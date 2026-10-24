"""Lightweight validation checks for download_wqp_results.py's station
selection logic (now sourced from confirmed network-connected stations,
not the raw candidate list).

Run directly: `python tests/test_download_wqp_results.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl import download_wqp_results as dwr
from etl.common.config import PROJECT_ROOT_ENV_VAR, load_config
from etl.download_wqp_results import (
    build_batches,
    covered_ranges,
    load_confirmed_stations,
    merge_windows,
    uncovered,
)


def _write_links(path, rows):
    pd.DataFrame(rows, columns=[
        "dam_id", "dam_name", "station_id", "year_removed", "relationship"
    ]).to_csv(path, index=False)


def _write_config(root, huc8s):
    (root / "config").mkdir(parents=True)
    huc8_yaml = "\n".join(f'  - "{h}"' for h in huc8s)
    (root / "config" / "test.yaml").write_text(
        f"""
case_study: test
huc8s:
{huc8_yaml}
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
"""
    )
    return load_config("test", project_root=root)


def test_only_connected_stations_are_selected():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _write_config(root, ["11111111"])

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", "S1", 2017, "upstream"),
            ("D1", "Dam One", "S2", 2017, "unconnected"),
        ])

        stations = load_confirmed_stations(cfg, 5, 5)
        assert stations["station_id"].tolist() == ["S1"]


def test_stations_merged_and_deduped_across_huc8s():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _write_config(root, ["11111111", "22222222"])

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", "S1", 2017, "upstream"),
            ("D1", "Dam One", "SHARED", 2017, "downstream"),
        ])
        _write_links(processed_dir / "dam_station_links_22222222.csv", [
            ("D2", "Dam Two", "S2", 2015, "same_reach"),
            ("D2", "Dam Two", "SHARED", 2015, "upstream"),
        ])

        stations = load_confirmed_stations(cfg, 5, 5)
        assert sorted(stations["station_id"]) == ["S1", "S2", "SHARED"]


def test_missing_huc8_link_table_is_skipped_not_fatal():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _write_config(root, ["11111111", "99999999"])

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", "S1", 2017, "upstream"),
        ])
        # No link table for 99999999 -- should warn, not raise.

        stations = load_confirmed_stations(cfg, 5, 5)
        assert stations["station_id"].tolist() == ["S1"]


def test_overlapping_windows_are_merged():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _write_config(root, ["11111111"])

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", "SHARED", 2010, "upstream"),
            ("D2", "Dam Two", "SHARED", 2015, "downstream"),
            ("D3", "Dam Three", "SHARED", 1990, "unconnected"),  # ignored
        ])

        stations = load_confirmed_stations(cfg, 5, 3)
        row = stations.set_index("station_id").loc["SHARED"]
        assert (row["start_year"], row["end_year"]) == (2005, 2018)


def test_separate_windows_for_distant_removal_years():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _write_config(root, ["11111111"])

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", "SHARED", 1986, "upstream"),
            ("D2", "Dam Two", "SHARED", 2018, "downstream"),
            ("D3", "Dam Three", "SHARED", 1990, "upstream"),  # overlaps D1
        ])

        stations = load_confirmed_stations(cfg, 5, 5)
        ranges = list(zip(stations["start_year"], stations["end_year"]))
        assert ranges == [(1981, 1995), (2013, 2023)]


def test_adjacent_windows_are_merged():
    assert merge_windows([(2000, 2005), (2006, 2010)]) == [(2000, 2010)]
    assert merge_windows([(2000, 2005), (2007, 2010)]) == [(2000, 2005), (2007, 2010)]


def test_stations_without_removal_year_are_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = _write_config(root, ["11111111"])

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", "S1", 2017, "upstream"),
            ("D2", "Dam Two", "S2", None, "upstream"),
            ("D3", "Dam Three", "S3", 0, "downstream"),
        ])

        stations = load_confirmed_stations(cfg, 5, 5)
        assert stations["station_id"].tolist() == ["S1"]


def test_batches_never_mix_date_ranges():
    stations = pd.DataFrame({
        "station_id": ["A", "B", "C", "D"],
        "start_year": [2005, 2005, 2005, 2010],
        "end_year": [2015, 2015, 2015, 2020],
    })

    batches = build_batches(stations, batch_size=2)
    assert batches == [
        (2005, 2015, ["A", "B"]),
        (2005, 2015, ["C"]),
        (2010, 2020, ["D"]),
    ]


def test_uncovered_keeps_only_ranges_not_downloaded():
    entries = {
        1: {"start_year": 2005, "end_year": 2015, "stations": "A;B"},
        2: {"start_year": 2016, "end_year": 2020, "stations": "B"},
    }
    covered = covered_ranges(entries)
    assert covered["B"] == [(2005, 2020)]  # adjacent ranges merge
    needed = pd.DataFrame({
        "station_id": ["A", "A", "B", "C"],
        "start_year": [2006, 2010, 2008, 2005],
        "end_year": [2014, 2018, 2019, 2015],
    })
    todo = uncovered(needed, covered)
    # A 2010-2018 is only partly covered; C is new.
    assert list(zip(todo["station_id"], todo["start_year"])) == [("A", 2010), ("C", 2005)]


def test_rerun_requests_only_new_or_failed_station_ranges():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_config(root, ["11111111"])
        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True)
        # Window 3/3 (the default) around 2010: 2007-2013.
        _write_links(processed_dir / "dam_station_links_11111111.csv", [
            ("D1", "Dam One", s, 2010, "downstream") for s in ["S1", "S2", "S3", "S4"]
        ])
        download = root / "data" / "raw" / "wqp" / "results" / "test" / "pilot_download"
        download.mkdir(parents=True)
        pd.DataFrame({"MonitoringLocationIdentifier": ["S1"]}).to_csv(
            download / "results_batch_001.csv", index=False)
        pd.DataFrame([
            {"batch": 1, "start_year": 2007, "end_year": 2013, "station_count": 2,
             "stations": "S1;S2", "data_profile": dwr.DATA_PROFILE, "status": "success",
             "file": "results_batch_001.csv", "rows": 1, "error": None},
            {"batch": 2, "start_year": 2007, "end_year": 2013, "station_count": 1,
             "stations": "S3", "data_profile": dwr.DATA_PROFILE, "status": "failed",
             "file": None, "rows": None, "error": "429"},
        ]).to_csv(download / "download_manifest.csv", index=False)

        requested = []

        def fake_fetch(batch, start_year, end_year):
            requested.append((tuple(batch), start_year, end_year))
            return pd.DataFrame({"MonitoringLocationIdentifier": batch})

        argv = ["download_wqp_results.py", "--case-study", "test"]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(dwr, "fetch_batch", fake_fetch):
            dwr.main()

        assert requested == [(("S3", "S4"), 2007, 2013)]  # S1, S2 reused; S3 failed before
        manifest = pd.read_csv(download / "download_manifest.csv")
        assert manifest["batch"].tolist() == [1, 2]
        assert manifest["status"].tolist() == ["success", "success"]
        assert manifest.loc[1, "stations"] == "S3;S4"


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
