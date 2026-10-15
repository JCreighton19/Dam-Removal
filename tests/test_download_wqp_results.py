"""Lightweight validation checks for download_wqp_results.py's station
selection logic (now sourced from confirmed network-connected stations,
not the raw candidate list).

Run directly: `python tests/test_download_wqp_results.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common.config import load_config
from etl.download_wqp_results import load_confirmed_stations


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

        stations = load_confirmed_stations(cfg)
        assert stations == ["S1"]


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

        stations = load_confirmed_stations(cfg)
        assert stations == ["S1", "S2", "SHARED"]


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

        stations = load_confirmed_stations(cfg)
        assert stations == ["S1"]


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
