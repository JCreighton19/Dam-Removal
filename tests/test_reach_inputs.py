"""Checks for etl/common/reach_inputs.py: load_core_tables.py and
download_wqp_results.py reading the stage 3-5 reach-matching outputs
(config link_source: reach_links).

Reuses the synthetic project of tests/test_load_core_tables.py (reaches
9 -> 1 -> 2 -> 3, and 5 on its own) and adds stage outputs:

    dams       D1 accepted on 2; D2 ambiguous (reservoir_name_mismatch) on 5;
               D3 accepted on 5, no links
    stations   S1 accepted on 3; S2 confluence-ambiguous, provisional 1;
               S3 accepted on 5; S4 accepted on 3
    links      D1-S1 downstream, both_accepted          -> loaded
               D1-S2 upstream, robust_to_reach_choice   -> loaded
               D1-S4 downstream, not usable             -> left out
               D2-S3 same_reach, not usable (dam)       -> left out

Run directly: `python tests/test_reach_inputs.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import pandas as pd
import pyogrio
from pyproj import Transformer

import test_load_core_tables as base
from etl import download_wqp_results as dwr
from etl import load_core_tables as lct
from etl.common import reach_inputs as ri
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC8 = base.HUC8
CASE = "pilot_test"
METRIC = "EPSG:32619"
TO_METRIC = Transformer.from_crs("EPSG:4269", METRIC, always_xy=True)


def _dam_csv():
    return pd.DataFrame({
        "AR_ID": [1, 2, 3],
        "NABI_ID": ["N1", None, None],
        "Dam_Name": ["Dam One", "Dam Two", "Dam Three"],
        "Year_Removed": [2017, 2015, 2010],
        "Latitude": [45.001, 45.2, 45.201],
        "Longitude": [-69.81, -69.75, -69.74],
        "City_County": ["Town"] * 3,
        "River": ["Test River"] * 3,
        "HUC8": ["Test Watershed"] * 3,
        "State": ["ME"] * 3,
        "Dam_Height_ft": [12, None, 5],
        "Dam_Length_ft": [100, None, None],
        "Year_Built": [1900, 0, 1950],
        "Original_Use": ["Mill"] * 3,
        "Type_Material": ["Stone"] * 3,
        "Reason_for_removal": ["Ecological"] * 3,
        "Miles_Restored_US": [1.5, None, None],
        "Miles_Restored_DS": [None] * 3,
        "Last_Updated": ["2026-02-10"] * 3,
    })


def _dam_reaches():
    return pd.DataFrame({
        "huc8": [HUC8] * 3,
        "AR_ID": ["1", "2", "3"],
        "Dam_Name": ["Dam One", "Dam Two", "Dam Three"],
        "Year_Removed": [2017, 2015, 2010],
        "datum_epsg": [4326] * 3,
        "datum_assumed": [True] * 3,
        "match_status": ["accepted", "ambiguous", "accepted"],
        "match_reason": ["reservoir_outlet", "reservoir_name_mismatch", "name_match"],
        "match_evidence": ["name_match", "name_mismatch", "name_match;far_snap"],
        "match_method": ["evidence_rules"] * 3,
        "NHDPlusID": pd.array([2, pd.NA, 5], dtype="Int64"),
        "snap_distance_m": [12.0, None, 200.0],
        "snap_fraction": [0.9, None, 0.6],
        "snapped_lon": [-69.81, None, -69.74],
        "snapped_lat": [45.0, None, 45.2],
    })


def _station_selection():
    ids = ["S1", "S2", "S3", "S4"]
    return pd.DataFrame({
        "MonitoringLocationIdentifier": ids,
        "MonitoringLocationName": ["Test River below dam", "Test River above dam",
                                   "Big Lake", "Test River"],
        "MonitoringLocationTypeName": ["River/Stream"] * 4,
        "OrganizationIdentifier": ["ORG"] * 4,
        "OrganizationFormalName": ["Org Name"] * 4,
        "LatitudeMeasure": [45.0, 45.0005, 45.2, 45.0],
        "LongitudeMeasure": [-69.75, -69.95, -69.79, -69.72],
        "StateCode": [23] * 4,
        "CountyCode": [5] * 4,
        "HorizontalCoordinateReferenceSystemDatumName": ["NAD83"] * 4,
        "HorizontalCollectionMethodName": ["GPS"] * 4,
        "DrainageAreaMeasure/MeasureValue": [None, 20.0, None, None],
        "DrainageAreaMeasure/MeasureUnitCode": [None, "sq mi", None, None],
        "source_huc8": [HUC8] * 4,
    })


def _station_reaches():
    lon = [-69.75, -69.95, -69.79, -69.72]
    lat = [45.0, 45.0005, 45.2, 45.0]
    x, y = TO_METRIC.transform(lon, lat)
    return pd.DataFrame({
        "huc8": [HUC8] * 4,
        "MonitoringLocationIdentifier": ["S1", "S2", "S3", "S4"],
        "MonitoringLocationName": ["n"] * 4,
        "LatitudeMeasure": lat,
        "LongitudeMeasure": lon,
        "datum_epsg": [4269] * 4,
        "datum_assumed": [False] * 4,
        "metric_crs": [METRIC] * 4,
        "x_m": x,
        "y_m": y,
        "snap_status": ["accepted", "ambiguous", "accepted", "accepted"],
        "snap_reason": ["name_match", "confluence_ambiguous", "name_match", "name_match"],
        "snap_evidence": ["name_match", "confluence_nearby", "far_snap", None],
        "match_method": ["evidence_rules"] * 4,
        "provisional_nhdplus_id": pd.array([3, 1, 5, 3], dtype="Int64"),
        "NHDPlusID": pd.array([3, pd.NA, 5, 3], dtype="Int64"),
        "snap_distance_m": [5.0, None, 7.0, 4.0],
        "snap_fraction": [0.5, None, 0.1, 0.8],
        "snapped_lon": [-69.75, None, -69.79, -69.72],
        "snapped_lat": [45.0, None, 45.2, 45.0],
    })


def _station_candidates():
    return pd.DataFrame({
        "huc8": [HUC8] * 3,
        "MonitoringLocationIdentifier": ["S2", "S2", "S1"],
        "NHDPlusID": pd.array([1, 9, 3], dtype="Int64"),
        "distance_m": [55.5, 60.0, 5.0],
    })


def _reach_links():
    rows = [
        ("1", "S1", 2, 3, "downstream", 6.0, 0.5, "accepted", "accepted", True, "both_accepted"),
        ("1", "S2", 2, 1, "upstream", 11.2, 0.5, "accepted", "ambiguous", True,
         "robust_to_reach_choice"),
        ("1", "S4", 2, 3, "downstream", 7.0, 0.8, "accepted", "accepted", False, None),
        ("2", "S3", 5, 5, "same_reach", 0.5, 0.1, "ambiguous", "accepted", False, None),
    ]
    return pd.DataFrame([{
        "huc8": HUC8, "dam_id": d, "station_id": s, "dam_reach_id": dr,
        "station_reach_id": sr, "relationship": rel, "network_distance_km": km,
        "distance_method": "network_path_fractional_endpoints", "dam_fraction": 0.9,
        "station_fraction": frac, "dam_status": ds, "dam_reason": "x",
        "station_status": ss, "station_reason": "x", "usable": usable,
        "station_in_mapped_reservoir": False, "usable_reason": reason,
    } for d, s, dr, sr, rel, km, frac, ds, ss, usable, reason in rows])


def _build_project(root):
    base._build_project(root)
    (root / "config" / f"{CASE}.yaml").write_text(
        f'case_study: {CASE}\nhuc8s: ["{HUC8}"]\ndam_source: data/raw/dams/dams.csv\n'
        "wbd_source: data/raw/wbd/wbd.gpkg\n")  # link_source defaults to reach_links

    (root / "data" / "raw" / "dams").mkdir(parents=True)
    _dam_csv().to_csv(root / "data" / "raw" / "dams" / "dams.csv", index=False)
    selection = root / "data" / "raw" / "wqp" / "results" / CASE
    selection.mkdir(parents=True)
    _station_selection().to_csv(selection / "pilot_station_selection.csv", index=False)

    processed = root / "data" / "processed"
    for stage, prefix, frame in [("dam_reaches", "dam_reaches", _dam_reaches()),
                                 ("station_reaches", "station_reaches", _station_reaches()),
                                 ("station_candidates", "station_candidates",
                                  _station_candidates()),
                                 ("reach_links", "reach_links", _reach_links())]:
        folder = processed / stage / CASE
        folder.mkdir(parents=True)
        frame.to_parquet(folder / f"{prefix}_{HUC8}.parquet", index=False)
    marker = processed / "_progress" / CASE / "link_reaches" / HUC8
    marker.parent.mkdir(parents=True)
    marker.touch()

    gdb = (processed / "nhdplus_extracted" / f"NHDPLUS_H_{HUC8}_HU8_GDB"
           / f"NHDPLUS_H_{HUC8}_HU8_GDB.gdb")
    pyogrio.write_dataframe(
        pd.DataFrame({"nhdplusid": [1.0, 2.0, 3.0, 5.0], "totdasqkm": [80.0, 90.0, 100.0, -9999],
                      "slope": [0.001, 0.002, 0.003, -9998]}),
        gdb, layer="NHDPlusFlowlineVAA", driver="OpenFileGDB", append=True)


def _config(root):
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}):
        return lct.load_config(CASE)


def test_usable_links_keep_usable_pairs_with_dam_year():
    links = ri.usable_links(_reach_links(), _dam_reaches())
    assert list(zip(links["dam_id"], links["station_id"])) == [("1", "S1"), ("1", "S2")]
    assert links["year_removed"].tolist() == [2017, 2017]
    assert links["usable_reason"].tolist() == ["both_accepted", "robust_to_reach_choice"]


def test_link_source_defaults_to_reach_links_and_rejects_typos():
    cfg = mock.Mock(get=lambda key, default=None: default)
    assert ri.link_source(cfg) == ri.REACH_LINKS
    cfg = mock.Mock(get=lambda key, default=None: "reach-links")
    try:
        ri.link_source(cfg)
        raise AssertionError("Expected ValueError")
    except ValueError as exc:
        assert "link_source" in str(exc)


def test_load_core_tables_from_reach_links():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        tables = lct.build_tables(_config(root), [HUC8])
        assert lct.validate_tables(tables) == []

    dams = tables["dam_removal"].set_index("AR_ID")
    assert sorted(dams.index) == ["1", "3"]  # D2 ambiguous: left out
    d1 = dams.loc["1"]
    assert d1["HUC8"] == HUC8 and d1["NHDPlusID"] == 2 and d1["snap_fraction"] == 0.9
    assert d1["Dam_Name"] == "Dam One" and d1["Year_built"] == 1900 and d1["NABI_ID"] == "N1"
    assert d1["snap_reason"] == "reservoir_outlet" and d1["match_method"] == "evidence_rules"
    assert d1["reach_slope"] == 0.002 and d1["reach_totdasqkm"] == 90.0
    assert d1["reach_feature_type"] == 460 and d1["reach_feature_code"] == 46006
    assert not d1["qc_far_snap"]
    assert abs(d1.geometry.x + 69.81) < 1e-4 and abs(d1.geometry.y - 45.0) < 1e-4
    d3 = dams.loc["3"]
    assert d3["qc_far_snap"] and pd.isna(d3["reach_slope"]) and pd.isna(d3["reach_totdasqkm"])

    links = tables["dam_station_link"]
    assert sorted(zip(links["AR_ID"], links["station_id"], links["relationship"])) == [
        ("1", "S1", "downstream"), ("1", "S2", "upstream")]

    stations = tables["wq_station"].set_index("station_id")
    assert sorted(stations.index) == ["S1", "S2"]  # S3, S4 only in unusable links
    s1, s2 = stations.loc["S1"], stations.loc["S2"]
    assert s1["NHDPlusID"] == 3 and s1["match_method"] == "evidence_rules"
    assert s1["snap_distance_m"] == 5.0 and s1["reach_slope"] == 0.003
    assert s1["station_name"] == "Test River below dam" and s1["county_code"] == "005"
    # S2: on its provisional reach, reported location, candidate distance.
    assert s2["NHDPlusID"] == 1 and s2["snap_fraction"] == 0.5
    assert s2["match_method"] == ri.PROVISIONAL and s2["snap_reason"] == "confluence_ambiguous"
    assert s2["snap_distance_m"] == 55.5
    assert abs(s2.geometry.x + 69.95) < 1e-4 and abs(s2.geometry.y - 45.0005) < 1e-4
    assert abs(s2["reported_drainage_area_km2"] - 20 * 2.58999) < 1e-6
    assert set(stations["huc8"]) == {HUC8}

    streams = set(tables["streams"]["nhdplus_id"])
    assert {1, 2, 3, 5} <= streams


def test_main_writes_the_warehouse_from_reach_links():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        out = base._run_main(root)
        assert len(pd.read_parquet(out / "dam_station_link.parquet")) == 2


def test_wqp_download_requests_usable_stations_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        stations = dwr.load_confirmed_stations(_config(root), 2, 3)
    assert stations.to_dict("records") == [
        {"station_id": "S1", "start_year": 2015, "end_year": 2020},
        {"station_id": "S2", "start_year": 2015, "end_year": 2020},
    ]


def test_missing_stage_outputs_say_how_to_fall_back():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        base._build_project(root)
        (root / "config" / f"{CASE}.yaml").write_text(
            f'case_study: {CASE}\nhuc8s: ["{HUC8}"]\ndam_source: data/raw/dams/dams.csv\n'
            "wbd_source: data/raw/wbd/wbd.gpkg\n")
        try:
            lct.build_tables(_config(root), [HUC8])
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError as exc:
            assert "link_source: snapped" in str(exc)


def test_huc8s_the_stages_never_ran_on_stop_the_load():
    # A HUC8 with no link_reaches marker (e.g. outside a --hu4 subset run) is
    # not "no links": the load stops instead of leaving it out.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        (root / "data" / "processed" / "_progress" / CASE / "link_reaches" / HUC8).unlink()
        for load in (lambda cfg: lct.build_tables(cfg, [HUC8]),
                     lambda cfg: dwr.load_confirmed_stations(cfg, 2, 3)):
            try:
                load(_config(root))
                raise AssertionError("Expected ValueError")
            except ValueError as exc:
                assert "never processed" in str(exc) and HUC8 in str(exc)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
