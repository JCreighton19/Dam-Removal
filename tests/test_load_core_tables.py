"""Validation checks for etl/load_core_tables.py.

Builds a throwaway project root with synthetic versions of every input
(snapped features gpkg, links CSV, WBD GeoPackage, NHDPlus geodatabase)
and runs the real `main()`, plus focused checks on the dedup, link
filtering, stream selection and validation rules.

Run directly: `python tests/test_load_core_tables.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import LineString, Point, Polygon

from etl.common.config import PROJECT_ROOT_ENV_VAR
from etl import load_core_tables as lct

HUC8 = "01010001"

# Three reaches flowing west to east: 1 -> 2 -> 3, plus 9 in another HUC8
# draining into 1, and 5 disconnected from the dam.
REACHES = {
    1: ("01010001000001", LineString([(-70.0, 45.0), (-69.9, 45.0)])),
    2: ("01010001000002", LineString([(-69.9, 45.0), (-69.8, 45.0)])),
    3: ("01010001000003", LineString([(-69.8, 45.0), (-69.7, 45.0)])),
    5: ("01010001000005", LineString([(-69.8, 45.2), (-69.7, 45.2)])),
    9: ("01010002000009", LineString([(-70.1, 45.0), (-70.0, 45.0)])),
}
FLOW = [(9, 1), (1, 2), (2, 3)]


def _snap_attrs(n, **overrides):
    """The match/reach columns snap_features.py writes, for n rows."""
    attrs = {
        "match_method": ["scored_candidates"] * n,
        "snap_reason": ["name_match"] * n,
        "snap_evidence": ["name_match"] * n,
        "reach_ftype": [460] * n,
        "reach_gnis_name": ["Test River"] * n,
        "reach_totdasqkm": [120.5] * n,
        "reach_slope": [0.002] * n,
        "datum_epsg": [4269] * n,
        "datum_assumed": [False] * n,
        "qc_far_snap": [False] * n,
        "qc_near_unsuitable_flowline": [False] * n,
    }
    attrs.update(overrides)
    return attrs


def _dams_snapped():
    """Dam D1 snapped twice (equidistant to reaches 2 and 3); D2 on reach 5."""
    rows = [
        ("D1", "Dam One", "2017", 2, 10.0, 0.9, (-69.81, 45.0)),
        ("D1", "Dam One", "2017", 3, 10.0, 0.0, (-69.80, 45.0)),
        ("D2", "Dam Two", "0", 5, 3.0, 0.5, (-69.75, 45.2)),
    ]
    return gpd.GeoDataFrame(
        {
            "AR_ID": [r[0] for r in rows],
            "NABI_ID": ["N1", "N1", None],
            "Dam_Name": [r[1] for r in rows],
            "Year_Removed": [r[2] for r in rows],
            "Latitude": [45.001, 45.001, 45.2],
            "Longitude": [-69.81, -69.81, -69.75],
            "City_County": ["Town", "Town", "Town"],
            "River": ["Test River"] * 3,
            "HUC8": ["Test Watershed"] * 3,
            "State": ["ME"] * 3,
            "Dam_Height_ft": ["12", "12", "unknown"],
            "Dam_Length_ft": ["100", "100", None],
            "Year_Built": ["1900", "1900", "0"],
            "Original_Use": ["Mill"] * 3,
            "Type_Material": ["Stone"] * 3,
            "Reason_for_removal": ["Ecological"] * 3,
            "Miles_Restored_US": ["1.5", "1.5", None],
            "Miles_Restored_DS": [None] * 3,
            "Last_Updated": ["2026-02-10", "2026-02-10", "not a date"],
            "NHDPlusID": [float(r[3]) for r in rows],
            "snap_distance_m": [r[4] for r in rows],
            "snap_fraction": [r[5] for r in rows],
            **_snap_attrs(len(rows), datum_epsg=[4326] * 3, datum_assumed=[True] * 3,
                          reach_slope=[0.002, 0.002, None]),
        },
        geometry=[Point(r[6]) for r in rows],
        crs="EPSG:4326",
    )


def _stations_snapped():
    """S1 downstream of D1 on reach 3; S2 upstream on reach 1; S3 unconnected."""
    rows = [
        ("S1", 3, 0.5, (-69.75, 45.0)),
        ("S2", 1, 0.5, (-69.95, 45.0)),
        ("S3", 5, 0.1, (-69.79, 45.2)),
    ]
    return gpd.GeoDataFrame(
        {
            "MonitoringLocationIdentifier": [r[0] for r in rows],
            "MonitoringLocationName": ["Station " + r[0] for r in rows],
            "MonitoringLocationTypeName": ["River/Stream"] * 3,
            "OrganizationIdentifier": ["ORG"] * 3,
            "OrganizationFormalName": ["Org Name"] * 3,
            "LatitudeMeasure": [45.0, 45.0, 45.2],
            "LongitudeMeasure": [-69.75, -69.95, -69.79],
            "StateCode": [23, 23, 23],
            "CountyCode": [5, 5, 5],
            "HorizontalCoordinateReferenceSystemDatumName": ["NAD83"] * 3,
            "HorizontalCollectionMethodName": ["GPS"] * 3,
            "NHDPlusID": [float(r[1]) for r in rows],
            "snap_distance_m": [5.0, 6.0, 7.0],
            "snap_fraction": [r[2] for r in rows],
            **_snap_attrs(len(rows), qc_far_snap=[False, True, False]),
            "reported_area_km2": [None, 50.0, None],
            "qc_non_riverine_name": [False, False, True],
        },
        geometry=[Point(r[3]) for r in rows],
        crs="EPSG:4326",
    )


def _links():
    """link_dams_stations.py output: one row per (dam snap, station)."""
    rows = [
        # D1 on reach 2 (kept snap) and reach 3 (dropped duplicate snap).
        ("D1", "S1", 2, 3, "downstream", 1.2, "network_path_fractional_endpoints"),
        ("D1", "S2", 2, 1, "upstream", 9.0, "network_path_fractional_endpoints"),
        ("D1", "S3", 2, 5, "unconnected", None, None),
        ("D1", "S1", 3, 3, "downstream", 4.0, "same_reach_fraction"),
        ("D1", "S2", 3, 1, "upstream", 15.0, "network_path_fractional_endpoints"),
        ("D2", "S3", 5, 5, "upstream", 3.1, "same_reach_fraction"),
    ]
    return pd.DataFrame(
        [
            {
                "huc8": HUC8, "dam_id": d, "dam_name": "x", "station_id": s,
                "dam_reach_id": dr, "year_removed": 2017, "river": "r",
                "station_reach_id": sr, "relationship": rel,
                "distance_to_dam_km": dist, "network_distance_km": dist,
                "distance_method": method,
            }
            for d, s, dr, sr, rel, dist, method in rows
        ]
    )


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        f"""
case_study: pilot_test
huc8s: ["{HUC8}"]
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
"""
    )

    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    snapped = processed / f"snapped_features_{HUC8}.gpkg"
    _stations_snapped().to_file(snapped, layer="stations_snapped", driver="GPKG")
    _dams_snapped().to_file(snapped, layer="dams_snapped", driver="GPKG", mode="a")
    _links().to_csv(processed / f"dam_station_links_{HUC8}.csv", index=False)

    wbd_dir = root / "data" / "raw" / "wbd"
    wbd_dir.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"huc8": [HUC8, "01010002"], "name": ["Test", "Other"],
         "states": ["ME", "ME"], "areasqkm": [1234.5, 99.0]},
        geometry=[
            Polygon([(-70.05, 44.9), (-69.6, 44.9), (-69.6, 45.3), (-70.05, 45.3)]),
            Polygon([(-70.2, 44.9), (-70.05, 44.9), (-70.05, 45.3), (-70.2, 45.3)]),
        ],
        crs="EPSG:4269",
    ).to_file(wbd_dir / "wbd.gpkg", layer="WBDHU8", driver="GPKG")

    # Lowercase field names, as in newer NHDPlus HR releases.
    gdb_dir = processed / "nhdplus_extracted" / f"NHDPLUS_H_{HUC8}_HU8_GDB"
    gdb_dir.mkdir(parents=True)
    gdb = gdb_dir / f"NHDPLUS_H_{HUC8}_HU8_GDB.gdb"
    gpd.GeoDataFrame(
        {
            "nhdplusid": [float(i) for i in REACHES],
            "reachcode": [code for code, _ in REACHES.values()],
            "gnis_name": ["Test River"] * len(REACHES),
            "lengthkm": [8.0] * len(REACHES),
            "ftype": [460] * len(REACHES),
            "fcode": [46006] * len(REACHES),
        },
        geometry=[line for _, line in REACHES.values()],
        crs="EPSG:4269",
    ).to_file(gdb, layer="NHDFlowline", driver="OpenFileGDB")
    pyogrio.write_dataframe(
        pd.DataFrame({
            "fromnhdpid": [float(a) for a, _ in FLOW],
            "tonhdpid": [float(b) for _, b in FLOW],
        }),
        gdb, layer="NHDPlusFlow", driver="OpenFileGDB", append=True,
    )


def _run_main(root):
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", ["load_core_tables.py", "--case-study", "pilot_test"]):
        lct.main()
    return root / "data" / "warehouse" / "pilot_test"


def test_main_writes_all_tables_matching_the_spec():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        out = _run_main(root)

        assert sorted(p.name for p in out.iterdir()) == sorted(
            f"{name}.parquet" for name in lct.SCHEMAS
        )

        dams = gpd.read_parquet(out / "dam_removal.parquet")
        assert list(dams.columns) == list(lct.SCHEMAS["dam_removal"])
        d1 = dams.set_index("AR_ID").loc["D1"]
        assert d1["NHDPlusID"] == 2  # tie on distance -> smallest NHDPlusID
        assert d1["HUC8"] == HUC8  # pipeline HUC8, not the CSV's watershed name
        assert d1["Year_built"] == 1900
        d2 = dams.set_index("AR_ID").loc["D2"]
        assert pd.isna(d2["Year_Removed"]) and pd.isna(d2["Year_built"])  # 0 -> null
        assert pd.isna(d2["Dam_Height_ft"]) and pd.isna(d2["Last_Updated"])

        links = pd.read_parquet(out / "dam_station_link.parquet")
        assert len(links) == 3
        d1_s1 = links[(links["AR_ID"] == "D1") & (links["station_id"] == "S1")]
        assert d1_s1["network_distance_km"].item() == 1.2  # row from the kept snap

        stations = gpd.read_parquet(out / "wq_station.parquet")
        assert set(stations["station_id"]) == {"S1", "S2", "S3"}
        assert set(stations["state_code"]) == {"23"}
        assert set(stations["county_code"]) == {"005"}
        s2 = stations.set_index("station_id").loc["S2"]
        assert s2["reach_gnis_name"] == "Test River" and s2["reach_feature_type"] == 460
        assert s2["reach_totdasqkm"] == 120.5 and s2["qc_far_snap"]
        assert s2["reported_drainage_area_km2"] == 50.0 and s2["match_method"] == "scored_candidates"
        assert stations.set_index("station_id").loc["S3", "qc_non_riverine_name"]
        assert d1["datum_assumed"] and d1["datum_epsg"] == 4326
        assert pd.isna(d2["reach_slope"])

        watersheds = gpd.read_parquet(out / "huc8_watershed.parquet")
        assert list(watersheds["huc8"]) == [HUC8]
        assert watersheds.crs.to_epsg() == 4326
        assert watersheds.geometry.geom_type.eq("MultiPolygon").all()

        streams = gpd.read_parquet(out / "streams.parquet")
        # 1-2-3 are connected to D1; 5 holds D2 and S3; 9 is upstream but
        # outside the case-study HUC8.
        assert set(streams["nhdplus_id"]) == {1, 2, 3, 5}
        assert not streams.geometry.has_z.any()


def test_failed_validation_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        out = root / "data" / "warehouse" / "pilot_test"

        with mock.patch.object(lct, "validate_tables", return_value=["forced failure"]):
            try:
                _run_main(root)
                raise AssertionError("Expected ValueError from failed validation.")
            except ValueError as exc:
                assert "no warehouse files were written" in str(exc)

        assert not out.exists()


def test_missing_inputs_are_all_named():
    with tempfile.TemporaryDirectory() as tmp:
        processed = Path(tmp)
        try:
            lct.check_inputs(processed, ["11111111", "22222222"])
            raise AssertionError("Expected FileNotFoundError.")
        except FileNotFoundError as exc:
            for huc8 in ("11111111", "22222222"):
                assert f"snapped_features_{huc8}.gpkg" in str(exc)
                assert f"dam_station_links_{huc8}.csv" in str(exc)


def test_dedupe_prefers_smaller_distance_then_smaller_reach():
    frame = pd.DataFrame({
        "AR_ID": ["A", "A", "B", "B"],
        "NHDPlusID": pd.array([7, 3, 4, 8], dtype="Int64"),
        "snap_distance_m": [1.0, 2.0, 5.0, 5.0],
    })
    kept = lct.dedupe_snaps(frame, "AR_ID", "test").set_index("AR_ID")["NHDPlusID"]
    assert kept["A"] == 7
    assert kept["B"] == 4


def test_validation_reports_broken_foreign_keys():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}):
            cfg = lct.load_config("pilot_test")
            tables = lct.build_tables(cfg, [HUC8])

    assert lct.validate_tables(tables) == []

    broken = dict(tables)
    broken["dam_station_link"] = tables["dam_station_link"].copy()
    broken["dam_station_link"].loc[0, "station_id"] = "NOT_A_STATION"
    broken["streams"] = tables["streams"][tables["streams"]["nhdplus_id"] != 5]

    errors = lct.validate_tables(broken)
    assert any("station_id not in wq_station" in e for e in errors)
    assert any("NHDPlusID not in streams" in e for e in errors)


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
