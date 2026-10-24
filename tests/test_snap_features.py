"""End-to-end checks for etl/snap_features.py on a synthetic project.

Flowlines (NAD83), around 45 N:

    reach 3  Big Ditch (canal) ------------------------------  lat 45.020
                                   |
                                   | reach 2 Otter Brook (tributary, 4 km2)
    reach 1  Mill River (100 km2) -+-------------------------  lat 45.000
             lon -70.00           -69.97                -69.95

Run directly: `python tests/test_snap_features.py`
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
from shapely.geometry import LineString, Polygon

from etl import compare_snapping as cs
from etl import snap_features as sf
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC8 = "01010001"

# id: (name, ftype, line, totdasqkm, slope)
REACHES = {
    1: ("Mill River", 460, LineString([(-70.00, 45.0), (-69.95, 45.0)]), 100.0, 0.001),
    2: ("Otter Brook", 460, LineString([(-69.97, 45.010), (-69.97, 45.0005)]), 4.0, -9998.0),
    3: ("Big Ditch", 336, LineString([(-70.00, 45.02), (-69.95, 45.02)]), -9999.0, 0.0),
}

# id: (name, lat, lon, datum, reported area in sq mi)
STATIONS = {
    "S_NEAR": ("MILL RIVER AT TOWN", 45.0003, -69.99, "NAD83", None),
    "S_NAMED": ("MILL RIVER BELOW BROOK", 45.0020, -69.9702, "WGS84", None),
    "S_AREA": ("SITE 12", 45.0010, -69.9702, "NAD83", 38.6),   # 38.6 sq mi = 100 km2
    "S_CANAL": ("DITCH SITE", 45.0198, -69.99, "NAD83", None),
    "S_FAR": ("SITE 9", 45.0080, -69.99, "NAD83", None),
    "S_ASTRO": ("MILL RIVER OLD SURVEY", 45.0003, -69.985, "ASTRO", None),
    "S_UNK": ("MILL RIVER AT DAM RD", 45.0002, -69.96, "UNKWN", None),
    "S_OUT": ("ELSEWHERE", 44.0, -71.0, "NAD83", None),
}

DAMS = pd.DataFrame({
    "AR_ID": ["D1", "D2"],
    "Dam_Name": ["Mill Dam", "Brook Dam"],
    "River": ["Mill River", None],
    "Latitude": [45.0004, 45.0060],
    "Longitude": [-69.98, -69.9701],
    "Year_Removed": [2010, 2012],
    **{column: [None, None] for column in [
        "NABI_ID", "City_County", "HUC8", "State", "Dam_Height_ft", "Dam_Length_ft", "Year_Built",
        "Original_Use", "Type_Material", "Reason_for_removal", "Miles_Restored_US",
        "Miles_Restored_DS", "Last_Updated"]},
})


def _build_project(root, overrides=""):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        f"""
case_study: pilot_test
huc8s: ["{HUC8}"]
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
"""
    )
    (root / "config" / "snap_overrides.csv").write_text(
        "feature_kind,feature_id,action,nhdplus_id,reason,reviewed_by,reviewed_on\n" + overrides
    )

    dam_dir = root / "data" / "raw" / "dams"
    dam_dir.mkdir(parents=True)
    DAMS.to_csv(dam_dir / "dams.csv", index=False)

    station_dir = root / "data" / "raw" / "wqp" / "results" / "pilot_test"
    station_dir.mkdir(parents=True)
    pd.DataFrame([
        {
            "MonitoringLocationIdentifier": sid, "MonitoringLocationName": name,
            "LatitudeMeasure": lat, "LongitudeMeasure": lon,
            "HorizontalCoordinateReferenceSystemDatumName": datum,
            "DrainageAreaMeasure/MeasureValue": area,
            "DrainageAreaMeasure/MeasureUnitCode": "sq mi" if area else None,
            "source_huc8": HUC8,
            "MonitoringLocationTypeName": "River/Stream", "OrganizationIdentifier": "ORG",
            "OrganizationFormalName": "Org", "StateCode": 23, "CountyCode": 5,
            "HorizontalCollectionMethodName": "GPS",
        }
        for sid, (name, lat, lon, datum, area) in STATIONS.items()
    ]).to_csv(station_dir / "pilot_station_selection.csv", index=False)

    wbd_dir = root / "data" / "raw" / "wbd"
    wbd_dir.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"huc8": [HUC8], "name": ["Test"], "states": ["ME"], "areasqkm": [500.0]},
        geometry=[Polygon([(-70.05, 44.95), (-69.9, 44.95), (-69.9, 45.05), (-70.05, 45.05)])],
        crs="EPSG:4269",
    ).to_file(wbd_dir / "wbd.gpkg", layer="WBDHU8", driver="GPKG")

    package = f"NHDPLUS_H_{HUC8}_HU8_GDB"
    gpkg = root / "data" / "processed" / "nhdplus_extracted" / package / f"{package}.gpkg"
    gpkg.parent.mkdir(parents=True)
    # Lowercase field names, as in newer NHDPlus HR releases.
    gpd.GeoDataFrame(
        {
            "nhdplusid": [float(i) for i in REACHES],
            "gnis_name": [r[0] for r in REACHES.values()],
            "ftype": [r[1] for r in REACHES.values()],
            "fcode": [r[1] * 100 for r in REACHES.values()],
            "reachcode": [f"{HUC8}00000{i}" for i in REACHES],
            "lengthkm": [4.0, 1.0, 4.0],
        },
        geometry=[r[2] for r in REACHES.values()],
        crs="EPSG:4269",
    ).to_file(gpkg, layer="NHDFlowline", driver="GPKG")
    pyogrio.write_dataframe(
        pd.DataFrame({
            "nhdplusid": [float(i) for i in REACHES],
            "totdasqkm": [r[3] for r in REACHES.values()],
            "slope": [r[4] for r in REACHES.values()],
        }),
        gpkg, layer="NHDPlusFlowlineVAA", driver="GPKG", append=True,
    )
    # Otter Brook flows into Mill River.
    pyogrio.write_dataframe(
        pd.DataFrame({"fromnhdpid": [2.0], "tonhdpid": [1.0]}),
        gpkg, layer="NHDPlusFlow", driver="GPKG", append=True,
    )


def _run_main(root, *extra):
    argv = ["snap_features.py", "--case-study", "pilot_test", *extra]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        sf.main()


def _diagnostics(folder):
    frame = pd.read_csv(folder / f"snap_diagnostics_{HUC8}.csv", dtype={"feature_id": str})
    return frame.set_index("feature_id")


def test_decisions_layers_and_reach_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        _run_main(root)
        out = root / "data" / "processed"
        diag = _diagnostics(out)

        assert "S_OUT" not in diag.index  # outside the watershed
        expected = {
            "S_NEAR": ("accepted", "name_match", 1),
            "S_NAMED": ("accepted", "named_reach_preferred", 1),
            "S_AREA": ("accepted", "drainage_area_match", 1),
            "S_CANAL": ("rejected", "only_unsuitable_flowline", None),
            "S_FAR": ("ambiguous", "far_without_support", 1),
            "S_ASTRO": ("rejected", "datum_unsupported", None),
            "S_UNK": ("accepted", "name_match", 1),
            "D1": ("accepted", "name_match", 1),
            "D2": ("accepted", "nearest_reach", 2),
        }
        for fid, (status, reason, reach) in expected.items():
            row = diag.loc[fid]
            assert (row["snap_status"], row["snap_reason"]) == (status, reason), (fid, row["snap_reason"])
            if reach is not None:
                assert row["NHDPlusID"] == reach, fid

        assert diag.loc["S_UNK", "datum_assumed"] and not diag.loc["S_NEAR", "datum_assumed"]
        assert diag.loc["D1", "datum_assumed"] and diag.loc["D1", "datum_epsg"] == 4326

        layers = {name for name, _ in pyogrio.list_layers(out / f"snapped_features_{HUC8}.gpkg")}
        assert layers == {"stations_snapped", "dams_snapped", "stations_original", "dams_original"}
        snapped = gpd.read_file(out / f"snapped_features_{HUC8}.gpkg", layer="stations_snapped")
        assert set(snapped["MonitoringLocationIdentifier"]) == {"S_NEAR", "S_NAMED", "S_AREA", "S_UNK"}
        assert (snapped.geometry.y - 45.0).abs().max() < 1e-5  # on Mill River (~1 m; UTM chord)
        original = gpd.read_file(out / f"snapped_features_{HUC8}.gpkg", layer="stations_original")
        assert len(original) == 7 and set(original["snap_status"]) == {"accepted", "ambiguous", "rejected"}

        near = snapped.set_index("MonitoringLocationIdentifier").loc["S_NEAR"]
        assert near["reach_gnis_name"] == "Mill River" and near["reach_ftype"] == 460
        assert near["reach_totdasqkm"] == 100.0 and near["reach_slope"] == 0.001
        assert 30 < near["snap_distance_m"] < 36 and 0 < near["snap_fraction"] < 1
        dams = gpd.read_file(out / f"snapped_features_{HUC8}.gpkg", layer="dams_snapped") \
            .set_index("AR_ID")
        assert pd.isna(dams.loc["D2", "reach_slope"])  # -9998 placeholder -> null

        candidates = pd.read_csv(out / f"snap_candidates_{HUC8}.csv")
        assert {"feature_id", "distance_m", "name_match", "eligible_ftype"} <= set(candidates.columns)
        assert (root / "data" / "processed" / "_progress" / "pilot_test" / "snap_features" / HUC8).exists()


def test_huc8_without_stations_still_writes_its_dams():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        stations = dict(STATIONS)
        try:
            STATIONS.clear()
            STATIONS["S_OUT"] = stations["S_OUT"]  # the only station is outside the HUC8
            _build_project(root)
            _run_main(root)
        finally:
            STATIONS.clear()
            STATIONS.update(stations)
        gpkg = root / "data" / "processed" / f"snapped_features_{HUC8}.gpkg"
        assert len(gpd.read_file(gpkg, layer="stations_snapped")) == 0
        assert set(gpd.read_file(gpkg, layer="dams_snapped")["AR_ID"]) == {"D1", "D2"}


def test_overrides_are_applied_and_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root, overrides=(
            "dam,D2,accept,1,dam is on the main stem per map,jc,2026-10-23\n"
            "station,S_NEAR,reject,,duplicate of another site,jc,2026-10-23\n"))
        _run_main(root)
        diag = _diagnostics(root / "data" / "processed")
        assert (diag.loc["D2", "snap_status"], diag.loc["D2", "snap_reason"]) == ("accepted", "manual_override")
        assert diag.loc["D2", "NHDPlusID"] == 1 and diag.loc["D2", "auto_reason"] == "nearest_reach"
        assert diag.loc["D2", "match_method"] == "manual_override"
        assert (diag.loc["S_NEAR", "snap_status"], diag.loc["S_NEAR", "auto_status"]) == ("rejected", "accepted")


def test_override_to_a_reach_outside_the_package_fails_the_huc8():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root, overrides="dam,D2,accept,999,typo,jc,2026-10-23\n")
        try:
            _run_main(root)
        except RuntimeError:
            pass
        else:
            raise AssertionError("expected the HUC8 to fail")
        assert not (root / "data" / "processed" / f"snapped_features_{HUC8}.gpkg").exists()


def test_validation_run_leaves_current_outputs_and_progress_alone():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        _run_main(root, "--output-dir", "data/processed/snap_validation", "--huc8", HUC8)
        assert (root / "data" / "processed" / "snap_validation" / f"snapped_features_{HUC8}.gpkg").exists()
        assert not (root / "data" / "processed" / f"snapped_features_{HUC8}.gpkg").exists()
        assert not (root / "data" / "processed" / "_progress" / "pilot_test" / "snap_features").exists()


def test_compare_reports_moved_reaches_and_known_failures():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        _run_main(root)  # the "current" outputs
        (root / "config" / "snap_overrides.csv").write_text(
            "feature_kind,feature_id,action,nhdplus_id,reason,reviewed_by,reviewed_on\n"
            "dam,D2,accept,1,main stem per map,jc,2026-10-23\n")
        _run_main(root, "--output-dir", "data/processed/snap_validation")
        argv = ["compare_snapping.py", "--case-study", "pilot_test"]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(cs, "KNOWN_WRONG_STATIONS", ["S_FAR"]):
            cs.main()
        comparison = pd.read_csv(root / "data" / "processed" / "snap_validation" / "snap_comparison.csv",
                                 dtype={"feature_id": str}).set_index("feature_id")
        assert comparison.loc["D2", "change"] == "moved_reach"
        assert comparison.loc["D1", "change"] == "same_reach"
        assert comparison.loc["S_FAR", "change"] == "ambiguous_not_in_old"

        # The HUC8s of known failures, from the current warehouse.
        warehouse = root / "data" / "warehouse" / "pilot_test"
        warehouse.mkdir(parents=True)
        pd.DataFrame({"station_id": ["S_NEAR"], "huc8": [HUC8]}).to_parquet(warehouse / "wq_station.parquet")
        pd.DataFrame({"AR_ID": ["D1"], "HUC8": [HUC8]}).to_parquet(warehouse / "dam_removal.parquet")
        cfg = cs.load_config("pilot_test", project_root=root)
        with mock.patch.object(cs, "KNOWN_WRONG_STATIONS", ["S_NEAR", "NOT_IN_WAREHOUSE"]):
            assert cs.known_failure_huc8s(cfg, None) == [HUC8]


def test_outputs_flow_through_linking_and_core_tables():
    from etl import link_dams_stations as lds
    from etl import load_core_tables as lct

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        _run_main(root)
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}):
            for module, name in [(lds, "link_dams_stations.py"), (lct, "load_core_tables.py")]:
                with mock.patch.object(sys, "argv", [name, "--case-study", "pilot_test"]):
                    module.main()

        links = pd.read_csv(root / "data" / "processed" / f"dam_station_links_{HUC8}.csv")
        # Only accepted points are linked.
        assert set(links["station_id"]) == {"S_NEAR", "S_NAMED", "S_AREA", "S_UNK"}
        warehouse = root / "data" / "warehouse" / "pilot_test"
        stations = gpd.read_parquet(warehouse / "wq_station.parquet").set_index("station_id")
        assert stations.loc["S_UNK", "datum_assumed"]
        assert stations.loc["S_AREA", "reported_drainage_area_km2"] > 99
        assert (stations["reach_totdasqkm"] == 100.0).all()
        dams = gpd.read_parquet(warehouse / "dam_removal.parquet").set_index("AR_ID")
        assert dams.loc["D2", "reach_gnis_name"] == "Otter Brook"
        assert pd.isna(dams.loc["D2", "reach_slope"]) and dams.loc["D1", "reach_slope"] == 0.001
        assert set(dams["match_method"]) == {"scored_candidates"}


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
