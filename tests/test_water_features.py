"""Checks for etl/common/water_features.py and etl/nhdplus_diagnostics.py
on a synthetic package where every relationship problem occurs once.

Flowlines (NHDPlusID: FType, WBArea_Permanent_Identifier):
    1: 460, none          ordinary stream
    2: 558, {area-1}      path through a wide river (NHDArea 460; id in
                          another case and with braces)
    3: 558, WB-1          path through a lake (NHDWaterbody 390)
    4: 558, none          artificial path without a polygon
    5: 558, GONE          polygon id that resolves to nothing
    6: 334, WB-1          connector naming a polygon

Run directly: `python tests/test_water_features.py`
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
from shapely.geometry import LineString, box

from etl import nhdplus_diagnostics as nd
from etl.common import water_features as wf
from etl.common.config import PROJECT_ROOT_ENV_VAR

FLOWLINES = pd.DataFrame({
    "NHDPlusID": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    "FType": [460, 558, 558, 558, 558, 334],
    "WBArea_Permanent_Identifier": [None, "{area-1}", "WB-1", " ", "GONE", "WB-1"],
})
AREAS = pd.DataFrame({"Permanent_Identifier": ["AREA-1", "AREA-2"], "NHDPlusID": [100.0, 101.0],
                      "FType": [460, 431]})
WATERBODIES = pd.DataFrame({"Permanent_Identifier": ["WB-1", "WB-2", "WB-3"],
                            "NHDPlusID": [200.0, 201.0, 202.0], "FType": [390, 390, 390]})
BURN = pd.DataFrame({"NHDPlusID": [100.0, 200.0, 201.0, 202.0, 999.0],
                     "OnOffNet": [1, 1, 0, 1, 1]})
CATCHMENTS = pd.DataFrame({"NHDPlusID": [1.0, 2.0, 3.0, 4.0, 77.0]})
VAA = pd.DataFrame({"NHDPlusID": [1.0, 2.0, 3.0, 4.0, 5.0],
                    "TotDASqKm": [10.0, 50.0, 60.0, 60.0, -9999.0]})
FLOW = pd.DataFrame({"FromNHDPID": [1.0, 2.0, 3.0, 4.0, 88.0],
                     "ToNHDPID": [2.0, 3.0, 0.0, 3.0, 1.0]})


def _checks():
    return wf.relationship_checks(FLOWLINES, AREAS, WATERBODIES, CATCHMENTS, VAA, FLOW, BURN)


def _ids(exceptions, kind):
    return sorted(exceptions.loc[exceptions["kind"] == kind, "record_id"].tolist())


def test_links_paths_to_polygons_by_stored_id():
    polygons = wf.water_polygons(AREAS, WATERBODIES)
    linked = wf.link_flowlines_to_polygons(FLOWLINES, polygons).set_index("NHDPlusID")
    assert linked.loc[2.0, "polygon_layer"] == "NHDArea"
    assert linked.loc[2.0, "polygon_ftype"] == 460
    assert linked.loc[3.0, "polygon_layer"] == "NHDWaterbody"
    assert linked["wbarea_status"].to_dict() == {
        1.0: "none", 2.0: "resolved", 3.0: "resolved", 4.0: "none", 5.0: "unresolved",
        6.0: "resolved"}


def test_counts():
    counts, _, _ = _checks()
    assert counts["n_artificial_path_558"] == 4
    assert counts["n_ap_with_polygon_id"] == 3
    assert counts["pct_ap_with_polygon_id"] == 75.0
    assert counts["n_ap_polygon_resolved"] == 2
    assert counts["n_ap_polygon_unresolved"] == 1
    assert (counts["n_ap_to_nhdarea"], counts["n_ap_to_nhdwaterbody"]) == (1, 1)
    assert counts["n_polygons_with_paths"] == 2
    # Matchable = 460/558 flowlines 1-5; 5 has no catchment.
    assert counts["n_matchable_without_catchment"] == 1
    assert counts["pct_matchable_with_catchment"] == 80.0


def test_exceptions():
    _, _, exceptions = _checks()
    assert _ids(exceptions, "artificial_path_without_polygon") == ["4"]
    assert _ids(exceptions, "polygon_id_unresolved") == ["5"]
    assert _ids(exceptions, "non_artificial_path_with_polygon") == ["6"]
    assert _ids(exceptions, "onnet_polygon_without_paths") == ["WB-3"]
    assert _ids(exceptions, "offnet_polygon_with_paths") == []
    assert _ids(exceptions, "burn_waterbody_unresolved") == ["999"]
    assert _ids(exceptions, "catchment_without_flowline") == ["77"]
    assert _ids(exceptions, "flowline_without_catchment") == ["5"]
    assert _ids(exceptions, "flowline_without_vaa") == []
    assert _ids(exceptions, "matchable_flowline_invalid_totdasqkm") == ["5"]
    assert _ids(exceptions, "flow_from_unknown_flowline") == ["88"]
    assert _ids(exceptions, "flow_to_unknown_flowline") == []
    assert _ids(exceptions, "matchable_flowline_not_in_flow") == ["5"]


def test_polygon_types():
    _, types, _ = _checks()
    types = types.set_index(["polygon_layer", "polygon_ftype"])
    assert types.loc[("NHDArea", 460), "n_linked_to_paths"] == 1
    assert types.loc[("NHDArea", 431), "n_linked_to_paths"] == 0
    lakes = types.loc[("NHDWaterbody", 390)]
    assert (lakes["n_polygons"], lakes["n_linked_to_paths"], lakes["n_on_network"]) == (3, 1, 2)


def test_duplicate_polygon_id_and_no_burn_layer():
    waterbodies = pd.concat([WATERBODIES, AREAS.iloc[[0]]], ignore_index=True)  # AREA-1 twice
    counts, types, exceptions = wf.relationship_checks(
        FLOWLINES, AREAS, waterbodies, CATCHMENTS, VAA, FLOW, burn=None)
    assert _ids(exceptions, "duplicate_polygon_permanent_id") == ["AREA-1", "AREA-1"]
    assert counts["n_burn_waterbody"] == 0
    assert types["n_on_off_unknown"].sum() == types["n_polygons"].sum()


def _write_package(gpkg, with_water=True):
    crs = "EPSG:4269"
    line = LineString([(0, 0), (1, 1)])
    gpd.GeoDataFrame(FLOWLINES.rename(columns=str.lower), geometry=[line] * 6, crs=crs).to_file(
        gpkg, layer="NHDFlowline", driver="GPKG")
    gpd.GeoDataFrame(CATCHMENTS.rename(columns=str.lower), geometry=[box(0, 0, 1, 1)] * 5,
                     crs=crs).to_file(gpkg, layer="NHDPlusCatchment", driver="GPKG")
    for name, frame in (("NHDPlusFlowlineVAA", VAA), ("NHDPlusFlow", FLOW)):
        pyogrio.write_dataframe(frame.rename(columns=str.lower), gpkg, layer=name, driver="GPKG")
    if with_water:
        for name, frame in (("NHDArea", AREAS), ("NHDWaterbody", WATERBODIES)):
            gpd.GeoDataFrame(frame.rename(columns=str.lower),
                             geometry=[box(0, 0, 1, 1)] * len(frame), crs=crs).to_file(
                gpkg, layer=name, driver="GPKG")
        gpd.GeoDataFrame(BURN.rename(columns=str.lower), geometry=[box(0, 0, 1, 1)] * len(BURN),
                         crs=crs).to_file(gpkg, layer="NHDPlusBurnWaterbody", driver="GPKG")


def test_diagnostics_script_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config").mkdir()
        (root / "config" / "diag_test.yaml").write_text(
            "case_study: diag_test\nhuc8s: ['01010001', '01010002', '01020001', '02040001']\n"
            "dam_source: x.csv\nwbd_source: x.gpkg\n")
        extracted = root / "data" / "processed" / "nhdplus_extracted"
        # 0101: one HU4 package covering two HUC8s. 0102: a package slimmed
        # before the water layers were kept. 0204: no package.
        for package, water in (("NHDPLUS_H_0101_HU4_20230713_GDB", True),
                               ("NHDPLUS_H_01020001_HU8_GDB", False)):
            (extracted / package).mkdir(parents=True)
            _write_package(extracted / package / f"{package}.gpkg", with_water=water)

        out = root / "diag"
        argv = ["nhdplus_diagnostics.py", "--case-study", "diag_test", "--output-dir", str(out)]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            nd.main()

        coverage = pd.read_csv(out / "huc8_coverage.csv", dtype={"huc8": str})
        assert coverage.set_index("huc8")["package_status"].to_dict() == {
            "01010001": "found", "01010002": "found", "01020001": "found",
            "02040001": "no_extracted_package"}
        summary = pd.read_csv(out / "package_summary.csv").set_index("package")
        assert len(summary) == 2  # the HU4 package is checked once
        full = summary.loc["NHDPLUS_H_0101_HU4_20230713_GDB"]
        assert full["status"] == "checked"
        assert full["pct_ap_with_polygon_id"] == 75.0
        assert full["exc_polygon_id_unresolved"] == 1
        slim = summary.loc["NHDPLUS_H_01020001_HU8_GDB"]
        assert slim["status"] == "checked_without_water_layers"
        assert slim["missing_layers"] == "NHDArea;NHDWaterbody;NHDPlusBurnWaterbody"
        exceptions = pd.read_csv(out / "exceptions.csv")
        assert set(exceptions["hu4"].astype(str).str.zfill(4)) == {"0101", "0102"}
        types = pd.read_csv(out / "polygon_types.csv")
        assert len(types) == 3  # the slim package has no polygons

        # --hu4 subset
        argv += ["--hu4", "0102"]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            nd.main()
        assert pd.read_csv(out / "package_summary.csv")["package"].tolist() == [
            "NHDPLUS_H_01020001_HU8_GDB"]


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
