"""Checks for etl/common/dam_matching.py and etl/match_dams.py on a
synthetic project, around 45 N:

    Mill River: reach 10 -> reservoir W1 (NHDWaterbody, path 11) -> reach 12
    W2: a reservoir with two outlets of similar size (paths 21 and 22)
    Otter Creek (reach 30): free-flowing, no reservoir

Dams:
    D_RES    at W1's outlet               -> structure 11, impounded 11, below 12
    D_NAME   same place, River names another stream -> ambiguous
    D_MULTI  at W2's main outlet          -> ambiguous (multiple_outlets)
    D_FREE   on Otter Creek               -> matched like a station (name_match)
    D_OVER   on Otter Creek, overridden to reject

Run directly: `python tests/test_dam_matching.py`
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

from etl import match_dams as md
from etl.common import dam_matching as dm
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC8 = "01010001"
CRS = "EPSG:4269"

# id: (name, ftype, line, totdasqkm, wbarea, divergence)
REACHES = {
    10: ("Mill River", 460, LineString([(-70.00, 45.0), (-69.98, 45.0)]), 90.0, None, 0),
    11: ("Mill River", 558, LineString([(-69.98, 45.0), (-69.96, 45.0)]), 100.0, "W1", 0),
    12: ("Mill River", 460, LineString([(-69.96, 45.0), (-69.94, 45.0)]), 101.0, None, 0),
    20: ("Pine Brook", 460, LineString([(-69.93, 45.035), (-69.92, 45.03)]), 50.0, None, 0),
    21: ("Pine Brook", 558, LineString([(-69.92, 45.03), (-69.92, 45.02)]), 50.0, "W2", 0),
    22: (None, 558, LineString([(-69.92, 45.03), (-69.91, 45.025)]), 45.0, "W2", 0),
    23: ("Pine Brook", 460, LineString([(-69.92, 45.02), (-69.92, 45.01)]), 50.0, None, 0),
    24: (None, 460, LineString([(-69.91, 45.025), (-69.905, 45.02)]), 45.0, None, 0),
    30: ("Otter Creek", 460, LineString([(-70.03, 45.04), (-70.00, 45.04)]), 20.0, None, 0),
}
FLOW = [(10, 11), (11, 12), (12, 0), (20, 21), (20, 22), (21, 23), (22, 24), (30, 0)]

# AR_ID: (river, lat, lon, year removed)
DAMS = {
    "D_RES": ("Mill River", 45.0003, -69.9605, 2010),
    "D_NAME": ("Beaver Creek", 45.0003, -69.9605, 2010),
    "D_MULTI": ("Pine Brook", 45.0203, -69.9202, 2012),
    "D_FREE": ("Otter Creek", 45.0402, -70.0200, 2015),
    "D_OVER": ("Otter Creek", 45.0401, -70.0100, 2015),
}


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "dam_test.yaml").write_text(
        f'case_study: dam_test\nhuc8s: ["{HUC8}"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n")
    (root / "config" / "snap_overrides.csv").write_text(
        "feature_kind,feature_id,action,nhdplus_id,reason,reviewed_by,reviewed_on\n"
        "dam,D_OVER,reject,,dam is on a canal (map),jc,2026-10-30\n")
    dam_dir = root / "data" / "raw" / "dams"
    dam_dir.mkdir(parents=True)
    pd.DataFrame([{"AR_ID": k, "Dam_Name": k, "River": r, "Latitude": lat, "Longitude": lon,
                   "Year_Removed": y} for k, (r, lat, lon, y) in DAMS.items()]).to_csv(
        dam_dir / "dams.csv", index=False)

    wbd_dir = root / "data" / "raw" / "wbd"
    wbd_dir.mkdir(parents=True)
    gpd.GeoDataFrame({"huc8": [HUC8], "name": ["A"]}, geometry=[box(-70.05, 44.95, -69.85, 45.06)],
                     crs=CRS).to_file(wbd_dir / "wbd.gpkg", layer="WBDHU8", driver="GPKG")

    package = f"NHDPLUS_H_{HUC8}_HU8_GDB"
    gpkg = root / "data" / "processed" / "nhdplus_extracted" / package / f"{package}.gpkg"
    gpkg.parent.mkdir(parents=True)
    ids = [float(i) for i in REACHES]
    gpd.GeoDataFrame({
        "nhdplusid": ids, "gnis_name": [r[0] for r in REACHES.values()],
        "ftype": [r[1] for r in REACHES.values()], "fcode": [r[1] * 100 for r in REACHES.values()],
        "wbarea_permanent_identifier": [r[4] for r in REACHES.values()],
    }, geometry=[r[2] for r in REACHES.values()], crs=CRS).to_file(
        gpkg, layer="NHDFlowline", driver="GPKG")
    pyogrio.write_dataframe(pd.DataFrame({
        "nhdplusid": ids, "totdasqkm": [r[3] for r in REACHES.values()],
        "divergence": [r[5] for r in REACHES.values()]}),
        gpkg, layer="NHDPlusFlowlineVAA", driver="GPKG")
    pyogrio.write_dataframe(pd.DataFrame({"fromnhdpid": [float(a) for a, _ in FLOW],
                                          "tonhdpid": [float(b) for _, b in FLOW]}),
                            gpkg, layer="NHDPlusFlow", driver="GPKG")
    gpd.GeoDataFrame({"permanent_identifier": ["W1", "W2"], "nhdplusid": [700.0, 701.0],
                      "ftype": [390, 390], "fdate": ["2015-06-01", "2005-01-01"]},
                     geometry=[box(-69.98, 44.998, -69.96, 45.002), box(-69.925, 45.02, -69.905, 45.031)],
                     crs=CRS).to_file(gpkg, layer="NHDWaterbody", driver="GPKG")
    gpd.GeoDataFrame({"permanent_identifier": pd.Series(dtype=str)}, geometry=[], crs=CRS).to_file(
        gpkg, layer="NHDArea", driver="GPKG")
    gpd.GeoDataFrame({"nhdplusid": [700.0, 701.0], "onoffnet": [1, 1]},
                     geometry=[box(0, 0, 1, 1)] * 2, crs=CRS).to_file(
        gpkg, layer="NHDPlusBurnWaterbody", driver="GPKG")
    gpd.GeoDataFrame({"nhdplusid": [12.0, 23.0, 30.0], "sourcefc": ["NHDPlusBurnLineEvent"] * 3},
                     geometry=[box(-70.05, 44.95, -69.85, 45.015), box(-69.95, 45.015, -69.85, 45.06),
                               box(-70.05, 45.015, -69.95, 45.06)], crs=CRS).to_file(
        gpkg, layer="NHDPlusCatchment", driver="GPKG")


def test_reservoir_outlets_and_diversions():
    flowlines = pd.DataFrame({
        "NHDPlusID": [21, 22, 25], "FType": [558, 558, 558], "wbarea_id": ["W2", "W2", "W2"],
        "totdasqkm": [50.0, 45.0, 50.0], "divergence": [1, 2, 0]})
    edges = pd.DataFrame({"from": [21, 22, 25], "to": [23, 24, 21]}).astype("Int64")
    outlets = dm.reservoir_outlets(["W2"], flowlines, edges).set_index("outlet_id")
    assert sorted(outlets.index) == [21, 22]          # 25 flows to 21 (inside W2)
    assert outlets.loc[21, "is_main_outlet"] and not outlets.loc[22, "is_main_outlet"]
    assert outlets.loc[22, "is_diversion"] and outlets.loc[21, "n_diversion_outlets"] == 1
    assert outlets.loc[21, "downstream_ids"] == "23" and outlets.loc[21, "n_paths"] == 3


def test_match_dams_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        argv = ["match_dams.py", "--case-study", "dam_test", "--output-dir", str(root / "out")]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            md.main()
        dams = pd.read_parquet(root / "out" / f"dam_reaches_{HUC8}.parquet").set_index("AR_ID")

        res = dams.loc["D_RES"]
        assert (res["match_status"], res["match_reason"]) == ("accepted", "reservoir_outlet")
        assert res["reservoir_status"] == "dam_at_outlet" and res["reservoir_id"] == "W1"
        assert res["NHDPlusID"] == 11                       # the structure flowline
        assert res["impounded_nhdplus_ids"] == "11" and res["downstream_nhdplus_ids"] == "12"
        assert res["n_outlets"] == 1 and res["dam_to_outlet_m"] < 60
        assert res["reservoir_mapped_after_removal"]       # polygon edited 2015 > removal 2010
        assert 0.9 < res["snap_fraction"] <= 1.0           # at the reservoir's downstream end
        assert res["Latitude"] == 45.0003 and res["datum_assumed"]  # reported, unchanged

        assert dams.loc["D_NAME", "match_reason"] == "reservoir_name_mismatch"
        assert pd.isna(dams.loc["D_NAME", "NHDPlusID"])
        assert dams.loc["D_NAME", "provisional_nhdplus_id"] == 11

        multi = dams.loc["D_MULTI"]
        assert (multi["match_status"], multi["match_reason"]) == ("ambiguous", "multiple_outlets")
        assert multi["n_outlets"] == 2 and multi["outlet_nhdplus_ids"] in ("21;22", "22;21")
        assert not multi["reservoir_mapped_after_removal"]

        free = dams.loc["D_FREE"]
        assert (free["match_status"], free["match_reason"]) == ("accepted", "name_match")
        assert free["reservoir_status"] == "no_reservoir_polygon" and free["NHDPlusID"] == 30
        assert pd.isna(free["impounded_nhdplus_ids"])

        over = dams.loc["D_OVER"]
        assert (over["match_status"], over["match_reason"]) == ("rejected", "manual_reject")
        assert over["auto_status"] == "accepted" and over["match_method"] == "manual_override"

        outlets = pd.read_parquet(root / "out" / f"dam_reservoir_outlets_{HUC8}.parquet")
        assert set(outlets.loc[outlets["AR_ID"] == "D_MULTI", "outlet_id"]) == {21, 22}


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
