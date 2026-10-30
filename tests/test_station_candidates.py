"""Checks for etl/station_candidates.py and etl/common/candidates.py on a
synthetic project (search radius 300 m), around 45 N:

    lake L1 (NHDWaterbody 390) with path 4                  pond P1 (no paths,
                                                             sink catchment)
    reach 2 Otter Brook (460, catchment C2) enters the river from the north
    wide river R1 (NHDArea 460, lat 44.997-45.003) with path 1 (Mill River,
    the centreline at lat 45.000)

Stations:
    S_BANK   inside R1 by the tributary mouth: the tributary is ~20 m away,
             the river's path ~310 m (beyond the radius) -> path 1 found
             through the polygon
    S_LAKE   inside L1, ~450 m from its path -> found through the polygon
    S_TRIB   in Otter Brook's catchment, ~390 m from it -> reach 2 (own) and
             Mill River (downstream) found through the catchment
    S_POND   in P1 and its sink catchment -> no candidates, flagged
    S_ASTRO  unsupported datum
    S_OTHER  in a HUC8 with no NHDPlus package

Run directly: `python tests/test_station_candidates.py`
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

from etl import station_candidates as sc
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC8, NO_PACKAGE_HUC8 = "01010001", "01010002"
CRS = "EPSG:4269"

# id: (name, ftype, line, totdasqkm, wbarea)
REACHES = {
    1: ("Mill River", 558, LineString([(-70.00, 45.0), (-69.95, 45.0)]), 100.0, "R1"),
    2: ("Otter Brook", 460, LineString([(-69.97, 45.010), (-69.97, 45.003)]), 4.0, None),
    4: ("Lake Brook", 558, LineString([(-69.94, 45.025), (-69.92, 45.025)]), 20.0, "L1"),
}
# name, lat, lon, datum
STATIONS = {
    "S_BANK": ("MILL RIVER AT BROOK", 45.0028, -69.9701, "NAD83"),
    "S_LAKE": ("LAKE SITE", 45.0290, -69.9210, "NAD83"),
    "S_TRIB": ("OTTER BROOK ABOVE MOUTH", 45.0060, -69.9750, "NAD83"),
    "S_POND": ("POND", 45.0350, -70.0300, "NAD83"),
    "S_ASTRO": ("OLD SURVEY", 45.0010, -69.9900, "ASTRO"),
    "S_OTHER": ("ELSEWHERE", 45.1000, -69.9800, "NAD83"),
}


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "cand_test.yaml").write_text(
        f'case_study: cand_test\nhuc8s: ["{HUC8}", "{NO_PACKAGE_HUC8}"]\n'
        "dam_source: x.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
        "station_candidates:\n  search_radius_m: 300\n")
    station_dir = root / "data" / "raw" / "wqp" / "results" / "cand_test"
    station_dir.mkdir(parents=True)
    pd.DataFrame([{
        "MonitoringLocationIdentifier": sid, "MonitoringLocationName": name,
        "LatitudeMeasure": lat, "LongitudeMeasure": lon,
        "HorizontalCoordinateReferenceSystemDatumName": datum,
        "source_huc8": HUC8, "MonitoringLocationTypeName": "River/Stream",
    } for sid, (name, lat, lon, datum) in STATIONS.items()]).to_csv(
        station_dir / "pilot_station_selection.csv", index=False)

    wbd_dir = root / "data" / "raw" / "wbd"
    wbd_dir.mkdir(parents=True)
    gpd.GeoDataFrame({"huc8": [HUC8, NO_PACKAGE_HUC8], "name": ["A", "B"]},
                     geometry=[box(-70.05, 44.95, -69.90, 45.05), box(-70.05, 45.05, -69.90, 45.15)],
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
    pyogrio.write_dataframe(pd.DataFrame({"nhdplusid": ids,
                                          "totdasqkm": [r[3] for r in REACHES.values()]}),
                            gpkg, layer="NHDPlusFlowlineVAA", driver="GPKG")
    # Otter Brook flows into Mill River.
    pyogrio.write_dataframe(pd.DataFrame({"fromnhdpid": [2.0, 1.0], "tonhdpid": [1.0, 0.0]}),
                            gpkg, layer="NHDPlusFlow", driver="GPKG")
    gpd.GeoDataFrame({"permanent_identifier": ["R1"], "nhdplusid": [500.0], "ftype": [460]},
                     geometry=[box(-70.0, 44.997, -69.95, 45.003)], crs=CRS).to_file(
        gpkg, layer="NHDArea", driver="GPKG")
    gpd.GeoDataFrame({"permanent_identifier": ["L1", "P1"], "nhdplusid": [600.0, 601.0],
                      "ftype": [390, 390]},
                     geometry=[box(-69.94, 45.02, -69.92, 45.03),
                               box(-70.032, 45.033, -70.028, 45.037)], crs=CRS).to_file(
        gpkg, layer="NHDWaterbody", driver="GPKG")
    gpd.GeoDataFrame({"nhdplusid": [600.0, 601.0], "onoffnet": [1, 1]},
                     geometry=[box(0, 0, 1, 1)] * 2, crs=CRS).to_file(
        gpkg, layer="NHDPlusBurnWaterbody", driver="GPKG")
    gpd.GeoDataFrame({"nhdplusid": [1.0, 2.0, 4.0, 601.0],
                      "sourcefc": ["NHDPlusBurnLineEvent"] * 3 + ["NHDPlusSink"]},
                     geometry=[box(-70.05, 44.95, -69.90, 45.003), box(-69.99, 45.003, -69.96, 45.015),
                               box(-69.95, 45.015, -69.90, 45.04), box(-70.05, 45.025, -70.02, 45.045)],
                     crs=CRS).to_file(gpkg, layer="NHDPlusCatchment", driver="GPKG")


def _run(root):
    out = root / "out"
    argv = ["station_candidates.py", "--case-study", "cand_test", "--output-dir", str(out)]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        sc.main()
    return out


def _candidates(out, station):
    frame = pd.read_parquet(out / f"station_candidates_{HUC8}.parquet")
    return frame[frame["MonitoringLocationIdentifier"] == station].set_index("NHDPlusID")


def test_station_candidates():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        out = _run(root)
        points = pd.read_parquet(out / f"station_points_{HUC8}.parquet") \
            .set_index("MonitoringLocationIdentifier")

        assert set(points.index) == {"S_BANK", "S_LAKE", "S_TRIB", "S_POND", "S_ASTRO"}
        assert points["candidate_status"].to_dict() == {
            "S_BANK": "candidates", "S_LAKE": "candidates", "S_TRIB": "candidates",
            "S_POND": "no_candidates", "S_ASTRO": "datum_unsupported"}
        assert points["water_layers_available"].all()
        # Reported coordinates are kept as given.
        assert points.loc["S_BANK", "LatitudeMeasure"] == 45.0028

        # Wide-river bank station: the tributary is nearest, but the river's
        # path comes through the polygon although it is beyond the radius.
        bank = _candidates(out, "S_BANK")
        assert points.loc["S_BANK", "point_polygon_layer"] == "NHDArea"
        assert points.loc["S_BANK", "point_polygon_ftype"] == 460
        assert points.loc["S_BANK", "nearest_eligible_id"] == 2
        assert bank.loc[1, "generation_methods"] == "polygon;catchment"
        assert bank.loc[1, "in_point_polygon"] and bank.loc[1, "path_polygon_layer"] == "NHDArea"
        assert 300 < bank.loc[1, "distance_m"] < 330
        assert bank.loc[2, "from_nearby"] and bank.loc[2, "distance_m"] < 30
        assert bank.loc[1, "name_match"] and not bank.loc[2, "name_match"]
        assert bank.loc[2, "distance_rank"] == 1 and bank.loc[1, "distance_rank"] == 2

        lake = _candidates(out, "S_LAKE")
        assert lake.loc[4, "from_polygon"] and not lake.loc[4, "from_nearby"]
        assert lake.loc[4, "path_polygon_ftype"] == 390

        # Tributary catchment: its own flowline and the main stem downstream.
        trib = _candidates(out, "S_TRIB")
        assert trib.loc[2, "catchment_relation"] == "own"
        assert trib.loc[1, "catchment_relation"] == "downstream"
        assert not trib["from_nearby"].any()

        # Isolated pond in a closed basin: flagged, no candidates.
        pond = points.loc["S_POND"]
        assert pond["in_sink_catchment"] and pond["n_point_polygons"] == 1
        assert not pond["point_polygon_has_paths"] and pond["n_candidates"] == 0

        other = pd.read_parquet(out / f"station_points_{NO_PACKAGE_HUC8}.parquet")
        assert other["candidate_status"].tolist() == ["no_nhdplus_package"]
        summary = pd.read_csv(out / "summary.csv", dtype={"huc8": str}).set_index("huc8")
        assert summary.loc[HUC8, "n_in_water_polygon"] == 2
        assert summary.loc[HUC8, "n_in_sink_catchment"] == 1
        assert summary.loc[NO_PACKAGE_HUC8, "n_no_nhdplus_package"] == 1


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
