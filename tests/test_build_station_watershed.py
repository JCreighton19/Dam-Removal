"""Checks for etl/build_station_watershed.py.

Builds two synthetic NHDPlus HR geodatabases (two HUC8 packages joined by
one reach crossing between them) and a warehouse.duckdb, runs the real
`main()`, and checks which stations get an upstream area, the areas, and
the truncated flag.

Network (all flowing east):

    package B (01010002)      package A (01010001)
    [8 not downloaded] -> 9 ->  1 -> 2 -> 3
                                     5 -/        (5 drains into 3)

Run directly: `python tests/test_build_station_watershed.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import LineString, box

from etl import build_station_watershed as bsw
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC_A, HUC_B = "01010001", "01010002"

# reach -> (package, x position, catchment area km2, published total drainage area)
REACHES = {
    9: (HUC_B, 0, 4.0, 14.0),   # 8 (10 km2) upstream of it was not downloaded
    1: (HUC_A, 1, 1.0, 15.0),
    2: (HUC_A, 2, 2.0, 17.0),
    5: (HUC_A, 3, 3.0, 3.0),
    3: (HUC_A, 4, 5.0, 25.0),
}
FLOW = {
    HUC_B: [(8, 9)],
    HUC_A: [(9, 1), (1, 2), (2, 3), (5, 3)],
}

# S3 is eligible downstream of everything; S5 is an eligible headwater;
# S2 has data but is not eligible, so it gets no row.
STATIONS = pd.DataFrame({
    "station_id": ["S2", "S3", "S5"],
    "NHDPlusID": [2, 3, 5],
    "huc8": [HUC_A] * 3,
})
PAIRS = pd.DataFrame({"station_id": ["S2", "S3", "S5"], "has_both": [False, True, True]})


def _write_gdb(root, huc8):
    gdb_dir = root / "data" / "processed" / "nhdplus_extracted" / f"NHDPLUS_H_{huc8}_HU8_GDB"
    gdb_dir.mkdir(parents=True)
    gdb = gdb_dir / f"NHDPLUS_H_{huc8}_HU8_GDB.gdb"
    ids = [r for r, (pkg, *_) in REACHES.items() if pkg == huc8]
    x = {r: REACHES[r][1] * 0.01 - 70 for r in ids}

    # Lowercase field names, as in newer NHDPlus HR releases.
    gpd.GeoDataFrame(
        {"nhdplusid": [float(r) for r in ids], "lengthkm": [1.0] * len(ids)},
        geometry=[LineString([(x[r], 45.0), (x[r] + 0.01, 45.0)]) for r in ids],
        crs="EPSG:4269",
    ).to_file(gdb, layer="NHDFlowline", driver="OpenFileGDB")
    gpd.GeoDataFrame(
        {"nhdplusid": [float(r) for r in ids], "areasqkm": [REACHES[r][2] for r in ids]},
        geometry=[box(x[r], 44.99, x[r] + 0.01, 45.01) for r in ids],
        crs="EPSG:4269",
    ).to_file(gdb, layer="NHDPlusCatchment", driver="OpenFileGDB", mode="a")
    pyogrio.write_dataframe(
        pd.DataFrame({
            "fromnhdpid": [float(a) for a, _ in FLOW[huc8]],
            "tonhdpid": [float(b) for _, b in FLOW[huc8]],
        }),
        gdb, layer="NHDPlusFlow", driver="OpenFileGDB", append=True,
    )
    pyogrio.write_dataframe(
        pd.DataFrame({
            "nhdplusid": [float(r) for r in ids],
            "totdasqkm": [REACHES[r][3] for r in ids],
        }),
        gdb, layer="NHDPlusFlowlineVAA", driver="OpenFileGDB", append=True,
    )


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        f'case_study: pilot_test\nhuc8s: ["{HUC_A}", "{HUC_B}"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
    )
    for huc8 in (HUC_A, HUC_B):
        _write_gdb(root, huc8)

    out = root / "data" / "warehouse" / "pilot_test"
    out.mkdir(parents=True)
    with duckdb.connect(str(out / "warehouse.duckdb")) as con:
        con.register("stations_df", STATIONS)
        con.register("pairs_df", PAIRS)
        con.execute("CREATE TABLE wq_station AS SELECT * FROM stations_df")
        con.execute("CREATE TABLE eligibility_pairs AS SELECT * FROM pairs_df")
    return out


def _run(root):
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", ["build_station_watershed.py", "--case-study", "pilot_test"]):
        bsw.main()


def test_upstream_areas_and_truncation():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _build_project(root)
        _run(root)

        table = gpd.read_parquet(out / "station_watershed.parquet").set_index("station_id")
        assert list(table.index) == ["S3", "S5"]  # S2 isn't eligible

        # S3: catchments 9 (other package), 1, 2, 5 and its own 3.
        s3 = table.loc["S3"]
        assert s3["n_catchments"] == 5
        assert s3["area_km2"] == 15.0
        assert s3["nhdplus_total_area_km2"] == 25.0
        assert abs(s3["area_coverage"] - 0.6) < 1e-9
        assert s3["truncated"]  # reach 8 flows in from outside the downloads

        # S5: a headwater, complete.
        s5 = table.loc["S5"]
        assert s5["n_catchments"] == 1
        assert s5["area_km2"] == 3.0
        assert s5["area_coverage"] == 1.0
        assert not s5["truncated"]

        assert table.crs.to_epsg() == 4326
        assert table.geometry.notna().all()
        assert s3.geometry.area > s5.geometry.area

        with duckdb.connect(str(out / "warehouse.duckdb"), read_only=True) as con:
            assert con.sql("SELECT count(*) FROM station_watershed").fetchone()[0] == 2


def test_schema_matches_spec():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _build_project(root)
        _run(root)
        table = gpd.read_parquet(out / "station_watershed.parquet")
        assert bsw.schema_errors("station_watershed", table, bsw.SCHEMA, ["station_id"]) == []


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
