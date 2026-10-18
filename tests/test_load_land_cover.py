"""Checks for etl/load_land_cover.py.

Two synthetic NHDPlus packages, one catchment each, and fake NLCD clips for
two years where the answer is known:

    catchment 1 (package A): forest (41); in 2011 its west half is nodata
    catchment 2 (package B): crops (82) in 2010, developed (21) in 2011

Station S1 drains both catchments (so its counts span two packages);
station S2 drains only catchment 2.

Run directly: `python tests/test_load_land_cover.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box

from etl import load_land_cover as llc
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC_A, HUC_B = "01010001", "01010002"
CATCHMENTS = {
    HUC_A: (1, box(-70.03, 44.99, -70.01, 45.01)),
    HUC_B: (2, box(-69.99, 44.99, -69.97, 45.01)),
}
ALBERS = "EPSG:5070"


def _package(huc8):
    return f"NHDPLUS_H_{huc8}_HU8_GDB"


def _write_clip(path, year, huc8):
    """A 30 m clip around the package's catchment with known classes."""
    left, bottom, right, top = transform_bounds("EPSG:4269", ALBERS, *CATCHMENTS[huc8][1].bounds)
    left, top = left - 300, top + 300
    width, height = int((right + 300 - left) // 30) + 1, int((top - (bottom - 300)) // 30) + 1
    if huc8 == HUC_A:
        values = np.full((height, width), 41, dtype="uint8")
        if year == 2011:
            values[:, : width // 2] = 250  # nodata
    else:
        values = np.full((height, width), 82 if year == 2010 else 21, dtype="uint8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="uint8", crs=ALBERS, nodata=250,
                       transform=from_origin(left, top, 30, 30)) as dst:
        dst.write(values, 1)


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        f'case_study: pilot_test\nhuc8s: ["{HUC_A}", "{HUC_B}"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
        "wq_window:\n  years_before: 0\n  years_after: 1\n"
    )
    for huc8, (cid, geom) in CATCHMENTS.items():
        gdb_dir = root / "data" / "processed" / "nhdplus_extracted" / _package(huc8)
        gdb_dir.mkdir(parents=True)
        gpd.GeoDataFrame({"nhdplusid": [float(cid)], "areasqkm": [1.0]},
                         geometry=[geom], crs="EPSG:4269").to_file(
            gdb_dir / f"{_package(huc8)}.gdb", layer="NHDPlusCatchment", driver="OpenFileGDB")
        for year in (2010, 2011):
            _write_clip(root / "data" / "raw" / "nlcd" / "annual_c1v2" / _package(huc8)
                        / f"LndCov_{year}.tif", year, huc8)

    area = {cid: gpd.GeoSeries([g], crs="EPSG:4269").to_crs(ALBERS).area.iloc[0] / 1e6
            for cid, g in CATCHMENTS.values()}
    out = root / "data" / "warehouse" / "pilot_test"
    out.mkdir(parents=True)
    pd.DataFrame({"station_id": ["S1", "S2"], "area_km2": [area[1] + area[2], area[2]]}) \
        .to_parquet(out / "station_watershed.parquet")
    # S3's window is before NLCD starts (1985), so it gets no rows.
    members = pd.DataFrame({"station_id": ["S1", "S1", "S2", "S3"], "catchment_id": [1, 2, 2, 1]})
    pairs = pd.DataFrame({"station_id": ["S1", "S2", "S3"], "removal_year": [2010, 2010, 1970],
                          "has_both": [True, True, True]})
    with duckdb.connect(str(out / "warehouse.duckdb")) as con:
        con.register("members_df", members)
        con.register("pairs_df", pairs)
        con.execute("CREATE TABLE station_catchment AS SELECT * FROM members_df")
        con.execute("CREATE TABLE eligibility_pairs AS SELECT * FROM pairs_df")
    return out


def _run(root):
    argv = ["load_land_cover.py", "--case-study", "pilot_test"]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        llc.main()


def _percent(table, station, year):
    rows = table[(table["station_id"] == station) & (table["year"] == year)]
    return rows.set_index("class_code")["percent_area"]


def test_land_cover_percentages():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _build_project(root)
        _run(root)
        table = pd.read_parquet(out / "land_cover.parquet")

        # One row per station x year x class.
        assert len(table) == 2 * 2 * 16
        assert "S3" not in set(table["station_id"])

        # S2 only drains catchment 2: all crops, then all developed.
        assert _percent(table, "S2", 2010)[82] == 100
        assert _percent(table, "S2", 2011)[21] == 100
        assert _percent(table, "S2", 2010).drop(82).eq(0).all()

        # S1 spans both packages: roughly half forest, half crops in 2010.
        s1 = _percent(table, "S1", 2010)
        assert abs(s1[41] + s1[82] - 100) < 1e-9
        assert 35 < s1[41] < 65

        # 2011: half of catchment 1 is nodata, so percentages are over the
        # valid pixels and valid_fraction drops to about 0.75.
        s1_2011 = table[(table["station_id"] == "S1") & (table["year"] == 2011)]
        assert abs(s1_2011["percent_area"].sum() - 100) < 1e-9
        assert 0.6 < s1_2011["valid_fraction"].iloc[0] < 0.9
        valid = table[(table["station_id"] == "S2")]["valid_fraction"]
        assert (valid > 0.9).all()

        codes = pd.read_parquet(out / "land_cover_codes.parquet")
        assert len(codes) == 16 and set(table["class_code"]) <= set(codes["class_code"])

        with duckdb.connect(str(out / "warehouse.duckdb"), read_only=True) as con:
            assert con.sql("SELECT count(*) FROM land_cover").fetchone()[0] == len(table)


def test_missing_clip_is_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        (root / "data" / "raw" / "nlcd" / "annual_c1v2" / _package(HUC_B) / "LndCov_2011.tif").unlink()
        try:
            _run(root)
        except FileNotFoundError as exc:
            assert "2011" in str(exc)
        else:
            raise AssertionError("expected a missing-clip error")


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
