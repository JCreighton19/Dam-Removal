"""Checks for etl/load_climate.py.

Two synthetic NHDPlus packages, one catchment each (smaller than a few
PRISM cells), and fake PRISM clips for one year where the answer is known:

    tmean: 10 C in package A's clip, 20 C in package B's;
           in July, package A's clip is all nodata
    ppt:   100 + month mm everywhere

Station S1 drains both catchments (so its weights span two packages);
station S2 drains only catchment 2; S3's window is before PRISM (1895).

Run directly: `python tests/test_load_climate.py`
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
from shapely.geometry import box

from etl import download_land_cover as dlc
from etl import load_climate as lcl
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC_A, HUC_B = "01010001", "01010002"
CATCHMENTS = {
    HUC_A: (1, box(-70.03, 44.99, -70.01, 45.01)),
    HUC_B: (2, box(-69.99, 44.99, -69.97, 45.01)),
}
TMEAN = {HUC_A: 10.0, HUC_B: 20.0}
CELL = 1 / 120
YEAR = 2010


def _package(huc8):
    return f"NHDPLUS_H_{huc8}_HU8_GDB"


def _write_clip(path, value):
    """A PRISM-like lon/lat clip around both test catchments."""
    width, height = int(0.2 / CELL), int(0.1 / CELL)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="float32", crs="EPSG:4269", nodata=-9999,
                       transform=from_origin(-70.1, 45.05, CELL, CELL)) as dst:
        dst.write(np.full((height, width), value, dtype="float32"), 1)


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        f'case_study: pilot_test\nhuc8s: ["{HUC_A}", "{HUC_B}"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
        "wq_window:\n  years_before: 0\n  years_after: 0\n"
    )
    prism_dir = root / "data" / "raw" / "prism"
    for huc8, (cid, geom) in CATCHMENTS.items():
        gdb_dir = root / "data" / "processed" / "nhdplus_extracted" / _package(huc8)
        gdb_dir.mkdir(parents=True)
        gpd.GeoDataFrame({"nhdplusid": [float(cid)], "areasqkm": [1.0]},
                         geometry=[geom], crs="EPSG:4269").to_file(
            gdb_dir / f"{_package(huc8)}.gdb", layer="NHDPlusCatchment", driver="OpenFileGDB")
        for month in range(1, 13):
            tmean = -9999 if (huc8 == HUC_A and month == 7) else TMEAN[huc8]
            _write_clip(lcl.clip_path(prism_dir, "tmean", _package(huc8), YEAR, month), tmean)
            _write_clip(lcl.clip_path(prism_dir, "ppt", _package(huc8), YEAR, month), 100 + month)

    area = {cid: gpd.GeoSeries([g], crs="EPSG:4269").to_crs(lcl.ALBERS).area.iloc[0] / 1e6
            for cid, g in CATCHMENTS.values()}
    out = root / "data" / "warehouse" / "pilot_test"
    out.mkdir(parents=True)
    pd.DataFrame({"station_id": ["S1", "S2", "S3"],
                  "area_km2": [area[1] + area[2], area[2], area[1]]}) \
        .to_parquet(out / "station_watershed.parquet")
    members = pd.DataFrame({"station_id": ["S1", "S1", "S2", "S3"],
                            "catchment_id": [1, 2, 2, 1]})
    pairs = pd.DataFrame({"station_id": ["S1", "S2", "S3"],
                          "removal_year": [YEAR, YEAR, 1850],
                          "has_both": [True, True, True]})
    with duckdb.connect(str(out / "warehouse.duckdb")) as con:
        con.register("members_df", members)
        con.register("pairs_df", pairs)
        con.execute("CREATE TABLE station_catchment AS SELECT * FROM members_df")
        con.execute("CREATE TABLE eligibility_pairs AS SELECT * FROM pairs_df")
    return out, prism_dir


def _run(root):
    argv = ["load_climate.py", "--case-study", "pilot_test"]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv), \
            mock.patch.object(lcl, "last_stable_year", lambda: 2025):
        lcl.main()


def test_monthly_area_weighted_means():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out, _ = _build_project(root)
        _run(root)
        table = pd.read_parquet(out / "climate.parquet").set_index(["station_id", "month"])

        # One row per station x year x month; S3 is before PRISM.
        assert len(table) == 2 * 12
        assert "S3" not in table.index.get_level_values("station_id")

        # S2 only drains catchment 2.
        assert (table.loc["S2", "temperature_mean_c"] == 20).all()
        assert (table.loc["S2", "valid_fraction"] > 0.95).all()

        # S1 spans both packages: area-weighted between 10 and 20.
        june = table.loc[("S1", 6)]
        assert 13 < june["temperature_mean_c"] < 17
        assert june["valid_fraction"] > 0.95

        # July: package A is nodata, so S1's temperature is B's only and
        # only about half its area has data.
        july = table.loc[("S1", 7)]
        assert july["temperature_mean_c"] == 20
        assert 0.3 < july["valid_fraction"] < 0.7

        # Precipitation is the same everywhere: exact for both stations.
        expected = 100 + table.index.get_level_values("month")
        assert np.allclose(table["precipitation_mm"], expected)

        with duckdb.connect(str(out / "warehouse.duckdb"), read_only=True) as con:
            assert con.sql("SELECT count(*) FROM climate").fetchone()[0] == len(table)

        # The clips were an intermediate: deleted after the successful run.
        assert not list((root / "data" / "raw" / "prism").rglob("*.tif"))


def test_missing_clip_is_an_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _, prism_dir = _build_project(root)
        lcl.clip_path(prism_dir, "ppt", _package(HUC_B), YEAR, 3).unlink()
        try:
            _run(root)
        except FileNotFoundError as exc:
            assert "ppt 2010-03" in str(exc)
        else:
            raise AssertionError("expected a missing-clip error")


def test_package_outside_prism_gives_null_climate():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out, prism_dir = _build_project(root)
        # As download_climate.py leaves a package outside the raster.
        for variable in ("tmean", "ppt"):
            package_b = prism_dir / variable / _package(HUC_B)
            for clip in package_b.glob("*.tif"):
                clip.unlink()
            dlc.mark_outside(package_b, "test raster")
        _run(root)
        table = pd.read_parquet(out / "climate.parquet").set_index(["station_id", "month"])

        # Rows are kept; S2 (only catchment 2) has no climate.
        assert len(table) == 2 * 12
        assert table.loc["S2", "temperature_mean_c"].isna().all()
        assert (table.loc["S2", "valid_fraction"] == 0).all()
        assert (table.loc[("S1", 6), "temperature_mean_c"]) == 10


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
