"""Checks for etl/download_land_cover.py.

Replaces the real download with a tiny fake NLCD zip, builds one synthetic
NHDPlus package plus a warehouse.duckdb, runs the real `main()`, and checks
which years are fetched, that each clip covers the package on the NLCD
grid, that downloads are cleaned up, and that a re-run skips finished years.

Run directly: `python tests/test_download_land_cover.py`
"""

import os
import shutil
import sys
import tempfile
import zipfile
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

from etl import download_land_cover as dlc
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC_A, HUC_B = "01010001", "01010002"
# Package A has an eligible station's catchment; package B doesn't.
CATCHMENTS = {HUC_A: [(1, box(-70.00, 44.99, -69.99, 45.01))],
              HUC_B: [(7, box(-71.00, 44.99, -70.99, 45.01))]}
ALBERS = "EPSG:5070"
# A package outside the fake raster, like Puerto Rico outside CONUS NLCD.
HUC_PR = "21010002"
OUTSIDE = {HUC_PR: [(9, box(-66.10, 18.30, -66.09, 18.31))]}


def _fake_nlcd_zip(path):
    """A small 30 m 'CONUS' raster around the test area, zipped like NLCD."""
    left, bottom, right, top = transform_bounds("EPSG:4326", ALBERS, -71.2, 44.8, -69.8, 45.2)
    width, height = int((right - left) // 30) + 1, int((top - bottom) // 30) + 1
    tif = path.with_suffix(".tif")
    with rasterio.open(tif, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="uint8", crs=ALBERS, nodata=250,
                       transform=from_origin(left, top, 30, 30)) as dst:
        dst.write(np.full((height, width), 41, dtype="uint8"), 1)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(tif, "Annual_NLCD_LndCov_2000_CU_C1V2.tif")
    tif.unlink()


def _build_project(root, outside=False):
    catchments = {**CATCHMENTS, **(OUTSIDE if outside else {})}
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        "case_study: pilot_test\nhuc8s: [" + ", ".join(f'"{h}"' for h in catchments) + "]\n"
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
        "wq_window:\n  years_before: 1\n  years_after: 1\n"
    )
    for huc8, rows in catchments.items():
        gdb_dir = root / "data" / "processed" / "nhdplus_extracted" / f"NHDPLUS_H_{huc8}_HU8_GDB"
        gdb_dir.mkdir(parents=True)
        gpd.GeoDataFrame(
            {"nhdplusid": [float(i) for i, _ in rows], "areasqkm": [1.0] * len(rows)},
            geometry=[g for _, g in rows], crs="EPSG:4269",
        ).to_file(gdb_dir / f"NHDPLUS_H_{huc8}_HU8_GDB.gdb", layer="NHDPlusCatchment",
                  driver="OpenFileGDB")

    out = root / "data" / "warehouse" / "pilot_test"
    out.mkdir(parents=True)
    pairs = pd.DataFrame({"station_id": ["S1", "S2", "S3", "S4"],
                          "removal_year": [1986, 2010, 1990, 2010],
                          "has_both": [True, True, False, outside]})
    members = pd.DataFrame({"station_id": ["S1", "S2", "S4"], "catchment_id": [1, 1, 9]})
    members.to_parquet(out / "station_catchment.parquet")
    with duckdb.connect(str(out / "warehouse.duckdb")) as con:
        con.register("pairs_df", pairs)
        con.register("members_df", members)
        con.execute("CREATE TABLE eligibility_pairs AS SELECT * FROM pairs_df")
        con.execute("CREATE TABLE station_catchment AS SELECT * FROM members_df")
    return root / "data" / "raw" / "nlcd" / "annual_c1v2"


def _run(root, fake_zip):
    calls = []

    def fake_download(url, path):
        calls.append(url)
        shutil.copy(fake_zip, path)

    argv = ["download_land_cover.py", "--case-study", "pilot_test"]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv), \
            mock.patch.object(dlc, "download_file", fake_download):
        dlc.main()
    return calls


def test_years_clips_cleanup_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nlcd_dir = _build_project(root)
        fake_zip = root / "fake.zip"
        _fake_nlcd_zip(fake_zip)

        calls = _run(root, fake_zip)

        # Windows: 1985-1987 (1984 is before NLCD) and 2009-2011; S3 isn't eligible.
        years = [int(url.split("LndCov_")[1][:4]) for url in calls]
        assert years == [1985, 1986, 1987, 2009, 2010, 2011]

        # Only package A has an eligible station's catchment.
        package_a = nlcd_dir / f"NHDPLUS_H_{HUC_A}_HU8_GDB"
        assert sorted(p.name for p in package_a.iterdir()) == [
            f"LndCov_{y}.tif" for y in years
        ]
        assert not (nlcd_dir / f"NHDPLUS_H_{HUC_B}_HU8_GDB").exists()

        # The clip covers the catchment, stays on the 30 m grid, keeps values.
        with rasterio.open(package_a / "LndCov_2010.tif") as clip:
            catchment = transform_bounds("EPSG:4269", clip.crs, *CATCHMENTS[HUC_A][0][1].bounds)
            assert clip.bounds.left <= catchment[0] and clip.bounds.right >= catchment[2]
            assert clip.bounds.bottom <= catchment[1] and clip.bounds.top >= catchment[3]
            assert clip.res == (30.0, 30.0)
            assert clip.nodata == 250
            assert set(np.unique(clip.read(1))) == {41}

        # Zips and full rasters are deleted.
        assert list((nlcd_dir / "_downloads").iterdir()) == []

        # A re-run downloads nothing.
        assert _run(root, fake_zip) == []

        # A later attempt after success exits before reading any package.
        done = root / "data" / "processed" / "_progress" / "pilot_test" / "download_nlcd"
        assert (done / "complete").exists()
        with mock.patch.object(dlc, "needed_packages", side_effect=AssertionError("read")):
            assert _run(root, fake_zip) == []


def test_package_outside_the_raster_is_marked_not_failed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nlcd_dir = _build_project(root, outside=True)
        fake_zip = root / "fake.zip"
        _fake_nlcd_zip(fake_zip)

        # Every year succeeds: package A is clipped, the outside one marked.
        calls = _run(root, fake_zip)
        assert len(calls) == 6
        assert len(list((nlcd_dir / f"NHDPLUS_H_{HUC_A}_HU8_GDB").glob("*.tif"))) == 6
        outside_dir = nlcd_dir / f"NHDPLUS_H_{HUC_PR}_HU8_GDB"
        assert dlc.outside_coverage(outside_dir)
        assert not list(outside_dir.glob("*.tif"))

        # The marker stops a re-run from downloading for it again.
        assert _run(root, fake_zip) == []


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
