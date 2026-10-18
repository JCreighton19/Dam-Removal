"""Checks for etl/download_climate.py.

Replaces the real download with a tiny fake PRISM zip, builds one synthetic
NHDPlus package plus a warehouse.duckdb, runs the real `main()`, and checks
which months are fetched (including pre-1981 years), the clips, cleanup,
resume, and that PRISM's "not a zip" answer is never retried.

Run directly: `python tests/test_download_climate.py`
"""

import os
import shutil
import sys
import tempfile
import zipfile
from datetime import date
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

from etl import download_climate as dcl
from etl.common.config import PROJECT_ROOT_ENV_VAR

HUC_A, HUC_B = "01010001", "01010002"
# Package A has an eligible station's catchment; package B doesn't.
CATCHMENTS = {HUC_A: (1, box(-70.00, 44.99, -69.98, 45.01)),
              HUC_B: (7, box(-71.00, 44.99, -70.98, 45.01))}
CELL = 1 / 120  # PRISM 800 m (30 arc-second) cell


def _fake_prism_zip(path, name):
    """A small lon/lat float32 raster around the test area, zipped like PRISM."""
    width, height = int(1.4 / CELL), int(0.4 / CELL)
    tif = path.with_suffix(".tif")
    with rasterio.open(tif, "w", driver="GTiff", width=width, height=height, count=1,
                       dtype="float32", crs="EPSG:4269", nodata=-9999,
                       transform=from_origin(-71.2, 45.2, CELL, CELL)) as dst:
        dst.write(np.full((height, width), 12.5, dtype="float32"), 1)
    with zipfile.ZipFile(path, "w") as archive:
        archive.write(tif, f"{name}.tif")
        archive.writestr(f"{name}.info.txt", "fake")
    tif.unlink()


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        f'case_study: pilot_test\nhuc8s: ["{HUC_A}", "{HUC_B}"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
        "wq_window:\n  years_before: 1\n  years_after: 0\n"
    )
    for huc8, (cid, geom) in CATCHMENTS.items():
        name = f"NHDPLUS_H_{huc8}_HU8_GDB"
        gdb_dir = root / "data" / "processed" / "nhdplus_extracted" / name
        gdb_dir.mkdir(parents=True)
        gpd.GeoDataFrame({"nhdplusid": [float(cid)], "areasqkm": [1.0]},
                         geometry=[geom], crs="EPSG:4269").to_file(
            gdb_dir / f"{name}.gdb", layer="NHDPlusCatchment", driver="OpenFileGDB")

    out = root / "data" / "warehouse" / "pilot_test"
    out.mkdir(parents=True)
    # S1: 1950 (before NLCD, still in PRISM); S2: its window ends after the
    # last final PRISM year; S3: not eligible.
    pairs = pd.DataFrame({"station_id": ["S1", "S2", "S3"],
                          "removal_year": [1950, 2026, 2000],
                          "has_both": [True, True, False]})
    members = pd.DataFrame({"station_id": ["S1", "S2"], "catchment_id": [1, 1]})
    members.to_parquet(out / "station_catchment.parquet")
    with duckdb.connect(str(out / "warehouse.duckdb")) as con:
        con.register("pairs_df", pairs)
        con.register("members_df", members)
        con.execute("CREATE TABLE eligibility_pairs AS SELECT * FROM pairs_df")
        con.execute("CREATE TABLE station_catchment AS SELECT * FROM members_df")
    return root / "data" / "raw" / "prism"


def _run(root, fake_zip):
    calls = []

    def fake_download(url, path):
        calls.append(url)
        shutil.copy(fake_zip, path)

    argv = ["download_climate.py", "--case-study", "pilot_test"]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv), \
            mock.patch.object(dcl, "download_file", fake_download), \
            mock.patch.object(dcl, "last_stable_year", lambda: 2025), \
            mock.patch.object(dcl, "PAUSE_SECONDS", 0):
        dcl.main()
    return calls


def test_months_clips_cleanup_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prism_dir = _build_project(root)
        fake_zip = root / "fake.zip"
        _fake_prism_zip(fake_zip, "prism_tmean_us_30s_195001")

        calls = _run(root, fake_zip)

        # 1949-1950 for S1 and 2025 for S2 (2026 isn't final), 12 months, 2 variables.
        assert len(calls) == 3 * 12 * 2
        assert calls[0].endswith("/800m/ppt/194901") and calls[-1].endswith("/800m/tmean/202512")

        # Only package A has an eligible station's catchment.
        package_a = prism_dir / "tmean" / f"NHDPLUS_H_{HUC_A}_HU8_GDB"
        assert len(list(package_a.iterdir())) == 36
        assert not (prism_dir / "tmean" / f"NHDPLUS_H_{HUC_B}_HU8_GDB").exists()

        with rasterio.open(package_a / "tmean_195006.tif") as clip:
            assert clip.crs.to_epsg() == 4269 and clip.nodata == -9999
            left, bottom, right, top = clip.bounds
            catchment = CATCHMENTS[HUC_A][1].bounds
            assert left <= catchment[0] and bottom <= catchment[1]
            assert right >= catchment[2] and top >= catchment[3]
            assert set(np.unique(clip.read(1))) == {12.5}

        # Zips are deleted, and a re-run downloads nothing.
        assert list((prism_dir / "_downloads").iterdir()) == []
        assert _run(root, fake_zip) == []


def test_not_a_zip_is_not_retried():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.zip"
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.iter_content.return_value = [b"You have exceeded the download limit"]
        with mock.patch.object(dcl.requests, "get", return_value=response) as get:
            try:
                dcl.download_file("https://example.org/x", path)
            except dcl.NotAZipError as exc:
                assert "download limit" in str(exc)
            else:
                raise AssertionError("expected NotAZipError")
        assert get.call_count == 1
        assert not path.exists() and not path.with_name("x.zip.part").exists()


def test_last_stable_year():
    assert dcl.last_stable_year(date(2026, 10, 18)) == 2025
    assert dcl.last_stable_year(date(2027, 3, 1)) == 2025
    assert dcl.last_stable_year(date(2027, 8, 1)) == 2026


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
