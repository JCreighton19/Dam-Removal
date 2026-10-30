"""Checks for etl/unzip_nhdplus.py.

Builds a fake NHDPlus HR package (a FileGDB with the four core layers,
one water layer, and a layer the pipeline doesn't read), zips it, and
checks that the job leaves only a slim GeoPackage that the pipeline's
readers find (missing water layers written empty), deletes the zip and the
geodatabase, converts an old extraction, skips finished packages on a
re-run, and adds the water layers to a GeoPackage slimmed before they
were kept without rewriting its other layers.

Run directly: `python tests/test_unzip_nhdplus.py`
"""

import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
import pyogrio
from shapely.geometry import LineString, box

from etl import unzip_nhdplus as unz
from etl.common.nhdplus import (CORE_LAYERS, SLIM_LAYERS, find_huc8_geodatabase,
                                missing_slim_layers, read_nhdplus_layer)


def _make_gdb(path):
    crs = "EPSG:4269"
    gpd.GeoDataFrame({"nhdplusid": [1.0, 2.0], "lengthkm": [1.0, 2.0]},
                     geometry=[LineString([(0, 0), (1, 1)]), LineString([(1, 1), (2, 2)])],
                     crs=crs).to_file(path, layer="NHDFlowline", driver="OpenFileGDB")
    gpd.GeoDataFrame({"nhdplusid": [1.0, 2.0], "areasqkm": [1.0, 1.0]},
                     geometry=[box(0, 0, 1, 1), box(1, 1, 2, 2)], crs=crs).to_file(
        path, layer="NHDPlusCatchment", driver="OpenFileGDB", append=True)
    # A water layer (NHDArea and NHDPlusBurnWaterbody are left out: written empty).
    gpd.GeoDataFrame({"permanent_identifier": ["{AB-1}"], "nhdplusid": [9.0], "ftype": [390]},
                     geometry=[box(0, 0, 3, 3)], crs=crs).to_file(
        path, layer="NHDWaterbody", driver="OpenFileGDB", append=True)
    # A layer the pipeline doesn't read: must not be kept.
    gpd.GeoDataFrame({"x": [1]}, geometry=[LineString([(0, 0), (3, 3)])], crs=crs).to_file(
        path, layer="NHDLine", driver="OpenFileGDB", append=True)
    pyogrio.write_dataframe(pd.DataFrame({"fromnhdpid": [1.0], "tonhdpid": [2.0]}),
                            path, layer="NHDPlusFlow", driver="OpenFileGDB", append=True)
    pyogrio.write_dataframe(pd.DataFrame({"nhdplusid": [1.0, 2.0], "totdasqkm": [1.0, 2.0]}),
                            path, layer="NHDPlusFlowlineVAA", driver="OpenFileGDB", append=True)


def _make_zip(zip_dir, package, work):
    gdb = work / f"{package}.gdb"
    _make_gdb(gdb)
    zip_path = zip_dir / f"{package}.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for file in gdb.rglob("*"):
            archive.write(file, file.relative_to(work))
    shutil.rmtree(gdb)
    return zip_path


def _run(zip_dir, out_dir):
    with mock.patch.object(unz, "ZIP_DIR", zip_dir), mock.patch.object(unz, "OUT_DIR", out_dir), \
            mock.patch.object(sys, "argv", ["unzip_nhdplus.py"]):
        unz.main()


def test_slims_deletes_and_resumes():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_dir, out_dir, work = tmp / "zips", tmp / "extracted", tmp / "work"
        for d in (zip_dir, out_dir, work):
            d.mkdir()

        new = "NHDPLUS_H_01010001_HU8_GDB"
        zip_path = _make_zip(zip_dir, new, work)
        # An extraction left by the old unzip job (full .gdb, zip already gone).
        old = "NHDPLUS_H_0102_HU4_GDB"
        (out_dir / old).mkdir()
        _make_gdb(out_dir / old / f"{old}.gdb")

        _run(zip_dir, out_dir)

        for package in (new, old):
            gpkg = out_dir / package / f"{package}.gpkg"
            assert [p.name for p in (out_dir / package).iterdir()] == [gpkg.name]
            assert sorted(name for name, _ in pyogrio.list_layers(gpkg)) == sorted(SLIM_LAYERS)
        assert not zip_path.exists()
        assert not list(out_dir.glob(".tmp_*"))

        # The pipeline's readers find and read the slim packages.
        found = find_huc8_geodatabase(out_dir, "01010001")
        assert found.suffix == ".gpkg"
        flow = read_nhdplus_layer(found, "NHDPlusFlow", columns=["FromNHDPID", "ToNHDPID"])
        assert list(flow.columns) == ["FromNHDPID", "ToNHDPID"]
        assert find_huc8_geodatabase(out_dir, "01020003").name == f"{old}.gpkg"

        # A re-run has nothing to do.
        before = (out_dir / new / f"{new}.gpkg").stat().st_mtime
        _run(zip_dir, out_dir)
        assert (out_dir / new / f"{new}.gpkg").stat().st_mtime == before

        # Water layers: the real one copied, the absent ones written empty.
        gpkg = out_dir / new / f"{new}.gpkg"
        assert len(pyogrio.read_dataframe(gpkg, layer="NHDWaterbody")) == 1
        for layer in ("NHDArea", "NHDPlusBurnWaterbody"):
            assert len(pyogrio.read_dataframe(gpkg, layer=layer)) == 0


def test_adds_water_layers_to_old_slim_package():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_dir, out_dir, work = tmp / "zips", tmp / "extracted", tmp / "work"
        for d in (zip_dir, out_dir, work):
            d.mkdir()
        package = "NHDPLUS_H_0103_HU4_20230713_GDB"
        zip_path = _make_zip(zip_dir, package, work)

        # A GeoPackage slimmed before the water layers were kept.
        gpkg = out_dir / package / f"{package}.gpkg"
        gpkg.parent.mkdir()
        old = gpd.GeoDataFrame({"nhdplusid": [7.0]}, geometry=[LineString([(5, 5), (6, 6)])],
                               crs="EPSG:4269")
        for layer in CORE_LAYERS:
            pyogrio.write_dataframe(old, gpkg, layer=layer, driver="GPKG")
        assert missing_slim_layers(gpkg) == ["NHDArea", "NHDWaterbody", "NHDPlusBurnWaterbody"]

        status = unz.process_zip(zip_path, out_dir)

        assert status.startswith("added NHDArea, NHDWaterbody, NHDPlusBurnWaterbody")
        assert missing_slim_layers(gpkg) == []
        assert not zip_path.exists()
        assert not list(out_dir.glob(".tmp_*")) and not list(gpkg.parent.glob("*.tmp"))
        # The existing layers are the old ones, not re-copied from the zip.
        flowlines = pyogrio.read_dataframe(gpkg, layer="NHDFlowline")
        assert flowlines["nhdplusid"].tolist() == [7.0]
        assert len(pyogrio.read_dataframe(gpkg, layer="NHDWaterbody")) == 1


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
