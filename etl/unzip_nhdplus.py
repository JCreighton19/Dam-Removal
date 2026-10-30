"""
Extract each downloaded NHDPlus HR package and keep only what the pipeline
reads, to save space.

For every zip in data/raw/nhdplus/ (from download_nhdplus.py):
1. Extract it to a temporary folder.
2. Copy the layers the pipeline uses (etl.common.nhdplus.SLIM_LAYERS: the
   core NHDFlowline, NHDPlusFlow, NHDPlusCatchment, NHDPlusFlowlineVAA,
   and the water layers NHDArea, NHDWaterbody, NHDPlusBurnWaterbody) into
       data/processed/nhdplus_extracted/<package>/<package>.gpkg
   (written to a .tmp file and renamed, so a half-written file is never
   mistaken for a finished one).
3. Delete the extracted geodatabase and the zip.

The full geodatabases hold dozens of other layers (lines, points, burn
line events, watershed boundaries, ...), so the GeoPackage is a fraction
of the size. download_nhdplus.py skips packages whose GeoPackage already
has every layer, so deleted zips aren't downloaded again.

A GeoPackage slimmed before the water layers were added gets only the
missing layers from the zip (added to a copy, then renamed); its existing
layers are not rewritten. A package without a water layer gets an empty
one, so it isn't downloaded again.

Packages extracted by the old unzip job (<package>/<package>.gdb with no
.gpkg) are converted the same way, and their geodatabase deleted.

Re-running skips finished packages, so a job stopped by the time limit
resumes where it left off.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
import pyogrio

from etl.common.nhdplus import CORE_LAYERS, SLIM_LAYERS, missing_slim_layers

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "data" / "raw" / "nhdplus"
OUT_DIR = ROOT / "data" / "processed" / "nhdplus_extracted"


def slim_path(out_dir, package):
    return out_dir / package / f"{package}.gpkg"


def slim_geodatabase(gdb_path, gpkg_path):
    """Copy SLIM_LAYERS from a FileGDB into a new GeoPackage."""
    tmp = gpkg_path.with_name(gpkg_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    copy_layers(gdb_path, tmp, SLIM_LAYERS)
    tmp.replace(gpkg_path)


def add_missing_layers(gdb_path, gpkg_path):
    """Add the SLIM_LAYERS a GeoPackage lacks, from the package's FileGDB.
    Written to a copy that replaces the original only when complete.
    Returns the layers added."""
    missing = missing_slim_layers(gpkg_path)
    if not missing:
        return []
    tmp = gpkg_path.with_name(gpkg_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    shutil.copy2(gpkg_path, tmp)
    try:
        copy_layers(gdb_path, tmp, missing)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(gpkg_path)
    return missing


def copy_layers(gdb_path, gpkg_path, layers):
    """Copy `layers` from a FileGDB into a GeoPackage (created if needed).
    A core layer missing from the FileGDB is an error; a missing water
    layer is written empty."""
    available = {str(name).lower(): str(name) for name, _ in pyogrio.list_layers(gdb_path)}
    missing = [layer for layer in layers
               if layer in CORE_LAYERS and layer.lower() not in available]
    if missing:
        raise ValueError(f"{gdb_path.name} is missing layers {missing}")

    with warnings.catch_warnings():
        # GDAL warns that the .tmp file isn't named .gpkg; it's renamed after.
        warnings.filterwarnings("ignore", message=".*non conformant file extension")
        for layer in layers:
            if layer.lower() in available:
                frame = pyogrio.read_dataframe(gdb_path, layer=available[layer.lower()])
            else:
                print(f"  {gdb_path.name} has no {layer}; writing it empty", flush=True)
                frame = empty_water_layer(gdb_path)
            pyogrio.write_dataframe(frame, gpkg_path, layer=layer, driver="GPKG",
                                    promote_to_multi=True)
            del frame


def empty_water_layer(gdb_path):
    """An empty polygon layer in the package's CRS (taken from NHDFlowline)."""
    crs = pyogrio.read_info(gdb_path, layer="NHDFlowline")["crs"]
    return gpd.GeoDataFrame(
        {"permanent_identifier": pd.Series(dtype="string"),
         "nhdplusid": pd.Series(dtype="float64"),
         "ftype": pd.Series(dtype="Int32")},
        geometry=gpd.GeoSeries([], crs=crs), crs=crs)


def find_gdb(folder):
    gdbs = sorted(folder.rglob("*.gdb"))
    if len(gdbs) != 1:
        raise ValueError(f"Expected one .gdb under {folder}, found {[g.name for g in gdbs]}")
    return gdbs[0]


def process_zip(zip_path, out_dir):
    package = zip_path.name.removesuffix(".zip")
    gpkg = slim_path(out_dir, package)
    if gpkg.exists():
        added = complete_from_zip(zip_path, gpkg, out_dir)
        zip_path.unlink()
        if added:
            return f"added {', '.join(added)} (zip deleted)"
        return "already slim (zip deleted)"
    if (out_dir / package / f"{package}.gdb").exists():
        status = process_old_extraction(out_dir / package)
        zip_path.unlink()
        return f"{status} (zip deleted)"

    tmp_dir = out_dir / f".tmp_{package}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        slim_geodatabase(find_gdb(tmp_dir), gpkg)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    # Anything an older extraction left next to the new GeoPackage.
    for leftover in gpkg.parent.iterdir():
        if leftover != gpkg:
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
    zip_path.unlink()
    return "extracted and slimmed"


def complete_from_zip(zip_path, gpkg, out_dir):
    """Add the layers an existing GeoPackage lacks from its zip."""
    if not missing_slim_layers(gpkg):
        return []
    tmp_dir = out_dir / f".tmp_{gpkg.stem}"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(tmp_dir)
        return add_missing_layers(find_gdb(tmp_dir), gpkg)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def process_old_extraction(package_dir):
    """A package extracted by the old unzip job: slim it, delete the rest."""
    package = package_dir.name
    gpkg = slim_path(package_dir.parent, package)
    if not gpkg.exists():
        slim_geodatabase(package_dir / f"{package}.gdb", gpkg)
    else:
        add_missing_layers(package_dir / f"{package}.gdb", gpkg)
    for leftover in package_dir.iterdir():
        if leftover != gpkg:
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
    return "converted existing geodatabase"


def folder_size_gb(path):
    if not path.exists():
        return 0.0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e9


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zips = sorted(ZIP_DIR.glob("NHDPLUS_H_*_GDB.zip"))
    # Old extractions whose zip is already gone (the rest are handled with their zip).
    old = sorted(
        d for d in OUT_DIR.glob("NHDPLUS_H_*_GDB")
        if (d / f"{d.name}.gdb").exists() and not (ZIP_DIR / f"{d.name}.zip").exists()
    )
    print(f"Zips to process: {len(zips)}; old extractions to convert: {len(old)}")
    print(f"Before: zips {folder_size_gb(ZIP_DIR):.1f} GB, "
          f"extracted {folder_size_gb(OUT_DIR):.1f} GB", flush=True)

    failed = []
    work = [(z, lambda z=z: process_zip(z, OUT_DIR)) for z in zips]
    work += [(d, lambda d=d: process_old_extraction(d)) for d in old]
    for path, step in work:
        try:
            status = step()
            print(f"{path.name}: {status}", flush=True)
        except Exception as exc:
            failed.append(path.name)
            print(f"{path.name}: FAILED: {exc}", flush=True)

    print(f"\nAfter: zips {folder_size_gb(ZIP_DIR):.1f} GB, "
          f"extracted {folder_size_gb(OUT_DIR):.1f} GB")
    if failed:
        print(f"Failed: {failed} -- re-run to retry")
        sys.exit(1)


if __name__ == "__main__":
    main()
