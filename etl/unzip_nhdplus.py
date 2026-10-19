"""
Extract each downloaded NHDPlus HR package and keep only what the pipeline
reads, to save space.

For every zip in data/raw/nhdplus/ (from download_nhdplus.py):
1. Extract it to a temporary folder.
2. Copy the four layers the pipeline uses (etl.common.nhdplus.SLIM_LAYERS:
   NHDFlowline, NHDPlusFlow, NHDPlusCatchment, NHDPlusFlowlineVAA) into
       data/processed/nhdplus_extracted/<package>/<package>.gpkg
   (written to a .tmp file and renamed, so a half-written file is never
   mistaken for a finished one).
3. Delete the extracted geodatabase and the zip.

The full geodatabases hold dozens of other layers (areas, waterbodies,
burn components, watershed boundaries, ...), so the GeoPackage is a
fraction of the size. download_nhdplus.py skips packages that already
have one, so deleted zips aren't downloaded again.

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

import pyogrio

from etl.common.nhdplus import SLIM_LAYERS

ROOT = Path(__file__).resolve().parents[1]
ZIP_DIR = ROOT / "data" / "raw" / "nhdplus"
OUT_DIR = ROOT / "data" / "processed" / "nhdplus_extracted"


def slim_path(out_dir, package):
    return out_dir / package / f"{package}.gpkg"


def slim_geodatabase(gdb_path, gpkg_path):
    """Copy SLIM_LAYERS from a FileGDB into a new GeoPackage."""
    available = {str(name).lower(): str(name) for name, _ in pyogrio.list_layers(gdb_path)}
    missing = [layer for layer in SLIM_LAYERS if layer.lower() not in available]
    if missing:
        raise ValueError(f"{gdb_path.name} is missing layers {missing}")

    tmp = gpkg_path.with_name(gpkg_path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    gpkg_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # GDAL warns that the .tmp file isn't named .gpkg; it's renamed below.
        warnings.filterwarnings("ignore", message=".*non conformant file extension")
        for layer in SLIM_LAYERS:
            frame = pyogrio.read_dataframe(gdb_path, layer=available[layer.lower()])
            pyogrio.write_dataframe(frame, tmp, layer=layer, driver="GPKG")
            del frame
    tmp.replace(gpkg_path)


def find_gdb(folder):
    gdbs = sorted(folder.rglob("*.gdb"))
    if len(gdbs) != 1:
        raise ValueError(f"Expected one .gdb under {folder}, found {[g.name for g in gdbs]}")
    return gdbs[0]


def process_zip(zip_path, out_dir):
    package = zip_path.name.removesuffix(".zip")
    gpkg = slim_path(out_dir, package)
    if gpkg.exists():
        zip_path.unlink()
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


def process_old_extraction(package_dir):
    """A package extracted by the old unzip job: slim it, delete the rest."""
    package = package_dir.name
    gpkg = slim_path(package_dir.parent, package)
    if not gpkg.exists():
        slim_geodatabase(package_dir / f"{package}.gdb", gpkg)
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
