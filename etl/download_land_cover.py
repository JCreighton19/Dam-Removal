"""
Download Annual NLCD land cover for the years and areas the eligible
stations need, clipped to each NHDPlus package (docs/warehouse_spec.md,
"Step 5 context data: decisions" and section 3.8).

Source: Annual NLCD Collection 1.2 (C1V2), CONUS, 1985-2025, from MRLC's
free direct-download server (no login). Each year is one ~1.35 GB zip.

Which years
-----------
Every year in removal_year +/- wq_window for the dam-station pairs with
`has_both` (eligibility_pairs in warehouse.duckdb), limited to the years
NLCD has.

Which areas
-----------
Every case-study NHDPlus package (HU8 or HU4 geodatabase) holding at least
one catchment in station_catchment.parquet (build_station_watershed.py).
Each year's raster is clipped to that package's catchment extent, on the
original NLCD 30 m grid (no reprojection), so every year lines up pixel
for pixel.

Per year: download the zip, extract the GeoTIFF, write one small clip per
package, then delete the zip and the full GeoTIFF. Clips are written to

    data/raw/nlcd/annual_c1v2/<package>/LndCov_<year>.tif

and shared across case studies. A re-run skips every year whose clips all
exist, so an interrupted job resumes where it stopped.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pyogrio
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.warehouse import warehouse_dir

URL_TEMPLATE = (
    "https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/"
    "Annual_NLCD_LndCov_{year}_CU_C1V2.zip"
)
FIRST_YEAR, LAST_YEAR = 1985, 2025

# Extra pixels around each package extent, so edge catchments are whole.
PAD_PIXELS = 10

DOWNLOAD_RETRIES = 3


# --------------------------------------------------
# What to download
# --------------------------------------------------

def needed_years(db_path, years_before, years_after, first_year=FIRST_YEAR, last_year=LAST_YEAR):
    """Years in some eligible pair's window that the source covers
    (NLCD by default)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        removal_years = [
            int(y) for (y,) in con.sql(
                "SELECT DISTINCT removal_year FROM eligibility_pairs WHERE has_both"
            ).fetchall()
        ]
    years = {
        year
        for removal in removal_years
        for year in range(removal - years_before, removal + years_after + 1)
    }
    return sorted(y for y in years if first_year <= y <= last_year)


def needed_packages(db_path, gdb_paths):
    """Packages holding at least one catchment upstream of an eligible
    station, with their catchment-layer extent and CRS."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        catchment_ids = {
            int(c) for (c,) in con.sql(
                "SELECT DISTINCT catchment_id FROM station_catchment"
            ).fetchall()
        }

    packages = {}
    for gdb_path in gdb_paths:
        fields = {str(f).lower(): str(f) for f in
                  pyogrio.read_info(gdb_path, layer="NHDPlusCatchment")["fields"]}
        ids = pyogrio.read_dataframe(
            gdb_path, layer="NHDPlusCatchment", columns=[fields["nhdplusid"]],
            read_geometry=False,
        ).iloc[:, 0]
        if catchment_ids.isdisjoint(int(i) for i in ids.dropna()):
            continue
        info = pyogrio.read_info(gdb_path, layer="NHDPlusCatchment", force_total_bounds=True)
        packages[gdb_path.stem] = {"bounds": info["total_bounds"], "crs": info["crs"]}
    return packages


def clip_path(nlcd_dir, package, year):
    return nlcd_dir / package / f"LndCov_{year}.tif"


# --------------------------------------------------
# Download and clip
# --------------------------------------------------

def download_file(url, path):
    """Stream url to path via a .part file, checking the size, with retries."""
    part = path.with_name(path.name + ".part")
    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                expected = int(response.headers.get("content-length", 0))
                with open(part, "wb") as out:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        out.write(chunk)
            if expected and part.stat().st_size != expected:
                raise IOError(f"got {part.stat().st_size:,} of {expected:,} bytes")
            part.replace(path)
            return
        except (requests.RequestException, IOError) as exc:
            part.unlink(missing_ok=True)
            if attempt == DOWNLOAD_RETRIES:
                raise
            print(f"    download failed ({exc}); retrying", flush=True)
            time.sleep(30 * attempt)


def extract_tif(zip_path, out_dir):
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".tif")]
        if len(names) != 1:
            raise ValueError(f"Expected one .tif in {zip_path.name}, found {names}")
        return Path(archive.extract(names[0], out_dir))


def clip_window(src, bounds, bounds_crs):
    """Pixel window covering bounds (plus padding), snapped to the grid."""
    left, bottom, right, top = transform_bounds(bounds_crs, src.crs, *bounds, densify_pts=21)
    window = from_bounds(left, bottom, right, top, transform=src.transform)
    col = math.floor(window.col_off) - PAD_PIXELS
    row = math.floor(window.row_off) - PAD_PIXELS
    width = math.ceil(window.col_off + window.width) + PAD_PIXELS - col
    height = math.ceil(window.row_off + window.height) + PAD_PIXELS - row
    return Window(col, row, width, height).intersection(Window(0, 0, src.width, src.height))


def write_clip(src, window, path):
    data = src.read(1, window=window)
    profile = src.profile.copy()
    profile.update(
        driver="GTiff", width=window.width, height=window.height,
        transform=src.window_transform(window),
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
    tmp.replace(path)


def process_year(year, packages, nlcd_dir):
    """Download one year, clip it to every package missing a clip, clean up."""
    missing = {name: info for name, info in packages.items()
               if not clip_path(nlcd_dir, name, year).exists()}
    if not missing:
        print(f"{year}: all {len(packages)} clips exist, skipped", flush=True)
        return "skipped"

    work_dir = nlcd_dir / "_downloads"
    work_dir.mkdir(parents=True, exist_ok=True)
    zip_path = work_dir / f"Annual_NLCD_LndCov_{year}_CU_C1V2.zip"
    tif_path = None

    try:
        print(f"{year}: downloading", flush=True)
        download_file(URL_TEMPLATE.format(year=year), zip_path)
        tif_path = extract_tif(zip_path, work_dir)
        zip_path.unlink()

        with rasterio.open(tif_path) as src:
            for name, info in missing.items():
                window = clip_window(src, info["bounds"], info["crs"])
                write_clip(src, window, clip_path(nlcd_dir, name, year))
                print(f"    {name}: {window.width:,} x {window.height:,} pixels", flush=True)
        return "downloaded"
    finally:
        zip_path.unlink(missing_ok=True)
        if tif_path is not None:
            tif_path.unlink(missing_ok=True)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    db_path = require_file(
        warehouse_dir(cfg) / "warehouse.duckdb", "warehouse.duckdb (run build_views.py first)"
    )
    require_file(
        warehouse_dir(cfg) / "station_catchment.parquet",
        "station_catchment table (run build_station_watershed.py first)",
    )

    wq_window = cfg.get("wq_window", {}) or {}
    years = needed_years(
        db_path, int(wq_window.get("years_before", 3)), int(wq_window.get("years_after", 3))
    )

    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = sorted({find_huc8_geodatabase(nhd_dir, h.zfill(8)) for h in cfg.huc8s})
    packages = needed_packages(db_path, gdb_paths)

    nlcd_dir = cfg.project_root / "data" / "raw" / "nlcd" / "annual_c1v2"
    print(f"Case study: {cfg.name}")
    print(f"Years ({len(years)}): {years[0] if years else '-'}-{years[-1] if years else '-'}")
    print(f"NHDPlus packages with eligible-station catchments: {len(packages)}")
    print(f"Output: {nlcd_dir}", flush=True)

    statuses = {}
    failed = []
    for year in years:
        try:
            statuses[year] = process_year(year, packages, nlcd_dir)
        except Exception as exc:
            failed.append(year)
            print(f"{year}: FAILED: {exc}", flush=True)

    print("\nDone:", {s: list(statuses.values()).count(s) for s in set(statuses.values())})
    if failed:
        print(f"Failed years: {failed} -- re-run to retry them")
        sys.exit(1)


if __name__ == "__main__":
    main()
