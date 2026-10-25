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
NLCD has. Station-years the existing land_cover table already holds for an
unchanged upstream area are not downloaded again (etl/common/covariates.py).

Which areas
-----------
Every case-study NHDPlus package (HU8 or HU4) holding at least one
catchment, in station_catchment.parquet (build_station_watershed.py), of a
station still to compute.
Each year's raster is clipped to the extent of those catchments (not the
whole package), on the original NLCD 30 m grid (no reprojection), so
every year lines up pixel for pixel. load_land_cover.py deletes the clips
once it has used them.

Per year: download the zip, extract the GeoTIFF, write one small clip per
package, then delete the zip and the full GeoTIFF. Clips are written to

    data/raw/nlcd/annual_c1v2/<package>/LndCov_<year>.tif

and shared across case studies. A re-run skips every year whose clips all
exist, so an interrupted job resumes where it stopped.

Annual NLCD covers only the contiguous US. A package whose catchments lie
entirely outside the raster (Alaska, Hawaii, Puerto Rico) gets no clips,
only an outside_coverage.txt marker, so it doesn't fail the year or make a
re-run download it again; its stations end up with null land cover
(missing_land_cover in analysis_measurement).
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
import time
import zipfile
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyogrio
import rasterio
import requests
from rasterio.errors import WindowError
from rasterio.warp import transform_bounds
from rasterio.windows import Window, from_bounds

from etl.common.config import default_case_study, load_config
from etl.common.covariates import eligible_station_years, split_reusable, window_years  # noqa: F401
from etl.common.io import require_file
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.progress import HucProgress, huc8s_with_outputs
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

def pending_work(cfg, db_path, table, first_year, last_year, rows_per_year,
                 nearest_if_outside=False):
    """(years to download, catchment IDs to clip) for the stations whose
    `table` rows can't be kept from the existing table."""
    wq_window = cfg.get("wq_window", {}) or {}
    membership, station_years = eligible_station_years(
        db_path, int(wq_window.get("years_before", 3)), int(wq_window.get("years_after", 3)),
        first_year, last_year, nearest_if_outside,
    )
    _, pending = split_reusable(warehouse_dir(cfg), table, membership, station_years,
                                rows_per_year)
    catchments = membership.loc[membership["station_id"].isin(pending["station_id"]),
                                "catchment_id"]
    return sorted(int(y) for y in pending["year"].unique()), {int(c) for c in catchments}


def needed_packages(catchment_ids, gdb_paths):
    """Packages holding at least one of the catchments, with the extent of
    those catchments (not the whole package, which can be far bigger) and
    their CRS."""
    packages = {}
    for gdb_path in gdb_paths:
        fields = {str(f).lower(): str(f) for f in
                  pyogrio.read_info(gdb_path, layer="NHDPlusCatchment")["fields"]}
        ids = pyogrio.read_dataframe(
            gdb_path, layer="NHDPlusCatchment", columns=[fields["nhdplusid"]],
            read_geometry=False, fid_as_index=True,
        ).iloc[:, 0]
        ids = pd.to_numeric(ids, errors="coerce")
        needed = ids[ids.isin(catchment_ids)]
        print(f"  {gdb_path.stem}: {len(needed):,} needed catchments", flush=True)
        if needed.empty:
            continue
        bounds = indexed_bounds(gdb_path, needed.index)
        if bounds is None:
            shapes = pyogrio.read_dataframe(gdb_path, layer="NHDPlusCatchment",
                                            fids=needed.index.tolist(), columns=[])
            bounds = tuple(shapes.total_bounds)
        info = pyogrio.read_info(gdb_path, layer="NHDPlusCatchment")
        packages[gdb_path.stem] = {"bounds": bounds, "crs": info["crs"]}
    return packages


def indexed_bounds(gdb_path, fids):
    """Combined extent of these catchment FIDs from the GeoPackage's spatial
    index, which holds each feature's bounding box. Reading polygons one FID
    at a time is a random read each, which on the shared filesystem took
    hours at national scale. None if there is no index (e.g. a .gdb)."""
    if gdb_path.suffix != ".gpkg":
        return None
    # immutable=1: read-only with no file locking, which can stall on NFS.
    with closing(sqlite3.connect(f"file:{gdb_path}?mode=ro&immutable=1", uri=True)) as con:
        row = con.execute(
            "SELECT table_name, column_name FROM gpkg_geometry_columns "
            "WHERE lower(table_name) = 'nhdpluscatchment'"
        ).fetchone()
        if row is None:
            return None
        rtree = f"rtree_{row[0]}_{row[1]}"
        if con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", (rtree,)).fetchone() is None:
            return None
        con.execute("CREATE TEMP TABLE wanted (id INTEGER PRIMARY KEY)")
        con.executemany("INSERT INTO wanted VALUES (?)", ((int(f),) for f in fids))
        bounds = con.execute(
            f'SELECT min(minx), min(miny), max(maxx), max(maxy) '
            f'FROM "{rtree}" JOIN wanted ON "{rtree}".id = wanted.id'
        ).fetchone()
    return None if bounds[0] is None else tuple(bounds)


def clip_path(nlcd_dir, package, year):
    return nlcd_dir / package / f"LndCov_{year}.tif"


# Written in a package's clip folder when its catchments are outside the
# raster, so later years and re-runs skip it instead of failing.
OUTSIDE_MARKER = "outside_coverage.txt"


def outside_coverage(package_dir):
    return (package_dir / OUTSIDE_MARKER).exists()


def mark_outside(package_dir, raster_name):
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / OUTSIDE_MARKER).write_text(
        f"Catchment extent is outside {raster_name}; no clips.\n")


def needs_clip(path):
    """A clip still to write: not on disk, and its package not outside the raster."""
    return not path.exists() and not outside_coverage(path.parent)


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
    """Pixel window covering bounds (plus padding), snapped to the grid, or
    None if bounds lie entirely outside the raster."""
    left, bottom, right, top = transform_bounds(bounds_crs, src.crs, *bounds, densify_pts=21)
    window = from_bounds(left, bottom, right, top, transform=src.transform)
    col = math.floor(window.col_off) - PAD_PIXELS
    row = math.floor(window.row_off) - PAD_PIXELS
    width = math.ceil(window.col_off + window.width) + PAD_PIXELS - col
    height = math.ceil(window.row_off + window.height) + PAD_PIXELS - row
    try:
        return Window(col, row, width, height).intersection(Window(0, 0, src.width, src.height))
    except WindowError:
        return None


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
               if needs_clip(clip_path(nlcd_dir, name, year))}
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
                if window is None:
                    mark_outside(nlcd_dir / name, "Annual NLCD (contiguous US)")
                    print(f"    {name}: outside NLCD coverage, skipped", flush=True)
                    continue
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
    # A later attempt queued by run_pipeline.sh after this stage already
    # succeeded has nothing to do; skip reading every package again.
    done_dir = HucProgress(cfg, "download_nlcd").folder
    if (done_dir / "complete").exists():
        print("download_nlcd already finished earlier in this run; nothing to do.")
        return

    db_path = require_file(
        warehouse_dir(cfg) / "warehouse.duckdb", "warehouse.duckdb (run build_views.py first)"
    )
    require_file(
        warehouse_dir(cfg) / "station_catchment.parquet",
        "station_catchment table (run build_station_watershed.py first)",
    )

    years, catchment_ids = pending_work(cfg, db_path, "land_cover", FIRST_YEAR, LAST_YEAR,
                                        rows_per_year=16, nearest_if_outside=True)

    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = sorted({find_huc8_geodatabase(nhd_dir, h) for h in huc8s_with_outputs(cfg)})
    packages = needed_packages(catchment_ids, gdb_paths) if years else {}

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
    (done_dir / "complete").touch()


if __name__ == "__main__":
    main()
