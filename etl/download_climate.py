"""
Download monthly PRISM precipitation and mean temperature for the years and
areas the eligible stations need, clipped to each NHDPlus package
(docs/warehouse_spec.md, "Step 5 context data: decisions" and section 3.10).

Source: PRISM AN monthly grids, 800 m, CONUS, 1895-present, from the free
PRISM web service (no login). Each month of each variable is one ~40 MB zip
holding a GeoTIFF (NAD83 lon/lat, nodata -9999).

PRISM allows each file to be downloaded at most twice per 24 hours, so:
- a zip is only deleted after all its clips are written (a failed run
  reuses it instead of downloading it again);
- a failed download is retried once at most;
- a response that isn't a zip (PRISM's "too many downloads" message) is
  never retried.

Which months
------------
All 12 months of every year in removal_year +/- wq_window for the
dam-station pairs with `has_both`, from 1895 up to the last year whose
months are all final (PRISM revises a month for 6 months).

Which areas
-----------
Every case-study NHDPlus package holding a catchment in station_catchment,
clipped to the extent of those catchments, as for land cover. Clips stay
on the PRISM grid and are written to

    data/raw/prism/<variable>/<package>/<variable>_<yyyymm>.tif

and shared across case studies. A re-run skips months whose clips all
exist. load_climate.py deletes the clips once it has used them.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rasterio
import requests

from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.warehouse import warehouse_dir
from etl.download_land_cover import clip_window, needed_packages, needed_years, write_clip

URL_TEMPLATE = "https://services.nacse.org/prism/data/get/us/800m/{variable}/{year}{month:02d}"
VARIABLES = ("ppt", "tmean")
FIRST_YEAR = 1895

# PRISM keeps revising a month for about 6 months before calling it final.
STABLE_AFTER_DAYS = 183

# Seconds between downloads, to be polite to the service.
PAUSE_SECONDS = 2

# Stop after this many failures in a row (e.g. the service is down).
MAX_CONSECUTIVE_FAILURES = 5


def last_stable_year(today=None):
    """The last year whose 12 months are all final."""
    return ((today or date.today()) - timedelta(days=STABLE_AFTER_DAYS)).year - 1


def clip_path(prism_dir, variable, package, year, month):
    return prism_dir / variable / package / f"{variable}_{year}{month:02d}.tif"


# --------------------------------------------------
# Download and clip
# --------------------------------------------------

class NotAZipError(Exception):
    """PRISM answered with a message instead of a zip (e.g. download limit)."""


def download_file(url, path):
    """Stream url to path via a .part file. One retry on network errors."""
    part = path.with_name(path.name + ".part")
    for attempt in (1, 2):
        try:
            with requests.get(url, stream=True, timeout=(30, 300)) as response:
                response.raise_for_status()
                with open(part, "wb") as out:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        out.write(chunk)
            if not zipfile.is_zipfile(part):
                message = part.read_bytes()[:300].decode("utf-8", "replace").strip()
                raise NotAZipError(f"not a zip: {message!r}")
            part.replace(path)
            return
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            print(f"    download failed ({exc}); retrying once", flush=True)
            time.sleep(60)
        finally:
            part.unlink(missing_ok=True)


def zip_tif(zip_path):
    """GDAL path of the one GeoTIFF inside a PRISM zip."""
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".tif")]
    if len(names) != 1:
        raise ValueError(f"Expected one .tif in {zip_path.name}, found {names}")
    return f"/vsizip/{zip_path}/{names[0]}"


def process_month(variable, year, month, packages, prism_dir):
    """Download one variable-month (unless a zip is left from a failed run),
    clip it to every package missing a clip, then delete the zip."""
    missing = {name: info for name, info in packages.items()
               if not clip_path(prism_dir, variable, name, year, month).exists()}
    if not missing:
        return "skipped"

    work_dir = prism_dir / "_downloads"
    work_dir.mkdir(parents=True, exist_ok=True)
    zip_path = work_dir / f"prism_{variable}_us_800m_{year}{month:02d}.zip"
    status = "reused zip"
    if not zip_path.exists():
        download_file(URL_TEMPLATE.format(variable=variable, year=year, month=month), zip_path)
        status = "downloaded"

    with rasterio.open(zip_tif(zip_path)) as src:
        for name, info in missing.items():
            window = clip_window(src, info["bounds"], info["crs"])
            write_clip(src, window, clip_path(prism_dir, variable, name, year, month))
    zip_path.unlink()
    return status


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
    last_year = last_stable_year()
    years = needed_years(
        db_path, int(wq_window.get("years_before", 3)), int(wq_window.get("years_after", 3)),
        first_year=FIRST_YEAR, last_year=last_year,
    )

    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = sorted({find_huc8_geodatabase(nhd_dir, h.zfill(8)) for h in cfg.huc8s})
    packages = needed_packages(db_path, gdb_paths)

    prism_dir = cfg.project_root / "data" / "raw" / "prism"
    print(f"Case study: {cfg.name}")
    print(f"Years ({len(years)}, final data through {last_year}): {years}")
    print(f"Variables: {', '.join(VARIABLES)}; files: {len(years) * 12 * len(VARIABLES)}")
    print(f"NHDPlus packages with eligible-station catchments: {len(packages)}")
    print(f"Output: {prism_dir}", flush=True)

    statuses = {}
    failed = []
    consecutive = 0
    for variable, year, month in [(v, y, m) for v in VARIABLES for y in years for m in range(1, 13)]:
        key = f"{variable} {year}-{month:02d}"
        try:
            statuses[key] = process_month(variable, year, month, packages, prism_dir)
            consecutive = 0
            if statuses[key] != "skipped":
                print(f"{key}: {statuses[key]}", flush=True)
                time.sleep(PAUSE_SECONDS)
        except Exception as exc:
            failed.append(key)
            consecutive += 1
            print(f"{key}: FAILED: {exc}", flush=True)
            if consecutive >= MAX_CONSECUTIVE_FAILURES:
                print(f"\nStopped after {consecutive} failures in a row.")
                break

    print("\nDone:", {s: list(statuses.values()).count(s) for s in sorted(set(statuses.values()))})
    if failed:
        print(f"Failed ({len(failed)}): {failed} -- re-run to retry them "
              "(each file can be downloaded at most twice per 24 hours)")
        sys.exit(1)


if __name__ == "__main__":
    main()
