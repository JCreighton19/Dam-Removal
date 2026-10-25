"""
Build climate.parquet: monthly mean temperature and total precipitation
averaged over each eligible station's upstream area (docs/warehouse_spec.md,
section 3.10).

Inputs
------
- warehouse.duckdb: eligibility_pairs (which years each station needs) and
  station_catchment (which catchments are upstream of each station).
- station_watershed.parquet: each station's upstream area, for
  valid_fraction.
- The case study's NHDPlus HR geodatabases: catchment polygons.
- The PRISM clips from download_climate.py:
  data/raw/prism/<variable>/<package>/<variable>_<yyyymm>.tif

How it works
------------
A PRISM cell (~800 m) is often bigger than a whole NHDPlus catchment, so
cells aren't assigned to catchments by their centers. Instead, per NHDPlus
package:
1. Burn the package's needed catchments into a 30 m grid (Albers), as for
   land cover.
2. Project the PRISM grid onto that 30 m grid once (nearest cell), so each
   30 m pixel knows which PRISM cell it's in.
3. Count 30 m pixels per (catchment, PRISM cell) pair and add them up to
   stations through station_catchment: a sparse stations x cells weight
   matrix, i.e. how much of each station's upstream area lies in each cell.
4. For each month, a station's value is the weighted mean of its cells'
   values (one sparse matrix product), skipping nodata cells.
Packages are summed, so an upstream area spanning packages is handled.

After a successful write the clips are deleted (they're only an
intermediate; --keep-clips keeps them).

Reuse: rows of the existing climate table are kept for stations whose
upstream catchments (catchment_hash) are unchanged; only new or changed
stations are computed (etl/common/covariates.py).

Rows: one per station x year x month, for the years inside some eligible
pair's window for that station (1895 onward).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import sparse

from etl.build_views import add_view
from etl.common.config import default_case_study, load_config
from etl.common.covariates import catchment_hashes, split_reusable
from etl.common.io import require_file
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.progress import huc8s_with_outputs
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic
from etl.download_climate import FIRST_YEAR, VARIABLES, clip_path, last_stable_year
from etl.download_land_cover import outside_coverage
from etl.load_land_cover import (BLOCK_ROWS, PIXEL_KM2, combine, delete_clip_folders,
                                 package_catchments, read_catchments, read_inputs)

SOURCE = "prism_an_800m_monthly"
ALBERS = "EPSG:5070"
GRID_M = 30

# PRISM variable -> climate column.
COLUMNS = {"tmean": "temperature_mean_c", "ppt": "precipitation_mm"}

# Physical limits for a monthly CONUS value (validation).
RANGES = {"temperature_mean_c": (-45.0, 45.0), "precipitation_mm": (0.0, 3000.0)}

SCHEMA = {
    "station_id": "string",
    "year": "int16",
    "month": "int8",
    "temperature_mean_c": "float64",
    "precipitation_mm": "float64",
    "valid_fraction": "float64",
    "source": "string",
    "catchment_hash": "string",
}


# --------------------------------------------------
# Weights: how much of each station's area lies in each PRISM cell
# --------------------------------------------------

def catchment_zones(gdb_path, fids):
    """30 m Albers grid of catchment numbers (1..n, 0 = none) covering the
    package's needed catchments, and its transform."""
    polygons = read_catchments(gdb_path, fids).to_crs(ALBERS)
    left, bottom, right, top = polygons.total_bounds
    left, bottom = math.floor(left / GRID_M) * GRID_M, math.floor(bottom / GRID_M) * GRID_M
    right, top = math.ceil(right / GRID_M) * GRID_M, math.ceil(top / GRID_M) * GRID_M
    shape = (int((top - bottom) / GRID_M), int((right - left) / GRID_M))
    transform = from_origin(left, top, GRID_M, GRID_M)
    zones = rasterize(zip(polygons.geometry.values, range(1, len(polygons) + 1)),
                      out_shape=shape, transform=transform, fill=0, dtype="int32")
    return zones, transform


def catchment_cell_weights(zones, zone_transform, prism_src, n_catchments):
    """Sparse catchments x PRISM cells matrix of 30 m pixel counts."""
    n_cells = prism_src.height * prism_src.width
    cell_ids = np.arange(n_cells, dtype=np.int32).reshape(prism_src.height, prism_src.width)
    cells = np.full(zones.shape, -1, dtype=np.int32)
    reproject(cell_ids, cells, src_transform=prism_src.transform, src_crs=prism_src.crs,
              src_nodata=-1, dst_transform=zone_transform, dst_crs=ALBERS, dst_nodata=-1,
              resampling=Resampling.nearest)

    keys, counts = [], []
    for row in range(0, zones.shape[0], BLOCK_ROWS):
        z, c = zones[row:row + BLOCK_ROWS], cells[row:row + BLOCK_ROWS]
        inside = (z > 0) & (c >= 0)
        k, n = np.unique(z[inside].astype(np.int64) * n_cells + c[inside], return_counts=True)
        keys.append(k)
        counts.append(n)
    keys, counts = np.concatenate(keys), np.concatenate(counts)
    # Duplicate (catchment, cell) pairs from different blocks are summed.
    return sparse.csr_matrix(
        (counts.astype(np.float64), (keys // n_cells - 1, keys % n_cells)),
        shape=(n_catchments, n_cells),
    )


# --------------------------------------------------
# Monthly sums
# --------------------------------------------------

def station_sums(membership, station_ids, prism_dir, packages, station_years):
    """{(variable, year, month): (weighted sums, valid pixel counts)} per
    station, summed over packages."""
    station_index = {s: i for i, s in enumerate(station_ids)}
    totals = {}

    for gdb_path, catchments in packages.items():
        package = gdb_path.stem
        if any(outside_coverage(prism_dir / v / package) for v in VARIABLES):
            # Outside PRISM (download_climate.py): its stations get null climate.
            print(f"{package}: outside PRISM coverage, skipped", flush=True)
            continue
        position = pd.Series(np.arange(len(catchments)), index=catchments.index)
        rows = membership[membership["catchment_id"].isin(catchments.index)]
        stations_x_catchments = sparse.csr_matrix(
            (np.ones(len(rows)),
             (rows["station_id"].map(station_index), rows["catchment_id"].map(position))),
            shape=(len(station_ids), len(catchments)),
        )
        years = sorted(station_years.loc[
            station_years["station_id"].isin(rows["station_id"]), "year"].unique())
        months = [(y, m) for y in years for m in range(1, 13)]
        missing = [f"{v} {y}-{m:02d}" for v in VARIABLES for y, m in months
                   if not clip_path(prism_dir, v, package, y, m).exists()]
        if missing:
            raise FileNotFoundError(
                f"PRISM clips missing for {package}: {missing[:12]}"
                f"{' ...' if len(missing) > 12 else ''} (run download_climate.py)"
            )

        print(f"{package}: {len(catchments):,} catchments, {len(years)} years", flush=True)
        zones, zone_transform = catchment_zones(gdb_path, catchments)
        reference = clip_path(prism_dir, VARIABLES[0], package, *months[0])
        with rasterio.open(reference) as src:
            grid = (src.shape, src.transform)
            weights = stations_x_catchments @ catchment_cell_weights(
                zones, zone_transform, src, len(catchments))
        del zones

        for variable in VARIABLES:
            for year, month in months:
                with rasterio.open(clip_path(prism_dir, variable, package, year, month)) as src:
                    if (src.shape, src.transform) != grid:
                        raise ValueError(f"{src.name} grid differs from {reference}")
                    values = src.read(1).ravel().astype(np.float64)
                    valid = np.isfinite(values)
                    if src.nodata is not None:
                        valid &= values != src.nodata
                sums = weights @ np.where(valid, values, 0.0)
                pixels = weights @ valid.astype(np.float64)
                key = (variable, year, month)
                if key in totals:
                    sums, pixels = sums + totals[key][0], pixels + totals[key][1]
                totals[key] = (sums, pixels)
    return totals


def build_climate(totals, station_ids, station_years, station_area):
    """One row per station x needed year x month."""
    frames = []
    # Months only packages outside PRISM needed have no sums: all null.
    none = (np.zeros(len(station_ids)), np.zeros(len(station_ids)))
    months = {(y, m) for _, y, m in totals} | {
        (int(y), m) for y in station_years["year"].unique() for m in range(1, 13)}
    for year, month in sorted(months):
        frame = pd.DataFrame({"station_id": station_ids, "year": year, "month": month})
        pixels = []
        for variable, column in COLUMNS.items():
            sums, valid = totals.get((variable, year, month), none)
            frame[column] = np.where(valid > 0, sums / np.where(valid > 0, valid, 1), np.nan)
            pixels.append(valid)
        # Area with data for both variables, over the station's upstream area.
        frame["valid_pixels"] = np.minimum(*pixels)
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True).merge(station_years, on=["station_id", "year"])
    table["valid_fraction"] = (
        table["valid_pixels"] * PIXEL_KM2 / table["station_id"].map(station_area)
    ).clip(upper=1.0)
    table["source"] = SOURCE
    return table


# --------------------------------------------------
# Validation (spec section 4)
# --------------------------------------------------

def validate(table, station_years):
    errors = schema_errors("climate", table, SCHEMA, ["station_id", "year", "month"])
    if errors:
        return errors
    for column, (low, high) in RANGES.items():
        values = table[column].dropna()
        if not values.between(low, high).all():
            errors.append(f"{column} outside [{low}, {high}]: "
                          f"min {values.min():.1f}, max {values.max():.1f}")
    if not table["valid_fraction"].between(0, 1).all():
        errors.append("valid_fraction outside [0, 1]")
    if len(table) != len(station_years) * 12:
        errors.append(f"{len(table):,} rows, expected {len(station_years) * 12:,} "
                      "(one per needed station-year and month)")
    if table["catchment_hash"].isna().any():
        errors.append("catchment_hash is null")
    return errors


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
    parser.add_argument(
        "--keep-clips",
        action="store_true",
        help="Keep the downloaded clips after a successful run (by default "
        "they're deleted to save space; re-running needs a re-download).",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    output_dir = warehouse_dir(cfg)
    db_path = require_file(output_dir / "warehouse.duckdb",
                           "warehouse.duckdb (run build_views.py first)")
    watershed_path = require_file(output_dir / "station_watershed.parquet",
                                  "station_watershed table (run build_station_watershed.py first)")

    wq_window = cfg.get("wq_window", {}) or {}
    membership, station_years = read_inputs(
        db_path, int(wq_window.get("years_before", 3)), int(wq_window.get("years_after", 3)),
        first_year=FIRST_YEAR, last_year=last_stable_year(),
    )
    hashes = catchment_hashes(membership)
    kept, pending = split_reusable(output_dir, "climate", membership, station_years, 12)
    todo = membership[membership["station_id"].isin(pending["station_id"])]
    station_ids = np.array(sorted(todo["station_id"].unique()))
    station_area = pd.read_parquet(watershed_path, columns=["station_id", "area_km2"]) \
        .set_index("station_id")["area_km2"]

    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = sorted({find_huc8_geodatabase(nhd_dir, h) for h in huc8s_with_outputs(cfg)})
    packages, not_found = package_catchments(gdb_paths, todo["catchment_id"].unique())

    print(f"Case study: {cfg.name}")
    print(f"Stations: {membership['station_id'].nunique():,}; station-years: "
          f"{len(station_years):,}; to compute: {len(station_ids):,} stations, "
          f"{len(pending):,} station-years")
    print(f"Packages: {len(packages)}; catchments not found in any package: {len(not_found):,}",
          flush=True)

    prism_dir = cfg.project_root / "data" / "raw" / "prism"
    new = None
    if len(pending):
        totals = station_sums(todo, station_ids, prism_dir, packages, pending)
        new = build_climate(totals, station_ids, pending, station_area)
    table = combine(kept, new, hashes, SCHEMA)

    errors = validate(table, station_years)
    if errors:
        raise ValueError("Validation failed; no warehouse files were written:\n  - "
                         + "\n  - ".join(errors))

    path = output_dir / "climate.parquet"
    write_parquet_atomic(table, path)
    add_view(db_path, "climate", path)
    print(f"Wrote {len(table):,} rows to {path}")

    print(f"Station-months with no data: {int(table['temperature_mean_c'].isna().sum()):,}")
    print(f"Station-months with valid_fraction < 0.9: "
          f"{int((table['valid_fraction'] < 0.9).sum()):,} of {len(table):,}")

    if not args.keep_clips:
        freed = delete_clip_folders(
            [prism_dir / variable / gdb.stem for variable in VARIABLES for gdb in packages])
        print(f"Deleted the PRISM clips used ({freed:.1f} GB); --keep-clips keeps them")


if __name__ == "__main__":
    main()
