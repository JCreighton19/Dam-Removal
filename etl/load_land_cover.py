"""
Build land_cover.parquet and land_cover_codes.parquet: the land-cover mix
of each eligible station's upstream area, per year (docs/warehouse_spec.md,
sections 3.8 and 3.9).

Inputs
------
- warehouse.duckdb: eligibility_pairs (which years each station needs) and
  station_catchment (which catchments are upstream of each station), from
  build_views.py and build_station_watershed.py.
- station_watershed.parquet: each station's upstream area, for
  valid_fraction.
- The case study's NHDPlus HR geodatabases: catchment polygons.
- The Annual NLCD clips from download_land_cover.py:
  data/raw/nlcd/annual_c1v2/<package>/LndCov_<year>.tif

How it works
------------
For each NHDPlus package (all years of a package share one 30 m grid):
1. Burn the package's needed catchments into that grid once: each pixel
   gets the number of the catchment its center falls in.
2. For each year, count pixels of each land-cover class per catchment in
   one pass (numpy bincount), reading the raster in blocks.
3. Add the catchment counts up to stations through station_catchment.
Counts from different packages are summed, so a station whose upstream
area spans packages is handled.

After a successful write the clips are deleted (they're only an
intermediate; --keep-clips keeps them).

Rows: one per station x year x class (all 16 classes, zeros included),
for the years inside some eligible pair's window for that station. Window
years outside NLCD (1985-2025) are replaced by the nearest NLCD year, so a
1973 removal still gets 1985 land cover.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from scipy import sparse

from etl.build_views import add_view
from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.progress import HucProgress, huc8s_with_outputs
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic
from etl.download_land_cover import (FIRST_YEAR, LAST_YEAR, clip_path, indexed_bounds,
                                     window_years)

SOURCE = "annual_nlcd_c1v2"
PIXEL_KM2 = 30 * 30 / 1e6
BLOCK_ROWS = 2048

# Annual NLCD legend (CONUS): code -> (label, group).
LAND_COVER_CODES = {
    11: ("Open Water", "Water"),
    12: ("Perennial Ice/Snow", "Water"),
    21: ("Developed, Open Space", "Developed"),
    22: ("Developed, Low Intensity", "Developed"),
    23: ("Developed, Medium Intensity", "Developed"),
    24: ("Developed, High Intensity", "Developed"),
    31: ("Barren Land", "Barren"),
    41: ("Deciduous Forest", "Forest"),
    42: ("Evergreen Forest", "Forest"),
    43: ("Mixed Forest", "Forest"),
    52: ("Shrub/Scrub", "Shrubland"),
    71: ("Grassland/Herbaceous", "Herbaceous"),
    81: ("Pasture/Hay", "Agriculture"),
    82: ("Cultivated Crops", "Agriculture"),
    90: ("Woody Wetlands", "Wetlands"),
    95: ("Emergent Herbaceous Wetlands", "Wetlands"),
}
CLASS_CODES = np.array(sorted(LAND_COVER_CODES))

SCHEMA = {
    "station_id": "string",
    "year": "int16",
    "class_code": "int16",
    "area_km2": "float64",
    "percent_area": "float64",
    "valid_fraction": "float64",
    "source": "string",
}
CODES_SCHEMA = {"class_code": "int16", "class_label": "string", "class_group": "string"}


# --------------------------------------------------
# Inputs
# --------------------------------------------------

def read_inputs(db_path, years_before, years_after, first_year=FIRST_YEAR, last_year=LAST_YEAR,
                nearest_if_outside=False):
    """station_catchment rows, and the years each station needs that the
    source covers (NLCD by default; see window_years for
    nearest_if_outside)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        membership = con.sql("SELECT station_id, catchment_id FROM station_catchment").df()
        removal = con.sql("""
            SELECT DISTINCT station_id, removal_year FROM eligibility_pairs WHERE has_both
        """).df()

    station_years = {
        (row.station_id, year)
        for row in removal.itertuples(index=False)
        for year in window_years(int(row.removal_year), years_before, years_after,
                                 first_year, last_year, nearest_if_outside)
    }
    station_years = pd.DataFrame(sorted(station_years), columns=["station_id", "year"])
    station_years = station_years[station_years["station_id"].isin(membership["station_id"])]
    # Stations whose whole window is outside the source's years get no rows.
    membership = membership[membership["station_id"].isin(station_years["station_id"])]
    return membership, station_years


def package_catchments(gdb_paths, catchment_ids):
    """For each package, the FIDs/IDs of the needed catchments it holds.
    A catchment is assigned to the first package holding it."""
    remaining = set(catchment_ids)
    packages = {}
    for gdb_path in gdb_paths:
        fields = {str(f).lower(): str(f) for f in
                  pyogrio.read_info(gdb_path, layer="NHDPlusCatchment")["fields"]}
        ids = pyogrio.read_dataframe(
            gdb_path, layer="NHDPlusCatchment", columns=[fields["nhdplusid"]],
            read_geometry=False, fid_as_index=True,
        ).iloc[:, 0]
        ids = pd.to_numeric(ids, errors="coerce").dropna().astype("int64")
        held = ids[ids.isin(remaining)]
        if held.empty:
            continue
        held = held[~held.duplicated()]
        packages[gdb_path] = pd.Series(held.index, index=held.values, name="fid")
        remaining -= set(held.values)
    return packages, remaining


# --------------------------------------------------
# Counting
# --------------------------------------------------

def read_catchments(gdb_path, fids):
    """Catchment polygons for these FIDs, in the order given. Reads the area
    they cover in one pass via the spatial index: fetching one FID at a
    time is a random read each, which took hours per package on the shared
    filesystem at national scale."""
    bounds = indexed_bounds(gdb_path, fids)
    if bounds is None:
        return pyogrio.read_dataframe(gdb_path, layer="NHDPlusCatchment",
                                      fids=fids.tolist(), columns=[])
    polygons = pyogrio.read_dataframe(gdb_path, layer="NHDPlusCatchment", bbox=bounds,
                                      columns=[], fid_as_index=True)
    return polygons.loc[fids.to_numpy()].reset_index(drop=True)


def burn_catchments(gdb_path, fids, src):
    """Grid of catchment numbers (1..n, 0 = none) on the clip's grid."""
    polygons = read_catchments(gdb_path, fids).to_crs(src.crs)
    shapes = zip(polygons.geometry.values, range(1, len(polygons) + 1))
    return rasterize(shapes, out_shape=(src.height, src.width), transform=src.transform,
                     fill=0, dtype="int32")


def count_classes(zones, raster_path, n_zones):
    """(n_zones + 1) x 256 pixel counts of each raster value per zone."""
    counts = np.zeros((n_zones + 1) * 256, dtype=np.int64)
    with rasterio.open(raster_path) as src:
        if (src.height, src.width) != zones.shape:
            raise ValueError(f"{raster_path} grid differs from the package's other years")
        for row in range(0, src.height, BLOCK_ROWS):
            height = min(BLOCK_ROWS, src.height - row)
            values = src.read(1, window=Window(0, row, src.width, height))
            block = zones[row:row + height]
            inside = block > 0
            index = block[inside].astype(np.int64) * 256 + values[inside]
            counts += np.bincount(index, minlength=counts.size)
    return counts.reshape(n_zones + 1, 256)


def station_counts(membership, station_ids, nlcd_dir, packages, station_years, done_dir):
    """{year: stations x 256 pixel counts}, summed over packages. Each
    package's counts are saved to done_dir, so a run stopped by the time
    limit resumes after the last finished package."""
    station_index = {s: i for i, s in enumerate(station_ids)}
    totals = {}

    def add(year, rows, counts):
        full = np.zeros((len(station_ids), 256))
        full[rows] = counts
        totals[year] = totals.get(year, 0) + full

    for gdb_path, catchments in packages.items():
        package = gdb_path.stem
        saved = done_dir / f"{package}.npz"
        if saved.exists():
            with np.load(saved) as done:
                for year, counts in zip(done["years"], done["counts"]):
                    add(int(year), done["rows"], counts)
            print(f"{package}: done earlier in this run, skipped", flush=True)
            continue

        # Stations x catchments-in-this-package membership matrix.
        position = pd.Series(np.arange(len(catchments)), index=catchments.index)
        rows = membership[membership["catchment_id"].isin(catchments.index)]
        matrix = sparse.csr_matrix(
            (np.ones(len(rows)),
             (rows["station_id"].map(station_index), rows["catchment_id"].map(position))),
            shape=(len(station_ids), len(catchments)),
        )
        years = sorted(station_years.loc[
            station_years["station_id"].isin(rows["station_id"]), "year"].unique())
        missing = [y for y in years if not clip_path(nlcd_dir, package, y).exists()]
        if missing:
            raise FileNotFoundError(
                f"NLCD clips missing for {package}, years {missing} (run download_land_cover.py)"
            )

        print(f"{package}: {len(catchments):,} catchments, {len(years)} years", flush=True)
        with rasterio.open(clip_path(nlcd_dir, package, years[0])) as src:
            zones = burn_catchments(gdb_path, catchments, src)

        stations = np.unique(rows["station_id"].map(station_index).to_numpy())
        package_counts = []
        for year in years:
            counts = count_classes(zones, clip_path(nlcd_dir, package, year), len(catchments))
            by_station = matrix @ counts[1:]  # drop zone 0 (outside every catchment)
            package_counts.append(by_station[stations].astype(np.int64))
            add(year, stations, package_counts[-1])

        tmp = saved.with_name(saved.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, years=np.array(years), rows=stations,
                                counts=np.stack(package_counts))
        tmp.replace(saved)
    return totals


def build_land_cover(totals, station_ids, station_years, station_area, nodata):
    """Long table: one row per station x needed year x class."""
    frames = []
    for year, counts in sorted(totals.items()):
        counts = np.asarray(counts)
        valid = counts.sum(axis=1) - (counts[:, nodata] if nodata is not None else 0)
        class_counts = counts[:, CLASS_CODES]
        frame = pd.DataFrame({
            "station_id": np.repeat(station_ids, len(CLASS_CODES)),
            "year": year,
            "class_code": np.tile(CLASS_CODES, len(station_ids)),
            "pixels": class_counts.ravel(),
            "valid_pixels": np.repeat(valid, len(CLASS_CODES)),
        })
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True).merge(station_years, on=["station_id", "year"])
    table["area_km2"] = table["pixels"] * PIXEL_KM2
    # Null when a station-year has no valid pixels at all.
    table["percent_area"] = table["pixels"] / table["valid_pixels"].where(
        table["valid_pixels"] > 0) * 100
    # Valid pixel area over the station's upstream area. Pixel-center
    # burning differs slightly from the published areas, so capped at 1.
    table["valid_fraction"] = (
        table["valid_pixels"] * PIXEL_KM2 / table["station_id"].map(station_area)
    ).clip(upper=1.0)
    table["source"] = SOURCE

    table = table.sort_values(["station_id", "year", "class_code"]).reset_index(drop=True)
    return table[list(SCHEMA)].astype(SCHEMA)


def delete_clip_folders(folders):
    """Delete per-package clip folders once their data is in the warehouse.
    Returns the GB freed."""
    freed = 0
    for folder in folders:
        if folder.exists():
            freed += sum(f.stat().st_size for f in folder.rglob("*") if f.is_file())
            shutil.rmtree(folder)
    return freed / 1e9


def land_cover_codes():
    return pd.DataFrame(
        [(code, label, group) for code, (label, group) in sorted(LAND_COVER_CODES.items())],
        columns=list(CODES_SCHEMA),
    ).astype(CODES_SCHEMA)


# --------------------------------------------------
# Validation (spec section 4)
# --------------------------------------------------

def validate(table, codes, station_years):
    errors = schema_errors("land_cover", table, SCHEMA, ["station_id", "year", "class_code"])
    errors += schema_errors("land_cover_codes", codes, CODES_SCHEMA, ["class_code"])
    if errors:
        return errors
    if not table["class_code"].isin(codes["class_code"]).all():
        errors.append("land_cover.class_code not in land_cover_codes")
    sums = table.groupby(["station_id", "year"])["percent_area"].sum(min_count=1).dropna()
    if not ((sums - 100).abs() < 0.01).all():
        errors.append(f"percent_area doesn't sum to 100 for {int(((sums - 100).abs() >= 0.01).sum())} station-years")
    if not table["valid_fraction"].between(0, 1).all():
        errors.append("valid_fraction outside [0, 1]")
    if len(table) != len(station_years) * len(CLASS_CODES):
        errors.append(f"{len(table):,} rows, expected {len(station_years) * len(CLASS_CODES):,} "
                      "(one per needed station-year and class)")
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
    # A later attempt queued by run_pipeline.sh after this stage already
    # succeeded has nothing to do (and the clips may already be deleted).
    done_dir = HucProgress(cfg, "load_land_cover").folder
    if (done_dir / "complete").exists():
        print("land_cover already written earlier in this run; nothing to do.")
        return

    output_dir = warehouse_dir(cfg)
    db_path = require_file(output_dir / "warehouse.duckdb",
                           "warehouse.duckdb (run build_views.py first)")
    watershed_path = require_file(output_dir / "station_watershed.parquet",
                                  "station_watershed table (run build_station_watershed.py first)")

    wq_window = cfg.get("wq_window", {}) or {}
    membership, station_years = read_inputs(
        db_path, int(wq_window.get("years_before", 3)), int(wq_window.get("years_after", 3)),
        nearest_if_outside=True,
    )
    station_ids = np.array(sorted(membership["station_id"].unique()))
    station_area = pd.read_parquet(watershed_path, columns=["station_id", "area_km2"]) \
        .set_index("station_id")["area_km2"]

    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = sorted({find_huc8_geodatabase(nhd_dir, h) for h in huc8s_with_outputs(cfg)})
    packages, not_found = package_catchments(gdb_paths, membership["catchment_id"].unique())

    print(f"Case study: {cfg.name}")
    print(f"Stations: {len(station_ids):,}; station-years: {len(station_years):,}")
    print(f"Packages: {len(packages)}; catchments not found in any package: {len(not_found):,}",
          flush=True)

    nlcd_dir = cfg.project_root / "data" / "raw" / "nlcd" / "annual_c1v2"
    totals = station_counts(membership, station_ids, nlcd_dir, packages, station_years,
                            done_dir)

    first_clip = next(nlcd_dir.glob("*/LndCov_*.tif"))
    with rasterio.open(first_clip) as src:
        nodata = int(src.nodata) if src.nodata is not None else None

    table = build_land_cover(totals, station_ids, station_years, station_area, nodata)
    codes = land_cover_codes()

    errors = validate(table, codes, station_years)
    if errors:
        raise ValueError("Validation failed; no warehouse files were written:\n  - "
                         + "\n  - ".join(errors))

    for name, frame in [("land_cover", table), ("land_cover_codes", codes)]:
        path = output_dir / f"{name}.parquet"
        write_parquet_atomic(frame, path)
        add_view(db_path, name, path)
        print(f"Wrote {len(frame):,} rows to {path}")
    (done_dir / "complete").touch()

    low = table.groupby(["station_id", "year"])["valid_fraction"].first()
    print(f"Station-years with valid_fraction < 0.9: {int((low < 0.9).sum()):,} of {len(low):,}")

    if not args.keep_clips:
        freed = delete_clip_folders([nlcd_dir / gdb.stem for gdb in packages])
        print(f"Deleted the NLCD clips used ({freed:.1f} GB); --keep-clips keeps them")


if __name__ == "__main__":
    main()
