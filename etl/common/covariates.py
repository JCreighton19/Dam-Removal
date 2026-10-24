"""Which station-years the land-cover and climate stages must compute, and
which rows they can keep from the existing tables (docs/warehouse_spec.md,
"Step 5 context data: decisions", Reuse).

A station's land cover and climate depend only on its upstream catchments
(station_catchment) and the year. Each row records the station's
catchment_hash; a later run keeps a row when the station's hash is
unchanged, and computes only new or changed stations, so a re-run after
re-snapping doesn't re-download NLCD/PRISM for every station.

Tables written before catchment_hash existed are matched through
station_catchment.prev.parquet, the membership build_station_watershed.py
replaced (kept by keep_previous_membership).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow.parquet as pq

PREVIOUS_MEMBERSHIP = "station_catchment.prev.parquet"
COVARIATE_TABLES = ["land_cover", "climate"]


def window_years(removal_year, years_before, years_after, first_year, last_year,
                 nearest_if_outside=False):
    """Years of one removal's window that the source covers. With
    nearest_if_outside, a year outside the source's range is replaced by the
    nearest covered year, so a window wholly before 1985 still gets NLCD
    1985 (land cover changes slowly; analysis rows flag the year used)."""
    years = range(removal_year - years_before, removal_year + years_after + 1)
    if nearest_if_outside:
        years = [min(max(y, first_year), last_year) for y in years]
    return {y for y in years if first_year <= y <= last_year}


def eligible_station_years(db_path, years_before, years_after, first_year, last_year,
                           nearest_if_outside=False):
    """station_catchment rows, and the (station, year) pairs the source
    covers inside some eligible pair's window."""
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
    return membership.reset_index(drop=True), station_years.reset_index(drop=True)


def catchment_hashes(membership):
    """station_id -> short hash of its sorted upstream catchment IDs."""
    if membership.empty:
        return pd.Series(dtype="string")
    return membership.groupby("station_id")["catchment_id"].agg(
        lambda ids: hashlib.sha256(
            ",".join(map(str, sorted(int(i) for i in ids))).encode()
        ).hexdigest()[:16]
    ).astype("string")


def keep_previous_membership(output_dir):
    """Before station_catchment is replaced: keep it as
    station_catchment.prev.parquet if a land_cover or climate table without
    catchment_hash was computed from it (and no copy is kept yet)."""
    output_dir = Path(output_dir)
    current = output_dir / "station_catchment.parquet"
    previous = output_dir / PREVIOUS_MEMBERSHIP
    if not current.exists() or previous.exists():
        return False
    for table in COVARIATE_TABLES:
        path = output_dir / f"{table}.parquet"
        if path.exists() and "catchment_hash" not in _columns(path):
            os.replace(current, previous)
            return True
    return False


def _columns(path):
    return set(pq.read_schema(path).names)


def split_reusable(output_dir, table, membership, station_years, rows_per_year):
    """(rows of the existing table to keep, station_years still to compute).

    A (station, year) is kept when the existing table has all
    rows_per_year rows for it (16 classes, 12 months) and the station's
    catchment_hash is unchanged.
    """
    output_dir = Path(output_dir)
    path = output_dir / f"{table}.parquet"
    current = catchment_hashes(membership)
    if not path.exists() or station_years.empty:
        return None, station_years

    existing = pd.read_parquet(path)
    if "catchment_hash" in existing.columns:
        old = existing.groupby("station_id")["catchment_hash"].first()
    else:
        previous = output_dir / PREVIOUS_MEMBERSHIP
        if not previous.exists():
            print(f"{table}: existing rows have no catchment_hash and no "
                  f"{PREVIOUS_MEMBERSHIP}; computing every station")
            return None, station_years
        old = catchment_hashes(pd.read_parquet(previous))
        existing["catchment_hash"] = existing["station_id"].map(old).astype("string")

    same = (current == old.reindex(current.index).astype("string")).fillna(False)
    unchanged = set(current.index[same.to_numpy(dtype=bool)])
    counts = existing.groupby(["station_id", "year"]).size()
    complete = set(counts.index[counts == rows_per_year])
    needed = list(zip(station_years["station_id"], station_years["year"]))
    keep = [(s, int(y)) in complete and s in unchanged for s, y in needed]
    keep = pd.Series(keep, index=station_years.index, dtype=bool)

    kept_keys = station_years[keep]
    kept = existing.merge(kept_keys, on=["station_id", "year"], how="inner")
    pending = station_years[~keep].reset_index(drop=True)
    print(f"{table}: keeping {len(kept_keys):,} station-years from the existing table "
          f"({kept_keys['station_id'].nunique():,} unchanged stations); "
          f"computing {len(pending):,} ({pending['station_id'].nunique():,} new or changed stations)")
    return kept, pending
