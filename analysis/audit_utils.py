"""Helpers for notebooks/preprocessing_audit.ipynb, a read-only data-quality audit.

Nothing here writes to the warehouse:
- connect() opens an in-memory DuckDB database whose views point at the
  existing Parquet files, and rebuilds the build_views.py eligibility views
  from the same SQL. warehouse.duckdb is never opened, so Slurm jobs are
  never locked out.
- save_output() only writes inside the audit output folder.
- read_vaa() opens NHDPlus GeoPackages read-only.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import duckdb
import pandas as pd
import pyogrio

from etl.build_views import ELIGIBILITY_VIEWS, OPTIONAL_TABLES, TABLES, table_view_sql

# Tables the audit can't run without; the rest are optional.
CORE_TABLES = ["dam_removal", "wq_station", "dam_station_link", "wq_measurement"]


# --------------------------------------------------
# Connection and outputs
# --------------------------------------------------

def connect(parquet_dir, years_before, years_after, min_samples, min_years,
            threads=None, memory_limit=None, temp_dir=None):
    """In-memory DuckDB with one view per existing warehouse Parquet file,
    plus the eligibility views of build_views.py. Returns (con, views)."""
    parquet_dir = Path(parquet_dir)
    missing = [t for t in CORE_TABLES if not (parquet_dir / f"{t}.parquet").exists()]
    if missing:
        raise FileNotFoundError(f"Missing core warehouse tables in {parquet_dir}: {missing}")

    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_dir:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{Path(temp_dir).resolve()}'")

    views = []
    for table in TABLES + OPTIONAL_TABLES:
        path = parquet_dir / f"{table}.parquet"
        if path.exists():
            con.execute(table_view_sql(table, path))
            views.append(table)

    con.execute(f"""
        CREATE VIEW eligibility_settings AS SELECT
            {int(years_before)} AS years_before,
            {int(years_after)} AS years_after,
            {int(min_samples)} AS min_samples,
            {int(min_years)} AS min_years
    """)
    con.execute(ELIGIBILITY_VIEWS)
    return con, views


def save_output(frame, name, out_dir):
    """Write a derived table to out_dir/name (.parquet or .csv), via a .tmp
    file. Refuses any path outside out_dir, so warehouse files can't be hit."""
    out_dir = Path(out_dir).resolve()
    path = (out_dir / name).resolve()
    if out_dir not in path.parents:
        raise ValueError(f"{path} is outside the audit output folder {out_dir}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False)
    elif path.suffix == ".csv":
        frame.to_csv(tmp, index=False)
    else:
        raise ValueError(f"Unsupported output type: {path.suffix}")
    tmp.replace(path)
    return path


# --------------------------------------------------
# SQL snippets
# --------------------------------------------------

def series_sql(parameter="parameter_std", fraction="sample_fraction_std"):
    """One label per parameter x fraction, as in analysis_eda.ipynb:
    "phosphorus (Total)", "water_temp"."""
    return (f"CASE WHEN {fraction} IS NULL THEN {parameter} "
            f"ELSE {parameter} || ' (' || {fraction} || ')' END")


def haversine_m_sql(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon columns."""
    return (f"2 * 6371000 * asin(sqrt("
            f"pow(sin(radians({lat2} - {lat1}) / 2), 2) + "
            f"cos(radians({lat1})) * cos(radians({lat2})) * "
            f"pow(sin(radians({lon2} - {lon1}) / 2), 2)))")


def do_saturation_sql(temp_c):
    """Oxygen solubility (mg/L) in fresh water at 1 atm (Benson & Krause, as in
    APHA Standard Methods 4500-O). No pressure or salinity correction."""
    t = f"({temp_c} + 273.15)"
    return (f"exp(-139.34411 + 1.575701e5 / {t} - 6.642308e7 / pow({t}, 2) "
            f"+ 1.243800e10 / pow({t}, 3) - 8.621949e11 / pow({t}, 4))")


def do_saturation_mg_l(temp_c):
    """Python version of do_saturation_sql, for checks."""
    t = temp_c + 273.15
    return math.exp(-139.34411 + 1.575701e5 / t - 6.642308e7 / t ** 2
                    + 1.243800e10 / t ** 3 - 8.621949e11 / t ** 4)


# --------------------------------------------------
# Counting
# --------------------------------------------------

def counts(con, relation):
    """Rows and distinct measurements, stations, dams, dam-station pairs and
    series in a SQL relation, using whichever of those columns it has."""
    columns = set(con.sql(f"SELECT * FROM ({relation}) LIMIT 0").columns)
    exprs = ["count(*) AS rows"]
    if "measurement_id" in columns:
        exprs.append("count(DISTINCT measurement_id) AS measurements")
    if "station_id" in columns:
        exprs.append("count(DISTINCT station_id) AS stations")
    if "AR_ID" in columns:
        exprs.append("count(DISTINCT AR_ID) AS dams")
    if {"AR_ID", "station_id"} <= columns:
        exprs.append("count(DISTINCT AR_ID || '|' || station_id) AS pairs")
    if "series" in columns:
        exprs.append("count(DISTINCT series) AS series")
    elif "parameter_std" in columns:
        exprs.append(f"count(DISTINCT {series_sql()}) AS series")
    return con.sql(f"SELECT {', '.join(exprs)} FROM ({relation})").df().iloc[0].to_dict()


def counts_table(con, relations):
    """counts() for several named relations, one row each."""
    return pd.DataFrame([{"stage": name, **counts(con, sql)} for name, sql in relations.items()])


# --------------------------------------------------
# Robust outlier statistics
# --------------------------------------------------

def robust_z(con, source, group_cols, out_table, min_n, reference_filter="TRUE"):
    """Create temp table out_table: source's rows plus robust statistics of x
    within each group, and a robust z-score.

    `source` needs columns x (the value on the scale to screen; null when
    not evaluable, e.g. a zero on a log scale) and the group columns. The
    median and spread come from rows passing `reference_filter` (e.g.
    uncensored values), so they aren't pulled by the rows being screened.

    method, per row:
      not_evaluable  x is null
      sparse         fewer than min_n reference values in the group
      mad            z = (x - median) / (1.4826 * MAD)
      meanad         MAD is 0 (over half the values identical):
                     z = (x - median) / (1.2533 * mean absolute deviation)
      zero_spread    every reference value is the same; no z
    """
    groups = ", ".join(group_cols)
    on = " AND ".join(f"s.{c} IS NOT DISTINCT FROM r.{c}" for c in group_cols)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE {out_table} AS
        WITH ref AS (
            SELECT {groups}, x FROM ({source}) WHERE x IS NOT NULL AND ({reference_filter})
        ),
        centre AS (
            SELECT {groups}, count(*) AS n_ref, median(x) AS med FROM ref GROUP BY ALL
        ),
        spread AS (
            SELECT {", ".join(f"c.{c}" for c in group_cols)}, c.n_ref, c.med,
                   median(abs(ref.x - c.med)) AS mad,
                   avg(abs(ref.x - c.med)) AS meanad
            FROM ref JOIN centre c USING ({groups})
            GROUP BY ALL
        )
        SELECT r.*, s.n_ref, s.med, s.mad, s.meanad,
               CASE
                   WHEN r.x IS NULL THEN 'not_evaluable'
                   WHEN coalesce(s.n_ref, 0) < {int(min_n)} THEN 'sparse'
                   WHEN s.mad > 0 THEN 'mad'
                   WHEN s.meanad > 0 THEN 'meanad'
                   ELSE 'zero_spread'
               END AS method,
               CASE
                   WHEN r.x IS NULL OR coalesce(s.n_ref, 0) < {int(min_n)} THEN NULL
                   WHEN s.mad > 0 THEN (r.x - s.med) / (1.4826 * s.mad)
                   WHEN s.meanad > 0 THEN (r.x - s.med) / (1.2533 * s.meanad)
               END AS robust_z
        FROM ({source}) r
        LEFT JOIN spread s ON {on}
    """)


# --------------------------------------------------
# Unit-factor matching
# --------------------------------------------------

# Ratios a unit or analyte-basis mix-up would produce (value / correct value).
UNIT_FACTORS = {
    "x1000 (mg vs ug)": 1000.0,
    "/1000 (ug vs mg)": 0.001,
    "NO3 vs N (x4.43)": 62.004 / 14.007,
    "N vs NO3 (/4.43)": 14.007 / 62.004,
    "PO4 vs P (x3.07)": 94.971 / 30.974,
    "P vs PO4 (/3.07)": 30.974 / 94.971,
    "NH4 vs N (x1.29)": 18.039 / 14.007,
    "N vs NH4 (/1.29)": 14.007 / 18.039,
}


def nearest_factor(ratio, tolerance=0.15, factors=UNIT_FACTORS):
    """Label of the unit factor `ratio` is within `tolerance` (relative, on a
    log scale) of, or None. `tolerance` is one number, or a dict of label ->
    tolerance (a x1000 slip can afford a wider band than a x1.29 one).
    Ratios near 1 never match."""
    if ratio is None or not ratio > 0 or math.isnan(ratio):
        return None
    best, best_distance = None, math.inf
    for label, factor in factors.items():
        allowed = tolerance[label] if isinstance(tolerance, dict) else tolerance
        distance = abs(math.log(ratio) - math.log(factor))
        if distance <= math.log(1 + allowed) and distance < best_distance:
            best, best_distance = label, distance
    return best


# --------------------------------------------------
# Names
# --------------------------------------------------

# Words that say what kind of water body or where on it, not which one.
GENERIC_NAME_WORDS = {
    "river", "r", "riv", "creek", "cr", "crk", "ck", "brook", "bk", "br", "stream", "str",
    "run", "the", "of", "at", "near", "nr", "above", "abv", "ab", "below", "blw", "bl",
    "upstream", "downstream", "us", "ds", "and", "trib", "tributary", "to", "site", "station",
}


def name_tokens(name):
    """Lowercase alphanumeric tokens of a name, minus GENERIC_NAME_WORDS."""
    if name is None or (isinstance(name, float) and math.isnan(name)) or pd.isna(name):
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", str(name).lower()) if t not in GENERIC_NAME_WORDS}


def name_matches(station_name, reach_name):
    """True if every distinctive word of the reach's GNIS name appears in the
    station name; None when the reach has no name to compare."""
    reach = name_tokens(reach_name)
    if not reach:
        return None
    return reach <= name_tokens(station_name)


# --------------------------------------------------
# NHDPlus value-added attributes
# --------------------------------------------------

def read_vaa(package_paths, ids, columns, progress=False):
    """NHDPlusFlowlineVAA rows for the given NHDPlusIDs from each package
    (GeoPackage read with sqlite, read-only; FileGDB via pyogrio).

    `columns` are the wanted VAA fields (any case); missing ones are skipped
    and reported. Each package's whole VAA table is scanned (it has no index
    on NHDPlusID), so this is slow on a network filesystem. Returns (frame
    with lowercase columns + package, notes).
    """
    ids = pd.Series(pd.unique(pd.Series(list(ids)).dropna().astype("int64")))
    wanted = [c.lower() for c in columns if c.lower() != "nhdplusid"]
    frames, notes = [], []
    package_paths = list(package_paths)
    for number, path in enumerate(package_paths, 1):
        path = Path(path)
        start = time.time()
        if path.suffix == ".gpkg":
            frame, missing = _read_vaa_gpkg(path, ids, wanted)
        else:
            frame, missing = _read_vaa_pyogrio(path, ids, wanted)
        if progress:
            found = 0 if frame is None else len(frame)
            print(f"  [{number}/{len(package_paths)}] {path.name}: {found} reaches "
                  f"({time.time() - start:.0f} s)", flush=True)
        if missing:
            notes.append(f"{path.name}: no VAA column(s) {missing}")
        if frame is None:
            notes.append(f"{path.name}: no NHDPlusFlowlineVAA layer")
            continue
        frame["package"] = path.stem
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["nhdplusid", *wanted, "package"]), notes
    vaa = pd.concat(frames, ignore_index=True)
    vaa["nhdplusid"] = pd.to_numeric(vaa["nhdplusid"], errors="coerce").astype("Int64")
    return vaa.drop_duplicates("nhdplusid"), notes


def _read_vaa_gpkg(path, ids, wanted):
    with closing(sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True)) as db:
        tables = {name.lower(): name for (name,) in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'")}
        table = tables.get("nhdplusflowlinevaa")
        if table is None:
            return None, []
        fields = {row[1].lower(): row[1] for row in db.execute(f'PRAGMA table_info("{table}")')}
        present = [c for c in wanted if c in fields]
        missing = [c for c in wanted if c not in fields]
        db.execute("CREATE TEMP TABLE wanted_ids (id INTEGER PRIMARY KEY)")
        db.executemany("INSERT OR IGNORE INTO wanted_ids VALUES (?)", ((int(i),) for i in ids))
        select = ", ".join([f'CAST(v."{fields["nhdplusid"]}" AS INTEGER) AS nhdplusid']
                           + [f'v."{fields[c]}" AS {c}' for c in present])
        frame = pd.read_sql_query(
            f'SELECT {select} FROM "{table}" v '
            f'WHERE CAST(v."{fields["nhdplusid"]}" AS INTEGER) IN (SELECT id FROM wanted_ids)', db)
    return frame, missing


def _read_vaa_pyogrio(path, ids, wanted):
    layers = {str(name).lower(): str(name) for name, _ in pyogrio.list_layers(path)}
    if "nhdplusflowlinevaa" not in layers:
        return None, []
    layer = layers["nhdplusflowlinevaa"]
    fields = {str(f).lower(): str(f) for f in pyogrio.read_info(path, layer=layer)["fields"]}
    present = [c for c in wanted if c in fields]
    missing = [c for c in wanted if c not in fields]
    frame = pyogrio.read_dataframe(path, layer=layer, read_geometry=False,
                                   columns=[fields["nhdplusid"]] + [fields[c] for c in present])
    frame.columns = [str(c).lower() for c in frame.columns]
    frame["nhdplusid"] = pd.to_numeric(frame["nhdplusid"], errors="coerce").astype("Int64")
    return frame[frame["nhdplusid"].isin(set(ids))], missing
