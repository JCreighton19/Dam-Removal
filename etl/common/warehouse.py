"""Shared helpers for writing and validating warehouse Parquet tables.

See docs/warehouse_spec.md, section 1 (layout and write rules) and
section 4 (validation checks).
"""

from __future__ import annotations

import os

import pandas as pd


def warehouse_dir(cfg):
    """data/warehouse/<case_study>/ for a loaded case-study config."""
    return cfg.project_root / "data" / "warehouse" / cfg.name


# Eligibility sensitivity variants (pre-specified 2026-11-07;
# docs/eligibility_sensitivity_prespec.md). The primary rule comes from the
# config's eligibility block (default 3 samples in 2 distinct years per side);
# a variant overrides it and writes its own warehouse_<name>.duckdb and
# analysis_measurement_<name>.parquet next to the primary files, which it
# never reads into, overwrites or modifies. min_months_single_year: a side
# with a single year of samples also needs samples in this many distinct
# calendar months (1 = no extra condition).
ELIGIBILITY_VARIANTS = {
    "min1yr": {"min_samples": 3, "min_years": 1, "min_months_single_year": 1,
               "label": "3 samples in >= 1 year per side (formal sensitivity analysis)"},
    "min1yr_2mo": {"min_samples": 3, "min_years": 1, "min_months_single_year": 2,
                   "label": "3 samples in >= 1 year per side; a single-year side needs >= 2 "
                            "months (secondary variant)"},
}


def check_variant(variant):
    """The variant's settings; ValueError for an unknown name."""
    if variant not in ELIGIBILITY_VARIANTS:
        raise ValueError(f"Unknown eligibility variant '{variant}'; choose from "
                         f"{sorted(ELIGIBILITY_VARIANTS)}")
    return ELIGIBILITY_VARIANTS[variant]


def views_db_path(parquet_dir, variant=None):
    """warehouse.duckdb (primary) or warehouse_<variant>.duckdb."""
    if variant is None:
        return parquet_dir / "warehouse.duckdb"
    check_variant(variant)
    return parquet_dir / f"warehouse_{variant}.duckdb"


def analysis_path(parquet_dir, variant=None):
    """analysis_measurement.parquet (primary) or analysis_measurement_<variant>.parquet."""
    if variant is None:
        return parquet_dir / "analysis_measurement.parquet"
    check_variant(variant)
    return parquet_dir / f"analysis_measurement_{variant}.parquet"


def progress_stage(stage, variant=None):
    """Progress-marker folder name: its own per variant, so a variant run is
    never skipped because the primary run of the stage already finished."""
    if variant is None:
        return stage
    check_variant(variant)
    return f"{stage}_{variant}"


def context_db_path(cfg, variant=None):
    """The views database whose eligible stations a context stage covers
    (build_station_watershed, download_nlcd, load_land_cover,
    download_climate, load_climate). Those stages write the shared context
    tables, so a variant must be at least as loose as the config's primary
    rule: its eligible stations then include every primary one, and the
    primary stations' context values don't change (they are per station and
    year), which build_analysis_dataset.py --verify-primary checks."""
    parquet_dir = warehouse_dir(cfg)
    if variant is None:
        return parquet_dir / "warehouse.duckdb"
    rule = check_variant(variant)
    eligibility = cfg.get("eligibility", {}) or {}
    primary = (int(eligibility.get("min_samples", 3)), int(eligibility.get("min_years", 2)))
    if rule["min_samples"] > primary[0] or rule["min_years"] > primary[1]:
        raise ValueError(f"Variant {variant} is stricter than the primary rule {primary}; context "
                         "stages run with it would drop primary stations' context")
    return views_db_path(parquet_dir, variant)


def add_variant_argument(parser, what):
    """--eligibility-variant for a stage (what: what the variant changes there)."""
    parser.add_argument(
        "--eligibility-variant", choices=sorted(ELIGIBILITY_VARIANTS), default=None,
        help=f"Sensitivity variant of the eligibility rule: {what}. Without it the primary "
             "rule from the config applies and nothing variant-specific is read or written.")


def write_parquet_atomic(frame, path):
    """Write to <path>.tmp, then rename over `path`, so a crashed job never
    leaves a half-written table."""
    tmp_path = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


def schema_errors(name, frame, schema, primary_key):
    """Column set, dtypes and primary key checks shared by every table.

    `schema` maps column name -> dtype string, in output order ("geometry"
    for a GeoPandas geometry column). Returns a list of messages, empty
    when the table conforms.
    """
    if list(frame.columns) != list(schema):
        return [f"{name}: columns {list(frame.columns)} != spec {list(schema)}"]

    errors = []
    for column, dtype in schema.items():
        actual = frame[column].dtype
        matches = (
            str(actual) == "geometry" if dtype == "geometry"
            else actual == pd.api.types.pandas_dtype(dtype)
        )
        if not matches:
            errors.append(f"{name}.{column}: dtype {actual}, expected {dtype}")

    key = frame[primary_key]
    if key.isna().any().any():
        errors.append(f"{name}: null primary key value(s)")
    if key.duplicated().any():
        dupes = key[key.duplicated(keep=False)].drop_duplicates().head(10)
        errors.append(f"{name}: duplicate primary key(s), e.g. {dupes.to_dict('records')}")

    return errors


# Tables later stages build from wq_station / dam_removal / wq_measurement,
# and the stage that rebuilds each. A station_id (or AR_ID, measurement_id)
# they hold that the core tables don't means the table comes from an earlier
# snapping or linking run.
DERIVED_TABLES = {
    "wq_measurement_daily": "build_daily_measurements.py",
    "station_watershed": "build_station_watershed.py",
    "station_catchment": "build_station_watershed.py",
    "land_cover": "load_land_cover.py",
    "climate": "load_climate.py",
    "analysis_measurement": "build_analysis_dataset.py",
}


def stale_reference_errors(con, tables=DERIVED_TABLES):
    """Messages for derived tables (views in `con`) that reference IDs the
    current core tables don't have. Tables not in `con` are skipped."""
    present = {row[0] for row in con.sql(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    checks = [(table, "station_id", "wq_station") for table in tables]
    checks += [("analysis_measurement", "AR_ID", "dam_removal"),
               ("analysis_measurement", "measurement_id", "wq_measurement")]
    errors = []
    for table, column, parent in checks:
        if table not in tables or table not in present:
            continue
        n = con.sql(f"""
            SELECT count(DISTINCT {column}) FROM {table}
            WHERE {column} NOT IN (SELECT {column} FROM {parent})
        """).fetchone()[0]
        if n:
            errors.append(f"{table}: {n:,} {column}(s) not in {parent}; stale, "
                          f"re-run {tables[table]}")
    return errors
