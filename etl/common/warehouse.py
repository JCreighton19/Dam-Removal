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
