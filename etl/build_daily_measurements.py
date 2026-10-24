"""
Build wq_measurement_daily.parquet: one value per station, day and kind of
reading, from wq_measurement (docs/warehouse_spec.md, section 3.6a).
wq_measurement itself is unchanged.

Grouping key (readings in one group are repeats of the same measurement):
    station_id, parameter_std, sample_fraction_std, sample_date,
    depth_m rounded to 0.1 m (null depth is its own group),
    activity_type (routine sample, field reading, composite, ...),
    method_id + method_context (analytical method),
    unit_basis_assumed (nutrient form stated vs assumed),
    turbidity_unit (turbidity only: NTU, FNU and NTRU come from different
    instruments and have no conversion, spec 3.6.4).
So depth profiles, different sample types and different methods on one
day stay separate rows; only true repeats are combined.

Value rule:
- the median of the group's uncensored values; censored values in the
  same group are left out (counted in n_censored_dropped), since a
  half-detection-limit stand-in would pull the median of real readings;
- if every value is censored, the median of the half-detection-limit
  values, with is_censored = true.
Quality-control replicates and blanks never reach this stage (filter 1,
spec 3.6.2), and result_unit_std is one unit per parameter, so values in a
group are directly comparable.

Before writing, the stage prints what the rule acts on: raw units per
parameter, and per parameter how many readings share a day, how many of
those days are depth profiles or mix sample types or methods, and the
spread inside combined groups. Check these on each new case study.

Run after clean_wq_results.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd

from etl.build_views import add_view
from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic

DATE = "date32[pyarrow]"

SCHEMA = {
    "daily_id": "int64",
    "station_id": "string",
    "parameter_std": "string",
    "sample_fraction_std": "string",
    "sample_date": DATE,
    "depth_m": "float64",
    "activity_type": "string",
    "method_id": "string",
    "method_context": "string",
    "unit_basis_assumed": "bool",
    "turbidity_unit": "string",
    "result_value_std": "float64",
    "result_unit_std": "string",
    "is_censored": "bool",
    "n_raw": "int64",
    "n_values": "int64",
    "n_censored_dropped": "int64",
    "value_min": "float64",
    "value_max": "float64",
    "provider": "string",
}

KEY = """station_id, parameter_std, sample_fraction_std, sample_date, depth_group, activity_type,
         method_id, method_context, unit_basis_assumed, turbidity_unit"""

KEYED = """
SELECT *,
       round(depth_m, 1) AS depth_group,
       CASE WHEN parameter_std = 'turbidity'
            THEN upper(replace(trim(result_unit_raw), ' ', '')) END AS turbidity_unit
FROM read_parquet('{path}')
"""

DAILY_SQL = f"""
WITH keyed AS ({KEYED}),
grouped AS (
    SELECT
        min(measurement_id) AS daily_id,
        {KEY},
        any_value(result_unit_std) AS result_unit_std,
        count(*) AS n_raw,
        count(*) FILTER (WHERE NOT is_censored) AS n_uncensored,
        median(result_value_std) FILTER (WHERE NOT is_censored) AS median_uncensored,
        min(result_value_std) FILTER (WHERE NOT is_censored) AS min_uncensored,
        max(result_value_std) FILTER (WHERE NOT is_censored) AS max_uncensored,
        median(result_value_std) AS median_all,
        min(result_value_std) AS min_all,
        max(result_value_std) AS max_all,
        min(provider) AS provider
    FROM keyed
    GROUP BY ALL
)
SELECT
    daily_id, station_id, parameter_std, sample_fraction_std, sample_date,
    depth_group AS depth_m, activity_type, method_id, method_context, unit_basis_assumed,
    turbidity_unit,
    CASE WHEN n_uncensored > 0 THEN median_uncensored ELSE median_all END AS result_value_std,
    result_unit_std,
    n_uncensored = 0 AS is_censored,
    n_raw,
    CASE WHEN n_uncensored > 0 THEN n_uncensored ELSE n_raw END AS n_values,
    CASE WHEN n_uncensored > 0 THEN n_raw - n_uncensored ELSE 0 END AS n_censored_dropped,
    CASE WHEN n_uncensored > 0 THEN min_uncensored ELSE min_all END AS value_min,
    CASE WHEN n_uncensored > 0 THEN max_uncensored ELSE max_all END AS value_max,
    provider
FROM grouped
ORDER BY station_id, parameter_std, sample_fraction_std, sample_date, daily_id
"""

# measurement_id -> daily_id (its group's smallest measurement_id), with the
# same key, so tables kept at the measurement grain (build_analysis_dataset)
# can be grouped exactly as wq_measurement_daily is.
DAILY_ID_SQL = f"""
SELECT measurement_id, min(measurement_id) OVER (PARTITION BY {KEY}) AS daily_id
FROM ({KEYED})
"""

UNITS_SQL = """
SELECT parameter_std, result_unit_std, result_unit_raw, count(*) AS rows
FROM read_parquet('{path}')
GROUP BY ALL
ORDER BY parameter_std, rows DESC
"""

STRUCTURE_SQL = f"""
WITH keyed AS ({KEYED}),
days AS (
    SELECT parameter_std, count(*) AS n,
           count(DISTINCT depth_group) AS depths,
           count(DISTINCT activity_type) AS types,
           count(DISTINCT coalesce(method_id, '-')) AS methods
    FROM keyed
    GROUP BY station_id, parameter_std, sample_fraction_std, sample_date
),
groups AS (
    SELECT parameter_std, count(*) AS n,
           bool_or(is_censored) AND NOT bool_and(is_censored) AS mixed,
           (max(result_value_std) - min(result_value_std)) / nullif(abs(median(result_value_std)), 0)
               AS spread
    FROM keyed
    GROUP BY {KEY}
)
SELECT
    d.parameter_std,
    sum(d.n)::BIGINT AS rows,
    count(*) AS station_days,
    round(100 * sum(d.n) FILTER (WHERE d.n > 1) / sum(d.n), 1) AS pct_rows_on_shared_days,
    count(*) FILTER (WHERE d.depths > 1) AS depth_profile_days,
    count(*) FILTER (WHERE d.types > 1) AS mixed_type_days,
    count(*) FILTER (WHERE d.methods > 1) AS mixed_method_days,
    any_value(g.daily_rows) AS daily_rows,
    any_value(g.mixed_censoring_groups) AS mixed_censoring_groups,
    any_value(g.median_spread) AS median_relative_spread
FROM days d
JOIN (
    SELECT parameter_std, count(*) AS daily_rows,
           count(*) FILTER (WHERE mixed) AS mixed_censoring_groups,
           round(median(spread) FILTER (WHERE n > 1), 3) AS median_spread
    FROM groups GROUP BY parameter_std
) g USING (parameter_std)
GROUP BY d.parameter_std
ORDER BY rows DESC
"""


def inspect(con, path):
    """Print the units and repeat structure the aggregation rule acts on."""
    units = con.sql(UNITS_SQL.format(path=path)).df()
    print("\n=== Units: raw units per parameter (result_unit_std must be one per parameter) ===")
    print(units.groupby(["parameter_std", "result_unit_std"])
          .apply(lambda g: ", ".join(f"{u} ({n:,})" for u, n in
                                     zip(g["result_unit_raw"].fillna("blank"), g["rows"])),
                 include_groups=False)
          .to_string())
    print("\n=== Readings sharing a station-day, by parameter ===")
    print(con.sql(STRUCTURE_SQL.format(path=path)).df().to_string(index=False))
    return units


def daily_ids(con, path):
    """measurement_id and daily_id for every row of wq_measurement."""
    return con.sql(DAILY_ID_SQL.format(path=path)).df()


def build_table(con, path):
    table = con.sql(DAILY_SQL.format(path=path)).df()
    table["sample_date"] = pd.to_datetime(table["sample_date"]).astype(DATE)
    return table.astype({c: t for c, t in SCHEMA.items() if c != "sample_date"})


def validate(table, measurement_rows, units):
    errors = schema_errors("wq_measurement_daily", table, SCHEMA, ["daily_id"])
    if errors:
        return errors
    if int(table["n_raw"].sum()) != measurement_rows:
        errors.append(f"groups cover {int(table['n_raw'].sum()):,} measurements, "
                      f"expected {measurement_rows:,} (each in exactly one group)")
    outside = ~table["result_value_std"].between(table["value_min"], table["value_max"])
    if outside.any():
        errors.append(f"{int(outside.sum())} value(s) outside their group's min-max")
    if (units.groupby("parameter_std")["result_unit_std"].nunique() > 1).any():
        errors.append("a parameter has more than one result_unit_std")
    return errors


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
    output_dir = warehouse_dir(cfg)
    source = require_file(output_dir / "wq_measurement.parquet",
                          "wq_measurement table (run clean_wq_results.py first)")
    path = str(source.resolve()).replace("'", "''")

    print(f"Case study: {cfg.name}")
    with duckdb.connect() as con:
        measurement_rows = con.sql(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        units = inspect(con, path)
        table = build_table(con, path)

    errors = validate(table, measurement_rows, units)
    if errors:
        raise ValueError("Validation failed; no warehouse files were written:\n  - "
                         + "\n  - ".join(errors))

    out = output_dir / "wq_measurement_daily.parquet"
    write_parquet_atomic(table, out)
    db_path = output_dir / "warehouse.duckdb"
    if db_path.exists():
        add_view(db_path, "wq_measurement_daily", out)
    print(f"\nWrote {len(table):,} daily rows from {measurement_rows:,} measurements "
          f"({100 * (1 - len(table) / max(measurement_rows, 1)):.1f}% fewer) to {out}")
    print(f"Days mixing censored and measured values: "
          f"{int((table['n_censored_dropped'] > 0).sum()):,} (censored readings left out)")


if __name__ == "__main__":
    main()
