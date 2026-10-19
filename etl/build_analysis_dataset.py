"""
Build analysis_measurement.parquet: the modeling table, one row per
measurement x eligible dam-station pair, with design, outcome, dam,
watershed, climate and land-cover columns (docs/warehouse_spec.md,
section 3.11).

Inputs
------
warehouse.duckdb views, so every rule lives in one place:
- measurement_window, eligibility_pairs (build_views.py)
- wq_measurement, dam_removal (core tables)
- station_watershed (build_station_watershed.py)
- land_cover, land_cover_codes (load_land_cover.py)
- climate (load_climate.py)

Design choices (agreed 2026-10-19)
----------------------------------
1. Rows: only pre / post measurements of dam x station x parameter x
   fraction pairs with has_both. The removal year and samples outside the
   window are left out.
2. A station linked to several dams gets one row per dam (as in
   measurement_window), so the key is (measurement_id, AR_ID).
3. Values are kept as cleaned: censored results are already half the
   detection limit (spec 3.6.3) and flagged by is_censored. Duplicate
   same-day samples are all kept; average them in modeling if needed.
4. Land cover is a fixed baseline per dam-station pair, not year by year:
   the last NLCD year before removal. Removal itself changes land cover
   (the drained reservoir becomes grass or wetland), so a year-by-year
   covariate would absorb part of the effect. If no year before removal
   exists (removals before 1986), the nearest year is used;
   land_cover_year shows which.
5. Climate is the sample month's (temperature, precipitation) plus the
   previous month's precipitation (rain before sampling drives turbidity
   and nutrients). The previous month is null for January of the first
   window year, which climate doesn't cover.
6. Context columns are left null rather than dropping rows; models decide
   what to exclude.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

from etl.build_views import add_view
from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic

REQUIRED_TABLES = ["station_watershed", "land_cover", "land_cover_codes", "climate"]

SCHEMA = {
    # Keys
    "measurement_id": "int64",
    "AR_ID": "string",
    "station_id": "string",
    "parameter_std": "string",
    "sample_fraction_std": "string",
    # Design
    "removal_year": "int16",
    "side": "string",
    "relationship": "string",
    "network_distance_km": "float64",
    "period": "string",
    "years_from_removal": "int16",
    # Time
    "sample_date": "datetime64[us]",
    "year": "int16",
    "month": "int8",
    "season": "string",
    # Outcome
    "result_value_std": "float64",
    "result_unit_std": "string",
    "is_censored": "bool",
    "unit_basis_assumed": "bool",
    "provider": "string",
    # Dam
    "dam_name": "string",
    "river": "string",
    "state": "string",
    "dam_height_ft": "float64",
    "year_built": "Int16",
    "dam_age_at_removal": "Int16",
    "original_use": "string",
    # Watershed
    "upstream_area_km2": "float64",
    "watershed_truncated": "boolean",
    # Climate
    "air_temp_mean_c": "float64",
    "precip_mm": "float64",
    "precip_prev_month_mm": "float64",
    # Land cover (baseline)
    "land_cover_year": "Int16",
    "pct_forest": "float64",
    "pct_developed": "float64",
    "pct_agriculture": "float64",
    "pct_wetlands": "float64",
    "pct_water": "float64",
    "pct_other": "float64",
}
PRIMARY_KEY = ["measurement_id", "AR_ID"]

ANALYSIS_SQL = """
WITH pairs AS (
    SELECT AR_ID, station_id, parameter_std, sample_fraction_std
    FROM eligibility_pairs
    WHERE has_both
),
rows AS (
    SELECT w.*
    FROM measurement_window w
    JOIN pairs p
      ON w.AR_ID = p.AR_ID
     AND w.station_id = p.station_id
     AND w.parameter_std = p.parameter_std
     AND w.sample_fraction_std IS NOT DISTINCT FROM p.sample_fraction_std
    WHERE w.period IN ('pre', 'post')
),
-- Land-cover groups as columns (they sum to 100 per station-year).
land_groups AS (
    SELECT
        l.station_id,
        l.year,
        sum(l.percent_area) FILTER (WHERE c.class_group = 'Forest') AS pct_forest,
        sum(l.percent_area) FILTER (WHERE c.class_group = 'Developed') AS pct_developed,
        sum(l.percent_area) FILTER (WHERE c.class_group = 'Agriculture') AS pct_agriculture,
        sum(l.percent_area) FILTER (WHERE c.class_group = 'Wetlands') AS pct_wetlands,
        sum(l.percent_area) FILTER (WHERE c.class_group = 'Water') AS pct_water,
        sum(l.percent_area) FILTER (
            WHERE c.class_group IN ('Shrubland', 'Herbaceous', 'Barren')) AS pct_other
    FROM land_cover l
    JOIN land_cover_codes c USING (class_code)
    GROUP BY ALL
),
-- Choice 4: last NLCD year before removal, else the nearest one.
baseline AS (
    SELECT p.AR_ID, p.station_id, g.year AS land_cover_year
    FROM (SELECT DISTINCT AR_ID, station_id, removal_year FROM rows) p
    JOIN land_groups g USING (station_id)
    QUALIFY row_number() OVER (
        PARTITION BY p.AR_ID, p.station_id
        ORDER BY g.year < p.removal_year DESC,
                 abs(g.year - (p.removal_year - 1)),
                 g.year
    ) = 1
)
SELECT
    r.measurement_id,
    r.AR_ID,
    r.station_id,
    r.parameter_std,
    r.sample_fraction_std,
    r.removal_year,
    r.side,
    r.relationship,
    r.network_distance_km,
    r.period,
    r.years_from_removal,
    r.sample_date,
    year(r.sample_date) AS year,
    month(r.sample_date) AS month,
    CASE
        WHEN month(r.sample_date) IN (12, 1, 2) THEN 'winter'
        WHEN month(r.sample_date) IN (3, 4, 5) THEN 'spring'
        WHEN month(r.sample_date) IN (6, 7, 8) THEN 'summer'
        ELSE 'fall'
    END AS season,
    r.result_value_std,
    r.result_unit_std,
    r.is_censored,
    r.unit_basis_assumed,
    m.provider,
    r.dam_name,
    d.River AS river,
    d.State AS state,
    d.Dam_Height_ft AS dam_height_ft,
    d.Year_built AS year_built,
    r.removal_year - d.Year_built AS dam_age_at_removal,
    d.Original_Use AS original_use,
    sw.area_km2 AS upstream_area_km2,
    sw.truncated AS watershed_truncated,
    c.temperature_mean_c AS air_temp_mean_c,
    c.precipitation_mm AS precip_mm,
    prev.precipitation_mm AS precip_prev_month_mm,
    b.land_cover_year,
    g.pct_forest,
    g.pct_developed,
    g.pct_agriculture,
    g.pct_wetlands,
    g.pct_water,
    g.pct_other
FROM rows r
JOIN wq_measurement m USING (measurement_id)
JOIN dam_removal d ON d.AR_ID = r.AR_ID
LEFT JOIN station_watershed sw ON sw.station_id = r.station_id
LEFT JOIN climate c
       ON c.station_id = r.station_id
      AND c.year = year(r.sample_date)
      AND c.month = month(r.sample_date)
LEFT JOIN climate prev
       ON prev.station_id = r.station_id
      AND prev.year * 12 + prev.month = year(r.sample_date) * 12 + month(r.sample_date) - 1
LEFT JOIN baseline b ON b.AR_ID = r.AR_ID AND b.station_id = r.station_id
LEFT JOIN land_groups g ON g.station_id = r.station_id AND g.year = b.land_cover_year
ORDER BY r.AR_ID, r.station_id, r.parameter_std, r.sample_date, r.measurement_id
"""

# Expected eligible-pair measurement count, to check no join dropped or
# duplicated rows.
EXPECTED_ROWS_SQL = """
SELECT count(*)
FROM measurement_window w
JOIN eligibility_pairs p
  ON w.AR_ID = p.AR_ID
 AND w.station_id = p.station_id
 AND w.parameter_std = p.parameter_std
 AND w.sample_fraction_std IS NOT DISTINCT FROM p.sample_fraction_std
WHERE p.has_both AND w.period IN ('pre', 'post')
"""


def build_table(db_path):
    with duckdb.connect(str(db_path), read_only=True) as con:
        table = con.sql(ANALYSIS_SQL).df()
        expected = con.sql(EXPECTED_ROWS_SQL).fetchone()[0]
    table["sample_date"] = table["sample_date"].astype("datetime64[us]")
    return table.astype(SCHEMA), expected


def validate(table, expected_rows):
    errors = schema_errors("analysis_measurement", table, SCHEMA, PRIMARY_KEY)
    if errors:
        return errors
    if len(table) != expected_rows:
        errors.append(f"{len(table):,} rows, expected {expected_rows:,} "
                      "(pre/post measurements of has_both pairs)")
    if not table["period"].isin(["pre", "post"]).all():
        errors.append("period not in {pre, post}")
    if (table["year"] == table["removal_year"]).any():
        errors.append("removal-year samples present")
    groups = table[["pct_forest", "pct_developed", "pct_agriculture",
                    "pct_wetlands", "pct_water", "pct_other"]].dropna()
    if not ((groups.sum(axis=1) - 100).abs() < 0.01).all():
        errors.append("land-cover groups don't sum to 100")
    return errors


def coverage_report(table):
    """Share of rows with each context column filled in."""
    columns = ["side", "upstream_area_km2", "air_temp_mean_c", "precip_prev_month_mm",
               "land_cover_year", "dam_height_ft", "year_built"]
    filled = table[columns].notna().mean().mul(100).round(1)
    baseline = table.drop_duplicates(["AR_ID", "station_id"])
    after = (baseline["land_cover_year"] >= baseline["removal_year"]).sum()
    lines = [f"  {c:<22} {v:5.1f}% of rows" for c, v in filled.items()]
    lines.append(f"  dam-station pairs whose land cover is from after removal: {after}")
    return "\n".join(lines)


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
    db_path = require_file(output_dir / "warehouse.duckdb",
                           "warehouse.duckdb (run build_views.py first)")
    for table in REQUIRED_TABLES:
        require_file(output_dir / f"{table}.parquet", f"{table} table (run the step 5 stages first)")

    table, expected = build_table(db_path)
    errors = validate(table, expected)
    if errors:
        raise ValueError("Validation failed; no warehouse files were written:\n  - "
                         + "\n  - ".join(errors))

    path = output_dir / "analysis_measurement.parquet"
    write_parquet_atomic(table, path)
    add_view(db_path, "analysis_measurement", path)

    print(f"Case study: {cfg.name}")
    print(f"Wrote {len(table):,} rows to {path}")
    print(f"Dams: {table['AR_ID'].nunique()}; stations: {table['station_id'].nunique()}; "
          f"parameters: {table['parameter_std'].nunique()}")
    print("\nRows per parameter and period:")
    counts = (table.assign(fraction=table["sample_fraction_std"].fillna("-"))
              .groupby(["parameter_std", "fraction", "period"]).size()
              .unstack(fill_value=0))
    print(counts.to_string())
    print("\nContext coverage:")
    print(coverage_report(table))


if __name__ == "__main__":
    main()
