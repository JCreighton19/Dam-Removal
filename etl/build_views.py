"""
Build data/warehouse/<case_study>/warehouse.duckdb: views only, over the
warehouse Parquet files (docs/warehouse_spec.md, section 5). It stores no
data, so it can be deleted and rebuilt at any time. Replaces
report_pilot_eligibility.py.

Views
-----
- One per Parquet table (dam_removal, wq_station, dam_station_link,
  huc8_watershed, streams, wq_measurement, wq_measurement_raw), so queries
  say FROM wq_measurement instead of a file path.
- eligibility_settings: the window and thresholds the views below use.
- measurement_window: one row per measurement x linked dam with a removal
  year. period is 'pre' / 'post' / 'removal_year' / 'outside'; the window
  is the same as report_pilot_eligibility.py (removal year excluded from
  both sides). side is the station's position relative to the dam;
  same_reach stations are placed by snap_fraction, which increases
  downstream along a reach (the assumption link_dams_stations.py makes).
- eligibility_pairs: one row per dam x station x parameter x fraction with
  samples and distinct years on each side; has_both when both sides meet
  the thresholds.
- eligibility_dams: one row per dam x parameter x fraction: downstream
  stations with both sides, and upstream ones (possible controls).
- eligibility_summary: one row per parameter x fraction: how many dams
  are eligible downstream, and how many also have an upstream control.

Thresholds come from the config's optional `eligibility` block
(min_samples, min_years per side; defaults 3 and 2) and the window from
`wq_window`. Change them and re-run this script; nothing else is rebuilt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb

from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.warehouse import warehouse_dir

TABLES = [
    "dam_removal",
    "wq_station",
    "dam_station_link",
    "huc8_watershed",
    "streams",
    "wq_measurement",
    "wq_measurement_raw",
]

# Context tables built after this script (build_station_watershed.py, …).
# They add their own view when written; a rebuild keeps any that exist.
OPTIONAL_TABLES = ["station_watershed"]

DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_YEARS = 2

ELIGIBILITY_VIEWS = """
CREATE VIEW measurement_window AS
WITH settings AS (SELECT * FROM eligibility_settings),
links AS (
    SELECT
        l.AR_ID,
        d.Dam_Name AS dam_name,
        d.Year_Removed AS removal_year,
        l.station_id,
        l.relationship,
        CASE
            WHEN l.relationship <> 'same_reach' THEN l.relationship
            WHEN s.snap_fraction > d.snap_fraction THEN 'downstream'
            WHEN s.snap_fraction < d.snap_fraction THEN 'upstream'
        END AS side,
        l.network_distance_km
    FROM dam_station_link l
    JOIN dam_removal d USING (AR_ID)
    JOIN wq_station s USING (station_id)
    WHERE d.Year_Removed IS NOT NULL
)
SELECT
    links.*,
    m.measurement_id,
    m.sample_date,
    year(m.sample_date) AS sample_year,
    year(m.sample_date) - links.removal_year AS years_from_removal,
    CASE
        WHEN year(m.sample_date) = links.removal_year THEN 'removal_year'
        WHEN year(m.sample_date) - links.removal_year
             BETWEEN -settings.years_before AND -1 THEN 'pre'
        WHEN year(m.sample_date) - links.removal_year
             BETWEEN 1 AND settings.years_after THEN 'post'
        ELSE 'outside'
    END AS period,
    m.parameter_std,
    m.sample_fraction_std,
    m.result_value_std,
    m.result_unit_std,
    m.is_censored,
    m.unit_basis_assumed
FROM links
JOIN wq_measurement m USING (station_id)
CROSS JOIN settings;

CREATE VIEW eligibility_pairs AS
WITH counts AS (
    SELECT
        AR_ID, dam_name, removal_year, station_id, relationship, side,
        network_distance_km, parameter_std, sample_fraction_std,
        count(*) FILTER (WHERE period = 'pre') AS n_pre,
        count(*) FILTER (WHERE period = 'post') AS n_post,
        count(DISTINCT sample_year) FILTER (WHERE period = 'pre') AS years_pre,
        count(DISTINCT sample_year) FILTER (WHERE period = 'post') AS years_post
    FROM measurement_window
    WHERE period IN ('pre', 'post')
    GROUP BY ALL
)
SELECT
    counts.*,
    n_pre >= min_samples AND years_pre >= min_years AS has_pre,
    n_post >= min_samples AND years_post >= min_years AS has_post,
    has_pre AND has_post AS has_both
FROM counts
CROSS JOIN eligibility_settings;

CREATE VIEW eligibility_dams AS
SELECT
    AR_ID, dam_name, removal_year, parameter_std, sample_fraction_std,
    count(*) FILTER (WHERE side = 'downstream' AND has_both) AS downstream_stations_both,
    count(*) FILTER (WHERE side = 'upstream' AND has_both) AS upstream_stations_both,
    downstream_stations_both > 0 AS has_both,
    downstream_stations_both > 0 AND upstream_stations_both > 0 AS has_upstream_control
FROM eligibility_pairs
GROUP BY ALL;

CREATE VIEW eligibility_summary AS
SELECT
    parameter_std,
    sample_fraction_std,
    count(*) FILTER (WHERE has_both) AS dams_eligible,
    count(*) FILTER (WHERE has_upstream_control) AS dams_with_upstream_control,
    (SELECT count(*) FROM dam_removal WHERE Year_Removed IS NOT NULL) AS dams_with_removal_year
FROM eligibility_dams
GROUP BY ALL
ORDER BY dams_eligible DESC, parameter_std;
"""


def table_view_sql(table, path):
    # Absolute paths, so the views work from any working directory.
    return f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{Path(path).resolve()}')"


def add_view(db_path, table, path):
    """Add (or replace) one table view in an existing warehouse.duckdb."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(table_view_sql(table, path))


def build_database(db_path, parquet_dir, years_before, years_after, min_samples, min_years):
    """Write the views to db_path (via a .tmp file, so a failed build never
    replaces a working database)."""
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    with duckdb.connect(str(tmp_path)) as con:
        for table in TABLES:
            con.execute(table_view_sql(table, parquet_dir / f"{table}.parquet"))
        for table in OPTIONAL_TABLES:
            if (parquet_dir / f"{table}.parquet").exists():
                con.execute(table_view_sql(table, parquet_dir / f"{table}.parquet"))

        con.execute(f"""
            CREATE VIEW eligibility_settings AS SELECT
                {int(years_before)} AS years_before,
                {int(years_after)} AS years_after,
                {int(min_samples)} AS min_samples,
                {int(min_years)} AS min_years
        """)
        con.execute(ELIGIBILITY_VIEWS)

        # Fails here, before replacing anything, if a view is broken.
        summary = con.sql("SELECT * FROM eligibility_summary").df()

    os.replace(tmp_path, db_path)
    return summary


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
    parquet_dir = warehouse_dir(cfg)

    for table in TABLES:
        require_file(
            parquet_dir / f"{table}.parquet",
            f"{table} table (run load_core_tables.py and clean_wq_results.py first)",
        )

    wq_window = cfg.get("wq_window", {}) or {}
    years_before = int(wq_window.get("years_before", 3))
    years_after = int(wq_window.get("years_after", 3))
    eligibility = cfg.get("eligibility", {}) or {}
    min_samples = int(eligibility.get("min_samples", DEFAULT_MIN_SAMPLES))
    min_years = int(eligibility.get("min_years", DEFAULT_MIN_YEARS))

    print(f"Case study: {cfg.name}")
    print(f"Window: {years_before} years before / {years_after} years after removal")
    print(f"Each side needs >= {min_samples} samples in >= {min_years} distinct years")

    db_path = parquet_dir / "warehouse.duckdb"
    summary = build_database(
        db_path, parquet_dir, years_before, years_after, min_samples, min_years
    )

    print(f"\nWrote {db_path}")
    print("\neligibility_summary:")
    print(summary.to_string(index=False) if len(summary) else "  (no eligible pairs)")


if __name__ == "__main__":
    main()
