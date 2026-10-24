"""
Build data/warehouse/<case_study>/warehouse.duckdb: views only, over the
warehouse Parquet files (docs/warehouse_spec.md, section 5). It stores no
data, so it can be deleted and rebuilt at any time. Replaces
report_pilot_eligibility.py.

Views
-----
- One per Parquet table (dam_removal, wq_station, dam_station_link,
  huc8_watershed, streams, wq_measurement, wq_measurement_raw, and the
  OPTIONAL_TABLES that exist), so queries say FROM wq_measurement instead
  of a file path.
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
- removal_link: dam_station_link with each dam's removal year and the
  station's side (the link part of measurement_window).
- pair_competing_removals: per dam-station pair, the other removals linked
  to the station whose removal year falls inside the pair's window.
- measurement_exposure: one row per measurement downstream of at least one
  removal whose window holds it: the nearest such removal, and whether the
  measurement can be assigned to it without ambiguity (spec 5,
  "Exposure"). An assignment says which removal is nearest upstream in
  space and time; it does not establish that the removal caused a change.
- station_qc: per station, the snapping flags and the saline flag (median
  specific conductance above the threshold).

Thresholds come from the config's optional `eligibility` block
(min_samples, min_years per side; defaults 3 and 2), the window from
`wq_window`, the saline threshold from `qc: {saline_sc_us_cm}` (default
3000) and the exposure distance ratio from `exposure: {min_distance_ratio}`
(default 2). Change them and re-run this script; nothing else is rebuilt.
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

# Tables built by other stages (build_daily_measurements.py,
# build_station_watershed.py, …). They add their own view when written; a
# rebuild keeps any that exist.
OPTIONAL_TABLES = ["wq_measurement_daily", "station_watershed", "station_catchment", "land_cover",
                   "land_cover_codes", "climate", "analysis_measurement"]

DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_YEARS = 2
# Median specific conductance (uS/cm) above which a station is flagged
# saline; fresh water is usually well under 1,500 (preprocessing audit).
DEFAULT_SALINE_SC = 3000
# A measurement is assigned to its nearest upstream removal only if the
# next one is more than this many times as far (preprocessing audit,
# section 9).
DEFAULT_MIN_DISTANCE_RATIO = 2.0


def settings_sql(years_before, years_after, min_samples, min_years,
                 saline_sc=DEFAULT_SALINE_SC, min_distance_ratio=DEFAULT_MIN_DISTANCE_RATIO):
    """The two one-row settings views the views below read."""
    return f"""
        CREATE VIEW eligibility_settings AS SELECT
            {int(years_before)} AS years_before,
            {int(years_after)} AS years_after,
            {int(min_samples)} AS min_samples,
            {int(min_years)} AS min_years;
        CREATE VIEW qc_settings AS SELECT
            {float(saline_sc)} AS saline_sc_us_cm,
            {float(min_distance_ratio)} AS exposure_min_distance_ratio;
    """


ELIGIBILITY_VIEWS = """
CREATE VIEW removal_link AS
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
WHERE d.Year_Removed IS NOT NULL;

CREATE VIEW measurement_window AS
WITH settings AS (SELECT * FROM eligibility_settings),
links AS (SELECT * FROM removal_link)
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

# Built after ELIGIBILITY_VIEWS; they need qc_settings and the snapping
# columns of wq_station (load_core_tables after the 2026-10-23 snapping).
EXPOSURE_QC_VIEWS = """
-- Other removals linked to the same station whose removal year is inside
-- this pair's window. "Upstream" ones are upstream of the station
-- (side = 'downstream': the station is downstream of them).
CREATE VIEW pair_competing_removals AS
SELECT
    p.AR_ID,
    p.station_id,
    count(o.AR_ID) AS n_other_removals_in_window,
    count(o.AR_ID) FILTER (WHERE o.side = 'downstream') AS n_other_upstream_removals_in_window,
    string_agg(o.AR_ID || ' (' || o.removal_year || ', station ' || coalesce(o.side, 'side unknown')
               || ')', '; ' ORDER BY o.removal_year, o.AR_ID) AS other_removals_in_window
FROM removal_link p
CROSS JOIN eligibility_settings s
LEFT JOIN removal_link o
       ON o.station_id = p.station_id
      AND o.AR_ID <> p.AR_ID
      AND o.removal_year BETWEEN p.removal_year - s.years_before AND p.removal_year + s.years_after
GROUP BY p.AR_ID, p.station_id;

-- Candidates: removals upstream of the measurement's station whose window
-- (removal year +/- wq_window) holds the sample year. The nearest by
-- network distance is assigned unless another upstream removal falls in
-- its window (competing_timing), the next is not clearly farther
-- (competing_distance), or a distance is unknown.
CREATE VIEW measurement_exposure AS
WITH candidates AS (
    SELECT
        measurement_id, station_id, AR_ID, removal_year, network_distance_km,
        row_number() OVER w AS rank,
        count(*) OVER (PARTITION BY measurement_id) AS n_candidates
    FROM measurement_window
    WHERE side = 'downstream' AND period <> 'outside'
    WINDOW w AS (PARTITION BY measurement_id ORDER BY network_distance_km NULLS LAST, AR_ID)
),
nearest AS (
    SELECT
        c1.measurement_id, c1.station_id, c1.n_candidates,
        c1.AR_ID AS nearest_AR_ID,
        c1.removal_year AS nearest_removal_year,
        c1.network_distance_km AS nearest_distance_km,
        c2.AR_ID AS next_AR_ID,
        c2.network_distance_km AS next_distance_km
    FROM candidates c1
    LEFT JOIN candidates c2 ON c2.measurement_id = c1.measurement_id AND c2.rank = 2
    WHERE c1.rank = 1
)
SELECT
    n.*,
    p.n_other_upstream_removals_in_window > 0 AS timing_conflict,
    CASE
        WHEN p.n_other_upstream_removals_in_window > 0 THEN 'competing_timing'
        WHEN n.n_candidates = 1 THEN 'assigned_sole'
        WHEN n.nearest_distance_km IS NULL OR n.next_distance_km IS NULL
            THEN 'unresolved_distance_unknown'
        WHEN n.next_distance_km <= q.exposure_min_distance_ratio * n.nearest_distance_km
            THEN 'competing_distance'
        ELSE 'assigned_nearest'
    END AS exposure_status,
    CASE WHEN exposure_status IN ('assigned_sole', 'assigned_nearest')
         THEN n.nearest_AR_ID END AS assigned_AR_ID
FROM nearest n
JOIN pair_competing_removals p ON p.AR_ID = n.nearest_AR_ID AND p.station_id = n.station_id
CROSS JOIN qc_settings q;

-- Station flags (none excludes anything): snapping (snap_features.py) and
-- saline water (median uncensored specific conductance; null if none).
CREATE VIEW station_qc AS
WITH sc AS (
    SELECT station_id, median(result_value_std) AS sc_median_us_cm, count(*) AS sc_samples
    FROM wq_measurement
    WHERE parameter_std = 'specific_conductance' AND NOT is_censored
    GROUP BY station_id
)
SELECT
    s.station_id,
    s.qc_far_snap,
    s.qc_near_unsuitable_flowline,
    s.qc_non_riverine_name,
    sc.sc_median_us_cm,
    sc.sc_samples,
    sc.sc_median_us_cm > q.saline_sc_us_cm AS qc_saline
FROM wq_station s
LEFT JOIN sc USING (station_id)
CROSS JOIN qc_settings q;
"""


def table_view_sql(table, path):
    # Absolute paths, so the views work from any working directory.
    return f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{Path(path).resolve()}')"


def add_view(db_path, table, path):
    """Add (or replace) one table view in an existing warehouse.duckdb."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(table_view_sql(table, path))


def build_database(db_path, parquet_dir, years_before, years_after, min_samples, min_years,
                   saline_sc=DEFAULT_SALINE_SC, min_distance_ratio=DEFAULT_MIN_DISTANCE_RATIO):
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

        con.execute(settings_sql(years_before, years_after, min_samples, min_years,
                                 saline_sc, min_distance_ratio))
        con.execute(ELIGIBILITY_VIEWS)
        con.execute(EXPOSURE_QC_VIEWS)

        # Fails here, before replacing anything, if a view is broken.
        summary = con.sql("SELECT * FROM eligibility_summary").df()
        for view in ["measurement_exposure", "station_qc"]:
            con.sql(f"SELECT * FROM {view} LIMIT 0").fetchall()

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
    saline_sc = float((cfg.get("qc", {}) or {}).get("saline_sc_us_cm", DEFAULT_SALINE_SC))
    min_distance_ratio = float((cfg.get("exposure", {}) or {}).get(
        "min_distance_ratio", DEFAULT_MIN_DISTANCE_RATIO))

    print(f"Case study: {cfg.name}")
    print(f"Window: {years_before} years before / {years_after} years after removal")
    print(f"Each side needs >= {min_samples} samples in >= {min_years} distinct years")
    print(f"Saline: median specific conductance > {saline_sc:g} uS/cm; exposure: next "
          f"removal must be > {min_distance_ratio:g}x as far as the nearest")

    db_path = parquet_dir / "warehouse.duckdb"
    summary = build_database(
        db_path, parquet_dir, years_before, years_after, min_samples, min_years,
        saline_sc, min_distance_ratio,
    )

    print(f"\nWrote {db_path}")
    print("\neligibility_summary:")
    print(summary.to_string(index=False) if len(summary) else "  (no eligible pairs)")


if __name__ == "__main__":
    main()
