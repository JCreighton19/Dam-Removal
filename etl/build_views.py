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
- removal_link_all: dam_station_link with each dam's removal year and the
  station's side.
- removal_link: removal_link_all within the link distance limits (the link
  part of measurement_window; every view below uses it).
- link_features: per link in removal_link, the drainage-area ratio
  (station / dam NHDPlus area), for upstream stations whether they may
  sit in the former reservoir (within dam height / reach slope of the dam),
  and the flow permanence of the station's and the dam's reach (NHD FCode;
  qc_non_perennial when either is intermittent or ephemeral).
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
3000) and the exposure settings from `exposure:` -- min_distance_ratio
(default 1), and max_downstream_km / max_upstream_km, the farthest a station
can be from a dam along the network to be linked to it (default: no limit;
docs/warehouse_spec.md section 5, "Link distance limits"). dam_station_link
itself stays complete. Change them and re-run this script; the stages after
it (watershed, land cover, climate, analysis dataset) then need re-running,
since they cover the eligible stations only.
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
from etl.common.warehouse import (add_variant_argument, check_variant, stale_reference_errors,
                                  views_db_path, warehouse_dir)

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
                   "land_cover_codes", "climate", "dam_reservoir", "analysis_measurement"]

DEFAULT_MIN_SAMPLES = 3
DEFAULT_MIN_YEARS = 2
# Median specific conductance (uS/cm) above which a station is flagged
# saline; fresh water is usually well under 1,500 (preprocessing audit).
DEFAULT_SALINE_SC = 3000
# A measurement is assigned to its nearest upstream removal only if the
# next one is more than this many times as far. 1 (since 2026-11-02): the
# nearest wins whenever it is strictly nearer; only exact ties stay
# competing_distance (advisor: associate a station below several removals
# with the closest one upstream rather than excluding it). The
# preprocessing audit (section 9) used 2; set exposure.min_distance_ratio: 2
# in the config to go back.
DEFAULT_MIN_DISTANCE_RATIO = 1.0

# exposure_status values that assign a measurement to one removal
# (measurement_exposure). assigned_overlapping: the nearest removal, but
# another removal upstream of the station also falls inside its window, so
# its before/after contrast may include that removal's effects; drop it from
# a model's treated statuses for a sensitivity check.
ASSIGNED_STATUSES = ("assigned_sole", "assigned_nearest", "assigned_overlapping")
# Farthest a station can be from a dam (network km) to be linked to it. None
# links every connected station (the behavior before 2026-10-26).
DEFAULT_MAX_DOWNSTREAM_KM = None
DEFAULT_MAX_UPSTREAM_KM = None

# NHDFlowline FCode -> flow permanence (link_features). The matched reaches
# are FType 460 StreamRiver or 558 ArtificialPath. An artificial path runs
# through a lake, reservoir or wide river, so it counts as perennial; 46000
# (StreamRiver without a hydrographic category) and any other code are
# 'unknown'. NHD's intermittent / ephemeral coding is known to be
# unreliable in places (the arid West especially), so it is a flag, not a
# filter: models decide (advisor, 2026-11-02: perennial streams are the main
# analysis).
FLOW_PERMANENCE = {
    46006: "perennial",
    55800: "perennial",
    46003: "intermittent",
    46007: "ephemeral",
}
NON_PERENNIAL = ("intermittent", "ephemeral")

# backwater_km (dam height / reach slope) needs a real slope. NHDPlus gives
# flat or uncomputable reaches a minimum slope of 1e-5, which turns a 3 m dam
# into a 300 km "reservoir" and marks every upstream control as inside it
# (stage 1 audit, 2026-11-04: 212 of 1,777 dams, backwater up to 3,292 km).
# Slopes at or below this are treated as unknown: backwater_km,
# reservoir_volume_proxy and upstream_in_former_reservoir are null for those
# dams (link_features; analysis/audits/reservoir_size_audit.py uses the same
# rule).
MIN_BACKWATER_SLOPE = 1e-5


def flow_permanence_sql(column):
    """SQL CASE mapping an FCode column to FLOW_PERMANENCE (null stays null)."""
    whens = " ".join(f"WHEN {int(code)} THEN '{label}'" for code, label in FLOW_PERMANENCE.items())
    return f"CASE WHEN {column} IS NULL THEN NULL ELSE CASE {column} {whens} ELSE 'unknown' END END"


def _sql_number(value):
    return "NULL::DOUBLE" if value is None else repr(float(value))


def settings_sql(years_before, years_after, min_samples, min_years,
                 saline_sc=DEFAULT_SALINE_SC, min_distance_ratio=DEFAULT_MIN_DISTANCE_RATIO,
                 max_downstream_km=DEFAULT_MAX_DOWNSTREAM_KM,
                 max_upstream_km=DEFAULT_MAX_UPSTREAM_KM, min_months_single_year=1):
    """The two one-row settings views the views below read. A null distance
    limit means no limit. min_months_single_year: a side with samples in a
    single year also needs them in this many distinct months (1, the
    default, adds no condition; eligibility sensitivity variants only)."""
    return f"""
        CREATE VIEW eligibility_settings AS SELECT
            {int(years_before)} AS years_before,
            {int(years_after)} AS years_after,
            {int(min_samples)} AS min_samples,
            {int(min_years)} AS min_years,
            {int(min_months_single_year)} AS min_months_single_year,
            {_sql_number(max_downstream_km)} AS max_downstream_km,
            {_sql_number(max_upstream_km)} AS max_upstream_km;
        CREATE VIEW qc_settings AS SELECT
            {float(saline_sc)} AS saline_sc_us_cm,
            {float(min_distance_ratio)} AS exposure_min_distance_ratio;
    """


ELIGIBILITY_VIEWS = """
CREATE VIEW removal_link_all AS
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

-- Links within the distance limits. A link with no distance or side is kept
-- only when no limit applies to it.
CREATE VIEW removal_link AS
SELECT l.*
FROM removal_link_all l
CROSS JOIN eligibility_settings s
WHERE CASE l.side
          WHEN 'downstream' THEN s.max_downstream_km IS NULL
                                 OR l.network_distance_km <= s.max_downstream_km
          WHEN 'upstream' THEN s.max_upstream_km IS NULL
                               OR l.network_distance_km <= s.max_upstream_km
          ELSE s.max_downstream_km IS NULL AND s.max_upstream_km IS NULL
      END;

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
        count(DISTINCT sample_year) FILTER (WHERE period = 'post') AS years_post,
        count(DISTINCT month(sample_date)) FILTER (WHERE period = 'pre') AS months_pre,
        count(DISTINCT month(sample_date)) FILTER (WHERE period = 'post') AS months_post
    FROM measurement_window
    WHERE period IN ('pre', 'post')
    GROUP BY ALL
)
SELECT
    counts.*,
    -- min_months_single_year only binds a side with samples in one year
    -- (min_years 1); at the default (1) or with min_years >= 2 it never does.
    n_pre >= min_samples AND years_pre >= min_years
        AND (years_pre > 1 OR months_pre >= min_months_single_year) AS has_pre,
    n_post >= min_samples AND years_post >= min_years
        AND (years_post > 1 OR months_post >= min_months_single_year) AS has_post,
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
# columns of wq_station (load_core_tables after the 2026-10-23 snapping) and
# reach_feature_code of wq_station / dam_removal (load_core_tables after
# 2026-11-02).
EXPOSURE_QC_VIEWS = """
-- Drainage-area ratio: how much of the station's water passed the dam
-- (about 1: mostly the dam's; 50: the dam drains ~2%). NHDPlus areas at
-- each reach's downstream end; non-positive values are NHDPlus placeholders.
-- backwater_km: rough reservoir length, dam height / reach slope; an
-- upstream station within it may have been in the former reservoir, i.e.
-- affected by the removal rather than a control (preprocessing audit,
-- section 6). Null when the slope is at the NHDPlus minimum
-- (MIN_BACKWATER_SLOPE): the length is unknown, not huge. Null also
-- without a positive dam height.
-- reservoir_volume_proxy: height_m^2 / slope * sqrt(dam drainage area km2),
-- a constructed size index (a wedge of height H and length H/S has a
-- cross-section of about H^2/S; channel width grows roughly with the square
-- root of drainage area). Not a measured volume or stored sediment; only its
-- ranks / logs mean anything. Same height and slope rules as backwater_km.
CREATE VIEW link_features AS
WITH areas AS (
    SELECT
        l.AR_ID,
        l.station_id,
        l.side,
        l.network_distance_km,
        CASE WHEN d.reach_totdasqkm > 0 THEN d.reach_totdasqkm END AS dam_drainage_area_km2,
        CASE WHEN s.reach_totdasqkm > 0 THEN s.reach_totdasqkm END AS station_drainage_area_km2,
        CASE WHEN d.Dam_Height_ft > 0 THEN d.Dam_Height_ft * 0.3048 END
            / CASE WHEN d.reach_slope > __MIN_BACKWATER_SLOPE__ THEN d.reach_slope END / 1000
            AS backwater_km,
        CASE WHEN d.Dam_Height_ft > 0 AND d.reach_slope > __MIN_BACKWATER_SLOPE__
                  AND d.reach_totdasqkm > 0
             THEN pow(d.Dam_Height_ft * 0.3048, 2) / d.reach_slope * sqrt(d.reach_totdasqkm)
        END AS reservoir_volume_proxy,
        d.reach_feature_code AS dam_reach_fcode,
        s.reach_feature_code AS station_reach_fcode
    FROM removal_link l
    JOIN dam_removal d USING (AR_ID)
    JOIN wq_station s USING (station_id)
)
SELECT
    *,
    station_drainage_area_km2 / dam_drainage_area_km2 AS drainage_area_ratio,
    CASE WHEN side = 'upstream' THEN network_distance_km <= backwater_km END
        AS upstream_in_former_reservoir,
    __STATION_PERMANENCE__ AS station_flow_permanence,
    __DAM_PERMANENCE__ AS dam_flow_permanence,
    -- True when either reach is intermittent or ephemeral, false when both
    -- are perennial, null otherwise (a code missing or 'unknown').
    CASE
        WHEN station_flow_permanence IN __NON_PERENNIAL__
          OR dam_flow_permanence IN __NON_PERENNIAL__ THEN true
        WHEN station_flow_permanence = 'perennial' AND dam_flow_permanence = 'perennial'
            THEN false
    END AS qc_non_perennial
FROM areas;

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
-- network distance is assigned unless the next is not clearly farther
-- (competing_distance) or a distance is unknown. When another upstream
-- removal falls inside the nearest one's window (timing_conflict), the
-- measurement is still assigned, as assigned_overlapping.
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
        WHEN n.n_candidates = 1 THEN 'assigned_sole'
        WHEN n.nearest_distance_km IS NULL OR n.next_distance_km IS NULL
            THEN 'unresolved_distance_unknown'
        WHEN n.next_distance_km <= q.exposure_min_distance_ratio * n.nearest_distance_km
            THEN 'competing_distance'
        ELSE 'assigned_nearest'
    END AS distance_status,
    CASE WHEN timing_conflict AND distance_status IN ('assigned_sole', 'assigned_nearest')
         THEN 'assigned_overlapping' ELSE distance_status END AS exposure_status,
    CASE WHEN exposure_status IN __ASSIGNED__
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
EXPOSURE_QC_VIEWS = (
    EXPOSURE_QC_VIEWS
    .replace("__STATION_PERMANENCE__", flow_permanence_sql("station_reach_fcode"))
    .replace("__DAM_PERMANENCE__", flow_permanence_sql("dam_reach_fcode"))
    .replace("__NON_PERENNIAL__", "(" + ", ".join(f"'{v}'" for v in NON_PERENNIAL) + ")")
    .replace("__ASSIGNED__", "(" + ", ".join(f"'{v}'" for v in ASSIGNED_STATUSES) + ")")
    .replace("__MIN_BACKWATER_SLOPE__", repr(MIN_BACKWATER_SLOPE))
)


def table_view_sql(table, path):
    # Absolute paths, so the views work from any working directory.
    return f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{Path(path).resolve()}')"


def add_view(db_path, table, path):
    """Add (or replace) one table view in an existing warehouse.duckdb."""
    with duckdb.connect(str(db_path)) as con:
        con.execute(table_view_sql(table, path))


def build_database(db_path, parquet_dir, years_before, years_after, min_samples, min_years,
                   saline_sc=DEFAULT_SALINE_SC, min_distance_ratio=DEFAULT_MIN_DISTANCE_RATIO,
                   max_downstream_km=DEFAULT_MAX_DOWNSTREAM_KM,
                   max_upstream_km=DEFAULT_MAX_UPSTREAM_KM, min_months_single_year=1):
    """Write the views to db_path (via a .tmp file, so a failed build never
    replaces a working database). Returns eligibility_summary and messages
    for derived tables that are stale (from an earlier snapping or linking
    run); those don't stop the build, since the stages that rebuild them run
    after this one."""
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    tmp_path.unlink(missing_ok=True)

    with duckdb.connect(str(tmp_path)) as con:
        for table in TABLES:
            con.execute(table_view_sql(table, parquet_dir / f"{table}.parquet"))
        for table in OPTIONAL_TABLES:
            if (parquet_dir / f"{table}.parquet").exists():
                con.execute(table_view_sql(table, parquet_dir / f"{table}.parquet"))

        con.execute(settings_sql(years_before, years_after, min_samples, min_years,
                                 saline_sc, min_distance_ratio, max_downstream_km, max_upstream_km,
                                 min_months_single_year))
        con.execute(ELIGIBILITY_VIEWS)
        con.execute(EXPOSURE_QC_VIEWS)

        # Fails here, before replacing anything, if a view is broken.
        summary = con.sql("SELECT * FROM eligibility_summary").df()
        for view in ["measurement_exposure", "station_qc", "link_features"]:
            con.sql(f"SELECT * FROM {view} LIMIT 0").fetchall()
        stale = stale_reference_errors(con)

    os.replace(tmp_path, db_path)
    return summary, stale


def _limit_text(km):
    return "none" if km is None else f"{float(km):g} km"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    add_variant_argument(parser, "writes warehouse_<variant>.duckdb with that rule; "
                                 "warehouse.duckdb is not touched")
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
    min_months_single_year = 1
    if args.eligibility_variant:
        rule = check_variant(args.eligibility_variant)
        min_samples, min_years = rule["min_samples"], rule["min_years"]
        min_months_single_year = rule["min_months_single_year"]
    saline_sc = float((cfg.get("qc", {}) or {}).get("saline_sc_us_cm", DEFAULT_SALINE_SC))
    exposure = cfg.get("exposure", {}) or {}
    min_distance_ratio = float(exposure.get("min_distance_ratio", DEFAULT_MIN_DISTANCE_RATIO))
    max_downstream_km = exposure.get("max_downstream_km", DEFAULT_MAX_DOWNSTREAM_KM)
    max_upstream_km = exposure.get("max_upstream_km", DEFAULT_MAX_UPSTREAM_KM)

    print(f"Case study: {cfg.name}")
    if args.eligibility_variant:
        print(f"SENSITIVITY VARIANT {args.eligibility_variant}: "
              f"{check_variant(args.eligibility_variant)['label']}; the primary "
              "warehouse.duckdb is not touched")
    print(f"Window: {years_before} years before / {years_after} years after removal")
    print(f"Each side needs >= {min_samples} samples in >= {min_years} distinct years"
          + (f" (a single-year side: in >= {min_months_single_year} months)"
             if min_months_single_year > 1 else ""))
    print(f"Saline: median specific conductance > {saline_sc:g} uS/cm; exposure: next "
          f"removal must be > {min_distance_ratio:g}x as far as the nearest")
    print(f"Link distance limits: downstream {_limit_text(max_downstream_km)}, "
          f"upstream {_limit_text(max_upstream_km)}")

    db_path = views_db_path(parquet_dir, args.eligibility_variant)
    summary, stale = build_database(
        db_path, parquet_dir, years_before, years_after, min_samples, min_years,
        saline_sc, min_distance_ratio, max_downstream_km, max_upstream_km,
        min_months_single_year,
    )

    print(f"\nWrote {db_path}")
    if stale:
        print("\nStale tables (from an earlier run; the stages after this one rebuild them, "
              "re-run those before using these tables):\n  - " + "\n  - ".join(stale))
    print("\neligibility_summary:")
    print(summary.to_string(index=False) if len(summary) else "  (no eligible pairs)")


if __name__ == "__main__":
    main()
