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
- wq_measurement_daily (build_daily_measurements.py), for daily_id
- station_watershed (build_station_watershed.py)
- land_cover, land_cover_codes (load_land_cover.py)
- climate (load_climate.py)
- dam_reservoir (build_dam_reservoir.py)

Design choices (agreed 2026-10-19)
----------------------------------
1. Rows: only pre / post measurements of dam x station x parameter x
   fraction pairs with has_both. The removal year and samples outside the
   window are left out.
2. A station linked to several dams gets one row per dam (as in
   measurement_window), so the key is (measurement_id, AR_ID).
3. Values are kept as cleaned: censored results are already half the
   detection limit (spec 3.6.3) and flagged by is_censored. Duplicate
   same-day samples are all kept; daily_id (added 2026-10-24) names the
   wq_measurement_daily row each one belongs to, so a model can group
   repeats exactly as the daily table does.
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
   what to exclude. missing_upstream_area / missing_climate /
   missing_land_cover say which are absent.

Added 2026-10-23 (spec 3.11, "Exposure and quality flags")
-----------------------------------------------------------
7. Exposure: every row keeps its dam (rule A), and exposure_status says
   whether the measurement is assigned to this dam as its nearest upstream
   removal in space and time (measurement_exposure view). Filter
   exposure_status IN build_views.ASSIGNED_STATUSES ('assigned_sole',
   'assigned_nearest', 'assigned_overlapping') for one exposure per
   measurement. Ties and unknown distances are flagged, not forced;
   other_removals_in_window lists the other removals for audit. (Before
   2026-11-02 a removal overlapping in time made the measurement
   competing_timing, unassigned; it is now assigned_overlapping.)
   An assignment is not evidence that the removal caused a change.
8. Station flags (station_qc view) and the drainage-area flag are carried,
   not applied. upstream_area_km2 is null unless drainage_area_flag = 'ok',
   so a placeholder, truncated or unverifiable area never enters as a value.

Added 2026-10-26
----------------
9. Links are limited by the config's exposure: max_downstream_km /
   max_upstream_km (build_views.py removal_link), so only pairs within
   those distances are eligible.
10. drainage_area_ratio (station / dam NHDPlus area; a dilution proxy, for
   modeling) and upstream_in_former_reservoir (an upstream station within
   the rough reservoir length; affected by the removal, not a control) come
   from the link_features view. Carried, not applied.
11. result_unit_raw, method_id, detection_limit and detection_limit_unit
   (as reported) are carried so analyses can check turbidity units, method
   changes and detection limits across the removal.
12. Refuses to run when a context table references stations wq_station
   doesn't have (left from an earlier snapping or linking run).

Added 2026-11-02
----------------
13. station_flow_permanence / dam_flow_permanence (perennial, intermittent,
   ephemeral, unknown; NHD FCode of each reach) and qc_non_perennial (either
   is intermittent or ephemeral; null when a code is unknown) come from the
   link_features view. Carried, not applied; the mixed model excludes
   qc_non_perennial rows by default (perennial streams as the main analysis).
14. discharge_same_day_cfs: median discharge (wq_measurement parameter
   discharge, measured with a sample) at the same station on the sample's
   day, and has_same_day_discharge. Suspended sediment means little without
   the flow it was carried by (advisor, 2026-11-02), so the mixed model
   keeps SSC / TSS rows only where it is known; it is also a flow covariate
   for any parameter. Discrete WQP flows only, not continuous gauge records.

Added 2026-11-04 (reservoir size; Stage 2 of the reservoir / distance work)
---------------------------------------------------------------------------
15. Dam size, three kinds kept apart and never imputed (null = unknown):
   - constructed, from link_features: backwater_km (dam height / reach
     slope, a rough reservoir length) and reservoir_volume_proxy
     (height_m^2 / slope * sqrt(dam drainage area), a size index, not a
     measured volume or stored sediment). Both null without a positive
     height, or when the reach slope is at the NHDPlus minimum
     (build_views.MIN_BACKWATER_SLOPE): the old backwater for those dams
     was up to 3,292 km.
   - measured, from dam_reservoir: reservoir_status, and
     reservoir_polygon_area_km2 / reservoir_polygon_timing for dams at a
     mapped reservoir outlet. Use timing edited_by_removal_year for an
     area mapped before the removal.
   - dam_drainage_area_km2 (the dam reach's NHDPlus TotDASqKm), the size
     model's river-size covariate: larger rivers have larger reservoirs
     (advisor), and the proxy contains it.
   main() prints how many dams have each, and downstream exposure status
   by distance band (overlapping removals grow with the distance window).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd

from etl.build_daily_measurements import daily_ids
from etl.build_views import ASSIGNED_STATUSES, add_view
from etl.common.config import default_case_study, load_config
from etl.common.io import require_file
from etl.common.warehouse import (DERIVED_TABLES, schema_errors, stale_reference_errors,
                                  warehouse_dir, write_parquet_atomic)

REQUIRED_TABLES = ["station_watershed", "land_cover", "land_cover_codes", "climate"]
# Choice 15; built by build_dam_reservoir.py (after build_views.py).
RESERVOIR_TABLE = "dam_reservoir"

# Downstream distance bands for the exposure report (network km, upper edge
# inclusive, as in build_views.removal_link).
DISTANCE_BANDS = [0, 5, 10, 15, 20, 30]

EXPOSURE_STATUSES = [
    "assigned_sole", "assigned_nearest", "assigned_overlapping", "assigned_to_other_removal",
    "competing_distance", "unresolved_distance_unknown", "upstream_control",
    "upstream_control_exposed", "unresolved_side",
]

SCHEMA = {
    # Keys
    "measurement_id": "int64",
    "daily_id": "int64",
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
    "drainage_area_ratio": "float64",
    "upstream_in_former_reservoir": "boolean",
    "station_flow_permanence": "string",
    "dam_flow_permanence": "string",
    "qc_non_perennial": "boolean",
    # Exposure
    "exposure_status": "string",
    "assigned_AR_ID": "string",
    "n_other_removals_in_window": "int64",
    "n_other_upstream_removals_in_window": "int64",
    "other_removals_in_window": "string",
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
    "result_unit_raw": "string",
    "method_id": "string",
    "detection_limit": "float64",
    "detection_limit_unit": "string",
    # Flow on the sample's day (choice 14)
    "discharge_same_day_cfs": "float64",
    "has_same_day_discharge": "bool",
    # Dam
    "dam_name": "string",
    "river": "string",
    "state": "string",
    "dam_height_ft": "float64",
    "year_built": "Int16",
    "dam_age_at_removal": "Int16",
    "original_use": "string",
    # Dam size (choice 15): constructed, measured, river size
    "backwater_km": "float64",
    "reservoir_volume_proxy": "float64",
    "reservoir_status": "string",
    "reservoir_polygon_area_km2": "float64",
    "reservoir_polygon_timing": "string",
    "dam_drainage_area_km2": "float64",
    # Station quality flags (none applied)
    "qc_far_snap": "boolean",
    "qc_near_unsuitable_flowline": "boolean",
    "qc_non_riverine_name": "boolean",
    "qc_saline": "boolean",
    # Watershed
    "upstream_area_km2": "float64",
    "watershed_truncated": "boolean",
    "drainage_area_flag": "string",
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
    # Missing context
    "missing_upstream_area": "bool",
    "missing_climate": "bool",
    "missing_land_cover": "bool",
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
-- Choice 14: discharge measured at the station on the sample's day.
discharge_day AS (
    SELECT station_id, sample_date, median(result_value_std) AS discharge_cfs
    FROM wq_measurement
    WHERE parameter_std = 'discharge' AND NOT is_censored
    GROUP BY station_id, sample_date
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
    lf.drainage_area_ratio,
    lf.upstream_in_former_reservoir,
    lf.station_flow_permanence,
    lf.dam_flow_permanence,
    lf.qc_non_perennial,
    CASE
        WHEN r.side = 'downstream' AND e.assigned_AR_ID = r.AR_ID THEN e.exposure_status
        WHEN r.side = 'downstream' AND e.assigned_AR_ID IS NOT NULL THEN 'assigned_to_other_removal'
        WHEN r.side = 'downstream' THEN e.exposure_status
        WHEN r.side = 'upstream' AND e.measurement_id IS NULL THEN 'upstream_control'
        WHEN r.side = 'upstream' THEN 'upstream_control_exposed'
        ELSE 'unresolved_side'
    END AS exposure_status,
    e.assigned_AR_ID,
    pc.n_other_removals_in_window,
    pc.n_other_upstream_removals_in_window,
    pc.other_removals_in_window,
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
    m.result_unit_raw,
    m.method_id,
    m.detection_limit,
    m.detection_limit_unit,
    q_day.discharge_cfs AS discharge_same_day_cfs,
    q_day.discharge_cfs IS NOT NULL AS has_same_day_discharge,
    r.dam_name,
    d.River AS river,
    d.State AS state,
    d.Dam_Height_ft AS dam_height_ft,
    d.Year_built AS year_built,
    r.removal_year - d.Year_built AS dam_age_at_removal,
    d.Original_Use AS original_use,
    lf.backwater_km,
    lf.reservoir_volume_proxy,
    dr.reservoir_status,
    dr.polygon_area_km2 AS reservoir_polygon_area_km2,
    dr.polygon_timing AS reservoir_polygon_timing,
    lf.dam_drainage_area_km2,
    q.qc_far_snap,
    q.qc_near_unsuitable_flowline,
    q.qc_non_riverine_name,
    q.qc_saline,
    CASE WHEN sw.drainage_area_flag = 'ok' THEN sw.area_km2 END AS upstream_area_km2,
    sw.truncated AS watershed_truncated,
    sw.drainage_area_flag,
    c.temperature_mean_c AS air_temp_mean_c,
    c.precipitation_mm AS precip_mm,
    prev.precipitation_mm AS precip_prev_month_mm,
    b.land_cover_year,
    g.pct_forest,
    g.pct_developed,
    g.pct_agriculture,
    g.pct_wetlands,
    g.pct_water,
    g.pct_other,
    upstream_area_km2 IS NULL AS missing_upstream_area,
    air_temp_mean_c IS NULL OR precip_mm IS NULL AS missing_climate,
    g.pct_forest IS NULL AS missing_land_cover
FROM rows r
JOIN wq_measurement m USING (measurement_id)
JOIN dam_removal d ON d.AR_ID = r.AR_ID
LEFT JOIN link_features lf ON lf.AR_ID = r.AR_ID AND lf.station_id = r.station_id
LEFT JOIN dam_reservoir dr ON dr.AR_ID = r.AR_ID
LEFT JOIN measurement_exposure e ON e.measurement_id = r.measurement_id
LEFT JOIN pair_competing_removals pc ON pc.AR_ID = r.AR_ID AND pc.station_id = r.station_id
LEFT JOIN station_qc q ON q.station_id = r.station_id
LEFT JOIN discharge_day q_day ON q_day.station_id = r.station_id AND q_day.sample_date = r.sample_date
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


def sql_path(path):
    return str(path.resolve()).replace("'", "''")


def stale_inputs(db_path):
    """Messages for context tables left from an earlier snapping or linking
    run (they reference stations wq_station no longer has), and for a
    dam_reservoir table that doesn't cover dam_removal's dams."""
    inputs = {t: stage for t, stage in DERIVED_TABLES.items() if t != "analysis_measurement"}
    with duckdb.connect(str(db_path), read_only=True) as con:
        errors = stale_reference_errors(con, inputs)
        n = con.sql(f"""
            SELECT count(*) FROM dam_removal
            WHERE AR_ID NOT IN (SELECT AR_ID FROM {RESERVOIR_TABLE})
        """).fetchone()[0]
    if n:
        errors.append(f"{RESERVOIR_TABLE}: {n:,} dam_removal AR_ID(s) missing; stale, "
                      "re-run build_dam_reservoir.py")
    return errors


def build_table(db_path):
    """The analysis rows, the expected row count and the daily_ids in
    wq_measurement_daily (to check daily_id against)."""
    output_dir = db_path.parent
    with duckdb.connect(str(db_path), read_only=True) as con:
        table = con.sql(ANALYSIS_SQL).df()
        expected = con.sql(EXPECTED_ROWS_SQL).fetchone()[0]
    with duckdb.connect() as con:
        ids = daily_ids(con, sql_path(output_dir / "wq_measurement.parquet"))
        daily = con.sql("SELECT daily_id FROM read_parquet('{}')".format(
            sql_path(output_dir / "wq_measurement_daily.parquet"))).df()["daily_id"]
    table = table.merge(ids, on="measurement_id", how="left", validate="many_to_one")
    table["sample_date"] = table["sample_date"].astype("datetime64[us]")
    return table[list(SCHEMA)].astype(SCHEMA), expected, set(daily)


def validate(table, expected_rows, daily_ids_written):
    errors = schema_errors("analysis_measurement", table, SCHEMA, PRIMARY_KEY)
    if errors:
        return errors
    if not table["daily_id"].isin(daily_ids_written).all():
        errors.append("daily_id not in wq_measurement_daily (re-run build_daily_measurements.py "
                      "after clean_wq_results.py)")
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
    if bad := sorted(set(table["exposure_status"].dropna()) - set(EXPOSURE_STATUSES)):
        errors.append(f"unexpected exposure_status values {bad}")
    if table["exposure_status"].isna().any():
        errors.append("null exposure_status")
    assigned = table["exposure_status"].isin(ASSIGNED_STATUSES)
    if (table.loc[assigned, "assigned_AR_ID"] != table.loc[assigned, "AR_ID"]).any():
        errors.append("assigned rows whose assigned_AR_ID differs from AR_ID")
    if (table.loc[assigned].duplicated("measurement_id")).any():
        errors.append("a measurement is assigned to more than one removal")
    if table.loc[table["drainage_area_flag"].ne("ok"), "upstream_area_km2"].notna().any():
        errors.append("upstream_area_km2 set for a station whose drainage area isn't ok")
    if (table["drainage_area_ratio"] <= 0).any():
        errors.append("drainage_area_ratio <= 0")
    if table.loc[table["side"].ne("upstream"), "upstream_in_former_reservoir"].notna().any():
        errors.append("upstream_in_former_reservoir set for a station that isn't upstream")
    for column in ["backwater_km", "reservoir_volume_proxy", "reservoir_polygon_area_km2",
                   "dam_drainage_area_km2"]:
        if (table[column] <= 0).any():
            errors.append(f"{column} <= 0")
    if table["reservoir_volume_proxy"].notna().gt(table["backwater_km"].notna()).any():
        errors.append("reservoir_volume_proxy without backwater_km (slope / height rules differ)")
    if table.loc[table["reservoir_status"].ne("dam_at_outlet").fillna(True),
                 "reservoir_polygon_area_km2"].notna().any():
        errors.append("reservoir_polygon_area_km2 set for a dam not at a reservoir outlet")
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


def size_report(table):
    """Dams with each size variable (choice 15), over all dams in the table
    and over dams with downstream rows."""
    dams = table.drop_duplicates("AR_ID")
    downstream = set(table.loc[table["side"].eq("downstream"), "AR_ID"])
    pre = dams["reservoir_polygon_area_km2"].where(
        dams["reservoir_polygon_timing"].eq("edited_by_removal_year"))
    columns = {"dam_height_ft": dams["dam_height_ft"],
               "backwater_km": dams["backwater_km"],
               "reservoir_volume_proxy": dams["reservoir_volume_proxy"],
               "reservoir_polygon_area_km2": dams["reservoir_polygon_area_km2"],
               "  mapped by the removal year": pre,
               "dam_drainage_area_km2": dams["dam_drainage_area_km2"]}
    is_down = dams["AR_ID"].isin(downstream)
    lines = [f"  {'':<30} {'all dams':>14} {'downstream':>14}",
             f"  {'dams':<30} {len(dams):>14,} {int(is_down.sum()):>14,}"]
    for name, values in columns.items():
        n_all, n_down = int(values.notna().sum()), int(values[is_down].notna().sum())
        pct_all = 100 * n_all / len(dams) if len(dams) else 0
        pct_down = 100 * n_down / is_down.sum() if is_down.any() else 0
        lines.append(f"  {name:<30} {n_all:>7,} {pct_all:5.1f}% {n_down:>7,} {pct_down:5.1f}%")
    return "\n".join(lines)


def distance_band(distance, edges=DISTANCE_BANDS):
    """'00-05', '05-10', ... and '30+' beyond the last edge; 'unknown' when
    null (upper edges inclusive)."""
    labels = [f"{lo:02d}-{hi:02d}" for lo, hi in zip(edges[:-1], edges[1:])]
    out = pd.cut(distance, bins=edges + [float("inf")], labels=labels + [f"{edges[-1]}+"],
                 right=True, include_lowest=True).astype("string")
    return out.fillna("unknown")


def exposure_by_band_report(table):
    """Downstream rows by exposure_status and distance band, and dams with
    assigned rows per band: how overlapping and competing removals grow
    with distance."""
    down = table[table["side"].eq("downstream")]
    if down.empty:
        return "  no downstream rows"
    band = distance_band(down["network_distance_km"])
    rows = pd.crosstab(down["exposure_status"], band, margins=True, margins_name="all")
    dams = down[down["exposure_status"].isin(ASSIGNED_STATUSES)].groupby(
        band[down["exposure_status"].isin(ASSIGNED_STATUSES)])["AR_ID"].nunique()
    return (rows.to_string() + "\n\n  dams with assigned rows per band:\n"
            + dams.to_string())


def exposure_report(table):
    """Rows and measurements per exposure_status."""
    report = table.groupby("exposure_status").agg(
        rows=("measurement_id", "size"), measurements=("measurement_id", "nunique"),
        dams=("AR_ID", "nunique"))
    report["pct_rows"] = (100 * report["rows"] / len(table)).round(1)
    return report.sort_values("rows", ascending=False).to_string()


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
    require_file(output_dir / f"{RESERVOIR_TABLE}.parquet",
                 f"{RESERVOIR_TABLE} table (run build_dam_reservoir.py after build_views.py)")
    require_file(output_dir / "wq_measurement_daily.parquet",
                 "wq_measurement_daily table (run build_daily_measurements.py first)")

    if stale := stale_inputs(db_path):
        raise ValueError("Context tables are from an earlier run; no warehouse files were "
                         "written:\n  - " + "\n  - ".join(stale))

    table, expected, daily_ids_written = build_table(db_path)
    errors = validate(table, expected, daily_ids_written)
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
    print(f"Distinct measurements: {table['measurement_id'].nunique():,}; "
          f"distinct daily values (daily_id): {table['daily_id'].nunique():,}")
    print("\nRows per parameter and period:")
    counts = (table.assign(fraction=table["sample_fraction_std"].fillna("-"))
              .groupby(["parameter_std", "fraction", "period"]).size()
              .unstack(fill_value=0))
    print(counts.to_string())
    print("\nContext coverage:")
    print(coverage_report(table))
    print("\nExposure (assigned_* rows give one exposure per measurement):")
    print(exposure_report(table))
    print("\nDownstream rows by exposure status and distance band (km):")
    print(exposure_by_band_report(table))
    print("\nDam size variables (dams with a value; choice 15, nothing imputed):")
    print(size_report(table))
    print("\nRows with station flags (not applied):")
    for column in ["qc_far_snap", "qc_near_unsuitable_flowline", "qc_non_riverine_name",
                   "qc_saline"]:
        print(f"  {column:<28} {int(table[column].fillna(False).sum()):>10,}")
    print("\nLink features (pairs):")
    pairs = table.drop_duplicates(["AR_ID", "station_id"])
    print(pairs.groupby("side")["drainage_area_ratio"].describe(percentiles=[0.5, 0.9])
          .round(2).to_string())
    print(f"  upstream pairs possibly in the former reservoir: "
          f"{int(pairs['upstream_in_former_reservoir'].fillna(False).sum())}")
    print("\nFlow permanence (pairs; station reach x dam reach):")
    print(pairs.groupby(["station_flow_permanence", "dam_flow_permanence"], dropna=False)
          .size().to_string())
    flagged = pairs[pairs["qc_non_perennial"].fillna(False).astype(bool)]
    print(f"  qc_non_perennial pairs: {len(flagged):,} ({flagged['AR_ID'].nunique()} dams); "
          f"rows: {int(table['qc_non_perennial'].fillna(False).sum()):,}")
    with_flow = table.groupby("parameter_std")["has_same_day_discharge"].agg(["size", "sum"])
    with_flow = with_flow[with_flow["sum"] > 0].rename(columns={"size": "rows",
                                                                 "sum": "with_same_day_discharge"})
    print("\nRows with discharge measured the same day (parameters that have any):")
    print(with_flow.to_string() if len(with_flow) else "  none")
    print("\nDrainage area flags (rows):")
    print(table["drainage_area_flag"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
