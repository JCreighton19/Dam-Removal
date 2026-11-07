"""
Dam funnel audit: why most dated removals never reach the analysis table
(read-only; writes only to data/processed/stage1_audit/<case_study>/dam_funnel/).

Every dam in the removal list gets the FIRST stage at which it drops out, in
the order the pipeline applies them, using the same views and rules as
build_views.py (analysis.audit_utils.connect: an in-memory DuckDB over the
warehouse Parquet files with the config's window, eligibility thresholds and
link distance limits; warehouse.duckdb is never opened):

    not_matched_no_record   in the dam list but in no dam_reaches file (outside
                            the case study's HUC8s, unusable coordinates, or no
                            stage output for its HUC8)
    match_rejected          dam matching rejected it (match_reason says why,
                            e.g. no_nhdplus_package)
    match_ambiguous         dam matching couldn't choose its reach
    accepted_not_loaded     accepted but not in dam_removal (should be none)
    no_removal_year         Year_Removed missing, so no before / after
    no_linked_station       no usable station up- or downstream at any distance
    links_beyond_limits     linked stations exist, but none within the distance
                            limits (config exposure: 30 km down / 10 km up)
    no_samples_in_window    stations within the limits, but no sample in the
                            5 years before or after (removal year excluded)
    one_side_only           samples before OR after, never both for one
                            station x parameter
    below_minimum           both sides, but fewer than min_samples samples or
                            min_years distinct years on a side
    in_analysis_table       has_both for at least one station x parameter: the
                            dams of analysis_measurement

For dams in the analysis table it also flags what the models need
(analysis_measurement): a downstream station whose samples are credited to
this removal, and an upstream control outside the former reservoir.

Diagnostics for deciding which losses are fixable (nothing is changed):
  - timing: losses by removal decade; post_window_incomplete = removal year +
    years_after is later than the last sample in the data.
  - thresholds: dams that would pass looser min_samples / min_years rules,
    from the existing counts (eligibility_pairs within the limits). The
    window itself (5 years) can't be widened this way: WQP results were only
    downloaded inside it.
  - station types: WQP station types in the raw downloads, which ones the
    config's station_types filter drops, and, per dam without a usable
    downstream station, river-like dropped stations within 5 km (straight
    line, same HUC8; an upper bound, not a network link).
  - watershed scope: for dams without a downstream station within the
    limit, a walk down the NHDPlus main path (up to the downstream limit)
    reports whether the river leaves the dam's HUC8 and whether the HUC8 it
    enters is in the case study. Stations there were never downloaded.
    --skip-network skips this (it reads each NHDPlus package's flow tables).

Outputs (dam_funnel/)
---------------------
    dam_funnel.csv             one row per listed dam: reason and diagnostics
    funnel_summary.csv         dams per reason, in pipeline order, with the
                               running count of dams left
    funnel_by_decade.csv       dated dams: reason x removal decade
    threshold_whatif.csv       dams with has_both under looser thresholds
    station_types.csv          raw WQP stations by type: kept / dropped
    model_needs.csv            analysis-table dams: assigned downstream rows
                               and upstream controls

Run on the cluster: AUDIT=funnel sbatch data/slurm/stage1_audits.slurm
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis.audit_utils import connect, save_output
from etl.build_views import ASSIGNED_STATUSES, DEFAULT_MIN_SAMPLES, DEFAULT_MIN_YEARS
from etl.common.config import default_case_study, load_config
from etl.common.reach_inputs import dam_id_text, read_per_huc8
from etl.common.warehouse import warehouse_dir

REASONS = [
    ("not_matched_no_record", "in the dam list but in no dam-matching output"),
    ("match_rejected", "dam matching rejected it"),
    ("match_ambiguous", "dam matching couldn't choose its river reach"),
    ("accepted_not_loaded", "matched but not in dam_removal"),
    ("no_removal_year", "no removal year"),
    ("no_linked_station", "no usable station on its river network"),
    ("links_beyond_limits", "stations only beyond the distance limits"),
    ("no_samples_in_window", "no samples within 5 years before or after"),
    ("one_side_only", "samples before or after, never both"),
    ("below_minimum", "both sides, but below the minimum samples / years"),
    ("in_analysis_table", "in the analysis table"),
]
REASON_ORDER = [r for r, _ in REASONS]
THRESHOLDS = [(3, 2), (2, 2), (3, 1), (2, 1), (1, 1)]
NEAR_KM = 5.0
RIVER_LIKE = re.compile(r"river|stream|riverine", re.IGNORECASE)


# --------------------------------------------------
# Per-dam status from the warehouse views
# --------------------------------------------------

DAM_STATUS_SQL = """
WITH links_all AS (
    SELECT AR_ID,
           count(*) AS links_any_distance,
           count(*) FILTER (WHERE side = 'downstream') AS downstream_links_any_distance,
           min(network_distance_km) FILTER (WHERE side = 'downstream') AS nearest_downstream_km,
           min(network_distance_km) FILTER (WHERE side = 'upstream') AS nearest_upstream_km
    FROM removal_link_all GROUP BY AR_ID
),
links_lim AS (
    SELECT AR_ID,
           count(*) AS links_within_limits,
           count(*) FILTER (WHERE side = 'downstream') AS downstream_links_within_limits
    FROM removal_link GROUP BY AR_ID
),
with_data AS (
    SELECT l.AR_ID, count(DISTINCT l.station_id) AS stations_with_any_data
    FROM removal_link l
    JOIN (SELECT DISTINCT station_id FROM wq_measurement) m USING (station_id)
    GROUP BY l.AR_ID
),
win AS (
    SELECT AR_ID,
           count(*) FILTER (WHERE period = 'pre') AS samples_pre,
           count(*) FILTER (WHERE period = 'post') AS samples_post,
           count(*) FILTER (WHERE period = 'removal_year') AS samples_removal_year,
           count(*) FILTER (WHERE period = 'outside') AS samples_outside_window
    FROM measurement_window GROUP BY AR_ID
),
pairs AS (
    SELECT AR_ID,
           bool_or(n_pre > 0 AND n_post > 0) AS any_pair_both_sides,
           bool_or(has_both) AS has_both,
           bool_or(has_both AND side = 'downstream') AS has_both_downstream,
           bool_or(has_both AND side = 'upstream') AS has_both_upstream,
           max(least(n_pre, n_post)) AS best_pair_min_samples,
           max(least(years_pre, years_post)) AS best_pair_min_years
    FROM eligibility_pairs GROUP BY AR_ID
)
SELECT d.AR_ID, d.Dam_Name AS dam_name, d.State AS state, d.HUC8 AS huc8,
       d.Year_Removed AS removal_year, d.NHDPlusID AS dam_nhdplus_id,
       d.snap_fraction AS dam_snap_fraction, d.Latitude AS latitude, d.Longitude AS longitude,
       la.* EXCLUDE (AR_ID), ll.* EXCLUDE (AR_ID), wd.* EXCLUDE (AR_ID),
       w.* EXCLUDE (AR_ID), p.* EXCLUDE (AR_ID)
FROM dam_removal d
LEFT JOIN links_all la USING (AR_ID)
LEFT JOIN links_lim ll USING (AR_ID)
LEFT JOIN with_data wd USING (AR_ID)
LEFT JOIN win w USING (AR_ID)
LEFT JOIN pairs p USING (AR_ID)
"""


def warehouse_reason(row):
    """First failing stage for a dam in dam_removal (REASONS order)."""
    if pd.isna(row["removal_year"]):
        return "no_removal_year"
    if not row["links_any_distance"] > 0:
        return "no_linked_station"
    if not row["links_within_limits"] > 0:
        return "links_beyond_limits"
    if not (row["samples_pre"] > 0 or row["samples_post"] > 0):
        return "no_samples_in_window"
    if row["any_pair_both_sides"] is not True:
        return "one_side_only"
    if row["has_both"] is not True:
        return "below_minimum"
    return "in_analysis_table"


def dam_status(con):
    frame = con.sql(DAM_STATUS_SQL).df()
    frame["AR_ID"] = dam_id_text(frame["AR_ID"])
    counts = [c for c in frame if c.startswith(("links_", "downstream_links", "stations_with",
                                                "samples_"))]
    frame[counts] = frame[counts].fillna(0).astype(int)
    for c in ("any_pair_both_sides", "has_both", "has_both_downstream", "has_both_upstream"):
        frame[c] = frame[c].astype("boolean").fillna(False).astype(bool)
    frame["reason"] = [warehouse_reason(r) for _, r in frame.iterrows()]
    return frame


def listed_and_matched(dam_list, dam_reaches):
    """(listed dams, their matching outcome) -> rows for dams that never
    reached dam_removal, with a reason. dam_list: AR_ID (+ name, state);
    dam_reaches: AR_ID, match_status, match_reason."""
    listed = dam_list.assign(AR_ID=dam_id_text(dam_list["AR_ID"])).drop_duplicates("AR_ID")
    matched = dam_reaches.assign(AR_ID=dam_id_text(dam_reaches["AR_ID"])).drop_duplicates("AR_ID")
    out = listed.merge(matched[["AR_ID", "huc8", "match_status", "match_reason"]],
                       on="AR_ID", how="left")
    out["reason"] = np.select(
        [out["match_status"].isna(), out["match_status"].eq("rejected"),
         out["match_status"].eq("ambiguous")],
        ["not_matched_no_record", "match_rejected", "match_ambiguous"], "accepted")
    return out


# --------------------------------------------------
# Diagnostics
# --------------------------------------------------

THRESHOLD_SQL = """
SELECT {s} AS min_samples, {y} AS min_years,
       count(DISTINCT AR_ID) AS dams,
       count(DISTINCT AR_ID) FILTER (WHERE side = 'downstream') AS dams_downstream
FROM eligibility_pairs
WHERE n_pre >= {s} AND years_pre >= {y} AND n_post >= {s} AND years_post >= {y}
"""


def threshold_whatif(con, current):
    rows = [con.sql(THRESHOLD_SQL.format(s=s, y=y)).df() for s, y in THRESHOLDS]
    out = pd.concat(rows, ignore_index=True)
    out["current_rule"] = [(s, y) == tuple(current) for s, y in THRESHOLDS]
    base = out.loc[out["current_rule"], "dams"]
    out["gain_vs_current"] = out["dams"] - (int(base.iloc[0]) if len(base) else 0)
    return out


def by_decade(dams):
    dated = dams[dams["removal_year"].notna()].copy()
    dated["decade"] = (dated["removal_year"].astype(int) // 10 * 10).astype(str) + "s"
    table = pd.crosstab(dated["decade"], pd.Categorical(dated["reason"], REASON_ORDER),
                        dropna=False)
    table = table.loc[:, table.sum() > 0]
    table.columns = [str(c) for c in table.columns]      # plain labels, so totals can be added
    table["dated_dams"] = table.sum(axis=1)
    table["share_in_analysis_%"] = 100 * table.get("in_analysis_table", 0) / table["dated_dams"]
    table.columns.name = None
    return table.reset_index()


def summary(all_dams):
    counts = all_dams["reason"].value_counts().reindex(REASON_ORDER, fill_value=0)
    out = pd.DataFrame({"reason": REASON_ORDER, "meaning": [m for _, m in REASONS],
                        "dams": counts.to_numpy()})
    total = int(out["dams"].sum())
    out["dams_left_after_this_stage"] = total - out["dams"].cumsum()
    out.loc[out["reason"].eq("in_analysis_table"), "dams_left_after_this_stage"] = \
        out.loc[out["reason"].eq("in_analysis_table"), "dams"]
    out["share_%"] = 100 * out["dams"] / total if total else np.nan
    return out


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def read_raw_stations(station_dir, huc8s):
    frames = []
    for huc8 in huc8s:
        path = Path(station_dir) / f"{huc8}.csv"
        if path.exists():
            frame = pd.read_csv(path, dtype=str, low_memory=False,
                                usecols=lambda c: c in {"MonitoringLocationIdentifier",
                                                        "MonitoringLocationTypeName",
                                                        "LatitudeMeasure", "LongitudeMeasure"})
            frames.append(frame.assign(huc8=huc8))
    if not frames:
        return pd.DataFrame(columns=["MonitoringLocationIdentifier", "MonitoringLocationTypeName",
                                     "LatitudeMeasure", "LongitudeMeasure", "huc8"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("MonitoringLocationIdentifier")


def station_types(raw, kept_types):
    raw = raw.assign(type=raw["MonitoringLocationTypeName"].fillna("(blank)"))
    out = raw.groupby("type").agg(stations=("MonitoringLocationIdentifier", "size"),
                                  huc8s=("huc8", "nunique")).reset_index()
    out["kept_by_filter"] = out["type"].isin(kept_types)
    out["river_like_but_dropped"] = ~out["kept_by_filter"] & out["type"].str.contains(RIVER_LIKE)
    return out.sort_values("stations", ascending=False)


def nearby_stations(dams, raw, kept_types, km=NEAR_KM):
    """Per dam: kept-type and dropped river-like stations within km (straight
    line, same HUC8)."""
    raw = raw.assign(lat=pd.to_numeric(raw["LatitudeMeasure"], errors="coerce"),
                     lon=pd.to_numeric(raw["LongitudeMeasure"], errors="coerce")).dropna(
        subset=["lat", "lon"])
    raw["kept"] = raw["MonitoringLocationTypeName"].isin(kept_types)
    raw["river_like_dropped"] = ~raw["kept"] & raw["MonitoringLocationTypeName"].fillna("") \
        .str.contains(RIVER_LIKE)
    by_huc8 = dict(tuple(raw.groupby("huc8")))
    out = []
    for _, d in dams.iterrows():
        group = by_huc8.get(str(d["huc8"]))
        if group is None or pd.isna(d["latitude"]):
            out.append((0, 0))
            continue
        near = haversine_km(d["latitude"], d["longitude"], group["lat"].to_numpy(),
                            group["lon"].to_numpy()) <= km
        out.append((int((near & group["kept"].to_numpy()).sum()),
                    int((near & group["river_like_dropped"].to_numpy()).sum())))
    return pd.DataFrame(out, index=dams.index,
                        columns=[f"kept_type_stations_within_{km:g}km",
                                 f"dropped_river_like_stations_within_{km:g}km"])


def walk_downstream(start_id, start_km, reach_code, length_km, next_reach, max_km):
    """Follow the main path down from a dam until it leaves the dam's HUC8,
    the package, or the network, or max_km is reached. Returns (end, km,
    next_huc8): end in left_huc8 / left_package / network_end / reached_limit."""
    huc8 = str(reach_code.get(start_id, ""))[:8]
    km, current = start_km, start_id
    for _ in range(100_000):
        if km > max_km:
            return "reached_limit", km, None
        nxt = next_reach.get(current)
        if nxt is None:
            return "network_end", km, None
        if nxt not in reach_code:
            return "left_package", km, None
        code = str(reach_code[nxt])[:8]
        if code != huc8:
            return "left_huc8", km, code
        km += float(length_km.get(nxt, 0) or 0)
        current = nxt
    return "network_end", km, None


def main_path(flow, divergence):
    """{from: to} along the main path: at a split, the branch that isn't a
    minor divergence (VAA Divergence 2), then the smaller id."""
    flow = flow.dropna().astype("int64")
    flow = flow[flow["ToNHDPID"] != 0]
    flow = flow.assign(minor=flow["ToNHDPID"].map(divergence).eq(2))
    flow = flow.sort_values(["FromNHDPID", "minor", "ToNHDPID"])
    return dict(flow.drop_duplicates("FromNHDPID")[["FromNHDPID", "ToNHDPID"]].to_numpy())


def scope_walks(dams, cfg, max_km):
    """For each dam (with dam_nhdplus_id, dam_snap_fraction, huc8): where its
    river goes within max_km."""
    from etl.common import water_features as wf
    from etl.common.nhdplus import find_huc8_geodatabase
    from etl.station_candidates import package_tables, read_layer

    scope = set(cfg.huc8s)
    extracted = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    out = pd.DataFrame(index=dams.index, columns=["river_end_within_limit", "river_km_to_end",
                                                  "next_huc8", "next_huc8_in_scope"])
    packages = {}
    for index, d in dams.iterrows():
        try:
            packages.setdefault(find_huc8_geodatabase(extracted, str(d["huc8"])), []).append(index)
        except FileNotFoundError:
            out.loc[index, "river_end_within_limit"] = "no_package"
    for number, (path, indexes) in enumerate(packages.items(), 1):
        print(f"  [{number}/{len(packages)}] {Path(path).name}: {len(indexes)} dams", flush=True)
        lines = read_layer(path, "NHDFlowline", ["NHDPlusID", "ReachCode", "LengthKM"],
                           geometry=False)
        lines = pd.DataFrame({"NHDPlusID": wf.clean_nhdplus_id(lines["NHDPlusID"]),
                              "ReachCode": lines["ReachCode"].astype(str),
                              "LengthKM": pd.to_numeric(lines["LengthKM"], errors="coerce")}) \
            .dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID")
        lines["NHDPlusID"] = lines["NHDPlusID"].astype("int64")
        vaa, flow = package_tables(path)
        divergence = ({} if vaa is None else
                      dict(zip(vaa["NHDPlusID"].astype("int64"), vaa["divergence"])))
        flow = pd.DataFrame({"FromNHDPID": wf.clean_nhdplus_id(flow["FromNHDPID"]),
                             "ToNHDPID": wf.clean_nhdplus_id(flow["ToNHDPID"])})
        nxt = main_path(flow, divergence)
        code = dict(zip(lines["NHDPlusID"], lines["ReachCode"]))
        length = dict(zip(lines["NHDPlusID"], lines["LengthKM"]))
        for index in indexes:
            d = dams.loc[index]
            if pd.isna(d["dam_nhdplus_id"]):
                out.loc[index, "river_end_within_limit"] = "no_reach"
                continue
            start = int(d["dam_nhdplus_id"])
            fraction = d["dam_snap_fraction"] if pd.notna(d["dam_snap_fraction"]) else 0.5
            offset = (1 - float(fraction)) * float(length.get(start, 0) or 0)
            end, km, next_huc8 = walk_downstream(start, offset, code, length, nxt, max_km)
            out.loc[index] = [end, km, next_huc8,
                              None if next_huc8 is None else next_huc8 in scope]
    return out


MODEL_NEEDS_SQL = f"""
SELECT AR_ID,
       bool_or(side = 'downstream' AND exposure_status IN {tuple(ASSIGNED_STATUSES)})
           AS has_assigned_downstream,
       bool_or(side = 'upstream' AND exposure_status = 'upstream_control'
               AND NOT coalesce(upstream_in_former_reservoir, FALSE)) AS has_upstream_control
FROM read_parquet('{{path}}')
GROUP BY AR_ID
"""


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--threads", type=int)
    parser.add_argument("--memory-limit", help="DuckDB memory limit, e.g. 12GB.")
    parser.add_argument("--skip-network", action="store_true",
                        help="Skip the NHDPlus walk that checks the watershed scope.")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    window = cfg.get("wq_window", {}) or {}
    eligibility = cfg.get("eligibility", {}) or {}
    exposure = cfg.get("exposure", {}) or {}
    min_rule = (eligibility.get("min_samples", DEFAULT_MIN_SAMPLES),
                eligibility.get("min_years", DEFAULT_MIN_YEARS))
    out_dir = cfg.project_root / "data" / "processed" / "stage1_audit" / cfg.name
    sub = "dam_funnel"
    max_down = exposure.get("max_downstream_km") or 30

    con, _ = connect(warehouse_dir(cfg), window.get("years_before", 5),
                     window.get("years_after", 5), *min_rule,
                     threads=args.threads, memory_limit=args.memory_limit,
                     temp_dir=out_dir / "_duckdb_tmp",
                     max_downstream_km=exposure.get("max_downstream_km"),
                     max_upstream_km=exposure.get("max_upstream_km"))

    # Stages inside the warehouse.
    loaded = dam_status(con)
    data_end_year = int(con.sql("SELECT max(year(sample_date)) FROM wq_measurement").fetchone()[0])
    loaded["post_window_incomplete"] = (loaded["removal_year"] + window.get("years_after", 5)
                                        > data_end_year)

    # Stages before it: the dam list and dam matching.
    dam_list = pd.read_csv(cfg.path("dam_source"), dtype=str, low_memory=False)
    dam_list = dam_list.rename(columns={"Dam_Name": "dam_name", "State": "state"})
    try:
        reaches = read_per_huc8(cfg, "dam_reaches", "dam_reaches", cfg.huc8s)
    except (FileNotFoundError, ValueError) as exc:
        print(f"No dam_reaches outputs ({exc}); matching stages left out.")
        reaches = pd.DataFrame(columns=["AR_ID", "huc8", "match_status", "match_reason"])
    early = listed_and_matched(dam_list[["AR_ID"] + [c for c in ("dam_name", "state")
                                                     if c in dam_list]], reaches)
    accepted_missing = early["reason"].eq("accepted") & ~early["AR_ID"].isin(loaded["AR_ID"])
    early.loc[accepted_missing, "reason"] = "accepted_not_loaded"
    early = early[~early["reason"].eq("accepted")]
    dams = pd.concat([early, loaded.merge(reaches.assign(AR_ID=dam_id_text(reaches["AR_ID"]))
                                          [["AR_ID", "match_status", "match_reason"]]
                                          .drop_duplicates("AR_ID"), on="AR_ID", how="left")],
                     ignore_index=True)

    # What the models need, for dams in the analysis table.
    analysis = warehouse_dir(cfg) / "analysis_measurement.parquet"
    if analysis.exists():
        needs = con.sql(MODEL_NEEDS_SQL.format(path=analysis.resolve())).df()
        needs["AR_ID"] = dam_id_text(needs["AR_ID"])
        dams = dams.merge(needs, on="AR_ID", how="left")
        in_table = dams["reason"].eq("in_analysis_table")
        check = set(needs["AR_ID"]) ^ set(dams.loc[in_table, "AR_ID"])
        print(f"analysis_measurement dams: {len(needs)}; funnel in_analysis_table: "
              f"{int(in_table.sum())}; differing: {len(check)} {sorted(check)[:10]}")

    # Coverage artifacts, for dams without a usable downstream station.
    no_down = dams["removal_year"].notna() & ~dams["has_both_downstream"].fillna(False).astype(bool)
    kept_types = list(cfg.get("station_types", ["Stream", "River/Stream"]))
    raw = read_raw_stations(cfg.project_root / "data" / "raw" / "wqp" / "stations", cfg.huc8s)
    types = station_types(raw, kept_types)
    dams = dams.join(nearby_stations(dams[no_down], raw, kept_types))
    walk_set = dams["removal_year"].notna() & dams["downstream_links_within_limits"].fillna(0).eq(0)
    if args.skip_network:
        print("Network walk skipped (--skip-network).")
    else:
        print(f"Walking downstream from {int(walk_set.sum())} dams without a downstream station "
              f"within {max_down} km:")
        dams = dams.join(scope_walks(dams[walk_set], cfg, float(max_down)))

    funnel = summary(dams)
    decade = by_decade(dams)
    whatif = threshold_whatif(con, min_rule)
    model = pd.DataFrame()
    if "has_assigned_downstream" in dams:
        t = dams[dams["reason"].eq("in_analysis_table")]
        down = t["has_assigned_downstream"].fillna(False).astype(bool)
        up = t["has_upstream_control"].fillna(False).astype(bool)
        model = pd.DataFrame({"group": ["in analysis table", "assigned downstream rows",
                                        "upstream control rows", "both (BACI possible)"],
                              "dams": [len(t), int(down.sum()), int(up.sum()),
                                       int((down & up).sum())]})
    outputs = [("dam_funnel.csv", dams), ("funnel_summary.csv", funnel),
               ("funnel_by_decade.csv", decade), ("threshold_whatif.csv", whatif),
               ("station_types.csv", types), ("model_needs.csv", model)]
    for name, frame in outputs:
        save_output(frame, f"{sub}/{name}", out_dir)
    con.close()

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(f"\nCase study: {cfg.name}; window {window}; thresholds {min_rule}; limits "
          f"{exposure.get('max_downstream_km')} km down / {exposure.get('max_upstream_km')} km up; "
          f"last sample year {data_end_year}")
    print("\n=== Funnel: where each listed dam drops out ===")
    print(funnel.to_string(index=False))
    print("\n=== Dated dams by removal decade ===")
    print(decade.to_string(index=False))
    dated = dams[dams["removal_year"].notna()]
    print(f"\nPost window incomplete (removal year + {window.get('years_after', 5)} > "
          f"{data_end_year}): {int(dated['post_window_incomplete'].fillna(False).sum())} dated dams; "
          f"by reason:")
    incomplete = dated[dated["post_window_incomplete"].fillna(False).astype(bool)]["reason"]
    print(incomplete.value_counts().to_string() if len(incomplete) else "  none")
    print("\n=== Looser thresholds (counts only; current rule marked) ===")
    print(whatif.to_string(index=False))
    print("\n=== WQP station types in the raw downloads ===")
    print(types.head(30).to_string(index=False))
    near = [c for c in dams if c.startswith("dropped_river_like")]
    if near:
        c = near[0]
        lost = dams[no_down]
        print(f"\nDams without a usable downstream station: {len(lost)}; with dropped river-like "
              f"stations within {NEAR_KM:g} km: {int((lost[c] > 0).sum())}; with kept-type "
              f"stations within {NEAR_KM:g} km: "
              f"{int((lost[f'kept_type_stations_within_{NEAR_KM:g}km'] > 0).sum())}")
    if "river_end_within_limit" in dams:
        walked = dams[walk_set]
        print(f"\n=== Where the river goes within {max_down} km (dams without a downstream "
              f"station within the limit: {len(walked)}) ===")
        print(walked.groupby(["river_end_within_limit", "next_huc8_in_scope"], dropna=False)
              .size().rename("dams").to_string())
    if len(model):
        print("\n=== What the models need, among analysis-table dams ===")
        print(model.to_string(index=False))
    print(f"\nWrote {', '.join(n for n, _ in outputs)} to {out_dir / sub}")


if __name__ == "__main__":
    main()
