"""
Expanded station recovery audit: how much useful data the unmatched stations
could add, and which ones are worth a manual review and a targeted download
(read-only for the pipeline; writes only to
data/processed/stage1_audit/<case_study>/station_recovery/ and, for the
summary stage, data/raw/wqp/summary/<case_study>/station_recovery/).

Separate from the accepted-station accuracy review (export_station_review.py).
The analysis population and matching decisions are not changed.

Stages (run in order; data/slurm/station_recovery.slurm):

  structure  (no network) every station the matching did not accept:
             ambiguous stations (linked to dams through their provisional
             reach; links unusable, samples never downloaded) and rejected
             stations (no reach, so no links). Breakdowns by matching reason,
             HUC2 region, network distance to the dam (ambiguous) or straight-
             line distance to the nearest dated accepted dam (rejected), side,
             and the dam's current state. Writes structure_stations.csv,
             structure_links.csv and structure_*.csv tables.
  summary    (light network) WQP's summary service for the ambiguous stations
             linked within the exposure limits: per station x year x
             characteristic, ResultCount and ActivityCount. These are counts,
             not results: one small request per SUMMARY_BATCH stations. Re-runs
             skip finished batches.
  estimate   maps characteristics to the warehouse parameters
             (wq_rules.PARAMETERS) and, per dam x station x parameter, counts
             results and years before / after removal in the window. A pair
             "qualifies" with the analysis rule (min_samples in min_years per
             side). Estimated gains per dam and parameter:
               new_baci_dam       a before/after-only dam gets a control for a
                                  parameter it has downstream rows for
               new_ba_dam         a dam not yet a before/after dam gets a
                                  downstream station
               new_baci_measure   a BACI dam gets a control for another
                                  parameter it has downstream rows for
               new_ba_measure     a before/after dam gets a downstream station
                                  for a parameter it lacks
               more_samples       an existing dam x parameter gets more samples
             Then per parameter: dams gained (before/after, BACI) vs today,
             and per station a priority. The 181 stations already downloaded
             (station_payoff_recount.py) are the calibration set: their
             summary-based prediction is compared with the real recount, and
             the stations confirmed there are merged in.

Caveats (printed with the results):
  - Summary counts are raw results: QC samples, other media, statistical
    summaries, etc. are not removed (the 181-station download kept ~24% of
    rows after cleaning). The calibration set measures how much this
    overstates.
  - The summary has no sample fraction, so phosphorus / orthophosphate are
    matched by parameter, not by fraction.
  - A provisional reach may be wrong (that is what the review decides), and a
    new downstream station isn't checked against closer competing removals.

Run on the cluster: sbatch data/slurm/station_recovery.slurm (STAGE=...)
"""

from __future__ import annotations

import argparse
import sys
import time
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import numpy as np
import pandas as pd
import requests

from analysis.audit_utils import connect, save_output, series_sql
from analysis.audits.dam_funnel_audit import MODEL_NEEDS_SQL, dam_status
from analysis.audits.export_dam_review import nearest_km
from analysis.audits.station_payoff_audit import prepare_links, usable_coverage
from analysis.audits.station_payoff_recount import qualifies
from analysis.mixed_model import SETTINGS as MODEL_SETTINGS
from etl.build_views import DEFAULT_MIN_SAMPLES, DEFAULT_MIN_YEARS
from etl.common import wq_rules
from etl.common.config import default_case_study, load_config
from etl.common.reach_inputs import STATION_ID, dam_id_text, read_per_huc8
from etl.common.warehouse import warehouse_dir

SUB = "station_recovery"
SUMMARY_URL = "https://www.waterqualitydata.us/data/summary/monitoringLocation/search"
SUMMARY_BATCH = 50
DISTANCE_BANDS = [0, 1, 5, 10, 20, 30, np.inf]
NEAR_KM = (1.0, 5.0, 10.0, 30.0)
GAINS = ["new_baci_dam", "new_ba_dam", "new_baci_measure", "new_ba_measure", "more_samples"]
ELIGIBILITY_GAINS = GAINS[:4]
MODEL_PARAMETERS = sorted({s.split(" (")[0] for s in MODEL_SETTINGS["series"]})


def out_dir(cfg):
    return cfg.project_root / "data" / "processed" / "stage1_audit" / cfg.name


def summary_dir(cfg):
    return cfg.project_root / "data" / "raw" / "wqp" / "summary" / cfg.name / SUB


def huc2(code):
    return code.astype("string").str.zfill(8).str[:2]


def band(km):
    return pd.cut(pd.to_numeric(km, errors="coerce"), DISTANCE_BANDS, right=True,
                  labels=["0-1", "1-5", "5-10", "10-20", "20-30", ">30"]).astype("string")


# --------------------------------------------------
# Dam state
# --------------------------------------------------

def dam_measures(analysis_path, settings=MODEL_SETTINGS):
    """Per dam x parameter: has_ba (assigned downstream rows) and has_control
    (upstream control rows outside the former reservoir), from
    analysis_measurement. Parameter level (fractions merged), to match the
    WQP summary."""
    treated = ", ".join(f"'{s}'" for s in settings["treated_exposure"])
    control = ", ".join(f"'{s}'" for s in settings["control_exposure"])
    frame = duckdb.sql(f"""
        SELECT CAST(AR_ID AS VARCHAR) AS dam_id, parameter_std AS parameter,
               bool_or(side = 'downstream' AND exposure_status IN ({treated})) AS has_ba,
               bool_or(side = 'upstream' AND exposure_status IN ({control})
                       AND NOT coalesce(upstream_in_former_reservoir, false)) AS has_control,
               count(*) AS rows
        FROM read_parquet('{Path(analysis_path).as_posix()}') GROUP BY ALL
    """).df()
    frame["dam_id"] = dam_id_text(frame["dam_id"])
    return frame


def dam_flags(measures):
    """dam_id -> ba_dam (any parameter with assigned downstream rows) and
    baci_dam (any parameter with both those and a control)."""
    both = measures.assign(both=measures["has_ba"].astype(bool) & measures["has_control"].astype(bool))
    return both.groupby("dam_id").agg(ba_dam=("has_ba", "any"), baci_dam=("both", "any"))


def dam_state(cfg, measures):
    """Per dated dam in dam_removal: funnel reason, model needs, usable
    coverage, ba_dam / baci_dam (any parameter)."""
    window = cfg.get("wq_window", {}) or {}
    eligibility = cfg.get("eligibility", {}) or {}
    exposure = cfg.get("exposure", {}) or {}
    con, _ = connect(warehouse_dir(cfg), window.get("years_before", 5), window.get("years_after", 5),
                     eligibility.get("min_samples", DEFAULT_MIN_SAMPLES),
                     eligibility.get("min_years", DEFAULT_MIN_YEARS),
                     temp_dir=out_dir(cfg) / "_duckdb_tmp",
                     max_downstream_km=exposure.get("max_downstream_km"),
                     max_upstream_km=exposure.get("max_upstream_km"))
    dams = dam_status(con).rename(columns={"AR_ID": "dam_id"}).set_index("dam_id")
    con.close()
    dams = dams.join(dam_flags(measures))
    dams[["ba_dam", "baci_dam"]] = dams[["ba_dam", "baci_dam"]].fillna(False).astype(bool)
    dams = dams[dams["removal_year"].notna()]
    dams["dam_category"] = np.select(
        [dams["baci_dam"], dams["ba_dam"], dams["reason"].eq("in_analysis_table")],
        ["baci", "ba_only", "in_table_no_ba"], "not_in_table:" + dams["reason"].astype(str))
    return dams


# --------------------------------------------------
# structure
# --------------------------------------------------

def unmatched_stations(cfg):
    """Every station_reaches row the matching did not accept."""
    reaches = read_per_huc8(cfg, "station_reaches", "station_reaches", cfg.huc8s)
    reaches = reaches.rename(columns={STATION_ID: "station_id"})
    reaches["station_id"] = reaches["station_id"].astype(str)
    reaches = reaches.drop_duplicates("station_id")
    keep = [c for c in ["station_id", "MonitoringLocationName", "huc8", "LatitudeMeasure",
                        "LongitudeMeasure", "snap_status", "snap_reason", "snap_evidence",
                        "provisional_nhdplus_id", "runner_up_id", "runner_up_name",
                        "n_eligible", "n_plausible", "point_polygon_layer"] if c in reaches]
    return reaches.loc[reaches["snap_status"].ne("accepted"), keep].reset_index(drop=True)


def rejected_proximity(rejected, dams_xy):
    """Straight-line km from each rejected station to the nearest dated
    accepted dam, and dams within NEAR_KM."""
    lon = pd.to_numeric(rejected["LongitudeMeasure"], errors="coerce")
    lat = pd.to_numeric(rejected["LatitudeMeasure"], errors="coerce")
    ok = lon.notna() & lat.notna()
    out = pd.DataFrame(index=rejected.index)
    out["nearest_dam_km"] = np.nan
    if ok.any() and len(dams_xy):
        out.loc[ok, "nearest_dam_km"] = nearest_km(lon[ok], lat[ok], dams_xy["lon"], dams_xy["lat"])
    return out


def structure(cfg):
    folder = out_dir(cfg)
    measures = dam_measures(warehouse_dir(cfg) / "analysis_measurement.parquet")
    dams = dam_state(cfg, measures)
    exposure = cfg.get("exposure", {}) or {}
    links = prepare_links(read_per_huc8(cfg, "reach_links", "reach_links", cfg.huc8s),
                          exposure.get("max_downstream_km"), exposure.get("max_upstream_km"))
    dams = dams.join(usable_coverage(links))
    stations = unmatched_stations(cfg)

    amb = links[~links["usable"] & links["station_status"].eq("ambiguous")
                & links["dam_status"].eq("accepted") & links["dam_id"].isin(dams.index)].copy()
    amb = amb.join(dams[["dam_category", "reason"]], on="dam_id")
    amb["distance_band"] = band(amb["network_distance_km"])
    amb["role"] = np.select([amb["side"].eq("downstream"), amb["is_control"],
                             amb["side"].eq("upstream")],
                            ["downstream", "upstream_control", "upstream_in_reservoir"], "unknown")

    reaches = read_per_huc8(cfg, "dam_reaches", "dam_reaches", cfg.huc8s)
    reaches["AR_ID"] = dam_id_text(reaches["AR_ID"])
    accepted = reaches[reaches["match_status"].eq("accepted") & reaches["AR_ID"].isin(dams.index)]
    dams_xy = pd.DataFrame({"lon": pd.to_numeric(accepted["snapped_lon"], errors="coerce"),
                            "lat": pd.to_numeric(accepted["snapped_lat"], errors="coerce")}).dropna()
    stations["huc2"] = huc2(stations["huc8"])
    stations = stations.join(rejected_proximity(stations, dams_xy))
    linked = amb.groupby("station_id").agg(
        linked_dams=("dam_id", "nunique"), in_limits_dams=("in_limits", "sum"),
        nearest_network_km=("network_distance_km", "min"),
        roles=("role", lambda s: ";".join(sorted(set(s)))))
    stations = stations.join(linked, on="station_id")
    stations["linked_dams"] = stations["linked_dams"].fillna(0).astype(int)
    stations["in_limits_dams"] = stations["in_limits_dams"].fillna(0).astype(int)
    stations["population"] = np.select(
        [stations["snap_status"].eq("ambiguous") & stations["in_limits_dams"].gt(0),
         stations["snap_status"].eq("ambiguous") & stations["linked_dams"].gt(0),
         stations["snap_status"].eq("ambiguous"),
         stations["nearest_dam_km"].le(NEAR_KM[1])],
        ["ambiguous_linked_in_limits", "ambiguous_linked_beyond_limits", "ambiguous_unlinked",
         "rejected_within_5km_of_dam"], "rejected_far")

    in_lim = amb[amb["in_limits"]]
    tables = {
        "structure_population.csv": stations.groupby(["population", "snap_status"]).size()
        .rename("stations").reset_index(),
        "structure_by_reason.csv": pd.crosstab(stations["snap_reason"].fillna("?"),
                                               stations["population"], margins=True).reset_index(),
        "structure_by_huc2.csv": pd.crosstab(stations["huc2"].fillna("?"),
                                             stations["population"], margins=True).reset_index(),
        "structure_links_by_band_role.csv": pd.crosstab(amb["distance_band"].fillna("?"),
                                                        amb["role"], margins=True).reset_index(),
        "structure_links_by_dam_category.csv": in_lim.groupby(["dam_category", "role"]).agg(
            links=("station_id", "size"), stations=("station_id", "nunique"),
            dams=("dam_id", "nunique")).reset_index(),
        "structure_rejected_near.csv": pd.DataFrame(
            [{"within_km": km, "rejected_stations": int(
                (stations["snap_status"].eq("rejected") & stations["nearest_dam_km"].le(km)).sum())}
             for km in NEAR_KM]),
    }
    link_columns = ["dam_id", "station_id", "side", "role", "relationship", "network_distance_km",
                    "distance_band", "in_limits", "station_reason", "station_in_mapped_reservoir",
                    "dam_category", "reason"]
    save_output(amb[link_columns], f"{SUB}/structure_links.csv", folder)
    save_output(stations, f"{SUB}/structure_stations.csv", folder)
    save_output(dams.reset_index()[["dam_id", "dam_name", "state", "huc8", "removal_year",
                                    "reason", "dam_category", "ba_dam", "baci_dam",
                                    "usable_downstream_in_limits", "usable_controls_in_limits"]],
                f"{SUB}/structure_dams.csv", folder)
    save_output(measures, f"{SUB}/dam_measures.csv", folder)
    for name, table in tables.items():
        save_output(table, f"{SUB}/{name}", folder)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Unmatched stations: {len(stations):,} "
          f"({stations['snap_status'].value_counts().to_dict()})")
    for name, table in tables.items():
        print(f"\n=== {name} ===")
        print(table.to_string(index=False))
    print(f"\nWrote {folder / SUB}")


# --------------------------------------------------
# summary
# --------------------------------------------------

def summary_targets(cfg):
    """Ambiguous stations linked within the limits to a dated accepted dam,
    plus the 181 already downloaded (calibration)."""
    links = pd.read_csv(out_dir(cfg) / SUB / "structure_links.csv", dtype={"station_id": str})
    ids = set(links.loc[links["in_limits"].astype(bool), "station_id"])
    payoff = out_dir(cfg) / "station_payoff" / "download_list.csv"
    if payoff.exists():
        ids |= set(pd.read_csv(payoff, dtype={"station_id": str})["station_id"])
    return sorted(ids)


def fetch_summary(ids, session, attempts=4):
    params = {"siteid": ";".join(ids), "summaryYears": "all", "dataProfile": "periodOfRecord",
              "mimeType": "csv", "zip": "no"}
    for attempt in range(attempts):
        response = session.get(SUMMARY_URL, params=params, timeout=300)
        if response.status_code == 200:
            text = response.text
            return pd.read_csv(StringIO(text), dtype=str) if text.strip() else pd.DataFrame()
        if response.status_code in (429, 500, 502, 503, 504):
            time.sleep(30 * (attempt + 1))
            continue
        response.raise_for_status()
    raise RuntimeError(f"WQP summary kept failing (HTTP {response.status_code})")


def summary(cfg):
    target = summary_targets(cfg)
    folder = summary_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.csv"
    manifest = (pd.read_csv(manifest_path, dtype=str) if manifest_path.exists()
                else pd.DataFrame(columns=["batch", "stations", "status", "rows"]))
    done = set(";".join(manifest.loc[manifest["status"].eq("ok"), "stations"]).split(";"))
    todo = [s for s in target if s not in done]
    batches = [todo[i:i + SUMMARY_BATCH] for i in range(0, len(todo), SUMMARY_BATCH)]
    first = int(pd.to_numeric(manifest["batch"], errors="coerce").max()) + 1 if len(manifest) else 1
    print(f"Summary targets: {len(target):,} stations; already done {len(target) - len(todo):,}; "
          f"{len(batches)} request(s) of up to {SUMMARY_BATCH}", flush=True)
    session = requests.Session()
    failures = 0
    for number, ids in enumerate(batches, start=first):
        try:
            frame = fetch_summary(ids, session)
            if len(frame):
                frame.to_csv(folder / f"summary_{number:04d}.csv", index=False)
            row = {"batch": number, "stations": ";".join(ids), "status": "ok", "rows": len(frame)}
            failures = 0
        except Exception as exc:          # noqa: BLE001 (record and continue)
            row = {"batch": number, "stations": ";".join(ids), "status": f"failed: {exc}",
                   "rows": 0}
            failures += 1
        manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
        manifest.to_csv(manifest_path, index=False)
        print(f"  batch {number}: {row['status']}, {row['rows']} rows", flush=True)
        if failures >= 5:
            print("Stopping after 5 failures in a row; re-run later (finished batches are kept).")
            sys.exit(1)
        time.sleep(1)
    failed = manifest["status"].str.startswith("failed").sum()
    print(f"Done; failed batches in the manifest: {failed} (re-run to retry)")
    if failed:
        sys.exit(1)


# --------------------------------------------------
# estimate
# --------------------------------------------------

def load_summary(folder):
    files = sorted(Path(folder).glob("summary_*.csv"))
    if not files:
        raise FileNotFoundError(f"No summary files in {folder}; run STAGE=summary first.")
    raw = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    out = pd.DataFrame({
        "station_id": raw["MonitoringLocationIdentifier"].str.strip(),
        "year": pd.to_numeric(raw["YearSummarized"], errors="coerce"),
        "parameter": raw["CharacteristicName"].map(wq_rules.PARAMETERS),
        "results": pd.to_numeric(raw["ResultCount"], errors="coerce").fillna(0),
        "activities": pd.to_numeric(raw.get("ActivityCount"), errors="coerce").fillna(0),
    })
    out = out.dropna(subset=["year", "parameter"])
    return out.groupby(["station_id", "year", "parameter"], as_index=False)[
        ["results", "activities"]].sum()


def estimated_pairs(summary_rows, links, years_before, years_after, min_samples, min_years):
    """Per dam x station x parameter: results and years before / after removal
    in the window (removal year left out), and whether it qualifies."""
    pairs = links[["dam_id", "station_id", "role", "removal_year"]].drop_duplicates() \
        .merge(summary_rows, on="station_id")
    offset = pairs["year"] - pairs["removal_year"]
    pairs["period"] = np.select([offset.between(-years_before, -1), offset.between(1, years_after)],
                                ["pre", "post"], "out")
    pairs = pairs[pairs["period"].ne("out") & pairs["results"].gt(0)]
    keys = ["dam_id", "station_id", "role", "parameter"]
    out = pairs.groupby(keys + ["period"]).agg(n=("results", "sum"), years=("year", "nunique")) \
        .unstack("period")
    out.columns = [f"{stat}_{period}" for stat, period in out.columns]
    for column in ["n_pre", "n_post", "years_pre", "years_post"]:
        if column not in out:
            out[column] = 0
    out = out.fillna(0).reset_index()
    # Months don't bind with min_years >= 2; give them a large value.
    out["months_pre"] = out["months_post"] = 12
    out["qualifies"] = qualifies(out, min_samples, min_years)
    return out


def pair_gains(pairs, measures, dams):
    """The gain (GAINS) each qualifying pair would give, or None."""
    m = measures.set_index(["dam_id", "parameter"])
    key = list(zip(pairs["dam_id"], pairs["parameter"]))
    has_ba = pd.Series([bool(m["has_ba"].get(k, False)) for k in key], index=pairs.index)
    has_ctl = pd.Series([bool(m["has_control"].get(k, False)) for k in key], index=pairs.index)
    ba_dam = pairs["dam_id"].map(dams["ba_dam"]).fillna(False).astype(bool)
    baci_dam = pairs["dam_id"].map(dams["baci_dam"]).fillna(False).astype(bool)
    down, ctl, ok = pairs["role"].eq("downstream"), pairs["role"].eq("upstream_control"), pairs["qualifies"]
    gain = np.select(
        [ok & ctl & has_ba & ~has_ctl & ~baci_dam,
         ok & down & ~ba_dam,
         ok & ctl & has_ba & ~has_ctl & baci_dam,
         ok & down & ba_dam & ~has_ba,
         ok & ((down & has_ba) | (ctl & has_ctl))],
        GAINS, "")
    gain = pd.Series(gain, index=pairs.index).astype(object)
    return gain.where(gain.ne(""), None)


def per_parameter(gained, measures):
    """Dams per parameter today and the estimated dams gained (BA, BACI)."""
    flagged = measures.assign(both=measures["has_ba"].astype(bool)
                              & measures["has_control"].astype(bool))
    today = flagged.groupby("parameter").agg(ba_dams=("has_ba", "sum"), baci_dams=("both", "sum"))
    new_ba = gained[gained["gain"].isin(["new_ba_dam", "new_ba_measure"])] \
        .groupby("parameter")["dam_id"].nunique().rename("ba_dams_gained")
    new_baci = gained[gained["gain"].isin(["new_baci_dam", "new_baci_measure"])] \
        .groupby("parameter")["dam_id"].nunique().rename("baci_dams_gained")
    more = gained[gained["gain"].eq("more_samples")].assign(
        n=lambda f: f["n_pre"] + f["n_post"]).groupby("parameter").agg(
        dams_with_more_samples=("dam_id", "nunique"), extra_results=("n", "sum"))
    out = today.join([new_ba, new_baci, more], how="outer").fillna(0).astype(int)
    out = out[out.index.isin(MODEL_PARAMETERS)]
    for kind in ("ba", "baci"):
        out[f"{kind}_dams_if_all_confirmed"] = out[f"{kind}_dams"] + out[f"{kind}_dams_gained"]
    out["baci_crosses_10"] = (out["baci_dams"] < 10) & (out["baci_dams_if_all_confirmed"] >= 10)
    out["baci_crosses_15"] = (out["baci_dams"] < 15) & (out["baci_dams_if_all_confirmed"] >= 15)
    return out.reset_index()


def station_priority(gained, stations, confirmed):
    """One row per station with any eligibility gain: best gain, dams and
    parameters, confirmed-by-download flag; ranked."""
    elig = gained[gained["gain"].isin(ELIGIBILITY_GAINS)]
    rank = {g: i for i, g in enumerate(GAINS)}
    by = elig.groupby("station_id")
    out = pd.DataFrame({
        "best_gain": by["gain"].agg(lambda s: min(s, key=rank.get)),
        "dams": by["dam_id"].agg(lambda s: ";".join(sorted(set(s)))),
        "n_dams": by["dam_id"].nunique(),
        "parameters": by["parameter"].agg(lambda s: ";".join(sorted(set(s)))),
        "n_dam_parameters": by.size(),
    }).reset_index()
    info = stations[[c for c in ["station_id", "MonitoringLocationName", "huc8", "huc2",
                                 "snap_reason", "provisional_nhdplus_id", "LatitudeMeasure",
                                 "LongitudeMeasure"] if c in stations]]
    out = out.merge(info, on="station_id", how="left")
    out["confirmed_by_download"] = out["station_id"].isin(confirmed)
    out["_r"] = out["best_gain"].map(rank)
    out = out.sort_values(["_r", "n_dams", "n_dam_parameters"], ascending=[True, False, False]) \
        .drop(columns="_r").reset_index(drop=True)
    out.insert(0, "priority_rank", np.arange(1, len(out) + 1))
    return out


def calibration(pairs, cfg):
    """Summary-based qualifies vs the real recount (station_payoff_recount)
    for the same dam x station x parameter, where both exist."""
    path = out_dir(cfg) / "station_payoff" / "recount_pairs.csv"
    wanted = out_dir(cfg) / "station_payoff" / "download_list.csv"
    if not path.exists() or not wanted.exists():
        return pd.DataFrame(), "no recount outputs; calibration skipped"
    real = pd.read_csv(path, dtype={"station_id": str, "dam_id": str})
    real["parameter"] = real["series"].str.split(" (", regex=False).str[0]
    real = real.groupby(["dam_id", "station_id", "parameter"])["has_both"].any().rename("real")
    tried = pd.read_csv(wanted, dtype={"station_id": str, "dam_id": str})
    tried = set(zip(tried["dam_id"], tried["station_id"]))
    est = pairs[[(d, s) in tried for d, s in zip(pairs["dam_id"], pairs["station_id"])]]
    est = est.groupby(["dam_id", "station_id", "parameter"])["qualifies"].any().rename("estimated")
    both = pd.concat([est, real], axis=1).fillna(False).astype(bool)
    table = pd.crosstab(both["estimated"], both["real"], rownames=["estimated"], colnames=["real"])
    tp = int((both["estimated"] & both["real"]).sum())
    precision = tp / max(int(both["estimated"].sum()), 1)
    recall = tp / max(int(both["real"].sum()), 1)
    return table, (f"calibration on {len(both)} dam x station x parameter pairs of the 181 "
                   f"downloaded stations: precision {precision:.2f}, recall {recall:.2f}")


def estimate(cfg):
    folder = out_dir(cfg)
    window = cfg.get("wq_window", {}) or {}
    eligibility = cfg.get("eligibility", {}) or {}
    yb, ya = int(window.get("years_before", 5)), int(window.get("years_after", 5))
    min_samples = int(eligibility.get("min_samples", DEFAULT_MIN_SAMPLES))
    min_years = int(eligibility.get("min_years", DEFAULT_MIN_YEARS))
    links = pd.read_csv(folder / SUB / "structure_links.csv", dtype={"station_id": str, "dam_id": str})
    links = links[links["in_limits"].astype(bool)
                  & links["role"].isin(["downstream", "upstream_control"])]
    dams = pd.read_csv(folder / SUB / "structure_dams.csv", dtype={"dam_id": str}).set_index("dam_id")
    measures = pd.read_csv(folder / SUB / "dam_measures.csv", dtype={"dam_id": str})
    dams[["ba_dam", "baci_dam"]] = dams[["ba_dam", "baci_dam"]].astype(bool)
    links = links.join(dams["removal_year"], on="dam_id")
    stations = pd.read_csv(folder / SUB / "structure_stations.csv", dtype={"station_id": str})

    pairs = estimated_pairs(load_summary(summary_dir(cfg)), links, yb, ya, min_samples, min_years)
    pairs["gain"] = pair_gains(pairs, measures, dams)
    gained = pairs[pairs["gain"].notna()]
    params = per_parameter(gained, measures)
    confirmed = set()
    recount = folder / "station_payoff" / "recount_stations.csv"
    if recount.exists():
        confirmed = set(pd.read_csv(recount, dtype={"station_id": str})["station_id"])
    priority = station_priority(gained, stations, confirmed)
    cal_table, cal_text = calibration(pairs, cfg)
    dam_gains = gained[gained["gain"].isin(ELIGIBILITY_GAINS)].groupby(["gain"]).agg(
        dams=("dam_id", "nunique"), stations=("station_id", "nunique"),
        dam_parameters=("parameter", "size")).reindex(ELIGIBILITY_GAINS).fillna(0).astype(int)

    for name, frame in [("estimate_pairs.csv", pairs), ("estimate_by_parameter.csv", params),
                        ("estimate_priority.csv", priority),
                        ("estimate_dam_gains.csv", dam_gains.reset_index())]:
        save_output(frame, f"{SUB}/{name}", folder)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_columns", 30)
    print(f"Rule {min_samples} results in {min_years} years per side (summary counts; see caveats)")
    print(f"Pairs with summary data: {len(pairs):,}; qualifying: {int(pairs['qualifies'].sum()):,}")
    print("\n=== Estimated eligibility gains (upper bounds before calibration) ===")
    print(dam_gains.to_string())
    print("\n=== Per parameter: dams today and estimated gains ===")
    print(params.to_string(index=False))
    print(f"\n=== Calibration ===\n{cal_text}")
    if len(cal_table):
        print(cal_table.to_string())
    print(f"\n=== Priority stations (eligibility gains): {len(priority)} ===")
    print(priority.head(60).to_string(index=False))
    print(f"\nWrote {folder / SUB}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--stage", choices=["structure", "summary", "estimate"], required=True)
    args = parser.parse_args()
    cfg = load_config(args.case_study)
    {"structure": structure, "summary": summary, "estimate": estimate}[args.stage](cfg)


if __name__ == "__main__":
    main()
