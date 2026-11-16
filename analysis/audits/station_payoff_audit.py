"""
Station payoff audit: which ambiguous (unmatched) stations could add sample
size if a review accepted their reach (read-only; writes only to
data/processed/stage1_audit/<case_study>/station_payoff/).

Why: link_reaches.py links an ambiguous station from its provisional reach,
but the link is not usable (only both-accepted pairs, or confluence-ambiguous
stations robust to the reach choice, are), and download_wqp_results.py only
downloads usable pairs. So these stations' samples were never downloaded and
never counted. Rejected stations have no reach, so no links, and aren't
counted here.

For every accepted, dated dam (in dam_removal) the audit takes its current
state from the warehouse (the same views as dam_funnel_audit.py) and the
ambiguous-station links inside the exposure limits (config `exposure`:
downstream <= max_downstream_km, upstream <= max_upstream_km; same-reach
links are split by position as in notebooks/qa/dam_funnel.ipynb section 5;
upstream stations in the dam's mapped reservoir don't count as controls).
What an ambiguous station could give a dam, best first:

    baci_control      the dam already has an assigned downstream station in
                      the analysis table but no upstream control: an
                      upstream station could make it a BACI dam
    first_downstream  the dam has no usable downstream station within the
                      limit: a downstream station could make before/after
                      possible
    dam_not_in_table  the dam has stations but isn't in the analysis table
                      (no samples / one side only / below the minimum): more
                      samples could bring it in
    extra_station     the dam is already in the analysis table: more data
                      for an existing model dam (precision only)

These are UPPER BOUNDS: the stations' samples were never downloaded, so it
is unknown whether they have enough data in the dam's window. The next step
(a targeted download) turns them into real counts. A station's reach may
also be wrong, which is why it needs review before use.

Outputs (station_payoff/)
-------------------------
    station_payoff_links.csv     one row per (dam, ambiguous station) link in
                                 the limits, with the dam's state and the gain
    station_payoff_stations.csv  one row per ambiguous station with any gain:
                                 matching reason / evidence, reported point,
                                 provisional and runner-up reach, best gain,
                                 dams it could help, review_rank
    station_payoff_dams.csv      one row per dam an ambiguous station could
                                 help: state, gain flags, stations per side
    station_payoff_summary.csv   dams and stations by gain (and by station
                                 matching reason)

Run on the cluster: AUDIT=station_payoff sbatch data/slurm/stage1_audits.slurm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from analysis.audit_utils import connect, save_output
from analysis.audits.dam_funnel_audit import MODEL_NEEDS_SQL, dam_status
from etl.build_views import DEFAULT_MIN_SAMPLES, DEFAULT_MIN_YEARS
from etl.common.config import default_case_study, load_config
from etl.common.reach_inputs import STATION_ID, dam_id_text, read_per_huc8
from etl.common.warehouse import warehouse_dir

GAINS = [
    ("baci_control", "an upstream control for a before/after dam without one (possible new BACI dam)"),
    ("first_downstream", "a first downstream station within the limit (possible new before/after dam)"),
    ("dam_not_in_table", "more samples for a dam with stations but not in the analysis table"),
    ("extra_station", "an extra station for a dam already in the analysis table"),
]
GAIN_ORDER = [g for g, _ in GAINS]
NOT_IN_TABLE = ("no_samples_in_window", "one_side_only", "below_minimum")
STATION_COLUMNS = [STATION_ID, "MonitoringLocationName", "huc8", "LatitudeMeasure",
                   "LongitudeMeasure", "snap_status", "snap_reason", "snap_evidence",
                   "provisional_nhdplus_id", "n_eligible", "n_plausible", "runner_up_id",
                   "runner_up_distance_m", "runner_up_name"]


# --------------------------------------------------
# Links
# --------------------------------------------------

def link_side(links):
    """upstream / downstream for every link; same-reach links by position
    (station further down the reach = downstream); None when level."""
    same = links["relationship"].eq("same_reach")
    down = links["station_fraction"] > links["dam_fraction"]
    up = links["station_fraction"] < links["dam_fraction"]
    side = links["relationship"].where(~same)
    side = side.mask(same & down, "downstream").mask(same & up, "upstream")
    return side.where(side.isin(["upstream", "downstream"]))


def in_limits(links, max_down, max_up):
    km = links["network_distance_km"]
    down = links["side"].eq("downstream") & (km <= max_down if max_down else True)
    up = links["side"].eq("upstream") & (km <= max_up if max_up else True)
    return down | up


def prepare_links(links, max_down, max_up):
    """Connected links with side, in_limits and is_control (upstream, not in
    the dam's mapped reservoir)."""
    out = links.copy()
    out["dam_id"] = dam_id_text(out["dam_id"])
    out["station_id"] = out["station_id"].astype(str)
    out["usable"] = out["usable"].astype("boolean").fillna(False).astype(bool)
    out["side"] = link_side(out)
    out["in_limits"] = in_limits(out, max_down, max_up)
    reservoir = out["station_in_mapped_reservoir"].astype("boolean").fillna(False).astype(bool)
    out["is_control"] = out["side"].eq("upstream") & ~reservoir
    return out


def usable_coverage(links):
    """Per dam: usable links in the limits, downstream and as controls."""
    usable = links[links["usable"] & links["in_limits"]]
    return usable.groupby("dam_id").agg(
        usable_downstream_in_limits=("side", lambda s: int(s.eq("downstream").sum())),
        usable_controls_in_limits=("is_control", "sum"),
    ).astype(int)


# --------------------------------------------------
# Gains
# --------------------------------------------------

def link_gains(candidate_links, dams):
    """For each (dam, ambiguous station) link in the limits, what the station
    could give the dam (GAINS), or None. `dams` is indexed by dam_id with
    reason, has_assigned_downstream, has_upstream_control and
    usable_downstream_in_limits."""
    state = ["reason", "has_assigned_downstream", "has_upstream_control",
             "usable_downstream_in_limits"]
    out = candidate_links.drop(columns=state, errors="ignore").join(dams[state], on="dam_id")
    in_table = out["reason"].eq("in_analysis_table")
    assigned = out["has_assigned_downstream"].astype("boolean").fillna(False).astype(bool)
    control = out["has_upstream_control"].astype("boolean").fillna(False).astype(bool)
    no_down = out["usable_downstream_in_limits"].fillna(0).eq(0)
    downstream = out["side"].eq("downstream")
    out["gain"] = np.select(
        [in_table & assigned & ~control & out["is_control"],
         no_down & downstream,
         out["reason"].isin(NOT_IN_TABLE) & (downstream | out["is_control"]),
         in_table & (downstream | out["is_control"])],
        GAIN_ORDER, "")
    out["gain"] = out["gain"].astype(object).where(out["gain"].ne(""), None)
    return out


def best_gain(gains):
    """The highest-ranked gain in an iterable (GAIN_ORDER), or None."""
    ranked = [GAIN_ORDER.index(g) for g in gains if g in GAIN_ORDER]
    return GAIN_ORDER[min(ranked)] if ranked else None


def dam_table(gained):
    """One row per dam with any gain: flags and stations per gain."""
    rows = []
    for dam_id, group in gained.groupby("dam_id"):
        row = {"dam_id": dam_id, "best_gain": best_gain(group["gain"])}
        for gain in GAIN_ORDER:
            row[f"stations_{gain}"] = int(group.loc[group["gain"].eq(gain), "station_id"].nunique())
        rows.append(row)
    columns = ["dam_id", "best_gain"] + [f"stations_{g}" for g in GAIN_ORDER]
    return pd.DataFrame(rows, columns=columns)


def station_table(gained, stations, downloaded):
    """One row per ambiguous station with any gain, with a review rank (best
    gain, then more dams helped, then closer)."""
    by = gained.groupby("station_id")

    def dams_with(gain):
        hit = gained[gained["gain"].eq(gain)].groupby("station_id")["dam_id"].nunique()
        return hit.reindex(by.size().index, fill_value=0)

    out = pd.DataFrame({
        "best_gain": by["gain"].agg(best_gain),
        "dams_helped": by["dam_id"].nunique(),
        "dams_baci_control": dams_with("baci_control"),
        "dams_first_downstream": dams_with("first_downstream"),
        "dam_ids": by["dam_id"].agg(lambda s: ";".join(sorted(set(s)))),
        "sides": by["side"].agg(lambda s: ";".join(sorted(set(s.dropna())))),
        "nearest_km": by["network_distance_km"].min(),
    }).reset_index()
    keep = [c for c in STATION_COLUMNS if c in stations.columns]
    info = stations[keep].rename(columns={STATION_ID: "station_id"})
    info["station_id"] = info["station_id"].astype(str)
    out = out.merge(info.drop_duplicates("station_id"), on="station_id", how="left")
    out["already_downloaded"] = out["station_id"].isin(downloaded)
    out["_gain_rank"] = out["best_gain"].map({g: i for i, g in enumerate(GAIN_ORDER)})
    out = out.sort_values(["_gain_rank", "dams_helped", "nearest_km"],
                          ascending=[True, False, True]).drop(columns="_gain_rank")
    out.insert(0, "review_rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def summary_table(dams_out, stations_out):
    rows = []
    for gain, meaning in GAINS:
        rows.append({"gain": gain, "meaning": meaning,
                     "dams_best_gain": int(dams_out["best_gain"].eq(gain).sum()),
                     "dams_any": int(dams_out[f"stations_{gain}"].gt(0).sum()),
                     "stations_best_gain": int(stations_out["best_gain"].eq(gain).sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--threads", type=int)
    parser.add_argument("--memory-limit", help="DuckDB memory limit, e.g. 12GB.")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    window = cfg.get("wq_window", {}) or {}
    eligibility = cfg.get("eligibility", {}) or {}
    exposure = cfg.get("exposure", {}) or {}
    min_rule = (eligibility.get("min_samples", DEFAULT_MIN_SAMPLES),
                eligibility.get("min_years", DEFAULT_MIN_YEARS))
    max_down, max_up = exposure.get("max_downstream_km"), exposure.get("max_upstream_km")
    out_dir = cfg.project_root / "data" / "processed" / "stage1_audit" / cfg.name
    sub = "station_payoff"

    con, _ = connect(warehouse_dir(cfg), window.get("years_before", 5),
                     window.get("years_after", 5), *min_rule,
                     threads=args.threads, memory_limit=args.memory_limit,
                     temp_dir=out_dir / "_duckdb_tmp",
                     max_downstream_km=max_down, max_upstream_km=max_up)

    # The dams' current state: funnel reason, usable coverage, model needs.
    dams = dam_status(con).set_index("AR_ID")
    analysis = warehouse_dir(cfg) / "analysis_measurement.parquet"
    if not analysis.exists():
        raise FileNotFoundError(f"{analysis} is missing; run build_analysis_dataset first.")
    needs = con.sql(MODEL_NEEDS_SQL.format(path=analysis.resolve())).df()
    needs["AR_ID"] = dam_id_text(needs["AR_ID"])
    dams = dams.join(needs.set_index("AR_ID"))
    downloaded = set(con.sql("SELECT DISTINCT station_id FROM wq_station").df()["station_id"]
                     .astype(str))
    con.close()

    links = prepare_links(read_per_huc8(cfg, "reach_links", "reach_links", cfg.huc8s),
                          max_down, max_up)
    dams = dams.join(usable_coverage(links), how="left")
    dams[["usable_downstream_in_limits", "usable_controls_in_limits"]] = \
        dams[["usable_downstream_in_limits", "usable_controls_in_limits"]].fillna(0).astype(int)
    dams.index.name = "dam_id"
    dated = dams[dams["removal_year"].notna()]

    candidates = links[~links["usable"] & links["station_status"].eq("ambiguous")
                       & links["dam_status"].eq("accepted") & links["in_limits"]
                       & links["dam_id"].isin(dated.index)]
    gained = link_gains(candidates, dated)
    gained = gained[gained["gain"].notna()]

    stations = read_per_huc8(cfg, "station_reaches", "station_reaches", cfg.huc8s)
    stations_out = station_table(gained, stations, downloaded)
    dams_out = dam_table(gained).merge(
        dated[["dam_name", "state", "huc8", "removal_year", "reason", "has_assigned_downstream",
               "has_upstream_control", "usable_downstream_in_limits",
               "usable_controls_in_limits"]].reset_index(), on="dam_id", how="left")
    dams_out["_rank"] = dams_out["best_gain"].map({g: i for i, g in enumerate(GAIN_ORDER)})
    dams_out = dams_out.sort_values(["_rank", "dam_id"]).drop(columns="_rank")
    summary = summary_table(dams_out, stations_out)

    link_columns = ["dam_id", "station_id", "huc8", "side", "relationship", "network_distance_km",
                    "station_reason", "station_in_mapped_reservoir", "reason",
                    "has_assigned_downstream", "has_upstream_control",
                    "usable_downstream_in_limits", "gain"]
    outputs = [("station_payoff_links.csv", gained[[c for c in link_columns if c in gained]]),
               ("station_payoff_stations.csv", stations_out),
               ("station_payoff_dams.csv", dams_out),
               ("station_payoff_summary.csv", summary)]
    for name, frame in outputs:
        save_output(frame, f"{sub}/{name}", out_dir)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(f"\nCase study: {cfg.name}; limits {max_down} km down / {max_up} km up; "
          f"rule {min_rule}")
    print(f"Connected links: {len(links):,}; ambiguous-station links to accepted dated dams in "
          f"the limits: {len(candidates):,} ({candidates['station_id'].nunique():,} stations, "
          f"{candidates['dam_id'].nunique():,} dams); with a possible gain: {len(gained):,}")
    print("\n=== Possible gains (UPPER BOUNDS: these stations' samples were never downloaded) ===")
    print(summary.to_string(index=False))
    print("\n=== Stations by matching reason and best gain ===")
    print(pd.crosstab(stations_out["snap_reason"].fillna("?"),
                      pd.Categorical(stations_out["best_gain"], GAIN_ORDER), margins=True,
                      dropna=False).to_string())
    print(f"\nStations already in wq_station (some data downloaded for another link): "
          f"{int(stations_out['already_downloaded'].sum())} of {len(stations_out)}")
    print("\n=== Dams by current funnel reason and best gain ===")
    print(pd.crosstab(dams_out["reason"].fillna("?"),
                      pd.Categorical(dams_out["best_gain"], GAIN_ORDER), margins=True,
                      dropna=False).to_string())
    print("\n=== Top 30 stations to look at first ===")
    show = ["review_rank", "station_id", "MonitoringLocationName", "snap_reason", "best_gain",
            "dams_helped", "nearest_km", "dam_ids"]
    print(stations_out[[c for c in show if c in stations_out]].head(30).to_string(index=False))
    print(f"\nWrote {out_dir / sub}")


if __name__ == "__main__":
    main()
