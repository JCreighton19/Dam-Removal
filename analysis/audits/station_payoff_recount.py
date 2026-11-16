"""
Station payoff, step 2: download the high-value ambiguous stations' samples
and count how many dams they would really add (read-only for the pipeline;
writes only to data/processed/stage1_audit/<case_study>/station_payoff/ and
data/raw/wqp/results/<case_study>/station_payoff/).

Stages (run in order; data/slurm/station_payoff.slurm runs all three):

    list     from the payoff audit (station_payoff_audit.py), the stations
             whose best gain is in --gains (default baci_control and
             first_downstream), one row per station and dam it could help
             -> download_list.csv (station_id, year_removed, dam_id, gain)
    (download: etl/download_wqp_results.py --stations-file download_list.csv
             --output-subdir station_payoff; same windows, batches and
             manifest as the pipeline's download, in its own folder)
    recount  cleans the downloaded rows with clean_wq_results.clean_results
             (the same rules as wq_measurement) and counts, per dam x new
             station x measure, samples and distinct years before / after
             removal inside the window (removal year left out), with the
             config's minimum rule, as build_views.eligibility_pairs does.
             Then, per dam:
               baci_control      becomes a BACI dam if a new control qualifies
                                 for a measure the dam already has assigned
                                 downstream rows for in analysis_measurement
               first_downstream  becomes a before/after dam if a new
                                 downstream station qualifies for any measure
             -> recount_pairs.csv, recount_dams.csv, recount_stations.csv
                (the stations that pay off: the QGIS review list)

Still optimistic in two ways: a station's reach may be wrong (that's what the
review decides), and a new downstream station isn't checked against closer
competing removals (build_analysis_dataset's exposure rule), so a few new
before/after dams could be credited to another removal.

Run on the cluster: sbatch data/slurm/station_payoff.slurm
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import numpy as np
import pandas as pd

from analysis.audit_utils import save_output, series_sql
from etl.build_views import ASSIGNED_STATUSES, DEFAULT_MIN_SAMPLES, DEFAULT_MIN_YEARS
from etl.clean_wq_results import clean_results, downloaded_batch_files, load_raw_results
from etl.common.config import default_case_study, load_config
from etl.common.reach_inputs import dam_id_text
from etl.common.warehouse import warehouse_dir

DEFAULT_GAINS = ("baci_control", "first_downstream")
SUBDIR = "station_payoff"


def audit_dir(cfg):
    return cfg.project_root / "data" / "processed" / "stage1_audit" / cfg.name


# --------------------------------------------------
# list
# --------------------------------------------------

def download_list(links, dams, gains):
    """Rows of the payoff links whose gain is in `gains`, with the dam's
    removal year: station_id, year_removed, dam_id, side, gain."""
    keep = links[links["gain"].isin(gains)].copy()
    keep["dam_id"] = dam_id_text(keep["dam_id"])
    years = dams.assign(dam_id=dam_id_text(dams["dam_id"])).set_index("dam_id")["removal_year"]
    keep["year_removed"] = keep["dam_id"].map(years)
    keep = keep[keep["year_removed"].notna()]
    return keep[["station_id", "year_removed", "dam_id", "side", "gain"]].drop_duplicates()


# --------------------------------------------------
# recount
# --------------------------------------------------

def pair_counts(measurements, wanted, years_before, years_after):
    """Per (dam, station, series): samples, distinct years and distinct months
    before and after removal inside the window, the removal year left out.
    measurements: station_id, sample_date, series; wanted: dam_id,
    station_id, year_removed, side, gain."""
    m = measurements[["station_id", "sample_date", "series"]].copy()
    m["year"] = pd.to_datetime(m["sample_date"]).dt.year
    m["month"] = pd.to_datetime(m["sample_date"]).dt.month
    joined = wanted.merge(m, on="station_id")
    offset = joined["year"] - joined["year_removed"].astype(int)
    joined["period"] = np.select([offset.between(-years_before, -1), offset.between(1, years_after)],
                                 ["pre", "post"], "out")
    joined = joined[joined["period"].ne("out")]
    keys = ["dam_id", "station_id", "side", "gain", "series"]
    out = joined.groupby(keys + ["period"]).agg(n=("year", "size"), years=("year", "nunique"),
                                                months=("month", "nunique")).unstack("period")
    out.columns = [f"{stat}_{period}" for stat, period in out.columns]
    for column in ["n_pre", "n_post", "years_pre", "years_post", "months_pre", "months_post"]:
        if column not in out:
            out[column] = 0
    return out.fillna(0).astype(int).reset_index()


def qualifies(pairs, min_samples, min_years, min_months_single_year=1):
    """has_both, with build_views.eligibility_pairs' rule."""
    def side(n, years, months):
        return (pairs[n] >= min_samples) & (pairs[years] >= min_years) \
            & ((pairs[years] > 1) | (pairs[months] >= min_months_single_year))
    return side("n_pre", "years_pre", "months_pre") & side("n_post", "years_post", "months_post")


def dam_outcomes(pairs, wanted, downstream_series):
    """Per dam in `wanted`: whether it becomes BACI (baci_control) or
    before/after (first_downstream), and the measures that do it.
    downstream_series: dam_id -> set of measures with assigned downstream
    rows in analysis_measurement."""
    good = pairs[pairs["has_both"]]
    rows = []
    for (dam_id, gain), group in wanted.groupby(["dam_id", "gain"]):
        mine = good[(good["dam_id"] == dam_id) & (good["gain"] == gain)]
        if gain == "baci_control":
            hits = mine[mine["side"].eq("upstream")
                        & mine["series"].isin(downstream_series.get(dam_id, set()))]
        else:
            hits = mine[mine["side"].eq("downstream")]
        rows.append({"dam_id": dam_id, "gain": gain, "becomes": len(hits) > 0,
                     "measures": ";".join(sorted(set(hits["series"]))),
                     "stations_that_qualify": ";".join(sorted(set(hits["station_id"]))),
                     "stations_tried": group["station_id"].nunique(),
                     "stations_with_any_data": int(pairs.loc[pairs["dam_id"].eq(dam_id),
                                                             "station_id"].nunique())})
    return pd.DataFrame(rows, columns=["dam_id", "gain", "becomes", "measures",
                                       "stations_that_qualify", "stations_tried",
                                       "stations_with_any_data"])


def paying_stations(outcomes, stations):
    """The stations that make a dam become BACI / before/after: the QGIS
    review list, with the payoff audit's matching details."""
    rows = [(s, o.dam_id, o.gain) for o in outcomes[outcomes["becomes"]].itertuples()
            for s in o.stations_that_qualify.split(";") if s]
    pay = pd.DataFrame(rows, columns=["station_id", "dam_id", "gain"])
    if pay.empty:
        return pay
    by = pay.groupby("station_id").agg(
        dams=("dam_id", lambda s: ";".join(sorted(set(s)))),
        new_baci_dams=("gain", lambda g: int((g == "baci_control").sum())),
        new_before_after_dams=("gain", lambda g: int((g == "first_downstream").sum()))).reset_index()
    info = stations.drop(columns=["dam_ids", "best_gain", "dams_helped"], errors="ignore")
    return by.merge(info, on="station_id", how="left").sort_values(
        ["new_baci_dams", "new_before_after_dams"], ascending=False).reset_index(drop=True)


def downstream_measures(analysis_path):
    """dam_id -> measures with assigned downstream rows in analysis_measurement."""
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT DISTINCT AR_ID, {series_sql()} AS series
        FROM read_parquet('{Path(analysis_path).resolve()}')
        WHERE side = 'downstream' AND exposure_status IN {tuple(ASSIGNED_STATUSES)}""").df()
    con.close()
    frame["AR_ID"] = dam_id_text(frame["AR_ID"])
    return frame.groupby("AR_ID")["series"].agg(set).to_dict()


def with_series(table):
    p, f = table["parameter_std"], table["sample_fraction_std"]
    return table.assign(series=np.where(f.isna(), p, p.astype(str) + " (" + f.astype(str) + ")"))


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--stage", choices=["list", "recount"], required=True)
    parser.add_argument("--gains", nargs="+", default=list(DEFAULT_GAINS),
                        help="Payoff-audit gains to download (default: %(default)s).")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    out_dir = audit_dir(cfg)
    folder = out_dir / SUBDIR
    window = cfg.get("wq_window", {}) or {}
    eligibility = cfg.get("eligibility", {}) or {}
    years_before, years_after = int(window.get("years_before", 5)), int(window.get("years_after", 5))
    min_samples = int(eligibility.get("min_samples", DEFAULT_MIN_SAMPLES))
    min_years = int(eligibility.get("min_years", DEFAULT_MIN_YEARS))
    min_months = int(eligibility.get("min_months_single_year", 1) or 1)

    if args.stage == "list":
        links = pd.read_csv(folder / "station_payoff_links.csv", dtype={"station_id": str})
        dams = pd.read_csv(folder / "station_payoff_dams.csv", dtype={"dam_id": str})
        wanted = download_list(links, dams, args.gains)
        save_output(wanted, f"{SUBDIR}/download_list.csv", out_dir)
        print(f"Download list: {wanted['station_id'].nunique():,} stations for "
              f"{wanted['dam_id'].nunique():,} dams (gains {args.gains})")
        print(wanted.groupby("gain").agg(stations=("station_id", "nunique"),
                                         dams=("dam_id", "nunique")).to_string())
        return

    wanted = pd.read_csv(folder / "download_list.csv", dtype={"station_id": str, "dam_id": str})
    files = downloaded_batch_files(cfg.wqp_results_dir / SUBDIR)
    raw = load_raw_results(files)
    print(f"Read {len(raw):,} downloaded rows from {len(files):,} batch file(s)")
    table = with_series(clean_results(raw, set(wanted["station_id"])))
    print(f"Cleaned rows: {len(table):,} at {table['station_id'].nunique():,} stations")

    pairs = pair_counts(table, wanted, years_before, years_after)
    pairs["has_both"] = qualifies(pairs, min_samples, min_years, min_months)
    analysis = warehouse_dir(cfg) / "analysis_measurement.parquet"
    outcomes = dam_outcomes(pairs, wanted, downstream_measures(analysis))
    stations = pd.read_csv(folder / "station_payoff_stations.csv", dtype={"station_id": str})
    pay = paying_stations(outcomes, stations)

    for name, frame in [("recount_pairs.csv", pairs), ("recount_dams.csv", outcomes),
                        ("recount_stations.csv", pay)]:
        save_output(frame, f"{SUBDIR}/{name}", out_dir)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 200)
    print(f"\nRule: {min_samples} samples in {min_years} years per side; window "
          f"-{years_before}/+{years_after} years")
    print("\n=== Dams that would really be added (if the stations' reaches are confirmed) ===")
    summary = outcomes.groupby("gain").agg(dams_tried=("dam_id", "size"),
                                           dams_with_any_data=("stations_with_any_data",
                                                               lambda s: int((s > 0).sum())),
                                           dams_added=("becomes", "sum"))
    print(summary.to_string())
    print("\n=== New BACI dams by measure ===")
    baci = outcomes[outcomes["becomes"] & outcomes["gain"].eq("baci_control")]
    print(baci["measures"].str.split(";").explode().value_counts().to_string()
          if len(baci) else "  none")
    print(f"\n=== Stations to review in QGIS: {len(pay)} ===")
    show = ["station_id", "MonitoringLocationName", "snap_reason", "new_baci_dams",
            "new_before_after_dams", "dams"]
    print(pay[[c for c in show if c in pay]].to_string(index=False) if len(pay) else "  none")
    print(f"\nWrote {folder}")


if __name__ == "__main__":
    main()
