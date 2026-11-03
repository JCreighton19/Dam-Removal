"""
Stage 1B: downstream station coverage by distance band (read-only; writes
only to data/processed/stage1_audit/<case_study>/).

Uses the existing warehouse Parquet files through analysis.audit_utils.connect
(an in-memory DuckDB; warehouse.duckdb is never opened) with NO link distance
limits, so removal_link, measurement_window and eligibility_pairs cover every
usable dam-station link at any distance -- the same views and window /
threshold rules build_views.py uses, only without the exposure cutoff.

Why that is enough without an ETL rerun: dam_station_link holds every usable
link at any distance (load_core_tables keeps them all; build_views applies
exposure.max_downstream_km later), and download_wqp_results.py requests every
station in a usable link. Two caveats, both reported:
  - robust_to_reach_choice links (confluence-ambiguous stations) are only
    usable inside the exposure window link_reaches.py ran with, so beyond the
    current cutoff they are missing; those counts are a lower bound.
  - link_reaches.py --max-km (Slurm MAX_KM) caps the network search; if it
    was set, links beyond it don't exist (see max_link_km below).

Distance bands (network km, downstream side): 00-05 = (0, 5], 05-10, 10-15,
15-20, 20-30, 30+ and distance_unknown. Upper edges are inclusive, as in
build_views.removal_link (distance <= max_downstream_km).

Outputs
-------
    coverage_by_band.csv     per series x band: dams / stations / pairs /
                             observations with any pre-or-post data, the same
                             for pairs meeting the eligibility thresholds on
                             both sides (has_both), dams whose has_both station
                             is the station's nearest linked upstream removal,
                             and has_both dams that also have an eligible
                             upstream control (within the config's
                             max_upstream_km, outside the rough former
                             reservoir)
    coverage_cumulative.csv  the same has_both counts for cutoffs 5, 10, 15,
                             20 and 30 km, and the gain over the current
                             cutoff
    link_download_check.csv  downstream links by band and whether their
                             station has any measurement in the dam's window
                             (shows whether data beyond the cutoff was
                             downloaded at all)
    coverage_scope.csv       table sizes and HUC8 coverage, to check the
                             warehouse is the national build

Run on the cluster through data/slurm/stage1_audits.slurm (AUDIT=coverage).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from analysis.audit_utils import connect, save_output, series_sql
from etl.build_views import DEFAULT_MIN_SAMPLES, DEFAULT_MIN_YEARS
from etl.common.config import default_case_study, load_config
from etl.common.warehouse import warehouse_dir

BAND_EDGES = [5, 10, 15, 20, 30]
CUTOFFS = [5, 10, 15, 20, 30]


def band_sql(column="network_distance_km", edges=BAND_EDGES):
    """CASE expression: '00-05', '05-10', ... '30+', 'distance_unknown'."""
    whens, low = [], 0
    for high in edges:
        whens.append(f"WHEN {column} <= {high} THEN '{low:02d}-{high:02d}'")
        low = high
    return (f"CASE WHEN {column} IS NULL THEN 'distance_unknown' {' '.join(whens)} "
            f"ELSE '{edges[-1]}+' END")


def coverage_sql(max_upstream_km):
    """Per downstream pair x series: band, obs, has_both, nearest-removal and
    upstream-control flags."""
    up_limit = "TRUE" if max_upstream_km is None else \
        f"u.network_distance_km <= {float(max_upstream_km)}"
    return f"""
    WITH backwater AS (
        SELECT AR_ID,
               Dam_Height_ft * 0.3048 / CASE WHEN reach_slope > 0 THEN reach_slope END / 1000
                   AS backwater_km
        FROM dam_removal
    ),
    nearest AS (
        SELECT AR_ID, station_id,
               row_number() OVER (PARTITION BY station_id
                                  ORDER BY network_distance_km NULLS LAST, AR_ID) = 1
                   AS is_nearest_removal
        FROM removal_link_all
        WHERE side = 'downstream'
    ),
    controls AS (
        SELECT DISTINCT u.AR_ID, {series_sql('u.parameter_std', 'u.sample_fraction_std')} AS series
        FROM eligibility_pairs u
        LEFT JOIN backwater b USING (AR_ID)
        WHERE u.side = 'upstream' AND u.has_both AND {up_limit}
          AND NOT coalesce(u.network_distance_km <= b.backwater_km, FALSE)
    )
    SELECT
        p.AR_ID, p.station_id, {series_sql('p.parameter_std', 'p.sample_fraction_std')} AS series,
        p.network_distance_km, {band_sql('p.network_distance_km')} AS band,
        p.n_pre + p.n_post AS obs, p.has_both,
        coalesce(n.is_nearest_removal, FALSE) AS is_nearest_removal,
        c.AR_ID IS NOT NULL AS has_upstream_control
    FROM eligibility_pairs p
    LEFT JOIN nearest n USING (AR_ID, station_id)
    LEFT JOIN controls c
           ON c.AR_ID = p.AR_ID
          AND c.series = {series_sql('p.parameter_std', 'p.sample_fraction_std')}
    WHERE p.side = 'downstream'
    """


def _summaries(pairs, keys):
    both = pairs[pairs["has_both"]]
    group = lambda frame: frame.groupby(keys, dropna=False)  # noqa: E731
    out = pd.concat({
        "dams_any": group(pairs)["AR_ID"].nunique(),
        "stations_any": group(pairs)["station_id"].nunique(),
        "pairs_any": group(pairs).size(),
        "obs_any": group(pairs)["obs"].sum(),
        "dams_has_both": group(both)["AR_ID"].nunique(),
        "stations_has_both": group(both)["station_id"].nunique(),
        "obs_has_both": group(both)["obs"].sum(),
        "dams_has_both_nearest": group(both[both["is_nearest_removal"]])["AR_ID"].nunique(),
        "dams_has_both_with_control":
            group(both[both["has_upstream_control"]])["AR_ID"].nunique(),
    }, axis=1)
    return out.fillna(0).astype("int64").reset_index()


def by_band(pairs):
    """coverage_by_band rows, plus an 'ALL' series per band."""
    per_series = _summaries(pairs, ["series", "band"])
    overall = _summaries(pairs.assign(series="ALL"), ["series", "band"])
    return pd.concat([overall, per_series], ignore_index=True)


def cumulative(pairs, current_cutoff, cutoffs=CUTOFFS):
    """has_both dam / station / observation counts within each cutoff, and
    the gain in dams over the current cutoff."""
    rows = []
    for cutoff in cutoffs:
        within = pairs[pairs["network_distance_km"] <= cutoff]
        summary = _summaries(within.assign(series="ALL"), ["series"])
        summary = pd.concat([summary, _summaries(within, ["series"])], ignore_index=True)
        rows.append(summary.assign(cutoff_km=cutoff))
    out = pd.concat(rows, ignore_index=True)
    if current_cutoff is not None:
        base = out[out["cutoff_km"] == current_cutoff].set_index("series")["dams_has_both"]
        out["dams_has_both_gain_vs_current"] = (out["dams_has_both"]
                                                - out["series"].map(base).fillna(0)).astype(int)
    columns = ["series", "cutoff_km"] + [c for c in out.columns if c not in ("series", "cutoff_km")]
    return out[columns].sort_values(["series", "cutoff_km"]).reset_index(drop=True)


LINK_CHECK_SQL = f"""
WITH links AS (
    SELECT AR_ID, station_id, network_distance_km, {band_sql('network_distance_km')} AS band
    FROM removal_link_all WHERE side = 'downstream'
),
in_window AS (
    SELECT DISTINCT AR_ID, station_id FROM measurement_window WHERE period <> 'outside'
),
any_data AS (SELECT DISTINCT station_id FROM wq_measurement)
SELECT band,
       count(*) AS links,
       count(DISTINCT l.station_id) AS stations,
       count(DISTINCT l.AR_ID) AS dams,
       count(w.AR_ID) AS links_with_window_data,
       count(DISTINCT l.station_id) FILTER (WHERE a.station_id IS NOT NULL)
           AS stations_with_any_data,
       max(network_distance_km) AS max_km
FROM links l
LEFT JOIN in_window w USING (AR_ID, station_id)
LEFT JOIN any_data a ON a.station_id = l.station_id
GROUP BY band ORDER BY band
"""


def scope(con, cfg):
    rows = {
        "case-study HUC8s (config)": len(cfg.huc8s),
        "dam_removal rows": con.sql("SELECT count(*) FROM dam_removal").fetchone()[0],
        "dam_removal with removal year":
            con.sql("SELECT count(*) FROM dam_removal WHERE Year_Removed IS NOT NULL").fetchone()[0],
        "dam_removal distinct HUC8": con.sql("SELECT count(DISTINCT HUC8) FROM dam_removal")
            .fetchone()[0],
        "wq_station rows": con.sql("SELECT count(*) FROM wq_station").fetchone()[0],
        "dam_station_link rows": con.sql("SELECT count(*) FROM dam_station_link").fetchone()[0],
        "max link km (any side)":
            con.sql("SELECT max(network_distance_km) FROM dam_station_link").fetchone()[0],
        "wq_measurement rows": con.sql("SELECT count(*) FROM wq_measurement").fetchone()[0],
        "last sample date": str(con.sql("SELECT max(sample_date) FROM wq_measurement")
                                .fetchone()[0]),
    }
    return pd.DataFrame({"item": list(rows), "value": [str(v) for v in rows.values()]})


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
    current_cutoff = exposure.get("max_downstream_km")
    max_upstream_km = exposure.get("max_upstream_km")
    out_dir = cfg.project_root / "data" / "processed" / "stage1_audit" / cfg.name

    con, _ = connect(warehouse_dir(cfg), window.get("years_before", 5),
                     window.get("years_after", 5),
                     eligibility.get("min_samples", DEFAULT_MIN_SAMPLES),
                     eligibility.get("min_years", DEFAULT_MIN_YEARS),
                     threads=args.threads, memory_limit=args.memory_limit,
                     temp_dir=out_dir / "_duckdb_tmp")
    pairs = con.sql(coverage_sql(max_upstream_km)).df()
    bands = by_band(pairs)
    cumul = cumulative(pairs, current_cutoff)
    links = con.sql(LINK_CHECK_SQL).df()
    sizes = scope(con, cfg)

    for name, frame in [("coverage_by_band.csv", bands), ("coverage_cumulative.csv", cumul),
                        ("link_download_check.csv", links), ("coverage_scope.csv", sizes)]:
        save_output(frame, name, out_dir)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 500)
    print(f"Case study: {cfg.name}; current downstream cutoff: {current_cutoff} km; upstream "
          f"controls within {max_upstream_km} km; window {window}; eligibility {eligibility}")
    print("\n=== Scope (check this is the national build) ===")
    print(sizes.to_string(index=False))
    print("\n=== Downstream links and downloaded data by band ===")
    print(links.to_string(index=False))
    print("\n=== Coverage by band, all series ===")
    print(bands[bands["series"] == "ALL"].to_string(index=False))
    print("\n=== Cumulative has_both coverage by cutoff ===")
    print(cumul.to_string(index=False))
    print("\n=== Coverage by band and series ===")
    print(bands[bands["series"] != "ALL"].to_string(index=False))
    print(f"\nNote: robust_to_reach_choice links beyond the cutoff link_reaches ran with are "
          f"not in dam_station_link, so counts beyond {current_cutoff} km are lower bounds.")
    print(f"Wrote coverage_by_band.csv, coverage_cumulative.csv, link_download_check.csv, "
          f"coverage_scope.csv to {out_dir}")


if __name__ == "__main__":
    main()
