"""
Accepted-station accuracy review: export every station used in the analysis
(analysis_measurement) with its matching decision, alternatives and analysis
roles, as one GeoPackage for QGIS, plus a risk tier and a stratified random
sample of the apparently strong assignments (read-only; writes only to
data/processed/station_review/<case_study>/).

This is the frozen baseline: it reads the current station_reaches,
station_candidates and analysis_measurement outputs and changes nothing.
It is separate from the recovery audits (station_payoff_audit.py,
station_recovery_audit.py), which look at stations NOT used.

Risk tier (from the matching evidence; transparent rules, not a model):

    manual  a manual override (config/snap_overrides.csv)
    high    a conflict the rules let through on name support alone
            (confluence_nearby, polygon_edge, far_snap, sink_catchment), a
            name mismatch, or an eligible alternative within TIE_M with a
            different name or very different drainage area, in each case
            WITHOUT drainage-area agreement; or snapped > HIGH_SNAP_M from a
            point outside any water polygon
    medium  any of those flags when drainage area agrees (it is the evidence
            that resolves a >= 5x size conflict), used although ambiguous
            (robust_to_reach_choice: either reach gives the same role), no
            drainage-area or name support, snapped > MEDIUM_SNAP_M, or an
            alternative within TIE_M
    low     no flag (drainage area or name agrees, close snap, no close rival)

(Revised 2026-11-16 before any review: the first version counted every
flagged conflict as high, which put 80% of the stations in it.)

Inspection set: every BACI station (treated or control for a dam x measure
that has both), every manual and high-risk station. The rest (medium and low
tier, not BACI) is sampled at random within strata (tier x role x feature
type x drainage-area class), at least MIN_PER_STRATUM or the tier's
fraction (SAMPLE_FRACTIONS) of each, with a fixed seed; sample_weight =
stratum size / sampled, so the errors found in the sample estimate the
undetected-error rate of each tier (docs/station_review.md).

Layers (EPSG:4326):

    stations          reported location (as given), all attributes, risk
                      tier, inspect / sample flags, and empty review fields
                      (verdict, correct_nhdplus_id, review_note)
    stations_snapped  where the station sits on its reach
    snap_lines        reported -> snapped
    chosen_reaches    the selected reach of each station
    flowlines         every flowline within --radius-km of a station (the
                      alternatives are among them; ids in alt*_id)
    dams              the analysis dams on their reach, with AR_ID and name

Also writes station_review.csv (the stations layer without geometry) and
sample_design.csv (strata, sizes, sampled, weights).

Run on the cluster (Slurm only): sbatch data/slurm/export_station_review.slurm
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import LineString

from analysis.audit_utils import series_sql
from analysis.audits.export_dam_review import METRIC_CRS, WGS84, flowlines_near
from analysis.mixed_model import SETTINGS as MODEL_SETTINGS
from etl.common.config import default_case_study, load_config
from etl.common.reach_inputs import STATION_ID, dam_id_text, read_per_huc8
from etl.common.warehouse import warehouse_dir

RADIUS_KM = 1.0
HIGH_SNAP_M = 100.0
MEDIUM_SNAP_M = 50.0
TIE_M = 25.0
AREA_RATIO = 5.0                      # as reach_selection's confluence_area_ratio
ELIGIBLE_FTYPES = {460, 558}          # StreamRiver, ArtificialPath
CONFLICT_CODES = {"confluence_nearby", "polygon_edge", "far_snap", "sink_catchment"}
N_ALTERNATIVES = 3
SAMPLE_FRACTIONS = {"medium": 0.25, "low": 0.15}
MIN_PER_STRATUM = 3
SEED = 20261116
TIERS = ["manual", "high", "medium", "low"]
VERDICTS = "correct | wrong_reach | wrong_side | not_a_stream | unsure"


# --------------------------------------------------
# Analysis roles
# --------------------------------------------------

def station_roles(path, settings=MODEL_SETTINGS):
    """One row per analysis station: dams, measures, rows, treated / control
    flags and whether it is a BACI station (treated or control for a dam x
    measure that has both). Roles as analysis/mixed_model.assign_roles, without
    the distance and flag restrictions."""
    treated = ", ".join(f"'{s}'" for s in settings["treated_exposure"])
    control = ", ".join(f"'{s}'" for s in settings["control_exposure"])
    return duckdb.sql(f"""
        WITH rows AS (
            SELECT CAST(station_id AS VARCHAR) AS station_id, CAST(AR_ID AS VARCHAR) AS AR_ID,
                   {series_sql()} AS series, network_distance_km,
                   side = 'downstream' AND exposure_status IN ({treated}) AS is_treated,
                   side = 'upstream' AND exposure_status IN ({control})
                       AND NOT coalesce(upstream_in_former_reservoir, false) AS is_control
            FROM read_parquet('{Path(path).as_posix()}')),
        dam_series AS (
            SELECT AR_ID, series, bool_or(is_treated) AND bool_or(is_control) AS baci
            FROM rows GROUP BY ALL)
        SELECT r.station_id,
               count(*) AS analysis_rows,
               count(DISTINCT r.AR_ID) AS n_dams,
               string_agg(DISTINCT r.AR_ID, ';' ORDER BY r.AR_ID) AS dam_ids,
               string_agg(DISTINCT r.series, ';' ORDER BY r.series) AS measures,
               bool_or(r.is_treated) AS treated_any,
               bool_or(r.is_control) AS control_any,
               bool_or((r.is_treated OR r.is_control) AND d.baci) AS baci_station,
               min(r.network_distance_km) AS nearest_dam_km
        FROM rows r JOIN dam_series d USING (AR_ID, series)
        GROUP BY r.station_id
    """).df()


def role_label(frame):
    treated, control = frame["treated_any"].fillna(False), frame["control_any"].fillna(False)
    return pd.Series(np.select([treated & control, treated, control], ["both", "treated", "control"],
                               "other"), index=frame.index)


# --------------------------------------------------
# Alternatives and risk
# --------------------------------------------------

def alternatives(candidates, chosen):
    """Per station: the chosen candidate's evidence and the N_ALTERNATIVES
    nearest other eligible candidates. candidates: station_candidates rows;
    chosen: station_id -> chosen NHDPlusID."""
    c = candidates.rename(columns={STATION_ID: "station_id"}).copy()
    c["station_id"] = c["station_id"].astype(str)
    c["NHDPlusID"] = pd.to_numeric(c["NHDPlusID"], errors="coerce").astype("Int64")
    c["is_chosen"] = c["NHDPlusID"].eq(c["station_id"].map(chosen).astype("Int64")).fillna(False)
    eligible = pd.to_numeric(c["FType"], errors="coerce").isin(ELIGIBLE_FTYPES)
    mine = c[c["is_chosen"]].drop_duplicates("station_id").set_index("station_id")
    out = pd.DataFrame(index=pd.Index(sorted(set(c["station_id"])), name="station_id"))
    out["chosen_distance_m"] = mine["distance_m"]
    out["chosen_area_ratio"] = mine["area_ratio"]
    out["chosen_name_match"] = mine["name_match"]
    out["chosen_area_km2"] = mine["totdasqkm"]
    rivals = c[eligible & ~c["is_chosen"]].sort_values(["station_id", "distance_m"])
    out["n_eligible_alternatives"] = rivals.groupby("station_id").size()
    for k in range(N_ALTERNATIVES):
        nth = rivals.groupby("station_id").nth(k).set_index("station_id")
        out[f"alt{k + 1}_id"] = nth["NHDPlusID"]
        out[f"alt{k + 1}_distance_m"] = nth["distance_m"]
        out[f"alt{k + 1}_ftype"] = pd.to_numeric(nth["FType"], errors="coerce")
        out[f"alt{k + 1}_name"] = nth["GNIS_Name"]
        out[f"alt{k + 1}_area_km2"] = nth["totdasqkm"]
    out["n_eligible_alternatives"] = out["n_eligible_alternatives"].fillna(0).astype(int)
    out["margin_m"] = out["alt1_distance_m"] - out["chosen_distance_m"]
    return out.reset_index()


def support(frame):
    """Evidence behind the choice: area (drainage area agrees), name, both, none."""
    evidence = frame["snap_evidence"].fillna("").str.split(";").apply(set)
    area = evidence.apply(lambda e: "area_agrees" in e)
    name = evidence.apply(lambda e: "name_match" in e)
    return pd.Series(np.select([area & name, area, name], ["area+name", "area", "name"], "none"),
                     index=frame.index)


def risk(frame):
    """risk_tier and risk_flags per station (see the module docstring).

    Drainage-area agreement is the evidence that resolves the conflicts the
    matching flags (a confluence rival is >= 5x larger or smaller), so a flag
    backed by it is medium, not high."""
    evidence = frame["snap_evidence"].fillna("").str.split(";").apply(set)
    supported = evidence.apply(lambda e: bool({"area_agrees", "name_match"} & e))
    area_ok = evidence.apply(lambda e: "area_agrees" in e)
    in_polygon = frame.get("point_polygon_layer", pd.Series(index=frame.index, dtype=object)).notna()
    snap = pd.to_numeric(frame["snap_distance_m"], errors="coerce")
    margin = pd.to_numeric(frame["margin_m"], errors="coerce")
    close_alt = margin.le(TIE_M).fillna(False)
    area = pd.to_numeric(frame["chosen_area_km2"], errors="coerce")
    alt_area = pd.to_numeric(frame["alt1_area_km2"], errors="coerce")
    size = (alt_area / area).where(area > 0)
    different = (frame["alt1_name"].fillna("").ne(frame["reach_gnis_name"].fillna(""))
                 | size.ge(AREA_RATIO) | size.le(1 / AREA_RATIO))
    flags = {
        "manual_override": frame["match_method"].eq("manual_override"),
        "used_while_ambiguous": frame["snap_status"].ne("accepted"),
        "conflict_let_through": evidence.apply(lambda e: bool(CONFLICT_CODES & e)),
        "name_mismatch": evidence.apply(lambda e: "name_mismatch" in e),
        "far_snap": snap.gt(HIGH_SNAP_M).fillna(False),
        "close_different_rival": close_alt & different.fillna(True),
        "no_area_or_name_support": ~supported,
        "snap_over_medium": snap.gt(MEDIUM_SNAP_M).fillna(False),
        "close_rival": close_alt,
    }
    flags = pd.DataFrame(flags, index=frame.index)
    # High: a conflict or rival the drainage area doesn't resolve, or a long snap
    # from a point that isn't inside a water polygon.
    high = ((flags["conflict_let_through"] | flags["name_mismatch"]
             | flags["close_different_rival"]) & ~area_ok) \
        | (flags["far_snap"] & ~in_polygon)
    medium = flags[["used_while_ambiguous", "conflict_let_through", "name_mismatch", "far_snap",
                    "close_different_rival", "no_area_or_name_support", "snap_over_medium",
                    "close_rival"]].any(axis=1)
    tier = np.select([flags["manual_override"], high, medium], TIERS[:3], "low")
    text = flags.apply(lambda row: ";".join(c for c, v in row.items() if v), axis=1)
    return pd.Series(tier, index=frame.index), text


def size_class(area):
    area = pd.to_numeric(area, errors="coerce")
    return pd.Series(np.select([area < 100, area < 1000, area >= 1000],
                               ["<100 km2", "100-1000 km2", ">=1000 km2"], "unknown"),
                     index=area.index)


def stratified_sample(frame, fractions=None, minimum=MIN_PER_STRATUM, seed=SEED):
    """(sampled flag, weight, stratum, design table) for the non-BACI stations
    of the sampled tiers (`fractions`: tier -> fraction): within each stratum
    max(minimum, ceil(fraction x size)), capped at the size; weight = size /
    sampled."""
    fractions = SAMPLE_FRACTIONS if fractions is None else fractions
    pool = frame["risk_tier"].isin(list(fractions)) & ~frame["baci_station"].fillna(False).astype(bool)
    ftype = pd.to_numeric(frame["reach_ftype"], errors="coerce").astype("Int64").astype("string")
    stratum = (frame["risk_tier"] + " | " + frame["role"] + " | ftype " + ftype.fillna("?")
               + " | " + size_class(frame["reach_totdasqkm"]))
    sampled = pd.Series(False, index=frame.index)
    weight = pd.Series(np.nan, index=frame.index)
    rows, rng = [], np.random.default_rng(seed)
    for name, members in stratum[pool].groupby(stratum[pool]):
        size = len(members)
        fraction = fractions[frame.loc[members.index[0], "risk_tier"]]
        take = min(size, max(minimum, math.ceil(fraction * size)))
        chosen = rng.choice(members.index.to_numpy(), size=take, replace=False)
        sampled[chosen] = True
        weight[chosen] = size / take
        rows.append({"tier": frame.loc[members.index[0], "risk_tier"], "stratum": name,
                     "stations": size, "sampled": take, "weight": size / take})
    design = pd.DataFrame(rows, columns=["tier", "stratum", "stations", "sampled", "weight"])
    return sampled, weight, stratum.where(pool), design


# --------------------------------------------------
# Output
# --------------------------------------------------

def _points(frame, lon, lat, crs):
    x, y = pd.to_numeric(frame[lon], errors="coerce"), pd.to_numeric(frame[lat], errors="coerce")
    keep = x.notna() & y.notna()
    return gpd.GeoDataFrame(frame[keep].reset_index(drop=True),
                            geometry=gpd.points_from_xy(x[keep], y[keep]), crs=crs).to_crs(WGS84)


def write_review(stations, flowlines, dams, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.gpkg")
    tmp.unlink(missing_ok=True)
    table = stations.copy()
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].astype("string")
    table["verdict"] = pd.Series("", index=table.index, dtype="string")
    table["correct_nhdplus_id"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
    table["review_note"] = pd.Series("", index=table.index, dtype="string")
    reported = _points(table, "LongitudeMeasure", "LatitudeMeasure", WGS84)
    pyogrio.write_dataframe(reported, tmp, layer="stations", driver="GPKG")
    snapped = _points(table[["station_id", "risk_tier", "role", "NHDPlusID", "snapped_lon",
                             "snapped_lat"]], "snapped_lon", "snapped_lat", "EPSG:4269")
    pyogrio.write_dataframe(snapped, tmp, layer="stations_snapped", driver="GPKG")
    both = reported[["station_id", "risk_tier", "snap_distance_m", "geometry"]].merge(
        snapped[["station_id", "geometry"]].rename(columns={"geometry": "snapped"}), on="station_id")
    lines = gpd.GeoDataFrame(both.drop(columns=["geometry", "snapped"]),
                             geometry=[LineString([a, b]) for a, b in
                                       zip(both["geometry"], both["snapped"])], crs=WGS84)
    pyogrio.write_dataframe(lines, tmp, layer="snap_lines", driver="GPKG")
    chosen = flowlines[flowlines["nhdplus_id"].isin(pd.to_numeric(stations["NHDPlusID"],
                                                                  errors="coerce").dropna())]
    for name, layer in (("chosen_reaches", chosen), ("flowlines", flowlines)):
        pyogrio.write_dataframe(layer, tmp, layer=name, driver="GPKG",
                                geometry_type="MultiLineString", promote_to_multi=True)
    pyogrio.write_dataframe(dams, tmp, layer="dams", driver="GPKG")
    tmp.replace(path)


def analysis_dams(cfg, dam_ids):
    reaches = read_per_huc8(cfg, "dam_reaches", "dam_reaches", cfg.huc8s)
    reaches["AR_ID"] = dam_id_text(reaches["AR_ID"])
    keep = reaches[reaches["AR_ID"].isin(dam_ids) & reaches["match_status"].eq("accepted")]
    keep = keep.drop_duplicates("AR_ID")[["AR_ID", "Dam_Name", "Year_Removed", "NHDPlusID",
                                          "snapped_lon", "snapped_lat"]]
    keep["NHDPlusID"] = pd.to_numeric(keep["NHDPlusID"], errors="coerce").astype("Int64")
    return _points(keep, "snapped_lon", "snapped_lat", "EPSG:4269")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM)
    parser.add_argument("--medium-fraction", type=float, default=SAMPLE_FRACTIONS["medium"])
    parser.add_argument("--low-fraction", type=float, default=SAMPLE_FRACTIONS["low"])
    parser.add_argument("--min-per-stratum", type=int, default=MIN_PER_STRATUM)
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    out_dir = cfg.project_root / "data" / "processed" / "station_review" / cfg.name
    roles = station_roles(warehouse_dir(cfg) / "analysis_measurement.parquet")
    roles["station_id"] = roles["station_id"].astype(str)
    print(f"Analysis stations: {len(roles):,}", flush=True)

    reaches = read_per_huc8(cfg, "station_reaches", "station_reaches", cfg.huc8s)
    reaches = reaches.rename(columns={STATION_ID: "station_id"})
    reaches["station_id"] = reaches["station_id"].astype(str)
    reaches = reaches[reaches["station_id"].isin(roles["station_id"])].drop_duplicates("station_id")
    reaches["NHDPlusID"] = pd.to_numeric(reaches["NHDPlusID"], errors="coerce").astype("Int64")
    # Ambiguous stations used through robust_to_reach_choice have only a provisional reach.
    provisional = pd.to_numeric(reaches.get("provisional_nhdplus_id"), errors="coerce").astype("Int64")
    reaches["NHDPlusID"] = reaches["NHDPlusID"].fillna(provisional)
    missing = sorted(set(roles["station_id"]) - set(reaches["station_id"]))
    if missing:
        print(f"WARNING: {len(missing)} analysis station(s) without a station_reaches row, "
              f"e.g. {missing[:5]}")

    huc8s = sorted(set(reaches["huc8"].astype(str).str.zfill(8)))
    candidates = read_per_huc8(cfg, "station_candidates", "station_candidates", huc8s)
    candidates = candidates[candidates[STATION_ID].astype(str).isin(roles["station_id"])]
    alts = alternatives(candidates, reaches.set_index("station_id")["NHDPlusID"])

    keep = [c for c in ["station_id", "MonitoringLocationName", "huc8", "LatitudeMeasure",
                        "LongitudeMeasure", "snapped_lon", "snapped_lat", "snap_status",
                        "snap_reason", "snap_evidence", "match_method", "auto_status",
                        "auto_reason", "NHDPlusID", "reach_gnis_name", "reach_ftype",
                        "reach_totdasqkm", "reach_area_ratio", "reach_generation_methods",
                        "reach_path_polygon_layer", "reach_catchment_relation",
                        "snap_distance_m", "n_eligible", "n_plausible", "runner_up_id",
                        "runner_up_distance_m", "runner_up_name", "point_polygon_layer",
                        "in_sink_catchment"] if c in reaches]
    stations = roles.merge(reaches[keep], on="station_id", how="left") \
        .merge(alts, on="station_id", how="left")
    stations["role"] = role_label(stations)
    stations["support"] = support(stations)
    stations["risk_tier"], stations["risk_flags"] = risk(stations)
    sampled, weight, stratum, design = stratified_sample(
        stations, {"medium": args.medium_fraction, "low": args.low_fraction}, args.min_per_stratum)
    stations["in_sample"], stations["sample_weight"], stations["stratum"] = sampled, weight, stratum
    baci = stations["baci_station"].fillna(False).astype(bool)
    stations["inspect_reason"] = np.select(
        [baci, stations["risk_tier"].isin(["manual", "high"]), sampled],
        ["baci", "risk_" + stations["risk_tier"], "random_sample"], "")
    order = {"baci": 0, "risk_manual": 1, "risk_high": 2, "random_sample": 3, "": 4}
    stations = stations.sort_values(["inspect_reason", "station_id"],
                                    key=lambda s: s.map(order) if s.name == "inspect_reason" else s)
    stations.insert(1, "review_order", np.where(stations["inspect_reason"].ne(""),
                                                np.arange(1, len(stations) + 1), 0))

    print("\nReading flowlines...", flush=True)
    rings = _points(stations[["station_id", "LongitudeMeasure", "LatitudeMeasure"]],
                    "LongitudeMeasure", "LatitudeMeasure", WGS84).to_crs(METRIC_CRS)
    rings = gpd.GeoDataFrame({"AR_ID": rings["station_id"]},
                             geometry=rings.buffer(args.radius_km * 1000), crs=METRIC_CRS)
    flowlines, notes = flowlines_near(rings, cfg.path("wbd_source"),
                                      cfg.project_root / "data" / "processed" / "nhdplus_extracted")
    for note in notes:
        print(f"  NOTE: {note}")
    dam_ids = set(dam_id_text(pd.Series(";".join(stations["dam_ids"].dropna()).split(";"))))
    dams = analysis_dams(cfg, dam_ids)

    output = out_dir / "station_review.gpkg"
    write_review(stations, flowlines, dams, output)
    stations.to_csv(out_dir / "station_review.csv", index=False)
    design.to_csv(out_dir / "sample_design.csv", index=False)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 100)
    print(f"\n=== Risk tier x role ({len(stations)} stations) ===")
    print(pd.crosstab(stations["risk_tier"], stations["role"], margins=True).to_string())
    print("\n=== Risk flags (a station can have several) ===")
    print(stations["risk_flags"].str.split(";").explode().replace("", np.nan).dropna()
          .value_counts().to_string())
    print("\n=== Risk flag x evidence support (stations) ===")
    flag_rows = stations[["support", "risk_flags"]].assign(
        flag=stations["risk_flags"].str.split(";")).explode("flag")
    flag_rows = flag_rows[flag_rows["flag"].fillna("").ne("")]
    print(pd.crosstab(flag_rows["flag"], flag_rows["support"], margins=True).to_string())
    print("\n=== Evidence support x tier ===")
    print(pd.crosstab(stations["support"], stations["risk_tier"], margins=True).to_string())
    print("\n=== Matching reason x tier ===")
    print(pd.crosstab(stations["snap_reason"].fillna("?"), stations["risk_tier"]).to_string())
    print("\n=== Inspection set ===")
    print(stations["inspect_reason"].replace("", "not inspected").value_counts().to_string())
    print(f"\nBACI stations: {int(baci.sum())}; of them high/medium/manual risk: "
          f"{int((baci & stations['risk_tier'].ne('low')).sum())}")
    print("\n=== Random sample (medium and low tier, not BACI) ===")
    print(design.groupby("tier")[["stations", "sampled"]].sum().to_string())
    print(design.to_string(index=False))
    print(f"\n{len(stations):,} stations, {len(flowlines):,} flowlines, {len(dams):,} dams "
          f"-> {output} ({output.stat().st_size / 1e6:,.1f} MB)")


if __name__ == "__main__":
    main()
