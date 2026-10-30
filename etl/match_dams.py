"""
Stage 4 of the reach-matching rework: match every dam removal to its
NHDPlus HR flowlines (rules in etl/common/dam_matching.py), apply manual
overrides (config/snap_overrides.csv, feature_kind dam), and place
accepted dams on their structure flowline by projecting the reported
point onto it. Reported coordinates are kept as given (the dam list has no
datum; they are taken as WGS84 and flagged datum_assumed, as in
snap_features.py).

Outputs in --output-dir (default data/processed/dam_reaches/<case_study>/):

    dam_reaches_<huc8>.parquet      one row per dam: match_status (accepted /
        ambiguous / rejected), match_reason, match_evidence, auto_* before
        overrides, match_method; the structure flowline (NHDPlusID, accepted
        only; provisional_nhdplus_id for review), snap_distance_m,
        snap_fraction, snapped coordinates; reservoir_status and the
        reservoir, impounded paths, outlets and downstream flowlines
    dam_reservoir_outlets_<huc8>.parquet  every outlet of every reservoir
        polygon near a dam: outlet path, flowlines below, drainage area,
        main / diversion, distance from the dam
    dam_candidates_<huc8>.parquet   every candidate reach of every dam
    summary.csv                     dams per HUC8 by status, reason and
                                    reservoir_status

The current snapping outputs are not touched.
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd

from etl.common import dam_matching as dm
from etl.common import reach_selection as rs
from etl.common import snapping as sn
from etl.common.config import default_case_study, load_config
from etl.common.dams import filter_to_sampled_dams, load_dams
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.wbd import clip_to_watersheds, load_huc8_watershed
from etl.select_station_reaches import apply_overrides
from etl.snap_features import (
    BBOX_MARGIN_DEG,
    DAM_DATUM_EPSG,
    DAM_ID,
    DAM_RIVER,
    NHDPLUS_CRS,
    OVERRIDES_FILE,
    locate_points,
    near_bounds,
)
from etl.station_candidates import load_package

KEEP = [DAM_ID, "Dam_Name", DAM_RIVER, "Year_Removed", "Latitude", "Longitude",
        "datum_epsg", "datum_assumed"]


def place(points, decisions, flowlines):
    """snap_fraction, snap_distance_m and snapped coordinates on the
    structure flowline of every accepted dam."""
    out = pd.DataFrame(index=points.index)
    reach = decisions["override_nhdplus_id"].astype("Int64").fillna(
        decisions["provisional_nhdplus_id"].astype("Int64"))
    out["NHDPlusID"] = reach.where(decisions["status"] == rs.ACCEPTED).astype("Int64")
    for column in ["snap_distance_m", "snap_fraction", "snapped_x_m", "snapped_y_m",
                   "snapped_lon", "snapped_lat"]:
        out[column] = np.nan
    out["reach_gnis_name"], out["reach_ftype"], out["reach_totdasqkm"] = None, pd.NA, np.nan
    chosen = out["NHDPlusID"].dropna()
    if chosen.empty:
        return out
    by_id = flowlines.set_index("NHDPlusID")
    missing = sorted(set(chosen) - set(by_id.index))
    if missing:
        raise ValueError(f"Structure / override reach(es) not near the dams: {missing}")
    lines = by_id.geometry.loc[chosen.to_numpy()].to_numpy()
    fractions, snapped, distances = rs.project_onto(lines, points.geometry.loc[chosen.index].to_numpy())
    snapped = gpd.GeoSeries(snapped, crs=points.crs)
    geographic = snapped.to_crs(NHDPLUS_CRS)
    out.loc[chosen.index, "snap_fraction"] = fractions
    out.loc[chosen.index, "snap_distance_m"] = distances
    out.loc[chosen.index, "snapped_x_m"] = snapped.x.to_numpy()
    out.loc[chosen.index, "snapped_y_m"] = snapped.y.to_numpy()
    out.loc[chosen.index, "snapped_lon"] = geographic.x.to_numpy()
    out.loc[chosen.index, "snapped_lat"] = geographic.y.to_numpy()
    out.loc[chosen.index, "reach_gnis_name"] = by_id.loc[chosen.to_numpy(), "GNIS_Name"].to_numpy()
    out.loc[chosen.index, "reach_ftype"] = by_id.loc[chosen.to_numpy(), "FType"].to_numpy()
    out.loc[chosen.index, "reach_totdasqkm"] = by_id.loc[chosen.to_numpy(), "totdasqkm"].to_numpy()

    overridden = decisions["override_nhdplus_id"].notna() & out["NHDPlusID"].notna()
    too_far = overridden & (out["snap_distance_m"] > sn.OVERRIDE_MAX_M)
    if too_far.any():
        raise ValueError(f"Override reach more than {sn.OVERRIDE_MAX_M:,} m from dam(s) "
                         f"{points.loc[too_far, DAM_ID].tolist()}")
    return out


def process_huc8(cfg, huc8, all_dams, rules, overrides, nhd_dir):
    watershed = load_huc8_watershed(cfg.path("wbd_source"), huc8)
    bounds = tuple(watershed.total_bounds)
    near = near_bounds(all_dams, "Longitude", "Latitude", bounds)
    if near.empty:
        return None
    try:
        gdb_path = find_huc8_geodatabase(nhd_dir, huc8)
    except FileNotFoundError:
        dams = locate_points(near, "Longitude", "Latitude", None, NHDPLUS_CRS, bounds,
                             assumed_epsg=DAM_DATUM_EPSG)
        dams = clip_to_watersheds(dams, watershed.to_crs(NHDPLUS_CRS))
        out = pd.DataFrame(dams[[c for c in KEEP if c in dams]]).assign(
            huc8=huc8, package=None, water_layers_available=False, match_status=rs.REJECTED,
            match_reason="no_nhdplus_package", reservoir_status="no_nhdplus_package")
        return out, pd.DataFrame(), pd.DataFrame()

    margin = BBOX_MARGIN_DEG
    flowlines, polygons, catchments, flow, water = load_package(
        gdb_path, (bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin))
    dams = locate_points(near, "Longitude", "Latitude", None, flowlines.crs, bounds,
                         assumed_epsg=DAM_DATUM_EPSG)
    dams = clip_to_watersheds(dams, watershed.to_crs(flowlines.crs)).reset_index(drop=True)
    if dams.empty:
        return None
    metric = watershed.estimate_utm_crs(datum_name="NAD83")
    points = dams.to_crs(metric)
    flowlines = flowlines.to_crs(metric)
    print(f"  package {gdb_path.name}; water layers: {water}; dams: {len(dams)}", flush=True)

    decisions, outlets, candidates = dm.match_dams(
        points, flowlines, polygons.to_crs(metric), catchments.to_crs(metric), flow,
        dams[DAM_RIVER].to_dict(), pd.to_numeric(dams["Year_Removed"], errors="coerce").to_dict(),
        rules)
    if not water:
        decisions["reservoir_status"] = "no_water_layers"
    decisions = apply_overrides(decisions, dams[DAM_ID], overrides, kind="dam")
    placed = place(points, decisions, flowlines)
    # An override reach has its own flowlines below.
    edges = dm.flow_edges(flow)
    moved = decisions["override_nhdplus_id"].notna() & placed["NHDPlusID"].notna()
    for index in decisions.index[moved]:
        decisions.loc[index, "downstream_nhdplus_ids"] = dm._below(edges, placed.loc[index, "NHDPlusID"])

    out = pd.DataFrame(dams[[c for c in KEEP if c in dams]]).copy()
    out.insert(0, "huc8", huc8)
    out["package"] = gdb_path.stem
    out["water_layers_available"] = water
    out["metric_crs"] = metric.to_string()
    out["x_m"], out["y_m"] = points.geometry.x, points.geometry.y
    out = out.join(decisions.rename(columns={"status": "match_status", "reason": "match_reason",
                                             "evidence": "match_evidence"}))
    out = out.join(placed)
    ids = dams[DAM_ID].astype(str)
    if len(outlets):
        outlets.insert(0, DAM_ID, ids.loc[outlets["point_index"]].to_numpy())
        outlets = outlets.drop(columns=["point_index"])
    if len(candidates):
        candidates.insert(0, DAM_ID, ids.loc[candidates["point_index"]].to_numpy())
        candidates = candidates.drop(columns=["point_index"])
    return out, outlets, candidates


def choose_huc8s(cfg, huc8s, hu4s):
    chosen = list(cfg.huc8s)
    if huc8s:
        chosen = [h for h in chosen if h in set(huc8s)]
    if hu4s:
        chosen = [h for h in chosen if h[:4] in set(hu4s)]
    return chosen


def print_summary(dams, outlets):
    print("\n=== Dams by status and reason ===")
    print(dams.groupby(["match_status", "match_reason"], dropna=False).size()
          .rename("dams").to_string())
    print("\n=== Dams by reservoir status ===")
    print(dams["reservoir_status"].value_counts(dropna=False).to_string())
    at = dams[dams["reservoir_status"] == "dam_at_outlet"]
    if len(at):
        print("\n=== Reservoirs dams sit at ===")
        print(f"outlets per reservoir: {at['n_outlets'].value_counts().sort_index().to_dict()}")
        print(f"with a diversion outlet: {int((at['n_diversion_outlets'] > 0).sum())}")
        print(f"reservoir edited after the removal year: "
              f"{int(at['reservoir_mapped_after_removal'].fillna(False).sum())}")
        print(f"dam to outlet (m): {at['dam_to_outlet_m'].describe().round(1).to_dict()}")
    if len(outlets):
        terminal = int(outlets.drop_duplicates(["permanent_id", "outlet_id"])["terminal"].sum())
        print(f"terminal outlets (no flowline below) among nearby reservoirs: {terminal}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--huc8", nargs="+", help="Only these case-study HUC8s.")
    parser.add_argument("--hu4", nargs="+", help="Only the case study's HUC8s in these HUC4s.")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    rules = dm.rules_from_config(cfg)
    processed = cfg.project_root / "data" / "processed"
    output_dir = args.output_dir or processed / "dam_reaches" / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = sn.load_overrides(args.overrides or cfg.project_root / OVERRIDES_FILE)
    all_dams = filter_to_sampled_dams(load_dams(cfg.path("dam_source")), cfg)
    all_dams = pd.DataFrame(all_dams.drop(columns="geometry")).reset_index(drop=True)
    huc8s = choose_huc8s(cfg, args.huc8, args.hu4)
    print(f"Case study: {cfg.name}; HUC8s: {len(huc8s)}; dams: {len(all_dams):,}; "
          f"rules: {rules}", flush=True)

    results, all_outlets, failed = [], [], []
    for number, huc8 in enumerate(huc8s, start=1):
        print(f"[{number}/{len(huc8s)}] HUC8 {huc8}", flush=True)
        try:
            result = process_huc8(cfg, huc8, all_dams, rules, overrides,
                                  processed / "nhdplus_extracted")
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            failed.append(huc8)
            continue
        if result is None:
            print("  no dams", flush=True)
            continue
        dams, outlets, candidates = result
        dams.to_parquet(output_dir / f"dam_reaches_{huc8}.parquet", index=False)
        outlets.to_parquet(output_dir / f"dam_reservoir_outlets_{huc8}.parquet", index=False)
        candidates.to_parquet(output_dir / f"dam_candidates_{huc8}.parquet", index=False)
        results.append(dams)
        all_outlets.append(outlets)
        print(f"  {dams['match_status'].value_counts().to_dict()}", flush=True)

    if results:
        dams = pd.concat(results, ignore_index=True)
        outlets = pd.concat(all_outlets, ignore_index=True)
        dams.groupby(["huc8", "match_status", "match_reason", "reservoir_status"], dropna=False) \
            .size().rename("dams").reset_index().to_csv(output_dir / "summary.csv", index=False)
        print_summary(dams, outlets)
    print(f"\nSaved to: {output_dir}")
    if failed:
        raise RuntimeError(f"{len(failed)} HUC8(s) failed: {failed}")


if __name__ == "__main__":
    main()
