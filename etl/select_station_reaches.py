"""
Stage 3 of the reach-matching rework: choose each station's reach from its
stage-2 candidates (rules in etl/common/reach_selection.py), apply manual
overrides (config/snap_overrides.csv), and place accepted stations on
their reach by projecting the reported point onto it. The reported
coordinates are kept as given; the projection only gives the network
position (snap_fraction) and the snapped coordinates, in separate columns.

Reads data/processed/station_candidates/<case_study>/ (etl/station_candidates.py)
and writes data/processed/station_reaches/<case_study>/ (--output-dir):

    station_reaches_<huc8>.parquet  one row per station: snap_status
        (accepted / ambiguous / rejected), snap_reason, snap_evidence,
        auto_status / auto_reason before overrides, match_method, the
        chosen NHDPlusID (accepted only) or provisional_nhdplus_id (for
        review), reach attributes, snap_distance_m, snap_fraction,
        snapped_x_m / snapped_y_m (metric_crs) and snapped_lon / snapped_lat
        (NAD83), the reported coordinates, runner-up candidate
    summary.csv                     stations per HUC8, status and reason

The current snapping outputs (snap_features.py) are not touched.
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import CRS

from etl.common import reach_selection as rs
from etl.common import snapping as sn
from etl.common import water_features as wf
from etl.common.config import default_case_study, load_config
from etl.common.nhdplus import find_huc8_geodatabase
from etl.snap_features import BBOX_MARGIN_DEG, NHDPLUS_CRS, OVERRIDES_FILE, STATION_ID
from etl.station_candidates import read_layer

REACH_COLUMNS = ["FType", "GNIS_Name", "totdasqkm", "area_ratio", "generation_methods",
                 "path_polygon_layer", "path_polygon_ftype", "catchment_relation"]


def apply_overrides(decisions, ids, overrides, kind="station"):
    """Reviewed manual decisions replace automatic ones. `ids` are the
    feature ids (aligned with decisions); `kind` is the overrides'
    feature_kind (station / dam). Returns decisions with auto_status,
    auto_reason, match_method and override_nhdplus_id (checked later
    against the flowlines)."""
    out = decisions.copy()
    out["auto_status"], out["auto_reason"] = out["status"], out["reason"]
    out["match_method"] = "evidence_rules"
    out["override_nhdplus_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    for index, feature in ids.astype(str).items():
        if (kind, feature) not in overrides.index:
            continue
        override = overrides.loc[(kind, feature)]
        out.loc[index, "match_method"] = "manual_override"
        out.loc[index, "evidence"] = ";".join(filter(None, [out.loc[index, "evidence"],
                                                            "manual_override"]))
        if override["action"] == "reject":
            out.loc[index, ["status", "reason"]] = [rs.REJECTED, "manual_reject"]
        else:
            out.loc[index, ["status", "reason"]] = [rs.ACCEPTED, "manual_override"]
            out.loc[index, "override_nhdplus_id"] = int(override["nhdplus_id"])
    return out


def place_on_reaches(points, decisions, candidates, gdb_path):
    """Reach attributes, snap_distance_m, snap_fraction and snapped
    coordinates for every accepted station."""
    out = pd.DataFrame(index=points.index)
    accepted = decisions["status"] == rs.ACCEPTED
    reach = decisions["override_nhdplus_id"].astype("Int64").fillna(
        decisions["provisional_nhdplus_id"].astype("Int64"))
    out["NHDPlusID"] = reach.where(accepted).astype("Int64")
    for column in ["snap_distance_m", "snap_fraction", "snapped_x_m", "snapped_y_m",
                   "snapped_lon", "snapped_lat"]:
        out[column] = np.nan
    chosen = out["NHDPlusID"].dropna()
    if chosen.empty:
        return out.join(pd.DataFrame(index=out.index, columns=REACH_COLUMNS))

    lon, lat = points["LongitudeMeasure"], points["LatitudeMeasure"]
    bbox = (lon.min() - BBOX_MARGIN_DEG, lat.min() - BBOX_MARGIN_DEG,
            lon.max() + BBOX_MARGIN_DEG, lat.max() + BBOX_MARGIN_DEG)
    lines = read_layer(gdb_path, "NHDFlowline", ["NHDPlusID", "FType", "GNIS_Name"], bbox)
    lines["NHDPlusID"] = wf.clean_nhdplus_id(lines["NHDPlusID"])
    lines = lines.dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID").set_index("NHDPlusID")
    missing = sorted(set(chosen) - set(lines.index))
    if missing:
        raise ValueError(f"Chosen or override reach(es) not in {gdb_path.name} near the "
                         f"stations: {missing}")

    metric = CRS.from_user_input(points["metric_crs"].dropna().iat[0])
    lines = lines.to_crs(metric)
    at = gpd.GeoSeries(gpd.points_from_xy(points.loc[chosen.index, "x_m"],
                                          points.loc[chosen.index, "y_m"]), crs=metric)
    fractions, snapped, distances = rs.project_onto(
        lines.geometry.loc[chosen.to_numpy()].to_numpy(), at.to_numpy())
    snapped = gpd.GeoSeries(snapped, crs=metric)
    geographic = snapped.to_crs(NHDPLUS_CRS)
    out.loc[chosen.index, "snap_fraction"] = fractions
    out.loc[chosen.index, "snap_distance_m"] = distances
    out.loc[chosen.index, "snapped_x_m"] = snapped.x.to_numpy()
    out.loc[chosen.index, "snapped_y_m"] = snapped.y.to_numpy()
    out.loc[chosen.index, "snapped_lon"] = geographic.x.to_numpy()
    out.loc[chosen.index, "snapped_lat"] = geographic.y.to_numpy()

    overridden = decisions["override_nhdplus_id"].notna() & accepted
    too_far = overridden & (out["snap_distance_m"] > sn.OVERRIDE_MAX_M)
    if too_far.any():
        raise ValueError(f"Override reach more than {sn.OVERRIDE_MAX_M:,} m from station(s) "
                         f"{points.loc[too_far, STATION_ID].tolist()}")

    # Reach attributes from the station's candidate row (an override reach
    # that wasn't a candidate gets what the flowline layer has).
    rows = candidates.drop_duplicates([STATION_ID, "NHDPlusID"]).set_index([STATION_ID, "NHDPlusID"])
    keys = list(zip(points.loc[chosen.index, STATION_ID].astype(str), chosen.astype("int64")))
    found = rows.reindex(keys)[REACH_COLUMNS]
    found.index = chosen.index
    fallback = lines.loc[chosen.to_numpy(), ["FType", "GNIS_Name"]].set_axis(chosen.index)
    found["FType"] = found["FType"].fillna(pd.to_numeric(fallback["FType"], errors="coerce"))
    found["GNIS_Name"] = found["GNIS_Name"].fillna(fallback["GNIS_Name"])
    return out.join(found.reindex(out.index))


def process_huc8(huc8, candidate_dir, nhd_dir, rules, overrides):
    points = pd.read_parquet(candidate_dir / f"station_points_{huc8}.parquet")
    candidates = pd.read_parquet(candidate_dir / f"station_candidates_{huc8}.parquet")
    points[STATION_ID] = points[STATION_ID].astype(str)
    candidates[STATION_ID] = candidates[STATION_ID].astype(str)
    candidates["NHDPlusID"] = candidates["NHDPlusID"].astype("Int64")

    decisions = rs.decide_stations(points, candidates, rules, STATION_ID)
    decisions = apply_overrides(decisions, points[STATION_ID], overrides)
    if (decisions["status"] == rs.ACCEPTED).any():
        placed = place_on_reaches(points, decisions, candidates,
                                  find_huc8_geodatabase(nhd_dir, huc8))
    else:
        placed = place_on_reaches(points, decisions, candidates, None)

    out = points.drop(columns=["nearest_eligible_id"], errors="ignore").copy()
    out["snap_status"] = decisions["status"]
    out["snap_reason"] = decisions["reason"]
    out["snap_evidence"] = decisions["evidence"]
    for column in ["auto_status", "auto_reason", "match_method", "provisional_nhdplus_id",
                   "n_eligible", "n_plausible", "runner_up_id", "runner_up_distance_m",
                   "runner_up_name", "runner_up_area_km2", "runner_up_methods"]:
        out[column] = decisions[column]
    out = out.join(placed.rename(columns={c: f"reach_{c.lower()}" for c in REACH_COLUMNS}))
    return out


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--huc8", nargs="+", help="Only these HUC8s.")
    parser.add_argument("--hu4", nargs="+", help="Only HUC8s in these HUC4s.")
    parser.add_argument("--candidate-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overrides", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    rules = rs.rules_from_config(cfg)
    processed = cfg.project_root / "data" / "processed"
    candidate_dir = args.candidate_dir or processed / "station_candidates" / cfg.name
    output_dir = args.output_dir or processed / "station_reaches" / cfg.name
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = sn.load_overrides(args.overrides or cfg.project_root / OVERRIDES_FILE)
    huc8s = sorted(p.stem.removeprefix("station_points_")
                   for p in candidate_dir.glob("station_points_*.parquet"))
    if args.huc8:
        huc8s = [h for h in huc8s if h in set(args.huc8)]
    if args.hu4:
        huc8s = [h for h in huc8s if h[:4] in set(args.hu4)]
    print(f"Case study: {cfg.name}; HUC8s with candidates: {len(huc8s)}; rules: {rules}; "
          f"manual overrides: {len(overrides)}", flush=True)

    results, failed = [], []
    for number, huc8 in enumerate(huc8s, start=1):
        try:
            out = process_huc8(huc8, candidate_dir, processed / "nhdplus_extracted", rules,
                               overrides)
        except Exception as exc:
            print(f"[{number}/{len(huc8s)}] HUC8 {huc8} FAILED: {exc}", flush=True)
            failed.append(huc8)
            continue
        out.to_parquet(output_dir / f"station_reaches_{huc8}.parquet", index=False)
        results.append(out[["huc8", "snap_status", "snap_reason"]])
        print(f"[{number}/{len(huc8s)}] HUC8 {huc8}: "
              f"{out['snap_status'].value_counts().to_dict()}", flush=True)

    if results:
        all_rows = pd.concat(results, ignore_index=True)
        summary = all_rows.value_counts().rename("stations").reset_index() \
            .sort_values(["huc8", "snap_status", "snap_reason"])
        summary.to_csv(output_dir / "summary.csv", index=False)
        print("\n=== Stations by status and reason (all HUC8s) ===")
        print(all_rows.groupby(["snap_status", "snap_reason"], dropna=False).size()
              .rename("stations").to_string())
    print(f"\nSaved to: {output_dir}")
    if failed:
        raise RuntimeError(f"{len(failed)} HUC8(s) failed: {failed}")


if __name__ == "__main__":
    main()
