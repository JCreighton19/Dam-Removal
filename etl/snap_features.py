"""
Match every dam and candidate station of a case study to an NHDPlus HR
reach (rules in etl/common/snapping.py and docs/warehouse_spec.md,
section 3.12).

For each HUC8:
1. Move each point from its reported datum to the flowlines' datum
   (NHDPlus HR: NAD83). Stations use the WQP datum field; the dam list has
   none, so dams are taken as WGS84 and flagged datum_assumed. A missing
   PROJ grid (e.g. NADCON for NAD27) stops the run before any HUC8 is
   processed.
2. List every flowline within the search radius, and choose a reach from
   distance, river name, drainage area and confluence checks. Canals,
   connectors, coastline, pipelines and conduits are never chosen.
3. Give each point snap_status accepted / ambiguous / rejected and a
   snap_reason; apply validated manual overrides (config/snap_overrides.csv).

Outputs per HUC8, in --output-dir (default data/processed/):

    snapped_features_<huc8>.gpkg
        dams_snapped, stations_snapped   accepted points only, on the reach
                                         (what link_dams and load_core_tables read)
        dams_original, stations_original every point at its reported
                                         location, with status and reason
    snap_diagnostics_<huc8>.csv          one row per point: decision, evidence,
                                         runner-up reach, datum transformation
    snap_candidates_<huc8>.csv           every candidate reach of every point

Validation runs: --output-dir data/processed/snap_validation writes there
instead and leaves the current outputs and progress markers alone; compare
with etl/compare_snapping.py before re-running the pipeline.
"""

from pathlib import Path
import argparse
import os
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.ops import nearest_points

from etl.common import snapping as sn
from etl.common.config import default_case_study, load_config
from etl.common.dams import filter_to_sampled_dams, load_dams
from etl.common.io import require_file
from etl.common.nhdplus import (
    find_huc8_geodatabase,
    is_hu4_fallback,
    read_nhdplus_layer,
)
from etl.common.progress import HucProgress
from etl.common.wbd import clip_to_watersheds, load_huc8_watershed

# --------------------------------------------------
# Configuration
# --------------------------------------------------

STREAM_LAYER = "NHDFlowline"
VAA_LAYER = "NHDPlusFlowlineVAA"
NHDPLUS_CRS = "EPSG:4269"  # NAD83, used for the datum check before reading packages

STATION_ID = "MonitoringLocationIdentifier"
STATION_NAME = "MonitoringLocationName"
STATION_DATUM = "HorizontalCoordinateReferenceSystemDatumName"
DAM_ID = "AR_ID"
DAM_RIVER = "River"
# The dam list gives no datum; its coordinates are taken as WGS84.
DAM_DATUM_EPSG = 4326

# Points are transformed only if their reported coordinates fall within
# the watershed's bounding box plus this margin (degrees; datum shifts are
# well under it).
BBOX_MARGIN_DEG = 0.05

OVERRIDES_FILE = Path("config") / "snap_overrides.csv"

# Keys of a snapping.decide_reach result, plus what snap_points adds.
DECISION_KEYS = [
    "status", "reason", "evidence", "stream_index", "n_candidates", "nearest_any_ftype",
    "nearest_any_distance_m", "near_unsuitable_flowline", "runner_up_id", "runner_up_distance_m",
    "runner_up_name", "runner_up_area_km2", "area_ratio", "auto_status", "auto_reason",
    "match_method",
]

# Columns added to every snapped layer and diagnostics row.
SNAP_COLUMNS = [
    "feature_id", "snap_status", "snap_reason", "snap_evidence", "auto_status",
    "auto_reason", "match_method", "NHDPlusID", "snap_distance_m", "snap_fraction",
    "reach_ftype", "reach_fcode", "reach_gnis_name", "reach_totdasqkm", "reach_slope",
    "datum_source", "datum_epsg", "datum_assumed", "datum_transform", "datum_accuracy_m",
    "original_x", "original_y", "snapped_x", "snapped_y", "n_candidates",
    "runner_up_id", "runner_up_distance_m", "runner_up_name", "runner_up_area_km2",
    "nearest_any_ftype", "nearest_any_distance_m", "reported_area_km2", "area_ratio",
    "qc_near_unsuitable_flowline", "qc_far_snap", "qc_non_riverine_name",
]


# --------------------------------------------------
# Loading candidate stations
# --------------------------------------------------

def load_candidate_stations(candidate_path):
    """The candidate stations from select_candidate_stations.py, one row per
    station with numeric, in-range coordinates (not yet a GeoDataFrame:
    the geometry depends on each station's datum)."""
    require_file(
        candidate_path,
        "Candidate station file (run select_candidate_stations.py first)",
    )

    stations = pd.read_csv(candidate_path, low_memory=False)

    for col in ["LatitudeMeasure", "LongitudeMeasure"]:
        stations[col] = pd.to_numeric(stations[col], errors="coerce")

    stations = stations.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
    stations = stations[
        stations["LatitudeMeasure"].between(-90, 90)
        & stations["LongitudeMeasure"].between(-180, 180)
    ].copy()
    if STATION_DATUM not in stations.columns:
        stations[STATION_DATUM] = pd.NA
    return stations.drop_duplicates(subset=STATION_ID).reset_index(drop=True)


def check_datums(stations):
    """Fail before processing anything if a reported datum needs a PROJ
    grid that isn't installed. Checked per datum and HUC2 region, over the
    stations' extent."""
    region = stations["source_huc8"].astype(str).str.zfill(8).str[:2] \
        if "source_huc8" in stations.columns else pd.Series("all", index=stations.index)
    epsg = stations[STATION_DATUM].map(lambda name: sn.resolve_datum(name)[0])
    for (code, _), group in stations.groupby([epsg, region], dropna=True):
        bounds = (group["LongitudeMeasure"].min(), group["LatitudeMeasure"].min(),
                  group["LongitudeMeasure"].max(), group["LatitudeMeasure"].max())
        sn.datum_transformer(int(code), NHDPLUS_CRS, bounds)
    print("Datum check: every reported datum has its PROJ transformation available",
          flush=True)


# --------------------------------------------------
# Flowlines
# --------------------------------------------------

def load_streams(gdb_path):
    """NHDFlowline with drainage area and slope from NHDPlusFlowlineVAA
    (placeholders such as -9999 / -9998 become NaN)."""
    streams = read_nhdplus_layer(
        gdb_path, STREAM_LAYER, columns=["NHDPlusID", "GNIS_Name", "FType", "FCode"]
    )
    if streams.crs is None:
        raise ValueError("The NHDFlowline layer has no defined CRS.")
    streams = streams[streams.geometry.notna() & ~streams.geometry.is_empty].copy()
    streams["NHDPlusID"] = pd.to_numeric(streams["NHDPlusID"], errors="coerce").astype("Int64")
    streams = streams.dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID")

    layers = {str(name).lower() for name, _ in pyogrio.list_layers(gdb_path)}
    if VAA_LAYER.lower() in layers:
        vaa = read_nhdplus_layer(gdb_path, VAA_LAYER, columns=["NHDPlusID", "TotDASqKm", "Slope"],
                                 ignore_geometry=True)
        vaa["NHDPlusID"] = pd.to_numeric(vaa["NHDPlusID"], errors="coerce").astype("Int64")
        vaa = vaa.dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID")
        streams = streams.merge(vaa, on="NHDPlusID", how="left")
    else:
        print(f"WARNING: {gdb_path.name} has no {VAA_LAYER}; no drainage areas or slopes",
              flush=True)
        streams["TotDASqKm"] = np.nan
        streams["Slope"] = np.nan
    streams["totdasqkm"] = sn.valid_area(streams["TotDASqKm"])
    streams["slope"] = sn.valid_slope(streams["Slope"])
    return gpd.GeoDataFrame(streams.reset_index(drop=True), crs=streams.crs)


# --------------------------------------------------
# Points
# --------------------------------------------------

def near_bounds(frame, lon_col, lat_col, bounds):
    """Rows whose reported coordinates are near the watershed's bounding box."""
    west, south, east, north = bounds
    keep = (frame[lon_col].between(west - BBOX_MARGIN_DEG, east + BBOX_MARGIN_DEG)
            & frame[lat_col].between(south - BBOX_MARGIN_DEG, north + BBOX_MARGIN_DEG))
    return frame[keep]


def locate_points(frame, lon_col, lat_col, datums, stream_crs, bounds, assumed_epsg=None):
    """GeoDataFrame of `frame` in the flowlines' CRS, after the datum
    transformation. Points with no real transformation keep their reported
    coordinates (used only to place them in a HUC8) and are rejected later."""
    moved = sn.transform_points(frame[lon_col], frame[lat_col], datums, stream_crs, bounds,
                                assumed_epsg=assumed_epsg)
    x = moved["x"].fillna(frame[lon_col])
    y = moved["y"].fillna(frame[lat_col])
    points = gpd.GeoDataFrame(frame.copy(), geometry=gpd.points_from_xy(x, y), crs=stream_crs)
    points["datum_source"] = None if datums is None else pd.Series(datums, index=frame.index)
    for column in ["datum_epsg", "datum_assumed", "datum_transform", "datum_accuracy_m"]:
        points[column] = moved[column]
    return points


def snap_points(points, streams_metric, rules, kind, names, reported_areas, overrides):
    """Decide a reach for each point (already in the metric CRS). Returns
    (points with SNAP_COLUMNS filled, candidate table)."""
    candidates = sn.find_candidates(points.geometry, streams_metric, rules.search_radius_m)
    by_point = dict(tuple(candidates.groupby("point_index"))) if len(candidates) else {}
    empty = candidates.iloc[0:0]

    rows = []
    for index, point in points.iterrows():
        feature_id = str(point["feature_id"])
        if point["datum_transform"] == "unsupported":
            decision = {"status": sn.REJECTED, "reason": "datum_unsupported", "evidence": [],
                        "stream_index": None, "n_candidates": 0}
        else:
            decision = sn.decide_reach(by_point.get(index, empty), names.get(index),
                                       reported_areas.get(index, np.nan), rules)
        decision["auto_status"], decision["auto_reason"] = decision["status"], decision["reason"]
        decision["match_method"] = "scored_candidates"
        apply_override(decision, overrides, kind, feature_id, point, streams_metric)
        rows.append(decision)

    decisions = pd.DataFrame(rows, index=points.index, columns=DECISION_KEYS)
    out = points.copy()
    out["snap_status"] = decisions["status"]
    out["snap_reason"] = decisions["reason"]
    out["snap_evidence"] = decisions["evidence"].map(lambda codes: ";".join(codes) or None)
    for column in ["auto_status", "auto_reason", "match_method", "n_candidates",
                   "runner_up_id", "runner_up_distance_m", "runner_up_name",
                   "runner_up_area_km2", "nearest_any_ftype", "nearest_any_distance_m",
                   "area_ratio"]:
        out[column] = decisions[column]
    out["qc_near_unsuitable_flowline"] = decisions["near_unsuitable_flowline"] \
        .astype("boolean").fillna(False).astype(bool)

    out["original_x"] = out.geometry.x
    out["original_y"] = out.geometry.y
    reach = decisions["stream_index"]
    snapped, fractions, distances = [], [], []
    for index, stream_index in reach.items():
        if stream_index is None or pd.isna(stream_index):
            snapped.append(None)
            fractions.append(np.nan)
            distances.append(np.nan)
            continue
        line = streams_metric.geometry.loc[stream_index]
        on_line = nearest_points(out.geometry.loc[index], line)[1]
        snapped.append(on_line)
        # Fraction along the line: 0 at its start, 1 at its end.
        fractions.append(line.project(on_line, normalized=True))
        distances.append(out.geometry.loc[index].distance(on_line))
    out["snapped_geometry"] = gpd.GeoSeries(snapped, index=out.index, crs=out.crs)
    out["snap_fraction"] = fractions
    out["snap_distance_m"] = distances
    out["snapped_x"] = [g.x if g is not None else np.nan for g in snapped]
    out["snapped_y"] = [g.y if g is not None else np.nan for g in snapped]

    chosen = reach.dropna()
    reaches = (streams_metric.loc[chosen.astype(int).to_numpy()]
               .set_axis(chosen.index).reindex(out.index))
    out["NHDPlusID"] = reaches["NHDPlusID"].to_numpy()
    out["reach_ftype"] = pd.to_numeric(reaches["FType"], errors="coerce").astype("Int16").to_numpy()
    out["reach_fcode"] = pd.to_numeric(reaches["FCode"], errors="coerce").astype("Int32").to_numpy()
    out["reach_gnis_name"] = reaches["GNIS_Name"].to_numpy()
    out["reach_totdasqkm"] = reaches["totdasqkm"].to_numpy()
    out["reach_slope"] = reaches["slope"].to_numpy()
    out["qc_far_snap"] = (out["snap_distance_m"] > rules.review_distance_m).fillna(False)
    out["reported_area_km2"] = pd.Series(reported_areas, dtype="float64").reindex(out.index)
    if kind == "station":
        out["qc_non_riverine_name"] = sn.non_riverine_name(out[STATION_NAME]).to_numpy()
    else:
        out["qc_non_riverine_name"] = False

    candidates = candidates.assign(
        feature_kind=kind,
        feature_id=points.loc[candidates["point_index"], "feature_id"].astype(str).to_numpy(),
        name_match=[sn.name_matches(names.get(i), g)
                    for i, g in zip(candidates["point_index"], candidates["GNIS_Name"])],
        eligible_ftype=pd.to_numeric(candidates["FType"], errors="coerce")
        .isin(list(sn.STREAM_FTYPES)).to_numpy(),
    ).drop(columns=["point_index", "stream_index"])
    return out, candidates


def apply_override(decision, overrides, kind, feature_id, point, streams_metric):
    """Replace an automatic decision with a reviewed manual one, if any."""
    if (kind, feature_id) not in overrides.index:
        return
    override = overrides.loc[(kind, feature_id)]
    decision["match_method"] = "manual_override"
    decision["evidence"] = list(decision.get("evidence") or []) + ["manual_override"]
    if override["action"] == "reject":
        decision.update(status=sn.REJECTED, reason="manual_reject", stream_index=None)
        return

    reach_id = int(override["nhdplus_id"])
    matches = streams_metric.index[streams_metric["NHDPlusID"] == reach_id]
    if len(matches) == 0:
        raise ValueError(f"Snap override for {kind} {feature_id}: reach {reach_id} is not "
                         "in this HUC8's NHDPlus package")
    if point["datum_transform"] == "unsupported":
        raise ValueError(f"Snap override for {kind} {feature_id}: its datum can't be "
                         "transformed, so the reach distance can't be checked")
    distance = point.geometry.distance(streams_metric.geometry.loc[matches[0]])
    if distance > sn.OVERRIDE_MAX_M:
        raise ValueError(f"Snap override for {kind} {feature_id}: reach {reach_id} is "
                         f"{distance:,.0f} m away (limit {sn.OVERRIDE_MAX_M:,} m)")
    decision.update(status=sn.ACCEPTED, reason="manual_override", stream_index=matches[0])


# --------------------------------------------------
# Main
# --------------------------------------------------

def write_outputs(output_dir, huc8, stations, dams, candidates):
    """The gpkg (written to a temporary file, then renamed) and the CSVs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"snapped_features_{huc8}.gpkg"
    tmp_file = output_dir / f"snapped_features_{huc8}.tmp.gpkg"
    tmp_file.unlink(missing_ok=True)

    layers = []
    for kind, frame in [("stations", stations), ("dams", dams)]:
        original = gpd.GeoDataFrame(frame.drop(columns="snapped_geometry"),
                                    geometry=frame.geometry, crs=frame.crs)
        accepted = frame[frame["snap_status"] == sn.ACCEPTED]
        on_reach = gpd.GeoDataFrame(accepted.drop(columns=["geometry", "snapped_geometry"]),
                                    geometry=accepted["snapped_geometry"], crs=frame.crs)
        layers += [(f"{kind}_original", original), (f"{kind}_snapped", on_reach)]

    for number, (layer, frame) in enumerate(layers):
        frame.to_crs("EPSG:4326").to_file(tmp_file, layer=layer, driver="GPKG",
                                          mode="w" if number == 0 else "a")
    os.replace(tmp_file, output_file)

    columns = ["feature_kind", "feature_name", "river"] + SNAP_COLUMNS
    diagnostics = pd.concat([
        pd.DataFrame(stations).assign(feature_kind="station", feature_name=stations.get(STATION_NAME),
                                      river=None)[columns],
        pd.DataFrame(dams).assign(feature_kind="dam", feature_name=dams.get("Dam_Name"),
                                  river=dams.get(DAM_RIVER))[columns],
    ], ignore_index=True)
    diagnostics.insert(0, "huc8", huc8)
    diagnostics.to_csv(output_dir / f"snap_diagnostics_{huc8}.csv", index=False)
    candidates.assign(huc8=huc8).to_csv(output_dir / f"snap_candidates_{huc8}.csv", index=False)
    return output_file, diagnostics


def process_huc8(cfg, huc8, all_stations, all_dams, rules, overrides, output_dir):
    dam_rules, station_rules = rules
    print(f"\n=== Processing HUC8: {huc8} ===")

    watershed = load_huc8_watershed(cfg.path("wbd_source"), huc8)
    bounds = tuple(watershed.total_bounds)

    # Load the already-extracted HUC8 geodatabase (an HU8 package, or an
    # HU4 fallback covering this HUC8's parent HUC4).
    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_path = find_huc8_geodatabase(nhd_dir, huc8)
    if is_hu4_fallback(gdb_path, huc8):
        print(f"Using HU4 fallback package for HUC8 {huc8}: {gdb_path.name}")

    streams = load_streams(gdb_path)
    stream_crs = streams.crs
    metric_crs = watershed.estimate_utm_crs(datum_name="NAD83")
    print(f"Flowline CRS: {stream_crs.name}; metric CRS: {metric_crs}")
    print(f"Loaded stream reaches: {len(streams):,}")

    near_stations = near_bounds(all_stations, "LongitudeMeasure", "LatitudeMeasure", bounds)
    near_dams = near_bounds(all_dams, "Longitude", "Latitude", bounds)
    stations = locate_points(near_stations, "LongitudeMeasure", "LatitudeMeasure",
                             near_stations[STATION_DATUM].tolist(), stream_crs, bounds)
    dams = locate_points(near_dams, "Longitude", "Latitude", None, stream_crs, bounds,
                         assumed_epsg=DAM_DATUM_EPSG)

    # Restrict points to the watershed (in their transformed location).
    area = watershed.to_crs(stream_crs)
    stations = clip_to_watersheds(stations, area)
    dams = clip_to_watersheds(dams, area)
    stations["feature_id"] = stations[STATION_ID].astype(str)
    dams["feature_id"] = dams[DAM_ID].astype(str)
    print(f"Stations inside watershed: {len(stations):,}")
    print(f"Dam removals inside watershed: {len(dams):,}")

    streams_metric = streams.to_crs(metric_crs)
    stations_out, station_candidates = snap_points(
        stations.to_crs(metric_crs), streams_metric, station_rules, "station",
        stations[STATION_NAME].to_dict() if STATION_NAME in stations else {},
        sn.reported_area_km2(stations).to_dict(), overrides,
    )
    dams_out, dam_candidates = snap_points(
        dams.to_crs(metric_crs), streams_metric, dam_rules, "dam",
        dams[DAM_RIVER].to_dict() if DAM_RIVER in dams else {}, {}, overrides,
    )

    output_file, diagnostics = write_outputs(
        output_dir, huc8, stations_out, dams_out,
        pd.concat([station_candidates, dam_candidates], ignore_index=True),
    )

    summary = diagnostics.groupby(["feature_kind", "snap_status", "snap_reason"], dropna=False) \
        .size().rename("points")
    print("\nSnap decisions:")
    print(summary.to_string())
    accepted = diagnostics[diagnostics["snap_status"] == sn.ACCEPTED]
    print("\nAccepted snap distances (m) by kind:")
    print(accepted.groupby("feature_kind")["snap_distance_m"].describe().round(1).to_string())
    print(f"\nSaved to: {output_file}")
    return set(zip(diagnostics["feature_kind"], diagnostics["feature_id"].astype(str)))


def choose_huc8s(cfg, requested, sample, seed):
    """The HUC8s to process: --huc8 values plus a random --sample of the
    case study's other HUC8s; every case-study HUC8 when neither is given."""
    huc8s = list(requested or [])
    if sample:
        others = sorted(set(cfg.huc8s) - set(huc8s))
        huc8s += sorted(random.Random(seed).sample(others, min(sample, len(others))))
    return huc8s or cfg.huc8s


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    parser.add_argument(
        "--huc8",
        nargs="+",
        default=None,
        help="Process only these HUC8(s). Defaults to every HUC8 listed in the "
        "case-study config, processed one at a time.",
    )
    parser.add_argument("--sample", type=int, default=0,
                        help="Also process this many randomly chosen case-study HUC8s "
                        "(for validation runs).")
    parser.add_argument("--seed", type=int, default=42, help="Seed for --sample.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Write here instead of data/processed/ (validation runs; "
                        "progress markers are then not used).")
    parser.add_argument("--overrides", type=Path, default=None,
                        help=f"Manual overrides CSV (default {OVERRIDES_FILE}).")
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    huc8_list = choose_huc8s(cfg, args.huc8, args.sample, args.seed)
    default_dir = cfg.project_root / "data" / "processed"
    output_dir = args.output_dir or default_dir
    output_dir = output_dir if output_dir.is_absolute() else cfg.project_root / output_dir
    validation_run = output_dir.resolve() != default_dir.resolve()

    rules = sn.rules_from_config(cfg)
    overrides = sn.load_overrides(args.overrides or cfg.project_root / OVERRIDES_FILE)
    print(f"Manual overrides: {len(overrides)}")
    print(f"Rules: dams {rules[0]}\n       stations {rules[1]}")
    print(f"Output: {output_dir}" + (" (validation run)" if validation_run else ""))

    candidate_path = cfg.wqp_results_dir / "pilot_station_selection.csv"

    # Loaded once and reused across HUC8s: the candidate file and dam
    # source cover the whole case study, not one HUC8 each.
    all_stations = load_candidate_stations(candidate_path)
    all_dams = load_dams(cfg.path("dam_source"))
    check_datums(all_stations)

    sampled_dams = filter_to_sampled_dams(all_dams, cfg)
    if len(sampled_dams) < len(all_dams):
        print(
            f"Restricting to the {len(sampled_dams)} sampled dam(s) in the "
            f"config (of {len(all_dams):,} in the dam source).",
            flush=True,
        )
    all_dams = pd.DataFrame(sampled_dams.drop(columns="geometry")).reset_index(drop=True)

    results = []
    seen = set()

    # Finished HUC8s are skipped, so a run stopped by the time limit resumes.
    progress = None if validation_run else HucProgress(cfg, "snap_features")
    for huc8 in huc8_list:
        if progress and progress.done(huc8):
            results.append((huc8, "skipped (done earlier in this run)"))
            continue
        try:
            seen |= process_huc8(cfg, huc8, all_stations, all_dams, rules, overrides, output_dir)
            if progress:
                progress.mark(huc8)
            results.append((huc8, "ok"))
        except FileNotFoundError as exc:
            # E.g. no NHDPlus HR package covers this HUC8. At national scale
            # a few are expected, so the stage carries on without them.
            print(f"SKIPPED HUC8 {huc8} (missing input): {exc}", file=sys.stderr)
            if progress:
                progress.skip(huc8, exc)
            results.append((huc8, f"skipped (missing input): {exc}"))
        except Exception as exc:
            print(f"FAILED HUC8 {huc8}: {exc}", file=sys.stderr)
            results.append((huc8, f"failed: {exc}"))

    if len(huc8_list) > 1:
        print("\n=== Summary ===")
        for huc8, status in results:
            print(f"  {huc8}: {status}")

    if huc8_list == cfg.huc8s:
        unused = sorted(set(overrides.index) - seen)
        if unused:
            print(f"WARNING: {len(unused)} override(s) matched no point: {unused}")

    if any(status.startswith("failed") for _, status in results):
        raise RuntimeError("One or more HUC8s failed; see log above.")


if __name__ == "__main__":
    main()
