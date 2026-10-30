"""
Stage 2 of the reach-matching rework: list the candidate reaches of every
candidate station, with the evidence for each (rules in
etl/common/candidates.py). Nothing is chosen here, and the current
snapping outputs (snap_features.py) are not touched.

For each HUC8:
1. Stations are located as in snap_features.py: reported coordinates moved
   from their reported datum to NAD83, then kept if inside the HUC8.
2. Candidates come from the water polygon the station is in (its
   artificial paths, by stored identifier), the catchment it is in (that
   flowline and its direct up/downstream neighbours), and every flowline
   within the search radius.

Outputs in --output-dir (default
data/processed/station_candidates/<case_study>/):

    station_points_<huc8>.parquet      one row per station: reported and
        datum-transformed coordinates, its water polygon and catchment,
        candidate counts, candidate_status
    station_candidates_<huc8>.parquet  one row per station and candidate
        reach: distance, FType, GNIS name match, drainage-area ratio,
        polygon and catchment relation, how it was found
    summary.csv                        one row per HUC8 processed

candidate_status: candidates | no_candidates | datum_unsupported |
no_nhdplus_package. A package without the water layers (see
download_streams.slurm) still gets catchment and nearby candidates;
water_layers_available records it.
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
import pyogrio

from etl.common import candidates as cand
from etl.common import snapping as sn
from etl.common import water_features as wf
from etl.common.config import default_case_study, load_config
from etl.common.nhdplus import (
    WATER_LAYERS,
    find_huc8_geodatabase,
    missing_slim_layers,
    normalize_nhdplus_columns,
)
from etl.common.wbd import clip_to_watersheds, load_huc8_watershed
from etl.snap_features import (
    BBOX_MARGIN_DEG,
    NHDPLUS_CRS,
    STATION_DATUM,
    STATION_ID,
    STATION_NAME,
    check_datums,
    load_candidate_stations,
    locate_points,
    near_bounds,
)

POINT_KEEP = [STATION_ID, STATION_NAME, "LatitudeMeasure", "LongitudeMeasure", STATION_DATUM,
              "datum_epsg", "datum_assumed", "datum_transform", "datum_accuracy_m"]


# --------------------------------------------------
# Reading one package (attributes by name, any case; geometry by bbox)
# --------------------------------------------------

def _layer_names(path):
    return {str(name).lower(): str(name) for name, _ in pyogrio.list_layers(path)}


def read_layer(path, layer, columns, bbox=None, geometry=True):
    """The columns of `layer` that exist (canonical names; missing ones
    added as NA), clipped to bbox (layer CRS) when given."""
    name = _layer_names(path).get(layer.lower())
    if name is None:
        return None
    fields = {str(f).lower(): str(f) for f in pyogrio.read_info(path, layer=name)["fields"]}
    frame = pyogrio.read_dataframe(path, layer=name, bbox=bbox, read_geometry=geometry,
                                   columns=[fields[c.lower()] for c in columns
                                            if c.lower() in fields])
    frame = normalize_nhdplus_columns(frame)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def load_package(gdb_path, bbox):
    """Flowlines (with drainage area and their water polygon), water
    polygons, catchments and the flow table near bbox."""
    flowlines = read_layer(gdb_path, "NHDFlowline", ["NHDPlusID", "GNIS_Name", "FType", "FCode",
                                                     "WBArea_Permanent_Identifier"], bbox)
    if flowlines.crs is None:
        raise ValueError("The NHDFlowline layer has no defined CRS.")
    flowlines = flowlines[flowlines.geometry.notna() & ~flowlines.geometry.is_empty]
    flowlines["NHDPlusID"] = wf.clean_nhdplus_id(flowlines["NHDPlusID"])
    flowlines["FType"] = pd.to_numeric(flowlines["FType"], errors="coerce")
    flowlines = flowlines.dropna(subset=["NHDPlusID"]).drop_duplicates("NHDPlusID")
    vaa = read_layer(gdb_path, "NHDPlusFlowlineVAA", ["NHDPlusID", "TotDASqKm"], geometry=False)
    if vaa is not None:
        vaa = pd.DataFrame({"NHDPlusID": wf.clean_nhdplus_id(vaa["NHDPlusID"]),
                            "TotDASqKm": vaa["TotDASqKm"]}).drop_duplicates("NHDPlusID")
        flowlines = flowlines.merge(vaa, on="NHDPlusID", how="left")
    else:
        flowlines["TotDASqKm"] = pd.NA
    flowlines["totdasqkm"] = sn.valid_area(flowlines["TotDASqKm"])

    water = not any(layer in WATER_LAYERS for layer in missing_slim_layers(gdb_path))
    shapes = []
    if water:
        for layer in ("NHDArea", "NHDWaterbody"):
            frame = read_layer(gdb_path, layer, ["Permanent_Identifier", "NHDPlusID", "FType",
                                                 "FCode"], bbox)
            if frame is not None and len(frame):
                shapes.append(frame.assign(_layer=layer))
    empty = pd.DataFrame(columns=["Permanent_Identifier", "NHDPlusID", "FType", "FCode"])
    areas = next((f for f in shapes if f["_layer"].iat[0] == "NHDArea"), empty)
    bodies = next((f for f in shapes if f["_layer"].iat[0] == "NHDWaterbody"), empty)
    attributes = wf.water_polygons(areas, bodies)
    geometry = [g for f in (areas, bodies) if isinstance(f, gpd.GeoDataFrame) for g in f.geometry]
    polygons = gpd.GeoDataFrame(attributes, geometry=geometry, crs=flowlines.crs)

    linked = wf.link_flowlines_to_polygons(flowlines, attributes)
    flowlines = flowlines.assign(wbarea_id=linked["wbarea_id"].to_numpy(),
                                 path_polygon_layer=linked["polygon_layer"].to_numpy(),
                                 path_polygon_ftype=linked["polygon_ftype"].to_numpy())

    catchments = read_layer(gdb_path, "NHDPlusCatchment", ["NHDPlusID", "SourceFC"], bbox)
    catchments = gpd.GeoDataFrame({
        "catchment_id": wf.clean_nhdplus_id(catchments["NHDPlusID"]),
        "sourcefc": catchments["SourceFC"].astype("string").str.strip().str.lower(),
    }, geometry=catchments.geometry.values, crs=catchments.crs)
    flow = read_layer(gdb_path, "NHDPlusFlow", ["FromNHDPID", "ToNHDPID"], geometry=False)
    return flowlines, polygons, catchments, pd.DataFrame(flow), water


# --------------------------------------------------
# One HUC8
# --------------------------------------------------

def _points_table(stations, huc8, package, water, status, metric_crs=None, summary=None):
    keep = [c for c in POINT_KEEP if c in stations.columns]
    out = pd.DataFrame(stations[keep]).copy()
    out.insert(0, "huc8", huc8)
    out["package"] = package
    out["water_layers_available"] = water
    out["metric_crs"] = None if metric_crs is None else metric_crs.to_string()
    out["x_m"] = stations.geometry.x if metric_crs is not None else pd.NA
    out["y_m"] = stations.geometry.y if metric_crs is not None else pd.NA
    if summary is not None:
        out = out.join(summary)
    out["candidate_status"] = status
    return out


def process_huc8(cfg, huc8, all_stations, rules, nhd_dir):
    watershed = load_huc8_watershed(cfg.path("wbd_source"), huc8)
    bounds = tuple(watershed.total_bounds)
    near = near_bounds(all_stations, "LongitudeMeasure", "LatitudeMeasure", bounds)

    try:
        gdb_path = find_huc8_geodatabase(nhd_dir, huc8)
    except FileNotFoundError:
        stations = locate_points(near, "LongitudeMeasure", "LatitudeMeasure",
                                 near[STATION_DATUM].tolist(), NHDPLUS_CRS, bounds)
        stations = clip_to_watersheds(stations, watershed.to_crs(NHDPLUS_CRS))
        points = _points_table(stations, huc8, None, False, "no_nhdplus_package")
        return points, pd.DataFrame(columns=["huc8", STATION_ID] + cand.CANDIDATE_COLUMNS[1:])

    margin = BBOX_MARGIN_DEG
    bbox = (bounds[0] - margin, bounds[1] - margin, bounds[2] + margin, bounds[3] + margin)
    flowlines, polygons, catchments, flow, water = load_package(gdb_path, bbox)
    stream_crs = flowlines.crs
    stations = locate_points(near, "LongitudeMeasure", "LatitudeMeasure",
                             near[STATION_DATUM].tolist(), stream_crs, bounds)
    stations = clip_to_watersheds(stations, watershed.to_crs(stream_crs))
    stations = stations.reset_index(drop=True)
    print(f"  package {gdb_path.name}; water layers: {water}; stations: {len(stations):,}; "
          f"flowlines: {len(flowlines):,}; polygons: {len(polygons):,}; "
          f"catchments: {len(catchments):,}", flush=True)

    metric_crs = watershed.estimate_utm_crs(datum_name="NAD83")
    supported = stations[stations["datum_transform"] != "unsupported"]
    points_m = supported.to_crs(metric_crs)
    names = supported[STATION_NAME].to_dict() if STATION_NAME in supported else {}
    areas = sn.reported_area_km2(supported).to_dict()
    summary, candidates = cand.build_candidates(
        points_m, flowlines.to_crs(metric_crs), polygons.to_crs(metric_crs),
        catchments.to_crs(metric_crs), flow, names, areas, rules)

    points = _points_table(stations.to_crs(metric_crs), huc8, gdb_path.stem, water, None,
                           metric_crs, summary)
    points["candidate_status"] = "candidates"
    points.loc[points["n_candidates"].fillna(0) == 0, "candidate_status"] = "no_candidates"
    points.loc[stations["datum_transform"] == "unsupported", "candidate_status"] = \
        "datum_unsupported"

    candidates.insert(0, "huc8", huc8)
    candidates.insert(1, STATION_ID,
                      stations.loc[candidates["point_index"], STATION_ID].astype(str).to_numpy())
    candidates = candidates.drop(columns="point_index")
    return points, candidates


def huc8_summary(huc8, points, candidates):
    status = points["candidate_status"].value_counts()
    return {
        "huc8": huc8,
        "package": points["package"].iat[0] if len(points) else None,
        "water_layers_available": bool(points["water_layers_available"].any()) if len(points)
        else None,
        "n_stations": len(points),
        **{f"n_{key}": int(status.get(key, 0)) for key in
           ("candidates", "no_candidates", "datum_unsupported", "no_nhdplus_package")},
        "n_in_water_polygon": int(points.get("point_polygon_has_paths",
                                             pd.Series(dtype="boolean")).fillna(False).sum()),
        "n_in_sink_catchment": int(points.get("in_sink_catchment",
                                              pd.Series(dtype=bool)).fillna(False).sum()),
        "n_candidate_rows": len(candidates),
        "median_candidates": float(points["n_candidates"].median())
        if "n_candidates" in points and len(points) else None,
    }


def choose_huc8s(cfg, huc8s, hu4s):
    """--huc8 values (case-study or not) plus the case study's HUC8s in
    --hu4; every case-study HUC8 when neither is given."""
    chosen = list(huc8s or [])
    if hu4s:
        chosen += [h for h in cfg.huc8s if h[:4] in set(hu4s) and h not in chosen]
    return chosen or cfg.huc8s


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--huc8", nargs="+", help="Process these HUC8s.")
    parser.add_argument("--hu4", nargs="+", help="Process the case study's HUC8s in these HUC4s.")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    rules = cand.rules_from_config(cfg)
    output_dir = args.output_dir or (cfg.project_root / "data" / "processed"
                                     / "station_candidates" / cfg.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    huc8s = choose_huc8s(cfg, args.huc8, args.hu4)
    print(f"Case study: {cfg.name}; HUC8s: {len(huc8s)}; rules: {rules}", flush=True)

    all_stations = load_candidate_stations(cfg.wqp_results_dir / "pilot_station_selection.csv")
    check_datums(all_stations)

    rows, failed = [], []
    for number, huc8 in enumerate(huc8s, start=1):
        print(f"[{number}/{len(huc8s)}] HUC8 {huc8}", flush=True)
        try:
            points, candidates = process_huc8(cfg, huc8, all_stations, rules, nhd_dir)
        except Exception as exc:
            print(f"  FAILED: {exc}", flush=True)
            failed.append(huc8)
            rows.append({"huc8": huc8, "error": str(exc)})
            continue
        points.to_parquet(output_dir / f"station_points_{huc8}.parquet", index=False)
        candidates.to_parquet(output_dir / f"station_candidates_{huc8}.parquet", index=False)
        rows.append(huc8_summary(huc8, points, candidates))
        print(f"  {points['candidate_status'].value_counts().to_dict()}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output_dir}")
    if failed:
        raise RuntimeError(f"{len(failed)} HUC8(s) failed: {failed}")


if __name__ == "__main__":
    main()
