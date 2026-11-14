"""
Export the dams to review by hand, with the flowlines around them, as one
GeoPackage for QGIS (read-only; writes only to data/processed/dam_review/
<case_study>/). Workflow: docs/dam_review.md.

Review set (dated dams only), one priority each, first that applies:

    1 flagged       known suspects (config/suspect_dam_matches.csv), the dams
                    flagged in the 2026-11-13 review (FLAGGED_2026_11_13) and
                    the "different path" / "path unknown" disagreements of
                    preprocessing_audit section 13b, if that output exists
    2 baci          in analysis_measurement with an assigned downstream row
                    and an upstream control outside the former reservoir
    3 analysis      any other dam in analysis_measurement
    4 recoverable   not in analysis_measurement, no accepted reach or no usable
                    link, and a WQP station of the case study within
                    --station-km (straight line): a better placement could
                    bring it in. Dams with no station that close are left out:
                    no placement can make them usable.

Layers (EPSG:4326):

    dams          reported location of each review dam: priority, match
                  status / reason, reservoir status, current reach (accepted,
                  else provisional) with its name and drainage area, flags;
                  empty reviewed / review_note fields for ticking dams off
    dams_current  where accepted dams are placed now (on their reach)
    flowlines     every NHDPlus HR flowline within --radius-km of a review
                  dam's reported or current location: nhdplus_id, gnis_name,
                  ftype, fcode, totdasqkm. Read from every package the circle
                  touches, so it includes side streams the warehouse lacks.
    corrections   empty point layer for the review: AR_ID, action
                  (move / reject), nhdplus_id, reason, source

Also writes dam_review_dams.csv (the dams layer without geometry).
etl/apply_dam_review.py turns the edited corrections layer into rows of
config/snap_overrides.csv.

Run on the cluster (Slurm only): sbatch data/slurm/export_dam_review.slurm
A wider circle for a few dams: DAMS="CO-022 WI-043" RADIUS_KM=6 sbatch ...
(writes dam_review_extra.gpkg, so the main export isn't overwritten).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from scipy.spatial import cKDTree

from analysis.mixed_model import SETTINGS as MODEL_SETTINGS
from etl.common import water_features as wf
from etl.common.config import default_case_study, load_config
from etl.common.dams import filter_to_sampled_dams, load_dams
from etl.common.io import find_column
from etl.common.nhdplus import find_huc8_geodatabase
from etl.common.reach_inputs import dam_id_text, read_per_huc8
from etl.common.warehouse import warehouse_dir
from etl.common.wbd import find_hu8_layer
from etl.station_candidates import read_layer

WGS84 = "EPSG:4326"
METRIC_CRS = "EPSG:5070"        # CONUS Albers; metres, fine for few-km circles
EARTH_KM = 6371.0
RADIUS_KM = 3.0
STATION_KM = 30.0
VAA_CHUNK = 2000
PRIORITIES = ["flagged", "baci", "analysis", "recoverable"]
# Dams flagged in the 2026-11-13 review of preprocessing_audit sections 13 / 13b.
FLAGGED_2026_11_13 = ["TN-032", "VT-034", "NJ-024", "PA-116", "OH-055", "OH-058",
                      "PA-059", "PA-232", "PA-273", "UT-004"]
FLAG_RELATIONS = ("different path", "path unknown")
CORRECTION_FIELDS = {"AR_ID": "string", "action": "string", "nhdplus_id": "Int64",
                     "reason": "string", "source": "string"}


# --------------------------------------------------
# Review set
# --------------------------------------------------

def is_dated(year):
    return pd.to_numeric(year, errors="coerce").fillna(0) > 0


def analysis_roles(path, settings=MODEL_SETTINGS):
    """AR_ID, in_analysis, baci from analysis_measurement (roles as in
    analysis/mixed_model.py, without the distance and flag restrictions)."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: build the analysis table first.")
    treated = ", ".join(f"'{s}'" for s in settings["treated_exposure"])
    control = ", ".join(f"'{s}'" for s in settings["control_exposure"])
    frame = duckdb.sql(f"""
        SELECT CAST(AR_ID AS VARCHAR) AS AR_ID,
               bool_or(side = 'downstream' AND exposure_status IN ({treated})) AS treated,
               bool_or(side = 'upstream' AND exposure_status IN ({control})
                       AND NOT coalesce(upstream_in_former_reservoir, false)) AS control
        FROM read_parquet('{Path(path).as_posix()}') GROUP BY 1
    """).df()
    frame["AR_ID"] = dam_id_text(frame["AR_ID"])
    return pd.DataFrame({"AR_ID": frame["AR_ID"], "in_analysis": True,
                         "baci": frame["treated"].fillna(False) & frame["control"].fillna(False)})


def _unit_vectors(lon, lat):
    lon, lat = np.radians(np.asarray(lon, float)), np.radians(np.asarray(lat, float))
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def nearest_km(lon, lat, other_lon, other_lat):
    """Great-circle distance (km) from each point to the nearest other point."""
    if len(other_lon) == 0:
        return np.full(len(lon), np.inf)
    chord, _ = cKDTree(_unit_vectors(other_lon, other_lat)).query(_unit_vectors(lon, lat))
    return 2 * EARTH_KM * np.arcsin(np.clip(chord / 2, 0, 1))


def assign_priority(frame, station_km=STATION_KM):
    """priority (first of PRIORITIES that applies, None = not reviewed) and
    priority_rank. Needs dated, flagged, baci, in_analysis, match_status,
    linked, nearest_station_km."""
    recoverable = (~frame["in_analysis"] & frame["nearest_station_km"].le(station_km)
                   & (frame["match_status"].ne("accepted") | ~frame["linked"]))
    conditions = [frame["flagged"], frame["baci"], frame["in_analysis"], recoverable]
    priority = pd.Series(np.select(conditions, PRIORITIES, default=""), index=frame.index)
    priority = priority.where(frame["dated"] & priority.ne(""))
    rank = priority.map({p: i + 1 for i, p in enumerate(PRIORITIES)}).astype("Int64")
    return priority, rank


def _optional_csv(path, columns):
    path = Path(path)
    if not path.exists():
        print(f"  (not found, skipped: {path})")
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype={"AR_ID": str}, low_memory=False)
    frame["AR_ID"] = dam_id_text(frame["AR_ID"])
    return frame[[c for c in columns if c in frame]]


def review_dams(cfg, station_km=STATION_KM):
    """Every listed dam with its review attributes and priority."""
    processed = cfg.project_root / "data" / "processed"
    dams = filter_to_sampled_dams(load_dams(cfg.path("dam_source")), cfg)
    dams = pd.DataFrame(dams.drop(columns="geometry"))
    dams["AR_ID"] = dam_id_text(dams["AR_ID"])
    dams = dams.drop_duplicates("AR_ID")
    keep = ["AR_ID", "Dam_Name", "River", "State", "Year_Removed", "Latitude", "Longitude"]
    dams = dams[[c for c in keep if c in dams]].reset_index(drop=True)
    dams["dated"] = is_dated(dams["Year_Removed"])

    reaches = read_per_huc8(cfg, "dam_reaches", "dam_reaches", cfg.huc8s)
    columns = ["AR_ID", "huc8", "match_status", "match_reason", "match_method", "reservoir_status",
               "NHDPlusID", "provisional_nhdplus_id", "reach_gnis_name", "reach_totdasqkm",
               "snapped_lon", "snapped_lat"]
    if len(reaches):
        reaches["AR_ID"] = dam_id_text(reaches["AR_ID"])
        reaches = reaches[[c for c in columns if c in reaches]].drop_duplicates("AR_ID")
        dams = dams.merge(reaches, on="AR_ID", how="left")
    for column in columns[1:]:
        if column not in dams:
            dams[column] = pd.NA
    dams["match_status"] = dams["match_status"].fillna("no_match_record")
    accepted = dams["match_status"].eq("accepted")
    dams["current_reach"] = pd.to_numeric(dams["NHDPlusID"], errors="coerce").where(
        accepted, pd.to_numeric(dams["provisional_nhdplus_id"], errors="coerce")).astype("Int64")

    warehouse = warehouse_dir(cfg)
    roles = analysis_roles(warehouse / "analysis_measurement.parquet")
    dams = dams.merge(roles, on="AR_ID", how="left")
    dams["in_analysis"] = dams["in_analysis"].astype("boolean").fillna(False).astype(bool)
    dams["baci"] = dams["baci"].astype("boolean").fillna(False).astype(bool)
    links = pd.read_parquet(warehouse / "dam_station_link.parquet", columns=["AR_ID"])
    dams["linked"] = dams["AR_ID"].isin(set(dam_id_text(links["AR_ID"])))

    stations = pd.read_csv(cfg.wqp_results_dir / "pilot_station_selection.csv",
                           usecols=["LatitudeMeasure", "LongitudeMeasure"], low_memory=False)
    stations = stations.apply(pd.to_numeric, errors="coerce").dropna()
    dams["nearest_station_km"] = nearest_km(dams["Longitude"], dams["Latitude"],
                                            stations["LongitudeMeasure"], stations["LatitudeMeasure"])

    suspects = _optional_csv(cfg.project_root / "config" / "suspect_dam_matches.csv", ["AR_ID"])
    audit = _optional_csv(processed / "preprocessing_audit" / cfg.name / "dam_nearest_vs_evidence.csv",
                          ["AR_ID", "relation", "nearest_reach", "nearest_name"])
    funnel = _optional_csv(processed / "stage1_audit" / cfg.name / "dam_funnel" / "dam_funnel.csv",
                           ["AR_ID", "reason"]).rename(columns={"reason": "funnel_reason"})
    dams = dams.merge(audit.rename(columns={"relation": "nearest_relation"}), on="AR_ID", how="left") \
        .merge(funnel, on="AR_ID", how="left")
    dams["known_suspect"] = dams["AR_ID"].isin(set(suspects["AR_ID"]))
    disagree = dams.get("nearest_relation", pd.Series(dtype=str)).astype("string") \
        .fillna("").str.startswith(FLAG_RELATIONS)
    dams["flagged"] = dams["known_suspect"] | dams["AR_ID"].isin(FLAGGED_2026_11_13) | disagree

    dams["priority"], dams["priority_rank"] = assign_priority(dams, station_km)
    return dams


# --------------------------------------------------
# Flowlines
# --------------------------------------------------

def circles(dams, radius_km):
    """One circle per dam (METRIC_CRS) around its reported and current points."""
    reported = gpd.GeoSeries(gpd.points_from_xy(dams["Longitude"], dams["Latitude"]), crs=WGS84)
    current = gpd.GeoSeries(gpd.points_from_xy(pd.to_numeric(dams["snapped_lon"], errors="coerce"),
                                               pd.to_numeric(dams["snapped_lat"], errors="coerce")),
                            crs="EPSG:4269")
    radius = radius_km * 1000
    shapes = reported.to_crs(METRIC_CRS).buffer(radius).to_numpy()
    has_current = current.notna() & ~current.is_empty & current.x.notna()
    if has_current.any():
        around = current[has_current].to_crs(METRIC_CRS).buffer(radius)
        shapes[has_current.to_numpy()] = [a.union(b) for a, b in
                                          zip(shapes[has_current.to_numpy()], around)]
    return gpd.GeoDataFrame({"AR_ID": dams["AR_ID"].to_numpy()}, geometry=shapes, crs=METRIC_CRS)


def hu8_polygons(wbd_path):
    layer = find_hu8_layer(wbd_path)
    fields = [str(f) for f in pyogrio.read_info(wbd_path, layer=layer)["fields"]]
    column = find_column(fields, "HUC8")
    wbd = gpd.read_file(wbd_path, layer=layer, columns=[column])
    return gpd.GeoDataFrame({"huc8": wbd[column].astype(str).str.zfill(8)},
                            geometry=wbd.geometry, crs=wbd.crs)


def _package_crs(path):
    return pyogrio.read_info(path, layer="NHDFlowline")["crs"]


def flowlines_near(rings, wbd_path, nhd_dir):
    """Every flowline intersecting a circle, from every package the circles
    touch, with drainage area. Returns (flowlines in WGS84, notes)."""
    hu8 = hu8_polygons(wbd_path).to_crs(METRIC_CRS)
    touched = gpd.sjoin(rings, hu8, predicate="intersects")[["AR_ID", "huc8"]]
    packages, notes = {}, []
    for huc8, group in touched.groupby("huc8"):
        try:
            path = find_huc8_geodatabase(nhd_dir, huc8)
        except FileNotFoundError:
            notes.append(f"no NHDPlus package for HUC8 {huc8} ({len(group)} dam circle(s))")
            continue
        packages.setdefault(path, set()).update(group["AR_ID"])

    frames = []
    for number, (path, ids) in enumerate(sorted(packages.items()), start=1):
        area = rings[rings["AR_ID"].isin(ids)]
        crs = _package_crs(path)
        bbox = tuple(area.to_crs(crs).total_bounds)
        lines = read_layer(path, "NHDFlowline", ["NHDPlusID", "GNIS_Name", "FType", "FCode"], bbox)
        if lines is None or lines.empty:
            continue
        lines["NHDPlusID"] = wf.clean_nhdplus_id(lines["NHDPlusID"])
        lines = lines.dropna(subset=["NHDPlusID"]).set_crs(crs, allow_override=True)
        inside = gpd.sjoin(lines.to_crs(METRIC_CRS), area[["geometry"]], predicate="intersects")
        lines = lines.loc[sorted(set(inside.index))]
        sizes = []
        ids_list = lines["NHDPlusID"].astype("int64").unique().tolist()
        for start in range(0, len(ids_list), VAA_CHUNK):
            vaa = read_layer(path, "NHDPlusFlowlineVAA", ["NHDPlusID", "TotDASqKm"], geometry=False,
                             nhdplus_ids=ids_list[start:start + VAA_CHUNK])
            if vaa is not None and len(vaa):
                sizes.append(pd.DataFrame(vaa))
        if sizes:
            vaa = pd.concat(sizes)
            vaa["NHDPlusID"] = wf.clean_nhdplus_id(vaa["NHDPlusID"])
            lines = lines.merge(vaa.drop_duplicates("NHDPlusID"), on="NHDPlusID", how="left")
        else:
            lines["TotDASqKm"] = np.nan
        frames.append(lines.to_crs(WGS84))
        print(f"  [{number}/{len(packages)}] {Path(path).name}: {len(lines):,} flowlines "
              f"for {len(ids)} dam(s)", flush=True)
    if not frames:
        return gpd.GeoDataFrame(columns=["nhdplus_id"], geometry=[], crs=WGS84), notes
    lines = pd.concat(frames, ignore_index=True).drop_duplicates("NHDPlusID")
    out = gpd.GeoDataFrame({
        "nhdplus_id": lines["NHDPlusID"].astype("int64"),
        "gnis_name": lines["GNIS_Name"].astype("string"),
        "ftype": pd.to_numeric(lines["FType"], errors="coerce").astype("Int64"),
        "fcode": pd.to_numeric(lines["FCode"], errors="coerce").astype("Int64"),
        "totdasqkm": pd.to_numeric(lines["TotDASqKm"], errors="coerce").where(lambda s: s > 0),
    }, geometry=lines.geometry.to_numpy(), crs=WGS84)
    return out, notes


# --------------------------------------------------
# Output
# --------------------------------------------------

DAM_LAYER_COLUMNS = ["AR_ID", "Dam_Name", "River", "State", "Year_Removed", "priority",
                     "priority_rank", "match_status", "match_reason", "match_method",
                     "reservoir_status", "current_reach", "reach_gnis_name", "reach_totdasqkm",
                     "in_analysis", "baci", "linked", "known_suspect", "nearest_relation",
                     "nearest_reach", "nearest_name", "funnel_reason", "nearest_station_km", "huc8"]


def write_review(dams, flowlines, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.gpkg")
    tmp.unlink(missing_ok=True)
    table = dams[[c for c in DAM_LAYER_COLUMNS if c in dams]].copy()
    table["nearest_station_km"] = table["nearest_station_km"].round(2)
    # For the reviewer's own bookkeeping in QGIS; apply_dam_review.py ignores them.
    table["reviewed"] = False
    table["review_note"] = pd.Series("", index=table.index, dtype="string")
    for column in ("current_reach", "nearest_reach"):
        if column in table:
            table[column] = pd.to_numeric(table[column], errors="coerce").astype("Int64")
    reported = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(dams["Longitude"], dams["Latitude"]),
                                crs=WGS84)
    pyogrio.write_dataframe(reported, tmp, layer="dams", driver="GPKG")

    placed = dams["match_status"].eq("accepted") & pd.to_numeric(dams["snapped_lon"], errors="coerce").notna()
    current = gpd.GeoDataFrame(
        dams.loc[placed, ["AR_ID", "current_reach", "reach_gnis_name"]].reset_index(drop=True),
        geometry=gpd.points_from_xy(pd.to_numeric(dams.loc[placed, "snapped_lon"]),
                                    pd.to_numeric(dams.loc[placed, "snapped_lat"])),
        crs="EPSG:4269").to_crs(WGS84)
    pyogrio.write_dataframe(current, tmp, layer="dams_current", driver="GPKG", geometry_type="Point")
    pyogrio.write_dataframe(flowlines, tmp, layer="flowlines", driver="GPKG",
                            geometry_type="MultiLineString", promote_to_multi=True)
    corrections = gpd.GeoDataFrame({c: pd.Series(dtype=t) for c, t in CORRECTION_FIELDS.items()},
                                   geometry=gpd.GeoSeries([], crs=WGS84), crs=WGS84)
    pyogrio.write_dataframe(corrections, tmp, layer="corrections", driver="GPKG",
                            geometry_type="Point")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--radius-km", type=float, default=RADIUS_KM)
    parser.add_argument("--station-km", type=float, default=STATION_KM)
    parser.add_argument("--dams", nargs="+", help="Only these AR_IDs (any priority, even unreviewed); "
                                                  "default output dam_review_extra.gpkg.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    out_dir = cfg.project_root / "data" / "processed" / "dam_review" / cfg.name
    output = args.output or out_dir / ("dam_review_extra.gpkg" if args.dams else "dam_review.gpkg")

    print(f"Case study: {cfg.name}; radius {args.radius_km} km; stations within {args.station_km} km",
          flush=True)
    dams = review_dams(cfg, args.station_km)
    if args.dams:
        wanted = set(dam_id_text(pd.Series(args.dams)))
        missing = sorted(wanted - set(dams["AR_ID"]))
        if missing:
            raise ValueError(f"Not in the dam list: {missing}")
        review = dams[dams["AR_ID"].isin(wanted)].copy()
        review["priority"] = review["priority"].fillna("requested")
    else:
        review = dams[dams["priority"].notna()].copy()
    review = review.sort_values(["priority_rank", "AR_ID"], na_position="last").reset_index(drop=True)

    print("\nListed dams:", len(dams), "| dated:", int(dams["dated"].sum()))
    print(review["priority"].value_counts().reindex(PRIORITIES + ["requested"]).dropna()
          .astype(int).to_string())
    print("\nReading flowlines...", flush=True)
    flowlines, notes = flowlines_near(circles(review, args.radius_km), cfg.path("wbd_source"),
                                      cfg.project_root / "data" / "processed" / "nhdplus_extracted")
    for note in notes:
        print(f"  NOTE: {note}")
    write_review(review, flowlines, output)
    review.drop(columns=["Latitude", "Longitude"]).assign(
        reported_lat=review["Latitude"], reported_lon=review["Longitude"]) \
        .to_csv(output.with_name(output.stem + "_dams.csv"), index=False)
    size_mb = output.stat().st_size / 1e6
    print(f"\n{len(review):,} dams, {len(flowlines):,} flowlines -> {output} ({size_mb:,.1f} MB)")


if __name__ == "__main__":
    main()
