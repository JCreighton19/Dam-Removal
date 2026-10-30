"""Candidate reaches for monitoring stations (stage 2 of the reach-matching
rework). Used by etl/station_candidates.py. Nothing here chooses a reach:
every candidate is kept with the evidence for and against it, and the
choice is made later from that evidence.

A station's candidates come from three sources:

1. Water polygon. The point lies in (or within polygon_tolerance_m of)
   an NHDArea or NHDWaterbody polygon that has artificial paths, i.e. a
   wide river or a lake/reservoir on the network. The paths are found by
   their stored WBArea_Permanent_Identifier (etl/common/water_features.py),
   not by geometry. Polygons without paths (isolated ponds, rapids,
   closed-basin sinks) give no candidates; NHDPlusBurnWaterbody.OnOffNet is
   not used, as sink waterbodies are flagged on-network too.
2. Catchment. The flowline whose NHDPlusCatchment contains the point, plus
   the flowlines directly upstream and downstream of it (NHDPlusFlow), so a
   point in a tributary's catchment next to the main stem also gets the main
   stem. A catchment with no flowline (SourceFC NHDPlusSink, a closed basin)
   gives none and is flagged in_sink_catchment.
3. Nearby. Every flowline within search_radius_m, as before.

All distances are from the reported location (after the datum
transformation) to the candidate flowline; the point is not moved.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely

from etl.common import snapping as sn
from etl.common import water_features as wf

SINK_SOURCE = "nhdplussink"

POINT_COLUMNS = [
    "n_point_polygons", "point_polygon_id", "point_polygon_layer", "point_polygon_ftype",
    "point_polygon_distance_m", "point_polygon_has_paths", "point_catchment_id",
    "point_catchment_sourcefc", "in_sink_catchment", "n_candidates", "n_eligible",
    "nearest_eligible_id", "nearest_eligible_distance_m",
]

CANDIDATE_COLUMNS = [
    "point_index", "NHDPlusID", "distance_m", "distance_rank", "FType", "FCode",
    "eligible_ftype", "GNIS_Name", "name_match", "totdasqkm", "reported_area_km2",
    "area_ratio", "area_agrees", "wbarea_id", "path_polygon_layer", "path_polygon_ftype",
    "in_point_polygon", "catchment_relation", "from_polygon", "from_catchment",
    "from_nearby", "generation_methods",
]


@dataclass(frozen=True)
class CandidateRules:
    """Limits for station candidate generation (config block
    `station_candidates`)."""
    # Nearby flowlines: as far as the current station snapping looks.
    search_radius_m: float = sn.STATION_RULES.search_radius_m
    # A point this close to a water polygon counts as in it: bank stations
    # just outside the mapped shoreline.
    polygon_tolerance_m: float = 30.0
    # Reported / NHDPlus drainage area in this range agrees (as snapping.py).
    area_agree: tuple = sn.STATION_RULES.area_agree


def rules_from_config(cfg):
    block = cfg.get("station_candidates", {}) or {}
    rules = {**CandidateRules().__dict__, **block}
    rules["area_agree"] = tuple(rules["area_agree"])
    return CandidateRules(**rules)


def _distances(points, flowlines, point_index, stream_index):
    return shapely.distance(points.geometry.loc[point_index].values,
                            flowlines.geometry.loc[stream_index].values)


def point_polygons(points, polygons, tolerance_m):
    """Every (point, water polygon) pair within tolerance_m: point_index,
    permanent_id, polygon_layer, polygon_ftype, distance_m (0 inside)."""
    columns = ["point_index", "permanent_id", "polygon_layer", "polygon_ftype", "distance_m"]
    if polygons.empty or points.empty:
        return pd.DataFrame(columns=columns)
    near = gpd.GeoDataFrame({"point_index": points.index},
                            geometry=points.geometry.buffer(tolerance_m).values, crs=points.crs)
    pairs = gpd.sjoin(near, polygons[["permanent_id", "polygon_layer", "polygon_ftype", "geometry"]],
                      predicate="intersects")
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    pairs["distance_m"] = shapely.distance(points.geometry.loc[pairs["point_index"]].values,
                                           polygons.geometry.loc[pairs["index_right"]].values)
    return pairs[columns].reset_index(drop=True)


def point_catchments(points, catchments):
    """The catchment each point falls in: point_index, catchment_id,
    sourcefc (lower case). A point on a shared edge takes the lower id."""
    columns = ["point_index", "catchment_id", "sourcefc"]
    if catchments.empty or points.empty:
        return pd.DataFrame(columns=columns)
    pts = gpd.GeoDataFrame({"point_index": points.index}, geometry=points.geometry.values,
                           crs=points.crs)
    pairs = gpd.sjoin(pts, catchments[["catchment_id", "sourcefc", "geometry"]],
                      predicate="intersects")
    pairs = pairs.sort_values(["point_index", "catchment_id"]).drop_duplicates("point_index")
    return pairs[columns].reset_index(drop=True)


def _rows(point_index, stream_index, source, relation=None):
    return pd.DataFrame({"point_index": np.asarray(point_index, dtype="int64"),
                         "stream_index": np.asarray(stream_index, dtype="int64"),
                         "source": source, "catchment_relation": relation})


def polygon_candidate_rows(in_polygon, flowlines, points, rules):
    """Artificial paths of each point's water polygons. Paths within the
    search radius; the polygon's nearest path if none is."""
    paths = flowlines[flowlines["wbarea_id"].notna()
                      & (flowlines["FType"] == wf.ARTIFICIAL_PATH)]
    if in_polygon.empty or paths.empty:
        return _rows([], [], "polygon")
    by_polygon = pd.DataFrame({"permanent_id": paths["wbarea_id"].to_numpy(),
                               "stream_index": paths.index})
    pairs = in_polygon[["point_index", "permanent_id"]].merge(by_polygon, on="permanent_id")
    if pairs.empty:
        return _rows([], [], "polygon")
    pairs["d"] = _distances(points, flowlines, pairs["point_index"], pairs["stream_index"])
    nearest = pairs.loc[pairs.groupby(["point_index", "permanent_id"])["d"].idxmin()]
    keep = pd.concat([pairs[pairs["d"] <= rules.search_radius_m], nearest])
    keep = keep.drop_duplicates(["point_index", "stream_index"])
    return _rows(keep["point_index"], keep["stream_index"], "polygon")


def catchment_candidate_rows(in_catchment, flowlines, flow):
    """The flowline of each point's catchment, and the flowlines directly
    up- and downstream of it (that are in `flowlines`)."""
    index_by_id = pd.Series(flowlines.index, index=flowlines["NHDPlusID"].to_numpy())
    index_by_id = index_by_id[~index_by_id.index.duplicated()]
    own = in_catchment[in_catchment["catchment_id"].isin(index_by_id.index)]
    if own.empty:
        return _rows([], [], "catchment")
    frames = [_rows(own["point_index"], index_by_id.loc[own["catchment_id"]], "catchment", "own")]
    edges = pd.DataFrame({"from": wf.clean_nhdplus_id(flow["FromNHDPID"]),
                          "to": wf.clean_nhdplus_id(flow["ToNHDPID"])}).dropna()
    for relation, key, other in (("upstream", "to", "from"), ("downstream", "from", "to")):
        pairs = own[["point_index", "catchment_id"]].merge(
            edges, left_on="catchment_id", right_on=key)
        pairs = pairs[pairs[other].isin(index_by_id.index)]
        if len(pairs):
            frames.append(_rows(pairs["point_index"], index_by_id.loc[pairs[other]],
                                "catchment", relation))
    return pd.concat(frames, ignore_index=True)


def nearby_candidate_rows(points, flowlines, rules):
    pairs = sn.find_candidates(points.geometry, flowlines, rules.search_radius_m)
    return _rows(pairs["point_index"], pairs["stream_index"], "nearby")


def build_candidates(points, flowlines, polygons, catchments, flow, names, reported_areas,
                     rules):
    """Candidate reaches for every point, with evidence.

    points      GeoDataFrame (metric CRS), any index
    flowlines   GeoDataFrame (same CRS): NHDPlusID, GNIS_Name, FType, FCode,
                wbarea_id (cleaned WBArea_Permanent_Identifier), totdasqkm
    polygons    GeoDataFrame (same CRS) of water_features.water_polygons rows
                with geometry; may be empty (package without water layers)
    catchments  GeoDataFrame (same CRS): catchment_id, sourcefc
    flow        NHDPlusFlow attributes: FromNHDPID, ToNHDPID
    names, reported_areas  dicts keyed by point index

    Returns (point columns POINT_COLUMNS indexed like points, candidates
    with CANDIDATE_COLUMNS, one row per point and reach).
    """
    flowlines = flowlines.reset_index(drop=True)
    polygons = polygons.reset_index(drop=True)
    with_paths = set(flowlines.loc[flowlines["FType"] == wf.ARTIFICIAL_PATH, "wbarea_id"].dropna())

    in_polygon = point_polygons(points, polygons, rules.polygon_tolerance_m)
    in_polygon["has_paths"] = in_polygon["permanent_id"].isin(with_paths)
    in_catchment = point_catchments(points, catchments)

    rows = pd.concat([
        polygon_candidate_rows(in_polygon[in_polygon["has_paths"]], flowlines, points, rules),
        catchment_candidate_rows(in_catchment, flowlines, flow),
        nearby_candidate_rows(points, flowlines, rules),
    ], ignore_index=True)

    candidates = _merge_sources(rows, flowlines, points, in_polygon, names, reported_areas, rules)
    summary = _point_summary(points, in_polygon, in_catchment, candidates)
    return summary, candidates


RELATION_ORDER = {"own": 0, "downstream": 1, "upstream": 2}


def _merge_sources(rows, flowlines, points, in_polygon, names, reported_areas, rules):
    if rows.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    rows = rows.assign(_order=rows["catchment_relation"].map(RELATION_ORDER).fillna(9))
    grouped = rows.sort_values("_order").groupby(["point_index", "stream_index"], sort=False)
    out = grouped.agg(
        from_polygon=("source", lambda s: (s == "polygon").any()),
        from_catchment=("source", lambda s: (s == "catchment").any()),
        from_nearby=("source", lambda s: (s == "nearby").any()),
        catchment_relation=("catchment_relation", "first"),
    ).reset_index()

    reach = flowlines.loc[out["stream_index"]].reset_index(drop=True)
    for column in ["NHDPlusID", "GNIS_Name", "FType", "FCode", "wbarea_id", "totdasqkm",
                   "path_polygon_layer", "path_polygon_ftype"]:
        out[column] = reach[column].to_numpy()
    out["distance_m"] = _distances(points, flowlines, out["point_index"], out["stream_index"])
    out["eligible_ftype"] = out["FType"].isin(list(sn.STREAM_FTYPES))
    out["name_match"] = [sn.name_matches(names.get(p), g)
                         for p, g in zip(out["point_index"], out["GNIS_Name"])]
    out["reported_area_km2"] = out["point_index"].map(pd.Series(reported_areas, dtype="float64"))
    out["area_ratio"] = out["totdasqkm"] / out["reported_area_km2"]
    out["area_agrees"] = out["area_ratio"].between(*rules.area_agree).where(
        out["area_ratio"].notna())
    point_polygon = set(zip(in_polygon["point_index"], in_polygon["permanent_id"]))
    out["in_point_polygon"] = [(p, w) in point_polygon
                               for p, w in zip(out["point_index"], out["wbarea_id"])]
    out["generation_methods"] = [
        ";".join(name for name, flag in (("polygon", a), ("catchment", b), ("nearby", c)) if flag)
        for a, b, c in zip(out["from_polygon"], out["from_catchment"], out["from_nearby"])]
    out = out.sort_values(["point_index", "distance_m", "NHDPlusID"]).reset_index(drop=True)
    eligible = out["eligible_ftype"]
    out["distance_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out.loc[eligible, "distance_rank"] = (out[eligible].groupby("point_index").cumcount() + 1)
    return out[CANDIDATE_COLUMNS]


def _point_summary(points, in_polygon, in_catchment, candidates):
    out = pd.DataFrame(index=points.index)
    out["n_point_polygons"] = in_polygon.groupby("point_index").size().reindex(
        out.index, fill_value=0)
    # The point's polygon: one with paths first, then the closest.
    first = (in_polygon.sort_values(["point_index", "has_paths", "distance_m"],
                                    ascending=[True, False, True])
             .drop_duplicates("point_index").set_index("point_index"))
    out["point_polygon_id"] = first["permanent_id"].reindex(out.index)
    out["point_polygon_layer"] = first["polygon_layer"].reindex(out.index)
    out["point_polygon_ftype"] = first["polygon_ftype"].reindex(out.index).astype("Int64")
    out["point_polygon_distance_m"] = first["distance_m"].reindex(out.index).astype("float64")
    out["point_polygon_has_paths"] = first["has_paths"].reindex(out.index).astype("boolean")
    catchment = in_catchment.set_index("point_index")
    out["point_catchment_id"] = catchment["catchment_id"].reindex(out.index).astype("Int64")
    out["point_catchment_sourcefc"] = catchment["sourcefc"].reindex(out.index)
    out["in_sink_catchment"] = (out["point_catchment_sourcefc"] == SINK_SOURCE).fillna(False)
    out["n_candidates"] = candidates.groupby("point_index").size().reindex(out.index, fill_value=0)
    eligible = candidates[candidates["eligible_ftype"]]
    out["n_eligible"] = eligible.groupby("point_index").size().reindex(out.index, fill_value=0)
    nearest = eligible.drop_duplicates("point_index").set_index("point_index")
    out["nearest_eligible_id"] = nearest["NHDPlusID"].reindex(out.index).astype("Int64")
    out["nearest_eligible_distance_m"] = nearest["distance_m"].reindex(out.index).astype("float64")
    return out[POINT_COLUMNS]
