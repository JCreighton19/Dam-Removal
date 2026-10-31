"""Matching dam removals to NHDPlus HR flowlines (stage 4 of the
reach-matching rework). Used by etl/match_dams.py.

Three flowline relationships are recorded separately for each dam:

    impounded   the artificial paths of the reservoir polygon the dam held
                back (NHDWaterbody, linked by WBArea_Permanent_Identifier)
    structure   the flowline the dam sits on: the reservoir's outlet path
                when the dam is at a reservoir outlet, otherwise the reach
                chosen from the dam's candidates
    downstream  the flowlines directly below the structure flowline
                (NHDPlusFlow)

The analysis needs the structure flowline and the dam's position on it
(snap_fraction): that splits the network into upstream and downstream of
the dam. The impounded paths say which upstream stations were in the
mapped reservoir.

Reservoir outlets: a path of the polygon with a flow connection to a
flowline that isn't a path of the polygon (or to nothing: a terminal
path). A polygon can have several; the main outlet is the one with the
largest drainage area, and an outlet on the minor side of a divergence
(VAA Divergence = 2) is a diversion. A dam within outlet_distance_m of a
reservoir's main outlet is placed on that outlet, unless:
    another outlet that isn't a diversion is of similar size
        -> ambiguous (multiple_outlets)
    the dam's River names neither the outlet path nor the flowline below
        -> ambiguous (reservoir_name_mismatch)
Dams with no reservoir polygon near them (often: the reservoir is gone
from the current hydrography after the removal, or was never mapped), or
not near a reservoir's outlet, are matched like stations
(reach_selection.decide_station_reach with the dam thresholds). Current
NHD can't show a drained impoundment; reservoir_fdate and
reservoir_mapped_after_removal record whether the mapped polygon postdates
the removal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shapely

from etl.common import candidates as cand
from etl.common import reach_selection as rs
from etl.common import snapping as sn
from etl.common import water_features as wf

RESERVOIR_LAYER = wf.WATERBODY_LAYER
MINOR_DIVERGENCE = 2


@dataclass(frozen=True)
class DamRules:
    """Thresholds for dam matching (config block `dam_matching`)."""
    search_radius_m: float = sn.DAM_RULES.search_radius_m
    review_distance_m: float = sn.DAM_RULES.review_distance_m
    # A dam sits on its reservoir's shoreline, so a wider tolerance than
    # for stations.
    polygon_tolerance_m: float = 100.0
    # A dam this close to a reservoir's main outlet is at that outlet.
    outlet_distance_m: float = 300.0
    tie_margin_m: float = sn.DAM_RULES.tie_margin_m
    # Outlets (or nearby reaches) within this drainage-area factor are of
    # similar size.
    confluence_area_ratio: float = sn.DAM_RULES.confluence_area_ratio

    def candidate_rules(self):
        return cand.CandidateRules(search_radius_m=self.search_radius_m,
                                   polygon_tolerance_m=self.polygon_tolerance_m)

    def selection_rules(self):
        return rs.SelectionRules(review_distance_m=self.review_distance_m,
                                 tie_margin_m=self.tie_margin_m,
                                 confluence_area_ratio=self.confluence_area_ratio)


def rules_from_config(cfg):
    return DamRules(**{**DamRules().__dict__, **(cfg.get("dam_matching", {}) or {})})


def _ids(values):
    return ";".join(str(int(v)) for v in values) or None


def _split_ids(value):
    """Inverse of _ids; None or NaN (a terminal outlet) gives []."""
    return [int(i) for i in value.split(";") if i] if isinstance(value, str) else []


def flow_edges(flow):
    return pd.DataFrame({"from": wf.clean_nhdplus_id(flow["FromNHDPID"]),
                         "to": wf.clean_nhdplus_id(flow["ToNHDPID"])}).dropna(subset=["from"])


def reservoir_outlets(polygon_ids, flowlines, edges):
    """One row per (reservoir polygon, outlet path) for the polygons in
    polygon_ids: permanent_id, outlet_id, downstream_ids, terminal,
    divergence, outlet_area_km2, is_main_outlet, is_diversion, n_outlets,
    n_diversion_outlets, n_paths, impounded_ids."""
    columns = ["permanent_id", "outlet_id", "downstream_ids", "terminal", "divergence",
               "outlet_area_km2", "is_main_outlet", "is_diversion", "n_outlets",
               "n_diversion_outlets", "n_paths", "impounded_ids"]
    paths = flowlines[flowlines["wbarea_id"].isin(set(polygon_ids))
                      & (flowlines["FType"] == wf.ARTIFICIAL_PATH)]
    if paths.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for polygon, group in paths.groupby("wbarea_id"):
        members = set(group["NHDPlusID"])
        out = edges[edges["from"].isin(members)]
        leaving = out[~out["to"].isin(members)]
        no_edges = members - set(out["from"])
        outlet_ids = sorted(set(leaving["from"]) | no_edges)
        info = group.set_index("NHDPlusID")
        for outlet in outlet_ids:
            below = leaving.loc[leaving["from"] == outlet, "to"].dropna()
            rows.append({"permanent_id": polygon, "outlet_id": int(outlet),
                         "downstream_ids": _ids(sorted(below)),
                         "terminal": below.empty,
                         "divergence": info.loc[outlet, "divergence"],
                         "outlet_area_km2": info.loc[outlet, "totdasqkm"],
                         "n_paths": len(members), "impounded_ids": _ids(sorted(members))})
    outlets = pd.DataFrame(rows)
    outlets["is_diversion"] = outlets["divergence"] == MINOR_DIVERGENCE
    outlets = outlets.assign(_area=outlets["outlet_area_km2"].fillna(-1)).sort_values(
        ["permanent_id", "is_diversion", "_area", "outlet_id"],
        ascending=[True, True, False, True]).drop(columns="_area")
    outlets["is_main_outlet"] = ~outlets.duplicated("permanent_id")
    counts = outlets.groupby("permanent_id").agg(n_outlets=("outlet_id", "size"),
                                                  n_diversion_outlets=("is_diversion", "sum"))
    outlets = outlets.join(counts, on="permanent_id")
    return outlets[columns].reset_index(drop=True)


def outlet_node(flowlines_by_id, outlet_id, downstream_ids):
    """The outlet path's downstream end: whichever end is nearer the
    flowline below it (the line's last vertex for a terminal path)."""
    coords = shapely.get_coordinates(flowlines_by_id.geometry.loc[outlet_id])
    first, last = shapely.points(coords[0]), shapely.points(coords[-1])
    below = [i for i in _split_ids(downstream_ids) if i in flowlines_by_id.index]
    if not below:
        return last
    target = flowlines_by_id.geometry.loc[below[0]]
    return first if shapely.distance(first, target) < shapely.distance(last, target) else last


def _name_ok(river, *names):
    """True / False if the dam's River matches any / none of the named
    flowlines; None if nothing to compare."""
    results = [sn.name_matches(river, name) for name in names]
    if any(r is True for r in results):
        return True
    return False if any(r is False for r in results) else None


def match_dams(points, flowlines, polygons, catchments, flow, rivers, removal_years, rules):
    """Match every dam (points in a metric CRS, any index).

    Returns (decisions indexed like points, reservoir rows, candidates).
    decisions: status, reason, evidence, provisional_nhdplus_id (structure
    flowline), runner-up, reservoir_status and the reservoir / outlet /
    impounded / downstream columns. reservoir rows: one per (dam, nearby
    reservoir outlet), for review of multi-outlet reservoirs.
    """
    flowlines = flowlines.reset_index(drop=True)
    by_id = flowlines.set_index("NHDPlusID")
    edges = flow_edges(flow)
    summary, candidates = cand.build_candidates(points, flowlines, polygons, catchments, flow,
                                                rivers, {}, rules.candidate_rules())

    reservoirs = polygons[polygons["polygon_layer"] == RESERVOIR_LAYER].reset_index(drop=True)
    near = cand.point_polygons(points, reservoirs, rules.outlet_distance_m)
    outlets = reservoir_outlets(near["permanent_id"].unique(), flowlines, edges)
    fdate = reservoirs.drop_duplicates("permanent_id").set_index("permanent_id")
    review = near.merge(outlets, on="permanent_id")
    if len(review):
        review["dam_to_outlet_m"] = [
            float(shapely.distance(points.geometry.loc[p],
                                   outlet_node(by_id, o, d)))
            for p, o, d in zip(review["point_index"], review["outlet_id"], review["downstream_ids"])]
        review["reservoir_ftype"] = review["permanent_id"].map(fdate["polygon_ftype"])
        review["reservoir_fdate"] = review["permanent_id"].map(fdate["polygon_fdate"])

    selection = rules.selection_rules()
    by_point = dict(tuple(candidates.groupby("point_index"))) if len(candidates) else {}
    empty = candidates.iloc[0:0]
    rows = []
    for index in points.index:
        decision = _decide(index, by_point.get(index, empty), summary.loc[index],
                           review[review["point_index"] == index] if len(review) else review,
                           rivers.get(index), removal_years.get(index), by_id, edges,
                           selection, rules)
        rows.append(decision)
    decisions = pd.DataFrame(rows, index=points.index)
    return decisions, review, candidates


def _below(edges, reach):
    return _ids(sorted(edges.loc[edges["from"] == reach, "to"].dropna()))


def _decide(index, candidates, point, review, river, year_removed, by_id, edges, selection,
            rules):
    base = {"reservoir_status": "no_reservoir_polygon", "reservoir_id": None,
            "reservoir_ftype": None, "reservoir_fdate": pd.NaT,
            "reservoir_mapped_after_removal": None, "impounded_nhdplus_ids": None,
            "n_impounded_paths": 0, "outlet_nhdplus_ids": None, "n_outlets": 0,
            "n_diversion_outlets": 0, "dam_to_outlet_m": np.nan,
            "n_reservoirs_nearby": int(review["permanent_id"].nunique()) if len(review) else 0}
    main = review[review["is_main_outlet"]] if len(review) else review
    at_outlet = main[main["dam_to_outlet_m"] <= rules.outlet_distance_m] if len(main) else main
    if len(at_outlet):
        res = at_outlet.sort_values(["dam_to_outlet_m", "outlet_id"]).iloc[0]
        all_outlets = review[review["permanent_id"] == res["permanent_id"]]
        fdate = res["reservoir_fdate"]
        base.update(
            reservoir_status="dam_at_outlet", reservoir_id=res["permanent_id"],
            reservoir_ftype=res["reservoir_ftype"], reservoir_fdate=fdate,
            reservoir_mapped_after_removal=(None if pd.isna(fdate) or pd.isna(year_removed)
                                            else bool(fdate.year > int(year_removed))),
            impounded_nhdplus_ids=res["impounded_ids"], n_impounded_paths=int(res["n_paths"]),
            outlet_nhdplus_ids=_ids(all_outlets["outlet_id"]), n_outlets=int(res["n_outlets"]),
            n_diversion_outlets=int(res["n_diversion_outlets"]),
            dam_to_outlet_m=float(res["dam_to_outlet_m"]))
        outlet = int(res["outlet_id"])
        evidence = ["reservoir_outlet"]
        decision = rs._decision(rs.ACCEPTED, "reservoir_outlet")
        decision.update(provisional_nhdplus_id=outlet)

        rivals = all_outlets[(all_outlets["outlet_id"] != outlet) & ~all_outlets["is_diversion"]]
        area = res["outlet_area_km2"]
        if len(rivals) and area > 0:
            ratio = rivals["outlet_area_km2"] / area
            if (ratio > 1 / rules.confluence_area_ratio).any():
                evidence.append("multiple_outlets")
                decision.update(status=rs.AMBIGUOUS, reason="multiple_outlets")
        if all_outlets["is_diversion"].any():
            evidence.append("diversion_outlet")
        names = [by_id.loc[outlet, "GNIS_Name"]] + [
            by_id.loc[b, "GNIS_Name"] for b in _split_ids(res["downstream_ids"]) if b in by_id.index]
        name = _name_ok(river, *names)
        if name is True:
            evidence.append("name_match")
        elif name is False:
            evidence.append("name_mismatch")
            if decision["status"] == rs.ACCEPTED:
                decision.update(status=rs.AMBIGUOUS, reason="reservoir_name_mismatch")
        decision["evidence"] = ";".join(evidence)
        decision.update(base)
        decision["downstream_nhdplus_ids"] = res["downstream_ids"]
        return decision

    if len(main):
        base["reservoir_status"] = "reservoir_not_at_outlet"
        base["dam_to_outlet_m"] = float(main["dam_to_outlet_m"].min())
    decision = rs.decide_station_reach(candidates, point, selection)
    if base["reservoir_status"] == "reservoir_not_at_outlet":
        decision["evidence"] = ";".join(filter(None, [decision["evidence"],
                                                      "reservoir_nearby_not_at_outlet"]))
    decision.update(base)
    reach = decision["provisional_nhdplus_id"]
    decision["downstream_nhdplus_ids"] = None if reach is None else _below(edges, reach)
    return decision
