"""Choosing a station's reach from its candidates (stage 3 of the
reach-matching rework). Candidates and their evidence come from stage 2
(etl/common/candidates.py); used by etl/select_station_reaches.py.

For one station, in order:

1. Eligible candidates: StreamRiver (460) and ArtificialPath (558) only.
   None -> rejected.
2. Plausible: eligible candidates within review_distance_m, a path of the
   water polygon the station is in, or the flowline of the station's own
   catchment. None -> ambiguous (far_without_support).
3. Evidence narrows the plausible set, strongest first; each step applies
   only when it leaves at least one candidate:
     drainage area  reported area agrees (ratio in area_agree); some
                    comparable but none agree -> ambiguous
     river name     the station name contains the reach's GNIS name
     structure      a path of the station's water polygon, else the
                    flowline of its own catchment
   The closest remaining candidate is chosen (ties: larger drainage area,
   then lower NHDPlusID).
4. Checks that make the choice ambiguous unless drainage area or name
   supports it:
     confluence     another plausible reach about as close (tie_margin_m),
                    or in the same water polygon, of very different
                    drainage area (confluence_area_ratio)
     polygon_edge   the station is just outside a water polygon and a
                    non-polygon reach is closer than the polygon's path
     far            the chosen reach is beyond review_distance_m and not a
                    path of the station's water polygon
     sink           the station is in a closed-basin catchment and the
                    reach isn't a path of its water polygon
5. Otherwise accepted, with the strongest evidence as the reason.

Nothing is forced: an ambiguous station keeps its provisional reach for
review but is not matched (only accepted stations are used).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from etl.common import snapping as sn

ACCEPTED, AMBIGUOUS, REJECTED = sn.ACCEPTED, sn.AMBIGUOUS, sn.REJECTED

DECISION_COLUMNS = [
    "status", "reason", "evidence", "provisional_nhdplus_id", "n_eligible", "n_plausible",
    "runner_up_id", "runner_up_distance_m", "runner_up_name", "runner_up_area_km2",
    "runner_up_methods",
]


@dataclass(frozen=True)
class SelectionRules:
    """Thresholds for choosing a station's reach (config block
    `station_selection`)."""
    # Beyond this, a reach needs drainage-area or name support (or to be a
    # path of the station's water polygon).
    review_distance_m: float = sn.STATION_RULES.review_distance_m
    # Reaches this much farther than the chosen one count as about as close.
    tie_margin_m: float = sn.STATION_RULES.tie_margin_m
    # Drainage areas differing by this factor are different-size rivers.
    confluence_area_ratio: float = sn.STATION_RULES.confluence_area_ratio


def rules_from_config(cfg):
    return SelectionRules(**{**SelectionRules().__dict__,
                             **(cfg.get("station_selection", {}) or {})})


def _decision(status, reason, evidence=(), chosen=None, eligible=None, plausible=None):
    out = dict.fromkeys(DECISION_COLUMNS)
    out.update(status=status, reason=reason, evidence=";".join(evidence) or None,
               n_eligible=0 if eligible is None else len(eligible),
               n_plausible=0 if plausible is None else len(plausible))
    if chosen is not None:
        out["provisional_nhdplus_id"] = int(chosen["NHDPlusID"])
        others = plausible.drop(index=chosen.name)
        if len(others):
            runner = others.iloc[0]
            out.update(runner_up_id=int(runner["NHDPlusID"]),
                       runner_up_distance_m=float(runner["distance_m"]),
                       runner_up_name=None if pd.isna(runner["GNIS_Name"]) else runner["GNIS_Name"],
                       runner_up_area_km2=float(runner["totdasqkm"]),
                       runner_up_methods=runner["generation_methods"])
    return out


def _order(frame):
    return frame.assign(_area=frame["totdasqkm"].fillna(-1)).sort_values(
        ["distance_m", "_area", "NHDPlusID"], ascending=[True, False, True]
    ).drop(columns="_area")


def decide_station_reach(candidates, point, rules):
    """Decide one station's reach.

    candidates  its stage-2 candidate rows
    point       its stage-2 point row (point_polygon_distance_m,
                point_polygon_has_paths, in_sink_catchment, ...)
    Returns a dict with DECISION_COLUMNS.
    """
    eligible = candidates[candidates["eligible_ftype"].astype(bool)]
    if eligible.empty:
        reason = ("sink_catchment_no_reach" if bool(point.get("in_sink_catchment"))
                  else "no_eligible_candidate")
        return _decision(REJECTED, reason, eligible=eligible)
    eligible = _order(eligible)
    in_polygon = eligible["in_point_polygon"].astype(bool)
    own = eligible["catchment_relation"] == "own"
    plausible = eligible[(eligible["distance_m"] <= rules.review_distance_m) | in_polygon | own]
    if plausible.empty:
        return _decision(AMBIGUOUS, "far_without_support", chosen=eligible.iloc[0],
                         eligible=eligible, plausible=eligible)

    evidence, pool = [], plausible
    # Drainage area.
    comparable = pool["area_agrees"].notna()
    if comparable.any():
        agrees = pool[pool["area_agrees"].fillna(False).astype(bool)]
        if agrees.empty:
            return _decision(AMBIGUOUS, "drainage_area_mismatch", ["area_mismatch"],
                             chosen=pool.iloc[0], eligible=eligible, plausible=plausible)
        pool = agrees
        evidence.append("area_agrees")
    # River name.
    named = pool[pool["name_match"].eq(True)]
    if len(named):
        pool = named
        evidence.append("name_match")
    elif pool["name_match"].eq(False).any():
        evidence.append("name_mismatch")
    # Structure.
    for column, value, code in (("in_point_polygon", True, "in_water_polygon"),
                                ("catchment_relation", "own", "own_catchment")):
        matches = pool[pool[column] == value]
        if len(matches):
            pool = matches
            evidence.append(code)
            break
    chosen = pool.iloc[0]
    plausible = pd.concat([plausible.loc[[chosen.name]], plausible.drop(index=chosen.name)])
    supported = "area_agrees" in evidence or "name_match" in evidence
    chosen_in_polygon = bool(chosen["in_point_polygon"])

    # Confluence: a very different-size reach about as close, or sharing
    # the station's polygon or catchment.
    others = plausible.drop(index=chosen.name)
    # (Inside a polygon, distance to each path's centreline says little.)
    close = others[(others["distance_m"] <= chosen["distance_m"] + rules.tie_margin_m)
                   | (others["in_point_polygon"].astype(bool) & chosen_in_polygon)]
    if chosen["totdasqkm"] > 0 and len(close):
        size = close["totdasqkm"] / chosen["totdasqkm"]
        if ((size >= rules.confluence_area_ratio) | (size <= 1 / rules.confluence_area_ratio)).any():
            evidence.append("confluence_nearby")
            if not supported:
                return _decision(AMBIGUOUS, "confluence_ambiguous", evidence, chosen,
                                 eligible, plausible)
    # Just outside a water polygon, with another reach closer than its path.
    edge = point.get("point_polygon_distance_m")
    if (chosen_in_polygon and not pd.isna(edge) and edge > 0
            and (~others["in_point_polygon"].astype(bool)
                 & (others["distance_m"] < chosen["distance_m"])).any()):
        evidence.append("polygon_edge")
        if not supported:
            return _decision(AMBIGUOUS, "polygon_edge_ambiguous", evidence, chosen,
                             eligible, plausible)
    if chosen["distance_m"] > rules.review_distance_m and not chosen_in_polygon:
        evidence.append("far_snap")
        if not supported:
            return _decision(AMBIGUOUS, "far_without_support", evidence, chosen,
                             eligible, plausible)
    if bool(point.get("in_sink_catchment")) and not chosen_in_polygon:
        evidence.append("sink_catchment")
        if not supported:
            return _decision(AMBIGUOUS, "sink_catchment", evidence, chosen, eligible, plausible)

    for code, reason in (("area_agrees", "drainage_area_match"), ("name_match", "name_match"),
                         ("in_water_polygon", "water_polygon"), ("own_catchment", "own_catchment")):
        if code in evidence:
            break
    else:
        reason = "nearest_plausible_reach"
    return _decision(ACCEPTED, reason, evidence, chosen, eligible, plausible)


def decide_stations(points, candidates, rules, key):
    """One decision per row of `points` (indexed like points), with
    candidates matched on the `key` column. Stations
    without candidates get rejected with their stage-2 candidate_status as
    the reason."""
    by_station = dict(tuple(candidates.groupby(key))) if len(candidates) else {}
    empty = candidates.iloc[0:0]
    rows = []
    for _, point in points.iterrows():
        status = point.get("candidate_status")
        if status in ("datum_unsupported", "no_nhdplus_package"):
            rows.append(_decision(REJECTED, status))
            continue
        rows.append(decide_station_reach(by_station.get(point[key], empty), point, rules))
    return pd.DataFrame(rows, index=points.index, columns=DECISION_COLUMNS)


def project_onto(lines, points):
    """For aligned arrays of shapely lines and points: (snap_fraction,
    snapped points, distance_m). The fraction runs 0 at the line's start
    to 1 at its end; the points themselves are not changed."""
    import shapely

    fractions = shapely.line_locate_point(lines, points, normalized=True)
    snapped = shapely.line_interpolate_point(lines, fractions, normalized=True)
    return np.asarray(fractions, dtype="float64"), snapped, shapely.distance(points, lines)
