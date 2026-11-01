"""Dam-station links on the evidence-based reaches (stage 5 of the
reach-matching rework). Used by etl/link_reaches.py.

Endpoints. Each dam (stage 4) and station (stage 3) gets one reach:

    accepted    its matched reach (NHDPlusID) and snap_fraction
    ambiguous   its provisional reach, with the reported point projected
                onto it for the fraction (provisional_fraction)
    rejected    no reach

A pair is usable only when both endpoints are accepted. Pairs with an
ambiguous endpoint are linked too, so the comparison can say how many links
each ambiguous reason costs; they are never used for analysis.

Relationship and distance follow etl/link_dams_stations.py, so old and new
links compare like for like: FromNHDPID -> ToNHDPID is downstream, a reach
both upstream and downstream of the dam (a loop) counts as upstream, on the
dam's own reach the fractions decide, and the distance is measured along the
shortest path between the two projected points (snap_fraction runs 0 at a
flowline's first vertex to 1 at its last, which assumes flowlines are
digitized downstream; digitized_downstream() checks that). One Dijkstra
search per dam each way replaces one shortest_path per pair.

Comparison with the old links (data/processed/dam_station_links_<huc8>.csv):
a pair "in window" is connected within the case study's exposure limits
(max_downstream_km / max_upstream_km), as build_views.removal_link defines
it. Each pair in either window gets a change:

    kept                 in both windows (relationship_changed and
                         distance_change_km say how much it moved)
    gained               in the new window only
    now_out_of_window    old window; usable now but farther / other side
    now_unconnected      old window; both endpoints have a reach now, but
                         the new reaches aren't connected
    now_blocked          old window; an endpoint isn't accepted now
                         (blocker says which and why)
    blocked              in the new window but not usable, and not in the
                         old window
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import shapely

from etl.common import reach_selection as rs

LINKED = ("upstream", "downstream", "same_reach")

ENDPOINT_COLUMNS = ["feature_id", "status", "reason", "reach_id", "fraction"]

LINK_COLUMNS = [
    "dam_id", "station_id", "dam_reach_id", "station_reach_id", "relationship",
    "network_distance_km", "distance_method", "dam_fraction", "station_fraction",
    "dam_status", "dam_reason", "station_status", "station_reason", "usable",
    "station_in_mapped_reservoir",
]

CHANGES = ["kept", "gained", "now_out_of_window", "now_unconnected", "now_blocked", "blocked"]


def endpoints(frame, id_column, status_column, reason_column):
    """One endpoint row per feature (see the module docstring). `frame`
    needs NHDPlusID, snap_fraction, provisional_nhdplus_id and
    provisional_fraction."""
    frame = frame.copy()
    for column in ("NHDPlusID", "provisional_nhdplus_id"):
        if column not in frame:
            frame[column] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for column in ("snap_fraction", "provisional_fraction"):
        if column not in frame:
            frame[column] = np.nan
    status = frame[status_column]
    accepted = status.eq(rs.ACCEPTED)
    ambiguous = status.eq(rs.AMBIGUOUS)
    reach = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    reach[accepted] = frame.loc[accepted, "NHDPlusID"].astype("Int64")
    reach[ambiguous] = frame.loc[ambiguous, "provisional_nhdplus_id"].astype("Int64")
    fraction = pd.Series(np.nan, index=frame.index)
    fraction[accepted] = pd.to_numeric(frame.loc[accepted, "snap_fraction"], errors="coerce")
    fraction[ambiguous] = pd.to_numeric(frame.loc[ambiguous, "provisional_fraction"],
                                        errors="coerce")
    return pd.DataFrame({"feature_id": frame[id_column].astype(str), "status": status,
                         "reason": frame[reason_column], "reach_id": reach,
                         "fraction": fraction})[ENDPOINT_COLUMNS].reset_index(drop=True)


def provisional_fractions(frame, lines_by_id):
    """snap_fraction of each row's reported point (x_m, y_m, in the CRS of
    lines_by_id) on its provisional_nhdplus_id; NaN without one, or when the
    reach isn't in lines_by_id."""
    out = pd.Series(np.nan, index=frame.index)
    if not {"provisional_nhdplus_id", "x_m", "y_m"} <= set(frame.columns):
        return out
    reach = frame["provisional_nhdplus_id"].astype("Int64")
    ok = reach.notna() & reach.isin(lines_by_id.index)
    ok &= frame["x_m"].notna() & frame["y_m"].notna()
    if not ok.any():
        return out
    points = shapely.points(frame.loc[ok, "x_m"].astype(float).to_numpy(),
                            frame.loc[ok, "y_m"].astype(float).to_numpy())
    fractions, _, _ = rs.project_onto(lines_by_id.geometry.loc[reach[ok].to_numpy()].to_numpy(),
                                      points)
    out[ok] = fractions
    return out


def _distance(relationship, dam_fraction, station_fraction, dam_length, station_length,
              path_km):
    """Along-network km between the two projected points; path_km is the
    Dijkstra distance (sum of the lengths of every reach on the path but the
    last)."""
    if relationship == "upstream":
        # station -> ... -> dam; path_km starts at the station's reach.
        return ((1 - station_fraction) * station_length + (path_km - station_length)
                + dam_fraction * dam_length)
    # dam -> ... -> station; path_km starts at the dam's reach.
    return ((1 - dam_fraction) * dam_length + (path_km - dam_length)
            + station_fraction * station_length)


def link_pairs(dams, stations, graph, length_by_reach, max_km=None, impounded=None):
    """Connected (dam, station) pairs between endpoint tables (endpoints()).

    graph / length_by_reach come from etl.common.network.build_reach_graph.
    max_km stops each search that far along the network (None: no limit),
    so pairs beyond it are left out. impounded maps dam_id to the set of
    its reservoir's paths (station_in_mapped_reservoir). Endpoints without
    a reach, or whose reach isn't in the graph, get no pairs.
    """
    impounded = impounded or {}
    stations = stations[stations["reach_id"].notna()]
    by_reach = {int(reach): group for reach, group in stations.groupby("reach_id")}
    reverse = graph.reverse(copy=False)
    # The search distance counts whole reaches (the dam's going down, the
    # station's going up), so search one reach farther than max_km and drop
    # pairs beyond it once the fractions are applied.
    longest_station = max((float(length_by_reach.get(r, 0) or 0) for r in by_reach), default=0)
    rows = []
    for dam in dams.itertuples(index=False):
        if pd.isna(dam.reach_id) or int(dam.reach_id) not in graph:
            continue
        reach = int(dam.reach_id)
        dam_length = float(length_by_reach.get(reach, np.nan) or np.nan)
        down = nx.single_source_dijkstra_path_length(
            graph, reach, weight="weight",
            cutoff=None if max_km is None else max_km + (dam_length if dam_length > 0 else 0))
        up = nx.single_source_dijkstra_path_length(
            reverse, reach, weight="weight",
            cutoff=None if max_km is None else max_km + longest_station)
        paths = impounded.get(dam.feature_id, set())
        for station_reach, group in by_reach.items():
            station_length = float(length_by_reach.get(station_reach, np.nan) or np.nan)
            for station in group.itertuples(index=False):
                fd, fs = dam.fraction, station.fraction
                if station_reach == reach:
                    if pd.isna(fd) or pd.isna(fs) or fd == fs:
                        relationship = "same_reach"
                        distance = 0.0 if fd == fs else np.nan
                    else:
                        relationship = "upstream" if fs < fd else "downstream"
                        distance = abs(fd - fs) * dam_length
                    method = "same_reach_fraction"
                elif station_reach in up:
                    relationship = "upstream"
                    distance = _distance(relationship, fd, fs, dam_length, station_length,
                                         up[station_reach])
                    method = "network_path_fractional_endpoints"
                elif station_reach in down:
                    relationship = "downstream"
                    distance = _distance(relationship, fd, fs, dam_length, station_length,
                                         down[station_reach])
                    method = "network_path_fractional_endpoints"
                else:
                    continue
                if max_km is not None and not pd.isna(distance) and distance > max_km:
                    continue
                rows.append({
                    "dam_id": dam.feature_id, "station_id": station.feature_id,
                    "dam_reach_id": reach, "station_reach_id": station_reach,
                    "relationship": relationship,
                    "network_distance_km": distance if not pd.isna(distance) else np.nan,
                    "distance_method": method if not pd.isna(distance) else None,
                    "dam_fraction": fd, "station_fraction": fs,
                    "dam_status": dam.status, "dam_reason": dam.reason,
                    "station_status": station.status, "station_reason": station.reason,
                    "usable": dam.status == rs.ACCEPTED and station.status == rs.ACCEPTED,
                    "station_in_mapped_reservoir": station_reach in paths,
                })
    out = pd.DataFrame(rows, columns=LINK_COLUMNS)
    out["dam_reach_id"] = out["dam_reach_id"].astype("Int64")
    out["station_reach_id"] = out["station_reach_id"].astype("Int64")
    return out


def in_window(relationship, distance_km, limits):
    """Boolean Series: connected within the exposure limits (dict with
    max_downstream_km / max_upstream_km, None = no limit). A same_reach
    pair (no side) is in only when neither limit is set, as in
    build_views.removal_link."""
    down, up = limits.get("max_downstream_km"), limits.get("max_upstream_km")
    distance = pd.to_numeric(distance_km, errors="coerce")
    downstream = relationship.eq("downstream") & (True if down is None else distance.le(down))
    upstream = relationship.eq("upstream") & (True if up is None else distance.le(up))
    same = relationship.eq("same_reach") & (down is None and up is None)
    return (downstream | upstream | same).fillna(False).astype(bool)


def old_connected(old):
    """The connected pairs of an old link table, one row per pair (a
    feature snapped to several reaches keeps its shortest link)."""
    old = old[old["relationship"].isin(LINKED)].copy()
    old["dam_id"] = old["dam_id"].astype(str)
    old["station_id"] = old["station_id"].astype(str)
    old["network_distance_km"] = pd.to_numeric(old["network_distance_km"], errors="coerce")
    old = old.sort_values(["dam_id", "station_id", "network_distance_km"], na_position="last")
    return old.drop_duplicates(["dam_id", "station_id"])[
        ["dam_id", "station_id", "relationship", "network_distance_km"]].reset_index(drop=True)


def _blocker(dam_status, dam_reason, station_status, station_reason):
    """Which endpoint stops a pair being used, and why ('dam:ambiguous:...',
    'station:not_in_stage3', ...); None when both are accepted."""
    for kind, status, reason in (("dam", dam_status, dam_reason),
                                 ("station", station_status, station_reason)):
        if pd.isna(status):
            stage = "stage4" if kind == "dam" else "stage3"
            return f"{kind}:not_in_{stage}"
        if status != rs.ACCEPTED:
            return f"{kind}:{status}:{reason}"
    return None


def compare_links(old, new, dams, stations, limits):
    """One row per pair in the old window, the new window, or both (see the
    module docstring). old: an old link table; new: link_pairs() output;
    dams / stations: endpoint tables."""
    old = old_connected(old)
    old["old_in_window"] = in_window(old["relationship"], old["network_distance_km"], limits)
    new = new.copy()
    new["new_in_window"] = in_window(new["relationship"], new["network_distance_km"], limits)
    merged = old.rename(columns={"relationship": "old_relationship",
                                 "network_distance_km": "old_distance_km"}).merge(
        new[["dam_id", "station_id", "relationship", "network_distance_km", "usable",
             "new_in_window", "station_in_mapped_reservoir"]].rename(
            columns={"relationship": "new_relationship",
                     "network_distance_km": "new_distance_km"}),
        on=["dam_id", "station_id"], how="outer")
    for column in ("old_in_window", "new_in_window", "usable"):
        merged[column] = merged[column].astype("boolean").fillna(False).astype(bool)
    merged = merged[merged["old_in_window"] | merged["new_in_window"]].copy()

    d = dams.drop_duplicates("feature_id").set_index("feature_id")
    s = stations.drop_duplicates("feature_id").set_index("feature_id")
    merged["dam_status"] = merged["dam_id"].map(d["status"])
    merged["dam_reason"] = merged["dam_id"].map(d["reason"])
    merged["station_status"] = merged["station_id"].map(s["status"])
    merged["station_reason"] = merged["station_id"].map(s["reason"])
    merged["blocker"] = [_blocker(*row) for row in merged[
        ["dam_status", "dam_reason", "station_status", "station_reason"]].itertuples(index=False)]

    both_reach = (merged["dam_id"].map(d["reach_id"]).notna()
                  & merged["station_id"].map(s["reach_id"]).notna())
    old_in, new_in, usable = merged["old_in_window"], merged["new_in_window"], merged["usable"]
    linked_now = merged["new_relationship"].notna()
    change = np.select(
        [old_in & new_in & usable,
         ~old_in & new_in & usable,
         old_in & usable & ~new_in,
         old_in & ~linked_now & merged["blocker"].isna() & both_reach,
         old_in,
         new_in],
        ["kept", "gained", "now_out_of_window", "now_unconnected", "now_blocked", "blocked"],
        default="")
    merged["change"] = change
    # A pair that is gone only because the new reaches don't connect has
    # no blocker; one whose endpoint lost its reach keeps it.
    merged.loc[merged["change"].isin(["kept", "gained", "now_out_of_window",
                                      "now_unconnected"]), "blocker"] = None
    merged["relationship_changed"] = (merged["change"] == "kept") & (
        merged["old_relationship"] != merged["new_relationship"])
    merged["distance_change_km"] = (merged["new_distance_km"] - merged["old_distance_km"]).where(
        merged["change"] == "kept")
    return merged.reset_index(drop=True)


def digitized_downstream(lines_by_id, edges, reach_ids):
    """Check snap_fraction's direction assumption on the given reaches:
    for each with a downstream flowline in lines_by_id (metric CRS), True
    when its last vertex is nearer that flowline than its first. Returns a
    Series indexed by reach id."""
    results = {}
    edges = edges.dropna(subset=["to"])
    below = edges.groupby("from")["to"].first()
    for reach in reach_ids:
        reach = int(reach)
        target = below.get(reach)
        if target is None or reach not in lines_by_id.index or int(target) not in lines_by_id.index:
            continue
        coords = shapely.get_coordinates(lines_by_id.geometry.loc[reach])
        if len(coords) < 2:
            continue
        line = lines_by_id.geometry.loc[int(target)]
        first = shapely.distance(shapely.points(coords[0]), line)
        last = shapely.distance(shapely.points(coords[-1]), line)
        if first == last:
            continue
        results[reach] = bool(last < first)
    return pd.Series(results, dtype=bool)
