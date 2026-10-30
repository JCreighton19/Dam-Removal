"""Relationships between NHDPlus HR flowlines, water polygons, catchments
and the flow network, built from stored identifiers (no geometry).

- Water polygon -> flowlines: NHDFlowline.WBArea_Permanent_Identifier
  names the NHDArea (e.g. a wide river, FType 460) or NHDWaterbody (lake,
  pond, reservoir, FType 390; swamp, 466) the flowline runs through.
  Artificial paths (FType 558) are the flowlines drawn through those
  polygons. Polygon identity is Permanent_Identifier, not NHDPlusID.
- Catchment -> flowline: NHDPlusCatchment.NHDPlusID = NHDFlowline.NHDPlusID.
- Burn waterbody -> polygon: NHDPlusBurnWaterbody.NHDPlusID = the polygon's
  NHDPlusID, with OnOffNet (1 = on the flow network).
- Flow: NHDPlusFlow FromNHDPID -> ToNHDPID (0 = no flowline, e.g. a
  network start or end).

Nothing here assumes these relationships are complete: every function
reports what doesn't resolve rather than dropping it, and
relationship_checks lists the exceptions (used by
etl/nhdplus_diagnostics.py, and later by station and dam matching).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ARTIFICIAL_PATH = 558
# Flowline types a station or dam can be matched to (as in snapping.py).
MATCHABLE_FTYPES = (460, 558)
AREA_LAYER = "NHDArea"
WATERBODY_LAYER = "NHDWaterbody"

# wbarea_status values
NO_POLYGON = "none"
RESOLVED = "resolved"
UNRESOLVED = "unresolved"


def clean_permanent_id(series):
    """Permanent identifiers as stripped, upper-case strings; blank -> NA.
    (GUID-style identifiers can differ in case and braces between layers.)"""
    text = series.astype("string").str.strip().str.upper().str.strip("{}")
    return text.mask(text == "")


def clean_nhdplus_id(series):
    """NHDPlusIDs as Int64; missing, non-numeric and 0 -> NA."""
    ids = pd.to_numeric(series, errors="coerce")
    return ids.where(ids > 0).round().astype("Int64")


def water_polygons(areas, waterbodies):
    """One table of the NHDArea and NHDWaterbody polygons: permanent_id,
    polygon_nhdplusid, polygon_layer, polygon_ftype, polygon_fcode."""
    frames = []
    for layer, frame in ((AREA_LAYER, areas), (WATERBODY_LAYER, waterbodies)):
        frames.append(pd.DataFrame({
            "permanent_id": clean_permanent_id(frame["Permanent_Identifier"]),
            "polygon_nhdplusid": clean_nhdplus_id(frame["NHDPlusID"]),
            "polygon_layer": layer,
            "polygon_ftype": pd.to_numeric(frame["FType"], errors="coerce").astype("Int64"),
            "polygon_fcode": pd.to_numeric(frame.get("FCode"), errors="coerce").astype("Int64")
            if "FCode" in frame else pd.Series(pd.NA, index=frame.index, dtype="Int64"),
        }))
    return pd.concat(frames, ignore_index=True)


def link_flowlines_to_polygons(flowlines, polygons):
    """Flowlines with the water polygon their WBArea_Permanent_Identifier
    names: wbarea_id, wbarea_status (none / resolved / unresolved),
    polygon_layer, polygon_ftype. A polygon identifier found in both layers
    (or twice) is resolved to its first row and reported by
    relationship_checks."""
    out = flowlines.copy()
    out["wbarea_id"] = clean_permanent_id(out["WBArea_Permanent_Identifier"])
    first = (polygons.dropna(subset=["permanent_id"]).drop_duplicates("permanent_id")
             .set_index("permanent_id"))
    # map, not merge: pandas can't merge an empty string key column.
    out["polygon_layer"] = out["wbarea_id"].map(first["polygon_layer"]).astype("object")
    out["polygon_ftype"] = out["wbarea_id"].map(first["polygon_ftype"]).astype("Int64")
    out["wbarea_status"] = np.select(
        [out["wbarea_id"].isna(), out["polygon_layer"].notna()],
        [NO_POLYGON, RESOLVED], UNRESOLVED)
    return out


def polygon_path_counts(linked, polygons, burn=None):
    """Polygons with n_flowlines and n_artificial_paths linked to them,
    and on_off_network from NHDPlusBurnWaterbody (1 / 0 / NA if absent)."""
    resolved = linked[linked["wbarea_status"] == RESOLVED]
    counts = resolved.groupby("wbarea_id").agg(
        n_flowlines=("NHDPlusID", "size"),
        n_artificial_paths=("FType", lambda f: int((f == ARTIFICIAL_PATH).sum())))
    out = polygons.copy()
    for column in ("n_flowlines", "n_artificial_paths"):
        out[column] = out["permanent_id"].map(counts[column]).fillna(0).astype("int64")
    out["on_off_network"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    if burn is not None and len(burn):
        flag = (pd.DataFrame({"id": clean_nhdplus_id(burn["NHDPlusID"]),
                              "flag": pd.to_numeric(burn["OnOffNet"], errors="coerce")})
                .dropna(subset=["id"]).drop_duplicates("id").set_index("id")["flag"])
        out["on_off_network"] = out["polygon_nhdplusid"].map(flag).astype("Int64")
    return out


def relationship_checks(flowlines, areas, waterbodies, catchments, vaa, flow, burn=None):
    """Check every stored relationship of one package.

    Inputs are attribute tables with canonical column names
    (etl.common.nhdplus.normalize_nhdplus_columns). `burn` may be None or
    empty for packages without NHDPlusBurnWaterbody.

    Returns (counts, polygon_types, exceptions):
      counts         dict of totals for the package summary
      polygon_types  one row per (polygon_layer, polygon_ftype): polygons,
                     linked to >= 1 artificial path, on-network, paths
      exceptions     one row per problem record: kind, record_id, detail
    """
    flowlines = flowlines.assign(NHDPlusID=clean_nhdplus_id(flowlines["NHDPlusID"]),
                                 FType=pd.to_numeric(flowlines["FType"], errors="coerce"))
    polygons = water_polygons(areas, waterbodies)
    linked = link_flowlines_to_polygons(flowlines, polygons)
    per_polygon = polygon_path_counts(linked, polygons, burn)
    ap = linked["FType"] == ARTIFICIAL_PATH
    matchable = linked["FType"].isin(MATCHABLE_FTYPES)
    flowline_ids = set(linked["NHDPlusID"].dropna())
    exceptions = []

    def add(kind, ids, detail=None):
        ids = pd.Series(ids).reset_index(drop=True)
        if len(ids):
            detail = (pd.Series(detail).reset_index(drop=True) if detail is not None
                      else pd.Series(pd.NA, index=ids.index, dtype="string"))
            exceptions.append(pd.DataFrame({"kind": kind, "record_id": ids.astype("string"),
                                            "detail": detail.astype("string")}))

    # Flowlines and their polygons.
    dup = linked["NHDPlusID"].dropna()
    add("duplicate_flowline_nhdplusid", dup[dup.duplicated()].unique())
    add("flowline_without_nhdplusid", linked.index[linked["NHDPlusID"].isna()].astype(str))
    no_poly = linked[ap & (linked["wbarea_status"] == NO_POLYGON)]
    add("artificial_path_without_polygon", no_poly["NHDPlusID"])
    bad = linked[linked["wbarea_status"] == UNRESOLVED]
    add("polygon_id_unresolved", bad["NHDPlusID"], "wbarea_id=" + bad["wbarea_id"])
    other = linked[~ap & (linked["wbarea_status"] != NO_POLYGON)]
    add("non_artificial_path_with_polygon", other["NHDPlusID"],
        "ftype=" + other["FType"].astype("Int64").astype("string")
        + " polygon=" + other["polygon_layer"].fillna("unresolved"))

    # Polygons.
    ids = polygons["permanent_id"].dropna()
    repeated = polygons[polygons["permanent_id"].isin(ids[ids.duplicated()])]
    add("duplicate_polygon_permanent_id", repeated["permanent_id"],
        "layer=" + repeated["polygon_layer"])
    add("polygon_without_permanent_id",
        polygons.loc[polygons["permanent_id"].isna(), "polygon_nhdplusid"])
    known = per_polygon["on_off_network"].notna()
    offnet = per_polygon[known & (per_polygon["on_off_network"] == 0)
                         & (per_polygon["n_artificial_paths"] > 0)]
    add("offnet_polygon_with_paths", offnet["permanent_id"],
        "layer=" + offnet["polygon_layer"] + " ftype="
        + offnet["polygon_ftype"].astype("string"))
    onnet = per_polygon[known & (per_polygon["on_off_network"] == 1)
                        & (per_polygon["n_artificial_paths"] == 0)]
    add("onnet_polygon_without_paths", onnet["permanent_id"],
        "layer=" + onnet["polygon_layer"] + " ftype="
        + onnet["polygon_ftype"].astype("string"))
    if burn is not None and len(burn):
        burn_ids = clean_nhdplus_id(burn["NHDPlusID"])
        add("burn_waterbody_unresolved",
            burn_ids[burn_ids.notna() & ~burn_ids.isin(polygons["polygon_nhdplusid"])])

    # Catchments, VAA.
    catch_ids = clean_nhdplus_id(catchments["NHDPlusID"])
    add("catchment_without_flowline", catch_ids[catch_ids.notna() & ~catch_ids.isin(flowline_ids)])
    no_catch = linked[matchable & ~linked["NHDPlusID"].isin(set(catch_ids.dropna()))]
    add("flowline_without_catchment", no_catch["NHDPlusID"],
        "ftype=" + no_catch["FType"].astype("Int64").astype("string"))
    vaa = vaa.assign(NHDPlusID=clean_nhdplus_id(vaa["NHDPlusID"]))
    no_vaa = linked[matchable & ~linked["NHDPlusID"].isin(set(vaa["NHDPlusID"].dropna()))]
    add("flowline_without_vaa", no_vaa["NHDPlusID"])
    area = pd.to_numeric(vaa.get("TotDASqKm"), errors="coerce") if "TotDASqKm" in vaa else None
    bad_area = (vaa.loc[~(area > 0), "NHDPlusID"] if area is not None
                else pd.Series(dtype="Int64"))
    bad_area = bad_area[bad_area.isin(set(linked.loc[matchable, "NHDPlusID"].dropna()))]
    add("matchable_flowline_invalid_totdasqkm", bad_area)

    # Flow table: ids that aren't flowlines of this package. Some are
    # expected at package boundaries (flow to/from a neighbouring HUC4).
    from_ids, to_ids = clean_nhdplus_id(flow["FromNHDPID"]), clean_nhdplus_id(flow["ToNHDPID"])
    add("flow_from_unknown_flowline", from_ids[from_ids.notna() & ~from_ids.isin(flowline_ids)])
    add("flow_to_unknown_flowline", to_ids[to_ids.notna() & ~to_ids.isin(flowline_ids)])
    in_flow = set(from_ids.dropna()) | set(to_ids.dropna())
    isolated = linked[matchable & ~linked["NHDPlusID"].isin(in_flow)]
    add("matchable_flowline_not_in_flow", isolated["NHDPlusID"])

    exceptions = (pd.concat(exceptions, ignore_index=True) if exceptions
                  else pd.DataFrame(columns=["kind", "record_id", "detail"]))

    status = linked.loc[ap, "wbarea_status"].value_counts()
    target = linked.loc[ap & (linked["wbarea_status"] == RESOLVED), "polygon_layer"].value_counts()
    n_ap = int(ap.sum())
    counts = {
        "n_flowlines": len(linked),
        "n_stream_river_460": int((linked["FType"] == 460).sum()),
        "n_artificial_path_558": n_ap,
        "n_other_ftype": int((~matchable).sum()),
        "n_ap_with_polygon_id": n_ap - int(status.get(NO_POLYGON, 0)),
        "pct_ap_with_polygon_id": round(100 * (n_ap - status.get(NO_POLYGON, 0)) / n_ap, 2)
        if n_ap else np.nan,
        "n_ap_polygon_resolved": int(status.get(RESOLVED, 0)),
        "n_ap_polygon_unresolved": int(status.get(UNRESOLVED, 0)),
        "n_ap_to_nhdarea": int(target.get(AREA_LAYER, 0)),
        "n_ap_to_nhdwaterbody": int(target.get(WATERBODY_LAYER, 0)),
        "n_nhdarea": len(areas),
        "n_nhdwaterbody": len(waterbodies),
        "n_polygons_with_paths": int((per_polygon["n_artificial_paths"] > 0).sum()),
        "n_burn_waterbody": 0 if burn is None else len(burn),
        "n_catchments": len(catchments),
        "n_matchable_without_catchment": len(no_catch),
        "pct_matchable_with_catchment": round(100 * (1 - len(no_catch) / matchable.sum()), 2)
        if matchable.sum() else np.nan,
        "n_flow_rows": len(flow),
    }
    for kind, n in exceptions["kind"].value_counts().items():
        counts[f"exc_{kind}"] = int(n)

    polygon_types = (per_polygon.assign(
        linked=per_polygon["n_artificial_paths"] > 0,
        on_network=per_polygon["on_off_network"] == 1)
        .groupby(["polygon_layer", "polygon_ftype"], dropna=False)
        .agg(n_polygons=("permanent_id", "size"),
             n_linked_to_paths=("linked", "sum"),
             n_on_network=("on_network", "sum"),
             n_on_off_unknown=("on_off_network", lambda s: int(s.isna().sum())),
             total_paths=("n_artificial_paths", "sum"),
             max_paths=("n_artificial_paths", "max"))
        .reset_index())
    return counts, polygon_types, exceptions
