"""Matching dams and stations to NHDPlus HR reaches (docs/warehouse_spec.md,
section 3.12). Used by snap_features.py.

Steps for each point:

1. Datum. The reported coordinates are moved from their reported horizontal
   datum to the flowlines' CRS (NHDPlus HR is NAD83) with the best PROJ
   transformation for the area. NAD27 needs the NADCON grid; without it
   PROJ silently falls back to a zero "ballpark" shift (tens of metres off),
   so a missing grid is an error (DatumTransformError), not a fallback.
2. Candidates. Every flowline within the search radius. Only stream types
   (StreamRiver, ArtificialPath) can be chosen; canals, connectors,
   coastline, pipelines and conduits never are.
3. Evidence. Distance, river name (station name or the dam's River vs the
   reach's GNIS name), drainage area (the station's WQP-reported area vs
   NHDPlus TotDASqKm) and nearby reaches of very different size
   (confluences).
4. Decision. Every point gets snap_status accepted / ambiguous / rejected
   and a snap_reason code (decide_reach). Only accepted points go on to
   linking; the rest stay in the diagnostics for review.
5. Manual overrides (config/snap_overrides.csv) replace the decision for
   reviewed points.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from pyproj.aoi import AreaOfInterest
from pyproj.transformer import TransformerGroup

# --------------------------------------------------
# Flowline types (NHDPlus HR NHDFlowline.FType)
# --------------------------------------------------

# A point can only be matched to these: a stream, or a river's path
# through a lake or wide river (ArtificialPath).
STREAM_FTYPES = {460: "StreamRiver", 558: "ArtificialPath"}
ARTIFICIAL_PATH = 558
# Named for the diagnostics; any other type is also unsuitable.
UNSUITABLE_FTYPES = {
    334: "Connector",
    336: "CanalDitch",
    420: "UndergroundConduit",
    428: "Pipeline",
    566: "Coastline",
}


@dataclass(frozen=True)
class SnapRules:
    """Distance limits and evidence thresholds for one kind of point."""

    # Reaches farther than this are never candidates: rejected.
    search_radius_m: float
    # Beyond this, a match needs a name or drainage-area match (or an
    # ArtificialPath through wide water) to be accepted.
    review_distance_m: float
    # Two candidates within this many metres of each other are "as close".
    tie_margin_m: float = 25.0
    # A reach as close but this many times bigger or smaller: confluence.
    confluence_area_ratio: float = 5.0
    # Reported / NHDPlus drainage area inside this range agrees. NHDPlus
    # gives the area at the reach's downstream end, so it is wide.
    area_agree: tuple = (0.5, 2.0)


# Dams are usually reported on the channel; the preprocessing audit
# reviewed dams moved more than 150 m. Stations on wide rivers can be
# several hundred metres from the centerline (ArtificialPath).
DAM_RULES = SnapRules(search_radius_m=500, review_distance_m=150)
STATION_RULES = SnapRules(search_radius_m=1000, review_distance_m=300)

# Manual overrides may point to a reach up to this far from the point.
OVERRIDE_MAX_M = 2000

ACCEPTED, AMBIGUOUS, REJECTED = "accepted", "ambiguous", "rejected"


def rules_from_config(cfg):
    """DAM_RULES / STATION_RULES with any values from the config's
    optional `snapping: {dams: {...}, stations: {...}}` block."""
    block = cfg.get("snapping", {}) or {}
    return (
        SnapRules(**{**DAM_RULES.__dict__, **(block.get("dams") or {})}),
        SnapRules(**{**STATION_RULES.__dict__, **(block.get("stations") or {})}),
    )


# --------------------------------------------------
# Datums
# --------------------------------------------------

# WQP HorizontalCoordinateReferenceSystemDatumName -> EPSG geographic CRS.
DATUM_EPSG = {
    "NAD83": 4269,
    "NAD27": 4267,
    "WGS84": 4326,
    "WGS72": 4322,
    "OLDHI": 4135,   # Old Hawaiian
    "PR": 4139,      # Puerto Rico
    "SGEOR": 4138,   # St. George Island
    "SLAWR": 4136,   # St. Lawrence Island
    "SPAUL": 4137,   # St. Paul Island
    "GUAM": 4675,    # Guam 1963
    "AMSMA": 4169,   # American Samoa 1962
    "JOHNS": 4725,   # Johnston Island 1961
    "WAKE": 4733,    # Wake Island 1952
}
# Unknown or unstated: assumed NAD83 (the NHDPlus datum) and flagged.
UNKNOWN_DATUMS = {"", "UNKWN", "OTHER", "UNKNOWN"}
ASSUMED_DATUM_EPSG = 4269

# A transformation less accurate than this (metres) with a better one
# available only through a missing grid is an error.
MAX_DATUM_ERROR_M = 10.0


class DatumTransformError(RuntimeError):
    """A needed PROJ transformation (grid file) is not installed."""


def resolve_datum(name):
    """(EPSG code or None if unsupported, assumed?) for a WQP datum name."""
    key = "" if name is None or pd.isna(name) else str(name).strip().upper()
    if key in UNKNOWN_DATUMS:
        return ASSUMED_DATUM_EPSG, True
    return DATUM_EPSG.get(key), False


def _is_ballpark(operation):
    return "ballpark" in operation.description.lower() or operation.accuracy is None \
        or operation.accuracy < 0


def choose_transformer(group, source, target):
    """The transformer to use from a pyproj TransformerGroup, or None if
    PROJ knows only a ballpark (no real) transformation.

    Raises DatumTransformError when the best usable transformation is a
    ballpark or worse than MAX_DATUM_ERROR_M while a grid-based one exists
    but its grid isn't installed.
    """
    usable = group.transformers[0] if group.transformers else None
    usable_ok = usable is not None and not _is_ballpark(usable) \
        and usable.accuracy <= MAX_DATUM_ERROR_M
    better = [
        op for op in group.unavailable_operations
        if op.accuracy is not None and op.accuracy >= 0
        and (usable is None or _is_ballpark(usable) or op.accuracy < usable.accuracy)
    ]
    if not usable_ok and better:
        grids = sorted({g.short_name for op in better for g in op.grids if not g.available})
        raise DatumTransformError(
            f"Transforming {source} to {target} needs PROJ grid(s) {grids}, which are not "
            "installed; without them PROJ would use a zero or approximate shift. Install "
            f"them once with `projsync --file {grids[0]}` (repeat per grid) or allow PROJ "
            "to fetch them with `export PROJ_NETWORK=ON`."
        )
    if usable is None or _is_ballpark(usable):
        return None
    return usable


def datum_transformer(source_epsg, target_crs, bounds):
    """Best transformer from source_epsg to target_crs over bounds
    (lon/lat west, south, east, north); None if only a ballpark exists."""
    source, target = CRS.from_epsg(source_epsg), CRS.from_user_input(target_crs)
    if source.equals(target):
        return Transformer.from_crs(source, target, always_xy=True)
    group = TransformerGroup(
        source, target, always_xy=True, area_of_interest=AreaOfInterest(*bounds)
    )
    return choose_transformer(group, source.name, target.name)


def transform_points(lon, lat, datum_names, target_crs, bounds, assumed_epsg=None):
    """Reported lon/lat in each point's datum -> x/y in target_crs.

    `datum_names` gives each point's WQP datum name (None for all when the
    source has no datum field; `assumed_epsg` is then used and flagged).
    Returns a DataFrame (same index) with x, y, datum_epsg, datum_assumed,
    datum_transform, datum_accuracy_m; x/y are NaN where no real
    transformation exists (datum_transform 'unsupported').
    """
    lon = pd.Series(lon, dtype="float64")
    lat = pd.Series(lat, dtype="float64", index=lon.index)
    if datum_names is None:
        resolved = [(assumed_epsg or ASSUMED_DATUM_EPSG, True)] * len(lon)
    else:
        resolved = [resolve_datum(name) for name in datum_names]

    out = pd.DataFrame(index=lon.index)
    out["datum_epsg"] = pd.array([epsg for epsg, _ in resolved], dtype="Int32")
    out["datum_assumed"] = [assumed for _, assumed in resolved]
    out["x"] = np.nan
    out["y"] = np.nan
    out["datum_transform"] = "unsupported"
    out["datum_accuracy_m"] = np.nan

    for epsg in out["datum_epsg"].dropna().unique():
        rows = out.index[out["datum_epsg"] == epsg]
        transformer = datum_transformer(int(epsg), target_crs, bounds)
        if transformer is None:
            continue
        x, y = transformer.transform(lon.loc[rows].to_numpy(), lat.loc[rows].to_numpy())
        out.loc[rows, "x"] = x
        out.loc[rows, "y"] = y
        out.loc[rows, "datum_transform"] = transformer.description
        accuracy = transformer.accuracy
        out.loc[rows, "datum_accuracy_m"] = 0.0 if accuracy is None or accuracy < 0 else accuracy
    return out


# --------------------------------------------------
# Names
# --------------------------------------------------

# Words that say what kind of water body or where on it, not which one.
GENERIC_NAME_WORDS = {
    "river", "r", "riv", "creek", "cr", "crk", "ck", "brook", "bk", "br", "stream", "str",
    "run", "the", "of", "at", "near", "nr", "above", "abv", "ab", "below", "blw", "bl",
    "upstream", "downstream", "us", "ds", "and", "trib", "tributary", "to", "site", "station",
}


def name_tokens(name):
    """Lowercase alphanumeric tokens of a name, minus GENERIC_NAME_WORDS."""
    if name is None or (isinstance(name, float) and math.isnan(name)) or pd.isna(name):
        return set()
    return {t for t in re.findall(r"[a-z0-9]+", str(name).lower()) if t not in GENERIC_NAME_WORDS}


def name_matches(station_name, reach_name):
    """True if every distinctive word of the reach's GNIS name appears in the
    station name; None when the reach has no name to compare."""
    reach = name_tokens(reach_name)
    if not reach:
        return None
    return reach <= name_tokens(station_name)


# A station name naming a lake, marsh, bay, ... (refined rule from the
# preprocessing audit, section 11): flagged unless the name also places the
# site on a stream or crossing ("MAXWELL POND-OUTLET", "KNOX MARSH RD BRIDGE").
WATER_BODY_WORDS = (r"\b(?:marsh|estuar\w*|bay|tidal|lake|pond|reservoir|impoundment|harbou?r|"
                    r"slough|lagoon|bayou|canal|ditch)\b")
RIVERINE_CONTEXT_WORDS = (r"\b(?:inlet|outlet|outflow|brook|creek|cr|crk|river|riv|stream|run|kill|"
                          r"trib\w*|branch|br|below|blw|bl|above|abv|ab|ds|us|dam|spillway|bridge|"
                          r"road|rd|rte|route|hwy|highway|street|st|ave)\b")


def non_riverine_name(names):
    """True where a station name suggests a lake, marsh, estuary, ... site."""
    text = pd.Series(names, dtype="string").fillna("").str.lower()
    return (text.str.contains(WATER_BODY_WORDS, regex=True)
            & ~text.str.contains(RIVERINE_CONTEXT_WORDS, regex=True)).astype(bool)


# --------------------------------------------------
# Reported drainage area (WQP station file)
# --------------------------------------------------

AREA_TO_KM2 = {"sqmi": 2.58999, "mi2": 2.58999, "sqkm": 1.0, "km2": 1.0,
               "acre": 0.00404686, "acres": 0.00404686, "ha": 0.01}
AREA_COLUMNS = [
    ("DrainageAreaMeasure/MeasureValue", "DrainageAreaMeasure/MeasureUnitCode"),
    ("ContributingDrainageAreaMeasure/MeasureValue",
     "ContributingDrainageAreaMeasure/MeasureUnitCode"),
]


def reported_area_km2(stations):
    """Agency-reported drainage area in km2 (contributing area as fallback);
    NaN where absent, non-positive or in an unknown unit."""
    area = pd.Series(np.nan, index=stations.index)
    for value_col, unit_col in AREA_COLUMNS:
        if value_col not in stations.columns or unit_col not in stations.columns:
            continue
        unit = stations[unit_col].astype("string").fillna("").str.lower() \
            .str.replace(r"[\s.]", "", regex=True)
        value = pd.to_numeric(stations[value_col], errors="coerce") * unit.map(AREA_TO_KM2) \
            .astype("float64")
        area = area.fillna(value.where(value > 0))
    return area


# --------------------------------------------------
# Candidates and decision
# --------------------------------------------------

def valid_area(series):
    """NHDPlus areas with the -9999 (and any non-positive) placeholder as NaN."""
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.where(values > 0)


def valid_slope(series):
    """NHDPlus slopes with the -9998 (and any negative) placeholder as NaN."""
    values = pd.to_numeric(series, errors="coerce").astype("float64")
    return values.where(values >= 0)


def find_candidates(points, streams, radius_m):
    """Every (point, reach) pair within radius_m, both in the same metric CRS.

    `points` is a GeoSeries; `streams` a GeoDataFrame with NHDPlusID,
    GNIS_Name, FType, totdasqkm. Returns a DataFrame with point_index,
    stream_index, distance_m and the reach attributes.
    """
    columns = ["point_index", "stream_index", "distance_m", "NHDPlusID", "GNIS_Name",
               "FType", "totdasqkm"]
    has_point = points.notna() & ~points.is_empty
    valid = points[has_point]
    if valid.empty or streams.empty:
        return pd.DataFrame(columns=columns)
    # Bounding-box query (works on every geopandas version), then exact distance.
    point_pos, stream_pos = streams.sindex.query(valid.buffer(radius_m).envelope.values)
    pairs = pd.DataFrame({
        "point_index": valid.index[point_pos],
        "stream_index": streams.index[stream_pos],
    })
    pairs["distance_m"] = valid.loc[pairs["point_index"]].distance(
        streams.geometry.loc[pairs["stream_index"]], align=False
    ).to_numpy()
    pairs = pairs[pairs["distance_m"] <= radius_m]
    attrs = streams.loc[pairs["stream_index"], ["NHDPlusID", "GNIS_Name", "FType", "totdasqkm"]]
    pairs = pd.concat([pairs.reset_index(drop=True), attrs.reset_index(drop=True)], axis=1)
    return pairs.sort_values(["point_index", "distance_m", "NHDPlusID"]).reset_index(drop=True)


def decide_reach(candidates, name, reported_area, rules):
    """Choose a reach for one point from its candidates (rows of
    find_candidates for that point).

    Returns a dict: status, reason, evidence (list of codes), stream_index
    (None if nothing chosen; for ambiguous points the provisional nearest
    eligible reach), plus diagnostics (runner-up, nearest unsuitable reach).
    """
    result = {"status": REJECTED, "reason": None, "evidence": [], "stream_index": None,
              "n_candidates": len(candidates), "nearest_any_ftype": None,
              "nearest_any_distance_m": np.nan, "near_unsuitable_flowline": False,
              "runner_up_id": None, "runner_up_distance_m": np.nan, "runner_up_name": None,
              "runner_up_area_km2": np.nan, "area_ratio": np.nan}
    if candidates.empty:
        result["reason"] = "no_flowline_within_limit"
        return result

    ftype = pd.to_numeric(candidates["FType"], errors="coerce")
    nearest_any = candidates.iloc[0]
    result["nearest_any_ftype"] = None if pd.isna(ftype.iloc[0]) else int(ftype.iloc[0])
    result["nearest_any_distance_m"] = float(nearest_any["distance_m"])

    eligible = candidates[ftype.isin(list(STREAM_FTYPES))].copy()
    if eligible.empty:
        result["reason"] = "only_unsuitable_flowline"
        result["near_unsuitable_flowline"] = True
        return result
    # A canal, coastline, ... clearly closer than the reach chosen below.
    result["near_unsuitable_flowline"] = bool(
        ftype.iloc[0] not in STREAM_FTYPES
        and nearest_any["distance_m"] + rules.tie_margin_m < eligible["distance_m"].iloc[0]
    )

    # Ties to 0.1 m (e.g. a point on the node between two reaches): the
    # bigger reach (the main stem), then the smaller ID, so runs are
    # deterministic.
    eligible = eligible.assign(_rounded=eligible["distance_m"].round(1),
                               _area=eligible["totdasqkm"].fillna(-1))
    eligible = eligible.sort_values(["_rounded", "_area", "NHDPlusID"],
                                    ascending=[True, False, True])
    nearest = eligible.iloc[0]
    evidence = []

    # Drainage area (stations with an agency-reported area).
    pool = eligible
    if not pd.isna(reported_area) and reported_area > 0:
        ratio = eligible["totdasqkm"] / reported_area
        agrees = ratio.between(*rules.area_agree)
        # As for names: area can move a point to another reach only within
        # the review distance. A small reach's TotDASqKm is taken at its
        # outlet, so a point on it can disagree 2x and still be right.
        reachable = (eligible["distance_m"] <= rules.review_distance_m) | (
            eligible.index == nearest.name)
        if (agrees & reachable).any():
            pool = eligible[agrees & reachable]
            evidence.append("area_agrees")
        elif agrees.any():
            result.update(status=AMBIGUOUS, reason="drainage_area_far",
                          stream_index=nearest["stream_index"],
                          area_ratio=float(ratio.loc[nearest.name]))
            result["evidence"] = ["area_mismatch", "area_agrees_far"]
            return _with_runner_up(result, eligible, nearest)
        elif ratio.notna().any():
            result.update(status=AMBIGUOUS, reason="drainage_area_mismatch",
                          stream_index=nearest["stream_index"],
                          area_ratio=float(ratio.loc[nearest.name]))
            result["evidence"] = ["area_mismatch"]
            return _with_runner_up(result, eligible, nearest)

    # River name.
    matches = [name_matches(name, reach) for reach in pool["GNIS_Name"]]
    named = pool[[m is True for m in matches]]
    chosen = pool.iloc[0]
    reason = "drainage_area_match" if "area_agrees" in evidence else "nearest_reach"
    if not named.empty:
        best_named = named.iloc[0]
        if best_named.name == chosen.name:
            evidence.append("name_match")
            if reason == "nearest_reach":
                reason = "name_match"
        elif best_named["distance_m"] <= rules.review_distance_m:
            chosen = best_named
            evidence += ["name_match", "named_reach_preferred"]
            reason = "named_reach_preferred"
        else:
            result.update(status=AMBIGUOUS, reason="named_reach_far",
                          stream_index=chosen["stream_index"], evidence=evidence)
            return _with_runner_up(result, eligible, chosen)
    elif matches and matches[0] is False:
        evidence.append("name_mismatch")

    supported = "name_match" in evidence or "area_agrees" in evidence
    if not pd.isna(reported_area) and chosen["totdasqkm"] > 0:
        result["area_ratio"] = float(chosen["totdasqkm"] / reported_area)

    # Confluence: another reach about as close but of very different size.
    others = eligible[(eligible.index != chosen.name)
                      & (eligible["distance_m"] <= chosen["distance_m"] + rules.tie_margin_m)]
    size = others["totdasqkm"] / chosen["totdasqkm"] if chosen["totdasqkm"] > 0 else pd.Series(
        dtype="float64")
    confluence = bool(((size >= rules.confluence_area_ratio)
                       | (size <= 1 / rules.confluence_area_ratio)).any())
    if confluence:
        evidence.append("confluence_nearby")
        if not supported:
            result.update(status=AMBIGUOUS, reason="confluence_ambiguous",
                          stream_index=chosen["stream_index"], evidence=evidence)
            return _with_runner_up(result, eligible, chosen)

    # Distance.
    if chosen["distance_m"] > rules.review_distance_m:
        evidence.append("far_snap")
        if not supported:
            if int(chosen["FType"]) == ARTIFICIAL_PATH:
                reason = "wide_water_far_snap"
            else:
                result.update(status=AMBIGUOUS, reason="far_without_support",
                              stream_index=chosen["stream_index"], evidence=evidence)
                return _with_runner_up(result, eligible, chosen)

    result.update(status=ACCEPTED, reason=reason, stream_index=chosen["stream_index"],
                  evidence=evidence)
    return _with_runner_up(result, eligible, chosen)


def _with_runner_up(result, eligible, chosen):
    others = eligible[eligible.index != chosen.name]
    if not others.empty:
        runner = others.iloc[0]
        result.update(runner_up_id=int(runner["NHDPlusID"]),
                      runner_up_distance_m=float(runner["distance_m"]),
                      runner_up_name=None if pd.isna(runner["GNIS_Name"]) else runner["GNIS_Name"],
                      runner_up_area_km2=float(runner["totdasqkm"]))
    return result


# --------------------------------------------------
# Manual overrides
# --------------------------------------------------

OVERRIDE_COLUMNS = ["feature_kind", "feature_id", "action", "nhdplus_id", "reason",
                    "reviewed_by", "reviewed_on"]
OVERRIDE_KINDS = {"dam", "station"}
OVERRIDE_ACTIONS = {"accept", "reject"}


def load_overrides(path):
    """Validated manual overrides, indexed by (feature_kind, feature_id).

    accept: use reach nhdplus_id (checked against the flowlines when
    applied). reject: drop the point from analysis. Every row needs a
    reason. Raises ValueError listing every problem.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS).set_index(["feature_kind", "feature_id"])
    frame = pd.read_csv(path, dtype="string", comment="#").apply(lambda s: s.str.strip())
    missing = sorted(set(OVERRIDE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    errors = []
    for number, row in frame.iterrows():
        where = f"{path.name} row {number + 2}"
        if row["feature_kind"] not in OVERRIDE_KINDS:
            errors.append(f"{where}: feature_kind {row['feature_kind']!r} not in {sorted(OVERRIDE_KINDS)}")
        if row["action"] not in OVERRIDE_ACTIONS:
            errors.append(f"{where}: action {row['action']!r} not in {sorted(OVERRIDE_ACTIONS)}")
        if pd.isna(row["feature_id"]) or not row["feature_id"]:
            errors.append(f"{where}: feature_id is empty")
        if pd.isna(row["reason"]) or not row["reason"]:
            errors.append(f"{where}: reason is empty")
        if row["action"] == "accept":
            if pd.isna(row["nhdplus_id"]) or not str(row["nhdplus_id"]).isdigit():
                errors.append(f"{where}: accept needs an integer nhdplus_id")
        elif not pd.isna(row["nhdplus_id"]) and row["nhdplus_id"]:
            errors.append(f"{where}: reject must not give an nhdplus_id")
    duplicated = frame.duplicated(["feature_kind", "feature_id"], keep=False)
    if duplicated.any():
        ids = sorted(set(frame.loc[duplicated, "feature_id"]))
        errors.append(f"{path.name}: more than one override for {ids}")
    if errors:
        raise ValueError("Invalid snap overrides:\n  - " + "\n  - ".join(errors))

    frame["nhdplus_id"] = pd.to_numeric(frame["nhdplus_id"]).astype("Int64")
    return frame.set_index(["feature_kind", "feature_id"])
