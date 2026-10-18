"""
Load a case study's core warehouse tables into Parquet.

Reads the per-HUC8 outputs the existing pipeline already produces and
writes the five tables specified in docs/warehouse_spec.md (sections
3.1-3.5) to data/warehouse/<case_study>/:

    dam_removal.parquet       GeoParquet: one row per dam, snapped point
    wq_station.parquet        GeoParquet: one row per network-connected station
    dam_station_link.parquet  Parquet: one row per connected (dam, station) pair
    huc8_watershed.parquet    GeoParquet: one row per case-study HUC8 polygon
    streams.parquet           GeoParquet: flowlines up/downstream of the dams

Inputs, required for every HUC8 in the case-study config:

    data/processed/snapped_features_<huc8>.gpkg   (snap_features.py)
    data/processed/dam_station_links_<huc8>.csv   (link_dams_stations.py)
    data/processed/nhdplus_extracted/...gdb       (unzip_nhdplus)
    the config's wbd_source GeoPackage

All five tables are built and validated (spec section 4) in memory before
anything is written, so a failed run leaves the previous warehouse files
untouched. Each file is written to <table>.parquet.tmp and then renamed
into place. The intermediate gpkg/CSV files are only read, never modified.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import MultiPolygon

from etl.common.config import default_case_study, load_config
from etl.common.io import find_column
from etl.common.links import CONNECTED_RELATIONSHIPS, connected_only, load_links
from etl.common.network import build_reach_graph
from etl.common.nhdplus import find_huc8_geodatabase, read_nhdplus_layer
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic
from etl.common.wbd import find_hu8_layer

# --------------------------------------------------
# Configuration
# --------------------------------------------------

WGS84 = "EPSG:4326"
MATCH_METHOD = "nearest_flowline"

# Flowlines are simplified for map rendering only; graph logic always
# reads the raw geodatabase (spec section 3.5).
STREAM_SIMPLIFY_M = 5.0
STREAM_COLUMNS = ["NHDPlusID", "ReachCode", "GNIS_Name", "LengthKM", "FType", "FCode"]

STATION_ID = "MonitoringLocationIdentifier"

# The HUC8 a snapped file belongs to. Kept separate from the dam CSV's own
# "HUC8" column, which holds a watershed name, not a code.
PIPELINE_HUC8 = "pipeline_huc8"

DATE = "date32[pyarrow]"

# Column order and dtype of every output table (spec sections 3.1-3.5).
SCHEMAS = {
    "dam_removal": {
        "AR_ID": "string",
        "NABI_ID": "string",
        "Dam_Name": "string",
        "Year_Removed": "Int16",
        "Latitude": "float64",
        "Longitude": "float64",
        "City_County": "string",
        "River": "string",
        "HUC8": "string",
        "State": "string",
        "Dam_Height_ft": "float64",
        "Dam_Length_ft": "float64",
        "Year_built": "Int16",
        "Original_Use": "string",
        "Type_Material": "string",
        "Reason_for_removal": "string",
        "Miles_Restored_US": "float64",
        "Miles_Restored_DS": "float64",
        "Last_Updated": DATE,
        "NHDPlusID": "int64",
        "snap_distance_m": "float64",
        "snap_fraction": "float64",
        "match_method": "string",
        "geometry": "geometry",
    },
    "wq_station": {
        "station_id": "string",
        "station_name": "string",
        "station_type": "string",
        "organization_id": "string",
        "organization_name": "string",
        "latitude": "float64",
        "longitude": "float64",
        "state_code": "string",
        "county_code": "string",
        "huc8": "string",
        "horizontal_datum": "string",
        "horizontal_method": "string",
        "NHDPlusID": "int64",
        "snap_distance_m": "float64",
        "snap_fraction": "float64",
        "match_method": "string",
        "geometry": "geometry",
    },
    "dam_station_link": {
        "AR_ID": "string",
        "station_id": "string",
        "relationship": "string",
        "network_distance_km": "float64",
        "distance_method": "string",
    },
    "huc8_watershed": {
        "huc8": "string",
        "name": "string",
        "states": "string",
        "areasqkm": "float64",
        "geometry": "geometry",
    },
    "streams": {
        "nhdplus_id": "int64",
        "reach_code": "string",
        "gnis_name": "string",
        "length_km": "float64",
        "feature_type": "Int16",
        "feature_code": "Int32",
        "geometry": "geometry",
    },
}

PRIMARY_KEYS = {
    "dam_removal": ["AR_ID"],
    "wq_station": ["station_id"],
    "dam_station_link": ["AR_ID", "station_id"],
    "huc8_watershed": ["huc8"],
    "streams": ["nhdplus_id"],
}

# Output column -> source column in the dam CSV (carried through the
# dams_snapped layer). The source spells Year_Built with a capital B.
DAM_SOURCE_COLUMNS = {
    "NABI_ID": "NABI_ID",
    "Dam_Name": "Dam_Name",
    "Year_Removed": "Year_Removed",
    "Latitude": "Latitude",
    "Longitude": "Longitude",
    "City_County": "City_County",
    "River": "River",
    "State": "State",
    "Dam_Height_ft": "Dam_Height_ft",
    "Dam_Length_ft": "Dam_Length_ft",
    "Year_built": "Year_Built",
    "Original_Use": "Original_Use",
    "Type_Material": "Type_Material",
    "Reason_for_removal": "Reason_for_removal",
    "Miles_Restored_US": "Miles_Restored_US",
    "Miles_Restored_DS": "Miles_Restored_DS",
    "Last_Updated": "Last_Updated",
}

# Output column -> WQP station column (carried through stations_snapped).
STATION_SOURCE_COLUMNS = {
    "station_name": "MonitoringLocationName",
    "station_type": "MonitoringLocationTypeName",
    "organization_id": "OrganizationIdentifier",
    "organization_name": "OrganizationFormalName",
    "latitude": "LatitudeMeasure",
    "longitude": "LongitudeMeasure",
    "state_code": "StateCode",
    "county_code": "CountyCode",
    "horizontal_datum": "HorizontalCoordinateReferenceSystemDatumName",
    "horizontal_method": "HorizontalCollectionMethodName",
}


# --------------------------------------------------
# Type conversion helpers
# --------------------------------------------------

def _source(frame, name, description):
    """Case-insensitive column lookup that fails loudly."""
    column = find_column(frame.columns, name)
    if column is None:
        raise ValueError(
            f"{description} is missing column '{name}'. "
            f"Available columns: {sorted(frame.columns)}"
        )
    return frame[column]


def to_text(series):
    """Strings with surrounding whitespace removed; blanks become null."""
    text = series.astype("string").str.strip()
    return text.mask(text == "")


def to_float(series):
    return pd.to_numeric(series, errors="coerce").astype("float64")


def to_year(series):
    """Non-numeric or 0 -> null (same rule as report_pilot_eligibility)."""
    years = pd.to_numeric(series, errors="coerce")
    return years.mask(years == 0).round().astype("Int16")


def to_id(series):
    """Numeric IDs (e.g. NHDPlusID, stored as float in the gpkg) as Int64."""
    return pd.to_numeric(series, errors="coerce").round().astype("Int64")


def to_date(series):
    return pd.to_datetime(series, errors="coerce").astype(DATE)


def to_code(series, width):
    """Numeric FIPS-style codes zero-padded to `width`; other values as-is."""
    numeric = pd.to_numeric(series, errors="coerce")
    padded = numeric.round().astype("Int64").astype("string").str.zfill(width)
    return padded.where(numeric.notna(), to_text(series))


# --------------------------------------------------
# Loading intermediate outputs
# --------------------------------------------------

def check_inputs(processed_dir, huc8s):
    """Fail up front, naming every missing file, before reading anything."""
    missing = []
    for huc8 in huc8s:
        for name in (f"snapped_features_{huc8}.gpkg", f"dam_station_links_{huc8}.csv"):
            if not (processed_dir / name).exists():
                missing.append(str(processed_dir / name))
    if missing:
        raise FileNotFoundError(
            "Missing pipeline outputs for this case study (run snap_features "
            f"and link_dams first): {missing}"
        )


def load_snapped_layer(processed_dir, huc8s, layer):
    """Concatenate one layer of every HUC8's snapped_features gpkg."""
    frames = []
    for huc8 in huc8s:
        frame = gpd.read_file(processed_dir / f"snapped_features_{huc8}.gpkg", layer=layer)
        frame[PIPELINE_HUC8] = huc8
        frames.append(frame.to_crs(WGS84))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=WGS84)


def load_all_links(processed_dir, huc8s):
    frames = [load_links(processed_dir / f"dam_station_links_{huc8}.csv") for huc8 in huc8s]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------
# Table builders
# --------------------------------------------------

def drop_unsnapped(frame, id_column, label):
    """Drop features that did not snap to any reach."""
    unsnapped = frame["NHDPlusID"].isna() | frame.geometry.isna()
    if unsnapped.any():
        ids = sorted(frame.loc[unsnapped, id_column].astype(str))
        print(f"{label}: dropped {unsnapped.sum():,} unsnapped feature(s): {ids}")
    return frame.loc[~unsnapped].copy()


def dedupe_snaps(frame, id_column, label):
    """One row per feature: smallest snap distance, then smallest NHDPlusID.

    A point equidistant to two reaches appears once per reach in the
    snapped layers (spec section 3.1, "Duplicate snaps").
    """
    ordered = frame.sort_values(
        [id_column, "snap_distance_m", "NHDPlusID"], kind="mergesort"
    )
    duplicate = ordered.duplicated(id_column, keep="first")
    if duplicate.any():
        print(f"{label}: dropped {duplicate.sum():,} duplicate snap row(s):")
        print(
            ordered.loc[duplicate, [id_column, "NHDPlusID", "snap_distance_m"]]
            .to_string(index=False)
        )
    return ordered.loc[~duplicate].reset_index(drop=True)


def build_dam_removal(dams_snapped):
    dams = dams_snapped.copy()
    dams["AR_ID"] = to_text(_source(dams, "AR_ID", "dams_snapped layer"))
    dams["NHDPlusID"] = to_id(dams["NHDPlusID"])
    dams = drop_unsnapped(dams, "AR_ID", "Dams")
    dams = dedupe_snaps(dams, "AR_ID", "Dams")

    def src(name):
        return _source(dams, name, "dams_snapped layer")

    table = pd.DataFrame({
        "AR_ID": dams["AR_ID"],
        "NABI_ID": to_text(src(DAM_SOURCE_COLUMNS["NABI_ID"])),
        "Dam_Name": to_text(src(DAM_SOURCE_COLUMNS["Dam_Name"])),
        "Year_Removed": to_year(src(DAM_SOURCE_COLUMNS["Year_Removed"])),
        "Latitude": to_float(src(DAM_SOURCE_COLUMNS["Latitude"])),
        "Longitude": to_float(src(DAM_SOURCE_COLUMNS["Longitude"])),
        "City_County": to_text(src(DAM_SOURCE_COLUMNS["City_County"])),
        "River": to_text(src(DAM_SOURCE_COLUMNS["River"])),
        "HUC8": dams[PIPELINE_HUC8].astype("string"),
        "State": to_text(src(DAM_SOURCE_COLUMNS["State"])),
        "Dam_Height_ft": to_float(src(DAM_SOURCE_COLUMNS["Dam_Height_ft"])),
        "Dam_Length_ft": to_float(src(DAM_SOURCE_COLUMNS["Dam_Length_ft"])),
        "Year_built": to_year(src(DAM_SOURCE_COLUMNS["Year_built"])),
        "Original_Use": to_text(src(DAM_SOURCE_COLUMNS["Original_Use"])),
        "Type_Material": to_text(src(DAM_SOURCE_COLUMNS["Type_Material"])),
        "Reason_for_removal": to_text(src(DAM_SOURCE_COLUMNS["Reason_for_removal"])),
        "Miles_Restored_US": to_float(src(DAM_SOURCE_COLUMNS["Miles_Restored_US"])),
        "Miles_Restored_DS": to_float(src(DAM_SOURCE_COLUMNS["Miles_Restored_DS"])),
        "Last_Updated": to_date(src(DAM_SOURCE_COLUMNS["Last_Updated"])),
        "NHDPlusID": dams["NHDPlusID"].astype("int64"),
        "snap_distance_m": to_float(dams["snap_distance_m"]),
        "snap_fraction": to_float(dams["snap_fraction"]),
        "match_method": pd.Series(MATCH_METHOD, index=dams.index, dtype="string"),
    })
    return gpd.GeoDataFrame(table, geometry=dams.geometry, crs=WGS84)


def build_station_candidates(stations_snapped):
    """Every snapped station, one row each. Narrowed to linked stations later."""
    stations = stations_snapped.copy()
    stations["station_id"] = to_text(_source(stations, STATION_ID, "stations_snapped layer"))
    stations["NHDPlusID"] = to_id(stations["NHDPlusID"])
    stations = drop_unsnapped(stations, "station_id", "Stations")
    return dedupe_snaps(stations, "station_id", "Stations")


def build_dam_station_link(links, dam_removal, stations):
    """Connected pairs whose dam and station rows survived deduplication.

    link_dams_stations.py emits one row per snap, so a duplicate-snapped
    dam or station has one link row per candidate reach. Only the row on
    the kept reach is retained.
    """
    connected = connected_only(links)
    connected["AR_ID"] = to_text(connected["dam_id"])
    connected["station_id"] = to_text(connected["station_id"])

    kept_dam_reach = dam_removal.set_index("AR_ID")["NHDPlusID"]
    kept_station_reach = stations.set_index("station_id")["NHDPlusID"]

    dam_matches = (
        connected["AR_ID"].map(kept_dam_reach).astype("Int64")
        == to_id(connected["dam_reach_id"])
    ).fillna(False)
    station_matches = (
        connected["station_id"].map(kept_station_reach).astype("Int64")
        == to_id(connected["station_reach_id"])
    ).fillna(False)

    kept = connected.loc[dam_matches & station_matches]
    print(
        f"Links: {len(links):,} rows, {len(connected):,} connected, "
        f"{len(kept):,} on kept dam/station reaches"
    )

    duplicate = kept.duplicated(["AR_ID", "station_id"], keep="first")
    if duplicate.any():
        print(f"Links: dropped {duplicate.sum():,} repeated (AR_ID, station_id) row(s)")
    kept = kept.loc[~duplicate]

    return pd.DataFrame({
        "AR_ID": kept["AR_ID"],
        "station_id": kept["station_id"],
        "relationship": kept["relationship"].astype("string"),
        "network_distance_km": to_float(kept["network_distance_km"]),
        "distance_method": to_text(kept["distance_method"]),
    }).reset_index(drop=True)


def build_wq_station(stations, dam_station_link):
    """Only stations network-connected to at least one dam (spec 3.2)."""
    linked = stations.loc[stations["station_id"].isin(dam_station_link["station_id"])]
    linked = linked.reset_index(drop=True)

    def src(name):
        return _source(linked, STATION_SOURCE_COLUMNS[name], "stations_snapped layer")

    table = pd.DataFrame({
        "station_id": linked["station_id"],
        "station_name": to_text(src("station_name")),
        "station_type": to_text(src("station_type")),
        "organization_id": to_text(src("organization_id")),
        "organization_name": to_text(src("organization_name")),
        "latitude": to_float(src("latitude")),
        "longitude": to_float(src("longitude")),
        "state_code": to_code(src("state_code"), 2),
        "county_code": to_code(src("county_code"), 3),
        "huc8": linked[PIPELINE_HUC8].astype("string"),
        "horizontal_datum": to_text(src("horizontal_datum")),
        "horizontal_method": to_text(src("horizontal_method")),
        "NHDPlusID": linked["NHDPlusID"].astype("int64"),
        "snap_distance_m": to_float(linked["snap_distance_m"]),
        "snap_fraction": to_float(linked["snap_fraction"]),
        "match_method": pd.Series(MATCH_METHOD, index=linked.index, dtype="string"),
    })
    return gpd.GeoDataFrame(table, geometry=linked.geometry, crs=WGS84)


def build_huc8_watershed(wbd_path, huc8s):
    """The case-study HUC8 polygons, read from the national WBD in one pass."""
    wbd = gpd.read_file(wbd_path, layer=find_hu8_layer(wbd_path))
    codes = _source(wbd, "huc8", "WBD HU8 layer").astype("string").str.zfill(8)
    selected = wbd.loc[codes.isin(huc8s)].copy()
    selected["huc8"] = codes.loc[selected.index]

    missing = sorted(set(huc8s) - set(selected["huc8"]))
    if missing:
        raise ValueError(f"HUC8(s) {missing} not found in {wbd_path}.")

    selected = selected.to_crs(WGS84)
    invalid = ~selected.geometry.is_valid
    if invalid.any():
        print(f"Watersheds: repaired {invalid.sum():,} invalid polygon(s) with make_valid")
        selected.loc[invalid, "geometry"] = selected.loc[invalid].geometry.make_valid()

    geometry = selected.geometry.apply(
        lambda geom: MultiPolygon([geom]) if geom.geom_type == "Polygon" else geom
    )

    table = pd.DataFrame({
        "huc8": selected["huc8"],
        "name": to_text(_source(selected, "name", "WBD HU8 layer")),
        "states": to_text(_source(selected, "states", "WBD HU8 layer")),
        "areasqkm": to_float(_source(selected, "areasqkm", "WBD HU8 layer")),
    }).reset_index(drop=True)
    return gpd.GeoDataFrame(table, geometry=geometry.reset_index(drop=True), crs=WGS84)


def select_stream_reaches(flowlines, flow, dam_reaches, station_reaches, huc8s):
    """Boolean mask over `flowlines` for the reaches `streams` keeps.

    Every reach upstream or downstream of a dam, limited to reaches whose
    ReachCode starts with a case-study HUC8 (spec 3.5). The dams' and
    stations' own snapped reaches are always kept, even across a HUC8
    boundary, so every NHDPlusID FK in the warehouse resolves.
    """
    graph, _, _ = build_reach_graph(flowlines, flow)

    network = set()
    for reach in dam_reaches:
        if reach in graph:
            network |= nx.ancestors(graph, reach) | nx.descendants(graph, reach)

    seeds = set(dam_reaches) | set(station_reaches)
    ids = to_id(flowlines["NHDPlusID"])
    in_case_study = flowlines["ReachCode"].astype("string").str[:8].isin(huc8s)
    return ((ids.isin(network) & in_case_study) | ids.isin(seeds)).fillna(False)


def build_streams(nhd_dir, huc8s, dam_removal, wq_station):
    # HUC8s sharing an HU4 fallback package share one geodatabase; read and
    # graph each geodatabase once.
    groups = defaultdict(list)
    for huc8 in huc8s:
        groups[find_huc8_geodatabase(nhd_dir, huc8)].append(huc8)

    frames = []
    for gdb_path, group in groups.items():
        print(f"Streams: reading {gdb_path.name} for HUC8(s) {group}")
        flowlines = read_nhdplus_layer(gdb_path, "NHDFlowline", columns=STREAM_COLUMNS)
        flow = read_nhdplus_layer(
            gdb_path, "NHDPlusFlow", columns=["FromNHDPID", "ToNHDPID"], ignore_geometry=True
        )

        dam_reaches = dam_removal.loc[dam_removal["HUC8"].isin(group), "NHDPlusID"]
        station_reaches = wq_station.loc[wq_station["huc8"].isin(group), "NHDPlusID"]
        keep = select_stream_reaches(flowlines, flow, dam_reaches, station_reaches, huc8s)
        selected = flowlines.loc[keep]
        print(f"Streams: kept {len(selected):,} of {len(flowlines):,} flowlines")

        if selected.empty:
            continue

        metric_crs = selected.estimate_utm_crs()
        geometry = (
            selected.geometry.force_2d()
            .to_crs(metric_crs)
            .simplify(STREAM_SIMPLIFY_M)
            .to_crs(WGS84)
        )

        table = pd.DataFrame({
            "nhdplus_id": to_id(selected["NHDPlusID"]).astype("int64"),
            "reach_code": to_text(selected["ReachCode"]),
            "gnis_name": to_text(selected["GNIS_Name"]),
            "length_km": to_float(selected["LengthKM"]),
            "feature_type": pd.to_numeric(selected["FType"], errors="coerce").astype("Int16"),
            "feature_code": pd.to_numeric(selected["FCode"], errors="coerce").astype("Int32"),
        })
        frames.append(gpd.GeoDataFrame(table, geometry=geometry, crs=WGS84))

    if not frames:
        empty = {col: pd.Series(dtype=dtype) for col, dtype in SCHEMAS["streams"].items()
                 if col != "geometry"}
        return gpd.GeoDataFrame(empty, geometry=gpd.GeoSeries([], crs=WGS84), crs=WGS84)

    streams = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=WGS84)
    return streams.drop_duplicates("nhdplus_id").reset_index(drop=True)


# --------------------------------------------------
# Validation (spec section 4)
# --------------------------------------------------

def _outside(values, allowed):
    """Distinct non-null values not in `allowed`, for error messages."""
    values = values.dropna()
    return sorted(set(values[~values.isin(allowed)].astype(str)))


def validate_tables(tables, current_year=None):
    """Return a list of every spec violation across all tables (empty = valid)."""
    current_year = current_year or date.today().year
    errors = []

    for name, frame in tables.items():
        expected = SCHEMAS[name]
        table_errors = schema_errors(name, frame, expected, PRIMARY_KEYS[name])
        errors.extend(table_errors)
        if table_errors:
            continue

        if "geometry" in expected:
            if frame.crs is None or frame.crs.to_epsg() != 4326:
                errors.append(f"{name}: CRS is {frame.crs}, expected EPSG:4326")
            if frame.geometry.isna().any() or frame.geometry.is_empty.any():
                errors.append(f"{name}: null or empty geometry")
            elif not frame.geometry.is_valid.all():
                errors.append(f"{name}: {(~frame.geometry.is_valid).sum()} invalid geometries")

    if errors:
        return errors

    dams = tables["dam_removal"]
    stations = tables["wq_station"]
    links = tables["dam_station_link"]
    watersheds = tables["huc8_watershed"]
    streams = tables["streams"]

    for name, column, width in [
        ("huc8_watershed", "huc8", 8),
        ("dam_removal", "HUC8", 8),
        ("wq_station", "huc8", 8),
    ]:
        values = tables[name][column].dropna()
        if bad := sorted(values[values.str.len() != width].unique())[:10]:
            errors.append(f"{name}.{column}: values not {width} characters long: {bad}")

    # Two-letter codes; dams on a state line list each, e.g. "MN; ND".
    states = dams["State"].dropna()
    if bad := sorted(states[~states.str.fullmatch(r"[A-Z]{2}(; [A-Z]{2})*")].unique())[:10]:
        errors.append(f"dam_removal.State: not two-letter codes separated by '; ': {bad}")

    if bad := _outside(dams["HUC8"], watersheds["huc8"]):
        errors.append(f"dam_removal.HUC8 not in huc8_watershed: {bad}")
    if bad := _outside(stations["huc8"], watersheds["huc8"]):
        errors.append(f"wq_station.huc8 not in huc8_watershed: {bad}")

    for name, frame in [("dam_removal", dams), ("wq_station", stations)]:
        if not frame["snap_fraction"].between(0, 1).all():
            errors.append(f"{name}.snap_fraction outside [0, 1]")

    years = dams["Year_Removed"].dropna()
    if not years.between(1800, current_year).all():
        errors.append(f"dam_removal.Year_Removed outside [1800, {current_year}]")

    if bad := _outside(links["AR_ID"], dams["AR_ID"]):
        errors.append(f"dam_station_link.AR_ID not in dam_removal: {bad}")
    if bad := _outside(links["station_id"], stations["station_id"]):
        errors.append(f"dam_station_link.station_id not in wq_station: {bad}")
    if bad := _outside(links["relationship"], list(CONNECTED_RELATIONSHIPS)):
        errors.append(f"dam_station_link.relationship has unexpected values: {bad}")
    if (links["network_distance_km"].dropna() < 0).any():
        errors.append("dam_station_link.network_distance_km has negative values")

    for name, frame in [("dam_removal", dams), ("wq_station", stations)]:
        if bad := _outside(frame["NHDPlusID"], streams["nhdplus_id"]):
            errors.append(f"{name}.NHDPlusID not in streams: {bad}")

    return errors


# --------------------------------------------------
# Main
# --------------------------------------------------

def build_tables(cfg, huc8s):
    processed_dir = cfg.project_root / "data" / "processed"
    check_inputs(processed_dir, huc8s)

    huc8_watershed = build_huc8_watershed(cfg.path("wbd_source"), huc8s)

    dam_removal = build_dam_removal(load_snapped_layer(processed_dir, huc8s, "dams_snapped"))
    stations = build_station_candidates(
        load_snapped_layer(processed_dir, huc8s, "stations_snapped")
    )
    dam_station_link = build_dam_station_link(
        load_all_links(processed_dir, huc8s), dam_removal, stations
    )
    wq_station = build_wq_station(stations, dam_station_link)

    streams = build_streams(
        processed_dir / "nhdplus_extracted", huc8s, dam_removal, wq_station
    )

    return {
        "dam_removal": dam_removal,
        "wq_station": wq_station,
        "dam_station_link": dam_station_link,
        "huc8_watershed": huc8_watershed,
        "streams": streams,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    huc8s = [huc8.zfill(8) for huc8 in cfg.huc8s]

    print(f"Case study: {cfg.name}")
    print(f"HUC8 scope: {huc8s}")

    tables = build_tables(cfg, huc8s)

    errors = validate_tables(tables)
    if errors:
        raise ValueError(
            "Validation failed; no warehouse files were written:\n  - "
            + "\n  - ".join(errors)
        )

    output_dir = warehouse_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Written ===")
    for name, frame in tables.items():
        path = output_dir / f"{name}.parquet"
        write_parquet_atomic(frame, path)
        print(f"  {name}: {len(frame):,} rows -> {path}")


if __name__ == "__main__":
    main()
