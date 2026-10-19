"""Shared helpers for loading HUC8 watershed boundaries from the national WBD."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import pyogrio

from .io import find_column, require_file


def find_hu8_layer(wbd_path):
    """Return the HU8 layer name inside a WBD GeoPackage."""
    layers = gpd.list_layers(wbd_path)
    layer_name = next(
        (name for name in layers["name"] if "hu8" in name.lower()),
        None,
    )
    if layer_name is None:
        raise ValueError(f"Could not find a HU8 layer in {wbd_path}.")
    return layer_name


def load_huc8_watersheds(wbd_path, huc8s):
    """Load several HUC8 watershed polygons as one GeoDataFrame (not dissolved).

    Reads only the requested rows (an SQL filter), not the whole national
    layer, so it stays fast for hundreds of HUC8s.
    """
    wbd_path = require_file(wbd_path, "WBD GeoPackage")
    layer_name = find_hu8_layer(wbd_path)
    fields = [str(f) for f in pyogrio.read_info(wbd_path, layer=layer_name)["fields"]]
    huc_col = find_column(fields, "HUC8")

    if huc_col is None:
        raise ValueError(
            f"No HUC8 column found in layer '{layer_name}'. Available columns: {fields}"
        )

    codes = sorted({str(h).zfill(8) for h in huc8s})
    in_list = ", ".join(f"'{code}'" for code in codes)
    wbd = gpd.read_file(wbd_path, layer=layer_name, where=f'"{huc_col}" IN ({in_list})')
    wbd = wbd[wbd[huc_col].astype(str).str.zfill(8).isin(codes)]

    missing = sorted(set(codes) - set(wbd[huc_col].astype(str).str.zfill(8)))
    if missing:
        raise ValueError(f"HUC8(s) {missing} not found in {wbd_path}.")

    return wbd.to_crs("EPSG:4326").reset_index(drop=True)


def load_huc8_watershed(wbd_path, huc8):
    """Load a single HUC8 watershed polygon from the national WBD GeoPackage."""
    return load_huc8_watersheds(wbd_path, [huc8])


def clip_to_watersheds(points, watersheds):
    """Keep only points that fall geographically inside the given watershed(s)."""
    return gpd.sjoin(
        points,
        watersheds[["geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns="index_right", errors="ignore")


def resolve_huc8_for_points(points, wbd_path):
    """Reverse lookup: for each point, find the HUC8 watershed it falls within.

    Returns the input points with a new "resolved_huc8" column (NaN for
    any point that doesn't fall inside any HUC8 polygon, e.g. outside the
    US or just offshore).
    """
    wbd_path = require_file(wbd_path, "WBD GeoPackage")
    layer_name = find_hu8_layer(wbd_path)
    wbd = gpd.read_file(wbd_path, layer=layer_name)
    huc_col = find_column(wbd.columns, "HUC8")

    if huc_col is None:
        raise ValueError(
            f"No HUC8 column found in layer '{layer_name}'. "
            f"Available columns: {wbd.columns.tolist()}"
        )

    wbd = wbd.to_crs("EPSG:4326").copy()
    wbd["resolved_huc8"] = wbd[huc_col].astype(str).str.zfill(8)

    points = points.to_crs("EPSG:4326")
    joined = gpd.sjoin(
        points,
        wbd[["resolved_huc8", "geometry"]],
        how="left",
        predicate="within",
    ).drop(columns="index_right", errors="ignore")

    return joined
