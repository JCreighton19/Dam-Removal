"""Shared helpers for loading HUC8 watershed boundaries from the national WBD."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

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


def load_huc8_watershed(wbd_path, huc8):
    """Load a single HUC8 watershed polygon from the national WBD GeoPackage."""
    wbd_path = require_file(wbd_path, "WBD GeoPackage")
    layer_name = find_hu8_layer(wbd_path)
    wbd = gpd.read_file(wbd_path, layer=layer_name)
    huc_col = find_column(wbd.columns, "HUC8")

    if huc_col is None:
        raise ValueError(
            f"No HUC8 column found in layer '{layer_name}'. "
            f"Available columns: {wbd.columns.tolist()}"
        )

    watershed = wbd[
        wbd[huc_col].astype(str).str.zfill(8) == str(huc8).zfill(8)
    ].copy()

    if watershed.empty:
        raise ValueError(f"HUC8 {huc8} was not found in {wbd_path}.")

    return watershed.to_crs("EPSG:4326")


def load_huc8_watersheds(wbd_path, huc8s):
    """Load several HUC8 watershed polygons as one GeoDataFrame (not dissolved)."""
    frames = [load_huc8_watershed(wbd_path, huc8) for huc8 in huc8s]
    combined = pd.concat(frames, ignore_index=True)
    return gpd.GeoDataFrame(combined, crs=frames[0].crs)


def clip_to_watersheds(points, watersheds):
    """Keep only points that fall geographically inside the given watershed(s)."""
    return gpd.sjoin(
        points,
        watersheds[["geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns="index_right", errors="ignore")
