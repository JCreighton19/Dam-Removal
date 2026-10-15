"""Shared helpers for loading the dam-removal source dataset."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from .io import find_column, require_file


def load_dams(dam_path):
    """Load the dam-removal CSV as a point GeoDataFrame in EPSG:4326."""
    dam_path = require_file(dam_path, "Dam removal CSV")
    dams = pd.read_csv(dam_path, low_memory=False)

    lat_col = find_column(dams.columns, "Latitude")
    lon_col = find_column(dams.columns, "Longitude")

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not find dam coordinate columns. "
            f"Available columns: {dams.columns.tolist()}"
        )

    dams[lat_col] = pd.to_numeric(dams[lat_col], errors="coerce")
    dams[lon_col] = pd.to_numeric(dams[lon_col], errors="coerce")
    dams = dams.dropna(subset=[lat_col, lon_col]).copy()

    dams = dams[
        dams[lat_col].between(-90, 90) & dams[lon_col].between(-180, 180)
    ].copy()

    return gpd.GeoDataFrame(
        dams,
        geometry=gpd.points_from_xy(dams[lon_col], dams[lat_col]),
        crs="EPSG:4326",
    )
