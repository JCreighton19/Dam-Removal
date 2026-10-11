from pathlib import Path
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
HUC8 = "19020401"
STREAM_LAYER = "NHDFlowline"
WQP_DIR = ROOT / "data" / "raw" / "wqp"
STATION_DIR = WQP_DIR / "stations"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / f"snapped_features_{HUC8}.gpkg"
DAM_FILE_NAME = "ARDamRemovalList_Figshare_Mar2026.csv"
WBD_FILE_NAME = "WBD_National_GPKG.gpkg"
SURFACE_WATER_TYPES = ["Stream", "River/Stream"]

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def find_file(filename, search_root):
    matches = list(search_root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {search_root}"
        )

    return matches[0]


def find_column(columns, expected):
    """Find a column name without case sensitivity."""
    return next(
        (col for col in columns if col.lower() == expected.lower()),
        None,
    )


def load_watershed():
    wbd_path = find_file(WBD_FILE_NAME, ROOT)
    layers = gpd.list_layers(wbd_path)

    layer_name = next(
        (
            name for name in layers["name"]
            if "hu8" in name.lower()
        ),
        None,
    )

    if layer_name is None:
        raise ValueError("Could not find a HU8 layer in the WBD file.")

    wbd = gpd.read_file(wbd_path, layer=layer_name)
    huc_col = find_column(wbd.columns, "HUC8")

    if huc_col is None:
        raise ValueError(
            f"No HUC8 column found. Available columns: {wbd.columns.tolist()}"
        )

    watershed = wbd[
        wbd[huc_col].astype(str).str.zfill(8) == HUC8
    ].copy()

    if watershed.empty:
        raise ValueError(f"HUC8 {HUC8} was not found.")

    return watershed.to_crs("EPSG:4326")


def load_stations():
    files = sorted(STATION_DIR.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No station CSVs found in {STATION_DIR}"
        )

    frames = []

    for file in files:
        df = pd.read_csv(file, low_memory=False)
        df["source_huc8"] = file.stem
        frames.append(df)

    stations = pd.concat(frames, ignore_index=True)

    stations = stations[
        stations["MonitoringLocationTypeName"].isin(
            SURFACE_WATER_TYPES
        )
    ].copy()

    for col in ["LatitudeMeasure", "LongitudeMeasure"]:
        stations[col] = pd.to_numeric(
            stations[col], errors="coerce"
        )

    stations = stations.dropna(
        subset=["LatitudeMeasure", "LongitudeMeasure"]
    )

    stations = stations[
        stations["LatitudeMeasure"].between(-90, 90)
        & stations["LongitudeMeasure"].between(-180, 180)
    ].copy()

    stations = stations.drop_duplicates(
        subset="MonitoringLocationIdentifier"
    )

    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations["LongitudeMeasure"],
            stations["LatitudeMeasure"],
        ),
        crs="EPSG:4326",
    )


def load_dams():
    dam_path = find_file(DAM_FILE_NAME, ROOT)
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
        dams[lat_col].between(-90, 90)
        & dams[lon_col].between(-180, 180)
    ].copy()

    return gpd.GeoDataFrame(
        dams,
        geometry=gpd.points_from_xy(
            dams[lon_col],
            dams[lat_col],
        ),
        crs="EPSG:4326",
    )


def keep_inside_watershed(points, watershed):
    """Keep points geographically inside the pilot watershed."""
    return gpd.sjoin(
        points,
        watershed[["geometry"]],
        how="inner",
        predicate="within",
    ).drop(columns="index_right", errors="ignore")


# --------------------------------------------------
# Snap points to their nearest flowline
# --------------------------------------------------

def snap_to_streams(points, streams, id_column):
    """
    Return points with nearest COMID, snap distance,
    snapped coordinates, and normalized position along
    the matched flowline.

    Coordinates and distances are calculated in a
    projected CRS with meter units.
    """
    points = points.to_crs(streams.crs).copy()
    stream_id_col = find_column(streams.columns, "NHDPlusID")

    if stream_id_col is None:
        raise ValueError(
            "COMID not found in NHDFlowline. "
            f"Available columns: {streams.columns.tolist()}"
        )

    stream_lookup = streams[
        [stream_id_col, "geometry"]
    ].copy().reset_index(drop=True)

    stream_lookup = stream_lookup[
        stream_lookup.geometry.notna()
        & ~stream_lookup.geometry.is_empty
    ].copy()

    # Nearest join retains each original point geometry.
    joined = gpd.sjoin_nearest(
        points,
        stream_lookup,
        how="left",
        distance_col="snap_distance_m",
    )

    # Preserve the original point location.
    joined["original_x"] = joined.geometry.x
    joined["original_y"] = joined.geometry.y

    snapped_geometries = []
    snap_fractions = []

    for _, row in joined.iterrows():
        right_index = row.get("index_right")

        if pd.isna(right_index):
            snapped_geometries.append(None)
            snap_fractions.append(float("nan"))
            continue

        point = row.geometry
        line = stream_lookup.loc[int(right_index), "geometry"]

        # Project the point onto the nearest point on the line.
        snapped_point = nearest_points(point, line)[1]

        # Fraction along the line: 0 at start, 1 at end.
        fraction = line.project(snapped_point, normalized=True)

        snapped_geometries.append(snapped_point)
        snap_fractions.append(fraction)

    joined["snap_fraction"] = snap_fractions
    joined["snapped_geometry"] = snapped_geometries

    # Store snapped coordinates while retaining the original
    # point geometry as the active geometry.
    joined["snapped_x"] = [
        geom.x if geom is not None else float("nan")
        for geom in snapped_geometries
    ]
    joined["snapped_y"] = [
        geom.y if geom is not None else float("nan")
        for geom in snapped_geometries
    ]

    # Normalize output column names.
    joined = joined.rename(
        columns={stream_id_col: "COMID"}
    )

    # Remove the extra join index, but keep the original point.
    joined = joined.drop(
        columns=["index_right"],
        errors="ignore",
    )

    joined["feature_id"] = joined[id_column].astype(str)

    return joined


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    print(f"Processing pilot HUC8: {HUC8}")

    watershed = load_watershed()
    stations = load_stations()
    dams = load_dams()

    # Choose a suitable local projected CRS in meters.
    metric_crs = watershed.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError(
            "Could not estimate a projected CRS for the watershed."
        )

    print(f"Metric CRS: {metric_crs}")

    # Restrict points to the watershed before snapping.
    stations = keep_inside_watershed(stations, watershed)
    dams = keep_inside_watershed(dams, watershed)

    print(f"Stations inside watershed: {len(stations):,}")
    print(f"Dam removals inside watershed: {len(dams):,}")

    # Load the already-extracted HUC8 geodatabase.
    nhd_dir = ROOT / "data" / "processed" / "nhdplus_extracted"
    gdb_paths = list(nhd_dir.rglob("*.gdb"))

    gdb_path = next(
        (path for path in gdb_paths if HUC8 in path.name),
        None,
    )

    if gdb_path is None:
        raise FileNotFoundError(
            f"No extracted HUC8 geodatabase found under {nhd_dir}"
        )

    streams = gpd.read_file(
        gdb_path,
        layer=STREAM_LAYER,
    )

    if streams.crs is None:
        raise ValueError("The NHDFlowline layer has no defined CRS.")

    streams = streams.to_crs(metric_crs)

    print(f"Loaded stream reaches: {len(streams):,}")

    # Snap both point datasets to the stream network.
    stations_snapped = snap_to_streams(
        stations,
        streams,
        "MonitoringLocationIdentifier",
    )

    dams_snapped = snap_to_streams(
        dams,
        streams,
        "Dam_Name",
    )

    print("\nStation snap distances (meters):")
    print(stations_snapped["snap_distance_m"].describe())
    print("\nDam snap distances (meters):")
    print(dams_snapped["snap_distance_m"].describe())

    # Write snapped points and original locations to GeoPackage.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Export in WGS84 so the layers are easy to inspect on a map.
    stations_out = stations_snapped.to_crs("EPSG:4326")
    dams_out = dams_snapped.to_crs("EPSG:4326")

    # Keep only tabular snap coordinates in the output.
    # The active geometry remains the original point location.
    stations_out = stations_out.drop(
        columns=["snapped_geometry"],
        errors="ignore",
    )
    dams_out = dams_out.drop(
        columns=["snapped_geometry"],
        errors="ignore",
    )

    stations_out.to_file(
        OUTPUT_FILE,
        layer="stations_snapped",
        driver="GPKG",
        mode="w",
    )

    dams_out.to_file(
        OUTPUT_FILE,
        layer="dams_snapped",
        driver="GPKG",
        mode="a",
    )

    print(f"\nSaved snapped features to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
