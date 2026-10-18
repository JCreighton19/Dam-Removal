from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points

from etl.common.config import default_case_study, load_config
from etl.common.dams import filter_to_sampled_dams, load_dams
from etl.common.io import require_file
from etl.common.nhdplus import (
    find_huc8_geodatabase,
    is_hu4_fallback,
    read_nhdplus_layer,
)
from etl.common.wbd import clip_to_watersheds, load_huc8_watershed

# --------------------------------------------------
# Configuration
# --------------------------------------------------

STREAM_LAYER = "NHDFlowline"


# --------------------------------------------------
# Loading candidate stations
# --------------------------------------------------

def load_candidate_stations(candidate_path):
    """Load the pass-1 candidate stations produced by
    select_candidate_stations.py. Snapping only operates on this
    dam-relevant candidate set, not on every WQP station ever downloaded.
    """
    require_file(
        candidate_path,
        "Candidate station file (run select_candidate_stations.py first)",
    )

    stations = pd.read_csv(candidate_path, low_memory=False)

    for col in ["LatitudeMeasure", "LongitudeMeasure"]:
        stations[col] = pd.to_numeric(stations[col], errors="coerce")

    stations = stations.dropna(subset=["LatitudeMeasure", "LongitudeMeasure"])
    stations = stations[
        stations["LatitudeMeasure"].between(-90, 90)
        & stations["LongitudeMeasure"].between(-180, 180)
    ].copy()
    stations = stations.drop_duplicates(subset="MonitoringLocationIdentifier")

    return gpd.GeoDataFrame(
        stations,
        geometry=gpd.points_from_xy(
            stations["LongitudeMeasure"],
            stations["LatitudeMeasure"],
        ),
        crs="EPSG:4326",
    )


def find_column(columns, expected):
    """Find a column name without case sensitivity."""
    return next(
        (col for col in columns if col.lower() == expected.lower()),
        None,
    )


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
        columns={stream_id_col: "NHDPlusID"}
    )

    # Remove the extra join index, but keep the original point.
    joined = joined.drop(
        columns=["index_right"],
        errors="ignore",
    )

    joined["feature_id"] = joined[id_column].astype(str)

    return joined


def print_duplicate_snaps(gdf, id_column, label):
    """
    Used to identify cases in which points are equidistant to multiple streams
    """
    counts = gdf.groupby(id_column, dropna=False).size()
    duplicate_ids = counts[counts > 1].index

    duplicates = gdf[gdf[id_column].isin(duplicate_ids)].copy()

    print(f"\n{label}: equidistant/multiple-match candidates")
    print(f"Features with multiple matches: {len(duplicate_ids):,}")
    print(f"Additional rows from multiple matches: {len(duplicates) - len(duplicate_ids):,}")

    if not duplicates.empty:
        print(
            duplicates[
                [id_column, "NHDPlusID", "snap_distance_m",
                 "original_x", "original_y", "snapped_x", "snapped_y"]
            ].sort_values(id_column).to_string(index=False)
        )


# --------------------------------------------------
# Main
# --------------------------------------------------

def process_huc8(cfg, huc8, all_stations, all_dams):
    output_dir = cfg.project_root / "data" / "processed"
    output_file = output_dir / f"snapped_features_{huc8}.gpkg"

    print(f"\n=== Processing HUC8: {huc8} ===")

    watershed = load_huc8_watershed(cfg.path("wbd_source"), huc8)
    stations = all_stations
    dams = all_dams

    # Choose a suitable local projected CRS in meters.
    metric_crs = watershed.estimate_utm_crs()

    if metric_crs is None:
        raise ValueError(
            "Could not estimate a projected CRS for the watershed."
        )

    print(f"Metric CRS: {metric_crs}")

    # Restrict points to the watershed before snapping.
    stations = clip_to_watersheds(stations, watershed)
    dams = clip_to_watersheds(dams, watershed)

    print(f"Stations inside watershed: {len(stations):,}")
    print(f"Dam removals inside watershed: {len(dams):,}")

    # Load the already-extracted HUC8 geodatabase (an HU8 package, or an
    # HU4 fallback covering this HUC8's parent HUC4).
    nhd_dir = cfg.project_root / "data" / "processed" / "nhdplus_extracted"
    gdb_path = find_huc8_geodatabase(nhd_dir, huc8)

    if is_hu4_fallback(gdb_path, huc8):
        print(f"Using HU4 fallback package for HUC8 {huc8}: {gdb_path.name}")

    streams = read_nhdplus_layer(gdb_path, STREAM_LAYER)

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

    print_duplicate_snaps(
        stations_snapped,
        "MonitoringLocationIdentifier",
        "Stations",
    )

    print_duplicate_snaps(
        dams_snapped,
        "Dam_Name",
        "Dam removals",
    )

    print("\nStation snap distances (meters):")
    print(stations_snapped["snap_distance_m"].describe())
    print("\nDam snap distances (meters):")
    print(dams_snapped["snap_distance_m"].describe())

    # --------------------------------------------------
    # Export original and snapped point geometries
    # --------------------------------------------------

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build separate GeoDataFrames using the snapped points
    # as their active geometry, retaining all original attributes.

    stations_on_streams = gpd.GeoDataFrame(
        stations_snapped.drop(
            columns=["geometry", "snapped_geometry"],
            errors="ignore",
        ),
        geometry=stations_snapped["snapped_geometry"],
        crs=stations_snapped.crs,
    )

    dams_on_streams = gpd.GeoDataFrame(
        dams_snapped.drop(
            columns=["geometry", "snapped_geometry"],
            errors="ignore",
        ),
        geometry=dams_snapped["snapped_geometry"],
        crs=dams_snapped.crs,
    )

    # Export all four layers in WGS84.
    stations_original = stations_snapped.to_crs("EPSG:4326")
    dams_original = dams_snapped.to_crs("EPSG:4326")
    stations_on_streams = stations_on_streams.to_crs("EPSG:4326")
    dams_on_streams = dams_on_streams.to_crs("EPSG:4326")

    # Write the original layers first.
    if output_file.exists(): # overwrite if already exists
        output_file.unlink()

    stations_original.to_file(
        output_file,
        layer="stations_original",
        driver="GPKG",
        mode="w",
    )

    dams_original.to_file(
        output_file,
        layer="dams_original",
        driver="GPKG",
        mode="a",
    )

    # Write the snapped layers.
    stations_on_streams.to_file(
        output_file,
        layer="stations_snapped",
        driver="GPKG",
        mode="a",
    )

    dams_on_streams.to_file(
        output_file,
        layer="dams_snapped",
        driver="GPKG",
        mode="a",
    )

    print(f"\nSaved original and snapped features to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    parser.add_argument(
        "--huc8",
        default=None,
        help="Process only this HUC8. Defaults to every HUC8 listed in the "
        "case-study config, processed one at a time.",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    huc8_list = [args.huc8] if args.huc8 else cfg.huc8s

    candidate_path = cfg.wqp_results_dir / "pilot_station_selection.csv"

    # Loaded once and reused across HUC8s: the candidate file and dam
    # source cover the whole case study, not one HUC8 each.
    all_stations = load_candidate_stations(candidate_path)
    all_dams = load_dams(cfg.path("dam_source"))

    sampled_dams = filter_to_sampled_dams(all_dams, cfg)
    if len(sampled_dams) < len(all_dams):
        print(
            f"Restricting to the {len(sampled_dams)} sampled dam(s) in the "
            f"config (of {len(all_dams):,} in the dam source).",
            flush=True,
        )
    all_dams = sampled_dams

    results = []

    for huc8 in huc8_list:
        try:
            process_huc8(cfg, huc8, all_stations, all_dams)
            results.append((huc8, "ok"))
        except Exception as exc:
            print(f"FAILED HUC8 {huc8}: {exc}", file=sys.stderr)
            results.append((huc8, f"failed: {exc}"))

    if len(huc8_list) > 1:
        print("\n=== Summary ===")
        for huc8, status in results:
            print(f"  {huc8}: {status}")

    if any(status != "ok" for _, status in results):
        raise RuntimeError("One or more HUC8s failed; see log above.")


if __name__ == "__main__":
    main()
