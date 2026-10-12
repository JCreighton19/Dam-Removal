from pathlib import Path
import argparse
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd


def find_column(df, candidates, description):
    """Find the first available column matching a list of candidates."""
    for column in candidates:
        if column in df.columns:
            return column
    return None


def clean_id(value):
    """Convert numeric IDs to integers while preserving missing values."""
    if pd.isna(value):
        return None
    try:
        value = int(value)
        return value if value != 0 else None
    except (ValueError, TypeError, OverflowError):
        return None


def calculate_network_distance(
    G,
    dam_reach,
    station_reach,
    dam_fraction,
    station_fraction,
    length_by_reach,
    relationship,
):
    """Estimate along-network distance between snapped locations, in km.

    Assumes snap_fraction increases in the direction of flow along each
    reach geometry. Validate this assumption for the NHDPlus data.
    """
    if relationship not in {"same_reach", "upstream", "downstream"}:
        return np.nan, None

    if pd.isna(dam_fraction) or pd.isna(station_fraction):
        return np.nan, None

    dam_fraction = float(dam_fraction)
    station_fraction = float(station_fraction)

    dam_length = length_by_reach.get(dam_reach, np.nan)
    station_length = length_by_reach.get(station_reach, np.nan)

    if pd.isna(dam_length) or pd.isna(station_length):
        return np.nan, None

    # Both locations lie on the same reach.
    if dam_reach == station_reach:
        distance = abs(dam_fraction - station_fraction) * dam_length
        return distance, "same_reach_fraction"

    # Find a directed path between the two reaches.
    if relationship == "upstream":
        source, target = station_reach, dam_reach
    else:
        source, target = dam_reach, station_reach

    try:
        path = nx.shortest_path(
            G,
            source=source,
            target=target,
            weight="weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return np.nan, None

    if len(path) < 2:
        return np.nan, None

    intermediate_length = sum(
        float(length_by_reach.get(reach, 0) or 0)
        for reach in path[1:-1]
    )

    if relationship == "upstream":
        # Travel downstream from the station to the dam.
        distance = (
            (1 - station_fraction) * station_length
            + intermediate_length
            + dam_fraction * dam_length
        )
    else:
        # Travel downstream from the dam to the station.
        distance = (
            (1 - dam_fraction) * dam_length
            + intermediate_length
            + station_fraction * station_length
        )

    return distance, "network_path_fractional_endpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--huc8", default="19020401")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    huc8 = args.huc8

    gdb_path = (
        root / "data" / "processed" / "nhdplus_extracted"
        / f"NHDPLUS_H_{huc8}_HU8_GDB"
        / f"NHDPLUS_H_{huc8}_HU8_GDB.gdb"
    )
    snapped_path = (
        root / "data" / "processed"
        / f"snapped_features_{huc8}.gpkg"
    )
    output_dir = root / "data" / "processed"
    output_path = output_dir / f"dam_station_links_{huc8}.csv"

    if not gdb_path.exists():
        raise FileNotFoundError(f"NHDPlus geodatabase not found: {gdb_path}")
    if not snapped_path.exists():
        raise FileNotFoundError(f"Snapped features not found: {snapped_path}")

    print(f"HUC8: {huc8}")
    print("Loading snapped dams and stations...")
    dams = gpd.read_file(snapped_path, layer="dams_snapped")
    stations = gpd.read_file(snapped_path, layer="stations_snapped")
    required = {"NHDPlusID", "snap_fraction", "snap_distance_m"}
    for name, frame in [("dams", dams), ("stations", stations)]:
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} layer is missing required columns: {sorted(missing)}"
            )

    # Use stable source identifiers when available; otherwise assign row IDs.
    dam_id_col = "AR_ID"
    if dam_id_col not in dams.columns:
        raise ValueError("Dam layer is missing the unique identifier AR_ID.")

    station_id_col = find_column(
        stations,
        [
            "MonitoringLocationIdentifier", "StationIdentifier",
            "station_id", "StationID", "site_no", "SiteNumber",
            "id", "ID",
        ],
        "station identifier",
    )

    dams = dams.reset_index(drop=True).copy()
    stations = stations.reset_index(drop=True).copy()

    dams["dam_id"] = (
        dams[dam_id_col].astype(str)
        if dam_id_col else [f"dam_{i:06d}" for i in range(len(dams))]
    )
    stations["station_id"] = (
        stations[station_id_col].astype(str)
        if station_id_col
        else [f"station_{i:06d}" for i in range(len(stations))]
    )

    dams["reach_id"] = dams["NHDPlusID"].map(clean_id)
    stations["reach_id"] = stations["NHDPlusID"].map(clean_id)

    print(f"Dams: {len(dams):,}")
    print(f"Stations: {len(stations):,}")
    print("Loading stream lengths and flow relationships...")

    streams = gpd.read_file(gdb_path, layer="NHDFlowline")
    flow = gpd.read_file(gdb_path, layer="NHDPlusFlow")
    streams["NHDPlusID"] = pd.to_numeric(
        streams["NHDPlusID"], errors="coerce"
    )

    streams = streams.dropna(subset=["NHDPlusID"]).copy()
    streams["NHDPlusID"] = streams["NHDPlusID"].astype("int64")
    streams = streams.drop_duplicates("NHDPlusID")
    length_by_reach = streams.set_index("NHDPlusID")["LengthKM"].to_dict()
    stream_ids = set(length_by_reach)

    flow["FromNHDPID"] = pd.to_numeric(
        flow["FromNHDPID"], errors="coerce"
    )
    flow["ToNHDPID"] = pd.to_numeric(
        flow["ToNHDPID"], errors="coerce"
    )

    valid = (
        flow["FromNHDPID"].notna()
        & flow["ToNHDPID"].notna()
        & flow["FromNHDPID"].ne(0)
        & flow["ToNHDPID"].ne(0)
        & flow["FromNHDPID"].isin(stream_ids)
        & flow["ToNHDPID"].isin(stream_ids)
    )
    valid_flow = flow.loc[valid].copy()

    # Assumption to validate: FromNHDPID -> ToNHDPID is downstream.
    G = nx.DiGraph()
    G.add_nodes_from(stream_ids)
    G.add_edges_from(
        zip(
            valid_flow["FromNHDPID"].astype("int64"),
            valid_flow["ToNHDPID"].astype("int64"),
        )
    )

    # Weight each edge by the length of the reach being exited.
    # This is used to select the shortest route through the network.
    for u, v in G.edges:
        G[u][v]["weight"] = float(length_by_reach.get(u, 0) or 0)

    print(f"Stream reaches: {len(stream_ids):,}")
    print(f"Directed connections: {G.number_of_edges():,}")
    print(f"Flow rows excluded: {(~valid).sum():,}")

    # Cache ancestor/descendant sets so each dam's graph traversal is
    # performed once, rather than once for every station.
    print("Linking dams to stations...")
    rows = []

    for dam in dams.itertuples(index=False):
        dam_id = getattr(dam, "dam_id")
        dam_name = getattr(dam, "Dam_Name", None)
        year_removed = getattr(dam, "Year_Removed", None)
        river = getattr(dam, "River", None)
        dam_reach = getattr(dam, "reach_id")
        dam_fraction = getattr(dam, "snap_fraction")
        dam_snap_distance = getattr(dam, "snap_distance_m")

        if dam_reach is not None and dam_reach in G:
            upstream = nx.ancestors(G, dam_reach)
            downstream = nx.descendants(G, dam_reach)
        else:
            upstream, downstream = set(), set()

        for station in stations.itertuples(index=False):
            station_id = getattr(station, "station_id")
            station_reach = getattr(station, "reach_id")
            station_fraction = getattr(station, "snap_fraction")
            station_snap_distance = getattr(station, "snap_distance_m")

            relationship = "unconnected"

            if dam_reach is not None and station_reach is not None:
                if dam_reach == station_reach and dam_reach in G:
                    if station_fraction < dam_fraction:
                        relationship = "upstream"
                    elif station_fraction > dam_fraction:
                        relationship = "downstream"
                    else:
                        relationship = "same_reach"
                elif station_reach in upstream:
                    relationship = "upstream"
                elif station_reach in downstream:
                    relationship = "downstream"

            distance_km, distance_method = calculate_network_distance(
                G=G,
                dam_reach=dam_reach,
                station_reach=station_reach,
                dam_fraction=dam_fraction,
                station_fraction=station_fraction,
                length_by_reach=length_by_reach,
                relationship=relationship,
            )

            rows.append({
                "huc8": huc8,
                "dam_id": dam_id,
                "dam_name": dam_name,
                "station_id": station_id,
                "dam_reach_id": dam_reach,
                "year_removed": year_removed,
                "river": river,
                "station_reach_id": station_reach,
                "relationship": relationship,
                "distance_to_dam_km": distance_km,
                "network_distance_km": distance_km,
                "distance_method": distance_method,
                "dam_snap_distance_m": dam_snap_distance,
                "station_snap_distance_m": station_snap_distance,
                "dam_snap_fraction": dam_fraction,
                "station_snap_fraction": station_fraction,
            })

    links = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    links.to_csv(output_path, index=False)

    print("\nRelationship counts:")
    print(links["relationship"].value_counts(dropna=False).to_string())
    print(f"\nOutput rows: {len(links):,}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
