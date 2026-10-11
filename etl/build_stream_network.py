"""
Build a directed stream-reach network from NHDPlus HR data.
"""

from pathlib import Path
import geopandas as gpd
import networkx as nx
import pandas as pd


# Project configuration.
ROOT = Path(__file__).resolve().parents[1]
HUC8 = "19020401"

STREAMS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "nhdplus_extracted"
    / f"NHDPLUS_H_{HUC8}_HU8_GDB"
    / f"NHDPLUS_H_{HUC8}_HU8_GDB.gdb"
)

STREAM_LAYER = "NHDFlowline"
FLOW_LAYER = "NHDPlusFlow"


def main():
    # Load stream reaches and their connectivity relationships.
    print("Loading NHDPlus HR data...")

    streams = gpd.read_file(
        STREAMS_PATH,
        layer=STREAM_LAYER,
    )

    flow = gpd.read_file(
        STREAMS_PATH,
        layer=FLOW_LAYER,
    )

    print(f"Stream reaches loaded: {len(streams):,}")
    print(f"Flow relationships loaded: {len(flow):,}")

    # Convert reach identifiers to integers for exact matching.
    # NHDPlusID values are identifiers, not continuous measurements.
    streams["NHDPlusID"] = pd.to_numeric(
        streams["NHDPlusID"], errors="raise"
    ).astype("int64")

    flow["from_id"] = pd.to_numeric(
        flow["FromNHDPID"], errors="raise"
    ).astype("int64")

    flow["to_id"] = pd.to_numeric(
        flow["ToNHDPID"], errors="raise"
    ).astype("int64")

    stream_ids = set(streams["NHDPlusID"].dropna())

    # Keep connections where both reach IDs exist in this dataset.
    valid = (
        flow["from_id"].ne(0)
        & flow["to_id"].ne(0)
        & flow["from_id"].isin(stream_ids)
        & flow["to_id"].isin(stream_ids)
    )

    valid_flow = flow.loc[valid]

    # Build a directed graph: each node is a reach;
    # each edge represents a connection between reaches.
    G = nx.DiGraph()
    G.add_nodes_from(stream_ids)
    G.add_edges_from(
        valid_flow[["from_id", "to_id"]].itertuples(
            index=False,
            name=None,
        )
    )

    # Report network diagnostics.
    print("\nStream network diagnostics")
    print(f"Reach nodes: {G.number_of_nodes():,}")
    print(f"Directed connections: {G.number_of_edges():,}")
    print(
        "Weakly connected components: "
        f"{nx.number_weakly_connected_components(G):,}"
    )
    print(f"Flow rows excluded: {len(flow) - len(valid_flow):,}")

    # Report reaches without incoming or outgoing edges.
    sources = [node for node in G if G.in_degree(node) == 0]
    outlets = [node for node in G if G.out_degree(node) == 0]

    print(f"Reaches with no incoming edge: {len(sources):,}")
    print(f"Reaches with no outgoing edge: {len(outlets):,}")


if __name__ == "__main__":
    main()