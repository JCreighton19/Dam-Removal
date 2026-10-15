"""
Build a directed stream-reach network from NHDPlus HR data and report
connectivity diagnostics. This is a diagnostic-only script -- it does not
persist the graph; `link_dams_stations.py` builds the same graph (via
etl/common/network.py) as part of producing dam-station links.
"""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import networkx as nx

from etl.common.config import load_config
from etl.common.network import build_reach_graph

STREAM_LAYER = "NHDFlowline"
FLOW_LAYER = "NHDPlusFlow"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-study",
        default="pilot_19020401",
        help="Name of a config file under config/ (without the .yaml extension). "
        "Used only to resolve the project root in this script.",
    )
    parser.add_argument(
        "--huc8",
        default=None,
        help="Defaults to the first HUC8 listed in the case-study config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    huc8 = args.huc8 or cfg.huc8s[0]

    streams_path = (
        cfg.project_root
        / "data"
        / "processed"
        / "nhdplus_extracted"
        / f"NHDPLUS_H_{huc8}_HU8_GDB"
        / f"NHDPLUS_H_{huc8}_HU8_GDB.gdb"
    )

    # Load stream reaches and their connectivity relationships.
    print(f"HUC8: {huc8}")
    print("Loading NHDPlus HR data...")

    streams = gpd.read_file(streams_path, layer=STREAM_LAYER)
    flow = gpd.read_file(streams_path, layer=FLOW_LAYER)

    print(f"Stream reaches loaded: {len(streams):,}")
    print(f"Flow relationships loaded: {len(flow):,}")

    # Assumption to validate: FromNHDPID -> ToNHDPID is downstream.
    G, length_by_reach, stats = build_reach_graph(streams, flow)

    # Report network diagnostics.
    print("\nStream network diagnostics")
    print(f"Reach nodes: {G.number_of_nodes():,}")
    print(f"Directed connections: {G.number_of_edges():,}")
    print(
        "Weakly connected components: "
        f"{nx.number_weakly_connected_components(G):,}"
    )
    print(f"Flow rows excluded: {stats['flow_rows_excluded']:,}")

    # Report reaches without incoming or outgoing edges.
    sources = [node for node in G if G.in_degree(node) == 0]
    outlets = [node for node in G if G.out_degree(node) == 0]

    print(f"Reaches with no incoming edge: {len(sources):,}")
    print(f"Reaches with no outgoing edge: {len(outlets):,}")


if __name__ == "__main__":
    main()
