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

import networkx as nx

from etl.common.config import default_case_study, load_config
from etl.common.network import build_reach_graph
from etl.common.nhdplus import (
    find_huc8_geodatabase,
    is_hu4_fallback,
    read_nhdplus_layer,
)

STREAM_LAYER = "NHDFlowline"
FLOW_LAYER = "NHDPlusFlow"


def process_huc8(root, huc8):
    nhd_dir = root / "data" / "processed" / "nhdplus_extracted"
    streams_path = find_huc8_geodatabase(nhd_dir, huc8)

    # Load stream reaches and their connectivity relationships.
    print(f"\n=== HUC8: {huc8} ===")

    if is_hu4_fallback(streams_path, huc8):
        print(f"Using HU4 fallback package: {streams_path.name}")

    print("Loading NHDPlus HR data...")

    streams = read_nhdplus_layer(streams_path, STREAM_LAYER)
    flow = read_nhdplus_layer(streams_path, FLOW_LAYER)

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension). "
        "Used only to resolve the project root in this script.",
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

    outcomes = []

    for huc8 in huc8_list:
        try:
            process_huc8(cfg.project_root, huc8)
            outcomes.append((huc8, "ok"))
        except Exception as exc:
            print(f"FAILED HUC8 {huc8}: {exc}", file=sys.stderr)
            outcomes.append((huc8, f"failed: {exc}"))

    if len(huc8_list) > 1:
        print("\n=== Summary ===")
        for huc8, status in outcomes:
            print(f"  {huc8}: {status}")

    if any(status != "ok" for _, status in outcomes):
        raise RuntimeError("One or more HUC8s failed; see log above.")


if __name__ == "__main__":
    main()
