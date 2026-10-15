"""Shared directed stream-reach network construction from NHDPlus HR data.

This logic previously existed, nearly identically, in both
`build_stream_network.py` and `link_dams_stations.py`. It now lives in one
place so both (and any future per-watershed script) build the exact same
graph from the exact same rules.

Assumption carried over from both originals, and worth re-stating: this
treats `FromNHDPID -> ToNHDPID` as the downstream direction. That was
spot-checked visually against mapped flow direction in
`notebooks/stream_network_exploration.ipynb`; it has not been validated
against the official NHDPlus documentation.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def build_reach_graph(streams: pd.DataFrame, flow: pd.DataFrame):
    """Build a directed reach graph from NHDFlowline/NHDPlusFlow tables.

    Parameters
    ----------
    streams : DataFrame-like with an "NHDPlusID" column and a "LengthKM" column.
    flow : DataFrame-like with "FromNHDPID" and "ToNHDPID" columns.

    Returns
    -------
    (graph, length_by_reach, stats)
        graph : a networkx.DiGraph with one node per reach and one edge per
            valid flow connection, weighted by the length of the reach the
            edge exits.
        length_by_reach : dict mapping NHDPlusID -> LengthKM.
        stats : dict with reach/flow row counts, for diagnostic logging.
    """
    if "NHDPlusID" not in streams.columns:
        raise ValueError("streams frame is missing an 'NHDPlusID' column.")
    if "LengthKM" not in streams.columns:
        raise ValueError("streams frame is missing a 'LengthKM' column.")
    if not {"FromNHDPID", "ToNHDPID"}.issubset(flow.columns):
        raise ValueError("flow frame must contain 'FromNHDPID' and 'ToNHDPID'.")

    streams = streams.copy()
    streams["NHDPlusID"] = pd.to_numeric(streams["NHDPlusID"], errors="coerce")
    streams = streams.dropna(subset=["NHDPlusID"]).copy()
    streams["NHDPlusID"] = streams["NHDPlusID"].astype("int64")
    streams = streams.drop_duplicates("NHDPlusID")

    length_by_reach = streams.set_index("NHDPlusID")["LengthKM"].to_dict()
    stream_ids = set(length_by_reach)

    flow = flow.copy()
    flow["FromNHDPID"] = pd.to_numeric(flow["FromNHDPID"], errors="coerce")
    flow["ToNHDPID"] = pd.to_numeric(flow["ToNHDPID"], errors="coerce")

    valid = (
        flow["FromNHDPID"].notna()
        & flow["ToNHDPID"].notna()
        & flow["FromNHDPID"].ne(0)
        & flow["ToNHDPID"].ne(0)
        & flow["FromNHDPID"].isin(stream_ids)
        & flow["ToNHDPID"].isin(stream_ids)
    )
    valid_flow = flow.loc[valid].copy()
    valid_flow["FromNHDPID"] = valid_flow["FromNHDPID"].astype("int64")
    valid_flow["ToNHDPID"] = valid_flow["ToNHDPID"].astype("int64")

    graph = nx.DiGraph()
    graph.add_nodes_from(stream_ids)
    graph.add_edges_from(zip(valid_flow["FromNHDPID"], valid_flow["ToNHDPID"]))

    # Weight each edge by the length of the reach being exited, so the
    # shortest weighted path approximates along-network travel distance.
    for u, v in graph.edges:
        graph[u][v]["weight"] = float(length_by_reach.get(u, 0) or 0)

    stats = {
        "stream_count": len(length_by_reach),
        "flow_rows_total": len(flow),
        "flow_rows_valid": int(valid.sum()),
        "flow_rows_excluded": int((~valid).sum()),
    }

    return graph, length_by_reach, stats
