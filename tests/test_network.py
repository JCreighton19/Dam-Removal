"""Lightweight validation checks for etl.common.network.build_reach_graph.

Run directly: `python tests/test_network.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common.network import build_reach_graph


def _streams(rows):
    return pd.DataFrame(rows, columns=["NHDPlusID", "LengthKM"])


def _flow(rows):
    return pd.DataFrame(rows, columns=["FromNHDPID", "ToNHDPID"])


def test_builds_expected_nodes_and_edges():
    streams = _streams([(1, 2.0), (2, 3.0), (3, 1.5)])
    flow = _flow([(1, 2), (2, 3)])

    graph, length_by_reach, stats = build_reach_graph(streams, flow)

    assert set(graph.nodes) == {1, 2, 3}
    assert set(graph.edges) == {(1, 2), (2, 3)}
    assert length_by_reach == {1: 2.0, 2: 3.0, 3: 1.5}
    assert stats["stream_count"] == 3
    assert stats["flow_rows_total"] == 2
    assert stats["flow_rows_valid"] == 2
    assert stats["flow_rows_excluded"] == 0


def test_edge_weight_is_length_of_source_reach():
    streams = _streams([(1, 5.0), (2, 9.0)])
    flow = _flow([(1, 2)])

    graph, _, _ = build_reach_graph(streams, flow)

    assert graph[1][2]["weight"] == 5.0


def test_excludes_flow_rows_referencing_unknown_reaches():
    streams = _streams([(1, 2.0), (2, 3.0)])
    # 2 -> 99 references a reach that doesn't exist in `streams`.
    flow = _flow([(1, 2), (2, 99)])

    graph, _, stats = build_reach_graph(streams, flow)

    assert set(graph.edges) == {(1, 2)}
    assert stats["flow_rows_valid"] == 1
    assert stats["flow_rows_excluded"] == 1


def test_excludes_zero_ids_and_duplicate_stream_rows():
    # Reach 1 appears twice (duplicate should be dropped); 0 is a sentinel,
    # not a real reach id, and should never produce an edge.
    streams = _streams([(1, 2.0), (1, 2.0), (2, 3.0)])
    flow = _flow([(0, 2), (1, 0), (1, 2)])

    graph, length_by_reach, stats = build_reach_graph(streams, flow)

    assert length_by_reach == {1: 2.0, 2: 3.0}
    assert set(graph.edges) == {(1, 2)}
    assert stats["flow_rows_excluded"] == 2


def test_raises_clear_error_on_missing_columns():
    streams = pd.DataFrame({"NHDPlusID": [1]})  # no LengthKM
    flow = _flow([(1, 1)])
    try:
        build_reach_graph(streams, flow)
        raise AssertionError("Expected a ValueError for a missing LengthKM column.")
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
