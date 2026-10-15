"""Lightweight validation checks for select_candidate_stations.py.

Run directly: `python tests/test_candidate_selection.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd

from etl.select_candidate_stations import (
    build_sensitivity_report,
    compute_nearest_dam,
    validate_candidates,
)


def _dams():
    # ~55.5 km apart (0.5 degrees of latitude).
    return gpd.GeoDataFrame(
        {"AR_ID": ["A1", "B1"]},
        geometry=gpd.points_from_xy([-71.0, -71.0], [42.0, 42.5]),
        crs="EPSG:4326",
    )


def _stations():
    return gpd.GeoDataFrame(
        {
            "MonitoringLocationIdentifier": ["S1", "S2", "S3"],
            "source_huc8": ["11111111", "11111111", "11111111"],
        },
        geometry=gpd.points_from_xy(
            [-71.0, -71.0, -71.0],
            [42.01, 42.49, 42.25],
        ),
        crs="EPSG:4326",
    )


def test_compute_nearest_dam_picks_closest_and_reasonable_distance():
    joined = compute_nearest_dam(_stations(), _dams())
    by_id = joined.set_index("MonitoringLocationIdentifier")

    # S1 sits ~1.1 km from A1, far from B1 -> nearest should be A1.
    assert by_id.loc["S1", "nearest_dam_id"] == "A1"
    assert 0.5 < by_id.loc["S1", "distance_to_nearest_dam_km"] < 2.0

    # S2 sits ~1.1 km from B1 -> nearest should be B1.
    assert by_id.loc["S2", "nearest_dam_id"] == "B1"
    assert 0.5 < by_id.loc["S2", "distance_to_nearest_dam_km"] < 2.0

    # S3 sits roughly halfway (~27-28 km either way).
    assert 20.0 < by_id.loc["S3", "distance_to_nearest_dam_km"] < 35.0

    # Required columns for downstream consumers are present.
    for col in ("MonitoringLocationIdentifier", "source_huc8", "nearest_dam_id", "distance_to_nearest_dam_km"):
        assert col in joined.columns


def test_sensitivity_report_counts_increase_with_distance():
    joined = compute_nearest_dam(_stations(), _dams())
    report = build_sensitivity_report(joined, threshold_km=5, sensitivity_km=[1, 5, 50])

    counts = report.set_index("distance_km")["candidate_station_count"]

    # At 1 km: only stations within 1 km of their nearest dam qualify.
    assert counts.loc[1] == 0
    # At 5 km: S1 and S2 (the ~1.1 km ones) qualify; S3 (~25-30 km) does not.
    assert counts.loc[5] == 2
    # At 50 km: all three qualify.
    assert counts.loc[50] == 3
    # Monotonically non-decreasing as the threshold widens.
    assert counts.loc[1] <= counts.loc[5] <= counts.loc[50]

    selected_row = report[report["distance_km"] == 5].iloc[0]
    assert bool(selected_row["is_selected_threshold"]) is True


def test_validate_candidates_rejects_empty_result():
    empty = pd.DataFrame(columns=[
        "MonitoringLocationIdentifier", "source_huc8",
        "nearest_dam_id", "distance_to_nearest_dam_km",
    ])
    try:
        validate_candidates(empty, threshold_km=5)
        raise AssertionError("Expected ValueError for an empty candidate table.")
    except ValueError:
        pass


def test_validate_candidates_rejects_missing_columns():
    bad = pd.DataFrame({"MonitoringLocationIdentifier": ["S1"]})
    try:
        validate_candidates(bad, threshold_km=5)
        raise AssertionError("Expected ValueError for missing columns.")
    except ValueError:
        pass


def test_validate_candidates_rejects_negative_distance():
    bad = pd.DataFrame({
        "MonitoringLocationIdentifier": ["S1"],
        "source_huc8": ["11111111"],
        "nearest_dam_id": ["A1"],
        "distance_to_nearest_dam_km": [-1.0],
    })
    try:
        validate_candidates(bad, threshold_km=5)
        raise AssertionError("Expected ValueError for a negative distance.")
    except ValueError:
        pass


def test_validate_candidates_accepts_well_formed_table():
    good = pd.DataFrame({
        "MonitoringLocationIdentifier": ["S1"],
        "source_huc8": ["11111111"],
        "nearest_dam_id": ["A1"],
        "distance_to_nearest_dam_km": [1.2],
    })
    validate_candidates(good, threshold_km=5)  # should not raise


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
