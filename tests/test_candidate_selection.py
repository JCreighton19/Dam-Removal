"""Lightweight validation checks for select_candidate_stations.py.

This script only scopes/merges/type-filters raw downloaded WQP stations --
it does not filter by distance to a dam. Relevance to a dam removal is
determined entirely downstream, by network connectivity in
snap_features.py / link_dams_stations.py.

Run directly: `python tests/test_candidate_selection.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.select_candidate_stations import load_scope_stations, validate_stations


def _write_station_file(station_dir, huc8, rows):
    pd.DataFrame(rows).to_csv(station_dir / f"{huc8}.csv", index=False)


def test_load_scope_stations_merges_multiple_huc8_files():
    with tempfile.TemporaryDirectory() as tmp:
        station_dir = Path(tmp)
        _write_station_file(station_dir, "11111111", {
            "MonitoringLocationIdentifier": ["S1"],
            "MonitoringLocationTypeName": ["Stream"],
            "LatitudeMeasure": [42.0],
            "LongitudeMeasure": [-71.0],
        })
        _write_station_file(station_dir, "22222222", {
            "MonitoringLocationIdentifier": ["S2"],
            "MonitoringLocationTypeName": ["River/Stream"],
            "LatitudeMeasure": [43.0],
            "LongitudeMeasure": [-72.0],
        })

        stations = load_scope_stations(
            station_dir, ["11111111", "22222222"], ["Stream", "River/Stream"]
        )

        assert set(stations["MonitoringLocationIdentifier"]) == {"S1", "S2"}
        assert set(stations["source_huc8"]) == {"11111111", "22222222"}


def test_load_scope_stations_filters_by_type_and_drops_invalid_coords():
    with tempfile.TemporaryDirectory() as tmp:
        station_dir = Path(tmp)
        _write_station_file(station_dir, "11111111", {
            "MonitoringLocationIdentifier": ["S1", "S2", "S3"],
            "MonitoringLocationTypeName": ["Stream", "Lake", "Stream"],
            "LatitudeMeasure": [42.0, 42.1, 999.0],  # S3 has an invalid latitude
            "LongitudeMeasure": [-71.0, -71.1, -71.0],
        })

        stations = load_scope_stations(station_dir, ["11111111"], ["Stream", "River/Stream"])

        # S2 excluded (wrong type), S3 excluded (invalid coordinate).
        assert set(stations["MonitoringLocationIdentifier"]) == {"S1"}


def test_load_scope_stations_drops_duplicate_station_ids():
    with tempfile.TemporaryDirectory() as tmp:
        station_dir = Path(tmp)
        _write_station_file(station_dir, "11111111", {
            "MonitoringLocationIdentifier": ["S1", "S1"],
            "MonitoringLocationTypeName": ["Stream", "Stream"],
            "LatitudeMeasure": [42.0, 42.0],
            "LongitudeMeasure": [-71.0, -71.0],
        })

        stations = load_scope_stations(station_dir, ["11111111"], ["Stream"])

        assert len(stations) == 1


def test_load_scope_stations_raises_clear_error_on_missing_huc8_file():
    with tempfile.TemporaryDirectory() as tmp:
        station_dir = Path(tmp)
        try:
            load_scope_stations(station_dir, ["99999999"], ["Stream"])
            raise AssertionError("Expected FileNotFoundError for a missing station file.")
        except FileNotFoundError:
            pass


def test_validate_stations_rejects_empty_result():
    empty = pd.DataFrame(columns=["MonitoringLocationIdentifier", "source_huc8"])
    try:
        validate_stations(empty)
        raise AssertionError("Expected ValueError for an empty station table.")
    except ValueError:
        pass


def test_validate_stations_rejects_missing_columns():
    bad = pd.DataFrame({"MonitoringLocationIdentifier": ["S1"]})
    try:
        validate_stations(bad)
        raise AssertionError("Expected ValueError for missing columns.")
    except ValueError:
        pass


def test_validate_stations_accepts_well_formed_table():
    good = pd.DataFrame({
        "MonitoringLocationIdentifier": ["S1"],
        "source_huc8": ["11111111"],
    })
    validate_stations(good)  # should not raise


def test_no_distance_or_dam_columns_are_produced():
    """Locks in the removal of the distance-based filter: the output has
    no nearest-dam or distance columns, and no dam/WBD inputs are touched."""
    with tempfile.TemporaryDirectory() as tmp:
        station_dir = Path(tmp)
        _write_station_file(station_dir, "11111111", {
            "MonitoringLocationIdentifier": ["S1"],
            "MonitoringLocationTypeName": ["Stream"],
            "LatitudeMeasure": [42.0],
            "LongitudeMeasure": [-71.0],
        })
        stations = load_scope_stations(station_dir, ["11111111"], ["Stream"])
        assert "nearest_dam_id" not in stations.columns
        assert "distance_to_nearest_dam_km" not in stations.columns
        assert "geometry" not in stations.columns


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
