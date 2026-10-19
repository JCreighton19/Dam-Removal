"""Lightweight validation checks for etl.common.wbd.

Run directly: `python tests/test_wbd.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
from shapely.geometry import box

import pyogrio

from etl.common.wbd import load_huc8_watersheds, resolve_huc8_for_points
from etl.slim_wbd import slim_wbd


def _write_wbd(path):
    watersheds = gpd.GeoDataFrame(
        {"huc8": ["11111111", "22222222"]},
        geometry=[
            box(-72.0, 42.0, -71.0, 43.0),   # west watershed
            box(-71.0, 42.0, -70.0, 43.0),   # east watershed, shares a border
        ],
        crs="EPSG:4326",
    )
    watersheds.to_file(path, layer="WBDHU8", driver="GPKG")


def test_resolve_huc8_for_points_matches_correct_watershed():
    with tempfile.TemporaryDirectory() as tmp:
        wbd_path = Path(tmp) / "wbd.gpkg"
        _write_wbd(wbd_path)

        points = gpd.GeoDataFrame(
            {"AR_ID": ["A", "B"]},
            geometry=gpd.points_from_xy([-71.5, -70.5], [42.5, 42.5]),
            crs="EPSG:4326",
        )

        resolved = resolve_huc8_for_points(points, wbd_path)
        by_id = resolved.set_index("AR_ID")

        assert by_id.loc["A", "resolved_huc8"] == "11111111"
        assert by_id.loc["B", "resolved_huc8"] == "22222222"


def test_resolve_huc8_for_points_leaves_unmatched_points_null():
    with tempfile.TemporaryDirectory() as tmp:
        wbd_path = Path(tmp) / "wbd.gpkg"
        _write_wbd(wbd_path)

        points = gpd.GeoDataFrame(
            {"AR_ID": ["OUTSIDE"]},
            geometry=gpd.points_from_xy([10.0], [10.0]),  # nowhere near either watershed
            crs="EPSG:4326",
        )

        resolved = resolve_huc8_for_points(points, wbd_path)

        assert resolved.iloc[0]["resolved_huc8"] is None or (
            resolved.iloc[0]["resolved_huc8"] != resolved.iloc[0]["resolved_huc8"]  # NaN
        )


def test_load_huc8_watersheds_reads_only_requested():
    with tempfile.TemporaryDirectory() as tmp:
        wbd_path = Path(tmp) / "wbd.gpkg"
        _write_wbd(wbd_path)
        loaded = load_huc8_watersheds(wbd_path, ["22222222"])
        assert loaded["huc8"].tolist() == ["22222222"]
        try:
            load_huc8_watersheds(wbd_path, ["33333333"])
        except ValueError as exc:
            assert "33333333" in str(exc)
        else:
            raise AssertionError("expected a missing-HUC8 error")


def test_slim_wbd_keeps_only_hu8():
    with tempfile.TemporaryDirectory() as tmp:
        wbd_path = Path(tmp) / "wbd.gpkg"
        _write_wbd(wbd_path)
        gpd.GeoDataFrame({"huc12": ["111111110101"]}, geometry=[box(-72, 42, -71.5, 42.5)],
                         crs="EPSG:4326").to_file(wbd_path, layer="WBDHU12", driver="GPKG")
        slim_wbd(wbd_path)
        assert [name for name, _ in pyogrio.list_layers(wbd_path)] == ["WBDHU8"]
        assert len(load_huc8_watersheds(wbd_path, ["11111111", "22222222"])) == 2
        assert slim_wbd(wbd_path) == "already slim"


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
