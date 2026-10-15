"""Lightweight validation checks for etl.common.wbd.

Run directly: `python tests/test_wbd.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
from shapely.geometry import box

from etl.common.wbd import resolve_huc8_for_points


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


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
