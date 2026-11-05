"""Checks for etl/common/reservoir.py and etl/build_dam_reservoir.py.

Run directly: `python tests/test_dam_reservoir.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl import build_dam_reservoir as bdr
from etl.common import reservoir as rv


def _inputs():
    dams = pd.DataFrame({"AR_ID": [1.0, 2.0, 3.0, 4.0],
                         "Year_Removed": pd.array([2015, 2015, 2015, 2015], dtype="Int16")})
    reaches = pd.DataFrame({
        "AR_ID": ["1", "2", "3"], "huc8": ["1010001"] * 3,
        "match_status": ["accepted"] * 3,
        "reservoir_status": ["dam_at_outlet", "dam_at_outlet", "reservoir_not_at_outlet"],
        "reservoir_id": ["{abc-1}", "xyz", "zzz"], "reservoir_ftype": [436, 390, 390],
        "reservoir_fdate": pd.to_datetime(["2010-01-01", "2019-01-01", "2010-01-01"]),
    })
    areas = pd.DataFrame({"reservoir_id": ["ABC-1", "XYZ", "ZZZ"],
                          "polygon_area_km2": [0.4, 2.0, 9.0],
                          "polygon_area_source": ["AreaSqKm", "geometry_5070", "AreaSqKm"]})
    return dams, reaches, areas


def test_one_row_per_dam_measured_only_at_outlet():
    dams, reaches, areas = _inputs()
    table = bdr.build_table(dams, reaches, areas).set_index("AR_ID")
    assert bdr.validate(table.reset_index(), dams) == []
    assert list(table.index) == ["1", "2", "3", "4"]             # AR_IDs as text, all dams
    # Ids match across case and braces.
    assert table.loc["1", "polygon_area_km2"] == 0.4
    assert table.loc["1", "polygon_timing"] == rv.PRE_REMOVAL
    assert table.loc["2", "polygon_timing"] == "edited_after_removal"
    # Not at the outlet: no measured area even though the polygon has one.
    assert pd.isna(table.loc["3", "polygon_area_km2"]) and pd.isna(table.loc["3", "polygon_timing"])
    # Not in Stage 4: everything null, nothing imputed.
    assert table.loc["4"].isna().all()


def test_pre_removal_area_and_no_areas():
    dams, reaches, areas = _inputs()
    out = rv.attach_polygons(dams.assign(AR_ID=["1", "2", "3", "4"]), reaches, areas)
    assert out["polygon_area_km2_pre_removal"].notna().tolist() == [True, False, False, False]
    empty = rv.attach_polygons(dams.assign(AR_ID=["1", "2", "3", "4"]), reaches, None)
    assert empty["polygon_area_km2"].isna().all()
    assert empty["polygon_timing"].notna().tolist() == [True, True, False, False]


def test_validate_catches_mismatch():
    dams, reaches, areas = _inputs()
    table = bdr.build_table(dams, reaches, areas)
    errors = bdr.validate(table.iloc[:3], dams)
    assert any("AR_IDs differ" in e for e in errors)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
