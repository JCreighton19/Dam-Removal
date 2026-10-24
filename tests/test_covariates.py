"""Checks for etl/common/covariates.py (reuse of land-cover/climate rows).

The stage-level reuse is checked in tests/test_load_land_cover.py.

Run directly: `python tests/test_covariates.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common import covariates as cv


def test_hash_depends_only_on_the_catchment_set():
    a = cv.catchment_hashes(pd.DataFrame({"station_id": ["S", "S"], "catchment_id": [2, 1]}))
    b = cv.catchment_hashes(pd.DataFrame({"station_id": ["S", "S"], "catchment_id": [1, 2]}))
    c = cv.catchment_hashes(pd.DataFrame({"station_id": ["S"], "catchment_id": [1]}))
    assert a["S"] == b["S"] != c["S"]


def test_previous_membership_is_kept_only_for_legacy_tables():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        membership = pd.DataFrame({"station_id": ["S"], "catchment_id": [1]})
        membership.to_parquet(out / "station_catchment.parquet")

        # A table with catchment_hash doesn't need the old membership.
        pd.DataFrame({"station_id": ["S"], "catchment_hash": ["x"]}).to_parquet(
            out / "climate.parquet")
        assert not cv.keep_previous_membership(out)

        # A legacy table does: the current membership becomes .prev, once.
        pd.DataFrame({"station_id": ["S"], "year": [2010]}).to_parquet(out / "land_cover.parquet")
        assert cv.keep_previous_membership(out)
        assert (out / cv.PREVIOUS_MEMBERSHIP).exists()
        membership.to_parquet(out / "station_catchment.parquet")
        assert not cv.keep_previous_membership(out)  # an older copy is already kept


def test_window_years_nearest_if_outside():
    assert cv.window_years(1970, 1, 1, 1985, 2025) == set()
    assert cv.window_years(1970, 1, 1, 1985, 2025, nearest_if_outside=True) == {1985}


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
