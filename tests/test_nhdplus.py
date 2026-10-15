"""Lightweight validation checks for etl.common.nhdplus.

Run directly: `python tests/test_nhdplus.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.common.nhdplus import find_huc8_geodatabase, is_hu4_fallback


def _make_gdb_dir(root, dirname):
    gdb = root / dirname / f"{dirname}.gdb"
    gdb.mkdir(parents=True)
    return gdb


def test_finds_exact_hu8_package():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        expected = _make_gdb_dir(root, "NHDPLUS_H_18070301_HU8_GDB")
        # An unrelated HU4 package for a different HUC4 should be ignored.
        _make_gdb_dir(root, "NHDPLUS_H_0106_HU4_GDB")

        found = find_huc8_geodatabase(root, "18070301")
        assert found == expected
        assert not is_hu4_fallback(found, "18070301")


def test_falls_back_to_hu4_package_when_no_hu8_exists():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        expected = _make_gdb_dir(root, "NHDPLUS_H_1807_HU4_GDB")

        found = find_huc8_geodatabase(root, "18070301")
        assert found == expected
        assert is_hu4_fallback(found, "18070301")


def test_prefers_exact_hu8_over_hu4_fallback_when_both_exist():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exact = _make_gdb_dir(root, "NHDPLUS_H_18070301_HU8_GDB")
        _make_gdb_dir(root, "NHDPLUS_H_1807_HU4_GDB")

        found = find_huc8_geodatabase(root, "18070301")
        assert found == exact


def test_substring_search_catches_unexpected_directory_naming():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        # Named differently than either expected pattern, but still
        # contains the HUC4 code -- the loose fallback should find it.
        weird = root / "extracted_1807_data" / "weird_1807_name.gdb"
        weird.mkdir(parents=True)

        found = find_huc8_geodatabase(root, "18070301")
        assert found == weird


def test_raises_clear_error_when_nothing_matches():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_gdb_dir(root, "NHDPLUS_H_0106_HU4_GDB")  # unrelated HUC4

        try:
            find_huc8_geodatabase(root, "18070301")
            raise AssertionError("Expected FileNotFoundError.")
        except FileNotFoundError as exc:
            assert "18070301" in str(exc)
            assert "1807" in str(exc)


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
