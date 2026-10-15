"""Lightweight validation checks for etl.common.links.

Run directly: `python tests/test_links.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common.links import connected_only, load_links


def _write_links(path, rows):
    pd.DataFrame(rows, columns=[
        "dam_id", "dam_name", "station_id", "year_removed", "relationship"
    ]).to_csv(path, index=False)


def test_load_links_raises_on_missing_columns():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "links.csv"
        pd.DataFrame({"dam_id": ["1"]}).to_csv(path, index=False)
        try:
            load_links(path)
            raise AssertionError("Expected ValueError for missing columns.")
        except ValueError:
            pass


def test_connected_only_excludes_unconnected_rows():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "links.csv"
        _write_links(path, [
            ("D1", "Dam One", "S1", 2017, "upstream"),
            ("D1", "Dam One", "S2", 2017, "downstream"),
            ("D1", "Dam One", "S3", 2017, "same_reach"),
            ("D1", "Dam One", "S4", 2017, "unconnected"),
        ])

        links = load_links(path)
        connected = connected_only(links)

        assert set(connected["station_id"]) == {"S1", "S2", "S3"}
        assert "S4" not in set(connected["station_id"])


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
