"""Lightweight validation checks for etl.common.dams.filter_to_sampled_dams.

Run directly: `python tests/test_dams.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.common.config import CaseStudyConfig
from etl.common.dams import filter_to_sampled_dams


def _cfg(**raw):
    return CaseStudyConfig(name="test", project_root=Path("."), source_path=Path("test.yaml"), raw=raw)


DAMS = pd.DataFrame({"AR_ID": [1, 2, 3], "Dam_Name": ["A", "B", "C"]})


def test_keeps_only_sampled_dams():
    cfg = _cfg(sampled_dams=[{"AR_ID": "1"}, {"AR_ID": "3"}])
    assert filter_to_sampled_dams(DAMS, cfg)["AR_ID"].tolist() == [1, 3]


def test_unchanged_without_sampled_dams():
    assert len(filter_to_sampled_dams(DAMS, _cfg())) == 3


def test_opt_out_keeps_every_dam():
    cfg = _cfg(sampled_dams=[{"AR_ID": "1"}], restrict_to_sampled_dams=False)
    assert len(filter_to_sampled_dams(DAMS, cfg)) == 3


def test_missing_sampled_dam_raises():
    cfg = _cfg(sampled_dams=[{"AR_ID": "1"}, {"AR_ID": "99"}])
    try:
        filter_to_sampled_dams(DAMS, cfg)
        raise AssertionError("Expected ValueError for a sampled dam not in the source.")
    except ValueError as exc:
        assert "99" in str(exc)


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
