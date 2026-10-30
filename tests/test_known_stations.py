"""Checks for etl/check_known_stations.py: the two checks stay separate.
Avoiding a known wrong reach is a pass of avoids_wrong_reach only; it never
makes correct_reach 'confirmed'.

The real regression run (the seven audit stations against the cluster's
stage-3 outputs) is `python etl/check_known_stations.py` in
data/slurm/select_station_reaches.slurm.

Run directly: `python tests/test_known_stations.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl import check_known_stations as cks

COLUMNS = ["station_id", "wrong_nhdplus_id", "wrong_source", "correct_nhdplus_id",
           "correct_river_name", "reviewed_by", "reviewed_on", "notes"]


def _known(*rows):
    return pd.DataFrame(rows, columns=COLUMNS).astype("string")


def _reaches(*rows):
    return pd.DataFrame(rows, columns=["MonitoringLocationIdentifier", "snap_status",
                                       "snap_reason", "NHDPlusID", "reach_gnis_name"])


def _check(known, reaches):
    return cks.check(known, reaches).set_index("station_id")


def test_avoiding_the_wrong_reach_is_not_confirmation():
    known = _known(
        ("ON_WRONG", "11", "a", None, None, None, None, None),
        ("MOVED", "12", "a", None, None, None, None, None),
        ("AMBIG", "13", "a", None, None, None, None, None),
        ("NO_WRONG", None, None, None, None, None, None, None),
    )
    reaches = _reaches(("ON_WRONG", "accepted", "name_match", 11, "X River"),
                       ("MOVED", "accepted", "water_polygon", 99, "Y River"),
                       ("AMBIG", "ambiguous", "confluence_ambiguous", None, None),
                       ("NO_WRONG", "accepted", "own_catchment", 5, None))
    result = _check(known, reaches)
    assert result["avoids_wrong_reach"].to_dict() == {
        "ON_WRONG": "fail", "MOVED": "pass", "AMBIG": "pass", "NO_WRONG": "unknown"}
    # None was reviewed, so none is confirmed, however its first check went.
    assert set(result["correct_reach"]) == {"not_reviewed"}
    assert not cks.report(result.reset_index())  # ON_WRONG fails the run


def test_reviewed_answers_strengthen_the_check():
    known = _known(
        ("BY_ID", "11", "a", "20", None, "jc", "2026-11-01", None),
        ("BY_ID_WRONG", "11", "a", "21", None, "jc", "2026-11-01", None),
        ("BY_NAME", "12", "a", None, "Mill River", "jc", "2026-11-01", None),
        ("AMBIG_REVIEWED", "13", "a", "30", None, "jc", "2026-11-01", None),
    )
    reaches = _reaches(("BY_ID", "accepted", "name_match", 20, "Mill River"),
                       ("BY_ID_WRONG", "accepted", "name_match", 22, "Mill River"),
                       ("BY_NAME", "accepted", "name_match", 40, "Mill River"),
                       ("AMBIG_REVIEWED", "ambiguous", "far_without_support", None, None))
    result = _check(known, reaches)
    assert result["correct_reach"].to_dict() == {
        "BY_ID": "confirmed", "BY_ID_WRONG": "not_confirmed", "BY_NAME": "confirmed",
        "AMBIG_REVIEWED": "not_confirmed"}
    # BY_ID_WRONG avoided its known wrong reach but is still not confirmed.
    assert result.loc["BY_ID_WRONG", "avoids_wrong_reach"] == "pass"
    assert not cks.report(result.reset_index())


def test_several_wrong_reaches_and_filling():
    known = _known(("S1", None, None, None, None, None, None, "audit"),
                   ("S2", None, None, None, None, None, None, "audit"))
    found = [("S1", 11, "old.gpkg"), ("S1", 12, "diag.csv"), ("S1", 11, "diag.csv")]
    filled = cks.fill_wrong_reaches(known, found)
    s1 = filled[filled["station_id"] == "S1"]
    assert sorted(s1["wrong_nhdplus_id"]) == ["11", "12"] and set(s1["notes"]) == {"audit"}
    # S2 found nowhere: its placeholder row stays.
    assert filled[filled["station_id"] == "S2"]["wrong_nhdplus_id"].isna().all()

    result = _check(filled, _reaches(("S1", "accepted", "name_match", 12, None)))
    assert result.loc["S1", "avoids_wrong_reach"] == "fail"
    assert result.loc["S2", "found"] == False  # noqa: E712
    assert result.loc["S2", "avoids_wrong_reach"] == "unknown"


def test_wrong_reaches_from_diagnostics_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "snap_diagnostics_01010001.csv"
        pd.DataFrame({"feature_kind": ["station", "station", "dam"],
                      "feature_id": ["S1", "S2", "S1"],
                      "NHDPlusID": [11.0, None, 99.0]}).to_csv(path, index=False)
        found = cks.wrong_reaches_from_diagnostics([path], {"S1", "S2"})
        assert found == [("S1", 11, str(path))]


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
