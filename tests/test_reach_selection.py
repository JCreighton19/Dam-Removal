"""Checks for etl/common/reach_selection.py (rules, on hand-built
candidate rows) and etl/select_station_reaches.py (end to end, on the
stage-2 synthetic project in tests/test_station_candidates.py).

Run directly: `python tests/test_reach_selection.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl import select_station_reaches as ssr
from etl.common import reach_selection as rs
from etl.common.config import PROJECT_ROOT_ENV_VAR
from tests.test_station_candidates import HUC8, NO_PACKAGE_HUC8, _build_project, _run

RULES = rs.SelectionRules()


def _cands(*rows):
    """Rows: (id, distance_m, totdasqkm, ftype, name_match, area_agrees,
    in_point_polygon, catchment_relation)."""
    frame = pd.DataFrame(rows, columns=["NHDPlusID", "distance_m", "totdasqkm", "FType",
                                        "name_match", "area_agrees", "in_point_polygon",
                                        "catchment_relation"])
    frame["eligible_ftype"] = frame["FType"].isin([460, 558])
    frame["GNIS_Name"] = None
    frame["generation_methods"] = "nearby"
    return frame


def _point(**values):
    return pd.Series({"point_polygon_distance_m": None, "in_sink_catchment": False, **values})


def _decide(candidates, **point):
    return rs.decide_station_reach(candidates, _point(**point), RULES)


def test_no_eligible_candidate_is_rejected():
    assert _decide(_cands((9, 10.0, 5.0, 336, None, None, False, None)))["status"] == "rejected"
    assert _decide(_cands(), in_sink_catchment=True)["reason"] == "sink_catchment_no_reach"


def test_wide_river_path_beats_closer_tributary_with_name_support():
    decision = _decide(_cands((2, 20.0, 4.0, 460, False, None, False, "upstream"),
                              (1, 310.0, 100.0, 558, True, None, True, "own")),
                       point_polygon_distance_m=0.0)
    assert (decision["status"], decision["reason"]) == ("accepted", "name_match")
    assert decision["provisional_nhdplus_id"] == 1 and decision["runner_up_id"] == 2
    assert "confluence_nearby" in decision["evidence"]


def test_confluence_without_support_is_ambiguous():
    decision = _decide(_cands((2, 20.0, 4.0, 460, None, None, False, None),
                              (1, 30.0, 100.0, 460, None, None, False, "own")))
    assert (decision["status"], decision["reason"]) == ("ambiguous", "confluence_ambiguous")


def test_paths_of_different_rivers_in_one_polygon_need_support():
    decision = _decide(_cands((1, 5.0, 70.0, 558, None, None, True, "own"),
                              (3, 390.0, 8800.0, 558, None, None, True, None)),
                       point_polygon_distance_m=0.0)
    assert decision["reason"] == "confluence_ambiguous"


def test_drainage_area_decides_and_mismatch_is_ambiguous():
    decision = _decide(_cands((2, 20.0, 4.0, 460, None, False, False, None),
                              (1, 120.0, 100.0, 460, None, True, False, None)))
    assert (decision["reason"], decision["provisional_nhdplus_id"]) == ("drainage_area_match", 1)
    mismatch = _decide(_cands((2, 20.0, 4.0, 460, None, False, False, None)))
    assert (mismatch["status"], mismatch["reason"]) == ("ambiguous", "drainage_area_mismatch")


def test_far_without_support_is_ambiguous_not_forced():
    decision = _decide(_cands((1, 450.0, 100.0, 460, None, None, False, "own")))
    assert (decision["status"], decision["reason"]) == ("ambiguous", "far_without_support")
    assert decision["provisional_nhdplus_id"] == 1  # kept for review only
    assert _decide(_cands((1, 800.0, 100.0, 460, None, None, False, None)))["reason"] \
        == "far_without_support"


def test_polygon_edge_with_closer_reach_is_ambiguous():
    decision = _decide(_cands((2, 8.0, 100.0, 460, None, None, False, "own"),
                              (1, 200.0, 100.0, 558, None, None, True, None)),
                       point_polygon_distance_m=15.0)
    assert decision["reason"] == "polygon_edge_ambiguous"


def test_sink_catchment_needs_support():
    decision = _decide(_cands((1, 50.0, 100.0, 460, None, None, False, None)),
                       in_sink_catchment=True)
    assert decision["reason"] == "sink_catchment"


def test_own_catchment_and_nearest():
    assert _decide(_cands((1, 40.0, 10.0, 460, None, None, False, "own")))["reason"] \
        == "own_catchment"
    assert _decide(_cands((1, 40.0, 10.0, 460, None, None, False, None)))["reason"] \
        == "nearest_plausible_reach"


def _select(root, *extra):
    argv = ["select_station_reaches.py", "--case-study", "cand_test",
            "--candidate-dir", str(root / "out"), "--output-dir", str(root / "reaches"), *extra]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        ssr.main()
    frame = pd.read_parquet(root / "reaches" / f"station_reaches_{HUC8}.parquet")
    return frame.set_index("MonitoringLocationIdentifier")


def test_end_to_end_selection_and_network_position():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        _run(root)
        out = _select(root)

        expected = {
            "S_BANK": ("accepted", "name_match", 1),
            "S_LAKE": ("accepted", "water_polygon", 4),
            "S_TRIB": ("accepted", "name_match", 2),
            "S_POND": ("rejected", "sink_catchment_no_reach", None),
            "S_ASTRO": ("rejected", "datum_unsupported", None),
        }
        for station, (status, reason, reach) in expected.items():
            row = out.loc[station]
            assert (row["snap_status"], row["snap_reason"]) == (status, reason), \
                (station, row["snap_reason"])
            assert (pd.isna(row["NHDPlusID"]) if reach is None else row["NHDPlusID"] == reach)
        assert "far_snap" in out.loc["S_TRIB", "snap_evidence"]

        # Network position: the reported point is projected onto Mill River
        # (lon -70.00 -> -69.95); the reported coordinates are unchanged.
        bank = out.loc["S_BANK"]
        assert abs(bank["snap_fraction"] - (70.00 - 69.9701) / 0.05) < 0.01
        assert abs(bank["snapped_lat"] - 45.0) < 1e-5 and abs(bank["snapped_lon"] + 69.9701) < 1e-4
        assert bank["LatitudeMeasure"] == 45.0028 and bank["LongitudeMeasure"] == -69.9701
        assert 300 < bank["snap_distance_m"] < 330
        assert bank["reach_gnis_name"] == "Mill River" and bank["reach_ftype"] == 558
        assert pd.isna(out.loc["S_POND", "snap_fraction"])

        other = pd.read_parquet(root / "reaches" / f"station_reaches_{NO_PACKAGE_HUC8}.parquet")
        assert other["snap_reason"].tolist() == ["no_nhdplus_package"]


def test_manual_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        _run(root)
        overrides = root / "overrides.csv"
        overrides.write_text(
            "feature_kind,feature_id,action,nhdplus_id,reason,reviewed_by,reviewed_on\n"
            "station,S_LAKE,reject,,not a river site,jc,2026-10-30\n"
            "station,S_TRIB,accept,1,on the main stem (map),jc,2026-10-30\n")
        out = _select(root, "--overrides", str(overrides))
        lake, trib = out.loc["S_LAKE"], out.loc["S_TRIB"]
        assert (lake["snap_status"], lake["snap_reason"]) == ("rejected", "manual_reject")
        assert (lake["auto_status"], lake["auto_reason"]) == ("accepted", "water_polygon")
        assert (trib["snap_status"], trib["NHDPlusID"]) == ("accepted", 1)
        assert trib["match_method"] == "manual_override" and trib["auto_reason"] == "name_match"
        assert abs(trib["snapped_lat"] - 45.0) < 1e-5  # placed on Mill River


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
