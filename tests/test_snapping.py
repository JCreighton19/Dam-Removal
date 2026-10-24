"""Checks for etl/common/snapping.py: datum handling, the reach decision
rules and the manual-override validation.

Run directly: `python tests/test_snapping.py`
"""

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from etl.common import snapping as sn

RULES = sn.SnapRules(search_radius_m=1000, review_distance_m=300)


def _candidates(rows):
    """rows: (NHDPlusID, distance_m, GNIS_Name, FType, totdasqkm)."""
    frame = pd.DataFrame(rows, columns=["NHDPlusID", "distance_m", "GNIS_Name", "FType", "totdasqkm"])
    frame["point_index"] = 0
    frame["stream_index"] = range(len(frame))
    return frame.sort_values("distance_m").reset_index(drop=True)


def _chosen(result, candidates):
    return int(candidates.set_index("stream_index").loc[result["stream_index"], "NHDPlusID"])


# --------------------------------------------------
# Datums
# --------------------------------------------------

def test_resolve_datum():
    assert sn.resolve_datum("NAD27") == (4267, False)
    assert sn.resolve_datum(" nad83 ") == (4269, False)
    assert sn.resolve_datum("UNKWN") == (sn.ASSUMED_DATUM_EPSG, True)
    assert sn.resolve_datum(None) == (sn.ASSUMED_DATUM_EPSG, True)
    assert sn.resolve_datum("ASTRO") == (None, False)  # no EPSG mapping: unsupported


def _op(description, accuracy, grids=()):
    return SimpleNamespace(
        description=description, accuracy=accuracy,
        grids=[SimpleNamespace(short_name=g, available=False) for g in grids],
    )


def test_missing_grid_for_a_ballpark_fallback_is_an_error():
    group = SimpleNamespace(
        transformers=[_op("Ballpark geographic offset from NAD27 to NAD83", -1.0)],
        unavailable_operations=[_op("NAD27 to NAD83 (1)", 0.15, ["us_noaa_conus.tif"])],
    )
    try:
        sn.choose_transformer(group, "NAD27", "NAD83")
    except sn.DatumTransformError as exc:
        assert "us_noaa_conus.tif" in str(exc) and "projsync" in str(exc)
    else:
        raise AssertionError("expected DatumTransformError")


def test_good_enough_transformation_is_used_despite_a_missing_better_grid():
    usable = _op("Inverse of NAD83 to WGS 84 (1)", 4.0)
    group = SimpleNamespace(transformers=[usable],
                            unavailable_operations=[_op("HPGN", 2.0, ["us_noaa_nehpgn.tif"])])
    assert sn.choose_transformer(group, "WGS 84", "NAD83") is usable


def test_ballpark_with_nothing_better_is_unsupported():
    group = SimpleNamespace(transformers=[_op("Ballpark geographic offset", -1.0)],
                            unavailable_operations=[])
    assert sn.choose_transformer(group, "Old Hawaiian", "NAD83") is None


def test_transform_points_flags_assumed_and_unsupported_datums():
    bounds = (-71, 41, -70, 42)
    out = sn.transform_points([-70.5, -70.5, -70.5], [41.5, 41.5, 41.5],
                              ["NAD83", "UNKWN", "ASTRO"], "EPSG:4269", bounds)
    assert out.loc[0, "x"] == -70.5 and out.loc[0, "y"] == 41.5  # NAD83 -> NAD83 unchanged
    assert not out.loc[0, "datum_assumed"] and out.loc[1, "datum_assumed"]
    assert out.loc[2, "datum_transform"] == "unsupported" and np.isnan(out.loc[2, "x"])

    wgs = sn.transform_points([-70.5], [41.5], None, "EPSG:4269", bounds, assumed_epsg=4326)
    assert abs(wgs.loc[0, "x"] + 70.5) < 1e-4 and wgs.loc[0, "datum_assumed"]


# --------------------------------------------------
# Reach decision
# --------------------------------------------------

def test_no_candidates_is_rejected():
    result = sn.decide_reach(_candidates([]), "X", np.nan, RULES)
    assert result["status"] == sn.REJECTED and result["reason"] == "no_flowline_within_limit"


def test_only_canals_nearby_is_rejected():
    candidates = _candidates([(1, 20, "Ditch", 336, 5.0), (2, 40, None, 566, np.nan)])
    result = sn.decide_reach(candidates, "X", np.nan, RULES)
    assert result["status"] == sn.REJECTED and result["reason"] == "only_unsuitable_flowline"


def test_nearer_canal_is_skipped_and_flagged():
    candidates = _candidates([(1, 10, "Canal", 336, 5.0), (2, 120, "Mill River", 460, 50.0)])
    result = sn.decide_reach(candidates, "MILL RIVER AT X", np.nan, RULES)
    assert result["status"] == sn.ACCEPTED and _chosen(result, candidates) == 2
    assert result["near_unsuitable_flowline"]


def test_nearest_stream_is_accepted():
    candidates = _candidates([(1, 30, None, 460, 10.0), (2, 200, None, 460, 12.0)])
    result = sn.decide_reach(candidates, "SITE 4", np.nan, RULES)
    assert result["status"] == sn.ACCEPTED and result["reason"] == "nearest_reach"
    assert _chosen(result, candidates) == 1 and result["runner_up_id"] == 2


def test_reported_area_picks_the_matching_reach():
    # Nearest is a small tributary; the reported area matches the main stem.
    candidates = _candidates([(1, 15, None, 460, 3.0), (2, 60, None, 460, 410.0)])
    result = sn.decide_reach(candidates, "SITE", 400.0, RULES)
    assert result["status"] == sn.ACCEPTED and result["reason"] == "drainage_area_match"
    assert _chosen(result, candidates) == 2


def test_reported_area_matching_only_a_far_reach_is_ambiguous():
    # On a small reach (5 m), area matches only a reach 700 m away: not moved.
    candidates = _candidates([(1, 5, None, 460, 3.0), (2, 700, None, 460, 410.0)])
    result = sn.decide_reach(candidates, "SITE", 400.0, RULES)
    assert result["status"] == sn.AMBIGUOUS and result["reason"] == "drainage_area_far"
    assert _chosen(result, candidates) == 1


def test_far_nearest_reach_with_matching_area_is_accepted():
    # The nearest reach itself agrees, so its distance doesn't block it.
    candidates = _candidates([(1, 450, None, 460, 410.0)])
    result = sn.decide_reach(candidates, "SITE", 400.0, RULES)
    assert result["status"] == sn.ACCEPTED and result["reason"] == "drainage_area_match"


def test_reported_area_matching_no_reach_is_ambiguous():
    candidates = _candidates([(1, 15, None, 460, 3.0), (2, 60, None, 460, 8.0)])
    result = sn.decide_reach(candidates, "SITE", 400.0, RULES)
    assert result["status"] == sn.AMBIGUOUS and result["reason"] == "drainage_area_mismatch"


def test_named_reach_is_preferred_within_review_distance():
    candidates = _candidates([(1, 20, "Otter Brook", 460, 5.0), (2, 90, "Mill River", 460, 6.0)])
    result = sn.decide_reach(candidates, "MILL R NR TOWN", np.nan, RULES)
    assert result["status"] == sn.ACCEPTED and result["reason"] == "named_reach_preferred"
    assert _chosen(result, candidates) == 2


def test_named_reach_too_far_is_ambiguous():
    candidates = _candidates([(1, 20, "Otter Brook", 460, 5.0), (2, 700, "Mill River", 460, 6.0)])
    result = sn.decide_reach(candidates, "MILL R NR TOWN", np.nan, RULES)
    assert result["status"] == sn.AMBIGUOUS and result["reason"] == "named_reach_far"


def test_confluence_without_evidence_is_ambiguous_and_with_a_name_accepted():
    candidates = _candidates([(1, 30, "Small Run", 460, 2.0), (2, 45, "Big River", 460, 900.0)])
    unnamed = sn.decide_reach(candidates, "SITE 7", np.nan, RULES)
    assert unnamed["status"] == sn.AMBIGUOUS and unnamed["reason"] == "confluence_ambiguous"
    named = sn.decide_reach(candidates, "BIG RIVER AT BRIDGE", np.nan, RULES)
    assert named["status"] == sn.ACCEPTED and _chosen(named, candidates) == 2


def test_far_snap_needs_support_unless_wide_water():
    stream = _candidates([(1, 450, None, 460, 50.0)])
    assert sn.decide_reach(stream, "SITE", np.nan, RULES)["reason"] == "far_without_support"
    wide = _candidates([(1, 450, None, 558, 50000.0)])
    result = sn.decide_reach(wide, "SITE", np.nan, RULES)
    assert result["status"] == sn.ACCEPTED and result["reason"] == "wide_water_far_snap"
    assert "far_snap" in result["evidence"]


def test_ties_prefer_the_main_stem():
    # A point on the node between a tributary and the main stem.
    candidates = _candidates([(7, 5.0, None, 460, 1.0), (3, 5.0, None, 460, 1.2)])
    result = sn.decide_reach(candidates, "SITE", np.nan, RULES)
    assert _chosen(result, candidates) == 3


# --------------------------------------------------
# Names and areas
# --------------------------------------------------

def test_non_riverine_name():
    flags = sn.non_riverine_name(["LAKE WINNIPESAUKEE MID", "MAXWELL POND-OUTLET",
                                  "KNOX MARSH RD BRIDGE", "MILL CREEK AT RT 9", None])
    assert flags.tolist() == [True, False, False, False, False]


def test_reported_area_km2():
    stations = pd.DataFrame({
        "DrainageAreaMeasure/MeasureValue": ["10", None, "-5"],
        "DrainageAreaMeasure/MeasureUnitCode": ["sq mi", None, "sq mi"],
        "ContributingDrainageAreaMeasure/MeasureValue": [None, "3", None],
        "ContributingDrainageAreaMeasure/MeasureUnitCode": [None, "sq km", None],
    })
    area = sn.reported_area_km2(stations)
    assert abs(area[0] - 25.8999) < 1e-3 and area[1] == 3.0 and np.isnan(area[2])


def test_placeholders_become_nan():
    assert np.isnan(sn.valid_area(pd.Series([-9999.0]))[0])
    assert np.isnan(sn.valid_slope(pd.Series([-9998.0]))[0])


# --------------------------------------------------
# Overrides
# --------------------------------------------------

def _overrides(text):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False)
    tmp.write("# comment\nfeature_kind,feature_id,action,nhdplus_id,reason,reviewed_by,reviewed_on\n"
              + text)
    tmp.close()
    return Path(tmp.name)


def test_valid_overrides_load():
    frame = sn.load_overrides(_overrides(
        "station,S1,reject,,tidal marsh,jc,2026-10-23\n"
        "dam,42,accept,1000123,main stem per map,jc,2026-10-23\n"))
    assert frame.loc[("dam", "42"), "nhdplus_id"] == 1000123
    assert frame.loc[("station", "S1"), "action"] == "reject"


def test_invalid_overrides_list_every_problem():
    try:
        sn.load_overrides(_overrides(
            "station,S1,accept,,no reach,jc,2026-10-23\n"
            "lake,S2,reject,,x,jc,2026-10-23\n"
            "dam,9,reject,123,,jc,2026-10-23\n"
            "station,S3,reject,,a,jc,2026-10-23\n"
            "station,S3,reject,,b,jc,2026-10-23\n"))
    except ValueError as exc:
        message = str(exc)
        for expected in ["accept needs an integer nhdplus_id", "feature_kind 'lake'",
                         "reason is empty", "reject must not give an nhdplus_id",
                         "more than one override"]:
            assert expected in message, expected
    else:
        raise AssertionError("expected ValueError")


def test_missing_overrides_file_is_empty():
    assert sn.load_overrides(Path("/nonexistent/overrides.csv")).empty


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
