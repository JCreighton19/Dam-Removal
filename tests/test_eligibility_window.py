"""Lightweight validation checks for report_pilot_eligibility.py's windowing
and eligibility-rollup logic -- the part of stage 1 with the most subtle
correctness requirements.

Run directly: `python tests/test_eligibility_window.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl.report_pilot_eligibility import (
    build_characteristic_report,
    build_dam_report,
    build_observation_counts,
    build_pair_report,
    classify_period,
    prepare_links,
    prepare_results,
)


def _links(rows):
    columns = ["dam_id", "dam_name", "station_id", "year_removed", "relationship"]
    return pd.DataFrame(rows, columns=columns)


def _results(rows):
    columns = ["MonitoringLocationIdentifier", "CharacteristicName", "ActivityStartDate"]
    return pd.DataFrame(rows, columns=columns)


def test_classify_period_excludes_removal_year_itself():
    # removal_year = 2017, years_before = years_after = 3
    assert classify_period(2017, 2017, 3, 3) is None
    assert classify_period(2016, 2017, 3, 3) == "pre"
    assert classify_period(2014, 2017, 3, 3) == "pre"
    assert classify_period(2013, 2017, 3, 3) is None  # outside the 3-year pre window
    assert classify_period(2018, 2017, 3, 3) == "post"
    assert classify_period(2020, 2017, 3, 3) == "post"
    assert classify_period(2021, 2017, 3, 3) is None  # outside the 3-year post window


def test_prepare_links_excludes_unconnected_and_missing_year():
    links = _links([
        ("D1", "Dam One", "S1", 2017, "upstream"),
        ("D1", "Dam One", "S2", 2017, "unconnected"),   # excluded: not network-connected
        ("D2", "Dam Two", "S3", None, "downstream"),     # excluded: missing removal year
        ("D3", "Dam Three", "S4", 0, "same_reach"),      # excluded: zero is not a real year
    ])

    pairs, excluded_missing_year = prepare_links(links)

    assert set(pairs["dam_id"]) == {"D1"}
    assert set(pairs["station_id"]) == {"S1"}
    assert set(excluded_missing_year["dam_id"]) == {"D2", "D3"}


def test_dam_does_not_count_both_from_two_different_stations():
    """The core requirement: a dam with one station covering only the
    pre-period and a DIFFERENT station covering only the post-period must
    NOT be reported as having a comparable before/after record."""
    links = _links([
        ("D1", "Dam One", "S1", 2017, "upstream"),    # pre-only station
        ("D1", "Dam One", "S2", 2017, "downstream"),  # post-only station
    ])
    results = _results([
        ("S1", "Turbidity", "2015-06-01"),  # pre window (2014-2016)
        ("S2", "Turbidity", "2019-06-01"),  # post window (2018-2020)
    ])

    pairs, _ = prepare_links(links)
    filtered = prepare_results(results, ["Turbidity"])
    counts = build_observation_counts(pairs, filtered, years_before=3, years_after=3)
    pair_report = build_pair_report(pairs, counts)
    dam_report = build_dam_report(pair_report)

    pair_by_station = pair_report.set_index("station_id")
    assert bool(pair_by_station.loc["S1", "has_pre"]) is True
    assert bool(pair_by_station.loc["S1", "has_post"]) is False
    assert bool(pair_by_station.loc["S1", "has_both"]) is False

    assert bool(pair_by_station.loc["S2", "has_pre"]) is False
    assert bool(pair_by_station.loc["S2", "has_post"]) is True
    assert bool(pair_by_station.loc["S2", "has_both"]) is False

    dam_row = dam_report[dam_report["dam_id"] == "D1"].iloc[0]
    # Looser, descriptive counts: the dam has SOME pre and SOME post coverage.
    assert bool(dam_row["has_pre"]) is True
    assert bool(dam_row["has_post"]) is True
    # But the strict eligibility flag must be False: no single station covers both.
    assert bool(dam_row["has_both"]) is False
    assert dam_row["pairs_with_both"] == 0


def test_dam_counts_both_when_same_station_covers_both_periods():
    links = _links([
        ("D1", "Dam One", "S1", 2017, "same_reach"),
    ])
    results = _results([
        ("S1", "Turbidity", "2015-06-01"),  # pre
        ("S1", "Turbidity", "2019-06-01"),  # post
    ])

    pairs, _ = prepare_links(links)
    filtered = prepare_results(results, ["Turbidity"])
    counts = build_observation_counts(pairs, filtered, years_before=3, years_after=3)
    pair_report = build_pair_report(pairs, counts)
    dam_report = build_dam_report(pair_report)

    assert bool(pair_report.iloc[0]["has_both"]) is True
    dam_row = dam_report[dam_report["dam_id"] == "D1"].iloc[0]
    assert bool(dam_row["has_both"]) is True
    assert dam_row["pairs_with_both"] == 1


def test_characteristic_report_keeps_characteristics_separate():
    """A pair with pre-period Turbidity and post-period pH should NOT be
    reported as 'both' for either characteristic individually -- the
    per-characteristic report must not pool across characteristics."""
    links = _links([
        ("D1", "Dam One", "S1", 2017, "same_reach"),
    ])
    results = _results([
        ("S1", "Turbidity", "2015-06-01"),  # pre
        ("S1", "pH", "2019-06-01"),         # post, different characteristic
    ])

    pairs, _ = prepare_links(links)
    filtered = prepare_results(results, ["Turbidity", "pH"])
    counts = build_observation_counts(pairs, filtered, years_before=3, years_after=3)
    characteristic_report = build_characteristic_report(counts)

    by_characteristic = characteristic_report.set_index("CharacteristicName")
    assert by_characteristic.loc["Turbidity", "pairs_with_pre"] == 1
    assert by_characteristic.loc["Turbidity", "pairs_with_post"] == 0
    assert by_characteristic.loc["Turbidity", "pairs_with_both"] == 0
    assert by_characteristic.loc["pH", "pairs_with_pre"] == 0
    assert by_characteristic.loc["pH", "pairs_with_post"] == 1
    assert by_characteristic.loc["pH", "pairs_with_both"] == 0

    # But the overall (any-characteristic) pair-level report DOES count
    # this pair as having both, since it has qualifying data in each
    # period for *some* characteristic -- a looser, documented distinction.
    pair_report = build_pair_report(pairs, counts)
    assert bool(pair_report.iloc[0]["has_both"]) is True


def test_characteristics_outside_the_configured_list_are_ignored():
    results = _results([
        ("S1", "Specific conductance", "2015-06-01"),  # not in the characteristic list
    ])
    filtered = prepare_results(results, ["Turbidity", "pH"])
    assert filtered.empty


def test_pair_with_no_observations_still_appears_with_false_flags():
    links = _links([
        ("D1", "Dam One", "S1", 2017, "upstream"),
    ])
    pairs, _ = prepare_links(links)
    empty_results = _results([])
    filtered = prepare_results(empty_results, ["Turbidity"])
    counts = build_observation_counts(pairs, filtered, years_before=3, years_after=3)
    pair_report = build_pair_report(pairs, counts)

    assert len(pair_report) == 1
    row = pair_report.iloc[0]
    assert bool(row["has_pre"]) is False
    assert bool(row["has_post"]) is False
    assert bool(row["has_both"]) is False
    assert row["n_pre_observations"] == 0
    assert row["n_post_observations"] == 0


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
