"""Checks for analysis/audits/station_payoff_audit.py: link sides and limits,
the gain each ambiguous station could give a dam, and the per-station and
per-dam tables, on small synthetic frames (no warehouse needed).

Run directly: `python tests/test_station_payoff.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from analysis.audits import station_payoff_audit as spa


def _links(rows):
    columns = ["dam_id", "station_id", "relationship", "network_distance_km", "dam_fraction",
               "station_fraction", "dam_status", "station_status", "station_reason", "usable",
               "station_in_mapped_reservoir"]
    return spa.prepare_links(pd.DataFrame(rows, columns=columns), max_down=30, max_up=10)


def _dams():
    # B_ONLY: before/after dam without a control; BACI: has both; NODOWN: stations
    # but none downstream; ONESIDE: not in the table.
    return pd.DataFrame({
        "reason": ["in_analysis_table", "in_analysis_table", "one_side_only", "one_side_only"],
        "has_assigned_downstream": [True, True, None, None],
        "has_upstream_control": [False, True, None, None],
        "usable_downstream_in_limits": [2, 3, 0, 1],
    }, index=pd.Index(["B_ONLY", "BACI", "NODOWN", "ONESIDE"], name="dam_id"))


def test_side_and_limits():
    links = _links([
        ("D", "s_down", "downstream", 29.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("D", "s_far", "downstream", 31.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("D", "s_up", "upstream", 9.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("D", "s_upfar", "upstream", 11.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("D", "s_same_dn", "same_reach", 0.1, 0.3, 0.6, "accepted", "ambiguous", "far", False, False),
        ("D", "s_same_up", "same_reach", 0.1, 0.6, 0.3, "accepted", "ambiguous", "far", False, False),
        ("D", "s_res", "upstream", 1.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, True),
    ]).set_index("station_id")
    assert links.loc["s_same_dn", "side"] == "downstream"
    assert links.loc["s_same_up", "side"] == "upstream"
    assert links["in_limits"].to_dict() == {"s_down": True, "s_far": False, "s_up": True,
                                            "s_upfar": False, "s_same_dn": True,
                                            "s_same_up": True, "s_res": True}
    assert links.loc["s_up", "is_control"] and not links.loc["s_res", "is_control"]


def test_gains():
    links = _links([
        ("B_ONLY", "up1", "upstream", 2.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("B_ONLY", "res1", "upstream", 1.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, True),
        ("B_ONLY", "dn1", "downstream", 2.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("BACI", "up1", "upstream", 3.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("NODOWN", "dn2", "downstream", 4.0, 0.5, 0.5, "accepted", "ambiguous", "confluence_ambiguous",
         False, False),
        ("ONESIDE", "up2", "upstream", 4.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
    ])
    gains = spa.link_gains(links, _dams()).set_index(["dam_id", "station_id"])["gain"]
    assert gains[("B_ONLY", "up1")] == "baci_control"
    assert gains[("B_ONLY", "res1")] is None            # in the reservoir: not a control
    assert gains[("B_ONLY", "dn1")] == "extra_station"
    assert gains[("BACI", "up1")] == "extra_station"
    assert gains[("NODOWN", "dn2")] == "first_downstream"
    assert gains[("ONESIDE", "up2")] == "dam_not_in_table"
    assert spa.best_gain(["extra_station", None, "first_downstream"]) == "first_downstream"
    assert spa.best_gain([None]) is None


def test_tables():
    links = _links([
        ("B_ONLY", "up1", "upstream", 2.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("BACI", "up1", "upstream", 3.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
        ("NODOWN", "dn2", "downstream", 4.0, 0.5, 0.5, "accepted", "ambiguous", "far", False, False),
    ])
    gained = spa.link_gains(links, _dams())
    gained = gained[gained["gain"].notna()]
    stations = pd.DataFrame({"MonitoringLocationIdentifier": ["up1", "dn2"],
                             "MonitoringLocationName": ["Upper", "Lower"],
                             "snap_reason": ["far_without_support", "confluence_ambiguous"]})
    out = spa.station_table(gained, stations, downloaded={"dn2"}).set_index("station_id")
    assert out.loc["up1", "best_gain"] == "baci_control"
    assert out.loc["up1", "dams_helped"] == 2 and out.loc["up1", "dams_baci_control"] == 1
    assert out.loc["up1", "review_rank"] == 1 and out.loc["dn2", "review_rank"] == 2
    assert out.loc["dn2", "already_downloaded"] and not out.loc["up1", "already_downloaded"]
    assert out.loc["up1", "MonitoringLocationName"] == "Upper"
    dams = spa.dam_table(gained).set_index("dam_id")
    assert dams.loc["B_ONLY", "best_gain"] == "baci_control"
    assert dams.loc["NODOWN", "stations_first_downstream"] == 1
    summary = spa.summary_table(dams.reset_index(), out.reset_index()).set_index("gain")
    assert summary.loc["baci_control", "dams_best_gain"] == 1
    assert summary.loc["extra_station", "dams_any"] == 1


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
