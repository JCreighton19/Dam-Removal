"""Checks for analysis/sensitivity_compare.py: the pre-specified labels
(docs/eligibility_sensitivity_prespec.md), material-improvement flags and the
added-dam comparison.

Run directly: `python tests/test_sensitivity_compare.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import sensitivity_compare as sc


def test_labels_follow_the_prespecification():
    det, nmc, inc = "detected", "no meaningful change", "inconclusive"
    assert sc.label(det, 1, 0.5, 1.5, det, 1.2, 0.7, 1.7) == "robust"
    assert sc.label(inc, 0.3, -0.2, 0.8, det, 0.4, 0.1, 0.7) == "sharper"
    assert sc.label(inc, 0.0, -0.1, 0.1, nmc, 0.01, -0.05, 0.07) == "sharper"
    assert sc.label(det, 1, 0.5, 1.5, inc, 0.8, -0.1, 1.7) == "weaker"
    assert sc.label(det, 1, 0.5, 1.5, det, -1, -1.5, -0.5) == "conflicting"      # flips, CI excludes 0
    assert sc.label(det, 0.2, 0.1, 0.3, nmc, 0.05, -0.05, 0.08) == "conflicting"  # detected <-> no change
    assert sc.label(inc, 0.1, -0.5, 0.7, inc, -0.1, -0.7, 0.5) == "direction unstable"
    assert sc.label("", np.nan, np.nan, np.nan, det, 1, 0.5, 1.5) == "not comparable"
    # Clarification 2026-11-08: both "no meaningful change" -> robust even when the signs differ.
    assert sc.label(nmc, -0.07, -0.3, 0.2, nmc, 0.08, -0.2, 0.3) == "robust"
    # A sign flip with a different verdict is still not robust.
    assert sc.label(nmc, 0.01, -0.05, 0.07, inc, -0.01, -0.9, 0.9) == "direction unstable"


def test_compare_flags_material_improvements():
    primary = pd.DataFrame({"series": ["ph", "tds", "water_temp"], "spec": "baci_distance",
                            "dams": [20, 8, 30], "effect_at_reference": [0.1, 0.02, -0.1],
                            "effect_ci_low": [0.0, -0.1, -0.5], "effect_ci_high": [0.2, 0.14, 0.3],
                            "mde80_effect": [0.15, 0.2, 0.6]})
    sensitivity = pd.DataFrame({"series": ["ph", "tds", "water_temp"], "spec": "baci_distance",
                                "dams": [24, 16, 34], "effect_at_reference": [0.11, 0.03, -0.12],
                                "effect_ci_low": [0.03, -0.05, -0.45], "effect_ci_high": [0.19, 0.11, 0.21],
                                "mde80_effect": [0.08, 0.12, 0.55]})
    sesoi = {"ph": 0.1, "tds": np.log10(1.1), "water_temp": 0.5}
    out = sc.compare(primary, sensitivity, ["series", "spec"], "effect_at_reference", "effect_ci_low",
                     "effect_ci_high", sesoi=sesoi, mde="mde80_effect").set_index("series")
    assert out.loc["ph", "label"] == "sharper"                 # inconclusive -> detected
    assert abs(out.loc["ph", "ci_width_ratio"] - 0.8) < 1e-12 and out.loc["ph", "ci_narrower_20pct"]
    assert out.loc["ph", "mde_now_below_sesoi"]                # 0.15 >= 0.1 -> 0.08 < 0.1
    assert out.loc["tds", "now_fittable"] and out.loc["tds", "no_longer_exploratory"]
    # 8 primary dams < 10: shown but not compared (no verdict, no label, no CI ratio).
    assert out.loc["tds", "below_min_dams_primary"] and out.loc["tds", "verdict_primary"] == ""
    assert out.loc["tds", "label"] == "not comparable" and pd.isna(out.loc["tds", "ci_width_ratio"])
    assert out.loc["tds", "est_primary"] == 0.02                # the estimate itself is untouched
    assert out.loc["water_temp", "verdict_primary"] == "no meaningful change"   # CI inside +/-0.5
    assert out.loc["water_temp", "label"] == "robust" and not out.loc["water_temp", "material_improvement"]
    # A measure fit in only one table is kept, not comparable.
    extra = sc.compare(primary.iloc[:1], sensitivity, ["series", "spec"], "effect_at_reference",
                       "effect_ci_low", "effect_ci_high", sesoi=sesoi).set_index("series")
    assert extra.loc["tds", "label"] == "not comparable" and pd.isna(extra.loc["tds", "dams_primary"])


def test_models_below_ten_dams_are_never_labelled():
    # Like the mixed models (fit from 5 dams): BACI TDS 5 -> 7 dams with a much narrower CI.
    primary = pd.DataFrame({"series": ["tds", "turbidity", "ph"], "model": "baci", "dams": [5, 7, 22],
                            "estimate": [0.05, 0.35, 0.10], "ci_low": [-0.3, -0.1, -0.01],
                            "ci_high": [0.4, 0.8, 0.21]})
    sensitivity = pd.DataFrame({"series": ["tds", "turbidity", "ph"], "model": "baci", "dams": [7, 12, 33],
                                "estimate": [0.03, 0.24, 0.07], "ci_low": [-0.09, -0.05, -0.02],
                                "ci_high": [0.15, 0.53, 0.17]})
    out = sc.compare(primary, sensitivity, ["series", "model"], "estimate", "ci_low", "ci_high",
                     sesoi={"tds": 0.04, "turbidity": 0.1, "ph": 0.1}).set_index("series")
    assert out.loc["tds", "label"] == "not comparable"
    assert not out.loc["tds", "material_improvement"]           # 7 dams: not fittable, CI ignored
    assert out.loc["turbidity", "label"] == "not comparable"    # primary 7 dams
    assert out.loc["turbidity", "now_fittable"] and out.loc["turbidity", "material_improvement"]
    assert out.loc["ph", "label"] == "robust" and pd.notna(out.loc["ph", "ci_width_ratio"])
    labelled = out[out["label"] != "not comparable"]
    assert (labelled[["dams_primary", "dams_sensitivity"]] >= sc.MIN_DAMS).all().all()


def test_negligible_sign_flip_is_robust_and_flagged():
    primary = pd.DataFrame({"series": ["water_temp"], "dams": [24], "estimate": [-0.07],
                            "ci_low": [-0.3], "ci_high": [0.2]})
    sensitivity = primary.assign(dams=36, estimate=0.08, ci_low=-0.2, ci_high=0.3)
    out = sc.compare(primary, sensitivity, ["series"], "estimate", "ci_low", "ci_high",
                     sesoi={"water_temp": 0.5}).iloc[0]
    assert out["verdict_primary"] == out["verdict_sensitivity"] == "no meaningful change"
    assert out["label"] == "robust" and out["negligible_sign_flip"]


def test_added_dams_comparison():
    rows = pd.DataFrame({
        "AR_ID": ["A", "A", "B", "C", "D", "E", "F"],
        "station_id": ["s1", "s2", "s3", "s4", "s5", "s6", "s7"],
        "side": ["downstream", "upstream", "downstream", "downstream", "downstream", "downstream", "downstream"],
        "dam_height_ft": [10, 10, 10, 10, 100, 100, 100], "removal_year": [2000] * 7,
        "network_distance_km": [2.0, 9.0, 4.0, 6.0, 2.0, 4.0, 6.0],
        "dam_drainage_area_km2": [10.0] * 7})
    chars = sc.dam_characteristics(rows)
    assert chars.loc["A", "downstream distance (km)"] == 2.0       # upstream station ignored
    assert chars.loc["D", "log10 dam height (ft)"] == 2.0
    table = sc.compare_dams(chars, ["A", "B", "C"], ["D", "E", "F"]).set_index("characteristic")
    assert table.loc["log10 dam height (ft)", "systematic"]        # 1.0 vs 2.0, no spread
    assert table.loc["log10 dam height (ft)", "smd_added_minus_primary"] == np.inf
    assert not table.loc["downstream distance (km)", "systematic"]  # same distribution
    assert abs(sc.smd([1, 2, 3], [2, 3, 4]) - 1.0) < 1e-12
    # Means are reported next to the medians (the SMD is computed from the means).
    assert table.loc["log10 dam height (ft)", "primary_mean"] == 1.0
    assert table.loc["log10 dam height (ft)", "added_mean"] == 2.0
    assert {"primary_median", "added_median"} <= set(table.columns)


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
