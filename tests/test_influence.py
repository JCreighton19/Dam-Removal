"""Checks for analysis/influence.py (leave-one-dam-out, dam bootstrap, BACI
components), on the simulated dams of tests/test_distance_models.py.

Run directly: `python tests/test_influence.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test_distance_models as sim
from analysis import distance_models as dm
from analysis import influence as inf

SETTINGS = sim.SETTINGS
OPTIONS = {"reference_km": 5.0, "size_variable": "proxy"}


def full_fit(daily, re_mode="intercepts"):
    return dm.fit_one(daily, "water_temp", "baci_distance", SETTINGS, re_mode=re_mode)


def test_leave_one_dam_out_finds_an_outlier_dam():
    daily = sim.simulate(n_dams=12)
    # D03's downstream stations jump by 6 after removal (three times the real effect).
    outlier = daily["AR_ID"].eq("D03") & daily["role"].eq("treated") & daily["post"].eq(1)
    daily.loc[outlier, "y"] += 6
    full = full_fit(daily)
    lodo = inf.leave_one_dam_out(full["rows"], "water_temp", "baci_distance", SETTINGS, OPTIONS,
                                 full)
    assert len(lodo) == 12 and lodo["error"].fillna("").eq("").all()
    by_dam = lodo.set_index("dam_left_out")
    assert by_dam.loc["D03", "influential"]
    assert by_dam.loc["D03", "delta_effect_at_reference_in_se"] <= -1   # removing it lowers the effect
    assert abs(by_dam.loc["D03", "modifier_slope"] - sim.SLOPE) < 0.2     # and restores the slope
    summary = inf.lodo_summary(lodo, {("water_temp", "baci_distance"): inf.summary_values(full)})
    assert summary.iloc[0]["most_influential_dam"] == "D03"
    assert "D03" in summary.iloc[0]["influential_dam_ids"]
    # Refits keep the full fit's reference ratio.
    assert by_dam["effect_5km"].notna().all()


def test_resampled_dams_are_relabeled():
    rows = full_fit(sim.simulate(n_dams=12))["rows"]
    draw = inf.resample_dams(rows, np.random.default_rng(0))
    assert draw["AR_ID"].nunique() == 12                       # repeats count as separate dams
    assert draw.groupby("AR_ID")["station_id"].nunique().eq(3).all()
    assert not set(draw["station_id"]) & set(rows["station_id"])


def test_bootstrap_summary_brackets_the_estimate():
    daily = sim.simulate(n_dams=12)
    full = full_fit(daily)
    draws, summary = inf.bootstrap(full["rows"], "water_temp", "baci_distance", SETTINGS, OPTIONS,
                                   full, draws=12, seed=1, workers=1)
    assert len(draws) == 12 and summary["draws_failed"] == 0
    assert summary["effect_at_reference_boot_low"] < full["row"]["effect_at_reference"] \
        < summary["effect_at_reference_boot_high"]
    assert summary["modifier_slope_boot_low"] < sim.SLOPE < summary["modifier_slope_boot_high"]


def test_components_split_upstream_and_downstream_change():
    daily = sim.simulate()
    full = full_fit(daily)
    by_dam, wide = inf.baci_components(full["rows"], "water_temp")
    assert set(by_dam["role"]) == {"control", "treated"} and len(wide) == 24
    # Simulated: +1 regional trend at every station; downstream adds ~2 at 5 km.
    assert abs(wide["change_upstream"].mean() - 1) < 0.35
    assert wide["downstream_minus_upstream"].mean() > 1
    summary = inf.components_summary(wide, full, "water_temp").set_index("quantity")
    model_up = summary.loc["model: upstream (control) change", "mean"]
    assert abs(model_up - 1) < 0.3
    assert abs(summary.loc["model: BACI effect at 5 km", "mean"] - 2) < 0.3
    traj = inf.trajectory(full["rows"], "water_temp")
    assert set(traj["years_from_removal"]) == {-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}
    post = traj[traj["years_from_removal"] > 0].groupby("role")["mean"].mean()
    assert post["treated"] > post["control"] > 0.5


def test_command_line_run_writes_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "analysis_measurement.parquet"
        sim.simulate(n_dams=12, raw=True).assign(parameter_std="ph").to_parquet(source,
                                                                                index=False)
        argv = ["influence.py", "--case-study", "test", "--input", str(source),
                "--output-dir", str(tmp / "out"), "--series", "ph", "--bootstrap", "10",
                "--workers", "2", "--exclude-dams", str(tmp / "none.csv")]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}):
            inf.main()
        for name in ("full_fits.csv", "lodo.csv", "lodo_summary.csv", "bootstrap_draws.csv",
                     "bootstrap_summary.csv", "components_ph.csv", "trajectory_ph.csv",
                     "components_summary.csv"):
            assert (tmp / "out" / name).exists(), name
        boot = pd.read_csv(tmp / "out" / "bootstrap_summary.csv")
        assert boot.iloc[0]["draws"] == 10


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
