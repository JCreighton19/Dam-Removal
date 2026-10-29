"""Checks for analysis/mixed_model.py.

Roles, the daily collapse and log scale on small hand-made rows, and that both
models recover a known effect on simulated dams (a regional trend of about +1
at both of a dam's stations after removal, about +2 more downstream).

Run directly: `python tests/test_mixed_model.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import mixed_model as mm

SETTINGS = {**mm.SETTINGS, "series": ["water_temp", "turbidity"]}


def row(**kw):
    base = {"AR_ID": "D1", "station_id": "S1", "parameter_std": "water_temp",
            "sample_fraction_std": None, "side": "downstream", "period": "pre",
            "sample_date": "2010-06-01", "month": 6, "result_value_std": 10.0,
            "exposure_status": "assigned_sole", "upstream_in_former_reservoir": None,
            "air_temp_mean_c": 20.0, "precip_prev_month_mm": 50.0}
    return {**base, **kw}


def simulate(n_dams=20, seed=0):
    """One treated and one control station per dam, monthly for 5 years each side.
    attrs["effect"] and attrs["trend"]: means of the dams' simulated effects and
    regional trends (the targets)."""
    rng = np.random.default_rng(seed)
    rows, effects, trends = [], [], []
    for d in range(n_dams):
        dam_level = rng.normal(0, 3)
        dam_trend = 1 + rng.normal(0, 0.5)
        dam_effect = 2 + rng.normal(0, 0.3)
        effects.append(dam_effect)
        trends.append(dam_trend)
        for side, status in [("downstream", "assigned_sole"), ("upstream", "upstream_control")]:
            station_level = rng.normal(0, 1)
            for year in range(-5, 6):
                if year == 0:
                    continue
                post = year > 0
                for month in range(1, 13):
                    air = 10 + 10 * np.sin(month / 12 * 2 * np.pi) + rng.normal(0, 2)
                    value = (dam_level + station_level + 0.5 * air + dam_trend * post
                             + dam_effect * (post and side == "downstream") + rng.normal(0, 1))
                    rows.append(row(AR_ID=f"D{d}", station_id=f"D{d}-{side}", side=side,
                                    exposure_status=status, period="post" if post else "pre",
                                    sample_date=f"{2010 + year}-{month:02d}-15", month=month,
                                    result_value_std=value, air_temp_mean_c=air))
    df = pd.DataFrame(rows)
    df.attrs["effect"] = float(np.mean(effects))
    df.attrs["trend"] = float(np.mean(trends))
    return df


def test_roles():
    df = pd.DataFrame([
        row(),                                                    # treated
        row(exposure_status="competing_timing"),                  # excluded
        row(side="upstream", exposure_status="upstream_control"),  # control
        row(side="upstream", exposure_status="upstream_control",
            upstream_in_former_reservoir=True),                   # excluded: in reservoir
        row(side="upstream", exposure_status="upstream_control_exposed"),  # excluded
    ])
    assert mm.assign_roles(df, SETTINGS).tolist() == [
        "treated", "excluded", "control", "excluded", "excluded"]
    kept = mm.assign_roles(df, {**SETTINGS, "exclude_former_reservoir": False})
    assert kept.tolist()[3] == "control"


def test_daily_collapse_and_log_scale():
    df = pd.DataFrame([
        row(result_value_std=10.0), row(result_value_std=11.0), row(result_value_std=30.0),
        row(parameter_std="turbidity", result_value_std=100.0),
        row(parameter_std="turbidity", sample_date="2010-06-02", result_value_std=0.0),
        row(parameter_std="nitrate", sample_fraction_std="Total"),   # not a modeled series
    ])
    daily = mm.prepare(df, SETTINGS).set_index(["series", "sample_date"])["y"]
    assert len(daily) == 3
    assert daily[("water_temp", "2010-06-01")] == 11.0               # median of repeats
    assert daily[("turbidity", "2010-06-01")] == 2.0                 # log10(100)
    assert daily[("turbidity", "2010-06-02")] == np.log10(1e-3)      # clipped at the floor


def test_recovers_known_effects():
    df = simulate()
    effect = df.attrs["effect"]
    daily = mm.prepare(df, SETTINGS)
    results, effects, residuals = mm.fit_all(daily, SETTINGS)
    r = results.set_index(["series", "model"])
    before_after = r.loc[("water_temp", "before_after")]
    baci = r.loc[("water_temp", "baci")]
    # Downstream changes by trend + effect; beyond upstream by the effect alone.
    assert before_after["ci_low"] < df.attrs["trend"] + effect < before_after["ci_high"]
    assert baci["ci_low"] < effect < baci["ci_high"]
    assert baci["fit_check"] == "ok"
    assert before_after["dams"] == baci["dams"] == 20
    assert before_after["re_structure"] == "dam slopes + station intercepts"
    # No turbidity rows: reported as skipped, not dropped.
    assert r.loc[("turbidity", "baci"), "error"].startswith("fewer than")
    # One effect per dam, centered near the average.
    dam = effects[(effects["series"] == "water_temp") & (effects["model"] == "baci")]
    assert len(dam) == 20 and abs(dam["dam_effect"].mean() - baci["estimate"]) < 0.1
    assert len(residuals) == before_after["days"] + baci["days"]


def test_rows_need_both_periods():
    df = simulate(n_dams=6)
    df = df[~((df["AR_ID"] == "D0") & (df["period"] == "post"))]   # D0 has no post data
    daily = mm.prepare(df, SETTINGS)
    rows = mm.model_rows(daily[daily["series"] == "water_temp"], "before_after", SETTINGS)
    assert "D0" not in set(rows["AR_ID"]) and rows["AR_ID"].nunique() == 5


def test_missing_columns_named():
    path = Path(__file__).resolve().parent / "_mixed_model_missing.csv"
    pd.DataFrame([row()]).drop(columns=["exposure_status"]).to_csv(path, index=False)
    try:
        mm.read_measurements(path, SETTINGS)
        raise AssertionError("expected a ValueError")
    except ValueError as exc:
        assert "exposure_status" in str(exc)
    finally:
        path.unlink()


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
