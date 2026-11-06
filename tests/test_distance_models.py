"""Checks for analysis/distance_models.py (Stage 3 distance / size models).

Simulated dams with one upstream control and two downstream stations: a
regional trend of +1 after removal at every station, plus a downstream effect
of 2 - 0.8 * (ln(1 + km) - ln(6)) (2 at 5 km, fading with distance). The
models should recover both, and the existing mixed-model specs must be
unchanged.

Run directly: `python tests/test_distance_models.py`
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import distance_models as dm
from analysis import mixed_model as mm

SETTINGS = {**mm.SETTINGS, "series": ["water_temp"], "max_downstream_km": 30,
            "carry_columns": dm.CARRY, "min_dams": 10, "exploratory_below": 15}
SLOPE = -0.8


def simulate(n_dams=24, seed=1, missing_size_every=0, raw=False, station_sd=0.5,
             effect_sd=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dams):
        dam_level = rng.normal(0, 2)
        dam_shift = rng.normal(0, effect_sd) if effect_sd else 0.0   # dam-specific effect
        proxy = None if missing_size_every and d % missing_size_every == 0 else 10 ** rng.uniform(2, 5)
        stations = [("upstream", "upstream_control", 3.0, 0.8)]
        stations += [("downstream", "assigned_sole", float(km), float(10 ** rng.uniform(0, 2)))
                     for km in rng.uniform(0.3, 25, size=2)]
        for i, (side, status, km, dar) in enumerate(stations):
            station_level = rng.normal(0, station_sd)
            effect = (2 + dam_shift + SLOPE * (np.log1p(km) - np.log1p(5))) \
                if side == "downstream" else 0
            for year in [y for y in range(-5, 6) if y != 0]:
                post = year > 0
                for month in (2, 5, 8, 11):
                    air = 10 + 8 * np.sin(month / 12 * 2 * np.pi) + rng.normal(0, 1)
                    value = (dam_level + station_level + 0.3 * air + 1.0 * post
                             + effect * post + rng.normal(0, 0.5))
                    rows.append({
                        "AR_ID": f"D{d:02d}", "station_id": f"D{d:02d}-S{i}",
                        "parameter_std": "water_temp", "sample_fraction_std": None,
                        "side": side, "period": "post" if post else "pre",
                        "sample_date": f"{2010 + year}-{month:02d}-15", "month": month,
                        "result_value_std": value, "exposure_status": status,
                        "upstream_in_former_reservoir": None, "qc_non_perennial": False,
                        "has_same_day_discharge": False, "air_temp_mean_c": air,
                        "precip_prev_month_mm": 50.0 + rng.normal(0, 5),
                        "network_distance_km": km, "drainage_area_ratio": dar,
                        "backwater_km": 0.5 + d % 4, "reservoir_volume_proxy": proxy,
                        "reservoir_polygon_area_km2": None, "reservoir_polygon_timing": None,
                        "dam_drainage_area_km2": 10 ** rng.uniform(0.5, 2.5) if i == 0 else None,
                        "removal_year": 2010, "years_from_removal": year,
                    })
    df = pd.DataFrame(rows)
    # Dam drainage area is a dam constant.
    df["dam_drainage_area_km2"] = df.groupby("AR_ID")["dam_drainage_area_km2"].transform("first")
    return df if raw else mm.prepare(df, SETTINGS)


def test_existing_specs_unchanged():
    assert set(mm.MODELS) == {"before_after", "baci"}
    assert mm.model_formula("baci", ["a"]) == "y ~ post + downstream + post_downstream + C(month) + z_a"
    assert mm.model_formula("before_after", []) == "y ~ post + C(month)"
    assert mm.SETTINGS["carry_columns"] == [] and mm.SETTINGS["max_downstream_km"] == 10


def test_discharge_check_is_added_without_touching_the_existing_models():
    assert "discharge" in dm.default_series() and "discharge" not in mm.SETTINGS["series"]
    assert dm.default_series()[:len(mm.SETTINGS["series"])] == mm.SETTINGS["series"]
    # Discharge is fit like any series (log10, same roles and rows).
    raw = simulate(raw=True)
    raw["parameter_std"] = "discharge"
    raw["result_value_std"] = 10 ** (1 + raw["result_value_std"] / 20)
    settings = {**SETTINGS, "series": ["discharge"]}
    daily = mm.prepare(raw, settings)
    assert daily["log_scale"].all()
    results, _, _ = dm.fit_all(daily, ["baci_distance"], settings)
    assert results.iloc[0]["error"] == "" and results.iloc[0]["unit"] == "log10 ft3/s"


def test_distance_is_downstream_only_and_zero_km_is_safe():
    daily = pd.DataFrame({
        "AR_ID": "D1", "station_id": ["U", "A", "B"], "role": ["control", "treated", "treated"],
        "network_distance_km": [4.0, 0.0, 5.0], "drainage_area_ratio": [0.5, 1.0, 10.0],
        "backwater_km": 2.0})
    rows, refs, dropped = dm.add_modifiers(daily, "baci_distance")
    ld = rows.set_index("station_id")["ld"]
    assert ld["U"] == 0                                     # upstream distance never used
    assert abs(ld["A"] - (np.log1p(0) - np.log1p(5))) < 1e-12   # 0 km is finite
    assert abs(ld["B"]) < 1e-12 and refs["reference_km"] == 5
    assert dropped == 0 and rows.set_index("station_id").loc["U", "ldar"] == 0
    # Reference ratio: the median of the downstream pairs (1 and 10).
    assert abs(refs["reference_dar"] - 5.5) < 1e-12


def test_terms_put_distance_on_the_effect():
    daily = pd.DataFrame({"post": [0, 1], "downstream": [1, 1], "post_downstream": [0, 1],
                          "ld": [0.5, 0.5], "ldar": [0.1, 0.1]})
    terms, effect = dm.term_columns(daily.copy(), "baci_distance")
    assert terms == ["post", "downstream", "post_downstream", "down_ld", "pd_ld",
                     "down_ldar", "pd_ldar"]
    assert effect == {"base": "post_downstream", "ld": "pd_ld", "ldar": "pd_ldar"}
    terms, effect = dm.term_columns(daily.copy(), "before_after_distance")
    assert terms == ["post", "ld", "post_ld", "ldar", "post_ldar"] and effect["ld"] == "post_ld"


def test_recovers_distance_decay():
    daily = simulate()
    results, coefficients, at = dm.fit_all(daily, ["baci_distance", "before_after_distance"],
                                           SETTINGS)
    results = results.set_index("spec")
    baci = results.loc["baci_distance"]
    assert baci["error"] == "" and baci["dams"] == 24 and baci["sample_flag"] == ""
    assert baci["modifier_ci_low"] < SLOPE < baci["modifier_ci_high"]
    assert baci["modifier_ci_high"] < 0                      # the decay is detected
    # Before/after includes the +1 regional trend; its slope is the same.
    ba = results.loc["before_after_distance"]
    assert ba["modifier_ci_low"] < SLOPE < ba["modifier_ci_high"]
    at = at[at["spec"] == "baci_distance"].set_index("distance_km")
    # Only distances inside the observed range (0.3-25 km) are reported.
    assert 30 not in at.index and {1, 5, 10, 20} <= set(at.index)
    assert abs(at.loc[5, "estimate"] - baci["effect_at_reference"]) < 1e-9
    assert at.loc[1, "estimate"] > at.loc[20, "estimate"]


def test_restrictions_touch_only_treated_rows_and_listed_dams():
    daily = simulate(n_dams=12)
    rows, counts = dm.restrict(daily, exclude_dams=["D00"], max_dar=10)
    assert "D00" not in set(rows["AR_ID"]) and counts["rows_dropped_excluded_dams"] > 0
    assert rows.loc[rows["role"].eq("treated"), "drainage_area_ratio"].le(10).all()
    controls = daily[daily["role"].eq("control") & daily["AR_ID"].ne("D00")]
    assert len(rows[rows["role"].eq("control")]) == len(controls)   # controls untouched
    assert daily["role"].equals(simulate(n_dams=12)["role"])          # assignment unchanged


def test_missing_size_is_dropped_and_counted_not_imputed():
    daily = simulate(missing_size_every=3)
    results, _, at = dm.fit_all(daily, ["baci_size"], SETTINGS)
    row = results.iloc[0]
    assert row["dams"] == 16 and row["dams_before_modifier"] == 24
    assert row["rows_dropped_missing_modifier"] > 0
    assert row["modifier"] == "size" and np.isfinite(row["modifier_slope"])
    assert set(at["size_percentile"]) == {10, 50, 90}
    assert np.isfinite(row["size_vs_drainage_r"]) and row["size_vs_drainage_vif"] >= 1
    assert row["sample_flag"] == ""                            # 16 dams: at or above 15


def test_model_frame_is_the_fitted_rows():
    daily = simulate(missing_size_every=3)
    for spec in ["baci_distance", "before_after_scaled_distance", "baci_size"]:
        rows, refs, meta = dm.model_frame(daily, "water_temp", spec, SETTINGS)
        fitted = dm.fit_one(daily, "water_temp", spec, SETTINGS)
        # fit_one then adds its interaction columns to the same rows.
        pd.testing.assert_frame_equal(rows, fitted["rows"][rows.columns])
        assert refs == fitted["refs"]
        assert meta["dams_before_modifier"] == fitted["row"]["dams_before_modifier"]


def test_small_samples_are_flagged_or_skipped():
    results, _, _ = dm.fit_all(simulate(n_dams=12), ["baci_distance"], SETTINGS)
    assert results.iloc[0]["sample_flag"].startswith("exploratory")
    results, _, _ = dm.fit_all(simulate(n_dams=8), ["baci_distance"], SETTINGS)
    assert results.iloc[0]["error"].startswith("fewer than 10")


def test_scaled_distance_reference_is_one_reservoir_length():
    daily = pd.DataFrame({"AR_ID": ["D1", "D1", "D2"], "station_id": ["A", "B", "C"],
                          "role": "treated", "network_distance_km": [2.0, 6.0, 1.0],
                          "drainage_area_ratio": 2.0, "backwater_km": [2.0, 2.0, None]})
    rows, _, dropped = dm.add_modifiers(daily, "baci_scaled_distance")
    ls = rows.set_index("station_id")["ls"]
    assert abs(ls["A"]) < 1e-12 and ls["B"] > 0 and "C" not in ls.index and dropped == 1


def test_dilution_and_scaled_specs_fit():
    results, _, at = dm.fit_all(simulate(), ["baci_distance_dilution", "baci_scaled_distance"],
                                SETTINGS)
    assert (results["error"] == "").all()
    dilution = at[at["spec"] == "baci_distance_dilution"]
    assert dilution["drainage_area_ratio"].dropna().isin(dm.DILUTION_RATIOS).all()
    assert at.loc[at["spec"] == "baci_scaled_distance", "reservoir_lengths"].notna().any()


def test_command_line_run_writes_outputs():
    import os
    import tempfile
    from unittest import mock
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "analysis_measurement.parquet"
        simulate(raw=True).to_parquet(source, index=False)
        pd.DataFrame({"AR_ID": ["D00"]}).to_csv(tmp / "exclude.csv", index=False)
        argv = ["distance_models.py", "--case-study", "test", "--input", str(source),
                "--output-dir", str(tmp / "out"), "--series", "water_temp",
                "--exclude-dams", str(tmp / "exclude.csv")]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}):
            dm.main()
        results = pd.read_csv(tmp / "out" / "results.csv")
        assert set(results["spec"]) == set(dm.DEFAULT_SPECS)
        assert (results["dams"] == 23).all()                     # D00 excluded
        used = pd.read_csv(tmp / "out" / "dams_used.csv")
        assert set(used["spec"]) == set(dm.DEFAULT_SPECS) and "D00" not in set(used["AR_ID"])
        baci = used[used["spec"] == "baci_distance"]
        assert len(baci) == 23 and (baci["control_stations"] == 1).all() \
            and (baci["treated_stations"] == 2).all()
        assert (used.loc[used["design"] == "before_after", "control_stations"] == 0).all()
        for name in ("coefficients.csv", "effects_at.csv", "settings.json"):
            assert (tmp / "out" / name).exists()


def test_singular_flag_reads_the_final_variances():
    results, _, _ = dm.fit_all(simulate(), ["baci_distance"], SETTINGS)
    fitted = results.iloc[0]
    assert not fitted["singular"] and fitted["var_station"] > 0.01 and fitted["var_dam_intercept"] > 1
    # No station-to-station differences: the station variance is estimated near 0.
    # How near depends on where the optimizer stops (it landed exactly on 0
    # locally but just above it on the cluster's library builds), so the fit is
    # checked loosely and the flag rule exactly, on a fit result with known values.
    results, _, _ = dm.fit_all(simulate(station_sd=0.0), ["baci_distance"], SETTINGS)
    flat = results.iloc[0]
    assert flat["var_station"] < 0.1 * fitted["var_station"]
    assert flat["singular"] == (flat["var_station"] < 1e-3 * flat["var_residual"]
                                or flat["var_dam_intercept"] < 1e-3 * flat["var_residual"])

    class Fit:   # what variance_components reads from a statsmodels MixedLM result
        scale = 0.25
        cov_re = pd.DataFrame([[2.0]], index=["Group"], columns=["Group"])
        vcomp = np.array([1e-5])
    flags = dm.variance_components(Fit())
    assert flags["singular"] and flags["near_zero_terms"] == "station"
    assert flags["var_dam_intercept"] == 2.0 and flags["var_residual"] == 0.25
    Fit.vcomp = np.array([0.05])
    assert not dm.variance_components(Fit())["singular"]


def test_auto_random_effects_follow_the_data():
    # Same effect at every dam: the dam slope of the effect is ~0, auto uses intercepts.
    same, _, _ = dm.fit_all(simulate(), ["baci_distance"], SETTINGS, re_mode="auto")
    assert same.iloc[0]["re_choice"].startswith("intercepts")
    assert same.iloc[0]["lrt_p"] >= 0.05 or "~0" in same.iloc[0]["re_choice"]
    assert same.iloc[0]["re_structure"].startswith("dam intercepts")
    # Dams differ in their effect: slopes are kept, and the interval widens.
    varied = simulate(effect_sd=1.0)
    slopes, _, _ = dm.fit_all(varied, ["baci_distance"], SETTINGS, re_mode="auto")
    flat, _, _ = dm.fit_all(varied, ["baci_distance"], SETTINGS, re_mode="intercepts")
    assert slopes.iloc[0]["re_choice"].startswith("slopes (LRT p")
    assert slopes.iloc[0]["var_dam_post_downstream"] > 0.3
    assert slopes.iloc[0]["effect_se"] > 2 * flat.iloc[0]["effect_se"]


def test_boundary_lrt_uses_bootstrap_when_chi_square_is_not_decisive():
    # Small real dam-to-dam variation: chi-square p (an upper bound) is 0.12.
    varied, _, _ = dm.fit_all(simulate(n_dams=16, seed=1, effect_sd=0.1), ["baci_distance"],
                              SETTINGS, re_mode="auto", lrt_bootstrap=39)
    row = varied.iloc[0]
    assert row["lrt_p_chi2"] > 0.05 and row["lrt_p_method"].startswith("parametric bootstrap")
    assert row["lrt_p"] < row["lrt_p_chi2"]          # the boundary-correct p is smaller
    assert row["lrt_bootstrap_draws"] >= 35
    # No dam-to-dam variation: the bootstrap p stays large and intercepts are kept.
    null, _, _ = dm.fit_all(simulate(n_dams=16, seed=1), ["baci_distance"], SETTINGS,
                            re_mode="auto", lrt_bootstrap=39)
    row = null.iloc[0]
    assert row["lrt_p"] >= 0.2 and row["re_choice"].startswith("intercepts")
    # Decisive chi-square: no simulation needed.
    clear, _, _ = dm.fit_all(simulate(effect_sd=1.0), ["baci_distance"], SETTINGS,
                             re_mode="auto", lrt_bootstrap=39)
    assert clear.iloc[0]["lrt_p_method"].startswith("chi-square (upper bound")


def test_minimum_detectable_effect():
    assert abs(dm.mde(1.0, 30) - (dm.stats.t.ppf(0.975, 29) + dm.stats.t.ppf(0.8, 29))) < 1e-12
    results, _, at = dm.fit_all(simulate(), ["baci_distance"], SETTINGS)
    row = results.iloc[0]
    assert abs(row["mde80_effect"] - dm.mde(row["effect_se"], row["dams"])) < 1e-12
    assert row["mde80_modifier"] > 0 and (at["mde80"] > 0).all()
    assert np.isfinite(row["cov_effect_modifier"])


def test_common_sample_and_ml_make_specs_comparable():
    daily = simulate()
    # Some dams lack backwater_km: scaled distance alone would fit fewer dams.
    daily.loc[daily["AR_ID"].isin(["D01", "D02", "D03"]), "backwater_km"] = np.nan
    plain, _, _ = dm.fit_all(daily, ["baci_distance", "baci_scaled_distance"], SETTINGS)
    assert plain.set_index("spec")["dams"].tolist() == [24, 21]
    common, _, _ = dm.fit_all(daily, ["baci_distance", "baci_scaled_distance"], SETTINGS,
                              reml=False, common_sample=True)
    common = common.set_index("spec")
    assert common["dams"].tolist() == [21, 21] and common["days"].nunique() == 1
    assert common["aic"].notna().all() and not common["reml"].any()
    assert "aic" not in plain or plain["aic"].isna().all()


def test_fixed_references_are_reused():
    daily = simulate()
    rows = mm.model_rows(daily, "baci", SETTINGS)
    _, refs, _ = dm.add_modifiers(rows, "baci_distance")
    subset = rows[rows["AR_ID"] != "D00"]
    _, own, _ = dm.add_modifiers(subset, "baci_distance")
    _, fixed, _ = dm.add_modifiers(subset, "baci_distance", fixed_refs=refs)
    assert fixed["reference_dar"] == refs["reference_dar"] != own["reference_dar"]


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
