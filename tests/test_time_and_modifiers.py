"""Checks for the time-since-removal (design 11) and dam / watershed
characteristic (design 12) extensions of analysis/distance_models.py.

Simulated dams with one upstream control and two downstream stations: a
regional +1 after removal everywhere, plus a downstream effect that can vary
with the year since removal, with dam height (dam level) and with the
downstream station's forest share (station level). The models should recover
each, leave the existing specs untouched, and say so when nothing varies.

Run directly: `python tests/test_time_and_modifiers.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import distance_models as dm
from analysis import mixed_model as mm

ALL_X = [s for s in dm.SPECS if "_xu_" in s]
SETTINGS = {**mm.SETTINGS, "series": ["water_temp"], "max_downstream_km": 30,
            "carry_columns": dm.carry_columns(ALL_X, all_characteristics=True),
            "min_dams": 10, "exploratory_below": 15, "time_bins": dm.DEFAULT_TIME_BINS}
YEARS = [y for y in range(-5, 6) if y != 0]


def simulate(n_dams=30, seed=3, profile=None, height_slope=0.0, forest_slope=0.0,
             early_pre_drift=0.0, raw=False, missing_height_every=0):
    """profile: {year: downstream effect} after removal (default 2 every year)."""
    rng = np.random.default_rng(seed)
    profile = profile or {y: 2.0 for y in YEARS if y > 0}
    log_height = rng.uniform(0.3, 1.5, n_dams)
    z_height = (log_height - log_height.mean()) / log_height.std(ddof=1)
    forest = rng.uniform(5, 95, (n_dams, 2))
    z_forest = (forest - forest.mean()) / forest.std(ddof=1)
    rows = []
    for d in range(n_dams):
        dam_level = rng.normal(0, 2)
        height = None if missing_height_every and d % missing_height_every == 0 \
            else float(10 ** log_height[d])
        stations = [("upstream", "upstream_control", 3.0, 0.8, None)]
        stations += [("downstream", "assigned_sole", float(km), float(10 ** rng.uniform(0, 2)), k)
                     for k, km in enumerate(rng.uniform(0.3, 25, size=2))]
        for i, (side, status, km, dar, k) in enumerate(stations):
            station_level = rng.normal(0, 0.5)
            pct_forest = float(forest[d, k]) if k is not None else float(rng.uniform(5, 95))
            for year in YEARS:
                post = year > 0
                effect = 0.0
                if side == "downstream" and post:
                    effect = (profile[year] + height_slope * z_height[d]
                              + forest_slope * z_forest[d, k])
                if side == "downstream" and year <= -3:
                    effect += early_pre_drift
                for month in (2, 5, 8, 11):
                    air = 10 + 8 * np.sin(month / 12 * 2 * np.pi) + rng.normal(0, 1)
                    rows.append({
                        "AR_ID": f"D{d:02d}", "station_id": f"D{d:02d}-S{i}",
                        "parameter_std": "water_temp", "sample_fraction_std": None,
                        "side": side, "period": "post" if post else "pre",
                        "sample_date": f"{2010 + year}-{month:02d}-15", "month": month,
                        "result_value_std": (dam_level + station_level + 0.3 * air + 1.0 * post
                                             + effect + rng.normal(0, 0.5)),
                        "exposure_status": status, "upstream_in_former_reservoir": None,
                        "qc_non_perennial": False, "has_same_day_discharge": False,
                        "air_temp_mean_c": air, "precip_prev_month_mm": 50.0 + rng.normal(0, 5),
                        "network_distance_km": km, "drainage_area_ratio": dar,
                        "backwater_km": 1.0 + d % 3, "reservoir_volume_proxy": 10 ** (2 + d % 4),
                        "reservoir_polygon_area_km2": 0.01 * (1 + d % 5),
                        "reservoir_polygon_timing": "edited_by_removal_year" if d % 2
                        else "edited_after_removal",
                        "dam_drainage_area_km2": 10 ** (0.5 + (d % 7) / 3),
                        "dam_height_ft": height, "pct_forest": pct_forest,
                        "pct_developed": 100 - pct_forest, "pct_agriculture": 0.0,
                        "removal_year": 2010, "years_from_removal": year,
                    })
    df = pd.DataFrame(rows)
    return df if raw else mm.prepare(df, SETTINGS)


def fit(daily, specs, settings=SETTINGS, **kw):
    return dm.fit_all(daily, specs, settings, **kw)


# --------------------------------------------------
# Existing specs are untouched
# --------------------------------------------------

def test_existing_specs_and_benchmark_are_unchanged():
    assert dm.SPECS["baci_distance"] == {"design": "baci", "station": ["ld", "ldar"], "dam": []}
    assert dm.SPECS["baci_size"] == {"design": "baci", "station": [], "dam": ["zsize", "zlda"]}
    assert dm.DEFAULT_SPECS == ["baci_distance", "before_after_distance",
                                "baci_scaled_distance", "before_after_scaled_distance"]
    # The benchmark is the mixed_model.py formula: same terms, same effect.
    rows = pd.DataFrame({"post": [0, 1], "downstream": [1, 1], "post_downstream": [0, 1]})
    terms, effect = dm.term_columns(rows.copy(), "baci_binary")
    assert terms == ["post", "downstream", "post_downstream"]
    assert dm.formula_for(terms, ["a"]) == mm.model_formula("baci", ["a"])
    assert effect == {"base": "post_downstream"}
    terms, _ = dm.term_columns(rows.copy(), "before_after_binary")
    assert dm.formula_for(terms, []) == mm.model_formula("before_after", [])


# --------------------------------------------------
# Design 11: time since removal
# --------------------------------------------------

def test_time_bins_are_validated():
    assert dm.parse_time_bins("pre:-5:-1;y1:1:1;y2_5:2:5") == [("pre", -5, -1), ("y1", 1, 1),
                                                               ("y2_5", 2, 5)]
    assert dm.parse_time_bins("pretrend")[0] == ("pre_late", -2, -1)
    for bad in ("pre:-5:1;y1:2:2",                 # spans the removal year
                "pre:-5:-1;y1:1:2;y2:2:3",         # overlap
                "y1:1:1;pre:-5:-1",                # reference must be pre
                "pre:-5:-3;late:-2:-1",            # no post bin
                "pre:-5:-1;y-1:1:1"):              # label isn't a word
        try:
            dm.parse_time_bins(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_time_terms_replace_post_with_bins():
    daily = simulate(n_dams=3)
    rows, refs, dropped = dm.add_modifiers(daily, "baci_time", time_bins=dm.DEFAULT_TIME_BINS)
    assert dropped == 0 and refs["time_reference"] == "pre"
    assert [c for c in rows if c.startswith("tb")] == ["tbpost_y1", "tbpost_y2_3", "tbpost_y4_5"]
    in_y23 = rows["years_from_removal"].isin([2, 3])
    assert rows.loc[in_y23, "tbpost_y2_3"].eq(1).all() and rows.loc[~in_y23, "tbpost_y2_3"].eq(0).all()
    terms, effect = dm.term_columns(rows, "baci_time")
    assert terms == ["tbpost_y1", "tbpost_y2_3", "tbpost_y4_5", "downstream",
                     "pd_tbpost_y1", "pd_tbpost_y2_3", "pd_tbpost_y4_5"]
    assert "post" not in terms and "post_downstream" not in terms
    assert effect["base"] == "pd_tbpost_y1" and effect["re_base"] == "post_downstream"
    assert effect["time:y4_5"] == "pd_tbpost_y4_5"
    # Years outside every bin are dropped and counted.
    _, _, dropped = dm.add_modifiers(daily, "baci_time", time_bins=[("pre", -2, -1), ("y1", 1, 1)])
    assert dropped == len(daily[~daily["years_from_removal"].isin([-2, -1, 1])])


def test_time_model_recovers_a_fading_effect_and_beats_the_binary_one():
    profile = {1: 3.0, 2: 2.0, 3: 2.0, 4: 1.0, 5: 1.0}
    daily = simulate(profile=profile)
    results, _, at = fit(daily, ["baci_binary", "baci_time"], reml=False)
    t = results.set_index("spec").loc["baci_time"]
    assert t["error"] == "" and t["time_change_p"] < 0.001 and np.isnan(t["pretrend_p"])
    by_bin = at[at["spec"].eq("baci_time")].set_index("time_bin")
    for b, truth in {"y1": 3.0, "y2_3": 2.0, "y4_5": 1.0}.items():
        assert by_bin.loc[b, "ci_low"] < truth < by_bin.loc[b, "ci_high"], b
    # The average over post bins (treated-day weights: 1, 2, 2 years) is the binary estimand.
    assert by_bin.loc["post average", "ci_low"] < 1.8 < by_bin.loc["post average", "ci_high"]
    assert by_bin.loc["post average", "treated_days_in_bin"] == by_bin.loc[
        ["y1", "y2_3", "y4_5"], "treated_days_in_bin"].sum()
    comparison = dm.time_comparison(results).iloc[0]
    assert comparison["same_sample"] and comparison["ml"]
    assert comparison["lrt_df"] == 4 and comparison["lrt_p"] < 0.001
    assert comparison["delta_aic_time_minus_binary"] < 0
    assert abs(comparison["binary_effect"] - 1.8) < 0.3


def test_constant_effect_shows_no_time_change():
    results, _, _ = fit(simulate(), ["baci_binary", "baci_time"], reml=False)
    t = results.set_index("spec").loc["baci_time"]
    assert t["time_change_p"] > 0.05
    comparison = dm.time_comparison(results).iloc[0]
    assert comparison["lrt_p"] > 0.05 and comparison["delta_aic_time_minus_binary"] > 0
    # A time fit below the nested binary likelihood is an optimizer failure, not a test.
    broken = results.copy()
    broken.loc[broken["spec"].eq("baci_time"), "llf"] -= 5
    row = dm.time_comparison(broken).iloc[0]
    assert np.isnan(row.get("lrt_p", np.nan)) and "stopped short" in row["lrt_note"]
    # REML fits: the comparison says why there is no likelihood test.
    results, _, _ = fit(simulate(), ["baci_binary", "baci_time"])
    assert "--ml" in dm.time_comparison(results).iloc[0]["lrt_note"]


def test_pretrend_placebo():
    settings = {**SETTINGS, "time_bins": dm.TIME_BINNINGS["pretrend"]}
    flat, _, _ = fit(simulate(), ["baci_time"], settings)
    assert flat.iloc[0]["pretrend_df"] == 1 and flat.iloc[0]["pretrend_p"] > 0.05
    drift, _, at = fit(simulate(early_pre_drift=1.0), ["baci_time"], settings)
    assert drift.iloc[0]["pretrend_p"] < 0.001
    pre = at[at["time_bin_period"].eq("pre")].set_index("time_bin")
    assert list(pre.index) == ["pre_early"] and pre.loc["pre_early", "ci_low"] > 0


def test_time_spec_keeps_dam_slopes_on_post():
    results, _, _ = fit(simulate(), ["baci_time"], re_mode="slopes")
    row = results.iloc[0]
    assert row["error"] == "" and row["re_choice"] == "slopes (requested)"
    assert "var_dam_post_downstream" in row and np.isfinite(row["var_dam_post_downstream"])


def test_intervals_report_dams_and_observations_by_role():
    daily = simulate(n_dams=12)
    # Remove the year-5 control data at three dams: they then identify y4_5 from one role only.
    daily = daily[~(daily["years_from_removal"].eq(5) & daily["AR_ID"].isin(["D00", "D01", "D02"])
                    & daily["role"].eq("control"))]
    _, _, at = fit(daily, ["baci_time"], {**SETTINGS, "min_dams": 5})
    at = at.set_index("time_bin")
    y1 = at.loc["y1"]
    assert y1["treated_dams_in_bin"] == 12 and y1["control_dams_in_bin"] == 12
    assert y1["dams_both_roles_in_bin"] == 12
    assert y1["treated_days_in_bin"] == 12 * 2 * 4 and y1["control_days_in_bin"] == 12 * 4
    late = at.loc["y4_5"]
    assert late["dams_both_roles_in_bin"] == 12          # year 4 still has controls everywhere
    assert late["control_days_in_bin"] == 12 * 4 * 2 - 3 * 4
    avg = at.loc["post average"]
    assert avg["treated_days_in_bin"] == 12 * 2 * 4 * 5
    _, _, ba = fit(daily, ["before_after_time"], {**SETTINGS, "min_dams": 5})
    assert ba["control_dams_in_bin"].isna().all() and ba["dams_both_roles_in_bin"].isna().all()


def test_ml_comparison_with_dam_slopes_needs_matching_random_effects():
    profile = {1: 3.0, 2: 2.0, 3: 2.0, 4: 1.0, 5: 1.0}
    results, _, _ = fit(simulate(profile=profile), ["baci_binary", "baci_time"],
                        re_mode="slopes", reml=False)
    assert results["re_choice"].eq("slopes (requested)").all()
    row = dm.time_comparison(results).iloc[0]
    assert row["same_sample"] and row["same_random_effects"] and row["ml"]
    assert row["binary_re_structure"] == row["time_re_structure"] == "dam slopes + station intercepts"
    assert row["lrt_df"] == 4 and row["lrt_p"] < 0.001       # only fixed effects differ
    # Different random effects (e.g. one fit fell back to intercepts): no likelihood test.
    mixed = results.copy()
    mixed.loc[mixed["spec"].eq("baci_time"), "re_structure"] = "dam intercepts + station intercepts"
    row = dm.time_comparison(mixed).iloc[0]
    assert not row["same_random_effects"] and np.isnan(row.get("lrt_p", np.nan))
    assert "random effects differ" in row["lrt_note"]


def test_coarse_bins_sensitivity():
    settings = {**SETTINGS, "time_bins": dm.TIME_BINNINGS["y1-2_y3-5"]}
    profile = {1: 3.0, 2: 3.0, 3: 1.0, 4: 1.0, 5: 1.0}
    results, _, at = fit(simulate(profile=profile), ["baci_binary", "baci_time"], settings)
    t = results.set_index("spec").loc["baci_time"]
    assert t["time_bins"] == "pre:-5:-1;y1_2:1:2;y3_5:3:5" and t["time_change_df"] == 1
    assert t["time_change_p"] < 0.001
    by_bin = at[at["spec"].eq("baci_time")].set_index("time_bin")
    assert list(by_bin.index) == ["y1_2", "y3_5", "post average"]
    for b, truth in {"y1_2": 3.0, "y3_5": 1.0}.items():
        assert by_bin.loc[b, "ci_low"] < truth < by_bin.loc[b, "ci_high"], b
    assert by_bin.loc["y1_2", "dams_both_roles_in_bin"] == 30
    assert dm.time_comparison(results).iloc[0]["same_sample"]


def test_time_coverage_audit():
    daily = simulate(n_dams=12)
    daily = daily[~(daily["years_from_removal"].eq(5) & daily["AR_ID"].isin(["D00", "D01", "D02"])
                    & daily["role"].eq("control"))]
    by_year, by_bin = dm.time_coverage(daily, SETTINGS)
    baci = by_year[by_year["design"].eq("baci")].set_index("years_from_removal")
    assert baci.loc[1, "dams_both_roles"] == 12 and baci.loc[5, "dams_both_roles"] == 9
    assert baci.loc[5, "treated_dams"] == 12 and baci.loc[5, "control_dams"] == 9
    yearly = by_bin[by_bin["design"].eq("baci") & by_bin["binning"].eq("yearly")]
    assert (yearly["min_dams_post_bins"] == 9).all() and not yearly["supported"].any()
    pooled = by_bin[by_bin["design"].eq("baci") & by_bin["binning"].eq("y1_y2-3_y4-5")]
    assert (pooled["min_dams_post_bins"] == 12).all() and pooled["supported"].all()
    assert pooled.set_index("bin").loc["pre", "reference"]


# --------------------------------------------------
# Design 12: characteristics as effect modifiers
# --------------------------------------------------

def test_characteristic_specs_put_the_modifier_on_the_effect():
    spec = dm.SPECS["baci_x_dam_height"]
    assert spec == {"design": "baci", "station": ["ld", "ldar"], "dam": ["m_dam_height"]}
    assert dm.SPECS["baci_xu_pct_forest"] == {"design": "baci", "station": ["m_pct_forest"],
                                              "dam": []}
    # Distance as a characteristic replaces ld, keeps the dilution adjustment.
    assert dm.SPECS["baci_x_distance"]["station"] == ["ldar", "m_distance"]
    rows, _, _ = dm.add_modifiers(simulate(n_dams=12), "baci_xu_dam_height")
    terms, effect = dm.term_columns(rows, "baci_xu_dam_height")
    assert terms == ["post", "downstream", "post_downstream", "m_dam_height",
                     "post_m_dam_height", "down_m_dam_height", "pd_m_dam_height"]
    assert effect["m_dam_height"] == "pd_m_dam_height"
    rows, _, _ = dm.add_modifiers(simulate(n_dams=12), "baci_xu_pct_forest")
    assert rows.loc[rows["role"].eq("control"), "m_pct_forest"].eq(0).all()   # downstream only
    terms, _ = dm.term_columns(rows, "baci_xu_pct_forest")
    assert terms[3:] == ["down_m_pct_forest", "pd_m_pct_forest"]


def test_recovers_dam_and_station_level_modification():
    daily = simulate(height_slope=1.0, forest_slope=-0.8)
    results, _, at = fit(daily, ["baci_xu_dam_height", "baci_x_dam_height",
                                 "baci_xu_pct_forest", "before_after_xu_dam_height"])
    results = results.set_index("spec")
    for spec, truth in [("baci_xu_dam_height", 1.0), ("baci_x_dam_height", 1.0),
                        ("baci_xu_pct_forest", -0.8), ("before_after_xu_dam_height", 1.0)]:
        r = results.loc[spec]
        assert r["error"] == "" and r["modifier"] == spec.split("_x", 1)[1].split("_", 1)[1]
        assert r["modifier_ci_low"] < truth < r["modifier_ci_high"], spec
        assert r["modifier_p_value"] < 0.001 and "per 1 SD" in r["modifier_scale"]
    assert results.loc["baci_xu_pct_forest", "modifier_level"] == "station"
    height = at[at["spec"].eq("baci_xu_dam_height")].set_index("characteristic_percentile")
    assert height.loc[10, "estimate"] < height.loc[50, "estimate"] < height.loc[90, "estimate"]
    assert height.loc[10, "characteristic_value"] < height.loc[90, "characteristic_value"]
    # No modification simulated: none detected.
    null, _, _ = fit(simulate(), ["baci_xu_dam_height"])
    assert null.iloc[0]["modifier_p_value"] > 0.05


def test_missing_values_are_dropped_and_counted_not_imputed():
    daily = simulate(missing_height_every=5)
    results, _, _ = fit(daily, ["baci_xu_dam_height"])
    row = results.iloc[0]
    assert row["dams"] == 24 and row["dams_before_modifier"] == 30
    assert row["rows_dropped_missing_modifier"] > 0
    # Station level: a treated station without a value loses its rows, not the dam.
    daily = simulate(n_dams=12)
    daily.loc[daily["station_id"].eq("D00-S1"), "pct_forest"] = np.nan
    rows, _, dropped = dm.add_modifiers(daily, "baci_xu_pct_forest")
    assert dropped == daily["station_id"].eq("D00-S1").sum()
    assert {"D00-S0", "D00-S2"} <= set(rows["station_id"])
    # Reservoir area counts only polygons mapped by the removal year.
    values = dm.characteristic_values(daily, "reservoir_area")
    after = daily["reservoir_polygon_timing"].eq("edited_after_removal")
    assert values[after].isna().all() and values[~after].notna().all()


def test_modifier_audit_correlations_and_combined_check():
    daily = simulate(n_dams=20, missing_height_every=4)
    availability, correlations = dm.modifier_audit(daily, SETTINGS)
    baci = availability[availability["design"].eq("baci")].set_index("characteristic")
    assert baci.loc["dam_height", "dams"] == 20 and baci.loc["dam_height", "dams_with_value"] == 15
    assert baci.loc["pct_forest", "treated_pairs_with_value"] == 40
    assert availability.set_index("characteristic").loc["stored_sediment", "available"] == False  # noqa: E712
    pair = correlations[correlations["design"].eq("baci")
                        & correlations["a"].eq("pct_forest") & correlations["b"].eq("pct_developed")]
    assert abs(pair.iloc[0]["spearman_rho"] + 1) < 1e-9          # 100 - forest
    check = dm.combined_check(daily, SETTINGS, ["dam_height", "pct_forest"]).set_index("design")
    assert check.loc["baci", "complete_dams"] == 15 and not check.loc["baci", "justified"]
    assert "complete dams < 20" in check.loc["baci", "reasons"]
    check = dm.combined_check(daily, SETTINGS, ["pct_forest", "pct_developed"]).set_index("design")
    assert "|rho| 1.00" in check.loc["baci", "reasons"]


def test_combined_spec_and_screen():
    dm.register_combined(["dam_height", "pct_forest"])
    try:
        spec = dm.SPECS["baci_x_combined"]
        assert spec["dam"] == ["m_dam_height"] and spec["station"] == ["ld", "ldar", "m_pct_forest"]
        daily = simulate(height_slope=1.0)
        results, coefficients, _ = fit(daily, ["baci_x_combined", "baci_x_dam_height",
                                               "baci_x_pct_forest"])
        combined = results.set_index("spec").loc["baci_x_combined"]
        assert combined["error"] == "" and pd.isna(combined.get("modifier"))
        terms = set(coefficients.loc[coefficients["spec"].eq("baci_x_combined"), "term"])
        assert {"pd_m_dam_height", "pd_m_pct_forest", "pd_ld", "pd_ldar"} <= terms
        assert coefficients["p_value"].between(0, 1).all()
        screen = dm.modifier_screen(results)
        assert set(screen["characteristic"]) == {"dam_height", "pct_forest"}   # not combined
        assert (screen["q_value"] >= screen["modifier_p_value"] - 1e-12).all()
        assert screen.set_index("characteristic").loc["dam_height", "q_value"] < 0.01
    finally:
        dm._characteristic_specs()                    # reset the placeholder
    try:
        dm.register_combined(["sediment"])
    except ValueError:
        pass
    else:
        raise AssertionError("unknown characteristic accepted")


def test_bh_qvalues():
    q = dm.bh_qvalues([0.01, 0.04, 0.03, np.nan, 0.5])
    # Sorted p 0.01, 0.03, 0.04, 0.5 -> p * 4 / rank 0.04, 0.06, 0.053, 0.5 -> step-up minimum.
    assert np.allclose(q[[0, 1, 2, 4]], [0.04, 0.16 / 3, 0.16 / 3, 0.5]) and np.isnan(q[3])


# --------------------------------------------------
# Command line
# --------------------------------------------------

def test_command_line_time_and_characteristic_runs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        source = tmp / "analysis_measurement.parquet"
        simulate(n_dams=14, raw=True).to_parquet(source, index=False)
        none = tmp / "none.csv"
        pd.DataFrame({"AR_ID": []}).to_csv(none, index=False)
        common = ["distance_models.py", "--case-study", "test", "--input", str(source),
                  "--series", "water_temp", "--exclude-dams", str(none)]

        def run(*extra, out):
            with mock.patch.object(sys, "argv", common + ["--output-dir", str(tmp / out),
                                                          *extra]), \
                    mock.patch.dict(os.environ, {}):
                dm.main()
            return tmp / out

        out = run("--audit-only", out="audit")
        assert {p.name for p in out.iterdir()} == {
            "time_coverage_by_year.csv", "time_coverage_by_bin.csv", "modifier_audit.csv",
            "modifier_correlations.csv", "settings.json"}
        out = run("--spec-sets", "time", "--ml", "--min-dams", "5", out="time")
        results = pd.read_csv(out / "results.csv")
        assert set(results["spec"]) == set(dm.SPEC_SETS["time"])
        assert results["analysis_role"].eq("primary").all()
        # A sensitivity run is labelled as such in every table and in settings.json.
        sens = run("--specs", "baci_binary", "baci_time", "--min-dams", "5", "--ml", "--re", "slopes",
                   "--time-bins", "y1-2_y3-5", "--analysis-role", "sensitivity", out="sens")
        for name in ("results.csv", "effects_at.csv", "coefficients.csv", "time_comparison.csv"):
            assert pd.read_csv(sens / name)["analysis_role"].eq("sensitivity").all(), name
        import json
        info = json.loads((sens / "settings.json").read_text())
        assert info["analysis_role"] == "sensitivity" and info["re"] == "slopes"
        assert info["time_bins"][1] == ["y1_2", 1, 2]
        compared = pd.read_csv(sens / "time_comparison.csv").iloc[0]
        assert compared["same_random_effects"] and compared["ml"]
        comparison = pd.read_csv(out / "time_comparison.csv")
        assert set(comparison["design"]) == {"baci", "before_after"} and comparison["ml"].all()
        assert (out / "time_coverage_by_bin.csv").exists()
        out = run("--specs", "baci_x_dam_height", "baci_x_combined", "--combined", "dam_height",
                  "pct_forest", out="chars")
        for name in ("modifier_screen.csv", "modifier_audit.csv", "combined_check.csv"):
            assert (out / name).exists(), name
        screen = pd.read_csv(out / "modifier_screen.csv")
        assert list(screen["characteristic"]) == ["dam_height"]
        # Too few dams for any fit: the comparison is empty but readable.
        out = run("--spec-sets", "time", "--min-dams", "50", out="few")
        assert pd.read_csv(out / "time_comparison.csv").empty
        dm._characteristic_specs()
        try:
            run("--specs", "baci_x_combined", out="bad")
        except SystemExit:
            pass
        else:
            raise AssertionError("combined spec ran without --combined")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
