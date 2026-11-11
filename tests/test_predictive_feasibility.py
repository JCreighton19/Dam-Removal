"""Checks for analysis/predictive_feasibility.py (Stage 1 audit).

Simulated measurement rows go through the frozen mixed_model.prepare, so the
row rules are the real ones. Dams have different numbers of downstream
stations at different distances (one beyond 10 km), some dams lack years on
one side, one dam is on the suspect list, some predictors are missing, and
land cover for one dam was mapped after its removal.

Run directly: `python tests/test_predictive_feasibility.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import mixed_model as mm
from analysis import predictive_feasibility as pf

SERIES = ["water_temp", "turbidity"]


def simulate(n_dams=12, seed=4):
    """Raw rows like analysis_measurement. Dam k has 1 + k % 3 stations at
    0.5, 4 and 8 km; dam D01 also has a station at 15 km. D02 has only one
    pre-removal year; D03 is a suspect dam; D04 has no post data at its
    second station."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_dams):
        dam = f"D{k:02d}"
        removal = 2005 + k % 5
        n_st = 1 + k % 3
        km = [0.5, 4.0, 8.0][:n_st] + ([15.0] if dam == "D01" else [])
        years = [y for y in range(-5, 6) if y != 0]
        if dam == "D02":
            years = [-1] + [y for y in years if y > 0]
        for s, d in enumerate(km):
            for yfr in years:
                if dam == "D04" and s == 1 and yfr > 0:
                    continue
                for month in (4, 7, 10):
                    for param in SERIES:
                        post = yfr > 0
                        rows.append({
                            "AR_ID": dam, "station_id": f"{dam}-S{s}", "parameter_std": param,
                            "sample_fraction_std": pd.NA, "side": "downstream",
                            "period": "post" if post else "pre",
                            "sample_date": pd.Timestamp(removal + yfr, month, 15), "month": month,
                            "result_value_std": float(rng.uniform(5, 15)),
                            "exposure_status": "assigned_sole", "upstream_in_former_reservoir": pd.NA,
                            "air_temp_mean_c": 10.0 if not post else 20.0,     # pre vs post differ
                            "precip_prev_month_mm": float(rng.uniform(20, 80)),
                            "precip_mm": 50.0, "qc_non_perennial": False, "has_same_day_discharge": True,
                            "network_distance_km": d, "years_from_removal": yfr, "removal_year": removal,
                            "drainage_area_ratio": 1.5 + s,
                            "land_cover_year": 2011 if dam == "D05" else 2001,   # D05: mapped after removal
                            "reservoir_polygon_timing": "edited_by_removal_year" if k % 2 else "after",
                            "dam_height_ft": np.nan if k % 4 == 0 else 3.0 + k,
                            "dam_drainage_area_km2": 10.0 * (k + 1),
                            "reservoir_volume_proxy": 100.0 * (k + 1),     # monotone with drainage area
                            "pct_agriculture": 10.0 + s, "pct_forest": 50.0 - k, "pct_developed": 5.0,
                            "dam_age_at_removal": 50 + k, "reservoir_polygon_area_km2": 0.1 * (k + 1),
                        })
    return pd.DataFrame(rows)


REGIONS = pd.DataFrame({"AR_ID": [f"D{k:02d}" for k in range(12)],
                        "HUC8": [1080201 if k < 6 else 18020158 for k in range(12)]})


def daily(df=None):
    settings = pf.audit_settings(SERIES)
    return mm.prepare(simulate() if df is None else df, settings)


def test_rows_follow_frozen_rules_and_distance():
    d = daily()
    primary = pf.treated_rows(d, exclude_dams=["D03"])
    window = pf.treated_rows(d, exclude_dams=["D03"], max_km=pf.WINDOW_KM)
    assert "D03" not in set(primary["AR_ID"])
    assert primary["network_distance_km"].max() <= 10 and window["network_distance_km"].max() == 15
    # D04's second station has no post data, so that pair is dropped; its first stays.
    assert set(primary.loc[primary["AR_ID"] == "D04", "station_id"]) == {"D04-S0"}
    # One value per dam x station x series x day.
    assert not primary.duplicated(["AR_ID", "station_id", "series", "sample_date"]).any()


def test_measure_audit_counts_dams_not_stations():
    rows = pf.treated_rows(daily(), exclude_dams=["D03"])
    m, per_dam = pf.measure_audit(rows, REGIONS, "10 km")
    t = m.set_index("series").loc["water_temp"]
    assert t["dams"] == 11                                     # 12 minus the suspect dam
    # D02 has one pre year: counted with >= 1 year each side, not >= 2.
    assert t["dams_years_ge1_each_side"] == 11 and t["dams_years_ge2_each_side"] == 10
    assert t["dams_estimable"] == 10
    # Unequal stations: dams with 1, 2 or 3 stations (D04 lost one pair).
    s = per_dam[per_dam["series"] == "water_temp"].set_index("AR_ID")["stations"]
    assert s["D00"] == 1 and s["D01"] == 2 and s["D02"] == 3 and s["D04"] == 1
    assert t["dams_one_station"] == int((s == 1).sum())
    # Regions from zero-padded HUC8.
    assert t["huc2_regions"] == 2 and set(per_dam["huc2"].dropna()) == {"01", "18"}
    assert not t["meets_size_rule"]                            # far fewer than 60 dams


def test_features_respect_pre_removal_and_missingness():
    rows = pf.treated_rows(daily(), exclude_dams=["D03"])
    f = pf.dam_features(rows).set_index("AR_ID")
    assert np.isclose(f.loc["D01", "pre_air_temp_c"], 10.0)       # pre days only (post are 20)
    assert pd.isna(f.loc["D05", "pct_forest"])                    # land cover mapped after removal
    assert pd.isna(f.loc["D00", "log_dam_height"]) and np.isclose(f.loc["D01", "log_dam_height"], np.log10(4))
    assert pd.isna(f.loc["D02", "log_reservoir_area_pre"])        # mapped after removal (k even)
    assert np.isclose(f.loc["D01", "log_reservoir_area_pre"], np.log10(0.2))
    _, per_dam = pf.measure_audit(rows, REGIONS, "10 km")
    a = pf.predictor_audit(f.reset_index(), per_dam)
    h = a[(a["series"] == "water_temp") & (a["predictor"] == "log_dam_height")].iloc[0]
    assert h["available"] < h["dams_estimable"] and 0 < h["completeness"] < 1


def test_correlated_predictors_and_cap():
    rows = pf.treated_rows(daily(), exclude_dams=["D03"])
    f = pf.dam_features(rows)
    _, pairs = pf.correlations(f)
    flagged = pairs[pairs["flag_abs_rho_gt_0_8"]]
    assert ((flagged["a"] == "log_dam_drainage_area") & (flagged["b"] == "log_reservoir_volume_proxy")).any()
    audit = pd.DataFrame([{"series": "x", "predictor": p, "primary_candidate": True, "meets_completeness": True}
                          for p in pf.PREDICTORS])
    chosen = pf.select_predictors(audit, flagged.assign(flag_abs_rho_gt_0_8=True), "x", n_dams=300)
    assert "log_dam_drainage_area" in chosen and "log_reservoir_volume_proxy" not in chosen
    assert len(pf.select_predictors(audit, flagged, "x", n_dams=45)) == 3      # floor(45 / 15)
    v = pf.vif(f, ["log_dam_drainage_area", "log_reservoir_volume_proxy", "dam_age_at_removal"])
    assert v["vif"].max() > 10                                                 # collinear trio


def test_distance_and_time_audits():
    d = daily()
    window = pf.treated_rows(d, exclude_dams=["D03"], max_km=pf.WINDOW_KM)
    dist, by_region = pf.distance_audit(window, REGIONS)
    t = dist.set_index("series").loc["water_temp"]
    assert t["station_km_min"] == 0.5 and t["dams_band_10_20km"] == 1 and t["dams_only_beyond_10km"] == 0
    # Dams with stations at 0.5, 4 and 8 km cover all three bands.
    assert t["dams_all_3_coverage_bands"] >= 1 and t["dams_ge2_stations_le_10km"] >= 1
    assert set(by_region["huc2"]) == {"01", "18"}
    primary = pf.treated_rows(d, exclude_dams=["D03"])
    temp, by_year = pf.temporal_audit(primary)
    tt = temp.set_index("series").loc["water_temp"]
    assert tt["rows_outside_periods"] == 0 and tt["dams_y1"] == tt["dams"]
    assert tt["dams_pre_early"] == tt["dams"] - 1                 # D02 has no years -5..-3
    assert 0 not in set(by_year["years_from_removal"])


def test_run_audit_writes_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        tables = pf.run_audit(daily(), REGIONS, ["D03"], tmp)
        assert (Path(tmp) / "report_audit.md").exists()
        assert (Path(tmp) / "audit_measures.csv").exists() and (Path(tmp) / "audit_distance.csv").exists()
        rec = tables["recommendations"].set_index("series")
        assert not rec.loc["water_temp", "meets_size_rule"]
        assert "Measures (primary window)" in (Path(tmp) / "report_audit.md").read_text()


# --------------------------------------------------
# Stage 2: targets
# --------------------------------------------------

B_LD, B_LDAR = -0.8, 0.5
STD = pd.Series({"series": "water_temp", "b_ld": B_LD, "se_ld": 0.1, "b_ldar": B_LDAR, "se_ldar": 0.2,
                 "reference_km": 5.0, "reference_dar": 2.0, "var_dam_post_30km": 1.0})


def simulate_daily(n_dams=40, seed=7, tau=1.0, year_sd=0.6, noise_sd=1.0):
    """Daily before/after rows (as treated_rows returns) with known dam effects at
    5 km / reference dilution: station change = true + B_LD * ld + B_LDAR * ldar.
    Dam k has 1-3 stations (one heavily sampled), D05 has a station without a
    dilution ratio, D06 a station at 20 km only, D07 one station with 10 days."""
    rng = np.random.default_rng(seed)
    truth, rows = {}, []
    for k in range(n_dams):
        dam = f"D{k:02d}"
        truth[dam] = rng.normal(0, tau)
        n_st = 1 + k % 3
        kms = [20.0] if dam == "D06" else ([3.0] if dam == "D07" else list(rng.uniform(0.2, 9.8, n_st)))
        for s, km in enumerate(kms):
            dar = np.nan if (dam == "D05" and s == 0) else float(rng.uniform(1.0, 8.0))
            ld = np.log1p(km) - np.log1p(5.0)
            ldar = 0.0 if np.isnan(dar) else np.log10(dar) - np.log10(2.0)
            change = truth[dam] + B_LD * ld + B_LDAR * ldar
            base = rng.normal(10, 2)
            per_year = 1 if dam == "D07" else (30 if s == 0 else 6)      # unequal sampling
            for yfr in [y for y in range(-5, 6) if y != 0]:
                shock = rng.normal(0, year_sd)                             # same-year correlation
                for _ in range(per_year):
                    month = int(rng.integers(1, 13))
                    air = rng.normal(0, 1)
                    y = (base + 2 * np.sin(month / 12 * 2 * np.pi) + 0.7 * air + shock
                         + (change if yfr > 0 else 0) + rng.normal(0, noise_sd))
                    rows.append({"AR_ID": dam, "station_id": f"{dam}-S{s}", "series": "water_temp",
                                 "role": "treated", "y": y, "post": int(yfr > 0), "month": month,
                                 "sample_date": pd.Timestamp(2010 + yfr, month, 1),
                                 "years_from_removal": yfr, "air_temp_mean_c": air,
                                 "precip_prev_month_mm": rng.uniform(0, 100),
                                 "network_distance_km": km, "drainage_area_ratio": dar})
    return pd.DataFrame(rows), pd.Series(truth)


def fit_dam_targets(daily_rows, standardize=True):
    st, V2, V1, _ = pf.fit_station_changes(daily_rows)
    st = st.merge(pf.station_terms(daily_rows, 5.0, 2.0), on=["AR_ID", "station_id"], how="left")
    return st, pf.dam_targets(st, V2, V1, STD, standardize=standardize)


def test_targets_recover_known_dam_effects():
    d, truth = simulate_daily()
    rows = pf.treated_rows(d, max_km=10)
    st, t = fit_dam_targets(rows)
    t = t.set_index("AR_ID")
    err = t["target"] - truth.reindex(t.index)
    assert t.index.nunique() == len(t) and "D06" not in t.index          # beyond 10 km
    assert np.corrcoef(t["target"], truth.reindex(t.index))[0, 1] > 0.9
    assert abs(err.mean()) < 0.25
    # Honest uncertainty despite year shocks: most 95% intervals cover the truth.
    covered = ((t["ci_low"] <= truth.reindex(t.index)) & (truth.reindex(t.index) <= t["ci_high"])).mean()
    assert covered >= 0.85


def test_equal_station_weights_and_standardization():
    d, _ = simulate_daily()
    rows = pf.treated_rows(d, max_km=10)
    st, t = fit_dam_targets(rows)
    t = t.set_index("AR_ID")
    # D02 has 3 stations, one with 5x the days: the dam value is their plain mean.
    s2 = st[st["AR_ID"] == "D02"]
    assert s2["days"].max() >= 4 * s2["days"].min()
    assert np.isclose(t.loc["D02", "target_unstandardized"], s2["delta"].mean())
    # Standardization = the frozen shift at the dam's mean ld / ldar.
    shift = B_LD * s2["ld"].mean() + B_LDAR * s2["ldar"].mean()
    assert np.isclose(t.loc["D02", "target_unstandardized"] - t.loc["D02", "target"], shift)
    # D05's station without a dilution ratio is out of the standardized target only.
    _, u = fit_dam_targets(rows, standardize=False)
    u = u.set_index("AR_ID")
    assert t.loc["D05", "stations"] == u.loc["D05", "stations"] - 1


def test_coefficient_uncertainty_and_ses():
    d, _ = simulate_daily()
    st, t = fit_dam_targets(pf.treated_rows(d, max_km=10))
    mld, mldar = t["mean_ld"].abs(), t["mean_ldar"].abs()
    assert np.allclose(t["se_coef_bound"], mld * 0.1 + mldar * 0.2)
    assert np.allclose(t["se_coef_rho0"], np.sqrt(mld ** 2 * 0.01 + mldar ** 2 * 0.04))
    assert (t["se_coef_bound"] >= t["se_coef_rho0"] - 1e-12).all()
    assert np.allclose(t["se_total"] ** 2, t["se_cr2"] ** 2 + t["se_coef_bound"] ** 2)
    assert (t["se_cr2"] > 0).all() and (t["clusters"] >= 4).all()
    # The station-year-means check gives similar estimates.
    assert np.corrcoef(t["target"], t["target_ym"])[0, 1] > 0.9
    # Nearest-station subset keeps alignment with V (index = position).
    nearest = st.loc[st[st["ldar"].notna()].groupby("AR_ID")["km"].idxmin()].sort_index()
    st2, V2, V1, _ = pf.fit_station_changes(pf.treated_rows(d, max_km=10))
    n = pf.dam_targets(nearest, V2, V1, STD).set_index("AR_ID")
    one = nearest[nearest["AR_ID"] == "D02"].index[0]
    assert np.isclose(n.loc["D02", "se_cr2"], np.sqrt(V2[one, one]))


def test_weak_support_flags():
    d, _ = simulate_daily()
    _, t = fit_dam_targets(pf.treated_rows(d, max_km=10))
    f = pf.flag_weak(t).set_index("AR_ID")
    assert f.loc["D07", "flag_few_days"] and f.loc["D07", "weak_support"]


def test_tau2_estimators_account_for_measurement_error():
    rng = np.random.default_rng(1)
    se = rng.uniform(0.5, 2.0, 2000)
    est = rng.normal(0, np.sqrt(1.0 + se ** 2))                  # true tau2 = 1
    assert abs(pf.tau2_pm(est, se) - 1) < 0.3 and abs(pf.tau2_dl(est, se) - 1) < 0.3
    assert np.var(est, ddof=1) > 2                                 # raw variance would overstate it
    null = rng.normal(0, se)                                       # tau2 = 0
    assert pf.tau2_pm(null, se) < 0.15 and pf.tau2_dl(null, se) < 0.15
    r, rd = pf.reliability(1.0, np.array([1.0, 1.0]))
    assert np.isclose(r, 0.5) and np.isclose(rd, 0.5)


def test_build_targets_versions_and_30km():
    d, _ = simulate_daily()
    rows10, rows30 = pf.treated_rows(d, max_km=10), pf.treated_rows(d, max_km=30)
    regions = pd.DataFrame({"AR_ID": d["AR_ID"].unique(), "HUC8": "01080201"})
    std = pd.DataFrame([STD])
    frozen10 = pd.DataFrame({"series": ["water_temp"], "effect_sd_between_dams": [1.0]})
    blups = pd.DataFrame({"AR_ID": ["D00", "D01"], "dam_effect": [0.1, 0.2], "series": "water_temp",
                          "model": "before_after"})
    targets, stations, comp, rel = pf.build_targets(rows10, rows30, std, frozen10, blups, regions)
    versions = set(targets["version"])
    assert {"primary_10km", "years_1_3", "sensitivity_30km", "unstandardized", "nearest_station",
            "frozen_blup_10km"} <= versions
    y13 = targets[targets["version"] == "years_1_3"]
    assert (y13["share_post_days_y4_5"] == 0).all()
    c30 = comp.set_index("version").loc["sensitivity_30km"]
    assert c30["dams_added_vs_primary"] == 1                       # D06, only beyond 10 km
    r = rel[(rel["version"] == "primary_10km") & (rel["se_kind"] == "se_total")].set_index("tau2_source")
    assert np.isclose(r.loc["frozen_10km_mixed (sd^2)", "tau2"], 1.0)   # SD squared
    assert r.loc["targets_paule_mandel", "tau2"] < r.loc["targets_paule_mandel", "raw_var_targets"]
    assert not r["stop_rule_met"].any()


def test_frozen_standardization_reads_outputs():
    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame({"series": ["ph"], "spec": ["before_after_distance"], "error": [""], "dams": [267],
                      "reference_km": [5.0], "reference_dar": [3.2], "var_dam_post": [0.02],
                      "re_structure": ["dam slopes"]}).to_csv(Path(tmp) / "results.csv", index=False)
        pd.DataFrame({"series": ["ph"] * 3, "spec": ["before_after_distance"] * 3,
                      "term": ["post", "post_ld", "post_ldar"], "estimate": [0.03, -0.006, 0.01],
                      "se": [0.01, 0.01, 0.02]}).to_csv(Path(tmp) / "coefficients.csv", index=False)
        s = pf.frozen_standardization(tmp).iloc[0]
        assert s["b_ld"] == -0.006 and s["se_ldar"] == 0.02 and s["reference_dar"] == 3.2
        assert s["var_dam_post_30km"] == 0.02


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
