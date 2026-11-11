"""
Predictive-feasibility analysis (docs/predictive_feasibility_prespec.md):
can pre-removal dam and watershed characteristics predict a dam's
before/after water-quality change downstream?

Post-hoc and read-only with respect to the frozen models
(docs/model_freeze.md): it reads analysis_measurement.parquet, the warehouse
dam table and the frozen mixed-model outputs, refits nothing frozen and
writes only to data/processed/predictive/<case_study>/.

Stages (run one at a time; each is a checkpoint in the pre-specification)
------------------------------------------------------------------------
audit   Stage 1, the data feasibility audit (no targets, no models):
        per measure, dams / stations / daily values, dams with enough
        years before and after, HUC2 / HUC4 coverage; predictor
        completeness, correlation, VIF and geography; downstream distance
        coverage; post-removal time coverage. Writes audit_*.csv and
        report_audit.md.
targets Stage 2 (Amendment 1), the dam-level targets, no models: one
        fixed-effects regression per measure, y = station + month +
        weather + delta_s * post, with station-year cluster-robust (CR2,
        CR1) covariances; each dam = the equal-weight mean of its stations'
        delta_s, standardized to 5 km and the reference dilution with the
        frozen before/after distance model's post_ld / post_ldar (their SEs
        propagated by the delta method, covariance bounded). Sensitivity
        targets (unstandardized, nearest station, years 1-3, frozen BLUPs,
        30 km), station-year-means check, weak-support flags, reliability.
        Writes targets*.csv and report_targets.md.
cv      Stage 3 (prespec 7-8, Amendment 2), from targets.csv: models A
        (overall mean), B (region mean), C (ridge, primary), D (small random
        forest, exploratory); leave-one-HUC2-out (primary) and HUC4-grouped
        5-fold x 20; predictor selection, imputation, scaling and penalty
        inside training folds; HUC2 cluster bootstrap; go / no-go;
        sensitivity analyses (weighted, complete cases, removal year, weak
        support excluded, years 1-3, 30 km) and conductance influence.
        Writes cv_*.csv and report_cv.md.

The rows are the frozen before/after rows, built with the frozen code
(mixed_model.prepare: treated roles, perennial filter, same-day discharge
for SSC / TSS, one median per dam x station x series x day, log10 scale),
then: suspect dams excluded (config/suspect_dam_matches.csv, as the distance
models), and each dam-station pair needing days both before and after.
"Primary" = treated stations within 10 km (the frozen before/after rule);
the 30 km warehouse window is audited alongside for comparison only.

Usage (cluster, via data/slurm/predictive_feasibility.slurm):
    python analysis/predictive_feasibility.py audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from analysis import mixed_model as mm
from etl.common.config import default_case_study, get_project_root

CANDIDATE_SERIES = ["water_temp", "dissolved_oxygen", "ph", "specific_conductance",
                    "phosphorus (Total)", "turbidity", "tds"]
AUDIT_ONLY_SERIES = ["ssc", "tss", "orthophosphate (Dissolved)"]   # go forward only if they qualify
PRIMARY_KM = 10.0
WINDOW_KM = 30.0
# Downstream distance bands (km, lower exclusive, upper inclusive; the first includes 0).
DISTANCE_BANDS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20), (20, 30)]
COVERAGE_BANDS = [(0, 2), (2, 5), (5, 10)]       # "coverage across 0-10 km"
# Post-removal periods present in the table: the removal year (0) and years
# beyond +5 are outside the analysis window (config wq_window; prespec 3.1).
PERIODS = {"pre_early": (-5, -3), "pre_late": (-2, -1),
           "y1": (1, 1), "y2_3": (2, 3), "y4_5": (4, 5)}
YEAR_THRESHOLDS = [1, 2, 3]
ESTIMABLE_MIN_YEARS = 2          # provisional (prespec 6): distinct sample years per side
MODEL_MIN_DAMS = 60              # prespec 4
MIN_REGION_DAMS = 5
MIN_REGIONS = 5
MIN_COMPLETENESS = 0.70          # prespec 5
MAX_ABS_RHO = 0.8
DAMS_PER_PREDICTOR = 15

# Prespec 5, in priority order: name -> (column, transform). Dam-level unless noted.
PREDICTORS = {
    "log_dam_height": ("dam_height_ft", "log10"),
    "log_dam_drainage_area": ("dam_drainage_area_km2", "log10"),
    "log_reservoir_volume_proxy": ("reservoir_volume_proxy", "log10"),
    "pct_agriculture": ("pct_agriculture", "station_median"),
    "pct_forest": ("pct_forest", "station_median"),
    "pct_developed": ("pct_developed", "station_median"),
    "pre_air_temp_c": ("air_temp_mean_c", "pre_mean"),
    "pre_precip_mm": ("precip_mm", "pre_mean"),
    "dam_age_at_removal": ("dam_age_at_removal", "identity"),
    "log_reservoir_area_pre": ("reservoir_polygon_area_km2", "log10_pre_removal"),
}
SENSITIVITY_FEATURES = {"removal_year": ("removal_year", "identity")}
CARRY = list(dict.fromkeys(
    ["network_distance_km", "years_from_removal", "removal_year", "land_cover_year",
     "reservoir_polygon_timing", "precip_mm", "drainage_area_ratio"]
    + [c for c, _ in PREDICTORS.values()]))

# Stage 2 (prespec Amendment 1)
TARGET_SERIES = ["water_temp", "dissolved_oxygen", "ph", "specific_conductance",
                 "phosphorus (Total)", "turbidity"]
STD_SPEC = "before_after_distance"           # frozen distance model whose post_ld / post_ldar standardize
COVARIATES = ["air_temp_mean_c", "precip_prev_month_mm"]
WEAK_MIN_CLUSTERS = 6
WEAK_MIN_DAYS = 20
WEAK_SE_MULTIPLE = 3.0
WEAK_YM_RATIO = (0.5, 2.0)
RELIABILITY_STOP = 0.15


def output_dir_for(case_study, project_root=None):
    root = Path(project_root) if project_root else get_project_root()
    return root / "data" / "processed" / "predictive" / case_study


def audit_settings(series=None):
    """The frozen before/after settings, every treated row in the 30 km window."""
    return {**mm.SETTINGS, "series": series or CANDIDATE_SERIES + AUDIT_ONLY_SERIES,
            "max_downstream_km": None, "carry_columns": CARRY}


# --------------------------------------------------
# Rows
# --------------------------------------------------

def treated_rows(daily, exclude_dams=(), max_km=PRIMARY_KM):
    """Before/after rows: treated, distance known and <= max_km, suspect dams out,
    and only dam-station pairs with days both before and after (as mm.model_rows)."""
    rows = daily[daily["role"].eq("treated") & ~daily["AR_ID"].isin(list(exclude_dams))].copy()
    rows["network_distance_km"] = pd.to_numeric(rows["network_distance_km"], errors="coerce")
    rows = rows[rows["network_distance_km"].le(max_km).fillna(False)]
    both = rows.groupby(["AR_ID", "station_id", "series"])["post"].transform("nunique").eq(2)
    rows = rows[both].copy()
    rows["year"] = pd.to_datetime(rows["sample_date"]).dt.year
    rows["years_from_removal"] = pd.to_numeric(rows["years_from_removal"], errors="coerce")
    return rows


def add_regions(frame, regions):
    """HUC2 / HUC4 from the warehouse dam table's HUC8 (prespec 2)."""
    r = regions[["AR_ID", "HUC8"]].drop_duplicates("AR_ID").copy()
    r["HUC8"] = r["HUC8"].astype("string").str.zfill(8)
    r["huc2"], r["huc4"] = r["HUC8"].str[:2], r["HUC8"].str[:4]
    return frame.merge(r[["AR_ID", "huc2", "huc4"]], on="AR_ID", how="left")


def dam_years(rows):
    """Per dam and series: distinct sample years before and after, days, stations."""
    g = rows.groupby(["series", "AR_ID"])
    out = g.agg(days=("y", "size"), stations=("station_id", "nunique")).reset_index()
    years = (rows.groupby(["series", "AR_ID", "post"])["year"].nunique()
             .unstack("post").reindex(columns=[0, 1]).fillna(0).astype(int))
    years.columns = ["pre_years", "post_years"]
    out = out.merge(years.reset_index(), on=["series", "AR_ID"], how="left")
    out["estimable"] = out[["pre_years", "post_years"]].min(axis=1).ge(ESTIMABLE_MIN_YEARS)
    return out


def _q(values, prefix):
    v = pd.Series(values, dtype=float).dropna()
    if v.empty:
        return {f"{prefix}_{k}": np.nan for k in ("min", "q25", "median", "mean", "q75", "max")}
    return {f"{prefix}_min": v.min(), f"{prefix}_q25": v.quantile(0.25), f"{prefix}_median": v.median(),
            f"{prefix}_mean": v.mean(), f"{prefix}_q75": v.quantile(0.75), f"{prefix}_max": v.max()}


# --------------------------------------------------
# Measures
# --------------------------------------------------

def measure_audit(rows, regions, window):
    """One row per series: counts, years before / after, regions, per-dam distributions."""
    per_dam = add_regions(dam_years(rows), regions)
    out = []
    for series, d in per_dam.groupby("series", sort=False):
        r = rows[rows["series"] == series]
        est = d[d["estimable"]]
        region_counts = est.groupby("huc2")["AR_ID"].nunique()
        row = {"series": series, "window": window, "dams": d["AR_ID"].nunique(),
               "stations": r[["AR_ID", "station_id"]].drop_duplicates().shape[0],
               "daily_values": len(r)}
        for k in YEAR_THRESHOLDS:
            row[f"dams_years_ge{k}_each_side"] = int(d[["pre_years", "post_years"]].min(axis=1).ge(k).sum())
        row.update({"dams_estimable": len(est), "huc2_regions": est["huc2"].nunique(),
                    "huc4_regions": est["huc4"].nunique(),
                    "huc2_regions_ge5_dams": int(region_counts.ge(MIN_REGION_DAMS).sum()),
                    "dams_missing_region": int(est["huc2"].isna().sum()),
                    "dams_one_station": int(d["stations"].eq(1).sum()),
                    **_q(d["days"], "days_per_dam"), **_q(d["stations"], "stations_per_dam")})
        row["meets_size_rule"] = bool(row["dams_estimable"] >= MODEL_MIN_DAMS
                                      and row["huc2_regions_ge5_dams"] >= MIN_REGIONS)
        out.append(row)
    return pd.DataFrame(out), per_dam


def dams_per_region(per_dam):
    est = per_dam[per_dam["estimable"]]
    return (est.groupby(["series", "huc2"])["AR_ID"].nunique().rename("dams").reset_index()
            .sort_values(["series", "dams"], ascending=[True, False]))


# --------------------------------------------------
# Predictors
# --------------------------------------------------

def _log10(x):
    x = pd.to_numeric(x, errors="coerce").astype(float)
    return np.log10(x.where(x > 0))


def dam_features(rows):
    """One row per dam with the prespec-5 predictors, from that dam's treated rows
    (all series). Leakage guards: land cover only if mapped before removal;
    climate from pre-removal sample days only; mapped reservoir area only if
    mapped by the removal year."""
    r = rows.drop_duplicates(["AR_ID", "station_id", "sample_date"])
    first = r.groupby("AR_ID").first()
    out = pd.DataFrame(index=first.index)
    lc_ok = (pd.to_numeric(r["land_cover_year"], errors="coerce")
             < pd.to_numeric(r["removal_year"], errors="coerce"))
    stations = r.assign(lc_ok=lc_ok).drop_duplicates(["AR_ID", "station_id"])
    pre = r[r["post"].eq(0)]
    for name, (col, how) in {**PREDICTORS, **SENSITIVITY_FEATURES}.items():
        if how == "log10":
            out[name] = _log10(first[col])
        elif how == "log10_pre_removal":
            ok = first["reservoir_polygon_timing"].eq("edited_by_removal_year")
            out[name] = _log10(first[col]).where(ok.fillna(False).astype(bool))
        elif how == "station_median":
            v = pd.to_numeric(stations[col], errors="coerce").where(stations["lc_ok"].fillna(False))
            out[name] = v.groupby(stations["AR_ID"]).median()
        elif how == "pre_mean":
            out[name] = pd.to_numeric(pre[col], errors="coerce").groupby(pre["AR_ID"]).mean()
        else:
            out[name] = pd.to_numeric(first[col], errors="coerce").astype(float)
    return out.reset_index()


def predictor_audit(features, per_dam):
    """Completeness of each predictor among each series' estimable dams."""
    out = []
    names = list(PREDICTORS) + list(SENSITIVITY_FEATURES)
    for series, d in per_dam[per_dam["estimable"]].groupby("series", sort=False):
        f = features[features["AR_ID"].isin(d["AR_ID"])]
        n = len(d)
        for name in names:
            avail = int(f[name].notna().sum()) if name in f else 0
            out.append({"series": series, "predictor": name, "dams_estimable": n, "available": avail,
                        "completeness": avail / n if n else np.nan,
                        "primary_candidate": name in PREDICTORS,
                        "meets_completeness": n > 0 and avail / n >= MIN_COMPLETENESS})
    return pd.DataFrame(out)


def correlations(features, names=None):
    """Spearman correlations (pairwise complete) and flagged pairs."""
    names = [n for n in (names or PREDICTORS) if n in features and features[n].notna().sum() > 2]
    rho = features[names].corr(method="spearman", min_periods=10)
    pairs = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            both = features[[a, b]].dropna()
            pairs.append({"a": a, "b": b, "spearman_rho": rho.loc[a, b], "dams_both": len(both),
                          "flag_abs_rho_gt_0_8": bool(abs(rho.loc[a, b]) > MAX_ABS_RHO)
                          if pd.notna(rho.loc[a, b]) else False})
    return rho, pd.DataFrame(pairs)


def vif(features, names):
    """Variance inflation factors on complete cases (diag of the inverse correlation)."""
    x = features[names].dropna()
    if len(x) <= len(names) + 1 or len(names) < 2:
        return pd.DataFrame({"predictor": names, "vif": np.nan, "complete_cases": len(x)})
    corr = np.corrcoef(x.to_numpy(float), rowvar=False)
    values = np.diag(np.linalg.pinv(corr))
    return pd.DataFrame({"predictor": names, "vif": values, "complete_cases": len(x)})


def select_predictors(pred_audit, pairs, series, n_dams):
    """Prespec 5 rules: completeness >= 70%, drop the lower-priority member of
    any |rho| > 0.8 pair, at most floor(n / 15), in priority order."""
    a = pred_audit[(pred_audit["series"] == series) & pred_audit["primary_candidate"]]
    keep = [p for p in PREDICTORS if p in set(a.loc[a["meets_completeness"], "predictor"])]
    flagged = pairs[pairs["flag_abs_rho_gt_0_8"]] if len(pairs) else pd.DataFrame(columns=["a", "b"])
    chosen = []
    for p in keep:
        partners = set(flagged.loc[flagged["a"] == p, "b"]) | set(flagged.loc[flagged["b"] == p, "a"])
        if not partners & set(chosen):
            chosen.append(p)
    return chosen[: max(n_dams // DAMS_PER_PREDICTOR, 0)]


def predictors_by_region(features, per_dam):
    est = per_dam[per_dam["estimable"]][["AR_ID", "huc2"]].drop_duplicates("AR_ID")
    f = features.merge(est, on="AR_ID")
    agg = {"dams": ("AR_ID", "nunique")}
    for name in PREDICTORS:
        agg[f"{name}_median"] = (name, "median")
        agg[f"{name}_missing"] = (name, lambda s: float(s.isna().mean()))
    return f.groupby("huc2").agg(**agg).reset_index()


# --------------------------------------------------
# Distance and time
# --------------------------------------------------

def _band(km):
    for lo, hi in DISTANCE_BANDS:
        if (km > lo or (lo == 0 and km >= 0)) and km <= hi:
            return f"{lo}-{hi}"
    return None


def distance_audit(rows_window, regions):
    """Per series: station distances (one per dam-station pair), dams per band,
    multi-station dams and coverage across 0-10 km; and by HUC2."""
    pairs = (rows_window.groupby(["series", "AR_ID", "station_id"])["network_distance_km"].first()
             .reset_index())
    pairs["band"] = pairs["network_distance_km"].map(_band)
    pairs = add_regions(pairs, regions)
    out, by_region = [], []
    for series, p in pairs.groupby("series", sort=False):
        within = p[p["network_distance_km"] <= PRIMARY_KM]
        d = within.groupby("AR_ID")["network_distance_km"]
        cov = within.assign(cb=within["network_distance_km"].map(
            lambda km: next((f"{lo}-{hi}" for lo, hi in COVERAGE_BANDS
                             if (km > lo or lo == 0) and km <= hi), None)))
        n_cov = cov.groupby("AR_ID")["cb"].nunique()
        row = {"series": series, **_q(within["network_distance_km"], "station_km"),
               "station_km_q10": within["network_distance_km"].quantile(0.1),
               "station_km_q90": within["network_distance_km"].quantile(0.9),
               "share_stations_le_1km": float((within["network_distance_km"] <= 1).mean()),
               "share_stations_le_2km": float((within["network_distance_km"] <= 2).mean()),
               "share_stations_le_5km": float((within["network_distance_km"] <= 5).mean()),
               **_q(d.min(), "nearest_station_km"),
               "dams_le_10km": within["AR_ID"].nunique(),
               "dams_ge2_stations_le_10km": int(d.size().ge(2).sum()),
               "dams_ge2_coverage_bands": int(n_cov.ge(2).sum()),
               "dams_all_3_coverage_bands": int(n_cov.ge(3).sum()),
               "dams_span_ge_3km": int((d.max() - d.min()).ge(3).sum()),
               "dams_le_30km": p["AR_ID"].nunique(),
               "dams_only_beyond_10km": int(p["AR_ID"].nunique() - within["AR_ID"].nunique())}
        for lo, hi in DISTANCE_BANDS:
            row[f"dams_band_{lo}_{hi}km"] = int(p.loc[p["band"] == f"{lo}-{hi}", "AR_ID"].nunique())
        out.append(row)
        for huc2, g in within.groupby("huc2"):
            by_region.append({"series": series, "huc2": huc2, "dams": g["AR_ID"].nunique(),
                              "stations": len(g), "median_km": g["network_distance_km"].median(),
                              "share_le_2km": float((g["network_distance_km"] <= 2).mean())})
    return pd.DataFrame(out), pd.DataFrame(by_region)


def temporal_audit(rows):
    """Per series: dams and stations with values in each period; per year from removal."""
    rows = rows.copy()
    rows["period_bin"] = None
    for name, (lo, hi) in PERIODS.items():
        rows.loc[rows["years_from_removal"].between(lo, hi), "period_bin"] = name
    out, by_year = [], []
    for series, r in rows.groupby("series", sort=False):
        row = {"series": series, "dams": r["AR_ID"].nunique(),
               "rows_outside_periods": int(r["period_bin"].isna().sum())}
        bins = r.groupby("AR_ID")["period_bin"].agg(lambda s: set(s.dropna()))
        for name in PERIODS:
            sub = r[r["period_bin"] == name]
            row[f"dams_{name}"] = sub["AR_ID"].nunique()
            row[f"stations_{name}"] = sub[["AR_ID", "station_id"]].drop_duplicates().shape[0]
            row[f"daily_values_{name}"] = len(sub)
        row["dams_all_post_periods"] = int(bins.map(lambda s: {"y1", "y2_3", "y4_5"} <= s).sum())
        row["dams_early_and_late"] = int(bins.map(lambda s: "y1" in s and "y4_5" in s).sum())
        row["dams_both_pre_periods"] = int(bins.map(lambda s: {"pre_early", "pre_late"} <= s).sum())
        out.append(row)
        yr = r.groupby("years_from_removal").agg(dams=("AR_ID", "nunique"), daily_values=("y", "size"))
        by_year.append(yr.reset_index().assign(series=series))
    return pd.DataFrame(out), pd.concat(by_year, ignore_index=True) if by_year else pd.DataFrame()


# --------------------------------------------------
# Frozen reference values (read-only)
# --------------------------------------------------

def frozen_reference(frozen_dir):
    """Before/after 10 km frozen model: dams, effect, between-dam SD; BLUP counts."""
    frozen_dir = Path(frozen_dir) if frozen_dir else None
    if frozen_dir is None or not (frozen_dir / "results.csv").exists():
        return pd.DataFrame()
    res = pd.read_csv(frozen_dir / "results.csv")
    res = res[res["model"].eq("before_after")]
    keep = ["series", "dams", "estimate", "se", "effect_sd_between_dams", "re_structure", "error"]
    out = res[[c for c in keep if c in res]].rename(columns={"dams": "frozen_dams"})
    if (frozen_dir / "dam_effects.csv").exists():
        blup = pd.read_csv(frozen_dir / "dam_effects.csv")
        blup = blup[blup["model"].eq("before_after")].groupby("series")["AR_ID"].nunique()
        out = out.merge(blup.rename("blup_dams").reset_index(), on="series", how="left")
    return out


# --------------------------------------------------
# Report
# --------------------------------------------------

def _md(frame, cols=None, floatfmt="{:.3g}"):
    f = frame[cols] if cols else frame
    if f.empty:
        return "_(none)_\n"
    header = "| " + " | ".join(map(str, f.columns)) + " |\n|" + "---|" * len(f.columns) + "\n"
    body = ""
    for _, r in f.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) and pd.notna(v)
                 else ("" if pd.isna(v) else str(v)) for v in r]
        body += "| " + " | ".join(cells) + " |\n"
    return header + body


def write_report(path, tables, recommendations):
    m = tables["measures"]
    prim = m[m["window"] == f"{PRIMARY_KM:g} km"]
    lines = [
        "# Predictive feasibility: Stage 1 data audit",
        "",
        "Read-only audit (docs/predictive_feasibility_prespec.md). No targets or models yet. Rows: frozen "
        "before/after rows (mixed_model.prepare), suspect dams excluded, dam-station pairs with days before "
        f"and after. Primary window: treated stations within {PRIMARY_KM:g} km. Estimable (provisional): "
        f"samples in >= {ESTIMABLE_MIN_YEARS} distinct years on each side.",
        "",
        "## Measures (primary window)",
        _md(prim, ["series", "dams", "stations", "daily_values", "dams_years_ge1_each_side",
                   "dams_years_ge2_each_side", "dams_years_ge3_each_side", "dams_estimable",
                   "huc2_regions", "huc2_regions_ge5_dams", "huc4_regions", "dams_one_station",
                   "meets_size_rule"]),
        f"Size rule (prespec 4): >= {MODEL_MIN_DAMS} estimable dams and >= {MIN_REGIONS} HUC2 regions with "
        f">= {MIN_REGION_DAMS} dams. Reliability (>= 0.15) is checked in Stage 2.",
        "",
        "## Same, 30 km warehouse window (comparison only)",
        _md(m[m["window"] != f"{PRIMARY_KM:g} km"], ["series", "dams", "stations", "dams_estimable",
                                                     "huc2_regions_ge5_dams", "meets_size_rule"]),
        "## Distance coverage",
        _md(tables["distance"], ["series", "station_km_min", "station_km_q25", "station_km_median",
                                 "station_km_mean", "station_km_q75", "station_km_max",
                                 "share_stations_le_2km", "nearest_station_km_median", "dams_le_10km",
                                 "dams_ge2_stations_le_10km", "dams_ge2_coverage_bands",
                                 "dams_all_3_coverage_bands", "dams_span_ge_3km", "dams_only_beyond_10km"]),
        "Coverage bands: 0-2, 2-5, 5-10 km. Station distances are one per dam-station pair within 10 km.",
        "",
        "## Time coverage (primary window)",
        _md(tables["temporal"], ["series", "dams", "dams_pre_early", "dams_pre_late", "dams_y1", "dams_y2_3",
                                 "dams_y4_5", "dams_all_post_periods", "dams_early_and_late",
                                 "rows_outside_periods"]),
        "The removal year (0) and years beyond +5 are not in the analysis table (window 5 / 5). Removal dates "
        "are known to the year only, so y1 spans about 1-24 months after removal.",
        "",
        "## Predictors",
        _md(tables["predictors"].pivot_table(index="predictor", columns="series", values="completeness")
            .reset_index()),
        f"Completeness among each measure's estimable dams; the rule is >= {MIN_COMPLETENESS:.0%}.",
        "",
        "Correlated pairs (|Spearman rho| > 0.8):",
        _md(tables["pairs"][tables["pairs"]["flag_abs_rho_gt_0_8"]] if len(tables["pairs"]) else tables["pairs"]),
        "VIF of the recommended predictors (complete cases):",
        _md(tables["vif"]),
        "## Frozen 10 km before/after model (reference, read-only)",
        _md(tables["frozen"]),
        "## Recommendation (from the prespec rules; Stage 2 still has to confirm reliability)",
        _md(recommendations),
    ]
    Path(path).write_text("\n".join(lines))


def run_audit(daily, regions, exclude_dams, out_dir, frozen_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary = treated_rows(daily, exclude_dams, PRIMARY_KM)
    window = treated_rows(daily, exclude_dams, WINDOW_KM)
    m10, per_dam = measure_audit(primary, regions, f"{PRIMARY_KM:g} km")
    m30, _ = measure_audit(window, regions, f"{WINDOW_KM:g} km")
    measures = pd.concat([m10, m30], ignore_index=True)
    features = dam_features(primary)
    pred = predictor_audit(features, per_dam)
    est_dams = per_dam.loc[per_dam["estimable"], "AR_ID"].unique()
    rho, pairs = correlations(features[features["AR_ID"].isin(est_dams)])
    distance, distance_region = distance_audit(window, regions)
    temporal, by_year = temporal_audit(primary)
    frozen = frozen_reference(frozen_dir)

    recs, vif_frames = [], []
    for _, r in m10.iterrows():
        chosen = select_predictors(pred, pairs, r["series"], int(r["dams_estimable"]))
        recs.append({"series": r["series"], "dams_estimable": int(r["dams_estimable"]),
                     "huc2_regions_ge5_dams": int(r["huc2_regions_ge5_dams"]),
                     "meets_size_rule": bool(r["meets_size_rule"]),
                     "audit_only_series": r["series"] in AUDIT_ONLY_SERIES,
                     "max_predictors": int(r["dams_estimable"]) // DAMS_PER_PREDICTOR,
                     "recommended_predictors": ", ".join(chosen)})
        if r["meets_size_rule"] and chosen:
            f = features[features["AR_ID"].isin(per_dam.loc[(per_dam["series"] == r["series"])
                                                            & per_dam["estimable"], "AR_ID"])]
            vif_frames.append(vif(f, chosen).assign(series=r["series"]))
    recommendations = pd.DataFrame(recs)
    vif_table = pd.concat(vif_frames, ignore_index=True) if vif_frames else pd.DataFrame()

    tables = {"measures": measures, "per_dam": per_dam, "regions": dams_per_region(per_dam),
              "features": features, "predictors": pred, "correlations": rho.reset_index(),
              "pairs": pairs, "vif": vif_table, "predictors_by_region": predictors_by_region(features, per_dam),
              "distance": distance, "distance_by_region": distance_region, "temporal": temporal,
              "temporal_by_year": by_year, "frozen": frozen, "recommendations": recommendations}
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"audit_{name}.csv", index=False)
    write_report(out_dir / "report_audit.md", tables, recommendations)
    return tables


# --------------------------------------------------
# Stage 2: dam-level targets (prespec Amendment 1)
# --------------------------------------------------

def frozen_standardization(stage3_dir):
    """Per series: the frozen before/after distance model's post_ld / post_ldar
    estimates and SEs, its reference distance and dilution, and its dam
    random-slope variance (var_dam_post, the between-dam variance of the
    change at the reference). Read only; their covariance is not saved."""
    stage3_dir = Path(stage3_dir)
    res = pd.read_csv(stage3_dir / "results.csv")
    coef = pd.read_csv(stage3_dir / "coefficients.csv")
    res = res[(res["spec"] == STD_SPEC) & res["error"].fillna("").eq("")]
    coef = coef[coef["spec"] == STD_SPEC]
    out = []
    for _, r in res.iterrows():
        c = coef[coef["series"] == r["series"]].set_index("term")
        if not {"post_ld", "post_ldar"} <= set(c.index):
            continue
        out.append({"series": r["series"], "b_ld": c.loc["post_ld", "estimate"],
                    "se_ld": c.loc["post_ld", "se"], "b_ldar": c.loc["post_ldar", "estimate"],
                    "se_ldar": c.loc["post_ldar", "se"],
                    "reference_km": float(r.get("reference_km", 5.0)),
                    "reference_dar": float(r["reference_dar"]),
                    "var_dam_post_30km": r.get("var_dam_post", np.nan), "frozen_dams": r["dams"],
                    "re_structure": r.get("re_structure", "")})
    return pd.DataFrame(out)


def station_terms(rows, reference_km, reference_dar):
    """One row per dam-station: distance, dilution ratio and the frozen model's
    ld = ln(1 + km) - ln(1 + reference_km), ldar = log10(ratio) - log10(reference)."""
    st = (rows.groupby(["AR_ID", "station_id"])
          .agg(km=("network_distance_km", "first"), dar=("drainage_area_ratio", "first"))
          .reset_index())
    km = pd.to_numeric(st["km"], errors="coerce").clip(lower=0)
    dar = pd.to_numeric(st["dar"], errors="coerce")
    st["ld"] = np.log1p(km) - np.log1p(reference_km)
    st["ldar"] = np.where(dar > 0, np.log10(dar.where(dar > 0)) - np.log10(reference_dar), np.nan)
    return st


def _inv_sqrt_psd(m, tol=1e-10):
    """(m)^(-1/2) for a symmetric PSD matrix; directions with eigenvalue ~0 get 0."""
    vals, vecs = np.linalg.eigh(m)
    inv = np.where(vals > tol, 1.0 / np.sqrt(np.clip(vals, tol, None)), 0.0)
    return (vecs * inv) @ vecs.T


def fit_station_changes(rows, covariates=COVARIATES):
    """The Amendment-1 regression: y = station + month + weather + delta_s * post.
    Returns (station table, V_cr2, V_cr1, adjusted values) where V_* are the
    sandwich covariances of the station deltas (station-year clusters)."""
    r = rows.dropna(subset=["y"] + covariates).copy()
    r = r[r.groupby(["AR_ID", "station_id"])["post"].transform("nunique").eq(2)]
    r["year"] = pd.to_datetime(r["sample_date"]).dt.year
    key = r["AR_ID"].astype(str) + "|" + r["station_id"].astype(str)
    st_codes, st_index = pd.factorize(key, sort=True)
    n, s = len(r), len(st_index)
    post = r["post"].to_numpy(float)
    months = sorted(r["month"].dropna().unique())[1:]           # first month is the reference
    cols = [np.eye(s)[st_codes], np.eye(s)[st_codes] * post[:, None]]
    cols.append(np.column_stack([(r["month"].to_numpy() == m).astype(float) for m in months])
                if months else np.empty((n, 0)))
    zcols = []
    for c in covariates:
        x = r[c].astype(float)
        if x.std() > 0:
            zcols.append(((x - x.mean()) / x.std()).to_numpy())
    cols.append(np.column_stack(zcols) if zcols else np.empty((n, 0)))
    X = np.hstack(cols)
    y = r["y"].to_numpy(float)
    bread = np.linalg.pinv(X.T @ X)
    beta = bread @ (X.T @ y)
    e = y - X @ beta
    p = np.linalg.matrix_rank(X)
    clusters, cl_index = pd.factorize(key + "|" + r["year"].astype(str))
    G = len(cl_index)
    XB = X @ bread
    order = np.argsort(clusters, kind="stable")
    bounds = np.flatnonzero(np.diff(np.r_[-1, clusters[order], G]))
    meat2 = np.zeros((X.shape[1], X.shape[1]))
    meat1 = np.zeros_like(meat2)
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = order[a:b]
        Xg, eg = X[idx], e[idx]
        u1 = Xg.T @ eg
        meat1 += np.outer(u1, u1)
        hgg = XB[idx] @ Xg.T
        u2 = Xg.T @ (_inv_sqrt_psd(np.eye(len(idx)) - hgg) @ eg)
        meat2 += np.outer(u2, u2)
    post_cols = np.arange(s, 2 * s)
    Bp = bread[post_cols]
    V2 = Bp @ meat2 @ Bp.T
    V1 = Bp @ meat1 @ Bp.T * (G / max(G - 1, 1)) * ((n - 1) / max(n - p, 1))

    # Station table and the station-year-means robustness check.
    adj = y - X[:, 2 * s:] @ beta[2 * s:]                      # month and weather removed
    r = r.assign(_key=key.to_numpy(), _adj=adj)
    ym = r.groupby(["_key", "post", "year"])["_adj"].mean().reset_index()
    stat = ym.groupby(["_key", "post"])["_adj"].agg(["mean", "var", "count"]).unstack("post")
    st = pd.DataFrame({"_key": st_index})
    st["AR_ID"] = st["_key"].str.split("|").str[0]
    st["station_id"] = st["_key"].str.split("|", n=1).str[1]
    st["delta"] = beta[post_cols]
    st["var_cr2"], st["var_cr1"] = np.diag(V2), np.diag(V1)
    g = r.groupby("_key")
    st["days"] = st["_key"].map(g.size())
    st["clusters"] = st["_key"].map(g["year"].nunique())
    st["pre_years"] = st["_key"].map(stat[("count", 0)])
    st["post_years"] = st["_key"].map(stat[("count", 1)])
    st["post_days_y4_5"] = st["_key"].map(g.apply(lambda d: int(d["years_from_removal"].between(4, 5).sum())
                                                  if "years_from_removal" in d else 0))
    st["post_days"] = st["_key"].map(g["post"].sum())
    st["delta_ym"] = st["_key"].map(stat[("mean", 1)] - stat[("mean", 0)])
    st["var_ym"] = st["_key"].map(stat[("var", 1)] / stat[("count", 1)] + stat[("var", 0)] / stat[("count", 0)])
    st["regression_rank_deficit"] = X.shape[1] - p
    return st, V2, V1, r


def dam_targets(st, V2, V1, std, standardize=True):
    """One row per dam: equal-weight station average, standardized with the frozen
    coefficients (std: one row of frozen_standardization), with conditional
    (CR2, CR1), coefficient and total SEs. st's index must be the station's
    position in V (as returned by fit_station_changes; a subset keeps it)."""
    out = []
    for dam, g in st.groupby("AR_ID", sort=False):
        if standardize:
            g = g[g["ldar"].notna()]
        if g.empty:
            continue
        idx, S = g.index.to_numpy(), len(g)
        w = np.full(S, 1.0 / S)
        var2 = float(w @ V2[np.ix_(idx, idx)] @ w)
        var1 = float(w @ V1[np.ix_(idx, idx)] @ w)
        mld, mldar = g["ld"].mean(), g["ldar"].mean() if standardize else 0.0
        shift = (std["b_ld"] * mld + std["b_ldar"] * mldar) if standardize else 0.0
        var_c0 = (mld ** 2 * std["se_ld"] ** 2 + mldar ** 2 * std["se_ldar"] ** 2) if standardize else 0.0
        var_cb = ((abs(mld) * std["se_ld"] + abs(mldar) * std["se_ldar"]) ** 2) if standardize else 0.0
        target = float(g["delta"].mean() - shift)
        G = int(g["clusters"].sum())
        se_total = np.sqrt(var2 + var_cb)
        t = stats.t.ppf(0.975, max(G - 1, 1))
        out.append({"AR_ID": dam, "target": target, "target_unstandardized": float(g["delta"].mean()),
                    "shift": float(shift), "se_cr2": np.sqrt(var2), "se_cr1": np.sqrt(var1),
                    "se_coef_rho0": np.sqrt(var_c0), "se_coef_bound": np.sqrt(var_cb),
                    "se_total": se_total, "se_total_rho0": np.sqrt(var2 + var_c0),
                    "ci_low": target - t * se_total, "ci_high": target + t * se_total, "df": max(G - 1, 1),
                    "stations": S, "days": int(g["days"].sum()), "clusters": G,
                    "min_station_clusters": int(g["clusters"].min()),
                    "pre_years": int(g["pre_years"].max()), "post_years": int(g["post_years"].max()),
                    "mean_km": float(g["km"].mean()), "min_km": float(g["km"].min()),
                    "max_km": float(g["km"].max()), "mean_ld": float(mld), "mean_ldar": float(mldar),
                    "share_post_days_y4_5": float(g["post_days_y4_5"].sum() / max(g["post_days"].sum(), 1)),
                    "target_ym": float(g["delta_ym"].mean() - shift),
                    "se_ym": float(np.sqrt(g["var_ym"].sum()) / S)})
    return pd.DataFrame(out)


def flag_weak(t):
    """Weak uncertainty support (Amendment 1)."""
    t = t.copy()
    med = t["se_cr2"].median()
    ratio = t["se_ym"] / t["se_cr2"]
    t["flag_few_clusters"] = t["clusters"] < WEAK_MIN_CLUSTERS
    t["flag_few_days"] = t["days"] < WEAK_MIN_DAYS
    t["flag_se_outlier"] = t["se_cr2"] > WEAK_SE_MULTIPLE * med
    t["flag_se_methods_disagree"] = ~ratio.between(*WEAK_YM_RATIO)
    t["weak_support"] = t[["flag_few_clusters", "flag_few_days", "flag_se_outlier",
                           "flag_se_methods_disagree"]].any(axis=1)
    return t


def tau2_dl(est, se):
    """DerSimonian-Laird between-unit variance (method of moments)."""
    y, v = np.asarray(est, float), np.asarray(se, float) ** 2
    w = 1 / v
    k = len(y)
    q = np.sum(w * (y - np.sum(w * y) / w.sum()) ** 2)
    c = w.sum() - np.sum(w ** 2) / w.sum()
    return max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0


def tau2_pm(est, se, tol=1e-10, max_iter=200):
    """Paule-Mandel: tau2 such that the generalized Q equals k - 1 (0 if Q(0) <= k - 1)."""
    y, v = np.asarray(est, float), np.asarray(se, float) ** 2
    k = len(y)

    def q(tau2):
        w = 1 / (v + tau2)
        return np.sum(w * (y - np.sum(w * y) / w.sum()) ** 2)

    if k < 2 or q(0.0) <= k - 1:
        return 0.0
    lo, hi = 0.0, max(np.var(y, ddof=1), 1e-12) * 10
    while q(hi) > k - 1:
        hi *= 2
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if q(mid) > k - 1 else (lo, mid)
        if hi - lo < tol * max(hi, 1e-12):
            break
    return (lo + hi) / 2


def reliability(tau2, se):
    se2 = np.asarray(se, float) ** 2
    if not np.isfinite(tau2) or len(se2) == 0:
        return np.nan, np.nan
    return tau2 / (tau2 + se2.mean()), float(np.mean(tau2 / (tau2 + se2)))


def reliability_table(targets, frozen10, std):
    """Reliability per series and target version, for each tau2 source and SE kind."""
    out = []
    for (version, series), t in targets.groupby(["version", "series"]):
        if version not in ("primary_10km", "sensitivity_30km") or len(t) < 3:
            continue
        sources = {"targets_paule_mandel": None, "targets_dersimonian_laird": None}
        if version == "primary_10km":
            sd = frozen10.set_index("series")["effect_sd_between_dams"].get(series, np.nan) \
                if len(frozen10) else np.nan
            sources["frozen_10km_mixed (sd^2)"] = sd ** 2
        sources["frozen_30km_distance (var_dam_post)"] = (
            std.set_index("series")["var_dam_post_30km"].get(series, np.nan) if len(std) else np.nan)
        for se_kind in ("se_total", "se_cr2"):
            for source, tau2 in sources.items():
                if source == "targets_paule_mandel":
                    tau2 = tau2_pm(t["target"], t[se_kind])
                elif source == "targets_dersimonian_laird":
                    tau2 = tau2_dl(t["target"], t[se_kind])
                r_mean, r_dams = reliability(tau2, t[se_kind])
                out.append({"version": version, "series": series, "dams": len(t), "tau2_source": source,
                            "se_kind": se_kind, "tau2": tau2, "tau": np.sqrt(tau2) if tau2 >= 0 else np.nan,
                            "mean_se2": float((t[se_kind] ** 2).mean()), "reliability": r_mean,
                            "mean_dam_reliability": r_dams,
                            "raw_var_targets": float(t["target"].var(ddof=1))})
    rel = pd.DataFrame(out)
    if len(rel):
        p = rel[rel["version"].eq("primary_10km")]
        stop = p.groupby("series").apply(lambda d: bool(
            (d.loc[(d["se_kind"] == "se_total") & (d["tau2_source"] == "targets_paule_mandel"),
                   "reliability"] < RELIABILITY_STOP).all()
            and (d.loc[d["tau2_source"].str.startswith("frozen"), "reliability"].dropna()
                 < RELIABILITY_STOP).all()))
        rel["stop_rule_met"] = rel["series"].map(stop).where(rel["version"].eq("primary_10km"))
    return rel


def compare_targets(targets):
    """Each sensitivity target vs the primary: dams, correlations, mean absolute difference."""
    prim = targets[targets["version"] == "primary_10km"][["series", "AR_ID", "target"]]
    out = []
    for (version, series), t in targets[targets["version"] != "primary_10km"].groupby(["version", "series"]):
        m = prim[prim["series"] == series].merge(t[["AR_ID", "target"]], on="AR_ID", suffixes=("", "_alt"))
        out.append({"version": version, "series": series, "dams_version": len(t), "dams_shared": len(m),
                    "dams_added_vs_primary": int((~t["AR_ID"].isin(prim.loc[prim["series"] == series, "AR_ID"])).sum()),
                    "pearson_r": m["target"].corr(m["target_alt"]) if len(m) > 2 else np.nan,
                    "spearman_r": m["target"].corr(m["target_alt"], method="spearman") if len(m) > 2 else np.nan,
                    "mean_abs_diff": float((m["target"] - m["target_alt"]).abs().mean()) if len(m) else np.nan,
                    "sd_primary": float(m["target"].std()) if len(m) > 1 else np.nan,
                    "mean_version": float(t["target"].mean()), "sd_version": float(t["target"].std())})
    return pd.DataFrame(out)


def build_targets(rows10, rows30, std, frozen10, blups, regions):
    """All target versions for every series in std. Returns (targets, stations, comparison, reliability)."""
    targets, stations = [], []
    for _, s in std.iterrows():
        series = s["series"]
        versions = {"primary_10km": rows10, "years_1_3": rows10[rows10["years_from_removal"].le(3)],
                    "sensitivity_30km": rows30}
        for version, rows in versions.items():
            r = rows[rows["series"] == series]
            if r.empty:
                continue
            st, V2, V1, _ = fit_station_changes(r)
            terms = station_terms(r, s["reference_km"], s["reference_dar"])
            st = st.merge(terms, on=["AR_ID", "station_id"], how="left", validate="one_to_one")
            # a left merge on unique keys keeps st's order: index = position in V
            t = dam_targets(st, V2, V1, s)
            targets.append(flag_weak(t).assign(version=version, series=series))
            stations.append(st.assign(version=version, series=series))
            if version == "primary_10km":
                u = dam_targets(st, V2, V1, s, standardize=False)
                targets.append(flag_weak(u).assign(version="unstandardized", series=series))
                nearest = st.loc[st[st["ldar"].notna()].groupby("AR_ID")["km"].idxmin()]
                n = dam_targets(nearest.sort_index(), V2, V1, s)  # index = positions in V
                targets.append(flag_weak(n).assign(version="nearest_station", series=series))
                excluded = st[st["ldar"].isna()]
                print(f"  {series}: {t.shape[0]} dams; {len(excluded)} stations without a dilution ratio "
                      f"left out of the standardized target")
        if len(blups):
            b = blups[(blups["series"] == series) & blups["model"].eq("before_after")]
            targets.append(pd.DataFrame({"AR_ID": b["AR_ID"], "target": b["dam_effect"],
                                         "version": "frozen_blup_10km", "series": series}))
    targets = pd.concat(targets, ignore_index=True)
    targets = add_regions(targets, regions)
    stations = pd.concat(stations, ignore_index=True)
    return targets, stations, compare_targets(targets), reliability_table(targets, frozen10, std)


def write_target_report(path, targets, std, comparison, rel):
    prim = targets[targets["version"] == "primary_10km"]
    summ = []
    for series, t in prim.groupby("series"):
        summ.append({"series": series, "dams": len(t), "huc2": t["huc2"].nunique(),
                     "target_mean": t["target"].mean(), "target_sd": t["target"].std(),
                     "target_q10": t["target"].quantile(0.1), "target_q90": t["target"].quantile(0.9),
                     "se_cr2_median": t["se_cr2"].median(), "se_cr1_median": t["se_cr1"].median(),
                     "se_total_median": t["se_total"].median(), "se_total_q90": t["se_total"].quantile(0.9),
                     "se_coef_bound_median": t["se_coef_bound"].median(),
                     "clusters_median": t["clusters"].median(), "clusters_min": t["clusters"].min(),
                     "weak_support": int(t["weak_support"].sum()),
                     "ym_vs_cr2_se_ratio_median": (t["se_ym"] / t["se_cr2"]).median(),
                     "corr_target_ym": t["target"].corr(t["target_ym"])})
    s30 = targets[targets["version"] == "sensitivity_30km"]
    add = []
    for series, t in s30.groupby("series"):
        p = prim[prim["series"] == series]
        add.append({"series": series, "dams_30km": len(t), "dams_10km": len(p),
                    "dams_added": int((~t["AR_ID"].isin(p["AR_ID"])).sum()),
                    "huc2_30km": t["huc2"].nunique(), "huc2_10km": p["huc2"].nunique(),
                    "huc2_added": len(set(t["huc2"].dropna()) - set(p["huc2"].dropna())),
                    "sd_30km": t["target"].std(), "sd_10km": p["target"].std()})
    rel_show = rel[rel["se_kind"].eq("se_total")] if len(rel) else rel
    lines = [
        "# Predictive feasibility: Stage 2 targets",
        "",
        "Dam-level targets (prespec Amendment 1). Model units: log10 for specific conductance, total "
        "phosphorus and turbidity. No predictive model has been fit.",
        "",
        "## Frozen standardization coefficients (before/after distance model, read only)",
        _md(std),
        "## Primary targets (10 km, standardized to 5 km and the reference dilution, years 1-5)",
        _md(pd.DataFrame(summ)),
        "## Reliability (total SE; conditional-SE rows in targets_reliability.csv)",
        _md(rel_show, ["version", "series", "dams", "tau2_source", "tau", "mean_se2", "reliability",
                       "mean_dam_reliability", "stop_rule_met"]),
        "## Sensitivity targets vs primary",
        _md(comparison),
        "## 30 km window: added dams and regions",
        _md(pd.DataFrame(add)),
    ]
    Path(path).write_text("\n".join(lines))


def run_targets(daily, regions, exclude_dams, out_dir, stage3_dir, frozen_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    std = frozen_standardization(stage3_dir)
    std = std[std["series"].isin(TARGET_SERIES)].reset_index(drop=True)
    frozen10 = frozen_reference(frozen_dir)
    blup_path = Path(frozen_dir) / "dam_effects.csv" if frozen_dir else None
    blups = pd.read_csv(blup_path) if blup_path and blup_path.exists() else pd.DataFrame()
    rows10 = treated_rows(daily, exclude_dams, PRIMARY_KM)
    rows30 = treated_rows(daily, exclude_dams, WINDOW_KM)
    targets, stations, comparison, rel = build_targets(rows10, rows30, std, frozen10, blups, regions)
    std.to_csv(out_dir / "targets_standardization.csv", index=False)
    targets.to_csv(out_dir / "targets.csv", index=False)
    stations.to_csv(out_dir / "targets_stations.csv", index=False)
    comparison.to_csv(out_dir / "targets_comparison.csv", index=False)
    rel.to_csv(out_dir / "targets_reliability.csv", index=False)
    write_target_report(out_dir / "report_targets.md", targets, std, comparison, rel)
    return targets, stations, comparison, rel


# --------------------------------------------------
# Stage 3: cross-validated prediction (prespec 7-8, Amendments 1-2)
# --------------------------------------------------

CV_PREDICTORS = [p for p in PREDICTORS if p != "log_reservoir_volume_proxy"]   # Amendment 1.5
MISSING_INDICATOR_SHARE = 0.10
LAMBDA_GRID = np.logspace(-2, 3, 11)
HUC4_SPLITS, HUC4_REPEATS = 5, 20
N_BOOT = 2000
SEED = 20261111
RF_PARAMS = {"n_estimators": 300, "max_depth": 3, "min_samples_leaf": 10}
R2_MIN, MAE_GAIN, MIN_FOLD_DAMS, AUC_MIN_CLASS = 0.05, 0.05, 10, 15
SMALL_REGIONS = "small_regions"
MODELS = ("A", "B", "C", "D")
INFLUENCE_SERIES = ["specific_conductance"]                 # Amendment 2b
TIMING_SERIES = ["phosphorus (Total)", "turbidity"]         # Amendment 2c (prespecified comparison)
# Prespec 7 and Amendment 2: name -> (target version, options). "primary" decides go / no-go.
CV_ANALYSES = {
    "primary": ("primary_10km", {}),
    "weighted": ("primary_10km", {"weighted": True}),
    "complete_cases": ("primary_10km", {"complete": True}),
    "removal_year": ("primary_10km", {"extra": ("removal_year",)}),
    "exclude_weak_support": ("primary_10km", {"exclude_weak": True}),
    "years_1_3": ("years_1_3", {}),
    "window_30km": ("sensitivity_30km", {"features": "30km"}),
}


def _rows(frame, mask):
    return frame.iloc[np.flatnonzero(np.asarray(mask))]


def region_folds(huc2, min_dams=MIN_REGION_DAMS):
    """Leave-one-HUC2-out labels; regions with < min_dams dams share one fold."""
    h = pd.Series(np.asarray(huc2), dtype="object").astype(str)
    counts = h.value_counts()
    small = set(counts[counts < min_dams].index)
    return np.where(h.isin(small), SMALL_REGIONS, h).astype(object)


def grouped_kfold(groups, n_splits, rng):
    """Whole groups to n_splits folds: groups shuffled, each to the fold with the fewest dams."""
    g = pd.Series(np.asarray(groups), dtype="object").astype(str)
    sizes = g.value_counts()
    load, fold_of = np.zeros(n_splits), {}
    for grp in rng.permutation(sizes.index.to_numpy()):
        k = int(np.argmin(load))
        fold_of[grp] = k
        load[k] += sizes[grp]
    return g.map(fold_of).to_numpy()


def fold_predictors(F, candidates=CV_PREDICTORS):
    """Prespec 5 + Amendments 1-2 on one training set (features only, never the
    target): >= 70% complete, skip if |rho| > 0.8 with one already chosen,
    indicator if >= 10% missing, cap floor(n / 15) counting indicators,
    priority order, stop at the first that doesn't fit. Returns (predictors, indicators)."""
    cap = len(F) // DAMS_PER_PREDICTOR
    ok = [p for p in candidates if p in F and F[p].notna().mean() >= MIN_COMPLETENESS
          and F[p].nunique(dropna=True) > 1]
    rho = F[ok].corr(method="spearman", min_periods=10) if ok else pd.DataFrame()
    chosen, indicators, used = [], [], 0
    for p in ok:
        if any(pd.notna(rho.loc[p, q]) and abs(rho.loc[p, q]) > MAX_ABS_RHO for q in chosen):
            continue
        ind = F[p].isna().mean() >= MISSING_INDICATOR_SHARE
        if used + 1 + int(ind) > cap:
            break
        chosen.append(p)
        indicators += [p] if ind else []
        used += 1 + int(ind)
    return chosen, indicators


class Design:
    """Median imputation + missing indicators + standardization, fitted on one training set."""

    def __init__(self, predictors, indicators=(), extra=()):
        self.cols = list(predictors) + [c for c in extra if c not in predictors]
        self.indicators = list(indicators)

    def _raw(self, F):
        x = F[self.cols].astype(float).fillna(self.median)
        ind = [F[p].isna().astype(float).rename(f"{p}_missing") for p in self.indicators]
        return pd.concat([x] + ind, axis=1).to_numpy(float)

    def fit(self, F):
        self.median = F[self.cols].astype(float).median()
        x = self._raw(F)
        self.mu, sd = x.mean(axis=0), x.std(axis=0)
        self.sd = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, F):
        return (self._raw(F) - self.mu) / self.sd


def ridge_fit(X, y, lam, w=None):
    """Weighted ridge with an unpenalized intercept; returns [intercept, coefficients]."""
    n, p = X.shape
    w = np.ones(n) if w is None else np.asarray(w, float)
    A = np.column_stack([np.ones(n), X])
    pen = np.eye(p + 1) * lam
    pen[0, 0] = 0.0
    AW = A.T * w
    return np.linalg.solve(AW @ A + pen, AW @ y)


def ridge_predict(coef, X):
    return coef[0] + X @ coef[1:]


def choose_lambda(F, y, w, design_args, groups):
    """Penalty with the lowest pooled inner-CV squared error; ties go to the larger penalty."""
    sse = np.zeros(len(LAMBDA_GRID))
    for g in pd.unique(groups):
        tr, te = groups != g, groups == g
        d = Design(*design_args).fit(_rows(F, tr))
        Xtr, Xte = d.transform(_rows(F, tr)), d.transform(_rows(F, te))
        for j, lam in enumerate(LAMBDA_GRID):
            sse[j] += np.sum((y[te] - ridge_predict(ridge_fit(Xtr, y[tr], lam, w[tr]), Xte)) ** 2)
    return float(LAMBDA_GRID[len(sse) - 1 - int(np.argmin(sse[::-1]))])


def training_weights(D):
    """1 / (tau2 + SE^2), tau2 by Paule-Mandel on these targets; scaled to mean 1."""
    tau2 = tau2_pm(D["target"], D["se_total"])
    w = 1.0 / (tau2 + D["se_total"].to_numpy(float) ** 2)
    return w / w.mean()


def cross_validate(data, folds, inner, rng, models=MODELS, weighted=False, extra=()):
    """Out-of-sample predictions, one row per dam. data: one row per dam with
    AR_ID, target, se_total, huc2, huc4 and the features. inner: "region"
    (inner leave-one-region-out) or "huc4" (inner grouped 5-fold)."""
    from sklearn.ensemble import RandomForestRegressor

    y_all = data["target"].to_numpy(float)
    folds = np.asarray(folds)
    out = []
    for f in pd.unique(folds):
        te = folds == f
        D_tr, D_te = _rows(data, ~te), _rows(data, te)
        y = y_all[~te]
        w = training_weights(D_tr) if weighted else np.ones(len(y))
        mean = float(np.average(y, weights=w))
        rec = pd.DataFrame({"AR_ID": D_te["AR_ID"].to_numpy(), "huc2": D_te["huc2"].to_numpy(),
                            "fold": str(f), "y": y_all[te], "pred_A": mean})
        if "B" in models:
            h = D_tr["huc2"].to_numpy()
            region = pd.Series(y * w).groupby(h).sum() / pd.Series(w).groupby(h).sum()
            rec["pred_B"] = D_te["huc2"].map(region).fillna(mean).to_numpy()
        preds, inds = fold_predictors(D_tr)
        args = (preds, inds, extra)
        rec["n_columns"] = len(Design(*args).cols) + len(inds)
        rec["predictors"] = ";".join(Design(*args).cols + [f"{p}_missing" for p in inds])
        rec["lambda"] = np.nan
        has_x = bool(Design(*args).cols)
        if "C" in models:
            if has_x:
                g = (region_folds(D_tr["huc2"]) if inner == "region"
                     else grouped_kfold(D_tr["huc4"], HUC4_SPLITS, rng))
                if len(pd.unique(g)) < 2:
                    g = grouped_kfold(D_tr["huc4"], HUC4_SPLITS, rng)
                lam = choose_lambda(D_tr, y, w, args, g)
                d = Design(*args).fit(D_tr)
                rec["pred_C"] = ridge_predict(ridge_fit(d.transform(D_tr), y, lam, w), d.transform(D_te))
                rec["lambda"] = lam
            else:
                rec["pred_C"] = mean
        if "D" in models:
            if has_x:
                d = Design(*args).fit(D_tr)
                rf = RandomForestRegressor(**RF_PARAMS, random_state=SEED, n_jobs=1)
                rf.fit(d.transform(D_tr), y, sample_weight=w)
                rec["pred_D"] = rf.predict(d.transform(D_te))
            else:
                rec["pred_D"] = mean
        out.append(rec)
    return pd.concat(out, ignore_index=True)


def _r2(y, pred, pred_a):
    sse_a = np.sum((y - pred_a) ** 2)
    return 1.0 - np.sum((y - pred) ** 2) / sse_a if sse_a > 0 else np.nan


def _auc(y, pred, threshold):
    cls = np.abs(y) > threshold
    if not np.isfinite(threshold) or min(cls.sum(), (~cls).sum()) < AUC_MIN_CLASS:
        return np.nan
    return float(stats.mannwhitneyu(np.abs(pred[cls]), np.abs(pred[~cls])).statistic
                 / (cls.sum() * (~cls).sum()))


def score(pred, models, threshold=np.nan):
    """Pooled out-of-sample metrics; R2 = 1 - SSE / SSE of model A (training-fold means)."""
    y, a = pred["y"].to_numpy(float), pred["pred_A"].to_numpy(float)
    out = []
    for m in models:
        p = pred[f"pred_{m}"].to_numpy(float)
        e = y - p
        out.append({"model": m, "dams": len(y), "r2": _r2(y, p, a), "mae": float(np.mean(np.abs(e))),
                    "rmse": float(np.sqrt(np.mean(e ** 2))),
                    "mae_vs_A": float(np.mean(np.abs(e)) / np.mean(np.abs(y - a))) if np.any(y != a) else np.nan,
                    "direction_accuracy": float(np.mean(np.sign(p) == np.sign(y))),
                    "auc_beyond_sesoi": _auc(y, p, threshold)})
    return pd.DataFrame(out)


def fold_table(pred, models):
    """Per fold: dams and mean squared error of each model, and whether it beats A."""
    out = []
    for f, g in pred.groupby("fold", sort=False):
        row = {"fold": f, "dams": len(g)}
        for m in models:
            row[f"mse_{m}"] = float(np.mean((g["y"] - g[f"pred_{m}"]) ** 2))
        for m in models:
            if m != "A":
                row[f"beats_A_{m}"] = row[f"mse_{m}"] < row["mse_A"]
        out.append(row)
    return pd.DataFrame(out)


def cluster_bootstrap(pred, models, rng, n_boot=N_BOOT):
    """R2 percentiles from resampling HUC2 regions (with replacement) of the
    out-of-sample predictions; approximate with few regions."""
    groups = [np.flatnonzero(pred["huc2"].to_numpy() == g) for g in pd.unique(pred["huc2"])]
    y, a = pred["y"].to_numpy(float), pred["pred_A"].to_numpy(float)
    draws = {m: np.empty(n_boot) for m in models}
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        for m in models:
            draws[m][b] = _r2(y[idx], pred[f"pred_{m}"].to_numpy(float)[idx], a[idx])
    return pd.DataFrame([{"model": m, "r2_boot_q05": np.nanpercentile(d, 5),
                          "r2_boot_q95": np.nanpercentile(d, 95),
                          "r2_boot_q025": np.nanpercentile(d, 2.5),
                          "r2_boot_q975": np.nanpercentile(d, 97.5)} for m, d in draws.items()])


def evaluate(data, rng, threshold=np.nan, models=MODELS, weighted=False, extra=(), huc4=True, boot=True):
    """Primary leave-one-region-out scheme (+ bootstrap) and repeated HUC4 scheme.
    Returns (loro predictions, metrics, folds, huc4 per-repeat metrics)."""
    loro = cross_validate(data, region_folds(data["huc2"]), "region", rng, models, weighted, extra)
    lm = [m for m in models if m != "B"]                         # B equals A under leave-one-region-out
    metrics = score(loro, lm, threshold).assign(scheme="loro")
    if boot:
        metrics = metrics.merge(cluster_bootstrap(loro, lm, rng), on="model", how="left")
    folds = fold_table(loro, lm)
    reps = []
    if huc4:
        for r in range(HUC4_REPEATS):
            p = cross_validate(data, grouped_kfold(data["huc4"], HUC4_SPLITS, rng), "huc4", rng,
                               models, weighted, extra)
            reps.append(score(p, models, threshold).assign(repeat=r))
        reps = pd.concat(reps, ignore_index=True)
        h = (reps.groupby("model", sort=False)
             .agg(dams=("dams", "first"), r2=("r2", "mean"), r2_sd_repeats=("r2", "std"),
                  mae=("mae", "mean"), rmse=("rmse", "mean"), mae_vs_A=("mae_vs_A", "mean"),
                  direction_accuracy=("direction_accuracy", "mean"),
                  auc_beyond_sesoi=("auc_beyond_sesoi", "mean"))
             .reset_index().assign(scheme="huc4_repeated"))
        metrics = pd.concat([metrics, h], ignore_index=True)
    else:
        reps = pd.DataFrame()
    return loro, metrics, folds, reps


def decide(metrics, folds):
    """Prespec 8 go / no-go for models C (primary) and D (exploratory)."""
    lo = metrics[metrics["scheme"] == "loro"].set_index("model")
    h4 = metrics[metrics["scheme"] == "huc4_repeated"].set_index("model")
    big = folds[folds["dams"] >= MIN_FOLD_DAMS]
    out = []
    for m in ("C", "D"):
        if m not in lo.index:
            continue
        r2 = lo.loc[m, "r2"]
        c = {"c1_r2_ge_0_05": bool(np.isfinite(r2) and r2 > 0 and r2 >= R2_MIN),
             "c2_beats_region_baseline": bool(m in h4.index and "B" in h4.index
                                              and h4.loc[m, "r2"] > h4.loc["B", "r2"]),
             "c3_beats_A_half_of_folds": bool(len(big) and big[f"beats_A_{m}"].mean() >= 0.5),
             "c4_boot90_excludes_0": bool(lo.loc[m].get("r2_boot_q05", np.nan) > 0),
             "c5_mae_5pct_lower": bool(lo.loc[m, "mae_vs_A"] <= 1 - MAE_GAIN)}
        out.append({"model": m, "role": "primary" if m == "C" else "exploratory",
                    "r2_loro": r2, "r2_huc4": h4["r2"].get(m, np.nan), "r2_huc4_B": h4["r2"].get("B", np.nan),
                    "folds_ge10_dams": len(big),
                    "share_folds_beating_A": float(big[f"beats_A_{m}"].mean()) if len(big) else np.nan,
                    "r2_boot_q05": lo.loc[m].get("r2_boot_q05", np.nan), "mae_vs_A": lo.loc[m, "mae_vs_A"],
                    **c, "go": all(c.values())})
    return pd.DataFrame(out)


def model_data(targets, features, series, version, exclude_weak=False, complete=False):
    """One row per dam for one series and target version, with its features.
    Dams without a region are dropped (and counted)."""
    t = targets[(targets["version"] == version) & (targets["series"] == series)]
    cols = ["AR_ID", "target", "se_total", "weak_support", "huc2", "huc4"]
    d = t[cols].merge(features, on="AR_ID", how="left")
    no_region = int(d["huc2"].isna().sum())
    d = d[d["huc2"].notna() & d["target"].notna()]
    if exclude_weak:
        d = d[~d["weak_support"].fillna(False).astype(bool)]
    if complete:
        sel, _ = fold_predictors(d)
        d = d.dropna(subset=sel)
    return d.reset_index(drop=True), no_region


def selection_table(loro, data, analysis, series):
    """How often each column was used across folds, and the full-sample selection."""
    full, inds = fold_predictors(data)
    full_cols = full + [f"{p}_missing" for p in inds]
    per_fold = loro.drop_duplicates("fold")["predictors"].fillna("")
    n = len(per_fold)
    used = pd.Series([c for s in per_fold for c in s.split(";") if c]).value_counts()
    names = list(dict.fromkeys(full_cols + list(used.index)))
    return pd.DataFrame([{"analysis": analysis, "series": series, "column": c,
                          "share_of_folds": used.get(c, 0) / n if n else np.nan,
                          "full_sample_selection": c in full_cols} for c in names])


def target_influence(targets, series, versions=("nearest_station", "frozen_blup_10km")):
    """Amendment 2b: per dam, the change in Pearson r (primary vs another
    version) when that dam is left out. Positive = the dam lowers r."""
    keep = ["AR_ID", "target", "stations", "mean_km", "min_km", "max_km", "days", "clusters",
            "se_total", "weak_support", "huc2"]
    prim = targets[(targets["version"] == "primary_10km") & (targets["series"] == series)]
    prim = prim[[c for c in keep if c in prim]]
    out = []
    for v in versions:
        alt = targets[(targets["version"] == v) & (targets["series"] == series)][["AR_ID", "target"]]
        m = prim.merge(alt.rename(columns={"target": "target_alt"}), on="AR_ID").reset_index(drop=True)
        if len(m) < 4:
            continue
        r_all = m["target"].corr(m["target_alt"])
        for i in range(len(m)):
            rest = m.drop(index=i)
            out.append({**m.loc[i].to_dict(), "version": v, "series": series, "pearson_all": r_all,
                        "pearson_without": rest["target"].corr(rest["target_alt"]),
                        "spearman_all": m["target"].corr(m["target_alt"], method="spearman")})
    out = pd.DataFrame(out)
    if len(out):
        out["r_gain_if_left_out"] = out["pearson_without"] - out["pearson_all"]
        out = out.sort_values(["version", "r_gain_if_left_out"], ascending=[True, False])
    return out


def cv_influence(data, rng_seed, threshold=np.nan):
    """Amendment 2b: leave-one-dam-out change in the leave-one-region-out R2 of model C."""
    rng = np.random.default_rng(rng_seed)
    base = score(cross_validate(data, region_folds(data["huc2"]), "region", rng, ("A", "C")),
                 ["C"], threshold)["r2"].iloc[0]
    out = []
    for i in range(len(data)):
        d = data.drop(index=i).reset_index(drop=True)
        rng = np.random.default_rng(rng_seed)
        r2 = score(cross_validate(d, region_folds(d["huc2"]), "region", rng, ("A", "C")),
                   ["C"], threshold)["r2"].iloc[0]
        out.append({"AR_ID": data.loc[i, "AR_ID"], "huc2": data.loc[i, "huc2"],
                    "target": data.loc[i, "target"], "r2_C_all": base, "r2_C_without": r2,
                    "r2_change_if_left_out": r2 - base})
    return pd.DataFrame(out).sort_values("r2_change_if_left_out", key=np.abs, ascending=False)


def _sesoi(series):
    from analysis.stage4_report import SESOI
    return float(SESOI.get(series, np.nan))


def _natural_mae(series, mae):
    """MAE in natural units; log10 measures as a typical % change."""
    if series.split(" (")[0] in mm.SETTINGS["log_parameters"]:
        return f"{(10 ** mae - 1) * 100:.1f}%"
    return f"{mae:.3g} {mm.unit_label(series, False)}".strip()


def run_cv_stage(targets, features, analyses=CV_ANALYSES, series_list=TARGET_SERIES, influence=True):
    """All analyses x series. Returns a dict of tables."""
    preds, metrics, folds, reps, decisions, selection, sizes = [], [], [], [], [], [], []
    for a_i, (analysis, (version, opt)) in enumerate(analyses.items()):
        feats = features[opt.get("features", "10km")]
        for s_i, series in enumerate(series_list):
            data, no_region = model_data(targets, feats, series, version,
                                         opt.get("exclude_weak", False), opt.get("complete", False))
            regions = data["huc2"].value_counts()
            sizes.append({"analysis": analysis, "series": series, "version": version, "dams": len(data),
                          "dams_no_region": no_region, "huc2_regions": len(regions),
                          "huc2_regions_ge5_dams": int(regions.ge(MIN_REGION_DAMS).sum()),
                          "loro_folds": len(pd.unique(region_folds(data["huc2"]))) if len(data) else 0})
            if len(data) < 2 * DAMS_PER_PREDICTOR or len(regions) < 2:
                continue
            rng = np.random.default_rng([SEED, s_i])                 # same seed per series in every analysis
            thr = _sesoi(series)
            loro, m, f, r = evaluate(data, rng, thr, weighted=opt.get("weighted", False),
                                     extra=opt.get("extra", ()))
            key = {"analysis": analysis, "series": series}
            preds.append(loro.assign(**key))
            metrics.append(m.assign(**key, mae_natural=m["mae"].map(lambda v: _natural_mae(series, v)),
                                    sesoi=thr))
            folds.append(f.assign(**key))
            if len(r):
                reps.append(r.assign(**key))
            decisions.append(decide(m, f).assign(**key))
            selection.append(selection_table(loro, data, analysis, series))
            print(f"  {analysis:22s} {series:22s} dams={len(data):4d}  "
                  f"R2 LORO C={m.loc[(m.scheme == 'loro') & (m.model == 'C'), 'r2'].iloc[0]:+.3f}")
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    tables = {"cv_sizes": pd.DataFrame(sizes), "cv_predictions": cat(preds), "cv_metrics": cat(metrics),
              "cv_folds": cat(folds), "cv_huc4_repeats": cat(reps), "cv_decisions": cat(decisions),
              "cv_selection": cat(selection)}
    if influence and "primary" in analyses:
        ti, ci = [], []
        for series in INFLUENCE_SERIES:
            ti.append(target_influence(targets, series))
            data, _ = model_data(targets, features["10km"], series, "primary_10km")
            if len(data) >= 2 * DAMS_PER_PREDICTOR:
                ci.append(cv_influence(data, [SEED, series_list.index(series)], _sesoi(series))
                          .assign(series=series))
        tables["cv_influence_targets"] = cat(ti)
        tables["cv_influence_r2"] = cat(ci)
    return tables


def write_cv_report(path, tables):
    m, d, s = tables["cv_metrics"], tables["cv_decisions"], tables["cv_sizes"]
    prim_m = m[m["analysis"] == "primary"] if len(m) else m
    show = ["series", "scheme", "model", "dams", "r2", "r2_boot_q05", "r2_boot_q95", "r2_sd_repeats",
            "mae", "mae_natural", "mae_vs_A", "rmse", "direction_accuracy", "auc_beyond_sesoi"]
    crit = ["series", "model", "role", "r2_loro", "r2_huc4", "r2_huc4_B", "share_folds_beating_A",
            "folds_ge10_dams", "r2_boot_q05", "mae_vs_A", "c1_r2_ge_0_05", "c2_beats_region_baseline",
            "c3_beats_A_half_of_folds", "c4_boot90_excludes_0", "c5_mae_5pct_lower", "go"]
    sens = (d[d["model"] == "C"].pivot_table(index="series", columns="analysis", values="r2_loro")
            .reset_index() if len(d) else pd.DataFrame())
    sens_go = (d[d["model"] == "C"].assign(go=d["go"].astype(int))
               .pivot_table(index="series", columns="analysis", values="go").reset_index() if len(d) else d)
    sel = tables["cv_selection"]
    sel = sel[sel["analysis"] == "primary"] if len(sel) else sel
    lines = [
        "# Predictive feasibility: Stage 3 cross-validation",
        "",
        f"Prespec sections 7-8 with Amendments 1-2. Seed {SEED}; ridge penalty grid 1e-2 to 1e3 "
        f"({len(LAMBDA_GRID)} values); HUC4 scheme {HUC4_SPLITS}-fold x {HUC4_REPEATS}; cluster bootstrap "
        f"{N_BOOT} draws over HUC2 regions (approximate with few regions). R2 = 1 - SSE / SSE of model A "
        "(training-fold means), on pooled out-of-sample predictions. Model B equals A under "
        "leave-one-region-out, so it is assessed under the HUC4 scheme. Units: the target's (log10 for "
        "conductance, phosphorus, turbidity; mae_natural as a % change). Prediction is association, not "
        "causation; the target is the average change over years 1-5.",
        "",
        "## Dams per analysis",
        _md(s),
        "## Primary metrics",
        _md(prim_m, [c for c in show if c in prim_m]),
        "## Go / no-go (primary analysis; model D exploratory)",
        _md(d[d["analysis"] == "primary"] if len(d) else d, [c for c in crit if c in d] if len(d) else None),
        "With 6 measures, one passing alone is reported cautiously (multiple comparisons).",
        "",
        "## Sensitivity analyses: leave-one-region-out R2 of model C",
        _md(sens),
        "Go flags of model C per analysis (1 = all five criteria met; sensitivity analyses don't change "
        "the decision):",
        _md(sens_go),
        f"Timing (Amendment 2c): the prespecified comparison is years_1_3 vs primary for "
        f"{', '.join(TIMING_SERIES)}.",
        "",
        "## Predictor selection (primary; share of leave-one-region-out folds)",
        _md(sel),
    ]
    if len(tables.get("cv_influence_targets", [])):
        ti = tables["cv_influence_targets"]
        lines += ["## Conductance: dams driving the Pearson / Spearman gap (Amendment 2b; top 5 per version)",
                  _md(ti.groupby("version").head(5),
                      [c for c in ["version", "AR_ID", "target", "target_alt", "stations", "min_km",
                                   "max_km", "days", "weak_support", "pearson_all", "pearson_without",
                                   "spearman_all"] if c in ti])]
    if len(tables.get("cv_influence_r2", [])):
        lines += ["## Conductance: leave-one-dam-out change in R2 of model C (top 10 by size)",
                  _md(tables["cv_influence_r2"].head(10)),
                  "No dam is removed from the primary analysis unless a data-quality problem is found "
                  "(Amendment 2b)."]
    Path(path).write_text("\n".join(lines))


def run_cv(daily, regions, exclude_dams, out_dir):
    """Stage 3: reads the Stage 2 targets.csv; features from the treated rows
    (10 km, and 30 km for the 30 km sensitivity)."""
    out_dir = Path(out_dir)
    path = out_dir / "targets.csv"
    if not path.exists():
        raise SystemExit(f"Run STAGE=targets first: {path} not found")
    targets = pd.read_csv(path, dtype={"AR_ID": str, "huc2": str, "huc4": str})
    targets["weak_support"] = targets["weak_support"].map({True: True, False: False, "True": True,
                                                           "False": False})
    features = {"10km": dam_features(treated_rows(daily, exclude_dams, PRIMARY_KM)),
                "30km": dam_features(treated_rows(daily, exclude_dams, WINDOW_KM))}
    tables = run_cv_stage(targets, features)
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    write_cv_report(out_dir / "report_cv.md", tables)
    return tables


def first_run_dir(base, run):
    for re_mode in ("auto", "slopes", "intercepts"):
        d = Path(base) / f"{run}__{re_mode}"
        if (d / "results.csv").exists():
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["audit", "targets", "cv"])
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--input", type=Path, help="analysis_measurement .parquet (default: warehouse).")
    parser.add_argument("--output-dir", type=Path, help="Default: data/processed/predictive/<case>/")
    parser.add_argument("--exclude-dams", type=Path,
                        default=get_project_root() / "config" / "suspect_dam_matches.csv")
    args = parser.parse_args()

    from etl.common.config import load_config
    from etl.common.warehouse import warehouse_dir
    wh = warehouse_dir(load_config(args.case_study))
    args.input = args.input or wh / "analysis_measurement.parquet"
    out_dir = args.output_dir or output_dir_for(args.case_study)
    frozen_dir = mm.output_dir_for(args.case_study)
    exclude = (pd.read_csv(args.exclude_dams, dtype=str)["AR_ID"].str.strip().dropna().tolist()
               if args.exclude_dams and Path(args.exclude_dams).exists() else [])
    print(f"Case study: {args.case_study}\nInput: {args.input}\nOutput: {out_dir}\n"
          f"Frozen models (read only): {frozen_dir}\nSuspect dams excluded: {len(exclude)}")

    settings = audit_settings(TARGET_SERIES if args.stage == "targets" else None)
    daily = mm.prepare(mm.read_measurements(args.input, settings), settings)
    regions = pd.read_parquet(wh / "dam_removal.parquet", columns=["AR_ID", "HUC8"])
    print(f"Daily values (all roles): {len(daily):,}")
    if args.stage == "cv":
        tables = run_cv(daily, regions, exclude, out_dir)
        d = tables["cv_decisions"]
        print(d[d["analysis"] == "primary"].to_string(index=False))
        print(f"\nWrote cv_*.csv and report_cv.md to {out_dir}")
        return
    if args.stage == "targets":
        stage3_dir = first_run_dir(frozen_dir / "stage3", "primary")
        if stage3_dir is None:
            raise SystemExit(f"No frozen stage3/primary__* outputs under {frozen_dir}")
        print(f"Frozen distance model (standardization, read only): {stage3_dir}")
        _, _, comparison, rel = run_targets(daily, regions, exclude, out_dir, stage3_dir, frozen_dir)
        print(rel[rel["se_kind"].eq("se_total")].to_string(index=False))
        print(f"\nWrote targets*.csv and report_targets.md to {out_dir}")
        return
    tables = run_audit(daily, regions, exclude, out_dir, frozen_dir)
    print("\nRecommendation:")
    print(tables["recommendations"].to_string(index=False))
    print(f"\nWrote audit_*.csv and report_audit.md to {out_dir}")


if __name__ == "__main__":
    main()
