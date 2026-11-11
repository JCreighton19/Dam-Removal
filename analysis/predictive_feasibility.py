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


def first_run_dir(base, run):
    for re_mode in ("auto", "slopes", "intercepts"):
        d = Path(base) / f"{run}__{re_mode}"
        if (d / "results.csv").exists():
            return d
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["audit", "targets"])
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
