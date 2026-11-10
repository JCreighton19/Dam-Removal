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
     "reservoir_polygon_timing", "precip_mm"]
    + [c for c, _ in PREDICTORS.values()]))


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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["audit"])
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

    settings = audit_settings()
    daily = mm.prepare(mm.read_measurements(args.input, settings), settings)
    regions = pd.read_parquet(wh / "dam_removal.parquet", columns=["AR_ID", "HUC8"])
    print(f"Daily values (all roles): {len(daily):,}")
    tables = run_audit(daily, regions, exclude, out_dir, frozen_dir)
    print("\nRecommendation:")
    print(tables["recommendations"].to_string(index=False))
    print(f"\nWrote audit_*.csv and report_audit.md to {out_dir}")


if __name__ == "__main__":
    main()
