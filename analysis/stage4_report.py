"""
Stage 4 report: consolidated tables and figures from the Stage 3-4 model runs,
for the capstone report and the Tableau dashboard. Reads
data/processed/mixed_model/<case_study>/stage3/<run>__<re>/ (written by
data/slurm/fit_distance_models.slurm) and stage4/influence/ (influence.py);
writes stage4/report/. Read-only on the warehouse. Fits nothing.

Usage (cluster: the last step of data/slurm/fit_stage4.slurm; locally on
copied outputs):
    python analysis/stage4_report.py
    python analysis/stage4_report.py --model-dir <copy of mixed_model/all_removals>

Outputs (stage4/report/)
------------------------
model_summary.csv        every fit of every run: counts, random-effects
                         choice and diagnostics, the effect at the reference
                         (5 km) and the distance / size slope with 95% CIs,
                         minimum detectable effects, and a classification
effects_by_distance.csv  tidy: effect and CI at 1-30 km (and reservoir
                         lengths / size percentiles) per run, series, spec;
                         the Tableau source
effects_curve.csv        the same on a fine distance grid inside each fit's
                         observed range, from the fitted coefficients and
                         their covariance (for curves with CI ribbons)
sensitivity_table.csv    BACI and before/after distance effects at 1, 5, 10
                         and 20 km across the primary and sensitivity runs
distance_comparison.csv  raw vs reservoir-length-scaled distance on the same
                         dams and days (common_sample run, ML): AIC / BIC
influence_summary.csv    leave-one-dam-out and bootstrap, per series
sesoi_thresholds.csv     the practical-significance thresholds used, in model
                         and plain units
sesoi_sensitivity.csv    primary estimates classified under 0.5x, 0.75x, 1x,
                         1.5x and 2x the thresholds, and whether the class
                         changes
figures/                 distance_response_<series>.png, sensitivity_<design>.png

Classification (per estimate)
-----------------------------
    detected               the 95% CI excludes 0
    no meaningful change   the ENTIRE 95% CI lies inside +/- the smallest
                           effect of interest (SESOI below): a small, precise
                           estimate
    inconclusive           neither: the data can't tell a meaningful effect
                           from none (usually the MDE is above the SESOI)
The SESOI values are analyst-selected practical-significance thresholds
(approved provisionally 2026-11-05), NOT established environmental or
regulatory standards. They only make explicit what "small" means here.
sesoi_sensitivity.csv shows which classifications depend on them. Change
them in SESOI and rerun the report (no refit needed). Log-scale series use
log10 units (log10(1.25) = a 25% change).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from analysis import mixed_model as mm
from etl.common.config import default_case_study

# Smallest effect of interest, in model units (see the module docstring).
SESOI = {
    "water_temp": 0.5,                          # deg C
    "dissolved_oxygen": 0.5,                    # mg/L
    "ph": 0.1,                                  # pH units
    "specific_conductance": np.log10(1.10),     # 10%
    "turbidity": np.log10(1.25),                # 25%
    "phosphorus (Total)": np.log10(1.20),       # 20%
    "orthophosphate (Dissolved)": np.log10(1.20),
    "tds": np.log10(1.10),
    "ssc": np.log10(1.25),
    "tss": np.log10(1.25),
}
SESOI_BASIS = ("analyst-selected practical-significance threshold (provisional); "
               "not an environmental or regulatory standard")
SESOI_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]
PRIMARY = ("primary", "auto")              # (run, random effects) of the main results
DISTANCE_SPECS = {"baci_distance": "BACI", "before_after_distance": "before/after"}
SENSITIVITY_RUNS = ["primary", "dar_le_100", "dar_le_10", "dar_le_2", "keep_suspects"]


def sesoi_label(series, sesoi=None, log_scale=None):
    """Plain-language threshold, e.g. '+/-0.5 deg C' or '+/-25% change'."""
    sesoi = SESOI.get(series) if sesoi is None else sesoi
    if sesoi is None or not np.isfinite(sesoi):
        return ""
    if log_scale is None:
        log_scale = series.split(" (")[0] in mm.SETTINGS["log_parameters"]
    if log_scale:
        return f"+/-{(10 ** sesoi - 1) * 100:.0f}% change"
    unit = mm.unit_label(series, False)
    return f"+/-{sesoi:g} {unit}".strip()


def sesoi_thresholds():
    return pd.DataFrame([{"series": series, "sesoi_model_units": value,
                          "scale": "log10" if series.split(" (")[0] in mm.SETTINGS["log_parameters"]
                          else "linear", "threshold": sesoi_label(series), "basis": SESOI_BASIS}
                         for series, value in SESOI.items()])


def classify(estimate, low, high, sesoi):
    if not np.isfinite([estimate, low, high]).all():
        return ""
    if low > 0 or high < 0:
        return "detected"
    if sesoi is not None and np.isfinite(sesoi) and -sesoi <= low and high <= sesoi:
        return "no meaningful change"
    return "inconclusive"


def slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")


# --------------------------------------------------
# Loading
# --------------------------------------------------

def load_runs(stage3_dir):
    """(results, effects_at, settings) over every <run>__<re> folder, with
    run, re and run settings columns. Folders without '__' (the first Stage 3
    runs, before the random-effects option) are skipped."""
    results, effects, settings = [], [], {}
    for folder in sorted(Path(stage3_dir).glob("*__*")):
        if not (folder / "results.csv").exists():
            continue
        info = json.loads((folder / "settings.json").read_text())
        run = info.get("run_name", folder.name).split("__")[0]
        re_mode = info.get("re", folder.name.split("__")[-1])
        tags = {"run": run, "re": re_mode, "run_folder": folder.name,
                "max_dar": info.get("max_dar"), "excluded_dams": len(info.get("excluded_dams", [])),
                "common_sample": info.get("common_sample", False), "ml": info.get("ml", False)}
        settings[folder.name] = info
        results.append(pd.read_csv(folder / "results.csv").assign(**tags))
        at_path = folder / "effects_at.csv"
        if at_path.exists() and at_path.stat().st_size > 1:
            effects.append(pd.read_csv(at_path).assign(**tags))
    if not results:
        raise FileNotFoundError(f"No <run>__<re> model outputs in {stage3_dir}")
    return (pd.concat(results, ignore_index=True),
            pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(), settings)


# --------------------------------------------------
# Tables
# --------------------------------------------------

SUMMARY_COLUMNS = [
    "run", "re", "series", "spec", "design", "dams", "stations", "treated_stations", "days",
    "sample_flag", "km_min", "km_max", "unit", "log_scale", "re_structure", "re_choice",
    "lrt_p", "singular", "near_zero_terms", "fit_check", "converged",
    "effect_at_reference", "effect_ci_low", "effect_ci_high", "mde80_effect", "effect_class",
    "modifier", "modifier_slope", "modifier_ci_low", "modifier_ci_high", "mde80_modifier",
    "slope_class", "sesoi", "sesoi_label", "lrt_stat", "lrt_df", "lrt_p_chi2", "lrt_p_method",
    "lrt_bootstrap_draws", "dams_before_modifier", "rows_dropped_missing_modifier",
    "rows_dropped_missing_covariates", "rows_dropped_excluded_dams", "rows_dropped_max_dar",
    "reference_km", "reference_dar", "size_vs_drainage_r", "size_vs_drainage_vif",
    "llf", "aic", "bic", "max_dar", "excluded_dams", "common_sample", "ml", "error",
]


def model_summary(results):
    out = results.copy()
    out["sesoi"] = out["series"].map(SESOI)
    out["sesoi_label"] = out["series"].map(sesoi_label)
    out["effect_class"] = [classify(e, lo, hi, s) for e, lo, hi, s in zip(
        out.get("effect_at_reference"), out.get("effect_ci_low"), out.get("effect_ci_high"),
        out["sesoi"])]
    # A slope has no natural "meaningful" size: only detected / inconclusive.
    out["slope_class"] = [classify(e, lo, hi, None) if pd.notna(e) else "" for e, lo, hi in zip(
        out.get("modifier_slope", pd.Series(np.nan, index=out.index)),
        out.get("modifier_ci_low", pd.Series(np.nan, index=out.index)),
        out.get("modifier_ci_high", pd.Series(np.nan, index=out.index)))]
    for column in SUMMARY_COLUMNS:
        if column not in out:
            out[column] = np.nan
    return out[SUMMARY_COLUMNS]


def effects_by_distance(effects):
    if effects.empty:
        return effects
    out = effects.copy()
    out["design"] = np.where(out["spec"].str.startswith("baci"), "BACI", "before/after")
    out["sesoi"] = out["series"].map(SESOI)
    out["sesoi_label"] = out["series"].map(sesoi_label)
    out["effect_class"] = [classify(e, lo, hi, s) for e, lo, hi, s in zip(
        out["estimate"], out["ci_low"], out["ci_high"], out["sesoi"])]
    out["primary"] = (out["run"] == PRIMARY[0]) & (out["re"] == PRIMARY[1])
    return out


def sesoi_sensitivity(by_distance, multipliers=SESOI_MULTIPLIERS):
    """Primary estimates classified under smaller / larger thresholds."""
    if by_distance.empty:
        return by_distance
    rows = by_distance[by_distance["primary"] & by_distance["sesoi"].notna()]
    out = rows[["series", "design", "spec", "at", "estimate", "ci_low", "ci_high", "dams"]].copy()
    for m in multipliers:
        out[f"class_at_{m:g}x"] = [classify(e, lo, hi, s * m) for e, lo, hi, s in zip(
            rows["estimate"], rows["ci_low"], rows["ci_high"], rows["sesoi"])]
    class_columns = [c for c in out if c.startswith("class_at_")]
    out["class_changes"] = out[class_columns].nunique(axis=1) > 1
    out["thresholds"] = [", ".join(f"{m:g}x = {sesoi_label(sr, SESOI[sr] * m)}"
                                   for m in multipliers) for sr in rows["series"]]
    return out


def effects_curve(results, step_km=0.25):
    """Effect and 95% CI on a distance grid inside each distance fit's
    observed range: b0 + x b1 with x = ln(1 + d) - ln(1 + reference),
    var = se0^2 + x^2 se1^2 + 2 x cov (the ratio at its reference)."""
    rows = []
    distance = results[results["modifier"].eq("distance")
                       & results["error"].fillna("").eq("")]
    for _, r in distance.iterrows():
        grid = np.arange(max(np.ceil(r["km_min"] / step_km) * step_km, step_km),
                         r["km_max"] + 1e-9, step_km)
        x = np.log1p(grid) - np.log1p(r["reference_km"])
        est = r["effect_at_reference"] + x * r["modifier_slope"]
        var = r["effect_se"] ** 2 + x ** 2 * r["modifier_se"] ** 2 + 2 * x * r["cov_effect_modifier"]
        se = np.sqrt(np.clip(var, 0, None))
        t = stats.t.ppf(0.975, max(r["dams"] - 1, 1))
        rows.append(pd.DataFrame({
            "run": r["run"], "re": r["re"], "series": r["series"], "spec": r["spec"],
            "design": "BACI" if r["design"] == "baci" else "before/after",
            "distance_km": grid, "estimate": est, "ci_low": est - t * se, "ci_high": est + t * se,
            "dams": r["dams"], "unit": r["unit"], "log_scale": r["log_scale"],
            "sample_flag": r["sample_flag"], "re_structure": r["re_structure"]}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def sensitivity_table(effects):
    """Distance effects at 1 / 5 / 10 / 20 km across runs (auto random effects,
    plus the intercept-only primary run)."""
    if effects.empty:
        return effects
    keep = effects[effects["spec"].isin(DISTANCE_SPECS) & effects["distance_km"].isin([1, 5, 10, 20])
                   & (effects["run"].isin(SENSITIVITY_RUNS) & effects["re"].eq("auto")
                      | effects["run"].eq("primary") & effects["re"].eq("intercepts"))]
    label = np.where(keep["re"].eq("intercepts"), keep["run"] + " (intercepts)", keep["run"])
    keep = keep.assign(model_run=label, design=keep["spec"].map(DISTANCE_SPECS))
    cols = ["series", "design", "distance_km", "model_run", "estimate", "ci_low", "ci_high",
            "dams", "effect_class", "re_choice"]
    return keep[[c for c in cols if c in keep]].sort_values(["series", "design", "distance_km",
                                                              "model_run"])


def distance_comparison(results):
    """Raw vs scaled distance on the common sample (ML): AIC difference."""
    common = results[results["common_sample"].astype(bool) & results["ml"].astype(bool)
                     & results["error"].fillna("").eq("")]
    rows = []
    for (series, design), group in common.groupby(["series", "design"]):
        raw = group[group["spec"].str.endswith("_distance") & ~group["spec"].str.contains("scaled")]
        scaled = group[group["spec"].str.contains("scaled")]
        if raw.empty or scaled.empty:
            continue
        r, s = raw.iloc[0], scaled.iloc[0]
        delta = s["aic"] - r["aic"]
        # Identical apart from the distance term: compare the formulas with
        # ld / ls (and their interactions) replaced by one placeholder.
        normalize = lambda f: re.sub(r"\bl[ds]\b", "D",  # noqa: E731
                                     re.sub(r"\b(down|pd|post)_l[ds]\b", r"\1_D", str(f)))
        rows.append({
            "series": series, "design": design, "dams": r["dams"], "days": r["days"],
            "same_sample": bool(r["dams"] == s["dams"] and r["days"] == s["days"]),
            "same_covariates": normalize(r.get("formula")) == normalize(s.get("formula")),
            "same_random_effects": r["re_structure"] == s["re_structure"],
            "both_ml": bool(r["ml"]) and bool(s["ml"]),
            "aic_raw": r["aic"], "aic_scaled": s["aic"], "delta_aic_scaled_minus_raw": delta,
            "bic_raw": r["bic"], "bic_scaled": s["bic"],
            "preferred": ("no clear difference" if abs(delta) < 2
                          else ("scaled distance" if delta < 0 else "raw distance")),
            "raw_slope": r["modifier_slope"], "raw_slope_ci_low": r["modifier_ci_low"],
            "raw_slope_ci_high": r["modifier_ci_high"],
            "scaled_slope": s["modifier_slope"], "scaled_slope_ci_low": s["modifier_ci_low"],
            "scaled_slope_ci_high": s["modifier_ci_high"], "re_structure": r["re_structure"]})
    return pd.DataFrame(rows)


def influence_summary(influence_dir):
    influence_dir = Path(influence_dir)
    parts = []
    for name in ("lodo_summary.csv", "bootstrap_summary.csv"):
        path = influence_dir / name
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for part in parts[1:]:
        out = out.merge(part, on=["series", "spec"], how="outer")
    return out


# --------------------------------------------------
# Figures
# --------------------------------------------------

def plot_distance_response(curve, points, series, axes=None, sesoi=None):
    """BACI and before/after panels: primary curve with 95% band, the
    intercept-only primary run dashed, evaluated distances as points."""
    import matplotlib.pyplot as plt
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    for ax, design in zip(axes, ["BACI", "before/after"]):
        sub = curve[(curve["series"] == series) & (curve["design"] == design)]
        main = sub[(sub["run"] == PRIMARY[0]) & (sub["re"] == PRIMARY[1])]
        alt = sub[(sub["run"] == PRIMARY[0]) & (sub["re"] == "intercepts")]
        ax.axhline(0, color="black", lw=0.8)
        if sesoi is not None and np.isfinite(sesoi):
            ax.axhspan(-sesoi, sesoi, color="0.85", alpha=0.6, lw=0,
                       label=f"+/- smallest effect of interest")
        title = f"{series}: {design}"
        if len(main):
            ax.fill_between(main["distance_km"], main["ci_low"], main["ci_high"], alpha=0.25,
                            color="C0", lw=0, label="95% CI (primary)")
            ax.plot(main["distance_km"], main["estimate"], color="C0",
                    label=f"primary ({main['re_structure'].iloc[0]})")
            flag = main["sample_flag"].iloc[0]
            flag = flag if isinstance(flag, str) else ""     # empty flags read back as NaN
            title += f"  ({int(main['dams'].iloc[0])} dams{', exploratory' if flag else ''})"
            ax.set_ylabel(main["unit"].iloc[0])
        else:
            ax.text(0.5, 0.5, "not fit (too few dams)", ha="center", transform=ax.transAxes)
        if len(alt):
            ax.plot(alt["distance_km"], alt["estimate"], color="C1", ls="--",
                    label="intercept-only (sensitivity)")
        pts = points[(points["series"] == series) & (points["design"] == design)
                     & points["primary"] & points["distance_km"].notna()] \
            if len(points) else pd.DataFrame()
        if len(pts):
            ax.errorbar(pts["distance_km"], pts["estimate"],
                        yerr=[pts["estimate"] - pts["ci_low"], pts["ci_high"] - pts["estimate"]],
                        fmt="o", color="C0", capsize=3, ms=4)
        ax.set_xscale("log")
        ax.set_xlabel("network distance downstream of dam (km, log scale)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="best")
    return axes


def plot_sensitivity(table, design, distance_km=5):
    """Forest plot: effect at one distance across runs, per series."""
    import matplotlib.pyplot as plt
    sub = table[(table["design"] == design) & (table["distance_km"] == distance_km)]
    series = list(dict.fromkeys(sub["series"]))
    fig, axes = plt.subplots(1, max(len(series), 1), figsize=(3.2 * max(len(series), 1), 3.4),
                             squeeze=False)
    for ax, s in zip(axes[0], series):
        g = sub[sub["series"] == s].reset_index(drop=True)
        ax.errorbar(g["estimate"], range(len(g)),
                    xerr=[g["estimate"] - g["ci_low"], g["ci_high"] - g["estimate"]],
                    fmt="o", capsize=3)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(g)), [f"{r} ({d})" for r, d in zip(g["model_run"], g["dams"])],
                      fontsize=7)
        ax.set_title(s, fontsize=9)
    fig.suptitle(f"{design}: effect at {distance_km} km across runs (dams)", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------
# Main
# --------------------------------------------------

def build(model_dir, out_dir, figures=True):
    """Write every report table (and figures); returns them as a dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results, effects, _ = load_runs(Path(model_dir) / "stage3")
    tables = {
        "model_summary": model_summary(results),
        "effects_by_distance": effects_by_distance(effects),
        "effects_curve": effects_curve(results),
        "distance_comparison": distance_comparison(results),
        "influence_summary": influence_summary(Path(model_dir) / "stage4" / "influence"),
    }
    tables["sensitivity_table"] = sensitivity_table(tables["effects_by_distance"])
    tables["sesoi_thresholds"] = sesoi_thresholds()
    tables["sesoi_sensitivity"] = sesoi_sensitivity(tables["effects_by_distance"])
    for name, frame in tables.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
    if figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        curve, points = tables["effects_curve"], tables["effects_by_distance"]
        for series in (curve["series"].unique() if len(curve) else []):
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
            plot_distance_response(curve, points, series, axes, SESOI.get(series))
            fig.tight_layout()
            fig.savefig(fig_dir / f"distance_response_{slug(series)}.png", dpi=150)
            plt.close(fig)
        for design in ("BACI", "before/after"):
            if len(tables["sensitivity_table"]):
                fig = plot_sensitivity(tables["sensitivity_table"], design)
                fig.savefig(fig_dir / f"sensitivity_{slug(design)}.png", dpi=150)
                plt.close(fig)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--model-dir", type=Path,
                        help="Default: data/processed/mixed_model/<case_study>/")
    parser.add_argument("--output-dir", type=Path, help="Default: <model-dir>/stage4/report/")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    model_dir = args.model_dir or mm.output_dir_for(args.case_study)
    out_dir = args.output_dir or model_dir / "stage4" / "report"
    tables = build(model_dir, out_dir, figures=not args.no_figures)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    summary = tables["model_summary"]
    primary = summary[(summary["run"] == PRIMARY[0]) & (summary["re"] == PRIMARY[1])
                      & summary["error"].fillna("").eq("")]
    print(f"Model outputs: {model_dir}\nReport: {out_dir}\n")
    print("Primary distance models (effect at 5 km, distance slope):")
    print(primary[["series", "spec", "dams", "re_choice", "effect_at_reference", "effect_ci_low",
                   "effect_ci_high", "mde80_effect", "effect_class", "modifier_slope",
                   "modifier_ci_low", "modifier_ci_high", "slope_class"]]
          .round(4).to_string(index=False))
    if len(tables["distance_comparison"]):
        print("\nRaw vs scaled distance (same sample, ML):")
        print(tables["distance_comparison"][["series", "design", "dams", "same_sample",
                                             "same_covariates", "same_random_effects", "both_ml",
                                             "delta_aic_scaled_minus_raw", "preferred"]]
              .round(2).to_string(index=False))
    sens = tables["sesoi_sensitivity"]
    if len(sens):
        changed = sens[sens["class_changes"]]
        print(f"\nThreshold sensitivity: {len(changed)} of {len(sens)} primary estimates change "
              f"class between 0.5x and 2x the thresholds (sesoi_sensitivity.csv).")
    if len(tables["influence_summary"]):
        print("\nInfluence:")
        cols = [c for c in ["series", "dams", "influential_dams", "influential_dam_ids",
                            "lodo_effect_min", "lodo_effect_max", "effect_at_reference_boot_low",
                            "effect_at_reference_boot_high"] if c in tables["influence_summary"]]
        print(tables["influence_summary"][cols].round(4).to_string(index=False))
    print(f"\nWrote {', '.join(f'{n}.csv' for n in tables)}"
          f"{' and figures/' if not args.no_figures else ''} to {out_dir}")


if __name__ == "__main__":
    main()
