"""
Mixed-effects models of the dam-removal effect, one fit per water-quality
series (parameter + fraction), on analysis_measurement.

Read by notebooks/mixed_model_results.ipynb. Nothing is written to the
warehouse; outputs go to data/processed/mixed_model/<case_study>/.

Usage
-----
On the cluster (reads the case study's analysis_measurement.parquet):
    python analysis/mixed_model.py
Locally, on an exported CSV (data/slurm/export_analysis_csv.slurm):
    python analysis/mixed_model.py --case-study all_removals \\
        --input analysis_measurement_all_removals.csv

Design
------
1. Roles, as in notebooks/analysis_robustness.ipynb:
   - treated: downstream, and this dam is the measurement's assigned
     removal (exposure_status assigned_sole / assigned_nearest);
   - control: upstream of this dam, not downstream of another removal in
     its window (upstream_control), not possibly in the former reservoir.
   Every other row is left out.
2. One value per dam, station, series and day (the median), so same-day
   repeats and continuous loggers don't count as extra evidence. Skewed
   parameters (LOG_PARAMETERS) are modeled as log10.
3. Two models per series:
   - before_after (treated rows only):
         y ~ post + C(month) + covariates
     The effect is `post`: the average change after removal.
   - baci (dams with both treated and control stations):
         y ~ post + downstream + post_downstream + C(month) + covariates
     The effect is `post_downstream`: the change downstream beyond the
     change upstream at the same dam (which removes the regional trend).
4. Random effects, grouped by dam: each dam gets its own baseline and its
   own effect (random slopes on post, and on post_downstream for baci),
   and each station within a dam its own baseline. The random slopes are
   what keep the interval honest: the effect is judged by how consistent
   it is across dams, not by how many samples there are. (The preview
   models in analysis_eda grouped by dam-station pair only, which gave
   intervals that were too narrow.) If a fit fails, the station term is
   dropped (re_structure says which was used); the slopes are never
   dropped.
5. Confidence intervals use a t distribution with dams - 1 degrees of
   freedom, since the dams are the independent units.

Changing the model
------------------
Everything a design change should touch is in SETTINGS below (roles,
series, covariates, minimum dams) or in model_formula(). REQUIRED_COLUMNS
lists the columns the script reads; a new dataset only needs those.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from etl.common import wq_rules
from etl.common.config import default_case_study, get_project_root

SETTINGS = {
    # Series (parameter, or "parameter (fraction)") to model.
    "series": ["water_temp", "dissolved_oxygen", "ph", "specific_conductance", "turbidity",
               "phosphorus (Total)", "orthophosphate (Dissolved)", "tds"],
    # Modeled as log10 (values clipped at LOG_FLOOR first).
    "log_parameters": ["turbidity", "specific_conductance", "nitrate", "ammonia",
                       "orthophosphate", "phosphorus", "tds"],
    "log_floor": 1e-3,
    # Roles (design 1).
    "treated_exposure": ["assigned_sole", "assigned_nearest"],
    "control_exposure": ["upstream_control"],
    "exclude_former_reservoir": True,
    # Boolean flag columns whose true rows are dropped, e.g. ["qc_saline"].
    "exclude_flags": [],
    # Continuous covariates, standardized (mean 0, SD 1) within each fit.
    # Rows missing any of them are dropped (counted in rows_dropped_missing).
    "covariates": ["air_temp_mean_c", "precip_prev_month_mm"],
    # A model is fit only with at least this many dams (with both roles, for baci).
    "min_dams": 5,
}

REQUIRED_COLUMNS = ["AR_ID", "station_id", "parameter_std", "sample_fraction_std", "side",
                    "period", "sample_date", "month", "result_value_std", "exposure_status",
                    "upstream_in_former_reservoir"]

# Effect term and random slopes of each model (design 3, 4).
MODELS = {
    "before_after": {"effect": "post", "slopes": ["post"]},
    "baci": {"effect": "post_downstream", "slopes": ["post", "post_downstream"]},
}


# --------------------------------------------------
# Data
# --------------------------------------------------

def read_measurements(path, settings=SETTINGS):
    """analysis_measurement (.parquet or .csv), only the columns the model uses."""
    path = Path(path)
    columns = list(dict.fromkeys(REQUIRED_COLUMNS + settings["covariates"]
                                 + settings["exclude_flags"]))
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, low_memory=False)
    if missing := [c for c in columns if c not in df.columns]:
        raise ValueError(f"{path} lacks columns the model needs: {missing}")
    return df[columns]


def series_name(df):
    """parameter, or 'parameter (fraction)' when there is a fraction."""
    return df["parameter_std"].where(
        df["sample_fraction_std"].isna(),
        df["parameter_std"] + " (" + df["sample_fraction_std"].fillna("") + ")")


def assign_roles(df, settings=SETTINGS):
    """'treated', 'control' or 'excluded' per row (design 1)."""
    in_reservoir = df["upstream_in_former_reservoir"].fillna(False).astype(bool)
    if not settings["exclude_former_reservoir"]:
        in_reservoir = pd.Series(False, index=df.index)
    flagged = pd.Series(False, index=df.index)
    for flag in settings["exclude_flags"]:
        flagged |= df[flag].fillna(False).astype(bool)
    treated = (df["side"] == "downstream") & df["exposure_status"].isin(settings["treated_exposure"])
    control = ((df["side"] == "upstream") & df["exposure_status"].isin(settings["control_exposure"])
               & ~in_reservoir)
    return pd.Series(np.select([treated & ~flagged, control & ~flagged],
                               ["treated", "control"], "excluded"), index=df.index)


def prepare(df, settings=SETTINGS):
    """One row per dam, station, series and day (design 2), with model columns:
    y, post, downstream, post_downstream."""
    df = df.assign(series=series_name(df), role=assign_roles(df, settings))
    df = df[df["series"].isin(settings["series"]) & (df["role"] != "excluded")]
    df = df.dropna(subset=["result_value_std"])

    log = df["parameter_std"].isin(settings["log_parameters"])
    value = df["result_value_std"].astype(float)
    df = df.assign(y=np.where(log, np.log10(value.clip(lower=settings["log_floor"])), value),
                   log_scale=log)

    key = ["AR_ID", "station_id", "series", "sample_date"]
    first = ["parameter_std", "role", "period", "month", "log_scale"] + settings["covariates"]
    daily = (df.groupby(key, as_index=False, sort=False)
             .agg(y=("y", "median"), **{c: (c, "first") for c in first}))
    daily["post"] = (daily["period"] == "post").astype(int)
    daily["downstream"] = (daily["role"] == "treated").astype(int)
    daily["post_downstream"] = daily["post"] * daily["downstream"]
    return daily


# --------------------------------------------------
# Model
# --------------------------------------------------

def model_formula(model, covariates):
    """Fixed effects of one model (design 3). Covariates enter as z-scores."""
    terms = ["post"] if model == "before_after" else ["post", "downstream", "post_downstream"]
    terms += ["C(month)"] + [f"z_{c}" for c in covariates]
    return "y ~ " + " + ".join(terms)


def model_rows(daily, model, settings=SETTINGS):
    """The rows one model is fit on: treated only, or dams with both roles."""
    if model == "before_after":
        rows = daily[daily["role"] == "treated"]
    else:
        roles = daily.groupby("AR_ID")["role"].nunique()
        rows = daily[daily["AR_ID"].isin(roles[roles == 2].index)]
    # A dam-station pair needs data on both sides of the removal to say anything.
    periods = rows.groupby(["AR_ID", "station_id"])["post"].transform("nunique")
    return rows[periods == 2]


def fit_model(rows, model, settings=SETTINGS):
    """Fit one mixed model. Returns (result, re_structure, warning text) or raises."""
    rows = rows.copy()
    # A covariate that doesn't vary in these rows can't be estimated; leave it out.
    covariates = [c for c in settings["covariates"] if rows[c].std() > 0]
    for c in covariates:
        rows[f"z_{c}"] = (rows[c] - rows[c].mean()) / rows[c].std()
    formula = model_formula(model, covariates)
    re_formula = "~ " + " + ".join(MODELS[model]["slopes"])
    attempts = [("dam slopes + station intercepts", {"station": "0 + C(station_id)"}),
                ("dam slopes", None)]
    errors = []
    for re_structure, vc in attempts:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = smf.mixedlm(formula, rows, groups=rows["AR_ID"], re_formula=re_formula,
                                     vc_formula=vc).fit(reml=True, method=["lbfgs", "powell"])
            except (np.linalg.LinAlgError, ValueError) as exc:
                errors.append(f"{re_structure}: {exc}")
                continue
        if result.converged and np.isfinite(result.bse[MODELS[model]["effect"]]):
            messages = sorted({str(w.message).split("\n")[0][:120] for w in caught})
            return result, re_structure, "; ".join(messages)
        errors.append(f"{re_structure}: did not converge")
    raise RuntimeError(" | ".join(errors))


def summarize(result, rows, series, model, re_structure, warning_text, n_dropped):
    """One results row: the effect, its t-based 95% CI (dams - 1 df) and sizes."""
    term = MODELS[model]["effect"]
    dams = rows["AR_ID"].nunique()
    estimate, se = result.fe_params[term], result.bse[term]
    t = stats.t.ppf(0.975, dams - 1)
    log_scale = bool(rows["log_scale"].iloc[0])
    # How much the effect varies between dams (SD of the dam random slope).
    slope_sd = float(np.sqrt(max(result.cov_re.loc[term, term], 0)))
    out = {
        "series": series, "model": model, "effect_term": term, "estimate": estimate, "se": se,
        "ci_low": estimate - t * se, "ci_high": estimate + t * se,
        "p_value": 2 * stats.t.sf(abs(estimate / se), dams - 1),
        "effect_sd_between_dams": slope_sd, "log_scale": log_scale,
        "unit": unit_label(series, log_scale), "dams": dams,
        "stations": rows["station_id"].nunique(), "days": len(rows),
        "rows_dropped_missing": n_dropped, "re_structure": re_structure,
        "fit_check": fit_check(warning_text), "warnings": warning_text, "error": "",
    }
    if log_scale:
        for key in ["estimate", "ci_low", "ci_high"]:
            out[f"pct_{key}"] = (10 ** out[key] - 1) * 100
    return out


def fit_check(warning_text):
    """'ok', or 'check' when the Hessian isn't positive definite. That usually
    comes with a random-effect variance estimated at 0 (harmless: the effect
    doesn't vary that way), but the interval should then be compared with the
    dam-bootstrap estimate of notebooks/analysis_robustness.ipynb."""
    return "check" if "Hessian" in warning_text else "ok"


def dam_effects(result, rows, series, model):
    """Each dam's own estimated effect: the average plus its random slope (shrunk
    toward the average when the dam has little data)."""
    term = MODELS[model]["effect"]
    sizes = rows.groupby("AR_ID").agg(days=("y", "size"), stations=("station_id", "nunique"))
    effects = {dam: result.fe_params[term] + re.get(term, 0.0)
               for dam, re in result.random_effects.items()}
    return (pd.Series(effects, name="dam_effect").rename_axis("AR_ID").reset_index()
            .merge(sizes, on="AR_ID").assign(series=series, model=model))


def unit_label(series, log_scale):
    unit = wq_rules.STANDARD_UNITS.get(series.split(" (")[0], "")
    return f"log10 {unit}" if log_scale else unit


def fit_all(daily, settings=SETTINGS):
    """Every series x model. Returns (results, dam_effects, residuals) DataFrames."""
    results, effects, residuals = [], [], []
    for series in settings["series"]:
        for model in MODELS:
            rows = model_rows(daily[daily["series"] == series], model, settings)
            complete = rows.dropna(subset=settings["covariates"])
            n_dropped = len(rows) - len(complete)
            base = {"series": series, "model": model, "effect_term": MODELS[model]["effect"],
                    "dams": complete["AR_ID"].nunique(), "days": len(complete),
                    "rows_dropped_missing": n_dropped}
            if base["dams"] < settings["min_dams"]:
                results.append({**base, "error": f"fewer than {settings['min_dams']} dams"})
                print(f"  {series:<28} {model:<13} skipped: {base['dams']} dams")
                continue
            try:
                result, re_structure, warning_text = fit_model(complete, model, settings)
            except RuntimeError as exc:
                results.append({**base, "error": str(exc)[:300]})
                print(f"  {series:<28} {model:<13} FAILED: {str(exc)[:100]}")
                continue
            row = summarize(result, complete, series, model, re_structure, warning_text, n_dropped)
            results.append(row)
            effects.append(dam_effects(result, complete, series, model))
            residuals.append(complete[["AR_ID", "station_id", "series", "role", "post"]]
                             .assign(model=model, fitted=result.fittedvalues.to_numpy(),
                                     residual=result.resid.to_numpy()))
            print(f"  {series:<28} {model:<13} {row['estimate']:+.4f} "
                  f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]  {row['dams']} dams  "
                  f"({re_structure})")
    return (pd.DataFrame(results),
            pd.concat(effects, ignore_index=True) if effects else pd.DataFrame(),
            pd.concat(residuals, ignore_index=True) if residuals else pd.DataFrame())


# --------------------------------------------------
# Main
# --------------------------------------------------

def output_dir_for(case_study, project_root=None):
    root = Path(project_root) if project_root else get_project_root()
    return root / "data" / "processed" / "mixed_model" / case_study


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study(),
                        help="Case study name (output folder; input unless --input is given).")
    parser.add_argument("--input", type=Path,
                        help="analysis_measurement .parquet or .csv (default: the case "
                             "study's warehouse table).")
    parser.add_argument("--output-dir", type=Path,
                        help="Default: data/processed/mixed_model/<case_study>/")
    parser.add_argument("--series", nargs="+", help="Override SETTINGS['series'].")
    args = parser.parse_args()

    settings = dict(SETTINGS)
    if args.series:
        settings["series"] = args.series
    if args.input is None:
        from etl.common.config import load_config
        from etl.common.warehouse import warehouse_dir
        args.input = warehouse_dir(load_config(args.case_study)) / "analysis_measurement.parquet"
    out_dir = args.output_dir or output_dir_for(args.case_study)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Case study: {args.case_study}\nInput: {args.input}\nOutput: {out_dir}")
    daily = prepare(read_measurements(args.input, settings), settings)
    print(f"Daily values (treated + control): {len(daily):,}; dams: {daily['AR_ID'].nunique()}")
    print("\nEffect [95% CI] per series and model:")
    results, effects, residuals = fit_all(daily, settings)

    results.to_csv(out_dir / "results.csv", index=False)
    effects.to_csv(out_dir / "dam_effects.csv", index=False)
    residuals.to_parquet(out_dir / "residuals.parquet", index=False)
    (out_dir / "settings.json").write_text(json.dumps(
        {**settings, "input": str(args.input), "models": MODELS,
         "formulas": {m: model_formula(m, settings["covariates"]) for m in MODELS}}, indent=2))
    print(f"\nWrote results.csv, dam_effects.csv, residuals.parquet, settings.json to {out_dir}")


if __name__ == "__main__":
    main()
