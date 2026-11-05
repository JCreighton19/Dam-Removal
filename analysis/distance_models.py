"""
Stage 3: does the dam-removal effect change with downstream distance, or
with reservoir size? Optional model specifications next to
analysis/mixed_model.py, whose before_after and baci models are unchanged
(this module reuses its roles, daily collapse and row rules). One fit per
water-quality series and specification. Nothing is written to the
warehouse; outputs go to data/processed/mixed_model/<case_study>/stage3/<run>/.

Usage (cluster: data/slurm/fit_distance_models.slurm runs the primary and
sensitivity runs):
    python analysis/distance_models.py --run-name primary
    python analysis/distance_models.py --run-name dar_le_10 --max-dar 10
    python analysis/distance_models.py --run-name size --specs baci_size before_after_size

Design (agreed 2026-11-05)
--------------------------
1. Rows: the mixed model's roles and daily values (design 1-2 of
   mixed_model.py), with treated rows up to --max-km (default 30, the
   warehouse window) instead of 10. Dams in --exclude-dams (default
   config/suspect_dam_matches.csv: likely dam-matching errors) are left out
   of every fit unless --keep-suspect-dams. --max-dar keeps only treated
   rows with drainage-area ratio <= the limit (sensitivity runs). A
   restriction removes rows; it never changes assignment or exposure.
2. Distance enters only downstream. ld = ln(1 + km) - ln(1 + 5) for
   treated rows and 0 for upstream controls, so an upstream distance is
   never read as exposure. ln(1 + km) handles 0 km and lets the effect
   change fastest near the dam. Reference 5 km (--reference-km): the
   effect coefficient is the effect at 5 km.
3. BACI distance (primary): the three-way post x downstream x ld. With ld
   zero for controls, post:ld is the same column as post:downstream:ld, so
   the fixed effects are
       y ~ post + downstream + post_downstream + down_ld + pd_ld
           + down_ldar + pd_ldar + C(month) + covariates
   pd_ld is how the BACI effect changes per unit of ln(1 + km). Before/after
   distance (supplementary): y ~ post + ld + post_ld + ldar + post_ldar + ...
4. Dilution. ldar = log10(drainage-area ratio) - log10(reference), the
   reference being the median ratio of the fit's downstream pairs. The ratio
   is a dilution proxy (station / dam NHDPlus drainage area), not a measured
   flow or dilution. It is a station-level constant, so as a plain covariate
   it would only shift station baselines (already absorbed by the station
   intercepts); it adjusts the effect only through pd_ldar (post_ldar), so
   the distance slope is estimated at a fixed dilution. Treated rows
   without a ratio are dropped (counted). *_distance_dilution adds the
   distance x dilution term (pd_ld_ldar).
5. Scaled distance (*_scaled_distance): ls = ln(1 + km / backwater_km) -
   ln(2), so the reference is one reservoir length; dams without
   backwater_km are dropped (counted).
6. Reservoir size (*_size), a separate exploratory model with no distance
   terms: zsize = standardized log10 of the size measure (--size-variable:
   reservoir_volume_proxy, a constructed index; or a mapped polygon area),
   zlda = standardized log10 dam drainage area (river size, advisor). Both
   are dam-level, standardized over the fit's dams; the effect and the
   regional trend may both vary with them:
       y ~ post + downstream + post_downstream + zsize + zlda + post_zsize
           + post_zlda + down_zsize + down_zlda + pd_zsize + pd_zlda + ...
   Dams without either value are dropped and counted; nothing is imputed.
7. Random effects: dam and station random intercepts (no dam random slopes
   unless --dam-slopes). If that fit fails, the station term is dropped;
   re_structure says which was used. singular flags a variance estimated at
   ~0; fit_check flags a non-positive-definite Hessian.
8. Intervals: t with dams - 1 degrees of freedom (dams are the
   independent units). A model is fit with at least --min-dams dams (10);
   below --exploratory-below (15) it is flagged exploratory.
9. effects_at.csv: the effect and its 95% CI at 1, 5, 10, 20, 30 km (scaled:
   0.5-20 reservoir lengths; size: the 10th / 50th / 90th percentile dam;
   dilution: also at ratios 2, 10, 100), only inside the fit's observed
   range, with the other modifiers at their reference. A wide interval
   means uncertain, not small; ci_half_width says how uncertain.

Observational associations: an effect here is a before/after difference
assigned to a removal, not proof that the removal caused it.
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

from analysis import mixed_model as mm
from etl.common.config import default_case_study, get_project_root

DISTANCES_KM = [1, 5, 10, 20, 30]
RESERVOIR_MULTIPLES = [0.5, 1, 2, 5, 10, 20]
DILUTION_RATIOS = [2, 10, 100]
SIZE_PERCENTILES = [10, 50, 90]
SIZE_VARIABLES = {
    "proxy": "reservoir_volume_proxy",
    "polygon_pre_removal": "reservoir_polygon_area_km2",   # mapped by the removal year only
    "polygon_any": "reservoir_polygon_area_km2",
}
CARRY = ["network_distance_km", "drainage_area_ratio", "backwater_km", "reservoir_volume_proxy",
         "reservoir_polygon_area_km2", "reservoir_polygon_timing", "dam_drainage_area_km2"]
DEFAULT_SPECS = ["baci_distance", "before_after_distance",
                 "baci_scaled_distance", "before_after_scaled_distance"]

# design: which rows (mixed_model.model_rows); station: modifiers that exist
# only downstream (0 for controls); dam: dam-level modifiers.
SPECS = {
    "baci_distance": {"design": "baci", "station": ["ld", "ldar"], "dam": []},
    "before_after_distance": {"design": "before_after", "station": ["ld", "ldar"], "dam": []},
    "baci_scaled_distance": {"design": "baci", "station": ["ls", "ldar"], "dam": []},
    "before_after_scaled_distance": {"design": "before_after", "station": ["ls", "ldar"],
                                     "dam": []},
    "baci_distance_dilution": {"design": "baci", "station": ["ld", "ldar", "ld_ldar"], "dam": []},
    "before_after_distance_dilution": {"design": "before_after",
                                       "station": ["ld", "ldar", "ld_ldar"], "dam": []},
    "baci_size": {"design": "baci", "station": [], "dam": ["zsize", "zlda"]},
    "before_after_size": {"design": "before_after", "station": [], "dam": ["zsize", "zlda"]},
}
# The modifier whose effect slope a results row reports.
PRIMARY_MODIFIER = {"ld": "distance", "ls": "scaled_distance", "zsize": "size"}


# --------------------------------------------------
# Rows and modifiers
# --------------------------------------------------

def restrict(daily, exclude_dams=(), max_dar=None):
    """Drop excluded dams (both roles) and, with max_dar, treated rows whose
    drainage-area ratio is above it or unknown. Returns (rows, counts)."""
    counts = {"rows_dropped_excluded_dams": int(daily["AR_ID"].isin(exclude_dams).sum())}
    rows = daily[~daily["AR_ID"].isin(exclude_dams)]
    if max_dar is not None:
        treated = rows["role"].eq("treated")
        keep = ~treated | rows["drainage_area_ratio"].le(max_dar).fillna(False)
        counts["rows_dropped_max_dar"] = int((~keep).sum())
        rows = rows[keep]
    return rows, counts


def add_modifiers(rows, spec, reference_km=5.0, size_variable="proxy"):
    """rows plus the spec's modifier columns, the rows that can't have them
    removed, and the reference values. Returns (rows, refs, n_dropped)."""
    rows = rows.copy()
    treated = rows["role"].eq("treated").to_numpy()
    need = SPECS[spec]["station"] + SPECS[spec]["dam"]
    keep = np.ones(len(rows), dtype=bool)
    refs = {}
    km = pd.to_numeric(rows["network_distance_km"], errors="coerce").to_numpy(dtype=float)
    if "ld" in need or "ls" in need or "ld_ldar" in need:
        keep &= ~treated | (np.isfinite(km) & (km >= 0))
    if "ld" in need or "ld_ldar" in need:
        refs["reference_km"] = reference_km
        rows["ld"] = np.where(treated, np.log1p(np.clip(km, 0, None)) - np.log1p(reference_km),
                              0.0)
    if "ls" in need:
        backwater = pd.to_numeric(rows["backwater_km"], errors="coerce").to_numpy(dtype=float)
        has = np.isfinite(backwater) & (backwater > 0)
        keep &= ~treated | has
        # A dam without backwater_km loses its treated rows, so model_rows
        # drops its controls too (no dam keeps only one role).
        refs["reference_reservoir_lengths"] = 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(has, np.clip(km, 0, None) / backwater, np.nan)
        rows["ls"] = np.where(treated, np.log1p(ratio) - np.log1p(1.0), 0.0)
    if "ldar" in need or "ld_ldar" in need:
        dar = pd.to_numeric(rows["drainage_area_ratio"], errors="coerce").to_numpy(dtype=float)
        has = np.isfinite(dar) & (dar > 0)
        keep &= ~treated | has
        pairs = rows.loc[treated & keep & has, ["AR_ID", "station_id", "drainage_area_ratio"]] \
            .drop_duplicates(["AR_ID", "station_id"])
        ref = float(pairs["drainage_area_ratio"].median()) if len(pairs) else 1.0
        refs["reference_dar"] = ref
        with np.errstate(divide="ignore", invalid="ignore"):
            rows["ldar"] = np.where(treated & has, np.log10(dar) - np.log10(ref), 0.0)
    if "ld_ldar" in need:
        rows["ld_ldar"] = rows["ld"] * rows["ldar"]
    if "zsize" in need:
        size = pd.to_numeric(rows[SIZE_VARIABLES[size_variable]], errors="coerce")
        if size_variable == "polygon_pre_removal":
            size = size.where(rows["reservoir_polygon_timing"].eq("edited_by_removal_year"))
        area = pd.to_numeric(rows["dam_drainage_area_km2"], errors="coerce")
        has = ((size > 0) & (area > 0)).fillna(False).to_numpy()
        keep &= has
        logs = {"zsize": np.log10(size.where(has)), "zlda": np.log10(area.where(has))}
        dams = pd.DataFrame({"AR_ID": rows["AR_ID"], **logs})[keep].drop_duplicates("AR_ID")
        # Standardized over dams (one value each), not over rows.
        for column, values in logs.items():
            mean, sd = dams[column].mean(), dams[column].std()
            refs[f"{column}_mean_log10"], refs[f"{column}_sd_log10"] = mean, sd
            rows[column] = ((values - mean) / sd).to_numpy() if sd and sd > 0 else 0.0
        refs["size_variable"] = size_variable
        # Collinearity of the two dam-level modifiers (two predictors: VIF = 1 / (1 - r^2)).
        r = dams["zsize"].corr(dams["zlda"]) if len(dams) > 2 else np.nan
        refs["size_vs_drainage_r"] = r
        refs["size_vs_drainage_vif"] = 1 / (1 - r ** 2) if np.isfinite(r) and abs(r) < 1 else np.nan
    n_dropped = int((~keep).sum())
    return rows[keep].copy(), refs, n_dropped


def term_columns(rows, spec):
    """Add the interaction columns; returns the fixed-effect terms (without
    month and covariates) and the effect terms: {'base': ..., modifier: term}."""
    design = SPECS[spec]["design"]
    if design == "baci":
        terms, effect = ["post", "downstream", "post_downstream"], {"base": "post_downstream"}
    else:
        terms, effect = ["post"], {"base": "post"}
    for m in SPECS[spec]["station"]:
        if design == "baci":
            rows[f"down_{m}"] = rows["downstream"] * rows[m]
            rows[f"pd_{m}"] = rows["post_downstream"] * rows[m]
            terms += [f"down_{m}", f"pd_{m}"]
            effect[m] = f"pd_{m}"
        else:
            rows[f"post_{m}"] = rows["post"] * rows[m]
            terms += [m, f"post_{m}"]
            effect[m] = f"post_{m}"
    for m in SPECS[spec]["dam"]:
        rows[f"post_{m}"] = rows["post"] * rows[m]
        terms += [m, f"post_{m}"]
        if design == "baci":
            rows[f"down_{m}"] = rows["downstream"] * rows[m]
            rows[f"pd_{m}"] = rows["post_downstream"] * rows[m]
            terms += [f"down_{m}", f"pd_{m}"]
            effect[m] = f"pd_{m}"
        else:
            effect[m] = f"post_{m}"
    return terms, effect


def formula_for(terms, covariates):
    return "y ~ " + " + ".join(terms + ["C(month)"] + [f"z_{c}" for c in covariates])


# --------------------------------------------------
# Fit
# --------------------------------------------------

def fit(rows, terms, effect, settings, dam_slopes=False):
    """Fit one model. Returns (result, re_structure, warning text, formula) or
    raises RuntimeError."""
    rows = rows.copy()
    covariates = [c for c in settings["covariates"] if rows[c].std() > 0]
    for c in covariates:
        rows[f"z_{c}"] = (rows[c] - rows[c].mean()) / rows[c].std()
    # A term that is constant here (e.g. no spread in distance) can't be estimated.
    terms = [t for t in terms if rows[t].std() > 0]
    if effect["base"] not in terms:
        raise RuntimeError(f"{effect['base']} does not vary")
    formula = formula_for(terms, covariates)
    station = {"station": "0 + C(station_id)"}
    attempts = [("dam + station intercepts", None, station), ("dam intercepts", None, None)]
    if dam_slopes:
        slope = "~ post" if effect["base"] == "post" else "~ post + post_downstream"
        attempts = [("dam slopes + station intercepts", slope, station)] + attempts
    wanted = [t for t in effect.values() if t in terms]
    errors = []
    for re_structure, re_formula, vc in attempts:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                result = smf.mixedlm(formula, rows, groups=rows["AR_ID"], re_formula=re_formula,
                                     vc_formula=vc).fit(reml=True, method=["lbfgs", "powell"])
            except (np.linalg.LinAlgError, ValueError) as exc:
                errors.append(f"{re_structure}: {exc}")
                continue
        if result.converged and all(np.isfinite(result.bse[t]) for t in wanted):
            messages = sorted({str(w.message).split("\n")[0][:120] for w in caught})
            return result, re_structure, "; ".join(messages), formula
        errors.append(f"{re_structure}: did not converge")
    raise RuntimeError(" | ".join(errors))


def is_singular(result, tolerance=1e-6):
    """A random-effect variance estimated at ~0 (relative to the residual)."""
    scale = float(result.scale) if np.isfinite(result.scale) and result.scale > 0 else 1.0
    variances = list(np.diag(np.asarray(result.cov_re)))
    variances += list(np.atleast_1d(getattr(result, "vcomp", [])))
    return any(v < tolerance * scale for v in variances)


def linear_effect(result, weights, dams):
    """Estimate, SE and t-based 95% CI of sum(weight * coefficient)."""
    names = list(weights)
    w = np.array([weights[n] for n in names], dtype=float)
    beta = result.fe_params[names].to_numpy()
    cov = result.cov_params().loc[names, names].to_numpy()
    estimate = float(w @ beta)
    se = float(np.sqrt(max(w @ cov @ w, 0)))
    t = stats.t.ppf(0.975, max(dams - 1, 1))
    return estimate, se, estimate - t * se, estimate + t * se


# --------------------------------------------------
# Reporting
# --------------------------------------------------

def evaluation_points(rows, spec, refs):
    """[(label, {modifier: value}, {reported column: value})] inside the
    observed range; the other modifiers stay at their reference (0)."""
    treated = rows[rows["role"].eq("treated")]
    km = pd.to_numeric(treated["network_distance_km"], errors="coerce")
    points = []
    station, dam = SPECS[spec]["station"], SPECS[spec]["dam"]
    if "ld" in station:
        ref = refs["reference_km"]
        for d in DISTANCES_KM:
            if km.min() <= d <= km.max():
                points.append((f"{d} km", {"ld": np.log1p(d) - np.log1p(ref)},
                               {"distance_km": d}))
    if "ls" in station:
        multiple = km / pd.to_numeric(treated["backwater_km"], errors="coerce")
        for m in RESERVOIR_MULTIPLES:
            if multiple.min() <= m <= multiple.max():
                points.append((f"{m}x reservoir length", {"ls": np.log1p(m) - np.log1p(1.0)},
                               {"reservoir_lengths": m}))
    if "ld_ldar" in station:
        dar = pd.to_numeric(treated["drainage_area_ratio"], errors="coerce")
        ref_dar = refs["reference_dar"]
        for d in DISTANCES_KM:
            for r in DILUTION_RATIOS:
                if km.min() <= d <= km.max() and dar.min() <= r <= dar.max():
                    ld = np.log1p(d) - np.log1p(refs["reference_km"])
                    ldar = np.log10(r) - np.log10(ref_dar)
                    points.append((f"{d} km, ratio {r}",
                                   {"ld": ld, "ldar": ldar, "ld_ldar": ld * ldar},
                                   {"distance_km": d, "drainage_area_ratio": r}))
    if "zsize" in dam:
        dams = rows.drop_duplicates("AR_ID")
        for p in SIZE_PERCENTILES:
            z = float(np.percentile(dams["zsize"], p))
            value = 10 ** (refs["zsize_mean_log10"] + z * refs["zsize_sd_log10"])
            points.append((f"size p{p}", {"zsize": z},
                           {"size_percentile": p, "size_value": value}))
    return points


def summarize(result, rows, series, spec, effect, refs, meta, settings):
    """(results row, coefficient rows, effects_at rows)."""
    dams = rows["AR_ID"].nunique()
    log_scale = bool(rows["log_scale"].iloc[0])
    unit = mm.unit_label(series, log_scale)
    treated = rows[rows["role"].eq("treated")]
    km = pd.to_numeric(treated["network_distance_km"], errors="coerce")
    flag = (f"exploratory (fewer than {settings['exploratory_below']} dams)"
            if dams < settings["exploratory_below"] else "")
    base = {"series": series, "spec": spec, "design": SPECS[spec]["design"], "dams": dams,
            "stations": rows["station_id"].nunique(),
            "treated_stations": treated["station_id"].nunique(), "days": len(rows),
            "treated_days": len(treated), "km_min": km.min(), "km_max": km.max(),
            "log_scale": log_scale, "unit": unit, "sample_flag": flag, **refs, **meta}
    est, se, lo, hi = linear_effect(result, {effect["base"]: 1.0}, dams)
    out = {**base, "effect_at_reference": est, "effect_se": se, "effect_ci_low": lo,
           "effect_ci_high": hi}
    for modifier, label in PRIMARY_MODIFIER.items():
        if modifier in effect and effect[modifier] in result.fe_params:
            s_est, s_se, s_lo, s_hi = linear_effect(result, {effect[modifier]: 1.0}, dams)
            out.update({"modifier": label, "modifier_term": effect[modifier],
                        "modifier_slope": s_est, "modifier_se": s_se,
                        "modifier_ci_low": s_lo, "modifier_ci_high": s_hi,
                        "modifier_p_value": 2 * stats.t.sf(abs(s_est / s_se), max(dams - 1, 1))
                        if s_se > 0 else np.nan})
    out.update({"converged": bool(result.converged), "singular": is_singular(result),
                "fit_check": mm.fit_check(meta.get("warnings", "")), "error": ""})

    coefficients = []
    for name in result.fe_params.index:
        c_est, c_se, c_lo, c_hi = linear_effect(result, {name: 1.0}, dams)
        coefficients.append({"series": series, "spec": spec, "term": name, "estimate": c_est,
                             "se": c_se, "ci_low": c_lo, "ci_high": c_hi})

    effects_at = []
    for label, values, reported in evaluation_points(rows, spec, refs):
        weights = {effect["base"]: 1.0}
        weights.update({effect[m]: v for m, v in values.items()
                        if m in effect and effect[m] in result.fe_params})
        e_est, e_se, e_lo, e_hi = linear_effect(result, weights, dams)
        row = {"series": series, "spec": spec, "at": label, **reported, "estimate": e_est,
               "se": e_se, "ci_low": e_lo, "ci_high": e_hi, "ci_half_width": (e_hi - e_lo) / 2,
               "ci_excludes_zero": bool(e_lo > 0 or e_hi < 0), "unit": unit, "dams": dams,
               "sample_flag": flag}
        if log_scale:
            for key in ("estimate", "ci_low", "ci_high"):
                row[f"pct_{key}"] = (10 ** row[key] - 1) * 100
        effects_at.append(row)
    return out, coefficients, effects_at


def fit_all(daily, specs, settings, exclude_dams=(), max_dar=None, reference_km=5.0,
            size_variable="proxy", dam_slopes=False):
    """Every series x spec. Returns (results, coefficients, effects_at)."""
    restricted, counts = restrict(daily, exclude_dams, max_dar)
    results, coefficients, effects_at = [], [], []
    for series in settings["series"]:
        for spec in specs:
            design = SPECS[spec]["design"]
            base_rows = mm.model_rows(restricted[restricted["series"] == series], design, settings)
            rows, refs, n_modifier = add_modifiers(base_rows, spec, reference_km, size_variable)
            rows = mm.model_rows(rows, design, settings)
            complete = rows.dropna(subset=settings["covariates"])
            meta = {**counts, "dams_before_modifier": base_rows["AR_ID"].nunique(),
                    "rows_dropped_missing_modifier": n_modifier,
                    "rows_dropped_missing_covariates": len(rows) - len(complete)}
            head = {"series": series, "spec": spec, "design": design,
                    "dams": complete["AR_ID"].nunique(), "days": len(complete), **meta}
            if head["dams"] < settings["min_dams"]:
                results.append({**head, "error": f"fewer than {settings['min_dams']} dams"})
                print(f"  {series:<28} {spec:<30} skipped: {head['dams']} dams")
                continue
            terms, effect = term_columns(complete, spec)
            try:
                result, re_structure, warning_text, formula = fit(complete, terms, effect,
                                                                  settings, dam_slopes)
            except RuntimeError as exc:
                results.append({**head, "error": str(exc)[:300]})
                print(f"  {series:<28} {spec:<30} FAILED: {str(exc)[:100]}")
                continue
            meta.update({"re_structure": re_structure, "warnings": warning_text,
                         "formula": formula})
            row, coefs, at = summarize(result, complete, series, spec, effect, refs, meta,
                                       settings)
            results.append(row)
            coefficients += coefs
            effects_at += at
            slope = (f"; {row['modifier']} slope {row['modifier_slope']:+.4f} "
                     f"[{row['modifier_ci_low']:+.4f}, {row['modifier_ci_high']:+.4f}]"
                     if "modifier" in row else "")
            print(f"  {series:<28} {spec:<30} at reference {row['effect_at_reference']:+.4f} "
                  f"[{row['effect_ci_low']:+.4f}, {row['effect_ci_high']:+.4f}]{slope}  "
                  f"{row['dams']} dams{' (exploratory)' if row['sample_flag'] else ''}"
                  f"{' SINGULAR' if row['singular'] else ''}")
    return pd.DataFrame(results), pd.DataFrame(coefficients), pd.DataFrame(effects_at)


# --------------------------------------------------
# Main
# --------------------------------------------------

def read_exclusions(path):
    if path is None or not Path(path).exists():
        return []
    return pd.read_csv(path, dtype=str)["AR_ID"].str.strip().dropna().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--input", type=Path, help="analysis_measurement .parquet or .csv.")
    parser.add_argument("--run-name", default="primary",
                        help="Output subfolder: data/processed/mixed_model/<case>/stage3/<run>/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--specs", nargs="+", default=DEFAULT_SPECS, choices=list(SPECS))
    parser.add_argument("--series", nargs="+", help="Override mixed_model SETTINGS['series'].")
    parser.add_argument("--max-km", type=float, default=30.0,
                        help="Treated rows up to this network distance (default 30).")
    parser.add_argument("--max-dar", type=float,
                        help="Keep treated rows with drainage-area ratio <= this (sensitivity).")
    parser.add_argument("--exclude-dams", type=Path,
                        default=get_project_root() / "config" / "suspect_dam_matches.csv",
                        help="CSV with an AR_ID column of dams to leave out.")
    parser.add_argument("--keep-suspect-dams", action="store_true",
                        help="Don't exclude the --exclude-dams list (sensitivity).")
    parser.add_argument("--size-variable", choices=list(SIZE_VARIABLES), default="proxy")
    parser.add_argument("--reference-km", type=float, default=5.0)
    parser.add_argument("--dam-slopes", action="store_true",
                        help="Try dam random slopes first (default: intercepts only).")
    parser.add_argument("--min-dams", type=int, default=10)
    parser.add_argument("--exploratory-below", type=int, default=15)
    args = parser.parse_args()

    settings = {**mm.SETTINGS, "max_downstream_km": args.max_km, "carry_columns": CARRY,
                "min_dams": args.min_dams, "exploratory_below": args.exploratory_below}
    if args.series:
        settings["series"] = args.series
    if args.input is None:
        from etl.common.config import load_config
        from etl.common.warehouse import warehouse_dir
        args.input = warehouse_dir(load_config(args.case_study)) / "analysis_measurement.parquet"
    out_dir = args.output_dir or (mm.output_dir_for(args.case_study) / "stage3" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    exclude = [] if args.keep_suspect_dams else read_exclusions(args.exclude_dams)

    print(f"Case study: {args.case_study}; run: {args.run_name}\nInput: {args.input}\n"
          f"Output: {out_dir}\nSpecs: {args.specs}\nTreated rows within {args.max_km} km; "
          f"max drainage-area ratio: {args.max_dar}; excluded dams: {len(exclude)}; "
          f"size variable: {args.size_variable}; dam slopes: {args.dam_slopes}")
    daily = mm.prepare(mm.read_measurements(args.input, settings), settings)
    print(f"Daily values (treated + control): {len(daily):,}; dams: {daily['AR_ID'].nunique()}")
    print("\nEffect at the reference [95% CI], and the modifier slope:")
    results, coefficients, effects_at = fit_all(
        daily, args.specs, settings, exclude, args.max_dar, args.reference_km,
        args.size_variable, args.dam_slopes)

    results.to_csv(out_dir / "results.csv", index=False)
    coefficients.to_csv(out_dir / "coefficients.csv", index=False)
    effects_at.to_csv(out_dir / "effects_at.csv", index=False)
    (out_dir / "settings.json").write_text(json.dumps(
        {**settings, "input": str(args.input), "run_name": args.run_name, "specs": args.specs,
         "spec_definitions": {s: SPECS[s] for s in args.specs}, "max_dar": args.max_dar,
         "excluded_dams": exclude, "size_variable": args.size_variable,
         "reference_km": args.reference_km, "dam_slopes": args.dam_slopes}, indent=2,
        default=str))
    print(f"\nWrote results.csv, coefficients.csv, effects_at.csv, settings.json to {out_dir}")


if __name__ == "__main__":
    main()
