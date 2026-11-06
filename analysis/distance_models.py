"""
Stage 3-4: does the dam-removal effect change with downstream distance, or
with reservoir size? Optional model specifications next to
analysis/mixed_model.py, whose before_after and baci models are unchanged
(this module reuses its roles, daily collapse and row rules). One fit per
water-quality series and specification. Nothing is written to the
warehouse; outputs go to data/processed/mixed_model/<case_study>/stage3/<run>/.

Usage (cluster: data/slurm/fit_distance_models.slurm runs the primary and
sensitivity runs):
    python analysis/distance_models.py --run-name primary --re auto
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
   backwater_km are dropped (counted). --common-sample fits every spec on
   the rows all distance specs can use (treated rows with distance, ratio
   and backwater_km), so raw and scaled distance compare on the same dams
   and days; with --ml the fits are maximum likelihood, so their AIC / BIC
   are comparable (REML likelihoods aren't across fixed effects).
6. Reservoir size (*_size), a separate exploratory model with no distance
   terms: zsize = standardized log10 of the size measure (--size-variable:
   reservoir_volume_proxy, a constructed index; or a mapped polygon area),
   zlda = standardized log10 dam drainage area (river size, advisor). Both
   are dam-level, standardized over the fit's dams; the effect and the
   regional trend may both vary with them:
       y ~ post + downstream + post_downstream + zsize + zlda + post_zsize
           + post_zlda + down_zsize + down_zlda + pd_zsize + pd_zlda + ...
   Dams without either value are dropped and counted; nothing is imputed.
7. Random effects (--re):
     intercepts  dam and station random intercepts
     slopes      dam random slopes on post (and post_downstream for BACI) plus
                 station intercepts: each dam may have its own effect, so the
                 interval reflects how much dams differ
     auto        both, and slopes are kept only where the data support them:
                 the dam slope variance of the effect term isn't ~0 AND a
                 likelihood-ratio test of slopes vs intercepts (same fixed
                 effects, so REML likelihoods compare) has p < 0.05.
                 Otherwise intercepts, the simpler model (Stage 4 primary;
                 temperature BACI is the known case).
                 The test is on the boundary: under "no slopes" the added
                 variances are 0, the edge of their range, so the statistic
                 doesn't follow chi-square with the added parameters as df
                 but a mixture of chi-squares with fewer df (Self & Liang
                 1987; Stram & Lee 1994). That mixture has less upper-tail
                 mass, so the plain chi-square p (lrt_p_chi2) is an upper
                 bound: p_chi2 < 0.05 already means the boundary-correct
                 p < 0.05, and slopes are kept. With 2 added slopes and their
                 covariances the mixture weights have no simple closed form,
                 so when p_chi2 >= 0.05 the decision uses a parametric
                 bootstrap (--lrt-bootstrap B, default 99): B data sets are
                 simulated from the fitted intercepts model, both models are
                 refit to each, and p = (1 + #{simulated statistic >=
                 observed}) / (1 + successful draws). lrt_p is the p the
                 decision used and lrt_p_method says which (a statistic of
                 ~0 gives p = 1 without simulation). Reported: lrt_stat,
                 lrt_df, lrt_p_chi2, lrt_p, lrt_p_method, lrt_bootstrap_draws.
   Each attempt falls back to dropping the station term if it fails.
   re_structure says what was fit, re_choice why. Each results row carries
   the final variances (var_residual, var_dam_*, var_station); singular /
   near_zero_terms flag any below 0.001 x the residual variance. fit_check
   'check' = the Hessian at the final estimates isn't positive definite.
   singular_warning_during_fit only records statsmodels' warning, which can
   come from intermediate optimizer steps.
8. Intervals: t with dams - 1 degrees of freedom (dams are the
   independent units). A model is fit with at least --min-dams dams (10);
   below --exploratory-below (15) it is flagged exploratory.
   mde80_*: the minimum detectable effect, (t_0.975 + t_0.80) x SE: the
   true effect this fit would detect (two-sided 5%) 80% of the time. An
   inconclusive estimate means "smaller than about this, or none".
9. dams_used.csv: which dams each fitted model used (stations and days by
   role), for maps (notebooks/analysis/distance_model_results.ipynb).
   effects_at.csv: the effect and its 95% CI at 1, 5, 10, 20, 30 km (scaled:
   0.5-20 reservoir lengths; size: the 10th / 50th / 90th percentile dam;
   dilution: also at ratios 2, 10, 100), only inside the fit's observed
   range, with the other modifiers at their reference. A wide interval
   means uncertain, not small; ci_half_width says how uncertain. results.csv
   carries cov_effect_modifier so a curve at any distance can be drawn
   (analysis/stage4_report.py).

10. Discharge check (added 2026-11-05): discharge (instantaneous flow
   measured with a sample, ft3/s, log10) is fit like the water-quality
   series. It is a check on sampling conditions, not a finding: small dam
   removals rarely change mean flow, so a "discharge effect" means the
   downstream samples after removal were taken at different flows than
   before (relative to the controls), and flow-sensitive results (turbidity,
   sediment, conductance, DO) could partly reflect that. The existing
   mixed_model.py series are unchanged; DIAGNOSTIC_SERIES are only added
   here when --series isn't given.

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
         "reservoir_polygon_area_km2", "reservoir_polygon_timing", "dam_drainage_area_km2",
         "removal_year", "years_from_removal"]
# Fit with the water-quality series, but read as checks (design 10).
DIAGNOSTIC_SERIES = ["discharge"]
DEFAULT_SPECS = ["baci_distance", "before_after_distance",
                 "baci_scaled_distance", "before_after_scaled_distance"]
RE_MODES = ["intercepts", "slopes", "auto"]
POWER = 0.80

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


def common_rows(rows):
    """Treated rows every distance spec can use (distance, drainage-area ratio
    and backwater_km all known), controls unchanged (design 5)."""
    treated = rows["role"].eq("treated")
    km = pd.to_numeric(rows["network_distance_km"], errors="coerce")
    dar = pd.to_numeric(rows["drainage_area_ratio"], errors="coerce")
    backwater = pd.to_numeric(rows["backwater_km"], errors="coerce")
    usable = (km >= 0) & (dar > 0) & (backwater > 0)
    return rows[~treated | usable.fillna(False)]


def add_modifiers(rows, spec, reference_km=5.0, size_variable="proxy", fixed_refs=None):
    """rows plus the spec's modifier columns, the rows that can't have them
    removed, and the reference values. fixed_refs (refs of a full-sample fit)
    keeps the reference ratio and size standardization fixed, e.g. for
    leave-one-dam-out and bootstrap refits. Returns (rows, refs, n_dropped)."""
    fixed_refs = fixed_refs or {}
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
        if "reference_dar" in fixed_refs:
            ref = float(fixed_refs["reference_dar"])
        else:
            pairs = rows.loc[treated & keep & has, ["AR_ID", "station_id",
                                                    "drainage_area_ratio"]] \
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
            mean = fixed_refs.get(f"{column}_mean_log10", dams[column].mean())
            sd = fixed_refs.get(f"{column}_sd_log10", dams[column].std())
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

def _attempts(re_formula, label):
    station = {"station": "0 + C(station_id)"}
    return [(f"{label} + station intercepts", re_formula, station), (label, re_formula, None)]


def random_effect_params(result):
    """Covariance parameters of the dam random effects plus variance components."""
    q = int(np.shape(result.cov_re)[0])
    return q * (q + 1) // 2 + len(np.atleast_1d(getattr(result, "vcomp", [])))


def _fit_once(formula, rows, re_formula, vc, reml):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return smf.mixedlm(formula, rows, groups=rows["AR_ID"], re_formula=re_formula,
                           vc_formula=vc).fit(reml=reml, method=["lbfgs", "powell"])


def _lrt_draw(task):
    """One parametric-bootstrap draw: simulate y from the intercepts model,
    refit both models, return the LRT statistic (nan when a fit fails)."""
    rows, formula, null, alt, mean, sds, seed, reml = task
    rng = np.random.default_rng(seed)
    dam_sd, station_sd, resid_sd = sds
    dams = rows["AR_ID"].to_numpy()
    y = mean + rng.normal(0, resid_sd, len(rows))
    unique_dams = pd.unique(dams)
    dam_effect = dict(zip(unique_dams, rng.normal(0, dam_sd, len(unique_dams))))
    y = y + np.array([dam_effect[d] for d in dams])
    if station_sd > 0:
        keys = rows["AR_ID"].astype(str) + "|" + rows["station_id"].astype(str)
        unique_keys = pd.unique(keys)
        st_effect = dict(zip(unique_keys, rng.normal(0, station_sd, len(unique_keys))))
        y = y + keys.map(st_effect).to_numpy()
    sim = rows.assign(y=y)
    try:
        r0 = _fit_once(formula, sim, null[0], null[1], reml)
        r1 = _fit_once(formula, sim, alt[0], alt[1], reml)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan
    if not (r0.converged and r1.converged):
        return np.nan
    return max(2 * (r1.llf - r0.llf), 0.0)


def bootstrap_lrt(rows, formula, null_result, null, alt, stat, draws, seed=0, workers=1,
                  reml=True):
    """Parametric-bootstrap p-value of the slopes-vs-intercepts LRT (design 7).
    null / alt: (re_formula, vc_formula) of the two fitted structures.
    Returns (p, successful draws)."""
    mean = np.asarray(null_result.model.exog) @ null_result.fe_params.to_numpy()
    dam_sd = float(np.sqrt(max(np.asarray(null_result.cov_re)[0, 0], 0)))
    vcomp = np.atleast_1d(getattr(null_result, "vcomp", []))
    station_sd = float(np.sqrt(max(vcomp[0], 0))) if len(vcomp) else 0.0
    sds = (dam_sd, station_sd, float(np.sqrt(null_result.scale)))
    tasks = [(rows, formula, null, alt, mean, sds, seed + i, reml) for i in range(draws)]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as pool:
            simulated = np.array(list(pool.map(_lrt_draw, tasks,
                                               chunksize=max(draws // workers, 1))))
    else:
        simulated = np.array([_lrt_draw(t) for t in tasks])
    simulated = simulated[np.isfinite(simulated)]
    p = (1 + np.sum(simulated >= stat - 1e-9)) / (1 + len(simulated))
    return float(p), int(len(simulated))


def fit(rows, terms, effect, settings, re_mode="intercepts", reml=True, lrt_bootstrap=99,
        lrt_workers=1, seed=0):
    """Fit one model (design 7). Returns (result, re_structure, warning text,
    formula, re_choice, re_test) or raises RuntimeError; re_test holds the
    likelihood-ratio test of auto mode (empty otherwise)."""
    if re_mode not in RE_MODES:
        raise ValueError(f"re_mode must be one of {RE_MODES}")
    rows = rows.copy()
    covariates = [c for c in settings["covariates"] if rows[c].std() > 0]
    for c in covariates:
        rows[f"z_{c}"] = (rows[c] - rows[c].mean()) / rows[c].std()
    # A term that is constant here (e.g. no spread in distance) can't be estimated.
    terms = [t for t in terms if rows[t].std() > 0]
    if effect["base"] not in terms:
        raise RuntimeError(f"{effect['base']} does not vary")
    formula = formula_for(terms, covariates)
    wanted = [t for t in effect.values() if t in terms]
    # re_formula "1" is required: with vc_formula and re_formula None,
    # statsmodels fits no dam random intercept, only the station components
    # (found 2026-11-05; the first Stage 3 cluster run had this).
    intercepts = _attempts("1", "dam intercepts")
    slope_formula = "~ post" if effect["base"] == "post" else "~ post + post_downstream"
    slopes = _attempts(slope_formula, "dam slopes")

    def run(attempts):
        errors = []
        for re_structure, re_formula, vc in attempts:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = smf.mixedlm(formula, rows, groups=rows["AR_ID"],
                                         re_formula=re_formula, vc_formula=vc) \
                        .fit(reml=reml, method=["lbfgs", "powell"])
                except (np.linalg.LinAlgError, ValueError) as exc:
                    errors.append(f"{re_structure}: {exc}")
                    continue
            if result.converged and all(np.isfinite(result.bse[t]) for t in wanted):
                messages = sorted({str(w.message).split("\n")[0][:120] for w in caught})
                return result, re_structure, "; ".join(messages), errors
            errors.append(f"{re_structure}: did not converge")
        return None, None, None, errors

    re_test = {}
    if re_mode == "intercepts":
        result, structure, text, errors = run(intercepts)
        choice = "intercepts (requested)"
    elif re_mode == "slopes":
        result, structure, text, errors = run(slopes)
        choice = "slopes (requested)"
        if result is None:
            result, structure, text, more = run(intercepts)
            errors += more
            choice = "intercepts (slopes fit failed)"
    else:
        s_fit = run(slopes)
        i_fit = run(intercepts)
        errors = s_fit[3] + i_fit[3]
        if s_fit[0] is None:
            result, structure, text, _ = i_fit
            choice = "intercepts (slopes fit failed)"
        elif i_fit[0] is None:
            result, structure, text, _ = s_fit
            choice = "slopes (intercepts fit failed)"
        else:
            stat = max(2 * (s_fit[0].llf - i_fit[0].llf), 0.0)
            df = max(random_effect_params(s_fit[0]) - random_effect_params(i_fit[0]), 1)
            p_chi2 = float(stats.chi2.sf(stat, df))
            re_test = {"lrt_stat": stat, "lrt_df": df, "lrt_p_chi2": p_chi2}
            near = [t.strip() for t in variance_components(s_fit[0])["near_zero_terms"].split(",")]
            if f"dam_{effect['base']}" in near:
                p_value, method = np.nan, "not tested: effect slope variance ~0"
            elif p_chi2 < 0.05:
                p_value, method = p_chi2, "chi-square (upper bound; boundary-correct p is smaller)"
            elif stat < 1e-8:
                p_value, method = 1.0, "statistic ~0"
            elif lrt_bootstrap > 0:
                structures = {label: (rf, vc) for label, rf, vc in intercepts + slopes}
                p_value, n_ok = bootstrap_lrt(rows, formula, i_fit[0], structures[i_fit[1]],
                                              structures[s_fit[1]], stat, lrt_bootstrap, seed,
                                              lrt_workers, reml)
                method = f"parametric bootstrap ({n_ok} draws)"
                re_test["lrt_bootstrap_draws"] = n_ok
            else:
                p_value, method = p_chi2, "chi-square (conservative; bootstrap not run)"
            re_test.update({"lrt_p": p_value, "lrt_p_method": method})
            if f"dam_{effect['base']}" in near:
                result, structure, text, _ = i_fit
                choice = f"intercepts (dam slope of {effect['base']} estimated at ~0)"
            elif p_value >= 0.05:
                result, structure, text, _ = i_fit
                choice = f"intercepts (slopes not supported, LRT p = {p_value:.3f})"
            else:
                result, structure, text, _ = s_fit
                choice = f"slopes (LRT p = {p_value:.2g})"
    if result is None:
        raise RuntimeError(" | ".join(errors))
    return result, structure, text, formula, choice, re_test


def variance_components(result, near_zero=1e-3):
    """The final random-effect variances, and which are near zero.

    Returns {'var_residual', 'var_dam_<term>' per dam random effect,
    'var_station' (when fit), 'singular', 'near_zero_terms'}. A variance
    below near_zero x the residual variance counts as estimated at ~0: the
    fit is on the boundary (singular), so that random effect isn't supported
    by the data and the structure could be simplified. statsmodels' warning
    "Random effects covariance is singular" is raised during the optimizer's
    intermediate steps too, so it alone doesn't mean the final fit is
    singular; this checks the final estimates."""
    scale = float(result.scale) if np.isfinite(result.scale) and result.scale > 0 else 1.0
    cov_re = result.cov_re
    names = [str(n) for n in getattr(cov_re, "columns", range(np.shape(cov_re)[0]))]
    variances = {f"var_dam_{'intercept' if n in ('Group', 'Intercept') else n}": float(v)
                 for n, v in zip(names, np.diag(np.asarray(cov_re)))}
    vcomp = np.atleast_1d(getattr(result, "vcomp", []))
    if len(vcomp):
        variances["var_station"] = float(vcomp[0])
    near = [k.removeprefix("var_") for k, v in variances.items() if v < near_zero * scale]
    return {"var_residual": scale, **variances, "singular": bool(near),
            "near_zero_terms": ", ".join(near)}


def information_criteria(result, reml):
    """Log-likelihood, parameter count, and AIC / BIC when fit by maximum
    likelihood (REML likelihoods aren't comparable across fixed effects)."""
    k_re = int(np.size(result.cov_re)) and (np.shape(result.cov_re)[0]
                                           * (np.shape(result.cov_re)[0] + 1) // 2)
    k = len(result.fe_params) + k_re + len(np.atleast_1d(getattr(result, "vcomp", []))) + 1
    llf = float(result.llf)
    out = {"llf": llf, "n_params": k, "reml": reml}
    if not reml:
        n = int(result.nobs)
        out.update({"aic": -2 * llf + 2 * k, "bic": -2 * llf + np.log(n) * k})
    return out


def mde(se, dams, power=POWER):
    """Minimum detectable effect at two-sided 5% and the given power (design 8)."""
    df = max(dams - 1, 1)
    return (stats.t.ppf(0.975, df) + stats.t.ppf(power, df)) * se


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


def summarize(result, rows, series, spec, effect, refs, meta, settings, reml=True):
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
    out = {**base, "effect_term": effect["base"], "effect_at_reference": est, "effect_se": se,
           "effect_ci_low": lo, "effect_ci_high": hi, "mde80_effect": mde(se, dams)}
    for modifier, label in PRIMARY_MODIFIER.items():
        if modifier in effect and effect[modifier] in result.fe_params:
            s_est, s_se, s_lo, s_hi = linear_effect(result, {effect[modifier]: 1.0}, dams)
            out.update({"modifier": label, "modifier_term": effect[modifier],
                        "modifier_slope": s_est, "modifier_se": s_se,
                        "modifier_ci_low": s_lo, "modifier_ci_high": s_hi,
                        "modifier_p_value": 2 * stats.t.sf(abs(s_est / s_se), max(dams - 1, 1))
                        if s_se > 0 else np.nan,
                        "mde80_modifier": mde(s_se, dams),
                        "cov_effect_modifier": float(result.cov_params().loc[
                            effect["base"], effect[modifier]])})
    warning_text = meta.get("warnings", "")
    out.update({"converged": bool(result.converged), **variance_components(result),
                **information_criteria(result, reml),
                "singular_warning_during_fit": "singular" in warning_text,
                "fit_check": mm.fit_check(warning_text), "error": ""})

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
               "ci_excludes_zero": bool(e_lo > 0 or e_hi < 0), "mde80": mde(e_se, dams),
               "unit": unit, "dams": dams, "sample_flag": flag,
               "re_structure": meta.get("re_structure", ""),
               "re_choice": meta.get("re_choice", "")}
        if log_scale:
            for key in ("estimate", "ci_low", "ci_high"):
                row[f"pct_{key}"] = (10 ** row[key] - 1) * 100
        effects_at.append(row)
    return out, coefficients, effects_at


def model_frame(restricted, series, spec, settings, reference_km=5.0, size_variable="proxy",
                common_sample=False, fixed_refs=None):
    """The exact rows one series x spec is fit on, without fitting (also read
    by notebooks/analysis/model_input_eda.ipynb). Returns (rows, refs, meta):
    meta counts the dams before the modifiers and the rows dropped for a
    missing modifier or covariate."""
    design = SPECS[spec]["design"]
    base_rows = mm.model_rows(restricted[restricted["series"] == series], design, settings)
    if common_sample:
        base_rows = mm.model_rows(common_rows(base_rows), design, settings)
    rows, refs, n_modifier = add_modifiers(base_rows, spec, reference_km, size_variable,
                                           fixed_refs)
    rows = mm.model_rows(rows, design, settings)
    complete = rows.dropna(subset=settings["covariates"])
    meta = {"dams_before_modifier": base_rows["AR_ID"].nunique(),
            "rows_dropped_missing_modifier": n_modifier,
            "rows_dropped_missing_covariates": len(rows) - len(complete)}
    return complete, refs, meta


def fit_one(restricted, series, spec, settings, counts=None, reference_km=5.0,
            size_variable="proxy", re_mode="intercepts", reml=True, common_sample=False,
            fixed_refs=None, lrt_bootstrap=99, lrt_workers=1):
    """One series x spec. Returns a dict: row (results row, with 'error' when
    skipped or failed), coefficients, effects_at, result (or None), rows."""
    design = SPECS[spec]["design"]
    complete, refs, frame_meta = model_frame(restricted, series, spec, settings, reference_km,
                                             size_variable, common_sample, fixed_refs)
    meta = {**(counts or {}), **frame_meta, "re_mode": re_mode, "common_sample": common_sample}
    head = {"series": series, "spec": spec, "design": design,
            "dams": complete["AR_ID"].nunique(), "days": len(complete), **meta}
    empty = {"coefficients": [], "effects_at": [], "result": None, "rows": complete,
             "refs": refs}
    if head["dams"] < settings["min_dams"]:
        return {**empty, "row": {**head, "error": f"fewer than {settings['min_dams']} dams"}}
    terms, effect = term_columns(complete, spec)
    try:
        result, re_structure, warning_text, formula, re_choice, re_test = fit(
            complete, terms, effect, settings, re_mode, reml, lrt_bootstrap, lrt_workers)
    except RuntimeError as exc:
        return {**empty, "row": {**head, "error": str(exc)[:300]}}
    meta.update({"re_structure": re_structure, "re_choice": re_choice, **re_test,
                 "warnings": warning_text, "formula": formula})
    row, coefs, at = summarize(result, complete, series, spec, effect, refs, meta, settings,
                               reml)
    return {"row": row, "coefficients": coefs, "effects_at": at, "result": result,
            "rows": complete, "refs": refs, "effect": effect}


def dams_used(rows, series, spec):
    """One row per dam in a fitted model: its stations and days by role."""
    out = rows.groupby("AR_ID").agg(
        treated_stations=("station_id", lambda s: s[rows.loc[s.index, "role"].eq("treated")].nunique()),
        control_stations=("station_id", lambda s: s[rows.loc[s.index, "role"].eq("control")].nunique()),
        days=("y", "size")).reset_index()
    return out.assign(series=series, spec=spec, design=SPECS[spec]["design"])


def fit_all(daily, specs, settings, exclude_dams=(), max_dar=None, reference_km=5.0,
            size_variable="proxy", re_mode="intercepts", reml=True, common_sample=False,
            lrt_bootstrap=99, lrt_workers=1, dams_out=None):
    """Every series x spec. Returns (results, coefficients, effects_at); with a
    list as dams_out, also appends each fitted model's dams_used() frame."""
    restricted, counts = restrict(daily, exclude_dams, max_dar)
    results, coefficients, effects_at = [], [], []
    for series in settings["series"]:
        for spec in specs:
            fitted = fit_one(restricted, series, spec, settings, counts, reference_km,
                             size_variable, re_mode, reml, common_sample,
                             lrt_bootstrap=lrt_bootstrap, lrt_workers=lrt_workers)
            row = fitted["row"]
            results.append(row)
            coefficients += fitted["coefficients"]
            effects_at += fitted["effects_at"]
            if dams_out is not None and not row.get("error"):
                dams_out.append(dams_used(fitted["rows"], series, spec))
            if row.get("error"):
                kind = "skipped" if row["error"].startswith("fewer") else "FAILED"
                detail = f"{row['dams']} dams" if kind == "skipped" else row["error"][:100]
                print(f"  {series:<28} {spec:<30} {kind}: {detail}")
                continue
            slope = (f"; {row['modifier']} slope {row['modifier_slope']:+.4f} "
                     f"[{row['modifier_ci_low']:+.4f}, {row['modifier_ci_high']:+.4f}]"
                     if "modifier" in row else "")
            print(f"  {series:<28} {spec:<30} at reference {row['effect_at_reference']:+.4f} "
                  f"[{row['effect_ci_low']:+.4f}, {row['effect_ci_high']:+.4f}]{slope}  "
                  f"{row['dams']} dams{' (exploratory)' if row['sample_flag'] else ''}"
                  f"{' SINGULAR' if row['singular'] else ''}  [{row['re_choice']}]")
    return pd.DataFrame(results), pd.DataFrame(coefficients), pd.DataFrame(effects_at)


# --------------------------------------------------
# Main
# --------------------------------------------------

def read_exclusions(path):
    if path is None or not Path(path).exists():
        return []
    return pd.read_csv(path, dtype=str)["AR_ID"].str.strip().dropna().tolist()


def base_parser(description):
    """Options shared with analysis/influence.py."""
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--input", type=Path, help="analysis_measurement .parquet or .csv.")
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
    parser.add_argument("--re", choices=RE_MODES, default="intercepts",
                        help="Random effects (design 7); the Stage 4 primary runs use auto.")
    parser.add_argument("--dam-slopes", action="store_true",
                        help="Same as --re slopes (kept for the Stage 3 commands).")
    parser.add_argument("--min-dams", type=int, default=10)
    parser.add_argument("--exploratory-below", type=int, default=15)
    parser.add_argument("--lrt-bootstrap", type=int, default=99,
                        help="Parametric-bootstrap draws for the auto slopes test when the "
                             "chi-square bound isn't decisive (design 7; 0 = conservative "
                             "chi-square only).")
    parser.add_argument("--workers", type=int,
                        default=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "1")),
                        help="Processes for bootstraps (default: the Slurm CPUs).")
    return parser


def default_series():
    """The mixed model's series plus the diagnostic ones (design 10)."""
    return list(dict.fromkeys(mm.SETTINGS["series"] + DIAGNOSTIC_SERIES))


def load_daily(args):
    """(daily values, settings, excluded dams) for the parsed options."""
    settings = {**mm.SETTINGS, "max_downstream_km": args.max_km, "carry_columns": CARRY,
                "min_dams": args.min_dams, "exploratory_below": args.exploratory_below}
    settings["series"] = args.series or default_series()
    if args.input is None:
        from etl.common.config import load_config
        from etl.common.warehouse import warehouse_dir
        args.input = warehouse_dir(load_config(args.case_study)) / "analysis_measurement.parquet"
    if args.dam_slopes:
        args.re = "slopes"
    exclude = [] if args.keep_suspect_dams else read_exclusions(args.exclude_dams)
    daily = mm.prepare(mm.read_measurements(args.input, settings), settings)
    return daily, settings, exclude


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--run-name", default="primary",
                        help="Output subfolder: data/processed/mixed_model/<case>/stage3/<run>/")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--specs", nargs="+", default=DEFAULT_SPECS, choices=list(SPECS))
    parser.add_argument("--common-sample", action="store_true",
                        help="Fit every spec on the rows all distance specs can use (design 5).")
    parser.add_argument("--ml", action="store_true",
                        help="Maximum likelihood instead of REML, for AIC / BIC comparisons.")
    args = parser.parse_args()

    out_dir = args.output_dir or (mm.output_dir_for(args.case_study) / "stage3" / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily, settings, exclude = load_daily(args)
    print(f"Case study: {args.case_study}; run: {args.run_name}\nInput: {args.input}\n"
          f"Output: {out_dir}\nSpecs: {args.specs}\nTreated rows within {args.max_km} km; "
          f"max drainage-area ratio: {args.max_dar}; excluded dams: {len(exclude)}; "
          f"size variable: {args.size_variable}; random effects: {args.re}; "
          f"common sample: {args.common_sample}; ML: {args.ml}")
    print(f"Daily values (treated + control): {len(daily):,}; dams: {daily['AR_ID'].nunique()}")
    print("\nEffect at the reference [95% CI], and the modifier slope:")
    dams = []
    results, coefficients, effects_at = fit_all(
        daily, args.specs, settings, exclude, args.max_dar, args.reference_km,
        args.size_variable, args.re, not args.ml, args.common_sample,
        args.lrt_bootstrap, args.workers, dams_out=dams)
    (pd.concat(dams, ignore_index=True) if dams else
     pd.DataFrame(columns=["AR_ID", "treated_stations", "control_stations", "days", "series",
                           "spec", "design"])).to_csv(out_dir / "dams_used.csv", index=False)

    results.to_csv(out_dir / "results.csv", index=False)
    coefficients.to_csv(out_dir / "coefficients.csv", index=False)
    effects_at.to_csv(out_dir / "effects_at.csv", index=False)
    (out_dir / "settings.json").write_text(json.dumps(
        {**settings, "input": str(args.input), "run_name": args.run_name, "specs": args.specs,
         "spec_definitions": {s: SPECS[s] for s in args.specs}, "max_dar": args.max_dar,
         "excluded_dams": exclude, "size_variable": args.size_variable,
         "reference_km": args.reference_km, "re": args.re, "common_sample": args.common_sample,
         "ml": args.ml}, indent=2, default=str))
    print(f"\nWrote results.csv, coefficients.csv, effects_at.csv, dams_used.csv, settings.json "
          f"to {out_dir}")


if __name__ == "__main__":
    main()
