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

11. Time since removal (baci_time, before_after_time; added 2026-11-06).
   The binary post term is replaced by categorical time-since-removal bins:
       y ~ tb_<bin>... + downstream + pd_tb_<bin>... + C(month) + covariates
   (before/after: y ~ tb_<bin>... + ...). tb_<bin> = 1 for rows whose
   years_from_removal (sample year - removal year) falls in that bin; the
   first bin (default: all pre years, -5..-1) is the reference. pd_tb_<bin>
   is the BACI effect in that interval. Removal dates are known only to the
   year, so the removal year itself is never in the data (measurement_window)
   and "year 1" = the first calendar year after removal, i.e. 1 to 24 months
   after it; a bin is a band of calendar years, not an exact elapsed time.
   Bins come from --time-bins (default DEFAULT_TIME_BINS, chosen from the
   coverage audit, time_coverage()); a pre bin other than the reference
   (e.g. TIME_BINNINGS["pretrend"]) is a placebo: its pd term should be ~0
   if downstream and upstream trended alike before removal. Reported:
   effects_at rows per bin, the effect averaged over the post bins
   (weighted by treated post days, comparable to the binary effect),
   time_change_F / _p (Wald F test, dams - 1 df, that the post bins are
   equal) and pretrend_F / _p (that the placebo pre bins are 0). Random
   slopes (--re slopes / auto) stay on post and post_downstream: each dam may
   have its own overall effect; the time profile is fixed. baci_binary /
   before_after_binary are the existing mixed_model.py formulas (no
   modifiers) fit on the same rows, the benchmark: time_comparison.csv puts
   them side by side and, when both are fit by ML (--ml), the likelihood-
   ratio test of time-varying vs constant effect and AIC / BIC, only when
   both have identical observations (dams, stations, days) and the same
   fitted random-effects structure (same_random_effects).
   Sensitivity runs (--analysis-role sensitivity; slurm time_ml_slopes and
   time_coarse): the ML comparison with dam random slopes in both models
   (an intercepts-only likelihood is overconfident where dams differ), and
   coarser 1-2 / 3-5 year bins. Each effects_at row reports the treated and
   control dams and daily observations behind its interval and the dams
   with both roles in it (what identifies a BACI interval).

12. Dam and watershed characteristics as effect modifiers (*_x_<name>,
   *_xu_<name>; added 2026-11-06). CHARACTERISTICS lists each candidate,
   its column, level and transform. A dam-level characteristic (reservoir
   volume / area, dam height, dam drainage area) is standardized over the
   fit's dams and enters like size (design 6): m + post_m + down_m + pd_m
   for BACI; a station-level one (baseline land cover of the downstream
   station's watershed, distance) exists only downstream and enters like
   distance: down_m + pd_m (before/after: m + post_m). pd_m (post_m) is how
   the effect changes per 1 SD of the characteristic: effect modification,
   an association across dams, not a causal effect of the characteristic.
   Adjustment vs explanation: month, air temperature and previous-month
   rain are adjustment covariates (they explain variation in water quality,
   not in the response); the dam and station random intercepts absorb any
   characteristic's main effect, so a characteristic added only as a plain
   covariate would say nothing about the response. *_x_ specs also adjust
   the effect for distance and dilution (ld, ldar, as in the primary
   distance models: pd_m is then the modification at a fixed distance and
   dilution); *_xu_ specs don't (sensitivity). Dams (dam-level) or treated
   rows (station-level) without the value are dropped and counted, never
   imputed, so each characteristic is fit on its own sample. Stored sediment
   is not available (UNAVAILABLE_CHARACTERISTICS). modifier_audit.csv and
   modifier_correlations.csv give availability, missingness and collinearity;
   modifier_screen.csv the screen with Benjamini-Hochberg q-values across
   series x characteristics per design; *_x_combined (--combined a b ...)
   fits several together, and combined_check.csv says whether the sample
   supports it (complete-case dams per modifier, correlations, VIF).

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

# Benchmark: the mixed_model.py formulas, no modifiers (design 11).
SPECS["baci_binary"] = {"design": "baci", "station": [], "dam": []}
SPECS["before_after_binary"] = {"design": "before_after", "station": [], "dam": []}
# Time since removal (design 11).
SPECS["baci_time"] = {"design": "baci", "station": [], "dam": [], "time": True}
SPECS["before_after_time"] = {"design": "before_after", "station": [], "dam": [], "time": True}

# (label, first year, last year) of years_from_removal; the first bin is the
# reference. Pre and post years never share a bin (year 0 is not in the data).
TIME_BINNINGS = {
    "yearly": [("pre", -5, -1), ("y1", 1, 1), ("y2", 2, 2), ("y3", 3, 3), ("y4", 4, 4),
               ("y5", 5, 5)],
    "y1_y2-3_y4-5": [("pre", -5, -1), ("y1", 1, 1), ("y2_3", 2, 3), ("y4_5", 4, 5)],
    "y1-2_y3-5": [("pre", -5, -1), ("y1_2", 1, 2), ("y3_5", 3, 5)],
    # Placebo: late pre years are the reference, early pre years should match them.
    "pretrend": [("pre_late", -2, -1), ("pre_early", -5, -3), ("y1", 1, 1), ("y2_3", 2, 3),
                 ("y4_5", 4, 5)],
}
# Chosen from the coverage audit (time_coverage) on the 2026-10-27 export,
# 10 km, suspect dams out: for temperature, DO and pH 17-24 dams have treated
# AND control days in every single post year, so year 1 (the short-term
# response, e.g. a sediment pulse) can stand alone and years 2-3 and 4-5 are
# pooled: >= 18 such dams per bin (conductance 13, total P 10; turbidity,
# orthophosphate and TDS stay below min_dams in any binning). Rerun the audit
# (--audit-only) on the current warehouse before relying on it.
DEFAULT_TIME_BINS = TIME_BINNINGS["y1_y2-3_y4-5"]

# Candidate effect modifiers (design 12). level: "dam" (one value per dam) or
# "station" (the downstream station's value; controls get 0). transform of the
# column before standardizing; values outside the transform's domain are unknown.
CHARACTERISTICS = {
    "reservoir_volume": {"column": "reservoir_volume_proxy", "level": "dam", "transform": "log10",
                         "label": "reservoir volume proxy (constructed index)"},
    "reservoir_area": {"column": "reservoir_polygon_area_km2", "level": "dam",
                       "transform": "log10", "pre_removal_only": True,
                       "label": "mapped reservoir area, km2 (mapped by the removal year)"},
    "dam_height": {"column": "dam_height_ft", "level": "dam", "transform": "log10",
                   "label": "dam height, ft"},
    "dam_drainage_area": {"column": "dam_drainage_area_km2", "level": "dam", "transform": "log10",
                          "label": "dam drainage area, km2"},
    "pct_forest": {"column": "pct_forest", "level": "station", "transform": "identity",
                   "label": "forest, % of the downstream station's watershed"},
    "pct_developed": {"column": "pct_developed", "level": "station", "transform": "identity",
                      "label": "developed, % of the downstream station's watershed"},
    "pct_agriculture": {"column": "pct_agriculture", "level": "station", "transform": "identity",
                        "label": "agriculture, % of the downstream station's watershed"},
    "distance": {"column": "network_distance_km", "level": "station", "transform": "log1p",
                 "label": "network distance downstream, km"},
}
UNAVAILABLE_CHARACTERISTICS = {
    "stored_sediment": "no sediment volume or mass in the AR list, NHD or the warehouse; "
                       "reservoir_volume_proxy (height^2 / slope x sqrt(area)) is the closest, "
                       "and it is a constructed index",
}
TRANSFORM_LABELS = {"log10": "log10", "log1p": "ln(1 + x)", "identity": ""}


def _characteristic_specs():
    """*_x_<name> (adjusted for distance and dilution) and *_xu_<name> specs."""
    for name, ch in CHARACTERISTICS.items():
        m = f"m_{name}"
        for design in ("baci", "before_after"):
            for suffix, adjust in (("x", True), ("xu", False)):
                # Distance itself replaces ld (same variable) and keeps the dilution adjustment.
                station = [a for a in ("ld", "ldar") if adjust and not (name == "distance"
                                                                        and a == "ld")]
                dam = []
                (station if ch["level"] == "station" else dam).append(m)
                SPECS[f"{design}_{suffix}_{name}"] = {"design": design, "station": station,
                                                      "dam": dam}
    # Placeholders, filled by register_combined().
    for design in ("baci", "before_after"):
        SPECS[f"{design}_x_combined"] = {"design": design, "station": ["ld", "ldar"], "dam": [],
                                         "combined": True}


_characteristic_specs()

SPEC_SETS = {
    "time": ["baci_binary", "baci_time", "before_after_binary", "before_after_time"],
    "modifier_screen": [f"{d}_x_{n}" for d in ("baci", "before_after") for n in CHARACTERISTICS],
    "modifier_screen_unadjusted": [f"{d}_xu_{n}" for d in ("baci", "before_after")
                                   for n in CHARACTERISTICS],
}


def register_combined(names):
    """Fill *_x_combined with the given characteristics (design 12)."""
    unknown = [n for n in names if n not in CHARACTERISTICS]
    if unknown:
        raise ValueError(f"Unknown characteristics {unknown}; choose from {list(CHARACTERISTICS)}")
    for design in ("baci", "before_after"):
        station = ["ldar"] + ([] if "distance" in names else ["ld"])
        station = [s for s in ("ld", "ldar") if s in station]
        dam = []
        for n in names:
            (station if CHARACTERISTICS[n]["level"] == "station" else dam).append(f"m_{n}")
        SPECS[f"{design}_x_combined"] = {"design": design, "station": station, "dam": dam,
                                         "combined": True, "names": list(names)}


def characteristic_columns(specs):
    """Warehouse columns the given specs' characteristics need."""
    columns = []
    for spec in specs:
        for m in SPECS[spec]["station"] + SPECS[spec]["dam"]:
            if m.startswith("m_"):
                ch = CHARACTERISTICS[m[2:]]
                columns += [ch["column"]] + (["reservoir_polygon_timing"]
                                             if ch.get("pre_removal_only") else [])
    return list(dict.fromkeys(columns))


def characteristic_values(rows, name):
    """The characteristic on its model scale (transform applied); NaN when unknown."""
    ch = CHARACTERISTICS[name]
    x = pd.to_numeric(rows[ch["column"]], errors="coerce").astype(float)
    if ch.get("pre_removal_only"):
        x = x.where(rows["reservoir_polygon_timing"].eq("edited_by_removal_year"))
    if ch["transform"] == "log10":
        return np.log10(x.where(x > 0))
    if ch["transform"] == "log1p":
        return np.log1p(x.where(x >= 0))
    return x


def parse_time_bins(text):
    """'pre:-5:-1;y1_2:1:2;y3_5:3:5' (or a TIME_BINNINGS name) -> [(label, lo, hi)]."""
    if text in TIME_BINNINGS:
        return TIME_BINNINGS[text]
    bins = []
    for part in text.split(";"):
        label, lo, hi = part.strip().split(":")
        bins.append((label.strip(), int(lo), int(hi)))
    check_time_bins(bins)
    return bins


def check_time_bins(bins):
    labels = [b[0] for b in bins]
    if len(set(labels)) != len(labels) or not all(re_label(lb) for lb in labels):
        raise ValueError(f"Time-bin labels must be unique words (letters, digits, _): {labels}")
    years = []
    for label, lo, hi in bins:
        if lo > hi or (lo < 0 < hi) or 0 in (lo, hi):
            raise ValueError(f"Bin {label} ({lo}..{hi}) must be all pre (<0) or all post (>0) "
                             "years; year 0 (the removal year) is not in the data")
        years += list(range(lo, hi + 1))
    if len(set(years)) != len(years):
        raise ValueError(f"Time bins overlap: {bins}")
    if not any(lo > 0 for _, lo, _ in bins):
        raise ValueError("Time bins need at least one post bin")
    if bins[0][1] > 0:
        raise ValueError(f"The first (reference) bin must be pre-removal: {bins[0]}")
    return bins


def re_label(label):
    return bool(label) and all(c.isalnum() or c == "_" for c in label)


def time_bin_columns(bins):
    """{label: column}: tbpre_<label> / tbpost_<label> for every non-reference bin."""
    return {label: f"tb{'post' if lo > 0 else 'pre'}_{label}" for label, lo, hi in bins[1:]}


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


def add_modifiers(rows, spec, reference_km=5.0, size_variable="proxy", fixed_refs=None,
                  time_bins=None):
    """rows plus the spec's modifier columns, the rows that can't have them
    removed, and the reference values. fixed_refs (refs of a full-sample fit)
    keeps the reference ratio and size standardization fixed, e.g. for
    leave-one-dam-out and bootstrap refits. time_bins: time specs' bins
    (default DEFAULT_TIME_BINS). Returns (rows, refs, n_dropped)."""
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
    for m in [x for x in need if x.startswith("m_")]:
        # Characteristic modifiers (design 12): standardized over dams (dam level)
        # or treated dam-station pairs (station level); never imputed.
        name, level = m[2:], CHARACTERISTICS[m[2:]]["level"]
        values = characteristic_values(rows, name).to_numpy(dtype=float)
        has = np.isfinite(values)
        if level == "dam":
            keep &= has
            units = pd.DataFrame({"AR_ID": rows["AR_ID"].to_numpy(), "v": values})[keep]
            units = units.drop_duplicates("AR_ID")
        else:
            keep &= ~treated | has
            units = pd.DataFrame({"AR_ID": rows["AR_ID"].to_numpy(),
                                  "station_id": rows["station_id"].to_numpy(), "v": values})
            units = units[treated & keep & has].drop_duplicates(["AR_ID", "station_id"])
        mean = fixed_refs.get(f"{m}_mean", units["v"].mean())
        sd = fixed_refs.get(f"{m}_sd", units["v"].std())
        refs[f"{m}_mean"], refs[f"{m}_sd"] = mean, sd
        z = (values - mean) / sd if sd and sd > 0 else np.zeros(len(rows))
        rows[m] = np.where(has, z, 0.0) if level == "dam" else np.where(treated & has, z, 0.0)
    if SPECS[spec].get("time"):
        # Time-since-removal bins (design 11); years outside every bin are dropped.
        bins = check_time_bins(list(time_bins or DEFAULT_TIME_BINS))
        years = pd.to_numeric(rows["years_from_removal"], errors="coerce").to_numpy(dtype=float)
        label = np.full(len(rows), None, dtype=object)
        for b, lo, hi in bins:
            label[(years >= lo) & (years <= hi)] = b
        keep &= pd.notna(label)
        for b, column in time_bin_columns(bins).items():
            rows[column] = (label == b).astype(float)
        refs["time_bins"] = ";".join(f"{b}:{lo}:{hi}" for b, lo, hi in bins)
        refs["time_reference"] = bins[0][0]
    n_dropped = int((~keep).sum())
    return rows[keep].copy(), refs, n_dropped


def term_columns(rows, spec):
    """Add the interaction columns; returns the fixed-effect terms (without
    month and covariates) and the effect terms: {'base': ..., modifier: term}."""
    design = SPECS[spec]["design"]
    if SPECS[spec].get("time"):
        # Design 11: a term per non-reference bin instead of post; the effect
        # terms are keyed 'time:<label>' ('base' = the first post bin), and
        # 're_base' keeps the random slopes on post / post_downstream.
        bins = [c for c in rows.columns if c.startswith(("tbpre_", "tbpost_"))]
        post_bins = [c for c in bins if c.startswith("tbpost_")]
        if design == "baci":
            for c in bins:
                rows[f"pd_{c}"] = rows["downstream"] * rows[c]
            terms = bins + ["downstream"] + [f"pd_{c}" for c in bins]
            effect = {"base": f"pd_{post_bins[0]}", "re_base": "post_downstream",
                      **{f"time:{c.split('_', 1)[1]}": f"pd_{c}" for c in bins}}
        else:
            terms = list(bins)
            effect = {"base": post_bins[0], "re_base": "post",
                      **{f"time:{c.split('_', 1)[1]}": c for c in bins}}
        return terms, effect
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
    # Random slopes are on post (and post_downstream for BACI), also for the
    # time specs, whose fixed effects split post into bins (design 11).
    re_base = effect.get("re_base", effect["base"])
    slope_formula = "~ post" if re_base == "post" else "~ post + post_downstream"
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
            if f"dam_{re_base}" in near:
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
            if f"dam_{re_base}" in near:
                result, structure, text, _ = i_fit
                choice = f"intercepts (dam slope of {re_base} estimated at ~0)"
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


def power(effect, se, dams, alpha=0.05):
    """Post-hoc design sensitivity: the chance a two-sided t test (dams - 1 df,
    as the model CIs) detects a TRUE effect of this size, given the fit's SE.

    Pass a practical threshold (SESOI), in the same units as `se` (log10 for
    log-scale series), never the estimate itself: power at the observed
    estimate only restates its p-value. Noncentral t, so power(mde(se, n), se, n)
    is about 0.8 (mde() uses the usual t_0.975 + t_0.80 shortcut). dams - 1 df
    is an approximation for mixed models (dams are the independent units).
    """
    effect, se = np.asarray(effect, float), np.asarray(se, float)
    df = np.maximum(np.asarray(dams, float) - 1, 1)
    crit = stats.t.ppf(1 - alpha / 2, df)
    with np.errstate(divide="ignore", invalid="ignore"):
        ncp = np.abs(effect) / se
        # The wrong-direction tail is ~0 for a large effect; scipy returns NaN there.
        out = stats.nct.sf(crit, df, ncp) + np.nan_to_num(stats.nct.cdf(-crit, df, ncp), nan=0.0)
    out = np.where((se > 0) & np.isfinite(ncp), np.clip(out, 0, 1), np.nan)
    return out.item() if out.ndim == 0 else out


def dams_needed(mde_now, sesoi, dams, power=POWER, max_dams=10_000):
    """Rough projection, NOT a sample-size requirement: how many dams would
    bring the MDE down to the threshold, if the SE shrank like
    1 / sqrt(dams) and new dams resembled the current ones (same variances,
    stations and sampling). Same units for mde_now and sesoi. NaN if an input
    is missing; inf if not reached by max_dams.
    """
    if not np.isfinite([mde_now, sesoi, dams]).all() or sesoi <= 0 or mde_now <= 0 or dams < 2:
        return np.nan
    se_now = mde_now / (stats.t.ppf(0.975, max(dams - 1, 1)) + stats.t.ppf(power, max(dams - 1, 1)))
    n = np.arange(2, max_dams + 1)
    projected = (stats.t.ppf(0.975, n - 1) + stats.t.ppf(power, n - 1)) * se_now * np.sqrt(dams / n)
    hit = np.flatnonzero(projected <= sesoi)
    return float(n[hit[0]]) if len(hit) else np.inf


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
    for m in [x for x in station + dam if x.startswith("m_")]:
        # Characteristics (design 12): the 10th / 50th / 90th percentile dam
        # (dam level) or downstream pair (station level), the others at their mean.
        name = m[2:]
        units = (rows.drop_duplicates("AR_ID") if CHARACTERISTICS[name]["level"] == "dam"
                 else treated.drop_duplicates(["AR_ID", "station_id"]))
        for p in SIZE_PERCENTILES:
            z = float(np.percentile(units[m], p))
            points.append((f"{name} p{p}", {m: z},
                           {"characteristic": name, "characteristic_percentile": p,
                            "characteristic_value": back_transform(
                                name, refs[f"{m}_mean"] + z * refs[f"{m}_sd"])}))
    return points


def back_transform(name, value):
    """A characteristic's model-scale value in its original units."""
    transform = CHARACTERISTICS[name]["transform"]
    return 10 ** value if transform == "log10" else np.expm1(value) if transform == "log1p" \
        else value


def primary_modifier(effect):
    """(modifier key, label) a results row reports the slope of: a single
    characteristic first (design 12), else distance / scaled distance / size;
    (None, None) for several characteristics or none."""
    characteristics = [k for k in effect if k.startswith("m_")]
    if characteristics:
        return (characteristics[0], characteristics[0][2:]) if len(characteristics) == 1 \
            else (None, None)
    for key, label in PRIMARY_MODIFIER.items():
        if key in effect:
            return key, label
    return None, None


def wald_f(result, contrasts, dams):
    """F test of contrasts @ beta = 0 ({name: weight} per row): (F, df1, p),
    with dams - 1 denominator df like the intervals (design 8)."""
    contrasts = [c for c in contrasts if all(n in result.fe_params for n in c)]
    if not contrasts:
        return np.nan, 0, np.nan
    names = sorted({n for c in contrasts for n in c})
    C = np.array([[c.get(n, 0.0) for n in names] for c in contrasts], dtype=float)
    beta = result.fe_params[names].to_numpy()
    cov = result.cov_params().loc[names, names].to_numpy()
    diff = C @ beta
    try:
        wald = float(diff @ np.linalg.solve(C @ cov @ C.T, diff))
    except np.linalg.LinAlgError:
        return np.nan, len(contrasts), np.nan
    df1 = len(contrasts)
    f = wald / df1
    return f, df1, float(stats.f.sf(f, df1, max(dams - 1, 1)))


def time_effects(result, rows, spec, effect, dams):
    """Design 11: (summary columns, [(label, weights, reported)]) for a time
    spec. The average effect weights the post bins by treated post days."""
    baci = SPECS[spec]["design"] == "baci"
    treated = rows[rows["role"].eq("treated")] if baci else rows
    control = rows[rows["role"].eq("control")] if baci else rows.iloc[0:0]
    bins = [(k.split(":", 1)[1], t) for k, t in effect.items() if k.startswith("time:")]
    post = [(b, t) for b, t in bins if f"tbpost_{b}" in rows and t in result.fe_params]
    pre = [(b, t) for b, t in bins if f"tbpre_{b}" in rows and t in result.fe_params]
    days = {b: float(treated[f"tbpost_{b}"].sum()) for b, _ in post}
    total = sum(days.values())
    weights = {t: days[b] / total for b, t in post} if total else {}
    change = wald_f(result, [{t: 1.0, post[0][1]: -1.0} for _, t in post[1:]], dams) \
        if len(post) > 1 else (np.nan, 0, np.nan)
    pretrend = wald_f(result, [{t: 1.0} for _, t in pre], dams)
    summary = {"time_post_bins": ", ".join(b for b, _ in post),
               "time_change_F": change[0], "time_change_df": change[1], "time_change_p": change[2],
               "pretrend_F": pretrend[0], "pretrend_df": pretrend[1], "pretrend_p": pretrend[2]}
    def counts(t_rows, c_rows):
        """Dams and daily observations behind an interval, by role. BACI
        identifies an interval's effect from dams with both roles in it."""
        both = len(set(t_rows["AR_ID"]) & set(c_rows["AR_ID"])) if baci else np.nan
        return {"treated_days_in_bin": len(t_rows), "treated_dams_in_bin": t_rows["AR_ID"].nunique(),
                "control_days_in_bin": len(c_rows) if baci else np.nan,
                "control_dams_in_bin": c_rows["AR_ID"].nunique() if baci else np.nan,
                "dams_both_roles_in_bin": both}

    points = []
    for kind, group in (("pre", pre), ("post", post)):
        for b, t in group:
            column = f"tb{kind}_{b}"
            points.append((b, {t: 1.0}, {"time_bin": b, "time_bin_period": kind,
                                         **counts(treated[treated[column] == 1],
                                                  control[control[column] == 1])}))
    if weights:
        points.append(("post average", weights, {"time_bin": "post average",
                                                 "time_bin_period": "post",
                                                 **counts(treated[treated["post"] == 1],
                                                          control[control["post"] == 1])}))
    return summary, weights, points


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
    time_points = None
    if SPECS[spec].get("time"):
        # Design 11: the headline effect is the post bins' average, weighted by
        # treated post days, so it is comparable to the binary effect.
        time_summary, weights, time_points = time_effects(result, rows, spec, effect, dams)
        est, se, lo, hi = linear_effect(result, weights, dams)
        out = {**base, "effect_term": "post average (treated-day weighted)",
               "effect_at_reference": est, "effect_se": se, "effect_ci_low": lo,
               "effect_ci_high": hi, "mde80_effect": mde(se, dams), **time_summary}
    else:
        est, se, lo, hi = linear_effect(result, {effect["base"]: 1.0}, dams)
        out = {**base, "effect_term": effect["base"], "effect_at_reference": est,
               "effect_se": se, "effect_ci_low": lo, "effect_ci_high": hi,
               "mde80_effect": mde(se, dams)}
    modifier, label = primary_modifier(effect)
    if modifier is not None and effect[modifier] in result.fe_params:
        s_est, s_se, s_lo, s_hi = linear_effect(result, {effect[modifier]: 1.0}, dams)
        out.update({"modifier": label, "modifier_term": effect[modifier],
                    "modifier_slope": s_est, "modifier_se": s_se,
                    "modifier_ci_low": s_lo, "modifier_ci_high": s_hi,
                    "modifier_p_value": 2 * stats.t.sf(abs(s_est / s_se), max(dams - 1, 1))
                    if s_se > 0 else np.nan,
                    "mde80_modifier": mde(s_se, dams),
                    "cov_effect_modifier": float(result.cov_params().loc[
                        effect["base"], effect[modifier]])})
        if modifier.startswith("m_"):
            ch = CHARACTERISTICS[label]
            scale = f"{TRANSFORM_LABELS[ch['transform']]} {ch['column']}".strip()
            out.update({"modifier_level": ch["level"],
                        "modifier_scale": f"per 1 SD of {scale} ({refs[f'{modifier}_sd']:.3g})"})
    warning_text = meta.get("warnings", "")
    out.update({"converged": bool(result.converged), **variance_components(result),
                **information_criteria(result, reml),
                "singular_warning_during_fit": "singular" in warning_text,
                "fit_check": mm.fit_check(warning_text), "error": ""})

    coefficients = []
    for name in result.fe_params.index:
        c_est, c_se, c_lo, c_hi = linear_effect(result, {name: 1.0}, dams)
        coefficients.append({"series": series, "spec": spec, "term": name, "estimate": c_est,
                             "se": c_se, "ci_low": c_lo, "ci_high": c_hi,
                             "p_value": 2 * stats.t.sf(abs(c_est / c_se), max(dams - 1, 1))
                             if c_se > 0 else np.nan})

    effects_at = []
    if time_points is not None:
        points = [(label, weights, reported) for label, weights, reported in time_points]
    else:
        points = []
        for label, values, reported in evaluation_points(rows, spec, refs):
            weights = {effect["base"]: 1.0}
            weights.update({effect[m]: v for m, v in values.items()
                            if m in effect and effect[m] in result.fe_params})
            points.append((label, weights, reported))
    for label, weights, reported in points:
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
                                           fixed_refs, settings.get("time_bins"))
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
# Audits and comparisons (designs 11, 12)
# --------------------------------------------------

def _design_rows(restricted, series, design, settings):
    """A design's rows before any modifier: the binary spec's sample."""
    rows = mm.model_rows(restricted[restricted["series"] == series], design, settings)
    return rows.dropna(subset=settings["covariates"])


def _role_counts(rows):
    treated, control = rows[rows["role"].eq("treated")], rows[rows["role"].eq("control")]
    return {"treated_dams": treated["AR_ID"].nunique(), "control_dams": control["AR_ID"].nunique(),
            "dams_both_roles": len(set(treated["AR_ID"]) & set(control["AR_ID"])),
            "treated_days": len(treated), "control_days": len(control)}


def time_coverage(restricted, settings, binnings=None):
    """Design 11 audit, before choosing bins: (by_year, by_bin). by_bin has,
    per series x design x binning, min_dams_post_bins = the fewest dams that
    identify any post bin (BACI: dams with treated AND control days in the
    bin; before/after: dams with treated days) and supported = that is at
    least min_dams."""
    binnings = binnings or TIME_BINNINGS
    by_year, by_bin = [], []
    for series in settings["series"]:
        for design in ("baci", "before_after"):
            rows = _design_rows(restricted, series, design, settings)
            if rows.empty:
                continue
            years = pd.to_numeric(rows["years_from_removal"], errors="coerce")
            for year, group in rows.groupby(years):
                by_year.append({"series": series, "design": design,
                                "years_from_removal": int(year), **_role_counts(group)})
            for name, bins in binnings.items():
                for label, lo, hi in bins:
                    counts = _role_counts(rows[years.between(lo, hi)])
                    by_bin.append({"series": series, "design": design, "binning": name,
                                   "bin": label, "years_lo": lo, "years_hi": hi,
                                   "reference": label == bins[0][0],
                                   "period": "post" if lo > 0 else "pre", **counts,
                                   "identifying_dams": counts["dams_both_roles"]
                                   if design == "baci" else counts["treated_dams"]})
    by_year, by_bin = pd.DataFrame(by_year), pd.DataFrame(by_bin)
    if len(by_bin):
        post = by_bin[by_bin["period"].eq("post")]
        least = post.groupby(["series", "design", "binning"])["identifying_dams"].min() \
            .rename("min_dams_post_bins")
        by_bin = by_bin.merge(least.reset_index(), on=["series", "design", "binning"], how="left")
        by_bin["supported"] = by_bin["min_dams_post_bins"] >= settings["min_dams"]
    return by_year, by_bin


def _dam_characteristics(rows):
    """One row per dam: each characteristic on its model scale (station-level
    ones averaged over the dam's treated pairs). Columns absent are skipped."""
    out = pd.DataFrame(index=pd.Index(rows["AR_ID"].unique(), name="AR_ID"))
    treated = rows[rows["role"].eq("treated")]
    for name, ch in CHARACTERISTICS.items():
        needed = [ch["column"]] + (["reservoir_polygon_timing"] if ch.get("pre_removal_only")
                                   else [])
        if not all(c in rows for c in needed):
            continue
        source = rows if ch["level"] == "dam" else treated
        values = characteristic_values(source, name)
        frame = source[["AR_ID", "station_id"]].assign(v=values.to_numpy())
        if ch["level"] == "dam":
            out[name] = frame.groupby("AR_ID")["v"].first()
        else:
            out[name] = frame.drop_duplicates(["AR_ID", "station_id"]).groupby("AR_ID")["v"].mean()
    return out


def modifier_audit(restricted, settings):
    """Design 12 audit: (availability, correlations). availability: per series
    x design x characteristic, the dams (and treated pairs) with a value, its
    spread in original units, and whether a fit is possible (>= min_dams).
    correlations: Spearman rho between characteristics over the dams of each
    design (any series), with the dams both are known for."""
    rows_out, dams_by_design = [], {}
    for series in settings["series"]:
        for design in ("baci", "before_after"):
            rows = _design_rows(restricted, series, design, settings)
            if rows.empty:
                continue
            dams_by_design.setdefault(design, []).append(rows)
            treated = rows[rows["role"].eq("treated")]
            n_dams = rows["AR_ID"].nunique()
            for name, ch in CHARACTERISTICS.items():
                base = {"series": series, "design": design, "characteristic": name,
                        "level": ch["level"], "column": ch["column"], "label": ch["label"],
                        "dams": n_dams}
                if ch["column"] not in rows:
                    rows_out.append({**base, "available": False,
                                     "note": f"{ch['column']} not in the input"})
                    continue
                source = rows if ch["level"] == "dam" else treated
                values = characteristic_values(source, name)
                known = source[values.notna().to_numpy()]
                dams_with = known["AR_ID"].nunique()
                units = (known.drop_duplicates("AR_ID") if ch["level"] == "dam"
                         else known.drop_duplicates(["AR_ID", "station_id"]))
                original = units[ch["column"]].astype(float)
                pairs = treated.drop_duplicates(["AR_ID", "station_id"])
                rows_out.append({
                    **base, "available": True, "dams_with_value": dams_with,
                    "dams_with_value_%": 100 * dams_with / n_dams if n_dams else np.nan,
                    "treated_pairs": len(pairs) if ch["level"] == "station" else np.nan,
                    "treated_pairs_with_value": len(units) if ch["level"] == "station" else np.nan,
                    "p10": original.quantile(0.1), "median": original.median(),
                    "p90": original.quantile(0.9),
                    "sd_model_scale": characteristic_values(units, name).std(),
                    "fit_possible": dams_with >= settings["min_dams"],
                    "note": "pre-removal mapped polygons only" if ch.get("pre_removal_only")
                    else ""})
    for name, why in UNAVAILABLE_CHARACTERISTICS.items():
        rows_out.append({"characteristic": name, "available": False, "note": why})
    correlations = []
    for design, frames in dams_by_design.items():
        table = _dam_characteristics(pd.concat(frames, ignore_index=True))
        names = list(table.columns)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = table[[a, b]].dropna()
                correlations.append({"design": design, "a": a, "b": b, "dams": len(both),
                                     "spearman_rho": both[a].corr(both[b], method="spearman")
                                     if len(both) > 2 else np.nan})
    return pd.DataFrame(rows_out), pd.DataFrame(correlations)


def combined_check(restricted, settings, names, dams_per_modifier=10, max_rho=0.7, max_vif=5.0):
    """Design 12: does each series x design support a combined model of
    `names`? Complete-case dams (every characteristic known), dams per
    modifier, the largest |Spearman rho| and VIF among them. justified needs
    >= dams_per_modifier dams per characteristic (and min_dams), |rho| <
    max_rho and VIF < max_vif. A rule of thumb, not a test."""
    out = []
    for series in settings["series"]:
        for design in ("baci", "before_after"):
            rows = _design_rows(restricted, series, design, settings)
            if rows.empty:
                continue
            table = _dam_characteristics(rows)
            missing = [n for n in names if n not in table]
            complete = table[[n for n in names if n in table]].dropna()
            n = len(complete) if not missing else 0
            rho = complete.corr(method="spearman").abs().where(
                ~np.eye(len(complete.columns), dtype=bool)).max().max() \
                if n > 2 and len(names) > 1 else np.nan
            vif = np.nan
            if n > len(names) + 1 and len(names) > 1:
                corr = np.corrcoef(complete.to_numpy(dtype=float), rowvar=False)
                try:
                    vif = float(np.max(np.diag(np.linalg.inv(corr))))
                except np.linalg.LinAlgError:
                    vif = np.inf
            need = max(settings["min_dams"], dams_per_modifier * len(names))
            reasons = ([f"missing columns {missing}"] if missing else []) \
                + ([f"{n} complete dams < {need}"] if n < need else []) \
                + ([f"|rho| {rho:.2f} >= {max_rho}"] if np.isfinite(rho) and rho >= max_rho
                   else []) \
                + ([f"VIF {vif:.1f} >= {max_vif}"] if np.isfinite(vif) and vif >= max_vif
                   else [])
            out.append({"series": series, "design": design, "characteristics": ", ".join(names),
                        "dams": rows["AR_ID"].nunique(), "complete_dams": n,
                        "dams_per_modifier": n / len(names), "max_abs_rho": rho, "max_vif": vif,
                        "justified": not reasons, "reasons": "; ".join(reasons)})
    return pd.DataFrame(out)


def bh_qvalues(p):
    """Benjamini-Hochberg q-values (NaN stays NaN)."""
    p = pd.Series(p, dtype=float)
    ok = p.dropna().sort_values()
    m = len(ok)
    if not m:
        return p
    q = (ok * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
    return q.reindex(p.index)


def modifier_screen(results):
    """Design 12: one row per series x characteristic fit, with the effect
    modification (per 1 SD), its CI, p and a BH q-value across the series x
    characteristics of each design and adjustment. q controls the share of
    false leads among the screen's 'detected' modifications; it doesn't make
    any one of them causal."""
    empty = pd.DataFrame(columns=["series", "design", "characteristic", "modifier_slope",
                                  "modifier_p_value", "q_value", "error"])
    if results.empty:
        return empty
    screen = results[results["spec"].str.contains(r"_xu?_") & ~results["spec"].str.endswith(
        "_combined")].copy()
    if screen.empty or "modifier_p_value" not in screen:
        return empty if screen.empty else screen.assign(q_value=np.nan)
    screen["characteristic"] = screen["spec"].str.replace(r"^.*?_xu?_", "", regex=True)
    screen["adjusted_for_distance_dilution"] = ~screen["spec"].str.contains("_xu_")
    screen["q_value"] = np.nan
    fitted = screen["error"].fillna("").eq("")
    for _, index in screen[fitted].groupby(["design", "adjusted_for_distance_dilution"]).groups.items():
        screen.loc[index, "q_value"] = bh_qvalues(screen.loc[index, "modifier_p_value"])
    keep = ["series", "design", "characteristic", "adjusted_for_distance_dilution", "spec", "dams",
            "dams_before_modifier", "sample_flag", "modifier_level", "modifier_scale",
            "modifier_slope", "modifier_ci_low", "modifier_ci_high", "modifier_p_value", "q_value",
            "mde80_modifier", "effect_at_reference", "effect_ci_low", "effect_ci_high",
            "re_choice", "converged", "singular", "fit_check", "error"]
    return screen[[c for c in keep if c in screen]].sort_values(
        ["design", "adjusted_for_distance_dilution", "series", "characteristic"])


def time_comparison(results):
    """Design 11: binary vs time-varying effect per series x design, fit on
    the same rows: estimates, CI widths, dams and days, the time tests, and
    (both fit by ML) the likelihood-ratio test and AIC / BIC."""
    out = []
    empty = pd.DataFrame(columns=["series", "design", "same_sample", "ml", "binary_effect",
                                  "time_average_effect", "time_change_p", "lrt_p"])
    if results.empty:
        return empty
    ok = results[results["error"].fillna("").eq("")]
    for (series, design), group in ok.groupby(["series", "design"]):
        group = group.set_index("spec")
        binary, time = f"{design}_binary", f"{design}_time"
        if binary not in group.index or time not in group.index:
            continue
        b, t = group.loc[binary], group.loc[time]
        same = bool(b["days"] == t["days"] and b["dams"] == t["dams"]
                    and b["stations"] == t["stations"])
        ml = not bool(b.get("reml", True)) and not bool(t.get("reml", True))
        same_re = b.get("re_structure") == t.get("re_structure")
        row = {"series": series, "design": design, "same_sample": same, "ml": ml,
               "same_random_effects": same_re, "binary_re_structure": b.get("re_structure"),
               "time_re_structure": t.get("re_structure"),
               "dams": b["dams"], "days": b["days"], "time_bins": t.get("time_bins"),
               "binary_effect": b["effect_at_reference"], "binary_ci_low": b["effect_ci_low"],
               "binary_ci_high": b["effect_ci_high"],
               "binary_ci_width": b["effect_ci_high"] - b["effect_ci_low"],
               "time_average_effect": t["effect_at_reference"],
               "time_average_ci_low": t["effect_ci_low"], "time_average_ci_high": t["effect_ci_high"],
               "time_average_ci_width": t["effect_ci_high"] - t["effect_ci_low"],
               "time_change_F": t.get("time_change_F"), "time_change_p": t.get("time_change_p"),
               "pretrend_p": t.get("pretrend_p"), "binary_re": b["re_choice"],
               "time_re": t["re_choice"], "unit": b["unit"], "sample_flag": b["sample_flag"]}
        row.update({"binary_fit_check": b.get("fit_check"), "time_fit_check": t.get("fit_check")})
        if same and ml and not same_re:
            row["lrt_note"] = (f"random effects differ ({b.get('re_structure')} vs "
                               f"{t.get('re_structure')}): likelihoods not comparable")
        elif same and ml and t["llf"] < b["llf"] - 1e-3:
            # The binary model is the time model with equal bins, so its maximum
            # likelihood can't be higher: the time fit stopped short of its maximum.
            row["lrt_note"] = (f"time fit's likelihood is below the nested binary fit's "
                               f"({t['llf']:.2f} < {b['llf']:.2f}): the optimizer stopped short; "
                               "no likelihood comparison")
        elif same and ml:
            stat = max(2 * (t["llf"] - b["llf"]), 0.0)
            df = int(t["n_params"] - b["n_params"])
            row.update({"lrt_stat": stat, "lrt_df": df,
                        "lrt_p": float(stats.chi2.sf(stat, df)) if df > 0 else np.nan,
                        "aic_binary": b["aic"], "aic_time": t["aic"],
                        "delta_aic_time_minus_binary": t["aic"] - b["aic"],
                        "delta_bic_time_minus_binary": t["bic"] - b["bic"]})
        else:
            row["lrt_note"] = ("refit with --ml (same random effects) to compare likelihoods"
                               if not ml else "samples differ; likelihoods not comparable")
        out.append(row)
    return pd.DataFrame(out) if out else empty


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
    parser.add_argument("--time-bins", default=None,
                        help="Time specs' bins (design 11): a TIME_BINNINGS name "
                             f"({', '.join(TIME_BINNINGS)}) or 'label:lo:hi;...', reference "
                             "first (default: the y1_y2-3_y4-5 binning).")
    parser.add_argument("--controls-known-outside", action="store_true",
                        help="Sensitivity: BACI controls only where upstream_in_former_reservoir "
                             "is False (docs/control_sensitivity_prespec.md).")
    parser.add_argument("--combined", nargs="+", choices=list(CHARACTERISTICS),
                        help="Characteristics of the *_x_combined specs (design 12).")
    return parser


def default_series():
    """The mixed model's series plus the diagnostic ones (design 10)."""
    return list(dict.fromkeys(mm.SETTINGS["series"] + DIAGNOSTIC_SERIES))


def carry_columns(specs, all_characteristics=False):
    """CARRY plus the warehouse columns the specs' characteristics need (all
    of them for the design 12 audit)."""
    extra = (characteristic_columns([s for s in SPECS if "_xu_" in s])
             if all_characteristics else characteristic_columns(specs))
    return list(dict.fromkeys(CARRY + extra))


def load_daily(args):
    """(daily values, settings, excluded dams) for the parsed options."""
    specs = getattr(args, "specs", None) or []
    if getattr(args, "combined", None):
        register_combined(args.combined)
    audit = getattr(args, "audit_only", False) or any("_x" in s for s in specs)
    settings = {**mm.SETTINGS, "max_downstream_km": args.max_km,
                "carry_columns": carry_columns(specs, all_characteristics=audit),
                "min_dams": args.min_dams, "exploratory_below": args.exploratory_below,
                "time_bins": parse_time_bins(args.time_bins) if getattr(args, "time_bins", None)
                else DEFAULT_TIME_BINS,
                "controls_known_outside_reservoir": bool(getattr(args, "controls_known_outside", False))}
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
                        help="Output subfolder: data/processed/mixed_model/<case>/<stage>/<run>/")
    parser.add_argument("--stage", default="stage3",
                        help="Output folder under mixed_model/<case>/ (stage3: the runs the "
                             "Stage 4 report reads; stage5: time and characteristic runs).")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--specs", nargs="+", choices=list(SPECS),
                        help=f"Default {DEFAULT_SPECS} unless --spec-sets is given.")
    parser.add_argument("--spec-sets", nargs="+", choices=list(SPEC_SETS),
                        help="Named groups of specs added to --specs (designs 11, 12).")
    parser.add_argument("--audit-only", action="store_true",
                        help="Write the time-coverage and characteristic audits and fit nothing.")
    parser.add_argument("--analysis-role", choices=["primary", "sensitivity"], default="primary",
                        help="Label written to every output table and settings.json: "
                             "sensitivity runs check a primary result, they don't replace it.")
    parser.add_argument("--common-sample", action="store_true",
                        help="Fit every spec on the rows all distance specs can use (design 5).")
    parser.add_argument("--ml", action="store_true",
                        help="Maximum likelihood instead of REML, for AIC / BIC comparisons.")
    args = parser.parse_args()
    args.specs = list(dict.fromkeys((args.specs or []) + [s for name in (args.spec_sets or [])
                                                          for s in SPEC_SETS[name]])) \
        or list(DEFAULT_SPECS)
    if any(SPECS[s].get("combined") for s in args.specs) and not args.combined:
        parser.error("*_x_combined specs need --combined <characteristic> ...")

    out_dir = args.output_dir or (mm.output_dir_for(args.case_study) / args.stage / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily, settings, exclude = load_daily(args)
    restricted, _ = restrict(daily, exclude, args.max_dar)
    time_specs = [s for s in args.specs if SPECS[s].get("time")]
    char_specs = [s for s in args.specs if "_x" in s]
    written = []
    if args.audit_only or time_specs:
        by_year, by_bin = time_coverage(restricted, settings)
        by_year.to_csv(out_dir / "time_coverage_by_year.csv", index=False)
        by_bin.to_csv(out_dir / "time_coverage_by_bin.csv", index=False)
        written += ["time_coverage_by_year.csv", "time_coverage_by_bin.csv"]
    if args.audit_only or char_specs:
        availability, correlations = modifier_audit(restricted, settings)
        availability.to_csv(out_dir / "modifier_audit.csv", index=False)
        correlations.to_csv(out_dir / "modifier_correlations.csv", index=False)
        written += ["modifier_audit.csv", "modifier_correlations.csv"]
    if args.combined:
        combined_check(restricted, settings, args.combined).to_csv(
            out_dir / "combined_check.csv", index=False)
        written.append("combined_check.csv")
    run_info = {**settings, "input": str(args.input), "run_name": args.run_name,
                "stage": args.stage, "specs": args.specs,
                "spec_definitions": {s: SPECS[s] for s in args.specs}, "max_dar": args.max_dar,
                "excluded_dams": exclude, "size_variable": args.size_variable,
                "reference_km": args.reference_km, "re": args.re,
                "common_sample": args.common_sample, "ml": args.ml, "combined": args.combined,
                "audit_only": args.audit_only, "analysis_role": args.analysis_role}
    if args.audit_only:
        (out_dir / "settings.json").write_text(json.dumps(run_info, indent=2, default=str))
        print(f"Audit only: wrote {', '.join(written)}, settings.json to {out_dir}")
        return
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

    role = {"analysis_role": args.analysis_role}
    results.assign(**role).to_csv(out_dir / "results.csv", index=False)
    coefficients.assign(**role).to_csv(out_dir / "coefficients.csv", index=False)
    effects_at.assign(**role).to_csv(out_dir / "effects_at.csv", index=False)
    if time_specs:
        time_comparison(results).assign(**role).to_csv(out_dir / "time_comparison.csv", index=False)
        written.append("time_comparison.csv")
    if char_specs:
        modifier_screen(results).assign(**role).to_csv(out_dir / "modifier_screen.csv", index=False)
        written.append("modifier_screen.csv")
    (out_dir / "settings.json").write_text(json.dumps(run_info, indent=2, default=str))
    print(f"\nWrote results.csv, coefficients.csv, effects_at.csv, dams_used.csv, "
          f"{''.join(w + ', ' for w in written)}settings.json to {out_dir}")


if __name__ == "__main__":
    main()
