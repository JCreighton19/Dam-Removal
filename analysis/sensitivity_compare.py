"""
Primary vs eligibility-sensitivity comparison (docs/eligibility_sensitivity_prespec.md).

The interpretation criteria were fixed before any sensitivity model was fit;
this module only applies them, for the results notebooks
(notebooks/analysis/distance_model_results.ipynb section 11,
mixed_model_results.ipynb section 7). Nothing here fits or writes anything.

Per estimate: the verdict (stage4_report.classify: detected / no meaningful
change / inconclusive, with the analyst-selected SESOI), then a label of the
sensitivity estimate against the primary one:

    robust              same verdict and same direction
    sharper             primary inconclusive -> sensitivity decided, same direction
    weaker              primary decided -> sensitivity inconclusive, same direction
    conflicting         direction flips while either CI excludes 0, or
                        detected <-> no meaningful change
    direction unstable  direction flips, both inconclusive

Material improvement for a measure: the sensitivity sample makes a model
fittable (>= MIN_DAMS) or drops the exploratory label (>= EXPLORATORY_BELOW),
or narrows the 95% CI by >= 20% (width ratio <= 0.8), or brings the minimum
detectable effect below the SESOI.

Added dams vs primary dams: standardized mean differences on dam and station
characteristics; |SMD| >= 0.25 is flagged as systematic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis import stage4_report as rep

MIN_DAMS, EXPLORATORY_BELOW = 10, 15
WIDTH_RATIO_MATERIAL = 0.8
SMD_FLAG = 0.25
DECIDED = ("detected", "no meaningful change")
# Characteristic -> (analysis_measurement column, transform, level). Level
# "station": the median over the dam's downstream stations.
CHARACTERISTICS = {
    "log10 dam height (ft)": ("dam_height_ft", "log10", "dam"),
    "log10 dam drainage area (km2)": ("dam_drainage_area_km2", "log10", "dam"),
    "removal year": ("removal_year", "identity", "dam"),
    "log10 reservoir volume proxy": ("reservoir_volume_proxy", "log10", "dam"),
    "downstream distance (km)": ("network_distance_km", "identity", "station"),
    "% forest": ("pct_forest", "identity", "station"),
    "% developed": ("pct_developed", "identity", "station"),
    "% agriculture": ("pct_agriculture", "identity", "station"),
}


def verdict(estimate, low, high, sesoi):
    return rep.classify(estimate, low, high, sesoi)


def label(p_verdict, p_estimate, p_low, p_high, s_verdict, s_estimate, s_low, s_high):
    """The pre-specified label of a sensitivity estimate against the primary one."""
    if not np.isfinite([p_estimate, s_estimate]).all() or not p_verdict or not s_verdict:
        return "not comparable"
    flipped = np.sign(p_estimate) * np.sign(s_estimate) < 0     # an estimate of exactly 0 has no direction
    excludes = (p_low > 0 or p_high < 0) or (s_low > 0 or s_high < 0)
    if flipped:
        return "conflicting" if excludes else "direction unstable"
    if {p_verdict, s_verdict} == set(DECIDED):
        return "conflicting"
    if p_verdict == s_verdict:
        return "robust"
    if p_verdict == "inconclusive" and s_verdict in DECIDED:
        return "sharper"
    return "weaker"


def compare(primary, sensitivity, keys, est, low, high, sesoi=None, dams="dams", mde=None):
    """One row per key present in either table: dams, estimates, CIs and
    verdicts of both, the label, the CI width ratio and material-improvement
    flags. sesoi: {series: threshold} (None: detected / inconclusive only).
    Rows not fit in one table (missing, or with an error) are kept with NaN."""
    cols = list(dict.fromkeys(keys + [dams, est, low, high] + ([mde] if mde else [])))
    p = primary[[c for c in cols if c in primary]].rename(columns={dams: "dams", est: "est",
                                                                  low: "low", high: "high", mde: "mde"})
    s = sensitivity[[c for c in cols if c in sensitivity]].rename(columns={dams: "dams", est: "est",
                                                                          low: "low", high: "high", mde: "mde"})
    out = p.merge(s, on=keys, how="outer", suffixes=("_primary", "_sensitivity"))
    threshold = out["series"].map(sesoi or {}) if "series" in out else pd.Series(np.nan, index=out.index)
    for side in ("primary", "sensitivity"):
        out[f"verdict_{side}"] = [
            verdict(e, lo, hi, t if sesoi else None) if pd.notna(e) else ""
            for e, lo, hi, t in zip(out[f"est_{side}"], out[f"low_{side}"], out[f"high_{side}"], threshold)]
    out["label"] = [label(*r) for r in out[["verdict_primary", "est_primary", "low_primary", "high_primary",
                                             "verdict_sensitivity", "est_sensitivity", "low_sensitivity",
                                             "high_sensitivity"]].itertuples(index=False)]
    out["ci_width_ratio"] = ((out["high_sensitivity"] - out["low_sensitivity"])
                             / (out["high_primary"] - out["low_primary"]))
    dp, ds = out["dams_primary"].fillna(0), out["dams_sensitivity"].fillna(0)
    out["now_fittable"] = (dp < MIN_DAMS) & (ds >= MIN_DAMS)
    out["no_longer_exploratory"] = (dp < EXPLORATORY_BELOW) & (ds >= EXPLORATORY_BELOW)
    out["ci_narrower_20pct"] = out["ci_width_ratio"] <= WIDTH_RATIO_MATERIAL
    if mde and "mde_primary" in out:
        out["mde_now_below_sesoi"] = (out["mde_primary"] >= threshold) & (out["mde_sensitivity"] < threshold)
    flags = [c for c in ("now_fittable", "no_longer_exploratory", "ci_narrower_20pct",
                         "mde_now_below_sesoi") if c in out]
    out["material_improvement"] = out[flags].fillna(False).any(axis=1)
    return out


def dam_characteristics(rows):
    """One row per dam from analysis_measurement rows: CHARACTERISTICS on
    their comparison scale (station-level ones: median over downstream rows)."""
    out = pd.DataFrame(index=pd.Index(rows["AR_ID"].astype(str).unique(), name="AR_ID"))
    by_dam = rows.assign(AR_ID=rows["AR_ID"].astype(str)).groupby("AR_ID")
    down = rows[rows["side"].eq("downstream")].assign(AR_ID=lambda f: f["AR_ID"].astype(str)) \
        .drop_duplicates(["AR_ID", "station_id"])
    for name, (column, transform, level) in CHARACTERISTICS.items():
        if column not in rows:
            continue
        values = (by_dam[column].first() if level == "dam"
                  else down.groupby("AR_ID")[column].median()).astype(float)
        if transform == "log10":
            values = np.log10(values.where(values > 0))
        out[name] = values
    return out


def smd(a, b):
    """Standardized mean difference (b - a) / pooled SD; NaN with < 2 values."""
    a, b = pd.Series(a, dtype=float).dropna(), pd.Series(b, dtype=float).dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var() + (len(b) - 1) * b.var()) / (len(a) + len(b) - 2))
    diff = b.mean() - a.mean()
    if pooled > 0:
        return diff / pooled
    return np.sign(diff) * np.inf if diff else 0.0     # no spread: any difference is systematic


def compare_dams(characteristics, primary_ids, added_ids):
    """Per characteristic: dams with a value, medians and the SMD of added vs
    primary dams; systematic = |SMD| >= SMD_FLAG."""
    p = characteristics.reindex(sorted(set(primary_ids)))
    a = characteristics.reindex(sorted(set(added_ids)))
    rows = []
    for name in characteristics.columns:
        d = smd(p[name], a[name])
        rows.append({"characteristic": name, "primary_dams": int(p[name].notna().sum()),
                     "added_dams": int(a[name].notna().sum()), "primary_median": p[name].median(),
                     "added_median": a[name].median(), "smd_added_minus_primary": d,
                     "systematic": bool(pd.notna(d) and abs(d) >= SMD_FLAG)})
    return pd.DataFrame(rows)
