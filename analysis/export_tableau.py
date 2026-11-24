"""
Small, flat CSV extracts for the Tableau dashboard. Read-only: reads the
warehouse, the frozen model outputs and the predictive-feasibility outputs,
fits nothing and writes only data/processed/tableau/<case_study>/ (a few MB,
meant to be copied to a laptop).

Usage (cluster: data/slurm/export_tableau.slurm):
    python analysis/export_tableau.py
    python analysis/export_tableau.py --model-dir <snapshot>/all_removals --no-geojson

Inputs
------
- data/warehouse/<case>/: dam_removal, wq_station, dam_station_link,
  analysis_measurement (primary eligibility rule), huc8_watershed
- the config's dam_source CSV (the whole American Rivers list, for the funnel)
- model outputs (--model-dir; default the 2026-11-18 frozen snapshot if it holds
  a copy, else the live data/processed/mixed_model/<case>/, which
  docs/model_freeze.md's sha256sum check confirms is identical):
  results.csv / dam_effects.csv (10 km mixed models), stage3/ (30 km distance
  models, the primary results), stage5/ (time since removal, characteristics)
- data/processed/predictive/<case>/ (cv_metrics, cv_decisions, cv_predictions,
  targets_reliability)

Outputs (one CSV per Tableau data source; a missing input skips its tables)
-------
dams.csv                 one row per dam in the American Rivers list: attributes,
                         original + snapped location, how far it got through the
                         pipeline (funnel_stage), which designs it is in
measure_funnel.csv       per measure: dams eligible downstream / with a control /
                         in each fitted model
stations.csv             linked stations, location, samples in the analysis table
dam_station_links.csv    dam-station pairs with both locations (MAKELINE), side,
                         distance, role and eligible measures
wq_monthly.csv           monthly median per dam x station x measure, eligible
                         pairs only (the dam explorer's time series)
results_10km.csv         before/after and BACI effect per measure (mixed models)
dam_effects.csv          each dam's own estimated effect (10 km before/after, BACI)
distance_summary.csv     primary distance models: effect at 5 km, distance slope,
                         classification, MDE, power at the threshold, dams needed
distance_effects.csv     effect at 1/5/10/20/30 km, primary + sensitivity runs
distance_curve.csv       effect on a fine distance grid (primary run), for ribbons
time_bins.csv            effect per time-since-removal bin (and the pre-trend bins)
modifier_screen.csv      characteristic x measure screen with q-values and findings
modifier_effects.csv     effect at the 10th / 50th / 90th percentile dam, per screen fit
predictability.csv       per measure: reliability, out-of-region R2, verdict
predicted_vs_actual.csv  per dam: actual vs predicted change (held-out region)
sesoi_thresholds.csv     the practical-significance thresholds
huc8_boundaries.geojson  simplified HUC8 outlines that hold a matched dam (optional)
README.txt               what was read, row counts, column notes

Units: every effect table has *_display columns in plain units (% change for
log10 series, else the series unit) next to the model-unit columns, and a
`classification` (detected / no meaningful change / inconclusive;
analysis/stage4_report.py). Thresholds are analyst-selected and provisional.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis import distance_models as dm
from analysis import mixed_model as mm
from analysis import stage4_report as rep
from etl.common.config import default_case_study, get_project_root, load_config
from etl.common.warehouse import warehouse_dir

FROZEN_TAG = "models-frozen-2026-11-18"

NAMES = {
    "water_temp": "Water temperature",
    "dissolved_oxygen": "Dissolved oxygen",
    "ph": "pH",
    "specific_conductance": "Specific conductance",
    "turbidity": "Turbidity",
    "phosphorus (Total)": "Total phosphorus",
    "orthophosphate (Dissolved)": "Dissolved orthophosphate",
    "tds": "Total dissolved solids",
    "ssc": "Suspended sediment (SSC)",
    "tss": "Total suspended solids",
    "discharge": "Discharge (sampling check)",
}
DESIGN_LABELS = {"before_after": "Before/after (downstream)", "baci": "BACI (upstream control)",
                 "before/after": "Before/after (downstream)", "BACI": "BACI (upstream control)"}
TREATED_WINDOW_KM = 30          # warehouse exposure window (distance models)
MIXED_KM = mm.SETTINGS["max_downstream_km"]
ROLE_RANK = {"excluded": 0, "control": 1, "treated": 2}
RANK_ROLE = {v: k for k, v in ROLE_RANK.items()}
FUNNEL = ["1. In American Rivers list", "2. Matched to a stream reach",
          "3. Monitoring station linked", "4. Eligible before/after data",
          "5. In a fitted model", "6. In a BACI model (upstream control)"]


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def is_log(series):
    s = pd.Series(series, dtype="object").astype(str).str.split(" (", regex=False).str[0]
    return s.isin(mm.SETTINGS["log_parameters"]).to_numpy()


def display_unit(series, unit=None):
    if is_log([series])[0]:
        return "% change"
    if unit is not None and isinstance(unit, str) and unit:
        return unit
    return mm.unit_label(series, False)


def add_display(frame, columns, series_col="series", unit_col=None):
    """<col>_display: % change for log10 series, else unchanged; plus display_unit,
    measure (plain name) and threshold columns."""
    if frame.empty:
        return frame
    out = frame.copy()
    log = is_log(out[series_col])
    for c in columns:
        if c in out:
            v = pd.to_numeric(out[c], errors="coerce")
            out[f"{c}_display"] = np.where(log, (10 ** v - 1) * 100, v)
    units = out[unit_col] if unit_col and unit_col in out else pd.Series([None] * len(out), index=out.index)
    out["display_unit"] = [display_unit(s, u) for s, u in zip(out[series_col], units)]
    out["measure"] = out[series_col].map(NAMES).fillna(out[series_col])
    out["threshold"] = out[series_col].map(rep.sesoi_label)
    sesoi = out[series_col].map(rep.SESOI).astype(float)
    out["threshold_display"] = np.where(log, (10 ** sesoi - 1) * 100, sesoi)
    return out


def classify(frame, est, lo, hi, series_col="series"):
    return [rep.classify(e, l, h, rep.SESOI.get(s)) for e, l, h, s in
            zip(pd.to_numeric(frame[est], errors="coerce"), pd.to_numeric(frame[lo], errors="coerce"),
                pd.to_numeric(frame[hi], errors="coerce"), frame[series_col])]


def ok_rows(frame):
    if frame.empty or "error" not in frame:
        return frame
    return frame[frame["error"].fillna("").astype(str).eq("")]


def read_csv(path, **kw):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False, **kw)


def stage_dir(base, run):
    """<base>/<run>__<re>, preferring auto (the reported runs)."""
    found = {p.name.split("__")[-1]: p for p in Path(base).glob(f"{run}__*")}
    return next((found[r] for r in ("auto", "slopes", "intercepts") if r in found), None)


def stage_csv(base, run, name):
    folder = stage_dir(base, run)
    return read_csv(folder / name) if folder else pd.DataFrame()


def default_model_dir(root, case):
    snapshot = root / "data" / "processed" / "frozen" / FROZEN_TAG / case
    if (snapshot / "results.csv").exists():
        return snapshot
    return mm.output_dir_for(case, root)


# --------------------------------------------------
# Warehouse tables
# --------------------------------------------------

def read_geo(path):
    """GeoParquet -> DataFrame with lat / lon of the (point) geometry."""
    import geopandas as gpd
    g = gpd.read_parquet(path)
    points = g.geom_type.eq("Point")
    if points.any():
        g["snapped_lat"] = g.geometry.y.where(points)
        g["snapped_lon"] = g.geometry.x.where(points)
    return pd.DataFrame(g.drop(columns=g.geometry.name))


def analysis_pairs(am):
    """Per dam x station x series: role, side, rows, years, from the analysis table."""
    am = am.assign(series=mm.series_name(am), role=mm.assign_roles(am, {**mm.SETTINGS,
                                                                       "max_downstream_km": None}))
    am["role_10km"] = mm.assign_roles(am)
    # A pair or month counts as treated if any of its rows is, else control.
    am["role_rank"] = am["role"].map(ROLE_RANK)
    am["role_10km_rank"] = am["role_10km"].map(ROLE_RANK)
    return am


def build_dams(ar, dam_removal, links, am, model_dams, suspects):
    ar = ar.copy()
    id_col = next(c for c in ar.columns if c.lower() == "ar_id")
    ar = ar.rename(columns={id_col: "AR_ID"})
    ar["AR_ID"] = ar["AR_ID"].astype(str).str.strip()
    ar = ar.drop_duplicates("AR_ID")
    keep = {"AR_ID": "AR_ID", "Dam_Name": "dam_name", "State": "state", "River": "river",
            "City_County": "city_county", "Year_Removed": "year_removed", "Year_Built": "year_built",
            "Dam_Height_ft": "dam_height_ft", "Dam_Length_ft": "dam_length_ft",
            "Original_Use": "original_use", "Type_Material": "type_material",
            "Reason_for_removal": "reason_for_removal", "Latitude": "latitude", "Longitude": "longitude"}
    lower = {c.lower(): c for c in ar.columns}
    cols = {lower[k.lower()]: v for k, v in keep.items() if k.lower() in lower}
    out = ar[list(cols)].rename(columns=cols)
    for c in ("year_removed", "year_built", "dam_height_ft", "dam_length_ft", "latitude", "longitude"):
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in ("year_removed", "year_built"):
        if c in out:
            out[c] = out[c].where(out[c] > 0)

    matched = dam_removal.assign(AR_ID=dam_removal["AR_ID"].astype(str))
    mcols = [c for c in ["AR_ID", "HUC8", "snapped_lat", "snapped_lon", "match_method",
                         "reach_gnis_name", "reach_totdasqkm"] if c in matched]
    out = out.merge(matched[mcols].rename(columns={"HUC8": "huc8",
                                                   "reach_totdasqkm": "drainage_area_km2"}),
                    on="AR_ID", how="left")
    out["matched_to_reach"] = out["AR_ID"].isin(set(matched["AR_ID"]))

    lk = links.assign(AR_ID=links["AR_ID"].astype(str),
                      km=pd.to_numeric(links["network_distance_km"], errors="coerce"))
    down = lk["relationship"].isin(["downstream", "same_reach"])
    counts = lk.groupby("AR_ID").agg(linked_stations=("station_id", "nunique"))
    counts["downstream_stations_30km"] = (lk[down & lk["km"].le(TREATED_WINDOW_KM)]
                                          .groupby("AR_ID")["station_id"].nunique())
    counts["upstream_stations"] = lk[lk["relationship"].eq("upstream")].groupby("AR_ID")["station_id"].nunique()
    out = out.merge(counts.reset_index(), on="AR_ID", how="left")
    for c in counts.columns:
        out[c] = out[c].fillna(0).astype(int)

    if len(am):
        treated = am[am["role"].eq("treated")]
        control = am[am["role"].eq("control")]
        elig = treated.groupby("AR_ID")["series"].agg(lambda s: ", ".join(sorted(set(s))))
        out["eligible_measures_downstream"] = out["AR_ID"].map(elig).fillna("")
        out["n_eligible_measures"] = out["AR_ID"].map(treated.groupby("AR_ID")["series"].nunique()).fillna(0).astype(int)
        out["has_upstream_control_data"] = out["AR_ID"].isin(set(control["AR_ID"]))
        out["first_sample_year"] = out["AR_ID"].map(am.groupby("AR_ID")["year"].min())
        out["last_sample_year"] = out["AR_ID"].map(am.groupby("AR_ID")["year"].max())
        out["eligible_data"] = out["AR_ID"].isin(set(treated["AR_ID"]))
    else:
        out["eligible_data"] = False
        out["has_upstream_control_data"] = False
    for name, ids in model_dams.items():
        out[name] = out["AR_ID"].isin(ids)
    none = pd.Series(False, index=out.index)
    any_model = out[list(model_dams)].any(axis=1) if model_dams else none
    baci_cols = [c for c in model_dams if "baci" in c]
    baci = out[baci_cols].any(axis=1) if baci_cols else none
    out["in_any_model"] = any_model
    out["suspect_match_excluded"] = out["AR_ID"].isin(suspects)

    stage = np.select(
        [baci, any_model, out["eligible_data"], out["linked_stations"] > 0, out["matched_to_reach"]],
        FUNNEL[5:0:-1], FUNNEL[0])
    out["funnel_stage"] = stage
    out["funnel_level"] = [FUNNEL.index(s) + 1 for s in stage]
    out["map_lat"] = out["snapped_lat"].fillna(out["latitude"]) if "snapped_lat" in out else out["latitude"]
    out["map_lon"] = out["snapped_lon"].fillna(out["longitude"]) if "snapped_lon" in out else out["longitude"]
    out["dam_age_at_removal"] = out["year_removed"] - out["year_built"]
    return out


def model_dam_sets(model_dir):
    """{flag column: set of AR_IDs} per design, from dam_effects (10 km) and the
    primary distance run's dams_used (30 km)."""
    sets = {}
    de = read_csv(model_dir / "dam_effects.csv", dtype={"AR_ID": str})
    for model in ("before_after", "baci"):
        if len(de):
            sets[f"in_{model}_10km"] = set(de.loc[de["model"].eq(model), "AR_ID"])
    used = stage_csv(model_dir / "stage3", "primary", "dams_used.csv")
    if len(used):
        used["AR_ID"] = used["AR_ID"].astype(str)
        for design in ("before_after", "baci"):
            sets[f"in_{design}_distance_30km"] = set(used.loc[used["design"].eq(design), "AR_ID"])
    return sets


def build_measure_funnel(am, model_dir):
    rows = []
    if len(am):
        for series, g in am.groupby("series"):
            t, c = g[g["role"].eq("treated")], g[g["role"].eq("control")]
            t10 = g[g["role_10km"].eq("treated")]
            rows += [
                {"series": series, "step": 1, "stage": "Eligible downstream data (within 30 km)",
                 "dams": t["AR_ID"].nunique()},
                {"series": series, "step": 2, "stage": "Eligible downstream data (within 10 km)",
                 "dams": t10["AR_ID"].nunique()},
                {"series": series, "step": 3, "stage": "Also has an upstream control",
                 "dams": len(set(t["AR_ID"]) & set(c["AR_ID"]))}]
    res = ok_rows(read_csv(model_dir / "results.csv"))
    for _, r in res.iterrows():
        rows.append({"series": r["series"], "step": 4 if r["model"] == "before_after" else 6,
                     "stage": f"In the 10 km {DESIGN_LABELS[r['model']]} model", "dams": r["dams"]})
    try:
        results, _, _ = rep.load_runs(model_dir / "stage3")
        prim = ok_rows(results[(results["run"] == rep.PRIMARY[0]) & (results["re"] == rep.PRIMARY[1])
                               & results["spec"].isin(rep.DISTANCE_SPECS)])
        for _, r in prim.iterrows():
            rows.append({"series": r["series"], "step": 5 if r["design"] == "before_after" else 7,
                         "stage": f"In the 30 km distance model, {DESIGN_LABELS[r['design']]}",
                         "dams": r["dams"]})
    except FileNotFoundError:
        pass
    out = pd.DataFrame(rows)
    return add_display(out, []).sort_values(["series", "step"]) if len(out) else out


def build_stations(stations, am):
    keep = [c for c in ["station_id", "station_name", "station_type", "organization_name", "latitude",
                        "longitude", "snapped_lat", "snapped_lon", "huc8", "reach_gnis_name",
                        "reach_totdasqkm", "qc_far_snap", "qc_non_riverine_name"] if c in stations]
    out = stations[keep].copy()
    if len(am):
        g = am.groupby("station_id")
        out["analysis_measurements"] = out["station_id"].map(g.size()).fillna(0).astype(int)
        out["measures"] = out["station_id"].map(g["series"].agg(lambda s: ", ".join(sorted(set(s))))).fillna("")
        out["first_year"] = out["station_id"].map(g["year"].min())
        out["last_year"] = out["station_id"].map(g["year"].max())
    return out


def build_links(links, dams, stations, am):
    out = links[["AR_ID", "station_id", "relationship", "network_distance_km"]].copy()
    out["AR_ID"] = out["AR_ID"].astype(str)
    d = dams.set_index("AR_ID")
    s = stations.set_index("station_id")
    out["dam_lat"], out["dam_lon"] = out["AR_ID"].map(d["map_lat"]), out["AR_ID"].map(d["map_lon"])
    s_lat = s["snapped_lat"].fillna(s["latitude"]) if "snapped_lat" in s else s["latitude"]
    s_lon = s["snapped_lon"].fillna(s["longitude"]) if "snapped_lon" in s else s["longitude"]
    out["station_lat"], out["station_lon"] = out["station_id"].map(s_lat), out["station_id"].map(s_lon)
    out["within_30km"] = pd.to_numeric(out["network_distance_km"], errors="coerce").le(TREATED_WINDOW_KM)
    if len(am):
        g = am.groupby(["AR_ID", "station_id"])
        pair = pd.DataFrame({
            "side": g["side"].agg(lambda x: x.dropna().mode().iat[0] if x.notna().any() else None),
            "role": g["role_rank"].max().map(RANK_ROLE),
            "role_10km": g["role_10km_rank"].max().map(RANK_ROLE),
            "eligible_measures": g["series"].agg(lambda x: ", ".join(sorted(set(x)))),
            "analysis_measurements": g.size(),
        }).reset_index()
        out = out.merge(pair, on=["AR_ID", "station_id"], how="left")
        out["eligible_pair"] = out["eligible_measures"].notna()
        out["role"] = out["role"].fillna("not eligible")
        out["role_10km"] = out["role_10km"].fillna("not eligible")
    return out


def build_wq_monthly(am):
    if am.empty:
        return am
    keep = am[am["series"].isin(mm.SETTINGS["series"])].dropna(subset=["result_value_std"])
    key = ["AR_ID", "station_id", "series", "year", "month"]
    out = (keep.groupby(key, as_index=False)
           .agg(median_value=("result_value_std", "median"), samples=("result_value_std", "size"),
                period=("period", "first"), years_from_removal=("years_from_removal", "first"),
                removal_year=("removal_year", "first"), side=("side", "first"),
                role_rank=("role_rank", "max"),
                network_distance_km=("network_distance_km", "first"),
                unit=("result_unit_std", "first")))
    out["role"] = out.pop("role_rank").map(RANK_ROLE)
    out["month_date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=15))
    out["measure"] = out["series"].map(NAMES).fillna(out["series"])
    return out


# --------------------------------------------------
# Model tables
# --------------------------------------------------

def build_results_10km(model_dir):
    res = ok_rows(read_csv(model_dir / "results.csv"))
    if res.empty:
        return res
    res = res[res["estimate"].notna()].copy()
    res["classification"] = classify(res, "estimate", "ci_low", "ci_high")
    res["design"] = res["model"].map(DESIGN_LABELS)
    res["exploratory"] = res["dams"] < 15
    keep = ["series", "model", "design", "estimate", "ci_low", "ci_high", "se", "p_value",
            "effect_sd_between_dams", "unit", "dams", "stations", "days", "re_structure",
            "classification", "exploratory"]
    return add_display(res[[c for c in keep if c in res]],
                       ["estimate", "ci_low", "ci_high", "effect_sd_between_dams"], unit_col="unit")


def build_dam_effects(model_dir, dams):
    de = read_csv(model_dir / "dam_effects.csv", dtype={"AR_ID": str})
    if de.empty:
        return de
    d = dams.set_index("AR_ID")
    for c in ("dam_name", "state", "map_lat", "map_lon", "year_removed", "dam_height_ft"):
        if c in d:
            de[c] = de["AR_ID"].map(d[c])
    de["design"] = de["model"].map(DESIGN_LABELS)
    return add_display(de, ["dam_effect"])


def build_distance(model_dir):
    try:
        results, effects, _ = rep.load_runs(model_dir / "stage3")
    except FileNotFoundError:
        return {}
    summary = rep.model_summary(results)
    prim = (results["run"] == rep.PRIMARY[0]) & (results["re"] == rep.PRIMARY[1])
    s = summary[prim.to_numpy() & summary["spec"].isin(rep.DISTANCE_SPECS).to_numpy()].copy()
    s = ok_rows(s)
    se = results.loc[s.index, "effect_se"].astype(float) if "effect_se" in results else np.nan
    s["effect_se"] = se
    s["power_at_threshold"] = dm.power(s["sesoi"].astype(float), s["effect_se"], s["dams"].astype(float))
    s["mde80_over_threshold"] = s["mde80_effect"] / s["sesoi"]
    s["dams_needed_rough"] = [dm.dams_needed(m, t, n) for m, t, n in
                              zip(s["mde80_effect"], s["sesoi"], s["dams"])]
    s["dams_needed_rough"] = s["dams_needed_rough"].replace(np.inf, np.nan)
    s["design"] = s["design"].map(DESIGN_LABELS).fillna(s["design"])
    s["exploratory"] = s["dams"] < 15
    keep = ["series", "spec", "design", "purpose", "dams", "stations", "km_min", "km_max", "unit",
            "re_choice", "effect_at_reference", "effect_ci_low", "effect_ci_high", "effect_class",
            "modifier_slope", "modifier_ci_low", "modifier_ci_high", "slope_class", "mde80_effect",
            "mde80_over_threshold", "power_at_threshold", "dams_needed_rough", "exploratory"]
    s = add_display(s[keep], ["effect_at_reference", "effect_ci_low", "effect_ci_high", "mde80_effect"],
                    unit_col="unit")

    by = rep.effects_by_distance(effects)
    if len(by):
        by = by[by["re"].eq("auto") & by["run"].isin(rep.SENSITIVITY_RUNS)
                & by["spec"].isin(rep.DISTANCE_SPECS) & by["distance_km"].notna()]
        by = by.assign(run_label=np.where(by["primary"], "primary", by["run"]),
                       design=by["design"].map(DESIGN_LABELS))
        keep = ["series", "design", "spec", "run_label", "primary", "distance_km", "estimate", "ci_low",
                "ci_high", "dams", "effect_class", "purpose"]
        by = add_display(by[[c for c in keep if c in by]], ["estimate", "ci_low", "ci_high"])

    curve = rep.effects_curve(results)
    if len(curve):
        curve = curve[(curve["run"] == rep.PRIMARY[0]) & (curve["re"] == rep.PRIMARY[1])]
        curve = curve.assign(design=curve["design"].map(DESIGN_LABELS))
        curve = add_display(curve.drop(columns=["run", "re"]), ["estimate", "ci_low", "ci_high"],
                            unit_col="unit")
    return {"distance_summary": s, "distance_effects": by, "distance_curve": curve}


def build_time(model_dir):
    base = model_dir / "stage5"
    at = stage_csv(base, "time", "effects_at.csv")
    pre = stage_csv(base, "time_pretrend", "effects_at.csv")
    parts = []
    if len(at):
        parts.append(at[at["time_bin_period"].eq("post")].assign(check="main"))
    if len(pre):
        parts.append(pre[pre["time_bin_period"].eq("pre")].assign(check="pre-trend (should be ~0)"))
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[out["spec"].isin(["baci_time", "before_after_time"])]
    out["design"] = out["spec"].str.replace("_time", "").map(DESIGN_LABELS)
    out["classification"] = classify(out, "estimate", "ci_low", "ci_high")
    out["bin_order"] = out.groupby(["series", "spec", "check"]).cumcount()
    keep = ["series", "design", "check", "time_bin", "time_bin_period", "bin_order", "estimate", "ci_low",
            "ci_high", "dams", "treated_dams_in_bin", "dams_both_roles_in_bin", "classification"]
    return add_display(out[[c for c in keep if c in out]], ["estimate", "ci_low", "ci_high"])


def build_modifiers(model_dir, q_max=0.10):
    base = model_dir / "stage5"
    screen = ok_rows(stage_csv(base, "modifiers", "modifier_screen.csv"))
    at = stage_csv(base, "modifiers", "effects_at.csv")
    if len(screen):
        screen = screen.copy()
        screen["ci_excludes_0"] = (screen["modifier_ci_low"] > 0) | (screen["modifier_ci_high"] < 0)
        screen["finding"] = screen["ci_excludes_0"] & (screen["q_value"] < q_max)
        screen["strength"] = np.select([screen["finding"], screen["ci_excludes_0"]],
                                       ["finding (q < 0.1)", "hint (not significant after correction)"],
                                       "no clear link")
        screen["slope_per_threshold"] = screen["modifier_slope"] / screen["series"].map(rep.SESOI)
        screen["design"] = screen["design"].map(DESIGN_LABELS).fillna(screen["design"])
        screen = add_display(screen, ["effect_at_reference", "effect_ci_low", "effect_ci_high"])
    if len(at):
        at = at[at["characteristic_percentile"].notna()]
        at = at.assign(design=np.where(at["spec"].str.startswith("baci"), DESIGN_LABELS["baci"],
                                       DESIGN_LABELS["before_after"]))
        keep = ["series", "design", "spec", "characteristic", "characteristic_percentile",
                "characteristic_value", "estimate", "ci_low", "ci_high", "dams"]
        at = add_display(at[[c for c in keep if c in at]], ["estimate", "ci_low", "ci_high"])
    return screen, at


def build_predictive(pred_dir):
    metrics = read_csv(pred_dir / "cv_metrics.csv")
    decisions = read_csv(pred_dir / "cv_decisions.csv")
    reliab = read_csv(pred_dir / "targets_reliability.csv")
    preds = read_csv(pred_dir / "cv_predictions.csv", dtype={"AR_ID": str, "huc2": str, "fold": str})
    if metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    loro = metrics[metrics["analysis"].eq("primary") & metrics["scheme"].eq("loro")]
    ridge = loro[loro["model"].eq("C")].set_index("series")
    mean_only = loro[loro["model"].eq("A")].set_index("series")
    out = pd.DataFrame(index=ridge.index)
    for c in ("dams", "r2", "r2_boot_q05", "r2_boot_q95", "mae_natural"):
        if c in ridge:
            out[c] = ridge[c]
    if "mae_natural" in mean_only:
        out["mae_natural_mean_only"] = mean_only["mae_natural"]
    out["r2_bar"] = 0.05
    if len(reliab):
        rel = (reliab[reliab["version"].eq("primary_10km") & reliab["se_kind"].eq("se_total")]
               .groupby("series")["reliability"].agg(["median", "min", "max"]))
        out = out.join(rel.add_prefix("reliability_"))
    if len(decisions):
        go = decisions[decisions["model"].eq("C")].copy()
        go["go"] = go["go"].astype(str).str.lower().eq("true")
        out["predictable"] = go[go["analysis"].eq("primary")].set_index("series")["go"]
        sens = go[~go["analysis"].eq("primary")].groupby("series")["go"].agg(["sum", "size"])
        out["sensitivity_passed"], out["sensitivity_total"] = sens["sum"], sens["size"]
        out["verdict"] = np.where(out["predictable"].fillna(False), "Predictable", "Not predictable")
    out = add_display(out.reset_index(), [])
    if len(preds):
        preds = preds[preds["analysis"].eq("primary")]
        keep = [c for c in ["AR_ID", "series", "huc2", "fold", "y", "pred_A", "pred_C"] if c in preds]
        preds = add_display(preds[keep], [c for c in ("y", "pred_C", "pred_A") if c in keep])
    return out, preds


def build_huc8_geojson(wh, dam_huc8s, path, tolerance=0.005):
    import geopandas as gpd
    g = gpd.read_parquet(wh / "huc8_watershed.parquet")
    g = g[g["huc8"].astype(str).isin(dam_huc8s)].copy()
    g["geometry"] = g.geometry.simplify(tolerance, preserve_topology=True)
    g[["huc8", "name", "states", "areasqkm", "geometry"]].to_file(path, driver="GeoJSON")
    return len(g)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case-study", default=default_case_study())
    parser.add_argument("--model-dir", type=Path, help="Default: the frozen snapshot copy, else the live dir")
    parser.add_argument("--predictive-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-geojson", action="store_true")
    args = parser.parse_args()

    root = get_project_root()
    cfg = load_config(args.case_study, root)
    wh = warehouse_dir(cfg)
    model_dir = args.model_dir or default_model_dir(root, cfg.name)
    pred_dir = args.predictive_dir or root / "data" / "processed" / "predictive" / cfg.name
    out_dir = args.output_dir or root / "data" / "processed" / "tableau" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Case study: {cfg.name}\nWarehouse: {wh}\nModels: {model_dir}\nPredictive: {pred_dir}\n"
          f"Output: {out_dir}\n")

    ar = pd.read_csv(cfg.path("dam_source"), low_memory=False)
    dam_removal = read_geo(wh / "dam_removal.parquet")
    stations = read_geo(wh / "wq_station.parquet")
    links = pd.read_parquet(wh / "dam_station_link.parquet")
    am_cols = ["AR_ID", "station_id", "parameter_std", "sample_fraction_std", "side", "period",
               "exposure_status", "upstream_in_former_reservoir", "network_distance_km", "year", "month",
               "years_from_removal", "removal_year", "result_value_std", "result_unit_std",
               "has_same_day_discharge"] + mm.SETTINGS["exclude_flags"]
    am = pd.read_parquet(wh / "analysis_measurement.parquet", columns=list(dict.fromkeys(am_cols)))
    am["AR_ID"] = am["AR_ID"].astype(str)
    am = analysis_pairs(am)
    suspects = set(read_csv(root / "config" / "suspect_dam_matches.csv", dtype=str).get("AR_ID", pd.Series(dtype=str)))

    tables = {}
    dams = build_dams(ar, dam_removal, links, am, model_dam_sets(model_dir), suspects)
    tables["dams"] = dams
    tables["measure_funnel"] = build_measure_funnel(am, model_dir)
    tables["stations"] = build_stations(stations, am)
    tables["dam_station_links"] = build_links(links, dams, stations, am)
    tables["wq_monthly"] = build_wq_monthly(am)
    tables["results_10km"] = build_results_10km(model_dir)
    tables["dam_effects"] = build_dam_effects(model_dir, dams)
    tables.update(build_distance(model_dir))
    tables["time_bins"] = build_time(model_dir)
    tables["modifier_screen"], tables["modifier_effects"] = build_modifiers(model_dir)
    tables["predictability"], tables["predicted_vs_actual"] = build_predictive(pred_dir)
    tables["sesoi_thresholds"] = add_display(rep.sesoi_thresholds(), [])

    lines = [f"Tableau extracts for {cfg.name}, written {dt.datetime.now():%Y-%m-%d %H:%M}",
             f"Models read from: {model_dir}", f"Predictive outputs: {pred_dir}",
             "Effects: *_display columns are plain units (% change for log10 measures).",
             "Thresholds are analyst-selected and provisional (docs/practical_significance_thresholds.md).",
             "Fewer than 15 dams = exploratory.", ""]
    for name, frame in tables.items():
        if frame is None or len(frame) == 0:
            print(f"  {name}: SKIPPED (no input)")
            lines.append(f"{name}.csv: not written (input missing)")
            continue
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        size = path.stat().st_size / 1e6
        print(f"  {name}.csv: {len(frame):,} rows, {size:.1f} MB")
        lines.append(f"{name}.csv: {len(frame):,} rows, {size:.1f} MB")

    if not args.no_geojson:
        try:
            huc8s = set(dams.loc[dams["matched_to_reach"], "huc8"].dropna().astype(str))
            n = build_huc8_geojson(wh, huc8s, out_dir / "huc8_boundaries.geojson")
            size = (out_dir / "huc8_boundaries.geojson").stat().st_size / 1e6
            print(f"  huc8_boundaries.geojson: {n:,} HUC8s, {size:.1f} MB")
            lines.append(f"huc8_boundaries.geojson: {n:,} HUC8s, {size:.1f} MB")
        except Exception as exc:          # optional layer; never fail the export for it
            print(f"  huc8_boundaries.geojson: SKIPPED ({exc})")

    funnel = dams["funnel_stage"].value_counts().reindex(FUNNEL).fillna(0).astype(int)
    lines += ["", "Dams by furthest funnel stage:"] + [f"  {k}: {v:,}" for k, v in funnel.items()]
    (out_dir / "README.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-len(funnel) - 1:]))
    total = sum(p.stat().st_size for p in out_dir.iterdir() if p.is_file()) / 1e6
    print(f"\nTotal: {total:.1f} MB in {out_dir}")


if __name__ == "__main__":
    main()
