"""
Stage 4: how much do the primary BACI distance results depend on individual
dams, and what drives the pH result? Read-only on the warehouse; outputs go to
data/processed/mixed_model/<case_study>/stage4/influence/.

Usage (cluster: data/slurm/fit_stage4.slurm):
    python analysis/influence.py --re auto --bootstrap 200 --workers 8

1. Full fit: analysis/distance_models.py fit_one for each series of --specs
   (default baci_distance), with the same rows, exclusions and options. Its
   random-effects structure (the auto choice) and reference values (ratio,
   size standardization) are then held fixed, so every refit below differs
   from it only in which dams it uses.
2. Leave-one-dam-out (lodo.csv, lodo_summary.csv): one refit per dam left
   out. For each dam: the effect at the reference (5 km), the distance slope,
   the effects at 1-30 km, and the change from the full fit in full-fit
   standard errors. A dam is influential when leaving it out moves the
   effect at the reference or the slope by at least one full-fit SE, or
   changes whether the 95% interval excludes zero.
3. Dam bootstrap (bootstrap_draws.csv, bootstrap_summary.csv): --bootstrap B
   resamples of dams with replacement (all of a dam's stations and days go
   together; a dam drawn twice is relabeled so it counts as two dams), refit
   each, percentile 95% interval and bootstrap SE next to the model interval.
   Draws whose fit fails are counted, not hidden.
4. BACI components (components_<series>.csv, components_summary.csv,
   trajectory_<series>.csv), for --components-series (default ph): each
   BACI dam's mean change after removal at its upstream controls and at its
   downstream stations (station means of the daily values, then averaged per
   dam and role), so a BACI effect can be read as "downstream rose", "upstream
   fell", or both; the model's own split (post = control change; post +
   post_downstream = downstream change at 5 km); and the dam-averaged yearly
   anomaly (value minus the station's pre-removal mean) for each role, years
   -5 to +5. Seasonal sampling differences between years are not removed from
   the trajectory, so read it as a description, not a model.

Observational associations; dams are the independent units throughout.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

from analysis import distance_models as dm
from analysis import mixed_model as mm

TRACKED_DISTANCES = dm.DISTANCES_KM


def fixed_mode(re_structure):
    return "slopes" if str(re_structure).startswith("dam slopes") else "intercepts"


def summary_values(fitted):
    """Effect at the reference, its SE / CI, the modifier slope and the
    effects at the tracked distances of one fit_one result."""
    row = fitted["row"]
    out = {"dams": row.get("dams"), "error": row.get("error", "")}
    if row.get("error"):
        return out
    out.update({k: row.get(k) for k in ("effect_at_reference", "effect_se", "effect_ci_low",
                                         "effect_ci_high", "modifier_slope", "modifier_se",
                                         "modifier_ci_low", "modifier_ci_high")})
    for at in fitted["effects_at"]:
        if "distance_km" in at and pd.notna(at.get("distance_km")) \
                and at["distance_km"] in TRACKED_DISTANCES:
            out[f"effect_{int(at['distance_km'])}km"] = at["estimate"]
    return out


def refit(rows, series, spec, settings, options, mode, refs):
    """fit_one on given rows with a fixed structure and references."""
    return dm.fit_one(rows, series, spec, settings, reference_km=options["reference_km"],
                      size_variable=options["size_variable"], re_mode=mode,
                      fixed_refs=refs)


# --------------------------------------------------
# Leave one dam out
# --------------------------------------------------

def leave_one_dam_out(rows, series, spec, settings, options, full):
    """One row per dam left out (design 2)."""
    mode, refs = fixed_mode(full["row"]["re_structure"]), full["refs"]
    base = summary_values(full)
    out = []
    for dam in sorted(rows["AR_ID"].unique()):
        fitted = refit(rows[rows["AR_ID"] != dam], series, spec, settings, options, mode, refs)
        values = summary_values(fitted)
        record = {"series": series, "spec": spec, "dam_left_out": dam, **values}
        if not values.get("error"):
            for key, se_key in (("effect_at_reference", "effect_se"),
                                ("modifier_slope", "modifier_se")):
                if pd.notna(values.get(key)) and base.get(se_key):
                    record[f"delta_{key}"] = values[key] - base[key]
                    record[f"delta_{key}_in_se"] = (values[key] - base[key]) / base[se_key]
            excludes = lambda v, lo, hi: bool(v[lo] > 0 or v[hi] < 0)  # noqa: E731
            record["effect_conclusion_changes"] = (
                excludes(values, "effect_ci_low", "effect_ci_high")
                != excludes(base, "effect_ci_low", "effect_ci_high"))
            record["slope_conclusion_changes"] = (
                pd.notna(values.get("modifier_ci_low")) and pd.notna(base.get("modifier_ci_low"))
                and excludes(values, "modifier_ci_low", "modifier_ci_high")
                != excludes(base, "modifier_ci_low", "modifier_ci_high"))
            record["influential"] = bool(
                abs(record.get("delta_effect_at_reference_in_se", 0)) >= 1
                or abs(record.get("delta_modifier_slope_in_se", 0)) >= 1
                or record["effect_conclusion_changes"] or record["slope_conclusion_changes"])
        out.append(record)
    return pd.DataFrame(out)


def lodo_summary(lodo, full_values):
    """Per series x spec: range of estimates and the influential dams."""
    rows = []
    for (series, spec), group in lodo.groupby(["series", "spec"]):
        ok = group[group["error"].fillna("").eq("")]
        full = full_values[(series, spec)]
        influential = ok[ok["influential"].fillna(False).astype(bool)]
        order = ok.assign(size=ok["delta_effect_at_reference_in_se"].abs()) \
            .sort_values("size", ascending=False)
        rows.append({
            "series": series, "spec": spec, "dams": full.get("dams"),
            "refits": len(group), "refits_failed": int(len(group) - len(ok)),
            "full_effect_at_reference": full.get("effect_at_reference"),
            "full_effect_ci_low": full.get("effect_ci_low"),
            "full_effect_ci_high": full.get("effect_ci_high"),
            "lodo_effect_min": ok["effect_at_reference"].min(),
            "lodo_effect_max": ok["effect_at_reference"].max(),
            "full_modifier_slope": full.get("modifier_slope"),
            "lodo_slope_min": ok["modifier_slope"].min(),
            "lodo_slope_max": ok["modifier_slope"].max(),
            "max_abs_delta_effect_in_se": order["size"].max(),
            "most_influential_dam": order["dam_left_out"].iloc[0] if len(order) else None,
            "influential_dams": int(len(influential)),
            "influential_dam_ids": ", ".join(influential["dam_left_out"].astype(str)),
            "effect_conclusion_changes": int(ok["effect_conclusion_changes"].sum()),
            "slope_conclusion_changes": int(ok["slope_conclusion_changes"].sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------
# Dam bootstrap
# --------------------------------------------------

def resample_dams(rows, rng):
    """All rows of len(dams) dams drawn with replacement; repeats relabeled."""
    dams = rows["AR_ID"].unique()
    drawn = rng.choice(dams, size=len(dams), replace=True)
    groups = {dam: g for dam, g in rows.groupby("AR_ID")}
    parts = []
    for i, dam in enumerate(drawn):
        part = groups[dam].copy()
        part["AR_ID"] = f"{dam}#{i}"
        part["station_id"] = part["station_id"].astype(str) + f"#{i}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _bootstrap_draw(task):
    rows, series, spec, settings, options, mode, refs, seed = task
    fitted = refit(resample_dams(rows, np.random.default_rng(seed)), series, spec, settings,
                   options, mode, refs)
    return {"series": series, "spec": spec, "seed": seed, **summary_values(fitted)}


def bootstrap(rows, series, spec, settings, options, full, draws, seed, workers):
    """(draw rows, summary row) for one series x spec (design 3)."""
    mode, refs = fixed_mode(full["row"]["re_structure"]), full["refs"]
    tasks = [(rows, series, spec, settings, options, mode, refs, seed + i) for i in range(draws)]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_bootstrap_draw, tasks, chunksize=max(draws // workers, 1)))
    else:
        results = [_bootstrap_draw(t) for t in tasks]
    draws_frame = pd.DataFrame(results)
    ok = draws_frame[draws_frame["error"].fillna("").eq("")]
    base = summary_values(full)
    summary = {"series": series, "spec": spec, "draws": draws, "draws_failed": draws - len(ok),
               "re_structure_fixed": full["row"]["re_structure"]}
    for key in ["effect_at_reference", "modifier_slope"] + \
            [f"effect_{d}km" for d in TRACKED_DISTANCES]:
        if key in ok and ok[key].notna().sum() >= 10:
            values = ok[key].dropna()
            summary.update({f"{key}_model": base.get(key), f"{key}_boot_se": values.std(),
                            f"{key}_boot_low": values.quantile(0.025),
                            f"{key}_boot_high": values.quantile(0.975)})
    for key, lo, hi in (("effect_at_reference", "effect_ci_low", "effect_ci_high"),
                        ("modifier_slope", "modifier_ci_low", "modifier_ci_high")):
        if f"{key}_boot_low" in summary:
            summary[f"{key}_model_ci_low"], summary[f"{key}_model_ci_high"] = base[lo], base[hi]
    return draws_frame, summary


# --------------------------------------------------
# BACI components (pH investigation)
# --------------------------------------------------

def baci_components(rows, series):
    """Per dam and role: mean pre / post value and change (station means
    first), and per dam the downstream-minus-upstream difference."""
    station = (rows.groupby(["AR_ID", "role", "station_id", "post"])["y"].mean()
               .unstack("post").rename(columns={0: "pre", 1: "post"}).dropna())
    station["change"] = station["post"] - station["pre"]
    by_dam = station.groupby(["AR_ID", "role"]).agg(
        stations=("change", "size"), pre_mean=("pre", "mean"), post_mean=("post", "mean"),
        change=("change", "mean")).reset_index()
    wide = by_dam.pivot(index="AR_ID", columns="role", values="change") \
        .rename(columns={"control": "change_upstream", "treated": "change_downstream"})
    wide["downstream_minus_upstream"] = wide["change_downstream"] - wide["change_upstream"]
    return by_dam.assign(series=series), wide.reset_index().assign(series=series)


def mean_ci(values):
    values = pd.Series(values).dropna()
    n = len(values)
    if n < 2:
        return {"n_dams": n, "mean": values.mean() if n else np.nan, "ci_low": np.nan,
                "ci_high": np.nan}
    half = stats.t.ppf(0.975, n - 1) * values.std() / np.sqrt(n)
    return {"n_dams": n, "mean": values.mean(), "ci_low": values.mean() - half,
            "ci_high": values.mean() + half}


def components_summary(wide, full, series):
    """Dam-averaged upstream / downstream change and the model's own split."""
    rows = []
    for column, label in (("change_upstream", "upstream (control) change, dam average"),
                          ("change_downstream", "downstream change, dam average"),
                          ("downstream_minus_upstream", "downstream minus upstream, dam average")):
        rows.append({"series": series, "quantity": label, **mean_ci(wide[column]),
                     "share_of_dams_positive": float((wide[column].dropna() > 0).mean())})
    result = full.get("result")
    if result is not None:
        dams = full["row"]["dams"]
        for weights, label in (({"post": 1.0}, "model: upstream (control) change"),
                               ({"post": 1.0, "post_downstream": 1.0},
                                "model: downstream change at 5 km"),
                               ({"post_downstream": 1.0}, "model: BACI effect at 5 km")):
            est, se, lo, hi = dm.linear_effect(result, weights, dams)
            rows.append({"series": series, "quantity": label, "n_dams": dams, "mean": est,
                         "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def trajectory(rows, series):
    """Dam-averaged yearly anomaly per role (value minus the station's
    pre-removal mean), years -5 to +5."""
    rows = rows.copy()
    pre_mean = rows[rows["post"] == 0].groupby("station_id")["y"].mean()
    rows["anomaly"] = rows["y"] - rows["station_id"].map(pre_mean)
    dam_year = (rows.dropna(subset=["anomaly", "years_from_removal"])
                .groupby(["AR_ID", "role", "years_from_removal"])["anomaly"].mean().reset_index())
    out = (dam_year.groupby(["role", "years_from_removal"])["anomaly"]
           .agg(dams="size", mean="mean", sd="std").reset_index())
    out["se"] = out["sd"] / np.sqrt(out["dams"])
    return out.assign(series=series)


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = dm.base_parser(__doc__)
    parser.add_argument("--specs", nargs="+", default=["baci_distance"], choices=list(dm.SPECS))
    parser.add_argument("--bootstrap", type=int, default=200, help="Draws (0 = skip).")
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--components-series", nargs="+", default=["ph"])
    parser.add_argument("--skip-lodo", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    out_dir = args.output_dir or (mm.output_dir_for(args.case_study) / "stage4" / "influence")
    out_dir.mkdir(parents=True, exist_ok=True)
    daily, settings, exclude = dm.load_daily(args)
    restricted, _ = dm.restrict(daily, exclude, args.max_dar)
    options = {"reference_km": args.reference_km, "size_variable": args.size_variable}
    print(f"Case study: {args.case_study}\nOutput: {out_dir}\nSpecs: {args.specs}; "
          f"random effects (full fit): {args.re}; bootstrap draws: {args.bootstrap}; "
          f"workers: {args.workers}; excluded dams: {len(exclude)}")

    lodo_frames, boot_frames, boot_summaries, full_values, full_rows = [], [], [], {}, []
    fits = {}
    for series in settings["series"]:
        for spec in args.specs:
            full = dm.fit_one(restricted, series, spec, settings,
                              reference_km=args.reference_km, size_variable=args.size_variable,
                              re_mode=args.re, lrt_bootstrap=args.lrt_bootstrap,
                              lrt_workers=args.workers)
            full_rows.append(full["row"])
            if full["row"].get("error"):
                print(f"  {series:<28} {spec:<16} skipped: {full['row']['error'][:80]}")
                continue
            fits[(series, spec)] = full
            full_values[(series, spec)] = summary_values(full)
            rows = full["rows"]
            print(f"  {series:<28} {spec:<16} {full['row']['dams']} dams, "
                  f"{full['row']['re_choice']}", flush=True)
            if not args.skip_lodo:
                lodo_frames.append(leave_one_dam_out(rows, series, spec, settings, options, full))
            if args.bootstrap > 0:
                draws, summary = bootstrap(rows, series, spec, settings, options, full,
                                           args.bootstrap, args.seed, args.workers)
                boot_frames.append(draws)
                boot_summaries.append(summary)

    pd.DataFrame(full_rows).to_csv(out_dir / "full_fits.csv", index=False)
    written = ["full_fits.csv"]
    if lodo_frames:
        lodo = pd.concat(lodo_frames, ignore_index=True)
        lodo.to_csv(out_dir / "lodo.csv", index=False)
        summary = lodo_summary(lodo, full_values)
        summary.to_csv(out_dir / "lodo_summary.csv", index=False)
        written += ["lodo.csv", "lodo_summary.csv"]
        print("\nLeave-one-dam-out:")
        print(summary[["series", "spec", "dams", "full_effect_at_reference", "lodo_effect_min",
                       "lodo_effect_max", "influential_dams", "influential_dam_ids"]]
              .to_string(index=False))
    if boot_frames:
        pd.concat(boot_frames, ignore_index=True).to_csv(out_dir / "bootstrap_draws.csv",
                                                         index=False)
        boot = pd.DataFrame(boot_summaries)
        boot.to_csv(out_dir / "bootstrap_summary.csv", index=False)
        written += ["bootstrap_draws.csv", "bootstrap_summary.csv"]
        print("\nDam bootstrap (effect at the reference):")
        cols = [c for c in ["series", "spec", "draws_failed", "effect_at_reference_model",
                            "effect_at_reference_model_ci_low", "effect_at_reference_model_ci_high",
                            "effect_at_reference_boot_low", "effect_at_reference_boot_high"]
                if c in boot]
        print(boot[cols].to_string(index=False))

    summaries = []
    for series in args.components_series:
        full = fits.get((series, "baci_distance")) or next(
            (f for (s, _), f in fits.items() if s == series), None)
        if full is None:
            print(f"\nComponents: no BACI fit for {series}")
            continue
        by_dam, wide = baci_components(full["rows"], series)
        wide.to_csv(out_dir / f"components_{series}.csv", index=False)
        by_dam.to_csv(out_dir / f"components_by_role_{series}.csv", index=False)
        trajectory(full["rows"], series).to_csv(out_dir / f"trajectory_{series}.csv", index=False)
        summaries.append(components_summary(wide, full, series))
        written += [f"components_{series}.csv", f"components_by_role_{series}.csv",
                    f"trajectory_{series}.csv"]
    if summaries:
        comp = pd.concat(summaries, ignore_index=True)
        comp.to_csv(out_dir / "components_summary.csv", index=False)
        written.append("components_summary.csv")
        print("\nBACI components:")
        print(comp.to_string(index=False))
    print(f"\nWrote {', '.join(written)} to {out_dir}")


if __name__ == "__main__":
    main()
