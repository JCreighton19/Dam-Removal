# Time since removal and dam characteristics: model extensions

Added 2026-11-06 to `analysis/distance_models.py` (designs 11 and 12 in its docstring). Both extend the
existing BACI mixed models. Nothing existing changes: the `mixed_model.py` models, the Stage 3–4 specs and
runs, and the Stage 4 report are untouched. New runs write to
`data/processed/mixed_model/<case_study>/stage5/<run>__<re>/`.

## 1. How the effect changes with time since removal

**Question.** Is the BACI effect the same in every year after removal, or does it change (for example a
short-term disturbance that fades, or a response that builds)?

**Specs.**

| Spec | Fixed effects | Effect |
|---|---|---|
| `baci_binary` (benchmark) | `post + downstream + post_downstream + C(month) + z_air_temp + z_precip_prev` | `post_downstream` |
| `baci_time` | `tb_<bin>… + downstream + pd_tb_<bin>… + C(month) + covariates` | `pd_tb_<bin>`: the BACI effect in that interval |
| `before_after_binary` / `before_after_time` | the same without `downstream` (treated rows only) | `post` / `tb_<bin>` |

`baci_binary` is exactly the `mixed_model.py` BACI formula, refit in the same run on the same rows, so the
comparison is like for like. `tb_<bin>` = 1 when the sample's `years_from_removal` falls in the bin; the
first bin (all pre years) is the reference. Random effects: dam and station intercepts, plus (with
`--re slopes` / `auto`) dam slopes on `post` and `post_downstream`. Each dam may have its own overall
effect; the time profile is shared.

**Removal dates are known only to the year.** The removal year itself is never in the data, and
`years_from_removal` = sample year − removal year. So "year 1" is the first calendar year after removal,
somewhere between 1 and 24 months after it. Bins are bands of calendar years, not exact elapsed time,
which is a reason to keep them coarse.

**Bins (chosen from the coverage audit).** Default `y1_y2-3_y4-5`: pre (−5…−1, reference) · year 1 ·
years 2–3 · years 4–5. In the 2026-10-27 export (10 km, suspect dams out), temperature, DO and pH have
17–24 dams with both downstream and upstream data in every single post year. That supports a standalone
year 1, where a short-term response such as a sediment pulse would show. Pooling years 2–3 and 4–5 keeps
≥ 18 such dams per bin (conductance 13, total P 10). Turbidity, orthophosphate and TDS are below
`min_dams` in any binning. **Rerun the audit on the current warehouse** (`RUNS="audit_10km"`) before
relying on this. Other binnings: `--time-bins yearly`, `y1-2_y3-5`, or `'pre:-5:-1;y1:1:1;…'`.

**What is reported.**
- `effects_at.csv`: the effect and 95% CI in each bin, with the treated dams and days behind it. Also a
  `post average` row (bins weighted by treated post days), comparable to the binary effect.
- `results.csv`: `time_change_F` / `time_change_p`, a Wald F test (dams − 1 df) that the post bins are
  equal. Also `pretrend_F` / `pretrend_p` when a placebo pre bin is fit.
- `time_comparison.csv`: binary vs time side by side, covering estimates, CI widths, dams and days, and
  both fits' `fit_check`. In the ML run (`time_ml`) it adds the likelihood-ratio test and AIC / BIC. If
  the time fit's likelihood comes out *below* the nested binary fit's, the optimizer stopped short; the
  test is then left blank with a note rather than reported.
- **Pre-trend check** (`time_pretrend`): pre years are split into late (−2…−1, reference) and early
  (−5…−3). The early-pre BACI term should be ~0. If it isn't, downstream and upstream were already
  diverging before removal, and the BACI effect can't be read as the removal's.

## 2. What explains differences between dams

**Question.** Is the BACI effect larger or smaller at dams with particular characteristics?

**Candidates** (`CHARACTERISTICS`).

| Name | Column | Level | Scale |
|---|---|---|---|
| `reservoir_volume` | `reservoir_volume_proxy` (constructed index) | dam | log10 |
| `reservoir_area` | `reservoir_polygon_area_km2`, mapped by the removal year only | dam | log10 |
| `dam_height` | `dam_height_ft` | dam | log10 |
| `dam_drainage_area` | `dam_drainage_area_km2` | dam | log10 |
| `pct_forest`, `pct_developed`, `pct_agriculture` | baseline land cover of the downstream station's watershed | station | % |
| `distance` | `network_distance_km` | station | ln(1 + km) |
| stored sediment | not available in any source | | |

**How a characteristic enters: as an effect modifier, not a covariate.** Each is standardized
(1 unit = 1 SD) over the fit's dams (dam level) or downstream dam-station pairs (station level). For BACI:
- **dam level:** `m + post_m + down_m + pd_m`;
- **station level:** `down_m + pd_m`, since it exists only downstream and is 0 for controls.

`pd_m` is the change in the BACI effect per 1 SD of the characteristic. For before/after the term is
`post_m`.

**Adjustment vs explanation.**
- **Adjustment variables** explain variation in water quality, not in the response: month, air
  temperature and previous-month rain. In the `*_x_` specs, distance and dilution (`ld`, `ldar`) are also
  adjustments of the effect.
- **Explanatory variables** are the characteristic's `pd_m`.
- The dam and station random intercepts absorb any characteristic's main effect, so adding a
  characteristic as a plain covariate would say nothing about the response.
- `*_xu_` specs drop the distance and dilution adjustment, as a sensitivity check.

**Missing values** are never imputed. A dam without a dam-level value is dropped. A downstream station
without a station-level value loses its rows. Each characteristic is therefore fit on its own sample;
`dams_before_modifier` vs `dams` shows the loss.

**Workflow.**
1. **Audit:** `RUNS=audit_30km`. `modifier_audit.csv` gives dams with a value, the spread and whether a fit
   is possible. `modifier_correlations.csv` gives Spearman ρ between characteristics over the dams.
2. **Screen:** `RUNS=modifiers` fits each characteristic alone (`*_x_<name>`). `modifier_screen.csv` has
   one row per series × characteristic: the effect modification per SD, its CI and p. It also carries a
   Benjamini–Hochberg q-value across series × characteristics within each design, because the screen runs
   many tests.
3. **Combined model, only if justified:** `COMBINED="a b" RUNS=modifiers_combined`. `combined_check.csv`
   flags whether each series supports it. The rule of thumb is ≥ 10 complete-case dams per characteristic
   (and `min_dams`), |ρ| < 0.7 and VIF < 5. With ~20–25 BACI dams per series this usually allows **one**
   characteristic; before/after (100+ dams) can take more but carries regional trends.

**Reading it.** A detected `pd_m` is an association across dams. Characteristics travel together (e.g.
forest vs agriculture ρ ≈ −0.6; reservoir size with dam height and river size), and dams with mapped
reservoirs are on smaller streams. So a modifier may stand in for something unmeasured. It does not show
that the characteristic causes the difference.

## Running (cluster, Slurm only)

```
RUNS="audit_10km audit_30km" sbatch data/slurm/fit_distance_models.slurm   # minutes; check first
RUNS="time time_ml time_pretrend" sbatch data/slurm/fit_distance_models.slurm
RUNS=modifiers sbatch data/slurm/fit_distance_models.slurm                 # longest: 16 specs x series
```

Use the default `RE=auto` for the reported runs. Intercepts-only fits are too confident for effects that
vary by dam: in the Stage 3 runs their intervals were about 3–4x narrower than with dam slopes. A local
dry run of the characteristic screen with intercepts showed the same overconfidence (land-cover p-values
around 1e-5 from 26 dams).

## Limitations

- Year-resolution removal dates: bins are calendar-year bands, and year 1 mixes 1–24 months after removal.
- Later bins rest on fewer dams (follow-up thins after year 3), and dams differ between bins. Treat the
  time profile as an average over a changing set of dams.
- The time profile is fixed across dams. Dam slopes cover each dam's overall effect, not its own trajectory.
- No stored-sediment data. Reservoir volume is a constructed index (height² / slope × √area), and mapped
  area exists for about a third of dams, mostly small streams.
- Station-level land cover is the downstream station's watershed, which includes, but is larger than, the
  dam's.
- Each screened characteristic uses its own complete-case sample, so effects are not on identical dams.
- The q-values limit false leads within the screen. They don't make any single association causal.
