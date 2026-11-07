# Pre-specification: eligibility sensitivity analysis (3 samples / 1 year)

**Written 2026-11-07, before any model was fit on the sensitivity data.** This note fixes what will be run,
how it will be compared with the primary analysis, and how the comparison will be read. Changes after the
results are seen must be recorded below with the reason.

## Decision

| | Rule (per side of the removal: before and after) | Role |
|---|---|---|
| **Primary** | ≥ 3 samples in ≥ 2 distinct years (`eligibility` in the config) | Primary analysis; all headline results |
| **`min1yr`** | ≥ 3 samples in ≥ 1 distinct year | **Formal sensitivity analysis** |
| **`min1yr_2mo`** | ≥ 3 samples in ≥ 1 distinct year; a side with a single year needs samples in ≥ 2 distinct months | **Secondary variant**: drops the single-month pairs |

The rule is applied per dam × station × measure, as now. 2 / 1 and 1 / 1 are **not** run: their extra pairs rest
on 1–2 samples per side, so a before/after difference would be close to one sample against one sample.

## Rationale

The dam funnel audit (`notebooks/qa/dam_funnel.ipynb`, 2026-11-07; `data/processed/stage1_audit/all_removals/dam_funnel/`)
showed:

- Requiring 2 distinct years costs 121 dams: 530 → 651 dams in the analysis table, 464 → 565 with downstream data.
  The sample count barely matters (2 / 2 adds 9).
- Under the same simplified rules: strict before/after dams 296 → 357 (+61), **strict BACI dams 47 → 69 (+22, +47%)**.
  Of the 22, 10 are existing dams that gain an upstream control and 12 are new. BACI is the analysis's bottleneck.
- BACI dams per measure, before the models' own filters: total P 23 → 39, TSS 24 → 40, TDS 13 → 17,
  dissolved orthophosphate 11 → 15.
- Of the 1,962 station × measure pairs that qualify only under 3 / 1, 88% have their single year spread over
  ≥ 3 months and 6% (117) over a single month. `min1yr_2mo` removes those 117.

## What stays fixed

Everything except the eligibility rule:
- the window (±5 years, removal year excluded), distance limits (30 km down / 10 km up), exposure rules;
- data cleaning, the daily collapse, the suspect-dam exclusions, covariates and flags;
- the model specifications, random-effects selection (`RE=auto`), minimum dams (10) and exploratory label
  (< 15), and the "too small to matter" thresholds (SESOI) in `analysis/stage4_report.py`.

The primary data and results are never overwritten. Variants write `warehouse_<variant>.duckdb`,
`analysis_measurement_<variant>.parquet` and `mixed_model/<case>/sensitivity_<variant>/`.

## Procedure

1. **Views:** `build_views --eligibility-variant <v>` for both variants.
2. **Context data (a partial ETL rerun):** the watershed, land-cover and climate stages cover only eligible stations,
   so they run once with `--eligibility-variant min1yr`, the loosest variant. Its stations include every primary
   station and every `min1yr_2mo` station. Previously computed stations are reused. Primary stations' values don't
   change (they're per station and year). `build_analysis_dataset --verify-primary` must then confirm that the
   primary table rebuilds identically.
3. **Analysis tables:** `build_analysis_dataset --eligibility-variant <v>` for both variants. The run must report 0
   stations without context.
4. **Models:** the existing mixed-model and distance-model jobs with `INPUT_VARIANT=<v>`: same specs, labelled
   `analysis_role = sensitivity`. Planned runs: the 10 km mixed models; distance runs `primary`, `size_proxy`,
   `time` and `time_ml_slopes`.
5. **Comparison:** `notebooks/analysis/distance_model_results.ipynb` section 11 and `mixed_model_results.ipynb`
   section 7.

## Interpretation criteria (fixed now)

The comparison is per measure × design (BACI, before/after). For the distance models it uses the effect at
5 km and the distance slope; for the mixed models, the 10 km effect. Each estimate gets the usual verdict:
*detected* (95% CI excludes 0), *no meaningful change* (whole CI inside ± SESOI) or *inconclusive*.

| Label | Meaning (sensitivity vs primary) |
|---|---|
| **robust** | same verdict and same direction |
| **sharper** | primary inconclusive → sensitivity detected or no meaningful change, same direction |
| **weaker** | primary detected or no meaningful change → sensitivity inconclusive, same direction |
| **conflicting** | direction flips while either CI excludes 0, or detected ↔ no meaningful change |
| **direction unstable** | direction flips, both inconclusive |

- **The primary analysis governs.** Primary findings labelled *robust* or *sharper* are reported as robust to the
  eligibility rule. *Weaker* or *conflicting* ones are reported as sensitive to it. A result detected **only** under
  3 / 1 is reported as suggestive, never as a headline finding.
- **Material improvement for a measure:** the sensitivity sample (a) makes a model fittable (≥ 10 dams) or removes
  the exploratory label (≥ 15), or (b) narrows the 95% CI of the effect by ≥ 20%, or brings the minimum detectable
  effect (80% power) below the SESOI.
- **Are the added dams different?** Added vs primary dams are compared on log dam height, log dam drainage area,
  removal year, downstream distance, land cover (% forest, developed, agriculture) and log reservoir-volume proxy.
  A standardized mean difference |SMD| ≥ 0.25 is flagged as systematic. A change in a result that coincides with
  such a difference is attributed to *who* was added, not to more data.
- **Single-month pairs:** conclusions are unaffected by them if every measure × design has the same label against
  the primary under `min1yr_2mo` as under `min1yr`. Any difference is reported.
- No new hypotheses, measures or specifications are added because of the sensitivity results.

## Limitations

- **Single-year sides.** About 25% of the added pairs have one year both before and after: a one-year-versus-one-year
  comparison, vulnerable to an unusual year such as a drought. BACI absorbs this when controls were sampled in
  the same years; before/after does not. Month effects handle season, not an unusual year.
- **Short follow-up.** The single-year side is most often after removal, so the added data lean toward the first
  years after removal, which also matters for the time-interval models.
- **Different dams.** Added dams may differ from the primary ones (checked above), so changes in estimates can
  reflect who is added as well as how much data.
- **Approximate dam counts in this note.** The audit counts above use simplified rules (no perennial, suspect-dam
  or covariate filters). The models' own dam counts will be lower. The comparison reports the actual model-level
  counts.
- **Influence checks not rerun.** Leave-one-dam-out and the dam bootstrap (Stage 4) are not part of this
  sensitivity analysis.

## Changes after results were seen

*(none yet)*
