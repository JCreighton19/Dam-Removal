# Model freeze

**Frozen 2026-11-10. Tag `models-frozen-2026-11-10`.** From here on the statistical models and their outputs are
fixed. Slides, the interpretation notebook, the predictive-model feasibility work and the Tableau export **read** the
frozen outputs and never refit them. Any later refit, threshold change or new specification needs a dated entry in the
change log below, with the reason, like the pre-specification documents.

## What is frozen

**Data.** The analysis tables built on 2026-11-08 17:40 with the data corrections (conductance and TDS floors,
non-detect codes, unflagged zeros; `docs/warehouse_spec.md` 3.6.3, filter 10).
- Primary eligibility rule: **3 samples in each of 2 years** per side.
- Sensitivity variants: 3 / 1, and 3 / 1 with ≥ 2 months.

**Models** (code at the tag):
- **Mixed models, 10 km:** `analysis/mixed_model.py`, before/after and BACI. Weather covariates, `C(month)`, dam
  random slopes and station intercepts, t intervals with dams − 1 df.
- **Distance models, 30 km:** `analysis/distance_models.py`, `RE=auto`. These are the **primary results**: the effect
  at 5 km and the distance slope, with dilution held fixed. The 12 suspect dams are excluded. Minimum 10 dams;
  fewer than 15 is exploratory.
- **Stage 4:** leave-one-dam-out influence and the dam bootstrap.
- **Stage 5:** time since removal (`time`, `time_ml`, `time_pretrend`, `time_ml_slopes`, `time_coarse`) and the
  characteristic screens (`modifiers`, `modifiers_unadjusted`, `modifiers_combined`, audits).
- **Sensitivity runs:** `sensitivity_min1yr`, `sensitivity_min1yr_2mo`, `sensitivity_known_controls`.

**Interpretation rules:**
- The thresholds in `analysis/stage4_report.py` `SESOI`: analyst-selected and provisional
  (`docs/practical_significance_thresholds.md`).
- The classification (detected / no meaningful change / inconclusive) and the sensitivity labels
  (`docs/eligibility_sensitivity_prespec.md`, `docs/control_sensitivity_prespec.md`).

**Post-hoc readings** (they add no models): MDE and power at the thresholds (`distance_models.power`,
`dams_needed`); dam-to-dam spread; the inconclusive-vs-no-change split.

## Where the frozen outputs are (cluster)

`data/processed/frozen/models-frozen-2026-11-10/`:
- `all_removals/`: a read-only copy of `data/processed/mixed_model/all_removals/` (or omitted to save space; see
  `MANIFEST.sha256`)
- `MANIFEST.sha256`: a checksum for every model output file
- `INPUTS.sha256`: checksums of `analysis_measurement*.parquet` (the tables the models were fit on)
- `COMMIT`: the git commit of the code

Check that the live outputs still match the frozen ones (no output = identical):

    cd data/processed/mixed_model && sha256sum -c --quiet ../frozen/models-frozen-2026-11-10/MANIFEST.sha256

The notebooks read the live folder. The interpretation, distance and time notebooks can read the snapshot instead
with `DAM_REMOVAL_MODEL_DIR=<snapshot>/all_removals`.

## What is still allowed without a change-log entry

- Wording, tables and figures in notebooks and docs that **read** the frozen outputs.
- New read-only analyses of the frozen outputs (e.g. the predictive-model feasibility notebook, the Tableau export).
  These are labelled post-hoc.

## Change log

*(none yet)*
