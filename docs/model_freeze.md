# Model freeze

**Frozen 2026-11-18 (re-freeze on the corrected dams and stations; supersedes the 2026-11-10 freeze). Tag
`models-frozen-2026-11-18`.** The 2026-11-10 snapshot is kept for the record (see the change log). From here on the statistical models and their outputs are
fixed. Slides, the interpretation notebook, the predictive-model feasibility work and the Tableau export **read** the
frozen outputs and never refit them. Any later refit, threshold change or new specification needs a dated entry in the
change log below, with the reason, like the pre-specification documents.

## What is frozen

**Data.** The primary analysis table built on 2026-11-18 11:05 (job 11972962) from the ETL rerun on the
manually reviewed dam locations and station reaches (change log, 2026-11-18), with the weather covariates loaded.
The sensitivity-variant tables were built on 2026-11-17 from the same corrected warehouse. The data corrections of
2026-11-08 still apply (conductance and TDS floors, non-detect codes, unflagged zeros; `docs/warehouse_spec.md`
3.6.3, filter 10).
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

`data/processed/frozen/models-frozen-2026-11-18/` (the superseded snapshot stays at
`data/processed/frozen/models-frozen-2026-11-10/`):
- `all_removals/`: a read-only copy of `data/processed/mixed_model/all_removals/` (or omitted to save space; see
  `MANIFEST.sha256`)
- `MANIFEST.sha256`: a checksum for every model output file
- `INPUTS.sha256`: checksums of `analysis_measurement*.parquet` (the tables the models were fit on)
- `COMMIT`: the git commit of the code

Check that the live outputs still match the frozen ones (no output = identical):

    cd data/processed/mixed_model && sha256sum -c --quiet ../frozen/models-frozen-2026-11-18/MANIFEST.sha256

The notebooks read the live folder. The interpretation, distance and time notebooks can read the snapshot instead
with `DAM_REMOVAL_MODEL_DIR=<snapshot>/all_removals`.

## What is still allowed without a change-log entry

- Wording, tables and figures in notebooks and docs that **read** the frozen outputs.
- New read-only analyses of the frozen outputs (e.g. the predictive-model feasibility notebook, the Tableau export).
  These are labelled post-hoc.

## Change log

- **2026-11-18: re-freeze on corrected dams and stations (all models refit).**
  - **Why.** The 2026-11-10 models included some misplaced dams (some matched to pond outlets or wrong reaches
    that no rule could fix) and some stations on the wrong reach. Every dam and station that could enter the
    analysis was reviewed by hand in QGIS before any new results were looked at.
  - **Dams** (`docs/dam_review.md`): corrected locations in `config/snap_overrides.csv` (`corrected_lat` /
    `corrected_lon`; 181 moves, 26 rejects after the MI-117 override was removed because its corrected point fell
    in the Lake Michigan HUC8). ETL rerun from `match_dams`. Funnel: accepted dams 1,861 → 1,956, analysis dams
    530 → 537, before/after 464 → 469, BACI (audit count) 47 → 52.
  - **Stations** (`docs/station_review.md`): all 723 analysis stations reviewed (a full census, not a sample): 693
    correct, 23 on the wrong reach, 1 not a stream, 6 unsure (kept on their current reach). The 24 errors are
    station overrides in `config/snap_overrides.csv`; ETL rerun from `select_station_reaches`.
  - **Missing weather fix.** The first rebuild of the primary table predated `load_climate`, leaving
    `precip_prev_month_mm` null in 16 rows (2 dams, 1 station); `VERIFY_PRIMARY` (job 11941943) caught it. The
    primary table was rebuilt after `load_climate` (job 11972962) and verified identical to a fresh rebuild
    (job 11981922; an earlier verify, job 11972963, was submitted with `ELIGIBILITY_VARIANT` set and stopped on the
    argument check without reading anything).
  - **Refit** with the frozen code and settings, no specification changes: mixed models, distance models
    (stage 3, `RE` auto / intercepts / slopes), stage 4, stage 5 time and characteristic runs (jobs
    11972964–11972971), the eligibility sensitivity variants (2026-11-17) and `modifiers_combined` with
    `COMBINED="dam_height pct_forest pct_agriculture"` (job 11982021). The combined model was not justified for any
    BACI measure (3–23 complete dams; it needs 30) and was justified for 9 before/after measures, where it added no
    new result. The 12 suspect dams in `config/suspect_dam_matches.csv` are still excluded from the distance models.
  - **Downstream.** Everything that reads the frozen outputs uses this snapshot. The predictive-feasibility stages
    built before this date (`audit`, `targets`) read the 2026-11-10 models and are rerun before `cv`.

- **2026-11-10 (recorded at the freeze; no model change).** The cluster working tree at the freeze differed from the
  tagged commit in two non-notebook files. The snapshot records both (`WORKTREE_STATUS.txt`,
  `WORKTREE_CONFIG.diff`, `known_station_reaches.cluster.csv`).
  1. `config/known_station_reaches.csv` had the wrong-reach IDs filled in for 6 stations by
     `etl/check_known_stations.py --fill-wrong-reaches`. The file is read only by that regression check, not by
     station matching, so it doesn't affect the analysis tables or models. That version was committed after the tag.
  2. `data/processed/huc4_list.txt` and `huc8_list.txt` were deleted on the cluster. No pipeline code reads them
     (HUC lists come from the case-study config), so there is no effect. They were restored from git.

  The snapshot was verified on 2026-11-10: `sha256sum -c` of `MANIFEST.sha256` reported no differences.
