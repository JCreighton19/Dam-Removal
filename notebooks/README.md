# Notebooks

Three folders, by when you use them. Every notebook finds the project root by
itself (the nearest folder above it with `etl/` and `config/`), so it runs from
any working directory. Notebooks that read the warehouse set `CASE_STUDY` in
their first code cell. They need the data on the cluster, except
`analysis/local_eda.ipynb`.

## Start here: interpreting the models

`statistical_model_interpretation.ipynb` is the presentation-oriented summary of the statistical modeling: the questions, the BA / BACI / distance / time models and their formulas in plain language, the key findings, robustness and limitations, and a summary for slides. Read-only: it loads the latest model outputs (`data/processed/mixed_model/<case_study>/`), checks they postdate the 8 Nov 2026 data corrections, and fits nothing. Run it on a compute node (data on the cluster).

## Typical order after a pipeline run

1. `qa/warehouse_qa.ipynb`
2. `qa/warehouse_integrity.ipynb`
3. `qa/preprocessing_audit.ipynb`
4. `qa/dar_sensitivity.ipynb`
5. `analysis/analysis_eda.ipynb`
6. `analysis/analysis_robustness.ipynb`
7. `analysis/model_input_eda.ipynb`
8. `sbatch data/slurm/fit_mixed_model.slurm`, then `analysis/mixed_model_results.ipynb`
9. `data/slurm/fit_distance_models.slurm` and `fit_stage4.slurm`, then
   `analysis/distance_model_results.ipynb`
10. `RUNS="audit_10km audit_30km"`, then `RUNS="time time_ml time_pretrend"` and
    `RUNS="modifiers modifiers_unadjusted"` (and the sensitivity checks
    `RUNS="time_ml_slopes time_coarse"`) with `fit_distance_models.slurm`, then
    `analysis/time_and_modifier_results.ipynb`
11. Predictive feasibility (after the model freeze): `STAGE=audit`, then `STAGE=targets`, then `STAGE=cv` with
    `data/slurm/predictive_feasibility.slurm`, reviewing `analysis/predictive_feasibility.ipynb` after each

## `qa/`: check the warehouse before analysis

| Notebook | What it's for | Run after |
|---|---|---|
| `warehouse_qa.ipynb` | Core and cleaned tables: files and row counts, key integrity (every count should be 0), dams / stations / links, map, `wq_measurement` parameters, censoring and units | `load_core_tables`, `clean_wq_results` |
| `warehouse_integrity.ipynb` | Stale derived tables, future-dated samples, what the link distance limits keep, raw-to-cleaned retention before vs after removal | the whole chain, through `build_analysis_dataset` |
| `preprocessing_audit.ipynb` | Read-only audit of `analysis_measurement`: duplicates, misplaced stations, outliers, competing removals, uneven sampling. Writes flag tables to `data/processed/preprocessing_audit/<case_study>/`. Needs about 32 GB, so run it on a compute node: `jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 notebooks/qa/preprocessing_audit.ipynb` | `build_analysis_dataset` |
| `dam_funnel.ipynb` | Where the listed removals drop out before the analysis table (counts and % per stage), rejected / ambiguous dam matches by reason, why some listed dams have no matching record, before-only vs after-only dams by decade (could a wider window help?), and the potential before/after and BACI payoff of accepting the ambiguous dams' provisional reaches. Also a looser-eligibility sensitivity (3/1, 2/1, 1/1 vs 3/2) with before/after and BACI gains, checked against the audit. Diagnostic only; reads the funnel audit, the matching and linking outputs and the warehouse | `AUDIT=funnel sbatch data/slurm/stage1_audits.slurm` |
| `conductance_plausibility.ipynb` | Read-only check for specific-conductance unit errors (added 2026-11-08 after MT-013's 387 → 0.5 µS/cm in the eligibility sensitivity review): the flagged dams' raw records (unit, organization, method), implausibly low values across `wq_measurement`, stations whose yearly median jumps ≥ 100×, and whether any reach the main or sensitivity analysis tables. Decides whether a cleaning rule is needed | `clean_wq_results`, `build_analysis_dataset` |
| `reservoir_station_audit.ipynb` | Phase 0 of a possible former-reservoir analysis (2026-11-08), read-only: upstream stations in the former reservoir under three separate signals (height/slope length, the dam's mapped NHD polygon via the station flowline's `WBArea_Permanent_Identifier`, artificial-path flowline) and their combinations; eligibility under 3 / 2 and 3 / 1; sample depth and pre/post organization changes; controls above the reservoir; unknown-status controls used today; dams per measure and design vs the 10 / 15-dam rules. Reads the NHDPlus packages (attributes only), so run it on a compute node | `build_analysis_dataset` (both variants), `build_dam_reservoir` |
| `dar_sensitivity.ipynb` | What limiting links by drainage-area ratio and distance would keep. Its "distance does nearly all the filtering" conclusion was drawn within 10 km; at the 30 km window (2026-11-04) the ratio rises with distance, so treat it as a 10 km result | `preprocessing_audit` (reads its `pair_features.parquet`) |

## `analysis/`: results

| Notebook | What it's for | Run after |
|---|---|---|
| `analysis_eda.ipynb` | Explores `analysis_measurement` before modeling: coverage, pre/post balance, value distributions, pooled before/after comparisons | `build_analysis_dataset` |
| `analysis_robustness.ipynb` | Before/after with one exposure per measurement, dam-level averages, a matched upstream control, measurement-consistency checks, by-dam event study | `build_analysis_dataset` |
| `model_input_eda.ipynb` | EDA on the exact rows the models are fit on (rebuilt with `mixed_model.prepare` / `model_rows` and `distance_models.model_frame`, fits nothing): where `analysis_measurement` rows are excluded and why, sample size per model, a map of the model dams (BACI vs before/after only, a layer per measure; reads the warehouse `dam_removal` table for coordinates), concentration in a few dams, pre/post balance (month, provider, censoring, turbidity unit), outliers, a model-free dam-level BACI check, distance / dilution spread, size, dam / watershed / climate characteristics of the model dams, stations and days (height, age, use, reservoir size, upstream area, land cover, climate; BACI vs before/after-only vs not-modeled dams, treated vs control stations), covariate shifts, and a checklist. `DAM_REMOVAL_ANALYSIS_INPUT` reads another file | `build_analysis_dataset` |
| `mixed_model_results.ipynb` | Reads the outputs of `analysis/mixed_model.py` (fits nothing itself, opens in seconds); section 7 compares them with the eligibility sensitivity runs (3 samples / 1 year) | `fit_mixed_model` |
| `distance_model_results.ipynb` | Stages 3-4: how each effect changes with distance downstream and with reservoir size; sensitivity runs, random-effects choice, leave-one-dam-out and bootstrap, raw vs scaled distance, pH upstream / downstream split, minimum detectable effects. Section 11: the pre-specified eligibility sensitivity analysis (3 / 1 and 3 / 1 with ≥2 months vs the main 3 / 2 rule; `docs/eligibility_sensitivity_prespec.md`). Reads `stage3/` and `stage4/` outputs; fits nothing | `fit_distance_models` (RE=auto and RE=intercepts), `fit_stage4` |
| `time_and_modifier_results.ipynb` | How the BACI effect changes with years since removal (year 1, years 2–3, years 4–5 vs one effect for all 5 years, pre-trend check), sensitivity checks (comparison with each dam's own effect; 1–2 / 3–5 year intervals) and which dam / watershed characteristics go with bigger or smaller effects (availability, correlations, screen with q-values, adjusted vs unadjusted, combined model, fit checks); section 13 compares the time models under the eligibility sensitivity rule. Reads `stage5/` and `sensitivity_<variant>/`; fits nothing | `fit_distance_models.slurm` stage5 runs |
| `predictive_feasibility.ipynb` | Can pre-removal dam traits predict each dam's water-quality change (`docs/predictive_feasibility_prespec.md`)? It covers the Stage 1 audit (dams per measure and region, predictor completeness, correlation and missingness by region, distance and time coverage) and the Stage 2 per-dam targets (distributions, 95% ranges, weak-support flags, reliability, agreement between target versions, the conductance outlier dams). Section 9 is the Stage 3 cross-validation: go / no-go, R² by model, observed vs predicted, per region, sensitivity analyses, predictor use. It deliberately has no target-vs-predictor plots before Stage 3. Reads `data/processed/predictive/<case_study>/`; fits nothing | `predictive_feasibility.slurm` (each stage) |
| `local_eda.ipynb` | Quick local EDA on the exported CSV (`analysis_measurement_all_removals.csv` at the repo root, from `data/slurm/export_analysis_csv.slurm`). Not in git (`.gitignore`) | `export_analysis_csv` |

## `exploration/`: early source exploration (historical)

Written while designing the pipeline, mostly on the pilot HUC8 19020401 or
`pilot_random5_seed42`. Several read outputs the pipeline no longer produces
(old `snapped_features_<huc8>.gpkg`, `dam_station_links_<huc8>.csv`), so they
may not rerun against the current data. Kept for reference.

| Notebook | Explores |
|---|---|
| `dams_exploration.ipynb` | The dam-removal list (Figshare, Mar 2026) |
| `wbd_exploration.ipynb` | Watershed boundaries (HUC8), and how the HUC4 / HUC8 lists were made |
| `stream_exploration.ipynb` | A sample NHDPlus HR stream dataset |
| `stream_network_exploration.ipynb` | The reach network from `build_stream_network` (diagnostics only) |
| `catchment_area_exploration.ipynb` | NHDPlus catchment areas |
| `wqs_station_exploration.ipynb` | Water-quality station metadata |
| `wqs_results_exploration.ipynb` | Downloaded WQP result records; the cleaning rules came from here |
| `pilot_network_analysis.ipynb` | How dams, stations, streams and watersheds relate spatially (pilot) |
| `link_table_analysis.ipynb` | The old `link_dams_stations` link table (pilot) |
| `relationship_map.html` | A saved interactive map from these explorations |
