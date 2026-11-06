# Notebooks

Three folders, by when you use them. Every notebook finds the project root by
itself (the nearest folder above it with `etl/` and `config/`), so it runs from
any working directory. Notebooks that read the warehouse set `CASE_STUDY` in
their first code cell. They need the data on the cluster, except
`analysis/local_eda.ipynb`.

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

## `qa/`: check the warehouse before analysis

| Notebook | What it's for | Run after |
|---|---|---|
| `warehouse_qa.ipynb` | Core and cleaned tables: files and row counts, key integrity (every count should be 0), dams / stations / links, map, `wq_measurement` parameters, censoring and units | `load_core_tables`, `clean_wq_results` |
| `warehouse_integrity.ipynb` | Stale derived tables, future-dated samples, what the link distance limits keep, raw-to-cleaned retention before vs after removal | the whole chain, through `build_analysis_dataset` |
| `preprocessing_audit.ipynb` | Read-only audit of `analysis_measurement`: duplicates, misplaced stations, outliers, competing removals, uneven sampling. Writes flag tables to `data/processed/preprocessing_audit/<case_study>/`. Needs about 32 GB, so run it on a compute node: `jupyter nbconvert --to html --execute --ExecutePreprocessor.timeout=-1 notebooks/qa/preprocessing_audit.ipynb` | `build_analysis_dataset` |
| `dar_sensitivity.ipynb` | What limiting links by drainage-area ratio and distance would keep. Its "distance does nearly all the filtering" conclusion was drawn within 10 km; at the 30 km window (2026-11-04) the ratio rises with distance, so treat it as a 10 km result | `preprocessing_audit` (reads its `pair_features.parquet`) |

## `analysis/`: results

| Notebook | What it's for | Run after |
|---|---|---|
| `analysis_eda.ipynb` | Explores `analysis_measurement` before modeling: coverage, pre/post balance, value distributions, pooled before/after comparisons | `build_analysis_dataset` |
| `analysis_robustness.ipynb` | Before/after with one exposure per measurement, dam-level averages, a matched upstream control, measurement-consistency checks, by-dam event study | `build_analysis_dataset` |
| `model_input_eda.ipynb` | EDA on the exact rows the models are fit on (rebuilt with `mixed_model.prepare` / `model_rows` and `distance_models.model_frame`, fits nothing): where `analysis_measurement` rows are excluded and why, sample size per model, concentration in a few dams, pre/post balance (month, provider, censoring, turbidity unit), outliers, a model-free dam-level BACI check, distance / dilution spread, size and covariates, and a checklist. `DAM_REMOVAL_ANALYSIS_INPUT` reads another file | `build_analysis_dataset` |
| `mixed_model_results.ipynb` | Reads the outputs of `analysis/mixed_model.py` (fits nothing itself, opens in seconds) | `fit_mixed_model` |
| `distance_model_results.ipynb` | Stages 3-4: how each effect changes with distance downstream and with reservoir size; sensitivity runs, random-effects choice, leave-one-dam-out and bootstrap, raw vs scaled distance, pH upstream / downstream split, minimum detectable effects. Reads `stage3/` and `stage4/` outputs; fits nothing | `fit_distance_models` (RE=auto and RE=intercepts), `fit_stage4` |
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
