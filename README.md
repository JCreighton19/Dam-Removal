# Dam Removal and Water Quality

DS5500 capstone project (Northeastern, team 4) studying how dam removals affect water quality upstream and downstream of the removed dam.

The pipeline links each dam removal to the water-quality monitoring stations on the same river network, downloads those stations' measurements from the years around the removal, and loads everything into a small Parquet + DuckDB warehouse for analysis.

## Data sources

| Source | Used for |
|---|---|
| American Rivers dam removal list (Figshare, Mar 2026) | Dam locations and removal years (placed manually in `data/raw/dams/`) |
| USGS Watershed Boundary Dataset (WBD), national GeoPackage | HUC8 watershed boundaries |
| NHDPlus High Resolution | Stream flowlines and the flow network |
| Water Quality Portal (WQP) | Monitoring stations and measurements |
| NLCD | Land cover (in progress) |

## Repository layout

```
config/          Case-study configs: which HUC8s and dams are in scope
etl/             Pipeline scripts (one per stage) and shared helpers in etl/common/
data/slurm/      One Slurm job per pipeline stage; case_study.env selects the config
docs/            Warehouse table spec (docs/warehouse_spec.md)
notebooks/       Exploration and QA notebooks
tests/           Lightweight tests, runnable with `python tests/<file>.py` or pytest
```

Data (`data/raw/`, `data/processed/`, `data/warehouse/`) is not tracked in git. WQP station selections and result downloads are stored per case study under `data/raw/wqp/results/<case_study>/`. It lives on the Explorer cluster at `/courses/DS5500.202710/teams/team4`.

## Setup

On Explorer:

```bash
sbatch create_env.slurm        # creates or updates the `dam-removal` conda env
```

Choose the case study in `data/slurm/case_study.env`, the one file every job reads. For a random sample of dams, set `PILOT_N_DAMS` and `PILOT_SEED`; `run_pipeline.sh` generates `config/pilot_random<N>_seed<SEED>.yaml` on its first run.

## Pipeline

To run stages 3–11 as one chain of Slurm jobs (each waits for the previous one to succeed), run from the login node:

```bash
bash data/slurm/run_pipeline.sh                      # every stage
bash data/slurm/run_pipeline.sh --from snap_features # resume from a stage
```

The stages, in order:

1. `download_huc8` → `unzip_huc8`: WBD boundaries
2. `sample_random_pilot` (optional): generate a config from a random sample of dams
3. `download_wqp_stations` → `select_candidate_stations`: WQP stations in the case-study HUC8s
4. `download_streams` → `unzip_nhdplus`: NHDPlus HR geodatabases
5. `build_stream_network`: network diagnostics
6. `snap_features`: snap dams and stations to the nearest flowline
7. `link_dams`: classify each station as upstream, downstream or unconnected relative to each dam
8. `download_wqp_results`: measurements within each dam's analysis window
9. `load_core_tables`: write the core warehouse tables (see below)
10. `clean_wq_results`: clean the downloaded results into `wq_measurement`
11. `build_views`: `warehouse.duckdb` views, including which dams have enough before/after data
12. `build_station_watershed`: upstream drainage area of each eligible station (context data, step 1 of 3)
13. `download_nlcd`: Annual NLCD land cover for the needed years, clipped per NHDPlus package
14. `load_land_cover`: land-cover percentages per station and year

## Warehouse

Curated tables are written as Parquet (GeoParquet for geometry) to `data/warehouse/<case_study>/` and queried with DuckDB. The column-level spec, cleaning rules and validation checks are in [`docs/warehouse_spec.md`](docs/warehouse_spec.md), based on the ERD in `DS5500 Capstone ERD final.jpeg`.

| Table | Status |
|---|---|
| `dam_removal`, `wq_station`, `dam_station_link`, `huc8_watershed`, `streams` | `load_core_tables.py` |
| `wq_measurement`, `wq_measurement_raw` | `clean_wq_results.py` (rules in `etl/common/wq_rules.py`) |
| `station_watershed` | `build_station_watershed.py` |
| `station_catchment` | `build_station_watershed.py` |
| `land_cover`, `land_cover_codes` | `load_land_cover.py` |
| `climate` | Planned |

`build_views.py` writes `data/warehouse/<case_study>/warehouse.duckdb`, which holds views only (no data). Open it with `duckdb.connect(path, read_only=True)` and query e.g. `eligibility_summary`. Eligibility thresholds come from the config's optional `eligibility: {min_samples: 3, min_years: 2}` block.

## Current status

Pilot case study: `pilot_random5_seed42` (5 randomly sampled dams, seed 42). Next: run `clean_wq_results` on the pilot, then replace the eligibility report with DuckDB views.
