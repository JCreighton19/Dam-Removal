# Re-running after the snapping change (2026-10-23)

What changed, how to check it before it replaces anything, and how to re-run. Column and rule details are in [`warehouse_spec.md`](warehouse_spec.md) (3.12 snapping, 3.6a daily table, 3.11 and section 5 for exposure and flags).

## What changed

| Change | Where | Effect on a re-run |
|---|---|---|
| Datum transformation, stream-type filter, distance limits, name/drainage-area/confluence checks, accepted/ambiguous/rejected status, overrides, diagnostics | `snap_features.py`, `etl/common/snapping.py`, `config/snap_overrides.csv` | Some points move to another reach or are no longer accepted, so links and eligibility change |
| Reach type, name, drainage area, slope, datum and snap flags in `dam_removal` and `wq_station` | `load_core_tables.py` | New columns; older snapped files are refused |
| Nearest-upstream-removal exposure, competing removals, station QC (saline etc.) | `build_views.py` views, `build_analysis_dataset.py` columns | New columns; the grain of `analysis_measurement` is unchanged |
| Drainage-area placeholders and truncation flagged | `build_station_watershed.py` | −9999 no longer stored; `upstream_area_km2` null unless `ok` |
| `method_id`, `method_context` | `clean_wq_results.py` | New columns; same `measurement_id`s |
| Daily aggregation | `build_daily_measurements.py` (new stage) | New table `wq_measurement_daily` |
| WQP downloads reused by station and date range | `download_wqp_results.py` | Only new or widened station windows are requested |
| Land cover and climate reused for stations whose upstream area is unchanged | `etl/common/covariates.py`, download and load stages | Only new or changed stations are downloaded and computed |

Stages to re-run: everything from `snap_features` on. `select_candidate_stations` and the NHDPlus downloads are unchanged.

## 1. Update the code and run the tests

```bash
cd /courses/DS5500.202710/teams/team4
git pull
module load anaconda3/2024.06 && conda activate dam-removal
for t in tests/test_*.py; do python "$t" > /dev/null && echo "ok   $t" || echo "FAIL $t"; done
```

## 2. Validate the snapping (nothing is replaced)

```bash
sbatch data/slurm/snap_validation.slurm              # known-failure HUC8s + 30 random ones
```

It snaps the HUC8s holding the seven stations confirmed wrong on the audit map and the audit's `likely wrong` / `check` dams and stations (from `data/processed/preprocessing_audit/<case_study>/`), plus a random sample, into `data/processed/snap_validation/`. It then runs `etl/compare_snapping.py`, which writes `snap_validation/snap_comparison.csv` and prints:

- **Old vs new:** `same_reach`, `moved_reach`, `now_ambiguous`, `now_rejected`, `newly_accepted`. In the random sample most points should keep their reach. Review every `moved_reach` row (old and new reach, names, distances).
- **Known failures:** none of the seven confirmed-wrong stations should still be `same_reach` and accepted. If one is, add a `reject` (or `accept` with the right reach) to `config/snap_overrides.csv`.
- **Decisions by reason:** if `ambiguous` or `rejected` is a large share, look at `snap_diagnostics_<huc8>.csv` (runner-up reach, name, area ratio) before loosening a limit in the config's `snapping:` block.
- **Datums:** NAD27 stations should show a grid-based `datum_transform` (e.g. "NAD27 to NAD83 (1)"), not "Ballpark". If PROJ can't get a grid, the job stops with its name: either keep `PROJ_NETWORK=ON` (set in the Slurm scripts) or install it once with `projsync --file us_noaa_conus.tif` (also `us_noaa_alaska.tif`, `us_noaa_hawaii.tif`, `us_noaa_prvi.tif` if those regions are in scope).

Record reviewed decisions in `config/snap_overrides.csv` (each needs a reason) and re-run the validation until the comparison looks right.

## 3. Back up the current outputs

The re-run replaces the snapped files, links and warehouse tables in place. Keep a copy until the new run is checked; check space first (the quota is about 80 GB):

```bash
source data/slurm/case_study.env
du -sh data/processed/snapped_features_*.gpkg data/processed/dam_station_links_*.csv \
       data/warehouse/$DAM_REMOVAL_CASE_STUDY 2>/dev/null | tail -3
BACKUP=data/backup/before_snapping_$(date +%Y%m%d)
mkdir -p $BACKUP/processed $BACKUP/warehouse
cp data/processed/snapped_features_*.gpkg data/processed/dam_station_links_*.csv $BACKUP/processed/
# wq_measurement_raw is rebuilt from the batch CSVs, so it isn't copied.
rsync -a --exclude wq_measurement_raw.parquet data/warehouse/$DAM_REMOVAL_CASE_STUDY/ $BACKUP/warehouse/
```

Leave `land_cover.parquet` and `climate.parquet` where they are: the re-run keeps their rows for unchanged stations.

## 4. Re-run from snapping

```bash
bash data/slurm/run_pipeline.sh --from snap_features
```

What to expect in the logs:

- `snap_features`: a decision table per HUC8, and a datum check before the first one.
- `download_wqp_results`: "Already downloaded: N station date ranges"; only new or widened windows are requested.
- `build_daily_measurements`: the units and same-day-readings tables. Check the units are what you expect and that the share of rows on shared days matches the audit (about 70% for water temperature).
- `build_station_watershed`: a count of `drainage_area_flag` values. On the first run it keeps the old membership as `station_catchment.prev.parquet`, so existing land-cover and climate rows can be matched.
- `download_nlcd` / `load_land_cover` / `download_climate` / `load_climate`: "keeping N station-years ... computing M". Only the M station-years are downloaded. PRISM allows two downloads of a file per day, so a large M may need the extra attempts `run_pipeline.sh` already queues.
- `build_analysis_dataset`: rows per `exposure_status`, station flag counts and drainage-area flags.

## 5. After the run

- Re-run `notebooks/warehouse_qa.ipynb` and `notebooks/preprocessing_audit.ipynb` and compare with the backup.
- One exposure per measurement: `WHERE exposure_status IN ('assigned_sole', 'assigned_nearest')`. Rows with `competing_*` or `unresolved_*` are flagged, not dropped. An assignment says which known removal is nearest upstream within the window; it is not evidence that the removal caused a change.
- Delete the backup once the new outputs are checked.
