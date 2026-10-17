# Warehouse Table Spec

Column-level spec for the curated Parquet tables queried with DuckDB. It
mirrors `DS5500 Capstone ERD final.jpeg`: every table and column in the ERD
is listed here with its source, type, the pipeline stage that writes it, and
the rule that produces it.

Status key used below:
- **Ready**: the inputs already exist in the current pipeline.
- **Step 2**: depends on the WQP cleaning rules (profiling not done yet).
- **Step 5**: depends on contextual-data work (NLCD/climate source not chosen yet).

---

## 1. Conventions

### Layout

```
data/warehouse/<case_study>/
    dam_removal.parquet
    dam_station_link.parquet
    huc8_watershed.parquet
    wq_station.parquet
    wq_measurement/            # partitioned once national-scale volume needs it
        huc8=<huc8>/part.parquet
    streams.parquet
    station_watershed.parquet
    land_cover.parquet
    land_cover_codes.parquet
    climate.parquet
```

- One directory per case study (e.g. `pilot_random5_seed42`), so case studies never overwrite each other.
- One file per table. Exception: `wq_measurement`, which is partitioned by `huc8` once it gets large.
- `data/warehouse/<case_study>/warehouse.duckdb` holds **views only** over these files. It is rebuildable at any time and never the source of truth.

### File format

- **Formats:** Parquet. Tables with a `geometry` column are written as GeoParquet (GeoPandas `to_parquet`), which DuckDB's `spatial` extension reads.
- **Geometry:** stored in EPSG:4326. Lengths and areas (`length_km`, `area_km2`, `areasqkm`, `network_distance_km`) are computed in a projected CRS before storage, never derived from 4326 coordinates.
- **Types:** ERD types map as follows.
  - `TEXT` → string
  - `CHAR(n)` → string, validated to length n
  - `SMALLINT` → int16
  - `INTEGER` → int32
  - `BIGINT` → int64
  - `DOUBLE` → float64
  - `BOOLEAN` → bool
  - `DATE` → date32
  - `TIME` → time64
  - IDs are always written explicitly as string or int64, never left to inference.
- **Missing values** are null. Source sentinels (e.g. `Year_Removed = 0`) are converted to null.
- **Column names** follow the ERD exactly. DuckDB is case-insensitive for identifiers, so the mix of source-style (`Dam_Name`) and snake_case (`station_id`) names is harmless.

### Write rules

- **Atomic:** write to `<table>.parquet.tmp`, then rename over the final file. A crashed job never leaves a half-written table.
- **Idempotent:** a stage always rewrites its whole table (or partition) for the case study. There are no appends or in-place updates.
- **Single writer:** only one job writes a given table at a time. Stages run sequentially, so this already holds.
- **Validated on write:** each writer runs the checks in section 4 for its table and fails instead of writing bad data.

### Generated keys

`wq_measurement.measurement_id` is a generated key, since WQP's legacy result profile has no result identifier:

```
measurement_id = signed int64 from the first 8 bytes of
                 SHA-256("|".join([OrganizationIdentifier, ActivityIdentifier,
                                   CharacteristicName, ResultSampleFractionText,
                                   ResultMeasureValue, ResultMeasure/MeasureUnitCode,
                                   ResultDetectionConditionText,
                                   ResultAnalyticalMethod/MethodIdentifier]))
```

Nulls are joined as empty strings. Hash collisions or genuine duplicate rows are resolved by the deduplication rule (step 2) before the write. A duplicate `measurement_id` at write time is an error.

---

## 2. Where each table comes from

| Table | Status | Written by (proposed stage) | Inputs |
|---|---|---|---|
| `dam_removal` | Ready | `load_core_tables.py` (new, after `link_dams`) | dam CSV, `snapped_features_<huc8>.gpkg` |
| `wq_station` | Ready | `load_core_tables.py` | `data/raw/wqp/stations/<huc8>.csv`, `snapped_features_<huc8>.gpkg` |
| `dam_station_link` | Ready | `load_core_tables.py` | `dam_station_links_<huc8>.csv` |
| `huc8_watershed` | Ready | `load_core_tables.py` | WBD GeoPackage (HU8 layer) |
| `streams` | Ready | `load_core_tables.py` | NHDPlus HR `NHDFlowline` + reach graph |
| `wq_measurement` | Step 2 | `clean_wq_results.py` (new, after `download_wqp_results`) | `pilot_download/results_batch_*.csv` |
| `station_watershed` | Step 5 | context stage (new) | NHDPlus HR `NHDPlusCatchment` + reach graph |
| `land_cover` | Step 5 | context stage (new) | Annual NLCD (source TBD) + `station_watershed` |
| `land_cover_codes` | Ready | static, written once | NLCD legend (section 3.9) |
| `climate` | Step 5 | context stage (new) | climate source TBD + `station_watershed` |

The two new loader stages leave the existing ETL scripts untouched. They read the outputs those scripts already produce (gpkg, CSV) and write the curated tables, so the existing per-stage outputs remain the intermediate layer.

---

## 3. Tables

### 3.1 `dam_removal`

- **Grain:** one row per dam removal in the case study.
- **PK:** `AR_ID`.
- **Rows:** dams kept by `snap_features.py` (`filter_to_sampled_dams` for sampled case studies) that snapped to a reach.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `AR_ID` | TEXT PK | dam CSV `AR_ID` | Cast to string |
| `NABI_ID` | TEXT | dam CSV `NABI_ID` | |
| `Dam_Name` | TEXT | dam CSV `Dam_Name` | |
| `Year_Removed` | SMALLINT | dam CSV `Year_Removed` | Non-numeric or 0 → null (same rule as `report_pilot_eligibility.prepare_links`) |
| `Latitude` | DOUBLE | dam CSV `Latitude` | **Original** coordinate |
| `Longitude` | DOUBLE | dam CSV `Longitude` | **Original** coordinate |
| `City_County` | TEXT | dam CSV `City_County` | |
| `River` | TEXT | dam CSV `River` | |
| `HUC8` | CHAR(8) | pipeline | The HUC8 the dam was clipped to in `snap_features.py` (authoritative), zero-padded. Not the CSV's own HUC8 field. |
| `State` | CHAR(2) | dam CSV `State` | |
| `Dam_Height_ft` | DOUBLE | dam CSV `Dam_Height_ft` | Non-numeric → null |
| `Dam_Length_ft` | DOUBLE | dam CSV `Dam_Length_ft` | Non-numeric → null |
| `Year_built` | SMALLINT | dam CSV **`Year_Built`** | Source spelling differs from the ERD; non-numeric or 0 → null |
| `Original_Use` | TEXT | dam CSV `Original_Use` | |
| `Type_Material` | TEXT | dam CSV `Type_Material` | |
| `Reason_for_removal` | TEXT | dam CSV `Reason_for_removal` | |
| `Miles_Restored_US` | DOUBLE | dam CSV `Miles_Restored_US` | |
| `Miles_Restored_DS` | DOUBLE | dam CSV `Miles_Restored_DS` | |
| `Last_Updated` | DATE | dam CSV `Last_Updated` | Parse; unparseable → null |
| `NHDPlusID` | BIGINT | gpkg `dams_snapped.NHDPlusID` | FK → `streams.nhdplus_id` |
| `snap_distance_m` | DOUBLE | gpkg `dams_snapped.snap_distance_m` | |
| `snap_fraction` | DOUBLE | gpkg `dams_snapped.snap_fraction` | 0 at the reach's start, 1 at its end |
| `match_method` | TEXT | constant | `nearest_flowline` |
| `geometry` | Point, 4326 | gpkg `dams_snapped` geometry | **Snapped** location |

**Duplicate snaps.** A dam equidistant to two reaches currently appears twice in `dams_snapped`. Keep one row per `AR_ID`: the smallest `snap_distance_m`, then the smallest `NHDPlusID` as a deterministic tie-break. Log every dropped duplicate.

### 3.2 `wq_station`

- **Grain:** one row per station.
- **PK:** `station_id`.
- **Rows:** stations that appear in `dam_station_link`, i.e. network-connected to at least one dam. This keeps the table small and guarantees every FK from the link table resolves.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `station_id` | TEXT PK | WQP `MonitoringLocationIdentifier` | |
| `station_name` | TEXT | WQP `MonitoringLocationName` | |
| `station_type` | TEXT | WQP `MonitoringLocationTypeName` | Already filtered to the config's `station_types` |
| `organization_id` | TEXT | WQP `OrganizationIdentifier` | |
| `organization_name` | TEXT | WQP `OrganizationFormalName` | |
| `latitude` | DOUBLE | WQP `LatitudeMeasure` | **Original** coordinate |
| `longitude` | DOUBLE | WQP `LongitudeMeasure` | **Original** coordinate |
| `state_code` | TEXT | WQP `StateCode` | |
| `county_code` | TEXT | WQP `CountyCode` | |
| `huc8` | CHAR(8) | pipeline | The HUC8 whose snapped file the station came from. FK → `huc8_watershed` |
| `horizontal_datum` | TEXT | WQP `HorizontalCoordinateReferenceSystemDatumName` | |
| `horizontal_method` | TEXT | WQP `HorizontalCollectionMethodName` | |
| `NHDPlusID` | BIGINT | gpkg `stations_snapped.NHDPlusID` | FK → `streams.nhdplus_id` |
| `snap_distance_m` | DOUBLE | gpkg `stations_snapped.snap_distance_m` | |
| `snap_fraction` | DOUBLE | gpkg `stations_snapped.snap_fraction` | |
| `match_method` | TEXT | constant | `nearest_flowline` |
| `geometry` | Point, 4326 | gpkg `stations_snapped` geometry | **Snapped** location |

Duplicate snaps are resolved the same way as for dams (section 3.1). A station near a HUC8 boundary that appears in two HUC8s' files keeps the row from the HUC8 its point falls within.

### 3.3 `dam_station_link`

- **Grain:** one row per network-connected (dam, station) pair.
- **PK:** `(AR_ID, station_id)`.
- **Rows:** `dam_station_links_<huc8>.csv` rows with `relationship` in (`upstream`, `downstream`, `same_reach`). `unconnected` pairs are **not stored**; they remain in the intermediate CSVs for QA.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `AR_ID` | TEXT | links `dam_id` | FK → `dam_removal` |
| `station_id` | TEXT | links `station_id` | FK → `wq_station` |
| `relationship` | TEXT | links `relationship` | One of `upstream`, `downstream`, `same_reach` |
| `network_distance_km` | DOUBLE | links `network_distance_km` | Null when no path or fraction was available |
| `distance_method` | TEXT | links `distance_method` | `same_reach_fraction` or `network_path_fractional_endpoints` |

The links CSV columns `dam_name`, `year_removed`, `river`, `distance_to_dam_km` (a duplicate of `network_distance_km`), the reach IDs and the snap columns are dropped. They live on `dam_removal` and `wq_station`.

### 3.4 `huc8_watershed`

- **Grain:** one row per HUC8 in the case study.
- **PK:** `huc8`.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `huc8` | CHAR(8) PK | WBD HU8 layer `huc8` | Zero-padded |
| `name` | TEXT | WBD `name` | |
| `states` | TEXT | WBD `states` | e.g. `"OH,PA"` |
| `areasqkm` | DOUBLE | WBD `areasqkm` | As published by WBD |
| `geometry` | MultiPolygon, 4326 | WBD geometry | Reprojected to 4326 |

### 3.5 `streams` (mapping helper)

- **Grain:** one row per NHDPlus HR flowline.
- **PK:** `nhdplus_id`.
- **Rows:** every reach upstream or downstream of a case-study dam (via the reach graph), plus the snapped reach of every `wq_station`, limited to reaches whose `ReachCode` starts with a case-study HUC8. The same scoping is used by the pilot map in `notebooks/stream_network_exploration.ipynb`.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `nhdplus_id` | BIGINT PK | `NHDFlowline.NHDPlusID` | Read via `read_nhdplus_layer` (case-insensitive) |
| `reach_code` | TEXT | `NHDFlowline.ReachCode` | First 8 characters = HUC8 |
| `gnis_name` | TEXT | `NHDFlowline.GNIS_Name` | |
| `length_km` | DOUBLE | `NHDFlowline.LengthKM` | As published |
| `feature_type` | SMALLINT | `NHDFlowline.FType` | e.g. 460 stream/river, 336 canal/ditch |
| `feature_code` | INTEGER | `NHDFlowline.FCode` | |
| `geometry` | MultiLineString, 4326 | `NHDFlowline` geometry | Z/M dropped; simplified (tolerance ~5 m) for map rendering |

`streams` is not the source of truth for the network. Graph logic always reads the raw NHDPlus geodatabase.

### 3.6 `wq_measurement` (Step 2)

- **Grain:** one row per cleaned WQP result.
- **PK:** `measurement_id`.
- **Rows:** results for `wq_station` stations that survive the step 2 filters (QC samples, non-water media and rejected statuses excluded; duplicates removed).

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `measurement_id` | BIGINT PK | generated | Section 1, "Generated keys" |
| `station_id` | TEXT | `MonitoringLocationIdentifier` | FK → `wq_station` |
| `organization_id` | TEXT | `OrganizationIdentifier` | |
| `activity_id` | TEXT | `ActivityIdentifier` | |
| `sample_date` | DATE | `ActivityStartDate` | Rows with unparseable dates dropped |
| `sample_time` | TIME | `ActivityStartTime/Time` | Null when absent |
| `time_zone` | TEXT | `ActivityStartTime/TimeZoneCode` | |
| `parameter` | TEXT | `CharacteristicName` | Verbatim (no merging of names) |
| `result_value_raw` | TEXT | `ResultMeasureValue` | Verbatim, e.g. `"<0.5"`, `"ND"` |
| `result_value_std` | DOUBLE | derived | Parsed value converted to `result_unit_std`. **Step 2:** conversion table and non-detect substitution rule |
| `result_unit_raw` | TEXT | `ResultMeasure/MeasureUnitCode` | |
| `result_unit_std` | TEXT | derived | One standard unit per `parameter` (+ fraction). **Step 2** |
| `detection_condition` | TEXT | `ResultDetectionConditionText` | |
| `detection_limit` | DOUBLE | `DetectionQuantitationLimitMeasure/MeasureValue` | |
| `detection_limit_unit` | TEXT | `DetectionQuantitationLimitMeasure/MeasureUnitCode` | |
| `is_censored` | BOOLEAN | derived | True if `detection_condition` indicates a non-detect, or `result_value_raw` starts with `<` or `>`. **Step 2:** finalize the list of non-detect strings |
| `sample_fraction` | TEXT | `ResultSampleFractionText` | |
| `activity_type` | TEXT | `ActivityTypeCode` | Retained after filtering, for audit |
| `media` | TEXT | `ActivityMediaName` | |
| `result_status` | TEXT | `ResultStatusIdentifier` | |
| `value_type` | TEXT | `ResultValueTypeName` | |
| `qualifier_code` | TEXT | `MeasureQualifierCode` | |
| `depth_m` | DOUBLE | `ActivityDepthHeightMeasure/MeasureValue` + `…/MeasureUnitCode` | Converted to meters; null when absent |

**Step 2 decisions** (from profiling the pilot downloads):
1. The standard unit for each parameter (+ fraction) and the conversion factors.
2. The non-detect rule: e.g. `result_value_std = detection_limit / 2` with `is_censored = true`, or null.
3. The excluded `ActivityTypeCode` values (QC blanks, spikes, replicates), the kept `ActivityMediaName` (`Water`) and the excluded `ResultStatusIdentifier` values (`Rejected`).
4. Deduplication: the definition of a duplicate result (e.g. same station, date, time, parameter, fraction and value from different providers).
5. Whether to also write an uncleaned `wq_measurement_raw` (all downloaded rows, typed) for audit. Recommended for the pilot.

### 3.7 `station_watershed` (Step 5)

- **Grain:** one row per `wq_station`.
- **PK:** `station_id`. 1:1 with `wq_station`.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `station_id` | TEXT PK | `wq_station` | FK → `wq_station` |
| `upstream_area_polygon` | MultiPolygon, 4326 | `NHDPlusCatchment` | Dissolved union of the catchments of the station's reach and all of its graph ancestors |
| `area_km2` | DOUBLE | derived | Sum of catchment areas (projected CRS) |
| `n_catchments` | INTEGER | derived | Number of catchments in the union |
| `truncated` | BOOLEAN | derived | True if any ancestor chain reaches the edge of the NHDPlus package (flow enters from outside it), so the upstream area is incomplete |

### 3.8 `land_cover` (Step 5)

- **Grain:** one row per station × year × NLCD class.
- **PK:** `(station_id, year, class_code)`.
- **Rows:** only years inside some dam window for that station (removal year ± `wq_window`, same windows as the WQP download), and only years the land-cover source has.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `station_id` | TEXT | `station_watershed` | FK → `station_watershed` |
| `year` | SMALLINT | land-cover source | |
| `class_code` | SMALLINT | raster value | FK → `land_cover_codes` |
| `area_km2` | DOUBLE | derived | Class area in the upstream area (projected/equal-area) |
| `percent_area` | DOUBLE | derived | `area_km2` / valid area × 100 |
| `valid_fraction` | DOUBLE | derived | Share of the upstream area with valid (non-nodata) pixels |
| `source` | TEXT | constant | e.g. `annual_nlcd_c1v2` |

Computed per catchment, then accumulated downstream to each station's reach (one pass in flow order), rather than masking each station's dissolved polygon. **Step 5:** choose the source (official AWS with credentials for every year 1985–2025, or the public COG mirror with missing years).

### 3.9 `land_cover_codes`

- **Grain:** one row per NLCD class.
- **PK:** `class_code`.
- Static. Written once, from the standard NLCD legend.

| `class_code` | `class_label` |
|---|---|
| 11 | Open Water |
| 12 | Perennial Ice/Snow |
| 21 | Developed, Open Space |
| 22 | Developed, Low Intensity |
| 23 | Developed, Medium Intensity |
| 24 | Developed, High Intensity |
| 31 | Barren Land |
| 41 | Deciduous Forest |
| 42 | Evergreen Forest |
| 43 | Mixed Forest |
| 52 | Shrub/Scrub |
| 71 | Grassland/Herbaceous |
| 81 | Pasture/Hay |
| 82 | Cultivated Crops |
| 90 | Woody Wetlands |
| 95 | Emergent Herbaceous Wetlands |

Optional extra column: `class_group` (Water, Developed, Barren, Forest, Shrubland, Herbaceous, Agriculture, Wetlands) for coarser summaries.

### 3.10 `climate` (Step 5)

- **Grain:** one row per station × year.
- **PK:** `(station_id, year)`.
- **Rows:** same year selection as `land_cover`.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `station_id` | TEXT | `station_watershed` | FK → `station_watershed` |
| `year` | SMALLINT | climate source | |
| `temperature_mean_c` | DOUBLE | derived | Area-weighted mean annual air temperature over the upstream area |
| `precipitation_mm` | DOUBLE | derived | Area-weighted annual total precipitation over the upstream area |
| `source` | TEXT | constant | e.g. `prism`, `gridmet`, `daymet` |

**Step 5:** choose the source.

---

## 4. Validation checks

Run by each writer before the atomic rename; any failure aborts the write.

| Table | Check |
|---|---|
| all | PK unique and non-null; column set and types match this spec |
| all with geometry | Geometry non-null and valid; CRS is EPSG:4326 |
| `dam_removal` | `HUC8` in `huc8_watershed`; `NHDPlusID` non-null; `snap_fraction` in [0, 1]; `Year_Removed` null or in [1800, current year] |
| `wq_station` | `huc8` in `huc8_watershed`; `NHDPlusID` non-null; `snap_fraction` in [0, 1] |
| `dam_station_link` | `AR_ID` in `dam_removal`; `station_id` in `wq_station`; `relationship` in the allowed set; `network_distance_km` ≥ 0 or null |
| `streams` | Every `dam_removal.NHDPlusID` and `wq_station.NHDPlusID` present |
| `wq_measurement` | `station_id` in `wq_station`; `sample_date` non-null; `result_unit_std` non-null wherever `result_value_std` is non-null; `result_value_std` within plausible range per parameter (step 2 range table) |
| `station_watershed` | One row per `wq_station`; `area_km2` > 0 |
| `land_cover` | `class_code` in `land_cover_codes`; `percent_area` sums to ~100 per (station, year); `valid_fraction` in [0, 1] |
| `climate` | `station_id` in `station_watershed`; values within physical ranges |

---

## 5. DuckDB views (built in step 4)

`warehouse.duckdb` defines views only, for example:

- One view per table, `CREATE VIEW dam_removal AS SELECT * FROM 'dam_removal.parquet'`, so notebooks don't hardcode paths.
- **`measurement_window`:** `dam_station_link` ⋈ `dam_removal` ⋈ `wq_measurement`, with each measurement classified `pre` / `post` / `removal_year` / `outside` using `wq_window` (same definition as `etl/report_pilot_eligibility.py`).
- **`eligibility_pairs` / `eligibility_dams`:** the SQL replacement for `report_pilot_eligibility.py`, checked against the current CSV reports on the pilot data before the script is retired.
- **Context joins:** `land_cover` and `climate` join to measurements on `station_id` and `year(sample_date) = year`. That's a join condition, not an FK.
