# Warehouse Table Spec

Column-level spec for the curated Parquet tables queried with DuckDB. It
mirrors `DS5500 Capstone ERD final.jpeg`: every table and column in the ERD
is listed here with its source, type, the pipeline stage that writes it, and
the rule that produces it.

Status key used below:
- **Ready**: the inputs already exist in the current pipeline.
- **Step 2**: depends on the WQP cleaning rules (decided 2026-10-17, section 3.6).
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

Nulls are joined as empty strings. Hash collisions or genuine duplicate rows are resolved by the deduplication rule (section 3.6.5) before the write. A duplicate `measurement_id` at write time is an error.

---

## 2. Where each table comes from

| Table | Status | Written by (proposed stage) | Inputs |
|---|---|---|---|
| `dam_removal` | Ready | `load_core_tables.py` (new, after `link_dams`) | dam CSV, `snapped_features_<huc8>.gpkg` |
| `wq_station` | Ready | `load_core_tables.py` | `data/raw/wqp/stations/<huc8>.csv`, `snapped_features_<huc8>.gpkg` |
| `dam_station_link` | Ready | `load_core_tables.py` | `dam_station_links_<huc8>.csv` |
| `huc8_watershed` | Ready | `load_core_tables.py` | WBD GeoPackage (HU8 layer) |
| `streams` | Ready | `load_core_tables.py` | NHDPlus HR `NHDFlowline` + reach graph |
| `wq_measurement` | Ready (rules in 3.6) | `clean_wq_results.py` (after `download_wqp_results` and `load_core_tables`) | `data/raw/wqp/results/<case_study>/pilot_download/results_batch_*.csv` listed in `download_manifest.csv`; `wq_station.parquet` |
| `wq_measurement_raw` | Ready | `clean_wq_results.py` | same as `wq_measurement`, unfiltered (3.6.6) |
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
- **Rows:** every reach upstream or downstream of a case-study dam (via the reach graph), limited to reaches whose `ReachCode` starts with a case-study HUC8, plus the snapped reach of every dam and `wq_station` (kept even across a HUC8 boundary, so every `NHDPlusID` FK resolves). The same scoping is used by the pilot map in `notebooks/stream_network_exploration.ipynb`.

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

### 3.6 `wq_measurement`

- **Grain:** one row per cleaned WQP result.
- **PK:** `measurement_id`.
- **Rows:** results for `wq_station` stations, limited to the parameters in the mapping table below, that survive the filters (section 3.6.2) and deduplication (3.6.5).
- **Rules source:** profiling of the `pilot_random5_seed42` downloads (`notebooks/wqs_results_exploration.ipynb`, about 10,150 target-parameter rows), decided 2026-10-17. The rule tables are implemented as data in `etl/common/wq_rules.py`, applied by `etl/clean_wq_results.py`. Re-check the value counts when the 30–50 dam sample is downloaded; unseen codes default to **kept and logged**, not silently dropped.

| Column | Type | Source | Rule / notes |
|---|---|---|---|
| `measurement_id` | BIGINT PK | generated | Section 1, "Generated keys" |
| `station_id` | TEXT | `MonitoringLocationIdentifier` | FK → `wq_station` |
| `organization_id` | TEXT | `OrganizationIdentifier` | |
| `provider` | TEXT | `ProviderName` | `NWIS` or `STORET` |
| `activity_id` | TEXT | `ActivityIdentifier` | |
| `sample_date` | DATE | `ActivityStartDate` | Rows with unparseable dates dropped |
| `sample_time` | TIME | `ActivityStartTime/Time` | Null when absent |
| `time_zone` | TEXT | `ActivityStartTime/TimeZoneCode` | |
| `parameter` | TEXT | `CharacteristicName` | Verbatim |
| `parameter_std` | TEXT | derived | Section 3.6.1 mapping. Analyses group on this, not `parameter` |
| `usgs_pcode` | TEXT | `USGSPCode` | Zero-padded to 5 characters; null for STORET |
| `result_value_raw` | TEXT | `ResultMeasureValue` | Verbatim, e.g. `"=6.59"`, `"## (Censored)"` |
| `result_value_std` | DOUBLE | derived | Section 3.6.3 (parsing, non-detects) and 3.6.4 (units) |
| `result_unit_raw` | TEXT | `ResultMeasure/MeasureUnitCode` | Verbatim |
| `result_unit_std` | TEXT | derived | The standard unit for `parameter_std` (3.6.1) |
| `unit_basis_assumed` | BOOLEAN | derived | True when the chemical form (as N / as P) was assumed, not stated (3.6.4) |
| `detection_condition` | TEXT | `ResultDetectionConditionText` | |
| `detection_limit` | DOUBLE | `DetectionQuantitationLimitMeasure/MeasureValue` | |
| `detection_limit_unit` | TEXT | `DetectionQuantitationLimitMeasure/MeasureUnitCode` | |
| `detection_limit_type` | TEXT | `DetectionQuantitationLimitTypeName` | |
| `is_censored` | BOOLEAN | derived | Section 3.6.3 |
| `sample_fraction` | TEXT | `ResultSampleFractionText` | Verbatim |
| `sample_fraction_std` | TEXT | derived | `Filtered, field` → `Dissolved`; null for field parameters (3.6.1) |
| `activity_type` | TEXT | `ActivityTypeCode` | Retained after filtering, for audit |
| `media` | TEXT | `ActivityMediaName` | |
| `result_status` | TEXT | `ResultStatusIdentifier` | `Preliminary` is kept; filter on this column if needed |
| `value_type` | TEXT | `ResultValueTypeName` | |
| `qualifier_code` | TEXT | `MeasureQualifierCode` | |
| `hydrologic_condition` | TEXT | `HydrologicCondition` | NWIS only (null for all STORET rows), so informational, not a filter |
| `depth_m` | DOUBLE | `ActivityDepthHeightMeasure/MeasureValue` + `…/MeasureUnitCode` | Converted to meters; null when absent |

#### 3.6.1 Parameter mapping and standard units

Rows whose `CharacteristicName` is not in this table are dropped. Field parameters have no meaningful fraction, so `sample_fraction_std` is null for them (`Total` and blank mean the same thing there). For nutrients the fraction matters, so analyses group on (`parameter_std`, `sample_fraction_std`).

| `parameter_std` | `CharacteristicName` (+ unit condition) | `result_unit_std` | Field parameter |
|---|---|---|---|
| `water_temp` | `Temperature, water` | `deg C` | yes |
| `specific_conductance` | `Specific conductance` | `uS/cm @25C` | yes |
| `dissolved_oxygen` | `Dissolved oxygen (DO)`, `Oxygen`, with a mg/L unit | `mg/L` | yes |
| `do_saturation` | `Dissolved oxygen (DO)`, `Oxygen`, with unit `%` or `% saturatn` | `%` | yes |
| `ph` | `pH` | `std units` | yes |
| `turbidity` | `Turbidity` | `NTU-equivalent` (raw unit kept; see 3.6.4) | yes |
| `nitrate` | `Nitrate` | `mg/L as N` | no |
| `ammonia` | `Ammonia and ammonium` | `mg/L as N` | no |
| `orthophosphate` | `Orthophosphate` | `mg/L as P` | no |
| `phosphorus` | `Phosphorus` | `mg/L as P` | no |
| `tds` | `Total dissolved solids` | `mg/L` | no |

Why one standardized name: providers name and unit the same measurement differently (NWIS `Oxygen` vs STORET `Dissolved oxygen (DO)`), which is the main known obstacle to combining WQP data ([Sprague et al., USGS](https://pubs.usgs.gov/publication/70179304)). Mapping once here keeps every analysis consistent.

#### 3.6.2 Row filters

Applied in this order. Each drop is counted by reason and written to the stage log.

| # | Rule | Pilot rows affected |
|---|---|---|
| 1 | Drop `ActivityTypeCode` starting with `Quality Control` (field replicates, blanks, lab duplicates). `Unknown` is kept (pilot: routine USGS-NJ field readings) | 878 |
| 2 | Keep `ActivityMediaName = Water` only; drop `ActivityMediaSubdivisionName` in (`Bottom material`, `Groundwater`) and any non-null `SampleAquifer` | 1 |
| 3 | Drop `ResultStatusIdentifier = Rejected` | 115 |
| 4 | Drop `ResultValueTypeName = Calculated` (pilot: all are NJ_MSU "Suspended" nitrate/orthophosphate, i.e. derived values) | 119 |
| 5 | Drop non-null `StatisticalBaseCode` (daily means/maxima, not single samples) | 0 |
| 6 | Drop fraction `Bed Sediment` and units `mg/kg` (sediment results) | 4 |
| 7 | Drop rows whose `MeasureQualifierCode`, split on `;`, includes `H` (holding time exceeded) | 2 |
| 8 | Drop NWIS alternate-form or derived parameter codes (below) | ~970 |
| 9 | Drop rows with unparseable `sample_date`, or that end with no `result_value_std` and no detection limit | `NO DATA` rows |
| 10 | Drop values outside the plausible range for their `parameter_std` (`PLAUSIBLE_RANGES` in `etl/common/wq_rules.py`, e.g. pH 0–14, water temperature −1–40 °C, DO 0–25 mg/L). Wide on purpose: catches unit and entry errors, not unusual readings | not yet profiled |

Also dropped and logged before filter 1: parameters not in the 3.6.1 mapping, and stations not in `wq_station`. Units that can't be resolved or converted (3.6.4) are dropped between filters 8 and 9.

NWIS reports some measurements twice under different parameter codes (same sample restated in another chemical form, or a lab re-measurement of a field value). Keep one:

| `parameter_std` | Keep pcode | Drop pcode | Reason |
|---|---|---|---|
| `ammonia` | 00608, 00610 (as N) | 71846, 71845 (as NH4) | Same result restated |
| `nitrate` | 00618 (as N) | 71851 (as NO3) | Same result restated |
| `orthophosphate` | 00671 (as P) | 00660 (as PO4) | Same result restated |
| `tds` | 70300 (residue at 180 °C) | 70301 (sum of constituents), 70302 (tons/day), 70303 (tons/acre-ft) | Calculated or load units |
| `specific_conductance` | 00095 (field) | 90095 (lab) | Field value preferred |
| `ph` | 00400 (field) | 00403 (lab) | Field value preferred |

#### 3.6.3 Value parsing and non-detects

- Strip whitespace and a leading `=` (`=6.59` → 6.59), then parse as a number.
- `is_censored = true` when any of:
  - `ResultDetectionConditionText` in (`Not Detected`, `Present Below Quantification Limit`, `Below Reporting Limit`), the three values seen in the pilot. Any other non-null value is logged for review;
  - `ResultMeasureValue` starts with `<` or equals `## (Censored)`;
  - `MeasureQualifierCode`, split on `;`, includes `K` (pilot: only on rows already dropped by filter 4; kept as a safeguard).
- **Non-detect value:** `result_value_std = detection_limit / 2`, converted to `result_unit_std` from the detection limit's unit. If no detection limit is reported but the value is `<x`, `x` is used as the limit; otherwise `result_value_std` is null (the row is dropped by filter 9).
  - Pilot: all 148 null-valued rows were non-detects (about 1.5% of rows). Substitution is acceptable at this rate ([EPA-adapted guidance, table in PMC10494629](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10494629/)) but biased when non-detects are common ([CLU-IN handout](https://clu-in.org/conf/tio/ltmo/Nondetects_handout.pdf)). `is_censored` is kept so ROS/MLE methods can replace it later; re-check the censored share per parameter at the 30–50 dam stage.
- `>` values (above a range) are kept at the reported value with `is_censored = true`. None in the pilot.
- `NO DATA` and other unparseable strings: null value (dropped by filter 9).

#### 3.6.4 Units

**Missing raw unit:** the pilot's blank-unit rows were non-detects, so the unit is taken from, in order: `detection_limit_unit`, then the pcode's unit (kept codes above), then `std units` for pH (STORET reports pH with no unit). Rows still without a unit are dropped.

**Conversions** (raw unit → `result_unit_std`). Unit strings are compared case-insensitively and with spaces removed (`mg/l as N` = `mg/L as N`).

| `parameter_std` | Raw unit | Factor | Notes |
|---|---|---|---|
| all mg/L parameters | `mg/l`, `mg/L` | 1 | Case only |
| all mg/L parameters | `ug/L`, `ug/l` | 0.001 | Not in pilot; included for scale-up |
| `specific_conductance` | `uS/cm`, `umho/cm`, `uS/cm @25C` | 1 | Identical units. A `ResultTemperatureBasisText` other than blank or `25 deg C` (any case) is logged |
| `dissolved_oxygen` | `mg/L` | 1 | |
| `do_saturation` | `%`, `% saturatn` | 1 | |
| `turbidity` | `NTU`, `NTRU`, `FNU` | 1 | See below |
| `nitrate`, `ammonia` | `mg/l as N` | 1 | |
| `nitrate`, `ammonia` | bare `mg/L` | 1 | **Assumed as N**; `unit_basis_assumed = true` |
| `orthophosphate`, `phosphorus` | `mg/l as P` | 1 | |
| `orthophosphate`, `phosphorus` | bare `mg/L` | 1 | **Assumed as P**; `unit_basis_assumed = true` |

Any other (parameter, unit) pair is dropped and logged, not guessed.

**Turbidity:** NTU (EPA 180.1), FNU (ISO 7027) and NTRU come from different instrument designs and have no conversion ([USGS TWRI 9-A6.7](https://pubs.usgs.gov/publication/twri09A6.7), [USGS memo 2004.03](https://water.usgs.gov/admin/memo/QW/qw04.03.html)). They are treated as comparable (factor 1) and `result_unit_raw` is kept, so analyses can test sensitivity by restricting to one unit.

**Bare mg/L nutrients** (about 230 STORET pilot rows): the chemical form decides the value by up to 4.4× (1 mg/L nitrate as N = 4.43 mg/L as NO3). `download_wqp_results.py` requests `dataProfile=resultPhysChem`, which adds `MethodSpeciationName`. When it is `as N`/`as NO3`/`as NH4` (nitrate, ammonia) or `as P`/`as PO4` (orthophosphate, phosphorus), the value is converted to the N or P basis (`SPECIATION_FACTORS` in `etl/common/wq_rules.py`) and not flagged. Rows with blank or mismatched speciation keep the as N / as P assumption and `unit_basis_assumed = true`. Checked 2026-10-17: NJ_MSU and 11113300 fill it for all bare mg/L nutrients (all as N / as P); 21OHIO_WQX leaves about 60% of orthophosphate blank.

#### 3.6.5 Deduplication

Run after filters and units, in two passes:

1. **Exact repeats** (the same result downloaded in more than one batch, e.g. overlapping dam windows at one station): same `measurement_id`. Keep one.
2. **Same measurement under two activities** (pilot: one Ohio pair, `…:SR` vs `…:FM`): same `station_id`, `sample_date`, `sample_time`, `parameter_std`, `sample_fraction_std` and `result_value_std`. Keep one, preferring `Sample-Routine`, then `Field Msr/Obs`, then the lowest `measurement_id`.

Results with the same key but **different** values are kept (they are genuinely separate readings, e.g. depth profiles); analyses aggregate per visit as needed. Pilot: after filters, only 2 rows were duplicated and no group spanned both providers.

#### 3.6.6 Audit table

`wq_measurement_raw.parquet`: every downloaded row for the case study before any filter: all WQP columns as verbatim strings, plus `measurement_id` and `source_file` (the batch CSV). It has no primary key, since exact repeats across batches are kept for audit. Only batch files the download manifest lists as `success` are read. Written for pilot and medium-sample case studies; skipped at national scale if size requires.

#### 3.6.7 Open items

None currently.

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
| `wq_measurement` | `station_id` in `wq_station`; `sample_date` non-null; `result_unit_std` non-null wherever `result_value_std` is non-null; `parameter_std` in the 3.6.1 mapping; `result_value_std` within plausible range per parameter (filter 10) |
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
