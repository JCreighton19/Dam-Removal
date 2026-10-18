# Warehouse Table Spec

Column-level spec for the curated Parquet tables queried with DuckDB. It
mirrors `DS5500 Capstone ERD final.jpeg`: every table and column in the ERD
is listed here with its source, type, the pipeline stage that writes it, and
the rule that produces it.

Status key used below:
- **Ready**: the inputs already exist in the current pipeline.
- **Step 2**: depends on the WQP cleaning rules (decided 2026-10-17, section 3.6).
- **Step 5**: contextual data (sources and method in "Step 5 context data: decisions").

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
| `land_cover` | Step 5 | `load_land_cover.py` (after `download_nlcd`) | Annual NLCD clips + `station_catchment` |
| `land_cover_codes` | Ready | static, written once | NLCD legend (section 3.9) |
| `climate` | Step 5 | `load_climate.py` (after `download_climate`) | PRISM monthly clips + `station_catchment` |

The two new loader stages leave the existing ETL scripts untouched. They read the outputs those scripts already produce (gpkg, CSV) and write the curated tables, so the existing per-stage outputs remain the intermediate layer.

---

## 3. Tables

### 3.1 `dam_removal`

- **Grain:** one row per dam removal in the case study.
- **PK:** `AR_ID`.
- **Rows:** dams kept by `snap_features.py` (`filter_to_sampled_dams` for sampled case studies) that snapped to a reach.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `AR_ID` | TEXT PK | American Rivers' ID for the removal; the dam's key | dam CSV `AR_ID` | Cast to string |
| `NABI_ID` | TEXT | ID in the National Aquatic Barrier Inventory, if linked | dam CSV `NABI_ID` | |
| `Dam_Name` | TEXT | Dam name | dam CSV `Dam_Name` | |
| `Year_Removed` | SMALLINT | Year the dam was removed; splits before from after | dam CSV `Year_Removed` | Non-numeric or 0 → null (same rule as `report_pilot_eligibility.prepare_links`) |
| `Latitude` | DOUBLE | Reported latitude | dam CSV `Latitude` | **Original** coordinate |
| `Longitude` | DOUBLE | Reported longitude | dam CSV `Longitude` | **Original** coordinate |
| `City_County` | TEXT | City or county | dam CSV `City_County` | |
| `River` | TEXT | River the dam was on | dam CSV `River` | |
| `HUC8` | CHAR(8) | 8-digit watershed the dam is in | pipeline | The HUC8 the dam was clipped to in `snap_features.py` (authoritative), zero-padded. Not the CSV's own HUC8 field. |
| `State` | TEXT | State(s) the dam is in | dam CSV `State` | Two-letter code; dams on a state line list each, e.g. `"MN; ND"` |
| `Dam_Height_ft` | DOUBLE | Dam height (ft) | dam CSV `Dam_Height_ft` | Non-numeric → null |
| `Dam_Length_ft` | DOUBLE | Dam length (ft) | dam CSV `Dam_Length_ft` | Non-numeric → null |
| `Year_built` | SMALLINT | Year the dam was built | dam CSV **`Year_Built`** | Source spelling differs from the ERD; non-numeric or 0 → null |
| `Original_Use` | TEXT | What the dam was built for (mill, hydropower, …) | dam CSV `Original_Use` | |
| `Type_Material` | TEXT | Construction type (earth, concrete, …) | dam CSV `Type_Material` | |
| `Reason_for_removal` | TEXT | Why it was removed | dam CSV `Reason_for_removal` | |
| `Miles_Restored_US` | DOUBLE | River miles reopened upstream | dam CSV `Miles_Restored_US` | |
| `Miles_Restored_DS` | DOUBLE | River miles reopened downstream | dam CSV `Miles_Restored_DS` | |
| `Last_Updated` | DATE | When the source record was last edited | dam CSV `Last_Updated` | Parse; unparseable → null |
| `NHDPlusID` | BIGINT | Stream reach the dam sits on | gpkg `dams_snapped.NHDPlusID` | FK → `streams.nhdplus_id` |
| `snap_distance_m` | DOUBLE | How far (m) the dam was moved to land on that reach | gpkg `dams_snapped.snap_distance_m` | |
| `snap_fraction` | DOUBLE | Position along the reach, 0 = upstream end, 1 = downstream end | gpkg `dams_snapped.snap_fraction` | 0 at the reach's start, 1 at its end |
| `match_method` | TEXT | How the dam was matched to a reach | constant | `nearest_flowline` |
| `geometry` | Point, 4326 | Location on the stream line | gpkg `dams_snapped` geometry | **Snapped** location |

**Duplicate snaps.** A dam equidistant to two reaches currently appears twice in `dams_snapped`. Keep one row per `AR_ID`: the smallest `snap_distance_m`, then the smallest `NHDPlusID` as a deterministic tie-break. Log every dropped duplicate.

### 3.2 `wq_station`

- **Grain:** one row per station.
- **PK:** `station_id`.
- **Rows:** stations that appear in `dam_station_link`, i.e. network-connected to at least one dam. This keeps the table small and guarantees every FK from the link table resolves.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `station_id` | TEXT PK | WQP station ID, e.g. `USGS-01108300` | WQP `MonitoringLocationIdentifier` | |
| `station_name` | TEXT | Station name | WQP `MonitoringLocationName` | |
| `station_type` | TEXT | Kind of site (river/stream, …) | WQP `MonitoringLocationTypeName` | Already filtered to the config's `station_types` |
| `organization_id` | TEXT | ID of the agency running the station | WQP `OrganizationIdentifier` | |
| `organization_name` | TEXT | Name of that agency | WQP `OrganizationFormalName` | |
| `latitude` | DOUBLE | Reported latitude | WQP `LatitudeMeasure` | **Original** coordinate |
| `longitude` | DOUBLE | Reported longitude | WQP `LongitudeMeasure` | **Original** coordinate |
| `state_code` | TEXT | State FIPS code | WQP `StateCode` | |
| `county_code` | TEXT | County FIPS code | WQP `CountyCode` | |
| `huc8` | CHAR(8) | 8-digit watershed the station is in | pipeline | The HUC8 whose snapped file the station came from. FK → `huc8_watershed` |
| `horizontal_datum` | TEXT | Coordinate datum the location was recorded in | WQP `HorizontalCoordinateReferenceSystemDatumName` | |
| `horizontal_method` | TEXT | How the location was measured (GPS, map, …) | WQP `HorizontalCollectionMethodName` | |
| `NHDPlusID` | BIGINT | Stream reach the station sits on | gpkg `stations_snapped.NHDPlusID` | FK → `streams.nhdplus_id` |
| `snap_distance_m` | DOUBLE | How far (m) the station was moved to land on that reach | gpkg `stations_snapped.snap_distance_m` | |
| `snap_fraction` | DOUBLE | Position along the reach, 0 = upstream end, 1 = downstream end | gpkg `stations_snapped.snap_fraction` | |
| `match_method` | TEXT | How the station was matched to a reach | constant | `nearest_flowline` |
| `geometry` | Point, 4326 | Location on the stream line | gpkg `stations_snapped` geometry | **Snapped** location |

Duplicate snaps are resolved the same way as for dams (section 3.1). A station near a HUC8 boundary that appears in two HUC8s' files keeps the row from the HUC8 its point falls within.

### 3.3 `dam_station_link`

- **Grain:** one row per network-connected (dam, station) pair.
- **PK:** `(AR_ID, station_id)`.
- **Rows:** `dam_station_links_<huc8>.csv` rows with `relationship` in (`upstream`, `downstream`, `same_reach`). `unconnected` pairs are **not stored**; they remain in the intermediate CSVs for QA.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `AR_ID` | TEXT | The dam | links `dam_id` | FK → `dam_removal` |
| `station_id` | TEXT | The station | links `station_id` | FK → `wq_station` |
| `relationship` | TEXT | Where the station is relative to the dam along the river | links `relationship` | One of `upstream`, `downstream`, `same_reach` |
| `network_distance_km` | DOUBLE | Distance between them along the river (km) | links `network_distance_km` | Null when no path or fraction was available |
| `distance_method` | TEXT | How that distance was computed | links `distance_method` | `same_reach_fraction` or `network_path_fractional_endpoints` |

The links CSV columns `dam_name`, `year_removed`, `river`, `distance_to_dam_km` (a duplicate of `network_distance_km`), the reach IDs and the snap columns are dropped. They live on `dam_removal` and `wq_station`.

### 3.4 `huc8_watershed`

- **Grain:** one row per HUC8 in the case study.
- **PK:** `huc8`.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `huc8` | CHAR(8) PK | 8-digit watershed code | WBD HU8 layer `huc8` | Zero-padded |
| `name` | TEXT | Watershed name | WBD `name` | |
| `states` | TEXT | States the watershed covers | WBD `states` | e.g. `"OH,PA"` |
| `areasqkm` | DOUBLE | Area (km²) | WBD `areasqkm` | As published by WBD |
| `geometry` | MultiPolygon, 4326 | Watershed boundary | WBD geometry | Reprojected to 4326 |

### 3.5 `streams` (mapping helper)

- **Grain:** one row per NHDPlus HR flowline.
- **PK:** `nhdplus_id`.
- **Rows:** every reach upstream or downstream of a case-study dam (via the reach graph), limited to reaches whose `ReachCode` starts with a case-study HUC8, plus the snapped reach of every dam and `wq_station` (kept even across a HUC8 boundary, so every `NHDPlusID` FK resolves). The same scoping is used by the pilot map in `notebooks/stream_network_exploration.ipynb`.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `nhdplus_id` | BIGINT PK | Reach ID; dams and stations point here | `NHDFlowline.NHDPlusID` | Read via `read_nhdplus_layer` (case-insensitive) |
| `reach_code` | TEXT | National reach code | `NHDFlowline.ReachCode` | First 8 characters = HUC8 |
| `gnis_name` | TEXT | River name | `NHDFlowline.GNIS_Name` | |
| `length_km` | DOUBLE | Reach length (km) | `NHDFlowline.LengthKM` | As published |
| `feature_type` | SMALLINT | Kind of channel (stream, canal, …) | `NHDFlowline.FType` | e.g. 460 stream/river, 336 canal/ditch |
| `feature_code` | INTEGER | More detailed channel type | `NHDFlowline.FCode` | |
| `geometry` | MultiLineString, 4326 | The reach line | `NHDFlowline` geometry | Z/M dropped; simplified (tolerance ~5 m) for map rendering |

`streams` is not the source of truth for the network. Graph logic always reads the raw NHDPlus geodatabase.

### 3.6 `wq_measurement`

- **Grain:** one row per cleaned WQP result.
- **PK:** `measurement_id`.
- **Rows:** results for `wq_station` stations, limited to the parameters in the mapping table below, that survive the filters (section 3.6.2) and deduplication (3.6.5).
- **Rules source:** profiling of the `pilot_random5_seed42` downloads (`notebooks/wqs_results_exploration.ipynb`, about 10,150 target-parameter rows), decided 2026-10-17. The rule tables are implemented as data in `etl/common/wq_rules.py`, applied by `etl/clean_wq_results.py`. Re-check the value counts when the 30–50 dam sample is downloaded; unseen codes default to **kept and logged**, not silently dropped.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `measurement_id` | BIGINT PK | Stable ID for the result | generated | Section 1, "Generated keys" |
| `station_id` | TEXT | Where it was measured | `MonitoringLocationIdentifier` | FK → `wq_station` |
| `organization_id` | TEXT | Agency that measured it | `OrganizationIdentifier` | |
| `provider` | TEXT | Database it came from: USGS (`NWIS`) or states/others (`STORET`) | `ProviderName` | `NWIS` or `STORET` |
| `activity_id` | TEXT | The sampling visit it belongs to | `ActivityIdentifier` | |
| `sample_date` | DATE | Date sampled | `ActivityStartDate` | Rows with unparseable dates dropped |
| `sample_time` | TIME | Time sampled | `ActivityStartTime/Time` | Null when absent |
| `time_zone` | TEXT | Time zone of `sample_time` | `ActivityStartTime/TimeZoneCode` | |
| `parameter` | TEXT | Original WQP parameter name | `CharacteristicName` | Verbatim |
| `parameter_std` | TEXT | Standard parameter name; analyze on this | derived | Section 3.6.1 mapping. Analyses group on this, not `parameter` |
| `usgs_pcode` | TEXT | USGS parameter code | `USGSPCode` | Zero-padded to 5 characters; null for STORET |
| `result_value_raw` | TEXT | Value exactly as reported | `ResultMeasureValue` | Verbatim, e.g. `"=6.59"`, `"## (Censored)"` |
| `result_value_std` | DOUBLE | Cleaned numeric value in the standard unit; analyze on this | derived | Section 3.6.3 (parsing, non-detects) and 3.6.4 (units) |
| `result_unit_raw` | TEXT | Unit as reported | `ResultMeasure/MeasureUnitCode` | Verbatim |
| `result_unit_std` | TEXT | Standard unit of `result_value_std` | derived | The standard unit for `parameter_std` (3.6.1) |
| `unit_basis_assumed` | BOOLEAN | True if the nutrient form (as N / as P) was guessed | derived | True when the chemical form (as N / as P) was assumed, not stated (3.6.4) |
| `detection_condition` | TEXT | Lab note such as `Not Detected` | `ResultDetectionConditionText` | |
| `detection_limit` | DOUBLE | Lowest value the lab could detect | `DetectionQuantitationLimitMeasure/MeasureValue` | |
| `detection_limit_unit` | TEXT | Unit of `detection_limit` | `DetectionQuantitationLimitMeasure/MeasureUnitCode` | |
| `detection_limit_type` | TEXT | Which kind of detection limit it is | `DetectionQuantitationLimitTypeName` | |
| `is_censored` | BOOLEAN | True if below detection; value set to half the limit | derived | Section 3.6.3 |
| `sample_fraction` | TEXT | Filter label as reported | `ResultSampleFractionText` | Verbatim |
| `sample_fraction_std` | TEXT | `Dissolved` (filtered) or `Total` (unfiltered); null when not applicable | derived | Trimmed; blank → null; `Filtered, field`, `Filtered, lab`, `Filtered field and/or lab` → `Dissolved`; `Unfiltered` → `Total`; null for field parameters and `tds` (3.6.1) |
| `activity_type` | TEXT | Kind of sample (routine sample, field measurement, …) | `ActivityTypeCode` | Retained after filtering, for audit |
| `media` | TEXT | What was sampled (always water) | `ActivityMediaName` | |
| `result_status` | TEXT | Review status (`Accepted`, `Final`, `Preliminary`, …) | `ResultStatusIdentifier` | `Preliminary` is kept; filter on this column if needed |
| `value_type` | TEXT | `Actual`, `Estimated`, … | `ResultValueTypeName` | |
| `qualifier_code` | TEXT | Lab flags on the result | `MeasureQualifierCode` | |
| `hydrologic_condition` | TEXT | Flow condition when sampled | `HydrologicCondition` | NWIS only (null for all STORET rows), so informational, not a filter |
| `depth_m` | DOUBLE | Sample depth (m) | `ActivityDepthHeightMeasure/MeasureValue` + `…/MeasureUnitCode` | Converted to meters; null when absent |

#### 3.6.1 Parameter mapping and standard units

Rows whose `CharacteristicName` is not in this table are dropped. Field parameters have no meaningful fraction, so `sample_fraction_std` is null for them (`Total` and blank mean the same thing there), and for `tds`, which is dissolved by definition. For nutrients the fraction matters, so analyses group on (`parameter_std`, `sample_fraction_std`).

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

### Step 5 context data: decisions (2026-10-18)

- **Which stations:** only stations in `eligibility_pairs` with `has_both` for at least one parameter (downstream sites and upstream controls). Context data for stations that never enter the analysis is skipped.
- **Upstream area:** built from NHDPlus HR catchments and the existing reach graph (3.7), not EPA StreamCat, so it matches the network used everywhere else.
- **Land cover:** Annual NLCD Collection 1.2 (C1V2), CONUS, 1985–2025, from MRLC's free direct-download server (`https://www.mrlc.gov/downloads/sciweb1/shared/mrlc/data-bundles/Annual_NLCD_LndCov_<yyyy>_CU_C1V2.zip`, one ~1.35 GB zip per year; the same files ScienceBase hosts, but ScienceBase needs a login for scripts). No AWS requester-pays bucket and no credentials. Each zip is downloaded, clipped to the case study and deleted (reading a window straight from the remote zip fails). Public domain; citation requested.
- **Climate:** PRISM AN time series, **monthly** grids, 800 m, 1895–present, via the free PRISM web service (`https://services.nacse.org/prism/data/get/us/800m/<ppt|tmean>/<yyyymm>`, one ~40 MB zip holding a GeoTIFF per variable-month). Monthly rather than annual because water quality (especially temperature) varies more with season than with removal, so measurements are matched to their month's climate. Each zip is clipped per NHDPlus package and deleted (keeping all of them would be ~40 GB). Only years whose 12 months are final are used (PRISM revises a month for ~6 months). Free to use; the PRISM Climate Group, the URL and the access date must be cited. Each file may be downloaded at most twice per 24 h, so a zip is kept until its clips are written, a failed download is retried once at most, and a non-zip answer (the limit message) is never retried.
- **Years:** the union of `removal_year ± wq_window` over the eligible pairs. Each source's raw files are cached once under `data/raw/` and shared across case studies.
- **Method:** class areas and climate means are computed per NHDPlus catchment, then summed or area-weighted to each station through `station_catchment` (3.8, 3.10), instead of masking each station's large dissolved polygon. Climate uses area weights rather than cell centers, because an 800 m PRISM cell is often bigger than a whole catchment.
- **Coverage:** both sources cover CONUS only. Stations in Alaska, Hawaii or Puerto Rico get no land cover or climate rows.

### 3.7 `station_watershed` (Step 5)

- **Grain:** one row per eligible station (see the decisions above).
- **PK:** `station_id`. FK → `wq_station`.

**Status:** built by `etl/build_station_watershed.py` (after `build_views`, which supplies the eligible stations). It reads every case-study NHDPlus HR geodatabase and stitches them into one reach graph, so reaches crossing between packages connect. The station's own catchment is included whole, even if the station sits partway along its reach.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `station_id` | TEXT PK | The station | `wq_station` | FK → `wq_station` |
| `NHDPlusID` | BIGINT | Reach the station sits on | `wq_station.NHDPlusID` | |
| `area_km2` | DOUBLE | Land area draining to the station (km²) | `NHDPlusCatchment.AreaSqKm` | Sum over the station's reach and all of its graph ancestors |
| `n_catchments` | BIGINT | Number of small catchments making up that area | derived | |
| `nhdplus_total_area_km2` | DOUBLE | NHDPlus's own published drainage area at the station's reach | `NHDPlusFlowlineVAA.TotDASqKm` | Null if the layer is absent |
| `area_coverage` | DOUBLE | Share of the published drainage area our upstream area covers | derived | `area_km2 / nhdplus_total_area_km2` |
| `truncated` | BOOLEAN | True if the upstream area is incomplete | derived | A flow row enters an upstream reach from a reach in no downloaded package, or `area_coverage` < 0.9 |
| `geometry` | MultiPolygon, 4326 | Outline of the upstream area | `NHDPlusCatchment` | Only with `--with-outlines` (null otherwise): dissolved union, simplified ~30 m; for map QA only (later stages use `station_catchment`). Slow for large rivers, so off at national scale |

### 3.7a `station_catchment`

- **Grain:** one row per (station, upstream catchment). A bridge table, written by `build_station_watershed.py` alongside `station_watershed`.
- **PK:** `(station_id, catchment_id)`.
- **Purpose:** `land_cover` and `climate` summarize each NHDPlus catchment once, then sum to stations through this table, instead of rebuilding the reach graph.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `station_id` | TEXT | The station | `station_watershed` | FK → `station_watershed` |
| `catchment_id` | BIGINT | One NHDPlus catchment upstream of it (incl. its own) | `NHDPlusCatchment.NHDPlusID` | Same ID as the catchment's reach |

### 3.8 `land_cover` (Step 5)

- **Grain:** one row per station × year × NLCD class.
- **PK:** `(station_id, year, class_code)`.
- **Rows:** for each station, the years inside some eligible pair's window (removal year ± `wq_window`, pairs with `has_both`) that NLCD has (1985–2025); all 16 classes per station-year, zeros included.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `station_id` | TEXT | The station | `station_watershed` | FK → `station_watershed` |
| `year` | SMALLINT | Land-cover year | NLCD clip year | |
| `class_code` | SMALLINT | Land-cover class | raster value | FK → `land_cover_codes` |
| `area_km2` | DOUBLE | Area of that class upstream of the station (km²) | derived | Pixel count × 0.0009 (30 m pixels in NLCD's equal-area projection) |
| `percent_area` | DOUBLE | Share of the upstream area in that class (%) | derived | Class pixels / valid pixels × 100; sums to 100 per station-year; null if no valid pixels |
| `valid_fraction` | DOUBLE | Share of the upstream area with usable land-cover data | derived | Valid-pixel area / `station_watershed.area_km2`, capped at 1 (pixel-center burning differs slightly from published areas). Low values mean nodata or catchments outside the clips |
| `source` | TEXT | Land-cover product | constant | `annual_nlcd_c1v2` |

Computed per catchment, then summed to each station through `station_catchment` (3.7a), rather than masking each station's dissolved polygon. **Source:** Annual NLCD C1V2 (decisions above); `source` = `annual_nlcd_c1v2`.

**Status:** raster download done (`etl/download_land_cover.py`, stage `download_nlcd`): for each needed year, the CONUS zip is downloaded, clipped to each NHDPlus package holding an eligible station's catchment (on the original 30 m NLCD grid, 10-pixel pad), and deleted. Clips: `data/raw/nlcd/annual_c1v2/<package>/LndCov_<year>.tif`, shared across case studies. Table load: `etl/load_land_cover.py` (stage `load_land_cover`). Per package, the needed catchments are burned into the clip grid once (pixel centers); each year's pixels are counted per catchment and class in one pass, then summed to stations through a sparse station × catchment matrix. Stations spanning packages are summed across them.

### 3.9 `land_cover_codes`

- **Grain:** one row per NLCD class.
- **PK:** `class_code`.
- Static, from the Annual NLCD legend; written by `load_land_cover.py`. Columns: `class_code` (SMALLINT), `class_label` (TEXT), `class_group` (TEXT: Water, Developed, Barren, Forest, Shrubland, Herbaceous, Agriculture, Wetlands, for coarser summaries).

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


### 3.10 `climate` (Step 5)

- **Grain:** one row per station × year × month.
- **PK:** `(station_id, year, month)`.
- **Rows:** for each station, all 12 months of the years inside some eligible pair's window (removal year ± `wq_window`, pairs with `has_both`), from 1895 to the last year whose months are all final. Unlike `land_cover`, pre-1985 removals are covered.
- **Status:** written by `etl/load_climate.py` from the clips of `etl/download_climate.py`.

| Column | Type | Meaning | Source | Rule / notes |
|---|---|---|---|---|
| `station_id` | TEXT | The station | `station_watershed` | FK → `station_watershed` |
| `year` | SMALLINT | Calendar year | PRISM clip | |
| `month` | TINYINT | Calendar month (1–12) | PRISM clip | |
| `temperature_mean_c` | DOUBLE | Mean air temperature over the upstream area that month (°C) | derived | Area-weighted mean of PRISM `tmean` cells; null if no cell has data |
| `precipitation_mm` | DOUBLE | Total precipitation that month, averaged over the upstream area (mm) | derived | Area-weighted mean of PRISM `ppt` cells; null if no cell has data |
| `valid_fraction` | DOUBLE | Share of the upstream area with PRISM data for both variables | derived | Valid area / `station_watershed.area_km2`, capped at 1 (PRISM is CONUS only) |
| `source` | TEXT | Data source | constant | `prism_an_800m_monthly` |

**How:** per NHDPlus package, the needed catchments are burned into a 30 m Albers grid and the PRISM grid is projected onto it once (nearest cell). Counting 30 m pixels per (catchment, PRISM cell) and summing through `station_catchment` gives each station's area in each cell; a month's value is the area-weighted mean over cells with data. Annual values are a `GROUP BY` away (mean temperature, summed precipitation).

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
| `station_watershed` | `station_id` in `wq_station`; `area_km2` > 0 |
| `station_catchment` | Unique `(station_id, catchment_id)` |
| `land_cover` | `class_code` in `land_cover_codes`; `percent_area` sums to ~100 per (station, year); `valid_fraction` in [0, 1]; one row per needed station-year × class |
| `climate` | `temperature_mean_c` in [−45, 45]; `precipitation_mm` in [0, 3000]; `valid_fraction` in [0, 1]; one row per needed station-year × month |

---

## 5. DuckDB views

**Status:** built by `etl/build_views.py` (last pipeline stage). `warehouse.duckdb` defines views only, over absolute Parquet paths:

- One view per table (`dam_removal`, `wq_station`, `dam_station_link`, `huc8_watershed`, `streams`, `wq_measurement`, `wq_measurement_raw`), so notebooks don't hardcode paths.
- **`eligibility_settings`:** one row: `years_before` / `years_after` (config `wq_window`) and `min_samples` / `min_years` (config `eligibility`, defaults 3 and 2).
- **`measurement_window`:** `dam_station_link` ⋈ `dam_removal` ⋈ `wq_station` ⋈ `wq_measurement`, one row per measurement × linked dam, for dams with a `Year_Removed`. `period` is `pre` / `post` / `removal_year` / `outside` (same definition as `etl/report_pilot_eligibility.py`: the removal year is in neither window). `side` is `relationship`, except `same_reach` stations are placed `downstream` / `upstream` by comparing `snap_fraction` with the dam's (null if equal).
- **`eligibility_pairs`:** one row per dam × station × `parameter_std` × `sample_fraction_std` with any pre/post data: `n_pre`, `n_post`, `years_pre`, `years_post` (distinct years); `has_pre` / `has_post` when a side has ≥ `min_samples` samples in ≥ `min_years` years; `has_both`.
- **`eligibility_dams`:** one row per dam × parameter × fraction: `downstream_stations_both`, `upstream_stations_both`; `has_both` (a downstream station has both sides) and `has_upstream_control` (an upstream one does too).
- **`eligibility_summary`:** one row per parameter × fraction: `dams_eligible`, `dams_with_upstream_control`, `dams_with_removal_year`.
- **Replaces `report_pilot_eligibility.py`**, which is out of the pipeline but kept until checked: on cleaned data the views should find the same or fewer dams (query `eligibility_pairs` with `n_pre > 0 AND n_post > 0` to match its one-sample rule), with differences explained by cleaning.
- **Context joins:** `land_cover` joins to measurements on `station_id` and `year(sample_date) = year`; `climate` also on `month(sample_date) = month`. Those are join conditions, not FKs.
