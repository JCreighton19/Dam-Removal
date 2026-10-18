"""
Clean downloaded WQP results into the wq_measurement warehouse table.

Applies the rules in docs/warehouse_spec.md section 3.6 (the rule tables
live in etl/common/wq_rules.py) and writes, to data/warehouse/<case_study>/:

    wq_measurement.parquet      one row per cleaned result
    wq_measurement_raw.parquet  every downloaded row, untouched, for audit

Inputs:

    data/raw/wqp/results/<case_study>/pilot_download/download_manifest.csv and the
        results_batch_*.csv files it lists as downloaded
        (download_wqp_results.py)
    data/warehouse/<case_study>/wq_station.parquet (load_core_tables.py),
        which every station_id must exist in

Steps, in order: scope to the mapped parameters and to wq_station
stations; row filters 1-8; resolve units; parse values and non-detects;
convert units; filters 9-10 (no value or date, implausible value);
deduplicate. The number of rows each step drops is printed, so a run can
be checked against the counts in the spec.

wq_measurement is validated (spec section 4) before anything is written;
a failed run leaves the previous files untouched. Must run after
load_core_tables.py.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pyarrow as pa

from etl.common import wq_rules as rules
from etl.common.config import default_case_study, load_config
from etl.common.io import require_columns, require_file
from etl.common.warehouse import schema_errors, warehouse_dir, write_parquet_atomic

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATE = "date32[pyarrow]"
TIME = pd.ArrowDtype(pa.time64("us"))

# Section 1, "Generated keys": the fields hashed into measurement_id.
MEASUREMENT_ID_FIELDS = [
    "OrganizationIdentifier",
    "ActivityIdentifier",
    "CharacteristicName",
    "ResultSampleFractionText",
    "ResultMeasureValue",
    "ResultMeasure/MeasureUnitCode",
    "ResultDetectionConditionText",
    "ResultAnalyticalMethod/MethodIdentifier",
]

# WQP columns the cleaning rules read.
REQUIRED_COLUMNS = set(MEASUREMENT_ID_FIELDS) | {
    "MonitoringLocationIdentifier",
    "ProviderName",
    "ActivityStartDate",
    "ActivityStartTime/Time",
    "ActivityStartTime/TimeZoneCode",
    "ActivityTypeCode",
    "ActivityMediaName",
    "ActivityMediaSubdivisionName",
    "SampleAquifer",
    "ResultStatusIdentifier",
    "ResultValueTypeName",
    "StatisticalBaseCode",
    "MeasureQualifierCode",
    "USGSPCode",
    "ResultTemperatureBasisText",
    "DetectionQuantitationLimitMeasure/MeasureValue",
    "DetectionQuantitationLimitMeasure/MeasureUnitCode",
    "DetectionQuantitationLimitTypeName",
    "HydrologicCondition",
    "ActivityDepthHeightMeasure/MeasureValue",
    "ActivityDepthHeightMeasure/MeasureUnitCode",
}

SCHEMA = {
    "measurement_id": "int64",
    "station_id": "string",
    "organization_id": "string",
    "provider": "string",
    "activity_id": "string",
    "sample_date": DATE,
    "sample_time": TIME,
    "time_zone": "string",
    "parameter": "string",
    "parameter_std": "string",
    "usgs_pcode": "string",
    "result_value_raw": "string",
    "result_value_std": "float64",
    "result_unit_raw": "string",
    "result_unit_std": "string",
    "unit_basis_assumed": "bool",
    "detection_condition": "string",
    "detection_limit": "float64",
    "detection_limit_unit": "string",
    "detection_limit_type": "string",
    "is_censored": "bool",
    "sample_fraction": "string",
    "sample_fraction_std": "string",
    "activity_type": "string",
    "media": "string",
    "result_status": "string",
    "value_type": "string",
    "qualifier_code": "string",
    "hydrologic_condition": "string",
    "depth_m": "float64",
}


# --------------------------------------------------
# Loading
# --------------------------------------------------

def downloaded_batch_files(download_dir):
    """The batch CSVs the download manifest lists as successfully downloaded.

    Reading from the manifest rather than globbing means leftover batch
    files from an earlier, differently-scoped run are never loaded.
    """
    manifest_path = require_file(
        download_dir / "download_manifest.csv",
        "WQP download manifest (run download_wqp_results.py first)",
    )
    manifest = pd.read_csv(manifest_path, dtype={"file": "string", "status": "string"})

    failed = manifest.loc[manifest["status"] == "failed", "batch"].tolist()
    if failed:
        raise RuntimeError(
            f"Download manifest has failed batches {failed}; re-run "
            "download_wqp_results.py until none fail."
        )

    listed = manifest.loc[manifest["status"] == "success", "file"].dropna()
    files = [require_file(download_dir / name, "Batch file listed in the manifest")
             for name in listed]

    unlisted = sorted(
        {p.name for p in download_dir.glob("results_batch_*.csv")} - set(listed)
    )
    if unlisted:
        print(f"Ignoring {len(unlisted)} batch file(s) not in the manifest, e.g. {unlisted[0]}")

    return files


def load_raw_results(files):
    """Every downloaded row, all columns as verbatim strings."""
    if not files:
        raise ValueError("The download manifest lists no batch files with data.")

    frames = []
    for path in files:
        frame = pd.read_csv(path, dtype="string", keep_default_na=False, na_values=[""])
        frame.insert(0, "source_file", path.name)
        frames.append(frame)

    raw = pd.concat(frames, ignore_index=True)
    require_columns(raw, REQUIRED_COLUMNS, "WQP result batches")
    raw.insert(0, "measurement_id", measurement_ids(raw))
    return raw


def measurement_ids(frame):
    """Signed int64 from the first 8 bytes of SHA-256 over the ID fields.

    Nulls are joined as empty strings (spec section 1, "Generated keys").
    """
    joined = frame[MEASUREMENT_ID_FIELDS[0]].fillna("")
    for field in MEASUREMENT_ID_FIELDS[1:]:
        joined = joined + "|" + frame[field].fillna("")
    return joined.map(
        lambda text: int.from_bytes(
            hashlib.sha256(text.encode("utf-8")).digest()[:8], "big", signed=True
        )
    ).astype("int64")


# --------------------------------------------------
# Cleaning
# --------------------------------------------------

class DropLog:
    """Applies drop masks and records how many rows each rule removed."""

    def __init__(self):
        self.rows = []

    def drop(self, frame, mask, rule):
        mask = mask.fillna(False).astype(bool)
        self.rows.append((rule, int(mask.sum())))
        return frame.loc[~mask]

    def print(self, start, end):
        print("\n=== Rows dropped by rule ===")
        width = max(len(rule) for rule, _ in self.rows)
        for rule, count in self.rows:
            print(f"  {rule:<{width}}  {count:>8,}")
        print(f"  {'Downloaded rows':<{width}}  {start:>8,}")
        print(f"  {'Kept rows':<{width}}  {end:>8,}")


def _split_codes(series):
    """Qualifier codes as sets: "J;MI" -> {"J", "MI"}."""
    return series.fillna("").str.split(";").map(lambda codes: {c.strip() for c in codes})


def _print_counts(title, series, limit=20):
    counts = series.value_counts(dropna=False)
    if counts.empty:
        return
    print(f"\n{title}:")
    print(counts.head(limit).to_string())


def apply_filters(df, station_ids, log):
    """Scope to mapped parameters and known stations, then filters 1-8 (3.6.2)."""
    df = log.drop(df, ~df["CharacteristicName"].isin(rules.PARAMETERS),
                  "Parameter not in the 3.6.1 mapping")
    df = log.drop(df, ~df["MonitoringLocationIdentifier"].str.strip().isin(station_ids),
                  "Station not in wq_station")

    df = log.drop(df, df["ActivityTypeCode"].str.startswith(rules.QC_ACTIVITY_PREFIX),
                  "1. Quality-control activity")
    df = log.drop(
        df,
        ~df["ActivityMediaName"].isin(rules.KEPT_MEDIA)
        | df["ActivityMediaSubdivisionName"].isin(rules.EXCLUDED_MEDIA_SUBDIVISIONS)
        | df["SampleAquifer"].notna(),
        "2. Not surface water",
    )
    df = log.drop(df, df["ResultStatusIdentifier"].isin(rules.EXCLUDED_STATUSES),
                  "3. Rejected status")
    df = log.drop(df, df["ResultValueTypeName"].isin(rules.EXCLUDED_VALUE_TYPES),
                  "4. Calculated value")
    df = log.drop(df, df["StatisticalBaseCode"].notna(), "5. Statistical summary")
    df = log.drop(
        df,
        df["ResultSampleFractionText"].isin(rules.SEDIMENT_FRACTIONS)
        | rules.normalize_unit(df["ResultMeasure/MeasureUnitCode"]).isin(rules.SEDIMENT_UNITS),
        "6. Sediment result",
    )
    df = log.drop(
        df,
        _split_codes(df["MeasureQualifierCode"]).map(lambda c: bool(c & rules.DROP_QUALIFIERS)),
        "7. Holding time exceeded",
    )
    df = log.drop(df, rules.normalize_pcode(df["USGSPCode"]).isin(rules.DROP_PCODES),
                  "8. NWIS alternate-form or lab code")
    return df.copy()


def resolve_units(df):
    """Fill a missing result unit (3.6.4) and assign parameter_std (3.6.1).

    Order: the reported unit, the detection-limit unit, the kept pcode's
    unit, then the parameter default (pH).
    """
    pcode = rules.normalize_pcode(df["USGSPCode"])
    parameter_std = df["CharacteristicName"].map(rules.PARAMETERS).astype("string")

    unit = (
        df["ResultMeasure/MeasureUnitCode"]
        .fillna(df["DetectionQuantitationLimitMeasure/MeasureUnitCode"])
        .fillna(pcode.map(rules.PCODE_UNITS).astype("string"))
        .fillna(parameter_std.map(rules.DEFAULT_UNITS).astype("string"))
    )

    is_saturation = (parameter_std == "dissolved_oxygen") & rules.normalize_unit(unit).isin(
        rules.DO_SATURATION_UNITS
    )
    df["parameter_std"] = parameter_std.mask(is_saturation, "do_saturation")
    df["usgs_pcode"] = pcode
    df["unit_resolved"] = unit
    return df


def parse_values(df):
    """Numeric value in its native unit, censoring and the unit to convert from (3.6.3)."""
    value = df["ResultMeasureValue"].str.strip().str.removeprefix("=")
    starts_lt = value.str.startswith("<").fillna(False)
    starts_gt = value.str.startswith(">").fillna(False)
    number = pd.to_numeric(value.str.replace(r"^[<>]=?", "", regex=True), errors="coerce")

    condition = df["ResultDetectionConditionText"]
    censored = (
        condition.isin(rules.CENSORED_CONDITIONS).fillna(False)
        | value.isin(rules.CENSORED_VALUES).fillna(False)
        | starts_lt
        | starts_gt
        | _split_codes(df["MeasureQualifierCode"]).map(lambda c: bool(c & rules.CENSORED_QUALIFIERS))
    )

    unknown = condition[condition.notna() & ~condition.isin(rules.CENSORED_CONDITIONS)]
    _print_counts("Detection conditions not in the censored list (kept as detected; review)", unknown)

    limit = pd.to_numeric(df["DetectionQuantitationLimitMeasure/MeasureValue"], errors="coerce")
    limit_unit = df["DetectionQuantitationLimitMeasure/MeasureUnitCode"].fillna(df["unit_resolved"])

    # Below-limit non-detects take half the detection limit. A "<0.5" value
    # with no separate limit states its own limit. ">" values keep their number.
    below = censored & ~starts_gt
    effective_limit = limit.where(limit.notna(), number.where(starts_lt))
    use_limit_unit = below & limit.notna()

    df["value_native"] = number.where(~below, effective_limit / 2)
    df["conversion_unit"] = df["unit_resolved"].where(~use_limit_unit, limit_unit)
    df["is_censored"] = censored.astype(bool)
    df["detection_limit"] = limit
    return df


def convert_units(df, log):
    """Convert value_native to result_unit_std; drop unconvertible pairs (3.6.4)."""
    keys = list(zip(df["parameter_std"], rules.normalize_unit(df["conversion_unit"]).fillna("")))
    factor = pd.Series([rules.CONVERSIONS.get(k, (None, False))[0] for k in keys],
                       index=df.index, dtype="float64")
    assumed = pd.Series([rules.CONVERSIONS.get(k, (None, False))[1] for k in keys],
                        index=df.index, dtype=bool)

    pairs = df["parameter_std"] + " | " + df["conversion_unit"]
    _print_counts(
        "Unconvertible (parameter_std | unit) pairs (dropped; add to wq_rules if valid)",
        pairs[factor.isna() & df["unit_resolved"].notna()],
    )

    # A bare mg/L nutrient's chemical form comes from MethodSpeciationName when
    # the download has it (resultPhysChem profile); otherwise it stays assumed.
    if "MethodSpeciationName" in df.columns:
        speciation = df["MethodSpeciationName"].str.strip().map(rules.SPECIATION_FACTORS)
        basis_matches = speciation.str[0] == df["parameter_std"].map(rules.NUTRIENT_BASIS)
        known = (assumed & basis_matches.fillna(False)).astype(bool)
        factor = factor.where(~known, factor * pd.to_numeric(speciation.str[1]))
        assumed = assumed & ~known

    df["result_value_std"] = df["value_native"] * factor
    df["result_unit_std"] = df["parameter_std"].map(rules.STANDARD_UNITS).astype("string")
    df["unit_basis_assumed"] = assumed
    df["sample_date_parsed"] = pd.to_datetime(df["ActivityStartDate"], errors="coerce")

    df = log.drop(df, df["unit_resolved"].isna(), "Unit missing after fallbacks")
    df = log.drop(df, factor.reindex(df.index).isna(), "Unit not convertible")
    df = log.drop(df, df["sample_date_parsed"].isna(), "9a. Unparseable sample date")
    df = log.drop(df, df["result_value_std"].isna(), "9b. No value and no detection limit")

    conductance = df["parameter_std"] == "specific_conductance"
    basis = rules.normalize_unit(df.loc[conductance, "ResultTemperatureBasisText"])
    _print_counts(
        "Specific conductance with a temperature basis other than 25 deg C (kept; review)",
        df.loc[conductance, "ResultTemperatureBasisText"][basis.notna() & ~basis.isin(rules.CONDUCTANCE_BASES)],
    )
    return df


def drop_implausible(df, log):
    low = df["parameter_std"].map(lambda p: rules.PLAUSIBLE_RANGES[p][0])
    high = df["parameter_std"].map(lambda p: rules.PLAUSIBLE_RANGES[p][1])
    outside = ~df["result_value_std"].between(low, high)
    _print_counts("Implausible values by parameter (dropped)", df.loc[outside, "parameter_std"])
    return log.drop(df, outside, "10. Outside plausible range")


def deduplicate(df, log):
    """Exact repeats, then one measurement filed under two activities (3.6.5)."""
    df = log.drop(df, df["measurement_id"].duplicated(), "Dedup 1. Repeated measurement_id")

    preference = {name: rank for rank, name in enumerate(rules.ACTIVITY_PREFERENCE)}
    df = df.assign(
        _activity_rank=df["ActivityTypeCode"].map(preference).fillna(len(preference))
    ).sort_values(["_activity_rank", "measurement_id"], kind="mergesort")

    key = [
        "MonitoringLocationIdentifier", "sample_date_parsed", "ActivityStartTime/Time",
        "parameter_std", "sample_fraction_std", "result_value_std",
    ]
    df = log.drop(df, df.duplicated(key, keep="first"), "Dedup 2. Same measurement, two activities")
    return df.drop(columns="_activity_rank").sort_values("measurement_id")


def to_meters(value, unit):
    factor = rules.normalize_unit(unit).map(rules.DEPTH_TO_METERS)
    return (pd.to_numeric(value, errors="coerce") * factor.astype("float64")).astype("float64")


def to_time(series):
    parsed = pd.to_datetime(series, format="%H:%M:%S", errors="coerce")
    times = [t.time() if pd.notna(t) else None for t in parsed]
    return pd.Series(pd.arrays.ArrowExtensionArray(pa.array(times, type=pa.time64("us"))),
                     index=series.index)


def build_table(df):
    """Rename and type the cleaned rows into the 3.6 schema."""
    is_field = df["parameter_std"].isin(rules.FIELD_PARAMETERS)
    fraction_std = df["ResultSampleFractionText"].replace(rules.FRACTION_ALIASES)

    table = pd.DataFrame({
        "measurement_id": df["measurement_id"],
        "station_id": df["MonitoringLocationIdentifier"].str.strip(),
        "organization_id": df["OrganizationIdentifier"],
        "provider": df["ProviderName"],
        "activity_id": df["ActivityIdentifier"],
        "sample_date": df["sample_date_parsed"].astype(DATE),
        "sample_time": to_time(df["ActivityStartTime/Time"]),
        "time_zone": df["ActivityStartTime/TimeZoneCode"],
        "parameter": df["CharacteristicName"],
        "parameter_std": df["parameter_std"],
        "usgs_pcode": df["usgs_pcode"],
        "result_value_raw": df["ResultMeasureValue"],
        "result_value_std": df["result_value_std"].astype("float64"),
        "result_unit_raw": df["ResultMeasure/MeasureUnitCode"],
        "result_unit_std": df["result_unit_std"],
        "unit_basis_assumed": df["unit_basis_assumed"].astype(bool),
        "detection_condition": df["ResultDetectionConditionText"],
        "detection_limit": df["detection_limit"].astype("float64"),
        "detection_limit_unit": df["DetectionQuantitationLimitMeasure/MeasureUnitCode"],
        "detection_limit_type": df["DetectionQuantitationLimitTypeName"],
        "is_censored": df["is_censored"].astype(bool),
        "sample_fraction": df["ResultSampleFractionText"],
        "sample_fraction_std": fraction_std.mask(is_field),
        "activity_type": df["ActivityTypeCode"],
        "media": df["ActivityMediaName"],
        "result_status": df["ResultStatusIdentifier"],
        "value_type": df["ResultValueTypeName"],
        "qualifier_code": df["MeasureQualifierCode"],
        "hydrologic_condition": df["HydrologicCondition"],
        "depth_m": to_meters(
            df["ActivityDepthHeightMeasure/MeasureValue"],
            df["ActivityDepthHeightMeasure/MeasureUnitCode"],
        ),
    }).reset_index(drop=True)

    for column, dtype in SCHEMA.items():
        if dtype == "string":
            table[column] = table[column].astype("string")
    return table


def clean_results(raw, station_ids):
    """Raw WQP rows -> wq_measurement rows, printing drops per rule."""
    log = DropLog()
    df = apply_filters(raw, station_ids, log)
    df = resolve_units(df)

    # Fraction standardization feeds dedup, so it is computed before it.
    df["sample_fraction_std"] = (
        df["ResultSampleFractionText"].replace(rules.FRACTION_ALIASES)
        .mask(df["parameter_std"].isin(rules.FIELD_PARAMETERS))
    )

    df = parse_values(df)
    df = convert_units(df, log)
    df = drop_implausible(df, log)
    df = deduplicate(df, log)

    log.print(len(raw), len(df))
    return build_table(df)


# --------------------------------------------------
# Validation (spec section 4)
# --------------------------------------------------

def validate(table, station_ids):
    errors = schema_errors("wq_measurement", table, SCHEMA, ["measurement_id"])
    if errors:
        return errors

    if bad := sorted(set(table["station_id"]) - set(station_ids))[:10]:
        errors.append(f"station_id not in wq_station, e.g. {bad}")
    if table["sample_date"].isna().any():
        errors.append("null sample_date")
    if (table["result_value_std"].notna() & table["result_unit_std"].isna()).any():
        errors.append("result_value_std without result_unit_std")
    if bad := sorted(set(table["parameter_std"].dropna()) - set(rules.STANDARD_UNITS)):
        errors.append(f"parameter_std not in the 3.6.1 mapping: {bad}")

    for parameter, (low, high) in rules.PLAUSIBLE_RANGES.items():
        values = table.loc[table["parameter_std"] == parameter, "result_value_std"]
        if not values.between(low, high).all():
            errors.append(f"{parameter}: values outside [{low}, {high}]")
    return errors


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--case-study",
        default=default_case_study(),
        help="Name of a config file under config/ (without the .yaml extension).",
    )
    args = parser.parse_args()

    cfg = load_config(args.case_study)
    output_dir = warehouse_dir(cfg)
    download_dir = cfg.wqp_results_dir / "pilot_download"

    print(f"Case study: {cfg.name}")

    station_path = require_file(
        output_dir / "wq_station.parquet",
        "wq_station table (run load_core_tables.py first)",
    )
    station_ids = set(pd.read_parquet(station_path, columns=["station_id"])["station_id"])
    print(f"Stations in wq_station: {len(station_ids):,}")

    files = downloaded_batch_files(download_dir)
    raw = load_raw_results(files)
    print(f"Read {len(raw):,} rows from {len(files):,} batch file(s)")

    table = clean_results(raw, station_ids)

    errors = validate(table, station_ids)
    if errors:
        raise ValueError(
            "Validation failed; no warehouse files were written:\n  - "
            + "\n  - ".join(errors)
        )

    print("\nRows by parameter_std:")
    print(table["parameter_std"].value_counts().to_string())
    print(f"Censored: {table['is_censored'].sum():,}; "
          f"unit basis assumed: {table['unit_basis_assumed'].sum():,}")

    write_parquet_atomic(raw, output_dir / "wq_measurement_raw.parquet")
    write_parquet_atomic(table, output_dir / "wq_measurement.parquet")
    print(f"\nWrote {len(raw):,} raw and {len(table):,} cleaned rows to {output_dir}")


if __name__ == "__main__":
    main()
