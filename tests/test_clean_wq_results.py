"""Validation checks for etl/clean_wq_results.py.

Builds synthetic WQP result batches with one row per cleaning rule
(docs/warehouse_spec.md 3.6), runs the real `main()` against a throwaway
project root, and checks which rows survive and how they are converted.

Run directly: `python tests/test_clean_wq_results.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl import clean_wq_results as cwr
from etl.common.config import PROJECT_ROOT_ENV_VAR

STATIONS = ["S1", "S2"]

DEFAULTS = {
    "OrganizationIdentifier": "ORG",
    "MonitoringLocationIdentifier": "S1",
    "ProviderName": "STORET",
    "ActivityStartDate": "2015-06-08",
    "ActivityStartTime/Time": "10:34:00",
    "ActivityStartTime/TimeZoneCode": "EST",
    "ActivityTypeCode": "Field Msr/Obs",
    "ActivityMediaName": "Water",
    "ActivityMediaSubdivisionName": "Surface Water",
    "CharacteristicName": "Temperature, water",
    "ResultMeasureValue": "17.9",
    "ResultMeasure/MeasureUnitCode": "deg C",
    "ResultStatusIdentifier": "Final",
    "ResultValueTypeName": "Actual",
}

# label -> overrides of DEFAULTS. Labels become ActivityIdentifiers.
ROWS = {
    "keep_temp": {},
    "qc_replicate": {"ActivityTypeCode": "Quality Control Sample-Field Replicate"},
    "sediment_media": {"ActivityMediaName": "Sediment"},
    "rejected": {"ResultStatusIdentifier": "Rejected"},
    "calculated": {"ResultValueTypeName": "Calculated"},
    "holding_time": {"MeasureQualifierCode": "J;H"},
    "nitrate_as_no3": {
        "ProviderName": "NWIS", "CharacteristicName": "Nitrate", "USGSPCode": "71851.0",
        "ResultSampleFractionText": "Dissolved", "ResultMeasureValue": "5.3",
        "ResultMeasure/MeasureUnitCode": "mg/l asNO3",
    },
    "keep_nitrate_as_n": {
        "ProviderName": "NWIS", "CharacteristicName": "Nitrate", "USGSPCode": "618.0",
        "ResultSampleFractionText": "Dissolved", "ResultMeasureValue": "1.2",
        "ResultMeasure/MeasureUnitCode": "mg/l as N",
    },
    "keep_nitrate_bare": {
        "CharacteristicName": "Nitrate", "ResultSampleFractionText": "Total",
        "ResultMeasureValue": "0.5", "ResultMeasure/MeasureUnitCode": "mg/L",
        "ActivityDepthHeightMeasure/MeasureValue": "3",
        "ActivityDepthHeightMeasure/MeasureUnitCode": "ft",
    },
    "keep_ammonia_nondetect": {
        "ProviderName": "NWIS", "CharacteristicName": "Ammonia and ammonium",
        "USGSPCode": "608", "ResultSampleFractionText": "Dissolved",
        "ResultMeasureValue": None, "ResultMeasure/MeasureUnitCode": None,
        "ResultDetectionConditionText": "Not Detected",
        "DetectionQuantitationLimitMeasure/MeasureValue": "0.02",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/l as N",
    },
    "keep_ph_equals": {
        "CharacteristicName": "pH", "ResultMeasureValue": "=6.59",
        "ResultMeasure/MeasureUnitCode": None,
    },
    "no_data": {
        "CharacteristicName": "pH", "ResultMeasureValue": "NO DATA",
        "ResultMeasure/MeasureUnitCode": None,
    },
    "keep_do_percent": {
        "CharacteristicName": "Dissolved oxygen (DO)", "ResultMeasureValue": "95",
        "ResultMeasure/MeasureUnitCode": "%",
    },
    "keep_oxygen_mgl": {
        "ProviderName": "NWIS", "CharacteristicName": "Oxygen", "USGSPCode": "300",
        "ResultSampleFractionText": "Dissolved", "ResultMeasureValue": "8.1",
        "ResultMeasure/MeasureUnitCode": "mg/l",
    },
    "keep_ortho_filtered": {
        "CharacteristicName": "Orthophosphate", "ResultSampleFractionText": "Filtered, field",
        "ResultMeasureValue": "0.03", "ResultMeasure/MeasureUnitCode": "mg/L",
    },
    "implausible_ph": {
        "CharacteristicName": "pH", "ResultMeasureValue": "45",
        "ResultMeasure/MeasureUnitCode": "std units",
    },
    "keep_sc_routine": {
        "CharacteristicName": "Specific conductance", "ActivityTypeCode": "Sample-Routine",
        "ActivityStartTime/Time": "11:40:00", "ResultMeasureValue": "357",
        "ResultMeasure/MeasureUnitCode": "uS/cm",
    },
    "sc_field_duplicate": {
        "CharacteristicName": "Specific conductance", "ActivityTypeCode": "Field Msr/Obs",
        "ActivityStartTime/Time": "11:40:00", "ResultMeasureValue": "357",
        "ResultMeasure/MeasureUnitCode": "uS/cm",
    },
    "fahrenheit": {"ResultMeasure/MeasureUnitCode": "deg F", "ResultMeasureValue": "64"},
    "unknown_station": {"MonitoringLocationIdentifier": "S9"},
    "out_of_scope_parameter": {"CharacteristicName": "Lead"},
    "keep_nitrate_speciated_n": {
        "CharacteristicName": "Nitrate", "ResultSampleFractionText": "Total",
        "ResultMeasureValue": "0.7", "ResultMeasure/MeasureUnitCode": "mg/L",
        "MethodSpeciationName": "as N",
    },
    "keep_nitrate_speciated_no3": {
        "CharacteristicName": "Nitrate", "ResultSampleFractionText": "Total",
        "ResultMeasureValue": "4.43", "ResultMeasure/MeasureUnitCode": "mg/L",
        "MethodSpeciationName": "as NO3",
    },
    "keep_phosphorus_lt": {
        "MonitoringLocationIdentifier": "S2", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "<0.5",
        "ResultMeasure/MeasureUnitCode": "mg/l as P",
    },
}

KEPT = sorted(label for label in ROWS if label.startswith("keep_"))


def _frame(labels):
    columns = sorted(cwr.REQUIRED_COLUMNS | {"MethodSpeciationName"})
    rows = []
    for label in labels:
        row = {col: None for col in columns}
        row.update(DEFAULTS)
        row.update(ROWS[label])
        row["ActivityIdentifier"] = label
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _build_project(root, manifest_status="success"):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        """
case_study: pilot_test
huc8s: ["01010001"]
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
"""
    )

    warehouse = root / "data" / "warehouse" / "pilot_test"
    warehouse.mkdir(parents=True)
    pd.DataFrame({"station_id": STATIONS}).to_parquet(warehouse / "wq_station.parquet")

    download = root / "data" / "raw" / "wqp" / "results" / "pilot_download"
    download.mkdir(parents=True)
    labels = list(ROWS)
    _frame(labels[:10]).to_csv(download / "results_batch_001.csv", index=False)
    # Batch 2 repeats keep_temp exactly (same measurement_id).
    _frame(labels[10:] + ["keep_temp"]).to_csv(download / "results_batch_002.csv", index=False)
    # Stale file from an earlier run: not in the manifest, so never read.
    pd.DataFrame({"junk": ["x"]}).to_csv(download / "results_batch_099.csv", index=False)

    pd.DataFrame({
        "batch": [1, 2, 3],
        "status": [manifest_status, "success", "zero_rows"],
        "file": ["results_batch_001.csv", "results_batch_002.csv", None],
    }).to_csv(download / "download_manifest.csv", index=False)
    return warehouse


def _run_main(root):
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", ["clean_wq_results.py", "--case-study", "pilot_test"]):
        cwr.main()


def test_main_applies_every_rule():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        warehouse = _build_project(root)
        _run_main(root)

        raw = pd.read_parquet(warehouse / "wq_measurement_raw.parquet")
        assert len(raw) == len(ROWS) + 1  # every downloaded row, repeat included

        table = pd.read_parquet(warehouse / "wq_measurement.parquet")
        assert list(table.columns) == list(cwr.SCHEMA)
        assert sorted(table["activity_id"]) == KEPT
        assert table["measurement_id"].is_unique

        row = table.set_index("activity_id")

        assert row.loc["keep_temp", "parameter_std"] == "water_temp"
        assert pd.isna(row.loc["keep_temp", "sample_fraction_std"])
        assert str(row.loc["keep_temp", "sample_time"]) == "10:34:00"

        assert row.loc["keep_nitrate_as_n", "usgs_pcode"] == "00618"
        assert not row.loc["keep_nitrate_as_n", "unit_basis_assumed"]
        assert row.loc["keep_nitrate_bare", "unit_basis_assumed"]
        assert row.loc["keep_nitrate_bare", "result_unit_std"] == "mg/L as N"
        assert abs(row.loc["keep_nitrate_bare", "depth_m"] - 0.9144) < 1e-9

        # MethodSpeciationName replaces the as-N assumption when present.
        assert not row.loc["keep_nitrate_speciated_n", "unit_basis_assumed"]
        assert row.loc["keep_nitrate_speciated_n", "result_value_std"] == 0.7
        no3 = row.loc["keep_nitrate_speciated_no3"]
        assert not no3["unit_basis_assumed"]
        assert abs(no3["result_value_std"] - 4.43 * 14.007 / 62.004) < 1e-12

        nondetect = row.loc["keep_ammonia_nondetect"]
        assert nondetect["is_censored"] and abs(nondetect["result_value_std"] - 0.01) < 1e-12

        lt = row.loc["keep_phosphorus_lt"]
        assert lt["is_censored"] and abs(lt["result_value_std"] - 0.25) < 1e-12

        assert row.loc["keep_ph_equals", "result_value_std"] == 6.59
        assert row.loc["keep_ph_equals", "result_unit_std"] == "std units"
        assert row.loc["keep_do_percent", "parameter_std"] == "do_saturation"
        assert row.loc["keep_oxygen_mgl", "parameter_std"] == "dissolved_oxygen"
        assert pd.isna(row.loc["keep_oxygen_mgl", "sample_fraction_std"])
        assert row.loc["keep_ortho_filtered", "sample_fraction_std"] == "Dissolved"
        assert row.loc["keep_sc_routine", "activity_type"] == "Sample-Routine"


def test_failed_batches_stop_the_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        warehouse = _build_project(root, manifest_status="failed")
        try:
            _run_main(root)
            raise AssertionError("Expected RuntimeError for failed batches.")
        except RuntimeError as exc:
            assert "failed batches" in str(exc)
        assert not (warehouse / "wq_measurement.parquet").exists()


def test_failed_validation_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        warehouse = _build_project(root)
        with mock.patch.object(cwr, "validate", return_value=["forced failure"]):
            try:
                _run_main(root)
                raise AssertionError("Expected ValueError from failed validation.")
            except ValueError as exc:
                assert "no warehouse files were written" in str(exc)
        assert not (warehouse / "wq_measurement.parquet").exists()
        assert not (warehouse / "wq_measurement_raw.parquet").exists()


def test_measurement_id_is_stable_and_null_safe():
    frame = _frame(["keep_temp", "keep_ammonia_nondetect"]).astype("string")
    first = cwr.measurement_ids(frame)
    second = cwr.measurement_ids(frame.copy())
    assert first.tolist() == second.tolist()
    assert first.nunique() == 2


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
