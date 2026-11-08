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
    "future_date": {"ActivityStartDate": "2999-09-13"},
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
        "ResultAnalyticalMethod/MethodIdentifier": " I-2545-90 ",
        "ResultAnalyticalMethod/MethodIdentifierContext": "USGS",
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
    # mS/cm reported as uS/cm (2026-11-08): below the 2 uS/cm floor, dropped.
    "sc_mscm_reported_as_uscm": {
        "CharacteristicName": "Specific conductance", "ResultMeasureValue": "0.503",
        "ResultMeasure/MeasureUnitCode": "uS/cm",
    },
    "sc_zero": {
        "CharacteristicName": "Specific conductance", "ResultMeasureValue": "0",
        "ResultMeasure/MeasureUnitCode": "uS/cm",
    },
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
    "keep_phosphorus_unfiltered": {
        "CharacteristicName": "Phosphorus", "ResultSampleFractionText": "Unfiltered",
        "ResultMeasureValue": "0.1", "ResultMeasure/MeasureUnitCode": "mg/l as P",
    },
    "keep_nitrate_filtered_lab": {
        "CharacteristicName": "Nitrate", "ResultSampleFractionText": "Filtered, lab",
        "ResultMeasureValue": "0.9", "ResultMeasure/MeasureUnitCode": "mg/l as N",
    },
    "keep_nitrate_blank_fraction": {
        "CharacteristicName": "Nitrate", "ResultSampleFractionText": " ",
        "ResultMeasureValue": "0.8", "ResultMeasure/MeasureUnitCode": "mg/l as N",
    },
    "keep_tds_dissolved": {
        "CharacteristicName": "Total dissolved solids", "ResultSampleFractionText": "Dissolved",
        "ResultMeasureValue": "250", "ResultMeasure/MeasureUnitCode": "mg/L",
    },
    # Very dilute but real water (snowmelt): above the floor, kept.
    "keep_sc_dilute": {
        "MonitoringLocationIdentifier": "S2", "CharacteristicName": "Specific conductance",
        "ResultMeasureValue": "3.5", "ResultMeasure/MeasureUnitCode": "uS/cm",
    },
    "keep_phosphorus_lt": {
        "MonitoringLocationIdentifier": "S2", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "<0.5",
        "ResultMeasure/MeasureUnitCode": "mg/l as P",
    },
    # Suspended sediment and discharge (added 2026-11-02).
    "keep_ssc": {
        "ProviderName": "NWIS", "CharacteristicName": "Suspended Sediment Concentration (SSC)",
        "USGSPCode": "80154", "ResultSampleFractionText": "Suspended",
        "ResultMeasureValue": "125", "ResultMeasure/MeasureUnitCode": None,
    },
    "keep_tss": {
        "CharacteristicName": "Total suspended solids", "ResultSampleFractionText": "Non-filterable",
        "ResultMeasureValue": "40", "ResultMeasure/MeasureUnitCode": "mg/L",
    },
    "keep_discharge_cfs": {
        "ProviderName": "NWIS", "CharacteristicName": "Stream flow, instantaneous",
        "USGSPCode": "00061", "ResultMeasureValue": "350", "ResultMeasure/MeasureUnitCode": "ft3/s",
    },
    "keep_flow_m3s": {
        "CharacteristicName": "Flow", "ResultMeasureValue": "2",
        "ResultMeasure/MeasureUnitCode": "m3/sec",
    },
    "daily_mean_flow": {
        "CharacteristicName": "Stream flow, mean. daily", "ResultMeasureValue": "300",
        "ResultMeasure/MeasureUnitCode": "ft3/s",
    },
    # Non-detects and zeros (2026-11-08). Each on its own date so deduplication
    # can't merge them. "keep_" rows survive; the others are dropped.
    "keep_nd_u_with_limit": {
        "ActivityStartDate": "2016-01-04", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0",
        "ResultMeasure/MeasureUnitCode": "mg/l as P", "MeasureQualifierCode": "U",
        "DetectionQuantitationLimitMeasure/MeasureValue": "0.01",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/l as P",
    },
    "keep_nd_dl_messy_case": {
        "ActivityStartDate": "2016-01-05", "CharacteristicName": "Total suspended solids",
        "ResultMeasureValue": "0", "ResultMeasure/MeasureUnitCode": "mg/L",
        "MeasureQualifierCode": " dl ;J", "DetectionQuantitationLimitMeasure/MeasureValue": "4",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/L",
    },
    "keep_nd_bdl": {
        "ActivityStartDate": "2016-01-06", "CharacteristicName": "Orthophosphate",
        "ResultSampleFractionText": "Dissolved", "ResultMeasureValue": "0.001",
        "ResultMeasure/MeasureUnitCode": "mg/l as P", "MeasureQualifierCode": "BDL",
        "DetectionQuantitationLimitMeasure/MeasureValue": "0.004",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/l as P",
    },
    "keep_nd_nd_ug": {
        "ActivityStartDate": "2016-01-07", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0",
        "ResultMeasure/MeasureUnitCode": "mg/l as P", "MeasureQualifierCode": "ND",
        "DetectionQuantitationLimitMeasure/MeasureValue": "20",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "ug/l as P",
    },
    "keep_nd_condition_new": {
        "ActivityStartDate": "2016-01-08", "CharacteristicName": "Total suspended solids",
        "ResultMeasureValue": "0", "ResultMeasure/MeasureUnitCode": "mg/L",
        "ResultDetectionConditionText": " not detected at detection limit ",
        "DetectionQuantitationLimitMeasure/MeasureValue": "2",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/L",
    },
    "nd_brl_no_limit": {
        "ActivityStartDate": "2016-01-09", "CharacteristicName": "Nitrate",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0.0",
        "ResultMeasure/MeasureUnitCode": "mg/l as N", "MeasureQualifierCode": "BRL",
    },
    "nd_u_zero_limit": {
        "ActivityStartDate": "2016-01-10", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0",
        "ResultMeasure/MeasureUnitCode": "mg/l as P", "MeasureQualifierCode": "U",
        "DetectionQuantitationLimitMeasure/MeasureValue": "0",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/l as P",
    },
    "nd_u_limit_unit_unconvertible": {
        "ActivityStartDate": "2016-01-11", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0",
        "ResultMeasure/MeasureUnitCode": "mg/l as P", "MeasureQualifierCode": "U",
        "DetectionQuantitationLimitMeasure/MeasureValue": "5",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "lbs/day",
    },
    "keep_zero_turbidity_with_limit": {
        "ActivityStartDate": "2016-01-12", "CharacteristicName": "Turbidity",
        "ResultMeasureValue": "0", "ResultMeasure/MeasureUnitCode": "NTU",
        "DetectionQuantitationLimitMeasure/MeasureValue": "0.1",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "NTU",
    },
    "zero_ortho_no_limit": {
        "ActivityStartDate": "2016-01-13", "CharacteristicName": "Orthophosphate",
        "ResultSampleFractionText": "Dissolved", "ResultMeasureValue": "0.0",
        "ResultMeasure/MeasureUnitCode": "mg/l as P",
    },
    "zero_tss_invalid_limit": {
        "ActivityStartDate": "2016-01-14", "CharacteristicName": "Total suspended solids",
        "ResultMeasureValue": "0", "ResultMeasure/MeasureUnitCode": "mg/L",
        "DetectionQuantitationLimitMeasure/MeasureValue": "-1",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/L",
    },
    "keep_discharge_zero": {
        "ActivityStartDate": "2016-01-15", "CharacteristicName": "Stream flow, instantaneous",
        "ResultMeasureValue": "0", "ResultMeasure/MeasureUnitCode": "ft3/s",
    },
    "keep_below_limit_positive": {
        "ActivityStartDate": "2016-01-16", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0.004",
        "ResultMeasure/MeasureUnitCode": "mg/l as P",
        "DetectionQuantitationLimitMeasure/MeasureValue": "0.01",
        "DetectionQuantitationLimitMeasure/MeasureUnitCode": "mg/l as P",
    },
    "keep_qualifier_not_a_substring": {
        "ActivityStartDate": "2016-01-17", "CharacteristicName": "Phosphorus",
        "ResultSampleFractionText": "Total", "ResultMeasureValue": "0.05",
        "ResultMeasure/MeasureUnitCode": "mg/l as P", "MeasureQualifierCode": "UND;DLX",
    },
    "tds_field_meter_gl": {
        "ActivityStartDate": "2016-01-18", "CharacteristicName": "Total dissolved solids",
        "ResultSampleFractionText": "Dissolved", "ResultMeasureValue": "0.55",
        "ResultMeasure/MeasureUnitCode": "mg/L",
    },
    "negative_flow": {
        "CharacteristicName": "Stream flow, instantaneous", "ResultMeasureValue": "-5",
        "ResultMeasure/MeasureUnitCode": "ft3/s",
    },
}

KEPT = sorted(label for label in ROWS if label.startswith("keep_"))


def _frame(labels):
    columns = sorted(cwr.REQUIRED_COLUMNS | {"MethodSpeciationName", cwr.METHOD_CONTEXT})
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

    download = root / "data" / "raw" / "wqp" / "results" / "pilot_test" / "pilot_download"
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
        assert row.loc["keep_nitrate_as_n", "method_id"] == "I-2545-90"
        assert row.loc["keep_nitrate_as_n", "method_context"] == "USGS"
        assert pd.isna(row.loc["keep_temp", "method_id"])
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
        assert row.loc["keep_phosphorus_unfiltered", "sample_fraction_std"] == "Total"
        assert row.loc["keep_nitrate_filtered_lab", "sample_fraction_std"] == "Dissolved"
        assert pd.isna(row.loc["keep_nitrate_blank_fraction", "sample_fraction_std"])
        assert pd.isna(row.loc["keep_tds_dissolved", "sample_fraction_std"])
        assert row.loc["keep_sc_routine", "activity_type"] == "Sample-Routine"
        assert row.loc["keep_sc_dilute", "result_value_std"] == 3.5   # sc_mscm_* and sc_zero dropped

        # SSC / TSS kept apart, no fraction; SSC unit from its pcode.
        assert row.loc["keep_ssc", "parameter_std"] == "ssc"
        assert row.loc["keep_ssc", "result_value_std"] == 125
        assert row.loc["keep_ssc", "result_unit_std"] == "mg/L"
        assert pd.isna(row.loc["keep_ssc", "sample_fraction_std"])
        assert row.loc["keep_tss", "parameter_std"] == "tss"
        assert pd.isna(row.loc["keep_tss", "sample_fraction_std"])
        # Discharge in ft3/s; daily means and negative flows are not kept.
        assert row.loc["keep_discharge_cfs", "parameter_std"] == "discharge"
        assert row.loc["keep_discharge_cfs", "result_value_std"] == 350
        assert abs(row.loc["keep_flow_m3s", "result_value_std"] - 70.6294) < 1e-9
        assert row.loc["keep_flow_m3s", "result_unit_std"] == "ft3/s"


def test_nondetects_and_zeros():
    """2026-11-08 rules: the added non-detect codes and condition, valid / missing /
    invalid / unconvertible detection limits, unflagged zeros, discharge zeros,
    positive below-limit values, exact (not substring) code matching, TDS floor."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        warehouse = _build_project(root)
        _run_main(root)
        row = pd.read_parquet(warehouse / "wq_measurement.parquet").set_index("activity_id")

        def check(label, value, censored):
            assert abs(row.loc[label, "result_value_std"] - value) < 1e-12, (label, row.loc[label, "result_value_std"])
            assert bool(row.loc[label, "is_censored"]) is censored, label

        check("keep_nd_u_with_limit", 0.005, True)          # U, half of 0.01
        check("keep_nd_dl_messy_case", 2.0, True)            # " dl " normalized, half of 4
        check("keep_nd_bdl", 0.002, True)                    # positive but flagged -> half the limit
        check("keep_nd_nd_ug", 0.01, True)                   # limit in ug/L converted: 20 ug/L / 2 = 0.01 mg/L
        check("keep_nd_condition_new", 1.0, True)            # condition matched after trim / case-fold
        check("keep_zero_turbidity_with_limit", 0.05, True)  # unflagged zero with a valid limit
        check("keep_discharge_zero", 0.0, False)             # zero flow is real
        check("keep_below_limit_positive", 0.004, False)     # unflagged positive below the limit: untouched
        check("keep_qualifier_not_a_substring", 0.05, False) # "UND", "DLX" are not U / ND / DL
        for dropped in ("nd_brl_no_limit", "nd_u_zero_limit", "nd_u_limit_unit_unconvertible",
                        "zero_ortho_no_limit", "zero_tss_invalid_limit", "tds_field_meter_gl"):
            assert dropped not in row.index, dropped


def test_zero_rule_covers_the_log_scale_outcomes():
    """ZERO_AS_NONDETECT = the model's log-scale measures minus discharge."""
    from analysis import mixed_model as mm
    from etl.common import wq_rules
    assert set(wq_rules.ZERO_AS_NONDETECT) == set(mm.SETTINGS["log_parameters"]) - {"discharge"}


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
