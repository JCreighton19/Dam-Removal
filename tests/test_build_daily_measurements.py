"""Checks for etl/build_daily_measurements.py.

Run directly: `python tests/test_build_daily_measurements.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from etl import build_daily_measurements as bdm
from etl.common.config import PROJECT_ROOT_ENV_VAR

# (measurement_id, station, parameter, date, depth, activity, method, value, censored, raw unit)
ROWS = [
    # Three repeats of one reading: one group, median 11.
    (1, "S1", "water_temp", "2015-06-01", 0.5, "Field Msr/Obs", None, 10.0, False, "deg C"),
    (2, "S1", "water_temp", "2015-06-01", 0.5, "Field Msr/Obs", None, 11.0, False, "deg C"),
    (3, "S1", "water_temp", "2015-06-01", 0.5, "Field Msr/Obs", None, 30.0, False, "deg C"),
    # A depth profile: separate rows per depth.
    (4, "S1", "water_temp", "2015-06-01", 5.0, "Field Msr/Obs", None, 8.0, False, "deg C"),
    # Same day, another sample type: separate.
    (5, "S1", "water_temp", "2015-06-01", 0.5, "Sample-Routine", None, 12.0, False, "deg C"),
    # Phosphorus: one detect and one non-detect by the same method -> the detect.
    (6, "S1", "phosphorus", "2015-06-01", None, "Sample-Routine", "M1", 0.08, False, "mg/l as P"),
    (7, "S1", "phosphorus", "2015-06-01", None, "Sample-Routine", "M1", 0.005, True, "mg/l as P"),
    # Another method the same day: separate.
    (8, "S1", "phosphorus", "2015-06-01", None, "Sample-Routine", "M2", 0.07, False, "mg/l as P"),
    # Two non-detects: censored median.
    (9, "S2", "phosphorus", "2015-06-01", None, "Sample-Routine", "M1", 0.01, True, "mg/l as P"),
    (10, "S2", "phosphorus", "2015-06-01", None, "Sample-Routine", "M1", 0.02, True, "mg/l as P"),
    # Turbidity in NTU and FNU the same day: separate.
    (11, "S2", "turbidity", "2015-06-01", None, "Field Msr/Obs", None, 5.0, False, "NTU"),
    (12, "S2", "turbidity", "2015-06-01", None, "Field Msr/Obs", None, 6.0, False, "FNU"),
]
STD_UNITS = {"water_temp": "deg C", "phosphorus": "mg/L as P", "turbidity": "NTU-equivalent"}


def _write(warehouse):
    frame = pd.DataFrame(ROWS, columns=["measurement_id", "station_id", "parameter_std",
                                        "sample_date", "depth_m", "activity_type", "method_id",
                                        "result_value_std", "is_censored", "result_unit_raw"])
    frame["sample_date"] = pd.to_datetime(frame["sample_date"]).dt.date
    frame["sample_fraction_std"] = pd.array(
        ["Total" if p == "phosphorus" else None for p in frame["parameter_std"]], dtype="string")
    frame["method_context"] = pd.array([None] * len(frame), dtype="string")
    frame["unit_basis_assumed"] = False
    frame["result_unit_std"] = frame["parameter_std"].map(STD_UNITS)
    frame["provider"] = "STORET"
    frame.to_parquet(warehouse / "wq_measurement.parquet", index=False)


def _run(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        'case_study: pilot_test\nhuc8s: ["01010001"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n")
    warehouse = root / "data" / "warehouse" / "pilot_test"
    warehouse.mkdir(parents=True)
    _write(warehouse)
    argv = ["build_daily_measurements.py", "--case-study", "pilot_test"]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        bdm.main()
    return pd.read_parquet(warehouse / "wq_measurement_daily.parquet").set_index("daily_id")


def test_groups_and_values():
    with tempfile.TemporaryDirectory() as tmp:
        daily = _run(Path(tmp))
    assert list(daily.reset_index().columns) == list(bdm.SCHEMA)
    assert len(daily) == 8 and daily["n_raw"].sum() == len(ROWS)

    repeats = daily.loc[1]
    assert repeats["result_value_std"] == 11.0 and repeats["n_raw"] == 3
    assert (repeats["value_min"], repeats["value_max"]) == (10.0, 30.0)
    assert daily.loc[4, "depth_m"] == 5.0 and daily.loc[5, "activity_type"] == "Sample-Routine"

    detect = daily.loc[6]
    assert detect["result_value_std"] == 0.08 and not detect["is_censored"]
    assert detect["n_censored_dropped"] == 1 and detect["n_values"] == 1
    assert daily.loc[8, "method_id"] == "M2"

    nondetect = daily.loc[9]
    assert nondetect["is_censored"] and abs(nondetect["result_value_std"] - 0.015) < 1e-12

    assert set(daily.loc[[11, 12], "turbidity_unit"]) == {"NTU", "FNU"}


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
