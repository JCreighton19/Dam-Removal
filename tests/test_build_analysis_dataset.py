"""Checks for etl/build_analysis_dataset.py.

Builds a tiny warehouse (core tables -> build_views views -> context
tables), runs the real `main()`, and checks the row selection and each
context join:

    D1 (removed 2015): station DOWN has 3 pre / 3 post samples (eligible),
        plus a removal-year and an outside-window sample (left out);
        station UP has post samples only (not eligible).
        Land cover for DOWN in 2013, 2014 and 2016: the baseline is 2014.
        Climate: temperature = month, precipitation = 10 x month.
    D2 (removed 1973): station OLD is eligible; NLCD only has 1985 and
        there is no climate, so land_cover_year is 1985 and climate is null.

Run directly: `python tests/test_build_analysis_dataset.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd

from etl import build_analysis_dataset as bad
from etl import build_views as bv
from etl.build_views import add_view
from etl.common.config import PROJECT_ROOT_ENV_VAR
from etl.load_land_cover import land_cover_codes

SAMPLES = {
    "DOWN": ["2005-06-01", "2012-01-15", "2013-06-01", "2013-07-01", "2015-06-01",
             "2016-06-01", "2017-06-01", "2018-03-01"],
    "UP": ["2016-06-01", "2017-06-01", "2018-06-01"],
    "OLD": ["1970-05-01", "1970-06-01", "1971-05-01", "1975-05-01", "1975-06-01", "1976-05-01"],
}


def _core_tables(directory):
    dams = pd.DataFrame({
        "AR_ID": ["D1", "D2"], "Dam_Name": ["One", "Two"],
        "Year_Removed": pd.array([2015, 1973], dtype="Int16"), "snap_fraction": [0.5, 0.5],
        "River": ["R1", "R2"], "State": ["ME", "NY"], "Dam_Height_ft": [10.0, None],
        "Year_built": pd.array([1900, None], dtype="Int16"), "Original_Use": ["Mill", None],
    })
    stations = pd.DataFrame({"station_id": ["DOWN", "UP", "OLD"], "snap_fraction": [0.5] * 3,
                             "qc_far_snap": [False, False, True],
                             "qc_near_unsuitable_flowline": False, "qc_non_riverine_name": False})
    links = pd.DataFrame({
        "AR_ID": ["D1", "D1", "D2"], "station_id": ["DOWN", "UP", "OLD"],
        "relationship": ["downstream", "upstream", "downstream"],
        "network_distance_km": [1.0, 2.0, 3.0],
    })
    rows = [(s, d) for s, dates in SAMPLES.items() for d in dates]
    measurements = pd.DataFrame(rows, columns=["station_id", "sample_date"])
    measurements["sample_date"] = pd.to_datetime(measurements["sample_date"]).dt.date
    measurements["measurement_id"] = range(len(measurements))
    measurements["parameter_std"] = "water_temp"
    measurements["sample_fraction_std"] = pd.array([None] * len(measurements), dtype="string")
    measurements["result_value_std"] = 10.0
    measurements["result_unit_std"] = "deg C"
    measurements["is_censored"] = False
    measurements["unit_basis_assumed"] = False
    measurements["provider"] = "NWIS"
    tables = {
        "dam_removal": dams, "wq_station": stations, "dam_station_link": links,
        "huc8_watershed": pd.DataFrame({"huc8": ["01010101"]}),
        "streams": pd.DataFrame({"nhdplus_id": [1]}),
        "wq_measurement": measurements, "wq_measurement_raw": pd.DataFrame({"x": [1]}),
    }
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)


def _context_tables(directory):
    codes = land_cover_codes()
    land = []
    for station, year, mix in [("DOWN", 2013, {41: 100}), ("DOWN", 2014, {41: 60, 21: 40}),
                               ("DOWN", 2016, {21: 100}), ("OLD", 1985, {82: 100})]:
        for code in codes["class_code"]:
            land.append({"station_id": station, "year": year, "class_code": code,
                         "percent_area": float(mix.get(code, 0))})
    climate = pd.DataFrame(
        [{"station_id": "DOWN", "year": y, "month": m, "temperature_mean_c": float(m),
          "precipitation_mm": 10.0 * m} for y in range(2010, 2021) for m in range(1, 13)])
    tables = {
        "station_watershed": pd.DataFrame({"station_id": ["DOWN", "UP", "OLD"],
                                           "area_km2": [50.0, 40.0, 30.0],
                                           "truncated": [False, False, True],
                                           "drainage_area_flag": ["ok", "ok", "truncated"]}),
        "land_cover": pd.DataFrame(land),
        "land_cover_codes": codes,
        "climate": climate,
    }
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
        add_view(directory / "warehouse.duckdb", name, directory / f"{name}.parquet")


def _build(root):
    (root / "config").mkdir()
    (root / "config" / "pilot_test.yaml").write_text(
        'case_study: pilot_test\nhuc8s: ["01010101"]\n'
        "dam_source: data/raw/dams/dams.csv\nwbd_source: data/raw/wbd/wbd.gpkg\n"
        "wq_window:\n  years_before: 5\n  years_after: 5\n"
    )
    out = root / "data" / "warehouse" / "pilot_test"
    out.mkdir(parents=True)
    _core_tables(out)
    bv.build_database(out / "warehouse.duckdb", out, 5, 5, 3, 2)
    _context_tables(out)
    return out


def test_rows_and_context_joins():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _build(root)
        argv = ["build_analysis_dataset.py", "--case-study", "pilot_test"]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            bad.main()

        table = pd.read_parquet(out / "analysis_measurement.parquet")
        # 3 pre + 3 post for DOWN and OLD; UP, the removal year and 2005 are out.
        assert len(table) == 12
        assert set(table["station_id"]) == {"DOWN", "OLD"}
        assert set(table["period"]) == {"pre", "post"}

        down = table[table["station_id"] == "DOWN"].set_index("sample_date")
        jan = down.loc[pd.Timestamp("2012-01-15")]
        assert jan["season"] == "winter" and jan["years_from_removal"] == -3
        assert jan["air_temp_mean_c"] == 1 and jan["precip_mm"] == 10
        assert jan["precip_prev_month_mm"] == 120  # December 2011
        assert (down["dam_age_at_removal"] == 115).all()

        # Baseline land cover: last year before removal (2014), for every row.
        assert (down["land_cover_year"] == 2014).all()
        assert (down["pct_forest"] == 60).all() and (down["pct_developed"] == 40).all()

        # Pre-1985 removal: nearest NLCD year, no climate.
        old = table[table["station_id"] == "OLD"]
        assert (old["land_cover_year"] == 1985).all() and (old["pct_agriculture"] == 100).all()
        assert old["air_temp_mean_c"].isna().all()
        assert old["watershed_truncated"].all() and old["year_built"].isna().all()

        # Exposure: each station is below one removal only.
        assert (down["exposure_status"] == "assigned_sole").all() and (down["assigned_AR_ID"] == "D1").all()
        assert (old["assigned_AR_ID"] == "D2").all() and (old["n_other_removals_in_window"] == 0).all()

        # Flags carried, not applied; a truncated area never enters as a value.
        assert old["qc_far_snap"].all() and not down["qc_far_snap"].any()
        assert table["qc_saline"].isna().all()  # no conductance measured: unknown
        assert old["upstream_area_km2"].isna().all() and old["missing_upstream_area"].all()
        assert (down["upstream_area_km2"] == 50.0).all() and not down["missing_upstream_area"].any()
        assert old["missing_climate"].all() and not down["missing_climate"].any()
        assert not table["missing_land_cover"].any()

        with duckdb.connect(str(out / "warehouse.duckdb"), read_only=True) as con:
            assert con.sql("SELECT count(*) FROM analysis_measurement").fetchone()[0] == 12


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
