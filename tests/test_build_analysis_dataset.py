"""Checks for etl/build_analysis_dataset.py.

Builds a tiny warehouse (core tables -> build_views views -> context
tables), runs the real `main()`, and checks the row selection and each
context join:

    D1 (removed 2015): station DOWN has 3 pre / 3 post samples (eligible),
        one post day read twice (two rows, one daily_id),
        plus a removal-year and an outside-window sample (left out);
        station UP has post samples only (not eligible).
        Land cover for DOWN in 2013, 2014 and 2016: the baseline is 2014.
        Climate: temperature = month, precipitation = 10 x month.
    D2 (removed 1973): station OLD is eligible; NLCD only has 1985 and
        there is no climate, so land_cover_year is 1985 and climate is null.
    Dam size: D1 (10 ft, slope 0.01, 20 km2) has backwater 0.3048 km and a
        mapped reservoir (0.5 km2, edited 2010, before its removal); D2 has
        no height or slope and no polygon, so every size column is null.

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
from etl import build_dam_reservoir as bdr
from etl import build_daily_measurements as bdm
from etl import build_views as bv
from etl.build_views import add_view
from etl.common.config import PROJECT_ROOT_ENV_VAR
from etl.load_land_cover import land_cover_codes

SAMPLES = {
    "DOWN": ["2005-06-01", "2012-01-15", "2013-06-01", "2013-07-01", "2015-06-01",
             "2016-06-01", "2016-06-01", "2017-06-01", "2018-03-01"],
    "UP": ["2016-06-01", "2017-06-01", "2018-06-01"],
    "OLD": ["1970-05-01", "1970-06-01", "1971-05-01", "1975-05-01", "1975-06-01", "1976-05-01"],
}


def _core_tables(directory):
    dams = pd.DataFrame({
        "AR_ID": ["D1", "D2"], "Dam_Name": ["One", "Two"],
        "Year_Removed": pd.array([2015, 1973], dtype="Int16"), "snap_fraction": [0.5, 0.5],
        "River": ["R1", "R2"], "State": ["ME", "NY"], "Dam_Height_ft": [10.0, None],
        "Year_built": pd.array([1900, None], dtype="Int16"), "Original_Use": ["Mill", None],
        "reach_totdasqkm": [20.0, 10.0], "reach_slope": [0.01, None],
        "reach_feature_code": pd.array([46006, 46006], dtype="Int32"),
    })
    stations = pd.DataFrame({"station_id": ["DOWN", "UP", "OLD"], "snap_fraction": [0.5] * 3,
                             "qc_far_snap": [False, False, True],
                             "qc_near_unsuitable_flowline": False, "qc_non_riverine_name": False,
                             "reach_totdasqkm": [50.0, 15.0, 30.0],
                             "reach_feature_code": pd.array([46006, 46007, None],
                                                            dtype="Int32")})
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
    measurements["depth_m"] = None
    measurements["activity_type"] = "Field Msr/Obs"
    for column in ["method_id", "method_context", "result_unit_raw", "detection_limit_unit"]:
        measurements[column] = pd.array([None] * len(measurements), dtype="string")
    measurements["detection_limit"] = None
    # Two discharge readings at DOWN on 2016-06-01 (median 300 ft3/s). One
    # sample per side is too few for a discharge pair, so they add no rows.
    flow = measurements.iloc[:2].copy()
    flow["sample_date"] = pd.to_datetime(["2016-06-01"] * 2).date
    flow["station_id"] = "DOWN"
    flow["measurement_id"] = [1000, 1001]
    flow["parameter_std"] = "discharge"
    flow["result_value_std"] = [200.0, 400.0]
    flow["result_unit_std"] = "ft3/s"
    measurements = pd.concat([measurements, flow], ignore_index=True)
    tables = {
        "dam_removal": dams, "wq_station": stations, "dam_station_link": links,
        "huc8_watershed": pd.DataFrame({"huc8": ["01010101"]}),
        "streams": pd.DataFrame({"nhdplus_id": [1]}),
        "wq_measurement": measurements, "wq_measurement_raw": pd.DataFrame({"x": [1]}),
    }
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    with duckdb.connect() as con:
        daily = bdm.build_table(con, str(directory / "wq_measurement.parquet"))
    daily.to_parquet(directory / "wq_measurement_daily.parquet", index=False)


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
    _reservoir_table(directory)


def _reservoir_table(directory):
    dams = pd.read_parquet(directory / "dam_removal.parquet", columns=["AR_ID", "Year_Removed"])
    reaches = pd.DataFrame({
        "AR_ID": ["D1", "D2"], "huc8": ["01010101"] * 2, "match_status": ["accepted"] * 2,
        "reservoir_status": ["dam_at_outlet", "no_reservoir_polygon"],
        "reservoir_id": ["{r1}", None], "reservoir_ftype": [436, None],
        "reservoir_fdate": pd.to_datetime(["2010-01-01", None]),
    })
    areas = pd.DataFrame({"reservoir_id": ["R1"], "polygon_area_km2": [0.5],
                          "polygon_area_source": ["AreaSqKm"]})
    table = bdr.build_table(dams, reaches, areas)
    assert bdr.validate(table, dams) == []
    table.to_parquet(directory / "dam_reservoir.parquet", index=False)
    add_view(directory / "warehouse.duckdb", "dam_reservoir", directory / "dam_reservoir.parquet")


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
        # 3 pre + 3 post for DOWN (one post day twice) and OLD; UP, the
        # removal year and 2005 are out.
        assert len(table) == 13
        assert set(table["station_id"]) == {"DOWN", "OLD"}
        assert set(table["period"]) == {"pre", "post"}

        # Same-day repeats share a daily_id; every other row has its own.
        down_rows = table[table["station_id"] == "DOWN"]
        repeat = down_rows[down_rows["sample_date"] == pd.Timestamp("2016-06-01")]
        assert len(repeat) == 2 and repeat["daily_id"].nunique() == 1
        assert repeat["daily_id"].iloc[0] == repeat["measurement_id"].min()
        assert table["daily_id"].nunique() == 12

        down = down_rows.drop_duplicates("sample_date").set_index("sample_date")
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

        # Link features: station / dam area (DOWN 50/20, OLD 30/10).
        assert (down["drainage_area_ratio"] == 2.5).all() and (old["drainage_area_ratio"] == 3).all()
        assert table["upstream_in_former_reservoir"].isna().all()  # no upstream rows
        # Same-day discharge: only DOWN's 2016-06-01 rows (two repeats) have it.
        flow_rows = table[table["has_same_day_discharge"]]
        assert len(flow_rows) == 2 and (flow_rows["sample_date"] == pd.Timestamp("2016-06-01")).all()
        assert (flow_rows["discharge_same_day_cfs"] == 300.0).all()
        assert table.loc[~table["has_same_day_discharge"], "discharge_same_day_cfs"].isna().all()
        # Flow permanence: DOWN and D1 perennial; OLD's reach has no FCode.
        assert (down["station_flow_permanence"] == "perennial").all()
        assert (down["dam_flow_permanence"] == "perennial").all()
        assert (down["qc_non_perennial"] == False).all()  # noqa: E712
        assert old["station_flow_permanence"].isna().all() and old["qc_non_perennial"].isna().all()

        # Flags carried, not applied; a truncated area never enters as a value.
        assert old["qc_far_snap"].all() and not down["qc_far_snap"].any()
        assert table["qc_saline"].isna().all()  # no conductance measured: unknown
        assert old["upstream_area_km2"].isna().all() and old["missing_upstream_area"].all()
        assert (down["upstream_area_km2"] == 50.0).all() and not down["missing_upstream_area"].any()
        assert old["missing_climate"].all() and not down["missing_climate"].any()
        assert not table["missing_land_cover"].any()

        # Dam size (choice 15): constructed from link_features, measured from
        # dam_reservoir, kept apart; nothing imputed for D2.
        assert (abs(down["backwater_km"] - 0.3048) < 1e-9).all()
        proxy = (10 * 0.3048) ** 2 / 0.01 * 20 ** 0.5
        assert (abs(down["reservoir_volume_proxy"] - proxy) < 1e-6).all()
        assert (down["reservoir_status"] == "dam_at_outlet").all()
        assert (down["reservoir_polygon_area_km2"] == 0.5).all()
        assert (down["reservoir_polygon_timing"] == "edited_by_removal_year").all()
        assert (down["dam_drainage_area_km2"] == 20).all()
        assert (old["reservoir_status"] == "no_reservoir_polygon").all()
        for column in ["backwater_km", "reservoir_volume_proxy", "reservoir_polygon_area_km2",
                       "reservoir_polygon_timing"]:
            assert old[column].isna().all(), column
        assert (old["dam_drainage_area_km2"] == 10).all()

        with duckdb.connect(str(out / "warehouse.duckdb"), read_only=True) as con:
            assert con.sql("SELECT count(*) FROM analysis_measurement").fetchone()[0] == 13


def test_stale_daily_table_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(Path(tmp))
        table, expected, written = bad.build_table(out / "warehouse.duckdb")
        assert bad.validate(table, expected, written) == []
        # wq_measurement_daily from an older clean_wq_results run.
        errors = bad.validate(table, expected, written - {table["daily_id"].iloc[0]})
        assert any("re-run build_daily_measurements" in e for e in errors)


def test_stale_context_table_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(Path(tmp))
        assert bad.stale_inputs(out / "warehouse.duckdb") == []
        # Climate from an earlier run, for a station wq_station no longer has.
        climate = pd.read_parquet(out / "climate.parquet")
        climate.loc[0, "station_id"] = "GONE"
        climate.to_parquet(out / "climate.parquet", index=False)
        stale = bad.stale_inputs(out / "warehouse.duckdb")
        assert len(stale) == 1 and "re-run load_climate.py" in stale[0]


def test_stale_reservoir_table_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(Path(tmp))
        # dam_reservoir from before a dam was added to dam_removal.
        reservoir = pd.read_parquet(out / "dam_reservoir.parquet")
        reservoir[reservoir["AR_ID"] != "D2"].to_parquet(out / "dam_reservoir.parquet",
                                                         index=False)
        stale = bad.stale_inputs(out / "warehouse.duckdb")
        assert len(stale) == 1 and "re-run build_dam_reservoir.py" in stale[0]


def test_reports_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = _build(Path(tmp))
        table, _, _ = bad.build_table(out / "warehouse.duckdb")
    size = bad.size_report(table)
    assert "reservoir_volume_proxy" in size and "mapped by the removal year" in size
    bands = bad.exposure_by_band_report(table)
    assert "00-05" in bands and "assigned_sole" in bands
    labels = bad.distance_band(pd.Series([0.0, 5.0, 5.01, 30.0, 31.0, None]))
    assert labels.tolist() == ["00-05", "00-05", "05-10", "20-30", "30+", "unknown"]


# --------------------------------------------------
# Eligibility sensitivity variants (docs/eligibility_sensitivity_prespec.md)
# --------------------------------------------------

def _add_thin_stations(out):
    """Two more stations below D1 with a single pre-removal year: THIN3 has
    samples in 3 months of 2014, THIN1 in one month only. Neither meets the
    primary 3 samples / 2 years; THIN3 has context, THIN1 has none."""
    import hashlib  # noqa: F401
    stations = pd.read_parquet(out / "wq_station.parquet")
    extra = stations.iloc[:2].copy()
    extra["station_id"] = ["THIN3", "THIN1"]
    pd.concat([stations, extra], ignore_index=True).to_parquet(out / "wq_station.parquet", index=False)
    links = pd.read_parquet(out / "dam_station_link.parquet")
    pd.concat([links, pd.DataFrame({"AR_ID": ["D1", "D1"], "station_id": ["THIN3", "THIN1"],
                                    "relationship": "downstream", "network_distance_km": [4.0, 5.0]})],
              ignore_index=True).to_parquet(out / "dam_station_link.parquet", index=False)
    measurements = pd.read_parquet(out / "wq_measurement.parquet")
    dates = {"THIN3": ["2014-05-01", "2014-06-01", "2014-07-01"],
             "THIN1": ["2014-06-01", "2014-06-10", "2014-06-20"]}
    rows = []
    for station, pre in dates.items():
        for d in pre + ["2016-06-01", "2017-06-01", "2018-06-01"]:
            rows.append((station, d))
    template = measurements[measurements["parameter_std"].eq("water_temp")].iloc[0]
    more = pd.DataFrame([template] * len(rows)).reset_index(drop=True)
    more["station_id"] = [r[0] for r in rows]
    more["sample_date"] = pd.to_datetime([r[1] for r in rows]).date
    more["measurement_id"] = range(5000, 5000 + len(rows))
    pd.concat([measurements, more], ignore_index=True).to_parquet(out / "wq_measurement.parquet",
                                                                  index=False)
    with duckdb.connect() as con:
        daily = bdm.build_table(con, str(out / "wq_measurement.parquet"))
    daily.to_parquet(out / "wq_measurement_daily.parquet", index=False)
    watershed = pd.read_parquet(out / "station_watershed.parquet")
    pd.concat([watershed, pd.DataFrame({"station_id": ["THIN3"], "area_km2": [60.0],
                                        "truncated": [False], "drainage_area_flag": ["ok"]})],
              ignore_index=True).to_parquet(out / "station_watershed.parquet", index=False)
    climate = pd.read_parquet(out / "climate.parquet")
    pd.concat([climate, climate[climate["station_id"].eq("DOWN")].assign(station_id="THIN3")],
              ignore_index=True).to_parquet(out / "climate.parquet", index=False)
    bv.build_database(out / "warehouse.duckdb", out, 5, 5, 3, 2)
    for name in ["station_watershed", "land_cover", "land_cover_codes", "climate", "dam_reservoir"]:
        add_view(out / "warehouse.duckdb", name, out / f"{name}.parquet")


def _digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(module, root, *extra):
    import contextlib
    import io
    argv = [module.__name__, "--case-study", "pilot_test", *extra]
    buffer = io.StringIO()
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buffer):
        module.main()
    return buffer.getvalue()


def test_eligibility_variants_write_their_own_files_and_leave_the_primary_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _build(root)
        _add_thin_stations(out)
        _run(bad, root)                                    # the primary table, as the pipeline writes it
        primary_path = out / "analysis_measurement.parquet"
        primary = pd.read_parquet(primary_path)
        before = {p: _digest(out / p) for p in ("analysis_measurement.parquet", "warehouse.duckdb")}

        for variant in ("min1yr", "min1yr_2mo"):
            _run(bv, root, "--eligibility-variant", variant)
            log = _run(bad, root, "--eligibility-variant", variant)
            assert (out / f"warehouse_{variant}.duckdb").exists()
            assert "SENSITIVITY VARIANT" in log
        after = {p: _digest(out / p) for p in before}
        assert after == before                              # primary files byte-for-byte unchanged

        min1 = pd.read_parquet(out / "analysis_measurement_min1yr.parquet")
        min1_2mo = pd.read_parquet(out / "analysis_measurement_min1yr_2mo.parquet")
        # The 3/2 baseline: THIN stations aren't eligible (one pre year).
        assert set(primary["station_id"]) == {"DOWN", "OLD"}
        assert set(min1["station_id"]) == {"DOWN", "OLD", "THIN3", "THIN1"}
        assert set(min1_2mo["station_id"]) == {"DOWN", "OLD", "THIN3"}   # THIN1: one month only
        # Every primary row is in the variant, unchanged.
        key = ["measurement_id", "AR_ID"]
        merged = primary.merge(min1, on=key, suffixes=("", "_v"))
        assert len(merged) == len(primary)
        assert (merged["result_value_std"] == merged["result_value_std_v"]).all()
        # THIN1 has no context: reported, and its rows lack climate.
        assert "WARNING: 1 stations have no context" in _run(bad, root, "--eligibility-variant", "min1yr")
        assert min1.loc[min1["station_id"].eq("THIN1"), "missing_climate"].all()
        assert not min1.loc[min1["station_id"].eq("THIN3"), "missing_climate"].any()

        # --verify-primary rebuilds in memory, writes nothing, and catches a changed file.
        assert "identical to the existing" in _run(bad, root, "--verify-primary")
        assert _digest(primary_path) == before["analysis_measurement.parquet"]
        primary.iloc[1:].to_parquet(primary_path, index=False)
        try:
            _run(bad, root, "--verify-primary")
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("a changed primary table passed --verify-primary")


def test_months_condition_only_binds_single_year_sides():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = _build(root)
        _add_thin_stations(out)
        query = "SELECT station_id, has_both FROM eligibility_pairs WHERE parameter_std = 'water_temp'"
        results = {}
        for name, (years, months) in {"primary": (2, 1), "primary_2mo": (2, 2), "min1yr": (1, 1),
                                      "min1yr_2mo": (1, 2)}.items():
            db = out / f"warehouse_{name}_test.duckdb"
            bv.build_database(db, out, 5, 5, 3, years, min_months_single_year=months)
            with duckdb.connect(str(db), read_only=True) as con:
                results[name] = dict(con.sql(query).fetchall())
    assert results["primary"] == results["primary_2mo"]           # no effect with 2 required years
    assert results["primary"]["THIN3"] is False and results["primary"]["DOWN"] is True
    assert results["min1yr"]["THIN1"] is True and results["min1yr_2mo"]["THIN1"] is False
    assert results["min1yr_2mo"]["THIN3"] is True


def test_unknown_variant_is_refused():
    from etl.common.warehouse import analysis_path, views_db_path
    for function in (views_db_path, analysis_path):
        try:
            function(Path("x"), "min2yr")
        except ValueError:
            continue
        raise AssertionError(f"{function.__name__} accepted an unknown variant")
    assert analysis_path(Path("x")) == Path("x") / "analysis_measurement.parquet"


def test_context_stages_only_accept_variants_at_least_as_loose_as_the_primary():
    from etl.common import warehouse as wh

    class Cfg:
        name, project_root = "pilot_test", Path("/tmp/x")
        def __init__(self, eligibility): self.raw = {"eligibility": eligibility}
        def get(self, key, default=None): return self.raw.get(key, default)

    assert wh.context_db_path(Cfg({})).name == "warehouse.duckdb"
    assert wh.context_db_path(Cfg({}), "min1yr").name == "warehouse_min1yr.duckdb"
    try:
        wh.context_db_path(Cfg({"min_samples": 3, "min_years": 1}), "min1yr")   # equal: fine
        wh.context_db_path(Cfg({"min_samples": 2, "min_years": 1}), "min1yr")   # stricter: refused
    except ValueError as exc:
        assert "stricter" in str(exc)
    else:
        raise AssertionError("a variant stricter than the primary rule was accepted")
    assert wh.progress_stage("download_climate") == "download_climate"
    assert wh.progress_stage("download_climate", "min1yr") == "download_climate_min1yr"


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
