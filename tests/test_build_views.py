"""Checks for etl/build_views.py.

Writes tiny synthetic warehouse Parquet files (only the columns the views
use), builds warehouse.duckdb from them, and checks the window edges,
thresholds and same_reach placement.

Run directly: `python tests/test_build_views.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import duckdb
import pandas as pd

from etl import build_views as bv

REMOVAL_YEAR = 2015

# Enough data on both sides for the default thresholds (3 samples, 2 years),
# plus one removal-year and one outside-the-window sample.
BOTH_SIDES = [2010, 2013, 2013, 2015, 2016, 2017, 2020, 2021, 2009]
# Pre side has 3 samples but only 1 distinct year.
ONE_PRE_YEAR = [2014, 2014, 2014, 2016, 2017, 2018]


def _with_qc(stations):
    """The snapping flag and reach columns station_qc and link_features read."""
    for column in ["qc_far_snap", "qc_near_unsuitable_flowline", "qc_non_riverine_name"]:
        stations[column] = False
    if "reach_totdasqkm" not in stations:
        stations["reach_totdasqkm"] = 100.0
    return stations


def _with_reach(dams):
    """The dam reach columns link_features reads."""
    for column, value in [("reach_totdasqkm", 100.0), ("reach_slope", 0.001),
                          ("Dam_Height_ft", 10.0)]:
        if column not in dams:
            dams[column] = value
    return dams


def _measurements(rows):
    """rows: (station_id, sample year, parameter_std, value)."""
    frame = pd.DataFrame(rows, columns=["station_id", "year", "parameter_std", "result_value_std"])
    frame["sample_date"] = pd.to_datetime(frame["year"].astype(str) + "-06-01").dt.date
    frame["measurement_id"] = range(len(frame))
    frame["sample_fraction_std"] = pd.array([None] * len(frame), dtype="string")
    frame["result_unit_std"] = "x"
    frame["is_censored"] = False
    frame["unit_basis_assumed"] = False
    return frame.drop(columns="year")


def write_tables(directory):
    # Dam reach: 100 km2, slope 0.001, 10 ft -> backwater about 3.05 km.
    dams = _with_reach(pd.DataFrame({
        "AR_ID": ["D1", "D2"],
        "Dam_Name": ["Dam one", "No year"],
        "Year_Removed": pd.array([REMOVAL_YEAR, None], dtype="Int16"),
        "snap_fraction": [0.3, 0.5],
    }))
    stations = _with_qc(pd.DataFrame({
        "station_id": ["DOWN", "UP", "SAME_BELOW", "SAME_ABOVE", "NOYEAR"],
        "snap_fraction": [0.5, 0.5, 0.8, 0.1, 0.5],
        "reach_totdasqkm": [250.0, 40.0, 100.0, 100.0, -9999.0],
    }))
    links = pd.DataFrame({
        "AR_ID": ["D1", "D1", "D1", "D1", "D2"],
        "station_id": ["DOWN", "UP", "SAME_BELOW", "SAME_ABOVE", "NOYEAR"],
        "relationship": ["downstream", "upstream", "same_reach", "same_reach", "downstream"],
        "network_distance_km": [1.0, 2.0, 0.1, 0.1, 1.0],
    })
    years = {
        "DOWN": BOTH_SIDES,
        "UP": ONE_PRE_YEAR,
        "SAME_BELOW": BOTH_SIDES,
        "SAME_ABOVE": BOTH_SIDES,
        "NOYEAR": BOTH_SIDES,
    }
    rows = [
        (station, f"{year}-06-01")
        for station, station_years in years.items()
        for year in station_years
    ]
    measurements = pd.DataFrame(rows, columns=["station_id", "sample_date"])
    measurements["sample_date"] = pd.to_datetime(measurements["sample_date"]).dt.date
    measurements["measurement_id"] = range(len(measurements))
    measurements["parameter_std"] = "water_temp"
    measurements["sample_fraction_std"] = pd.array([None] * len(measurements), dtype="string")
    measurements["result_value_std"] = 10.0
    measurements["result_unit_std"] = "deg C"
    measurements["is_censored"] = False
    measurements["unit_basis_assumed"] = False

    tables = {
        "dam_removal": dams,
        "wq_station": stations,
        "dam_station_link": links,
        "huc8_watershed": pd.DataFrame({"huc8": ["01010101"]}),
        "streams": pd.DataFrame({"nhdplus_id": [1]}),
        "wq_measurement": measurements,
        "wq_measurement_raw": pd.DataFrame({"x": [1]}),
    }
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)


def build(directory, **limits):
    write_tables(directory)
    db_path = directory / "warehouse.duckdb"
    bv.build_database(db_path, directory, years_before=5, years_after=5,
                      min_samples=3, min_years=2, **limits)
    return duckdb.connect(str(db_path), read_only=True)


def test_periods_match_window_definition():
    with tempfile.TemporaryDirectory() as tmp, build(Path(tmp)) as con:
        periods = dict(con.sql("""
            SELECT sample_year, any_value(period) FROM measurement_window
            WHERE station_id = 'DOWN' GROUP BY sample_year
        """).fetchall())
    assert periods == {
        2009: "outside", 2010: "pre", 2013: "pre", 2015: "removal_year",
        2016: "post", 2017: "post", 2020: "post", 2021: "outside",
    }


def test_dams_without_removal_year_are_excluded():
    with tempfile.TemporaryDirectory() as tmp, build(Path(tmp)) as con:
        n = con.sql("SELECT count(*) FROM measurement_window WHERE AR_ID = 'D2'").fetchone()[0]
    assert n == 0


def test_same_reach_placed_by_snap_fraction():
    with tempfile.TemporaryDirectory() as tmp, build(Path(tmp)) as con:
        sides = dict(con.sql(
            "SELECT DISTINCT station_id, side FROM measurement_window"
        ).fetchall())
    assert sides == {
        "DOWN": "downstream", "UP": "upstream",
        "SAME_BELOW": "downstream", "SAME_ABOVE": "upstream",
    }


def test_thresholds_and_rollups():
    with tempfile.TemporaryDirectory() as tmp, build(Path(tmp)) as con:
        pairs = con.sql("""
            SELECT station_id, n_pre, years_pre, n_post, years_post, has_both
            FROM eligibility_pairs ORDER BY station_id
        """).df().set_index("station_id")
        dam = con.sql("SELECT * FROM eligibility_dams").df().iloc[0]
        summary = con.sql("SELECT * FROM eligibility_summary").df().iloc[0]

    assert pairs.loc["DOWN"].tolist() == [3, 2, 3, 3, True]
    assert pairs.loc["UP"].tolist() == [3, 1, 3, 3, False]  # one pre year only
    assert dam["downstream_stations_both"] == 2  # DOWN, SAME_BELOW
    assert dam["upstream_stations_both"] == 1  # SAME_ABOVE
    assert dam["has_both"] and dam["has_upstream_control"]
    assert summary["dams_eligible"] == 1
    assert summary["dams_with_removal_year"] == 1


# Exposure scenarios (window 5 years each side). Distances in km.
#   ST_SOLE  below A only                                     -> assigned_sole
#   ST_NEAR  below A (1 km, 2010) and F (10 km, 2017)          -> 2013: assigned_nearest A
#   ST_DIST  below A (1 km, 2010) and G (1.5 km, 2018)         -> 2014: competing_distance
#   ST_TIME  below A (1 km, 2010) and H (20 km, 2012)          -> competing_timing
EXPOSURE_LINKS = [
    ("A", "ST_SOLE", 1.0), ("A", "ST_NEAR", 1.0), ("F", "ST_NEAR", 10.0),
    ("A", "ST_DIST", 1.0), ("G", "ST_DIST", 1.5), ("A", "ST_TIME", 1.0), ("H", "ST_TIME", 20.0),
]
EXPOSURE_YEARS = {"A": 2010, "F": 2017, "G": 2018, "H": 2012}


def build_exposure(directory):
    dams = _with_reach(pd.DataFrame({
        "AR_ID": list(EXPOSURE_YEARS), "Dam_Name": list(EXPOSURE_YEARS),
        "Year_Removed": pd.array(list(EXPOSURE_YEARS.values()), dtype="Int16"),
        "snap_fraction": 0.5,
    }))
    stations = _with_qc(pd.DataFrame({
        "station_id": ["ST_SOLE", "ST_NEAR", "ST_DIST", "ST_TIME"], "snap_fraction": 0.5}))
    links = pd.DataFrame(EXPOSURE_LINKS, columns=["AR_ID", "station_id", "network_distance_km"])
    links["relationship"] = "downstream"
    measurements = _measurements([
        ("ST_SOLE", 2012, "water_temp", 10.0),
        ("ST_NEAR", 2013, "water_temp", 10.0), ("ST_NEAR", 2008, "water_temp", 10.0),
        ("ST_NEAR", 2020, "water_temp", 10.0),
        ("ST_DIST", 2014, "water_temp", 10.0),
        ("ST_TIME", 2008, "water_temp", 10.0),
        # Saline: median 4500 > 3000. ST_NEAR is fresh; the others have no conductance.
        ("ST_SOLE", 2012, "specific_conductance", 4000.0),
        ("ST_SOLE", 2012, "specific_conductance", 4500.0),
        ("ST_SOLE", 2012, "specific_conductance", 5000.0),
        ("ST_NEAR", 2012, "specific_conductance", 300.0),
    ])
    tables = {
        "dam_removal": dams, "wq_station": stations, "dam_station_link": links,
        "huc8_watershed": pd.DataFrame({"huc8": ["01010101"]}),
        "streams": pd.DataFrame({"nhdplus_id": [1]}),
        "wq_measurement": measurements, "wq_measurement_raw": pd.DataFrame({"x": [1]}),
    }
    for name, frame in tables.items():
        frame.to_parquet(directory / f"{name}.parquet", index=False)
    db_path = directory / "warehouse.duckdb"
    bv.build_database(db_path, directory, years_before=5, years_after=5,
                      min_samples=1, min_years=1)
    return duckdb.connect(str(db_path), read_only=True)


def test_exposure_assignment():
    with tempfile.TemporaryDirectory() as tmp, build_exposure(Path(tmp)) as con:
        exposure = con.sql("""
            SELECT station_id, year(m.sample_date) AS year, e.exposure_status, e.assigned_AR_ID,
                   e.nearest_AR_ID, e.next_AR_ID
            FROM measurement_exposure e JOIN wq_measurement m USING (measurement_id, station_id)
            WHERE m.parameter_std = 'water_temp'
        """).df()
    got = {(r.station_id, r.year): (r.exposure_status, None if pd.isna(r.assigned_AR_ID)
                                    else r.assigned_AR_ID, r.nearest_AR_ID)
           for r in exposure.itertuples()}
    assert got == {
        ("ST_SOLE", 2012): ("assigned_sole", "A", "A"),
        ("ST_NEAR", 2013): ("assigned_nearest", "A", "A"),   # F is 10x as far
        ("ST_NEAR", 2008): ("assigned_sole", "A", "A"),      # outside F's window
        ("ST_NEAR", 2020): ("assigned_sole", "F", "F"),      # outside A's window
        ("ST_DIST", 2014): ("competing_distance", None, "A"),
        ("ST_TIME", 2008): ("competing_timing", None, "A"),  # H removed inside A's window
    }


def test_pair_competing_removals_lists_other_removals():
    with tempfile.TemporaryDirectory() as tmp, build_exposure(Path(tmp)) as con:
        pairs = con.sql("SELECT * FROM pair_competing_removals").df() \
            .set_index(["AR_ID", "station_id"])
    assert pairs.loc[("A", "ST_TIME"), "n_other_upstream_removals_in_window"] == 1
    assert pairs.loc[("A", "ST_TIME"), "other_removals_in_window"] == "H (2012, station downstream)"
    assert pairs.loc[("A", "ST_NEAR"), "n_other_removals_in_window"] == 0  # F (2017) outside A's window


def test_station_qc_saline_flag():
    with tempfile.TemporaryDirectory() as tmp, build_exposure(Path(tmp)) as con:
        qc = con.sql("SELECT * FROM station_qc").df().set_index("station_id")
    assert qc.loc["ST_SOLE", "qc_saline"] == True  # noqa: E712
    assert qc.loc["ST_SOLE", "sc_median_us_cm"] == 4500
    assert qc.loc["ST_NEAR", "qc_saline"] == False  # noqa: E712
    assert pd.isna(qc.loc["ST_DIST", "qc_saline"])  # no conductance: unknown, not fresh


def test_distance_limits_filter_links():
    # DOWN is 1 km below, SAME_BELOW 0.1 km below, UP 2 km above, SAME_ABOVE 0.1 km above.
    with tempfile.TemporaryDirectory() as tmp, \
            build(Path(tmp), max_downstream_km=0.5, max_upstream_km=1.0) as con:
        kept = {r[0] for r in con.sql("SELECT station_id FROM removal_link").fetchall()}
        every = {r[0] for r in con.sql("SELECT station_id FROM removal_link_all").fetchall()}
        windowed = {r[0] for r in con.sql(
            "SELECT DISTINCT station_id FROM measurement_window").fetchall()}
    assert kept == windowed == {"SAME_BELOW", "SAME_ABOVE"}
    assert every == {"DOWN", "UP", "SAME_BELOW", "SAME_ABOVE"}  # D2 has no removal year


def test_no_limits_keep_every_link():
    with tempfile.TemporaryDirectory() as tmp, build(Path(tmp)) as con:
        n_kept = con.sql("SELECT count(*) FROM removal_link").fetchone()[0]
        n_all = con.sql("SELECT count(*) FROM removal_link_all").fetchone()[0]
    assert n_kept == n_all == 4


def test_link_features():
    with tempfile.TemporaryDirectory() as tmp, build(Path(tmp)) as con:
        f = con.sql("SELECT * FROM link_features").df().set_index("station_id")
    assert f.loc["DOWN", "drainage_area_ratio"] == 2.5
    assert f.loc["UP", "drainage_area_ratio"] == 0.4
    assert abs(f.loc["UP", "backwater_km"] - 3.048) < 1e-9
    assert f.loc["UP", "upstream_in_former_reservoir"] == True  # noqa: E712  2 km < 3.05 km
    assert pd.isna(f.loc["DOWN", "upstream_in_former_reservoir"])  # downstream: not applicable


def test_stale_derived_tables_are_reported():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        write_tables(directory)
        # Climate from an earlier run, for a station wq_station no longer has.
        pd.DataFrame({"station_id": ["DOWN", "GONE"], "year": [2015, 2015], "month": [6, 6]}) \
            .to_parquet(directory / "climate.parquet", index=False)
        _, stale = bv.build_database(directory / "warehouse.duckdb", directory, 5, 5, 3, 2)
    assert len(stale) == 1 and stale[0].startswith("climate: 1 station_id(s) not in wq_station")


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
