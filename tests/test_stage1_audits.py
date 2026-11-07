"""Checks for the Stage 1 read-only audits (analysis/audits/).

Run directly: `python tests/test_stage1_audits.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import duckdb
import numpy as np
import pandas as pd

import test_build_views as views_fixture
from analysis.audit_utils import connect
from analysis.audits import dam_funnel_audit as fun
from analysis.audits import downstream_coverage_audit as cov
from analysis.audits import reservoir_size_audit as res
from etl.common.config import PROJECT_ROOT_ENV_VAR, load_config

CASE = "audit_test"


# --------------------------------------------------
# Reservoir size
# --------------------------------------------------

def test_size_variables_units_and_invalid_values():
    frame = pd.DataFrame({
        "Dam_Height_ft": [10.0, 10.0, 0.0, 10.0, 10.0, None],
        "reach_slope": [0.001, 0.0, 0.001, 0.001, 1e-5, 0.001],
        "reach_totdasqkm": [100.0, 100.0, 100.0, -9999.0, 100.0, 100.0],
    })
    out = res.size_variables(frame)
    # Same formula as build_views.link_features: 10 ft / 0.001 = 3.048 km.
    assert abs(out.loc[0, "backwater_km"] - 3.048) < 1e-12
    assert abs(out.loc[0, "reservoir_volume_proxy"] - 3.048 ** 2 / 0.001 * 10) < 1e-6
    assert out.loc[1, ["backwater_km", "reservoir_volume_proxy"]].isna().all()   # zero slope
    assert out.loc[2, ["height_m", "backwater_km"]].isna().all()                 # zero height
    assert pd.notna(out.loc[3, "backwater_km"]) and pd.isna(out.loc[3, "reservoir_volume_proxy"])
    assert out.loc[4, "slope_at_floor"] == True and out.loc[0, "slope_at_floor"] == False  # noqa: E712
    # At the NHDPlus minimum slope, backwater and proxy are unknown (as in build_views).
    assert out.loc[4, ["backwater_km", "reservoir_volume_proxy"]].isna().all()
    assert pd.isna(out.loc[1, "slope_at_floor"])
    assert out.loc[5, ["height_m", "reservoir_volume_proxy"]].isna().all()


def test_polygon_timing():
    fdate = pd.Series(pd.to_datetime(["2010-05-01", "2016-01-01", None, "2015-12-31"]))
    years = pd.Series([2015, 2015, 2015, 2015])
    assert res.polygon_timing(fdate, years).tolist() == [
        "edited_by_removal_year", "edited_after_removal", "date_unknown",
        "edited_by_removal_year"]


def test_storage_file_detection_and_matching():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nid.csv"
        pd.DataFrame({
            "Federal ID": ["ME00001", "me00002 ", "ME00001", "NY00003"],
            "NID Storage (Acre-Ft)": [100, 0, 5, 50],
            "Normal Storage (Acre-Ft)": [80, None, 4, 40],
            "Surface Area (Acres)": [10, 2, 1, 5],
        }).to_csv(path, index=False)
        storage, notes = res.read_storage_file(path)
    assert any("'Federal ID'" in n for n in notes)
    assert any("duplicate ids (first kept): 1" in n for n in notes)
    row = storage.set_index("storage_id")
    assert row.loc["ME00001", "nid_storage_acre_ft"] == 100
    assert pd.isna(row.loc["ME00002", "nid_storage_acre_ft"])        # 0 -> missing
    assert abs(row.loc["ME00001", "nid_surface_area_km2"] - 10 * res.ACRE_TO_KM2) < 1e-12

    found, matched = res.measured_storage(pd.Series(["ME00001", None, "ZZ1", "ny00003"]), storage)
    assert matched.tolist() == [True, False, False, True]
    assert found["normal_storage_acre_ft"].tolist()[0] == 80 and found.iloc[1].isna().all()


def test_storage_file_without_columns_is_refused():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.csv"
        pd.DataFrame({"name": ["a"], "value": [1]}).to_csv(path, index=False)
        try:
            res.read_storage_file(path)
            raise AssertionError("expected ValueError")
        except ValueError as exc:
            assert "--id-column" in str(exc) and "name" in str(exc)


def test_correlations_report_insufficient_samples():
    rng = np.random.default_rng(0)
    x = 10 ** rng.normal(0, 1, 30)
    table = pd.DataFrame({"area": x * 3, "proxy": x, "few": [np.nan] * 25 + [1.0] * 5})
    out = res.correlations(table, ["area", "few"], ["proxy"], min_n=10).set_index("measured")
    assert out.loc["area", "n"] == 30 and abs(out.loc["area", "pearson_log10"] - 1) < 1e-9
    assert out.loc["few", "note"].startswith("insufficient")


def _reservoir_project(root):
    (root / "config").mkdir()
    (root / "config" / f"{CASE}.yaml").write_text(
        f'case_study: {CASE}\nhuc8s: ["01010001"]\ndam_source: dams.csv\nwbd_source: wbd.gpkg\n')
    pd.DataFrame({
        "AR_ID": ["A", "B", "C", "D"], "NABI_ID": ["N1", None, "N3", None],
        "NID_ID": ["ME00001", None, "ME00009", None], "State": ["ME"] * 4,
        "Year_Removed": ["2015", "2012", "2018", None], "Dam_Height_ft": ["10", "20", "5", "8"],
    }).to_csv(root / "dams.csv", index=False)
    warehouse = root / "data" / "warehouse" / CASE
    warehouse.mkdir(parents=True)
    pd.DataFrame({"AR_ID": ["A", "B"], "Dam_Height_ft": [10.0, 20.0],
                  "reach_slope": [0.001, 0.002], "reach_totdasqkm": [100.0, 400.0]}) \
        .to_parquet(warehouse / "dam_removal.parquet", index=False)
    reaches = root / "data" / "processed" / "dam_reaches" / CASE
    reaches.mkdir(parents=True)
    pd.DataFrame({
        "AR_ID": ["A", "B", "C"], "huc8": ["01010001"] * 3,
        "match_status": ["accepted", "accepted", "ambiguous"],
        "reservoir_status": ["dam_at_outlet", "no_reservoir_polygon", "dam_at_outlet"],
        "reservoir_id": ["{abc}", None, "XYZ"], "reservoir_ftype": [390, None, 436],
        "reservoir_fdate": pd.to_datetime(["2010-01-01", None, "2020-01-01"]),
    }).to_parquet(reaches / "dam_reaches_01010001.parquet", index=False)
    pd.DataFrame({"NIDID": ["ME00001"], "NORM_STOR": [250.0]}).to_csv(root / "nabd.csv",
                                                                      index=False)


def test_reservoir_build_keeps_measured_and_constructed_apart():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _reservoir_project(root)
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}):
            cfg = load_config(CASE)
            table, notes = res.build(cfg, storage_file=root / "nabd.csv", read_polygons=False)
    table = table.set_index("AR_ID")
    assert sorted(table.index) == ["A", "B", "C"]                 # D has no removal year
    assert table.loc["A", "in_analysis"] and not table.loc["C", "in_analysis"]
    assert abs(table.loc["A", "backwater_km"] - 3.048) < 1e-12
    # C is outside dam_removal: height from the dam list, no slope -> no proxy.
    assert abs(table.loc["C", "height_m"] - 5 * 0.3048) < 1e-12
    assert pd.isna(table.loc["C", "reservoir_volume_proxy"])
    assert table.loc["A", "polygon_timing"] == "edited_by_removal_year"
    assert table.loc["C", "polygon_timing"] == "edited_after_removal"
    assert pd.isna(table.loc["B", "polygon_timing"])
    # Storage matched on NID_ID only; nothing imputed.
    assert table.loc["A", "normal_storage_acre_ft"] == 250
    assert table["normal_storage_acre_ft"].notna().sum() == 1
    assert table["storage_matched_by_NID_ID"].tolist() == [True, False, False]
    assert any("matched on NID_ID: 1 of 2" in n for n in notes)
    # NABI_ID is never a join key, even when the file's ids look like it.
    assert "storage_matched_by_NABI_ID" not in table
    assert not any("NABI_ID" in n for n in notes)

    cov_table = res.coverage(table.reset_index(), ["normal_storage_acre_ft",
                                                   "storage_matched_by_NID_ID"],
                             {"all": pd.Series(True, index=range(3))})
    assert cov_table["with_value"].tolist() == [1, 1]


def test_storage_never_matches_on_nabi_id():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _reservoir_project(root)
        # A file keyed by A's NABI_ID ("N1"): must not match anything.
        pd.DataFrame({"NIDID": ["N1"], "NORM_STOR": [999.0]}).to_csv(root / "nabd.csv",
                                                                     index=False)
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}):
            table, _ = res.build(load_config(CASE), storage_file=root / "nabd.csv",
                                 read_polygons=False)
    assert table["normal_storage_acre_ft"].isna().all()
    assert not table["storage_matched_by_NID_ID"].any()


def test_collinearity_with_drainage_area():
    area = np.array([1, 10, 100, 1000, 10, 100, 1000, 1, 100, 10, 1000, 1.0])
    table = pd.DataFrame({"reach_totdasqkm": area, "proxy": area ** 2,
                          "noise": np.resize([5.0, 1.0, 3.0, 2.0], 12), "few": [1.0] + [None] * 11})
    out = res.collinearity(table, ["proxy", "noise", "few", "absent"]).set_index("size")
    assert out.loc["proxy", "pearson_log10"] == 1 and out.loc["proxy", "vif"] > 1000
    assert out.loc["noise", "vif"] < 5 and out.loc["noise", "n"] == 12
    assert "insufficient" in out.loc["few", "note"] and "absent" not in out.index


# --------------------------------------------------
# Downstream coverage
# --------------------------------------------------

def test_band_edges_are_upper_inclusive():
    with duckdb.connect() as con:
        got = con.sql(f"SELECT d, {cov.band_sql('d')} FROM (VALUES (0.1), (5.0), (5.01), "
                      f"(30.0), (31.0), (NULL)) t(d)").fetchall()
    assert [b for _, b in got] == ["00-05", "00-05", "05-10", "20-30", "30+",
                                   "distance_unknown"]


def test_coverage_counts_from_unlimited_views():
    # build_views fixture: dam D1 (2015) with DOWN 1 km and SAME_BELOW 0.1 km
    # downstream (both with data on both sides); UP 2 km upstream has only
    # one pre year, so D1 has no eligible control.
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        views_fixture.write_tables(directory)
        con, _ = connect(directory, 5, 5, 3, 2)
        pairs = con.sql(cov.coverage_sql(max_upstream_km=10)).df()
        links = con.sql(cov.LINK_CHECK_SQL).df()
        con.close()
    bands = cov.by_band(pairs).set_index(["series", "band"])
    row = bands.loc[("ALL", "00-05")]
    assert row["dams_has_both"] == 1 and row["stations_has_both"] == 2
    assert row["dams_has_both_with_control"] == 0
    assert row["obs_has_both"] == 2 * 6  # 3 pre + 3 post each (2009, 2021 outside; 2015 removal year)
    assert row["dams_has_both_nearest"] == 1

    cumulative = cov.cumulative(pairs, current_cutoff=10).set_index(["series", "cutoff_km"])
    assert cumulative.loc[("ALL", 30), "dams_has_both"] == 1
    assert cumulative.loc[("ALL", 30), "dams_has_both_gain_vs_current"] == 0

    link = links.set_index("band").loc["00-05"]
    assert link["links"] == 2 and link["links_with_window_data"] == 2


def test_band_labels_match_band_sql():
    distance = pd.Series([0.0, 0.1, 5.0, 5.01, 30.0, 31.0, None])
    assert cov.band_labels(distance).tolist() == ["00-05", "00-05", "00-05", "05-10", "20-30",
                                                  "30+", "distance_unknown"]


def test_confluence_blocked_links():
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        pd.DataFrame({
            "dam_id": [1.0, 1.0, 2.0, 2.0, 3.0, 9.0],
            "station_id": ["U1", "C1", "C2", "O1", "C3", "C9"],
            "relationship": ["downstream"] * 5 + ["downstream"],
            "network_distance_km": [4.0, 12.0, 18.0, 3.0, 25.0, 2.0],
            "dam_status": ["accepted", "accepted", "accepted", "ambiguous", "accepted",
                           "accepted"],
            "station_status": ["accepted", "ambiguous", "ambiguous", "accepted", "ambiguous",
                               "ambiguous"],
            "station_reason": [None, "confluence_ambiguous", "confluence_ambiguous", None,
                               "far_without_support", "confluence_ambiguous"],
            "usable": [True, False, False, False, False, False],
        }).to_parquet(folder / "reach_links_01010001.parquet", index=False)
        pd.DataFrame({"dam_id": [1.0], "station_id": ["X"], "relationship": ["upstream"],
                      "network_distance_km": [1.0], "dam_status": ["accepted"],
                      "station_status": ["ambiguous"], "station_reason": ["confluence_ambiguous"],
                      "usable": [False]}).to_parquet(folder / "reach_links_01010002.parquet",
                                                     index=False)
        links = cov.read_reach_links(folder)
    assert len(links) == 6                                        # the upstream pair is left out
    assert links.set_index("station_id")["link_class"].to_dict() == {
        "U1": "usable", "C1": "confluence_blocked", "C2": "confluence_blocked",
        "O1": "other_blocked", "C3": "other_blocked", "C9": "confluence_blocked"}
    # Dam 9 has no removal year (not in the set): left out.
    by_band, gain = cov.confluence_tables(links, {"1", "2", "3"}, {"C2"})
    gain = gain.set_index("cutoff_km")
    assert gain.loc[10, "dams_usable"] == 1 and gain.loc[10, "dams_only_confluence_blocked"] == 0
    # Within 20 km: dam 2's only downstream links are blocked (C2 confluence).
    assert gain.loc[20, "dams_only_confluence_blocked"] == 1
    assert gain.loc[20, "confluence_blocked_pairs"] == 2
    assert gain.loc[20, "confluence_blocked_stations_with_any_data"] == 1
    row = by_band.set_index(["band", "link_class"]).loc[("15-20", "confluence_blocked")]
    assert row["pairs"] == 1 and row["stations_with_any_data"] == 1


# --------------------------------------------------
# Dam funnel
# --------------------------------------------------

def _funnel_warehouse(directory):
    """The build_views fixture (D1 in the table, D2 without a year) plus one
    dam per later funnel stage: D3 no link, D4 only a link at 50 km, D5 a
    station with no data, D6 pre-removal data only, D7 one pre year only."""
    views_fixture.write_tables(directory)
    dams = pd.read_parquet(directory / "dam_removal.parquet")
    extra = pd.DataFrame({"AR_ID": [f"D{i}" for i in range(3, 8)], "Dam_Name": "x",
                          "Year_Removed": pd.array([2015] * 5, dtype="Int16"),
                          "snap_fraction": 0.5})
    dams = views_fixture._with_reach(pd.concat([dams, extra], ignore_index=True))
    dams["State"], dams["HUC8"] = "ME", "01010101"
    dams["NHDPlusID"], dams["Latitude"], dams["Longitude"] = 1, 45.0, -69.0
    stations = pd.read_parquet(directory / "wq_station.parquet")
    new = views_fixture._with_qc(pd.DataFrame({"station_id": ["FAR", "NODATA", "PREONLY", "THIN"],
                                               "snap_fraction": 0.5}))
    stations = pd.concat([stations, new], ignore_index=True)
    links = pd.concat([pd.read_parquet(directory / "dam_station_link.parquet"), pd.DataFrame({
        "AR_ID": ["D4", "D5", "D6", "D7"], "station_id": ["FAR", "NODATA", "PREONLY", "THIN"],
        "relationship": "downstream", "network_distance_km": [50.0, 2.0, 2.0, 2.0]})],
        ignore_index=True)
    measurements = pd.read_parquet(directory / "wq_measurement.parquet")
    more = [("FAR", y) for y in views_fixture.BOTH_SIDES] + \
           [("PREONLY", y) for y in (2011, 2012, 2013, 2014)] + \
           [("THIN", y) for y in views_fixture.ONE_PRE_YEAR]
    more = pd.DataFrame(more, columns=["station_id", "year"])
    more["sample_date"] = pd.to_datetime(more["year"].astype(str) + "-06-01").dt.date
    more = more.drop(columns="year").assign(
        measurement_id=range(len(measurements), len(measurements) + len(more)),
        parameter_std="water_temp", result_value_std=10.0, result_unit_std="deg C",
        is_censored=False, unit_basis_assumed=False)
    more["sample_fraction_std"] = pd.array([None] * len(more), dtype="string")
    measurements = pd.concat([measurements, more[measurements.columns]], ignore_index=True)
    for name, frame in (("dam_removal", dams), ("wq_station", stations),
                        ("dam_station_link", links), ("wq_measurement", measurements)):
        frame.to_parquet(directory / f"{name}.parquet", index=False)


def test_funnel_gives_each_dam_its_first_failing_stage():
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _funnel_warehouse(directory)
        con, _ = connect(directory, 5, 5, 3, 2, max_downstream_km=30, max_upstream_km=10)
        status = fun.dam_status(con).set_index("AR_ID")
        whatif = fun.threshold_whatif(con, (3, 2)).set_index(["min_samples", "min_years"])
        con.close()
    assert status["reason"].to_dict() == {
        "D1": "in_analysis_table", "D2": "no_removal_year", "D3": "no_linked_station",
        "D4": "links_beyond_limits", "D5": "no_samples_in_window", "D6": "one_side_only",
        "D7": "below_minimum"}
    assert status.loc["D4", "nearest_downstream_km"] == 50 and status.loc["D4", "links_within_limits"] == 0
    assert status.loc["D5", "stations_with_any_data"] == 0
    assert status.loc["D7", "best_pair_min_years"] == 1 and status.loc["D1", "has_both_downstream"]
    # D7 (3 pre samples in one year) passes once one year is enough; D1 passes every rule.
    assert whatif.loc[(3, 2), "dams"] == 1 and whatif.loc[(3, 2), "current_rule"]
    assert whatif.loc[(3, 1), "dams"] == 2 and whatif.loc[(3, 1), "gain_vs_current"] == 1
    # D6 never has post data, at any threshold.
    assert whatif.loc[(1, 1), "dams"] == 2


def test_funnel_early_stages_summary_and_decades():
    listed = pd.DataFrame({"AR_ID": ["D1", "L1", "L2", "L3", "7.0"]})
    reaches = pd.DataFrame({"AR_ID": ["D1", "L2", "L3", "7"], "huc8": "01010101",
                            "match_status": ["accepted", "rejected", "ambiguous", "accepted"],
                            "match_reason": ["name_match", "no_nhdplus_package",
                                             "confluence_ambiguous", "name_match"]})
    early = fun.listed_and_matched(listed, reaches).set_index("AR_ID")
    assert early["reason"].to_dict() == {"D1": "accepted", "L1": "not_matched_no_record",
                                         "L2": "match_rejected", "L3": "match_ambiguous",
                                         "7": "accepted"}            # "7.0" and 7 are one dam
    dams = pd.DataFrame({"reason": ["match_rejected", "no_removal_year", "in_analysis_table",
                                    "below_minimum", "in_analysis_table"],
                         "removal_year": [None, None, 1995, 2012, 2018]})
    s = fun.summary(dams).set_index("reason")
    assert s.loc["in_analysis_table", "dams"] == 2 and s.loc["match_rejected", "dams"] == 1
    assert s.loc["no_removal_year", "dams_left_after_this_stage"] == 3
    assert list(s.index) == fun.REASON_ORDER
    decade = fun.by_decade(dams).set_index("decade")
    assert decade.loc["2010s", "dated_dams"] == 2 and decade.loc["2010s", "share_in_analysis_%"] == 50
    assert "match_rejected" not in decade           # undated dams aren't in the decade table


def test_downstream_walk_finds_where_the_river_leaves_the_huc8():
    # 1 -> 2 -> 3 in HUC8 01010101, then 4 in 01010102; 3 also splits to 9 (a minor path).
    flow = pd.DataFrame({"FromNHDPID": [1, 2, 3, 3, 4], "ToNHDPID": [2, 3, 9, 4, 0]})
    nxt = fun.main_path(flow, {9: 2, 4: 0})
    assert nxt == {1: 2, 2: 3, 3: 4}                   # minor divergence and terminal 0 skipped
    code = {1: "01010101000001", 2: "01010101000002", 3: "01010101000003", 4: "01010102000001"}
    length = {1: 2.0, 2: 3.0, 3: 4.0, 4: 10.0}
    assert fun.walk_downstream(1, 1.0, code, length, nxt, 30) == ("left_huc8", 8.0, "01010102")
    assert fun.walk_downstream(1, 1.0, code, length, nxt, 5)[0] == "reached_limit"
    assert fun.walk_downstream(4, 0.0, code, length, nxt, 30)[0] == "network_end"
    del code[4]
    assert fun.walk_downstream(1, 1.0, code, length, nxt, 30)[0] == "left_package"


def test_station_types_and_nearby_counts():
    raw = pd.DataFrame({
        "MonitoringLocationIdentifier": ["A", "B", "C", "D", "E"],
        "MonitoringLocationTypeName": ["River/Stream", "River/Stream Perennial", "Lake",
                                       "Riverine Impoundment", "River/Stream Intermittent"],
        "LatitudeMeasure": ["45.0", "45.01", "45.0", "45.5", "45.02"],
        "LongitudeMeasure": ["-69.0", "-69.0", "-69.0", "-69.0", "-69.0"],
        "huc8": "01010101"})
    types = fun.station_types(raw, ["Stream", "River/Stream"]).set_index("type")
    assert types.loc["River/Stream", "kept_by_filter"]
    assert types.loc["River/Stream Perennial", "river_like_but_dropped"]
    assert not types.loc["Lake", "river_like_but_dropped"]
    dams = pd.DataFrame({"huc8": ["01010101", "09999999"], "latitude": [45.0, 45.0],
                         "longitude": [-69.0, -69.0]})
    near = fun.nearby_stations(dams, raw, ["Stream", "River/Stream"])
    # Within 5 km of dam 0: A (kept), B and E (dropped river-like); D is ~55 km away; C is a lake.
    assert near.iloc[0].tolist() == [1, 2]
    assert near.iloc[1].tolist() == [0, 0]            # no stations downloaded for that HUC8


def test_funnel_command_line_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config").mkdir()
        (root / "config" / f"{CASE}.yaml").write_text(
            f'case_study: {CASE}\nhuc8s: ["01010101"]\ndam_source: dams.csv\nwbd_source: wbd.gpkg\n'
            "exposure:\n  max_downstream_km: 30\n  max_upstream_km: 10\n")
        warehouse = root / "data" / "warehouse" / CASE
        warehouse.mkdir(parents=True)
        _funnel_warehouse(warehouse)
        pd.DataFrame({"AR_ID": ["D1", "D3"], "side": ["downstream", "upstream"],
                      "exposure_status": ["assigned_sole", "upstream_control"],
                      "upstream_in_former_reservoir": [None, False]}).to_parquet(
            warehouse / "analysis_measurement.parquet", index=False)
        ids = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
        pd.DataFrame({"AR_ID": ids + ["L1", "L2"], "Dam_Name": "x", "State": "ME"}).to_csv(
            root / "dams.csv", index=False)
        reaches = root / "data" / "processed" / "dam_reaches" / CASE
        reaches.mkdir(parents=True)
        pd.DataFrame({"AR_ID": ids + ["L2"], "huc8": "01010101",
                      "match_status": ["accepted"] * 7 + ["rejected"],
                      "match_reason": ["name_match"] * 7 + ["no_nhdplus_package"]}).to_parquet(
            reaches / "dam_reaches_01010101.parquet", index=False)
        stations = root / "data" / "raw" / "wqp" / "stations"
        stations.mkdir(parents=True)
        pd.DataFrame({"MonitoringLocationIdentifier": ["S1", "S2"],
                      "MonitoringLocationTypeName": ["River/Stream", "River/Stream Perennial"],
                      "LatitudeMeasure": [45.0, 45.01], "LongitudeMeasure": [-69.0, -69.0]}).to_csv(
            stations / "01010101.csv", index=False)
        argv = ["dam_funnel_audit.py", "--case-study", CASE, "--skip-network"]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            fun.main()
        out = root / "data" / "processed" / "stage1_audit" / CASE / "dam_funnel"
        dams = pd.read_csv(out / "dam_funnel.csv", dtype={"AR_ID": str}).set_index("AR_ID")
        summary = pd.read_csv(out / "funnel_summary.csv").set_index("reason")
        types = pd.read_csv(out / "station_types.csv").set_index("type")
        model = pd.read_csv(out / "model_needs.csv").set_index("group")
        assert set((out / n).name for n in out.iterdir()) >= {"funnel_by_decade.csv",
                                                              "threshold_whatif.csv"}
    assert dams.loc["L1", "reason"] == "not_matched_no_record"
    assert dams.loc["L2", "reason"] == "match_rejected" and dams.loc["L2", "match_reason"] == "no_nhdplus_package"
    assert dams.loc["D4", "reason"] == "links_beyond_limits" and dams.loc["D1", "reason"] == "in_analysis_table"
    assert len(dams) == 9 and summary["dams"].sum() == 9
    assert summary.loc["in_analysis_table", "dams"] == 1
    # D3 (no station) has the dropped River/Stream Perennial station and the kept one within 5 km.
    assert dams.loc["D3", "dropped_river_like_stations_within_5km"] == 1
    assert dams.loc["D3", "kept_type_stations_within_5km"] == 1
    assert pd.isna(dams.loc["D1", "dropped_river_like_stations_within_5km"])  # has a usable station
    assert types.loc["River/Stream Perennial", "river_like_but_dropped"]
    assert model.loc["in analysis table", "dams"] == 1
    assert model.loc["assigned downstream rows", "dams"] == 1
    assert model.loc["both (BACI possible)", "dams"] == 0


def test_variant_views_reproduce_the_audit_threshold_counts():
    from etl import build_views as bv
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        _funnel_warehouse(directory)
        con, _ = connect(directory, 5, 5, 3, 2, max_downstream_km=30, max_upstream_km=10)
        whatif = fun.threshold_whatif(con, (3, 2)).set_index(["min_samples", "min_years"])
        con.close()
        for rule in [(3, 2), (3, 1)]:
            db = directory / f"w_{rule[1]}.duckdb"
            bv.build_database(db, directory, 5, 5, *rule, max_downstream_km=30, max_upstream_km=10)
            with duckdb.connect(str(db), read_only=True) as c:
                dams, down = c.sql("""SELECT count(DISTINCT AR_ID),
                                              count(DISTINCT AR_ID) FILTER (WHERE side = 'downstream')
                                       FROM eligibility_pairs WHERE has_both""").fetchone()
            assert (dams, down) == (whatif.loc[rule, "dams"], whatif.loc[rule, "dams_downstream"]), rule
    assert whatif.loc[(3, 1), "dams"] == whatif.loc[(3, 2), "dams"] + 1      # D7 (one pre year) is added


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
