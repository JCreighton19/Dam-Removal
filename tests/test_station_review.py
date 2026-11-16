"""Checks for the station QA audits on small synthetic frames (no warehouse):
analysis/audits/export_station_review.py (alternatives, risk tiers, stratified
sample) and analysis/audits/station_recovery_audit.py (summary mapping,
estimated pairs, gains, per-parameter table, priority).

Run directly: `python tests/test_station_review.py`
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from analysis.audits import export_station_review as esr
from analysis.audits import station_recovery_audit as sra

ID = "MonitoringLocationIdentifier"


def _candidates():
    return pd.DataFrame({
        ID: ["a", "a", "a", "b", "b"],
        "NHDPlusID": [1, 2, 3, 10, 11],
        "distance_m": [10.0, 20.0, 400.0, 5.0, 300.0],
        "FType": [460, 460, 336, 558, 460],          # 336 canal: not eligible
        "GNIS_Name": ["Main R", "Side Cr", "Ditch", "Lake", "Main R"],
        "totdasqkm": [500.0, 20.0, 1.0, 900.0, 800.0],
        "area_ratio": [1.0, 25.0, None, None, None],
        "name_match": [True, False, None, None, None],
    })


def test_alternatives():
    alts = esr.alternatives(_candidates(), pd.Series({"a": 1, "b": 10})).set_index("station_id")
    assert alts.loc["a", "alt1_id"] == 2 and alts.loc["a", "n_eligible_alternatives"] == 1
    assert alts.loc["a", "margin_m"] == 10.0
    assert alts.loc["b", "alt1_id"] == 11 and alts.loc["b", "chosen_distance_m"] == 5.0


def _station(**kw):
    base = {"snap_status": "accepted", "snap_evidence": "area_agrees;name_match",
            "match_method": "evidence_rules", "snap_distance_m": 10.0, "margin_m": 200.0,
            "chosen_area_km2": 500.0, "alt1_area_km2": 480.0, "alt1_name": "Main R",
            "reach_gnis_name": "Main R"}
    base.update(kw)
    return base


def test_risk_tiers():
    frame = pd.DataFrame([
        _station(),                                                     # low
        _station(snap_evidence="own_catchment"),                        # medium: no support
        _station(snap_evidence="name_match;confluence_nearby"),         # high: conflict, name only
        _station(snap_evidence="area_agrees;confluence_nearby"),        # medium: area resolves it
        _station(snap_status="ambiguous"),                              # medium: robust, same role
        _station(snap_distance_m=150.0),                                # high: far, not in polygon
        _station(snap_distance_m=150.0, point_polygon_layer="NHDArea"),  # medium: far, in polygon
        _station(margin_m=10.0, alt1_area_km2=20.0,
                 snap_evidence="name_match"),                           # high: close different rival
        _station(margin_m=10.0, alt1_area_km2=20.0),                    # medium: area agrees
        _station(margin_m=10.0),                                        # medium: close rival
        _station(match_method="manual_override", snap_status="ambiguous"),  # manual
    ])
    tier, flags = esr.risk(frame)
    assert tier.tolist() == ["low", "medium", "high", "medium", "medium", "high", "medium",
                             "high", "medium", "medium", "manual"], tier.tolist()
    assert "close_different_rival" in flags[7] and flags[0] == ""
    assert esr.support(frame).tolist()[:4] == ["area+name", "none", "name", "area"]


def test_strata_ftype_format():
    frame = pd.DataFrame({"risk_tier": ["low"] * 4, "baci_station": [False] * 4,
                          "role": ["treated"] * 4, "reach_ftype": [460, 460.0, "460", None],
                          "reach_totdasqkm": [50.0] * 4})
    design = esr.stratified_sample(frame, {"low": 1.0}, minimum=1, seed=1)[3]
    assert design["stratum"].tolist() == ["low | treated | ftype 460 | <100 km2",
                                          "low | treated | ftype ? | <100 km2"]


def test_stratified_sample():
    n = 50
    frame = pd.DataFrame({
        "risk_tier": ["low"] * 36 + ["high"] * 4 + ["medium"] * 10,
        "baci_station": [False] * 30 + [True] * 6 + [False] * 14,
        "role": ["treated"] * 20 + ["control"] * 20 + ["treated"] * 10,
        "reach_ftype": [460] * n,
        "reach_totdasqkm": [50.0] * 10 + [500.0] * 40,
    })
    fractions = {"medium": 0.5, "low": 0.2}
    sampled, weight, stratum, design = esr.stratified_sample(frame, fractions, minimum=3, seed=1)
    pool = frame["risk_tier"].isin(["low", "medium"]) & ~frame["baci_station"]
    assert not sampled[~pool].any()                       # BACI and high are inspected, not sampled
    assert design["stations"].sum() == int(pool.sum())
    assert set(design["tier"]) == {"low", "medium"}
    for row in design.itertuples():                       # max(3, ceil(f n)), capped at n
        f = fractions[row.tier]
        assert row.sampled == min(row.stations, max(3, int(np.ceil(f * row.stations))))
        assert abs(row.weight - row.stations / row.sampled) < 1e-9
    # Weighted sample size reproduces the pool size.
    assert abs(weight[sampled].sum() - pool.sum()) < 1e-9
    again = esr.stratified_sample(frame, fractions, minimum=3, seed=1)[0]
    assert sampled.equals(again)                          # fixed seed: reproducible


def test_summary_mapping_and_pairs():
    raw = pd.DataFrame({
        "MonitoringLocationIdentifier": ["s1"] * 6 + ["s2"],
        "YearSummarized": ["2006", "2008", "2011", "2013", "2010", "2008", "2008"],
        "CharacteristicName": ["pH"] * 4 + ["pH", "Optical density", "pH"],
        "ResultCount": ["2", "2", "2", "2", "9", "50", "1"],
        "ActivityCount": ["2", "2", "2", "2", "9", "50", "1"],
    })
    with tempfile.TemporaryDirectory() as folder:
        raw.to_csv(Path(folder) / "summary_0001.csv", index=False)
        rows = sra.load_summary(folder)
    assert set(rows["parameter"]) == {"ph"}                # unmapped characteristic dropped
    links = pd.DataFrame({"dam_id": ["D", "D"], "station_id": ["s1", "s2"],
                          "role": ["upstream_control", "downstream"], "removal_year": [2010, 2010]})
    pairs = sra.estimated_pairs(rows, links, 5, 5, 3, 2).set_index("station_id")
    s1 = pairs.loc["s1"]                                   # 2010 = removal year, left out
    assert (s1["n_pre"], s1["years_pre"], s1["n_post"], s1["years_post"]) == (4, 2, 4, 2)
    assert s1["qualifies"] and not pairs.loc["s2", "qualifies"]


def test_gains_parameters_priority():
    measures = pd.DataFrame({
        "dam_id": ["BA", "BA", "BACI", "BACI"],
        "parameter": ["ph", "turbidity", "ph", "water_temp"],
        "has_ba": [True, True, True, True], "has_control": [False, False, True, False],
        "rows": [1, 1, 1, 1]})
    dams = pd.DataFrame({"ba_dam": [True, True, False], "baci_dam": [False, True, False]},
                        index=pd.Index(["BA", "BACI", "NEW"], name="dam_id"))
    pairs = pd.DataFrame({
        "dam_id": ["BA", "BACI", "NEW", "BA", "BACI", "BA"],
        "station_id": ["c1", "c2", "d1", "d2", "c3", "c4"],
        "role": ["upstream_control", "upstream_control", "downstream", "downstream",
                 "upstream_control", "upstream_control"],
        "parameter": ["ph", "water_temp", "ph", "water_temp", "ph", "nitrate"],
        "qualifies": [True, True, True, True, True, True],
        "n_pre": [5] * 6, "n_post": [5] * 6})
    gains = sra.pair_gains(pairs, measures, dams).tolist()
    assert gains == ["new_baci_dam", "new_baci_measure", "new_ba_dam", "new_ba_measure",
                     "more_samples", None]             # nitrate control: no downstream nitrate
    pairs["gain"] = gains
    gained = pairs[pairs["gain"].notna()]
    params = sra.per_parameter(gained, measures).set_index("parameter")
    assert params.loc["ph", "baci_dams"] == 1 and params.loc["ph", "baci_dams_gained"] == 1
    assert params.loc["water_temp", "ba_dams_gained"] == 1
    stations = pd.DataFrame({"station_id": ["c1", "c2", "d1", "d2"], "snap_reason": ["x"] * 4})
    pri = sra.station_priority(gained, stations, confirmed={"d1"})
    assert pri["station_id"].tolist()[0] == "c1"
    assert pri.set_index("station_id").loc["d1", "confirmed_by_download"]
    assert "c3" not in set(pri["station_id"])          # more_samples only: not an eligibility gain


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
