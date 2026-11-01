"""Checks for etl/common/reach_links.py and etl/link_reaches.py on a
synthetic network around 45 N (every reach 1 km, digitized downstream):

    reach 4 (tributary) --\\
    reach 1 ---> reach 2 ---> reach 3        reach 5 (separate network)

Dams:  D1 accepted on 2 at 0.5 (its reservoir's path is 1)
       D2 ambiguous, provisional reach 3, reported point at its middle
Stations:
    S_UP    accepted on 1 at 0.5      upstream of D1, 1.0 km, in reservoir
    S_DN    accepted on 3 at 0.25     downstream of D1, 0.75 km
    S_SAME  accepted on 2 at 0.8      downstream of D1 on its reach, 0.3 km
    S_TRIB  ambiguous (confluence), provisional 4   upstream of D1, not usable
    S_ISO   accepted on 5             not connected
    S_REJ   rejected

Run directly: `python tests/test_reach_links.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
from shapely.geometry import LineString

from etl import link_reaches as lr
from etl.common import reach_links as rl
from etl.common.config import PROJECT_ROOT_ENV_VAR
from etl.common.network import build_reach_graph

HUC8 = "01010001"
CRS = "EPSG:4269"
METRIC = "EPSG:26919"

LINES = {
    1: LineString([(-70.00, 45.0), (-69.99, 45.0)]),
    2: LineString([(-69.99, 45.0), (-69.98, 45.0)]),
    3: LineString([(-69.98, 45.0), (-69.97, 45.0)]),
    4: LineString([(-69.99, 45.01), (-69.99, 45.0)]),
    5: LineString([(-69.95, 45.02), (-69.94, 45.02)]),
}
FLOW = [(1, 2), (2, 3), (4, 2), (3, 0), (5, 0)]

# id: (status, reason, reach, fraction, provisional, lon, lat)
STATIONS = {
    "S_UP": ("accepted", "name_match", 1, 0.5, 1, -69.995, 45.0001),
    "S_DN": ("accepted", "own_catchment", 3, 0.25, 3, -69.9775, 45.0001),
    "S_SAME": ("accepted", "name_match", 2, 0.8, 2, -69.982, 45.0001),
    "S_TRIB": ("ambiguous", "confluence_ambiguous", None, None, 4, -69.9901, 45.005),
    "S_ISO": ("accepted", "name_match", 5, 0.5, 5, -69.945, 45.0201),
    "S_REJ": ("rejected", "no_eligible_candidate", None, None, None, -69.96, 45.03),
}
# id: (status, reason, reach, fraction, provisional, lon, lat, impounded)
DAMS = {
    "1001": ("accepted", "reservoir_outlet", 2, 0.5, 2, -69.985, 45.0001, "1"),
    "1002": ("ambiguous", "reservoir_name_mismatch", None, None, 3, -69.975, 45.0001, None),
}
OLD = [  # dam_id, station_id, relationship, km
    ("1001", "S_UP", "upstream", 1.0),
    ("1001", "S_DN", "downstream", 0.9),
    ("1001", "S_TRIB", "upstream", 1.0),
    ("1001", "S_ISO", "downstream", 2.0),
    ("1001", "S_REJ", "downstream", 1.0),
    ("1001", "S_SAME", "unconnected", np.nan),
    ("1001", "S_SAME", "unconnected", np.nan),
]


def _metric(lon, lat):
    return Transformer.from_crs(CRS, METRIC, always_xy=True).transform(lon, lat)


def _graph():
    streams = pd.DataFrame({"NHDPlusID": list(LINES), "LengthKM": [1.0] * len(LINES)})
    flow = pd.DataFrame(FLOW, columns=["FromNHDPID", "ToNHDPID"])
    graph, lengths, _ = build_reach_graph(streams, flow)
    return graph, lengths


def _ends(rows):
    return pd.DataFrame([{"feature_id": k, "status": v[0], "reason": v[1],
                          "reach_id": v[2] if v[0] == "accepted" else v[4],
                          "fraction": v[3] if v[0] == "accepted" else 0.5}
                         for k, v in rows.items() if v[0] != "rejected"]).astype(
        {"reach_id": "Int64"})


def test_link_pairs_relationships_and_distances():
    graph, lengths = _graph()
    links = rl.link_pairs(_ends(DAMS), _ends(STATIONS), graph, lengths,
                          impounded={"1001": {1}}).set_index(["dam_id", "station_id"])
    d1 = links.loc["1001"]
    assert set(d1.index) == {"S_UP", "S_DN", "S_SAME", "S_TRIB"}   # S_ISO unconnected
    assert d1.loc["S_UP", "relationship"] == "upstream"
    assert np.isclose(d1.loc["S_UP", "network_distance_km"], 1.0)
    assert np.isclose(d1.loc["S_DN", "network_distance_km"], 0.75)
    assert d1.loc["S_SAME", "relationship"] == "downstream"
    assert np.isclose(d1.loc["S_SAME", "network_distance_km"], 0.3)
    assert d1.loc["S_SAME", "distance_method"] == "same_reach_fraction"
    assert d1.loc["S_UP", "station_in_mapped_reservoir"]
    assert not d1.loc["S_DN", "station_in_mapped_reservoir"]
    assert d1.loc["S_UP", "usable"] and not d1.loc["S_TRIB", "usable"]
    # The ambiguous dam is linked from its provisional reach, never usable.
    d2 = links.loc["1002"]
    assert d2.loc["S_DN", "relationship"] == "upstream"
    assert np.isclose(d2.loc["S_DN", "network_distance_km"], 0.25)
    assert not d2["usable"].any()


def test_link_pairs_max_km_drops_far_pairs():
    graph, lengths = _graph()
    links = rl.link_pairs(_ends(DAMS), _ends(STATIONS), graph, lengths, max_km=0.8)
    near = set(links.loc[links["dam_id"] == "1001", "station_id"])
    assert near == {"S_DN", "S_SAME"}


def test_link_pairs_matches_old_shortest_path_distance():
    # Same answer as link_dams_stations.calculate_network_distance.
    from etl.link_dams_stations import calculate_network_distance
    graph, lengths = _graph()
    links = rl.link_pairs(_ends(DAMS), _ends(STATIONS), graph, lengths).set_index(
        ["dam_id", "station_id"])
    for station, rel in (("S_UP", "upstream"), ("S_DN", "downstream"), ("S_TRIB", "upstream")):
        row = links.loc[("1001", station)]
        old, _ = calculate_network_distance(graph, 2, int(row["station_reach_id"]), 0.5,
                                            row["station_fraction"], lengths, rel)
        assert np.isclose(old, row["network_distance_km"]), station


def test_in_window():
    rel = pd.Series(["downstream", "upstream", "same_reach", "downstream", "unconnected"])
    km = pd.Series([5.0, 12.0, 0.0, np.nan, 1.0])
    limited = rl.in_window(rel, km, {"max_downstream_km": 10, "max_upstream_km": 10})
    assert limited.tolist() == [True, False, False, False, False]
    open_ = rl.in_window(rel, km, {"max_downstream_km": None, "max_upstream_km": None})
    assert open_.tolist() == [True, True, True, True, False]


def test_compare_links_out_of_window_and_blockers():
    stations = pd.DataFrame({"feature_id": ["A", "B"], "status": ["accepted", "ambiguous"],
                             "reason": ["name_match", "confluence_ambiguous"],
                             "reach_id": pd.array([1, 2], dtype="Int64"), "fraction": [0.5, 0.5]})
    dams = pd.DataFrame({"feature_id": ["D"], "status": ["accepted"], "reason": ["name_match"],
                         "reach_id": pd.array([3], dtype="Int64"), "fraction": [0.5]})
    old = pd.DataFrame({"dam_id": ["D", "D"], "station_id": ["A", "B"],
                        "relationship": ["downstream", "downstream"],
                        "network_distance_km": [5.0, 5.0]})
    new = pd.DataFrame({"dam_id": ["D", "D"], "station_id": ["A", "B"],
                        "relationship": ["downstream", "downstream"],
                        "network_distance_km": [15.0, 5.0], "usable": [True, False],
                        "station_in_mapped_reservoir": [False, False]})
    out = rl.compare_links(old, new, dams, stations,
                           {"max_downstream_km": 10, "max_upstream_km": 10}).set_index("station_id")
    assert out.loc["A", "change"] == "now_out_of_window" and pd.isna(out.loc["A", "blocker"])
    assert out.loc["B", "change"] == "now_blocked"
    assert out.loc["B", "blocker"] == "station:ambiguous:confluence_ambiguous"


def test_provisional_fractions_and_digitization():
    lines = gpd.GeoSeries(LINES, crs=CRS).to_crs(METRIC)
    lines = gpd.GeoDataFrame(geometry=lines)
    x, y = _metric(-69.975, 45.0)
    frame = pd.DataFrame({"provisional_nhdplus_id": pd.array([3, None, 99], dtype="Int64"),
                          "x_m": [x, x, x], "y_m": [y, y, y]})
    fractions = rl.provisional_fractions(frame, lines)
    assert np.isclose(fractions.iat[0], 0.5, atol=0.01)
    assert fractions.iloc[1:].isna().all()          # no reach / reach not loaded

    edges = pd.DataFrame({"from": [1, 2, 4, 3], "to": [2, 3, 2, None]}).astype("Int64")
    flipped = lines.copy()
    flipped.loc[4, "geometry"] = LineString(list(lines.geometry.loc[4].coords)[::-1])
    assert rl.digitized_downstream(lines, edges, [1, 2, 3, 4]).to_dict() == \
        {1: True, 2: True, 4: True}                 # 3 has nothing below it
    assert not rl.digitized_downstream(flipped, edges, [4]).iat[0]


def _build_project(root):
    (root / "config").mkdir()
    (root / "config" / "links_test.yaml").write_text(
        f'case_study: links_test\nhuc8s: ["{HUC8}"]\ndam_source: d.csv\nwbd_source: w.gpkg\n'
        "exposure:\n  max_downstream_km: 10\n  max_upstream_km: 10\n")
    processed = root / "data" / "processed"
    package = f"NHDPLUS_H_{HUC8}_HU8_GDB"
    gpkg = processed / "nhdplus_extracted" / package / f"{package}.gpkg"
    gpkg.parent.mkdir(parents=True)
    ids = [float(i) for i in LINES]
    gpd.GeoDataFrame({"nhdplusid": ids, "lengthkm": [1.0] * len(ids)},
                     geometry=list(LINES.values()), crs=CRS).to_file(
        gpkg, layer="NHDFlowline", driver="GPKG")
    pyogrio.write_dataframe(pd.DataFrame({"fromnhdpid": [float(a) for a, _ in FLOW],
                                          "tonhdpid": [float(b) for _, b in FLOW]}),
                            gpkg, layer="NHDPlusFlow", driver="GPKG")

    station_dir = processed / "station_reaches" / "links_test"
    station_dir.mkdir(parents=True)
    rows = []
    for sid, (status, reason, reach, fraction, provisional, lon, lat) in STATIONS.items():
        x, y = _metric(lon, lat)
        rows.append({"huc8": HUC8, "MonitoringLocationIdentifier": sid,
                     "LongitudeMeasure": lon, "LatitudeMeasure": lat, "metric_crs": METRIC,
                     "x_m": x, "y_m": y, "snap_status": status, "snap_reason": reason,
                     "NHDPlusID": reach, "snap_fraction": fraction,
                     "provisional_nhdplus_id": provisional})
    pd.DataFrame(rows).astype({"NHDPlusID": "Int64", "provisional_nhdplus_id": "Int64"}) \
        .to_parquet(station_dir / f"station_reaches_{HUC8}.parquet", index=False)

    dam_dir = processed / "dam_reaches" / "links_test"
    dam_dir.mkdir(parents=True)
    rows = []
    for did, (status, reason, reach, fraction, provisional, lon, lat, imp) in DAMS.items():
        x, y = _metric(lon, lat)
        rows.append({"huc8": HUC8, "AR_ID": int(did), "Longitude": lon, "Latitude": lat,
                     "metric_crs": METRIC, "x_m": x, "y_m": y, "match_status": status,
                     "match_reason": reason, "NHDPlusID": reach, "snap_fraction": fraction,
                     "provisional_nhdplus_id": provisional, "impounded_nhdplus_ids": imp})
    pd.DataFrame(rows).astype({"NHDPlusID": "Int64", "provisional_nhdplus_id": "Int64"}) \
        .to_parquet(dam_dir / f"dam_reaches_{HUC8}.parquet", index=False)

    pd.DataFrame(OLD, columns=["dam_id", "station_id", "relationship", "network_distance_km"]) \
        .to_csv(processed / f"dam_station_links_{HUC8}.csv", index=False)


def test_link_reaches_end_to_end():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _build_project(root)
        out_dir = root / "out"
        argv = ["link_reaches.py", "--case-study", "links_test", "--output-dir", str(out_dir)]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            lr.main()

        links = pd.read_parquet(out_dir / f"reach_links_{HUC8}.parquet") \
            .set_index(["dam_id", "station_id"])
        d1 = links.loc["1001"]
        assert d1.loc["S_UP", "usable"] and d1.loc["S_UP", "station_in_mapped_reservoir"]
        assert np.isclose(d1.loc["S_DN", "network_distance_km"], 0.75)
        # S_TRIB's provisional fraction comes from projecting its point onto reach 4.
        assert d1.loc["S_TRIB", "relationship"] == "upstream"
        assert 0.4 < d1.loc["S_TRIB", "station_fraction"] < 0.6
        # D2's fraction too (point at the middle of reach 3).
        assert np.isclose(links.loc[("1002", "S_DN"), "network_distance_km"], 0.25, atol=0.02)

        cmp = pd.read_parquet(out_dir / f"link_comparison_{HUC8}.parquet") \
            .set_index(["dam_id", "station_id"])
        change = cmp["change"].to_dict()
        assert change[("1001", "S_UP")] == "kept"
        assert change[("1001", "S_DN")] == "kept"
        assert np.isclose(cmp.loc[("1001", "S_DN"), "distance_change_km"], -0.15)
        assert change[("1001", "S_SAME")] == "gained"
        assert change[("1001", "S_ISO")] == "now_unconnected"
        assert change[("1001", "S_TRIB")] == "now_blocked"
        assert cmp.loc[("1001", "S_TRIB"), "blocker"] == "station:ambiguous:confluence_ambiguous"
        assert cmp.loc[("1001", "S_REJ"), "blocker"] == "station:rejected:no_eligible_candidate"
        assert change[("1002", "S_DN")] == "blocked"
        assert cmp.loc[("1002", "S_DN"), "blocker"] == "dam:ambiguous:reservoir_name_mismatch"

        summary = pd.read_csv(out_dir / "summary.csv").iloc[0]
        assert summary["usable"] == 3 and summary["kept"] == 2
        assert summary["digitized_downstream"] == summary["digitized_checked"] > 0


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
