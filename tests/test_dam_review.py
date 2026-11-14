"""Checks for the manual dam review: analysis/audits/export_dam_review.py
(review set, priorities, flowline extraction, GeoPackage layers) and
etl/apply_dam_review.py (corrections -> config/snap_overrides.csv), on the
synthetic project of tests/test_dam_matching.py.

Run directly: `python tests/test_dam_review.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import Point

import test_dam_matching as tdm
from analysis.audits import export_dam_review as edr
from etl import apply_dam_review as adr
from etl import match_dams as md
from etl.common import snapping as sn
from etl.common.config import PROJECT_ROOT_ENV_VAR


def test_nearest_km():
    km = edr.nearest_km([-70.0, -70.0], [45.0, 46.0], [-70.0], [45.0])
    assert km[0] < 1e-6 and abs(km[1] - 111.2) < 0.5
    assert np.isinf(edr.nearest_km([-70.0], [45.0], [], [])[0])


def test_assign_priority():
    frame = pd.DataFrame({
        "dated":       [True, True, True, True, True, True, False],
        "flagged":     [True, False, False, False, False, False, False],
        "baci":        [True, True, False, False, False, False, False],
        "in_analysis": [True, True, True, False, False, False, False],
        "match_status": ["accepted", "accepted", "accepted", "ambiguous", "accepted", "rejected",
                         "rejected"],
        "linked":      [True, True, True, False, True, False, False],
        "nearest_station_km": [1, 1, 1, 5, 5, 50, 1],
    })
    priority, rank = edr.assign_priority(frame)
    assert priority.iloc[:4].tolist() == ["flagged", "baci", "analysis", "recoverable"]
    assert priority.iloc[4:].isna().all()     # linked but no data; no station; undated
    assert rank.iloc[:4].tolist() == [1, 2, 3, 4]


def _project(root):
    """The dam_test project, matched, with a minimal warehouse and stations."""
    tdm._build_project(root)
    argv = ["match_dams.py", "--case-study", "dam_test"]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        md.main()
    warehouse = root / "data" / "warehouse" / "dam_test"
    warehouse.mkdir(parents=True)
    pd.DataFrame({
        "AR_ID": ["D_RES", "D_RES", "D_FREE"],
        "side": ["downstream", "upstream", "downstream"],
        "exposure_status": ["assigned_sole", "upstream_control", "assigned_sole"],
        "upstream_in_former_reservoir": [None, False, None],
    }).to_parquet(warehouse / "analysis_measurement.parquet")
    pd.DataFrame({"AR_ID": ["D_RES", "D_FREE"]}).to_parquet(warehouse / "dam_station_link.parquet")
    results = root / "data" / "raw" / "wqp" / "results" / "dam_test"
    results.mkdir(parents=True)
    pd.DataFrame({"MonitoringLocationIdentifier": ["S1"], "LatitudeMeasure": [45.02],
                  "LongitudeMeasure": [-69.95]}).to_csv(results / "pilot_station_selection.csv",
                                                        index=False)


def _export(root, extra=()):
    argv = ["export_dam_review.py", "--case-study", "dam_test", "--radius-km", "3", *extra]
    with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
            mock.patch.object(sys, "argv", argv):
        edr.main()


def test_export_review_geopackage():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _project(root)
        _export(root)
        gpkg = root / "data" / "processed" / "dam_review" / "dam_test" / "dam_review.gpkg"
        layers = set(pyogrio.list_layers(gpkg)[:, 0])
        assert layers == {"dams", "dams_current", "flowlines", "corrections"}

        dams = gpd.read_file(gpkg, layer="dams").set_index("AR_ID")
        assert dams.loc["D_RES", "priority"] == "baci" and dams.loc["D_FREE", "priority"] == "analysis"
        # Ambiguous / rejected dams with a station within 30 km can be recovered.
        for dam in ("D_NAME", "D_MULTI", "D_OVER"):
            assert dams.loc[dam, "priority"] == "recoverable", dam
        assert dams.loc["D_NAME", "current_reach"] == 11       # provisional reach
        assert dams.loc["D_RES", "current_reach"] == 11 and dams.crs.to_epsg() == 4326

        lines = gpd.read_file(gpkg, layer="flowlines").set_index("nhdplus_id")
        assert {10, 11, 12, 30} <= set(lines.index)
        assert lines.loc[11, "gnis_name"] == "Mill River" and lines.loc[11, "totdasqkm"] == 100.0
        assert len(gpd.read_file(gpkg, layer="corrections")) == 0
        assert len(gpd.read_file(gpkg, layer="dams_current")) == 2   # accepted dams
        assert (gpkg.parent / "dam_review_dams.csv").exists()

        # A requested subset goes to its own file.
        _export(root, ["--dams", "D_FREE"])
        extra = gpd.read_file(gpkg.parent / "dam_review_extra.gpkg", layer="dams")
        assert extra["AR_ID"].tolist() == ["D_FREE"]


def _corrections(gpkg, rows):
    frame = gpd.GeoDataFrame(
        {"AR_ID": [r[0] for r in rows], "action": [r[1] for r in rows],
         "nhdplus_id": pd.array([r[2] for r in rows], dtype="Int64"),
         "reason": [r[3] for r in rows], "source": ["imagery"] * len(rows)},
        geometry=[r[4] for r in rows], crs="EPSG:4326")
    pyogrio.write_dataframe(frame, gpkg, layer="corrections", driver="GPKG")


def _apply(gpkg, overrides, *extra):
    argv = ["apply_dam_review.py", "--review", str(gpkg), "--reviewer", "jc",
            "--reviewed-on", "2026-11-14", "--overrides", str(overrides), *extra]
    with mock.patch.object(sys, "argv", argv):
        adr.main()


def test_apply_corrections_writes_overrides():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _project(root)
        _export(root)
        gpkg = root / "data" / "processed" / "dam_review" / "dam_test" / "dam_review.gpkg"
        _corrections(gpkg, [
            ("D_FREE", "move", 10, "dam was on Mill River", Point(-69.99, 45.0)),
            ("D_MULTI", "reject", None, "channel not mapped in NHDPlus", Point(-69.92, 45.02)),
        ])
        overrides = root / "config" / "snap_overrides.csv"   # holds a D_OVER reject
        _apply(gpkg, overrides)

        frame = sn.load_overrides(overrides)
        free = frame.loc[("dam", "D_FREE")]
        assert free["action"] == "accept" and free["nhdplus_id"] == 10
        assert (free["corrected_lat"], free["corrected_lon"]) == (45.0, -69.99)
        assert free["reason"] == "dam was on Mill River [source: imagery]"
        assert frame.loc[("dam", "D_MULTI"), "action"] == "reject"
        assert ("dam", "D_OVER") in frame.index                     # other rows kept

        # The new overrides run through match_dams.
        argv = ["match_dams.py", "--case-study", "dam_test", "--output-dir", str(root / "out")]
        with mock.patch.dict(os.environ, {PROJECT_ROOT_ENV_VAR: str(root)}), \
                mock.patch.object(sys, "argv", argv):
            md.main()
        dams = pd.read_parquet(root / "out" / f"dam_reaches_{tdm.HUC8}.parquet").set_index("AR_ID")
        assert dams.loc["D_FREE", "NHDPlusID"] == 10
        assert dams.loc["D_FREE", "coordinates_source"] == "manual_correction"
        assert dams.loc["D_MULTI", "match_reason"] == "manual_reject"


def test_apply_rejects_bad_corrections_and_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _project(root)
        _export(root)
        gpkg = root / "data" / "processed" / "dam_review" / "dam_test" / "dam_review.gpkg"
        _corrections(gpkg, [
            ("D_FREE", "move", 10, "off the line", Point(-69.99, 45.005)),   # ~550 m away
            ("D_RES", "move", 999, "unknown reach", Point(-69.97, 45.0)),
            ("D_NAME", "move", 11, "", Point(-69.97, 45.0)),
            ("NOPE", "reject", None, "not a review dam", Point(-69.97, 45.0)),
            ("D_OVER", "relocate", None, "bad action", Point(-69.97, 45.0)),
            ("D_OVER", "reject", 30, "reject with a reach", Point(-69.97, 45.0)),
        ])
        overrides = root / "config" / "snap_overrides.csv"
        before = overrides.read_text()
        try:
            _apply(gpkg, overrides)
        except SystemExit as exc:
            assert exc.code == 1
        else:
            raise AssertionError("expected SystemExit")
        assert overrides.read_text() == before
        corrections, ids, lines = adr.read_review(gpkg)
        message = "\n".join(adr.check_corrections(corrections, ids, lines))
        for expected in ["m from flowline 10", "999 is not in the flowlines layer", "reason is empty",
                         "not in the review's dams layer", "action 'relocate'",
                         "reject must not give an nhdplus_id", "more than one correction"]:
            assert expected in message, expected


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
