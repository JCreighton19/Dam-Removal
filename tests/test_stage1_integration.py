"""End-to-end integration checks for the stage-1 scripts' I/O paths
(config loading, file validation, atomic writes, manifests) against
synthetic fixtures -- not just their pure logic functions.

These build a throwaway project root under a temp directory with a
minimal WBD GeoPackage, dam CSV, station CSVs, and (for the eligibility
report) link/results CSVs, then run each script's real `main()`.

Run directly: `python tests/test_stage1_integration.py`
(Slower than the other test_*.py files since it does real file I/O.)
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from etl.common.config import PROJECT_ROOT_ENV_VAR


def _build_project(root: Path, huc8="19020401"):
    (root / "config").mkdir(parents=True)
    (root / "config" / "pilot_test.yaml").write_text(
        f"""
case_study: pilot_test
huc8s: ["{huc8}"]
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
station_types: ["Stream", "River/Stream"]
candidate_distance_km: 5
candidate_distance_sensitivity_km: [1, 5, 50]
wq_window:
  years_before: 3
  years_after: 3
wq_characteristics: ["Turbidity", "pH"]
"""
    )

    wbd_dir = root / "data" / "raw" / "wbd"
    wbd_dir.mkdir(parents=True)
    watershed = gpd.GeoDataFrame(
        {"huc8": [huc8]},
        geometry=[box(-71.3, 41.8, -70.7, 42.7)],
        crs="EPSG:4326",
    )
    watershed.to_file(wbd_dir / "wbd.gpkg", layer="WBDHU8", driver="GPKG")

    dams_dir = root / "data" / "raw" / "dams"
    dams_dir.mkdir(parents=True)
    pd.DataFrame({
        "AR_ID": [1, 2],
        "Dam_Name": ["Dam One", "Dam Two"],
        "Latitude": [42.0, 42.5],
        "Longitude": [-71.0, -71.0],
        "Year_Removed": [2017, 2015],
    }).to_csv(dams_dir / "dams.csv", index=False)

    stations_dir = root / "data" / "raw" / "wqp" / "stations"
    stations_dir.mkdir(parents=True)
    pd.DataFrame({
        "MonitoringLocationIdentifier": ["S1", "S2", "S3"],
        "MonitoringLocationName": ["Near Dam One", "Near Dam Two", "Far from both"],
        "MonitoringLocationTypeName": ["Stream", "River/Stream", "Stream"],
        "LatitudeMeasure": [42.01, 42.49, 90.0],  # S3 has an invalid latitude
        "LongitudeMeasure": [-71.0, -71.0, -71.0],
    }).to_csv(stations_dir / f"{huc8}.csv", index=False)

    return root


def test_select_candidate_stations_end_to_end(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_project(Path(tmp))
        monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(root))
        monkeypatch.setattr(
            sys, "argv", ["select_candidate_stations.py", "--case-study", "pilot_test"]
        )

        # Re-import so the module picks up the patched env var fresh.
        import importlib
        import etl.select_candidate_stations as mod
        importlib.reload(mod)
        mod.main()

        output_path = root / "data" / "raw" / "wqp" / "results" / "pilot_station_selection.csv"
        sensitivity_path = output_path.with_name("pilot_station_selection_sensitivity.csv")
        manifest_path = output_path.with_name("pilot_station_selection_manifest.csv")

        assert output_path.exists()
        assert sensitivity_path.exists()
        assert manifest_path.exists()

        candidates = pd.read_csv(output_path)
        # S3 has an invalid latitude and must be dropped before distance
        # computation; S1/S2 are each ~1.1 km from a dam, within 5 km.
        assert set(candidates["MonitoringLocationIdentifier"]) == {"S1", "S2"}
        assert "nearest_dam_id" in candidates.columns
        assert "distance_to_nearest_dam_km" in candidates.columns

        manifest = pd.read_csv(manifest_path)
        assert manifest.loc[0, "status"] == "ok"
        assert manifest.loc[0, "candidate_station_count"] == 2


def test_select_candidate_stations_does_not_overwrite_on_missing_dam_file(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_project(Path(tmp))

        # Seed an existing "good" candidate file, then break the dam source
        # so the run fails -- the existing file must survive untouched.
        output_path = root / "data" / "raw" / "wqp" / "results" / "pilot_station_selection.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("MonitoringLocationIdentifier\nPRE_EXISTING\n")

        (root / "data" / "raw" / "dams" / "dams.csv").unlink()

        monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(root))
        monkeypatch.setattr(
            sys, "argv", ["select_candidate_stations.py", "--case-study", "pilot_test"]
        )

        import importlib
        import etl.select_candidate_stations as mod
        importlib.reload(mod)

        try:
            mod.main()
            raise AssertionError("Expected the missing dam file to raise.")
        except FileNotFoundError:
            pass

        # The pre-existing candidate file must be untouched.
        assert output_path.read_text() == "MonitoringLocationIdentifier\nPRE_EXISTING\n"

        manifest_path = output_path.with_name("pilot_station_selection_manifest.csv")
        manifest = pd.read_csv(manifest_path)
        assert manifest.loc[0, "status"] == "failed"


def test_report_pilot_eligibility_detects_missing_input(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_project(Path(tmp))
        # Deliberately do not create dam_station_links_<huc8>.csv.

        monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(root))
        monkeypatch.setattr(
            sys, "argv",
            ["report_pilot_eligibility.py", "--case-study", "pilot_test", "--huc8", "19020401"],
        )

        import importlib
        import etl.report_pilot_eligibility as mod
        importlib.reload(mod)

        try:
            mod.main()
            raise AssertionError("Expected a FileNotFoundError for the missing link table.")
        except FileNotFoundError:
            pass

        manifest_path = root / "data" / "processed" / "pilot_eligibility_manifest_19020401.csv"
        manifest = pd.read_csv(manifest_path)
        assert manifest.loc[0, "status"] == "missing_input"
        # The manifest must not claim "0 eligible" -- those fields are left blank, not 0.
        assert pd.isna(manifest.loc[0, "pairs_eligible_both"])


def test_report_pilot_eligibility_end_to_end_ok_run(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = _build_project(Path(tmp))

        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "dam_id": ["1", "1"],
            "dam_name": ["Dam One", "Dam One"],
            "station_id": ["S1", "S2"],
            "year_removed": [2017, 2017],
            "relationship": ["upstream", "downstream"],
            "distance_to_dam_km": [1.0, 2.0],
        }).to_csv(processed_dir / "dam_station_links_19020401.csv", index=False)

        results_dir = root / "data" / "raw" / "wqp" / "results" / "pilot_download"
        results_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "MonitoringLocationIdentifier": ["S1", "S2"],
            "CharacteristicName": ["Turbidity", "Turbidity"],
            "ActivityStartDate": ["2015-06-01", "2019-06-01"],
        }).to_csv(results_dir / "results_batch_001.csv", index=False)

        monkeypatch.setenv(PROJECT_ROOT_ENV_VAR, str(root))
        monkeypatch.setattr(
            sys, "argv",
            ["report_pilot_eligibility.py", "--case-study", "pilot_test", "--huc8", "19020401"],
        )

        import importlib
        import etl.report_pilot_eligibility as mod
        importlib.reload(mod)
        mod.main()

        manifest = pd.read_csv(processed_dir / "pilot_eligibility_manifest_19020401.csv")
        assert manifest.loc[0, "status"] == "ok"
        assert manifest.loc[0, "dams_eligible_pre"] == 1
        assert manifest.loc[0, "dams_eligible_post"] == 1
        # Same "different stations don't count as both" rule, verified
        # through the real file-based entry point this time.
        assert manifest.loc[0, "dams_eligible_both"] == 0
        assert manifest.loc[0, "pairs_eligible_both"] == 0


# --- Minimal monkeypatch shim, so this file runs without pytest installed ---

class _FakeMonkeypatch:
    def __init__(self):
        self._restores = []

    def setenv(self, key, value):
        import os
        old = os.environ.get(key)
        self._restores.append(("env", key, old))
        os.environ[key] = value

    def setattr(self, obj, name, value):
        old = getattr(obj, name)
        self._restores.append(("attr", obj, name, old))
        setattr(obj, name, value)

    def undo(self):
        import os
        for entry in reversed(self._restores):
            if entry[0] == "env":
                _, key, old = entry
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old
            else:
                _, obj, name, old = entry
                setattr(obj, name, old)


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        mp = _FakeMonkeypatch()
        try:
            test(mp)
        finally:
            mp.undo()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
