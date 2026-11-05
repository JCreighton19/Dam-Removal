"""Checks for analysis/stage4_report.py: real model runs on the simulated dams
of tests/test_distance_models.py, written the way the Slurm job writes them,
then the report built from them.

Run directly: `python tests/test_stage4_report.py`
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import test_distance_models as sim
from analysis import distance_models as dm
from analysis import influence as inf
from analysis import stage4_report as rep


def test_classify():
    assert rep.classify(0.6, 0.2, 1.0, 0.5) == "detected"
    assert rep.classify(0.05, -0.1, 0.2, 0.5) == "no meaningful change"   # small and precise
    assert rep.classify(0.05, -0.8, 0.9, 0.5) == "inconclusive"           # uncertain
    assert rep.classify(0.05, -0.1, 0.6, 0.5) == "inconclusive"           # CI not entirely inside
    assert rep.classify(0.0, -0.5, 0.5, 0.5) == "no meaningful change"    # bounds inclusive
    assert rep.classify(0.05, -0.1, 0.2, None) == "inconclusive"
    assert rep.classify(np.nan, 0, 1, 0.5) == ""


def _run(model_dir, source, run, re_mode, *extra):
    argv = ["distance_models.py", "--case-study", "test", "--input", str(source),
            "--output-dir", str(model_dir / "stage3" / f"{run}__{re_mode}"),
            "--run-name", f"{run}__{re_mode}", "--re", re_mode, "--series", "water_temp",
            "--exclude-dams", str(model_dir / "none.csv"), *extra]
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}):
        dm.main()


def build_fixture(model_dir):
    source = model_dir / "analysis_measurement.parquet"
    raw = sim.simulate(raw=True)
    raw.loc[raw["AR_ID"].isin(["D01", "D02"]), "backwater_km"] = np.nan
    raw.to_parquet(source, index=False)
    _run(model_dir, source, "primary", "auto")
    _run(model_dir, source, "primary", "intercepts")
    _run(model_dir, source, "dar_le_10", "auto", "--specs", "baci_distance",
         "before_after_distance", "--max-dar", "10")
    _run(model_dir, source, "common_sample", "intercepts", "--common-sample", "--ml")
    argv = ["influence.py", "--case-study", "test", "--input", str(source), "--series",
            "water_temp", "--bootstrap", "10", "--workers", "1", "--re", "auto",
            "--components-series", "water_temp", "--exclude-dams", str(model_dir / "none.csv"),
            "--output-dir", str(model_dir / "stage4" / "influence")]
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}):
        inf.main()


def test_report_tables_and_figures():
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        build_fixture(model_dir)
        (model_dir / "stage3" / "old_run").mkdir()        # pre-Stage 4 folder: ignored
        out = model_dir / "stage4" / "report"
        tables = rep.build(model_dir, out)
        for name in ("model_summary", "effects_by_distance", "effects_curve",
                     "sensitivity_table", "distance_comparison", "influence_summary",
                     "sesoi_thresholds", "sesoi_sensitivity"):
            assert (out / f"{name}.csv").exists(), name
        assert (out / "figures" / "distance_response_water_temp.png").exists()
        assert (out / "figures" / "sensitivity_baci.png").exists()
        import matplotlib.pyplot as plt
        axes = rep.plot_distance_response(tables["effects_curve"], tables["effects_by_distance"],
                                          "water_temp", sesoi=0.5)
        assert "exploratory" not in axes[0].get_title()           # 24 dams: not flagged
        plt.close("all")

        summary = tables["model_summary"]
        assert set(summary["run"]) == {"primary", "dar_le_10", "common_sample"}
        primary = summary[(summary["run"] == "primary") & (summary["re"] == "auto")
                          & (summary["spec"] == "baci_distance")].iloc[0]
        assert primary["effect_class"] == "detected" and primary["slope_class"] == "detected"
        assert primary["mde80_effect"] > 0 and primary["re_choice"].startswith("intercepts")

        # The curve passes through the reported effect at 5 km.
        curve = tables["effects_curve"]
        at5 = curve[(curve["run"] == "primary") & (curve["re"] == "auto")
                    & (curve["spec"] == "baci_distance") & np.isclose(curve["distance_km"], 5)]
        assert abs(at5["estimate"].iloc[0] - primary["effect_at_reference"]) < 1e-9

        comparison = tables["distance_comparison"].set_index("design")
        assert comparison.loc["baci", "same_sample"] and comparison.loc["baci", "dams"] == 22
        assert comparison.loc["baci", "same_covariates"] and comparison.loc["baci", "both_ml"]
        assert comparison.loc["baci", "same_random_effects"]

        thresholds = tables["sesoi_thresholds"].set_index("series")
        assert thresholds.loc["turbidity", "threshold"] == "+/-25% change"
        assert thresholds.loc["water_temp", "threshold"] == "+/-0.5 deg C"
        assert thresholds["basis"].str.contains("not an environmental").all()
        sens = tables["sesoi_sensitivity"]
        assert {f"class_at_{m:g}x" for m in rep.SESOI_MULTIPLIERS} <= set(sens.columns)
        assert not sens["class_changes"].any()          # simulated effects are large: all detected
        assert comparison.loc["baci", "preferred"] in {"raw distance", "scaled distance",
                                                       "no clear difference"}

        sens = tables["sensitivity_table"]
        assert {"primary", "primary (intercepts)", "dar_le_10"} <= set(sens["model_run"])
        assert tables["influence_summary"].iloc[0]["refits"] == 24


def test_results_notebook_runs_on_the_outputs():
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        print("  (nbclient not installed: notebook check skipped)")
        return
    notebook = Path(__file__).resolve().parents[1] / "notebooks" / "analysis" / \
        "distance_model_results.ipynb"
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        build_fixture(model_dir)
        nb = nbformat.read(notebook, as_version=4)
        with mock.patch.dict(os.environ, {"DAM_REMOVAL_MODEL_DIR": str(model_dir)}):
            NotebookClient(nb, timeout=300, kernel_name="python3",
                           resources={"metadata": {"path": str(notebook.parent)}}).execute()
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
              if o.get("output_type") == "error"]
    assert not errors, errors[0].get("evalue")


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items())
             if name.startswith("test_") and callable(obj)]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
