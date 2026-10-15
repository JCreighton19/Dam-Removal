"""Lightweight validation checks for etl.common.config.

Run directly: `python tests/test_config.py`
Also discoverable by pytest, if installed: `pytest tests/`
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.common.config import ConfigError, load_config


def _write_config(config_dir, name, contents):
    path = Path(config_dir) / f"{name}.yaml"
    path.write_text(contents)
    return path


def test_load_config_reads_required_fields():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_dir = root / "config"
        config_dir.mkdir()
        _write_config(
            config_dir,
            "demo",
            """
case_study: demo
huc8s: ["00000000"]
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
candidate_distance_km: 5
""",
        )

        cfg = load_config("demo", project_root=root)

        assert cfg.name == "demo"
        assert cfg.huc8s == ["00000000"]
        assert cfg.get("candidate_distance_km") == 5
        assert cfg.path("dam_source") == root / "data/raw/dams/dams.csv"


def test_load_config_missing_file_raises_clear_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "config").mkdir()
        try:
            load_config("does_not_exist", project_root=root)
            raise AssertionError("Expected ConfigError for a missing config file.")
        except ConfigError:
            pass


def test_load_config_missing_required_key_raises_clear_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_dir = root / "config"
        config_dir.mkdir()
        _write_config(
            config_dir,
            "incomplete",
            """
case_study: incomplete
huc8s: ["00000000"]
""",
        )
        try:
            load_config("incomplete", project_root=root)
            raise AssertionError("Expected ConfigError for missing required keys.")
        except ConfigError:
            pass


def test_load_config_empty_huc8_list_raises_clear_error():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        config_dir = root / "config"
        config_dir.mkdir()
        _write_config(
            config_dir,
            "empty_scope",
            """
case_study: empty_scope
huc8s: []
dam_source: data/raw/dams/dams.csv
wbd_source: data/raw/wbd/wbd.gpkg
""",
        )
        try:
            load_config("empty_scope", project_root=root)
            raise AssertionError("Expected ConfigError for an empty huc8s list.")
        except ConfigError:
            pass


def test_project_root_env_var_overrides_default():
    from etl.common.config import PROJECT_ROOT_ENV_VAR, get_project_root

    with tempfile.TemporaryDirectory() as tmp:
        old_value = os.environ.get(PROJECT_ROOT_ENV_VAR)
        os.environ[PROJECT_ROOT_ENV_VAR] = tmp
        try:
            assert get_project_root() == Path(tmp).resolve()
        finally:
            if old_value is None:
                os.environ.pop(PROJECT_ROOT_ENV_VAR, None)
            else:
                os.environ[PROJECT_ROOT_ENV_VAR] = old_value


def test_default_case_study_follows_env_var_with_fallback():
    from etl.common.config import (
        CASE_STUDY_ENV_VAR,
        DEFAULT_CASE_STUDY,
        default_case_study,
    )

    old_value = os.environ.get(CASE_STUDY_ENV_VAR)
    try:
        os.environ.pop(CASE_STUDY_ENV_VAR, None)
        assert default_case_study() == DEFAULT_CASE_STUDY

        os.environ[CASE_STUDY_ENV_VAR] = "some_other_case_study"
        assert default_case_study() == "some_other_case_study"
    finally:
        if old_value is None:
            os.environ.pop(CASE_STUDY_ENV_VAR, None)
        else:
            os.environ[CASE_STUDY_ENV_VAR] = old_value


if __name__ == "__main__":
    tests = [
        obj for name, obj in list(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} tests passed.")
