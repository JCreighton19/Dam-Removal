"""Shared case-study configuration loading and project-root resolution.

Every ETL script that needs to know "which watershed(s) am I running for"
or "where are the source files" should go through `load_config()` instead
of hardcoding constants. Swapping the pilot case study (or eventually
scaling to a national run) means pointing at a different file under
`config/`, not editing script internals.

Path resolution is anchored to this file's location on disk (not the
process's current working directory), so scripts behave the same whether
invoked as `python etl/foo.py` from the repo root, from inside `etl/`, or
via a Slurm job with its own working directory. Set the
`DAM_REMOVAL_PROJECT_ROOT` environment variable to override this (e.g. if
the code is ever deployed somewhere other than a checkout of this repo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT_ENV_VAR = "DAM_REMOVAL_PROJECT_ROOT"
CONFIG_DIR_NAME = "config"

# Keys every case-study config must define. Individual scripts validate
# any additional keys they specifically depend on (e.g. `wq_characteristics`).
REQUIRED_KEYS = ("case_study", "huc8s", "dam_source", "wbd_source")


class ConfigError(ValueError):
    """Raised when a case-study configuration is missing or invalid."""


def get_project_root() -> Path:
    """Resolve the project root, independent of the current working directory."""
    env_value = os.environ.get(PROJECT_ROOT_ENV_VAR)

    if env_value:
        root = Path(env_value).expanduser().resolve()
        if not root.is_dir():
            raise ConfigError(
                f"${PROJECT_ROOT_ENV_VAR} is set to '{root}', but that "
                "directory does not exist."
            )
        return root

    # This file lives at <project_root>/etl/common/config.py.
    return Path(__file__).resolve().parents[2]


def _config_path(name: str, project_root: Path) -> Path:
    if name.endswith((".yml", ".yaml")):
        candidate = Path(name)
        return candidate if candidate.is_absolute() else project_root / candidate

    return project_root / CONFIG_DIR_NAME / f"{name}.yaml"


@dataclass
class CaseStudyConfig:
    """A loaded case-study config, with convenience accessors."""

    name: str
    project_root: Path
    source_path: Path
    raw: dict = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, key: str) -> Path:
        """Resolve a config value that names a project-relative path."""
        value = self.raw.get(key)
        if value is None:
            raise ConfigError(
                f"Case-study config '{self.source_path}' is missing required "
                f"path key '{key}'."
            )
        candidate = Path(value)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    @property
    def huc8s(self) -> list:
        return [str(huc8) for huc8 in self.raw.get("huc8s", [])]


def load_config(name: str, project_root: Path | None = None) -> CaseStudyConfig:
    """Load a case-study config by name (e.g. "pilot_19020401").

    `name` may also be a path to a .yaml file, absolute or relative to the
    project root, for one-off configs that don't live under `config/`.
    """
    root = project_root or get_project_root()
    path = _config_path(name, root)

    if not path.exists():
        config_dir = root / CONFIG_DIR_NAME
        available = sorted(p.stem for p in config_dir.glob("*.yaml")) if config_dir.is_dir() else []
        raise ConfigError(
            f"Case-study config not found: {path}\n"
            f"Available configs under {config_dir}: {available or '(none found)'}"
        )

    with path.open("r") as handle:
        raw = yaml.safe_load(handle) or {}

    missing = [key for key in REQUIRED_KEYS if key not in raw]
    if missing:
        raise ConfigError(
            f"Case-study config '{path}' is missing required keys: {missing}"
        )

    if not raw["huc8s"]:
        raise ConfigError(
            f"Case-study config '{path}' must list at least one HUC8 in 'huc8s'."
        )

    return CaseStudyConfig(
        name=raw.get("case_study", name),
        project_root=root,
        source_path=path,
        raw=raw,
    )
