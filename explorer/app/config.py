# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
import functools
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIGS = (
    ROOT / "config.yaml",
    ROOT / "config.yml",
    ROOT / "config.json",
)
CONFIG_ENV = "SIMJIT_EXPLORER_TARGETS_FILE"


class ConfigError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ConfigVersion:
    path: str
    configured: bool
    exists: bool
    mtime_ns: int
    size: int


def config_path() -> Path | None:
    configured = os.environ.get(CONFIG_ENV)
    if configured:
        return Path(configured)
    for path in DEFAULT_CONFIGS:
        if path.exists():
            return path
    return None


def config_version() -> ConfigVersion:
    configured = bool(os.environ.get(CONFIG_ENV))
    path = config_path()
    if path is None:
        return ConfigVersion("", configured, False, 0, 0)
    try:
        stat = path.stat()
    except OSError:
        return ConfigVersion(str(path), configured, False, 0, 0)
    return ConfigVersion(str(path), configured, True, stat.st_mtime_ns, stat.st_size)


def load_config() -> dict[str, Any]:
    return _load_config_cached(config_version())


@functools.lru_cache(maxsize=4)
def _load_config_cached(version: ConfigVersion) -> dict[str, Any]:
    if not version.path:
        return {}
    path = Path(version.path)
    if not version.exists:
        raise ConfigError(f"failed to read benchmark target config {path}: file does not exist")
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml
            except Exception as exc:
                raise ConfigError(
                    f"failed to read benchmark target config {path}: PyYAML is required for YAML config"
                ) from exc
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(f"failed to read benchmark target config {path}: {exc}") from exc
    if data is None:
        return {}
    if isinstance(data, list):
        return {"targets": data}
    if not isinstance(data, dict):
        raise ConfigError("benchmark target config must be an object")
    return data


def clear_config_cache() -> None:
    _load_config_cached.cache_clear()
