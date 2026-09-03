# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
import functools
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .config import (
    CONFIG_ENV,
    DEFAULT_CONFIGS,
    ROOT,
    ConfigError,
    clear_config_cache,
    config_path,
    config_version,
    load_config as _load_config,
)
from .benchmark_model import BenchmarkErrorResult, BenchmarkResult, coerce_benchmark_result

REPO_ROOT = ROOT.parent
PYTHON_BUILD_DIR_ENV = "SIMJIT_PYTHON_BUILD_DIR"


@dataclasses.dataclass(frozen=True)
class TargetProbe:
    ok: bool
    arch_family: str
    machine: str = ""
    system: str = ""
    python: str = ""
    gxx: str = ""
    clang: str = ""
    llvm_config: str = ""
    error: str = ""


@dataclasses.dataclass(frozen=True)
class BenchmarkTarget:
    id: str
    label: str
    arch_family: str
    public: bool = True
    env: tuple[tuple[str, str], ...] = ()


@dataclasses.dataclass(frozen=True)
class BenchmarkTargetChoice:
    id: str
    label: str
    display_label: str
    arch_family: str
    available: bool
    health: str


class TargetError(RuntimeError):
    pass


def load_config() -> dict[str, Any]:
    try:
        return _load_config()
    except ConfigError as exc:
        raise TargetError(str(exc)) from exc


def host_arch_family() -> str:
    return machine_to_arch_family(platform.machine())


def machine_to_arch_family(machine: str) -> str:
    value = machine.strip().lower()
    if value in {"arm64", "aarch64"}:
        return "arm"
    if value in {"x86_64", "amd64", "x64"}:
        return "x86"
    return "unknown"


def arch_family_for_code_arch(arch: str, target: BenchmarkTarget) -> str:
    if arch == "native":
        if target.arch_family != "unknown":
            return target.arch_family
        return host_arch_family()
    if arch in {"x86", "x86-ymm"}:
        return "x86"
    if arch == "arm":
        return "arm"
    return "unknown"


def target_can_run_arch(target: BenchmarkTarget, arch: str) -> bool:
    if target.arch_family == "unknown":
        return False
    return arch_family_for_code_arch(arch, target) == target.arch_family


def _local_label() -> str:
    family = host_arch_family()
    if family == "arm":
        return "Local arm64"
    if family == "x86":
        return "Local x86-64"
    return f"Local {platform.machine() or 'unknown'}"


def _local_probe(target: BenchmarkTarget) -> TargetProbe:
    env = _local_probe_env(target)
    return TargetProbe(
        ok=True,
        arch_family=host_arch_family(),
        machine=platform.machine(),
        system=platform.system(),
        python=platform.python_version(),
        gxx=shutil.which("g++") or "",
        clang=env.get("LLVM_CLANG") or shutil.which("clang") or "",
        llvm_config=env.get("LLVM_CONFIG") or shutil.which("llvm-config") or "",
    )


def benchmark_targets() -> tuple[BenchmarkTarget, ...]:
    return _benchmark_targets_cached(config_version())


@functools.lru_cache(maxsize=4)
def _benchmark_targets_cached(_version) -> tuple[BenchmarkTarget, ...]:
    raw = load_config()
    local_config = (
        raw.get("local", {}) if isinstance(raw.get("local", {}), dict) else {}
    )
    configured_targets = raw.get("targets", []) if isinstance(raw, dict) else []
    if configured_targets not in (None, []):
        raise TargetError("Explorer benchmark targets only support the local host")
    local_env = _parse_target_env(local_config.get("env", {}), "local")
    return (
        BenchmarkTarget(
            id="local",
            label=str(local_config.get("label") or _local_label()),
            arch_family=host_arch_family(),
            public=_bool_config(local_config.get("public", True)),
            env=local_env,
        ),
    )


def _parse_target_env(raw: Any, target_id: str) -> tuple[tuple[str, str], ...]:
    if raw in (None, {}):
        return ()
    if not isinstance(raw, dict):
        raise TargetError(f"benchmark target {target_id!r} env must be an object")
    env: list[tuple[str, str]] = []
    for key, value in sorted(raw.items()):
        name = str(key).strip()
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            raise TargetError(
                f"benchmark target {target_id!r} has invalid env name {name!r}"
            )
        env.append((name, str(value)))
    return tuple(env)


def _bool_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def get_target(target_id: str) -> BenchmarkTarget:
    for target in benchmark_targets():
        if target.id == target_id:
            return target
    raise TargetError(f"unknown benchmark target {target_id!r}")


def target_choices() -> tuple[BenchmarkTargetChoice, ...]:
    choices: list[BenchmarkTargetChoice] = []
    for target in benchmark_targets():
        probe = probe_target(target)
        choices.append(
            BenchmarkTargetChoice(
                id=target.id,
                label=target.label,
                display_label=_target_display_label(target, probe),
                arch_family=probe.arch_family,
                available=probe.ok,
                health=_probe_health(probe),
            )
        )
    return tuple(choices)


def _target_display_label(target: BenchmarkTarget, probe: TargetProbe) -> str:
    if not probe.ok:
        return f"{target.label} - offline"
    family = probe.arch_family if probe.arch_family != "unknown" else "unknown arch"
    missing = []
    if not probe.gxx:
        missing.append("g++")
    suffix = f"{family}"
    if missing:
        suffix += "; missing " + ", ".join(missing)
    return f"{target.label} - {suffix}"


def _probe_health(probe: TargetProbe) -> str:
    if not probe.ok:
        return probe.error or "unavailable"
    parts = [probe.system, probe.machine]
    if probe.python:
        parts.append(f"python {probe.python}")
    missing = []
    if not probe.gxx:
        missing.append("g++")
    if missing:
        parts.append("missing " + ", ".join(missing))
    return " / ".join(part for part in parts if part)


def probe_target(target: BenchmarkTarget) -> TargetProbe:
    return _probe_target_cached(config_version(), target)


@functools.lru_cache(maxsize=16)
def _probe_target_cached(_version, target: BenchmarkTarget) -> TargetProbe:
    return _local_probe(target)


class LocalBenchmarkExecutor:
    def __init__(self, target: BenchmarkTarget):
        self.target = target

    def run(self, payload: dict[str, Any], *, timeout: int = 60) -> BenchmarkResult:
        env = dict(os.environ)
        env.update(dict(self.target.env))
        env["PYTHONPATH"] = _worker_pythonpath(env)
        proc = subprocess.run(
            [os.sys.executable, "-m", "explorer.app.bench_worker"],
            input=json.dumps(payload),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout,
            check=False,
        )
        return _parse_worker_result(proc, "local benchmark")


def _worker_pythonpath(env: dict[str, str]) -> str:
    parts = [str(REPO_ROOT)]
    build_dir = _python_build_dir(env)
    if build_dir is not None:
        parts.append(str(build_dir))
    parts.extend(
        [
            str(REPO_ROOT / "python"),
            str(REPO_ROOT / "python" / "src"),
            str(REPO_ROOT / "python" / "tests"),
        ]
    )
    existing = env.get("PYTHONPATH")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def _python_build_dir(env: dict[str, str]) -> Path | None:
    configured = env.get(PYTHON_BUILD_DIR_ENV)
    if configured:
        return Path(configured)
    dev_build = REPO_ROOT / "build" / "python-dev"
    if any(dev_build.glob("simjit/_simjit*.so")):
        return dev_build
    python_build = REPO_ROOT / "build" / "python"
    for extension in python_build.glob("**/simjit/_simjit*.so"):
        return extension.parent.parent
    return None


def _parse_worker_result(
    proc: subprocess.CompletedProcess[str], label: str
) -> BenchmarkResult:
    if proc.returncode != 0:
        return BenchmarkErrorResult(
            proc.stderr.strip()
            or proc.stdout.strip()
            or f"{label} exited with {proc.returncode}"
        )
    try:
        return coerce_benchmark_result(json.loads(proc.stdout))
    except json.JSONDecodeError as exc:
        return BenchmarkErrorResult(f"invalid benchmark JSON from {label}: {exc}: {proc.stdout}")


def clear_target_caches() -> None:
    clear_config_cache()
    _benchmark_targets_cached.cache_clear()
    _probe_target_cached.cache_clear()


def _local_probe_env(target: BenchmarkTarget) -> dict[str, str]:
    env = dict(os.environ)
    env.update(dict(target.env))
    return env
