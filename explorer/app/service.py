# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
import functools
import platform
import subprocess
from typing import Any

import simjit as sj
import simjit.ir as ir

from . import querylang
from .benchmark_model import BenchmarkErrorResult, BenchmarkResult
from .limits import LimitError, public_limits, target_is_public, validate_benchmark_payload
from .lisp_format import FormatError, format_lisp
from .samples import DEFAULT_QUERY
from .targets import (
    LocalBenchmarkExecutor,
    TargetError,
    benchmark_targets,
    get_target,
    probe_target,
    target_choices,
)


BENCHMARK_PROVIDERS = (
    ("google", "Google Benchmark"),
    ("python", "Python API"),
    ("all", "All providers"),
)

INPUT_MODES = (
    ("expression_sql", "Expression SQL"),
    ("serialized_hir", "Serialized HIR"),
)

ARCH_ALIASES = {
    "auto": "native",
    "native": "native",
    "x86": "x86",
    "avx512": "x86",
    "amd64-avx512": "x86",
    "x86-ymm": "x86-ymm",
    "avx512-ymm": "x86-ymm",
    "amd64-avx512-ymm": "x86-ymm",
    "arm": "arm",
    "aarch64": "arm",
    "neon": "arm",
    "arm64": "arm",
    "arm64-neon": "arm",
}


@dataclasses.dataclass(frozen=True)
class PlatformChoice:
    value: str
    label: str
    runnable: bool
    arch_family: str


@dataclasses.dataclass(frozen=True)
class CompileResult:
    ok: bool
    query: str
    sections: dict[str, str]
    diagnostics: list[str]
    error: str = ""
    arch: str = "native"
    input_mode: str = "expression_sql"


@dataclasses.dataclass(frozen=True)
class BenchmarkOptions:
    rows: int = 100_000
    warmups: int = 3
    runs: int = 10
    null_density: float = 0.1
    output: str = "pyarrow"
    arch: str = "native"
    benchmark_target: str = "local"
    provider: str = "google"


@dataclasses.dataclass(frozen=True)
class BenchmarkAvailability:
    runnable: bool
    reason: str = ""
    title: str = ""
    platform_label: str = ""
    native_arch: str = ""

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def normalize_arch(arch: str) -> str:
    value = arch.strip().lower()
    try:
        return ARCH_ALIASES[value]
    except KeyError as exc:
        choices = ", ".join(["native", "x86", "x86-ymm", "arm"])
        raise ValueError(f"unsupported platform {arch!r}; expected one of: {choices}") from exc


def normalize_input_mode(input_mode: str) -> str:
    value = input_mode.strip().lower().replace("-", "_")
    if value in {"expression_sql", "sql", "expr_sql"}:
        return "expression_sql"
    if value in {"serialized_hir", "hir", "serialized"}:
        return "serialized_hir"
    choices = ", ".join(value for value, _ in INPUT_MODES)
    raise ValueError(f"unsupported input mode {input_mode!r}; expected one of: {choices}")


def _host_family() -> str:
    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm"
    if machine in {"x86_64", "amd64"}:
        return "x86"
    return "unknown"


def _host_label() -> str:
    family = _host_family()
    if family == "arm":
        return "arm64"
    if family == "x86":
        return "x86-64"
    return platform.machine() or "unknown"


def arch_label(arch: str) -> str:
    labels = {
        "native": f"Native ({_host_label()})",
        "x86": "x86 AVX-512",
        "x86-ymm": "x86 AVX-512 YMM",
        "arm": "Arm64 NEON",
    }
    return labels[normalize_arch(arch)]


def _native_platform_arch(arch_family: str) -> str:
    if arch_family == "arm":
        return "arm"
    return "x86"


def default_platform_arch(benchmark_target: str = "local") -> str:
    try:
        target = get_target(benchmark_target)
        probe = probe_target(target)
    except Exception:
        return _native_platform_arch(_host_family())
    if probe.ok and probe.arch_family != "unknown":
        return _native_platform_arch(probe.arch_family)
    return _native_platform_arch(target.arch_family)


def _benchmark_platform_label(arch: str, probe) -> str:
    value = normalize_arch(arch)
    if value == "native":
        return arch_label(_native_platform_arch(probe.arch_family))
    return arch_label(value)


def is_benchmark_arch_runnable(arch: str, benchmark_target: str = "local") -> bool:
    value = normalize_arch(arch)
    try:
        target = get_target(benchmark_target)
        probe = probe_target(target)
    except Exception:
        return False
    if not probe.ok or probe.arch_family == "unknown":
        return False
    if value == "native":
        return True
    return value == _native_platform_arch(probe.arch_family)


def platform_choices() -> tuple[PlatformChoice, ...]:
    return (
        PlatformChoice("x86", arch_label("x86"), is_benchmark_arch_runnable("x86"), "x86"),
        PlatformChoice("x86-ymm", arch_label("x86-ymm"), False, "x86"),
        PlatformChoice("arm", arch_label("arm"), is_benchmark_arch_runnable("arm"), "arm"),
    )


def benchmark_target_choices():
    caps = public_limits()
    return tuple(choice for choice in target_choices() if target_is_public(get_target(choice.id), caps))


def default_benchmark_target() -> str:
    caps = public_limits()
    targets = tuple(target for target in benchmark_targets() if target_is_public(target, caps))
    return targets[0].id if targets else "local"


def benchmark_availability(
    input_mode: str,
    arch: str,
    benchmark_target: str = "local",
) -> BenchmarkAvailability:
    try:
        normalized_mode = normalize_input_mode(input_mode)
    except ValueError as exc:
        return BenchmarkAvailability(False, str(exc), str(exc))
    try:
        normalized_arch = normalize_arch(arch)
    except ValueError as exc:
        return BenchmarkAvailability(False, str(exc), str(exc))
    try:
        target = get_target(benchmark_target or default_benchmark_target())
        probe = probe_target(target)
    except TargetError as exc:
        return BenchmarkAvailability(False, str(exc), str(exc))

    native_arch = ""
    platform_label = ""
    if probe.ok and probe.arch_family != "unknown":
        native_arch = _native_platform_arch(probe.arch_family)
        platform_label = _benchmark_platform_label(normalized_arch, probe)
    elif target.arch_family != "unknown":
        native_arch = _native_platform_arch(target.arch_family)
        platform_label = arch_label(normalized_arch if normalized_arch != "native" else native_arch)

    if normalized_mode != "expression_sql":
        reason = "Benchmarks require Expression SQL input mode"
        return BenchmarkAvailability(False, reason, reason, platform_label, native_arch)

    caps = public_limits()
    if not target_is_public(target, caps):
        reason = f"benchmark target {target.label} is not enabled for this demo"
        return BenchmarkAvailability(False, reason, reason, platform_label, native_arch)
    if not probe.ok:
        reason = probe.error or f"benchmark target {target.id!r} is unavailable"
        return BenchmarkAvailability(False, reason, reason, platform_label, native_arch)
    if probe.arch_family == "unknown":
        reason = "benchmark target architecture is unknown"
        return BenchmarkAvailability(False, reason, reason, platform_label, native_arch)
    if normalized_arch != "native" and normalized_arch != native_arch:
        reason = (
            f"benchmark platform {arch_label(normalized_arch)} cannot execute on target "
            f"{target.label} ({probe.arch_family}); compile-only inspection is still available"
        )
        return BenchmarkAvailability(False, reason, reason, platform_label, native_arch)

    return BenchmarkAvailability(True, "", "", platform_label or arch_label(native_arch), native_arch)


def benchmark_availability_matrix() -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    arches = ("native", *(choice.value for choice in platform_choices()))
    matrix: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for input_mode, _ in INPUT_MODES:
        matrix[input_mode] = {}
        for target in benchmark_target_choices():
            matrix[input_mode][target.id] = {
                arch: benchmark_availability(input_mode, arch, target.id).to_json_dict()
                for arch in arches
            }
    return matrix


def _schema_for_inspection(lowered: querylang.LoweredQuery) -> dict[str, ir.ScalarType | tuple[ir.ScalarType, bool]]:
    return {
        name: (dt, lowered.nullable.get(name, False))
        for name, dt in lowered.schema.items()
    }


def _is_mir_section_header(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and all(ch.isupper() or ch.isspace() for ch in stripped)


def format_mir_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _is_mir_section_header(stripped):
            lines.append(stripped)
        else:
            lines.append(f"  {stripped}")
    return "\n".join(lines)


def _is_asmjit_label(line: str) -> bool:
    if not line.endswith(":"):
        return False
    label = line[:-1]
    return bool(label) and all(ch.isalnum() or ch in "._$" for ch in label)


def format_asmjit_text(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(".section") or _is_asmjit_label(stripped):
            lines.append(stripped)
        else:
            lines.append(f"  {stripped}")
    return "\n".join(lines)


def _sections_from_inspection(inspection: sj.InspectionResult, diagnostics: list[str]) -> dict[str, str]:
    serialized = inspection.serialized
    if serialized:
        try:
            serialized = format_lisp(serialized)
        except FormatError as exc:
            diagnostics.append(f"Serialized HIR formatting failed: {exc}")

    return {
        "C++": inspection.cpp,
        "AsmJit": format_asmjit_text(inspection.asm),
        "LLVM IR": inspection.llvm_ir,
        "Serialized HIR": serialized,
        "HIR": inspection.hir,
        "Vectorizer": inspection.vectorizer,
        "MIR": format_mir_text(inspection.mir),
    }


def compile_query(
    query: str,
    policy: str = "best_effort",
    arch: str = "native",
    input_mode: str = "expression_sql",
) -> CompileResult:
    try:
        normalized_arch = normalize_arch(arch)
        normalized_mode = normalize_input_mode(input_mode)
    except Exception as exc:
        return CompileResult(False, query, {}, [], error=str(exc), arch=str(arch))
    return _compile_query(query, policy, normalized_arch, normalized_mode)


@functools.lru_cache(maxsize=64)
def _compile_query(query: str, policy: str, arch: str, input_mode: str) -> CompileResult:
    try:
        if input_mode == "serialized_hir":
            inspection = sj.inspect_serialized(query, policy=policy, arch=arch)
        else:
            parsed = querylang.parse(query)
            lowered = querylang.lower(parsed)
            inspection = sj.inspect(
                lowered.program,
                _schema_for_inspection(lowered),
                policy=policy,
                arch=arch,
            )
        diagnostics = list(inspection.diagnostics)
        sections = _sections_from_inspection(inspection, diagnostics)
        if inspection.vectorization_exception and inspection.vectorization_exception not in diagnostics:
            diagnostics.append(inspection.vectorization_exception)
        return CompileResult(True, query, sections, diagnostics, arch=arch, input_mode=input_mode)
    except Exception as exc:
        return CompileResult(False, query, {}, [], error=str(exc), arch=arch, input_mode=input_mode)


def run_benchmark(query: str, options: BenchmarkOptions) -> BenchmarkResult:
    arch = normalize_arch(options.arch)
    target_id = options.benchmark_target or default_benchmark_target()
    availability = benchmark_availability("expression_sql", arch, target_id)
    if not availability.runnable:
        return BenchmarkErrorResult(availability.reason)
    try:
        target = get_target(target_id)
        probe = probe_target(target)
    except TargetError as exc:
        return BenchmarkErrorResult(str(exc))
    caps = public_limits()

    try:
        payload = validate_benchmark_payload(
            {
                "query": query,
                "arch": "native",
                "rows": options.rows,
                "warmups": options.warmups,
                "runs": options.runs,
                "null_density": options.null_density,
                "output": options.output,
                "provider": options.provider,
            },
            caps,
        )
    except LimitError as exc:
        return BenchmarkErrorResult(str(exc))

    executor = LocalBenchmarkExecutor(target)
    try:
        result = executor.run(payload, timeout=caps.max_benchmark_seconds)
    except subprocess.TimeoutExpired as exc:
        return BenchmarkErrorResult(f"benchmark target {target.label} timed out after {exc.timeout:g}s")
    except TimeoutError as exc:
        return BenchmarkErrorResult(f"benchmark target {target.label} timed out: {exc}")
    except Exception as exc:
        return BenchmarkErrorResult(str(exc))
    if isinstance(result, BenchmarkErrorResult):
        return result
    return dataclasses.replace(
        result,
        target_id=target.id,
        target_label=target.label,
        target_arch=probe.arch_family,
        target_health=probe.error if not probe.ok else "",
        platform_label=availability.platform_label,
    )
