# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


class BenchmarkModelError(ValueError):
    pass


@dataclasses.dataclass
class BenchmarkImplementationRow:
    name: str
    key: str
    source: str
    api: str
    compile_mode: str
    backend: str
    opt: str
    chart_label: str
    main: bool
    hot_us: float
    compile_us: float | None
    rows_per_second: float | None
    throughput_gbps: float | None
    compile_samples_us: list[float] | None = None
    compile_warmups: int | None = None
    compile_runs: int | None = None
    compile_boundary: str | None = None
    speedup: float | None = None
    bar_percent: float = 0.0
    bar_label: str = "-"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BenchmarkImplementationRow":
        return cls(
            name=_required_str(raw, "name"),
            key=_required_str(raw, "key"),
            source=_required_str(raw, "source"),
            api=_required_str(raw, "api"),
            compile_mode=_required_str(raw, "compile_mode"),
            backend=_required_str(raw, "backend"),
            opt=_required_str(raw, "opt"),
            chart_label=_required_str(raw, "chart_label"),
            main=bool(_required(raw, "main")),
            hot_us=_required_float(raw, "hot_us"),
            compile_us=_optional_float(raw.get("compile_us")),
            rows_per_second=_optional_float(raw.get("rows_per_second")),
            throughput_gbps=_optional_float(raw.get("throughput_gbps")),
            compile_samples_us=(
                [float(value) for value in raw["compile_samples_us"]]
                if raw.get("compile_samples_us") is not None
                else None
            ),
            compile_warmups=(
                int(raw["compile_warmups"])
                if raw.get("compile_warmups") is not None
                else None
            ),
            compile_runs=(
                int(raw["compile_runs"])
                if raw.get("compile_runs") is not None
                else None
            ),
            compile_boundary=_optional_str(raw.get("compile_boundary")),
            speedup=_optional_float(raw.get("speedup")),
            bar_percent=float(raw.get("bar_percent", 0.0)),
            bar_label=str(raw.get("bar_label", "-")),
        )

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def __setitem__(self, name: str, value: Any) -> None:
        setattr(self, name, value)

    def to_json_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class BenchmarkRunResult:
    rows: int
    warmups: int
    runs: int
    null_density: float
    output: str
    arch: str
    provider: str
    provider_label: str
    compile_us: float | None
    median_us: float | None
    compile_ms: float | None
    median_ms: float | None
    rows_per_second: float | None
    throughput_gbps: float | None
    cache_hits: int
    cache_misses: int
    scalar_outputs: tuple[str, ...]
    implementations: tuple[BenchmarkImplementationRow, ...]
    diagnostics: tuple[str, ...] = ()
    google_benchmark: Mapping[str, Any] | None = None
    target_id: str = ""
    target_label: str = ""
    target_arch: str = ""
    target_health: str = ""
    platform_label: str = ""
    ok: bool = True

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BenchmarkRunResult":
        implementations = tuple(
            coerce_implementation_row(row)
            for row in _required_sequence(raw, "implementations")
        )
        return cls(
            rows=_required_int(raw, "rows"),
            warmups=_required_int(raw, "warmups"),
            runs=_required_int(raw, "runs"),
            null_density=_required_float(raw, "null_density"),
            output=_required_str(raw, "output"),
            arch=_required_str(raw, "arch"),
            provider=_required_str(raw, "provider"),
            provider_label=_required_str(raw, "provider_label"),
            compile_us=_optional_float(raw.get("compile_us")),
            median_us=_optional_float(raw.get("median_us")),
            compile_ms=_optional_float(raw.get("compile_ms")),
            median_ms=_optional_float(raw.get("median_ms")),
            rows_per_second=_optional_float(raw.get("rows_per_second")),
            throughput_gbps=_optional_float(raw.get("throughput_gbps")),
            cache_hits=int(raw.get("cache_hits", 0)),
            cache_misses=int(raw.get("cache_misses", 0)),
            scalar_outputs=tuple(str(value) for value in raw.get("scalar_outputs", ())),
            implementations=implementations,
            diagnostics=tuple(str(value) for value in raw.get("diagnostics", ())),
            google_benchmark=raw.get("google_benchmark"),
            target_id=str(raw.get("target_id", "")),
            target_label=str(raw.get("target_label", "")),
            target_arch=str(raw.get("target_arch", "")),
            target_health=str(raw.get("target_health", "")),
            platform_label=str(raw.get("platform_label", "")),
        )

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def to_json_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["ok"] = True
        return data


@dataclasses.dataclass(frozen=True)
class BenchmarkErrorResult:
    error: str
    diagnostics: tuple[str, ...] = ()
    ok: bool = False

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "BenchmarkErrorResult":
        return cls(
            error=str(raw.get("error") or "benchmark failed"),
            diagnostics=tuple(str(value) for value in raw.get("diagnostics", ())),
        )

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def to_json_dict(self) -> dict[str, Any]:
        return {"ok": False, "error": self.error, "diagnostics": list(self.diagnostics)}


BenchmarkResult = BenchmarkRunResult | BenchmarkErrorResult


def coerce_implementation_row(raw: BenchmarkImplementationRow | Mapping[str, Any]) -> BenchmarkImplementationRow:
    if isinstance(raw, BenchmarkImplementationRow):
        return raw
    if not isinstance(raw, Mapping):
        raise BenchmarkModelError(f"benchmark implementation row must be an object, got {type(raw).__name__}")
    return BenchmarkImplementationRow.from_mapping(raw)


def coerce_benchmark_result(raw: BenchmarkResult | Mapping[str, Any]) -> BenchmarkResult:
    if isinstance(raw, (BenchmarkRunResult, BenchmarkErrorResult)):
        return raw
    if not isinstance(raw, Mapping):
        raise BenchmarkModelError(f"benchmark result must be an object, got {type(raw).__name__}")
    if bool(raw.get("ok")):
        return BenchmarkRunResult.from_mapping(raw)
    return BenchmarkErrorResult.from_mapping(raw)


def to_json_dict(result: BenchmarkResult | BenchmarkImplementationRow | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, (BenchmarkRunResult, BenchmarkErrorResult, BenchmarkImplementationRow)):
        return result.to_json_dict()
    coerced = coerce_benchmark_result(result)
    return coerced.to_json_dict()


def _required(raw: Mapping[str, Any], name: str) -> Any:
    if name not in raw:
        raise BenchmarkModelError(f"benchmark result is missing required field {name!r}")
    return raw[name]


def _required_sequence(raw: Mapping[str, Any], name: str) -> tuple[Any, ...]:
    value = _required(raw, name)
    if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
        raise BenchmarkModelError(f"benchmark result field {name!r} must be a sequence")
    return tuple(value)


def _required_str(raw: Mapping[str, Any], name: str) -> str:
    return str(_required(raw, name))


def _required_int(raw: Mapping[str, Any], name: str) -> int:
    try:
        return int(_required(raw, name))
    except (TypeError, ValueError) as exc:
        raise BenchmarkModelError(f"benchmark result field {name!r} must be an integer") from exc


def _required_float(raw: Mapping[str, Any], name: str) -> float:
    try:
        return float(_required(raw, name))
    except (TypeError, ValueError) as exc:
        raise BenchmarkModelError(f"benchmark result field {name!r} must be a number") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise BenchmarkModelError(f"benchmark result optional numeric field has non-numeric value {value!r}") from exc


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
