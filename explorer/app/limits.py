# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
from typing import Any

from .config import load_config
from .targets import BenchmarkTarget


@dataclasses.dataclass(frozen=True)
class PublicLimits:
    max_query_bytes: int = 65_536
    max_share_query_bytes: int = 131_072
    max_form_bytes: int = 200_000
    max_form_fields: int = 32
    max_form_value_bytes: int = 131_072
    max_rows: int = 1_000_000
    max_warmups: int = 10
    max_runs: int = 20
    max_null_density: float = 1.0
    max_benchmark_seconds: int = 120
    allowed_providers: tuple[str, ...] = ("google", "python", "all")
    allowed_outputs: tuple[str, ...] = ("pyarrow",)
    allowed_targets: tuple[str, ...] = ()


class LimitError(ValueError):
    pass


def public_limits() -> PublicLimits:
    raw = load_config()
    security = raw.get("security", {}) if isinstance(raw.get("security", {}), dict) else {}
    limits = raw.get("public_limits", {}) if isinstance(raw.get("public_limits", {}), dict) else {}
    merged = {**security, **limits}
    return PublicLimits(
        max_query_bytes=_int_limit(merged, "max_query_bytes", PublicLimits.max_query_bytes),
        max_share_query_bytes=_int_limit(merged, "max_share_query_bytes", PublicLimits.max_share_query_bytes),
        max_form_bytes=_int_limit(merged, "max_form_bytes", PublicLimits.max_form_bytes),
        max_form_fields=_int_limit(merged, "max_form_fields", PublicLimits.max_form_fields),
        max_form_value_bytes=_int_limit(merged, "max_form_value_bytes", PublicLimits.max_form_value_bytes),
        max_rows=_int_limit(merged, "max_rows", PublicLimits.max_rows),
        max_warmups=_int_limit(merged, "max_warmups", PublicLimits.max_warmups),
        max_runs=_int_limit(merged, "max_runs", PublicLimits.max_runs),
        max_null_density=_float_limit(merged, "max_null_density", PublicLimits.max_null_density),
        max_benchmark_seconds=_int_limit(merged, "max_benchmark_seconds", PublicLimits.max_benchmark_seconds),
        allowed_providers=_string_tuple(merged.get("allowed_providers"), PublicLimits.allowed_providers),
        allowed_outputs=_string_tuple(merged.get("allowed_outputs"), PublicLimits.allowed_outputs),
        allowed_targets=_string_tuple(merged.get("allowed_targets"), PublicLimits.allowed_targets),
    )


def target_is_public(target: BenchmarkTarget, limits: PublicLimits | None = None) -> bool:
    caps = limits or public_limits()
    if caps.allowed_targets and target.id not in caps.allowed_targets:
        return False
    return target.public


def validate_benchmark_payload(payload: dict[str, Any], limits: PublicLimits | None = None) -> dict[str, Any]:
    caps = limits or public_limits()
    query = str(payload.get("query", ""))
    _check_bytes(query, caps.max_query_bytes, "query")
    rows = _coerce_int(payload.get("rows", 100_000), "rows", minimum=1)
    warmups = _coerce_int(payload.get("warmups", 3), "warmups", minimum=0)
    runs = _coerce_int(payload.get("runs", 10), "runs", minimum=1)
    null_density = _coerce_float(payload.get("null_density", 0.1), "null_density", minimum=0.0)
    provider = str(payload.get("provider", "google")).strip().lower()
    output = str(payload.get("output", "pyarrow")).strip().lower()

    if rows > caps.max_rows:
        raise LimitError(f"rows exceeds public limit ({rows} > {caps.max_rows})")
    if warmups > caps.max_warmups:
        raise LimitError(f"warmups exceeds public limit ({warmups} > {caps.max_warmups})")
    if runs > caps.max_runs:
        raise LimitError(f"runs exceeds public limit ({runs} > {caps.max_runs})")
    if null_density > caps.max_null_density:
        raise LimitError(
            f"null density exceeds public limit ({null_density:g} > {caps.max_null_density:g})"
        )
    if provider not in caps.allowed_providers:
        raise LimitError(f"benchmark provider {provider!r} is not enabled for this demo")
    if output not in caps.allowed_outputs:
        raise LimitError(f"benchmark output {output!r} is not enabled for this demo")

    normalized = dict(payload)
    normalized.update(
        {
            "query": query,
            "rows": rows,
            "warmups": warmups,
            "runs": runs,
            "null_density": null_density,
            "provider": provider,
            "output": output,
        }
    )
    return normalized


def _check_bytes(value: str, maximum: int, label: str) -> None:
    size = len(value.encode("utf-8"))
    if size > maximum:
        raise LimitError(f"{label} is too large ({size} bytes > {maximum} bytes)")


def _coerce_int(raw: Any, label: str, *, minimum: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise LimitError(f"{label} must be an integer") from exc
    if value < minimum:
        raise LimitError(f"{label} must be at least {minimum}")
    return value


def _coerce_float(raw: Any, label: str, *, minimum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise LimitError(f"{label} must be a number") from exc
    if value < minimum:
        raise LimitError(f"{label} must be at least {minimum:g}")
    return value


def _int_limit(raw: dict[str, Any], name: str, default: int) -> int:
    try:
        return max(1, int(raw.get(name, default)))
    except (TypeError, ValueError):
        return default


def _float_limit(raw: dict[str, Any], name: str, default: float) -> float:
    try:
        return max(0.0, float(raw.get(name, default)))
    except (TypeError, ValueError):
        return default


def _string_tuple(raw: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if raw is None:
        return default
    if isinstance(raw, str):
        values = [raw]
    elif isinstance(raw, list):
        values = raw
    else:
        return default
    return tuple(str(value).strip().lower() for value in values if str(value).strip())
