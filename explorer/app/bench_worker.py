# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import simjit.ir as ir
from simjit import _simjit as sj_ext

import simjit as sj

from . import querylang
from .benchmark_model import BenchmarkImplementationRow, BenchmarkRunResult, to_json_dict
from .limits import validate_benchmark_payload
from .service import _schema_for_inspection

CPP_PARAM_RE = re.compile(
    r"\bvoid\s+expr\s*\(\s*size_t\s+\w+\s*,(?P<params>.*?)\)\s*\{", re.S
)
CPP_POINTER_PARAM_RE = re.compile(r"^(?:const\s+)?(?P<type>.+?)\s*\*")
HIR_ARG_RE = re.compile(r"^@\d+\s+arg\s+dtype=(?P<dtype>[a-z0-9]+)\s+kind=", re.M)
DTYPE_TO_CPP = {
    "i1": "uint8_t",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "i128": "__int128",
    "f32": "float",
    "f64": "double",
}
PROVIDER_LABELS = {
    "google": "Google Benchmark",
    "python": "Python API",
    "all": "All providers",
}

IMPLEMENTATION_METADATA = {
    "asmjit_vectorized_py": {
        "api": "Python",
        "compile_mode": "SIMD",
        "backend": "AsmJit",
        "opt": "-",
    },
    "asmjit_scalar_py": {
        "api": "Python",
        "compile_mode": "Scalar",
        "backend": "AsmJit",
        "opt": "-",
    },
    "asmjit_vectorized": {
        "api": "C++",
        "compile_mode": "SIMD",
        "backend": "AsmJit",
        "opt": "-",
    },
    "asmjit_scalar": {
        "api": "C++",
        "compile_mode": "Scalar",
        "backend": "AsmJit",
        "opt": "-",
    },
    "cpp_vectorized": {
        "api": "C++",
        "compile_mode": "SIMD",
        "backend": "C++ emitter",
        "opt": "O1",
    },
    "cpp_scalar": {
        "api": "C++",
        "compile_mode": "Scalar",
        "backend": "C++ emitter",
        "opt": "O1",
    },
    "cpp_scalar_o3": {
        "api": "C++",
        "compile_mode": "Scalar",
        "backend": "C++ emitter",
        "opt": "O3",
    },
    "llvm_vectorized": {
        "api": "C++",
        "compile_mode": "SIMD",
        "backend": "LLVM",
        "opt": "O1",
    },
    "llvm_scalar": {
        "api": "C++",
        "compile_mode": "Scalar",
        "backend": "LLVM",
        "opt": "O1",
    },
    "llvm_scalar_o3": {
        "api": "C++",
        "compile_mode": "Scalar",
        "backend": "LLVM",
        "opt": "O3",
    },
}

API_ORDER = {"Python": 0, "C++": 1}
BACKEND_ORDER = {"AsmJit": 0, "C++ emitter": 1, "LLVM": 2}
COMPILE_MODE_ORDER = {"SIMD": 0, "Scalar": 1}

CPP_BASELINE_OPT = "O1"
CPP_AGGRESSIVE_OPT = "O3"
LLVM_BASELINE_OPT = "O1"
LLVM_BASELINE_PROBE_OPT = "O1"
LLVM_AGGRESSIVE_OPT = "O3"
LLVM_TARGET_MARCH_ENV = "SIMJIT_EXPLORER_LLVM_MARCH"
SCALAR_OUTPUT_TYPES = (
    ir.ArithAggExpr,
    ir.PredicateAggExpr,
    ir.CountIfExpr,
)
TYPE_TO_NUMPY = {
    sj.I1: np.dtype(np.bool_),
    sj.I8: np.dtype(np.int8),
    sj.I16: np.dtype(np.int16),
    sj.I32: np.dtype(np.int32),
    sj.I64: np.dtype(np.int64),
    sj.U8: np.dtype(np.uint8),
    sj.U16: np.dtype(np.uint16),
    sj.U32: np.dtype(np.uint32),
    sj.U64: np.dtype(np.uint64),
    sj.F32: np.dtype(np.float32),
    sj.F64: np.dtype(np.float64),
}


def numpy_dtype(dt) -> np.dtype:
    if dt.name == "timestamp64":
        return np.dtype(f"datetime64[{dt.unit}]")
    if dt not in TYPE_TO_NUMPY:
        raise TypeError(f"unsupported simjit scalar type {dt}")
    return TYPE_TO_NUMPY[dt]


def _values(dt, rows: int, rng: np.random.Generator):
    if dt == sj.I1:
        return rng.integers(0, 2, size=rows, dtype=np.int8).astype(bool)
    if dt == sj.F32:
        return rng.normal(size=rows).astype(np.float32)
    if dt == sj.F64:
        return rng.normal(size=rows).astype(np.float64)
    dtype = numpy_dtype(dt)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        low = max(1, int(info.min))
        high = min(1000, int(info.max)) + 1
        return rng.integers(low, high, size=rows, dtype=dtype)
    return rng.integers(1, 1000, size=rows, dtype=dtype)


def _pack_bits(values) -> bytes:
    data = bytearray((len(values) + 7) // 8)
    for idx, value in enumerate(values):
        if bool(value):
            data[idx // 8] |= 1 << (idx % 8)
    return bytes(data)


def make_inputs(
    schema: dict[str, Any],
    nullable: dict[str, bool],
    *,
    rows: int,
    null_density: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(42)
    inputs: dict[str, Any] = {}
    for name, dt in schema.items():
        values = _values(dt, rows, rng)
        null = None
        if nullable.get(name, False) and null_density > 0:
            is_valid = rng.random(rows) >= null_density
            null = ir.NullEncoding(
                kind="mask_bitpacked",
                buf=_pack_bits(is_valid),
                true_means_null=False,
            )
        if dt == sj.I1:
            inputs[name] = ir.BufferHandle(
                ty=dt,
                buf=_pack_bits(values),
                length=rows,
                aligned=True,
                bitpacked=True,
                null=null,
            )
        else:
            inputs[name] = ir.BufferHandle(
                ty=dt,
                buf=values,
                length=rows,
                aligned=True,
                bitpacked=False,
                null=null,
            )
    return inputs


def run(payload: dict[str, Any]) -> BenchmarkRunResult:
    payload = validate_benchmark_payload(payload)
    parsed = querylang.parse(payload["query"])
    lowered = querylang.lower(parsed)
    rows = int(payload.get("rows", 100_000))
    warmups = int(payload.get("warmups", 3))
    runs = int(payload.get("runs", 10))
    null_density = float(payload.get("null_density", 0.1))
    output = str(payload.get("output", "pyarrow"))
    arch = str(payload.get("arch", "native"))
    provider = str(payload.get("provider", "google")).strip().lower()
    if provider not in PROVIDER_LABELS:
        raise ValueError(f"unsupported benchmark provider {provider!r}")
    inputs = make_inputs(
        lowered.schema,
        lowered.nullable,
        rows=rows,
        null_density=null_density,
    )

    outputs = lowered.program.to_dsl()
    scalar_outputs = scalar_output_names(outputs)
    bytes_processed = sum(_type_width(dt) * rows for dt in lowered.schema.values())
    implementations: list[BenchmarkImplementationRow] = []
    diagnostics: list[str] = []
    vectorized_api = None
    scalar_api = None
    gbench: dict[str, Any] | None = None

    def measure_prepare_for_display(policy) -> float:
        return measure_python_prepare_us(
            policy,
            outputs,
            inputs,
            output,
            arch,
            warmups,
            max(1, min(runs, 10)),
        )

    def measure_native_compile_for_display(policy) -> dict[str, Any]:
        return dict(
            sj_ext.benchmark_hir_jit_compile(
                outputs,
                inputs,
                output,
                "asmjit",
                policy,
                "O1",
                arch,
                warmups,
                max(1, min(runs, 10)),
            )
        )

    if provider in {"python", "all"}:
        try:
            vectorized_compile_us = measure_prepare_for_display(
                sj.CompilePolicy.Vectorized
            )
            vectorized_api = run_python_api_kernel(
                "Simjit SIMD / Python API",
                "asmjit_vectorized_py",
                sj.CompilePolicy.Vectorized,
                outputs,
                inputs,
                output,
                rows,
                bytes_processed,
                warmups,
                runs,
                arch,
                compile_us=vectorized_compile_us,
            )
            implementations.append(vectorized_api["row"])
        except Exception as exc:
            diagnostics.append(f"Simjit SIMD / Python API skipped: {exc}")
        scalar_compile_us = measure_prepare_for_display(sj.CompilePolicy.Scalar)
        scalar_api = run_python_api_kernel(
            "Simjit scalar / Python API",
            "asmjit_scalar_py",
            sj.CompilePolicy.Scalar,
            outputs,
            inputs,
            output,
            rows,
            bytes_processed,
            warmups,
            runs,
            arch,
            compile_us=scalar_compile_us,
        )
        implementations.append(scalar_api["row"])

    if provider in {"google", "all"}:
        scalar_inspection = sj.inspect(
            lowered.program,
            _schema_for_inspection(lowered),
            output=output,
            policy="scalar",
            arch=arch,
        )
        scalar_compile_measurement = measure_native_compile_for_display(
            sj.CompilePolicy.Scalar
        )
        scalar_compile_us = float(scalar_compile_measurement["compile_us"])
        vectorized_inspection = None
        vectorized_compile_us: float | None = None
        jit_compile_measurements = {
            "asmjit_scalar": scalar_compile_measurement,
        }
        try:
            vectorized_inspection = sj.inspect(
                lowered.program,
                _schema_for_inspection(lowered),
                output=output,
                policy="vectorized",
                arch=arch,
            )
            vectorized_compile_measurement = measure_native_compile_for_display(
                sj.CompilePolicy.Vectorized
            )
            vectorized_compile_us = float(vectorized_compile_measurement["compile_us"])
            jit_compile_measurements["asmjit_vectorized"] = (
                vectorized_compile_measurement
            )
        except Exception as exc:
            diagnostics.append(f"SIMD compile timing skipped: {exc}")
        asmjit_variants = []
        if vectorized_inspection is not None:
            try:
                asmjit_variants.append(
                    capture_asmjit_machine_code(
                        "Simjit SIMD / AsmJit",
                        "asmjit_vectorized",
                        sj.CompilePolicy.Vectorized,
                        outputs,
                        inputs,
                        output,
                        arch,
                        vectorized_compile_us,
                    )
                )
            except Exception as exc:
                diagnostics.append(f"Simjit SIMD / AsmJit skipped: {exc}")
        asmjit_variants.append(
            capture_asmjit_machine_code(
                "Simjit scalar / AsmJit",
                "asmjit_scalar",
                sj.CompilePolicy.Scalar,
                outputs,
                inputs,
                output,
                arch,
                scalar_compile_us,
            )
        )
        gbench = run_google_benchmark(
            vectorized_inspection,
            scalar_inspection,
            asmjit_variants=asmjit_variants,
            rows=rows,
            repetitions=max(1, min(runs, 10)),
            bytes_processed=bytes_processed,
            jit_compile_measurements=jit_compile_measurements,
        )
        if gbench["ok"]:
            implementations.extend(gbench["implementations"])
        diagnostics.extend(gbench.get("diagnostics", []))

    implementations = _ordered_implementation_rows(implementations)
    _add_bar_widths(implementations)
    representative = (
        vectorized_api["row"]
        if vectorized_api
        else (
            scalar_api["row"]
            if scalar_api
            else (implementations[0] if implementations else {})
        )
    )
    compile_us = representative.get("compile_us")
    median_us = representative.get("hot_us")
    return BenchmarkRunResult(
        rows=rows,
        warmups=warmups,
        runs=runs,
        null_density=null_density,
        output=output,
        arch=arch,
        provider=provider,
        provider_label=PROVIDER_LABELS[provider],
        compile_us=compile_us,
        median_us=median_us,
        compile_ms=compile_us / 1000.0 if compile_us else None,
        median_ms=median_us / 1000.0 if median_us else None,
        rows_per_second=representative.get("rows_per_second"),
        throughput_gbps=representative.get("throughput_gbps"),
        cache_hits=(vectorized_api["cache_hits"] if vectorized_api else 0)
        + (scalar_api["cache_hits"] if scalar_api else 0),
        cache_misses=(vectorized_api["cache_misses"] if vectorized_api else 0)
        + (scalar_api["cache_misses"] if scalar_api else 0),
        scalar_outputs=tuple(sorted(scalar_outputs)),
        implementations=tuple(implementations),
        diagnostics=tuple(diagnostics),
        google_benchmark=gbench,
    )


def scalar_output_names(outputs) -> set[str]:
    return {
        name
        for name, expr in outputs
        if isinstance(expr, SCALAR_OUTPUT_TYPES)
    }


def capture_asmjit_machine_code(
    name: str,
    key: str,
    policy,
    outputs,
    inputs,
    output: str,
    arch: str,
    compile_us: float | None,
) -> dict[str, Any]:
    session = sj.Session(arch=arch)
    session.debug_options.capture_on_success = True
    session.debug_options.stages = sj.DebugStage.MachineCode
    if policy is not None:
        session.policy = policy
    session.prepare_program(outputs, inputs, output)
    return {
        "name": name,
        "key": key,
        "code": bytes(session.debug_snapshot.machine_code),
        "compile_us": compile_us,
    }


def run_python_api_kernel(
    name: str,
    key: str,
    policy,
    outputs,
    inputs,
    output: str,
    rows: int,
    bytes_processed: float,
    warmups: int,
    runs: int,
    arch: str,
    compile_us: float | None = None,
) -> dict[str, Any]:
    session = sj.Session(arch=arch)
    if policy is not None:
        session.policy = policy
    start_compile = time.perf_counter()
    kernel = session.prepare_program(outputs, inputs, output)
    compile_seconds = time.perf_counter() - start_compile

    for _ in range(warmups):
        kernel.run()

    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        kernel.run()
        timings.append(time.perf_counter() - start)

    median = _median(timings)
    stats = session.statistics()
    measured_compile_us = compile_seconds * 1_000_000.0
    reported_compile_us = measured_compile_us if compile_us is None else compile_us
    median_us = median * 1_000_000.0
    return {
        "compile_us": reported_compile_us,
        "measured_compile_us": measured_compile_us,
        "median_us": median_us,
        "cache_hits": stats.cache_hits,
        "cache_misses": stats.cache_misses,
        "row": _implementation_row(
            name,
            key,
            median_us,
            rows,
            bytes_processed,
            compile_us=reported_compile_us,
            compile_boundary="python-program-to-prepared-kernel",
            source="Python API",
        ),
    }


def measure_python_prepare_us(
    policy,
    outputs,
    inputs,
    output: str,
    arch: str,
    warmups: int,
    runs: int,
) -> float:
    def prepare_once() -> float:
        session = sj.Session(arch=arch)
        if policy is not None:
            session.policy = policy
        start = time.perf_counter()
        session.prepare_program(outputs, inputs, output)
        return time.perf_counter() - start

    for _ in range(max(1, warmups)):
        prepare_once()
    timings = [prepare_once() for _ in range(max(1, runs))]
    return _median(timings) * 1_000_000.0


def _implementation_row(
    name: str,
    key: str,
    hot_us: float,
    rows: int,
    bytes_processed: float,
    *,
    compile_us: float | None,
    compile_boundary: str | None = None,
    source: str,
) -> BenchmarkImplementationRow:
    seconds = hot_us / 1_000_000.0
    metadata = _implementation_metadata(name, key)
    return BenchmarkImplementationRow(
        name=name,
        key=key,
        source=source,
        api=metadata["api"],
        compile_mode=metadata["compile_mode"],
        backend=metadata["backend"],
        opt=metadata["opt"],
        chart_label=_implementation_chart_label(metadata),
        main=_is_main_implementation(key),
        hot_us=hot_us,
        compile_us=compile_us,
        compile_boundary=compile_boundary,
        rows_per_second=rows / seconds if seconds > 0 else None,
        throughput_gbps=bytes_processed / seconds / 1_000_000_000
        if seconds > 0
        else None,
    )


def _implementation_metadata(name: str, key: str) -> dict[str, str]:
    metadata = IMPLEMENTATION_METADATA.get(key)
    if metadata is not None:
        return dict(metadata)

    lowered_name = name.lower()
    backend = "Other"
    if "asmjit" in lowered_name or "python api" in lowered_name:
        backend = "AsmJit"
    elif "c++" in lowered_name:
        backend = "C++ emitter"
    elif "llvm" in lowered_name:
        backend = "LLVM"

    opt = "O3" if "o3" in lowered_name else ("O1" if backend in {"C++ emitter", "LLVM"} else "-")
    return {
        "api": "Python" if "python api" in lowered_name else "C++",
        "compile_mode": "SIMD" if "simd" in lowered_name else "Scalar",
        "backend": backend,
        "opt": opt,
    }


def _implementation_chart_label(metadata: dict[str, str]) -> str:
    parts = [metadata["api"], metadata["compile_mode"], metadata["backend"]]
    if metadata["opt"] != "-":
        parts.append(metadata["opt"])
    return " · ".join(parts)


def _is_main_implementation(key: str) -> bool:
    return key.startswith("asmjit_")


def _add_bar_widths(rows: list[BenchmarkImplementationRow]) -> None:
    hot_values = [row["hot_us"] for row in rows if row.get("hot_us")]
    fastest = min(hot_values) if hot_values else None
    throughput_values = [
        row["throughput_gbps"]
        for row in rows
        if row.get("throughput_gbps") is not None and row["throughput_gbps"] > 0
    ]
    rows_per_second_values = [
        row["rows_per_second"]
        for row in rows
        if row.get("rows_per_second") is not None and row["rows_per_second"] > 0
    ]
    max_throughput = max(throughput_values) if throughput_values else None
    max_rows_per_second = (
        max(rows_per_second_values) if rows_per_second_values else None
    )
    for row in rows:
        hot_us = row.get("hot_us")
        if hot_us and fastest:
            row["speedup"] = hot_us / fastest if fastest > 0 else None
        throughput = row.get("throughput_gbps")
        if throughput is not None and max_throughput:
            row["bar_percent"] = max(3.0, throughput / max_throughput * 100.0)
            row["bar_label"] = f"{throughput:.2f} GB/s"
            continue
        rows_per_second = row.get("rows_per_second")
        if rows_per_second is not None and max_rows_per_second:
            row["bar_percent"] = max(3.0, rows_per_second / max_rows_per_second * 100.0)
            row["bar_label"] = f"{rows_per_second / 1_000_000.0:.2f} M rows/s"


def _ordered_implementation_rows(rows: list[BenchmarkImplementationRow]) -> list[BenchmarkImplementationRow]:
    return [
        row
        for _, row in sorted(
            enumerate(rows),
            key=lambda item: (
                _backend_order(item[1]),
                _mode_order(item[1]),
                _api_order(item[1]),
                _opt_order(item[1]),
                item[0],
            ),
        )
    ]


def _backend_order(row: BenchmarkImplementationRow) -> int:
    backend = row.get("backend")
    if backend in BACKEND_ORDER:
        return BACKEND_ORDER[backend]
    name = str(row.get("name", "")).lower()
    if "asmjit" in name or "python api" in name:
        return BACKEND_ORDER["AsmJit"]
    if "c++" in name:
        return BACKEND_ORDER["C++ emitter"]
    if "llvm" in name:
        return BACKEND_ORDER["LLVM"]
    return 3


def _mode_order(row: BenchmarkImplementationRow) -> int:
    compile_mode = row.get("compile_mode")
    if compile_mode in COMPILE_MODE_ORDER:
        return COMPILE_MODE_ORDER[compile_mode]
    name = str(row.get("name", "")).lower()
    if "simd" in name:
        return COMPILE_MODE_ORDER["SIMD"]
    if "scalar" in name:
        return COMPILE_MODE_ORDER["Scalar"]
    return 2


def _api_order(row: BenchmarkImplementationRow) -> int:
    api = row.get("api")
    if api in API_ORDER:
        return API_ORDER[api]
    name = str(row.get("name", "")).lower()
    return API_ORDER["Python"] if "python api" in name else API_ORDER["C++"]


def _opt_order(row: BenchmarkImplementationRow) -> int:
    opt = str(row.get("opt", "-"))
    if opt == "-":
        return 0
    if opt == "O1":
        return 1
    if opt == "O3":
        return 2
    return 3


def run_google_benchmark(
    vectorized_inspection,
    scalar_inspection,
    *,
    asmjit_variants: list[dict[str, Any]],
    rows: int,
    repetitions: int,
    bytes_processed: float,
    jit_compile_measurements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    variants = [
        *[
            (item["name"], item["key"], "asm", item["code"], False, item["compile_us"])
            for item in asmjit_variants
        ],
    ]
    if vectorized_inspection is not None:
        variants.extend(
            [
                (
                    "Simjit SIMD / C++",
                    "cpp_vectorized",
                    "cpp",
                    vectorized_inspection.cpp.strip(),
                    False,
                    None,
                ),
                (
                    "Simjit SIMD / LLVM",
                    "llvm_vectorized",
                    "llvm",
                    vectorized_inspection.llvm_ir.strip(),
                    False,
                    None,
                ),
            ]
        )
    variants.extend(
        [
            (
                "Simjit scalar / C++",
                "cpp_scalar",
                "cpp",
                scalar_inspection.cpp.strip(),
                False,
                None,
            ),
            (
                "Simjit scalar / C++ O3",
                "cpp_scalar_o3",
                "cpp",
                scalar_inspection.cpp.strip(),
                True,
                None,
            ),
            (
                "Simjit scalar / LLVM",
                "llvm_scalar",
                "llvm",
                scalar_inspection.llvm_ir.strip(),
                False,
                None,
            ),
            (
                "Simjit scalar / LLVM O3",
                "llvm_scalar_o3",
                "llvm",
                scalar_inspection.llvm_ir.strip(),
                True,
                None,
            ),
        ]
    )
    diagnostics: list[str] = []
    if not any(code for _, _, _, code, _, _ in variants):
        return {
            "ok": False,
            "diagnostics": ["Google Benchmark skipped: emitters returned no code"],
            "implementations": [],
        }

    gxx = shutil.which("g++")
    clang = os.environ.get("LLVM_CLANG") or shutil.which("clang")
    if not gxx:
        return {
            "ok": False,
            "diagnostics": ["Google Benchmark skipped: g++ was not found"],
            "implementations": [],
        }

    try:
        param_types = gbench_pointer_types(variants, scalar_inspection.hir)
    except ValueError as exc:
        return {
            "ok": False,
            "diagnostics": [f"Google Benchmark skipped: {exc}"],
            "implementations": [],
        }

    with tempfile.TemporaryDirectory(prefix="simjit-explorer-bench-") as tmp:
        tmp_dir = Path(tmp)
        objects: list[str] = []
        symbols: list[tuple[str, str]] = []
        compile_times: dict[str, tuple[float, str]] = {}

        try:
            for label, symbol, kind, code, opt_o3, compile_us in variants:
                if not code:
                    diagnostics.append(
                        f"Google Benchmark skipped {label}: emitter returned no code"
                    )
                    continue
                try:
                    if kind == "asm":
                        asm_src = tmp_dir / f"{symbol}.c"
                        asm_obj = tmp_dir / f"{symbol}.o"
                        asm_src.write_text(
                            wrap_asmjit_machine_code(code, symbol), encoding="utf-8"
                        )
                        run_cmd([gxx, "-c", str(asm_src), "-o", str(asm_obj)])
                        objects.append(str(asm_obj))
                        symbols.append((label, symbol))
                        if compile_us is not None:
                            compile_times[symbol] = (
                                float(compile_us),
                                "constructed-hir-to-executable-pointer",
                            )
                    elif kind == "cpp":
                        cpp_src = tmp_dir / f"{symbol}.cpp"
                        cpp_obj = tmp_dir / f"{symbol}.o"
                        cpp_src.write_text(
                            wrap_cpp_source(rename_cpp_function(code, symbol)),
                            encoding="utf-8",
                        )
                        run_cmd(
                            [
                                gxx,
                                "-std=c++20",
                                *cpp_compile_flags(opt_o3),
                                "-fwrapv",
                                "-march=native",
                                "-c",
                                str(cpp_src),
                                "-o",
                                str(cpp_obj),
                            ]
                        )
                        objects.append(str(cpp_obj))
                        symbols.append((label, symbol))
                    elif kind == "llvm" and clang:
                        llvm_src = tmp_dir / f"{symbol}.ll"
                        llvm_obj = tmp_dir / f"{symbol}.o"
                        llvm_src.write_text(
                            rename_llvm_function(code, symbol), encoding="utf-8"
                        )
                        run_cmd(
                            [
                                clang,
                                *llvm_clang_compile_flags(opt_o3),
                                "-c",
                                str(llvm_src),
                                "-o",
                                str(llvm_obj),
                            ]
                        )
                        objects.append(str(llvm_obj))
                        symbols.append((label, symbol))
                    elif kind == "llvm":
                        diagnostics.append(
                            f"Google Benchmark skipped {label}: clang was not found"
                        )
                except Exception as exc:
                    diagnostics.append(f"Google Benchmark skipped {label}: {exc}")

            if not symbols:
                return {
                    "ok": False,
                    "diagnostics": diagnostics
                    or ["Google Benchmark had no compilable rows"],
                    "implementations": [],
                }

            bench_src = tmp_dir / "bench.cpp"
            bench_bin = tmp_dir / "bench"
            bench_json = tmp_dir / "bench.json"
            bench_src.write_text(
                generate_gbench_source(param_types, symbols, rows), encoding="utf-8"
            )
            link_cmd = [
                gxx,
                "-std=c++20",
                "-O1",
                "-fwrapv",
                "-march=native",
                *objects,
                str(bench_src),
                "-o",
                str(bench_bin),
                *benchmark_include_lib_flags(),
                "-lbenchmark",
                "-lpthread",
            ]
            run_cmd(link_cmd)
            run_cmd(
                [
                    str(bench_bin),
                    "--benchmark_format=json",
                    "--benchmark_out=" + str(bench_json),
                    "--benchmark_out_format=json",
                    "--benchmark_repetitions=" + str(repetitions),
                    "--benchmark_report_aggregates_only=false",
                    "--benchmark_min_time=0.01",
                ],
                timeout=60,
            )
            raw = json.loads(bench_json.read_text(encoding="utf-8"))
            implementations = _ordered_implementation_rows(
                parse_gbench_results(raw, symbols, rows, bytes_processed)
            )
            vectorized_ir = (
                vectorized_inspection.llvm_ir.strip()
                if vectorized_inspection is not None
                else ""
            )
            jit_compile_measurements.update(
                probe_llvm_compile_times(
                    vectorized_ir,
                    scalar_inspection.llvm_ir.strip(),
                    diagnostics,
                )
            )
            for row in implementations:
                measurement = jit_compile_measurements.get(row["key"])
                if measurement is not None:
                    row["compile_us"] = float(measurement["compile_us"])
                    row["compile_samples_us"] = [
                        float(value)
                        for value in measurement["compile_samples_us"]
                    ]
                    row["compile_warmups"] = int(measurement["compile_warmups"])
                    row["compile_runs"] = int(measurement["compile_runs"])
                    row["compile_boundary"] = str(measurement["compile_boundary"])
                elif row["key"] in compile_times:
                    row["compile_us"], row["compile_boundary"] = compile_times[
                        row["key"]
                    ]
            return {
                "ok": True,
                "diagnostics": diagnostics,
                "implementations": implementations,
            }
        except Exception as exc:
            diagnostics.append(f"Google Benchmark unavailable: {exc}")
            return {"ok": False, "diagnostics": diagnostics, "implementations": []}


def probe_llvm_compile_times(
    vectorized_ir: str, scalar_ir: str, diagnostics: list[str]
) -> dict[str, dict[str, Any]]:
    if not vectorized_ir and not scalar_ir:
        return {}
    try:
        probe = importlib.import_module("explorer_llvm_probe")
    except Exception as exc:
        diagnostics.append(f"LLVM compile timing unavailable: {exc}")
        return {}

    timings: dict[str, dict[str, Any]] = {}
    jobs = [
        ("llvm_scalar", scalar_ir, LLVM_BASELINE_PROBE_OPT),
        ("llvm_scalar_o3", scalar_ir, LLVM_AGGRESSIVE_OPT),
        ("llvm_vectorized", vectorized_ir, LLVM_BASELINE_PROBE_OPT),
    ]
    for key, llvm_ir, opt in jobs:
        if not llvm_ir:
            continue
        try:
            warmup = probe.compile_ir(llvm_ir, opt=opt)
            samples = [
                float(probe.compile_ir(llvm_ir, opt=opt)["compile_us"])
                for _ in range(5)
            ]
            timings[key] = {
                "compile_us": _median(samples),
                "compile_samples_us": samples,
                "compile_warmups": 1,
                "compile_runs": len(samples),
                "compile_boundary": str(warmup["compile_boundary"]),
                "backend": "llvm",
                "llvm_opt": opt,
            }
        except Exception as exc:
            diagnostics.append(f"LLVM compile timing failed for {key}: {exc}")
    return timings


def llvm_clang_compile_flags(opt_o3: bool) -> list[str]:
    # Baseline LLVM rows should show Simjit's chosen lowering, not LLVM's
    # loop-vectorized version of scalar IR. O3 is the explicit compiler path.
    opt = LLVM_AGGRESSIVE_OPT if opt_o3 else LLVM_BASELINE_OPT
    flags = ["-ffast-math", f"-{opt}"]
    march = llvm_target_march()
    if march:
        flags.append(f"-march={march}")
    return flags


def llvm_target_march() -> str | None:
    value = os.environ.get(LLVM_TARGET_MARCH_ENV)
    if value is not None:
        value = value.strip()
        if not value or value.lower() in {"none", "generic", "off"}:
            return None
        return llvm_cpu_alias(value)

    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64"}:
        return "native"
    return None


def llvm_cpu_alias(cpu: str) -> str:
    if cpu.lower() == "zen4":
        return "znver4"
    return cpu


def cpp_compile_flags(opt_o3: bool) -> list[str]:
    opt = CPP_AGGRESSIVE_OPT if opt_o3 else CPP_BASELINE_OPT
    return [f"-{opt}"]


def parse_cpp_pointer_types(cpp: str) -> list[str]:
    match = CPP_PARAM_RE.search(cpp)
    if not match:
        raise ValueError("could not parse emitted C++ function signature")
    params = [part.strip() for part in match.group("params").split(",") if part.strip()]
    types: list[str] = []
    for param in params:
        type_match = CPP_POINTER_PARAM_RE.match(param)
        if not type_match:
            raise ValueError(
                f"expected pointer argument in C++ signature, got {param!r}"
            )
        types.append(type_match.group("type").strip())
    return types


def parse_hir_pointer_types(hir: str) -> list[str]:
    types: list[str] = []
    for match in HIR_ARG_RE.finditer(hir):
        dtype = match.group("dtype")
        if dtype not in DTYPE_TO_CPP:
            raise ValueError(f"unsupported HIR argument dtype {dtype!r}")
        types.append(DTYPE_TO_CPP[dtype])
    if not types:
        raise ValueError("could not parse HIR argument signature")
    return types


def gbench_pointer_types(
    variants: list[tuple[str, str, str, Any, bool, float | None]],
    hir: str,
) -> list[str]:
    signature_cpp = next(
        (code for _, _, kind, code, _, _ in variants if kind == "cpp" and code), ""
    )
    if signature_cpp:
        return parse_cpp_pointer_types(signature_cpp)
    return parse_hir_pointer_types(hir)


def rename_cpp_function(cpp: str, name: str) -> str:
    return re.sub(r"\bvoid\s+expr\s*\(", f"void {name}(", cpp, count=1)


def rename_llvm_function(llvm_ir: str, name: str) -> str:
    return llvm_ir.replace("@expr(", f"@{name}(", 1)


def wrap_cpp_source(cpp: str) -> str:
    return f"""#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <bit>
#include <math.h>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#define bit_cast std::bit_cast

extern "C" {{
{cpp}
}}
"""


def wrap_asmjit_machine_code(code: bytes, symbol: str) -> str:
    bytes_per_line = 12
    chunks = []
    for offset in range(0, len(code), bytes_per_line):
        chunk = ", ".join(
            f"0x{byte:02x}" for byte in code[offset : offset + bytes_per_line]
        )
        chunks.append(f"  {chunk},")
    body = "\n".join(chunks) if chunks else "  0x00,"
    return f"""#if defined(__APPLE__)
__attribute__((section("__TEXT,__text"), aligned(16)))
#else
__attribute__((section(".text"), aligned(16)))
#endif
unsigned char {symbol}[] = {{
{body}
}};
"""


def generate_gbench_source(
    param_types: list[str], symbols: list[tuple[str, str]], rows: int
) -> str:
    pointer_types = ", ".join(f"{ty}*" for ty in param_types)
    declarations = "\n".join(
        f'extern "C" void {symbol}(size_t, {pointer_types});' for _, symbol in symbols
    )
    buffers = "\n".join(
        f"  auto arg{i} = make_buffer<{ty}>(rows);" for i, ty in enumerate(param_types)
    )
    fills = "\n".join(
        f"  fill_buffer(arg{i}.get(), rows, {i + 1}u);" for i in range(len(param_types))
    )
    args = ", ".join(f"arg{i}.get()" for i in range(len(param_types)))
    benches = "\n".join(
        f"""
static void B_{symbol}(benchmark::State& state) {{
  run_one(state, {symbol});
}}
BENCHMARK(B_{symbol})->Arg({rows});
"""
        for _, symbol in symbols
    )
    return f"""#include <benchmark/benchmark.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <type_traits>

{declarations}

template <class T>
struct FreeDeleter {{
  void operator()(T* ptr) const {{ free(ptr); }}
}};

template <class T>
using Buffer = std::unique_ptr<T, FreeDeleter<T>>;

template <class T>
Buffer<T> make_buffer(size_t rows) {{
  void* ptr = nullptr;
  if (posix_memalign(&ptr, 64, std::max<size_t>(1, rows) * sizeof(T)) != 0) {{
    throw std::bad_alloc();
  }}
  return Buffer<T>(static_cast<T*>(ptr));
}}

static uint32_t xorshift32(uint32_t* state) {{
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  return *state = x;
}}

template <class T>
void fill_buffer(T* data, size_t rows, uint32_t seed) {{
  uint32_t state = 0x12312fdaU ^ (seed * 0x9e3779b9U);
  for (size_t i = 0; i < rows; ++i) {{
    uint32_t value = xorshift32(&state);
    if constexpr (std::is_floating_point_v<T>) {{
      data[i] = static_cast<T>(static_cast<int32_t>(value) / 17.0);
    }} else {{
      data[i] = static_cast<T>(value);
    }}
  }}
}}

using func_dtype = void (*)(size_t, {pointer_types});

static void run_one(benchmark::State& state, func_dtype fn) {{
  size_t rows = static_cast<size_t>(state.range(0));
{buffers}
{fills}
  for (auto _ : state) {{
    fn(rows, {args});
    benchmark::ClobberMemory();
  }}
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(rows));
}}

{benches}

BENCHMARK_MAIN();
"""


def benchmark_include_lib_flags() -> list[str]:
    flags: list[str] = []
    for prefix in (Path("/opt/homebrew"), Path("/usr/local")):
        include = prefix / "include"
        lib = prefix / "lib"
        if (include / "benchmark" / "benchmark.h").exists():
            flags.append(f"-I{include}")
        if lib.exists():
            flags.append(f"-L{lib}")
    return flags


def run_cmd(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "command failed").strip()
        raise RuntimeError(f"{cmd[0]} failed: {detail}")
    return proc


def parse_gbench_results(
    raw: dict[str, Any],
    symbols: list[tuple[str, str]],
    rows: int,
    bytes_processed: float,
) -> list[BenchmarkImplementationRow]:
    results: list[BenchmarkImplementationRow] = []
    benchmarks = raw.get("benchmarks", [])
    for label, symbol in symbols:
        entries = [
            entry for entry in benchmarks if f"B_{symbol}" in entry.get("name", "")
        ]
        if not entries:
            continue
        chosen = next(
            (entry for entry in entries if entry.get("aggregate_name") == "median"),
            None,
        )
        if chosen is None:
            chosen = next(
                (entry for entry in entries if "aggregate_name" not in entry),
                entries[0],
            )
        hot_us = time_to_us(
            float(chosen.get("real_time", chosen.get("cpu_time", 0.0))),
            chosen.get("time_unit", "ns"),
        )
        row = _implementation_row(
            label,
            symbol,
            hot_us,
            rows,
            bytes_processed,
            compile_us=None,
            source="Google Benchmark",
        )
        if chosen.get("items_per_second"):
            row["rows_per_second"] = float(chosen["items_per_second"])
        results.append(row)
    return results


def time_to_us(value: float, unit: str) -> float:
    if unit == "ns":
        return value / 1000.0
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    return value


def _type_width(dt) -> float:
    if dt == sj.I1:
        return 1.0 / 8.0
    return float(numpy_dtype(dt).itemsize)


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
        print(json.dumps(to_json_dict(run(payload)), sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
