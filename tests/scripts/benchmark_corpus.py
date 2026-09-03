#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

"""Benchmark runtime bundle cases and write resumable structured JSONL."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
BENCHMARK_ROWS = 4096
ROOT = Path(__file__).resolve().parents[2]
LOCAL_RUNNER = Path(os.environ.get("SIMJIT_LOCAL_RUNNER", ROOT / "build/debug/local_runner"))
TIME_UNIT_TO_NS = {
    "ns": 1.0,
    "us": 1_000.0,
    "ms": 1_000_000.0,
    "s": 1_000_000_000.0,
}


def iter_bundle(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield item


def case_key(item: dict[str, Any]) -> str:
    return "|".join(
        (
            str(item.get("suite", "")),
            str(item.get("id", "")),
            str(item.get("variant", "")),
        )
    )


def completed_keys(path: Path, retry_errors: bool) -> set[str]:
    if not path.exists():
        return set()
    result: set[str] = set()
    good_end = 0
    with path.open("rb") as stream:
        while line := stream.readline():
            if not line.strip():
                good_end = stream.tell()
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if stream.read(1):
                    raise
                break
            good_end = stream.tell()
            if record.get("record_type") != "case":
                continue
            if retry_errors and record.get("status") != "success":
                continue
            key = record.get("key")
            if isinstance(key, str):
                result.add(key)
    if good_end != path.stat().st_size:
        with path.open("r+b") as stream:
            stream.truncate(good_end)
    return result


def benchmark_rows(
    raw: dict[str, Any],
    bundle_to_implementation: dict[str, str],
    rows: int = BENCHMARK_ROWS,
) -> list[dict[str, Any]]:
    result = []
    for row in raw.get("benchmarks", []):
        if row.get("run_type", "iteration") != "iteration":
            continue
        time_unit = str(row.get("time_unit", "ns"))
        if time_unit not in TIME_UNIT_TO_NS:
            raise ValueError(f"unsupported Google Benchmark time unit: {time_unit}")
        scale = TIME_UNIT_TO_NS[time_unit]
        cpu_time_ns = float(row["cpu_time"]) * scale
        real_time_ns = float(row["real_time"]) * scale
        benchmark_name = str(row["name"])
        bundle_name = benchmark_name.rsplit("/", 1)[-1]
        result.append(
            {
                "name": benchmark_name,
                "function_name": bundle_name,
                "implementation": bundle_to_implementation.get(
                    bundle_name,
                    "unknown",
                ),
                "iterations": int(row["iterations"]),
                "cpu_time_ns": cpu_time_ns,
                "real_time_ns": real_time_ns,
                "loop_throughput_rows_per_ns": rows / cpu_time_ns,
            }
        )
    return result


def case_record(
    item: dict[str, Any],
    raw: dict[str, Any],
    code_names: list[str],
    bundle_to_implementation: dict[str, str],
    minimum_time_seconds: float,
    repetitions: int,
    include_o3: bool,
) -> dict[str, Any]:
    vectorized = str(item.get("variant", "")).endswith("-vector")
    available_implementations = set(
        implementation_names(code_names, include_o3, vectorized=vectorized)
    )
    execution_paths = {}
    for implementation in available_implementations:
        if implementation == "asmjit":
            execution_paths[implementation] = (
                "automatic-vectorized"
                if vectorized
                else "automatic-scalar-fallback"
            )
        elif implementation == "asmjit-scalar":
            execution_paths[implementation] = "forced-scalar"
        elif not vectorized:
            execution_paths[implementation] = "automatic-scalar-fallback"
        elif "simd" in implementation or implementation.startswith("cpp-o"):
            execution_paths[implementation] = "simd-input"
        else:
            execution_paths[implementation] = "forced-scalar"
    return {
        "record_type": "case",
        "schema_version": SCHEMA_VERSION,
        "key": case_key(item),
        "status": "success",
        "n": item.get("n"),
        "id": item.get("id"),
        "suite": item.get("suite"),
        "suite_idx": item.get("suite_idx"),
        "case_idx": item.get("case_idx"),
        "variant": item.get("variant"),
        "expected": item.get("expected"),
        "file": item.get("file"),
        "line": item.get("line"),
        "schema": item.get("schema"),
        "ir": {
            "hir": {"text": item.get("src", "")},
            "mir": {"text": item.get("mir", "")},
        },
        "benchmark": {
            "code_names": code_names,
            "execution_paths": execution_paths,
            "row_count": BENCHMARK_ROWS,
            "minimum_time_seconds": minimum_time_seconds,
            "repetitions": repetitions,
            "include_o3": include_o3,
            "context": raw.get("context", {}),
            "measurements": benchmark_rows(raw, bundle_to_implementation),
        },
    }


def error_record(item: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        "record_type": "case",
        "schema_version": SCHEMA_VERSION,
        "key": case_key(item),
        "status": "error",
        "n": item.get("n"),
        "id": item.get("id"),
        "suite": item.get("suite"),
        "variant": item.get("variant"),
        "file": item.get("file"),
        "line": item.get("line"),
        "ir": {
            "hir": {"text": item.get("src", "")},
            "mir": {"text": item.get("mir", "")},
        },
        "error": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
    }


def implementation_names(
    code_names: list[str],
    include_o3: bool,
    *,
    vectorized: bool = True,
) -> list[str]:
    result = []
    for code_name in code_names:
        if code_name == "asmjit_s":
            result.append("asmjit-scalar")
        elif code_name == "llvm_s":
            result.append("llvm-o1")
            if include_o3:
                result.append("llvm-o3")
        elif code_name == "cpp_s":
            result.append("cpp-scalar-o1")
            if include_o3:
                result.append("cpp-scalar-o3")
        elif code_name == "llvm":
            prefix = "llvm-simd" if vectorized else "llvm"
            result.append(f"{prefix}-o1")
            if include_o3:
                result.append(f"{prefix}-o3")
        elif code_name == "cpp":
            prefix = "cpp" if vectorized else "cpp-scalar"
            result.append(f"{prefix}-o1")
            if include_o3:
                result.append(f"{prefix}-o3")
        else:
            result.append(code_name)
    return result


def bundle_implementations(
    code_names: list[str],
    include_o3: bool,
    *,
    vectorized: bool,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for code_name in code_names:
        if code_name == "asmjit":
            result["asmjit"] = "asmjit"
        elif code_name == "asmjit_s":
            result["asmjit_s"] = "asmjit-scalar"
        elif code_name == "llvm":
            prefix = "llvm-simd" if vectorized else "llvm"
            result["llvm"] = f"{prefix}-o1"
            if include_o3:
                result["llvm_o3"] = f"{prefix}-o3"
        elif code_name == "llvm_s":
            result["llvm_s"] = "llvm-o1"
            if include_o3:
                result["llvm_s_o3"] = "llvm-o3"
        elif code_name == "cpp":
            prefix = "cpp" if vectorized else "cpp-scalar"
            result["cpp"] = f"{prefix}-o1"
            if include_o3:
                result["cpp_o3"] = f"{prefix}-o3"
        elif code_name == "cpp_s":
            result["cpp_s"] = "cpp-scalar-o1"
            if include_o3:
                result["cpp_s_o3"] = "cpp-scalar-o3"
        else:
            result[code_name] = code_name
    return result


def run_native_benchmark(
    item: dict[str, Any],
    code_names: list[str],
    *,
    minimum_time_seconds: float,
    repetitions: int,
    include_o3: bool,
) -> dict[str, Any]:
    if not LOCAL_RUNNER.is_file():
        raise RuntimeError(f"missing native local runner: {LOCAL_RUNNER}")
    with tempfile.TemporaryDirectory(prefix="simjit-benchmark-corpus-") as directory:
        temp = Path(directory)
        bundle = temp / "case.jsonl"
        output = temp / "benchmark.json"
        bundle.write_text(json.dumps(item, sort_keys=True) + "\n", encoding="utf-8")
        command = [
            str(LOCAL_RUNNER),
            "--bench",
            "--bench-o3",
            "all" if include_o3 else "none",
            "--file",
            str(bundle),
            "--rows",
            str(BENCHMARK_ROWS),
            f"--benchmark_out={output}",
            "--benchmark_out_format=json",
            f"--benchmark_min_time={minimum_time_seconds:g}s",
            f"--benchmark_repetitions={repetitions}",
            "--benchmark_report_aggregates_only=false",
        ]
        for code_name in code_names:
            command.extend(("--code", code_name))
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "native local runner failed").strip()
            raise RuntimeError(detail)
        if not output.is_file():
            raise RuntimeError("native local runner produced no Google Benchmark JSON")
        return json.loads(output.read_text(encoding="utf-8"))


def selected_code_names(
    item: dict[str, Any], requested: list[str]
) -> list[str]:
    available = {
        str(code.get("name", "")) for code in item.get("codes", [])
    }
    selected = [name for name in requested if name in available]
    if str(item.get("variant", "")).endswith("-vector"):
        missing_pairs = []
        for vector_name, scalar_name in (
            ("asmjit", "asmjit_s"),
            ("llvm", "llvm_s"),
            ("cpp", "cpp_s"),
        ):
            pair = {vector_name, scalar_name}
            if pair.issubset(requested) and not pair.issubset(selected):
                missing_pairs.extend(sorted(pair - set(selected)))
        if missing_pairs:
            raise ValueError(
                "vectorized case lacks requested paired artifacts: "
                + ",".join(missing_pairs)
            )
    return selected


def run_header(args: argparse.Namespace, code_names: list[str]) -> dict[str, Any]:
    return {
        "record_type": "run",
        "schema_version": SCHEMA_VERSION,
        "producer": "simjit-native-bundle-benchmark",
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "runner": {"command": str(LOCAL_RUNNER)},
        "settings": {
            "code_names": code_names,
            "implementations": implementation_names(
                code_names,
                not args.no_o3,
            ),
            "row_count": BENCHMARK_ROWS,
            "minimum_time_seconds": args.minimum_time,
            "repetitions": args.repetitions,
            "include_o3": not args.no_o3,
            "native_compile_flags": "in-process Clang/LLVM O1 and O3, -fwrapv, native target",
            "vectorized_only": not args.include_fallbacks,
            "suites": sorted(args.suite),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-time", type=float, default=0.5)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--code-name", action="append", default=[])
    parser.add_argument("--suite", action="append", default=[])
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-fallbacks", action="store_true")
    parser.add_argument("--no-o3", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_time <= 0:
        raise SystemExit("--minimum-time must be positive")
    if args.repetitions <= 0:
        raise SystemExit("--repetitions must be positive")
    if args.first < 0 or args.limit < 0:
        raise SystemExit("--first and --limit must be non-negative")
    if args.output.exists() and not args.resume:
        raise SystemExit(f"output already exists; use --resume: {args.output}")

    code_names = args.code_name or ["asmjit", "llvm", "cpp"]
    selected_suites = set(args.suite)
    done = completed_keys(args.output, args.retry_errors) if args.resume else set()
    mode = (
        "a"
        if args.output.exists()
        and args.output.stat().st_size > 0
        and args.resume
        else "w"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    with args.output.open(mode, encoding="utf-8") as output:
        if mode == "w":
            output.write(
                json.dumps(run_header(args, code_names), sort_keys=True) + "\n"
            )
            output.flush()

        for item in iter_bundle(args.file):
            item_n = int(item.get("n", -1))
            if item_n < args.first:
                continue
            if selected_suites and item.get("suite") not in selected_suites:
                continue
            if not args.include_fallbacks and not str(
                item.get("variant", "")
            ).endswith("-vector"):
                continue
            key = case_key(item)
            if key in done:
                continue

            print(
                f"[{processed + 1}] {key}",
                file=sys.stderr,
                flush=True,
            )
            try:
                case_code_names = selected_code_names(item, code_names)
                if not case_code_names:
                    raise ValueError(
                        "bundle case has none of the requested code names: "
                        + ",".join(code_names)
                    )
                vectorized = str(item.get("variant", "")).endswith("-vector")
                bundle_to_implementation = bundle_implementations(
                    case_code_names,
                    not args.no_o3,
                    vectorized=vectorized,
                )
                raw = run_native_benchmark(
                    item,
                    case_code_names,
                    minimum_time_seconds=args.minimum_time,
                    repetitions=args.repetitions,
                    include_o3=not args.no_o3,
                )
                observed = {
                    row["implementation"]
                    for row in benchmark_rows(raw, bundle_to_implementation)
                }
                expected = set(
                    implementation_names(
                        case_code_names,
                        not args.no_o3,
                        vectorized=vectorized,
                    )
                )
                if observed != expected:
                    raise ValueError(
                        "unexpected benchmark implementation coverage: "
                        f"expected={sorted(expected)} "
                        f"observed={sorted(observed)}"
                    )
                record = case_record(
                    item,
                    raw,
                    case_code_names,
                    bundle_to_implementation,
                    args.minimum_time,
                    args.repetitions,
                    not args.no_o3,
                )
            except Exception as exc:
                record = error_record(item, exc)
                print(f"{key}: {exc}", file=sys.stderr, flush=True)
                if args.fail_fast:
                    output.write(json.dumps(record, sort_keys=True) + "\n")
                    output.flush()
                    raise

            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
            processed += 1
            if args.limit > 0 and processed >= args.limit:
                break


if __name__ == "__main__":
    main()
