#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from explorer.app import bench_worker
from explorer.app.benchmark_model import BenchmarkRunResult, to_json_dict
from explorer.app.samples import SAMPLES, Sample


def main() -> int:
    args = parse_args()
    selected = select_samples(args.sample, args.limit)

    if args.dummy:
        install_dummy_google_benchmark()

    started = time.perf_counter()
    cases = []
    for index, sample in enumerate(selected, start=1):
        print(
            f"[{index}/{len(selected)}] {sample.id}: {sample.title}",
            file=sys.stderr,
            flush=True,
        )
        cases.append(run_sample(sample, args))

    document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "host": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "config": {
            "rows": args.rows,
            "warmups": args.warmups,
            "runs": args.runs,
            "null_density": args.null_density,
            "output": args.output,
            "provider": "google",
            "arch": "native",
            "dummy": args.dummy,
            "sample_count": len(selected),
        },
        "elapsed_wall_s": time.perf_counter() - started,
        "ok": all(case["ok"] for case in cases),
        "cases": cases,
    }

    encoded = json.dumps(document, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    return 1 if args.strict and not document["ok"] else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run every benchmarkable Explorer sample on the host and emit one JSON "
            "document containing Google Benchmark rows and compile timings."
        )
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=[],
        help=(
            "Explorer sample id to run. Repeat to run multiple ids. Defaults "
            "to all benchmarkable samples."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected samples. Useful for smoke checks.",
    )
    parser.add_argument("--rows", type=int, default=100_000)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--null-density", type=float, default=0.1)
    parser.add_argument("--output", choices=("pyarrow", "numpy"), default="pyarrow")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help=(
            "Exercise the batch runner without compiling/linking/running Google "
            "Benchmark. Simjit lowering and compile timing still run."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with status 1 if any selected case fails.",
    )
    return parser.parse_args()


def select_samples(sample_ids: list[str], limit: int | None) -> list[Sample]:
    benchmarkable = [sample for sample in SAMPLES if sample.benchmarkable]
    by_id = {sample.id: sample for sample in benchmarkable}
    if sample_ids:
        missing = [sample_id for sample_id in sample_ids if sample_id not in by_id]
        if missing:
            available = ", ".join(sorted(by_id))
            raise SystemExit(
                f"unknown or non-benchmarkable sample id(s): {', '.join(missing)}\n"
                f"available: {available}"
            )
        selected = [by_id[sample_id] for sample_id in sample_ids]
    else:
        selected = benchmarkable
    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be positive")
        selected = selected[:limit]
    return selected


def run_sample(sample: Sample, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    payload = {
        "query": sample.query,
        "arch": "native",
        "rows": args.rows,
        "warmups": args.warmups,
        "runs": args.runs,
        "null_density": args.null_density,
        "output": args.output,
        "provider": "google",
    }
    try:
        result = bench_worker.run(payload)
        benchmark = normalized_result(result)
        google = benchmark.get("google_benchmark") or {}
        google_ok = bool(google.get("ok")) and bool(benchmark.get("implementations"))
        return {
            "id": sample.id,
            "title": sample.title,
            "group": sample.group,
            "tags": list(sample.tags),
            "ok": bool(result.ok) and google_ok,
            "elapsed_wall_s": time.perf_counter() - started,
            "error": google_error(benchmark) if not google_ok else "",
            "benchmark": benchmark,
        }
    except Exception as exc:
        return {
            "id": sample.id,
            "title": sample.title,
            "group": sample.group,
            "tags": list(sample.tags),
            "ok": False,
            "elapsed_wall_s": time.perf_counter() - started,
            "error": str(exc),
        }


def google_error(benchmark: dict[str, Any]) -> str:
    google = benchmark.get("google_benchmark") or {}
    diagnostics = google.get("diagnostics") or benchmark.get("diagnostics") or []
    if diagnostics:
        return "; ".join(str(item) for item in diagnostics)
    return "Google Benchmark produced no implementation rows"


def normalized_result(result: BenchmarkRunResult) -> dict[str, Any]:
    data = to_json_dict(result)
    data["implementations"] = [
        {
            "name": row["name"],
            "key": row["key"],
            "source": row["source"],
            "api": row["api"],
            "compile_mode": row["compile_mode"],
            "backend": row["backend"],
            "opt": row["opt"],
            "hot_us": row["hot_us"],
            "compile_us": row["compile_us"],
            "compile_samples_us": row.get("compile_samples_us"),
            "compile_warmups": row.get("compile_warmups"),
            "compile_runs": row.get("compile_runs"),
            "compile_boundary": row.get("compile_boundary"),
            "rows_per_second": row["rows_per_second"],
            "throughput_gbps": row["throughput_gbps"],
            "speedup": row["speedup"],
        }
        for row in data.get("implementations", [])
        if row.get("source") == "Google Benchmark"
    ]
    return data


def install_dummy_google_benchmark() -> None:
    def fake_run_google_benchmark(
        vectorized_inspection,
        scalar_inspection,
        *,
        asmjit_variants,
        rows,
        repetitions,
        bytes_processed,
        jit_compile_measurements,
    ):
        del (
            vectorized_inspection,
            scalar_inspection,
            repetitions,
            jit_compile_measurements,
        )
        implementations = []
        hot_base = 10.0
        for idx, variant in enumerate(asmjit_variants):
            implementations.append(
                bench_worker._implementation_row(
                    variant["name"],
                    variant["key"],
                    hot_base + idx,
                    rows,
                    bytes_processed,
                    compile_us=variant.get("compile_us"),
                    source="Google Benchmark",
                )
            )
        bench_worker._add_bar_widths(implementations)
        return {
            "ok": True,
            "diagnostics": ["dummy Google Benchmark result"],
            "implementations": implementations,
        }

    bench_worker.run_google_benchmark = fake_run_google_benchmark


if __name__ == "__main__":
    raise SystemExit(main())
