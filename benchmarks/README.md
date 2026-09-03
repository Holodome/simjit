# Benchmarks

This directory contains the benchmark runners used for the Explorer demo page.
Visualization and report-rendering scripts are not kept here.

Run commands from the repository root.

## Python Demo Benchmarks

Build the Python extension and run the benchmark set used by the demo page:

```sh
NO_LTO=1 scripts/dev demo-bench
```

Outputs are written under `build/demo-python-bench/`:

- `python-demo-bench.jsonl`
- `python-demo-bench.md`

For a shorter local smoke while iterating:

```sh
NO_LTO=1 \
PYTHON_DEMO_BENCH_SIZES=1024 \
PYTHON_DEMO_BENCH_REPEATS=1 \
PYTHON_DEMO_BENCH_MIN_TIME=0.01 \
scripts/dev demo-bench
```

The Python paper benchmark runs the ten E1-E10 workloads with NumPy, Numba,
and Simjit:

```sh
NO_LTO=1 scripts/dev python-bench
```

Use `--workloads` for a subset. The default invocation runs all ten
workloads and writes `bench-results/python-bench-external.{jsonl,md}`.

The demo-specific runner is `benchmarks/python/demo_bench.py`; `scripts/dev
demo-bench` passes the exact workload and output options used by the
Explorer demo data refresh path.

## Arrow/Gandiva Benchmark

This benchmark compares Arrow compute, Gandiva, and Simjit C++ paths. It
requires Arrow, Gandiva, and Google Benchmark to be discoverable by CMake.

```sh
NO_LTO=1 scripts/dev jit-bench
build/reldebug/jit-bench --benchmark_format=json > /tmp/simjit-arrow-gandiva.json
```

The source is `benchmarks/arrow-gandiva/bench.cpp`.

## Runtime Corpus Benchmarks

Benchmark a test-runner JSONL bundle with structured, resumable output:

```sh
make local-runner
python3 tests/scripts/benchmark_corpus.py \
  --file /tmp/tests.jsonl \
  --output /tmp/simjit-benchmark-corpus.jsonl \
  --minimum-time 0.1 \
  --repetitions 1 \
  --resume
```

The corpus adapter drives `build/debug/local_runner --bench`, parses its Google Benchmark JSON, and writes the existing
schema-v2 JSONL records. It supports suite and implementation filtering, scalar fallbacks, O3 variants, retrying errors,
and fail-fast execution. Output paths are caller-selected and benchmark results are not tracked in this repository.

## Explorer Batch Benchmarks

Run benchmarkable Explorer samples without the web UI and emit one JSON document:

```sh
scripts/py scripts/run-explorer-benchmarks.py \
  --limit 1 \
  --rows 10000 \
  --warmups 1 \
  --runs 1 \
  --output-json /tmp/simjit-explorer-benchmarks.json
```

Use `--dummy` for an orchestration smoke test that skips Google Benchmark compilation and execution.

## DuckDB Benchmark

The DuckDB runner expects a local DuckDB checkout. It builds DuckDB with the
Simjit extension config unless `SKIP_BUILD=1` is set.

```sh
DUCKDB_DIR=/path/to/duckdb \
RESULT_DIR=/tmp/simjit-duckdb \
ROWS=10000000 \
HOT_RUNS=7 \
NULL_DENSITIES=0,1,10,50,90 \
NULL_PATTERNS=periodic,random,runs \
python3 benchmarks/duckdb/bench_simjit_fragments.py
```

Outputs are JSON and Markdown files in `RESULT_DIR`, plus the last generated
SQL and raw DuckDB output for inspection.

For a shorter smoke:

```sh
DUCKDB_DIR=/path/to/duckdb \
RESULT_DIR=/tmp/simjit-duckdb-smoke \
ROWS=10000 \
HOT_RUNS=1 \
NULL_DENSITIES=0,10 \
NULL_PATTERNS=periodic \
python3 benchmarks/duckdb/bench_simjit_fragments.py
```
