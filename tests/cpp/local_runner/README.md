# Native JSON bundle runner

`build/debug/local_runner` executes existing native `tests.jsonl` bundles without temporary source files, object files,
shared libraries, subprocess compilation, or `dlopen`. It is a developer-only target and requires the C++, LLVM, and
AsmJit backends plus Clang headers, `libclang-cpp`, and Google Benchmark.

```sh
make local-runner
make run-local-runner
```

The runner parses JSON lines concurrently, balances C++ and LLVM expressions into in-memory compilation groups,
materializes raw AsmJit code in worker-local runtimes, and schedules a case as soon as all of its implementations are
ready. Cases execute concurrently, while implementations within a case execute sequentially against the same generated
inputs. Results are retained and reported in JSON input order.

The C++ path uses Clang `EmitLLVMOnlyAction` and an in-memory VFS. The LLVM path links group-local modules and runs the
selected O1 or O3 pipeline with the native target machine. Both paths materialize through persistent worker-owned
`LLVMSession` instances. Each session disables ORC compilation threads and is used by only its scheduler worker, so the
runner's task pool owns all compilation concurrency. AsmJit likewise uses one persistent runtime for each worker.

The runner is native-architecture only and rejects `--isolate`. Python code entries are counted and skipped,
inspection-only code is ignored, and correctness cases with fewer than two stable native implementations are counted as
comparison-skipped.

Use `--bench` to benchmark native implementations in-process through Google Benchmark. Benchmark mode uses 4,096 rows
by default, prepares benchmark descriptors concurrently, initializes buffers outside each measured loop, measures
implementations serially, and adds O3 copies of scalar C++ and LLVM implementations to match the Python benchmark
workflow. `--rows` overrides the benchmark row count. Standard Google Benchmark `--benchmark_*` options are forwarded
when written in `--option=value` form, for example:

```sh
build/debug/local_runner --bench --file tests.jsonl --only-n 0 \
  --benchmark_min_time=0.1s --benchmark_repetitions=3

make run-local-runner \
  LOCAL_RUNNER_ARGS='--bench --limit 1 --benchmark_min_time=0.1s'
```

Use `--only-n`, `--suite`, or `--limit` for focused benchmarking. Google Benchmark calibrates every implementation
independently, so benchmarking an entire native bundle is intentionally much slower than correctness execution.
Use repeatable `--code` options to select bundle code kinds such as `asmjit_s` and `llvm_s`. `--bench-o3` controls
whether O3 copies are generated for no implementations, scalar implementations only (the default), or all selected C++
and LLVM implementations.

Output is concise by default: successful case labels and structured-error skip lines are suppressed. Use
`--verbose-item` when debugging to restore per-record output.

Use `--timings` to report submitted/completed task counts, active high-water marks, additive per-queue thread CPU time,
each queue's first-start to last-finish wall span, and end-to-end wall time. Compilation averages are reported separately
for C++ text and LLVM IR text, at O1 and O3; every average covers parsing/compilation through executable-pointer lookup
and is amortized by the number of functions in its compilation groups. `--workers`, `--rows`, and `--seed` make
concurrency and generated inputs explicit.
