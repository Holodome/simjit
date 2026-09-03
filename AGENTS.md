# AGENTS.md

## Project Overview

Simjit is a C++17 JIT/vectorization project for fast columnar expression execution. It has two user-facing layers:

- C++ API and CLI for direct expression construction, lowering, codegen inspection, and testing.
- Python package for NumPy/Arrow-style column expressions, runtime specialization, caching, null handling, and
  zero-copy buffer execution.

The main code generation backends are:

- AsmJit backend: primary production backend and the Python runtime execution backend.
- LLVM backend: inspection/reference backend enabled through the system `llvm-config`.
- C++ backend: debugging/reference backend, useful for readable emitted code and test comparison.

## Project Importance Map

1. The C++ library is the most important part of the project. AsmJit is the production backend. C++ and LLVM backends
   are also fully supported, except for unavoidable language-defined UB differences. LLVM is especially useful as an
   optimized reference when run through `clang -O3`; a typical inspection flow is:

   ```sh
   build/debug/simjit-cli -s "$HIR" --llvm > /tmp/case.ll
   clang -march=native -O3 -c /tmp/case.ll -o /tmp/case.o
   objdump -d -j .text /tmp/case.o
   ```

   On Mach-O targets, use the platform text section name, for example
   `llvm-objdump -d --section=__TEXT,__text /tmp/case.o`.

2. Explorer is the user-facing landing page. Its three showcased benchmarks are the second-most-important project
   surface. Simjit non-functional goals are low memory usage, low compile time, and predictable heuristic algorithms.
   The main bottleneck should remain AsmJit's register allocator.
3. The Python library is important, but it can tolerate adapter-level inconsistencies better than the C++ layer because
   the C++ validation and lowering pipeline should catch invalid programs.
4. Correctness is primarily tested with tests that produce executable programs, not by inspecting output text. Runner
   scripts execute generated programs with random data.
5. The `fuzz/` workflow is extremely efficient at finding backend bugs.
6. The largest known testing gap is end-to-end intent coverage: a HIR construction bug can still generate valid code
   that does not match the intended expression. This is unavoidable, so keep builder/API tests focused on intent as well
   as generated-program correctness.

Important public headers:

- `src/simjit/simjit.h`: core expression builder handles, operations, datatypes, argument kinds, and optimizations.
- `src/simjit/errors.h`: structured error taxonomy (`ErrorModule`, `ErrorKind`, `ErrorSubKind`, `ErrorInfo`).
- `src/simjit/compiler.h`: lowering helpers, build limits, backend emitters, serialization gate.
- `src/simjit/jit.h`: JIT context, compile policy, cache API, debug snapshots, typed function wrappers.
- `src/simjit/nullable.h`: SQL-style nullable expression builder.
- `src/simjit/asmjit.h`: AsmJit session and executable-memory ownership.
- `src/simjit/dynamic_value.h`: dynamic value/predicate handle used by higher-level layers.

Users should be able to include just `simjit.h` to use the builder.
Users should be able to include just `compiler.h` to create the builder and run compilation as sequence of function calls.
Users should be able to include just `jit.h` to create the JIT runtime.

Core lowering pipeline:

- HIR: `src/simjit/core/hir.*`
- vectorizer: `src/simjit/core/vectorizer.*`
- MIR: `src/simjit/core/mir.*`
- backend emitters: `src/simjit/core/{aj,llvm,cpp}/`
- expression/datatype helpers: `src/simjit/core/expr.*`
- optional serialization: `src/simjit/core/serialize.cpp`

## Build Commands

Prefer top-level `Makefile` wrappers over calling CMake directly. The wrappers enable serialization by default and keep
build directories consistent.

- Build debug CLI: `make debug`
- Build release CLI: `make release`
- Build RelWithDebInfo library: `make reldebug`
- Build the main and integration test binaries: `make test`
- Build tests in RelWithDebInfo: `make test-reldebug`
- Build Arrow/Gandiva JIT benchmark target: `scripts/dev jit-bench`
- Build fuzz driver tools: `make fuzz-tools`
- Build explorer LLVM probe module: `scripts/dev explorer-probe`

Useful build knobs:

- `NO_LLVM=1`: disable LLVM backend.
- `NO_ASMJIT=1`: disable AsmJit backend.
- `NO_CPP=1`: disable C++ backend.
- `NO_X86=1`: disable AsmJit x86 backend while keeping Arm.
- `NO_ARM=1`: disable AsmJit Arm backend while keeping x86.
- `NO_LTO=1`: disable LTO for release-style builds.
- `LTO_MODE=THIN|FULL|AUTO`: choose LTO flavor.
- `ASAN=1`, `UBSAN=1`, `COVERAGE=1`: enable sanitizer or coverage build flags.
- `WORKERS=N`: override parallel build worker count.

Clean behavior:

- `make clean` removes reports and test dumps but keeps `build/`.
- `make distclean` removes `build/`, reports, and test dumps.

## Python Workflow

The Python package is in `python/` and uses `scikit-build-core` plus `pybind11`. It requires Python 3.10+. NumPy and
Arrow are optional package extras, though most local development and tests use NumPy.

Use the repo launcher for Python scripts because it selects the interpreter and sets:

- `PYTHONPATH=build/python-dev:python:python/src:python/tests`

Important commands:

- Build local optimized extension: `make py`
- Build local debug extension: `make py-debug`
- Build release wheel: `make py-release`
- Run Python unit tests: `make py-test`
- Run Python e2e tests: `make py-e2e`
- Run Python benchmarks: `scripts/dev python-bench`
- Run serialized bug-report debug-path smoke: `scripts/py python/tests/debug_path_smoke.py`
- Run explorer demo Python benchmarks: `scripts/dev demo-bench`
- Regenerate Python stub file: `make regen-pyi`
- Run Python unit tests through the launcher: `scripts/py -m pytest -q python/tests/test.py`
- Inspect launcher environment: `scripts/py --debug-env -m pytest -q python/tests/test.py`

Environment overrides:

- `SIMJIT_PYTHON=/path/to/python`
- `SIMJIT_PYTHON_BUILD_DIR=/path/to/build/python-dev`

The default Python build enables AsmJit for runtime execution and AsmJit inspection output. LLVM and C++ inspection
emitters are opt-in for the Python extension: use `PY_LLVM=1 PY_CPP=1 make py`, or pass
`-DSIMJIT_PYTHON_ENABLE_LLVM=ON -DSIMJIT_PYTHON_ENABLE_CPP=ON` to CMake. The Python build enables both x86 and Arm
AsmJit emitters for cross-architecture inspection unless `SIMJIT_PYTHON_ENABLE_CROSS_ASMJIT=OFF` is passed to CMake.

## Test Commands

Important entry points:

- Build tests only: `make test`
- Run compiler-output and IR-shape inspections: `build/debug/integration_test`
- Run compiler test cases without runtime checking: `make run-min-tests`
- Run the complete project workflow: `make run-all-tests`
- Run build configuration: `make check-builds`
- Build and check install profiles: `make check-installs`
- Run static analysis build: `make static-analysis`
- Merge configured CMake compile databases: `make compile-commands`
- Generate full JSONL bundle: `make dump-tests-json`
- Run native runtime checks on the full bundle: `make run-local-runner`
- Serve local JSONL/code viewer: `scripts/dev results-serve`

The main test runner is:

- `build/debug/test --suite=all --arch=all --mode=all`

Test runner options:

- `--suite <csv>` supports
  `int,float,nullable,tpcds,general,libdivide,agg,invalid_type,invalid_builder,misc,ternarylogic,all`.
- `--arch <native|x86|x86-ymm|arm|all>`
- `--mode <auto|scalar|novect|all>`
- `--emit <csv>` supports `cpp,llvm,asmjit,all`.
- `--validate-serialization`
- `--dump-json <path>`
- `--arena-stats`
- `--log-failures`
- `--log <csv>` supports `hir,vectorizer,mir,all`.
- `--test-id <suite:idx>`, `--test-num <n>`, `--test-at <file:line>`
- `--iterations <n>`

Useful focused runs:

- Single suite: `build/debug/test --suite=int --arch=x86 --mode=auto`
- Native smoke: `build/debug/test --suite=general --arch=native --mode=auto`
- One backend: `build/debug/test --suite=float --emit=asmjit --arch=arm --mode=scalar`
- Serialization validation: `build/debug/test --suite=nullable --validate-serialization`
- Debug lowering logs: `build/debug/test --suite=int --test-num 0 --log=hir,vectorizer,mir`

### Generated-Code Execution Correctness

Do not treat `make test` or a successful `build/debug/test` invocation as generated-code execution correctness:

- `make test` only builds the test binary.
- `build/debug/test` constructs, lowers, emits, and optionally dumps test programs. Its success and timing labels are not
  proof that the dumped native program was executed and compared by the runtime runner.

For codegen correctness, build the native local runner, dump a native JSONL bundle, and execute it:

```sh
build/debug/test --suite=int --arch=native --mode=all \
  --test-num 0 --dump-json /tmp/simjit-case.jsonl
make local-runner
build/debug/local_runner --file /tmp/simjit-case.jsonl
```

Use `--test-id`, `--test-num`, or `--test-at` to select one test. `--mode=all` dumps both auto/vectorized
and scalar variants. Confirm that `local_runner` exits successfully and prints `Testing complete!`; that is the
execution check. Use `make run-local-runner` for the full native bundle or follow the equivalent dump-and-run steps in
`tests/scripts/run-all-tests.sh`.

Do not use `--iterations` to claim additional runtime coverage. It repeats compiler/test-harness work and duplicates
dumped cases; it does not make `local_runner` exercise additional runtime inputs.

Do not try to execute foreign-architecture machine code locally: x86/x86-ymm binaries are not runnable on Arm hosts, and
Arm binaries are not runnable on x86 hosts. Use `--arch=native` for local execution, or use foreign `--arch` values only
for compile-time inspection, JSON dumps, or codegen diagnostics that do not execute the generated function.

Environment variable aliases:

- `TI`, `TF`, `TDS`, `TGENERAL`, `TAGG` select suites.
- `SCALAR`, `NOVECT`, `ARM` map to mode/arch.
- `LLVM`, `ASMJIT`, `NOCPP` control emission.
- `DUMP` enables JSONL dump to `tests.jsonl` unless `--dump-json` is supplied.
- `LOG` enables failure HIR logging.
- `N` sets iteration count.

CLI script tests are in `tests/scripts/run-cli-tests.sh`.

## Full Workflow

`make run-all-tests` runs `tests/scripts/run-all-tests.sh` and performs:

- `make test`
- compiler-output and IR-shape checks through `build/debug/integration_test`
- main test runner over all suites, arches, and modes
- JSONL dump and serialization grammar validation
- native execution of the full all-suite JSONL bundle
- `make debug`
- `make py`, `make py-test`, `make py-e2e`
- Python serialized debug-path smoke through `python/tests/debug_path_smoke.py`
- CLI script tests

Use this full workflow when touching cross-cutting behavior, CLI parsing/output, Python bindings/runtime, or serialization
format.

## Coverage

To collect coverage:

1. `COVERAGE=1 make test`
2. Run the tests you care about, or `make run-all-tests`.
3. `./tests/scripts/collect-coverage.sh`

Coverage intentionally excludes:

- `thirdparty`
- `simjit-cli`
- assertion/unreachable-only lines

## Repository Layout

- `src/`: C++ library sources.
- `src/simjit/core/`: HIR, vectorizer, MIR, pipeline, serialization, and backend internals.
- `python/`: Python package, pybind11 bindings, and Python tests.
- `tests/cpp/`: C++ test suites and runner.
- `tests/scripts/`: CLI, install, coverage, serialization, workflow, and runtime-bundle test helpers.
- `simjit-cli/`: CLI parser and driver.
- `benchmarks/`: Python, Arrow/Gandiva, and DuckDB benchmark entry points.
- `benchmarks/duckdb/`: DuckDB extension, benchmark drivers, and benchmark assets.
- `explorer/`: local web explorer service, LLVM probe module, and Docker packaging.
- `fuzz/`: fuzzing driver, report rendering, and generated-program workflow.
- `scripts/`: developer orchestration and utilities, including `scripts/dev` for benchmarks and Explorer tasks.
- `cmake/`: reusable CMake modules for the library and LTO setup.
- `thirdparty/`: vendored dependencies (`asmjit`, `headerlisp`, `libdivide`, `pybind11`).

## Code Style And Conventions

- Keep added text/code ASCII unless the file already uses non-ASCII.
- Prefer top-level Makefile targets for routine work; use CMake directly only for uncommon configuration work.
- Prefer short, direct error messages that identify the violated backend, IR, or API assumption.
- Prefer structured errors from `src/simjit/errors.h` over ad-hoc exception text when adding validation.
- Real validation should happen as early as practical in public API, HIR, vectorizer, or MIR. Backend sanity checks should
  be last-resort protection.
- Public handles (`Value`, `Predicate`, `MaybeValue`, `MaybePredicate`, `Argument`) are lightweight builder-owned handles;
  do not extend their lifetime beyond function construction.
- `Maybe*::value()` is a contract check: callers should test `is_valid()` first.
- Keep scalar fallback behavior explicit. Best-effort JIT policy may fall back from vectorized lowering to scalar, but
  tests should record known vectorization limitations.
- The Python layer should preserve zero-copy buffer behavior and keep NumPy/Arrow-facing semantics explicit.
- Follow clang-format style as specified in `.clang-format`. Use `make clang-format` for broad formatting passes.
- Follow clang-tidy and cppcheck settings as specified in `src/CMakeLists.txt` and `.clang-tidy`. For a full check, run
  `make distclean` first, then `make static-analysis`. Do not run full semantic analysis too often; it is
  computationally expensive.

## Testing Expectations

- Prefer targeted test runs while iterating.
- If you change public builder validation or error classification, add/update tests for `ErrorModule`, `ErrorKind`, and
  `ErrorSubKind` expectations where applicable.
- If you change vectorization, run vector tests for both x86 and Arm metadata paths where possible.
- If you change backend instruction selection, prefer focused backend runs first, for example:
  - `build/debug/test --suite=int,float --emit=asmjit --arch=x86 --mode=auto`
  - `build/debug/test --suite=int,float --emit=asmjit --arch=arm --mode=auto`
- If you change serialization, run `build/debug/test ... --validate-serialization` and the JSON/grammar validation path.
- If you change Python bindings/runtime, run at least `make py-test`; use `make py-e2e` for execution semantics.
- If you change CLI parsing, emitted text, Python package behavior, or cross-cutting build flags, run `make run-all-tests`
  when feasible.
- If you change CMake options, backend gates, or target setup, run `make check-builds` when feasible.
- If you fix a codegen bug, add targeted regression test using `int_test.cpp` as a template - no direct output checks.
- If you add a feature or a flag, add targeted regression test as done in `misc_test.cpp`. Inspecting output is allowed
  if that is required to confirm the state of inner structures.

## Test Suite Notes

The C++ test runner uses explicit per-variant metadata for:

- `x86-scalar`
- `x86-vector`
- `arm-scalar`
- `arm-vector`

Common presets and helpers live in `tests/cpp/test.h`, including:

- `PASS_ALL`
- `ONLY_SCALAR`
- `BUG_X86_VECTOR`
- `BUG_ARM_VECTOR`
- `BUG_ALL_VECTOR`
- `LIMIT_X86_VECTOR`
- `LIMIT_ARM_VECTOR`
- `LIMIT_ALL_VECTOR`
- `BUG_ALL`
- fluent helpers such as `.only(...)`, `.skip(...)`, `.bug(...)`, `.limitation(...)`, `.unstable_*_only(...)`,
  `.vectorization_failure(...)`, and `.structured_error(...)`

Use the smallest accurate preset. Prefer `limitation(...)` for expected unsupported behavior, `bug(...)` for known
incorrect behavior, and structured error expectations when validation should fail with a specific classification.

## Backend Notes

- AsmJit x86 targets AVX-512 by default. The `x86-ymm` arch exists for targeted YMM-width investigations, but it is not
  part of routine validation unless the change directly concerns that path.
- Arm AsmJit backend targets AArch64 NEON.
- Cross-architecture AsmJit emitters are useful for inspection, but generated machine code must run only on matching
  hardware.
- LLVM backend is enabled through the system LLVM installation discovered by `llvm-config` or `LLVM_CONFIG`.
- C++ backend is for debugging, readable output, and reference comparison.
- Optional `libdivide` support is enabled by default through `thirdparty/libdivide` and backs constant integer division
  optimizations when available.
- `SIMJIT_ENABLE_NATIVE_ARCH` is on by default for the library build; be mindful of host-specific codegen assumptions.

## Serialization And Fuzzing Notes

- Serialization is controlled by `SIMJIT_ENABLE_SERIALIZATION`; Makefile wrappers enable it by default.
- Serializer/parser live in `src/simjit/core/serialize.cpp`.
- Grammar file is `tests/scripts/serialization-format.lark`.
- Grammar validation script: `tests/scripts/validate-tests-json-serialization.sh`.
- Fuzz tools require serialization and are built with `make fuzz-tools`.
- Fuzz workflow documentation: `fuzz/README.md`.
- The unified fuzz driver is `fuzz/simjit-fuzz`.

### Handling Serialized Bug Reports

Bug reports often come as one serialized `(func ...)` expression. Save it as a `.simjit` file and start with
`simjit-cli`; do not route a single case through the fuzz tool unless you need fuzz-specific minimization or corpus
processing.

1. Build the CLI:
   `make debug`
2. Inspect the generated forms. Enable only the backend you want to inspect; if lowering or emission fails,
   `simjit-cli` reports the failing enabled backend:
   `build/debug/simjit-cli --serialized-file /tmp/case.simjit --arch x86 --asmjit --print-hir --print-mir`
3. Inspect generated assembly first for obvious inconsistencies:
   `build/debug/simjit-cli --serialized-file /tmp/case.simjit --arch x86 --asmjit`
4. Generate a one-line `tests.jsonl`-compatible bundle for the local runner and disassembly tools:
   `build/debug/simjit-cli --serialized-file /tmp/case.simjit --arch x86 --dump-json /tmp/case-x86.jsonl`
5. Run runtime comparison only on matching hardware:
   `build/debug/local_runner --file /tmp/case-x86.jsonl`
6. Benchmark native implementations in-process:
   `build/debug/local_runner --bench --file /tmp/case-x86.jsonl`

Architecture notes:

- Use `--arch arm` on Apple Silicon/AArch64 hosts and `--arch x86` on x86_64 hosts when executing generated code.
- Do not execute foreign-architecture bundles locally. Use foreign `--arch` values only for compile-time inspection,
  JSON dumps, or disassembly, then move the bundle to matching hardware for execution.

AsmJit inspection notes:

- A recurring x86 failure mode is invalid machine code encoding emitted by AsmJit without a useful error. The textual
  AsmJit assembly can look plausible while the actual bytes are invalid.
- When x86 output looks suspicious, disassemble the raw AsmJit blob from the JSONL bundle:
  `scripts/objdump-asmjit-json.py /tmp/case-x86.jsonl --arch x86 --show-raw-insn`
- True AsmJit code emission bugs are rare but possible. Disassembling the raw bytes is still the fastest way to
  distinguish Simjit instruction selection bugs from backend encoding bugs.

Runtime-difference notes:

- LLVM and C++ outputs can differ from AsmJit if the generated code contains C/C++/LLVM UB. The fuzzer tries to avoid
  UB-generating programs, but reports can still contain missed cases.
- If C++ or LLVM disagrees with AsmJit, first decide whether the serialized expression admits language-defined UB before
  treating it as a backend correctness bug.

The smoke test for this workflow is `scripts/py python/tests/debug_path_smoke.py`. It gets serialized HIR from the Python
library, runs the `simjit-cli` inspection commands above, generates a one-case JSONL bundle, runs `local_runner`, and
disassembles the raw AsmJit blob with `objdump-asmjit-json.py`.

## DuckDB Benchmark Notes

- DuckDB benchmark and extension code is in `benchmarks/duckdb/`.
