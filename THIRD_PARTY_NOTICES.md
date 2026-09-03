# Third-Party Notices

This project includes vendored and copied third-party code. Not every
component listed here is present in every build output or install profile, but
the notices are kept together so source archives, static-library installs, and
Python builds have one place to audit.

## C++ Library And Install Package

### AsmJit

- Project: AsmJit
- Location: `thirdparty/asmjit`
- Installed in C++ package: static library when `SIMJIT_ENABLE_ASMJIT=ON`;
  headers only in the `toolkit` install profile
- License: Zlib
- License files:
  - `thirdparty/asmjit/LICENSE.md`
  - `thirdparty/asmjit/db/LICENSE.md`

### libdivide

- Project: libdivide
- Location: `thirdparty/libdivide`
- Installed in C++ package: not as public headers, but compiled into Simjit
  when `SIMJIT_ENABLE_LIBDIVIDE=ON`
- License: dual Zlib or Boost Software License 1.0
- License file: `thirdparty/libdivide/LICENSE.txt`

### expected-lite

- Project: expected-lite
- Location: `src/simjit/detail/expected.h`
- Installed in C++ package: public support header in both `jit` and `toolkit`
  install profiles
- Local modification: wrapped into the `simjit::nonstd` namespace
- License: Boost Software License 1.0
- License file: `thirdparty/LICENSES/Boost-1.0.txt`

### span-lite

- Project: span-lite
- Location: `src/simjit/detail/span.h`
- Installed in C++ package: public support header in both `jit` and `toolkit`
  install profiles
- Local modification: wrapped into the `simjit::nonstd` namespace
- License: Boost Software License 1.0
- License file: `thirdparty/LICENSES/Boost-1.0.txt`

### headerlisp

- Project: headerlisp
- Location: `thirdparty/headerlisp`
- Installed in C++ package: not as public headers, but used by serialization
  support when `SIMJIT_ENABLE_SERIALIZATION=ON`
- License: MIT
- License file: `thirdparty/headerlisp/LICENSE`

## Python Build

### pybind11

- Project: pybind11
- Location: `thirdparty/pybind11`
- Installed in C++ package: no
- Used by: Python extension build
- License: BSD-3-Clause
- License file: `thirdparty/pybind11/LICENSE`

## Explorer Static Assets

### Highlight.js

- Project: Highlight.js
- Location: `explorer/app/static/vendor/highlightjs`
- Used by: Explorer syntax highlighting
- License: BSD-3-Clause
- License notice: embedded in `explorer/app/static/vendor/highlightjs/highlight.min.js`

## Non-Vendored Dependencies

LLVM, Arrow, Gandiva, DuckDB, Google Benchmark, NumPy, PyArrow, Nginx, and
system packages used by development, benchmarks, or Docker images are not
vendored into this project tree. They remain governed by their own
installations and distribution terms.
