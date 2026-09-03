# Validation Environment

Validation requires two native Linux machines:

- x86_64 with AVX-512F.
- AArch64.

Generated machine code must execute only on its matching native architecture.

## Dependencies

Both runners need:

- Bash, Git, CMake 3.25+, Ninja, Make, rsync, and a C++17/C++20 compiler.
- Clang, LLVM development files and `llvm-config`. Simjit and Gandiva must load
  the same LLVM build; two ABI-compatible-looking LLVM installations in one
  process can still conflict through LLVM's C++ symbols.
- `clang-format`, `clang-tidy`, and `cppcheck`.
- Apache Arrow C++ with Compute and Gandiva, plus Google Benchmark. On systems
  using Apache Arrow's APT repository the relevant packages are
  `libarrow-dev`, `libgandiva-dev`, and `libbenchmark-dev`.
- Python 3.10+ development headers and the packages in
  `scripts/validate-python-requirements.txt`:

  ```sh
  python3 -m pip install -r scripts/validate-python-requirements.txt
  ```

- Racket and Xsmith. Install Xsmith into the environment ahead of validation:

  ```sh
  raco pkg install --auto --batch \
    'https://gitlab.flux.utah.edu/xsmith/xsmith.git?path=xsmith#5dd69736542e0b30a5158de81c0dba4868fc8646'
  ```

- A DuckDB git checkout pinned to
  `932e831ce75bd4b6edd1d9e1f9136da6f52bf634`.

The Xsmith package checksum is
`5dd69736542e0b30a5158de81c0dba4868fc8646`. Changes to either dependency pin
should be reviewed explicitly.

## Container Image

Build the dependency image for both `linux/amd64` and `linux/arm64`. Install
Racket and Xsmith during the image build; the upstream
[Xsmith installation guide](https://docs.racket-lang.org/xsmith/How_to_Install_Xsmith.html)
uses `raco pkg`. Keep the source checkout outside the image so one immutable
dependency image can validate different revisions.

The container must run natively. In particular, an emulated amd64 container on
an Arm host does not provide AVX-512 execution coverage.
