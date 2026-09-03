#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
base_dir="${1:-"$root/build/install-check"}"
workers="${2:-4}"
cmake_bin="${CMAKE:-cmake}"
cxx_bin="${CXX:-c++}"

configure_and_install() {
  local profile="$1"
  local build_dir="$base_dir/$profile-build"
  local install_dir="$base_dir/$profile-install"

  local args=(
    -S "$root"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_INSTALL_PREFIX="$install_dir"
    -DSIMJIT_DEFAULT_BUILD_DEV_TARGETS=OFF
    -DSIMJIT_BUILD_CLI=OFF
    -DSIMJIT_BUILD_TESTS=OFF
    -DSIMJIT_BUILD_BENCHMARKS=OFF
    -DSIMJIT_BUILD_FUZZ_TOOLS=OFF
    -DSIMJIT_BUILD_EXPLORER_LLVM_PROBE=OFF
    -DSIMJIT_ENABLE_LLVM=OFF
    -DSIMJIT_ENABLE_NATIVE_ARCH=OFF
    -DSIMJIT_ENABLE_SERIALIZATION=ON
    -DSIMJIT_ENABLE_LTO=OFF
    -DSIMJIT_INSTALL_PROFILE="$profile"
  )

  if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
    args=(-G "$CMAKE_GENERATOR" "${args[@]}")
  fi

  "$cmake_bin" "${args[@]}"
  "$cmake_bin" --build "$build_dir" --target install --parallel "$workers"
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    printf 'missing expected file: %s\n' "$path" >&2
    exit 1
  fi
}

require_contains() {
  local path="$1"
  local text="$2"
  if ! grep -Fq "$text" "$path"; then
    printf 'expected %s to contain: %s\n' "$path" "$text" >&2
    exit 1
  fi
}

reject_file() {
  local path="$1"
  if [[ -e "$path" ]]; then
    printf 'unexpected installed file: %s\n' "$path" >&2
    exit 1
  fi
}

reject_include() {
  local install_dir="$1"
  local include_name="$2"
  local source="$base_dir/negative-${include_name//\//_}.cpp"
  local object="$base_dir/negative-${include_name//\//_}.o"

  printf '#include <%s>\nint main() { return 0; }\n' "$include_name" > "$source"
  if "$cxx_bin" -std=c++17 -I "$install_dir/include" -c "$source" -o "$object" >/dev/null 2>&1; then
    printf 'unexpectedly compiled forbidden include: %s\n' "$include_name" >&2
    exit 1
  fi
}

write_jit_consumer() {
  local consumer_dir="$1"

  mkdir -p "$consumer_dir"
  cat > "$consumer_dir/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.25)
project(simjit_jit_consumer LANGUAGES CXX)

find_package(simjit 0.1 CONFIG REQUIRED)

if(NOT SIMJIT_VERSION STREQUAL "0.1.0")
  message(FATAL_ERROR "unexpected simjit version: ${SIMJIT_VERSION}")
endif()

foreach(std IN ITEMS 17 20)
  add_executable(jit_smoke_${std} main.cpp)
  set_target_properties(jit_smoke_${std} PROPERTIES
    CXX_STANDARD ${std}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  target_link_libraries(jit_smoke_${std} PRIVATE simjit::simjit)
endforeach()
CMAKE

  cat > "$consumer_dir/main.cpp" <<'CPP'
#include <simjit/compiler.h>
#include <simjit/dynamic_value.h>
#include <simjit/jit.h>
#include <simjit/nullable.h>

#include <cstdint>

using namespace simjit;
using namespace simjit::jit;
using namespace simjit::types;

static void build_sum_i32(FunctionBuilder &b) {
    Value src = b.input_arg(I32);
    Argument dst = b.arg(I32);
    b.sum(src, dst);
}

int main() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto func = vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "install-smoke-sum", build_sum_i32);

    int32_t input[1000];
    for (int32_t i = 0; i < 1000; ++i) {
        input[i] = i;
    }

    int32_t output = 0;
    func(1000, input, &output);

    if (output != 499500) {
        return 1;
    }

    DynamicValue dynamic{};
    return dynamic.is_valid() ? 2 : 0;
}
CPP
}

write_toolkit_consumer() {
  local consumer_dir="$1"

  mkdir -p "$consumer_dir"
  cat > "$consumer_dir/CMakeLists.txt" <<'CMAKE'
cmake_minimum_required(VERSION 3.25)
project(simjit_toolkit_consumer LANGUAGES CXX)

find_package(simjit 0.1 CONFIG REQUIRED)

if(NOT SIMJIT_VERSION STREQUAL "0.1.0")
  message(FATAL_ERROR "unexpected simjit version: ${SIMJIT_VERSION}")
endif()

foreach(std IN ITEMS 17 20)
  add_executable(toolkit_smoke_${std} main.cpp)
  set_target_properties(toolkit_smoke_${std} PROPERTIES
    CXX_STANDARD ${std}
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  target_link_libraries(toolkit_smoke_${std} PRIVATE simjit::simjit)
endforeach()
CMAKE

  cat > "$consumer_dir/main.cpp" <<'CPP'
#include <simjit/asmjit.h>
#include <simjit/core/cpp/x86_intrin.h>
#include <simjit/core/hir.h>
#include <simjit/core/mir.h>
#include <simjit/core/vectorizer.h>
#include <simjit/jit.h>

#include <cstdint>

using namespace simjit;
using namespace simjit::jit;
using namespace simjit::types;

static void build_sum_i32(FunctionBuilder &b) {
    Value src = b.input_arg(I32);
    Argument dst = b.arg(I32);
    b.sum(src, dst);
}

int main() {
    AsmjitSession session(Arch::Native);
    (void)session.host_supports_vectorization();

    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto func = vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "toolkit-install-smoke-sum", build_sum_i32);

    int32_t input[1000];
    for (int32_t i = 0; i < 1000; ++i) {
        input[i] = i;
    }

    int32_t output = 0;
    func(1000, input, &output);

    return output == 499500 ? 0 : 1;
}
CPP
}

build_and_run_consumer() {
  local profile="$1"
  local install_dir="$base_dir/$profile-install"
  local consumer_dir="$base_dir/$profile-consumer"
  local build_dir="$consumer_dir/build"
  shift
  local targets=("$@")

  local args=(
    -S "$consumer_dir"
    -B "$build_dir"
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_PREFIX_PATH="$install_dir"
  )
  if [[ -n "${CMAKE_GENERATOR:-}" ]]; then
    args=(-G "$CMAKE_GENERATOR" "${args[@]}")
  fi

  "$cmake_bin" "${args[@]}"
  "$cmake_bin" --build "$build_dir" --parallel "$workers"

  for target in "${targets[@]}"; do
    "$build_dir/$target"
  done
}

check_cmake_package_metadata() {
  local install_dir="$1"
  local config="$install_dir/lib/cmake/simjit/simjitConfig.cmake"
  local version_config="$install_dir/lib/cmake/simjit/simjitConfigVersion.cmake"

  require_file "$config"
  require_file "$version_config"
  require_contains "$config" "SPDX-License-Identifier: Zlib"
  require_contains "$config" 'set(SIMJIT_VERSION "0.1.0")'
  require_contains "$version_config" "SPDX-License-Identifier: Zlib"
  require_contains "$version_config" 'set(PACKAGE_VERSION "0.1.0")'
}

"$cmake_bin" -E rm -rf "$base_dir"
"$cmake_bin" -E make_directory "$base_dir"

configure_and_install jit

jit_install="$base_dir/jit-install"
require_file "$jit_install/include/simjit/simjit.h"
require_file "$jit_install/include/simjit/errors.h"
require_file "$jit_install/include/simjit/compiler.h"
require_file "$jit_install/include/simjit/jit.h"
require_file "$jit_install/include/simjit/nullable.h"
require_file "$jit_install/include/simjit/dynamic_value.h"
require_file "$jit_install/include/simjit/detail/arena.h"
require_file "$jit_install/include/simjit/detail/span.h"
require_file "$jit_install/include/simjit/detail/expected.h"
check_cmake_package_metadata "$jit_install"
require_file "$jit_install/share/doc/simjit/LICENSE"
require_file "$jit_install/share/doc/simjit/THIRD_PARTY_NOTICES.md"
require_file "$jit_install/share/doc/simjit/licenses/asmjit/LICENSE.md"
require_file "$jit_install/share/doc/simjit/licenses/asmjit-db/LICENSE.md"
require_file "$jit_install/share/doc/simjit/licenses/boost/Boost-1.0.txt"
require_file "$jit_install/share/doc/simjit/licenses/headerlisp/LICENSE"
require_file "$jit_install/share/doc/simjit/licenses/libdivide/LICENSE.txt"
require_file "$jit_install/share/doc/simjit/licenses/pybind11/LICENSE"
reject_file "$jit_install/benchmarks"
reject_file "$jit_install/examples"
reject_file "$jit_install/tests"
reject_file "$jit_install/include/simjit/asmjit.h"
reject_file "$jit_install/include/simjit/core/hir.h"
reject_file "$jit_install/include/simjit/core/mir.h"
reject_include "$jit_install" "simjit/asmjit.h"
reject_include "$jit_install" "simjit/core/hir.h"

write_jit_consumer "$base_dir/jit-consumer"
build_and_run_consumer jit jit_smoke_17 jit_smoke_20

configure_and_install toolkit

toolkit_install="$base_dir/toolkit-install"
require_file "$toolkit_install/include/simjit/asmjit.h"
require_file "$toolkit_install/include/simjit/core/hir.h"
require_file "$toolkit_install/include/simjit/core/mir.h"
require_file "$toolkit_install/include/simjit/core/vectorizer.h"
require_file "$toolkit_install/include/simjit/core/cpp/x86_intrin.h"
require_file "$toolkit_install/include/asmjit/core.h"
require_file "$toolkit_install/include/asmjit/core/jitruntime.h"
check_cmake_package_metadata "$toolkit_install"
require_file "$toolkit_install/share/doc/simjit/LICENSE"
require_file "$toolkit_install/share/doc/simjit/THIRD_PARTY_NOTICES.md"
require_file "$toolkit_install/share/doc/simjit/licenses/asmjit/LICENSE.md"
require_file "$toolkit_install/share/doc/simjit/licenses/asmjit-db/LICENSE.md"
require_file "$toolkit_install/share/doc/simjit/licenses/boost/Boost-1.0.txt"
require_file "$toolkit_install/share/doc/simjit/licenses/headerlisp/LICENSE"
require_file "$toolkit_install/share/doc/simjit/licenses/libdivide/LICENSE.txt"
require_file "$toolkit_install/share/doc/simjit/licenses/pybind11/LICENSE"
reject_file "$toolkit_install/benchmarks"
reject_file "$toolkit_install/examples"
reject_file "$toolkit_install/tests"

write_toolkit_consumer "$base_dir/toolkit-consumer"
build_and_run_consumer toolkit toolkit_smoke_17 toolkit_smoke_20

printf 'install profile checks passed: %s\n' "$base_dir"
