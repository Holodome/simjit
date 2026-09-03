add_executable(test cpp/test.cpp)
target_link_libraries(test PRIVATE simjit)
target_compile_features(test PRIVATE cxx_std_20)
set_target_properties(test PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
simjit_apply_warnings(test)
simjit_apply_lto(test)

add_executable(integration_test cpp/integration_test.cpp)
target_link_libraries(integration_test PRIVATE simjit)
target_compile_features(integration_test PRIVATE cxx_std_20)
set_target_properties(integration_test PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
simjit_apply_warnings(integration_test)
simjit_apply_lto(integration_test)

set(LOCAL_RUNNER_MISSING_DEPENDENCIES)
if(NOT SIMJIT_ENABLE_LLVM)
  list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES "LLVM backend (SIMJIT_ENABLE_LLVM=OFF)")
endif()
if(NOT SIMJIT_ENABLE_ASMJIT)
  list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES "AsmJit backend (SIMJIT_ENABLE_ASMJIT=OFF)")
endif()
if(NOT SIMJIT_ENABLE_CPP)
  list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES "C++ backend (SIMJIT_ENABLE_CPP=OFF)")
endif()

if(SIMJIT_ENABLE_LLVM)
  simjit_query_llvm_config(LOCAL_RUNNER_LLVM
    COMPONENTS all)
  simjit_find_llvm_clang(
    LOCAL_RUNNER_CLANG
    LOCAL_RUNNER_CLANG_DIAGNOSTICS
    "${LOCAL_RUNNER_LLVM_BINDIR}")
  find_library(LOCAL_RUNNER_CLANG_CPP
    NAMES clang-cpp
    HINTS "${LOCAL_RUNNER_LLVM_LIBDIR}"
    NO_DEFAULT_PATH)
  if(NOT LOCAL_RUNNER_CLANG_CPP)
    file(GLOB LOCAL_RUNNER_CLANG_CPP_CANDIDATES
      "${LOCAL_RUNNER_LLVM_LIBDIR}/libclang-cpp.so.${SIMJIT_REQUIRED_LLVM_MAJOR}*"
      "${LOCAL_RUNNER_LLVM_LIBDIR}/libclang-cpp.${SIMJIT_REQUIRED_LLVM_MAJOR}*.dylib")
    list(SORT LOCAL_RUNNER_CLANG_CPP_CANDIDATES COMPARE NATURAL ORDER DESCENDING)
    list(LENGTH LOCAL_RUNNER_CLANG_CPP_CANDIDATES LOCAL_RUNNER_CLANG_CPP_COUNT)
    if(LOCAL_RUNNER_CLANG_CPP_COUNT GREATER 0)
      list(GET LOCAL_RUNNER_CLANG_CPP_CANDIDATES 0 LOCAL_RUNNER_CLANG_CPP)
    endif()
  endif()
  find_path(LOCAL_RUNNER_CLANG_INCLUDE_DIR
    NAMES clang/CodeGen/CodeGenAction.h
    HINTS "${LOCAL_RUNNER_LLVM_INCLUDEDIR}"
    NO_DEFAULT_PATH)
  find_package(benchmark QUIET CONFIG)

  if(NOT LOCAL_RUNNER_CLANG)
    list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES "${LOCAL_RUNNER_CLANG_DIAGNOSTICS}")
  endif()
  if(NOT LOCAL_RUNNER_CLANG_CPP)
    list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES
      "LLVM ${SIMJIT_REQUIRED_LLVM_MAJOR} libclang-cpp in ${LOCAL_RUNNER_LLVM_LIBDIR}")
  endif()
  if(NOT LOCAL_RUNNER_CLANG_INCLUDE_DIR)
    list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES
      "LLVM ${SIMJIT_REQUIRED_LLVM_MAJOR} Clang CodeGen headers in ${LOCAL_RUNNER_LLVM_INCLUDEDIR}")
  endif()
  if(NOT TARGET benchmark::benchmark)
    list(APPEND LOCAL_RUNNER_MISSING_DEPENDENCIES "Google Benchmark CMake package (benchmark::benchmark)")
  endif()
endif()

if(NOT LOCAL_RUNNER_MISSING_DEPENDENCIES)
  add_library(local_runner_core STATIC
    cpp/local_runner/bundle.cpp
    cpp/local_runner/planner.cpp
    cpp/local_runner/compiler.cpp
    cpp/local_runner/runtime.cpp)
  target_link_libraries(local_runner_core PUBLIC
    simjit
    ${LOCAL_RUNNER_CLANG_CPP}
    ${LOCAL_RUNNER_LLVM_LDFLAGS}
    ${LOCAL_RUNNER_LLVM_LIBS}
    benchmark::benchmark)
  target_include_directories(local_runner_core SYSTEM PRIVATE
    ${LOCAL_RUNNER_CLANG_INCLUDE_DIR}
    ${LOCAL_RUNNER_LLVM_INCLUDEDIR})
  set(LOCAL_RUNNER_LLVM_COMPILE_FLAGS ${LOCAL_RUNNER_LLVM_CXXFLAGS})
  list(FILTER LOCAL_RUNNER_LLVM_COMPILE_FLAGS EXCLUDE REGEX "^-std=")
  target_compile_options(local_runner_core PRIVATE
    ${LOCAL_RUNNER_LLVM_COMPILE_FLAGS})
  target_compile_definitions(local_runner_core PRIVATE
    SIMJIT_LOCAL_RUNNER_CLANGXX="${LOCAL_RUNNER_CLANG}")
  target_compile_features(local_runner_core PUBLIC cxx_std_20)
  set_target_properties(local_runner_core PROPERTIES
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF)
  simjit_apply_warnings(local_runner_core)
  simjit_apply_lto(local_runner_core)

  add_executable(local_runner cpp/local_runner/main.cpp)
  target_link_libraries(local_runner PRIVATE local_runner_core)
  target_include_directories(local_runner SYSTEM PRIVATE
    ${LOCAL_RUNNER_LLVM_INCLUDEDIR})
  target_compile_options(local_runner PRIVATE
    ${LOCAL_RUNNER_LLVM_COMPILE_FLAGS})
  set_target_properties(local_runner PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  simjit_apply_warnings(local_runner)
  simjit_apply_lto(local_runner)
else()
  message(STATUS "local_runner unavailable; missing dependencies:")
  foreach(dependency IN LISTS LOCAL_RUNNER_MISSING_DEPENDENCIES)
    message(STATUS "  - ${dependency}")
  endforeach()
  string(JOIN ", " LOCAL_RUNNER_MISSING_DEPENDENCIES_TEXT ${LOCAL_RUNNER_MISSING_DEPENDENCIES})
  add_custom_target(local_runner
    COMMAND ${CMAKE_COMMAND} -E echo
      "local_runner unavailable: ${LOCAL_RUNNER_MISSING_DEPENDENCIES_TEXT}"
    COMMAND ${CMAKE_COMMAND} -E false
    VERBATIM)
endif()
