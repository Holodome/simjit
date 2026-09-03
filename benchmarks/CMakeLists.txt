find_package(Arrow REQUIRED)
find_package(ArrowCompute REQUIRED)
find_package(benchmark REQUIRED)

add_executable(jit-bench
  arrow-gandiva/bench.cpp)
target_link_libraries(jit-bench PRIVATE
  simjit
  Arrow::arrow_shared
  arrow_compute
  gandiva
  benchmark::benchmark)
target_compile_features(jit-bench PRIVATE cxx_std_20)
target_compile_options(jit-bench PRIVATE -frtti)
set_target_properties(jit-bench PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
simjit_apply_lto(jit-bench)
