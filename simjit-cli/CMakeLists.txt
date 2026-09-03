add_executable(simjit-cli
  main.cpp)
target_link_libraries(simjit-cli PRIVATE simjit)
target_compile_features(simjit-cli PRIVATE cxx_std_17)
set_target_properties(simjit-cli PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
simjit_apply_warnings(simjit-cli)
simjit_apply_lto(simjit-cli)
