if(NOT SIMJIT_ENABLE_SERIALIZATION)
  message(FATAL_ERROR "SIMJIT_BUILD_FUZZ_TOOLS=ON requires SIMJIT_ENABLE_SERIALIZATION=ON")
endif()

add_executable(simjit-fuzz-tool simjit_fuzz_tool.cpp)
target_link_libraries(simjit-fuzz-tool PRIVATE simjit)
target_include_directories(simjit-fuzz-tool SYSTEM PRIVATE
  "${PROJECT_SOURCE_DIR}/thirdparty/headerlisp")
target_compile_features(simjit-fuzz-tool PRIVATE cxx_std_20)
set_target_properties(simjit-fuzz-tool PROPERTIES
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
simjit_apply_warnings(simjit-fuzz-tool)
simjit_apply_lto(simjit-fuzz-tool)
