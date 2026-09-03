set(PYBIND11_FINDPYTHON ON)
set(SIMJIT_EXPLORER_PROBE_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}"
    CACHE PATH "Directory for the explorer LLVM probe extension")

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

if(NOT TARGET pybind11::module)
  add_subdirectory(
    "${PROJECT_SOURCE_DIR}/thirdparty/pybind11"
    "${CMAKE_CURRENT_BINARY_DIR}/pybind11")
endif()

if(NOT SIMJIT_ENABLE_LLVM)
  message(FATAL_ERROR "Explorer LLVM probe requires SIMJIT_ENABLE_LLVM=ON")
endif()

pybind11_add_module(explorer_llvm_probe probe/llvm_probe.cpp)
target_link_libraries(explorer_llvm_probe PRIVATE
  simjit)
set_target_properties(explorer_llvm_probe PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  LIBRARY_OUTPUT_DIRECTORY "${SIMJIT_EXPLORER_PROBE_OUTPUT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${SIMJIT_EXPLORER_PROBE_OUTPUT_DIR}")
simjit_apply_lto(explorer_llvm_probe)
