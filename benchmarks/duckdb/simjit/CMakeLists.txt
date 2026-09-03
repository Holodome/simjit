cmake_minimum_required(VERSION 3.25)
project(SimjitDuckDBExtension LANGUAGES CXX)

get_filename_component(SIMJIT_DUCKDB_EXTENSION_DIR "${CMAKE_CURRENT_LIST_DIR}" ABSOLUTE)
get_filename_component(SIMJIT_SOURCE_ROOT "${SIMJIT_DUCKDB_EXTENSION_DIR}/../../.." ABSOLUTE)

set(SIMJIT_ENABLE_LLVM OFF CACHE BOOL "" FORCE)
set(SIMJIT_ENABLE_CPP OFF CACHE BOOL "" FORCE)
set(SIMJIT_ENABLE_ASMJIT ON CACHE BOOL "" FORCE)
set(SIMJIT_ENABLE_NATIVE_ARCH OFF CACHE BOOL "" FORCE)

add_subdirectory("${SIMJIT_SOURCE_ROOT}/src" "${CMAKE_CURRENT_BINARY_DIR}/simjit_core")

build_loadable_extension(simjit "" simjit_extension.cpp)
set_property(TARGET simjit_loadable_extension APPEND PROPERTY LINK_LIBRARIES simjit)
