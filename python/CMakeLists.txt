cmake_minimum_required(VERSION 3.25)
include("${CMAKE_CURRENT_SOURCE_DIR}/../cmake/simjitVersion.cmake")
project(simjit_python VERSION "${SIMJIT_VERSION}" LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PYBIND11_FINDPYTHON ON)
set(SIMJIT_PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_BINARY_DIR}/simjit" CACHE PATH "Directory containing the Python simjit package")
option(SIMJIT_PYTHON_ENABLE_LLVM "Enable LLVM inspection output in the Python extension" OFF)
option(SIMJIT_PYTHON_ENABLE_CPP "Enable C++ inspection output in the Python extension" OFF)
option(SIMJIT_PYTHON_ENABLE_CROSS_ASMJIT "Enable non-host AsmJit emitters for inspection output" ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" SIMJIT_PYTHON_PROCESSOR)
set(SIMJIT_PYTHON_ENABLE_X86 OFF)
set(SIMJIT_PYTHON_ENABLE_ARM OFF)

if(SIMJIT_PYTHON_ENABLE_CROSS_ASMJIT)
  set(SIMJIT_PYTHON_ENABLE_X86 ON)
  set(SIMJIT_PYTHON_ENABLE_ARM ON)
elseif(SIMJIT_PYTHON_PROCESSOR MATCHES "^(x86_64|amd64|x64)$")
  set(SIMJIT_PYTHON_ENABLE_X86 ON)
elseif(SIMJIT_PYTHON_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(SIMJIT_PYTHON_ENABLE_ARM ON)
else()
  message(FATAL_ERROR "Unsupported Python wheel target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(SIMJIT_DEFAULT_BUILD_DEV_TARGETS OFF CACHE BOOL "Disable Simjit dev targets in Python build" FORCE)
set(SIMJIT_ENABLE_LLVM ${SIMJIT_PYTHON_ENABLE_LLVM} CACHE BOOL "Enable optional LLVM inspection emitter in Python extension" FORCE)
set(SIMJIT_ENABLE_ASMJIT ON CACHE BOOL "Enable AsmJit backend in Python build" FORCE)
set(SIMJIT_ENABLE_ASMJIT_X86 ${SIMJIT_PYTHON_ENABLE_X86} CACHE BOOL "Enable x86 AsmJit backend in Python build" FORCE)
set(SIMJIT_ENABLE_ASMJIT_ARM ${SIMJIT_PYTHON_ENABLE_ARM} CACHE BOOL "Enable Arm AsmJit backend in Python build" FORCE)
set(SIMJIT_ENABLE_CPP ${SIMJIT_PYTHON_ENABLE_CPP} CACHE BOOL "Enable optional C++ inspection emitter in Python extension" FORCE)
set(SIMJIT_ENABLE_SERIALIZATION ON CACHE BOOL "Enable serialization in Python build" FORCE)
set(SIMJIT_ENABLE_LIBDIVIDE ON CACHE BOOL "Enable libdivide in Python build" FORCE)
set(SIMJIT_ENABLE_NATIVE_ARCH OFF CACHE BOOL "Disable native-arch codegen in Python build" FORCE)

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../thirdparty/pybind11" "${CMAKE_CURRENT_BINARY_DIR}/pybind11")

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../src" "${CMAKE_CURRENT_BINARY_DIR}/simjit_core")
simjit_configure_lto()

pybind11_add_module(_simjit
  bindings/bindings.cpp
  bindings/module_api.cpp
  bindings/py_adapter.cpp
  bindings/native_builder.cpp
  bindings/resolver.cpp)
target_link_libraries(_simjit PRIVATE simjit)
set_target_properties(_simjit PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
simjit_apply_lto(_simjit)
simjit_apply_warnings(_simjit)
target_enable_static_analysis(_simjit)
file(MAKE_DIRECTORY "${SIMJIT_PYTHON_PACKAGE_DIR}")
set_target_properties(_simjit PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${SIMJIT_PYTHON_PACKAGE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${SIMJIT_PYTHON_PACKAGE_DIR}")

install(TARGETS _simjit LIBRARY DESTINATION simjit RUNTIME DESTINATION simjit)
install(FILES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/simjit/_simjit.pyi"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/simjit/py.typed"
  DESTINATION simjit)
