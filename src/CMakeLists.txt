get_filename_component(SIMJIT_SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
include("${SIMJIT_SOURCE_ROOT}/cmake/simjitVersion.cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(SIMJIT_ENABLE_LLVM "Include LLVM backend based on system LLVM installation" ON)
option(SIMJIT_ENABLE_ASMJIT "Include AsmJit backend based on AsmJit code" ON)
option(SIMJIT_ENABLE_ASMJIT_X86 "Include x86 AsmJit backend" ON)
option(SIMJIT_ENABLE_ASMJIT_ARM "Include AArch64 AsmJit backend" ON)
option(SIMJIT_ENABLE_CPP "Include C++ backend" ON)
option(SIMJIT_ENABLE_LIBDIVIDE "Enable optional libdivide support from thirdparty/libdivide" ON)
option(SIMJIT_ENABLE_SERIALIZATION "Enable HIR serialization and deserialization support" OFF)
option(SIMJIT_ENABLE_NATIVE_ARCH "Build simjit itself with native CPU flags" ON)
option(SIMJIT_ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(SIMJIT_ENABLE_ASAN "Enable ASAN" OFF)
option(SIMJIT_ENABLE_UBSAN "Enable UBSAN" OFF)
option(SIMJIT_ENABLE_LTO "Enable LTO/IPO for release-style builds" ON)
set(SIMJIT_LTO_MODE "AUTO" CACHE STRING "LTO mode: AUTO, THIN, or FULL")
set_property(CACHE SIMJIT_LTO_MODE PROPERTY STRINGS AUTO THIN FULL)
if(CMAKE_SOURCE_DIR STREQUAL SIMJIT_SOURCE_ROOT)
  set(SIMJIT_ENABLE_INSTALL_DEFAULT ON)
else()
  set(SIMJIT_ENABLE_INSTALL_DEFAULT OFF)
endif()
option(SIMJIT_ENABLE_INSTALL "Install the Simjit C++ package" ${SIMJIT_ENABLE_INSTALL_DEFAULT})
set(SIMJIT_INSTALL_PROFILE "jit" CACHE STRING "Install profile: jit or toolkit")
set_property(CACHE SIMJIT_INSTALL_PROFILE PROPERTY STRINGS jit toolkit)
string(TOLOWER "${SIMJIT_INSTALL_PROFILE}" SIMJIT_INSTALL_PROFILE)
set(SIMJIT_INSTALL_PROFILE "${SIMJIT_INSTALL_PROFILE}" CACHE STRING "Install profile: jit or toolkit" FORCE)
if(NOT SIMJIT_INSTALL_PROFILE STREQUAL "jit" AND NOT SIMJIT_INSTALL_PROFILE STREQUAL "toolkit")
  message(FATAL_ERROR "SIMJIT_INSTALL_PROFILE must be 'jit' or 'toolkit'; got '${SIMJIT_INSTALL_PROFILE}'")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

function(target_enable_static_analysis TARGET_NAME)
    set(_sa_env_var "ENABLE_STATIC_ANALYSIS")

    if(NOT TARGET "${TARGET_NAME}")
        message(FATAL_ERROR
            "target_enable_static_analysis: '${TARGET_NAME}' is not a CMake target")
    endif()

    set(_sa_enabled OFF)

    if(DEFINED ENV{${_sa_env_var}})
        set(_sa_env_value "$ENV{${_sa_env_var}}")
        string(STRIP "${_sa_env_value}" _sa_env_value)
        string(TOUPPER "${_sa_env_value}" _sa_env_value_upper)

        if(NOT _sa_env_value_upper STREQUAL "" AND
           NOT _sa_env_value_upper STREQUAL "0" AND
           NOT _sa_env_value_upper STREQUAL "OFF" AND
           NOT _sa_env_value_upper STREQUAL "FALSE" AND
           NOT _sa_env_value_upper STREQUAL "NO")
            set(_sa_enabled ON)
        endif()
    endif()

    if(_sa_enabled)
        # Allow passing an alias target by resolving it to the real target.
        get_target_property(_sa_aliased_target "${TARGET_NAME}" ALIASED_TARGET)
        if(_sa_aliased_target)
            set(_sa_target "${_sa_aliased_target}")
        else()
            set(_sa_target "${TARGET_NAME}")
        endif()

        get_filename_component(_sa_source_root "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
        set(_sa_any_tool OFF)

        find_program(_sa_clang_tidy_exe NAMES clang-tidy NO_CACHE)
        if(_sa_clang_tidy_exe)
            set(_sa_clang_tidy_cmd
                "${_sa_clang_tidy_exe}"
                "--header-filter=^${_sa_source_root}/.*"
                "--exclude-header-filter=^${_sa_source_root}/src/simjit/detail/(expected|span)\\.h$"
                "--warnings-as-errors=*"
            )

            set_property(TARGET "${_sa_target}"
                PROPERTY CXX_CLANG_TIDY "${_sa_clang_tidy_cmd}")
            set(_sa_any_tool ON)
        else()
            message(WARNING
                "ENABLE_STATIC_ANALYSIS is set, but clang-tidy was not found")
        endif()

        find_program(_sa_cppcheck_exe NAMES cppcheck NO_CACHE)
        if(_sa_cppcheck_exe)
            set(_sa_cppcheck_cmd
                "${_sa_cppcheck_exe}"
                "--enable=warning,style,performance,portability"
                "--inline-suppr"
                "--suppress=missingIncludeSystem"
                "--suppress=cstyleCast"
                "--suppress=uninitMemberVarPrivate"
                "--suppress=noExplicitConstructor"
                "--suppress=preprocessorErrorDirective:*/llvm-c/DataTypes.h"
                "--suppress=*:*/src/simjit/detail/expected.h"
                "--suppress=*:*/src/simjit/detail/span.h"
                "--suppress=useStlAlgorithm"
                "--suppress=functionStatic"
                "--suppress=constVariablePointer"
                "--suppress=unreachableCode"
                "--error-exitcode=1"
            )

            get_target_property(_sa_cxx_standard "${_sa_target}" CXX_STANDARD)
            if(_sa_cxx_standard)
                list(APPEND _sa_cppcheck_cmd "--std=c++${_sa_cxx_standard}")
            elseif(CMAKE_CXX_STANDARD)
                list(APPEND _sa_cppcheck_cmd "--std=c++${CMAKE_CXX_STANDARD}")
            endif()

            set_property(TARGET "${_sa_target}"
                PROPERTY CXX_CPPCHECK "${_sa_cppcheck_cmd}")
            set(_sa_any_tool ON)
        else()
            message(WARNING
                "ENABLE_STATIC_ANALYSIS is set, but cppcheck was not found")
        endif()

        if(_sa_any_tool)
            message(STATUS
                "Static analysis enabled for target '${_sa_target}' via ${_sa_env_var}")
        endif()
    endif()
endfunction()

macro(simjit_configure_lto)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
  set(SIMJIT_LTO_FLAG "")
  set(SIMJIT_LTO_MODE_UPPER "")
  if(SIMJIT_ENABLE_LTO)
    string(TOUPPER "${SIMJIT_LTO_MODE}" SIMJIT_LTO_MODE_UPPER)

    if(SIMJIT_LTO_MODE_UPPER STREQUAL "AUTO")
      include(CheckIPOSupported)
      check_ipo_supported(RESULT SIMJIT_IPO_SUPPORTED OUTPUT SIMJIT_IPO_ERROR LANGUAGES CXX)
      if(NOT SIMJIT_IPO_SUPPORTED)
        message(FATAL_ERROR "SIMJIT_ENABLE_LTO=ON is not supported by this toolchain: ${SIMJIT_IPO_ERROR}")
      endif()
    elseif(SIMJIT_LTO_MODE_UPPER STREQUAL "THIN" OR SIMJIT_LTO_MODE_UPPER STREQUAL "FULL")
      if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        string(TOLOWER "${SIMJIT_LTO_MODE_UPPER}" SIMJIT_LTO_FLAG_KIND)
        set(SIMJIT_LTO_FLAG "-flto=${SIMJIT_LTO_FLAG_KIND}")
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND SIMJIT_LTO_MODE_UPPER STREQUAL "FULL")
        set(SIMJIT_LTO_FLAG "-flto")
      elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        message(FATAL_ERROR "SIMJIT_LTO_MODE=THIN is not supported by GCC; use AUTO or FULL")
      else()
        message(FATAL_ERROR "SIMJIT_LTO_MODE=${SIMJIT_LTO_MODE_UPPER} is not supported by ${CMAKE_CXX_COMPILER_ID}")
      endif()
    else()
      message(FATAL_ERROR "SIMJIT_LTO_MODE must be AUTO, THIN, or FULL; got ${SIMJIT_LTO_MODE}")
    endif()
  endif()
endmacro()

function(simjit_apply_lto target)
  if(NOT SIMJIT_ENABLE_LTO)
    return()
  endif()

  string(TOUPPER "${SIMJIT_LTO_MODE}" mode)
  if(mode STREQUAL "AUTO")
    set_target_properties(${target} PROPERTIES
      INTERPROCEDURAL_OPTIMIZATION_RELEASE ON
      INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON
      INTERPROCEDURAL_OPTIMIZATION_MINSIZEREL ON)
    return()
  endif()

  if(NOT SIMJIT_LTO_FLAG)
    message(FATAL_ERROR "simjit_configure_lto() must run before simjit_apply_lto(${target})")
  endif()

  target_compile_options(${target} PRIVATE
    "$<$<CONFIG:Release>:${SIMJIT_LTO_FLAG}>"
    "$<$<CONFIG:RelWithDebInfo>:${SIMJIT_LTO_FLAG}>"
    "$<$<CONFIG:MinSizeRel>:${SIMJIT_LTO_FLAG}>")

  get_target_property(target_type ${target} TYPE)
  if(NOT target_type STREQUAL "STATIC_LIBRARY")
    target_link_options(${target} PRIVATE
      "$<$<CONFIG:Release>:${SIMJIT_LTO_FLAG}>"
      "$<$<CONFIG:RelWithDebInfo>:${SIMJIT_LTO_FLAG}>"
      "$<$<CONFIG:MinSizeRel>:${SIMJIT_LTO_FLAG}>")
  endif()
endfunction()

function(simjit_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE
      -Wall
      -Wextra
      -pedantic
      -Wshadow
      -Wimplicit-fallthrough
      -Wnull-dereference
      -Wcast-align
      -Wwrite-strings
      -Wundef
      -Wnon-virtual-dtor
      -Wextra-semi
    )
  endif()
endfunction()

function(simjit_apply_sanitizers target)
  get_target_property(target_type ${target} TYPE)

  if(SIMJIT_ENABLE_ASAN)
    target_link_libraries(${target} PRIVATE -fsanitize=address)
    target_compile_options(${target} PRIVATE -fsanitize=address)
  endif()

  if(SIMJIT_ENABLE_UBSAN)
    target_link_libraries(${target} PRIVATE -fsanitize=undefined)
    target_compile_options(${target} PRIVATE -fsanitize=undefined)
  endif()

  if(SIMJIT_ENABLE_COVERAGE)
    set(coverage_flags -fprofile-arcs -ftest-coverage)
    target_compile_options(${target} PUBLIC ${coverage_flags})
    if(target_type STREQUAL "STATIC_LIBRARY")
      set_property(TARGET ${target} APPEND PROPERTY INTERFACE_LINK_OPTIONS ${coverage_flags})
    else()
      target_link_options(${target} PUBLIC ${coverage_flags})
    endif()
  endif()
endfunction()

set(SIMJIT_REQUIRED_LLVM_MAJOR 22 CACHE INTERNAL "Required LLVM major version" FORCE)

function(simjit_find_llvm_config output)
  set(candidates)
  if(DEFINED LLVM_CONFIG AND NOT LLVM_CONFIG STREQUAL "")
    list(APPEND candidates "${LLVM_CONFIG}")
  elseif(NOT "$ENV{LLVM_CONFIG}" STREQUAL "")
    list(APPEND candidates "$ENV{LLVM_CONFIG}")
  endif()
  list(APPEND candidates llvm-config "llvm-config-${SIMJIT_REQUIRED_LLVM_MAJOR}")
  list(REMOVE_DUPLICATES candidates)

  set(attempts)
  foreach(candidate IN LISTS candidates)
    unset(executable)
    if(IS_ABSOLUTE "${candidate}")
      if(EXISTS "${candidate}")
        set(executable "${candidate}")
      endif()
    else()
      find_program(executable NAMES "${candidate}" NO_CACHE)
    endif()

    if(NOT executable)
      list(APPEND attempts "${candidate}: not found")
      continue()
    endif()

    execute_process(
      COMMAND "${executable}" --version
      RESULT_VARIABLE version_result
      OUTPUT_VARIABLE version
      ERROR_VARIABLE version_error
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT version_result EQUAL 0 OR NOT version MATCHES "^([0-9]+)\\.")
      list(APPEND attempts "${executable}: could not determine version")
      continue()
    endif()
    if(NOT CMAKE_MATCH_1 STREQUAL "${SIMJIT_REQUIRED_LLVM_MAJOR}")
      list(APPEND attempts "${executable}: version ${version}")
      continue()
    endif()

    set(LLVM_CONFIG "${executable}" CACHE FILEPATH "llvm-config for LLVM ${SIMJIT_REQUIRED_LLVM_MAJOR}" FORCE)
    set(${output} "${executable}" PARENT_SCOPE)
    return()
  endforeach()

  string(JOIN ", " attempts_text ${attempts})
  message(FATAL_ERROR
    "LLVM ${SIMJIT_REQUIRED_LLVM_MAJOR} is required, but no matching llvm-config was found. "
    "Tried ${attempts_text}. Set LLVM_CONFIG to the LLVM ${SIMJIT_REQUIRED_LLVM_MAJOR} executable.")
endfunction()

function(simjit_find_llvm_clang output diagnostics llvm_bindir)
  set(candidates)
  if(DEFINED CLANG_EXECUTABLE AND NOT CLANG_EXECUTABLE STREQUAL "")
    list(APPEND candidates "${CLANG_EXECUTABLE}")
  elseif(NOT "$ENV{LLVM_CLANG}" STREQUAL "")
    list(APPEND candidates "$ENV{LLVM_CLANG}")
  endif()
  list(APPEND candidates
    "${llvm_bindir}/clang"
    clang
    "${llvm_bindir}/clang-${SIMJIT_REQUIRED_LLVM_MAJOR}"
    "clang-${SIMJIT_REQUIRED_LLVM_MAJOR}")
  list(REMOVE_DUPLICATES candidates)

  set(attempts)
  foreach(candidate IN LISTS candidates)
    unset(executable)
    if(IS_ABSOLUTE "${candidate}")
      if(EXISTS "${candidate}")
        set(executable "${candidate}")
      endif()
    else()
      find_program(executable NAMES "${candidate}" NO_CACHE)
    endif()

    if(NOT executable)
      list(APPEND attempts "${candidate}: not found")
      continue()
    endif()

    execute_process(
      COMMAND "${executable}" --version
      RESULT_VARIABLE version_result
      OUTPUT_VARIABLE version
      ERROR_VARIABLE version_error
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT version_result EQUAL 0 OR NOT version MATCHES "version[ \\t]+([0-9]+)\\.")
      list(APPEND attempts "${executable}: could not determine version")
      continue()
    endif()
    if(NOT CMAKE_MATCH_1 STREQUAL "${SIMJIT_REQUIRED_LLVM_MAJOR}")
      string(REGEX MATCH "^[^\n]*" version_line "${version}")
      list(APPEND attempts "${executable}: ${version_line}")
      continue()
    endif()

    set(CLANG_EXECUTABLE "${executable}" CACHE FILEPATH "Clang ${SIMJIT_REQUIRED_LLVM_MAJOR} executable" FORCE)
    set(${output} "${executable}" PARENT_SCOPE)
    set(${diagnostics} "" PARENT_SCOPE)
    return()
  endforeach()

  string(JOIN ", " attempts_text ${attempts})
  set(${output} "" PARENT_SCOPE)
  set(${diagnostics}
    "Clang ${SIMJIT_REQUIRED_LLVM_MAJOR} executable (tried ${attempts_text})"
    PARENT_SCOPE)
endfunction()

function(simjit_query_llvm_config prefix)
  cmake_parse_arguments(SIMJIT_LLVM "" "" "COMPONENTS" ${ARGN})
  if(NOT SIMJIT_LLVM_COMPONENTS)
    set(SIMJIT_LLVM_COMPONENTS all)
  endif()

  simjit_find_llvm_config(llvm_config)

  execute_process(
    COMMAND "${llvm_config}" --version
    OUTPUT_VARIABLE version
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${llvm_config}" --bindir
    OUTPUT_VARIABLE bindir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${llvm_config}" --libdir
    OUTPUT_VARIABLE libdir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${llvm_config}" --includedir
    OUTPUT_VARIABLE includedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${llvm_config}" --cxxflags
    OUTPUT_VARIABLE cxxflags
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${llvm_config}" --ldflags
    OUTPUT_VARIABLE ldflags
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  execute_process(
    COMMAND "${llvm_config}" --libs ${SIMJIT_LLVM_COMPONENTS}
    OUTPUT_VARIABLE libs
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)

  separate_arguments(cxxflags_list NATIVE_COMMAND "${cxxflags}")
  separate_arguments(ldflags_list NATIVE_COMMAND "${ldflags}")
  separate_arguments(libs_list NATIVE_COMMAND "${libs}")
  list(FILTER cxxflags_list EXCLUDE REGEX "^-fno-exceptions$")

  set(${prefix}_VERSION "${version}" PARENT_SCOPE)
  set(${prefix}_BINDIR "${bindir}" PARENT_SCOPE)
  set(${prefix}_LIBDIR "${libdir}" PARENT_SCOPE)
  set(${prefix}_INCLUDEDIR "${includedir}" PARENT_SCOPE)
  set(${prefix}_CXXFLAGS "${cxxflags_list}" PARENT_SCOPE)
  set(${prefix}_LDFLAGS "${ldflags_list}" PARENT_SCOPE)
  set(${prefix}_LIBS "${libs_list}" PARENT_SCOPE)
endfunction()

simjit_configure_lto()

if(TARGET simjit)
  return()
endif()

set(simjit_sources
  "${SIMJIT_SOURCE_ROOT}/src/simjit/core/expr.cpp"
  "${SIMJIT_SOURCE_ROOT}/src/simjit/core/hir.cpp"
  "${SIMJIT_SOURCE_ROOT}/src/simjit/core/mir.cpp"
  "${SIMJIT_SOURCE_ROOT}/src/simjit/core/vectorizer.cpp"
  "${SIMJIT_SOURCE_ROOT}/src/simjit/function_builder.cpp"
  "${SIMJIT_SOURCE_ROOT}/src/simjit/nullable.cpp"
)

if(SIMJIT_ENABLE_SERIALIZATION)
  list(APPEND simjit_sources "${SIMJIT_SOURCE_ROOT}/src/simjit/core/serialize.cpp")
endif()

if(SIMJIT_ENABLE_LLVM)
  list(APPEND simjit_sources "${SIMJIT_SOURCE_ROOT}/src/simjit/core/llvm/emitter.cpp")
endif()

if(SIMJIT_ENABLE_ASMJIT)
  list(APPEND simjit_sources
    "${SIMJIT_SOURCE_ROOT}/src/simjit/asmjit.cpp"
    "${SIMJIT_SOURCE_ROOT}/src/simjit/jit.cpp")

  set(ASMJIT_DIR "${SIMJIT_SOURCE_ROOT}/thirdparty/asmjit")
  set(ASMJIT_STATIC ON CACHE BOOL "Build 'asmjit' as a static library" FORCE)
  set(ASMJIT_NO_UJIT ON CACHE BOOL "Disable AsmJit's UniCompiler backend" FORCE)
  set(ASMJIT_NO_INSTALL ON CACHE BOOL "Disable AsmJit install rules" FORCE)
  set(ASMJIT_NO_X86 OFF CACHE BOOL "Disable X86/X64 backend" FORCE)
  set(ASMJIT_NO_AARCH64 OFF CACHE BOOL "Disable AArch64 backend" FORCE)

  if(SIMJIT_ENABLE_ASMJIT_ARM)
    if(NOT SIMJIT_ENABLE_ASMJIT_X86)
      set(ASMJIT_NO_X86 ON CACHE BOOL "Disable X86/X64 backend" FORCE)
    endif()
    list(APPEND simjit_sources "${SIMJIT_SOURCE_ROOT}/src/simjit/core/aj/a64.cpp")
  endif()
  if(SIMJIT_ENABLE_ASMJIT_X86)
    if(NOT SIMJIT_ENABLE_ASMJIT_ARM)
      set(ASMJIT_NO_AARCH64 ON CACHE BOOL "Disable AArch64 backend" FORCE)
    endif()
    list(APPEND simjit_sources "${SIMJIT_SOURCE_ROOT}/src/simjit/core/aj/x86.cpp")
  endif()

  add_subdirectory("${ASMJIT_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/simjit_asmjit")
endif()

if(SIMJIT_ENABLE_CPP)
  list(APPEND simjit_sources
    "${SIMJIT_SOURCE_ROOT}/src/simjit/core/cpp/x86_intrin.cpp"
    "${SIMJIT_SOURCE_ROOT}/src/simjit/core/cpp/emitter_base.cpp"
    "${SIMJIT_SOURCE_ROOT}/src/simjit/core/cpp/emitter_x86.cpp"
    "${SIMJIT_SOURCE_ROOT}/src/simjit/core/cpp/emitter_arm.cpp"
    "${SIMJIT_SOURCE_ROOT}/src/simjit/core/cpp/emitter.cpp")
endif()

add_library(simjit STATIC ${simjit_sources})
set_target_properties(simjit PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_compile_features(simjit PUBLIC cxx_std_17)

target_include_directories(simjit
  PUBLIC
    "$<BUILD_INTERFACE:${SIMJIT_SOURCE_ROOT}/src>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
if(SIMJIT_ENABLE_SERIALIZATION)
  target_include_directories(simjit SYSTEM PRIVATE "${SIMJIT_SOURCE_ROOT}/thirdparty/headerlisp")
endif()
simjit_apply_warnings(simjit)
target_compile_options(simjit PRIVATE -fexceptions)
if(SIMJIT_ENABLE_NATIVE_ARCH)
  target_compile_options(simjit PRIVATE -march=native)
endif()
simjit_apply_lto(simjit)

set(SIMJIT_DEF_ENABLE_LLVM 0)
set(SIMJIT_DEF_ENABLE_ASMJIT 0)
set(SIMJIT_DEF_ENABLE_ASMJIT_ARM 0)
set(SIMJIT_DEF_ENABLE_ASMJIT_X86 0)
set(SIMJIT_DEF_ENABLE_CPP 0)
set(SIMJIT_DEF_ENABLE_SERIALIZATION 0)
if(SIMJIT_ENABLE_LLVM)
  set(SIMJIT_DEF_ENABLE_LLVM 1)
endif()
if(SIMJIT_ENABLE_ASMJIT)
  set(SIMJIT_DEF_ENABLE_ASMJIT 1)
  if(SIMJIT_ENABLE_ASMJIT_ARM)
    set(SIMJIT_DEF_ENABLE_ASMJIT_ARM 1)
  endif()
  if(SIMJIT_ENABLE_ASMJIT_X86)
    set(SIMJIT_DEF_ENABLE_ASMJIT_X86 1)
  endif()
endif()
if(SIMJIT_ENABLE_CPP)
  set(SIMJIT_DEF_ENABLE_CPP 1)
endif()
if(SIMJIT_ENABLE_SERIALIZATION)
  set(SIMJIT_DEF_ENABLE_SERIALIZATION 1)
endif()
set(SIMJIT_DEF_USE_LIBDIVIDE 0)
if(SIMJIT_ENABLE_LIBDIVIDE)
  set(SIMJIT_DEF_USE_LIBDIVIDE 1)
endif()

target_compile_definitions(simjit
  PUBLIC
    SIMJIT_LLVM_BACKEND=${SIMJIT_DEF_ENABLE_LLVM}
    SIMJIT_ASMJIT_BACKEND=${SIMJIT_DEF_ENABLE_ASMJIT}
    SIMJIT_ASMJIT_BACKEND_ARM=${SIMJIT_DEF_ENABLE_ASMJIT_ARM}
    SIMJIT_ASMJIT_BACKEND_X86=${SIMJIT_DEF_ENABLE_ASMJIT_X86}
    SIMJIT_CPP_BACKEND=${SIMJIT_DEF_ENABLE_CPP}
    SIMJIT_USE_LIBDIVIDE=${SIMJIT_DEF_USE_LIBDIVIDE}
    SIMJIT_ENABLE_SERIALIZATION=${SIMJIT_DEF_ENABLE_SERIALIZATION}
)

if(SIMJIT_ENABLE_LIBDIVIDE)
  set(SIMJIT_LIBDIVIDE_DIR "${SIMJIT_SOURCE_ROOT}/thirdparty/libdivide")
  if(NOT EXISTS "${SIMJIT_LIBDIVIDE_DIR}/libdivide.h")
    message(FATAL_ERROR "SIMJIT_ENABLE_LIBDIVIDE=ON requires thirdparty/libdivide/libdivide.h")
  endif()
  target_include_directories(simjit SYSTEM PRIVATE "${SIMJIT_LIBDIVIDE_DIR}")
endif()

if(SIMJIT_ENABLE_LLVM)
  simjit_query_llvm_config(SIMJIT_LLVM COMPONENTS all)
  target_include_directories(simjit SYSTEM PRIVATE ${SIMJIT_LLVM_INCLUDEDIR})
  target_link_libraries(simjit PUBLIC ${SIMJIT_LLVM_LDFLAGS} ${SIMJIT_LLVM_LIBS})
  target_compile_options(simjit PRIVATE ${SIMJIT_LLVM_CXXFLAGS})
endif()

if(SIMJIT_ENABLE_ASMJIT)
  target_compile_options(asmjit PRIVATE -w)
  target_link_libraries(simjit PUBLIC asmjit::asmjit)
  get_target_property(asmjit_include_dirs asmjit::asmjit INTERFACE_INCLUDE_DIRECTORIES)
  if(asmjit_include_dirs)
    target_include_directories(simjit SYSTEM PUBLIC ${asmjit_include_dirs})
  endif()
endif()

simjit_apply_sanitizers(simjit)
target_enable_static_analysis(simjit)

if(SIMJIT_ENABLE_INSTALL)

function(simjit_install_header header)
  get_filename_component(header_dir "${header}" DIRECTORY)
  if(header_dir STREQUAL "")
    set(header_destination "${CMAKE_INSTALL_INCLUDEDIR}/simjit")
  else()
    set(header_destination "${CMAKE_INSTALL_INCLUDEDIR}/simjit/${header_dir}")
  endif()
  install(
    FILES "${SIMJIT_SOURCE_ROOT}/src/simjit/${header}"
    DESTINATION "${header_destination}")
endfunction()

function(simjit_install_notice_file notice_file notice_destination)
  if(EXISTS "${notice_file}")
    install(
      FILES "${notice_file}"
      DESTINATION "${notice_destination}")
  endif()
endfunction()

set(simjit_jit_install_headers
  simjit.h
  errors.h
  compiler.h
  jit.h
  nullable.h
  dynamic_value.h
  detail/arena.h
  detail/span.h
  detail/expected.h
)

install(
  TARGETS simjit
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")

install(
  FILES "${SIMJIT_SOURCE_ROOT}/THIRD_PARTY_NOTICES.md"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}")
install(
  FILES "${SIMJIT_SOURCE_ROOT}/LICENSE"
  DESTINATION "${CMAKE_INSTALL_DOCDIR}")
simjit_install_notice_file(
  "${SIMJIT_SOURCE_ROOT}/thirdparty/asmjit/LICENSE.md"
  "${CMAKE_INSTALL_DOCDIR}/licenses/asmjit")
simjit_install_notice_file(
  "${SIMJIT_SOURCE_ROOT}/thirdparty/asmjit/db/LICENSE.md"
  "${CMAKE_INSTALL_DOCDIR}/licenses/asmjit-db")
simjit_install_notice_file(
  "${SIMJIT_SOURCE_ROOT}/thirdparty/libdivide/LICENSE.txt"
  "${CMAKE_INSTALL_DOCDIR}/licenses/libdivide")
simjit_install_notice_file(
  "${SIMJIT_SOURCE_ROOT}/thirdparty/headerlisp/LICENSE"
  "${CMAKE_INSTALL_DOCDIR}/licenses/headerlisp")
simjit_install_notice_file(
  "${SIMJIT_SOURCE_ROOT}/thirdparty/pybind11/LICENSE"
  "${CMAKE_INSTALL_DOCDIR}/licenses/pybind11")
simjit_install_notice_file(
  "${SIMJIT_SOURCE_ROOT}/thirdparty/LICENSES/Boost-1.0.txt"
  "${CMAKE_INSTALL_DOCDIR}/licenses/boost")

if(SIMJIT_ENABLE_ASMJIT)
  install(
    TARGETS asmjit
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

if(SIMJIT_INSTALL_PROFILE STREQUAL "jit")
  foreach(header IN LISTS simjit_jit_install_headers)
    simjit_install_header("${header}")
  endforeach()
else()
  install(
    DIRECTORY "${SIMJIT_SOURCE_ROOT}/src/simjit/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/simjit"
    FILES_MATCHING
      PATTERN "*.h"
      PATTERN "*.hpp")

  if(SIMJIT_ENABLE_ASMJIT)
    install(
      DIRECTORY "${SIMJIT_SOURCE_ROOT}/thirdparty/asmjit/asmjit"
      DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
      FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")
  endif()
endif()

set(SIMJIT_INSTALL_CONFIG_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/simjit")
set(SIMJIT_CONFIG_PROFILE "${SIMJIT_INSTALL_PROFILE}")
set(SIMJIT_CONFIG_ENABLE_ASMJIT "${SIMJIT_DEF_ENABLE_ASMJIT}")
set(SIMJIT_CONFIG_INSTALLS_ASMJIT_HEADERS 0)
if(SIMJIT_INSTALL_PROFILE STREQUAL "toolkit" AND SIMJIT_ENABLE_ASMJIT)
  set(SIMJIT_CONFIG_INSTALLS_ASMJIT_HEADERS 1)
endif()
set(SIMJIT_CONFIG_SIMJIT_LIBRARY_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}simjit${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(SIMJIT_CONFIG_ASMJIT_LIBRARY_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}asmjit${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(SIMJIT_CONFIG_COMPILE_DEFINITIONS
  "SIMJIT_LLVM_BACKEND=${SIMJIT_DEF_ENABLE_LLVM}"
  "SIMJIT_ASMJIT_BACKEND=${SIMJIT_DEF_ENABLE_ASMJIT}"
  "SIMJIT_ASMJIT_BACKEND_ARM=${SIMJIT_DEF_ENABLE_ASMJIT_ARM}"
  "SIMJIT_ASMJIT_BACKEND_X86=${SIMJIT_DEF_ENABLE_ASMJIT_X86}"
  "SIMJIT_CPP_BACKEND=${SIMJIT_DEF_ENABLE_CPP}"
  "SIMJIT_USE_LIBDIVIDE=${SIMJIT_DEF_USE_LIBDIVIDE}"
  "SIMJIT_ENABLE_SERIALIZATION=${SIMJIT_DEF_ENABLE_SERIALIZATION}")
set(SIMJIT_CONFIG_LLVM_LDFLAGS "")
set(SIMJIT_CONFIG_LLVM_LIBS "")
if(SIMJIT_ENABLE_LLVM)
  set(SIMJIT_CONFIG_LLVM_LDFLAGS "${SIMJIT_LLVM_LDFLAGS}")
  set(SIMJIT_CONFIG_LLVM_LIBS "${SIMJIT_LLVM_LIBS}")
endif()

configure_package_config_file(
  "${SIMJIT_SOURCE_ROOT}/cmake/simjitConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/simjitConfig.cmake"
  INSTALL_DESTINATION "${SIMJIT_INSTALL_CONFIG_DIR}"
  PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/simjitConfigVersion.cmake.raw"
  VERSION "${SIMJIT_VERSION}"
  COMPATIBILITY SameMinorVersion)
file(READ
  "${CMAKE_CURRENT_BINARY_DIR}/simjitConfigVersion.cmake.raw"
  SIMJIT_CONFIG_VERSION_CONTENTS)
file(WRITE
  "${CMAKE_CURRENT_BINARY_DIR}/simjitConfigVersion.cmake"
  "# This file is part of Simjit project <https://simjit.org>\n"
  "#\n"
  "# See LICENSE for license and copyright information\n"
  "# SPDX-License-Identifier: Zlib\n\n"
  "${SIMJIT_CONFIG_VERSION_CONTENTS}")
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/simjitConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/simjitConfigVersion.cmake"
  DESTINATION "${SIMJIT_INSTALL_CONFIG_DIR}")

endif()
