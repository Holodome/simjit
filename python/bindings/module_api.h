// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "simjit/jit.h"
#include "simjit/simjit.h"

namespace simjit_python {
namespace py = pybind11;
namespace sj = simjit;

class SafetyCheckFailed : public std::runtime_error {
public:
    SafetyCheckFailed() : std::runtime_error("Simjit safety check failed") {}
};

struct SessionState {
    py::module_ mod;
    sj::jit::JitContext jit;
    size_t generation = 0;

    explicit SessionState(const py::module_ &mod);
    SessionState(const py::module_ &mod, sj::Arch arch);
};

struct PreparedKernel {
    std::shared_ptr<SessionState> state;
    void *fn = nullptr;
    size_t n = 0;
    std::vector<void *> ptrs{};
    size_t generation = 0;
    py::object buffers;
    py::object resolved;
    std::string identifier{};

    void run();
};

struct PreparedProgramState;

struct PreparedProgram {
    std::shared_ptr<PreparedProgramState> state;

    void run(py::handle inputs = py::none());
    py::dict run_fresh();
    py::tuple run_fresh_values();
    void release_outputs();
    py::dict output_buffers() const;
    py::dict result() const;
    const std::string &identifier() const;
};

struct Session {
    std::shared_ptr<SessionState> state;

    explicit Session(const py::module_ &mod, std::string_view arch_name = "native");

    void run_native(const py::dict &buffers, py::handle outputs, size_t n);
    py::dict run_program(py::handle outputs, py::handle inputs, std::string_view output_kind);
    PreparedProgram prepare_program(py::handle outputs, py::handle inputs, std::string_view output_kind);
    bool release(std::string_view identifier);
    void clear();
};

void run_native(const py::module_ &mod, const py::dict &buffers, py::handle outputs, size_t n);
size_t infer_native_length(const py::module_ &mod, const py::dict &buffers, py::handle outputs);
py::dict run_program(const py::module_ &mod, py::handle outputs, py::handle inputs, std::string_view output_kind);
py::dict inspect_native_function(const py::dict &buffers, py::handle outputs, size_t n, std::string_view policy_name,
                                 std::string_view arch_name);
py::dict inspect_program_function(py::handle outputs, py::handle inputs, std::string_view output_kind,
                                  std::string_view policy_name, std::string_view arch_name);
py::dict inspect_schema_function(py::handle outputs, py::handle schema, std::string_view output_kind,
                                 std::string_view policy_name, std::string_view arch_name);
py::dict benchmark_hir_jit_compile(py::handle outputs, py::handle inputs, std::string_view output_kind,
                                   std::string_view backend, sj::jit::CompilePolicy policy, std::string_view llvm_opt,
                                   std::string_view arch_name, int warmups, int runs);
py::dict inspect_serialized_function(std::string_view serialized, std::string_view policy_name,
                                     std::string_view arch_name);

} // namespace simjit_python
