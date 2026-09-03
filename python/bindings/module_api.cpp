// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "module_api.h"

#include "py_adapter.h"

#include <stdexcept>

namespace simjit_python {

SessionState::SessionState(const py::module_ &module) : mod(module), jit() {
    configure_session_jit_context(jit);
}

SessionState::SessionState(const py::module_ &module, sj::Arch arch) : mod(module), jit(arch) {
    configure_session_jit_context(jit);
}

void PreparedKernel::run() {
    run_prepared_kernel_state(*this);
}

void PreparedProgram::run(py::handle inputs) {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    run_prepared_program_state(*state, inputs);
}

py::dict PreparedProgram::run_fresh() {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    return run_prepared_program_fresh_state(*state);
}

py::tuple PreparedProgram::run_fresh_values() {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    return run_prepared_program_fresh_values_state(*state);
}

void PreparedProgram::release_outputs() {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    release_prepared_program_outputs(*state);
}

py::dict PreparedProgram::output_buffers() const {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    return prepared_program_output_buffers(*state);
}

py::dict PreparedProgram::result() const {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    return prepared_program_result(*state);
}

const std::string &PreparedProgram::identifier() const {
    if (state == nullptr) { throw std::runtime_error("prepared program is not initialized"); }
    return prepared_program_identifier(*state);
}

Session::Session(const py::module_ &mod, std::string_view arch_name)
    : state(std::make_shared<SessionState>(mod, parse_session_arch(arch_name))) {
}

void Session::run_native(const py::dict &buffers, py::handle outputs, size_t n) {
    run_explicit_native(*state, buffers, outputs, n);
}

py::dict Session::run_program(py::handle outputs, py::handle inputs, std::string_view output_kind) {
    return run_native_runtime_program_with_session(state, outputs, inputs, output_kind);
}

PreparedProgram Session::prepare_program(py::handle outputs, py::handle inputs, std::string_view output_kind) {
    return PreparedProgram{prepare_program_state(state, outputs, inputs, output_kind)};
}

bool Session::release(std::string_view identifier) {
    bool released = state->jit.delete_cached_function(identifier);
    if (released) { ++state->generation; }
    return released;
}

void Session::clear() {
    state->jit.clear();
    ++state->generation;
}

void run_native(const py::module_ &mod, const py::dict &buffers, py::handle outputs, size_t n) {
    Session session(mod);
    session.run_native(buffers, outputs, n);
}

size_t infer_native_length(const py::module_ &, const py::dict &buffers, py::handle outputs) {
    return infer_native_length_from_buffers(buffers, outputs);
}

py::dict run_program(const py::module_ &mod, py::handle outputs, py::handle inputs, std::string_view output_kind) {
    Session session(mod);
    return session.run_program(outputs, inputs, output_kind);
}

} // namespace simjit_python
