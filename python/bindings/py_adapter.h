// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "module_api.h"

namespace simjit_python {

void configure_session_jit_context(sj::jit::JitContext &jit);
sj::Arch parse_session_arch(std::string_view arch_name);
void run_prepared_kernel_state(const PreparedKernel &kernel);

void run_explicit_native(SessionState &state, const py::dict &buffers, py::handle outputs, size_t n);
size_t infer_native_length_from_buffers(const py::dict &buffers, py::handle outputs);
py::dict run_native_runtime_program_with_session(std::shared_ptr<SessionState> state, py::handle outputs,
                                                 py::handle inputs, std::string_view output_kind);

std::shared_ptr<PreparedProgramState> prepare_program_state(std::shared_ptr<SessionState> state, py::handle outputs,
                                                            py::handle inputs, std::string_view output_kind);
void run_prepared_program_state(PreparedProgramState &state, py::handle inputs);
py::dict run_prepared_program_fresh_state(PreparedProgramState &state);
py::tuple run_prepared_program_fresh_values_state(PreparedProgramState &state);
void release_prepared_program_outputs(PreparedProgramState &state);
py::dict prepared_program_output_buffers(const PreparedProgramState &state);
py::dict prepared_program_result(const PreparedProgramState &state);
const std::string &prepared_program_identifier(const PreparedProgramState &state);

} // namespace simjit_python
