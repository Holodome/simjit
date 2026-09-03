// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/cpp/emitter_internal.h"

namespace simjit {
namespace cpp_backend {

static std::unique_ptr<CppEmitterBase> make_cpp_emitter(const Function *func) {
    if (func->ctx->arch == Arch::Arm64_NEON) { return make_arm_neon_cpp_emitter(func); }
    return make_x86_cpp_emitter(func);
}

static std::string emit_cpp_source(const Function *func) {
    return make_cpp_emitter(func)->emit_source();
}

} // namespace cpp_backend

std::string emit_cpp_source(const mir::Function *func) {
    return cpp_backend::emit_cpp_source(func);
}

} // namespace simjit
