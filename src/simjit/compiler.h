// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/detail/arena.h"
#include "simjit/detail/expected.h"
#include "simjit/simjit.h"

#include <vector>

namespace simjit {

namespace hir {
struct Function;
struct Step;

std::string print_function(const Function *);
} // namespace hir

#if SIMJIT_ASMJIT_BACKEND
class AsmjitSession;
#endif

namespace vect {
struct Function;

nonstd::expected<Function *, ErrorInfo> try_hir_to_vect(const hir::Function *func);
Function *hir_to_vect(const hir::Function *func);

std::string print_function(const Function *);
} // namespace vect

namespace mir {
struct Function;

Function *hir_to_mir(const hir::Function *func);
Function *vect_to_mir(const vect::Function *vect);

std::string print_function(const Function *);
} // namespace mir

static constexpr size_t MaxFunctionArgumentCount = 32;

struct BuildLimits {
    size_t max_hir_roots = 32;
    size_t max_hir_live_steps = 256;
    size_t max_vector_roots = 16;
    size_t max_argument_count = MaxFunctionArgumentCount;
    size_t max_cached_functions = SIZE_MAX;
};

struct Context {
    Context() = delete;
    explicit Context(MemoryArena &arena_init) : arena(&arena_init) {}
    Context(MemoryArena &ar, std::string_view n) : arena(&ar), symbol_name(n) {}
    Context(MemoryArena &ar, std::string_view n, CodeTransformations o, Arch a)
        : arena(&ar), symbol_name(n), transformations(o), arch(a) {}

    MemoryArena *arena = nullptr;

    // Function name/symbol name in binary
    std::string symbol_name = "expr";

    CodeTransformations transformations = CodeTransformations::All;

    Arch arch = Arch::Native;

    BuildLimits build_limits{};
};

// Prepare function for compilation to specific target. This performs vectorization.
// This is mostly a convenience shortcut when direct access to vect::Function is not required.
// For debuggability it is advised to implement this transformation manually, in order to be able to print vectorization
// information.
inline mir::Function *lower_vectorized(const hir::Function *function) {
    const vect::Function *vec = vect::hir_to_vect(function);
    return mir::vect_to_mir(vec);
}

inline nonstd::expected<mir::Function *, ErrorInfo> try_lower_vectorized(const hir::Function *function) {
    auto vec = vect::try_hir_to_vect(function);
    if (!vec) { return nonstd::unexpected<ErrorInfo>(std::move(vec.error())); }
    try {
        return mir::vect_to_mir(vec.value());
    } catch (const SimjitException &e) {
        return nonstd::unexpected<ErrorInfo>(e.info());
    } catch (const std::exception &e) {
        return nonstd::unexpected<ErrorInfo>(
            ErrorInfo{ErrorModule::MIR, ErrorKind::InternalInvariant, ErrorSubKind::ExternalFailure, e.what()});
    }
}

// Prepare function for compilation to specific target. This does not perform vectorization - only scalar code can be
// executed. Provided to mirror 'lower_vectorized'.
inline mir::Function *lower_scalar(const hir::Function *function) {
    return mir::hir_to_mir(function);
}

#if SIMJIT_CPP_BACKEND
// Compile function to C++. Returns C++ code as string.
std::string emit_cpp_source(const mir::Function *function);
#endif

#if SIMJIT_LLVM_BACKEND
// Compile function to LLVM bitcode. Returns LLVM code as string.
std::string emit_llvm_ir(const mir::Function *function);
#endif

#if SIMJIT_ASMJIT_BACKEND
#if !SIMJIT_ASMJIT_BACKEND_ARM && !SIMJIT_ASMJIT_BACKEND_X86
#error If asmjit backend is enabled, at least one platform must be selected
#endif

struct AsmjitCompileOptions {
    bool emit_machine_code = true;
    bool emit_asm_code = false;
    AsmjitSession *session = nullptr;
};

struct AsmjitCompileResult {
    std::vector<uint8_t> machine_code{};
    std::string asm_code{};
};

void compile_asmjit(const mir::Function *function, const AsmjitCompileOptions &opts, AsmjitCompileResult &result);

#if SIMJIT_ASMJIT_BACKEND_ARM
void compile_asmjit_arm(const mir::Function *func, const AsmjitCompileOptions &opts, AsmjitCompileResult &result);
#endif
#if SIMJIT_ASMJIT_BACKEND_X86
void compile_asmjit_x86(const mir::Function *func, const AsmjitCompileOptions &opts, AsmjitCompileResult &result);
#endif

#endif

#if SIMJIT_ENABLE_SERIALIZATION
std::string serialize(const hir::Function *func);
void deserialize(std::string_view str, FunctionBuilder &builder);
#endif

} // namespace simjit
