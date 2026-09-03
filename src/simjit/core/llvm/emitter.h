// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/compiler.h"

#include <cstdint>
#include <memory>
#include <string_view>

#if !SIMJIT_LLVM_BACKEND
#error "simjit/core/llvm/emitter.h requires SIMJIT_LLVM_BACKEND"
#endif

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace simjit {
namespace jit {
enum class CompilePolicy : uint8_t;
} // namespace jit

namespace llvm_backend {

enum class LLVMOptLevel : uint8_t {
    O1,
    O3
};

// Module must be destroyed before the context it references. Member
// destruction is in reverse declaration order, so keep context first.
struct LLVMModuleOwner {
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::Module> module;

    LLVMModuleOwner() noexcept;
    LLVMModuleOwner(std::unique_ptr<llvm::LLVMContext> context_init,
                    std::unique_ptr<llvm::Module> module_init) noexcept;
    ~LLVMModuleOwner() noexcept;

    LLVMModuleOwner(const LLVMModuleOwner &) = delete;
    LLVMModuleOwner(LLVMModuleOwner &&) noexcept;
    LLVMModuleOwner &operator=(const LLVMModuleOwner &) = delete;
    LLVMModuleOwner &operator=(LLVMModuleOwner &&) noexcept;
};

// Internal LLVM backend session. The session is deliberately single-threaded;
// callers provide concurrency with one persistent session per worker.
class LLVMSession {
public:
    LLVMSession(Arch arch, LLVMOptLevel opt_level);
    ~LLVMSession() noexcept;

    LLVMSession(const LLVMSession &) = delete;
    LLVMSession(LLVMSession &&) = delete;
    LLVMSession &operator=(const LLVMSession &) = delete;
    LLVMSession &operator=(LLVMSession &&) = delete;

    Arch arch() const noexcept;
    LLVMOptLevel opt_level() const noexcept;

    // Sets native target metadata, verifies, and runs the configured pass
    // pipeline. Callers with an already optimized Clang module may skip this.
    void optimize_module(LLVMModuleOwner &owner);
    // Sets native target metadata and transfers ownership to ORC.
    void add_module(LLVMModuleOwner owner);
    // Lookup forces materialization and returns an executable pointer.
    void *lookup(std::string_view symbol);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Low-level construction boundaries. Session setup is intentionally separate
// so benchmarks can exclude it from their measured interval.
LLVMModuleOwner build_llvm_module(const mir::Function *function);
LLVMModuleOwner parse_llvm_ir(std::string_view ir);

// Canonical compilation boundaries used by benchmarks. Every function ends at
// executable-pointer lookup and therefore includes forced ORC materialization.
void *compile_hir(const hir::Function *function, LLVMSession &session, jit::CompilePolicy policy);
void *compile_mir(const mir::Function *function, LLVMSession &session);
void *compile_ir(std::string_view ir, std::string_view symbol, LLVMSession &session);

} // namespace llvm_backend
} // namespace simjit
