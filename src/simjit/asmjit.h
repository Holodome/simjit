// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <cstddef>
#include <memory>

#include "simjit/simjit.h"

#if !SIMJIT_ASMJIT_BACKEND
#error "simjit/asmjit.h requires SIMJIT_ASMJIT_BACKEND"
#endif

#include "asmjit/core.h"
#include "asmjit/core/jitruntime.h"

namespace simjit {

struct AsmjitAllocatorStatistics {
    size_t block_count = 0;
    size_t allocation_count = 0;
    size_t used_memory = 0;
    size_t reserved_memory = 0;
    size_t overhead_memory = 0;
};

class AsmjitSession {
public:
    explicit AsmjitSession(Arch arch = Arch::Native);
    ~AsmjitSession() noexcept;

    AsmjitSession(const AsmjitSession &) = delete;
    AsmjitSession(AsmjitSession &&) = delete;
    AsmjitSession &operator=(const AsmjitSession &) = delete;
    AsmjitSession &operator=(AsmjitSession &&) = delete;

    Arch arch() const noexcept { return arch_; }
    void reset() noexcept;

    asmjit::CodeHolder &code_holder() noexcept { return code_holder_; }
    const asmjit::CodeHolder &code_holder() const noexcept { return code_holder_; }
    asmjit::BaseCompiler &compiler() noexcept { return *compiler_; }
    const asmjit::BaseCompiler &compiler() const noexcept { return *compiler_; }

    Arch host_arch() const;
    bool host_supports_x86_backend() const noexcept;
    bool host_supports_vectorization() const noexcept;

    void *add_compiled_function();
    void release_compiled_function(void *fn_ptr);
    AsmjitAllocatorStatistics allocator_statistics() const noexcept;

private:
    Arch arch_;
    asmjit::CodeHolder code_holder_{};
    asmjit::JitRuntime runtime_{};
    std::unique_ptr<asmjit::BaseCompiler> compiler_{};
    std::unique_ptr<asmjit::ErrorHandler> error_handler_{};
};

} // namespace simjit
