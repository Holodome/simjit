// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/asmjit.h"

#include "simjit/detail/base.h"

#if SIMJIT_ASMJIT_BACKEND_ARM
#include "asmjit/a64.h"
#endif
#if SIMJIT_ASMJIT_BACKEND_X86
#include "asmjit/x86/x86compiler.h"
#endif

namespace simjit {

namespace {

#define aj_messed_up(...) simjit_exception(ErrorModule::AsmJit, {}, {}, __VA_ARGS__)

class AjErrorHandler : public asmjit::ErrorHandler {
public:
    void handle_error(asmjit::Error, const char *message, asmjit::BaseEmitter *) override {
        aj_messed_up("AsmJit error: %s", message);
    }
};

static asmjit::Environment create_environment(Arch arch) {
    if (is_x86_arch(arch)) {
#if SIMJIT_ASMJIT_BACKEND_X86
        return asmjit::Environment(asmjit::Arch::kX64);
#else
        aj_messed_up("x86 backend is not available");
#endif
    }
    if (arch == Arch::Arm64_NEON) {
#if SIMJIT_ASMJIT_BACKEND_ARM
        return asmjit::Environment(asmjit::Arch::kAArch64);
#else
        aj_messed_up("a64 backend is not available");
#endif
    }
    aj_messed_up("unknown architecture");
}

static std::unique_ptr<asmjit::BaseCompiler> create_compiler(Arch arch) {
    if (is_x86_arch(arch)) {
#if SIMJIT_ASMJIT_BACKEND_X86
        return std::make_unique<asmjit::x86::Compiler>();
#else
        aj_messed_up("x86 backend is not available");
#endif
    }
    if (arch == Arch::Arm64_NEON) {
#if SIMJIT_ASMJIT_BACKEND_ARM
        return std::make_unique<asmjit::a64::Compiler>();
#else
        aj_messed_up("a64 backend is not available");
#endif
    }
    aj_messed_up("unknown architecture");
}

static Arch map_host_arch(asmjit::Arch arch) {
    switch (arch) {
#if SIMJIT_ASMJIT_BACKEND_X86
    case asmjit::Arch::kX64: return Arch::Amd64_AVX512;
#endif
#if SIMJIT_ASMJIT_BACKEND_ARM
    case asmjit::Arch::kAArch64: return Arch::Arm64_NEON;
#endif
    default: aj_messed_up("unsupported host architecture %d", (int)arch);
    }
}

static bool runtime_supports_vectorization(const asmjit::JitRuntime &runtime) noexcept {
    switch (runtime.arch()) {
#if SIMJIT_ASMJIT_BACKEND_X86
    case asmjit::Arch::kX64: {
        const auto &feat = runtime.cpu_features();
        const auto &x86 = feat.x86();
        return x86.has_avx512_bitalg() && x86.has_avx512_bw() && x86.has_avx512_cd() && x86.has_avx512_dq() &&
               x86.has_avx512_f() && x86.has_avx512_vbmi() && x86.has_avx512_vbmi2() && x86.has_avx512_vl() &&
               x86.has_avx512_vpopcntdq() && x86.has_gfni();
    }
#endif
#if SIMJIT_ASMJIT_BACKEND_ARM
    case asmjit::Arch::kAArch64: return true;
#endif
    default: return false;
    }
}

} // namespace

AsmjitSession::AsmjitSession(Arch arch) : arch_(arch) {
    if (auto err = code_holder_.init(create_environment(arch_)); err != asmjit::kErrorOk) {
        aj_messed_up("failed to init code holder: %s", asmjit::DebugUtils::error_as_string(err));
    }
    error_handler_ = std::make_unique<AjErrorHandler>();
    code_holder_.set_error_handler(error_handler_.get());
    compiler_ = create_compiler(arch_);
    code_holder_.attach(compiler_.get());
}

AsmjitSession::~AsmjitSession() noexcept = default;

void AsmjitSession::reset() noexcept {
    code_holder_.reinit();
}

Arch AsmjitSession::host_arch() const {
    // map_host_arch can throw but we already checked the arch
    return map_host_arch(runtime_.arch());
}

bool AsmjitSession::host_supports_vectorization() const noexcept {
    return runtime_supports_vectorization(runtime_);
}

bool AsmjitSession::host_supports_x86_backend() const noexcept {
#if SIMJIT_ASMJIT_BACKEND_X86
    if (runtime_.arch() != asmjit::Arch::kX64) { return false; }
    const auto &feat = runtime_.cpu_features();
    const auto &x86 = feat.x86();
    return x86.has_bmi2() && x86.has_bmi();
#else
    return false;
#endif
}

void *AsmjitSession::add_compiled_function() {
    if (arch_ != host_arch() && !(is_x86_arch(arch_) && is_x86_arch(host_arch()))) {
        aj_messed_up("cannot add compiled code for non-host architecture to JIT runtime");
    }
    void *fn_ptr = nullptr;
    if (asmjit::Error err = runtime_.add(&fn_ptr, &code_holder_); err != asmjit::Error::kOk) {
        aj_messed_up("failed to add JIT function: %s", asmjit::DebugUtils::error_as_string(err));
    }
    return fn_ptr;
}

void AsmjitSession::release_compiled_function(void *fn_ptr) {
    if (asmjit::Error err = runtime_.release(fn_ptr); err != asmjit::Error::kOk) {
        aj_messed_up("failed to release JIT function: %s", asmjit::DebugUtils::error_as_string(err));
    }
}

AsmjitAllocatorStatistics AsmjitSession::allocator_statistics() const noexcept {
    auto inner = runtime_.allocator().statistics();
    AsmjitAllocatorStatistics result;
    result.allocation_count = inner.allocation_count();
    result.block_count = inner.block_count();
    result.overhead_memory = inner.overhead_size();
    result.reserved_memory = inner.reserved_size();
    result.used_memory = inner.used_size();
    return result;
}

} // namespace simjit
