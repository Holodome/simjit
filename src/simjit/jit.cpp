// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/jit.h"
#include "simjit/asmjit.h"
#include "simjit/compiler.h"
#include "simjit/core/vectorizer.h"

#include <array>
#include <chrono>
#include <unordered_map>
#include <utility>

namespace simjit {
namespace jit {

namespace {

#define jit_failure(_subkind, ...) simjit_exception(ErrorModule::JIT, ErrorKind::JitFailure, _subkind, __VA_ARGS__)

static const char *show_arch(Arch arch) {
    switch (arch) {
    case Arch::Amd64_AVX512: return "x86-avx512";
    case Arch::Amd64_AVX512_YMM: return "x86-avx512-ymm";
    case Arch::Arm64_NEON: return "arm64-neon";
    }
    return "unknown";
}

static bool can_jit_target_on_host(Arch target, Arch host) {
    return target == host || (is_x86_arch(target) && is_x86_arch(host));
}

struct FunctionRecord {
    void *fn_ptr{};
    std::vector<ArgumentDecl> args;
    std::vector<hir::ArgumentUsage> usages;
    std::string identifier;
};

class FunctionCache {
public:
    FunctionRecord *find(std::string_view identifier) {
        auto [first, last] = records_.equal_range(hash_identifier(identifier));
        for (auto it = first; it != last; ++it) {
            if (std::string_view{it->second.identifier} == identifier) { return &it->second; }
        }
        return nullptr;
    }

    const FunctionRecord *find(std::string_view identifier) const {
        auto [first, last] = records_.equal_range(hash_identifier(identifier));
        for (auto it = first; it != last; ++it) {
            if (std::string_view{it->second.identifier} == identifier) { return &it->second; }
        }
        return nullptr;
    }

    void insert_or_assign(std::string_view identifier, FunctionRecord record) {
        record.identifier = std::string{identifier};

        auto [first, last] = records_.equal_range(hash_identifier(identifier));
        for (auto it = first; it != last; ++it) {
            if (std::string_view{it->second.identifier} == identifier) {
                it->second = std::move(record);
                return;
            }
        }
        records_.emplace(hash_identifier(identifier), std::move(record));
    }

    bool erase(std::string_view identifier) {
        auto [first, last] = records_.equal_range(hash_identifier(identifier));
        for (auto it = first; it != last; ++it) {
            if (std::string_view{it->second.identifier} == identifier) {
                records_.erase(it);
                return true;
            }
        }
        return false;
    }

    size_t size() const { return records_.size(); }

    std::vector<std::string> identifiers() const {
        std::vector<std::string> out;
        out.reserve(records_.size());
        for (const auto &[_, record] : records_) {
            out.push_back(record.identifier);
        }
        return out;
    }

private:
    static size_t hash_identifier(std::string_view identifier) { return std::hash<std::string_view>{}(identifier); }

    std::unordered_multimap<size_t, FunctionRecord> records_{};
};

} // namespace

static const char *policy_name(CompilePolicy policy) {
    switch (policy) {
    case CompilePolicy::BestEffort: return "best-effort";
    case CompilePolicy::Vectorized: return "vectorized";
    case CompilePolicy::Scalar: return "scalar";
    }
    SIMJIT_UNREACHABLE();
}

static const char *show_jit_argument_role(JitArgumentRole role) {
    switch (role) {
    case JitArgumentRole::InputArray: return "input array";
    case JitArgumentRole::InputConstant: return "input constant";
    case JitArgumentRole::OutputArray: return "output array";
    case JitArgumentRole::OutputScalar: return "output scalar";
    case JitArgumentRole::OutputSafetyCheck: return "output safety check";
    }
    SIMJIT_UNREACHABLE();
}

static JitArgumentRole derived_jit_argument_role(const hir::ArgumentUsage &usage) {
    if (bool(usage & hir::ArgumentUsage::SafetyCheck)) { return JitArgumentRole::OutputSafetyCheck; }
    if (bool(usage & hir::ArgumentUsage::AggregateWrite)) { return JitArgumentRole::OutputScalar; }
    if (bool(usage & (hir::ArgumentUsage::SequentialWrite | hir::ArgumentUsage::RandomWrite |
                      hir::ArgumentUsage::AppendWrite))) {
        return JitArgumentRole::OutputArray;
    }
    if (bool(usage & hir::ArgumentUsage::SplatRead) && !bool(usage & hir::ArgumentUsage::ArrayRead)) {
        return JitArgumentRole::InputConstant;
    }
    return JitArgumentRole::InputArray;
}

static bool can_best_effort_fallback(const ErrorInfo &error) {
    bool vectorizer_rejection =
        (error.module == ErrorModule::Vectorizer) &&
        ((error.kind == ErrorKind::VectorizationFailed) ||
         (error.kind == ErrorKind::Unsupported && error.subkind == ErrorSubKind::UnsupportedFeature));
    bool backend_rejection =
        (error.kind == ErrorKind::Unsupported) && (error.subkind == ErrorSubKind::UnsupportedBackendFeature);
    return vectorizer_rejection || backend_rejection;
}

static void typecheck_function(std::string_view identifier, size_t decl_count, const ArgumentDecl *decls,
                               const hir::ArgumentUsage *usages, const CallerInfo *caller) {
    if (caller == nullptr) { return; }

    if (decl_count != caller->expected_arg_count) {
        jit_failure(ErrorSubKind::ArgumentMismatch, "%.*s Argument count mismatch: expected %zu, got %zu",
                    (int)identifier.length(), identifier.data(), caller->expected_arg_count, decl_count);
    }
    for (size_t i = 0; i < decl_count; ++i) {
        const ArgumentDecl &arg = decls[i];
        JitArgumentRole actual_role = derived_jit_argument_role(usages[i]);
        if (caller->expected_roles[i] != actual_role) {
            jit_failure(ErrorSubKind::ArgumentMismatch, "[%.*s] Argument [%zu] role mismatch: expected %s, got %s",
                        (int)identifier.length(), identifier.data(), i,
                        show_jit_argument_role(caller->expected_roles[i]), show_jit_argument_role(actual_role));
        }
        if (caller->expected_types[i] != arg.dtype) {
            jit_failure(ErrorSubKind::ArgumentMismatch, "[%.*s] Argument [%zu] type mismatch: expected %s, got %s",
                        (int)identifier.length(), identifier.data(), i, show_scalar_dtype(caller->expected_types[i]),
                        show_scalar_dtype(arg.dtype));
        }
    }
}

class JitContextImpl {
public:
    JitContextImpl() = delete;
    explicit JitContextImpl(Arch arch) : ctx_(arena_, "expr", CodeTransformations::All, arch), session_(arch) {
        Arch host_arch = session_.host_arch();
        if (!can_jit_target_on_host(ctx_.arch, host_arch)) {
            jit_failure(ErrorSubKind::UnsupportedHostFeature,
                        "JIT target %s cannot execute on host %s; use inspection emitters for "
                        "cross-target code generation",
                        show_arch(ctx_.arch), show_arch(host_arch));
        }
        if (is_x86_arch(host_arch)) {
            if (!session_.host_supports_x86_backend()) {
                jit_failure(ErrorSubKind::UnsupportedHostFeature,
                            "Amd64 machine should support BMI2 instruction set (Haswell and newer)");
            }
            host_supports_vectorization_ = session_.host_supports_vectorization();
            ctx_.host_supported_vector_special_ops = session_.host_supported_vector_special_ops();
        } else if (host_arch == Arch::Arm64_NEON) {
            host_supports_vectorization_ = true;
            ctx_.host_supported_vector_special_ops = session_.host_supported_vector_special_ops();
        } else {
            jit_failure(ErrorSubKind::UnsupportedHostFeature, "Unknown host architecture %d", (int)host_arch);
        }
    }

    void *call_asmjit(const mir::Function *func, bool emit_machine_code, bool emit_asm, AsmjitCompileResult &result) {
        AsmjitCompileOptions opts{emit_machine_code, emit_asm, &session_};
        compile_asmjit(func, opts, result);
        return session_.add_compiled_function();
    }

    void *compile(const mir::Function *func) {
        AsmjitCompileResult result{};
        return call_asmjit(func, false, false, result);
    }

    void require_host_vectorization() const {
        if (!host_supports_vectorization_) {
            jit_failure(ErrorSubKind::UnsupportedHostFeature, "Host machine does not support vectorization");
        }
    }

    mir::Function *lower_function(const hir::Function *fn) {
        switch (policy_) {
        case CompilePolicy::BestEffort:
            if (host_supports_vectorization_ && !fn->is_scalar_only()) {
                auto vectorized = try_lower_vectorized(fn);
                if (vectorized) { return vectorized.value(); }
                if (!can_best_effort_fallback(vectorized.error())) {
                    throw SimjitException(std::move(vectorized.error()));
                }
            }
            return lower_scalar(fn);
        case CompilePolicy::Vectorized:
            require_host_vectorization();
            if (fn->is_scalar_only()) {
                if (fn->vectorization_hint == hir::VectorizationHint::UnsupportedSpecialOps) {
                    jit_failure(ErrorSubKind::UnsupportedFeature,
                                "Failed to compile vectorized: unsupported vector special operations: %s",
                                hir::show_special_ops(fn->unsupported_vector_special_ops()).c_str());
                }
                jit_failure(ErrorSubKind::UnsupportedFeature,
                            "Failed to compile vectorized: function is marked as only scalar");
            }
            return lower_vectorized(fn);
        case CompilePolicy::Scalar: return lower_scalar(fn);
        }
        SIMJIT_UNREACHABLE();
    }

    void *lower_and_compile_function(const hir::Function *fn) {
        switch (policy_) {
        case CompilePolicy::BestEffort:
            if (host_supports_vectorization_ && !fn->is_scalar_only()) {
                auto vectorized = try_lower_vectorized(fn);
                if (vectorized) {
                    try {
                        return compile(vectorized.value());
                    } catch (const SimjitException &e) {
                        if (!can_best_effort_fallback(e.info())) { throw; }
                        if (debug_options_.record_vectorization_fail_exception) {
                            debug_snapshot_.vectorization_exception = e.info().verbose();
                        }
                    }
                } else if (can_best_effort_fallback(vectorized.error())) {
                    if (debug_options_.record_vectorization_fail_exception) {
                        debug_snapshot_.vectorization_exception = vectorized.error().verbose();
                    }
                } else {
                    throw SimjitException(std::move(vectorized.error()));
                }
            }
            return compile(lower_scalar(fn));
        case CompilePolicy::Vectorized:
            require_host_vectorization();
            if (fn->is_scalar_only()) {
                if (fn->vectorization_hint == hir::VectorizationHint::UnsupportedSpecialOps) {
                    jit_failure(ErrorSubKind::UnsupportedFeature,
                                "Failed to compile vectorized: unsupported vector special operations: %s",
                                hir::show_special_ops(fn->unsupported_vector_special_ops()).c_str());
                }
                jit_failure(ErrorSubKind::UnsupportedFeature,
                            "Failed to compile vectorized: function is marked as only scalar");
            }
            return compile(lower_vectorized(fn));
        case CompilePolicy::Scalar: return compile(lower_scalar(fn));
        }
        SIMJIT_UNREACHABLE();
    }

    void *find_and_typecheck_function(std::string_view identifier, const CallerInfo *caller) {
        if (const FunctionRecord *record = funcs_.find(identifier)) {
            typecheck_function(identifier, record->args.size(), record->args.data(), record->usages.data(), caller);
            return record->fn_ptr;
        }
        return nullptr;
    }

    void save_function(const hir::Function *fn, std::string_view identifier, void *fn_ptr) {
        FunctionRecord record{fn_ptr, {}, {}, {}};
        record.args.insert(record.args.end(), fn->args.begin(), fn->args.end());
        record.usages.assign(fn->argument_usage.begin(), fn->argument_usage.end());
        funcs_.insert_or_assign(identifier, std::move(record));
    }

    void *compile_internal(std::string_view identifier, const hir::Function *hir, const CallerInfo *caller) {
        typecheck_function(identifier, hir->args.size(), hir->args.data(), hir->argument_usage.data(), caller);
        if (funcs_.size() >= ctx_.build_limits.max_cached_functions) {
            jit_failure(ErrorSubKind::CacheLimitExceeded, "JIT function cache is full (%zu >= %zu)", funcs_.size(),
                        ctx_.build_limits.max_cached_functions);
        }
        void *result = lower_and_compile_function(hir);
        save_function(hir, identifier, result);
        return result;
    }

    Statistics statistics() const {
        Statistics s;
        s.function_count = funcs_.size();
        s.last_compilation_arena_used_memory = arena_.total_bytes_used();
        s.last_compilation_arena_reserved_memory = arena_.total_bytes_allocated();
        s.cache_hits = cache_hits_;
        s.cache_misses = cache_misses_;
        s.compilation_attempts = compilation_attempts_;
        s.compilation_failures = compilation_failures_;
        s.compilation_successes = compilation_successes_;
        s.last_compilation_ns = last_compilation_ns_;

        auto inner = session_.allocator_statistics();
        s.jit_memory_allocation_count = inner.allocation_count;
        s.jit_memory_block_count = inner.block_count;
        s.jit_overhead_memory = inner.overhead_memory;
        s.jit_reserved_memory = inner.reserved_memory;
        s.jit_used_memory = inner.used_memory;

        return s;
    }

    void *find_cached_function(std::string_view identifier, const CallerInfo *caller);
    void *build_and_compile(std::string_view identifier, const function_ref<void(FunctionBuilder &)> &build_fn,
                            const CallerInfo *caller);
    void capture_debug_information(const hir::Function *fn) noexcept;
    void reset_current_compilation();
    hir::Function *build_hir(const function_ref<void(FunctionBuilder &)> &build_fn);
    std::string bug_report() const;

    bool release(std::string_view identifier) {
        if (const FunctionRecord *record = funcs_.find(identifier)) {
            session_.release_compiled_function(record->fn_ptr);
            funcs_.erase(identifier);
            return true;
        }
        return false;
    }

    std::vector<std::string> function_identifiers() const { return funcs_.identifiers(); }

private:
    MemoryArena arena_{};
    Context ctx_;
    AsmjitSession session_;
    FunctionCache funcs_{};

    bool host_supports_vectorization_ = false;
    CompilePolicy policy_ = CompilePolicy::BestEffort;
    DebugOptions debug_options_{};
    DebugSnapshot debug_snapshot_{};
    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;
    size_t compilation_attempts_ = 0;
    size_t compilation_successes_ = 0;
    size_t compilation_failures_ = 0;
    uint64_t last_compilation_ns_ = 0;

    friend class JitContext;
};

JitContext::JitContext() : impl_(new JitContextImpl(Arch::Native)) {
}

JitContext::JitContext(Arch arch) : impl_(new JitContextImpl(arch)) {
}

JitContext::JitContext(JitContext &&other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

JitContext::~JitContext() noexcept {
    delete impl_;
}

void JitContext::set_policy(CompilePolicy compile_policy) {
    if (compile_policy == CompilePolicy::Vectorized) { impl_->require_host_vectorization(); }
    impl_->policy_ = compile_policy;
}

CompilePolicy JitContext::policy() const noexcept {
    return impl_->policy_;
}

void JitContext::set_transformations(CodeTransformations transform_flags) noexcept {
    impl_->ctx_.transformations = transform_flags;
}

CodeTransformations JitContext::transformations() const noexcept {
    return impl_->ctx_.transformations;
}

void JitContext::set_build_limits(const BuildLimits &limits) noexcept {
    impl_->ctx_.build_limits = limits;
}

const BuildLimits &JitContext::build_limits() const noexcept {
    return impl_->ctx_.build_limits;
}

DebugOptions &JitContext::debug_options() noexcept {
    return impl_->debug_options_;
}

const DebugSnapshot &JitContext::debug_snapshot() const noexcept {
    return impl_->debug_snapshot_;
}

JitContext &JitContext::operator=(JitContext &&other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

static void capture_asmjit(const mir::Function *func, bool emit_machine_code, bool emit_asm,
                           AsmjitCompileResult &result) {
    AsmjitCompileOptions opts{emit_machine_code, emit_asm, nullptr};
    compile_asmjit(func, opts, result);
}

void JitContextImpl::capture_debug_information(const hir::Function *fn) noexcept {
    // HIR is captured independently. If it failed, nothing to do.
    if (fn == nullptr) { return; }

    if (bool(debug_options_.stages & DebugStage::HIR)) {
        try {
            debug_snapshot_.hir = hir::print_function(fn);
        } catch (...) {}
    }
#if SIMJIT_ENABLE_SERIALIZATION
    if (bool(debug_options_.stages & DebugStage::Serialized)) {
        try {
            debug_snapshot_.serialized = serialize(fn);
        } catch (...) {}
    }
#endif
    if (bool(debug_options_.stages & DebugStage::Vectorizer) && policy_ != CompilePolicy::Scalar) {
        try {
            const auto *vec_result = vect::hir_to_vect(fn);
            debug_snapshot_.vectorizer = vect::print_function(vec_result);
        } catch (...) {}
    }
    if (bool(debug_options_.stages & DebugStage::MIR)) {
        try {
            const auto *mir = lower_function(fn);
            debug_snapshot_.mir = mir::print_function(mir);
        } catch (...) {}
    }
    if (bool(debug_options_.stages & DebugStage::ASM) || bool(debug_options_.stages & DebugStage::MachineCode)) {
        try {
            const auto *mir = lower_function(fn);
            AsmjitCompileResult result{};
            capture_asmjit(mir, bool(debug_options_.stages & DebugStage::MachineCode),
                           bool(debug_options_.stages & DebugStage::ASM), result);
            debug_snapshot_.asm_code = std::move(result.asm_code);
            debug_snapshot_.machine_code = std::move(result.machine_code);
        } catch (...) {}
    }
}

void JitContextImpl::reset_current_compilation() {
    arena_.clear();
    debug_snapshot_ = {};
}

hir::Function *JitContextImpl::build_hir(const function_ref<void(FunctionBuilder &)> &build_fn) {
    bool capture_hir = debug_options_.enabled() && bool(debug_options_.stages & DebugStage::HIR);
#if SIMJIT_ENABLE_SERIALIZATION
    bool capture_serialized = debug_options_.enabled() && bool(debug_options_.stages & DebugStage::Serialized);
#endif

    FunctionBuilder builder{ctx_};
    build_fn(builder);
    hir::Function *fn = builder.build();

    // Eager capture for HIR info. This makes sure we have enough debug information available to easily reproduce
    // the error. Obviously, this has non-zero overhead. However, we don't expect debug information to be enabled
    // always.
    if (capture_hir) { debug_snapshot_.hir = hir::print_function(fn); }
#if SIMJIT_ENABLE_SERIALIZATION
    if (capture_serialized) { debug_snapshot_.serialized = serialize(fn); }
#endif
    return fn;
}

void *JitContextImpl::find_cached_function(std::string_view identifier, const CallerInfo *caller) {
    if (void *result = find_and_typecheck_function(identifier, caller)) {
        ++cache_hits_;
        return result;
    }
    ++cache_misses_;
    return nullptr;
}

void *JitContextImpl::build_and_compile(std::string_view identifier,
                                        const function_ref<void(FunctionBuilder &)> &build_fn,
                                        const CallerInfo *caller) {
    if (auto result = find_and_typecheck_function(identifier, caller)) {
        ++cache_hits_;
        return result;
    }

    ++cache_misses_;
    ++compilation_attempts_;
    const auto compilation_start = std::chrono::steady_clock::now();

    void *result = nullptr;
    hir::Function *hir = nullptr;
    try {
        reset_current_compilation();
        hir = build_hir(build_fn);
        result = compile_internal(identifier, hir, caller);
    } catch (...) {
        last_compilation_ns_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - compilation_start)
                .count());
        ++compilation_failures_;
        if (debug_options_.capture_on_error) { capture_debug_information(hir); }
        throw;
    }
    last_compilation_ns_ = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - compilation_start)
            .count());
    if (debug_options_.capture_on_success) { capture_debug_information(hir); }
    ++compilation_successes_;
    return result;
}

static void append_section(std::string &out, const char *name, std::string_view content) {
    out += "=== ";
    out += name;
    out += " ===\n";
    if (content.empty()) {
        out += "<empty>\n";
    } else {
        out.append(content.data(), content.size());
        if (content.back() != '\n') { out += '\n'; }
    }
}

static std::string hex_bytes(const std::vector<uint8_t> &bytes) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 3);
    for (size_t i = 0; i < bytes.size(); ++i) {
        if (i != 0) { out += ' '; }
        uint32_t b = bytes[i];
        out += kHex[b >> 4u];
        out += kHex[b & 0xfu];
    }
    return out;
}

std::string JitContextImpl::bug_report() const {
    std::string out;
    out.reserve(4096);
    const Statistics s = statistics();
    const auto identifiers = function_identifiers();

    out += "=== context ===\n";
    simjit::format_to(out, "policy: %s\n", policy_name(policy_));
    simjit::format_to(out, "transformations: 0x%llx\n", (unsigned long long)ctx_.transformations);
    simjit::format_to(out, "debug_stages: 0x%llx\n", (unsigned long long)debug_options_.stages);
    simjit::format_to(out, "capture_on_success: %s\n", debug_options_.capture_on_success ? "true" : "false");
    simjit::format_to(out, "capture_on_error: %s\n", debug_options_.capture_on_error ? "true" : "false");
    simjit::format_to(out, "record_vectorization_fail_exception: %s\n",
                      debug_options_.record_vectorization_fail_exception ? "true" : "false");
    simjit::format_to(out, "function_count: %zu\n", s.function_count);
    simjit::format_to(out, "cache_hits: %zu\n", s.cache_hits);
    simjit::format_to(out, "cache_misses: %zu\n", s.cache_misses);
    simjit::format_to(out, "compilation_attempts: %zu\n", s.compilation_attempts);
    simjit::format_to(out, "compilation_successes: %zu\n", s.compilation_successes);
    simjit::format_to(out, "compilation_failures: %zu\n", s.compilation_failures);
    simjit::format_to(out, "last_compilation_ns: %llu\n", (unsigned long long)s.last_compilation_ns);
    simjit::format_to(out, "last_compilation_arena_used_memory: %zu\n", s.last_compilation_arena_used_memory);
    simjit::format_to(out, "last_compilation_arena_reserved_memory: %zu\n", s.last_compilation_arena_reserved_memory);
    simjit::format_to(out, "jit_memory_block_count: %zu\n", s.jit_memory_block_count);
    simjit::format_to(out, "jit_memory_allocation_count: %zu\n", s.jit_memory_allocation_count);
    simjit::format_to(out, "jit_used_memory: %zu\n", s.jit_used_memory);
    simjit::format_to(out, "jit_reserved_memory: %zu\n", s.jit_reserved_memory);
    simjit::format_to(out, "jit_overhead_memory: %zu\n", s.jit_overhead_memory);

    out += "function_identifiers:";
    if (identifiers.empty()) {
        out += " <empty>\n";
    } else {
        out += '\n';
        for (const std::string &identifier : identifiers) {
            out += "- ";
            out += identifier;
            out += '\n';
        }
    }

    append_section(out, "vectorization exception", debug_snapshot_.vectorization_exception);
    append_section(out, "HIR", debug_snapshot_.hir);
#if SIMJIT_ENABLE_SERIALIZATION
    append_section(out, "serialized HIR", debug_snapshot_.serialized);
#endif
    append_section(out, "vectorizer", debug_snapshot_.vectorizer);
    append_section(out, "MIR", debug_snapshot_.mir);
    append_section(out, "ASM", debug_snapshot_.asm_code);
    out += "=== machine code ===\n";
    out += simjit::format("size: %zu\n", debug_snapshot_.machine_code.size());
    if (!debug_snapshot_.machine_code.empty()) { out += hex_bytes(debug_snapshot_.machine_code) + '\n'; }
    return out;
}

void *JitContext::find_cached_function(std::string_view identifier, const CallerInfo *caller) {
    return impl_->find_cached_function(identifier, caller);
}

void *JitContext::build_and_compile(std::string_view identifier, const function_ref<void(FunctionBuilder &)> &build_fn,
                                    const CallerInfo *caller) {
    return impl_->build_and_compile(identifier, build_fn, caller);
}

Statistics JitContext::statistics() const noexcept {
    return impl_->statistics();
}

std::vector<std::string> JitContext::function_identifiers() const {
    return impl_->function_identifiers();
}

std::string JitContext::bug_report() const {
    return impl_->bug_report();
}

bool JitContext::delete_cached_function(std::string_view identifier) {
    return impl_->release(identifier);
}

void JitContext::clear() {
    CompilePolicy saved_policy = impl_->policy_;
    DebugOptions saved_debug_options = impl_->debug_options_;
    CodeTransformations saved_transformations = impl_->ctx_.transformations;
    BuildLimits saved_build_limits = impl_->ctx_.build_limits;
    Arch saved_arch = impl_->ctx_.arch;

    delete impl_;
    impl_ = new JitContextImpl(saved_arch);
    impl_->policy_ = saved_policy;
    impl_->debug_options_ = saved_debug_options;
    impl_->ctx_.transformations = saved_transformations;
    impl_->ctx_.build_limits = saved_build_limits;
}

template <size_t... Idxs> using RawFunctionPtr = void (*)(size_t, decltype((void)Idxs, (void *)nullptr)...);

template <size_t... Idxs>
static void call_fn_ptr_impl(void *fn, size_t n, nonstd::span<void *> args, std::index_sequence<Idxs...>) {
    (void)args;
    using FnPtr = RawFunctionPtr<Idxs...>;
    ((FnPtr)(fn))(n, args[Idxs]...);
}

template <size_t N> static void call_fn_ptr_n(void *fn, size_t n, nonstd::span<void *> args) {
    SIMJIT_ASSERT(args.size() == N);
    call_fn_ptr_impl(fn, n, args, std::make_index_sequence<N>{});
}

using CallFnPtrThunk = void (*)(void *, size_t, nonstd::span<void *>);

template <size_t... Idxs> static constexpr auto make_call_fn_ptr_table(std::index_sequence<Idxs...>) {
    return std::array<CallFnPtrThunk, sizeof...(Idxs)>{&call_fn_ptr_n<Idxs>...};
}

void call_fn_ptr(void *fn, size_t n, nonstd::span<void *> args) {
    static constexpr auto dispatch_table =
        make_call_fn_ptr_table(std::make_index_sequence<MaxFunctionArgumentCount + 1>{});
    if (args.size() >= dispatch_table.size() || args.empty()) {
        jit_failure(ErrorSubKind::LimitExceeded, "Don't support argument count %zu", args.size());
    }
    dispatch_table[args.size()](fn, n, args);
}

} // namespace jit
} // namespace simjit
