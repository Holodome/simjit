// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/jit.h"
#include "simjit/asmjit.h"
#include "simjit/core/vectorizer.h"

#include <array>
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
    void *fn_ptr;
    std::vector<ArgumentDecl> args;
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

static ArgumentKind inner_kind(ArgumentKind kind) {
    // We have additional types that are meaningless to user, since for them these are all just read-only arrays.
    if (kind == ArgumentKind::SrcGatherArr) { return ArgumentKind::SrcArr; }
    if (kind == ArgumentKind::SrcIdxArr) { return ArgumentKind::SrcArr; }
    return kind;
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
                               const CallerInfo *caller) {
    if (caller == nullptr) { return; }

    if (decl_count != caller->expected_arg_count) {
        jit_failure(ErrorSubKind::ArgumentMismatch, "%.*s Argument count mismatch: expected %zu, got %zu",
                    (int)identifier.length(), identifier.data(), caller->expected_arg_count, decl_count);
    }
    for (size_t i = 0; i < decl_count; ++i) {
        const ArgumentDecl &arg = decls[i];
        auto actual_kind = inner_kind(arg.kind);
        if (caller->expected_kinds[i] != actual_kind) {
            jit_failure(ErrorSubKind::ArgumentMismatch, "[%.*s] Argument [%zu] kind mismatch: expected %s, got %s",
                        (int)identifier.length(), identifier.data(), i,
                        show_argument_kind(caller->expected_kinds[i]).c_str(), show_argument_kind(actual_kind).c_str());
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
    explicit JitContextImpl(JitContext *parent) : session_(parent->ctx_.arch), parent_(parent) {
        Arch host_arch = session_.host_arch();
        if (!can_jit_target_on_host(parent_->ctx_.arch, host_arch)) {
            jit_failure(ErrorSubKind::UnsupportedHostFeature,
                        "JIT target %s cannot execute on host %s; use inspection emitters for "
                        "cross-target code generation",
                        show_arch(parent_->ctx_.arch), show_arch(host_arch));
        }
        if (is_x86_arch(host_arch)) {
            if (!session_.host_supports_x86_backend()) {
                jit_failure(ErrorSubKind::UnsupportedHostFeature,
                            "Amd64 machine should support BMI2 instruction set (Haswell and newer)");
            }
            host_supports_vectorization_ = session_.host_supports_vectorization();
        } else if (host_arch == Arch::Arm64_NEON) {
            host_supports_vectorization_ = true;
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

    mir::Function *lower_function(const hir::Function *fn) {
        switch (parent_->policy_) {
        case CompilePolicy::BestEffort:
            if (host_supports_vectorization_) {
                auto vectorized = try_lower_vectorized(fn);
                if (vectorized) { return vectorized.value(); }
                if (!can_best_effort_fallback(vectorized.error())) {
                    throw SimjitException(std::move(vectorized.error()));
                }
            }
            return lower_scalar(fn);
        case CompilePolicy::Vectorized:
            if (!host_supports_vectorization_) {
                jit_failure(ErrorSubKind::UnsupportedHostFeature, "Host machine does not support vectorization");
            }
            return lower_vectorized(fn);
        case CompilePolicy::Scalar: return lower_scalar(fn);
        }
        SIMJIT_UNREACHABLE();
    }

    void *lower_and_compile_function(const hir::Function *fn) {
        switch (parent_->policy_) {
        case CompilePolicy::BestEffort:
            if (!fn->scalar_only) {
                auto vectorized = try_lower_vectorized(fn);
                if (vectorized) {
                    try {
                        return compile(vectorized.value());
                    } catch (const SimjitException &e) {
                        if (!can_best_effort_fallback(e.info())) { throw; }
                        if (parent_->debug_options_.record_vectorization_fail_exception) {
                            parent_->debug_snapshot_.vectorization_exception = e.info().verbose();
                        }
                    }
                } else if (can_best_effort_fallback(vectorized.error())) {
                    if (parent_->debug_options_.record_vectorization_fail_exception) {
                        parent_->debug_snapshot_.vectorization_exception = vectorized.error().verbose();
                    }
                } else {
                    throw SimjitException(std::move(vectorized.error()));
                }
            }
            return compile(lower_scalar(fn));
        case CompilePolicy::Vectorized:
            if (fn->scalar_only) {
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
            typecheck_function(identifier, record->args.size(), record->args.data(), caller);
            return record->fn_ptr;
        }
        return nullptr;
    }

    void save_function(const hir::Function *fn, std::string_view identifier, void *fn_ptr) {
        FunctionRecord record{fn_ptr, {}, {}};
        record.args.insert(record.args.end(), fn->args.begin(), fn->args.end());
        funcs_.insert_or_assign(identifier, std::move(record));
    }

    void *compile_internal(std::string_view identifier, const hir::Function *hir, const CallerInfo *caller) {
        typecheck_function(identifier, hir->args.size(), hir->args.data(), caller);
        if (funcs_.size() >= parent_->ctx_.build_limits.max_cached_functions) {
            jit_failure(ErrorSubKind::CacheLimitExceeded, "JIT function cache is full (%zu >= %zu)", funcs_.size(),
                        parent_->ctx_.build_limits.max_cached_functions);
        }
        void *result = lower_and_compile_function(hir);
        save_function(hir, identifier, result);
        return result;
    }

    Statistics statistics() const {
        Statistics s;
        s.function_count = funcs_.size();
        s.last_compilation_arena_used_memory = parent_->arena_.total_bytes_used();
        s.last_compilation_arena_reserved_memory = parent_->arena_.total_bytes_allocated();
        s.cache_hits = parent_->cache_hits_;
        s.cache_misses = parent_->cache_misses_;
        s.compilation_attempts = parent_->compilation_attempts_;
        s.compilation_failures = parent_->compilation_failures_;
        s.compilation_successes = parent_->compilation_successes_;

        auto inner = session_.allocator_statistics();
        s.jit_memory_allocation_count = inner.allocation_count;
        s.jit_memory_block_count = inner.block_count;
        s.jit_overhead_memory = inner.overhead_memory;
        s.jit_reserved_memory = inner.reserved_memory;
        s.jit_used_memory = inner.used_memory;

        return s;
    }

    bool release(std::string_view identifier) {
        if (const FunctionRecord *record = funcs_.find(identifier)) {
            session_.release_compiled_function(record->fn_ptr);
            funcs_.erase(identifier);
            return true;
        }
        return false;
    }

    std::vector<std::string> function_identifiers() const { return funcs_.identifiers(); }

    void rebind_parent(JitContext *parent) { parent_ = parent; }

private:
    AsmjitSession session_;
    JitContext *parent_ = nullptr;
    FunctionCache funcs_{};

    bool host_supports_vectorization_ = false;
};

JitContext::JitContext() : ctx_(arena_) {
    impl_ = new JitContextImpl(this);
}

JitContext::JitContext(Arch arch) : ctx_(arena_, "expr", CodeTransformations::All, arch) {
    impl_ = new JitContextImpl(this);
}

JitContext::JitContext(JitContext &&other) noexcept
    : impl_(other.impl_), policy_(other.policy_), debug_options_(other.debug_options_),
      debug_snapshot_(std::move(other.debug_snapshot_)), cache_hits_(other.cache_hits_),
      cache_misses_(other.cache_misses_), compilation_attempts_(other.compilation_attempts_),
      compilation_successes_(other.compilation_successes_), compilation_failures_(other.compilation_failures_),
      arena_(std::move(other.arena_)), ctx_(std::move(other.ctx_)) {
    ctx_.arena = &arena_;
    if (impl_ != nullptr) { impl_->rebind_parent(this); }

    other.impl_ = nullptr;
    other.ctx_.arena = &other.arena_;
}

JitContext::~JitContext() noexcept {
    delete impl_;
}

JitContext &JitContext::operator=(JitContext &&other) noexcept {
    if (this != &other) {
        delete impl_;

        impl_ = other.impl_;
        policy_ = other.policy_;
        debug_options_ = other.debug_options_;
        debug_snapshot_ = std::move(other.debug_snapshot_);
        cache_hits_ = other.cache_hits_;
        cache_misses_ = other.cache_misses_;
        compilation_attempts_ = other.compilation_attempts_;
        compilation_successes_ = other.compilation_successes_;
        compilation_failures_ = other.compilation_failures_;
        arena_ = std::move(other.arena_);
        ctx_ = std::move(other.ctx_);
        ctx_.arena = &arena_;
        if (impl_ != nullptr) { impl_->rebind_parent(this); }

        other.impl_ = nullptr;
        other.ctx_.arena = &other.arena_;
    }
    return *this;
}

static void capture_asmjit(const mir::Function *func, bool emit_machine_code, bool emit_asm,
                           AsmjitCompileResult &result) {
    AsmjitCompileOptions opts{emit_machine_code, emit_asm, nullptr};
    compile_asmjit(func, opts, result);
}

void JitContext::capture_debug_information(const hir::Function *fn) noexcept {
    // HIR is captured independently. If it failed, nothing to do.
    if (fn == nullptr) { return; }

    if (bool(debug_options_.stages & DebugStage::HIR)) {
        try {
            debug_snapshot_.hir = hir::print_function(fn);
#if SIMJIT_ENABLE_SERIALIZATION
            debug_snapshot_.serialized = serialize(fn);
#endif
        } catch (...) {}
    }
    if (bool(debug_options_.stages & DebugStage::Vectorizer) && policy_ != CompilePolicy::Scalar) {
        try {
            const auto *vec_result = vect::hir_to_vect(fn);
            debug_snapshot_.vectorizer = vect::print_function(vec_result);
        } catch (...) {}
    }
    if (bool(debug_options_.stages & DebugStage::MIR)) {
        try {
            const auto *mir = impl_->lower_function(fn);
            debug_snapshot_.mir = mir::print_function(mir);
        } catch (...) {}
    }
    if (bool(debug_options_.stages & DebugStage::ASM) || bool(debug_options_.stages & DebugStage::MachineCode)) {
        try {
            const auto *mir = impl_->lower_function(fn);
            AsmjitCompileResult result{};
            capture_asmjit(mir, bool(debug_options_.stages & DebugStage::MachineCode),
                           bool(debug_options_.stages & DebugStage::ASM), result);
            debug_snapshot_.asm_code = std::move(result.asm_code);
            debug_snapshot_.machine_code = std::move(result.machine_code);
        } catch (...) {}
    }
}

void JitContext::reset_current_compilation() {
    arena_.clear();
    debug_snapshot_ = {};
}

void *JitContext::find_and_typecheck_function(std::string_view identifier, const CallerInfo *caller) {
    return impl_->find_and_typecheck_function(identifier, caller);
}

void *JitContext::find_cached_function(std::string_view identifier, const CallerInfo *caller) {
    if (void *result = find_and_typecheck_function(identifier, caller)) {
        ++cache_hits_;
        return result;
    }
    ++cache_misses_;
    return nullptr;
}

void *JitContext::compile(std::string_view identifier, const hir::Function *hir, const CallerInfo *caller) {
    return impl_->compile_internal(identifier, hir, caller);
}

Statistics JitContext::statistics() const noexcept {
    return impl_->statistics();
}

std::vector<std::string> JitContext::function_identifiers() const {
    return impl_->function_identifiers();
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
        uint8_t b = bytes[i];
        out += kHex[b >> 4];
        out += kHex[b & 0xf];
    }
    return out;
}

std::string JitContext::bug_report() const {
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

bool JitContext::delete_cached_function(std::string_view identifier) {
    return impl_->release(identifier);
}

void JitContext::clear() {
    CompilePolicy p = policy_;
    DebugOptions d = debug_options_;
    CodeTransformations t = ctx_.transformations;
    BuildLimits l = ctx_.build_limits;
    Arch a = ctx_.arch;
    std::string n = ctx_.symbol_name;

    delete impl_;
    impl_ = nullptr;

    arena_.clear();
    ctx_ = Context(arena_, n, t, a);
    ctx_.build_limits = l;
    debug_snapshot_ = {};
    cache_hits_ = 0;
    cache_misses_ = 0;
    compilation_attempts_ = 0;
    compilation_successes_ = 0;
    compilation_failures_ = 0;

    policy_ = p;
    debug_options_ = d;
    impl_ = new JitContextImpl(this);
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
