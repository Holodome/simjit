// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/compiler.h"
#include "simjit/detail/expected.h"

#include <optional>

namespace simjit {
namespace jit {

// How JIT compiler will approach compilation
enum class CompilePolicy : uint8_t {
    // Try vectorization, if it fails - ignore errors, try scalar
    BestEffort,
    // Only vectorized
    Vectorized,
    // Only scalar
    Scalar
};

enum class DebugStage : uint8_t {
    HIR = 1,
    Vectorizer = 2,
    MIR = 4,
    ASM = 8,
    MachineCode = 16,

    All = HIR | Vectorizer | MIR | ASM | MachineCode,
};
SIMJIT_DEFINE_ENUM_FLAGS(DebugStage)

struct DebugOptions {
    bool capture_on_error = false;
    bool capture_on_success = false;
    bool record_vectorization_fail_exception = false;
    DebugStage stages = DebugStage::HIR;

    constexpr bool enabled() const noexcept { return capture_on_error || capture_on_success; }
};

struct DebugSnapshot {
    std::string vectorization_exception{};
    std::string hir{};
    std::string serialized{};
    std::string vectorizer{};
    std::string mir{};
    std::string asm_code{};
    std::vector<uint8_t> machine_code{};
};

struct CallerInfo {
    const ScalarDataType *expected_types = nullptr;
    const ArgumentKind *expected_kinds = nullptr;
    size_t expected_arg_count = SIZE_MAX;
};

// Statistics for current state of JitContext.
struct Statistics {
    // How many distinct functions/identifiers are there?
    size_t function_count;
    size_t cache_hits;
    size_t cache_misses;
    size_t compilation_attempts;
    size_t compilation_successes;
    size_t compilation_failures;

    // Compilation memory arena reserved memory size.
    size_t last_compilation_arena_used_memory;
    // Compilation memory arena reserved memory size.
    size_t last_compilation_arena_reserved_memory;
    // Information from asmjit::JitAllocator::Statistics.
    // https://asmjit.com/doc/structasmjit_1_1JitAllocator_1_1Statistics.html
    size_t jit_memory_block_count;
    size_t jit_memory_allocation_count;
    size_t jit_used_memory;
    size_t jit_reserved_memory;
    size_t jit_overhead_memory;
};

// Use PIMPL to avoid bringing private state information. This also keeps header cleaner.
class JitContextImpl;

// JitContext owns the compiled-function cache and executable memory for its lifetime.
//
// Usage is as following: keep one context for a workload, compile functions into it by stable identifiers, reuse
// returned functions through cache hits, and release individual identifiers or clear the context when their generated
// code is no longer needed.
//
// Function pointers returned from this context are lightweight execution handles into that cache. They remain valid
// while the owning JitContext keeps the matching identifier alive. Use the matching argument types, keep the context
// alive, and do not call a function after its identifier has been released or the context has been cleared.
class JitContext {
    friend class JitContextImpl;

public:
    JitContext();
    explicit JitContext(Arch arch);
    ~JitContext() noexcept;

    JitContext(const JitContext &) = delete;
    JitContext(JitContext &&other) noexcept;
    JitContext &operator=(const JitContext &) = delete;
    JitContext &operator=(JitContext &&other) noexcept;

    void set_policy(CompilePolicy policy) noexcept { policy_ = policy; }
    CompilePolicy policy() const noexcept { return policy_; }
    void set_transformations(CodeTransformations transformations) noexcept { ctx_.transformations = transformations; }
    CodeTransformations transformations() const noexcept { return ctx_.transformations; }
    void set_build_limits(const BuildLimits &limits) noexcept { ctx_.build_limits = limits; }
    const BuildLimits &build_limits() const noexcept { return ctx_.build_limits; }

    DebugOptions &debug_options() noexcept { return debug_options_; }
    // DebugSnapshot always contains the result of the most recent compilation attempt on this JitContext.
    const DebugSnapshot &debug_snapshot() const noexcept { return debug_snapshot_; }

    Statistics statistics() const noexcept;
    std::vector<std::string> function_identifiers() const;
    std::string bug_report() const;

    // Look up a compiled function by identifier without invoking a builder.
    // Returns nullptr on cache miss. If caller is provided and the identifier exists, declared argument
    // count/kinds/types are checked before returning the cached function pointer.
    void *find_cached_function(std::string_view identifier, const CallerInfo *caller = nullptr);

    // Same cache operation as release(), named for callers that think in cache terms.
    bool delete_cached_function(std::string_view identifier);
    void clear();

    // identifier is the cache key. On cache hit, the stored function is reused without rebuilding the expression or
    // comparing it with build_fn. If caller is provided, only the declared argument count/kinds/types are checked.
    // Use stable, unique identifiers for distinct expressions.
    template <typename BuildFn>
    void *build_and_compile(std::string_view identifier, BuildFn build_fn, const CallerInfo *caller = nullptr) {
        if (auto result = find_and_typecheck_function(identifier, caller)) {
            ++cache_hits_;
            return result;
        }

        ++cache_misses_;
        ++compilation_attempts_;

        void *result = nullptr;
        hir::Function *hir = nullptr;
        try {
            reset_current_compilation();
            hir = build_hir(build_fn);
            result = compile(identifier, hir, caller);
        } catch (...) {
            ++compilation_failures_;
            if (debug_options_.capture_on_error) { capture_debug_information(hir); }
            throw;
        }
        if (debug_options_.capture_on_success) { capture_debug_information(hir); }
        ++compilation_successes_;
        return result;
    }

private:
    // identifier is used as unique key that can be used to save and look up compiled functions. This function looks up
    // already compiled function. If not found, it returns nullptr. Additionally does type checking.
    void *find_and_typecheck_function(std::string_view identifier, const CallerInfo *caller);

    // JitContext stores data that is local to last compilation, for example memory arena and debug state. This is
    // called in beginning of each new compilation.
    void reset_current_compilation();

    template <typename BuildFn> hir::Function *build_hir(BuildFn build_fn) {
        bool debug_info = debug_options_.enabled() && bool(debug_options_.stages & DebugStage::HIR);

        FunctionBuilder builder{ctx_};
        build_fn(builder);
        hir::Function *fn = builder.build();

        // Eager capture for HIR info. This makes sure we have enough debug information available to easily reproduce
        // the error. Obviously, this has non-zero overhead. However, we don't expect debug information to be enabled
        // always.
        if (debug_info) {
            debug_snapshot_.hir = hir::print_function(fn);
#if SIMJIT_ENABLE_SERIALIZATION
            debug_snapshot_.serialized = serialize(fn);
#endif
        }
        return fn;
    }

    void *compile(std::string_view identifier, const hir::Function *hir, const CallerInfo *caller);

    // Fills debug_snapshot_ according to debug_options_. This function is guaranteed not to throw,
    // since it can be called in catch {} block.
    void capture_debug_information(const hir::Function *fn) noexcept;

    JitContextImpl *impl_ = nullptr;

    CompilePolicy policy_ = CompilePolicy::BestEffort;
    DebugOptions debug_options_{};
    DebugSnapshot debug_snapshot_{};

    size_t cache_hits_ = 0;
    size_t cache_misses_ = 0;
    size_t compilation_attempts_ = 0;
    size_t compilation_successes_ = 0;
    size_t compilation_failures_ = 0;

    MemoryArena arena_{};
    Context ctx_;
};

// Create special type to indicate bit array. This is made so that library user has to cast his array to (Bitmask *)
// when using it with input/output array. Because size is computed differently (8 bits per byte), this has to be done
// knowingly.
class Bitmask;

template <ScalarDataType s> struct ArrayTypeMap {
    using type = typename ScalarDataTypeMap<s>::type;
};
template <> struct ArrayTypeMap<ScalarDataType::I1> {
    using type = Bitmask;
};

template <ArgumentKind k, typename T, ScalarDataType s> struct JitTypeWrapper {
    constexpr static ArgumentKind kind = k;
    constexpr static ScalarDataType scalar = s;
    using type = T;
};

template <ScalarDataType s>
using InputArr = JitTypeWrapper<ArgumentKind::SrcArr, const typename ArrayTypeMap<s>::type *, s>;
template <ScalarDataType s>
using InputConst = JitTypeWrapper<ArgumentKind::SrcConst, const typename ScalarDataTypeMap<s>::type *, s>;
template <ScalarDataType s> using OutputArr = JitTypeWrapper<ArgumentKind::Dst, typename ArrayTypeMap<s>::type *, s>;
template <ScalarDataType s>
using OutputScalar = JitTypeWrapper<ArgumentKind::DstAgg, typename ScalarDataTypeMap<s>::type *, s>;
template <ScalarDataType s>
using OutputSafetyCheck = JitTypeWrapper<ArgumentKind::DstSafetyCheck, typename ScalarDataTypeMap<s>::type *, s>;

void call_fn_ptr(void *fn, size_t n, nonstd::span<void *> args);

template <typename... Types> struct FunctionHolder {
    using FnPtr = void (*)(size_t, Types...);

    FunctionHolder() = delete;
    explicit FunctionHolder(FnPtr x) : fn(x) {}

    // Thin typed wrapper over a JIT function pointer owned by JitContext.
    void operator()(size_t n, Types... args) {
        SIMJIT_ASSERT(fn != nullptr);
        fn(n, args...);
    }

    FnPtr fn = nullptr;
};

template <typename... Args, typename BuildFn>
inline FunctionHolder<typename Args::type...> vectorized_function(JitContext &ctx, std::string_view identifier,
                                                                  BuildFn build_fn) {
    constexpr ScalarDataType mapped_types[] = {Args::scalar...};
    constexpr ArgumentKind kinds[] = {Args::kind...};
    CallerInfo caller{mapped_types, kinds, sizeof...(Args)};
    void *fn = ctx.build_and_compile(identifier, build_fn, &caller);
    using Holder = FunctionHolder<typename Args::type...>;
    return Holder{(typename Holder::FnPtr)fn};
}

template <typename... Args>
inline std::optional<FunctionHolder<typename Args::type...>> find_vectorized_function(JitContext &ctx,
                                                                                      std::string_view identifier) {
    constexpr ScalarDataType mapped_types[] = {Args::scalar...};
    constexpr ArgumentKind kinds[] = {Args::kind...};
    CallerInfo caller{mapped_types, kinds, sizeof...(Args)};
    void *fn = ctx.find_cached_function(identifier, &caller);
    if (fn == nullptr) return {};
    using Holder = FunctionHolder<typename Args::type...>;
    return Holder{(typename Holder::FnPtr)fn};
}

#define SIMJIT_STRINGIFY(x) #x
#define SIMJIT_TOSTRING(x) SIMJIT_STRINGIFY(x)
#define SIMJIT_UNIQUE_FUNCTION_IDENTIFIER __FILE__ ":" SIMJIT_TOSTRING(__LINE__) ":" SIMJIT_TOSTRING(__COUNTER__)

template <typename... Args, typename BuildFn>
inline simjit::nonstd::expected<FunctionHolder<typename Args::type...>, ErrorInfo>
try_vectorized_function(JitContext &ctx, std::string_view identifier, BuildFn build_fn) {
    try {
        auto fn = vectorized_function<Args...>(ctx, identifier, build_fn);
        return {fn};
    } catch (const SimjitException &e) {
        return simjit::nonstd::unexpected<ErrorInfo>(e.info());
    } catch (const std::exception &e) { //
        return simjit::nonstd::unexpected<ErrorInfo>(
            ErrorInfo{ErrorModule::Generic, ErrorKind::InternalInvariant, ErrorSubKind::ExternalFailure, e.what()});
    }
}

} // namespace jit
} // namespace simjit
