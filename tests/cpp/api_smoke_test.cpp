// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/asmjit.h"
#include "simjit/compiler.h"
#include "simjit/core/hir.h"
#include "simjit/jit.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace simjit;
using namespace simjit::jit;
using namespace simjit::types;

#if SIMJIT_ASMJIT_BACKEND
static void require_api_smoke(bool condition, const char *message) {
    if (!condition) { throw std::runtime_error(std::string("api smoke failed: ") + message); }
}

static JitContext &&api_smoke_move_ref(JitContext &ctx) {
    return static_cast<JitContext &&>(ctx);
}

static void require_api_smoke_contains(const std::string &text, const char *needle, const char *message) {
    require_api_smoke(text.find(needle) != std::string::npos, message);
}

static void require_api_smoke_identifier(const std::vector<std::string> &identifiers, const char *needle,
                                         const char *message) {
    for (const std::string &identifier : identifiers) {
        if (identifier == needle) { return; }
    }
    throw std::runtime_error(std::string("api smoke failed: ") + message);
}

static void require_api_smoke_error_info(const ErrorInfo &info, ErrorModule module, ErrorKind kind,
                                         ErrorSubKind subkind, const char *message) {
    require_api_smoke(info.module == module, message);
    require_api_smoke(info.kind == kind, message);
    require_api_smoke(info.subkind == subkind, message);
}

template <typename Fn>
static ErrorInfo expect_api_smoke_simjit_error(Fn fn, ErrorModule module, ErrorKind kind, ErrorSubKind subkind,
                                               const char *message) {
    try {
        fn();
    } catch (const SimjitException &e) {
        ErrorInfo info = e.info();
        require_api_smoke_error_info(info, module, kind, subkind, message);
        return info;
    } catch (const std::exception &e) {
        throw std::runtime_error(std::string("api smoke failed: ") + message +
                                 " (unexpected std::exception: " + e.what() + ")");
    } catch (...) {
        throw std::runtime_error(std::string("api smoke failed: ") + message + " (unexpected non-std exception)");
    }

    throw std::runtime_error(std::string("api smoke failed: ") + message + " (expected exception)");
}

static void api_smoke_build_sum_i32(FunctionBuilder &b) {
    Value src = b.input_arg(I32);
    Argument dst = b.arg(I32);
    b.sum(src, dst);
}

static void api_smoke_build_scaled_sum_i64(FunctionBuilder &b) {
    Value src = b.input_arg(I32);
    Argument dst = b.arg(I64);
    b.sum(b.mul(b.sext(src, I64), b.i64(123)), dst);
}

static void api_smoke_build_many_outputs(FunctionBuilder &b) {
    Value src = b.input_arg(I32);
    for (size_t i = 0; i < BuildLimits{}.max_vector_roots + 1; ++i) {
        b.output_arg(src);
    }
}

static void api_smoke_raw_build_callbacks() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    int32_t offset = 7;
    void *temporary_lambda = ctx.build_and_compile("api-smoke-temporary-builder", [offset](FunctionBuilder &b) {
        Value src = b.input_arg(I32);
        b.sum(b.add(src, b.i32(offset)), b.arg(I32));
    });
    require_api_smoke(temporary_lambda != nullptr, "temporary builder lambda");

    auto mutable_builder = [offset](FunctionBuilder &b) mutable {
        Value src = b.input_arg(I32);
        b.sum(b.add(src, b.i32(offset)), b.arg(I32));
    };
    void *mutable_callable = ctx.build_and_compile("api-smoke-mutable-builder", mutable_builder);
    require_api_smoke(mutable_callable != nullptr, "mutable builder callable");

    const auto const_builder = [](FunctionBuilder &b) { api_smoke_build_sum_i32(b); };
    void *const_callable = ctx.build_and_compile("api-smoke-const-builder", const_builder);
    require_api_smoke(const_callable != nullptr, "const builder callable");
}

static Arch api_smoke_host_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return Arch::Arm64_NEON;
#elif defined(__x86_64__) || defined(_M_X64)
    return Arch::Amd64_AVX512;
#else
#error "Unsupported public API smoke host architecture"
#endif
}

static bool api_smoke_host_supports_vectorization() {
    AsmjitSession session(Arch::Native);
    return session.host_supports_vectorization();
}

template <typename BuildFn>
static void require_api_smoke_cascade_scalar_special_op(hir::SpecialOp expected, BuildFn build_fn,
                                                        const char *message) {
    MemoryArena arena{};
    Context context{arena, "cascade-lake", CodeTransformations::All, Arch::Amd64_AVX512};
    hir::SpecialOp ice_lake = hir::supported_vector_special_ops_for_arch(Arch::Amd64_AVX512);
    context.host_supported_vector_special_ops =
        ice_lake & ~(hir::SpecialOp::SmallPack | hir::SpecialOp::ArbitraryBitPermute | hir::SpecialOp::I8ConstantShift |
                     hir::SpecialOp::SmallPopcount | hir::SpecialOp::LargePopcount);
    FunctionBuilder builder{context};
    build_fn(builder);
    hir::Function *fn = builder.build();
    require_api_smoke(fn->is_scalar_only(), message);
    require_api_smoke(bool(fn->unsupported_vector_special_ops() & expected), message);
}

static void api_smoke_special_op_capabilities() {
    using hir::SpecialOp;

    SpecialOp ice_lake = hir::supported_vector_special_ops_for_arch(Arch::Amd64_AVX512);
    require_api_smoke(bool(ice_lake & SpecialOp::SmallPack), "Ice Lake small pack capability");
    require_api_smoke(bool(ice_lake & SpecialOp::ArbitraryBitPermute), "Ice Lake GFNI capability");
    require_api_smoke(bool(ice_lake & SpecialOp::SmallPopcount), "Ice Lake small popcount capability");
    require_api_smoke(bool(ice_lake & SpecialOp::LargePopcount), "Ice Lake large popcount capability");

    SpecialOp arm = hir::supported_vector_special_ops_for_arch(Arch::Arm64_NEON);
    require_api_smoke(bool(arm & SpecialOp::SmallPack), "Arm small pack capability");
    require_api_smoke(bool(arm & SpecialOp::SmallPopcount), "Arm small popcount capability");
    require_api_smoke(bool(arm & SpecialOp::LargePopcount), "Arm large popcount capability");
    require_api_smoke(!bool(arm & SpecialOp::ArbitraryBitPermute), "Arm arbitrary bit permute capability");

    MemoryArena scalar_arena{};
    Context scalar_context{scalar_arena, "cascade-lake", CodeTransformations::All, Arch::Amd64_AVX512};
    // Synthetic Cascade Lake host mask: retain the AVX-512 baseline but exclude the Ice Lake extensions.
    scalar_context.host_supported_vector_special_ops =
        ice_lake & ~(SpecialOp::SmallPack | SpecialOp::ArbitraryBitPermute | SpecialOp::I8ConstantShift |
                     SpecialOp::SmallPopcount | SpecialOp::LargePopcount);
    FunctionBuilder scalar_builder{scalar_context};
    scalar_builder.pack(scalar_builder.input_arg(I8), scalar_builder.input_predicate_arg(), scalar_builder.arg(I8),
                        scalar_builder.arg(I64));
    hir::Function *scalar_fn = scalar_builder.build();
    require_api_smoke(scalar_fn->is_scalar_only(), "Cascade small pack forces scalar");
    require_api_smoke(bool(scalar_fn->unsupported_vector_special_ops() & SpecialOp::SmallPack),
                      "Cascade small pack reason");
    require_api_smoke_contains(hir::print_function(scalar_fn), "unsupported-vector-special-ops=small-pack",
                               "Cascade special op HIR report");

    require_api_smoke_cascade_scalar_special_op(
        SpecialOp::SmallPopcount, [](FunctionBuilder &b) { b.output_arg(b.popcnt(b.input_arg(I8))); },
        "Cascade small popcount forces scalar");
    require_api_smoke_cascade_scalar_special_op(
        SpecialOp::LargePopcount, [](FunctionBuilder &b) { b.output_arg(b.popcnt(b.input_arg(I32))); },
        "Cascade large popcount forces scalar");
    require_api_smoke_cascade_scalar_special_op(
        SpecialOp::ArbitraryBitPermute,
        [](FunctionBuilder &b) { b.output_arg(b.permute_i8_bits(b.input_arg(I8), 1, 2, 3, 4, 5, 6, 7, 0)); },
        "Cascade arbitrary bit permute forces scalar");
    require_api_smoke_cascade_scalar_special_op(
        SpecialOp::I8ConstantShift, [](FunctionBuilder &b) { b.output_arg(b.sll(b.input_arg(I8), b.i8(1))); },
        "Cascade i8 constant shift forces scalar");

    MemoryArena vector_arena{};
    Context vector_context{vector_arena, "cascade-lake", CodeTransformations::All, Arch::Amd64_AVX512};
    vector_context.host_supported_vector_special_ops = scalar_context.host_supported_vector_special_ops;
    FunctionBuilder vector_builder{vector_context};
    vector_builder.output_arg(vector_builder.input_arg(I32));
    hir::Function *vector_fn = vector_builder.build();
    require_api_smoke(!vector_fn->is_scalar_only(), "Cascade baseline vector HIR remains eligible");
}

static void api_smoke_sum() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);
    require_api_smoke(ctx.statistics().last_compilation_ns == 0, "initial compilation timing");

    auto func = vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-sum", api_smoke_build_sum_i32);
    const uint64_t compilation_ns = ctx.statistics().last_compilation_ns;
    require_api_smoke(compilation_ns > 0, "successful compilation timing");

    auto cached = find_vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-sum");
    require_api_smoke(cached.has_value(), "sum cache hit");
    require_api_smoke(ctx.statistics().last_compilation_ns == compilation_ns, "cache hit preserves compilation timing");

    int32_t input[1000];
    for (int32_t i = 0; i < 1000; ++i) {
        input[i] = i;
    }

    int32_t output = 0;
    func(1000, input, &output);
    require_api_smoke(output == 499500, "sum result");
}

static void api_smoke_try_compile_and_debug() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);
    ctx.debug_options().capture_on_error = true;
    ctx.debug_options().stages = DebugStage::HIR;

    auto wrong_arg_count =
        try_vectorized_function<InputArr<I32>>(ctx, "api-smoke-wrong-count", api_smoke_build_sum_i32);
    require_api_smoke(!wrong_arg_count, "wrong prototype should fail");
    require_api_smoke(!ctx.debug_snapshot().hir.empty(), "debug HIR should be captured on error");
}

static void api_smoke_serialized_debug_stage() {
#if SIMJIT_ENABLE_SERIALIZATION
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);
    ctx.debug_options().capture_on_success = true;
    ctx.debug_options().stages = DebugStage::Serialized;

    (void)vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-serialized-debug-stage",
                                                                api_smoke_build_sum_i32);
    const DebugSnapshot &snapshot = ctx.debug_snapshot();
    require_api_smoke(snapshot.hir.empty(), "serialized debug stage does not capture HIR");
    require_api_smoke(!snapshot.serialized.empty(), "serialized debug stage captures serialized HIR");
#endif
}

static void api_smoke_try_compile_generic_error() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto result = try_vectorized_function<InputArr<I32>>(
        ctx, "api-smoke-std-exception", [](FunctionBuilder &) { throw std::runtime_error("builder exploded"); });

    require_api_smoke(!result, "std::exception should be returned as expected error");
    require_api_smoke_error_info(result.error(), ErrorModule::Generic, ErrorKind::InternalInvariant,
                                 ErrorSubKind::ExternalFailure, "std::exception error info");
    require_api_smoke_contains(result.error().message, "builder exploded", "std::exception message");

    Statistics stats = ctx.statistics();
    require_api_smoke(stats.compilation_attempts == 1, "std::exception attempt count");
    require_api_smoke(stats.compilation_failures == 1, "std::exception failure count");
    require_api_smoke(stats.compilation_successes == 0, "std::exception success count");
    require_api_smoke(stats.last_compilation_ns > 0, "std::exception compilation timing");
}

static void api_smoke_cache_and_casts() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto sum_func =
        vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-cache-sum", api_smoke_build_sum_i32);
    auto scaled_func = vectorized_function<InputArr<I32>, OutputScalar<I64>>(ctx, "api-smoke-cache-scaled",
                                                                             api_smoke_build_scaled_sum_i64);

    int32_t input[1000];
    for (int32_t i = 0; i < 1000; ++i) {
        input[i] = i;
    }

    int32_t sum = 0;
    int64_t scaled = 0;
    sum_func(1000, input, &sum);
    scaled_func(1000, input, &scaled);

    require_api_smoke(sum == 499500, "cached sum result");
    require_api_smoke(scaled == 61438500, "scaled i64 sum result");

    ctx.delete_cached_function("api-smoke-cache-scaled");
}

static void api_smoke_cache_error_paths() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto sum_func =
        vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-cache-errors", api_smoke_build_sum_i32);
    (void)sum_func;

    require_api_smoke(ctx.find_cached_function("api-smoke-cache-errors") != nullptr, "raw cache hit");
    require_api_smoke(ctx.find_cached_function("api-smoke-missing") == nullptr, "raw cache miss");

    auto cached = find_vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-cache-errors");
    require_api_smoke(cached.has_value(), "typed cache hit");
    auto missing = find_vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-missing");
    require_api_smoke(!missing.has_value(), "typed cache miss");

    ErrorInfo count_mismatch = expect_api_smoke_simjit_error(
        [&]() { (void)find_vectorized_function<InputArr<I32>>(ctx, "api-smoke-cache-errors"); }, ErrorModule::JIT,
        ErrorKind::JitFailure, ErrorSubKind::ArgumentMismatch, "cached argument count mismatch");
    require_api_smoke_contains(count_mismatch.message, "Argument count mismatch", "count mismatch message");

    ErrorInfo role_mismatch = expect_api_smoke_simjit_error(
        [&]() { (void)find_vectorized_function<InputConst<I32>, OutputScalar<I32>>(ctx, "api-smoke-cache-errors"); },
        ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::ArgumentMismatch, "cached argument role mismatch");
    require_api_smoke_contains(role_mismatch.message, "role mismatch", "role mismatch message");

    ErrorInfo type_mismatch = expect_api_smoke_simjit_error(
        [&]() { (void)find_vectorized_function<InputArr<I32>, OutputScalar<I64>>(ctx, "api-smoke-cache-errors"); },
        ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::ArgumentMismatch, "cached argument type mismatch");
    require_api_smoke_contains(type_mismatch.message, "type mismatch", "type mismatch message");

    require_api_smoke(ctx.delete_cached_function("api-smoke-cache-errors"), "delete cached function");
    require_api_smoke(!ctx.delete_cached_function("api-smoke-cache-errors"), "delete missing cached function");
}

static void api_smoke_cache_limit_error() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    BuildLimits limits = ctx.build_limits();
    limits.max_cached_functions = 0;
    ctx.set_build_limits(limits);

    ErrorInfo info = expect_api_smoke_simjit_error(
        [&]() {
            (void)vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-cache-limit",
                                                                        api_smoke_build_sum_i32);
        },
        ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::CacheLimitExceeded, "cache limit exceeded");
    require_api_smoke_contains(info.message, "JIT function cache is full", "cache limit message");

    Statistics stats = ctx.statistics();
    require_api_smoke(stats.function_count == 0, "cache limit function count");
    require_api_smoke(stats.compilation_attempts == 1, "cache limit attempt count");
    require_api_smoke(stats.compilation_failures == 1, "cache limit failure count");
    require_api_smoke(stats.compilation_successes == 0, "cache limit success count");
}

static void api_smoke_vectorized_policy_error() {
    JitContext ctx{};
    if (!api_smoke_host_supports_vectorization()) {
        ErrorInfo info = expect_api_smoke_simjit_error(
            [&]() { ctx.set_policy(CompilePolicy::Vectorized); }, ErrorModule::JIT, ErrorKind::JitFailure,
            ErrorSubKind::UnsupportedHostFeature, "vectorized policy host capability");
        require_api_smoke_contains(info.message, "does not support vectorization", "vectorized policy host message");
        require_api_smoke(ctx.policy() == CompilePolicy::BestEffort,
                          "rejected vectorized policy leaves policy unchanged");
        return;
    }
    ctx.set_policy(CompilePolicy::Vectorized);

    ErrorInfo info = expect_api_smoke_simjit_error(
        [&]() {
            (void)vectorized_function<InputArr<I32>, OutputArr<I32>>(ctx, "api-smoke-vectorized-scalar-only",
                                                                     [](FunctionBuilder &b) {
                                                                         b.scalar_only();
                                                                         b.output_arg(b.input_arg(I32));
                                                                     });
        },
        ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::UnsupportedFeature, "vectorized scalar-only function");
    require_api_smoke_contains(info.message, "only scalar", "vectorized scalar-only message");
}

static void api_smoke_best_effort_records_vectorization_error() {
    JitContext ctx{};
    ctx.debug_options().record_vectorization_fail_exception = true;

    void *fn = ctx.build_and_compile("api-smoke-best-effort-vectorization-error", api_smoke_build_many_outputs);
    require_api_smoke(fn != nullptr, "best-effort fallback compile");
    if (api_smoke_host_supports_vectorization()) {
        require_api_smoke(!ctx.debug_snapshot().vectorization_exception.empty(), "recorded vectorization error");
        require_api_smoke_contains(ctx.debug_snapshot().vectorization_exception, "too many roots",
                                   "vectorization error details");
    } else {
        require_api_smoke(ctx.debug_snapshot().vectorization_exception.empty(),
                          "no vectorization attempt without host support");
    }
}

static void api_smoke_raw_call_errors() {
    expect_api_smoke_simjit_error([&]() { call_fn_ptr(nullptr, 0, nonstd::span<void *>{nullptr, 0}); },
                                  ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::LimitExceeded,
                                  "empty raw call args");

    std::array<void *, MaxFunctionArgumentCount + 1> too_many_args{};
    expect_api_smoke_simjit_error(
        [&]() { call_fn_ptr(nullptr, 0, nonstd::span<void *>{too_many_args.data(), too_many_args.size()}); },
        ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::LimitExceeded, "too many raw call args");
}

static void api_smoke_bug_report() {
    JitContext empty_ctx{};
    std::string empty_report = empty_ctx.bug_report();
    require_api_smoke_contains(empty_report, "policy: best-effort", "empty bug report policy");
    require_api_smoke_contains(empty_report, "function_identifiers: <empty>", "empty bug report identifiers");
    require_api_smoke_contains(empty_report, "=== HIR ===\n<empty>", "empty bug report section");

    if (api_smoke_host_supports_vectorization()) {
        JitContext vectorized_ctx{};
        vectorized_ctx.set_policy(CompilePolicy::Vectorized);
        require_api_smoke_contains(vectorized_ctx.bug_report(), "policy: vectorized", "vectorized bug report policy");
    }

    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);
    ctx.debug_options().capture_on_success = true;
    ctx.debug_options().stages = DebugStage::All;

    auto func =
        vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-bug-report", api_smoke_build_sum_i32);
    (void)func;

    const DebugSnapshot &snapshot = ctx.debug_snapshot();
    require_api_smoke(!snapshot.hir.empty(), "bug report HIR capture");
#if SIMJIT_ENABLE_SERIALIZATION
    require_api_smoke(!snapshot.serialized.empty(), "bug report serialized HIR capture");
#endif
    require_api_smoke(!snapshot.mir.empty(), "bug report MIR capture");
    require_api_smoke(!snapshot.asm_code.empty(), "bug report ASM capture");
    require_api_smoke(!snapshot.machine_code.empty(), "bug report machine code capture");

    std::string report = ctx.bug_report();
    require_api_smoke_contains(report, "policy: scalar", "bug report scalar policy");
    require_api_smoke_contains(report, "- api-smoke-bug-report", "bug report identifier");
    require_api_smoke_contains(report, "=== HIR ===", "bug report HIR section");
    require_api_smoke_contains(report, "=== MIR ===", "bug report MIR section");
    require_api_smoke_contains(report, "=== ASM ===", "bug report ASM section");
    require_api_smoke_contains(report, "=== machine code ===", "bug report machine code section");
}

static void api_smoke_jit_context_arch_constructor() {
    JitContext ctx{api_smoke_host_arch()};
    ctx.set_policy(CompilePolicy::Scalar);

    auto func =
        vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-arch-ctor", api_smoke_build_sum_i32);

    int32_t input[3] = {7, 8, 9};
    int32_t output = 0;
    func(3, input, &output);
    require_api_smoke(output == 24, "arch constructor function result");
    require_api_smoke(ctx.statistics().function_count == 1, "arch constructor function count");
}

static void api_smoke_jit_context_move_constructor() {
    static_assert(!std::is_copy_constructible_v<JitContext>, "JitContext must not be copy constructible");
    static_assert(!std::is_copy_assignable_v<JitContext>, "JitContext must not be copy assignable");
    static_assert(std::is_move_constructible_v<JitContext>, "JitContext must be move constructible");
    static_assert(std::is_move_assignable_v<JitContext>, "JitContext must be move assignable");

    JitContext source{};
    source.set_policy(CompilePolicy::Scalar);
    source.set_transformations(CodeTransformations::No);
    BuildLimits limits = source.build_limits();
    limits.max_cached_functions = 4;
    source.set_build_limits(limits);
    source.debug_options().capture_on_success = true;
    source.debug_options().stages = DebugStage::HIR;

    auto original =
        vectorized_function<InputArr<I32>, OutputScalar<I32>>(source, "api-smoke-move-ctor", api_smoke_build_sum_i32);
    (void)original;
    require_api_smoke(source.statistics().function_count == 1, "move ctor source function count before move");
    require_api_smoke(!source.debug_snapshot().hir.empty(), "move ctor source debug snapshot before move");

    JitContext moved{std::move(source)};

    require_api_smoke(moved.policy() == CompilePolicy::Scalar, "move ctor policy");
    require_api_smoke(moved.transformations() == CodeTransformations::No, "move ctor transformations");
    require_api_smoke(moved.build_limits().max_cached_functions == 4, "move ctor build limits");
    require_api_smoke(moved.debug_options().capture_on_success, "move ctor debug options");
    require_api_smoke(moved.debug_options().stages == DebugStage::HIR, "move ctor debug stages");
    require_api_smoke(!moved.debug_snapshot().hir.empty(), "move ctor debug snapshot");
    require_api_smoke(moved.statistics().function_count == 1, "move ctor moved function count");
    require_api_smoke(moved.statistics().last_compilation_ns > 0, "move ctor compilation timing");
    require_api_smoke_identifier(moved.function_identifiers(), "api-smoke-move-ctor", "move ctor function identifier");

    auto cached = find_vectorized_function<InputArr<I32>, OutputScalar<I32>>(moved, "api-smoke-move-ctor");
    require_api_smoke(cached.has_value(), "move ctor typed cache lookup");

    int32_t input[4] = {1, 2, 3, 4};
    int32_t output = 0;
    cached.value()(4, input, &output);
    require_api_smoke(output == 10, "move ctor cached function result");

    require_api_smoke(moved.delete_cached_function("api-smoke-move-ctor"), "move ctor delete cached function");
    require_api_smoke(moved.statistics().function_count == 0, "move ctor function count after delete");
}

static void api_smoke_jit_context_move_assignment() {
    JitContext old_owner{};
    old_owner.set_policy(CompilePolicy::Scalar);
    (void)vectorized_function<InputArr<I32>, OutputScalar<I32>>(old_owner, "api-smoke-move-assign-old",
                                                                api_smoke_build_sum_i32);
    require_api_smoke(old_owner.statistics().function_count == 1, "move assignment old owner function count");

    JitContext source{};
    source.set_policy(CompilePolicy::Scalar);
    source.debug_options().capture_on_success = true;
    source.debug_options().stages = DebugStage::All;
    (void)vectorized_function<InputArr<I32>, OutputScalar<I32>>(source, "api-smoke-move-assign-new",
                                                                api_smoke_build_sum_i32);
    require_api_smoke(source.statistics().function_count == 1, "move assignment source function count before move");
    require_api_smoke(!source.debug_snapshot().asm_code.empty(), "move assignment source ASM snapshot before move");

    old_owner = std::move(source);

    require_api_smoke(old_owner.policy() == CompilePolicy::Scalar, "move assignment policy");
    require_api_smoke(old_owner.debug_options().capture_on_success, "move assignment debug options");
    require_api_smoke(old_owner.debug_options().stages == DebugStage::All, "move assignment debug stages");
    require_api_smoke(!old_owner.debug_snapshot().asm_code.empty(), "move assignment debug snapshot");
    require_api_smoke(old_owner.statistics().function_count == 1, "move assignment moved function count");
    require_api_smoke_identifier(old_owner.function_identifiers(), "api-smoke-move-assign-new",
                                 "move assignment new function identifier");
    require_api_smoke(old_owner.find_cached_function("api-smoke-move-assign-old") == nullptr,
                      "move assignment old cache replaced");

    auto cached = find_vectorized_function<InputArr<I32>, OutputScalar<I32>>(old_owner, "api-smoke-move-assign-new");
    require_api_smoke(cached.has_value(), "move assignment typed cache lookup");

    int32_t input[5] = {5, 4, 3, 2, 1};
    int32_t output = 0;
    cached.value()(5, input, &output);
    require_api_smoke(output == 15, "move assignment cached function result");

    old_owner = api_smoke_move_ref(old_owner);
    require_api_smoke(old_owner.statistics().function_count == 1, "self move assignment keeps function count");
    auto cached_after_self_move =
        find_vectorized_function<InputArr<I32>, OutputScalar<I32>>(old_owner, "api-smoke-move-assign-new");
    require_api_smoke(cached_after_self_move.has_value(), "self move assignment keeps cache entry");

    old_owner.clear();
    require_api_smoke(old_owner.policy() == CompilePolicy::Scalar, "move assignment clear preserves policy");
    require_api_smoke(old_owner.debug_options().capture_on_success, "move assignment clear preserves debug options");
    require_api_smoke(old_owner.statistics().function_count == 0, "move assignment clear function count");
    require_api_smoke(old_owner.statistics().last_compilation_ns == 0,
                      "move assignment clear resets compilation timing");
}

static void api_smoke_bitmask_roundtrip() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto isnan_func = vectorized_function<InputArr<F32>, OutputArr<I1>>(ctx, "api-smoke-isnan", [](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I1);
        b.store(b.isnan(x), dst);
    });

    auto count_func =
        vectorized_function<InputArr<I1>, OutputScalar<I64>>(ctx, "api-smoke-count-bits", [](FunctionBuilder &b) {
            Predicate x = b.input_predicate_arg();
            Argument dst = b.arg(I64);
            b.countif(x, dst);
        });

    float input[1000];
    for (int i = 0; i < 1000; ++i) {
        input[i] = (i % 2 == 0) ? static_cast<float>(i) : std::numeric_limits<float>::quiet_NaN();
    }

    uint8_t bits[(1000 + 7) / 8] = {};
    isnan_func(1000, input, reinterpret_cast<Bitmask *>(bits));

    int64_t count = 0;
    count_func(1000, reinterpret_cast<Bitmask *>(bits), &count);
    require_api_smoke(count == 500, "bitmask count");
}

static void api_smoke_transform_and_aggregate() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto func =
        vectorized_function<InputArr<I32>, OutputArr<I32>, OutputScalar<I64>, OutputScalar<I32>, OutputScalar<I32>>(
            ctx, "api-smoke-transform-aggregate", [](FunctionBuilder &b) {
                Value x = b.input_arg(I32);
                Argument transformed_out = b.arg(I32);
                Argument sum_out = b.arg(I64);
                Argument min_out = b.arg(I32);
                Argument max_out = b.arg(I32);

                Predicate cond = b.cmp_gt(x, b.i32(100));
                Value transformed =
                    b.select(cond, b.add(b.mul(x, b.i32(3)), b.i32(10)), b.sub(b.mul(x, b.i32(2)), b.i32(5)));

                b.store(transformed, transformed_out);
                b.sum_if(b.sext(transformed, I64), b.cmp_gt(transformed, b.i32(50)), sum_out);
                b.min_agg(transformed, min_out);
                b.max_agg_if(transformed, b.cmp_lt(transformed, b.i32(200)), max_out);
            });

    constexpr int N = 1000;
    int32_t input[N];
    int32_t transformed[N] = {};
    int64_t sum = 0;
    int32_t min_value = 0;
    int32_t max_value = 0;

    for (int i = 0; i < N; ++i) {
        input[i] = i;
    }

    func(N, input, transformed, &sum, &min_value, &max_value);

    int64_t expected_sum = 0;
    int32_t expected_min = std::numeric_limits<int32_t>::max();
    int32_t expected_max = std::numeric_limits<int32_t>::min();
    for (int i = 0; i < N; ++i) {
        int32_t expected = input[i] > 100 ? input[i] * 3 + 10 : input[i] * 2 - 5;
        require_api_smoke(transformed[i] == expected, "transformed output");
        if (expected > 50) { expected_sum += expected; }
        if (expected < expected_min) { expected_min = expected; }
        if (expected < 200 && expected > expected_max) { expected_max = expected; }
    }

    require_api_smoke(sum == expected_sum, "transform aggregate sum");
    require_api_smoke(min_value == expected_min, "transform aggregate min");
    require_api_smoke(max_value == expected_max, "transform aggregate max");
}
#endif

static void api_smoke_load_deinterleave() {
    using hir::SpecialOp;

    // Repeated calls share lane steps (CSE), and the carrier keeps its explicit
    // interleaved-half declaration.
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument src = builder.arg_interleaved(I32);
        DeinterleavedPair first = builder.load2(src);
        DeinterleavedPair second = builder.load2(src);
        require_api_smoke(first.even.step_ == second.even.step_, "deinterleave even lane CSE");
        require_api_smoke(first.odd.step_ == second.odd.step_, "deinterleave odd lane CSE");
        require_api_smoke(first.even.step_ != first.odd.step_, "deinterleave lanes are distinct steps");
        builder.store(builder.add(first.even, builder.i32(1)), builder.arg(I32));
        builder.store(second.odd, builder.arg(I32));
        hir::Function *fn = builder.build();
        require_api_smoke(fn->args[src.idx_].dtype == I64 && fn->args[src.idx_].interleaved &&
                              fn->args[src.idx_].interleaved_half_dtype == I32,
                          "deinterleave carrier declaration");
        size_t lane_count = 0;
        std::vector<uint8_t> traversal_state(fn->step_id_count, 0);
        for (hir::Step *root : fn->step_roots) {
            hir::traverse_steps_postorder_unique(root, traversal_state, [&](hir::Step *step) {
                if (!step->is(hir::StepKind::LoadDeinterleave)) { return; }
                auto &data = step->step_data<hir::StepKind::LoadDeinterleave>();
                require_api_smoke(data.lane_count == 2, "deinterleave lane count is 2");
                require_api_smoke(data.lane < data.lane_count, "deinterleave lane index in range");
                ++lane_count;
            });
        }
        require_api_smoke(lane_count == 2, "CSE collapses repeated deinterleave to two lane steps");
    }

    // x86 vectorizes deinterleave through vpermi2var.
    {
        SpecialOp x86_ops = hir::supported_vector_special_ops_for_arch(Arch::Amd64_AVX512);
        require_api_smoke(bool(x86_ops & SpecialOp::SmallDeinterleave), "x86 small deinterleave capability");
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Amd64_AVX512};
        FunctionBuilder builder{context};
        DeinterleavedPair pair = builder.load2(builder.arg_interleaved(I32));
        builder.store(pair.even, builder.arg(I32));
        builder.store(pair.odd, builder.arg(I32));
        hir::Function *fn = builder.build();
        require_api_smoke(!fn->is_scalar_only(), "x86 deinterleave vectorizes");
    }

    // i8 lanes require VBMI; hosts without it stage them to scalar while wider lanes stay vectorized.
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Amd64_AVX512};
        context.host_supported_vector_special_ops =
            hir::supported_vector_special_ops_for_arch(Arch::Amd64_AVX512) & ~SpecialOp::SmallDeinterleave;
        FunctionBuilder builder{context};
        DeinterleavedPair i8_pair = builder.load2(builder.arg_interleaved(I8));
        builder.store(i8_pair.even, builder.arg(I8));
        builder.store(i8_pair.odd, builder.arg(I8));
        builder.store(builder.load2(builder.arg_interleaved(I32)).even, builder.arg(I32));
        hir::Function *fn = builder.build();
        require_api_smoke(bool(fn->special_ops & SpecialOp::SmallDeinterleave), "i8 deinterleave op recorded");
        require_api_smoke(fn->is_scalar_only(), "i8 deinterleave stages to scalar without VBMI");
        require_api_smoke(bool(fn->unsupported_vector_special_ops() & SpecialOp::SmallDeinterleave),
                          "i8 deinterleave unsupported without VBMI");
    }

    // A gather fed by a deinterleave lane marks the interleaved input as a bounded index source.
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument pairs = builder.arg_interleaved(I32);
        DeinterleavedPair lanes = builder.load2(pairs);
        builder.store(builder.gather(lanes.even, builder.arg(I32)), builder.arg(I32));
        hir::Function *fn = builder.build();
        require_api_smoke(bool(fn->argument_usage[pairs.idx_] & hir::ArgumentUsage::BoundedIndex),
                          "deinterleaved gather index marks input bounded");
        require_api_smoke(fn->args[pairs.idx_].interleaved, "deinterleaved gather input stays marked");
    }

    // The single-lane API produces the same leaves as the pair API (cross-API CSE).
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument src = builder.arg_interleaved(I32);
        Value single = builder.load2_1(src);
        DeinterleavedPair pair = builder.load2(src);
        require_api_smoke(single.step_ == pair.odd.step_, "single-lane API unifies with pair odd lane");
        require_api_smoke(pair.even.step_ != nullptr, "pair even lane materialized");
        builder.store(single, builder.arg(I32));
        hir::Function *fn = builder.build();
        size_t lane_count = 0;
        std::vector<uint8_t> traversal_state(fn->step_id_count, 0);
        for (hir::Step *root : fn->step_roots) {
            hir::traverse_steps_postorder_unique(root, traversal_state, [&](hir::Step *step) {
                if (step->is(hir::StepKind::LoadDeinterleave)) { ++lane_count; }
            });
        }
        require_api_smoke(lane_count == 1, "only the consumed lane is reachable");
    }
    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            builder.load_deinterleave(builder.arg_interleaved(I32), 2, 2);
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::InvalidConfiguration,
        "out-of-range deinterleave lane is rejected");
    // N-way interleaving is not implemented yet: the general API is restricted to pairs.
    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            builder.load_deinterleave(builder.arg_interleaved(I32), 3, 0);
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::InvalidConfiguration,
        "3-way deinterleave is rejected");

    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            builder.load2(builder.arg(I32));
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::TypeError, "unmarked deinterleave is rejected");
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument arg = builder.arg(I128);
        builder.mark_arg_interleave(arg, I64);
        DeinterleavedPair pair = builder.load2(arg);
        require_api_smoke(pair.even.dtype() == I64 && pair.odd.dtype() == I64, "i128 carrier yields i64 pair values");
    }

    // store_interleave: pair operations use explicit declarations, while ordinary
    // carrier accesses remain legal on the same argument.
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument dst = builder.arg_interleaved(I32);
        DeinterleavedPair pair = builder.load2(builder.arg_interleaved(I32));
        builder.store_interleave(dst, pair.even, pair.odd);
        hir::Function *fn = builder.build();
        require_api_smoke(bool(fn->argument_usage[dst.idx_] & hir::ArgumentUsage::SequentialWrite),
                          "store interleave marks dst sequential write");
        require_api_smoke(fn->args[dst.idx_].interleaved, "store interleave destination is marked");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument dst = builder.arg_interleaved(I32);
        DeinterleavedPair pair = builder.load2(builder.arg_interleaved(I32));
        Predicate cond = builder.input_predicate_arg();
        builder.cond_store_interleave(pair.even, pair.odd, cond, dst, LoadStoreKind::Aligned);
        hir::Function *fn = builder.build();
        hir::Step *root = fn->step_roots[0];
        require_api_smoke(root->is(hir::StepKind::StoreInterleave), "conditional interleave store root kind");
        require_api_smoke(root->step_data<hir::StepKind::StoreInterleave>().cond == cond.step_,
                          "conditional interleave store keeps predicate");
        require_api_smoke(root->step_data<hir::StepKind::StoreInterleave>().kind == LoadStoreKind::Aligned,
                          "conditional interleave store keeps store kind");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument dst = builder.arg_interleaved(I32);
        DeinterleavedPair pair = builder.load2(builder.arg_interleaved(I32));
        builder.cond_store_interleave(pair.even, pair.odd, builder.true_(), dst);
        hir::Function *fn = builder.build();
        require_api_smoke(fn->step_roots[0]->step_data<hir::StepKind::StoreInterleave>().cond == nullptr,
                          "true conditional interleave store simplifies to unconditional");
    }
    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            builder.store_interleave(builder.arg(I32), builder.i32(0), builder.i32(1));
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::TypeError, "unmarked store-interleave is rejected");
    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            Argument dst = builder.arg_interleaved(I64);
            builder.store_interleave(dst, builder.i32(0), builder.i32(1));
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::TypeError, "mismatched interleave half is rejected");
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument mixed_src = builder.arg_interleaved(I32);
        Argument raw_dst = builder.arg(I64);
        Argument pair_dst = builder.arg_interleaved(I32);
        Value raw = builder.load(mixed_src);
        DeinterleavedPair pair = builder.load2(mixed_src);
        builder.store(raw, raw_dst);
        builder.store_interleave(pair_dst, pair.even, pair.odd);
        hir::Function *fn = builder.build();
        require_api_smoke(fn->step_roots.size() == 2, "mixed carrier and pair accesses both remain roots");
        require_api_smoke(fn->step_roots[0]->is(hir::StepKind::Store), "mixed carrier access keeps ordinary store");
        require_api_smoke(fn->step_roots[1]->is(hir::StepKind::StoreInterleave),
                          "mixed pair access keeps interleaved store");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument left = builder.arg(I128);
        Argument right = builder.arg(I128);
        Argument dst = builder.arg(I128);
        Predicate cond = builder.input_predicate_arg();
        builder.cond_store(builder.add(builder.load(left), builder.load(right)), cond, dst, LoadStoreKind::Aligned);
        hir::Function *fn = builder.build();
        require_api_smoke(fn->args[left.idx_].interleaved && fn->args[left.idx_].interleaved_half_dtype == I64,
                          "ordinary i128 input becomes i64 interleaved carrier");
        hir::Step *root = fn->step_roots[0];
        require_api_smoke(root->is(hir::StepKind::StoreInterleave), "ordinary i128 store rewrites to pair store");
        const auto &store = root->step_data<hir::StepKind::StoreInterleave>();
        require_api_smoke(store.cond == cond.step_ && store.kind == LoadStoreKind::Aligned,
                          "i128 store preserves predicate and kind");
        size_t pair_loads = 0;
        size_t i64_adds = 0;
        std::vector<uint8_t> traversal_state(fn->step_id_count, 0);
        for (hir::Step *step_root : fn->step_roots) {
            hir::traverse_steps_postorder_unique(step_root, traversal_state, [&](hir::Step *step) {
                require_api_smoke(step->dtype != I128, "rewritten i128 value is not reachable");
                if (step->is(hir::StepKind::LoadDeinterleave)) { ++pair_loads; }
                if (step->is(hir::StepKind::ArithBinary) && step->dtype == I64) { ++i64_adds; }
            });
        }
        require_api_smoke(pair_loads == 4, "i128 add has four pair load lanes");
        require_api_smoke(i64_adds == 3, "i128 add has low, high, and carry additions");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument src = builder.arg(I128);
        builder.store(builder.rotl(builder.load(src), builder.con(64, I128)), builder.arg(I128));
        hir::Function *fn = builder.build();
        size_t arith_count = 0;
        std::vector<uint8_t> traversal_state(fn->step_id_count, 0);
        for (hir::Step *root : fn->step_roots) {
            hir::traverse_steps_postorder_unique(root, traversal_state, [&](hir::Step *step) {
                require_api_smoke(step->dtype != I128, "i128 rotate rewrite leaves no i128 step");
                if (step->is(hir::StepKind::ArithBinary)) { ++arith_count; }
            });
        }
        require_api_smoke(arith_count == 0, "i128 rotate by 64 only swaps pair halves");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument src = builder.arg(I128);
        Argument count = builder.arg(I128);
        builder.store(builder.sll(builder.load(src), builder.load(count)), builder.arg(I128));
        hir::Function *fn = builder.build();
        size_t i64_funnels = 0;
        std::vector<uint8_t> traversal_state(fn->step_id_count, 0);
        for (hir::Step *root : fn->step_roots) {
            hir::traverse_steps_postorder_unique(root, traversal_state, [&](hir::Step *step) {
                require_api_smoke(step->dtype != I128, "variable i128 shift rewrite leaves no i128 step");
                if (step->is(hir::StepKind::FunnelShift)) { ++i64_funnels; }
            });
        }
        require_api_smoke(i64_funnels == 1, "variable i128 shift uses one i64 funnel shift");
        require_api_smoke(bool(fn->special_ops & SpecialOp::FunnelShift), "i128 shift records funnel shift use");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument left = builder.arg(I128);
        Argument right = builder.arg(I128);
        builder.store(builder.mul(builder.load(left), builder.load(right)), builder.arg(I128));
        hir::Function *fn = builder.build();
        size_t mul32_count = 0;
        std::vector<uint8_t> traversal_state(fn->step_id_count, 0);
        for (hir::Step *root : fn->step_roots) {
            hir::traverse_steps_postorder_unique(root, traversal_state, [&](hir::Step *step) {
                require_api_smoke(step->dtype != I128, "i128 multiply rewrite leaves no i128 step");
                if (step->is(hir::StepKind::ArithBinary) &&
                    step->step_data<hir::StepKind::ArithBinary>().op == ArithBinaryOp::Mul64ZE) {
                    ++mul32_count;
                }
            });
        }
        require_api_smoke(mul32_count == 10, "i128 multiply uses ten 32x32 partial products");
    }
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All & ~CodeTransformations::I128Rewrite, Arch::Arm64_NEON};
        FunctionBuilder builder{context};
        Argument src = builder.arg(I128);
        Argument dst = builder.arg(I128);
        builder.store(builder.load(src), dst);
        expect_api_smoke_simjit_error([&] { (void)builder.build(); }, ErrorModule::HIR, ErrorKind::Unsupported,
                                      ErrorSubKind::UnsupportedFeature,
                                      "build rejects unlowered i128 when rewrite is disabled");
    }
}

static void api_smoke_funnel_shift() {
    using hir::SpecialOp;

    {
        SpecialOp x86_ops = hir::supported_vector_special_ops_for_arch(Arch::Amd64_AVX512);
        require_api_smoke(bool(x86_ops & SpecialOp::FunnelShift), "x86 funnel shift capability");
        SpecialOp arm_ops = hir::supported_vector_special_ops_for_arch(Arch::Arm64_NEON);
        require_api_smoke(bool(arm_ops & SpecialOp::FunnelShift), "arm funnel shift capability");
    }

    // x86 vectorizes funnel shifts through vpshldv, and the special op is recorded.
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Amd64_AVX512};
        FunctionBuilder builder{context};
        Value y = builder.input_arg(I32);
        Value z = builder.input_arg(I32);
        Value w = builder.input_arg(I32);
        builder.store(builder.funnel_shift_left(y, z, w), builder.arg(I32));
        hir::Function *fn = builder.build();
        require_api_smoke(bool(fn->special_ops & SpecialOp::FunnelShift), "funnel shift op recorded");
        require_api_smoke(!fn->is_scalar_only(), "x86 funnel shift vectorizes");
    }

    // Hosts without VBMI2 stage funnel shift functions to scalar.
    {
        MemoryArena arena{};
        Context context{arena, "expr", CodeTransformations::All, Arch::Amd64_AVX512};
        context.host_supported_vector_special_ops =
            hir::supported_vector_special_ops_for_arch(Arch::Amd64_AVX512) & ~SpecialOp::FunnelShift;
        FunctionBuilder builder{context};
        Value y = builder.input_arg(I32);
        Value z = builder.input_arg(I32);
        Value w = builder.input_arg(I32);
        builder.store(builder.funnel_shift_left(y, z, w), builder.arg(I32));
        hir::Function *fn = builder.build();
        require_api_smoke(fn->is_scalar_only(), "funnel shift stages to scalar without VBMI2");
        require_api_smoke(bool(fn->unsupported_vector_special_ops() & SpecialOp::FunnelShift),
                          "funnel shift unsupported without VBMI2");
    }

    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            builder.funnel_shift_left(builder.i8(1), builder.i8(2), builder.i8(3));
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::TypeError, "i8 funnel shift is rejected");
    expect_api_smoke_simjit_error(
        [&] {
            MemoryArena arena{};
            Context context{arena, "expr", CodeTransformations::All, Arch::Arm64_NEON};
            FunctionBuilder builder{context};
            builder.funnel_shift_right(builder.i64(1), builder.i32(2), builder.i64(3));
        },
        ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::TypeError, "mixed funnel shift types are rejected");
}

static void run_public_api_smoke_tests() {
#if SIMJIT_ASMJIT_BACKEND
    api_smoke_sum();
    api_smoke_special_op_capabilities();
    api_smoke_load_deinterleave();
    api_smoke_funnel_shift();
    api_smoke_raw_build_callbacks();
    api_smoke_try_compile_and_debug();
    api_smoke_serialized_debug_stage();
    api_smoke_try_compile_generic_error();
    api_smoke_cache_and_casts();
    api_smoke_cache_error_paths();
    api_smoke_cache_limit_error();
    api_smoke_vectorized_policy_error();
    api_smoke_best_effort_records_vectorization_error();
    api_smoke_raw_call_errors();
    api_smoke_bug_report();
    api_smoke_jit_context_arch_constructor();
    api_smoke_jit_context_move_constructor();
    api_smoke_jit_context_move_assignment();
    api_smoke_bitmask_roundtrip();
    api_smoke_transform_and_aggregate();
#endif
}
