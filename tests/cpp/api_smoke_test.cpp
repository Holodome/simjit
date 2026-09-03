// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

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

static Arch api_smoke_host_arch() {
#if defined(__aarch64__) || defined(_M_ARM64)
    return Arch::Arm64_NEON;
#elif defined(__x86_64__) || defined(_M_X64)
    return Arch::Amd64_AVX512;
#else
#error "Unsupported public API smoke host architecture"
#endif
}

static void api_smoke_sum() {
    JitContext ctx{};
    ctx.set_policy(CompilePolicy::Scalar);

    auto func = vectorized_function<InputArr<I32>, OutputScalar<I32>>(ctx, "api-smoke-sum", api_smoke_build_sum_i32);

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

    ErrorInfo kind_mismatch = expect_api_smoke_simjit_error(
        [&]() { (void)find_vectorized_function<InputConst<I32>, OutputScalar<I32>>(ctx, "api-smoke-cache-errors"); },
        ErrorModule::JIT, ErrorKind::JitFailure, ErrorSubKind::ArgumentMismatch, "cached argument kind mismatch");
    require_api_smoke_contains(kind_mismatch.message, "kind mismatch", "kind mismatch message");

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
    require_api_smoke(!ctx.debug_snapshot().vectorization_exception.empty(), "recorded vectorization error");
    require_api_smoke_contains(ctx.debug_snapshot().vectorization_exception, "too many roots",
                               "vectorization error details");
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

    JitContext vectorized_ctx{};
    vectorized_ctx.set_policy(CompilePolicy::Vectorized);
    require_api_smoke_contains(vectorized_ctx.bug_report(), "policy: vectorized", "vectorized bug report policy");

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
    old_owner.set_policy(CompilePolicy::Vectorized);
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

static void run_public_api_smoke_tests() {
#if SIMJIT_ASMJIT_BACKEND
    api_smoke_sum();
    api_smoke_try_compile_and_debug();
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
