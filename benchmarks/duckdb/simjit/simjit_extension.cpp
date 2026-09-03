// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "duckdb.hpp"
#include "duckdb/common/vector/constant_vector.hpp"
#include "duckdb/common/vector/dictionary_vector.hpp"
#include "duckdb/main/extension/extension_loader.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "simjit/jit.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

namespace duckdb {
namespace {

using namespace simjit::types;

constexpr const char *SIMJIT_HELLO_MESSAGE = "hello world from simjit duckdb extension";
constexpr idx_t SIMJIT_DUCKDB_VECTOR_SIZE = 2048;
constexpr size_t SIMJIT_ALL_VALID_MASK_ENTRIES = (SIMJIT_DUCKDB_VECTOR_SIZE + 63) / 64;

using AddI64Kernel = simjit::jit::FunctionHolder<const int64_t *, const int64_t *, int64_t *>;
using NetI64Kernel = simjit::jit::FunctionHolder<const int64_t *, const int64_t *, int64_t *>;
using OutputBoolArr = simjit::jit::JitTypeWrapper<simjit::ArgumentKind::Dst, bool *, I8>;
using ShipFilterKernel = simjit::jit::FunctionHolder<const int32_t *, const int32_t *, bool *>;
using SumAddI64Kernel = simjit::jit::FunctionHolder<const int64_t *, const int64_t *, int64_t *>;
using SumNetI64Kernel = simjit::jit::FunctionHolder<const int64_t *, const int64_t *, int64_t *>;
using CountShipFilterKernel = simjit::jit::FunctionHolder<const int32_t *, const int32_t *, int64_t *>;
using Q1MultiKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const int64_t *, int64_t *, int64_t *, int64_t *>;
using NullableRevenueKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const int64_t *, const int64_t *,
                                const simjit::jit::Bitmask *, const simjit::jit::Bitmask *,
                                const simjit::jit::Bitmask *, int64_t *, int64_t *, int64_t *, int64_t *>;
using Q19MaskKernel = simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const int64_t *, const int64_t *,
                                                  const int64_t *, const int64_t *, int64_t *, int64_t *>;
using SalesWideFlatKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const int64_t *, const int8_t *, const int8_t *,
                                int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesWideNullableFlatKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const simjit::jit::Bitmask *, const int64_t *,
                                const int8_t *, const int8_t *, int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesWideGatherKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const int64_t *, const int8_t *, const int32_t *,
                                const int8_t *, const int32_t *, int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesWideNullableGatherKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const simjit::jit::Bitmask *, const int64_t *,
                                const int8_t *, const int32_t *, const int8_t *, const int32_t *, int64_t *, int64_t *,
                                int64_t *, int64_t *, int64_t *>;
using SalesWideSpecKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const int64_t *, const int32_t *, const int32_t *,
                                int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesWideNullableSpecKernel =
    simjit::jit::FunctionHolder<const int64_t *, const int64_t *, const simjit::jit::Bitmask *, const int64_t *,
                                const int32_t *, const int32_t *, int64_t *, int64_t *, int64_t *, int64_t *,
                                int64_t *>;
using SalesMixedFlatKernel =
    simjit::jit::FunctionHolder<const int16_t *, const int64_t *, const int16_t *, const int8_t *, const int8_t *,
                                int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesMixedNullableFlatKernel =
    simjit::jit::FunctionHolder<const int16_t *, const int64_t *, const simjit::jit::Bitmask *, const int16_t *,
                                const int8_t *, const int8_t *, int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesMixedGatherKernel =
    simjit::jit::FunctionHolder<const int16_t *, const int64_t *, const int16_t *, const int8_t *, const int32_t *,
                                const int8_t *, const int32_t *, int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesMixedNullableGatherKernel =
    simjit::jit::FunctionHolder<const int16_t *, const int64_t *, const simjit::jit::Bitmask *, const int16_t *,
                                const int8_t *, const int32_t *, const int8_t *, const int32_t *, int64_t *, int64_t *,
                                int64_t *, int64_t *, int64_t *>;
using SalesMixedSpecKernel =
    simjit::jit::FunctionHolder<const int16_t *, const int64_t *, const int16_t *, const int32_t *, const int32_t *,
                                int64_t *, int64_t *, int64_t *, int64_t *, int64_t *>;
using SalesMixedNullableSpecKernel =
    simjit::jit::FunctionHolder<const int16_t *, const int64_t *, const simjit::jit::Bitmask *, const int16_t *,
                                const int32_t *, const int32_t *, int64_t *, int64_t *, int64_t *, int64_t *,
                                int64_t *>;

static std::atomic<uint64_t> kernel_generation{1};

template <class HOLDER> class LazyKernel {
public:
    explicit LazyKernel(std::string name) : name(std::move(name)) {
        context.set_policy(simjit::jit::CompilePolicy::BestEffort);
        context.debug_options().record_vectorization_fail_exception = true;
        seen_generation = kernel_generation.load(std::memory_order_relaxed);
    }

    template <class BUILD> HOLDER &Get(BUILD build) {
        ResetIfStale();
        if (holder) { return *holder; }

        const auto before = std::chrono::steady_clock::now();
        auto compiled = build(context, name.c_str());
        holder = std::make_unique<HOLDER>(std::move(compiled));
        const auto after = std::chrono::steady_clock::now();
        last_compile_us =
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(after - before).count());

        const auto &reason = context.debug_snapshot().vectorization_exception;
        vectorizer_status = reason.empty() ? "vectorized" : "scalar_fallback:" + reason;
        return *holder;
    }

    void ResetLocal() {
        holder.reset();
        context.clear();
        last_compile_us = 0;
        vectorizer_status = "not_compiled";
        seen_generation = kernel_generation.load(std::memory_order_relaxed);
    }

    uint64_t LastCompileUs() const {
        if (IsStale()) { return 0; }
        return last_compile_us;
    }

    std::string VectorizerStatus() const {
        if (IsStale()) { return "not_compiled"; }
        return vectorizer_status;
    }

    std::string name;

private:
    bool IsStale() const { return seen_generation != kernel_generation.load(std::memory_order_relaxed); }

    void ResetIfStale() {
        if (IsStale()) { ResetLocal(); }
    }

    simjit::jit::JitContext context;
    std::unique_ptr<HOLDER> holder;
    uint64_t last_compile_us = 0;
    uint64_t seen_generation = 0;
    std::string vectorizer_status = "not_compiled";
};

static LazyKernel<AddI64Kernel> &AddI64KernelCache() {
    thread_local LazyKernel<AddI64Kernel> kernel("simjit_add_i64_nn");
    return kernel;
}

static LazyKernel<NetI64Kernel> &NetI64KernelCache() {
    thread_local LazyKernel<NetI64Kernel> kernel("simjit_net_i64_nn");
    return kernel;
}

static LazyKernel<ShipFilterKernel> &ShipFilterKernelCache() {
    thread_local LazyKernel<ShipFilterKernel> kernel("simjit_ship_filter_i32_nn");
    return kernel;
}

static LazyKernel<SumAddI64Kernel> &SumAddI64KernelCache() {
    thread_local LazyKernel<SumAddI64Kernel> kernel("simjit_sum_add_i64_nn");
    return kernel;
}

static LazyKernel<SumNetI64Kernel> &SumNetI64KernelCache() {
    thread_local LazyKernel<SumNetI64Kernel> kernel("simjit_sum_net_i64_nn");
    return kernel;
}

static LazyKernel<CountShipFilterKernel> &CountShipFilterKernelCache() {
    thread_local LazyKernel<CountShipFilterKernel> kernel("simjit_count_ship_filter_i32_nn");
    return kernel;
}

static LazyKernel<Q1MultiKernel> &Q1MultiKernelCache() {
    thread_local LazyKernel<Q1MultiKernel> kernel("simjit_q1_multi_i64");
    return kernel;
}

static LazyKernel<NullableRevenueKernel> &NullableRevenueKernelCache() {
    thread_local LazyKernel<NullableRevenueKernel> kernel("simjit_nullable_revenue_i64");
    return kernel;
}

static LazyKernel<Q19MaskKernel> &Q19MaskKernelCache() {
    thread_local LazyKernel<Q19MaskKernel> kernel("simjit_q19_mask_i64");
    return kernel;
}

static std::string SalesKernelName(const char *numeric, bool nullable_price, bool const_discount, const char *cat) {
    return std::string("simjit_sales_") + numeric + "__price_" + (nullable_price ? "nullable10" : "nn") + "__disc_" +
           (const_discount ? "const" : "column") + "__cat_" + cat;
}

#define SALES_KERNEL_CACHE(NAME, HOLDER, NUMERIC, NULLABLE_PRICE, CONST_DISCOUNT, CAT)                         \
    static LazyKernel<HOLDER> &NAME() {                                                                        \
        thread_local LazyKernel<HOLDER> kernel(SalesKernelName(NUMERIC, NULLABLE_PRICE, CONST_DISCOUNT, CAT)); \
        return kernel;                                                                                         \
    }

SALES_KERNEL_CACHE(SalesWideFlatColumnCache, SalesWideFlatKernel, "wide_i64", false, false, "flat")
SALES_KERNEL_CACHE(SalesWideFlatConstCache, SalesWideFlatKernel, "wide_i64", false, true, "flat")
SALES_KERNEL_CACHE(SalesWideNullableFlatColumnCache, SalesWideNullableFlatKernel, "wide_i64", true, false, "flat")
SALES_KERNEL_CACHE(SalesWideNullableFlatConstCache, SalesWideNullableFlatKernel, "wide_i64", true, true, "flat")
SALES_KERNEL_CACHE(SalesWideGatherColumnCache, SalesWideGatherKernel, "wide_i64", false, false, "dict_gather")
SALES_KERNEL_CACHE(SalesWideGatherConstCache, SalesWideGatherKernel, "wide_i64", false, true, "dict_gather")
SALES_KERNEL_CACHE(SalesWideNullableGatherColumnCache, SalesWideNullableGatherKernel, "wide_i64", true, false,
                   "dict_gather")
SALES_KERNEL_CACHE(SalesWideNullableGatherConstCache, SalesWideNullableGatherKernel, "wide_i64", true, true,
                   "dict_gather")
SALES_KERNEL_CACHE(SalesWideSpecColumnCache, SalesWideSpecKernel, "wide_i64", false, false, "dict_spec")
SALES_KERNEL_CACHE(SalesWideSpecConstCache, SalesWideSpecKernel, "wide_i64", false, true, "dict_spec")
SALES_KERNEL_CACHE(SalesWideNullableSpecColumnCache, SalesWideNullableSpecKernel, "wide_i64", true, false, "dict_spec")
SALES_KERNEL_CACHE(SalesWideNullableSpecConstCache, SalesWideNullableSpecKernel, "wide_i64", true, true, "dict_spec")

SALES_KERNEL_CACHE(SalesMixedFlatColumnCache, SalesMixedFlatKernel, "mixed_narrow", false, false, "flat")
SALES_KERNEL_CACHE(SalesMixedFlatConstCache, SalesMixedFlatKernel, "mixed_narrow", false, true, "flat")
SALES_KERNEL_CACHE(SalesMixedNullableFlatColumnCache, SalesMixedNullableFlatKernel, "mixed_narrow", true, false, "flat")
SALES_KERNEL_CACHE(SalesMixedNullableFlatConstCache, SalesMixedNullableFlatKernel, "mixed_narrow", true, true, "flat")
SALES_KERNEL_CACHE(SalesMixedGatherColumnCache, SalesMixedGatherKernel, "mixed_narrow", false, false, "dict_gather")
SALES_KERNEL_CACHE(SalesMixedGatherConstCache, SalesMixedGatherKernel, "mixed_narrow", false, true, "dict_gather")
SALES_KERNEL_CACHE(SalesMixedNullableGatherColumnCache, SalesMixedNullableGatherKernel, "mixed_narrow", true, false,
                   "dict_gather")
SALES_KERNEL_CACHE(SalesMixedNullableGatherConstCache, SalesMixedNullableGatherKernel, "mixed_narrow", true, true,
                   "dict_gather")
SALES_KERNEL_CACHE(SalesMixedSpecColumnCache, SalesMixedSpecKernel, "mixed_narrow", false, false, "dict_spec")
SALES_KERNEL_CACHE(SalesMixedSpecConstCache, SalesMixedSpecKernel, "mixed_narrow", false, true, "dict_spec")
SALES_KERNEL_CACHE(SalesMixedNullableSpecColumnCache, SalesMixedNullableSpecKernel, "mixed_narrow", true, false,
                   "dict_spec")
SALES_KERNEL_CACHE(SalesMixedNullableSpecConstCache, SalesMixedNullableSpecKernel, "mixed_narrow", true, true,
                   "dict_spec")

#undef SALES_KERNEL_CACHE

static void PrintHello() {
    std::printf("%s\n", SIMJIT_HELLO_MESSAGE);
    std::fflush(stdout);
}

static AddI64Kernel BuildAddI64(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
                                            simjit::jit::OutputArr<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto left = builder.input_arg(I64);
            auto right = builder.input_arg(I64);
            builder.output_arg(builder.add(left, right));
        });
}

static NetI64Kernel BuildNetI64(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
                                            simjit::jit::OutputArr<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto price = builder.input_arg(I64);
            auto discount = builder.input_arg(I64);
            auto base = builder.mul(price, builder.sub(builder.i64(10000), discount));
            builder.output_arg(builder.div(base, builder.i64(10000)));
        });
}

static ShipFilterKernel BuildShipFilter(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I32>, simjit::jit::InputArr<I32>, OutputBoolArr>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto shipdate = builder.input_arg(I32);
            auto quantity = builder.input_arg(I32);
            auto pred =
                builder.and_(builder.cmp_le(shipdate, builder.i32(10592)), builder.cmp_lt(quantity, builder.i32(24)));
            builder.output_arg(builder.bit2bool(pred));
        });
}

static SumAddI64Kernel BuildSumAddI64(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
                                            simjit::jit::OutputScalar<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto left = builder.input_arg(I64);
            auto right = builder.input_arg(I64);
            auto sum = builder.arg(I64);
            builder.sum(builder.add(left, right), sum);
        });
}

static SumNetI64Kernel BuildSumNetI64(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
                                            simjit::jit::OutputScalar<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto price = builder.input_arg(I64);
            auto discount = builder.input_arg(I64);
            auto sum = builder.arg(I64);
            auto base = builder.mul(price, builder.sub(builder.i64(10000), discount));
            builder.sum(builder.div(base, builder.i64(10000)), sum);
        });
}

static CountShipFilterKernel BuildCountShipFilter(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I32>, simjit::jit::InputArr<I32>,
                                            simjit::jit::OutputScalar<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto shipdate = builder.input_arg(I32);
            auto quantity = builder.input_arg(I32);
            auto count = builder.arg(I64);
            auto pred =
                builder.and_(builder.cmp_le(shipdate, builder.i32(10592)), builder.cmp_lt(quantity, builder.i32(24)));
            builder.countif(pred, count);
        });
}

static Q1MultiKernel BuildQ1Multi(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
                                            simjit::jit::InputArr<I64>, simjit::jit::OutputScalar<I64>,
                                            simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto price = builder.input_arg(I64);
            auto discount = builder.input_arg(I64);
            auto tax = builder.input_arg(I64);
            auto sum_net = builder.arg(I64);
            auto sum_charge = builder.arg(I64);
            auto row_count = builder.arg(I64);

            auto discounted = builder.mul(price, builder.sub(builder.i64(10000), discount));
            auto net = builder.div(discounted, builder.i64(10000));
            auto charge =
                builder.div(builder.mul(discounted, builder.add(builder.i64(10000), tax)), builder.i64(100000000));
            builder.sum(net, sum_net);
            builder.sum(charge, sum_charge);
            builder.sum(builder.i64(1), row_count);
        });
}

static NullableRevenueKernel BuildNullableRevenue(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<
        simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
        simjit::jit::InputArr<I1>, simjit::jit::InputArr<I1>, simjit::jit::InputArr<I1>, simjit::jit::OutputScalar<I64>,
        simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>(
        context, name, [](simjit::FunctionBuilder &builder) {
            auto quantity_raw = builder.input_arg(I64);
            auto price = builder.input_arg(I64);
            auto discount_raw = builder.input_arg(I64);
            auto delay = builder.input_arg(I64);
            auto quantity_valid = builder.input_predicate_arg();
            auto price_valid = builder.input_predicate_arg();
            auto discount_valid = builder.input_predicate_arg();
            auto sum_valid_net = builder.arg(I64);
            auto sum_delay_net = builder.arg(I64);
            auto count_valid = builder.arg(I64);
            auto count_delay = builder.arg(I64);

            auto quantity = builder.select(quantity_valid, quantity_raw, builder.i64(0));
            auto discount = builder.select(discount_valid, discount_raw, builder.i64(0));
            auto valid = builder.and_(price_valid, builder.and_(builder.cmp_ne(price, builder.i64(0)),
                                                                builder.cmp_gt(quantity, builder.i64(0))));
            auto delay_valid = builder.and_(valid, builder.cmp_gt(delay, builder.i64(30)));
            auto net = builder.div(builder.mul(builder.mul(quantity, price), builder.sub(builder.i64(10000), discount)),
                                   builder.i64(10000));
            builder.sum_if(net, valid, sum_valid_net);
            builder.sum_if(net, delay_valid, sum_delay_net);
            builder.countif(valid, count_valid);
            builder.countif(delay_valid, count_delay);
        });
}

static simjit::Predicate Between(simjit::FunctionBuilder &builder, simjit::Value value, int64_t min_value,
                                 int64_t max_value) {
    return builder.and_(builder.cmp_ge(value, builder.i64(min_value)), builder.cmp_le(value, builder.i64(max_value)));
}

static Q19MaskKernel BuildQ19Mask(simjit::jit::JitContext &context, const char *name) {
    return simjit::jit::vectorized_function<
        simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>,
        simjit::jit::InputArr<I64>, simjit::jit::InputArr<I64>, simjit::jit::OutputScalar<I64>,
        simjit::jit::OutputScalar<I64>>(context, name, [](simjit::FunctionBuilder &builder) {
        auto brand = builder.input_arg(I64);
        auto container = builder.input_arg(I64);
        auto quantity = builder.input_arg(I64);
        auto size = builder.input_arg(I64);
        auto price = builder.input_arg(I64);
        auto discount = builder.input_arg(I64);
        auto row_count = builder.arg(I64);
        auto revenue_sum = builder.arg(I64);

        auto g1 =
            builder.and_(builder.cmp_eq(brand, builder.i64(12)),
                         builder.and_(Between(builder, container, 1, 4),
                                      builder.and_(Between(builder, quantity, 1, 11), Between(builder, size, 1, 5))));
        auto g2 =
            builder.and_(builder.cmp_eq(brand, builder.i64(23)),
                         builder.and_(Between(builder, container, 5, 8),
                                      builder.and_(Between(builder, quantity, 10, 20), Between(builder, size, 1, 10))));
        auto g3 =
            builder.and_(builder.cmp_eq(brand, builder.i64(34)),
                         builder.and_(Between(builder, container, 9, 12),
                                      builder.and_(Between(builder, quantity, 20, 30), Between(builder, size, 1, 15))));
        auto pred = builder.or_(g1, builder.or_(g2, g3));
        auto revenue = builder.div(builder.mul(price, builder.sub(builder.i64(10000), discount)), builder.i64(10000));
        builder.countif(pred, row_count);
        builder.sum_if(revenue, pred, revenue_sum);
    });
}

static simjit::Value SalesToI64(simjit::FunctionBuilder &builder, simjit::Value value) {
    if (value.dtype() == I64) { return value; }
    return builder.signed_cast(value, I64);
}

static void BuildSalesBody(simjit::FunctionBuilder &builder, simjit::Value quantity_raw, simjit::Value price,
                           simjit::MaybePredicate price_valid, bool nullable_price, simjit::Value discount_raw,
                           simjit::Value return_flag_raw, simjit::Value channel_raw) {
    auto quantity = SalesToI64(builder, quantity_raw);
    auto discount = SalesToI64(builder, discount_raw);
    auto return_flag = SalesToI64(builder, return_flag_raw);
    auto channel = SalesToI64(builder, channel_raw);
    auto sum_revenue = builder.arg(I64);
    auto channel2_revenue = builder.arg(I64);
    auto bulk_count = builder.arg(I64);
    auto revenue_rows = builder.arg(I64);
    auto channel2_rows = builder.arg(I64);

    auto discounted = builder.mul(builder.mul(quantity, price), builder.sub(builder.i64(10000), discount));
    auto net = builder.div(discounted, builder.i64(10000));
    auto signed_net =
        builder.select(builder.cmp_eq(return_flag, builder.i64(1)), builder.sub(builder.i64(0), net), net);
    auto channel2 = builder.cmp_eq(channel, builder.i64(2));
    if (nullable_price) {
        auto valid = price_valid.value();
        builder.sum_if(signed_net, valid, sum_revenue);
        builder.sum_if(signed_net, builder.and_(valid, channel2), channel2_revenue);
        builder.countif(valid, revenue_rows);
        builder.countif(builder.and_(valid, channel2), channel2_rows);
    } else {
        builder.sum(signed_net, sum_revenue);
        builder.sum_if(signed_net, channel2, channel2_revenue);
        builder.sum(builder.i64(1), revenue_rows);
        builder.countif(channel2, channel2_rows);
    }
    builder.countif(builder.cmp_ge(quantity, builder.i64(10)), bulk_count);
}

#define BUILD_SALES_FLAT_NN(NAME, HOLDER, QWRAP, QDTYPE, DWRAP, DDTYPE, DISCOUNT_LOAD)                               \
    static HOLDER NAME(simjit::jit::JitContext &context, const char *name) {                                         \
        return simjit::jit::vectorized_function<QWRAP, simjit::jit::InputArr<I64>, DWRAP, simjit::jit::InputArr<I8>, \
                                                simjit::jit::InputArr<I8>, simjit::jit::OutputScalar<I64>,           \
                                                simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>,      \
                                                simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>(     \
            context, name, [](simjit::FunctionBuilder &builder) {                                                    \
                auto quantity = builder.input_arg(QDTYPE);                                                           \
                auto price = builder.input_arg(I64);                                                                 \
                auto discount = builder.DISCOUNT_LOAD(DDTYPE);                                                       \
                auto return_flag = builder.input_arg(I8);                                                            \
                auto channel = builder.input_arg(I8);                                                                \
                BuildSalesBody(builder, quantity, price, simjit::MaybePredicate(nullptr), false, discount,           \
                               return_flag, channel);                                                                \
            });                                                                                                      \
    }

#define BUILD_SALES_FLAT_NULLABLE(NAME, HOLDER, QWRAP, QDTYPE, DWRAP, DDTYPE, DISCOUNT_LOAD)                 \
    static HOLDER NAME(simjit::jit::JitContext &context, const char *name) {                                 \
        return simjit::jit::vectorized_function<                                                             \
            QWRAP, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I1>, DWRAP, simjit::jit::InputArr<I8>,  \
            simjit::jit::InputArr<I8>, simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>,       \
            simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>( \
            context, name, [](simjit::FunctionBuilder &builder) {                                            \
                auto quantity = builder.input_arg(QDTYPE);                                                   \
                auto price = builder.input_arg(I64);                                                         \
                auto price_valid = builder.input_predicate_arg();                                            \
                auto discount = builder.DISCOUNT_LOAD(DDTYPE);                                               \
                auto return_flag = builder.input_arg(I8);                                                    \
                auto channel = builder.input_arg(I8);                                                        \
                BuildSalesBody(builder, quantity, price, price_valid, true, discount, return_flag, channel); \
            });                                                                                              \
    }

#define BUILD_SALES_GATHER_NN(NAME, HOLDER, QWRAP, QDTYPE, DWRAP, DDTYPE, DISCOUNT_LOAD)                             \
    static HOLDER NAME(simjit::jit::JitContext &context, const char *name) {                                         \
        return simjit::jit::vectorized_function<QWRAP, simjit::jit::InputArr<I64>, DWRAP, simjit::jit::InputArr<I8>, \
                                                simjit::jit::InputArr<I32>, simjit::jit::InputArr<I8>,               \
                                                simjit::jit::InputArr<I32>, simjit::jit::OutputScalar<I64>,          \
                                                simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>,      \
                                                simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>(     \
            context, name, [](simjit::FunctionBuilder &builder) {                                                    \
                auto quantity = builder.input_arg(QDTYPE);                                                           \
                auto price = builder.input_arg(I64);                                                                 \
                auto discount = builder.DISCOUNT_LOAD(DDTYPE);                                                       \
                auto return_child = builder.arg(I8);                                                                 \
                auto return_idx = builder.input_arg(I32);                                                            \
                auto channel_child = builder.arg(I8);                                                                \
                auto channel_idx = builder.input_arg(I32);                                                           \
                auto return_flag = builder.gather(return_idx, return_child);                                         \
                auto channel = builder.gather(channel_idx, channel_child);                                           \
                BuildSalesBody(builder, quantity, price, simjit::MaybePredicate(nullptr), false, discount,           \
                               return_flag, channel);                                                                \
            });                                                                                                      \
    }

#define BUILD_SALES_GATHER_NULLABLE(NAME, HOLDER, QWRAP, QDTYPE, DWRAP, DDTYPE, DISCOUNT_LOAD)               \
    static HOLDER NAME(simjit::jit::JitContext &context, const char *name) {                                 \
        return simjit::jit::vectorized_function<                                                             \
            QWRAP, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I1>, DWRAP, simjit::jit::InputArr<I8>,  \
            simjit::jit::InputArr<I32>, simjit::jit::InputArr<I8>, simjit::jit::InputArr<I32>,               \
            simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>,  \
            simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>(                                 \
            context, name, [](simjit::FunctionBuilder &builder) {                                            \
                auto quantity = builder.input_arg(QDTYPE);                                                   \
                auto price = builder.input_arg(I64);                                                         \
                auto price_valid = builder.input_predicate_arg();                                            \
                auto discount = builder.DISCOUNT_LOAD(DDTYPE);                                               \
                auto return_child = builder.arg(I8);                                                         \
                auto return_idx = builder.input_arg(I32);                                                    \
                auto channel_child = builder.arg(I8);                                                        \
                auto channel_idx = builder.input_arg(I32);                                                   \
                auto return_flag = builder.gather(return_idx, return_child);                                 \
                auto channel = builder.gather(channel_idx, channel_child);                                   \
                BuildSalesBody(builder, quantity, price, price_valid, true, discount, return_flag, channel); \
            });                                                                                              \
    }

#define BUILD_SALES_SPEC_NN(NAME, HOLDER, QWRAP, QDTYPE, DWRAP, DDTYPE, DISCOUNT_LOAD)                                \
    static HOLDER NAME(simjit::jit::JitContext &context, const char *name) {                                          \
        return simjit::jit::vectorized_function<QWRAP, simjit::jit::InputArr<I64>, DWRAP, simjit::jit::InputArr<I32>, \
                                                simjit::jit::InputArr<I32>, simjit::jit::OutputScalar<I64>,           \
                                                simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>,       \
                                                simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>(      \
            context, name, [](simjit::FunctionBuilder &builder) {                                                     \
                auto quantity = builder.input_arg(QDTYPE);                                                            \
                auto price = builder.input_arg(I64);                                                                  \
                auto discount = builder.DISCOUNT_LOAD(DDTYPE);                                                        \
                auto return_flag = builder.input_arg(I32);                                                            \
                auto channel = builder.input_arg(I32);                                                                \
                BuildSalesBody(builder, quantity, price, simjit::MaybePredicate(nullptr), false, discount,            \
                               return_flag, channel);                                                                 \
            });                                                                                                       \
    }

#define BUILD_SALES_SPEC_NULLABLE(NAME, HOLDER, QWRAP, QDTYPE, DWRAP, DDTYPE, DISCOUNT_LOAD)                 \
    static HOLDER NAME(simjit::jit::JitContext &context, const char *name) {                                 \
        return simjit::jit::vectorized_function<                                                             \
            QWRAP, simjit::jit::InputArr<I64>, simjit::jit::InputArr<I1>, DWRAP, simjit::jit::InputArr<I32>, \
            simjit::jit::InputArr<I32>, simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>,      \
            simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>, simjit::jit::OutputScalar<I64>>( \
            context, name, [](simjit::FunctionBuilder &builder) {                                            \
                auto quantity = builder.input_arg(QDTYPE);                                                   \
                auto price = builder.input_arg(I64);                                                         \
                auto price_valid = builder.input_predicate_arg();                                            \
                auto discount = builder.DISCOUNT_LOAD(DDTYPE);                                               \
                auto return_flag = builder.input_arg(I32);                                                   \
                auto channel = builder.input_arg(I32);                                                       \
                BuildSalesBody(builder, quantity, price, price_valid, true, discount, return_flag, channel); \
            });                                                                                              \
    }

#define BUILD_SALES_SET(PREFIX, QWRAP, QDTYPE, DTYPE, NUMNAME)                                                       \
    BUILD_SALES_FLAT_NN(Build##PREFIX##FlatColumn, PREFIX##FlatKernel, QWRAP, QDTYPE, simjit::jit::InputArr<DTYPE>,  \
                        DTYPE, input_arg)                                                                            \
    BUILD_SALES_FLAT_NN(Build##PREFIX##FlatConst, PREFIX##FlatKernel, QWRAP, QDTYPE, simjit::jit::InputConst<DTYPE>, \
                        DTYPE, input_splat_arg)                                                                      \
    BUILD_SALES_FLAT_NULLABLE(Build##PREFIX##NullableFlatColumn, PREFIX##NullableFlatKernel, QWRAP, QDTYPE,          \
                              simjit::jit::InputArr<DTYPE>, DTYPE, input_arg)                                        \
    BUILD_SALES_FLAT_NULLABLE(Build##PREFIX##NullableFlatConst, PREFIX##NullableFlatKernel, QWRAP, QDTYPE,           \
                              simjit::jit::InputConst<DTYPE>, DTYPE, input_splat_arg)                                \
    BUILD_SALES_GATHER_NN(Build##PREFIX##GatherColumn, PREFIX##GatherKernel, QWRAP, QDTYPE,                          \
                          simjit::jit::InputArr<DTYPE>, DTYPE, input_arg)                                            \
    BUILD_SALES_GATHER_NN(Build##PREFIX##GatherConst, PREFIX##GatherKernel, QWRAP, QDTYPE,                           \
                          simjit::jit::InputConst<DTYPE>, DTYPE, input_splat_arg)                                    \
    BUILD_SALES_GATHER_NULLABLE(Build##PREFIX##NullableGatherColumn, PREFIX##NullableGatherKernel, QWRAP, QDTYPE,    \
                                simjit::jit::InputArr<DTYPE>, DTYPE, input_arg)                                      \
    BUILD_SALES_GATHER_NULLABLE(Build##PREFIX##NullableGatherConst, PREFIX##NullableGatherKernel, QWRAP, QDTYPE,     \
                                simjit::jit::InputConst<DTYPE>, DTYPE, input_splat_arg)                              \
    BUILD_SALES_SPEC_NN(Build##PREFIX##SpecColumn, PREFIX##SpecKernel, QWRAP, QDTYPE, simjit::jit::InputArr<DTYPE>,  \
                        DTYPE, input_arg)                                                                            \
    BUILD_SALES_SPEC_NN(Build##PREFIX##SpecConst, PREFIX##SpecKernel, QWRAP, QDTYPE, simjit::jit::InputConst<DTYPE>, \
                        DTYPE, input_splat_arg)                                                                      \
    BUILD_SALES_SPEC_NULLABLE(Build##PREFIX##NullableSpecColumn, PREFIX##NullableSpecKernel, QWRAP, QDTYPE,          \
                              simjit::jit::InputArr<DTYPE>, DTYPE, input_arg)                                        \
    BUILD_SALES_SPEC_NULLABLE(Build##PREFIX##NullableSpecConst, PREFIX##NullableSpecKernel, QWRAP, QDTYPE,           \
                              simjit::jit::InputConst<DTYPE>, DTYPE, input_splat_arg)

BUILD_SALES_SET(SalesWide, simjit::jit::InputArr<I64>, I64, I64, "wide_i64")
BUILD_SALES_SET(SalesMixed, simjit::jit::InputArr<I16>, I16, I16, "mixed_narrow")

#undef BUILD_SALES_SET
#undef BUILD_SALES_SPEC_NULLABLE
#undef BUILD_SALES_SPEC_NN
#undef BUILD_SALES_GATHER_NULLABLE
#undef BUILD_SALES_GATHER_NN
#undef BUILD_SALES_FLAT_NULLABLE
#undef BUILD_SALES_FLAT_NN

static uint64_t LastCompileUsByName(const string &name) {
    if (name == "simjit_add_i64_nn") { return AddI64KernelCache().LastCompileUs(); }
    if (name == "simjit_net_i64_nn") { return NetI64KernelCache().LastCompileUs(); }
    if (name == "simjit_ship_filter_i32_nn") { return ShipFilterKernelCache().LastCompileUs(); }
    if (name == "simjit_sum_add_i64_nn") { return SumAddI64KernelCache().LastCompileUs(); }
    if (name == "simjit_sum_net_i64_nn") { return SumNetI64KernelCache().LastCompileUs(); }
    if (name == "simjit_count_ship_filter_i32_nn") { return CountShipFilterKernelCache().LastCompileUs(); }
    if (name == "simjit_q1_multi_i64") { return Q1MultiKernelCache().LastCompileUs(); }
    if (name == "simjit_nullable_revenue_i64") { return NullableRevenueKernelCache().LastCompileUs(); }
    if (name == "simjit_q19_mask_i64") { return Q19MaskKernelCache().LastCompileUs(); }
#define SALES_LAST_COMPILE(CACHE) \
    if (name == CACHE().name) { return CACHE().LastCompileUs(); }
    SALES_LAST_COMPILE(SalesWideFlatColumnCache)
    SALES_LAST_COMPILE(SalesWideFlatConstCache)
    SALES_LAST_COMPILE(SalesWideNullableFlatColumnCache)
    SALES_LAST_COMPILE(SalesWideNullableFlatConstCache)
    SALES_LAST_COMPILE(SalesWideGatherColumnCache)
    SALES_LAST_COMPILE(SalesWideGatherConstCache)
    SALES_LAST_COMPILE(SalesWideNullableGatherColumnCache)
    SALES_LAST_COMPILE(SalesWideNullableGatherConstCache)
    SALES_LAST_COMPILE(SalesWideSpecColumnCache)
    SALES_LAST_COMPILE(SalesWideSpecConstCache)
    SALES_LAST_COMPILE(SalesWideNullableSpecColumnCache)
    SALES_LAST_COMPILE(SalesWideNullableSpecConstCache)
    SALES_LAST_COMPILE(SalesMixedFlatColumnCache)
    SALES_LAST_COMPILE(SalesMixedFlatConstCache)
    SALES_LAST_COMPILE(SalesMixedNullableFlatColumnCache)
    SALES_LAST_COMPILE(SalesMixedNullableFlatConstCache)
    SALES_LAST_COMPILE(SalesMixedGatherColumnCache)
    SALES_LAST_COMPILE(SalesMixedGatherConstCache)
    SALES_LAST_COMPILE(SalesMixedNullableGatherColumnCache)
    SALES_LAST_COMPILE(SalesMixedNullableGatherConstCache)
    SALES_LAST_COMPILE(SalesMixedSpecColumnCache)
    SALES_LAST_COMPILE(SalesMixedSpecConstCache)
    SALES_LAST_COMPILE(SalesMixedNullableSpecColumnCache)
    SALES_LAST_COMPILE(SalesMixedNullableSpecConstCache)
#undef SALES_LAST_COMPILE
    throw InvalidInputException("unknown simjit kernel name: %s", name);
}

static string VectorizerStatusByName(const string &name) {
    if (name == "simjit_add_i64_nn") { return AddI64KernelCache().VectorizerStatus(); }
    if (name == "simjit_net_i64_nn") { return NetI64KernelCache().VectorizerStatus(); }
    if (name == "simjit_ship_filter_i32_nn") { return ShipFilterKernelCache().VectorizerStatus(); }
    if (name == "simjit_sum_add_i64_nn") { return SumAddI64KernelCache().VectorizerStatus(); }
    if (name == "simjit_sum_net_i64_nn") { return SumNetI64KernelCache().VectorizerStatus(); }
    if (name == "simjit_count_ship_filter_i32_nn") { return CountShipFilterKernelCache().VectorizerStatus(); }
    if (name == "simjit_q1_multi_i64") { return Q1MultiKernelCache().VectorizerStatus(); }
    if (name == "simjit_nullable_revenue_i64") { return NullableRevenueKernelCache().VectorizerStatus(); }
    if (name == "simjit_q19_mask_i64") { return Q19MaskKernelCache().VectorizerStatus(); }
#define SALES_VECTORIZER_STATUS(CACHE) \
    if (name == CACHE().name) { return CACHE().VectorizerStatus(); }
    SALES_VECTORIZER_STATUS(SalesWideFlatColumnCache)
    SALES_VECTORIZER_STATUS(SalesWideFlatConstCache)
    SALES_VECTORIZER_STATUS(SalesWideNullableFlatColumnCache)
    SALES_VECTORIZER_STATUS(SalesWideNullableFlatConstCache)
    SALES_VECTORIZER_STATUS(SalesWideGatherColumnCache)
    SALES_VECTORIZER_STATUS(SalesWideGatherConstCache)
    SALES_VECTORIZER_STATUS(SalesWideNullableGatherColumnCache)
    SALES_VECTORIZER_STATUS(SalesWideNullableGatherConstCache)
    SALES_VECTORIZER_STATUS(SalesWideSpecColumnCache)
    SALES_VECTORIZER_STATUS(SalesWideSpecConstCache)
    SALES_VECTORIZER_STATUS(SalesWideNullableSpecColumnCache)
    SALES_VECTORIZER_STATUS(SalesWideNullableSpecConstCache)
    SALES_VECTORIZER_STATUS(SalesMixedFlatColumnCache)
    SALES_VECTORIZER_STATUS(SalesMixedFlatConstCache)
    SALES_VECTORIZER_STATUS(SalesMixedNullableFlatColumnCache)
    SALES_VECTORIZER_STATUS(SalesMixedNullableFlatConstCache)
    SALES_VECTORIZER_STATUS(SalesMixedGatherColumnCache)
    SALES_VECTORIZER_STATUS(SalesMixedGatherConstCache)
    SALES_VECTORIZER_STATUS(SalesMixedNullableGatherColumnCache)
    SALES_VECTORIZER_STATUS(SalesMixedNullableGatherConstCache)
    SALES_VECTORIZER_STATUS(SalesMixedSpecColumnCache)
    SALES_VECTORIZER_STATUS(SalesMixedSpecConstCache)
    SALES_VECTORIZER_STATUS(SalesMixedNullableSpecColumnCache)
    SALES_VECTORIZER_STATUS(SalesMixedNullableSpecConstCache)
#undef SALES_VECTORIZER_STATUS
    throw InvalidInputException("unknown simjit kernel name: %s", name);
}

static void ResetAllKernels() {
    kernel_generation.fetch_add(1, std::memory_order_relaxed);
}

static void RequireNoNulls(Vector &input, const char *function_name) {
    if (input.GetVectorType() == VectorType::CONSTANT_VECTOR) {
        if (ConstantVector::IsNull(input)) {
            throw InvalidInputException("%s does not support NULL inputs", function_name);
        }
        return;
    }
    if (!FlatVector::Validity(input).CannotHaveNull()) {
        throw InvalidInputException("%s does not support NULL inputs", function_name);
    }
}

static void SimjitHelloFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    PrintHello();
    result.Reference(Value(SIMJIT_HELLO_MESSAGE), count_t(args.size()));
}

static void SimjitResetKernelsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    ResetAllKernels();
    result.Reference(Value::BOOLEAN(true), count_t(args.size()));
}

static void SimjitLastCompileUsFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    args.Flatten();
    auto count = args.size();
    auto names = FlatVector::GetData<string_t>(args.data[0]);
    auto output = FlatVector::GetDataMutable<int64_t>(result);
    for (idx_t i = 0; i < count; ++i) {
        output[i] = static_cast<int64_t>(LastCompileUsByName(names[i].GetString()));
    }
}

static void SimjitVectorizerStatusFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    args.Flatten();
    auto count = args.size();
    auto names = FlatVector::GetData<string_t>(args.data[0]);
    auto output = FlatVector::GetDataMutable<string_t>(result);
    for (idx_t i = 0; i < count; ++i) {
        output[i] = StringVector::AddString(result, VectorizerStatusByName(names[i].GetString()));
    }
}

static void SimjitAddI64Function(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    args.Flatten();
    RequireNoNulls(args.data[0], "simjit_add_i64_nn");
    RequireNoNulls(args.data[1], "simjit_add_i64_nn");
    result.SetVectorType(VectorType::FLAT_VECTOR);
    AddI64KernelCache().Get(BuildAddI64)(args.size(), FlatVector::GetData<int64_t>(args.data[0]),
                                         FlatVector::GetData<int64_t>(args.data[1]),
                                         FlatVector::GetDataMutable<int64_t>(result));
}

static void SimjitNetI64Function(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    args.Flatten();
    RequireNoNulls(args.data[0], "simjit_net_i64_nn");
    RequireNoNulls(args.data[1], "simjit_net_i64_nn");
    result.SetVectorType(VectorType::FLAT_VECTOR);
    NetI64KernelCache().Get(BuildNetI64)(args.size(), FlatVector::GetData<int64_t>(args.data[0]),
                                         FlatVector::GetData<int64_t>(args.data[1]),
                                         FlatVector::GetDataMutable<int64_t>(result));
}

static void SimjitShipFilterFunction(DataChunk &args, ExpressionState &state, Vector &result) {
    (void)state;
    args.Flatten();
    RequireNoNulls(args.data[0], "simjit_ship_filter_i32_nn");
    RequireNoNulls(args.data[1], "simjit_ship_filter_i32_nn");
    result.SetVectorType(VectorType::FLAT_VECTOR);
    ShipFilterKernelCache().Get(BuildShipFilter)(args.size(), FlatVector::GetData<int32_t>(args.data[0]),
                                                 FlatVector::GetData<int32_t>(args.data[1]),
                                                 FlatVector::GetDataMutable<bool>(result));
}

static void SimjitIdentityDictionaryI8(DataChunk &args, ExpressionState &state, Vector &result, const string &dict_id) {
    (void)state;
    args.Flatten();
    RequireNoNulls(args.data[0], dict_id.c_str());
    constexpr idx_t DICTIONARY_SIZE = 128;
    const auto count = args.size();
    auto dict = DictionaryVector::CreateReusableDictionary(LogicalType::TINYINT, DICTIONARY_SIZE);
    auto child = FlatVector::GetDataMutable<int8_t>(dict->data);
    for (idx_t i = 0; i < DICTIONARY_SIZE; ++i) {
        child[i] = UnsafeNumericCast<int8_t>(i);
    }
    dict->id = dict_id;

    SelectionVector sel(count);
    auto input = FlatVector::GetData<int8_t>(args.data[0]);
    for (idx_t i = 0; i < count; ++i) {
        const auto code = static_cast<int>(input[i]);
        if (code < 0 || code >= static_cast<int>(DICTIONARY_SIZE)) {
            throw InvalidInputException("%s expects non-negative tinyint dictionary codes below 128", dict_id.c_str());
        }
        sel.set_index(i, UnsafeNumericCast<idx_t>(code));
    }
    result.Dictionary(dict, sel, count);
}

static void SimjitDictGatherI8Function(DataChunk &args, ExpressionState &state, Vector &result) {
    SimjitIdentityDictionaryI8(args, state, result, "simjit_dict_gather_i8");
}

static void SimjitDictSpecI8Function(DataChunk &args, ExpressionState &state, Vector &result) {
    SimjitIdentityDictionaryI8(args, state, result, "simjit_dict_spec_i8");
}

static bool TryAddI64(int64_t left, int64_t right, int64_t &result) {
    return !__builtin_add_overflow(left, right, &result);
}

static unique_ptr<BaseStatistics> SimjitAddI64Stats(ClientContext &context, FunctionStatisticsInput &input) {
    (void)context;
    if (input.child_stats.size() != 2) { return nullptr; }

    auto &left_stats = input.child_stats[0];
    auto &right_stats = input.child_stats[1];
    if (!NumericStats::HasMinMax(left_stats) || !NumericStats::HasMinMax(right_stats)) { return nullptr; }

    int64_t min_value;
    int64_t max_value;
    if (!TryAddI64(NumericStats::GetMin<int64_t>(left_stats), NumericStats::GetMin<int64_t>(right_stats), min_value) ||
        !TryAddI64(NumericStats::GetMax<int64_t>(left_stats), NumericStats::GetMax<int64_t>(right_stats), max_value)) {
        return nullptr;
    }

    auto result = NumericStats::CreateEmpty(input.expr.GetReturnType());
    NumericStats::SetMin(result, Value::BIGINT(min_value));
    NumericStats::SetMax(result, Value::BIGINT(max_value));
    result.CombineValidity(left_stats, right_stats);
    return result.ToUnique();
}

struct Q1State {
    int64_t sum_net = 0;
    int64_t sum_charge = 0;
    int64_t row_count = 0;
};

struct I64SumState {
    int64_t sum = 0;
    int64_t row_count = 0;
};

struct I64CountState {
    int64_t count = 0;
};

struct NullableRevenueState {
    int64_t sum_valid_net = 0;
    int64_t sum_delay_net = 0;
    int64_t count_valid = 0;
    int64_t count_delay = 0;
};

struct Q19State {
    int64_t row_count = 0;
    int64_t revenue = 0;
};

struct SalesState {
    int64_t sum_revenue = 0;
    int64_t channel2_revenue = 0;
    int64_t bulk_count = 0;
    int64_t revenue_rows = 0;
    int64_t channel2_rows = 0;
};

template <class STATE> static idx_t StateSize(const BoundAggregateFunction &function) {
    (void)function;
    return sizeof(STATE);
}

template <class STATE> static void StateInitialize(const BoundAggregateFunction &function, data_ptr_t state) {
    (void)function;
    new (state) STATE();
}

template <class STATE> static STATE *GetStatePtr(Vector &states, UnifiedVectorFormat &format, idx_t row) {
    auto data = UnifiedVectorFormat::GetData<STATE *>(format);
    return data[format.sel->get_index(row)];
}

template <class STATE> static STATE &GetSingleUngroupedState(Vector &states, idx_t count) {
    (void)count;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    return *GetStatePtr<STATE>(states, state_format, 0);
}

template <class STATE, class MERGE>
static void CombineStates(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count,
                          MERGE merge) {
    (void)aggr_input_data;
    UnifiedVectorFormat source_format;
    UnifiedVectorFormat target_format;
    source.ToUnifiedFormat(source_format);
    target.ToUnifiedFormat(target_format);
    for (idx_t i = 0; i < count; ++i) {
        merge(*GetStatePtr<STATE>(source, source_format, i), *GetStatePtr<STATE>(target, target_format, i));
    }
}

static const uint64_t *AllValidMaskData(idx_t count) {
    static thread_local const std::array<uint64_t, SIMJIT_ALL_VALID_MASK_ENTRIES> all_valid = []() {
        std::array<uint64_t, SIMJIT_ALL_VALID_MASK_ENTRIES> result{};
        result.fill(~uint64_t(0));
        return result;
    }();
    if (count <= SIMJIT_DUCKDB_VECTOR_SIZE) { return all_valid.data(); }
    thread_local std::vector<uint64_t> oversized_all_valid;
    oversized_all_valid.assign(ValidityMask::EntryCount(count), ~uint64_t(0));
    return oversized_all_valid.data();
}

static const uint64_t *ValidityData(Vector &input, idx_t count) {
    auto &validity = FlatVector::Validity(input);
    if (validity.CannotHaveNull()) { return AllValidMaskData(count); }
    return validity.GetData();
}

static LogicalType Q1ReturnType() {
    child_list_t<LogicalType> children;
    children.emplace_back("sum_net", LogicalType::HUGEINT);
    children.emplace_back("sum_charge", LogicalType::HUGEINT);
    children.emplace_back("row_count", LogicalType::BIGINT);
    return LogicalType::STRUCT(std::move(children));
}

static LogicalType NullableRevenueReturnType() {
    child_list_t<LogicalType> children;
    children.emplace_back("sum_valid_net", LogicalType::HUGEINT);
    children.emplace_back("sum_delay_net", LogicalType::HUGEINT);
    children.emplace_back("count_valid", LogicalType::BIGINT);
    return LogicalType::STRUCT(std::move(children));
}

static LogicalType Q19ReturnType() {
    child_list_t<LogicalType> children;
    children.emplace_back("row_count", LogicalType::BIGINT);
    children.emplace_back("revenue", LogicalType::HUGEINT);
    return LogicalType::STRUCT(std::move(children));
}

static LogicalType SalesReturnType() {
    child_list_t<LogicalType> children;
    children.emplace_back("sum_revenue", LogicalType::HUGEINT);
    children.emplace_back("channel2_revenue", LogicalType::HUGEINT);
    children.emplace_back("bulk_count", LogicalType::BIGINT);
    return LogicalType::STRUCT(std::move(children));
}

static Value HugeintValueOrNull(int64_t value, bool has_value) {
    if (!has_value) { return Value(LogicalType::HUGEINT); }
    return Value::HUGEINT(Hugeint::Convert(value));
}

static void SumAddUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                         idx_t count) {
    (void)aggr_input_data;
    if (input_count != 2 || count == 0) { return; }
    for (idx_t i = 0; i < input_count; ++i) {
        inputs[i].Flatten();
        RequireNoNulls(inputs[i], "simjit_sum_add_i64_nn");
    }
    auto &state = GetSingleUngroupedState<I64SumState>(states, count);
    int64_t sum = 0;
    SumAddI64KernelCache().Get(BuildSumAddI64)(count, FlatVector::GetData<int64_t>(inputs[0]),
                                               FlatVector::GetData<int64_t>(inputs[1]), &sum);
    state.sum += sum;
    state.row_count += UnsafeNumericCast<int64_t>(count);
}

static void SumNetUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                         idx_t count) {
    (void)aggr_input_data;
    if (input_count != 2 || count == 0) { return; }
    for (idx_t i = 0; i < input_count; ++i) {
        inputs[i].Flatten();
        RequireNoNulls(inputs[i], "simjit_sum_net_i64_nn");
    }
    auto &state = GetSingleUngroupedState<I64SumState>(states, count);
    int64_t sum = 0;
    SumNetI64KernelCache().Get(BuildSumNetI64)(count, FlatVector::GetData<int64_t>(inputs[0]),
                                               FlatVector::GetData<int64_t>(inputs[1]), &sum);
    state.sum += sum;
    state.row_count += UnsafeNumericCast<int64_t>(count);
}

static void SumI64Combine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
    CombineStates<I64SumState>(source, target, aggr_input_data, count,
                               [](const I64SumState &source, I64SumState &target) {
                                   target.sum += source.sum;
                                   target.row_count += source.row_count;
                               });
}

static void SumI64Finalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                           idx_t offset) {
    (void)aggr_input_data;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    auto output = FlatVector::GetDataMutable<hugeint_t>(result);
    for (idx_t i = 0; i < count; ++i) {
        auto &state = *GetStatePtr<I64SumState>(states, state_format, i);
        if (state.row_count == 0) {
            FlatVector::SetNull(result, offset + i, true);
        } else {
            output[offset + i] = Hugeint::Convert(state.sum);
        }
    }
}

static void CountShipFilterUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &states, idx_t count) {
    (void)aggr_input_data;
    if (input_count != 2 || count == 0) { return; }
    for (idx_t i = 0; i < input_count; ++i) {
        inputs[i].Flatten();
        RequireNoNulls(inputs[i], "simjit_count_ship_filter_i32_nn");
    }
    auto &state = GetSingleUngroupedState<I64CountState>(states, count);
    int64_t matched = 0;
    CountShipFilterKernelCache().Get(BuildCountShipFilter)(count, FlatVector::GetData<int32_t>(inputs[0]),
                                                           FlatVector::GetData<int32_t>(inputs[1]), &matched);
    state.count += matched;
}

static void CountI64Combine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
    CombineStates<I64CountState>(
        source, target, aggr_input_data, count,
        [](const I64CountState &source, I64CountState &target) { target.count += source.count; });
}

static void CountI64Finalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                             idx_t offset) {
    (void)aggr_input_data;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    auto output = FlatVector::GetDataMutable<int64_t>(result);
    for (idx_t i = 0; i < count; ++i) {
        output[offset + i] = GetStatePtr<I64CountState>(states, state_format, i)->count;
    }
}

static void Q1Update(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                     idx_t count) {
    (void)aggr_input_data;
    if (input_count != 3 || count == 0) { return; }
    for (idx_t i = 0; i < input_count; ++i) {
        inputs[i].Flatten();
        RequireNoNulls(inputs[i], "simjit_q1_multi_i64");
    }
    auto &state = GetSingleUngroupedState<Q1State>(states, count);
    int64_t sum_net = 0;
    int64_t sum_charge = 0;
    int64_t row_count = 0;
    Q1MultiKernelCache().Get(BuildQ1Multi)(count, FlatVector::GetData<int64_t>(inputs[0]),
                                           FlatVector::GetData<int64_t>(inputs[1]),
                                           FlatVector::GetData<int64_t>(inputs[2]), &sum_net, &sum_charge, &row_count);
    state.sum_net += sum_net;
    state.sum_charge += sum_charge;
    state.row_count += row_count;
}

static void Q1Combine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
    CombineStates<Q1State>(source, target, aggr_input_data, count, [](const Q1State &source, Q1State &target) {
        target.sum_net += source.sum_net;
        target.sum_charge += source.sum_charge;
        target.row_count += source.row_count;
    });
}

static void Q1Finalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count, idx_t offset) {
    (void)aggr_input_data;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    for (idx_t i = 0; i < count; ++i) {
        auto &state = *GetStatePtr<Q1State>(states, state_format, i);
        result.SetValue(offset + i,
                        Value::STRUCT({{"sum_net", HugeintValueOrNull(state.sum_net, state.row_count > 0)},
                                       {"sum_charge", HugeintValueOrNull(state.sum_charge, state.row_count > 0)},
                                       {"row_count", Value::BIGINT(state.row_count)}}));
    }
}

static void NullableRevenueUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count,
                                  Vector &states, idx_t count) {
    (void)aggr_input_data;
    if (input_count != 4 || count == 0) { return; }
    for (idx_t i = 0; i < input_count; ++i) {
        inputs[i].Flatten();
    }
    RequireNoNulls(inputs[3], "simjit_nullable_revenue_i64 delay");

    auto &state = GetSingleUngroupedState<NullableRevenueState>(states, count);
    int64_t sum_valid_net = 0;
    int64_t sum_delay_net = 0;
    int64_t count_valid = 0;
    int64_t count_delay = 0;
    const auto *quantity_validity = reinterpret_cast<const simjit::jit::Bitmask *>(ValidityData(inputs[0], count));
    const auto *price_validity = reinterpret_cast<const simjit::jit::Bitmask *>(ValidityData(inputs[1], count));
    const auto *discount_validity = reinterpret_cast<const simjit::jit::Bitmask *>(ValidityData(inputs[2], count));

    NullableRevenueKernelCache().Get(BuildNullableRevenue)(
        count, FlatVector::GetData<int64_t>(inputs[0]), FlatVector::GetData<int64_t>(inputs[1]),
        FlatVector::GetData<int64_t>(inputs[2]), FlatVector::GetData<int64_t>(inputs[3]), quantity_validity,
        price_validity, discount_validity, &sum_valid_net, &sum_delay_net, &count_valid, &count_delay);
    state.sum_valid_net += sum_valid_net;
    state.sum_delay_net += sum_delay_net;
    state.count_valid += count_valid;
    state.count_delay += count_delay;
}

static void NullableRevenueCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
    CombineStates<NullableRevenueState>(source, target, aggr_input_data, count,
                                        [](const NullableRevenueState &source, NullableRevenueState &target) {
                                            target.sum_valid_net += source.sum_valid_net;
                                            target.sum_delay_net += source.sum_delay_net;
                                            target.count_valid += source.count_valid;
                                            target.count_delay += source.count_delay;
                                        });
}

static void NullableRevenueFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                                    idx_t offset) {
    (void)aggr_input_data;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    for (idx_t i = 0; i < count; ++i) {
        auto &state = *GetStatePtr<NullableRevenueState>(states, state_format, i);
        result.SetValue(
            offset + i,
            Value::STRUCT({{"sum_valid_net", HugeintValueOrNull(state.sum_valid_net, state.count_valid > 0)},
                           {"sum_delay_net", HugeintValueOrNull(state.sum_delay_net, state.count_delay > 0)},
                           {"count_valid", Value::BIGINT(state.count_valid)}}));
    }
}

static void Q19Update(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                      idx_t count) {
    (void)aggr_input_data;
    if (input_count != 6 || count == 0) { return; }
    for (idx_t i = 0; i < input_count; ++i) {
        inputs[i].Flatten();
        RequireNoNulls(inputs[i], "simjit_q19_mask_i64");
    }
    auto &state = GetSingleUngroupedState<Q19State>(states, count);
    int64_t row_count = 0;
    int64_t revenue = 0;
    Q19MaskKernelCache().Get(BuildQ19Mask)(
        count, FlatVector::GetData<int64_t>(inputs[0]), FlatVector::GetData<int64_t>(inputs[1]),
        FlatVector::GetData<int64_t>(inputs[2]), FlatVector::GetData<int64_t>(inputs[3]),
        FlatVector::GetData<int64_t>(inputs[4]), FlatVector::GetData<int64_t>(inputs[5]), &row_count, &revenue);
    state.row_count += row_count;
    state.revenue += revenue;
}

static void Q19Combine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
    CombineStates<Q19State>(source, target, aggr_input_data, count, [](const Q19State &source, Q19State &target) {
        target.row_count += source.row_count;
        target.revenue += source.revenue;
    });
}

static void Q19Finalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                        idx_t offset) {
    (void)aggr_input_data;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    for (idx_t i = 0; i < count; ++i) {
        auto &state = *GetStatePtr<Q19State>(states, state_format, i);
        result.SetValue(offset + i,
                        Value::STRUCT({{"row_count", Value::BIGINT(state.row_count)},
                                       {"revenue", HugeintValueOrNull(state.revenue, state.row_count > 0)}}));
    }
}

enum class SalesCatMode {
    Flat,
    DictGather,
    DictSpec
};

static SalesCatMode DetectSalesCatMode(Vector &return_flag, Vector &channel) {
    const auto return_type = return_flag.GetVectorType();
    const auto channel_type = channel.GetVectorType();
    if (return_type == VectorType::DICTIONARY_VECTOR && channel_type == VectorType::DICTIONARY_VECTOR) {
        const auto &return_id = DictionaryVector::DictionaryId(return_flag);
        const auto &channel_id = DictionaryVector::DictionaryId(channel);
        if (return_id == "simjit_dict_spec_i8" && channel_id == "simjit_dict_spec_i8") {
            return SalesCatMode::DictSpec;
        }
        return SalesCatMode::DictGather;
    }
    if (return_type == VectorType::DICTIONARY_VECTOR || channel_type == VectorType::DICTIONARY_VECTOR) {
        throw InvalidInputException("sales categorical inputs must both be dictionary vectors or both be flat");
    }
    return SalesCatMode::Flat;
}

static const int32_t *DictionarySelI32(Vector &input) {
    return reinterpret_cast<const int32_t *>(DictionaryVector::SelVector(input).data());
}

static const int8_t *DictionaryChildI8(Vector &input, const char *name) {
    auto &child = DictionaryVector::Child(input);
    RequireNoNulls(child, name);
    return FlatVector::GetData<int8_t>(child);
}

template <class Q, class D, bool MIXED_NARROW>
static void SalesUpdateCommon(Vector inputs[], idx_t input_count, Vector &states, idx_t count) {
    if (input_count != 5 || count == 0) { return; }

    auto &quantity = inputs[0];
    auto &price = inputs[1];
    auto &discount = inputs[2];
    auto &return_flag = inputs[3];
    auto &channel = inputs[4];
    const auto cat_mode = DetectSalesCatMode(return_flag, channel);
    const bool const_discount = discount.GetVectorType() == VectorType::CONSTANT_VECTOR;

    quantity.Flatten();
    price.Flatten();
    if (!const_discount) { discount.Flatten(); }
    RequireNoNulls(quantity, MIXED_NARROW ? "simjit_sales_mixed_narrow quantity" : "simjit_sales_wide_i64 quantity");
    RequireNoNulls(discount, MIXED_NARROW ? "simjit_sales_mixed_narrow discount" : "simjit_sales_wide_i64 discount");
    const bool nullable_price = !FlatVector::Validity(price).CannotHaveNull();

    auto &state = GetSingleUngroupedState<SalesState>(states, count);
    int64_t sum_revenue = 0;
    int64_t channel2_revenue = 0;
    int64_t bulk_count = 0;
    int64_t revenue_rows = 0;
    int64_t channel2_rows = 0;
    auto quantity_data = FlatVector::GetData<Q>(quantity);
    auto price_data = FlatVector::GetData<int64_t>(price);
    auto discount_data = const_discount ? ConstantVector::GetData<D>(discount) : FlatVector::GetData<D>(discount);
    auto price_validity = reinterpret_cast<const simjit::jit::Bitmask *>(ValidityData(price, count));

#define SALES_ACCUMULATE()                      \
    state.sum_revenue += sum_revenue;           \
    state.channel2_revenue += channel2_revenue; \
    state.bulk_count += bulk_count;             \
    state.revenue_rows += revenue_rows;         \
    state.channel2_rows += channel2_rows

    if (cat_mode == SalesCatMode::Flat) {
        return_flag.Flatten();
        channel.Flatten();
        RequireNoNulls(return_flag, "simjit sales return_flag");
        RequireNoNulls(channel, "simjit sales channel");
        auto return_data = FlatVector::GetData<int8_t>(return_flag);
        auto channel_data = FlatVector::GetData<int8_t>(channel);
        if constexpr (MIXED_NARROW) {
            if (nullable_price) {
                if (const_discount) {
                    SalesMixedNullableFlatConstCache().Get(BuildSalesMixedNullableFlatConst)(
                        count, quantity_data, price_data, price_validity, discount_data, return_data, channel_data,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                } else {
                    SalesMixedNullableFlatColumnCache().Get(BuildSalesMixedNullableFlatColumn)(
                        count, quantity_data, price_data, price_validity, discount_data, return_data, channel_data,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                }
            } else if (const_discount) {
                SalesMixedFlatConstCache().Get(BuildSalesMixedFlatConst)(
                    count, quantity_data, price_data, discount_data, return_data, channel_data, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            } else {
                SalesMixedFlatColumnCache().Get(BuildSalesMixedFlatColumn)(
                    count, quantity_data, price_data, discount_data, return_data, channel_data, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            }
        } else {
            if (nullable_price) {
                if (const_discount) {
                    SalesWideNullableFlatConstCache().Get(BuildSalesWideNullableFlatConst)(
                        count, quantity_data, price_data, price_validity, discount_data, return_data, channel_data,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                } else {
                    SalesWideNullableFlatColumnCache().Get(BuildSalesWideNullableFlatColumn)(
                        count, quantity_data, price_data, price_validity, discount_data, return_data, channel_data,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                }
            } else if (const_discount) {
                SalesWideFlatConstCache().Get(BuildSalesWideFlatConst)(
                    count, quantity_data, price_data, discount_data, return_data, channel_data, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            } else {
                SalesWideFlatColumnCache().Get(BuildSalesWideFlatColumn)(
                    count, quantity_data, price_data, discount_data, return_data, channel_data, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            }
        }
        SALES_ACCUMULATE();
        return;
    }

    auto return_child = DictionaryChildI8(return_flag, "simjit sales return_flag dictionary");
    auto channel_child = DictionaryChildI8(channel, "simjit sales channel dictionary");
    auto return_idx = DictionarySelI32(return_flag);
    auto channel_idx = DictionarySelI32(channel);
    if (cat_mode == SalesCatMode::DictSpec) {
        if constexpr (MIXED_NARROW) {
            if (nullable_price) {
                if (const_discount) {
                    SalesMixedNullableSpecConstCache().Get(BuildSalesMixedNullableSpecConst)(
                        count, quantity_data, price_data, price_validity, discount_data, return_idx, channel_idx,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                } else {
                    SalesMixedNullableSpecColumnCache().Get(BuildSalesMixedNullableSpecColumn)(
                        count, quantity_data, price_data, price_validity, discount_data, return_idx, channel_idx,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                }
            } else if (const_discount) {
                SalesMixedSpecConstCache().Get(BuildSalesMixedSpecConst)(
                    count, quantity_data, price_data, discount_data, return_idx, channel_idx, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            } else {
                SalesMixedSpecColumnCache().Get(BuildSalesMixedSpecColumn)(
                    count, quantity_data, price_data, discount_data, return_idx, channel_idx, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            }
        } else {
            if (nullable_price) {
                if (const_discount) {
                    SalesWideNullableSpecConstCache().Get(BuildSalesWideNullableSpecConst)(
                        count, quantity_data, price_data, price_validity, discount_data, return_idx, channel_idx,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                } else {
                    SalesWideNullableSpecColumnCache().Get(BuildSalesWideNullableSpecColumn)(
                        count, quantity_data, price_data, price_validity, discount_data, return_idx, channel_idx,
                        &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
                }
            } else if (const_discount) {
                SalesWideSpecConstCache().Get(BuildSalesWideSpecConst)(
                    count, quantity_data, price_data, discount_data, return_idx, channel_idx, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            } else {
                SalesWideSpecColumnCache().Get(BuildSalesWideSpecColumn)(
                    count, quantity_data, price_data, discount_data, return_idx, channel_idx, &sum_revenue,
                    &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
            }
        }
        SALES_ACCUMULATE();
        return;
    }

    if constexpr (MIXED_NARROW) {
        if (nullable_price) {
            if (const_discount) {
                SalesMixedNullableGatherConstCache().Get(BuildSalesMixedNullableGatherConst)(
                    count, quantity_data, price_data, price_validity, discount_data, return_child, return_idx,
                    channel_child, channel_idx, &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows,
                    &channel2_rows);
            } else {
                SalesMixedNullableGatherColumnCache().Get(BuildSalesMixedNullableGatherColumn)(
                    count, quantity_data, price_data, price_validity, discount_data, return_child, return_idx,
                    channel_child, channel_idx, &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows,
                    &channel2_rows);
            }
        } else if (const_discount) {
            SalesMixedGatherConstCache().Get(BuildSalesMixedGatherConst)(
                count, quantity_data, price_data, discount_data, return_child, return_idx, channel_child, channel_idx,
                &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
        } else {
            SalesMixedGatherColumnCache().Get(BuildSalesMixedGatherColumn)(
                count, quantity_data, price_data, discount_data, return_child, return_idx, channel_child, channel_idx,
                &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
        }
    } else {
        if (nullable_price) {
            if (const_discount) {
                SalesWideNullableGatherConstCache().Get(BuildSalesWideNullableGatherConst)(
                    count, quantity_data, price_data, price_validity, discount_data, return_child, return_idx,
                    channel_child, channel_idx, &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows,
                    &channel2_rows);
            } else {
                SalesWideNullableGatherColumnCache().Get(BuildSalesWideNullableGatherColumn)(
                    count, quantity_data, price_data, price_validity, discount_data, return_child, return_idx,
                    channel_child, channel_idx, &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows,
                    &channel2_rows);
            }
        } else if (const_discount) {
            SalesWideGatherConstCache().Get(BuildSalesWideGatherConst)(
                count, quantity_data, price_data, discount_data, return_child, return_idx, channel_child, channel_idx,
                &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
        } else {
            SalesWideGatherColumnCache().Get(BuildSalesWideGatherColumn)(
                count, quantity_data, price_data, discount_data, return_child, return_idx, channel_child, channel_idx,
                &sum_revenue, &channel2_revenue, &bulk_count, &revenue_rows, &channel2_rows);
        }
    }
    SALES_ACCUMULATE();
#undef SALES_ACCUMULATE
}

static void SalesWideUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                            idx_t count) {
    (void)aggr_input_data;
    SalesUpdateCommon<int64_t, int64_t, false>(inputs, input_count, states, count);
}

static void SalesMixedUpdate(Vector inputs[], AggregateInputData &aggr_input_data, idx_t input_count, Vector &states,
                             idx_t count) {
    (void)aggr_input_data;
    SalesUpdateCommon<int16_t, int16_t, true>(inputs, input_count, states, count);
}

static void SalesCombine(Vector &source, Vector &target, AggregateInputData &aggr_input_data, idx_t count) {
    CombineStates<SalesState>(source, target, aggr_input_data, count, [](const SalesState &source, SalesState &target) {
        target.sum_revenue += source.sum_revenue;
        target.channel2_revenue += source.channel2_revenue;
        target.bulk_count += source.bulk_count;
        target.revenue_rows += source.revenue_rows;
        target.channel2_rows += source.channel2_rows;
    });
}

static void SalesFinalize(Vector &states, AggregateInputData &aggr_input_data, Vector &result, idx_t count,
                          idx_t offset) {
    (void)aggr_input_data;
    UnifiedVectorFormat state_format;
    states.ToUnifiedFormat(state_format);
    for (idx_t i = 0; i < count; ++i) {
        auto &state = *GetStatePtr<SalesState>(states, state_format, i);
        result.SetValue(
            offset + i,
            Value::STRUCT({{"sum_revenue", HugeintValueOrNull(state.sum_revenue, state.revenue_rows > 0)},
                           {"channel2_revenue", HugeintValueOrNull(state.channel2_revenue, state.channel2_rows > 0)},
                           {"bulk_count", Value::BIGINT(state.bulk_count)}}));
    }
}

static void RegisterScalarFunctions(ExtensionLoader &loader) {
    PrintHello();
    loader.RegisterFunction(ScalarFunction("simjit_hello", {}, LogicalType::VARCHAR, SimjitHelloFunction));
    loader.RegisterFunction(
        ScalarFunction("simjit_reset_kernels", {}, LogicalType::BOOLEAN, SimjitResetKernelsFunction));
    loader.RegisterFunction(ScalarFunction("simjit_last_compile_us", {LogicalType::VARCHAR}, LogicalType::BIGINT,
                                           SimjitLastCompileUsFunction));
    loader.RegisterFunction(ScalarFunction("simjit_vectorizer_status", {LogicalType::VARCHAR}, LogicalType::VARCHAR,
                                           SimjitVectorizerStatusFunction));

    ScalarFunction add_function("simjit_add_i64_nn", {LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::BIGINT,
                                SimjitAddI64Function);
    add_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
    add_function.SetStatisticsCallback(SimjitAddI64Stats);
    loader.RegisterFunction(add_function);

    ScalarFunction net_function("simjit_net_i64_nn", {LogicalType::BIGINT, LogicalType::BIGINT}, LogicalType::BIGINT,
                                SimjitNetI64Function);
    net_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
    loader.RegisterFunction(net_function);

    ScalarFunction ship_filter_function("simjit_ship_filter_i32_nn", {LogicalType::INTEGER, LogicalType::INTEGER},
                                        LogicalType::BOOLEAN, SimjitShipFilterFunction);
    ship_filter_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
    loader.RegisterFunction(ship_filter_function);

    ScalarFunction dict_gather_function("simjit_dict_gather_i8", {LogicalType::TINYINT}, LogicalType::TINYINT,
                                        SimjitDictGatherI8Function);
    dict_gather_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
    loader.RegisterFunction(dict_gather_function);

    ScalarFunction dict_spec_function("simjit_dict_spec_i8", {LogicalType::TINYINT}, LogicalType::TINYINT,
                                      SimjitDictSpecI8Function);
    dict_spec_function.SetNullHandling(FunctionNullHandling::SPECIAL_HANDLING);
    loader.RegisterFunction(dict_spec_function);
}

static void RegisterAggregateFunctions(ExtensionLoader &loader) {
    loader.RegisterFunction(AggregateFunction("simjit_sum_add_i64_nn", {LogicalType::BIGINT, LogicalType::BIGINT},
                                              LogicalType::HUGEINT, StateSize<I64SumState>,
                                              StateInitialize<I64SumState>, SumAddUpdate, SumI64Combine, SumI64Finalize,
                                              FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(AggregateFunction("simjit_sum_net_i64_nn", {LogicalType::BIGINT, LogicalType::BIGINT},
                                              LogicalType::HUGEINT, StateSize<I64SumState>,
                                              StateInitialize<I64SumState>, SumNetUpdate, SumI64Combine, SumI64Finalize,
                                              FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(AggregateFunction(
        "simjit_count_ship_filter_i32_nn", {LogicalType::INTEGER, LogicalType::INTEGER}, LogicalType::BIGINT,
        StateSize<I64CountState>, StateInitialize<I64CountState>, CountShipFilterUpdate, CountI64Combine,
        CountI64Finalize, FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(AggregateFunction("simjit_q1_multi_i64",
                                              {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
                                              Q1ReturnType(), StateSize<Q1State>, StateInitialize<Q1State>, Q1Update,
                                              Q1Combine, Q1Finalize, FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(
        AggregateFunction("simjit_nullable_revenue_i64",
                          {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
                          NullableRevenueReturnType(), StateSize<NullableRevenueState>,
                          StateInitialize<NullableRevenueState>, NullableRevenueUpdate, NullableRevenueCombine,
                          NullableRevenueFinalize, FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(AggregateFunction("simjit_q19_mask_i64",
                                              {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT,
                                               LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT},
                                              Q19ReturnType(), StateSize<Q19State>, StateInitialize<Q19State>,
                                              Q19Update, Q19Combine, Q19Finalize,
                                              FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(AggregateFunction(
        "simjit_sales_wide_i64",
        {LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::BIGINT, LogicalType::TINYINT, LogicalType::TINYINT},
        SalesReturnType(), StateSize<SalesState>, StateInitialize<SalesState>, SalesWideUpdate, SalesCombine,
        SalesFinalize, FunctionNullHandling::SPECIAL_HANDLING));

    loader.RegisterFunction(AggregateFunction(
        "simjit_sales_mixed_narrow",
        {LogicalType::SMALLINT, LogicalType::BIGINT, LogicalType::SMALLINT, LogicalType::TINYINT, LogicalType::TINYINT},
        SalesReturnType(), StateSize<SalesState>, StateInitialize<SalesState>, SalesMixedUpdate, SalesCombine,
        SalesFinalize, FunctionNullHandling::SPECIAL_HANDLING));
}

static void LoadInternal(ExtensionLoader &loader) {
    RegisterScalarFunctions(loader);
    RegisterAggregateFunctions(loader);
}

} // namespace
} // namespace duckdb

extern "C" {

DUCKDB_CPP_EXTENSION_ENTRY(simjit, loader) {
    duckdb::LoadInternal(loader);
}
}
