// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/detail/span.h"
#include "simjit/simjit.h"

#include <optional>

namespace simjit {
namespace nullable {

template <typename T> struct NullableValueT {
    T v;
    MaybePredicate null{};

    constexpr NullableValueT() noexcept = default;
    // Allow implicit conversion
    constexpr NullableValueT(T x) noexcept : v(x) {}
    constexpr NullableValueT(T x, MaybePredicate nx) noexcept : v(x), null(nx) {}
    constexpr NullableValueT(T x, Predicate nx) noexcept : v(x), null(nx) {}
};

using NullableValue = NullableValueT<Value>;
using NullablePredicate = NullableValueT<Predicate>;

class NullableBuilder {
public:
    NullableBuilder() = delete;
    explicit NullableBuilder(FunctionBuilder *ll_builder) noexcept : base_(ll_builder) {}

    NullableValue null_value(ScalarDataType dtype);
    NullableValue zeronull(NullableValue arg);

    NullableValue nbool_load(Argument var, Argument null);
    NullablePredicate nbool_load_predicate(Argument var, Argument null);
    NullableValue nbit_load(Argument var, Argument null);
    NullablePredicate nbit_load_predicate(Argument var, Argument null);
    NullableValue nbool_load_ext(Argument var, Argument null, bool true_means_null);
    NullablePredicate nbool_load_predicate_ext(Argument var, Argument null, bool true_means_null);
    NullableValue nbit_load_ext(Argument var, Argument null, bool true_means_null);
    NullablePredicate nbit_load_predicate_ext(Argument var, Argument null, bool true_means_null);
    NullableValue nval_load(Argument var, Value nval);

    NullableValue load_splat(Argument var, Argument null);
    NullablePredicate load_predicate_splat(Argument var, Argument null);
    NullableValue nbit_load_splat_ext(Argument var, Argument null, bool true_means_null);
    NullablePredicate nbit_load_predicate_splat_ext(Argument var, Argument null, bool true_means_null);
    NullableValue nbool_load_splat_ext(Argument var, Argument null, bool true_means_null);
    NullablePredicate nbool_load_predicate_splat_ext(Argument var, Argument null, bool true_means_null);

    NullableValue bit2bool(NullablePredicate arg);
    NullablePredicate bool2bit(NullableValue arg);

    NullableValue nbool_gather(Argument var, Argument null, Value idx);
    NullableValue nbool_gather_ext(Argument var, Argument null, Value idx, bool true_means_null);
    NullableValue nval_gather(Argument var, Value idx, Value nval);

    MaybePredicate combine_selection(MaybePredicate sel1, MaybePredicate sel2);

    NullableValue arith_binary(NullableValue left, NullableValue right, ArithBinaryOp op);
    NullableValue arith_binary_checked(NullableValue left, NullableValue right, ArithBinaryOp op);

    NullableValue add(NullableValue left, NullableValue right);
    NullableValue sub(NullableValue left, NullableValue right);
    NullableValue mul(NullableValue left, NullableValue right);
    NullableValue and_(NullableValue left, NullableValue right);
    NullableValue or_(NullableValue left, NullableValue right);
    NullableValue div(NullableValue left, NullableValue right, bool null_on_invalid_division = false);
    NullableValue mod(NullableValue left, NullableValue right, bool null_on_invalid_division = false);

    NullableValue add_checked(NullableValue left, NullableValue right);
    NullableValue sub_checked(NullableValue left, NullableValue right);
    NullableValue mul_checked(NullableValue left, NullableValue right);

    NullableValue greatest(NullableValue left, NullableValue right);
    NullableValue greatest(nonstd::span<const NullableValue> args);
    NullableValue least(NullableValue left, NullableValue right);
    NullableValue least(nonstd::span<const NullableValue> args);

    NullableValue int_cast(NullableValue arg, ScalarDataType to, IntCastKind kind);
    NullableValue cast(NullableValue arg, ScalarDataType to);
    NullableValue signed_cast(NullableValue arg, ScalarDataType to);
    NullableValue unsigned_cast(NullableValue arg, ScalarDataType to);
    NullableValue trunc(NullableValue arg, ScalarDataType to);
    NullableValue trunc_checked(NullableValue arg, ScalarDataType to);
    NullableValue sext(NullableValue arg, ScalarDataType to);
    NullableValue zext(NullableValue arg, ScalarDataType to);
    NullableValue float_cast(NullableValue arg, ScalarDataType to, bool is_unsigned = false);
    NullableValue negate(NullableValue arg);
    NullableValue negate_checked(NullableValue arg);
    NullableValue arith_unary(NullableValue arg, ArithUnaryOp op);
    NullableValue abs(NullableValue arg);
    NullableValue abs_checked(NullableValue arg);
    NullablePredicate not_(NullablePredicate arg);

    NullablePredicate cmp(NullableValue left, NullableValue right, CmpOp op);
    NullablePredicate cmp_eq(NullableValue left, NullableValue right);
    NullablePredicate cmp_ne(NullableValue left, NullableValue right);
    NullablePredicate cmp_gt(NullableValue left, NullableValue right);
    NullablePredicate cmp_ge(NullableValue left, NullableValue right);
    NullablePredicate cmp_lt(NullableValue left, NullableValue right);
    NullablePredicate cmp_le(NullableValue left, NullableValue right);
    NullablePredicate bit_test(NullableValue left, NullableValue right);
    NullablePredicate bit_testn(NullableValue left, NullableValue right);
    NullablePredicate is_positive(NullableValue arg);
    NullablePredicate is_negative(NullableValue arg);

    NullablePredicate predicate_binary(NullablePredicate left, NullablePredicate right, PredicateBinaryOp op);
    NullablePredicate and_(NullablePredicate left, NullablePredicate right);
    NullablePredicate and_(nonstd::span<const NullablePredicate> args);
    NullablePredicate or_(NullablePredicate left, NullablePredicate right);
    NullablePredicate or_(nonstd::span<const NullablePredicate> args);

    Predicate is_null(NullableValue arg);
    Predicate is_not_null(NullableValue arg);
    NullablePredicate true_unless_null(NullableValue arg);
    Predicate is_true(NullablePredicate arg);
    Predicate is_not_true(NullablePredicate arg);
    Predicate is_false(NullablePredicate arg);
    Predicate is_not_false(NullablePredicate arg);

    Predicate is_distinct(NullableValue left, NullableValue right);
    Predicate is_not_distinct(NullableValue left, NullableValue right);
    NullableValue nullif(NullableValue left, NullableValue right);

    NullableValue if_else_null(NullablePredicate cond, NullableValue truthy);
    NullableValue if_else(NullablePredicate cond, NullableValue truthy, NullableValue falsy);
    NullableValue coalesce(nonstd::span<const NullableValue> args);
    NullableValue case_x_when(NullableValue arg, nonstd::span<const std::pair<NullableValue, NullableValue>> cases,
                              std::optional<NullableValue> default_case = {});
    NullableValue case_when(nonstd::span<const std::pair<NullablePredicate, NullableValue>> cases,
                            std::optional<NullableValue> default_case = {});

    void nbit_store(NullableValue arg, Argument dst, Argument dst_null);
    void nbit_store(NullablePredicate arg, Argument dst, Argument dst_null);
    void nbool_store(NullableValue arg, Argument dst, Argument dst_null);
    void nbool_store(NullablePredicate arg, Argument dst, Argument dst_null);
    void nbit_store_ext(NullableValue arg, Argument dst, Argument dst_null, bool true_means_null);
    void nbit_store_ext(NullablePredicate arg, Argument dst, Argument dst_null, bool true_means_null);
    void nbool_store_ext(NullableValue arg, Argument dst, Argument dst_null, bool true_means_null);
    void nbool_store_ext(NullablePredicate arg, Argument dst, Argument dst_null, bool true_means_null);

    void nval_store(NullableValue arg, Argument dst, Value nval);
    void nbool_scatter(NullableValue arg, Argument dst, Argument dst_null, Value idx);
    void nbool_scatter_ext(NullableValue arg, Argument dst, Argument dst_null, Value idx, bool true_means_null);
    void nval_scatter(NullableValue arg, Argument dst, Value idx, Value nval);

    void arith_agg(NullableValue arg, ArithBinaryOp op, Argument dst);
    void cond_arith_agg(NullableValue arg, NullablePredicate cond, ArithBinaryOp op, Argument dst);
    void sum(NullableValue arg, Argument dst);
    void min_agg(NullableValue arg, Argument dst);
    void max_agg(NullableValue arg, Argument dst);
    void grouped_arith_agg(NullableValue arg, Value idx, ArithBinaryOp op, Argument table);
    void grouped_cond_arith_agg(NullableValue arg, NullablePredicate pr, Value idx, ArithBinaryOp op, Argument table);
    void grouped_sum(NullableValue arg, Value idx, Argument table);
    void grouped_min(NullableValue arg, Value idx, Argument table);
    void grouped_max(NullableValue arg, Value idx, Argument table);
    void countif(NullablePredicate arg, Argument dst);
    void count_notnull(NullableValue arg, Argument dst);
    void has_nulls(NullableValue arg, Argument dst);
    void all(NullablePredicate arg, Argument dst);
    void any(NullablePredicate arg, Argument dst);
    void find_true_indices(NullablePredicate arg, Argument dst, Argument dst_size);
    void update_true_indices(Value idx, NullablePredicate arg, Argument dst, Argument dst_size);
    void find_notnull_indices(NullableValue arg, Argument dst, Argument dst_size);
    void find_null_indices(NullableValue arg, Argument dst, Argument dst_size);

private:
    FunctionBuilder *base_ = nullptr;
};

} // namespace nullable
} // namespace simjit
