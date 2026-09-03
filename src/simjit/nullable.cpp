// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/nullable.h"

#include "simjit/core/expr.h"
#include "simjit/simjit.h"

namespace simjit {
namespace nullable {

#define nullable_invalid_config(...) \
    simjit_exception(ErrorModule::Nullable, ErrorKind::InvalidInput, ErrorSubKind::InvalidConfiguration, __VA_ARGS__)

template <typename T, typename F> static T fold_bin_op(NullableBuilder *builder, nonstd::span<const T> exprs, F fn) {
    SIMJIT_ASSERT(!exprs.empty());
    if (exprs.size() == 1) { return exprs[0]; }
    size_t middle = exprs.size() / 2;
    auto lhs = exprs.subspan(0, middle);
    auto rhs = exprs.subspan(middle, exprs.size() - middle);
    SIMJIT_ASSERT(lhs.size() + rhs.size() == exprs.size());
    T left = fold_bin_op(builder, lhs, fn);
    T right = fold_bin_op(builder, rhs, fn);
    return {fn(left.v, right.v), builder->combine_selection(left.null, right.null)};
}

static NullableValue safe_int_div_mod(NullableBuilder *builder, FunctionBuilder *base, NullableValue left,
                                      NullableValue right, ArithBinaryOp op, bool null_on_invalid_division) {
    ScalarDataType dtype = left.v.dtype();
    SIMJIT_ASSERT(is_simple_int_dtype(dtype));
    SIMJIT_ASSERT(op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod);

    Value value = base->arith_binary(left.v, right.v, op, ArithBinaryOpFlags::SafeDivision);
    MaybePredicate null = builder->combine_selection(left.null, right.null);
    // FIXME: This is very inefficient... But we don't expose the safety check mask otherwise.
    // I honestly just want to remove this thing completely.
    if (null_on_invalid_division) {
        Predicate rhs_zero = base->cmp_eq(right.v, base->con(0, dtype));
        Predicate overflow = base->and_(base->cmp_eq(left.v, base->con_internal(scalar_dtype_min(dtype), dtype)),
                                        base->cmp_eq(right.v, base->con(-1, dtype)));
        Predicate invalid = base->or_(rhs_zero, overflow);
        null = builder->combine_selection(null, invalid);
    }
    return {value, null};
}

static Predicate normalize_null_mask(FunctionBuilder *base, Predicate mask, bool true_means_null) {
    return true_means_null ? mask : base->not_(mask);
}

static Predicate store_null_mask(FunctionBuilder *base, MaybePredicate mask, bool true_means_null) {
    if (mask.is_valid()) { return normalize_null_mask(base, mask.value(), true_means_null); }
    return true_means_null ? base->false_() : base->true_();
}

NullableValue NullableBuilder::null_value(ScalarDataType dtype) {
    return {base_->con(0, dtype), base_->true_()};
}

NullableValue NullableBuilder::zeronull(NullableValue arg) {
    if (arg.null.is_valid()) { return base_->zero_select(arg.v, arg.null.value()); }
    return arg;
}

NullableValue NullableBuilder::nbool_load(Argument var, Argument null) {
    return nbool_load_ext(var, null, true);
}

NullablePredicate NullableBuilder::nbool_load_predicate(Argument var, Argument null) {
    return nbool_load_predicate_ext(var, null, true);
}

NullableValue NullableBuilder::nbit_load(Argument var, Argument null) {
    return nbit_load_ext(var, null, true);
}

NullablePredicate NullableBuilder::nbit_load_predicate(Argument var, Argument null) {
    return nbit_load_predicate_ext(var, null, true);
}

NullableValue NullableBuilder::nbool_load_ext(Argument var, Argument null, bool true_means_null) {
    Value x = base_->load(var);
    Predicate y = base_->bool2bit(base_->load(null));
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullablePredicate NullableBuilder::nbool_load_predicate_ext(Argument var, Argument null, bool true_means_null) {
    Predicate x = base_->load_predicate(var);
    Predicate y = base_->bool2bit(base_->load(null));
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullableValue NullableBuilder::nbit_load_ext(Argument var, Argument null, bool true_means_null) {
    Value x = base_->load(var);
    Predicate y = base_->load_predicate(null);
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullablePredicate NullableBuilder::nbit_load_predicate_ext(Argument var, Argument null, bool true_means_null) {
    Predicate x = base_->load_predicate(var);
    Predicate y = base_->load_predicate(null);
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullableValue NullableBuilder::nval_load(Argument var, Value nval) {
    Value x = base_->load(var);
    Predicate y = base_->cmp_eq(x, nval);
    return {x, y};
}

NullableValue NullableBuilder::load_splat(Argument var, Argument null) {
    return nbit_load_splat_ext(var, null, true);
}

NullablePredicate NullableBuilder::load_predicate_splat(Argument var, Argument null) {
    return nbit_load_predicate_splat_ext(var, null, true);
}

NullableValue NullableBuilder::nbit_load_splat_ext(Argument var, Argument null, bool true_means_null) {
    Value x = base_->load_splat(var);
    Predicate y = base_->load_predicate_splat(null);
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullablePredicate NullableBuilder::nbit_load_predicate_splat_ext(Argument var, Argument null, bool true_means_null) {
    Predicate x = base_->load_predicate_splat(var);
    Predicate y = base_->load_predicate_splat(null);
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullableValue NullableBuilder::nbool_load_splat_ext(Argument var, Argument null, bool true_means_null) {
    Value x = base_->load_splat(var);
    Predicate y = base_->bool2bit(base_->load_splat(null));
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullablePredicate NullableBuilder::nbool_load_predicate_splat_ext(Argument var, Argument null, bool true_means_null) {
    Predicate x = base_->load_predicate_splat(var);
    Predicate y = base_->bool2bit(base_->load_splat(null));
    return {x, normalize_null_mask(base_, y, true_means_null)};
}

NullableValue NullableBuilder::bit2bool(NullablePredicate arg) {
    return {base_->bit2bool(arg.v), arg.null};
}

NullablePredicate NullableBuilder::bool2bit(NullableValue arg) {
    return {base_->bool2bit(arg.v), arg.null};
}

NullableValue NullableBuilder::nbool_gather(Argument var, Argument null, Value idx) {
    return nbool_gather_ext(var, null, idx, true);
}

NullableValue NullableBuilder::nbool_gather_ext(Argument var, Argument null, Value idx, bool true_means_null) {
    Value x = base_->gather(idx, var);
    Value y = base_->gather(idx, null);
    return {x, normalize_null_mask(base_, base_->bool2bit(y), true_means_null)};
}

NullableValue NullableBuilder::nval_gather(Argument var, Value idx, Value nval) {
    Value x = base_->gather(idx, var);
    return {x, base_->cmp_eq(x, nval)};
}

MaybePredicate NullableBuilder::combine_selection(MaybePredicate sel1, MaybePredicate sel2) {
    if (sel1.is_valid() && sel2.is_valid()) { return base_->or_(sel1.value(), sel2.value()); }
    if (sel1.is_valid()) { return sel1; }
    if (sel2.is_valid()) { return sel2; }
    return {};
}

NullableValue NullableBuilder::arith_binary(NullableValue left, NullableValue right, ArithBinaryOp op) {
    Value x = base_->arith_binary(left.v, right.v, op);
    return {x, combine_selection(left.null, right.null)};
}

NullableValue NullableBuilder::arith_binary_checked(NullableValue left, NullableValue right, ArithBinaryOp op) {
    MaybePredicate is_null = combine_selection(left.null, right.null);
    if (is_null.is_valid()) {
        Value x = base_->checked_op(base_->arith_binary(left.v, right.v, op), base_->not_(is_null.value()));
        return {x, is_null};
    }
    return base_->arith_binary(left.v, right.v, op, ArithBinaryOpFlags::SafetyCheck);
}

NullableValue NullableBuilder::add(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::Add);
}

NullableValue NullableBuilder::sub(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::Sub);
}

NullableValue NullableBuilder::mul(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::Mul);
}

NullableValue NullableBuilder::and_(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::And);
}

NullableValue NullableBuilder::or_(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::Or);
}

NullableValue NullableBuilder::div(NullableValue left, NullableValue right, bool null_on_invalid_division) {
    if (left.v.dtype() == right.v.dtype() && is_simple_int_dtype(left.v.dtype())) {
        return safe_int_div_mod(this, base_, left, right, ArithBinaryOp::Div, null_on_invalid_division);
    }
    return arith_binary(left, right, ArithBinaryOp::Div);
}

NullableValue NullableBuilder::mod(NullableValue left, NullableValue right, bool null_on_invalid_division) {
    if (left.v.dtype() == right.v.dtype() && is_simple_int_dtype(left.v.dtype())) {
        return safe_int_div_mod(this, base_, left, right, ArithBinaryOp::Mod, null_on_invalid_division);
    }
    return arith_binary(left, right, ArithBinaryOp::Mod);
}

NullableValue NullableBuilder::greatest(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::Max);
}

NullableValue NullableBuilder::greatest(nonstd::span<const NullableValue> args) {
    if (args.empty()) { nullable_invalid_config("greatest argument list can't be empty"); }
    return fold_bin_op(this, args, [&](Value l, Value r) { return base_->max(l, r); });
}

NullableValue NullableBuilder::least(NullableValue left, NullableValue right) {
    return arith_binary(left, right, ArithBinaryOp::Min);
}

NullableValue NullableBuilder::least(nonstd::span<const NullableValue> args) {
    if (args.empty()) { nullable_invalid_config("least argument list can't be empty"); }
    return fold_bin_op(this, args, [&](Value l, Value r) { return base_->min(l, r); });
}

NullableValue NullableBuilder::add_checked(NullableValue left, NullableValue right) {
    return arith_binary_checked(left, right, ArithBinaryOp::Add);
}
NullableValue NullableBuilder::sub_checked(NullableValue left, NullableValue right) {
    return arith_binary_checked(left, right, ArithBinaryOp::Sub);
}
NullableValue NullableBuilder::mul_checked(NullableValue left, NullableValue right) {
    return arith_binary_checked(left, right, ArithBinaryOp::Mul);
}

NullableValue NullableBuilder::int_cast(NullableValue arg, ScalarDataType to, IntCastKind kind) {
    return {base_->int_cast(arg.v, to, kind), arg.null};
}

NullableValue NullableBuilder::trunc_checked(NullableValue arg, ScalarDataType to) {
    if (arg.null.is_valid()) {
        return {base_->checked_op(base_->int_cast(arg.v, to, IntCastKind::Trunc), base_->not_(arg.null.value())),
                arg.null};
    }
    return base_->int_cast(arg.v, to, IntCastKind::Trunc, true);
}

NullableValue NullableBuilder::cast(NullableValue arg, ScalarDataType to) {
    return signed_cast(arg, to);
}

NullableValue NullableBuilder::signed_cast(NullableValue arg, ScalarDataType to) {
    return {base_->signed_cast(arg.v, to), arg.null};
}

NullableValue NullableBuilder::unsigned_cast(NullableValue arg, ScalarDataType to) {
    return {base_->unsigned_cast(arg.v, to), arg.null};
}

NullableValue NullableBuilder::trunc(NullableValue arg, ScalarDataType to) {
    return int_cast(arg, to, IntCastKind::Trunc);
}

NullableValue NullableBuilder::sext(NullableValue arg, ScalarDataType to) {
    return int_cast(arg, to, IntCastKind::Sext);
}

NullableValue NullableBuilder::zext(NullableValue arg, ScalarDataType to) {
    return int_cast(arg, to, IntCastKind::Zext);
}

NullableValue NullableBuilder::float_cast(NullableValue arg, ScalarDataType to, bool is_unsigned) {
    return {base_->float_cast(arg.v, to, is_unsigned), arg.null};
}

NullableValue NullableBuilder::negate(NullableValue arg) {
    return {base_->negate(arg.v), arg.null};
}

NullableValue NullableBuilder::arith_unary(NullableValue arg, ArithUnaryOp op) {
    return {base_->arith_unary(arg.v, op), arg.null};
}

NullableValue NullableBuilder::abs(NullableValue arg) {
    return arith_unary(arg, ArithUnaryOp::Abs);
}

NullableValue NullableBuilder::negate_checked(NullableValue arg) {
    if (arg.null.is_valid()) {
        return {base_->checked_op(base_->negate(arg.v), base_->not_(arg.null.value())), arg.null};
    }
    return base_->negate_checked(arg.v);
}

NullableValue NullableBuilder::abs_checked(NullableValue arg) {
    if (arg.null.is_valid()) { return {base_->checked_op(base_->abs(arg.v), base_->not_(arg.null.value())), arg.null}; }
    return base_->abs_checked(arg.v);
}

NullablePredicate NullableBuilder::not_(NullablePredicate arg) {
    return {base_->not_(arg.v), arg.null};
}

NullablePredicate NullableBuilder::cmp(NullableValue left, NullableValue right, CmpOp op) {
    Predicate x = base_->cmp(left.v, right.v, op);
    return {x, combine_selection(left.null, right.null)};
}

NullablePredicate NullableBuilder::cmp_eq(NullableValue left, NullableValue right) {
    return cmp(left, right, CmpOp::Equal);
}

NullablePredicate NullableBuilder::cmp_ne(NullableValue left, NullableValue right) {
    return cmp(left, right, CmpOp::NotEqual);
}

NullablePredicate NullableBuilder::cmp_gt(NullableValue left, NullableValue right) {
    return cmp(left, right, CmpOp::Greater);
}

NullablePredicate NullableBuilder::cmp_ge(NullableValue left, NullableValue right) {
    return cmp(left, right, CmpOp::GreaterEqual);
}

NullablePredicate NullableBuilder::cmp_lt(NullableValue left, NullableValue right) {
    return cmp(left, right, CmpOp::Less);
}

NullablePredicate NullableBuilder::cmp_le(NullableValue left, NullableValue right) {
    return cmp(left, right, CmpOp::LessEqual);
}

NullablePredicate NullableBuilder::bit_test(NullableValue left, NullableValue right) {
    return cmp_ne(and_(left, right), base_->con(0, left.v.dtype()));
}

NullablePredicate NullableBuilder::bit_testn(NullableValue left, NullableValue right) {
    return cmp_eq(and_(left, right), base_->con(0, left.v.dtype()));
}

NullablePredicate NullableBuilder::is_positive(NullableValue arg) {
    return cmp_gt(arg, base_->con(0, arg.v.dtype()));
}

NullablePredicate NullableBuilder::is_negative(NullableValue arg) {
    return cmp_lt(arg, base_->con(0, arg.v.dtype()));
}

NullablePredicate NullableBuilder::predicate_binary(NullablePredicate left, NullablePredicate right,
                                                    PredicateBinaryOp op) {
    Predicate x = base_->predicate_binary(left.v, right.v, op);
    return {x, combine_selection(left.null, right.null)};
}

NullablePredicate NullableBuilder::and_(NullablePredicate left, NullablePredicate right) {
    return predicate_binary(left, right, PredicateBinaryOp::And);
}

NullablePredicate NullableBuilder::and_(nonstd::span<const NullablePredicate> args) {
    if (args.empty()) { nullable_invalid_config("and argument list can't be empty"); }
    return fold_bin_op(this, args, [&](Predicate l, Predicate r) { return base_->and_(l, r); });
}

NullablePredicate NullableBuilder::or_(NullablePredicate left, NullablePredicate right) {
    return predicate_binary(left, right, PredicateBinaryOp::Or);
}

NullablePredicate NullableBuilder::or_(nonstd::span<const NullablePredicate> args) {
    if (args.empty()) { nullable_invalid_config("or argument list can't be empty"); }
    return fold_bin_op(this, args, [&](Predicate l, Predicate r) { return base_->or_(l, r); });
}

Predicate NullableBuilder::is_null(NullableValue arg) {
    if (arg.null.is_valid()) { return arg.null.value(); }
    return base_->false_();
}

Predicate NullableBuilder::is_not_null(NullableValue arg) {
    if (arg.null.is_valid()) { return base_->not_(arg.null.value()); }
    return base_->true_();
}

NullablePredicate NullableBuilder::true_unless_null(NullableValue arg) {
    if (arg.null.is_valid()) { return {base_->not_(arg.null.value()), arg.null.value()}; }
    return base_->true_();
}

Predicate NullableBuilder::is_true(NullablePredicate arg) {
    if (arg.null.is_valid()) { return base_->andnot(arg.null.value(), arg.v); }
    return arg.v;
}

Predicate NullableBuilder::is_not_true(NullablePredicate arg) {
    Predicate x = base_->not_(arg.v);
    return combine_selection(arg.null, x).value();
}

Predicate NullableBuilder::is_false(NullablePredicate arg) {
    return base_->not_(combine_selection(arg.v, arg.null).value());
}

Predicate NullableBuilder::is_not_false(NullablePredicate arg) {
    return combine_selection(arg.null, arg.v).value();
}

Predicate NullableBuilder::is_distinct(NullableValue left, NullableValue right) {
    Predicate x = base_->cmp_ne(left.v, right.v);
    // (NE & !LN & !RN) | (LN ^ RN)
    if (left.null.is_valid() && right.null.is_valid()) {
        return base_->or_(base_->andnot(left.null.value(), x), base_->xor_(left.null.value(), right.null.value()));
    }
    if (left.null.is_valid()) { return base_->or_(x, left.null.value()); }
    if (right.null.is_valid()) { return base_->or_(x, right.null.value()); }
    return x;
}

Predicate NullableBuilder::is_not_distinct(NullableValue left, NullableValue right) {
    Predicate x = base_->cmp_eq(left.v, right.v);
    // (EQ & !LN & !RN) | (LN & RN)
    if (left.null.is_valid() && right.null.is_valid()) {
        return base_->and_(base_->or_(left.null.value(), x), base_->xnor(left.null.value(), right.null.value()));
    }
    if (left.null.is_valid()) { return base_->andnot(left.null.value(), x); }
    if (right.null.is_valid()) { return base_->andnot(right.null.value(), x); }
    return x;
}

NullableValue NullableBuilder::nullif(NullableValue left, NullableValue right) {
    NullablePredicate x = cmp_eq(left, right);
    return {left.v, combine_selection(left.null, is_true(x))};
}

NullableValue NullableBuilder::if_else_null(NullablePredicate cond, NullableValue truthy) {
    return {truthy.v, combine_selection(base_->not_(cond.v), combine_selection(cond.null, truthy.null))};
}

NullableValue NullableBuilder::if_else(NullablePredicate cond, NullableValue truthy, NullableValue falsy) {
    Predicate take_truthy = is_true(cond);
    Value selected = base_->select(take_truthy, truthy.v, falsy.v);

    if (truthy.null.is_valid() && falsy.null.is_valid()) {
        Predicate selected_null =
            base_->or_(base_->and_(take_truthy, truthy.null.value()), base_->andnot(take_truthy, falsy.null.value()));
        return {selected, selected_null};
    }
    if (truthy.null.is_valid()) { return {selected, base_->and_(take_truthy, truthy.null.value())}; }
    if (falsy.null.is_valid()) { return {selected, base_->andnot(take_truthy, falsy.null.value())}; }
    return {selected};
}

NullableValue NullableBuilder::coalesce(nonstd::span<const NullableValue> args) {
    if (args.empty()) { nullable_invalid_config("coalesce can't have less than 1 argument"); }

    NullableValue result = args[0];
    for (size_t i = 1; i < args.size(); ++i) {
        const NullableValue &arg = args[i];

        if (!result.null.is_valid()) { return result; }
        if (!arg.null.is_valid()) {
            result.v = base_->select(result.null.value(), arg.v, result.v);
            result.null = {};
            return result;
        }

        result.v = base_->select(result.null.value(), arg.v, result.v);
        result.null = base_->and_(result.null.value(), arg.null.value());
    }
    return result;
}

static NullableValue first_case_x_when(NullableBuilder &nb, NullableValue arg,
                                       nonstd::span<const std::pair<NullableValue, NullableValue>> cases,
                                       std::optional<NullableValue> default_case) {
    size_t idx = cases.size();
    if (default_case.has_value()) {
        return nb.if_else(nb.cmp(arg, cases[idx - 1].first, CmpOp::Equal), cases[idx - 1].second, *default_case);
    }
    return nb.if_else_null(nb.cmp(arg, cases[idx - 1].first, CmpOp::Equal), cases[idx - 1].second);
}

NullableValue NullableBuilder::case_x_when(NullableValue arg,
                                           nonstd::span<const std::pair<NullableValue, NullableValue>> cases,
                                           std::optional<NullableValue> default_case) {
    if (cases.empty()) { nullable_invalid_config("case can't be empty"); }
    size_t idx = cases.size();
    NullableValue result = first_case_x_when(*this, arg, cases, default_case);
    --idx;
    for (; idx != 0; --idx) {
        result = if_else(cmp(arg, cases[idx - 1].first, CmpOp::Equal), cases[idx - 1].second, result);
    }
    return result;
}

static NullableValue first_case_when(NullableBuilder &nb,
                                     nonstd::span<const std::pair<NullablePredicate, NullableValue>> cases,
                                     std::optional<NullableValue> default_case) {
    size_t idx = cases.size();
    if (default_case.has_value()) { return nb.if_else(cases[idx - 1].first, cases[idx - 1].second, *default_case); }
    return nb.if_else_null(cases[idx - 1].first, cases[idx - 1].second);
}

NullableValue NullableBuilder::case_when(nonstd::span<const std::pair<NullablePredicate, NullableValue>> cases,
                                         std::optional<NullableValue> default_case) {
    if (cases.empty()) { nullable_invalid_config("case can't be empty"); }
    size_t idx = cases.size();
    NullableValue result = first_case_when(*this, cases, default_case);
    --idx;
    for (; idx != 0; --idx) {
        result = if_else(cases[idx - 1].first, cases[idx - 1].second, result);
    }
    return result;
}

void NullableBuilder::nbit_store(NullableValue arg, Argument dst, Argument dst_null) {
    nbit_store_ext(arg, dst, dst_null, true);
}

void NullableBuilder::nbit_store(NullablePredicate arg, Argument dst, Argument dst_null) {
    nbit_store_ext(arg, dst, dst_null, true);
}

void NullableBuilder::nbool_store(NullableValue arg, Argument dst, Argument dst_null) {
    nbool_store_ext(arg, dst, dst_null, true);
}

void NullableBuilder::nbool_store(NullablePredicate arg, Argument dst, Argument dst_null) {
    nbool_store_ext(arg, dst, dst_null, true);
}

void NullableBuilder::nbit_store_ext(NullableValue arg, Argument dst, Argument dst_null, bool true_means_null) {
    base_->store(arg.v, dst);
    base_->store(store_null_mask(base_, arg.null, true_means_null), dst_null);
}

void NullableBuilder::nbit_store_ext(NullablePredicate arg, Argument dst, Argument dst_null, bool true_means_null) {
    base_->store(arg.v, dst);
    base_->store(store_null_mask(base_, arg.null, true_means_null), dst_null);
}

void NullableBuilder::nbool_store_ext(NullableValue arg, Argument dst, Argument dst_null, bool true_means_null) {
    base_->store(arg.v, dst);
    base_->store(base_->bit2bool(store_null_mask(base_, arg.null, true_means_null)), dst_null);
}

void NullableBuilder::nbool_store_ext(NullablePredicate arg, Argument dst, Argument dst_null, bool true_means_null) {
    base_->store(arg.v, dst);
    base_->store(base_->bit2bool(store_null_mask(base_, arg.null, true_means_null)), dst_null);
}

void NullableBuilder::nval_store(NullableValue arg, Argument dst, Value nval) {
    if (arg.null.is_valid()) {
        base_->store(base_->select(arg.null.value(), nval, arg.v), dst);
    } else {
        base_->store(arg.v, dst);
    }
}

void NullableBuilder::nbool_scatter(NullableValue arg, Argument dst, Argument dst_null, Value idx) {
    nbool_scatter_ext(arg, dst, dst_null, idx, true);
}

void NullableBuilder::nbool_scatter_ext(NullableValue arg, Argument dst, Argument dst_null, Value idx,
                                        bool true_means_null) {
    base_->scatter(arg.v, idx, dst);
    base_->scatter(base_->bit2bool(store_null_mask(base_, arg.null, true_means_null)), idx, dst_null);
}

void NullableBuilder::nval_scatter(NullableValue arg, Argument dst, Value idx, Value nval) {
    if (arg.null.is_valid()) {
        base_->scatter(base_->select(arg.null.value(), nval, arg.v), idx, dst);
    } else {
        base_->scatter(arg.v, idx, dst);
    }
}

void NullableBuilder::arith_agg(NullableValue arg, ArithBinaryOp op, Argument dst) {
    if (arg.null.is_valid()) {
        base_->cond_arith_agg(arg.v, base_->not_(arg.null.value()), op, dst);
    } else {
        base_->arith_agg(arg.v, op, dst);
    }
}

void NullableBuilder::cond_arith_agg(NullableValue arg, NullablePredicate cond, ArithBinaryOp op, Argument dst) {
    if (arg.null.is_valid()) {
        base_->cond_arith_agg(arg.v, is_true(and_(cond, base_->not_(arg.null.value()))), op, dst);
    } else {
        base_->cond_arith_agg(arg.v, is_true(cond), op, dst);
    }
}

void NullableBuilder::sum(NullableValue arg, Argument dst) {
    arith_agg(arg, ArithBinaryOp::Add, dst);
}

void NullableBuilder::min_agg(NullableValue arg, Argument dst) {
    arith_agg(arg, ArithBinaryOp::Min, dst);
}

void NullableBuilder::max_agg(NullableValue arg, Argument dst) {
    arith_agg(arg, ArithBinaryOp::Max, dst);
}

void NullableBuilder::grouped_arith_agg(NullableValue arg, Value idx, ArithBinaryOp op, Argument table) {
    if (arg.null.is_valid()) {
        base_->grouped_cond_arith_agg(arg.v, base_->not_(arg.null.value()), idx, op, table);
    } else {
        base_->grouped_arith_agg(arg.v, idx, op, table);
    }
}

void NullableBuilder::grouped_cond_arith_agg(NullableValue arg, NullablePredicate cond, Value idx, ArithBinaryOp op,
                                             Argument table) {
    if (arg.null.is_valid()) {
        base_->grouped_cond_arith_agg(arg.v, is_true(and_(cond, base_->not_(arg.null.value()))), idx, op, table);
    } else {
        base_->grouped_cond_arith_agg(arg.v, is_true(cond), idx, op, table);
    }
}

void NullableBuilder::grouped_sum(NullableValue arg, Value idx, Argument table) {
    grouped_arith_agg(arg, idx, ArithBinaryOp::Add, table);
}

void NullableBuilder::grouped_min(NullableValue arg, Value idx, Argument table) {
    grouped_arith_agg(arg, idx, ArithBinaryOp::Min, table);
}

void NullableBuilder::grouped_max(NullableValue arg, Value idx, Argument table) {
    grouped_arith_agg(arg, idx, ArithBinaryOp::Max, table);
}

void NullableBuilder::countif(NullablePredicate arg, Argument dst) {
    base_->countif(is_true(arg), dst);
}

void NullableBuilder::count_notnull(NullableValue arg, Argument dst) {
    base_->countif(is_not_null(arg), dst);
}

void NullableBuilder::has_nulls(NullableValue arg, Argument dst) {
    base_->or_agg(is_null(arg), dst);
}

void NullableBuilder::all(NullablePredicate arg, Argument dst) {
    base_->and_agg(is_true(arg), dst);
}

void NullableBuilder::any(NullablePredicate arg, Argument dst) {
    base_->or_agg(is_true(arg), dst);
}

void NullableBuilder::find_true_indices(NullablePredicate arg, Argument dst, Argument dst_size) {
    base_->pack(base_->index(ScalarDataType::I32), is_true(arg), dst, dst_size);
}

void NullableBuilder::update_true_indices(Value idx, NullablePredicate arg, Argument dst, Argument dst_size) {
    base_->pack(idx, is_true(arg), dst, dst_size);
}

void NullableBuilder::find_notnull_indices(NullableValue arg, Argument dst, Argument dst_size) {
    base_->pack(base_->index(ScalarDataType::I32), is_not_null(arg), dst, dst_size);
}

void NullableBuilder::find_null_indices(NullableValue arg, Argument dst, Argument dst_size) {
    base_->pack(base_->index(ScalarDataType::I32), is_null(arg), dst, dst_size);
}

} // namespace nullable
} // namespace simjit
