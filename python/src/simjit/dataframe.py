# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

"""
High-level dataframe-style API for simjit.

Example:

    import simjit as sj
    import simjit.dataframe as sdf

    t = sdf.table({
        "x": sj.i32,
        "y": sj.i32,
        "price": sj.f64,
        "discount": sj.f64,
    })

    program = sdf.select({
        "sum_xy": t.x + t.y,
        "gross": t.price * (1.0 - t.discount),
        "flag": ((t.x > 0) & (t.y < 100)).ifelse(1, 0),
        "total_gross": t.price.sum(),
    })

    lowered = program.to_dsl()
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn, TypeAlias, cast as typing_cast

from . import ir
from .integrations import _numpy as numpy_integration, _pyarrow as pyarrow_integration

ScalarType = ir.ScalarType
I8 = ir.I8
I16 = ir.I16
I32 = ir.I32
I64 = ir.I64
U8 = ir.U8
U16 = ir.U16
U32 = ir.U32
U64 = ir.U64
F32 = ir.F32
F64 = ir.F64
I1 = ir.I1
timestamp64 = ir.timestamp64

TimestampValue: TypeAlias = numpy_integration.TimestampValue


def _is_float_type(dt: ir.ScalarType | None) -> bool:
    return dt in {F32, F64}


def as_expr(value: ValueLike) -> ir.Expr:
    if isinstance(value, ir.Expr):
        return value
    if isinstance(value, Expr):
        return value.to_dsl()
    if isinstance(value, bool):
        return ir.ConstExpr(dt=I1, value=value)
    if isinstance(value, (int, float)):
        return ir.ConstExpr(dt=None, value=value)
    raise TypeError(f"cannot convert {type(value).__name__} to Expr")


def as_predicate_expr(value: PredicateLike) -> ir.Expr:
    if isinstance(value, ir.Expr):
        if value.dt != I1:
            raise TypeError("expected predicate-typed ir.Expr")
        return value
    if isinstance(value, Expr):
        expr = value.to_dsl()
        if expr.dt != I1:
            raise TypeError("expected predicate-typed expression")
        return expr
    if isinstance(value, bool):
        return ir.ConstExpr(dt=I1, value=value)
    raise TypeError(f"cannot convert {type(value).__name__} to predicate Expr")


def const(value: float | int | bool, dt: ir.ScalarType | None = None) -> Expr:
    if dt is None and isinstance(value, bool):
        dt = I1
    return Expr(ir.ConstExpr(dt=dt, value=value))


def load(
    name: str,
    dt: ir.ScalarType | None = None,
    kind: ir.LoadStoreKind = ir.LoadStoreKind.Unaligned,
) -> Expr:
    return Expr(ir.LoadExpr(dt=dt, name=name, kind=kind))


def load_splat(name: str, dt: ir.ScalarType | None = None) -> Expr:
    return Expr(ir.LoadSplatExpr(dt=dt, name=name))


def gather(idx: ValueLike, name: str, dt: ir.ScalarType | None = None) -> Expr:
    return Expr(ir.GatherExpr(dt=dt, idx=as_expr(idx), name=name))


def index(dt: ir.ScalarType | None = None) -> Expr:
    return Expr(ir.IndexExpr(dt=dt))


def arith_binary(
    op: ir.ArithBinaryOp, lhs: ValueLike, rhs: ValueLike, checked: bool = False
) -> Expr:
    lhs_expr = as_expr(lhs)
    rhs_expr = as_expr(rhs)
    return Expr(
        ir.ArithBinaryExpr(
            dt=None,
            op=op,
            lhs=lhs_expr,
            rhs=rhs_expr,
            checked=checked,
        )
    )


def predicate_binary(
    op: ir.PredicateBinaryOp, lhs: PredicateLike, rhs: PredicateLike
) -> Expr:
    return Expr(
        ir.PredicateBinaryExpr(
            dt=I1,
            op=op,
            lhs=as_predicate_expr(lhs),
            rhs=as_predicate_expr(rhs),
        )
    )


def arith_unary(
    op: ir.ArithUnaryOp, arg: ValueLike, checked: bool = False
) -> Expr:
    arg_expr = as_expr(arg)
    return Expr(
        ir.ArithUnaryExpr(
            dt=arg_expr.dt, op=op, arg=arg_expr, checked=checked
        )
    )


def predicate_not(arg: PredicateLike) -> Expr:
    return Expr(ir.PredicateNotExpr(dt=I1, arg=as_predicate_expr(arg)))


def cmp(
    op: ir.CompareOp, lhs: ValueLike, rhs: ValueLike, unsigned: bool = False
) -> Expr:
    return Expr(
        ir.CompareExpr(
            dt=I1,
            op=op,
            lhs=as_expr(lhs),
            rhs=as_expr(rhs),
            unsigned=unsigned,
        )
    )


def int_cast(
    kind: ir.IntCastKind,
    dt: ir.ScalarType | None,
    arg: ValueLike,
    checked: bool = False,
) -> Expr:
    return Expr(
        ir.IntCastExpr(dt=dt, kind=kind, arg=as_expr(arg), checked=checked)
    )


def float_cast(
    dt: ir.ScalarType | None, arg: ValueLike, is_unsigned: bool = False
) -> Expr:
    return Expr(ir.FloatCastExpr(dt=dt, arg=as_expr(arg), is_unsigned=is_unsigned))


def bitcast(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return Expr(ir.BitCastExpr(dt=dt, arg=as_expr(arg)))


def function(
    name: ir.FunctionName, *args: ValueLike, dt: ir.ScalarType | None = None
) -> Expr:
    lowered_args: list[ir.Expr] = []
    for arg in args:
        if isinstance(arg, Expr):
            lowered_args.append(arg.to_dsl())
        elif isinstance(arg, ir.Expr):
            lowered_args.append(arg)
        elif isinstance(arg, bool):
            lowered_args.append(as_predicate_expr(arg))
        else:
            lowered_args.append(as_expr(arg))
    expr = ir.FunctionExpr(dt=dt, name=name, args=tuple(lowered_args))
    return Expr(expr)


def select(cond: PredicateLike, truthy: ValueLike, falsy: ValueLike) -> Expr:
    truthy_expr = as_expr(truthy)
    falsy_expr = as_expr(falsy)
    return Expr(
        ir.SelectExpr(
            dt=None,
            cond=as_predicate_expr(cond),
            truthy=truthy_expr,
            falsy=falsy_expr,
        )
    )


def fpclass(arg: ValueLike, flags: ir.FpClassFlags) -> Expr:
    return Expr(ir.FpClassExpr(dt=I1, arg=as_expr(arg), flags=flags))


def permute(arg: ValueLike, permute_idxs: int, is_bit: bool) -> Expr:
    arg_expr = as_expr(arg)
    return Expr(
        ir.PermuteExpr(
            dt=None,
            arg=arg_expr,
            permute_idxs=permute_idxs,
            is_bit=is_bit,
        )
    )


def store(
    value: ValueLike,
    kind: ir.LoadStoreKind = ir.LoadStoreKind.Unaligned,
    cond: PredicateLike | None = None,
) -> Expr:
    value_expr = as_expr(value)
    cond_expr = None if cond is None else as_predicate_expr(cond)
    return Expr(ir.StoreExpr(dt=None, value=value_expr, kind=kind, cond=cond_expr))


def scatter(
    value: ValueLike, idx: ValueLike, cond: PredicateLike | None = None
) -> Expr:
    value_expr = as_expr(value)
    cond_expr = None if cond is None else as_predicate_expr(cond)
    return Expr(
        ir.ScatterExpr(
            dt=None,
            value=value_expr,
            idx=as_expr(idx),
            cond=cond_expr,
        )
    )


def pack(value: ValueLike, cond: PredicateLike, dst_size: str | None = None) -> Expr:
    value_expr = as_expr(value)
    return Expr(
        ir.PackExpr(
            dt=None,
            value=value_expr,
            cond=as_predicate_expr(cond),
            dst_size=dst_size,
        )
    )


def arith_agg(
    op: ir.ArithBinaryOp, arg: ValueLike, cond: PredicateLike | None = None
) -> FrozenExpr:
    arg_expr = as_expr(arg)
    cond_expr = None if cond is None else as_predicate_expr(cond)
    return FrozenExpr(ir.ArithAggExpr(dt=None, op=op, arg=arg_expr, cond=cond_expr))


def predicate_agg(op: ir.PredicateBinaryOp, arg: PredicateLike) -> FrozenExpr:
    return FrozenExpr(ir.PredicateAggExpr(dt=I1, op=op, arg=as_predicate_expr(arg)))


def count_if(
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    grouped_table = _require_grouped_table(group_idx, table)
    if grouped_table is None:
        return FrozenExpr(ir.CountIfExpr(dt=I64, cond=as_predicate_expr(cond)))
    assert group_idx is not None
    return grouped_arith_agg(
        ir.ArithBinaryOp.Add, i64(1), group_idx, grouped_table, cond=cond
    )


def _require_grouped_table(
    group_idx: ValueLike | None, table: str | None
) -> str | None:
    if group_idx is None:
        if table is not None:
            raise TypeError("table requires group_idx")
        return None
    if table is None:
        raise TypeError("group_idx requires table")
    if not isinstance(table, str):
        raise TypeError("table must be a string")
    return table


def _arith_agg_or_grouped(
    op: ir.ArithBinaryOp,
    arg: ValueLike,
    cond: PredicateLike | None = None,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    grouped_table = _require_grouped_table(group_idx, table)
    if grouped_table is None:
        return arith_agg(op, arg, cond=cond)
    assert group_idx is not None
    return grouped_arith_agg(op, arg, group_idx, grouped_table, cond=cond)


def _count_if_or_grouped(
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return count_if(cond, group_idx=group_idx, table=table)


def grouped_arith_agg(
    op: ir.ArithBinaryOp,
    arg: ValueLike,
    idx: ValueLike,
    table: str,
    cond: PredicateLike | None = None,
) -> FrozenExpr:
    arg_expr = as_expr(arg)
    cond_expr = None if cond is None else as_predicate_expr(cond)
    return FrozenExpr(
        ir.GroupedArithAggExpr(
            dt=None,
            op=op,
            arg=arg_expr,
            idx=as_expr(idx),
            table=table,
            cond=cond_expr,
        )
    )


def add(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Add, lhs, rhs)


def add_checked(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Add, lhs, rhs, checked=True)


def sub(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Sub, lhs, rhs)


def sub_checked(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Sub, lhs, rhs, checked=True)


def mul(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Mul, lhs, rhs)


def mul_checked(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Mul, lhs, rhs, checked=True)


def div(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Div, lhs, rhs)


def udiv(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.UDiv, lhs, rhs)


def mod(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Mod, lhs, rhs)


def umod(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.UMod, lhs, rhs)


def min(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Min, lhs, rhs)


def max(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.Max, lhs, rhs)


def umin(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.UMin, lhs, rhs)


def umax(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.UMax, lhs, rhs)


def and_(lhs: ValueLike, rhs: ValueLike) -> Expr:
    lhs_expr = as_expr(lhs)
    rhs_expr = as_expr(rhs)
    if lhs_expr.dt == I1 and rhs_expr.dt == I1:
        return predicate_binary(ir.PredicateBinaryOp.And, lhs_expr, rhs_expr)
    return arith_binary(ir.ArithBinaryOp.And, lhs_expr, rhs_expr)


def or_(lhs: ValueLike, rhs: ValueLike) -> Expr:
    lhs_expr = as_expr(lhs)
    rhs_expr = as_expr(rhs)
    if lhs_expr.dt == I1 and rhs_expr.dt == I1:
        return predicate_binary(ir.PredicateBinaryOp.Or, lhs_expr, rhs_expr)
    return arith_binary(ir.ArithBinaryOp.Or, lhs_expr, rhs_expr)


def xor(lhs: ValueLike, rhs: ValueLike) -> Expr:
    lhs_expr = as_expr(lhs)
    rhs_expr = as_expr(rhs)
    if lhs_expr.dt == I1 and rhs_expr.dt == I1:
        return predicate_binary(ir.PredicateBinaryOp.Xor, lhs_expr, rhs_expr)
    return arith_binary(ir.ArithBinaryOp.Xor, lhs_expr, rhs_expr)


def andnot(lhs: ValueLike, rhs: ValueLike) -> Expr:
    lhs_expr = as_expr(lhs)
    rhs_expr = as_expr(rhs)
    if lhs_expr.dt == I1 and rhs_expr.dt == I1:
        return predicate_binary(ir.PredicateBinaryOp.AndNot, lhs_expr, rhs_expr)
    return arith_binary(ir.ArithBinaryOp.AndNot, lhs_expr, rhs_expr)


def sll(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.ShiftLeftLogical, lhs, rhs)


def srl(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.ShiftRightLogical, lhs, rhs)


def sra(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.ShiftRightArith, lhs, rhs)


def rotl(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.RotateLeft, lhs, rhs)


def rotr(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return arith_binary(ir.ArithBinaryOp.RotateRight, lhs, rhs)


def predicate_and(lhs: PredicateLike, rhs: PredicateLike) -> Expr:
    return predicate_binary(ir.PredicateBinaryOp.And, lhs, rhs)


def predicate_or(lhs: PredicateLike, rhs: PredicateLike) -> Expr:
    return predicate_binary(ir.PredicateBinaryOp.Or, lhs, rhs)


def predicate_xor(lhs: PredicateLike, rhs: PredicateLike) -> Expr:
    return predicate_binary(ir.PredicateBinaryOp.Xor, lhs, rhs)


def predicate_andnot(lhs: PredicateLike, rhs: PredicateLike) -> Expr:
    return predicate_binary(ir.PredicateBinaryOp.AndNot, lhs, rhs)


def predicate_xnor(lhs: PredicateLike, rhs: PredicateLike) -> Expr:
    return predicate_binary(ir.PredicateBinaryOp.XNor, lhs, rhs)


def negate(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Negate, arg)


def negate_checked(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Negate, arg, checked=True)


def abs(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Abs, arg)


def abs_checked(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Abs, arg, checked=True)


def not_(arg: ValueLike | PredicateLike) -> Expr:
    if isinstance(arg, bool):
        return predicate_not(arg)
    arg_expr = as_expr(arg)
    if arg_expr.dt == I1:
        return predicate_not(arg_expr)
    return arith_unary(ir.ArithUnaryOp.Not, arg_expr)


def lzcnt(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Lzcnt, arg)


def tzcnt(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Tzcnt, arg)


def popcnt(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Popcount, arg)


def round_(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.RoundNearest, arg)


def floor(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.RoundDown, arg)


def ceil(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.RoundUp, arg)


def round2zero(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.RoundTruncate, arg)


def sqrt(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Sqrt, arg)


def rsqrt(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Rsqrt, arg)


def rcp(arg: ValueLike) -> Expr:
    return arith_unary(ir.ArithUnaryOp.Rcp, arg)


def cmp_eq(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.Equal, lhs, rhs)


def cmp_ne(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.NotEqual, lhs, rhs)


def cmp_gt(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.Greater, lhs, rhs)


def cmp_ge(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.GreaterEqual, lhs, rhs)


def cmp_lt(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.Less, lhs, rhs)


def cmp_le(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.LessEqual, lhs, rhs)


def cmp_ugt(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.Greater, lhs, rhs, unsigned=True)


def cmp_uge(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.GreaterEqual, lhs, rhs, unsigned=True)


def cmp_ult(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.Less, lhs, rhs, unsigned=True)


def cmp_ule(lhs: ValueLike, rhs: ValueLike) -> Expr:
    return cmp(ir.CompareOp.LessEqual, lhs, rhs, unsigned=True)


def trunc(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return int_cast(ir.IntCastKind.Trunc, dt, arg)


def trunc_checked(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return int_cast(ir.IntCastKind.Trunc, dt, arg, checked=True)


def sext(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return int_cast(ir.IntCastKind.Sext, dt, arg)


def zext(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return int_cast(ir.IntCastKind.Zext, dt, arg)


def signed_cast(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return int_cast(ir.IntCastKind.Signed, dt, arg)


def unsigned_cast(dt: ir.ScalarType | None, arg: ValueLike) -> Expr:
    return int_cast(ir.IntCastKind.Unsigned, dt, arg)


def sum(
    arg: ValueLike, *, group_idx: ValueLike | None = None, table: str | None = None
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Add, arg, group_idx=group_idx, table=table
    )


def sum_if(
    arg: ValueLike,
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Add, arg, cond=cond, group_idx=group_idx, table=table
    )


def product(
    arg: ValueLike, *, group_idx: ValueLike | None = None, table: str | None = None
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Mul, arg, group_idx=group_idx, table=table
    )


def product_if(
    arg: ValueLike,
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Mul, arg, cond=cond, group_idx=group_idx, table=table
    )


def min_agg(
    arg: ValueLike, *, group_idx: ValueLike | None = None, table: str | None = None
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Min, arg, group_idx=group_idx, table=table
    )


def min_agg_if(
    arg: ValueLike,
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Min, arg, cond=cond, group_idx=group_idx, table=table
    )


def max_agg(
    arg: ValueLike, *, group_idx: ValueLike | None = None, table: str | None = None
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Max, arg, group_idx=group_idx, table=table
    )


def max_agg_if(
    arg: ValueLike,
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.Max, arg, cond=cond, group_idx=group_idx, table=table
    )


def umin_agg(
    arg: ValueLike, *, group_idx: ValueLike | None = None, table: str | None = None
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.UMin, arg, group_idx=group_idx, table=table
    )


def umin_agg_if(
    arg: ValueLike,
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.UMin, arg, cond=cond, group_idx=group_idx, table=table
    )


def umax_agg(
    arg: ValueLike, *, group_idx: ValueLike | None = None, table: str | None = None
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.UMax, arg, group_idx=group_idx, table=table
    )


def umax_agg_if(
    arg: ValueLike,
    cond: PredicateLike,
    *,
    group_idx: ValueLike | None = None,
    table: str | None = None,
) -> FrozenExpr:
    return _arith_agg_or_grouped(
        ir.ArithBinaryOp.UMax, arg, cond=cond, group_idx=group_idx, table=table
    )


def and_agg(arg: PredicateLike) -> FrozenExpr:
    return predicate_agg(ir.PredicateBinaryOp.And, arg)


def or_agg(arg: PredicateLike) -> FrozenExpr:
    return predicate_agg(ir.PredicateBinaryOp.Or, arg)


def xor_agg(arg: PredicateLike) -> FrozenExpr:
    return predicate_agg(ir.PredicateBinaryOp.Xor, arg)


def andnot_agg(arg: PredicateLike) -> FrozenExpr:
    return predicate_agg(ir.PredicateBinaryOp.AndNot, arg)


def grouped_sum(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Add, arg, idx, table)


def grouped_sum_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Add, arg, idx, table, cond=cond)


def grouped_product(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Mul, arg, idx, table)


def grouped_product_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Mul, arg, idx, table, cond=cond)


def grouped_min(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Min, arg, idx, table)


def grouped_min_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Min, arg, idx, table, cond=cond)


def grouped_max(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Max, arg, idx, table)


def grouped_max_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Max, arg, idx, table, cond=cond)


def grouped_umin(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.UMin, arg, idx, table)


def grouped_umin_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.UMin, arg, idx, table, cond=cond)


def grouped_umax(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.UMax, arg, idx, table)


def grouped_umax_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.UMax, arg, idx, table, cond=cond)


def grouped_and(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.And, arg, idx, table)


def grouped_and_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.And, arg, idx, table, cond=cond)


def grouped_or(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Or, arg, idx, table)


def grouped_or_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Or, arg, idx, table, cond=cond)


def grouped_xor(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Xor, arg, idx, table)


def grouped_xor_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.Xor, arg, idx, table, cond=cond)


def grouped_andnot(arg: ValueLike, idx: ValueLike, table: str) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.AndNot, arg, idx, table)


def grouped_andnot_if(
    arg: ValueLike, cond: PredicateLike, idx: ValueLike, table: str
) -> FrozenExpr:
    return grouped_arith_agg(ir.ArithBinaryOp.AndNot, arg, idx, table, cond=cond)


def _is_timestamp_type(dt: ir.ScalarType | None) -> bool:
    return dt is not None and dt.name == "timestamp64"


def _is_int_type(dt: ir.ScalarType | None) -> bool:
    return dt in {I8, U8, I16, U16, I32, U32, I64, U64}


def cast(expr: ValueLike, dt: ScalarType) -> Expr:
    if isinstance(expr, (float, int)):
        return const(expr, dt)

    wrapped = _coerce_value_expr(expr)
    if wrapped.dt == dt:
        return Expr(wrapped.to_dsl())
    if _is_timestamp_type(dt):
        return int_cast(ir.IntCastKind.Cast, dt, wrapped)
    if _is_timestamp_type(wrapped.dt) and _is_int_type(dt):
        return int_cast(ir.IntCastKind.Cast, dt, wrapped)
    if _is_float_type(dt):
        return float_cast(dt, wrapped)
    if _is_int_type(dt):
        return int_cast(ir.IntCastKind.Cast, dt, wrapped)
    return bitcast(dt, wrapped)


def i8(expr: ValueLike) -> Expr:
    return cast(expr, I8)


def i16(expr: ValueLike) -> Expr:
    return cast(expr, I16)


def i32(expr: ValueLike) -> Expr:
    return cast(expr, I32)


def i64(expr: ValueLike) -> Expr:
    return cast(expr, I64)


def u8(expr: ValueLike) -> Expr:
    return cast(expr, U8)


def u16(expr: ValueLike) -> Expr:
    return cast(expr, U16)


def u32(expr: ValueLike) -> Expr:
    return cast(expr, U32)


def u64(expr: ValueLike) -> Expr:
    return cast(expr, U64)


def f32(expr: ValueLike) -> Expr:
    return cast(expr, F32)


def f64(expr: ValueLike) -> Expr:
    return cast(expr, F64)


def _normalize_timestamp_literal(
    value: TimestampValue,
    unit: str,
) -> tuple[int, str | None]:
    return numpy_integration.normalize_timestamp_literal(value, unit)


def timestamp(
    value: TimestampValue,
    unit: ir.TimestampUnit = "ns",
) -> Expr:
    raw, tz = _normalize_timestamp_literal(value, unit)
    return Expr(ir.ConstExpr(dt=timestamp64(unit, tz), value=raw))


def ts_s(value: TimestampValue) -> Expr:
    return timestamp(value, "s")


def ts_ms(value: TimestampValue) -> Expr:
    return timestamp(value, "ms")


def ts_us(value: TimestampValue) -> Expr:
    return timestamp(value, "us")


def ts_ns(value: TimestampValue) -> Expr:
    return timestamp(value, "ns")


def year(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.Year, arg, dt=I32)


def month(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.Month, arg, dt=I32)


def day(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.Day, arg, dt=I32)


def hour(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.Hour, arg, dt=I32)


def minute(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.Minute, arg, dt=I32)


def second(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.Second, arg, dt=I32)


def day_of_week(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.DayOfWeek, arg, dt=I32)


def log2(arg: ValueLike) -> Expr:
    arg_expr = as_expr(arg)
    return function(ir.FunctionName.Log2, arg_expr, dt=arg_expr.dt)


def log2_no_zero(arg: ValueLike) -> Expr:
    arg_expr = as_expr(arg)
    return function(ir.FunctionName.Log2NoZero, arg_expr, dt=arg_expr.dt)


def byteswap(arg: ValueLike) -> Expr:
    arg_expr = as_expr(arg)
    return function(ir.FunctionName.Byteswap, arg_expr, dt=arg_expr.dt)


def bit_floor(arg: ValueLike) -> Expr:
    arg_expr = as_expr(arg)
    return function(ir.FunctionName.BitFloor, arg_expr, dt=arg_expr.dt)


def bit_ceil(arg: ValueLike) -> Expr:
    arg_expr = as_expr(arg)
    return function(ir.FunctionName.BitCeil, arg_expr, dt=arg_expr.dt)


def coalesce(*args: ValueLike) -> Expr:
    if not args:
        raise ValueError("coalesce requires at least one argument")
    arg_exprs = tuple(_coerce_value_expr(arg).to_dsl() for arg in args)
    return function(ir.FunctionName.Coalesce, *arg_exprs)


def nullif(lhs: ValueLike, rhs: ValueLike) -> Expr:
    lhs_expr = _coerce_value_expr(lhs).to_dsl()
    rhs_expr = _coerce_value_expr(rhs).to_dsl()
    return function(ir.FunctionName.NullIf, lhs_expr, rhs_expr)


def is_null(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.IsNull, _coerce_value_expr(arg).to_dsl(), dt=I1)


def is_not_null(arg: ValueLike) -> Expr:
    return function(ir.FunctionName.IsNotNull, _coerce_value_expr(arg).to_dsl(), dt=I1)


def _wrap_existing(expr: ir.Expr) -> Expr:
    return Expr(expr)


def _coerce_value_expr(value: ValueLike) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, ir.Expr):
        wrapped = _wrap_existing(value)
        return wrapped
    if isinstance(value, (int, float)):
        return const(value)
    if isinstance(value, bool):
        raise TypeError("expected value expression, got predicate expression")
    raise TypeError(f"cannot convert {type(value).__name__} to Expr")


def _coerce_predicate_expr(value: PredicateLike) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, ir.Expr):
        wrapped = _wrap_existing(value)
        return wrapped
    if isinstance(value, bool):
        result = const(value)
        return result
    raise TypeError(f"cannot convert {type(value).__name__} to Expr")


class Expr:
    """Base high-level expression wrapper."""

    __slots__: tuple[str, ...] = ("_expr",)

    def __init__(self, expr: ir.Expr):
        self._expr: ir.Expr = expr

    @property
    def dt(self) -> ir.ScalarType | None:
        return self._expr.dt

    def to_dsl(self) -> ir.Expr:
        return self._expr

    def __bool__(self) -> bool:
        raise TypeError("simjit dataframe Expr cannot be used as a Python truth value")

    def __add__(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.Add, self, _coerce_value_expr(other))

    def __radd__(self, other: ValueLike) -> Expr:
        return _coerce_value_expr(other).__add__(self)

    def __sub__(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.Sub, self, _coerce_value_expr(other))

    def __rsub__(self, other: ValueLike) -> Expr:
        return _coerce_value_expr(other).__sub__(self)

    def __mul__(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.Mul, self, _coerce_value_expr(other))

    def __rmul__(self, other: ValueLike) -> Expr:
        return _coerce_value_expr(other).__mul__(self)

    def __truediv__(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.Div, self, _coerce_value_expr(other))

    def __rtruediv__(self, other: ValueLike) -> Expr:
        return _coerce_value_expr(other).__truediv__(self)

    def __neg__(self) -> Expr:
        return arith_unary(ir.ArithUnaryOp.Negate, self)

    def __abs__(self) -> Expr:
        return arith_unary(ir.ArithUnaryOp.Abs, self)

    def __lt__(self, other: ValueLike) -> Expr:
        return cmp(ir.CompareOp.Less, self, _coerce_value_expr(other))

    def __le__(self, other: ValueLike) -> Expr:
        return cmp(ir.CompareOp.LessEqual, self, _coerce_value_expr(other))

    def __gt__(self, other: ValueLike) -> Expr:
        return cmp(ir.CompareOp.Greater, self, _coerce_value_expr(other))

    def __ge__(self, other: ValueLike) -> Expr:
        return cmp(ir.CompareOp.GreaterEqual, self, _coerce_value_expr(other))

    def __eq__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: ValueLike
    ) -> Expr:
        return cmp(ir.CompareOp.Equal, self, _coerce_value_expr(other))

    def __ne__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, other: ValueLike
    ) -> Expr:
        return cmp(ir.CompareOp.NotEqual, self, _coerce_value_expr(other))

    def sum(
        self,
        where: Expr | None = None,
        *,
        group_idx: ValueLike | None = None,
        table: str | None = None,
    ) -> FrozenExpr:
        return _arith_agg_or_grouped(
            ir.ArithBinaryOp.Add, self, cond=where, group_idx=group_idx, table=table
        )

    def product(
        self,
        where: Expr | None = None,
        *,
        group_idx: ValueLike | None = None,
        table: str | None = None,
    ) -> FrozenExpr:
        return _arith_agg_or_grouped(
            ir.ArithBinaryOp.Mul, self, cond=where, group_idx=group_idx, table=table
        )

    def min(
        self,
        where: Expr | None = None,
        *,
        group_idx: ValueLike | None = None,
        table: str | None = None,
    ) -> FrozenExpr:
        return _arith_agg_or_grouped(
            ir.ArithBinaryOp.Min, self, cond=where, group_idx=group_idx, table=table
        )

    def max(
        self,
        where: Expr | None = None,
        *,
        group_idx: ValueLike | None = None,
        table: str | None = None,
    ) -> FrozenExpr:
        return _arith_agg_or_grouped(
            ir.ArithBinaryOp.Max, self, cond=where, group_idx=group_idx, table=table
        )

    def umin(
        self,
        where: Expr | None = None,
        *,
        group_idx: ValueLike | None = None,
        table: str | None = None,
    ) -> FrozenExpr:
        return _arith_agg_or_grouped(
            ir.ArithBinaryOp.UMin, self, cond=where, group_idx=group_idx, table=table
        )

    def umax(
        self,
        where: Expr | None = None,
        *,
        group_idx: ValueLike | None = None,
        table: str | None = None,
    ) -> FrozenExpr:
        return _arith_agg_or_grouped(
            ir.ArithBinaryOp.UMax, self, cond=where, group_idx=group_idx, table=table
        )

    def bit_and(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.And, self, _coerce_value_expr(other))

    def bit_or(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.Or, self, _coerce_value_expr(other))

    def bit_xor(self, other: ValueLike) -> Expr:
        return arith_binary(ir.ArithBinaryOp.Xor, self, _coerce_value_expr(other))

    def shift_left(self, other: ValueLike) -> Expr:
        return arith_binary(
            ir.ArithBinaryOp.ShiftLeftLogical, self, _coerce_value_expr(other)
        )

    def shift_right_logical(self, other: ValueLike) -> Expr:
        return arith_binary(
            ir.ArithBinaryOp.ShiftRightLogical, self, _coerce_value_expr(other)
        )

    def shift_right_arithmetic(self, other: ValueLike) -> Expr:
        return arith_binary(
            ir.ArithBinaryOp.ShiftRightArith, self, _coerce_value_expr(other)
        )

    def popcnt(self) -> Expr:
        return arith_unary(ir.ArithUnaryOp.Popcount, self)

    def lzcnt(self) -> Expr:
        return arith_unary(ir.ArithUnaryOp.Lzcnt, self)

    def tzcnt(self) -> Expr:
        return arith_unary(ir.ArithUnaryOp.Tzcnt, self)

    def cast(self, dt: ir.ScalarType) -> Expr:
        return cast(self, dt)

    def year(self) -> Expr:
        return year(self)

    def month(self) -> Expr:
        return month(self)

    def day(self) -> Expr:
        return day(self)

    def hour(self) -> Expr:
        return hour(self)

    def minute(self) -> Expr:
        return minute(self)

    def second(self) -> Expr:
        return second(self)

    def day_of_week(self) -> Expr:
        return day_of_week(self)

    def log2(self) -> Expr:
        return log2(self)

    def log2_no_zero(self) -> Expr:
        return log2_no_zero(self)

    def byteswap(self) -> Expr:
        return byteswap(self)

    def bit_floor(self) -> Expr:
        return bit_floor(self)

    def bit_ceil(self) -> Expr:
        return bit_ceil(self)

    def coalesce(self, *others: ValueLike) -> Expr:
        return coalesce(self, *others)

    def nullif(self, other: ValueLike) -> Expr:
        return nullif(self, other)

    def is_null(self) -> Expr:
        return is_null(self)

    def is_not_null(self) -> Expr:
        return is_not_null(self)

    def __and__(self, other: PredicateLike) -> Expr:
        return predicate_binary(
            ir.PredicateBinaryOp.And, self, _coerce_predicate_expr(other)
        )

    def __rand__(self, other: PredicateLike) -> Expr:
        return _coerce_predicate_expr(other).__and__(self)

    def __or__(self, other: PredicateLike) -> Expr:
        return predicate_binary(
            ir.PredicateBinaryOp.Or, self, _coerce_predicate_expr(other)
        )

    def __ror__(self, other: PredicateLike) -> Expr:
        return _coerce_predicate_expr(other).__or__(self)

    def __xor__(self, other: PredicateLike) -> Expr:
        return predicate_binary(
            ir.PredicateBinaryOp.Xor, self, _coerce_predicate_expr(other)
        )

    def __rxor__(self, other: PredicateLike) -> Expr:
        return _coerce_predicate_expr(other).__xor__(self)

    def __invert__(self) -> Expr:
        return predicate_not(self)

    def ifelse(self, truthy: ValueLike, falsy: ValueLike) -> Expr:
        return select(self, truthy, falsy)

    def count(
        self, *, group_idx: ValueLike | None = None, table: str | None = None
    ) -> FrozenExpr:
        return _count_if_or_grouped(self, group_idx=group_idx, table=table)


class FrozenExpr(Expr):
    pass


PredicateLike: TypeAlias = ir.Expr | bool | Expr
ValueLike: TypeAlias = ir.Expr | int | float | Expr
OutputValue: TypeAlias = Expr | ir.Expr | int | float | bool


class Table:
    """Schema object whose attributes expose symbolic columns."""

    def __init__(self, schema: Mapping[str, ir.ScalarType | None]):
        self.schema: dict[str, ir.ScalarType | None] = dict(schema)

    def __getattr__(self, name: str) -> Expr:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self.column(name)
        except KeyError as exc:
            raise AttributeError(f"unknown column {name!r}") from exc

    def column(self, name: str) -> Expr:
        if name not in self.schema:
            raise KeyError(f"unknown column {name!r}")
        dt = self.schema[name]
        return load(name, dt)


class Program:
    """High-level immutable program description."""

    def __init__(self, outputs: Mapping[str, OutputValue]):
        if not outputs:
            raise ValueError("outputs can't be empty")
        self.outputs: dict[str, OutputValue] = dict(outputs)

    def to_dsl(self) -> list[tuple[str, ir.Expr]]:
        lowered: list[tuple[str, ir.Expr]] = []
        for name, expr in self.outputs.items():
            if isinstance(expr, Expr):
                lowered.append((name, expr.to_dsl()))
                continue
            if isinstance(expr, ir.Expr):
                lowered.append((name, expr))
                continue
            lowered.append((name, const(expr).to_dsl()))
        return lowered

    def compile(self) -> NoReturn:
        raise NotImplementedError(
            "Program.compile is not implemented for dataframe API v1"
        )


def _merge_named_mapping(
    name: str,
    mapping: object | None,
    kwargs: Mapping[str, ir.ScalarType | None],
) -> dict[str, ir.ScalarType | None]:
    if mapping is None:
        return dict(kwargs)
    if kwargs:
        raise TypeError(
            f"{name} accepts either a mapping or keyword arguments, not both"
        )
    pyarrow_schema = pyarrow_integration.maybe_schema_from_pyarrow(mapping)
    if pyarrow_schema is not None:
        result: dict[str, ir.ScalarType | None] = {}
        result.update(pyarrow_schema)
        return result
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} expected a mapping or PyArrow schema")
    typed_mapping = typing_cast(Mapping[str, ir.ScalarType | None], mapping)
    return dict(typed_mapping)


def table(
    schema: object | None = None,
    **kwargs: ir.ScalarType | None,
) -> Table:
    return Table(_merge_named_mapping("table", schema, kwargs))


def col(name: str, dt: ir.ScalarType | None = None) -> Expr:
    return load(name, dt)


def _normalize_program_outputs(
    outputs: (
        Mapping[str, OutputValue]
        | list[OutputValue]
        | tuple[OutputValue, ...]
        | OutputValue
    ),
) -> dict[str, OutputValue]:
    if isinstance(outputs, Mapping):
        return dict(outputs)
    if isinstance(outputs, (list, tuple)):
        return {f"result_{i}": expr for i, expr in enumerate(outputs)}
    return {"result_0": outputs}


def query(
    outputs: (
        Mapping[str, OutputValue]
        | list[OutputValue]
        | tuple[OutputValue, ...]
        | OutputValue
        | None
    ) = None,
    **kwargs: OutputValue,
) -> Program:
    if outputs is None:
        return Program(dict(kwargs))
    if kwargs:
        raise TypeError("query accepts either outputs or keyword outputs, not both")
    if isinstance(outputs, Mapping):
        return Program(outputs)
    return Program(_normalize_program_outputs(outputs))
