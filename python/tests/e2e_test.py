# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

import array
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import simjit.dataframe as sdf
import simjit.ir as ir

import simjit as sj
from simjit import _simjit as simjit_ext

try:
    import pyarrow as pa
except ImportError:
    pa = None


def assert_array_equal(actual: object, expected: object):
    np.testing.assert_array_equal(actual, expected)


def assert_close(actual, expected, rtol: float = 1e-6):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=0)


def assert_raises(exc_type, fn, contains=None):
    with pytest.raises(exc_type) as exc_info:
        fn()
    exc = exc_info.value
    if contains is not None:
        assert contains in str(exc), f"expected {contains!r} in {exc!r}"
    return exc


def run_ir(buffers, outputs, n):
    normalized = {}
    for name, handle in buffers.items():
        buf = handle.buf
        if handle.ty.name == "timestamp64" and isinstance(buf, np.ndarray):
            if np.issubdtype(buf.dtype, np.datetime64):
                buf = buf.view(np.int64)
        normalized[name] = ir.BufferHandle(
            ty=handle.ty,
            buf=buf,
            length=handle.length,
            aligned=handle.aligned,
            bitpacked=handle.bitpacked,
            null=handle.null,
        )
    simjit_ext.run_native(normalized, list(outputs.items()), n)


def run_program(buffers, program, n):
    simjit_ext.run_native(buffers, program.to_dsl(), n)


def run_program_raw(buffers, program, n):
    run_ir(buffers, dict(program.to_dsl()), n)


def popcount_i32(x):
    return (
        np.unpackbits(x.astype(np.uint32).view(np.uint8), bitorder="little")
        .reshape(len(x), 32)
        .sum(axis=1)
        .astype(np.int32)
    )


def lzcnt_i32(x):
    out = np.empty(len(x), dtype=np.int32)
    for i, value in enumerate(x.astype(np.uint32)):
        out[i] = 32 if value == 0 else 32 - int(value).bit_length()
    return out


def tzcnt_i32(x):
    out = np.empty(len(x), dtype=np.int32)
    for i, value in enumerate(x.astype(np.uint32)):
        unsigned = int(value)
        out[i] = 32 if unsigned == 0 else (unsigned & -unsigned).bit_length() - 1
    return out


def test_dsl_store_const():
    arr = np.zeros(100, dtype=np.int32)
    run_ir(
        {"a": ir.BufferHandle(ty=ir.I32, buf=arr)},
        {
            "a": ir.StoreExpr(
                dt=ir.I32,
                value=ir.ConstExpr(dt=ir.I32, value=42),
                kind=ir.LoadStoreKind.Unaligned,
                cond=None,
            )
        },
        100,
    )
    assert_array_equal(arr, np.full(100, 42, dtype=np.int32))


def test_dsl_store_const_no_store():
    arr = np.zeros(100, dtype=np.int32)
    run_ir(
        {"a": ir.BufferHandle(ty=ir.I32, buf=arr)},
        {"a": ir.ConstExpr(dt=ir.I32, value=42)},
        100,
    )
    assert_array_equal(arr, np.full(100, 42, dtype=np.int32))


def test_program_typed_arithmetic_chain():
    a = np.arange(1000, dtype=np.int32)
    b = np.arange(1000, dtype=np.int32) + 3
    c = np.arange(1000, dtype=np.int32) * 2
    d = np.arange(1000, dtype=np.int32) - 5
    result = np.zeros(1000, dtype=np.int32)

    prog = sj.query(
        {
            "result": (sj.col("a", sj.I32) + sj.col("b", sj.I32))
            * (sj.col("c", sj.I32) - sj.col("d", sj.I32))
        }
    )
    run_program(
        {
            "a": ir.BufferHandle(ty=ir.I32, buf=a),
            "b": ir.BufferHandle(ty=ir.I32, buf=b),
            "c": ir.BufferHandle(ty=ir.I32, buf=c),
            "d": ir.BufferHandle(ty=ir.I32, buf=d),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        prog,
        len(a),
    )
    assert_array_equal(result, (a + b) * (c - d))


def test_program_typed_bit_manipulation_chain():
    a = np.random.randint(0, 1024, 256, dtype=np.int16)
    b = np.random.randint(0, 1024, 256, dtype=np.int16)
    c = np.random.randint(0, 1024, 256, dtype=np.int16)
    result = np.zeros(256, dtype=np.int16)

    expr = (
        sj.col("a", sj.I16)
        .bit_and(0xFF)
        .bit_or(sj.col("b", sj.I16).shift_left(2))
        .bit_xor(sj.col("c", sj.I16))
    )
    run_program(
        {
            "a": ir.BufferHandle(ty=ir.I16, buf=a),
            "b": ir.BufferHandle(ty=ir.I16, buf=b),
            "c": ir.BufferHandle(ty=ir.I16, buf=c),
            "result": ir.BufferHandle(ty=ir.I16, buf=result),
        },
        sj.query({"result": expr}),
        len(a),
    )
    expected = ((a & np.int16(0xFF)) | (b << np.int16(2))) ^ c
    assert_array_equal(result, expected.astype(np.int16))


def test_program_typed_unary_mix():
    x = np.random.randint(-1000, 1000, 256, dtype=np.int32)
    y = np.random.randint(0, 1000, 256, dtype=np.int32)
    z = np.random.randint(0, 1000, 256, dtype=np.int32)
    result = np.zeros(256, dtype=np.int32)

    expr = (
        abs(-sj.col("x", sj.I32))
        + sj.col("y", sj.I32).popcnt()
        + sj.col("z", sj.I32).lzcnt()
    )
    run_program(
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "z": ir.BufferHandle(ty=ir.I32, buf=z),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        sj.query({"result": expr}),
        len(x),
    )
    expected = (np.abs(-x) + popcount_i32(y) + lzcnt_i32(z)).astype(np.int32)
    assert_array_equal(result, expected)


def test_program_tzcnt():
    value = np.array([0, 1, 2, 3, 4, 8, 12, -1, -(1 << 31)], dtype=np.int32)
    result = np.zeros(len(value), dtype=np.int32)
    run_program(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        sj.query({"result": sj.col("value", sj.I32).tzcnt()}),
        len(value),
    )
    assert_array_equal(result, tzcnt_i32(value))


def test_program_explicit_casts():
    x = np.arange(128, dtype=np.int16)
    y = np.arange(128, dtype=np.int16) * 3
    k = np.full(128, 7, dtype=np.int32)
    result = np.zeros(128, dtype=np.int32)

    expr = sj.col("x", sj.I16).cast(sj.I32) + (
        sj.col("y", sj.I16).cast(sj.I32) * sj.col("k", sj.I32)
    )
    run_program(
        {
            "x": ir.BufferHandle(ty=ir.I16, buf=x),
            "y": ir.BufferHandle(ty=ir.I16, buf=y),
            "k": ir.BufferHandle(ty=ir.I32, buf=k),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        sj.query({"result": expr}),
        len(x),
    )
    assert_array_equal(result, x.astype(np.int32) + y.astype(np.int32) * k)


def test_program_predicates_and_ifelse():
    a = np.random.randint(-50, 150, 1000, dtype=np.int32)
    b = np.random.randint(-50, 150, 1000, dtype=np.int32)
    flag = np.zeros(1000, dtype=np.int32)
    mask = np.zeros(1000, dtype=np.bool_)

    expr = ((sj.col("a", sj.I32) > 0) & (sj.col("b", sj.I32) < 100)).ifelse(
        sj.i32(1), sj.i32(0)
    )
    pred = (sj.col("a", sj.I32) > 0) & (sj.col("b", sj.I32) < 100)
    run_program(
        {
            "a": ir.BufferHandle(ty=ir.I32, buf=a),
            "b": ir.BufferHandle(ty=ir.I32, buf=b),
            "flag": ir.BufferHandle(ty=ir.I32, buf=flag),
            "mask": ir.BufferHandle(ty=ir.I1, buf=mask, bitpacked=False),
        },
        sj.query({"flag": expr, "mask": pred}),
        len(a),
    )
    expected_mask = (a > 0) & (b < 100)
    assert_array_equal(flag, expected_mask.astype(np.int32))
    assert_array_equal(mask, expected_mask)


def test_program_float_cast_and_fpclass():
    x = np.array([0, 1, 2, 3, -4, 5], dtype=np.int32)
    f = np.array([0.0, math.inf, -math.inf, math.nan, 1.5, 2.5], dtype=np.float32)
    casted = np.zeros(len(x), dtype=np.float32)
    flags = np.zeros(len(f), dtype=np.bool_)

    run_program_raw(
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "f": ir.BufferHandle(ty=ir.F32, buf=f),
            "casted": ir.BufferHandle(ty=ir.F32, buf=casted),
            "flags": ir.BufferHandle(ty=ir.I1, buf=flags, bitpacked=False),
        },
        sj.query(
            {
                "casted": sj.float_cast(sj.F32, sj.load("x", sj.I32)),
                "flags": sj.fpclass(
                    sj.load("f", sj.F32),
                    ir.FpClassFlags.Infinite
                    | ir.FpClassFlags.Nan
                    | ir.FpClassFlags.Zero,
                ),
            }
        ),
        len(x),
    )
    assert_close(casted, x.astype(np.float32))
    expected_flags = np.isinf(f) | np.isnan(f) | (f == 0)
    assert_array_equal(flags, expected_flags)


def test_nullable_dsl_casts_preserve_null_masks():
    x16 = np.array([-2, -1, 3, 4], dtype=np.int16)
    x32 = np.array([-2, -1, 3, 4], dtype=np.int32)
    x_null = np.array([False, True, False, True], dtype=np.bool_)
    int_out = np.zeros(len(x16), dtype=np.int32)
    int_null = np.zeros(len(x16), dtype=np.bool_)
    float_out = np.zeros(len(x32), dtype=np.float32)
    float_null = np.zeros(len(x32), dtype=np.bool_)

    run_ir(
        {
            "x16": ir.BufferHandle(
                ty=ir.I16,
                buf=x16,
                null=ir.NullEncoding(kind="mask_bool", buf=x_null),
            ),
            "x32": ir.BufferHandle(
                ty=ir.I32,
                buf=x32,
                null=ir.NullEncoding(kind="mask_bool", buf=x_null),
            ),
            "int_out": ir.BufferHandle(
                ty=ir.I32,
                buf=int_out,
                null=ir.NullEncoding(kind="mask_bool", buf=int_null),
            ),
            "float_out": ir.BufferHandle(
                ty=ir.F32,
                buf=float_out,
                null=ir.NullEncoding(kind="mask_bool", buf=float_null),
            ),
        },
        {
            "int_out": ir.StoreExpr(
                dt=ir.I32,
                value=ir.IntCastExpr(
                    dt=ir.I32,
                    kind=ir.IntCastKind.Signed,
                    arg=ir.LoadExpr(dt=ir.I16, name="x16"),
                ),
            ),
            "float_out": ir.StoreExpr(
                dt=ir.F32,
                value=ir.FloatCastExpr(
                    dt=ir.F32,
                    arg=ir.LoadExpr(dt=ir.I32, name="x32"),
                ),
            ),
        },
        len(x16),
    )

    assert_array_equal(int_out, x16.astype(np.int32))
    assert_array_equal(int_null, x_null)
    assert_close(float_out, x32.astype(np.float32))
    assert_array_equal(float_null, x_null)


def test_program_bitcast():
    raw = np.array([0x3F800000, 0x40000000, 0x40400000, 0x40800000], dtype=np.int32)
    result = np.zeros(len(raw), dtype=np.float32)
    run_program(
        {
            "raw": ir.BufferHandle(ty=ir.I32, buf=raw),
            "result": ir.BufferHandle(ty=ir.F32, buf=result),
        },
        sj.query({"result": sj.bitcast(sj.F32, sj.load("raw", sj.I32))}),
        len(raw),
    )
    assert_close(result, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    raw_unsigned = np.array([0, 1, 0x7FFFFFFF, 0x80000000], dtype=np.uint32)
    reinterpreted = np.zeros(len(raw_unsigned), dtype=np.int32)
    run_program(
        {
            "raw": ir.BufferHandle(ty=ir.U32, buf=raw_unsigned),
            "result": ir.BufferHandle(ty=ir.I32, buf=reinterpreted),
        },
        sj.query({"result": sj.bitcast(sj.I32, sj.load("raw", sj.U32))}),
        len(raw_unsigned),
    )
    assert_array_equal(reinterpreted.view(np.uint32), raw_unsigned)


def test_dsl_cmp():
    a = np.random.randint(0, 1000, 1000, dtype=np.int32)
    b = np.random.randint(0, 1000, 1000, dtype=np.int32)
    result = np.zeros(1000, dtype=np.bool_)
    run_ir(
        {
            "a": ir.BufferHandle(ty=ir.I32, buf=a),
            "b": ir.BufferHandle(ty=ir.I32, buf=b),
            "result": ir.BufferHandle(ty=ir.I1, buf=result, bitpacked=False),
        },
        {
            "result": ir.StoreExpr(
                dt=ir.I1,
                value=ir.CompareExpr(
                    dt=ir.I1,
                    op=ir.CompareOp.Less,
                    lhs=ir.LoadExpr(dt=ir.I32, name="a"),
                    rhs=ir.LoadExpr(dt=ir.I32, name="b"),
                ),
            )
        },
        1000,
    )
    assert_array_equal(result, a < b)


def test_dsl_mixed_sign_sensitive_ops_fail_in_binding():
    a = np.array([1, 2, 3], dtype=np.uint32)
    b = np.array([1, 1, 1], dtype=np.int32)
    out_div = np.zeros(3, dtype=np.uint32)
    out_cmp = np.zeros(3, dtype=np.bool_)

    exc = assert_raises(
        ValueError,
        lambda: run_ir(
            {
                "a": ir.BufferHandle(ty=ir.U32, buf=a),
                "b": ir.BufferHandle(ty=ir.I32, buf=b),
                "out_div": ir.BufferHandle(ty=ir.U32, buf=out_div),
            },
            {
                "out_div": ir.ArithBinaryExpr(
                    dt=ir.U32,
                    op=ir.ArithBinaryOp.Div,
                    lhs=ir.LoadExpr(dt=ir.U32, name="a"),
                    rhs=ir.LoadExpr(dt=ir.I32, name="b"),
                )
            },
            len(a),
        ),
    )
    assert "incompatible numeric types at arithmetic: u32 vs i32" in str(exc)

    exc = assert_raises(
        ValueError,
        lambda: run_ir(
            {
                "a": ir.BufferHandle(ty=ir.U32, buf=a),
                "b": ir.BufferHandle(ty=ir.I32, buf=b),
                "out_cmp": ir.BufferHandle(ty=ir.I1, buf=out_cmp, bitpacked=False),
            },
            {
                "out_cmp": ir.CompareExpr(
                    dt=ir.I1,
                    op=ir.CompareOp.Greater,
                    lhs=ir.LoadExpr(dt=ir.U32, name="a"),
                    rhs=ir.LoadExpr(dt=ir.I32, name="b"),
                    unsigned=False,
                )
            },
            len(a),
        ),
    )
    assert "incompatible numeric types at compare: u32 vs i32" in str(exc)


def test_dsl_load_splat():
    scalar = np.array([7], dtype=np.int32)
    result = np.zeros(64, dtype=np.int32)
    run_ir(
        {
            "scalar": ir.BufferHandle(ty=ir.I32, buf=scalar),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        {"result": ir.LoadSplatExpr(dt=ir.I32, name="scalar")},
        len(result),
    )
    assert_array_equal(result, np.full(len(result), 7, dtype=np.int32))


def test_dsl_nullable_sentinel_load_splat():
    scalar = np.array([-1], dtype=np.int32)
    result = np.zeros(8, dtype=np.int32)
    result_null = np.zeros(8, dtype=np.bool_)
    run_ir(
        {
            "scalar": ir.BufferHandle(
                ty=ir.I32,
                buf=scalar,
                null=ir.NullEncoding(kind="sentinel", sentinel=-1),
            ),
            "result": ir.BufferHandle(
                ty=ir.I32,
                buf=result,
                null=ir.NullEncoding(kind="mask_bool", buf=result_null),
            ),
        },
        {"result": ir.LoadSplatExpr(dt=ir.I32, name="scalar")},
        len(result),
    )
    assert_array_equal(result, np.full(len(result), -1, dtype=np.int32))
    assert_array_equal(result_null, np.ones(len(result), dtype=np.bool_))


def test_dsl_nullable_unsigned_compare():
    lhs = np.array([0, 1 << 31, 7, 9], dtype=np.uint32)
    rhs = np.array([1, 0, 8, 9], dtype=np.uint32)
    lhs_null = np.array([False, False, True, False], dtype=np.bool_)
    result = np.zeros(4, dtype=np.bool_)
    result_null = np.zeros(4, dtype=np.bool_)
    run_ir(
        {
            "lhs": ir.BufferHandle(
                ty=ir.U32,
                buf=lhs,
                null=ir.NullEncoding(kind="mask_bool", buf=lhs_null),
            ),
            "rhs": ir.BufferHandle(ty=ir.U32, buf=rhs),
            "result": ir.BufferHandle(
                ty=ir.I1,
                buf=result,
                bitpacked=False,
                null=ir.NullEncoding(kind="mask_bool", buf=result_null),
            ),
        },
        {
            "result": ir.CompareExpr(
                dt=ir.I1,
                op=ir.CompareOp.Greater,
                lhs=ir.LoadExpr(dt=ir.U32, name="lhs"),
                rhs=ir.LoadExpr(dt=ir.U32, name="rhs"),
                unsigned=True,
            )
        },
        len(lhs),
    )
    assert_array_equal(result, lhs > rhs)
    assert_array_equal(result_null, lhs_null)


def test_program_raw_load_splat_accepts_one_element_input_array():
    scalar = np.array([11], dtype=np.int32)
    result = np.zeros(32, dtype=np.int32)
    run_program_raw(
        {
            "scalar": ir.BufferHandle(ty=ir.I32, buf=scalar),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        sj.query(result=sj.load_splat("scalar", sj.I32)),
        len(result),
    )
    assert_array_equal(result, np.full(len(result), 11, dtype=np.int32))


def test_dsl_gather():
    idx = np.array([3, 1, 0, 2, 3, 1], dtype=np.int32)
    src = np.array([10, 20, 30, 40], dtype=np.int32)
    result = np.zeros(len(idx), dtype=np.int32)
    run_ir(
        {
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "src": ir.BufferHandle(ty=ir.I32, buf=src),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        {
            "result": ir.GatherExpr(
                dt=ir.I32,
                idx=ir.LoadExpr(dt=ir.I32, name="idx"),
                name="src",
            )
        },
        len(idx),
    )
    assert_array_equal(result, src[idx])


def test_program_raw_gather_with_public_builder():
    idx = np.array([2, 0, 3, 1], dtype=np.int32)
    src = np.array([100, 200, 300, 400], dtype=np.int32)
    result = np.zeros(len(idx), dtype=np.int32)
    run_program_raw(
        {
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "src": ir.BufferHandle(ty=ir.I32, buf=src),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
        },
        sj.query(result=sj.gather(sj.load("idx", sj.I32), "src", sj.I32)),
        len(idx),
    )
    assert_array_equal(result, src[idx])


def test_dsl_index():
    result = np.zeros(32, dtype=np.int32)
    run_ir(
        {"result": ir.BufferHandle(ty=ir.I32, buf=result)},
        {"result": ir.IndexExpr(dt=ir.I32)},
        len(result),
    )
    assert_array_equal(result, np.arange(len(result), dtype=np.int32))


def test_program_aggregates_and_count():
    value = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    pred = np.array([True, False, True, False, True], dtype=np.bool_)
    total = np.zeros(1, dtype=np.int64)
    total_if = np.zeros(1, dtype=np.int64)
    product = np.zeros(1, dtype=np.int64)
    count = np.zeros(1, dtype=np.int64)

    run_program(
        {
            "value": ir.BufferHandle(ty=ir.I64, buf=value),
            "pred": ir.BufferHandle(ty=ir.I1, buf=pred, bitpacked=False),
            "sum": ir.BufferHandle(ty=ir.I64, buf=total),
            "sum_if": ir.BufferHandle(ty=ir.I64, buf=total_if),
            "product": ir.BufferHandle(ty=ir.I64, buf=product),
            "count": ir.BufferHandle(ty=ir.I64, buf=count),
        },
        sj.query(
            {
                "sum": sj.col("value", sj.I64).sum(),
                "sum_if": sj.col("value", sj.I64).sum(where=sj.col("pred", sj.I1)),
                "product": sj.col("value", sj.I64).product(),
                "count": sj.col("pred", sj.I1).count(),
            }
        ),
        len(value),
    )
    assert total[0] == value.sum()
    assert total_if[0] == value[pred].sum()
    assert product[0] == value.prod()
    assert count[0] == pred.sum()


def test_dsl_grouped_aggregate():
    value = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    idx = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    table = np.zeros(2, dtype=np.int32)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "table": ir.BufferHandle(ty=ir.I32, buf=table),
        },
        {
            "grouped": ir.GroupedArithAggExpr(
                dt=ir.I32,
                op=ir.ArithBinaryOp.Add,
                arg=ir.LoadExpr(dt=ir.I32, name="value"),
                idx=ir.LoadExpr(dt=ir.I32, name="idx"),
                table="table",
            )
        },
        len(value),
    )
    assert_array_equal(table, np.array([9, 12], dtype=np.int32))


def test_dataframe_group_idx_aggregates():
    value = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    idx = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    pred = np.array([True, False, True, True, False, True], dtype=np.bool_)
    sum_table = np.zeros(2, dtype=np.int32)
    count_table = np.zeros(2, dtype=np.int64)
    t = sj.table({"value": sj.I32, "idx": sj.I32, "pred": sj.I1})

    run_program(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "pred": ir.BufferHandle(ty=ir.I1, buf=pred, bitpacked=False),
            "sum_table": ir.BufferHandle(ty=ir.I32, buf=sum_table),
            "count_table": ir.BufferHandle(ty=ir.I64, buf=count_table),
        },
        sj.query(
            {
                "grouped_sum": t.value.sum(group_idx=t.idx, table="sum_table"),
                "grouped_count": t.pred.count(group_idx=t.idx, table="count_table"),
            }
        ),
        len(value),
    )

    assert_array_equal(sum_table, np.array([9, 12], dtype=np.int32))
    assert_array_equal(count_table, np.array([2, 2], dtype=np.int64))


def test_dsl_scatter():
    value = np.array([10, 20, 30, 40], dtype=np.int32)
    idx = np.array([3, 1, 0, 2], dtype=np.int32)
    dst = np.zeros(4, dtype=np.int32)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "dst": ir.BufferHandle(ty=ir.I32, buf=dst),
        },
        {
            "dst": ir.ScatterExpr(
                dt=ir.I32,
                value=ir.LoadExpr(dt=ir.I32, name="value"),
                idx=ir.LoadExpr(dt=ir.I32, name="idx"),
            )
        },
        len(value),
    )
    assert_array_equal(dst, np.array([30, 20, 40, 10], dtype=np.int32))


def test_program_raw_scatter_with_public_builder():
    value = np.array([7, 8, 9, 10], dtype=np.int32)
    idx = np.array([1, 3, 0, 2], dtype=np.int32)
    dst = np.zeros(4, dtype=np.int32)
    run_program(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "dst": ir.BufferHandle(ty=ir.I32, buf=dst),
        },
        sj.query(dst=sj.scatter(sj.load("value", sj.I32), sj.load("idx", sj.I32))),
        len(value),
    )
    assert_array_equal(dst, np.array([9, 7, 10, 8], dtype=np.int32))


def test_dsl_pack():
    value = np.array([10, 20, 30, 40, 50], dtype=np.int32)
    pred = np.array([True, False, True, False, True], dtype=np.bool_)
    packed = np.zeros(len(value), dtype=np.int32)
    packed_size = np.zeros(1, dtype=np.int64)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value),
            "pred": ir.BufferHandle(ty=ir.I1, buf=pred, bitpacked=False),
            "packed": ir.BufferHandle(ty=ir.I32, buf=packed),
            "packed_size": ir.BufferHandle(ty=ir.I64, buf=packed_size),
        },
        {
            "packed": ir.PackExpr(
                dt=ir.I32,
                value=ir.LoadExpr(dt=ir.I32, name="value"),
                cond=ir.LoadExpr(dt=ir.I1, name="pred"),
                dst_size="packed_size",
            )
        },
        len(value),
    )
    expected = value[pred]
    assert packed_size[0] == len(expected)
    assert_array_equal(packed[: len(expected)], expected)


def test_missing_input_buffer_fails():
    result = np.zeros(8, dtype=np.int32)

    def run():
        run_program(
            {"result": ir.BufferHandle(ty=ir.I32, buf=result)},
            sj.query({"result": sj.col("a", sj.I32) + 1}),
            len(result),
        )

    assert_raises(ValueError, run, "argument a not found in schema")


def test_native_builder_missing_output_error_is_index_error():
    a = np.arange(8, dtype=np.int32)

    def run():
        run_program(
            {"a": ir.BufferHandle(ty=ir.I32, buf=a)},
            sj.query({"result": sj.col("a", sj.I32) + 1}),
            len(a),
        )

    assert_raises(IndexError, run, "buffer result is missing")


def test_excess_buffer_entries_are_ignored():
    a = np.arange(8, dtype=np.int32)
    result = np.zeros(8, dtype=np.int32)
    unused = np.zeros(8, dtype=np.int32)
    run_program(
        {
            "a": ir.BufferHandle(ty=ir.I32, buf=a),
            "result": ir.BufferHandle(ty=ir.I32, buf=result),
            "unused": ir.BufferHandle(ty=ir.I32, buf=unused),
        },
        sj.query({"result": sj.col("a", sj.I32) + 1}),
        len(a),
    )
    assert_array_equal(result, a + 1)
    assert_array_equal(unused, np.zeros_like(unused))


def test_program_select_accepts_expression_sequence():
    x = np.arange(8, dtype=np.int32)
    y = np.arange(8, dtype=np.int32) * 2
    result_0 = np.zeros(8, dtype=np.int32)
    result_1 = np.zeros(8, dtype=np.int32)
    run_program(
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "result_0": ir.BufferHandle(ty=ir.I32, buf=result_0),
            "result_1": ir.BufferHandle(ty=ir.I32, buf=result_1),
        },
        sj.query((sj.col("x", sj.I32) + 1, sj.col("y", sj.I32) - 2)),
        len(x),
    )
    assert_array_equal(result_0, x + 1)
    assert_array_equal(result_1, y - 2)


def test_program_select_accepts_single_expression():
    x = np.arange(8, dtype=np.int32)
    result_0 = np.zeros(8, dtype=np.int32)
    run_program(
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "result_0": ir.BufferHandle(ty=ir.I32, buf=result_0),
        },
        sj.query(sj.col("x", sj.I32) + 1),
        len(x),
    )
    assert_array_equal(result_0, x + 1)


def test_program_table_and_query_accept_kwargs():
    a = np.arange(16, dtype=np.int32)
    b = np.arange(16, dtype=np.int32) + 5
    sum_xy = np.zeros(16, dtype=np.int32)

    t = sj.table(a=sj.I32, b=sj.I32)
    run_program(
        {
            "a": ir.BufferHandle(ty=ir.I32, buf=a),
            "b": ir.BufferHandle(ty=ir.I32, buf=b),
            "sum_xy": ir.BufferHandle(ty=ir.I32, buf=sum_xy),
        },
        sj.query(sum_xy=t.a + t.b),
        len(a),
    )
    assert_array_equal(sum_xy, a + b)


def test_program_run_uses_binding_for_projection():
    program = sj.query(sum_xy=sj.col("x") + sj.col("y"))
    result = sj.run_program(
        program,
        {
            "x": np.array([1, 2, 3], dtype=np.int32),
            "y": np.array([10, 20, 30], dtype=np.int32),
        },
    )
    assert_array_equal(result["sum_xy"], np.array([11, 22, 33], dtype=np.int32))
    assert_array_equal(result.sum_xy, np.array([11, 22, 33], dtype=np.int32))


def test_program_run_returns_result_wrapper():
    program = sj.query(
        sum_xy=sj.col("x") + sj.col("y"), total=sj.col("x", sj.I64).sum()
    )
    result = sj.run_program(
        program,
        {
            "x": np.array([1, 2, 3], dtype=np.int64),
            "y": np.array([10, 20, 30], dtype=np.int64),
        },
    )
    assert isinstance(result, sj.Result)
    assert tuple(result.keys()) == ("sum_xy", "total")
    assert_array_equal(result.sum_xy, np.array([11, 22, 33], dtype=np.int64))
    assert result.total == 6
    assert_array_equal(
        result.to_dict()["sum_xy"], np.array([11, 22, 33], dtype=np.int64)
    )
    assert result.to_dict()["total"] == 6
    assert_raises(AttributeError, lambda: result.missing)


@pytest.fixture(
    params=("int32", "float32", "numpy-bool", "pyarrow-bool"),
    ids=("int32-add", "float32-add", "numpy-bool", "pyarrow-bitpacked-bool"),
)
def session_case(request):
    if request.param == "int32":
        dtype, values = np.int32, ([1, 2, 3], [10, 20, 30])
    elif request.param == "float32":
        dtype, values = np.float32, ([1.5, 2.5, 3.5], [10.0, 20.0, 30.0])
    else:
        dtype, values = np.bool_, ([True, False, True, True], [True, True, False, True])

    inputs = {
        "x": np.array(values[0], dtype=dtype),
        "y": np.array(values[1], dtype=dtype),
    }
    if request.param == "pyarrow-bool":
        arrow = pytest.importorskip("pyarrow")
        inputs = {name: arrow.array(value) for name, value in inputs.items()}

    if request.param.endswith("bool"):
        program = sj.query(result=sj.col("x", sj.I1) & sj.col("y", sj.I1))
        expected = np.logical_and(values[0], values[1])
        output_name = "result"
    else:
        program = sj.query(sum_xy=sj.col("x") + sj.col("y"))
        expected = np.add(values[0], values[1], dtype=dtype)
        output_name = "sum_xy"
    return program, inputs, output_name, expected


def test_program_run_reuses_explicit_session_cache(session_case):
    program, inputs, output_name, expected = session_case
    session = simjit_ext.Session()
    session.debug_options.capture_on_success = True

    first = sj.run_program(program, inputs, session=session)
    assert_array_equal(first[output_name], expected)
    first_stats = session.statistics()
    assert first_stats.function_count == 1
    assert first_stats.cache_hits == 0
    identifiers = session.function_identifiers()
    assert len(identifiers) == 1
    assert session.debug_snapshot.hir

    second = sj.run_program(program, inputs, session=session)
    assert_array_equal(second[output_name], expected)
    second_stats = session.statistics()
    assert second_stats.function_count == 1
    assert second_stats.cache_hits > first_stats.cache_hits
    assert session.function_identifiers() == identifiers

    assert session.release(identifiers[0]) is True
    assert session.statistics().function_count == 0

    third = sj.run_program(program, inputs, session=session)
    assert_array_equal(third[output_name], expected)
    assert session.statistics().function_count == 1


def test_session_clear_resets_cache_and_preserves_configuration(session_case):
    program, inputs, output_name, expected = session_case
    session = simjit_ext.Session()
    session.policy = simjit_ext.CompilePolicy.Scalar
    session.transformations = simjit_ext.CodeTransformations.No
    session.debug_options.capture_on_error = True

    result = sj.run_program(program, inputs, session=session)
    assert_array_equal(result[output_name], expected)
    assert session.statistics().function_count == 1

    session.clear()
    stats = session.statistics()
    assert stats.function_count == 0
    assert stats.cache_hits == 0
    assert stats.cache_misses == 0
    assert stats.compilation_attempts == 0
    assert session.policy == simjit_ext.CompilePolicy.Scalar
    assert session.transformations == simjit_ext.CodeTransformations.No
    assert session.debug_options.capture_on_error is True
    assert session.function_identifiers() == []


def test_native_prepared_kernel_lowers_unresolved_graph():
    x = np.array([1, 2, 3, 4], dtype=np.int16)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    pred = np.array([True, False, True, True], dtype=np.bool_)
    program = sj.query(
        total=sj.col("x") + sj.col("y"),
        picked=sj.col("pred", sj.I1).ifelse(sj.col("x"), sj.col("y")),
        filled=sj.coalesce(sj.col("x"), sj.col("y")),
        count_pos=(sj.col("y") > 15).count(),
    )

    expected = sj.run_program(program, {"x": x, "y": y, "pred": pred})
    session = simjit_ext.Session()
    prepared = session.prepare_program(
        program.to_dsl(), {"x": x, "y": y, "pred": pred}, "numpy"
    )
    assert prepared.identifier.startswith("native:")
    prepared.run()
    result = prepared.result()

    assert result["total"].dtype == np.int32
    assert result["picked"].dtype == np.int32
    assert result["filled"].dtype == np.int32
    assert_array_equal(result["total"], expected.total)
    assert_array_equal(result["picked"], expected.picked)
    assert_array_equal(result["filled"], expected.filled)
    assert result["count_pos"] == expected.count_pos


def test_native_session_run_reuses_unresolved_graph_cache():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    program = sj.query(result=sj.col("x") + sj.col("y"))
    outputs = program.to_dsl()
    session = simjit_ext.Session()

    result = session.run_program(outputs, {"x": x, "y": y}, "numpy")
    assert_array_equal(result["result"], x + y)
    first_stats = session.statistics()
    identifiers = session.function_identifiers()
    assert first_stats.function_count == 1
    assert first_stats.cache_hits == 0
    assert first_stats.cache_misses == 1
    assert len(identifiers) == 1
    assert identifiers[0].startswith("native:")

    result = session.run_program(outputs, {"x": x, "y": y}, "numpy")
    assert_array_equal(result["result"], x + y)
    second_stats = session.statistics()
    assert second_stats.function_count == 1
    assert second_stats.cache_hits > first_stats.cache_hits
    assert session.function_identifiers() == identifiers


def test_native_session_run_program_allocates_numpy_outputs():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    program = sj.query(
        result=sj.col("x") + sj.col("y"), total=sj.col("x", sj.I32).sum()
    )
    session = simjit_ext.Session()
    outputs = program.to_dsl()

    result = session.run_program(outputs, {"x": x, "y": y}, "numpy")

    assert_array_equal(result["result"], x + y)
    assert result["total"] == int(np.sum(x))
    first_stats = session.statistics()
    identifiers = session.function_identifiers()
    assert len(identifiers) == 1
    assert identifiers[0].startswith("native:")

    x2 = np.array([5, 6, 7, 8], dtype=np.int32)
    y2 = np.array([1, 2, 3, 4], dtype=np.int32)
    result = session.run_program(outputs, {"x": x2, "y": y2}, "numpy")

    assert_array_equal(result["result"], x2 + y2)
    assert result["total"] == int(np.sum(x2))
    second_stats = session.statistics()
    assert second_stats.function_count == 1
    assert second_stats.cache_hits > first_stats.cache_hits
    assert session.function_identifiers() == identifiers


def test_native_session_run_program_cache_misses_on_dtype_change():
    program = sj.query(result=sj.col("x") + 1)
    outputs = program.to_dsl()
    session = simjit_ext.Session()

    result32 = session.run_program(
        outputs, {"x": np.array([1, 2, 3], dtype=np.int32)}, "numpy"
    )
    assert_array_equal(result32["result"], np.array([2, 3, 4], dtype=np.int32))
    first_stats = session.statistics()
    assert first_stats.function_count == 1

    result64 = session.run_program(
        outputs, {"x": np.array([1, 2, 3], dtype=np.int64)}, "numpy"
    )
    assert_array_equal(result64["result"], np.array([2, 3, 4], dtype=np.int64))
    second_stats = session.statistics()
    assert second_stats.function_count == 2
    assert second_stats.cache_misses > first_stats.cache_misses


def test_native_session_prepare_program_reuses_outputs_and_invalidates():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    program = sj.query(
        result=sj.col("x") + sj.col("y"), total=sj.col("x", sj.I32).sum()
    )
    outputs = program.to_dsl()
    session = simjit_ext.Session()

    prepared = session.prepare_program(outputs, {"x": x, "y": y}, "numpy")
    assert prepared.identifier.startswith("native:")
    assert session.statistics().function_count == 1
    prepared.run()
    result = prepared.result()
    output_buffers = prepared.output_buffers()
    assert_array_equal(result["result"], x + y)
    assert result["total"] == int(np.sum(x))
    assert output_buffers["result"] is result["result"]
    assert output_buffers["total"].shape == (1,)
    assert output_buffers["total"][0] == int(np.sum(x))

    x[:] = [5, 6, 7, 8]
    y[:] = [1, 2, 3, 4]
    prepared.run()
    result = prepared.result()
    assert_array_equal(result["result"], x + y)
    assert result["total"] == int(np.sum(x))
    assert prepared.output_buffers()["result"] is output_buffers["result"]
    assert prepared.output_buffers()["total"] is output_buffers["total"]

    prepared.release_outputs()
    fresh = prepared.run_fresh()
    assert_array_equal(fresh["result"], x + y)
    assert fresh["total"] == int(np.sum(x))
    assert prepared.output_buffers()["result"] is fresh["result"]
    assert_raises(
        RuntimeError,
        prepared.run_fresh,
        contains="prepared outputs must be released",
    )
    prepared.release_outputs()
    fresh_again = prepared.run_fresh()
    assert_array_equal(fresh_again["result"], x + y)
    assert prepared.output_buffers()["result"] is fresh_again["result"]
    prepared.release_outputs()
    fresh_values = prepared.run_fresh_values()
    assert_array_equal(fresh_values[0], x + y)
    assert fresh_values[1] == int(np.sum(x))

    first_stats = session.statistics()
    prepared_2 = session.prepare_program(outputs, {"x": x, "y": y}, "numpy")
    second_stats = session.statistics()
    assert prepared_2.identifier == prepared.identifier
    assert second_stats.function_count == 1
    assert second_stats.cache_hits > first_stats.cache_hits

    assert session.release(prepared.identifier) is True
    assert_raises(
        RuntimeError, prepared.run, contains="prepared program was invalidated"
    )


def test_native_session_prepare_program_accepts_nullable_pyarrow_outputs(pa):
    x = pa.array([1, None, 3, 4], type=pa.int32())
    outputs = sj.query(result=sj.col("x", sj.I32)).to_dsl()
    prepared = simjit_ext.Session().prepare_program(outputs, {"x": x}, "pyarrow")

    prepared.run()
    result = prepared.result()

    assert result["result"].to_pylist() == [1, None, 3, 4]


def test_prepared_runner_rebinds_inputs_and_reuses_outputs():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    program = sj.query(result=sj.col("x") + sj.col("y"))
    session = sj.Session()

    runner = sj.prepare_program(program, {"x": x, "y": y}, session=session)
    first_stats = session.statistics()
    assert runner.run() is None
    output_array = runner.outputs.result
    assert_array_equal(output_array, x + y)
    assert runner.result().result is output_array

    x2 = np.array([5, 6, 7, 8], dtype=np.int32)
    y2 = np.array([1, 2, 3, 4], dtype=np.int32)
    assert runner.run({"x": x2, "y": y2}) is None
    second_stats = session.statistics()
    assert runner.outputs.result is output_array
    assert_array_equal(runner.outputs.result, x2 + y2)
    assert second_stats.compilation_attempts == first_stats.compilation_attempts
    assert second_stats.function_count == first_stats.function_count
    assert runner.identifier in session.function_identifiers()


def test_prepared_runner_exposes_scalar_output_buffers():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    program = sj.query(total=sj.col("x", sj.I32).sum())
    runner = sj.prepare_program(program, {"x": x})

    assert runner.outputs.total.shape == (1,)
    assert runner.run() is None
    assert runner.scalar("total") == int(np.sum(x))
    assert runner.scalars() == {"total": int(np.sum(x))}
    assert runner.result().total == int(np.sum(x))

    x[:] = [10, 20, 30, 40]
    assert runner.run() is None
    assert runner.outputs.total.shape == (1,)
    assert runner.scalar("total") == int(np.sum(x))


def test_resolver_identifier_is_stable_across_equivalent_python_graphs():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    session = sj.Session()

    runner_a = sj.prepare_program(
        sj.query(result=sj.col("x") + sj.col("y")), {"x": x, "y": y}, session=session
    )
    runner_b = sj.prepare_program(
        sj.query(result=sj.col("x") + sj.col("y")), {"x": x, "y": y}, session=session
    )
    runner_renamed = sj.prepare_program(
        sj.query(renamed=sj.col("a") + sj.col("b")), {"a": x, "b": y}, session=session
    )

    assert runner_a.identifier == runner_b.identifier
    assert runner_a.identifier == runner_renamed.identifier
    assert "result" not in runner_a.identifier
    assert session.statistics().function_count == 1

    runner_aliased = sj.prepare_program(
        sj.query(renamed=sj.col("a") + sj.col("a")), {"a": x}, session=session
    )
    assert runner_aliased.identifier != runner_a.identifier
    assert session.statistics().function_count == 2

    assert runner_a.run() is None
    assert runner_b.run() is None
    assert runner_renamed.run() is None
    assert_array_equal(runner_a.outputs.result, x + y)
    assert_array_equal(runner_b.outputs.result, x + y)
    assert_array_equal(runner_renamed.outputs.renamed, x + y)


def test_resolver_identifier_includes_compilation_configuration():
    x = np.array([1, 2, 3], dtype=np.int32)
    program = sj.query(result=sj.col("x") + 1)
    session = sj.Session()

    default_runner = sj.prepare_program(program, {"x": x}, session=session)
    session.policy = sj.CompilePolicy.Scalar
    scalar_runner = sj.prepare_program(program, {"x": x}, session=session)
    session.transformations = sj.CodeTransformations.No
    untransformed_runner = sj.prepare_program(program, {"x": x}, session=session)

    assert default_runner.identifier != scalar_runner.identifier
    assert scalar_runner.identifier != untransformed_runner.identifier
    assert session.statistics().function_count == 3
    assert "policy=" in default_runner.identifier
    assert "transformations=" in default_runner.identifier
    assert default_runner.run({"x": x}) is None


def test_prepared_runner_rejects_incompatible_rebinding_and_invalidates():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    program = sj.query(result=sj.col("x") + sj.col("y"))
    session = sj.Session()
    runner = sj.prepare_program(program, {"x": x, "y": y}, session=session)

    assert_raises(
        ValueError,
        lambda: sj.prepare_program(program, {"x": x, "y": y}, output="pyarrow"),
        contains="output='numpy'",
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": x.astype(np.int64), "y": y}),
        contains="type mismatch",
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": np.array([1, 2], dtype=np.int32), "y": y[:2]}),
        contains="length mismatch",
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": x}),
        contains="missing input",
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": x, "y": y, "z": x}),
        contains="unexpected input",
    )
    masked = ir.BufferHandle(
        ty=ir.I32,
        buf=x,
        null=ir.NullEncoding(
            kind="mask_bool",
            buf=np.array([False, False, False], dtype=np.bool_),
            true_means_null=True,
        ),
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": masked, "y": y}),
        contains="null transport mismatch",
    )

    assert session.release(runner.identifier) is True
    assert_raises(RuntimeError, runner.run, contains="prepared program was invalidated")


def test_prepared_runner_rejects_sentinel_rebind_changes():
    x = np.array([1, -1, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    program = sj.query(result=sj.coalesce(sj.col("x", sj.I32), sj.col("y", sj.I32)))
    runner = sj.prepare_program(
        program,
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(kind="sentinel", sentinel=-1),
            ),
            "y": y,
        },
    )

    assert runner.run() is None
    assert_array_equal(runner.outputs.result, np.array([1, 20, 3], dtype=np.int32))
    assert_raises(
        ValueError,
        lambda: runner.run(
            {
                "x": ir.BufferHandle(
                    ty=ir.I32,
                    buf=x,
                    null=ir.NullEncoding(kind="sentinel", sentinel=-2),
                ),
                "y": y,
            }
        ),
        contains="null transport mismatch",
    )


def test_run_into_writes_explicit_numpy_and_scalar_outputs():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    out = np.empty(4, dtype=np.int32)
    total = np.empty(1, dtype=np.int32)
    program = sj.query(
        result=sj.col("x") + sj.col("y"), total=sj.col("x", sj.I32).sum()
    )

    result = sj.run_into(
        program,
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
        },
        {
            "result": ir.BufferHandle(ty=ir.I32, buf=out),
            "total": ir.BufferHandle(ty=ir.I32, buf=total, length=1),
        },
    )

    assert result.result is out
    assert result.total is total
    assert_array_equal(out, x + y)
    assert total[0] == np.sum(x)


def test_run_into_reuses_session_cache_and_validates_buffers():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    out = np.empty(3, dtype=np.int32)
    program = sj.query(result=sj.col("x") + sj.col("y"))
    inputs = {
        "x": ir.BufferHandle(ty=ir.I32, buf=x),
        "y": ir.BufferHandle(ty=ir.I32, buf=y),
    }
    outputs = {"result": ir.BufferHandle(ty=ir.I32, buf=out)}
    session = sj.Session()

    sj.run_into(program, inputs, outputs, session=session)
    first_stats = session.statistics()
    sj.run_into(program, inputs, outputs, session=session)
    second_stats = session.statistics()
    assert second_stats.function_count == 1
    assert second_stats.cache_hits > first_stats.cache_hits

    assert_raises(
        ValueError,
        lambda: sj.run_into(program, inputs, {}, session=session),
        contains="missing output",
    )
    assert_raises(
        ValueError,
        lambda: sj.run_into(
            program,
            inputs,
            {
                "result": outputs["result"],
                "extra": ir.BufferHandle(ty=ir.I32, buf=np.empty(3, dtype=np.int32)),
            },
            session=session,
        ),
        contains="unexpected output",
    )
    assert_raises(
        TypeError,
        lambda: sj.run_into(
            program, {"x": x, "y": inputs["y"]}, outputs, session=session
        ),
        contains="BufferHandle",
    )
    assert_raises(
        ValueError,
        lambda: sj.run_into(
            program,
            inputs,
            {"result": ir.BufferHandle(ty=ir.I64, buf=np.empty(3, dtype=np.int64))},
            session=session,
        ),
        contains="type conflict",
    )
    assert_raises(
        ValueError,
        lambda: sj.run_into(
            program,
            inputs,
            {"result": ir.BufferHandle(ty=ir.I32, buf=np.empty(2, dtype=np.int32))},
            session=session,
        ),
        contains="length",
    )
    assert_raises(
        ValueError,
        lambda: sj.run_into(
            program,
            {
                "x": inputs["x"],
                "y": ir.BufferHandle(ty=ir.I32, buf=np.array([1, 2], dtype=np.int32)),
            },
            outputs,
            session=session,
        ),
        contains="same length",
    )


def test_native_inspect_lowers_unresolved_graph_to_hir():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    program = sj.query(result=(sj.col("x") + sj.i32(1)) * sj.col("y"))
    native = simjit_ext.inspect_program(
        program.to_dsl(), {"x": x, "y": y}, "numpy", "best_effort", "native"
    )

    assert native["hir"]
    assert native["serialized"]
    assert native["mir"]


def test_program_run_uses_native_runtime_by_default(numpy_function):
    session = simjit_ext.Session()
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    result = (
        numpy_function(x=x, y=y)
        .output(
            "result",
            sj.col("x") + sj.col("y"),
            lambda inputs: inputs["x"] + inputs["y"],
        )
        .run(session=session)
    )

    assert_array_equal(result.result, x + y)
    identifiers = session.function_identifiers()
    assert len(identifiers) == 1
    assert identifiers[0].startswith("native:")


def test_program_run_accepts_raw_runtime_inputs_in_fresh_process():
    code = r"""
import numpy as np
import simjit as sj

program = sj.query(result=sj.col("x") + sj.col("y"))
result = sj.run_program(
    program,
    {
        "x": np.array([1, 2, 3], dtype=np.int32),
        "y": np.array([10, 20, 30], dtype=np.int32),
    },
)
assert result.result.tolist() == [11, 22, 33]

try:
    import pyarrow as pa
except ImportError:
    pa = None

if pa is not None:
    program = sj.query(result=sj.col("x", sj.I32) + sj.col("y", sj.I32))
    result = sj.run_program(
        program,
        {
            "x": pa.array([1, 2, None], type=pa.int32()),
            "y": pa.array([10, 20, 30], type=pa.int32()),
        },
        output="pyarrow",
    )
    assert result.result.to_pylist() == [11, 22, None]

print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path.cwd(),
        env=dict(os.environ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "ok" in proc.stdout


def test_program_run_default_native_runtime_supports_pyarrow_bool_transport(pa):
    program = sj.query(result=sj.col("x", sj.I1) & sj.col("y", sj.I1))

    result = sj.run_program(
        program,
        {
            "x": pa.array([True, False, True, True], type=pa.bool_()),
            "y": pa.array([True, True, False, True], type=pa.bool_()),
        },
        output="pyarrow",
    )

    assert isinstance(result.result, pa.Array)
    assert result.result.type == pa.bool_()
    assert result.result.to_pylist() == [True, False, False, True]


def test_program_run_uses_binding_for_aggregate():
    program = sj.query(
        total=sj.col("x", sj.I64).sum(), count_pos=(sj.col("x", sj.I64) > 0).count()
    )
    result = sj.run_program(program, {"x": np.array([1, -2, 3, 4], dtype=np.int64)})
    assert result["total"] == 6
    assert result["count_pos"] == 3


def test_program_run_accepts_memoryview_inputs():
    x_arr = np.array([1, 2, 3], dtype=np.int32)
    y_arr = np.array([10, 20, 30], dtype=np.int32)
    program = sj.query(sum_xy=sj.col("x") + sj.col("y"))
    result = sj.run_program(program, {"x": memoryview(x_arr), "y": memoryview(y_arr)})
    assert_array_equal(result.sum_xy, np.array([11, 22, 33], dtype=np.int32))


def test_program_run_accepts_buffer_handle_inputs():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = array.array("i", [10, 20, 30])
    program = sj.query(sum_xy=sj.col("x") + sj.col("y"))
    result = sj.run_program(
        program,
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
        },
    )
    assert_array_equal(result.sum_xy, np.array([11, 22, 33], dtype=np.int32))


def test_program_run_supports_generic_log2_function(numpy_function):
    (
        numpy_function(x=np.array([0, 1, 2, 3, 4, 8], dtype=np.int32))
        .output(
            "result",
            sj.log2(sj.col("x", sj.I32)),
            np.array([0, 0, 1, 1, 2, 3], dtype=np.int32),
        )
        .run()
    )


def test_program_run_supports_load_splat_with_one_element_input():
    program = sj.query(result=sj.col("x") + sj.load_splat("scalar", sj.I32))
    result = sj.run_program(
        program,
        {
            "x": np.array([1, 2, 3, 4], dtype=np.int32),
            "scalar": np.array([7], dtype=np.int32),
        },
    )
    assert_array_equal(result.result, np.array([8, 9, 10, 11], dtype=np.int32))


def test_program_run_supports_gather_with_table_length_different_from_index_length():
    program = sj.query(result=sj.gather(sj.col("idx", sj.I32), "src", sj.I32))
    result = sj.run_program(
        program,
        {
            "idx": np.array([4, 1, 3], dtype=np.int32),
            "src": np.array([10, 20, 30, 40, 50, 60], dtype=np.int32),
        },
    )
    assert_array_equal(result.result, np.array([50, 20, 40], dtype=np.int32))


def test_program_run_accepts_unsigned_numpy_inputs():
    program = sj.query(
        sum_xy=sj.col("x") + sj.col("y"),
        min_xy=sj.col("x").min(where=(sj.col("x") > sj.u32(0))),
    )
    result = sj.run_program(
        program,
        {
            "x": np.array([1, 4, 7], dtype=np.uint32),
            "y": np.array([10, 20, 30], dtype=np.uint32),
        },
    )
    assert_array_equal(result.sum_xy, np.array([11, 24, 37], dtype=np.uint32))
    assert result.min_xy == np.uint32(1).item()


def test_program_run_accepts_unsigned_comparisons():
    program = sj.query(
        gt=sj.cmp_ugt(sj.col("x"), sj.col("y")),
        div=sj.udiv(sj.col("x"), sj.col("y")),
    )
    result = sj.run_program(
        program,
        {
            "x": np.array([1, 10, 15], dtype=np.uint32),
            "y": np.array([2, 5, 4], dtype=np.uint32),
        },
    )
    assert_array_equal(result.gt, np.array([False, True, True], dtype=np.bool_))
    assert_array_equal(result.div, np.array([0, 2, 3], dtype=np.uint32))


def test_program_run_rejects_mixed_sign_arithmetic():
    program = sj.query(result=sj.col("x") + sj.col("y"))

    exc = assert_raises(
        ValueError,
        lambda: sj.run_program(
            program,
            {
                "x": np.array([1, 2, 3], dtype=np.uint32),
                "y": np.array([4, 5, 6], dtype=np.int32),
            },
        ),
    )
    assert "mixed signedness" in str(exc) or "incompatible numeric types" in str(exc)


def test_program_run_supports_same_sign_mixed_width_promotion(numpy_function):
    x = np.array([1, 2, 3], dtype=np.int16)
    y = np.array([10, 20, 30], dtype=np.int32)
    pred = np.array([True, False, True], dtype=np.bool_)
    result = (
        numpy_function(x=x, y=y, pred=pred)
        .output(
            "total",
            sj.col("x", sj.I16) + sj.col("y", sj.I32),
            lambda inputs: inputs["x"] + inputs["y"],
        )
        .output(
            "picked",
            sj.col("pred", sj.I1).ifelse(sj.col("x", sj.I16), sj.col("y", sj.I32)),
            np.array([1, 20, 3], dtype=np.int32),
        )
        .output(
            "filled",
            sj.coalesce(sj.col("x", sj.I16), sj.col("y", sj.I32)),
            x.astype(np.int32),
        )
        .run()
    )

    assert result.total.dtype == np.int32
    assert result.picked.dtype == np.int32
    assert result.filled.dtype == np.int32


def test_program_run_allows_explicit_sign_casts():
    program = sj.query(
        signed_sum=sj.signed_cast(sj.I32, sj.col("x")) + sj.col("y"),
        unsigned_sum=sj.col("x") + sj.unsigned_cast(sj.U32, sj.col("y")),
    )
    result = sj.run_program(
        program,
        {
            "x": np.array([1, 2, 3], dtype=np.uint32),
            "y": np.array([4, 5, 6], dtype=np.int32),
        },
    )
    assert_array_equal(result.signed_sum, np.array([5, 7, 9], dtype=np.int32))
    assert_array_equal(result.unsigned_sum, np.array([5, 7, 9], dtype=np.uint32))


def test_program_run_float_cast_from_unsigned_inputs(numpy_function):
    (
        numpy_function(x=np.array([1, 2, 3], dtype=np.uint32))
        .output(
            "casted",
            sj.float_cast(sj.F32, sj.col("x")),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            rtol=1e-6,
        )
        .run()
    )


def test_program_run_accepts_pyarrow_primitive_inputs(pa):
    program = sj.query(sum_xy=sj.col("x") + sj.col("y"))
    result = sj.run_program(
        program,
        {
            "x": pa.array([1, 2, 3], type=pa.int32()),
            "y": pa.array([10, 20, 30], type=pa.int32()),
        },
    )
    assert_array_equal(result.sum_xy, np.array([11, 22, 33], dtype=np.int32))


def test_program_run_supports_nullable_pyarrow_projection_and_functions(pa):
    x = pa.array([1, None, 3], type=pa.int32())
    y = pa.array([10, 20, None], type=pa.int32())
    program = sj.query(
        sum_xy=sj.col("x", sj.I32) + sj.col("y", sj.I32),
        filled=sj.coalesce(sj.col("x", sj.I32), sj.col("y", sj.I32), sj.i32(0)),
        missing=sj.is_null(sj.col("x", sj.I32)),
        present=sj.is_not_null(sj.col("y", sj.I32)),
    )
    result = sj.run_program(program, {"x": x, "y": y}, output="pyarrow")

    assert result.sum_xy.to_pylist() == [11, None, None]
    assert result.filled.to_pylist() == [1, 20, 3]
    assert result.missing.to_pylist() == [False, True, False]
    assert result.present.to_pylist() == [True, True, False]


def test_program_run_supports_nullable_pyarrow_aggregates(pa):
    x = pa.array([1, None, 3, None], type=pa.int32())
    program = sj.query(
        total=sj.col("x", sj.I32).sum(),
        count=sj.count_if(sj.col("x", sj.I32) > sj.i32(1)),
    )
    result = sj.run_program(program, {"x": x}, output="pyarrow")

    assert result.total.as_py() == 4
    assert result.count.as_py() == 1


def test_program_run_accepts_nullable_pyarrow_booleans(pa):
    pred = pa.array([True, None, False, True, None], type=pa.bool_())
    other = pa.array([True, True, True, False, False], type=pa.bool_())
    program = sj.query(
        pred_out=sj.col("pred", sj.I1),
        both=sj.col("pred", sj.I1) & sj.col("other", sj.I1),
        count=sj.col("pred", sj.I1).count(),
    )
    result = sj.run_program(program, {"pred": pred, "other": other}, output="pyarrow")

    assert result.pred_out.to_pylist() == [True, None, False, True, None]
    assert result.both.to_pylist() == [True, None, False, False, None]
    assert result.count.as_py() == 2


def test_dsl_raw_nullable_mask_bool_buffers():
    x = np.array([1, 0, 3], dtype=np.int32)
    x_null = np.array([False, True, False], dtype=np.bool_)
    out = np.zeros(3, dtype=np.int32)
    out_null = np.zeros(3, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=x_null,
                    true_means_null=True,
                ),
            ),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=out,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=out_null,
                    true_means_null=True,
                ),
            ),
        },
        sj.query(out=sj.coalesce(sj.load("x", sj.I32), sj.i32(7))),
        len(x),
    )

    assert_array_equal(out, np.array([1, 7, 3], dtype=np.int32))
    assert_array_equal(out_null, np.array([False, False, False], dtype=np.bool_))


def test_dsl_raw_mixed_nullable_and_plain_operations():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    x_null = np.array([False, True, False, False], dtype=np.bool_)
    y = np.array([10, 30, 40, 20], dtype=np.int32)
    sum_out = np.zeros(4, dtype=np.int32)
    sum_null = np.zeros(4, dtype=np.bool_)
    cmp_out = np.zeros(4, dtype=np.bool_)
    cmp_null = np.zeros(4, dtype=np.bool_)
    select_out = np.zeros(4, dtype=np.int32)
    select_null = np.zeros(4, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=x_null,
                    true_means_null=True,
                ),
            ),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "sum_out": ir.BufferHandle(
                ty=ir.I32,
                buf=sum_out,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=sum_null,
                    true_means_null=True,
                ),
            ),
            "cmp_out": ir.BufferHandle(
                ty=ir.I1,
                buf=cmp_out,
                bitpacked=False,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=cmp_null,
                    true_means_null=True,
                ),
            ),
            "select_out": ir.BufferHandle(
                ty=ir.I32,
                buf=select_out,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=select_null,
                    true_means_null=True,
                ),
            ),
        },
        sj.query(
            sum_out=sj.load("x", sj.I32) + sj.load("y", sj.I32),
            cmp_out=sj.load("x", sj.I32) < sj.load("y", sj.I32),
            select_out=sj.select(
                sj.load("y", sj.I32) > sj.i32(25),
                sj.load("x", sj.I32),
                sj.i32(0),
            ),
        ),
        len(x),
    )

    assert_array_equal(sum_out, x + y)
    assert_array_equal(sum_null, x_null)
    assert_array_equal(cmp_out, x < y)
    assert_array_equal(cmp_null, x_null)
    assert_array_equal(select_out, np.array([0, 2, 3, 0], dtype=np.int32))
    assert_array_equal(select_null, np.array([False, True, False, False]))


def test_dsl_raw_nullable_function_lifts_are_explicit():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    x_null = np.array([False, True, False, False], dtype=np.bool_)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    coalesced = np.zeros(4, dtype=np.int32)
    coalesced_null = np.ones(4, dtype=np.bool_)
    nullif_out = np.zeros(4, dtype=np.int32)
    nullif_null = np.zeros(4, dtype=np.bool_)
    is_null_out = np.zeros(4, dtype=np.bool_)
    is_not_null_out = np.zeros(4, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=x_null,
                    true_means_null=True,
                ),
            ),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "coalesced": ir.BufferHandle(
                ty=ir.I32,
                buf=coalesced,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=coalesced_null,
                    true_means_null=True,
                ),
            ),
            "nullif_out": ir.BufferHandle(
                ty=ir.I32,
                buf=nullif_out,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=nullif_null,
                    true_means_null=True,
                ),
            ),
            "is_null_out": ir.BufferHandle(
                ty=ir.I1,
                buf=is_null_out,
                bitpacked=False,
            ),
            "is_not_null_out": ir.BufferHandle(
                ty=ir.I1,
                buf=is_not_null_out,
                bitpacked=False,
            ),
        },
        sj.query(
            coalesced=sj.coalesce(sj.load("y", sj.I32), sj.load("x", sj.I32)),
            nullif_out=sj.nullif(sj.load("y", sj.I32), sj.i32(20)),
            is_null_out=sj.is_null(sj.load("y", sj.I32)),
            is_not_null_out=sj.is_not_null(sj.load("y", sj.I32)),
        ),
        len(x),
    )

    assert_array_equal(coalesced, y)
    assert_array_equal(coalesced_null, np.array([False, False, False, False]))
    assert_array_equal(nullif_out, y)
    assert_array_equal(nullif_null, np.array([False, True, False, False]))
    assert_array_equal(is_null_out, np.array([False, False, False, False]))
    assert_array_equal(is_not_null_out, np.array([True, True, True, True]))


def test_dsl_raw_plain_value_stored_to_nullable_output_writes_all_not_null():
    x = np.array([1, 2, 3], dtype=np.int32)
    out = np.zeros(3, dtype=np.int32)
    out_null = np.ones(3, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(ty=ir.I32, buf=x),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=out,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=out_null,
                    true_means_null=True,
                ),
            ),
        },
        sj.query(out=sj.load("x", sj.I32) + sj.i32(5)),
        len(x),
    )

    assert_array_equal(out, np.array([6, 7, 8], dtype=np.int32))
    assert_array_equal(out_null, np.array([False, False, False]))


def test_dsl_raw_nullable_value_unwraps_are_rejected():
    idx = np.array([0, 1, 2], dtype=np.int32)
    idx_null = np.array([False, True, False], dtype=np.bool_)
    src = np.array([10, 20, 30], dtype=np.int32)
    out = np.zeros(3, dtype=np.int32)

    def run_nullable_index():
        run_program(
            {
                "idx": ir.BufferHandle(
                    ty=ir.I32,
                    buf=idx,
                    null=ir.NullEncoding(
                        kind="mask_bool",
                        buf=idx_null,
                        true_means_null=True,
                    ),
                ),
                "src": ir.BufferHandle(ty=ir.I32, buf=src),
                "out": ir.BufferHandle(ty=ir.I32, buf=out),
            },
            sj.query(out=sj.gather(sj.load("idx", sj.I32), "src", sj.I32)),
            len(idx),
        )

    assert_raises(ValueError, run_nullable_index, "nullable value is not allowed")

    cond = np.array([True, False, True], dtype=np.bool_)
    cond_null = np.array([False, True, False], dtype=np.bool_)
    stored = np.zeros(3, dtype=np.int32)

    def run_nullable_condition():
        run_ir(
            {
                "cond": ir.BufferHandle(
                    ty=ir.I1,
                    buf=cond,
                    bitpacked=False,
                    null=ir.NullEncoding(
                        kind="mask_bool",
                        buf=cond_null,
                        true_means_null=True,
                    ),
                ),
                "stored": ir.BufferHandle(ty=ir.I32, buf=stored),
            },
            {
                "stored": ir.StoreExpr(
                    dt=ir.I32,
                    value=ir.ConstExpr(dt=ir.I32, value=7),
                    cond=ir.LoadExpr(dt=ir.I1, name="cond"),
                )
            },
            len(cond),
        )

    assert_raises(
        ValueError, run_nullable_condition, "nullable predicate is not allowed"
    )


def test_dsl_raw_native_builder_finalizers():
    value_i32 = np.array([10, 20, 30, 40], dtype=np.int32)
    idx = np.array([3, 1, 0, 2], dtype=np.int32)
    cond = np.array([True, False, True, False], dtype=np.bool_)

    dst = np.zeros(4, dtype=np.int32)
    dst_null = np.ones(4, dtype=np.bool_)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value_i32),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "dst": ir.BufferHandle(
                ty=ir.I32,
                buf=dst,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=dst_null,
                    true_means_null=True,
                ),
            ),
        },
        {
            "dst": ir.ScatterExpr(
                dt=ir.I32,
                value=ir.LoadExpr(dt=ir.I32, name="value"),
                idx=ir.LoadExpr(dt=ir.I32, name="idx"),
            )
        },
        len(value_i32),
    )
    assert_array_equal(dst, np.array([30, 20, 40, 10], dtype=np.int32))
    assert_array_equal(dst_null, np.zeros(4, dtype=np.bool_))

    dst = np.zeros(4, dtype=np.int32)
    dst_null = np.ones(4, dtype=np.bool_)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I32, buf=value_i32),
            "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
            "dst": ir.BufferHandle(
                ty=ir.I32,
                buf=dst,
                null=ir.NullEncoding(
                    kind="mask_bool",
                    buf=dst_null,
                    true_means_null=True,
                ),
            ),
        },
        {
            "dst": ir.StoreExpr(
                dt=ir.I32,
                value=ir.LoadExpr(dt=ir.I32, name="value"),
                cond=ir.LoadExpr(dt=ir.I1, name="cond"),
            )
        },
        len(value_i32),
    )
    assert_array_equal(dst, np.array([10, 0, 30, 0], dtype=np.int32))
    assert_array_equal(dst_null, np.array([False, True, False, True]))

    value_i1 = np.array([True, True, False, False], dtype=np.bool_)

    dst = np.zeros(4, dtype=np.bool_)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I1, buf=value_i1, bitpacked=False),
            "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
            "dst": ir.BufferHandle(ty=ir.I1, buf=dst, bitpacked=False),
        },
        {
            "dst": ir.StoreExpr(
                dt=ir.I1,
                value=ir.LoadExpr(dt=ir.I1, name="value"),
                cond=ir.LoadExpr(dt=ir.I1, name="cond"),
            )
        },
        len(value_i1),
    )
    assert_array_equal(dst, np.array([True, False, False, False]))

    dst = np.zeros(4, dtype=np.bool_)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I1, buf=value_i1, bitpacked=False),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "dst": ir.BufferHandle(ty=ir.I1, buf=dst, bitpacked=False),
        },
        {
            "dst": ir.ScatterExpr(
                dt=ir.I1,
                value=ir.LoadExpr(dt=ir.I1, name="value"),
                idx=ir.LoadExpr(dt=ir.I32, name="idx"),
            )
        },
        len(value_i1),
    )
    assert_array_equal(dst, np.array([False, True, False, True]))

    dst = np.zeros(4, dtype=np.bool_)
    dst_size = np.zeros(1, dtype=np.int64)
    run_ir(
        {
            "value": ir.BufferHandle(ty=ir.I1, buf=value_i1, bitpacked=False),
            "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
            "dst": ir.BufferHandle(ty=ir.I1, buf=dst, bitpacked=False),
            "dst_size": ir.BufferHandle(ty=ir.I64, buf=dst_size),
        },
        {
            "dst": ir.PackExpr(
                dt=ir.I1,
                value=ir.LoadExpr(dt=ir.I1, name="value"),
                cond=ir.LoadExpr(dt=ir.I1, name="cond"),
                dst_size="dst_size",
            )
        },
        len(value_i1),
    )
    assert dst_size[0] == 2
    assert_array_equal(dst[:2], np.array([True, False]))


def test_dsl_raw_nullable_pack_and_predicate_aggregate():
    value = np.array([10, 20, 30, 40], dtype=np.int32)
    value_null = np.array([False, True, False, True], dtype=np.bool_)
    cond = np.array([True, True, False, True], dtype=np.bool_)
    packed = np.zeros(4, dtype=np.int32)
    packed_null = np.zeros(4, dtype=np.bool_)
    packed_size = np.zeros(1, dtype=np.int64)
    run_ir(
        {
            "value": ir.BufferHandle(
                ty=ir.I32,
                buf=value,
                null=ir.NullEncoding(kind="mask_bool", buf=value_null),
            ),
            "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
            "packed": ir.BufferHandle(
                ty=ir.I32,
                buf=packed,
                null=ir.NullEncoding(kind="mask_bool", buf=packed_null),
            ),
            "packed_size": ir.BufferHandle(ty=ir.I64, buf=packed_size),
        },
        {
            "packed": ir.PackExpr(
                dt=ir.I32,
                value=ir.LoadExpr(dt=ir.I32, name="value"),
                cond=ir.LoadExpr(dt=ir.I1, name="cond"),
                dst_size="packed_size",
            )
        },
        len(value),
    )
    assert packed_size[0] == 3
    assert_array_equal(packed[:3], np.array([10, 20, 40], dtype=np.int32))
    assert_array_equal(packed_null[:3], np.array([False, True, True]))

    scattered = np.zeros(4, dtype=np.int32)
    scattered_null = np.zeros(4, dtype=np.bool_)
    idx = np.array([3, 1, 0, 2], dtype=np.int32)
    run_ir(
        {
            "value": ir.BufferHandle(
                ty=ir.I32,
                buf=value,
                null=ir.NullEncoding(kind="mask_bool", buf=value_null),
            ),
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
            "scattered": ir.BufferHandle(
                ty=ir.I32,
                buf=scattered,
                null=ir.NullEncoding(kind="mask_bool", buf=scattered_null),
            ),
        },
        {
            "scattered": ir.ScatterExpr(
                dt=ir.I32,
                value=ir.LoadExpr(dt=ir.I32, name="value"),
                idx=ir.LoadExpr(dt=ir.I32, name="idx"),
                cond=ir.LoadExpr(dt=ir.I1, name="cond"),
            )
        },
        len(value),
    )
    assert_array_equal(scattered, np.array([0, 20, 40, 10], dtype=np.int32))
    assert_array_equal(scattered_null, np.array([False, True, True, False]))

    predicate = np.array([False, True, True, False], dtype=np.bool_)
    predicate_null = np.array([True, False, False, True], dtype=np.bool_)
    packed_predicate = np.zeros(4, dtype=np.bool_)
    packed_predicate_null = np.zeros(4, dtype=np.bool_)
    packed_predicate_size = np.zeros(1, dtype=np.int64)
    run_ir(
        {
            "predicate": ir.BufferHandle(
                ty=ir.I1,
                buf=predicate,
                bitpacked=False,
                null=ir.NullEncoding(kind="mask_bool", buf=predicate_null),
            ),
            "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
            "packed": ir.BufferHandle(
                ty=ir.I1,
                buf=packed_predicate,
                bitpacked=False,
                null=ir.NullEncoding(kind="mask_bool", buf=packed_predicate_null),
            ),
            "packed_size": ir.BufferHandle(ty=ir.I64, buf=packed_predicate_size),
        },
        {
            "packed": ir.PackExpr(
                dt=ir.I1,
                value=ir.LoadExpr(dt=ir.I1, name="predicate"),
                cond=ir.LoadExpr(dt=ir.I1, name="cond"),
                dst_size="packed_size",
            )
        },
        len(predicate),
    )
    assert packed_predicate_size[0] == 3
    assert_array_equal(packed_predicate[:3], np.array([False, True, False]))
    assert_array_equal(packed_predicate_null[:3], np.array([True, False, True]))

    aggregate = np.zeros(1, dtype=np.bool_)
    run_ir(
        {
            "predicate": ir.BufferHandle(
                ty=ir.I1,
                buf=predicate,
                bitpacked=False,
                null=ir.NullEncoding(kind="mask_bool", buf=predicate_null),
            ),
            "aggregate": ir.BufferHandle(ty=ir.I1, buf=aggregate, bitpacked=False),
        },
        {
            "aggregate": ir.PredicateAggExpr(
                dt=ir.I1,
                op=ir.PredicateBinaryOp.Or,
                arg=ir.LoadExpr(dt=ir.I1, name="predicate"),
            )
        },
        len(predicate),
    )
    assert aggregate[0]


def test_dsl_raw_bitpacked_random_access_outputs_are_unsupported():
    value = np.array([True, False, True, False], dtype=np.bool_)
    idx = np.arange(4, dtype=np.int32)
    cond = np.ones(4, dtype=np.bool_)

    def run_scatter():
        run_ir(
            {
                "value": ir.BufferHandle(ty=ir.I1, buf=value, bitpacked=False),
                "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
                "dst": ir.BufferHandle(ty=ir.I1, buf=np.zeros(1, dtype=np.uint8), bitpacked=True),
            },
            {
                "dst": ir.ScatterExpr(
                    dt=ir.I1,
                    value=ir.LoadExpr(dt=ir.I1, name="value"),
                    idx=ir.LoadExpr(dt=ir.I32, name="idx"),
                )
            },
            len(value),
        )

    assert_raises(ValueError, run_scatter, "bitpacked scatter is unsupported")

    def run_pack():
        run_ir(
            {
                "value": ir.BufferHandle(ty=ir.I1, buf=value, bitpacked=False),
                "cond": ir.BufferHandle(ty=ir.I1, buf=cond, bitpacked=False),
                "dst": ir.BufferHandle(ty=ir.I1, buf=np.zeros(1, dtype=np.uint8), bitpacked=True),
                "dst_size": ir.BufferHandle(ty=ir.I64, buf=np.zeros(1, dtype=np.int64)),
            },
            {
                "dst": ir.PackExpr(
                    dt=ir.I1,
                    value=ir.LoadExpr(dt=ir.I1, name="value"),
                    cond=ir.LoadExpr(dt=ir.I1, name="cond"),
                    dst_size="dst_size",
                )
            },
            len(value),
        )

    assert_raises(ValueError, run_pack, "bitpacked pack is unsupported")


def test_dsl_raw_nullable_inverted_mask_bool_buffers():
    x = np.array([1, 2, 3, 4], dtype=np.int32)
    x_valid = np.array([True, False, True, False], dtype=np.bool_)
    out = np.zeros(4, dtype=np.int32)
    out_valid = np.zeros(4, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=x_valid, true_means_null=False
                ),
            ),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=out,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=out_valid, true_means_null=False
                ),
            ),
        },
        sj.query(out=sj.load("x", sj.I32)),
        len(x),
    )

    assert_array_equal(out, x)
    assert_array_equal(out_valid, x_valid)

    bit_valid = np.packbits(x_valid, bitorder="little")
    bit_out = np.zeros(4, dtype=np.int32)
    bit_out_valid = np.zeros(1, dtype=np.uint8)
    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bitpacked", buf=bit_valid, true_means_null=False
                ),
            ),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=bit_out,
                null=ir.NullEncoding(
                    kind="mask_bitpacked", buf=bit_out_valid, true_means_null=False
                ),
            ),
        },
        sj.query(out=sj.load("x", sj.I32)),
        len(x),
    )

    assert_array_equal(bit_out, x)
    assert_array_equal(
        np.unpackbits(bit_out_valid, bitorder="little")[: len(x)], x_valid
    )

    coalesced = np.zeros(4, dtype=np.int32)
    coalesced_valid = np.zeros(4, dtype=np.bool_)
    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=x_valid, true_means_null=False
                ),
            ),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=coalesced,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=coalesced_valid, true_means_null=False
                ),
            ),
        },
        sj.query(out=sj.coalesce(sj.load("x", sj.I32), sj.i32(7))),
        len(x),
    )

    assert_array_equal(coalesced, np.array([1, 7, 3, 7], dtype=np.int32))
    assert_array_equal(
        coalesced_valid, np.array([True, True, True, True], dtype=np.bool_)
    )

    scalar = np.array([11], dtype=np.int32)
    scalar_valid = np.array([False], dtype=np.bool_)
    splat = np.zeros(4, dtype=np.int32)
    splat_valid = np.zeros(4, dtype=np.bool_)
    run_program(
        {
            "scalar": ir.BufferHandle(
                ty=ir.I32,
                buf=scalar,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=scalar_valid, true_means_null=False
                ),
            ),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=splat,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=splat_valid, true_means_null=False
                ),
            ),
        },
        sj.query(out=sj.load_splat("scalar", sj.I32)),
        len(splat),
    )

    assert_array_equal(splat, np.full(4, 11, dtype=np.int32))
    assert_array_equal(
        splat_valid, np.array([False, False, False, False], dtype=np.bool_)
    )

    idx = np.array([3, 1, 0, 2], dtype=np.int32)
    src = np.array([10, 20, 30, 40], dtype=np.int32)
    src_valid = np.array([True, False, True, False], dtype=np.bool_)
    gathered = np.zeros(4, dtype=np.int32)
    gathered_valid = np.zeros(4, dtype=np.bool_)
    run_program(
        {
            "idx": ir.BufferHandle(ty=ir.I32, buf=idx),
            "src": ir.BufferHandle(
                ty=ir.I32,
                buf=src,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=src_valid, true_means_null=False
                ),
            ),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=gathered,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=gathered_valid, true_means_null=False
                ),
            ),
        },
        sj.query(out=sj.gather(sj.load("idx", sj.I32), "src", sj.I32)),
        len(idx),
    )

    assert_array_equal(gathered, np.array([40, 20, 10, 30], dtype=np.int32))
    assert_array_equal(
        gathered_valid, np.array([False, False, True, True], dtype=np.bool_)
    )

    value = np.array([10, 20, 30, 40], dtype=np.int32)
    value_valid = np.array([True, False, True, False], dtype=np.bool_)
    scatter_idx = np.array([3, 1, 0, 2], dtype=np.int32)
    dst = np.zeros(4, dtype=np.int32)
    dst_valid = np.zeros(4, dtype=np.bool_)
    run_program(
        {
            "value": ir.BufferHandle(
                ty=ir.I32,
                buf=value,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=value_valid, true_means_null=False
                ),
            ),
            "idx": ir.BufferHandle(ty=ir.I32, buf=scatter_idx),
            "dst": ir.BufferHandle(
                ty=ir.I32,
                buf=dst,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=dst_valid, true_means_null=False
                ),
            ),
        },
        sj.query(dst=sj.scatter(sj.load("value", sj.I32), sj.load("idx", sj.I32))),
        len(value),
    )

    assert_array_equal(dst, np.array([30, 20, 40, 10], dtype=np.int32))
    assert_array_equal(dst_valid, np.array([True, False, False, True], dtype=np.bool_))


def test_program_run_accepts_mixed_pyarrow_and_numpy_inputs(pa):
    program = sj.query(
        sum_xy=sj.col("x") + sj.col("y"),
        gt_20=(sj.col("x") + sj.col("y")) > 20,
    )
    result = sj.run_program(
        program,
        {
            "x": pa.array([1, 2, 3], type=pa.int32()),
            "y": np.array([10, 20, 30], dtype=np.int32),
        },
    )
    assert_array_equal(result.sum_xy, np.array([11, 22, 33], dtype=np.int32))
    assert_array_equal(result.gt_20, np.array([False, True, True], dtype=np.bool_))


def test_program_run_returns_pyarrow_outputs(pa):
    program = sj.query(
        sum_xy=sj.col("x") + sj.col("y"),
        gt_20=(sj.col("x") + sj.col("y")) > 20,
    )
    result = sj.run_program(
        program,
        {
            "x": pa.array([1, 2, 3], type=pa.int32()),
            "y": np.array([10, 20, 30], dtype=np.int32),
        },
        output="pyarrow",
    )
    assert isinstance(result.sum_xy, pa.Array)
    assert isinstance(result.gt_20, pa.Array)
    assert result.sum_xy.type == pa.int32()
    assert result.gt_20.type == pa.bool_()
    assert result.sum_xy.to_pylist() == [11, 22, 33]
    assert result.gt_20.to_pylist() == [False, True, True]


def test_program_run_accepts_pyarrow_boolean_input(pa):
    program = sj.query(
        flag=sj.col("pred", sj.I1).ifelse(sj.i32(1), sj.i32(0)),
        count=sj.col("pred", sj.I1).count(),
    )
    result = sj.run_program(
        program, {"pred": pa.array([True, False, True, True], type=pa.bool_())}
    )
    assert_array_equal(result.flag, np.array([1, 0, 1, 1], dtype=np.int32))
    assert result.count == 3


def test_program_run_accepts_bitpacked_pyarrow_boolean_in_comparison_flow(pa):
    pred = pa.array([True, False, True, True], type=pa.bool_())
    x = np.array([5, 10, 20, 30], dtype=np.int32)
    program = sj.query(
        selected=sj.col("pred", sj.I1).ifelse(sj.col("x", sj.I32), sj.i32(0)),
        gt_15=sj.col("pred", sj.I1).ifelse(sj.col("x", sj.I32), sj.i32(0)) > 15,
        gated=(sj.col("x", sj.I32) > 8) & sj.col("pred", sj.I1),
    )
    result = sj.run_program(program, {"pred": pred, "x": x})
    expected_selected = np.array([5, 0, 20, 30], dtype=np.int32)
    assert_array_equal(result.selected, expected_selected)
    assert_array_equal(result.gt_15, expected_selected > 15)
    assert_array_equal(
        result.gated, np.array([False, False, True, True], dtype=np.bool_)
    )


def test_program_run_accepts_numpy_timestamp_projection_and_extractors():
    ts = np.array(
        ["2024-01-01T00:00:00", "2024-01-02T03:04:05"],
        dtype="datetime64[s]",
    )
    program = sj.query(
        ts_out=sj.col("ts"),
        year=sj.year(sj.col("ts")),
        hour=sj.hour(sj.col("ts")),
        day_of_week=sj.day_of_week(sj.col("ts")),
    )
    result = sj.run_program(program, {"ts": ts})
    assert_array_equal(result.ts_out, ts)
    assert_array_equal(result.year, np.array([2024, 2024], dtype=np.int32))
    assert_array_equal(result.hour, np.array([0, 3], dtype=np.int32))
    assert_array_equal(result.day_of_week, np.array([0, 1], dtype=np.int32))


def test_program_run_accepts_mixed_numpy_and_pyarrow_timestamps(pa):
    lhs = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00"],
        dtype="datetime64[s]",
    )
    rhs = pa.array([1704067190, 1704067200], type=pa.timestamp("s"))
    program = sj.query(
        gt=sj.col("lhs") > sj.col("rhs"),
        diff=sj.col("lhs") - sj.col("rhs"),
        shifted=sj.col("lhs") + sj.i64(10),
    )
    result = sj.run_program(program, {"lhs": lhs, "rhs": rhs})
    assert_array_equal(result.gt, np.array([True, True], dtype=np.bool_))
    assert_array_equal(result.diff, np.array([10, 60], dtype=np.int64))
    assert_array_equal(
        result.shifted,
        np.array(["2024-01-01T00:00:10", "2024-01-01T00:01:10"], dtype="datetime64[s]"),
    )


def test_program_run_accepts_pyarrow_timezone_metadata_stably(pa):
    ts = pa.array([1704067200, 1704153600], type=pa.timestamp("s", tz="Europe/Moscow"))
    program = sj.query(ts_out=sj.col("ts"), min_ts=sj.col("ts").min())
    result = sj.run_program(program, {"ts": ts})
    assert_array_equal(
        result.ts_out,
        np.array(["2024-01-01T00:00:00", "2024-01-02T00:00:00"], dtype="datetime64[s]"),
    )
    assert result.min_ts == np.datetime64("2024-01-01T00:00:00", "s")


def test_program_run_returns_pyarrow_timestamp_outputs(pa):
    ts = pa.array([1704067200, 1704153600], type=pa.timestamp("s", tz="Europe/Moscow"))
    program = sj.query(ts_out=sj.col("ts"), min_ts=sj.col("ts").min())
    result = sj.run_program(program, {"ts": ts}, output="pyarrow")
    assert isinstance(result.ts_out, pa.Array)
    assert result.ts_out.type == pa.timestamp("s", tz="UTC")
    assert result.ts_out.to_pylist() == ts.cast(pa.timestamp("s", tz="UTC")).to_pylist()
    assert result.min_ts.type == pa.timestamp("s", tz="UTC")
    assert result.min_ts.as_py() == ts.cast(pa.timestamp("s", tz="UTC"))[0].as_py()


def test_add_no_overflow(numpy_function):
    function = numpy_function(
        x=np.array([1, 2, 3], dtype=np.int32),
        y=np.array([10, 20, 30], dtype=np.int32),
    )
    x = sj.col("x")
    y = sj.col("y")
    result = sdf.add_checked(x, y)

    function.output("sum", result, lambda inputs: inputs["x"] + inputs["y"])
    function.run()


def test_add_safety_check_failed(numpy_function):
    function = numpy_function(
        x=np.array([np.iinfo(np.int32).max], dtype=np.int32),
        y=np.array([1], dtype=np.int32),
    )
    x = sj.col("x")
    y = sj.col("y")
    result = sdf.add_checked(x, y)

    function.output("sum", result, np.array([np.iinfo(np.int32).min], dtype=np.int32))
    with pytest.raises(sj.SafetyCheckFailed):
        function.run()


def test_nullable_add_ignores_safety_failure_on_null_row():
    x = np.array([1, np.iinfo(np.int32).max, 10], dtype=np.int32)
    x_null = np.array([False, True, False], dtype=np.bool_)
    y = np.array([2, 1, -3], dtype=np.int32)
    out = np.empty(3, dtype=np.int32)
    out_null = np.empty(3, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=x_null, true_means_null=True
                ),
            ),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=out,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=out_null, true_means_null=True
                ),
            ),
        },
        sj.query(out=sdf.add_checked(sj.col("x"), sj.col("y"))),
        len(x),
    )

    assert_array_equal(out, np.array([3, np.iinfo(np.int32).min, 7], dtype=np.int32))
    assert_array_equal(out_null, x_null)


def test_nullable_sub_ignores_safety_failure_on_null_row():
    x = np.array([10, np.iinfo(np.int32).min, -5], dtype=np.int32)
    x_null = np.array([False, True, False], dtype=np.bool_)
    y = np.array([3, 1, 2], dtype=np.int32)
    out = np.empty(3, dtype=np.int32)
    out_null = np.empty(3, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=x_null, true_means_null=True
                ),
            ),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=out,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=out_null, true_means_null=True
                ),
            ),
        },
        sj.query(out=sdf.sub_checked(sj.col("x"), sj.col("y"))),
        len(x),
    )

    assert_array_equal(out, np.array([7, np.iinfo(np.int32).max, -7], dtype=np.int32))
    assert_array_equal(out_null, x_null)


def test_nullable_mul_ignores_safety_failure_on_null_row():
    x = np.array([3, np.iinfo(np.int32).max, -4], dtype=np.int32)
    x_null = np.array([False, True, False], dtype=np.bool_)
    y = np.array([7, 2, 5], dtype=np.int32)
    out = np.empty(3, dtype=np.int32)
    out_null = np.empty(3, dtype=np.bool_)

    run_program(
        {
            "x": ir.BufferHandle(
                ty=ir.I32,
                buf=x,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=x_null, true_means_null=True
                ),
            ),
            "y": ir.BufferHandle(ty=ir.I32, buf=y),
            "out": ir.BufferHandle(
                ty=ir.I32,
                buf=out,
                null=ir.NullEncoding(
                    kind="mask_bool", buf=out_null, true_means_null=True
                ),
            ),
        },
        sj.query(out=sdf.mul_checked(sj.col("x"), sj.col("y"))),
        len(x),
    )

    assert_array_equal(out, np.array([21, -2, -20], dtype=np.int32))
    assert_array_equal(out_null, x_null)


def test_nullable_checked_unary_ignores_safety_failure_on_null_row():
    value = np.array([-7, np.iinfo(np.int32).min, 9], dtype=np.int32)
    value_null = np.array([False, True, False], dtype=np.bool_)
    negated = np.empty(3, dtype=np.int32)
    negated_null = np.empty(3, dtype=np.bool_)
    absolute = np.empty(3, dtype=np.int32)
    absolute_null = np.empty(3, dtype=np.bool_)
    run_program(
        {
            "value": ir.BufferHandle(
                ty=ir.I32,
                buf=value,
                null=ir.NullEncoding(kind="mask_bool", buf=value_null),
            ),
            "negated": ir.BufferHandle(
                ty=ir.I32,
                buf=negated,
                null=ir.NullEncoding(kind="mask_bool", buf=negated_null),
            ),
            "absolute": ir.BufferHandle(
                ty=ir.I32,
                buf=absolute,
                null=ir.NullEncoding(kind="mask_bool", buf=absolute_null),
            ),
        },
        sj.query(
            negated=sdf.negate_checked(sj.col("value")),
            absolute=sdf.abs_checked(sj.col("value")),
        ),
        len(value),
    )
    assert_array_equal(negated, np.array([7, np.iinfo(np.int32).min, -9]))
    assert_array_equal(absolute, np.array([7, np.iinfo(np.int32).min, 9]))
    assert_array_equal(negated_null, value_null)
    assert_array_equal(absolute_null, value_null)


def test_sub_no_overflow(numpy_function):
    function = numpy_function(
        x=np.array([1, 2, 3], dtype=np.int32),
        y=np.array([10, 20, 30], dtype=np.int32),
    )
    x = sj.col("x")
    y = sj.col("y")
    result = sdf.sub_checked(x, y)

    function.output("sum", result, lambda inputs: inputs["x"] - inputs["y"])
    function.run()


def test_sub_safety_check_failed(numpy_function):
    function = numpy_function(
        x=np.array([np.iinfo(np.int32).max], dtype=np.int32),
        y=np.array([-1], dtype=np.int32),
    )
    x = sj.col("x")
    y = sj.col("y")
    result = sdf.sub_checked(x, y)

    function.output("sum", result, np.array([np.iinfo(np.int32).min], dtype=np.int32))
    with pytest.raises(sj.SafetyCheckFailed):
        function.run()


def test_mul_no_overflow(numpy_function):
    function = numpy_function(
        x=np.array([1, 2, 3], dtype=np.int32),
        y=np.array([10, 20, 30], dtype=np.int32),
    )
    x = sj.col("x")
    y = sj.col("y")
    result = sdf.mul_checked(x, y)

    function.output("sum", result, lambda inputs: inputs["x"] * inputs["y"])
    function.run()


def test_mul_safety_check_failed(numpy_function):
    function = numpy_function(
        x=np.array([np.iinfo(np.int32).max], dtype=np.int32),
        y=np.array([np.iinfo(np.int32).max], dtype=np.int32),
    )
    x = sj.col("x")
    y = sj.col("y")
    result = sdf.mul_checked(x, y)

    function.output("sum", result, np.array([np.iinfo(np.int32).min], dtype=np.int32))
    with pytest.raises(sj.SafetyCheckFailed):
        function.run()


def test_dsl_function_expr_timestamp_lowering():
    ts = np.array(["2024-01-02T03:04:05"], dtype="datetime64[s]")
    out_year = np.zeros(1, dtype=np.int32)
    out_month = np.zeros(1, dtype=np.int32)
    out_second = np.zeros(1, dtype=np.int32)
    run_ir(
        {
            "ts": ir.BufferHandle(ty=ir.timestamp64("s"), buf=ts),
            "out_year": ir.BufferHandle(ty=ir.I32, buf=out_year),
            "out_month": ir.BufferHandle(ty=ir.I32, buf=out_month),
            "out_second": ir.BufferHandle(ty=ir.I32, buf=out_second),
        },
        {
            "out_year": ir.FunctionExpr(
                dt=ir.I32,
                name=ir.FunctionName.Year,
                args=(ir.LoadExpr(dt=ir.timestamp64("s"), name="ts"),),
            ),
            "out_month": ir.FunctionExpr(
                dt=ir.I32,
                name=ir.FunctionName.Month,
                args=(ir.LoadExpr(dt=ir.timestamp64("s"), name="ts"),),
            ),
            "out_second": ir.FunctionExpr(
                dt=ir.I32,
                name=ir.FunctionName.Second,
                args=(ir.LoadExpr(dt=ir.timestamp64("s"), name="ts"),),
            ),
        },
        len(ts),
    )
    assert_array_equal(out_year, np.array([2024], dtype=np.int32))
    assert_array_equal(out_month, np.array([1], dtype=np.int32))
    assert_array_equal(out_second, np.array([5], dtype=np.int32))
