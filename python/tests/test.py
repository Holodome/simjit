# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

import datetime as _datetime
import platform
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import simjit as sj
import simjit.dataframe as sdf
import simjit.ir as ir
from simjit import _simjit as simjit_ext

try:
    import pyarrow as pa
except ImportError:
    pa = None


def make_pa_array(values, ty):
    assert pa is not None
    valid = [value is not None for value in values]
    validity = None if all(valid) else pa.py_buffer(pack_bits(valid))
    clean = [0 if value is None else value for value in values]
    if pa.types.is_boolean(ty):
        data = pa.py_buffer(pack_bits([bool(value) for value in clean]))
    elif pa.types.is_int32(ty):
        data = pa.py_buffer(np.asarray(clean, dtype=np.int32))
    elif pa.types.is_uint32(ty):
        data = pa.py_buffer(np.asarray(clean, dtype=np.uint32))
    elif pa.types.is_float64(ty):
        data = pa.py_buffer(np.asarray(clean, dtype=np.float64))
    elif pa.types.is_timestamp(ty):
        data = pa.py_buffer(np.asarray(clean, dtype=np.int64))
    else:
        raise TypeError(f"unsupported test Arrow type {ty}")
    return pa.Array.from_buffers(ty, len(values), [validity, data])


def pack_bits(values):
    data = bytearray((len(values) + 7) // 8)
    for idx, value in enumerate(values):
        if value:
            data[idx // 8] |= 1 << (idx % 8)
    return bytes(data)


def assert_is(expr, cls, dt):
    assert isinstance(expr, cls), f"expected {cls.__name__}, got {type(expr).__name__}"
    assert expr.dt == dt, f"expected dt={dt}, got {expr.dt}"


def assert_raises(exc_type, fn, contains=None):
    with pytest.raises(exc_type) as exc_info:
        fn()
    exc = exc_info.value
    if contains is not None:
        assert contains in str(exc), f"expected {contains!r} in {exc!r}"
    return exc


def test_explicitly_typed_arithmetic_chain():
    a = sdf.load("a", ir.I32)
    b = sdf.load("b", ir.I32)
    c = sdf.load("c", ir.I32)
    d = sdf.load("d", ir.I32)

    expr = sdf.mul(sdf.add(a, b), sdf.sub(c, d)).to_dsl()

    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.op is ir.ArithBinaryOp.Mul
    assert expr.lhs.op is ir.ArithBinaryOp.Add
    assert expr.rhs.op is ir.ArithBinaryOp.Sub
    assert expr.lhs.lhs.name == "a"
    assert expr.lhs.rhs.name == "b"
    assert expr.rhs.lhs.name == "c"
    assert expr.rhs.rhs.name == "d"


def test_typed_bit_manipulation_chain():
    a = sdf.load("a", ir.I16)
    b = sdf.load("b", ir.I16)
    c = sdf.load("c", ir.I16)
    mask = sdf.const(0xFF, ir.I16)

    expr = sdf.xor(
        sdf.or_(sdf.and_(a, mask), sdf.sll(b, sdf.const(2, ir.I16))), c
    ).to_dsl()

    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.op is ir.ArithBinaryOp.Xor
    assert expr.lhs.op is ir.ArithBinaryOp.Or
    assert expr.lhs.lhs.op is ir.ArithBinaryOp.And
    assert expr.lhs.rhs.op is ir.ArithBinaryOp.ShiftLeftLogical
    assert expr.rhs.name == "c"


def test_typed_unary_mix_from_general_examples():
    x = sdf.load("x", ir.I32)
    y = sdf.load("y", ir.I32)
    z = sdf.load("z", ir.I32)

    expr = sdf.add(
        sdf.add(sdf.abs(sdf.negate(x)), sdf.popcnt(y)), sdf.lzcnt(z)
    ).to_dsl()

    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.op is ir.ArithBinaryOp.Add
    assert expr.lhs.op is ir.ArithBinaryOp.Add
    assert expr.lhs.lhs.op is ir.ArithUnaryOp.Abs
    assert expr.lhs.lhs.dt == ir.I32
    assert expr.lhs.lhs.arg.op is ir.ArithUnaryOp.Negate
    assert expr.lhs.lhs.arg.dt == ir.I32
    assert expr.lhs.rhs.op is ir.ArithUnaryOp.Popcount
    assert expr.lhs.rhs.dt == ir.I32
    assert expr.rhs.op is ir.ArithUnaryOp.Lzcnt
    assert expr.rhs.dt == ir.I32


def test_untyped_expression_infers_none():
    x = sdf.load("x")
    y = sdf.load("y")

    expr = sdf.add(sdf.mul(sdf.const(5), x), sdf.sub(y, sdf.const(3))).to_dsl()

    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.lhs.dt is None
    assert expr.rhs.dt is None
    assert expr.lhs.lhs.value == 5
    assert expr.rhs.rhs.value == 3


def test_untyped_expression_with_explicit_casts():
    x = sdf.load("x")
    y = sdf.load("y")

    x_i32 = sdf.sext(ir.I32, x)
    y_i32 = sdf.zext(ir.I32, y)
    expr = sdf.add(x_i32, sdf.mul(y_i32, sdf.i32(7))).to_dsl()

    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.lhs.kind is ir.IntCastKind.Sext
    assert expr.lhs.dt == ir.I32
    assert expr.rhs.op is ir.ArithBinaryOp.Mul
    assert expr.rhs.lhs.kind is ir.IntCastKind.Zext
    assert expr.rhs.lhs.dt == ir.I32


def test_unsigned_types_and_helpers_are_public():
    expr = sdf.add(sdf.u32(7), sdf.load("x", ir.U32)).to_dsl()
    signed = sdf.signed_cast(ir.I32, sdf.load("u", ir.U32)).to_dsl()
    unsigned = sdf.unsigned_cast(ir.U32, sdf.load("i", ir.I32)).to_dsl()

    assert ir.U8.name == "u8"
    assert ir.U16.name == "u16"
    assert ir.U32.name == "u32"
    assert ir.U64.name == "u64"
    assert_is(expr, ir.ArithBinaryExpr, None)
    assert_is(signed, ir.IntCastExpr, ir.I32)
    assert signed.kind is ir.IntCastKind.Signed
    assert_is(unsigned, ir.IntCastExpr, ir.U32)
    assert unsigned.kind is ir.IntCastKind.Unsigned


def test_native_enums_are_imported_through_ir():
    enum_members = {
        "LoadStoreKind": ("Aligned", "Unaligned"),
        "ArithBinaryOp": (
            "Add",
            "Sub",
            "Mul",
            "Mul64SE",
            "Mul64ZE",
            "Div",
            "UDiv",
            "Mod",
            "UMod",
            "Min",
            "Max",
            "UMin",
            "UMax",
            "And",
            "Or",
            "Xor",
            "AndNot",
            "ShiftLeftLogical",
            "ShiftRightLogical",
            "ShiftRightArith",
            "RotateLeft",
            "RotateRight",
        ),
        "PredicateBinaryOp": ("And", "Or", "Xor", "AndNot", "XNor"),
        "ArithUnaryOp": (
            "Not",
            "Negate",
            "Abs",
            "Lzcnt",
            "Tzcnt",
            "Popcount",
            "RoundNearest",
            "RoundDown",
            "RoundUp",
            "RoundTruncate",
            "Rcp",
            "Sqrt",
            "Rsqrt",
        ),
        "CompareOp": (
            "Less",
            "Greater",
            "LessEqual",
            "GreaterEqual",
            "Equal",
            "NotEqual",
        ),
        "IntCastKind": ("Cast", "Signed", "Unsigned", "Trunc", "Sext", "Zext"),
        "FpClassFlags": ("Infinite", "Nan", "Subnormal", "Zero"),
        "FunctionName": (
            "Year",
            "Month",
            "Day",
            "Hour",
            "Minute",
            "Second",
            "DayOfWeek",
            "Log2",
            "Log2NoZero",
            "Byteswap",
            "BitFloor",
            "BitCeil",
            "Coalesce",
            "NullIf",
            "IsNull",
            "IsNotNull",
        ),
    }

    for enum_name, member_names in enum_members.items():
        enum_cls = getattr(ir, enum_name)
        assert enum_cls is getattr(simjit_ext, enum_name)
        for member_name in member_names:
            assert getattr(enum_cls, member_name) is getattr(
                getattr(simjit_ext, enum_name), member_name
            )

    assert ir.LoadExpr(dt=ir.I32, name="x").kind is ir.LoadStoreKind.Unaligned
    assert (
        ir.StoreExpr(dt=ir.I32, value=ir.LoadExpr(dt=ir.I32, name="x")).kind
        is ir.LoadStoreKind.Unaligned
    )
    flags = ir.FpClassFlags.Infinite | ir.FpClassFlags.Nan | ir.FpClassFlags.Zero
    assert isinstance(flags, ir.FpClassFlags)


def test_timestamp64_type_and_literal_helpers():
    ts_ty = ir.timestamp64("ms")
    aware_ty = ir.timestamp64("ns", "UTC")
    expr = sdf.timestamp("2024-01-02T03:04:05", unit="s").to_dsl()
    np_expr = sdf.timestamp(np.datetime64("2024-01-02T03:04:05"), unit="s").to_dsl()
    aware = sdf.timestamp(
        _datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=_datetime.timezone.utc),
        unit="s",
    ).to_dsl()
    extracted = sdf.year(sdf.load("ts", ts_ty))

    assert str(ts_ty) == "timestamp64[ms]"
    assert str(aware_ty) == "timestamp64[ns, UTC]"
    assert_is(expr, ir.ConstExpr, ir.timestamp64("s"))
    assert expr.value == int(np.datetime64("2024-01-02T03:04:05", "s").astype(np.int64))
    assert_is(np_expr, ir.ConstExpr, ir.timestamp64("s"))
    assert np_expr.value == expr.value
    assert_is(aware, ir.ConstExpr, ir.timestamp64("s", "UTC"))
    assert_is(
        extracted.to_dsl() if hasattr(extracted, "to_dsl") else extracted,
        ir.FunctionExpr,
        ir.I32,
    )


def test_base_import_and_explicit_buffers_without_numpy_or_pyarrow():
    code = r"""
import array
import datetime
import importlib.abc
import sys


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ImportError("blocked numpy")
        if fullname == "pyarrow" or fullname.startswith("pyarrow."):
            raise ImportError("blocked pyarrow")
        return None


sys.meta_path.insert(0, Blocker())
for name in list(sys.modules):
    if name == "numpy" or name.startswith("numpy."):
        sys.modules.pop(name, None)
    if name == "pyarrow" or name.startswith("pyarrow."):
        sys.modules.pop(name, None)

import simjit as sj
import simjit.dataframe as sdf
import simjit.ir as ir
from simjit import _simjit

expr = sdf.add(sdf.load("x", ir.I32), sdf.const(1, ir.I32)).to_dsl()
out = array.array("i", [0, 0, 0])
_simjit.run_native(
    {
        "x": ir.BufferHandle(ty=ir.I32, buf=array.array("i", [1, 2, 3])),
        "out": ir.BufferHandle(ty=ir.I32, buf=out),
    },
    [("out", ir.StoreExpr(dt=ir.I32, value=expr))],
    3,
)
assert list(out) == [2, 3, 4]

run_into_out = array.array("i", [0, 0, 0])
run_into_result = sj.run_into(
    sdf.query({"result": sdf.add(sdf.load("x", ir.I32), sdf.const(1, ir.I32))}),
    {"x": ir.BufferHandle(ty=ir.I32, buf=array.array("i", [1, 2, 3]))},
    {"result": ir.BufferHandle(ty=ir.I32, buf=run_into_out)},
)
assert list(run_into_out) == [2, 3, 4]
assert run_into_result.result is run_into_out

assert sdf.timestamp(123, unit="s").to_dsl().value == 123
assert sdf.timestamp("2024-01-02T03:04:05", unit="s").to_dsl().value == 1704164645
assert (
    sdf.timestamp(datetime.datetime(2024, 1, 2, 3, 4, 5), unit="s").to_dsl().value
    == 1704164645
)
aware = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
assert sdf.timestamp(aware, unit="s").to_dsl().dt == ir.timestamp64("s", "UTC")

program = sdf.query({"result": sdf.add(sdf.load("x", ir.I32), sdf.const(1, ir.I32))})
try:
    sj.run_program(
        program,
        {"x": ir.BufferHandle(ty=ir.I32, buf=array.array("i", [1, 2, 3]))},
    )
except (ImportError, RuntimeError) as exc:
    assert "blocked" in str(exc) or "requires numpy" in str(exc)
else:
    raise AssertionError("expected numpy output dependency error")

try:
    sj.run_program(
        program,
        {"x": ir.BufferHandle(ty=ir.I32, buf=array.array("i", [1, 2, 3]))},
        output="pyarrow",
    )
except (ImportError, RuntimeError) as exc:
    assert "blocked" in str(exc) or "requires pyarrow" in str(exc)
else:
    raise AssertionError("expected pyarrow output dependency error")
"""
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_predicates_and_select():
    x = sdf.load("x", ir.I32)
    y = sdf.load("y", ir.I32)

    mask = sdf.and_(sdf.cmp_gt(x, 0), sdf.cmp_lt(y, 100))
    expr = sdf.select(mask, 1, 0).to_dsl()
    mask = mask.to_dsl()

    assert_is(mask, ir.PredicateBinaryExpr, ir.I1)
    assert mask.op is ir.PredicateBinaryOp.And
    assert mask.lhs.op is ir.CompareOp.Greater
    assert mask.rhs.op is ir.CompareOp.Less

    assert_is(expr, ir.SelectExpr, None)
    assert expr.cond is mask
    assert expr.truthy.value == 1
    assert expr.falsy.value == 0


def test_aggregates_infer_type_from_argument():
    value = sdf.load("value", ir.I64)
    pred = sdf.load("pred", ir.I1)

    total = sdf.sum(value).to_dsl()
    total_if = sdf.sum_if(value, pred).to_dsl()
    product = sdf.product(sdf.mul(value, sdf.i64(2))).to_dsl()
    count = sdf.count_if(pred).to_dsl()

    assert_is(total, ir.ArithAggExpr, None)
    assert total.op is ir.ArithBinaryOp.Add
    assert_is(total_if, ir.ArithAggExpr, None)
    assert total_if.cond is pred.to_dsl()
    assert_is(product, ir.ArithAggExpr, None)
    assert product.op is ir.ArithBinaryOp.Mul
    assert_is(count, ir.CountIfExpr, ir.I64)


def test_grouped_aggregates_infer_type_from_argument():
    value = sdf.load("value", ir.I32)
    idx = sdf.load("idx", ir.I32)
    pred = sdf.load("pred", ir.I1)

    grouped = sdf.grouped_max(value, idx, "tbl").to_dsl()
    grouped_if = sdf.grouped_and_if(value, pred, idx, "tbl").to_dsl()
    method_grouped = value.sum(group_idx=idx, table="sum_tbl").to_dsl()
    method_grouped_if = value.max(where=pred, group_idx=idx, table="max_tbl").to_dsl()
    method_count = pred.count(group_idx=idx, table="count_tbl").to_dsl()
    function_grouped = sdf.sum(value, group_idx=idx, table="sum_fn_tbl").to_dsl()
    function_count = sdf.count_if(pred, group_idx=idx, table="count_fn_tbl").to_dsl()

    assert_is(grouped, ir.GroupedArithAggExpr, None)
    assert grouped.op is ir.ArithBinaryOp.Max
    assert grouped.table == "tbl"
    assert grouped.idx is idx.to_dsl()

    assert_is(grouped_if, ir.GroupedArithAggExpr, None)
    assert grouped_if.op is ir.ArithBinaryOp.And
    assert grouped_if.cond is pred.to_dsl()

    assert_is(method_grouped, ir.GroupedArithAggExpr, None)
    assert method_grouped.op is ir.ArithBinaryOp.Add
    assert method_grouped.idx is idx.to_dsl()
    assert method_grouped.table == "sum_tbl"

    assert_is(method_grouped_if, ir.GroupedArithAggExpr, None)
    assert method_grouped_if.op is ir.ArithBinaryOp.Max
    assert method_grouped_if.cond is pred.to_dsl()
    assert method_grouped_if.table == "max_tbl"

    assert_is(method_count, ir.GroupedArithAggExpr, None)
    assert method_count.op is ir.ArithBinaryOp.Add
    assert method_count.arg.value == 1
    assert method_count.arg.dt == ir.I64
    assert method_count.cond is pred.to_dsl()
    assert method_count.table == "count_tbl"

    assert_is(function_grouped, ir.GroupedArithAggExpr, None)
    assert function_grouped.table == "sum_fn_tbl"

    assert_is(function_count, ir.GroupedArithAggExpr, None)
    assert function_count.table == "count_fn_tbl"


def test_memory_style_nodes():
    idx = sdf.load("idx", ir.I32)
    value = sdf.load("value", ir.I32)
    pred = sdf.load("pred", ir.I1)

    gathered = sdf.gather(idx, "src", ir.I32).to_dsl()
    stored = sdf.store(value, cond=pred).to_dsl()
    scattered = sdf.scatter(value, idx, cond=pred).to_dsl()
    packed = sdf.pack(value, pred).to_dsl()

    assert_is(gathered, ir.GatherExpr, ir.I32)
    assert gathered.idx is idx.to_dsl()
    assert gathered.name == "src"

    assert_is(stored, ir.StoreExpr, None)
    assert stored.cond is pred.to_dsl()

    assert_is(scattered, ir.ScatterExpr, None)
    assert scattered.idx is idx.to_dsl()
    assert scattered.cond is pred.to_dsl()

    assert_is(packed, ir.PackExpr, None)
    assert packed.cond is pred.to_dsl()
    assert packed.dst_size is None


def test_dataframe_typed_arithmetic_chain():
    t = sdf.table({"a": ir.I32, "b": ir.I32, "c": ir.I32, "d": ir.I32})
    lowered = sdf.query({"result": (t.a + t.b) * (t.c - t.d)}).to_dsl()

    assert len(lowered) == 1
    name, expr = lowered[0]
    assert name == "result"
    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.op is ir.ArithBinaryOp.Mul
    assert expr.lhs.op is ir.ArithBinaryOp.Add
    assert expr.rhs.op is ir.ArithBinaryOp.Sub
    assert expr.lhs.lhs.name == "a"
    assert expr.lhs.rhs.name == "b"
    assert expr.rhs.lhs.name == "c"
    assert expr.rhs.rhs.name == "d"


def test_dataframe_typed_bit_manipulation_chain():
    t = sdf.table({"a": ir.I16, "b": ir.I16, "c": ir.I16})
    expr = t.a.bit_and(0xFF).bit_or(t.b.shift_left(2)).bit_xor(t.c)
    lowered = sdf.query({"result": expr}).to_dsl()

    _, node = lowered[0]
    assert_is(node, ir.ArithBinaryExpr, None)
    assert node.op is ir.ArithBinaryOp.Xor
    assert node.lhs.op is ir.ArithBinaryOp.Or
    assert node.lhs.lhs.op is ir.ArithBinaryOp.And
    assert node.lhs.rhs.op is ir.ArithBinaryOp.ShiftLeftLogical
    assert node.rhs.name == "c"


def test_dataframe_typed_unary_mix():
    t = sdf.table({"x": ir.I32, "y": ir.I32, "z": ir.I32})
    expr = abs(-t.x) + t.y.popcnt() + t.z.lzcnt()
    lowered = sdf.query({"result": expr}).to_dsl()

    _, node = lowered[0]
    assert_is(node, ir.ArithBinaryExpr, None)
    assert node.op is ir.ArithBinaryOp.Add
    assert node.lhs.op is ir.ArithBinaryOp.Add
    assert node.lhs.lhs.op is ir.ArithUnaryOp.Abs
    assert node.lhs.lhs.dt == ir.I32
    assert node.lhs.lhs.arg.op is ir.ArithUnaryOp.Negate
    assert node.lhs.lhs.arg.dt == ir.I32
    assert node.lhs.rhs.op is ir.ArithUnaryOp.Popcount
    assert node.lhs.rhs.dt == ir.I32
    assert node.rhs.op is ir.ArithUnaryOp.Lzcnt
    assert node.rhs.dt == ir.I32


def test_dataframe_literals_follow_schema_types():
    t = sdf.table({"x": ir.I32, "y": ir.I32})
    lowered = sdf.query({"result": (5 * t.x) + (t.y - 3)}).to_dsl()

    _, expr = lowered[0]
    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.lhs.op is ir.ArithBinaryOp.Mul
    assert expr.lhs.lhs.value == 5
    assert expr.rhs.op is ir.ArithBinaryOp.Sub
    assert expr.rhs.rhs.value == 3


def test_dataframe_explicit_casts_lower():
    t = sdf.table({"x": ir.I16, "y": ir.I16})
    expr = t.x.cast(ir.I32) + (t.y.cast(ir.I32) * sdf.col("k", ir.I32).cast(ir.I32))
    lowered = sdf.query({"result": expr}).to_dsl()

    _, node = lowered[0]
    assert_is(node, ir.ArithBinaryExpr, None)
    assert node.lhs.kind is ir.IntCastKind.Cast
    assert node.lhs.dt == ir.I32
    assert node.rhs.op is ir.ArithBinaryOp.Mul
    assert node.rhs.lhs.kind is ir.IntCastKind.Cast
    assert node.rhs.lhs.dt == ir.I32
    assert node.rhs.rhs.dt == ir.I32


def test_dataframe_predicates_and_ifelse():
    t = sdf.table({"x": ir.I32, "y": ir.I32})
    mask = (t.x > 0) & (t.y < 100)
    lowered = sdf.query({"flag": mask.ifelse(1, 0)}).to_dsl()

    _, expr = lowered[0]
    assert_is(expr, ir.SelectExpr, None)
    assert expr.cond.op is ir.PredicateBinaryOp.And
    assert expr.cond.lhs.op is ir.CompareOp.Greater
    assert expr.cond.rhs.op is ir.CompareOp.Less
    assert expr.truthy.value == 1
    assert expr.falsy.value == 0


def test_dataframe_reductions_and_filtered_reductions():
    t = sdf.table({"value": ir.I64, "pred": ir.I1})
    lowered = sdf.query(
        {
            "sum": t.value.sum(),
            "sum_if": t.value.sum(where=t.pred),
            "product": t.value.product(),
            "count": t.pred.count(),
        }
    ).to_dsl()

    assert_is(lowered[0][1], ir.ArithAggExpr, None)
    assert lowered[0][1].op is ir.ArithBinaryOp.Add
    assert_is(lowered[1][1], ir.ArithAggExpr, None)
    assert lowered[1][1].cond.name == "pred"
    assert_is(lowered[2][1], ir.ArithAggExpr, None)
    assert lowered[2][1].op is ir.ArithBinaryOp.Mul
    assert_is(lowered[3][1], ir.CountIfExpr, ir.I64)


def test_dataframe_expr_truthiness_is_blocked():
    t = sdf.table({"x": ir.I32, "pred": ir.I1})
    for expr in (t.x, t.pred):
        try:
            bool(expr)
        except TypeError as exc:
            assert "truth value" in str(exc)
        else:
            raise AssertionError("expected dataframe Expr.__bool__ to raise TypeError")


def test_dataframe_schema_enforcement():
    t = sdf.table({"x": ir.I32})
    assert t.x.to_dsl().dt == ir.I32
    try:
        t.missing
    except AttributeError as exc:
        assert "unknown column" in str(exc)
    else:
        raise AssertionError("expected missing column access to fail")


def test_dataframe_compile_not_supported_and_grouping_not_supported():
    t = sdf.table({"x": ir.I32, "pred": ir.I1})
    program = sdf.query({"x": t.x + 1})

    try:
        program.compile()
    except NotImplementedError as exc:
        assert "not implemented" in str(exc)
    else:
        raise AssertionError("expected Program.compile() to raise NotImplementedError")

    try:
        t.x.sum(by=t.x)
    except TypeError:
        pass
    else:
        raise AssertionError("expected by-key grouped sum to be unsupported")

    try:
        t.pred.count(by=t.x)
    except TypeError:
        pass
    else:
        raise AssertionError("expected by-key grouped count to be unsupported")

    grouped_sum = t.x.sum(group_idx=t.x, table="sum_table").to_dsl()
    assert_is(grouped_sum, ir.GroupedArithAggExpr, None)
    assert grouped_sum.table == "sum_table"

    grouped_count = t.pred.count(group_idx=t.x, table="count_table").to_dsl()
    assert_is(grouped_count, ir.GroupedArithAggExpr, None)
    assert grouped_count.table == "count_table"

    try:
        t.x.sum(group_idx=t.x)
    except TypeError as exc:
        assert "group_idx requires table" in str(exc)
    else:
        raise AssertionError("expected missing grouped output table to fail")

    try:
        t.x.sum(table="sum_table")
    except TypeError as exc:
        assert "table requires group_idx" in str(exc)
    else:
        raise AssertionError("expected grouped output table without index to fail")


def test_dataframe_select_accepts_raw_dsl_expr():
    lowered = sdf.query({"raw": sdf.load("raw", ir.I32)}).to_dsl()
    assert len(lowered) == 1
    assert lowered[0][0] == "raw"
    assert_is(lowered[0][1], ir.LoadExpr, ir.I32)


def test_dataframe_select_accepts_expression_sequence():
    t = sdf.table({"x": ir.I32, "y": ir.I32})
    lowered = sdf.query((t.x + 1, t.y - 2)).to_dsl()

    assert len(lowered) == 2
    assert lowered[0][0] == "result_0"
    assert lowered[1][0] == "result_1"
    assert_is(lowered[0][1], ir.ArithBinaryExpr, None)
    assert_is(lowered[1][1], ir.ArithBinaryExpr, None)


def test_dataframe_select_accepts_single_expression():
    t = sdf.table({"x": ir.I32})
    lowered = sdf.query(t.x + 1).to_dsl()

    assert len(lowered) == 1
    assert lowered[0][0] == "result_0"
    assert_is(lowered[0][1], ir.ArithBinaryExpr, None)


def test_dataframe_select_accepts_scalar_output():
    lowered = sdf.query(7).to_dsl()

    assert len(lowered) == 1
    assert lowered[0][0] == "result_0"
    assert_is(lowered[0][1], ir.ConstExpr, None)
    assert lowered[0][1].value == 7


def test_dataframe_table_accepts_kwargs():
    t = sdf.table(x=ir.I32, y=ir.I64)
    assert t.x.to_dsl().dt == ir.I32
    assert t.y.to_dsl().dt == ir.I64


def test_dataframe_table_accepts_pyarrow_batch_table_and_schema(pa):
    batch = pa.record_batch(
        [
            make_pa_array([1, 2], pa.int32()),
            make_pa_array([True, False], pa.bool_()),
            make_pa_array([1.0, 2.0], pa.float64()),
        ],
        names=["x", "pred", "score"],
    )
    table = pa.table(
        {
            "x": make_pa_array([1, 2], pa.int32()),
            "pred": make_pa_array([True, False], pa.bool_()),
            "score": make_pa_array([1.0, 2.0], pa.float64()),
        }
    )
    schema = batch.schema

    for source in (batch, table, schema):
        t = sdf.table(source)
        assert t.x.to_dsl().dt == ir.I32
        assert t.pred.to_dsl().dt == ir.I1
        assert t.score.to_dsl().dt == ir.F64


def test_dataframe_table_accepts_pyarrow_unsigned_type(pa):
    schema = pa.schema([("u", pa.uint32())])
    t = sdf.table(schema)
    assert t.u.to_dsl().dt == ir.U32


def test_dataframe_query_accepts_kwargs():
    t = sdf.table(x=ir.I32, y=ir.I32)
    lowered = sdf.query(sum_xy=t.x + t.y, flag=(t.x > 0).ifelse(1, 0)).to_dsl()

    assert len(lowered) == 2
    assert lowered[0][0] == "sum_xy"
    assert lowered[1][0] == "flag"
    assert_is(lowered[0][1], ir.ArithBinaryExpr, None)
    assert_is(lowered[1][1], ir.SelectExpr, None)


def test_dataframe_table_and_query_reject_mixed_mapping_and_kwargs():
    assert_raises = False
    try:
        sdf.table({"x": ir.I32}, y=ir.I32)
    except TypeError as exc:
        assert "either a mapping or keyword arguments" in str(exc)
        assert_raises = True
    if not assert_raises:
        raise AssertionError("expected table mixed arguments to fail")

    assert_raises = False
    try:
        sdf.query({"x": sdf.col("x")}, y=sdf.col("y"))
    except TypeError as exc:
        assert "either outputs or keyword outputs" in str(exc)
        assert_raises = True
    if not assert_raises:
        raise AssertionError("expected query mixed arguments to fail")


def test_adhoc_typed_arithmetic_chain():
    a = sdf.col("a", ir.I32)
    b = sdf.col("b", ir.I32)
    c = sdf.col("c", ir.I32)
    d = sdf.col("d", ir.I32)
    lowered = sdf.query({"result": (a + b) * (c - d)}).to_dsl()

    _, expr = lowered[0]
    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.op is ir.ArithBinaryOp.Mul
    assert expr.lhs.op is ir.ArithBinaryOp.Add
    assert expr.rhs.op is ir.ArithBinaryOp.Sub
    assert expr.lhs.lhs.name == "a"
    assert expr.lhs.rhs.name == "b"
    assert expr.rhs.lhs.name == "c"
    assert expr.rhs.rhs.name == "d"


def test_adhoc_typed_bit_manipulation_chain():
    a = sdf.col("a", ir.I16)
    b = sdf.col("b", ir.I16)
    c = sdf.col("c", ir.I16)
    expr = a.bit_and(0xFF).bit_or(b.shift_left(2)).bit_xor(c)
    lowered = sdf.query({"result": expr}).to_dsl()

    _, node = lowered[0]
    assert_is(node, ir.ArithBinaryExpr, None)
    assert node.op is ir.ArithBinaryOp.Xor
    assert node.lhs.op is ir.ArithBinaryOp.Or
    assert node.lhs.lhs.op is ir.ArithBinaryOp.And
    assert node.lhs.rhs.op is ir.ArithBinaryOp.ShiftLeftLogical
    assert node.rhs.name == "c"


def test_adhoc_typed_unary_mix():
    x = sdf.col("x", ir.I32)
    y = sdf.col("y", ir.I32)
    z = sdf.col("z", ir.I32)
    lowered = sdf.query({"result": abs(-x) + y.popcnt() + z.lzcnt()}).to_dsl()

    _, node = lowered[0]
    assert_is(node, ir.ArithBinaryExpr, None)
    assert node.op is ir.ArithBinaryOp.Add
    assert node.lhs.op is ir.ArithBinaryOp.Add
    assert node.lhs.lhs.op is ir.ArithUnaryOp.Abs
    assert node.lhs.lhs.dt == ir.I32
    assert node.lhs.lhs.arg.op is ir.ArithUnaryOp.Negate
    assert node.lhs.lhs.arg.dt == ir.I32
    assert node.lhs.rhs.op is ir.ArithUnaryOp.Popcount
    assert node.lhs.rhs.dt == ir.I32
    assert node.rhs.op is ir.ArithUnaryOp.Lzcnt
    assert node.rhs.dt == ir.I32


def test_adhoc_untyped_expression_infers_none():
    x = sdf.col("x")
    y = sdf.col("y")
    lowered = sdf.query({"result": (5 * x) + (y - 3)}).to_dsl()

    _, expr = lowered[0]
    assert_is(expr, ir.ArithBinaryExpr, None)
    assert expr.lhs.dt is None
    assert expr.rhs.dt is None
    assert expr.lhs.lhs.value == 5
    assert expr.rhs.rhs.value == 3


def test_adhoc_untyped_expression_with_explicit_casts():
    x = sdf.col("x")
    y = sdf.col("y")
    lowered = sdf.query(
        {"result": sdf.i32(x) + (sdf.i32(y) * sdf.col("k", ir.I32))}
    ).to_dsl()

    _, node = lowered[0]
    assert_is(node, ir.ArithBinaryExpr, None)
    assert node.lhs.kind is ir.IntCastKind.Cast
    assert node.lhs.dt == ir.I32
    assert node.rhs.op is ir.ArithBinaryOp.Mul
    assert node.rhs.lhs.kind is ir.IntCastKind.Cast
    assert node.rhs.lhs.dt == ir.I32
    assert node.rhs.rhs.dt == ir.I32


def test_adhoc_cast_accepts_raw_ir_expression():
    raw = ir.LoadExpr(dt=ir.I32, name="x")
    lowered = sdf.i64(raw).to_dsl()

    assert_is(lowered, ir.IntCastExpr, ir.I64)
    assert lowered.kind is ir.IntCastKind.Cast
    assert lowered.arg is raw


def test_adhoc_predicates_and_ifelse():
    x = sdf.col("x", ir.I32)
    y = sdf.col("y", ir.I32)
    lowered = sdf.query({"flag": ((x > 0) & (y < 100)).ifelse(1, 0)}).to_dsl()

    _, expr = lowered[0]
    assert_is(expr, ir.SelectExpr, None)
    assert expr.cond.op is ir.PredicateBinaryOp.And
    assert expr.cond.lhs.op is ir.CompareOp.Greater
    assert expr.cond.rhs.op is ir.CompareOp.Less
    assert expr.truthy.value == 1
    assert expr.falsy.value == 0


def test_adhoc_reductions_and_filtered_reductions():
    value = sdf.col("value", ir.I64)
    pred = sdf.col("pred", ir.I1)
    lowered = sdf.query(
        {
            "sum": value.sum(),
            "sum_if": value.sum(where=pred),
            "product": value.product(),
            "count": pred.count(),
        }
    ).to_dsl()

    assert_is(lowered[0][1], ir.ArithAggExpr, None)
    assert lowered[0][1].op is ir.ArithBinaryOp.Add
    assert_is(lowered[1][1], ir.ArithAggExpr, None)
    assert lowered[1][1].cond.name == "pred"
    assert_is(lowered[2][1], ir.ArithAggExpr, None)
    assert lowered[2][1].op is ir.ArithBinaryOp.Mul
    assert_is(lowered[3][1], ir.CountIfExpr, ir.I64)


def test_adhoc_expr_truthiness_is_blocked():
    x = sdf.col("x", ir.I32)
    pred = sdf.col("pred", ir.I1)
    for expr in (x, pred):
        try:
            bool(expr)
        except TypeError as exc:
            assert "truth value" in str(exc)
        else:
            raise AssertionError("expected ad-hoc Expr.__bool__ to raise TypeError")


def test_runtime_program_run_rejects_python_lists():
    program = sdf.query({"x": sdf.col("x")})

    try:
        sj.run_program(program, {"x": [1, 2, 3]})
    except TypeError as exc:
        assert "buffer protocol" in str(exc)
    else:
        raise AssertionError("expected python list input to fail")


def test_runtime_program_run_rejects_unknown_output_kind():
    program = sdf.query({"x": sdf.col("x")})

    try:
        sj.run_program(
            program, {"x": np.array([1, 2, 3], dtype=np.int32)}, output="invalid"
        )
    except ValueError as exc:
        assert "output must be either" in str(exc)
    else:
        raise AssertionError("expected invalid output kind to fail")


def test_runtime_inspect_accepts_schema_without_buffers():
    program = sdf.query({"result": sdf.col("x", ir.I32) + sdf.i32(1)})

    inspection = sj.inspect(program, {"x": ir.I32}, policy="scalar")

    assert isinstance(inspection, sj.InspectionResult)
    assert "binary dtype=i32 op=add" in inspection.hir
    assert "MAIN LOOP" in inspection.mir


def test_runtime_program_run_rejects_nullable_numpy_output(pa):
    program = sdf.query(result=sdf.col("x", ir.I32))

    try:
        sj.run_program(program, {"x": make_pa_array([1, None, 3], pa.int32())})
    except ValueError as exc:
        assert "non-nullable output" in str(exc) or "null output metadata" in str(exc)
    else:
        raise AssertionError("expected nullable numpy output to fail")


def test_runtime_inspect_rejects_invalid_policy_arch_and_schema():
    program = sdf.query({"result": sdf.col("x", ir.I32) + sdf.i32(1)})

    assert_raises(
        ValueError,
        lambda: sj.inspect(program, {"x": ir.I32}, policy="turbo"),
        contains="unknown compile policy turbo",
    )
    assert_raises(
        ValueError,
        lambda: sj.inspect(program, {"x": ir.I32}, arch="riscv"),
        contains="unknown arch riscv",
    )
    assert_raises(
        TypeError,
        lambda: sj.inspect(program, 42),
        contains="inspection schema must be a mapping",
    )
    assert_raises(
        TypeError,
        lambda: sj.inspect(program, {"x": (ir.I32, True, "extra")}),
        contains="inspection schema values must be a type or (type, nullable)",
    )


def test_hir_jit_benchmark_contract():
    program = sdf.query({"result": sdf.col("x", ir.I32) + sdf.i32(1)})
    inputs = {"x": np.array([1, 2, 3], dtype=np.int32)}

    result = simjit_ext.benchmark_hir_jit_compile(
        program.to_dsl(), inputs, warmups=0, runs=1
    )
    assert result["backend"] == "asmjit"
    assert result["policy"] == "best_effort"
    assert result["llvm_opt"] is None
    assert result["compile_boundary"] == "constructed-hir-to-executable-pointer"
    assert result["compile_warmups"] == 0
    assert result["compile_runs"] == 1
    assert len(result["compile_samples_us"]) == 1
    assert result["compile_us"] == result["compile_samples_us"][0]

    if sj.inspect(program, {"x": ir.I32}).llvm_ir:
        for llvm_opt in ("O1", "O3"):
            llvm_result = simjit_ext.benchmark_hir_jit_compile(
                program.to_dsl(),
                inputs,
                backend="llvm",
                llvm_opt=llvm_opt,
                warmups=0,
                runs=1,
            )
            assert llvm_result["backend"] == "llvm"
            assert llvm_result["llvm_opt"] == llvm_opt
            assert (
                llvm_result["compile_boundary"]
                == "constructed-hir-to-executable-pointer"
            )
            assert len(llvm_result["compile_samples_us"]) == 1

    assert_raises(
        ValueError,
        lambda: simjit_ext.benchmark_hir_jit_compile(
            program.to_dsl(), inputs, warmups=-1
        ),
        contains="warmups must be non-negative",
    )
    assert_raises(
        ValueError,
        lambda: simjit_ext.benchmark_hir_jit_compile(program.to_dsl(), inputs, runs=0),
        contains="runs must be positive",
    )
    assert_raises(
        ValueError,
        lambda: simjit_ext.benchmark_hir_jit_compile(
            program.to_dsl(), inputs, backend="unknown", warmups=0, runs=1
        ),
        contains="backend must be asmjit or llvm",
    )
    assert_raises(
        ValueError,
        lambda: simjit_ext.benchmark_hir_jit_compile(
            program.to_dsl(), inputs, llvm_opt="O2", warmups=0, runs=1
        ),
        contains="llvm_opt must be O1 or O3",
    )


def test_runtime_program_rejects_invalid_input_shapes_and_metadata():
    program = sdf.query({"result": sdf.col("x", ir.I32) + sdf.i32(1)})

    assert_raises(
        TypeError,
        lambda: sj.run_program(program, 42),
        contains="Program.run inputs must be a mapping",
    )
    assert_raises(
        ValueError,
        lambda: sj.run_program(program, {}),
        contains="Program.run requires at least one input buffer",
    )
    assert_raises(
        TypeError,
        lambda: sj.run_program(
            program, {"x": np.array([1 + 2j, 3 + 4j], dtype=np.complex64)}
        ),
        contains="unsupported numpy dtype complex64",
    )
    assert_raises(
        ValueError,
        lambda: sj.run_program(program, {"x": np.arange(8, dtype=np.int32)[::2]}),
        contains="input x must be C-contiguous",
    )
    assert_raises(
        TypeError,
        lambda: sj.run_program(
            program,
            {
                "x": ir.BufferHandle(
                    ty=ir.I32,
                    buf=np.array([1, 2, 3], dtype=np.int32),
                    bitpacked=True,
                )
            },
        ),
        contains="uses bitpacked transport for non-I1 type",
    )
    assert_raises(
        TypeError,
        lambda: sj.run_program(
            program,
            {
                "x": ir.BufferHandle(
                    ty=ir.I32,
                    buf=np.array([1, 2, 3], dtype=np.int32),
                    null=ir.NullEncoding(kind="sentinel"),
                )
            },
        ),
        contains="sentinel null encoding requires a sentinel value",
    )
    assert_raises(
        TypeError,
        lambda: sj.run_program(
            program,
            {
                "x": ir.BufferHandle(
                    ty=ir.I32,
                    buf=np.array([1, 2, 3], dtype=np.int32),
                    null=ir.NullEncoding(kind="mask_bool"),
                )
            },
        ),
        contains="mask null encoding requires a mask buffer",
    )
    assert_raises(
        TypeError,
        lambda: sj.run_program(
            program,
            {
                "x": ir.BufferHandle(
                    ty=ir.I32,
                    buf=np.array([1, 2, 3], dtype=np.int32),
                    null=ir.NullEncoding(kind="unknown"),
                )
            },
        ),
        contains="uses unsupported null encoding unknown",
    )


def test_prepared_runner_rejects_bad_rebinds_and_invalidates():
    x = np.array([1, 2, 3], dtype=np.int32)
    y = np.array([10, 20, 30], dtype=np.int32)
    program = sdf.query({"result": sdf.col("x") + sdf.col("y")})
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
        lambda: runner.run({"x": x}),
        contains="missing input y for prepared program",
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": x, "y": y, "z": x}),
        contains="unexpected input z for prepared program",
    )
    assert_raises(
        ValueError,
        lambda: runner.run({"x": np.array([1, 2], dtype=np.int32), "y": y[:2]}),
        contains="length mismatch",
    )

    session.clear()
    assert_raises(
        RuntimeError,
        runner.run,
        contains="prepared program was invalidated",
    )


def test_session_rejects_non_host_architecture():
    machine = platform.machine().lower()
    incompatible = "x86" if machine in {"arm64", "aarch64"} else "arm"
    try:
        sj.Session(arch=incompatible)
    except Exception as exc:
        assert "cannot execute on host" in str(exc)
    else:
        raise AssertionError("expected non-host Session architecture to fail")
