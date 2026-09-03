# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

"""

Python-side simjit IR.
It is very close to simjit HIR, but allows unresolved types (at data structure level),
as well as it is treated more declaratively (i1 can be bit-packed or bool array, depending on metadata).
This IR is also used from C++-side bindings, so it has to be somewhat stable.

"""

from __future__ import annotations

import dataclasses
from typing import Literal

from ._simjit import (
    ArithBinaryOp as ArithBinaryOp,
    ArithUnaryOp as ArithUnaryOp,
    CompareOp as CompareOp,
    FpClassFlags as FpClassFlags,
    FunctionName as FunctionName,
    IntCastKind as IntCastKind,
    LoadStoreKind as LoadStoreKind,
    PredicateBinaryOp as PredicateBinaryOp,
)

TimestampUnit = Literal["s", "ms", "us", "ns"]


@dataclasses.dataclass(frozen=True)
class ScalarType:
    name: str
    unit: str | None = None
    tz: str | None = None

    def __str__(self) -> str:
        if self.name == "timestamp64":
            if self.tz is not None:
                return f"timestamp64[{self.unit}, {self.tz}]"
            return f"timestamp64[{self.unit}]"
        return self.name


I8 = ScalarType("i8")
I16 = ScalarType("i16")
I32 = ScalarType("i32")
I64 = ScalarType("i64")
U8 = ScalarType("u8")
U16 = ScalarType("u16")
U32 = ScalarType("u32")
U64 = ScalarType("u64")
F32 = ScalarType("f32")
F64 = ScalarType("f64")
I1 = ScalarType("i1")

_TIMESTAMP_UNITS = {"s", "ms", "us", "ns"}


def timestamp64(unit: TimestampUnit = "ns", tz: str | None = None) -> ScalarType:
    if unit not in _TIMESTAMP_UNITS:
        raise ValueError(f"unsupported timestamp unit {unit!r}")
    if tz is not None and tz != "UTC":
        raise ValueError(f"unsupported timestamp timezone {tz!r}")
    return ScalarType("timestamp64", unit=unit, tz=tz)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class Expr:
    dt: ScalarType | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ConstExpr(Expr):
    value: float | int | bool


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class LoadExpr(Expr):
    name: str
    kind: LoadStoreKind = LoadStoreKind.Unaligned


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class LoadSplatExpr(Expr):
    name: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class GatherExpr(Expr):
    idx: Expr
    name: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class IndexExpr(Expr):
    pass


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ArithBinaryExpr(Expr):
    op: ArithBinaryOp
    lhs: Expr
    rhs: Expr
    checked: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PredicateBinaryExpr(Expr):
    op: PredicateBinaryOp
    lhs: Expr
    rhs: Expr


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ArithUnaryExpr(Expr):
    op: ArithUnaryOp
    arg: Expr
    checked: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PredicateNotExpr(Expr):
    arg: Expr


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CompareExpr(Expr):
    op: CompareOp
    lhs: Expr
    rhs: Expr
    unsigned: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class IntCastExpr(Expr):
    kind: IntCastKind
    arg: Expr
    checked: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FloatCastExpr(Expr):
    arg: Expr
    is_unsigned: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class BitCastExpr(Expr):
    arg: Expr


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FunctionExpr(Expr):
    name: FunctionName
    args: tuple[Expr, ...]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class SelectExpr(Expr):
    cond: Expr
    truthy: Expr
    falsy: Expr


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FpClassExpr(Expr):
    arg: Expr
    flags: FpClassFlags


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PermuteExpr(Expr):
    arg: Expr
    permute_idxs: int
    is_bit: bool


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class StoreExpr(Expr):
    value: Expr
    kind: LoadStoreKind = LoadStoreKind.Unaligned
    cond: Expr | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ScatterExpr(Expr):
    value: Expr
    idx: Expr
    cond: Expr | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PackExpr(Expr):
    value: Expr
    cond: Expr
    dst_size: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ArithAggExpr(Expr):
    op: ArithBinaryOp
    arg: Expr
    cond: Expr | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class PredicateAggExpr(Expr):
    op: PredicateBinaryOp
    arg: Expr


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class CountIfExpr(Expr):
    cond: Expr


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class GroupedArithAggExpr(Expr):
    op: ArithBinaryOp
    arg: Expr
    idx: Expr
    table: str
    cond: Expr | None = None


@dataclasses.dataclass(frozen=True)
class NullEncoding:
    kind: Literal["mask_bitpacked", "mask_bool", "sentinel"]
    buf: object | None = None
    true_means_null: bool = True
    sentinel: object | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class BufferHandle:
    ty: ScalarType

    # Python 3.10 has no public Buffer protocol type.
    buf: object
    # Logical element count. Useful for non-array buffer transports like Arrow buffers.
    length: int | None = None
    # numpy is unaligned, arrow is aligned
    aligned: bool = False
    # Only for i1 type: numpy stores as bool_ (8 bits), arrow stores bitpacked
    bitpacked: bool = False
    # Optional null transport metadata.
    null: NullEncoding | None = None
