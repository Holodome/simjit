# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

# PyArrow does not currently expose complete typing metadata for these runtime APIs.
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false, reportUnknownVariableType=false
from collections.abc import Iterator
from typing import Protocol, cast

from .. import ir

try:
    import pyarrow as pa
except ImportError:
    pa = None

class _TimestampType(Protocol):
    unit: ir.TimestampUnit
    tz: object | None


class _Field(Protocol):
    name: str
    type: object


class _Schema(Protocol):
    def __iter__(self) -> Iterator[_Field]: ...


class _HasSchema(Protocol):
    schema: _Schema


def is_available() -> bool:
    return pa is not None


def is_schema_like(value: object) -> bool:
    return pa is not None and isinstance(value, (pa.RecordBatch, pa.Table, pa.Schema))


def _arrow_type_to_ir(name: str, arrow_type: object) -> ir.ScalarType:
    assert pa is not None
    if pa.types.is_timestamp(arrow_type):
        timestamp_type = cast(_TimestampType, arrow_type)
        return ir.timestamp64(
            timestamp_type.unit, "UTC" if timestamp_type.tz is not None else None
        )
    if pa.types.is_boolean(arrow_type):
        return ir.I1
    if pa.types.is_int8(arrow_type):
        return ir.I8
    if pa.types.is_int16(arrow_type):
        return ir.I16
    if pa.types.is_int32(arrow_type):
        return ir.I32
    if pa.types.is_int64(arrow_type):
        return ir.I64
    if pa.types.is_uint8(arrow_type):
        return ir.U8
    if pa.types.is_uint16(arrow_type):
        return ir.U16
    if pa.types.is_uint32(arrow_type):
        return ir.U32
    if pa.types.is_uint64(arrow_type):
        return ir.U64
    if pa.types.is_float32(arrow_type):
        return ir.F32
    if pa.types.is_float64(arrow_type):
        return ir.F64
    raise TypeError(f"column {name!r} uses unsupported Arrow type {arrow_type}")


def schema_from_pyarrow(value: object) -> dict[str, ir.ScalarType]:
    if pa is None:
        raise RuntimeError("pyarrow schema support requires pyarrow to be installed")
    if isinstance(value, pa.RecordBatch):
        schema = cast(_HasSchema, value).schema
    elif isinstance(value, pa.Table):
        schema = cast(_HasSchema, value).schema
    elif isinstance(value, pa.Schema):
        schema = cast(_Schema, value)
    else:
        raise TypeError("expected pyarrow RecordBatch, Table, or Schema")

    result: dict[str, ir.ScalarType] = {}
    for field in schema:
        result[field.name] = _arrow_type_to_ir(field.name, field.type)
    return result


def maybe_schema_from_pyarrow(value: object) -> dict[str, ir.ScalarType] | None:
    if not is_schema_like(value):
        return None
    return schema_from_pyarrow(value)
