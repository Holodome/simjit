# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
from typing import Any

import simjit.ir as ir

import simjit as sj

from . import ast


class LowerError(ValueError):
    pass


TYPE_MAP = {
    "bool": ir.I1,
    "i1": ir.I1,
    "i8": ir.I8,
    "i16": ir.I16,
    "i32": ir.I32,
    "i64": ir.I64,
    "u8": ir.U8,
    "u16": ir.U16,
    "u32": ir.U32,
    "u64": ir.U64,
    "f32": ir.F32,
    "f64": ir.F64,
}

CASTS = {
    "i8": sj.i8,
    "i16": sj.i16,
    "i32": sj.i32,
    "i64": sj.i64,
    "u8": sj.u8,
    "u16": sj.u16,
    "u32": sj.u32,
    "u64": sj.u64,
    "f32": sj.f32,
    "f64": sj.f64,
}


@dataclasses.dataclass(frozen=True)
class LoweredQuery:
    program: sj.Program
    schema: dict[str, ir.ScalarType]
    nullable: dict[str, bool]


def dtype_from_name(name: str) -> ir.ScalarType:
    try:
        return TYPE_MAP[name.lower()]
    except KeyError as exc:
        raise LowerError(f"unsupported type {name!r}") from exc


class Lowerer:
    def __init__(self, program: ast.Program):
        self.program_ast = program
        self.schema: dict[str, ir.ScalarType] = {}
        self.nullable: dict[str, bool] = {}
        self.env: dict[str, Any] = {}

    def lower(self) -> LoweredQuery:
        for column in self.program_ast.columns:
            if column.name in self.schema:
                raise LowerError(f"duplicate input column {column.name!r}")
            dt = dtype_from_name(column.type_name)
            self.schema[column.name] = dt
            self.nullable[column.name] = column.nullable
            self.env[column.name] = sj.col(column.name, dt)

        for binding in self.program_ast.bindings:
            if binding.name in self.env:
                raise LowerError(f"duplicate name {binding.name!r}")
            self.env[binding.name] = self.expr(binding.expr)

        outputs = {}
        for output in self.program_ast.outputs:
            if output.name in outputs:
                raise LowerError(f"duplicate output {output.name!r}")
            outputs[output.name] = self.output_expr(output)
        return LoweredQuery(sj.query(outputs), dict(self.schema), dict(self.nullable))

    def output_expr(self, output: ast.Output):
        if output.filter is None:
            return self.expr(output.expr)
        return self.filtered_output(output.expr, output.filter)

    def filtered_output(self, expr: ast.Expr, filter_expr: ast.Expr):
        cond = self.expr(filter_expr)
        if not isinstance(expr, ast.Call):
            raise LowerError("FILTER can only be applied to aggregate calls")
        name = expr.name.lower()
        args = [self.expr(arg) for arg in expr.args]
        if name == "sum" and len(args) == 1:
            return sj.sum_if(args[0], cond)
        if name == "product" and len(args) == 1:
            return sj.product_if(args[0], cond)
        if name == "min" and len(args) == 1:
            return sj.min_agg_if(args[0], cond)
        if name == "max" and len(args) == 1:
            return sj.max_agg_if(args[0], cond)
        if name == "count_if" and len(args) == 1:
            return sj.count_if(sj.and_(args[0], cond))
        raise LowerError(f"FILTER is not supported for {name}")

    def expr(self, expr: ast.Expr):
        if isinstance(expr, ast.Name):
            try:
                return self.env[expr.value]
            except KeyError as exc:
                raise LowerError(f"unknown name {expr.value!r}") from exc
        if isinstance(expr, ast.Literal):
            return expr.value
        if isinstance(expr, ast.Unary):
            arg = self.expr(expr.arg)
            if expr.op == "not":
                return sj.not_(arg)
            if expr.op == "neg":
                return sj.negate(arg)
            if expr.op == "~":
                return sj.not_(arg)
            raise LowerError(f"unsupported unary operator {expr.op}")
        if isinstance(expr, ast.Binary):
            return self.binary(expr)
        if isinstance(expr, ast.Call):
            return self.call(expr)
        raise LowerError(f"unsupported expression {expr!r}")

    def binary(self, expr: ast.Binary):
        lhs = self.expr(expr.lhs)
        rhs = self.expr(expr.rhs)
        if expr.op == "+":
            return sj.add(lhs, rhs)
        if expr.op == "-":
            return sj.sub(lhs, rhs)
        if expr.op == "*":
            return sj.mul(lhs, rhs)
        if expr.op == "/":
            return sj.div(lhs, rhs)
        if expr.op == "%":
            return sj.mod(lhs, rhs)
        if expr.op == "and":
            return sj.and_(lhs, rhs)
        if expr.op == "&":
            return sj.and_(lhs, rhs)
        if expr.op == "or":
            return sj.or_(lhs, rhs)
        if expr.op == "|":
            return sj.or_(lhs, rhs)
        if expr.op == "#":
            return sj.xor(lhs, rhs)
        if expr.op == "<<":
            return sj.sll(lhs, rhs)
        if expr.op == ">>":
            return sj.sra(lhs, rhs)
        if expr.op == ">>u":
            return sj.srl(lhs, rhs)
        if expr.op in {"=", "=="}:
            return sj.cmp_eq(lhs, rhs)
        if expr.op in {"!=", "<>"}:
            return sj.cmp_ne(lhs, rhs)
        if expr.op == "<":
            return sj.cmp_lt(lhs, rhs)
        if expr.op == "<=":
            return sj.cmp_le(lhs, rhs)
        if expr.op == ">":
            return sj.cmp_gt(lhs, rhs)
        if expr.op == ">=":
            return sj.cmp_ge(lhs, rhs)
        raise LowerError(f"unsupported binary operator {expr.op}")

    def call(self, expr: ast.Call):
        name = expr.name.lower()
        args = [self.expr(arg) for arg in expr.args]
        if name in CASTS:
            self.require_arity(name, args, 1)
            return CASTS[name](args[0])
        if name == "coalesce":
            return sj.coalesce(*args)
        if name == "nullif":
            self.require_arity(name, args, 2)
            return sj.nullif(args[0], args[1])
        if name == "is_null":
            self.require_arity(name, args, 1)
            return sj.is_null(args[0])
        if name == "is_not_null":
            self.require_arity(name, args, 1)
            return sj.is_not_null(args[0])
        if name == "sum":
            self.require_arity(name, args, 1)
            return sj.sum(args[0])
        if name == "product":
            self.require_arity(name, args, 1)
            return sj.product(args[0])
        if name == "min":
            if len(args) == 1:
                return sj.min_agg(args[0])
            self.require_arity(name, args, 2)
            return sj.min(args[0], args[1])
        if name == "max":
            if len(args) == 1:
                return sj.max_agg(args[0])
            self.require_arity(name, args, 2)
            return sj.max(args[0], args[1])
        if name == "least":
            self.require_arity(name, args, 2)
            return sj.min(args[0], args[1])
        if name == "greatest":
            self.require_arity(name, args, 2)
            return sj.max(args[0], args[1])
        if name == "uleast":
            self.require_arity(name, args, 2)
            return sj.umin(args[0], args[1])
        if name == "ugreatest":
            self.require_arity(name, args, 2)
            return sj.umax(args[0], args[1])
        if name == "count_if":
            self.require_arity(name, args, 1)
            return sj.count_if(args[0])
        if name == "ifelse":
            self.require_arity(name, args, 3)
            return sj.select(args[0], args[1], args[2])
        if name == "and":
            self.require_arity(name, args, 2)
            return sj.and_(args[0], args[1])
        if name == "or":
            self.require_arity(name, args, 2)
            return sj.or_(args[0], args[1])
        if name == "not":
            self.require_arity(name, args, 1)
            return sj.not_(args[0])
        if hasattr(sj, name):
            fn = getattr(sj, name)
            return fn(*args)
        raise LowerError(f"unsupported function {name}")

    @staticmethod
    def require_arity(name: str, args: list[Any], count: int) -> None:
        if len(args) != count:
            raise LowerError(f"{name} expects {count} argument(s), got {len(args)}")


def lower(program: ast.Program) -> LoweredQuery:
    return Lowerer(program).lower()
