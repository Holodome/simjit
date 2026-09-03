# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
import functools
from typing import Any

from lark import Lark, Token, Transformer, UnexpectedInput
from lark.exceptions import VisitError

from . import ast


class ParseError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class _InputName:
    value: str


@dataclasses.dataclass(frozen=True)
class _Alias:
    value: str


@dataclasses.dataclass(frozen=True)
class _ParsedOutput:
    expr: ast.Expr
    filter: ast.Expr | None
    alias: str | None


def _token_value(value: Any) -> str:
    if isinstance(value, Token):
        return str(value.value)
    return str(value)


def _binary_op(value: Any) -> str:
    op = _token_value(value)
    upper = op.upper()
    if upper == "AND":
        return "and"
    if upper == "OR":
        return "or"
    return op


class _AstBuilder(Transformer):
    def program(self, children: list[Any]) -> ast.Program:
        input_name = ""
        columns: list[ast.Column] = []
        bindings: list[ast.Binding] = []
        outputs: list[ast.Output] | None = None

        for child in children:
            if isinstance(child, _InputName):
                input_name = child.value
            elif isinstance(child, list):
                if all(isinstance(item, ast.Column) for item in child):
                    columns = child
                elif all(isinstance(item, ast.Binding) for item in child):
                    bindings = child
                elif all(isinstance(item, ast.Output) for item in child):
                    outputs = child

        if outputs is None:
            raise ParseError("expected SELECT output list")
        return ast.Program(input_name, columns, bindings, outputs)

    def input_name(self, children: list[Any]) -> _InputName:
        return _InputName(_token_value(children[0]))

    def columns(self, children: list[Any]) -> list[ast.Column]:
        return list(children)

    def column(self, children: list[Any]) -> ast.Column:
        name = _token_value(children[0])
        type_name = _token_value(children[1]).lower()
        nullable = len(children) > 2
        return ast.Column(name, type_name, nullable)

    def nullable(self, children: list[Any]) -> bool:
        return True

    def with_clause(self, children: list[Any]) -> list[ast.Binding]:
        return children[0]

    def bindings(self, children: list[Any]) -> list[ast.Binding]:
        return list(children)

    def binding(self, children: list[Any]) -> ast.Binding:
        return ast.Binding(_token_value(children[0]), children[1])

    def outputs(self, children: list[Any]) -> list[ast.Output]:
        outputs: list[ast.Output] = []
        for index, child in enumerate(children):
            if not isinstance(child, _ParsedOutput):
                raise ParseError(f"unexpected output parse node {child!r}")
            name = child.alias or f"result_{index}"
            outputs.append(ast.Output(name, child.expr, child.filter))
        return outputs

    def output(self, children: list[Any]) -> _ParsedOutput:
        expr = children[0]
        filter_expr: ast.Expr | None = None
        alias: str | None = None
        for child in children[1:]:
            if isinstance(child, _Alias):
                alias = child.value
            else:
                filter_expr = child
        return _ParsedOutput(expr, filter_expr, alias)

    def filter(self, children: list[Any]) -> ast.Expr:
        return children[0]

    def alias_as(self, children: list[Any]) -> _Alias:
        return _Alias(_token_value(children[0]))

    def alias_shorthand(self, children: list[Any]) -> _Alias:
        return _Alias(_token_value(children[0]))

    def binary(self, children: list[Any]) -> ast.Expr:
        return ast.Binary(_binary_op(children[1]), children[0], children[2])

    def unary_not(self, children: list[Any]) -> ast.Expr:
        return ast.Unary("not", children[-1])

    def null_predicate(self, children: list[Any]) -> ast.Expr:
        expr = children[0]
        negated = any(isinstance(child, Token) and child.type == "NOT" for child in children[1:])
        return ast.Call("is_not_null" if negated else "is_null", [expr])

    def unary_symbol(self, children: list[Any]) -> ast.Expr:
        op = _token_value(children[0])
        expr = children[1]
        if op == "+":
            return expr
        if op == "-":
            return ast.Unary("neg", expr)
        return ast.Unary("~", expr)

    def true(self, children: list[Any]) -> ast.Expr:
        return ast.Literal(True)

    def false(self, children: list[Any]) -> ast.Expr:
        return ast.Literal(False)

    def number(self, children: list[Any]) -> ast.Expr:
        value = _token_value(children[0])
        if "." in value:
            return ast.Literal(float(value))
        return ast.Literal(int(value))

    def call(self, children: list[Any]) -> ast.Expr:
        name = _token_value(children[0]).lower()
        args: list[ast.Expr] = []
        if len(children) > 1:
            args = children[1]
        return ast.Call(name, args)

    def args(self, children: list[Any]) -> list[ast.Expr]:
        return list(children)

    def name(self, children: list[Any]) -> ast.Expr:
        return ast.Name(_token_value(children[0]))


@functools.lru_cache(maxsize=1)
def _parser() -> Lark:
    return Lark.open(
        "query.lark",
        rel_to=__file__,
        parser="lalr",
        start="program",
        maybe_placeholders=False,
        propagate_positions=True,
    )


def _format_parse_error(text: str, exc: UnexpectedInput) -> str:
    pos = exc.pos_in_stream
    if pos is None:
        pos = len(text)
    snippet = text[max(0, pos - 10) : pos]
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    got = getattr(getattr(exc, "token", None), "value", None)
    if got:
        return f"unexpected token {got!r} at byte {pos} (line {line}, column {column}): `{snippet}`"
    return f"unexpected syntax at byte {pos} (line {line}, column {column}): `{snippet}`"


def parse(text: str) -> ast.Program:
    try:
        tree = _parser().parse(text)
    except UnexpectedInput as exc:
        raise ParseError(_format_parse_error(text, exc)) from exc
    try:
        return _AstBuilder().transform(tree)
    except VisitError as exc:
        if isinstance(exc.orig_exc, ParseError):
            raise exc.orig_exc from exc
        raise ParseError(str(exc.orig_exc)) from exc
    except ParseError:
        raise
    except Exception as exc:
        raise ParseError(str(exc)) from exc
