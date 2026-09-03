# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class Column:
    name: str
    type_name: str
    nullable: bool = False


@dataclasses.dataclass(frozen=True)
class Binding:
    name: str
    expr: "Expr"


@dataclasses.dataclass(frozen=True)
class Output:
    name: str
    expr: "Expr"
    filter: "Expr | None" = None


@dataclasses.dataclass(frozen=True)
class Program:
    input_name: str
    columns: list[Column]
    bindings: list[Binding]
    outputs: list[Output]


class Expr:
    pass


@dataclasses.dataclass(frozen=True)
class Name(Expr):
    value: str


@dataclasses.dataclass(frozen=True)
class Literal(Expr):
    value: Any


@dataclasses.dataclass(frozen=True)
class Unary(Expr):
    op: str
    arg: Expr


@dataclasses.dataclass(frozen=True)
class Binary(Expr):
    op: str
    lhs: Expr
    rhs: Expr


@dataclasses.dataclass(frozen=True)
class Call(Expr):
    name: str
    args: list[Expr]
