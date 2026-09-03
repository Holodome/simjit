# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_COLUMN_LIMIT = 40
DEFAULT_SPECIAL_NAMES = ("arg", "step")


@dataclass(frozen=True)
class Token:
    type: str
    value: str


class FormatError(ValueError):
    pass


def tokenize(code: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    length = len(code)
    while i < length:
        ch = code[i]
        if ch.isspace():
            i += 1
            continue
        if ch in "()":
            tokens.append(Token("paren", ch))
            i += 1
            continue
        if ch == '"':
            start = i
            i += 1
            while i < length:
                if code[i] == "\\":
                    i += 2
                    continue
                if code[i] == '"':
                    i += 1
                    break
                i += 1
            else:
                raise FormatError("unterminated string")
            tokens.append(Token("string", code[start:i]))
            continue

        start = i
        while i < length and not code[i].isspace() and code[i] not in "()":
            i += 1
        word = code[start:i]
        if not word:
            continue
        tokens.append(Token("number" if _is_number(word) else "symbol", word))
    return tokens


def _is_number(word: str) -> bool:
    if word.startswith("-"):
        word = word[1:]
    if not word:
        return False
    if word.count(".") > 1:
        return False
    if "." in word:
        left, right = word.split(".", 1)
        return bool(left) and bool(right) and left.isdigit() and right.isdigit()
    return word.isdigit()


def parse(tokens: list[Token]) -> list[object]:
    root: list[object] = []
    stack: list[list[object]] = [root]
    for token in tokens:
        if token.type == "paren" and token.value == "(":
            new_list: list[object] = []
            stack[-1].append(new_list)
            stack.append(new_list)
            continue
        if token.type == "paren" and token.value == ")":
            if len(stack) == 1:
                raise FormatError("unmatched )")
            stack.pop()
            continue
        stack[-1].append(token.value)
    if len(stack) != 1:
        raise FormatError("unmatched (")
    return root


def inline_repr(node: object) -> str:
    if not isinstance(node, list):
        return str(node)
    if not node:
        return "()"
    return "(" + " ".join(inline_repr(child) for child in node) + ")"


def _format_node(node: object, indent: str, column_limit: int, special_names: set[str]) -> str:
    if not isinstance(node, list):
        return str(node)
    if not node:
        return "()"

    inline_version = inline_repr(node)
    first = node[0]
    is_special = not isinstance(first, list) and str(first) in special_names
    if is_special or len(inline_version) <= column_limit:
        return inline_version

    pieces = ["(" + _format_node(node[0], indent + "  ", column_limit, special_names)]
    for child in node[1:]:
        pieces.append("\n" + indent + "  " + _format_node(child, indent + "  ", column_limit, special_names))
    pieces.append(")")
    return "".join(pieces)


def format_lisp(
    code: str,
    column_limit: int = DEFAULT_COLUMN_LIMIT,
    special_names: tuple[str, ...] | list[str] = DEFAULT_SPECIAL_NAMES,
) -> str:
    special_set = set(special_names)
    tokens = tokenize(code)
    ast = parse(tokens)
    return "\n".join(_format_node(node, "", column_limit, special_set) for node in ast)
