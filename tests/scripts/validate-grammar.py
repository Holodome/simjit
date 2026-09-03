#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

"""
Validate serialized Simjit HIR s-expressions against the local Lark grammar.

Behavior:
- Reads the grammar from tests/scripts/serialization-format.lark.
- Reads one candidate serialized function from stdin, or validates serialized
  entries from a JSONL bundle with --json-file.
- Exits with status 0 on success and 1 on mismatch or grammar-load failure.
- Exits with status 2 if the required Python dependency is missing.

Dependency to install:
- lark
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GRAMMAR_PATH = ROOT / "tests" / "scripts" / "serialization-format.lark"


class MissingDependencyError(RuntimeError):
    pass


def _load_lark() -> tuple[Any, type[BaseException]]:
    try:
        from lark import Lark, UnexpectedInput
    except ModuleNotFoundError as exc:
        if exc.name == "lark":
            raise MissingDependencyError(
                "Missing Python dependency 'lark'. Install lark>=1.3 or run through scripts/py."
            ) from exc
        raise
    return Lark, UnexpectedInput


def _format_parse_error(exc: BaseException) -> str:
    pos = getattr(exc, "pos_in_stream", None)
    line = getattr(exc, "line", None)
    column = getattr(exc, "column", None)
    token = getattr(exc, "token", None)

    parts = ["Input does not match grammar"]
    if token is not None:
        parts.append(f"near token {getattr(token, 'value', token)!r}")
    if pos is not None:
        parts.append(f"at offset {pos}")
    if line is not None and column is not None:
        parts.append(f"(line {line}, column {column})")
    if len(parts) == 1:
        message = str(exc).splitlines()
        if message:
            parts.append(message[0])
    return " ".join(parts)


def build_parser() -> Any:
    if not GRAMMAR_PATH.is_file():
        raise FileNotFoundError(f"Grammar file not found: {GRAMMAR_PATH}")

    Lark, _ = _load_lark()
    return Lark.open(
        str(GRAMMAR_PATH),
        parser="lalr",
        start="document",
        maybe_placeholders=False,
        rel_to=__file__,
    )


def validate_single(parser: Any, payload: str) -> str | None:
    _, UnexpectedInput = _load_lark()
    try:
        parser.parse(payload)
    except UnexpectedInput as exc:
        return _format_parse_error(exc)
    return None


def draw_progress(processed: int, total: int, ok_count: int, failed_count: int) -> None:
    if not sys.stderr.isatty():
        return

    bar_width = 40
    filled = 0
    if total > 0:
        filled = processed * bar_width // total
    empty = bar_width - filled
    sys.stderr.write(
        f"\r[{'#' * filled}{'-' * empty}] {processed}/{total} ok={ok_count} failed={failed_count}"
    )
    sys.stderr.flush()


def clear_progress() -> None:
    if sys.stderr.isatty():
        sys.stderr.write("\r\033[2K")
        sys.stderr.flush()


def load_bundle_entries(bundle_path: Path) -> list[dict]:
    try:
        text = bundle_path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read bundle file {bundle_path}: {exc}") from exc

    stripped = text.strip()
    if not stripped:
        return []

    entries: list[dict] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse bundle JSONL line {line_no} in {bundle_path}: {exc}") from exc
        if not isinstance(entry, dict):
            raise RuntimeError(f"Bundle line {line_no} in {bundle_path} is not a JSON object")
        entries.append(entry)
    return entries


def validate_json_file(parser: Any, json_path: Path) -> int:
    try:
        entries = load_bundle_entries(json_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    tests = [entry for entry in entries if entry.get("serialized")]
    total = len(tests)
    unique_total = len({entry["serialized"] for entry in tests})
    ok_count = 0
    failed_count = 0
    cache: dict[str, str | None] = {}

    draw_progress(0, total, ok_count, failed_count)

    try:
        for idx, entry in enumerate(tests, start=1):
            test_id = entry.get("id")
            if test_id is None:
                test_id = str(entry.get("n", -1))

            payload = entry["serialized"]
            error = cache.get(payload)
            if error is None and payload not in cache:
                error = validate_single(parser, payload)
                cache[payload] = error
            if error is None:
                ok_count += 1
            else:
                clear_progress()
                print(error, file=sys.stderr)
                print(f"Serialization does not match grammar for test {test_id}", file=sys.stderr)
                failed_count += 1

            draw_progress(idx, total, ok_count, failed_count)
    except KeyboardInterrupt:
        clear_progress()
        return 130

    clear_progress()
    print(
        f"Summary: total={total} unique={unique_total} ok={ok_count} failed={failed_count}",
        file=sys.stderr,
    )
    return 0 if failed_count == 0 else 1


def main() -> int:
    json_file: Path | None = None
    if len(sys.argv) == 3 and sys.argv[1] == "--json-file":
        json_file = Path(sys.argv[2])
    elif len(sys.argv) != 1:
        print("usage: validate-grammar.py [--json-file bundle.jsonl]", file=sys.stderr)
        return 1

    try:
        parser = build_parser()
    except MissingDependencyError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Failed to load grammar: {exc}", file=sys.stderr)
        return 1

    if json_file is not None:
        return validate_json_file(parser, json_file)

    payload = sys.stdin.read()
    if not payload:
        print("No input provided on stdin", file=sys.stderr)
        return 1

    error = validate_single(parser, payload)
    if error is not None:
        print(error, file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
