#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMMENT_PREFIX_BY_SUFFIX = {
    ".c": "//",
    ".cc": "//",
    ".cpp": "//",
    ".cxx": "//",
    ".h": "//",
    ".hh": "//",
    ".hpp": "//",
    ".hxx": "//",
    ".js": "//",
    ".mjs": "//",
    ".py": "#",
    ".pyi": "#",
    ".sh": "#",
    ".rkt": ";;",
}

PREFIX_BY_SHEBANG_NAME = {
    "python": "#",
    "python3": "#",
    "bash": "#",
    "sh": "#",
    "zsh": "#",
}

SKIP_DIRS = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "test-dump",
    "thirdparty",
}

SKIP_PATHS = {
    "explorer/app/static/vendor",
    # Copied third-party support headers retain their own upstream notices.
    "src/simjit/detail/expected.h",
    "src/simjit/detail/span.h",
}

HEADER_LINES = (
    "This file is part of Simjit project <https://simjit.org>",
    "",
    "See LICENSE for license and copyright information",
    "SPDX-License-Identifier: Zlib",
)

LEGACY_HEADER_LINES = (
    (
        "This file is part of simjit project <https://simjit.org>",
        "",
        "See LICENSE for license and copyright information",
        "SPDX-License-Identifier: Zlib",
    ),
)


def path_is_skipped(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1]):
        return True
    return any(rel == skipped or rel.startswith(skipped + "/") for skipped in SKIP_PATHS)


def shebang_comment_prefix(text: str) -> str | None:
    if not text.startswith("#!"):
        return None
    first_line = text.splitlines()[0]
    executable = first_line.rsplit("/", 1)[-1]
    executable = executable.split()[0]
    if executable == "env":
        parts = first_line.split()
        executable = parts[1] if len(parts) > 1 else ""
    return PREFIX_BY_SHEBANG_NAME.get(executable)


def comment_prefix_for(path: Path, text: str) -> str | None:
    return COMMENT_PREFIX_BY_SUFFIX.get(path.suffix) or shebang_comment_prefix(text)


def rendered_header(prefix: str, lines: tuple[str, ...] = HEADER_LINES) -> str:
    rendered_lines = [prefix if line == "" else f"{prefix} {line}" for line in lines]
    return "\n".join(rendered_lines) + "\n\n"


def insertion_offset(text: str, prefix: str) -> int:
    if not text.startswith("#!"):
        return 0

    first_newline = text.find("\n")
    if first_newline < 0:
        return len(text)

    offset = first_newline + 1
    if prefix == "#":
        next_newline = text.find("\n", offset)
        second_line = text[offset:] if next_newline < 0 else text[offset:next_newline]
        if "coding" in second_line and ("-*" in second_line or "coding:" in second_line):
            offset = len(text) if next_newline < 0 else next_newline + 1
    return offset


def header_end_offset(text: str, prefix: str) -> int:
    offset = insertion_offset(text, prefix)
    rendered_headers = (
        rendered_header(prefix),
        *(rendered_header(prefix, lines) for lines in LEGACY_HEADER_LINES),
    )

    while True:
        for header in rendered_headers:
            if text.startswith(header, offset):
                offset += len(header)
                break
        else:
            return offset


def add_header(path: Path, *, dry_run: bool) -> bool:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("ascii")

    prefix = comment_prefix_for(path, text)
    if prefix is None:
        return False

    offset = insertion_offset(text, prefix)
    header_end = header_end_offset(text, prefix)
    updated = text[:offset] + rendered_header(prefix) + text[header_end:]
    if updated == text:
        return False
    if not dry_run:
        path.write_text(updated, encoding="utf-8", newline="")
    return True


def iter_source_files() -> list[Path]:
    paths: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path_is_skipped(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if comment_prefix_for(path, text) is not None:
            paths.append(path)
    return sorted(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add Simjit license headers to repo-owned source files.")
    parser.add_argument("--check", action="store_true", help="Report files missing the header and exit non-zero.")
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be updated.")
    parser.add_argument("paths", nargs="*", type=Path, help="Files to check or update. Defaults to all source files.")
    args = parser.parse_args()

    paths = []
    if args.paths:
        for value in args.paths:
            path = value if value.is_absolute() else ROOT / value
            path = path.resolve()
            path.relative_to(ROOT)
            if not path.is_file():
                parser.error(f"not a file: {value}")
            paths.append(path)
    else:
        paths = iter_source_files()

    changed = []
    for path in paths:
        if add_header(path, dry_run=args.check or args.dry_run):
            changed.append(path.relative_to(ROOT).as_posix())

    for rel in changed:
        print(rel)
    if args.check and changed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
