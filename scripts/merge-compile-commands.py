#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compile_db_path(path: Path) -> Path:
    if path.is_dir():
        return path / "compile_commands.json"
    return path


def entry_file_key(entry: dict[str, Any], db_dir: Path) -> str | None:
    file_name = entry.get("file")
    if not isinstance(file_name, str) or not file_name:
        return None

    file_path = Path(file_name)
    if not file_path.is_absolute():
        directory = entry.get("directory")
        base = Path(directory) if isinstance(directory, str) and directory else db_dir
        file_path = base / file_path

    return str(file_path.resolve())


def read_compile_db(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"{path} is not a compile_commands.json array")

    entries: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object compile command")
        entries.append(item)
    return entries


def merge_compile_dbs(paths: list[Path]) -> list[dict[str, Any]]:
    entries_by_file: dict[str, dict[str, Any]] = {}

    for raw_path in paths:
        db_path = compile_db_path(raw_path)
        if not db_path.exists():
            continue

        for entry in read_compile_db(db_path):
            key = entry_file_key(entry, db_path.parent)
            if key is None:
                continue
            entries_by_file.setdefault(key, entry)

    return [entries_by_file[key] for key in sorted(entries_by_file)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge CMake compile command databases.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    merged = merge_compile_dbs(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = args.output.with_suffix(args.output.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(args.output)

    print(f"wrote {args.output} with {len(merged)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
