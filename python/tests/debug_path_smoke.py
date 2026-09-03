#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import simjit as sj


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def host_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        return "arm"
    if machine in ("x86_64", "amd64"):
        return "x86"
    raise RuntimeError(f"unsupported host architecture for debug-path smoke: {machine}")


def run_cmd(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(cmd))
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {' '.join(cmd)}")
    return completed


def sample_program():
    return sj.query(result=(sj.col("x", sj.I32) + sj.i32(1)) * sj.col("y", sj.I32))


def inspect_serialized(program) -> str:
    inspection = sj.inspect(program, {"x": sj.I32, "y": sj.I32}, policy="best_effort", arch="native")
    if not inspection.serialized:
        raise RuntimeError("inspect() did not produce serialized HIR")
    return inspection.serialized


def run_and_get_serialized(program) -> str:
    session = sj.Session()
    session.debug_options.capture_on_success = True
    session.debug_options.stages = sj.DebugStage.All

    x = np.array([1, 2, 3, 4], dtype=np.int32)
    y = np.array([10, 20, 30, 40], dtype=np.int32)
    result = sj.run_program(program, {"x": x, "y": y}, session=session)
    expected = (x + 1) * y
    if result.result.tolist() != expected.tolist():
        raise RuntimeError(f"unexpected run result: {result.result.tolist()} != {expected.tolist()}")

    serialized = session.debug_snapshot.serialized
    if not serialized:
        raise RuntimeError("run_program() debug snapshot did not produce serialized HIR")
    return serialized


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def smoke_debug_path(args: argparse.Namespace) -> None:
    root = repo_root()
    simjit_cli = root / "build" / "debug" / "simjit-cli"
    local_runner = root / "build" / "debug" / "local_runner"
    objdump_asmjit = root / "scripts" / "objdump-asmjit-json.py"

    if not simjit_cli.exists():
        raise RuntimeError(f"missing simjit-cli binary: {simjit_cli}")
    if not local_runner.exists():
        raise RuntimeError(f"missing local_runner binary: {local_runner}")

    program = sample_program()
    serialized_from_inspect = inspect_serialized(program)
    serialized = serialized_from_inspect

    if not args.skip_run:
        serialized_from_run = run_and_get_serialized(program)
        if serialized_from_run != serialized_from_inspect:
            raise RuntimeError("serialized HIR differs between inspect() and run_program() debug snapshot")
        serialized = serialized_from_run

    temp_ctx = tempfile.TemporaryDirectory(prefix="simjit-debug-path-smoke-")
    temp_dir = Path(temp_ctx.name)
    try:
        case_path = temp_dir / "case.simjit"
        json_path = temp_dir / f"case-{args.arch}.jsonl"
        write_text(case_path, serialized)

        run_cmd(
            [
                str(simjit_cli),
                "--serialized-file",
                str(case_path),
                "--arch",
                args.arch,
                "--asmjit",
                "--print-hir",
                "--print-mir",
            ],
            cwd=root,
        )
        run_cmd(
            [str(simjit_cli), "--serialized-file", str(case_path), "--arch", args.arch, "--asmjit"],
            cwd=root,
        )
        run_cmd(
            [
                str(simjit_cli),
                "--serialized-file",
                str(case_path),
                "--arch",
                args.arch,
                "--dump-json",
                str(json_path),
            ],
            cwd=root,
        )
        run_cmd([str(local_runner), "--file", str(json_path)], cwd=root)
        run_cmd(
            [
                sys.executable,
                str(objdump_asmjit),
                str(json_path),
                "--arch",
                args.arch,
                "--show-raw-insn",
            ],
            cwd=root,
        )

        if args.keep_temp:
            keep_dir = Path(args.keep_temp).resolve()
            keep_dir.mkdir(parents=True, exist_ok=True)
            write_text(keep_dir / case_path.name, serialized)
            (keep_dir / json_path.name).write_bytes(json_path.read_bytes())
            print(f"kept debug smoke artifacts in {keep_dir}")
    finally:
        temp_ctx.cleanup()

    print("debug path smoke passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the serialized bug-report debug path.")
    parser.add_argument("--arch", choices=("x86", "arm"), default=host_arch())
    parser.add_argument("--skip-run", action="store_true", help="Get serialized HIR only through inspect().")
    parser.add_argument("--keep-temp", help="Copy generated .simjit and .jsonl artifacts to this directory.")
    return parser.parse_args()


def main() -> None:
    smoke_debug_path(parse_args())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"debug_path_smoke: {exc}", file=sys.stderr)
        sys.exit(1)
