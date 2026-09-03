#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from simjit_lisp_format import FormatError, format_lisp

FUZZ_DIR = Path(__file__).resolve().parent
ROOT = FUZZ_DIR.parent
RACKET_GENERATOR = FUZZ_DIR / "simjit-xsmith.rkt"
FUZZ_TOOL = ROOT / "build" / "debug" / "simjit-fuzz-tool"
LOCAL_RUNNER = ROOT / "build" / "debug" / "local_runner"
VALIDATE_GRAMMAR = ROOT / "tests" / "scripts" / "validate-grammar.py"

DEFAULT_MAX_DEPTH = 6

DEFAULT_PROGRESS_WIDTH = 28

PROFILE_FEATURES: dict[str, dict[str, bool]] = {
    "default": {
        "gather": False,
        "scatter": False,
        "pack": False,
        "permute": False,
        "sum128": False,
        "fpclass": False,
    },
    "gather-scatter": {
        "gather": True,
        "scatter": True,
        "pack": False,
        "permute": False,
        "sum128": False,
        "fpclass": False,
    },
    "full": {
        "gather": True,
        "scatter": True,
        "pack": True,
        "permute": True,
        "sum128": True,
        "fpclass": True,
    },
}

PRESET_CHOICES = ("syntax", "builder-validate", "compile-validate", "compile-observe", "scalar-diff", "vector-observe", "vector-strict")
ARCH_CHOICES = ("native", "x86", "arm")
ARTIFACT_KIND_CHOICES = ("program", "corpus", "bundle", "outcome")
RUNTIME_PRESETS = {"scalar-diff", "vector-observe", "vector-strict"}
VECTOR_PRESETS = {"compile-validate", "compile-observe", "vector-observe", "vector-strict"}


class GeneratedSyntaxError(RuntimeError):
    def __init__(self, message: str, payload: str):
        self.payload = payload
        super().__init__(message)


class ProgressBar:
    def __init__(self, total: int, *, label: str, stream=None, enabled: bool = True):
        self.total = max(total, 0)
        self.label = label
        self.stream = sys.stderr if stream is None else stream
        self.enabled = enabled and self.stream is not None
        self.is_tty = bool(self.enabled and hasattr(self.stream, "isatty") and self.stream.isatty())
        self.current = 0
        self.last_render = 0.0
        self.last_detail = ""
        self.last_line = ""
        self.line_mode_step = max(1, self.total // 20) if self.total > 0 else 1
        if self.enabled:
            self._render(force=True, detail="starting")

    def update(self, current: int, *, detail: str = "") -> None:
        if not self.enabled:
            return
        self.current = max(0, min(current, self.total)) if self.total > 0 else max(0, current)
        now = time.monotonic()
        should_force = self.current >= self.total or detail != self.last_detail
        should_render = should_force or (now - self.last_render) >= 0.1
        if not should_render:
            return
        self.last_detail = detail
        self._render(force=should_force, detail=detail)

    def done(self, *, detail: str = "done") -> None:
        if not self.enabled:
            return
        self.current = self.total
        self.last_detail = detail
        self._render(force=True, detail=detail)
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()

    def _render(self, *, force: bool, detail: str) -> None:
        self.last_render = time.monotonic()
        total = max(self.total, 1)
        ratio = min(1.0, self.current / total)
        percent = int(ratio * 100)
        detail_text = f" {detail}" if detail else ""
        if self.is_tty:
            filled = int(ratio * DEFAULT_PROGRESS_WIDTH)
            empty = DEFAULT_PROGRESS_WIDTH - filled
            line = (
                f"{self.label} "
                f"[{'#' * filled}{'.' * empty}] "
                f"{self.current}/{self.total} {percent:3d}%{detail_text}"
            )
            clear_width = max(0, len(self.last_line) - len(line))
            self.last_line = line
            self.stream.write("\r" + line + (" " * clear_width))
            self.stream.flush()
            return

        if not force and self.current not in {0, self.total} and (self.current % self.line_mode_step) != 0:
            return
        line = f"{self.label}: {self.current}/{self.total} ({percent}%)"
        if detail:
            line += f" {detail}"
        if line == self.last_line and not force:
            return
        self.last_line = line
        self.stream.write(line + "\n")
        self.stream.flush()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@functools.cache
def validate_grammar_module():
    return _load_module("simjit_validate_grammar", VALIDATE_GRAMMAR)


@functools.cache
def grammar_parser():
    return validate_grammar_module().build_parser()


def validate_serialized_text(payload: str) -> str | None:
    return validate_grammar_module().validate_single(grammar_parser(), payload)


def ensure_fuzz_tool() -> Path:
    if FUZZ_TOOL.is_file():
        return FUZZ_TOOL
    subprocess.run(["make", "fuzz-tools"], cwd=ROOT, check=True)
    if not FUZZ_TOOL.is_file():
        raise RuntimeError(f"expected fuzz tool at {FUZZ_TOOL}")
    return FUZZ_TOOL


def ensure_local_runner() -> Path:
    if LOCAL_RUNNER.is_file():
        return LOCAL_RUNNER
    subprocess.run(["make", "local-runner"], cwd=ROOT, check=True)
    if not LOCAL_RUNNER.is_file():
        raise RuntimeError(f"expected local runner at {LOCAL_RUNNER}")
    return LOCAL_RUNNER


def feature_args_for_profile(profile: str) -> list[str]:
    if profile not in PROFILE_FEATURES:
        raise RuntimeError(f"unknown profile '{profile}'")
    result: list[str] = []
    for feature, enabled in PROFILE_FEATURES[profile].items():
        result.extend([f"--with-{feature}", "true" if enabled else "false"])
    return result


def run_raw_xsmith(seed: int, profile: str, max_depth: int = DEFAULT_MAX_DEPTH) -> str:
    records = run_raw_xsmith_batch(
        base_seed=seed,
        count=1,
        start_index=0,
        profile=profile,
        max_depth=max_depth,
    )
    return records[0]["serialized"]


def run_raw_xsmith_batch(base_seed: int, count: int, start_index: int, profile: str,
                         max_depth: int = DEFAULT_MAX_DEPTH) -> list[dict[str, Any]]:
    cmd = [
        "racket",
        str(RACKET_GENERATOR),
        "--seed",
        str(base_seed),
        "--max-depth",
        str(max_depth),
        "--profile-name",
        profile,
        "--count",
        str(count),
        "--start-index",
        str(start_index),
        "--jsonl",
        *feature_args_for_profile(profile),
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = proc.stderr.strip()
        stdout = proc.stdout.strip()
        detail = stderr or stdout or "xsmith generator failed"
        raise RuntimeError(detail)
    records: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSON from xsmith generator: {exc}") from exc
        payload = record.get("serialized")
        if not isinstance(payload, str):
            raise RuntimeError("xsmith generator record is missing string 'serialized'")
        record["profile"] = profile
        records.append(record)
    if len(records) != count:
        raise RuntimeError(f"xsmith generator returned {len(records)} programs, expected {count}")
    return records


def canonicalize_program(serialized: str) -> str:
    tool = ensure_fuzz_tool()
    proc = subprocess.run(
        [str(tool), "canonicalize"],
        cwd=ROOT,
        input=serialized,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "canonicalization failed"
        raise RuntimeError(detail)
    return proc.stdout.strip()


def build_single_program(seed: int, profile: str, max_depth: int = DEFAULT_MAX_DEPTH) -> str:
    raw = run_raw_xsmith(seed=seed, profile=profile, max_depth=max_depth)
    return build_program_from_raw(raw)


def build_program_from_raw(raw: str) -> str:
    error = validate_serialized_text(raw)
    if error is not None:
        raise GeneratedSyntaxError(error, raw)
    return canonicalize_program(raw)


def build_canonical_chunk_best_effort(base_seed: int, count: int, start_index: int, profile: str,
                                      max_depth: int = DEFAULT_MAX_DEPTH) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in run_raw_xsmith_batch(
        base_seed=base_seed,
        count=count,
        start_index=start_index,
        profile=profile,
        max_depth=max_depth,
    ):
        try:
            canonical = build_program_from_raw(record["serialized"])
        except Exception as exc:
            failures.append(
                {
                    "base_seed": int(record["base_seed"]),
                    "program_index": int(record["program_index"]),
                    "program_seed": int(record["program_seed"]),
                    "profile": profile,
                    "serialized": getattr(exc, "payload", record["serialized"]),
                    "error": str(exc),
                }
            )
            continue
        result.append(
            {
                "base_seed": int(record["base_seed"]),
                "program_index": int(record["program_index"]),
                "program_seed": int(record["program_seed"]),
                "profile": profile,
                "serialized": canonical,
            }
        )
    return result, failures


def chunk_plan(start_index: int, count: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size <= 0:
        raise RuntimeError("batch_size must be positive")
    plan: list[tuple[int, int]] = []
    remaining = count
    cursor = start_index
    while remaining > 0:
        current = min(batch_size, remaining)
        plan.append((cursor, current))
        cursor += current
        remaining -= current
    return plan


def metadata_dict(
    *,
    base_seed: int = -1,
    program_index: int = -1,
    program_seed: int = -1,
    profile: str = "default",
    preset: str = "scalar-diff",
    arch: str = "native",
) -> dict[str, Any]:
    return {
        "base_seed": base_seed,
        "program_index": program_index,
        "program_seed": program_seed,
        "profile": profile,
        "preset": preset,
        "arch": arch,
    }


def empty_item(serialized: str, meta: dict[str, Any], *, scalar_status: str, vector_status: str,
               errors: dict[str, str] | None = None,
               error_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    errors = errors or {}
    error_metadata = error_metadata or {}
    variant_mode = "vector" if meta["preset"] in VECTOR_PRESETS else "scalar"
    return {
        "n": meta["program_index"],
        "id": f"xsmith-{meta['base_seed']}-{meta['program_index']}"
        if meta["base_seed"] >= 0 and meta["program_index"] >= 0
        else f"xsmith-{meta['program_seed']}",
        "suite": "xsmith",
        "variant": f"{meta['arch']}-{variant_mode}",
        "expected": "pass",
        "file": "fuzz/generated",
        "line": 0,
        "schema": {"args": []},
        "serialized": serialized,
        "codes": [],
        "base_seed": meta["base_seed"],
        "program_index": meta["program_index"],
        "program_seed": meta["program_seed"],
        "profile": meta["profile"],
        "preset": meta["preset"],
        "arch": meta["arch"],
        "hir_scalar_only": False,
        "scalar_status": scalar_status,
        "vector_status": vector_status,
        "errors": errors,
        "error_metadata": error_metadata,
    }


def compile_bundle_item(serialized: str, preset: str, meta: dict[str, Any], *,
                        wait_for_debugger: bool = False, emit_cpp: bool = False) -> tuple[dict[str, Any], bool]:
    syntax_error = validate_serialized_text(serialized)
    if syntax_error is not None:
        return (
            empty_item(
                serialized,
                meta,
                scalar_status="not-requested",
                vector_status="not-requested",
                errors={"syntax": syntax_error},
            ),
            False,
        )

    if preset == "syntax":
        return (
            empty_item(
                serialized,
                meta,
                scalar_status="not-requested",
                vector_status="not-requested",
            ),
            True,
        )

    tool = ensure_fuzz_tool()
    cmd = [
        str(tool),
        "compile",
        "--preset",
        preset,
        "--base-seed",
        str(meta["base_seed"]),
        "--program-index",
        str(meta["program_index"]),
        "--program-seed",
        str(meta["program_seed"]),
        "--profile",
        meta["profile"],
        "--arch",
        meta["arch"],
    ]
    if wait_for_debugger:
        cmd.append("--wait-for-debugger")
    if emit_cpp:
        cmd.append("--cpp")
    if wait_for_debugger:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            input=serialized,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
        )
    else:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            input=serialized,
            text=True,
            capture_output=True,
        )

    stdout = proc.stdout.strip()
    if stdout:
        item = json.loads(stdout)
    else:
        stage_error = "compiler failed"
        if proc.stderr:
            stage_error = proc.stderr.strip() or stage_error
        item = empty_item(
            serialized,
            meta,
            scalar_status="failed",
            vector_status="not-requested" if preset in {"builder-validate", "scalar-diff"} else "failed",
            errors={"stage": stage_error},
        )
    return item, proc.returncode == 0


def artifact_name(meta: dict[str, Any]) -> str:
    if meta["program_seed"] >= 0:
        return f"seed-{meta['program_seed']}-idx-{meta['program_index']}"
    return f"idx-{meta['program_index']}"


def clear_fuzz_artifacts(base_dir: Path | None = None) -> None:
    target_dir = base_dir if base_dir is not None else ROOT / "test-dump" / "fuzz"
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)


def save_artifact(bucket: str, meta: dict[str, Any], serialized: str, *, item: dict[str, Any] | None = None,
                  error: str | None = None, base_dir: Path | None = None) -> None:
    target_root = base_dir if base_dir is not None else ROOT / "test-dump" / "fuzz"
    target_dir = target_root / bucket
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = artifact_name(meta)
    artifact_payload = serialized
    try:
        artifact_payload = format_lisp(serialized)
    except FormatError:
        artifact_payload = serialized
    if artifact_payload and not artifact_payload.endswith("\n"):
        artifact_payload += "\n"
    (target_dir / f"{stem}.simjit").write_text(artifact_payload, encoding="utf-8")

    payload: dict[str, Any] = dict(meta)
    if error is not None:
        payload["error"] = error
    if item is not None:
        payload["item"] = item
    (target_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def format_timing_stats(label: str, seconds: float, count: int) -> str:
    if count <= 0:
        return f"{label}={seconds:.3f}s (items=0)"
    ms_per_item = (seconds * 1000.0) / count
    items_per_second = count / seconds if seconds > 0 else float("inf")
    return (
        f"{label}={seconds:.3f}s"
        f" ({ms_per_item:.1f} ms/item, {items_per_second:.1f} items/s, items={count})"
    )


def write_text_artifact(path: Path | str | None, payload: str) -> None:
    if path is None or str(path) == "-":
        sys.stdout.write(payload)
        if payload and not payload.endswith("\n"):
            sys.stdout.write("\n")
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")


def write_json_artifact(path: Path | str | None, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    write_text_artifact(path, text + "\n")


def write_bundle_artifact(path: Path | str | None, items: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(item, sort_keys=True) + "\n" for item in items)
    write_text_artifact(path, payload)


def write_jsonl_artifact(path: Path | str | None, records: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    write_text_artifact(path, payload)


def load_corpus_records_from_text(text: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid corpus JSON at line {line_no}: {exc}") from exc
        if not isinstance(record, dict):
            raise RuntimeError(f"invalid corpus JSON at line {line_no}: expected object")
        payload = record.get("serialized")
        if not isinstance(payload, str):
            raise RuntimeError(f"invalid corpus JSON at line {line_no}: missing string 'serialized'")
        records.append(record)
    return records


def load_bundle_items_from_text(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        raise RuntimeError("bundle input is empty")

    items = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid bundle JSONL at line {line_no}: {exc}") from exc
        if not isinstance(item, dict):
            raise RuntimeError(f"bundle item {line_no} must be a JSON object")
        items.append(item)
    return items


def minimize_program(serialized: str, roots: str) -> str:
    tool = ensure_fuzz_tool()
    proc = subprocess.run(
        [str(tool), "minimize", "--roots", roots],
        cwd=ROOT,
        input=serialized,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "minimize failed"
        raise RuntimeError(detail)
    return proc.stdout.strip()


def validate_program(serialized: str, mode: str, *, meta: dict[str, Any] | None = None) -> None:
    if mode == "syntax":
        error = validate_serialized_text(serialized)
        if error is not None:
            raise RuntimeError(error)
        return
    if mode == "builder":
        builder_meta = meta or metadata_dict(program_index=0, program_seed=0, preset="builder-validate")
        item, ok = compile_bundle_item(serialized, "builder-validate", builder_meta)
        if not ok or item.get("scalar_status") != "validated":
            errors = item.get("errors", {})
            if isinstance(errors, dict) and errors:
                raise RuntimeError("; ".join(f"{k}: {v}" for k, v in errors.items()))
            raise RuntimeError("builder validation failed")
        return
    raise RuntimeError(f"invalid validation mode '{mode}'")


def validate_corpus_records(records: list[dict[str, Any]], mode: str, *,
                            jobs: int = 0, progress: ProgressBar | None = None) -> list[tuple[int, str]]:
    failures: list[tuple[int, str]] = []
    total = len(records)
    worker_count = jobs or min(total, os.cpu_count() or 1) if total else 1
    worker_count = max(1, worker_count)

    def validate_one(index: int, record: dict[str, Any]) -> tuple[int, str | None]:
        try:
            meta = metadata_dict(
                base_seed=int(record.get("base_seed", -1)),
                program_index=int(record.get("program_index", index)),
                program_seed=int(record.get("program_seed", index)),
                profile=str(record.get("profile", "default")),
                preset="builder-validate" if mode == "builder" else "syntax",
            )
            validate_program(record["serialized"], mode, meta=meta)
            return index, None
        except Exception as exc:
            return index, str(exc)

    pending: dict[int, str | None] = {}
    next_index = 0
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(validate_one, index, record): index
            for index, record in enumerate(records)
        }
        for future in concurrent.futures.as_completed(future_map):
            index, error = future.result()
            pending[index] = error
            while next_index in pending:
                error = pending.pop(next_index)
                if error is not None:
                    failures.append((next_index, error))
                if progress is not None:
                    detail = f"idx={next_index}"
                    if error is not None:
                        detail = f"failed idx={next_index}"
                    progress.update(next_index + 1, detail=detail)
                next_index += 1
    return failures


def item_has_code(item: dict[str, Any], name: str) -> bool:
    return any(code["name"] == name for code in item.get("codes", []))


class ComparisonFailure(RuntimeError):
    pass


def run_native_stage(item: dict[str, Any], code_names: list[str]) -> None:
    missing = [name for name in code_names if not item_has_code(item, name)]
    if missing:
        raise RuntimeError("missing code bundle: " + ",".join(missing))
    command = [str(ensure_local_runner()), "--file", "-", "--workers", "1"]
    for code_name in code_names:
        command.extend(("--code", code_name))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        input=json.dumps(item, sort_keys=True) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode == 0:
        return
    detail = (completed.stderr or completed.stdout or "native local runner failed").strip()
    if " mismatches" in detail and " != " in detail:
        raise ComparisonFailure(detail)
    raise RuntimeError(detail)


def run_scalar_stage(item: dict[str, Any]):
    return run_native_stage(item, ["asmjit_s", "llvm_s"])


def run_vector_stage(item: dict[str, Any], code_name: str):
    return run_native_stage(item, ["asmjit_s", code_name])


def comparison_payload_from_exception(exc: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    symbols = getattr(exc, "symbols", None)
    equality_matrix = getattr(exc, "equality_matrix", None)
    mismatches = getattr(exc, "mismatches", None)
    if symbols:
        payload["symbols"] = symbols
    if equality_matrix:
        payload["equality_matrix"] = equality_matrix
    if mismatches:
        payload["mismatches"] = mismatches
    return payload


def bundle_success(item: dict[str, Any], ok: bool, preset: str) -> bool:
    if preset == "syntax":
        return ok
    if preset == "builder-validate":
        return ok and item.get("scalar_status") == "validated"
    if item.get("scalar_status") != "pass":
        return False
    if preset == "compile-validate":
        return item.get("vector_status") in {"pass", "scalar-only"}
    if preset == "compile-observe":
        return True
    if preset == "scalar-diff":
        return True
    vector_status = item.get("vector_status")
    if preset == "vector-observe":
        return vector_status in {"pass", "scalar-only"}
    if preset == "vector-strict":
        return vector_status == "pass"
    return ok


def compile_failure_bucket(item: dict[str, Any], preset: str) -> str:
    if preset == "syntax":
        return "syntax-fail"
    if preset == "builder-validate":
        return "builder-fail"
    if item.get("scalar_status") != "pass":
        return "scalar-fail"
    return "vector-fail"


def outcome_item_base(item: dict[str, Any]) -> dict[str, Any]:
    errors = item.get("errors", {})
    if not isinstance(errors, dict):
        errors = {"stage": str(errors)}
    error_metadata = item.get("error_metadata", {})
    if not isinstance(error_metadata, dict):
        error_metadata = {}
    return {
        "id": item.get("id", "xsmith"),
        "base_seed": item.get("base_seed", -1),
        "program_index": item.get("program_index", -1),
        "program_seed": item.get("program_seed", -1),
        "profile": item.get("profile", "default"),
        "preset": item.get("preset", "scalar-diff"),
        "arch": item.get("arch", item.get("variant", "native")),
        "serialized": item.get("serialized", ""),
        "hir_scalar_only": bool(item.get("hir_scalar_only", False)),
        "scalar_status": item.get("scalar_status", "failed"),
        "vector_status": item.get("vector_status", "not-requested"),
        "errors": errors,
        "error_metadata": error_metadata,
    }


def execute_bundle_item(item: dict[str, Any]) -> dict[str, Any]:
    outcome = outcome_item_base(item)
    preset = str(item.get("preset", "scalar-diff"))

    if outcome["scalar_status"] == "validated":
        outcome["status"] = "builder-fail"
        return outcome
    if outcome["scalar_status"] != "pass":
        outcome["status"] = "syntax-fail" if "syntax" in outcome["errors"] else "scalar-fail"
        return outcome

    vector_requested = preset in VECTOR_PRESETS
    vector_status = outcome["vector_status"]
    if preset != "compile-observe" and vector_requested and vector_status not in {"pass", "scalar-only", "not-requested"}:
        outcome["status"] = "vector-fail"
        return outcome

    try:
        run_scalar_stage(item)
    except Exception as exc:
        if exc.__class__.__name__ == "ComparisonFailure":
            outcome["status"] = "mismatch"
            outcome["comparison"] = comparison_payload_from_exception(exc)
            outcome["errors"] = {"comparison": str(exc)}
            outcome["error_metadata"] = {}
        else:
            outcome["status"] = "scalar-fail"
            outcome["errors"] = {"stage": str(exc)}
            outcome["error_metadata"] = {}
        return outcome

    if vector_requested and preset == "vector-strict":
        for code_name in ("asmjit", "llvm"):
            if not item_has_code(item, code_name):
                outcome["status"] = "vector-fail"
                outcome["errors"] = {"stage": f"missing vector code bundle: {code_name}"}
                outcome["error_metadata"] = {}
                return outcome

    if vector_requested:
        for code_name in ("asmjit", "llvm"):
            if not item_has_code(item, code_name):
                continue
            try:
                run_vector_stage(item, code_name)
            except Exception as exc:
                if exc.__class__.__name__ == "ComparisonFailure":
                    outcome["status"] = "mismatch"
                    outcome["comparison"] = comparison_payload_from_exception(exc)
                    outcome["errors"] = {"comparison": str(exc)}
                    outcome["error_metadata"] = {}
                else:
                    outcome["status"] = "vector-fail"
                    outcome["errors"] = {"stage": str(exc)}
                    outcome["error_metadata"] = {}
                return outcome

    outcome["status"] = "pass"
    outcome["errors"] = {}
    outcome["error_metadata"] = {}
    return outcome


def summarize_outcome_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "total": len(items),
        "passed": 0,
        "syntax_fail": 0,
        "builder_fail": 0,
        "scalar_fail": 0,
        "vector_fail": 0,
        "mismatch": 0,
        "crash": 0,
    }
    if items:
        summary["preset"] = items[0].get("preset")
        summary["arch"] = items[0].get("arch")
    for item in items:
        status = item.get("status")
        if status == "pass":
            summary["passed"] += 1
        elif status == "syntax-fail":
            summary["syntax_fail"] += 1
        elif status == "builder-fail":
            summary["builder_fail"] += 1
        elif status == "scalar-fail":
            summary["scalar_fail"] += 1
        elif status == "vector-fail":
            summary["vector_fail"] += 1
        elif status == "mismatch":
            summary["mismatch"] += 1
        elif status == "crash":
            summary["crash"] += 1
    return summary
