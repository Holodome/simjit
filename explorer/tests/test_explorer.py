# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import base64
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import simjit as sj

import explorer.app.samples as samples_module
import explorer.app.bench_worker as bench_worker_module
import explorer.app.service as service_module
import explorer.app.targets as targets_module
from explorer.app import querylang
from explorer.app.benchmark_model import (
    BenchmarkErrorResult,
    BenchmarkImplementationRow,
    BenchmarkModelError,
    BenchmarkRunResult,
    coerce_benchmark_result,
    to_json_dict,
)
from explorer.app.bench_worker import (
    _add_bar_widths,
    gbench_pointer_types,
    make_inputs,
    parse_cpp_pointer_types,
    parse_hir_pointer_types,
    time_to_us,
    wrap_asmjit_machine_code,
    wrap_cpp_source,
)
from explorer.app.limits import LimitError, PublicLimits, public_limits, validate_benchmark_payload
from explorer.app.lisp_format import format_lisp
from explorer.app.querylang import ast as query_ast
from explorer.app.samples import DEFAULT_SAMPLE_ID, SAMPLE_GROUPS, SAMPLES, SAMPLES_BY_ID
from explorer.app.service import (
    DEFAULT_QUERY,
    BenchmarkOptions,
    benchmark_availability,
    benchmark_availability_matrix,
    compile_query,
    format_asmjit_text,
    format_mir_text,
    is_benchmark_arch_runnable,
    platform_choices,
    run_benchmark,
)
from explorer.app.targets import BenchmarkTarget, TargetProbe


def _run_serialization_validator(payload: str):
    root = Path(__file__).resolve().parents[2]
    script = root / "tests" / "scripts" / "validate-grammar.py"
    return subprocess.run(
        [os.sys.executable, str(script)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def test_serialization_lark_validator_cases():
    valid_payloads = [
        '(func (args) (steps (step 0 const i32 "0x1")) (roots (step 0)))',
        (
            '(func (args) (steps (step 0 const i32 "0x1") '
            '(step 1 permute i32 (step 0) "0x01" ())) (roots (step 1)))'
        ),
        (
            '(func (args (arg 0 f64 src-arr)) '
            '(steps (step 0 load f64 (arg 0) unaligned) '
            '(step 1 fpclass i1 (step 0) ("inf" "nan" sub zer))) '
            '(roots (step 1)))'
        ),
    ]
    for payload in valid_payloads:
        result = _run_serialization_validator(payload)
        assert result.returncode == 0, result.stderr

    invalid_payloads = [
        "(func (args))",
        '(func (args) (steps (step 0 const nope "0x1")) (roots (step 0)))',
        '(func (args) (steps (step 0 const i32 "0x1")) (roots (step 0))) trailing',
    ]
    for payload in invalid_payloads:
        result = _run_serialization_validator(payload)
        assert result.returncode == 1
        assert "Input does not match grammar" in result.stderr


def test_querylang_lowers_expression_sql():
    parsed = querylang.parse(
        """INPUT (x i32, y i32?);
        WITH z = coalesce(y, 0)
        SELECT x + z result, sum(z) FILTER (WHERE x > 0) total;"""
    )
    lowered = querylang.lower(parsed)
    outputs = lowered.program.to_dsl()

    assert parsed.input_name == ""
    assert lowered.schema["x"].name == "i32"
    assert lowered.nullable["y"] is True
    assert [name for name, _ in outputs] == ["result", "total"]

    legacy_input_name = querylang.parse("INPUT t (x i32); SELECT x;")
    assert legacy_input_name.input_name == "t"
    assert legacy_input_name.outputs[0].name == "result_0"

    explicit_as = querylang.parse("INPUT t (x i32); SELECT x AS y;")
    shorthand = querylang.parse("INPUT t (x i32); SELECT x y;")
    assert explicit_as.outputs == shorthand.outputs

    unnamed = querylang.parse(
        """INPUT t (x i32);
        SELECT x + 1, sum(x) FILTER (WHERE x > 0);"""
    )
    assert [output.name for output in unnamed.outputs] == ["result_0", "result_1"]
    unnamed_lowered = querylang.lower(unnamed)
    assert [name for name, _ in unnamed_lowered.program.to_dsl()] == [
        "result_0",
        "result_1",
    ]


def test_querylang_allows_keyword_function_names_for_bitwise_samples():
    parsed = querylang.parse("INPUT t (x i32, y i32); SELECT and(x, y) AS v;")
    lowered = querylang.lower(parsed)
    assert [name for name, _ in lowered.program.to_dsl()] == ["v"]


def test_querylang_lark_parser_edges():
    parsed = querylang.parse(
        """-- parser edge coverage
        INPUT t (a i32, b i32?, c i32);
        SELECT
          a + b * c AS add_mul,
          (a + b) * c grouped_mul,
          a < b OR b > c AND NOT FALSE AS pred,
          and(a, b) AS keyword_fn,
          -a + ~b AS unary_mix;"""
    )

    assert parsed.input_name == "t"
    assert parsed.columns[1].nullable is True
    assert [output.name for output in parsed.outputs] == [
        "add_mul",
        "grouped_mul",
        "pred",
        "keyword_fn",
        "unary_mix",
    ]

    add_mul = parsed.outputs[0].expr
    assert isinstance(add_mul, query_ast.Binary)
    assert add_mul.op == "+"
    assert isinstance(add_mul.rhs, query_ast.Binary)
    assert add_mul.rhs.op == "*"

    grouped = parsed.outputs[1].expr
    assert isinstance(grouped, query_ast.Binary)
    assert grouped.op == "*"
    assert isinstance(grouped.lhs, query_ast.Binary)
    assert grouped.lhs.op == "+"

    pred = parsed.outputs[2].expr
    assert isinstance(pred, query_ast.Binary)
    assert pred.op == "or"
    assert isinstance(pred.rhs, query_ast.Binary)
    assert pred.rhs.op == "and"
    assert isinstance(pred.rhs.rhs, query_ast.Unary)
    assert pred.rhs.rhs.op == "not"

    keyword_fn = parsed.outputs[3].expr
    assert isinstance(keyword_fn, query_ast.Call)
    assert keyword_fn.name == "and"

    unary_mix = parsed.outputs[4].expr
    assert isinstance(unary_mix, query_ast.Binary)
    assert unary_mix.op == "+"
    assert isinstance(unary_mix.lhs, query_ast.Unary)
    assert unary_mix.lhs.op == "neg"
    assert isinstance(unary_mix.rhs, query_ast.Unary)
    assert unary_mix.rhs.op == "~"

    try:
        querylang.parse("INPUT t (x i32);\nSELECT x + ;")
    except querylang.ParseError as exc:
        message = str(exc)
        assert "byte" in message
        assert "line" in message
        assert "column" in message
    else:
        raise AssertionError("expected syntax error")


def test_querylang_lowers_sql_minmax_aliases():
    parsed = querylang.parse(
        """INPUT t (x i32, y i32);
        SELECT
          least(x, y) lo,
          greatest(x, y) hi,
          uleast(x, y) ulo,
          ugreatest(x, y) uhi;"""
    )
    lowered = querylang.lower(parsed)
    assert [name for name, _ in lowered.program.to_dsl()] == ["lo", "hi", "ulo", "uhi"]


def test_querylang_lowers_sql_null_predicates():
    parsed = querylang.parse(
        """INPUT t (x i32?, y i32?);
        WITH
          present = y IS NOT NULL
        SELECT
          x IS NULL AS missing_x,
          present AS present_y,
          is_not_null(y) AS legacy_present_y;"""
    )
    lowered = querylang.lower(parsed)
    assert [name for name, _ in lowered.program.to_dsl()] == [
        "missing_x",
        "present_y",
        "legacy_present_y",
    ]


def test_querylang_preserves_symbolic_operator_syntax():
    parsed = querylang.parse(
        """INPUT t (a i32, b i32);
        SELECT
          (a & b) | (a # b) AS mix,
          ~a AS inverted,
          a << 1 AS shifted_left,
          a >> 2 AS shifted_right;"""
    )
    mix = parsed.outputs[0].expr
    assert isinstance(mix, query_ast.Binary)
    assert mix.op == "|"
    assert isinstance(mix.lhs, query_ast.Binary)
    assert mix.lhs.op == "&"
    assert isinstance(mix.rhs, query_ast.Binary)
    assert mix.rhs.op == "#"
    inverted = parsed.outputs[1].expr
    assert isinstance(inverted, query_ast.Unary)
    assert inverted.op == "~"
    shifted_left = parsed.outputs[2].expr
    assert isinstance(shifted_left, query_ast.Binary)
    assert shifted_left.op == "<<"
    shifted_right = parsed.outputs[3].expr
    assert isinstance(shifted_right, query_ast.Binary)
    assert shifted_right.op == ">>"

    lowered = querylang.lower(parsed)
    assert [name for name, _ in lowered.program.to_dsl()] == [
        "mix",
        "inverted",
        "shifted_left",
        "shifted_right",
    ]


def test_sample_catalog_is_grouped_and_parseable():
    ids = [sample.id for sample in SAMPLES]
    assert len(ids) == len(set(ids))
    assert DEFAULT_SAMPLE_ID in SAMPLES_BY_ID
    assert samples_module.DEFAULT_QUERY == SAMPLES_BY_ID[DEFAULT_SAMPLE_ID].query

    for group in SAMPLE_GROUPS:
        assert group.samples
        for sample in group.samples:
            assert sample.query.strip()
            assert sample.group == group.title
            assert sample.input_mode == "expression_sql"
            parsed = querylang.parse(sample.query)
            lowered = querylang.lower(parsed)
            assert lowered.program.to_dsl()


def test_compile_query_reports_outputs():
    result = compile_query(DEFAULT_QUERY)
    assert result.ok, result.error
    assert result.sections["Serialized HIR"]
    assert "\n  (args" in result.sections["Serialized HIR"]
    assert result.sections["HIR"]
    assert result.sections["MIR"]
    invalid = compile_query(DEFAULT_QUERY, arch="sparc")
    assert not invalid.ok
    assert "unsupported platform" in invalid.error

    serialized_result = compile_query(
        result.sections["Serialized HIR"],
        input_mode="serialized_hir",
    )
    assert serialized_result.ok, serialized_result.error
    assert serialized_result.sections["HIR"]
    assert serialized_result.sections["MIR"]


def test_lisp_formatter_keeps_serialized_atoms_readable():
    formatted = format_lisp(
        "(func (args (arg 0 i32 src-arr) (arg 1 i32 dst-arr)) "
        '(steps (step 0 const i32 "0x1") (step 1 store i32 (step 0) (arg 1) unaligned)) '
        "(roots (step 1)))"
    )
    assert formatted.startswith("(func\n")
    assert "(arg 0 i32 src-arr)" in formatted
    assert '(step 0 const i32 "0x1")' in formatted


def test_codegen_formatters_indent_body_lines():
    mir = format_mir_text(
        "PROLOGUE\n%1 <- const dtype=i32 value=1\nMAIN LOOP\n%2 <- load"
    )
    assert mir == "PROLOGUE\n  %1 <- const dtype=i32 value=1\nMAIN LOOP\n  %2 <- load"

    asm = format_asmjit_text(
        ".section .text {#0}\nL1:\nmov x0, x1\nmain_loop:\nadd x0, x0, 1"
    )
    assert asm == ".section .text {#0}\nL1:\n  mov x0, x1\nmain_loop:\n  add x0, x0, 1"


def test_benchmark_helpers_use_us_and_cpp_signature():
    cpp = "void expr(size_t nelems, const int32_t * __restrict arg0, uint8_t * __restrict arg1) {}"
    assert parse_cpp_pointer_types(cpp) == ["int32_t", "uint8_t"]
    hir = """\
@0 arg dtype=i32 kind=src-arr
@1 arg dtype=i1 kind=dst-arr
"""
    assert parse_hir_pointer_types(hir) == ["int32_t", "uint8_t"]
    assert gbench_pointer_types([], hir) == ["int32_t", "uint8_t"]
    assert gbench_pointer_types(
        [("cpp", "cpp", "cpp", cpp, False, None)], hir
    ) == ["int32_t", "uint8_t"]
    wrapped = wrap_cpp_source("void expr(size_t nelems) {}")
    assert "#include <immintrin.h>" in wrapped
    assert "#include <arm_neon.h>" in wrapped
    wrapped_asm = wrap_asmjit_machine_code(b"\x90\xc3", "expr_asmjit")
    assert "section" in wrapped_asm
    assert "unsigned char expr_asmjit[]" in wrapped_asm
    assert "0x90, 0xc3" in wrapped_asm
    assert time_to_us(42, "ns") == 0.042
    assert time_to_us(42, "us") == 42
    assert time_to_us(42, "ms") == 42_000
    original_march = os.environ.get(bench_worker_module.LLVM_TARGET_MARCH_ENV)
    try:
        os.environ[bench_worker_module.LLVM_TARGET_MARCH_ENV] = "none"
        assert bench_worker_module.llvm_clang_compile_flags(False) == [
            "-ffast-math",
            "-O1",
        ]
        assert bench_worker_module.llvm_clang_compile_flags(True) == [
            "-ffast-math",
            "-O3",
        ]
        os.environ[bench_worker_module.LLVM_TARGET_MARCH_ENV] = "znver4"
        assert bench_worker_module.llvm_clang_compile_flags(False) == [
            "-ffast-math",
            "-O1",
            "-march=znver4",
        ]
        os.environ[bench_worker_module.LLVM_TARGET_MARCH_ENV] = "zen4"
        assert bench_worker_module.llvm_clang_compile_flags(False) == [
            "-ffast-math",
            "-O1",
            "-march=znver4",
        ]
    finally:
        if original_march is None:
            os.environ.pop(bench_worker_module.LLVM_TARGET_MARCH_ENV, None)
        else:
            os.environ[bench_worker_module.LLVM_TARGET_MARCH_ENV] = original_march
    assert bench_worker_module.LLVM_BASELINE_PROBE_OPT == "O1"
    assert bench_worker_module.cpp_compile_flags(False) == ["-O1"]
    assert bench_worker_module.cpp_compile_flags(True) == ["-O3"]

    rows = [
        {"hot_us": 2.0, "throughput_gbps": 10.0, "rows_per_second": 100.0},
        {"hot_us": 1.0, "throughput_gbps": 20.0, "rows_per_second": 200.0},
    ]
    _add_bar_widths(rows)
    assert rows[0]["bar_percent"] == 50.0
    assert rows[1]["bar_percent"] == 100.0
    assert rows[1]["bar_label"] == "20.00 GB/s"
    mixed_rows = [
        {"name": "Simjit scalar / LLVM"},
        {"name": "Simjit scalar / C++"},
        {"name": "Simjit scalar / C++ O3"},
        {"name": "Simjit SIMD / LLVM"},
        {"name": "Simjit scalar / AsmJit"},
        {"name": "Simjit SIMD / C++"},
        {"name": "Simjit SIMD / AsmJit"},
    ]
    assert [
        row["name"]
        for row in bench_worker_module._ordered_implementation_rows(mixed_rows)
    ] == [
        "Simjit SIMD / AsmJit",
        "Simjit scalar / AsmJit",
        "Simjit SIMD / C++",
        "Simjit scalar / C++",
        "Simjit scalar / C++ O3",
        "Simjit SIMD / LLVM",
        "Simjit scalar / LLVM",
    ]
    metadata = bench_worker_module._implementation_metadata(
        "Simjit SIMD / Python API", "asmjit_vectorized_py"
    )
    assert metadata == {
        "api": "Python",
        "compile_mode": "SIMD",
        "backend": "AsmJit",
        "opt": "-",
    }


def test_prepare_measurement_always_discards_one_warmup():
    prepare_calls = 0

    class FakeSession:
        def __init__(self, *, arch):
            self.arch = arch

        def prepare_program(self, outputs, inputs, output):
            nonlocal prepare_calls
            prepare_calls += 1

    original_session = bench_worker_module.sj.Session
    try:
        bench_worker_module.sj.Session = FakeSession
        bench_worker_module.measure_python_prepare_us(
            None,
            [],
            {},
            "pyarrow",
            "native",
            warmups=0,
            runs=3,
        )
    finally:
        bench_worker_module.sj.Session = original_session

    assert prepare_calls == 4


def test_llvm_compile_probe_uses_warmed_median():
    calls = []
    samples = {
        ("scalar-ir", "O1"): [100.0, 7.0, 5.0, 6.0, 8.0, 9.0],
        ("scalar-ir", "O3"): [100.0, 17.0, 15.0, 16.0, 18.0, 19.0],
        ("vector-ir", "O1"): [100.0, 27.0, 25.0, 26.0, 28.0, 29.0],
    }

    class FakeProbe:
        def compile_ir(self, llvm_ir, *, opt):
            calls.append((llvm_ir, opt))
            return {
                "compile_us": samples[(llvm_ir, opt)].pop(0),
                "compile_boundary": "llvm-ir-to-executable-pointer",
            }

    original_import_module = bench_worker_module.importlib.import_module
    try:
        bench_worker_module.importlib.import_module = lambda name: FakeProbe()
        diagnostics = []
        timings = bench_worker_module.probe_llvm_compile_times(
            "vector-ir", "scalar-ir", diagnostics
        )
    finally:
        bench_worker_module.importlib.import_module = original_import_module

    assert diagnostics == []
    assert timings == {
        "llvm_scalar": {
            "compile_us": 7.0,
            "compile_samples_us": [7.0, 5.0, 6.0, 8.0, 9.0],
            "compile_warmups": 1,
            "compile_runs": 5,
            "compile_boundary": "llvm-ir-to-executable-pointer",
            "backend": "llvm",
            "llvm_opt": "O1",
        },
        "llvm_scalar_o3": {
            "compile_us": 17.0,
            "compile_samples_us": [17.0, 15.0, 16.0, 18.0, 19.0],
            "compile_warmups": 1,
            "compile_runs": 5,
            "compile_boundary": "llvm-ir-to-executable-pointer",
            "backend": "llvm",
            "llvm_opt": "O3",
        },
        "llvm_vectorized": {
            "compile_us": 27.0,
            "compile_samples_us": [27.0, 25.0, 26.0, 28.0, 29.0],
            "compile_warmups": 1,
            "compile_runs": 5,
            "compile_boundary": "llvm-ir-to-executable-pointer",
            "backend": "llvm",
            "llvm_opt": "O1",
        },
    }
    assert calls.count(("scalar-ir", "O1")) == 6
    assert calls.count(("scalar-ir", "O3")) == 6
    assert calls.count(("vector-ir", "O1")) == 6


def _benchmark_row_payload(**overrides):
    row = {
        "name": "Simjit SIMD / Python API",
        "key": "asmjit_vectorized_py",
        "source": "Python API",
        "api": "Python",
        "compile_mode": "SIMD",
        "backend": "AsmJit",
        "opt": "-",
        "chart_label": "Python · SIMD · AsmJit",
        "main": True,
        "hot_us": 1.0,
        "compile_us": 2.0,
        "rows_per_second": 3.0,
        "throughput_gbps": 4.0,
        "speedup": None,
        "bar_percent": 100.0,
        "bar_label": "4.00 GB/s",
    }
    row.update(overrides)
    return row


def _benchmark_result_payload(**overrides):
    result = {
        "ok": True,
        "rows": 8,
        "warmups": 0,
        "runs": 1,
        "null_density": 0.0,
        "output": "pyarrow",
        "arch": "native",
        "provider": "python",
        "provider_label": "Python API",
        "compile_us": 2.0,
        "median_us": 1.0,
        "compile_ms": 0.002,
        "median_ms": 0.001,
        "rows_per_second": 3.0,
        "throughput_gbps": 4.0,
        "cache_hits": 0,
        "cache_misses": 1,
        "scalar_outputs": [],
        "implementations": [_benchmark_row_payload()],
        "diagnostics": [],
        "google_benchmark": None,
    }
    result.update(overrides)
    return result


def test_benchmark_model_coerces_worker_payloads():
    result = coerce_benchmark_result(_benchmark_result_payload())

    assert isinstance(result, BenchmarkRunResult)
    assert result.ok is True
    assert result.implementations[0].name == "Simjit SIMD / Python API"
    assert result.implementations[0].key == "asmjit_vectorized_py"
    assert to_json_dict(result)["implementations"][0]["chart_label"] == "Python · SIMD · AsmJit"

    error = coerce_benchmark_result({"ok": False, "error": "worker failed"})
    assert isinstance(error, BenchmarkErrorResult)
    assert error.error == "worker failed"

    broken_row = _benchmark_row_payload()
    broken_row.pop("key")
    try:
        coerce_benchmark_result(_benchmark_result_payload(implementations=[broken_row]))
    except BenchmarkModelError as exc:
        assert "missing required field 'key'" in str(exc)
    else:
        raise AssertionError("expected missing benchmark row key to fail")


def test_python_benchmark_displays_simd_above_scalar():
    result = _run_stubbed_python_benchmark()
    names = [row.name for row in result.implementations]
    assert names == ["Simjit SIMD / Python API", "Simjit scalar / Python API"]
    assert [row.chart_label for row in result.implementations] == [
        "Python · SIMD · AsmJit",
        "Python · Scalar · AsmJit",
    ]
    assert [row.main for row in result.implementations] == [True, True]
    assert not any("best-effort" in name for name in names)
    assert result.median_us == 1.0
    assert result.implementations[0].bar_percent == 100.0


def test_python_benchmark_omits_failed_simd_row():
    result = _run_stubbed_python_benchmark(fail_simd=True)
    names = [row.name for row in result.implementations]
    assert names == ["Simjit scalar / Python API"]
    assert not any("best-effort" in name for name in names)
    assert any("Simjit SIMD / Python API skipped" in item for item in result.diagnostics)
    assert result.median_us == 2.0


def test_all_provider_keeps_python_and_google_rows():
    result = _run_stubbed_python_benchmark(provider="all")
    pairs = [(row.chart_label, row.source) for row in result.implementations]

    assert pairs == [
        ("Python · SIMD · AsmJit", "Python API"),
        ("C++ · SIMD · AsmJit", "Google Benchmark"),
        ("Python · Scalar · AsmJit", "Python API"),
        ("C++ · Scalar · AsmJit", "Google Benchmark"),
        ("C++ · SIMD · C++ emitter · O1", "Google Benchmark"),
    ]
    assert [row.main for row in result.implementations] == [
        True,
        True,
        True,
        True,
        False,
    ]
    assert len({row.key for row in result.implementations}) == len(
        result.implementations
    )
    compile_by_key = {
        row.key: row.compile_us for row in result.implementations
    }
    assert compile_by_key["asmjit_vectorized_py"] == 15.0
    assert compile_by_key["asmjit_vectorized"] == 16.0
    assert compile_by_key["asmjit_scalar_py"] == 25.0
    assert compile_by_key["asmjit_scalar"] == 26.0


def _run_stubbed_python_benchmark(
    *,
    fail_simd: bool = False,
    provider: str = "python",
):
    class DummyProgram:
        def to_dsl(self):
            return []

    lowered = SimpleNamespace(schema={"x": sj.I32}, nullable={}, program=DummyProgram())
    originals = {
        "parse": bench_worker_module.querylang.parse,
        "lower": bench_worker_module.querylang.lower,
        "make_inputs": bench_worker_module.make_inputs,
        "run_python_api_kernel": bench_worker_module.run_python_api_kernel,
        "measure_python_prepare_us": bench_worker_module.measure_python_prepare_us,
        "benchmark_hir_jit_compile": bench_worker_module.sj_ext.benchmark_hir_jit_compile,
        "inspect": bench_worker_module.sj.inspect,
        "capture_asmjit_machine_code": bench_worker_module.capture_asmjit_machine_code,
        "run_google_benchmark": bench_worker_module.run_google_benchmark,
    }

    def fake_run_python_api_kernel(
        name,
        key,
        policy,
        outputs,
        inputs,
        output,
        rows,
        bytes_processed,
        warmups,
        runs,
        arch,
        compile_us=None,
    ):
        if policy == sj.CompilePolicy.Vectorized and fail_simd:
            raise RuntimeError("unsupported vectorization")
        hot_us = 1.0 if policy == sj.CompilePolicy.Vectorized else 2.0
        compile_us = compile_us or (10.0 if policy == sj.CompilePolicy.Vectorized else 20.0)
        return {
            "compile_us": compile_us,
            "median_us": hot_us,
            "cache_hits": 0,
            "cache_misses": 1,
            "row": bench_worker_module._implementation_row(
                name,
                key,
                hot_us,
                rows,
                bytes_processed,
                compile_us=compile_us,
                source="Python API",
            ),
        }

    def fake_google_benchmark(
        vectorized_inspection,
        scalar_inspection,
        *,
        asmjit_variants,
        rows,
        repetitions,
        bytes_processed,
        jit_compile_measurements,
    ):
        assert set(jit_compile_measurements) == {
            "asmjit_scalar",
            "asmjit_vectorized",
        }
        compile_times = {item["key"]: item["compile_us"] for item in asmjit_variants}
        return {
            "ok": True,
            "diagnostics": [],
            "implementations": [
                bench_worker_module._implementation_row(
                    "Simjit SIMD / AsmJit",
                    "asmjit_vectorized",
                    1.5,
                    rows,
                    bytes_processed,
                    compile_us=compile_times["asmjit_vectorized"],
                    source="Google Benchmark",
                ),
                bench_worker_module._implementation_row(
                    "Simjit scalar / AsmJit",
                    "asmjit_scalar",
                    2.5,
                    rows,
                    bytes_processed,
                    compile_us=compile_times["asmjit_scalar"],
                    source="Google Benchmark",
                ),
                bench_worker_module._implementation_row(
                    "Simjit SIMD / C++",
                    "cpp_vectorized",
                    3.0,
                    rows,
                    bytes_processed,
                    compile_us=None,
                    source="Google Benchmark",
                ),
            ],
        }

    try:
        bench_worker_module.querylang.parse = lambda query: object()
        bench_worker_module.querylang.lower = lambda parsed: lowered
        bench_worker_module.make_inputs = lambda schema, nullable, rows, null_density: {}
        bench_worker_module.run_python_api_kernel = fake_run_python_api_kernel
        prepare_counts = {
            sj.CompilePolicy.Vectorized: 0,
            sj.CompilePolicy.Scalar: 0,
        }

        def fake_measure_python_prepare_us(
            policy, outputs, inputs, output, arch, warmups, runs
        ):
            base = 15.0 if policy == sj.CompilePolicy.Vectorized else 25.0
            value = base + prepare_counts[policy]
            prepare_counts[policy] += 1
            return value

        bench_worker_module.measure_python_prepare_us = fake_measure_python_prepare_us

        def fake_benchmark_hir_jit_compile(
            outputs, inputs, output, backend, policy, llvm_opt, arch, warmups, runs
        ):
            assert backend == "asmjit"
            assert llvm_opt == "O1"
            return {
                "compile_us": 16.0
                if policy == sj.CompilePolicy.Vectorized
                else 26.0,
                "compile_boundary": "constructed-hir-to-executable-pointer",
            }

        bench_worker_module.sj_ext.benchmark_hir_jit_compile = (
            fake_benchmark_hir_jit_compile
        )

        def fake_inspect(program, schema, output, policy, arch):
            assert output == "pyarrow"
            return SimpleNamespace(
                cpp="void expr(size_t nelems, int32_t * arg0) {}",
                llvm_ir="",
                asmjit_compile_us=9000.0,
            )

        bench_worker_module.sj.inspect = fake_inspect
        bench_worker_module.capture_asmjit_machine_code = (
            lambda name, key, policy, outputs, inputs, output, arch, compile_us: {
                "name": name,
                "key": key,
                "code": b"\xc3",
                "compile_us": compile_us,
            }
        )
        bench_worker_module.run_google_benchmark = fake_google_benchmark
        return bench_worker_module.run(
            {
                "query": "INPUT t (x i32); SELECT x AS y;",
                "provider": provider,
                "rows": 8,
                "warmups": 0,
                "runs": 1,
                "null_density": 0.0,
                "output": "pyarrow",
                "arch": "native",
            }
        )
    finally:
        bench_worker_module.querylang.parse = originals["parse"]
        bench_worker_module.querylang.lower = originals["lower"]
        bench_worker_module.make_inputs = originals["make_inputs"]
        bench_worker_module.run_python_api_kernel = originals["run_python_api_kernel"]
        bench_worker_module.measure_python_prepare_us = originals[
            "measure_python_prepare_us"
        ]
        bench_worker_module.sj_ext.benchmark_hir_jit_compile = originals[
            "benchmark_hir_jit_compile"
        ]
        bench_worker_module.sj.inspect = originals["inspect"]
        bench_worker_module.capture_asmjit_machine_code = originals[
            "capture_asmjit_machine_code"
        ]
        bench_worker_module.run_google_benchmark = originals["run_google_benchmark"]


def test_benchmark_inputs_fit_small_integer_dtypes():
    inputs = make_inputs({"x": sj.I8}, {}, rows=64, null_density=0.0)
    values = inputs["x"].buf

    assert values.dtype.name == "int8"
    assert values.min() >= 1
    assert values.max() <= 127


def test_public_limit_validation_rejects_direct_worker_payloads():
    caps = PublicLimits(
        max_rows=10, max_runs=3, max_warmups=2, allowed_providers=("python",)
    )
    normalized = validate_benchmark_payload(
        {
            "query": "INPUT t (x i32); SELECT x y;",
            "rows": "10",
            "warmups": "2",
            "runs": "3",
            "null_density": "0",
            "provider": "python",
            "output": "pyarrow",
        },
        caps,
    )
    assert normalized["rows"] == 10
    assert normalized["provider"] == "python"
    try:
        validate_benchmark_payload({**normalized, "rows": 11}, caps)
    except LimitError as exc:
        assert "rows exceeds public limit" in str(exc)
    else:
        raise AssertionError("expected row cap failure")
    try:
        validate_benchmark_payload({**normalized, "provider": "google"}, caps)
    except LimitError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("expected provider cap failure")


def test_platform_choices_hide_native_and_keep_cross_arch_benchmarks_compile_only():
    assert is_benchmark_arch_runnable("native")
    choices = platform_choices()
    assert [choice.value for choice in choices] == ["x86", "x86-ymm", "arm"]
    assert all("Native" not in choice.label for choice in choices)
    target_id = service_module.default_benchmark_target()
    native_arch = service_module.default_platform_arch(target_id)
    available = benchmark_availability("expression_sql", native_arch, target_id)
    assert available.runnable
    assert available.native_arch in {"x86", "arm"}
    serialized = benchmark_availability("serialized_hir", native_arch, target_id)
    assert not serialized.runnable
    assert "Expression SQL" in serialized.reason
    matrix = benchmark_availability_matrix()
    assert matrix["expression_sql"][target_id][native_arch]["runnable"] is True
    assert matrix["serialized_hir"][target_id][native_arch]["reason"] == serialized.reason
    blocked = next(
        (choice for choice in choices if not choice.runnable), None
    )
    if blocked is None:
        return
    blocked_availability = benchmark_availability("expression_sql", blocked.value, target_id)
    assert not blocked_availability.runnable
    assert matrix["expression_sql"][target_id][blocked.value]["reason"] == blocked_availability.reason

    benchmark = run_benchmark(
        DEFAULT_QUERY,
        BenchmarkOptions(rows=1, warmups=0, runs=1, null_density=0, arch=blocked.value),
    )
    assert not benchmark.ok
    assert "cannot execute" in benchmark.error


def test_local_target_config_and_rejects_remote_targets():
    previous = os.environ.get(targets_module.CONFIG_ENV)
    with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
        json.dump(
            {
                "local": {
                    "label": "Local configured",
                    "env": {
                        "LLVM_CONFIG": "local-llvm-config",
                        "LLVM_CLANG": "local-clang",
                    },
                },
                "targets": [],
            },
            tmp,
        )
        tmp.flush()
        os.environ[targets_module.CONFIG_ENV] = tmp.name
        targets_module.clear_target_caches()
        loaded = targets_module.benchmark_targets()
        assert [target.id for target in loaded] == ["local"]
        assert loaded[0].label == "Local configured"
        assert loaded[0].env == (
            ("LLVM_CLANG", "local-clang"),
            ("LLVM_CONFIG", "local-llvm-config"),
        )
        assert targets_module.machine_to_arch_family("x86_64") == "x86"
        assert targets_module.machine_to_arch_family("arm64") == "arm"
        assert (
            targets_module.arch_family_for_code_arch("native", loaded[0])
            == targets_module.host_arch_family()
        )
    if previous is None:
        os.environ.pop(targets_module.CONFIG_ENV, None)
    else:
        os.environ[targets_module.CONFIG_ENV] = previous
    targets_module.clear_target_caches()

    with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
        json.dump(
            {
                "targets": [
                    {
                        "id": "other-host",
                        "label": "Unsupported target",
                        "arch_family": "x86",
                    }
                ]
            },
            tmp,
        )
        tmp.flush()
        os.environ[targets_module.CONFIG_ENV] = tmp.name
        targets_module.clear_target_caches()
        try:
            targets_module.benchmark_targets()
        except targets_module.TargetError as exc:
            assert "only support the local host" in str(exc)
        else:
            raise AssertionError("expected remote target config to be rejected")
    if previous is None:
        os.environ.pop(targets_module.CONFIG_ENV, None)
    else:
        os.environ[targets_module.CONFIG_ENV] = previous
    targets_module.clear_target_caches()


def test_config_snapshot_reloads_targets_and_limits_without_manual_clear():
    previous = os.environ.get(targets_module.CONFIG_ENV)
    targets_module.clear_target_caches()
    try:
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as first, tempfile.NamedTemporaryFile("w+", suffix=".json") as second:
            json.dump(
                {
                    "local": {"label": "Config one"},
                    "security": {"max_rows": 11},
                    "targets": [],
                },
                first,
            )
            first.flush()
            json.dump(
                {
                    "local": {"label": "Config two"},
                    "security": {"max_rows": 22},
                    "targets": [],
                },
                second,
            )
            second.flush()

            os.environ[targets_module.CONFIG_ENV] = first.name
            assert targets_module.benchmark_targets()[0].label == "Config one"
            assert public_limits().max_rows == 11

            os.environ[targets_module.CONFIG_ENV] = second.name
            assert targets_module.benchmark_targets()[0].label == "Config two"
            assert public_limits().max_rows == 22

            second_path = Path(second.name)
            second_path.write_text(
                json.dumps(
                    {
                        "local": {"label": "Config two reloaded"},
                        "security": {"max_rows": 33},
                        "targets": [],
                    }
                ),
                encoding="utf-8",
            )
            assert targets_module.benchmark_targets()[0].label == "Config two reloaded"
            assert public_limits().max_rows == 33

            targets_module.clear_target_caches()
            assert targets_module.benchmark_targets()[0].label == "Config two reloaded"
    finally:
        if previous is None:
            os.environ.pop(targets_module.CONFIG_ENV, None)
        else:
            os.environ[targets_module.CONFIG_ENV] = previous
        targets_module.clear_target_caches()


def test_local_benchmark_worker_uses_repo_pythonpath():
    captured = {}
    previous_run = targets_module.subprocess.run
    previous_pythonpath = os.environ.get("PYTHONPATH")
    try:
        os.environ["PYTHONPATH"] = "existing-pythonpath"

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured.update(kwargs)
            return SimpleNamespace(
                returncode=0,
                stdout=json.dumps(_benchmark_result_payload()),
                stderr="",
            )

        targets_module.subprocess.run = fake_run
        target = BenchmarkTarget(
            id="local",
            label="Local",
            arch_family=targets_module.host_arch_family(),
            env=((targets_module.PYTHON_BUILD_DIR_ENV, "/tmp/simjit-python-build"),),
        )
        result = targets_module.LocalBenchmarkExecutor(target).run(
            {"query": DEFAULT_QUERY},
            timeout=7,
        )

        assert result.ok
        assert captured["cmd"] == [
            os.sys.executable,
            "-m",
            "explorer.app.bench_worker",
        ]
        assert captured["cwd"] == targets_module.REPO_ROOT
        assert captured["timeout"] == 7
        path_parts = captured["env"]["PYTHONPATH"].split(os.pathsep)
        assert path_parts[0] == str(targets_module.REPO_ROOT)
        assert path_parts[1] == "/tmp/simjit-python-build"
        assert str(targets_module.REPO_ROOT / "python") in path_parts
        assert str(targets_module.REPO_ROOT / "python" / "src") in path_parts
        assert path_parts[-1] == "existing-pythonpath"
    finally:
        targets_module.subprocess.run = previous_run
        if previous_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = previous_pythonpath


def test_target_choices_include_probe_status():
    target = BenchmarkTarget(
        id="fake",
        label="Fake x86",
        arch_family="x86",
    )
    ok_choice = targets_module._target_display_label(
        target,
        TargetProbe(
            ok=True,
            arch_family="x86",
            machine="x86_64",
            system="Linux",
            python="3.12.3",
            gxx="/usr/bin/g++",
        ),
    )
    assert ok_choice == "Fake x86 - x86"
    missing_choice = targets_module._target_display_label(
        target,
        TargetProbe(ok=True, arch_family="x86", machine="x86_64", system="Linux"),
    )
    assert missing_choice == "Fake x86 - x86; missing g++"
    offline_choice = targets_module._target_display_label(
        target, TargetProbe(ok=False, arch_family="unknown", error="probe failed")
    )
    assert offline_choice == "Fake x86 - offline"


def test_run_benchmark_rejects_unknown_target():
    previous = os.environ.get(targets_module.CONFIG_ENV)
    os.environ.pop(targets_module.CONFIG_ENV, None)
    targets_module.clear_target_caches()
    try:
        benchmark = run_benchmark(
            DEFAULT_QUERY,
            BenchmarkOptions(
                rows=1,
                warmups=0,
                runs=1,
                null_density=0,
                arch="x86",
                benchmark_target="x86-linux",
                provider="google",
            ),
        )
        assert not benchmark.ok
        assert "unknown benchmark target" in benchmark.error
    finally:
        if previous is None:
            os.environ.pop(targets_module.CONFIG_ENV, None)
        else:
            os.environ[targets_module.CONFIG_ENV] = previous
        targets_module.clear_target_caches()


def test_routes_render_success_and_errors_when_fastapi_is_available():
    if os.environ.get("SIMJIT_EXPLORER_TEST_ROUTES") != "1":
        pytest.skip("route tests are disabled")
    if importlib.util.find_spec("fastapi") is None:
        pytest.skip("fastapi is not installed")

    from fastapi.testclient import TestClient

    import explorer.app as app_module

    client = TestClient(app_module.app)
    cross_origin = client.post(
        "/compile",
        data={"query": "INPUT t (x i32); SELECT x y;"},
        headers={"Origin": "http://evil.example"},
    )
    assert cross_origin.status_code == 403
    assert "cross-origin POST rejected" in cross_origin.text

    ok = client.post("/compile", data={"query": "INPUT t (x i32); SELECT x + 1 AS y;"})
    assert ok.status_code == 200

    bad = client.post("/compile", data={"query": "INPUT t (x nope); SELECT x AS y;"})
    assert bad.status_code == 200
    assert "unsupported type" in bad.text

    fragment = client.post(
        "/compile",
        data={
            "query": "INPUT t (x i32); SELECT x + 1 AS y;",
            "input_mode": "expression_sql",
        },
        headers={"X-Simjit-Explorer-Fragment": "results"},
    )
    assert fragment.status_code == 200
    assert "<!doctype html>" not in fragment.text.lower()

    old_compile = app_module.compile_query
    try:
        app_module.compile_query = (
            lambda query, arch="native", input_mode="expression_sql": (
                service_module.CompileResult(
                    True,
                    query,
                    {"C++": "<script>alert(1)</script> &"},
                    [],
                    arch=arch,
                    input_mode=input_mode,
                )
            )
        )
        escaped = client.post(
            "/compile",
            data={"query": "INPUT t (x i32); SELECT x y;"},
            headers={"X-Simjit-Explorer-Fragment": "results"},
        )
        assert escaped.status_code == 200
        assert "<script>alert(1)</script>" not in escaped.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt; &amp;" in escaped.text
    finally:
        app_module.compile_query = old_compile

    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["Cache-Control"] == "no-store"
    assert client.get("/demo").status_code == 200

    q6 = client.get("/?example=tpch_q6_filter&provider=python&rows=1234&runs=7")
    assert q6.status_code == 200
    assert "q6_revenue" in q6.text
    assert "example=tpch_q6_filter" in q6.text
    assert "provider=python" in q6.text
    assert "rows=1234" in q6.text
    assert "runs=7" in q6.text

    shared_query = "INPUT t (x i32); SELECT x + 42 AS shared_result;"
    encoded_query = base64.urlsafe_b64encode(shared_query.encode()).decode().rstrip("=")
    shared = client.get(
        "/", params={"query": encoded_query, "input_mode": "expression_sql"}
    )
    assert shared.status_code == 200
    assert "shared_result" in shared.text

    bad_shared = client.get("/", params={"query": "%%%not-base64%%%"})
    assert bad_shared.status_code == 200
    assert "invalid base64 query parameter" in bad_shared.text

    previous_config_for_limits = os.environ.get(targets_module.CONFIG_ENV)
    try:
        with tempfile.NamedTemporaryFile("w+", suffix=".json") as tmp:
            json.dump(
                {"security": {"max_query_bytes": 8, "max_form_bytes": 200000}}, tmp
            )
            tmp.flush()
            os.environ[targets_module.CONFIG_ENV] = tmp.name
            too_large_compile = client.post(
                "/compile", data={"query": "INPUT t (x i32); SELECT x y;"}
            )
            assert too_large_compile.status_code == 200
            assert "query is too large" in too_large_compile.text
            encoded_large = (
                base64.urlsafe_b64encode(b"too large query").decode().rstrip("=")
            )
            too_large_shared = client.get("/", params={"query": encoded_large})
            assert too_large_shared.status_code == 200
            assert "shared query is too large" in too_large_shared.text
    finally:
        if previous_config_for_limits is None:
            os.environ.pop(targets_module.CONFIG_ENV, None)
        else:
            os.environ[targets_module.CONFIG_ENV] = previous_config_for_limits

    fallback = client.get("/?example=missing")
    assert fallback.status_code == 200
    assert "total_x" in fallback.text
    assert "q6_revenue" not in fallback.text

    redirect = client.get("/examples/tpch_q6_filter", follow_redirects=False)
    assert redirect.status_code == 303
    assert redirect.headers["location"] == "../?example=tpch_q6_filter"

    bad_redirect = client.get("/examples/nope", follow_redirects=False)
    assert bad_redirect.status_code == 303
    assert bad_redirect.headers["location"] == "../"

    mounted = TestClient(app_module.app, root_path="/explorer")
    mounted_page = mounted.get("/")
    assert mounted_page.status_code == 200
    assert mounted.get("/demo").status_code == 200
    assert "/explorer/static/system.css" in mounted_page.text
    assert "/explorer/static/style.css" in mounted_page.text
    assert 'formaction="compile"' in mounted_page.text
    assert 'href="?example=simple_arithmetic' in mounted_page.text

    blocked = next(
        (choice for choice in platform_choices() if not choice.runnable), None
    )
    if blocked is not None:
        benchmark = client.post(
            "/benchmark",
            data={
                "query": "INPUT t (x i32); SELECT x + 1 AS y;",
                "rows": "1",
                "warmups": "0",
                "runs": "1",
                "null_density": "0",
                "arch": blocked.value,
                "benchmark_target": "local",
                "provider": "python",
            },
        )
        assert benchmark.status_code == 200
        assert "cannot execute" in benchmark.text

    invalid_benchmark = client.post(
        "/benchmark",
        data={
            "query": "INPUT t (x i32); SELECT x + 1 AS y;",
            "rows": "nope",
            "warmups": "0",
            "runs": "1",
            "null_density": "0",
            "provider": "python",
        },
    )
    assert invalid_benchmark.status_code == 200
    assert "rows must be an integer" in invalid_benchmark.text

    captured_providers = []
    captured_arches = []
    old_run_benchmark = app_module.run_benchmark
    try:
        def fake_run_benchmark(query, options):
            captured_providers.append(options.provider)
            captured_arches.append(options.arch)
            return BenchmarkRunResult(
                rows=options.rows,
                warmups=options.warmups,
                runs=options.runs,
                null_density=options.null_density,
                output=options.output,
                arch=options.arch,
                provider=options.provider,
                provider_label="All providers",
                compile_us=1.0,
                median_us=1.0,
                compile_ms=0.001,
                median_ms=0.001,
                rows_per_second=None,
                throughput_gbps=None,
                cache_hits=0,
                cache_misses=0,
                scalar_outputs=(),
                implementations=(
                    BenchmarkImplementationRow.from_mapping(
                        {
                        "name": "Simjit SIMD / Python API",
                        "key": "asmjit_vectorized_py",
                        "api": "Python",
                        "compile_mode": "SIMD",
                        "backend": "AsmJit",
                        "opt": "-",
                        "chart_label": "Python · SIMD · AsmJit",
                        "source": "Python API",
                        "hot_us": 1.0,
                        "speedup": None,
                        "rows_per_second": None,
                        "throughput_gbps": None,
                        "compile_us": 1.0,
                        "bar_percent": 100.0,
                        "main": True,
                        "bar_label": "1.00 M rows/s",
                        }
                    ),
                    BenchmarkImplementationRow.from_mapping(
                        {
                        "name": "Simjit SIMD / AsmJit",
                        "key": "asmjit_vectorized",
                        "api": "C++",
                        "compile_mode": "SIMD",
                        "backend": "AsmJit",
                        "opt": "-",
                        "chart_label": "C++ · SIMD · AsmJit",
                        "source": "Google Benchmark",
                        "hot_us": 1.0,
                        "speedup": None,
                        "rows_per_second": None,
                        "throughput_gbps": None,
                        "compile_us": 1.0,
                        "bar_percent": 100.0,
                        "main": True,
                        "bar_label": "1.00 M rows/s",
                        }
                    ),
                ),
                diagnostics=(),
            )

        app_module.run_benchmark = fake_run_benchmark
        all_provider = client.post(
            "/benchmark",
            data={
                "query": "INPUT t (x i32); SELECT x + 1 AS y;",
                "rows": "1",
                "warmups": "0",
                "runs": "1",
                "null_density": "0",
                "provider": "all",
            },
            headers={"X-Simjit-Explorer-Fragment": "results"},
        )
        assert all_provider.status_code == 200
        assert captured_providers == ["all"]
        assert captured_arches == ["native"]
    finally:
        app_module.run_benchmark = old_run_benchmark

    previous_config = os.environ.get(targets_module.CONFIG_ENV)
    try:
        os.environ.pop(targets_module.CONFIG_ENV, None)
        targets_module.clear_target_caches()
        dead_page = client.get("/?benchmark_target=dead-runner")
        assert dead_page.status_code == 200
        forged = client.post(
            "/benchmark",
            data={
                "query": "INPUT t (x i32); SELECT x + 1 AS y;",
                "rows": "1",
                "warmups": "0",
                "runs": "1",
                "null_density": "0",
                "arch": "x86",
                "benchmark_target": "dead-runner",
                "provider": "python",
            },
        )
        assert forged.status_code == 200
        assert "unknown benchmark target" in forged.text
    finally:
        if previous_config is None:
            os.environ.pop(targets_module.CONFIG_ENV, None)
        else:
            os.environ[targets_module.CONFIG_ENV] = previous_config
        targets_module.clear_target_caches()
