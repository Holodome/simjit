#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_corpus


class TestScalarFallbackCollection:
    def test_implementation_names_include_explicit_scalar(self) -> None:
        assert benchmark_corpus.implementation_names(
            ["asmjit", "asmjit_s", "llvm", "llvm_s", "cpp", "cpp_s"],
            True,
        ) == [
            "asmjit",
            "asmjit-scalar",
            "llvm-simd-o1",
            "llvm-simd-o3",
            "llvm-o1",
            "llvm-o3",
            "cpp-o1",
            "cpp-o3",
            "cpp-scalar-o1",
            "cpp-scalar-o3",
        ]

    def test_fallback_artifacts_are_labeled_as_scalar(self) -> None:
        assert benchmark_corpus.implementation_names(
            ["asmjit", "llvm", "cpp"],
            True,
            vectorized=False,
        ) == [
            "asmjit",
            "llvm-o1",
            "llvm-o3",
            "cpp-scalar-o1",
            "cpp-scalar-o3",
        ]

    def test_fallback_bundles_map_to_scalar_implementations(self) -> None:
        assert benchmark_corpus.bundle_implementations(
            ["asmjit", "llvm", "cpp"],
            True,
            vectorized=False,
        ) == {
            "asmjit": "asmjit",
            "llvm": "llvm-o1",
            "llvm_o3": "llvm-o3",
            "cpp": "cpp-scalar-o1",
            "cpp_o3": "cpp-scalar-o3",
        }

    def test_native_benchmark_names_are_parsed(self) -> None:
        raw = {
            "benchmarks": [
                {
                    "name": "7/general:7/arm-vector/0/llvm_o3",
                    "iterations": 10,
                    "cpu_time": 2.0,
                    "real_time": 3.0,
                    "time_unit": "us",
                }
            ]
        }
        assert benchmark_corpus.benchmark_rows(
            raw, {"llvm_o3": "llvm-simd-o3"}, rows=100
        )[0] == {
            "name": "7/general:7/arm-vector/0/llvm_o3",
            "function_name": "llvm_o3",
            "implementation": "llvm-simd-o3",
            "iterations": 10,
            "cpu_time_ns": 2000.0,
            "real_time_ns": 3000.0,
            "loop_throughput_rows_per_ns": 0.05,
        }

    def test_vector_case_requires_paired_asmjit_artifacts(self) -> None:
        item = {
            "variant": "x86-vector",
            "codes": [{"name": "asmjit"}, {"name": "llvm"}],
        }
        with pytest.raises(ValueError, match="asmjit_s"):
            benchmark_corpus.selected_code_names(
                item,
                ["asmjit", "asmjit_s", "llvm"],
            )

    def test_vector_case_selects_scalar_pair(self) -> None:
        item = {
            "variant": "x86-vector",
            "codes": [
                {"name": "asmjit"},
                {"name": "asmjit_s"},
                {"name": "llvm"},
            ],
        }
        assert benchmark_corpus.selected_code_names(
            item,
            ["asmjit", "asmjit_s", "llvm", "cpp"],
        ) == ["asmjit", "asmjit_s", "llvm"]

    def test_vector_case_selects_all_scalar_pairs(self) -> None:
        names = ["asmjit", "asmjit_s", "llvm", "llvm_s", "cpp", "cpp_s"]
        item = {
            "variant": "x86-vector",
            "codes": [{"name": name} for name in names],
        }
        assert benchmark_corpus.selected_code_names(item, names) == names

    def test_vector_case_requires_scalar_llvm_when_requested(self) -> None:
        item = {
            "variant": "x86-vector",
            "codes": [{"name": "llvm"}],
        }
        with pytest.raises(ValueError, match="llvm_s"):
            benchmark_corpus.selected_code_names(
                item,
                ["llvm", "llvm_s"],
            )

    def test_automatic_fallback_does_not_require_duplicate_scalar(self) -> None:
        item = {
            "variant": "x86-scalar",
            "codes": [{"name": "asmjit"}],
        }
        assert benchmark_corpus.selected_code_names(
            item,
            ["asmjit", "asmjit_s"],
        ) == ["asmjit"]
