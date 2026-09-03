#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


set -eo pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root_dir"

read -r -a simjit_cmd <<< "${SIMJIT:-build/debug/simjit-cli}"
extra_flags=()
if [ -n "${LLVM:-}" ]; then
    extra_flags+=(--llvm)
fi
if [ -n "${ASMJIT:-}" ]; then
    extra_flags+=(--asmjit)
fi

passed=0
failed=0

function run_case() {
    local name=$1
    local arch=$2
    local serialized=$3
    shift 3

    echo "== Running ${name} =="
    if command -v bat > /dev/null; then
        if "${simjit_cmd[@]}" --arch "$arch" -s "$serialized" "$@" "${extra_flags[@]}" | bat -l cpp -p --paging=never; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    else
        if "${simjit_cmd[@]}" --arch "$arch" -s "$serialized" "$@" "${extra_flags[@]}"; then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    fi
}

function run_json_case() {
    local name=$1
    local arch=$2
    local serialized=$3
    shift 3

    echo "== Running ${name} =="
    local json_path
    local expr_path
    json_path=$(mktemp "${TMPDIR:-/tmp}/simjit-cli-json.XXXXXX")
    expr_path=$(mktemp "${TMPDIR:-/tmp}/simjit-cli-expr.XXXXXX")
    printf '%s' "$serialized" > "$expr_path"
    if "${simjit_cmd[@]}" --arch "$arch" --serialized-file "$expr_path" --dump-json "$json_path" "$@"; then
        if python3 - "$json_path" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    lines = [line for line in f.read().splitlines() if line.strip()]

assert len(lines) == 1
item = json.loads(lines[0])
assert item["id"] == "cli:0"
assert item["suite"] == "cli"
assert item["variant"] == "x86-vector"
assert item["schema"]["args"] == [
    {"dtype": "i32", "kind": "in"},
    {"dtype": "i32", "kind": "out"},
]
assert item["serialized"].startswith("(func ")
names = {code["name"] for code in item["codes"]}
assert "asmjit" in names
assert "asmjit_asm" in names
PY
        then
            passed=$((passed + 1))
        else
            failed=$((failed + 1))
        fi
    else
        failed=$((failed + 1))
    fi
    rm -f "$json_path"
    rm -f "$expr_path"
}

function gen_pair() {
    local serialized=$1
    local name=$2

    run_case "tpch/${name}_vect" x86 "$serialized" --name "${name}_vect"
    run_case "tpch/${name}_scalar" x86 "$serialized" --name "${name}_scalar" --scalar
}

add_i32='(func (args (arg 0 i32 src-arr) (arg 1 i32 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 const i32 "0x1") (step 2 binary i32 add (step 0) (step 1)) (step 3 store i32 (step 2) (arg 1) unaligned)) (roots (step 3)))'
predicate_i1='(func (args (arg 0 i32 src-arr) (arg 1 i1 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 const i32 "0x0") (step 2 cmp i1 gt (step 0) (step 1)) (step 3 store i1 (step 2) (arg 1) unaligned)) (roots (step 3)))'
masked_store='(func (args (arg 0 i32 src-arr) (arg 1 i32 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 const i32 "0x0") (step 2 cmp i1 ne (step 0) (step 1)) (step 3 binary i32 add (step 0) (step 0)) (step 4 store i32 (step 3) (arg 1) unaligned (step 2))) (roots (step 4)))'
sum_i32='(func (args (arg 0 i32 src-arr) (arg 1 i32 dst-scalar)) (accs (acc 0 i32 (arg 1) (step 1))) (steps (step 0 load i32 (arg 0) unaligned) (step 1 acc-arith-bin i32 add (step 0) (acc 0))) (roots (step 1)))'

wide_add='(func (args (arg 0 i32 src-arr) (arg 1 i32 src-arr) (arg 2 i32 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 load i32 (arg 1) unaligned) (step 2 binary i32 add (step 0) (step 1)) (step 3 store i32 (step 2) (arg 2) unaligned)) (roots (step 3)))'
widen_mul='(func (args (arg 0 i32 src-arr) (arg 1 i64 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 int-cast i64 sext (step 0)) (step 2 const i64 "0x7b") (step 3 binary i64 mul (step 1) (step 2)) (step 4 store i64 (step 3) (arg 1) unaligned)) (roots (step 4)))'
select_mask='(func (args (arg 0 i32 src-arr) (arg 1 i8 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 const i32 "0x64") (step 2 cmp i1 gt (step 0) (step 1)) (step 3 const i8 "0x1") (step 4 const i8 "0x0") (step 5 select i8 (step 2) (step 3) (step 4)) (step 6 store i8 (step 5) (arg 1) unaligned)) (roots (step 6)))'

db_masked_store='(func (args (arg 0 i32 src-arr) (arg 1 i32 src-arr) (arg 2 i1 src-arr) (arg 3 i32 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 load i32 (arg 1) unaligned) (step 2 load i1 (arg 2) unaligned) (step 3 binary i32 add (step 0) (step 1)) (step 4 store i32 (step 3) (arg 3) unaligned (step 2))) (roots (step 4)))'
db_mask_result='(func (args (arg 0 i32 src-arr) (arg 1 i32 src-arr) (arg 2 i1 dst-arr)) (steps (step 0 load i32 (arg 0) unaligned) (step 1 load i32 (arg 1) unaligned) (step 2 cmp i1 eq (step 0) (step 1)) (step 3 store i1 (step 2) (arg 2) unaligned)) (roots (step 3)))'

sum_i64='(func (args (arg 0 i64 src-arr) (arg 1 i64 dst-scalar)) (accs (acc 0 i64 (arg 1) (step 1))) (steps (step 0 load i64 (arg 0) unaligned) (step 1 acc-arith-bin i64 add (step 0) (acc 0))) (roots (step 1)))'
sum_product_i64='(func (args (arg 0 i64 src-arr) (arg 1 i64 src-arr) (arg 2 i64 dst-scalar)) (accs (acc 0 i64 (arg 2) (step 3))) (steps (step 0 load i64 (arg 0) unaligned) (step 1 load i64 (arg 1) unaligned) (step 2 binary i64 mul (step 0) (step 1)) (step 3 acc-arith-bin i64 add (step 2) (acc 0))) (roots (step 3)))'

run_case "samples/add_i32" avx512 "$add_i32"
run_case "samples/add_i32_scalar_hir" avx512 "$add_i32" --scalar --print-hir
run_case "samples/predicate_i1_mir" avx512 "$predicate_i1" --print-mir
run_case "samples/masked_store_asmjit" avx512 "$masked_store" --asmjit
run_case "samples/sum_i32_serialized" avx512 "$sum_i32" --serialized
run_json_case "samples/add_i32_json" avx512 "$add_i32"

run_case "vect/wide_add" avx512 "$wide_add"
run_case "vect/widen_mul" avx512 "$widen_mul"
run_case "vect/select_mask_mir" avx512 "$select_mask" --print-mir

run_case "db/masked_store" avx512 "$db_masked_store"
run_case "db/mask_result_hir" avx512 "$db_mask_result" --print-hir

gen_pair "$sum_i32" "sum_i32"
gen_pair "$sum_i64" "sum_i64"
gen_pair "$sum_product_i64" "sum_product_i64"

total=$((passed + failed))

echo
echo "CLI test summary: ${passed} passed, ${failed} failed, ${total} total"

if [ "$failed" -eq 0 ]; then
    echo "All CLI tests passed."
    exit 0
fi

exit 1
