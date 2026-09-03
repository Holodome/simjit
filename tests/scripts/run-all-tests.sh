#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


set -u -o pipefail

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
json_out="${SIMJIT_TEST_JSON:-$root_dir/test-dump/tests.jsonl}"

declare -a step_names=()
declare -a step_statuses=()
declare -a step_details=()

failed=0

record_step() {
    local name=$1
    local status=$2
    local details=$3
    step_names+=("$name")
    step_statuses+=("$status")
    step_details+=("$details")
    if [[ "$status" == "FAIL" ]]; then
        failed=1
    fi
}

run_step() {
    local name=$1
    shift
    local rc

    echo "== $name =="
    "$@"
    rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "PASS: $name"
        echo
        record_step "$name" "PASS" ""
        return 0
    fi

    echo "FAIL($rc): $name"
    echo
    record_step "$name" "FAIL" "exit=$rc"
    return "$rc"
}

skip_step() {
    local name=$1
    local reason=$2
    echo "== $name =="
    echo "SKIP: $reason"
    echo
    record_step "$name" "SKIP" "$reason"
}

print_summary() {
    local pass_count=0
    local fail_count=0
    local skip_count=0

    echo "Summary:"
    for i in "${!step_names[@]}"; do
        local name=${step_names[$i]}
        local status=${step_statuses[$i]}
        local detail=${step_details[$i]}
        case "$status" in
            PASS) pass_count=$((pass_count + 1)) ;;
            FAIL) fail_count=$((fail_count + 1)) ;;
            SKIP) skip_count=$((skip_count + 1)) ;;
        esac
        if [[ -n "$detail" ]]; then
            printf '  %-5s %s (%s)\n' "$status" "$name" "$detail"
        else
            printf '  %-5s %s\n' "$status" "$name"
        fi
    done

    echo "  total: $((pass_count + fail_count + skip_count))"
    echo "  passed: $pass_count"
    echo "  failed: $fail_count"
    echo "  skipped: $skip_count"
}

cd "$root_dir" || exit 1
mkdir -p "$(dirname "$json_out")"
rm -f "$json_out"

build_test_ok=0
build_local_runner_ok=0
build_cli_ok=0
build_python_ok=0

if run_step "Build test binaries" make test; then
    build_test_ok=1
fi

if run_step "Build native local runner" make local-runner; then
    build_local_runner_ok=1
fi

if [[ $build_test_ok -eq 1 && -x build/debug/integration_test ]]; then
    run_step "Run integration tests" build/debug/integration_test || true
else
    skip_step "Run integration tests" "integration test binary was not built"
fi

if [[ $build_test_ok -eq 1 && -x build/debug/test ]]; then
    run_step "Run main test runner" build/debug/test --suite=all --arch=all --mode=all || true
else
    skip_step "Run main test runner" "test binary was not built"
fi

if [[ $build_test_ok -eq 1 && -x build/debug/test ]]; then
    run_step "Generate tests JSONL" build/debug/test --suite=all --arch=native --mode=all --dump-json "$json_out" || true
else
    skip_step "Generate tests JSONL" "test binary was not built"
fi

if [[ -f "$json_out" ]]; then
    run_step "Validate serialization grammar" tests/scripts/validate-tests-json-serialization.sh "$json_out" || true
else
    skip_step "Validate serialization grammar" "tests JSON dump was not produced"
fi

if [[ -f "$json_out" && $build_local_runner_ok -eq 1 ]]; then
    run_step "Run native bundle tests (all suites)" build/debug/local_runner --file "$json_out" || true
else
    skip_step "Run native bundle tests (all suites)" "native local runner or full JSON dump was not produced"
fi

if run_step "Build simjit-cli" make debug; then
    build_cli_ok=1
fi

if run_step "Build Python extension" make py; then
    build_python_ok=1
fi

run_step "Run benchmark runner unit tests" scripts/py -m pytest -q tests/scripts/test_benchmark_corpus.py || true

if [[ $build_cli_ok -eq 1 && $build_python_ok -eq 1 ]]; then
    run_step "Run Python debug path smoke" scripts/py python/tests/debug_path_smoke.py || true
else
    skip_step "Run Python debug path smoke" "simjit-cli or python extension was not built"
fi

if [[ $build_python_ok -eq 1 ]]; then
    run_step "Run Python unit tests" make py-test || true
    run_step "Run Python e2e tests" make py-e2e || true
else
    skip_step "Run Python unit tests" "python extension was not built"
    skip_step "Run Python e2e tests" "python extension was not built"
fi

if [[ $build_cli_ok -eq 1 && -x build/debug/simjit-cli ]]; then
    run_step "Run CLI script tests" tests/scripts/run-cli-tests.sh || true
else
    skip_step "Run CLI script tests" "simjit-cli was not built"
fi

print_summary

if [[ $failed -ne 0 ]]; then
    exit 1
fi

exit 0
