#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


# Validate serialized HIR snippets stored in a tests bundle dump.
#
# This script expects one argument: a JSONL bundle file produced by the test
# dump path in tests/cpp/test.cpp.
#
# Behavior:
# - Tracks total/ok/failed counters and renders a progress bar on stderr.
# - Stops drawing the progress bar permanently if validation commands print any
#   output, so diagnostic text does not fight with terminal redraws.
# - Handles Ctrl-C for the whole script, not just the currently running Python
#   validator process.
# - Prints a final summary with total, ok, and failed counts.
#
# Exit codes:
# - 0: all serialized entries matched the grammar
# - 1: at least one entry failed validation, or usage/file checks failed
# - 130: interrupted by Ctrl-C

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: tests/scripts/validate-tests-json-serialization.sh <tests.jsonl>" >&2
    exit 1
fi

root_dir=$(cd "$(dirname "$0")/../.." && pwd)
json_file=$1

if [[ ! -f "$json_file" ]]; then
    echo "file not found: $json_file" >&2
    exit 1
fi

exec python3 "$root_dir/tests/scripts/validate-grammar.py" --json-file "$json_file"
