#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

gcovr -e simjit-cli -e thirdparty -e tests -e src/simjit/detail/expected.h -e src/simjit/detail/span.h --exclude-lines-by-pattern '.*assert.*' --exclude-lines-by-pattern '.*SIMJIT_ASSERT.*' --exclude-lines-by-pattern '.*SIMJIT_UNREACHABLE.*' --exclude-lines-by-pattern '.*INVALID_FLOAT_CASES.*' --print-summary --html-details -o report.html 
open report.html
