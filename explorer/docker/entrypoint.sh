#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

set -euo pipefail

uvicorn_pid=""
nginx_pid=""

terminate() {
  if [[ -n "${uvicorn_pid}" ]]; then
    kill -TERM "${uvicorn_pid}" 2>/dev/null || true
  fi
  if [[ -n "${nginx_pid}" ]]; then
    kill -TERM "${nginx_pid}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
}

trap terminate INT TERM

python -m uvicorn explorer.app:app \
  --host "${SIMJIT_EXPLORER_HOST:-127.0.0.1}" \
  --port "${SIMJIT_EXPLORER_PORT:-8001}" \
  --root-path "${SIMJIT_EXPLORER_ROOT_PATH:-/explorer}" &
uvicorn_pid="$!"

nginx -g "daemon off;" &
nginx_pid="$!"

wait -n "${uvicorn_pid}" "${nginx_pid}"
status="$?"
terminate
exit "${status}"
