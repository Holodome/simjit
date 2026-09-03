#!/usr/bin/env bash
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


set -oex pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

make distclean 
make 

make distclean 
make release

make distclean 
make reldebug

make distclean 
make test

make distclean 
make NO_ASMJIT=1

make distclean 
make NO_LLVM=1

make distclean 
make NO_CPP=1 

make distclean 
make NO_LLVM=1 NO_CPP=1

make distclean 
make NO_ASMJIT=1 NO_CPP=1

make distclean 
make NO_LLVM=1 NO_CPP=1 NO_X86=1

make distclean 
make NO_LLVM=1 NO_CPP=1 NO_ARM=1

echo All good!
