# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import sys
from pathlib import Path


def _fix_repo_python_path() -> None:
    repo_python_dir = Path(__file__).resolve().parent
    repo_python_dir_str = str(repo_python_dir)

    while repo_python_dir_str in sys.path:
        sys.path.remove(repo_python_dir_str)
    sys.path.append(repo_python_dir_str)


_fix_repo_python_path()
