# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

import pytest

from numpy_testing import NumpyFunctionCase


@pytest.fixture
def numpy_function():
    """Factory for a Simjit function executed and checked against NumPy."""
    return NumpyFunctionCase


@pytest.fixture
def pa():
    """Provide PyArrow or skip only the tests that require it."""
    return pytest.importorskip("pyarrow")
