# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

import simjit as sj

ExpectedValue = object | Callable[[Mapping[str, np.ndarray]], object]


@dataclass(frozen=True)
class _Output:
    expression: object
    expected: ExpectedValue
    rtol: float | None
    atol: float


class NumpyFunctionCase:
    """Build, execute, and verify one Simjit function against NumPy."""

    def __init__(self, **inputs: object):
        self.inputs: dict[str, np.ndarray] = {
            name: np.asarray(value) for name, value in inputs.items()
        }
        self._outputs: dict[str, _Output] = {}

    def output(
        self,
        name: str,
        expression: object,
        expected: ExpectedValue,
        *,
        rtol: float | None = None,
        atol: float = 0,
    ) -> "NumpyFunctionCase":
        """Add an output and its expected value, returning this case for chaining.

        ``expected`` may be a value or a function accepting the normalized input
        mapping. Comparisons are exact unless ``rtol`` is supplied.
        """
        if name in self._outputs:
            raise ValueError(f"duplicate output: {name}")
        self._outputs[name] = _Output(expression, expected, rtol, atol)
        return self

    def build(self) -> sj.Program:
        if not self._outputs:
            raise ValueError("a NumPy function case must have at least one output")
        return sj.query(
            {name: output.expression for name, output in self._outputs.items()}
        )

    def run(self, *, session: sj.Session | None = None):
        if session is None:
            result = sj.run_program(self.build(), self.inputs)
        else:
            result = sj.run_program(self.build(), self.inputs, session=session)

        for name, output in self._outputs.items():
            expected = output.expected
            if callable(expected):
                expected = expected(self.inputs)
            actual = result[name]
            if output.rtol is None:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=output.rtol,
                    atol=output.atol,
                )

        return result
