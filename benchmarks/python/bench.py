# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import argparse
import dataclasses
import gc
import importlib
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import simjit as sj
from simjit import _simjit as simjit_ext


DEFAULT_SIZES = (1_024, 4_096, 65_536, 1_048_576, 16_777_216)
DEFAULT_MIN_TIME = 0.20
DEFAULT_REPEATS = 5


@dataclass(frozen=True)
class OptionalModule:
    name: str
    module: Any | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.module is not None


def import_optional(name: str) -> OptionalModule:
    try:
        return OptionalModule(name, importlib.import_module(name))
    except ImportError as exc:
        return OptionalModule(name, None, str(exc))


numba_mod = import_optional("numba")
numba = numba_mod.module


@dataclass(frozen=True)
class DataSet:
    rows: int
    numpy: dict[str, np.ndarray]


@dataclass(frozen=True)
class Workload:
    name: str
    output: str
    bytes_per_row: int
    simjit_program: Any
    expected: Callable[[DataSet], Any]
    numba_factory: Callable[[], Callable[[DataSet], Any]] | None = None


@dataclass(frozen=True)
class BenchCase:
    library: str
    workload: Workload
    fn: Callable[[DataSet], Any]
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class Timing:
    seconds: float
    iterations: int


def make_external_data(rows: int) -> DataSet:
    idx = np.arange(rows, dtype=np.int64)
    arrays = {
        "price": np.int64(10_000) + (idx % 1000) * np.int64(110),
        "discount": ((idx % 10) * 100).astype(np.int32),
        "tax": ((idx % 7) * 100).astype(np.int32),
        "cost": (idx % 700) * np.int64(9),
        "penalty": (idx % 13) * np.int64(25),
        "lower": np.full(rows, np.int64(-25_000), dtype=np.int64),
        "upper": np.full(rows, np.int64(2_500_000), dtype=np.int64),
        "target": (idx % 500) * np.int64(125),
        "x": (idx % 97) * np.int64(4),
        "y": (idx % 89) * np.int64(2),
        "z": (idx % 17) + np.int64(8),
        "x0": np.full(rows, np.int64(48), dtype=np.int64),
        "y0": np.full(rows, np.int64(32), dtype=np.int64),
        "open": (idx % 251) * np.int64(110),
        "close": ((idx * 7) % 251) * np.int64(105),
        "limit": (idx % 23) * np.int64(100) + np.int64(200),
        "quantity": (idx % 50).astype(np.int32),
        "shipdate": (19940101 + (idx % 730)).astype(np.int32),
        "discount_bp": ((idx * 17) % 2000).astype(np.int32),
        "tax_bp": ((idx * 11) % 1000).astype(np.int32),
        "delay": ((idx * 5) % 60).astype(np.int32),
    }
    return DataSet(rows=rows, numpy=arrays)


EXTERNAL_INPUT_COLUMNS = (
    "price", "cost", "penalty", "lower", "upper", "target", "x", "y", "z",
    "open", "close", "limit", "x0", "y0", "discount", "tax", "quantity",
    "shipdate", "discount_bp", "tax_bp", "delay",
)


def external_input_checksum(data: DataSet) -> int:
    value = 1469598103934665603
    mask = (1 << 64) - 1
    for name in EXTERNAL_INPUT_COLUMNS:
        for item in np.ascontiguousarray(data.numpy[name]):
            value = ((value ^ (int(item) & mask)) * 1099511628211) & mask
    return value


def make_external_workloads() -> list[Workload]:
    """Arithmetic-heavy workloads shared with the native external comparison."""
    q = lambda name: sj.col(name, sj.I64)
    i = lambda name: sj.col(name, sj.I32)
    price, cost = map(q, ("price", "cost"))
    discount_i32, tax_i32 = i("discount"), i("tax")
    discount, tax = sj.i64(discount_i32), sj.i64(tax_i32)
    quantity, shipdate = i("quantity"), i("shipdate")
    c = lambda value: sj.const(int(value), sj.I64)
    scale = c(10_000)
    revenue = price * (scale - discount) / scale
    charge = revenue * (scale + tax) / scale
    margin = charge - cost

    def numpy_pricing(d: DataSet):
        disc = d.numpy["price"] * (10_000 - d.numpy["discount"].astype(np.int64)) // 10_000
        return disc, disc * (10_000 + d.numpy["tax"].astype(np.int64)) // 10_000

    def numpy_bounded(d: DataSet):
        adjusted = (d.numpy["price"] * d.numpy["quantity"].astype(np.int64)
                    * (10_000 - d.numpy["discount"].astype(np.int64)) // 10_000 - d.numpy["penalty"])
        return (np.minimum(np.maximum(adjusted, d.numpy["lower"]), d.numpy["upper"]),
                np.abs(adjusted - d.numpy["target"]))

    def numpy_dag(d: DataSet):
        x, y, z = d.numpy["x"], d.numpy["y"], d.numpy["z"]
        xy = x * y
        base = (x + y) * z
        adjusted = base - xy
        return base + adjusted, np.minimum(base, adjusted), np.maximum(base, adjusted)

    def numpy_conditional(d: DataSet):
        disc = d.numpy["price"] * (10_000 - d.numpy["discount"].astype(np.int64)) // 10_000
        chg = disc * (10_000 + d.numpy["tax"].astype(np.int64)) // 10_000
        score = chg - d.numpy["cost"]
        adjusted = np.where(d.numpy["quantity"] >= 25,
                            score * 110 // 100, score * 90 // 100).astype(np.int64)
        return adjusted, np.maximum(score, 0), np.minimum(chg, d.numpy["price"]), np.abs(score)

    def numpy_range(d: DataSet):
        lo = np.minimum(d.numpy["open"], d.numpy["close"])
        hi = np.maximum(d.numpy["open"], d.numpy["close"])
        body = np.abs(d.numpy["close"] - d.numpy["open"])
        return lo, hi, hi - lo, np.minimum(body, d.numpy["limit"])

    def selection_mask(d: DataSet, charge=None):
        if charge is None:
            rev = d.numpy["price"] * (10_000 - d.numpy["discount"].astype(np.int64)) // 10_000
            charge = rev * (10_000 + d.numpy["tax"].astype(np.int64)) // 10_000
        return ((d.numpy["shipdate"] >= 19940101) & (d.numpy["shipdate"] <= 19940501)
                & (d.numpy["discount"] >= 500) & (d.numpy["discount"] <= 700)
                & (d.numpy["quantity"] < 24) & ((charge - d.numpy["cost"]) > 5_000))

    def numpy_selection(d: DataSet):
        return np.flatnonzero(selection_mask(d)).astype(np.int64)

    def numpy_q6(d: DataSet):
        rev = d.numpy["price"] * (10_000 - d.numpy["discount"].astype(np.int64)) // 10_000
        charge = rev * (10_000 + d.numpy["tax"].astype(np.int64)) // 10_000
        return np.sum(rev, where=selection_mask(d, charge), dtype=np.int64)

    def numpy_summary(d: DataSet):
        disc = d.numpy["price"] * (10_000 - d.numpy["discount"].astype(np.int64)) // 10_000
        chg = disc * (10_000 + d.numpy["tax"].astype(np.int64)) // 10_000
        return (np.sum(d.numpy["quantity"], dtype=np.int64), np.sum(d.numpy["price"], dtype=np.int64),
                np.sum(disc, dtype=np.int64), np.sum(chg, dtype=np.int64),
                np.min(chg), np.max(chg))

    def line_charge(d: DataSet):
        gross = d.numpy["quantity"].astype(np.int64) * d.numpy["price"]
        net = gross * (10_000 - d.numpy["discount_bp"].astype(np.int64)) // 10_000
        return net * (10_000 + d.numpy["tax_bp"].astype(np.int64)) // 10_000

    def numpy_segmented(d: DataSet):
        chg = line_charge(d)
        return (np.sum(chg, where=d.numpy["delay"] > 30, dtype=np.int64),
                np.sum(chg, where=d.numpy["quantity"] >= 25, dtype=np.int64),
                np.sum(chg, where=chg > 1_000_000, dtype=np.int64), np.min(chg), np.max(chg))

    def numpy_moments(d: DataSet):
        cx, cy = d.numpy["x"] - d.numpy["x0"], d.numpy["y"] - d.numpy["y0"]
        xy = cx * cy
        return (np.sum(cx, dtype=np.int64), np.sum(cy, dtype=np.int64),
                np.sum(cx * cx, dtype=np.int64), np.sum(xy, dtype=np.int64),
                np.sum(cy * cy, dtype=np.int64), np.min(xy), np.max(xy))

    penalty, lower, upper, target = map(q, ("penalty", "lower", "upper", "target"))
    adjusted = price * sj.i64(quantity) * (scale - discount) / scale - penalty
    x, y, z = map(q, ("x", "y", "z"))
    xy, base = x * y, (x + y) * z
    dag_adjusted = base - xy
    open_, close, limit = map(q, ("open", "close", "limit"))
    lo, hi = sj.min(open_, close), sj.max(open_, close)
    predicate = ((shipdate >= 19940101) & (shipdate <= 19940501) &
                 (discount_i32 >= 500) & (discount_i32 <= 700) &
                 (quantity < 24) & (margin > c(5_000)))
    discount_bp, tax_bp, delay = i("discount_bp"), i("tax_bp"), i("delay")
    gross = sj.i64(quantity) * price
    net = gross * (scale - sj.i64(discount_bp)) / scale
    line = net * (scale + sj.i64(tax_bp)) / scale
    x0, y0 = q("x0"), q("y0")
    cx, cy = x - x0, y - y0
    cross = cx * cy

    specs = [
        ("pricing_projection", "tuple_array_i64", 36, sj.query(disc_price=revenue, charge=charge), numpy_pricing),
        ("bounded_order_score", "tuple_array_i64", 68,
         sj.query(clamped=sj.min(sj.max(adjusted, lower), upper), deviation=sj.abs(adjusted - target)), numpy_bounded),
        ("shared_arithmetic_dag", "tuple_array_i64", 48,
         sj.query(combined=base + dag_adjusted, least=sj.min(base, dag_adjusted), greatest=sj.max(base, dag_adjusted)), numpy_dag),
        ("conditional_pricing", "tuple_array_i64", 60,
         sj.query(adjusted=(quantity >= 25).ifelse(margin * c(110) / c(100), margin * c(90) / c(100)),
                  positive=sj.max(margin, c(0)), capped=sj.min(charge, price), magnitude=sj.abs(margin)), numpy_conditional),
        ("range_projection", "tuple_array_i64", 56,
         sj.query(lower=lo, upper=hi, span=hi - lo, bounded_body=sj.min(sj.abs(close - open_), limit)), numpy_range),
        ("revenue_selection", "array_i64", 32,
         sj.query(indices=sj.pack(sj.index(sj.I64), predicate, dst_size="selection_count")), numpy_selection),
        ("tpch_q6_revenue", "scalar_i64", 32, sj.query(revenue=revenue.sum(predicate)), numpy_q6),
        ("pricing_summary", "tuple_scalar_i64", 20,
         sj.query(sum_quantity=sj.i64(quantity).sum(), sum_price=price.sum(), sum_disc_price=revenue.sum(),
                  sum_charge=charge.sum(), min_charge=charge.min(), max_charge=charge.max()), numpy_summary),
        ("segmented_line_totals", "tuple_scalar_i64", 24,
         sj.query(late=line.sum(delay > 30), large=line.sum(quantity >= 25),
                  expensive=line.sum(line > c(1_000_000)), minimum=line.min(), maximum=line.max()), numpy_segmented),
        ("statistical_moments", "tuple_scalar_i64", 32,
         sj.query(sum_x=cx.sum(), sum_y=cy.sum(), sum_xx=(cx * cx).sum(), sum_xy=cross.sum(),
                  sum_yy=(cy * cy).sum(), min_xy=cross.min(), max_xy=cross.max()), numpy_moments),
    ]
    return [Workload(name=name, output=output, bytes_per_row=bpr, simjit_program=program,
                     expected=expected, numba_factory=make_numba_external(name))
            for name, output, bpr, program, expected in specs]


def make_numba_external(name: str) -> Callable[[], Callable[[DataSet], Any]]:
    def factory() -> Callable[[DataSet], Any]:
        if name == "pricing_projection":
            @numba.njit
            def kernel(price, discount, tax, disc, charge):
                for j in range(price.size):
                    disc[j] = price[j] * (10_000 - np.int64(discount[j])) // 10_000
                    charge[j] = disc[j] * (10_000 + np.int64(tax[j])) // 10_000
            def run(d):
                a = np.empty(d.rows, np.int64); b = np.empty(d.rows, np.int64)
                kernel(d.numpy["price"], d.numpy["discount"], d.numpy["tax"], a, b); return a, b
            return run
        if name == "bounded_order_score":
            @numba.njit
            def kernel(price, qty, discount, penalty, lower, upper, target, out, dev):
                for j in range(price.size):
                    v = price[j] * np.int64(qty[j]) * (10_000 - np.int64(discount[j])) // 10_000 - penalty[j]
                    out[j] = min(max(v, lower[j]), upper[j]); dev[j] = abs(v - target[j])
            def run(d):
                a = np.empty(d.rows, np.int64); b = np.empty(d.rows, np.int64)
                kernel(*(d.numpy[k] for k in ("price", "quantity", "discount", "penalty", "lower", "upper", "target")), a, b); return a, b
            return run
        if name == "shared_arithmetic_dag":
            @numba.njit
            def kernel(x, y, z, combined, least, greatest):
                for j in range(x.size):
                    xy = x[j] * y[j]; base = (x[j] + y[j]) * z[j]; adjusted = base - xy
                    combined[j] = base + adjusted; least[j] = min(base, adjusted); greatest[j] = max(base, adjusted)
            def run(d):
                out = tuple(np.empty(d.rows, np.int64) for _ in range(3))
                kernel(d.numpy["x"], d.numpy["y"], d.numpy["z"], *out); return out
            return run
        if name == "conditional_pricing":
            @numba.njit
            def kernel(price, discount, tax, cost, qty, adjusted, positive, capped, magnitude):
                for j in range(price.size):
                    revenue = price[j] * (10_000 - np.int64(discount[j])) // 10_000
                    charge = revenue * (10_000 + np.int64(tax[j])) // 10_000; score = charge - cost[j]
                    adjusted[j] = score * (110 if qty[j] >= 25 else 90) // 100
                    positive[j] = max(score, 0); capped[j] = min(charge, price[j]); magnitude[j] = abs(score)
            def run(d):
                out = tuple(np.empty(d.rows, np.int64) for _ in range(4))
                kernel(*(d.numpy[k] for k in ("price", "discount", "tax", "cost", "quantity")), *out); return out
            return run
        if name == "range_projection":
            @numba.njit
            def kernel(open_, close, limit, lower, upper, span, bounded):
                for j in range(open_.size):
                    lo = min(open_[j], close[j]); hi = max(open_[j], close[j])
                    lower[j] = lo; upper[j] = hi; span[j] = hi - lo; bounded[j] = min(abs(close[j] - open_[j]), limit[j])
            def run(d):
                out = tuple(np.empty(d.rows, np.int64) for _ in range(4))
                kernel(d.numpy["open"], d.numpy["close"], d.numpy["limit"], *out); return out
            return run
        if name == "revenue_selection":
            @numba.njit
            def kernel(price, discount, tax, cost, qty, shipdate):
                out = np.empty(price.size, np.int64); count = 0
                for j in range(price.size):
                    revenue = price[j] * (10_000 - np.int64(discount[j])) // 10_000
                    margin = revenue * (10_000 + np.int64(tax[j])) // 10_000 - cost[j]
                    if (shipdate[j] >= 19940101 and shipdate[j] <= 19940501 and discount[j] >= 500
                            and discount[j] <= 700 and qty[j] < 24 and margin > 5_000):
                        out[count] = j; count += 1
                return out[:count]
            return lambda d: kernel(*(d.numpy[k] for k in ("price", "discount", "tax", "cost", "quantity", "shipdate")))
        if name == "tpch_q6_revenue":
            @numba.njit
            def kernel(price, discount, tax, cost, qty, shipdate):
                total = np.int64(0)
                for j in range(price.size):
                    revenue = price[j] * (10_000 - np.int64(discount[j])) // 10_000
                    margin = revenue * (10_000 + np.int64(tax[j])) // 10_000 - cost[j]
                    if (shipdate[j] >= 19940101 and shipdate[j] <= 19940501 and discount[j] >= 500
                            and discount[j] <= 700 and qty[j] < 24 and margin > 5_000):
                        total += revenue
                return total
            return lambda d: kernel(*(d.numpy[k] for k in ("price", "discount", "tax", "cost", "quantity", "shipdate")))
        if name == "pricing_summary":
            @numba.njit
            def kernel(price, discount, tax, qty):
                sq = np.int64(0); sp = np.int64(0); sd = np.int64(0); sc = np.int64(0)
                mn = np.int64(9_223_372_036_854_775_807); mx = np.int64(-9_223_372_036_854_775_807)
                for j in range(price.size):
                    disc = price[j] * (10_000 - np.int64(discount[j])) // 10_000
                    charge = disc * (10_000 + np.int64(tax[j])) // 10_000
                    sq += qty[j]; sp += price[j]; sd += disc; sc += charge; mn = min(mn, charge); mx = max(mx, charge)
                return sq, sp, sd, sc, mn, mx
            return lambda d: kernel(*(d.numpy[k] for k in ("price", "discount", "tax", "quantity")))
        if name == "segmented_line_totals":
            @numba.njit
            def kernel(price, qty, discount_bp, tax_bp, delay):
                late = np.int64(0); large = np.int64(0); expensive = np.int64(0)
                mn = np.int64(9_223_372_036_854_775_807); mx = np.int64(-9_223_372_036_854_775_807)
                for j in range(price.size):
                    net = np.int64(qty[j]) * price[j] * (10_000 - np.int64(discount_bp[j])) // 10_000
                    charge = net * (10_000 + np.int64(tax_bp[j])) // 10_000
                    if delay[j] > 30: late += charge
                    if qty[j] >= 25: large += charge
                    if charge > 1_000_000: expensive += charge
                    mn = min(mn, charge); mx = max(mx, charge)
                return late, large, expensive, mn, mx
            return lambda d: kernel(*(d.numpy[k] for k in ("price", "quantity", "discount_bp", "tax_bp", "delay")))
        if name == "statistical_moments":
            @numba.njit
            def kernel(x, y, x0, y0):
                sx = np.int64(0); sy = np.int64(0); sxx = np.int64(0); sxy = np.int64(0); syy = np.int64(0)
                mn = np.int64(9_223_372_036_854_775_807); mx = np.int64(-9_223_372_036_854_775_807)
                for j in range(x.size):
                    cx = x[j] - x0[j]; cy = y[j] - y0[j]; xy = cx * cy
                    sx += cx; sy += cy; sxx += cx * cx; sxy += xy; syy += cy * cy; mn = min(mn, xy); mx = max(mx, xy)
                return sx, sy, sxx, sxy, syy, mn, mx
            return lambda d: kernel(*(d.numpy[k] for k in ("x", "y", "x0", "y0")))
        raise ValueError(f"unknown external workload: {name}")
    return factory


def make_simjit_fn(workload: Workload, *, policy: str) -> Callable[[DataSet], Any]:
    cache: dict[int, Callable[[], Any]] = {}
    simjit_stats: dict[str, Any] = {}

    def prepare(data: DataSet) -> Callable[[], Any]:
        inputs = data.numpy
        selection_count = None
        if workload.name == "revenue_selection":
            inputs = dict(inputs)
            selection_count = np.zeros(1, dtype=np.int64)
            inputs["selection_count"] = selection_count
        outputs = workload.simjit_program.to_dsl()
        session = simjit_ext.Session()
        session.policy = {
            "best_effort": simjit_ext.CompilePolicy.BestEffort,
            "vectorized": simjit_ext.CompilePolicy.Vectorized,
            "scalar": simjit_ext.CompilePolicy.Scalar,
        }[policy]
        prepared = session.prepare_program(outputs, inputs, "numpy")
        prepared.release_outputs()
        result_names = tuple(item[0] for item in outputs)
        stats = session.statistics()
        simjit_stats.update(
            {
                "function_count": stats.function_count,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "compilation_attempts": stats.compilation_attempts,
                "compilation_successes": stats.compilation_successes,
                "compilation_failures": stats.compilation_failures,
                "compile_policy": policy,
            }
        )

        def run_prepared():
            result = prepared.run_fresh_values()
            if len(result_names) == 1:
                value = result[0]
                if selection_count is not None:
                    return value[: int(selection_count[0])]
                return value
            return result

        run_prepared.cleanup = prepared.release_outputs

        return run_prepared

    def run(data: DataSet):
        key = id(data)
        if key not in cache:
            # Intentionally prepare on the first measured call so first_seconds
            # includes simjit compilation, matching numba's first-call semantics.
            cache[key] = prepare(data)
        return cache[key]()

    run.simjit_stats = simjit_stats
    run.cleanup = lambda: [getattr(fn, "cleanup", lambda: None)() for fn in cache.values()]
    return run


def build_cases(
    workloads: list[Workload], libraries: set[str], *, simjit_policy: str
) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for workload in workloads:
        if "numpy" in libraries:
            cases.append(BenchCase("numpy", workload, workload.expected))
        if "numba" in libraries:
            if numba is None:
                cases.append(
                    BenchCase("numba", workload, unavailable_fn, numba_mod.error)
                )
            elif workload.numba_factory is not None:
                cases.append(BenchCase("numba", workload, workload.numba_factory()))
        if "simjit_numpy" in libraries:
            cases.append(
                BenchCase(
                    "simjit_numpy",
                    workload,
                    make_simjit_fn(workload, policy=simjit_policy),
                )
            )
    return cases


def unavailable_fn(_data: DataSet):
    raise RuntimeError("benchmark dependency is unavailable")


def to_numpy(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return value.as_py()
    return value


def validate_result(workload: Workload, actual: Any, expected: Any) -> None:
    if isinstance(expected, tuple):
        if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
            raise AssertionError(
                f"{workload.name}: expected {len(expected)} outputs, got {type(actual).__name__}"
            )
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            validate_result(
                dataclasses.replace(workload, name=f"{workload.name}[{index}]"),
                actual_item,
                expected_item,
            )
        return
    actual = to_numpy(actual)
    expected = to_numpy(expected)
    expected_dtype = np.asarray(expected).dtype
    float_rtol = 1e-5 if expected_dtype == np.dtype(np.float32) else 1e-12
    float_atol = 1e-5 if expected_dtype == np.dtype(np.float32) else 1e-12
    if isinstance(expected, np.ndarray):
        actual_arr = np.asarray(actual)
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(
                actual_arr, expected, rtol=float_rtol, atol=float_atol
            )
        else:
            np.testing.assert_array_equal(actual_arr, expected)
        return
    if isinstance(expected, np.generic):
        expected = expected.item()
    if isinstance(actual, np.generic):
        actual = actual.item()
    if isinstance(expected, float):
        if not math.isclose(
            float(actual),
            float(expected),
            rel_tol=float_rtol,
            abs_tol=float_atol,
        ):
            raise AssertionError(
                f"{workload.name}: expected {expected!r}, got {actual!r}"
            )
    elif int(actual) != int(expected):
        raise AssertionError(f"{workload.name}: expected {expected!r}, got {actual!r}")


def consume(value: Any) -> None:
    value = to_numpy(value)
    if isinstance(value, np.ndarray):
        if value.size:
            scalar = value.reshape(-1)[0]
            if isinstance(scalar, np.generic):
                scalar.item()
        return
    if isinstance(value, np.generic):
        value.item()


def time_once(
    fn: Callable[[], Any], iterations: int, cleanup: Callable[[], None]
) -> Timing:
    elapsed = 0
    for _ in range(iterations):
        start = time.perf_counter_ns()
        value = fn()
        elapsed += time.perf_counter_ns() - start
        consume(value)
        del value
        cleanup()
    return Timing(seconds=elapsed / 1e9 / iterations, iterations=iterations)


def choose_iterations(
    fn: Callable[[], Any], min_time: float, cleanup: Callable[[], None]
) -> int:
    iterations = 1
    while True:
        timing = time_once(fn, iterations, cleanup)
        total = timing.seconds * iterations
        if total >= min_time or iterations >= 1 << 30:
            return iterations
        if total <= 0:
            iterations *= 10
            continue
        scale = max(2, int(math.ceil(min_time / total)))
        iterations *= min(scale, 10)


def measure(
    fn: Callable[[], Any],
    *,
    repeats: int,
    min_time: float,
    validate: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    cleanup = getattr(fn, "cleanup", lambda: None)
    first_start = time.perf_counter_ns()
    first_value = fn()
    first_seconds = (time.perf_counter_ns() - first_start) / 1e9
    consume(first_value)
    if validate is not None:
        validate(first_value)
    del first_value
    cleanup()

    # One explicit warm run after first-run timing avoids timing one-off lazy setup.
    warm_value = fn()
    consume(warm_value)
    del warm_value
    cleanup()
    iterations = choose_iterations(fn, min_time, cleanup)

    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            samples.append(time_once(fn, iterations, cleanup).seconds)
    finally:
        if gc_was_enabled:
            gc.enable()

    return {
        "first_seconds": first_seconds,
        "hot_min_seconds": min(samples),
        "hot_median_seconds": statistics.median(samples),
        "hot_samples_seconds": samples,
        "iterations_per_repeat": iterations,
    }


def rows_per_second(rows: int, seconds: float) -> float:
    return rows / seconds if seconds > 0 else float("inf")


def bytes_per_second(rows: int, bytes_per_row: int, seconds: float) -> float:
    return rows * bytes_per_row / seconds if seconds > 0 else float("inf")


def environment() -> dict[str, Any]:
    def version(mod: OptionalModule) -> str | None:
        if mod.module is None:
            return None
        return getattr(mod.module, "__version__", "unknown")

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "numba": version(numba_mod),
        "simjit_extension": True,
    }


def write_markdown(path: Path, records: list[dict[str, Any]]) -> None:
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("status") == "ok":
            by_workload.setdefault(record["workload"], []).append(record)

    lines = [
        "# Python Benchmark Results",
        "",
        "Lower hot median seconds is better.",
        "",
    ]
    for workload in sorted(by_workload):
        lines.append(f"## {workload}")
        lines.append("")
        lines.append("| rows | library | first ms | hot median us | rows/s | GB/s |")
        lines.append("| ---: | --- | ---: | ---: | ---: | ---: |")
        for record in sorted(
            by_workload[workload], key=lambda r: (r["rows"], r["hot_median_seconds"])
        ):
            lines.append(
                "| {rows} | {library} | {first_ms:.3f} | {hot_us:.3f} | {rps:.3g} | {gbps:.3g} |".format(
                    rows=record["rows"],
                    library=record["library"],
                    first_ms=record["first_seconds"] * 1e3,
                    hot_us=record["hot_median_seconds"] * 1e6,
                    rps=record["hot_rows_per_second"],
                    gbps=record["hot_bytes_per_second"] / 1e9,
                )
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(
        int(part.replace("_", "")) for part in value.split(",") if part.strip()
    )


def parse_libraries(value: str) -> set[str]:
    aliases = {
        "simjit": {"simjit_numpy"},
        "all": {"numpy", "numba", "simjit_numpy"},
    }
    result: set[str] = set()
    for part in value.split(","):
        name = part.strip()
        if not name:
            continue
        result.update(aliases.get(name, {name}))
    allowed = {"numpy", "numba", "simjit_numpy"}
    unknown = result - allowed
    if unknown:
        raise ValueError(f"unknown libraries: {', '.join(sorted(unknown))}")
    return result


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark simjit Python API against Python compute libraries."
    )
    parser.add_argument("--sizes", default=",".join(str(v) for v in DEFAULT_SIZES))
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--min-time", type=float, default=DEFAULT_MIN_TIME)
    parser.add_argument("--libraries", default="numpy,numba,simjit_numpy")
    parser.add_argument(
        "--workloads",
        default="",
        help="Comma-separated E1-E10 workload names. Defaults to all ten.",
    )
    parser.add_argument(
        "--simjit-policy",
        choices=("best_effort", "vectorized", "scalar"),
        default="best_effort",
    )
    parser.add_argument("--jsonl", default="bench-results/python-bench-external.jsonl")
    parser.add_argument("--markdown", default="bench-results/python-bench-external.md")
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args(argv)

    sizes = parse_sizes(args.sizes)
    libraries = parse_libraries(args.libraries)
    workloads = make_external_workloads()

    if args.workloads:
        requested = [name.strip() for name in args.workloads.split(",") if name.strip()]
        available = {workload.name: workload for workload in workloads}
        unknown = [name for name in requested if name not in available]
        if unknown:
            raise ValueError(f"unknown workloads: {', '.join(unknown)}")
        workloads = [available[name] for name in requested]

    env = environment()

    jsonl_path = Path(args.jsonl)
    markdown_path = Path(args.markdown)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as out:
        for rows in sizes:
            data = make_external_data(rows)
            data_checksum = external_input_checksum(data)
            cases = build_cases(workloads, libraries, simjit_policy=args.simjit_policy)
            expected_by_workload = {w.name: w.expected(data) for w in workloads}
            for case in cases:
                base = {
                    "status": "ok",
                    "workload": case.workload.name,
                    "library": case.library,
                    "rows": rows,
                    "output": case.workload.output,
                    "bytes_per_row": case.workload.bytes_per_row,
                    "execution_timing_boundary": "prepared-inputs-no-outputs-to-result-ready",
                    "input_checksum": data_checksum,
                    "simjit_policy": args.simjit_policy if case.library.startswith("simjit_") else None,
                    "environment": env,
                }
                if case.unavailable_reason is not None:
                    record = {
                        **base,
                        "status": "skipped",
                        "reason": case.unavailable_reason,
                    }
                    out.write(json.dumps(record, sort_keys=True) + "\n")
                    out.flush()
                    records.append(record)
                    print(
                        f"skip {rows:>9} {case.workload.name:<18} {case.library}: {case.unavailable_reason}"
                    )
                    continue

                fn = lambda case=case, data=data: case.fn(data)
                fn.cleanup = getattr(case.fn, "cleanup", lambda: None)
                try:
                    validator = None
                    if not args.no_validate:
                        expected = expected_by_workload[case.workload.name]
                        validator = (
                            lambda value, workload=case.workload, expected=expected: (
                                validate_result(workload, value, expected)
                            )
                        )
                    timing = measure(
                        fn,
                        repeats=args.repeats,
                        min_time=args.min_time,
                        validate=validator,
                    )
                    hot = timing["hot_median_seconds"]
                    record = {
                        **base,
                        **timing,
                        "hot_rows_per_second": rows_per_second(rows, hot),
                        "hot_bytes_per_second": bytes_per_second(
                            rows, case.workload.bytes_per_row, hot
                        ),
                    }
                    simjit_stats = getattr(case.fn, "simjit_stats", None)
                    if simjit_stats is not None:
                        record["simjit_stats"] = dict(simjit_stats)
                    print(
                        f"ok   {rows:>9} {case.workload.name:<18} {case.library:<14} "
                        f"first={record['first_seconds'] * 1e3:8.3f} ms "
                        f"hot={record['hot_median_seconds'] * 1e6:9.3f} us"
                    )
                except Exception as exc:
                    record = {**base, "status": "failed", "reason": repr(exc)}
                    print(
                        f"fail {rows:>9} {case.workload.name:<18} {case.library}: {exc!r}"
                    )
                    raise exc
                out.write(json.dumps(record, sort_keys=True) + "\n")
                out.flush()
                records.append(record)

    write_markdown(markdown_path, records)
    print(f"wrote {jsonl_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
