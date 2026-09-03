# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

import simjit as sj
from simjit import _simjit as simjit_ext

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import pyarrow as pa
    import pyarrow.compute as pc
except ImportError:
    pa = None
    pc = None

try:
    import numba
except ImportError:
    numba = None

DEFAULT_SIZES = (1_024, 4_096, 65_536, 1_048_576)
DEFAULT_REPEATS = 5
DEFAULT_MIN_TIME = 0.20
DEFAULT_NULL_PATTERNS = ("periodic",)
DEFAULT_NULL_MODS = (7,)
DEFAULT_FULL_NULL_PATTERNS = ("periodic", "random", "runs")
DEFAULT_FULL_NULL_MODS = (4, 7, 15, 63)
DEFAULT_RANDOM_SEED = 0
DEFAULT_RUN_MEAN = 32
DEFAULT_WORKLOADS = ("nullable_revenue_multiagg",)
DEBUG_SIMJIT = False


@dataclass(frozen=True)
class DataSet:
    rows: int
    null_pattern: str
    null_mod: int
    seed: int
    run_mean: int
    numpy: dict[str, np.ndarray]
    arrow: dict[str, Any]


@dataclass(frozen=True)
class Timing:
    seconds: float
    iterations: int


def make_simjit_session() -> Any:
    session = simjit_ext.Session()
    if DEBUG_SIMJIT:
        session.debug_options.capture_on_success = True
        session.debug_options.stages = simjit_ext.DebugStage.All
        session.debug_options.record_vectorization_fail_exception = True
    return session


def print_simjit_debug(label: str, session: Any) -> None:
    if not DEBUG_SIMJIT:
        return

    print(f"=== simjit debug: {label}: bug report ===")
    print(session.bug_report())


def make_random_validity(
    rng: np.random.Generator, rows: int, null_probability: float
) -> np.ndarray:
    return rng.random(rows) >= null_probability


def make_run_validity(
    rng: np.random.Generator,
    rows: int,
    null_probability: float,
    run_mean: int,
) -> np.ndarray:
    values = np.empty(rows, dtype=np.bool_)
    pos = 0
    run_mean = max(run_mean, 1)
    while pos < rows:
        is_valid = rng.random() >= null_probability
        length = int(rng.geometric(1.0 / run_mean))
        end = min(rows, pos + max(length, 1))
        values[pos:end] = is_valid
        pos = end
    return values


def make_validity(
    idx: np.ndarray,
    *,
    pattern: str,
    null_mod: int,
    offset: int,
    rng: np.random.Generator,
    run_mean: int,
) -> np.ndarray:
    if null_mod <= 0:
        raise ValueError("null_mod must be positive")
    if pattern == "none":
        return np.ones(idx.size, dtype=np.bool_)

    null_probability = 1.0 / null_mod
    if pattern == "periodic":
        return (idx % null_mod) != 0
    if pattern == "random":
        return make_random_validity(rng, idx.size, null_probability)
    if pattern == "runs":
        return make_run_validity(rng, idx.size, null_probability, run_mean + offset)
    raise ValueError(f"unknown null pattern {pattern!r}")


def make_data(
    rows: int,
    *,
    null_mod: int = 7,
    null_pattern: str = "periodic",
    seed: int = DEFAULT_RANDOM_SEED,
    run_mean: int = DEFAULT_RUN_MEAN,
) -> DataSet:
    idx = np.arange(rows, dtype=np.int64)
    qty_raw = ((idx % 50) + 1).astype(np.int32)
    unit_price_raw = (((idx * 17) % 10_000) + 1).astype(np.int32)
    discount_raw = ((idx * 3) % 5_000).astype(np.int32)
    delay = ((idx * 11) % 60).astype(np.int32)

    rng = np.random.default_rng(seed + rows * 17 + null_mod * 1_003)
    qty_valid = make_validity(
        idx,
        pattern=null_pattern,
        null_mod=null_mod,
        offset=0,
        rng=rng,
        run_mean=run_mean,
    )
    unit_valid = make_validity(
        idx,
        pattern=null_pattern,
        null_mod=null_mod + 2,
        offset=2,
        rng=rng,
        run_mean=run_mean,
    )
    discount_valid = make_validity(
        idx,
        pattern=null_pattern,
        null_mod=null_mod + 4,
        offset=4,
        rng=rng,
        run_mean=run_mean,
    )

    qty = np.where(qty_valid, qty_raw, 0).astype(np.int32)
    unit_price = np.where(unit_valid, unit_price_raw, 0).astype(np.int32)
    discount = np.where(discount_valid, discount_raw, 0).astype(np.int32)

    arrays = {
        "qty_raw": qty_raw,
        "unit_price_raw": unit_price_raw,
        "discount_bp_raw": discount_raw,
        "qty": qty,
        "unit_price": unit_price,
        "discount_bp": discount,
        "delay": delay,
        "qty_valid": qty_valid,
        "unit_price_valid": unit_valid,
        "discount_valid": discount_valid,
    }

    arrow: dict[str, Any] = {}
    if pa is not None:
        arrow = {
            "qty_raw": pa.array(qty_raw, type=pa.int32()),
            "unit_price_raw": pa.array(unit_price_raw, type=pa.int32()),
            "discount_bp_raw": pa.array(discount_raw, type=pa.int32()),
            "qty": pa.array(
                [int(v) if ok else None for v, ok in zip(qty_raw, qty_valid)],
                type=pa.int32(),
            ),
            "unit_price": pa.array(
                [int(v) if ok else None for v, ok in zip(unit_price_raw, unit_valid)],
                type=pa.int32(),
            ),
            "discount_bp": pa.array(
                [int(v) if ok else None for v, ok in zip(discount_raw, discount_valid)],
                type=pa.int32(),
            ),
            "delay": pa.array(delay, type=pa.int32()),
        }
    return DataSet(
        rows=rows,
        null_pattern=null_pattern,
        null_mod=null_mod,
        seed=seed,
        run_mean=run_mean,
        numpy=arrays,
        arrow=arrow,
    )


def dataset_stats(data: DataSet) -> dict[str, float]:
    qty_null = float(np.mean(~data.numpy["qty_valid"]))
    unit_null = float(np.mean(~data.numpy["unit_price_valid"]))
    discount_null = float(np.mean(~data.numpy["discount_valid"]))
    unit_valid = data.numpy["unit_price_valid"] & (data.numpy["unit_price"] != 0)
    valid = unit_valid & data.numpy["qty_valid"] & (data.numpy["qty"] > 0)
    late = data.numpy["delay"] > 30
    high_qty = valid & (data.numpy["qty"] >= 25)
    expensive = valid & (data.numpy["unit_price"] >= 5_000)
    return {
        "qty_null_fraction": qty_null,
        "unit_price_null_fraction": unit_null,
        "discount_null_fraction": discount_null,
        "revenue_valid_fraction": float(np.mean(valid)),
        "late_revenue_valid_fraction": float(np.mean(valid & late)),
        "high_qty_valid_fraction": float(np.mean(high_qty)),
        "expensive_valid_fraction": float(np.mean(expensive)),
    }


def expected_nullable_revenue(data: DataSet) -> tuple[int, int, int]:
    q = np.where(data.numpy["qty_valid"], data.numpy["qty"], 0).astype(np.int64)
    unit_valid = data.numpy["unit_price_valid"] & (data.numpy["unit_price"] != 0)
    up = np.where(unit_valid, data.numpy["unit_price"], 0).astype(np.int64)
    d = np.where(data.numpy["discount_valid"], data.numpy["discount_bp"], 0).astype(
        np.int64
    )
    gross = q * up
    net = gross * (10_000 - d) // 10_000
    valid = unit_valid & (q > 0)
    late = data.numpy["delay"] > 30
    return (
        int(np.sum(np.where(valid, net, 0))),
        int(np.sum(np.where(valid & late, net, 0))),
        int(np.sum(valid)),
    )


def expected_split_valids_revenue(data: DataSet) -> tuple[int, int, int]:
    q = np.where(data.numpy["qty_valid"], data.numpy["qty"], 0).astype(np.int64)
    unit_valid = data.numpy["unit_price_valid"] & (data.numpy["unit_price"] != 0)
    up = np.where(unit_valid, data.numpy["unit_price"], 0).astype(np.int64)
    d = np.where(data.numpy["discount_valid"], data.numpy["discount_bp"], 0).astype(
        np.int64
    )
    gross = q * up
    net = gross * (10_000 - d) // 10_000
    base_valid = unit_valid & (q > 0)
    high_qty = base_valid & (q >= 25)
    expensive = base_valid & (up >= 5_000)
    late = base_valid & (data.numpy["delay"] > 30)
    return (
        int(np.sum(np.where(high_qty, net, 0))),
        int(np.sum(np.where(expensive, net, 0))),
        int(np.sum(late)),
    )


def expected_unconditional_div_split_valids(data: DataSet) -> tuple[int, int, int]:
    q_raw = data.numpy["qty_raw"].astype(np.int64)
    up_raw = data.numpy["unit_price_raw"].astype(np.int64)
    d_raw = data.numpy["discount_bp_raw"].astype(np.int64)
    net = q_raw * up_raw * (10_000 - d_raw) // 10_000

    q = np.where(data.numpy["qty_valid"], data.numpy["qty"], 0).astype(np.int64)
    unit_valid = data.numpy["unit_price_valid"] & (data.numpy["unit_price"] != 0)
    up = np.where(unit_valid, data.numpy["unit_price"], 0).astype(np.int64)
    base_valid = unit_valid & (q > 0)
    high_qty = base_valid & (q >= 25)
    expensive = base_valid & (up >= 5_000)
    late = base_valid & (data.numpy["delay"] > 30)
    return (
        int(np.sum(np.where(high_qty, net, 0))),
        int(np.sum(np.where(expensive, net, 0))),
        int(np.sum(late)),
    )


def expected_store_projection(
    data: DataSet,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q_raw = data.numpy["qty_raw"].astype(np.int64)
    up_raw = data.numpy["unit_price_raw"].astype(np.int64)
    d_raw = data.numpy["discount_bp_raw"].astype(np.int64)
    net = q_raw * up_raw * (10_000 - d_raw) // 10_000

    q = np.where(data.numpy["qty_valid"], data.numpy["qty"], 0).astype(np.int64)
    unit_valid = data.numpy["unit_price_valid"] & (data.numpy["unit_price"] != 0)
    up = np.where(unit_valid, data.numpy["unit_price"], 0).astype(np.int64)
    base_valid = unit_valid & (q > 0)
    high_qty = base_valid & (q >= 25)
    expensive = base_valid & (up >= 5_000)
    late = base_valid & (data.numpy["delay"] > 30)
    return (
        np.where(high_qty, net, 0).astype(np.int64),
        np.where(expensive, net, 0).astype(np.int64),
        late.astype(np.int64),
    )


def projection_sentinel(
    high_qty: np.ndarray, expensive: np.ndarray, late: np.ndarray
) -> tuple[int, int, int, int, int, int]:
    mid = high_qty.size // 2
    last = high_qty.size - 1
    return (
        int(high_qty[0]),
        int(expensive[mid]),
        int(late[last]),
        int(high_qty[mid]),
        int(expensive[last]),
        int(late[0]),
    )


def expected_store_projection_sentinel(
    data: DataSet,
) -> tuple[int, int, int, int, int, int]:
    return projection_sentinel(*expected_store_projection(data))


def assert_projection_outputs(
    outputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    expected: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    for got, want in zip(outputs, expected):
        if not np.array_equal(got, want):
            raise AssertionError("projection output mismatch")


def make_numpy_nullable_revenue(data: DataSet) -> Callable[[], tuple[int, int, int]]:
    def run() -> tuple[int, int, int]:
        return expected_nullable_revenue(data)

    return run


def make_numpy_split_valids_revenue(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    def run() -> tuple[int, int, int]:
        return expected_split_valids_revenue(data)

    return run


def make_numpy_unconditional_div_split_valids(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    def run() -> tuple[int, int, int]:
        return expected_unconditional_div_split_valids(data)

    return run


def make_numpy_store_projection(data: DataSet) -> Callable[[], tuple[int, ...]]:
    expected = expected_store_projection(data)
    sentinel = projection_sentinel(*expected)

    def run() -> tuple[int, ...]:
        return sentinel

    return run


def make_numba_nullable_revenue(data: DataSet) -> Callable[[], tuple[int, int, int]]:
    if numba is None:
        raise RuntimeError("numba is not available")

    @numba.njit
    def kernel(
        qty, qty_valid, unit_price, unit_valid_in, discount, discount_valid, delay
    ):
        total = 0
        late_total = 0
        count = 0
        for i in range(qty.size):
            q = qty[i] if qty_valid[i] else 0
            unit_valid = unit_valid_in[i] and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            d = discount[i] if discount_valid[i] else 0
            gross = np.int64(q) * np.int64(up)
            net = gross * (10_000 - np.int64(d)) // 10_000
            valid = unit_valid and q > 0
            if valid:
                total += net
                count += 1
                if delay[i] > 30:
                    late_total += net
        return total, late_total, count

    def run() -> tuple[int, int, int]:
        return kernel(
            data.numpy["qty"],
            data.numpy["qty_valid"],
            data.numpy["unit_price"],
            data.numpy["unit_price_valid"],
            data.numpy["discount_bp"],
            data.numpy["discount_valid"],
            data.numpy["delay"],
        )

    return run


def make_numba_split_valids_revenue(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if numba is None:
        raise RuntimeError("numba is not available")

    @numba.njit
    def kernel(
        qty, qty_valid, unit_price, unit_valid_in, discount, discount_valid, delay
    ):
        high_qty_total = 0
        expensive_total = 0
        late_count = 0
        for i in range(qty.size):
            q = qty[i] if qty_valid[i] else 0
            unit_valid = unit_valid_in[i] and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            d = discount[i] if discount_valid[i] else 0
            gross = np.int64(q) * np.int64(up)
            net = gross * (10_000 - np.int64(d)) // 10_000
            base_valid = unit_valid and q > 0
            if base_valid and q >= 25:
                high_qty_total += net
            if base_valid and up >= 5_000:
                expensive_total += net
            if base_valid and delay[i] > 30:
                late_count += 1
        return high_qty_total, expensive_total, late_count

    def run() -> tuple[int, int, int]:
        return kernel(
            data.numpy["qty"],
            data.numpy["qty_valid"],
            data.numpy["unit_price"],
            data.numpy["unit_price_valid"],
            data.numpy["discount_bp"],
            data.numpy["discount_valid"],
            data.numpy["delay"],
        )

    return run


def make_numba_unconditional_div_split_valids(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if numba is None:
        raise RuntimeError("numba is not available")

    @numba.njit
    def kernel(
        qty_raw,
        unit_price_raw,
        discount_raw,
        qty,
        qty_valid,
        unit_price,
        unit_valid_in,
        delay,
    ):
        high_qty_total = 0
        expensive_total = 0
        late_count = 0
        for i in range(qty.size):
            gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
            net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000

            q = qty[i] if qty_valid[i] else 0
            unit_valid = unit_valid_in[i] and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            base_valid = unit_valid and q > 0
            if base_valid and q >= 25:
                high_qty_total += net
            if base_valid and up >= 5_000:
                expensive_total += net
            if base_valid and delay[i] > 30:
                late_count += 1
        return high_qty_total, expensive_total, late_count

    def run() -> tuple[int, int, int]:
        return kernel(
            data.numpy["qty_raw"],
            data.numpy["unit_price_raw"],
            data.numpy["discount_bp_raw"],
            data.numpy["qty"],
            data.numpy["qty_valid"],
            data.numpy["unit_price"],
            data.numpy["unit_price_valid"],
            data.numpy["delay"],
        )

    return run


def make_numba_conditional_store_projection(
    data: DataSet,
) -> Callable[[], tuple[int, ...]]:
    if numba is None:
        raise RuntimeError("numba is not available")

    high_qty_out = np.zeros(data.rows, dtype=np.int64)
    expensive_out = np.zeros(data.rows, dtype=np.int64)
    late_out = np.zeros(data.rows, dtype=np.int64)
    expected = expected_store_projection(data)
    sentinel = projection_sentinel(*expected)

    @numba.njit
    def kernel(
        qty_raw,
        unit_price_raw,
        discount_raw,
        qty,
        qty_valid,
        unit_price,
        unit_valid_in,
        delay,
        high_qty_out,
        expensive_out,
        late_out,
    ):
        for i in range(qty.size):
            gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
            net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000

            q = qty[i] if qty_valid[i] else 0
            unit_valid = unit_valid_in[i] and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            base_valid = unit_valid and q > 0
            if base_valid and q >= 25:
                high_qty_out[i] = net
            if base_valid and up >= 5_000:
                expensive_out[i] = net
            if base_valid and delay[i] > 30:
                late_out[i] = 1

    def run_kernel() -> None:
        kernel(
            data.numpy["qty_raw"],
            data.numpy["unit_price_raw"],
            data.numpy["discount_bp_raw"],
            data.numpy["qty"],
            data.numpy["qty_valid"],
            data.numpy["unit_price"],
            data.numpy["unit_price_valid"],
            data.numpy["delay"],
            high_qty_out,
            expensive_out,
            late_out,
        )

    run_kernel()
    assert_projection_outputs((high_qty_out, expensive_out, late_out), expected)

    def run() -> tuple[int, ...]:
        run_kernel()
        return sentinel

    return run


def make_numba_select_store_projection(data: DataSet) -> Callable[[], tuple[int, ...]]:
    if numba is None:
        raise RuntimeError("numba is not available")

    high_qty_out = np.empty(data.rows, dtype=np.int64)
    expensive_out = np.empty(data.rows, dtype=np.int64)
    late_out = np.empty(data.rows, dtype=np.int64)
    expected = expected_store_projection(data)
    sentinel = projection_sentinel(*expected)

    @numba.njit
    def kernel(
        qty_raw,
        unit_price_raw,
        discount_raw,
        qty,
        qty_valid,
        unit_price,
        unit_valid_in,
        delay,
        high_qty_out,
        expensive_out,
        late_out,
    ):
        for i in range(qty.size):
            gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
            net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000

            q = qty[i] if qty_valid[i] else 0
            unit_valid = unit_valid_in[i] and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            base_valid = unit_valid and q > 0
            high_qty_out[i] = net if base_valid and q >= 25 else 0
            expensive_out[i] = net if base_valid and up >= 5_000 else 0
            late_out[i] = 1 if base_valid and delay[i] > 30 else 0

    def run_kernel() -> None:
        kernel(
            data.numpy["qty_raw"],
            data.numpy["unit_price_raw"],
            data.numpy["discount_bp_raw"],
            data.numpy["qty"],
            data.numpy["qty_valid"],
            data.numpy["unit_price"],
            data.numpy["unit_price_valid"],
            data.numpy["delay"],
            high_qty_out,
            expensive_out,
            late_out,
        )

    run_kernel()
    assert_projection_outputs((high_qty_out, expensive_out, late_out), expected)

    def run() -> tuple[int, ...]:
        run_kernel()
        return sentinel

    return run


def arrow_int32_buffers(data: DataSet, name: str) -> tuple[np.ndarray, np.ndarray]:
    if pa is None:
        raise RuntimeError("pyarrow is not available")
    arr = data.arrow[name]
    if arr.offset != 0:
        raise RuntimeError("benchmark does not support sliced Arrow arrays")
    validity_buf, values_buf = arr.buffers()
    values = np.frombuffer(values_buf, dtype=np.int32, count=len(arr))
    if validity_buf is None:
        validity = np.empty(0, dtype=np.uint8)
    else:
        validity = np.frombuffer(
            validity_buf, dtype=np.uint8, count=(len(arr) + 7) // 8
        )
    return values, validity


def make_numba_arrow_bits_nullable_revenue(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if numba is None:
        raise RuntimeError("numba is not available")
    if pa is None:
        raise RuntimeError("pyarrow is required for bitmask inputs")

    qty, qty_validity = arrow_int32_buffers(data, "qty")
    unit_price, unit_validity = arrow_int32_buffers(data, "unit_price")
    discount, discount_validity = arrow_int32_buffers(data, "discount_bp")
    delay = np.frombuffer(
        data.arrow["delay"].buffers()[1], dtype=np.int32, count=data.rows
    )

    @numba.njit
    def is_valid(bits, i):
        if bits.size == 0:
            return True
        return (bits[i >> 3] & (1 << (i & 7))) != 0

    @numba.njit
    def kernel(qty, qty_bits, unit_price, unit_bits, discount, discount_bits, delay):
        total = 0
        late_total = 0
        count = 0
        for i in range(qty.size):
            q = qty[i] if is_valid(qty_bits, i) else 0
            unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            d = discount[i] if is_valid(discount_bits, i) else 0
            gross = np.int64(q) * np.int64(up)
            net = gross * (10_000 - np.int64(d)) // 10_000
            valid = unit_valid and q > 0
            if valid:
                total += net
                count += 1
                if delay[i] > 30:
                    late_total += net
        return total, late_total, count

    def run() -> tuple[int, int, int]:
        return kernel(
            qty,
            qty_validity,
            unit_price,
            unit_validity,
            discount,
            discount_validity,
            delay,
        )

    return run


def make_numba_arrow_bits_split_valids_revenue(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if numba is None:
        raise RuntimeError("numba is not available")
    if pa is None:
        raise RuntimeError("pyarrow is required for bitmask inputs")

    qty, qty_validity = arrow_int32_buffers(data, "qty")
    unit_price, unit_validity = arrow_int32_buffers(data, "unit_price")
    discount, discount_validity = arrow_int32_buffers(data, "discount_bp")
    delay = np.frombuffer(
        data.arrow["delay"].buffers()[1], dtype=np.int32, count=data.rows
    )

    @numba.njit
    def is_valid(bits, i):
        if bits.size == 0:
            return True
        return (bits[i >> 3] & (1 << (i & 7))) != 0

    @numba.njit
    def kernel(qty, qty_bits, unit_price, unit_bits, discount, discount_bits, delay):
        high_qty_total = 0
        expensive_total = 0
        late_count = 0
        for i in range(qty.size):
            q = qty[i] if is_valid(qty_bits, i) else 0
            unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            d = discount[i] if is_valid(discount_bits, i) else 0
            gross = np.int64(q) * np.int64(up)
            net = gross * (10_000 - np.int64(d)) // 10_000
            base_valid = unit_valid and q > 0
            if base_valid and q >= 25:
                high_qty_total += net
            if base_valid and up >= 5_000:
                expensive_total += net
            if base_valid and delay[i] > 30:
                late_count += 1
        return high_qty_total, expensive_total, late_count

    def run() -> tuple[int, int, int]:
        return kernel(
            qty,
            qty_validity,
            unit_price,
            unit_validity,
            discount,
            discount_validity,
            delay,
        )

    return run


def make_numba_arrow_bits_unconditional_div_split_valids(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if numba is None:
        raise RuntimeError("numba is not available")
    if pa is None:
        raise RuntimeError("pyarrow is required for bitmask inputs")

    qty_raw, _ = arrow_int32_buffers(data, "qty_raw")
    unit_price_raw, _ = arrow_int32_buffers(data, "unit_price_raw")
    discount_raw, _ = arrow_int32_buffers(data, "discount_bp_raw")
    qty, qty_validity = arrow_int32_buffers(data, "qty")
    unit_price, unit_validity = arrow_int32_buffers(data, "unit_price")
    delay = np.frombuffer(
        data.arrow["delay"].buffers()[1], dtype=np.int32, count=data.rows
    )

    @numba.njit
    def is_valid(bits, i):
        if bits.size == 0:
            return True
        return (bits[i >> 3] & (1 << (i & 7))) != 0

    @numba.njit
    def kernel(
        qty_raw,
        unit_price_raw,
        discount_raw,
        qty,
        qty_bits,
        unit_price,
        unit_bits,
        delay,
    ):
        high_qty_total = 0
        expensive_total = 0
        late_count = 0
        for i in range(qty.size):
            gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
            net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000

            q = qty[i] if is_valid(qty_bits, i) else 0
            unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            base_valid = unit_valid and q > 0
            if base_valid and q >= 25:
                high_qty_total += net
            if base_valid and up >= 5_000:
                expensive_total += net
            if base_valid and delay[i] > 30:
                late_count += 1
        return high_qty_total, expensive_total, late_count

    def run() -> tuple[int, int, int]:
        return kernel(
            qty_raw,
            unit_price_raw,
            discount_raw,
            qty,
            qty_validity,
            unit_price,
            unit_validity,
            delay,
        )

    return run


def make_numba_arrow_bits_conditional_store_projection(
    data: DataSet,
) -> Callable[[], tuple[int, ...]]:
    if numba is None:
        raise RuntimeError("numba is not available")
    if pa is None:
        raise RuntimeError("pyarrow is required for bitmask inputs")

    qty_raw, _ = arrow_int32_buffers(data, "qty_raw")
    unit_price_raw, _ = arrow_int32_buffers(data, "unit_price_raw")
    discount_raw, _ = arrow_int32_buffers(data, "discount_bp_raw")
    qty, qty_validity = arrow_int32_buffers(data, "qty")
    unit_price, unit_validity = arrow_int32_buffers(data, "unit_price")
    delay = np.frombuffer(
        data.arrow["delay"].buffers()[1], dtype=np.int32, count=data.rows
    )
    high_qty_out = np.zeros(data.rows, dtype=np.int64)
    expensive_out = np.zeros(data.rows, dtype=np.int64)
    late_out = np.zeros(data.rows, dtype=np.int64)
    expected = expected_store_projection(data)
    sentinel = projection_sentinel(*expected)

    @numba.njit
    def is_valid(bits, i):
        if bits.size == 0:
            return True
        return (bits[i >> 3] & (1 << (i & 7))) != 0

    @numba.njit
    def kernel(
        qty_raw,
        unit_price_raw,
        discount_raw,
        qty,
        qty_bits,
        unit_price,
        unit_bits,
        delay,
        high_qty_out,
        expensive_out,
        late_out,
    ):
        for i in range(qty.size):
            gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
            net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000

            q = qty[i] if is_valid(qty_bits, i) else 0
            unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            base_valid = unit_valid and q > 0
            if base_valid and q >= 25:
                high_qty_out[i] = net
            if base_valid and up >= 5_000:
                expensive_out[i] = net
            if base_valid and delay[i] > 30:
                late_out[i] = 1

    def run_kernel() -> None:
        kernel(
            qty_raw,
            unit_price_raw,
            discount_raw,
            qty,
            qty_validity,
            unit_price,
            unit_validity,
            delay,
            high_qty_out,
            expensive_out,
            late_out,
        )

    run_kernel()
    assert_projection_outputs((high_qty_out, expensive_out, late_out), expected)

    def run() -> tuple[int, ...]:
        run_kernel()
        return sentinel

    return run


def make_numba_arrow_bits_select_store_projection(
    data: DataSet,
) -> Callable[[], tuple[int, ...]]:
    if numba is None:
        raise RuntimeError("numba is not available")
    if pa is None:
        raise RuntimeError("pyarrow is required for bitmask inputs")

    qty_raw, _ = arrow_int32_buffers(data, "qty_raw")
    unit_price_raw, _ = arrow_int32_buffers(data, "unit_price_raw")
    discount_raw, _ = arrow_int32_buffers(data, "discount_bp_raw")
    qty, qty_validity = arrow_int32_buffers(data, "qty")
    unit_price, unit_validity = arrow_int32_buffers(data, "unit_price")
    delay = np.frombuffer(
        data.arrow["delay"].buffers()[1], dtype=np.int32, count=data.rows
    )
    high_qty_out = np.empty(data.rows, dtype=np.int64)
    expensive_out = np.empty(data.rows, dtype=np.int64)
    late_out = np.empty(data.rows, dtype=np.int64)
    expected = expected_store_projection(data)
    sentinel = projection_sentinel(*expected)

    @numba.njit
    def is_valid(bits, i):
        if bits.size == 0:
            return True
        return (bits[i >> 3] & (1 << (i & 7))) != 0

    @numba.njit
    def kernel(
        qty_raw,
        unit_price_raw,
        discount_raw,
        qty,
        qty_bits,
        unit_price,
        unit_bits,
        delay,
        high_qty_out,
        expensive_out,
        late_out,
    ):
        for i in range(qty.size):
            gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
            net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000

            q = qty[i] if is_valid(qty_bits, i) else 0
            unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
            up = unit_price[i] if unit_valid else 0
            base_valid = unit_valid and q > 0
            high_qty_out[i] = net if base_valid and q >= 25 else 0
            expensive_out[i] = net if base_valid and up >= 5_000 else 0
            late_out[i] = 1 if base_valid and delay[i] > 30 else 0

    def run_kernel() -> None:
        kernel(
            qty_raw,
            unit_price_raw,
            discount_raw,
            qty,
            qty_validity,
            unit_price,
            unit_validity,
            delay,
            high_qty_out,
            expensive_out,
            late_out,
        )

    run_kernel()
    assert_projection_outputs((high_qty_out, expensive_out, late_out), expected)

    def run() -> tuple[int, ...]:
        run_kernel()
        return sentinel

    return run


def make_pyarrow_nullable_revenue(data: DataSet) -> Callable[[], tuple[int, int, int]]:
    if pa is None or pc is None:
        raise RuntimeError("pyarrow is not available")

    def run() -> tuple[int, int, int]:
        qty = pc.fill_null(data.arrow["qty"], 0)
        unit_valid = pc.and_kleene(
            pc.is_valid(data.arrow["unit_price"]),
            pc.not_equal(data.arrow["unit_price"], 0),
        )
        unit_price = pc.if_else(unit_valid, data.arrow["unit_price"], 0)
        discount = pc.fill_null(data.arrow["discount_bp"], 0)
        gross = pc.multiply(pc.cast(qty, pa.int64()), pc.cast(unit_price, pa.int64()))
        raw = pc.multiply(gross, pc.subtract(10_000, pc.cast(discount, pa.int64())))
        net = pc.divide(raw, 10_000)
        valid = pc.and_kleene(unit_valid, pc.greater(qty, 0))
        late = pc.greater(data.arrow["delay"], 30)
        total = pc.sum(pc.if_else(valid, net, 0)).as_py()
        late_total = pc.sum(pc.if_else(pc.and_kleene(valid, late), net, 0)).as_py()
        count = pc.sum(pc.cast(valid, pa.int64())).as_py()
        return int(total), int(late_total), int(count)

    return run


def make_pyarrow_split_valids_revenue(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if pa is None or pc is None:
        raise RuntimeError("pyarrow is not available")

    def run() -> tuple[int, int, int]:
        qty = pc.fill_null(data.arrow["qty"], 0)
        unit_valid = pc.and_kleene(
            pc.is_valid(data.arrow["unit_price"]),
            pc.not_equal(data.arrow["unit_price"], 0),
        )
        unit_price = pc.if_else(unit_valid, data.arrow["unit_price"], 0)
        discount = pc.fill_null(data.arrow["discount_bp"], 0)
        gross = pc.multiply(pc.cast(qty, pa.int64()), pc.cast(unit_price, pa.int64()))
        raw = pc.multiply(gross, pc.subtract(10_000, pc.cast(discount, pa.int64())))
        net = pc.divide(raw, 10_000)
        base_valid = pc.and_kleene(unit_valid, pc.greater(qty, 0))
        high_qty = pc.and_kleene(base_valid, pc.greater_equal(qty, 25))
        expensive = pc.and_kleene(base_valid, pc.greater_equal(unit_price, 5_000))
        late = pc.and_kleene(base_valid, pc.greater(data.arrow["delay"], 30))
        high_qty_total = pc.sum(pc.if_else(high_qty, net, 0)).as_py()
        expensive_total = pc.sum(pc.if_else(expensive, net, 0)).as_py()
        late_count = pc.sum(pc.cast(late, pa.int64())).as_py()
        return int(high_qty_total), int(expensive_total), int(late_count)

    return run


def make_pyarrow_unconditional_div_split_valids(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if pa is None or pc is None:
        raise RuntimeError("pyarrow is not available")

    def run() -> tuple[int, int, int]:
        qty_raw = data.arrow["qty_raw"]
        unit_price_raw = data.arrow["unit_price_raw"]
        discount_raw = data.arrow["discount_bp_raw"]
        gross = pc.multiply(
            pc.cast(qty_raw, pa.int64()), pc.cast(unit_price_raw, pa.int64())
        )
        raw = pc.multiply(gross, pc.subtract(10_000, pc.cast(discount_raw, pa.int64())))
        net = pc.divide(raw, 10_000)

        qty = pc.fill_null(data.arrow["qty"], 0)
        unit_valid = pc.and_kleene(
            pc.is_valid(data.arrow["unit_price"]),
            pc.not_equal(data.arrow["unit_price"], 0),
        )
        unit_price = pc.if_else(unit_valid, data.arrow["unit_price"], 0)
        base_valid = pc.and_kleene(unit_valid, pc.greater(qty, 0))
        high_qty = pc.and_kleene(base_valid, pc.greater_equal(qty, 25))
        expensive = pc.and_kleene(base_valid, pc.greater_equal(unit_price, 5_000))
        late = pc.and_kleene(base_valid, pc.greater(data.arrow["delay"], 30))
        high_qty_total = pc.sum(pc.if_else(high_qty, net, 0)).as_py()
        expensive_total = pc.sum(pc.if_else(expensive, net, 0)).as_py()
        late_count = pc.sum(pc.cast(late, pa.int64())).as_py()
        return int(high_qty_total), int(expensive_total), int(late_count)

    return run


def make_pyarrow_store_projection(
    data: DataSet,
) -> Callable[[], tuple[int, int, int, int, int, int]]:
    if pa is None or pc is None:
        raise RuntimeError("pyarrow is not available")

    def run() -> tuple[int, int, int, int, int, int]:
        qty_raw = data.arrow["qty_raw"]
        unit_price_raw = data.arrow["unit_price_raw"]
        discount_raw = data.arrow["discount_bp_raw"]
        qty = pc.fill_null(data.arrow["qty"], 0)
        unit_valid = pc.and_kleene(
            pc.is_valid(data.arrow["unit_price"]),
            pc.not_equal(data.arrow["unit_price"], 0),
        )
        unit_price = pc.if_else(unit_valid, data.arrow["unit_price"], 0)

        gross = pc.multiply(
            pc.cast(qty_raw, pa.int64()), pc.cast(unit_price_raw, pa.int64())
        )
        raw = pc.multiply(gross, pc.subtract(10_000, pc.cast(discount_raw, pa.int64())))
        net = pc.divide(raw, 10_000)
        base_valid = pc.and_kleene(unit_valid, pc.greater(qty, 0))

        high_qty = pc.and_kleene(base_valid, pc.greater_equal(qty, 25))
        expensive = pc.and_kleene(base_valid, pc.greater_equal(unit_price, 5_000))
        late = pc.and_kleene(base_valid, pc.greater(data.arrow["delay"], 30))

        high_qty_out = pc.if_else(high_qty, net, 0)
        expensive_out = pc.if_else(expensive, net, 0)
        late_out = pc.cast(late, pa.int64())
        mid = data.rows // 2
        last = data.rows - 1
        return (
            to_py_int(high_qty_out[0]),
            to_py_int(expensive_out[mid]),
            to_py_int(late_out[last]),
            to_py_int(high_qty_out[mid]),
            to_py_int(expensive_out[last]),
            to_py_int(late_out[0]),
        )

    return run


def make_simjit_nullable_revenue(data: DataSet) -> Callable[[], tuple[int, int, int]]:
    if pa is None:
        raise RuntimeError("pyarrow is required for nullable simjit inputs")

    qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
    unit_raw = sj.col("unit_price", sj.I32)
    unit_price = sj.nullif(unit_raw, sj.i32(0))
    unit_price_value = sj.coalesce(unit_price, sj.i32(0))
    discount = sj.coalesce(sj.col("discount_bp", sj.I32), sj.i32(0))
    delay = sj.col("delay", sj.I32)

    gross = qty.cast(sj.I64) * unit_price_value.cast(sj.I64)
    net = (gross * (sj.i64(10_000) - discount.cast(sj.I64))) / sj.i64(10_000)
    valid = unit_price.is_not_null() & (qty > sj.i32(0))
    late = delay > sj.i32(30)

    program = sj.query(
        total_net=net.sum(where=valid),
        late_net=net.sum(where=valid & late),
        valid_count=valid.count(),
    )

    session = make_simjit_session()
    runner = sj.prepare_program(program, data.arrow, session=session)
    print_simjit_debug("nullable_revenue_multiagg", session)
    runner.run()
    result = runner.outputs

    def run() -> tuple[int, int, int]:
        runner.run()
        return (
            to_py_int(result["total_net"][0]),
            to_py_int(result["late_net"][0]),
            to_py_int(result["valid_count"][0]),
        )

    return run


def make_simjit_split_valids_revenue(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if pa is None:
        raise RuntimeError("pyarrow is required for nullable simjit inputs")

    qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
    unit_raw = sj.col("unit_price", sj.I32)
    unit_price = sj.nullif(unit_raw, sj.i32(0))
    unit_price_value = sj.coalesce(unit_price, sj.i32(0))
    discount = sj.coalesce(sj.col("discount_bp", sj.I32), sj.i32(0))
    delay = sj.col("delay", sj.I32)

    gross = qty.cast(sj.I64) * unit_price_value.cast(sj.I64)
    net = (gross * (sj.i64(10_000) - discount.cast(sj.I64))) / sj.i64(10_000)
    base_valid = unit_price.is_not_null() & (qty > sj.i32(0))
    high_qty = base_valid & (qty >= sj.i32(25))
    expensive = base_valid & (unit_price_value >= sj.i32(5_000))
    late = base_valid & (delay > sj.i32(30))

    program = sj.query(
        high_qty_net=net.sum(where=high_qty),
        expensive_net=net.sum(where=expensive),
        late_count=late.count(),
    )

    session = make_simjit_session()
    runner = sj.prepare_program(program, data.arrow, session=session)
    print_simjit_debug("split_valids_revenue", session)
    runner.run()
    result = runner.outputs

    def run() -> tuple[int, int, int]:
        runner.run()
        return (
            to_py_int(result["high_qty_net"][0]),
            to_py_int(result["expensive_net"][0]),
            to_py_int(result["late_count"][0]),
        )

    return run


def make_simjit_unconditional_div_split_valids(
    data: DataSet,
) -> Callable[[], tuple[int, int, int]]:
    if pa is None:
        raise RuntimeError("pyarrow is required for nullable simjit inputs")

    qty_raw = sj.col("qty_raw", sj.I32)
    unit_price_raw = sj.col("unit_price_raw", sj.I32)
    discount_raw = sj.col("discount_bp_raw", sj.I32)
    qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
    unit_raw = sj.col("unit_price", sj.I32)
    unit_price = sj.nullif(unit_raw, sj.i32(0))
    unit_price_value = sj.coalesce(unit_price, sj.i32(0))
    delay = sj.col("delay", sj.I32)

    gross = qty_raw.cast(sj.I64) * unit_price_raw.cast(sj.I64)
    net = (gross * (sj.i64(10_000) - discount_raw.cast(sj.I64))) / sj.i64(10_000)
    base_valid = unit_price.is_not_null() & (qty > sj.i32(0))
    high_qty = base_valid & (qty >= sj.i32(25))
    expensive = base_valid & (unit_price_value >= sj.i32(5_000))
    late = base_valid & (delay > sj.i32(30))

    program = sj.query(
        high_qty_net=net.sum(where=high_qty),
        expensive_net=net.sum(where=expensive),
        late_count=late.count(),
    )

    session = make_simjit_session()
    runner = sj.prepare_program(program, data.arrow, session=session)
    print_simjit_debug("unconditional_div_split_valids", session)
    runner.run()
    result = runner.outputs

    def run() -> tuple[int, int, int]:
        runner.run()
        return (
            to_py_int(result["high_qty_net"][0]),
            to_py_int(result["expensive_net"][0]),
            to_py_int(result["late_count"][0]),
        )

    return run


def simjit_projection_exprs() -> tuple[Any, Any, Any, Any]:
    qty_raw = sj.col("qty_raw", sj.I32)
    unit_price_raw = sj.col("unit_price_raw", sj.I32)
    discount_raw = sj.col("discount_bp_raw", sj.I32)
    qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
    unit_raw = sj.col("unit_price", sj.I32)
    unit_price = sj.nullif(unit_raw, sj.i32(0))
    unit_price_value = sj.coalesce(unit_price, sj.i32(0))
    delay = sj.col("delay", sj.I32)

    gross = qty_raw.cast(sj.I64) * unit_price_raw.cast(sj.I64)
    net = (gross * (sj.i64(10_000) - discount_raw.cast(sj.I64))) / sj.i64(10_000)
    base_valid = unit_price.is_not_null() & (qty > sj.i32(0))
    high_qty = base_valid & (qty >= sj.i32(25))
    expensive = base_valid & (unit_price_value >= sj.i32(5_000))
    late = base_valid & (delay > sj.i32(30))
    return net, high_qty, expensive, late


def prepare_simjit_projection(
    data: DataSet, program: Any, debug_label: str
) -> tuple[Any, tuple[np.ndarray, np.ndarray, np.ndarray], tuple[int, ...]]:
    outputs = program.to_dsl()
    session = make_simjit_session()
    prepared = session.prepare_program(outputs, data.arrow, "numpy")
    print_simjit_debug(debug_label, session)
    prepared.run()
    result = prepared.result()
    arrays = (
        result["high_qty_out"],
        result["expensive_out"],
        result["late_out"],
    )
    for array in arrays:
        array.fill(0)
    expected = expected_store_projection(data)
    sentinel = projection_sentinel(*expected)
    prepared.run()
    assert_projection_outputs(arrays, expected)
    return prepared, arrays, sentinel


def make_simjit_conditional_store_projection(
    data: DataSet,
) -> Callable[[], tuple[int, ...]]:
    if pa is None:
        raise RuntimeError("pyarrow is required for nullable simjit inputs")

    net, high_qty, expensive, late = simjit_projection_exprs()
    program = sj.query(
        high_qty_out=sj.store(net, cond=high_qty),
        expensive_out=sj.store(net, cond=expensive),
        late_out=sj.store(sj.i64(1), cond=late),
    )
    kernel, _, sentinel = prepare_simjit_projection(
        data, program, "conditional_store_projection"
    )

    def run() -> tuple[int, ...]:
        kernel.run()
        return sentinel

    return run


def make_simjit_select_store_projection(
    data: DataSet,
) -> Callable[[], tuple[int, ...]]:
    if pa is None:
        raise RuntimeError("pyarrow is required for nullable simjit inputs")

    net, high_qty, expensive, late = simjit_projection_exprs()
    program = sj.query(
        high_qty_out=high_qty.ifelse(net, sj.i64(0)),
        expensive_out=expensive.ifelse(net, sj.i64(0)),
        late_out=late.ifelse(sj.i64(1), sj.i64(0)),
    )
    kernel, _, sentinel = prepare_simjit_projection(
        data, program, "select_store_projection"
    )

    def run() -> tuple[int, ...]:
        kernel.run()
        return sentinel

    return run


def to_py_int(value: Any) -> int:
    if hasattr(value, "as_py"):
        return int(value.as_py())
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def consume(value: Any) -> None:
    if isinstance(value, tuple):
        for item in value:
            int(item)
        return
    if hasattr(value, "as_py"):
        value.as_py()


def time_once(fn: Callable[[], Any], iterations: int) -> Timing:
    start = time.perf_counter_ns()
    for _ in range(iterations):
        consume(fn())
    elapsed = time.perf_counter_ns() - start
    return Timing(seconds=elapsed / 1e9 / iterations, iterations=iterations)


def choose_iterations(fn: Callable[[], Any], min_time: float) -> int:
    iterations = 1
    while True:
        timing = time_once(fn, iterations)
        total = timing.seconds * iterations
        if total >= min_time or iterations >= 1 << 28:
            return iterations
        if total <= 0:
            iterations *= 10
            continue
        scale = max(2, int(math.ceil(min_time / total)))
        iterations *= min(scale, 10)


def measure(
    fn: Callable[[], Any], *, repeats: int, min_time: float, expected
) -> dict[str, Any]:
    first_start = time.perf_counter_ns()
    first_value = fn()
    consume(first_value)
    first_seconds = (time.perf_counter_ns() - first_start) / 1e9
    if tuple(int(x) for x in first_value) != expected:
        raise AssertionError(f"expected {expected!r}, got {first_value!r}")

    consume(fn())
    iterations = choose_iterations(fn, min_time)
    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            samples.append(time_once(fn, iterations).seconds)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "first_seconds": first_seconds,
        "hot_median_seconds": statistics.median(samples),
        "hot_min_seconds": min(samples),
        "hot_samples_seconds": samples,
        "iterations_per_repeat": iterations,
    }


def parse_sizes(value: str) -> tuple[int, ...]:
    return tuple(
        int(part.replace("_", "")) for part in value.split(",") if part.strip()
    )


def parse_csv_names(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(
        int(part.replace("_", "")) for part in value.split(",") if part.strip()
    )


def write_markdown(
    path: Path, records: list[dict[str, Any]], heatmap_paths: list[Path]
) -> None:
    lines = [
        "# Simjit Selling Benchmark",
        "",
        "Lower hot median time is better.",
        "",
    ]
    if heatmap_paths:
        lines.extend(["## Heatmaps", ""])
        for image in heatmap_paths:
            label = image.stem.replace("_", " ")
            rel = image.relative_to(path.parent)
            lines.extend([f"### {label}", "", f"![{label}]({rel})", ""])

    workloads = sorted({r["workload"] for r in records})
    for workload in workloads:
        group = [r for r in records if r["workload"] == workload]
        ok = [r for r in group if r["status"] == "ok"]
        skipped = [r for r in group if r["status"] != "ok"]
        lines.extend([f"## {workload}", ""])
        if ok:
            best = min(ok, key=lambda r: r["hot_median_seconds"])
            lines.append(
                "Summary: best hot time is `{library}` at `{rows}` rows "
                "({hot_us:.3f} us).".format(
                    library=best["library"],
                    rows=best["rows"],
                    hot_us=best["hot_median_seconds"] * 1e6,
                )
            )
        else:
            lines.append("Summary: no implementation completed successfully.")
        if skipped:
            names = ", ".join(sorted({r["library"] for r in skipped}))
            lines.append(f"Skipped implementations: {names}.")
        lines.extend(["", table_for_records(group), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def table_for_records(records: list[dict[str, Any]]) -> str:
    lines = [
        "| pattern | null mod | rows | library | valid % | first ms | hot median us | rows/s | status |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in records:
        if r["status"] == "ok":
            lines.append(
                "| {pattern} | {null_mod} | {rows} | {library} | {valid_pct:.1f} | "
                "{first_ms:.3f} | {hot_us:.3f} | {rps:.3g} | ok |".format(
                    pattern=r["null_pattern"],
                    null_mod=r["null_mod"],
                    rows=r["rows"],
                    library=r["library"],
                    valid_pct=r["revenue_valid_fraction"] * 100.0,
                    first_ms=r["first_seconds"] * 1e3,
                    hot_us=r["hot_median_seconds"] * 1e6,
                    rps=r["rows"] / r["hot_median_seconds"],
                )
            )
        else:
            lines.append(
                f"| {r['null_pattern']} | {r['null_mod']} | {r['rows']} | "
                f"{r['library']} | {r.get('revenue_valid_fraction', 0.0) * 100.0:.1f} | "
                f"- | - | - | {r['reason']} |"
            )
    return "\n".join(lines)


def write_heatmaps(
    records: list[dict[str, Any]], output_dir: Path, *, baseline: str = "simjit_arrow"
) -> list[Path]:
    try:
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "simjit-matplotlib"),
        )
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        return []

    ok = [r for r in records if r["status"] == "ok"]
    if not ok:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    workloads = sorted({r["workload"] for r in ok})

    for workload in workloads:
        workload_group = [r for r in ok if r["workload"] == workload]
        patterns = sorted({r["null_pattern"] for r in workload_group})
        libraries = sorted({r["library"] for r in workload_group})

        for pattern in patterns:
            group = [r for r in workload_group if r["null_pattern"] == pattern]
            rows_values = sorted({r["rows"] for r in group})
            null_mod_values = sorted({r["null_mod"] for r in group})
            if not rows_values or not null_mod_values:
                continue

            fig, axes = plt.subplots(
                1,
                len(libraries),
                figsize=(max(4, 3.1 * len(libraries)), 4.2),
                squeeze=False,
            )
            for ax, library in zip(axes[0], libraries):
                matrix = np.full((len(null_mod_values), len(rows_values)), np.nan)
                for r in group:
                    if r["library"] != library:
                        continue
                    y = null_mod_values.index(r["null_mod"])
                    x = rows_values.index(r["rows"])
                    matrix[y, x] = r["hot_rows_per_second"]
                positive = matrix[np.isfinite(matrix) & (matrix > 0)]
                norm = None
                if positive.size:
                    vmin = float(np.min(positive))
                    vmax = float(np.max(positive))
                    if vmin < vmax:
                        norm = LogNorm(vmin=vmin, vmax=vmax)
                image = ax.imshow(matrix, aspect="auto", origin="lower", norm=norm)
                ax.set_title(library)
                ax.set_xticks(
                    range(len(rows_values)),
                    [str(x) for x in rows_values],
                    rotation=45,
                    ha="right",
                )
                ax.set_yticks(
                    range(len(null_mod_values)), [str(x) for x in null_mod_values]
                )
                ax.set_xlabel("rows")
                ax.set_ylabel("null_mod")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="rows/s")
            fig.suptitle(f"Hot throughput, {workload}, {pattern} nulls")
            fig.tight_layout()
            out = output_dir / f"{workload}_{pattern}_rows_per_second.png"
            fig.savefig(out, dpi=160)
            plt.close(fig)
            paths.append(out)

            baseline_records = {
                (r["rows"], r["null_mod"]): r for r in group if r["library"] == baseline
            }
            if not baseline_records:
                continue

            compare_libraries = [
                library for library in libraries if library != baseline
            ]
            if not compare_libraries:
                continue
            fig, axes = plt.subplots(
                1,
                len(compare_libraries),
                figsize=(max(4, 3.1 * len(compare_libraries)), 4.2),
                squeeze=False,
            )
            for ax, library in zip(axes[0], compare_libraries):
                matrix = np.full((len(null_mod_values), len(rows_values)), np.nan)
                for r in group:
                    if r["library"] != library:
                        continue
                    base = baseline_records.get((r["rows"], r["null_mod"]))
                    if base is None:
                        continue
                    y = null_mod_values.index(r["null_mod"])
                    x = rows_values.index(r["rows"])
                    matrix[y, x] = (
                        r["hot_rows_per_second"] / base["hot_rows_per_second"]
                    )
                finite = matrix[np.isfinite(matrix)]
                vmax = float(np.max(finite)) if finite.size else 1.0
                image = ax.imshow(
                    matrix, aspect="auto", origin="lower", vmin=0.0, vmax=vmax
                )
                ax.set_title(f"{library} / {baseline}")
                ax.set_xticks(
                    range(len(rows_values)),
                    [str(x) for x in rows_values],
                    rotation=45,
                    ha="right",
                )
                ax.set_yticks(
                    range(len(null_mod_values)), [str(x) for x in null_mod_values]
                )
                ax.set_xlabel("rows")
                ax.set_ylabel("null_mod")
                fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="speedup")
            fig.suptitle(f"Speedup over {baseline}, {workload}, {pattern} nulls")
            fig.tight_layout()
            out = output_dir / f"{workload}_{pattern}_speedup_vs_{baseline}.png"
            fig.savefig(out, dpi=160)
            plt.close(fig)
            paths.append(out)

    return paths


def main(argv: list[str]) -> int:
    global DEBUG_SIMJIT

    parser = argparse.ArgumentParser(
        description="Sales-oriented end-to-end benchmarks for Simjit."
    )
    parser.add_argument("--sizes", default=",".join(str(v) for v in DEFAULT_SIZES))
    parser.add_argument(
        "--null-patterns",
        default=",".join(DEFAULT_NULL_PATTERNS),
        help="Comma-separated null patterns: none, periodic, random, runs.",
    )
    parser.add_argument(
        "--null-mods",
        default=",".join(str(v) for v in DEFAULT_NULL_MODS),
        help="Comma-separated base null-rate denominators. Smaller values mean more nulls.",
    )
    parser.add_argument(
        "--full-factorial",
        action="store_true",
        help="Use the built-in factorial null-pattern/null-density preset.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--run-mean", type=int, default=DEFAULT_RUN_MEAN)
    parser.add_argument(
        "--libraries",
        default="all",
        help="Comma-separated implementation names, or all.",
    )
    parser.add_argument(
        "--workloads",
        default=",".join(DEFAULT_WORKLOADS),
        help=(
            "Comma-separated workload names: nullable_revenue_multiagg, "
            "split_valids_revenue, unconditional_div_split_valids, "
            "conditional_store_projection, select_store_projection."
        ),
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--min-time", type=float, default=DEFAULT_MIN_TIME)
    parser.add_argument("--jsonl", default="bench-results/demo-bench.jsonl")
    parser.add_argument("--markdown", default="bench-results/demo-bench.md")
    parser.add_argument(
        "--heatmap-dir",
        default="bench-results/demo-heatmaps",
        help="Directory for optional matplotlib heatmaps.",
    )
    parser.add_argument("--no-heatmaps", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Capture and print Simjit HIR, MIR, and asm for each compiled Simjit workload.",
    )
    args = parser.parse_args(argv)
    DEBUG_SIMJIT = args.debug

    sizes = parse_sizes(args.sizes)
    if args.full_factorial:
        null_patterns = DEFAULT_FULL_NULL_PATTERNS
        null_mods = DEFAULT_FULL_NULL_MODS
    else:
        null_patterns = parse_csv_names(args.null_patterns)
        null_mods = parse_ints(args.null_mods)

    jsonl_path = Path(args.jsonl)
    markdown_path = Path(args.markdown)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    base_factories: list[tuple[str, Callable[[DataSet], Callable[[], Any]]]] = [
        ("numpy_expanded_masks", make_numpy_nullable_revenue),
        ("numba_byte_masks", make_numba_nullable_revenue),
        ("numba_arrow_bits", make_numba_arrow_bits_nullable_revenue),
        ("pyarrow_compute", make_pyarrow_nullable_revenue),
        ("simjit_arrow", make_simjit_nullable_revenue),
    ]
    split_factories: list[tuple[str, Callable[[DataSet], Callable[[], Any]]]] = [
        ("numpy_expanded_masks", make_numpy_split_valids_revenue),
        ("numba_byte_masks", make_numba_split_valids_revenue),
        ("numba_arrow_bits", make_numba_arrow_bits_split_valids_revenue),
        ("pyarrow_compute", make_pyarrow_split_valids_revenue),
        ("simjit_arrow", make_simjit_split_valids_revenue),
    ]
    unconditional_div_factories: list[
        tuple[str, Callable[[DataSet], Callable[[], Any]]]
    ] = [
        ("numpy_expanded_masks", make_numpy_unconditional_div_split_valids),
        ("numba_byte_masks", make_numba_unconditional_div_split_valids),
        (
            "numba_arrow_bits",
            make_numba_arrow_bits_unconditional_div_split_valids,
        ),
        ("pyarrow_compute", make_pyarrow_unconditional_div_split_valids),
        ("simjit_arrow", make_simjit_unconditional_div_split_valids),
    ]
    conditional_store_factories: list[
        tuple[str, Callable[[DataSet], Callable[[], Any]]]
    ] = [
        ("numpy_expanded_masks", make_numpy_store_projection),
        ("numba_byte_masks", make_numba_conditional_store_projection),
        (
            "numba_arrow_bits",
            make_numba_arrow_bits_conditional_store_projection,
        ),
        ("pyarrow_compute", make_pyarrow_store_projection),
        ("simjit_arrow", make_simjit_conditional_store_projection),
    ]
    select_store_factories: list[tuple[str, Callable[[DataSet], Callable[[], Any]]]] = [
        ("numpy_expanded_masks", make_numpy_store_projection),
        ("numba_byte_masks", make_numba_select_store_projection),
        ("numba_arrow_bits", make_numba_arrow_bits_select_store_projection),
        ("pyarrow_compute", make_pyarrow_store_projection),
        ("simjit_arrow", make_simjit_select_store_projection),
    ]
    workload_configs = {
        "nullable_revenue_multiagg": (expected_nullable_revenue, base_factories),
        "split_valids_revenue": (expected_split_valids_revenue, split_factories),
        "unconditional_div_split_valids": (
            expected_unconditional_div_split_valids,
            unconditional_div_factories,
        ),
        "conditional_store_projection": (
            expected_store_projection_sentinel,
            conditional_store_factories,
        ),
        "select_store_projection": (
            expected_store_projection_sentinel,
            select_store_factories,
        ),
    }
    selected_workloads = parse_csv_names(args.workloads)
    unknown_workloads = set(selected_workloads) - set(workload_configs)
    if unknown_workloads:
        raise ValueError(f"unknown workloads: {', '.join(sorted(unknown_workloads))}")

    if args.libraries != "all":
        selected = set(parse_csv_names(args.libraries))
        known = {
            name for _, factories in workload_configs.values() for name, _ in factories
        }
        unknown = selected - known
        if unknown:
            raise ValueError(f"unknown libraries: {', '.join(sorted(unknown))}")

    records: list[dict[str, Any]] = []
    with jsonl_path.open("w", encoding="utf-8") as out:
        for workload in selected_workloads:
            expected_fn, factories = workload_configs[workload]
            if args.libraries != "all":
                selected = set(parse_csv_names(args.libraries))
                factories = [
                    (name, factory) for name, factory in factories if name in selected
                ]
            for null_pattern in null_patterns:
                for null_mod in null_mods:
                    for rows in sizes:
                        data = make_data(
                            rows,
                            null_mod=null_mod,
                            null_pattern=null_pattern,
                            seed=args.seed,
                            run_mean=args.run_mean,
                        )
                        expected = expected_fn(data)
                        stats = dataset_stats(data)
                        for library, factory in factories:
                            base = {
                                "workload": workload,
                                "rows": rows,
                                "null_pattern": null_pattern,
                                "null_mod": null_mod,
                                "seed": args.seed,
                                "run_mean": args.run_mean,
                                "library": library,
                                **stats,
                            }
                            try:
                                fn = factory(data)
                                timing = measure(
                                    fn,
                                    repeats=args.repeats,
                                    min_time=args.min_time,
                                    expected=expected,
                                )
                                record = {
                                    **base,
                                    "status": "ok",
                                    **timing,
                                    "hot_rows_per_second": rows
                                    / timing["hot_median_seconds"],
                                }
                                print(
                                    f"ok   {workload:<23} {null_pattern:<8} null_mod={null_mod:<3} "
                                    f"{rows:>9} {library:<20} "
                                    f"valid={stats['revenue_valid_fraction'] * 100:5.1f}% "
                                    f"first={record['first_seconds'] * 1e3:8.3f} ms "
                                    f"hot={record['hot_median_seconds'] * 1e6:9.3f} us"
                                )
                            except Exception as exc:
                                record = {
                                    **base,
                                    "status": "skipped",
                                    "reason": repr(exc),
                                }
                                print(
                                    f"skip {workload:<23} {null_pattern:<8} null_mod={null_mod:<3} "
                                    f"{rows:>9} {library:<20} {exc!r}"
                                )
                            out.write(json.dumps(record, sort_keys=True) + "\n")
                            out.flush()
                            records.append(record)

    heatmap_paths: list[Path] = []
    if not args.no_heatmaps:
        heatmap_paths = write_heatmaps(records, Path(args.heatmap_dir))
    write_markdown(markdown_path, records, heatmap_paths)
    print(f"wrote {jsonl_path}")
    print(f"wrote {markdown_path}")
    if not args.no_heatmaps:
        print(f"wrote heatmaps in {args.heatmap_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
