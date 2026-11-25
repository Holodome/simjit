# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license information
# SPDX-License-Identifier: Zlib

"""Randomized compatibility suite for Arrow Decimal128 scale-0 <-> signed i128.

Decimal128 scale-0 is accepted purely as a physical 16-byte two's-complement
carrier for the existing i128 path; no decimal semantics, conversion buffers, or
public dtype are introduced. Two comparison groups:

- Arrow Compute: the same operation is executed by simjit and by a
  pyarrow.compute kernel over identical Decimal128 inputs, and results are
  compared after casting widened Arrow results back to decimal128(38, 0).
- Manual i128 reference: small explicit Python helpers model signed/unsigned
  128-bit arithmetic, and results are compared through Decimal to_pylist, which
  round-trips the full i128 range even where values exceed the declared 38-digit
  precision. General operand ranges keep typical results inside decimal128(38, 0)
  so the decimal interpretation stays meaningful; curated boundary values
  deliberately push into the i128 corners the rewrite exists for.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

pa = pytest.importorskip("pyarrow")

import pyarrow.compute as pc  # noqa: E402
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

import simjit as sj  # noqa: E402
import simjit.ir as ir  # noqa: E402

ROW_COUNT = 32
SETTINGS = settings(deadline=None, max_examples=25)

DEC38 = pa.decimal128(38, 0)
INPUT_DEC = pa.decimal128(18, 0)

MASK128 = (1 << 128) - 1
COMPUTE_LIMIT = 10**18 - 1
MANUAL_LIMIT = 10**37 - 1


def to_signed(value: int) -> int:
    value &= MASK128
    return value - (1 << 128) if value >= (1 << 127) else value


def to_unsigned(value: int) -> int:
    return value & MASK128


def nullable(ints):
    return st.one_of(st.none(), ints)


def dec_array(values, dtype=INPUT_DEC):
    return pa.array([None if v is None else Decimal(v) for v in values], type=dtype)


def raw_dec_array(values, dtype=DEC38):
    """Build a scale-0 decimal array from raw i128 values.

    from_buffers skips decimal precision validation, so curated boundary values
    beyond the declared 38 digits (the i128 corners) stay representable; the
    native adapter binds the data buffer directly either way.
    """

    data = bytearray()
    validity = bytearray((len(values) + 7) // 8)
    null_count = 0
    for i, value in enumerate(values):
        if value is None:
            null_count += 1
            data += b"\0" * 16
        else:
            validity[i >> 3] |= 1 << (i & 7)
            data += (int(value) & MASK128).to_bytes(16, "little")
    buffers = [pa.py_buffer(bytes(validity)), pa.py_buffer(bytes(data))]
    return pa.Array.from_buffers(dtype, len(values), buffers, null_count=null_count)


def run_binary(session, build, left, right, dtype=INPUT_DEC):
    program = sj.query(result=build(sj.col("x", sj.I128), sj.col("y", sj.I128)))
    inputs = {"x": left, "y": right}
    if dtype is not None:
        inputs = {name: dec_array(values, dtype) for name, values in inputs.items()}
    result = sj.run_program(
        program,
        inputs,
        output="pyarrow",
        session=session,
    )
    return result["result"]


def run_unary(session, build, values, dtype=INPUT_DEC):
    program = sj.query(result=build(sj.col("x", sj.I128)))
    inputs = {"x": values}
    if dtype is not None:
        inputs = {name: dec_array(values, dtype) for name, values in inputs.items()}
    result = sj.run_program(
        program, inputs, output="pyarrow", session=session
    )
    return result["result"]


def run_manual_binary(session, build, left, right):
    return run_binary(session, build, raw_dec_array(left), raw_dec_array(right), None)


def run_manual_unary(session, build, values):
    return run_unary(session, build, raw_dec_array(values), None)


def arrow_value_list(result) -> list[int | None]:
    array = result
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.type != DEC38:
        array = pc.cast(array, DEC38)
    assert isinstance(array, pa.Array)
    assert array.type == DEC38
    return [None if v is None else int(v) for v in array.to_pylist()]


def kernel_value_list(kernel_result) -> list[int | None]:
    array = kernel_result
    if isinstance(array, pa.ChunkedArray):
        array = array.combine_chunks()
    if array.type != DEC38:
        array = pc.cast(array, DEC38)
    return [None if v is None else int(v) for v in array.to_pylist()]


def zip_with(left, right, fn):
    return [None if a is None or b is None else fn(a, b) for a, b in zip(left, right)]


def map_with(values, fn):
    return [None if v is None else fn(v) for v in values]


@pytest.fixture(scope="module")
def session():
    return sj.Session()


COMPUTE_INTS = nullable(st.integers(min_value=-COMPUTE_LIMIT, max_value=COMPUTE_LIMIT))
COMPUTE_LISTS = st.lists(COMPUTE_INTS, min_size=ROW_COUNT, max_size=ROW_COUNT)


@SETTINGS
@given(l=COMPUTE_LISTS, r=COMPUTE_LISTS)
def test_compute_add_matches_arrow(session, l, r):
    result = run_binary(session, lambda x, y: x + y, l, r)
    assert arrow_value_list(result) == kernel_value_list(pc.add(dec_array(l), dec_array(r)))


@SETTINGS
@given(l=COMPUTE_LISTS, r=COMPUTE_LISTS)
def test_compute_sub_matches_arrow(session, l, r):
    result = run_binary(session, lambda x, y: x - y, l, r)
    assert arrow_value_list(result) == kernel_value_list(
        pc.subtract(dec_array(l), dec_array(r))
    )


@SETTINGS
@given(l=COMPUTE_LISTS, r=COMPUTE_LISTS)
def test_compute_mul_matches_arrow(session, l, r):
    result = run_binary(session, lambda x, y: x * y, l, r)
    assert arrow_value_list(result) == kernel_value_list(
        pc.multiply(dec_array(l), dec_array(r))
    )


DIV_PAIRS = st.lists(
    st.tuples(
        nullable(st.integers(min_value=-(10**9 - 1), max_value=10**9 - 1)),
        st.integers(min_value=-(10**9 - 1), max_value=10**9 - 1).filter(lambda v: v != 0),
    ),
    min_size=ROW_COUNT,
    max_size=ROW_COUNT,
)


@SETTINGS
@given(pairs=DIV_PAIRS)
def test_compute_exact_div_matches_arrow(session, pairs):
    # Exact multiples only: Arrow's fractional-scale division result must cast
    # back to scale 0 without loss, so both sides agree regardless of rounding.
    left = [None if q is None or d is None else q * d for q, d in pairs]
    right = [None if d is None else d for _, d in pairs]
    result = run_binary(session, lambda x, y: sj.div(x, y), left, right)
    assert arrow_value_list(result) == kernel_value_list(
        pc.divide(dec_array(left), dec_array(right))
    )


@SETTINGS
@given(l=COMPUTE_LISTS, r=COMPUTE_LISTS)
def test_compute_min_max_matches_arrow(session, l, r):
    result_min = run_binary(session, lambda x, y: sj.min(x, y), l, r)
    result_max = run_binary(session, lambda x, y: sj.max(x, y), l, r)
    assert arrow_value_list(result_min) == kernel_value_list(
        pc.min_element_wise(dec_array(l), dec_array(r), skip_nulls=False)
    )
    assert arrow_value_list(result_max) == kernel_value_list(
        pc.max_element_wise(dec_array(l), dec_array(r), skip_nulls=False)
    )


@SETTINGS
@given(values=COMPUTE_LISTS)
def test_compute_negate_abs_matches_arrow(session, values):
    result_neg = run_unary(session, lambda x: sj.negate(x), values)
    result_abs = run_unary(session, lambda x: sj.abs(x), values)
    array = dec_array(values)
    assert arrow_value_list(result_neg) == kernel_value_list(pc.negate(array))
    assert arrow_value_list(result_abs) == kernel_value_list(pc.abs(array))


COMPARISONS = [
    ("eq", sj.cmp_eq, pc.equal),
    ("ne", sj.cmp_ne, pc.not_equal),
    ("gt", sj.cmp_gt, pc.greater),
    ("ge", sj.cmp_ge, pc.greater_equal),
    ("lt", sj.cmp_lt, pc.less),
    ("le", sj.cmp_le, pc.less_equal),
]


@pytest.mark.parametrize("name", [item[0] for item in COMPARISONS])
@SETTINGS
@given(l=COMPUTE_LISTS, r=COMPUTE_LISTS)
def test_compute_comparisons_match_arrow(session, name, l, r):
    build = {item[0]: item[1] for item in COMPARISONS}[name]
    kernel = {item[0]: item[2] for item in COMPARISONS}[name]
    result = run_binary(session, build, l, r)
    assert isinstance(result, pa.Array)
    assert result.type == pa.bool_()
    assert result.to_pylist() == kernel(dec_array(l), dec_array(r)).to_pylist()


MANUAL_INTS = nullable(st.integers(min_value=-MANUAL_LIMIT, max_value=MANUAL_LIMIT))
MANUAL_OPERANDS = st.one_of(
    MANUAL_INTS,
    st.sampled_from(
        [
            0,
            1,
            -1,
            (1 << 63) - 1,
            1 << 63,
            (1 << 64) - 1,
            1 << 64,
            (1 << 64) + 3,
            -(1 << 64) - 3,
            0xDEAD_BEEF << 64 | 12345,
            12345 << 64 | 0xDEAD_BEEF,
            (1 << 127) - 1,
            -(1 << 127),
        ]
    ),
)
MANUAL_BINARY_LISTS = st.lists(MANUAL_OPERANDS, min_size=ROW_COUNT, max_size=ROW_COUNT)
UNSIGNED_DIVIDENDS = nullable(
    st.one_of(
        st.integers(min_value=0, max_value=MANUAL_LIMIT),
        st.sampled_from([0, 1, (1 << 64) - 1, 1 << 64, (1 << 127) - 1]),
    )
)
DIVISORS = st.one_of(
    st.none(),
    st.integers(min_value=-MANUAL_LIMIT, max_value=MANUAL_LIMIT).filter(lambda v: v != 0),
)
# Shifts and rotates stay non-null: the nullable i128 rewrite rejects flagged
# shift/rotate ops with a structured error, so they are outside this suite.
NONNULL_SHIFT_AMOUNTS = st.one_of(
    st.integers(min_value=0, max_value=200),
    st.sampled_from([0, 1, 63, 64, 65, 127, 128, 129, 255]),
)
SHIFT_CASES = st.lists(
    st.tuples(
        st.one_of(
            st.integers(min_value=-MANUAL_LIMIT, max_value=MANUAL_LIMIT),
            st.sampled_from(
                [
                    0,
                    1,
                    -1,
                    (1 << 64) - 1,
                    1 << 64,
                    0xDEAD_BEEF << 64 | 12345,
                    (1 << 127) - 1,
                    -(1 << 127),
                ]
            ),
        ),
        NONNULL_SHIFT_AMOUNTS,
    ),
    min_size=ROW_COUNT,
    max_size=ROW_COUNT,
)


def trunc_divmod(a, b):
    quotient = abs(a) // abs(b)
    if (a < 0) != (b < 0):
        quotient = -quotient
    return quotient, a - quotient * b


def ref_sll(x, amount):
    return to_signed(to_unsigned(x) << (amount % 128))


def ref_srl(x, amount):
    return to_signed(to_unsigned(x) >> (amount % 128))


def ref_sra(x, amount):
    return to_signed(to_signed(x) >> min(amount % 128, 127))


def ref_rotl(x, amount):
    amount %= 128
    value = to_unsigned(x)
    if amount == 0:
        return to_signed(value)
    return to_signed((value << amount) | (value >> (128 - amount)))


def ref_rotr(x, amount):
    amount %= 128
    value = to_unsigned(x)
    if amount == 0:
        return to_signed(value)
    return to_signed((value >> amount) | (value << (128 - amount)))


def ref_popcnt(x):
    return to_unsigned(x).bit_count()


def ref_lzcnt(x):
    return 128 - to_unsigned(x).bit_length()


def ref_tzcnt(x):
    value = to_unsigned(x)
    return 128 if value == 0 else (value & -value).bit_length() - 1


@SETTINGS
@given(l=MANUAL_BINARY_LISTS, r=MANUAL_BINARY_LISTS)
def test_manual_mod_matches_reference(session, l, r):
    divisors = [1 if v == 0 else v for v in r]
    result = run_manual_binary(session, lambda x, y: sj.mod(x, y), l, divisors)
    assert arrow_value_list(result) == zip_with(
        l, divisors, lambda a, b: trunc_divmod(a, b)[1]
    )


@SETTINGS
@given(
    l=st.lists(UNSIGNED_DIVIDENDS, min_size=ROW_COUNT, max_size=ROW_COUNT),
    r=st.lists(DIVISORS, min_size=ROW_COUNT, max_size=ROW_COUNT),
)
def test_manual_udiv_umod_matches_reference(session, l, r):
    dividends = [to_unsigned(v) if v is not None else None for v in l]
    divisors = [to_unsigned(v) if v is not None else None for v in r]
    result_div = run_manual_binary(session, lambda x, y: sj.udiv(x, y), l, r)
    result_mod = run_manual_binary(session, lambda x, y: sj.umod(x, y), l, r)
    assert arrow_value_list(result_div) == zip_with(dividends, divisors, lambda a, b: a // b)
    assert arrow_value_list(result_mod) == zip_with(dividends, divisors, lambda a, b: a % b)


@SETTINGS
@given(l=MANUAL_BINARY_LISTS, r=MANUAL_BINARY_LISTS)
def test_manual_umin_umax_matches_reference(session, l, r):
    def unsigned_min(a, b):
        return to_signed(min(to_unsigned(a), to_unsigned(b)))

    def unsigned_max(a, b):
        return to_signed(max(to_unsigned(a), to_unsigned(b)))

    result_min = run_manual_binary(session, lambda x, y: sj.umin(x, y), l, r)
    result_max = run_manual_binary(session, lambda x, y: sj.umax(x, y), l, r)
    assert arrow_value_list(result_min) == zip_with(l, r, unsigned_min)
    assert arrow_value_list(result_max) == zip_with(l, r, unsigned_max)


@SETTINGS
@given(l=MANUAL_BINARY_LISTS, r=MANUAL_BINARY_LISTS)
def test_manual_bitwise_matches_reference(session, l, r):
    cases = [
        ("and", lambda x, y: sj.and_(x, y), lambda a, b: a & b),
        ("or", lambda x, y: sj.or_(x, y), lambda a, b: a | b),
        ("xor", lambda x, y: sj.xor(x, y), lambda a, b: a ^ b),
        ("andnot", lambda x, y: sj.andnot(x, y), lambda a, b: (~a) & b),
    ]
    for _, build, reference in cases:
        result = run_manual_binary(session, build, l, r)
        assert arrow_value_list(result) == zip_with(
            l, r, lambda a, b, fn=reference: to_signed(fn(a, b))
        )


@SETTINGS
@given(values=MANUAL_BINARY_LISTS)
def test_manual_not_matches_reference(session, values):
    result = run_manual_unary(session, lambda x: sj.not_(x), values)
    assert arrow_value_list(result) == map_with(values, lambda v: to_signed(~v))


@SETTINGS
@given(pairs=SHIFT_CASES)
def test_manual_sll_srl_sra_matches_reference(session, pairs):
    values = [v for v, _ in pairs]
    amounts = [a for _, a in pairs]
    result_sll = run_manual_binary(session, lambda x, y: sj.sll(x, y), values, amounts)
    result_srl = run_manual_binary(session, lambda x, y: sj.srl(x, y), values, amounts)
    result_sra = run_manual_binary(session, lambda x, y: sj.sra(x, y), values, amounts)
    assert arrow_value_list(result_sll) == zip_with(values, amounts, ref_sll)
    assert arrow_value_list(result_srl) == zip_with(values, amounts, ref_srl)
    assert arrow_value_list(result_sra) == zip_with(values, amounts, ref_sra)


@SETTINGS
@given(pairs=SHIFT_CASES)
def test_manual_rotl_rotr_matches_reference(session, pairs):
    values = [v for v, _ in pairs]
    amounts = [a for _, a in pairs]
    result_rotl = run_manual_binary(session, lambda x, y: sj.rotl(x, y), values, amounts)
    result_rotr = run_manual_binary(session, lambda x, y: sj.rotr(x, y), values, amounts)
    assert arrow_value_list(result_rotl) == zip_with(values, amounts, ref_rotl)
    assert arrow_value_list(result_rotr) == zip_with(values, amounts, ref_rotr)


@SETTINGS
@given(values=MANUAL_BINARY_LISTS)
def test_manual_bit_counts_match_reference(session, values):
    result_popcnt = run_manual_unary(session, lambda x: sj.popcnt(x), values)
    result_lzcnt = run_manual_unary(session, lambda x: sj.lzcnt(x), values)
    result_tzcnt = run_manual_unary(session, lambda x: sj.tzcnt(x), values)
    assert arrow_value_list(result_popcnt) == map_with(values, ref_popcnt)
    assert arrow_value_list(result_lzcnt) == map_with(values, ref_lzcnt)
    assert arrow_value_list(result_tzcnt) == map_with(values, ref_tzcnt)


def test_schema_maps_decimal128_to_i128():
    from simjit.integrations._pyarrow import schema_from_pyarrow

    schema = pa.schema(
        [
            ("wide", pa.decimal128(38, 0)),
            ("mid", pa.decimal128(18, 0)),
            ("small", pa.decimal128(10, 0)),
        ]
    )
    assert schema_from_pyarrow(schema) == {
        "wide": ir.I128,
        "mid": ir.I128,
        "small": ir.I128,
    }


def test_schema_maps_decimal_record_batch_table_and_single_chunk():
    from simjit.integrations._pyarrow import schema_from_pyarrow

    array = pa.array([Decimal(1), Decimal(2), Decimal(3)], type=DEC38)
    batch = pa.RecordBatch.from_arrays([array], names=["x"])
    table = pa.Table.from_batches([batch])
    assert schema_from_pyarrow(batch) == {"x": ir.I128}
    assert schema_from_pyarrow(table) == {"x": ir.I128}

    single_chunk = pa.chunked_array([array])
    assert single_chunk.num_chunks == 1
    result = sj.run_program(
        sj.query(result=sj.col("x", sj.I128)), {"x": single_chunk}, output="pyarrow"
    )
    assert arrow_value_list(result["result"]) == [1, 2, 3]


def test_rejects_nonzero_scale_and_decimal256():
    from simjit.integrations._pyarrow import schema_from_pyarrow

    for bad_type in [pa.decimal128(18, 1), pa.decimal128(38, -2)]:
        with pytest.raises(TypeError, match="scale-0"):
            schema_from_pyarrow(pa.schema([("x", bad_type)]))
    with pytest.raises(TypeError, match="scale-0"):
        sj.run_program(
            sj.query(result=sj.col("x", sj.I128)),
            {"x": pa.array([Decimal("1.5")], type=pa.decimal128(18, 1))},
            output="pyarrow",
        )
    with pytest.raises(TypeError, match="decimal256"):
        schema_from_pyarrow(pa.schema([("x", pa.decimal256(40, 0))]))
    with pytest.raises(TypeError, match="decimal256"):
        sj.run_program(
            sj.query(result=sj.col("x", sj.I128)),
            {"x": pa.array([Decimal(1)], type=pa.decimal256(40, 0))},
            output="pyarrow",
        )


def test_rejects_u128_pyarrow_output():
    import numpy as np

    source = np.zeros(1, dtype=np.dtype("V16"))
    program = sj.query(result=sj.col("x", sj.U128))
    with pytest.raises(TypeError, match="unsigned 128-bit"):
        sj.run_program(
            program,
            {"x": ir.BufferHandle(ty=ir.U128, buf=source)},
            output="pyarrow",
        )


def test_decimal_round_trips_full_i128_range_through_i128_lowering():
    values = [0, 1, -1, (1 << 64) + 3, (1 << 127) - 1, -(1 << 127)]
    program = sj.query(result=sj.col("x", sj.I128))
    info = sj.inspect(program, {"x": ir.I128})
    assert "i128" in info.serialized
    assert "int-cast" not in info.serialized
    result = sj.run_program(program, {"x": raw_dec_array(values)}, output="pyarrow")
    assert arrow_value_list(result["result"]) == values


def test_decimal_arithmetic_uses_i128_lowering_without_conversion():
    program = sj.query(result=sj.col("x", sj.I128) + sj.col("y", sj.I128))
    info = sj.inspect(program, {"x": ir.I128, "y": ir.I128})
    assert "(arg 0 i128" in info.serialized
    assert "(arg 1 i128" in info.serialized
    assert "int-cast" not in info.serialized
