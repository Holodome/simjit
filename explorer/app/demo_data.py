# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import math

PYTHON_ROWS = [1024, 4096, 65536, 1048576]
SPEEDUP_COLOR_LOW = 0.35
SPEEDUP_COLOR_HIGH = 3.5
PYTHON_SPEEDUP_LOSS_LOW = 0.5
PYTHON_NUMBA_SPEEDUP_HIGH = 5.0
PYTHON_PYARROW_SPEEDUP_HIGH = 160.0
BENCHMARK_ENVIRONMENT = {
    "title": "Host details",
    "items": [
        "Zen 4",
        "4 VCPU",
        "8 GB RAM",
        "Linux 6.8.0-111-generic",
        "Ubuntu 24.04.4 LTS",
        "2396.398 MHz",
    ],
}
PYTHON_BENCHMARK_ENVIRONMENT = {
    **BENCHMARK_ENVIRONMENT,
    "sections": [
        {
            "title": "Python stack",
            "items": [
                "Python 3.12.3",
                "NumPy 2.3.4",
                "PyArrow 22.0.0",
                "Numba 0.65.1",
            ],
        }
    ],
}
ENGINE_BENCHMARK_ENVIRONMENT = {
    **BENCHMARK_ENVIRONMENT,
    "sections": [
        {
            "title": "Arrow/Gandiva build",
            "items": [
                "Arrow a0d2885b101a",
                "GCC 13.3.0",
                "compiled on target machine",
            ],
        }
    ],
}
DUCKDB_BENCHMARK_ENVIRONMENT = {
    **BENCHMARK_ENVIRONMENT,
    "sections": [
        {
            "title": "Arm Host details",
            "items": [
                "Apple M4",
            ],
        },
        {
            "title": "DuckDB build",
            "items": [
                "DuckDB 932e831ce75b",
                "compiled on target machine",
            ],
        },
    ],
}
PROJECT_LINKS = {
    "asmjit": "https://asmjit.com/",
    "arrow": "https://arrow.apache.org/",
    "arrow_compute": "https://arrow.apache.org/docs/cpp/compute.html",
    "duckdb": "https://duckdb.org/",
    "gandiva": "https://arrow.apache.org/docs/cpp/gandiva.html",
    "numpy": "https://numpy.org/",
    "python_buffer": "https://docs.python.org/3/c-api/buffer.html",
    "pyarrow": "https://arrow.apache.org/docs/python/",
    "pyarrow_compute": "https://arrow.apache.org/docs/python/compute.html",
}
DUCKDB_BUILDER_SNIPPETS = {
    "Between": """static simjit::Predicate Between(simjit::FunctionBuilder &b,
                                 simjit::Value value,
                                 int64_t lo,
                                 int64_t hi) {
    return b.and_(b.cmp_ge(value, b.i64(lo)), b.cmp_le(value, b.i64(hi)));
}""",
    "BuildSumAddI64": """static SumAddI64Kernel BuildSumAddI64(simjit::jit::JitContext &ctx,
                                      const char *name) {
    return simjit::jit::vectorized_function<InputArr<I64>, InputArr<I64>,
                                            OutputScalar<I64>>(
        ctx, name, [](simjit::FunctionBuilder &b) {
            auto a = b.input_arg(I64);
            auto b_col = b.input_arg(I64);
            auto sum = b.arg(I64);
            b.sum(b.add(a, b_col), sum);
        });
}""",
    "BuildSumNetI64": """static SumNetI64Kernel BuildSumNetI64(simjit::jit::JitContext &ctx,
                                      const char *name) {
    return simjit::jit::vectorized_function<InputArr<I64>, InputArr<I64>,
                                            OutputScalar<I64>>(
        ctx, name, [](simjit::FunctionBuilder &b) {
            auto price = b.input_arg(I64);
            auto discount = b.input_arg(I64);
            auto net = b.div(b.mul(price, b.sub(b.i64(10000), discount)), b.i64(10000));
            b.sum(net, b.arg(I64));
        });
}""",
    "BuildCountShipFilter": """static CountShipFilterKernel BuildCountShipFilter(simjit::jit::JitContext &ctx,
                                                const char *name) {
    return simjit::jit::vectorized_function<InputArr<I32>, InputArr<I32>,
                                            OutputScalar<I64>>(
        ctx, name, [](simjit::FunctionBuilder &b) {
            auto shipdate = b.input_arg(I32);
            auto quantity = b.input_arg(I32);
            auto pred = b.and_(b.cmp_le(shipdate, b.i32(10592)),
                               b.cmp_lt(quantity, b.i32(24)));
            b.countif(pred, b.arg(I64));
        });
}""",
    "BuildQ1Multi": """static Q1MultiKernel BuildQ1Multi(simjit::jit::JitContext &ctx,
                                const char *name) {
    return simjit::jit::vectorized_function<InputArr<I64>, InputArr<I64>,
                                            InputArr<I64>, OutputScalar<I64>,
                                            OutputScalar<I64>, OutputScalar<I64>>(
        ctx, name, [](simjit::FunctionBuilder &b) {
            auto price = b.input_arg(I64);
            auto discount = b.input_arg(I64);
            auto tax = b.input_arg(I64);
            auto discounted = b.mul(price, b.sub(b.i64(10000), discount));
            auto net = b.div(discounted, b.i64(10000));
            auto charge = b.div(b.mul(discounted, b.add(b.i64(10000), tax)),
                                b.i64(100000000));
            b.sum(net, b.arg(I64));
            b.sum(charge, b.arg(I64));
            b.sum(b.i64(1), b.arg(I64));
        });
}""",
    "BuildQ19Mask": """static Q19MaskKernel BuildQ19Mask(simjit::jit::JitContext &ctx,
                                const char *name) {
    return simjit::jit::vectorized_function<InputArr<I64>, InputArr<I64>,
                                            InputArr<I64>, InputArr<I64>,
                                            InputArr<I64>, InputArr<I64>,
                                            OutputScalar<I64>, OutputScalar<I64>>(
        ctx, name, [](simjit::FunctionBuilder &b) {
            auto brand = b.input_arg(I64);
            auto container = b.input_arg(I64);
            auto quantity = b.input_arg(I64);
            auto size = b.input_arg(I64);
            auto price = b.input_arg(I64);
            auto discount = b.input_arg(I64);
            auto g1 = b.and_(b.cmp_eq(brand, b.i64(12)),
                             b.and_(Between(b, container, 1, 4),
                                    b.and_(Between(b, quantity, 1, 11),
                                           Between(b, size, 1, 5))));
            auto g2 = b.and_(b.cmp_eq(brand, b.i64(23)),
                             b.and_(Between(b, container, 5, 8),
                                    b.and_(Between(b, quantity, 10, 20),
                                           Between(b, size, 1, 10))));
            auto pred = b.or_(g1, g2);
            auto revenue = b.div(b.mul(price, b.sub(b.i64(10000), discount)),
                                 b.i64(10000));
            b.countif(pred, b.arg(I64));
            b.sum_if(revenue, pred, b.arg(I64));
        });
}""",
    "BuildNullableRevenue": """static NullableRevenueKernel BuildNullableRevenue(simjit::jit::JitContext &ctx,
                                                const char *name) {
    return simjit::jit::vectorized_function<InputArr<I64>, InputArr<I64>,
                                            InputArr<I64>, InputArr<I64>,
                                            InputArr<I1>, InputArr<I1>,
                                            InputArr<I1>, OutputScalar<I64>,
                                            OutputScalar<I64>, OutputScalar<I64>,
                                            OutputScalar<I64>>(
        ctx, name, [](simjit::FunctionBuilder &b) {
            auto qty_raw = b.input_arg(I64);
            auto price = b.input_arg(I64);
            auto discount_raw = b.input_arg(I64);
            auto delay = b.input_arg(I64);
            auto qty_valid = b.input_predicate_arg();
            auto price_valid = b.input_predicate_arg();
            auto discount_valid = b.input_predicate_arg();
            auto qty = b.select(qty_valid, qty_raw, b.i64(0));
            auto discount = b.select(discount_valid, discount_raw, b.i64(0));
            auto valid = b.and_(price_valid, b.cmp_gt(qty, b.i64(0)));
            auto late = b.and_(valid, b.cmp_gt(delay, b.i64(30)));
            auto net = b.div(b.mul(b.mul(qty, price), b.sub(b.i64(10000), discount)),
                             b.i64(10000));
            b.sum_if(net, valid, b.arg(I64));
            b.sum_if(net, late, b.arg(I64));
            b.countif(valid, b.arg(I64));
            b.countif(late, b.arg(I64));
        });
}""",
}

ENGINE_BENCH_SNIPPETS = {
    "BM_Arrow_Revenue": """static void BM_Arrow_Revenue(benchmark::State &state) {
    auto price = batch->GetColumnByName("l_extendedprice");
    auto discount = batch->GetColumnByName("l_discount");
    for (auto _ : state) {
        auto one_minus_discount = arrow::compute::CallFunction(
            "subtract", {arrow::Datum(arrow::MakeScalar<double>(1.0)), discount});
        auto revenue = arrow::compute::CallFunction(
            "multiply", {price, one_minus_discount.ValueOrDie()});
        benchmark::DoNotOptimize(revenue);
    }
}""",
    "BM_Gandiva_Revenue": """static void BM_Gandiva_Revenue(benchmark::State &state) {
    auto price = gandiva::TreeExprBuilder::MakeField(field("l_extendedprice", arrow::float64()));
    auto discount = gandiva::TreeExprBuilder::MakeField(field("l_discount", arrow::float64()));
    auto expr = gandiva::TreeExprBuilder::MakeFunction(
        "multiply",
        {price, gandiva::TreeExprBuilder::MakeFunction(
            "subtract", {gandiva::TreeExprBuilder::MakeLiteral(1.0), discount},
            arrow::float64())},
        arrow::float64());
    RunGandivaProject(state, gandiva::TreeExprBuilder::MakeExpression(expr, field("revenue", arrow::float64())));
}""",
    "simjit_revenue_builder": """static void simjit_revenue_builder(simjit::FunctionBuilder &b) {
    auto price = b.input_arg(F64);
    auto discount = b.input_arg(F64);
    auto revenue = b.mul(price, b.sub(b.f64(1.0), discount));
    b.store(revenue, b.arg(F64));
}""",
    "BM_simjit_Revenue": """static void BM_simjit_Revenue(benchmark::State &state) {
    simjit::jit::JitContext ctx;
    auto fn = vectorized_function<InputArr<F64>, InputArr<F64>, OutputArr<F64>>(
        ctx, "revenue", simjit_revenue_builder);
    for (auto _ : state) {
        fn(kNumRows, price_data, discount_data, result_data);
        benchmark::DoNotOptimize(result_data);
    }
}""",
    "BM_Arrow_Filter_DateQty": """static void BM_Arrow_Filter_DateQty(benchmark::State &state) {
    auto ship = batch->GetColumnByName("l_shipdate");
    auto qty = batch->GetColumnByName("l_quantity");
    for (auto _ : state) {
        auto c1 = arrow::compute::CallFunction("less_equal", {ship, arrow::Datum(arrow::MakeScalar<int32_t>(10592))});
        auto c2 = arrow::compute::CallFunction("less", {qty, arrow::Datum(arrow::MakeScalar<int32_t>(24))});
        auto both = arrow::compute::CallFunction("and_kleene", {c1.ValueOrDie(), c2.ValueOrDie()});
        benchmark::DoNotOptimize(both);
    }
}""",
    "BM_Gandiva_Filter_DateQty": """static void BM_Gandiva_Filter_DateQty(benchmark::State &state) {
    auto ship = gandiva::TreeExprBuilder::MakeField(field("l_shipdate", arrow::int32()));
    auto qty = gandiva::TreeExprBuilder::MakeField(field("l_quantity", arrow::int32()));
    auto c1 = gandiva::TreeExprBuilder::MakeFunction(
        "less_than_or_equal_to", {ship, gandiva::TreeExprBuilder::MakeLiteral(10592)}, arrow::boolean());
    auto c2 = gandiva::TreeExprBuilder::MakeFunction(
        "less_than", {qty, gandiva::TreeExprBuilder::MakeLiteral(24)}, arrow::boolean());
    RunGandivaProject(state, gandiva::TreeExprBuilder::MakeExpression(
        gandiva::TreeExprBuilder::MakeAnd({c1, c2}), field("res", arrow::boolean())));
}""",
    "simjit_Filter_DateQty_builder": """static void simjit_Filter_DateQty_builder(simjit::FunctionBuilder &b) {
    auto ship = b.input_arg(I32);
    auto qty = b.input_arg(I32);
    auto pred = b.and_(b.cmp_le(ship, b.i32(10592)), b.cmp_lt(qty, b.i32(24)));
    b.store(pred, b.arg(I1));
}""",
    "BM_simjit_DateQty": """static void BM_simjit_DateQty(benchmark::State &state) {
    simjit::jit::JitContext ctx;
    auto fn = vectorized_function<InputArr<I32>, InputArr<I32>, OutputArr<I1>>(
        ctx, "date_qty", simjit_Filter_DateQty_builder);
    for (auto _ : state) {
        fn(kNumRows, ship_data, qty_data, result_bits);
        benchmark::DoNotOptimize(result_bits);
    }
}""",
}


def _cpp_function_source(name: str) -> str:
    return DUCKDB_BUILDER_SNIPPETS[name]


def _engine_bench_function_source(name: str) -> str:
    return ENGINE_BENCH_SNIPPETS[name]


def _engine_bench_sources(*names: str) -> str:
    return "\n\n".join(_engine_bench_function_source(name) for name in names)


def _duckdb_builder_source(name: str, helpers: tuple[str, ...] = ()) -> str:
    snippets = [_cpp_function_source(helper) for helper in helpers]
    snippets.append(_cpp_function_source(name))
    return "\n\n".join(snippets)


PYTHON_WORKLOADS = [
    ("conditional_store_projection", "Conditional projection"),
    ("select_store_projection", "CASE projection"),
    ("nullable_revenue_multiagg", "Revenue aggregates"),
    ("split_valids_revenue", "Split validity aggregates"),
    ("unconditional_div_split_valids", "Unconditional division"),
]

PYTHON_RECORDS = [
    ("conditional_store_projection", 1024, "numba_arrow_bits", 0.000014, 253465346.534653),
    ("conditional_store_projection", 1024, "pyarrow_compute", 0.00111, 7419859.71827720),
    ("conditional_store_projection", 1024, "simjit_arrow", 0.000003, 808846761.453397),
    ("conditional_store_projection", 4096, "numba_arrow_bits", 0.000024, 318135922.330097),
    ("conditional_store_projection", 4096, "pyarrow_compute", 0.000763, 22275033.5811358),
    ("conditional_store_projection", 4096, "simjit_arrow", 0.000006, 1212551805.80225),
    ("conditional_store_projection", 65536, "numba_arrow_bits", 0.000207, 333829132.622914),
    ("conditional_store_projection", 65536, "pyarrow_compute", 0.004304, 58894841.0398261),
    ("conditional_store_projection", 65536, "simjit_arrow", 0.000077, 1316803632.78346),
    ("conditional_store_projection", 1048576, "numba_arrow_bits", 0.003472, 316007473.931650),
    ("conditional_store_projection", 1048576, "pyarrow_compute", 0.057317, 60219059.7399265),
    ("conditional_store_projection", 1048576, "simjit_arrow", 0.001785, 651086027.160459),
    ("select_store_projection", 1024, "numba_arrow_bits", 0.000029, 218290343.210403),
    ("select_store_projection", 1024, "pyarrow_compute", 0.00047, 7392541.04159748),
    ("select_store_projection", 1024, "simjit_arrow", 0.000004, 764179104.477612),
    ("select_store_projection", 4096, "numba_arrow_bits", 0.000024, 264070659.531945),
    ("select_store_projection", 4096, "pyarrow_compute", 0.000704, 22194046.1870239),
    ("select_store_projection", 4096, "simjit_arrow", 0.000006, 1098123324.39678),
    ("select_store_projection", 65536, "numba_arrow_bits", 0.000241, 279540355.397070),
    ("select_store_projection", 65536, "pyarrow_compute", 0.001637, 58689988.8684607),
    ("select_store_projection", 65536, "simjit_arrow", 0.000064, 1224171103.01672),
    ("select_store_projection", 1048576, "numba_arrow_bits", 0.003824, 280223606.773624),
    ("select_store_projection", 1048576, "pyarrow_compute", 0.017911, 60773242.7474841),
    ("select_store_projection", 1048576, "simjit_arrow", 0.001553, 676770043.301108),
    ("nullable_revenue_multiagg", 1024, "numba_arrow_bits", 0.180885, 257157207.433451),
    ("nullable_revenue_multiagg", 1024, "pyarrow_compute", 0.000582, 7745019.43818355),
    ("nullable_revenue_multiagg", 1024, "simjit_arrow", 0.000024, 356173913.043478),
    ("nullable_revenue_multiagg", 4096, "numba_arrow_bits", 0.178448, 294803512.307471),
    ("nullable_revenue_multiagg", 4096, "pyarrow_compute", 0.00052, 22934444.2204753),
    ("nullable_revenue_multiagg", 4096, "simjit_arrow", 0.000023, 757256424.477722),
    ("nullable_revenue_multiagg", 65536, "numba_arrow_bits", 0.179129, 308647266.322239),
    ("nullable_revenue_multiagg", 65536, "pyarrow_compute", 0.001533, 61987817.3357043),
    ("nullable_revenue_multiagg", 65536, "simjit_arrow", 0.000071, 1176800143.65236),
    ("nullable_revenue_multiagg", 1048576, "numba_arrow_bits", 0.184493, 310413262.285376),
    ("nullable_revenue_multiagg", 1048576, "pyarrow_compute", 0.017651, 63056230.9563388),
    ("nullable_revenue_multiagg", 1048576, "simjit_arrow", 0.000886, 1239617111.97831),
    ("split_valids_revenue", 1024, "numba_arrow_bits", 0.236876, 245976459.284170),
    ("split_valids_revenue", 1024, "pyarrow_compute", 0.000499, 6748518.81874559),
    ("split_valids_revenue", 1024, "simjit_arrow", 0.000021, 357541899.441341),
    ("split_valids_revenue", 4096, "numba_arrow_bits", 0.189086, 281376657.278285),
    ("split_valids_revenue", 4096, "pyarrow_compute", 0.000575, 20712393.0500212),
    ("split_valids_revenue", 4096, "simjit_arrow", 0.000023, 756138083.810227),
    ("split_valids_revenue", 65536, "numba_arrow_bits", 0.194189, 293115786.460570),
    ("split_valids_revenue", 65536, "pyarrow_compute", 0.001732, 55255960.1498764),
    ("split_valids_revenue", 65536, "simjit_arrow", 0.000075, 1148344138.77694),
    ("split_valids_revenue", 1048576, "numba_arrow_bits", 0.196338, 294558548.152127),
    ("split_valids_revenue", 1048576, "pyarrow_compute", 0.018633, 57698181.6403001),
    ("split_valids_revenue", 1048576, "simjit_arrow", 0.000903, 1205403864.36999),
    ("unconditional_div_split_valids", 1024, "numba_arrow_bits", 0.22286, 275713516.424340),
    ("unconditional_div_split_valids", 1024, "pyarrow_compute", 0.000498, 7013938.83352170),
    ("unconditional_div_split_valids", 1024, "simjit_arrow", 0.000023, 366237482.117310),
    ("unconditional_div_split_valids", 4096, "numba_arrow_bits", 0.28464, 331204010.673567),
    ("unconditional_div_split_valids", 4096, "pyarrow_compute", 0.000546, 21256941.2008926),
    ("unconditional_div_split_valids", 4096, "simjit_arrow", 0.000022, 818545163.868905),
    ("unconditional_div_split_valids", 65536, "numba_arrow_bits", 0.225411, 345913078.360375),
    ("unconditional_div_split_valids", 65536, "pyarrow_compute", 0.001679, 57130453.5061009),
    ("unconditional_div_split_valids", 65536, "simjit_arrow", 0.000066, 1279450236.22662),
    ("unconditional_div_split_valids", 1048576, "numba_arrow_bits", 0.233704, 349054342.673619),
    ("unconditional_div_split_valids", 1048576, "pyarrow_compute", 0.020635, 60230114.6322608),
    ("unconditional_div_split_valids", 1048576, "simjit_arrow", 0.000996, 1254801346.49513),
]

PYTHON_README_CODE = """import numpy as np
import simjit as sj

program = sj.query(
    net=sj.col("price") * (1.0 - sj.col("discount")),
    is_large=(sj.col("qty") > 100).ifelse(1, 0),
)

result = program.run({
    "price": np.array([10.0, 20.0, 30.0]),
    "discount": np.array([0.1, 0.2, 0.0]),
    "qty": np.array([50, 150, 200]),
})"""

README_NUMPY_CODE = """import numpy as np

price = arrays["price"]
discount = arrays["discount"]
qty = arrays["qty"]

result = {
    "net": price * (1.0 - discount),
    "is_large": np.where(qty > 100, 1, 0),
}"""

README_PYARROW_COMPUTE_CODE = """import pyarrow as pa
import pyarrow.compute as pc

net = pc.multiply(
    table["price"],
    pc.subtract(pa.scalar(1.0), table["discount"]),
)
is_large = pc.if_else(
    pc.greater(table["qty"], pa.scalar(100, type=pa.int32())),
    pa.scalar(1, type=pa.int32()),
    pa.scalar(0, type=pa.int32()),
)

result = pa.table({"net": net, "is_large": is_large})"""

README_SIMJIT_API_CODE = """static void build_readme_projection(FunctionBuilder &b) {
    Value price = b.input_arg(F64);
    Value discount = b.input_arg(F64);
    Value qty = b.input_arg(I32);

    Argument net_out = b.arg(F64);
    Argument is_large_out = b.arg(I32);

    Value net = b.mul(price, b.sub(b.f64(1.0), discount));
    Predicate large = b.cmp_gt(qty, b.i32(100));

    b.store(net, net_out);
    b.store(b.select(large, b.i32(1), b.i32(0)), is_large_out);
}"""

SIMJIT_NULLABLE_CODE = """qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
unit_raw = sj.col("unit_price", sj.I32)
unit_price = sj.nullif(unit_raw, sj.i32(0))
discount = sj.coalesce(sj.col("discount_bp", sj.I32), sj.i32(0))

gross = qty.cast(sj.I64) * sj.coalesce(unit_price, sj.i32(0)).cast(sj.I64)
net = (gross * (sj.i64(10_000) - discount.cast(sj.I64))) / sj.i64(10_000)
valid = unit_price.is_not_null() & (qty > sj.i32(0))
late = sj.col("delay", sj.I32) > sj.i32(30)

program = sj.query(
    total_net=net.sum(where=valid),
    late_net=net.sum(where=valid & late),
    valid_count=valid.count(),
)"""

NULLABLE_SIMJIT_API_CODE = """static void build_nullable_revenue(FunctionBuilder &b) {
    Value qty = b.input_arg(I32);
    Value unit_price = b.input_arg(I32);
    Value discount_bp = b.input_arg(I32);
    Value delay = b.input_arg(I32);

    Predicate unit_valid = b.input_predicate_arg();
    Predicate discount_valid = b.input_predicate_arg();

    Value discount = b.select(discount_valid, discount_bp, b.i32(0));
    Predicate valid = b.and_(unit_valid, b.cmp_gt(qty, b.i32(0)));
    Predicate late = b.cmp_gt(delay, b.i32(30));

    Value gross = b.mul(b.sext(qty, I64), b.sext(unit_price, I64));
    Value scale = b.sub(b.i64(10000), b.sext(discount, I64));
    Value net = b.div(b.mul(gross, scale), b.i64(10000));

    b.sum_if(net, valid, b.arg(I64));
    b.sum_if(net, b.and_(valid, late), b.arg(I64));
    b.countif(valid, b.arg(I64));
}"""

NUMBA_BITMAP_CODE = """@numba.njit
def is_valid(bits, i):
    return bits.size == 0 or (bits[i >> 3] & (1 << (i & 7))) != 0

@numba.njit
def kernel(qty, qty_bits, unit_price, unit_bits, discount, discount_bits, delay):
    total = late_total = count = 0
    for i in range(qty.size):
        q = qty[i] if is_valid(qty_bits, i) else 0
        up_ok = is_valid(unit_bits, i) and unit_price[i] != 0
        up = unit_price[i] if up_ok else 0
        d = discount[i] if is_valid(discount_bits, i) else 0
        net = (int64(q) * int64(up)) * (10_000 - int64(d)) // 10_000
        if up_ok and q > 0:
            total += net; count += 1
            if delay[i] > 30: late_total += net
    return total, late_total, count"""

PYARROW_PROJECTION_CODE = """qty = pc.fill_null(data.arrow["qty"], 0)
unit_valid = pc.and_kleene(
    pc.is_valid(data.arrow["unit_price"]),
    pc.not_equal(data.arrow["unit_price"], 0),
)
unit_price = pc.if_else(unit_valid, data.arrow["unit_price"], 0)

gross = pc.multiply(
    pc.cast(data.arrow["qty_raw"], pa.int64()),
    pc.cast(data.arrow["unit_price_raw"], pa.int64()),
)
raw = pc.multiply(
    gross,
    pc.subtract(10_000, pc.cast(data.arrow["discount_bp_raw"], pa.int64())),
)
net = pc.divide(raw, 10_000)
base_valid = pc.and_kleene(unit_valid, pc.greater(qty, 0))

high_qty_out = pc.if_else(pc.and_kleene(base_valid, pc.greater_equal(qty, 25)), net, 0)
expensive_out = pc.if_else(
    pc.and_kleene(base_valid, pc.greater_equal(unit_price, 5_000)),
    net,
    0,
)
late_out = pc.if_else(pc.and_kleene(base_valid, pc.greater(data.arrow["delay"], 30)), 1, 0)"""

PYARROW_NULLABLE_REVENUE_CODE = """qty = pc.fill_null(data.arrow["qty"], 0)
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
count = pc.sum(pc.cast(valid, pa.int64())).as_py()"""

PYARROW_SPLIT_VALIDITY_CODE = """qty = pc.fill_null(data.arrow["qty"], 0)
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
late_count = pc.sum(pc.cast(late, pa.int64())).as_py()"""

PYARROW_UNCONDITIONAL_DIV_CODE = """gross = pc.multiply(
    pc.cast(data.arrow["qty_raw"], pa.int64()),
    pc.cast(data.arrow["unit_price_raw"], pa.int64()),
)
raw = pc.multiply(
    gross,
    pc.subtract(10_000, pc.cast(data.arrow["discount_bp_raw"], pa.int64())),
)
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
late_count = pc.sum(pc.cast(late, pa.int64())).as_py()"""

SIMJIT_PROJECTION_EXPRS_CODE = """qty_raw = sj.col("qty_raw", sj.I32)
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
late = base_valid & (delay > sj.i32(30))"""

PYTHON_WORKLOAD_SOURCES = {
    "conditional_store_projection": {
        "simjit": f"""{SIMJIT_PROJECTION_EXPRS_CODE}

program = sj.query(
    high_qty_out=sj.store(net, cond=high_qty),
    expensive_out=sj.store(net, cond=expensive),
    late_out=sj.store(sj.i64(1), cond=late),
)

kernel, outputs, sentinel = prepare_simjit_projection(
    data, program, "conditional_store_projection"
)""",
        "numba": """qty_raw, _ = arrow_int32_buffers(data, "qty_raw")
unit_price_raw, _ = arrow_int32_buffers(data, "unit_price_raw")
discount_raw, _ = arrow_int32_buffers(data, "discount_bp_raw")
qty, qty_bits = arrow_int32_buffers(data, "qty")
unit_price, unit_bits = arrow_int32_buffers(data, "unit_price")

@numba.njit
def kernel(qty_raw, unit_price_raw, discount_raw, qty, qty_bits, unit_price, unit_bits, delay):
    for i in range(qty.size):
        gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
        net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000
        q = qty[i] if is_valid(qty_bits, i) else 0
        unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
        base_valid = unit_valid and q > 0
        if base_valid and q >= 25:
            high_qty_out[i] = net""",
        "pyarrow": PYARROW_PROJECTION_CODE,
    },
    "select_store_projection": {
        "simjit": f"""{SIMJIT_PROJECTION_EXPRS_CODE}

program = sj.query(
    high_qty_out=high_qty.ifelse(net, sj.i64(0)),
    expensive_out=expensive.ifelse(net, sj.i64(0)),
    late_out=late.ifelse(sj.i64(1), sj.i64(0)),
)

kernel, outputs, sentinel = prepare_simjit_projection(
    data, program, "select_store_projection"
)""",
        "numba": """@numba.njit
def kernel(qty_raw, unit_price_raw, discount_raw, qty, qty_bits, unit_price, unit_bits, delay):
    for i in range(qty.size):
        gross = np.int64(qty_raw[i]) * np.int64(unit_price_raw[i])
        net = gross * (10_000 - np.int64(discount_raw[i])) // 10_000
        q = qty[i] if is_valid(qty_bits, i) else 0
        unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
        up = unit_price[i] if unit_valid else 0
        base_valid = unit_valid and q > 0
        high_qty_out[i] = net if base_valid and q >= 25 else 0
        expensive_out[i] = net if base_valid and up >= 5_000 else 0
        late_out[i] = 1 if base_valid and delay[i] > 30 else 0""",
        "pyarrow": PYARROW_PROJECTION_CODE,
    },
    "nullable_revenue_multiagg": {
        "simjit": SIMJIT_NULLABLE_CODE,
        "numba": NUMBA_BITMAP_CODE,
        "pyarrow": PYARROW_NULLABLE_REVENUE_CODE,
    },
    "split_valids_revenue": {
        "simjit": """qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
unit_price = sj.nullif(sj.col("unit_price", sj.I32), sj.i32(0))
unit_price_value = sj.coalesce(unit_price, sj.i32(0))
discount = sj.coalesce(sj.col("discount_bp", sj.I32), sj.i32(0))
delay = sj.col("delay", sj.I32)

gross = qty.cast(sj.I64) * unit_price_value.cast(sj.I64)
net = (gross * (sj.i64(10_000) - discount.cast(sj.I64))) / sj.i64(10_000)
base_valid = unit_price.is_not_null() & (qty > sj.i32(0))

program = sj.query(
    high_qty_net=net.sum(where=base_valid & (qty >= sj.i32(25))),
    expensive_net=net.sum(where=base_valid & (unit_price_value >= sj.i32(5_000))),
    late_count=(base_valid & (delay > sj.i32(30))).count(),
)""",
        "numba": """@numba.njit
def kernel(qty, qty_bits, unit_price, unit_bits, discount, discount_bits, delay):
    high_qty_total = 0
    expensive_total = 0
    late_count = 0
    for i in range(qty.size):
        q = qty[i] if is_valid(qty_bits, i) else 0
        unit_valid = is_valid(unit_bits, i) and unit_price[i] != 0
        up = unit_price[i] if unit_valid else 0
        d = discount[i] if is_valid(discount_bits, i) else 0
        net = np.int64(q) * np.int64(up) * (10_000 - np.int64(d)) // 10_000
        base_valid = unit_valid and q > 0
        if base_valid and q >= 25:
            high_qty_total += net
        if base_valid and up >= 5_000:
            expensive_total += net
        if base_valid and delay[i] > 30:
            late_count += 1""",
        "pyarrow": PYARROW_SPLIT_VALIDITY_CODE,
    },
    "unconditional_div_split_valids": {
        "simjit": """qty_raw = sj.col("qty_raw", sj.I32)
unit_price_raw = sj.col("unit_price_raw", sj.I32)
discount_raw = sj.col("discount_bp_raw", sj.I32)
qty = sj.coalesce(sj.col("qty", sj.I32), sj.i32(0))
unit_price = sj.nullif(sj.col("unit_price", sj.I32), sj.i32(0))
unit_price_value = sj.coalesce(unit_price, sj.i32(0))

gross = qty_raw.cast(sj.I64) * unit_price_raw.cast(sj.I64)
net = (gross * (sj.i64(10_000) - discount_raw.cast(sj.I64))) / sj.i64(10_000)
base_valid = unit_price.is_not_null() & (qty > sj.i32(0))

program = sj.query(
    high_qty_net=net.sum(where=base_valid & (qty >= sj.i32(25))),
    expensive_net=net.sum(where=base_valid & (unit_price_value >= sj.i32(5_000))),
    late_count=(base_valid & (sj.col("delay", sj.I32) > sj.i32(30))).count(),
)""",
        "numba": """@numba.njit
def kernel(qty_raw, unit_price_raw, discount_raw, qty, qty_bits, unit_price, unit_bits, delay):
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
            expensive_total += net""",
        "pyarrow": PYARROW_UNCONDITIONAL_DIV_CODE,
    },
}

CPP_ENGINE_CODE = """static void build_revenue_kernel(FunctionBuilder &b) {
    Value price = b.input_arg(I64);
    Value discount = b.input_arg(I32);
    Argument revenue_out = b.arg(I128);
    Value net = b.mul(price, b.sext(b.sub(b.i32(10000), discount), I64));
    b.sum(b.sext(net, I128), revenue_out);
}
auto kernel = vectorized_function<
    InputArr<I64>, InputArr<I32>, OutputScalar<I128>
>(ctx, "revenue_fragment", build_revenue_kernel);

auto price = std::static_pointer_cast<arrow::Int64Array>(batch->column(0));
auto discount = std::static_pointer_cast<arrow::Int32Array>(batch->column(1));

// raw_values() is Arrow's typed view of the values buffer: buffers[1].
const int64_t *price_data = price->raw_values();
const int32_t *discount_data = discount->raw_values();

__int128 revenue = 0;
kernel(batch->num_rows(), price_data, discount_data, &revenue);
aggregate_state.revenue += revenue;  // OutputScalar<I128> result"""

DUCKDB_ROWS = [
    {
        "id": "sum_add_i64",
        "label": "sum(a + b)",
        "note": "fused add aggregate",
        "sql": """SELECT sum(a + b)
FROM t;""",
        "simjit_sql": """SELECT simjit_sum_add_i64_nn(a, b)
FROM t;""",
        "builder": _duckdb_builder_source("BuildSumAddI64"),
        "x86": (21.0, 19.0, 1.11),
        "arm": (7.0, 8.0, 0.88),
    },
    {
        "id": "sum_net_i64",
        "label": "sum(net)",
        "note": "fused revenue aggregate",
        "sql": """SELECT sum(price * (10000 - discount_bp) // 10000)
FROM t;""",
        "simjit_sql": """SELECT simjit_sum_net_i64_nn(price, discount_bp)
FROM t;""",
        "builder": _duckdb_builder_source("BuildSumNetI64"),
        "x86": (61.0, 21.0, 2.90),
        "arm": (28.0, 11.0, 2.55),
    },
    {
        "id": "count_ship_filter",
        "label": "count(ship filter)",
        "note": "fused predicate count",
        "sql": """SELECT count(*)
FROM t
WHERE shipdate <= 10592
  AND quantity_i32 < 24;""",
        "simjit_sql": """SELECT simjit_count_ship_filter_i32_nn(shipdate, quantity_i32)
FROM t;""",
        "builder": _duckdb_builder_source("BuildCountShipFilter"),
        "x86": (31.0, 15.0, 2.07),
        "arm": (12.0, 5.0, 2.40),
    },
    {
        "id": "q1_multi",
        "label": "Q1-shaped multi-agg",
        "note": "shared expression, 2 sums + count",
        "sql": """SELECT
  sum(price * (10000 - discount_bp) // 10000) AS sum_net,
  sum(price * (10000 - discount_bp) * (10000 + tax_bp) // 100000000) AS sum_charge,
  count(*) AS row_count
FROM t;""",
        "simjit_sql": """SELECT (r).sum_net AS sum_net,
       (r).sum_charge AS sum_charge,
       (r).row_count AS row_count
FROM (
  SELECT simjit_q1_multi_i64(price, discount_bp, tax_bp) AS r
  FROM t
);""",
        "builder": _duckdb_builder_source("BuildQ1Multi"),
        "x86": (111.0, 32.0, 3.47),
        "arm": (54.0, 17.0, 3.18),
    },
    {
        "id": "q19_mask",
        "label": "Q19-shaped predicate",
        "note": "complex boolean DAG + revenue",
        "sql": """SELECT
  count(*) AS row_count,
  sum(price * (10000 - discount_bp) // 10000) AS revenue
FROM t
WHERE (
    brand = 12 AND container BETWEEN 1 AND 4
    AND quantity BETWEEN 1 AND 11 AND size BETWEEN 1 AND 5
  )
  OR (
    brand = 23 AND container BETWEEN 5 AND 8
    AND quantity BETWEEN 10 AND 20 AND size BETWEEN 1 AND 10
  )
  OR (
    brand = 34 AND container BETWEEN 9 AND 12
    AND quantity BETWEEN 20 AND 30 AND size BETWEEN 1 AND 15
  );""",
        "simjit_sql": """SELECT (r).row_count AS row_count,
       (r).revenue AS revenue
FROM (
  SELECT simjit_q19_mask_i64(brand, container, quantity, size, price, discount_bp) AS r
  FROM t
);""",
        "builder": _duckdb_builder_source("BuildQ19Mask", helpers=("Between",)),
        "x86": (374.0, 50.0, 7.48),
        "arm": (226.0, 36.0, 6.28),
    },
    {
        "id": "nullable_revenue_random_50",
        "label": "nullable revenue, random 50%",
        "note": "validity + filtered aggregates",
        "sql": """WITH x AS (
  SELECT coalesce(quantity, 0) AS quantity,
         nullif(price, 0) AS price,
         coalesce(discount_bp, 0) AS discount_bp,
         delay
  FROM nullable_random_50
), y AS (
  SELECT quantity, price, discount_bp, delay,
         price IS NOT NULL AND quantity > 0 AS valid,
         quantity * price * (10000 - discount_bp) // 10000 AS net
  FROM x
)
SELECT
  sum(net) FILTER (WHERE valid) AS sum_valid_net,
  sum(net) FILTER (WHERE valid AND delay > 30) AS sum_delay_net,
  count(*) FILTER (WHERE valid) AS count_valid
FROM y;""",
        "simjit_sql": """SELECT (r).sum_valid_net AS sum_valid_net,
       (r).sum_delay_net AS sum_delay_net,
       (r).count_valid AS count_valid
FROM (
  SELECT simjit_nullable_revenue_i64(quantity, price, discount_bp, delay) AS r
  FROM nullable_random_50
);""",
        "builder": _duckdb_builder_source("BuildNullableRevenue"),
        "x86": (405.0, 34.0, 11.91),
        "arm": (251.0, 28.0, 8.96),
    },
]


def demo_context() -> dict:
    python = _python_context()
    cpp = _cpp_context()
    return {
        "hero_title": "Simjit",
        "hero_description": (
            "A JIT compiler for columnar expressions: AVX‑512/NEON kernels, "
            "< 50 µs compile time."
        ),
        "pitch_lines": [
            "NumPy/Arrow column expressions for Python users; expression fragments for C++ and database engines.",
        ],
        "feature_chips": [
            "Application hosts",
            "Core library",
            "Vectorizer",
            "C++ emitter",
            "LLVM emitter",
            "AsmJit backend",
        ],
        "readme_code": PYTHON_README_CODE,
        "readme_numpy_code": README_NUMPY_CODE,
        "readme_pyarrow_compute_code": README_PYARROW_COMPUTE_CODE,
        "readme_simjit_api_code": README_SIMJIT_API_CODE,
        "cpp_engine_code": CPP_ENGINE_CODE,
        "benchmark_environment": BENCHMARK_ENVIRONMENT,
        "python_benchmark_environment": PYTHON_BENCHMARK_ENVIRONMENT,
        "duckdb_benchmark_environment": DUCKDB_BENCHMARK_ENVIRONMENT,
        "project_links": PROJECT_LINKS,
        "python_panel": python,
        "cpp_panel": cpp,
        "scope_panel": _scope_context(),
    }


def _python_context() -> dict:
    baseline_specs = [
        {
            "id": "numba",
            "label": "vs Numba",
            "axis_label": "Numba",
            "library": "numba_arrow_bits",
            "color_scale": "speedup_log",
            "color_low": PYTHON_SPEEDUP_LOSS_LOW,
            "color_high": PYTHON_NUMBA_SPEEDUP_HIGH,
        },
        {
            "id": "pyarrow",
            "label": "vs pyarrow.compute",
            "axis_label": "pyarrow.compute",
            "library": "pyarrow_compute",
            "color_scale": "speedup_log",
            "color_low": PYTHON_SPEEDUP_LOSS_LOW,
            "color_high": PYTHON_PYARROW_SPEEDUP_HIGH,
        },
    ]
    by_key = {
        (workload, rows, library): {
            "first_seconds": first_seconds,
            "hot_rows_per_second": hot_rows_per_second,
        }
        for workload, rows, library, first_seconds, hot_rows_per_second in PYTHON_RECORDS
    }
    cells = []
    for workload, label in PYTHON_WORKLOADS:
        row_cells = []
        for rows in PYTHON_ROWS:
            simjit = by_key[(workload, rows, "simjit_arrow")]
            simjit_rps = simjit["hot_rows_per_second"]
            simjit_time = rows / simjit_rps
            baseline_cells = {}
            for spec in baseline_specs:
                baseline = by_key[(workload, rows, spec["library"])]
                speedup = simjit_rps / baseline["hot_rows_per_second"]
                baseline_time = rows / baseline["hot_rows_per_second"]
                baseline_cells[spec["id"]] = {
                    "rows": _rows_label(rows),
                    "speedup": speedup,
                    "speedup_label": _ratio_label(speedup),
                    "time_label": (
                        f"{_python_time_label(baseline_time)} to "
                        f"{_python_time_label(simjit_time)}"
                    ),
                    "speed_class": _speed_bucket(
                        speedup,
                        spec["color_low"],
                        spec["color_high"],
                        spec["color_scale"],
                    ),
                }
            row_cells.append({**baseline_cells["numba"], "baselines": baseline_cells})
        sources = PYTHON_WORKLOAD_SOURCES[workload]
        cells.append(
            {
                "id": workload,
                "label": label,
                "simjit_source": sources["simjit"],
                "numba_source": sources["numba"],
                "pyarrow_source": sources["pyarrow"],
                "cells": row_cells,
            }
        )

    return {
        "baselines": baseline_specs,
        "row_labels": [
            {
                "label": _rows_label(rows),
            }
            for rows in PYTHON_ROWS
        ],
        "heatmap": cells,
        "use_guidance": [
            {
                "title": "Use when",
                "text": "You have reusable column logic over NumPy or nullable Arrow data: ETL transforms, feature engineering, and interactive dataframe-style transforms.",
            },
            {
                "title": "Not when",
                "text": "The work is a tiny one-off array, arbitrary Python object/control-flow code, or whole-query planning. Simjit is a column-expression compiler.",
            },
        ],
    }


def _cpp_context() -> dict:
    rendered_rows = []
    for row in DUCKDB_ROWS:
        platforms = []
        for platform in ["x86", "arm"]:
            native_ms, simjit_ms, speedup = row[platform]
            platforms.append(
                {
                    "name": platform.upper(),
                    "speedup": speedup,
                    "speedup_label": f"{speedup:.2f}x",
                    "timing_label": f"{native_ms:.0f} ms to {simjit_ms:.0f} ms",
                    "speed_class": _bucket(
                        speedup, SPEEDUP_COLOR_LOW, SPEEDUP_COLOR_HIGH
                    ),
                }
            )
        rendered_rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "note": row["note"],
                "sql": row["sql"],
                "simjit_sql": row["simjit_sql"],
                "builder": row["builder"],
                "platforms": platforms,
            }
        )

    return {
        "engine_benchmark_environment": ENGINE_BENCHMARK_ENVIRONMENT,
        "engine_comparisons": [
            {
                "id": "arithmetic_projection",
                "name": "Arithmetic projection",
                "workload": "price * (1.0 - discount), 4096 rows",
                "arrow_compute": "6,769 ns",
                "gandiva": "1,878 ns",
                "simjit": "1,329 ns",
                "note": "Compile time: Gandiva about 20 ms, Simjit about 26 µs.",
                "sources": {
                    "arrow_compute": _engine_bench_function_source("BM_Arrow_Revenue"),
                    "gandiva": _engine_bench_function_source("BM_Gandiva_Revenue"),
                    "simjit": _engine_bench_sources(
                        "simjit_revenue_builder", "BM_simjit_Revenue"
                    ),
                },
                "bars": [
                    {
                        "label": "Arrow Compute",
                        "time": "6,769 ns",
                        "throughput": "1.0x",
                        "width": 20,
                        "kind": "arrow",
                    },
                    {
                        "label": "Gandiva",
                        "time": "1,878 ns",
                        "throughput": "3.6x",
                        "width": 71,
                        "kind": "gandiva",
                    },
                    {
                        "label": "Simjit",
                        "time": "1,329 ns",
                        "throughput": "5.1x",
                        "width": 100,
                        "kind": "simjit",
                    },
                ],
            },
            {
                "id": "logical_filter",
                "name": "Logical filter",
                "workload": "shipdate <= date AND quantity < 24, 4096 rows",
                "arrow_compute": "8,779 ns",
                "gandiva": "8,251 ns",
                "simjit": "247 ns",
                "note": "Bitmask output, matching vectorized-engine selection shape.",
                "info_title": "Why this outlier?",
                "info": [
                    "The generic Arrow kernel shape is hard for a compiler to auto-vectorize.",
                    "This filter benchmark shows the bitmap path that originally motivated Simjit.",
                ],
                "sources": {
                    "arrow_compute": _engine_bench_function_source(
                        "BM_Arrow_Filter_DateQty"
                    ),
                    "gandiva": _engine_bench_function_source(
                        "BM_Gandiva_Filter_DateQty"
                    ),
                    "simjit": _engine_bench_sources(
                        "simjit_Filter_DateQty_builder", "BM_simjit_DateQty"
                    ),
                },
                "bars": [
                    {
                        "label": "Arrow Compute",
                        "time": "8,779 ns",
                        "throughput": "1.0x",
                        "width": 3,
                        "kind": "arrow",
                    },
                    {
                        "label": "Gandiva",
                        "time": "8,251 ns",
                        "throughput": "1.1x",
                        "width": 3,
                        "kind": "gandiva",
                    },
                    {
                        "label": "Simjit",
                        "time": "247 ns",
                        "throughput": "35.5x",
                        "width": 100,
                        "kind": "simjit",
                    },
                ],
            },
        ],
        "rows": rendered_rows,
        "duckdb_caveat": (
            "The POC uses hand-written extension functions; the intended integration is lowering DuckDB "
            "expression trees into Simjit expression builders."
        ),
        "fusion_evidence": {
            "title": "Primitive loops collapse into one vector loop",
            "text": (
                "Simjit removes temporaries from expression processing by computing the whole expression "
                "inside one JIT-compiled vectorized primitive. The main benefit is less memory I/O: "
                "intermediate values stay in registers instead of being written to temporary columns. SIMD "
                "instructions further speed up the primitive and make compact data layouts, such as "
                "bitpacked validity masks, cheaper to process."
            ),
            "expression": "sum(price * (1 - discount))",
        },
    }


def _scope_context() -> dict:
    return {
        "groups": [
            {
                "name": "Feature set",
                "items": [
                    "x86-64 AVX-512 and ARMv8 NEON targets, scalar and vector codegen paths.",
                    "Arithmetic, logical, compare, select, cast, conditional-store, and floating-point kernels, with peephole optimizations and i32/i64 overflow checks.",
                    "Scalar and grouped aggregations, filtered variants, and conditional vector compaction (pack).",
                    "Nullable expressions: coalesce, nullif, is_valid, is_not_null — validity is part of the expression DAG.",
                ],
            },
            {
                "name": "Engine integration",
                "items": [
                    "C++ builder API is the primary interface; Python lowers NumPy and Arrow expressions into it.",
                    "AsmJit runtime backend; LLVM IR and C++ emitters for inspection and debugging.",
                    "Kernels bind to raw pointers and row counts; the host keeps storage, validity, scheduling, and aggregation state.",
                    "Expressions that can't vectorize fall back to scalar codegen.",
                ],
            },
        ]
    }


def _bucket(value: float, low: float, high: float) -> int:
    if high <= low:
        return 0
    ratio = (value - low) / (high - low)
    return max(0, min(10, round(ratio * 10)))


def _log_bucket(value: float, low: float, high: float) -> int:
    if value <= 0 or low <= 0 or high <= low:
        return 0
    ratio = math.log(value / low) / math.log(high / low)
    return max(0, min(10, math.floor(ratio * 10)))


def _speedup_log_bucket(value: float, loss_low: float, gain_high: float) -> int:
    if value <= 0:
        return 0
    if value < 1.0:
        if loss_low >= 1.0:
            return 4
        ratio = (value - loss_low) / (1.0 - loss_low)
        return max(0, min(4, round(ratio * 4)))
    if gain_high <= 1.0:
        return 4
    ratio = math.log(value) / math.log(gain_high)
    return max(4, min(10, 4 + round(ratio * 6)))


def _speed_bucket(value: float, low: float, high: float, scale: str) -> int:
    if scale == "speedup_log":
        return _speedup_log_bucket(value, low, high)
    if scale == "log":
        return _log_bucket(value, low, high)
    return _bucket(value, low, high)


def _percent(value: float, maximum: float) -> float:
    if maximum <= 0:
        return 0.0
    return round(max(0.0, min(100.0, value / maximum * 100.0)), 2)


def _rows_label(rows: int) -> str:
    if rows >= 1_000_000:
        return f"{rows / 1_000_000:.1f}M"
    if rows >= 1000:
        return f"{rows // 1000}K"
    return str(rows)


def _python_time_label(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    return f"{seconds * 1000:.2f} ms"


def _ratio_label(value: float) -> str:
    return f"{value:.1f}x"
