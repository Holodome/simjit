// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include <benchmark/benchmark.h>

#include "simjit/asmjit.h"
#include "simjit/compiler.h"
#include "simjit/jit.h"

#include <arrow/api.h>
#include <arrow/compute/api.h>
#include <gandiva/configuration.h>
#include <gandiva/filter.h>
#include <gandiva/projector.h>
#include <gandiva/selection_vector.h>
#include <gandiva/tree_expr_builder.h>

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace simjit {

static void *compile_asmjit_from_hir(hir::Function *hir, AsmjitSession &session, jit::CompilePolicy policy) {
    mir::Function *mir = nullptr;
    switch (policy) {
    case jit::CompilePolicy::Vectorized: mir = lower_vectorized(hir); break;
    case jit::CompilePolicy::Scalar: mir = lower_scalar(hir); break;
    case jit::CompilePolicy::BestEffort: {
        auto vectorized = try_lower_vectorized(hir);
        mir = vectorized ? vectorized.value() : lower_scalar(hir);
        break;
    }
    }
    AsmjitCompileResult result{};
    compile_asmjit(mir, AsmjitCompileOptions{false, false, &session}, result);
    return session.add_compiled_function();
}

} // namespace simjit

namespace {

using simjit::types::I32;
using simjit::types::I64;

enum class Backend {
    Arrow,
    Gandiva,
    SimjitScalar,
    SimjitSimd
};
enum class Workload {
    PricingProjection,
    BoundedOrderScore,
    SharedArithmeticDag,
    ConditionalPricing,
    RangeProjection,
    RevenueSelection,
    TpchQ6Revenue,
    PricingSummary,
    SegmentedLineTotals,
    StatisticalMoments,
};

constexpr Workload Workloads[] = {
    Workload::PricingProjection,  Workload::BoundedOrderScore, Workload::SharedArithmeticDag,
    Workload::ConditionalPricing, Workload::RangeProjection,   Workload::RevenueSelection,
    Workload::TpchQ6Revenue,      Workload::PricingSummary,    Workload::SegmentedLineTotals,
    Workload::StatisticalMoments,
};
constexpr Backend Backends[] = {Backend::Arrow, Backend::Gandiva, Backend::SimjitScalar, Backend::SimjitSimd};
int64_t g_rows = 4096;

const char *show(Backend backend) {
    switch (backend) {
    case Backend::Arrow: return "arrow_compute";
    case Backend::Gandiva: return "gandiva";
    case Backend::SimjitScalar: return "simjit_scalar";
    case Backend::SimjitSimd: return "simjit_simd";
    }
    return "unknown";
}

const char *show(Workload workload) {
    switch (workload) {
    case Workload::PricingProjection: return "pricing_projection";
    case Workload::BoundedOrderScore: return "bounded_order_score";
    case Workload::SharedArithmeticDag: return "shared_arithmetic_dag";
    case Workload::ConditionalPricing: return "conditional_pricing";
    case Workload::RangeProjection: return "range_projection";
    case Workload::RevenueSelection: return "revenue_selection";
    case Workload::TpchQ6Revenue: return "tpch_q6_revenue";
    case Workload::PricingSummary: return "pricing_summary";
    case Workload::SegmentedLineTotals: return "segmented_line_totals";
    case Workload::StatisticalMoments: return "statistical_moments";
    }
    return "unknown";
}

bool supported(Backend backend, Workload workload) {
#if defined(__aarch64__)
    // The canonical corpus is x86-64 AVX-512. Keep host execution tests
    // separate from the explicit x86 lowering check below.
    if (backend == Backend::SimjitSimd) return false;
#endif
    return backend != Backend::Gandiva || workload <= Workload::RevenueSelection;
}

bool is_projection(Workload workload) {
    return workload <= Workload::RangeProjection;
}
bool is_selection(Workload workload) {
    return workload == Workload::RevenueSelection;
}

void check_arrow(const arrow::Status &status) {
    if (!status.ok()) throw std::runtime_error(status.ToString());
}

template <typename T> T unwrap(arrow::Result<T> result) {
    if (!result.ok()) throw std::runtime_error(result.status().ToString());
    return std::move(result).ValueOrDie();
}

arrow::Datum call_arrow(std::string_view name, std::vector<arrow::Datum> args) {
    return unwrap(arrow::compute::CallFunction(std::string(name), std::move(args)));
}

template <typename Builder, typename ValueFn> std::shared_ptr<arrow::Array> make_array(int64_t rows, ValueFn value) {
    Builder builder;
    check_arrow(builder.Reserve(rows));
    for (int64_t i = 0; i < rows; ++i)
        check_arrow(builder.Append(value(i)));
    std::shared_ptr<arrow::Array> result;
    check_arrow(builder.Finish(&result));
    return result;
}

struct Data {
    int64_t rows = 0;
    std::shared_ptr<arrow::Schema> schema;
    std::shared_ptr<arrow::RecordBatch> batch;
    std::vector<std::shared_ptr<arrow::Int64Array>> q;
    std::vector<std::shared_ptr<arrow::Int32Array>> i;
};

enum QCol {
    Price,
    Cost,
    Penalty,
    Lower,
    Upper,
    Target,
    X,
    Y,
    Z,
    Open,
    Close,
    Limit,
    X0,
    Y0
};
enum ICol {
    Discount,
    Tax,
    Quantity,
    Shipdate,
    DiscountBp,
    TaxBp,
    Delay
};

std::shared_ptr<Data> make_data(int64_t rows, std::string_view selection_shape = "normal") {
    auto data = std::make_shared<Data>();
    data->rows = rows;
    auto q = [&](auto fn) {
        return std::static_pointer_cast<arrow::Int64Array>(make_array<arrow::Int64Builder>(rows, fn));
    };
    auto i = [&](auto fn) {
        return std::static_pointer_cast<arrow::Int32Array>(make_array<arrow::Int32Builder>(rows, fn));
    };
    data->q = {
        q([=](int64_t n) { return selection_shape == "full" ? INT64_C(100000) : INT64_C(10000) + n % 1000 * 110; }),
        q([=](int64_t n) { return selection_shape == "full" ? INT64_C(0) : n % 700 * 9; }),
        q([](int64_t n) { return n % 13 * 25; }),
        q([](int64_t) { return INT64_C(-25000); }),
        q([](int64_t) { return INT64_C(2500000); }),
        q([](int64_t n) { return n % 500 * 125; }),
        q([](int64_t n) { return n % 97 * 4; }),
        q([](int64_t n) { return n % 89 * 2; }),
        q([](int64_t n) { return n % 17 + 8; }),
        q([](int64_t n) { return n % 251 * 110; }),
        q([](int64_t n) { return (n * 7) % 251 * 105; }),
        q([](int64_t n) { return n % 23 * 100 + 200; }),
        q([](int64_t) { return INT64_C(48); }),
        q([](int64_t) { return INT64_C(32); }),
    };
    data->i = {
        i([=](int64_t n) { return selection_shape == "full" ? 600 : int32_t(n % 10 * 100); }),
        i([](int64_t n) { return int32_t(n % 7 * 100); }),
        i([=](int64_t n) {
            if (selection_shape == "empty") return 50;
            if (selection_shape == "full") return 1;
            return int32_t(n % 50);
        }),
        i([=](int64_t n) { return selection_shape == "empty" ? 19950101 : int32_t(19940101 + n % 730); }),
        i([](int64_t n) { return int32_t((n * 17) % 2000); }),
        i([](int64_t n) { return int32_t((n * 11) % 1000); }),
        i([](int64_t n) { return int32_t((n * 5) % 60); }),
    };
    const char *qnames[] = {"price", "cost", "penalty", "lower", "upper", "target", "x",
                            "y",     "z",    "open",    "close", "limit", "x0",     "y0"};
    const char *inames[] = {"discount", "tax", "quantity", "shipdate", "discount_bp", "tax_bp", "delay"};
    arrow::FieldVector fields;
    arrow::ArrayVector arrays;
    for (size_t n = 0; n < data->q.size(); ++n) {
        fields.push_back(arrow::field(qnames[n], arrow::int64(), false));
        arrays.push_back(data->q[n]);
    }
    for (size_t n = 0; n < data->i.size(); ++n) {
        fields.push_back(arrow::field(inames[n], arrow::int32(), false));
        arrays.push_back(data->i[n]);
    }
    data->schema = arrow::schema(fields);
    data->batch = arrow::RecordBatch::Make(data->schema, rows, arrays);
    return data;
}

int64_t qv(const Data &data, QCol col, int64_t row) {
    return data.q[size_t(col)]->Value(row);
}
int32_t iv(const Data &data, ICol col, int64_t row) {
    return data.i[size_t(col)]->Value(row);
}

struct Result {
    std::vector<std::vector<int64_t>> arrays;
    std::vector<int64_t> scalars;
    std::vector<int64_t> indices;
};

bool selected(const Data &d, int64_t row, int64_t margin) {
    return iv(d, Shipdate, row) >= 19940101 && iv(d, Shipdate, row) <= 19940501 && iv(d, Discount, row) >= 500 &&
           iv(d, Discount, row) <= 700 && iv(d, Quantity, row) < 24 && margin > 5000;
}

Result reference(const Data &d, Workload workload) {
    Result out;
    const size_t outputs[] = {2, 2, 3, 4, 4};
    if (is_projection(workload)) out.arrays.assign(outputs[size_t(workload)], std::vector<int64_t>(size_t(d.rows)));
    if (workload == Workload::TpchQ6Revenue) out.scalars.assign(1, 0);
    if (workload == Workload::PricingSummary) {
        out.scalars = {0, 0, 0, 0, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()};
    }
    if (workload == Workload::SegmentedLineTotals)
        out.scalars = {0, 0, 0, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()};
    if (workload == Workload::StatisticalMoments)
        out.scalars = {0, 0, 0, 0, 0, std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min()};
    for (int64_t row = 0; row < d.rows; ++row) {
        int64_t price = qv(d, Price, row), discount = iv(d, Discount, row), tax = iv(d, Tax, row);
        int64_t revenue = price * (10000 - discount) / 10000;
        int64_t charge = revenue * (10000 + tax) / 10000, margin = charge - qv(d, Cost, row);
        if (workload == Workload::PricingProjection) {
            out.arrays[0][row] = revenue;
            out.arrays[1][row] = charge;
        } else if (workload == Workload::BoundedOrderScore) {
            int64_t adjusted = price * int64_t(iv(d, Quantity, row)) * (10000 - discount) / 10000 - qv(d, Penalty, row);
            out.arrays[0][row] = std::min(std::max(adjusted, qv(d, Lower, row)), qv(d, Upper, row));
            out.arrays[1][row] = std::abs(adjusted - qv(d, Target, row));
        } else if (workload == Workload::SharedArithmeticDag) {
            int64_t x = qv(d, X, row), y = qv(d, Y, row), base = (x + y) * qv(d, Z, row), adjusted = base - x * y;
            out.arrays[0][row] = base + adjusted;
            out.arrays[1][row] = std::min(base, adjusted);
            out.arrays[2][row] = std::max(base, adjusted);
        } else if (workload == Workload::ConditionalPricing) {
            out.arrays[0][row] = margin * (iv(d, Quantity, row) >= 25 ? 110 : 90) / 100;
            out.arrays[1][row] = std::max(margin, INT64_C(0));
            out.arrays[2][row] = std::min(charge, price);
            out.arrays[3][row] = std::abs(margin);
        } else if (workload == Workload::RangeProjection) {
            int64_t open = qv(d, Open, row), close = qv(d, Close, row), lo = std::min(open, close),
                    hi = std::max(open, close);
            out.arrays[0][row] = lo;
            out.arrays[1][row] = hi;
            out.arrays[2][row] = hi - lo;
            out.arrays[3][row] = std::min(std::abs(close - open), qv(d, Limit, row));
        } else if (workload == Workload::RevenueSelection) {
            if (selected(d, row, margin)) out.indices.push_back(row);
        } else if (workload == Workload::TpchQ6Revenue) {
            if (selected(d, row, margin)) out.scalars[0] += revenue;
        } else if (workload == Workload::PricingSummary) {
            out.scalars[0] += int64_t(iv(d, Quantity, row));
            out.scalars[1] += price;
            out.scalars[2] += revenue;
            out.scalars[3] += charge;
            out.scalars[4] = std::min(out.scalars[4], charge);
            out.scalars[5] = std::max(out.scalars[5], charge);
        } else if (workload == Workload::SegmentedLineTotals) {
            int64_t net = int64_t(iv(d, Quantity, row)) * price * (10000 - int64_t(iv(d, DiscountBp, row))) / 10000;
            int64_t line = net * (10000 + int64_t(iv(d, TaxBp, row))) / 10000;
            if (iv(d, Delay, row) > 30) out.scalars[0] += line;
            if (iv(d, Quantity, row) >= 25) out.scalars[1] += line;
            if (line > 1000000) out.scalars[2] += line;
            out.scalars[3] = std::min(out.scalars[3], line);
            out.scalars[4] = std::max(out.scalars[4], line);
        } else {
            int64_t cx = qv(d, X, row) - qv(d, X0, row), cy = qv(d, Y, row) - qv(d, Y0, row), xy = cx * cy;
            out.scalars[0] += cx;
            out.scalars[1] += cy;
            out.scalars[2] += cx * cx;
            out.scalars[3] += xy;
            out.scalars[4] += cy * cy;
            out.scalars[5] = std::min(out.scalars[5], xy);
            out.scalars[6] = std::max(out.scalars[6], xy);
        }
    }
    return out;
}

uint64_t hash_step(uint64_t hash, uint64_t value) {
    return (hash ^ value) * UINT64_C(1099511628211);
}
uint64_t input_checksum(const Data &data) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const auto &column : data.q)
        for (int64_t row = 0; row < data.rows; ++row)
            hash = hash_step(hash, uint64_t(column->Value(row)));
    for (const auto &column : data.i)
        for (int64_t row = 0; row < data.rows; ++row)
            hash = hash_step(hash, uint64_t(int64_t(column->Value(row))));
    return hash;
}
uint64_t result_checksum(const Result &result) {
    uint64_t hash = UINT64_C(1469598103934665603);
    for (const auto &array : result.arrays)
        for (int64_t value : array)
            hash = hash_step(hash, uint64_t(value));
    for (int64_t value : result.scalars)
        hash = hash_step(hash, uint64_t(value));
    for (int64_t value : result.indices)
        hash = hash_step(hash, uint64_t(value));
    return hash;
}

void verify_result(const Result &actual, const Result &expected, Workload workload) {
    if (actual.arrays != expected.arrays || actual.scalars != expected.scalars || actual.indices != expected.indices)
        throw std::runtime_error(std::string(show(workload)) + " result shape/index mismatch");
}

struct Prepared {
    std::function<void()> run;
    std::function<Result()> result;
    std::function<void()> release;
    Result expected;
    int64_t result_count = 0;
};

uint64_t validate(Prepared &prepared, Workload workload) {
    prepared.release();
    prepared.run();
    Result actual = prepared.result();
    verify_result(actual, prepared.expected, workload);
    prepared.release();
    return result_checksum(prepared.expected);
}

arrow::Datum qdatum(const Data &d, QCol col) {
    return arrow::Datum(d.q[size_t(col)]);
}
arrow::Datum idatum(const Data &d, ICol col) {
    return arrow::Datum(d.i[size_t(col)]);
}
arrow::Datum qs(int64_t value) {
    return arrow::Datum(std::make_shared<arrow::Int64Scalar>(value));
}
arrow::Datum is(int32_t value) {
    return arrow::Datum(value);
}
arrow::Datum amin(arrow::Datum a, arrow::Datum b) {
    return call_arrow("if_else", {call_arrow("less", {a, b}), a, b});
}
arrow::Datum amax(arrow::Datum a, arrow::Datum b) {
    return call_arrow("if_else", {call_arrow("greater", {a, b}), a, b});
}
arrow::Datum aabs(arrow::Datum a) {
    return call_arrow("if_else", {call_arrow("less", {a, qs(0)}), call_arrow("subtract", {qs(0), a}), a});
}

std::pair<arrow::Datum, arrow::Datum> arrow_pricing(const Data &d) {
    auto discount = unwrap(arrow::compute::Cast(idatum(d, Discount), arrow::int64()));
    auto tax = unwrap(arrow::compute::Cast(idatum(d, Tax), arrow::int64()));
    auto revenue =
        call_arrow("divide", {call_arrow("multiply", {qdatum(d, Price), call_arrow("subtract", {qs(10000), discount})}),
                              qs(10000)});
    return {revenue,
            call_arrow("divide", {call_arrow("multiply", {revenue, call_arrow("add", {qs(10000), tax})}), qs(10000)})};
}

arrow::Datum arrow_predicate(const Data &d, arrow::Datum margin) {
    std::vector<arrow::Datum> terms = {call_arrow("greater_equal", {idatum(d, Shipdate), is(19940101)}),
                                       call_arrow("less_equal", {idatum(d, Shipdate), is(19940501)}),
                                       call_arrow("greater_equal", {idatum(d, Discount), is(500)}),
                                       call_arrow("less_equal", {idatum(d, Discount), is(700)}),
                                       call_arrow("less", {idatum(d, Quantity), is(24)}),
                                       call_arrow("greater", {margin, qs(5000)})};
    auto result = terms[0];
    for (size_t n = 1; n < terms.size(); ++n)
        result = call_arrow("and_kleene", {result, terms[n]});
    return result;
}

int64_t arrow_scalar_i64(const arrow::Datum &datum) {
    auto scalar = datum.scalar();
    switch (scalar->type->id()) {
    case arrow::Type::INT32: return std::static_pointer_cast<arrow::Int32Scalar>(scalar)->value;
    case arrow::Type::INT64: return std::static_pointer_cast<arrow::Int64Scalar>(scalar)->value;
    default: throw std::runtime_error("unexpected Arrow integer aggregate scalar type");
    }
}

std::pair<int64_t, int64_t> arrow_minmax(const arrow::Datum &values) {
    auto scalar = std::static_pointer_cast<arrow::StructScalar>(unwrap(arrow::compute::MinMax(values)).scalar());
    return {std::static_pointer_cast<arrow::Int64Scalar>(scalar->value[0])->value,
            std::static_pointer_cast<arrow::Int64Scalar>(scalar->value[1])->value};
}

Result arrow_to_result(const std::vector<arrow::Datum> &values, Workload workload) {
    Result result;
    if (is_projection(workload)) {
        for (const auto &value : values) {
            auto array = std::static_pointer_cast<arrow::Int64Array>(value.make_array());
            result.arrays.emplace_back(array->raw_values(), array->raw_values() + array->length());
        }
    } else if (is_selection(workload)) {
        auto array = std::static_pointer_cast<arrow::UInt64Array>(values[0].make_array());
        for (int64_t n = 0; n < array->length(); ++n)
            result.indices.push_back(int64_t(array->Value(n)));
    } else {
        for (const auto &value : values)
            result.scalars.push_back(arrow_scalar_i64(value));
    }
    return result;
}

Prepared prepare_arrow(const std::shared_ptr<Data> &data, Workload workload) {
    auto output = std::make_shared<std::vector<arrow::Datum>>();
    auto run = [data, workload, output] {
        auto i64 = [&](ICol col) { return unwrap(arrow::compute::Cast(idatum(*data, col), arrow::int64())); };
        if (workload == Workload::PricingProjection) {
            auto [revenue, charge] = arrow_pricing(*data);
            *output = {revenue, charge};
        } else if (workload == Workload::BoundedOrderScore) {
            auto numerator = call_arrow("multiply", {call_arrow("multiply", {qdatum(*data, Price), i64(Quantity)}),
                                                     call_arrow("subtract", {qs(10000), i64(Discount)})});
            auto adjusted =
                call_arrow("subtract", {call_arrow("divide", {numerator, qs(10000)}), qdatum(*data, Penalty)});
            *output = {amin(amax(adjusted, qdatum(*data, Lower)), qdatum(*data, Upper)),
                       aabs(call_arrow("subtract", {adjusted, qdatum(*data, Target)}))};
        } else if (workload == Workload::SharedArithmeticDag) {
            auto base =
                call_arrow("multiply", {call_arrow("add", {qdatum(*data, X), qdatum(*data, Y)}), qdatum(*data, Z)});
            auto adjusted =
                call_arrow("subtract", {base, call_arrow("multiply", {qdatum(*data, X), qdatum(*data, Y)})});
            *output = {call_arrow("add", {base, adjusted}), amin(base, adjusted), amax(base, adjusted)};
        } else if (workload == Workload::ConditionalPricing) {
            auto [revenue, charge] = arrow_pricing(*data);
            auto margin = call_arrow("subtract", {charge, qdatum(*data, Cost)});
            auto adjusted =
                call_arrow("if_else", {call_arrow("greater_equal", {idatum(*data, Quantity), is(25)}),
                                       call_arrow("divide", {call_arrow("multiply", {margin, qs(110)}), qs(100)}),
                                       call_arrow("divide", {call_arrow("multiply", {margin, qs(90)}), qs(100)})});
            *output = {adjusted, amax(margin, qs(0)), amin(charge, qdatum(*data, Price)), aabs(margin)};
        } else if (workload == Workload::RangeProjection) {
            auto lo = amin(qdatum(*data, Open), qdatum(*data, Close)),
                 hi = amax(qdatum(*data, Open), qdatum(*data, Close));
            *output = {
                lo, hi, call_arrow("subtract", {hi, lo}),
                amin(aabs(call_arrow("subtract", {qdatum(*data, Close), qdatum(*data, Open)})), qdatum(*data, Limit))};
        } else if (workload == Workload::RevenueSelection) {
            auto [revenue, charge] = arrow_pricing(*data);
            auto margin = call_arrow("subtract", {charge, qdatum(*data, Cost)});
            output->push_back(call_arrow("indices_nonzero", {arrow_predicate(*data, margin)}));
        } else if (workload == Workload::TpchQ6Revenue) {
            auto [revenue, charge] = arrow_pricing(*data);
            auto margin = call_arrow("subtract", {charge, qdatum(*data, Cost)});
            auto filtered = call_arrow("filter", {revenue, arrow_predicate(*data, margin)});
            output->push_back(unwrap(arrow::compute::Sum(filtered)));
        } else if (workload == Workload::PricingSummary) {
            auto [revenue, charge] = arrow_pricing(*data);
            auto minmax = arrow_minmax(charge);
            *output = {unwrap(arrow::compute::Sum(idatum(*data, Quantity))),
                       unwrap(arrow::compute::Sum(qdatum(*data, Price))),
                       unwrap(arrow::compute::Sum(revenue)),
                       unwrap(arrow::compute::Sum(charge)),
                       qs(minmax.first),
                       qs(minmax.second)};
        } else if (workload == Workload::SegmentedLineTotals) {
            auto net = call_arrow(
                "divide", {call_arrow("multiply", {call_arrow("multiply", {i64(Quantity), qdatum(*data, Price)}),
                                                   call_arrow("subtract", {qs(10000), i64(DiscountBp)})}),
                           qs(10000)});
            auto line = call_arrow(
                "divide", {call_arrow("multiply", {net, call_arrow("add", {qs(10000), i64(TaxBp)})}), qs(10000)});
            auto sum_if = [&](arrow::Datum pred) {
                return unwrap(arrow::compute::Sum(call_arrow("filter", {line, pred})));
            };
            auto minmax = arrow_minmax(line);
            *output = {sum_if(call_arrow("greater", {idatum(*data, Delay), is(30)})),
                       sum_if(call_arrow("greater_equal", {idatum(*data, Quantity), is(25)})),
                       sum_if(call_arrow("greater", {line, qs(1000000)})), qs(minmax.first), qs(minmax.second)};
        } else {
            auto cx = call_arrow("subtract", {qdatum(*data, X), qdatum(*data, X0)}),
                 cy = call_arrow("subtract", {qdatum(*data, Y), qdatum(*data, Y0)});
            auto cross = call_arrow("multiply", {cx, cy});
            auto minmax = arrow_minmax(cross);
            *output = {unwrap(arrow::compute::Sum(cx)),
                       unwrap(arrow::compute::Sum(cy)),
                       unwrap(arrow::compute::Sum(call_arrow("multiply", {cx, cx}))),
                       unwrap(arrow::compute::Sum(cross)),
                       unwrap(arrow::compute::Sum(call_arrow("multiply", {cy, cy}))),
                       qs(minmax.first),
                       qs(minmax.second)};
        }
    };
    Result expected = reference(*data, workload);
    return {run, [output, workload] { return arrow_to_result(*output, workload); }, [output] { output->clear(); },
            expected, is_selection(workload) ? int64_t(expected.indices.size()) : data->rows};
}

gandiva::NodePtr gcall(const char *name, gandiva::NodeVector args, std::shared_ptr<arrow::DataType> type) {
    return gandiva::TreeExprBuilder::MakeFunction(name, std::move(args), std::move(type));
}
gandiva::NodePtr gq(std::string_view name) {
    return gandiva::TreeExprBuilder::MakeField(arrow::field(std::string(name), arrow::int64(), false));
}
gandiva::NodePtr gi(std::string_view name) {
    return gandiva::TreeExprBuilder::MakeField(arrow::field(std::string(name), arrow::int32(), false));
}
gandiva::NodePtr gql(int64_t value) {
    return gandiva::TreeExprBuilder::MakeLiteral(value);
}
gandiva::NodePtr gil(int32_t value) {
    return gandiva::TreeExprBuilder::MakeLiteral(value);
}
gandiva::NodePtr ga(gandiva::NodePtr a, gandiva::NodePtr b) {
    return gcall("add", {a, b}, arrow::int64());
}
gandiva::NodePtr gs(gandiva::NodePtr a, gandiva::NodePtr b) {
    return gcall("subtract", {a, b}, arrow::int64());
}
gandiva::NodePtr gm(gandiva::NodePtr a, gandiva::NodePtr b) {
    return gcall("multiply", {a, b}, arrow::int64());
}
gandiva::NodePtr gd(gandiva::NodePtr a, gandiva::NodePtr b) {
    return gcall("divide", {a, b}, arrow::int64());
}
gandiva::NodePtr gmin(gandiva::NodePtr a, gandiva::NodePtr b) {
    return gandiva::TreeExprBuilder::MakeIf(gcall("less_than", {a, b}, arrow::boolean()), a, b, arrow::int64());
}
gandiva::NodePtr gmax(gandiva::NodePtr a, gandiva::NodePtr b) {
    return gandiva::TreeExprBuilder::MakeIf(gcall("greater_than", {a, b}, arrow::boolean()), a, b, arrow::int64());
}
gandiva::NodePtr gabs(gandiva::NodePtr a) {
    return gandiva::TreeExprBuilder::MakeIf(gcall("less_than", {a, gql(0)}, arrow::boolean()), gs(gql(0), a), a,
                                            arrow::int64());
}

struct GandivaProgram {
    gandiva::ExpressionVector expressions;
    gandiva::ConditionPtr condition;
};
GandivaProgram build_gandiva(Workload workload) {
    if (!supported(Backend::Gandiva, workload))
        throw std::runtime_error("Gandiva does not support aggregation workloads");
    auto price = gq("price"), cost = gq("cost");
    auto discount = gcall("castBIGINT", {gi("discount")}, arrow::int64());
    auto tax = gcall("castBIGINT", {gi("tax")}, arrow::int64());
    auto revenue = gd(gm(price, gs(gql(10000), discount)), gql(10000));
    auto charge = gd(gm(revenue, ga(gql(10000), tax)), gql(10000)), margin = gs(charge, cost);
    gandiva::NodeVector roots;
    if (workload == Workload::PricingProjection)
        roots = {revenue, charge};
    else if (workload == Workload::BoundedOrderScore) {
        auto qty = gcall("castBIGINT", {gi("quantity")}, arrow::int64());
        auto adjusted = gs(gd(gm(gm(price, qty), gs(gql(10000), discount)), gql(10000)), gq("penalty"));
        roots = {gmin(gmax(adjusted, gq("lower")), gq("upper")), gabs(gs(adjusted, gq("target")))};
    } else if (workload == Workload::SharedArithmeticDag) {
        auto x = gq("x"), y = gq("y"), base = gm(ga(x, y), gq("z")), adjusted = gs(base, gm(x, y));
        roots = {ga(base, adjusted), gmin(base, adjusted), gmax(base, adjusted)};
    } else if (workload == Workload::ConditionalPricing) {
        auto large = gcall("greater_than_or_equal_to", {gi("quantity"), gil(25)}, arrow::boolean());
        roots = {gandiva::TreeExprBuilder::MakeIf(large, gd(gm(margin, gql(110)), gql(100)),
                                                  gd(gm(margin, gql(90)), gql(100)), arrow::int64()),
                 gmax(margin, gql(0)), gmin(charge, price), gabs(margin)};
    } else if (workload == Workload::RangeProjection) {
        auto open = gq("open"), close = gq("close"), lo = gmin(open, close), hi = gmax(open, close);
        roots = {lo, hi, gs(hi, lo), gmin(gabs(gs(close, open)), gq("limit"))};
    } else {
        auto terms = gandiva::TreeExprBuilder::MakeAnd({
            gcall("greater_than_or_equal_to", {gi("shipdate"), gil(19940101)}, arrow::boolean()),
            gcall("less_than_or_equal_to", {gi("shipdate"), gil(19940501)}, arrow::boolean()),
            gcall("greater_than_or_equal_to", {gi("discount"), gil(500)}, arrow::boolean()),
            gcall("less_than_or_equal_to", {gi("discount"), gil(700)}, arrow::boolean()),
            gcall("less_than", {gi("quantity"), gil(24)}, arrow::boolean()),
            gcall("greater_than", {margin, gql(5000)}, arrow::boolean()),
        });
        return {{}, gandiva::TreeExprBuilder::MakeCondition(terms)};
    }
    gandiva::ExpressionVector expressions;
    for (size_t n = 0; n < roots.size(); ++n)
        expressions.push_back(gandiva::TreeExprBuilder::MakeExpression(
            roots[n], arrow::field("out" + std::to_string(n), arrow::int64())));
    return {expressions, nullptr};
}

Prepared prepare_gandiva(const std::shared_ptr<Data> &data, Workload workload,
                         std::shared_ptr<gandiva::Projector> projector = nullptr,
                         std::shared_ptr<gandiva::Filter> filter = nullptr) {
    auto program = build_gandiva(workload);
    auto config = gandiva::ConfigurationBuilder::DefaultConfiguration();
    auto outputs = std::make_shared<arrow::ArrayVector>();
    auto selection = std::make_shared<std::shared_ptr<gandiva::SelectionVector>>();
    std::function<void()> run;
    std::function<Result()> result;
    if (is_projection(workload)) {
        if (!projector) check_arrow(gandiva::Projector::Make(data->schema, program.expressions, config, &projector));
        run = [data, projector, outputs] {
            check_arrow(projector->Evaluate(*data->batch, arrow::default_memory_pool(), outputs.get()));
        };
        result = [outputs] {
            Result r;
            for (const auto &base : *outputs) {
                auto array = std::static_pointer_cast<arrow::Int64Array>(base);
                r.arrays.emplace_back(array->raw_values(), array->raw_values() + array->length());
            }
            return r;
        };
    } else {
        if (!filter) check_arrow(gandiva::Filter::Make(data->schema, program.condition, config, &filter));
        run = [data, filter, selection] {
            std::shared_ptr<gandiva::SelectionVector> next;
            check_arrow(gandiva::SelectionVector::MakeInt64(data->rows, arrow::default_memory_pool(), &next));
            check_arrow(filter->Evaluate(*data->batch, next));
            *selection = std::move(next);
        };
        result = [selection] {
            Result r;
            auto array = std::static_pointer_cast<arrow::UInt64Array>((*selection)->ToArray());
            for (int64_t n = 0; n < array->length(); ++n)
                r.indices.push_back(int64_t(array->Value(n)));
            return r;
        };
    }
    Result expected = reference(*data, workload);
    auto release = [outputs, selection] {
        outputs->clear();
        selection->reset();
    };
    return {run, result, release, expected, is_selection(workload) ? int64_t(expected.indices.size()) : data->rows};
}

simjit::hir::Function *build_simjit_hir(simjit::Context &context, Workload workload) {
    simjit::FunctionBuilder b{context};
    if (workload == Workload::PricingProjection) {
        auto price = b.input_arg(I64), discount = b.signed_cast(b.input_arg(I32), I64),
             tax = b.signed_cast(b.input_arg(I32), I64);
        auto revenue = b.div(b.mul(price, b.sub(b.i64(10000), discount)), b.i64(10000));
        auto charge = b.div(b.mul(revenue, b.add(b.i64(10000), tax)), b.i64(10000));
        b.store(revenue, b.arg(I64));
        b.store(charge, b.arg(I64));
    } else if (workload == Workload::BoundedOrderScore) {
        auto price = b.input_arg(I64), qty = b.input_arg(I32), discount = b.signed_cast(b.input_arg(I32), I64),
             penalty = b.input_arg(I64);
        auto lower = b.input_arg(I64), upper = b.input_arg(I64), target = b.input_arg(I64);
        auto adjusted = b.sub(
            b.div(b.mul(b.mul(price, b.signed_cast(qty, I64)), b.sub(b.i64(10000), discount)), b.i64(10000)), penalty);
        b.store(b.min(b.max(adjusted, lower), upper), b.arg(I64));
        b.store(b.abs(b.sub(adjusted, target)), b.arg(I64));
    } else if (workload == Workload::SharedArithmeticDag) {
        auto x = b.input_arg(I64), y = b.input_arg(I64), z = b.input_arg(I64);
        auto base = b.mul(b.add(x, y), z), adjusted = b.sub(base, b.mul(x, y));
        b.store(b.add(base, adjusted), b.arg(I64));
        b.store(b.min(base, adjusted), b.arg(I64));
        b.store(b.max(base, adjusted), b.arg(I64));
    } else if (workload == Workload::ConditionalPricing) {
        auto price = b.input_arg(I64), discount = b.signed_cast(b.input_arg(I32), I64),
             tax = b.signed_cast(b.input_arg(I32), I64), cost = b.input_arg(I64), qty = b.input_arg(I32);
        auto revenue = b.div(b.mul(price, b.sub(b.i64(10000), discount)), b.i64(10000)),
             charge = b.div(b.mul(revenue, b.add(b.i64(10000), tax)), b.i64(10000)), margin = b.sub(charge, cost);
        b.store(b.select(b.cmp_ge(qty, b.i32(25)), b.div(b.mul(margin, b.i64(110)), b.i64(100)),
                         b.div(b.mul(margin, b.i64(90)), b.i64(100))),
                b.arg(I64));
        b.store(b.max(margin, b.i64(0)), b.arg(I64));
        b.store(b.min(charge, price), b.arg(I64));
        b.store(b.abs(margin), b.arg(I64));
    } else if (workload == Workload::RangeProjection) {
        auto open = b.input_arg(I64), close = b.input_arg(I64), limit = b.input_arg(I64);
        auto lo = b.min(open, close), hi = b.max(open, close);
        b.store(lo, b.arg(I64));
        b.store(hi, b.arg(I64));
        b.store(b.sub(hi, lo), b.arg(I64));
        b.store(b.min(b.abs(b.sub(close, open)), limit), b.arg(I64));
    } else {
        auto price = b.input_arg(I64);
        if (workload == Workload::RevenueSelection || workload == Workload::TpchQ6Revenue) {
            auto discount_i32 = b.input_arg(I32), tax_i32 = b.input_arg(I32), cost = b.input_arg(I64),
                 qty = b.input_arg(I32), shipdate = b.input_arg(I32);
            auto discount = b.signed_cast(discount_i32, I64), tax = b.signed_cast(tax_i32, I64);
            auto revenue = b.div(b.mul(price, b.sub(b.i64(10000), discount)), b.i64(10000)),
                 charge = b.div(b.mul(revenue, b.add(b.i64(10000), tax)), b.i64(10000)), margin = b.sub(charge, cost);
            auto pred = b.and_(b.cmp_ge(shipdate, b.i32(19940101)),
                               b.and_(b.cmp_le(shipdate, b.i32(19940501)),
                                      b.and_(b.cmp_ge(discount_i32, b.i32(500)),
                                             b.and_(b.cmp_le(discount_i32, b.i32(700)),
                                                    b.and_(b.cmp_lt(qty, b.i32(24)), b.cmp_gt(margin, b.i64(5000)))))));
            if (workload == Workload::RevenueSelection)
                b.pack(b.index(I64), pred, b.arg(I64), b.arg(I64));
            else
                b.sum_if(revenue, pred, b.arg(I64));
        } else if (workload == Workload::PricingSummary) {
            auto discount = b.signed_cast(b.input_arg(I32), I64), tax = b.signed_cast(b.input_arg(I32), I64),
                 qty = b.input_arg(I32);
            auto revenue = b.div(b.mul(price, b.sub(b.i64(10000), discount)), b.i64(10000));
            auto charge = b.div(b.mul(revenue, b.add(b.i64(10000), tax)), b.i64(10000));
            b.sum(b.signed_cast(qty, I64), b.arg(I64));
            b.sum(price, b.arg(I64));
            b.sum(revenue, b.arg(I64));
            b.sum(charge, b.arg(I64));
            b.min_agg(charge, b.arg(I64));
            b.max_agg(charge, b.arg(I64));
        } else if (workload == Workload::SegmentedLineTotals) {
            auto qty = b.input_arg(I32), dbp = b.input_arg(I32), tbp = b.input_arg(I32), delay = b.input_arg(I32);
            auto net = b.div(b.mul(b.mul(b.signed_cast(qty, I64), price), b.sub(b.i64(10000), b.signed_cast(dbp, I64))),
                             b.i64(10000));
            auto line = b.div(b.mul(net, b.add(b.i64(10000), b.signed_cast(tbp, I64))), b.i64(10000));
            b.sum_if(line, b.cmp_gt(delay, b.i32(30)), b.arg(I64));
            b.sum_if(line, b.cmp_ge(qty, b.i32(25)), b.arg(I64));
            b.sum_if(line, b.cmp_gt(line, b.i64(1000000)), b.arg(I64));
            b.min_agg(line, b.arg(I64));
            b.max_agg(line, b.arg(I64));
        } else {
            auto y = b.input_arg(I64), x0 = b.input_arg(I64), y0 = b.input_arg(I64);
            auto cx = b.sub(price, x0), cy = b.sub(y, y0), cross = b.mul(cx, cy);
            b.sum(cx, b.arg(I64));
            b.sum(cy, b.arg(I64));
            b.sum(b.mul(cx, cx), b.arg(I64));
            b.sum(cross, b.arg(I64));
            b.sum(b.mul(cy, cy), b.arg(I64));
            b.min_agg(cross, b.arg(I64));
            b.max_agg(cross, b.arg(I64));
        }
    }
    return b.build();
}

Prepared prepare_simjit(const std::shared_ptr<Data> &d, Workload w, simjit::jit::CompilePolicy policy,
                        simjit::hir::Function *hir = nullptr, std::shared_ptr<simjit::MemoryArena> arena = nullptr,
                        std::shared_ptr<simjit::AsmjitSession> session = nullptr, void *compiled = nullptr) {
    if (!arena) arena = std::make_shared<simjit::MemoryArena>();
    auto context =
        std::make_shared<simjit::Context>(*arena, "external", simjit::CodeTransformations::All, simjit::Arch::Native);
    if (!hir) hir = build_simjit_hir(*context, w);
    if (!session) session = std::make_shared<simjit::AsmjitSession>(simjit::Arch::Native);
    if (!compiled) compiled = simjit::compile_asmjit_from_hir(hir, *session, policy);
    if (!compiled) throw std::runtime_error("Simjit returned a null function pointer");
    Result expected = reference(*d, w);
    const size_t array_outputs = expected.arrays.size();
    const size_t scalar_outputs = expected.scalars.size();
    auto out = std::make_shared<std::unique_ptr<Result>>();
    auto run = [=] {
        auto next = std::make_unique<Result>();
        next->arrays.resize(array_outputs);
        for (auto &array : next->arrays)
            array.resize(size_t(d->rows));
        if (is_selection(w)) next->indices.resize(size_t(d->rows));
        next->scalars.assign(scalar_outputs, 0);
        if (next->scalars.size() >= 5) {
            next->scalars[next->scalars.size() - 2] = std::numeric_limits<int64_t>::max();
            next->scalars.back() = std::numeric_limits<int64_t>::min();
        }

        std::vector<void *> args;
        auto addq = [&](QCol c) { args.push_back(const_cast<int64_t *>(d->q[size_t(c)]->raw_values())); };
        auto addi = [&](ICol c) { args.push_back(const_cast<int32_t *>(d->i[size_t(c)]->raw_values())); };
        auto add_array = [&](size_t n) { args.push_back(next->arrays[n].data()); };
        auto add_scalar = [&](size_t n) { args.push_back(&next->scalars[n]); };
        int64_t count = 0;
        if (w == Workload::PricingProjection) {
            addq(Price);
            addi(Discount);
            addi(Tax);
            add_array(0);
            add_array(1);
        } else if (w == Workload::BoundedOrderScore) {
            addq(Price);
            addi(Quantity);
            addi(Discount);
            addq(Penalty);
            addq(Lower);
            addq(Upper);
            addq(Target);
            add_array(0);
            add_array(1);
        } else if (w == Workload::SharedArithmeticDag) {
            addq(X);
            addq(Y);
            addq(Z);
            add_array(0);
            add_array(1);
            add_array(2);
        } else if (w == Workload::ConditionalPricing) {
            addq(Price);
            addi(Discount);
            addi(Tax);
            addq(Cost);
            addi(Quantity);
            add_array(0);
            add_array(1);
            add_array(2);
            add_array(3);
        } else if (w == Workload::RangeProjection) {
            addq(Open);
            addq(Close);
            addq(Limit);
            add_array(0);
            add_array(1);
            add_array(2);
            add_array(3);
        } else if (w == Workload::RevenueSelection) {
            addq(Price);
            addi(Discount);
            addi(Tax);
            addq(Cost);
            addi(Quantity);
            addi(Shipdate);
            args.push_back(next->indices.data());
            args.push_back(&count);
        } else if (w == Workload::TpchQ6Revenue) {
            addq(Price);
            addi(Discount);
            addi(Tax);
            addq(Cost);
            addi(Quantity);
            addi(Shipdate);
            add_scalar(0);
        } else if (w == Workload::PricingSummary) {
            addq(Price);
            addi(Discount);
            addi(Tax);
            addi(Quantity);
            for (size_t n = 0; n < 6; ++n)
                add_scalar(n);
        } else if (w == Workload::SegmentedLineTotals) {
            addq(Price);
            addi(Quantity);
            addi(DiscountBp);
            addi(TaxBp);
            addi(Delay);
            for (size_t n = 0; n < 5; ++n)
                add_scalar(n);
        } else {
            addq(X);
            addq(Y);
            addq(X0);
            addq(Y0);
            for (size_t n = 0; n < 7; ++n)
                add_scalar(n);
        }
        simjit::jit::call_fn_ptr(compiled, size_t(d->rows), simjit::nonstd::span<void *>(args.data(), args.size()));
        if (is_selection(w)) next->indices.resize(size_t(count));
        *out = std::move(next);
    };
    auto result = [out, arena, context, session] {
        if (!*out) throw std::runtime_error("Simjit output is not available");
        return **out;
    };
    return {run, result, [out] { out->reset(); }, expected,
            is_selection(w) ? int64_t(expected.indices.size()) : d->rows};
}

Prepared prepare(Backend backend, Workload workload, int64_t rows, std::string_view selection_shape = "normal") {
    if (!supported(backend, workload)) throw std::runtime_error("unsupported backend/workload pair");
    auto data = make_data(rows, selection_shape);
    if (backend == Backend::Arrow) return prepare_arrow(data, workload);
    if (backend == Backend::Gandiva) return prepare_gandiva(data, workload);
    return prepare_simjit(data, workload,
                          backend == Backend::SimjitScalar ? simjit::jit::CompilePolicy::Scalar
                                                           : simjit::jit::CompilePolicy::Vectorized);
}

Backend parse_backend(std::string_view value) {
    if (value == "gandiva") return Backend::Gandiva;
    if (value == "simjit_scalar") return Backend::SimjitScalar;
    if (value == "simjit_simd") return Backend::SimjitSimd;
    throw std::runtime_error("unknown compile backend");
}
Workload parse_workload(std::string_view value) {
    for (Workload workload : Workloads)
        if (value == show(workload)) return workload;
    throw std::runtime_error("unknown workload");
}

double compile_sample(Backend backend, Workload workload, int64_t rows, uint64_t &checksum, int64_t &count,
                      bool &cache_hit) {
    if (!supported(backend, workload) || backend == Backend::Arrow)
        throw std::runtime_error("unsupported compilation pair");
    auto data = make_data(rows);
    cache_hit = false;
    Prepared prepared;
    auto begin = std::chrono::steady_clock::now();
    if (backend == Backend::Gandiva) {
        auto program = build_gandiva(workload);
        auto config = gandiva::ConfigurationBuilder::DefaultConfiguration();
        if (is_projection(workload)) {
            std::shared_ptr<gandiva::Projector> projector;
            begin = std::chrono::steady_clock::now();
            check_arrow(gandiva::Projector::Make(data->schema, program.expressions, config, &projector));
            auto end = std::chrono::steady_clock::now();
            cache_hit = projector->GetBuiltFromCache();
            prepared = prepare_gandiva(data, workload, projector, nullptr);
            checksum = validate(prepared, workload);
            count = prepared.result_count;
            return std::chrono::duration<double, std::micro>(end - begin).count();
        }
        std::shared_ptr<gandiva::Filter> filter;
        begin = std::chrono::steady_clock::now();
        check_arrow(gandiva::Filter::Make(data->schema, program.condition, config, &filter));
        auto end = std::chrono::steady_clock::now();
        cache_hit = filter->GetBuiltFromCache();
        prepared = prepare_gandiva(data, workload, nullptr, filter);
        checksum = validate(prepared, workload);
        count = prepared.result_count;
        return std::chrono::duration<double, std::micro>(end - begin).count();
    }
    auto arena = std::make_shared<simjit::MemoryArena>();
    auto context =
        std::make_shared<simjit::Context>(*arena, "external", simjit::CodeTransformations::All, simjit::Arch::Native);
    auto *hir = build_simjit_hir(*context, workload);
    auto session = std::make_shared<simjit::AsmjitSession>(simjit::Arch::Native);
    auto policy =
        backend == Backend::SimjitScalar ? simjit::jit::CompilePolicy::Scalar : simjit::jit::CompilePolicy::Vectorized;
    begin = std::chrono::steady_clock::now();
    void *fn = simjit::compile_asmjit_from_hir(hir, *session, policy);
    auto end = std::chrono::steady_clock::now();
    prepared = prepare_simjit(data, workload, policy, hir, arena, session, fn);
    checksum = validate(prepared, workload);
    count = prepared.result_count;
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

void benchmark_execution(benchmark::State &state, Backend backend, Workload workload) {
    auto prepared = prepare(backend, workload, g_rows);
    auto checksum = validate(prepared, workload);
    benchmark::DoNotOptimize(checksum);
    for (auto _ : state) {
        prepared.release();
        auto begin = std::chrono::steady_clock::now();
        prepared.run();
        auto end = std::chrono::steady_clock::now();
        state.SetIterationTime(std::chrono::duration<double>(end - begin).count());
        benchmark::ClobberMemory();
    }
    prepared.release();
    state.SetItemsProcessed(state.iterations() * g_rows);
    state.counters["rows"] = double(g_rows);
}

void register_benchmarks() {
    for (Backend backend : Backends)
        for (Workload workload : Workloads)
            if (supported(backend, workload)) {
                std::string name = std::string(show(backend)) + "/" + show(workload);
                benchmark::RegisterBenchmark(name.c_str(), [backend, workload](benchmark::State &state) {
                    benchmark_execution(state, backend, workload);
                })->UseManualTime();
            }
}

int verify_x86_vectorization() {
    int cases = 0;
    for (Workload workload : Workloads) {
        simjit::MemoryArena arena;
        simjit::Context context(arena, "external-x86-check", simjit::CodeTransformations::All,
                                simjit::Arch::Amd64_AVX512);
        auto *hir = build_simjit_hir(context, workload);
        auto *mir = simjit::lower_vectorized(hir);
        simjit::AsmjitSession session(simjit::Arch::Amd64_AVX512);
        simjit::AsmjitCompileResult result;
        simjit::compile_asmjit(mir, simjit::AsmjitCompileOptions{false, false, &session}, result);
        ++cases;
    }
    return cases;
}

} // namespace

int main(int argc, char **argv) {
    try {
        check_arrow(arrow::Initialize({}));
        check_arrow(arrow::compute::Initialize());
        std::string compile_spec;
        bool self_test = false;
        bool print_input_checksum = false;
        bool verify_x86 = false;
        std::vector<char *> benchmark_args{argv[0]};
        for (int n = 1; n < argc; ++n) {
            std::string_view arg = argv[n];
            if (arg.starts_with("--compile-sample="))
                compile_spec = std::string(arg.substr(std::strlen("--compile-sample=")));
            else if (arg.starts_with("--rows="))
                g_rows = std::stoll(std::string(arg.substr(std::strlen("--rows="))));
            else if (arg == "--self-test")
                self_test = true;
            else if (arg == "--input-checksum")
                print_input_checksum = true;
            else if (arg == "--verify-x86-vectorization")
                verify_x86 = true;
            else
                benchmark_args.push_back(argv[n]);
        }
        if (g_rows <= 0) throw std::runtime_error("rows must be positive");
        if (verify_x86) {
            std::cout << "{\"x86_vectorization\":true,\"arch\":\"amd64-avx512\",\"cases\":"
                      << verify_x86_vectorization() << "}\n";
            return 0;
        }
        if (print_input_checksum) {
            auto data = make_data(g_rows);
            std::cout << "{\"rows\":" << g_rows << ",\"input_checksum\":" << input_checksum(*data) << "}\n";
            return 0;
        }
        if (self_test) {
            int cases = 0;
            for (const auto &[rows, shape] : std::vector<std::pair<int64_t, const char *>>{
                     {127, "normal"}, {4096, "normal"}, {127, "empty"}, {127, "full"}})
                for (Backend backend : Backends)
                    for (Workload workload : Workloads)
                        if (supported(backend, workload)) {
                            auto p = prepare(backend, workload, rows, shape);
                            validate(p, workload);
                            ++cases;
                        }
            std::cout << "{\"self_test\":true,\"cases\":" << cases << "}\n";
            return 0;
        }
        if (!compile_spec.empty()) {
            auto comma = compile_spec.find(',');
            if (comma == std::string::npos) throw std::runtime_error("compile sample must be BACKEND,WORKLOAD");
            Backend backend = parse_backend(std::string_view(compile_spec).substr(0, comma));
            Workload workload = parse_workload(std::string_view(compile_spec).substr(comma + 1));
            uint64_t checksum = 0;
            int64_t count = 0;
            bool cache_hit = false;
            double us = compile_sample(backend, workload, g_rows, checksum, count, cache_hit);
            std::cout << "{\"backend\":\"" << show(backend) << "\",\"workload\":\"" << show(workload)
                      << "\",\"compile_us\":" << us << ",\"cache_hit\":" << (cache_hit ? "true" : "false")
                      << ",\"checksum\":" << checksum << ",\"result_count\":" << count << "}\n";
            return cache_hit ? 2 : 0;
        }
        int benchmark_argc = int(benchmark_args.size());
        register_benchmarks();
        benchmark::Initialize(&benchmark_argc, benchmark_args.data());
        if (benchmark::ReportUnrecognizedArguments(benchmark_argc, benchmark_args.data())) return 2;
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "jit-bench: " << error.what() << '\n';
        return 1;
    }
}
