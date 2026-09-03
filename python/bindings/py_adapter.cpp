// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "py_adapter.h"
#include "native_builder.h"

#include <pybind11/numpy.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "simjit/asmjit.h"
#include "simjit/compiler.h"
#include "simjit/core/expr.h"
#if SIMJIT_LLVM_BACKEND
#include "simjit/core/llvm/emitter.h"
#endif
#include "simjit/dynamic_value.h"
#include "simjit/nullable.h"

namespace simjit_python {
namespace {
using namespace pybind11::literals;

#define DSL_ATTR_LIST(X)   \
    X(ScalarType)          \
    X(I8)                  \
    X(I16)                 \
    X(I32)                 \
    X(I64)                 \
    X(U8)                  \
    X(U16)                 \
    X(U32)                 \
    X(U64)                 \
    X(F32)                 \
    X(F64)                 \
    X(I1)                  \
    X(Expr)                \
    X(ConstExpr)           \
    X(LoadExpr)            \
    X(LoadSplatExpr)       \
    X(GatherExpr)          \
    X(IndexExpr)           \
    X(ArithBinaryExpr)     \
    X(PredicateBinaryExpr) \
    X(ArithUnaryExpr)      \
    X(PredicateNotExpr)    \
    X(CompareExpr)         \
    X(IntCastExpr)         \
    X(FloatCastExpr)       \
    X(BitCastExpr)         \
    X(FunctionExpr)        \
    X(SelectExpr)          \
    X(FpClassExpr)         \
    X(PermuteExpr)         \
    X(StoreExpr)           \
    X(ScatterExpr)         \
    X(PackExpr)            \
    X(ArithAggExpr)        \
    X(PredicateAggExpr)    \
    X(CountIfExpr)         \
    X(GroupedArithAggExpr) \
    X(NullEncoding)        \
    X(BufferHandle)

struct PyBufferHandle : BufferDesc {
    using NullEncoding = BufferNullDesc;
    using NullKind = BufferNullKind;

    py::buffer buf{};
    py::object null_buf{};
};

struct DslModule {
    DslModule() = delete;
    DslModule(const DslModule &) = delete;
    DslModule(DslModule &&) = delete;
    DslModule &operator=(const DslModule &) = delete;
    DslModule &operator=(DslModule &&) = delete;

    py::module_ mod;
#define X(_name) py::object _name;
    DSL_ATTR_LIST(X)
#undef X
};

static const DslModule &get_dsl() {
    // Keep the cached Python objects alive until process exit to avoid Python destruction-order dependencies.
    static const DslModule *instance = []() -> const DslModule * {
        auto mod = py::module_::import("simjit.ir");

        alignas(DslModule) static unsigned char memory[sizeof(DslModule)];
        return new (memory) DslModule{mod,
#define X(_name) mod.attr(#_name),
                                      DSL_ATTR_LIST(X)
#undef X
        };
    }();
    return *instance;
}

#define SV(x) static_cast<int>((x).size()), ((x).data() == nullptr ? "" : (x).data())

static std::string_view py_string_view(py::handle value, std::string_view field_name) {
    if (!PyUnicode_Check(value.ptr())) { throw py::type_error(sj::format("%.*s must be str", SV(field_name))); }
    Py_ssize_t size = 0;
    const char *data = PyUnicode_AsUTF8AndSize(value.ptr(), &size);
    if (data == nullptr) { throw py::error_already_set(); }
    if (size < 0) { throw py::value_error(sj::format("%.*s has invalid size", SV(field_name))); }
    return std::string_view(data, static_cast<size_t>(size));
}

template <typename Enum> static Enum py_enum_value(py::handle value, std::string_view field_name) {
    try {
        return value.cast<Enum>();
    } catch (const py::cast_error &) {
        throw py::type_error(sj::format("%.*s must be a native enum value", SV(field_name)));
    }
}

static TimestampTimezone parse_timestamp_timezone_value(std::string_view value) {
    if (value == "UTC") { return TimestampTimezone::UTC; }
    throw std::invalid_argument(sj::format("Invalid timestamp timezone %.*s", SV(value)));
}

static TimestampTimezone parse_timestamp_timezone(py::handle tz, std::string_view field_name) {
    if (tz.is_none()) { return TimestampTimezone::None; }
    std::string_view value = py_string_view(tz, field_name);
    return parse_timestamp_timezone_value(value);
}

static TimestampUnit parse_timestamp_unit(std::string_view unit) {
    if (unit == "s") { return TimestampUnit::Seconds; }
    if (unit == "ms") { return TimestampUnit::Milliseconds; }
    if (unit == "us") { return TimestampUnit::Microseconds; }
    if (unit == "ns") { return TimestampUnit::Nanoseconds; }
    throw std::invalid_argument(sj::format("Invalid timestamp unit %.*s", SV(unit)));
}

static LogicalType parse_type(const py::handle &ty) {
    const auto &dsl = get_dsl();
    if (ty.is(dsl.I1)) { return {sj::ScalarDataType::I1, false}; }
    if (ty.is(dsl.I8)) { return {sj::ScalarDataType::I8, false}; }
    if (ty.is(dsl.I16)) { return {sj::ScalarDataType::I16, false}; }
    if (ty.is(dsl.I32)) { return {sj::ScalarDataType::I32, false}; }
    if (ty.is(dsl.I64)) { return {sj::ScalarDataType::I64, false}; }
    if (ty.is(dsl.U8)) { return {sj::ScalarDataType::I8, true}; }
    if (ty.is(dsl.U16)) { return {sj::ScalarDataType::I16, true}; }
    if (ty.is(dsl.U32)) { return {sj::ScalarDataType::I32, true}; }
    if (ty.is(dsl.U64)) { return {sj::ScalarDataType::I64, true}; }
    if (ty.is(dsl.F32)) { return {sj::ScalarDataType::F32, false}; }
    if (ty.is(dsl.F64)) { return {sj::ScalarDataType::F64, false}; }
    if (py::hasattr(ty, "name") && py_string_view(ty.attr("name"), "type name") == "timestamp64") {
        TimestampUnit unit = parse_timestamp_unit(py_string_view(ty.attr("unit"), "timestamp unit"));
        TimestampTimezone timezone = parse_timestamp_timezone(ty.attr("tz"), "timestamp timezone");
        return {sj::ScalarDataType::I64, false, SemanticKind::Timestamp64, unit, timezone};
    }
    throw py::value_error(sj::format("Invalid type %s", py::str(ty).cast<std::string>().c_str()));
}

static py::str py_string_from_view(std::string_view value);

static std::string_view borrow_string_view(py::handle value, std::string_view field_name) {
    if (!PyUnicode_Check(value.ptr())) { throw py::type_error(sj::format("%.*s must be str", SV(field_name))); }
    Py_ssize_t size = 0;
    const char *data = PyUnicode_AsUTF8AndSize(value.ptr(), &size);
    if (data == nullptr) { throw py::error_already_set(); }
    if (size < 0) { throw py::value_error(sj::format("%.*s has invalid size", SV(field_name))); }
    return std::string_view(data, static_cast<size_t>(size));
}

static bool is_sequence_like(py::handle value) {
    return PySequence_Check(value.ptr()) != 0 && !PyUnicode_Check(value.ptr());
}

static py::object sequence_item(py::handle seq, size_t idx) {
    PyObject *item = PySequence_GetItem(seq.ptr(), static_cast<Py_ssize_t>(idx));
    if (item == nullptr) { throw py::error_already_set(); }
    return py::reinterpret_steal<py::object>(item);
}

static size_t sequence_size(py::handle seq, std::string_view name) {
    if (!is_sequence_like(seq)) { throw py::type_error(sj::format("%.*s must be a sequence", SV(name))); }
    Py_ssize_t size = PySequence_Size(seq.ptr());
    if (size < 0) { throw py::error_already_set(); }
    return static_cast<size_t>(size);
}

static LogicalType parse_dsl_type(const py::handle &ty) {
    const auto &dsl = get_dsl();
    if (ty.is(dsl.I1)) { return {sj::ScalarDataType::I1, false}; }
    if (ty.is(dsl.I8)) { return {sj::ScalarDataType::I8, false}; }
    if (ty.is(dsl.I16)) { return {sj::ScalarDataType::I16, false}; }
    if (ty.is(dsl.I32)) { return {sj::ScalarDataType::I32, false}; }
    if (ty.is(dsl.I64)) { return {sj::ScalarDataType::I64, false}; }
    if (ty.is(dsl.U8)) { return {sj::ScalarDataType::I8, true}; }
    if (ty.is(dsl.U16)) { return {sj::ScalarDataType::I16, true}; }
    if (ty.is(dsl.U32)) { return {sj::ScalarDataType::I32, true}; }
    if (ty.is(dsl.U64)) { return {sj::ScalarDataType::I64, true}; }
    if (ty.is(dsl.F32)) { return {sj::ScalarDataType::F32, false}; }
    if (ty.is(dsl.F64)) { return {sj::ScalarDataType::F64, false}; }
    if (py::hasattr(ty, "name") && py_string_view(ty.attr("name"), "type name") == "timestamp64") {
        py::handle unit = ty.attr("unit");
        TimestampUnit parsed_unit = parse_timestamp_unit(borrow_string_view(unit, "timestamp unit"));
        TimestampTimezone timezone = parse_timestamp_timezone(ty.attr("tz"), "timestamp timezone");
        return {sj::ScalarDataType::I64, false, SemanticKind::Timestamp64, parsed_unit, timezone};
    }
    throw py::value_error(sj::format("Invalid type %s", py::str(ty).cast<std::string>().c_str()));
}

static bool parse_optional_dsl_type(const py::handle &ty, LogicalType *out) {
    if (ty.is_none()) { return false; }
    *out = parse_dsl_type(ty);
    return true;
}

static DslNodeKind dsl_node_kind(const py::handle &expr) {
    const auto &dsl = get_dsl();
    if (!py::isinstance(expr, dsl.Expr)) {
        throw py::type_error(sj::format("expected simjit.ir.Expr, got %s", py::str(expr).cast<std::string>().c_str()));
    }
    if (py::isinstance(expr, dsl.ConstExpr)) { return DslNodeKind::Const; }
    if (py::isinstance(expr, dsl.LoadExpr)) { return DslNodeKind::Load; }
    if (py::isinstance(expr, dsl.LoadSplatExpr)) { return DslNodeKind::LoadSplat; }
    if (py::isinstance(expr, dsl.GatherExpr)) { return DslNodeKind::Gather; }
    if (py::isinstance(expr, dsl.IndexExpr)) { return DslNodeKind::Index; }
    if (py::isinstance(expr, dsl.ArithBinaryExpr)) { return DslNodeKind::ArithBinary; }
    if (py::isinstance(expr, dsl.PredicateBinaryExpr)) { return DslNodeKind::PredicateBinary; }
    if (py::isinstance(expr, dsl.ArithUnaryExpr)) { return DslNodeKind::ArithUnary; }
    if (py::isinstance(expr, dsl.PredicateNotExpr)) { return DslNodeKind::PredicateNot; }
    if (py::isinstance(expr, dsl.CompareExpr)) { return DslNodeKind::Compare; }
    if (py::isinstance(expr, dsl.IntCastExpr)) { return DslNodeKind::IntCast; }
    if (py::isinstance(expr, dsl.FloatCastExpr)) { return DslNodeKind::FloatCast; }
    if (py::isinstance(expr, dsl.BitCastExpr)) { return DslNodeKind::BitCast; }
    if (py::isinstance(expr, dsl.FunctionExpr)) { return DslNodeKind::Function; }
    if (py::isinstance(expr, dsl.SelectExpr)) { return DslNodeKind::Select; }
    if (py::isinstance(expr, dsl.FpClassExpr)) { return DslNodeKind::FpClass; }
    if (py::isinstance(expr, dsl.PermuteExpr)) { return DslNodeKind::Permute; }
    if (py::isinstance(expr, dsl.StoreExpr)) { return DslNodeKind::Store; }
    if (py::isinstance(expr, dsl.ScatterExpr)) { return DslNodeKind::Scatter; }
    if (py::isinstance(expr, dsl.PackExpr)) { return DslNodeKind::Pack; }
    if (py::isinstance(expr, dsl.ArithAggExpr)) { return DslNodeKind::ArithAgg; }
    if (py::isinstance(expr, dsl.PredicateAggExpr)) { return DslNodeKind::PredicateAgg; }
    if (py::isinstance(expr, dsl.CountIfExpr)) { return DslNodeKind::CountIf; }
    if (py::isinstance(expr, dsl.GroupedArithAggExpr)) { return DslNodeKind::GroupedArithAgg; }
    throw py::type_error(
        sj::format("unsupported simjit.ir.Expr subclass %s", py::str(expr).cast<std::string>().c_str()));
}

static ConstPayload parse_const_payload(const py::handle &expr, bool has_declared_type,
                                        const LogicalType &declared_type) {
    py::handle value = expr.attr("value");
    ConstPayload payload{};
    if (PyBool_Check(value.ptr())) {
        payload.kind = ConstPayloadKind::Bool;
        payload.bool_value = value.cast<bool>();
        return payload;
    }
    if (PyFloat_Check(value.ptr())) {
        payload.kind = ConstPayloadKind::Float;
        payload.float_value = value.cast<double>();
        return payload;
    }
    if (PyLong_Check(value.ptr())) {
        if (has_declared_type && declared_type.is_unsigned) {
            payload.kind = ConstPayloadKind::UnsignedInt;
            payload.unsigned_int = PyLong_AsUnsignedLongLong(value.ptr());
            if (PyErr_Occurred()) { throw py::error_already_set(); }
            return payload;
        }
        long long signed_value = PyLong_AsLongLong(value.ptr());
        if (PyErr_Occurred()) {
            if (!PyErr_ExceptionMatches(PyExc_OverflowError)) { throw py::error_already_set(); }
            PyErr_Clear();
            payload.kind = ConstPayloadKind::UnsignedInt;
            payload.unsigned_int = PyLong_AsUnsignedLongLong(value.ptr());
            if (PyErr_Occurred()) { throw py::error_already_set(); }
            return payload;
        }
        payload.kind = ConstPayloadKind::SignedInt;
        payload.signed_int = signed_value;
        return payload;
    }
    throw py::type_error("ConstExpr.value must be bool, int, or float");
}

struct DslImportState {
    struct PendingChild {
        py::handle expr{};
        NodeId parent = 0;
        uint16_t index = 0;
    };

    DslProgram program{};
    std::vector<PendingChild> pending_children{};
    uint32_t node_cursor = 0;
    uint32_t edge_cursor = 0;
    uint32_t output_cursor = 0;
    uint32_t string_cursor = 0;
    std::unordered_map<uintptr_t, NodeId> expr_memo{};

    explicit DslImportState(size_t output_count, DslImportLimits limits = {}) {
        program.outputs = program.arena.alloc_array<DslOutput>(output_count);
        program.nodes = program.arena.alloc_array<DslNode>(limits.node_capacity);
        program.declared_types = program.arena.alloc_array<DslTypeSlot>(limits.node_capacity);
        program.resolved_types = program.arena.alloc_array<DslTypeSlot>(limits.node_capacity);
        program.child_edges = program.arena.alloc_array<NodeId>(limits.edge_capacity);
        program.strings = program.arena.alloc_array<std::string_view>(limits.string_capacity);
        pending_children.reserve(limits.node_capacity);
        expr_memo.reserve(limits.node_capacity);
    }

    DslStringRef native_string(std::string_view value) {
        if (string_cursor >= program.strings.size()) { throw py::value_error("native DSL string limit exceeded"); }
        uint32_t id = string_cursor++;
        char *data = static_cast<char *>(program.arena.alloc(value.size()));
        std::memcpy(data, value.data() == nullptr ? "" : value.data(), value.size());
        program.strings[id] = std::string_view(data, value.size());
        return DslStringRef::native(id);
    }

    DslStringRef copy_string(py::handle value, std::string_view field_name) {
        return native_string(borrow_string_view(value, field_name));
    }

    std::string_view copy_string_view(py::handle value, std::string_view field_name) {
        DslStringRef ref = copy_string(value, field_name);
        return program.strings[ref.native_id()];
    }

    void reserve_children(DslNode &node, size_t count) {
        if (count > std::numeric_limits<uint16_t>::max()) {
            throw py::value_error("native DSL node has too many child edges");
        }
        if (count > program.child_edges.size() - edge_cursor) {
            throw py::value_error("native DSL edge limit exceeded");
        }
        node.first_child = edge_cursor;
        node.child_count = static_cast<uint16_t>(count);
        edge_cursor += static_cast<uint32_t>(count);
    }

    void push_child(NodeId parent, uint32_t index, const py::handle &child) {
        const DslNode &node = program.nodes[parent];
        SIMJIT_ASSERT(index < node.child_count);
        if (index > std::numeric_limits<uint16_t>::max()) { throw py::value_error("native DSL child index overflow"); }
        if (pending_children.size() >= program.nodes.size()) {
            throw py::value_error("native DSL task limit exceeded");
        }
        pending_children.push_back(PendingChild{child, parent, static_cast<uint16_t>(index)});
    }

    NodeId append_expr(const py::handle &expr) {
        // Import runs synchronously with the GIL held. The Python outputs object keeps the expression graph alive for
        // the duration of import, so PyObject addresses are stable enough for this import-local identity memo. Never
        // persist these keys outside DslImportState.
        uintptr_t expr_key = reinterpret_cast<uintptr_t>(expr.ptr());
        if (auto it = expr_memo.find(expr_key); it != expr_memo.end()) { return it->second; }

        DslNodeKind kind = dsl_node_kind(expr);
        if (node_cursor >= program.nodes.size()) { throw py::value_error("native DSL node limit exceeded"); }
        NodeId node_id = node_cursor++;
        expr_memo.emplace(expr_key, node_id);
        DslNode &node = program.nodes[node_id];
        DslTypeSlot declared{};
        declared.has = parse_optional_dsl_type(expr.attr("dt"), &declared.type);
        program.declared_types[node_id] = declared;
        program.resolved_types[node_id] = {};
        node.kind = kind;
        DslNode *step = &node;
        switch (node.kind) {
            SIMJIT_MATCH (DslNodeKind::Const) {
                reserve_children(node, 0);
                data = parse_const_payload(expr, declared.has, declared.type);
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::Load) {
                reserve_children(node, 0);
                data.name = copy_string(expr.attr("name"), "LoadExpr.name");
                data.kind = py_enum_value<sj::LoadStoreKind>(expr.attr("kind"), "LoadExpr.kind");
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::LoadSplat) {
                reserve_children(node, 0);
                data.name = copy_string(expr.attr("name"), "LoadSplatExpr.name");
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::Gather) {
                reserve_children(node, 1);
                data.name = copy_string(expr.attr("name"), "GatherExpr.name");
                push_child(node_id, 0, expr.attr("idx"));
                return node_id;
            }
        case DslNodeKind::Index:
            reserve_children(node, 0);
            return node_id;
            SIMJIT_MATCH (DslNodeKind::ArithBinary) {
                reserve_children(node, 2);
                data.op = py_enum_value<sj::ArithBinaryOp>(expr.attr("op"), "ArithBinaryExpr.op");
                data.checked = expr.attr("checked").cast<bool>();
                push_child(node_id, 0, expr.attr("lhs"));
                push_child(node_id, 1, expr.attr("rhs"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::PredicateBinary) {
                reserve_children(node, 2);
                data.op = py_enum_value<sj::PredicateBinaryOp>(expr.attr("op"), "PredicateBinaryExpr.op");
                push_child(node_id, 0, expr.attr("lhs"));
                push_child(node_id, 1, expr.attr("rhs"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::ArithUnary) {
                reserve_children(node, 1);
                data.op = py_enum_value<sj::ArithUnaryOp>(expr.attr("op"), "ArithUnaryExpr.op");
                data.checked = expr.attr("checked").cast<bool>();
                push_child(node_id, 0, expr.attr("arg"));
                return node_id;
            }
        case DslNodeKind::PredicateNot:
            reserve_children(node, 1);
            push_child(node_id, 0, expr.attr("arg"));
            return node_id;
            SIMJIT_MATCH (DslNodeKind::Compare) {
                reserve_children(node, 2);
                data.op = py_enum_value<sj::CmpOp>(expr.attr("op"), "CompareExpr.op");
                data.is_unsigned = expr.attr("unsigned").cast<bool>();
                push_child(node_id, 0, expr.attr("lhs"));
                push_child(node_id, 1, expr.attr("rhs"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::IntCast) {
                reserve_children(node, 1);
                data.kind = py_enum_value<DslIntCastKind>(expr.attr("kind"), "IntCastExpr.kind");
                data.checked = expr.attr("checked").cast<bool>();
                push_child(node_id, 0, expr.attr("arg"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::FloatCast) {
                reserve_children(node, 1);
                data.is_unsigned = expr.attr("is_unsigned").cast<bool>();
                push_child(node_id, 0, expr.attr("arg"));
                return node_id;
            }
        case DslNodeKind::BitCast:
            reserve_children(node, 1);
            push_child(node_id, 0, expr.attr("arg"));
            return node_id;
            SIMJIT_MATCH (DslNodeKind::Function) {
                py::tuple args = expr.attr("args").cast<py::tuple>();
                reserve_children(node, args.size());
                data.kind = py_enum_value<DslFunctionKind>(expr.attr("name"), "FunctionExpr.name");
                if (find_dsl_function_spec(data.kind) == nullptr) {
                    throw py::type_error("FunctionExpr.name must be a public FunctionName value");
                }
                for (uint32_t i = 0; i < node.child_count; ++i) {
                    push_child(node_id, i, args[i]);
                }
                return node_id;
            }
        case DslNodeKind::Select:
            reserve_children(node, 3);
            push_child(node_id, 0, expr.attr("cond"));
            push_child(node_id, 1, expr.attr("truthy"));
            push_child(node_id, 2, expr.attr("falsy"));
            return node_id;
            SIMJIT_MATCH (DslNodeKind::FpClass) {
                reserve_children(node, 1);
                data.flags = py_enum_value<sj::FpClass>(expr.attr("flags"), "FpClassExpr.flags");
                push_child(node_id, 0, expr.attr("arg"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::Permute) {
                reserve_children(node, 1);
                data.idxs = expr.attr("permute_idxs").cast<uint64_t>();
                data.is_bit = expr.attr("is_bit").cast<bool>();
                push_child(node_id, 0, expr.attr("arg"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::Store) {
                py::object cond = expr.attr("cond");
                data.has_cond = !cond.is_none();
                reserve_children(node, data.has_cond ? 2 : 1);
                data.kind = py_enum_value<sj::LoadStoreKind>(expr.attr("kind"), "StoreExpr.kind");
                push_child(node_id, 0, expr.attr("value"));
                if (data.has_cond) { push_child(node_id, 1, cond); }
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::Scatter) {
                py::object cond = expr.attr("cond");
                data.has_child = !cond.is_none();
                reserve_children(node, data.has_child ? 3 : 2);
                push_child(node_id, 0, expr.attr("value"));
                push_child(node_id, 1, expr.attr("idx"));
                if (data.has_child) { push_child(node_id, 2, cond); }
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::Pack) {
                reserve_children(node, 2);
                py::object dst_size = expr.attr("dst_size");
                if (!dst_size.is_none()) { data.dst_size = copy_string(dst_size, "PackExpr.dst_size"); }
                push_child(node_id, 0, expr.attr("value"));
                push_child(node_id, 1, expr.attr("cond"));
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::ArithAgg) {
                py::object cond = expr.attr("cond");
                data.has_cond = !cond.is_none();
                reserve_children(node, data.has_cond ? 2 : 1);
                data.op = py_enum_value<sj::ArithBinaryOp>(expr.attr("op"), "ArithAggExpr.op");
                push_child(node_id, 0, expr.attr("arg"));
                if (data.has_cond) { push_child(node_id, 1, cond); }
                return node_id;
            }
            SIMJIT_MATCH (DslNodeKind::PredicateAgg) {
                reserve_children(node, 1);
                data.op = py_enum_value<sj::PredicateBinaryOp>(expr.attr("op"), "PredicateAggExpr.op");
                push_child(node_id, 0, expr.attr("arg"));
                return node_id;
            }
        case DslNodeKind::CountIf:
            reserve_children(node, 1);
            push_child(node_id, 0, expr.attr("cond"));
            return node_id;
            SIMJIT_MATCH (DslNodeKind::GroupedArithAgg) {
                py::object cond = expr.attr("cond");
                data.has_cond = !cond.is_none();
                reserve_children(node, data.has_cond ? 3 : 2);
                data.op = py_enum_value<sj::ArithBinaryOp>(expr.attr("op"), "GroupedArithAggExpr.op");
                data.table = copy_string(expr.attr("table"), "GroupedArithAggExpr.table");
                push_child(node_id, 0, expr.attr("arg"));
                push_child(node_id, 1, expr.attr("idx"));
                if (data.has_cond) { push_child(node_id, 2, cond); }
                return node_id;
            }
        }
        SIMJIT_UNREACHABLE();
    }

    void import_output(py::handle item) {
        if (!is_sequence_like(item) || sequence_size(item, "output entry") != 2) {
            throw py::type_error("expected a sequence of (name, Expr) pairs");
        }
        py::object name = sequence_item(item, 0);
        py::object expr = sequence_item(item, 1);
        if (output_cursor >= program.outputs.size()) { throw py::value_error("native DSL output count mismatch"); }
        DslOutput &out = program.outputs[output_cursor++];
        out.name = copy_string_view(name, "output name");
        out.root = append_expr(expr);
    }

    DslProgram finish(size_t expected_output_count) {
        if (output_cursor != expected_output_count) { throw py::value_error("native DSL import count mismatch"); }
        while (!pending_children.empty()) {
            PendingChild child = pending_children.back();
            pending_children.pop_back();
            const DslNode &parent = program.nodes[child.parent];
            SIMJIT_ASSERT(child.index < parent.child_count);
            program.child_edges[parent.first_child + child.index] = append_expr(child.expr);
        }
        program.node_count = node_cursor;
        program.edge_count = edge_cursor;
        program.output_count = output_cursor;
        program.string_count = string_cursor;
        return std::move(program);
    }
};

static DslProgram import_dsl_program(py::handle outputs, const DslImportLimits &limits = {}) {
    size_t output_count = sequence_size(outputs, "outputs");
    DslImportState state{output_count, limits};
    for (size_t i = 0; i < output_count; ++i) {
        py::object item = sequence_item(outputs, i);
        state.import_output(item);
    }
    return state.finish(output_count);
}

static py::str py_string_from_view(std::string_view value) {
    if (value.size() > static_cast<size_t>(std::numeric_limits<Py_ssize_t>::max())) {
        throw py::value_error("string is too large");
    }
    PyObject *obj =
        PyUnicode_FromStringAndSize(value.data() == nullptr ? "" : value.data(), static_cast<Py_ssize_t>(value.size()));
    if (obj == nullptr) { throw py::error_already_set(); }
    return py::reinterpret_steal<py::str>(obj);
}

static ConstPayload parse_sentinel_payload(py::handle value, const LogicalType &ty) {
    ConstPayload payload{};
    if (value.is_none()) { return payload; }
    if (ty.is_float()) {
        payload.kind = ConstPayloadKind::Float;
        payload.float_value = py::float_(py::reinterpret_borrow<py::object>(value)).cast<double>();
        return payload;
    }
    if (ty.dtype == sj::ScalarDataType::I1) {
        payload.kind = ConstPayloadKind::Bool;
        payload.bool_value = value.cast<bool>();
        return payload;
    }
    if (ty.is_unsigned) {
        payload.kind = ConstPayloadKind::UnsignedInt;
        payload.unsigned_int = value.cast<uint64_t>();
        return payload;
    }
    payload.kind = ConstPayloadKind::SignedInt;
    payload.signed_int = value.cast<int64_t>();
    return payload;
}

static NameMap<PyBufferHandle> parse_buffers(const py::dict &buffers) {
    NameMap<PyBufferHandle> result{};
    result.reserve(py::len(buffers));
    for (auto [k, v] : buffers) {
        std::string_view key = py_string_view(k, "buffer name");
        py::buffer buf = v.attr("buf");
        LogicalType ty = parse_type(v.attr("ty"));
        py::buffer_info info = buf.request();
        size_t length = 0;
        if (!v.attr("length").is_none()) {
            length = v.attr("length").cast<size_t>();
        } else if (info.ndim > 0 && !info.shape.empty()) {
            length = info.shape[0];
        } else {
            length = info.size;
        }
        PyBufferHandle::NullEncoding null{};
        py::object null_buf = py::none();
        py::object py_null = v.attr("null");
        if (!py_null.is_none()) {
            std::string_view kind = py_string_view(py_null.attr("kind"), "null kind");
            if (kind == "mask_bitpacked") {
                null.kind = PyBufferHandle::NullKind::MaskBitpacked;
            } else if (kind == "mask_bool") {
                null.kind = PyBufferHandle::NullKind::MaskBool;
            } else if (kind == "sentinel") {
                null.kind = PyBufferHandle::NullKind::Sentinel;
            } else {
                throw py::value_error(sj::format("unsupported null encoding %.*s", SV(kind)));
            }
            null_buf = py_null.attr("buf");
            null.true_means_null = py_null.attr("true_means_null").cast<bool>();
            null.sentinel = parse_sentinel_payload(py_null.attr("sentinel"), ty);
        }
        PyBufferHandle handle{};
        handle.ty = ty;
        handle.buf = buf;
        handle.length = length;
        handle.null = null;
        handle.null_buf = null_buf;
        handle.aligned = v.attr("aligned").cast<bool>();
        handle.bitpacked = v.attr("bitpacked").cast<bool>();
        result.insert_or_assign(key, std::move(handle));
    }
    return result;
}

static NameMap<BufferDesc> buffer_descs_from_handles(const NameMap<PyBufferHandle> &buffers) {
    NameMap<BufferDesc> result{};
    result.reserve(buffers.size());
    buffers.for_each_entry([&](const NameMapEntry<PyBufferHandle> &entry) {
        const BufferDesc &desc = entry.value;
        result.insert_or_assign(entry.name, desc);
    });
    return result;
}

static NameMap<BufferDesc> parse_buffer_descs(const py::dict &buffers) {
    return buffer_descs_from_handles(parse_buffers(buffers));
}

static NameMap<PyBufferHandle> parse_schema_input_handles(py::handle schema) {
    if (PyMapping_Check(schema.ptr()) == 0) { throw py::type_error("inspection schema must be a mapping"); }

    py::dict raw = py::module_::import("builtins").attr("dict")(schema);
    NameMap<PyBufferHandle> result{};
    result.reserve(py::len(raw));
    for (auto [k, v] : raw) {
        std::string_view name = py_string_view(k, "schema name");
        py::handle ty_obj = v;
        bool nullable = false;
        py::object ty_holder;
        if (is_sequence_like(v)) {
            if (sequence_size(v, "schema value") != 2) {
                throw py::type_error("inspection schema values must be a type or (type, nullable)");
            }
            ty_holder = sequence_item(v, 0);
            py::object nullable_obj = sequence_item(v, 1);
            ty_obj = ty_holder;
            nullable = nullable_obj.cast<bool>();
        }

        PyBufferHandle handle{};
        handle.ty = parse_type(ty_obj);
        handle.length = 1;
        handle.aligned = false;
        handle.bitpacked = false;
        if (nullable) {
            handle.null.kind = PyBufferHandle::NullKind::MaskBool;
            handle.null.true_means_null = true;
        }
        result.insert_or_assign(name, std::move(handle));
    }
    return result;
}

static NativeRuntimeOutputKind parse_native_runtime_output_kind(std::string_view output) {
    if (output == "numpy") { return NativeRuntimeOutputKind::Numpy; }
    if (output == "pyarrow") { return NativeRuntimeOutputKind::Arrow; }
    throw py::value_error("output must be either 'numpy' or 'pyarrow'");
}

static std::string_view timestamp_unit_code(TimestampUnit unit) {
    switch (unit) {
    case TimestampUnit::Seconds: return "s";
    case TimestampUnit::Milliseconds: return "ms";
    case TimestampUnit::Microseconds: return "us";
    case TimestampUnit::Nanoseconds: return "ns";
    }
    SIMJIT_UNREACHABLE();
}

static py::object py_timezone(TimestampTimezone timezone) {
    switch (timezone) {
    case TimestampTimezone::None: return py::none();
    case TimestampTimezone::UTC: return py::str("UTC");
    }
    SIMJIT_UNREACHABLE();
}

static py::object py_dsl_type(const LogicalType &ty) {
    const DslModule &dsl = get_dsl();
    if (ty.is_timestamp()) {
        return dsl.mod.attr("timestamp64")(py_string_from_view(timestamp_unit_code(ty.unit)), py_timezone(ty.timezone));
    }
    switch (ty.dtype) {
    case sj::ScalarDataType::I1: return dsl.I1;
    case sj::ScalarDataType::I8: return ty.is_unsigned ? dsl.U8 : dsl.I8;
    case sj::ScalarDataType::I16: return ty.is_unsigned ? dsl.U16 : dsl.I16;
    case sj::ScalarDataType::I32: return ty.is_unsigned ? dsl.U32 : dsl.I32;
    case sj::ScalarDataType::I64: return ty.is_unsigned ? dsl.U64 : dsl.I64;
    case sj::ScalarDataType::F32: return dsl.F32;
    case sj::ScalarDataType::F64: return dsl.F64;
    case sj::ScalarDataType::I128: break;
    }
    throw py::type_error(sj::format("unsupported simjit scalar type %s", ty.name()));
}

struct NativeRuntimeModuleCache {
    py::object numpy{};
    py::object pyarrow{};
    bool pyarrow_checked = false;
    bool pyarrow_available = false;
};

static NativeRuntimeModuleCache &native_runtime_modules() {
    // Match get_dsl(): these Python objects intentionally live until process teardown.
    static auto *cache = new NativeRuntimeModuleCache{};
    return *cache;
}

static py::module_ import_numpy() {
    NativeRuntimeModuleCache &cache = native_runtime_modules();
    if (cache.numpy.ptr() == nullptr) { cache.numpy = py::module_::import("numpy"); }
    return py::reinterpret_borrow<py::module_>(cache.numpy);
}

static py::object try_import_pyarrow() {
    NativeRuntimeModuleCache &cache = native_runtime_modules();
    if (cache.pyarrow_checked) { return cache.pyarrow_available ? cache.pyarrow : py::none(); }

    PyObject *mod = PyImport_ImportModule("pyarrow");
    if (mod == nullptr) {
        if (PyErr_ExceptionMatches(PyExc_ImportError)) {
            PyErr_Clear();
            cache.pyarrow_checked = true;
            cache.pyarrow_available = false;
            return py::none();
        }
        throw py::error_already_set();
    }
    cache.pyarrow = py::reinterpret_steal<py::object>(mod);
    cache.pyarrow_checked = true;
    cache.pyarrow_available = true;
    return cache.pyarrow;
}

static py::module_ import_pyarrow_for_output() {
    py::object pa = try_import_pyarrow();
    if (pa.is_none()) { throw std::runtime_error("pyarrow output requires pyarrow to be installed"); }
    return py::reinterpret_borrow<py::module_>(pa);
}

static bool is_numpy_array(py::handle value) {
    py::module_ np = import_numpy();
    return py::isinstance(value, np.attr("ndarray"));
}

static py::object normalize_transport_buffer(const LogicalType &ty, py::handle buf) {
    if (!ty.is_timestamp() || !is_numpy_array(buf)) { return py::reinterpret_borrow<py::object>(buf); }
    py::object dtype = buf.attr("dtype");
    std::string_view kind = py_string_view(dtype.attr("kind"), "numpy dtype kind");
    if (kind != "M") { return py::reinterpret_borrow<py::object>(buf); }
    py::module_ np = import_numpy();
    return buf.attr("view")(np.attr("int64"));
}

static py::object make_null_encoding(std::string_view kind, py::handle buf, bool true_means_null,
                                     py::handle sentinel = py::none()) {
    return get_dsl().NullEncoding("kind"_a = py_string_from_view(kind), "buf"_a = buf,
                                  "true_means_null"_a = py::bool_(true_means_null), "sentinel"_a = sentinel);
}

static py::object make_buffer_handle(const LogicalType &ty, py::handle buf, size_t length, bool aligned, bool bitpacked,
                                     py::handle null = py::none()) {
    if (bitpacked && ty.dtype != sj::ScalarDataType::I1) {
        throw py::type_error(sj::format("bitpacked transport is only supported for I1, got %s", ty.name()));
    }
    return get_dsl().BufferHandle("ty"_a = py_dsl_type(ty), "buf"_a = normalize_transport_buffer(ty, buf),
                                  "length"_a = length, "aligned"_a = py::bool_(aligned),
                                  "bitpacked"_a = py::bool_(bitpacked), "null"_a = null);
}

static void validate_1d_c_contiguous(std::string_view name, const py::buffer_info &info, std::string_view what) {
    if (info.ndim != 1) { throw py::value_error(sj::format("%.*s %.*s must be a 1D buffer", SV(what), SV(name))); }
    if (!info.strides.empty() && info.shape[0] > 1 && info.strides[0] != info.itemsize) {
        throw py::value_error(sj::format("%.*s %.*s must be C-contiguous", SV(what), SV(name)));
    }
}

static size_t buffer_length(const py::buffer_info &info) {
    if (info.ndim > 0 && !info.shape.empty()) { return static_cast<size_t>(info.shape[0]); }
    return static_cast<size_t>(info.size);
}

static std::string_view normalize_buffer_format(std::string_view fmt) {
    if (fmt.empty()) { throw py::type_error("buffer format is missing"); }
    if (fmt[0] == '@' || fmt[0] == '=' || fmt[0] == '<' || fmt[0] == '>' || fmt[0] == '!') { fmt.remove_prefix(1); }
    return fmt;
}

static LogicalType infer_type_from_buffer_format(std::string_view name, std::string_view fmt) {
    fmt = normalize_buffer_format(fmt);
    if (fmt == "M") {
        throw py::type_error(
            sj::format("input %.*s uses datetime buffer format without explicit source semantics", SV(name)));
    }
    if (fmt == "?") { return LogicalType::simple(sj::ScalarDataType::I1); }
    if (fmt == "b") { return LogicalType::simple(sj::ScalarDataType::I8); }
    if (fmt == "B") { return LogicalType::simple(sj::ScalarDataType::I8, true); }
    if (fmt == "h") { return LogicalType::simple(sj::ScalarDataType::I16); }
    if (fmt == "H") { return LogicalType::simple(sj::ScalarDataType::I16, true); }
    if (fmt == "i") { return LogicalType::simple(sj::ScalarDataType::I32); }
    if (fmt == "I") { return LogicalType::simple(sj::ScalarDataType::I32, true); }
    if (fmt == "q") { return LogicalType::simple(sj::ScalarDataType::I64); }
    if (fmt == "Q") { return LogicalType::simple(sj::ScalarDataType::I64, true); }
    if (fmt == "f") { return LogicalType::simple(sj::ScalarDataType::F32); }
    if (fmt == "d") { return LogicalType::simple(sj::ScalarDataType::F64); }
    throw py::type_error(sj::format("input %.*s uses unsupported buffer format %.*s", SV(name), SV(fmt)));
}

static LogicalType infer_type_from_numpy_dtype(py::handle dtype_obj) {
    py::module_ np = import_numpy();
    py::object dtype = np.attr("dtype")(dtype_obj);
    std::string_view kind = py_string_view(dtype.attr("kind"), "numpy dtype kind");
    if (kind == "M") {
        py::tuple data = np.attr("datetime_data")(dtype).cast<py::tuple>();
        std::string_view unit = py_string_view(data[0], "numpy datetime unit");
        return {sj::ScalarDataType::I64, false, SemanticKind::Timestamp64, parse_timestamp_unit(unit),
                TimestampTimezone::None};
    }
    std::string_view name = py_string_view(dtype.attr("name"), "numpy dtype name");
    if (name == "bool") { return LogicalType::simple(sj::ScalarDataType::I1); }
    if (name == "int8") { return LogicalType::simple(sj::ScalarDataType::I8); }
    if (name == "int16") { return LogicalType::simple(sj::ScalarDataType::I16); }
    if (name == "int32") { return LogicalType::simple(sj::ScalarDataType::I32); }
    if (name == "int64") { return LogicalType::simple(sj::ScalarDataType::I64); }
    if (name == "uint8") { return LogicalType::simple(sj::ScalarDataType::I8, true); }
    if (name == "uint16") { return LogicalType::simple(sj::ScalarDataType::I16, true); }
    if (name == "uint32") { return LogicalType::simple(sj::ScalarDataType::I32, true); }
    if (name == "uint64") { return LogicalType::simple(sj::ScalarDataType::I64, true); }
    if (name == "float32") { return LogicalType::simple(sj::ScalarDataType::F32); }
    if (name == "float64") { return LogicalType::simple(sj::ScalarDataType::F64); }
    throw py::type_error(sj::format("unsupported numpy dtype %.*s", SV(name)));
}

static py::buffer require_buffer(std::string_view name, py::handle value, std::string_view what) {
    if (PyObject_CheckBuffer(value.ptr()) == 0) {
        throw py::type_error(sj::format("%.*s %.*s must support the buffer protocol", SV(what), SV(name)));
    }
    return py::reinterpret_borrow<py::object>(value).cast<py::buffer>();
}

static void validate_null_mask_buffer(std::string_view name, py::handle buf) {
    py::buffer mask = require_buffer(name, buf, "input null mask");
    validate_1d_c_contiguous(name, mask.request(), "input null mask");
}

static py::object validate_null_encoding(std::string_view name, py::handle value) {
    if (value.is_none()) { return py::none(); }
    std::string_view kind = py_string_view(value.attr("kind"), "null kind");
    if (kind == "sentinel") {
        if (value.attr("sentinel").is_none()) {
            throw py::type_error(sj::format("input %.*s sentinel null encoding requires a sentinel value", SV(name)));
        }
        return py::reinterpret_borrow<py::object>(value);
    }
    if (kind != "mask_bitpacked" && kind != "mask_bool") {
        throw py::type_error(sj::format("input %.*s uses unsupported null encoding %.*s", SV(name), SV(kind)));
    }
    py::object buf = value.attr("buf");
    if (buf.is_none()) {
        throw py::type_error(sj::format("input %.*s mask null encoding requires a mask buffer", SV(name)));
    }
    validate_null_mask_buffer(name, buf);
    return py::reinterpret_borrow<py::object>(value);
}

static py::object describe_handle_input(std::string_view name, py::handle value) {
    LogicalType ty = parse_type(value.attr("ty"));
    bool bitpacked = value.attr("bitpacked").cast<bool>();
    if (bitpacked && ty.dtype != sj::ScalarDataType::I1) {
        throw py::type_error(sj::format("input %.*s uses bitpacked transport for non-I1 type %s", SV(name), ty.name()));
    }
    py::object buf_obj = value.attr("buf");
    py::buffer buf = require_buffer(name, buf_obj, "input");
    py::buffer_info info = buf.request();
    validate_1d_c_contiguous(name, info, "input");
    py::object length_obj = value.attr("length");
    size_t length = length_obj.is_none() ? buffer_length(info) : length_obj.cast<size_t>();
    py::object null = validate_null_encoding(name, value.attr("null"));
    return make_buffer_handle(ty, buf_obj, length, value.attr("aligned").cast<bool>(), bitpacked, null);
}

static py::object describe_numpy_input(std::string_view name, py::handle value) {
    LogicalType ty = infer_type_from_numpy_dtype(value.attr("dtype"));
    py::object buf_obj = py::reinterpret_borrow<py::object>(value);
    if (ty.is_timestamp()) { buf_obj = buf_obj.attr("view")(import_numpy().attr("int64")); }
    py::buffer buf = require_buffer(name, buf_obj, "input");
    py::buffer_info info = buf.request();
    validate_1d_c_contiguous(name, info, "input");
    return make_buffer_handle(ty, buf_obj, buffer_length(info), false, false);
}

static LogicalType arrow_type_to_ir(std::string_view name, py::handle arrow_type, py::handle pa) {
    py::object types = pa.attr("types");
    if (types.attr("is_timestamp")(arrow_type).cast<bool>()) {
        TimestampUnit unit = parse_timestamp_unit(py_string_view(arrow_type.attr("unit"), "Arrow timestamp unit"));
        TimestampTimezone timezone = arrow_type.attr("tz").is_none() ? TimestampTimezone::None : TimestampTimezone::UTC;
        return {sj::ScalarDataType::I64, false, SemanticKind::Timestamp64, unit, timezone};
    }
    if (types.attr("is_boolean")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I1); }
    if (types.attr("is_int8")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I8); }
    if (types.attr("is_int16")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I16); }
    if (types.attr("is_int32")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I32); }
    if (types.attr("is_int64")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I64); }
    if (types.attr("is_uint8")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I8, true); }
    if (types.attr("is_uint16")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I16, true); }
    if (types.attr("is_uint32")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I32, true); }
    if (types.attr("is_uint64")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::I64, true); }
    if (types.attr("is_float32")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::F32); }
    if (types.attr("is_float64")(arrow_type).cast<bool>()) { return LogicalType::simple(sj::ScalarDataType::F64); }
    throw py::type_error(sj::format("input %.*s uses unsupported Arrow type %s", SV(name),
                                    py::str(arrow_type).cast<std::string>().c_str()));
}

static py::object describe_pyarrow_array(std::string_view name, py::handle value, py::handle pa) {
    py::object array = py::reinterpret_borrow<py::object>(value);
    if (py::isinstance(array, pa.attr("ChunkedArray"))) {
        size_t chunks = array.attr("num_chunks").cast<size_t>();
        if (chunks != 1) {
            throw py::type_error(sj::format("input %.*s uses a ChunkedArray with %zu chunks", SV(name), chunks));
        }
        array = array.attr("chunk")(0);
    }
    if (!py::isinstance(array, pa.attr("Array"))) {
        throw py::type_error(sj::format("input %.*s is not a supported Arrow array", SV(name)));
    }

    LogicalType ty = arrow_type_to_ir(name, array.attr("type"), pa);
    py::object data_buffer = sequence_item(array.attr("buffers")(), 1);
    if (data_buffer.is_none()) { throw py::type_error(sj::format("input %.*s has no data buffer", SV(name))); }
    if (array.attr("offset").cast<size_t>() != 0) { throw py::type_error("don't support pyarrow.Array.offset"); }

    py::object null = py::none();
    if (array.attr("null_count").cast<size_t>() != 0) {
        py::object validity = sequence_item(array.attr("buffers")(), 0);
        if (validity.is_none()) {
            throw py::type_error(sj::format("input %.*s has null_count but no validity buffer", SV(name)));
        }
        null = make_null_encoding("mask_bitpacked", validity, false);
    }

    size_t length = py::len(array);
    if (ty.dtype == sj::ScalarDataType::I1) { return make_buffer_handle(ty, data_buffer, length, false, true, null); }

    if (!py::hasattr(array.attr("type"), "bit_width")) {
        throw py::type_error(sj::format("input %.*s uses unsupported Arrow type %s", SV(name),
                                        py::str(array.attr("type")).cast<std::string>().c_str()));
    }
    size_t bit_width = array.attr("type").attr("bit_width").cast<size_t>();
    if (bit_width == 0 || bit_width % 8 != 0) {
        throw py::type_error(sj::format("input %.*s uses unsupported Arrow type %s", SV(name),
                                        py::str(array.attr("type")).cast<std::string>().c_str()));
    }
    return make_buffer_handle(ty, data_buffer, length, true, false, null);
}

static py::object describe_buffer_source(std::string_view name, py::handle value) {
    py::buffer buf = require_buffer(name, value, "input");
    py::buffer_info info = buf.request();
    validate_1d_c_contiguous(name, info, "input");
    LogicalType ty = infer_type_from_buffer_format(name, info.format);
    return make_buffer_handle(ty, value, buffer_length(info), false, false);
}

static py::dict normalize_native_runtime_inputs(py::handle inputs) {
    if (PyMapping_Check(inputs.ptr()) == 0) {
        throw py::type_error("Program.run inputs must be a mapping of name to buffer-like");
    }
    py::dict raw = py::module_::import("builtins").attr("dict")(inputs);
    if (py::len(raw) == 0) { throw py::value_error("Program.run requires at least one input buffer"); }

    py::object pa = try_import_pyarrow();
    py::module_ np = import_numpy();
    const DslModule &dsl = get_dsl();
    py::dict handles;
    for (auto [k, v] : raw) {
        std::string_view name = py_string_view(k, "input name");
        py::object handle;
        if (py::isinstance(v, dsl.BufferHandle)) {
            handle = describe_handle_input(name, v);
        } else if (!pa.is_none() &&
                   (py::isinstance(v, pa.attr("Array")) || py::isinstance(v, pa.attr("ChunkedArray")))) {
            handle = describe_pyarrow_array(name, v, pa);
        } else if (py::isinstance(v, np.attr("ndarray"))) {
            handle = describe_numpy_input(name, v);
        } else {
            handle = describe_buffer_source(name, v);
        }
        handles[k] = handle;
    }
    return handles;
}

static void configure_jit_context(sj::jit::JitContext &jit) {
    jit.set_policy(sj::jit::CompilePolicy::BestEffort);
    sj::jit::DebugOptions &debug_opts = jit.debug_options();
    debug_opts.stages = sj::jit::DebugStage::HIR;
}

static sj::jit::CompilePolicy parse_compile_policy(std::string_view value) {
    if (value == "best_effort" || value == "best-effort" || value == "auto") {
        return sj::jit::CompilePolicy::BestEffort;
    }
    if (value == "vectorized" || value == "vector") { return sj::jit::CompilePolicy::Vectorized; }
    if (value == "scalar") { return sj::jit::CompilePolicy::Scalar; }
    throw py::value_error(sj::format("unknown compile policy %.*s", SV(value)));
}

static sj::Arch parse_arch(std::string_view value) {
    if (value == "native" || value == "auto") { return sj::Arch::Native; }
    if (value == "x86" || value == "avx512" || value == "amd64-avx512") { return sj::Arch::Amd64_AVX512; }
    if (value == "x86-ymm" || value == "avx512-ymm" || value == "amd64-avx512-ymm") {
        return sj::Arch::Amd64_AVX512_YMM;
    }
    if (value == "arm" || value == "aarch64" || value == "neon" || value == "arm64-neon") {
        return sj::Arch::Arm64_NEON;
    }
    throw py::value_error(sj::format("unknown arch %.*s", SV(value)));
}

static DslTypeMap make_dsl_type_map(sj::MemoryArena &arena, size_t capacity) {
    return DslTypeMap{arena.alloc_array<DslTypeEntry>(capacity), 0};
}

static DslTypeMap parse_buffer_input_type_map(const py::dict &buffers, sj::MemoryArena &arena, size_t extra_capacity) {
    DslTypeMap out = make_dsl_type_map(arena, py::len(buffers) + extra_capacity);
    for (auto [k, v] : buffers) {
        std::string_view name = py_string_view(k, "buffer name");
        out.insert_or_assign(name, parse_type(v.attr("ty")));
    }
    return out;
}

static DslTypeMap make_buffer_input_type_map(const NameMap<PyBufferHandle> &buffers, sj::MemoryArena &arena,
                                             size_t extra_capacity) {
    DslTypeMap out = make_dsl_type_map(arena, buffers.size() + extra_capacity);
    buffers.for_each_entry(
        [&](const NameMapEntry<PyBufferHandle> &entry) { out.insert_or_assign(entry.name, entry.value.ty); });
    return out;
}

static bool program_has_output_name(const DslProgram &program, std::string_view name) {
    for (uint32_t i = 0; i < program.output_count; ++i) {
        if (program.outputs[i].name == name) { return true; }
    }
    return false;
}

static DslTypeMap parse_buffer_output_type_map(const py::dict &buffers, const DslProgram &program,
                                               sj::MemoryArena &arena) {
    DslTypeMap out = make_dsl_type_map(arena, program.output_count);
    for (auto [k, v] : buffers) {
        std::string_view name = py_string_view(k, "buffer name");
        if (!program_has_output_name(program, name)) { continue; }
        out.insert_or_assign(name, parse_type(v.attr("ty")));
    }
    return out;
}

static DslProgram import_and_resolve_native_program(py::handle outputs, const py::dict &buffers) {
    DslProgram program = import_dsl_program(outputs);
    resolve_native_dsl_program(program, parse_buffer_input_type_map(buffers, program.arena, program.nodes.size()),
                               parse_buffer_output_type_map(buffers, program, program.arena));
    return program;
}

static DslProgram import_and_resolve_native_runtime_program(py::handle outputs,
                                                            const NameMap<PyBufferHandle> &input_buffers) {
    DslProgram program = import_dsl_program(outputs);
    resolve_native_dsl_program(program, make_buffer_input_type_map(input_buffers, program.arena, program.nodes.size()),
                               make_dsl_type_map(program.arena, program.output_count));
    return program;
}

static void append_signature_type(std::string &out, LogicalType ty) {
    sj::format_to(out, "T%d,%d,%d,%d,%d;", static_cast<int>(ty.dtype), static_cast<int>(ty.is_unsigned),
                  static_cast<int>(ty.semantic), static_cast<int>(ty.unit), static_cast<int>(ty.timezone));
}

static void append_signature_type_slot(std::string &out, DslTypeSlot slot) {
    sj::format_to(out, "H%d;", static_cast<int>(slot.has));
    if (slot.has) { append_signature_type(out, slot.type); }
}

static void append_signature_const(std::string &out, ConstPayload payload) {
    sj::format_to(out, "C%d,", static_cast<int>(payload.kind));
    switch (payload.kind) {
    case ConstPayloadKind::None: break;
    case ConstPayloadKind::Bool: sj::format_to(out, "%d", static_cast<int>(payload.bool_value)); break;
    case ConstPayloadKind::SignedInt: sj::format_to(out, "%lld", static_cast<long long>(payload.signed_int)); break;
    case ConstPayloadKind::UnsignedInt:
        sj::format_to(out, "%llu", static_cast<unsigned long long>(payload.unsigned_int));
        break;
    case ConstPayloadKind::Float: {
        uint64_t bits = 0;
        std::memcpy(&bits, &payload.float_value, sizeof(bits));
        sj::format_to(out, "%llx", static_cast<unsigned long long>(bits));
        break;
    }
    }
    out.push_back(';');
}

static void append_signature_sentinel(std::string &out, ConstPayload payload, const LogicalType &ty) {
    if (payload.kind == ConstPayloadKind::None) {
        out.append("V-;");
        return;
    }
    out.append("V+");
    if (ty.is_float()) {
        double d = 0.0;
        if (payload.kind == ConstPayloadKind::Float) {
            d = payload.float_value;
        } else if (payload.kind == ConstPayloadKind::SignedInt) {
            d = static_cast<double>(payload.signed_int);
        } else if (payload.kind == ConstPayloadKind::UnsignedInt) {
            d = static_cast<double>(payload.unsigned_int);
        } else {
            d = payload.bool_value ? 1.0 : 0.0;
        }
        uint64_t bits = 0;
        std::memcpy(&bits, &d, sizeof(bits));
        sj::format_to(out, "%llx", static_cast<unsigned long long>(bits));
    } else if (ty.dtype == sj::ScalarDataType::I1) {
        bool value = payload.kind == ConstPayloadKind::Bool ? payload.bool_value : false;
        sj::format_to(out, "%d", static_cast<int>(value));
    } else if (ty.is_unsigned) {
        uint64_t value = payload.kind == ConstPayloadKind::UnsignedInt ? payload.unsigned_int
                         : payload.kind == ConstPayloadKind::SignedInt ? static_cast<uint64_t>(payload.signed_int)
                                                                       : static_cast<uint64_t>(payload.bool_value);
        sj::format_to(out, "%llu", static_cast<unsigned long long>(value));
    } else {
        int64_t value = payload.kind == ConstPayloadKind::SignedInt     ? payload.signed_int
                        : payload.kind == ConstPayloadKind::UnsignedInt ? static_cast<int64_t>(payload.unsigned_int)
                                                                        : static_cast<int64_t>(payload.bool_value);
        sj::format_to(out, "%lld", static_cast<long long>(value));
    }
    out.push_back(';');
}

struct NativePointerPositions {
    NameMap<size_t> inputs{};
    NameMap<size_t> outputs{};

    explicit NativePointerPositions(const std::vector<NativePointerBinding> &plan) {
        inputs.reserve(plan.size());
        outputs.reserve(plan.size());
        for (const NativePointerBinding &binding : plan) {
            if (binding.null_buffer) { continue; }
            (binding.writable ? outputs : inputs).insert_or_assign(binding.name, binding.slot);
        }
    }

    size_t input(std::string_view name) const { return require(inputs, name, "input"); }
    size_t output(std::string_view name) const { return require(outputs, name, "output"); }
    const size_t *find_output(std::string_view name) const { return outputs.find(name); }

private:
    static size_t require(const NameMap<size_t> &positions, std::string_view name, const char *kind) {
        if (const size_t *position = positions.find(name)) { return *position; }
        throw std::logic_error(sj::format("missing %s pointer position for %.*s", kind, SV(name)));
    }
};

static void append_signature_node_payload(std::string &out, const DslProgram &program, const DslNode &node,
                                          const NativePointerPositions &positions) {
    const DslNode *step = &node;
    switch (node.kind) {
        SIMJIT_MATCH (DslNodeKind::Const) {
            append_signature_const(out, data);
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Load) {
            sj::format_to(out, "L%zu,%d;", positions.input(program.string(data.name)), static_cast<int>(data.kind));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::LoadSplat) {
            sj::format_to(out, "LS%zu;", positions.input(program.string(data.name)));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Gather) {
            sj::format_to(out, "G%zu;", positions.input(program.string(data.name)));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::ArithBinary) {
            sj::format_to(out, "AB%d,%d;", static_cast<int>(data.op), static_cast<int>(data.checked));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::PredicateBinary) {
            sj::format_to(out, "PB%d;", static_cast<int>(data.op));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::ArithUnary) {
            sj::format_to(out, "AU%d,%d;", static_cast<int>(data.op), static_cast<int>(data.checked));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Compare) {
            sj::format_to(out, "CMP%d,%d;", static_cast<int>(data.op), static_cast<int>(data.is_unsigned));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::IntCast) {
            sj::format_to(out, "IC%d,%d;", static_cast<int>(data.kind), static_cast<int>(data.checked));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::FloatCast) {
            sj::format_to(out, "FC%d;", static_cast<int>(data.is_unsigned));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Function) {
            sj::format_to(out, "FN%d,", static_cast<int>(data.kind));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::FpClass) {
            sj::format_to(out, "FPC%u;", static_cast<unsigned>(data.flags));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Permute) {
            sj::format_to(out, "PM%llu,%d;", static_cast<unsigned long long>(data.idxs), static_cast<int>(data.is_bit));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Store) {
            sj::format_to(out, "ST%d,%d;", static_cast<int>(data.kind), static_cast<int>(data.has_cond));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Scatter) {
            sj::format_to(out, "SC%d;", static_cast<int>(data.has_child));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::Pack) {
            if (data.dst_size.is_none()) {
                out.append("PK-;");
            } else {
                sj::format_to(out, "PK%zu;", positions.output(program.string(data.dst_size)));
            }
            return;
        }
        SIMJIT_MATCH (DslNodeKind::ArithAgg) {
            sj::format_to(out, "AA%d,%d;", static_cast<int>(data.op), static_cast<int>(data.has_cond));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::PredicateAgg) {
            sj::format_to(out, "PA%d;", static_cast<int>(data.op));
            return;
        }
        SIMJIT_MATCH (DslNodeKind::GroupedArithAgg) {
            sj::format_to(out, "GA%d,%d,%zu;", static_cast<int>(data.op), static_cast<int>(data.has_cond),
                          positions.output(program.string(data.table)));
            return;
        }
    case DslNodeKind::Index:
    case DslNodeKind::PredicateNot:
    case DslNodeKind::BitCast:
    case DslNodeKind::Select:
    case DslNodeKind::CountIf: out.push_back(';'); return;
    }
    SIMJIT_UNREACHABLE();
}

static void append_signature_buffer(std::string &out, const NativePointerBinding &binding, const BufferDesc &desc) {
    sj::format_to(out, "@%zu,W%d,N%d,", binding.slot, static_cast<int>(binding.writable),
                  static_cast<int>(binding.null_buffer));
    append_signature_type(out, desc.ty);
    sj::format_to(out, "A%d,B%d,K%d,D%d,", static_cast<int>(desc.aligned), static_cast<int>(desc.bitpacked),
                  static_cast<int>(desc.null.kind), static_cast<int>(desc.null.true_means_null));
    if (desc.null.kind == BufferNullKind::Sentinel) { append_signature_sentinel(out, desc.null.sentinel, desc.ty); }
    out.push_back('|');
}

static std::string make_resolved_native_function_identifier(const DslProgram &program,
                                                            const NameMap<BufferDesc> &buffers,
                                                            const std::vector<NativePointerBinding> &pointer_plan,
                                                            NativeRuntimeOutputKind output_kind,
                                                            sj::jit::CompilePolicy policy,
                                                            sj::CodeTransformations transformations) {
    std::vector<NativePointerBinding> sorted_bindings = pointer_plan;
    std::sort(sorted_bindings.begin(), sorted_bindings.end(),
              [](const NativePointerBinding &lhs, const NativePointerBinding &rhs) { return lhs.slot < rhs.slot; });
    NativePointerPositions positions(pointer_plan);

    std::string out;
    out.reserve(program.node_count * 48 + program.edge_count * 8 + program.output_count * 32 +
                sorted_bindings.size() * 48 + 96);
    out.append("native:struct:v2|");
    sj::format_to(out, "policy=%d|transformations=%u|output_kind=%d|", static_cast<int>(policy),
                  static_cast<unsigned>(transformations), static_cast<int>(output_kind));
    sj::format_to(out, "outputs=%u|", static_cast<unsigned>(program.output_count));
    for (uint32_t i = 0; i < program.output_count; ++i) {
        const DslOutput &item = program.outputs[i];
        if (const size_t *position = positions.find_output(item.name)) {
            sj::format_to(out, "@%zu:%u;", *position, static_cast<unsigned>(item.root));
        } else {
            sj::format_to(out, "@-:%u;", static_cast<unsigned>(item.root));
        }
    }
    sj::format_to(out, "|nodes=%u|", static_cast<unsigned>(program.node_count));
    for (uint32_t i = 0; i < program.node_count; ++i) {
        const DslNode &node = program.nodes[i];
        sj::format_to(out, "#%u:%d,", static_cast<unsigned>(i), static_cast<int>(node.kind));
        append_signature_type_slot(out, program.resolved_types[i]);
        append_signature_node_payload(out, program, node, positions);
        sj::format_to(out, "[%u:", static_cast<unsigned>(node.child_count));
        for (uint32_t child = 0; child < node.child_count; ++child) {
            sj::format_to(out, "%u,", static_cast<unsigned>(program.child_edges[node.first_child + child]));
        }
        out.append("]");
    }
    out.append("|buffers=");
    for (const NativePointerBinding &binding : sorted_bindings) {
        const BufferDesc *desc = buffers.find(binding.name);
        if (desc == nullptr) {
            throw std::logic_error(sj::format("missing buffer descriptor for pointer position %zu", binding.slot));
        }
        append_signature_buffer(out, binding, *desc);
    }
    return out;
}

static std::string make_resolved_native_function_identifier(const DslProgram &program, const py::dict &buffers,
                                                            const std::vector<NativePointerBinding> &pointer_plan,
                                                            NativeRuntimeOutputKind output_kind,
                                                            sj::jit::CompilePolicy policy,
                                                            sj::CodeTransformations transformations) {
    return make_resolved_native_function_identifier(program, parse_buffer_descs(buffers), pointer_plan, output_kind,
                                                    policy, transformations);
}

[[noreturn]] static void translate_native_builder_error(const NativeBuilderError &error) {
    switch (error.kind()) {
    case NativeBuilderErrorKind::Type: throw py::type_error(error.what());
    case NativeBuilderErrorKind::Value: throw py::value_error(error.what());
    case NativeBuilderErrorKind::Index: throw py::index_error(error.what());
    }
    SIMJIT_UNREACHABLE();
}

template <typename Fn> static auto run_native_builder_call(Fn &&fn) -> decltype(fn()) {
    try {
        return fn();
    } catch (const NativeBuilderError &error) { translate_native_builder_error(error); }
}

static const PyBufferHandle &require_bound_buffer(const NameMap<PyBufferHandle> &buffers, std::string_view name) {
    if (const PyBufferHandle *buf = buffers.find(name)) { return *buf; }
    throw py::value_error(sj::format("buffer %.*s is missing", SV(name)));
}

static std::vector<void *> materialize_pointer_table(const NameMap<PyBufferHandle> &buffers,
                                                     const std::vector<NativePointerBinding> &plan) {
    size_t slot_count = 0;
    for (const NativePointerBinding &binding : plan) {
        slot_count = std::max(slot_count, binding.slot + 1);
    }

    std::vector<void *> ptrs(slot_count, nullptr);
    for (const NativePointerBinding &binding : plan) {
        if (ptrs[binding.slot] != nullptr) {
            const char *kind = binding.null_buffer ? "null argument" : "argument";
            throw py::value_error(sj::format("conflicting %s name %.*s", kind, SV(binding.name)));
        }

        const PyBufferHandle &handle = require_bound_buffer(buffers, binding.name);
        py::buffer buf = binding.null_buffer ? handle.null_buf.cast<py::buffer>() : handle.buf;
        ptrs[binding.slot] = binding.writable ? buf.request(true).ptr : buf.request().ptr;
    }
    return ptrs;
}

static std::vector<NativePointerBinding>
lower_resolved_native_and_collect_pointer_plan(const NameMap<BufferDesc> &buffers, const DslProgram &program, size_t n,
                                               sj::FunctionBuilder &builder) {
    return run_native_builder_call([&]() { return build_native_pointer_plan(builder, buffers, program, n); });
}

static std::vector<void *>
build_resolved_native_pointer_table(const py::dict &buffers, const DslProgram &program, size_t n,
                                    std::vector<NativePointerBinding> *pointer_plan_out = nullptr) {
    NameMap<PyBufferHandle> parsed_buffers = parse_buffers(buffers);
    NameMap<BufferDesc> buffer_descs = buffer_descs_from_handles(parsed_buffers);
    sj::MemoryArena arena{};
    sj::Context ctx(arena);
    sj::FunctionBuilder builder(ctx);
    std::vector<NativePointerBinding> plan =
        lower_resolved_native_and_collect_pointer_plan(buffer_descs, program, n, builder);
    std::vector<void *> ptrs = materialize_pointer_table(parsed_buffers, plan);
    if (pointer_plan_out != nullptr) { *pointer_plan_out = std::move(plan); }
    return ptrs;
}

static void *build_resolved_native_function(sj::jit::JitContext &jit, const py::dict &buffers,
                                            const DslProgram &program, size_t n, const std::string &identifier) {
    NameMap<BufferDesc> buffer_descs = parse_buffer_descs(buffers);
    return jit.build_and_compile(identifier, [&](sj::FunctionBuilder &builder) {
        lower_resolved_native_and_collect_pointer_plan(buffer_descs, program, n, builder);
    });
}

#if SIMJIT_ASMJIT_BACKEND
static double measure_asmjit_backend_compile_us(const sj::mir::Function *mir, sj::Arch arch) {
    sj::AsmjitSession warmup_session(arch);
    sj::AsmjitCompileOptions warmup_opts{};
    warmup_opts.session = &warmup_session;
    sj::AsmjitCompileResult warmup_result{};
    sj::compile_asmjit(mir, warmup_opts, warmup_result);

    sj::AsmjitSession timing_session(arch);
    sj::AsmjitCompileOptions timing_opts{};
    timing_opts.session = &timing_session;
    sj::AsmjitCompileResult timing_result{};

    auto t1 = std::chrono::high_resolution_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    sj::compile_asmjit(mir, timing_opts, timing_result);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::micro> us_double = t2 - t1;
    return us_double.count();
}
#endif

static py::dict inspect_hir_function(sj::hir::Function *hir, sj::jit::CompilePolicy policy, sj::Arch arch) {
    py::list diagnostics{};
    py::dict out{};
    out["hir"] = sj::hir::print_function(hir);
#if SIMJIT_ENABLE_SERIALIZATION
    out["serialized"] = sj::serialize(hir);
#else
    out["serialized"] = "";
    diagnostics.append("serialization backend is not enabled");
#endif

    std::string vectorization_exception{};
    std::string vectorizer_text{};
    sj::mir::Function *mir = nullptr;
    if (policy == sj::jit::CompilePolicy::Scalar) {
        mir = sj::lower_scalar(hir);
    } else {
        auto vec = sj::vect::try_hir_to_vect(hir);
        if (vec) {
            vectorizer_text = sj::vect::print_function(vec.value());
            mir = sj::mir::vect_to_mir(vec.value());
        } else {
            vectorization_exception = vec.error().verbose();
            if (policy == sj::jit::CompilePolicy::Vectorized) { throw py::value_error(vectorization_exception); }
            diagnostics.append(vectorization_exception);
            mir = sj::lower_scalar(hir);
        }
    }

    out["vectorizer"] = vectorizer_text;
    out["vectorization_exception"] = vectorization_exception;
    out["mir"] = sj::mir::print_function(mir);

#if SIMJIT_ASMJIT_BACKEND
    try {
        out["asmjit_compile_us"] = measure_asmjit_backend_compile_us(mir, arch);
        sj::AsmjitCompileOptions asm_opts{false, true, nullptr};
        sj::AsmjitCompileResult asm_result{};
        sj::compile_asmjit(mir, asm_opts, asm_result);
        out["asm_code"] = std::move(asm_result.asm_code);
    } catch (const std::exception &e) {
        out["asmjit_compile_us"] = py::none();
        out["asm_code"] = "";
        diagnostics.append(sj::format("AsmJit output failed: %s", e.what()));
    }
#else
    out["asmjit_compile_us"] = py::none();
    out["asm_code"] = "";
    diagnostics.append("AsmJit backend is not enabled");
#endif

#if SIMJIT_CPP_BACKEND
    try {
        out["cpp"] = sj::emit_cpp_source(mir);
    } catch (const std::exception &e) {
        try {
            sj::mir::Function *scalar_mir = sj::lower_scalar(hir);
            out["cpp"] = sj::emit_cpp_source(scalar_mir);
            diagnostics.append(sj::format("C++ output fell back to scalar MIR: %s", e.what()));
        } catch (const std::exception &fallback) {
            out["cpp"] = "";
            diagnostics.append(sj::format("C++ output failed: %s", fallback.what()));
        }
    }
#else
    out["cpp"] = "";
    diagnostics.append("C++ backend is not enabled");
#endif

#if SIMJIT_LLVM_BACKEND
    try {
        out["llvm_ir"] = sj::emit_llvm_ir(mir);
    } catch (const std::exception &e) {
        out["llvm_ir"] = "";
        diagnostics.append(sj::format("LLVM output failed: %s", e.what()));
    }
#else
    out["llvm_ir"] = "";
    diagnostics.append("LLVM backend is not enabled");
#endif
    out["diagnostics"] = diagnostics;
    return out;
}

} // namespace

py::dict inspect_native_function(const py::dict &buffers, py::handle outputs, size_t n, std::string_view policy_name,
                                 std::string_view arch_name) {
    sj::MemoryArena arena{};
    sj::Arch arch = parse_arch(arch_name);
    sj::Context ctx(arena, "expr", sj::CodeTransformations::All, arch);
    sj::FunctionBuilder builder(ctx);
    DslProgram program = import_and_resolve_native_program(outputs, buffers);
    lower_resolved_native_and_collect_pointer_plan(parse_buffer_descs(buffers), program, n, builder);

    sj::hir::Function *hir = builder.build();
    sj::jit::CompilePolicy policy = parse_compile_policy(policy_name);
    return inspect_hir_function(hir, policy, arch);
}

py::dict inspect_serialized_function(std::string_view serialized, std::string_view policy_name,
                                     std::string_view arch_name) {
#if SIMJIT_ENABLE_SERIALIZATION
    sj::MemoryArena arena{};
    sj::Arch arch = parse_arch(arch_name);
    sj::Context ctx(arena, "expr", sj::CodeTransformations::All, arch);
    sj::FunctionBuilder builder(ctx);
    sj::deserialize(serialized, builder);
    sj::hir::Function *hir = builder.build();
    sj::jit::CompilePolicy policy = parse_compile_policy(policy_name);
    return inspect_hir_function(hir, policy, arch);
#else
    throw py::value_error("serialization backend is not enabled");
#endif
}

namespace {

static void execute_function(void *fn, size_t n, const std::vector<void *> &ptrs) {
    sj::jit::call_fn_ptr(fn, n, sj::nonstd::span<void *>{const_cast<void **>(ptrs.data()), ptrs.size()});
}

struct NativeRuntimeOutput {
    std::string_view name{};
    py::object data{};
    py::object validity{};
    LogicalType ty{};
    size_t length = 0;
    bool scalar = false;
    void *raw_data = nullptr;
};

struct NativeRuntimePlan {
    DslProgram program{};
    py::dict buffers{};
    std::vector<NativeRuntimeOutput> outputs{};
    py::object safety_check_buffer{};
    size_t n = 0;
    bool has_safety_check = false;
    NativeRuntimeOutputKind output_kind = NativeRuntimeOutputKind::Numpy;
};

static void copy_dict_items(py::dict &dst, const py::dict &src) {
    for (auto [k, v] : src) {
        dst[k] = v;
    }
}

static py::object numpy_dtype(const LogicalType &ty, py::handle np) {
    if (ty.is_timestamp()) {
        switch (ty.unit) {
        case TimestampUnit::Seconds: return np.attr("dtype")("datetime64[s]");
        case TimestampUnit::Milliseconds: return np.attr("dtype")("datetime64[ms]");
        case TimestampUnit::Microseconds: return np.attr("dtype")("datetime64[us]");
        case TimestampUnit::Nanoseconds: return np.attr("dtype")("datetime64[ns]");
        }
        SIMJIT_UNREACHABLE();
    }
    switch (ty.dtype) {
    case sj::ScalarDataType::I1: return np.attr("bool_");
    case sj::ScalarDataType::I8: return ty.is_unsigned ? np.attr("uint8") : np.attr("int8");
    case sj::ScalarDataType::I16: return ty.is_unsigned ? np.attr("uint16") : np.attr("int16");
    case sj::ScalarDataType::I32: return ty.is_unsigned ? np.attr("uint32") : np.attr("int32");
    case sj::ScalarDataType::I64: return ty.is_unsigned ? np.attr("uint64") : np.attr("int64");
    case sj::ScalarDataType::F32: return np.attr("float32");
    case sj::ScalarDataType::F64: return np.attr("float64");
    case sj::ScalarDataType::I128: break;
    }
    throw py::type_error(sj::format("unsupported simjit scalar type %s", ty.name()));
}

static py::object pyarrow_type(const LogicalType &ty, py::handle pa) {
    if (ty.is_timestamp()) {
        return pa.attr("timestamp")(py_string_from_view(timestamp_unit_code(ty.unit)),
                                    "tz"_a = py_timezone(ty.timezone));
    }
    switch (ty.dtype) {
    case sj::ScalarDataType::I1: return pa.attr("bool_")();
    case sj::ScalarDataType::I8: return ty.is_unsigned ? pa.attr("uint8")() : pa.attr("int8")();
    case sj::ScalarDataType::I16: return ty.is_unsigned ? pa.attr("uint16")() : pa.attr("int16")();
    case sj::ScalarDataType::I32: return ty.is_unsigned ? pa.attr("uint32")() : pa.attr("int32")();
    case sj::ScalarDataType::I64: return ty.is_unsigned ? pa.attr("uint64")() : pa.attr("int64")();
    case sj::ScalarDataType::F32: return pa.attr("float32")();
    case sj::ScalarDataType::F64: return pa.attr("float64")();
    case sj::ScalarDataType::I128: break;
    }
    throw py::type_error(sj::format("unsupported simjit scalar type %s", ty.name()));
}

static void fill_writable_buffer(py::handle buf, unsigned char byte) {
    py::buffer buffer = py::reinterpret_borrow<py::object>(buf).cast<py::buffer>();
    py::buffer_info info = buffer.request(true);
    std::memset(info.ptr, byte, static_cast<size_t>(info.size) * static_cast<size_t>(info.itemsize));
}

static NativeRuntimeOutput allocate_numpy_output(py::dict &buffers, const NativeRuntimeOutputSpec &spec, py::handle np,
                                                 bool bind_output) {
    py::object array = np.attr("empty")(py::int_(spec.length), numpy_dtype(spec.ty, np));
    if (bind_output) {
        buffers[py_string_from_view(spec.name)] = make_buffer_handle(spec.ty, array, spec.length, false, false);
    }
    return NativeRuntimeOutput{spec.name, array, py::none(), spec.ty, spec.length, spec.scalar, nullptr};
}

static NativeRuntimeOutput allocate_pyarrow_output(py::dict &buffers, const NativeRuntimeOutputSpec &spec,
                                                   py::handle pa, bool bind_output) {
    py::object data = pa.attr("allocate_buffer")(py::int_(spec.data_bytes));
    if (spec.bitpacked) { fill_writable_buffer(data, 0); }

    py::object validity = py::none();
    py::object null = py::none();
    if (spec.nullable) {
        validity = pa.attr("allocate_buffer")(py::int_(spec.validity_bytes));
        fill_writable_buffer(validity, 0xFF);
        null = make_null_encoding("mask_bitpacked", validity, false);
    }

    if (bind_output) {
        buffers[py_string_from_view(spec.name)] =
            make_buffer_handle(spec.ty, data, spec.length, true, spec.bitpacked, null);
    }
    return NativeRuntimeOutput{spec.name, data, validity, spec.ty, spec.length, spec.scalar, nullptr};
}

static NativeRuntimeOutput allocate_native_scalar_output(const NativeRuntimeOutputSpec &spec, void *storage) {
    return NativeRuntimeOutput{spec.name, py::none(), py::none(), spec.ty, 1, true, storage};
}

static void allocate_native_runtime_outputs(py::dict &buffers, std::vector<NativeRuntimeOutput> &outputs,
                                            NativeRuntimeOutputKind output_kind,
                                            const std::vector<NativeRuntimeOutputSpec> &specs, py::handle output_module,
                                            bool bind_outputs) {
    outputs.reserve(specs.size());
    for (const NativeRuntimeOutputSpec &spec : specs) {
        if (output_kind == NativeRuntimeOutputKind::Numpy) {
            outputs.push_back(allocate_numpy_output(buffers, spec, output_module, bind_outputs));
        } else {
            outputs.push_back(allocate_pyarrow_output(buffers, spec, output_module, bind_outputs));
        }
    }
}

static void allocate_native_runtime_outputs(py::dict &buffers, std::vector<NativeRuntimeOutput> &outputs,
                                            const DslProgram &program, NativeRuntimeOutputKind output_kind, size_t n,
                                            const NameMap<BufferDesc> &inputs) {
    py::object output_module = output_kind == NativeRuntimeOutputKind::Numpy ? py::object(import_numpy())
                                                                             : py::object(import_pyarrow_for_output());
    allocate_native_runtime_outputs(buffers, outputs, output_kind,
                                    plan_native_runtime_outputs(program, inputs, output_kind, n), output_module, true);
}

static void allocate_native_runtime_outputs(NativeRuntimePlan &plan, const NameMap<BufferDesc> &inputs) {
    allocate_native_runtime_outputs(plan.buffers, plan.outputs, plan.program, plan.output_kind, plan.n, inputs);
}

static void allocate_native_safety_check(NativeRuntimePlan &plan) {
    if (!native_program_requires_safety_check(plan.program)) { return; }
    py::str name = py_string_from_view(NATIVE_SAFETY_CHECK_BUFFER);
    if (plan.buffers.contains(name) || program_has_output_name(plan.program, NATIVE_SAFETY_CHECK_BUFFER)) {
        throw py::value_error("reserved internal safety-check buffer name is in use");
    }
    static constexpr char ZERO = 0;
    plan.safety_check_buffer = py::bytearray(&ZERO, 1);
    plan.buffers[name] =
        make_buffer_handle(LogicalType::simple(sj::ScalarDataType::I8), plan.safety_check_buffer, 1, false, false);
    plan.has_safety_check = true;
}

static void reset_native_safety_check(py::handle buffer) {
    py::buffer_info info = buffer.cast<py::buffer>().request(true);
    *static_cast<uint8_t *>(info.ptr) = 0;
}

static void check_native_safety_check(py::handle buffer) {
    py::buffer_info info = buffer.cast<py::buffer>().request();
    if (*static_cast<const uint8_t *>(info.ptr) != 0) { throw SafetyCheckFailed{}; }
}

static size_t choose_native_execution_length(const DslProgram &program, const NameMap<BufferDesc> &inputs) {
    bool has_driver = false;
    size_t length = 0;
    for (uint32_t i = 0; i < program.node_count; ++i) {
        const DslNode &node = program.nodes[i];
        if (node.kind != DslNodeKind::Load) { continue; }
        const BufferDesc *buf = inputs.find(program.string(node.step_data<DslNodeKind::Load>().name));
        if (buf == nullptr) { continue; }
        if (!has_driver) {
            length = buf->length;
            has_driver = true;
            continue;
        }
        if (buf->length != length) {
            throw std::invalid_argument("all vector-driving input buffers must have the same length");
        }
    }
    if (has_driver) { return length; }
    if (inputs.empty()) { throw std::invalid_argument("Program.run requires at least one input buffer"); }
    inputs.for_each_entry([&](const NameMapEntry<BufferDesc> &entry) {
        if (!has_driver) {
            length = entry.value.length;
            has_driver = true;
        }
    });
    return length;
}

static NativeRuntimePlan prepare_native_runtime_plan(py::handle outputs, py::handle inputs,
                                                     std::string_view output_kind) {
    NativeRuntimePlan plan{};
    plan.output_kind = parse_native_runtime_output_kind(output_kind);
    py::dict input_handles = normalize_native_runtime_inputs(inputs);
    NameMap<PyBufferHandle> input_map = parse_buffers(input_handles);
    NameMap<BufferDesc> input_descs = buffer_descs_from_handles(input_map);
    plan.program = import_and_resolve_native_runtime_program(outputs, input_map);
    plan.n = choose_native_execution_length(plan.program, input_descs);
    copy_dict_items(plan.buffers, input_handles);
    allocate_native_runtime_outputs(plan, input_descs);
    allocate_native_safety_check(plan);
    return plan;
}

static py::object finalize_numpy_output(const NativeRuntimeOutput &out) {
    if (!out.scalar) { return out.data; }
    if (out.raw_data != nullptr) {
        switch (out.ty.dtype) {
        case sj::ScalarDataType::I1: return py::bool_(*static_cast<const bool *>(out.raw_data));
        case sj::ScalarDataType::I8:
            return out.ty.is_unsigned ? py::object(py::int_(*static_cast<const uint8_t *>(out.raw_data)))
                                      : py::object(py::int_(*static_cast<const int8_t *>(out.raw_data)));
        case sj::ScalarDataType::I16:
            return out.ty.is_unsigned ? py::object(py::int_(*static_cast<const uint16_t *>(out.raw_data)))
                                      : py::object(py::int_(*static_cast<const int16_t *>(out.raw_data)));
        case sj::ScalarDataType::I32:
            return out.ty.is_unsigned ? py::object(py::int_(*static_cast<const uint32_t *>(out.raw_data)))
                                      : py::object(py::int_(*static_cast<const int32_t *>(out.raw_data)));
        case sj::ScalarDataType::I64:
            return out.ty.is_unsigned ? py::object(py::int_(*static_cast<const uint64_t *>(out.raw_data)))
                                      : py::object(py::int_(*static_cast<const int64_t *>(out.raw_data)));
        case sj::ScalarDataType::F32: return py::float_(*static_cast<const float *>(out.raw_data));
        case sj::ScalarDataType::F64: return py::float_(*static_cast<const double *>(out.raw_data));
        case sj::ScalarDataType::I128: break;
        }
        throw py::type_error(sj::format("unsupported simjit scalar type %s", out.ty.name()));
    }
    py::object item = out.data.attr("__getitem__")(0);
    if (out.ty.is_timestamp()) { return item; }
    return item.attr("item")();
}

static py::object finalize_pyarrow_output(const NativeRuntimeOutput &out, py::handle pa) {
    py::list buffers;
    buffers.append(out.validity);
    buffers.append(out.data);
    py::object array = pa.attr("Array").attr("from_buffers")(pyarrow_type(out.ty, pa), py::int_(out.length), buffers);
    if (out.scalar) { return array.attr("__getitem__")(0); }
    return array;
}

static py::dict finalize_native_runtime_outputs(NativeRuntimeOutputKind output_kind,
                                                const std::vector<NativeRuntimeOutput> &outputs) {
    py::dict results;
    py::object pa = py::none();
    if (output_kind == NativeRuntimeOutputKind::Arrow) { pa = import_pyarrow_for_output(); }
    for (const NativeRuntimeOutput &out : outputs) {
        if (output_kind == NativeRuntimeOutputKind::Numpy) {
            results[py_string_from_view(out.name)] = finalize_numpy_output(out);
        } else {
            results[py_string_from_view(out.name)] = finalize_pyarrow_output(out, pa);
        }
    }
    return results;
}

static py::tuple finalize_native_runtime_output_values(NativeRuntimeOutputKind output_kind,
                                                       const std::vector<NativeRuntimeOutput> &outputs) {
    py::tuple results(outputs.size());
    py::object pa = py::none();
    if (output_kind == NativeRuntimeOutputKind::Arrow) { pa = import_pyarrow_for_output(); }
    for (size_t i = 0; i < outputs.size(); ++i) {
        results[i] = output_kind == NativeRuntimeOutputKind::Numpy ? finalize_numpy_output(outputs[i])
                                                                   : finalize_pyarrow_output(outputs[i], pa);
    }
    return results;
}

static py::dict inspect_native_runtime_plan(const NativeRuntimePlan &plan, std::string_view policy_name,
                                            std::string_view arch_name) {
    sj::MemoryArena arena{};
    sj::Arch arch = parse_arch(arch_name);
    sj::Context ctx(arena, "expr", sj::CodeTransformations::All, arch);
    sj::FunctionBuilder builder(ctx);
    lower_resolved_native_and_collect_pointer_plan(parse_buffer_descs(plan.buffers), plan.program, plan.n, builder);

    sj::hir::Function *hir = builder.build();
    sj::jit::CompilePolicy policy = parse_compile_policy(policy_name);
    return inspect_hir_function(hir, policy, arch);
}

static void add_schema_output_handles(NameMap<PyBufferHandle> &buffers, const DslProgram &program,
                                      const NameMap<BufferDesc> &inputs, NativeRuntimeOutputKind output_kind,
                                      size_t n) {
    std::vector<NativeRuntimeOutputSpec> specs = plan_native_runtime_outputs(program, inputs, output_kind, n);
    for (const NativeRuntimeOutputSpec &spec : specs) {
        PyBufferHandle handle{};
        handle.ty = spec.ty;
        handle.length = spec.length;
        handle.aligned = output_kind == NativeRuntimeOutputKind::Arrow;
        handle.bitpacked = spec.bitpacked;
        if (spec.nullable) {
            handle.null.kind = PyBufferHandle::NullKind::MaskBitpacked;
            handle.null.true_means_null = false;
        }
        buffers.insert_or_assign(spec.name, std::move(handle));
    }
}

} // namespace

py::dict inspect_program_function(py::handle outputs, py::handle inputs, std::string_view output_kind,
                                  std::string_view policy_name, std::string_view arch_name) {
    NativeRuntimePlan plan = prepare_native_runtime_plan(outputs, inputs, output_kind);
    return inspect_native_runtime_plan(plan, policy_name, arch_name);
}

py::dict inspect_schema_function(py::handle outputs, py::handle schema, std::string_view output_kind,
                                 std::string_view policy_name, std::string_view arch_name) {
    NativeRuntimeOutputKind parsed_output_kind = parse_native_runtime_output_kind(output_kind);
    NameMap<PyBufferHandle> buffers = parse_schema_input_handles(schema);
    NameMap<BufferDesc> input_descs = buffer_descs_from_handles(buffers);
    DslProgram program = import_and_resolve_native_runtime_program(outputs, buffers);
    constexpr size_t n = 1;
    add_schema_output_handles(buffers, program, input_descs, parsed_output_kind, n);
    if (native_program_requires_safety_check(program)) {
        if (buffers.find(NATIVE_SAFETY_CHECK_BUFFER) != nullptr ||
            program_has_output_name(program, NATIVE_SAFETY_CHECK_BUFFER)) {
            throw py::value_error("reserved internal safety-check buffer name is in use");
        }
        PyBufferHandle handle{};
        handle.ty = LogicalType::simple(sj::ScalarDataType::I8);
        handle.length = 1;
        buffers.insert_or_assign(NATIVE_SAFETY_CHECK_BUFFER, std::move(handle));
    }

    sj::MemoryArena arena{};
    sj::Arch arch = parse_arch(arch_name);
    sj::Context ctx(arena, "expr", sj::CodeTransformations::All, arch);
    sj::FunctionBuilder builder(ctx);
    lower_resolved_native_and_collect_pointer_plan(buffer_descs_from_handles(buffers), program, n, builder);

    sj::hir::Function *hir = builder.build();
    sj::jit::CompilePolicy policy = parse_compile_policy(policy_name);
    return inspect_hir_function(hir, policy, arch);
}

py::dict benchmark_hir_jit_compile(py::handle outputs, py::handle inputs, std::string_view output_kind,
                                   std::string_view backend, sj::jit::CompilePolicy policy, std::string_view llvm_opt,
                                   std::string_view arch_name, int warmups, int runs) {
    if (warmups < 0) { throw py::value_error("warmups must be non-negative"); }
    if (runs <= 0) { throw py::value_error("runs must be positive"); }
    if (backend != "asmjit" && backend != "llvm") { throw py::value_error("backend must be asmjit or llvm"); }
    if (llvm_opt != "O1" && llvm_opt != "O3") { throw py::value_error("llvm_opt must be O1 or O3"); }

    NativeRuntimePlan plan = prepare_native_runtime_plan(outputs, inputs, output_kind);
    sj::Arch arch = parse_arch(arch_name);

    auto measure_once = [&]() -> double {
        sj::MemoryArena arena{};
        sj::Context ctx(arena, "expr", sj::CodeTransformations::All, arch);
        sj::FunctionBuilder builder(ctx);
        lower_resolved_native_and_collect_pointer_plan(parse_buffer_descs(plan.buffers), plan.program, plan.n, builder);
        sj::hir::Function *hir = builder.build();
        if (hir == nullptr) { throw std::runtime_error("HIR construction returned null"); }

        void *fn = nullptr;
        double elapsed_us = 0.0;
        if (backend == "asmjit") {
            sj::AsmjitSession session(arch);
            auto start = std::chrono::steady_clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            sj::mir::Function *mir = nullptr;
            switch (policy) {
            case sj::jit::CompilePolicy::Scalar: mir = sj::lower_scalar(hir); break;
            case sj::jit::CompilePolicy::Vectorized: mir = sj::lower_vectorized(hir); break;
            case sj::jit::CompilePolicy::BestEffort: {
                auto vectorized = sj::try_lower_vectorized(hir);
                mir = vectorized ? vectorized.value() : sj::lower_scalar(hir);
                break;
            }
            }
            sj::AsmjitCompileOptions options{};
            options.emit_machine_code = false;
            options.emit_asm_code = false;
            options.session = &session;
            sj::AsmjitCompileResult result{};
            sj::compile_asmjit(mir, options, result);
            fn = session.add_compiled_function();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            auto finish = std::chrono::steady_clock::now();
            elapsed_us = std::chrono::duration<double, std::micro>(finish - start).count();
        } else {
#if SIMJIT_LLVM_BACKEND
            auto level = llvm_opt == "O1" ? sj::llvm_backend::LLVMOptLevel::O1 : sj::llvm_backend::LLVMOptLevel::O3;
            sj::llvm_backend::LLVMSession session(arch, level);
            auto start = std::chrono::steady_clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            fn = sj::llvm_backend::compile_hir(hir, session, policy);
            std::atomic_signal_fence(std::memory_order_seq_cst);
            auto finish = std::chrono::steady_clock::now();
            elapsed_us = std::chrono::duration<double, std::micro>(finish - start).count();
#else
            throw py::value_error("LLVM JIT benchmark requires an LLVM-enabled Python extension");
#endif
        }

        if (fn == nullptr) { throw std::runtime_error("JIT compile returned a null executable function pointer"); }
        return elapsed_us;
    };

    for (int i = 0; i < warmups; ++i) {
        (void)measure_once();
    }

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(runs));
    for (int i = 0; i < runs; ++i) {
        timings.push_back(measure_once());
    }
    std::vector<double> sorted = timings;
    std::sort(sorted.begin(), sorted.end());
    size_t mid = sorted.size() / 2;
    double median = (sorted.size() % 2) == 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2.0;

    py::dict result;
    result["compile_us"] = median;
    result["compile_samples_us"] = timings;
    result["compile_warmups"] = warmups;
    result["compile_runs"] = runs;
    result["compile_boundary"] = "constructed-hir-to-executable-pointer";
    result["backend"] = backend;
    result["policy"] = policy == sj::jit::CompilePolicy::Scalar       ? "scalar"
                       : policy == sj::jit::CompilePolicy::Vectorized ? "vectorized"
                                                                      : "best_effort";
    if (backend == "llvm")
        result["llvm_opt"] = std::string(llvm_opt);
    else
        result["llvm_opt"] = py::none();
    return result;
}

struct PreparedProgramState {
    std::shared_ptr<SessionState> session_state;
    void *fn = nullptr;
    size_t n = 0;
    std::vector<void *> ptrs{};
    std::vector<void *> pointer_template{};
    std::vector<NativePointerBinding> pointer_plan{};
    std::vector<NativeRuntimeOutputSpec> output_specs{};
    size_t generation = 0;
    py::object output_graph{};
    py::object output_module{};
    DslProgram program{};
    py::dict buffers{};
    std::vector<NativeRuntimeOutput> outputs{};
    py::object safety_check_buffer{};
    std::vector<std::max_align_t> scalar_storage{};
    bool has_safety_check = false;
    NativeRuntimeOutputKind output_kind = NativeRuntimeOutputKind::Numpy;
    sj::jit::CompilePolicy compile_policy = sj::jit::CompilePolicy::BestEffort;
    sj::CodeTransformations transformations = sj::CodeTransformations::All;
    std::string identifier{};
};

namespace {

static std::shared_ptr<PreparedProgramState>
prepare_native_runtime_program_state(std::shared_ptr<SessionState> session_state, py::handle outputs, py::handle inputs,
                                     std::string_view output_kind) {
    NativeRuntimePlan plan = prepare_native_runtime_plan(outputs, inputs, output_kind);
    NameMap<PyBufferHandle> parsed_buffers = parse_buffers(plan.buffers);
    NameMap<BufferDesc> buffer_descs = buffer_descs_from_handles(parsed_buffers);
    sj::MemoryArena pointer_arena{};
    sj::Context pointer_context(pointer_arena);
    sj::FunctionBuilder pointer_builder(pointer_context);
    std::vector<NativePointerBinding> pointer_plan =
        lower_resolved_native_and_collect_pointer_plan(buffer_descs, plan.program, plan.n, pointer_builder);
    std::vector<void *> ptrs = materialize_pointer_table(parsed_buffers, pointer_plan);
    sj::jit::CompilePolicy compile_policy = session_state->jit.policy();
    sj::CodeTransformations transformations = session_state->jit.transformations();
    std::string identifier = make_resolved_native_function_identifier(
        plan.program, plan.buffers, pointer_plan, plan.output_kind, compile_policy, transformations);
    void *fn = build_resolved_native_function(session_state->jit, plan.buffers, plan.program, plan.n, identifier);

    auto prepared = std::make_shared<PreparedProgramState>();
    prepared->session_state = std::move(session_state);
    prepared->fn = fn;
    prepared->n = plan.n;
    prepared->ptrs = std::move(ptrs);
    prepared->pointer_template = prepared->ptrs;
    prepared->pointer_plan = std::move(pointer_plan);
    prepared->output_specs = plan_native_runtime_outputs(plan.program, buffer_descs, plan.output_kind, plan.n);
    prepared->generation = prepared->session_state->generation;
    prepared->output_graph = py::reinterpret_borrow<py::object>(outputs);
    prepared->output_module = plan.output_kind == NativeRuntimeOutputKind::Numpy
                                  ? py::object(import_numpy())
                                  : py::object(import_pyarrow_for_output());
    prepared->program = std::move(plan.program);
    prepared->buffers = std::move(plan.buffers);
    prepared->outputs = std::move(plan.outputs);
    prepared->safety_check_buffer = std::move(plan.safety_check_buffer);
    prepared->has_safety_check = plan.has_safety_check;
    prepared->output_kind = plan.output_kind;
    prepared->compile_policy = compile_policy;
    prepared->transformations = transformations;
    prepared->identifier = std::move(identifier);
    return prepared;
}

static void check_prepared_program_valid(const PreparedProgramState &state) {
    if (state.session_state == nullptr || state.session_state->generation != state.generation) {
        throw std::runtime_error("prepared program was invalidated by Session.clear() or Session.release()");
    }
}

static bool buffer_desc_has_same_sentinel(const BufferDesc &lhs, const BufferDesc &rhs) {
    if (lhs.null.kind != BufferNullKind::Sentinel) { return true; }
    std::string lhs_sig;
    std::string rhs_sig;
    append_signature_sentinel(lhs_sig, lhs.null.sentinel, lhs.ty);
    append_signature_sentinel(rhs_sig, rhs.null.sentinel, rhs.ty);
    return lhs_sig == rhs_sig;
}

static void validate_prepared_input_desc(std::string_view name, const BufferDesc &expected, const BufferDesc &actual) {
    if (expected.ty != actual.ty) {
        throw std::invalid_argument(sj::format("prepared input %.*s type mismatch: expected %s, got %s", SV(name),
                                               expected.ty.name(), actual.ty.name()));
    }
    if (expected.length != actual.length) {
        throw std::invalid_argument(sj::format("prepared input %.*s length mismatch: expected %zu, got %zu", SV(name),
                                               expected.length, actual.length));
    }
    if (expected.aligned != actual.aligned) {
        throw std::invalid_argument(sj::format("prepared input %.*s alignment mismatch", SV(name)));
    }
    if (expected.bitpacked != actual.bitpacked) {
        throw std::invalid_argument(sj::format("prepared input %.*s bitpacking mismatch", SV(name)));
    }
    if (expected.null.kind != actual.null.kind || expected.null.true_means_null != actual.null.true_means_null ||
        !buffer_desc_has_same_sentinel(expected, actual)) {
        throw std::invalid_argument(sj::format("prepared input %.*s null transport mismatch", SV(name)));
    }
}

static py::dict rebind_prepared_program_buffers(const PreparedProgramState &state, py::handle inputs) {
    py::dict input_handles = normalize_native_runtime_inputs(inputs);
    NameMap<PyBufferHandle> expected_buffers = parse_buffers(state.buffers);
    NameMap<PyBufferHandle> actual_inputs = parse_buffers(input_handles);

    expected_buffers.for_each_entry([&](const NameMapEntry<PyBufferHandle> &entry) {
        if (entry.name == NATIVE_SAFETY_CHECK_BUFFER) { return; }
        if (program_has_output_name(state.program, entry.name)) { return; }
        const PyBufferHandle *actual = actual_inputs.find(entry.name);
        if (actual == nullptr) {
            throw py::value_error(sj::format("missing input %.*s for prepared program", SV(entry.name)));
        }
        validate_prepared_input_desc(entry.name, entry.value, *actual);
    });

    actual_inputs.for_each_entry([&](const NameMapEntry<PyBufferHandle> &entry) {
        const PyBufferHandle *expected = expected_buffers.find(entry.name);
        if (expected == nullptr || program_has_output_name(state.program, entry.name)) {
            throw py::value_error(sj::format("unexpected input %.*s for prepared program", SV(entry.name)));
        }
    });

    py::dict rebound;
    copy_dict_items(rebound, input_handles);
    if (state.has_safety_check) {
        rebound[py_string_from_view(NATIVE_SAFETY_CHECK_BUFFER)] =
            state.buffers[py_string_from_view(NATIVE_SAFETY_CHECK_BUFFER)];
    }
    for (const NativeRuntimeOutput &out : state.outputs) {
        py::object name = py_string_from_view(out.name);
        rebound[name] = state.buffers[name];
    }

    std::string identifier = make_resolved_native_function_identifier(
        state.program, rebound, state.pointer_plan, state.output_kind, state.compile_policy, state.transformations);
    if (identifier != state.identifier) { throw py::value_error("prepared input signature mismatch"); }
    return rebound;
}

static py::dict run_native_runtime_program(std::shared_ptr<SessionState> session_state, py::handle outputs,
                                           py::handle inputs, std::string_view output_kind) {
    std::shared_ptr<PreparedProgramState> prepared =
        prepare_native_runtime_program_state(std::move(session_state), outputs, inputs, output_kind);
    check_prepared_program_valid(*prepared);
    if (prepared->has_safety_check) { reset_native_safety_check(prepared->safety_check_buffer); }
    execute_function(prepared->fn, prepared->n, prepared->ptrs);
    if (prepared->has_safety_check) { check_native_safety_check(prepared->safety_check_buffer); }
    return finalize_native_runtime_outputs(prepared->output_kind, prepared->outputs);
}

} // namespace

void configure_session_jit_context(sj::jit::JitContext &jit) {
    configure_jit_context(jit);
}

sj::Arch parse_session_arch(std::string_view arch_name) {
    return parse_arch(arch_name);
}

void run_prepared_kernel_state(const PreparedKernel &kernel) {
    if (kernel.state == nullptr || kernel.state->generation != kernel.generation) {
        throw std::runtime_error("prepared kernel was invalidated by Session.clear() or Session.release()");
    }
    py::gil_scoped_release release;
    execute_function(kernel.fn, kernel.n, kernel.ptrs);
}

void run_explicit_native(SessionState &state, const py::dict &buffers, py::handle outputs, size_t n) {
    py::dict bound_buffers;
    copy_dict_items(bound_buffers, buffers);
    DslProgram program = import_and_resolve_native_program(outputs, bound_buffers);
    py::object safety_check_buffer;
    bool has_safety_check = native_program_requires_safety_check(program);
    if (has_safety_check) {
        py::str name = py_string_from_view(NATIVE_SAFETY_CHECK_BUFFER);
        if (bound_buffers.contains(name) || program_has_output_name(program, NATIVE_SAFETY_CHECK_BUFFER)) {
            throw py::value_error("reserved internal safety-check buffer name is in use");
        }
        static constexpr char ZERO = 0;
        safety_check_buffer = py::bytearray(&ZERO, 1);
        bound_buffers[name] =
            make_buffer_handle(LogicalType::simple(sj::ScalarDataType::I8), safety_check_buffer, 1, false, false);
    }
    std::vector<NativePointerBinding> pointer_plan;
    std::vector<void *> ptrs = build_resolved_native_pointer_table(bound_buffers, program, n, &pointer_plan);
    std::string identifier = make_resolved_native_function_identifier(program, bound_buffers, pointer_plan,
                                                                      NativeRuntimeOutputKind::Explicit,
                                                                      state.jit.policy(), state.jit.transformations());
    void *fn = build_resolved_native_function(state.jit, bound_buffers, program, n, identifier);
    execute_function(fn, n, ptrs);
    if (has_safety_check) { check_native_safety_check(safety_check_buffer); }
}

size_t infer_native_length_from_buffers(const py::dict &buffers, py::handle outputs) {
    NameMap<PyBufferHandle> parsed_buffers = parse_buffers(buffers);
    NameMap<BufferDesc> parsed_descs = buffer_descs_from_handles(parsed_buffers);
    DslProgram program = import_and_resolve_native_program(outputs, buffers);
    return choose_native_execution_length(program, parsed_descs);
}

py::dict run_native_runtime_program_with_session(std::shared_ptr<SessionState> state, py::handle outputs,
                                                 py::handle inputs, std::string_view output_kind) {
    return run_native_runtime_program(std::move(state), outputs, inputs, output_kind);
}

std::shared_ptr<PreparedProgramState> prepare_program_state(std::shared_ptr<SessionState> state, py::handle outputs,
                                                            py::handle inputs, std::string_view output_kind) {
    return prepare_native_runtime_program_state(std::move(state), outputs, inputs, output_kind);
}

void run_prepared_program_state(PreparedProgramState &state, py::handle inputs) {
    check_prepared_program_valid(state);
    if (!inputs.is_none()) {
        py::dict rebound_buffers = rebind_prepared_program_buffers(state, inputs);
        std::vector<void *> rebound_ptrs = build_resolved_native_pointer_table(rebound_buffers, state.program, state.n);
        state.buffers = std::move(rebound_buffers);
        state.ptrs = std::move(rebound_ptrs);
    }
    if (state.has_safety_check) { reset_native_safety_check(state.safety_check_buffer); }
    {
        py::gil_scoped_release release;
        execute_function(state.fn, state.n, state.ptrs);
    }
    if (state.has_safety_check) { check_native_safety_check(state.safety_check_buffer); }
}

void release_prepared_program_outputs(PreparedProgramState &state) {
    check_prepared_program_valid(state);
    for (const NativeRuntimeOutput &out : state.outputs) {
        state.buffers.attr("pop")(py_string_from_view(out.name), py::none());
    }
    state.outputs.clear();
    state.scalar_storage = {};
    state.ptrs.clear();
}

static void patch_prepared_output_pointers(std::vector<void *> &ptrs, const std::vector<NativeRuntimeOutput> &outputs,
                                           const std::vector<NativePointerBinding> &pointer_plan) {
    for (const NativePointerBinding &binding : pointer_plan) {
        auto output = std::find_if(outputs.begin(), outputs.end(), [&](const NativeRuntimeOutput &candidate) {
            return candidate.name == binding.name;
        });
        if (output == outputs.end()) { continue; }
        if (!binding.null_buffer && output->raw_data != nullptr) {
            ptrs[binding.slot] = output->raw_data;
            continue;
        }
        py::buffer buffer = binding.null_buffer ? output->validity.cast<py::buffer>() : output->data.cast<py::buffer>();
        ptrs[binding.slot] = buffer.request(true).ptr;
    }
}

template <typename Finalize>
static auto run_prepared_program_fresh_impl(PreparedProgramState &state, bool native_scalars, Finalize finalize) {
    check_prepared_program_valid(state);
    if (!state.outputs.empty()) { throw std::runtime_error("prepared outputs must be released before run_fresh()"); }

    std::vector<NativeRuntimeOutput> outputs;
    state.scalar_storage = {};
    if (native_scalars && state.output_kind == NativeRuntimeOutputKind::Numpy) {
        state.scalar_storage.resize(static_cast<size_t>(
            std::count_if(state.output_specs.begin(), state.output_specs.end(),
                          [](const NativeRuntimeOutputSpec &spec) { return spec.scalar && !spec.ty.is_timestamp(); })));
    }
    outputs.reserve(state.output_specs.size());
    size_t scalar_index = 0;
    for (const NativeRuntimeOutputSpec &spec : state.output_specs) {
        if (native_scalars && state.output_kind == NativeRuntimeOutputKind::Numpy && spec.scalar &&
            !spec.ty.is_timestamp()) {
            outputs.push_back(allocate_native_scalar_output(spec, &state.scalar_storage[scalar_index++]));
        } else if (state.output_kind == NativeRuntimeOutputKind::Numpy) {
            outputs.push_back(allocate_numpy_output(state.buffers, spec, state.output_module, false));
        } else {
            outputs.push_back(allocate_pyarrow_output(state.buffers, spec, state.output_module, false));
        }
    }
    std::vector<void *> ptrs = state.pointer_template;
    patch_prepared_output_pointers(ptrs, outputs, state.pointer_plan);

    state.outputs = std::move(outputs);
    state.ptrs = std::move(ptrs);
    if (state.has_safety_check) { reset_native_safety_check(state.safety_check_buffer); }
    {
        py::gil_scoped_release release;
        execute_function(state.fn, state.n, state.ptrs);
    }
    if (state.has_safety_check) { check_native_safety_check(state.safety_check_buffer); }
    return finalize(state.output_kind, state.outputs);
}

py::dict run_prepared_program_fresh_state(PreparedProgramState &state) {
    return run_prepared_program_fresh_impl(state, false, finalize_native_runtime_outputs);
}

py::tuple run_prepared_program_fresh_values_state(PreparedProgramState &state) {
    return run_prepared_program_fresh_impl(state, true, finalize_native_runtime_output_values);
}

py::dict prepared_program_output_buffers(const PreparedProgramState &state) {
    check_prepared_program_valid(state);
    py::dict results;
    for (const NativeRuntimeOutput &out : state.outputs) {
        results[py_string_from_view(out.name)] = out.data;
    }
    return results;
}

py::dict prepared_program_result(const PreparedProgramState &state) {
    check_prepared_program_valid(state);
    return finalize_native_runtime_outputs(state.output_kind, state.outputs);
}

const std::string &prepared_program_identifier(const PreparedProgramState &state) {
    return state.identifier;
}

} // namespace simjit_python
