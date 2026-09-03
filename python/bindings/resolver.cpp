// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "resolver.h"

#include "simjit/core/expr.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace simjit_python {
#define SV(x) static_cast<int>((x).size()), ((x).data() == nullptr ? "" : (x).data())

static constexpr DslFunctionSpec DSL_FUNCTION_SPECS[] = {
    {DslFunctionKind::Year, "year", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::Month, "month", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::Day, "day", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::Hour, "hour", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::Minute, "minute", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::Second, "second", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::DayOfWeek, "day_of_week", DslFunctionGroup::TimestampExtract, 1, 1},
    {DslFunctionKind::Log2, "log2", DslFunctionGroup::IntegerUnary, 1, 1},
    {DslFunctionKind::Log2NoZero, "log2_no_zero", DslFunctionGroup::IntegerUnary, 1, 1},
    {DslFunctionKind::Byteswap, "byteswap", DslFunctionGroup::IntegerUnary, 1, 1},
    {DslFunctionKind::BitFloor, "bit_floor", DslFunctionGroup::IntegerUnary, 1, 1},
    {DslFunctionKind::BitCeil, "bit_ceil", DslFunctionGroup::IntegerUnary, 1, 1},
    {DslFunctionKind::Coalesce, "coalesce", DslFunctionGroup::Coalesce, 1, kDslVariadicFunctionArgs},
    {DslFunctionKind::NullIf, "nullif", DslFunctionGroup::NullIf, 2, 2},
    {DslFunctionKind::IsNull, "is_null", DslFunctionGroup::NullPredicate, 1, 1},
    {DslFunctionKind::IsNotNull, "is_not_null", DslFunctionGroup::NullPredicate, 1, 1},
};

bool DslFunctionSpec::accepts_arg_count(uint32_t count) const {
    return count >= min_args && (max_args == kDslVariadicFunctionArgs || count <= max_args);
}

const DslFunctionSpec *find_dsl_function_spec(DslFunctionKind kind) {
    if (kind == DslFunctionKind::Unknown) { return nullptr; }
    for (const DslFunctionSpec &spec : DSL_FUNCTION_SPECS) {
        if (spec.kind == kind) { return &spec; }
    }
    return nullptr;
}

std::string_view DslProgram::string(DslStringRef ref) const {
    SIMJIT_ASSERT(!ref.is_none());
    uint32_t id = ref.native_id();
    SIMJIT_ASSERT(id < string_count);
    return strings[id];
}

std::optional<std::string_view> DslProgram::optional_string(DslStringRef ref) const {
    if (ref.is_none()) { return std::nullopt; }
    return string(ref);
}

std::string_view DslProgram::required_string(DslStringRef ref, std::string_view field_name) const {
    if (ref.is_none()) { throw std::invalid_argument(sj::format("native DSL missing %.*s", SV(field_name))); }
    return string(ref);
}

static const char *dsl_node_kind_name(DslNodeKind kind) {
    switch (kind) {
    case DslNodeKind::Const: return "ConstExpr";
    case DslNodeKind::Load: return "LoadExpr";
    case DslNodeKind::LoadSplat: return "LoadSplatExpr";
    case DslNodeKind::Gather: return "GatherExpr";
    case DslNodeKind::Index: return "IndexExpr";
    case DslNodeKind::ArithBinary: return "ArithBinaryExpr";
    case DslNodeKind::PredicateBinary: return "PredicateBinaryExpr";
    case DslNodeKind::ArithUnary: return "ArithUnaryExpr";
    case DslNodeKind::PredicateNot: return "PredicateNotExpr";
    case DslNodeKind::Compare: return "CompareExpr";
    case DslNodeKind::IntCast: return "IntCastExpr";
    case DslNodeKind::FloatCast: return "FloatCastExpr";
    case DslNodeKind::BitCast: return "BitCastExpr";
    case DslNodeKind::Function: return "FunctionExpr";
    case DslNodeKind::Select: return "SelectExpr";
    case DslNodeKind::FpClass: return "FpClassExpr";
    case DslNodeKind::Permute: return "PermuteExpr";
    case DslNodeKind::Store: return "StoreExpr";
    case DslNodeKind::Scatter: return "ScatterExpr";
    case DslNodeKind::Pack: return "PackExpr";
    case DslNodeKind::ArithAgg: return "ArithAggExpr";
    case DslNodeKind::PredicateAgg: return "PredicateAggExpr";
    case DslNodeKind::CountIf: return "CountIfExpr";
    case DslNodeKind::GroupedArithAgg: return "GroupedArithAggExpr";
    }
    SIMJIT_UNREACHABLE();
}

size_t DslTypeMap::hash_name(std::string_view name) {
    return std::hash<std::string_view>{}(name);
}

DslTypeEntry *DslTypeMap::begin() {
    return entries.empty() ? nullptr : entries.data();
}
DslTypeEntry *DslTypeMap::end() {
    return entries.empty() ? nullptr : entries.data() + count;
}
const DslTypeEntry *DslTypeMap::begin() const {
    return entries.empty() ? nullptr : entries.data();
}
const DslTypeEntry *DslTypeMap::end() const {
    return entries.empty() ? nullptr : entries.data() + count;
}

DslTypeEntry *DslTypeMap::find(std::string_view name) {
    size_t h = hash_name(name);
    for (DslTypeEntry &entry : *this) {
        if (entry.hash == h && entry.name == name) { return &entry; }
    }
    return nullptr;
}

const DslTypeEntry *DslTypeMap::find(std::string_view name) const {
    size_t h = hash_name(name);
    for (const DslTypeEntry &entry : *this) {
        if (entry.hash == h && entry.name == name) { return &entry; }
    }
    return nullptr;
}

void DslTypeMap::insert_or_assign(std::string_view name, LogicalType type) {
    if (DslTypeEntry *entry = find(name)) {
        entry->type = type;
        return;
    }
    if (count >= entries.size()) { throw std::invalid_argument("native DSL type map capacity exceeded"); }
    entries[count++] = DslTypeEntry{hash_name(name), name, type};
}

static const char *dsl_timestamp_type_name(TimestampUnit unit, TimestampTimezone timezone) {
    bool utc = timezone == TimestampTimezone::UTC;
    switch (unit) {
    case TimestampUnit::Seconds: return utc ? "timestamp64[s, UTC]" : "timestamp64[s]";
    case TimestampUnit::Milliseconds: return utc ? "timestamp64[ms, UTC]" : "timestamp64[ms]";
    case TimestampUnit::Microseconds: return utc ? "timestamp64[us, UTC]" : "timestamp64[us]";
    case TimestampUnit::Nanoseconds: return utc ? "timestamp64[ns, UTC]" : "timestamp64[ns]";
    }
    SIMJIT_UNREACHABLE();
}

const char *LogicalType::name() const {
    if (semantic == SemanticKind::Timestamp64) { return dsl_timestamp_type_name(unit, timezone); }
    switch (dtype) {
    case sj::ScalarDataType::I1: return "i1";
    case sj::ScalarDataType::I8: return is_unsigned ? "u8" : "i8";
    case sj::ScalarDataType::I16: return is_unsigned ? "u16" : "i16";
    case sj::ScalarDataType::I32: return is_unsigned ? "u32" : "i32";
    case sj::ScalarDataType::I64: return is_unsigned ? "u64" : "i64";
    case sj::ScalarDataType::F32: return "f32";
    case sj::ScalarDataType::F64: return "f64";
    case sj::ScalarDataType::I128: return "i128";
    }
    SIMJIT_UNREACHABLE();
}

struct TypeSlot {
    bool has = false;
    LogicalType type{};

    static TypeSlot none() { return {}; }
    static TypeSlot of(LogicalType type) { return TypeSlot{true, type}; }
    static TypeSlot i1() { return of(LogicalType::simple(sj::ScalarDataType::I1)); }
    static TypeSlot i32() { return of(LogicalType::simple(sj::ScalarDataType::I32)); }
    static TypeSlot i64() { return of(LogicalType::simple(sj::ScalarDataType::I64)); }

    bool operator==(const TypeSlot &rhs) const {
        if (has != rhs.has) { return false; }
        return !has || type == rhs.type;
    }
    bool operator!=(const TypeSlot &rhs) const { return !(*this == rhs); }

    const char *name() const {
        if (!has) { return "None"; }
        return type.name();
    }

    bool is_timestamp() const { return has && type.semantic == SemanticKind::Timestamp64; }

    bool is_predicate() const {
        return has && type.semantic == SemanticKind::Plain && type.dtype == sj::ScalarDataType::I1;
    }

    bool is_signed_int() const {
        return has && type.semantic == SemanticKind::Plain && !type.is_unsigned && sj::is_simple_int_dtype(type.dtype);
    }

    bool is_unsigned_int() const {
        return has && type.semantic == SemanticKind::Plain && type.is_unsigned && sj::is_simple_int_dtype(type.dtype);
    }

    bool is_int() const { return is_signed_int() || is_unsigned_int(); }

    bool is_float() const {
        return has && type.semantic == SemanticKind::Plain &&
               (type.dtype == sj::ScalarDataType::F32 || type.dtype == sj::ScalarDataType::F64);
    }

    bool is_value_type() const { return has && !is_predicate(); }

    uint32_t bit_width() const {
        if (!is_int()) { return 0; }
        return sj::scalar_dtype_bits(type.dtype);
    }

    bool same_timestamp(const TypeSlot &rhs) const { return has && rhs.has && type == rhs.type; }

    bool same_sign(const TypeSlot &rhs) const {
        if (!has || !rhs.has) { return true; }
        if (!is_int() || !rhs.is_int()) { return true; }
        return (is_unsigned_int() && rhs.is_unsigned_int()) || (is_signed_int() && rhs.is_signed_int());
    }

    bool same_width(const TypeSlot &rhs) const {
        uint32_t lhs_bits = bit_width();
        uint32_t rhs_bits = rhs.bit_width();
        return lhs_bits != 0 && lhs_bits == rhs_bits;
    }

    bool can_promote_numeric_to(const TypeSlot &dst) const {
        if (!has || !dst.has || (*this == dst)) { return false; }
        if (is_signed_int() && dst.is_signed_int()) { return bit_width() < dst.bit_width(); }
        if (is_unsigned_int() && dst.is_unsigned_int()) { return bit_width() < dst.bit_width(); }
        if (is_float() && dst.is_float()) {
            return type.dtype == sj::ScalarDataType::F32 && dst.type.dtype == sj::ScalarDataType::F64;
        }
        return false;
    }

    void require_int_if_known(std::string_view where, std::string_view message) const {
        if (!has || is_int()) { return; }
        throw_requirement_error(where, message);
    }

    void require_float_if_known(std::string_view where, std::string_view message) const {
        if (!has || is_float()) { return; }
        throw_requirement_error(where, message);
    }

    void require_value_type_if_known(std::string_view where, std::string_view message) const {
        if (!has || is_value_type()) { return; }
        throw_requirement_error(where, message);
    }

private:
    void throw_requirement_error(std::string_view where, std::string_view message) const {
        throw std::invalid_argument(sj::format("%.*s at %.*s, got %s", SV(message), SV(where), type.name()));
    }
};

static TypeSlot unify_hard(const TypeSlot &current, const TypeSlot &incoming, std::string_view where) {
    if (!current.has) { return incoming; }
    if (!incoming.has) { return current; }
    if (current != incoming) {
        throw std::invalid_argument(
            sj::format("type conflict at %.*s: %s vs %s", SV(where), current.type.name(), incoming.type.name()));
    }
    return current;
}

static LogicalType wider_int(const LogicalType &type) {
    TypeSlot slot = TypeSlot::of(type);
    if (!slot.is_int() || slot.bit_width() < 64) {
        return LogicalType::simple(sj::ScalarDataType::I64, slot.is_unsigned_int());
    }
    return type;
}

static TypeSlot promote_same_sign_numeric_pair(std::string_view where, const TypeSlot &lhs, const TypeSlot &rhs) {
    if (!lhs.has) { return rhs; }
    if (!rhs.has) { return lhs; }
    if (lhs == rhs) { return lhs; }
    if (lhs.is_float() && rhs.is_float()) {
        return TypeSlot::of(
            LogicalType::simple(lhs.type.dtype == sj::ScalarDataType::F64 || rhs.type.dtype == sj::ScalarDataType::F64
                                    ? sj::ScalarDataType::F64
                                    : sj::ScalarDataType::F32));
    }
    if (lhs.is_signed_int() && rhs.is_signed_int()) { return lhs.bit_width() >= rhs.bit_width() ? lhs : rhs; }
    if (lhs.is_unsigned_int() && rhs.is_unsigned_int()) { return lhs.bit_width() >= rhs.bit_width() ? lhs : rhs; }
    throw std::invalid_argument(
        sj::format("incompatible numeric types at %.*s: %s vs %s", SV(where), lhs.name(), rhs.name()));
}

static TypeSlot choose_promoted_numeric_type(std::string_view where, const TypeSlot &constraint, const TypeSlot &a,
                                             const TypeSlot &b = TypeSlot::none(),
                                             const TypeSlot &c = TypeSlot::none()) {
    TypeSlot chosen{};
    TypeSlot values[] = {a, b, c};
    for (TypeSlot value : values) {
        if (!value.has) { continue; }
        if (value.is_timestamp()) {
            throw std::invalid_argument(
                sj::format("timestamp type is not valid plain numeric type at %.*s", SV(where)));
        }
        if (value.is_predicate()) {
            throw std::invalid_argument(sj::format("predicate type is not valid numeric type at %.*s", SV(where)));
        }
        chosen = promote_same_sign_numeric_pair(where, chosen, value);
    }
    return unify_hard(constraint, chosen, where);
}

static TypeSlot choose_bitwise_type(std::string_view where, const TypeSlot &chosen, const TypeSlot &lhs,
                                    const TypeSlot &rhs) {
    if (chosen.has) {
        if (lhs.has && !lhs.is_int()) {
            throw std::invalid_argument(sj::format("bitwise operation requires integer type at %.*s", SV(where)));
        }
        if (rhs.has && !rhs.is_int()) {
            throw std::invalid_argument(sj::format("bitwise operation requires integer type at %.*s", SV(where)));
        }
        if (lhs.has && chosen != lhs && !(chosen.same_width(lhs) && chosen.is_int() && lhs.is_int())) {
            throw std::invalid_argument(
                sj::format("type conflict at %.*s: %s vs %s", SV(where), chosen.type.name(), lhs.type.name()));
        }
        if (rhs.has && chosen != rhs && !(chosen.same_width(rhs) && chosen.is_int() && rhs.is_int())) {
            throw std::invalid_argument(
                sj::format("type conflict at %.*s: %s vs %s", SV(where), chosen.type.name(), rhs.type.name()));
        }
        return chosen;
    }
    if (!lhs.has) { return rhs; }
    if (!rhs.has) { return lhs; }
    if (!lhs.is_int() || !rhs.is_int()) {
        throw std::invalid_argument(sj::format("bitwise operation requires integer type at %.*s", SV(where)));
    }
    if (lhs == rhs) { return lhs; }
    if (!lhs.same_width(rhs)) {
        throw std::invalid_argument(
            sj::format("bitwise operation requires same-width integer operands at %.*s: %s vs %s", SV(where),
                       lhs.type.name(), rhs.type.name()));
    }
    return lhs;
}

static bool is_bitwise_binary_op(sj::ArithBinaryOp op) {
    switch (op) {
    case sj::ArithBinaryOp::And:
    case sj::ArithBinaryOp::Or:
    case sj::ArithBinaryOp::Xor:
    case sj::ArithBinaryOp::AndNot:
    case sj::ArithBinaryOp::ShiftLeftLogical:
    case sj::ArithBinaryOp::ShiftRightLogical:
    case sj::ArithBinaryOp::ShiftRightArith:
    case sj::ArithBinaryOp::RotateLeft:
    case sj::ArithBinaryOp::RotateRight: return true;
    case sj::ArithBinaryOp::Add:
    case sj::ArithBinaryOp::Sub:
    case sj::ArithBinaryOp::Mul:
    case sj::ArithBinaryOp::Mul64SE:
    case sj::ArithBinaryOp::Mul64ZE:
    case sj::ArithBinaryOp::Div:
    case sj::ArithBinaryOp::UDiv:
    case sj::ArithBinaryOp::Mod:
    case sj::ArithBinaryOp::UMod:
    case sj::ArithBinaryOp::Min:
    case sj::ArithBinaryOp::Max:
    case sj::ArithBinaryOp::UMin:
    case sj::ArithBinaryOp::UMax: return false;
    }
    SIMJIT_UNREACHABLE();
}

static bool is_float_unary_op(sj::ArithUnaryOp op) {
    switch (op) {
    case sj::ArithUnaryOp::RoundNearest:
    case sj::ArithUnaryOp::RoundDown:
    case sj::ArithUnaryOp::RoundUp:
    case sj::ArithUnaryOp::RoundTruncate:
    case sj::ArithUnaryOp::Sqrt:
    case sj::ArithUnaryOp::Rsqrt:
    case sj::ArithUnaryOp::Rcp: return true;
    case sj::ArithUnaryOp::Not:
    case sj::ArithUnaryOp::Negate:
    case sj::ArithUnaryOp::Abs:
    case sj::ArithUnaryOp::Lzcnt:
    case sj::ArithUnaryOp::Tzcnt:
    case sj::ArithUnaryOp::Popcount: return false;
    }
    SIMJIT_UNREACHABLE();
}

static bool is_int_unary_op(sj::ArithUnaryOp op) {
    switch (op) {
    case sj::ArithUnaryOp::Not:
    case sj::ArithUnaryOp::Lzcnt:
    case sj::ArithUnaryOp::Tzcnt:
    case sj::ArithUnaryOp::Popcount: return true;
    case sj::ArithUnaryOp::Negate:
    case sj::ArithUnaryOp::Abs:
    case sj::ArithUnaryOp::RoundNearest:
    case sj::ArithUnaryOp::RoundDown:
    case sj::ArithUnaryOp::RoundUp:
    case sj::ArithUnaryOp::RoundTruncate:
    case sj::ArithUnaryOp::Sqrt:
    case sj::ArithUnaryOp::Rsqrt:
    case sj::ArithUnaryOp::Rcp: return false;
    }
    SIMJIT_UNREACHABLE();
}

[[noreturn]] static void throw_output_type_conflict(const DslOutput &out, const TypeSlot &expected,
                                                    const TypeSlot &actual) {
    throw std::invalid_argument(
        sj::format("output type conflict for %.*s: expected %s, got %s", SV(out.name), expected.name(), actual.name()));
}

class NativeDslResolver {
public:
    NativeDslResolver(DslProgram &program_arg, DslTypeMap input_types_arg, DslTypeMap output_types_arg)
        : program(program_arg), input_types(input_types_arg), output_types(output_types_arg) {}

    size_t resolve_in_place() {
        collect_graph();
        expected_types.assign(program.nodes.size(), TypeSlot::none());
        edge_expected.assign(program.child_edges.size(), TypeSlot::none());
        for (uint32_t i = 0; i < program.output_count; ++i) {
            const DslOutput &out = program.outputs[i];
            add_expected_type(out.root, output_type_for(out.name), "output");
        }

        for (NodeId id : postorder) {
            assign_local(id, false, false);
        }
        for (auto it = postorder.rbegin(); it != postorder.rend(); ++it) {
            assign_local(*it, false, false);
        }
        std::fill(edge_expected.begin(), edge_expected.end(), TypeSlot::none());
        for (NodeId id : postorder) {
            assign_local(id, true, true);
        }

        rewrite_expected_edges();
        recompute_resolved_types();
        if (rewrite_output_roots()) { recompute_resolved_types(); }
        simplify_output_roots();
        recompute_resolved_types();

        for (uint32_t i = 0; i < program.output_count; ++i) {
            const DslOutput &out = program.outputs[i];
            TypeSlot expected = output_type_for(out.name);
            TypeSlot actual = resolved_type_slot(out.root, "output");
            if (expected.has && expected != actual) {
                throw std::invalid_argument(sj::format("output type conflict for %.*s: expected %s, got %s",
                                                       SV(out.name), expected.type.name(), actual.name()));
            }
        }
        return 1;
    }

private:
    DslProgram &program;
    DslTypeMap input_types;
    DslTypeMap output_types;
    size_t inserted_cast_count = 0;
    uint32_t graph_node_count = 0;
    std::vector<NodeId> postorder{};
    std::vector<TypeSlot> expected_types{};
    std::vector<TypeSlot> edge_expected{};
    bool mutate_expected_types = true;

    TypeSlot output_type_for(std::string_view name) const {
        const DslTypeEntry *entry = output_types.find(name);
        if (entry == nullptr) { return TypeSlot::none(); }
        return TypeSlot::of(entry->type);
    }

    TypeSlot input_type_for(std::string_view name) const {
        const DslTypeEntry *entry = input_types.find(name);
        if (entry == nullptr) { return TypeSlot::none(); }
        return TypeSlot::of(entry->type);
    }

    void set_input_type(std::string_view name, const TypeSlot &incoming, std::string_view where) {
        if (!incoming.has) { return; }
        DslTypeEntry *entry = input_types.find(name);
        TypeSlot current = entry == nullptr ? TypeSlot::none() : TypeSlot::of(entry->type);
        TypeSlot resolved = unify_hard(current, incoming, where);
        if (current != resolved) { input_types.insert_or_assign(name, resolved.type); }
    }

    NodeId child(NodeId id, uint32_t index) const {
        const DslNode &node = program.nodes[id];
        SIMJIT_ASSERT(index < node.child_count);
        return program.child_edges[node.first_child + index];
    }

    void set_child(NodeId id, uint32_t index, NodeId value) {
        const DslNode &node = program.nodes[id];
        SIMJIT_ASSERT(index < node.child_count);
        program.child_edges[node.first_child + index] = value;
    }

    const DslTypeSlot &declared_slot(NodeId id) const {
        SIMJIT_ASSERT(id < program.node_count);
        return program.declared_types[id];
    }

    const DslTypeSlot &resolved_slot(NodeId id) const {
        SIMJIT_ASSERT(id < program.node_count);
        return program.resolved_types[id];
    }

    TypeSlot maybe_declared_type(NodeId id) const {
        const DslTypeSlot &declared = declared_slot(id);
        return declared.has ? TypeSlot::of(declared.type) : TypeSlot::none();
    }

    bool declared_is_timestamp(NodeId id) const {
        const DslTypeSlot &declared = declared_slot(id);
        return declared.has && declared.type.semantic == SemanticKind::Timestamp64;
    }

    TypeSlot constrain_declared_type(NodeId id, const TypeSlot &incoming, std::string_view where) const {
        const DslTypeSlot &declared = declared_slot(id);
        if (!declared.has) { return incoming; }
        return unify_hard(TypeSlot::of(declared.type), incoming, where);
    }

    TypeSlot maybe_resolved_type(NodeId id) const {
        const DslTypeSlot &resolved = resolved_slot(id);
        return resolved.has ? TypeSlot::of(resolved.type) : TypeSlot::none();
    }

    TypeSlot resolved_type_slot(NodeId id, std::string_view where) const {
        const DslTypeSlot &resolved = resolved_slot(id);
        if (!resolved.has) { throw std::invalid_argument(sj::format("unresolved type at %.*s", SV(where))); }
        return TypeSlot::of(resolved.type);
    }

    void set_resolved_type(NodeId id, const TypeSlot &type) {
        program.resolved_types[id] = DslTypeSlot{type.type, type.has};
    }

    NodeId append_cast(DslNodeKind kind, const TypeSlot &target, NodeId arg) {
        if (!target.has) { throw std::invalid_argument("native DSL resolver cannot append cast without target type"); }
        if (program.node_count >= program.nodes.size()) {
            throw std::invalid_argument("native DSL resolver node capacity exceeded");
        }
        if (program.edge_count >= program.child_edges.size()) {
            throw std::invalid_argument("native DSL resolver edge capacity exceeded");
        }
        NodeId id = program.node_count++;
        DslNode &node = program.nodes[id];
        node = DslNode{};
        program.declared_types[id] = DslTypeSlot{target.type, true};
        program.resolved_types[id] = DslTypeSlot{target.type, true};
        node.kind = kind;
        node.first_child = program.edge_count;
        node.child_count = 1;
        if (kind == DslNodeKind::IntCast) { node.step_data<DslNodeKind::IntCast>().kind = DslIntCastKind::Cast; }
        program.child_edges[program.edge_count++] = arg;
        inserted_cast_count += 1;
        return id;
    }

    NodeId coerce_promoted_node(NodeId id, const TypeSlot &expected, std::string_view where) {
        TypeSlot actual = maybe_resolved_type(id);
        if (!expected.has || !actual.has || actual == expected) { return id; }
        if (!actual.can_promote_numeric_to(expected)) {
            throw std::invalid_argument(
                sj::format("type conflict at %.*s: %s vs %s", SV(where), actual.type.name(), expected.type.name()));
        }
        if (actual.is_int() && expected.is_int()) { return append_cast(DslNodeKind::IntCast, expected, id); }
        if (actual.is_float() && expected.is_float()) { return append_cast(DslNodeKind::FloatCast, expected, id); }
        throw std::invalid_argument(
            sj::format("type conflict at %.*s: %s vs %s", SV(where), actual.type.name(), expected.type.name()));
    }

    bool is_final_output_node(DslNodeKind kind) const {
        switch (kind) {
        case DslNodeKind::Store:
        case DslNodeKind::Scatter:
        case DslNodeKind::Pack:
        case DslNodeKind::ArithAgg:
        case DslNodeKind::PredicateAgg:
        case DslNodeKind::CountIf:
        case DslNodeKind::GroupedArithAgg: return true;
        case DslNodeKind::Const:
        case DslNodeKind::Load:
        case DslNodeKind::LoadSplat:
        case DslNodeKind::Index:
        case DslNodeKind::Gather:
        case DslNodeKind::ArithBinary:
        case DslNodeKind::PredicateBinary:
        case DslNodeKind::ArithUnary:
        case DslNodeKind::PredicateNot:
        case DslNodeKind::Compare:
        case DslNodeKind::IntCast:
        case DslNodeKind::FloatCast:
        case DslNodeKind::BitCast:
        case DslNodeKind::Select:
        case DslNodeKind::FpClass:
        case DslNodeKind::Permute:
        case DslNodeKind::Function: return false;
        }
        SIMJIT_UNREACHABLE();
    }

    void collect_graph() {
        graph_node_count = program.node_count;
        postorder.clear();
        std::vector<uint8_t> state(program.nodes.size(), 0);

        struct Frame {
            NodeId id = 0;
            uint32_t next_child = 0;
        };

        for (uint32_t output_idx = 0; output_idx < program.output_count; ++output_idx) {
            NodeId root = program.outputs[output_idx].root;
            if (root >= graph_node_count) { throw std::invalid_argument("native DSL output root is out of range"); }
            if (state[root] == 2) { continue; }
            if (state[root] == 1) { throw std::invalid_argument("native DSL expression graph contains a cycle"); }

            std::vector<Frame> stack;
            state[root] = 1;
            stack.push_back(Frame{root, 0});
            while (!stack.empty()) {
                Frame &frame = stack.back();
                const DslNode &node = program.nodes[frame.id];
                if (frame.next_child < node.child_count) {
                    NodeId kid = child(frame.id, frame.next_child++);
                    if (kid >= graph_node_count) {
                        throw std::invalid_argument("native DSL child node is out of range");
                    }
                    if (state[kid] == 0) {
                        state[kid] = 1;
                        stack.push_back(Frame{kid, 0});
                    } else if (state[kid] == 1) {
                        throw std::invalid_argument("native DSL expression graph contains a cycle");
                    }
                    continue;
                }
                state[frame.id] = 2;
                postorder.push_back(frame.id);
                stack.pop_back();
            }
        }
    }

    void add_expected_type(NodeId id, const TypeSlot &incoming, std::string_view where) {
        if (!incoming.has) { return; }
        if (id >= expected_types.size()) {
            throw std::invalid_argument("native DSL expected type node is out of range");
        }
        expected_types[id] = unify_hard(expected_types[id], incoming, where);
    }

    void expect_child(NodeId id, uint32_t index, const TypeSlot &incoming, std::string_view where, bool record_edge) {
        if (!incoming.has) { return; }
        NodeId kid = child(id, index);
        if (mutate_expected_types) { add_expected_type(kid, incoming, where); }
        if (record_edge) {
            uint32_t edge = program.nodes[id].first_child + index;
            if (edge >= edge_expected.size()) {
                throw std::invalid_argument("native DSL expected edge is out of range");
            }
            edge_expected[edge] = unify_hard(edge_expected[edge], incoming, where);
        }
    }

    TypeSlot expected_type(NodeId id) const {
        if (id >= expected_types.size()) { return TypeSlot::none(); }
        return expected_types[id];
    }

    void assign_const(NodeId id, bool final) {
        TypeSlot dt = constrain_declared_type(id, expected_type(id), "const");
        set_resolved_type(id, dt);
        if (final && !dt.has) { throw std::invalid_argument("unresolved const type at expression"); }
    }

    void assign_load(NodeId id, bool final) {
        DslNode &node = program.nodes[id];
        DslStringRef name_ref = node.kind == DslNodeKind::Load ? node.step_data<DslNodeKind::Load>().name
                                                               : node.step_data<DslNodeKind::LoadSplat>().name;
        std::string_view name = program.required_string(name_ref, "load name");
        TypeSlot dt = constrain_declared_type(id, input_type_for(name), "load");
        if (!dt.has && expected_type(id).has) { dt = expected_type(id); }
        if (dt.has) { set_input_type(name, dt, "load"); }
        set_resolved_type(id, dt);
        if (final && !dt.has) {
            throw std::invalid_argument(sj::format("unresolved load type for %.*s at expression", SV(name)));
        }
    }

    void assign_index(NodeId id) {
        TypeSlot dt = constrain_declared_type(id, expected_type(id), "index");
        if (!dt.has) { dt = TypeSlot::i64(); }
        set_resolved_type(id, dt);
    }

    void assign_gather(NodeId id, bool final) {
        maybe_resolved_type(child(id, 0)).require_int_if_known("gather index", "gather index must be integer");
        TypeSlot dt = constrain_declared_type(id, expected_type(id), "gather");
        set_resolved_type(id, dt);
        if (final && !dt.has) { throw std::invalid_argument("unresolved gather result type at expression"); }
    }

    void assign_arith_binary(NodeId id, bool final, bool record_edges) {
        const DslNode &node = program.nodes[id];
        TypeSlot lhs_dt = maybe_resolved_type(child(id, 0));
        TypeSlot rhs_dt = maybe_resolved_type(child(id, 1));
        sj::ArithBinaryOp op = node.step_data<DslNodeKind::ArithBinary>().op;
        const char *op_name = sj::show_arith_binary_op(op);
        bool is_bitwise = is_bitwise_binary_op(op);
        TypeSlot dt = constrain_declared_type(id, expected_type(id), is_bitwise ? "bitwise" : "arithmetic");

        if (lhs_dt.is_timestamp() || rhs_dt.is_timestamp() || dt.is_timestamp()) {
            if (op == sj::ArithBinaryOp::Min || op == sj::ArithBinaryOp::Max) {
                if (!lhs_dt.same_timestamp(rhs_dt)) {
                    throw std::invalid_argument(
                        sj::format("timestamp operands for %s must match exactly at expression: %s vs %s", op_name,
                                   lhs_dt.name(), rhs_dt.name()));
                }
                dt = unify_hard(dt, lhs_dt, "arithmetic");
                expect_child(id, 0, dt, "arithmetic", record_edges);
                expect_child(id, 1, dt, "arithmetic", record_edges);
                set_resolved_type(id, dt);
                return;
            }
            if (op == sj::ArithBinaryOp::Sub && lhs_dt.is_timestamp() && rhs_dt.is_timestamp()) {
                if (!lhs_dt.same_timestamp(rhs_dt)) {
                    throw std::invalid_argument(
                        sj::format("timestamp operands for subtraction must match exactly at expression: %s vs %s",
                                   lhs_dt.name(), rhs_dt.name()));
                }
                dt = unify_hard(dt, TypeSlot::i64(), "arithmetic");
                if (!dt.has) { dt = TypeSlot::i64(); }
                expect_child(id, 0, lhs_dt, "arithmetic", record_edges);
                expect_child(id, 1, rhs_dt, "arithmetic", record_edges);
                set_resolved_type(id, dt);
                return;
            }
            if ((op == sj::ArithBinaryOp::Add || op == sj::ArithBinaryOp::Sub) && lhs_dt.is_timestamp() &&
                rhs_dt == TypeSlot::i64()) {
                dt = unify_hard(dt, lhs_dt, "arithmetic");
                if (!dt.has) { dt = lhs_dt; }
                expect_child(id, 0, dt, "arithmetic", record_edges);
                expect_child(id, 1, TypeSlot::i64(), "arithmetic", record_edges);
                set_resolved_type(id, dt);
                return;
            }
            if (op == sj::ArithBinaryOp::Add && rhs_dt.is_timestamp() && lhs_dt == TypeSlot::i64()) {
                dt = unify_hard(dt, rhs_dt, "arithmetic");
                if (!dt.has) { dt = rhs_dt; }
                expect_child(id, 0, TypeSlot::i64(), "arithmetic", record_edges);
                expect_child(id, 1, dt, "arithmetic", record_edges);
                set_resolved_type(id, dt);
                return;
            }
            throw std::invalid_argument(sj::format("timestamp operation %s is not allowed at expression", op_name));
        }

        TypeSlot lhs_expected{};
        TypeSlot rhs_expected{};
        if (is_bitwise) {
            dt = choose_bitwise_type("bitwise", dt, lhs_dt, rhs_dt);
            dt.require_int_if_known("bitwise", sj::format("%s requires integer type", op_name));
            bool mixed_sign_same_width =
                lhs_dt.is_int() && rhs_dt.is_int() && lhs_dt.same_width(rhs_dt) && !lhs_dt.same_sign(rhs_dt);
            lhs_expected = mixed_sign_same_width ? lhs_dt : dt;
            rhs_expected = mixed_sign_same_width ? rhs_dt : dt;
        } else {
            dt = choose_promoted_numeric_type("arithmetic", dt, lhs_dt, rhs_dt);
            dt.require_value_type_if_known("arithmetic", sj::format("%s requires numeric value type", op_name));
            if (lhs_dt.is_int() && rhs_dt.is_int() && !lhs_dt.same_sign(rhs_dt)) {
                throw std::invalid_argument(sj::format("mixed signedness is not allowed for %s at expression: %s vs %s",
                                                       op_name, lhs_dt.type.name(), rhs_dt.type.name()));
            }
            lhs_expected = dt;
            rhs_expected = dt;
        }
        expect_child(id, 0, lhs_expected, is_bitwise ? "bitwise" : "arithmetic", record_edges);
        expect_child(id, 1, rhs_expected, is_bitwise ? "bitwise" : "arithmetic", record_edges);
        if (final && !dt.has) { throw std::invalid_argument("unresolved arithmetic type at expression"); }
        set_resolved_type(id, dt);
    }

    void assign_arith_unary(NodeId id, bool final, bool record_edges) {
        DslNode &node = program.nodes[id];
        TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
        TypeSlot dt = constrain_declared_type(id, expected_type(id), "unary");
        if (!dt.has) { dt = arg_dt; }
        sj::ArithUnaryOp op = node.step_data<DslNodeKind::ArithUnary>().op;
        std::string_view op_name = sj::show_arith_unary_op(op);
        TypeSlot check = dt.has ? dt : arg_dt;
        if (check.is_timestamp()) {
            throw std::invalid_argument(
                sj::format("timestamp unary operation %.*s is not allowed at expression", SV(op_name)));
        }
        if (is_float_unary_op(op)) {
            check.require_float_if_known("unary", sj::format("%.*s requires float type", SV(op_name)));
        }
        if (is_int_unary_op(op)) {
            check.require_int_if_known("unary", sj::format("%.*s requires integer type", SV(op_name)));
        }
        if (op == sj::ArithUnaryOp::Negate || op == sj::ArithUnaryOp::Abs) {
            check.require_value_type_if_known("unary", sj::format("%.*s requires numeric value type", SV(op_name)));
        }
        expect_child(id, 0, dt, "unary", record_edges);
        if (final && !dt.has) { throw std::invalid_argument("unresolved unary type at expression"); }
        set_resolved_type(id, dt);
    }

    void assign_compare(NodeId id, bool final, bool record_edges) {
        const DslNode &node = program.nodes[id];
        TypeSlot lhs_dt = maybe_resolved_type(child(id, 0));
        TypeSlot rhs_dt = maybe_resolved_type(child(id, 1));
        if (lhs_dt.is_timestamp() || rhs_dt.is_timestamp()) {
            if (node.step_data<DslNodeKind::Compare>().is_unsigned) {
                throw std::invalid_argument("unsigned compare is not allowed for timestamps at expression");
            }
            if (!lhs_dt.same_timestamp(rhs_dt)) {
                throw std::invalid_argument(
                    sj::format("timestamp compare operands must match exactly at expression: %s vs %s", lhs_dt.name(),
                               rhs_dt.name()));
            }
            expect_child(id, 0, lhs_dt, "compare", record_edges);
            expect_child(id, 1, rhs_dt, "compare", record_edges);
            set_resolved_type(id, TypeSlot::i1());
            return;
        }
        TypeSlot operand_dt{};
        if (lhs_dt.has && rhs_dt.has && lhs_dt.is_predicate() && rhs_dt.is_predicate()) {
            operand_dt = TypeSlot::i1();
        } else {
            operand_dt = choose_promoted_numeric_type("compare", TypeSlot::none(), lhs_dt, rhs_dt);
        }
        if (lhs_dt.is_int() && rhs_dt.is_int() && !lhs_dt.same_sign(rhs_dt)) {
            throw std::invalid_argument(
                sj::format("mixed signedness is not allowed for compare at expression: %s vs %s", lhs_dt.type.name(),
                           rhs_dt.type.name()));
        }
        if (node.step_data<DslNodeKind::Compare>().is_unsigned) {
            operand_dt.require_int_if_known("compare", "unsigned compare requires integer type");
        } else {
            operand_dt.require_value_type_if_known("compare", "compare requires value type");
        }
        expect_child(id, 0, operand_dt, "compare", record_edges);
        expect_child(id, 1, operand_dt, "compare", record_edges);
        if (final && !operand_dt.has) { throw std::invalid_argument("unresolved compare operand type at expression"); }
        set_resolved_type(id, TypeSlot::i1());
    }

    void assign_int_cast(NodeId id, bool final, bool record_edges) {
        const DslNode &node = program.nodes[id];
        TypeSlot target = constrain_declared_type(id, expected_type(id), "int cast");
        if (!target.has) { throw std::invalid_argument("int cast target type is required at expression"); }
        DslIntCastKind kind = node.step_data<DslNodeKind::IntCast>().kind;
        TypeSlot arg_expected{};
        if (target.is_timestamp()) {
            arg_expected = TypeSlot::i64();
            if (kind != DslIntCastKind::Cast) {
                throw std::invalid_argument("only generic cast to timestamp is allowed at expression");
            }
        } else if (kind == DslIntCastKind::Sext || kind == DslIntCastKind::Zext) {
            arg_expected = target;
        } else if (kind == DslIntCastKind::Trunc) {
            arg_expected = TypeSlot::of(wider_int(target.type));
        }
        expect_child(id, 0, arg_expected, "int cast", record_edges);

        TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
        if (target.is_timestamp()) {
            if ((final || arg_dt.has) && arg_dt != TypeSlot::i64()) {
                throw std::invalid_argument("timestamp cast source must be I64 at expression");
            }
            set_resolved_type(id, target);
            return;
        }
        if (arg_dt.is_timestamp()) {
            if (kind != DslIntCastKind::Cast) {
                throw std::invalid_argument("only generic cast from timestamp is allowed at expression");
            }
            if (target != TypeSlot::i64()) {
                throw std::invalid_argument("timestamp can only cast to I64 at expression");
            }
            set_resolved_type(id, target);
            return;
        }
        arg_dt.require_int_if_known("int cast source", "int cast source must be integer");
        target.require_int_if_known("int cast target", "int cast target must be integer");
        set_resolved_type(id, target);
    }

    void assign_float_cast(NodeId id) {
        TypeSlot target = constrain_declared_type(id, expected_type(id), "float cast");
        if (!target.has) { throw std::invalid_argument("float cast target type is required at expression"); }
        if (maybe_resolved_type(child(id, 0)).is_timestamp()) {
            throw std::invalid_argument("float cast is not allowed for timestamps at expression");
        }
        set_resolved_type(id, target);
    }

    void assign_bit_cast(NodeId id, bool final, bool record_edges) {
        TypeSlot target = constrain_declared_type(id, expected_type(id), "bitcast");
        if (!target.has) { throw std::invalid_argument("bitcast target type is required at expression"); }
        expect_child(id, 0, maybe_declared_type(child(id, 0)), "bitcast", record_edges);
        TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
        if (target.is_timestamp() || arg_dt.is_timestamp()) {
            throw std::invalid_argument("bitcast is not allowed for timestamps at expression");
        }
        if (final && !arg_dt.has) { throw std::invalid_argument("bitcast source type is unresolved at expression"); }
        set_resolved_type(id, target);
    }

    void assign_select(NodeId id, bool final, bool record_edges) {
        expect_child(id, 0, TypeSlot::i1(), "select", record_edges);
        TypeSlot truthy_dt = maybe_resolved_type(child(id, 1));
        TypeSlot falsy_dt = maybe_resolved_type(child(id, 2));
        TypeSlot branch_dt = constrain_declared_type(id, expected_type(id), "select");
        if (truthy_dt.is_timestamp() || falsy_dt.is_timestamp() || branch_dt.is_timestamp()) {
            branch_dt = unify_hard(branch_dt, truthy_dt, "select");
            branch_dt = unify_hard(branch_dt, falsy_dt, "select");
        } else if (truthy_dt.is_predicate() && falsy_dt.is_predicate() && !branch_dt.has) {
            branch_dt = TypeSlot::i1();
        } else {
            branch_dt = choose_promoted_numeric_type("select", branch_dt, truthy_dt, falsy_dt);
        }
        expect_child(id, 1, branch_dt, "select", record_edges);
        expect_child(id, 2, branch_dt, "select", record_edges);
        if (final && !branch_dt.has) { throw std::invalid_argument("unresolved select branch type at expression"); }
        set_resolved_type(id, branch_dt);
    }

    void assign_store_like(NodeId id, bool record_edges) {
        TypeSlot value_expected = constrain_declared_type(id, expected_type(id), "store");
        expect_child(id, 0, value_expected, "store", record_edges);
        if (program.nodes[id].step_data<DslNodeKind::Store>().has_cond) {
            expect_child(id, 1, TypeSlot::i1(), "store", record_edges);
        }
        set_resolved_type(id, value_expected.has ? value_expected : maybe_resolved_type(child(id, 0)));
    }

    void assign_scatter(NodeId id, bool record_edges) {
        TypeSlot value_expected = constrain_declared_type(id, expected_type(id), "scatter");
        expect_child(id, 0, value_expected, "scatter", record_edges);
        maybe_resolved_type(child(id, 1)).require_int_if_known("scatter index", "scatter index must be integer");
        if (program.nodes[id].step_data<DslNodeKind::Scatter>().has_child) {
            expect_child(id, 2, TypeSlot::i1(), "scatter", record_edges);
        }
        set_resolved_type(id, value_expected.has ? value_expected : maybe_resolved_type(child(id, 0)));
    }

    void assign_pack(NodeId id, bool record_edges) {
        TypeSlot value_expected = constrain_declared_type(id, expected_type(id), "pack");
        expect_child(id, 0, value_expected, "pack", record_edges);
        expect_child(id, 1, TypeSlot::i1(), "pack", record_edges);
        set_resolved_type(id, value_expected.has ? value_expected : maybe_resolved_type(child(id, 0)));
    }

    void assign_arith_agg(NodeId id, bool final, bool record_edges) {
        DslNode &node = program.nodes[id];
        TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
        TypeSlot agg_dt{};
        if (arg_dt.is_timestamp() || declared_is_timestamp(id) || expected_type(id).is_timestamp()) {
            sj::ArithBinaryOp op = node.step_data<DslNodeKind::ArithAgg>().op;
            std::string_view op_name = sj::show_arith_binary_op(op);
            if (op != sj::ArithBinaryOp::Min && op != sj::ArithBinaryOp::Max) {
                throw std::invalid_argument(
                    sj::format("timestamp aggregate %.*s is not allowed at expression", SV(op_name)));
            }
            agg_dt = constrain_declared_type(id, expected_type(id), "aggregate");
            if (!agg_dt.has) { agg_dt = arg_dt; }
        } else {
            agg_dt = constrain_declared_type(id, expected_type(id), "aggregate");
            if (!agg_dt.has) { agg_dt = arg_dt; }
            TypeSlot check = agg_dt.has ? agg_dt : arg_dt;
            check.require_value_type_if_known("aggregate", "aggregate requires numeric value type");
        }
        expect_child(id, 0, agg_dt, "aggregate", record_edges);
        if (node.step_data<DslNodeKind::ArithAgg>().has_cond) {
            expect_child(id, 1, TypeSlot::i1(), "aggregate", record_edges);
        }
        if (final && !agg_dt.has) { throw std::invalid_argument("unresolved aggregate type at expression"); }
        set_resolved_type(id, agg_dt);
    }

    void assign_grouped_arith_agg(NodeId id, bool final, bool record_edges) {
        maybe_resolved_type(child(id, 1))
            .require_int_if_known("grouped aggregate index", "grouped aggregate index must be integer");
        TypeSlot agg_dt = constrain_declared_type(id, expected_type(id), "grouped aggregate");
        if (!agg_dt.has) { agg_dt = maybe_resolved_type(child(id, 0)); }
        expect_child(id, 0, agg_dt, "grouped aggregate", record_edges);
        if (program.nodes[id].step_data<DslNodeKind::GroupedArithAgg>().has_cond) {
            expect_child(id, 2, TypeSlot::i1(), "grouped aggregate", record_edges);
        }
        if (final && !agg_dt.has) { throw std::invalid_argument("unresolved grouped aggregate type at expression"); }
        set_resolved_type(id, agg_dt);
    }

    void assign_function(NodeId id, bool final, bool record_edges) {
        const DslNode &node = program.nodes[id];
        const DslFunctionSpec &spec = *find_dsl_function_spec(node.step_data<DslNodeKind::Function>().kind);
        check_function_arity(spec, node.child_count);
        switch (spec.group) {
        case DslFunctionGroup::TimestampExtract: {
            TypeSlot dt = constrain_declared_type(id, expected_type(id), "function");
            if (!dt.has) { dt = TypeSlot::i32(); }
            if (dt != TypeSlot::i32()) {
                throw std::invalid_argument(
                    sj::format("function %.*s result type must be I32 at expression", SV(spec.name)));
            }
            TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
            if (arg_dt.has && !arg_dt.is_timestamp()) {
                throw std::invalid_argument(
                    sj::format("function %.*s requires timestamp input at expression", SV(spec.name)));
            }
            if (final && !arg_dt.has) {
                throw std::invalid_argument(
                    sj::format("function %.*s requires a resolvable timestamp input at expression", SV(spec.name)));
            }
            set_resolved_type(id, dt);
            return;
        }
        case DslFunctionGroup::Coalesce:
        case DslFunctionGroup::NullIf: {
            TypeSlot dt{};
            for (uint32_t i = 0; i < node.child_count; ++i) {
                TypeSlot arg_dt = maybe_resolved_type(child(id, i));
                if (arg_dt.is_predicate()) {
                    throw std::invalid_argument(
                        sj::format("function %.*s requires value inputs at expression", SV(spec.name)));
                }
                dt = choose_promoted_numeric_type("function", TypeSlot::none(), dt, arg_dt);
            }
            dt = unify_hard(constrain_declared_type(id, expected_type(id), "function"), dt, "function");
            if (!dt.has) {
                throw std::invalid_argument(
                    sj::format("function %.*s requires a resolvable argument type at expression", SV(spec.name)));
            }
            for (uint32_t i = 0; i < node.child_count; ++i) {
                expect_child(id, i, dt, "function", record_edges);
                TypeSlot arg_dt = maybe_resolved_type(child(id, i));
                if (arg_dt.has && arg_dt != dt && !arg_dt.can_promote_numeric_to(dt)) {
                    throw std::invalid_argument(sj::format("function %.*s argument %u must have type %s at expression",
                                                           SV(spec.name), i, dt.type.name()));
                }
                arg_dt.require_value_type_if_known("function argument", "function requires value inputs");
            }
            set_resolved_type(id, dt);
            return;
        }
        case DslFunctionGroup::NullPredicate: {
            TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
            arg_dt.require_value_type_if_known("function argument", "function requires value input");
            if (final && !arg_dt.has) {
                throw std::invalid_argument(
                    sj::format("function %.*s requires a resolvable value input at expression", SV(spec.name)));
            }
            TypeSlot dt = constrain_declared_type(id, expected_type(id), "function");
            if (!dt.has) { dt = TypeSlot::i1(); }
            if (dt != TypeSlot::i1()) {
                throw std::invalid_argument(
                    sj::format("function %.*s result type must be I1 at expression", SV(spec.name)));
            }
            set_resolved_type(id, dt);
            return;
        }
        case DslFunctionGroup::IntegerUnary: {
            TypeSlot arg_dt = maybe_resolved_type(child(id, 0));
            TypeSlot dt = constrain_declared_type(id, expected_type(id), "function");
            if (!dt.has) { dt = arg_dt; }
            if (final && !dt.has) {
                throw std::invalid_argument(
                    sj::format("function %.*s requires a resolvable argument type at expression", SV(spec.name)));
            }
            expect_child(id, 0, dt, "function", record_edges);
            arg_dt.require_int_if_known("function argument", "function requires integer input");
            if (dt.has && arg_dt.has && dt != arg_dt && !arg_dt.can_promote_numeric_to(dt)) {
                throw std::invalid_argument(
                    sj::format("function %.*s result type must match argument type at expression", SV(spec.name)));
            }
            set_resolved_type(id, dt);
            return;
        }
        }
        SIMJIT_UNREACHABLE();
    }

    void assign_local(NodeId id, bool final, bool record_edges) {
        const DslNode &node = program.nodes[id];
        switch (node.kind) {
        case DslNodeKind::Const: assign_const(id, final); return;
        case DslNodeKind::Load:
        case DslNodeKind::LoadSplat: assign_load(id, final); return;
        case DslNodeKind::Index: assign_index(id); return;
        case DslNodeKind::Gather: assign_gather(id, final); return;
        case DslNodeKind::ArithBinary: assign_arith_binary(id, final, record_edges); return;
        case DslNodeKind::PredicateBinary:
            expect_child(id, 0, TypeSlot::i1(), "predicate", record_edges);
            expect_child(id, 1, TypeSlot::i1(), "predicate", record_edges);
            set_resolved_type(id, TypeSlot::i1());
            return;
        case DslNodeKind::ArithUnary: assign_arith_unary(id, final, record_edges); return;
        case DslNodeKind::PredicateNot:
            expect_child(id, 0, TypeSlot::i1(), "predicate", record_edges);
            set_resolved_type(id, TypeSlot::i1());
            return;
        case DslNodeKind::Compare: assign_compare(id, final, record_edges); return;
        case DslNodeKind::IntCast: assign_int_cast(id, final, record_edges); return;
        case DslNodeKind::FloatCast: assign_float_cast(id); return;
        case DslNodeKind::BitCast: assign_bit_cast(id, final, record_edges); return;
        case DslNodeKind::Select: assign_select(id, final, record_edges); return;
        case DslNodeKind::FpClass:
            maybe_resolved_type(child(id, 0)).require_float_if_known("fpclass input", "fpclass requires float input");
            set_resolved_type(id, TypeSlot::i1());
            return;
        case DslNodeKind::Permute: {
            TypeSlot arg_expected = constrain_declared_type(id, expected_type(id), "permute");
            expect_child(id, 0, arg_expected, "permute", record_edges);
            set_resolved_type(id, arg_expected.has ? arg_expected : maybe_resolved_type(child(id, 0)));
            return;
        }
        case DslNodeKind::Store: assign_store_like(id, record_edges); return;
        case DslNodeKind::Scatter: assign_scatter(id, record_edges); return;
        case DslNodeKind::Pack: assign_pack(id, record_edges); return;
        case DslNodeKind::ArithAgg: assign_arith_agg(id, final, record_edges); return;
        case DslNodeKind::PredicateAgg:
            expect_child(id, 0, TypeSlot::i1(), "predicate aggregate", record_edges);
            set_resolved_type(id, TypeSlot::i1());
            return;
        case DslNodeKind::CountIf:
            expect_child(id, 0, TypeSlot::i1(), "count_if", record_edges);
            set_resolved_type(id, TypeSlot::i64());
            return;
        case DslNodeKind::GroupedArithAgg: assign_grouped_arith_agg(id, final, record_edges); return;
        case DslNodeKind::Function: assign_function(id, final, record_edges); return;
        }
        SIMJIT_UNREACHABLE();
    }

    void rewrite_expected_edges() {
        for (NodeId id : postorder) {
            const DslNode &node = program.nodes[id];
            for (uint32_t i = 0; i < node.child_count; ++i) {
                uint32_t edge = node.first_child + i;
                if (edge >= edge_expected.size()) { continue; }
                TypeSlot target = edge_expected[edge];
                if (!target.has) { continue; }
                set_child(id, i, coerce_promoted_node(child(id, i), target, dsl_node_kind_name(node.kind)));
            }
        }
    }

    void recompute_resolved_types() {
        collect_graph();
        bool previous_mutate_expected_types = mutate_expected_types;
        mutate_expected_types = false;
        for (NodeId id : postorder) {
            assign_local(id, true, false);
        }
        mutate_expected_types = previous_mutate_expected_types;
    }

    bool rewrite_output_roots() {
        bool changed_root = false;
        for (uint32_t i = 0; i < program.output_count; ++i) {
            DslOutput &out = program.outputs[i];
            TypeSlot expected = output_type_for(out.name);
            TypeSlot actual = maybe_resolved_type(out.root);
            if (!expected.has || !actual.has || actual == expected) { continue; }
            if (is_final_output_node(program.nodes[out.root].kind)) {
                throw_output_type_conflict(out, expected, actual);
            }
            if (!actual.can_promote_numeric_to(expected)) { throw_output_type_conflict(out, expected, actual); }
            if (actual.is_int() && expected.is_int()) {
                out.root = append_cast(DslNodeKind::IntCast, expected, out.root);
                changed_root = true;
                continue;
            }
            if (actual.is_float() && expected.is_float()) {
                out.root = append_cast(DslNodeKind::FloatCast, expected, out.root);
                changed_root = true;
                continue;
            }
            throw_output_type_conflict(out, expected, actual);
        }
        return changed_root;
    }

    void simplify_output_roots() {
        collect_graph();
        constexpr NodeId kInvalidNode = std::numeric_limits<NodeId>::max();
        std::vector<NodeId> simplified(program.nodes.size(), kInvalidNode);
        for (NodeId id : postorder) {
            const DslNode &node = program.nodes[id];
            for (uint32_t i = 0; i < node.child_count; ++i) {
                NodeId kid = child(id, i);
                if (kid < simplified.size() && simplified[kid] != kInvalidNode) { set_child(id, i, simplified[kid]); }
            }
            simplified[id] = is_noop_cast(id) ? child(id, 0) : id;
        }
        for (uint32_t i = 0; i < program.output_count; ++i) {
            NodeId root = program.outputs[i].root;
            if (root < simplified.size() && simplified[root] != kInvalidNode) {
                program.outputs[i].root = simplified[root];
            }
        }
        collect_graph();
    }

    void check_function_arity(const DslFunctionSpec &spec, uint32_t count) const {
        if (spec.accepts_arg_count(count)) { return; }
        if (spec.min_args == spec.max_args) {
            throw std::invalid_argument(sj::format("function %.*s expects %u argument%s at expression", SV(spec.name),
                                                   static_cast<unsigned>(spec.min_args),
                                                   spec.min_args == 1 ? "" : "s"));
        }
        if (spec.max_args == kDslVariadicFunctionArgs) {
            throw std::invalid_argument(sj::format("function %.*s expects at least %u argument%s at expression",
                                                   SV(spec.name), static_cast<unsigned>(spec.min_args),
                                                   spec.min_args == 1 ? "" : "s"));
        }
        throw std::invalid_argument(sj::format("function %.*s expects %u to %u arguments at expression", SV(spec.name),
                                               static_cast<unsigned>(spec.min_args),
                                               static_cast<unsigned>(spec.max_args)));
    }

    bool is_noop_cast(NodeId id) const {
        const DslNode &node = program.nodes[id];
        if (node.child_count != 1) { return false; }
        if (node.kind != DslNodeKind::IntCast && node.kind != DslNodeKind::FloatCast &&
            node.kind != DslNodeKind::BitCast) {
            return false;
        }
        TypeSlot resolved = resolved_type_slot(id, "cast simplification");
        TypeSlot arg = resolved_type_slot(child(id, 0), "cast simplification input");
        if (arg != resolved) { return false; }
        if (node.kind == DslNodeKind::IntCast) {
            return node.step_data<DslNodeKind::IntCast>().kind != DslIntCastKind::Trunc;
        }
        return true;
    }
};

void resolve_native_dsl_program(DslProgram &program, DslTypeMap input_types, DslTypeMap output_types) {
    NativeDslResolver resolver{program, input_types, output_types};
    resolver.resolve_in_place();
}

static bool native_output_is_scalar(const DslNode &node) {
    return node.kind == DslNodeKind::ArithAgg || node.kind == DslNodeKind::PredicateAgg ||
           node.kind == DslNodeKind::CountIf;
}

NodeId native_child(const DslProgram &program, NodeId id, uint32_t index) {
    const DslNode &node = program.nodes[id];
    SIMJIT_ASSERT(index < node.child_count);
    return program.child_edges[node.first_child + index];
}

static bool buffer_is_nullable(const NameMap<BufferDesc> &buffers, std::string_view name) {
    const BufferDesc *buf = buffers.find(name);
    return buf == nullptr || buf->null.kind != BufferNullKind::None;
}

static bool native_expr_may_be_nullable(const DslProgram &program, NodeId id, const NameMap<BufferDesc> &inputs) {
    const DslNode &node = program.nodes[id];
    const DslNode *step = &node;
    switch (node.kind) {
        SIMJIT_MATCH (DslNodeKind::Load) { return buffer_is_nullable(inputs, program.string(data.name)); }
        SIMJIT_MATCH (DslNodeKind::LoadSplat) { return buffer_is_nullable(inputs, program.string(data.name)); }
    case DslNodeKind::Const:
    case DslNodeKind::Index:
    case DslNodeKind::ArithAgg:
    case DslNodeKind::PredicateAgg:
    case DslNodeKind::CountIf:
    case DslNodeKind::GroupedArithAgg:
        return false;
        SIMJIT_MATCH (DslNodeKind::Gather) {
            return buffer_is_nullable(inputs, program.string(data.name)) ||
                   native_expr_may_be_nullable(program, native_child(program, id, 0), inputs);
        }
        SIMJIT_MATCH (DslNodeKind::Function) {
            if (data.kind == DslFunctionKind::Coalesce) {
                for (uint32_t i = 0; i < node.child_count; ++i) {
                    if (!native_expr_may_be_nullable(program, native_child(program, id, i), inputs)) { return false; }
                }
                return true;
            }
            if (data.kind == DslFunctionKind::NullIf) { return true; }
            if (data.kind == DslFunctionKind::IsNull || data.kind == DslFunctionKind::IsNotNull) { return false; }
            break;
        }
    case DslNodeKind::ArithBinary:
    case DslNodeKind::PredicateBinary:
    case DslNodeKind::ArithUnary:
    case DslNodeKind::PredicateNot:
    case DslNodeKind::Compare:
    case DslNodeKind::IntCast:
    case DslNodeKind::FloatCast:
    case DslNodeKind::BitCast:
    case DslNodeKind::Select:
    case DslNodeKind::FpClass:
    case DslNodeKind::Permute:
    case DslNodeKind::Store:
    case DslNodeKind::Scatter:
    case DslNodeKind::Pack: break;
    }
    for (uint32_t i = 0; i < node.child_count; ++i) {
        if (native_expr_may_be_nullable(program, native_child(program, id, i), inputs)) { return true; }
    }
    return false;
}

static LogicalType resolved_output_type(const DslProgram &program, const DslOutput &out) {
    const DslTypeSlot &slot = program.resolved_types[out.root];
    if (!slot.has) { throw std::invalid_argument(sj::format("resolved output %.*s is missing dtype", SV(out.name))); }
    return slot.type;
}

static size_t logical_item_size(const LogicalType &ty) {
    if (ty.is_timestamp()) { return 8; }
    switch (ty.dtype) {
    case sj::ScalarDataType::I1: return 1;
    case sj::ScalarDataType::I8: return 1;
    case sj::ScalarDataType::I16: return 2;
    case sj::ScalarDataType::I32: return 4;
    case sj::ScalarDataType::I64: return 8;
    case sj::ScalarDataType::F32: return 4;
    case sj::ScalarDataType::F64: return 8;
    case sj::ScalarDataType::I128: break;
    }
    throw std::invalid_argument(sj::format("unsupported simjit scalar type %s", ty.name()));
}

std::vector<NativeRuntimeOutputSpec> plan_native_runtime_outputs(const DslProgram &program,
                                                                 const NameMap<BufferDesc> &inputs,
                                                                 NativeRuntimeOutputKind output_kind, size_t n) {
    std::vector<NativeRuntimeOutputSpec> specs;
    specs.reserve(program.output_count);
    for (uint32_t i = 0; i < program.output_count; ++i) {
        const DslOutput &out = program.outputs[i];
        LogicalType ty = resolved_output_type(program, out);
        bool scalar = native_output_is_scalar(program.nodes[out.root]);
        size_t length = scalar ? 1 : n;
        bool bitpacked = output_kind == NativeRuntimeOutputKind::Arrow && ty.dtype == sj::ScalarDataType::I1;
        bool nullable =
            output_kind == NativeRuntimeOutputKind::Arrow && native_expr_may_be_nullable(program, out.root, inputs);
        size_t data_bytes = bitpacked ? (length + 7) / 8 : logical_item_size(ty) * length;
        specs.push_back(NativeRuntimeOutputSpec{out.name, ty, length, scalar, bitpacked, nullable, data_bytes,
                                                nullable ? (length + 7) / 8 : 0});
    }
    return specs;
}

} // namespace simjit_python
