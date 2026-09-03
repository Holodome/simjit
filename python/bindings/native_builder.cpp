// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "native_builder.h"

#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "simjit/core/expr.h"
#include "simjit/dynamic_value.h"
#include "simjit/nullable.h"

namespace simjit_python {

bool native_program_requires_safety_check(const DslProgram &program) {
    for (uint32_t i = 0; i < program.node_count; ++i) {
        const DslNode &node = program.nodes[i];
        switch (node.kind) {
        case DslNodeKind::ArithBinary:
            if (node.step_data<DslNodeKind::ArithBinary>().checked) { return true; }
            break;
        case DslNodeKind::ArithUnary:
            if (node.step_data<DslNodeKind::ArithUnary>().checked) { return true; }
            break;
        case DslNodeKind::IntCast:
            if (node.step_data<DslNodeKind::IntCast>().checked) { return true; }
            break;
        default: break;
        }
    }
    return false;
}

namespace {

#define SV(x) static_cast<int>((x).size()), ((x).data() == nullptr ? "" : (x).data())

#define throw_native_type_error(message) throw NativeBuilderError(NativeBuilderErrorKind::Type, message)
#define throw_native_value_error(message) throw NativeBuilderError(NativeBuilderErrorKind::Value, message)
#define throw_native_index_error(message) throw NativeBuilderError(NativeBuilderErrorKind::Index, message)

static sj::Value floor_div_const(sj::FunctionBuilder &builder, sj::Value arg, int64_t divisor) {
    sj::Value divisor_v = builder.i64(divisor);
    sj::Value q = builder.div(arg, divisor_v);
    sj::Value r = builder.mod(arg, divisor_v);
    sj::Predicate needs_adjust = builder.and_(builder.cmp_ne(r, builder.i64(0)), builder.cmp_lt(arg, builder.i64(0)));
    return builder.sub(q, builder.select(needs_adjust, builder.i64(1), builder.i64(0)));
}

static sj::Value floor_mod_const(sj::FunctionBuilder &builder, sj::Value arg, int64_t divisor) {
    sj::Value divisor_v = builder.i64(divisor);
    sj::Value r = builder.mod(arg, divisor_v);
    sj::Predicate is_neg = builder.cmp_lt(r, builder.i64(0));
    return builder.select(is_neg, builder.add(r, divisor_v), r);
}

struct CivilDateValues {
    sj::Value year;
    sj::Value month;
    sj::Value day;
};

// FIXME: Don't return all three at once
static CivilDateValues civil_from_days(sj::FunctionBuilder &builder, sj::Value days_since_epoch) {
    sj::Value z = builder.add(days_since_epoch, builder.i64(719468));
    sj::Value era_adj = builder.select(builder.cmp_ge(z, builder.i64(0)), z, builder.sub(z, builder.i64(146096)));
    sj::Value era = builder.div(era_adj, builder.i64(146097));
    sj::Value doe = builder.sub(z, builder.mul(era, builder.i64(146097)));
    sj::Value yoe = builder.div(builder.sub(builder.add(builder.sub(doe, builder.div(doe, builder.i64(1460))),
                                                        builder.div(doe, builder.i64(36524))),
                                            builder.div(doe, builder.i64(146096))),
                                builder.i64(365));
    sj::Value y = builder.add(yoe, builder.mul(era, builder.i64(400)));
    sj::Value doy =
        builder.sub(doe, builder.add(builder.add(builder.mul(builder.i64(365), yoe), builder.div(yoe, builder.i64(4))),
                                     builder.negate(builder.div(yoe, builder.i64(100)))));
    sj::Value mp = builder.div(builder.add(builder.mul(builder.i64(5), doy), builder.i64(2)), builder.i64(153));
    sj::Value d = builder.add(
        builder.sub(doy, builder.div(builder.add(builder.mul(builder.i64(153), mp), builder.i64(2)), builder.i64(5))),
        builder.i64(1));
    sj::Value m = builder.add(mp, builder.select(builder.cmp_lt(mp, builder.i64(10)), builder.i64(3), builder.i64(-9)));
    y = builder.add(y, builder.select(builder.cmp_le(m, builder.i64(2)), builder.i64(1), builder.i64(0)));
    return {y, m, d};
}

static bool logical_same_sign(const LogicalType &lhs, const LogicalType &rhs) {
    if (!lhs.is_int() || !rhs.is_int()) { return true; }
    return lhs.is_unsigned == rhs.is_unsigned;
}

static void check_no_mixed_sign(std::string_view op_name, const LogicalType &lhs, const LogicalType &rhs) {
    if (lhs.is_int() && rhs.is_int() && !logical_same_sign(lhs, rhs)) {
        throw_native_value_error(sj::format(
            "mixed signedness is not allowed for %.*s: left=%s right=%s", (int)op_name.length(), op_name.data(),
            lhs.is_unsigned ? "unsigned" : "signed", rhs.is_unsigned ? "unsigned" : "signed"));
    }
}

static bool logical_same_timestamp(const LogicalType &lhs, const LogicalType &rhs) {
    return lhs.is_timestamp() && rhs.is_timestamp() && lhs.unit == rhs.unit && lhs.timezone == rhs.timezone;
}

static sj::ArithBinaryOpFlags arith_binary_flags(sj::ArithBinaryOp op, bool checked) {
    switch (op) {
    case sj::ArithBinaryOp::ShiftLeftLogical:
    case sj::ArithBinaryOp::ShiftRightLogical:
    case sj::ArithBinaryOp::ShiftRightArith:
    case sj::ArithBinaryOp::RotateLeft:
    case sj::ArithBinaryOp::RotateRight:
        return checked ? sj::ArithBinaryOpFlags::SafetyCheck : sj::ArithBinaryOpFlags::ShiftWraparound;
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
    case sj::ArithBinaryOp::UMax:
    case sj::ArithBinaryOp::And:
    case sj::ArithBinaryOp::Or:
    case sj::ArithBinaryOp::Xor:
    case sj::ArithBinaryOp::AndNot: return checked ? sj::ArithBinaryOpFlags::SafetyCheck : sj::ArithBinaryOpFlags::No;
    }
    SIMJIT_UNREACHABLE();
}

static int64_t timestamp_unit_scale(TimestampUnit unit) {
    switch (unit) {
    case TimestampUnit::Seconds: return 1;
    case TimestampUnit::Milliseconds: return 1000;
    case TimestampUnit::Microseconds: return 1000000;
    case TimestampUnit::Nanoseconds: return 1000000000;
    }
    SIMJIT_UNREACHABLE();
}

static bool dsl_node_is_final(DslNodeKind kind) {
    return kind == DslNodeKind::Store || kind == DslNodeKind::Scatter || kind == DslNodeKind::Pack ||
           kind == DslNodeKind::ArithAgg || kind == DslNodeKind::PredicateAgg || kind == DslNodeKind::CountIf ||
           kind == DslNodeKind::GroupedArithAgg;
}

enum class LoweredValueKind {
    Invalid,
    Value,
    Predicate,
    NullableValue,
    NullablePredicate,
};

class LoweredValue {
public:
    LoweredValue() : kind(LoweredValueKind::Invalid) {}
    LoweredValue(sj::Value x) : kind(LoweredValueKind::Value), inner(x) {}
    LoweredValue(sj::Predicate x) : kind(LoweredValueKind::Predicate), inner(x) {}
    LoweredValue(sj::nullable::NullableValue x) : kind(LoweredValueKind::NullableValue), inner(x.v), null(x.null) {}
    LoweredValue(sj::nullable::NullablePredicate x)
        : kind(LoweredValueKind::NullablePredicate), inner(x.v), null(x.null) {}

    static LoweredValue nullable_value(sj::nullable::NullableValue x) { return LoweredValue(x); }
    static LoweredValue nullable_predicate(sj::nullable::NullablePredicate x) { return LoweredValue(x); }

    sj::Value as_value() const {
        if (kind != LoweredValueKind::Value) throw std::runtime_error("Invalid value");
        return value_payload();
    }

    sj::Predicate as_predicate() const {
        if (kind != LoweredValueKind::Predicate) throw std::runtime_error("Invalid predicate");
        return predicate_payload();
    }

    sj::nullable::NullableValue as_nullable_value() const {
        if (kind != LoweredValueKind::NullableValue) throw std::runtime_error("Invalid nullable value");
        return {value_payload(), null};
    }

    sj::nullable::NullablePredicate as_nullable_predicate() const {
        if (kind != LoweredValueKind::NullablePredicate) throw std::runtime_error("Invalid nullable predicate");
        return {predicate_payload(), null};
    }

    sj::nullable::NullableValue to_nullable_value() const {
        if (kind == LoweredValueKind::Value) { return {value_payload(), sj::MaybePredicate{}}; }
        if (kind == LoweredValueKind::NullableValue) { return as_nullable_value(); }
        throw std::runtime_error("Invalid value");
    }

    sj::nullable::NullablePredicate to_nullable_predicate() const {
        if (kind == LoweredValueKind::Predicate) { return {predicate_payload(), sj::MaybePredicate{}}; }
        if (kind == LoweredValueKind::NullablePredicate) { return as_nullable_predicate(); }
        throw std::runtime_error("Invalid predicate");
    }

    sj::Value as_known_non_null_value(std::string_view context) const {
        if (kind == LoweredValueKind::Value) { return value_payload(); }
        if (kind == LoweredValueKind::NullableValue) {
            if (null.is_valid()) {
                throw_native_value_error(sj::format("nullable value is not allowed for %.*s", SV(context)));
            }
            return value_payload();
        }
        throw std::runtime_error("Invalid value");
    }

    sj::Predicate as_known_non_null_predicate(std::string_view context) const {
        if (kind == LoweredValueKind::Predicate) { return predicate_payload(); }
        if (kind == LoweredValueKind::NullablePredicate) {
            if (null.is_valid()) {
                throw_native_value_error(sj::format("nullable predicate is not allowed for %.*s", SV(context)));
            }
            return predicate_payload();
        }
        throw std::runtime_error("Invalid predicate");
    }

    bool is_predicate() const {
        return kind == LoweredValueKind::Predicate || kind == LoweredValueKind::NullablePredicate;
    }
    bool is_value() const { return kind == LoweredValueKind::Value || kind == LoweredValueKind::NullableValue; }
    bool is_nullable() const {
        return kind == LoweredValueKind::NullableValue || kind == LoweredValueKind::NullablePredicate;
    }
    bool is_nullable_value() const { return kind == LoweredValueKind::NullableValue; }
    bool is_nullable_predicate() const { return kind == LoweredValueKind::NullablePredicate; }

    bool is_valid() const { return kind != LoweredValueKind::Invalid; }
    bool has_null() const { return null.is_valid(); }

private:
    sj::Value value_payload() const {
        auto x = inner.as_value();
        SIMJIT_ASSERT(x);
        return x.value();
    }

    sj::Predicate predicate_payload() const {
        auto x = inner.as_predicate();
        SIMJIT_ASSERT(x);
        return x.value();
    }

    LoweredValueKind kind;
    sj::DynamicValue inner{};
    sj::MaybePredicate null{};
};

static sj::ScalarDataType mask_load_type(const BufferNullDesc &null, std::string_view name) {
    sj::ScalarDataType dt = sj::ScalarDataType::I8;
    if (null.kind == BufferNullKind::MaskBitpacked) {
        dt = sj::ScalarDataType::I1;
    } else if (null.kind == BufferNullKind::MaskBool) {
        dt = sj::ScalarDataType::I8;
    } else {
        throw_native_value_error(sj::format("buffer %.*s does not have a mask transport", SV(name)));
    }
    return dt;
}

struct NativeBuilderImpl {
    sj::FunctionBuilder &builder;
    sj::nullable::NullableBuilder nbuilder;

    sj::MemoryArena prep_arena{};
    NameMap<LogicalType> schema{};
    NameMap<sj::Argument> load_args{};
    NameMap<sj::Argument> load_null_args{};
    NameMap<sj::Argument> outputs{};
    NameMap<sj::Argument> output_nulls{};
    NameMap<BufferDesc> buffers{};
    NameMap<BufferUsageFlags> buffer_usage{};
    std::vector<NativePointerBinding> output_aliases{};
    std::optional<sj::Argument> safety_check_arg{};
    const DslProgram *native_program = nullptr;

    explicit NativeBuilderImpl(sj::FunctionBuilder &builder_arg) : builder(builder_arg), nbuilder(&builder_arg) {}

    std::string_view arena_concat(std::string_view lhs, std::string_view rhs) {
        size_t size = lhs.size() + rhs.size();
        char *data = static_cast<char *>(prep_arena.alloc(size));
        if (data == nullptr) { throw std::bad_alloc{}; }
        std::memcpy(data, lhs.data(), lhs.size());
        std::memcpy(data + lhs.size(), rhs.data(), rhs.size());
        return std::string_view(data, size);
    }

    const BufferDesc &get_buffer(std::string_view name) const {
        if (const BufferDesc *buf = buffers.find(name)) { return *buf; }
        throw_native_index_error(sj::format("buffer %.*s is missing", SV(name)));
    }

    sj::Argument get_arg(std::string_view name) {
        if (const sj::Argument *arg = load_args.find(name)) { return *arg; }
        const LogicalType *name_ty = schema.find(name);
        if (name_ty == nullptr) { throw_native_value_error(sj::format("argument %.*s not found in schema", SV(name))); }
        sj::ScalarDataType dt = name_ty->dtype;
        if (!get_buffer(name).bitpacked && dt == sj::ScalarDataType::I1) { dt = sj::ScalarDataType::I8; }
        sj::Argument idx = builder.arg(dt);
        load_args.insert_or_assign(name, idx);
        return idx;
    }

    void mark_buffer_usage(std::string_view name, BufferUsageFlags flag) {
        buffer_usage.get_or_insert(name, BufferUsageFlags::BufferUsageNone) |= flag;
    }

    sj::Argument get_output(std::string_view name, const LogicalType &ty) {
        sj::ScalarDataType dt = ty.dtype;
        if (!get_buffer(name).bitpacked && dt == sj::ScalarDataType::I1) { dt = sj::ScalarDataType::I8; }

        if (const sj::Argument *out = outputs.find(name)) {
            const LogicalType *type_it = schema.find(name);
            if (type_it != nullptr && (type_it->dtype != ty.dtype || type_it->is_unsigned != ty.is_unsigned)) {
                throw_native_value_error("...");
            }
            return *out;
        }

        sj::Argument out = builder.arg(dt);
        outputs.insert_or_assign(name, out);
        schema.insert_or_assign(name, ty);
        return out;
    }

    sj::Argument get_pack_size_output_alias(std::string_view name) {
        // The value and null-mask packs have identical conditions but independent accumulators. Give the second
        // accumulator its own ABI argument and bind both size arguments to the same user buffer.
        sj::Argument out = builder.arg(sj::ScalarDataType::I64);
        output_aliases.push_back(NativePointerBinding{out.idx_, name, false, true});
        return out;
    }

    sj::Argument get_predicate_output(std::string_view name) {
        if (const sj::Argument *out = outputs.find(name)) { return *out; }
        sj::Argument out = builder.arg(sj::ScalarDataType::I1);
        outputs.insert_or_assign(name, out);
        schema.insert_or_assign(name, {sj::ScalarDataType::I1, false});
        return out;
    }

    bool buffer_is_nullable(std::string_view name) const {
        const BufferDesc *buf = buffers.find(name);
        return buf != nullptr && buf->null.kind != BufferNullKind::None;
    }

    sj::Argument get_null_arg(std::string_view name) {
        if (const sj::Argument *arg = load_null_args.find(name)) { return *arg; }
        const BufferDesc &buf = get_buffer(name);
        sj::ScalarDataType dt = mask_load_type(buf.null, name);
        sj::Argument idx = builder.arg(dt);
        load_null_args.insert_or_assign(name, idx);
        return idx;
    }

    sj::Argument get_null_output(std::string_view name) {
        if (const sj::Argument *out = output_nulls.find(name)) { return *out; }
        const BufferDesc &buf = get_buffer(name);
        sj::ScalarDataType dt = mask_load_type(buf.null, name);
        sj::Argument out = builder.arg(dt);
        output_nulls.insert_or_assign(name, out);
        return out;
    }

    sj::Value sentinel_value(ConstPayload payload, const LogicalType &ty) {
        auto signed_value = [&]() -> int64_t {
            switch (payload.kind) {
            case ConstPayloadKind::SignedInt: return payload.signed_int;
            case ConstPayloadKind::UnsignedInt: return static_cast<int64_t>(payload.unsigned_int);
            case ConstPayloadKind::Bool: return payload.bool_value ? 1 : 0;
            case ConstPayloadKind::Float: return static_cast<int64_t>(payload.float_value);
            case ConstPayloadKind::None: break;
            }
            throw_native_type_error("invalid sentinel type");
        };
        auto unsigned_value = [&]() -> uint64_t {
            switch (payload.kind) {
            case ConstPayloadKind::UnsignedInt: return payload.unsigned_int;
            case ConstPayloadKind::SignedInt: return static_cast<uint64_t>(payload.signed_int);
            case ConstPayloadKind::Bool: return payload.bool_value ? 1u : 0u;
            case ConstPayloadKind::Float: return static_cast<uint64_t>(payload.float_value);
            case ConstPayloadKind::None: break;
            }
            throw_native_type_error("invalid sentinel type");
        };
        auto float_value = [&]() -> double {
            switch (payload.kind) {
            case ConstPayloadKind::Float: return payload.float_value;
            case ConstPayloadKind::SignedInt: return static_cast<double>(payload.signed_int);
            case ConstPayloadKind::UnsignedInt: return static_cast<double>(payload.unsigned_int);
            case ConstPayloadKind::Bool: return payload.bool_value ? 1.0 : 0.0;
            case ConstPayloadKind::None: break;
            }
            throw_native_type_error("invalid sentinel type");
        };

        if (ty.is_timestamp()) { return builder.i64(signed_value()); }
        switch (ty.dtype) {
        case sj::ScalarDataType::I8:
            return ty.is_unsigned ? builder.u8(static_cast<uint8_t>(unsigned_value()))
                                  : builder.i8(static_cast<int8_t>(signed_value()));
        case sj::ScalarDataType::I16:
            return ty.is_unsigned ? builder.u16(static_cast<uint16_t>(unsigned_value()))
                                  : builder.i16(static_cast<int16_t>(signed_value()));
        case sj::ScalarDataType::I32:
            return ty.is_unsigned ? builder.u32(static_cast<uint32_t>(unsigned_value()))
                                  : builder.i32(static_cast<int32_t>(signed_value()));
        case sj::ScalarDataType::I64:
            return ty.is_unsigned ? builder.u64(unsigned_value()) : builder.i64(signed_value());
        case sj::ScalarDataType::F32: return builder.f32(static_cast<float>(float_value()));
        case sj::ScalarDataType::F64: return builder.f64(float_value());
        case sj::ScalarDataType::I1: break;
        case sj::ScalarDataType::I128: break;
        }
        throw_native_type_error("invalid sentinel type");
    }

    sj::Value timestamp_to_seconds(sj::Value arg, TimestampUnit unit) {
        return floor_div_const(builder, arg, timestamp_unit_scale(unit));
    }

    sj::Value timestamp_to_days(sj::Value arg, TimestampUnit unit) {
        return floor_div_const(builder, arg, 86400 * timestamp_unit_scale(unit));
    }

    sj::Value lower_timestamp_year(sj::Value arg, const LogicalType &ty) {
        CivilDateValues civil = civil_from_days(builder, timestamp_to_days(arg, ty.unit));
        return builder.signed_cast(civil.year, sj::ScalarDataType::I32);
    }

    sj::Value lower_timestamp_month(sj::Value arg, const LogicalType &ty) {
        CivilDateValues civil = civil_from_days(builder, timestamp_to_days(arg, ty.unit));
        return builder.signed_cast(civil.month, sj::ScalarDataType::I32);
    }

    sj::Value lower_timestamp_day(sj::Value arg, const LogicalType &ty) {
        CivilDateValues civil = civil_from_days(builder, timestamp_to_days(arg, ty.unit));
        return builder.signed_cast(civil.day, sj::ScalarDataType::I32);
    }

    sj::Value lower_timestamp_hour(sj::Value arg, const LogicalType &ty) {
        sj::Value seconds = timestamp_to_seconds(arg, ty.unit);
        sj::Value sec_in_day = floor_mod_const(builder, seconds, 86400);
        return builder.signed_cast(builder.div(sec_in_day, builder.i64(3600)), sj::ScalarDataType::I32);
    }

    sj::Value lower_timestamp_minute(sj::Value arg, const LogicalType &ty) {
        sj::Value seconds = timestamp_to_seconds(arg, ty.unit);
        sj::Value sec_in_day = floor_mod_const(builder, seconds, 86400);
        sj::Value minute_val = builder.div(builder.mod(sec_in_day, builder.i64(3600)), builder.i64(60));
        return builder.signed_cast(minute_val, sj::ScalarDataType::I32);
    }

    sj::Value lower_timestamp_second(sj::Value arg, const LogicalType &ty) {
        sj::Value seconds = timestamp_to_seconds(arg, ty.unit);
        sj::Value second_val = floor_mod_const(builder, seconds, 60);
        return builder.signed_cast(second_val, sj::ScalarDataType::I32);
    }

    sj::Value lower_timestamp_day_of_week(sj::Value arg, const LogicalType &ty) {
        sj::Value days = timestamp_to_days(arg, ty.unit);
        sj::Value dow = floor_mod_const(builder, builder.add(days, builder.i64(3)), 7);
        return builder.signed_cast(dow, sj::ScalarDataType::I32);
    }

    const DslProgram &native_program_ref() const {
        SIMJIT_ASSERT(native_program != nullptr);
        return *native_program;
    }

    const DslNode &native_node(NodeId id) const {
        const DslProgram &program = native_program_ref();
        SIMJIT_ASSERT(id < program.node_count);
        return program.nodes[id];
    }

    std::string_view native_string(DslStringRef id) const { return native_program_ref().string(id); }

    std::string_view native_required_string(DslStringRef id, std::string_view field_name) const {
        return native_program_ref().required_string(id, field_name);
    }

    NodeId native_child(NodeId id, uint32_t index) const {
        const DslProgram &program = native_program_ref();
        const DslNode &node = native_node(id);
        SIMJIT_ASSERT(index < node.child_count);
        return program.child_edges[node.first_child + index];
    }

    LogicalType native_expr_type(NodeId id) const {
        const DslTypeSlot &resolved = native_program_ref().resolved_types[id];
        SIMJIT_ASSERT(resolved.has);
        return resolved.type;
    }

    sj::Value native_value_expr(NodeId id) { return native_dsl_expr(id).as_known_non_null_value("value expression"); }

    sj::Predicate native_predicate_expr(NodeId id) {
        return native_dsl_expr(id).as_known_non_null_predicate("predicate expression");
    }

    LoweredValue const_expr(NodeId id, const LogicalType &ty) {
        const ConstPayload payload = native_node(id).step_data<DslNodeKind::Const>();
        auto signed_value = [&]() -> int64_t {
            switch (payload.kind) {
            case ConstPayloadKind::SignedInt: return payload.signed_int;
            case ConstPayloadKind::UnsignedInt:
                if (payload.unsigned_int > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
                    throw_native_value_error("integer constant does not fit signed target");
                }
                return static_cast<int64_t>(payload.unsigned_int);
            case ConstPayloadKind::Bool:
            case ConstPayloadKind::Float:
            case ConstPayloadKind::None: throw_native_type_error("integer constant expected");
            }
            SIMJIT_UNREACHABLE();
        };
        auto unsigned_value = [&]() -> uint64_t {
            switch (payload.kind) {
            case ConstPayloadKind::UnsignedInt: return payload.unsigned_int;
            case ConstPayloadKind::SignedInt:
                if (payload.signed_int < 0) { throw_native_value_error("negative integer constant for unsigned type"); }
                return static_cast<uint64_t>(payload.signed_int);
            case ConstPayloadKind::Bool:
            case ConstPayloadKind::Float:
            case ConstPayloadKind::None: throw_native_type_error("integer constant expected");
            }
            SIMJIT_UNREACHABLE();
        };
        auto float_value = [&]() -> double {
            switch (payload.kind) {
            case ConstPayloadKind::Float: return payload.float_value;
            case ConstPayloadKind::SignedInt: return static_cast<double>(payload.signed_int);
            case ConstPayloadKind::UnsignedInt: return static_cast<double>(payload.unsigned_int);
            case ConstPayloadKind::Bool:
            case ConstPayloadKind::None: throw_native_type_error("float constant expected");
            }
            SIMJIT_UNREACHABLE();
        };

        if (ty.is_timestamp()) { return builder.i64(signed_value()); }
        switch (ty.dtype) {
        case sj::ScalarDataType::I8:
            return ty.is_unsigned ? builder.u8(static_cast<uint8_t>(unsigned_value()))
                                  : builder.i8(static_cast<int8_t>(signed_value()));
        case sj::ScalarDataType::I16:
            return ty.is_unsigned ? builder.u16(static_cast<uint16_t>(unsigned_value()))
                                  : builder.i16(static_cast<int16_t>(signed_value()));
        case sj::ScalarDataType::I32:
            return ty.is_unsigned ? builder.u32(static_cast<uint32_t>(unsigned_value()))
                                  : builder.i32(static_cast<int32_t>(signed_value()));
        case sj::ScalarDataType::I64:
            return ty.is_unsigned ? builder.u64(unsigned_value()) : builder.i64(signed_value());
        case sj::ScalarDataType::F32: return builder.f32(static_cast<float>(float_value()));
        case sj::ScalarDataType::F64: return builder.f64(float_value());
        case sj::ScalarDataType::I1:
            if (payload.kind != ConstPayloadKind::Bool) { throw_native_type_error("bool constant expected"); }
            return payload.bool_value ? builder.true_() : builder.false_();
        case sj::ScalarDataType::I128: throw_native_value_error("i128 constants are not supported");
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue load_expr(NodeId id, const LogicalType &ty) {
        const DslNode &node = native_node(id);
        std::string_view name = native_required_string(node.step_data<DslNodeKind::Load>().name, "LoadExpr.name");
        mark_buffer_usage(name, BufferUsageFlags::BufferUsageInputVector);
        sj::Argument arg = get_arg(name);
        const BufferDesc &buf = get_buffer(name);

        switch (buf.null.kind) {
        case BufferNullKind::None: {
            if (ty.dtype == sj::ScalarDataType::I1) {
                if (buf.bitpacked) { return builder.load_predicate(arg); }
                return builder.bool2bit(builder.load(arg));
            }
            return builder.load(arg, node.step_data<DslNodeKind::Load>().kind);
        }
        case BufferNullKind::MaskBitpacked: {
            if (ty.dtype == sj::ScalarDataType::I1) {
                return buf.bitpacked
                           ? nbuilder.nbit_load_predicate_ext(arg, get_null_arg(name), buf.null.true_means_null)
                           : nbuilder.bool2bit(
                                 nbuilder.nbit_load_ext(arg, get_null_arg(name), buf.null.true_means_null));
            }
            return nbuilder.nbit_load_ext(arg, get_null_arg(name), buf.null.true_means_null);
        }
        case BufferNullKind::MaskBool: {
            if (ty.dtype == sj::ScalarDataType::I1) {
                return buf.bitpacked
                           ? nbuilder.nbool_load_predicate_ext(arg, get_null_arg(name), buf.null.true_means_null)
                           : nbuilder.bool2bit(
                                 nbuilder.nbool_load_ext(arg, get_null_arg(name), buf.null.true_means_null));
            }
            return nbuilder.nbool_load_ext(arg, get_null_arg(name), buf.null.true_means_null);
        }
        case BufferNullKind::Sentinel:
            if (ty.dtype == simjit::ScalarDataType::I1) {
                throw_native_value_error("don't support sentinel null i1 load");
            }
            return nbuilder.nval_load(arg, sentinel_value(buf.null.sentinel, ty));
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue load_splat_expr(NodeId id) {
        std::string_view name =
            native_required_string(native_node(id).step_data<DslNodeKind::LoadSplat>().name, "LoadSplatExpr.name");
        mark_buffer_usage(name, BufferUsageFlags::BufferUsageInputSplat);
        const BufferDesc &buf = get_buffer(name);
        switch (buf.null.kind) {
        case BufferNullKind::None:
            if (buf.ty.dtype == simjit::ScalarDataType::I1) return builder.load_predicate_splat(get_arg(name));
            return builder.load_splat(get_arg(name));
        case BufferNullKind::MaskBitpacked:
            if (buf.ty.dtype == simjit::ScalarDataType::I1)
                return buf.bitpacked ? nbuilder.nbit_load_predicate_splat_ext(get_arg(name), get_null_arg(name),
                                                                              buf.null.true_means_null)
                                     : nbuilder.bool2bit(nbuilder.nbit_load_splat_ext(get_arg(name), get_null_arg(name),
                                                                                      buf.null.true_means_null));
            return nbuilder.nbit_load_splat_ext(get_arg(name), get_null_arg(name), buf.null.true_means_null);
        case BufferNullKind::MaskBool:
            if (buf.ty.dtype == simjit::ScalarDataType::I1)
                return buf.bitpacked ? nbuilder.nbool_load_predicate_splat_ext(get_arg(name), get_null_arg(name),
                                                                               buf.null.true_means_null)
                                     : nbuilder.bool2bit(nbuilder.nbool_load_splat_ext(
                                           get_arg(name), get_null_arg(name), buf.null.true_means_null));
            return nbuilder.nbool_load_splat_ext(get_arg(name), get_null_arg(name), buf.null.true_means_null);
        case BufferNullKind::Sentinel: {
            sj::Value value = builder.load_splat(get_arg(name));
            sj::Value sentinel = sentinel_value(buf.null.sentinel, buf.ty);
            return sj::nullable::NullableValue{value, builder.cmp_eq(value, sentinel)};
        }
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue gather_expr(NodeId id, const LogicalType &ty) {
        sj::Value idx = native_value_expr(native_child(id, 0));
        std::string_view name =
            native_required_string(native_node(id).step_data<DslNodeKind::Gather>().name, "GatherExpr.name");
        mark_buffer_usage(name, BufferUsageFlags::BufferUsageInputTable);
        const BufferDesc &buf = get_buffer(name);
        switch (buf.null.kind) {
        case BufferNullKind::None: return builder.gather(idx, get_arg(name));
        case BufferNullKind::MaskBitpacked: throw_native_value_error("bitpacked nullable gather is unsupported");
        case BufferNullKind::MaskBool: {
            return nbuilder.nbool_gather_ext(get_arg(name), get_null_arg(name), idx, buf.null.true_means_null);
        }
        case BufferNullKind::Sentinel:
            return nbuilder.nval_gather(get_arg(name), idx, sentinel_value(buf.null.sentinel, ty));
        }
        SIMJIT_UNREACHABLE();
    }

    sj::Value index_expr(NodeId, const LogicalType &ty) { return builder.index(ty.dtype); }

    LoweredValue arith_binary_expr(NodeId id) {
        const DslNode &node = native_node(id);
        sj::ArithBinaryOp op = node.step_data<DslNodeKind::ArithBinary>().op;
        const char *op_name = sj::show_arith_binary_op(op);
        LogicalType out_ty = native_expr_type(id);
        NodeId lhs_id = native_child(id, 0);
        NodeId rhs_id = native_child(id, 1);
        LogicalType lhs_ty = native_expr_type(lhs_id);
        LogicalType rhs_ty = native_expr_type(rhs_id);
        LoweredValue lhs_val = native_dsl_expr(lhs_id);
        LoweredValue rhs_val = native_dsl_expr(rhs_id);
        bool nullable = lhs_val.is_nullable() || rhs_val.is_nullable();
        if (lhs_ty.is_timestamp() || rhs_ty.is_timestamp() || out_ty.is_timestamp()) {
            auto lower_timestamp_arith = [&]() -> LoweredValue {
                if (nullable) {
                    return nbuilder.arith_binary(lhs_val.to_nullable_value(), rhs_val.to_nullable_value(), op);
                }
                return builder.arith_binary(lhs_val.as_value(), rhs_val.as_value(), op);
            };
            if ((op == sj::ArithBinaryOp::Min || op == sj::ArithBinaryOp::Max) && lhs_ty.is_timestamp() &&
                rhs_ty.is_timestamp()) {
                if (!logical_same_timestamp(lhs_ty, rhs_ty)) {
                    throw_native_value_error("timestamp operands must match");
                }
                return lower_timestamp_arith();
            }
            if (op == sj::ArithBinaryOp::Sub && lhs_ty.is_timestamp() && rhs_ty.is_timestamp()) {
                if (!logical_same_timestamp(lhs_ty, rhs_ty)) {
                    throw_native_value_error("timestamp operands must match");
                }
                return lower_timestamp_arith();
            }
            if ((op == sj::ArithBinaryOp::Add || op == sj::ArithBinaryOp::Sub) && lhs_ty.is_timestamp() &&
                rhs_ty.dtype == sj::ScalarDataType::I64 && !rhs_ty.is_timestamp()) {
                return lower_timestamp_arith();
            }
            if (op == sj::ArithBinaryOp::Add && lhs_ty.dtype == sj::ScalarDataType::I64 && !lhs_ty.is_timestamp() &&
                rhs_ty.is_timestamp()) {
                return lower_timestamp_arith();
            }
            throw_native_value_error(sj::format("timestamp operation %s is not allowed", op_name));
        }
        if ((op == sj::ArithBinaryOp::Div || op == sj::ArithBinaryOp::Mod || op == sj::ArithBinaryOp::Min ||
             op == sj::ArithBinaryOp::Max) &&
            lhs_ty.is_int() && rhs_ty.is_int()) {
            check_no_mixed_sign(op_name, lhs_ty, rhs_ty);
        }
        if ((op == sj::ArithBinaryOp::Div || op == sj::ArithBinaryOp::Mod || op == sj::ArithBinaryOp::Min ||
             op == sj::ArithBinaryOp::Max) &&
            lhs_ty.is_int() && rhs_ty.is_int() && lhs_ty.is_unsigned) {
            if (op == sj::ArithBinaryOp::Div) { op = sj::ArithBinaryOp::UDiv; }
            if (op == sj::ArithBinaryOp::Mod) { op = sj::ArithBinaryOp::UMod; }
            if (op == sj::ArithBinaryOp::Min) { op = sj::ArithBinaryOp::UMin; }
            if (op == sj::ArithBinaryOp::Max) { op = sj::ArithBinaryOp::UMax; }
        }
        sj::ArithBinaryOpFlags flags = arith_binary_flags(op, node.step_data<DslNodeKind::ArithBinary>().checked);
        if (nullable) {
            if (node.step_data<DslNodeKind::ArithBinary>().checked) {
                return nbuilder.arith_binary_checked(lhs_val.to_nullable_value(), rhs_val.to_nullable_value(), op);
            }
            return nbuilder.arith_binary(lhs_val.to_nullable_value(), rhs_val.to_nullable_value(), op);
        }
        return builder.arith_binary(lhs_val.as_value(), rhs_val.as_value(), op, flags);
    }

    LoweredValue predicate_binary_expr(NodeId id) {
        const DslNode &node = native_node(id);
        sj::PredicateBinaryOp op = node.step_data<DslNodeKind::PredicateBinary>().op;
        LoweredValue lhs = native_dsl_expr(native_child(id, 0));
        LoweredValue rhs = native_dsl_expr(native_child(id, 1));
        if (lhs.is_nullable() || rhs.is_nullable()) {
            return nbuilder.predicate_binary(lhs.to_nullable_predicate(), rhs.to_nullable_predicate(), op);
        }
        return builder.predicate_binary(lhs.as_predicate(), rhs.as_predicate(), op);
    }

    LoweredValue arith_unary_expr(NodeId id) {
        const DslNode &node = native_node(id);
        sj::ArithUnaryOp op = node.step_data<DslNodeKind::ArithUnary>().op;
        LoweredValue arg = native_dsl_expr(native_child(id, 0));
        if (arg.is_nullable()) {
            if (node.step_data<DslNodeKind::ArithUnary>().checked) {
                switch (op) {
                case sj::ArithUnaryOp::Negate: return nbuilder.negate_checked(arg.to_nullable_value());
                case sj::ArithUnaryOp::Abs: return nbuilder.abs_checked(arg.to_nullable_value());
                default: throw_native_value_error("checked unary operation is not supported for this operation");
                }
            }
            return nbuilder.arith_unary(arg.to_nullable_value(), op);
        }
        return builder.arith_unary(arg.as_value(), op, node.step_data<DslNodeKind::ArithUnary>().checked);
    }

    LoweredValue predicate_not_expr(NodeId id) {
        LoweredValue arg = native_dsl_expr(native_child(id, 0));
        if (arg.is_nullable()) { return nbuilder.not_(arg.to_nullable_predicate()); }
        return builder.not_(arg.as_predicate());
    }

    LoweredValue compare_expr(NodeId id) {
        const DslNode &node = native_node(id);
        sj::CmpOp op = node.step_data<DslNodeKind::Compare>().op;
        NodeId lhs_id = native_child(id, 0);
        NodeId rhs_id = native_child(id, 1);
        LoweredValue lhs_val = native_dsl_expr(lhs_id);
        LoweredValue rhs_val = native_dsl_expr(rhs_id);
        LogicalType lhs_ty = native_expr_type(lhs_id);
        LogicalType rhs_ty = native_expr_type(rhs_id);
        bool is_unsigned = node.step_data<DslNodeKind::Compare>().is_unsigned;
        if (lhs_ty.is_timestamp() || rhs_ty.is_timestamp()) {
            if (is_unsigned) { throw_native_value_error("unsigned compare is not allowed for timestamps"); }
            if (!logical_same_timestamp(lhs_ty, rhs_ty)) { throw_native_value_error("timestamp operands must match"); }
            bool nullable = lhs_val.is_nullable() || rhs_val.is_nullable();
            if (nullable) { return nbuilder.cmp(lhs_val.to_nullable_value(), rhs_val.to_nullable_value(), op); }
            return builder.cmp(lhs_val.as_value(), rhs_val.as_value(), op, false);
        }
        check_no_mixed_sign("compare", lhs_ty, rhs_ty);
        if (!is_unsigned && lhs_ty.is_int() && rhs_ty.is_int() && logical_same_sign(lhs_ty, rhs_ty)) {
            is_unsigned = lhs_ty.is_unsigned;
        }
        if (lhs_val.is_nullable() || rhs_val.is_nullable()) {
            sj::nullable::NullableValue lhs = lhs_val.to_nullable_value();
            sj::nullable::NullableValue rhs = rhs_val.to_nullable_value();
            if (!is_unsigned) { return nbuilder.cmp(lhs, rhs, op); }
            sj::MaybePredicate null;
            if (lhs.null.is_valid() && rhs.null.is_valid())
                null = builder.or_(lhs.null.value(), rhs.null.value());
            else if (lhs.null.is_valid())
                null = lhs.null;
            else
                null = rhs.null;
            return sj::nullable::NullablePredicate{builder.cmp(lhs.v, rhs.v, op, true), null};
        }
        return builder.cmp(lhs_val.as_value(), rhs_val.as_value(), op);
    }

    LoweredValue int_cast_expr(NodeId id, LogicalType ty) {
        const DslNode &node = native_node(id);
        DslIntCastKind cast_kind = node.step_data<DslNodeKind::IntCast>().kind;
        NodeId arg_id = native_child(id, 0);
        LoweredValue arg_val = native_dsl_expr(arg_id);
        LogicalType arg_ty = native_expr_type(arg_id);
        if (ty.is_timestamp()) {
            if (cast_kind != DslIntCastKind::Cast) {
                throw_native_value_error("only generic cast to timestamp is allowed");
            }
            if (arg_ty.dtype != sj::ScalarDataType::I64 || arg_ty.is_timestamp()) {
                throw_native_value_error("timestamp cast source must be i64");
            }
            return arg_val;
        }
        if (arg_ty.is_timestamp()) {
            if (cast_kind != DslIntCastKind::Cast || ty.dtype != sj::ScalarDataType::I64) {
                throw_native_value_error("timestamp can only cast to i64 with generic cast");
            }
            return arg_val;
        }
        switch (cast_kind) {
        case DslIntCastKind::Cast: {
            if (arg_ty.dtype == ty.dtype) { return arg_val; }
            if (arg_val.is_nullable()) {
                return ty.is_unsigned ? nbuilder.unsigned_cast(arg_val.to_nullable_value(), ty.dtype)
                                      : nbuilder.signed_cast(arg_val.to_nullable_value(), ty.dtype);
            }
            sj::Value arg = arg_val.as_value();
            if (ty.is_unsigned) { return builder.unsigned_cast(arg, ty.dtype); }
            return builder.signed_cast(arg, ty.dtype);
        }
        case DslIntCastKind::Signed: {
            if (arg_ty.dtype == ty.dtype) { return arg_val; }
            if (arg_val.is_nullable()) { return nbuilder.signed_cast(arg_val.to_nullable_value(), ty.dtype); }
            return builder.signed_cast(arg_val.as_value(), ty.dtype);
        }
        case DslIntCastKind::Unsigned: {
            if (arg_ty.dtype == ty.dtype) { return arg_val; }
            if (arg_val.is_nullable()) { return nbuilder.unsigned_cast(arg_val.to_nullable_value(), ty.dtype); }
            return builder.unsigned_cast(arg_val.as_value(), ty.dtype);
        }
        case DslIntCastKind::Trunc: {
            if (arg_val.is_nullable()) {
                if (node.step_data<DslNodeKind::IntCast>().checked) {
                    return nbuilder.trunc_checked(arg_val.to_nullable_value(), ty.dtype);
                }
                return nbuilder.trunc(arg_val.to_nullable_value(), ty.dtype);
            }
            if (node.step_data<DslNodeKind::IntCast>().checked) {
                return builder.trunc_checked(arg_val.as_value(), ty.dtype);
            }
            return builder.trunc(arg_val.as_value(), ty.dtype);
        }
        case DslIntCastKind::Sext: {
            if (arg_val.is_nullable()) { return nbuilder.sext(arg_val.to_nullable_value(), ty.dtype); }
            return builder.sext(arg_val.as_value(), ty.dtype);
        }
        case DslIntCastKind::Zext: {
            if (arg_val.is_nullable()) { return nbuilder.zext(arg_val.to_nullable_value(), ty.dtype); }
            return builder.zext(arg_val.as_value(), ty.dtype);
        }
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue float_cast_expr(NodeId id, LogicalType ty) {
        NodeId arg_id = native_child(id, 0);
        LoweredValue arg_val = native_dsl_expr(arg_id);
        LogicalType arg_ty = native_expr_type(arg_id);
        if (arg_ty.is_timestamp()) { throw_native_value_error("float cast is not allowed for timestamps"); }
        bool is_unsigned = native_node(id).step_data<DslNodeKind::FloatCast>().is_unsigned;
        if (!is_unsigned) { is_unsigned = arg_ty.is_unsigned; }
        if (arg_val.is_nullable()) { return nbuilder.float_cast(arg_val.to_nullable_value(), ty.dtype, is_unsigned); }
        return builder.float_cast(arg_val.as_value(), ty.dtype, is_unsigned);
    }

    LoweredValue bit_cast_expr(NodeId id, LogicalType ty) {
        NodeId arg_id = native_child(id, 0);
        LoweredValue arg_val = native_dsl_expr(arg_id);
        if (ty.is_timestamp() || native_expr_type(arg_id).is_timestamp()) {
            throw_native_value_error("bitcast is not allowed for timestamps");
        }
        sj::Value arg = arg_val.as_known_non_null_value("bitcast");
        if (arg.dtype() == ty.dtype) { return arg; }
        return builder.bitcast(arg, ty.dtype);
    }

    LoweredValue select_expr(NodeId id) {
        LoweredValue cond = native_dsl_expr(native_child(id, 0));
        LoweredValue truthy = native_dsl_expr(native_child(id, 1));
        LoweredValue falsy = native_dsl_expr(native_child(id, 2));
        if (!cond.is_nullable() && !truthy.is_nullable() && !falsy.is_nullable()) {
            return builder.select(cond.as_predicate(), truthy.as_value(), falsy.as_value());
        }
        return nbuilder.if_else(cond.to_nullable_predicate(), truthy.to_nullable_value(), falsy.to_nullable_value());
    }

    LoweredValue fpclass_expr(NodeId id) {
        LoweredValue arg = native_dsl_expr(native_child(id, 0));
        return builder.fpclass(arg.as_known_non_null_value("fpclass"),
                               native_node(id).step_data<DslNodeKind::FpClass>().flags);
    }

    LoweredValue permute_expr(NodeId id) {
        LoweredValue arg = native_dsl_expr(native_child(id, 0));
        const DslNode &node = native_node(id);
        return builder.permute(arg.as_known_non_null_value("permute"), node.step_data<DslNodeKind::Permute>().idxs,
                               node.step_data<DslNodeKind::Permute>().is_bit);
    }

    sj::Predicate encoded_null_mask(sj::MaybePredicate null, bool true_means_null) {
        sj::Predicate mask = null.is_valid() ? null.value() : builder.false_();
        return true_means_null ? mask : builder.not_(mask);
    }

    sj::nullable::NullableValue as_bool_value(LoweredValue value) {
        if (value.is_predicate()) { return nbuilder.bit2bool(value.to_nullable_predicate()); }
        return value.to_nullable_value();
    }

    void conditional_store_lowered_value(std::string_view output_name, sj::Argument out, LoweredValue value,
                                         sj::Predicate cond) {
        const BufferDesc &buf = get_buffer(output_name);
        if (buf.null.kind == BufferNullKind::None) {
            if (value.has_null()) {
                throw_native_value_error(
                    sj::format("nullable output %.*s requires null output metadata", SV(output_name)));
            }
            if (value.is_predicate()) {
                sj::Predicate predicate = value.as_known_non_null_predicate("conditional store");
                if (buf.bitpacked)
                    builder.cond_store(predicate, cond, out);
                else
                    builder.cond_store(builder.bit2bool(predicate), cond, out);
            } else {
                builder.cond_store(value.as_known_non_null_value("conditional store"), cond, out,
                                   sj::LoadStoreKind::Unaligned);
            }
            return;
        }

        sj::nullable::NullableValue stored = as_bool_value(value);
        if (buf.null.kind == BufferNullKind::Sentinel) {
            if (value.is_predicate()) {
                throw_native_value_error("sentinel null output is unsupported for predicates");
            }
            sj::Predicate is_null = stored.null.is_valid() ? stored.null.value() : builder.false_();
            builder.cond_store(builder.select(is_null, sentinel_value(buf.null.sentinel, buf.ty), stored.v), cond, out,
                               sj::LoadStoreKind::Unaligned);
            return;
        }

        if (value.is_predicate() && buf.bitpacked)
            builder.cond_store(value.to_nullable_predicate().v, cond, out);
        else
            builder.cond_store(stored.v, cond, out, sj::LoadStoreKind::Unaligned);

        sj::Argument out_null = get_null_output(output_name);
        sj::Predicate null_mask = encoded_null_mask(stored.null, buf.null.true_means_null);
        if (buf.null.kind == BufferNullKind::MaskBitpacked)
            builder.cond_store(null_mask, cond, out_null);
        else
            builder.cond_store(builder.bit2bool(null_mask), cond, out_null, sj::LoadStoreKind::Unaligned);
    }

    void store_expr(NodeId id, LogicalType ty, std::string_view output_name) {
        const DslNode &node = native_node(id);
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputVector);
        sj::Argument out = get_output(output_name, ty);
        LoweredValue lowered = native_dsl_expr(native_child(id, 0));
        if (node.step_data<DslNodeKind::Store>().has_cond) {
            conditional_store_lowered_value(output_name, out, lowered, native_predicate_expr(native_child(id, 1)));
            return;
        }
        if (lowered.has_null() || buffer_is_nullable(output_name)) {
            store_lowered_value(output_name, out, lowered);
            return;
        }
        if (ty.dtype == sj::ScalarDataType::I1) {
            sj::Predicate arg = lowered.as_known_non_null_predicate("store");
            if (get_buffer(output_name).bitpacked) {
                builder.store(arg, out);
                return;
            }
            sj::Value x = builder.bit2bool(arg);
            builder.store(x, out);
            return;
        }

        sj::Value arg = lowered.as_known_non_null_value("store");
        if (!node.step_data<DslNodeKind::Store>().has_cond) {
            builder.store(arg, out, simjit::LoadStoreKind::Unaligned);
        } else {
            sj::Predicate predicate = native_predicate_expr(native_child(id, 1));
            builder.cond_store(arg, predicate, out, simjit::LoadStoreKind::Unaligned);
        }
    }

    void scatter_expr(NodeId id, LogicalType ty, std::string_view output_name) {
        const DslNode &node = native_node(id);
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputTable);
        sj::Argument out = get_output(output_name, ty);
        LoweredValue value = native_dsl_expr(native_child(id, 0));
        sj::Value idx = native_value_expr(native_child(id, 1));
        const BufferDesc &buf = get_buffer(output_name);
        if (buf.bitpacked || buf.null.kind == BufferNullKind::MaskBitpacked) {
            throw_native_value_error("bitpacked scatter is unsupported");
        }

        sj::nullable::NullableValue stored = as_bool_value(value);
        if (stored.null.is_valid() && !buffer_is_nullable(output_name)) {
            throw_native_value_error(
                sj::format("nullable value can't be scattered into non-nullable output %.*s", SV(output_name)));
        }
        sj::MaybePredicate cond;
        if (node.step_data<DslNodeKind::Scatter>().has_child) { cond = native_predicate_expr(native_child(id, 2)); }
        if (buf.null.kind == BufferNullKind::Sentinel) {
            if (value.is_predicate()) {
                throw_native_value_error("sentinel null output is unsupported for predicates");
            }
            sj::Predicate is_null = stored.null.is_valid() ? stored.null.value() : builder.false_();
            stored.v = builder.select(is_null, sentinel_value(buf.null.sentinel, ty), stored.v);
        }
        builder.cond_scatter(stored.v, idx, cond, out);
        if (buf.null.kind == BufferNullKind::MaskBool) {
            sj::Predicate null_mask = encoded_null_mask(stored.null, buf.null.true_means_null);
            builder.cond_scatter(builder.bit2bool(null_mask), idx, cond, get_null_output(output_name));
        }
    }

    void pack_expr(NodeId id, const LogicalType &ty, std::string_view output_name) {
        const DslNode &node = native_node(id);
        std::string_view dst_size = !node.step_data<DslNodeKind::Pack>().dst_size.is_none()
                                        ? native_string(node.step_data<DslNodeKind::Pack>().dst_size)
                                        : arena_concat(output_name, "_size");
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputTable);
        mark_buffer_usage(dst_size, BufferUsageFlags::BufferUsageOutputScalar);
        LoweredValue value = native_dsl_expr(native_child(id, 0));
        const BufferDesc &buf = get_buffer(output_name);
        if (buf.bitpacked || buf.null.kind == BufferNullKind::MaskBitpacked) {
            throw_native_value_error("bitpacked pack is unsupported");
        }
        sj::nullable::NullableValue stored = as_bool_value(value);
        if (stored.null.is_valid() && !buffer_is_nullable(output_name)) {
            throw_native_value_error(
                sj::format("nullable value can't be packed into non-nullable output %.*s", SV(output_name)));
        }
        sj::Predicate cond = native_predicate_expr(native_child(id, 1));
        sj::Argument size_out = get_output(dst_size, {sj::ScalarDataType::I64, false});
        if (buf.null.kind == BufferNullKind::Sentinel) {
            if (value.is_predicate()) {
                throw_native_value_error("sentinel null output is unsupported for predicates");
            }
            sj::Predicate is_null = stored.null.is_valid() ? stored.null.value() : builder.false_();
            stored.v = builder.select(is_null, sentinel_value(buf.null.sentinel, ty), stored.v);
        }
        builder.pack(stored.v, cond, get_output(output_name, ty), size_out);
        if (buf.null.kind == BufferNullKind::MaskBool) {
            sj::Predicate null_mask = encoded_null_mask(stored.null, buf.null.true_means_null);
            builder.pack(builder.bit2bool(null_mask), cond, get_null_output(output_name),
                         get_pack_size_output_alias(dst_size));
        }
    }

    void arith_agg_expr(NodeId id, const LogicalType &ty, std::string_view output_name) {
        const DslNode &node = native_node(id);
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputScalar);
        NodeId arg_id = native_child(id, 0);
        LoweredValue arg = native_dsl_expr(arg_id);
        sj::ArithBinaryOp op = node.step_data<DslNodeKind::ArithAgg>().op;
        const char *op_name = sj::show_arith_binary_op(op);
        LogicalType arg_ty = native_expr_type(arg_id);
        if (arg_ty.is_timestamp()) {
            if (op != sj::ArithBinaryOp::Min && op != sj::ArithBinaryOp::Max) {
                throw_native_value_error(sj::format("timestamp aggregate %s is not allowed", op_name));
            }
        }
        sj::Argument out = get_output(output_name, ty);
        if (node.step_data<DslNodeKind::ArithAgg>().has_cond) {
            nbuilder.cond_arith_agg(arg.to_nullable_value(),
                                    native_dsl_expr(native_child(id, 1)).to_nullable_predicate(), op, out);
        } else {
            nbuilder.arith_agg(arg.to_nullable_value(), op, out);
        }
    }

    void predicate_agg_expr(NodeId id, std::string_view output_name) {
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputScalar);
        LoweredValue arg = native_dsl_expr(native_child(id, 0));
        sj::PredicateBinaryOp op = native_node(id).step_data<DslNodeKind::PredicateAgg>().op;
        builder.predicate_agg(nbuilder.is_true(arg.to_nullable_predicate()), op, get_predicate_output(output_name));
    }

    void count_if_expr(NodeId id, std::string_view output_name) {
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputScalar);
        LoweredValue cond = native_dsl_expr(native_child(id, 0));
        sj::Argument out = get_output(output_name, {sj::ScalarDataType::I64, false});
        if (cond.is_nullable()) {
            nbuilder.countif(cond.as_nullable_predicate(), out);
        } else {
            builder.countif(cond.as_predicate(), out);
        }
    }

    void grouped_arith_agg_expr(NodeId id, const LogicalType &ty) {
        const DslNode &node = native_node(id);
        LoweredValue arg = native_dsl_expr(native_child(id, 0));
        sj::Value idx = native_value_expr(native_child(id, 1));
        sj::ArithBinaryOp op = node.step_data<DslNodeKind::GroupedArithAgg>().op;
        std::string_view table_name =
            native_required_string(node.step_data<DslNodeKind::GroupedArithAgg>().table, "GroupedArithAggExpr.table");
        mark_buffer_usage(table_name, BufferUsageFlags::BufferUsageOutputTable);
        sj::Argument table = get_output(table_name, ty);
        if (node.step_data<DslNodeKind::GroupedArithAgg>().has_cond) {
            nbuilder.grouped_cond_arith_agg(
                arg.to_nullable_value(), native_dsl_expr(native_child(id, 2)).to_nullable_predicate(), idx, op, table);
        } else {
            nbuilder.grouped_arith_agg(arg.to_nullable_value(), idx, op, table);
        }
    }

    template <sj::Value (sj::FunctionBuilder::*Fn)(sj::Value)>
    LoweredValue lower_integer_builder_function(std::string_view name, NodeId arg_id) {
        LoweredValue arg_val = native_dsl_expr(arg_id);
        return (builder.*Fn)(arg_val.as_known_non_null_value(name));
    }

    template <sj::Value (NativeBuilderImpl::*Fn)(sj::Value, const LogicalType &)>
    LoweredValue lower_timestamp_builder_function(std::string_view name, NodeId arg_id) {
        LoweredValue arg_val = native_dsl_expr(arg_id);
        return (this->*Fn)(arg_val.as_known_non_null_value(name), native_expr_type(arg_id));
    }

    LoweredValue timestamp_function_expr(DslFunctionKind function, std::string_view name, NodeId arg_id) {
        switch (function) {
        case DslFunctionKind::Year:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_year>(name, arg_id);
        case DslFunctionKind::Month:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_month>(name, arg_id);
        case DslFunctionKind::Day:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_day>(name, arg_id);
        case DslFunctionKind::Hour:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_hour>(name, arg_id);
        case DslFunctionKind::Minute:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_minute>(name, arg_id);
        case DslFunctionKind::Second:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_second>(name, arg_id);
        case DslFunctionKind::DayOfWeek:
            return lower_timestamp_builder_function<&NativeBuilderImpl::lower_timestamp_day_of_week>(name, arg_id);
        case DslFunctionKind::Unknown:
        case DslFunctionKind::Log2:
        case DslFunctionKind::Log2NoZero:
        case DslFunctionKind::Byteswap:
        case DslFunctionKind::BitFloor:
        case DslFunctionKind::BitCeil:
        case DslFunctionKind::Coalesce:
        case DslFunctionKind::NullIf:
        case DslFunctionKind::IsNull:
        case DslFunctionKind::IsNotNull: break;
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue integer_function_expr(DslFunctionKind function, std::string_view name, NodeId arg_id) {
        switch (function) {
        case DslFunctionKind::Log2: return lower_integer_builder_function<&sj::FunctionBuilder::log2>(name, arg_id);
        case DslFunctionKind::Log2NoZero:
            return lower_integer_builder_function<&sj::FunctionBuilder::log2_no_zero>(name, arg_id);
        case DslFunctionKind::Byteswap:
            return lower_integer_builder_function<&sj::FunctionBuilder::byteswap>(name, arg_id);
        case DslFunctionKind::BitFloor:
            return lower_integer_builder_function<&sj::FunctionBuilder::bit_floor>(name, arg_id);
        case DslFunctionKind::BitCeil:
            return lower_integer_builder_function<&sj::FunctionBuilder::bit_ceil>(name, arg_id);
        case DslFunctionKind::Unknown:
        case DslFunctionKind::Year:
        case DslFunctionKind::Month:
        case DslFunctionKind::Day:
        case DslFunctionKind::Hour:
        case DslFunctionKind::Minute:
        case DslFunctionKind::Second:
        case DslFunctionKind::DayOfWeek:
        case DslFunctionKind::Coalesce:
        case DslFunctionKind::NullIf:
        case DslFunctionKind::IsNull:
        case DslFunctionKind::IsNotNull: break;
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue coalesce_function_expr(NodeId id) {
        const DslNode &node = native_node(id);
        std::vector<sj::nullable::NullableValue> lowered;
        lowered.reserve(node.child_count);
        for (uint32_t i = 0; i < node.child_count; ++i) {
            lowered.push_back(native_dsl_expr(native_child(id, i)).to_nullable_value());
        }
        return nbuilder.coalesce(lowered);
    }

    LoweredValue null_predicate_function_expr(DslFunctionKind function, NodeId arg_id) {
        switch (function) {
        case DslFunctionKind::IsNull: return nbuilder.is_null(native_dsl_expr(arg_id).to_nullable_value());
        case DslFunctionKind::IsNotNull: return nbuilder.is_not_null(native_dsl_expr(arg_id).to_nullable_value());
        case DslFunctionKind::Unknown:
        case DslFunctionKind::Year:
        case DslFunctionKind::Month:
        case DslFunctionKind::Day:
        case DslFunctionKind::Hour:
        case DslFunctionKind::Minute:
        case DslFunctionKind::Second:
        case DslFunctionKind::DayOfWeek:
        case DslFunctionKind::Log2:
        case DslFunctionKind::Log2NoZero:
        case DslFunctionKind::Byteswap:
        case DslFunctionKind::BitFloor:
        case DslFunctionKind::BitCeil:
        case DslFunctionKind::Coalesce:
        case DslFunctionKind::NullIf: break;
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue function_expr(NodeId id) {
        const DslNode &node = native_node(id);
        const DslFunctionSpec *spec = find_dsl_function_spec(node.step_data<DslNodeKind::Function>().kind);
        SIMJIT_ASSERT(spec != nullptr);
        SIMJIT_ASSERT(spec->accepts_arg_count(node.child_count));
        switch (spec->group) {
        case DslFunctionGroup::TimestampExtract:
            return timestamp_function_expr(spec->kind, spec->name, native_child(id, 0));
        case DslFunctionGroup::IntegerUnary: return integer_function_expr(spec->kind, spec->name, native_child(id, 0));
        case DslFunctionGroup::Coalesce: return coalesce_function_expr(id);
        case DslFunctionGroup::NullIf:
            return nbuilder.nullif(native_dsl_expr(native_child(id, 0)).to_nullable_value(),
                                   native_dsl_expr(native_child(id, 1)).to_nullable_value());
        case DslFunctionGroup::NullPredicate: return null_predicate_function_expr(spec->kind, native_child(id, 0));
        }
        SIMJIT_UNREACHABLE();
    }

    LoweredValue native_dsl_expr(NodeId id, std::string_view output_name = {}) {
        const DslNode &node = native_node(id);
        LogicalType ty = native_expr_type(id);
        switch (node.kind) {
        case DslNodeKind::Const: return const_expr(id, ty);
        case DslNodeKind::Load: return load_expr(id, ty);
        case DslNodeKind::LoadSplat: return load_splat_expr(id);
        case DslNodeKind::Gather: return gather_expr(id, ty);
        case DslNodeKind::Index: return index_expr(id, ty);
        case DslNodeKind::ArithBinary: return arith_binary_expr(id);
        case DslNodeKind::PredicateBinary: return predicate_binary_expr(id);
        case DslNodeKind::ArithUnary: return arith_unary_expr(id);
        case DslNodeKind::PredicateNot: return predicate_not_expr(id);
        case DslNodeKind::Compare: return compare_expr(id);
        case DslNodeKind::IntCast: return int_cast_expr(id, ty);
        case DslNodeKind::FloatCast: return float_cast_expr(id, ty);
        case DslNodeKind::BitCast: return bit_cast_expr(id, ty);
        case DslNodeKind::Function: return function_expr(id);
        case DslNodeKind::Select: return select_expr(id);
        case DslNodeKind::FpClass: return fpclass_expr(id);
        case DslNodeKind::Permute: return permute_expr(id);
        case DslNodeKind::Store: store_expr(id, ty, output_name); return {};
        case DslNodeKind::Scatter: scatter_expr(id, ty, output_name); return {};
        case DslNodeKind::Pack: pack_expr(id, ty, output_name); return {};
        case DslNodeKind::ArithAgg: arith_agg_expr(id, ty, output_name); return {};
        case DslNodeKind::PredicateAgg: predicate_agg_expr(id, output_name); return {};
        case DslNodeKind::CountIf: count_if_expr(id, output_name); return {};
        case DslNodeKind::GroupedArithAgg: grouped_arith_agg_expr(id, ty); return {};
        }
        SIMJIT_UNREACHABLE();
    }

    void lower_resolved_outputs(const DslProgram &program) {
        if (native_program_requires_safety_check(program)) {
            const BufferDesc &buf = get_buffer(NATIVE_SAFETY_CHECK_BUFFER);
            if (buf.ty.dtype != sj::ScalarDataType::I8 || buf.length != 1 || buf.bitpacked ||
                buf.null.kind != BufferNullKind::None) {
                throw_native_value_error("invalid internal safety-check buffer");
            }
            safety_check_arg = builder.arg_safety_check();
        }
        native_program = &program;
        for (uint32_t i = 0; i < program.output_count; ++i) {
            const DslOutput &out = program.outputs[i];
            lower_native_named_expr(out.name, out.root);
        }
        native_program = nullptr;
    }

    void lower_native_named_expr(std::string_view output_name, NodeId root) {
        LoweredValue val = native_dsl_expr(root, output_name);
        if (!dsl_node_is_final(native_node(root).kind)) { materialize_output(output_name, root, val); }
    }

    void materialize_output(std::string_view output_name, NodeId root, LoweredValue val) {
        SIMJIT_ASSERT(val.is_valid());
        LogicalType out_ty = native_expr_type(root);
        mark_buffer_usage(output_name, BufferUsageFlags::BufferUsageOutputVector);
        sj::Argument out = get_output(output_name, out_ty);
        store_lowered_value(output_name, out, val);
    }

    void init_schema(NameMap<BufferDesc> b) {
        this->buffers = std::move(b);
        size_t capacity = this->buffers.size() * 2 + 8;
        schema.reserve(capacity);
        load_args.reserve(capacity);
        load_null_args.reserve(capacity);
        outputs.reserve(capacity);
        output_nulls.reserve(capacity);
        buffer_usage.reserve(capacity);
        this->buffers.for_each_entry(
            [&](const NameMapEntry<BufferDesc> &entry) { this->schema.insert_or_assign(entry.name, entry.value.ty); });
    }

    void validate_lengths(size_t n) const {
        buffer_usage.for_each_entry([&](const NameMapEntry<BufferUsageFlags> &entry) {
            std::string_view name = entry.name;
            BufferUsageFlags usage = entry.value;
            const BufferDesc *buf = buffers.find(name);
            if (buf == nullptr) { throw_native_value_error(sj::format("buffer %.*s is missing", SV(name))); }

            size_t length = buf->length;
            bool needs_vector = bool(usage & BufferUsageFlags::BufferUsageInputVector) ||
                                bool(usage & BufferUsageFlags::BufferUsageOutputVector);
            bool needs_scalar =
                !bool(usage & BufferUsageFlags::BufferUsageInputVector) &&
                !bool(usage & BufferUsageFlags::BufferUsageOutputVector) &&
                !bool(usage & BufferUsageFlags::BufferUsageInputTable) &&
                !bool(usage & BufferUsageFlags::BufferUsageOutputTable) &&
                bool(usage & (BufferUsageFlags::BufferUsageInputSplat | BufferUsageFlags::BufferUsageOutputScalar));

            if (needs_vector && length != n) {
                throw_native_value_error(sj::format("buffer %.*s must have length %zu, got %zu", SV(name), n, length));
            }
            if (needs_scalar && length != 1) {
                throw_native_value_error(sj::format("buffer %.*s must have length 1, got %zu", SV(name), length));
            }
            if (bool(usage & BufferUsageFlags::BufferUsageOutputScalar) &&
                !bool(usage & BufferUsageFlags::BufferUsageOutputVector) && length != 1) {
                throw_native_value_error(
                    sj::format("output buffer %.*s must have length 1, got %zu", SV(name), length));
            }
        });
    }

    void store_lowered_value(std::string_view output_name, sj::Argument out, LoweredValue val) {
        const BufferDesc &buf = get_buffer(output_name);
        if (buf.null.kind == BufferNullKind::None) {
            if (val.has_null()) {
                throw_native_value_error(
                    sj::format("nullable output %.*s requires null output metadata", SV(output_name)));
            }
            if (val.is_predicate()) {
                sj::Predicate pr = val.as_known_non_null_predicate("store");
                if (buf.bitpacked) {
                    builder.store(pr, out);
                } else {
                    builder.store(builder.bit2bool(pr), out);
                }
                return;
            }
            builder.store(val.as_known_non_null_value("store"), out, sj::LoadStoreKind::Unaligned);
            return;
        }
        if (buf.null.kind == BufferNullKind::Sentinel) {
            if (val.has_null()) {
                if (val.is_predicate()) {
                    throw_native_value_error("sentinel null output is unsupported for predicates");
                }
                nbuilder.nval_store(val.as_nullable_value(), out, sentinel_value(buf.null.sentinel, buf.ty));
                return;
            }
            if (val.is_predicate()) {
                sj::Predicate pr = val.as_known_non_null_predicate("store");
                if (buf.bitpacked) {
                    builder.store(pr, out);
                } else {
                    builder.store(builder.bit2bool(pr), out);
                }
                return;
            }
            builder.store(val.as_known_non_null_value("store"), out, sj::LoadStoreKind::Unaligned);
            return;
        }
        sj::Argument out_null = get_null_output(output_name);
        if (val.is_predicate()) {
            if (buf.bitpacked) {
                if (buf.null.kind == BufferNullKind::MaskBitpacked) {
                    nbuilder.nbit_store_ext(val.to_nullable_predicate(), out, out_null, buf.null.true_means_null);
                } else {
                    nbuilder.nbool_store_ext(val.to_nullable_predicate(), out, out_null, buf.null.true_means_null);
                }
            } else {
                sj::nullable::NullableValue bool_value = nbuilder.bit2bool(val.to_nullable_predicate());
                if (buf.null.kind == BufferNullKind::MaskBitpacked) {
                    nbuilder.nbit_store_ext(bool_value, out, out_null, buf.null.true_means_null);
                } else {
                    nbuilder.nbool_store_ext(bool_value, out, out_null, buf.null.true_means_null);
                }
            }
            return;
        }
        if (buf.null.kind == BufferNullKind::MaskBitpacked) {
            nbuilder.nbit_store_ext(val.to_nullable_value(), out, out_null, buf.null.true_means_null);
        } else {
            nbuilder.nbool_store_ext(val.to_nullable_value(), out, out_null, buf.null.true_means_null);
        }
    }

    std::vector<NativePointerBinding> build_pointer_plan() {
        std::vector<NativePointerBinding> plan;
        plan.reserve(load_args.size() + load_null_args.size() + outputs.size() + output_nulls.size() +
                     output_aliases.size() + (safety_check_arg.has_value() ? 1 : 0));
        append_input_pointers(plan);
        append_output_pointers(plan);
        plan.insert(plan.end(), output_aliases.begin(), output_aliases.end());
        if (safety_check_arg) {
            append_pointer_binding(plan, NATIVE_SAFETY_CHECK_BUFFER, safety_check_arg->idx_, true);
        }
        return plan;
    }

    void append_input_pointers(std::vector<NativePointerBinding> &plan) {
        load_args.for_each_entry([&](const NameMapEntry<sj::Argument> &entry) {
            append_pointer_binding(plan, entry.name, entry.value.idx_, false);
        });
        load_null_args.for_each_entry([&](const NameMapEntry<sj::Argument> &entry) {
            append_null_pointer_binding(plan, entry.name, entry.value.idx_, false);
        });
    }

    void append_output_pointers(std::vector<NativePointerBinding> &plan) {
        outputs.for_each_entry([&](const NameMapEntry<sj::Argument> &entry) {
            append_pointer_binding(plan, entry.name, entry.value.idx_, true);
        });
        output_nulls.for_each_entry([&](const NameMapEntry<sj::Argument> &entry) {
            append_null_pointer_binding(plan, entry.name, entry.value.idx_, true);
        });
    }

    void append_pointer_binding(std::vector<NativePointerBinding> &plan, std::string_view name, size_t idx,
                                bool writable) {
        plan.push_back(NativePointerBinding{idx, name, false, writable});
    }

    void append_null_pointer_binding(std::vector<NativePointerBinding> &plan, std::string_view name, size_t idx,
                                     bool writable) {
        plan.push_back(NativePointerBinding{idx, name, true, writable});
    }
};

} // namespace

std::vector<NativePointerBinding> build_native_pointer_plan(sj::FunctionBuilder &builder, NameMap<BufferDesc> buffers,
                                                            const DslProgram &program, size_t n) {
    NativeBuilderImpl native_builder{builder};
    native_builder.init_schema(std::move(buffers));
    native_builder.lower_resolved_outputs(program);
    native_builder.validate_lengths(n);
    return native_builder.build_pointer_plan();
}

} // namespace simjit_python
