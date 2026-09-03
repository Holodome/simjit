// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "simjit/compiler.h"
#include "simjit/simjit.h"

namespace simjit_python {
namespace sj = simjit;

enum class DslIntCastKind : uint8_t {
    Cast,
    Signed,
    Unsigned,
    Trunc,
    Sext,
    Zext,
};

enum class DslFunctionKind : uint8_t {
    Unknown,
    Year,
    Month,
    Day,
    Hour,
    Minute,
    Second,
    DayOfWeek,
    Log2,
    Log2NoZero,
    Byteswap,
    BitFloor,
    BitCeil,
    Coalesce,
    NullIf,
    IsNull,
    IsNotNull,
};

enum class SemanticKind : uint8_t {
    Plain,
    Timestamp64,
};

enum class TimestampUnit : uint8_t {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
};

enum class TimestampTimezone : uint8_t {
    None,
    UTC,
};

struct LogicalType {
    sj::ScalarDataType dtype = sj::ScalarDataType::I1;
    bool is_unsigned = false;
    SemanticKind semantic = SemanticKind::Plain;
    TimestampUnit unit = TimestampUnit::Nanoseconds;
    TimestampTimezone timezone = TimestampTimezone::None;

    LogicalType() = default;
    LogicalType(const LogicalType &) = default;
    LogicalType(LogicalType &&) = default;
    LogicalType &operator=(const LogicalType &) = default;
    LogicalType &operator=(LogicalType &&) = default;

    static LogicalType simple(sj::ScalarDataType dtype, bool is_unsigned = false) {
        return LogicalType{dtype, is_unsigned, SemanticKind::Plain};
    }

    bool is_int() const {
        return dtype == sj::ScalarDataType::I8 || dtype == sj::ScalarDataType::I16 ||
               dtype == sj::ScalarDataType::I32 || dtype == sj::ScalarDataType::I64;
    }
    bool is_float() const { return dtype == sj::ScalarDataType::F32 || dtype == sj::ScalarDataType::F64; }
    bool is_timestamp() const { return semantic == SemanticKind::Timestamp64; }

    const char *name() const;

    bool operator==(const LogicalType &rhs) const {
        auto &lhs = *this;
        if (lhs.dtype != rhs.dtype || lhs.is_unsigned != rhs.is_unsigned || lhs.semantic != rhs.semantic) {
            return false;
        }
        if (lhs.semantic == SemanticKind::Timestamp64) { return lhs.unit == rhs.unit && lhs.timezone == rhs.timezone; }
        return true;
    }
    bool operator!=(const LogicalType &rhs) const { return !(*this == rhs); }
};

enum class BufferNullKind : uint8_t {
    None,
    MaskBitpacked,
    MaskBool,
    Sentinel,
};

enum class ConstPayloadKind : uint8_t {
    None,
    Bool,
    SignedInt,
    UnsignedInt,
    Float,
};

struct ConstPayload {
    ConstPayloadKind kind = ConstPayloadKind::None;
    union {
        bool bool_value;
        int64_t signed_int;
        uint64_t unsigned_int;
        double float_value;
    };
};

struct BufferNullDesc {
    BufferNullKind kind = BufferNullKind::None;
    bool true_means_null = true;
    ConstPayload sentinel{};
};

struct BufferDesc {
    LogicalType ty{};
    size_t length = 0;
    BufferNullDesc null{};
    bool aligned = false;
    bool bitpacked = false;
};

enum class BufferUsageFlags : uint8_t {
    BufferUsageNone = 0,
    BufferUsageInputVector = 1u << 0,
    BufferUsageInputSplat = 1u << 1,
    BufferUsageInputTable = 1u << 2,
    BufferUsageOutputVector = 1u << 3,
    BufferUsageOutputScalar = 1u << 4,
    BufferUsageOutputTable = 1u << 5,
};

SIMJIT_DEFINE_ENUM_FLAGS(BufferUsageFlags)

enum class NativeRuntimeOutputKind : uint8_t {
    Numpy,
    Arrow,
    Explicit,
};

template <typename T> struct NameMapEntry {
    size_t hash = 0;
    std::string_view name{};
    T value{};
};

template <typename T> class NameMap {
    using Storage = std::unordered_multimap<size_t, NameMapEntry<T>>;

public:
    void reserve(size_t capacity) { entries_.reserve(capacity); }
    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }

    template <typename F> void for_each_entry(F fn) const {
        for (const auto &item : entries_) {
            fn(item.second);
        }
    }

    T *find(std::string_view name) {
        size_t h = hash_name(name);
        auto [first, last] = entries_.equal_range(h);
        for (auto it = first; it != last; ++it) {
            NameMapEntry<T> &entry = it->second;
            if (entry.name == name) { return &entry.value; }
        }
        return nullptr;
    }

    const T *find(std::string_view name) const {
        size_t h = hash_name(name);
        auto [first, last] = entries_.equal_range(h);
        for (auto it = first; it != last; ++it) {
            const NameMapEntry<T> &entry = it->second;
            if (entry.name == name) { return &entry.value; }
        }
        return nullptr;
    }

    T &insert_or_assign(std::string_view name, T value) {
        size_t h = hash_name(name);
        auto [first, last] = entries_.equal_range(h);
        for (auto it = first; it != last; ++it) {
            NameMapEntry<T> &entry = it->second;
            if (entry.name == name) {
                entry.value = std::move(value);
                return entry.value;
            }
        }
        auto it = entries_.emplace(h, NameMapEntry<T>{h, name, std::move(value)});
        return it->second.value;
    }

    T &get_or_insert(std::string_view name, T initial = T{}) {
        size_t h = hash_name(name);
        auto [first, last] = entries_.equal_range(h);
        for (auto it = first; it != last; ++it) {
            if (it->second.name == name) { return it->second.value; }
        }
        auto it = entries_.emplace(h, NameMapEntry<T>{h, name, std::move(initial)});
        return it->second.value;
    }

private:
    static size_t hash_name(std::string_view name) { return std::hash<std::string_view>{}(name); }

    Storage entries_{};
};

using NodeId = uint32_t;

struct DslStringRef {
    static constexpr uint32_t kNone = std::numeric_limits<uint32_t>::max();

    uint32_t id = kNone;

    static DslStringRef native(uint32_t id) { return DslStringRef{id}; }
    bool is_none() const { return id == kNone; }
    bool is_native() const { return !is_none(); }
    uint32_t native_id() const { return id; }
};

enum class DslNodeKind : uint8_t {
    Const,
    Load,
    LoadSplat,
    Gather,
    Index,
    ArithBinary,
    PredicateBinary,
    ArithUnary,
    PredicateNot,
    Compare,
    IntCast,
    FloatCast,
    BitCast,
    Function,
    Select,
    FpClass,
    Permute,
    Store,
    Scatter,
    Pack,
    ArithAgg,
    PredicateAgg,
    CountIf,
    GroupedArithAgg,
};

struct DslNode {
    struct LoadData {
        DslStringRef name{};
        sj::LoadStoreKind kind = sj::LoadStoreKind::Unaligned;
    };
    struct NamedData {
        DslStringRef name{};
    };
    struct ArithBinaryData {
        sj::ArithBinaryOp op = sj::ArithBinaryOp::Add;
        bool checked = false;
    };
    struct PredicateBinaryData {
        sj::PredicateBinaryOp op = sj::PredicateBinaryOp::And;
    };
    struct ArithUnaryData {
        sj::ArithUnaryOp op = sj::ArithUnaryOp::Negate;
        bool checked = false;
    };
    struct CompareData {
        sj::CmpOp op = sj::CmpOp::Equal;
        bool is_unsigned = false;
    };
    struct IntCastData {
        DslIntCastKind kind = DslIntCastKind::Cast;
        bool checked = false;
    };
    struct FloatCastData {
        bool is_unsigned = false;
    };
    struct FunctionData {
        DslFunctionKind kind = DslFunctionKind::Unknown;
    };
    struct FpClassData {
        sj::FpClass flags{};
    };
    struct PermuteData {
        uint64_t idxs = 0;
        bool is_bit = false;
    };
    struct StoreData {
        sj::LoadStoreKind kind = sj::LoadStoreKind::Unaligned;
        bool has_cond = false;
    };
    struct OptionalChildData {
        bool has_child = false;
    };
    struct PackData {
        DslStringRef dst_size{};
    };
    struct ArithAggData {
        sj::ArithBinaryOp op = sj::ArithBinaryOp::Add;
        bool has_cond = false;
    };
    struct PredicateAggData {
        sj::PredicateBinaryOp op = sj::PredicateBinaryOp::And;
    };
    struct GroupedArithAggData {
        sj::ArithBinaryOp op = sj::ArithBinaryOp::Add;
        DslStringRef table{};
        bool has_cond = false;
    };
    template <DslNodeKind Kind> struct Data;
    template <DslNodeKind Kind> const typename Data<Kind>::T &step_data() const noexcept;
    template <DslNodeKind Kind> typename Data<Kind>::T &step_data() noexcept;

    uint32_t first_child = 0;
    uint16_t child_count = 0;
    DslNodeKind kind = DslNodeKind::Const;

private:
    union {
        ConstPayload constant;
        LoadData load;
        NamedData named;
        ArithBinaryData arith_binary;
        PredicateBinaryData predicate_binary;
        ArithUnaryData arith_unary;
        CompareData compare;
        IntCastData int_cast;
        FloatCastData float_cast;
        FunctionData function;
        FpClassData fpclass;
        PermuteData permute;
        StoreData store;
        OptionalChildData optional_child;
        PackData pack;
        ArithAggData arith_agg;
        PredicateAggData predicate_agg;
        GroupedArithAggData grouped_arith_agg;
    };

public:
    DslNode() noexcept : constant{} {}
};

#define DSL_STEP_DATA_LIST(X)            \
    X(Const, constant)                   \
    X(Load, load)                        \
    X(LoadSplat, named)                  \
    X(Gather, named)                     \
    X(ArithBinary, arith_binary)         \
    X(PredicateBinary, predicate_binary) \
    X(ArithUnary, arith_unary)           \
    X(Compare, compare)                  \
    X(IntCast, int_cast)                 \
    X(FloatCast, float_cast)             \
    X(Function, function)                \
    X(FpClass, fpclass)                  \
    X(Permute, permute)                  \
    X(Store, store)                      \
    X(Scatter, optional_child)           \
    X(Pack, pack)                        \
    X(ArithAgg, arith_agg)               \
    X(PredicateAgg, predicate_agg)       \
    X(GroupedArithAgg, grouped_arith_agg)

#define ASSOC_STEP_DATA(_kind, _field)                     \
    template <> struct DslNode::Data<DslNodeKind::_kind> { \
        using T = decltype(DslNode::_field);               \
    };

DSL_STEP_DATA_LIST(ASSOC_STEP_DATA)

#undef ASSOC_STEP_DATA

#define STEP_DATA(_kind, _field)                                                                                 \
    template <>                                                                                                  \
    inline const DslNode::Data<DslNodeKind::_kind>::T &DslNode::step_data<DslNodeKind::_kind>() const noexcept { \
        SIMJIT_ASSERT(kind == DslNodeKind::_kind);                                                               \
        return _field;                                                                                           \
    }                                                                                                            \
    template <> inline DslNode::Data<DslNodeKind::_kind>::T &DslNode::step_data<DslNodeKind::_kind>() noexcept { \
        SIMJIT_ASSERT(kind == DslNodeKind::_kind);                                                               \
        return _field;                                                                                           \
    }

DSL_STEP_DATA_LIST(STEP_DATA)

#undef STEP_DATA
#undef DSL_STEP_DATA_LIST

struct DslTypeSlot {
    LogicalType type{};
    bool has = false;
};

struct DslOutput {
    std::string_view name{};
    NodeId root = 0;
};

struct DslProgram {
    sj::MemoryArena arena{};
    sj::ArenaArray<DslNode> nodes{};
    sj::ArenaArray<DslTypeSlot> declared_types{};
    sj::ArenaArray<DslTypeSlot> resolved_types{};
    sj::ArenaArray<NodeId> child_edges{};
    sj::ArenaArray<DslOutput> outputs{};
    sj::ArenaArray<std::string_view> strings{};
    uint32_t node_count = 0;
    uint32_t edge_count = 0;
    uint32_t output_count = 0;
    uint32_t string_count = 0;

    std::string_view string(DslStringRef ref) const;
    std::optional<std::string_view> optional_string(DslStringRef ref) const;
    std::string_view required_string(DslStringRef ref, std::string_view field_name) const;
};

struct DslImportLimits {
    size_t node_capacity = 4096;
    size_t edge_capacity = 4096;
    size_t string_capacity = 4096;
};

enum class DslFunctionGroup : uint8_t {
    TimestampExtract,
    IntegerUnary,
    Coalesce,
    NullIf,
    NullPredicate,
};

static constexpr uint16_t kDslVariadicFunctionArgs = std::numeric_limits<uint16_t>::max();

struct DslFunctionSpec {
    DslFunctionKind kind;
    std::string_view name;
    DslFunctionGroup group;
    uint16_t min_args;
    uint16_t max_args;

    bool accepts_arg_count(uint32_t count) const;
};

struct DslTypeEntry {
    size_t hash = 0;
    std::string_view name{};
    LogicalType type{};
};

struct DslTypeMap {
    sj::ArenaArray<DslTypeEntry> entries{};
    uint32_t count = 0;

    static size_t hash_name(std::string_view name);

    DslTypeEntry *begin();
    DslTypeEntry *end();
    const DslTypeEntry *begin() const;
    const DslTypeEntry *end() const;

    DslTypeEntry *find(std::string_view name);
    const DslTypeEntry *find(std::string_view name) const;
    void insert_or_assign(std::string_view name, LogicalType type);
};

struct NativeRuntimeOutputSpec {
    std::string_view name{};
    LogicalType ty{};
    size_t length = 0;
    bool scalar = false;
    bool bitpacked = false;
    bool nullable = false;
    size_t data_bytes = 0;
    size_t validity_bytes = 0;
};

const DslFunctionSpec *find_dsl_function_spec(DslFunctionKind kind);

void resolve_native_dsl_program(DslProgram &program, DslTypeMap input_types, DslTypeMap output_types);
NodeId native_child(const DslProgram &program, NodeId id, uint32_t index);
std::vector<NativeRuntimeOutputSpec> plan_native_runtime_outputs(const DslProgram &program,
                                                                 const NameMap<BufferDesc> &inputs,
                                                                 NativeRuntimeOutputKind output_kind, size_t n);

} // namespace simjit_python
