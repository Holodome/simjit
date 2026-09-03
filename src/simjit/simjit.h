// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/errors.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <type_traits>

// This macro is used in places that should be unreachable. It is not the same thing as assert/exception - this should
// only be used in cases where this state is *really* unreachable, assuming memory/data structures are not corrupted.
// For example if variable of enum type has invalid value.
#define SIMJIT_UNREACHABLE() __builtin_unreachable()

#ifndef SIMJIT_ASSERT
#ifdef NDEBUG
#define SIMJIT_ASSERT(_x)                \
    do {                                 \
        if (!(_x)) SIMJIT_UNREACHABLE(); \
    } while (0)
#else
#define SIMJIT_ASSERT(_x) assert(_x)
#endif
#endif

#define SIMJIT_DEFINE_ENUM_FLAGS(T)                                            \
    [[maybe_unused]] constexpr T operator~(T a) noexcept {                     \
        return T(~std::underlying_type_t<T>(a));                               \
    }                                                                          \
                                                                               \
    [[maybe_unused]] constexpr T operator|(T a, T b) noexcept {                \
        return T(std::underlying_type_t<T>(a) | std::underlying_type_t<T>(b)); \
    }                                                                          \
    [[maybe_unused]] constexpr T operator&(T a, T b) noexcept {                \
        return T(std::underlying_type_t<T>(a) & std::underlying_type_t<T>(b)); \
    }                                                                          \
    [[maybe_unused]] constexpr T operator^(T a, T b) noexcept {                \
        return T(std::underlying_type_t<T>(a) ^ std::underlying_type_t<T>(b)); \
    }                                                                          \
                                                                               \
    [[maybe_unused]] constexpr T &operator|=(T &a, T b) noexcept {             \
        a = T(std::underlying_type_t<T>(a) | std::underlying_type_t<T>(b));    \
        return a;                                                              \
    }                                                                          \
    [[maybe_unused]] constexpr T &operator&=(T &a, T b) noexcept {             \
        a = T(std::underlying_type_t<T>(a) & std::underlying_type_t<T>(b));    \
        return a;                                                              \
    }                                                                          \
    [[maybe_unused]] constexpr T &operator^=(T &a, T b) noexcept {             \
        a = T(std::underlying_type_t<T>(a) ^ std::underlying_type_t<T>(b));    \
        return a;                                                              \
    }

namespace simjit {

enum class ScalarDataType : uint8_t {
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
    F32 = 4,
    F64 = 5,
    I1,
    I128
};

namespace types {
constexpr ScalarDataType I1 = ScalarDataType::I1;
constexpr ScalarDataType I128 = ScalarDataType::I128;
constexpr ScalarDataType I8 = ScalarDataType::I8;
constexpr ScalarDataType I16 = ScalarDataType::I16;
constexpr ScalarDataType I32 = ScalarDataType::I32;
constexpr ScalarDataType I64 = ScalarDataType::I64;
constexpr ScalarDataType F32 = ScalarDataType::F32;
constexpr ScalarDataType F64 = ScalarDataType::F64;
} // namespace types

struct ConstData {
    ScalarDataType dtype = ScalarDataType::I64;
    union {
        uint64_t bits;
        int64_t i64_value;
        double f64_value;
        float f32_value;
    };

    ConstData() noexcept : bits(0) {}

    static ConstData i1(bool x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I1;
        result.bits = x ? 1 : 0;
        return result;
    }
    static ConstData i8(int8_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I8;
        result.bits = (uint8_t)x;
        return result;
    }
    static ConstData u8(uint8_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I8;
        result.bits = x;
        return result;
    }
    static ConstData i16(int16_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I16;
        result.bits = (uint16_t)x;
        return result;
    }
    static ConstData u16(uint16_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I16;
        result.bits = x;
        return result;
    }
    static ConstData i32(int32_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I32;
        result.bits = (uint32_t)x;
        return result;
    }
    static ConstData u32(uint32_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I32;
        result.bits = x;
        return result;
    }
    static ConstData i64(int64_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I64;
        result.i64_value = x;
        return result;
    }
    static ConstData u64(uint64_t x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::I64;
        result.bits = x;
        return result;
    }
    static ConstData f64(double x) noexcept {
        ConstData result;
        result.dtype = ScalarDataType::F64;
        result.f64_value = x;
        return result;
    }
    static ConstData f32(float x) noexcept {
        ConstData result{};
        result.dtype = ScalarDataType::F32;
        result.bits = 0;
        result.f32_value = x;
        return result;
    }
    static ConstData f32_bits(uint32_t x) noexcept {
        ConstData result{};
        result.dtype = ScalarDataType::F32;
        result.bits = x;
        return result;
    }
    static ConstData f64_bits(uint64_t x) noexcept {
        ConstData result{};
        result.dtype = ScalarDataType::F64;
        result.bits = x;
        return result;
    }

    constexpr uint64_t raw_bits() const noexcept { return bits; }
    constexpr static int64_t sign_extend_bits(uint64_t x, unsigned width) noexcept {
        SIMJIT_ASSERT(width > 0 && width < 64);
        uint64_t mask = (UINT64_C(1) << width) - 1;
        x &= mask;
        uint64_t sign_bit = UINT64_C(1) << (width - 1);
        if ((x & sign_bit) == 0) { return (int64_t)x; }
        return -(int64_t)(((~x) & mask) + 1);
    }

    constexpr uint64_t as_unsigned() const noexcept {
        switch (dtype) {
        case ScalarDataType::I1: return bits & 1;
        case ScalarDataType::I8: return bits & UINT8_MAX;
        case ScalarDataType::I16: return bits & UINT16_MAX;
        case ScalarDataType::I32: return bits & UINT32_MAX;
        case ScalarDataType::I64: return bits;
        case ScalarDataType::F32: return bits & UINT32_MAX;
        case ScalarDataType::F64: return bits;
        case ScalarDataType::I128: SIMJIT_ASSERT(0);
        }
        SIMJIT_UNREACHABLE();
    }
    constexpr int64_t as_signed() const noexcept {
        switch (dtype) {
        case ScalarDataType::I1: return (int64_t)(bits & 1);
        case ScalarDataType::I8: return sign_extend_bits(bits, 8);
        case ScalarDataType::I16: return sign_extend_bits(bits, 16);
        case ScalarDataType::I32: return sign_extend_bits(bits, 32);
        case ScalarDataType::I64: return i64_value;
        case ScalarDataType::F32:
        case ScalarDataType::F64:
        case ScalarDataType::I128: SIMJIT_ASSERT(0);
        }
        SIMJIT_UNREACHABLE();
    }

    constexpr int8_t as_i8() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I8);
        return (int8_t)sign_extend_bits(bits, 8);
    }
    constexpr uint8_t as_u8() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I8);
        return (uint8_t)(bits & UINT8_MAX);
    }
    constexpr int16_t as_i16() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I16);
        return (int16_t)sign_extend_bits(bits, 16);
    }
    constexpr uint16_t as_u16() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I16);
        return (uint16_t)(bits & UINT16_MAX);
    }
    constexpr int32_t as_i32() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I32);
        return (int32_t)sign_extend_bits(bits, 32);
    }
    constexpr uint32_t as_u32() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I32);
        return (uint32_t)(bits & UINT32_MAX);
    }
    constexpr int64_t as_i64() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I64);
        return i64_value;
    }
    constexpr uint64_t as_u64() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::I64);
        return bits;
    }
    constexpr uint32_t as_f32_bits() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::F32);
        return (uint32_t)bits;
    }
    constexpr uint64_t as_f64_bits() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::F64);
        return bits;
    }
    constexpr float as_f32() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::F32);
        return f32_value;
    }
    constexpr double as_f64() const noexcept {
        SIMJIT_ASSERT(dtype == ScalarDataType::F64);
        return f64_value;
    }

    ConstData retag(ScalarDataType to) const noexcept {
        switch (to) {
        case ScalarDataType::I1: return i1(as_unsigned() != 0);
        case ScalarDataType::I8: return u8((uint8_t)as_unsigned());
        case ScalarDataType::I16: return u16((uint16_t)as_unsigned());
        case ScalarDataType::I32: return u32((uint32_t)as_unsigned());
        case ScalarDataType::I64: return u64(as_unsigned());
        case ScalarDataType::F32: return f32_bits((uint32_t)as_unsigned());
        case ScalarDataType::F64: return f64_bits(as_unsigned());
        case ScalarDataType::I128: SIMJIT_ASSERT(0);
        }
        SIMJIT_UNREACHABLE();
    }

    bool operator==(ConstData right) const noexcept { return dtype == right.dtype && raw_bits() == right.raw_bits(); }
    bool operator!=(ConstData right) const noexcept { return !(*this == right); }

    constexpr bool is_zero() const noexcept { return as_unsigned() == 0; }
    constexpr bool is_all_ones() const noexcept {
        switch (dtype) {
        case ScalarDataType::I1: return as_unsigned() == 1;
        case ScalarDataType::I8: return as_unsigned() == UINT8_MAX;
        case ScalarDataType::I16: return as_unsigned() == UINT16_MAX;
        case ScalarDataType::I32: return as_unsigned() == UINT32_MAX;
        case ScalarDataType::I64: return as_unsigned() == UINT64_MAX;
        case ScalarDataType::F32:
        case ScalarDataType::F64:
        case ScalarDataType::I128: SIMJIT_ASSERT(0);
        }
        SIMJIT_UNREACHABLE();
    }
};

template <ScalarDataType X> struct ScalarDataTypeMap;
#define SIMJIT_MAP_TYPE(_scalar, _cpp)                              \
    template <> struct ScalarDataTypeMap<ScalarDataType::_scalar> { \
        using type = _cpp;                                          \
    };
SIMJIT_MAP_TYPE(I8, int8_t)
SIMJIT_MAP_TYPE(I16, int16_t)
SIMJIT_MAP_TYPE(I32, int32_t)
SIMJIT_MAP_TYPE(I64, int64_t)
SIMJIT_MAP_TYPE(F32, float)
SIMJIT_MAP_TYPE(F64, double)
SIMJIT_MAP_TYPE(I1, bool)
SIMJIT_MAP_TYPE(I128, __int128_t)
#undef SIMJIT_MAP_TYPE

enum class ArithUnaryOp : uint8_t {
    Not,
    Negate,
    Abs,
    Lzcnt,
    Tzcnt,
    Popcount,
    RoundNearest,  // round to nearest int
    RoundDown,     // round to -inf
    RoundUp,       // round to +inf
    RoundTruncate, // round to lower abs value
    Rcp,           // 1 / x, 0.5 ULP
    Sqrt,          // sqrt(x), 0.5 ULP
    Rsqrt,         // 1 / sqrt(x), 0.5 ULP
};

enum class ArithBinaryOp : uint8_t {
    Add,
    Sub,
    Mul,
    // Multiply lower 32 bits sign-extended to 64. Input must be already 64 bits. This compiles down to VPMULDQ.
    Mul64SE,
    // Multiply lower 32 bits zero-extended to 64. Input must be already 64 bits. This compiles down to VPMULUDQ.
    Mul64ZE,
    Div,
    UDiv,
    Mod,
    UMod,
    Min,
    Max,
    UMin,
    UMax,
    And,
    Or,
    Xor,
    AndNot,
    ShiftRightArith,
    ShiftRightLogical,
    ShiftLeftLogical,
    RotateLeft,
    RotateRight
};

enum class FpClass : uint8_t {
    NO = 0x0,
    FPC_INFINITE = 0x1,
    FPC_NAN = 0x2,
    FPC_SUBNORMAL = 0x4,
    FPC_ZERO = 0x8,
};
SIMJIT_DEFINE_ENUM_FLAGS(FpClass)

enum class ArithBinaryOpFlags : uint8_t {
    No = 0,
    SafetyCheck = 0x1,
    // For shifts and rotates, normalize the count modulo the scalar bit width before lowering.
    ShiftWraparound = 0x2,
    // For integer division/modulo,
    // replace invalid scalar divisors with 1 before lowering so target instructions do not fault.
    SafeDivision = 0x4
};
SIMJIT_DEFINE_ENUM_FLAGS(ArithBinaryOpFlags)

enum class CmpOp : uint8_t {
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    Equal,
    NotEqual,
};

enum class IntCastKind : uint8_t {
    Trunc,
    Sext,
    Zext,
};

enum class PredicateBinaryOp : uint8_t {
    And,
    Or,
    Xor,
    AndNot,
    XNor,
};

enum class LoadStoreKind : uint8_t {
    Aligned,
    Unaligned,
};

enum class CodeTransformations : uint16_t {
    No = 0,
    // Allow use of 32x32 -> 64 bit multiply
    MuldqInst = 1 << 0,
    // Replace multiplication by constant with series of shift+add instructions
    MulConstPeephole = 1 << 1,
    // Invert logical operations if that simplifies expression
    LogicalPeephole = 1 << 2,
    // Rewrite comparisons (a <= X) && (X <= b) - analogous to SQL BETWEEN (and variations) to sub+cmp
    BetweenPeephole = 1 << 3,
    // Allow reactive unrolling that equalizes the widths of different root groups.
    Unroll = 1 << 4,
    // Give duplicated aggregate work independent accumulators. Without this, unrolled copies update one shared
    // accumulator.
    AccSplit = 1 << 5,
    // Mask combining optimization - join together multiple mask registers, if possible
    MaskCombine = 1 << 6,
    // Allow use of ternarylogic instruction to replace series of bit operations
    TernarylogicInst = 1 << 7,
    // Allow use of fma instructions
    FmaInst = 1 << 8,
    // Expand i8 vector multiplications and shifts on targets that do not have a native byte multiply instruction.
    // Rewrite i8 shifts into permutes.
    // Implement i8 and i16 lzcnt.
    SmallArith = 1 << 9,
    // Replace integer division by constant with multiply + shift. Requires libdivide compiled in (SIMJIT_USE_LIBDIVIDE)
    ConstDiv = 1 << 10,
    // Collapse chained integer casts that do not have safety checks.
    CastPeephole = 1 << 11,
    // Split mixed integer/float casts so coefficient-changing work happens in one domain before same-width
    // cross-domain conversion. Up-casts prefer integer-domain widening; down-casts prefer float-domain narrowing.
    // This enables i8/i16 float casts through i32, and exposes vector-friendly normalization nodes. For float-to-int
    // down-casts, vectorized code can round in the narrower float domain before integer conversion.
    CastDecomposition = 1 << 12,
    // Increase vector loop width for small expression graphs when the node-count budget allows it.
    ProactiveUnroll = 1 << 13,
    // Remove simple identity binary operations during HIR construction.
    BinaryIdentityPeephole = 1 << 14,
    // Remove constant conditions during HIR construction.
    ConstantConditionPeephole = 1 << 15,

    All = MuldqInst | MulConstPeephole | LogicalPeephole | BetweenPeephole | Unroll | AccSplit | MaskCombine |
          TernarylogicInst | FmaInst | SmallArith | CastPeephole | ProactiveUnroll | CastDecomposition |
          BinaryIdentityPeephole |
          ConstantConditionPeephole
#if SIMJIT_USE_LIBDIVIDE
          // Actually we can just set it always, but I want to make it clear that libdivide is required.
          | ConstDiv
#endif
        ,
};
SIMJIT_DEFINE_ENUM_FLAGS(CodeTransformations)

class MemoryArena;

enum class Arch : uint8_t {
    Amd64_AVX512,
    Amd64_AVX512_YMM,
    Arm64_NEON,
#if defined(_M_X64) || defined(__x86_64__)
    Native = Amd64_AVX512,
#elif defined(_M_ARM64) || defined(__arm64__) || defined(__aarch64__)
    Native = Arm64_NEON,
#endif
};

constexpr bool is_x86_arch(Arch arch) noexcept {
    return arch == Arch::Amd64_AVX512 || arch == Arch::Amd64_AVX512_YMM;
}

// Automatic classification for arguments.
enum class ArgumentKind : uint8_t {
    // Value that has not been used
    Undefined = 0,
    SrcArr = 1,
    SrcGatherArr = 2,
    SrcIdxArr = 4,
    SrcConst = 8,
    Dst = 16,
    DstAgg = 32,
    DstSafetyCheck = 64,
    Table = SrcGatherArr | Dst
};
SIMJIT_DEFINE_ENUM_FLAGS(ArgumentKind)

using ArgumentIdx = size_t;

// Public expression objects are lightweight handles.
//
// Value, Predicate, MaybeValue, MaybePredicate, and Argument are small wrappers around builder-owned HIR pointers or
// indices. They are designed to be copied freely while constructing one function and to stay close to raw pointer cost
// on the expression-building path.
template <typename Tag> struct ArgumentT {
    ArgumentIdx idx_ = SIZE_MAX;
};

using Argument = ArgumentT<struct ArgumentTag>;

namespace hir {
struct Step;
struct Function;
} // namespace hir

// Non-owning handle to a HIR step. MUST be not empty (contain valid step).
// For optional values use MaybeValueT.
template <typename T> struct ValueT {
    hir::Step *step_ = nullptr;

    ValueT() = delete;
    constexpr ValueT(const ValueT<T> &) noexcept = default;
    constexpr ValueT(ValueT<T> &&) noexcept = default;
    constexpr ValueT &operator=(const ValueT<T> &) noexcept = default;
    constexpr ValueT &operator=(ValueT<T> &&) noexcept = default;

    ValueT(int) = delete;
    ValueT(double) = delete;
    ValueT(std::nullptr_t) = delete;
    constexpr ValueT(hir::Step *step) noexcept : step_(step) { SIMJIT_ASSERT(step_); }

    // Step is guaranteed to have datatype as first member.
    // We don't include hir here, and can't access it directly.
    ScalarDataType dtype() const noexcept { return *reinterpret_cast<ScalarDataType *>(step_); }
};

using Value = ValueT<struct ValueTag>;
using Predicate = ValueT<struct PredicateTag>;

template <typename T> struct MaybeValueT {
    hir::Step *step_ = nullptr;

    constexpr MaybeValueT() noexcept = default;
    constexpr MaybeValueT(const MaybeValueT<T> &) noexcept = default;
    constexpr MaybeValueT(MaybeValueT<T> &&) noexcept = default;
    constexpr MaybeValueT &operator=(const MaybeValueT<T> &) noexcept = default;
    constexpr MaybeValueT &operator=(MaybeValueT<T> &&) noexcept = default;

    MaybeValueT(int) = delete;
    MaybeValueT(double) = delete;
    constexpr explicit MaybeValueT(std::nullptr_t) noexcept {}
    constexpr MaybeValueT(hir::Step *step) noexcept : step_(step) {}
    constexpr MaybeValueT(ValueT<T> v) noexcept : step_(v.step_) {}

    constexpr ValueT<T> value() const noexcept {
        // The contract is that users of this class should explicitly check is_valid() and only then call this function.
        // Calling value() on unchecked value is breach of contract.
        SIMJIT_ASSERT(step_ != nullptr);
        return ValueT<T>(step_);
    }

    constexpr bool is_valid() const noexcept { return step_ != nullptr; }
};

using MaybeValue = MaybeValueT<struct ValueTag>;
using MaybePredicate = MaybeValueT<struct PredicateTag>;

// compiler.h
struct Context;

class FunctionBuilder {
public:
    FunctionBuilder() = delete;
    explicit FunctionBuilder(Context &ctx);
    FunctionBuilder(const FunctionBuilder &) = delete;
    FunctionBuilder(FunctionBuilder &&other) noexcept;
    FunctionBuilder &operator=(const FunctionBuilder &) = delete;
    FunctionBuilder &operator=(FunctionBuilder &&other) noexcept;
    ~FunctionBuilder() noexcept;

    // Can be used to disable vectorization for given expression. This can be used for 2 things:
    // 1. Preserve semantics for operations that need sequential behavior. This is the case for grouped aggregations,
    //   since their behavior is different vectorized and not-vectorized (vectorized can't process duplicate indices).
    // 2. Disable vectorization early if expression can't be vectorized anyway (due to lack of ISA support).
    //   This is internal usage.
    void scalar_only() noexcept;
    hir::Function *build();

    Argument arg(ScalarDataType dtype, ArgumentKind kind = ArgumentKind::Undefined);
    Argument arg_safety_check();

    // Stores

    void store(Value arg, Argument dst, LoadStoreKind kind = LoadStoreKind::Unaligned);
    void store(Predicate arg, Argument dst);
    void cond_store(Value arg, MaybePredicate cond, Argument dst, LoadStoreKind kind = LoadStoreKind::Unaligned);
    void cond_store(Predicate arg, MaybePredicate cond, Argument dst);
    void pack(Value arg, Predicate cond, Argument dst, Argument dst_size);
    void scatter(Value arg, Value idx, Argument dst);
    void cond_scatter(Value arg, Value idx, MaybePredicate cond, Argument dst);

    // Aggregates

    void arith_agg(Value arg, ArithBinaryOp op, Argument dst);
    void cond_arith_agg(Value arg, MaybePredicate cond, ArithBinaryOp op, Argument dst);
    void predicate_agg(Predicate arg, PredicateBinaryOp op, Argument dst);
    void grouped_arith_agg(Value arg, Value idx, ArithBinaryOp op, Argument table);
    void grouped_cond_arith_agg(Value arg, MaybePredicate cond, Value idx, ArithBinaryOp op, Argument table);

    // Loads and constants

    Value load(Argument arg, LoadStoreKind kind = LoadStoreKind::Unaligned);
    Predicate load_predicate(Argument arg);

    Value gather(Value idx, Argument arg);
    Value load_splat(Argument arg);
    Predicate load_predicate_splat(Argument arg);
    Value con_internal(ConstData data, ScalarDataType dtype);

    // Arithmetic and predicate operations

    Value arith_binary(Value left, Value right, ArithBinaryOp op, ArithBinaryOpFlags flags = ArithBinaryOpFlags::No);
    Value checked_op(Value op, MaybePredicate mask = {});
    Value arith_unary(Value arg, ArithUnaryOp op, bool safety_check = false);
    Predicate predicate_not(Predicate arg);
    Predicate predicate_binary(Predicate left, Predicate right, PredicateBinaryOp op);
    Value select(Predicate cond, Value truthy, Value falsy);
    Predicate select(Predicate cond, Predicate truthy, Predicate falsy);

    // Casts

    Value int_cast(Value arg, ScalarDataType dtype, IntCastKind kind, bool safety_check = false);
    Value float_cast(Value arg, ScalarDataType dtype, bool is_unsigned = false);
    Value bitcast(Value arg, ScalarDataType to);

    // Comparisons

    Predicate cmp(Value left, Value right, CmpOp op, bool is_unsigned = false);
    Predicate fpclass(Value arg, FpClass flags);

    //
    // Misc operations
    //

    Value permute(Value arg, uint64_t permute_idxs, bool is_bit);
    Value index(ScalarDataType dtype);

    //
    // High-level stuff
    //

    //
    // Aggregates
    //

    void sum(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::Add, dst); }
    void product(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::Mul, dst); }
    void min_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::Min, dst); }
    void max_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::Max, dst); }
    void umin_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::UMin, dst); }
    void umax_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::UMax, dst); }
    void and_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::And, dst); }
    void or_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::Or, dst); }
    void xor_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::Xor, dst); }
    void andnot_agg(Value arg, Argument dst) { arith_agg(arg, ArithBinaryOp::AndNot, dst); }
    void sum_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::Add, dst); }
    void product_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::Mul, dst); }
    void min_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::Min, dst); }
    void max_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::Max, dst); }
    void umin_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::UMin, dst); }
    void umax_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::UMax, dst); }
    void and_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::And, dst); }
    void or_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::Or, dst); }
    void xor_agg_if(Value arg, Predicate cond, Argument dst) { cond_arith_agg(arg, cond, ArithBinaryOp::Xor, dst); }
    void andnot_agg_if(Value arg, Predicate cond, Argument dst) {
        cond_arith_agg(arg, cond, ArithBinaryOp::AndNot, dst);
    }

    void and_agg(Predicate arg, Argument dst) { predicate_agg(arg, PredicateBinaryOp::And, dst); }
    void or_agg(Predicate arg, Argument dst) { predicate_agg(arg, PredicateBinaryOp::Or, dst); }
    void andnot_agg(Predicate arg, Argument dst) { predicate_agg(arg, PredicateBinaryOp::AndNot, dst); }
    void xor_agg(Predicate arg, Argument dst) { predicate_agg(arg, PredicateBinaryOp::Xor, dst); }
    void countif(Predicate cond, Argument dst);

    // Grouped aggregates
    void grouped_sum(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::Add, table); }
    void grouped_product(Value arg, Value idx, Argument table) {
        grouped_arith_agg(arg, idx, ArithBinaryOp::Mul, table);
    }
    void grouped_min(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::Min, table); }
    void grouped_max(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::Max, table); }
    void grouped_umin(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::UMin, table); }
    void grouped_umax(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::UMax, table); }
    void grouped_and(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::And, table); }
    void grouped_or(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::Or, table); }
    void grouped_xor(Value arg, Value idx, Argument table) { grouped_arith_agg(arg, idx, ArithBinaryOp::Xor, table); }
    void grouped_andnot(Value arg, Value idx, Argument table) {
        grouped_arith_agg(arg, idx, ArithBinaryOp::AndNot, table);
    }
    void grouped_sum_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::Add, table);
    }
    void grouped_product_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::Mul, table);
    }
    void grouped_min_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::Min, table);
    }
    void grouped_max_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::Max, table);
    }
    void grouped_umin_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::UMin, table);
    }
    void grouped_umax_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::UMax, table);
    }
    void grouped_and_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::And, table);
    }
    void grouped_or_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::Or, table);
    }
    void grouped_xor_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::Xor, table);
    }
    void grouped_andnot_if(Value arg, Predicate cond, Value idx, Argument table) {
        grouped_cond_arith_agg(arg, cond, idx, ArithBinaryOp::AndNot, table);
    }

    //
    // Loads and constants
    //

    Predicate input_predicate_arg() { return load_predicate(arg(ScalarDataType::I1)); }
    Value input_arg(ScalarDataType dtype, LoadStoreKind kind = LoadStoreKind::Unaligned) {
        return load(arg(dtype), kind);
    }
    Value input_splat_arg(ScalarDataType dtype) { return load_splat(arg(dtype)); }
    Predicate input_predicate_splat_arg() { return load_predicate_splat(arg(ScalarDataType::I1)); }

    void output_arg(Value value, LoadStoreKind kind = LoadStoreKind::Unaligned) {
        store(value, arg(value.dtype()), kind);
    }
    void output_arg(Predicate value) { store(value, arg(ScalarDataType::I1)); }

    template <ScalarDataType Ty> Value con(typename ScalarDataTypeMap<Ty>::type data) {
        if constexpr (Ty == ScalarDataType::F32) {
            return con_internal(ConstData::f32(data), ScalarDataType::F32);
        } else if constexpr (Ty == ScalarDataType::F64) {
            return con_internal(ConstData::f64(data), ScalarDataType::F64);
        } else {
            return con_internal(ConstData::i64((int64_t)data), Ty);
        }
    }
    template <ScalarDataType Ty> Value ucon(std::make_unsigned_t<typename ScalarDataTypeMap<Ty>::type> data) {
        static_assert(Ty != ScalarDataType::F32 && Ty != ScalarDataType::F64);
        // Zero extend
        return con_internal(ConstData::u64((uint64_t)data), Ty);
    }
    Value con(int64_t data, ScalarDataType dt) { return con_internal(ConstData::i64(data), dt); }
    Value i8(int8_t data) { return con<ScalarDataType::I8>(data); }
    Value u8(uint8_t data) { return ucon<ScalarDataType::I8>(data); }
    Value i16(int16_t data) { return con<ScalarDataType::I16>(data); }
    Value u16(uint16_t data) { return ucon<ScalarDataType::I16>(data); }
    Value i32(int32_t data) { return con<ScalarDataType::I32>(data); }
    Value u32(uint32_t data) { return ucon<ScalarDataType::I32>(data); }
    Value i64(int64_t data) { return con<ScalarDataType::I64>(data); }
    Value u64(uint64_t data) { return ucon<ScalarDataType::I64>(data); }
    Value f32(float data) { return con<ScalarDataType::F32>(data); }
    Value f64(double data) { return con<ScalarDataType::F64>(data); }
    Predicate i1(bool data);

    Predicate true_() { return i1(true); }
    Predicate false_() { return i1(false); }

    //
    // Binary integer arithmetic
    //

    Value add(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Add); }
    Value sub(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Sub); }
    Value mul(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Mul); }
    // if safe_division = false, might fault on invalid input (0 or INT_MIN/-1)
    Value div(Value left, Value right, bool safe_division = true) {
        return arith_binary(left, right, ArithBinaryOp::Div,
                            safe_division ? ArithBinaryOpFlags::SafeDivision : ArithBinaryOpFlags::No);
    }
    Value udiv(Value left, Value right, bool safe_division = true) {
        return arith_binary(left, right, ArithBinaryOp::UDiv,
                            safe_division ? ArithBinaryOpFlags::SafeDivision : ArithBinaryOpFlags::No);
    }
    Value mod(Value left, Value right, bool safe_division = true) {
        return arith_binary(left, right, ArithBinaryOp::Mod,
                            safe_division ? ArithBinaryOpFlags::SafeDivision : ArithBinaryOpFlags::No);
    }
    Value umod(Value left, Value right, bool safe_division = true) {
        return arith_binary(left, right, ArithBinaryOp::UMod,
                            safe_division ? ArithBinaryOpFlags::SafeDivision : ArithBinaryOpFlags::No);
    }
    Value min(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Min); }
    Value max(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Max); }
    Value umin(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::UMin); }
    Value umax(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::UMax); }
    Value or_(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Or); }
    Value and_(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::And); }
    Value xor_(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::Xor); }
    Value xnor(Value left, Value right) { return not_(xor_(left, right)); }
    Value andnot(Value left, Value right) { return arith_binary(left, right, ArithBinaryOp::AndNot); }
    Value sll(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::ShiftLeftLogical, ArithBinaryOpFlags::ShiftWraparound);
    }
    Value srl(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::ShiftRightLogical, ArithBinaryOpFlags::ShiftWraparound);
    }
    Value sra(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::ShiftRightArith, ArithBinaryOpFlags::ShiftWraparound);
    }
    Value rotl(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::RotateLeft, ArithBinaryOpFlags::ShiftWraparound);
    }
    Value rotr(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::RotateRight, ArithBinaryOpFlags::ShiftWraparound);
    }

    //
    // Checked binary
    //
    Value add_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::Add, ArithBinaryOpFlags::SafetyCheck);
    }
    Value sub_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::Sub, ArithBinaryOpFlags::SafetyCheck);
    }
    Value mul_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::Mul, ArithBinaryOpFlags::SafetyCheck);
    }
    Value div_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::Div, ArithBinaryOpFlags::SafetyCheck);
    }
    Value udiv_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::UDiv, ArithBinaryOpFlags::SafetyCheck);
    }
    Value mod_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::Mod, ArithBinaryOpFlags::SafetyCheck);
    }
    Value umod_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::UMod, ArithBinaryOpFlags::SafetyCheck);
    }
    Value sll_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::ShiftLeftLogical, ArithBinaryOpFlags::SafetyCheck);
    }
    Value srl_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::ShiftRightLogical, ArithBinaryOpFlags::SafetyCheck);
    }
    Value sra_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::ShiftRightArith, ArithBinaryOpFlags::SafetyCheck);
    }
    Value rotl_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::RotateLeft, ArithBinaryOpFlags::SafetyCheck);
    }
    Value rotr_checked(Value left, Value right) {
        return arith_binary(left, right, ArithBinaryOp::RotateRight, ArithBinaryOpFlags::SafetyCheck);
    }

    //
    // Unary arithmetic
    //

    Value negate(Value arg) { return arith_unary(arg, ArithUnaryOp::Negate); }
    Value negate_checked(Value arg) { return arith_unary(arg, ArithUnaryOp::Negate, true); }
    Value abs(Value arg) { return arith_unary(arg, ArithUnaryOp::Abs); }
    Value abs_checked(Value arg) { return arith_unary(arg, ArithUnaryOp::Abs, true); }
    Value not_(Value arg) { return arith_unary(arg, ArithUnaryOp::Not); }
    Value lzcnt(Value arg) { return arith_unary(arg, ArithUnaryOp::Lzcnt); }
    Value tzcnt(Value arg) { return arith_unary(arg, ArithUnaryOp::Tzcnt); }
    Value popcnt(Value arg) { return arith_unary(arg, ArithUnaryOp::Popcount); }
    Value round_nearest_even(Value arg) { return arith_unary(arg, ArithUnaryOp::RoundNearest); }
    Value round_down(Value arg) { return arith_unary(arg, ArithUnaryOp::RoundDown); }
    Value round_up(Value arg) { return arith_unary(arg, ArithUnaryOp::RoundUp); }
    Value round_toward_zero(Value arg) { return arith_unary(arg, ArithUnaryOp::RoundTruncate); }
    Value sqrt(Value arg) { return arith_unary(arg, ArithUnaryOp::Sqrt); }
    Value rsqrt(Value arg) { return arith_unary(arg, ArithUnaryOp::Rsqrt); }
    Value rcp(Value arg) { return arith_unary(arg, ArithUnaryOp::Rcp); }

    //
    // Casts
    //

    Value trunc(Value arg, ScalarDataType dtype) { return int_cast(arg, dtype, IntCastKind::Trunc); }
    Value trunc_checked(Value arg, ScalarDataType dtype) { return int_cast(arg, dtype, IntCastKind::Trunc, true); }
    Value sext(Value arg, ScalarDataType dtype) { return int_cast(arg, dtype, IntCastKind::Sext, false); }
    Value zext(Value arg, ScalarDataType dtype) { return int_cast(arg, dtype, IntCastKind::Zext, false); }
    Value signed_cast(Value arg, ScalarDataType to);
    Value unsigned_cast(Value arg, ScalarDataType to);

    //
    // Comparisons
    //

    Predicate cmp_eq(Value left, Value right) { return cmp(left, right, CmpOp::Equal); }
    Predicate cmp_ne(Value left, Value right) { return cmp(left, right, CmpOp::NotEqual); }
    Predicate cmp_gt(Value left, Value right) { return cmp(left, right, CmpOp::Greater); }
    Predicate cmp_ge(Value left, Value right) { return cmp(left, right, CmpOp::GreaterEqual); }
    Predicate cmp_lt(Value left, Value right) { return cmp(left, right, CmpOp::Less); }
    Predicate cmp_le(Value left, Value right) { return cmp(left, right, CmpOp::LessEqual); }
    Predicate cmp_ueq(Value left, Value right) { return cmp(left, right, CmpOp::Equal, true); }
    Predicate cmp_une(Value left, Value right) { return cmp(left, right, CmpOp::NotEqual, true); }
    Predicate cmp_ugt(Value left, Value right) { return cmp(left, right, CmpOp::Greater, true); }
    Predicate cmp_uge(Value left, Value right) { return cmp(left, right, CmpOp::GreaterEqual, true); }
    Predicate cmp_ult(Value left, Value right) { return cmp(left, right, CmpOp::Less, true); }
    Predicate cmp_ule(Value left, Value right) { return cmp(left, right, CmpOp::LessEqual, true); }
    Predicate bit_test(Value left, Value right) { return cmp_ne(and_(left, right), con(0, left.dtype())); }
    Predicate bit_testn(Value left, Value right) { return cmp_eq(and_(left, right), con(0, left.dtype())); }
    Predicate is_positive(Value arg) { return cmp_gt(arg, con(0, arg.dtype())); }
    Predicate is_negative(Value arg) { return cmp_lt(arg, con(0, arg.dtype())); }
    Predicate bool2bit(Value arg) { return cmp_ne(arg, con(0, arg.dtype())); }

    //
    // Fpclass
    //

    Predicate isnan(Value arg) { return fpclass(arg, FpClass::FPC_NAN); }
    Predicate isinf(Value arg) { return fpclass(arg, FpClass::FPC_INFINITE); }
    Predicate isfinite(Value arg) { return not_(fpclass(arg, FpClass::FPC_INFINITE | FpClass::FPC_NAN)); }
    Predicate isnormal(Value arg) {
        return not_(
            fpclass(arg, FpClass::FPC_INFINITE | FpClass::FPC_NAN | FpClass::FPC_SUBNORMAL | FpClass::FPC_ZERO));
    }

    //
    // Predicate binary
    //

    Predicate not_(Predicate x) { return predicate_not(x); }
    Predicate or_(Predicate left, Predicate right) { return predicate_binary(left, right, PredicateBinaryOp::Or); }
    Predicate and_(Predicate left, Predicate right) { return predicate_binary(left, right, PredicateBinaryOp::And); }
    Predicate andnot(Predicate left, Predicate right) {
        return predicate_binary(left, right, PredicateBinaryOp::AndNot);
    }
    Predicate xnor(Predicate left, Predicate right) { return predicate_binary(left, right, PredicateBinaryOp::XNor); }
    Predicate xor_(Predicate left, Predicate right) { return predicate_binary(left, right, PredicateBinaryOp::Xor); }

    //
    // Blend
    //

    Value zero_select(Value truthy, Predicate cond) { return select(cond, truthy, con(0, truthy.dtype())); }
    Value bit2bool(Predicate cond) { return zero_select(con<ScalarDataType::I8>(1), cond); }

    //
    // Bit stuff
    //

    // Compute log2. If argument is zero, behavior is undefined
    Value log2_no_zero(Value arg);
    Value log2(Value arg);
    Predicate has_single_bit(Value arg);
    Value byteswap(Value arg);
    Value bit_floor(Value arg);
    Value bit_ceil(Value arg);

    Value permute_i64_i8(Value arg, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7,
                         uint8_t a8);
    Value permute_i64_i16(Value arg, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4);
    Value permute_i64_i32(Value arg, uint8_t a1, uint8_t a2);
    Value permute_i32_i8(Value arg, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4);
    Value permute_i32_i16(Value arg, uint8_t a1, uint8_t a2);
    Value permute_i16_i8(Value arg, uint8_t a1, uint8_t a2);

    Value permute_i8_bits(Value arg, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7,
                          uint8_t a8);

    Value reverse_bits_i8(Value arg) { return permute_i8_bits(arg, 7, 6, 5, 4, 3, 2, 1, 0); }
    Value reverse_bits_full(Value arg) { return byteswap(reverse_bits_i8(arg)); }
    Value replicate_ith_bit_i8(Value arg, int bit_idx) {
        return permute_i8_bits(arg, bit_idx, bit_idx, bit_idx, bit_idx, bit_idx, bit_idx, bit_idx, bit_idx);
    }

    //
    // Library functions
    //

    // if arg < 0 ? -1 : 1
    Value sign_no_zero(Value arg);
    // if arg < 0 ? -1 : arg == 0 ? 0 : 1
    Value sign(Value arg);

    // if sign < 0 ? -arg : arg
    Value copysign_no_zero(Value sign, Value arg);
    // if sign < 0 ? -arg sign == 0 ? 0 : arg
    Value copysign(Value sign, Value arg);

private:
    struct FunctionBuilderImpl *impl_ = nullptr;
};

} // namespace simjit
