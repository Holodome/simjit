// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/compiler.h"
#include "simjit/core/expr.h"
#include "simjit/core/hir.h"
#include "simjit/detail/base.h"
#include "simjit/simjit.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define unsupported(...) \
    simjit_exception(ErrorModule::HIR, ErrorKind::Unsupported, ErrorSubKind::UnsupportedFeature, __VA_ARGS__)
#define unsupported_limit(...) \
    simjit_exception(ErrorModule::HIR, ErrorKind::Unsupported, ErrorSubKind::LimitExceeded, __VA_ARGS__)
#define invalid_input(...) \
    simjit_exception(ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::InvalidConfiguration, __VA_ARGS__)
#define invalid_type(...) \
    simjit_exception(ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::TypeError, __VA_ARGS__)
#define invalid_access(...) \
    simjit_exception(ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::InvalidArgumentAccess, __VA_ARGS__)
#define missing_output(...) \
    simjit_exception(ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::MissingRequiredOutput, __VA_ARGS__)

#define check_value(_x) \
    if (!is_valid_value(_x)) invalid_input("invalid value")
#define check_predicate(_x) \
    if (!is_valid_predicate(_x)) invalid_input("invalid predicate")

namespace simjit {
namespace hir {

using namespace simjit::types;

static bool is_valid_value(MaybeValue x) noexcept {
    return x.is_valid() && x.value().dtype() != I1;
}

static bool is_valid_predicate(MaybePredicate x) noexcept {
    return x.is_valid() && x.value().dtype() == I1;
}

static bool is_compact_float_cast_int_dtype(ScalarDataType dtype) noexcept {
    return dtype == I32 || dtype == I64;
}

static ScalarDataType int_dtype_for_float(ScalarDataType dtype) noexcept {
    SIMJIT_ASSERT(is_float_dtype(dtype));
    return scalar_dtype_size(dtype) == 4 ? I32 : I64;
}

static ScalarDataType float_dtype_for_int(ScalarDataType dtype) noexcept {
    SIMJIT_ASSERT(is_simple_int_dtype(dtype));
    return scalar_dtype_size(dtype) <= 4 ? F32 : F64;
}

enum class FloatCastGroup : uint8_t {
    SmallInt,
    FullInt,
    Float,
    Count,
};

enum class SizeRelation : uint8_t {
    Smaller,
    Same,
    Larger,
    Count,
};

enum class FloatCastRewrite : uint8_t {
    Original,
    FloatIdentity,
    IntBridgeToTargetFloat,
    FloatBridgeFromSourceInt,
    FloatBridgeToTargetInt,
    FloatBridgeToSmallInt,
};

static bool classify_float_cast_group(ScalarDataType dtype, FloatCastGroup *group) noexcept {
    if (dtype == I8 || dtype == I16) {
        *group = FloatCastGroup::SmallInt;
        return true;
    }
    if (is_compact_float_cast_int_dtype(dtype)) {
        *group = FloatCastGroup::FullInt;
        return true;
    }
    if (is_float_dtype(dtype)) {
        *group = FloatCastGroup::Float;
        return true;
    }
    return false;
}

static SizeRelation size_relation(ScalarDataType from, ScalarDataType to) noexcept {
    size_t from_size = scalar_dtype_size(from);
    size_t to_size = scalar_dtype_size(to);
    if (from_size < to_size) { return SizeRelation::Smaller; }
    if (from_size > to_size) { return SizeRelation::Larger; }
    return SizeRelation::Same;
}

static FloatCastRewrite classify_float_cast_rewrite(ScalarDataType from, ScalarDataType to) noexcept {
    FloatCastGroup from_group;
    FloatCastGroup to_group;
    if (!classify_float_cast_group(from, &from_group) || !classify_float_cast_group(to, &to_group)) {
        return FloatCastRewrite::Original;
    }

    using R = FloatCastRewrite;
    static constexpr R O = R::Original;
    static constexpr R BIF = R::IntBridgeToTargetFloat;
    static constexpr R BFI = R::FloatBridgeFromSourceInt;
    static constexpr R BFTI = R::FloatBridgeToTargetInt;
    static constexpr R BFSI = R::FloatBridgeToSmallInt;
    static constexpr R ID = R::FloatIdentity;
    static constexpr R matrix[(size_t)FloatCastGroup::Count][(size_t)FloatCastGroup::Count]
                             [(size_t)SizeRelation::Count] = {
                                 // From small integer.
                                 {
                                     {O, O, O},
                                     {O, O, O},
                                     {BIF, BIF, BIF},
                                 },
                                 // From i32/i64.
                                 {
                                     {O, O, O},
                                     {O, O, O},
                                     {BIF, O, BFI},
                                 },
                                 // From float.
                                 {
                                     {BFSI, BFSI, BFSI},
                                     {BFTI, O, BFTI},
                                     {O, ID, O},
                                 },
    };
    return matrix[(size_t)from_group][(size_t)to_group][(size_t)size_relation(from, to)];
}

static CmpOp invert_cmp_op(CmpOp op) noexcept {
    switch (op) {
    case CmpOp::Less: return CmpOp::GreaterEqual;
    case CmpOp::Greater: return CmpOp::LessEqual;
    case CmpOp::LessEqual: return CmpOp::Greater;
    case CmpOp::GreaterEqual: return CmpOp::Less;
    case CmpOp::Equal: return CmpOp::NotEqual;
    case CmpOp::NotEqual: return CmpOp::Equal;
    }
    SIMJIT_UNREACHABLE();
}

static bool is_equality_cmp_op(CmpOp op) noexcept {
    return op == CmpOp::Equal || op == CmpOp::NotEqual;
}

static int rhs_cmp_preference(const Step *step) noexcept {
    if (step->is(StepKind::Const)) { return 2; }
    if (step->is(StepKind::Load) || step->is(StepKind::LoadSplat)) { return 1; }
    return 0;
}

static bool is_i8_constant_shift_or_rotate(ArithBinaryOp op, const Step *right) noexcept {
    if (!right->is(StepKind::Const)) { return false; }
    return op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftRightLogical ||
           op == ArithBinaryOp::ShiftLeftLogical || op == ArithBinaryOp::RotateLeft || op == ArithBinaryOp::RotateRight;
}

static SpecialOp popcount_special_op(ScalarDataType dtype) noexcept {
    return dtype == I8 || dtype == I16 ? SpecialOp::SmallPopcount : SpecialOp::LargePopcount;
}

static SpecialOp arith_binary_special_ops(const Step *step) noexcept {
    if (!step->is(StepKind::ArithBinary)) { return SpecialOp::None; }

    SpecialOp result = SpecialOp::None;
    const auto &data = step->step_data<StepKind::ArithBinary>();
    if (step->dtype == I64 && data.op == ArithBinaryOp::Mul) { result |= SpecialOp::I64Mul; }
    // x86 i8 constant shifts and rotates use the GFNI permute peephole in MIR.
    if (step->dtype == I8 && is_i8_constant_shift_or_rotate(data.op, data.right)) {
        result |= SpecialOp::I8ConstantShift;
    }
    return result;
}

static SpecialOp arith_unary_special_ops(const Step *step) noexcept {
    if (!step->is(StepKind::ArithUnary)) { return SpecialOp::None; }

    const auto &data = step->step_data<StepKind::ArithUnary>();
    if (data.op == ArithUnaryOp::Popcount || data.op == ArithUnaryOp::Tzcnt) {
        return popcount_special_op(step->dtype);
    }
    if (data.op == ArithUnaryOp::Lzcnt && (step->dtype == I8 || step->dtype == I16)) {
        // x86 lowers small lzcnt through vector popcount.
        return SpecialOp::SmallPopcount;
    }
    return SpecialOp::None;
}

} // namespace hir

using namespace ::simjit::hir;

struct StepPair {
    Step *left;
    Step *right;
};

static StepPair ordered_step_pair(Step *left, Step *right) noexcept {
    if (std::less<Step *>{}(right, left)) { std::swap(left, right); }
    return {left, right};
}

static bool is_commutative_arith_cse_op(ArithBinaryOp op, ScalarDataType dtype) noexcept {
    if (!is_int_dtype(dtype)) { return false; }
    switch (op) {
    case ArithBinaryOp::Add:
    case ArithBinaryOp::Mul:
    case ArithBinaryOp::Mul64SE:
    case ArithBinaryOp::Mul64ZE:
    case ArithBinaryOp::Min:
    case ArithBinaryOp::Max:
    case ArithBinaryOp::UMin:
    case ArithBinaryOp::UMax:
    case ArithBinaryOp::And:
    case ArithBinaryOp::Or:
    case ArithBinaryOp::Xor: return true;
    default: return false;
    }
}

static bool is_commutative_predicate_cse_op(PredicateBinaryOp op) noexcept {
    switch (op) {
    case PredicateBinaryOp::And:
    case PredicateBinaryOp::Or:
    case PredicateBinaryOp::Xor:
    case PredicateBinaryOp::XNor: return true;
    case PredicateBinaryOp::AndNot: return false;
    }
    SIMJIT_UNREACHABLE();
}

enum class ConstClass : uint8_t {
    Other,
    Zero,
    One,
    AllOnes,
    BoolFalse,
    BoolTrue,
};

static ConstClass classify_const(const Step *step) noexcept {
    if (!step->is(StepKind::Const)) return ConstClass::Other;
    const ConstData &data = step->step_data<StepKind::Const>();
    if (data.dtype == I1) { return data.as_unsigned() == 0 ? ConstClass::BoolFalse : ConstClass::BoolTrue; }
    if (data.dtype != I128 && data.is_zero()) { return ConstClass::Zero; }
    switch (data.dtype) {
    case I1: SIMJIT_UNREACHABLE();
    case I8:
    case I16:
    case I32:
    case I64:
        if (data.as_unsigned() == 1) return ConstClass::One;
        if (data.is_all_ones()) return ConstClass::AllOnes;
        return ConstClass::Other;
    case F32:
        if (data.as_f32_bits() == ConstData::f32(1.0f).as_f32_bits()) return ConstClass::One;
        if (data.as_f32_bits() == UINT32_MAX) return ConstClass::AllOnes;
        return ConstClass::Other;
    case F64:
        if (data.as_f64_bits() == ConstData::f64(1.0).as_f64_bits()) return ConstClass::One;
        if (data.as_f64_bits() == UINT64_MAX) return ConstClass::AllOnes;
        return ConstClass::Other;
    case I128: return ConstClass::Other;
    }
    SIMJIT_UNREACHABLE();
}

static Step *simplify_optional_cond(Context *ctx, Step *cond) noexcept {
    if (bool(ctx->transformations & CodeTransformations::ConstantConditionPeephole) && cond != nullptr &&
        classify_const(cond) == ConstClass::BoolTrue) {
        return nullptr;
    }
    return cond;
}

struct FunctionBuilderImpl {
    StepMaker sm;
    Context *ctx;
    std::vector<ArgumentDecl> args{};
    std::optional<ArgumentIdx> safety_check_arg{};
    std::vector<Accumulator> accs{};
    std::vector<Step *> step_roots{};
    struct RememberedExpr {
        Step *key;
        Step *result;
    };
    std::unordered_multimap<uint64_t, RememberedExpr> hash_map{};
    std::unordered_map<Step *, Step *> reverse_cse{};
    SpecialOp special_ops = SpecialOp::None;
    // Disable vectorization early to avoid relying on exceptions further during compilation process.
    VectorizationHint vectorization_hint = VectorizationHint::None;

    FunctionBuilderImpl() = delete;
    explicit FunctionBuilderImpl(Context *x) noexcept : sm(x->arena), ctx(x) {}

    ArgumentDecl &get_arg(ArgumentIdx idx) { return args.at(idx); }

    template <StepKind kind, typename F> Step *cse(uint64_t hash, F f) {
        auto [first, last] = hash_map.equal_range(hash);
        for (auto it = first; it != last; ++it) {
            const RememberedExpr &x = it->second;
            if (x.key->kind == kind && f(x.key->step_data<kind>())) { return x.result; }
        }
        return nullptr;
    }
    template <StepKind kind, typename F> Step *cse(uint64_t hash, ScalarDataType dtype, F f) {
        auto [first, last] = hash_map.equal_range(hash);
        for (auto it = first; it != last; ++it) {
            const RememberedExpr &x = it->second;
            if (x.key->dtype == dtype && x.key->kind == kind && f(x.key->step_data<kind>())) { return x.result; }
        }
        return nullptr;
    }

    Step *remember_expr(uint64_t hash, Step *s, Step *key = nullptr) {
        if (key == nullptr) { key = s; }
        hash_map.emplace(hash, RememberedExpr{key, s});
        if (key != s) { reverse_cse.insert_or_assign(s, key); }
        return s;
    }

    Step *original_expr(Step *result) const {
        auto it = reverse_cse.find(result);
        return it == reverse_cse.end() ? result : it->second;
    }

    Step *arith_agg(Step *arg, Step *cond, ArithBinaryOp op, const Accumulator &acc) {
        cond = simplify_optional_cond(ctx, cond);
        ScalarDataType dtype = acc.dtype;
        if (dtype == I128) {
            SIMJIT_ASSERT(arg->dtype == I128);
            return sm.i128_agg({op, arg, nullptr, acc.idx, cond}, dtype);
        }
        SIMJIT_ASSERT(arg->dtype == dtype);
        return sm.acc_arith_bin({op, arg, acc.idx, cond}, dtype);
    }

    Step *countif(Step *arg, const Accumulator &acc) {
        return sm.countif({ArithBinaryOp::Add, arg, acc.idx}, acc.dtype);
    }

    Step *predicate_agg(Step *arg, PredicateBinaryOp op, const Accumulator &acc) {
        SIMJIT_ASSERT(arg->dtype == I1);
        SIMJIT_ASSERT(acc.dtype == I1);
        return sm.acc_predicate_bin({op, arg, acc.idx}, I1);
    }
    Step *scatter(Step *arg, Step *idx, Step *cond, const ArgumentDecl &dst) {
        cond = simplify_optional_cond(ctx, cond);
        ScalarDataType dtype = arg->dtype;
        SIMJIT_ASSERT(dst.dtype == dtype || (dst.interleaved && dst.interleaved_half_dtype == dtype));
        return sm.scatter({arg, idx, dst.idx, cond}, dtype);
    }
    Step *store_interleave(Step *even, Step *odd, Step *cond, const ArgumentDecl &dst, LoadStoreKind kind) {
        cond = simplify_optional_cond(ctx, cond);
        SIMJIT_ASSERT(dst.interleaved);
        SIMJIT_ASSERT(dst.interleaved_half_dtype == even->dtype);
        SIMJIT_ASSERT(dst.interleaved_half_dtype == odd->dtype);
        return sm.store_interleave({even, odd, dst.idx, kind, cond}, even->dtype);
    }
    Step *pack(Step *arg, Step *cond, const ArgumentDecl &dst, const Accumulator &dst_size_acc) {
        ScalarDataType dtype = arg->dtype;
        SIMJIT_ASSERT(dst.dtype == dtype);
        SIMJIT_ASSERT(dst_size_acc.dtype == I64);
        return sm.pack({arg, cond, dst.idx, dst_size_acc.idx}, dtype);
    }
    Step *store(Step *arg, Step *cond, const ArgumentDecl &dst, LoadStoreKind kind) {
        cond = simplify_optional_cond(ctx, cond);
        ScalarDataType dtype = arg->dtype;
        SIMJIT_ASSERT(dst.dtype == dtype);
        return sm.store({arg, dst.idx, kind, cond}, dtype);
    }

    Step *load(const ArgumentDecl &arg, LoadStoreKind kind) { return sm.load({arg.idx, kind}, arg.dtype); }

    Step *load_deinterleave(const ArgumentDecl &arg, LoadStoreKind kind, uint8_t lane_count, uint8_t lane) {
        SIMJIT_ASSERT(arg.interleaved);
        return sm.load_deinterleave({arg.idx, kind, lane_count, lane}, arg.interleaved_half_dtype);
    }

    Step *gather(Step *idx, const ArgumentDecl &dst, ScalarDataType dtype) { return sm.gather({idx, dst.idx}, dtype); }

    Step *con(ConstData data, ScalarDataType dtype) { return sm.con(data, dtype); }

    Step *load_splat(const ArgumentDecl &arg, ScalarDataType dtype, size_t byte_offset = 0) {
        SIMJIT_ASSERT(dtype == arg.dtype || (arg.interleaved && dtype == arg.interleaved_half_dtype));
        return sm.load_splat({arg.idx, LoadStoreKind::Unaligned, byte_offset}, dtype);
    }

    Step *zero_const(ScalarDataType dtype) {
        switch (dtype) {
        case I1:
        case I8:
        case I16:
        case I32:
        case I64:
        case F32:
        case F64: return sm.con(ConstData::u64(0).retag(dtype), dtype);
        case I128: return sm.con(ConstData::i64(0), I128);
        }
        SIMJIT_UNREACHABLE();
    }

    Step *bool_const(bool value) { return sm.con(ConstData::i1(value), I1); }

    Step *binary_identity_peephole(Step *left, Step *right, ArithBinaryOp op) {
        ConstClass left_class = classify_const(left);
        ConstClass right_class = classify_const(right);
        switch (op) {
        case ArithBinaryOp::Add:
            if (left_class == ConstClass::Zero) return right;
            if (right_class == ConstClass::Zero) return left;
            return nullptr;
        case ArithBinaryOp::Sub:
            if (right_class == ConstClass::Zero) return left;
            return nullptr;
        case ArithBinaryOp::Mul:
            if (left_class == ConstClass::One) return right;
            if (right_class == ConstClass::One) return left;
            if (is_int_dtype(left->dtype)) {
                if (left_class == ConstClass::Zero) return left;
                if (right_class == ConstClass::Zero) return right;
            }
            return nullptr;
        case ArithBinaryOp::Div:
        case ArithBinaryOp::UDiv:
            if (right_class == ConstClass::One) return left;
            return nullptr;
        case ArithBinaryOp::Min:
        case ArithBinaryOp::Max:
        case ArithBinaryOp::UMin:
        case ArithBinaryOp::UMax:
            if (left == right) return left;
            return nullptr;
        case ArithBinaryOp::And:
            if (left == right) return left;
            if (left_class == ConstClass::AllOnes) return right;
            if (right_class == ConstClass::AllOnes) return left;
            return nullptr;
        case ArithBinaryOp::Or:
            if (left == right) return left;
            if (left_class == ConstClass::Zero) return right;
            if (right_class == ConstClass::Zero) return left;
            return nullptr;
        case ArithBinaryOp::Xor:
            if (left_class == ConstClass::Zero) return right;
            if (right_class == ConstClass::Zero) return left;
            if (left == right) return zero_const(left->dtype);
            return nullptr;
        case ArithBinaryOp::ShiftRightArith:
        case ArithBinaryOp::ShiftRightLogical:
        case ArithBinaryOp::ShiftLeftLogical:
        case ArithBinaryOp::RotateLeft:
        case ArithBinaryOp::RotateRight:
            if (right_class == ConstClass::Zero) return left;
            return nullptr;
        case ArithBinaryOp::Mul64SE:
        case ArithBinaryOp::Mul64ZE:
        case ArithBinaryOp::Mod:
        case ArithBinaryOp::UMod:
        case ArithBinaryOp::AndNot: return nullptr;
        }
        SIMJIT_UNREACHABLE();
    }

    Step *binary_identity_peephole(const Step *left, const Step *right, CmpOp op, bool is_unsigned) {
        ConstClass left_class = classify_const(left);
        ConstClass right_class = classify_const(right);
        if (is_int_dtype(left->dtype) && left == right) {
            switch (op) {
            case CmpOp::Equal:
            case CmpOp::LessEqual:
            case CmpOp::GreaterEqual: return bool_const(true);
            case CmpOp::Less:
            case CmpOp::Greater:
            case CmpOp::NotEqual: return bool_const(false);
            }
            SIMJIT_UNREACHABLE();
        }
        if (is_unsigned && is_int_dtype(left->dtype)) {
            if (op == CmpOp::GreaterEqual && right_class == ConstClass::Zero) return bool_const(true);
            if (op == CmpOp::Less && right_class == ConstClass::Zero) return bool_const(false);
            if (op == CmpOp::LessEqual && left_class == ConstClass::Zero) return bool_const(true);
            if (op == CmpOp::Greater && left_class == ConstClass::Zero) return bool_const(false);
        }
        return nullptr;
    }

    Step *binary_identity_peephole(Step *left, Step *right, PredicateBinaryOp op) {
        ConstClass left_class = classify_const(left);
        ConstClass right_class = classify_const(right);
        switch (op) {
        case PredicateBinaryOp::And:
            if (left == right) return left;
            if (left_class == ConstClass::BoolTrue) return right;
            if (right_class == ConstClass::BoolTrue) return left;
            if (left_class == ConstClass::BoolFalse) return left;
            if (right_class == ConstClass::BoolFalse) return right;
            return nullptr;
        case PredicateBinaryOp::Or:
            if (left == right) return left;
            if (left_class == ConstClass::BoolFalse) return right;
            if (right_class == ConstClass::BoolFalse) return left;
            if (left_class == ConstClass::BoolTrue) return left;
            if (right_class == ConstClass::BoolTrue) return right;
            return nullptr;
        case PredicateBinaryOp::Xor:
            if (left_class == ConstClass::BoolFalse) return right;
            if (right_class == ConstClass::BoolFalse) return left;
            if (left == right) return bool_const(false);
            return nullptr;
        case PredicateBinaryOp::AndNot:
            if (left == right) return bool_const(false);
            if (left_class == ConstClass::BoolFalse) return right;
            if (right_class == ConstClass::BoolFalse) return right;
            return nullptr;
        case PredicateBinaryOp::XNor:
            if (left == right) return bool_const(true);
            return nullptr;
        }
        SIMJIT_UNREACHABLE();
    }

    Step *arith_binary(Step *left, Step *right, ArithBinaryOp op, ArithBinaryOpFlags flags, Step **original) {
        SIMJIT_ASSERT(!bool(flags & ArithBinaryOpFlags::SafetyCheck));
        Step *result = *original = sm.arith_bin({op, left, right, flags}, left->dtype);
        if (bool(ctx->transformations & CodeTransformations::BinaryIdentityPeephole)) {
            if (Step *peepholed = binary_identity_peephole(left, right, op)) { return peepholed; }
        }
        if (op == ArithBinaryOp::Div || op == ArithBinaryOp::UDiv || op == ArithBinaryOp::Mod ||
            op == ArithBinaryOp::UMod) {
            // i128 division is lowered to a scalar compiler-runtime call. Keep
            // this before the simple-integer constant-division peephole.
            if (left->dtype == I128) {
                vectorization_hint = VectorizationHint::ScalarOnly;
            } else if (is_simple_int_dtype(left->dtype)) {
#if SIMJIT_USE_LIBDIVIDE
                {
                    bool vector_const_div = false;
                    if (bool(ctx->transformations & CodeTransformations::ConstDiv) && right->is(StepKind::Const) &&
                        (left->dtype == I16 || left->dtype == I32 || left->dtype == I64)) {
                        auto const_data = right->step_data<StepKind::Const>();
                        uint64_t rhs = (op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod)
                                           ? (uint64_t)const_data.as_signed()
                                           : const_data.as_unsigned();
                        vector_const_div = rhs != 0;
                    }
                    if (!vector_const_div) { vectorization_hint = VectorizationHint::ScalarOnly; }
                }
#else
                vectorization_hint = VectorizationHint::ScalarOnly;
#endif
            }
        }

        // Ad-hoc strength reduction for mul
        if (op == ArithBinaryOp::Mul && is_int_dtype(left->dtype)) {
            if (bool(ctx->transformations & CodeTransformations::MulConstPeephole)) {
                if (Step *peepholed = int_mul_peephole(left, right)) { return peepholed; }
            }
            // Change instruction if both of its arguments are 32-bit
            if (is_simple_int_dtype(left->dtype) && bool(ctx->transformations & CodeTransformations::MuldqInst)) {
                if (auto new_op = mul32_64_peephole(left, right)) {
                    return sm.arith_bin({*new_op, left, right}, left->dtype);
                }
            }
        }
        return result;
    }

    Step *arith_unary(Step *arg, ArithUnaryOp op) { return sm.arith_un({op, arg}, arg->dtype); }
    Step *predicate_not(Step *arg, Step **original) {
        *original = sm.predicate_not(arg, arg->dtype);
        if (bool(ctx->transformations & CodeTransformations::LogicalPeephole)) {
            if (Step *inverted = try_invert_cond(arg)) return inverted;
        }
        return *original;
    }
    Step *int_cast(Step *arg, ScalarDataType dtype, IntCastKind kind, Step **original) {
        *original = sm.int_cast({kind, arg}, dtype);
        if (bool(ctx->transformations & CodeTransformations::CastPeephole)) {
            if (Step *peepholed = int_cast_peephole(arg, dtype, kind)) { return peepholed; }
        }
        return *original;
    }
    Step *int_cast_for_float_decomposition(Step *arg, ScalarDataType dtype, bool is_unsigned) {
        if (arg->dtype == dtype) { return arg; }
        IntCastKind kind = scalar_dtype_size(arg->dtype) < scalar_dtype_size(dtype)
                               ? (is_unsigned ? IntCastKind::Zext : IntCastKind::Sext)
                               : IntCastKind::Trunc;
        Step *original = nullptr;
        return int_cast(arg, dtype, kind, &original);
    }
    Step *float_cast(Step *arg, ScalarDataType dtype, bool is_unsigned, Step **original) {
        *original = sm.float_cast({arg, is_unsigned}, dtype);
        if (!bool(ctx->transformations & CodeTransformations::CastDecomposition)) { return *original; }

        ScalarDataType from = arg->dtype;
        switch (classify_float_cast_rewrite(from, dtype)) {
        case FloatCastRewrite::Original: return *original;
        case FloatCastRewrite::FloatIdentity: return arg;
        case FloatCastRewrite::IntBridgeToTargetFloat: {
            Step *bridge = int_cast_for_float_decomposition(arg, int_dtype_for_float(dtype), is_unsigned);
            Step *result_original = nullptr;
            return float_cast(bridge, dtype, is_unsigned, &result_original);
        }
        case FloatCastRewrite::FloatBridgeFromSourceInt: {
            ScalarDataType same_width_float = float_dtype_for_int(from);
            Step *bridge_original = nullptr;
            Step *bridge = float_cast(arg, same_width_float, is_unsigned, &bridge_original);
            Step *result_original = nullptr;
            return float_cast(bridge, dtype, false, &result_original);
        }
        case FloatCastRewrite::FloatBridgeToTargetInt: {
            ScalarDataType same_width_float = float_dtype_for_int(dtype);
            Step *bridge_original = nullptr;
            Step *bridge = float_cast(arg, same_width_float, false, &bridge_original);
            Step *result_original = nullptr;
            return float_cast(bridge, dtype, is_unsigned, &result_original);
        }
        case FloatCastRewrite::FloatBridgeToSmallInt: {
            ScalarDataType same_width_float = float_dtype_for_int(dtype);
            ScalarDataType same_width_int = int_dtype_for_float(same_width_float);
            Step *float_original = nullptr;
            Step *float_bridge = float_cast(arg, same_width_float, false, &float_original);
            Step *int_original = nullptr;
            Step *int_bridge = float_cast(float_bridge, same_width_int, is_unsigned, &int_original);
            return int_cast_for_float_decomposition(int_bridge, dtype, is_unsigned);
        }
        }
        SIMJIT_UNREACHABLE();
    }
    Step *bitcast(Step *arg, ScalarDataType dtype) { return sm.bitcast(arg, dtype); }
    Step *fpclass(Step *arg, FpClass flags) { return sm.fpclass({flags, arg}, I1); }
    Step *cmp(Step *left, Step *right, CmpOp op, bool is_unsigned, Step **original) {
        SIMJIT_ASSERT(left->dtype == right->dtype);
        *original = sm.cmp({op, left, right, is_unsigned}, I1);
        if (bool(ctx->transformations & CodeTransformations::BinaryIdentityPeephole)) {
            if (Step *result = binary_identity_peephole(left, right, op, is_unsigned)) { return result; }
        }
        if (Step *result = cmp_peephole(left, right, op, is_unsigned)) { return result; }
        return *original;
    }
    Step *predicate_binary(Step *left, Step *right, PredicateBinaryOp op, Step **original) {
        SIMJIT_ASSERT(left->dtype == right->dtype);
        *original = sm.predicate_bin({left, right, op}, left->dtype);
        if (bool(ctx->transformations & CodeTransformations::BinaryIdentityPeephole)) {
            if (Step *result = binary_identity_peephole(left, right, op)) { return result; }
        }
        if (Step *result = predicate_binary_and_peephole(left, right, op)) { return result; }
        if (Step *result = predicate_binary_andnot_peephole(left, right, op)) { return result; }
        if (Step *result = predicate_binary_or_peephole(left, right, op)) { return result; }
        return *original;
    }
    Step *select(Step *cond, Step *truthy, Step *falsy, Step **original) {
        SIMJIT_ASSERT(falsy->dtype == truthy->dtype);
        SIMJIT_ASSERT(cond->dtype == I1);
        *original = sm.select({cond, truthy, falsy}, falsy->dtype);
        if (Step *result = select_peephole(cond, truthy, falsy)) { return result; }
        return *original;
    }
    Step *index(ScalarDataType dtype) { return sm.index({}, dtype); }

    Step *byte_permute(Step *arg, uint64_t permute) {
        SIMJIT_ASSERT(arg->dtype != I1 && arg->dtype != I128);
        return sm.permute({arg, permute, false}, arg->dtype);
    }

    Step *bit_permute(Step *arg, uint64_t permute) {
        SIMJIT_ASSERT(arg->dtype != I1 && arg->dtype != I128);
        return sm.permute({arg, permute, true}, arg->dtype);
    }

    Step *funnel_shift(Step *left, Step *right, Step *amount, FunnelShiftKind kind) {
        return sm.funnel_shift({left, right, amount, kind}, left->dtype);
    }

    Step *int_mul_peephole(Step *left, Step *right) {
        ScalarDataType dtype = left->dtype;

        Step *arg = nullptr;
        int64_t const_v;
        if (left->is(StepKind::Const)) {
            arg = right;
            const_v = left->step_data<StepKind::Const>().as_signed();
        } else if (right->is(StepKind::Const)) {
            arg = left;
            const_v = right->step_data<StepKind::Const>().as_signed();
        } else {
            return nullptr;
        }

        size_t bit_count = scalar_dtype_bits(dtype);
        ArithBinaryOpFlags shift_flags = dtype == I128 ? ArithBinaryOpFlags::ShiftWraparound : ArithBinaryOpFlags::No;

        auto multiply_pos = [&](uint64_t n, bool is_neg) -> Step * {
            SIMJIT_ASSERT(n != 0);
            // Optimizer should have handled it, but whatever
            if (n == 1) { return arg; }
            if (n == 2) { return sm.arith_bin({ArithBinaryOp::Add, arg, arg}, dtype); }
            if (has_single_bit(n)) {
                size_t count = nonzero_log2(n);
                SIMJIT_ASSERT(count < bit_count);
                return sm.arith_bin(
                    {ArithBinaryOp::ShiftLeftLogical, arg, sm.con(ConstData::i64((int64_t)count), dtype), shift_flags},
                    dtype);
            }
            // 2^n + 1 => (x << n) + x
            if (has_single_bit(n - 1)) {
                size_t count = nonzero_log2(n - 1);
                SIMJIT_ASSERT(count < bit_count);
                Step *tmp = sm.arith_bin(
                    {ArithBinaryOp::ShiftLeftLogical, arg, sm.con(ConstData::i64((int64_t)count), dtype), shift_flags},
                    dtype);
                return sm.arith_bin({ArithBinaryOp::Add, tmp, arg}, dtype);
            }
            // 2^n - 1 => (x << n) - x
            if (has_single_bit(n + 1)) {
                size_t count = nonzero_log2(n + 1);
                if (count >= bit_count) { return nullptr; }
                SIMJIT_ASSERT(count < bit_count);
                Step *tmp = sm.arith_bin(
                    {ArithBinaryOp::ShiftLeftLogical, arg, sm.con(ConstData::i64((int64_t)count), dtype), shift_flags},
                    dtype);
                return sm.arith_bin({ArithBinaryOp::Sub, tmp, arg}, dtype);
            }
            // 2^n + 2^m => (x << n) + (x << m)
            if (!is_neg && popcount(n) == 2) {
                size_t first_shift = nonzero_log2(n);
                size_t second_shift = nonzero_log2(n & ~(1llu << first_shift));
                SIMJIT_ASSERT(((1llu << first_shift) | (1llu << second_shift)) == n);
                SIMJIT_ASSERT(first_shift < bit_count);
                SIMJIT_ASSERT(second_shift < bit_count);
                Step *first_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)first_shift), dtype), shift_flags},
                                 dtype);
                Step *second_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)second_shift), dtype), shift_flags},
                                 dtype);
                return sm.arith_bin({ArithBinaryOp::Add, first_shift_s, second_shift_s}, dtype);
            }
            // 2^n - 2 => (x << n) - x - x
            if (has_single_bit(n + 2)) {
                size_t count = nonzero_log2(n + 2);
                if (count >= bit_count) { return nullptr; }
                SIMJIT_ASSERT(count < bit_count);
                Step *shifted =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, sm.con(ConstData::i64((int64_t)count), dtype),
                                  shift_flags},
                                 dtype);
                Step *tmp = sm.arith_bin({ArithBinaryOp::Add, arg, arg}, dtype);
                return sm.arith_bin({ArithBinaryOp::Sub, shifted, tmp}, dtype);
            }
            // 2^n + 2^m + 1 => (x << n) + (x << m) + x
            if ((dtype == I64 || dtype == I128) && popcount(n & ~1llu) == 2 && (n & 1llu) == 1llu) {
                size_t first_shift = nonzero_log2(n & ~1llu);
                size_t second_shift = nonzero_log2((n & ~1llu) & ~(1llu << first_shift));
                SIMJIT_ASSERT(((1llu << first_shift) | (1llu << second_shift) | 1llu) == n);
                SIMJIT_ASSERT(first_shift < bit_count);
                SIMJIT_ASSERT(second_shift < bit_count);
                Step *first_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)first_shift), dtype), shift_flags},
                                 dtype);
                Step *second_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)second_shift), dtype), shift_flags},
                                 dtype);
                Step *tmp = sm.arith_bin({ArithBinaryOp::Add, first_shift_s, second_shift_s}, dtype);
                return sm.arith_bin({ArithBinaryOp::Add, tmp, arg}, dtype);
            }
            // 2^n + 2^m + 2^p => (x << n) + (x << m) + (x << p)
            if ((dtype == I64 || dtype == I128) && popcount(n) == 3) {
                size_t first_shift = nonzero_log2(n);
                size_t second_shift = nonzero_log2(n & ~(1llu << first_shift));
                size_t third_shift = nonzero_log2(n & (~(1llu << first_shift) & ~(1llu << second_shift)));

                SIMJIT_ASSERT(((1llu << first_shift) | (1llu << second_shift) | (1llu << third_shift)) == n);
                SIMJIT_ASSERT(first_shift < bit_count);
                SIMJIT_ASSERT(second_shift < bit_count);
                SIMJIT_ASSERT(third_shift < bit_count);
                Step *first_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)first_shift), dtype), shift_flags},
                                 dtype);
                Step *second_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)second_shift), dtype), shift_flags},
                                 dtype);
                Step *first_sum_s = sm.arith_bin({ArithBinaryOp::Add, first_shift_s, second_shift_s}, dtype);
                Step *third_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg,
                                  sm.con(ConstData::i64((int64_t)third_shift), dtype), shift_flags},
                                 dtype);
                return sm.arith_bin({ArithBinaryOp::Add, first_sum_s, third_shift_s}, dtype);
            }
            return nullptr;
        };

        if (const_v == 0) { return sm.con(ConstData::i64(0), dtype); }
        if (const_v > 0) { return multiply_pos(const_v, false); }
        SIMJIT_ASSERT(const_v < 0);
        if (const_v == -1) { return sm.arith_un({ArithUnaryOp::Negate, arg}, left->dtype); }
        if (const_v == std::numeric_limits<int64_t>::min()) { return nullptr; }
        uint64_t mn = -const_v;
        if (Step *last = multiply_pos(mn, true)) {
            // If the last instruction was subtraction, we can swap operands places
            if (last->is(StepKind::ArithBinary) && last->step_data<StepKind::ArithBinary>().op == ArithBinaryOp::Sub) {
                auto &dat = last->step_data<StepKind::ArithBinary>();
                std::swap(dat.left, dat.right);
            } else {
                last = sm.arith_un({ArithUnaryOp::Negate, last}, dtype);
            }
            return last;
        }
        return nullptr;
    }

    static std::optional<ArithBinaryOp> mul32_64_peephole(Step *left, Step *right) {
        enum ChildTestResult : uint8_t {
            None,
            Const,
            Signed,
            Unsigned,
        };
        if (left->dtype != I64) return {};
        auto child_is_applicable = [](Step *expr) -> ChildTestResult {
            if (expr->is(StepKind::IntCast) && expr->step_data<StepKind::IntCast>().kind == IntCastKind::Sext)
                return Signed;
            if (expr->is(StepKind::IntCast) && expr->step_data<StepKind::IntCast>().kind == IntCastKind::Zext)
                return Unsigned;
            if (!expr->is(StepKind::Const)) { return {}; }
            int64_t data = expr->step_data<StepKind::Const>().as_signed();
            return ((int64_t)(int32_t)data == data) ? Const : None;
        };
        auto l = child_is_applicable(left);
        auto r = child_is_applicable(right);
        if (l == None || r == None) { return {}; }
        if (l == Const && r == Const) { return ArithBinaryOp::Mul64SE; }
        if ((l == Const || l == Signed) && (r == Const || r == Signed)) { return ArithBinaryOp::Mul64SE; }
        if ((l == Const || l == Unsigned) && (r == Const || r == Unsigned)) { return ArithBinaryOp::Mul64ZE; }
        return {};
    }

    Step *int_cast_peephole(const Step *arg, ScalarDataType dtype, IntCastKind kind) {
        if (!arg->is(StepKind::IntCast)) { return nullptr; }
        const auto &inner = arg->step_data<StepKind::IntCast>();

        ScalarDataType src_dtype = inner.arg->dtype;
        size_t src_size = scalar_dtype_size(src_dtype);
        size_t dst_size = scalar_dtype_size(dtype);

        if (kind == IntCastKind::Trunc) {
            if (inner.kind == IntCastKind::Trunc) {
                Step *original = nullptr;
                return int_cast(inner.arg, dtype, IntCastKind::Trunc, &original);
            }

            if (inner.kind == IntCastKind::Sext || inner.kind == IntCastKind::Zext) {
                if (dst_size == src_size) { return inner.arg; }
                Step *original = nullptr;
                if (dst_size < src_size) { return int_cast(inner.arg, dtype, IntCastKind::Trunc, &original); }
                return int_cast(inner.arg, dtype, inner.kind, &original);
            }
            SIMJIT_UNREACHABLE();
        }

        if (inner.kind == kind) {
            Step *original = nullptr;
            return int_cast(inner.arg, dtype, kind, &original);
        }
        if (kind == IntCastKind::Sext && inner.kind == IntCastKind::Zext) {
            Step *original = nullptr;
            return int_cast(inner.arg, dtype, IntCastKind::Zext, &original);
        }
        return nullptr;
    }

    Step *try_invert_cond(const Step *step) {
        if (step->is(StepKind::PredicateBinary)) {
            const auto &data = step->step_data<StepKind::PredicateBinary>();
            switch (data.op) {
            case PredicateBinaryOp::And: {
                Step *inverted_left = try_invert_cond(data.left);
                Step *inverted_right = try_invert_cond(data.right);
                if (inverted_left != nullptr && inverted_right != nullptr) {
                    return sm.predicate_bin({inverted_left, inverted_right, PredicateBinaryOp::Or}, step->dtype);
                }
                break;
            }
            case PredicateBinaryOp::Or: {
                Step *inverted_left = try_invert_cond(data.left);
                Step *inverted_right = try_invert_cond(data.right);
                if (inverted_left != nullptr && inverted_right != nullptr) {
                    return sm.predicate_bin({inverted_left, inverted_right, PredicateBinaryOp::And}, step->dtype);
                }
                break;
            }
            case PredicateBinaryOp::Xor:
                return sm.predicate_bin({data.left, data.right, PredicateBinaryOp::XNor}, step->dtype);
            case PredicateBinaryOp::AndNot: {
                // !(!a && b) -> a || !b
                if (Step *inverted_right = try_invert_cond(data.right); inverted_right) {
                    return sm.predicate_bin({inverted_right, data.left, PredicateBinaryOp::Or}, step->dtype);
                }
                break;
            }
            case PredicateBinaryOp::XNor:
                return sm.predicate_bin({data.left, data.right, PredicateBinaryOp::Xor}, step->dtype);
            }
        } else if (step->is(StepKind::Compare)) {
            const auto &data = step->step_data<StepKind::Compare>();
            if (is_float_dtype(data.left->dtype) && !is_equality_cmp_op(data.op)) return nullptr;
            CmpOp op = invert_cmp_op(data.op);
            if (Step *result = cmp_peephole(data.left, data.right, op, data.is_unsigned)) { return result; }
            return sm.cmp({op, data.left, data.right, data.is_unsigned}, step->dtype);
        } else if (step->is(StepKind::PredicateNot)) {
            Step *x = step->step_data<StepKind::PredicateNot>();
            return x;
        }
        return nullptr;
    }

    Step *cmp_peephole(Step *left, Step *right, CmpOp op, bool is_unsigned) {
        if (!bool(ctx->transformations & CodeTransformations::LogicalPeephole)) return nullptr;
        if (!is_equality_cmp_op(op)) return nullptr;
        if (rhs_cmp_preference(left) <= rhs_cmp_preference(right)) return nullptr;
        return sm.cmp({op, right, left, is_unsigned}, I1);
    }

    Step *predicate_binary_and_peephole(Step *lbin_left, Step *lbin_right, PredicateBinaryOp lbin_op) {
        if (lbin_op != PredicateBinaryOp::And) return nullptr;

        auto is_greater_or_equal_zero = [&](Step *e) -> Step * {
            if (!e->is(StepKind::Compare)) return nullptr;
            auto e_data = e->step_data<StepKind::Compare>();
            if (!is_simple_int_dtype(e_data.left->dtype)) return nullptr;
            CmpOp op = e_data.op;
            Step *left = e_data.left;
            Step *right = e_data.right;
            if (op == CmpOp::LessEqual && left->is(StepKind::Const) && left->step_data<StepKind::Const>().is_zero())
                return right;
            if (op == CmpOp::GreaterEqual && right->is(StepKind::Const) &&
                right->step_data<StepKind::Const>().is_zero())
                return left;
            return nullptr;
        };
        auto is_less_or_less_equal_something = [](Step *e) -> std::tuple<Step *, Step *, CmpOp> {
            if (!e->is(StepKind::Compare)) return {nullptr, nullptr, {}};
            auto e_data = e->step_data<StepKind::Compare>();
            if (!is_simple_int_dtype(e_data.left->dtype)) return {nullptr, nullptr, {}};
            CmpOp op = e_data.op;
            Step *left = e_data.left;
            Step *right = e_data.right;
            if (op == CmpOp::LessEqual || op == CmpOp::Less) return {left, right, op};
            if (op == CmpOp::GreaterEqual || op == CmpOp::Greater)
                return {right, left, op == CmpOp::GreaterEqual ? CmpOp::LessEqual : CmpOp::Less};
            return {nullptr, nullptr, {}};
        };
        auto is_greater_or_equal_something = [](Step *e) -> std::tuple<Step *, Step *, CmpOp> {
            if (!e->is(StepKind::Compare)) return {nullptr, nullptr, {}};
            auto e_data = e->step_data<StepKind::Compare>();
            if (!is_simple_int_dtype(e_data.left->dtype)) return {nullptr, nullptr, {}};
            CmpOp op = e_data.op;
            Step *left = e_data.left;
            Step *right = e_data.right;
            if (op == CmpOp::GreaterEqual || op == CmpOp::Greater) return {left, right, op};
            if (op == CmpOp::LessEqual || op == CmpOp::Less)
                return {right, left, op == CmpOp::LessEqual ? CmpOp::Greater : CmpOp::GreaterEqual};
            return {nullptr, nullptr, {}};
        };

        auto try_opt_between0 = [&](Step *left, Step *right) -> Step * {
            Step *compared1 = is_greater_or_equal_zero(left);
            auto [compared2, bound, new_op] = is_less_or_less_equal_something(right);
            if (!compared1 || !compared2) return nullptr;
            if (compared1 != compared2) return nullptr;
            // We need known positive RHS, otherwise original condition is always false, and rewrite won't be equivalent
            if (!bound->is(StepKind::Const) || bound->step_data<StepKind::Const>().as_signed() <= 0) return nullptr;

            return sm.cmp({new_op, compared1, bound, true}, I1);
        };
        auto try_opt_between = [&](Step *left, Step *right) -> Step * {
            auto [compared1, bound1, op1] = is_greater_or_equal_something(left);
            auto [compared2, bound2, op2] = is_less_or_less_equal_something(right);
            if (!compared1 || !compared2) return nullptr;
            if (compared1 != compared2) return nullptr;
            if (!bound1->is(StepKind::Const) || !bound2->is(StepKind::Const)) return nullptr;

            int64_t min_val = bound1->step_data<StepKind::Const>().as_signed();
            if (op1 == CmpOp::Greater) { ++min_val; }
            int64_t max_val = bound2->step_data<StepKind::Const>().as_signed();
            if (op2 == CmpOp::Less) { --max_val; }
            int64_t range = max_val - min_val;
            if (range <= 0) return nullptr;

            Step *min_val_step = sm.con(ConstData::i64(min_val), bound1->dtype);
            Step *range_step = sm.con(ConstData::i64(range), bound1->dtype);
            Step *adjusted = sm.arith_bin({ArithBinaryOp::Sub, compared1, min_val_step}, compared1->dtype);
            return sm.cmp({CmpOp::LessEqual, adjusted, range_step, true}, I1);
        };
        auto try_use_andnot = [&](Step *left, Step *right) -> Step * {
            // Two NOT in AND is very strange, but easy to handle
            if (left->is(StepKind::PredicateNot) && right->is(StepKind::PredicateNot)) {
                // Just de-morgan
                Step *left_step = left->step_data<StepKind::PredicateNot>();
                Step *right_step = right->step_data<StepKind::PredicateNot>();
                return sm.predicate_not(sm.predicate_bin({left_step, right_step, PredicateBinaryOp::Or}, I1), I1);
            }
            if (left->is(StepKind::PredicateNot)) {
                return sm.predicate_bin({left->step_data<StepKind::PredicateNot>(), right, PredicateBinaryOp::AndNot},
                                        I1);
            }
            if (right->is(StepKind::PredicateNot)) {
                return sm.predicate_bin({right->step_data<StepKind::PredicateNot>(), left, PredicateBinaryOp::AndNot},
                                        I1);
            }
            return nullptr;
        };

        if (bool(ctx->transformations & CodeTransformations::BetweenPeephole)) {
            if (Step *r = try_opt_between0(lbin_left, lbin_right)) return r;
            if (Step *r = try_opt_between0(lbin_right, lbin_left)) return r;
            if (Step *r = try_opt_between(lbin_left, lbin_right)) return r;
            if (Step *r = try_opt_between(lbin_right, lbin_left)) return r;
        }
        if (bool(ctx->transformations & CodeTransformations::LogicalPeephole)) {
            if (Step *r = try_use_andnot(lbin_left, lbin_right)) return r;
        }
        return nullptr;
    }

    Step *predicate_binary_or_peephole(Step *left, Step *right, PredicateBinaryOp op) {
        if (op != PredicateBinaryOp::Or) return nullptr;

        // !A | !B = !(A & B)
        if (left->is(StepKind::PredicateNot) && right->is(StepKind::PredicateNot)) {
            // Just de-morgan
            Step *left_step = left->step_data<StepKind::PredicateNot>();
            Step *right_step = right->step_data<StepKind::PredicateNot>();
            return sm.predicate_not(sm.predicate_bin({left_step, right_step, PredicateBinaryOp::And}, I1), I1);
        }

        // !A | B = !(A & !B)
        if (left->is(StepKind::PredicateNot)) {
            // Just de-morgan
            Step *left_step = left->step_data<StepKind::PredicateNot>();
            return sm.predicate_not(sm.predicate_bin({right, left_step, PredicateBinaryOp::AndNot}, I1), I1);
        }
        if (right->is(StepKind::PredicateNot)) {
            // Just de-morgan
            Step *right_step = right->step_data<StepKind::PredicateNot>();
            return sm.predicate_not(sm.predicate_bin({left, right_step, PredicateBinaryOp::AndNot}, I1), I1);
        }

        return nullptr;
    }

    Step *predicate_binary_andnot_peephole(const Step *left, Step *right, PredicateBinaryOp op) {
        if (op != PredicateBinaryOp::AndNot) return nullptr;
        if (!bool(ctx->transformations & CodeTransformations::LogicalPeephole)) return nullptr;

        if (left->is(StepKind::PredicateNot)) {
            Step *left_step = left->step_data<StepKind::PredicateNot>();
            if (right->is(StepKind::PredicateNot)) {
                return sm.predicate_bin(
                    {right->step_data<StepKind::PredicateNot>(), left_step, PredicateBinaryOp::AndNot}, I1);
            }
            return sm.predicate_bin({left_step, right, PredicateBinaryOp::And}, I1);
        }
        return nullptr;
    }

    Step *select_peephole(const Step *cond, Step *truthy, Step *falsy) {
        ConstClass cond_class = classify_const(cond);
        if (bool(ctx->transformations & CodeTransformations::ConstantConditionPeephole)) {
            if (cond_class == ConstClass::BoolTrue) return truthy;
            if (cond_class == ConstClass::BoolFalse) return falsy;
        }
        if (!bool(ctx->transformations & CodeTransformations::LogicalPeephole)) return nullptr;
        if (truthy == falsy) return truthy;
        if (!cond->is(StepKind::PredicateNot)) return nullptr;
        return sm.select({cond->step_data<StepKind::PredicateNot>(), falsy, truthy}, falsy->dtype);
    }

    void ensure_has_safety_check_arg() const {
        if (!safety_check_arg) {
            missing_output("Must call 'arg_safety_check' to support expressions with safety checking");
        }
    }
};

FunctionBuilder::FunctionBuilder(Context &context) {
    // This should be valid because Context can only be constructed with arena &
    SIMJIT_ASSERT(context.arena != nullptr);
    impl_ = new FunctionBuilderImpl(&context);
}

FunctionBuilder::FunctionBuilder(FunctionBuilder &&other) noexcept : impl_(other.impl_) {
    other.impl_ = nullptr;
}

FunctionBuilder &FunctionBuilder::operator=(FunctionBuilder &&other) noexcept {
    if (this != &other) {
        delete impl_;
        impl_ = other.impl_;
        other.impl_ = nullptr;
    }
    return *this;
}

FunctionBuilder::~FunctionBuilder() noexcept {
    delete impl_;
}

Argument FunctionBuilder::arg(ScalarDataType dtype) {
    size_t max_args = impl_->ctx->build_limits.max_argument_count;
    if (max_args > MaxFunctionArgumentCount) { max_args = MaxFunctionArgumentCount; }
    if (impl_->args.size() >= max_args) {
        unsupported_limit("HIR has too many arguments (%zu > %zu)", impl_->args.size() + 1, max_args);
    }
    impl_->args.push_back(ArgumentDecl{dtype, impl_->args.size()});
    return {impl_->args.back().idx};
}

static ScalarDataType interleaved_carrier_dtype(ScalarDataType half_dtype) {
    switch (half_dtype) {
    case I8: return I16;
    case I16: return I32;
    case I32:
    case F32: return I64;
    case I64:
    case F64: return I128;
    case I1:
    case I128: break;
    }
    invalid_type("invalid interleaved half type %s", show_scalar_dtype(half_dtype));
}

void FunctionBuilder::mark_arg_interleave(Argument argument, ScalarDataType half_dtype) {
    ArgumentDecl &decl = impl_->get_arg(argument.idx_);
    ScalarDataType carrier_dtype = interleaved_carrier_dtype(half_dtype);
    if (decl.dtype != carrier_dtype) {
        invalid_type("interleaved half type %s requires %s carrier, got %s", show_scalar_dtype(half_dtype),
                     show_scalar_dtype(carrier_dtype), show_scalar_dtype(decl.dtype));
    }
    if (decl.interleaved && decl.interleaved_half_dtype != half_dtype) {
        invalid_type("argument %zu is already interleaved as %s", argument.idx_,
                     show_scalar_dtype(decl.interleaved_half_dtype));
    }
    decl.interleaved = true;
    decl.interleaved_half_dtype = half_dtype;
}

Argument FunctionBuilder::arg_interleaved(ScalarDataType half_dtype) {
    Argument result = arg(interleaved_carrier_dtype(half_dtype));
    mark_arg_interleave(result, half_dtype);
    return result;
}

static void check_store_dst(const ArgumentDecl &decl, const Step *arg, StepKind kind) {
    if (decl.dtype != arg->dtype) {
        invalid_type("Step %s argument and destination have different types %s %s", show_step_kind(kind),
                     show_scalar_dtype(arg->dtype), show_scalar_dtype(decl.dtype));
    }
}

struct HirComplexityStats {
    size_t roots = 0;
    size_t live_steps = 0;
};

static void check_hir_root_limit(FunctionBuilderImpl *impl) {
    size_t root_count = impl->step_roots.size();
    size_t max_roots = impl->ctx->build_limits.max_hir_roots;
    if (root_count > max_roots) { unsupported_limit("HIR has too many roots (%zu > %zu)", root_count, max_roots); }
}

static HirComplexityStats check_hir_complexity(FunctionBuilderImpl *impl) {
    std::vector<uint8_t> state(impl->sm.max_id(), 0);
    HirComplexityStats stats{impl->step_roots.size(), 0};
    size_t max_live_steps = impl->ctx->build_limits.max_hir_live_steps;

    for (Step *root : impl->step_roots) {
        traverse_steps_postorder_unique(root, state, [&](Step *) {
            ++stats.live_steps;
            if (stats.live_steps > max_live_steps) {
                unsupported_limit("HIR has too many live steps (%zu > %zu)", stats.live_steps, max_live_steps);
            }
        });
    }
    return stats;
}

struct I128RewriteState {
    struct I128Pair {
        Step *low;
        Step *high;
    };
    struct RewrittenStep {
        enum class Kind : uint8_t {
            Scalar,
            Pair,
        };

        Kind kind = Kind::Scalar;
        union {
            Step *scalar;
            I128Pair pair;
        };

        RewrittenStep() : scalar(nullptr) {}

        static RewrittenStep make_scalar(Step *step) {
            RewrittenStep result;
            result.scalar = step;
            return result;
        }

        static RewrittenStep make_pair(I128Pair value) {
            RewrittenStep result;
            result.kind = Kind::Pair;
            result.pair = value;
            return result;
        }

        bool is_pair() const { return kind == Kind::Pair; }
    };
    struct ScalarChildRewriter {
        I128RewriteState &state;
        Step *consumer;

        void operator()(Step *&child) const { child = state.rewrite_scalar(child, consumer); }
    };

    Function *func;
    StepMaker &sm;
    size_t original_step_count;
    std::vector<RewrittenStep> rewritten;
    std::vector<Step *> additional_roots;
    ArenaBitmap rewritten_steps;
    Step *zero = nullptr;
    Step *one = nullptr;
    Step *ones = nullptr;
    Step *mask32 = nullptr;
    Step *sixty3 = nullptr;
    Step *sixty4 = nullptr;
    Step *sixty127 = nullptr;

    I128RewriteState(FunctionBuilderImpl *impl, Function *func)
        : func(func), sm(impl->sm), original_step_count(sm.max_id()), rewritten(original_step_count),
          rewritten_steps(ArenaBitmap::create(impl->ctx->arena, original_step_count)) {}

    void mark_i128_argument(ArgumentIdx idx) {
        ArgumentDecl &arg = func->args[idx];
        if (arg.dtype != I128) { unsupported("ordinary i128 access requires an i128 {i64,i64} carrier"); }
        if (!arg.interleaved) {
            arg.interleaved = true;
            arg.interleaved_half_dtype = I64;
        }
        if (arg.interleaved_half_dtype != I64) {
            unsupported("ordinary i128 access requires an i128 {i64,i64} carrier");
        }
    }

    template <typename Make> static Step *cached_step(Step **cache, Make make) {
        if (*cache == nullptr) { *cache = make(); }
        return *cache;
    }

    Step *rewrite_scalar(Step *child, const Step *consumer) {
        RewrittenStep rewritten_child = rewrite(child);
        if (rewritten_child.is_pair()) {
            unsupported("i128 value is not supported by %s", show_step_kind(consumer->kind));
        }
        return rewritten_child.scalar;
    }

    I128Pair rewrite_pair_child(Step *child, const Step *consumer) {
        RewrittenStep rewritten_child = rewrite(child);
        (void)consumer;
        SIMJIT_ASSERT(rewritten_child.is_pair());
        return rewritten_child.pair;
    }

    Step *get_i64_one() {
        return cached_step(&one, [this] { return sm.con(ConstData::i64(1), I64); });
    }

    Step *get_i64_ones() {
        return cached_step(&ones, [this] { return sm.con(ConstData::u64(scalar_dtype_umax(I64)), I64); });
    }

    Step *get_i64_mask32() {
        return cached_step(&mask32, [this] { return sm.con(ConstData::u64(0xffffffffull), I64); });
    }

    Step *get_i64_zero() {
        return cached_step(&zero, [this] { return sm.con(ConstData::i64(0), I64); });
    }

    Step *get_i64_63() {
        return cached_step(&sixty3, [this] { return sm.con(ConstData::i64(63), I64); });
    }

    Step *get_i64_64() {
        return cached_step(&sixty4, [this] { return sm.con(ConstData::i64(64), I64); });
    }

    Step *get_i64_127() {
        return cached_step(&sixty127, [this] { return sm.con(ConstData::i64(127), I64); });
    }

    RewrittenStep rewrite(Step *step) {
        SIMJIT_ASSERT(step->id < original_step_count);
        RewrittenStep &result = rewritten[step->id];
        if (rewritten_steps.get(step->id)) { return result; }
        rewritten_steps.set(step->id);
        result = rewrite_internal(step);
        return result;
    }

    I128Pair negate_i128(I128Pair arg) {
        // lo = 0-al; hi = ~ah + (al == 0)
        Step *low = sm.arith_bin({ArithBinaryOp::Sub, get_i64_zero(), arg.low}, I64);
        Step *high = sm.arith_un({ArithUnaryOp::Not, arg.high}, I64);
        Step *carry_i64 =
            sm.select({sm.cmp({CmpOp::Equal, arg.low, get_i64_zero()}, I1), get_i64_one(), get_i64_zero()}, I64);
        high = sm.arith_bin({ArithBinaryOp::Add, high, carry_i64}, I64);
        return {low, high};
    }

    I128Pair add_i128(I128Pair left, I128Pair right) {
        Step *low = sm.arith_bin({ArithBinaryOp::Add, left.low, right.low}, I64);
        Step *high = sm.arith_bin({ArithBinaryOp::Add, left.high, right.high}, I64);
        Step *carry = sm.cmp({CmpOp::Less, low, left.low, true}, I1);
        Step *carry_i64 = sm.select({carry, get_i64_one(), get_i64_zero()}, I64);
        high = sm.arith_bin({ArithBinaryOp::Add, high, carry_i64}, I64);
        return {low, high};
    }

    I128Pair sub_i128(I128Pair left, I128Pair right) {
        Step *low = sm.arith_bin({ArithBinaryOp::Sub, left.low, right.low}, I64);
        Step *high = sm.arith_bin({ArithBinaryOp::Sub, left.high, right.high}, I64);
        Step *borrow = sm.cmp({CmpOp::Less, left.low, right.low, true}, I1);
        Step *borrow_i64 = sm.select({borrow, get_i64_one(), get_i64_zero()}, I64);
        high = sm.arith_bin({ArithBinaryOp::Sub, high, borrow_i64}, I64);
        return {low, high};
    }

    I128Pair safe_i128_divisor(I128Pair left, I128Pair right, ArithBinaryOp op) {
        Step *low_zero = sm.cmp({CmpOp::Equal, right.low, get_i64_zero()}, I1);
        Step *high_zero = sm.cmp({CmpOp::Equal, right.high, get_i64_zero()}, I1);
        Step *invalid = sm.predicate_bin({low_zero, high_zero, PredicateBinaryOp::And}, I1);
        if (op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod) {
            Step *left_low_zero = sm.cmp({CmpOp::Equal, left.low, get_i64_zero()}, I1);
            Step *left_high_min =
                sm.cmp({CmpOp::Equal, left.high, sm.con(ConstData::i64(std::numeric_limits<int64_t>::min()), I64)}, I1);
            Step *right_low_ones = sm.cmp({CmpOp::Equal, right.low, get_i64_ones()}, I1);
            Step *right_high_ones = sm.cmp({CmpOp::Equal, right.high, get_i64_ones()}, I1);
            Step *left_is_min = sm.predicate_bin({left_low_zero, left_high_min, PredicateBinaryOp::And}, I1);
            Step *right_is_minus_one = sm.predicate_bin({right_low_ones, right_high_ones, PredicateBinaryOp::And}, I1);
            Step *overflow = sm.predicate_bin({left_is_min, right_is_minus_one, PredicateBinaryOp::And}, I1);
            invalid = sm.predicate_bin({invalid, overflow, PredicateBinaryOp::Or}, I1);
        }
        return select_i128(invalid, {get_i64_one(), get_i64_zero()}, right);
    }

    I128Pair i128_div(I128Pair left, I128Pair right, ArithBinaryOp op) {
        Step *low = sm.i128_div({op, left.low, left.high, right.low, right.high, nullptr, false}, I64);
        Step *high = sm.i128_div({op, left.low, left.high, right.low, right.high, low, true}, I64);
        low->step_data<StepKind::I128Div>().partner = high;
        return {low, high};
    }

    I128Pair checked_i128_pair(CheckedOpKind low_kind, CheckedOpKind high_kind, I128Pair result, Step *mask,
                               I128Pair left, I128Pair right = {}) {
        Step *low =
            sm.checked_op({low_kind, result.low, mask, nullptr, left.low, left.high, right.low, right.high}, I64);
        Step *high =
            sm.checked_op({high_kind, result.high, mask, low, left.low, left.high, right.low, right.high}, I64);
        low->step_data<StepKind::CheckedOp>().partner = high;
        return {low, high};
    }

    struct ShiftCount {
        Step *within_word;
        Step *at_least_word;
    };

    ShiftCount normalize_shift_count(Step *count) {
        Step *normalized = sm.arith_bin({ArithBinaryOp::And, count, get_i64_127()}, I64);
        Step *within_word = sm.arith_bin({ArithBinaryOp::And, normalized, get_i64_63()}, I64);
        Step *at_least_word = sm.cmp({CmpOp::GreaterEqual, normalized, get_i64_64(), true}, I1);
        return {within_word, at_least_word};
    }

    I128Pair shift_i128(ArithBinaryOp op, I128Pair arg, Step *count) {
        func->special_ops |= SpecialOp::FunnelShift;
        ShiftCount normalized = normalize_shift_count(count);
        Step *k = normalized.within_word;
        I128Pair below{};
        I128Pair above{};
        switch (op) {
        case ArithBinaryOp::ShiftLeftLogical:
            below = {sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg.low, k}, I64),
                     sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Left}, I64)};
            above = {get_i64_zero(), sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg.low, k}, I64)};
            break;
        case ArithBinaryOp::ShiftRightLogical:
            below = {sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Right}, I64),
                     sm.arith_bin({ArithBinaryOp::ShiftRightLogical, arg.high, k}, I64)};
            above = {sm.arith_bin({ArithBinaryOp::ShiftRightLogical, arg.high, k}, I64), get_i64_zero()};
            break;
        case ArithBinaryOp::ShiftRightArith:
            below = {sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Right}, I64),
                     sm.arith_bin({ArithBinaryOp::ShiftRightArith, arg.high, k}, I64)};
            above = {sm.arith_bin({ArithBinaryOp::ShiftRightArith, arg.high, k}, I64),
                     sm.arith_bin({ArithBinaryOp::ShiftRightArith, arg.high, get_i64_63()}, I64)};
            break;
        case ArithBinaryOp::RotateLeft:
            below = {sm.funnel_shift({arg.low, arg.high, k, FunnelShiftKind::Left}, I64),
                     sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Left}, I64)};
            above = {sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Left}, I64),
                     sm.funnel_shift({arg.low, arg.high, k, FunnelShiftKind::Left}, I64)};
            break;
        case ArithBinaryOp::RotateRight:
            below = {sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Right}, I64),
                     sm.funnel_shift({arg.low, arg.high, k, FunnelShiftKind::Right}, I64)};
            above = {sm.funnel_shift({arg.low, arg.high, k, FunnelShiftKind::Right}, I64),
                     sm.funnel_shift({arg.high, arg.low, k, FunnelShiftKind::Right}, I64)};
            break;
        default: SIMJIT_UNREACHABLE();
        }
        return select_i128(normalized.at_least_word, above, below);
    }

    I128Pair shift_i128_const(ArithBinaryOp op, I128Pair arg, size_t count) {
        count &= 127;
        if (count == 0) { return arg; }
        size_t k = count & 63;
        switch (op) {
        case ArithBinaryOp::ShiftLeftLogical: {
            if (count < 64) {
                func->special_ops |= SpecialOp::FunnelShift;
                Step *k_step = sm.con(ConstData::i64((int64_t)k), I64);
                return {sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg.low, k_step}, I64),
                        sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Left}, I64)};
            }
            if (k == 0) { return {get_i64_zero(), arg.low}; }
            return {
                get_i64_zero(),
                sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg.low, sm.con(ConstData::i64((int64_t)k), I64)}, I64)};
        }
        case ArithBinaryOp::ShiftRightLogical: {
            if (count < 64) {
                func->special_ops |= SpecialOp::FunnelShift;
                Step *k_step = sm.con(ConstData::i64((int64_t)k), I64);
                return {sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Right}, I64),
                        sm.arith_bin({ArithBinaryOp::ShiftRightLogical, arg.high, k_step}, I64)};
            }
            if (k == 0) { return {arg.high, get_i64_zero()}; }
            return {sm.arith_bin({ArithBinaryOp::ShiftRightLogical, arg.high, sm.con(ConstData::i64((int64_t)k), I64)},
                                 I64),
                    get_i64_zero()};
        }
        case ArithBinaryOp::ShiftRightArith: {
            if (count < 64) {
                func->special_ops |= SpecialOp::FunnelShift;
                Step *k_step = sm.con(ConstData::i64((int64_t)k), I64);
                return {sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Right}, I64),
                        sm.arith_bin({ArithBinaryOp::ShiftRightArith, arg.high, k_step}, I64)};
            }
            Step *sign = sm.arith_bin({ArithBinaryOp::ShiftRightArith, arg.high, get_i64_63()}, I64);
            if (k == 0) { return {arg.high, sign}; }
            return {
                sm.arith_bin({ArithBinaryOp::ShiftRightArith, arg.high, sm.con(ConstData::i64((int64_t)k), I64)}, I64),
                sign};
        }
        case ArithBinaryOp::RotateLeft: {
            if (count == 64) { return {arg.high, arg.low}; }
            func->special_ops |= SpecialOp::FunnelShift;
            Step *k_step = sm.con(ConstData::i64((int64_t)k), I64);
            if (count < 64) {
                return {sm.funnel_shift({arg.low, arg.high, k_step, FunnelShiftKind::Left}, I64),
                        sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Left}, I64)};
            }
            return {sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Left}, I64),
                    sm.funnel_shift({arg.low, arg.high, k_step, FunnelShiftKind::Left}, I64)};
        }
        case ArithBinaryOp::RotateRight: {
            if (count == 64) { return {arg.high, arg.low}; }
            func->special_ops |= SpecialOp::FunnelShift;
            Step *k_step = sm.con(ConstData::i64((int64_t)k), I64);
            if (count < 64) {
                return {sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Right}, I64),
                        sm.funnel_shift({arg.low, arg.high, k_step, FunnelShiftKind::Right}, I64)};
            }
            return {sm.funnel_shift({arg.low, arg.high, k_step, FunnelShiftKind::Right}, I64),
                    sm.funnel_shift({arg.high, arg.low, k_step, FunnelShiftKind::Right}, I64)};
        }
        default: SIMJIT_UNREACHABLE();
        }
        SIMJIT_UNREACHABLE();
    }

    struct U64Parts {
        Step *low;
        Step *high;
    };

    U64Parts split_u64(Step *value) {
        return {value, sm.arith_bin({ArithBinaryOp::ShiftRightLogical, value, sm.con(ConstData::i64(32), I64)}, I64)};
    }

    I128Pair umul64_wide(U64Parts left, U64Parts right) {
        // Mul64ZE already selects each input's low 32 bits. Keep the low limbs unmasked so the backend can
        // lower them directly, and share the shifted high limbs with the cross products below.
        Step *x0y0 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.low, right.low}, I64);
        Step *x0y1 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.low, right.high}, I64);
        Step *x1y0 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.high, right.low}, I64);
        Step *x1y1 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.high, right.high}, I64);
        Step *middle = sm.arith_bin({ArithBinaryOp::ShiftRightLogical, x0y0, sm.con(ConstData::i64(32), I64)}, I64);
        middle = sm.arith_bin(
            {ArithBinaryOp::Add, middle, sm.arith_bin({ArithBinaryOp::And, x0y1, get_i64_mask32()}, I64)}, I64);
        middle = sm.arith_bin(
            {ArithBinaryOp::Add, middle, sm.arith_bin({ArithBinaryOp::And, x1y0, get_i64_mask32()}, I64)}, I64);
        Step *low =
            sm.arith_bin({ArithBinaryOp::Add, x0y0,
                          sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, x0y1, sm.con(ConstData::i64(32), I64)}, I64)},
                         I64);
        low =
            sm.arith_bin({ArithBinaryOp::Add, low,
                          sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, x1y0, sm.con(ConstData::i64(32), I64)}, I64)},
                         I64);
        Step *high =
            sm.arith_bin({ArithBinaryOp::Add, x1y1,
                          sm.arith_bin({ArithBinaryOp::ShiftRightLogical, x0y1, sm.con(ConstData::i64(32), I64)}, I64)},
                         I64);
        high =
            sm.arith_bin({ArithBinaryOp::Add, high,
                          sm.arith_bin({ArithBinaryOp::ShiftRightLogical, x1y0, sm.con(ConstData::i64(32), I64)}, I64)},
                         I64);
        high = sm.arith_bin(
            {ArithBinaryOp::Add, high,
             sm.arith_bin({ArithBinaryOp::ShiftRightLogical, middle, sm.con(ConstData::i64(32), I64)}, I64)},
            I64);
        return {low, high};
    }

    Step *umul64_low(U64Parts left, U64Parts right) {
        Step *x0y0 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.low, right.low}, I64);
        Step *x0y1 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.low, right.high}, I64);
        Step *x1y0 = sm.arith_bin({ArithBinaryOp::Mul64ZE, left.high, right.low}, I64);
        Step *low =
            sm.arith_bin({ArithBinaryOp::Add, x0y0,
                          sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, x0y1, sm.con(ConstData::i64(32), I64)}, I64)},
                         I64);
        return sm.arith_bin(
            {ArithBinaryOp::Add, low,
             sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, x1y0, sm.con(ConstData::i64(32), I64)}, I64)},
            I64);
    }

    I128Pair multiply_i128(I128Pair left, I128Pair right) {
        U64Parts left_low = split_u64(left.low);
        U64Parts left_high = split_u64(left.high);
        U64Parts right_low = split_u64(right.low);
        U64Parts right_high = split_u64(right.high);
        I128Pair low_product = umul64_wide(left_low, right_low);
        Step *high = sm.arith_bin({ArithBinaryOp::Add, low_product.high, umul64_low(left_low, right_high)}, I64);
        high = sm.arith_bin({ArithBinaryOp::Add, high, umul64_low(left_high, right_low)}, I64);
        return {low_product.low, high};
    }

    I128Pair select_i128(Step *cond, I128Pair truthy, I128Pair falsy) {
        Step *low = sm.select({cond, truthy.low, falsy.low}, I64);
        Step *high = sm.select({cond, truthy.high, falsy.high}, I64);
        return {low, high};
    }

    Step *cmp_i128(I128Pair left, I128Pair right, CmpOp op, bool is_unsigned) {
        if (op == CmpOp::Equal) {
            Step *lo_cond = sm.cmp({op, left.low, right.low}, I1);
            Step *hi_cond = sm.cmp({op, left.high, right.high}, I1);
            return sm.predicate_bin({lo_cond, hi_cond, PredicateBinaryOp::And}, I1);
        }
        if (op == CmpOp::NotEqual) {
            Step *lo_cond = sm.cmp({op, left.low, right.low}, I1);
            Step *hi_cond = sm.cmp({op, left.high, right.high}, I1);
            return sm.predicate_bin({lo_cond, hi_cond, PredicateBinaryOp::Or}, I1);
        }

        switch (op) {
        case CmpOp::Less:
        case CmpOp::Greater: {
            // (ah < bh) | ((ah == bh) & (al <u bl))
            Step *hi_cond = sm.cmp({op, left.high, right.high, is_unsigned}, I1);
            Step *lo_cond = sm.cmp({op, left.low, right.low, true}, I1);
            Step *hi_eq = sm.cmp({CmpOp::Equal, left.high, right.high}, I1);
            Step *lo = sm.predicate_bin({hi_eq, lo_cond, PredicateBinaryOp::And}, I1);
            return sm.predicate_bin({hi_cond, lo, PredicateBinaryOp::Or}, I1);
        }
        case CmpOp::LessEqual:
        case CmpOp::GreaterEqual: {
            CmpOp strict_op = op == CmpOp::LessEqual ? CmpOp::Less : CmpOp::Greater;
            Step *hi_strict = sm.cmp({strict_op, left.high, right.high, is_unsigned}, I1);
            Step *lo_cond = sm.cmp({op, left.low, right.low, true}, I1);
            Step *hi_eq = sm.cmp({CmpOp::Equal, left.high, right.high}, I1);
            Step *lo = sm.predicate_bin({hi_eq, lo_cond, PredicateBinaryOp::And}, I1);
            return sm.predicate_bin({hi_strict, lo, PredicateBinaryOp::Or}, I1);
        }
        default: SIMJIT_UNREACHABLE();
        }
        SIMJIT_UNREACHABLE();
    }

    Step *i64_to_float_for_i128_rewrite(Step *arg, ScalarDataType dtype, bool is_unsigned) {
        if (dtype == F64) { return sm.float_cast({arg, is_unsigned}, F64); }
        Step *wide = sm.float_cast({arg, is_unsigned}, F64);
        return sm.float_cast({wide, false}, F32);
    }

    Step *float_to_i64_for_i128_rewrite(Step *arg, bool is_unsigned) {
        if (arg->dtype == F64) { return sm.float_cast({arg, is_unsigned}, I64); }
        Step *wide = sm.float_cast({arg, false}, F64);
        return sm.float_cast({wide, is_unsigned}, I64);
    }

    Step *i128_to_float(I128Pair arg, ScalarDataType dtype, bool is_unsigned) {
        I128Pair magnitude = arg;
        Step *negative = nullptr;
        if (!is_unsigned) {
            negative = sm.cmp({CmpOp::Less, arg.high, get_i64_zero()}, I1);
            magnitude = select_i128(negative, negate_i128(arg), arg);
        }

        Step *scale = dtype == F32 ? sm.con(ConstData::f32(0x1p64f), F32) : sm.con(ConstData::f64(0x1p64), F64);
        Step *high = i64_to_float_for_i128_rewrite(magnitude.high, dtype, true);
        Step *low = i64_to_float_for_i128_rewrite(magnitude.low, dtype, true);
        Step *result =
            sm.arith_bin({ArithBinaryOp::Add, sm.arith_bin({ArithBinaryOp::Mul, high, scale}, dtype), low}, dtype);
        if (negative != nullptr) {
            result = sm.select({negative, sm.arith_un({ArithUnaryOp::Negate, result}, dtype), result}, dtype);
        }
        return result;
    }

    I128Pair float_to_i128(Step *arg, bool is_unsigned) {
        ScalarDataType dtype = arg->dtype;
        Step *magnitude = arg;
        Step *negative = nullptr;
        if (!is_unsigned) {
            Step *float_zero = dtype == F32 ? sm.con(ConstData::f32(0.0f), F32) : sm.con(ConstData::f64(0.0), F64);
            negative = sm.cmp({CmpOp::Less, arg, float_zero}, I1);
            magnitude = sm.arith_un({ArithUnaryOp::Abs, arg}, dtype);
        }

        Step *scale = dtype == F32 ? sm.con(ConstData::f32(0x1p64f), F32) : sm.con(ConstData::f64(0x1p64), F64);
        Step *inverse_scale =
            dtype == F32 ? sm.con(ConstData::f32(0x1p-64f), F32) : sm.con(ConstData::f64(0x1p-64), F64);
        Step *high =
            float_to_i64_for_i128_rewrite(sm.arith_bin({ArithBinaryOp::Mul, magnitude, inverse_scale}, dtype), true);
        Step *high_as_float = i64_to_float_for_i128_rewrite(high, dtype, true);
        Step *low = float_to_i64_for_i128_rewrite(
            sm.arith_bin(
                {ArithBinaryOp::Sub, magnitude, sm.arith_bin({ArithBinaryOp::Mul, high_as_float, scale}, dtype)},
                dtype),
            true);
        I128Pair result{low, high};
        if (negative != nullptr) { result = select_i128(negative, negate_i128(result), result); }
        return result;
    }

    RewrittenStep rewrite_internal(Step *step) {
        SIMJIT_ASSERT(step->id < original_step_count);

        switch (step->kind) {
            SIMJIT_MATCH (StepKind::Const) {
                if (step->dtype != I128) break;
                SIMJIT_ASSERT(data.dtype == I64);
                int64_t val = data.as_i64();
                Step *lo = sm.con(data.retag(I64), I64);
                Step *hi = nullptr;
                if (val < 0) {
                    hi = get_i64_ones();
                } else {
                    hi = get_i64_zero();
                }
                return RewrittenStep::make_pair({lo, hi});
            }
            SIMJIT_MATCH (StepKind::Load) {
                if (step->dtype != I128) break;
                mark_i128_argument(data.idx);
                return RewrittenStep::make_pair({sm.load_deinterleave({data.idx, data.kind, 2, 0}, I64),
                                                 sm.load_deinterleave({data.idx, data.kind, 2, 1}, I64)});
            }
            SIMJIT_MATCH (StepKind::LoadSplat) {
                if (step->dtype != I128) break;
                if (data.byte_offset > std::numeric_limits<size_t>::max() - sizeof(uint64_t)) {
                    invalid_input("i128 load_splat byte offset overflows");
                }
                mark_i128_argument(data.idx);
                return RewrittenStep::make_pair(
                    {sm.load_splat({data.idx, data.kind, data.byte_offset}, I64),
                     sm.load_splat({data.idx, data.kind, data.byte_offset + sizeof(uint64_t)}, I64)});
            }
            SIMJIT_MATCH (StepKind::Gather) {
                if (step->dtype != I128) break;
                mark_i128_argument(data.data);
                Step *idx = rewrite_scalar(data.idx, step);
                Step *twice_idx = sm.arith_bin({ArithBinaryOp::Add, idx, idx}, idx->dtype);
                Step *odd_idx = sm.arith_bin(
                    {ArithBinaryOp::Add, twice_idx, sm.con(ConstData::i64(1).retag(idx->dtype), idx->dtype)},
                    idx->dtype);
                return RewrittenStep::make_pair(
                    {sm.gather({twice_idx, data.data}, I64), sm.gather({odd_idx, data.data}, I64)});
            }

            SIMJIT_MATCH (StepKind::ArithBinary) {
                if (step->dtype != I128) break;
                bool is_shift = data.op == ArithBinaryOp::ShiftRightArith ||
                                data.op == ArithBinaryOp::ShiftRightLogical ||
                                data.op == ArithBinaryOp::ShiftLeftLogical || data.op == ArithBinaryOp::RotateLeft ||
                                data.op == ArithBinaryOp::RotateRight;
                bool is_division = data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::UDiv ||
                                   data.op == ArithBinaryOp::Mod || data.op == ArithBinaryOp::UMod;
                if ((is_shift && data.flags != ArithBinaryOpFlags::ShiftWraparound) ||
                    (!is_shift && !is_division && data.flags != ArithBinaryOpFlags::No) ||
                    (is_division && data.flags != ArithBinaryOpFlags::No &&
                     data.flags != ArithBinaryOpFlags::SafeDivision)) {
                    unsupported("i128 rewrite does not support flags for %s", show_arith_binary_op(data.op));
                }
                I128Pair left = rewrite_pair_child(data.left, step);
                if (is_shift && data.right->is(StepKind::Const)) {
                    const ConstData &count = data.right->step_data<StepKind::Const>();
                    SIMJIT_ASSERT(data.right->dtype == I128);
                    return RewrittenStep::make_pair(shift_i128_const(data.op, left, count.as_unsigned()));
                }
                I128Pair right = rewrite_pair_child(data.right, step);
                switch (data.op) {
                case ArithBinaryOp::Add: return RewrittenStep::make_pair(add_i128(left, right));
                case ArithBinaryOp::Sub: return RewrittenStep::make_pair(sub_i128(left, right));
                case ArithBinaryOp::Mul: return RewrittenStep::make_pair(multiply_i128(left, right));
                case ArithBinaryOp::Div:
                case ArithBinaryOp::UDiv:
                case ArithBinaryOp::Mod:
                case ArithBinaryOp::UMod:
                    if (data.flags == ArithBinaryOpFlags::SafeDivision) {
                        right = safe_i128_divisor(left, right, data.op);
                    }
                    return RewrittenStep::make_pair(i128_div(left, right, data.op));
                case ArithBinaryOp::ShiftRightArith:
                case ArithBinaryOp::ShiftRightLogical:
                case ArithBinaryOp::ShiftLeftLogical:
                case ArithBinaryOp::RotateLeft:
                case ArithBinaryOp::RotateRight: return RewrittenStep::make_pair(shift_i128(data.op, left, right.low));
                case ArithBinaryOp::And:
                case ArithBinaryOp::Or:
                case ArithBinaryOp::Xor:
                case ArithBinaryOp::AndNot: {
                    ArithBinaryOp op = data.op;
                    Step *low = sm.arith_bin({op, left.low, right.low}, I64);
                    Step *high = sm.arith_bin({op, left.high, right.high}, I64);
                    return RewrittenStep::make_pair({low, high});
                }
                case ArithBinaryOp::Min: {
                    Step *cond = cmp_i128(left, right, CmpOp::Less, false);
                    return RewrittenStep::make_pair(select_i128(cond, left, right));
                }
                case ArithBinaryOp::Max: {
                    Step *cond = cmp_i128(left, right, CmpOp::Less, false);
                    return RewrittenStep::make_pair(select_i128(cond, right, left));
                }
                case ArithBinaryOp::UMin: {
                    Step *cond = cmp_i128(left, right, CmpOp::Less, true);
                    return RewrittenStep::make_pair(select_i128(cond, left, right));
                }
                case ArithBinaryOp::UMax: {
                    Step *cond = cmp_i128(left, right, CmpOp::Less, true);
                    return RewrittenStep::make_pair(select_i128(cond, right, left));
                }
                default: break;
                }
                break;
            }
            SIMJIT_MATCH (StepKind::CheckedOp) {
                Step *mask = data.mask == nullptr ? nullptr : rewrite_scalar(data.mask, step);
                if (data.op->is(StepKind::IntCast)) {
                    const auto &cast = data.op->step_data<StepKind::IntCast>();
                    if (cast.arg->dtype != I128) { break; }
                    SIMJIT_ASSERT(data.kind == CheckedOpKind::Trunc);
                    SIMJIT_ASSERT(cast.kind == IntCastKind::Trunc);
                    SIMJIT_ASSERT(is_simple_int_dtype(step->dtype));
                    SIMJIT_ASSERT(step->dtype != I128);
                    I128Pair source = rewrite_pair_child(cast.arg, step);
                    Step *result = source.low;
                    if (step->dtype != I64) { result = sm.int_cast({IntCastKind::Trunc, result}, step->dtype); }
                    return RewrittenStep::make_scalar(sm.checked_op(
                        {CheckedOpKind::I128Trunc, result, mask, nullptr, source.low, source.high}, step->dtype));
                }
                if (step->dtype != I128) break;
                if (data.op->is(StepKind::ArithUnary)) {
                    const auto &op = data.op->step_data<StepKind::ArithUnary>();
                    SIMJIT_ASSERT(op.op == ArithUnaryOp::Abs || op.op == ArithUnaryOp::Negate);
                    I128Pair arg = rewrite_pair_child(op.arg, step);
                    I128Pair result =
                        op.op == ArithUnaryOp::Abs
                            ? select_i128(sm.cmp({CmpOp::Less, arg.high, get_i64_zero()}, I1), negate_i128(arg), arg)
                            : negate_i128(arg);
                    return RewrittenStep::make_pair(checked_i128_pair(
                        op.op == ArithUnaryOp::Abs ? CheckedOpKind::I128AbsLow : CheckedOpKind::I128NegateLow,
                        op.op == ArithUnaryOp::Abs ? CheckedOpKind::I128AbsHigh : CheckedOpKind::I128NegateHigh, result,
                        mask, arg));
                }
                SIMJIT_ASSERT(data.op->is(StepKind::ArithBinary));
                const auto &op = data.op->step_data<StepKind::ArithBinary>();
                SIMJIT_ASSERT(op.flags == ArithBinaryOpFlags::No);
                I128Pair left = rewrite_pair_child(op.left, step);
                I128Pair right = rewrite_pair_child(op.right, step);
                I128Pair result{};
                CheckedOpKind low_kind{};
                CheckedOpKind high_kind{};
                switch (op.op) {
                case ArithBinaryOp::Add:
                    result = add_i128(left, right);
                    low_kind = CheckedOpKind::I128AddLow;
                    high_kind = CheckedOpKind::I128AddHigh;
                    break;
                case ArithBinaryOp::Sub:
                    result = sub_i128(left, right);
                    low_kind = CheckedOpKind::I128SubLow;
                    high_kind = CheckedOpKind::I128SubHigh;
                    break;
                case ArithBinaryOp::Mul:
                    result = multiply_i128(left, right);
                    low_kind = CheckedOpKind::I128MulLow;
                    high_kind = CheckedOpKind::I128MulHigh;
                    break;
                case ArithBinaryOp::Div:
                case ArithBinaryOp::UDiv:
                case ArithBinaryOp::Mod:
                case ArithBinaryOp::UMod:
                    result = i128_div(left, right, op.op);
                    return RewrittenStep::make_pair(
                        checked_i128_pair(CheckedOpKind::I128Div, CheckedOpKind::I128Div, result, mask, left, right));
                case ArithBinaryOp::ShiftLeftLogical:
                    result = shift_i128(op.op, left, right.low);
                    low_kind = CheckedOpKind::I128SllLow;
                    high_kind = CheckedOpKind::I128SllHigh;
                    break;
                case ArithBinaryOp::ShiftRightLogical:
                    result = shift_i128(op.op, left, right.low);
                    low_kind = CheckedOpKind::I128SrlLow;
                    high_kind = CheckedOpKind::I128SrlHigh;
                    break;
                case ArithBinaryOp::ShiftRightArith:
                    result = shift_i128(op.op, left, right.low);
                    low_kind = CheckedOpKind::I128SraLow;
                    high_kind = CheckedOpKind::I128SraHigh;
                    break;
                case ArithBinaryOp::RotateLeft:
                    result = shift_i128(op.op, left, right.low);
                    low_kind = CheckedOpKind::I128RotlLow;
                    high_kind = CheckedOpKind::I128RotlHigh;
                    break;
                case ArithBinaryOp::RotateRight:
                    result = shift_i128(op.op, left, right.low);
                    low_kind = CheckedOpKind::I128RotrLow;
                    high_kind = CheckedOpKind::I128RotrHigh;
                    break;
                default: SIMJIT_UNREACHABLE();
                }
                return RewrittenStep::make_pair(checked_i128_pair(low_kind, high_kind, result, mask, left, right));
            }
            SIMJIT_MATCH (StepKind::Select) {
                if (step->dtype != I128) break;
                I128Pair truthy = rewrite_pair_child(data.truthy, step);
                I128Pair falsy = rewrite_pair_child(data.falsy, step);
                Step *cond = rewrite_scalar(data.cond, step);
                return RewrittenStep::make_pair(select_i128(cond, truthy, falsy));
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                switch (data.kind) {
                case IntCastKind::Trunc: {
                    if (data.arg->dtype != I128) break;
                    I128Pair pair = rewrite_pair_child(data.arg, step);
                    Step *result = pair.low;
                    if (step->dtype != I64) { result = sm.int_cast({IntCastKind::Trunc, result}, step->dtype); }
                    return RewrittenStep::make_scalar(result);
                }
                case IntCastKind::Sext: {
                    // {x, sra(x, 63)}
                    if (step->dtype != I128) break;
                    Step *low = rewrite_scalar(data.arg, step);
                    if (low->dtype != I64) { low = sm.int_cast({IntCastKind::Sext, low}, I64); }
                    Step *high = sm.arith_bin({ArithBinaryOp::ShiftRightArith, low, get_i64_63()}, I64);
                    return RewrittenStep::make_pair({low, high});
                }
                case IntCastKind::Zext: {
                    // {x, 0}
                    if (step->dtype != I128) break;
                    Step *low = rewrite_scalar(data.arg, step);
                    if (low->dtype != I64) { low = sm.int_cast({IntCastKind::Zext, low}, I64); }
                    Step *high = get_i64_zero();
                    return RewrittenStep::make_pair({low, high});
                }
                }
                break;
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                if (data.arg->dtype == I128) {
                    I128Pair arg = rewrite_pair_child(data.arg, step);
                    return RewrittenStep::make_scalar(i128_to_float(arg, step->dtype, data.is_unsigned));
                }
                if (step->dtype == I128) {
                    Step *arg = rewrite_scalar(data.arg, step);
                    return RewrittenStep::make_pair(float_to_i128(arg, data.is_unsigned));
                }
                break;
            }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                if (step->dtype != I128) break;
                I128Pair pair = rewrite_pair_child(data.arg, step);
                switch (data.op) {
                case ArithUnaryOp::Not: {
                    ArithUnaryOp op = data.op;
                    Step *low = sm.arith_un({op, pair.low}, I64);
                    Step *high = sm.arith_un({op, pair.high}, I64);
                    return RewrittenStep::make_pair({low, high});
                }
                case ArithUnaryOp::Negate: {
                    return RewrittenStep::make_pair(negate_i128(pair));
                }
                case ArithUnaryOp::Abs: {
                    // select(ah < 0, neg(a), a)
                    I128Pair negated = negate_i128(pair);
                    Step *cond = sm.cmp({CmpOp::Less, pair.high, get_i64_zero()}, I1);
                    I128Pair result = select_i128(cond, negated, pair);
                    return RewrittenStep::make_pair(result);
                }
                case ArithUnaryOp::Popcount: {
                    ArithUnaryOp op = data.op;
                    Step *low = sm.arith_un({op, pair.low}, I64);
                    Step *high = sm.arith_un({op, pair.high}, I64);
                    Step *result = sm.arith_bin({ArithBinaryOp::Add, low, high}, I64);
                    return RewrittenStep::make_pair({result, get_i64_zero()});
                }
                case ArithUnaryOp::Lzcnt: {
                    Step *low = sm.arith_un({data.op, pair.low}, I64);
                    Step *high = sm.arith_un({data.op, pair.high}, I64);
                    Step *low_with_high_width = sm.arith_bin({ArithBinaryOp::Add, low, get_i64_64()}, I64);
                    Step *high_is_zero = sm.cmp({CmpOp::Equal, pair.high, get_i64_zero()}, I1);
                    Step *result = sm.select({high_is_zero, low_with_high_width, high}, I64);
                    return RewrittenStep::make_pair({result, get_i64_zero()});
                }
                case ArithUnaryOp::Tzcnt: {
                    Step *low = sm.arith_un({data.op, pair.low}, I64);
                    Step *high = sm.arith_un({data.op, pair.high}, I64);
                    Step *high_with_low_width = sm.arith_bin({ArithBinaryOp::Add, high, get_i64_64()}, I64);
                    Step *low_is_zero = sm.cmp({CmpOp::Equal, pair.low, get_i64_zero()}, I1);
                    Step *result = sm.select({low_is_zero, high_with_low_width, low}, I64);
                    return RewrittenStep::make_pair({result, get_i64_zero()});
                }
                default: break;
                }
                break;
            }
            SIMJIT_MATCH (StepKind::Compare) {
                if (data.left->dtype != I128) { break; }
                I128Pair left = rewrite_pair_child(data.left, step);
                I128Pair right = rewrite_pair_child(data.right, step);
                return RewrittenStep::make_scalar(cmp_i128(left, right, data.op, data.is_unsigned));
            }
            SIMJIT_MATCH (StepKind::Store) {
                if (step->dtype != I128) break;
                mark_i128_argument(data.addr);
                I128Pair value = rewrite_pair_child(data.what, step);
                Step *cond = data.cond == nullptr ? nullptr : rewrite_scalar(data.cond, step);
                return RewrittenStep::make_scalar(
                    sm.store_interleave({value.low, value.high, data.addr, data.kind, cond}, I64));
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                if (step->dtype != I128) break;
                mark_i128_argument(data.dst);
                I128Pair value = rewrite_pair_child(data.arg, step);
                Step *idx = rewrite_scalar(data.idx, step);
                Step *twice_idx = sm.arith_bin({ArithBinaryOp::Add, idx, idx}, idx->dtype);
                Step *odd_idx = sm.arith_bin(
                    {ArithBinaryOp::Add, twice_idx, sm.con(ConstData::i64(1).retag(idx->dtype), idx->dtype)},
                    idx->dtype);
                Step *cond = data.cond == nullptr ? nullptr : rewrite_scalar(data.cond, step);
                Step *low = sm.scatter({value.low, twice_idx, data.dst, cond}, I64);
                additional_roots.push_back(sm.scatter({value.high, odd_idx, data.dst, cond}, I64));
                return RewrittenStep::make_scalar(low);
            }
            SIMJIT_MATCH (StepKind::Pack) {
                if (step->dtype != I128) break;
                mark_i128_argument(data.dst);
                I128Pair value = rewrite_pair_child(data.arg, step);
                Step *cond = rewrite_scalar(data.cond, step);
                return RewrittenStep::make_scalar(
                    sm.pack128({value.low, value.high, cond, data.dst, data.dst_size_acc}, I64));
            }
            SIMJIT_MATCH (StepKind::AccI128) {
                SIMJIT_ASSERT(data.high == nullptr);
                I128Pair value = rewrite_pair_child(data.low, step);
                data.low = value.low;
                data.high = value.high;
                if (data.cond != nullptr) { data.cond = rewrite_scalar(data.cond, step); }
                return RewrittenStep::make_scalar(step);
            }

        default: break;
        }

        if (step->dtype == I128 && !step->is(StepKind::AccI128)) {
            unsupported("i128 rewrite does not support %s", show_step_kind(step->kind));
        }

        step_recurse(step, ScalarChildRewriter{*this, step});
        return RewrittenStep::make_scalar(step);
    }

    void rewrite_roots() {
        std::vector<Step *> roots;
        roots.reserve(func->step_roots.size());
        for (Step *root : func->step_roots) {
            SIMJIT_ASSERT(additional_roots.empty());
            RewrittenStep rewritten_root = rewrite(root);
            SIMJIT_ASSERT(!rewritten_root.is_pair());
            roots.push_back(rewritten_root.scalar);
            roots.insert(roots.end(), additional_roots.begin(), additional_roots.end());
            additional_roots.clear();
        }
        if (roots.size() > func->ctx->build_limits.max_hir_roots) {
            unsupported_limit("HIR has too many roots after i128 rewrite (%zu > %zu)", roots.size(),
                              func->ctx->build_limits.max_hir_roots);
        }
        func->step_roots = func->ctx->arena->copy_array<Step *>(roots);
        func->step_id_count = sm.max_id();
    }
};

// i128 is a public transfer type. Keep its desugaring here, at the boundary
// between builder expressions and finalized HIR, so all later stages only see
// the existing i64 interleaved-pair representation.
static void rewrite_i128_hir(FunctionBuilderImpl *impl, Function *func) {
    // Synthetic steps are ordinary builder steps. Original i128 steps remain
    // allocated but become unreachable as their consumers are replaced.
    I128RewriteState state(impl, func);
    state.rewrite_roots();
}

static void validate_i128_hir(const Function *func) {
    std::vector<uint8_t> state(func->step_id_count, 0);
    for (Step *root : func->step_roots) {
        traverse_steps_postorder_unique(root, state, [&](Step *step) {
            if (step->is(StepKind::AccI128)) {
                const auto &data = step->step_data<StepKind::AccI128>();
                if (data.high == nullptr || data.low->dtype != I64 || data.high->dtype != I64) {
                    unsupported("unlowered i128 aggregate in finalized HIR; enable I128Rewrite");
                }
                ArithBinaryOp op = data.op;
                if (op != ArithBinaryOp::Add && op != ArithBinaryOp::Min && op != ArithBinaryOp::Max &&
                    op != ArithBinaryOp::UMin && op != ArithBinaryOp::UMax && op != ArithBinaryOp::Mul &&
                    op != ArithBinaryOp::And && op != ArithBinaryOp::Or && op != ArithBinaryOp::Xor &&
                    op != ArithBinaryOp::AndNot) {
                    unsupported("i128 aggregation does not support %s", show_arith_binary_op(op));
                }
            } else if (step->dtype == I128) {
                unsupported("unlowered i128 %s in finalized HIR; enable I128Rewrite", show_step_kind(step->kind));
            }
        });
    }
}

static void check_arith_agg_op(ArithBinaryOp op) {
    if (op != ArithBinaryOp::Add && op != ArithBinaryOp::Mul && op != ArithBinaryOp::Min && op != ArithBinaryOp::Max &&
        op != ArithBinaryOp::UMin && op != ArithBinaryOp::UMax && op != ArithBinaryOp::And && op != ArithBinaryOp::Or &&
        op != ArithBinaryOp::Xor && op != ArithBinaryOp::AndNot) {
        invalid_input("Arith binary op %s can't be used in arith agg", show_arith_binary_op(op));
    }
}

static void check_predicate_agg_op(PredicateBinaryOp op) {
    if (op != PredicateBinaryOp::And && op != PredicateBinaryOp::Or && op != PredicateBinaryOp::Xor &&
        op != PredicateBinaryOp::AndNot) {
        invalid_input("Predicate binary op %s can't be used in predicate agg", show_predicate_binary_op(op));
    }
}

Argument FunctionBuilder::arg_safety_check() {
    if (impl_->safety_check_arg) { invalid_input("Already have safety check argument %zu", *impl_->safety_check_arg); }
    Argument x = arg(I8);
    impl_->safety_check_arg = x.idx_;
    return x;
}

void FunctionBuilder::store(Value value, Argument dst, LoadStoreKind kind) {
    cond_store(value, {}, dst, kind);
}

void FunctionBuilder::store(Predicate value, Argument dst) {
    cond_store(value, {}, dst);
}

void FunctionBuilder::cond_store(Value value, MaybePredicate cond, Argument dst, LoadStoreKind kind) {
    check_value(value);

    const ArgumentDecl &func_arg = impl_->get_arg(dst.idx_);
    check_store_dst(func_arg, value.step_, StepKind::Store);

    auto *step = impl_->store(value.step_, cond.step_, func_arg, kind);
    impl_->step_roots.push_back(step);
}

void FunctionBuilder::cond_store(Predicate value, MaybePredicate cond, Argument dst) {
    check_predicate(value);

    const ArgumentDecl &func_arg = impl_->get_arg(dst.idx_);
    check_store_dst(func_arg, value.step_, StepKind::Store);

    if (cond.is_valid()) {
        Predicate old = load_predicate(dst);
        value = select(cond.value(), value, old);
    }

    auto *step = impl_->store(value.step_, nullptr, func_arg, LoadStoreKind::Unaligned);
    impl_->step_roots.push_back(step);
}

void FunctionBuilder::pack(Value value, Predicate cond, Argument dst, Argument dst_size) {
    check_value(value);
    check_predicate(cond);
    const ArgumentDecl &func_arg = impl_->get_arg(dst.idx_);
    check_store_dst(func_arg, value.step_, StepKind::Pack);

    const ArgumentDecl &size_arg = impl_->get_arg(dst_size.idx_);
    if (size_arg.dtype != I64) { invalid_type("Pack result size must be i64"); }
    if (value.dtype() == I8 || value.dtype() == I16) { impl_->special_ops |= SpecialOp::SmallPack; }
    impl_->accs.push_back(Accumulator{I64, AccIdx{impl_->accs.size()}, dst_size.idx_, nullptr});
    Accumulator &acc = impl_->accs.back();
    auto *step = impl_->pack(value.step_, cond.step_, func_arg, acc);
    acc.agg_expr = step;
    impl_->step_roots.push_back(step);
}

void FunctionBuilder::scatter(Value value, Value idx, Argument dst) {
    cond_scatter(value, idx, {}, dst);
}

void FunctionBuilder::cond_scatter(Value value, Value idx, MaybePredicate cond, Argument dst) {
    check_value(value);
    check_value(idx);
    if (!is_simple_int_dtype(idx.dtype())) {
        invalid_type("Only i8, i16, i32 and i64 indices are supported in scatter. Got %s",
                     show_scalar_dtype(idx.dtype()));
    }
    const ArgumentDecl &func_arg = impl_->get_arg(dst.idx_);
    if (func_arg.dtype == I8 || func_arg.dtype == I16) {
        // no SIMD ISA supports scatter for i8/i16.
        scalar_only();
    }
    Step *cond_step = simplify_optional_cond(impl_->ctx, cond.step_);
    impl_->special_ops |= cond_step != nullptr ? SpecialOp::CondScatter : SpecialOp::Scatter;
    if (func_arg.dtype != value.dtype() &&
        (!func_arg.interleaved || func_arg.interleaved_half_dtype != value.dtype())) {
        invalid_type("Step scatter argument and destination have different types %s %s",
                     show_scalar_dtype(value.dtype()), show_scalar_dtype(func_arg.dtype));
    }
    if (idx.dtype() == I8 || idx.dtype() == I16) { idx = zext(idx, I32); }

    auto *step = impl_->scatter(value.step_, idx.step_, cond_step, func_arg);
    impl_->step_roots.push_back(step);
}

void FunctionBuilder::store_interleave(Argument dst, Value even, Value odd, LoadStoreKind kind) {
    cond_store_interleave(even, odd, {}, dst, kind);
}

void FunctionBuilder::cond_store_interleave(Value even, Value odd, MaybePredicate cond, Argument dst,
                                            LoadStoreKind kind) {
    check_value(even);
    check_value(odd);
    if (even.dtype() != odd.dtype()) {
        invalid_type("Store interleave even and odd arguments should have same type, got %s, %s",
                     show_scalar_dtype(even.dtype()), show_scalar_dtype(odd.dtype()));
    }
    const ArgumentDecl &func_arg = impl_->get_arg(dst.idx_);
    if (!func_arg.interleaved) { invalid_type("store_interleave requires an interleaved argument"); }
    if (func_arg.interleaved_half_dtype != even.dtype()) {
        invalid_type("store_interleave values have type %s but argument halves are %s", show_scalar_dtype(even.dtype()),
                     show_scalar_dtype(func_arg.interleaved_half_dtype));
    }
    if (func_arg.interleaved_half_dtype == I8) { impl_->special_ops |= SpecialOp::SmallDeinterleave; }

    auto *step = impl_->store_interleave(even.step_, odd.step_, cond.step_, func_arg, kind);
    impl_->step_roots.push_back(step);
}

void FunctionBuilder::arith_agg(Value value, ArithBinaryOp op, Argument dst) {
    cond_arith_agg(value, {}, op, dst);
}

void FunctionBuilder::cond_arith_agg(Value value, MaybePredicate cond, ArithBinaryOp op, Argument dst) {
    check_value(value);
    check_arith_agg_op(op);
    const ArgumentDecl &dst_info = impl_->get_arg(dst.idx_);
    ScalarDataType dtype = dst_info.dtype;
    bool widening_sum = op == ArithBinaryOp::Add && is_int_dtype(value.dtype()) && is_int_dtype(dtype) &&
                        scalar_dtype_size(value.dtype()) < scalar_dtype_size(dtype);
    if (widening_sum) {
        if (!bool(impl_->ctx->transformations & CodeTransformations::SumAutoSext)) {
            unsupported("widening sum requires SumAutoSext or an explicit sext");
        }
        Step *original = nullptr;
        value.step_ = impl_->int_cast(value.step_, dtype, IntCastKind::Sext, &original);
    }
    if (dst_info.dtype == I128) {
        if (value.dtype() != I128) {
            unsupported("Only support calculating 128-bit aggregates of i128 values, got %s",
                        show_scalar_dtype(value.dtype()));
        }
        if (op != ArithBinaryOp::Add && op != ArithBinaryOp::Min && op != ArithBinaryOp::Max &&
            op != ArithBinaryOp::UMin && op != ArithBinaryOp::UMax && op != ArithBinaryOp::Mul &&
            op != ArithBinaryOp::And && op != ArithBinaryOp::Or && op != ArithBinaryOp::Xor &&
            op != ArithBinaryOp::AndNot) {
            unsupported("Don't support 128-bit %s agg", show_arith_binary_op(op));
        }
    } else {
        check_store_dst(dst_info, value.step_, StepKind::AccArithBinary);
    }

    if (op == ArithBinaryOp::Mul) {
        if (value.dtype() == I64) { impl_->special_ops |= SpecialOp::I64Mul; }
    }

    impl_->accs.push_back(Accumulator{dtype, AccIdx{impl_->accs.size()}, dst.idx_, nullptr});
    Accumulator &acc = impl_->accs.back();
    auto *step = impl_->arith_agg(value.step_, cond.step_, op, acc);
    acc.agg_expr = step;
    impl_->step_roots.push_back(step);
}

void FunctionBuilder::grouped_arith_agg(Value value, Value idx, ArithBinaryOp op, Argument table) {
    check_value(value);
    check_value(idx);
    scalar_only();

    Value loaded = gather(idx, table);
    Value updated = arith_binary(value, loaded, op);
    scatter(updated, idx, table);
}

void FunctionBuilder::grouped_cond_arith_agg(Value value, MaybePredicate cond, Value idx, ArithBinaryOp op,
                                             Argument table) {
    check_value(value);
    check_predicate(cond);
    check_value(idx);
    scalar_only();

    Value loaded = gather(idx, table);
    Value updated = arith_binary(value, loaded, op);
    updated = select(cond.value(), updated, loaded);
    scatter(updated, idx, table);
}

void FunctionBuilder::predicate_agg(Predicate value, PredicateBinaryOp op, Argument dst) {
    check_predicate(value);
    check_predicate_agg_op(op);

    const ArgumentDecl &dst_info = impl_->get_arg(dst.idx_);
    check_store_dst(dst_info, value.step_, StepKind::AccPredicateBinary);

    ScalarDataType dtype = value.dtype();
    impl_->accs.push_back(Accumulator{dtype, AccIdx{impl_->accs.size()}, dst.idx_, nullptr});
    Accumulator &acc = impl_->accs.back();
    auto *step = impl_->predicate_agg(value.step_, op, acc);
    acc.agg_expr = step;
    impl_->step_roots.push_back(step);
}

static uint64_t xorshift64(uint64_t x) noexcept {
    x ^= x << 13ull;
    x ^= x >> 7ull;
    x ^= x << 17ull;
    return x;
}

template <typename T> static uint64_t to_uint64(T value) noexcept {
    return size_t(value);
}

template <> uint64_t to_uint64(Argument value) noexcept {
    return value.idx_;
}

template <> uint64_t to_uint64(Value value) noexcept {
    return (uint64_t)(uintptr_t)value.step_;
}

template <> uint64_t to_uint64(Predicate value) noexcept {
    return (uint64_t)(uintptr_t)value.step_;
}

template <> uint64_t to_uint64(Step *value) noexcept {
    return (uint64_t)(uintptr_t)value;
}

template <typename... Args> static uint64_t make_hash(StepKind kind, Args... args) noexcept {
    return xorshift64((((uint64_t)kind) << 32ull) | (to_uint64(args) + ...));
}

Value FunctionBuilder::load(Argument argument, LoadStoreKind kind) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype == I1) { invalid_type("load should not be used with i1 type"); }

    uint64_t hash = make_hash(StepKind::Load, argument, kind);
    if (auto *result = impl_->cse<StepKind::Load>(
            hash, [argument, kind](const auto &x) { return x.idx == argument.idx_ && x.kind == kind; })) {
        return result;
    }

    Step *s = impl_->load(func_arg, kind);
    return impl_->remember_expr(hash, s);
}

DeinterleavedPair FunctionBuilder::load2(Argument argument, LoadStoreKind kind) {
    return {load2_0(argument, kind), load2_1(argument, kind)};
}

Value FunctionBuilder::load2_0(Argument argument, LoadStoreKind kind) {
    return load_deinterleave(argument, 2, 0, kind);
}

Value FunctionBuilder::load2_1(Argument argument, LoadStoreKind kind) {
    return load_deinterleave(argument, 2, 1, kind);
}

Value FunctionBuilder::load_deinterleave(Argument argument, uint8_t lane_count, uint8_t lane, LoadStoreKind kind) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (!func_arg.interleaved) { invalid_type("load_deinterleave requires an interleaved argument"); }
    if (lane_count != 2) {
        invalid_input("load_deinterleave supports only 2-way interleaved loads, got %u lanes", lane_count);
    }
    if (lane >= lane_count) { invalid_input("load_deinterleave lane %u out of %u lanes", lane, lane_count); }
    if (func_arg.interleaved_half_dtype == I8) { impl_->special_ops |= SpecialOp::SmallDeinterleave; }

    uint64_t hash = make_hash(StepKind::LoadDeinterleave, argument, kind, lane_count, lane);
    if (auto *result = impl_->cse<StepKind::LoadDeinterleave>(hash, [argument, kind, lane_count, lane](const auto &x) {
            return x.idx == argument.idx_ && x.kind == kind && x.lane_count == lane_count && x.lane == lane;
        })) {
        return result;
    }
    return impl_->remember_expr(hash, impl_->load_deinterleave(func_arg, kind, lane_count, lane));
}

Predicate FunctionBuilder::load_predicate(Argument argument) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype != I1) { invalid_type("load_predicate should use I1 dtype"); }

    uint64_t hash = make_hash(StepKind::Load, argument);
    if (auto *result = impl_->cse<StepKind::Load>(hash, [argument](const auto &x) { return x.idx == argument.idx_; })) {
        return result;
    }

    Step *s = impl_->load(func_arg, LoadStoreKind::Unaligned);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::gather(Value idx, Argument argument) {
    return gather(idx, argument, impl_->get_arg(argument.idx_).dtype);
}

Value FunctionBuilder::gather(Value idx, Argument argument, ScalarDataType dtype) {
    check_value(idx);

    if (!is_simple_int_dtype(idx.dtype())) {
        invalid_type("Only i8, i16, i32 and i64 indices are supported in gather. Got %s",
                     show_scalar_dtype(idx.dtype()));
    }
    if (idx.dtype() == I8 || idx.dtype() == I16) { idx = zext(idx, I32); }
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (dtype != func_arg.dtype && (!func_arg.interleaved || dtype != func_arg.interleaved_half_dtype)) {
        invalid_type("gather type %s does not match argument carrier %s or interleaved half type",
                     show_scalar_dtype(dtype), show_scalar_dtype(func_arg.dtype));
    }
    uint64_t hash = make_hash(StepKind::Gather, argument, idx, dtype);
    if (auto *result = impl_->cse<StepKind::Gather>(
            hash, dtype, [idx, argument](const auto &x) { return x.idx == idx.step_ && x.data == argument.idx_; })) {
        return result;
    }
    if (dtype == I8 || dtype == I16) { impl_->special_ops |= SpecialOp::SmallGather; }
    impl_->special_ops |= SpecialOp::Gather;
    Step *s = impl_->gather(idx.step_, func_arg, dtype);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::load_splat(Argument argument, size_t byte_offset) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    return load_splat(argument, func_arg.dtype, byte_offset);
}

Value FunctionBuilder::load_splat(Argument argument, ScalarDataType dtype, size_t byte_offset) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (dtype == I1) { invalid_type("load should not be used with i1 type"); }
    if (dtype != func_arg.dtype && (!func_arg.interleaved || dtype != func_arg.interleaved_half_dtype)) {
        invalid_type("load_splat type does not match argument");
    }

    uint64_t hash = make_hash(StepKind::LoadSplat, argument, dtype, byte_offset);
    if (auto *result = impl_->cse<StepKind::LoadSplat>(hash, dtype, [argument, byte_offset](const auto &x) {
            return x.idx == argument.idx_ && x.byte_offset == byte_offset;
        })) {
        return result;
    }
    Step *s = impl_->load_splat(func_arg, dtype, byte_offset);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::load_predicate_splat(Argument argument) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype != I1) { invalid_type("load_predicate_splat should use I1 dtype"); }

    uint64_t hash = make_hash(StepKind::LoadSplat, argument);
    if (auto *result =
            impl_->cse<StepKind::LoadSplat>(hash, [argument](const auto &x) { return x.idx == argument.idx_; })) {
        return result;
    }
    Step *s = impl_->load_splat(func_arg, I1);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::con_internal(ConstData data, ScalarDataType dtype) {
    if (is_float_dtype(dtype)) {
    } else if (dtype == I128) {
    } else if (!(data.as_signed() >= scalar_dtype_min(dtype).as_signed() &&
                 data.as_signed() <= scalar_dtype_max(dtype).as_signed()) &&
               !(data.as_unsigned() <= scalar_dtype_umax(dtype))) {
        invalid_type("Constant value %lld (unsigned %llu) does not fit into allowed range for type %s",
                     (long long)data.as_signed(), (unsigned long long)data.as_unsigned(), show_scalar_dtype(dtype));
    }
    if (dtype != I128) { data = data.retag(dtype); }

    uint64_t hash = make_hash(StepKind::Const, data.raw_bits(), dtype);
    if (auto *result = impl_->cse<StepKind::Const>(hash, dtype, [data](const auto &x) { return x == data; })) {
        return result;
    }
    Step *s = impl_->con(data, dtype);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::i1(bool data) {
    ScalarDataType dtype = I1;
    uint64_t hash = make_hash(StepKind::Const, data, dtype);
    if (auto *result =
            impl_->cse<StepKind::Const>(hash, dtype, [data](const auto &x) { return x == ConstData::i1(data); })) {
        return result;
    }
    Step *s = impl_->con(ConstData::i1(data), dtype);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::arith_binary(Value left, Value right, ArithBinaryOp op, ArithBinaryOpFlags flags) {
    check_value(left);
    check_value(right);

    bool is_division_op =
        op == ArithBinaryOp::Div || op == ArithBinaryOp::UDiv || op == ArithBinaryOp::Mod || op == ArithBinaryOp::UMod;

    if (left.dtype() != right.dtype()) {
        invalid_type("Binary operation %s left and right side should have same type, got %s, %s",
                     show_arith_binary_op(op), show_scalar_dtype(left.dtype()), show_scalar_dtype(right.dtype()));
    }

    if ((op == ArithBinaryOp::Mul64SE || op == ArithBinaryOp::Mul64ZE) && left.dtype() != I64) {
        invalid_type("Binary operation %s requires i64 operands, got %s", show_arith_binary_op(op),
                     show_scalar_dtype(left.dtype()));
    }

    bool safety_check = bool(flags & ArithBinaryOpFlags::SafetyCheck);
    if (bool(flags & ArithBinaryOpFlags::SafeDivision)) {
        if (!is_division_op) {
            invalid_input("Safe division flag is not supported for binary arithmetic operation %s",
                          show_arith_binary_op(op));
        }
        if (!is_simple_int_dtype(left.dtype()) && left.dtype() != I128 && !is_float_dtype(left.dtype())) {
            unsupported("Safe division is not supported for op %s type %s", show_arith_binary_op(op),
                        show_scalar_dtype(left.dtype()));
        }
    }

    if (bool(flags & ArithBinaryOpFlags::ShiftWraparound)) {
        if (safety_check) {
            invalid_input("Can't define both safety check and wraparound for op %s", show_arith_binary_op(op));
        }
        if (bool(flags & ArithBinaryOpFlags::SafeDivision)) {
            invalid_input("Can't define both shift safety check and safe division for op %s", show_arith_binary_op(op));
        }
    }

    if (safety_check) {
        flags &= ~ArithBinaryOpFlags::SafetyCheck;
        if (is_division_op) { flags &= ~ArithBinaryOpFlags::SafeDivision; }
        // A bit dumb recursion
        return checked_op(arith_binary(left, right, op, flags));
    }

    StepPair lookup = is_commutative_arith_cse_op(op, left.dtype()) ? ordered_step_pair(left.step_, right.step_)
                                                                    : StepPair{left.step_, right.step_};
    uint64_t hash = make_hash(StepKind::ArithBinary, lookup.left, lookup.right, op, flags);
    if (auto *result = impl_->cse<StepKind::ArithBinary>(hash, [lookup, op, flags](const auto &x) {
            StepPair key = is_commutative_arith_cse_op(op, x.left->dtype) ? ordered_step_pair(x.left, x.right)
                                                                          : StepPair{x.left, x.right};
            return key.left == lookup.left && key.right == lookup.right && x.op == op && x.flags == flags;
        })) {
        return result;
    }

    Step *key = nullptr;
    Step *s = impl_->arith_binary(left.step_, right.step_, op, flags, &key);
    impl_->special_ops |= arith_binary_special_ops(s);
    return impl_->remember_expr(hash, s, key);
}

Value FunctionBuilder::checked_op(Value value, MaybePredicate mask) {
    check_value(value);
    if (mask.is_valid()) { check_predicate(mask); }
    if (value.step_->is(StepKind::ArithBinary)) {
        ArithBinaryOp optimized_op = value.step_->step_data<StepKind::ArithBinary>().op;
        if (optimized_op == ArithBinaryOp::Mul64SE || optimized_op == ArithBinaryOp::Mul64ZE) {
            impl_->ensure_has_safety_check_arg();
            return value;
        }
    }
    Step *operation = impl_->original_expr(value.step_);
    if (operation->is(StepKind::ArithBinary)) {
        const auto &arith = operation->step_data<StepKind::ArithBinary>();
        ArithBinaryOp op = arith.op;
        bool is_division_op = op == ArithBinaryOp::Div || op == ArithBinaryOp::UDiv || op == ArithBinaryOp::Mod ||
                              op == ArithBinaryOp::UMod;
        bool is_shift_or_rotate = op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftRightLogical ||
                                  op == ArithBinaryOp::ShiftLeftLogical || op == ArithBinaryOp::RotateLeft ||
                                  op == ArithBinaryOp::RotateRight;
        if (op != ArithBinaryOp::Add && op != ArithBinaryOp::Sub && op != ArithBinaryOp::Mul && !is_shift_or_rotate &&
            !is_division_op) {
            invalid_input("safety checking is not supported for binary arithmetic operation %s",
                          show_arith_binary_op(op));
        }
        bool supports_all_int_widths =
            op == ArithBinaryOp::Add || op == ArithBinaryOp::Sub || op == ArithBinaryOp::Mul || is_shift_or_rotate;
        bool supported_type = value.dtype() == I32 || value.dtype() == I64 || value.dtype() == I128 ||
                              (supports_all_int_widths && is_simple_int_dtype(value.dtype()));
        if (!supported_type) {
            if (supports_all_int_widths) {
                unsupported("safety checking is not supported for op %s type %s; Only integer types are supported",
                            show_arith_binary_op(op), show_scalar_dtype(value.dtype()));
            }
            unsupported("safety checking is not supported for op %s type %s; Only support i32, i64",
                        show_arith_binary_op(op), show_scalar_dtype(value.dtype()));
        }
        if (arith.flags != ArithBinaryOpFlags::No) {
            invalid_input("Checked operation must not already have arithmetic flags");
        }
    } else if (operation->is(StepKind::IntCast)) {
        auto &data = operation->step_data<StepKind::IntCast>();
        if (data.kind != IntCastKind::Trunc) {
            invalid_input("Do not support safety checking for %s cast. Only truncate cast can be unsafe",
                          show_int_cast_kind(data.kind));
        }
        if (data.arg->dtype == I128 && !is_simple_int_dtype(value.dtype())) {
            unsupported("safety checking i128 truncation only supports integer destinations");
        }
    } else if (operation->is(StepKind::ArithUnary)) {
        auto &data = operation->step_data<StepKind::ArithUnary>();
        if (data.op != ArithUnaryOp::Abs && data.op != ArithUnaryOp::Negate) {
            invalid_input("safety checking is not supported for unary arithmetic operation %s",
                          show_arith_unary_op(data.op));
        }
        if (!is_simple_int_dtype(value.dtype()) && value.dtype() != I128) {
            unsupported("safety checking is not supported for op %s type %s; Only integer types are supported",
                        show_arith_unary_op(data.op), show_scalar_dtype(value.dtype()));
        }
    } else {
        invalid_input("Unsupported checked operation %s", show_step_kind(operation->kind));
    }

    CheckedOpKind kind{};
    if (operation->is(StepKind::ArithBinary)) {
        switch (operation->step_data<StepKind::ArithBinary>().op) {
        case ArithBinaryOp::Add: kind = CheckedOpKind::Add; break;
        case ArithBinaryOp::Sub: kind = CheckedOpKind::Sub; break;
        case ArithBinaryOp::Mul: kind = CheckedOpKind::Mul; break;
        case ArithBinaryOp::Div: kind = CheckedOpKind::Div; break;
        case ArithBinaryOp::UDiv: kind = CheckedOpKind::UDiv; break;
        case ArithBinaryOp::Mod: kind = CheckedOpKind::Mod; break;
        case ArithBinaryOp::UMod: kind = CheckedOpKind::UMod; break;
        case ArithBinaryOp::ShiftRightArith: kind = CheckedOpKind::ShiftRightArith; break;
        case ArithBinaryOp::ShiftRightLogical: kind = CheckedOpKind::ShiftRightLogical; break;
        case ArithBinaryOp::ShiftLeftLogical: kind = CheckedOpKind::ShiftLeftLogical; break;
        case ArithBinaryOp::RotateLeft: kind = CheckedOpKind::RotateLeft; break;
        case ArithBinaryOp::RotateRight: kind = CheckedOpKind::RotateRight; break;
        default: SIMJIT_UNREACHABLE();
        }
    } else if (operation->is(StepKind::IntCast)) {
        kind = CheckedOpKind::Trunc;
    } else if (operation->is(StepKind::ArithUnary)) {
        kind = operation->step_data<StepKind::ArithUnary>().op == ArithUnaryOp::Abs ? CheckedOpKind::Abs
                                                                                    : CheckedOpKind::Negate;
    } else {
        SIMJIT_UNREACHABLE();
    }

    impl_->ensure_has_safety_check_arg();
    if (value.dtype() == I128 ||
        (operation->is(StepKind::IntCast) && operation->step_data<StepKind::IntCast>().arg->dtype == I128)) {
        impl_->vectorization_hint = VectorizationHint::ScalarOnly;
    }

    Step *mask_step = mask.is_valid() ? mask.value().step_ : nullptr;
    mask_step = simplify_optional_cond(impl_->ctx, mask_step);
    uint64_t hash = make_hash(StepKind::CheckedOp, kind, operation, mask_step);
    if (auto *result = impl_->cse<StepKind::CheckedOp>(hash, [kind, operation, mask_step](const auto &x) {
            return x.kind == kind && x.op == operation && x.mask == mask_step;
        })) {
        return result;
    }
    impl_->special_ops |= arith_binary_special_ops(operation);
    Step *result = impl_->sm.checked_op({kind, operation, mask_step}, value.dtype());
    return impl_->remember_expr(hash, result);
}

Value FunctionBuilder::arith_unary(Value value, ArithUnaryOp op, bool safety_check) {
    check_value(value);

    if (safety_check) {
        // a bit dumb recursion
        return checked_op(arith_unary(value, op, false));
    }

    uint64_t hash = make_hash(StepKind::ArithUnary, value, op, safety_check);
    if (auto *result = impl_->cse<StepKind::ArithUnary>(
            hash, [value, op](const auto &x) { return x.arg == value.step_ && x.op == op; })) {
        return result;
    }
    switch (op) {
    case ArithUnaryOp::Not:
    case ArithUnaryOp::Negate:
    case ArithUnaryOp::Abs:
        if (!(is_int_dtype(value.dtype()) || is_float_dtype(value.dtype()))) {
            invalid_type("Invalid unary arithmetic operation %s for type %s", show_arith_unary_op(op),
                         show_scalar_dtype(value.dtype()));
        }
        break;
    case ArithUnaryOp::Lzcnt:
    case ArithUnaryOp::Tzcnt:
    case ArithUnaryOp::Popcount:
        if (!is_int_dtype(value.dtype())) {
            invalid_type("Invalid unary arithmetic operation %s for type %s", show_arith_unary_op(op),
                         show_scalar_dtype(value.dtype()));
        }
        break;
    case ArithUnaryOp::RoundNearest:
    case ArithUnaryOp::RoundDown:
    case ArithUnaryOp::RoundUp:
    case ArithUnaryOp::RoundTruncate:
    case ArithUnaryOp::Rcp:
    case ArithUnaryOp::Sqrt:
    case ArithUnaryOp::Rsqrt:
        if (!is_float_dtype(value.dtype())) {
            invalid_type("Invalid unary arithmetic operation %s for type %s", show_arith_unary_op(op),
                         show_scalar_dtype(value.dtype()));
        }
        break;
    }

    Step *s = impl_->arith_unary(value.step_, op);
    impl_->special_ops |= arith_unary_special_ops(s);
    return impl_->remember_expr(hash, s);
}

void FunctionBuilder::countif(Predicate value, Argument dst) {
    check_predicate(value);

    ArgumentDecl &dst_info = impl_->get_arg(dst.idx_);
    ScalarDataType dtype = dst_info.dtype;
    if (dtype != I64) { invalid_type("Only support calculating 64-bit countif, got %s", show_scalar_dtype(dtype)); }

    impl_->accs.push_back(Accumulator{dtype, AccIdx{impl_->accs.size()}, dst.idx_, nullptr});
    Accumulator &acc = impl_->accs.back();
    auto *step = impl_->countif(value.step_, acc);
    acc.agg_expr = step;
    impl_->step_roots.push_back(step);
}

Predicate FunctionBuilder::predicate_not(Predicate value) {
    check_predicate(value);

    uint64_t hash = make_hash(StepKind::PredicateNot, value);
    if (auto *result = impl_->cse<StepKind::PredicateNot>(hash, [value](const auto &x) { return x == value.step_; })) {
        return result;
    }
    Step *key = nullptr;
    Step *s = impl_->predicate_not(value.step_, &key);
    return impl_->remember_expr(hash, s, key);
}

Value FunctionBuilder::int_cast(Value value, ScalarDataType dtype, IntCastKind kind, bool safety_check) {
    check_value(value);
    if (value.dtype() == dtype) { invalid_type("Invalid cast from type %s to itself", show_scalar_dtype(dtype)); }
    if (!is_int_dtype(value.dtype()) || !is_int_dtype(dtype)) {
        invalid_type("Invalid int_cast from %s to %s", show_scalar_dtype(value.dtype()), show_scalar_dtype(dtype));
    }
    {
        ScalarDataType from = value.dtype();
        ScalarDataType to = dtype;
        bool extend = scalar_dtype_size(from) < scalar_dtype_size(to);
        switch (kind) {
        case IntCastKind::Trunc:
            if (extend) {
                invalid_type("Invalid truncate from type %s to type %s", show_scalar_dtype(from),
                             show_scalar_dtype(to));
            }
            break;
        case IntCastKind::Sext:
            if (!extend) {
                invalid_type("Invalid sign extend from type %s to type %s", show_scalar_dtype(from),
                             show_scalar_dtype(to));
            }
            break;
        case IntCastKind::Zext:
            if (!extend) {
                invalid_type("Invalid zero extend from type %s to type %s", show_scalar_dtype(from),
                             show_scalar_dtype(to));
            }
            break;
        }
    }

    if (safety_check) {
        if (kind != IntCastKind::Trunc) {
            invalid_input("Do not support safety checking for %s cast. Only truncate cast can be unsafe",
                          show_int_cast_kind(kind));
        }
        impl_->ensure_has_safety_check_arg();
        return checked_op(int_cast(value, dtype, kind, false));
    }

    uint64_t hash = make_hash(StepKind::IntCast, value, dtype, kind);
    if (auto *result = impl_->cse<StepKind::IntCast>(
            hash, dtype, [value, kind](const auto &x) { return x.arg == value.step_ && x.kind == kind; })) {
        return result;
    }
    Step *key = nullptr;
    Step *s = impl_->int_cast(value.step_, dtype, kind, &key);
    return impl_->remember_expr(hash, s, key);
}

Value FunctionBuilder::float_cast(Value value, ScalarDataType dtype, bool is_unsigned) {
    check_value(value);

    bool valid_compact = (is_float_dtype(dtype) && is_float_dtype(value.dtype())) ||
                         (is_compact_float_cast_int_dtype(value.dtype()) && is_float_dtype(dtype)) ||
                         (is_compact_float_cast_int_dtype(dtype) && is_float_dtype(value.dtype()));
    bool valid_i128 =
        (value.dtype() == I128 && is_float_dtype(dtype)) || (dtype == I128 && is_float_dtype(value.dtype()));
    bool valid_decomposed = bool(impl_->ctx->transformations & CodeTransformations::CastDecomposition) &&
                            ((is_simple_int_dtype(value.dtype()) && is_float_dtype(dtype)) ||
                             (is_simple_int_dtype(dtype) && is_float_dtype(value.dtype())));
    if (!valid_compact && !valid_i128 && !valid_decomposed) {
        invalid_type("Invalid float_cast %s -> %s", show_scalar_dtype(value.dtype()), show_scalar_dtype(dtype));
    }
    uint64_t hash = make_hash(StepKind::FloatCast, value, dtype, is_unsigned);
    if (auto *result = impl_->cse<StepKind::FloatCast>(hash, dtype, [value, is_unsigned](const auto &x) {
            return x.arg == value.step_ && x.is_unsigned == is_unsigned;
        })) {
        return result;
    }
    Step *key = nullptr;
    Step *s = impl_->float_cast(value.step_, dtype, is_unsigned, &key);
    return impl_->remember_expr(hash, s, key);
}

Value FunctionBuilder::bitcast(Value value, ScalarDataType dtype) {
    check_value(value);

    if (!((value.dtype() == F32 && dtype == I32) || (value.dtype() == F64 && dtype == I64) ||
          (value.dtype() == I32 && dtype == F32) || (value.dtype() == I64 && dtype == F64))) {
        invalid_type("Invalid bitcast %s -> %s", show_scalar_dtype(value.dtype()), show_scalar_dtype(dtype));
    }

    uint64_t hash = make_hash(StepKind::BitCast, value, dtype);
    if (auto *result =
            impl_->cse<StepKind::BitCast>(hash, dtype, [value](const auto &x) { return x == value.step_; })) {
        return result;
    }
    Step *s = impl_->bitcast(value.step_, dtype);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::cmp(Value left, Value right, CmpOp op, bool is_unsigned) {
    check_value(left);
    check_value(right);

    if (left.dtype() != right.dtype()) {
        invalid_type("Compare left and right side should have same type, got %s, %s", show_scalar_dtype(left.dtype()),
                     show_scalar_dtype(right.dtype()));
    }
    if (is_unsigned && is_float_dtype(left.dtype())) {
        invalid_input("Unsigned cmp is not supported for floating-point types (got %s)",
                      show_scalar_dtype(left.dtype()));
    }

    uint64_t hash = make_hash(StepKind::Compare, left, right, op, is_unsigned);
    if (auto *result = impl_->cse<StepKind::Compare>(hash, [left, right, op, is_unsigned](const auto &x) {
            return x.left == left.step_ && x.right == right.step_ && x.op == op && x.is_unsigned == is_unsigned;
        })) {
        return result;
    }
    Step *key = nullptr;
    Step *s = impl_->cmp(left.step_, right.step_, op, is_unsigned, &key);
    return impl_->remember_expr(hash, s, key);
}

Predicate FunctionBuilder::fpclass(Value value, FpClass flags) {
    check_value(value);

    if (!is_float_dtype(value.dtype())) {
        invalid_type("Fpclass argument must have floating point type, got %s", show_scalar_dtype(value.dtype()));
    }

    uint64_t hash = make_hash(StepKind::Fpclass, value, flags);
    if (auto *result = impl_->cse<StepKind::Fpclass>(
            hash, [value, flags](const auto &x) { return x.arg == value.step_ && x.flags == flags; })) {
        return result;
    }

    Step *s = impl_->fpclass(value.step_, flags);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::predicate_binary(Predicate left, Predicate right, PredicateBinaryOp op) {
    check_predicate(left);
    check_predicate(right);

    StepPair lookup = is_commutative_predicate_cse_op(op) ? ordered_step_pair(left.step_, right.step_)
                                                          : StepPair{left.step_, right.step_};
    uint64_t hash = make_hash(StepKind::PredicateBinary, lookup.left, lookup.right, op);
    if (auto *result = impl_->cse<StepKind::PredicateBinary>(hash, [lookup, op](const auto &x) {
            StepPair key =
                is_commutative_predicate_cse_op(op) ? ordered_step_pair(x.left, x.right) : StepPair{x.left, x.right};
            return key.left == lookup.left && key.right == lookup.right && x.op == op;
        })) {
        return result;
    }

    Step *key = nullptr;
    Step *s = impl_->predicate_binary(left.step_, right.step_, op, &key);
    return impl_->remember_expr(hash, s, key);
}

Value FunctionBuilder::select(Predicate cond, Value truthy, Value falsy) {
    check_predicate(cond);
    check_value(truthy);
    check_value(falsy);

    if (falsy.dtype() != truthy.dtype()) {
        invalid_type("Select true and false side should have same type, got %s, %s", show_scalar_dtype(falsy.dtype()),
                     show_scalar_dtype(truthy.dtype()));
    }

    uint64_t hash = make_hash(StepKind::Select, cond, truthy, falsy);
    if (auto *result = impl_->cse<StepKind::Select>(hash, [cond, falsy, truthy](const auto &x) {
            return x.falsy == falsy.step_ && x.truthy == truthy.step_ && x.cond == cond.step_;
        })) {
        return result;
    }
    Step *key = nullptr;
    Step *s = impl_->select(cond.step_, truthy.step_, falsy.step_, &key);
    return impl_->remember_expr(hash, s, key);
}

Predicate FunctionBuilder::select(Predicate cond, Predicate truthy, Predicate falsy) {
    check_predicate(cond);
    check_predicate(truthy);
    check_predicate(falsy);

    if (falsy.dtype() != truthy.dtype()) {
        invalid_type("Select true and false side should have same type, got %s, %s", show_scalar_dtype(falsy.dtype()),
                     show_scalar_dtype(truthy.dtype()));
    }

    uint64_t hash = make_hash(StepKind::Select, cond, truthy, falsy);
    if (auto *result = impl_->cse<StepKind::Select>(hash, [cond, falsy, truthy](const auto &x) {
            return x.falsy == falsy.step_ && x.truthy == truthy.step_ && x.cond == cond.step_;
        })) {
        return result;
    }
    Predicate result = or_(and_(cond, truthy), and_(not_(cond), falsy));
    Step *key = nullptr;
    (void)impl_->select(cond.step_, truthy.step_, falsy.step_, &key);
    return Predicate{impl_->remember_expr(hash, result.step_, key)};
}

Value FunctionBuilder::index(ScalarDataType dtype) {
    if (!is_simple_int_dtype(dtype) && !is_float_dtype(dtype)) {
        invalid_type("Can't have index of %s type", show_scalar_dtype(dtype));
    }
    uint64_t hash = make_hash(StepKind::Index, dtype);
    if (auto *result = impl_->cse<StepKind::Index>(hash, dtype, [](auto &) { return true; })) { return result; }
    Step *s = impl_->index(dtype);
    return impl_->remember_expr(hash, s);
}

static void check_permute_idx(uint8_t x) {
    if (x >= 8) { invalid_input("Permute byte index can't be larger or equal 8"); }
}

Value FunctionBuilder::funnel_shift(Value left, Value right, Value amount, FunnelShiftKind kind) {
    check_value(left);
    check_value(right);
    check_value(amount);

    if (left.dtype() != right.dtype() || left.dtype() != amount.dtype()) {
        invalid_type("Funnel shift arguments should have same type, got %s, %s, %s", show_scalar_dtype(left.dtype()),
                     show_scalar_dtype(right.dtype()), show_scalar_dtype(amount.dtype()));
    }
    if (left.dtype() != I16 && left.dtype() != I32 && left.dtype() != I64) {
        invalid_type("Funnel shift supports i16, i32 and i64 arguments, got %s", show_scalar_dtype(left.dtype()));
    }

    impl_->special_ops |= SpecialOp::FunnelShift;

    uint64_t hash = make_hash(StepKind::FunnelShift, left.step_, right.step_, amount.step_, kind);
    if (auto *result = impl_->cse<StepKind::FunnelShift>(hash, [left, right, amount, kind](const auto &x) {
            return x.left == left.step_ && x.right == right.step_ && x.amount == amount.step_ && x.kind == kind;
        })) {
        return result;
    }
    Step *s = impl_->funnel_shift(left.step_, right.step_, amount.step_, kind);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::permute(Value value, uint64_t permute_idxs, bool is_bit) {
    check_value(value);

    if (!is_simple_int_dtype(value.dtype())) {
        invalid_type("Byte permute input must be one of the i8,i16,i32,i64 types. Got %s",
                     show_scalar_dtype(value.dtype()));
    }
    if (is_bit && permute_idxs != REVERSE_BITS) { impl_->special_ops |= SpecialOp::ArbitraryBitPermute; }

    uint64_t hash = make_hash(StepKind::Permute, value, permute_idxs);
    if (auto *result = impl_->cse<StepKind::Permute>(hash, [value, permute_idxs, is_bit](const auto &x) {
            return x.is_bit == is_bit && x.arg == value.step_ && x.permute == permute_idxs;
        })) {
        return result;
    }
    Step *s = is_bit ? impl_->bit_permute(value.step_, permute_idxs) : impl_->byte_permute(value.step_, permute_idxs);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::permute_i64_i8(Value value, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5,
                                      uint8_t a6, uint8_t a7, uint8_t a8) {
    check_permute_idx(a1);
    check_permute_idx(a2);
    check_permute_idx(a3);
    check_permute_idx(a4);
    check_permute_idx(a5);
    check_permute_idx(a6);
    check_permute_idx(a7);
    check_permute_idx(a8);
    if (!is_simple_int_dtype(value.dtype())) {
        invalid_type("Byte permute input must be one of the i8,i16,i32,i64 types. Got %s",
                     show_scalar_dtype(value.dtype()));
    }
    uint64_t permute_idxs = combine_i8_to_i64(a1, a2, a3, a4, a5, a6, a7, a8);
    return permute(value, permute_idxs, false);
}

Value FunctionBuilder::permute_i64_i16(Value value, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4) {
    if (value.dtype() != I64) {
        invalid_type("permute_i64_i16 input type is not i64 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1 * 2, a1 * 2 + 1, a2 * 2, a2 * 2 + 1, a3 * 2, a3 * 2 + 1, a4 * 2, a4 * 2 + 1);
}

Value FunctionBuilder::permute_i64_i32(Value value, uint8_t a1, uint8_t a2) {
    if (value.dtype() != I64) {
        invalid_type("permute_i64_i32 input type is not i64 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1 * 4, a1 * 4 + 1, a1 * 4 + 2, a1 * 4 + 3, a2 * 4, a2 * 4 + 1, a2 * 4 + 2,
                          a2 * 4 + 3);
}

Value FunctionBuilder::permute_i32_i8(Value value, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4) {
    if (value.dtype() != I32) {
        invalid_type("permute_i32_i8 input type is not i32 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1, a2, a3, a4, a1 + 4, a2 + 4, a3 + 4, a4 + 4);
}

Value FunctionBuilder::permute_i32_i16(Value value, uint8_t a1, uint8_t a2) {
    if (value.dtype() != I32) {
        invalid_type("permute_i32_i16 input type is not i32 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1 * 2, a1 * 2 + 1, a2 * 2, a2 * 2 + 1, a1 * 2 + 4, a1 * 2 + 5, a2 * 2 + 4,
                          a2 * 2 + 5);
}

Value FunctionBuilder::permute_i16_i8(Value value, uint8_t a1, uint8_t a2) {
    if (value.dtype() != I16) {
        invalid_type("permute_i16_i8 input type is not i16 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1, a2, a1 + 2, a2 + 2, a1 + 4, a2 + 4, a1 + 6, a2 + 6);
}

Value FunctionBuilder::permute_i8_bits(Value value, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5,
                                       uint8_t a6, uint8_t a7, uint8_t a8) {
    auto transformidx_ = [](uint8_t idx) -> uint8_t {
        if (idx == (uint8_t)-1) { return 0; }
        idx = idx + 1;
        if (idx > 8) { invalid_input("Permute bit index can't be larger than 8"); }
        return idx;
    };
    a1 = transformidx_(a1);
    a2 = transformidx_(a2);
    a3 = transformidx_(a3);
    a4 = transformidx_(a4);
    a5 = transformidx_(a5);
    a6 = transformidx_(a6);
    a7 = transformidx_(a7);
    a8 = transformidx_(a8);
    if (!is_simple_int_dtype(value.dtype())) {
        invalid_type("Bit permute input must be one of the i8,i16,i32,i64 types. Got %s",
                     show_scalar_dtype(value.dtype()));
    }
    uint64_t permute_idxs = combine_i8_to_i64(a1, a2, a3, a4, a5, a6, a7, a8);
    return permute(value, permute_idxs, true);
}

Function *FunctionBuilder::build() {
    MemoryArena *arena = impl_->ctx->arena;

    if (impl_->step_roots.empty()) { invalid_input("Malformed function with no expressions"); }
    check_hir_root_limit(impl_);
    (void)check_hir_complexity(impl_);

    Function *func = arena->create<Function>();
    func->ctx = impl_->ctx;
    func->step_id_count = impl_->sm.max_id();
    func->args = arena->copy_array<ArgumentDecl>(impl_->args);
    func->accs = arena->copy_array<Accumulator>(impl_->accs);
    func->step_roots = arena->copy_array<Step *>(impl_->step_roots);
    func->safety_check_arg = impl_->safety_check_arg;
    func->special_ops = impl_->special_ops;
    if (bool(impl_->ctx->transformations & CodeTransformations::I128Rewrite)) { rewrite_i128_hir(impl_, func); }
    validate_i128_hir(func);
    analyze_argument_usage(func);
    func->vectorization_hint = impl_->vectorization_hint;
    if (func->vectorization_hint == VectorizationHint::None &&
        func->unsupported_vector_special_ops() != SpecialOp::None) {
        func->vectorization_hint = VectorizationHint::UnsupportedSpecialOps;
    }

    return func;
}

Value FunctionBuilder::signed_cast(Value value, ScalarDataType to) {
    ScalarDataType from = value.dtype();
    if (is_float_dtype(from) || is_float_dtype(to)) { return float_cast(value, to); }
    if (scalar_dtype_size(from) < scalar_dtype_size(to)) { return sext(value, to); }
    return trunc(value, to);
}

Value FunctionBuilder::unsigned_cast(Value value, ScalarDataType to) {
    ScalarDataType from = value.dtype();
    if (is_float_dtype(from) || is_float_dtype(to)) { return float_cast(value, to, true); }
    if (scalar_dtype_size(from) < scalar_dtype_size(to)) { return zext(value, to); }
    return trunc(value, to);
}

static void check_int_bit_helper_arg(ScalarDataType dtype, const char *func_name) {
    if (!is_int_dtype(dtype) || dtype == I1) {
        invalid_type("%s argument must be one of the i8,i16,i32,i64,i128 types. Got %s", func_name,
                     show_scalar_dtype(dtype));
    }
}

Value FunctionBuilder::log2_no_zero(Value value) {
    ScalarDataType dtype = value.dtype();
    check_int_bit_helper_arg(dtype, "log2_no_zero");
    size_t bits = scalar_dtype_bits(dtype) - 1;
    return sub(con((int64_t)bits, dtype), lzcnt(value));
}

Value FunctionBuilder::log2(Value value) {
    ScalarDataType dtype = value.dtype();
    check_int_bit_helper_arg(dtype, "log2");
    auto is_predicate_notzero = cmp_ne(value, con(0, dtype));
    size_t bits = scalar_dtype_bits(dtype) - 1;
    return zero_select(sub(con((int64_t)bits, dtype), lzcnt(value)), is_predicate_notzero);
}

Predicate FunctionBuilder::has_single_bit(Value value) {
    check_int_bit_helper_arg(value.dtype(), "has_single_bit");
    return bit_testn(value, sub(value, con(1, value.dtype())));
}

Value FunctionBuilder::byteswap(Value value) {
    switch (value.dtype()) {
    case I8: invalid_type("can't byteswap i8");
    case I16: return permute_i16_i8(value, 1, 0);
    case I32: return permute_i32_i8(value, 3, 2, 1, 0);
    case I64: return permute_i64_i8(value, 7, 6, 5, 4, 3, 2, 1, 0);
    case I1:
    case I128: invalid_type("invalid input type %s to byteswap", show_scalar_dtype(value.dtype()));
    case F32:
    case F64: invalid_type("can't byteswap float type %s", show_scalar_dtype(value.dtype()));
    }
    SIMJIT_UNREACHABLE();
}

Value FunctionBuilder::bit_floor(Value value) {
    ScalarDataType dtype = value.dtype();
    check_int_bit_helper_arg(dtype, "bit_floor");
    return zero_select(sll(con(1, dtype), sub(con((int64_t)scalar_dtype_bits(dtype) - 1, dtype), lzcnt(value))),
                       cmp_ne(value, con(0, dtype)));
}

Value FunctionBuilder::bit_ceil(Value value) {
    ScalarDataType dtype = value.dtype();
    check_int_bit_helper_arg(dtype, "bit_ceil");
    // log2
    Value x = sub(con((int64_t)scalar_dtype_bits(dtype), dtype), lzcnt(sub(value, con(1, dtype))));
    return select(cmp_ule(value, con(1, dtype)), con(1, dtype), sll(con(1, dtype), x));
}

Value FunctionBuilder::sign_no_zero(Value value) {
    ScalarDataType dtype = value.dtype();
    if (is_float_dtype(dtype)) {
        // Really nothing special in float case
        auto is_neg = is_negative(value);
        auto one = con(1, dtype);
        auto mone = con(-1, dtype);
        return select(is_neg, mone, one);
    }
    size_t bits = scalar_dtype_bits(dtype) - 1;
    auto neg = sra(value, con((int64_t)bits, dtype));
    return or_(neg, con(1, dtype));
}

Value FunctionBuilder::sign(Value value) {
    ScalarDataType dtype = value.dtype();
    if (is_float_dtype(dtype)) {
        // Really nothing special in float case
        auto is_neg = is_negative(value);
        auto is_predicate_notzero = cmp_ne(value, con(0, dtype));
        auto one = con(1, dtype);
        auto mone = con(-1, dtype);
        return select(is_neg, mone, zero_select(one, is_predicate_notzero));
    }
    size_t bits = scalar_dtype_bits(dtype) - 1;
    auto is_zero = cmp_ne(value, con(0, dtype));
    auto neg = sra(value, con((int64_t)bits, dtype));
    return zero_select(or_(neg, con(1, dtype)), is_zero);
}

Value FunctionBuilder::copysign_no_zero(Value sign_value, Value value) {
    return select(is_negative(sign_value), negate(value), value);
}

Value FunctionBuilder::copysign(Value sign_value, Value value) {
    auto is_nonzero = cmp_ne(sign_value, con(0, sign_value.dtype()));
    auto a_neg = select(is_negative(sign_value), negate(value), value);
    return zero_select(a_neg, is_nonzero);
}

void FunctionBuilder::scalar_only() noexcept {
    impl_->vectorization_hint = VectorizationHint::ScalarOnly;
}

std::string ErrorInfo::verbose() const {
    std::string result{};
    result.reserve(128);
#if SIMJIT_ERROR_SOURCE_LOCATION
    result += file;
    simjit::format_to(result, ":%d ", line);
#endif
    result += "module=";
    simjit::format_to(result, "0x%llx", (unsigned long long)module);
    result += " kind=";
    simjit::format_to(result, "0x%llx", (unsigned long long)kind);
    result += " subkind=";
    simjit::format_to(result, "0x%llx", (unsigned long long)subkind);
    result += ": ";
    result += message;
    return result;
}

} // namespace simjit
