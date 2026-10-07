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
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#define messed_up(...) simjit_messed_up_m(ErrorModule::HIR, __VA_ARGS__)

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

static bool is_valid_value(MaybeValue x) noexcept {
    return x.is_valid() && x.value().dtype() != ScalarDataType::I1;
}

static bool is_valid_predicate(MaybePredicate x) noexcept {
    return x.is_valid() && x.value().dtype() == ScalarDataType::I1;
}

static bool is_compact_float_cast_int_dtype(ScalarDataType dtype) noexcept {
    return dtype == ScalarDataType::I32 || dtype == ScalarDataType::I64;
}

static ScalarDataType int_dtype_for_float(ScalarDataType dtype) noexcept {
    SIMJIT_ASSERT(is_float_dtype(dtype));
    return scalar_dtype_size(dtype) == 4 ? ScalarDataType::I32 : ScalarDataType::I64;
}

static ScalarDataType float_dtype_for_int(ScalarDataType dtype) noexcept {
    SIMJIT_ASSERT(is_simple_int_dtype(dtype));
    return scalar_dtype_size(dtype) <= 4 ? ScalarDataType::F32 : ScalarDataType::F64;
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
    if (dtype == ScalarDataType::I8 || dtype == ScalarDataType::I16) {
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
    return dtype == ScalarDataType::I8 || dtype == ScalarDataType::I16 ? SpecialOp::SmallPopcount
                                                                       : SpecialOp::LargePopcount;
}

static SpecialOp arith_binary_special_ops(const Step *step) noexcept {
    if (!step->is(StepKind::ArithBinary)) { return SpecialOp::None; }

    SpecialOp result = SpecialOp::None;
    const auto &data = step->step_data<StepKind::ArithBinary>();
    if (step->dtype == ScalarDataType::I64 && data.op == ArithBinaryOp::Mul) { result |= SpecialOp::I64Mul; }
    // x86 i8 constant shifts and rotates use the GFNI permute peephole in MIR.
    if (step->dtype == ScalarDataType::I8 && is_i8_constant_shift_or_rotate(data.op, data.right)) {
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
    if (data.op == ArithUnaryOp::Lzcnt && (step->dtype == ScalarDataType::I8 || step->dtype == ScalarDataType::I16)) {
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
    if (!is_simple_int_dtype(dtype)) { return false; }
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
    if (data.dtype == ScalarDataType::I1) {
        return data.as_unsigned() == 0 ? ConstClass::BoolFalse : ConstClass::BoolTrue;
    }
    if (data.dtype != ScalarDataType::I128 && data.is_zero()) { return ConstClass::Zero; }
    switch (data.dtype) {
    case ScalarDataType::I1: SIMJIT_UNREACHABLE();
    case ScalarDataType::I8:
    case ScalarDataType::I16:
    case ScalarDataType::I32:
    case ScalarDataType::I64:
        if (data.as_unsigned() == 1) return ConstClass::One;
        if (data.is_all_ones()) return ConstClass::AllOnes;
        return ConstClass::Other;
    case ScalarDataType::F32:
        if (data.as_f32_bits() == ConstData::f32(1.0f).as_f32_bits()) return ConstClass::One;
        if (data.as_f32_bits() == UINT32_MAX) return ConstClass::AllOnes;
        return ConstClass::Other;
    case ScalarDataType::F64:
        if (data.as_f64_bits() == ConstData::f64(1.0).as_f64_bits()) return ConstClass::One;
        if (data.as_f64_bits() == UINT64_MAX) return ConstClass::AllOnes;
        return ConstClass::Other;
    case ScalarDataType::I128: return ConstClass::Other;
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
        if (dtype == ScalarDataType::I128) {
            SIMJIT_ASSERT(arg->dtype == ScalarDataType::I64);
            return sm.sum128({op, arg, acc.idx, cond}, dtype);
        }
        SIMJIT_ASSERT(arg->dtype == dtype);
        return sm.acc_arith_bin({op, arg, acc.idx, cond}, dtype);
    }

    Step *countif(Step *arg, const Accumulator &acc) {
        return sm.countif({ArithBinaryOp::Add, arg, acc.idx}, acc.dtype);
    }

    Step *predicate_agg(Step *arg, PredicateBinaryOp op, const Accumulator &acc) {
        SIMJIT_ASSERT(arg->dtype == ScalarDataType::I1);
        SIMJIT_ASSERT(acc.dtype == ScalarDataType::I1);
        return sm.acc_predicate_bin({op, arg, acc.idx}, ScalarDataType::I1);
    }
    Step *scatter(Step *arg, Step *idx, Step *cond, const ArgumentDecl &dst) {
        cond = simplify_optional_cond(ctx, cond);
        ScalarDataType dtype = arg->dtype;
        SIMJIT_ASSERT(dst.dtype == dtype);
        return sm.scatter({arg, idx, dst.idx, cond}, dtype);
    }
    Step *pack(Step *arg, Step *cond, const ArgumentDecl &dst, const Accumulator &dst_size_acc) {
        ScalarDataType dtype = arg->dtype;
        SIMJIT_ASSERT(dst.dtype == dtype);
        SIMJIT_ASSERT(dst_size_acc.dtype == ScalarDataType::I64);
        return sm.pack({arg, cond, dst.idx, dst_size_acc.idx}, dtype);
    }
    Step *store(Step *arg, Step *cond, const ArgumentDecl &dst, LoadStoreKind kind) {
        cond = simplify_optional_cond(ctx, cond);
        ScalarDataType dtype = arg->dtype;
        SIMJIT_ASSERT(dst.dtype == dtype);
        return sm.store({arg, dst.idx, kind, cond}, dtype);
    }

    Step *load(const ArgumentDecl &arg, LoadStoreKind kind) { return sm.load({arg.idx, kind}, arg.dtype); }

    Step *gather(Step *idx, const ArgumentDecl &dst) { return sm.gather({idx, dst.idx}, dst.dtype); }

    Step *con(ConstData data, ScalarDataType dtype) { return sm.con(data, dtype); }

    Step *load_splat(const ArgumentDecl &arg) { return sm.load_splat({arg.idx, LoadStoreKind::Unaligned}, arg.dtype); }

    Step *zero_const(ScalarDataType dtype) {
        switch (dtype) {
        case ScalarDataType::I1:
        case ScalarDataType::I8:
        case ScalarDataType::I16:
        case ScalarDataType::I32:
        case ScalarDataType::I64:
        case ScalarDataType::F32:
        case ScalarDataType::F64: return sm.con(ConstData::u64(0).retag(dtype), dtype);
        case ScalarDataType::I128: return nullptr;
        }
        SIMJIT_UNREACHABLE();
    }

    Step *bool_const(bool value) { return sm.con(ConstData::i1(value), ScalarDataType::I1); }

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
            if (is_simple_int_dtype(left->dtype)) {
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
        if (is_simple_int_dtype(left->dtype) && left == right) {
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
        if (is_unsigned && is_simple_int_dtype(left->dtype)) {
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
        if ((op == ArithBinaryOp::Div || op == ArithBinaryOp::UDiv || op == ArithBinaryOp::Mod ||
             op == ArithBinaryOp::UMod) &&
            is_simple_int_dtype(left->dtype)) {
#if SIMJIT_USE_LIBDIVIDE
            {
                bool vector_const_div = false;
                if (bool(ctx->transformations & CodeTransformations::ConstDiv) && right->is(StepKind::Const) &&
                    (left->dtype == ScalarDataType::I16 || left->dtype == ScalarDataType::I32 ||
                     left->dtype == ScalarDataType::I64)) {
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

        // Ad-hoc strength reduction for mul
        if (op == ArithBinaryOp::Mul && is_simple_int_dtype(left->dtype)) {
            if (bool(ctx->transformations & CodeTransformations::MulConstPeephole)) {
                if (Step *peepholed = int_mul_peephole(left, right)) { return peepholed; }
            }
            // Change instruction if both of its arguments are 32-bit
            if (bool(ctx->transformations & CodeTransformations::MuldqInst)) {
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
    Step *fpclass(Step *arg, FpClass flags) { return sm.fpclass({flags, arg}, ScalarDataType::I1); }
    Step *cmp(Step *left, Step *right, CmpOp op, bool is_unsigned, Step **original) {
        SIMJIT_ASSERT(left->dtype == right->dtype);
        *original = sm.cmp({op, left, right, is_unsigned}, ScalarDataType::I1);
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
        SIMJIT_ASSERT(cond->dtype == ScalarDataType::I1);
        *original = sm.select({cond, truthy, falsy}, falsy->dtype);
        if (Step *result = select_peephole(cond, truthy, falsy)) { return result; }
        return *original;
    }
    Step *index(ScalarDataType dtype) { return sm.index({}, dtype); }

    Step *byte_permute(Step *arg, uint64_t permute) {
        SIMJIT_ASSERT(arg->dtype != ScalarDataType::I1 && arg->dtype != ScalarDataType::I128);
        return sm.permute({arg, permute, false}, arg->dtype);
    }

    Step *bit_permute(Step *arg, uint64_t permute) {
        SIMJIT_ASSERT(arg->dtype != ScalarDataType::I1 && arg->dtype != ScalarDataType::I128);
        return sm.permute({arg, permute, true}, arg->dtype);
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

        auto make_con = [&](uint64_t x) { return sm.con(ConstData::u64(x).retag(dtype), dtype); };
        size_t bit_count = scalar_dtype_bits(dtype);

        auto multiply_pos = [&](uint64_t n, bool is_neg) -> Step * {
            SIMJIT_ASSERT(n != 0);
            // Optimizer should have handled it, but whatever
            if (n == 1) { return arg; }
            if (n == 2) { return sm.arith_bin({ArithBinaryOp::Add, arg, arg}, dtype); }
            if (has_single_bit(n)) {
                size_t count = nonzero_log2(n);
                SIMJIT_ASSERT(count < bit_count);
                return sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(count)}, dtype);
            }
            // 2^n + 1 => (x << n) + x
            if (has_single_bit(n - 1)) {
                size_t count = nonzero_log2(n - 1);
                SIMJIT_ASSERT(count < bit_count);
                Step *tmp = sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(count)}, dtype);
                return sm.arith_bin({ArithBinaryOp::Add, tmp, arg}, dtype);
            }
            // 2^n - 1 => (x << n) - x
            if (has_single_bit(n + 1)) {
                size_t count = nonzero_log2(n + 1);
                if (count >= bit_count) { return nullptr; }
                SIMJIT_ASSERT(count < bit_count);
                Step *tmp = sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(count)}, dtype);
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
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(first_shift)}, dtype);
                Step *second_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(second_shift)}, dtype);
                return sm.arith_bin({ArithBinaryOp::Add, first_shift_s, second_shift_s}, dtype);
            }
            // 2^n - 2 => (x << n) - x - x
            if (has_single_bit(n + 2)) {
                size_t count = nonzero_log2(n + 2);
                if (count >= bit_count) { return nullptr; }
                SIMJIT_ASSERT(count < bit_count);
                Step *shifted = sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(count)}, dtype);
                Step *tmp = sm.arith_bin({ArithBinaryOp::Add, arg, arg}, dtype);
                return sm.arith_bin({ArithBinaryOp::Sub, shifted, tmp}, dtype);
            }
            // 2^n + 2^m + 1 => (x << n) + (x << m) + x
            if (dtype == ScalarDataType::I64 && popcount(n & ~1llu) == 2 && (n & 1llu) == 1llu) {
                size_t first_shift = nonzero_log2(n & ~1llu);
                size_t second_shift = nonzero_log2((n & ~1llu) & ~(1llu << first_shift));
                SIMJIT_ASSERT(((1llu << first_shift) | (1llu << second_shift) | 1llu) == n);
                SIMJIT_ASSERT(first_shift < bit_count);
                SIMJIT_ASSERT(second_shift < bit_count);
                Step *first_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(first_shift)}, dtype);
                Step *second_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(second_shift)}, dtype);
                Step *tmp = sm.arith_bin({ArithBinaryOp::Add, first_shift_s, second_shift_s}, dtype);
                return sm.arith_bin({ArithBinaryOp::Add, tmp, arg}, dtype);
            }
            // 2^n + 2^m + 2^p => (x << n) + (x << m) + (x << p)
            if (dtype == ScalarDataType::I64 && popcount(n) == 3) {
                size_t first_shift = nonzero_log2(n);
                size_t second_shift = nonzero_log2(n & ~(1llu << first_shift));
                size_t third_shift = nonzero_log2(n & (~(1llu << first_shift) & ~(1llu << second_shift)));

                SIMJIT_ASSERT(((1llu << first_shift) | (1llu << second_shift) | (1llu << third_shift)) == n);
                SIMJIT_ASSERT(first_shift < bit_count);
                SIMJIT_ASSERT(second_shift < bit_count);
                SIMJIT_ASSERT(third_shift < bit_count);
                Step *first_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(first_shift)}, dtype);
                Step *second_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(second_shift)}, dtype);
                Step *first_sum_s = sm.arith_bin({ArithBinaryOp::Add, first_shift_s, second_shift_s}, dtype);
                Step *third_shift_s =
                    sm.arith_bin({ArithBinaryOp::ShiftLeftLogical, arg, make_con(third_shift)}, dtype);
                return sm.arith_bin({ArithBinaryOp::Add, first_sum_s, third_shift_s}, dtype);
            }
            return nullptr;
        };

        if (const_v > 0) { return multiply_pos(const_v, false); }
        if (const_v < 0) {
            if (const_v == -1) { return sm.arith_un({ArithUnaryOp::Negate, arg}, left->dtype); }
            if (const_v == std::numeric_limits<int64_t>::min()) { return nullptr; }
            uint64_t mn = -const_v;
            if (Step *last = multiply_pos(mn, true)) {
                // If the last instruction was subtraction, we can swap operands places
                if (last->is(StepKind::ArithBinary) &&
                    last->step_data<StepKind::ArithBinary>().op == ArithBinaryOp::Sub) {
                    auto &dat = last->step_data<StepKind::ArithBinary>();
                    std::swap(dat.left, dat.right);
                } else {
                    last = sm.arith_un({ArithUnaryOp::Negate, last}, dtype);
                }
                return last;
            }
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
        if (left->dtype != ScalarDataType::I64) return {};
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
        return sm.cmp({op, right, left, is_unsigned}, ScalarDataType::I1);
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

            return sm.cmp({new_op, compared1, bound, true}, ScalarDataType::I1);
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
            return sm.cmp({CmpOp::LessEqual, adjusted, range_step, true}, ScalarDataType::I1);
        };
        auto try_use_andnot = [&](Step *left, Step *right) -> Step * {
            // Two NOT in AND is very strange, but easy to handle
            if (left->is(StepKind::PredicateNot) && right->is(StepKind::PredicateNot)) {
                // Just de-morgan
                Step *left_step = left->step_data<StepKind::PredicateNot>();
                Step *right_step = right->step_data<StepKind::PredicateNot>();
                return sm.predicate_not(
                    sm.predicate_bin({left_step, right_step, PredicateBinaryOp::Or}, ScalarDataType::I1),
                    ScalarDataType::I1);
            }
            if (left->is(StepKind::PredicateNot)) {
                return sm.predicate_bin({left->step_data<StepKind::PredicateNot>(), right, PredicateBinaryOp::AndNot},
                                        ScalarDataType::I1);
            }
            if (right->is(StepKind::PredicateNot)) {
                return sm.predicate_bin({right->step_data<StepKind::PredicateNot>(), left, PredicateBinaryOp::AndNot},
                                        ScalarDataType::I1);
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
            return sm.predicate_not(
                sm.predicate_bin({left_step, right_step, PredicateBinaryOp::And}, ScalarDataType::I1),
                ScalarDataType::I1);
        }

        // !A | B = !(A & !B)
        if (left->is(StepKind::PredicateNot)) {
            // Just de-morgan
            Step *left_step = left->step_data<StepKind::PredicateNot>();
            return sm.predicate_not(sm.predicate_bin({right, left_step, PredicateBinaryOp::AndNot}, ScalarDataType::I1),
                                    ScalarDataType::I1);
        }
        if (right->is(StepKind::PredicateNot)) {
            // Just de-morgan
            Step *right_step = right->step_data<StepKind::PredicateNot>();
            return sm.predicate_not(sm.predicate_bin({left, right_step, PredicateBinaryOp::AndNot}, ScalarDataType::I1),
                                    ScalarDataType::I1);
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
                    {right->step_data<StepKind::PredicateNot>(), left_step, PredicateBinaryOp::AndNot},
                    ScalarDataType::I1);
            }
            return sm.predicate_bin({left_step, right, PredicateBinaryOp::And}, ScalarDataType::I1);
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
    Argument x = arg(ScalarDataType::I8);
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
    if (size_arg.dtype != ScalarDataType::I64) { invalid_type("Pack result size must be i64"); }
    if (value.dtype() == ScalarDataType::I8 || value.dtype() == ScalarDataType::I16) {
        impl_->special_ops |= SpecialOp::SmallPack;
    }
    impl_->accs.push_back(Accumulator{ScalarDataType::I64, AccIdx{impl_->accs.size()}, dst_size.idx_, nullptr});
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
    if (func_arg.dtype == ScalarDataType::I8 || func_arg.dtype == ScalarDataType::I16) {
        // no SIMD ISA supports scatter for i8/i16.
        scalar_only();
    }
    Step *cond_step = simplify_optional_cond(impl_->ctx, cond.step_);
    impl_->special_ops |= cond_step != nullptr ? SpecialOp::CondScatter : SpecialOp::Scatter;
    check_store_dst(func_arg, value.step_, StepKind::Scatter);
    if (idx.dtype() == ScalarDataType::I8 || idx.dtype() == ScalarDataType::I16) {
        idx = zext(idx, ScalarDataType::I32);
    }

    auto *step = impl_->scatter(value.step_, idx.step_, cond_step, func_arg);
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
    if (dst_info.dtype == ScalarDataType::I128) {
        if (value.dtype() != ScalarDataType::I64) {
            unsupported("Only support calculating 128-bit sum of i64 values, got %s", show_scalar_dtype(value.dtype()));
        }
        if (op != ArithBinaryOp::Add) { unsupported("Don't support 128-bit %s agg", show_arith_binary_op(op)); }
    } else {
        check_store_dst(dst_info, value.step_, StepKind::AccArithBinary);
    }

    if (op == ArithBinaryOp::Mul) {
        if (value.dtype() == ScalarDataType::I64) { impl_->special_ops |= SpecialOp::I64Mul; }
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
    if (func_arg.dtype == ScalarDataType::I1) { invalid_type("load should not be used with i1 type"); }

    uint64_t hash = make_hash(StepKind::Load, argument, kind);
    if (auto *result = impl_->cse<StepKind::Load>(
            hash, [argument, kind](const auto &x) { return x.idx == argument.idx_ && x.kind == kind; })) {
        return result;
    }

    Step *s = impl_->load(func_arg, kind);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::load_predicate(Argument argument) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype != ScalarDataType::I1) { invalid_type("load_predicate should use I1 dtype"); }

    uint64_t hash = make_hash(StepKind::Load, argument);
    if (auto *result = impl_->cse<StepKind::Load>(hash, [argument](const auto &x) { return x.idx == argument.idx_; })) {
        return result;
    }

    Step *s = impl_->load(func_arg, LoadStoreKind::Unaligned);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::gather(Value idx, Argument argument) {
    check_value(idx);

    if (!is_simple_int_dtype(idx.dtype())) {
        invalid_type("Only i8, i16, i32 and i64 indices are supported in gather. Got %s",
                     show_scalar_dtype(idx.dtype()));
    }
    if (idx.dtype() == ScalarDataType::I8 || idx.dtype() == ScalarDataType::I16) {
        idx = zext(idx, ScalarDataType::I32);
    }
    uint64_t hash = make_hash(StepKind::Gather, argument, idx);
    if (auto *result = impl_->cse<StepKind::Gather>(
            hash, [idx, argument](const auto &x) { return x.idx == idx.step_ && x.data == argument.idx_; })) {
        return result;
    }
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype == ScalarDataType::I8 || func_arg.dtype == ScalarDataType::I16) {
        impl_->special_ops |= SpecialOp::SmallGather;
    }
    impl_->special_ops |= SpecialOp::Gather;
    Step *s = impl_->gather(idx.step_, func_arg);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::load_splat(Argument argument) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype == ScalarDataType::I1) { invalid_type("load should not be used with i1 type"); }

    uint64_t hash = make_hash(StepKind::LoadSplat, argument);
    if (auto *result =
            impl_->cse<StepKind::LoadSplat>(hash, [argument](const auto &x) { return x.idx == argument.idx_; })) {
        return result;
    }
    Step *s = impl_->load_splat(func_arg);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::load_predicate_splat(Argument argument) {
    const ArgumentDecl &func_arg = impl_->get_arg(argument.idx_);
    if (func_arg.dtype != ScalarDataType::I1) { invalid_type("load_predicate_splat should use I1 dtype"); }

    uint64_t hash = make_hash(StepKind::LoadSplat, argument);
    if (auto *result =
            impl_->cse<StepKind::LoadSplat>(hash, [argument](const auto &x) { return x.idx == argument.idx_; })) {
        return result;
    }
    Step *s = impl_->load_splat(func_arg);
    return impl_->remember_expr(hash, s);
}

Value FunctionBuilder::con_internal(ConstData data, ScalarDataType dtype) {
    if (is_float_dtype(dtype)) {
    } else if (!(data.as_signed() >= scalar_dtype_min(dtype).as_signed() &&
                 data.as_signed() <= scalar_dtype_max(dtype).as_signed()) &&
               !(data.as_unsigned() <= scalar_dtype_umax(dtype))) {
        invalid_type("Constant value %lld (unsigned %llu) does not fit into allowed range for type %s",
                     (long long)data.as_signed(), (unsigned long long)data.as_unsigned(), show_scalar_dtype(dtype));
    }
    data = data.retag(dtype);

    uint64_t hash = make_hash(StepKind::Const, data.raw_bits(), dtype);
    if (auto *result = impl_->cse<StepKind::Const>(hash, dtype, [data](const auto &x) { return x == data; })) {
        return result;
    }
    Step *s = impl_->con(data, dtype);
    return impl_->remember_expr(hash, s);
}

Predicate FunctionBuilder::i1(bool data) {
    ScalarDataType dtype = ScalarDataType::I1;
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

    bool safety_check = bool(flags & ArithBinaryOpFlags::SafetyCheck);
    if (bool(flags & ArithBinaryOpFlags::SafeDivision)) {
        if (!is_division_op) {
            invalid_input("Safe division flag is not supported for binary arithmetic operation %s",
                          show_arith_binary_op(op));
        }
        if (!is_simple_int_dtype(left.dtype()) && !is_float_dtype(left.dtype())) {
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
        bool supported_type = value.dtype() == ScalarDataType::I32 || value.dtype() == ScalarDataType::I64 ||
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
    } else if (operation->is(StepKind::ArithUnary)) {
        auto &data = operation->step_data<StepKind::ArithUnary>();
        if (data.op != ArithUnaryOp::Abs && data.op != ArithUnaryOp::Negate) {
            invalid_input("safety checking is not supported for unary arithmetic operation %s",
                          show_arith_unary_op(data.op));
        }
        if (!is_simple_int_dtype(value.dtype())) {
            unsupported("safety checking is not supported for op %s type %s; Only integer types are supported",
                        show_arith_unary_op(data.op), show_scalar_dtype(value.dtype()));
        }
    } else {
        invalid_input("Unsupported checked operation %s", show_step_kind(operation->kind));
    }

    impl_->ensure_has_safety_check_arg();

    Step *mask_step = mask.is_valid() ? mask.value().step_ : nullptr;
    mask_step = simplify_optional_cond(impl_->ctx, mask_step);
    uint64_t hash = make_hash(StepKind::CheckedOp, operation, mask_step);
    if (auto *result = impl_->cse<StepKind::CheckedOp>(
            hash, [operation, mask_step](const auto &x) { return x.op == operation && x.mask == mask_step; })) {
        return result;
    }
    impl_->special_ops |= arith_binary_special_ops(operation);
    Step *result = impl_->sm.checked_op({operation, mask_step}, value.dtype());
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
        if (!(is_simple_int_dtype(value.dtype()) || is_float_dtype(value.dtype()))) {
            invalid_type("Invalid unary arithmetic operation %s for type %s", show_arith_unary_op(op),
                         show_scalar_dtype(value.dtype()));
        }
        break;
    case ArithUnaryOp::Lzcnt:
    case ArithUnaryOp::Tzcnt:
    case ArithUnaryOp::Popcount:
        if (!is_simple_int_dtype(value.dtype())) {
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
    if (dtype != ScalarDataType::I64) {
        invalid_type("Only support calculating 64-bit countif, got %s", show_scalar_dtype(dtype));
    }

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
    if (!is_simple_int_dtype(value.dtype()) || !is_simple_int_dtype(dtype)) {
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
    bool valid_decomposed = bool(impl_->ctx->transformations & CodeTransformations::CastDecomposition) &&
                            ((is_simple_int_dtype(value.dtype()) && is_float_dtype(dtype)) ||
                             (is_simple_int_dtype(dtype) && is_float_dtype(value.dtype())));
    if (!valid_compact && !valid_decomposed) {
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

    if (!((value.dtype() == ScalarDataType::F32 && dtype == ScalarDataType::I32) ||
          (value.dtype() == ScalarDataType::F64 && dtype == ScalarDataType::I64) ||
          (value.dtype() == ScalarDataType::I32 && dtype == ScalarDataType::F32) ||
          (value.dtype() == ScalarDataType::I64 && dtype == ScalarDataType::F64))) {
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
    if (value.dtype() != ScalarDataType::I64) {
        invalid_type("permute_i64_i16 input type is not i64 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1 * 2, a1 * 2 + 1, a2 * 2, a2 * 2 + 1, a3 * 2, a3 * 2 + 1, a4 * 2, a4 * 2 + 1);
}

Value FunctionBuilder::permute_i64_i32(Value value, uint8_t a1, uint8_t a2) {
    if (value.dtype() != ScalarDataType::I64) {
        invalid_type("permute_i64_i32 input type is not i64 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1 * 4, a1 * 4 + 1, a1 * 4 + 2, a1 * 4 + 3, a2 * 4, a2 * 4 + 1, a2 * 4 + 2,
                          a2 * 4 + 3);
}

Value FunctionBuilder::permute_i32_i8(Value value, uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4) {
    if (value.dtype() != ScalarDataType::I32) {
        invalid_type("permute_i32_i8 input type is not i32 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1, a2, a3, a4, a1 + 4, a2 + 4, a3 + 4, a4 + 4);
}

Value FunctionBuilder::permute_i32_i16(Value value, uint8_t a1, uint8_t a2) {
    if (value.dtype() != ScalarDataType::I32) {
        invalid_type("permute_i32_i16 input type is not i32 (%s)", show_scalar_dtype(value.dtype()));
    }
    return permute_i64_i8(value, a1 * 2, a1 * 2 + 1, a2 * 2, a2 * 2 + 1, a1 * 2 + 4, a1 * 2 + 5, a2 * 2 + 4,
                          a2 * 2 + 5);
}

Value FunctionBuilder::permute_i16_i8(Value value, uint8_t a1, uint8_t a2) {
    if (value.dtype() != ScalarDataType::I16) {
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

static void check_simple_int_bit_helper_arg(ScalarDataType dtype, const char *func_name) {
    if (!is_simple_int_dtype(dtype)) {
        invalid_type("%s argument must be one of the i8,i16,i32,i64 types. Got %s", func_name,
                     show_scalar_dtype(dtype));
    }
}

Value FunctionBuilder::log2_no_zero(Value value) {
    ScalarDataType dtype = value.dtype();
    check_simple_int_bit_helper_arg(dtype, "log2_no_zero");
    size_t bits = scalar_dtype_bits(dtype) - 1;
    return sub(con((int64_t)bits, dtype), lzcnt(value));
}

Value FunctionBuilder::log2(Value value) {
    ScalarDataType dtype = value.dtype();
    check_simple_int_bit_helper_arg(dtype, "log2");
    auto is_predicate_notzero = cmp_ne(value, con(0, dtype));
    size_t bits = scalar_dtype_bits(dtype) - 1;
    return zero_select(sub(con((int64_t)bits, dtype), lzcnt(value)), is_predicate_notzero);
}

Predicate FunctionBuilder::has_single_bit(Value value) {
    check_simple_int_bit_helper_arg(value.dtype(), "has_single_bit");
    return bit_testn(value, sub(value, con(1, value.dtype())));
}

Value FunctionBuilder::byteswap(Value value) {
    switch (value.dtype()) {
    case ScalarDataType::I8: invalid_type("can't byteswap i8");
    case ScalarDataType::I16: return permute_i16_i8(value, 1, 0);
    case ScalarDataType::I32: return permute_i32_i8(value, 3, 2, 1, 0);
    case ScalarDataType::I64: return permute_i64_i8(value, 7, 6, 5, 4, 3, 2, 1, 0);
    case ScalarDataType::I1:
    case ScalarDataType::I128: invalid_type("invalid input type %s to byteswap", show_scalar_dtype(value.dtype()));
    case ScalarDataType::F32:
    case ScalarDataType::F64: invalid_type("can't byteswap float type %s", show_scalar_dtype(value.dtype()));
    }
    SIMJIT_UNREACHABLE();
}

Value FunctionBuilder::bit_floor(Value value) {
    ScalarDataType dtype = value.dtype();
    check_simple_int_bit_helper_arg(dtype, "bit_floor");
    return zero_select(sll(con(1, dtype), sub(con((int64_t)scalar_dtype_bits(dtype) - 1, dtype), lzcnt(value))),
                       cmp_ne(value, con(0, dtype)));
}

Value FunctionBuilder::bit_ceil(Value value) {
    ScalarDataType dtype = value.dtype();
    check_simple_int_bit_helper_arg(dtype, "bit_ceil");
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
