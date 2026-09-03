// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/cpp/emitter_internal.h"
#include "simjit/core/cpp/x86_intrin.h"
#include "simjit/core/x86.h"

namespace simjit {
namespace cpp_backend {

#define messed_up(...) simjit_exception(ErrorModule::CPP, {}, {}, __VA_ARGS__)
#define unsupported(...) \
    simjit_exception(ErrorModule::CPP, ErrorKind::Unsupported, ErrorSubKind::UnsupportedBackendFeature, __VA_ARGS__)

static const char *vec_dtype_cpp(x86::Vector vec) {
    switch (vec.reg) {
    case x86::VecRegisterKind::XMM:
        if (vec.dtype == VecElemType::F32) return "__m128";
        if (vec.dtype == VecElemType::F64) return "__m128d";
        return "__m128i";
    case x86::VecRegisterKind::YMM:
        if (vec.dtype == VecElemType::F32) return "__m256";
        if (vec.dtype == VecElemType::F64) return "__m256d";
        return "__m256i";
    case x86::VecRegisterKind::ZMM:
        if (vec.dtype == VecElemType::F32) return "__m512";
        if (vec.dtype == VecElemType::F64) return "__m512d";
        return "__m512i";
    }
    SIMJIT_UNREACHABLE();
}

static const char *vec_dtype_cpp(VecDataType dtype) {
    return vec_dtype_cpp(x86::vec_to_x86(dtype));
}

static const char *cmp_op_to_int_imm(CmpOp op) {
    switch (op) {
    case CmpOp::Less: return "_MM_CMPINT_LT";
    case CmpOp::Greater: return "_MM_CMPINT_NLE";
    case CmpOp::LessEqual: return "_MM_CMPINT_LE";
    case CmpOp::GreaterEqual: return "_MM_CMPINT_NLT";
    case CmpOp::Equal: return "_MM_CMPINT_EQ";
    case CmpOp::NotEqual: return "_MM_CMPINT_NE";
    }
    SIMJIT_UNREACHABLE();
}

static const char *cmp_op_to_fp_imm(CmpOp op) {
    switch (op) {
    case CmpOp::Less: return "_CMP_LT_OQ";
    case CmpOp::Greater: return "_CMP_GT_OQ";
    case CmpOp::LessEqual: return "_CMP_LE_OQ";
    case CmpOp::GreaterEqual: return "_CMP_GE_OQ";
    case CmpOp::Equal: return "_CMP_EQ_OQ";
    case CmpOp::NotEqual: return "_CMP_NEQ_UQ";
    }
    SIMJIT_UNREACHABLE();
}

static const char *mask_dtype_cpp(MaskDataType dtype) {
    switch (dtype) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: return "__mmask8";
    case MaskDataType::M16: return "__mmask16";
    case MaskDataType::M32: return "__mmask32";
    case MaskDataType::M64: return "__mmask64";
    }
    SIMJIT_UNREACHABLE();
}

static const char *dtype_to_cpp(DataType dtype) {
    switch (dtype.kind) {
    case DataTypeKind::Scalar: return scalar_dtype_to_cpp(dtype.scalar);
    case DataTypeKind::Vec: return vec_dtype_cpp(dtype.vec);
    case DataTypeKind::Mask: return mask_dtype_cpp(dtype.mask);
    }
    SIMJIT_UNREACHABLE();
}

static const char *mask_all_ones(MaskDataType dt) {
    switch (dt) {
    case MaskDataType::M2: return "0x03";
    case MaskDataType::M4: return "0x0f";
    case MaskDataType::M8: return "0xff";
    case MaskDataType::M16: return "0xffff";
    case MaskDataType::M32: return "0xffffffff";
    case MaskDataType::M64: return "0xffffffffffffffff";
    }
    SIMJIT_UNREACHABLE();
}

static bool mask_uses_partial_byte(MaskDataType dt) {
    return dt == MaskDataType::M2 || dt == MaskDataType::M4;
}

struct ArgumentRef {
    ArgumentRef() = delete;
    ArgumentRef(const char *v) : var(v) {}
    ArgumentRef(std::string v) : var(std::move(v)) {}
    ArgumentRef(const Step *s) : var(s) {}

    const char *to_str(nonstd::span<std::string const> map) const {
        if (std::holds_alternative<const char *>(var)) { return std::get<const char *>(var); }
        if (std::holds_alternative<std::string>(var)) { return std::get<std::string>(var).c_str(); }
        const Step *step = std::get<const Step *>(var);
        if (map[step->id].empty()) {
            messed_up("step %s (id=%zu) does not have an assigned C++ name", show_step_kind(step->kind),
                      size_t(step->id));
        }
        return map[step->id].c_str();
    }

    std::variant<const char *, std::string, const Step *> var;
};

struct IntrinsicCaller {
    IntrinsicCaller() = delete;
    IntrinsicCaller(const IntrinsicCaller &) = delete;
    IntrinsicCaller(IntrinsicCaller &&) = delete;
    IntrinsicCaller &operator=(const IntrinsicCaller &) = delete;
    IntrinsicCaller &operator=(IntrinsicCaller &&) = delete;
    IntrinsicCaller(const x86::Intrinsic &x, nonstd::span<std::string const> m) : i(x), map(m) {}

    std::string call() const { return format("%s()", i.name()); }
    std::string call(const ArgumentRef &a) const { return format("%s(%s)", i.name(), a.to_str(map)); }
    std::string call(const ArgumentRef &a, const ArgumentRef &b) const {
        return format("%s(%s, %s)", i.name(), a.to_str(map), b.to_str(map));
    }
    std::string call(const ArgumentRef &a, const ArgumentRef &b, const ArgumentRef &c) const {
        return format("%s(%s, %s, %s)", i.name(), a.to_str(map), b.to_str(map), c.to_str(map));
    }
    std::string call(const ArgumentRef &a, const ArgumentRef &b, const ArgumentRef &c, const ArgumentRef &d) const {
        return format("%s(%s, %s, %s, %s)", i.name(), a.to_str(map), b.to_str(map), c.to_str(map), d.to_str(map));
    }
    std::string call(const ArgumentRef &a, const ArgumentRef &b, const ArgumentRef &c, const ArgumentRef &d,
                     const ArgumentRef &e) const {
        return format("%s(%s, %s, %s, %s, %s)", i.name(), a.to_str(map), b.to_str(map), c.to_str(map), d.to_str(map),
                      e.to_str(map));
    }

    const x86::Intrinsic &i;
    nonstd::span<std::string const> map;
};

struct X86CppEmitter : CppEmitterBase {
    using CppEmitterBase::CppEmitterBase;

    const char *cpp_dtype(DataType dtype) const override { return dtype_to_cpp(dtype); }

    struct DataTypeRef {
        DataTypeRef() = delete;
        DataTypeRef(const Step *step) : dtype(step->dtype) {}
        DataTypeRef(DataType dt) : dtype(dt) {}
        DataTypeRef(ScalarDataType dt) : dtype(dt) {}
        DataTypeRef(VecDataType dt) : dtype(dt) {}
        DataTypeRef(MaskDataType dt) : dtype(dt) {}

        DataType get() const { return dtype; }

        DataType dtype;
    };

    static void check_return_type(const x86::Intrinsic &in, const Step *caller) {
        DataType step_dtype = caller->dtype;
        if (step_dtype != in.return_dtype()) {
            messed_up("got %s vs expected %s: data type of step and called intrinsic (%s) do not match",
                      show_dtype(step_dtype), show_dtype(in.return_dtype()), in.name());
        }
    }
    static void check_argument_count(const x86::Intrinsic &in, size_t count) {
        if (in.arity() != count) {
            messed_up("intrinsic %s has %zu arguments, but tried to call with %zu", in.name(), in.arity(), count);
        }
    }
    static void check_nth_arg(const x86::Intrinsic &in, DataTypeRef a, size_t nth) {
        DataType test_dtype = a.get();
        DataType arg_dtype = in.arg_dtype(nth);
        if (test_dtype != arg_dtype) {
            messed_up("got %s vs expected %s: invalid %zu argument data type intrinsic %s", show_dtype(test_dtype),
                      show_dtype(arg_dtype), nth, in.name());
        }
    }

    template <typename... Args> std::string unsafe_intrin_call(const x86::Intrinsic &in, Args &&...args) const {
        constexpr size_t arity = sizeof...(Args);
        check_argument_count(in, arity);
        auto caller = IntrinsicCaller{in, seen};
        return caller.call(args...);
    }

    IntrinsicCaller typecheck_intrin_call(const x86::Intrinsic &in, const Step *caller) const {
        check_argument_count(in, 0);
        check_return_type(in, caller);
        return IntrinsicCaller{in, seen};
    }
    IntrinsicCaller typecheck_intrin_call(const x86::Intrinsic &in, const Step *caller, DataTypeRef a) const {
        check_argument_count(in, 1);
        check_nth_arg(in, a, 0);
        check_return_type(in, caller);
        return IntrinsicCaller{in, seen};
    }
    IntrinsicCaller typecheck_intrin_call(const x86::Intrinsic &in, const Step *caller, DataTypeRef a,
                                          DataTypeRef b) const {
        check_argument_count(in, 2);
        check_nth_arg(in, a, 0);
        check_nth_arg(in, b, 1);
        check_return_type(in, caller);
        return IntrinsicCaller{in, seen};
    }
    IntrinsicCaller typecheck_intrin_call(const x86::Intrinsic &in, const Step *caller, DataTypeRef a, DataTypeRef b,
                                          DataTypeRef c) const {
        check_argument_count(in, 3);
        check_nth_arg(in, a, 0);
        check_nth_arg(in, b, 1);
        check_nth_arg(in, c, 2);
        check_return_type(in, caller);
        return IntrinsicCaller{in, seen};
    }
    IntrinsicCaller typecheck_intrin_call(const x86::Intrinsic &in, const Step *caller, DataTypeRef a, DataTypeRef b,
                                          DataTypeRef c, DataTypeRef d) const {
        check_argument_count(in, 4);
        check_nth_arg(in, a, 0);
        check_nth_arg(in, b, 1);
        check_nth_arg(in, c, 2);
        check_nth_arg(in, d, 3);
        check_return_type(in, caller);
        return IntrinsicCaller{in, seen};
    }

    template <typename... Args>
    std::string call_intrin(const x86::Intrinsic &in, const Step *caller, Args &&...args) const {
        check_return_type(in, caller);
        check_argument_count(in, sizeof...(args));
        // Do not use perfect forwarding because we do it twice
        auto checked = typecheck_intrin_call(in, caller, args...);
        return checked.call(args...);
    }

    template <typename... Args>
    std::string call_intrin_var(const x86::Intrinsic &in, const Step *caller, Args &&...args) const {
        return cpp_var_decl(caller) + call_intrin(in, caller, std::forward<Args>(args)...);
    }

    template <typename... Args>
    std::string call_intrin_var(const IntrinsicCaller &in, const Step *caller, Args &&...args) const {
        SIMJIT_ASSERT(sizeof...(Args) == in.i.arity()); // Should be already checked since IntrinsicCaller is used
        return cpp_var_decl(caller) + in.call(std::forward<Args>(args)...);
    }

    bool step_supports_mask_pushdown(const Step *step, bool has_merge) const {
        if (use_counts[step->id] != 1 || !step->dtype.is_vec()) { return false; }

        VecDataType vdtype = step->dtype.as_vec();
        switch (step->kind) {
        case StepKind::Load: {
            const auto &data = step->step_data<StepKind::Load>();
            auto &map = has_merge ? (data.kind == LoadStoreKind::Aligned ? x86::mask_loada_map : x86::mask_loadu_map)
                                  : (data.kind == LoadStoreKind::Aligned ? x86::maskz_loada_map : x86::maskz_loadu_map);
            return map.lookup_nothrow(vdtype) != nullptr;
        }
        case StepKind::Gather: {
            const auto &data = step->step_data<StepKind::Gather>();
            const auto *intrin = data.idx->dtype.as_vec().elem == VecElemType::I32
                                     ? x86::mask_i32gather_map.lookup_nothrow(vdtype)
                                     : x86::mask_i64gather_map.lookup_nothrow(vdtype);
            return intrin != nullptr;
        }
        case StepKind::ArithBinary: {
            auto &data = step->step_data<StepKind::ArithBinary>();
            const x86::Intrinsic *intrin = nullptr;
            if (data.right->is(StepKind::Const)) {
                switch (data.op) {
                case ArithBinaryOp::ShiftRightArith:
                case ArithBinaryOp::ShiftRightLogical:
                case ArithBinaryOp::ShiftLeftLogical:
                case ArithBinaryOp::RotateLeft:
                case ArithBinaryOp::RotateRight:
                    intrin = has_merge ? x86::mask_vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype)
                                       : x86::maskz_vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype);
                    break;
                default: break;
                }
            }
            if (intrin != nullptr) { return true; }

            if (!vdtype.is_float()) {
                switch (data.op) {
                case ArithBinaryOp::Mul64ZE:
                case ArithBinaryOp::Mul64SE:
                case ArithBinaryOp::Div:
                case ArithBinaryOp::UDiv:
                case ArithBinaryOp::Mod:
                case ArithBinaryOp::UMod: return false;
                default: break;
                }
            }
            auto &map = vdtype.is_float()
                            ? (has_merge ? x86::mask_float_binary_map(data.op) : x86::maskz_float_binary_map(data.op))
                            : (has_merge ? x86::mask_arith_binary_map(data.op) : x86::maskz_arith_binary_map(data.op));
            return map.lookup_nothrow(vdtype) != nullptr;
        }
        case StepKind::ArithUnary: {
            auto &data = step->step_data<StepKind::ArithUnary>();
            switch (data.op) {
            case ArithUnaryOp::Not:
            case ArithUnaryOp::Negate: return false;
            case ArithUnaryOp::RoundDown:
            case ArithUnaryOp::RoundUp:
            case ArithUnaryOp::RoundNearest:
            case ArithUnaryOp::RoundTruncate:
                return (has_merge ? x86::mask_roundscale_map : x86::maskz_roundscale_map).lookup_nothrow(vdtype) !=
                       nullptr;
            default: break;
            }
            auto &map = has_merge ? x86::mask_arith_unary_map(data.op) : x86::maskz_arith_unary_map(data.op);
            return map.lookup_nothrow(vdtype) != nullptr;
        }
        case StepKind::IntCast: {
            auto &data = step->step_data<StepKind::IntCast>();
            if (!data.arg->dtype.is_vec()) { return false; }
            VecDataType arg_type = data.arg->dtype.as_vec();
            const auto &map = data.kind == IntCastKind::Zext ? (has_merge ? x86::mask_zext_map : x86::maskz_zext_map)
                                                             : (has_merge ? x86::mask_cvt_map : x86::maskz_cvt_map);
            return map.lookup_nothrow(arg_type, vdtype) != nullptr;
        }
        default: return false;
        }
    }

    bool compare_supports_mask_pushdown(const Step *step, const Step *mask_step) const {
        if (use_counts[step->id] != 1 || !step->is(StepKind::Compare) || !step->dtype.is_mask()) { return false; }
        if (!enable_vector_peepholes() || mask_step->is(StepKind::AccLoad)) { return false; }
        if (!mask_step->dtype.is_mask() || mask_step->dtype != step->dtype) { return false; }
        const auto &data = step->step_data<StepKind::Compare>();
        if (!data.left->dtype.is_vec()) { return false; }
        VecDataType dtype = data.left->dtype.as_vec();
        if (zero_compare_test_supported(step, true)) { return true; }
        const auto &map = data.is_unsigned ? x86::mask_cmpu_mask_map : x86::mask_cmp_mask_map;
        return map.lookup_nothrow(dtype) != nullptr;
    }

    struct ZeroCompareTestArgs {
        const Step *left{};
        const Step *right{};
        VecDataType dtype{};
    };

    static std::optional<ZeroCompareTestArgs> zero_compare_test_args(const Step *step) {
        if (!step->is(StepKind::Compare) || !step->dtype.is_mask()) { return std::nullopt; }
        const auto &data = step->step_data<StepKind::Compare>();
        if (data.op != CmpOp::Equal && data.op != CmpOp::NotEqual) { return std::nullopt; }

        Step *arg = step_is_zero(data.right) ? data.left : nullptr;
        if (arg == nullptr && step_is_zero(data.left)) { arg = data.right; }
        if (arg == nullptr || step_is_zero(arg) || !arg->dtype.is_vec()) { return std::nullopt; }

        VecDataType dtype = arg->dtype.as_vec();
        if (!dtype.is_int()) { return std::nullopt; }
        if (arg->is(StepKind::ArithBinary)) {
            const auto &bin = arg->step_data<StepKind::ArithBinary>();
            if (bin.op == ArithBinaryOp::And) { return ZeroCompareTestArgs{bin.left, bin.right, dtype}; }
        }
        return ZeroCompareTestArgs{arg, arg, dtype};
    }

    bool zero_compare_test_supported(const Step *step, bool masked) const {
        auto args = zero_compare_test_args(step);
        if (!args.has_value()) { return false; }
        const auto &data = step->step_data<StepKind::Compare>();
        const auto &map = data.op == CmpOp::Equal ? (masked ? x86::mask_testn_map : x86::testn_map)
                                                  : (masked ? x86::mask_test_map : x86::test_map);
        return map.lookup_nothrow(args->dtype) != nullptr;
    }

    std::optional<std::string> zero_compare_test_expr(const Step *step, const MaskPushdownInfo *mask_pushdown) const {
        auto args = zero_compare_test_args(step);
        if (!args.has_value()) { return std::nullopt; }
        const auto &data = step->step_data<StepKind::Compare>();
        if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) { return std::nullopt; }
        const bool masked = mask_pushdown != nullptr;
        const auto &map = data.op == CmpOp::Equal ? (masked ? x86::mask_testn_map : x86::testn_map)
                                                  : (masked ? x86::mask_test_map : x86::test_map);
        const auto *intrin = map.lookup_nothrow(args->dtype);
        if (intrin == nullptr) { return std::nullopt; }
        if (masked) {
            auto caller = typecheck_intrin_call(*intrin, step, mask_pushdown->mask, args->left, args->right);
            return caller.call(mask_pushdown->mask, args->left, args->right);
        }
        auto caller = typecheck_intrin_call(*intrin, step, args->left, args->right);
        return caller.call(args->left, args->right);
    }

    std::optional<std::string> masked_compare_expr(const Step *cmp_step, const Step *mask_step) const {
        if (!compare_supports_mask_pushdown(cmp_step, mask_step)) { return std::nullopt; }
        MaskPushdownInfo pushdown{mask_step, nullptr, cmp_step};
        if (auto expr = zero_compare_test_expr(cmp_step, &pushdown)) { return expr; }
        const auto &data = cmp_step->step_data<StepKind::Compare>();
        VecDataType dtype = data.left->dtype.as_vec();
        const char *imm = dtype.is_float() ? cmp_op_to_fp_imm(data.op) : cmp_op_to_int_imm(data.op);
        const auto &intrin =
            data.is_unsigned ? x86::mask_cmpu_mask_map.lookup(dtype) : x86::mask_cmp_mask_map.lookup(dtype);
        auto caller = typecheck_intrin_call(intrin, cmp_step, mask_step, data.left, data.right, ScalarDataType::I32);
        return caller.call(mask_step, data.left, data.right, imm);
    }

    void record_backend_peephole_uses(Step *step) override {
        auto mark_named_operand = [&](const Step *operand) { ++peephole_named_use_counts[operand->id]; };

        auto mark_masked_compare_pushdown = [&](Step *cmp_step, Step *mask_step) {
            ++peephole_step_use_counts[cmp_step->id];
            mark_named_operand(mask_step);

            auto args = zero_compare_test_args(cmp_step);
            if (args.has_value() && zero_compare_test_supported(cmp_step, true)) {
                mark_named_operand(args->left);
                mark_named_operand(args->right);
                return;
            }

            const auto &cmp_data = cmp_step->step_data<StepKind::Compare>();
            mark_named_operand(cmp_data.left);
            mark_named_operand(cmp_data.right);
        };

        if (enable_vector_peepholes() && step->is(StepKind::ArithBinary) && step->dtype.is_vec()) {
            auto &data = step->step_data<StepKind::ArithBinary>();
            if (data.right->is(StepKind::Const)) {
                switch (data.op) {
                case ArithBinaryOp::ShiftRightArith:
                case ArithBinaryOp::ShiftRightLogical:
                case ArithBinaryOp::ShiftLeftLogical:
                case ArithBinaryOp::RotateLeft:
                case ArithBinaryOp::RotateRight:
                    if (x86::vector_immediate_shift_rotate_map(data.op).lookup_nothrow(step->dtype.as_vec())) {
                        ++peephole_const_use_counts[data.right->id];
                    }
                    break;
                default: break;
                }
            }
        }

        if (enable_vector_peepholes() && step->is(StepKind::Compare) && step->dtype.is_mask()) {
            auto &data = step->step_data<StepKind::Compare>();
            if (data.left->dtype.is_vec()) {
                VecDataType vdtype = data.left->dtype.as_vec();
                if (vdtype.is_int()) {
                    if (data.op == CmpOp::Equal || data.op == CmpOp::NotEqual) {
                        Step *zero = step_is_zero(data.right) ? data.right : nullptr;
                        if (zero == nullptr && step_is_zero(data.left)) { zero = data.left; }
                        Step *arg = zero == data.right ? data.left : data.right;
                        const auto *intrin =
                            (data.op == CmpOp::Equal ? x86::testn_map : x86::test_map).lookup_nothrow(vdtype);
                        if (zero != nullptr && !step_is_zero(arg) && intrin != nullptr) {
                            ++peephole_const_use_counts[zero->id];
                            if (arg->is(StepKind::ArithBinary)) {
                                auto &bin = arg->step_data<StepKind::ArithBinary>();
                                if (bin.op == ArithBinaryOp::And) { ++peephole_step_use_counts[arg->id]; }
                            }
                        }
                    } else if (data.op == CmpOp::Less && !data.is_unsigned && step_is_zero(data.right) &&
                               x86::mov_mask_map.lookup_nothrow(vdtype) != nullptr) {
                        ++peephole_const_use_counts[data.right->id];
                    }
                }
            }
        }

        if (enable_vector_peepholes() && step->is(StepKind::Select) && step->dtype.is_vec()) {
            auto &data = step->step_data<StepKind::Select>();
            VecDataType vdtype = step->dtype.as_vec();
            if (step_is_zero(data.falsy)) {
                if (vdtype.is_int() && data.truthy->is(StepKind::Const) &&
                    x86::movm_map.lookup_nothrow(vdtype) != nullptr) {
                    uint64_t all_ones = scalar_dtype_umax(vec_elem_to_scalar(vdtype.elem));
                    if ((data.truthy->step_data<StepKind::Const>().as_unsigned() & all_ones) == all_ones) {
                        ++peephole_const_use_counts[data.truthy->id];
                        ++peephole_const_use_counts[data.falsy->id];
                        return;
                    }
                }
                if (x86::maskz_mov_map.lookup_nothrow(vdtype) != nullptr) {
                    ++peephole_const_use_counts[data.falsy->id];
                    if (step_supports_mask_pushdown(data.truthy, false)) {
                        ++peephole_named_use_counts[data.cond->id];
                        ++peephole_step_use_counts[data.truthy->id];
                    }
                }
            } else if (x86::mask_blend_map.lookup_nothrow(vdtype) != nullptr &&
                       step_supports_mask_pushdown(data.truthy, true)) {
                ++peephole_named_use_counts[data.cond->id];
                ++peephole_step_use_counts[data.truthy->id];
            }
        }

        if (enable_vector_peepholes() && step->is(StepKind::MaskBinary) && step->dtype.is_mask()) {
            auto &data = step->step_data<StepKind::MaskBinary>();
            if (data.op == PredicateBinaryOp::And) {
                if (compare_supports_mask_pushdown(data.left, data.right)) {
                    mark_masked_compare_pushdown(data.left, data.right);
                } else if (compare_supports_mask_pushdown(data.right, data.left)) {
                    mark_masked_compare_pushdown(data.right, data.left);
                }
            }
        }

        if (enable_vector_peepholes() && step->is(StepKind::Store) && step->dtype.is_vec()) {
            auto &data = step->step_data<StepKind::Store>();
            if (data.what->is(StepKind::IntCast)) {
                auto &cast = data.what->step_data<StepKind::IntCast>();
                if (cast.kind == IntCastKind::Trunc && cast.arg->dtype.is_vec() &&
                    x86::cvt_storeu_map.lookup_nothrow(cast.arg->dtype.as_vec(), step->dtype.as_vec()) != nullptr) {
                    ++peephole_step_use_counts[data.what->id];
                }
            }
        }

        if (enable_vector_peepholes() && step->is(StepKind::CondStore) && step->dtype.is_vec()) {
            auto &data = step->step_data<StepKind::CondStore>();
            if (data.arg->is(StepKind::IntCast)) {
                auto &cast = data.arg->step_data<StepKind::IntCast>();
                if (cast.kind == IntCastKind::Trunc && cast.arg->dtype.is_vec() &&
                    x86::cvt_storeu_map.lookup_nothrow(cast.arg->dtype.as_vec(), step->dtype.as_vec()) != nullptr) {
                    ++peephole_step_use_counts[data.arg->id];
                }
            }
        }
    }

    static bool x86_is_float_minmax(VecDataType vdtype, ArithBinaryOp op) {
        return vdtype.is_float() && (op == ArithBinaryOp::Min || op == ArithBinaryOp::Max);
    }

    std::string x86_vector_arith_binary_expr(VecDataType vdtype, ArithBinaryOp op, const std::string &left,
                                             const std::string &right,
                                             const std::optional<std::string> &merge = std::nullopt,
                                             const std::optional<std::string> &mask = std::nullopt) const {
        if (x86_is_float_minmax(vdtype, op) && (!mask.has_value() || (merge.has_value() && *merge == left))) {
            const auto &cmp_intrin =
                mask.has_value() ? x86::mask_cmp_mask_map.lookup(vdtype) : x86::cmp_mask_map.lookup(vdtype);
            std::string ordered_mask = mask.has_value()
                                           ? unsafe_intrin_call(cmp_intrin, *mask, right, right, "_CMP_ORD_Q")
                                           : unsafe_intrin_call(cmp_intrin, right, right, "_CMP_ORD_Q");
            const auto &intrin = x86::mask_float_binary_map(op).lookup(vdtype);
            return unsafe_intrin_call(intrin, merge.value_or(left), ordered_mask, left, right);
        }

        std::string result;
        if (mask.has_value()) {
            if (merge.has_value()) {
                const auto &intrin = vdtype.is_float() ? x86::mask_float_binary_map(op).lookup(vdtype)
                                                       : x86::mask_arith_binary_map(op).lookup(vdtype);
                result = unsafe_intrin_call(intrin, *merge, *mask, left, right);
            } else {
                const auto &intrin = vdtype.is_float() ? x86::maskz_float_binary_map(op).lookup(vdtype)
                                                       : x86::maskz_arith_binary_map(op).lookup(vdtype);
                result = unsafe_intrin_call(intrin, *mask, left, right);
            }
        } else {
            const auto &intrin =
                vdtype.is_float() ? x86::float_binary_map(op).lookup(vdtype) : x86::arith_binary_map(op).lookup(vdtype);
            result = unsafe_intrin_call(intrin, left, right);
        }

        if (!x86_is_float_minmax(vdtype, op)) { return result; }

        const auto &cmp_intrin =
            mask.has_value() ? x86::mask_cmp_mask_map.lookup(vdtype) : x86::cmp_mask_map.lookup(vdtype);
        std::string nan_mask = mask.has_value() ? unsafe_intrin_call(cmp_intrin, *mask, right, right, "_CMP_UNORD_Q")
                                                : unsafe_intrin_call(cmp_intrin, right, right, "_CMP_UNORD_Q");
        const auto &mov_intrin = x86::mask_mov_map.lookup(vdtype);
        return unsafe_intrin_call(mov_intrin, result, nan_mask, left);
    }

    std::string vector_arith_binary_expr(const Step *step, const ArithBinData &data, const Step *acc_load, AccId acc,
                                         const Step *mask = nullptr) {
        std::string left = show_inline_operand(data.left, acc_load, acc);
        VecDataType vdtype = step->dtype.as_vec();
        const x86::Intrinsic *imm_intrin = nullptr;
        if (data.right->is(StepKind::Const)) {
            switch (data.op) {
            case ArithBinaryOp::ShiftRightArith:
            case ArithBinaryOp::ShiftRightLogical:
            case ArithBinaryOp::ShiftLeftLogical:
            case ArithBinaryOp::RotateLeft:
            case ArithBinaryOp::RotateRight:
                imm_intrin = mask != nullptr
                                 ? x86::mask_vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype)
                                 : x86::vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype);
                break;
            default: break;
            }
        }
        if (imm_intrin != nullptr) {
            if (mask != nullptr) {
                auto caller = typecheck_intrin_call(*imm_intrin, step, vdtype, mask, data.left, ScalarDataType::I32);
                return caller.call(format("acc%zu", func->accs.index(acc)), mask, left,
                                   std::to_string((long long)data.right->step_data<StepKind::Const>().as_signed()));
            }
            auto caller = typecheck_intrin_call(*imm_intrin, step, data.left, ScalarDataType::I32);
            return caller.call(left, std::to_string((long long)data.right->step_data<StepKind::Const>().as_signed()));
        }
        std::string right = show_inline_operand(data.right, acc_load, acc);
        if (mask != nullptr) {
            return x86_vector_arith_binary_expr(vdtype, data.op, left, right, format("acc%zu", func->accs.index(acc)),
                                                show(mask));
        }
        return x86_vector_arith_binary_expr(vdtype, data.op, left, right);
    }

    std::string mask_binary_expr(const Step *step, const PredicateBinData &data, const Step *acc_load, AccId acc) {
        std::string left = show_inline_operand(data.left, acc_load, acc);
        std::string right = show_inline_operand(data.right, acc_load, acc);
        MaskDataType mdtype = step->dtype.as_mask();
        if (mdtype == MaskDataType::M2 || mdtype == MaskDataType::M4) mdtype = MaskDataType::M8;
        const auto &intrin = x86::binary_op_mask_map(data.op).lookup(mdtype);
        return unsafe_intrin_call(intrin, left, right);
    }

    std::string vec_low_half_expr(VecDataType arg_type, const char *arg) const {
        auto maybe_half = vec_dtype_half(arg_type);
        SIMJIT_ASSERT(maybe_half.has_value());
        VecDataType half_type = *maybe_half;
        const auto &intrin = x86::compiler_downcast_map.lookup(arg_type, half_type);
        return unsafe_intrin_call(intrin, arg);
    }

    std::string vec_high_half_expr(VecDataType arg_type, const char *arg) const {
        auto maybe_half = vec_dtype_half(arg_type);
        SIMJIT_ASSERT(maybe_half.has_value());
        VecDataType half_type = *maybe_half;
        const auto &intrin = x86::extract_map.lookup(arg_type, half_type);
        return unsafe_intrin_call(intrin, arg, "1");
    }

    std::string vec_widen_half_to_cpp(const Step *step, const Step *arg, bool is_unsigned, bool high_half) const {
        VecDataType arg_type = arg->dtype.as_vec();
        VecDataType dst_type = step->dtype.as_vec();
        auto maybe_half = vec_dtype_half(arg_type);
        SIMJIT_ASSERT(maybe_half.has_value());
        VecDataType half_type = *maybe_half;

        const x86::Vector arg_x86 = x86::vec_to_x86(arg_type);
        const x86::Vector dst_x86 = x86::vec_to_x86(dst_type);
        if (arg_x86.reg == x86::VecRegisterKind::XMM && dst_x86.reg == x86::VecRegisterKind::XMM) {
            const auto &intrin = arg_type.is_float() ? x86::float_cast_map.lookup(arg_type, dst_type)
                                 : is_unsigned       ? x86::zext_map.lookup(arg_type, dst_type)
                                                     : x86::cvt_map.lookup(arg_type, dst_type);
            auto caller = typecheck_intrin_call(intrin, step, arg_type);
            const char *arg_expr = show(arg);
            std::string half_expr = arg_expr;
            if (high_half && arg_type.is_float()) {
                half_expr = format("_mm_movehl_ps(%s, %s)", arg_expr, arg_expr);
            } else if (high_half) {
                half_expr = format("_mm_srli_si128(%s, 8)", arg_expr);
            }
            return cpp_var_decl(step) + caller.call(half_expr);
        }

        std::string half_expr =
            high_half ? vec_high_half_expr(arg_type, show(arg)) : vec_low_half_expr(arg_type, show(arg));

        const auto &intrin = arg_type.is_float() ? x86::float_cast_map.lookup(half_type, dst_type)
                             : is_unsigned       ? x86::zext_map.lookup(half_type, dst_type)
                                                 : x86::cvt_map.lookup(half_type, dst_type);
        auto caller = typecheck_intrin_call(intrin, step, half_type);
        return cpp_var_decl(step) + caller.call(half_expr);
    }

    std::string vec_narrow_combine_to_cpp(const Step *step, const VecNarrowCombineData &data) const {
        VecDataType vdtype = step->dtype.as_vec();
        x86::VecRegisterKind reg = x86::vec_to_x86(vdtype).reg;
        VecDataType wide_type = data.low->dtype.as_vec();

        const char *name = show(step);

        switch (reg) {
        case x86::VecRegisterKind::XMM: {
            const auto &trunc = x86::cvt_map.lookup(wide_type, vdtype);
            std::string low_expr = unsafe_intrin_call(trunc, show(data.low));
            std::string high_expr = unsafe_intrin_call(trunc, show(data.high));
            return format("%s %s;\n"
                          "{\n"
                          "%s %s_lo = %s;\n"
                          "%s %s_hi = %s;\n"
                          "%s = _mm_unpacklo_epi64(%s_lo, %s_hi);\n"
                          "}",
                          dtype_to_cpp(vdtype), name,                    //
                          dtype_to_cpp(vdtype), name, low_expr.c_str(),  //
                          dtype_to_cpp(vdtype), name, high_expr.c_str(), //
                          name, name, name);
        }
        case x86::VecRegisterKind::YMM: {
            auto maybe_half_dtype = vec_dtype_half(vdtype);
            SIMJIT_ASSERT(maybe_half_dtype.has_value());
            VecDataType half_type = *maybe_half_dtype;
            const auto &trunc = x86::cvt_map.lookup(wide_type, half_type);
            std::string low_expr = unsafe_intrin_call(trunc, show(data.low));
            std::string high_expr = unsafe_intrin_call(trunc, show(data.high));
            return format("%s %s;\n"
                          "{\n"
                          "%s %s_lo = %s;\n"
                          "%s %s_hi = %s;\n"
                          "%s = _mm256_inserti128_si256(_mm256_castsi128_si256(%s_lo), %s_hi, 1);\n"
                          "}",
                          dtype_to_cpp(vdtype), name,                       //
                          dtype_to_cpp(half_type), name, low_expr.c_str(),  //
                          dtype_to_cpp(half_type), name, high_expr.c_str(), //
                          name, name, name);
        }
        case x86::VecRegisterKind::ZMM: {
            auto maybe_half_dtype = vec_dtype_half(vdtype);
            SIMJIT_ASSERT(maybe_half_dtype.has_value());
            VecDataType half_type = *maybe_half_dtype;
            const auto &trunc = x86::cvt_map.lookup(wide_type, half_type);
            std::string low_expr = unsafe_intrin_call(trunc, show(data.low));
            std::string high_expr = unsafe_intrin_call(trunc, show(data.high));
            return format("%s %s;\n"
                          "{\n"
                          "%s %s_lo = %s;\n"
                          "%s %s_hi = %s;\n"
                          "%s = _mm512_inserti64x4(_mm512_castsi256_si512(%s_lo), %s_hi, 1);\n"
                          "}",
                          dtype_to_cpp(vdtype), name,                       //
                          dtype_to_cpp(half_type), name, low_expr.c_str(),  //
                          dtype_to_cpp(half_type), name, high_expr.c_str(), //
                          name, name, name);
        }
        }
        SIMJIT_UNREACHABLE();
    }

    std::string vec_float_narrow_combine_to_cpp(const Step *step, const VecFloatNarrowCombineData &data) const {
        VecDataType vdtype = step->dtype.as_vec();
        VecDataType wide_type = data.low->dtype.as_vec();
        if (vdtype.elem != VecElemType::F32 || wide_type.elem != VecElemType::F64) {
            messed_up("unsupported float narrow combine %s <- %s", show_vec_dtype(vdtype), show_vec_dtype(wide_type));
        }

        const char *name = show(step);

        switch (x86::vec_to_x86(vdtype).reg) {
        case x86::VecRegisterKind::XMM: {
            const auto &cast = x86::float_cast_map.lookup(wide_type, vdtype);
            std::string low_expr = unsafe_intrin_call(cast, show(data.low));
            std::string high_expr = unsafe_intrin_call(cast, show(data.high));
            return format("%s %s;\n"
                          "{\n"
                          "%s %s_lo = %s;\n"
                          "%s %s_hi = %s;\n"
                          "%s = _mm_movelh_ps(%s_lo, %s_hi);\n"
                          "}",
                          dtype_to_cpp(vdtype), name,                    //
                          dtype_to_cpp(vdtype), name, low_expr.c_str(),  //
                          dtype_to_cpp(vdtype), name, high_expr.c_str(), //
                          name, name, name);
        }
        case x86::VecRegisterKind::YMM: {
            auto maybe_half_dtype = vec_dtype_half(vdtype);
            SIMJIT_ASSERT(maybe_half_dtype.has_value());
            VecDataType half_type = *maybe_half_dtype;
            const auto &cast = x86::float_cast_map.lookup(wide_type, half_type);
            std::string low_expr = unsafe_intrin_call(cast, show(data.low));
            std::string high_expr = unsafe_intrin_call(cast, show(data.high));
            return format("%s %s;\n"
                          "{\n"
                          "%s %s_lo = %s;\n"
                          "%s %s_hi = %s;\n"
                          "%s = _mm256_insertf128_ps(_mm256_castps128_ps256(%s_lo), %s_hi, 1);\n"
                          "}",
                          dtype_to_cpp(vdtype), name,                       //
                          dtype_to_cpp(half_type), name, low_expr.c_str(),  //
                          dtype_to_cpp(half_type), name, high_expr.c_str(), //
                          name, name, name);
        }
        case x86::VecRegisterKind::ZMM: {
            auto maybe_half_dtype = vec_dtype_half(vdtype);
            SIMJIT_ASSERT(maybe_half_dtype.has_value());
            VecDataType half_type = *maybe_half_dtype;
            const auto &cast = x86::float_cast_map.lookup(wide_type, half_type);
            std::string low_expr = unsafe_intrin_call(cast, show(data.low));
            std::string high_expr = unsafe_intrin_call(cast, show(data.high));
            return format("%s %s;\n"
                          "{\n"
                          "%s %s_lo = %s;\n"
                          "%s %s_hi = %s;\n"
                          "%s = _mm512_insertf32x8(_mm512_castps256_ps512(%s_lo), %s_hi, 1);\n"
                          "}",
                          dtype_to_cpp(vdtype), name,                       //
                          dtype_to_cpp(half_type), name, low_expr.c_str(),  //
                          dtype_to_cpp(half_type), name, high_expr.c_str(), //
                          name, name, name);
        }
        }
        SIMJIT_UNREACHABLE();
    }

    bool can_inline_acc_store_arg(const Step *arg) const {
        if (use_counts[arg->id] != 1) { return false; }
        return arg->is(StepKind::Const) || arg->is(StepKind::LoadSplat);
    }

    bool suppress_compacted_acc_store_args(const Step *step, ArenaBitmap &suppressed) const override {
        if (!step->is(StepKind::AccStore)) { return false; }

        auto &store = step->step_data<StepKind::AccStore>();
        const Step *arg = store.arg;
        // We can only collapse load -> binary -> store into a single "acc = acc op rhs"
        // line when the binary result is consumed by this store only.
        if (use_counts[arg->id] != 1) { return false; }

        if (arg->is(StepKind::ArithBinary)) {
            auto &bin = arg->step_data<StepKind::ArithBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return false; }
            suppressed.set(arg->id);
            suppressed.set(acc_load->id);
            return true;
        }

        if (arg->is(StepKind::MaskBinary)) {
            auto &bin = arg->step_data<StepKind::MaskBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return false; }
            suppressed.set(arg->id);
            suppressed.set(acc_load->id);
            return true;
        }

        if (arg->is(StepKind::FMA)) {
            auto &fma = arg->step_data<StepKind::FMA>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {fma.x1, fma.x2, fma.x3});
            if (acc_load == nullptr) { return false; }
            suppressed.set(arg->id);
            suppressed.set(acc_load->id);
            return true;
        }

        if (arg->is(StepKind::Select) && arg->dtype.is_scalar()) {
            auto &select = arg->step_data<StepKind::Select>();
            if (!select.falsy->is(StepKind::AccLoad) || select.falsy->step_data<StepKind::AccLoad>() != store.acc) {
                return false;
            }

            if (select.truthy->is(StepKind::ArithBinary) && use_counts[select.truthy->id] == 1) {
                auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
                const Step *acc_load =
                    find_compacted_acc_load(store.acc, {bin.left, bin.right}, {bin.left, bin.right, select.falsy});
                if (acc_load == nullptr) { return false; }
                if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return false; }

                suppressed.set(arg->id);
                suppressed.set(select.truthy->id);
                suppressed.set(acc_load->id);
                suppressed.set(select.falsy->id);
                return true;
            }

            if (use_counts[select.falsy->id] != 1) { return false; }
            suppressed.set(arg->id);
            suppressed.set(select.falsy->id);
            return true;
        }

        if (arg->is(StepKind::Select) && step->dtype.is_vec()) {
            auto &select = arg->step_data<StepKind::Select>();
            if (!select.truthy->is(StepKind::ArithBinary) || !select.falsy->is(StepKind::AccLoad)) { return false; }
            if (select.falsy->step_data<StepKind::AccLoad>() != store.acc) { return false; }
            if (!step_supports_mask_pushdown(select.truthy, true)) { return false; }

            auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
            const Step *acc_load =
                find_compacted_acc_load(store.acc, {bin.left, bin.right}, {bin.left, bin.right, select.falsy});
            if (acc_load == nullptr) { return false; }
            if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return false; }

            suppressed.set(arg->id);
            suppressed.set(select.truthy->id);
            suppressed.set(acc_load->id);
            suppressed.set(select.falsy->id);
            return true;
        }

        if (step->is(StepKind::AccStore)) {
            const auto &data = step->step_data<StepKind::AccStore>();
            if (can_inline_acc_store_arg(data.arg)) {
                suppressed.set(data.arg->id);
                return true;
            }
        }
        return false;
    }

    std::optional<std::string> inline_acc_store_arg_expr(const Step *arg) const {
        if (!can_inline_acc_store_arg(arg)) { return std::nullopt; }

        if (arg->is(StepKind::Const)) {
            const auto &data = arg->step_data<StepKind::Const>();
            if (arg->dtype.is_scalar()) { return const_data_to_cpp(data, arg->dtype.as_scalar()); }
            if (arg->dtype.is_mask()) {
                return std::string(data.as_unsigned() != 0 ? mask_all_ones(arg->dtype.as_mask()) : "0");
            }

            VecDataType dtype = arg->dtype.as_vec();
            if (data.is_zero()) {
                const auto &intrin = x86::setzero_map.lookup(dtype);
                auto caller = typecheck_intrin_call(intrin, arg);
                return caller.call();
            }

            const auto &intrin = x86::set1_map.lookup(dtype);
            auto caller = typecheck_intrin_call(intrin, arg, dtype.to_scalar());
            return caller.call(const_data_to_cpp(data, dtype.to_scalar()));
        }

        if (arg->is(StepKind::LoadSplat)) {
            const auto &data = arg->step_data<StepKind::LoadSplat>();
            if (arg->dtype.is_scalar()) { return format("*arg%zu", data.addr.arg); }
            if (arg->dtype.is_mask()) {
                return format("(*arg%zu & 1) ? %s : 0", data.addr.arg, mask_all_ones(arg->dtype.as_mask()));
            }

            VecDataType dtype = arg->dtype.as_vec();
            const auto &intrin = x86::set1_map.lookup(dtype);
            return unsafe_intrin_call(intrin, format("*arg%zu", data.addr.arg));
        }

        SIMJIT_UNREACHABLE();
    }

    std::optional<std::string> backend_compact_acc_store_to_cpp(const Step *step) override {
        if (!step->is(StepKind::AccStore)) { return std::nullopt; }

        auto &store = step->step_data<StepKind::AccStore>();
        const Step *arg = store.arg;
        if (use_counts[arg->id] != 1) { return std::nullopt; }

        if (arg->is(StepKind::ArithBinary)) {
            auto &bin = arg->step_data<StepKind::ArithBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return std::nullopt; }
            if (step->dtype.is_scalar()) {
                std::string left = show_inline_operand(bin.left, acc_load, store.acc);
                std::string right = show_inline_operand(bin.right, acc_load, store.acc);
                return scalar_arith_binary_expr(step->dtype.as_scalar(), bin, left.c_str(), right.c_str());
            }
            return vector_arith_binary_expr(arg, bin, acc_load, store.acc);
        }

        if (arg->is(StepKind::MaskBinary)) {
            auto &bin = arg->step_data<StepKind::MaskBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return std::nullopt; }
            return mask_binary_expr(arg, bin, acc_load, store.acc);
        }

        if (arg->is(StepKind::FMA)) {
            auto &fma = arg->step_data<StepKind::FMA>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {fma.x1, fma.x2, fma.x3});
            if (acc_load == nullptr) { return std::nullopt; }
            const auto &intrin = x86::fma_map(fma.kind).lookup(arg->dtype.as_vec());
            auto caller = typecheck_intrin_call(intrin, arg, fma.x1, fma.x2, fma.x3);
            std::string x1 = show_inline_operand(fma.x1, acc_load, store.acc);
            std::string x2 = show_inline_operand(fma.x2, acc_load, store.acc);
            std::string x3 = show_inline_operand(fma.x3, acc_load, store.acc);
            return caller.call(x1, x2, x3);
        }

        if (arg->is(StepKind::Select) && arg->dtype.is_scalar()) {
            auto &select = arg->step_data<StepKind::Select>();
            if (!select.falsy->is(StepKind::AccLoad) || select.falsy->step_data<StepKind::AccLoad>() != store.acc) {
                return std::nullopt;
            }

            std::string cond = show_scalar_operand(select.cond);
            if (select.truthy->is(StepKind::ArithBinary) && use_counts[select.truthy->id] == 1) {
                auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
                const Step *acc_load =
                    find_compacted_acc_load(store.acc, {bin.left, bin.right}, {bin.left, bin.right, select.falsy});
                if (acc_load == nullptr) { return std::nullopt; }
                if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return std::nullopt; }

                std::string left = show_inline_operand(bin.left, acc_load, store.acc);
                std::string right = show_inline_operand(bin.right, acc_load, store.acc);
                std::string truthy = scalar_arith_binary_expr(arg->dtype.as_scalar(), bin, left.c_str(), right.c_str());
                std::string falsy = show_inline_operand(select.falsy, acc_load, store.acc);
                return format("%s ? %s : %s", cond.c_str(), truthy.c_str(), falsy.c_str());
            }

            if (use_counts[select.falsy->id] != 1) { return std::nullopt; }
            return format("%s ? %s : acc%zu", cond.c_str(), show_scalar_operand(select.truthy),
                          func->accs.index(store.acc));
        }

        if (arg->is(StepKind::Select) && step->dtype.is_vec()) {
            auto &select = arg->step_data<StepKind::Select>();
            if (!select.truthy->is(StepKind::ArithBinary) || !select.falsy->is(StepKind::AccLoad)) {
                return std::nullopt;
            }
            if (select.falsy->step_data<StepKind::AccLoad>() != store.acc) { return std::nullopt; }
            if (!step_supports_mask_pushdown(select.truthy, true)) { return std::nullopt; }

            auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
            const Step *acc_load =
                find_compacted_acc_load(store.acc, {bin.left, bin.right}, {bin.left, bin.right, select.falsy});
            if (acc_load == nullptr) { return std::nullopt; }
            if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return std::nullopt; }

            return vector_arith_binary_expr(select.truthy, bin, acc_load, store.acc, select.cond);
        }

        if (step->is(StepKind::AccStore)) {
            const auto &data = step->step_data<StepKind::AccStore>();
            if (auto expr = inline_acc_store_arg_expr(data.arg)) { return expr; }
        }
        return std::nullopt;
    }

    std::string store_sum128_to_cpp(const StoreSum128Data &data) override {
        std::string result = "{\n"
                             "uint64_t sum128_low = 0;\n"
                             "uint64_t sum128_carry = 0;\n"
                             "uint64_t sum128_prev = 0;\n";
        size_t low_idx = 0;
        for (auto *low : data.low_steps) {
            if (low->dtype.is_vec()) {
                VecDataType vdtype = low->dtype.as_vec();
                std::string values = format("sum128_values_%zu", low_idx++);
                const auto &storeu = x86::storeu_map.lookup(vdtype);
                format_to(result, "alignas(64) uint64_t %s[%zu];\n", values.c_str(), vdtype.nelems());
                result +=
                    unsafe_intrin_call(storeu, format("(%s *)%s", dtype_to_cpp(vdtype), values.c_str()), show(low)) +
                    ";\n";
                format_to(result,
                          "for (size_t sum128_i = 0; "
                          "sum128_i < %zu; "
                          "++sum128_i) {\n"
                          "sum128_prev = sum128_low;\n"
                          "sum128_low += %s[sum128_i];\n"
                          "sum128_carry += (sum128_low < sum128_prev);\n"
                          "}\n",
                          vdtype.nelems(), //
                          values.c_str());
                continue;
            }
            format_to(result,
                      "sum128_prev = sum128_low;\n"
                      "sum128_low += (uint64_t)%s;\n"
                      "sum128_carry += (sum128_low < sum128_prev);\n",
                      show_scalar_operand(low));
        }
        format_to(result,
                  "__int128 sum128_hi = (__int128)%s + "
                  "(__int128)sum128_carry;\n"
                  "*arg%zu = (sum128_hi << 64) + "
                  "(unsigned __int128)sum128_low;\n",
                  show_scalar_operand(data.hi_combined), //
                  data.dst);
        result += "}";
        return result;
    }

    std::string backend_step_to_cpp(const Step *step, const MaskPushdownInfo *mask_pushdown) override {
        const Step *result_step = mask_pushdown == nullptr ? step : mask_pushdown->result;
        SIMJIT_ASSERT(step->dtype == result_step->dtype);
        if (step->dtype.is_scalar() && is_scalar_step(step->kind)) { return scalar_step_to_cpp(step); }
        switch (step->kind) {
        case StepKind::AggResult:
        case StepKind::ScalarIndex:
        case StepKind::ScalarArithBinaryOverflow:
        case StepKind::ScalarPermute:
        case StepKind::StoreSum128:
        case StepKind::ConstDiv:
            messed_up("this is scalar instruction");
            SIMJIT_MATCH (StepKind::LoadSplat) {
                if (step->dtype.is_mask()) {
                    return cpp_var_decl(step) +
                           format("(*arg%zu & 1) ? %s : 0", data.addr.arg, mask_all_ones(step->dtype.as_mask()));
                }
                VecDataType dtype = step->dtype.as_vec();
                const auto &intrin = x86::set1_map.lookup(dtype);
                return cpp_var_decl(step) + unsafe_intrin_call(intrin, format("*arg%zu", data.addr.arg));
            }
            SIMJIT_MATCH (StepKind::Const) {
                if (step->dtype.is_mask()) {
                    return cpp_var_decl(step) +
                           std::string(data.as_unsigned() != 0 ? mask_all_ones(step->dtype.as_mask()) : "0");
                }

                VecDataType vec_dtype = step->dtype.as_vec();
                if (data.is_zero()) {
                    const auto &intrin = x86::setzero_map.lookup(vec_dtype);
                    return call_intrin_var(intrin, step);
                }
                const auto &intrin = x86::set1_map.lookup(vec_dtype);
                auto caller = typecheck_intrin_call(intrin, step, vec_dtype.to_scalar());
                return call_intrin_var(caller, step, const_data_to_cpp(data, vec_dtype.to_scalar()));
            }
            SIMJIT_MATCH (StepKind::VecIndex) {
                VecDataType vdtype = step->dtype.as_vec();
                const auto &add_intrin = x86::add_map.lookup(vdtype);
                std::string result{};
                result += cpp_var_decl(step) + format("acc%zu;\n", func->accs.index(data.acc));
                result +=
                    format("acc%zu = ", func->accs.index(data.acc)) + unsafe_intrin_call(add_intrin, step, data.inc);
                return result;
            }
            SIMJIT_MATCH (StepKind::Load) {
                const ArgumentDecl &arg = func->args[data.addr.arg];
                SIMJIT_ASSERT(mask_pushdown == nullptr || !step->dtype.is_mask());
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    if (mask_uses_partial_byte(mdtype)) {
                        std::string row = data.addr.offset == 0 ? "i" : format("(i + %zu)", data.addr.offset);
                        return cpp_var_decl(step) + format("(__mmask8)(((arg%zu[%s >> 3] >> (%s & 7)) & %s)",
                                                           data.addr.arg, row.c_str(), row.c_str(),
                                                           mask_all_ones(mdtype));
                    }
                    const auto &intrin = x86::load_mask_map.lookup(mdtype);
                    std::string offset_str =
                        data.addr.offset != 0 ? format(" + %zu", data.addr.offset / mask_dtype_bits(mdtype)) : "";
                    std::string expr = unsafe_intrin_call(
                        intrin, format("(%s *)arg%zu + ((i >> %zu)%s)", mask_dtype_cpp(mdtype), data.addr.arg,
                                       mask_dtype_bits_log2(mdtype), offset_str.c_str()));
                    if (mask_uses_partial_byte(mdtype)) {
                        expr = format("(%s & %s)", expr.c_str(), mask_all_ones(mdtype));
                    }
                    return cpp_var_decl(step) + expr;
                }
                std::string offset_str = data.addr.offset != 0 ? format(" + %zu", data.addr.offset) : "";
                VecDataType vec_dtype = step->dtype.as_vec();

                if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                    auto &map = data.kind == LoadStoreKind::Aligned ? x86::mask_loada_map : x86::mask_loadu_map;
                    const auto &intrin = map.lookup(vec_dtype);
                    DataType func_arg_dtype = intrin.arg_dtype(2);
                    check_return_type(intrin, result_step);
                    check_nth_arg(intrin, mask_pushdown->merge, 0);
                    check_nth_arg(intrin, mask_pushdown->mask, 1);
                    if (DataType{arg.dtype} != func_arg_dtype && !vec_dtype.is_float()) {
                        return cpp_var_decl(result_step) +
                               unsafe_intrin_call(intrin, mask_pushdown->merge, mask_pushdown->mask,
                                                  format("(const %s *)(arg%zu + i%s)", dtype_to_cpp(func_arg_dtype),
                                                         data.addr.arg, offset_str.c_str()));
                    }
                    return cpp_var_decl(result_step) +
                           unsafe_intrin_call(intrin, mask_pushdown->merge, mask_pushdown->mask,
                                              format("arg%zu + i%s", data.addr.arg, offset_str.c_str()));
                }

                if (mask_pushdown != nullptr) {
                    auto &map = data.kind == LoadStoreKind::Aligned ? x86::maskz_loada_map : x86::maskz_loadu_map;
                    const auto &intrin = map.lookup(vec_dtype);
                    DataType func_arg_dtype = intrin.arg_dtype(1);
                    check_return_type(intrin, result_step);
                    check_nth_arg(intrin, mask_pushdown->mask, 0);
                    if (DataType{arg.dtype} != func_arg_dtype && !vec_dtype.is_float()) {
                        return cpp_var_decl(result_step) +
                               unsafe_intrin_call(intrin, mask_pushdown->mask,
                                                  format("(const %s *)(arg%zu + i%s)", dtype_to_cpp(func_arg_dtype),
                                                         data.addr.arg, offset_str.c_str()));
                    }
                    return cpp_var_decl(result_step) +
                           unsafe_intrin_call(intrin, mask_pushdown->mask,
                                              format("arg%zu + i%s", data.addr.arg, offset_str.c_str()));
                }

                auto &map = data.kind == LoadStoreKind::Aligned ? x86::loada_map : x86::loadu_map;
                const auto &intrin = map.lookup(vec_dtype);
                DataType func_arg_dtype = intrin.arg_dtype(0);
                check_return_type(intrin, step);
                if (DataType{arg.dtype} != func_arg_dtype && !vec_dtype.is_float()) {
                    return cpp_var_decl(step) +
                           unsafe_intrin_call(intrin, format("(const %s *)(arg%zu + i%s)", dtype_to_cpp(func_arg_dtype),
                                                             data.addr.arg, offset_str.c_str()));
                }
                return cpp_var_decl(step) +
                       unsafe_intrin_call(intrin, format("arg%zu + i%s", data.addr.arg, offset_str.c_str()));
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                VecDataType vdtype = step->dtype.as_vec();
                const x86::Intrinsic *imm_intrin = nullptr;
                if (data.right->is(StepKind::Const)) {
                    switch (data.op) {
                    case ArithBinaryOp::ShiftRightArith:
                    case ArithBinaryOp::ShiftRightLogical:
                    case ArithBinaryOp::ShiftLeftLogical:
                    case ArithBinaryOp::RotateLeft:
                    case ArithBinaryOp::RotateRight:
                        if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                            imm_intrin = x86::mask_vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype);
                        } else if (mask_pushdown != nullptr) {
                            imm_intrin = x86::maskz_vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype);
                        } else {
                            imm_intrin = x86::vector_immediate_shift_rotate_map(data.op).lookup_nothrow(vdtype);
                        }
                        break;
                    default: break;
                    }
                }
                if (imm_intrin != nullptr) {
                    if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                        auto caller = typecheck_intrin_call(*imm_intrin, result_step, mask_pushdown->merge,
                                                            mask_pushdown->mask, data.left, ScalarDataType::I32);
                        return cpp_var_decl(result_step) +
                               caller.call(mask_pushdown->merge, mask_pushdown->mask, data.left,
                                           std::to_string(data.right->step_data<StepKind::Const>().as_signed()));
                    }
                    if (mask_pushdown != nullptr) {
                        auto caller = typecheck_intrin_call(*imm_intrin, result_step, mask_pushdown->mask, data.left,
                                                            ScalarDataType::I32);
                        return cpp_var_decl(result_step) +
                               caller.call(mask_pushdown->mask, data.left,
                                           std::to_string(data.right->step_data<StepKind::Const>().as_signed()));
                    }
                    auto caller = typecheck_intrin_call(*imm_intrin, step, data.left, ScalarDataType::I32);
                    return cpp_var_decl(step) +
                           caller.call(data.left, std::to_string(data.right->step_data<StepKind::Const>().as_signed()));
                }

                if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                    return cpp_var_decl(result_step) +
                           x86_vector_arith_binary_expr(vdtype, data.op, show(data.left), show(data.right),
                                                        show(mask_pushdown->merge), show(mask_pushdown->mask));
                }

                if (mask_pushdown != nullptr) {
                    return cpp_var_decl(result_step) + x86_vector_arith_binary_expr(vdtype, data.op, show(data.left),
                                                                                    show(data.right), std::nullopt,
                                                                                    show(mask_pushdown->mask));
                }

                return cpp_var_decl(step) +
                       x86_vector_arith_binary_expr(vdtype, data.op, show(data.left), show(data.right));
            }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                SIMJIT_ASSERT(data.op != ArithUnaryOp::Not);
                if (data.op == ArithUnaryOp::RoundNearest || data.op == ArithUnaryOp::RoundDown ||
                    data.op == ArithUnaryOp::RoundUp || data.op == ArithUnaryOp::RoundTruncate) {
                    const char *imm;
                    switch (data.op) {
                    case ArithUnaryOp::RoundNearest: imm = "_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC"; break;
                    case ArithUnaryOp::RoundDown: imm = "_MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC"; break;
                    case ArithUnaryOp::RoundUp: imm = "_MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC"; break;
                    case ArithUnaryOp::RoundTruncate: imm = "_MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC"; break;
                    default: SIMJIT_UNREACHABLE();
                    }
                    if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                        const auto &intrin = x86::mask_roundscale_map.lookup(step->dtype.as_vec());
                        auto caller = typecheck_intrin_call(intrin, result_step, mask_pushdown->merge,
                                                            mask_pushdown->mask, data.arg, ScalarDataType::I32);
                        return cpp_var_decl(result_step) +
                               caller.call(mask_pushdown->merge, mask_pushdown->mask, data.arg, imm);
                    }
                    if (mask_pushdown != nullptr) {
                        const auto &intrin = x86::maskz_roundscale_map.lookup(step->dtype.as_vec());
                        auto caller = typecheck_intrin_call(intrin, result_step, mask_pushdown->mask, data.arg,
                                                            ScalarDataType::I32);
                        return cpp_var_decl(result_step) + caller.call(mask_pushdown->mask, data.arg, imm);
                    }
                    const auto &intrin = x86::roundscale_map.lookup(step->dtype.as_vec());
                    auto caller = typecheck_intrin_call(intrin, step, data.arg, ScalarDataType::I32);
                    return cpp_var_decl(step) + caller.call(data.arg, imm);
                }

                if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                    const auto &intrin = x86::mask_arith_unary_map(data.op).lookup(step->dtype.as_vec());
                    auto caller =
                        typecheck_intrin_call(intrin, result_step, mask_pushdown->merge, mask_pushdown->mask, data.arg);
                    return cpp_var_decl(result_step) + caller.call(mask_pushdown->merge, mask_pushdown->mask, data.arg);
                }

                if (mask_pushdown != nullptr) {
                    const auto &intrin = x86::maskz_arith_unary_map(data.op).lookup(step->dtype.as_vec());
                    auto caller = typecheck_intrin_call(intrin, result_step, mask_pushdown->mask, data.arg);
                    return cpp_var_decl(result_step) + caller.call(mask_pushdown->mask, data.arg);
                }

                const auto &intrin = x86::arith_unary_map(data.op).lookup(step->dtype.as_vec());
                return call_intrin_var(intrin, step, data.arg);
            }
            SIMJIT_MATCH (StepKind::Store) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    if (mask_uses_partial_byte(mdtype)) {
                        size_t step_idx = step->id;
                        std::string row = data.addr.offset == 0 ? "i" : format("(i + %zu)", data.addr.offset);
                        std::string what = show(data.what);
                        return format(R"STR({
        uint8_t *mask_byte_%zu = arg%zu + (%s >> 3);
        uint8_t mask_shift_%zu = (uint8_t)(%s & 7);
        uint8_t mask_bits_%zu = (uint8_t)(%s << mask_shift_%zu);
        *mask_byte_%zu = (uint8_t)((*mask_byte_%zu & ~mask_bits_%zu) | (((uint8_t)%s << mask_shift_%zu) & mask_bits_%zu));
    })STR",
                                      step_idx, data.addr.arg, row.c_str(),      //
                                      step_idx, row.c_str(),                     //
                                      step_idx, mask_all_ones(mdtype), step_idx, //
                                      step_idx, step_idx, step_idx, what.c_str(), step_idx, step_idx);
                    }
                    const auto &intrin = x86::store_mask_map.lookup(mdtype);
                    std::string offset_str =
                        data.addr.offset != 0 ? format(" + %zu", data.addr.offset / mask_dtype_bits(mdtype)) : "";
                    return unsafe_intrin_call(intrin,
                                              format("(%s *)arg%zu + ((i >> %zu)%s)", mask_dtype_cpp(mdtype),
                                                     data.addr.arg, mask_dtype_bits_log2(mdtype), offset_str.c_str()),
                                              show(data.what));
                }
                std::string offset_str = data.addr.offset != 0 ? format(" + %zu", data.addr.offset) : "";
                VecDataType vec_dtype = step->dtype.as_vec();
                if (data.what->is(StepKind::IntCast)) {
                    auto &cast = data.what->step_data<StepKind::IntCast>();
                    if (cast.kind == IntCastKind::Trunc && cast.arg->dtype.is_vec()) {
                        const auto *intrin = x86::cvt_storeu_map.lookup_nothrow(cast.arg->dtype.as_vec(), vec_dtype);
                        if (intrin != nullptr) {
                            const ArgumentDecl &arg = func->args[data.addr.arg];
                            DataType ptr_dtype = intrin->arg_dtype(0);
                            auto caller = typecheck_intrin_call(*intrin, step, ptr_dtype, vec_dtype.mask(), cast.arg);
                            std::string ptr = ptr_dtype == arg.dtype
                                                  ? format("arg%zu + i%s", data.addr.arg, offset_str.c_str())
                                                  : format("(%s *)(arg%zu + i%s)", dtype_to_cpp(ptr_dtype),
                                                           data.addr.arg, offset_str.c_str());
                            return caller.call(ptr, mask_all_ones(vec_dtype.mask()), cast.arg);
                        }
                    }
                }
                auto &map = data.kind == LoadStoreKind::Aligned ? x86::storea_map : x86::storeu_map;
                const auto &intrin = map.lookup(vec_dtype);
                const ArgumentDecl &arg = func->args[data.addr.arg];
                DataType ptr_dtype = intrin.arg_dtype(0);
                auto caller = typecheck_intrin_call(intrin, step, ptr_dtype, data.what);
                if (ptr_dtype == arg.dtype || vec_dtype.is_float()) {
                    return caller.call(format("arg%zu + i%s", data.addr.arg, offset_str.c_str()), data.what);
                }
                return caller.call(
                    format("(%s *)(arg%zu + i%s)", dtype_to_cpp(ptr_dtype), data.addr.arg, offset_str.c_str()),
                    data.what);
            }
            SIMJIT_MATCH (StepKind::CondStore) {
                std::string offset_str = data.addr.offset != 0 ? format(" + %zu", data.addr.offset) : "";
                VecDataType vec_dtype = step->dtype.as_vec();
                if (data.arg->is(StepKind::IntCast)) {
                    auto &cast = data.arg->step_data<StepKind::IntCast>();
                    if (cast.kind == IntCastKind::Trunc && cast.arg->dtype.is_vec()) {
                        const auto *intrin = x86::cvt_storeu_map.lookup_nothrow(cast.arg->dtype.as_vec(), vec_dtype);
                        if (intrin != nullptr) {
                            const ArgumentDecl &arg = func->args[data.addr.arg];
                            DataType ptr_dtype = intrin->arg_dtype(0);
                            auto caller = typecheck_intrin_call(*intrin, step, ptr_dtype, data.cond, cast.arg);
                            if (ptr_dtype == arg.dtype) {
                                return caller.call(format("arg%zu + i%s", data.addr.arg, offset_str.c_str()), data.cond,
                                                   cast.arg);
                            }
                            return caller.call(format("(%s *)(arg%zu + i%s)", dtype_to_cpp(ptr_dtype), data.addr.arg,
                                                      offset_str.c_str()),
                                               data.cond, cast.arg);
                        }
                    }
                }
                const auto &intrin = data.kind == LoadStoreKind::Aligned ? x86::mask_storea_map.lookup(vec_dtype)
                                                                         : x86::mask_storeu_map.lookup(vec_dtype);
                const ArgumentDecl &arg = func->args[data.addr.arg];
                DataType ptr_dtype = intrin.arg_dtype(0);
                auto caller = typecheck_intrin_call(intrin, step, ptr_dtype, data.cond, data.arg);
                if (ptr_dtype == arg.dtype) {
                    return caller.call(format("arg%zu + i%s", data.addr.arg, offset_str.c_str()), data.cond, data.arg);
                }
                return caller.call(
                    format("(%s)(arg%zu + i%s)", dtype_to_cpp(ptr_dtype), data.addr.arg, offset_str.c_str()), data.cond,
                    data.arg);
            }
            SIMJIT_MATCH (StepKind::Pack) {
                MaskDataType mdtype = data.cond->dtype.as_mask();
                VecDataType vdtype = step->dtype.as_vec();
                std::string result{};
                size_t step_idx = step->id;

                const auto &popcnt_intrin = x86::scalar_popcnt_map.lookup(mask_dtype_to_scalar(mdtype));
                const auto &compress_intrin = x86::maskz_compress_map.lookup(vdtype);
                const auto &storeu_intrin = x86::storeu_map.lookup(vdtype);

                format_to(result, "size_t base%zu = acc%zu;\n", step_idx, func->accs.index(data.acc));
                result +=
                    format("int popcnt%zu = %s;\n", step_idx, unsafe_intrin_call(popcnt_intrin, data.cond).c_str());
                format_to(result, "%s t_%zu = %s;\n", vec_dtype_cpp(vdtype), step_idx,
                          unsafe_intrin_call(compress_intrin, data.cond, data.arg).c_str());
                result += unsafe_intrin_call(storeu_intrin,
                                             (vdtype.is_float() ? "" : format("(%s *)(", vec_dtype_cpp(vdtype))) +
                                                 format("arg%zu + base%zu", data.dst, step_idx) +
                                                 (vdtype.is_float() ? "" : std::string(")")),
                                             format("t_%zu", step_idx)) +
                          ";\n";
                format_to(result, "acc%zu = base%zu + popcnt%zu", func->accs.index(data.acc), step_idx, step_idx);

                return result;
            }
            SIMJIT_MATCH (StepKind::Gather) {
                VecDataType vec_dtype = step->dtype.as_vec();
                MaskDataType mdtype = vec_dtype.mask();
                const auto &intrin = data.idx->dtype.as_vec().elem == VecElemType::I32
                                         ? x86::mask_i32gather_map.lookup(vec_dtype)
                                         : x86::mask_i64gather_map.lookup(vec_dtype);

                if (mask_pushdown != nullptr && mask_pushdown->merge != nullptr) {
                    return cpp_var_decl(result_step) +
                           unsafe_intrin_call(intrin, mask_pushdown->merge, mask_pushdown->mask, data.idx,
                                              format("arg%zu", data.data),
                                              std::to_string(vec_dtype.element_size_bytes()));
                }

                if (mask_pushdown != nullptr) {
                    const auto &setzero_intrin = x86::setzero_map.lookup(vec_dtype);
                    return cpp_var_decl(result_step) +
                           unsafe_intrin_call(intrin, unsafe_intrin_call(setzero_intrin), mask_pushdown->mask, data.idx,
                                              format("arg%zu", data.data),
                                              std::to_string(vec_dtype.element_size_bytes()));
                }

                const auto &undefined_intrin = x86::undefined_map.lookup(vec_dtype);
                return cpp_var_decl(step) +
                       unsafe_intrin_call(intrin, unsafe_intrin_call(undefined_intrin), mask_all_ones(mdtype), data.idx,
                                          format("arg%zu", data.data), std::to_string(vec_dtype.element_size_bytes()));
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                VecDataType vdtype = step->dtype.as_vec();
                const ArgumentDecl &data_arg = func->args[data.dst];
                const auto &intrin = data.idx->dtype.as_vec().elem == VecElemType::I32
                                         ? x86::i32scatter_map.lookup(vdtype)
                                         : x86::i64scatter_map.lookup(vdtype);
                auto caller =
                    typecheck_intrin_call(intrin, step, data_arg.dtype, data.idx, data.arg, ScalarDataType::I32);
                return caller.call(format("arg%zu", data.dst), data.idx, data.arg,
                                   std::to_string(vdtype.element_size_bytes()));
            }
            SIMJIT_MATCH (StepKind::CondScatter) {
                VecDataType vdtype = step->dtype.as_vec();
                const auto &intrin = data.idx->dtype.as_vec().elem == VecElemType::I32
                                         ? x86::mask_i32scatter_map.lookup(vdtype)
                                         : x86::mask_i64scatter_map.lookup(vdtype);
                return unsafe_intrin_call(intrin, format("arg%zu", data.dst), data.cond, data.idx, data.arg,
                                          std::to_string(vdtype.element_size_bytes()));
            }
            SIMJIT_MATCH (StepKind::Compare) {
                VecDataType vec_dtype = data.left->dtype.as_vec();
                if (auto expr = zero_compare_test_expr(step, mask_pushdown)) {
                    return cpp_var_decl(result_step) + *expr;
                }
                if (mask_pushdown == nullptr && vec_dtype.is_int()) {
                    if (data.op == CmpOp::Less && !data.is_unsigned && step_is_zero(data.right)) {
                        const auto &intrin = x86::mov_mask_map.lookup(vec_dtype);
                        return call_intrin_var(intrin, step, data.left);
                    }
                }
                const char *imm = vec_dtype.is_float() ? cmp_op_to_fp_imm(data.op) : cmp_op_to_int_imm(data.op);
                if (mask_pushdown != nullptr) {
                    auto expr = masked_compare_expr(step, mask_pushdown->mask);
                    if (!expr.has_value()) { messed_up("compare does not support mask pushdown"); }
                    return cpp_var_decl(result_step) + *expr;
                }
                const auto &intrin =
                    data.is_unsigned ? x86::cmpu_mask_map.lookup(vec_dtype) : x86::cmp_mask_map.lookup(vec_dtype);
                auto caller = typecheck_intrin_call(intrin, result_step, data.left, data.right, ScalarDataType::I32);
                return call_intrin_var(caller, result_step, data.left, data.right, imm);
            }
            SIMJIT_MATCH (StepKind::AccLoad) { return cpp_var_decl(step) + format("acc%zu", func->accs.index(data)); }
            SIMJIT_MATCH (StepKind::AccStore) { return format_acc_store(data.acc, show_scalar_operand(data.arg)); }
            SIMJIT_MATCH2 (StepKind::VecWidenHighHalf, StepKind::VecFloatWidenHighHalf) {
                return vec_widen_half_to_cpp(step, data.arg, data.is_unsigned, true);
            }
            SIMJIT_MATCH2 (StepKind::VecWidenLowHalf, StepKind::VecFloatWidenLowHalf) {
                return vec_widen_half_to_cpp(step, data.arg, data.is_unsigned, false);
            }
            SIMJIT_MATCH (StepKind::VecNarrowCombine) { return vec_narrow_combine_to_cpp(step, data); }
            SIMJIT_MATCH (StepKind::VecFloatNarrowCombine) { return vec_float_narrow_combine_to_cpp(step, data); }
            SIMJIT_MATCH (StepKind::IntCast) {
                if (mask_pushdown != nullptr) {
                    const auto &map = data.kind == IntCastKind::Zext
                                          ? (mask_pushdown->merge != nullptr ? x86::mask_zext_map : x86::maskz_zext_map)
                                          : (mask_pushdown->merge != nullptr ? x86::mask_cvt_map : x86::maskz_cvt_map);
                    const auto &intrin = map.lookup(data.arg->dtype.as_vec(), step->dtype.as_vec());
                    if (mask_pushdown->merge != nullptr) {
                        auto caller = typecheck_intrin_call(intrin, result_step, mask_pushdown->merge,
                                                            mask_pushdown->mask, data.arg);
                        return cpp_var_decl(result_step) +
                               caller.call(mask_pushdown->merge, mask_pushdown->mask, data.arg);
                    }
                    auto caller = typecheck_intrin_call(intrin, result_step, mask_pushdown->mask, data.arg);
                    return cpp_var_decl(result_step) + caller.call(mask_pushdown->mask, data.arg);
                }
                switch (data.kind) {
                case IntCastKind::Trunc:
                case IntCastKind::Sext: {
                    const auto &intrin = x86::cvt_map.lookup(data.arg->dtype.as_vec(), step->dtype.as_vec());
                    return call_intrin_var(intrin, step, data.arg);
                }
                case IntCastKind::Zext: {
                    const auto &intrin = x86::zext_map.lookup(data.arg->dtype.as_vec(), step->dtype.as_vec());
                    return call_intrin_var(intrin, step, data.arg);
                }
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                const auto &intrin = data.is_unsigned
                                         ? x86::float_ucast_map.lookup(data.arg->dtype.as_vec(), step->dtype.as_vec())
                                         : x86::float_cast_map.lookup(data.arg->dtype.as_vec(), step->dtype.as_vec());
                return call_intrin_var(intrin, step, data.arg);
            }
            SIMJIT_MATCH (StepKind::BitCast) {
                VecDataType vdtype = step->dtype.as_vec();
                const auto &intrin = x86::bitcast_map.lookup(vdtype);
                return call_intrin_var(intrin, step, data);
            }
            SIMJIT_MATCH (StepKind::FMA) {
                const auto &intrin = x86::fma_map(data.kind).lookup(step->dtype.as_vec());
                return call_intrin_var(intrin, step, data.x1, data.x2, data.x3);
            }
            SIMJIT_MATCH (StepKind::PredicateNot) {
                MaskDataType mdtype = step->dtype.as_mask();
                MaskDataType lookup_type = mdtype;
                if (lookup_type == MaskDataType::M2 || lookup_type == MaskDataType::M4) lookup_type = MaskDataType::M8;
                const auto &intrin = x86::not_mask_map.lookup(lookup_type);
                std::string expr = unsafe_intrin_call(intrin, data);
                if (mask_uses_partial_byte(mdtype)) { expr = format("(%s & %s)", expr.c_str(), mask_all_ones(mdtype)); }
                return cpp_var_decl(step) + expr;
            }
            SIMJIT_MATCH (StepKind::MaskBinary) {
                MaskDataType mdtype = step->dtype.as_mask();
                auto try_pushdown_into_compare = [&](const Step *cmp_step,
                                                     const Step *mask_step) -> std::optional<std::string> {
                    if (data.op != PredicateBinaryOp::And) { return std::nullopt; }
                    auto expr = masked_compare_expr(cmp_step, mask_step);
                    if (!expr.has_value()) { return std::nullopt; }
                    return cpp_var_decl(step) + *expr;
                };
                if (auto result = try_pushdown_into_compare(data.left, data.right)) { return *result; }
                if (auto result = try_pushdown_into_compare(data.right, data.left)) { return *result; }

                MaskDataType lookup_type = mdtype;
                if (lookup_type == MaskDataType::M2 || lookup_type == MaskDataType::M4) lookup_type = MaskDataType::M8;
                const auto &intrin = x86::binary_op_mask_map(data.op).lookup(lookup_type);
                std::string expr = unsafe_intrin_call(intrin, data.left, data.right);
                if (data.op == PredicateBinaryOp::XNor && mask_uses_partial_byte(mdtype)) {
                    expr = format("(%s & %s)", expr.c_str(), mask_all_ones(mdtype));
                }
                return cpp_var_decl(step) + expr;
            }
            SIMJIT_MATCH (StepKind::MaskCount) {
                MaskDataType mdtype = data->dtype.as_mask();
                MaskDataType lookup_type = mdtype;
                if (lookup_type == MaskDataType::M2 || lookup_type == MaskDataType::M4) lookup_type = MaskDataType::M8;
                const auto &intrin = x86::cvtmask_u_map.lookup(lookup_type);
                std::string expr = unsafe_intrin_call(intrin, data);
                if (mask_uses_partial_byte(mdtype)) { expr = format("(%s & %s)", expr.c_str(), mask_all_ones(mdtype)); }
                return cpp_var_decl(step) + format("__builtin_popcountll(%s)", expr.c_str());
            }
            SIMJIT_MATCH (StepKind::VecReduce) {
                uint32_t step_idx = step->id;
                VecDataType vdtype = data.arg->dtype.as_vec();
                x86::VecRegisterKind reg = x86::vec_to_x86(vdtype).reg;
                VecDataType reduce_dtype = x86::x86_to_vec(x86::Vector{x86::VecRegisterKind::ZMM, vdtype.elem});
                std::string arg = show(data.arg);
                std::string mask = format("0x%llx", (unsigned long long)((1ull << vdtype.nelems()) - 1));

                /* GCC (unlike clang, or LLVM at least) does not have xor reduce intrinsic */
                if (data.op == ArithBinaryOp::Xor) {
                    if (vdtype.is_float()) { unsupported("xor reduce is not supported"); }
                    std::string result = format("%s %s;", dtype_to_cpp(step->dtype), show(step));

                    std::string current_arg = show(data.arg);
                    if (reg == x86::VecRegisterKind::ZMM) {
                        format_to(result,
                                  R"STR(
__m256i low256_%u = _mm512_castsi512_si256(%s);
__m256i high256_%u = _mm512_extracti64x4_epi64(%s, 1);
__m256i reduced256_%u = _mm256_xor_si256(low256_%u, high256_%u);)STR",
                                  step_idx, current_arg.c_str(), //
                                  step_idx, current_arg.c_str(), //
                                  step_idx, step_idx, step_idx);
                        current_arg = format("reduced256_%u", step_idx);
                    }

                    if (reg == x86::VecRegisterKind::ZMM || reg == x86::VecRegisterKind::YMM) {
                        format_to(result,
                                  R"STR(
__m128i low128_%u = _mm256_castsi256_si128(%s);
__m128i high128_%u = _mm256_extracti128_si256(%s, 1);
__m128i reduced128_%u = _mm_xor_si128(low128_%u, high128_%u);)STR",
                                  step_idx, current_arg.c_str(), //
                                  step_idx, current_arg.c_str(), //
                                  step_idx, step_idx, step_idx);
                        current_arg = format("reduced128_%u", step_idx);
                    }

                    if (vdtype.elem == VecElemType::I32) {
                        format_to(result,
                                  R"STR(
__m128i shifted64_%u = _mm_shuffle_epi32(%s, _MM_SHUFFLE(1, 0, 3, 2));
__m128i reduced128_1_%u = _mm_xor_si128(%s, shifted64_%u);
__m128i shifted32_%u = _mm_shuffle_epi32(reduced128_1_%u, _MM_SHUFFLE(2, 3, 0, 1));
__m128i reduced128_2_%u = _mm_xor_si128(reduced128_1_%u, shifted32_%u);
%s = _mm_cvtsi128_si32(reduced128_2_%u);)STR",
                                  step_idx, current_arg.c_str(),           //
                                  step_idx, current_arg.c_str(), step_idx, //
                                  step_idx, step_idx,                      //
                                  step_idx, step_idx, step_idx,            //
                                  show(step), step_idx);
                    } else {
                        format_to(result,
                                  R"STR(
__m128i shifted64_%u = _mm_shuffle_epi32(%s, _MM_SHUFFLE(1, 0, 3, 2));
__m128i reduced128_1_%u = _mm_xor_si128(%s, shifted64_%u);
%s = _mm_cvtsi128_si64(reduced128_1_%u);)STR",
                                  step_idx, current_arg.c_str(),           //
                                  step_idx, current_arg.c_str(), step_idx, //
                                  show(step), step_idx);
                    }

                    return result;
                }

                if (reg == x86::VecRegisterKind::XMM) {
                    if (vdtype.elem == VecElemType::F32)
                        arg = format("_mm512_castps128_ps512(%s)", arg.c_str());
                    else if (vdtype.elem == VecElemType::F64)
                        arg = format("_mm512_castpd128_pd512(%s)", arg.c_str());
                    else
                        arg = format("_mm512_castsi128_si512(%s)", arg.c_str());
                } else if (reg == x86::VecRegisterKind::YMM) {
                    if (vdtype.elem == VecElemType::F32)
                        arg = format("_mm512_castps256_ps512(%s)", arg.c_str());
                    else if (vdtype.elem == VecElemType::F64)
                        arg = format("_mm512_castpd256_pd512(%s)", arg.c_str());
                    else
                        arg = format("_mm512_castsi256_si512(%s)", arg.c_str());
                }
                if (x86_is_float_minmax(vdtype, data.op)) {
                    std::string non_nan_mask = format("reduce_non_nan_mask_%u", step->id);
                    const auto &cmp_intrin = x86::mask_cmp_mask_map.lookup(reduce_dtype);
                    const auto &intrin = x86::reduce_map(data.op).lookup(reduce_dtype);
                    std::string reduce_mask = format("(%s ? %s : (%s)1)", non_nan_mask.c_str(), non_nan_mask.c_str(),
                                                     mask_dtype_cpp(reduce_dtype.mask()));
                    auto caller = typecheck_intrin_call(intrin, step, reduce_dtype.mask(), reduce_dtype);
                    std::string ordered_mask = unsafe_intrin_call(cmp_intrin, mask, arg, arg, "_CMP_ORD_Q");
                    return format("%s %s = %s;\n"
                                  "%s%s",
                                  mask_dtype_cpp(reduce_dtype.mask()), non_nan_mask.c_str(), ordered_mask.c_str(), //
                                  cpp_var_decl(step).c_str(), caller.call(reduce_mask, arg).c_str());
                }
                if (reg == x86::VecRegisterKind::ZMM) {
                    const auto &intrin = x86::unmasked_reduce_map(data.op).lookup(reduce_dtype);
                    auto caller = typecheck_intrin_call(intrin, step, reduce_dtype);
                    return cpp_var_decl(step) + caller.call(arg);
                }

                const auto &intrin = x86::reduce_map(data.op).lookup(reduce_dtype);
                auto caller = typecheck_intrin_call(intrin, step, reduce_dtype.mask(), reduce_dtype);
                return cpp_var_decl(step) + caller.call(mask, arg);
            }
            SIMJIT_MATCH (StepKind::MaskReduce) {
                MaskDataType mask = data.arg->dtype.as_mask();
                MaskDataType lookup_type = mask;
                if (mask == MaskDataType::M2 || mask == MaskDataType::M4) lookup_type = MaskDataType::M8;

                switch (data.op) {
                case PredicateBinaryOp::And: {
                    const auto &intrin = x86::ktestc_map.lookup(lookup_type);
                    return cpp_var_decl(step) + unsafe_intrin_call(intrin, show(data.arg), mask_all_ones(mask));
                }
                case PredicateBinaryOp::Or: {
                    const auto &intrin = x86::ktestz_map.lookup(lookup_type);
                    return cpp_var_decl(step) + "!" + unsafe_intrin_call(intrin, show(data.arg), mask_all_ones(mask));
                }
                case PredicateBinaryOp::Xor: {
                    ScalarDataType scalar = mask_dtype_to_scalar(mask);
                    const auto &intrin = x86::scalar_popcnt_map.lookup(scalar);
                    return cpp_var_decl(step) + format("%s & 1", unsafe_intrin_call(intrin, show(data.arg)).c_str());
                }
                case PredicateBinaryOp::AndNot:
                case PredicateBinaryOp::XNor:
                    messed_up("don't support MaskReduce for %s op", show_predicate_binary_op(data.op));
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::MaskCombine) {
                MaskDataType mdtype = step->dtype.as_mask();
                switch (mdtype) {
                case MaskDataType::M2:
                    return cpp_var_decl(step) + format("(%s << 1) | %s", show(data.right), show(data.left));
                case MaskDataType::M4:
                    return cpp_var_decl(step) + format("(%s << 2) | %s", show(data.right), show(data.left));
                case MaskDataType::M8:
                    return cpp_var_decl(step) + format("(%s << 4) | %s", show(data.right), show(data.left));
                default: break;
                }
                const auto &intrin = x86::kunpack_map.lookup(mdtype);
                return call_intrin_var(intrin, step, data.right, data.left);
            }
            SIMJIT_MATCH (StepKind::Select) {
                VecDataType vdtype = step->dtype.as_vec();
                if (step_is_zero(data.falsy)) {
                    if (vdtype.is_int() && data.truthy->is(StepKind::Const)) {
                        uint64_t all_ones = scalar_dtype_umax(vec_elem_to_scalar(vdtype.elem));
                        if ((data.truthy->step_data<StepKind::Const>().as_unsigned() & all_ones) == all_ones) {
                            const auto &intrin = x86::movm_map.lookup(vdtype);
                            return call_intrin_var(intrin, step, data.cond);
                        }
                    }
                    if (step_supports_mask_pushdown(data.truthy, false)) {
                        MaskPushdownInfo pushdown{data.cond, nullptr, step};
                        return step_to_cpp(data.truthy, &pushdown);
                    }
                    const auto &intrin = x86::maskz_mov_map.lookup(vdtype);
                    return call_intrin_var(intrin, step, data.cond, data.truthy);
                }
                if (step_supports_mask_pushdown(data.truthy, true)) {
                    MaskPushdownInfo pushdown{data.cond, data.falsy, step};
                    return step_to_cpp(data.truthy, &pushdown);
                }
                const auto &intrin = x86::mask_blend_map.lookup(vdtype);
                return call_intrin_var(intrin, step, data.cond, data.falsy, data.truthy);
            }
            SIMJIT_MATCH (StepKind::Ternarylogic) {
                const auto &intrin = x86::ternarylogic_map.lookup(data.lookup_type);
                return cpp_var_decl(step) +
                       unsafe_intrin_call(intrin, data.a, data.b, data.c, std::to_string(data.fun));
            }
            SIMJIT_MATCH (StepKind::VecConst) {
                VecDataType vdtype = step->dtype.as_vec();
                size_t idx = step->id;
                std::string result;
                switch (vdtype.elem) {
                case VecElemType::I8:
                    result = format("static const uint8_t const_mem_%zu[] = {", idx);
                    for (size_t i = 0; i < vdtype.nelems(); ++i) {
                        simjit::format_to(result, "0x%02hhx, ", ((uint8_t *)data.mem)[i]);
                    }
                    result += "};";
                    break;
                case VecElemType::I16:
                    result = format("static const uint16_t const_mem_%zu[] = {", idx);
                    for (size_t i = 0; i < vdtype.nelems(); ++i) {
                        simjit::format_to(result, "0x%04hx, ", ((uint16_t *)data.mem)[i]);
                    }
                    result += "};";
                    break;
                case VecElemType::I32:
                    result = format("static const uint32_t const_mem_%zu[] = {", idx);
                    for (size_t i = 0; i < vdtype.nelems(); ++i) {
                        simjit::format_to(result, "0x%08x, ", ((uint32_t *)data.mem)[i]);
                    }
                    result += "};";
                    break;
                case VecElemType::I64:
                    result = format("static const uint64_t const_mem_%zu[] = {", idx);
                    for (size_t i = 0; i < vdtype.nelems(); ++i) {
                        simjit::format_to(result, "0x%016llx, ", (unsigned long long)(((uint64_t *)data.mem)[i]));
                    }
                    result += "};";
                    break;
                case VecElemType::F32:
                case VecElemType::F64: messed_up("unexpected float type in vecconst");
                }

                const auto &intrin = x86::loadu_map.lookup(vdtype);
                result += cpp_var_decl(step) +
                          unsafe_intrin_call(intrin, format("(%s *)const_mem_%zu", dtype_to_cpp(vdtype), idx));
                return result;
            }
            SIMJIT_MATCH (StepKind::VecPermute) {
                VecDataType vdtype = step->dtype.as_vec();
                if (data.is_bit) {
                    const auto &intrin = x86::gp2affine_map.lookup(vdtype);
                    return cpp_var_decl(step) + unsafe_intrin_call(intrin, data.arg, data.permute_idxs, "0");
                }
                const auto &intrin = x86::shuffle8_map.lookup(vdtype);
                return call_intrin_var(intrin, step, data.arg, data.permute_idxs);
            }
            SIMJIT_MATCH (StepKind::Fpclass) {
                VecDataType vdtype = data.arg->dtype.as_vec();
                const auto &intrin = x86::fpclass_map.lookup(vdtype);
                auto caller = typecheck_intrin_call(intrin, step, data.arg, ScalarDataType::I32);
                std::string classes{};
                SIMJIT_ASSERT(classes.empty());
                auto append = [&](std::string_view a) {
                    if (classes.empty())
                        classes = a;
                    else
                        classes += " | " + std::string{a};
                };
                if (bool(data.flags & FpClass::FPC_NAN)) { append("0x01 | 0x80"); }
                if (bool(data.flags & FpClass::FPC_INFINITE)) { append("0x08 | 0x10"); }
                if (bool(data.flags & FpClass::FPC_SUBNORMAL)) { append("0x20"); }
                if (bool(data.flags & FpClass::FPC_ZERO)) { append("0x02 | 0x04"); }
                return cpp_var_decl(step) + caller.call(data.arg, classes);
            }
        }
        SIMJIT_UNREACHABLE();
    }
};

std::unique_ptr<CppEmitterBase> make_x86_cpp_emitter(const Function *func) {
    return std::make_unique<X86CppEmitter>(func);
}

#undef unsupported
#undef messed_up

} // namespace cpp_backend
} // namespace simjit
