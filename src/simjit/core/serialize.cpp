// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/hir.h"

#include "headerlisp.h"
#include "simjit/detail/base.h"
#include "simjit/dynamic_value.h"
#include "simjit/simjit.h"

#include <utility>

namespace hl = headerlisp;

#define SV(x) int((x).size()), ((x).data() == nullptr ? "" : (x).data())
#define serialization_error(...)                                                                  \
    simjit_exception(simjit::ErrorModule::Serialization, simjit::ErrorKind::SerializationFailure, \
                     simjit::ErrorSubKind::SerializationParseError, __VA_ARGS__)

#define DEFAULT_TO_LIST(_dt, _show)                                      \
    template <> struct hl::to_list<_dt> {                                \
        hl::value operator()(_dt x) { return hl::make_value(_show(x)); } \
    }

#define INT_TO_FROM_LIST(_dt)                                          \
    template <> struct hl::to_list<_dt> {                              \
        hl::value operator()(_dt x) { return hl::make_value((int)x); } \
    };                                                                 \
    template <> struct hl::from_list<_dt> {                            \
        _dt operator()(hl::value x) { return (_dt)x.as_int(); }        \
    }

DEFAULT_TO_LIST(simjit::ScalarDataType, simjit::show_scalar_dtype);
DEFAULT_TO_LIST(simjit::VecElemType, simjit::show_vec_elem_type);
DEFAULT_TO_LIST(simjit::VecSize, simjit::show_vec_size);
DEFAULT_TO_LIST(simjit::MaskDataType, simjit::show_mask_dtype);
DEFAULT_TO_LIST(simjit::ArithUnaryOp, simjit::show_arith_unary_op);
DEFAULT_TO_LIST(simjit::ArithBinaryOp, simjit::show_arith_binary_op);
DEFAULT_TO_LIST(simjit::CmpOp, simjit::show_cmp_op);
DEFAULT_TO_LIST(simjit::IntCastKind, simjit::show_int_cast_kind);
DEFAULT_TO_LIST(simjit::PredicateBinaryOp, simjit::show_predicate_binary_op);
DEFAULT_TO_LIST(simjit::LoadStoreKind, simjit::show_load_store_kind);
DEFAULT_TO_LIST(simjit::hir::StepKind, simjit::hir::show_step_kind);
INT_TO_FROM_LIST(size_t);

template <> struct hl::from_list<simjit::ScalarDataType> {
    simjit::ScalarDataType operator()(hl::value dt) {
        std::string_view s = dt.as_string_view();
        if (s == "i8") return simjit::ScalarDataType::I8;
        if (s == "i16") return simjit::ScalarDataType::I16;
        if (s == "i32") return simjit::ScalarDataType::I32;
        if (s == "i64") return simjit::ScalarDataType::I64;
        if (s == "i1") return simjit::ScalarDataType::I1;
        if (s == "f32") return simjit::ScalarDataType::F32;
        if (s == "f64") return simjit::ScalarDataType::F64;
        if (s == "i128") return simjit::ScalarDataType::I128;
        serialization_error("Invalid ScalarDataType value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::VecElemType> {
    simjit::VecElemType operator()(hl::value dt) {
        std::string_view s = dt.as_string_view();
        if (s == "i8") return simjit::VecElemType::I8;
        if (s == "i16") return simjit::VecElemType::I16;
        if (s == "i32") return simjit::VecElemType::I32;
        if (s == "i64") return simjit::VecElemType::I64;
        if (s == "f32") return simjit::VecElemType::F32;
        if (s == "f64") return simjit::VecElemType::F64;
        serialization_error("Invalid VecElemType value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::MaskDataType> {
    simjit::MaskDataType operator()(hl::value dt) {
        std::string_view s = dt.as_string_view();
        if (s == "m2" || s == "M2") return simjit::MaskDataType::M2;
        if (s == "m4" || s == "M4") return simjit::MaskDataType::M4;
        if (s == "m8" || s == "M8") return simjit::MaskDataType::M8;
        if (s == "m16" || s == "M16") return simjit::MaskDataType::M16;
        if (s == "m32" || s == "M32") return simjit::MaskDataType::M32;
        if (s == "m64" || s == "M64") return simjit::MaskDataType::M64;
        serialization_error("Invalid MaskDataType value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::VecSize> {
    simjit::VecSize operator()(hl::value dt) {
        std::string_view s = dt.as_string_view();
        if (s == "x2") return simjit::VecSize::X2;
        if (s == "x4") return simjit::VecSize::X4;
        if (s == "x8") return simjit::VecSize::X8;
        if (s == "x16") return simjit::VecSize::X16;
        if (s == "x32") return simjit::VecSize::X32;
        if (s == "x64") return simjit::VecSize::X64;
        serialization_error("Invalid VecSize value %.*s", SV(s));
    }
};

template <> struct hl::to_list<simjit::VecDataType> {
    hl::value operator()(simjit::VecDataType dt) { return hl::cons(dt.elem, dt.size); }
};
template <> struct hl::from_list<simjit::VecDataType> {
    simjit::VecDataType operator()(hl::value dt) {
        auto [a, b] = hl::unapply_cons(dt);
        return simjit::VecDataType{b.as<simjit::VecSize>(), a.as<simjit::VecElemType>()};
    }
};

template <> struct hl::to_list<simjit::DataType> {
    hl::value operator()(simjit::DataType dt) {
        switch (dt.kind) {
        case simjit::DataTypeKind::Scalar: return hl::list("scalar", dt.as_scalar());
        case simjit::DataTypeKind::Vec: return hl::list("vec", dt.as_vec());
        case simjit::DataTypeKind::Mask: return hl::list("mask", dt.as_mask());
        }
        SIMJIT_UNREACHABLE();
    }
};
template <> struct hl::from_list<simjit::DataType> {
    simjit::DataType operator()(hl::value dt) {
        if (auto m = hl::match(dt, "scalar", hl::cap_as<simjit::ScalarDataType>)) {
            auto [value] = *m;
            return value;
        }
        if (auto m = hl::match(dt, "vec", hl::cap_as<simjit::VecDataType>)) {
            auto [value] = *m;
            return value;
        }
        if (auto m = hl::match(dt, "mask", hl::cap_as<simjit::MaskDataType>)) {
            auto [value] = *m;
            return value;
        }
        serialization_error("Invalid DataType value %s", hl::print(dt).c_str());
    }
};

template <> struct hl::from_list<simjit::ArithBinaryOp> {
    simjit::ArithBinaryOp operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "add" || s == "+") return simjit::ArithBinaryOp::Add;
        if (s == "sub" || s == "-") return simjit::ArithBinaryOp::Sub;
        if (s == "mul" || s == "*") return simjit::ArithBinaryOp::Mul;
        if (s == "mulse" || s == "*SE") return simjit::ArithBinaryOp::Mul64SE;
        if (s == "mulze" || s == "*ZE") return simjit::ArithBinaryOp::Mul64ZE;
        if (s == "min") return simjit::ArithBinaryOp::Min;
        if (s == "max") return simjit::ArithBinaryOp::Max;
        if (s == "umin") return simjit::ArithBinaryOp::UMin;
        if (s == "umax") return simjit::ArithBinaryOp::UMax;
        if (s == "and" || s == "&") return simjit::ArithBinaryOp::And;
        if (s == "or" || s == "|") return simjit::ArithBinaryOp::Or;
        if (s == "xor" || s == "^") return simjit::ArithBinaryOp::Xor;
        if (s == "andnot" || s == "!&") return simjit::ArithBinaryOp::AndNot;
        if (s == "shra") return simjit::ArithBinaryOp::ShiftRightArith;
        if (s == "shrl") return simjit::ArithBinaryOp::ShiftRightLogical;
        if (s == "shll") return simjit::ArithBinaryOp::ShiftLeftLogical;
        if (s == "rol") return simjit::ArithBinaryOp::RotateLeft;
        if (s == "ror") return simjit::ArithBinaryOp::RotateRight;
        if (s == "div" || s == "/") return simjit::ArithBinaryOp::Div;
        if (s == "udiv" || s == "u/") return simjit::ArithBinaryOp::UDiv;
        if (s == "mod" || s == "%") return simjit::ArithBinaryOp::Mod;
        if (s == "umod" || s == "u%") return simjit::ArithBinaryOp::UMod;
        serialization_error("Invalid ArithBinaryOp value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::ArithUnaryOp> {
    simjit::ArithUnaryOp operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "not" || s == "~") return simjit::ArithUnaryOp::Not;
        if (s == "neg" || s == "negate" || s == "-") return simjit::ArithUnaryOp::Negate;
        if (s == "abs") return simjit::ArithUnaryOp::Abs;
        if (s == "lzcnt") return simjit::ArithUnaryOp::Lzcnt;
        if (s == "tzcnt") return simjit::ArithUnaryOp::Tzcnt;
        if (s == "popcount" || s == "popcnt") return simjit::ArithUnaryOp::Popcount;
        if (s == "round") return simjit::ArithUnaryOp::RoundNearest;
        if (s == "floor") return simjit::ArithUnaryOp::RoundDown;
        if (s == "ceil") return simjit::ArithUnaryOp::RoundUp;
        if (s == "trunc") return simjit::ArithUnaryOp::RoundTruncate;
        if (s == "rcp") return simjit::ArithUnaryOp::Rcp;
        if (s == "sqrt") return simjit::ArithUnaryOp::Sqrt;
        if (s == "rsqrt") return simjit::ArithUnaryOp::Rsqrt;
        serialization_error("Invalid ArithUnaryOp value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::CmpOp> {
    simjit::CmpOp operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "lt") return simjit::CmpOp::Less;
        if (s == "gt") return simjit::CmpOp::Greater;
        if (s == "le") return simjit::CmpOp::LessEqual;
        if (s == "ge") return simjit::CmpOp::GreaterEqual;
        if (s == "eq") return simjit::CmpOp::Equal;
        if (s == "ne") return simjit::CmpOp::NotEqual;
        serialization_error("Invalid CmpOp value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::PredicateBinaryOp> {
    simjit::PredicateBinaryOp operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "and" || s == "&") return simjit::PredicateBinaryOp::And;
        if (s == "or" || s == "|") return simjit::PredicateBinaryOp::Or;
        if (s == "xor" || s == "^") return simjit::PredicateBinaryOp::Xor;
        if (s == "andnot" || s == "!&") return simjit::PredicateBinaryOp::AndNot;
        if (s == "xnor" || s == "!^") return simjit::PredicateBinaryOp::XNor;
        serialization_error("Invalid PredicateBinaryOp value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::IntCastKind> {
    simjit::IntCastKind operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "trunc") return simjit::IntCastKind::Trunc;
        if (s == "sext") return simjit::IntCastKind::Sext;
        if (s == "zext") return simjit::IntCastKind::Zext;
        serialization_error("Invalid IntCastKind value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::LoadStoreKind> {
    simjit::LoadStoreKind operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "aligned") return simjit::LoadStoreKind::Aligned;
        if (s == "unaligned") return simjit::LoadStoreKind::Unaligned;
        serialization_error("Invalid LoadStoreKind value %.*s", SV(s));
    }
};

template <> struct hl::from_list<simjit::hir::StepKind> {
    simjit::hir::StepKind operator()(hl::value x) {
        std::string_view s = x.as_string_view();
        if (s == "load-splat") return simjit::hir::StepKind::LoadSplat;
        if (s == "const") return simjit::hir::StepKind::Const;
        if (s == "load") return simjit::hir::StepKind::Load;
        if (s == "gather") return simjit::hir::StepKind::Gather;
        if (s == "binary") return simjit::hir::StepKind::ArithBinary;
        if (s == "checked-op") return simjit::hir::StepKind::CheckedOp;
        if (s == "predicate-binary") return simjit::hir::StepKind::PredicateBinary;
        if (s == "unary") return simjit::hir::StepKind::ArithUnary;
        if (s == "int-cast") return simjit::hir::StepKind::IntCast;
        if (s == "float-cast") return simjit::hir::StepKind::FloatCast;
        if (s == "store") return simjit::hir::StepKind::Store;
        if (s == "cmp") return simjit::hir::StepKind::Compare;
        if (s == "acc-arith-bin") return simjit::hir::StepKind::AccArithBinary;
        if (s == "acc-predicate-bin") return simjit::hir::StepKind::AccPredicateBinary;
        if (s == "countif") return simjit::hir::StepKind::Countif;
        if (s == "predicate-not") return simjit::hir::StepKind::PredicateNot;
        if (s == "select") return simjit::hir::StepKind::Select;
        if (s == "index") return simjit::hir::StepKind::Index;
        if (s == "scatter") return simjit::hir::StepKind::Scatter;
        if (s == "pack") return simjit::hir::StepKind::Pack;
        if (s == "sum128") return simjit::hir::StepKind::AccSum128;
        if (s == "permute") return simjit::hir::StepKind::Permute;
        if (s == "bitcast") return simjit::hir::StepKind::BitCast;
        if (s == "fpclass") return simjit::hir::StepKind::Fpclass;
        serialization_error("Invalid StepKind value %.*s", SV(s));
    }
};

template <> struct hl::to_list<simjit::FpClass> {
    hl::value operator()(simjit::FpClass x) {
        hl::list_builder b{};
        if (bool(x & simjit::FpClass::FPC_INFINITE)) b.add("inf");
        if (bool(x & simjit::FpClass::FPC_NAN)) b.add("nan");
        if (bool(x & simjit::FpClass::FPC_SUBNORMAL)) b.add("sub");
        if (bool(x & simjit::FpClass::FPC_ZERO)) b.add("zer");
        if (!b.empty()) { return hl::car(b.list()); }
        return b.list();
    }
};

template <> struct hl::from_list<simjit::FpClass> {
    simjit::FpClass operator()(hl::value x) {
        if (hl::is_string(x)) { x = hl::list(x); }

        simjit::FpClass result{};
        for (std::string_view it : x.iter().as<std::string_view>()) {
            if (it == "inf")
                result |= simjit::FpClass::FPC_INFINITE;
            else if (it == "nan")
                result |= simjit::FpClass::FPC_NAN;
            else if (it == "sub")
                result |= simjit::FpClass::FPC_SUBNORMAL;
            else if (it == "zer")
                result |= simjit::FpClass::FPC_ZERO;
            else
                serialization_error("Invalid FpClass entry %.*s", SV(it));
        }

        return result;
    }
};

template <> struct hl::to_list<simjit::ArithBinaryOpFlags> {
    hl::value operator()(simjit::ArithBinaryOpFlags x) {
        hl::list_builder b{};
        if (bool(x & simjit::ArithBinaryOpFlags::SafetyCheck)) b.add("overflow");
        if (bool(x & simjit::ArithBinaryOpFlags::ShiftWraparound)) b.add("shift-wraparound");
        if (bool(x & simjit::ArithBinaryOpFlags::SafeDivision)) b.add("safe-division");
        if (b.size() == 1) { return hl::car(b.list()); }
        return b.list();
    }
};

template <> struct hl::from_list<simjit::ArithBinaryOpFlags> {
    simjit::ArithBinaryOpFlags operator()(hl::value x) {
        simjit::ArithBinaryOpFlags result{};
        if (hl::is_string(x)) { x = hl::list(x); }

        for (std::string_view it : x.iter().as<std::string_view>()) {
            if (it == "overflow")
                result |= simjit::ArithBinaryOpFlags::SafetyCheck;
            else if (it == "shift-wraparound")
                result |= simjit::ArithBinaryOpFlags::ShiftWraparound;
            else if (it == "safe-division")
                result |= simjit::ArithBinaryOpFlags::SafeDivision;
            else
                serialization_error("Invalid ArithBinaryOpFlags entry %.*s", SV(it));
        }

        return result;
    }
};

template <> struct hl::to_list<simjit::ArgumentKind> {
    hl::value operator()(simjit::ArgumentKind x) {
        hl::list_builder b{};
        if (bool(x & simjit::ArgumentKind::SrcArr)) b.add("src-arr");
        if (bool(x & simjit::ArgumentKind::SrcIdxArr)) b.add("src-idx-arr");
        if (bool(x & simjit::ArgumentKind::SrcConst)) b.add("src-const");
        if (bool(x & simjit::ArgumentKind::Dst)) b.add("dst-arr");
        if (bool(x & simjit::ArgumentKind::DstAgg)) b.add("dst-scalar");
        if (bool(x & simjit::ArgumentKind::SrcGatherArr)) b.add("src-gather-arr");
        if (bool(x & simjit::ArgumentKind::DstSafetyCheck)) b.add("dst-overflow");
        if (b.size() == 1) { return hl::car(b.list()); }
        return b.list();
    }
};

template <> struct hl::from_list<simjit::ArgumentKind> {
    simjit::ArgumentKind operator()(hl::value x) {
        simjit::ArgumentKind result{};
        if (hl::is_string(x)) { x = hl::list(x); }

        for (std::string_view it : x.iter().as<std::string_view>()) {
            if (it == "src-arr")
                result |= simjit::ArgumentKind::SrcArr;
            else if (it == "src-idx-arr")
                result |= simjit::ArgumentKind::SrcIdxArr;
            else if (it == "src-const")
                result |= simjit::ArgumentKind::SrcConst;
            else if (it == "dst-arr")
                result |= simjit::ArgumentKind::Dst;
            else if (it == "dst-scalar")
                result |= simjit::ArgumentKind::DstAgg;
            else if (it == "src-gather-arr")
                result |= simjit::ArgumentKind::SrcGatherArr;
            else if (it == "dst-overflow")
                result |= simjit::ArgumentKind::DstSafetyCheck;
            else
                serialization_error("Invalid ArgumentKind entry %.*s", SV(it));
        }
        return result;
    }
};

namespace simjit {

using namespace hir;

#define GUARD_DESERIALIZE(_cond)                                                                                     \
    do {                                                                                                             \
        if (!(_cond))                                                                                                \
            serialization_error("[%s] Deserialization error: check %s failed", debug_context_str().c_str(), #_cond); \
    } while (0)

#define GUARD_EQUAL(_left, _right)                                                                                \
    do {                                                                                                          \
        auto __l = hl::make_value(_left);                                                                         \
        auto __r = hl::make_value(_right);                                                                        \
        if (!hl::is_equal(__l, __r))                                                                              \
            serialization_error("[%s] Deserialization error: %s is not equal to %s", debug_context_str().c_str(), \
                                hl::print(__l).c_str(), hl::print(__r).c_str());                                  \
    } while (0)

static thread_local std::vector<std::string> debug_context_stack;

namespace {
struct DebugContextGuard {
    DebugContextGuard() = delete;
    explicit DebugContextGuard(std::string name) { debug_context_stack.push_back(std::move(name)); }
    explicit DebugContextGuard(hl::value value) { debug_context_stack.push_back(hl::print(value)); }
    ~DebugContextGuard() {
        // Don't clear context if exception was thrown
        if (std::uncaught_exceptions() == 0) debug_context_stack.pop_back();
    }
};
} // namespace

static std::string debug_context_str() {
    std::string result;
    result.reserve(32);
    for (const auto &s : debug_context_stack) {
        if (!result.empty()) { result += "|"; }
        result += s;
    }
    return result;
}

static hl::value serialize_args(const Function *func) {
    hl::list_builder b{};
    size_t i = 0;
    for (const auto &arg : func->args) {
        SIMJIT_ASSERT(arg.idx == i);
        hl::value it = hl::list("arg", arg.idx, arg.dtype, arg.kind);
        b.add(it);
        ++i;
    }
    (void)i;
    return hl::cons("args", b.list());
}

static hl::value serialize_step_ref(const Step *step, const std::vector<size_t> &id_map) {
    return hl::list("step", id_map[step->id]);
}

static hl::value serialize_arg_ref(ArgumentIdx idx) {
    return hl::list("arg", idx);
}

static hl::value serialize_acc_ref(AccIdx idx) {
    return hl::list("acc", idx);
}

static hl::value serialize_accs(const Function *func, const std::vector<size_t> &id_map) {
    hl::list_builder b{};
    for (const auto &acc : func->accs) {
        hl::value it = hl::list("acc", acc.idx, acc.dtype, serialize_arg_ref(acc.dst_arg),
                                serialize_step_ref(acc.agg_expr, id_map));
        b.add(it);
    }
    return hl::cons("accs", b.list());
}

static hl::value serialize_step(const Step *step, const std::vector<size_t> &id_map) {
    hl::value id = hl::make_value((int)id_map[step->id]);
    hl::value kind = hl::make_value(step->kind);
    hl::value dtype = hl::make_value(step->dtype);
#define ref(_x) serialize_step_ref(_x, id_map)
    hl::value payload;
    switch (step->kind) {
        SIMJIT_MATCH (StepKind::Const) {
            payload = hl::list(simjit::format("0x%llx", (unsigned long long)data.raw_bits()));
            break;
        }
        SIMJIT_MATCH (StepKind::Load) {
            payload = hl::list(serialize_arg_ref(data.idx), data.kind);
            break;
        }
        SIMJIT_MATCH (StepKind::LoadSplat) {
            payload = hl::list(serialize_arg_ref(data.idx));
            break;
        }
        SIMJIT_MATCH (StepKind::Index) { break; }
        SIMJIT_MATCH (StepKind::Gather) {
            payload = hl::list(ref(data.idx), serialize_arg_ref(data.data));
            break;
        }
        SIMJIT_MATCH (StepKind::ArithBinary) {
            payload = hl::list_dot(data.op, ref(data.left), ref(data.right),
                                   data.flags != ArithBinaryOpFlags::No ? hl::list(data.flags) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::CheckedOp) {
            payload = hl::list_dot(ref(data.op), data.mask ? hl::list(ref(data.mask)) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::ArithUnary) {
            payload = hl::list(data.op, ref(data.arg));
            break;
        }
        SIMJIT_MATCH (StepKind::Store) {
            payload = hl::list_dot(ref(data.what), serialize_arg_ref(data.addr), data.kind,
                                   data.cond ? hl::list(ref(data.cond)) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::Compare) {
            payload =
                hl::list_dot(data.op, ref(data.left), ref(data.right), data.is_unsigned ? hl::list(true) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::IntCast) {
            payload = hl::list(data.kind, ref(data.arg));
            break;
        }
        SIMJIT_MATCH (StepKind::FloatCast) {
            payload = hl::list_dot(ref(data.arg), data.is_unsigned ? hl::list(true) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::PredicateNot) {
            payload = hl::list(ref(data));
            break;
        }
        SIMJIT_MATCH (StepKind::PredicateBinary) {
            payload = hl::list(data.op, ref(data.left), ref(data.right));
            break;
        }
        SIMJIT_MATCH (StepKind::AccPredicateBinary) {
            payload = hl::list(data.op, ref(data.arg), serialize_acc_ref(data.acc));
            break;
        }
        SIMJIT_MATCH2 (StepKind::AccArithBinary, StepKind::AccSum128) {
            payload = hl::list_dot(data.op, ref(data.arg), serialize_acc_ref(data.acc),
                                   data.cond ? hl::list(ref(data.cond)) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::Countif) {
            payload = hl::list(ref(data.arg), serialize_acc_ref(data.acc));
            break;
        }
        SIMJIT_MATCH (StepKind::Select) {
            payload = hl::list(ref(data.cond), ref(data.truthy), ref(data.falsy));
            break;
        }
        SIMJIT_MATCH (StepKind::Scatter) {
            payload = hl::list_dot(ref(data.arg), ref(data.idx), serialize_arg_ref(data.dst),
                                   data.cond ? hl::list(ref(data.cond)) : nullptr);
            break;
        }
        SIMJIT_MATCH (StepKind::Pack) {
            payload = hl::list(ref(data.arg), ref(data.cond), serialize_arg_ref(data.dst),
                               serialize_acc_ref(data.dst_size_acc));
            break;
        }
        SIMJIT_MATCH (StepKind::Permute) {
            payload = hl::list(ref(data.arg), simjit::format("0x%llx", (unsigned long long)data.permute), data.is_bit);
            break;
        }
        SIMJIT_MATCH (StepKind::BitCast) {
            payload = hl::list(ref(data));
            break;
        }
        SIMJIT_MATCH (StepKind::Fpclass) {
            payload = hl::list(ref(data.arg), data.flags);
            break;
        }
    }
#undef ref
    return hl::list_dot("step", id, kind, dtype, payload);
}

static hl::value serialize_headerlisp(const Function *func) {
    hl::value steps_head, steps_tail;
    hl::value roots_head, roots_tail;

    size_t id_counter = 0;
    std::vector<size_t> id_map(func->step_id_count, 0);
    std::vector<uint8_t> traversal_state(func->step_id_count, 0);
    for (Step *root : func->step_roots) {
        traverse_steps_postorder_unique(root, traversal_state, [&](Step *step) {
            id_map[step->id] = id_counter++;
            hl::add_last(steps_head, steps_tail, serialize_step(step, id_map));
        });
        hl::add_last(roots_head, roots_tail, serialize_step_ref(root, id_map));
    }

    hl::list_builder b{};
    b.add("func");
    b.add(serialize_args(func));
    if (!func->accs.empty()) { b.add(serialize_accs(func, id_map)); }
    b.add(hl::cons("steps", steps_head));
    b.add(hl::cons("roots", roots_head));
    if (func->scalar_only) { b.add(hl::list("scalar-only")); }
    return b.list();
}

std::string serialize(const Function *func) {
    hl::context_guard guard{};
    hl::value serialized = serialize_headerlisp(func);
    return hl::print(serialized);
}

static void process_deserialized_args(hl::value args, FunctionBuilder &builder) {
    DebugContextGuard g{"args"};

    for (auto arg : args.iter()) {
        DebugContextGuard guard{arg};

        auto [idx, dtype, kind] = hl::require_match("invalid arg", arg, "arg", hl::cap_as<int>,
                                                    hl::cap_as<ScalarDataType>, hl::cap_as<ArgumentKind>);
        if (kind == ArgumentKind::DstSafetyCheck) {
            Argument arg_id = builder.arg_safety_check();
            GUARD_DESERIALIZE((int)arg_id.idx_ == idx);
        } else {
            Argument arg_id = builder.arg(dtype, kind);
            GUARD_DESERIALIZE((int)arg_id.idx_ == idx);
        }
    }
}

static int parse_step_ref(hl::value v) {
    DebugContextGuard guard{v};
    auto [idx] = hl::require_match("Invalid step ref", v, "step", hl::cap_as<int>);
    return idx;
}

static Argument parse_arg_ref(hl::value v) {
    DebugContextGuard guard{v};
    auto [idx] = hl::require_match("Invalid arg ref", v, "arg", hl::cap_as<int>);
    return Argument{(size_t)idx};
}

static AccIdx parse_acc_ref(hl::value v) {
    DebugContextGuard guard{v};
    auto [idx] = hl::require_match("Invalid acc ref", v, "acc", hl::cap_as<int>);
    return (size_t)idx;
}

static DynamicValue process_deserialized_step(hl::value step, FunctionBuilder &builder,
                                              const std::unordered_map<int, DynamicValue> &step_map,
                                              const std::unordered_map<AccIdx, Argument> &acc_arg_map) {
    DebugContextGuard guard{step};

    auto [kind, dtype, payload] = hl::require_match(hl::print(step).c_str(), step, hl::cap_as<hir::StepKind>,
                                                    hl::cap_as<ScalarDataType>, hl::rest);

    auto ref_value = [&step_map](hl::value v) {
        auto maybe_x = step_map.at(parse_step_ref(v)).as_value();
        if (!maybe_x) serialization_error("non-existent step %s", hl::print(v).c_str());
        return *maybe_x;
    };

    auto ref_predicate = [&step_map](hl::value v) {
        auto maybe_x = step_map.at(parse_step_ref(v)).as_predicate();
        if (!maybe_x) serialization_error("non-existent step %s", hl::print(v).c_str());
        return *maybe_x;
    };

#define ref(_v) step_map.at(parse_step_ref(_v))
#define bind_step(...) hl::require_match(hl::print(step).c_str(), payload, __VA_ARGS__)

    switch (kind) {
    case StepKind::Const: {
        auto [x] = bind_step(hl::cap);
        if (hl::is_num(x)) { return builder.con_internal(ConstData::i64(x.as_int()), dtype); }
        uint64_t data = strtoull(x.as_c_str(), nullptr, 16);
        if (dtype == ScalarDataType::I1) { return builder.i1(data); }
        // No type-specific logic, plain bits all the time.
        return builder.con_internal(ConstData::u64(data), dtype);
    }
    case StepKind::ArithBinary: {
        auto [op, l, r, flags] =
            bind_step(hl::cap_as<ArithBinaryOp>, hl::cap, hl::cap, hl::or_nil_as<ArithBinaryOpFlags>);
        return builder.arith_binary(ref_value(l), ref_value(r), op, flags);
    }
    case StepKind::CheckedOp: {
        auto [op, mask] = bind_step(hl::cap, hl::or_nil);
        MaybePredicate predicate{};
        if (!hl::is_nil(mask)) { predicate = ref_predicate(mask); }
        return builder.checked_op(ref_value(op), predicate);
    }
    case StepKind::ArithUnary: {
        auto [op, arg, overflow] = bind_step(hl::cap_as<ArithUnaryOp>, hl::cap, hl::or_nil_as<bool>);
        return builder.arith_unary(ref_value(arg), op, overflow);
    }
    case StepKind::Compare: {
        auto [op, l, r, is_unsigned] = bind_step(hl::cap_as<CmpOp>, hl::cap, hl::cap, hl::or_nil_as<bool>);
        return builder.cmp(ref_value(l), ref_value(r), op, is_unsigned);
    }
    case StepKind::IntCast: {
        auto [op, arg, overflow] = bind_step(hl::cap_as<IntCastKind>, hl::cap, hl::or_nil_as<bool>);
        return builder.int_cast(ref_value(arg), dtype, op, overflow);
    }
    case StepKind::FloatCast: {
        auto [arg, is_unsigned] = bind_step(hl::cap, hl::or_nil_as<bool>);
        return builder.float_cast(ref_value(arg), dtype, is_unsigned);
    }
    case StepKind::BitCast: {
        auto [arg] = bind_step(hl::cap);
        return builder.bitcast(ref_value(arg), dtype);
    }
    case StepKind::PredicateBinary: {
        auto [op, l, r] = bind_step(hl::cap_as<PredicateBinaryOp>, hl::cap, hl::cap);
        return builder.predicate_binary(ref_predicate(l), ref_predicate(r), op);
    }
    case StepKind::PredicateNot: {
        auto [arg] = bind_step(hl::cap);
        return builder.predicate_not(ref_predicate(arg));
    }
    case StepKind::Select: {
        auto [cond, truthy, falsy] = bind_step(hl::cap, hl::cap, hl::cap);
        Predicate c = ref_predicate(cond);
        DynamicValue t = ref(truthy);
        DynamicValue f = ref(falsy);
        auto t_value = t.as_value();
        auto f_value = f.as_value();
        if (t_value && f_value) { return builder.select(c, *t_value, *f_value); }
        auto t_predicate = t.as_predicate();
        auto f_predicate = f.as_predicate();
        if (t_predicate && f_predicate) { return builder.select(c, *t_predicate, *f_predicate); }
        serialization_error("Invalid select");
    }
    case StepKind::Index: {
        return builder.index(dtype);
    }
    case StepKind::Permute: {
        auto [arg, perm, bit] = bind_step(hl::cap, hl::cap, hl::cap_as<bool>);
        uint64_t permute_values = strtoull(perm.as_c_str(), nullptr, 16);
        return builder.permute(ref_value(arg), permute_values, bit);
    }
    case StepKind::Fpclass: {
        auto [arg, cl] = bind_step(hl::cap, hl::cap_as<FpClass>);
        return builder.fpclass(ref_value(arg), cl);
    }
    case StepKind::Load: {
        auto [func_arg, load_kind] = bind_step(hl::cap, hl::cap_as<LoadStoreKind>);
        if (dtype == ScalarDataType::I1) { return builder.load_predicate(parse_arg_ref(func_arg)); }
        return builder.load(parse_arg_ref(func_arg), load_kind);
    }
    case StepKind::Gather: {
        auto [idx, func_arg] = bind_step(hl::cap, hl::cap);
        return builder.gather(ref_value(idx), parse_arg_ref(func_arg));
    }
    case StepKind::LoadSplat: {
        auto [func_arg] = bind_step(hl::cap);
        if (dtype == ScalarDataType::I1) { return builder.load_predicate_splat(parse_arg_ref(func_arg)); }
        return builder.load_splat(parse_arg_ref(func_arg));
    }
    case StepKind::Store: {
        auto [arg, func_arg, store_kind, cond] = bind_step(hl::cap, hl::cap, hl::cap_as<LoadStoreKind>, hl::maybe);

        if (dtype == ScalarDataType::I1) {
            if (cond) {
                builder.cond_store(ref_predicate(arg), ref_predicate(cond.value()), parse_arg_ref(func_arg));
            } else {
                builder.store(ref_predicate(arg), parse_arg_ref(func_arg));
            }
        } else {
            if (cond) {
                builder.cond_store(ref_value(arg), ref_predicate(cond.value()), parse_arg_ref(func_arg), store_kind);
            } else {
                builder.store(ref_value(arg), parse_arg_ref(func_arg), store_kind);
            }
        }
        return {};
    }
    case StepKind::Pack: {
        auto [arg, cond, dst, acc] = bind_step(hl::cap, hl::cap, hl::cap, hl::cap);
        builder.pack(ref_value(arg), ref_predicate(cond), parse_arg_ref(dst), acc_arg_map.at(parse_acc_ref(acc)));
        return {};
    }
    case StepKind::Scatter: {
        auto [arg, idx, dst, cond] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        if (cond) {
            builder.cond_scatter(ref_value(arg), ref_value(idx), ref_predicate(cond.value()), parse_arg_ref(dst));
        } else {
            builder.scatter(ref_value(arg), ref_value(idx), parse_arg_ref(dst));
        }
        return {};
    }
    case StepKind::AccArithBinary:
    case StepKind::AccSum128: {
        auto [op, arg, acc, cond] = bind_step(hl::cap_as<ArithBinaryOp>, hl::cap, hl::cap, hl::maybe);
        if (cond) {
            builder.cond_arith_agg(ref_value(arg), ref_predicate(cond.value()), op, acc_arg_map.at(parse_acc_ref(acc)));
        } else {
            builder.arith_agg(ref_value(arg), op, acc_arg_map.at(parse_acc_ref(acc)));
        }
        return {};
    }
    case StepKind::Countif: {
        auto [arg, acc] = bind_step(hl::cap, hl::cap);
        if (dtype != ScalarDataType::I64) {
            serialization_error("Invalid countif dtype %s, expected i64", show_scalar_dtype(dtype));
        }
        builder.countif(ref_predicate(arg), acc_arg_map.at(parse_acc_ref(acc)));
        return {};
    }
    case StepKind::AccPredicateBinary: {
        auto [op, arg, acc] = bind_step(hl::cap_as<PredicateBinaryOp>, hl::cap, hl::cap);
        builder.predicate_agg(ref_predicate(arg), op, acc_arg_map.at(parse_acc_ref(acc)));
        return {};
    }
    }
#undef bind_step
#undef ref

    SIMJIT_UNREACHABLE();
}

static void process_deserialized_accs(hl::value accs, std::unordered_map<AccIdx, Argument> &acc_arg_map) {
    DebugContextGuard guard("accs");

    int idx = 0;
    for (auto acc : accs.iter()) {
        auto [arg, other] = hl::require_match("invalid acc definition", acc, "acc", idx, hl::_, hl::cap, hl::rest);
        acc_arg_map[idx] = parse_arg_ref(arg);
        idx++;
    }
}
static void process_deserialized_steps(hl::value steps, FunctionBuilder &builder,
                                       const std::unordered_map<AccIdx, Argument> &acc_arg_map) {
    DebugContextGuard guard("steps");

    int idx = 0;
    std::unordered_map<int, DynamicValue> step_map;
    for (auto step : steps.iter()) {
        auto [other] = hl::require_match("invalid step", step, "step", idx, hl::rest);
        auto v = process_deserialized_step(other, builder, step_map, acc_arg_map);
        step_map[idx] = v;

        idx++;
    }
}

static void process_deserialized(hl::value value, FunctionBuilder &builder) {
    GUARD_EQUAL(hl::car(value), "func");
    value = hl::cdr(value);

    hl::value args = hl::assoc_ref("args", value);
    process_deserialized_args(args, builder);
    hl::value accs = hl::assoc_ref("accs", value);
    std::unordered_map<AccIdx, Argument> acc_arg_map;
    if (accs) { process_deserialized_accs(accs, acc_arg_map); }
    hl::value steps = hl::assoc_ref("steps", value);
    process_deserialized_steps(steps, builder, acc_arg_map);
    if (auto scalar_only = hl::assoc("scalar-only", value)) { builder.scalar_only(); }
}

void deserialize(std::string_view str, FunctionBuilder &builder) {
    debug_context_stack.clear();

    DebugContextGuard debug_guard("deserialize");

    try {
        hl::context_guard guard{};
        hl::value deserialized = hl::read(str);
        process_deserialized(deserialized, builder);
    } catch (hl::hl_exception &e) { serialization_error("[%s] %s", debug_context_str().c_str(), e.what()); }
}

} // namespace simjit
