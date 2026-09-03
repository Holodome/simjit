// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "expr.h"

#include <limits>
#include <math.h>

#define messed_up(...) simjit_exception(ErrorModule::Generic, {}, {}, __VA_ARGS__)

namespace simjit {

const char *show_scalar_dtype(ScalarDataType dtype) noexcept {
    switch (dtype) {
    case ScalarDataType::I1: return "i1";
    case ScalarDataType::I8: return "i8";
    case ScalarDataType::I16: return "i16";
    case ScalarDataType::I32: return "i32";
    case ScalarDataType::I64: return "i64";
    case ScalarDataType::I128: return "i128";
    case ScalarDataType::F32: return "f32";
    case ScalarDataType::F64: return "f64";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_arith_binary_op(ArithBinaryOp op) noexcept {
    switch (op) {
    case ArithBinaryOp::Add: return "add";
    case ArithBinaryOp::Sub: return "sub";
    case ArithBinaryOp::Mul: return "mul";
    case ArithBinaryOp::Mul64SE: return "mulse";
    case ArithBinaryOp::Mul64ZE: return "mulze";
    case ArithBinaryOp::Min: return "min";
    case ArithBinaryOp::Max: return "max";
    case ArithBinaryOp::UMin: return "umin";
    case ArithBinaryOp::UMax: return "umax";
    case ArithBinaryOp::And: return "and";
    case ArithBinaryOp::Or: return "or";
    case ArithBinaryOp::Xor: return "xor";
    case ArithBinaryOp::AndNot: return "andnot";
    case ArithBinaryOp::ShiftRightArith: return "shra";
    case ArithBinaryOp::ShiftRightLogical: return "shrl";
    case ArithBinaryOp::ShiftLeftLogical: return "shll";
    case ArithBinaryOp::RotateLeft: return "rol";
    case ArithBinaryOp::RotateRight: return "ror";
    case ArithBinaryOp::Div: return "div";
    case ArithBinaryOp::UDiv: return "udiv";
    case ArithBinaryOp::Mod: return "mod";
    case ArithBinaryOp::UMod: return "umod";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_arith_unary_op(ArithUnaryOp op) noexcept {
    switch (op) {
    case ArithUnaryOp::Not: return "not";
    case ArithUnaryOp::Negate: return "neg";
    case ArithUnaryOp::Abs: return "abs";
    case ArithUnaryOp::Lzcnt: return "lzcnt";
    case ArithUnaryOp::Tzcnt: return "tzcnt";
    case ArithUnaryOp::Popcount: return "popcnt";
    case ArithUnaryOp::RoundNearest: return "round";
    case ArithUnaryOp::RoundDown: return "floor";
    case ArithUnaryOp::RoundUp: return "ceil";
    case ArithUnaryOp::RoundTruncate: return "trunc";
    case ArithUnaryOp::Rcp: return "rcp";
    case ArithUnaryOp::Sqrt: return "sqrt";
    case ArithUnaryOp::Rsqrt: return "rsqrt";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_vec_elem_type(VecElemType dtype) noexcept {
    switch (dtype) {
    case VecElemType::I8: return "i8";
    case VecElemType::I16: return "i16";
    case VecElemType::I32: return "i32";
    case VecElemType::I64: return "i64";
    case VecElemType::F32: return "f32";
    case VecElemType::F64: return "f64";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_vec_size(VecSize x) noexcept {
    switch (x) {
    case VecSize::X2: return "x2";
    case VecSize::X4: return "x4";
    case VecSize::X8: return "x8";
    case VecSize::X16: return "x16";
    case VecSize::X32: return "x32";
    case VecSize::X64: return "x64";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_mask_dtype(MaskDataType dtype) noexcept {
    switch (dtype) {
    case MaskDataType::M2: return "m2";
    case MaskDataType::M4: return "m4";
    case MaskDataType::M8: return "m8";
    case MaskDataType::M16: return "m16";
    case MaskDataType::M32: return "m32";
    case MaskDataType::M64: return "m64";
    }
    SIMJIT_UNREACHABLE();
}

#define ALL_SIZES(X) X(2), X(4), X(8), X(16), X(32), X(64)
#define MAKE_DEF0(_size, _dtype) (#_dtype "x" #_size)
#define MAKE_DEF(_size)                                                                                               \
    MAKE_DEF0(_size, i8), MAKE_DEF0(_size, i16), MAKE_DEF0(_size, i32), MAKE_DEF0(_size, i64), MAKE_DEF0(_size, f32), \
        MAKE_DEF0(_size, f64)

const char *show_vec_dtype(VecDataType dtype) noexcept {
    static const char *strs[] = {ALL_SIZES(MAKE_DEF)};
    return strs[(size_t)dtype.size * 6 + (size_t)dtype.elem];
}

#undef MAKE_DEF
#undef MAKE_DEF0
#undef ALL_SIZES

const char *show_dtype(DataType dtype) noexcept {
    switch (dtype.kind) {
    case DataTypeKind::Scalar: return show_scalar_dtype(dtype.scalar);
    case DataTypeKind::Vec: return show_vec_dtype(dtype.vec);
    case DataTypeKind::Mask: return show_mask_dtype(dtype.mask);
    }
    SIMJIT_UNREACHABLE();
}

const char *show_cmp_op(CmpOp op) noexcept {
    switch (op) {
    case CmpOp::Less: return "lt";
    case CmpOp::Greater: return "gt";
    case CmpOp::LessEqual: return "le";
    case CmpOp::GreaterEqual: return "ge";
    case CmpOp::Equal: return "eq";
    case CmpOp::NotEqual: return "ne";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_predicate_binary_op(PredicateBinaryOp op) noexcept {
    switch (op) {
    case PredicateBinaryOp::And: return "and";
    case PredicateBinaryOp::Or: return "or";
    case PredicateBinaryOp::AndNot: return "andnot";
    case PredicateBinaryOp::XNor: return "xnor";
    case PredicateBinaryOp::Xor: return "xor";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_int_cast_kind(IntCastKind kind) noexcept {
    switch (kind) {
    case IntCastKind::Trunc: return "trunc";
    case IntCastKind::Sext: return "sext";
    case IntCastKind::Zext: return "zext";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_load_store_kind(LoadStoreKind kind) noexcept {
    switch (kind) {
    case LoadStoreKind::Aligned: return "aligned";
    case LoadStoreKind::Unaligned: return "unaligned";
    }
    SIMJIT_UNREACHABLE();
}

ConstData scalar_dtype_max(ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I128: messed_up("invalid dtype for dtype_max");
    case ScalarDataType::I1: return ConstData::i1(true);
    case ScalarDataType::I8: return ConstData::i8(INT8_MAX);
    case ScalarDataType::I16: return ConstData::i16(INT16_MAX);
    case ScalarDataType::I32: return ConstData::i32(INT32_MAX);
    case ScalarDataType::I64: return ConstData::i64(INT64_MAX);
    case ScalarDataType::F32: return ConstData::f32(HUGE_VALF);
    case ScalarDataType::F64: return ConstData::f64(HUGE_VAL);
    }
    SIMJIT_UNREACHABLE();
}

uint64_t scalar_dtype_umax(ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I1: return 1;
    case ScalarDataType::I8: return (int64_t)(uint64_t)UINT8_MAX;
    case ScalarDataType::I16: return (int64_t)(uint64_t)UINT16_MAX;
    case ScalarDataType::I32: return (int64_t)(uint64_t)UINT32_MAX;
    case ScalarDataType::I64: return (int64_t)(uint64_t)UINT64_MAX;
    case ScalarDataType::I128:
    case ScalarDataType::F32:
    case ScalarDataType::F64: messed_up("invalid dtype %s for dtype_max", show_scalar_dtype(dt));
    }
    SIMJIT_UNREACHABLE();
}

ConstData scalar_dtype_min(ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I128: messed_up("invalid dtype %s for dtype_max", show_scalar_dtype(dt));
    case ScalarDataType::I1: return ConstData::i1(false);
    case ScalarDataType::I8: return ConstData::i8(INT8_MIN);
    case ScalarDataType::I16: return ConstData::i16(INT16_MIN);
    case ScalarDataType::I32: return ConstData::i32(INT32_MIN);
    case ScalarDataType::I64: return ConstData::i64(INT64_MIN);
    case ScalarDataType::F32: return ConstData::f32(-HUGE_VALF);
    case ScalarDataType::F64: return ConstData::f64(-HUGE_VAL);
    }
    SIMJIT_UNREACHABLE();
}

std::string show_argument_kind(ArgumentKind kind) {
    if (kind == ArgumentKind::Undefined) { return "undefined"; }
    if (kind == ArgumentKind::Table) { return "table"; }

    std::string result{};
    auto append = [&](std::string_view v) {
        if (!result.empty()) { result += "|"; }
        result += v;
    };

    if (bool(kind & ArgumentKind::SrcArr)) append("src-arr");
    if (bool(kind & ArgumentKind::SrcGatherArr)) append("src-gather-arr");
    if (bool(kind & ArgumentKind::SrcIdxArr)) append("src-idx-arr");
    if (bool(kind & ArgumentKind::SrcConst)) append("src-const");
    if (bool(kind & ArgumentKind::Dst)) append("dst-arr");
    if (bool(kind & ArgumentKind::DstAgg)) append("dst-scalar");
    if (bool(kind & ArgumentKind::DstSafetyCheck)) append("dst-safety-check");

    return result.empty() ? "unknown" : result;
}

std::string show_fpclass(FpClass flags) {
    std::string result{};
    auto append = [&](std::string_view v) {
        if (!result.empty()) { result += "|"; };
        result += v;
    };

    if (bool(flags & FpClass::FPC_INFINITE)) append("inf");
    if (bool(flags & FpClass::FPC_NAN)) append("nan");
    if (bool(flags & FpClass::FPC_SUBNORMAL)) append("subnormal");
    if (bool(flags & FpClass::FPC_ZERO)) append("zero");

    if (result.empty()) result = "no";

    return result;
}

std::string show_arith_binary_flags(ArithBinaryOpFlags flags) {
    std::string result{};
    auto append = [&](std::string_view v) {
        if (!result.empty()) { result += "|"; };
        result += v;
    };
    if (bool(flags & ArithBinaryOpFlags::SafetyCheck)) append("safety-check");
    if (bool(flags & ArithBinaryOpFlags::ShiftWraparound)) append("shift-wraparound");
    if (bool(flags & ArithBinaryOpFlags::SafeDivision)) append("safe-division");

    if (result.empty()) result = "no";
    return result;
}

const char *show_error_module(ErrorModule module) noexcept {
    switch (module) {
    case ErrorModule::Generic: return "Generic";
    case ErrorModule::HIR: return "HIR";
    case ErrorModule::Vectorizer: return "Vectorizer";
    case ErrorModule::MIR: return "MIR";
    case ErrorModule::LLVM: return "LLVM";
    case ErrorModule::CPP: return "CPP";
    case ErrorModule::AsmJit: return "AsmJit";
    case ErrorModule::X86: return "X86";
    case ErrorModule::A64: return "A64";
    case ErrorModule::Serialization: return "Serialization";
    case ErrorModule::JIT: return "JIT";
    case ErrorModule::Nullable: return "Nullable";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_error_kind(ErrorKind kind) noexcept {
    switch (kind) {
    case ErrorKind::InvalidInput: return "InvalidInput";
    case ErrorKind::Unsupported: return "Unsupported";
    case ErrorKind::VectorizationFailed: return "VectorizationFailed";
    case ErrorKind::InternalInvariant: return "InternalInvariant";
    case ErrorKind::SerializationFailure: return "SerializationFailure";
    case ErrorKind::JitFailure: return "JitFailure";
    }
    SIMJIT_UNREACHABLE();
}

const char *show_error_subkind(ErrorSubKind subkind) noexcept {
    switch (subkind) {
    case ErrorSubKind::None: return "None";
    case ErrorSubKind::TypeError: return "TypeError";
    case ErrorSubKind::LimitExceeded: return "LimitExceeded";
    case ErrorSubKind::InvalidConfiguration: return "InvalidConfiguration";
    case ErrorSubKind::InvalidArgumentAccess: return "InvalidArgumentAccess";
    case ErrorSubKind::MissingRequiredOutput: return "MissingRequiredOutput";
    case ErrorSubKind::ArgumentMismatch: return "ArgumentMismatch";
    case ErrorSubKind::CacheLimitExceeded: return "CacheLimitExceeded";
    case ErrorSubKind::UnsupportedFeature: return "UnsupportedFeature";
    case ErrorSubKind::UnsupportedBackendFeature: return "UnsupportedBackendFeature";
    case ErrorSubKind::UnsupportedHostFeature: return "UnsupportedHostFeature";
    case ErrorSubKind::SerializationParseError: return "SerializationParseError";
    case ErrorSubKind::SerializationFormatError: return "SerializationFormatError";
    case ErrorSubKind::ExternalFailure: return "ExternalFailure";
    case ErrorSubKind::UnsupportedSpecialOps: return "UnsupportedSpecialOps";
    case ErrorSubKind::TooManyRoots: return "TooManyRoots";
    case ErrorSubKind::UnresolvedCarrierDType: return "UnresolvedCarrierDType";
    case ErrorSubKind::ConflictingGraphCoefficient: return "ConflictingGraphCoefficient";
    case ErrorSubKind::GraphCoefficientLimitExceeded: return "GraphCoefficientLimitExceeded";
    case ErrorSubKind::CoefficientRangeNeedsNormalization: return "CoefficientRangeNeedsNormalization";
    case ErrorSubKind::UpcastSubjectUnreachable: return "UpcastSubjectUnreachable";
    case ErrorSubKind::UpcastDidNotConverge: return "UpcastDidNotConverge";
    case ErrorSubKind::ComponentRangeSubjectUnreachable: return "ComponentRangeSubjectUnreachable";
    case ErrorSubKind::ComponentRangeDidNotConverge: return "ComponentRangeDidNotConverge";
    case ErrorSubKind::MaskCombineTooWide: return "MaskCombineTooWide";
    case ErrorSubKind::DowncastCombineItemWidthMismatch: return "DowncastCombineItemWidthMismatch";
    case ErrorSubKind::FloatDowncastCombineItemWidthMismatch: return "FloatDowncastCombineItemWidthMismatch";
    case ErrorSubKind::SyntheticIntCastItemWidthMismatch: return "SyntheticIntCastItemWidthMismatch";
    case ErrorSubKind::UpcastHalfItemWidthMismatch: return "UpcastHalfItemWidthMismatch";
    case ErrorSubKind::RootWidthsNotPowerOfTwo: return "RootWidthsNotPowerOfTwo";
    case ErrorSubKind::RootWidthsNotDivisible: return "RootWidthsNotDivisible";
    case ErrorSubKind::RootWidthsMismatch: return "RootWidthsMismatch";
    case ErrorSubKind::MaskDTypeTooWide: return "MaskDTypeTooWide";
    case ErrorSubKind::WidthMismatch: return "WidthMismatch";
    }
    SIMJIT_UNREACHABLE();
}

} // namespace simjit
