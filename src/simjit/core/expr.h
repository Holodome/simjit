// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/detail/base.h"

namespace simjit {

enum class VecElemType : uint8_t {
    // Use [0, 5] to ease up conversion to ScalarDataType
    I8 = 0,
    I16 = 1,
    I32 = 2,
    I64 = 3,
    F32 = 4,
    F64 = 5
};

enum class VecSize : uint8_t {
    X2 = 0,
    X4 = 1,
    X8 = 2,
    X16 = 3,
    X32 = 4,
    X64 = 5
};

enum class MaskDataType : uint8_t {
    M2 = 0,
    M4 = 1,
    M8 = 2,
    M16 = 3,
    M32 = 4,
    M64 = 5
};

constexpr bool vec_elem_is_float(VecElemType elem) noexcept {
    return elem == VecElemType::F32 || elem == VecElemType::F64;
}
constexpr bool vec_elem_is_int(VecElemType elem) noexcept {
    return elem == VecElemType::I8 || elem == VecElemType::I16 || elem == VecElemType::I32 || elem == VecElemType::I64;
}

constexpr bool is_simple_int_dtype(ScalarDataType dtype) noexcept {
    return dtype == ScalarDataType::I8 || dtype == ScalarDataType::I16 || dtype == ScalarDataType::I32 ||
           dtype == ScalarDataType::I64;
}

constexpr bool is_float_dtype(ScalarDataType dtype) noexcept {
    return dtype == ScalarDataType::F32 || dtype == ScalarDataType::F64;
}

constexpr size_t scalar_dtype_size(ScalarDataType dtype) noexcept {
    switch (dtype) {
    case ScalarDataType::I8: return 1;
    case ScalarDataType::I16: return 2;
    case ScalarDataType::I32: return 4;
    case ScalarDataType::I64: return 8;
    case ScalarDataType::I1: return 1;
    case ScalarDataType::I128: return 16;
    case ScalarDataType::F32: return 4;
    case ScalarDataType::F64: return 8;
    }
    SIMJIT_UNREACHABLE();
}

constexpr size_t scalar_dtype_bits(ScalarDataType sdtype) noexcept {
    if (sdtype == ScalarDataType::I1) return 1;
    return scalar_dtype_size(sdtype) * 8;
}

constexpr size_t scalar_dtype_size_log2(ScalarDataType dtype) noexcept {
    switch (dtype) {
    case ScalarDataType::I8: return 0;
    case ScalarDataType::I16: return 1;
    case ScalarDataType::I32: return 2;
    case ScalarDataType::I64: return 3;
    case ScalarDataType::I1: return 0;
    case ScalarDataType::I128: return 4;
    case ScalarDataType::F32: return 2;
    case ScalarDataType::F64: return 3;
    }
    SIMJIT_UNREACHABLE();
}

constexpr size_t vec_elem_size_bytes(VecElemType dtype) noexcept {
    switch (dtype) {
    case VecElemType::I8: return 1;
    case VecElemType::I16: return 2;
    case VecElemType::I32: return 4;
    case VecElemType::I64: return 8;
    case VecElemType::F32: return 4;
    case VecElemType::F64: return 8;
    }
    SIMJIT_UNREACHABLE();
}

constexpr size_t vec_elem_size_bytes_log2(VecElemType dtype) noexcept {
    switch (dtype) {
    case VecElemType::I8: return 0;
    case VecElemType::I16: return 1;
    case VecElemType::I32: return 2;
    case VecElemType::I64: return 3;
    case VecElemType::F32: return 2;
    case VecElemType::F64: return 3;
    }
    SIMJIT_UNREACHABLE();
}

constexpr ScalarDataType vec_elem_to_scalar(VecElemType dtype) noexcept {
    // 1-1 map
    return (ScalarDataType)(int)dtype;
}

constexpr std::optional<VecElemType> vec_elem_from_scalar(ScalarDataType dtype) noexcept {
    switch (dtype) {
    case ScalarDataType::I8: return VecElemType::I8;
    case ScalarDataType::I16: return VecElemType::I16;
    case ScalarDataType::I32: return VecElemType::I32;
    case ScalarDataType::I64: return VecElemType::I64;
    case ScalarDataType::F32: return VecElemType::F32;
    case ScalarDataType::F64: return VecElemType::F64;
    case ScalarDataType::I1:
    case ScalarDataType::I128: break;
    }
    return {};
}

constexpr size_t vec_size_num(VecSize kind) noexcept {
    SIMJIT_ASSERT(kind <= VecSize::X64);
    return (size_t)2 << (size_t)kind;
}

static_assert(vec_size_num(VecSize::X2) == 2);
static_assert(vec_size_num(VecSize::X4) == 4);
static_assert(vec_size_num(VecSize::X8) == 8);
static_assert(vec_size_num(VecSize::X16) == 16);
static_assert(vec_size_num(VecSize::X32) == 32);
static_assert(vec_size_num(VecSize::X64) == 64);

constexpr size_t mask_dtype_bits(MaskDataType mask) noexcept {
    return (size_t)2 << (size_t)mask;
}

constexpr size_t mask_dtype_bits_log2(MaskDataType mask) noexcept {
    return (size_t)mask + 1;
}

constexpr std::optional<MaskDataType> double_mask(MaskDataType mask) noexcept {
    switch (mask) {
    case MaskDataType::M2: return MaskDataType::M4;
    case MaskDataType::M4: return MaskDataType::M8;
    case MaskDataType::M8: return MaskDataType::M16;
    case MaskDataType::M16: return MaskDataType::M32;
    case MaskDataType::M32: return MaskDataType::M64;
    case MaskDataType::M64: return {};
    }
    SIMJIT_UNREACHABLE();
}

constexpr std::optional<MaskDataType> half_mask(MaskDataType mask) noexcept {
    switch (mask) {
    case MaskDataType::M2: return {};
    case MaskDataType::M4: return MaskDataType::M2;
    case MaskDataType::M8: return MaskDataType::M4;
    case MaskDataType::M16: return MaskDataType::M8;
    case MaskDataType::M32: return MaskDataType::M16;
    case MaskDataType::M64: return MaskDataType::M32;
    }
    SIMJIT_UNREACHABLE();
}

constexpr MaskDataType vec_mask_dtype(VecSize size) noexcept {
    // 1-1 map
    return (MaskDataType)(int)size;
}

struct VecDataType {
    VecSize size;
    VecElemType elem;

    constexpr bool operator==(VecDataType right) const noexcept { return size == right.size && elem == right.elem; }
    constexpr bool operator!=(VecDataType right) const noexcept { return size != right.size || elem != right.elem; }

    constexpr ScalarDataType to_scalar() const noexcept { return vec_elem_to_scalar(elem); }
    constexpr size_t element_size_bytes() const noexcept { return vec_elem_size_bytes(elem); }
    constexpr size_t element_size_bits() const noexcept { return element_size_bytes() * 8; }
    constexpr size_t element_size_bytes_log2() const noexcept { return vec_elem_size_bytes_log2(elem); }
    constexpr size_t nelems() const noexcept { return vec_size_num(size); }
    constexpr size_t size_bytes() const noexcept { return element_size_bytes() * nelems(); }
    constexpr MaskDataType mask() const noexcept { return vec_mask_dtype(size); }
    constexpr bool is_float() const noexcept { return vec_elem_is_float(elem); }
    constexpr bool is_int() const noexcept { return vec_elem_is_int(elem); }
};

// Use std::optional to force error handling
constexpr std::optional<VecDataType> vec_dtype_half(VecDataType dtype) noexcept {
    switch (dtype.size) {
    case VecSize::X2: return {};
    case VecSize::X4: return VecDataType{VecSize::X2, dtype.elem};
    case VecSize::X8: return VecDataType{VecSize::X4, dtype.elem};
    case VecSize::X16: return VecDataType{VecSize::X8, dtype.elem};
    case VecSize::X32: return VecDataType{VecSize::X16, dtype.elem};
    case VecSize::X64: return VecDataType{VecSize::X32, dtype.elem};
    }
    SIMJIT_UNREACHABLE();
}

constexpr std::optional<VecDataType> vec_dtype_twice(VecDataType dtype) noexcept {
    switch (dtype.size) {
    case VecSize::X2: return VecDataType{VecSize::X4, dtype.elem};
    case VecSize::X4: return VecDataType{VecSize::X8, dtype.elem};
    case VecSize::X8: return VecDataType{VecSize::X16, dtype.elem};
    case VecSize::X16: return VecDataType{VecSize::X32, dtype.elem};
    case VecSize::X32: return VecDataType{VecSize::X64, dtype.elem};
    case VecSize::X64: return {};
    }
    SIMJIT_UNREACHABLE();
}

constexpr ScalarDataType mask_dtype_to_scalar(MaskDataType dtype) noexcept {
    switch (dtype) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: return ScalarDataType::I8;
    case MaskDataType::M16: return ScalarDataType::I16;
    case MaskDataType::M32: return ScalarDataType::I32;
    case MaskDataType::M64: return ScalarDataType::I64;
    }
    SIMJIT_UNREACHABLE();
}

constexpr std::optional<ArithBinaryOp> arith_op_from_predicate(PredicateBinaryOp op) noexcept {
    switch (op) {
    case PredicateBinaryOp::And: return ArithBinaryOp::And;
    case PredicateBinaryOp::Or: return ArithBinaryOp::Or;
    case PredicateBinaryOp::Xor: return ArithBinaryOp::Xor;
    case PredicateBinaryOp::AndNot: return ArithBinaryOp::AndNot;
    case PredicateBinaryOp::XNor: return std::nullopt;
    }
    SIMJIT_UNREACHABLE();
}

enum class DataTypeKind : uint8_t {
    Scalar,
    Vec,
    Mask,
};

struct DataType {
    DataTypeKind kind;
    union {
        ScalarDataType scalar;
        VecDataType vec;
        MaskDataType mask;
    };

    constexpr bool is_scalar() const noexcept { return kind == DataTypeKind::Scalar; }
    constexpr bool is_vec() const noexcept { return kind == DataTypeKind::Vec; }
    constexpr bool is_mask() const noexcept { return kind == DataTypeKind::Mask; }

    // Zero-init to a deterministic scalar type.
    constexpr explicit DataType() noexcept : kind(DataTypeKind::Scalar), scalar(ScalarDataType::I8) {}
    constexpr DataType(ScalarDataType dtype) noexcept : kind(DataTypeKind::Scalar), scalar(dtype) {}
    constexpr DataType(VecDataType dtype) noexcept : kind(DataTypeKind::Vec), vec(dtype) {}
    constexpr DataType(MaskDataType dtype) noexcept : kind(DataTypeKind::Mask), mask(dtype) {}

    ScalarDataType as_scalar() const noexcept {
        SIMJIT_ASSERT(is_scalar());
        return scalar;
    }

    VecDataType as_vec() const noexcept {
        SIMJIT_ASSERT(is_vec());
        return vec;
    }

    MaskDataType as_mask() const noexcept {
        SIMJIT_ASSERT(is_mask());
        return mask;
    }

    constexpr bool operator==(DataType right) const noexcept {
        if (kind != right.kind) return false;
        switch (kind) {
        case DataTypeKind::Scalar: return scalar == right.scalar;
        case DataTypeKind::Vec: return vec == right.vec;
        case DataTypeKind::Mask: return mask == right.mask;
        }
        SIMJIT_UNREACHABLE();
    }
    constexpr bool operator!=(DataType right) const noexcept { return !(*this == right); }
};

struct ArgumentDecl {
    ScalarDataType dtype;
    ArgumentIdx idx;
    ArgumentKind kind;
};

ConstData scalar_dtype_max(ScalarDataType dt);
ConstData scalar_dtype_min(ScalarDataType dt);
uint64_t scalar_dtype_umax(ScalarDataType dt);

const char *show_scalar_dtype(ScalarDataType dtype) noexcept;
const char *show_mask_dtype(MaskDataType dtype) noexcept;
const char *show_vec_dtype(VecDataType dtype) noexcept;
const char *show_vec_elem_type(VecElemType dtype) noexcept;
const char *show_vec_size(VecSize dtype) noexcept;
const char *show_dtype(DataType dtype) noexcept;
const char *show_arith_binary_op(ArithBinaryOp op) noexcept;
const char *show_arith_unary_op(ArithUnaryOp op) noexcept;
const char *show_cmp_op(CmpOp op) noexcept;
const char *show_predicate_binary_op(PredicateBinaryOp op) noexcept;
const char *show_arith_agg_kind(ArithBinaryOp op) noexcept;
const char *show_predicate_agg_kind(PredicateBinaryOp op) noexcept;
const char *show_int_cast_kind(IntCastKind kind) noexcept;
const char *show_load_store_kind(LoadStoreKind kind) noexcept;
std::string show_argument_kind(ArgumentKind kind);
std::string show_fpclass(FpClass flags);
std::string show_arith_binary_flags(ArithBinaryOpFlags flags);

const char *show_error_module(ErrorModule module) noexcept;
const char *show_error_kind(ErrorKind kind) noexcept;
const char *show_error_subkind(ErrorSubKind subkind) noexcept;

inline std::string show_const_data(ConstData data, ScalarDataType dt) {
    if (dt == ScalarDataType::F32) { return simjit::format("%a", data.as_f32()); }
    if (dt == ScalarDataType::F64) { return simjit::format("%a", data.as_f64()); }
    return std::to_string(data.as_signed());
}

} // namespace simjit
