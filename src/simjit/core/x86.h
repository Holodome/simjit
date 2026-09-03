// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/core/expr.h"

namespace simjit {
namespace x86 {

#define x86_messed_up(...) simjit_exception(ErrorModule::X86, {}, {}, __VA_ARGS__)

enum class VecRegisterKind : uint8_t {
    XMM,
    YMM,
    ZMM
};

struct Vector {
    VecRegisterKind reg;
    VecElemType dtype;
};

constexpr Vector vec_to_x86(VecDataType dtype) {
    switch (dtype.elem) {
    case VecElemType::I8:
        switch (dtype.size) {
        case VecSize::X16: return Vector{VecRegisterKind::XMM, dtype.elem};
        case VecSize::X32: return Vector{VecRegisterKind::YMM, dtype.elem};
        case VecSize::X64: return Vector{VecRegisterKind::ZMM, dtype.elem};
        default: break;
        }
        break;
    case VecElemType::I16:
        switch (dtype.size) {
        case VecSize::X8: return Vector{VecRegisterKind::XMM, dtype.elem};
        case VecSize::X16: return Vector{VecRegisterKind::YMM, dtype.elem};
        case VecSize::X32: return Vector{VecRegisterKind::ZMM, dtype.elem};
        default: break;
        }
        break;
    case VecElemType::I32:
    case VecElemType::F32:
        switch (dtype.size) {
        case VecSize::X4: return Vector{VecRegisterKind::XMM, dtype.elem};
        case VecSize::X8: return Vector{VecRegisterKind::YMM, dtype.elem};
        case VecSize::X16: return Vector{VecRegisterKind::ZMM, dtype.elem};
        default: break;
        }
        break;
    case VecElemType::I64:
    case VecElemType::F64:
        switch (dtype.size) {
        case VecSize::X2: return Vector{VecRegisterKind::XMM, dtype.elem};
        case VecSize::X4: return Vector{VecRegisterKind::YMM, dtype.elem};
        case VecSize::X8: return Vector{VecRegisterKind::ZMM, dtype.elem};
        default: break;
        }
        break;
    }
    x86_messed_up("Invalid data type vector_to_x86 %s", show_vec_dtype(dtype));
}

constexpr VecDataType x86_to_vec(x86::Vector vec) {
    VecElemType dtype = vec.dtype;
    switch (vec.reg) {
    case VecRegisterKind::XMM:
        switch (dtype) {
        case VecElemType::I8: return VecDataType{VecSize::X16, dtype};
        case VecElemType::I16: return VecDataType{VecSize::X8, dtype};
        case VecElemType::I32:
        case VecElemType::F32: return VecDataType{VecSize::X4, dtype};
        case VecElemType::I64:
        case VecElemType::F64: return VecDataType{VecSize::X2, dtype};
        }
        SIMJIT_UNREACHABLE();
    case VecRegisterKind::YMM:
        switch (dtype) {
        case VecElemType::I8: return VecDataType{VecSize::X32, dtype};
        case VecElemType::I16: return VecDataType{VecSize::X16, dtype};
        case VecElemType::I32:
        case VecElemType::F32: return VecDataType{VecSize::X8, dtype};
        case VecElemType::I64:
        case VecElemType::F64: return VecDataType{VecSize::X4, dtype};
        }
        SIMJIT_UNREACHABLE();
    case VecRegisterKind::ZMM:
        switch (dtype) {
        case VecElemType::I8: return VecDataType{VecSize::X64, dtype};
        case VecElemType::I16: return VecDataType{VecSize::X32, dtype};
        case VecElemType::I32:
        case VecElemType::F32: return VecDataType{VecSize::X16, dtype};
        case VecElemType::I64:
        case VecElemType::F64: return VecDataType{VecSize::X8, dtype};
        }
        SIMJIT_UNREACHABLE();
    }
    SIMJIT_UNREACHABLE();
}

namespace types {
constexpr Vector XMMI8V = Vector{VecRegisterKind::XMM, VecElemType::I8};
constexpr Vector XMMI16V = Vector{VecRegisterKind::XMM, VecElemType::I16};
constexpr Vector XMMI32V = Vector{VecRegisterKind::XMM, VecElemType::I32};
constexpr Vector XMMI64V = Vector{VecRegisterKind::XMM, VecElemType::I64};
constexpr Vector XMMF32V = Vector{VecRegisterKind::XMM, VecElemType::F32};
constexpr Vector XMMF64V = Vector{VecRegisterKind::XMM, VecElemType::F64};

constexpr Vector YMMI8V = Vector{VecRegisterKind::YMM, VecElemType::I8};
constexpr Vector YMMI16V = Vector{VecRegisterKind::YMM, VecElemType::I16};
constexpr Vector YMMI32V = Vector{VecRegisterKind::YMM, VecElemType::I32};
constexpr Vector YMMI64V = Vector{VecRegisterKind::YMM, VecElemType::I64};
constexpr Vector YMMF32V = Vector{VecRegisterKind::YMM, VecElemType::F32};
constexpr Vector YMMF64V = Vector{VecRegisterKind::YMM, VecElemType::F64};

constexpr Vector ZMMI8V = Vector{VecRegisterKind ::ZMM, VecElemType::I8};
constexpr Vector ZMMI16V = Vector{VecRegisterKind::ZMM, VecElemType::I16};
constexpr Vector ZMMI32V = Vector{VecRegisterKind::ZMM, VecElemType::I32};
constexpr Vector ZMMI64V = Vector{VecRegisterKind::ZMM, VecElemType::I64};
constexpr Vector ZMMF32V = Vector{VecRegisterKind::ZMM, VecElemType::F32};
constexpr Vector ZMMF64V = Vector{VecRegisterKind::ZMM, VecElemType::F64};

constexpr VecDataType XMMI8 = x86_to_vec(XMMI8V);
constexpr VecDataType XMMI16 = x86_to_vec(XMMI16V);
constexpr VecDataType XMMI32 = x86_to_vec(XMMI32V);
constexpr VecDataType XMMI64 = x86_to_vec(XMMI64V);
constexpr VecDataType XMMF32 = x86_to_vec(XMMF32V);
constexpr VecDataType XMMF64 = x86_to_vec(XMMF64V);

constexpr VecDataType YMMI8 = x86_to_vec(YMMI8V);
constexpr VecDataType YMMI16 = x86_to_vec(YMMI16V);
constexpr VecDataType YMMI32 = x86_to_vec(YMMI32V);
constexpr VecDataType YMMI64 = x86_to_vec(YMMI64V);
constexpr VecDataType YMMF32 = x86_to_vec(YMMF32V);
constexpr VecDataType YMMF64 = x86_to_vec(YMMF64V);

constexpr VecDataType ZMMI8 = x86_to_vec(ZMMI8V);
constexpr VecDataType ZMMI16 = x86_to_vec(ZMMI16V);
constexpr VecDataType ZMMI32 = x86_to_vec(ZMMI32V);
constexpr VecDataType ZMMI64 = x86_to_vec(ZMMI64V);
constexpr VecDataType ZMMF32 = x86_to_vec(ZMMF32V);
constexpr VecDataType ZMMF64 = x86_to_vec(ZMMF64V);
} // namespace types

} // namespace x86
} // namespace simjit
