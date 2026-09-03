// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/cpp/x86_intrin.h"
#include "simjit/core/expr.h"
#include "simjit/core/mir.h"

// NOLINTBEGIN(bugprone-throwing-static-initialization)

namespace simjit {
namespace x86 {

using namespace ::simjit::x86::types;
using namespace ::simjit::types;

constexpr auto M2 = MaskDataType::M2;
constexpr auto M4 = MaskDataType::M4;
constexpr auto M8 = MaskDataType::M8;
constexpr auto M16 = MaskDataType::M16;
constexpr auto M32 = MaskDataType::M32;
constexpr auto M64 = MaskDataType::M64;

// NOLINTNEXTLINE(bugprone-suspicious-include): generated implementation is intentionally included in this namespace.
#include "x86_intrin.generated.cpp"

constexpr auto CT = CurrentType;
constexpr auto CMT = CurrentMaskType;

const UnaryIntrinsicMap<VecDataType> &arith_binary_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return add_map;
    case ArithBinaryOp::Sub: return sub_map;
    case ArithBinaryOp::Mul: return mullo_map;
    case ArithBinaryOp::Mul64ZE: return umul_map;
    case ArithBinaryOp::Mul64SE: return mul_map;
    case ArithBinaryOp::Min: return min_map;
    case ArithBinaryOp::Max: return max_map;
    case ArithBinaryOp::UMin: return umin_map;
    case ArithBinaryOp::UMax: return umax_map;
    case ArithBinaryOp::And: return and_map;
    case ArithBinaryOp::Or: return or_map;
    case ArithBinaryOp::Xor: return xor_map;
    case ArithBinaryOp::AndNot: return andnot_map;
    case ArithBinaryOp::ShiftRightArith: return srav_map;
    case ArithBinaryOp::ShiftRightLogical: return srlv_map;
    case ArithBinaryOp::ShiftLeftLogical: return sllv_map;
    case ArithBinaryOp::RotateLeft: return rolv_map;
    case ArithBinaryOp::RotateRight: return rorv_map;
    default: x86_messed_up("missing intrinsic for int %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &vector_immediate_shift_rotate_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::ShiftRightArith: return srai_map;
    case ArithBinaryOp::ShiftRightLogical: return srli_map;
    case ArithBinaryOp::ShiftLeftLogical: return slli_map;
    case ArithBinaryOp::RotateLeft: return roli_map;
    case ArithBinaryOp::RotateRight: return rori_map;
    default: x86_messed_up("missing immediate intrinsic for %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &maskz_vector_immediate_shift_rotate_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::ShiftRightArith: return maskz_srai_map;
    case ArithBinaryOp::ShiftRightLogical: return maskz_srli_map;
    case ArithBinaryOp::ShiftLeftLogical: return maskz_slli_map;
    case ArithBinaryOp::RotateLeft: return maskz_roli_map;
    case ArithBinaryOp::RotateRight: return maskz_rori_map;
    default: x86_messed_up("missing maskz immediate intrinsic for %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &mask_vector_immediate_shift_rotate_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::ShiftRightArith: return mask_srai_map;
    case ArithBinaryOp::ShiftRightLogical: return mask_srli_map;
    case ArithBinaryOp::ShiftLeftLogical: return mask_slli_map;
    case ArithBinaryOp::RotateLeft: return mask_roli_map;
    case ArithBinaryOp::RotateRight: return mask_rori_map;
    default: x86_messed_up("missing mask immediate intrinsic for %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &float_binary_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return add_map;
    case ArithBinaryOp::Sub: return sub_map;
    case ArithBinaryOp::Mul: return float_mul_map;
    case ArithBinaryOp::Min: return min_map;
    case ArithBinaryOp::Max: return max_map;
    case ArithBinaryOp::And: return and_map;
    case ArithBinaryOp::Or: return or_map;
    case ArithBinaryOp::Xor: return xor_map;
    case ArithBinaryOp::AndNot: return andnot_map;
    case ArithBinaryOp::Div: return div_map;
    default: x86_messed_up("missing intrinsic for float %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &maskz_arith_binary_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return maskz_add_map;
    case ArithBinaryOp::Sub: return maskz_sub_map;
    case ArithBinaryOp::Mul: return maskz_mullo_map;
    case ArithBinaryOp::Min: return maskz_min_map;
    case ArithBinaryOp::Max: return maskz_max_map;
    case ArithBinaryOp::UMin: return maskz_umin_map;
    case ArithBinaryOp::UMax: return maskz_umax_map;
    case ArithBinaryOp::And: return maskz_and_map;
    case ArithBinaryOp::Or: return maskz_or_map;
    case ArithBinaryOp::Xor: return maskz_xor_map;
    case ArithBinaryOp::AndNot: return maskz_andnot_map;
    case ArithBinaryOp::ShiftRightArith: return maskz_srav_map;
    case ArithBinaryOp::ShiftRightLogical: return maskz_srlv_map;
    case ArithBinaryOp::ShiftLeftLogical: return maskz_sllv_map;
    case ArithBinaryOp::RotateLeft: return maskz_rolv_map;
    case ArithBinaryOp::RotateRight: return maskz_rorv_map;
    default: x86_messed_up("missing maskz intrinsic for int %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &maskz_float_binary_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return maskz_add_map;
    case ArithBinaryOp::Sub: return maskz_sub_map;
    case ArithBinaryOp::Mul: return maskz_float_mul_map;
    case ArithBinaryOp::Min: return maskz_min_map;
    case ArithBinaryOp::Max: return maskz_max_map;
    case ArithBinaryOp::And: return maskz_and_map;
    case ArithBinaryOp::Or: return maskz_or_map;
    case ArithBinaryOp::Xor: return maskz_xor_map;
    case ArithBinaryOp::AndNot: return maskz_andnot_map;
    case ArithBinaryOp::Div: return maskz_float_div_map;
    default: x86_messed_up("missing maskz intrinsic for float %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &mask_float_binary_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return mask_add_map;
    case ArithBinaryOp::Sub: return mask_sub_map;
    case ArithBinaryOp::Mul: return mask_float_mul_map;
    case ArithBinaryOp::Min: return mask_min_map;
    case ArithBinaryOp::Max: return mask_max_map;
    case ArithBinaryOp::And: return mask_and_map;
    case ArithBinaryOp::Or: return mask_or_map;
    case ArithBinaryOp::Xor: return mask_xor_map;
    case ArithBinaryOp::AndNot: return mask_andnot_map;
    case ArithBinaryOp::Div: return mask_float_div_map;
    default: x86_messed_up("missing mask intrinsic for float %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const UnaryIntrinsicMap<VecDataType> &mask_arith_binary_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return mask_add_map;
    case ArithBinaryOp::Sub: return mask_sub_map;
    case ArithBinaryOp::Mul: return mask_mullo_map;
    case ArithBinaryOp::Min: return mask_min_map;
    case ArithBinaryOp::Max: return mask_max_map;
    case ArithBinaryOp::UMin: return mask_umin_map;
    case ArithBinaryOp::UMax: return mask_umax_map;
    case ArithBinaryOp::And: return mask_and_map;
    case ArithBinaryOp::Or: return mask_or_map;
    case ArithBinaryOp::Xor: return mask_xor_map;
    case ArithBinaryOp::AndNot: return mask_andnot_map;
    case ArithBinaryOp::ShiftRightArith: return mask_srav_map;
    case ArithBinaryOp::ShiftRightLogical: return mask_srlv_map;
    case ArithBinaryOp::ShiftLeftLogical: return mask_sllv_map;
    case ArithBinaryOp::RotateLeft: return mask_rolv_map;
    case ArithBinaryOp::RotateRight: return mask_rorv_map;
    default: x86_messed_up("missing mask intrinsic for int %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const VecIntrinsicUnaryMap roundscale_map{
    "roundscale",

    IXMMF32{CT, "_mm_roundscale_ps", CT, I32},
    IXMMF64{CT, "_mm_roundscale_pd", CT, I32},

    IYMMF32{CT, "_mm256_roundscale_ps", CT, I32},
    IYMMF64{CT, "_mm256_roundscale_pd", CT, I32},

    IZMMF32{CT, "_mm512_roundscale_ps", CT, I32},
    IZMMF64{CT, "_mm512_roundscale_pd", CT, I32},
};

const VecIntrinsicUnaryMap maskz_roundscale_map{
    "maskz_roundscale",

    IXMMF32{CT, "_mm_maskz_roundscale_ps", CMT, CT, I32},
    IXMMF64{CT, "_mm_maskz_roundscale_pd", CMT, CT, I32},

    IYMMF32{CT, "_mm256_maskz_roundscale_ps", CMT, CT, I32},
    IYMMF64{CT, "_mm256_maskz_roundscale_pd", CMT, CT, I32},

    IZMMF32{CT, "_mm512_maskz_roundscale_ps", CMT, CT, I32},
    IZMMF64{CT, "_mm512_maskz_roundscale_pd", CMT, CT, I32},
};

const VecIntrinsicUnaryMap mask_roundscale_map{
    "mask_roundscale",

    IXMMF32{CT, "_mm_mask_roundscale_ps", CT, CMT, CT, I32},
    IXMMF64{CT, "_mm_mask_roundscale_pd", CT, CMT, CT, I32},

    IYMMF32{CT, "_mm256_mask_roundscale_ps", CT, CMT, CT, I32},
    IYMMF64{CT, "_mm256_mask_roundscale_pd", CT, CMT, CT, I32},

    IZMMF32{CT, "_mm512_mask_roundscale_ps", CT, CMT, CT, I32},
    IZMMF64{CT, "_mm512_mask_roundscale_pd", CT, CMT, CT, I32},
};

const VecIntrinsicUnaryMap fpclass_map{
    "fpclass",

    IXMMF32{CMT, "_mm_fpclass_ps_mask", CT, I32},
    IXMMF64{CMT, "_mm_fpclass_pd_mask", CT, I32},

    IYMMF32{CMT, "_mm256_fpclass_ps_mask", CT, I32},
    IYMMF64{CMT, "_mm256_fpclass_pd_mask", CT, I32},

    IZMMF32{CMT, "_mm512_fpclass_ps_mask", CT, I32},
    IZMMF64{CMT, "_mm512_fpclass_pd_mask", CT, I32},
};

const VecIntrinsicUnaryMap set1_map{
    "set1",

    IXMMI8{CT, "_mm_set1_epi8", CurrentScalarType},
    IXMMI16{CT, "_mm_set1_epi16", CurrentScalarType},
    IXMMI32{CT, "_mm_set1_epi32", CurrentScalarType},
    IXMMI64{CT, "_mm_set1_epi64x", CurrentScalarType},
    IXMMF32{CT, "_mm_set1_ps", CurrentScalarType},
    IXMMF64{CT, "_mm_set1_pd", CurrentScalarType},

    IYMMI8{CT, "_mm256_set1_epi8", CurrentScalarType},
    IYMMI16{CT, "_mm256_set1_epi16", CurrentScalarType},
    IYMMI32{CT, "_mm256_set1_epi32", CurrentScalarType},
    IYMMI64{CT, "_mm256_set1_epi64x", CurrentScalarType},
    IYMMF32{CT, "_mm256_set1_ps", CurrentScalarType},
    IYMMF64{CT, "_mm256_set1_pd", CurrentScalarType},

    IZMMI8{CT, "_mm512_set1_epi8", CurrentScalarType},
    IZMMI16{CT, "_mm512_set1_epi16", CurrentScalarType},
    IZMMI32{CT, "_mm512_set1_epi32", CurrentScalarType},
    IZMMI64{CT, "_mm512_set1_epi64", CurrentScalarType},
    IZMMF32{CT, "_mm512_set1_ps", CurrentScalarType},
    IZMMF64{CT, "_mm512_set1_pd", CurrentScalarType},
};

const VecIntrinsicUnaryMap mask_i32gather_map{
    "mask_i32gather",

    IXMMI32(CT, "_mm_mmask_i32gather_epi32", CT, CMT, CT, I32, I32),
    IXMMI64(CT, "_mm_mmask_i32gather_epi64", CT, CMT, XMMI32, I64, I32),
    IXMMF32(CT, "_mm_mmask_i32gather_ps", CT, CMT, CT, F32, I32),
    IXMMF64(CT, "_mm_mmask_i32gather_pd", CT, CMT, XMMI32, F64, I32),

    IYMMI32(CT, "_mm256_mmask_i32gather_epi32", CT, CMT, CT, I32, I32),
    IYMMI64(CT, "_mm256_mmask_i32gather_epi64", CT, CMT, XMMI32, I64, I32),
    IYMMF32(CT, "_mm256_mmask_i32gather_ps", CT, CMT, CT, F32, I32),
    IYMMF64(CT, "_mm256_mmask_i32gather_pd", CT, CMT, XMMI32, F64, I32),

    IZMMI32(CT, "_mm512_mask_i32gather_epi32", CT, CMT, CT, I32, I32),
    IZMMI64(CT, "_mm512_mask_i32gather_epi64", CT, CMT, YMMI32, I64, I32),
    IZMMF32(CT, "_mm512_mask_i32gather_ps", CT, CMT, CT, F32, I32),
    IZMMF64(CT, "_mm512_mask_i32gather_pd", CT, CMT, YMMI32, F64, I32),
};

const VecIntrinsicUnaryMap mask_i64gather_map{
    "mask_i64gather",

    IXMMI32(CT, "_mm256_mmask_i64gather_epi32", CT, CMT, YMMI64, I32, I32),
    IXMMI64(CT, "_mm_mmask_i64gather_epi64", CT, CMT, XMMI64, I64, I32),
    IXMMF32(CT, "_mm256_mmask_i64gather_ps", CT, CMT, YMMI64, F32, I32),
    IXMMF64(CT, "_mm_mmask_i64gather_pd", CT, CMT, XMMI64, F64, I32),

    IYMMI32(CT, "_mm512_mask_i64gather_epi32", CT, CMT, ZMMI64, I32, I32),
    IYMMI64(CT, "_mm256_mmask_i64gather_epi64", CT, CMT, YMMI64, I64, I32),
    IYMMF32(CT, "_mm512_mask_i64gather_ps", CT, CMT, ZMMI64, F32, I32),
    IYMMF64(CT, "_mm256_mmask_i64gather_pd", CT, CMT, YMMI64, F64, I32),

    IZMMI64(CT, "_mm512_mask_i64gather_epi64", CT, CMT, ZMMI64, I64, I32),
    IZMMF64(CT, "_mm512_mask_i64gather_pd", CT, CMT, ZMMI64, F64, I32),
};

const VecIntrinsicUnaryMap i32scatter_map{
    "i32scatter",

    IXMMI32(CT, "_mm_i32scatter_epi32", I32, XMMI32, CT, I32),
    IXMMI64(CT, "_mm_i32scatter_epi64", I64, XMMI32, CT, I32),
    IXMMF32(CT, "_mm_i32scatter_ps", F32, XMMI32, CT, I32),
    IXMMF64(CT, "_mm_i32scatter_pd", F64, XMMI32, CT, I32),

    IYMMI32(CT, "_mm256_i32scatter_epi32", I32, YMMI32, CT, I32),
    IYMMI64(CT, "_mm256_i32scatter_epi64", I64, XMMI32, CT, I32),
    IYMMF32(CT, "_mm256_i32scatter_ps", F32, YMMI32, CT, I32),
    IYMMF64(CT, "_mm256_i32scatter_pd", F64, XMMI32, CT, I32),

    IZMMI32(CT, "_mm512_i32scatter_epi32", I32, ZMMI32, CT, I32),
    IZMMI64(CT, "_mm512_i32scatter_epi64", I64, YMMI32, CT, I32),
    IZMMF32(CT, "_mm512_i32scatter_ps", F32, ZMMI32, CT, I32),
    IZMMF64(CT, "_mm512_i32scatter_pd", F64, YMMI32, CT, I32),
};

const VecIntrinsicUnaryMap i64scatter_map{
    "i64scatter",

    IXMMI32(CT, "_mm256_i64scatter_epi32", I32, YMMI64, XMMI32, I32),
    IXMMI64(CT, "_mm_i64scatter_epi64", I64, XMMI64, XMMI64, I32),
    IXMMF32(CT, "_mm256_i64scatter_ps", F32, YMMI64, XMMF32, I32),
    IXMMF64(CT, "_mm_i64scatter_pd", F64, XMMI64, XMMF64, I32),

    IYMMI32(CT, "_mm512_i64scatter_epi32", I32, ZMMI64, YMMI32, I32),
    IYMMI64(CT, "_mm256_i64scatter_epi64", I64, YMMI64, YMMI64, I32),
    IYMMF32(CT, "_mm512_i64scatter_ps", F32, ZMMI64, YMMF32, I32),
    IYMMF64(CT, "_mm256_i64scatter_pd", F64, YMMI64, YMMF64, I32),

    IZMMI64(CT, "_mm512_i64scatter_epi64", I64, ZMMI64, ZMMI64, I32),
    IZMMF64(CT, "_mm512_i64scatter_pd", F64, ZMMI64, ZMMF64, I32),
};

const VecIntrinsicUnaryMap mask_i32scatter_map{
    "mask_i32scatter",

    IXMMI32(CT, "_mm_mask_i32scatter_epi32", I32, CMT, XMMI32, CT, I32),
    IXMMI64(CT, "_mm_mask_i32scatter_epi64", I64, CMT, XMMI32, CT, I32),
    IXMMF32(CT, "_mm_mask_i32scatter_ps", F32, CMT, XMMI32, CT, I32),
    IXMMF64(CT, "_mm_mask_i32scatter_pd", F64, CMT, XMMI32, CT, I32),

    IYMMI32(CT, "_mm256_mask_i32scatter_epi32", I32, CMT, YMMI32, CT, I32),
    IYMMI64(CT, "_mm256_mask_i32scatter_epi64", I64, CMT, XMMI32, CT, I32),
    IYMMF32(CT, "_mm256_mask_i32scatter_ps", F32, CMT, YMMI32, CT, I32),
    IYMMF64(CT, "_mm256_mask_i32scatter_pd", F64, CMT, XMMI32, CT, I32),

    IZMMI32(CT, "_mm512_mask_i32scatter_epi32", I32, CMT, ZMMI32, CT, I32),
    IZMMI64(CT, "_mm512_mask_i32scatter_epi64", I64, CMT, YMMI32, CT, I32),
    IZMMF32(CT, "_mm512_mask_i32scatter_ps", F32, CMT, ZMMI32, CT, I32),
    IZMMF64(CT, "_mm512_mask_i32scatter_pd", F64, CMT, YMMI32, CT, I32),
};

const VecIntrinsicUnaryMap mask_i64scatter_map{
    "mask_i64scatter",

    IXMMI32(CT, "_mm256_mask_i64scatter_epi32", I32, CMT, YMMI64, XMMI32, I32),
    IXMMI64(CT, "_mm_mask_i64scatter_epi64", I64, CMT, XMMI64, XMMI64, I32),
    IXMMF32(CT, "_mm256_mask_i64scatter_ps", F32, CMT, YMMI64, XMMF32, I32),
    IXMMF64(CT, "_mm_mask_i64scatter_pd", F64, CMT, XMMI64, XMMF64, I32),

    IYMMI32(CT, "_mm512_mask_i64scatter_epi32", I32, CMT, ZMMI64, YMMI32, I32),
    IYMMI64(CT, "_mm256_mask_i64scatter_epi64", I64, CMT, YMMI64, YMMI64, I32),
    IYMMF32(CT, "_mm512_mask_i64scatter_ps", F32, CMT, ZMMI64, YMMF32, I32),
    IYMMF64(CT, "_mm256_mask_i64scatter_pd", F64, CMT, YMMI64, YMMF64, I32),

    IZMMI64(CT, "_mm512_mask_i64scatter_epi64", I64, CMT, ZMMI64, ZMMI64, I32),
    IZMMF64(CT, "_mm512_mask_i64scatter_pd", F64, CMT, ZMMI64, ZMMF64, I32),
};

const VecBinaryIntrinsicMap extract_map{
    "extract",

    binary_intrin(XMMI8, IYMMI8(XMMI8, "_mm256_extractf128_si256", CT, I32)),
    binary_intrin(XMMI16, IYMMI16(XMMI16, "_mm256_extractf128_si256", CT, I32)),
    binary_intrin(XMMI32, IYMMI32(XMMI32, "_mm256_extractf128_si256", CT, I32)),
    binary_intrin(XMMI64, IYMMI64(XMMI64, "_mm256_extractf128_si256", CT, I32)),
    binary_intrin(XMMF32, IYMMF32(XMMF32, "_mm256_extractf32x4_ps", CT, I32)),
    binary_intrin(XMMF64, IYMMF64(XMMF64, "_mm256_extractf64x2_ps", CT, I32)),

    binary_intrin(XMMI8, IZMMI8(XMMI8, "_mm512_extracti32x4_epi32", CT, I32)),
    binary_intrin(XMMI16, IZMMI16(XMMI16, "_mm512_extracti32x4_epi32", CT, I32)),
    binary_intrin(XMMI32, IZMMI32(XMMI32, "_mm512_extracti32x4_epi32", CT, I32)),
    binary_intrin(XMMI64, IZMMI64(XMMI64, "_mm512_extracti64x2_epi64", CT, I32)),
    binary_intrin(XMMF32, IZMMF32(XMMF32, "_mm512_extractf32x4_ps", CT, I32)),
    binary_intrin(XMMF64, IZMMF64(XMMF64, "_mm512_extractf64x2_pd", CT, I32)),

    binary_intrin(YMMI8, IZMMI8(YMMI8, "_mm512_extracti32x8_epi32", CT, I32)),
    binary_intrin(YMMI16, IZMMI16(YMMI16, "_mm512_extracti32x8_epi32", CT, I32)),
    binary_intrin(YMMI32, IZMMI32(YMMI32, "_mm512_extracti32x8_epi32", CT, I32)),
    binary_intrin(YMMI64, IZMMI64(YMMI64, "_mm512_extracti64x4_epi64", CT, I32)),
    binary_intrin(YMMF32, IZMMF32(YMMF32, "_mm512_extractf32x8_ps", CT, I32)),
    binary_intrin(YMMF64, IZMMF64(YMMF64, "_mm512_extractf64x4_pd", CT, I32)),
};

const VecBinaryIntrinsicMap maskz_extract_map{
    "maskz_extract",

    binary_intrin(XMMI8, IYMMI8(XMMI8, "_mm256_maskz_extractf128_si256", CMT, CT, I32)),
    binary_intrin(XMMI16, IYMMI16(XMMI16, "_mm256_maskz_extractf128_si256", CMT, CT, I32)),
    binary_intrin(XMMI32, IYMMI32(XMMI32, "_mm256_maskz_extractf128_si256", CMT, CT, I32)),
    binary_intrin(XMMI64, IYMMI64(XMMI64, "_mm256_maskz_extractf128_si256", CMT, CT, I32)),

    binary_intrin(XMMI8, IZMMI8(XMMI8, "_mm512_maskz_extracti32x4_epi32", CMT, CT, I32)),
    binary_intrin(XMMI16, IZMMI16(XMMI16, "_mm512_maskz_extracti32x4_epi32", CMT, CT, I32)),
    binary_intrin(XMMI32, IZMMI32(XMMI32, "_mm512_maskz_extracti32x4_epi32", CMT, CT, I32)),
    binary_intrin(XMMI64, IZMMI64(XMMI64, "_mm512_maskz_extracti64x2_epi64", CMT, CT, I32)),

    binary_intrin(YMMI8, IZMMI8(YMMI8, "_mm512_maskz_extracti32x8_epi32", CMT, CT, I32)),
    binary_intrin(YMMI16, IZMMI16(YMMI16, "_mm512_maskz_extracti32x8_epi32", CMT, CT, I32)),
    binary_intrin(YMMI32, IZMMI32(YMMI32, "_mm512_maskz_extracti32x8_epi32", CMT, CT, I32)),
    binary_intrin(YMMI64, IZMMI64(YMMI64, "_mm512_maskz_extracti64x4_epi64", CMT, CT, I32)),
};

const VecBinaryIntrinsicMap mask_extract_map{
    "mask_extract",

    binary_intrin(XMMI8, IYMMI8(XMMI8, "_mm256_mask_extractf128_si256", XMMI8, CMT, CT, I32)),
    binary_intrin(XMMI16, IYMMI16(XMMI16, "_mm256_mask_extractf128_si256", XMMI16, CMT, CT, I32)),
    binary_intrin(XMMI32, IYMMI32(XMMI32, "_mm256_mask_extractf128_si256", XMMI32, CMT, CT, I32)),
    binary_intrin(XMMI64, IYMMI64(XMMI64, "_mm256_mask_extractf128_si256", XMMI64, CMT, CT, I32)),

    binary_intrin(XMMI8, IZMMI8(XMMI8, "_mm512_mask_extracti32x4_epi32", XMMI8, CMT, CT, I32)),
    binary_intrin(XMMI16, IZMMI16(XMMI16, "_mm512_mask_extracti32x4_epi32", XMMI16, CMT, CT, I32)),
    binary_intrin(XMMI32, IZMMI32(XMMI32, "_mm512_mask_extracti32x4_epi32", XMMI32, CMT, CT, I32)),
    binary_intrin(XMMI64, IZMMI64(XMMI64, "_mm512_mask_extracti64x2_epi64", XMMI64, CMT, CT, I32)),

    binary_intrin(YMMI8, IZMMI8(YMMI8, "_mm512_mask_extracti32x8_epi32", YMMI8, CMT, CT, I32)),
    binary_intrin(YMMI16, IZMMI16(YMMI16, "_mm512_mask_extracti32x8_epi32", YMMI16, CMT, CT, I32)),
    binary_intrin(YMMI32, IZMMI32(YMMI32, "_mm512_mask_extracti32x8_epi32", YMMI32, CMT, CT, I32)),
    binary_intrin(YMMI64, IZMMI64(YMMI64, "_mm512_mask_extracti64x4_epi64", YMMI64, CMT, CT, I32)),
};

const VecBinaryIntrinsicMap cvt_map{
    "cvt",

    binary_intrin(XMMI16, IXMMI8(XMMI16, "_mm_cvtepi8_epi16", CT)),
    binary_intrin(YMMI16, IXMMI8(YMMI16, "_mm256_cvtepi8_epi16", CT)),
    binary_intrin(ZMMI16, IYMMI8(ZMMI16, "_mm512_cvtepi8_epi16", CT)),
    binary_intrin(XMMI32, IXMMI8(XMMI32, "_mm_cvtepi8_epi32", CT)),
    binary_intrin(ZMMI32, IXMMI8(ZMMI32, "_mm512_cvtepi8_epi32", CT)),
    binary_intrin(XMMI64, IXMMI8(XMMI64, "_mm_cvtepi8_epi64", CT)),

    binary_intrin(XMMI8, IXMMI16(XMMI8, "_mm_cvtepi16_epi8", CT)),
    binary_intrin(XMMI8, IYMMI16(XMMI8, "_mm256_cvtepi16_epi8", CT)),
    binary_intrin(YMMI8, IZMMI16(YMMI8, "_mm512_cvtepi16_epi8", CT)),
    binary_intrin(XMMI32, IXMMI16(XMMI32, "_mm_cvtepi16_epi32", CT)),
    binary_intrin(XMMI16, IXMMI32(XMMI16, "_mm_cvtepi32_epi16", CT)),
    binary_intrin(YMMI32, IXMMI16(YMMI32, "_mm256_cvtepi16_epi32", CT)),
    binary_intrin(ZMMI32, IYMMI16(ZMMI32, "_mm512_cvtepi16_epi32", CT)),
    binary_intrin(XMMI64, IXMMI16(XMMI64, "_mm_cvtepi16_epi64", CT)),
    binary_intrin(ZMMI64, IXMMI16(ZMMI64, "_mm512_cvtepi16_epi64", CT)),

    binary_intrin(XMMI8, IXMMI32(XMMI8, "_mm_cvtepi32_epi8", CT)),
    binary_intrin(XMMI8, IZMMI32(XMMI8, "_mm512_cvtepi32_epi8", CT)),
    binary_intrin(XMMI16, IYMMI32(XMMI16, "_mm256_cvtepi32_epi16", CT)),
    binary_intrin(YMMI16, IZMMI32(YMMI16, "_mm512_cvtepi32_epi16", CT)),
    binary_intrin(XMMI64, IXMMI32(XMMI64, "_mm_cvtepi32_epi64", CT)),
    binary_intrin(XMMI32, IXMMI64(XMMI32, "_mm_cvtepi64_epi32", CT)),
    binary_intrin(YMMI64, IXMMI32(YMMI64, "_mm256_cvtepi32_epi64", CT)),
    binary_intrin(ZMMI64, IYMMI32(ZMMI64, "_mm512_cvtepi32_epi64", CT)),

    binary_intrin(XMMI8, IXMMI64(XMMI8, "_mm_cvtepi64_epi8", CT)),
    binary_intrin(XMMI16, IXMMI64(XMMI16, "_mm_cvtepi64_epi16", CT)),
    binary_intrin(XMMI16, IZMMI64(XMMI16, "_mm512_cvtepi64_epi16", CT)),
    binary_intrin(XMMI32, IYMMI64(XMMI32, "_mm256_cvtepi64_epi32", CT)),
    binary_intrin(YMMI32, IZMMI64(YMMI32, "_mm512_cvtepi64_epi32", CT)),
};

const VecBinaryIntrinsicMap maskz_cvt_map{
    "maskz_cvt",

    binary_intrin(YMMI16, IXMMI8(YMMI16, "_mm256_maskz_cvtepi8_epi16", M16, CT)),
    binary_intrin(ZMMI16, IYMMI8(ZMMI16, "_mm512_maskz_cvtepi8_epi16", M32, CT)),
    binary_intrin(ZMMI32, IXMMI8(ZMMI32, "_mm512_maskz_cvtepi8_epi32", M16, CT)),

    binary_intrin(XMMI8, IYMMI16(XMMI8, "_mm256_maskz_cvtepi16_epi8", M16, CT)),
    binary_intrin(YMMI8, IZMMI16(YMMI8, "_mm512_maskz_cvtepi16_epi8", M32, CT)),
    binary_intrin(YMMI32, IXMMI16(YMMI32, "_mm256_maskz_cvtepi16_epi32", M8, CT)),
    binary_intrin(ZMMI32, IYMMI16(ZMMI32, "_mm512_maskz_cvtepi16_epi32", M16, CT)),
    binary_intrin(ZMMI64, IXMMI16(ZMMI64, "_mm512_maskz_cvtepi16_epi64", M8, CT)),

    binary_intrin(XMMI8, IZMMI32(XMMI8, "_mm512_maskz_cvtepi32_epi8", M16, CT)),
    binary_intrin(XMMI16, IYMMI32(XMMI16, "_mm256_maskz_cvtepi32_epi16", M8, CT)),
    binary_intrin(YMMI16, IZMMI32(YMMI16, "_mm512_maskz_cvtepi32_epi16", M16, CT)),
    binary_intrin(YMMI64, IXMMI32(YMMI64, "_mm256_maskz_cvtepi32_epi64", M4, CT)),
    binary_intrin(ZMMI64, IYMMI32(ZMMI64, "_mm512_maskz_cvtepi32_epi64", M8, CT)),

    binary_intrin(XMMI16, IZMMI64(XMMI16, "_mm512_maskz_cvtepi64_epi16", M8, CT)),
    binary_intrin(XMMI32, IYMMI64(XMMI32, "_mm256_maskz_cvtepi64_epi32", M4, CT)),
    binary_intrin(YMMI32, IZMMI64(YMMI32, "_mm512_maskz_cvtepi64_epi32", M8, CT)),
};

const VecBinaryIntrinsicMap mask_cvt_map{
    "mask_cvt",

    binary_intrin(YMMI16, IXMMI8(YMMI16, "_mm256_mask_cvtepi8_epi16", YMMI16, M16, CT)),
    binary_intrin(ZMMI16, IYMMI8(ZMMI16, "_mm512_mask_cvtepi8_epi16", ZMMI16, M32, CT)),
    binary_intrin(ZMMI32, IXMMI8(ZMMI32, "_mm512_mask_cvtepi8_epi32", ZMMI32, M16, CT)),

    binary_intrin(XMMI8, IYMMI16(XMMI8, "_mm256_mask_cvtepi16_epi8", XMMI8, M16, CT)),
    binary_intrin(YMMI8, IZMMI16(YMMI8, "_mm512_mask_cvtepi16_epi8", YMMI8, M32, CT)),
    binary_intrin(YMMI32, IXMMI16(YMMI32, "_mm256_mask_cvtepi16_epi32", YMMI32, M8, CT)),
    binary_intrin(ZMMI32, IYMMI16(ZMMI32, "_mm512_mask_cvtepi16_epi32", ZMMI32, M16, CT)),
    binary_intrin(ZMMI64, IXMMI16(ZMMI64, "_mm512_mask_cvtepi16_epi64", ZMMI64, M8, CT)),

    binary_intrin(XMMI8, IZMMI32(XMMI8, "_mm512_mask_cvtepi32_epi8", XMMI8, M16, CT)),
    binary_intrin(XMMI16, IYMMI32(XMMI16, "_mm256_mask_cvtepi32_epi16", XMMI16, M8, CT)),
    binary_intrin(YMMI16, IZMMI32(YMMI16, "_mm512_mask_cvtepi32_epi16", YMMI16, M16, CT)),
    binary_intrin(YMMI64, IXMMI32(YMMI64, "_mm256_mask_cvtepi32_epi64", YMMI64, M4, CT)),
    binary_intrin(ZMMI64, IYMMI32(ZMMI64, "_mm512_mask_cvtepi32_epi64", ZMMI64, M8, CT)),

    binary_intrin(XMMI16, IZMMI64(XMMI16, "_mm512_mask_cvtepi64_epi16", XMMI16, M8, CT)),
    binary_intrin(XMMI32, IYMMI64(XMMI32, "_mm256_mask_cvtepi64_epi32", XMMI32, M4, CT)),
    binary_intrin(YMMI32, IZMMI64(YMMI32, "_mm512_mask_cvtepi64_epi32", YMMI32, M8, CT)),
};

const VecBinaryIntrinsicMap float_cast_map{
    "float_cast",

    binary_intrin(XMMF64, IXMMF32(XMMF64, "_mm_cvtps_pd", CT)),
    binary_intrin(YMMF64, IXMMF32(YMMF64, "_mm256_cvtps_pd", CT)),
    binary_intrin(ZMMF64, IYMMF32(ZMMF64, "_mm512_cvtps_pd", CT)),

    binary_intrin(XMMF32, IYMMF64(XMMF32, "_mm256_cvtpd_ps", CT)),
    binary_intrin(YMMF32, IZMMF64(YMMF32, "_mm512_cvtpd_ps", CT)),

    binary_intrin(XMMF32, IXMMI32(XMMF32, "_mm_cvtepi32_ps", CT)),
    binary_intrin(YMMF32, IYMMI32(YMMF32, "_mm256_cvtepi32_ps", CT)),
    binary_intrin(ZMMF32, IZMMI32(ZMMF32, "_mm512_cvtepi32_ps", CT)),

    binary_intrin(YMMF64, IXMMI32(YMMF64, "_mm256_cvtepi32_pd", CT)),
    binary_intrin(ZMMF64, IYMMI32(ZMMF64, "_mm512_cvtepi32_pd", CT)),

    binary_intrin(XMMF32, IYMMI64(XMMF32, "_mm256_cvtepi64_ps", CT)),
    binary_intrin(YMMF32, IZMMI64(YMMF32, "_mm512_cvtepi64_ps", CT)),

    binary_intrin(XMMF64, IXMMI64(XMMF64, "_mm_cvtepi64_pd", CT)),
    binary_intrin(YMMF64, IYMMI64(YMMF64, "_mm256_cvtepi64_pd", CT)),
    binary_intrin(ZMMF64, IZMMI64(ZMMF64, "_mm512_cvtepi64_pd", CT)),

    binary_intrin(XMMI32, IXMMF32(XMMI32, "_mm_cvttps_epi32", CT)),
    binary_intrin(YMMI32, IYMMF32(YMMI32, "_mm256_cvttps_epi32", CT)),
    binary_intrin(ZMMI32, IZMMF32(ZMMI32, "_mm512_cvttps_epi32", CT)),

    binary_intrin(XMMI32, IYMMF64(XMMI32, "_mm256_cvttpd_epi32", CT)),
    binary_intrin(YMMI32, IZMMF64(YMMI32, "_mm512_cvttpd_epi32", CT)),

    binary_intrin(YMMI64, IXMMF32(YMMI64, "_mm256_cvttps_epi64", CT)),
    binary_intrin(ZMMI64, IYMMF32(ZMMI64, "_mm512_cvttps_epi64", CT)),

    binary_intrin(XMMI64, IXMMF64(XMMI64, "_mm_cvttpd_epi64", CT)),
    binary_intrin(YMMI64, IYMMF64(YMMI64, "_mm256_cvttpd_epi64", CT)),
    binary_intrin(ZMMI64, IZMMF64(ZMMI64, "_mm512_cvttpd_epi64", CT)),
};

const VecBinaryIntrinsicMap float_ucast_map{
    "float_ucast",

    binary_intrin(XMMF32, IXMMI32(XMMF32, "_mm_cvtepu32_ps", CT)),
    binary_intrin(YMMF32, IYMMI32(YMMF32, "_mm256_cvtepu32_ps", CT)),
    binary_intrin(ZMMF32, IZMMI32(ZMMF32, "_mm512_cvtepu32_ps", CT)),

    binary_intrin(YMMF64, IXMMI32(YMMF64, "_mm256_cvtepu32_pd", CT)),
    binary_intrin(ZMMF64, IYMMI32(ZMMF64, "_mm512_cvtepu32_pd", CT)),

    binary_intrin(XMMF32, IYMMI64(XMMF32, "_mm256_cvtepu64_ps", CT)),
    binary_intrin(YMMF32, IZMMI64(YMMF32, "_mm512_cvtepu64_ps", CT)),

    binary_intrin(XMMF64, IXMMI64(XMMF64, "_mm_cvtepu64_pd", CT)),
    binary_intrin(YMMF64, IYMMI64(YMMF64, "_mm256_cvtepu64_pd", CT)),
    binary_intrin(ZMMF64, IZMMI64(ZMMF64, "_mm512_cvtepu64_pd", CT)),

    binary_intrin(XMMI32, IXMMF32(XMMI32, "_mm_cvttps_epu32", CT)),
    binary_intrin(YMMI32, IYMMF32(YMMI32, "_mm256_cvttps_epu32", CT)),
    binary_intrin(ZMMI32, IZMMF32(ZMMI32, "_mm512_cvttps_epu32", CT)),

    binary_intrin(XMMI32, IYMMF64(XMMI32, "_mm256_cvttpd_epu32", CT)),
    binary_intrin(YMMI32, IZMMF64(YMMI32, "_mm512_cvttpd_epu32", CT)),

    binary_intrin(YMMI64, IXMMF32(YMMI64, "_mm256_cvttps_epu64", CT)),
    binary_intrin(ZMMI64, IYMMF32(ZMMI64, "_mm512_cvttps_epu64", CT)),

    binary_intrin(XMMI64, IXMMF64(XMMI64, "_mm_cvttpd_epu64", CT)),
    binary_intrin(YMMI64, IYMMF64(YMMI64, "_mm256_cvttpd_epu64", CT)),
    binary_intrin(ZMMI64, IZMMF64(ZMMI64, "_mm512_cvttpd_epu64", CT)),
};

const VecIntrinsicUnaryMap bitcast_map{
    "bitcast",

    IXMMI32(CT, "_mm_castps_si128", XMMF32),
    IXMMI64(CT, "_mm_castpd_si128", XMMF64),
    IXMMF32(CT, "_mm_castsi128_ps", XMMI32),
    IXMMF64(CT, "_mm_castsi128_pd", XMMI64),

    IYMMI32(CT, "_mm256_castps_si256", YMMF32),
    IYMMI64(CT, "_mm256_castpd_si256", YMMF64),
    IYMMF32(CT, "_mm256_castsi256_ps", YMMI32),
    IYMMF64(CT, "_mm256_castsi256_pd", YMMI64),

    IZMMI32(CT, "_mm512_castps_si512", ZMMF32),
    IZMMI64(CT, "_mm512_castpd_si512", ZMMF64),
    IZMMF32(CT, "_mm512_castsi512_ps", ZMMI32),
    IZMMF64(CT, "_mm512_castsi512_pd", ZMMI64),
};

const VecIntrinsicUnaryMap fmadd_map{
    "fmadd",

    IXMMF32(CT, "_mm_fmadd_ps", CT, CT, CT),
    IXMMF64(CT, "_mm_fmadd_pd", CT, CT, CT),

    IYMMF32(CT, "_mm256_fmadd_ps", CT, CT, CT),
    IYMMF64(CT, "_mm256_fmadd_pd", CT, CT, CT),

    IZMMF32(CT, "_mm512_fmadd_ps", CT, CT, CT),
    IZMMF64(CT, "_mm512_fmadd_pd", CT, CT, CT),
};

const VecIntrinsicUnaryMap fmsub_map{
    "fmsub",

    IXMMF32(CT, "_mm_fmsub_ps", CT, CT, CT),
    IXMMF64(CT, "_mm_fmsub_pd", CT, CT, CT),

    IYMMF32(CT, "_mm256_fmsub_ps", CT, CT, CT),
    IYMMF64(CT, "_mm256_fmsub_pd", CT, CT, CT),

    IZMMF32(CT, "_mm512_fmsub_ps", CT, CT, CT),
    IZMMF64(CT, "_mm512_fmsub_pd", CT, CT, CT),
};

const VecIntrinsicUnaryMap fnmadd_map{
    "fnmadd",

    IXMMF32(CT, "_mm_fnmadd_ps", CT, CT, CT),
    IXMMF64(CT, "_mm_fnmadd_pd", CT, CT, CT),

    IYMMF32(CT, "_mm256_fnmadd_ps", CT, CT, CT),
    IYMMF64(CT, "_mm256_fnmadd_pd", CT, CT, CT),

    IZMMF32(CT, "_mm512_fnmadd_ps", CT, CT, CT),
    IZMMF64(CT, "_mm512_fnmadd_pd", CT, CT, CT),
};

const VecIntrinsicUnaryMap fnmsub_map{
    "fnmsub",

    IXMMF32(CT, "_mm_fnmsub_ps", CT, CT, CT),
    IXMMF64(CT, "_mm_fnmsub_pd", CT, CT, CT),

    IYMMF32(CT, "_mm256_fnmsub_ps", CT, CT, CT),
    IYMMF64(CT, "_mm256_fnmsub_pd", CT, CT, CT),

    IZMMF32(CT, "_mm512_fnmsub_ps", CT, CT, CT),
    IZMMF64(CT, "_mm512_fnmsub_pd", CT, CT, CT),
};

const VecIntrinsicUnaryMap &fma_map(mir::FmaKind kind) {
    switch (kind) {
    case mir::FmaKind::FMA: return fmadd_map;
    case mir::FmaKind::FMS: return fmsub_map;
    case mir::FmaKind::FNMA: return fnmadd_map;
    case mir::FmaKind::FNMS: return fnmsub_map;
    }
    SIMJIT_UNREACHABLE();
}

const VecBinaryIntrinsicMap zext_map{
    "zext",

    binary_intrin(XMMI16, IXMMI8(XMMI16, "_mm_cvtepu8_epi16", CT)),
    binary_intrin(YMMI16, IXMMI8(YMMI16, "_mm256_cvtepu8_epi16", CT)),
    binary_intrin(ZMMI16, IYMMI8(ZMMI16, "_mm512_cvtepu8_epi16", CT)),
    binary_intrin(XMMI32, IXMMI8(XMMI32, "_mm_cvtepu8_epi32", CT)),
    binary_intrin(ZMMI32, IXMMI8(ZMMI32, "_mm512_cvtepu8_epi32", CT)),
    binary_intrin(XMMI64, IXMMI8(XMMI64, "_mm_cvtepu8_epi64", CT)),

    binary_intrin(XMMI32, IXMMI16(XMMI32, "_mm_cvtepu16_epi32", CT)),
    binary_intrin(YMMI32, IXMMI16(YMMI32, "_mm256_cvtepu16_epi32", CT)),
    binary_intrin(ZMMI32, IYMMI16(ZMMI32, "_mm512_cvtepu16_epi32", CT)),
    binary_intrin(XMMI64, IXMMI16(XMMI64, "_mm_cvtepu16_epi64", CT)),
    binary_intrin(ZMMI64, IXMMI16(ZMMI64, "_mm512_cvtepu16_epi64", CT)),

    binary_intrin(XMMI64, IXMMI32(XMMI64, "_mm_cvtepu32_epi64", CT)),
    binary_intrin(YMMI64, IXMMI32(YMMI64, "_mm256_cvtepu32_epi64", CT)),
    binary_intrin(ZMMI64, IYMMI32(ZMMI64, "_mm512_cvtepu32_epi64", CT)),
};

const VecBinaryIntrinsicMap maskz_zext_map{
    "maskz_zext",

    binary_intrin(YMMI16, IXMMI8(YMMI16, "_mm256_maskz_cvtepu8_epi16", M16, CT)),
    binary_intrin(ZMMI16, IYMMI8(ZMMI16, "_mm512_maskz_cvtepu8_epi16", M32, CT)),
    binary_intrin(ZMMI32, IXMMI8(ZMMI32, "_mm512_maskz_cvtepu8_epi32", M16, CT)),

    binary_intrin(YMMI32, IXMMI16(YMMI32, "_mm256_maskz_cvtepu16_epi32", M8, CT)),
    binary_intrin(ZMMI32, IYMMI16(ZMMI32, "_mm512_maskz_cvtepu16_epi32", M16, CT)),
    binary_intrin(ZMMI64, IXMMI16(ZMMI64, "_mm512_maskz_cvtepu16_epi64", M8, CT)),

    binary_intrin(YMMI64, IXMMI32(YMMI64, "_mm256_maskz_cvtepu32_epi64", M4, CT)),
    binary_intrin(ZMMI64, IYMMI32(ZMMI64, "_mm512_maskz_cvtepu32_epi64", M8, CT)),
};

const VecBinaryIntrinsicMap mask_zext_map{
    "mask_zext",

    binary_intrin(YMMI16, IXMMI8(YMMI16, "_mm256_mask_cvtepu8_epi16", YMMI16, M16, CT)),
    binary_intrin(ZMMI16, IYMMI8(ZMMI16, "_mm512_mask_cvtepu8_epi16", ZMMI16, M32, CT)),
    binary_intrin(ZMMI32, IXMMI8(ZMMI32, "_mm512_mask_cvtepu8_epi32", ZMMI32, M16, CT)),

    binary_intrin(YMMI32, IXMMI16(YMMI32, "_mm256_mask_cvtepu16_epi32", YMMI32, M8, CT)),
    binary_intrin(ZMMI32, IYMMI16(ZMMI32, "_mm512_mask_cvtepu16_epi32", ZMMI32, M16, CT)),
    binary_intrin(ZMMI64, IXMMI16(ZMMI64, "_mm512_mask_cvtepu16_epi64", ZMMI64, M8, CT)),

    binary_intrin(YMMI64, IXMMI32(YMMI64, "_mm256_mask_cvtepu32_epi64", YMMI64, M4, CT)),
    binary_intrin(ZMMI64, IYMMI32(ZMMI64, "_mm512_mask_cvtepu32_epi64", ZMMI64, M8, CT)),
};

const VecBinaryIntrinsicMap cvt_storeu_map{
    "cvt_storeu",

    binary_intrin(XMMI8, IYMMI16(XMMI8, "_mm256_mask_cvtepi16_storeu_epi8", I8, M16, CT)),
    binary_intrin(YMMI8, IZMMI16(YMMI8, "_mm512_mask_cvtepi16_storeu_epi8", I8, M32, CT)),

    binary_intrin(XMMI8, IZMMI32(XMMI8, "_mm512_mask_cvtepi32_storeu_epi8", I8, M16, CT)),
    binary_intrin(XMMI16, IYMMI32(XMMI16, "_mm256_mask_cvtepi32_storeu_epi16", I16, M8, CT)),
    binary_intrin(YMMI16, IZMMI32(YMMI16, "_mm512_mask_cvtepi32_storeu_epi16", I16, M16, CT)),

    binary_intrin(XMMI16, IZMMI64(XMMI16, "_mm512_mask_cvtepi64_storeu_epi16", I16, M8, CT)),
    binary_intrin(XMMI32, IYMMI64(XMMI32, "_mm256_mask_cvtepi64_storeu_epi32", I32, M4, CT)),
    binary_intrin(YMMI32, IZMMI64(YMMI32, "_mm512_mask_cvtepi64_storeu_epi32", I32, M8, CT)),
};

const VecBinaryIntrinsicMap signed_saturate_map{
    "unsigned_saturate",

    binary_intrin(XMMI8, IYMMI16(XMMI8, "_mm256_cvtsepi16_epi8", CT)),
    binary_intrin(YMMI8, IZMMI16(YMMI8, "_mm512_cvtsepi16_epi8", CT)),

    binary_intrin(XMMI8, IZMMI32(XMMI8, "_mm512_cvtsepi32_epi8", CT)),
    binary_intrin(XMMI16, IYMMI32(YMMI16, "_mm256_cvtsepi32_epi16", CT)),
    binary_intrin(YMMI16, IZMMI32(YMMI16, "_mm512_cvtsepi32_epi16", CT)),

    binary_intrin(XMMI16, IZMMI64(XMMI16, "_mm512_cvtsepi64_epi16", CT)),
    binary_intrin(XMMI32, IYMMI64(XMMI32, "_mm256_cvtsepi64_epi32", CT)),
    binary_intrin(YMMI32, IZMMI64(YMMI32, "_mm512_cvtsepi64_epi32", CT)),
};

const VecBinaryIntrinsicMap unsigned_saturate_map{
    "signed_saturate",

    binary_intrin(XMMI8, IYMMI16(XMMI8, "_mm256_cvtsuepi16_epi8", CT)),
    binary_intrin(YMMI8, IZMMI16(YMMI8, "_mm512_cvtusepi16_epi8", CT)),

    binary_intrin(XMMI8, IZMMI32(XMMI8, "_mm512_cvtusepi32_epi8", CT)),
    binary_intrin(XMMI16, IYMMI32(YMMI16, "_mm256_cvtusepi32_epi16", CT)),
    binary_intrin(YMMI16, IZMMI32(YMMI16, "_mm512_cvtusepi32_epi16", CT)),

    binary_intrin(XMMI16, IZMMI64(XMMI16, "_mm512_cvtusepi64_epi16", CT)),
    binary_intrin(XMMI32, IYMMI64(XMMI32, "_mm256_cvtusepi64_epi32", CT)),
    binary_intrin(YMMI32, IZMMI64(YMMI32, "_mm512_cvtusepi64_epi32", CT)),
};

const MaskIntrinsicUnaryMap and_mask_map{
    "and_mask",

    IM8(CT, "_kand_mask8", CT, CT),
    IM16(CT, "_kand_mask16", CT, CT),
    IM32(CT, "_kand_mask32", CT, CT),
    IM64(CT, "_kand_mask64", CT, CT),
};

const MaskIntrinsicUnaryMap or_mask_map{
    "or_mask",

    IM8(CT, "_kor_mask8", CT, CT),
    IM16(CT, "_kor_mask16", CT, CT),
    IM32(CT, "_kor_mask32", CT, CT),
    IM64(CT, "_kor_mask64", CT, CT),
};

const MaskIntrinsicUnaryMap andn_mask_map{
    "andn_mask",

    IM8(CT, "_kandn_mask8", CT, CT),
    IM16(CT, "_kandn_mask16", CT, CT),
    IM32(CT, "_kandn_mask32", CT, CT),
    IM64(CT, "_kandn_mask64", CT, CT),
};

const MaskIntrinsicUnaryMap xnor_mask_map{
    "xnor_mask",

    IM8(CT, "_kxnor_mask8", CT, CT),
    IM16(CT, "_kxnor_mask16", CT, CT),
    IM32(CT, "_kxnor_mask32", CT, CT),
    IM64(CT, "_kxnor_mask64", CT, CT),
};

const MaskIntrinsicUnaryMap xor_mask_map{
    "xor_mask",

    IM8(CT, "_kxor_mask8", CT, CT),
    IM16(CT, "_kxor_mask16", CT, CT),
    IM32(CT, "_kxor_mask32", CT, CT),
    IM64(CT, "_kxor_mask64", CT, CT),
};

const MaskIntrinsicUnaryMap &binary_op_mask_map(PredicateBinaryOp op) {
    switch (op) {
    case PredicateBinaryOp::And: return and_mask_map;
    case PredicateBinaryOp::Or: return or_mask_map;
    case PredicateBinaryOp::AndNot: return andn_mask_map;
    case PredicateBinaryOp::XNor: return xnor_mask_map;
    case PredicateBinaryOp::Xor: return xor_mask_map;
    }
    SIMJIT_UNREACHABLE();
}

const MaskIntrinsicUnaryMap not_mask_map{
    "not_mask",

    IM8(CT, "_knot_mask8", CT),
    IM16(CT, "_knot_mask16", CT),
    IM32(CT, "_knot_mask32", CT),
    IM64(CT, "_knot_mask64", CT),
};

const MaskIntrinsicUnaryMap cvtmask_u_map{
    "cvtmask_u",

    IM8(I8, "_cvtmask8_u32", CT),
    IM16(I16, "_cvtmask16_u32", CT),
    IM32(I32, "_cvtmask32_u32", CT),
    IM64(I64, "_cvtmask64_u64", CT),
};

const MaskIntrinsicUnaryMap cvtu_mask_map{
    "cvtu_mask",

    IM8(CT, "_cvtu32_mask8", I32),
    IM16(CT, "_cvtu32_mask16", I32),
    IM32(CT, "_cvtu32_mask32", I32),
    IM64(CT, "_cvtu64_mask64", I64),
};

const MaskIntrinsicUnaryMap kunpack_map{
    "kunpack",

    IM16(CT, "_mm512_kunpackb", M8, M8),
    IM32(CT, "_mm512_kunpackw", M16, M16),
    IM64(CT, "_mm512_kunpackd", M32, M32),
};

const MaskIntrinsicUnaryMap ktestc_map{
    "ktestc",

    IM8(CT, "_ktestc_mask8_u8", CT, CT),
    IM16(CT, "_ktestc_mask16_u8", CT, CT),
    IM32(CT, "_ktestc_mask32_u8", CT, CT),
    IM64(CT, "_ktestc_mask64_u8", CT, CT),
};

const MaskIntrinsicUnaryMap ktestz_map{
    "ktestz",

    IM8(CT, "_ktestz_mask8_u8", CT, CT),
    IM16(CT, "_ktestz_mask16_u8", CT, CT),
    IM32(CT, "_ktestz_mask32_u8", CT, CT),
    IM64(CT, "_ktestz_mask64_u8", CT, CT),
};

const MaskIntrinsicUnaryMap kortestz_map{
    "kortestz",

    IM8(CT, "_kortestz_mask8_u8", CT, CT),
    IM16(CT, "_kortestz_mask16_u8", CT, CT),
    IM32(CT, "_kortestz_mask32_u8", CT, CT),
    IM64(CT, "_kortestz_mask64_u8", CT, CT),
};

const MaskIntrinsicUnaryMap kortestc_map{
    "kortestc",

    IM8(CT, "_kortestc_mask8_u8", CT, CT),
    IM16(CT, "_kortestc_mask16_u8", CT, CT),
    IM32(CT, "_kortestc_mask32_u8", CT, CT),
    IM64(CT, "_kortestc_mask64_u8", CT, CT),
};

const VecBinaryIntrinsicMap compiler_downcast_map{
    "downcast",

    // ymm -> xmm
    binary_intrin(XMMI8, IYMMI8(XMMI8, "_mm256_castsi256_si128", CT)),
    binary_intrin(XMMI16, IYMMI16(XMMI16, "_mm256_castsi256_si128", CT)),
    binary_intrin(XMMI32, IYMMI32(XMMI32, "_mm256_castsi256_si128", CT)),
    binary_intrin(XMMI64, IYMMI64(XMMI64, "_mm256_castsi256_si128", CT)),
    binary_intrin(XMMF32, IYMMF32(XMMF32, "_mm256_castps256_ps128", CT)),
    binary_intrin(XMMF64, IYMMF64(XMMF64, "_mm256_castpd256_pd128", CT)),

    // zmm -> xmm
    binary_intrin(XMMI8, IZMMI8(XMMI8, "_mm512_castsi512_si128", CT)),
    binary_intrin(XMMI16, IZMMI16(XMMI16, "_mm512_castsi512_si128", CT)),
    binary_intrin(XMMI32, IZMMI32(XMMI32, "_mm512_castsi512_si128", CT)),
    binary_intrin(XMMI64, IZMMI64(XMMI64, "_mm512_castsi512_si128", CT)),
    binary_intrin(XMMF32, IZMMF32(XMMF32, "_mm512_castps512_ps128", CT)),
    binary_intrin(XMMF64, IZMMF64(XMMF64, "_mm512_castpd512_pd128", CT)),

    // zmm -> ymm
    binary_intrin(YMMI8, IZMMI8(YMMI8, "_mm512_castsi512_si256", CT)),
    binary_intrin(YMMI16, IZMMI16(YMMI16, "_mm512_castsi512_si256", CT)),
    binary_intrin(YMMI32, IZMMI32(YMMI32, "_mm512_castsi512_si256", CT)),
    binary_intrin(YMMI64, IZMMI64(YMMI64, "_mm512_castsi512_si256", CT)),
    binary_intrin(YMMF32, IZMMF32(YMMF32, "_mm512_castps512_ps256", CT)),
    binary_intrin(YMMF64, IZMMF64(YMMF64, "_mm512_castpd512_pd256", CT)),
};

const ScalarIntrinsicUnaryMap scalar_popcnt_map{
    "scalar_popcnt",

    ISI8(CT, "_mm_popcnt_u32", CT),
    ISI16(CT, "_mm_popcnt_u32", CT),
    ISI32(CT, "_mm_popcnt_u32", CT),
    ISI64(CT, "_mm_popcnt_u64", CT),
};

const ScalarIntrinsicUnaryMap pext_map{
    "pext",

    ISI32(CT, "_pext_u32", CT, CT),
    ISI64(CT, "_pext_u64", CT, CT),
};
const ScalarIntrinsicUnaryMap pdep_map{
    "pdep",

    ISI32(CT, "_pdep_u32", CT, CT),
    ISI64(CT, "_pdep_u64", CT, CT),
};
const ScalarIntrinsicUnaryMap andn_map{
    "andn",

    ISI32(CT, "_andn_u32", CT, CT),
    ISI64(CT, "_andn_u64", CT, CT),
};
const ScalarIntrinsicUnaryMap blsmsk_map{
    "blsmsk",

    ISI32(CT, "_blsmsk_u32", CT),
    ISI64(CT, "_blsmsk_u64", CT),
};
const ScalarIntrinsicUnaryMap tzcnt_map{
    "tzcnt",

    ISI32(CT, "_tzcnt_u32", CT),
    ISI64(CT, "_tzcnt_u64", CT),
};

const VecIntrinsicUnaryMap &arith_unary_map(ArithUnaryOp op) {
    switch (op) {
    case ArithUnaryOp::Abs: return abs_map;
    case ArithUnaryOp::Lzcnt: return lzcnt_map;
    case ArithUnaryOp::Popcount: return popcnt_map;
    case ArithUnaryOp::Not:
    case ArithUnaryOp::Negate:
    case ArithUnaryOp::Tzcnt:
    case ArithUnaryOp::RoundNearest:
    case ArithUnaryOp::RoundDown:
    case ArithUnaryOp::RoundUp:
    case ArithUnaryOp::RoundTruncate:
        x86_messed_up("unary op %s doesn't have direct intrinsic", show_arith_unary_op(op));
    case ArithUnaryOp::Rcp: return rcp_map;
    case ArithUnaryOp::Sqrt: return sqrt_map;
    case ArithUnaryOp::Rsqrt: return rsqrt_map;
    }
    SIMJIT_UNREACHABLE();
}

const VecIntrinsicUnaryMap &maskz_arith_unary_map(ArithUnaryOp op) {
    switch (op) {
    case ArithUnaryOp::Abs: return maskz_abs_map;
    case ArithUnaryOp::Lzcnt: return maskz_lzcnt_map;
    case ArithUnaryOp::Popcount: return maskz_popcnt_map;
    case ArithUnaryOp::Not:
    case ArithUnaryOp::Negate:
    case ArithUnaryOp::Tzcnt:
    case ArithUnaryOp::RoundNearest:
    case ArithUnaryOp::RoundDown:
    case ArithUnaryOp::RoundUp:
    case ArithUnaryOp::RoundTruncate:
        x86_messed_up("unary op %s doesn't have direct intrinsic", show_arith_unary_op(op));
    case ArithUnaryOp::Rcp: return maskz_rcp_map;
    case ArithUnaryOp::Sqrt: return maskz_sqrt_map;
    case ArithUnaryOp::Rsqrt: return maskz_rsqrt_map;
    }
    SIMJIT_UNREACHABLE();
}

const VecIntrinsicUnaryMap &mask_arith_unary_map(ArithUnaryOp op) {
    switch (op) {
    case ArithUnaryOp::Abs: return mask_abs_map;
    case ArithUnaryOp::Lzcnt: return mask_lzcnt_map;
    case ArithUnaryOp::Popcount: return mask_popcnt_map;
    case ArithUnaryOp::Not:
    case ArithUnaryOp::Negate:
    case ArithUnaryOp::Tzcnt:
    case ArithUnaryOp::RoundNearest:
    case ArithUnaryOp::RoundDown:
    case ArithUnaryOp::RoundUp:
    case ArithUnaryOp::RoundTruncate:
        x86_messed_up("unary op %s doesn't have direct intrinsic", show_arith_unary_op(op));
    case ArithUnaryOp::Rcp: return mask_rcp_map;
    case ArithUnaryOp::Sqrt: return mask_sqrt_map;
    case ArithUnaryOp::Rsqrt: return mask_rsqrt_map;
    }
    SIMJIT_UNREACHABLE();
}

const VecIntrinsicUnaryMap reduce_add_map{
    "reduce_add",

    IZMMI32(I32, "_mm512_mask_reduce_add_epi32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_add_epi64", M8, ZMMI64),
    IZMMF32(F32, "_mm512_mask_reduce_add_ps", M16, ZMMF32),
    IZMMF64(F64, "_mm512_mask_reduce_add_pd", M8, ZMMF64),
};

const VecIntrinsicUnaryMap reduce_mul_map{
    "reduce_mul",

    IZMMI32(I32, "_mm512_mask_reduce_mul_epi32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_mul_epi64", M8, ZMMI64),
    IZMMF32(F32, "_mm512_mask_reduce_mul_ps", M16, ZMMF32),
    IZMMF64(F64, "_mm512_mask_reduce_mul_pd", M8, ZMMF64),
};

const VecIntrinsicUnaryMap reduce_min_map{
    "reduce_min",

    IZMMI32(I32, "_mm512_mask_reduce_min_epi32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_min_epi64", M8, ZMMI64),
    IZMMF32(F32, "_mm512_mask_reduce_min_ps", M16, ZMMF32),
    IZMMF64(F64, "_mm512_mask_reduce_min_pd", M8, ZMMF64),
};

const VecIntrinsicUnaryMap reduce_max_map{
    "reduce_max",

    IZMMI32(I32, "_mm512_mask_reduce_max_epi32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_max_epi64", M8, ZMMI64),
    IZMMF32(F32, "_mm512_mask_reduce_max_ps", M16, ZMMF32),
    IZMMF64(F64, "_mm512_mask_reduce_max_pd", M8, ZMMF64),
};

const VecIntrinsicUnaryMap reduce_umin_map{
    "reduce_umin",

    IZMMI32(I32, "_mm512_mask_reduce_min_epu32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_min_epu64", M8, ZMMI64),
};

const VecIntrinsicUnaryMap reduce_umax_map{
    "reduce_umax",

    IZMMI32(I32, "_mm512_mask_reduce_max_epu32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_max_epu64", M8, ZMMI64),
};

const VecIntrinsicUnaryMap reduce_and_map{
    "reduce_and",

    IZMMI32(I32, "_mm512_mask_reduce_and_epi32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_and_epi64", M8, ZMMI64),
};

const VecIntrinsicUnaryMap reduce_or_map{
    "reduce_or",

    IZMMI32(I32, "_mm512_mask_reduce_or_epi32", M16, ZMMI32),
    IZMMI64(I64, "_mm512_mask_reduce_or_epi64", M8, ZMMI64),
};

const VecIntrinsicUnaryMap unmasked_reduce_add_map{
    "unmasked_reduce_add",

    IZMMI32(I32, "_mm512_reduce_add_epi32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_add_epi64", ZMMI64),
    IZMMF32(F32, "_mm512_reduce_add_ps", ZMMF32),
    IZMMF64(F64, "_mm512_reduce_add_pd", ZMMF64),
};

const VecIntrinsicUnaryMap unmasked_reduce_mul_map{
    "unmasked_reduce_mul",

    IZMMI32(I32, "_mm512_reduce_mul_epi32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_mul_epi64", ZMMI64),
    IZMMF32(F32, "_mm512_reduce_mul_ps", ZMMF32),
    IZMMF64(F64, "_mm512_reduce_mul_pd", ZMMF64),
};

const VecIntrinsicUnaryMap unmasked_reduce_min_map{
    "unmasked_reduce_min",

    IZMMI32(I32, "_mm512_reduce_min_epi32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_min_epi64", ZMMI64),
    IZMMF32(F32, "_mm512_reduce_min_ps", ZMMF32),
    IZMMF64(F64, "_mm512_reduce_min_pd", ZMMF64),
};

const VecIntrinsicUnaryMap unmasked_reduce_max_map{
    "unmasked_reduce_max",

    IZMMI32(I32, "_mm512_reduce_max_epi32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_max_epi64", ZMMI64),
    IZMMF32(F32, "_mm512_reduce_max_ps", ZMMF32),
    IZMMF64(F64, "_mm512_reduce_max_pd", ZMMF64),
};

const VecIntrinsicUnaryMap unmasked_reduce_umin_map{
    "unmasked_reduce_umin",

    IZMMI32(I32, "_mm512_reduce_min_epu32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_min_epu64", ZMMI64),
};

const VecIntrinsicUnaryMap unmasked_reduce_umax_map{
    "unmasked_reduce_umax",

    IZMMI32(I32, "_mm512_reduce_max_epu32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_max_epu64", ZMMI64),
};

const VecIntrinsicUnaryMap unmasked_reduce_and_map{
    "unmasked_reduce_and",

    IZMMI32(I32, "_mm512_reduce_and_epi32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_and_epi64", ZMMI64),
};

const VecIntrinsicUnaryMap unmasked_reduce_or_map{
    "unmasked_reduce_or",

    IZMMI32(I32, "_mm512_reduce_or_epi32", ZMMI32),
    IZMMI64(I64, "_mm512_reduce_or_epi64", ZMMI64),
};

const VecIntrinsicUnaryMap &reduce_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return reduce_add_map;
    case ArithBinaryOp::Mul: return reduce_mul_map;
    case ArithBinaryOp::Min: return reduce_min_map;
    case ArithBinaryOp::Max: return reduce_max_map;
    case ArithBinaryOp::UMin: return reduce_umin_map;
    case ArithBinaryOp::UMax: return reduce_umax_map;
    case ArithBinaryOp::And: return reduce_and_map;
    case ArithBinaryOp::Or: return reduce_or_map;
    case ArithBinaryOp::Sub:
    case ArithBinaryOp::Mul64SE:
    case ArithBinaryOp::Mul64ZE:
    case ArithBinaryOp::Div:
    case ArithBinaryOp::Mod:
    case ArithBinaryOp::UDiv:
    case ArithBinaryOp::UMod:
    case ArithBinaryOp::Xor:
    case ArithBinaryOp::AndNot:
    case ArithBinaryOp::ShiftRightArith:
    case ArithBinaryOp::ShiftRightLogical:
    case ArithBinaryOp::ShiftLeftLogical:
    case ArithBinaryOp::RotateLeft:
    case ArithBinaryOp::RotateRight: x86_messed_up("missing reduce intrinsic for %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const VecIntrinsicUnaryMap &unmasked_reduce_map(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return unmasked_reduce_add_map;
    case ArithBinaryOp::Mul: return unmasked_reduce_mul_map;
    case ArithBinaryOp::Min: return unmasked_reduce_min_map;
    case ArithBinaryOp::Max: return unmasked_reduce_max_map;
    case ArithBinaryOp::UMin: return unmasked_reduce_umin_map;
    case ArithBinaryOp::UMax: return unmasked_reduce_umax_map;
    case ArithBinaryOp::And: return unmasked_reduce_and_map;
    case ArithBinaryOp::Or: return unmasked_reduce_or_map;
    case ArithBinaryOp::Sub:
    case ArithBinaryOp::Mul64SE:
    case ArithBinaryOp::Mul64ZE:
    case ArithBinaryOp::Div:
    case ArithBinaryOp::Mod:
    case ArithBinaryOp::UDiv:
    case ArithBinaryOp::UMod:
    case ArithBinaryOp::Xor:
    case ArithBinaryOp::AndNot:
    case ArithBinaryOp::ShiftRightArith:
    case ArithBinaryOp::ShiftRightLogical:
    case ArithBinaryOp::ShiftLeftLogical:
    case ArithBinaryOp::RotateLeft:
    case ArithBinaryOp::RotateRight: x86_messed_up("missing reduce intrinsic for %s", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

const VecIntrinsicUnaryMap mov_mask_map{
    "mov_mask",

    IXMMI8(CMT, "_mm_movepi8_mask", CT),
    IXMMI16(CMT, "_mm_movepi16_mask", CT),
    IXMMI32(CMT, "_mm_movepi32_mask", CT),
    IXMMI64(CMT, "_mm_movepi64_mask", CT),

    IYMMI8(CMT, "_mm256_movepi8_mask", CT),
    IYMMI16(CMT, "_mm256_movepi16_mask", CT),
    IYMMI32(CMT, "_mm256_movepi32_mask", CT),
    IYMMI64(CMT, "_mm256_movepi64_mask", CT),

    IZMMI8(CMT, "_mm512_movepi8_mask", CT),
    IZMMI16(CMT, "_mm512_movepi16_mask", CT),
    IZMMI32(CMT, "_mm512_movepi32_mask", CT),
    IZMMI64(CMT, "_mm512_movepi64_mask", CT),
};

const VecIntrinsicUnaryMap movm_map{
    "movm",

    IXMMI8(CT, "_mm_movm_epi8", CMT),
    IXMMI16(CT, "_mm_movm_epi16", CMT),
    IXMMI32(CT, "_mm_movm_epi32", CMT),
    IXMMI64(CT, "_mm_movm_epi64", CMT),

    IYMMI8(CT, "_mm256_movm_epi8", CMT),
    IYMMI16(CT, "_mm256_movm_epi16", CMT),
    IYMMI32(CT, "_mm256_movm_epi32", CMT),
    IYMMI64(CT, "_mm256_movm_epi64", CMT),

    IZMMI8(CT, "_mm512_movm_epi8", CMT),
    IZMMI16(CT, "_mm512_movm_epi16", CMT),
    IZMMI32(CT, "_mm512_movm_epi32", CMT),
    IZMMI64(CT, "_mm512_movm_epi64", CMT),
};

const VecIntrinsicUnaryMap undefined_map{
    "undefined",

    IXMMI8{CT, "_mm_undefined_si128"},
    IXMMI16{CT, "_mm_undefined_si128"},
    IXMMI32{CT, "_mm_undefined_si128"},
    IXMMI64{CT, "_mm_undefined_si128"},
    IXMMF32{CT, "_mm_undefined_ps"},
    IXMMF64{CT, "_mm_undefined_pd"},

    IYMMI8{CT, "_mm256_undefined_si256"},
    IYMMI16{CT, "_mm256_undefined_si256"},
    IYMMI32{CT, "_mm256_undefined_si256"},
    IYMMI64{CT, "_mm256_undefined_si256"},
    IYMMF32{CT, "_mm256_undefined_ps"},
    IYMMF64{CT, "_mm256_undefined_pd"},

    IZMMI8{CT, "_mm512_undefined_epi32"},
    IZMMI16{CT, "_mm512_undefined_epi32"},
    IZMMI32{CT, "_mm512_undefined_epi32"},
    IZMMI64{CT, "_mm512_undefined_epi32"},
    IZMMF32{CT, "_mm512_undefined_ps"},
    IZMMF64{CT, "_mm512_undefined_pd"},
};

const VecIntrinsicUnaryMap gp2affine_map{
    "gp2affine",

    IXMMI8{CT, "_mm_gf2p8affine_epi64_epi8", CT, CT, I32},
    IXMMI16{CT, "_mm_gf2p8affine_epi64_epi8", CT, CT, I32},
    IXMMI32{CT, "_mm_gf2p8affine_epi64_epi8", CT, CT, I32},
    IXMMI64{CT, "_mm_gf2p8affine_epi64_epi8", CT, CT, I32},

    IYMMI8{CT, "_mm256_gf2p8affine_epi64_epi8", CT, CT, I32},
    IYMMI16{CT, "_mm256_gf2p8affine_epi64_epi8", CT, CT, I32},
    IYMMI32{CT, "_mm256_gf2p8affine_epi64_epi8", CT, CT, I32},
    IYMMI64{CT, "_mm256_gf2p8affine_epi64_epi8", CT, CT, I32},

    IZMMI8{CT, "_mm512_gf2p8affine_epi64_epi8", CT, CT, I32},
    IZMMI16{CT, "_mm512_gf2p8affine_epi64_epi8", CT, CT, I32},
    IZMMI32{CT, "_mm512_gf2p8affine_epi64_epi8", CT, CT, I32},
    IZMMI64{CT, "_mm512_gf2p8affine_epi64_epi8", CT, CT, I32},
};

const VecIntrinsicUnaryMap permb_map{
    "permb",

    IXMMI8{CT, "_mm_permutexvar_epi8", CT, CT},
    IXMMI16{CT, "_mm_permutexvar_epi8", CT, CT},
    IXMMI32{CT, "_mm_permutexvar_epi8", CT, CT},
    IXMMI64{CT, "_mm_permutexvar_epi8", CT, CT},

    IYMMI8{CT, "_mm256_permutexvar_epi8", CT, CT},
    IYMMI16{CT, "_mm256_permutexvar_epi8", CT, CT},
    IYMMI32{CT, "_mm256_permutexvar_epi8", CT, CT},
    IYMMI64{CT, "_mm256_permutexvar_epi8", CT, CT},

    IZMMI8{CT, "_mm512_permutexvar_epi8", CT, CT},
    IZMMI16{CT, "_mm512_permutexvar_epi8", CT, CT},
    IZMMI32{CT, "_mm512_permutexvar_epi8", CT, CT},
    IZMMI64{CT, "_mm512_permutexvar_epi8", CT, CT},
};

const VecIntrinsicUnaryMap shuffle8_map{
    "shuffle8",

    IXMMI8{CT, "_mm_shuffle_epi8", CT, CT},
    IXMMI16{CT, "_mm_shuffle_epi8", CT, CT},
    IXMMI32{CT, "_mm_shuffle_epi8", CT, CT},
    IXMMI64{CT, "_mm_shuffle_epi8", CT, CT},

    IYMMI8{CT, "_mm256_shuffle_epi8", CT, CT},
    IYMMI16{CT, "_mm256_shuffle_epi8", CT, CT},
    IYMMI32{CT, "_mm256_shuffle_epi8", CT, CT},
    IYMMI64{CT, "_mm256_shuffle_epi8", CT, CT},

    IZMMI8{CT, "_mm512_shuffle_epi8", CT, CT},
    IZMMI16{CT, "_mm512_shuffle_epi8", CT, CT},
    IZMMI32{CT, "_mm512_shuffle_epi8", CT, CT},
    IZMMI64{CT, "_mm512_shuffle_epi8", CT, CT},
};

const MaskIntrinsicUnaryMap load_mask_map{
    "load_mask",

    IM8(CT, "_load_mask8", CT),
    IM16(CT, "_load_mask16", CT),
    IM32(CT, "_load_mask32", CT),
    IM64(CT, "_load_mask64", CT),
};
const MaskIntrinsicUnaryMap store_mask_map{
    "store_mask",

    IM8(CT, "_store_mask8", CT, CT),
    IM16(CT, "_store_mask16", CT, CT),
    IM32(CT, "_store_mask32", CT, CT),
    IM64(CT, "_store_mask64", CT, CT),
};

const VecIntrinsicUnaryMap mask_storea_map{
    "mask_storea_map",
    IXMMI8(XMMI8, "_mm_mask_storeu_epi8", XMMI8, MaskDataType::M16, XMMI8),
    IXMMI16(XMMI16, "_mm_mask_storeu_epi16", XMMI16, MaskDataType::M8, XMMI16),
    IXMMI32(XMMI32, "_mm_mask_store_epi32", XMMI32, MaskDataType::M4, XMMI32),
    IXMMI64(XMMI64, "_mm_mask_store_epi64", XMMI64, MaskDataType::M2, XMMI64),
    IXMMF32(XMMF32, "_mm_mask_store_ps", XMMF32, MaskDataType::M4, XMMF32),
    IXMMF64(XMMF64, "_mm_mask_store_pd", XMMF64, MaskDataType::M2, XMMF64),
    IYMMI8(YMMI8, "_mm256_mask_storeu_epi8", YMMI8, MaskDataType::M32, YMMI8),
    IYMMI16(YMMI16, "_mm256_mask_storeu_epi16", YMMI16, MaskDataType::M16, YMMI16),
    IYMMI32(YMMI32, "_mm256_mask_store_epi32", YMMI32, MaskDataType::M8, YMMI32),
    IYMMI64(YMMI64, "_mm256_mask_store_epi64", YMMI64, MaskDataType::M4, YMMI64),
    IYMMF32(YMMF32, "_mm256_mask_store_ps", YMMF32, MaskDataType::M8, YMMF32),
    IYMMF64(YMMF64, "_mm256_mask_store_pd", YMMF64, MaskDataType::M4, YMMF64),
    IZMMI8(ZMMI8, "_mm512_mask_storeu_epi8", ScalarDataType::I8, MaskDataType::M64, ZMMI8),
    IZMMI16(ZMMI16, "_mm512_mask_storeu_epi16", ScalarDataType::I16, MaskDataType::M32, ZMMI16),
    IZMMI32(ZMMI32, "_mm512_mask_store_epi32", ScalarDataType::I32, MaskDataType::M16, ZMMI32),
    IZMMI64(ZMMI64, "_mm512_mask_store_epi64", ScalarDataType::I64, MaskDataType::M8, ZMMI64),
    IZMMF32(ZMMF32, "_mm512_mask_store_ps", ScalarDataType::F32, MaskDataType::M16, ZMMF32),
    IZMMF64(ZMMF64, "_mm512_mask_store_pd", ScalarDataType::F64, MaskDataType::M8, ZMMF64),
};

} // namespace x86
} // namespace simjit

// NOLINTEND(bugprone-throwing-static-initialization)
