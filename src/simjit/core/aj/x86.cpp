// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/x86.h"
#include "simjit/asmjit.h"
#include "simjit/core/expr.h"
#include "simjit/core/mir.h"
#include "simjit/simjit.h"

#include "asmjit/x86/x86compiler.h"
#include "asmjit/x86/x86globals.h"

#define messed_up(...) simjit_exception(ErrorModule::X86, {}, {}, __VA_ARGS__)
#define unsupported(...) \
    simjit_exception(ErrorModule::X86, ErrorKind::Unsupported, ErrorSubKind::UnsupportedBackendFeature, __VA_ARGS__)

#define INVALID_FLOAT_CASES \
    case VecElemType::F32:  \
    case VecElemType::F64: messed_up("Unexpected float")

// In asmjit backend we do a lot of small codegen transformations.
// Just in case I have to debug them some time later, add option to disable them selectively.
// I Do not want to make this user-facing arguments because they should all work out of the box.
// Macros also make it easy to spot these things in code.
#define SIMJIT_X64_ASMJIT_INLINE_MEM 1
#define SIMJIT_X64_ASMJIT_CONST_OPS 1
/* This one does not have performance impact, just makes code smaller */
#define SIMJIT_X64_ASMJIT_INLINE_ACC_INIT 1
/* This one does not have performance impact, just makes code smaller */
#define SIMJIT_X64_ASMJIT_INLINE_ACC_BIN 1
#define SIMJIT_X64_ASMJIT_INLINE_SCALAR_COND 1
#define SIMJIT_X64_ASMJIT_INLINE_VEC_CAST 1
/* Some comparisons with zero can be rewritten. This is not synonym of SIMJIT_X64_ASMJIT_CONST_OPS */
#define SIMJIT_X64_ASMJIT_ZERO_CMP 1
#define SIMJIT_X64_ASMJIT_ZEROBLEND 1
#define SIMJIT_X64_ASMJIT_MASK_PUSHDOWN 1
#define SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT 1
#define SIMJIT_X64_REPAIR_MINMAX 1

namespace simjit {
namespace asmjit_backend {

using namespace ::simjit::mir;

namespace aj = asmjit;
namespace ax86 = asmjit::x86;

using MaskR = ax86::KReg;
using GpR = ax86::Gp;
using VecR = ax86::Vec;
using AnyR = aj::Reg;

namespace {
struct ArgInfo {
    const ArgumentDecl *arg;
    GpR gp;
    ax86::Mem spilled{};
};

struct MaskPushdownInfo {
    MaskR mask{};
    VecR vec{};
    const Step *vec_step{};
};
} // namespace

static bool maybe_apply_mask_pushdown(ax86::Compiler &cc, const MaskPushdownInfo *info, const VecR &result) {
    if (info == nullptr) { return false; }
    if (info->vec.is_valid()) {
        SIMJIT_ASSERT(info->mask.is_valid());
        cc.vmovdqa64(result, info->vec);
        cc.k(info->mask);
        return true;
    }
    if (info->mask.is_valid()) {
        cc.k(info->mask).z();
        return true;
    }
    SIMJIT_ASSERT(0);
    return false;
}

// Used when mask vector is result of instruction, like cmp.
static bool maybe_apply_mask_pushdown(ax86::Compiler &cc, const MaskPushdownInfo *info) {
    if (info == nullptr) { return false; }
    SIMJIT_ASSERT(info->mask.is_valid());
    SIMJIT_ASSERT(!info->vec.is_valid());
    cc.k(info->mask);
    return true;
}

static bool same_vec_reg(const VecR &left, const VecR &right) noexcept {
    return left.is_valid() && right.is_valid() && left.id() == right.id();
}

static void scalar_mov(ax86::Compiler &cc, const AnyR &a, const AnyR &b, ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I8:
    case ScalarDataType::I16:
    case ScalarDataType::I32:
    case ScalarDataType::I64: cc.mov(a.as<GpR>(), b.as<GpR>()); break;
    // NOTE: Use packed move instructions because vmovss/vmovsd will actually do the same thing with three operand form
    case ScalarDataType::F32: cc.vmovaps(a.as<VecR>(), b.as<VecR>()); break;
    case ScalarDataType::F64: cc.vmovapd(a.as<VecR>(), b.as<VecR>()); break;
    case ScalarDataType::I1:
    case ScalarDataType::I128: messed_up("invalid scalar mov for type %s", show_scalar_dtype(dt));
    }
}
static void scalar_mov(ax86::Compiler &cc, const AnyR &a, const ax86::Mem &b, ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I8:
    case ScalarDataType::I16:
    case ScalarDataType::I32:
    case ScalarDataType::I64: cc.mov(a.as<GpR>(), b); break;
    case ScalarDataType::F32: cc.vmovss(a.as<VecR>(), b); break;
    case ScalarDataType::F64: cc.vmovsd(a.as<VecR>(), b); break;
    case ScalarDataType::I1:
    case ScalarDataType::I128: messed_up("invalid scalar mov for type %s", show_scalar_dtype(dt));
    }
}
static void scalar_mov(ax86::Compiler &cc, const ax86::Mem &a, const AnyR &b, ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I8:
    case ScalarDataType::I16:
    case ScalarDataType::I32:
    case ScalarDataType::I64: cc.mov(a, b.as<GpR>()); break;
    case ScalarDataType::F32: cc.vmovss(a, b.as<VecR>()); break;
    case ScalarDataType::F64: cc.vmovsd(a, b.as<VecR>()); break;
    case ScalarDataType::I1:
    case ScalarDataType::I128: messed_up("invalid scalar mov for type %s", show_scalar_dtype(dt));
    }
}

static void broadcast_f64(ax86::Compiler &cc, const VecR &dst, const ax86::Mem &src) {
    // VEX VBROADCASTSD has no XMM destination form. For two-lane F64 vectors
    // VMOVDDUP is the valid equivalent.
    if (dst.is_vec128()) {
        cc.vmovddup(dst, src);
    } else {
        cc.vbroadcastsd(dst, src);
    }
}

static aj::TypeId scalar_dtype_to_asmjit(ScalarDataType dtype) {
    switch (dtype) {
    case ScalarDataType::I1:
    case ScalarDataType::I8: return aj::TypeId::kInt8;
    case ScalarDataType::I16: return aj::TypeId::kInt16;
    case ScalarDataType::I32: return aj::TypeId::kInt32;
    case ScalarDataType::I64: return aj::TypeId::kInt64;
    case ScalarDataType::F32: return aj::TypeId::kFloat32;
    case ScalarDataType::F64: return aj::TypeId::kFloat64;
    case ScalarDataType::I128: messed_up("cannot map scalar type %s to asmjit", show_scalar_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

static aj::TypeId mask_dtype_to_asmjit(MaskDataType mdtype) noexcept {
    switch (mdtype) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: return aj::TypeId::kMask8;
    case MaskDataType::M16: return aj::TypeId::kMask16;
    case MaskDataType::M32: return aj::TypeId::kMask32;
    case MaskDataType::M64: return aj::TypeId::kMask64;
    }
    SIMJIT_UNREACHABLE();
}

static bool mask_uses_partial_byte(MaskDataType mdtype) noexcept {
    return mdtype == MaskDataType::M2 || mdtype == MaskDataType::M4;
}

static size_t mask_valid_bits_u32(MaskDataType mdtype) noexcept {
    SIMJIT_ASSERT(mask_dtype_bits(mdtype) < 32);
    return (size_t(1) << mask_dtype_bits(mdtype)) - 1;
}

static ax86::Mem mem_offset(const GpR &base, const GpR &i, ScalarDataType dtype, int32_t offset = 0) {
    switch (dtype) {
    case ScalarDataType::I1: messed_up("can't have i1 mem");
    case ScalarDataType::I8: return ax86::byte_ptr(base, i, 0, offset);
    case ScalarDataType::I16: return ax86::word_ptr(base, i, 1, offset);
    case ScalarDataType::I32:
    case ScalarDataType::F32: return ax86::dword_ptr(base, i, 2, offset);
    case ScalarDataType::I64:
    case ScalarDataType::F64: return ax86::qword_ptr(base, i, 3, offset);
    case ScalarDataType::I128: messed_up("can't have i128 mem");
    }
    SIMJIT_UNREACHABLE();
}

static ax86::Mem vec_mem_offset(const GpR &base, const GpR &i, size_t offset, VecDataType dtype) {
    x86::Vector vec = x86::vec_to_x86(dtype);
    size_t log2_size = dtype.element_size_bytes_log2();
    switch (vec.reg) {
    case x86::VecRegisterKind::XMM: return ax86::xmmword_ptr(base, i, log2_size, (int)offset); break;
    case x86::VecRegisterKind::YMM: return ax86::ymmword_ptr(base, i, log2_size, (int)offset); break;
    case x86::VecRegisterKind::ZMM: return ax86::zmmword_ptr(base, i, log2_size, (int)offset); break;
    }
    SIMJIT_UNREACHABLE();
}

template <typename T>
static void vec_unary(ax86::Compiler &cc, ArithUnaryOp op, VecDataType vdtype, const VecR &result, const T &arg) {
    bool is_f32 = vdtype.elem == VecElemType::F32;
    switch (op) {
    case ArithUnaryOp::Not:
    case ArithUnaryOp::Negate: messed_up("vector unary not and minus should have been rewritten");
    case ArithUnaryOp::Tzcnt: messed_up("vector tzcnt should have been rewritten");
    case ArithUnaryOp::Abs:
        switch (vdtype.elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: cc.vpabsb(result, arg); break;
        case VecElemType::I16: cc.vpabsw(result, arg); break;
        case VecElemType::I32: cc.vpabsd(result, arg); break;
        case VecElemType::I64: cc.vpabsq(result, arg); break;
        }
        break;
    case ArithUnaryOp::Lzcnt:
        switch (vdtype.elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16: unsupported("Do not support i8/i16 lzcnt");
        case VecElemType::I32: cc.vplzcntd(result, arg); break;
        case VecElemType::I64: cc.vplzcntq(result, arg); break;
        }
        break;
    case ArithUnaryOp::Popcount:
        switch (vdtype.elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: cc.vpopcntb(result, arg); break;
        case VecElemType::I16: cc.vpopcntw(result, arg); break;
        case VecElemType::I32: cc.vpopcntd(result, arg); break;
        case VecElemType::I64: cc.vpopcntq(result, arg); break;
        }
        break;
    case ArithUnaryOp::RoundNearest:
        if (is_f32)
            cc.vrndscaleps(result, arg, ax86::RoundImm::kNearest | ax86::RoundImm::kSuppress);
        else
            cc.vrndscalepd(result, arg, ax86::RoundImm::kNearest | ax86::RoundImm::kSuppress);
        break;
    case ArithUnaryOp::RoundDown:
        if (is_f32)
            cc.vrndscaleps(result, arg, ax86::RoundImm::kDown | ax86::RoundImm::kSuppress);
        else
            cc.vrndscalepd(result, arg, ax86::RoundImm::kDown | ax86::RoundImm::kSuppress);
        break;
    case ArithUnaryOp::RoundUp:
        if (is_f32)
            cc.vrndscaleps(result, arg, ax86::RoundImm::kUp | ax86::RoundImm::kSuppress);
        else
            cc.vrndscalepd(result, arg, ax86::RoundImm::kUp | ax86::RoundImm::kSuppress);
        break;
    case ArithUnaryOp::RoundTruncate:
        if (is_f32)
            cc.vrndscaleps(result, arg, ax86::RoundImm::kTrunc | ax86::RoundImm::kSuppress);
        else
            cc.vrndscalepd(result, arg, ax86::RoundImm::kTrunc | ax86::RoundImm::kSuppress);
        break;
    case ArithUnaryOp::Rcp:
        if (is_f32)
            cc.vrcp14ps(result, arg);
        else
            cc.vrcp14pd(result, arg);
        break;
    case ArithUnaryOp::Sqrt:
        if (is_f32)
            cc.vsqrtps(result, arg);
        else
            cc.vsqrtpd(result, arg);
        break;
    case ArithUnaryOp::Rsqrt:
        if (is_f32)
            cc.vrsqrt14ps(result, arg);
        else
            cc.vrsqrt14pd(result, arg);
        break;
    }
}

template <typename T>
static void vec_binary_dispatch(ax86::Compiler &cc, ArithBinaryOp op, VecElemType elem, const VecR &result,
                                const VecR &left, const T &right) {
    switch (op) {
    case ArithBinaryOp::Add:
        switch (elem) {
        case VecElemType::I8: cc.vpaddb(result, left, right); break;
        case VecElemType::I16: cc.vpaddw(result, left, right); break;
        case VecElemType::I32: cc.vpaddd(result, left, right); break;
        case VecElemType::I64: cc.vpaddq(result, left, right); break;
        case VecElemType::F32: cc.vaddps(result, left, right); break;
        case VecElemType::F64: cc.vaddpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Sub:
        switch (elem) {
        case VecElemType::I8: cc.vpsubb(result, left, right); break;
        case VecElemType::I16: cc.vpsubw(result, left, right); break;
        case VecElemType::I32: cc.vpsubd(result, left, right); break;
        case VecElemType::I64: cc.vpsubq(result, left, right); break;
        case VecElemType::F32: cc.vsubps(result, left, right); break;
        case VecElemType::F64: cc.vsubpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Mul64SE: cc.vpmuldq(result, left, right); break;
    case ArithBinaryOp::Mul64ZE: cc.vpmuludq(result, left, right); break;
    case ArithBinaryOp::Mul:
        switch (elem) {
        case VecElemType::I8: unsupported("Do not support i8 mul");
        case VecElemType::I16: cc.vpmullw(result, left, right); break;
        case VecElemType::I32: cc.vpmulld(result, left, right); break;
        case VecElemType::I64: cc.vpmullq(result, left, right); break;
        case VecElemType::F32: cc.vmulps(result, left, right); break;
        case VecElemType::F64: cc.vmulpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Div:
        switch (elem) {
        case VecElemType::I8:
        case VecElemType::I16:
        case VecElemType::I32:
        case VecElemType::I64: unsupported("Do not support int div");
        case VecElemType::F32: cc.vdivps(result, left, right); break;
        case VecElemType::F64: cc.vdivpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Min:
        switch (elem) {
        case VecElemType::I8: cc.vpminsb(result, left, right); break;
        case VecElemType::I16: cc.vpminsw(result, left, right); break;
        case VecElemType::I32: cc.vpminsd(result, left, right); break;
        case VecElemType::I64: cc.vpminsq(result, left, right); break;
        case VecElemType::F32: cc.vminps(result, left, right); break;
        case VecElemType::F64: cc.vminpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Max:
        switch (elem) {
        case VecElemType::I8: cc.vpmaxsb(result, left, right); break;
        case VecElemType::I16: cc.vpmaxsw(result, left, right); break;
        case VecElemType::I32: cc.vpmaxsd(result, left, right); break;
        case VecElemType::I64: cc.vpmaxsq(result, left, right); break;
        case VecElemType::F32: cc.vmaxps(result, left, right); break;
        case VecElemType::F64: cc.vmaxpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::UMin:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: cc.vpminub(result, left, right); break;
        case VecElemType::I16: cc.vpminuw(result, left, right); break;
        case VecElemType::I32: cc.vpminud(result, left, right); break;
        case VecElemType::I64: cc.vpminuq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::UMax:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: cc.vpmaxub(result, left, right); break;
        case VecElemType::I16: cc.vpmaxuw(result, left, right); break;
        case VecElemType::I32: cc.vpmaxud(result, left, right); break;
        case VecElemType::I64: cc.vpmaxuq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::And:
        switch (elem) {
        case VecElemType::I8:
        case VecElemType::I16:
            SIMJIT_ASSERT(!cc.has_extra_reg());
            cc.vpandq(result, left, right);
            break;
        case VecElemType::I32: cc.vpandd(result, left, right); break;
        case VecElemType::I64: cc.vpandq(result, left, right); break;
        case VecElemType::F32: cc.vandps(result, left, right); break;
        case VecElemType::F64: cc.vandpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Or:
        switch (elem) {
        case VecElemType::I8:
        case VecElemType::I16:
            SIMJIT_ASSERT(!cc.has_extra_reg());
            cc.vporq(result, left, right);
            break;
        case VecElemType::I32: cc.vpord(result, left, right); break;
        case VecElemType::I64: cc.vporq(result, left, right); break;
        case VecElemType::F32: cc.vorps(result, left, right); break;
        case VecElemType::F64: cc.vorpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::Xor:
        switch (elem) {
        case VecElemType::I8:
        case VecElemType::I16:
            SIMJIT_ASSERT(!cc.has_extra_reg());
            cc.vpxorq(result, left, right);
            break;
        case VecElemType::I32: cc.vpxord(result, left, right); break;
        case VecElemType::I64: cc.vpxorq(result, left, right); break;
        case VecElemType::F32: cc.vxorps(result, left, right); break;
        case VecElemType::F64: cc.vxorpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::AndNot:
        switch (elem) {
        case VecElemType::I8:
        case VecElemType::I16:
            SIMJIT_ASSERT(!cc.has_extra_reg());
            cc.vpandnq(result, left, right);
            break;
        case VecElemType::I32: cc.vpandnd(result, left, right); break;
        case VecElemType::I64: cc.vpandnq(result, left, right); break;
        case VecElemType::F32: cc.vandnps(result, left, right); break;
        case VecElemType::F64: cc.vandnpd(result, left, right); break;
        }
        break;
    case ArithBinaryOp::ShiftRightArith:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: unsupported("Do not support i8 sra");
        case VecElemType::I16: cc.vpsravw(result, left, right); break;
        case VecElemType::I32: cc.vpsravd(result, left, right); break;
        case VecElemType::I64: cc.vpsravq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::ShiftRightLogical:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: unsupported("Do not support i8 srl");
        case VecElemType::I16: cc.vpsrlvw(result, left, right); break;
        case VecElemType::I32: cc.vpsrlvd(result, left, right); break;
        case VecElemType::I64: cc.vpsrlvq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::ShiftLeftLogical:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: unsupported("Do not support i8 sll");
        case VecElemType::I16: cc.vpsllvw(result, left, right); break;
        case VecElemType::I32: cc.vpsllvd(result, left, right); break;
        case VecElemType::I64: cc.vpsllvq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::RotateLeft:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16: unsupported("Do not support i8/i16 rol");
        case VecElemType::I32: cc.vprolvd(result, left, right); break;
        case VecElemType::I64: cc.vprolvq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::RotateRight:
        switch (elem) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16: unsupported("Do not support i8/i16 ror");
        case VecElemType::I32: cc.vprorvd(result, left, right); break;
        case VecElemType::I64: cc.vprorvq(result, left, right); break;
        }
        break;
    case ArithBinaryOp::UDiv:
    case ArithBinaryOp::Mod:
    case ArithBinaryOp::UMod: unsupported("Do not support int div");
    }
}

static bool vec_binary_can_swap_mem_operand(VecDataType vdtype, ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add:
    case ArithBinaryOp::Mul:
    case ArithBinaryOp::And:
    case ArithBinaryOp::Or:
    case ArithBinaryOp::Xor: return true;
    case ArithBinaryOp::Min:
    case ArithBinaryOp::Max: {
        if (vdtype.is_float() && SIMJIT_X64_REPAIR_MINMAX) { return false; }
        return true;
    }
    case ArithBinaryOp::Mul64SE:
    case ArithBinaryOp::Mul64ZE:
    case ArithBinaryOp::UMin:
    case ArithBinaryOp::UMax: return vdtype.is_int();
    case ArithBinaryOp::Sub:
    case ArithBinaryOp::Div:
    case ArithBinaryOp::UDiv:
    case ArithBinaryOp::Mod:
    case ArithBinaryOp::UMod:
    case ArithBinaryOp::AndNot:
    case ArithBinaryOp::ShiftRightArith:
    case ArithBinaryOp::ShiftRightLogical:
    case ArithBinaryOp::ShiftLeftLogical:
    case ArithBinaryOp::RotateLeft:
    case ArithBinaryOp::RotateRight: return false;
    }
    SIMJIT_UNREACHABLE();
}

template <typename T>
static void vec_int_cmp(ax86::Compiler &cc, CmpOp op, VecDataType vdtype, bool is_unsigned, const MaskR &result,
                        const VecR &left, const T &right) {
    ax86::VPCmpImm cmp_imm{};
    switch (op) {
    case CmpOp::Less: cmp_imm = ax86::VPCmpImm::kLT; break;
    case CmpOp::Greater: cmp_imm = ax86::VPCmpImm::kGT; break;
    case CmpOp::LessEqual: cmp_imm = ax86::VPCmpImm::kLE; break;
    case CmpOp::GreaterEqual: cmp_imm = ax86::VPCmpImm::kGE; break;
    case CmpOp::Equal: cmp_imm = ax86::VPCmpImm::kEQ; break;
    case CmpOp::NotEqual: cmp_imm = ax86::VPCmpImm::kNE; break;
    }

    switch (vdtype.elem) {
        INVALID_FLOAT_CASES;
    case VecElemType::I8:
        if (is_unsigned) {
            cc.vpcmpub(result, left, right, cmp_imm);
        } else {
            cc.vpcmpb(result, left, right, cmp_imm);
        }
        break;
    case VecElemType::I16:
        if (is_unsigned) {
            cc.vpcmpuw(result, left, right, cmp_imm);
        } else {
            cc.vpcmpw(result, left, right, cmp_imm);
        }
        break;
    case VecElemType::I32:
        if (is_unsigned) {
            cc.vpcmpud(result, left, right, cmp_imm);
        } else {
            cc.vpcmpd(result, left, right, cmp_imm);
        }
        break;
    case VecElemType::I64:
        if (is_unsigned) {
            cc.vpcmpuq(result, left, right, cmp_imm);
        } else {
            cc.vpcmpq(result, left, right, cmp_imm);
        }
        break;
    }
}

static CmpOp swap_cmp_operands(CmpOp op) noexcept {
    switch (op) {
    case CmpOp::Less: return CmpOp::Greater;
    case CmpOp::Greater: return CmpOp::Less;
    case CmpOp::LessEqual: return CmpOp::GreaterEqual;
    case CmpOp::GreaterEqual: return CmpOp::LessEqual;
    case CmpOp::Equal: return CmpOp::Equal;
    case CmpOp::NotEqual: return CmpOp::NotEqual;
    }
    SIMJIT_UNREACHABLE();
}

template <typename T>
static void vec_float_cmp(ax86::Compiler &cc, CmpOp op, VecDataType vdtype, const MaskR &result, const VecR &left,
                          const T &right) {
    ax86::VCmpImm cmp_imm{};
    switch (op) {
    case CmpOp::Less: cmp_imm = ax86::VCmpImm::kLT_OQ; break;
    case CmpOp::Greater: cmp_imm = ax86::VCmpImm::kGT_OQ; break;
    case CmpOp::LessEqual: cmp_imm = ax86::VCmpImm::kLE_OQ; break;
    case CmpOp::GreaterEqual: cmp_imm = ax86::VCmpImm::kGE_OQ; break;
    case CmpOp::Equal: cmp_imm = ax86::VCmpImm::kEQ_OQ; break;
    case CmpOp::NotEqual: cmp_imm = ax86::VCmpImm::kNEQ_UQ; break;
    }

    switch (vdtype.elem) {
    case VecElemType::I8:
    case VecElemType::I16:
    case VecElemType::I32:
    case VecElemType::I64:
        messed_up("vector float compare requires float element type, got %s", show_vec_elem_type(vdtype.elem));
    case VecElemType::F32: cc.vcmpps(result, left, right, cmp_imm); break;
    case VecElemType::F64: cc.vcmppd(result, left, right, cmp_imm); break;
    }
}

template <typename T>
static void vec_sext(ax86::Compiler &cc, VecElemType from, VecElemType to, const VecR &dst, const T &arg) {
    switch (from) {
        INVALID_FLOAT_CASES;
    case VecElemType::I8:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
            messed_up("invalid vector sign-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I16: cc.vpmovsxbw(dst, arg); break;
        case VecElemType::I32: cc.vpmovsxbd(dst, arg); break;
        case VecElemType::I64:
            messed_up("invalid vector sign-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        }
        break;
    case VecElemType::I16:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16:
            messed_up("invalid vector sign-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I32: cc.vpmovsxwd(dst, arg); break;
        case VecElemType::I64: cc.vpmovsxwq(dst, arg); break;
        }
        break;
    case VecElemType::I32:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16:
        case VecElemType::I32:
            messed_up("invalid vector sign-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I64: cc.vpmovsxdq(dst, arg); break;
        }
        break;
    case VecElemType::I64:
        messed_up("invalid vector sign-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
    }
}

template <typename T>
static void vec_zext(ax86::Compiler &cc, VecElemType from, VecElemType to, const VecR &dst, const T &arg) {
    switch (from) {
        INVALID_FLOAT_CASES;
    case VecElemType::I8:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
            messed_up("invalid vector zero-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I16: cc.vpmovzxbw(dst, arg); break;
        case VecElemType::I32: cc.vpmovzxbd(dst, arg); break;
        case VecElemType::I64:
            messed_up("invalid vector zero-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        }
        break;
    case VecElemType::I16:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16:
            messed_up("invalid vector zero-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I32: cc.vpmovzxwd(dst, arg); break;
        case VecElemType::I64: cc.vpmovzxwq(dst, arg); break;
        }
        break;
    case VecElemType::I32:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16:
        case VecElemType::I32:
            messed_up("invalid vector zero-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I64: cc.vpmovzxdq(dst, arg); break;
        }
        break;
    case VecElemType::I64:
        messed_up("invalid vector zero-extension from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
    }
}

template <typename T>
static void vec_trunc(ax86::Compiler &cc, VecElemType from, VecElemType to, const T &dst, const VecR &arg) {
    switch (from) {
        INVALID_FLOAT_CASES;
    case VecElemType::I8:
        messed_up("invalid vector truncation from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
    case VecElemType::I16:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: cc.vpmovwb(dst, arg); break;
        case VecElemType::I16:
        case VecElemType::I32:
        case VecElemType::I64:
            messed_up("invalid vector truncation from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        }
        break;
    case VecElemType::I32:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: cc.vpmovdb(dst, arg); break;
        case VecElemType::I16: cc.vpmovdw(dst, arg); break;
        case VecElemType::I32:
        case VecElemType::I64:
            messed_up("invalid vector truncation from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        }
        break;
    case VecElemType::I64:
        switch (to) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
            messed_up("invalid vector truncation from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I16: cc.vpmovqw(dst, arg); break;
        case VecElemType::I32: cc.vpmovqd(dst, arg); break;
        case VecElemType::I64:
            messed_up("invalid vector truncation from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        }
        break;
    }
}

template <typename T>
static void vec_float_cast(ax86::Compiler &cc, VecElemType from, VecElemType to, bool is_unsigned, const VecR &dst,
                           const T &arg) {
    // when casting float -> int use round to zero mode. This is same as LLVM's fptosi behavior
    switch (from) {
    case VecElemType::I8:
    case VecElemType::I16:
        messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
    case VecElemType::I32:
        switch (to) {
        case VecElemType::I8:
        case VecElemType::I16:
        case VecElemType::I32:
        case VecElemType::I64:
            messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::F32:
            if (is_unsigned)
                cc.vcvtudq2ps(dst, arg);
            else
                cc.vcvtdq2ps(dst, arg);
            break;
        case VecElemType::F64:
            if (is_unsigned)
                cc.vcvtudq2pd(dst, arg);
            else
                cc.vcvtdq2pd(dst, arg);
            break;
        }
        break;
    case VecElemType::I64:
        switch (to) {
        case VecElemType::I8:
        case VecElemType::I16:
        case VecElemType::I32:
        case VecElemType::I64:
            messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::F32:
            if (is_unsigned)
                cc.vcvtuqq2ps(dst, arg);
            else
                cc.vcvtqq2ps(dst, arg);
            break;
        case VecElemType::F64:
            if (is_unsigned)
                cc.vcvtuqq2pd(dst, arg);
            else
                cc.vcvtqq2pd(dst, arg);
            break;
        }
        break;
    case VecElemType::F32:
        switch (to) {
        case VecElemType::I8:
        case VecElemType::I16:
            messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I32:
            if (is_unsigned)
                cc.vcvttps2udq(dst, arg);
            else
                cc.vcvttps2dq(dst, arg);
            break;
        case VecElemType::I64:
            if (is_unsigned)
                cc.vcvttps2uqq(dst, arg);
            else
                cc.vcvttps2qq(dst, arg);
            break;
        case VecElemType::F32:
            messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::F64: cc.vcvtps2pd(dst, arg); break;
        }
        break;
    case VecElemType::F64:
        switch (to) {
        case VecElemType::I8:
        case VecElemType::I16:
            messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        case VecElemType::I32:
            if (is_unsigned)
                cc.vcvttpd2udq(dst, arg);
            else
                cc.vcvttpd2dq(dst, arg);
            break;
        case VecElemType::I64:
            if (is_unsigned)
                cc.vcvttpd2uqq(dst, arg);
            else
                cc.vcvttpd2qq(dst, arg);
            break;
        case VecElemType::F32: cc.vcvtpd2ps(dst, arg); break;
        case VecElemType::F64:
            messed_up("invalid vector float cast from %s to %s", show_vec_elem_type(from), show_vec_elem_type(to));
        }
    }
}

template <typename T>
static void vec_shift_imm(ax86::Compiler &cc, ArithBinaryOp op, VecElemType type, const VecR &result, const T &left,
                          const aj::Imm &imm) {
    if (op == ArithBinaryOp::ShiftRightArith) {
        switch (type) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: unsupported("Do not support i8 sra");
        case VecElemType::I16: cc.vpsraw(result, left, imm); break;
        case VecElemType::I32: cc.vpsrad(result, left, imm); break;
        case VecElemType::I64: cc.vpsraq(result, left, imm); break;
        }
    } else if (op == ArithBinaryOp::ShiftRightLogical) {
        switch (type) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: unsupported("Do not support i8 srl");
        case VecElemType::I16: cc.vpsrlw(result, left, imm); break;
        case VecElemType::I32: cc.vpsrld(result, left, imm); break;
        case VecElemType::I64: cc.vpsrlq(result, left, imm); break;
        }
    } else if (op == ArithBinaryOp::ShiftLeftLogical) {
        switch (type) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8: unsupported("Do not support i8 sll");
        case VecElemType::I16: cc.vpsllw(result, left, imm); break;
        case VecElemType::I32: cc.vpslld(result, left, imm); break;
        case VecElemType::I64: cc.vpsllq(result, left, imm); break;
        }
    } else if (op == ArithBinaryOp::RotateLeft) {
        switch (type) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16: unsupported("Do not support i8/i16 rol");
        case VecElemType::I32: cc.vprold(result, left, imm); break;
        case VecElemType::I64: cc.vprolq(result, left, imm); break;
        }
    } else if (op == ArithBinaryOp::RotateRight) {
        switch (type) {
            INVALID_FLOAT_CASES;
        case VecElemType::I8:
        case VecElemType::I16: unsupported("Do not support i8/i16 ror");
        case VecElemType::I32: cc.vprord(result, left, imm); break;
        case VecElemType::I64: cc.vprorq(result, left, imm); break;
        }
    }
}

template <typename T>
static void vec_fma(ax86::Compiler &cc, FmaKind kind, VecElemType type, const VecR &x1, const VecR &x2, const T &x3) {
    bool is_f32 = type == VecElemType::F32;
    switch (kind) {
    case FmaKind::FMA:
        if (is_f32)
            cc.vfmadd213ps(x1, x2, x3);
        else
            cc.vfmadd213pd(x1, x2, x3);
        break;
    case FmaKind::FMS:
        if (is_f32)
            cc.vfmsub213ps(x1, x2, x3);
        else
            cc.vfmsub213pd(x1, x2, x3);
        break;
    case FmaKind::FNMA:
        if (is_f32)
            cc.vfnmadd213ps(x1, x2, x3);
        else
            cc.vfnmadd213pd(x1, x2, x3);
        break;
    case FmaKind::FNMS:
        if (is_f32)
            cc.vfnmsub213ps(x1, x2, x3);
        else
            cc.vfnmsub213pd(x1, x2, x3);
        break;
    }
}

static void scalar_fma(ax86::Compiler &cc, FmaKind kind, ScalarDataType type, const VecR &x1, const VecR &x2,
                       const VecR &x3) {
    bool is_f32 = type == ScalarDataType::F32;
    switch (kind) {
    case FmaKind::FMA:
        if (is_f32)
            cc.vfmadd213ss(x1, x2, x3);
        else
            cc.vfmadd213sd(x1, x2, x3);
        break;
    case FmaKind::FMS:
        if (is_f32)
            cc.vfmsub213ss(x1, x2, x3);
        else
            cc.vfmsub213sd(x1, x2, x3);
        break;
    case FmaKind::FNMA:
        if (is_f32)
            cc.vfnmadd213ss(x1, x2, x3);
        else
            cc.vfnmadd213sd(x1, x2, x3);
        break;
    case FmaKind::FNMS:
        if (is_f32)
            cc.vfnmsub213ss(x1, x2, x3);
        else
            cc.vfnmsub213sd(x1, x2, x3);
        break;
    }
}

template <typename T>
static void vec_fma_acc(ax86::Compiler &cc, FmaKind kind, VecElemType type, const VecR &acc, const VecR &x1,
                        const T &x2) {
    bool is_f32 = type == VecElemType::F32;
    switch (kind) {
    case FmaKind::FMA:
        if (is_f32)
            cc.vfmadd231ps(acc, x1, x2);
        else
            cc.vfmadd231pd(acc, x1, x2);
        break;
    case FmaKind::FMS:
        if (is_f32)
            cc.vfmsub231ps(acc, x1, x2);
        else
            cc.vfmsub231pd(acc, x1, x2);
        break;
    case FmaKind::FNMA:
        if (is_f32)
            cc.vfnmadd231ps(acc, x1, x2);
        else
            cc.vfnmadd231pd(acc, x1, x2);
        break;
    case FmaKind::FNMS:
        if (is_f32)
            cc.vfnmsub231ps(acc, x1, x2);
        else
            cc.vfnmsub231pd(acc, x1, x2);
        break;
    }
}

static ax86::VFPClassImm make_fpclass_imm(FpClass flags) noexcept {
    return (bool(flags & FpClass::FPC_INFINITE) ? (ax86::VFPClassImm::kNInf | ax86::VFPClassImm::kPInf)
                                                : ax86::VFPClassImm::kNone) |
           (bool(flags & FpClass::FPC_NAN) ? (ax86::VFPClassImm::kQNaN | ax86::VFPClassImm::kSNaN)
                                           : ax86::VFPClassImm::kNone) |
           (bool(flags & FpClass::FPC_SUBNORMAL) ? (ax86::VFPClassImm::kDenormal) : ax86::VFPClassImm::kNone) |
           (bool(flags & FpClass::FPC_ZERO) ? (ax86::VFPClassImm::kNZero | ax86::VFPClassImm::kPZero)
                                            : ax86::VFPClassImm::kNone);
}

static void load_mask(ax86::Compiler &cc, MaskDataType type, const MaskR &dst, const ax86::Mem &src) {
    switch (type) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: cc.kmovb(dst, src); break;
    case MaskDataType::M16: cc.kmovw(dst, src); break;
    case MaskDataType::M32: cc.kmovd(dst, src); break;
    case MaskDataType::M64: cc.kmovq(dst, src); break;
    }
}

static void store_mask(ax86::Compiler &cc, MaskDataType type, const ax86::Mem &dst, const MaskR &src) {
    switch (type) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: cc.kmovb(dst, src); break;
    case MaskDataType::M16: cc.kmovw(dst, src); break;
    case MaskDataType::M32: cc.kmovd(dst, src); break;
    case MaskDataType::M64: cc.kmovq(dst, src); break;
    }
}

static void copy_mask(ax86::Compiler &cc, MaskDataType type, const MaskR &dst, const MaskR &src) {
    switch (type) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: cc.kmovb(dst, src); break;
    case MaskDataType::M16: cc.kmovw(dst, src); break;
    case MaskDataType::M32: cc.kmovd(dst, src); break;
    case MaskDataType::M64: cc.kmovq(dst, src); break;
    }
}

static void mask_to_gp(ax86::Compiler &cc, MaskDataType type, const GpR &dst, const MaskR &src) {
    // TODO: Revisit this extra move. AsmJit RA currently mishandles a spilled mask used by kmov-to-GP: it can reload
    // the spill into k0 but leave the GP destination unchanged. Materializing through a mask temp gives RA a valid
    // reload-then-kmov shape, at the cost of a possible redundant kmov when the source was already in a mask register.
    MaskR tmp = cc.new_k(mask_dtype_to_asmjit(type));
    copy_mask(cc, type, tmp, src);

    // AsmJit treats narrow kmov-to-gp as preserving part of the old GPR value, which can make short-lived temps look
    // live across loops. Clear the virtual register first to break that false dependency.
    if (type != MaskDataType::M64) { cc.xor_(dst.r32(), dst.r32()); }
    switch (type) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: cc.kmovb(dst, tmp); break;
    case MaskDataType::M16: cc.kmovw(dst, tmp); break;
    case MaskDataType::M32: cc.kmovd(dst, tmp); break;
    case MaskDataType::M64: cc.kmovq(dst, tmp); break;
    }
}

static void mask_and(ax86::Compiler &cc, MaskDataType type, const MaskR &dst, const MaskR &left, const MaskR &right) {
    switch (type) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: cc.kandb(dst, left, right); break;
    case MaskDataType::M16: cc.kandw(dst, left, right); break;
    case MaskDataType::M32: cc.kandd(dst, left, right); break;
    case MaskDataType::M64: cc.kandq(dst, left, right); break;
    }
}

static void mask_zero(ax86::Compiler &cc, MaskDataType type, const MaskR &dst) {
    switch (type) {
    case MaskDataType::M2:
    case MaskDataType::M4:
    case MaskDataType::M8: cc.kxorb(dst, dst, dst); break;
    case MaskDataType::M16: cc.kxorw(dst, dst, dst); break;
    case MaskDataType::M32: cc.kxord(dst, dst, dst); break;
    case MaskDataType::M64: cc.kxorq(dst, dst, dst); break;
    }
}

static bool step_has_mask_form(const Step *step) noexcept {
    switch (step->kind) {
        SIMJIT_MATCH (StepKind::Load) return true;
        SIMJIT_MATCH (StepKind::Gather) return true;
        SIMJIT_MATCH (StepKind::ArithBinary) {
            if (data.op == ArithBinaryOp::And || data.op == ArithBinaryOp::Xor || data.op == ArithBinaryOp::Or ||
                data.op == ArithBinaryOp::AndNot) {
                VecDataType vdtype = step->dtype.as_vec();
                return vdtype.elem == VecElemType::I32 || vdtype.elem == VecElemType::I64;
            }
            return true;
        }
        SIMJIT_MATCH (StepKind::ArithUnary) return true;
        SIMJIT_MATCH (StepKind::IntCast) return true;
        SIMJIT_MATCH (StepKind::FloatCast) return true;
    default: break;
    }
    return false;
}

static void bump_refcount(uint16_t &refcount) noexcept {
    if (refcount != UINT16_MAX) { ++refcount; }
}

static bool is_shift_rotate(ArithBinaryOp op) noexcept {
    return op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftLeftLogical ||
           op == ArithBinaryOp::ShiftRightLogical || op == ArithBinaryOp::RotateLeft ||
           op == ArithBinaryOp::RotateRight;
}

static bool scalar_const_binary_can_fold(ScalarDataType sdtype, ArithBinaryOp op, const ConstData &data) noexcept {
    if (!is_simple_int_dtype(sdtype)) { return false; }
    if (!(op == ArithBinaryOp::Add || op == ArithBinaryOp::Sub || op == ArithBinaryOp::Mul ||
          op == ArithBinaryOp::And || op == ArithBinaryOp::Or || op == ArithBinaryOp::Xor || is_shift_rotate(op))) {
        return false;
    }
    return (int64_t)(int32_t)data.as_signed() == data.as_signed();
}
namespace {
struct CompileState {
    MemoryArena *arena;
    ax86::Compiler &cc;
    GpR counter{};
    GpR row_count{};
    ArenaArray<ArgInfo> args{};
    ArenaArray<AnyR> accs{};
    ArenaArray<AnyR> step_map{};
    ArenaArray<uint16_t> refcounts{};
    ArenaArray<uint16_t> folded_const_refcounts{};
    ArenaArray<uint8_t> main_loop_acc_uses{};
    ArenaArray<uint8_t> acc_initialized{};
    const mir::Function *mir_func = nullptr;
    bool has_main_loop = false;

    // This is only used for small shift amounts (until shift=6, dividing by 64)
    GpR shifted_counter[7]{};

    std::unordered_map<uint64_t, aj::Label> bit_permute_luts{};

    aj::Label remainder_label{};
    aj::Label end_label{};
    // Spill arguments that are only used in epilogue
    bool spill_epilogue_args = true;

    void clear_shifted_counters() {
        for (auto &c : shifted_counter) {
            c = {};
        }
    }

    GpR get_or_insert_shifted_counter(size_t shift) {
        if (shifted_counter[shift].is_valid()) { return shifted_counter[shift]; }
        GpR idx = cc.new_gp64();
        cc.mov(idx, counter);
        cc.shr(idx, shift);
        shifted_counter[shift] = idx;
        return idx;
    }

    ax86::Mem scalar_load_mem(const ArgInfo &arg, ScalarDataType dtype) const {
        return mem_offset(arg.gp, counter, dtype);
    }
    ax86::Mem vec_load_mem(const ArgumentAddress &addr, VecDataType dtype) const {
        return vec_mem_offset(args[addr.arg].gp, counter, addr.offset * dtype.element_size_bytes(), dtype);
    }

    bool step_supports_mask_pushdown(const Step *step) const noexcept {
        if (refcounts[step->id] > 1) return false;
        return step_has_mask_form(step);
    }

    void require_acc_initialized(AccId acc) const {
        size_t idx = mir_func->accs.index(acc);
        if (!acc_initialized[idx]) { messed_up("x86 codegen used accumulator %zu before initialization", idx); }
    }

    void mark_acc_initialized(AccId acc) noexcept { acc_initialized[mir_func->accs.index(acc)] = 1; }

    bool const_is_folded_root(const Step *step) const {
        if (!step->is(StepKind::Const)) return false;
        return refcounts[step->id] != 0 && refcounts[step->id] == folded_const_refcounts[step->id];
    }

    void record_folded_const_ref(Step *step) noexcept {
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::AccStore) {
                if (!SIMJIT_X64_ASMJIT_INLINE_ACC_INIT || !data.arg->is(StepKind::Const)) { return; }
                if (step->dtype.is_scalar()) {
                    if (is_simple_int_dtype(step->dtype.as_scalar())) {
                        bump_refcount(folded_const_refcounts[data.arg->id]);
                    }
                } else if (step->dtype.is_vec() || step->dtype.is_mask()) {
                    bump_refcount(folded_const_refcounts[data.arg->id]);
                }
                return;
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                if (!SIMJIT_X64_ASMJIT_CONST_OPS || !data.right->is(StepKind::Const)) { return; }
                if (step->dtype.is_vec()) {
                    if (is_shift_rotate(data.op)) { bump_refcount(folded_const_refcounts[data.right->id]); }
                } else if (step->dtype.is_scalar()) {
                    if (scalar_const_binary_can_fold(step->dtype.as_scalar(), data.op,
                                                     data.right->step_data<StepKind::Const>())) {
                        bump_refcount(folded_const_refcounts[data.right->id]);
                    }
                }
                return;
            }
            SIMJIT_MATCH (StepKind::Compare) {
                if (!SIMJIT_X64_ASMJIT_CONST_OPS || !data.right->is(StepKind::Const)) { return; }
                if (data.left->dtype.is_scalar() && is_simple_int_dtype(data.left->dtype.as_scalar())) {
                    bump_refcount(folded_const_refcounts[data.right->id]);
                } else if (SIMJIT_X64_ASMJIT_ZERO_CMP && data.left->dtype.is_vec() &&
                           data.left->dtype.as_vec().is_int() && step_is_zero(data.right) &&
                           (data.op == CmpOp::Less || data.op == CmpOp::Equal || data.op == CmpOp::NotEqual)) {
                    bump_refcount(folded_const_refcounts[data.right->id]);
                }
                return;
            }
            SIMJIT_MATCH (StepKind::Select) {
                if (!SIMJIT_X64_ASMJIT_ZEROBLEND || !step->dtype.is_vec() || !step_is_zero(data.falsy)) { return; }
                bump_refcount(folded_const_refcounts[data.falsy->id]);
                VecDataType vdtype = step->dtype.as_vec();
                if (vdtype.is_int() && data.truthy->is(StepKind::Const) &&
                    data.truthy->step_data<StepKind::Const>().as_unsigned() ==
                        scalar_dtype_umax(vec_elem_to_scalar(vdtype.elem))) {
                    bump_refcount(folded_const_refcounts[data.truthy->id]);
                }
                return;
            }
        default: return;
        }
    }

    void do_int_cmp(const GpR &left, const Step *right_step, CmpOp op) {
        if (right_step->is(StepKind::Load) && should_inline_mem(right_step)) {
            const auto &right_data = right_step->step_data<StepKind::Load>();
            ax86::Mem mem = scalar_load_mem(args[right_data.addr.arg], right_step->dtype.as_scalar());
            cc.cmp(left, mem);
        } else if (SIMJIT_X64_ASMJIT_CONST_OPS && right_step->is(StepKind::Const)) {
            int64_t right_data = right_step->step_data<StepKind::Const>().as_signed();
            if (right_data == 0 && op == CmpOp::Equal) {
                cc.test(left, left);
            } else if (right_step->dtype.as_scalar() != ScalarDataType::I64 ||
                       (int64_t)(int32_t)right_data == right_data) {
                // x86 encodes compare immediates as signed imm32. For r64 compares,
                // larger positive constants must be materialized in a register.
                aj::Imm imm = (int32_t)right_data;
                cc.cmp(left, imm);
            } else {
                SIMJIT_ASSERT(right_step->dtype == ScalarDataType::I64);
                GpR reg = cc.new_gp64();
                cc.movabs(reg, right_data);
                cc.cmp(left, reg);
            }
        } else {
            GpR right = int_subexpr(right_step);
            cc.cmp(left, right);
        }
    }

    void do_float_cmp(const MaskR &result, const VecR &left, const VecR &right, CmpOp op, ScalarDataType sdtype) {
        ax86::VCmpImm cmp_imm{};
        switch (op) {
        case CmpOp::Less: cmp_imm = ax86::VCmpImm::kLT_OQ; break;
        case CmpOp::Greater: cmp_imm = ax86::VCmpImm::kGT_OQ; break;
        case CmpOp::LessEqual: cmp_imm = ax86::VCmpImm::kLE_OQ; break;
        case CmpOp::GreaterEqual: cmp_imm = ax86::VCmpImm::kGE_OQ; break;
        case CmpOp::Equal: cmp_imm = ax86::VCmpImm::kEQ_OQ; break;
        case CmpOp::NotEqual: cmp_imm = ax86::VCmpImm::kNEQ_UQ; break;
        }

        switch (sdtype) {
        case ScalarDataType::F32: cc.vcmpss(result, left, right, cmp_imm); break;
        case ScalarDataType::F64: cc.vcmpsd(result, left, right, cmp_imm); break;
        default: messed_up("scalar float compare requires float type, got %s", show_scalar_dtype(sdtype));
        }
    }

    void flags2bool(const GpR &dst, CmpOp op, bool is_unsigned) {
        switch (op) {
        case CmpOp::Less:
            if (is_unsigned) {
                cc.setb(dst);
            } else {
                cc.setl(dst);
            }
            break;
        case CmpOp::Greater:
            if (is_unsigned) {
                cc.seta(dst);
            } else {
                cc.setg(dst);
            }
            break;
        case CmpOp::LessEqual:
            if (is_unsigned) {
                cc.setbe(dst);
            } else {
                cc.setle(dst);
            }
            break;
        case CmpOp::GreaterEqual:
            if (is_unsigned) {
                cc.setae(dst);
            } else {
                cc.setge(dst);
            }
            break;
        case CmpOp::Equal: cc.sete(dst); break;
        case CmpOp::NotEqual: cc.setne(dst); break;
        }
    }

    void flags2cmov(const GpR &dst, const GpR &right, CmpOp op, bool is_unsigned) {
        switch (op) {
        case CmpOp::Less:
            if (is_unsigned) {
                cc.cmovb(dst, right);
            } else {
                cc.cmovl(dst, right);
            }
            break;
        case CmpOp::Greater:
            if (is_unsigned) {
                cc.cmova(dst, right);
            } else {
                cc.cmovg(dst, right);
            }
            break;
        case CmpOp::LessEqual:
            if (is_unsigned) {
                cc.cmovbe(dst, right);
            } else {
                cc.cmovle(dst, right);
            }
            break;
        case CmpOp::GreaterEqual:
            if (is_unsigned) {
                cc.cmovae(dst, right);
            } else {
                cc.cmovge(dst, right);
            }
            break;
        case CmpOp::Equal: cc.cmove(dst, right); break;
        case CmpOp::NotEqual: cc.cmovne(dst, right); break;
        }
    }

    bool should_inline_mem(const Step *step) const noexcept {
        if (!SIMJIT_X64_ASMJIT_INLINE_MEM) return false;
        if (!step->is(StepKind::Load)) return false;
        if (refcounts[step->id] > 1) return false;
        return (step->dtype.is_scalar() &&
                (is_simple_int_dtype(step->dtype.as_scalar()) || is_float_dtype(step->dtype.as_scalar()))) ||
               step->dtype.is_vec();
    }

    void init_scalar_int_const(const GpR &result, ScalarDataType sdtype, int64_t data) {
        if (data == 0) {
            cc.xor_(result, result);
        } else if (sdtype == ScalarDataType::I64 && ((data > (int64_t)INT32_MAX) || (data < (int64_t)INT32_MIN))) {
            // regular mov can have 32-bit immediate, for 64-bit need movabs
            cc.movabs(result, data);
        } else {
            cc.mov(result, data);
        }
    }

    void init_vec_const(const VecR &reg, VecDataType vdtype, ConstData data) {
        if (vdtype.elem == VecElemType::F32) {
            if (data.is_zero()) {
                cc.vxorps(reg, reg, reg);
            } else {
                ax86::Mem mem = cc.new_float_const(aj::ConstPoolScope::kLocal, data.as_f32());
                cc.vbroadcastss(reg, mem);
            }
        } else if (vdtype.elem == VecElemType::F64) {
            if (data.is_zero()) {
                cc.vxorpd(reg, reg, reg);
            } else {
                ax86::Mem mem = cc.new_double_const(aj::ConstPoolScope::kLocal, data.as_f64());
                broadcast_f64(cc, reg, mem);
            }
        } else {
            if (data.is_zero()) {
                cc.vpxor(reg, reg, reg);
            } else if (data.as_signed() == -1) {
                cc.vpternlogq(reg, reg, reg, ax86::TLogImm::k1);
            } else {
                uint64_t bits = data.as_unsigned();
                ax86::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, &bits, vdtype.element_size_bytes());
                switch (vdtype.elem) {
                    INVALID_FLOAT_CASES;
                case VecElemType::I8: cc.vpbroadcastb(reg, mem); break;
                case VecElemType::I16: cc.vpbroadcastw(reg, mem); break;
                case VecElemType::I32: cc.vpbroadcastd(reg, mem); break;
                case VecElemType::I64: cc.vpbroadcastq(reg, mem); break;
                }
            }
        }
    }

    void init_mask_const(const MaskR &reg, MaskDataType mdtype, int64_t data) {
        if (data == 0) {
            mask_zero(cc, mdtype, reg);
        } else {
            SIMJIT_ASSERT(data == 1);
            if (mask_uses_partial_byte(mdtype)) {
                init_partial_byte_mask_const(reg, mdtype);
                return;
            }
            switch (mdtype) {
            case MaskDataType::M8: cc.kxnorb(reg, reg, reg); break;
            case MaskDataType::M2:
            case MaskDataType::M4: SIMJIT_UNREACHABLE();
            case MaskDataType::M16: cc.kxnorw(reg, reg, reg); break;
            case MaskDataType::M32: cc.kxnord(reg, reg, reg); break;
            case MaskDataType::M64: cc.kxnorq(reg, reg, reg); break;
            }
        }
    }

    void scalar_float_arith_binary(ScalarDataType sdtype, ArithBinaryOp op, const VecR &dst, const VecR &left,
                                   const Step *right_step, bool inline_mem) {
        bool is_f32 = sdtype == ScalarDataType::F32;
        if (inline_mem && should_inline_mem(right_step) &&
            (!SIMJIT_X64_REPAIR_MINMAX || (op != ArithBinaryOp::Min && op != ArithBinaryOp::Max))) {
            const auto &right_data = right_step->step_data<StepKind::Load>();
            ax86::Mem mem = scalar_load_mem(args[right_data.addr.arg], sdtype);
            switch (op) {
            case ArithBinaryOp::Add:
                if (is_f32)
                    cc.vaddss(dst, left, mem);
                else
                    cc.vaddsd(dst, left, mem);
                break;
            case ArithBinaryOp::Sub:
                if (is_f32)
                    cc.vsubss(dst, left, mem);
                else
                    cc.vsubsd(dst, left, mem);
                break;
            case ArithBinaryOp::Mul:
                if (is_f32)
                    cc.vmulss(dst, left, mem);
                else
                    cc.vmulsd(dst, left, mem);
                break;
            case ArithBinaryOp::Div:
                if (is_f32)
                    cc.vdivss(dst, left, mem);
                else
                    cc.vdivsd(dst, left, mem);
                break;
            case ArithBinaryOp::Min:
                if (is_f32)
                    cc.vminss(dst, left, mem);
                else
                    cc.vminsd(dst, left, mem);
                break;
            case ArithBinaryOp::Max:
                if (is_f32)
                    cc.vmaxss(dst, left, mem);
                else
                    cc.vmaxsd(dst, left, mem);
                break;
            case ArithBinaryOp::And:
                if (is_f32)
                    cc.vandps(dst, left, mem);
                else
                    cc.vandpd(dst, left, mem);
                break;
            case ArithBinaryOp::Or:
                if (is_f32)
                    cc.vorps(dst, left, mem);
                else
                    cc.vorpd(dst, left, mem);
                break;
            case ArithBinaryOp::Xor:
                if (is_f32)
                    cc.vxorps(dst, left, mem);
                else
                    cc.vxorpd(dst, left, mem);
                break;
            case ArithBinaryOp::AndNot:
                if (is_f32)
                    cc.vandnps(dst, left, mem);
                else
                    cc.vandnpd(dst, left, mem);
                break;
            default: messed_up("unsupported scalar float binary op %s in inline-mem path", show_arith_binary_op(op));
            }
            return;
        }

        VecR right = float_subexpr(right_step);
        switch (op) {
        case ArithBinaryOp::Add:
            if (is_f32)
                cc.vaddss(dst, left, right);
            else
                cc.vaddsd(dst, left, right);
            break;
        case ArithBinaryOp::Sub:
            if (is_f32)
                cc.vsubss(dst, left, right);
            else
                cc.vsubsd(dst, left, right);
            break;
        case ArithBinaryOp::Mul:
            if (is_f32)
                cc.vmulss(dst, left, right);
            else
                cc.vmulsd(dst, left, right);
            break;
        case ArithBinaryOp::Div:
            if (is_f32)
                cc.vdivss(dst, left, right);
            else
                cc.vdivsd(dst, left, right);
            break;
        case ArithBinaryOp::Min:
            if (is_f32)
                cc.vminss(dst, left, right);
            else
                cc.vminsd(dst, left, right);
            break;
        case ArithBinaryOp::Max:
            if (is_f32)
                cc.vmaxss(dst, left, right);
            else
                cc.vmaxsd(dst, left, right);
            break;
        case ArithBinaryOp::And:
            if (is_f32)
                cc.vandps(dst, left, right);
            else
                cc.vandpd(dst, left, right);
            break;
        case ArithBinaryOp::Or:
            if (is_f32)
                cc.vorps(dst, left, right);
            else
                cc.vorpd(dst, left, right);
            break;
        case ArithBinaryOp::Xor:
            if (is_f32)
                cc.vxorps(dst, left, right);
            else
                cc.vxorpd(dst, left, right);
            break;
        case ArithBinaryOp::AndNot:
            if (is_f32)
                cc.vandnps(dst, left, right);
            else
                cc.vandnpd(dst, left, right);
            break;
        default: messed_up("unsupported op %s for floats", show_arith_binary_op(op));
        }
        if (SIMJIT_X64_REPAIR_MINMAX && (op == ArithBinaryOp::Min || op == ArithBinaryOp::Max) &&
            is_float_dtype(sdtype)) {
            // Note that we break 'no avx512' rule here, but xmm codegen is so verbose for this, and this was not a
            // hard rule anyway
            MaskR m = cc.new_k8();
            if (sdtype == ScalarDataType::F32) {
                cc.vcmpss(m, right, right, ax86::CmpImm::kUNORD);
                cc.k(m).vmovaps(dst, left);
            } else {
                cc.vcmpsd(m, right, right, ax86::CmpImm::kUNORD);
                cc.k(m).vmovapd(dst, left);
            }
        }
    }

    void scalar_int_arith_binary(ScalarDataType sdtype, ArithBinaryOp op, const GpR &left, const Step *right_step,
                                 bool inline_mem) {
        if (inline_mem && should_inline_mem(right_step) &&
            (op == ArithBinaryOp::Add || op == ArithBinaryOp::Sub || op == ArithBinaryOp::Mul ||
             op == ArithBinaryOp::And || op == ArithBinaryOp::Or || op == ArithBinaryOp::Xor ||
             op == ArithBinaryOp::AndNot)) {
            const auto &right_data = right_step->step_data<StepKind::Load>();
            ax86::Mem mem = scalar_load_mem(args[right_data.addr.arg], sdtype);
            switch (op) {
            case ArithBinaryOp::Add: cc.add(left, mem); break;
            case ArithBinaryOp::Sub: cc.sub(left, mem); break;
            case ArithBinaryOp::Mul: cc.imul(left, mem); break;
            case ArithBinaryOp::And: cc.and_(left, mem); break;
            case ArithBinaryOp::Or: cc.or_(left, mem); break;
            case ArithBinaryOp::Xor: cc.xor_(left, mem); break;
            case ArithBinaryOp::AndNot: cc.andn(left, left, mem); break;
            default: messed_up("unsupported scalar int binary op %s in inline-mem path", show_arith_binary_op(op));
            }
            return;
        }
        if (SIMJIT_X64_ASMJIT_CONST_OPS && right_step->is(StepKind::Const) &&
            (op == ArithBinaryOp::Add || op == ArithBinaryOp::Sub || op == ArithBinaryOp::Mul ||
             op == ArithBinaryOp::And || op == ArithBinaryOp::Or || op == ArithBinaryOp::Xor ||
             op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftRightLogical ||
             op == ArithBinaryOp::ShiftLeftLogical || op == ArithBinaryOp::RotateLeft ||
             op == ArithBinaryOp::RotateRight)) {
            const auto &right_data = right_step->step_data<StepKind::Const>();
            // Only allow 32 bit operands
            if ((int64_t)(int32_t)right_data.as_signed() == right_data.as_signed()) {
                int64_t imm_value = right_data.as_signed();
                if ((sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) && is_shift_rotate(op)) {
                    imm_value &= (int64_t)scalar_dtype_bits(sdtype) - 1;
                }
                aj::Imm imm = imm_value;
                switch (op) {
                case ArithBinaryOp::Add: cc.add(left, imm); break;
                case ArithBinaryOp::Sub: cc.sub(left, imm); break;
                case ArithBinaryOp::Mul: cc.imul(left, imm); break;
                case ArithBinaryOp::And: cc.and_(left, imm); break;
                case ArithBinaryOp::Or: cc.or_(left, imm); break;
                case ArithBinaryOp::Xor: cc.xor_(left, imm); break;
                case ArithBinaryOp::ShiftRightArith: cc.sar(left, imm); break;
                case ArithBinaryOp::ShiftRightLogical: cc.shr(left, imm); break;
                case ArithBinaryOp::ShiftLeftLogical: cc.sal(left, imm); break;
                case ArithBinaryOp::RotateLeft: cc.rol(left, imm); break;
                case ArithBinaryOp::RotateRight: cc.ror(left, imm); break;
                default: SIMJIT_UNREACHABLE(); break;
                }
                return;
            }
        }
        GpR right = int_subexpr(right_step);
        if ((sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) && is_shift_rotate(op)) {
            GpR masked = cc.new_gp32();
            cc.mov(masked, right.r32());
            cc.and_(masked, scalar_dtype_bits(sdtype) - 1);
            right = masked;
        }
        switch (op) {
        case ArithBinaryOp::Add: cc.add(left, right); break;
        case ArithBinaryOp::Sub: cc.sub(left, right); break;
        case ArithBinaryOp::Mul:
        case ArithBinaryOp::Mul64SE:
        case ArithBinaryOp::Mul64ZE: cc.imul(left, right); break;
        case ArithBinaryOp::Div: {
            GpR tmp = create_int_reg(sdtype);
            // Asmjit uses unconventional syntax for idiv instruction (and cwd for that matter)
            // syntax is: idiv <high> <low> <divisor>
            // usually <high> is dx, <low> is ax and divisor can be arbitrary.
            // Since we don't do division of double length (i32/i16), we can sign-extend ax into dx:ax.
            if (sdtype == ScalarDataType::I32)
                cc.cdq(tmp, left);
            else if (sdtype == ScalarDataType::I64)
                cc.cqo(tmp, left);
            else
                messed_up("scalar signed division only supports i32/i64, got %s", show_scalar_dtype(sdtype));
            cc.idiv(tmp, left, right);
            break;
        }
        case ArithBinaryOp::UDiv: {
            GpR tmp = create_int_reg(sdtype);
            // Here syntax is same with idiv (see comment above), but we need to zero-extend, which is easier.
            cc.xor_(tmp, tmp);
            cc.div(tmp, left, right);
            break;
        }
        case ArithBinaryOp::Mod: {
            GpR tmp = create_int_reg(sdtype);
            if (sdtype == ScalarDataType::I32)
                cc.cdq(tmp, left);
            else if (sdtype == ScalarDataType::I64)
                cc.cqo(tmp, left);
            else
                messed_up("scalar signed modulo only supports i32/i64, got %s", show_scalar_dtype(sdtype));
            cc.idiv(tmp, left, right);
            cc.mov(left, tmp);
            break;
        }
        case ArithBinaryOp::UMod: {
            GpR tmp = create_int_reg(sdtype);
            cc.xor_(tmp, tmp);
            cc.div(tmp, left, right);
            cc.mov(left, tmp);
            break;
        }
        case ArithBinaryOp::Min:
            cc.cmp(left, right);
            cc.cmovg(left, right);
            break;
        case ArithBinaryOp::Max:
            cc.cmp(left, right);
            cc.cmovl(left, right);
            break;
        case ArithBinaryOp::UMin:
            cc.cmp(left, right);
            cc.cmova(left, right);
            break;
        case ArithBinaryOp::UMax:
            cc.cmp(left, right);
            cc.cmovb(left, right);
            break;
        case ArithBinaryOp::And: cc.and_(left, right); break;
        case ArithBinaryOp::Or: cc.or_(left, right); break;
        case ArithBinaryOp::Xor: cc.xor_(left, right); break;
        case ArithBinaryOp::AndNot: cc.andn(left, left, right); break;
        case ArithBinaryOp::ShiftRightArith:
            // BMI2 variable shifts only support 32/64-bit operands. Keep i8/i16 on legacy shifts.
            if (sdtype == ScalarDataType::I32 || sdtype == ScalarDataType::I64)
                cc.sarx(left, left, right);
            else
                cc.sar(left, right);
            break;
        case ArithBinaryOp::ShiftRightLogical:
            // BMI2 variable shifts only support 32/64-bit operands. Keep i8/i16 on legacy shifts.
            if (sdtype == ScalarDataType::I32 || sdtype == ScalarDataType::I64)
                cc.shrx(left, left, right);
            else
                cc.shr(left, right);
            break;
        case ArithBinaryOp::ShiftLeftLogical:
            // BMI2 variable shifts only support 32/64-bit operands. Keep i8/i16 on legacy shifts.
            if (sdtype == ScalarDataType::I32 || sdtype == ScalarDataType::I64)
                cc.shlx(left, left, right);
            else
                cc.sal(left, right);
            break;
        case ArithBinaryOp::RotateLeft: cc.rol(left, right); break;
        case ArithBinaryOp::RotateRight: cc.ror(left, right); break;
        }
    }

    GpR scalar_const_div(const ConstDivData &data, ScalarDataType sdtype) {
        bool is_signed = data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::Mod;
        bool is_mod = data.op == ArithBinaryOp::Mod || data.op == ArithBinaryOp::UMod;
        size_t bits = scalar_dtype_bits(sdtype);
        GpR numerator = int_subexpr(data.numerator);
        GpR quotient = create_int_reg(sdtype);

        if (!data.has_magic) {
            cc.mov(quotient, numerator);
            if (is_signed) {
                GpR tweak = create_int_reg(sdtype);
                GpR mask = int_subexpr(data.round_mask);
                cc.mov(tweak, numerator);
                cc.sar(tweak, bits - 1);
                cc.and_(tweak, mask);
                cc.add(quotient, tweak);
                if (data.shift != 0) { cc.sar(quotient, data.shift); }
                if (data.negative_divisor) { cc.neg(quotient); }
            } else if (data.shift != 0) {
                cc.shr(quotient, data.shift);
            }
        } else {
            GpR low = create_int_reg(sdtype);
            GpR high = create_int_reg(sdtype);
            GpR magic = int_subexpr(data.magic);
            if (is_signed) {
                cc.mov(low, numerator);
                cc.imul(high, low, magic);
            } else {
                GpR mulx_src = create_int_reg(sdtype);
                cc.mov(mulx_src, numerator);
                cc.mulx(high, low, magic, mulx_src);
            }
            cc.mov(quotient, high);

            if (data.has_add) {
                if (is_signed) {
                    if (data.negative_divisor) {
                        cc.sub(quotient, numerator);
                    } else {
                        cc.add(quotient, numerator);
                    }
                } else {
                    GpR tmp = create_int_reg(sdtype);
                    cc.mov(tmp, numerator);
                    cc.sub(tmp, quotient);
                    cc.shr(tmp, 1);
                    cc.add(quotient, tmp);
                }
            }
            if (data.shift != 0) {
                if (is_signed) {
                    cc.sar(quotient, data.shift);
                } else {
                    cc.shr(quotient, data.shift);
                }
            }
            if (is_signed) {
                GpR sign = create_int_reg(sdtype);
                cc.mov(sign, quotient);
                cc.shr(sign, bits - 1);
                cc.add(quotient, sign);
            }
        }

        if (!is_mod) { return quotient; }

        GpR product = create_int_reg(sdtype);
        GpR divisor = int_subexpr(data.divisor);
        cc.mov(product, quotient);
        cc.imul(product, divisor);
        GpR remainder = create_int_reg(sdtype);
        cc.mov(remainder, numerator);
        cc.sub(remainder, product);
        return remainder;
    }

    void vec_binary(ArithBinaryOp op, VecDataType vdtype, const VecR &result, const VecR &left, const VecR &right,
                    const MaskPushdownInfo *mask_pushdown = nullptr, bool mask_merge_is_left = false) {
        bool repair_minmax =
            SIMJIT_X64_REPAIR_MINMAX && vdtype.is_float() && (op == ArithBinaryOp::Min || op == ArithBinaryOp::Max);
        VecR stable_right = right;
        if (repair_minmax && same_vec_reg(result, right) && !same_vec_reg(left, right)) {
            stable_right = create_vec_reg(vdtype);
            if (vdtype.elem == VecElemType::F32)
                cc.vmovaps(stable_right, right);
            else
                cc.vmovapd(stable_right, right);
        }
        bool fold_repair = repair_minmax && (mask_pushdown == nullptr || mask_merge_is_left);
        if (fold_repair) {
            MaskR ordered = create_mask_reg(vdtype.mask());
            if (mask_pushdown != nullptr) { cc.k(mask_pushdown->mask); }
            if (vdtype.elem == VecElemType::F32) {
                cc.vcmpps(ordered, stable_right, stable_right, ax86::CmpImm::kORD);
                if (!same_vec_reg(result, left)) { cc.vmovaps(result, left); }
            } else {
                cc.vcmppd(ordered, stable_right, stable_right, ax86::CmpImm::kORD);
                if (!same_vec_reg(result, left)) { cc.vmovapd(result, left); }
            }
            cc.k(ordered);
            vec_binary_dispatch(cc, op, vdtype.elem, result, left, stable_right);
            return;
        }

        vec_binary_dispatch(cc, op, vdtype.elem, result, left, stable_right);
        if (repair_minmax) {
            MaskR m = create_mask_reg(vdtype.mask());
            if (vdtype.elem == VecElemType::F32) {
                if (mask_pushdown != nullptr) { cc.k(mask_pushdown->mask); }
                cc.vcmpps(m, stable_right, stable_right, ax86::CmpImm::kUNORD);
                cc.k(m).vmovaps(result, left);
            } else {
                if (mask_pushdown != nullptr) { cc.k(mask_pushdown->mask); }
                cc.vcmppd(m, stable_right, stable_right, ax86::CmpImm::kUNORD);
                cc.k(m).vmovapd(result, left);
            }
        }
    }

    void vec_binary(ArithBinaryOp op, VecDataType vdtype, const VecR &result, const VecR &left,
                    const ax86::Mem &right) {
        vec_binary_dispatch(cc, op, vdtype.elem, result, left, right);
        if (SIMJIT_X64_REPAIR_MINMAX && vdtype.is_float() && (op == ArithBinaryOp::Min || op == ArithBinaryOp::Max)) {
            SIMJIT_ASSERT(0);
        }
    }

    template <typename T>
    void scalar_cast(bool is_sext, ScalarDataType from, ScalarDataType to, const GpR &result, const T &arg) {
        if (is_sext) {
            if (from == ScalarDataType::I32 && to == ScalarDataType::I64) {
                cc.movsxd(result, arg);
            } else {
                cc.movsx(result, arg);
            }
        } else {
            // rex prefix zeroes upper part of quadword register with doubleword operand
            if (from == ScalarDataType::I32 && to == ScalarDataType::I64) {
                cc.mov(result.r32(), arg);
            } else {
                cc.movzx(result, arg);
            }
        }
    }

    void vec_arith_binary(VecDataType vdtype, ArithBinaryOp op, const VecR &result, const Step *left_step,
                          const Step *right_step, bool inline_mem, const MaskPushdownInfo *mask_pushdown) {
        if (SIMJIT_X64_ASMJIT_CONST_OPS && should_inline_mem(left_step) && right_step->is(StepKind::Const) &&
            (op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftLeftLogical ||
             op == ArithBinaryOp::ShiftRightLogical || op == ArithBinaryOp::RotateLeft ||
             op == ArithBinaryOp::RotateRight)) {
            const auto &right_data = left_step->step_data<StepKind::Load>();
            ax86::Mem mem = vec_load_mem(right_data.addr, vdtype);
            aj::Imm imm = right_step->step_data<StepKind::Const>().as_signed();
            maybe_apply_mask_pushdown(cc, mask_pushdown, result);
            vec_shift_imm(cc, op, vdtype.elem, result, mem, imm);
            return;
        }

        if (inline_mem && left_step->is(StepKind::Load) && should_inline_mem(left_step) &&
            !(right_step->is(StepKind::Load) && should_inline_mem(right_step)) &&
            vec_binary_can_swap_mem_operand(vdtype, op)) {
            SIMJIT_ASSERT(!(vdtype.is_float() && SIMJIT_X64_REPAIR_MINMAX) ||
                          (op != ArithBinaryOp::Min && op != ArithBinaryOp::Max));
            const auto &left_data = left_step->step_data<StepKind::Load>();
            ax86::Mem mem = vec_load_mem(left_data.addr, vdtype);
            VecR right = vec_subexpr(right_step);
            maybe_apply_mask_pushdown(cc, mask_pushdown, result);
            vec_binary(op, vdtype, result, right, mem);
            return;
        }

        VecR left = vec_subexpr(left_step);
        if (SIMJIT_X64_ASMJIT_CONST_OPS && right_step->is(StepKind::Const) &&
            (op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftLeftLogical ||
             op == ArithBinaryOp::ShiftRightLogical || op == ArithBinaryOp::RotateLeft ||
             op == ArithBinaryOp::RotateRight)) {
            aj::Imm imm = right_step->step_data<StepKind::Const>().as_signed();
            maybe_apply_mask_pushdown(cc, mask_pushdown, result);
            vec_shift_imm(cc, op, vdtype.elem, result, left, imm);
            return;
        }

        if (right_step->is(StepKind::Load) && inline_mem && should_inline_mem(right_step) &&
            (!vdtype.is_float() || !SIMJIT_X64_REPAIR_MINMAX ||
             (op != ArithBinaryOp::Min && op != ArithBinaryOp::Max))) {
            const auto &right_data = right_step->step_data<StepKind::Load>();
            ax86::Mem mem = vec_load_mem(right_data.addr, vdtype);
            maybe_apply_mask_pushdown(cc, mask_pushdown, result);
            vec_binary(op, vdtype, result, left, mem);
            return;
        }

        VecR right = vec_subexpr(right_step);
        bool fold_repaired_minmax =
            SIMJIT_X64_REPAIR_MINMAX && vdtype.is_float() && (op == ArithBinaryOp::Min || op == ArithBinaryOp::Max) &&
            (mask_pushdown == nullptr || (mask_pushdown->vec.is_valid() && mask_pushdown->vec_step == left_step));
        if (!fold_repaired_minmax) { maybe_apply_mask_pushdown(cc, mask_pushdown, result); }
        vec_binary(op, vdtype, result, left, right, mask_pushdown, fold_repaired_minmax);
    }

    AnyR scalar_subexpr(const Step *step) {
        ScalarDataType sdtype = step->dtype.as_scalar();
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::Const) {
                AnyR result = create_scalar_reg(sdtype);
                if (sdtype == ScalarDataType::F32) {
                    VecR x = result.as<VecR>();
                    if (data.is_zero()) {
                        cc.vxorps(x, x, x);
                    } else {
                        ax86::Mem mem = cc.new_float_const(aj::ConstPoolScope::kLocal, data.as_f32());
                        cc.vmovss(x, mem);
                    }
                } else if (sdtype == ScalarDataType::F64) {
                    VecR x = result.as<VecR>();
                    if (data.is_zero()) {
                        cc.vxorpd(x, x, x);
                    } else {
                        ax86::Mem mem = cc.new_double_const(aj::ConstPoolScope::kLocal, data.as_f64());
                        cc.vmovsd(x, mem);
                    }
                } else {
                    init_scalar_int_const(result.as<GpR>(), sdtype, data.as_signed());
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Load) {
                const ArgInfo &arg = args[data.addr.arg];
                if (sdtype == ScalarDataType::I1) {
                    GpR result = cc.new_gp64();
                    GpR idx = get_or_insert_shifted_counter(6);
                    cc.mov(result, mem_offset(arg.gp, idx, ScalarDataType::I64));
                    cc.bt(result, counter);
                    cc.setc(result.r8());
                    return result.r8();
                }

                ax86::Mem mem = scalar_load_mem(arg, sdtype);
                AnyR result = create_scalar_reg(sdtype);
                scalar_mov(cc, result, mem, sdtype);
                return result;
            }
            SIMJIT_MATCH (StepKind::LoadSplat) {
                const ArgInfo &arg = args[data.addr.arg];
                AnyR result = create_scalar_reg(sdtype);
                ax86::Mem mem{};
                switch (sdtype) {
                case ScalarDataType::I128: messed_up("can't do i128 LoadSplat");
                case ScalarDataType::I1: mem = asmjit::x86::byte_ptr(arg.gp); break;
                case ScalarDataType::I8: mem = asmjit::x86::byte_ptr(arg.gp); break;
                case ScalarDataType::I16: mem = asmjit::x86::word_ptr(arg.gp); break;
                case ScalarDataType::I32:
                case ScalarDataType::F32: mem = asmjit::x86::dword_ptr(arg.gp); break;
                case ScalarDataType::I64:
                case ScalarDataType::F64: mem = asmjit::x86::qword_ptr(arg.gp); break;
                }
                scalar_mov(cc, result, mem, sdtype == ScalarDataType::I1 ? ScalarDataType::I8 : sdtype);
                if (sdtype == ScalarDataType::I1) { cc.and_(result.as<GpR>(), 1); }
                return result;
            }
            SIMJIT_MATCH (StepKind::Gather) {
                const ArgInfo &arg = args[data.data];
                GpR idx = int_subexpr(data.idx);
                AnyR result = create_scalar_reg(sdtype);
                ax86::Mem mem = mem_offset(arg.gp, idx, sdtype);
                scalar_mov(cc, result, mem, sdtype);
                return result;
            }
            SIMJIT_MATCH (StepKind::Store) {
                const ArgInfo &arg = args[data.addr.arg];
                if (sdtype == ScalarDataType::I1) {
                    GpR what = int_subexpr(data.what);
                    GpR idx = get_or_insert_shifted_counter(6);
                    GpR x = cc.new_gp64();
                    cc.mov(x, mem_offset(arg.gp, idx, ScalarDataType::I64));
                    cc.btr(x, counter);
                    GpR in = cc.new_gp64();
                    cc.xor_(in, in);
                    cc.mov(in.r8(), what);
                    cc.shlx(in, in, counter.r64());
                    cc.or_(x, in);
                    cc.mov(mem_offset(arg.gp, idx, ScalarDataType::I64), x);
                    return {};
                }
                AnyR what = subexpr(data.what);
                ax86::Mem mem = mem_offset(arg.gp, counter, sdtype);
                scalar_mov(cc, mem, what, sdtype);
                return {};
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                if (is_float_dtype(sdtype)) {
                    VecR left = float_subexpr(data.left);
                    VecR result = cc.new_vec128_f32x1();
                    scalar_float_arith_binary(sdtype, data.op, result, left, data.right, data.right != data.left);
                    return result;
                }
                GpR left = int_subexpr(data.left);
                GpR result = create_int_reg(sdtype);
                cc.mov(result, left);
                scalar_int_arith_binary(sdtype, data.op, result, data.right, data.right != data.left);

                return result;
            }
            SIMJIT_MATCH (StepKind::FMA) {
                VecR x1 = float_subexpr(data.x1);
                VecR x2 = float_subexpr(data.x2);
                VecR x3 = float_subexpr(data.x3);
                VecR result = cc.new_vec128_f32x1();
                if (sdtype == ScalarDataType::F32)
                    cc.vmovaps(result, x1);
                else
                    cc.vmovapd(result, x1);
                scalar_fma(cc, data.kind, sdtype, result, x2, x3);
                return result;
            }
            SIMJIT_MATCH (StepKind::ConstDiv) { return scalar_const_div(data, sdtype); }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                if (is_simple_int_dtype(sdtype)) {
                    GpR arg = int_subexpr(data.arg);
                    GpR result = create_int_reg(sdtype);
                    cc.mov(result, arg);

                    switch (data.op) {
                    case ArithUnaryOp::Not: cc.not_(result); break;
                    case ArithUnaryOp::Negate: cc.neg(result); break;
                    case ArithUnaryOp::Abs:
                        cc.cmp(arg, 0);
                        cc.neg(result);
                        cc.cmov(ax86::CondCode::kSignedLT, result, arg);
                        break;
                    case ArithUnaryOp::Lzcnt:
                        // X86 has no 8-bit LZCNT form. Use a 32-bit operation for small scalars and remove the
                        // leading bits outside the logical source width.
                        if (sdtype == ScalarDataType::I8) {
                            cc.and_(result.r32(), 0xff);
                            cc.lzcnt(result.r32(), result.r32());
                            cc.sub(result.r32(), 24);
                        } else if (sdtype == ScalarDataType::I16) {
                            cc.and_(result.r32(), 0xffff);
                            cc.lzcnt(result.r32(), result.r32());
                            cc.sub(result.r32(), 16);
                        } else {
                            cc.lzcnt(result, arg);
                        }
                        break;
                    case ArithUnaryOp::Tzcnt:
                        // Add a sentinel immediately above small scalar widths so zero produces the logical width.
                        if (sdtype == ScalarDataType::I8) {
                            cc.and_(result.r32(), 0xff);
                            cc.or_(result.r32(), 0x100);
                            cc.tzcnt(result.r32(), result.r32());
                        } else if (sdtype == ScalarDataType::I16) {
                            cc.and_(result.r32(), 0xffff);
                            cc.or_(result.r32(), 0x10000);
                            cc.tzcnt(result.r32(), result.r32());
                        } else {
                            cc.tzcnt(result, arg);
                        }
                        break;
                    case ArithUnaryOp::Popcount:
                        // There is no 8 bit version, make sure we don't count garbage values
                        if (sdtype == ScalarDataType::I8) {
                            cc.and_(result.r32(), 0xff);
                            cc.popcnt(result.r32(), result.r32());
                            break;
                        }
                        cc.popcnt(result, arg);
                        break;
                    default: messed_up("unsupported scalar int unary op %s", show_arith_unary_op(data.op));
                    }
                    return result;
                }

                VecR arg = float_subexpr(data.arg);
                VecR result = cc.new_vec128_f32x1();
                switch (data.op) {
                case ArithUnaryOp::Abs:
                case ArithUnaryOp::Lzcnt:
                case ArithUnaryOp::Tzcnt:
                case ArithUnaryOp::Popcount:
                    messed_up("scalar float unary op %s should have been rewritten before x86 emission",
                              show_arith_unary_op(data.op));
                case ArithUnaryOp::Not: {
                    if (sdtype == ScalarDataType::F32) {
                        uint32_t sign = 0xFFFFFFFF;
                        ax86::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, &sign, sizeof(sign));
                        cc.vandnps(result, arg, mem._1to4());
                    } else {
                        uint64_t sign = 0xFFFFFFFFFFFFFFFF;
                        ax86::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, &sign, sizeof(sign));
                        cc.vandnps(result, arg, mem._1to2());
                    }
                    break;
                }
                case ArithUnaryOp::Negate: {
                    // ad-hoc minus because it is not very important
                    if (sdtype == ScalarDataType::F32) {
                        uint32_t sign = 0x80000000;
                        ax86::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, &sign, sizeof(sign));
                        cc.vxorps(result, arg, mem._1to4());
                    } else {
                        uint64_t sign = 0x8000000000000000;
                        ax86::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, &sign, sizeof(sign));
                        cc.vxorpd(result, arg, mem._1to2());
                    }
                    break;
                }
                case ArithUnaryOp::RoundNearest:
                case ArithUnaryOp::RoundDown:
                case ArithUnaryOp::RoundUp:
                case ArithUnaryOp::RoundTruncate:
                case ArithUnaryOp::Rcp:
                case ArithUnaryOp::Sqrt:
                case ArithUnaryOp::Rsqrt: {
                    auto maybe_vec = vec_elem_from_scalar(sdtype);
                    SIMJIT_ASSERT(maybe_vec.has_value());
                    vec_unary(cc, data.op, x86::x86_to_vec(x86::Vector{x86::VecRegisterKind::XMM, *maybe_vec}), result,
                              arg);
                    break;
                }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                GpR result = create_int_reg(sdtype);
                ScalarDataType from = data.arg->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                bool is_sext = data.kind == IntCastKind::Sext;

                if (scalar_dtype_size(from) > scalar_dtype_size(to)) {
                    GpR arg = int_subexpr(data.arg);
                    SIMJIT_ASSERT(data.kind == IntCastKind::Trunc);
                    switch (to) {
                    case ScalarDataType::I8: cc.mov(result, arg.r8()); break;
                    case ScalarDataType::I16: cc.mov(result, arg.r16()); break;
                    case ScalarDataType::I32: cc.mov(result, arg.r32()); break;
                    case ScalarDataType::I64:
                        messed_up("invalid scalar truncation from %s to %s", show_scalar_dtype(from),
                                  show_scalar_dtype(to));
                    case ScalarDataType::I1:
                    case ScalarDataType::I128:
                    case ScalarDataType::F32:
                    case ScalarDataType::F64:
                        messed_up("invalid cast from %s to %s", show_scalar_dtype(from), show_scalar_dtype(to));
                    }
                    return result;
                }
                if (data.arg->is(StepKind::Load) && should_inline_mem(data.arg)) {
                    const auto &load_data = data.arg->step_data<StepKind::Load>();
                    ax86::Mem mem = scalar_load_mem(args[load_data.addr.arg], data.arg->dtype.as_scalar());
                    scalar_cast(is_sext, from, to, result, mem);
                    return result;
                }
                GpR arg = int_subexpr(data.arg);
                scalar_cast(is_sext, from, to, result, arg);
                return result;
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                VecR arg;
                VecR cvt_res = cc.new_vec128();
                ScalarDataType from = data.arg->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                if (is_float_dtype(from)) {
                    arg = float_subexpr(data.arg);
                } else {
                    GpR arg_tmp = int_subexpr(data.arg);
                    arg = cc.new_vec128();
                    if (from == ScalarDataType::I32) {
                        cc.vmovd(arg, arg_tmp);
                    } else {
                        cc.vmovq(arg, arg_tmp);
                    }
                }
                auto from_vec = vec_elem_from_scalar(from);
                auto to_vec = vec_elem_from_scalar(to);
                SIMJIT_ASSERT(from_vec.has_value());
                SIMJIT_ASSERT(to_vec.has_value());
                vec_float_cast(cc, *from_vec, *to_vec, data.is_unsigned, cvt_res, arg);
                AnyR result = cvt_res;
                if (!is_float_dtype(to)) {
                    if (to == ScalarDataType::I32) {
                        GpR result_gp = cc.new_gp32();
                        cc.vmovd(result_gp, cvt_res);
                        result = result_gp;
                    } else {
                        GpR result_gp = cc.new_gp64();
                        cc.vmovq(result_gp, cvt_res);
                        result = result_gp;
                    }
                }

                return result;
            }
            SIMJIT_MATCH (StepKind::BitCast) {
                ScalarDataType from = data->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                SIMJIT_ASSERT((is_simple_int_dtype(from) && is_float_dtype(to)) ||
                              (is_simple_int_dtype(to) && is_float_dtype(from)));
                if (is_simple_int_dtype(from)) {
                    SIMJIT_ASSERT(is_float_dtype(to));
                    GpR arg = int_subexpr(data);
                    VecR result = cc.new_vec128();
                    if (to == ScalarDataType::F32) {
                        cc.vmovd(result, arg);
                    } else {
                        cc.vmovq(result, arg);
                    }
                    return result;
                }
                SIMJIT_ASSERT(is_simple_int_dtype(to));
                VecR arg = float_subexpr(data);
                GpR result;
                if (to == ScalarDataType::I32) {
                    result = cc.new_gp32();
                    cc.vmovd(result, arg);
                } else {
                    result = cc.new_gp64();
                    cc.vmovq(result, arg);
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Compare) {
                SIMJIT_ASSERT(data.left->dtype == data.right->dtype);
                ScalarDataType arg_dtype = data.left->dtype.as_scalar();
                if (is_float_dtype(arg_dtype)) {
                    // Main complexity here comes from handling NAN cases, since we implement ordered comparison.
                    VecR left = float_subexpr(data.left);
                    VecR right = float_subexpr(data.right);

                    CmpOp op = data.op;
                    switch (data.op) {
                    case CmpOp::Less:
                        std::swap(left, right);
                        op = CmpOp::Greater;
                        break;
                    case CmpOp::LessEqual:
                        std::swap(left, right);
                        op = CmpOp::GreaterEqual;
                        break;
                    default: break;
                    }

                    if (arg_dtype == ScalarDataType::F32) {
                        // ucomiss does not raise exception on nan (unlike comiss)
                        cc.vucomiss(left, right);
                    } else {
                        SIMJIT_ASSERT(arg_dtype == ScalarDataType::F64);
                        cc.vucomisd(left, right);
                    }
                    GpR result = create_int_reg(sdtype);
                    flags2bool(result, op, true);
                    if (op == CmpOp::Equal) {
                        GpR tmp = cc.new_gp8();
                        cc.setnp(tmp);
                        cc.and_(result, tmp);
                    } else if (op == CmpOp::NotEqual) {
                        GpR tmp = cc.new_gp8();
                        cc.setp(tmp);
                        cc.or_(result, tmp);
                    }
                    return result;
                }

                // Hir should have tried to have right side be a memory/const operand if possible
                GpR left = int_subexpr(data.left);
                do_int_cmp(left, data.right, data.op);

                GpR result = create_int_reg(sdtype);
                flags2bool(result, data.op, data.is_unsigned);
                return result;
            }
            SIMJIT_MATCH (StepKind::AggResult) {
                const ArgInfo &info = args[data.dst];
                AnyR arg = subexpr(data.arg);
                ax86::Mem mem{};
                switch (step->dtype.as_scalar()) {
                case ScalarDataType::I8: mem = ax86::byte_ptr(info.gp); break;
                case ScalarDataType::I16: mem = ax86::word_ptr(info.gp); break;
                case ScalarDataType::I32:
                case ScalarDataType::F32: mem = ax86::dword_ptr(info.gp); break;
                case ScalarDataType::I64:
                case ScalarDataType::F64: mem = ax86::qword_ptr(info.gp); break;
                case ScalarDataType::I1: messed_up("i1 aggs should've been rewritten to use i8");
                case ScalarDataType::I128: messed_up("i128 aggs should've been handled with special opcodes");
                }
                scalar_mov(cc, mem, arg, sdtype);
                return {};
            }
            SIMJIT_MATCH (StepKind::StoreSum128) {
                const ArgInfo &info = args[data.dst];
                GpR zero = cc.new_gp64();
                cc.mov(zero, 0);
                GpR high = int_subexpr(data.hi_combined);
                GpR low = cc.new_gp64();
                cc.mov(low, 0);
                for (Step *lo_step : data.low_steps) {
                    if (lo_step->dtype.is_scalar()) {
                        GpR lo_val = int_subexpr(lo_step);
                        cc.add(low, lo_val);
                        cc.adc(high, 0);
                    } else {
                        VecDataType vdtype = lo_step->dtype.as_vec();
                        VecR vec = vec_subexpr(lo_step);
                        ax86::Mem mem = cc.new_stack(vdtype.size_bytes(), 8);
                        GpR lo_val = cc.new_gp64();
                        cc.vmovdqu64(mem, vec);
                        for (size_t i = 0; i < vdtype.nelems(); ++i) {
                            int64_t off = (int64_t)(i * vdtype.element_size_bytes());
                            cc.mov(lo_val, mem.clone_adjusted(off));
                            cc.add(low, lo_val);
                            cc.adc(high, 0);
                        }
                    }
                }

                ax86::Mem low_mem = ax86::qword_ptr(info.gp);
                ax86::Mem high_mem = ax86::qword_ptr(info.gp, 8);
                cc.mov(low_mem, low);
                cc.mov(high_mem, high);
                return {};
            }
            SIMJIT_MATCH (StepKind::AccLoad) {
                require_acc_initialized(data);
                return accs[mir_func->accs.index(data)];
            }
            SIMJIT_MATCH (StepKind::AccStore) {
                AnyR acc_reg = accs[mir_func->accs.index(data.acc)];
                if (SIMJIT_X64_ASMJIT_INLINE_ACC_INIT && is_simple_int_dtype(sdtype) && data.arg->is(StepKind::Const)) {
                    // inline const to avoid machine code growing excessively
                    init_scalar_int_const(acc_reg.as<GpR>(), sdtype,
                                          data.arg->step_data<StepKind::Const>().as_signed());
                    mark_acc_initialized(data.acc);
                    return {};
                }
                if (SIMJIT_X64_ASMJIT_INLINE_ACC_BIN && is_simple_int_dtype(sdtype) &&
                    data.arg->is(StepKind::ArithBinary)) {
                    // Optimize common case where we load + bin + store acc
                    auto &bin_data = data.arg->step_data<StepKind::ArithBinary>();
                    if (bin_data.left->is(StepKind::AccLoad) &&
                        bin_data.left->step_data<StepKind::AccLoad>() == data.acc) {
                        require_acc_initialized(data.acc);
                        scalar_int_arith_binary(sdtype, bin_data.op, acc_reg.as<GpR>(), bin_data.right, true);
                        mark_acc_initialized(data.acc);
                        return {};
                    }
                }
                // NOTE: could add similar optimization for float accs, but I am too lazy
                AnyR arg = subexpr(data.arg);
                scalar_mov(cc, acc_reg, arg, sdtype);
                mark_acc_initialized(data.acc);
                return {};
            }
            SIMJIT_MATCH (StepKind::PredicateNot) {
                GpR arg = int_subexpr(data);
                GpR result = create_int_reg(sdtype);
                cc.mov(result, arg);
                cc.xor_(result, 1);
                return result;
            }
            SIMJIT_MATCH (StepKind::Select) {
                if (is_float_dtype(sdtype)) {
                    VecR falsy = float_subexpr(data.falsy);
                    VecR truthy = float_subexpr(data.truthy);
                    GpR cond = int_subexpr(data.cond);
                    AnyR dst = create_scalar_reg(sdtype);

                    // stupid solution with jump
                    aj::Label skip_label = cc.new_anonymous_label("skip");

                    scalar_mov(cc, dst, falsy, sdtype);
                    cc.test(cond, cond);
                    cc.jz(skip_label);
                    scalar_mov(cc, dst, truthy, sdtype);
                    cc.bind(skip_label);
                    return dst;
                }
                GpR falsy = int_subexpr(data.falsy);
                GpR truthy = int_subexpr(data.truthy);
                GpR result = create_int_reg(sdtype);
                cc.mov(result, falsy);

                if (SIMJIT_X64_ASMJIT_INLINE_SCALAR_COND && data.cond->is(StepKind::Compare)) {
                    auto &cond_data = data.cond->step_data<StepKind::Compare>();
                    if (!is_float_dtype(cond_data.left->dtype.as_scalar())) {
                        GpR cmp_left = int_subexpr(cond_data.left);
                        do_int_cmp(cmp_left, cond_data.right, cond_data.op);
                        flags2cmov(result, truthy, cond_data.op, cond_data.is_unsigned);
                        return result;
                    }
                }
                GpR cond = int_subexpr(data.cond);
                cc.test(cond, cond);
                cc.cmovne(result, truthy);
                return result;
            }
            SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
                GpR left = int_subexpr(data.left);
                GpR right = int_subexpr(data.right);
                GpR result = create_int_reg(sdtype);
                cc.mov(result, left);
                GpR overflow_flag = cc.new_gp8();
                switch (data.op) {
                case ArithBinaryOp::Add:
                    cc.add(result, right);
                    cc.seto(overflow_flag);
                    break;
                case ArithBinaryOp::Sub:
                    cc.sub(result, right);
                    cc.seto(overflow_flag);
                    break;
                case ArithBinaryOp::Mul:
                    cc.imul(result, right);
                    cc.seto(overflow_flag);
                    break;
                default: messed_up("invalid overflow step");
                }
                if (data.mask != nullptr) { cc.and_(overflow_flag, int_subexpr(data.mask)); }
                require_acc_initialized(data.overflow_flag);
                GpR overflow_acc = accs[mir_func->accs.index(data.overflow_flag)].as<GpR>();
                cc.or_(overflow_acc, overflow_flag);
                return result;
            }
            SIMJIT_MATCH (StepKind::ScalarIndex) {
                if (sdtype == ScalarDataType::I32) { return counter.r32(); }
                return counter;
            }
            SIMJIT_MATCH (StepKind::ScalarPermute) {
                GpR result = create_int_reg(sdtype);
                GpR arg = int_subexpr(data.arg);
                if (!data.is_bit && sdtype == ScalarDataType::I16 && data.permute == REVERSE_BYTES_I16) {
                    cc.mov(result, arg);
                    cc.rol(result, 8);
                    return result;
                }
                if (!data.is_bit && ((sdtype == ScalarDataType::I32 && data.permute == REVERSE_BYTES_I32) ||
                                     (sdtype == ScalarDataType::I64 && data.permute == REVERSE_BYTES_I64))) {
                    cc.mov(result, arg);
                    cc.bswap(result);
                    return result;
                }

                GpR tmp = create_int_reg(sdtype);
                size_t dtype_size = scalar_dtype_size(step->dtype.as_scalar());
                if (data.is_bit) {
                    if (auto it = bit_permute_luts.find(data.permute); it != bit_permute_luts.end()) {
                    } else {
                        aj::Label label = cc.new_label();
                        bit_permute_luts[data.permute] = label;
                    }

                    GpR addr = cc.new_gp64();
                    aj::Label label = bit_permute_luts.at(data.permute);
                    cc.lea(addr, ax86::ptr(label));
                    cc.xor_(result, result);
                    for (size_t i = 0; i < dtype_size; ++i) {
                        cc.mov(tmp, arg);
                        if (i != 0) { cc.shr(tmp, i * 8); }
                        cc.movzx(tmp.r32(), tmp.r8());
                        cc.movzx(tmp.r64(), ax86::byte_ptr(addr, tmp.r32()));
                        if (i != 0) { cc.shl(tmp, i * 8); }
                        cc.or_(result, tmp);
                    }
                } else {
                    cc.xor_(result, result);
                    for (size_t i = 0; i < dtype_size; ++i) {
                        cc.mov(tmp, arg);
                        size_t imm = ((data.permute >> (i * 8)) & 0xff) * 8;
                        if (imm != 0) cc.shr(tmp, imm);
                        cc.and_(tmp, 0xff);
                        if (i != 0) cc.shl(tmp, i * 8);
                        cc.or_(result, tmp);
                    }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                const ArgInfo &arg_info = args[data.dst];
                AnyR arg = subexpr(data.arg);
                GpR idx = int_subexpr(data.idx);
                ax86::Mem mem = mem_offset(arg_info.gp, idx, sdtype);
                scalar_mov(cc, mem, arg, sdtype);
                return {};
            }
            SIMJIT_MATCH (StepKind::CondScatter) {
                const ArgInfo &arg_info = args[data.dst];
                AnyR arg = subexpr(data.arg);
                GpR cond = int_subexpr(data.cond);
                GpR idx = int_subexpr(data.idx);
                ax86::Mem mem = mem_offset(arg_info.gp, idx, sdtype);
                aj::Label skip_label = cc.new_anonymous_label("skip");
                cc.test(cond, cond);
                cc.jz(skip_label);
                scalar_mov(cc, mem, arg, sdtype);
                cc.bind(skip_label);
                return {};
            }
            SIMJIT_MATCH (StepKind::Pack) {
                const ArgInfo &arg_info = args[data.dst];
                require_acc_initialized(data.acc);
                GpR acc = accs[mir_func->accs.index(data.acc)].as<GpR>();
                AnyR arg = subexpr(data.arg);
                GpR cond = int_subexpr(data.cond);
                ax86::Mem mem = mem_offset(arg_info.gp, acc, sdtype);
                aj::Label skip_label = cc.new_anonymous_label("skip");
                cc.test(cond, cond);
                cc.jz(skip_label);
                scalar_mov(cc, mem, arg, sdtype);
                cc.inc(acc);
                cc.bind(skip_label);
                return {};
            }
            SIMJIT_MATCH (StepKind::CondStore) {
                const ArgInfo &arg_info = args[data.addr.arg];
                AnyR arg = subexpr(data.arg);
                GpR cond = int_subexpr(data.cond);
                ax86::Mem mem = mem_offset(arg_info.gp, counter, sdtype);
                aj::Label skip_label = cc.new_anonymous_label("skip");
                cc.test(cond, cond);
                cc.jz(skip_label);
                scalar_mov(cc, mem, arg, sdtype);
                cc.bind(skip_label);
                return {};
            }
            SIMJIT_MATCH (StepKind::Fpclass) {
                VecR arg = float_subexpr(data.arg);
                MaskR mask_result = cc.new_kb();
                GpR result = cc.new_gp32();

                ax86::VFPClassImm imm = make_fpclass_imm(data.flags);
                if (data.arg->dtype.as_scalar() == ScalarDataType::F32) {
                    cc.vfpclassss(mask_result, arg, imm);
                } else {
                    cc.vfpclasssd(mask_result, arg, imm);
                }
                mask_to_gp(cc, MaskDataType::M8, result, mask_result);
                return result.r8();
            }
        default: messed_up("unexpected step %s in scalar context", show_step_kind(step->kind));
        }
        messed_up("scalar_subexpr fell through for step %s", show_step_kind(step->kind));
    }

    VecR create_vec_reg(VecDataType vdtype) {
        x86::Vector vec = x86::vec_to_x86(vdtype);
        switch (vec.reg) {
        case x86::VecRegisterKind::XMM: return cc.new_vec128();
        case x86::VecRegisterKind::YMM: return cc.new_vec256();
        case x86::VecRegisterKind::ZMM: return cc.new_vec512();
        }
        SIMJIT_UNREACHABLE();
    }

    VecR create_zero_vec_reg(VecDataType vdtype) {
        VecR reg = create_vec_reg(vdtype);
        switch (vdtype.elem) {
        case VecElemType::I8:
        case VecElemType::I16:
        case VecElemType::I32:
        case VecElemType::I64: cc.vpxor(reg, reg, reg); break;
        case VecElemType::F32: cc.vxorps(reg, reg, reg); break;
        case VecElemType::F64: cc.vxorpd(reg, reg, reg); break;
        }
        return reg;
    }

    AnyR create_scalar_reg(ScalarDataType scalar) {
        if (scalar == ScalarDataType::F32) { return cc.new_vec128_f32x1(); }
        if (scalar == ScalarDataType::F64) { return cc.new_vec128_f64x1(); }
        return cc.new_gp(scalar_dtype_to_asmjit(scalar));
    }
    GpR create_int_reg(ScalarDataType scalar) { return cc.new_gp(scalar_dtype_to_asmjit(scalar)); }
    MaskR create_mask_reg(MaskDataType mask) { return cc.new_k(mask_dtype_to_asmjit(mask)); }

    void gp_to_mask(MaskDataType mdtype, const MaskR &dst, const GpR &src) {
        switch (mdtype) {
        case MaskDataType::M2:
        case MaskDataType::M4:
        case MaskDataType::M8: cc.kmovb(dst, src); break;
        case MaskDataType::M16: cc.kmovw(dst, src); break;
        case MaskDataType::M32: cc.kmovd(dst, src); break;
        case MaskDataType::M64: cc.kmovq(dst, src); break;
        }
    }

    MaskR copy_mask_for_destructive_use(MaskDataType mdtype, const MaskR &src) {
        GpR bits = mdtype == MaskDataType::M64 ? cc.new_gp64() : cc.new_gp32();
        mask_to_gp(cc, mdtype, bits, src);
        MaskR dst = create_mask_reg(mdtype);
        gp_to_mask(mdtype, dst, bits);
        return dst;
    }

    void init_partial_byte_mask_const(const MaskR &reg, MaskDataType mdtype) {
        SIMJIT_ASSERT(mask_uses_partial_byte(mdtype));
        GpR gp = cc.new_gp32();
        cc.mov(gp, mask_valid_bits_u32(mdtype));
        cc.kmovb(reg, gp);
    }

    void canonicalize_partial_byte_mask(const MaskR &reg, MaskDataType mdtype) {
        if (!mask_uses_partial_byte(mdtype)) { return; }
        MaskR valid = create_mask_reg(mdtype);
        init_partial_byte_mask_const(valid, mdtype);
        cc.kandb(reg, reg, valid);
    }

    SIMJIT_ALWAYS_INLINE AnyR subexpr(const Step *step, const MaskPushdownInfo *mask_pushdown = nullptr) {
        if (step_map[step->id].is_valid()) { return step_map[step->id]; }
        SIMJIT_ASSERT(!const_is_folded_root(step));
        AnyR v = subexpr_internal(step, mask_pushdown);
        step_map[step->id] = v;
        return v;
    }

    SIMJIT_ALWAYS_INLINE GpR int_subexpr(const Step *step) {
        AnyR reg = subexpr(step);
        if (!reg.is_gp()) {
            messed_up("expected gp(int) for step %s(%s)", mir::show_step_kind(step->kind), show_dtype(step->dtype));
        }
        return reg.as<GpR>();
    }
    SIMJIT_ALWAYS_INLINE VecR float_subexpr(const Step *step) {
        AnyR reg = subexpr(step);
        if (!reg.is_vec128()) {
            messed_up("expected xmm(float) for step %s(%s)", mir::show_step_kind(step->kind), show_dtype(step->dtype));
        }
        return reg.as<VecR>();
    }

    SIMJIT_ALWAYS_INLINE VecR vec_subexpr(const Step *step, const MaskPushdownInfo *pushdown = nullptr) {
        AnyR reg = subexpr(step, pushdown);
        if (!reg.is_vec()) {
            messed_up("expected vector for step %s(%s)", mir::show_step_kind(step->kind), show_dtype(step->dtype));
        }
        return reg.as<VecR>();
    }
    SIMJIT_ALWAYS_INLINE MaskR mask_subexpr(const Step *step, const MaskPushdownInfo *pushdown = nullptr) {
        AnyR reg = subexpr(step, pushdown);
        if (!reg.is_mask_reg()) {
            messed_up("expected mask for step %s(%s)", mir::show_step_kind(step->kind), show_dtype(step->dtype));
        }
        return reg.as<MaskR>();
    }

    AnyR subexpr_internal(const Step *step, const MaskPushdownInfo *mask_pushdown) {
        if (step->dtype.is_scalar() && is_scalar_step(step->kind)) { return scalar_subexpr(step); }

        switch (step->kind) {
        case StepKind::AggResult:
        case StepKind::ScalarIndex:
        case StepKind::ScalarArithBinaryOverflow:
        case StepKind::ScalarPermute:
        case StepKind::StoreSum128:
        case StepKind::ConstDiv:
            messed_up("this is scalar instruction");

            SIMJIT_MATCH (StepKind::Const) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    MaskR reg = create_mask_reg(mdtype);
                    init_mask_const(reg, mdtype, data.as_signed());
                    return reg;
                }

                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                init_vec_const(reg, vdtype, data);
                return reg;
            }
            SIMJIT_MATCH (StepKind::VecConst) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                // Do not try to use aligned load here because we can't guarantee 64 byte align
                ax86::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, data.mem, vdtype.size_bytes());
                switch (vdtype.elem) {
                case VecElemType::I8:
                case VecElemType::I16:
                case VecElemType::I32:
                case VecElemType::I64: cc.vmovdqu64(reg, mem); break;
                case VecElemType::F32: cc.vmovups(reg, mem); break;
                case VecElemType::F64: cc.vmovupd(reg, mem); break;
                }
                return reg;
            }

            SIMJIT_MATCH (StepKind::Load) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    MaskR reg = create_mask_reg(mdtype);
                    if (mask_uses_partial_byte(mdtype)) {
                        size_t bit_count = mask_dtype_bits(mdtype);
                        size_t bit_offset = data.addr.offset & 7;
                        SIMJIT_ASSERT(bit_offset + bit_count <= 8);
                        GpR byte_idx = get_or_insert_shifted_counter(3);
                        ax86::Mem mem =
                            ax86::byte_ptr(args[data.addr.arg].gp, byte_idx, 0, (int32_t)(data.addr.offset >> 3));
                        GpR gp = cc.new_gp32();
                        cc.movzx(gp, mem);
                        if (bit_offset != 0) { cc.shr(gp, bit_offset); }
                        cc.and_(gp, mask_valid_bits_u32(mdtype));
                        cc.kmovb(reg, gp);
                        return reg;
                    }
                    size_t shift_amount = mask_dtype_bits_log2(mdtype);
                    GpR gp = get_or_insert_shifted_counter(shift_amount);
                    ax86::Mem mem = mem_offset(args[data.addr.arg].gp, gp, mask_dtype_to_scalar(mdtype),
                                               (int32_t)((data.addr.offset >> shift_amount) *
                                                         scalar_dtype_size(mask_dtype_to_scalar(mdtype))));
                    load_mask(cc, step->dtype.as_mask(), reg, mem);
                    return reg;
                }
                ax86::Mem mem = vec_load_mem(data.addr, step->dtype.as_vec());
                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                maybe_apply_mask_pushdown(cc, mask_pushdown, reg);
                switch (data.kind) {
                case LoadStoreKind::Aligned:
                    switch (vdtype.elem) {
                        // NOTE: I8, I16 are unaligned
                    case VecElemType::I8: cc.vmovdqu8(reg, mem); break;
                    case VecElemType::I16: cc.vmovdqu16(reg, mem); break;
                    case VecElemType::I32: cc.vmovdqa32(reg, mem); break;
                    case VecElemType::I64: cc.vmovdqa64(reg, mem); break;
                    case VecElemType::F32: cc.vmovaps(reg, mem); break;
                    case VecElemType::F64: cc.vmovapd(reg, mem); break;
                    }
                    break;
                case LoadStoreKind::Unaligned:
                    switch (vdtype.elem) {
                    case VecElemType::I8: cc.vmovdqu8(reg, mem); break;
                    case VecElemType::I16: cc.vmovdqu16(reg, mem); break;
                    case VecElemType::I32: cc.vmovdqu32(reg, mem); break;
                    case VecElemType::I64: cc.vmovdqu64(reg, mem); break;
                    case VecElemType::F32: cc.vmovups(reg, mem); break;
                    case VecElemType::F64: cc.vmovupd(reg, mem); break;
                    }
                    break;
                }
                return reg;
            }
            SIMJIT_MATCH (StepKind::LoadSplat) {
                const ArgInfo &arg = args[data.addr.arg];
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    MaskR reg = create_mask_reg(mdtype);
                    GpR gp = cc.new_gp8();
                    cc.mov(gp, ax86::byte_ptr(arg.gp));
                    cc.and_(gp.r64(), 1);
                    cc.neg(gp.r64());
                    switch (mdtype) {
                    case MaskDataType::M2:
                        cc.and_(gp, 3);
                        cc.kmovb(reg, gp);
                        break;
                    case MaskDataType::M4:
                        cc.and_(gp, 15);
                        cc.kmovb(reg, gp);
                        break;
                    case MaskDataType::M8: cc.kmovb(reg, gp); break;
                    case MaskDataType::M16: cc.kmovw(reg, gp); break;
                    case MaskDataType::M32: cc.kmovd(reg, gp); break;
                    case MaskDataType::M64: cc.kmovq(reg, gp); break;
                    }
                    return reg;
                }

                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                switch (vdtype.elem) {
                case VecElemType::I8: cc.vpbroadcastb(reg, ax86::byte_ptr(arg.gp)); break;
                case VecElemType::I16: cc.vpbroadcastw(reg, ax86::word_ptr(arg.gp)); break;
                case VecElemType::I32: cc.vpbroadcastd(reg, ax86::dword_ptr(arg.gp)); break;
                case VecElemType::I64: cc.vpbroadcastq(reg, ax86::qword_ptr(arg.gp)); break;
                case VecElemType::F32: cc.vbroadcastss(reg, ax86::dword_ptr(arg.gp)); break;
                case VecElemType::F64: broadcast_f64(cc, reg, ax86::qword_ptr(arg.gp)); break;
                }
                return reg;
            }
            SIMJIT_MATCH (StepKind::Gather) {
                VecDataType vdtype = step->dtype.as_vec();
                const ArgInfo &arg = args[data.data];
                VecR idx = vec_subexpr(data.idx);
                VecR result = create_vec_reg(vdtype);
                MaskDataType mdtype = vdtype.mask();

                // Xor because otherwise asmjit liveness analysis goes crazy, it is not actually needed
                cc.vpxor(result, result, result);
                if (mask_pushdown) {
                    SIMJIT_ASSERT(mask_pushdown->mask.is_valid());
                    MaskR gather_mask = copy_mask_for_destructive_use(mdtype, mask_pushdown->mask);
                    if (mask_pushdown->vec.is_valid()) {
                        cc.vmovdqa64(result, mask_pushdown->vec);
                        cc.k(gather_mask);
                    } else if (mask_pushdown->mask.is_valid()) {
                        // Note that we don't need {z} because we already did xor.
                        // Additionally, {z} in gather would cause UD which we want to avoid.
                        cc.k(gather_mask);
                    } else {
                        SIMJIT_UNREACHABLE();
                    }
                } else {
                    MaskR k = create_mask_reg(mdtype);
                    init_mask_const(k, mdtype, 1);
                    cc.k(k);
                }
                if (data.idx->dtype.as_vec().elem == VecElemType::I32) {
                    switch (vdtype.elem) {
                    case VecElemType::I8:
                    case VecElemType::I16: unsupported("Do not support i8/i16 gather");
                    case VecElemType::I32: cc.vpgatherdd(result, ax86::ptr(arg.gp, idx, 2)); break;
                    case VecElemType::I64: cc.vpgatherdq(result, ax86::ptr(arg.gp, idx, 3)); break;
                    case VecElemType::F32: cc.vgatherdps(result, ax86::ptr(arg.gp, idx, 2)); break;
                    case VecElemType::F64: cc.vgatherdpd(result, ax86::ptr(arg.gp, idx, 3)); break;
                    }
                } else {
                    switch (vdtype.elem) {
                    case VecElemType::I8:
                    case VecElemType::I16: unsupported("Do not support i8/i16 gather");
                    case VecElemType::I32: cc.vpgatherqd(result, ax86::ptr(arg.gp, idx, 2)); break;
                    case VecElemType::I64: cc.vpgatherqq(result, ax86::ptr(arg.gp, idx, 3)); break;
                    case VecElemType::F32: cc.vgatherqps(result, ax86::ptr(arg.gp, idx, 2)); break;
                    case VecElemType::F64: cc.vgatherqpd(result, ax86::ptr(arg.gp, idx, 3)); break;
                    }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Store) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    MaskR arg = mask_subexpr(data.what);
                    if (mask_uses_partial_byte(mdtype)) {
                        size_t bit_count = mask_dtype_bits(mdtype);
                        size_t bit_offset = data.addr.offset & 7;
                        uint32_t write_mask = mask_valid_bits_u32(mdtype) << bit_offset;
                        SIMJIT_ASSERT(bit_offset + bit_count <= 8);
                        GpR byte_idx = get_or_insert_shifted_counter(3);
                        ax86::Mem mem =
                            ax86::byte_ptr(args[data.addr.arg].gp, byte_idx, 0, (int32_t)(data.addr.offset >> 3));
                        GpR bits = cc.new_gp32();
                        GpR old = cc.new_gp32();
                        mask_to_gp(cc, mdtype, bits, arg);
                        if (bit_offset != 0) { cc.shl(bits, bit_offset); }
                        cc.and_(bits, write_mask);
                        cc.movzx(old, mem);
                        cc.and_(old, ~write_mask & 0xffu);
                        cc.or_(old, bits);
                        cc.mov(mem, old.r8());
                        return {};
                    }
                    size_t shift_amount = mask_dtype_bits_log2(mdtype);
                    GpR gp = get_or_insert_shifted_counter(shift_amount);
                    ax86::Mem mem = mem_offset(args[data.addr.arg].gp, gp, mask_dtype_to_scalar(mdtype),
                                               (int32_t)((data.addr.offset >> shift_amount) *
                                                         scalar_dtype_size(mask_dtype_to_scalar(mdtype))));
                    store_mask(cc, step->dtype.as_mask(), mem, arg);
                    return {};
                }
                ax86::Mem mem = vec_load_mem(data.addr, step->dtype.as_vec());
                if (SIMJIT_X64_ASMJIT_INLINE_VEC_CAST && data.what->is(StepKind::IntCast)) {
                    auto &cast_data = data.what->step_data<StepKind::IntCast>();
                    VecElemType from = cast_data.arg->dtype.as_vec().elem;
                    VecElemType to = data.what->dtype.as_vec().elem;
                    if (cast_data.kind == IntCastKind::Trunc) {
                        VecR arg = vec_subexpr(cast_data.arg);
                        vec_trunc(cc, from, to, mem, arg);
                        return {};
                    }
                }
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = vec_subexpr(data.what);
                switch (data.kind) {
                case LoadStoreKind::Aligned:
                    switch (vdtype.elem) {
                        // NOTE: I8, I16 are unaligned
                    case VecElemType::I8: cc.vmovdqu8(mem, arg); break;
                    case VecElemType::I16: cc.vmovdqu16(mem, arg); break;
                    case VecElemType::I32: cc.vmovdqa32(mem, arg); break;
                    case VecElemType::I64: cc.vmovdqa64(mem, arg); break;
                    case VecElemType::F32: cc.vmovaps(mem, arg); break;
                    case VecElemType::F64: cc.vmovapd(mem, arg); break;
                    }
                    break;
                case LoadStoreKind::Unaligned:
                    switch (vdtype.elem) {
                    case VecElemType::I8: cc.vmovdqu8(mem, arg); break;
                    case VecElemType::I16: cc.vmovdqu16(mem, arg); break;
                    case VecElemType::I32: cc.vmovdqu32(mem, arg); break;
                    case VecElemType::I64: cc.vmovdqu64(mem, arg); break;
                    case VecElemType::F32: cc.vmovups(mem, arg); break;
                    case VecElemType::F64: cc.vmovupd(mem, arg); break;
                    }
                    break;
                }
                return {};
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR result = create_vec_reg(vdtype);
                vec_arith_binary(vdtype, data.op, result, data.left, data.right, data.right != data.left,
                                 mask_pushdown);

                return result;
            }
            SIMJIT_MATCH (StepKind::FMA) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR a = vec_subexpr(data.x1);
                VecR b = vec_subexpr(data.x2);
                VecR result = create_vec_reg(vdtype);
                if (vdtype.elem == VecElemType::F32) {
                    cc.vmovaps(result, a);
                } else {
                    cc.vmovapd(result, a);
                }

                // NOTE: FMA does not support mask pushdown same as other steps because it has different merge
                // masking behavior and zero masking is not that useful on its own
                if (data.x3->is(StepKind::Load) && should_inline_mem(data.x3)) {
                    const auto &arg_data = data.x3->step_data<StepKind::Load>();
                    ax86::Mem mem = vec_load_mem(arg_data.addr, step->dtype.as_vec());
                    vec_fma(cc, data.kind, vdtype.elem, result, b, mem);
                    return result;
                }
                VecR c = vec_subexpr(data.x3);
                vec_fma(cc, data.kind, vdtype.elem, result, b, c);
                return result;
            }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR result = create_vec_reg(vdtype);

                if (data.arg->is(StepKind::Load) && should_inline_mem(data.arg)) {
                    const auto &arg_data = data.arg->step_data<StepKind::Load>();
                    ax86::Mem mem = vec_load_mem(arg_data.addr, step->dtype.as_vec());
                    maybe_apply_mask_pushdown(cc, mask_pushdown, result);
                    vec_unary(cc, data.op, vdtype, result, mem);
                    return result;
                }
                VecR arg = vec_subexpr(data.arg);
                maybe_apply_mask_pushdown(cc, mask_pushdown, result);
                vec_unary(cc, data.op, vdtype, result, arg);
                return result;
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR dst = create_vec_reg(vdtype);

                VecElemType from = data.arg->dtype.as_vec().elem;
                VecElemType to = step->dtype.as_vec().elem;
                if (data.arg->is(StepKind::Load) &&
                    (data.kind == IntCastKind::Sext || data.kind == IntCastKind::Zext) && should_inline_mem(data.arg)) {
                    const auto &arg_data = data.arg->step_data<StepKind::Load>();
                    ax86::Mem mem = vec_load_mem(arg_data.addr, data.arg->dtype.as_vec());
                    maybe_apply_mask_pushdown(cc, mask_pushdown, dst);

                    if (data.kind == IntCastKind::Sext) {
                        vec_sext(cc, from, to, dst, mem);
                    } else {
                        vec_zext(cc, from, to, dst, mem);
                    }
                    return dst;
                }

                VecR arg = vec_subexpr(data.arg);
                maybe_apply_mask_pushdown(cc, mask_pushdown, dst);
                switch (data.kind) {
                case IntCastKind::Trunc: vec_trunc(cc, from, to, dst, arg); break;
                case IntCastKind::Sext: vec_sext(cc, from, to, dst, arg); break;
                case IntCastKind::Zext: vec_zext(cc, from, to, dst, arg); break;
                }
                return dst;
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR dst = create_vec_reg(vdtype);
                VecElemType from = data.arg->dtype.as_vec().elem;
                VecElemType to = step->dtype.as_vec().elem;

                if (data.arg->is(StepKind::Load) && should_inline_mem(data.arg)) {
                    const auto &arg_data = data.arg->step_data<StepKind::Load>();
                    ax86::Mem mem = vec_load_mem(arg_data.addr, data.arg->dtype.as_vec());
                    maybe_apply_mask_pushdown(cc, mask_pushdown, dst);
                    vec_float_cast(cc, from, to, data.is_unsigned, dst, mem);
                    return dst;
                }

                VecR arg = vec_subexpr(data.arg);
                maybe_apply_mask_pushdown(cc, mask_pushdown, dst);
                vec_float_cast(cc, from, to, data.is_unsigned, dst, arg);
                return dst;
            }
            SIMJIT_MATCH (StepKind::BitCast) {
                // here we Do not actually do anything
                return vec_subexpr(data);
            }
            SIMJIT_MATCH (StepKind::Compare) {
                MaskDataType mdtype = step->dtype.as_mask();
                VecDataType vdtype = data.left->dtype.as_vec();
                MaskR result = create_mask_reg(mdtype);

                // test/testn
                if (SIMJIT_X64_ASMJIT_ZERO_CMP && vdtype.is_int() && step_is_zero(data.right) &&
                    (data.left->is(StepKind::ArithBinary) &&
                     data.left->step_data<StepKind::ArithBinary>().op == ArithBinaryOp::And) &&
                    (data.op == CmpOp::Equal || data.op == CmpOp::NotEqual)) {
                    const auto &bin = data.left->step_data<StepKind::ArithBinary>();
                    VecR left = vec_subexpr(bin.left);

                    if (bin.right->is(StepKind::Load) && should_inline_mem(bin.right)) {
                        const auto &right_data = bin.right->step_data<StepKind::Load>();
                        ax86::Mem mem = vec_load_mem(right_data.addr, vdtype);
                        maybe_apply_mask_pushdown(cc, mask_pushdown);
                        if (data.op == CmpOp::NotEqual) {
                            switch (vdtype.elem) {
                                INVALID_FLOAT_CASES;
                            case VecElemType::I8: cc.vptestmb(result, left, mem); break;
                            case VecElemType::I16: cc.vptestmw(result, left, mem); break;
                            case VecElemType::I32: cc.vptestmd(result, left, mem); break;
                            case VecElemType::I64: cc.vptestmq(result, left, mem); break;
                            }
                        } else {
                            switch (vdtype.elem) {
                                INVALID_FLOAT_CASES;
                            case VecElemType::I8: cc.vptestnmb(result, left, mem); break;
                            case VecElemType::I16: cc.vptestnmw(result, left, mem); break;
                            case VecElemType::I32: cc.vptestnmd(result, left, mem); break;
                            case VecElemType::I64: cc.vptestnmq(result, left, mem); break;
                            }
                        }
                        return result;
                    }

                    VecR right = vec_subexpr(bin.right);
                    maybe_apply_mask_pushdown(cc, mask_pushdown);
                    if (data.op == CmpOp::NotEqual) {
                        switch (vdtype.elem) {
                            INVALID_FLOAT_CASES;
                        case VecElemType::I8: cc.vptestmb(result, left, right); break;
                        case VecElemType::I16: cc.vptestmw(result, left, right); break;
                        case VecElemType::I32: cc.vptestmd(result, left, right); break;
                        case VecElemType::I64: cc.vptestmq(result, left, right); break;
                        }
                    } else {
                        switch (vdtype.elem) {
                            INVALID_FLOAT_CASES;
                        case VecElemType::I8: cc.vptestnmb(result, left, right); break;
                        case VecElemType::I16: cc.vptestnmw(result, left, right); break;
                        case VecElemType::I32: cc.vptestnmd(result, left, right); break;
                        case VecElemType::I64: cc.vptestnmq(result, left, right); break;
                        }
                    }
                    return result;
                }

                if (SIMJIT_X64_ASMJIT_ZERO_CMP && vdtype.is_int() && step_is_zero(data.right) &&
                    (data.op == CmpOp::Less || data.op == CmpOp::Equal || data.op == CmpOp::NotEqual)) {
                    VecR left = vec_subexpr(data.left);
                    switch (data.op) {
                    case CmpOp::Less:
                        if (data.is_unsigned) {
                            mask_zero(cc, mdtype, result);
                        } else {
                            switch (vdtype.elem) {
                                INVALID_FLOAT_CASES;
                            case VecElemType::I8: cc.vpmovb2m(result, left); break;
                            case VecElemType::I16: cc.vpmovw2m(result, left); break;
                            case VecElemType::I32: cc.vpmovd2m(result, left); break;
                            case VecElemType::I64: cc.vpmovq2m(result, left); break;
                            }
                        }
                        // For simplicity we treat all comparisons that they support mask pushdown. This peephole does
                        // not - but we manually implement what it is supposed to do.
                        if (mask_pushdown != nullptr) {
                            SIMJIT_ASSERT(mask_pushdown->mask.is_valid());
                            SIMJIT_ASSERT(!mask_pushdown->vec.is_valid());
                            mask_and(cc, mdtype, result, result, mask_pushdown->mask);
                        }
                        break;
                    case CmpOp::Greater:
                    case CmpOp::LessEqual:
                    case CmpOp::GreaterEqual:
                        messed_up("zero-compare fast path does not support op %s", show_cmp_op(data.op));
                    case CmpOp::Equal:
                        maybe_apply_mask_pushdown(cc, mask_pushdown);
                        switch (vdtype.elem) {
                            INVALID_FLOAT_CASES;
                        case VecElemType::I8: cc.vptestnmb(result, left, left); break;
                        case VecElemType::I16: cc.vptestnmw(result, left, left); break;
                        case VecElemType::I32: cc.vptestnmd(result, left, left); break;
                        case VecElemType::I64: cc.vptestnmq(result, left, left); break;
                        }
                        break;
                    case CmpOp::NotEqual:
                        maybe_apply_mask_pushdown(cc, mask_pushdown);
                        switch (vdtype.elem) {
                            INVALID_FLOAT_CASES;
                        case VecElemType::I8: cc.vptestmb(result, left, left); break;
                        case VecElemType::I16: cc.vptestmw(result, left, left); break;
                        case VecElemType::I32: cc.vptestmd(result, left, left); break;
                        case VecElemType::I64: cc.vptestmq(result, left, left); break;
                        }
                        break;
                    }
                    return result;
                }

                if (data.left->is(StepKind::Load) && should_inline_mem(data.left) &&
                    !(data.right->is(StepKind::Load) && should_inline_mem(data.right))) {
                    const auto &left_data = data.left->step_data<StepKind::Load>();
                    ax86::Mem mem = vec_load_mem(left_data.addr, data.left->dtype.as_vec());
                    VecR right = vec_subexpr(data.right);
                    CmpOp swapped_op = swap_cmp_operands(data.op);
                    maybe_apply_mask_pushdown(cc, mask_pushdown);
                    if (vdtype.is_int())
                        vec_int_cmp(cc, swapped_op, vdtype, data.is_unsigned, result, right, mem);
                    else
                        vec_float_cmp(cc, swapped_op, vdtype, result, right, mem);
                    return result;
                }

                VecR left = vec_subexpr(data.left);

                if (data.right->is(StepKind::Load) && should_inline_mem(data.right)) {
                    const auto &right_data = data.right->step_data<StepKind::Load>();
                    ax86::Mem mem = vec_load_mem(right_data.addr, data.right->dtype.as_vec());
                    maybe_apply_mask_pushdown(cc, mask_pushdown);
                    if (vdtype.is_int())
                        vec_int_cmp(cc, data.op, vdtype, data.is_unsigned, result, left, mem);
                    else
                        vec_float_cmp(cc, data.op, vdtype, result, left, mem);
                    return result;
                }

                VecR right = vec_subexpr(data.right);
                maybe_apply_mask_pushdown(cc, mask_pushdown);
                if (vdtype.is_int())
                    vec_int_cmp(cc, data.op, vdtype, data.is_unsigned, result, left, right);
                else
                    vec_float_cmp(cc, data.op, vdtype, result, left, right);
                return result;
            }
            SIMJIT_MATCH (StepKind::VecReduce) {
                VecDataType vdtype = data.arg->dtype.as_vec();
                x86::Vector vec = x86::vec_to_x86(vdtype);
                VecR arg_vec = vec_subexpr(data.arg);

                AnyR result = create_scalar_reg(step->dtype.as_scalar());
                // Note the codegen path for repaired float min/max is inefficient, but who cares

                // In XMM reductions we don't have a way to specify data type argument to vec_binary more preciely, but
                // that does not matter because we will extract lowest element anyway
                auto reduce_xmm_i32 = [&](const VecR &arg) {
                    if (vdtype.is_int()) {
                        VecR tmp = cc.new_vec128();
                        cc.vpunpckhqdq(tmp, arg, arg);
                        vec_binary(data.op, x86::types::XMMI32, tmp, arg, tmp);
                        VecR tmp1 = cc.new_vec128();
                        cc.vpshuflw(tmp1, tmp, 0x4e);
                        vec_binary(data.op, x86::types::XMMI32, tmp, tmp, tmp1);
                        cc.vmovd(result.as<GpR>(), tmp);
                    } else {
                        VecR tmp = cc.new_vec128();
                        cc.vpermilps(tmp, arg, 0x4e);
                        vec_binary(data.op, x86::types::XMMF32, tmp, tmp, arg);
                        VecR tmp1 = cc.new_vec128();
                        cc.vpermilps(tmp1, tmp, 0x55);
                        vec_binary(data.op, x86::types::XMMF32, tmp, tmp, tmp1);
                        VecR result_xmm = result.as<VecR>();
                        // mov only lowest f32
                        cc.vxorps(result_xmm, result_xmm, result_xmm);
                        cc.vmovss(result_xmm, result_xmm, tmp);
                    }
                };
                auto reduce_xmm_i64 = [&](const VecR &arg) {
                    if (vdtype.is_int()) {
                        VecR tmp = cc.new_vec128();
                        cc.vpunpckhqdq(tmp, arg, arg);
                        vec_binary(data.op, x86::types::XMMI64, tmp, arg, tmp);
                        cc.vmovq(result.as<GpR>(), tmp);
                    } else {
                        VecR tmp = cc.new_vec128();
                        cc.vunpckhpd(tmp, arg, arg);
                        vec_binary(data.op, x86::types::XMMF64, tmp, arg, tmp);
                        VecR result_xmm = result.as<VecR>();
                        cc.vxorpd(result_xmm, result_xmm, result_xmm);
                        cc.vmovsd(result_xmm, result_xmm, tmp);
                    }
                };

                auto reduce_ymm_i32 = [&](const VecR &arg) {
                    if (vdtype.is_int()) {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec128();
                        cc.vextracti32x4(tmp, arg, 1);
                        vec_binary(data.op, x86::types::XMMI32, tmp, low, tmp);
                        reduce_xmm_i32(tmp);
                    } else {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec128();
                        cc.vextractf32x4(tmp, arg, 1);
                        vec_binary(data.op, x86::types::XMMF32, tmp, low, tmp);
                        reduce_xmm_i32(tmp);
                    }
                };
                auto reduce_ymm_i64 = [&](const VecR &arg) {
                    if (vdtype.is_int()) {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec128();
                        cc.vextracti64x2(tmp, arg, 1);
                        vec_binary(data.op, x86::types::XMMI64, tmp, low, tmp);
                        reduce_xmm_i64(tmp);
                    } else {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec128();
                        cc.vextractf64x2(tmp, arg, 1);
                        vec_binary(data.op, x86::types::XMMF64, tmp, low, tmp);
                        reduce_xmm_i64(tmp);
                    }
                };
                auto reduce_zmm_i32 = [&](const VecR &arg) {
                    if (vdtype.is_int()) {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec256();
                        cc.vextracti32x8(tmp, arg, 1);
                        vec_binary(data.op, x86::types::YMMI32, tmp, low, tmp);
                        reduce_ymm_i32(tmp);
                    } else {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec256();
                        cc.vextractf32x8(tmp, arg, 1);
                        vec_binary(data.op, x86::types::YMMF32, tmp, low, tmp);
                        reduce_ymm_i32(tmp);
                    }
                };
                auto reduce_zmm_i64 = [&](const VecR &arg) {
                    if (vdtype.is_int()) {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec256();
                        cc.vextracti64x4(tmp, arg, 1);
                        vec_binary(data.op, x86::types::YMMI64, tmp, low, tmp);
                        reduce_ymm_i64(tmp);
                    } else {
                        VecR low = arg.half();
                        VecR tmp = cc.new_vec256();
                        cc.vextractf64x4(tmp, arg, 1);
                        vec_binary(data.op, x86::types::YMMF64, tmp, low, tmp);
                        reduce_ymm_i64(tmp);
                    }
                };

                switch (vec.reg) {
                case x86::VecRegisterKind::XMM:
                    if (vdtype.elem == VecElemType::I32 || vdtype.elem == VecElemType::F32) {
                        reduce_xmm_i32(arg_vec.xmm());
                    } else {
                        reduce_xmm_i64(arg_vec.xmm());
                    }
                    break;
                case x86::VecRegisterKind::YMM:
                    if (vdtype.elem == VecElemType::I32 || vdtype.elem == VecElemType::F32) {
                        reduce_ymm_i32(arg_vec.ymm());
                    } else {
                        reduce_ymm_i64(arg_vec.ymm());
                    }
                    break;
                case x86::VecRegisterKind::ZMM:
                    if (vdtype.elem == VecElemType::I32 || vdtype.elem == VecElemType::F32) {
                        reduce_zmm_i32(arg_vec.zmm());
                    } else {
                        reduce_zmm_i64(arg_vec.zmm());
                    }
                    break;
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::MaskReduce) {
                MaskDataType mdtype = data.arg->dtype.as_mask();
                MaskR arg = mask_subexpr(data.arg);
                GpR result;

                // We can use self-test versions for large masks, but that would still require special case
                // handling for small masks. To keep it simple always use generic path.
                auto test_with_ones = [&]() {
                    MaskR all_ones = create_mask_reg(mdtype);
                    init_mask_const(all_ones, mdtype, 1);

                    switch (mdtype) {
                    case MaskDataType::M2:
                    case MaskDataType::M4:
                    case MaskDataType::M8: cc.ktestb(arg, all_ones); break;
                    case MaskDataType::M16: cc.ktestw(arg, all_ones); break;
                    case MaskDataType::M32: cc.ktestd(arg, all_ones); break;
                    case MaskDataType::M64: cc.ktestq(arg, all_ones); break;
                    }
                };

                switch (data.op) {
                case PredicateBinaryOp::And:
                    test_with_ones();
                    // E !arg == 0 => A arg != 0
                    result = cc.new_gp8();
                    cc.setc(result.r8());
                    break;
                case PredicateBinaryOp::Or:
                    test_with_ones();
                    // !(A arg == 0) => E arg != 0
                    result = cc.new_gp8();
                    cc.setnz(result.r8());
                    break;
                case PredicateBinaryOp::Xor: {
                    result = create_int_reg(mask_dtype_to_scalar(mdtype));
                    mask_to_gp(cc, mdtype, result, arg);
                    cc.popcnt(result, result);
                    cc.and_(result, 1);
                    break;
                }
                case PredicateBinaryOp::AndNot:
                case PredicateBinaryOp::XNor: messed_up("invalid mask reduce op %s", show_predicate_binary_op(data.op));
                }
                return result.r8();
            }
            SIMJIT_MATCH (StepKind::AccLoad) {
                require_acc_initialized(data);
                AnyR acc_reg = accs[mir_func->accs.index(data)];
                return acc_reg;
            }
            SIMJIT_MATCH (StepKind::AccStore) {
                AnyR acc_reg = accs[mir_func->accs.index(data.acc)];

                if (SIMJIT_X64_ASMJIT_INLINE_ACC_INIT && data.arg->is(StepKind::Const)) {
                    // inline const to avoid machine code growing excessively
                    const ConstData &x = data.arg->step_data<StepKind::Const>();
                    if (step->dtype.is_mask()) {
                        init_mask_const(acc_reg.as<MaskR>(), step->dtype.as_mask(), x.as_signed());
                    } else if (step->dtype.is_vec()) {
                        init_vec_const(acc_reg.as<VecR>(), step->dtype.as_vec(), x);
                    } else {
                        messed_up("accumulator const initialization expects mask or vector type, got %s",
                                  show_dtype(step->dtype));
                    }
                    mark_acc_initialized(data.acc);
                    return {};
                }
                if (SIMJIT_X64_ASMJIT_INLINE_ACC_BIN && data.arg->is(StepKind::ArithBinary) && step->dtype.is_vec()) {
                    VecDataType vdtype = step->dtype.as_vec();
                    // Optimize common case where we load + bin + store acc
                    const auto &bin_data = data.arg->step_data<StepKind::ArithBinary>();
                    if (bin_data.left->is(StepKind::AccLoad) &&
                        bin_data.left->step_data<StepKind::AccLoad>() == data.acc) {
                        require_acc_initialized(data.acc);
                        VecR a = acc_reg.as<VecR>();
                        vec_arith_binary(vdtype, bin_data.op, a, bin_data.left, bin_data.right, true, {});
                        mark_acc_initialized(data.acc);
                        return {};
                    }
                }

                if (SIMJIT_X64_ASMJIT_INLINE_ACC_BIN && data.arg->is(StepKind::FMA) && step->dtype.is_vec() &&
                    refcounts[data.arg->id] == 1) {
                    VecDataType vdtype = step->dtype.as_vec();
                    const auto &fma_data = data.arg->step_data<StepKind::FMA>();
                    if (fma_data.x3->is(StepKind::AccLoad) && fma_data.x3->step_data<StepKind::AccLoad>() == data.acc &&
                        refcounts[fma_data.x3->id] == 1) {
                        require_acc_initialized(data.acc);
                        VecR acc = acc_reg.as<VecR>();
                        if (fma_data.x2->is(StepKind::Load) && should_inline_mem(fma_data.x2)) {
                            VecR x1 = vec_subexpr(fma_data.x1);
                            const auto &x2_data = fma_data.x2->step_data<StepKind::Load>();
                            ax86::Mem x2 = vec_load_mem(x2_data.addr, fma_data.x2->dtype.as_vec());
                            vec_fma_acc(cc, fma_data.kind, vdtype.elem, acc, x1, x2);
                        } else if (fma_data.x1->is(StepKind::Load) && should_inline_mem(fma_data.x1)) {
                            VecR x2 = vec_subexpr(fma_data.x2);
                            const auto &x1_data = fma_data.x1->step_data<StepKind::Load>();
                            ax86::Mem x1 = vec_load_mem(x1_data.addr, fma_data.x1->dtype.as_vec());
                            vec_fma_acc(cc, fma_data.kind, vdtype.elem, acc, x2, x1);
                        } else {
                            VecR x1 = vec_subexpr(fma_data.x1);
                            VecR x2 = vec_subexpr(fma_data.x2);
                            vec_fma_acc(cc, fma_data.kind, vdtype.elem, acc, x1, x2);
                        }
                        mark_acc_initialized(data.acc);
                        return {};
                    }
                }

                if (SIMJIT_X64_ASMJIT_INLINE_ACC_BIN && SIMJIT_X64_ASMJIT_MASK_PUSHDOWN &&
                    data.arg->is(StepKind::Select) && step->dtype.is_vec()) {
                    VecDataType vdtype = step->dtype.as_vec();
                    // Optimize less common case where we have conditional update of accumulator like in sum_if
                    auto &select_data = data.arg->step_data<StepKind::Select>();
                    if (select_data.truthy->is(StepKind::ArithBinary) && select_data.falsy->is(StepKind::AccLoad) &&
                        select_data.falsy->step_data<StepKind::AccLoad>() == data.acc) {
                        const auto &bin_data = select_data.truthy->step_data<StepKind::ArithBinary>();
                        // note simplified step_has_mask_form condition, not step_supports_mask_pushdown
                        if (bin_data.left->is(StepKind::AccLoad) &&
                            bin_data.left->step_data<StepKind::AccLoad>() == data.acc &&
                            step_has_mask_form(select_data.truthy)) {
                            require_acc_initialized(data.acc);
                            VecR a = acc_reg.as<VecR>();
                            MaskPushdownInfo pushdown{mask_subexpr(select_data.cond), a, select_data.falsy};
                            vec_arith_binary(vdtype, bin_data.op, a, bin_data.left, bin_data.right, true, &pushdown);
                            mark_acc_initialized(data.acc);
                            return {};
                        }
                    }
                }

                AnyR arg = subexpr(data.arg);
                if (acc_reg != arg) {
                    if (step->dtype.is_mask()) {
                        MaskR a = acc_reg.as<MaskR>();
                        MaskR b = arg.as<MaskR>();
                        copy_mask(cc, step->dtype.as_mask(), a, b);
                    } else if (step->dtype.is_vec()) {
                        VecElemType elem = step->dtype.as_vec().elem;
                        VecR a = acc_reg.as<VecR>();
                        VecR b = arg.as<VecR>();
                        switch (elem) {
                        case VecElemType::I8:
                        case VecElemType::I16:
                        case VecElemType::I32:
                        case VecElemType::I64: cc.vmovdqa64(a, b); break;
                        case VecElemType::F32: cc.vmovaps(a, b); break;
                        case VecElemType::F64: cc.vmovapd(a, b); break;
                        }
                    } else {
                        messed_up("accumulator store expects mask or vector type, got %s", show_dtype(step->dtype));
                    }
                }
                mark_acc_initialized(data.acc);
                return {};
            }
            SIMJIT_MATCH2 (StepKind::VecWidenLowHalf, StepKind::VecFloatWidenLowHalf) {
                VecDataType arg_type = data.arg->dtype.as_vec();
                VecDataType vdtype = step->dtype.as_vec();
                VecR dst = create_vec_reg(vdtype);
                VecR arg = vec_subexpr(data.arg);
                if (vdtype.is_float()) {
                    cc.vcvtps2pd(dst, arg);
                } else if (data.is_unsigned) {
                    vec_zext(cc, arg_type.elem, vdtype.elem, dst, arg);
                } else {
                    vec_sext(cc, arg_type.elem, vdtype.elem, dst, arg);
                }
                return dst;
            }
            SIMJIT_MATCH2 (StepKind::VecWidenHighHalf, StepKind::VecFloatWidenHighHalf) {
                VecDataType arg_type = data.arg->dtype.as_vec();
                VecDataType vdtype = step->dtype.as_vec();
                x86::Vector vec = x86::vec_to_x86(vdtype);
                VecR dst = create_vec_reg(vdtype);
                VecR arg = vec_subexpr(data.arg);
                switch (vec.reg) {
                case x86::VecRegisterKind::XMM:
                    if (vdtype.is_int()) {
                        cc.vpsrldq(dst.xmm(), arg.xmm(), 8);
                    } else {
                        cc.vmovhlps(dst.xmm(), arg.xmm(), arg.xmm());
                    }
                    arg = dst;
                    break;
                case x86::VecRegisterKind::YMM:
                    if (vdtype.is_int()) {
                        cc.vextracti32x4(dst.xmm(), arg, 1);
                    } else {
                        cc.vextractf32x4(dst.xmm(), arg, 1);
                    }
                    arg = dst;
                    break;
                case x86::VecRegisterKind::ZMM:
                    if (vdtype.is_int()) {
                        cc.vextracti32x8(dst.ymm(), arg, 1);
                    } else {
                        cc.vextractf32x8(dst.ymm(), arg, 1);
                    }
                    arg = dst;
                    break;
                }
                if (vdtype.is_float()) {
                    cc.vcvtps2pd(dst, arg);
                } else if (data.is_unsigned) {
                    vec_zext(cc, arg_type.elem, vdtype.elem, dst, arg);
                } else {
                    vec_sext(cc, arg_type.elem, vdtype.elem, dst, arg);
                }
                return dst;
            }
            SIMJIT_MATCH (StepKind::VecNarrowCombine) {
                VecDataType vdtype = step->dtype.as_vec();
                x86::VecRegisterKind reg = x86::vec_to_x86(vdtype).reg;
                VecR low = vec_subexpr(data.low);
                VecR high = vec_subexpr(data.high);
                VecR result = create_vec_reg(vdtype);
                switch (reg) {
                case x86::VecRegisterKind::XMM: {
                    VecR high_narrow = create_vec_reg(vdtype);
                    vec_trunc(cc, data.low->dtype.as_vec().elem, vdtype.elem, result.xmm(), low);
                    vec_trunc(cc, data.high->dtype.as_vec().elem, vdtype.elem, high_narrow.xmm(), high);
                    cc.vpunpcklqdq(result.xmm(), result.xmm(), high_narrow.xmm());
                    break;
                }
                case x86::VecRegisterKind::YMM: {
                    auto maybe_half_dtype = vec_dtype_half(vdtype);
                    SIMJIT_ASSERT(maybe_half_dtype.has_value());
                    VecDataType half_dtype = *maybe_half_dtype;
                    VecR high_narrow = create_vec_reg(half_dtype);
                    vec_trunc(cc, data.high->dtype.as_vec().elem, vdtype.elem, high_narrow, high);
                    vec_trunc(cc, data.low->dtype.as_vec().elem, vdtype.elem, result.xmm(), low);
                    cc.vinserti128(result.ymm(), result.ymm(), high_narrow.xmm(), 1);
                    break;
                }
                case x86::VecRegisterKind::ZMM: {
                    auto maybe_half_dtype = vec_dtype_half(vdtype);
                    SIMJIT_ASSERT(maybe_half_dtype.has_value());
                    VecDataType half_dtype = *maybe_half_dtype;
                    VecR high_narrow = create_vec_reg(half_dtype);
                    vec_trunc(cc, data.high->dtype.as_vec().elem, vdtype.elem, high_narrow, high);
                    vec_trunc(cc, data.low->dtype.as_vec().elem, vdtype.elem, result.ymm(), low);
                    cc.vinserti64x4(result.zmm(), result.zmm(), high_narrow.ymm(), 1);
                    break;
                }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::VecFloatNarrowCombine) {
                VecDataType vdtype = step->dtype.as_vec();
                VecDataType wide_type = data.low->dtype.as_vec();
                if (vdtype.elem != VecElemType::F32 || wide_type.elem != VecElemType::F64) {
                    messed_up("unsupported float narrow combine %s <- %s", show_vec_dtype(vdtype),
                              show_vec_dtype(wide_type));
                }

                VecR low = vec_subexpr(data.low);
                VecR high = vec_subexpr(data.high);
                VecR result = create_vec_reg(vdtype);
                switch (x86::vec_to_x86(vdtype).reg) {
                case x86::VecRegisterKind::XMM: {
                    VecR high_narrow = create_vec_reg(vdtype);
                    vec_float_cast(cc, VecElemType::F64, VecElemType::F32, false, result.xmm(), low);
                    vec_float_cast(cc, VecElemType::F64, VecElemType::F32, false, high_narrow.xmm(), high);
                    cc.vmovlhps(result.xmm(), result.xmm(), high_narrow.xmm());
                    break;
                }
                case x86::VecRegisterKind::YMM: {
                    auto maybe_half_dtype = vec_dtype_half(vdtype);
                    SIMJIT_ASSERT(maybe_half_dtype.has_value());
                    VecDataType half_dtype = *maybe_half_dtype;
                    VecR high_narrow = create_vec_reg(half_dtype);
                    vec_float_cast(cc, VecElemType::F64, VecElemType::F32, false, high_narrow, high);
                    vec_float_cast(cc, VecElemType::F64, VecElemType::F32, false, result.xmm(), low);
                    cc.vinsertf128(result.ymm(), result.ymm(), high_narrow.xmm(), 1);
                    break;
                }
                case x86::VecRegisterKind::ZMM: {
                    auto maybe_half_dtype = vec_dtype_half(vdtype);
                    SIMJIT_ASSERT(maybe_half_dtype.has_value());
                    VecDataType half_dtype = *maybe_half_dtype;
                    VecR high_narrow = create_vec_reg(half_dtype);
                    vec_float_cast(cc, VecElemType::F64, VecElemType::F32, false, high_narrow, high);
                    vec_float_cast(cc, VecElemType::F64, VecElemType::F32, false, result.ymm(), low);
                    cc.vinsertf32x8(result.zmm(), result.zmm(), high_narrow.ymm(), 1);
                    break;
                }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::MaskBinary) {
                MaskDataType mdtype = step->dtype.as_mask();
                auto try_pushdown_into_compare = [&](Step *cmp_step, Step *mask_step) -> std::optional<MaskR> {
                    if (!cmp_step->is(StepKind::Compare)) { return std::nullopt; }
                    if (refcounts[cmp_step->id] > 1) { return std::nullopt; }
                    MaskPushdownInfo pushdown{mask_subexpr(mask_step)};
                    return mask_subexpr(cmp_step, &pushdown);
                };

                if (SIMJIT_X64_ASMJIT_MASK_PUSHDOWN && data.op == PredicateBinaryOp::And) {
                    if (auto result = try_pushdown_into_compare(data.left, data.right)) { return *result; }
                    if (auto result = try_pushdown_into_compare(data.right, data.left)) { return *result; }
                }

                MaskR left = mask_subexpr(data.left);
                MaskR right = mask_subexpr(data.right);
                MaskR result = create_mask_reg(mdtype);
                switch (data.op) {
                case PredicateBinaryOp::And: mask_and(cc, mdtype, result, left, right); break;
                case PredicateBinaryOp::Or:
                    switch (mdtype) {
                    case MaskDataType::M2:
                    case MaskDataType::M4:
                    case MaskDataType::M8: cc.korb(result, left, right); break;
                    case MaskDataType::M16: cc.korw(result, left, right); break;
                    case MaskDataType::M32: cc.kord(result, left, right); break;
                    case MaskDataType::M64: cc.korq(result, left, right); break;
                    }
                    break;
                case PredicateBinaryOp::Xor:
                    switch (mdtype) {
                    case MaskDataType::M2:
                    case MaskDataType::M4:
                    case MaskDataType::M8: cc.kxorb(result, left, right); break;
                    case MaskDataType::M16: cc.kxorw(result, left, right); break;
                    case MaskDataType::M32: cc.kxord(result, left, right); break;
                    case MaskDataType::M64: cc.kxorq(result, left, right); break;
                    }
                    break;
                case PredicateBinaryOp::AndNot:
                    switch (mdtype) {
                    case MaskDataType::M2:
                    case MaskDataType::M4:
                    case MaskDataType::M8: cc.kandnb(result, left, right); break;
                    case MaskDataType::M16: cc.kandnw(result, left, right); break;
                    case MaskDataType::M32: cc.kandnd(result, left, right); break;
                    case MaskDataType::M64: cc.kandnq(result, left, right); break;
                    }
                    break;
                case PredicateBinaryOp::XNor:
                    switch (mdtype) {
                    case MaskDataType::M2:
                    case MaskDataType::M4:
                    case MaskDataType::M8: cc.kxnorb(result, left, right); break;
                    case MaskDataType::M16: cc.kxnorw(result, left, right); break;
                    case MaskDataType::M32: cc.kxnord(result, left, right); break;
                    case MaskDataType::M64: cc.kxnorq(result, left, right); break;
                    }
                    canonicalize_partial_byte_mask(result, mdtype);
                    break;
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::MaskCount) {
                MaskR mask = mask_subexpr(data);
                GpR result = create_int_reg(step->dtype.as_scalar());
                mask_to_gp(cc, data->dtype.as_mask(), result, mask);
                cc.popcnt(result, result);
                return result;
            }
            SIMJIT_MATCH (StepKind::MaskCombine) {
                MaskDataType mdtype = step->dtype.as_mask();
                MaskR left = mask_subexpr(data.left);
                MaskR right = mask_subexpr(data.right);
                // ordering is swapped for cpu instructions
                std::swap(left, right);
                MaskR result = create_mask_reg(mdtype);
                switch (mdtype) {
                case MaskDataType::M2: messed_up("invalid MaskCombine");
                case MaskDataType::M4:
                    cc.kmovb(result, left);
                    cc.kshiftlb(result, result, 2);
                    cc.korb(result, result, right);
                    break;
                case MaskDataType::M8:
                    cc.kmovb(result, left);
                    cc.kshiftlb(result, result, 4);
                    cc.korb(result, result, right);
                    break;
                case MaskDataType::M16: cc.kunpckbw(result, left, right); break;
                case MaskDataType::M32: cc.kunpckwd(result, left, right); break;
                case MaskDataType::M64: cc.kunpckdq(result, left, right); break;
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::PredicateNot) {
                MaskDataType mdtype = step->dtype.as_mask();
                MaskR mask = mask_subexpr(data);
                MaskR result = create_mask_reg(mdtype);
                switch (mdtype) {
                case MaskDataType::M2:
                case MaskDataType::M4:
                case MaskDataType::M8: cc.knotb(result, mask); break;
                case MaskDataType::M16: cc.knotw(result, mask); break;
                case MaskDataType::M32: cc.knotd(result, mask); break;
                case MaskDataType::M64: cc.knotq(result, mask); break;
                }
                canonicalize_partial_byte_mask(result, mdtype);
                return result;
            }
            SIMJIT_MATCH (StepKind::Select) {
                VecDataType vdtype = step->dtype.as_vec();
                const Step *truthy_step = data.truthy;
                const Step *falsy_step = data.falsy;
                MaskR cond = mask_subexpr(data.cond);

                if (SIMJIT_X64_ASMJIT_ZEROBLEND && step_is_zero(falsy_step)) {
                    if (vdtype.is_int() && truthy_step->is(StepKind::Const) &&
                        truthy_step->step_data<StepKind::Const>().as_unsigned() ==
                            scalar_dtype_umax(vec_elem_to_scalar(vdtype.elem))) {
                        VecR result = create_vec_reg(vdtype);
                        switch (vdtype.elem) {
                            INVALID_FLOAT_CASES;
                        case VecElemType::I8: cc.vpmovm2b(result, cond); break;
                        case VecElemType::I16: cc.vpmovm2w(result, cond); break;
                        case VecElemType::I32: cc.vpmovm2d(result, cond); break;
                        case VecElemType::I64: cc.vpmovm2q(result, cond); break;
                        }
                        return result;
                    }
                    if (SIMJIT_X64_ASMJIT_MASK_PUSHDOWN && step_supports_mask_pushdown(truthy_step)) {
                        MaskPushdownInfo pushdown{cond};
                        return vec_subexpr(truthy_step, &pushdown);
                    }
                    VecR result = create_zero_vec_reg(vdtype);
                    VecR truthy = vec_subexpr(truthy_step);
                    switch (vdtype.elem) {
                    case VecElemType::I8: cc.k(cond).z().vmovdqu8(result, truthy); break;
                    case VecElemType::I16: cc.k(cond).z().vmovdqu16(result, truthy); break;
                    case VecElemType::I32: cc.k(cond).z().vmovdqa32(result, truthy); break;
                    case VecElemType::I64: cc.k(cond).z().vmovdqa64(result, truthy); break;
                    case VecElemType::F32: cc.k(cond).z().vmovaps(result, truthy); break;
                    case VecElemType::F64: cc.k(cond).z().vmovapd(result, truthy); break;
                    }
                    return result;
                }

                VecR falsy = vec_subexpr(falsy_step);
                if (SIMJIT_X64_ASMJIT_MASK_PUSHDOWN && step_supports_mask_pushdown(truthy_step)) {
                    MaskPushdownInfo pushdown{cond, falsy, falsy_step};
                    return vec_subexpr(truthy_step, &pushdown);
                }
                // xor because otherwise asmjit liveness analysis goes crazy, it is not actually needed
                VecR result = create_zero_vec_reg(vdtype);

                // Here we can add special case when truthy is load, but it will never be hit because mask pushdown
                // handles it
                VecR truthy = vec_subexpr(truthy_step);
                switch (vdtype.elem) {
                case VecElemType::I8: cc.k(cond).vpblendmb(result, falsy, truthy); break;
                case VecElemType::I16: cc.k(cond).vpblendmw(result, falsy, truthy); break;
                case VecElemType::I32: cc.k(cond).vpblendmd(result, falsy, truthy); break;
                case VecElemType::I64: cc.k(cond).vpblendmq(result, falsy, truthy); break;
                case VecElemType::F32: cc.k(cond).vblendmps(result, falsy, truthy); break;
                case VecElemType::F64: cc.k(cond).vblendmpd(result, falsy, truthy); break;
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::VecIndex) {
                VecDataType dtype = step->dtype.as_vec();
                require_acc_initialized(data.acc);
                VecR acc = accs[mir_func->accs.index(data.acc)].as<VecR>();
                VecR inc = vec_subexpr(data.inc);
                VecR result = create_vec_reg(dtype);
                cc.vmovdqa64(result, acc);
                vec_binary(ArithBinaryOp::Add, dtype, acc, acc, inc);
                return result;
            }
            SIMJIT_MATCH (StepKind::Pack) {
                ArgInfo &info = args[data.dst];
                MaskDataType mdtype = data.cond->dtype.as_mask();
                VecDataType dtype = step->dtype.as_vec();
                require_acc_initialized(data.acc);
                GpR acc = accs[mir_func->accs.index(data.acc)].as<GpR>();
                VecR arg = vec_subexpr(data.arg);
                MaskR cond = mask_subexpr(data.cond);
                // Xor because otherwise asmjit liveness analysis goes crazy, it is not actually needed
                VecR result = create_zero_vec_reg(dtype);
                switch (dtype.elem) {
                case VecElemType::I8: cc.k(cond).vpcompressb(result, arg); break;
                case VecElemType::I16: cc.k(cond).vpcompressw(result, arg); break;
                case VecElemType::I32: cc.k(cond).vpcompressd(result, arg); break;
                case VecElemType::I64: cc.k(cond).vpcompressq(result, arg); break;
                case VecElemType::F32: cc.k(cond).vcompressps(result, arg); break;
                case VecElemType::F64: cc.k(cond).vcompresspd(result, arg); break;
                }
                ax86::Mem off = vec_mem_offset(info.gp, acc, 0, dtype);
                switch (dtype.elem) {
                case VecElemType::I8:
                case VecElemType::I16:
                case VecElemType::I32:
                case VecElemType::I64: cc.vmovdqu64(off, result); break;
                case VecElemType::F32: cc.vmovups(off, result); break;
                case VecElemType::F64: cc.vmovupd(off, result); break;
                }

                GpR popcnt_gp = mdtype == MaskDataType::M64 ? cc.new_gp64() : cc.new_gp32();
                mask_to_gp(cc, mdtype, popcnt_gp, cond);
                cc.popcnt(popcnt_gp, popcnt_gp); // result is sign-extended to 64 if 32
                cc.add(acc, popcnt_gp.r64());
                return {};
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                VecDataType vdtype = step->dtype.as_vec();
                const ArgInfo &arg_info = args[data.dst];
                VecR arg = vec_subexpr(data.arg);
                VecR idx = vec_subexpr(data.idx);
                MaskR k = cc.new_k64();
                cc.kxnorq(k, k, k);
                if (data.idx->dtype.as_vec().elem == VecElemType::I32) {
                    switch (vdtype.elem) {
                    case VecElemType::I8:
                    case VecElemType::I16: unsupported("Do not support i8/i16 scatter");
                    case VecElemType::I32: cc.k(k).vpscatterdd(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::I64: cc.k(k).vpscatterdq(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    case VecElemType::F32: cc.k(k).vscatterdps(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::F64: cc.k(k).vscatterdpd(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    }
                } else {
                    switch (vdtype.elem) {
                    case VecElemType::I8:
                    case VecElemType::I16: unsupported("Do not support i8/i16 scatter");
                    case VecElemType::I32: cc.k(k).vpscatterqd(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::I64: cc.k(k).vpscatterqq(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    case VecElemType::F32: cc.k(k).vscatterqps(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::F64: cc.k(k).vscatterqpd(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    }
                }
                return {};
            }
            SIMJIT_MATCH (StepKind::CondScatter) {
                VecDataType vdtype = step->dtype.as_vec();
                const ArgInfo &arg_info = args[data.dst];
                VecR arg = vec_subexpr(data.arg);
                VecR idx = vec_subexpr(data.idx);
                MaskR k = mask_subexpr(data.cond);
                MaskR k_tmp = cc.new_k64();
                cc.kmovq(k_tmp, k);
                if (data.idx->dtype.as_vec().elem == VecElemType::I32) {
                    switch (vdtype.elem) {
                    case VecElemType::I8:
                    case VecElemType::I16: messed_up("Do not support i8/i16 scatter");
                    case VecElemType::I32: cc.k(k).vpscatterdd(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::I64: cc.k(k).vpscatterdq(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    case VecElemType::F32: cc.k(k).vscatterdps(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::F64: cc.k(k).vscatterdpd(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    }
                } else {
                    switch (vdtype.elem) {
                    case VecElemType::I8:
                    case VecElemType::I16: messed_up("Do not support i8/i16 scatter");
                    case VecElemType::I32: cc.k(k).vpscatterqd(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::I64: cc.k(k).vpscatterqq(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    case VecElemType::F32: cc.k(k).vscatterqps(ax86::ptr(arg_info.gp, idx, 2), arg); break;
                    case VecElemType::F64: cc.k(k).vscatterqpd(ax86::ptr(arg_info.gp, idx, 3), arg); break;
                    }
                }
                cc.kmovq(k, k_tmp);
                return {};
            }
            SIMJIT_MATCH (StepKind::Ternarylogic) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR a = vec_subexpr(data.a);
                VecR b = vec_subexpr(data.b);
                VecR c = vec_subexpr(data.c);

                VecR result = create_vec_reg(vdtype);
                cc.vmovdqa64(result, a);

                cc.vpternlogq(result, b, c, data.fun);
                return result;
            }
            SIMJIT_MATCH (StepKind::CondStore) {
                VecDataType vdtype = step->dtype.as_vec();
                ax86::Mem mem = vec_load_mem(data.addr, vdtype);
                MaskR cond = mask_subexpr(data.cond);
                if (SIMJIT_X64_ASMJIT_INLINE_VEC_CAST && data.arg->is(StepKind::IntCast)) {
                    auto &cast_data = data.arg->step_data<StepKind::IntCast>();
                    VecElemType from = cast_data.arg->dtype.as_vec().elem;
                    VecElemType to = data.arg->dtype.as_vec().elem;
                    if (cast_data.kind == IntCastKind::Trunc) {
                        VecR arg = vec_subexpr(cast_data.arg);
                        vec_trunc(cc.k(cond), from, to, mem, arg);
                        return {};
                    }
                }
                VecR arg = vec_subexpr(data.arg);
                if (data.kind == LoadStoreKind::Aligned) {
                    switch (vdtype.elem) {
                    case VecElemType::I8: cc.k(cond).vmovdqu8(mem, arg); break;
                    case VecElemType::I16: cc.k(cond).vmovdqu16(mem, arg); break;
                    case VecElemType::I32: cc.k(cond).vmovdqa32(mem, arg); break;
                    case VecElemType::I64: cc.k(cond).vmovdqa64(mem, arg); break;
                    case VecElemType::F32: cc.k(cond).vmovaps(mem, arg); break;
                    case VecElemType::F64: cc.k(cond).vmovapd(mem, arg); break;
                    }
                } else {
                    switch (vdtype.elem) {
                    case VecElemType::I8: cc.k(cond).vmovdqu8(mem, arg); break;
                    case VecElemType::I16: cc.k(cond).vmovdqu16(mem, arg); break;
                    case VecElemType::I32: cc.k(cond).vmovdqu32(mem, arg); break;
                    case VecElemType::I64: cc.k(cond).vmovdqu64(mem, arg); break;
                    case VecElemType::F32: cc.k(cond).vmovups(mem, arg); break;
                    case VecElemType::F64: cc.k(cond).vmovupd(mem, arg); break;
                    }
                }
                return {};
            }
            SIMJIT_MATCH (StepKind::VecPermute) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = vec_subexpr(data.arg);
                VecR idxs = vec_subexpr(data.permute_idxs);
                VecR result = create_vec_reg(vdtype);
                if (data.is_bit) {
                    cc.vgf2p8affineqb(result, arg, idxs, 0);
                } else {
                    // TODO: Use type-specific versions
                    cc.vpshufb(result, arg, idxs);
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Fpclass) {
                VecDataType vdtype = data.arg->dtype.as_vec();
                MaskDataType mdtype = step->dtype.as_mask();
                MaskR result = create_mask_reg(mdtype);
                VecR arg = vec_subexpr(data.arg);

                ax86::VFPClassImm imm = make_fpclass_imm(data.flags);

                if (vdtype.elem == VecElemType::F32) {
                    cc.vfpclassps(result, arg, imm);
                } else {
                    cc.vfpclasspd(result, arg, imm);
                }
                return result;
            }
        }
        SIMJIT_UNREACHABLE();
    }

    void compile_steps(nonstd::span<Step *const> steps) {
        clear_shifted_counters();
        for (const Step *root : steps) {
            subexpr(root);
        }
    }

    bool prologue_root_is_delayed(Step *root) const {
        if (!SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT) { return false; }
        if (!has_main_loop) { return false; }
        if (!root->is(StepKind::AccStore)) { return false; }
        if (!root->dtype.is_scalar()) { return false; }
        AccId acc = root->step_data<StepKind::AccStore>().acc;
        if (main_loop_acc_uses[mir_func->accs.index(acc)]) { return false; }
        SIMJIT_ASSERT(SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT);
        return true;
    }

    bool has_delayed_prologue_steps(nonstd::span<Step *const> steps) const {
        for (Step *root : steps) {
            if (prologue_root_is_delayed(root)) { return true; }
        }
        return false;
    }

    void compile_prologue_steps(nonstd::span<Step *const> steps, bool delayed = false) {
        if (delayed) { SIMJIT_ASSERT(SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT); }
        clear_shifted_counters();
        for (Step *root : steps) {
            if (prologue_root_is_delayed(root) != delayed) { continue; }
            if (const_is_folded_root(root)) { continue; }
            subexpr(root);
        }
    }

    void emit_bit_luts() {
        for (const auto &[permute, label] : bit_permute_luts) {
            cc.align(aj::AlignMode::kData, 16);
            cc.bind(label);
            std::vector<uint8_t> data = generate_bit_permute_lut(permute);
            cc.embed(data.data(), data.size());
        }
    }

    void record_refcounts(nonstd::span<Step *const> steps, bool count_folded_consts = false,
                          bool count_main_loop_acc_uses = false) {
        auto add_refcounts = [&](Step *s) { bump_refcount(refcounts[s->id]); };
        traverse_steps_postorder_unique(refcounts.size(), steps, [&](Step *x) {
            step_recurse(x, add_refcounts);
            if (count_main_loop_acc_uses) {
                if (x->is(StepKind::AccLoad)) {
                    main_loop_acc_uses[mir_func->accs.index(x->step_data<StepKind::AccLoad>())] = 1;
                } else if (x->is(StepKind::AccStore)) {
                    main_loop_acc_uses[mir_func->accs.index(x->step_data<StepKind::AccStore>().acc)] = 1;
                } else if (x->is(StepKind::VecIndex)) {
                    main_loop_acc_uses[mir_func->accs.index(x->step_data<StepKind::VecIndex>().acc)] = 1;
                } else if (x->is(StepKind::Pack)) {
                    main_loop_acc_uses[mir_func->accs.index(x->step_data<StepKind::Pack>().acc)] = 1;
                } else if (x->is(StepKind::ScalarArithBinaryOverflow)) {
                    main_loop_acc_uses[mir_func->accs.index(
                        x->step_data<StepKind::ScalarArithBinaryOverflow>().overflow_flag)] = 1;
                }
            }
            if (count_folded_consts) { record_folded_const_ref(x); }
        });
    }

    void init(const mir::Function *func) {
        mir_func = func;
        step_map = arena->alloc_array<AnyR>(func->step_id_count);
        refcounts = arena->alloc_array<uint16_t>(func->step_id_count);
        args = arena->alloc_array<ArgInfo>(func->args.size());
        accs = arena->alloc_array<AnyR>(func->accs.count);
        folded_const_refcounts = arena->alloc_array<uint16_t>(func->step_id_count);
        main_loop_acc_uses = arena->alloc_array<uint8_t>(func->accs.count);
        acc_initialized = arena->alloc_array<uint8_t>(func->accs.count);
        for (uint8_t &x : main_loop_acc_uses) {
            x = 0;
        }
        for (uint8_t &x : acc_initialized) {
            x = 0;
        }

        row_count = cc.new_gp64("n");
        counter = cc.new_gp64("i");
        if (!func->remainder_roots.empty()) { remainder_label = cc.new_named_label("remainder"); }
        end_label = cc.new_named_label("end");

        has_main_loop = !func->main_loop_roots.empty();
        if (!func->main_loop_roots.empty()) record_refcounts(func->main_loop_roots, true, true);
        if (!func->remainder_roots.empty()) record_refcounts(func->remainder_roots, true);

        spill_epilogue_args = !func->main_loop_roots.empty();
    }

    void init_args(nonstd::span<ArgumentDecl const> func_args, aj::FuncNode *func_node) {
        func_node->set_arg(0, row_count);

        for (const ArgumentDecl &arg : func_args) {
            GpR gp = cc.new_gp64();
            ArgInfo info{&arg, gp};
            func_node->set_arg(arg.idx + 1, gp);
            args[arg.idx] = std::move(info);

            // Attempt to help asmjit in register allocation - it loads all arguments to registers, even if they are
            // needed only much later. Here we manually spill them to stack, and restore in the epilogue. If we Do not
            // do it, it starts spilling register that are used in remainder loop, which results in spills happening
            // inside loop. Spilling arguments is essentially free, however. This results in a little stupid code if
            // these arguments came from stack, where we load them from stack just to put back again. But this should
            // not matter in performance sense, mostly because we do this stuff only once, and have no dependencies on
            // these values.
            if (spill_epilogue_args && (arg.kind == ArgumentKind::DstSafetyCheck || arg.kind == ArgumentKind::DstAgg)) {
                ax86::Mem mem = cc.new_stack(8, 8);
                cc.mov(mem, gp);
                args[arg.idx].spilled = mem;
            }
        }
    }

    void init_accs(nonstd::span<Step *const> prologue_roots) {
        for (const Step *step : prologue_roots) {
            if (!step->is(StepKind::AccStore)) continue;
            const auto &store = step->step_data<StepKind::AccStore>();
            size_t idx = mir_func->accs.index(store.acc);
            switch (step->dtype.kind) {
            case DataTypeKind::Scalar: {
                ScalarDataType dtype = step->dtype.as_scalar();
                accs[idx] = create_scalar_reg(dtype);
                break;
            }
            case DataTypeKind::Vec: {
                VecDataType dtype = step->dtype.as_vec();
                accs[idx] = create_vec_reg(dtype);
                break;
            }
            case DataTypeKind::Mask: {
                MaskDataType dtype = step->dtype.as_mask();
                accs[idx] = create_mask_reg(dtype);
                break;
            }
            }
        }
    }

    void reload_spilled_arguments(nonstd::span<ArgumentDecl const> func_args) {
        if (!spill_epilogue_args) return;

        for (auto it = func_args.rbegin(); it < func_args.rend(); ++it) {
            const ArgumentDecl &arg = *it;
            if (arg.kind != ArgumentKind::DstSafetyCheck && arg.kind != ArgumentKind::DstAgg) { continue; }

            // Allocate new virtual register instead of using old one.
            // Asmjit register allocator is very happy about this approach.
            GpR reg = cc.new_gp64();
            cc.mov(reg, args[arg.idx].spilled);
            args[arg.idx].gp = reg;
        }
    }
};
} // namespace

static aj::FuncNode *create_func_node(size_t arg_count, ax86::Compiler &cc) {
    aj::FuncSignature signature{};
    signature.set_ret(aj::TypeId::kVoid);
    signature.add_arg(aj::TypeId::kUInt64);
    for (size_t i = 0; i < arg_count; ++i) {
        signature.add_arg(aj::TypeId::kUIntPtr);
    }

    aj::FuncNode *func_node = cc.add_func(signature);
    if (func_node == nullptr) {
        messed_up("Failed to create function node (probably code holder is not initialized correctly)");
    }
    return func_node;
}

static void compile_asmjit(const mir::Function *func, ax86::Compiler &cc) {
    aj::FuncNode *func_node = create_func_node(func->args.size(), cc);
    func_node->frame().set_avx_enabled();
    func_node->frame().set_avx512_enabled();
    func_node->frame().set_avx_cleanup();

    CompileState state{func->ctx->arena, cc};
    state.init(func);
    state.init_args(func->args, func_node);
    state.init_accs(func->prologue_roots);
    bool has_delayed_prologue = state.has_delayed_prologue_steps(func->prologue_roots);
    aj::Label delayed_prologue_label{};
    if (has_delayed_prologue) {
        SIMJIT_ASSERT(SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT);
        delayed_prologue_label = cc.new_named_label("delayed_prologue");
    }

    // Prologue

    state.compile_prologue_steps(func->prologue_roots);
    cc.xor_(state.counter, state.counter);
    cc.test(state.row_count, state.row_count);
    cc.je(has_delayed_prologue ? delayed_prologue_label : state.end_label);

    // Main loop

    if (!func->main_loop_roots.empty()) {
        aj::Label main_loop_label = cc.new_named_label("main_loop");
        if (!func->remainder_roots.empty()) {
            cc.cmp(state.row_count, func->loop_width);
            cc.jb(has_delayed_prologue ? delayed_prologue_label : state.remainder_label);
        }
        GpR last_vec_idx = cc.new_gp64();
        cc.mov(last_vec_idx, state.row_count);
        cc.and_(last_vec_idx, ~(func->loop_width - 1));
        // Align the code to favour uop cache
        cc.align(aj::AlignMode::kCode, 32);
        cc.bind(main_loop_label);
        state.compile_steps(func->main_loop_roots);
        cc.add(state.counter, func->loop_width);
        cc.cmp(state.counter, last_vec_idx);
        cc.jb(main_loop_label);
    }

    // Scalar remainder

    if (!func->remainder_roots.empty()) {
        if (!func->main_loop_roots.empty() && !has_delayed_prologue) {
            cc.cmp(state.counter, state.row_count);
            cc.je(state.end_label);
        }

        if (has_delayed_prologue) {
            SIMJIT_ASSERT(SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT);
            cc.bind(delayed_prologue_label);
            state.compile_prologue_steps(func->prologue_roots, true);
            cc.cmp(state.counter, state.row_count);
            cc.je(state.end_label);
        }

        cc.bind(state.remainder_label);
        state.compile_steps(func->remainder_roots);
        cc.inc(state.counter);
        cc.cmp(state.counter, state.row_count);
        cc.jne(state.remainder_label);
    } else if (has_delayed_prologue) {
        SIMJIT_ASSERT(SIMJIT_X64_ASMJIT_DELAY_SCALAR_ACC_INIT);
        cc.bind(delayed_prologue_label);
        state.compile_prologue_steps(func->prologue_roots, true);
    }

    // Epilogue

    cc.bind(state.end_label);
    state.reload_spilled_arguments(func->args);
    state.compile_steps(func->epilogue_roots);
    cc.ret();
    cc.end_func();
    state.emit_bit_luts();
    cc.finalize();
}

static void handle_compilation_result(asmjit::StringLogger *logger, asmjit::CodeHolder &code,
                                      const AsmjitCompileOptions &opts, AsmjitCompileResult &result) {
    if (opts.emit_asm_code && logger != nullptr) {
        result.asm_code = std::string{logger->data(), logger->data() + logger->data_size()};
    }
    if (opts.emit_machine_code) {
        for (auto *sec : code.sections_by_order()) {
            const auto &buffer = sec->buffer();
            result.machine_code.insert(result.machine_code.end(), buffer.begin(), buffer.end());
        }
    }
}

static void compile_asmjit(const mir::Function *func, const AsmjitCompileOptions &opts, AsmjitCompileResult &result) {
    struct {
        std::unique_ptr<aj::StringLogger> logger;
        std::unique_ptr<AsmjitSession> session;
    } tmp_state;

    ax86::Compiler *cc = nullptr;
    aj::CodeHolder *code = nullptr;

    if (opts.session != nullptr) {
        if (!is_x86_arch(opts.session->arch())) { messed_up("asmjit x86 compile requires x86 session"); }
        opts.session->reset();
        cc = static_cast<ax86::Compiler *>(&opts.session->compiler());
        code = &opts.session->code_holder();
    } else {
        tmp_state.session = std::make_unique<AsmjitSession>(func->ctx->arch);
        cc = static_cast<ax86::Compiler *>(&tmp_state.session->compiler());
        code = &tmp_state.session->code_holder();
    }

    if (opts.emit_asm_code) {
        tmp_state.logger = std::make_unique<aj::StringLogger>();
        code->set_logger(tmp_state.logger.get());
    } else if (code->logger() != nullptr) {
        code->reset_logger();
    }

    compile_asmjit(func, *cc);
    handle_compilation_result(tmp_state.logger.get(), *code, opts, result);
}

} // namespace asmjit_backend

void compile_asmjit_x86(const mir::Function *func, const AsmjitCompileOptions &opts, AsmjitCompileResult &result) {
    asmjit_backend::compile_asmjit(func, opts, result);
}

} // namespace simjit
