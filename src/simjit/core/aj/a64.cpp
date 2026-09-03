// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/asmjit.h"
#include "simjit/core/expr.h"
#include "simjit/core/mir.h"
#include "simjit/detail/base.h"
#include "simjit/simjit.h"

#include "asmjit/a64.h"

#define messed_up(...) simjit_exception(ErrorModule::A64, {}, {}, __VA_ARGS__)
#define unsupported(...) \
    simjit_exception(ErrorModule::A64, ErrorKind::Unsupported, ErrorSubKind::UnsupportedBackendFeature, __VA_ARGS__)

// See comment about similar macros in mir_compiler_asmjit.cpp.
// Note there is no option to disable special constants.
#define SIMJIT_A64_ASMJIT_CONST_OPS 1
#define SIMJIT_A64_ASMJIT_ZEROBLEND 1
#define SIMJIT_A64_ASMJIT_INDEX_REGS 1
#define SIMJIT_A64_ASMJIT_LDP 1
#define SIMJIT_A64_ASMJIT_STP 1
#define SIMJIT_A64_ASMJIT_INLINE_SCALAR_COND 1
#define SIMJIT_A64_ASMJIT_BLEND_SUB 1

namespace simjit {
namespace asmjit_backend {
namespace arm {
using namespace ::simjit::mir;

namespace aj = asmjit;
namespace aja64 = asmjit::a64;

using GpR = aja64::Gp;
using VecR = aja64::Vec;
using AnyR = aj::Reg;

static constexpr bool is_valid_mask_shift_writer_flush_dtype(ScalarDataType dtype) noexcept {
    return dtype == ScalarDataType::I8 || dtype == ScalarDataType::I16;
}

static constexpr ScalarDataType MASK_SHIFT_WRITER_FLUSH_DTYPE = ScalarDataType::I16;
static_assert(is_valid_mask_shift_writer_flush_dtype(MASK_SHIFT_WRITER_FLUSH_DTYPE));
static constexpr unsigned MASK_SHIFT_WRITER_FLUSH_BITS = scalar_dtype_bits(MASK_SHIFT_WRITER_FLUSH_DTYPE);
static constexpr int MASK_SHIFT_WRITER_FLUSH_BYTES = (int)scalar_dtype_size(MASK_SHIFT_WRITER_FLUSH_DTYPE);
static_assert(has_single_bit(MASK_SHIFT_WRITER_FLUSH_BITS));

namespace SpecialConstant {
enum : uint64_t {
    Bits,            // 0x8040201008040201ull, used in different mask ops
    Zero8One8,       // [0 x8, 1 x8], used in Load M16
    I32_32,          // (i32)32
    I64_64,          // (i64)64
    F32_Inf,         // (f32)0x7F800000
    F64_Inf,         // (f64)0x7FF0000000000000
    F32_Mant,        // (f32)0x007FFFFF
    F64_Mant,        // (f64)0x000FFFFFFFFFFFFF
    I8_1,            // (i64)1
    I8_MaskBits,     // [1u8, 2u8, ..., 128u8, 1u8, 2u8, ..., 128u8]
    I8_PackIndices,  // pointer to array of i8 pack indices
    I16_MaskBits,    // [1u16, 2u16, ..., 128u16]
    I16_PackIndices, // pointer to array of i16 pack indices
    I32_MaskBits,    // [1u32, 2u32, 4u32, 8u32]
    I32_PackIndices, // pointer to array of i32 pack indices
    I64_MaskBits,    // [1u64, 2u64]
    I64_PackIndices, // pointer to array of i64 pack indices
    Count
};
} // namespace SpecialConstant

namespace {
struct SuggestedReg {
    AnyR reg{};
    bool acknowledge = false;

    AnyR take() {
        acknowledge = true;
        return reg;
    }
};

struct ArgInfo {
    const ArgumentDecl *arg;
    aja64::Gp gp;
};

struct IndexRegState {
    GpR gp{};
    size_t offset = 0;
};

// MSS = mask shift store
struct MSSWriterState {
    bool active = false;
    GpR dst_ptr{};
    GpR acc{};
};

struct SmallPackIndexTables {
    uint8_t i8[256][16]{};
    uint8_t i16[256][16]{};
};

constexpr SmallPackIndexTables make_small_pack_index_tables() noexcept {
    SmallPackIndexTables tables{};
    for (size_t mask = 0; mask != 256; ++mask) {
        for (size_t i = 0; i != 16; ++i) {
            tables.i8[mask][i] = 16;
            tables.i16[mask][i] = 16;
        }
        size_t out_i8 = 0;
        size_t out_i16 = 0;
        for (size_t lane = 0; lane != 8; ++lane) {
            if ((mask & (size_t{1} << lane)) == 0) { continue; }
            tables.i8[mask][out_i8++] = uint8_t(lane);
            tables.i16[mask][out_i16++] = uint8_t(lane * 2);
            tables.i16[mask][out_i16++] = uint8_t(lane * 2 + 1);
        }
    }
    return tables;
}

enum class MSSCandidateState {
    Empty,
    VectorOnly,
    ScalarOnly,
    VectorAndScalar,
    Rejected,
};

struct MSSCandidate {
    MSSCandidateState state = MSSCandidateState::Empty;
    size_t vector_cursor = 0;
    size_t scalar_cursor = 0;
};

} // namespace

static aja64::CondCode map_cmp_op_signed(CmpOp op) noexcept {
    switch (op) {
    case CmpOp::Less: return aja64::CondCode::kLT; break;
    case CmpOp::Greater: return aja64::CondCode::kGT; break;
    case CmpOp::LessEqual: return aja64::CondCode::kLE; break;
    case CmpOp::GreaterEqual: return aja64::CondCode::kGE; break;
    case CmpOp::Equal: return aja64::CondCode::kEQ; break;
    case CmpOp::NotEqual: return aja64::CondCode::kNE; break;
    }
    SIMJIT_UNREACHABLE();
}

static aja64::CondCode map_cmp_op_unsigned(CmpOp op) noexcept {
    switch (op) {
    case CmpOp::Less: return aja64::CondCode::kLO; break;
    case CmpOp::Greater: return aja64::CondCode::kHI; break;
    case CmpOp::LessEqual: return aja64::CondCode::kLS; break;
    case CmpOp::GreaterEqual: return aja64::CondCode::kHS; break;
    case CmpOp::Equal: return aja64::CondCode::kEQ; break;
    case CmpOp::NotEqual: return aja64::CondCode::kNE; break;
    }
    SIMJIT_UNREACHABLE();
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
    case ScalarDataType::I128: messed_up("Invalid register type %s", show_scalar_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

static aja64::Mem mem_offset(const GpR &base, const GpR &i, ScalarDataType dtype) noexcept {
    return aja64::Mem(base, i, aja64::lsl((uint32_t)scalar_dtype_size_log2(dtype)));
}

static void store(aja64::Compiler &cc, const AnyR &arg, const aja64::Mem &mem, ScalarDataType sdtype) {
    switch (sdtype) {
    case ScalarDataType::I8: cc.strb(arg.as<GpR>(), mem); break;
    case ScalarDataType::I16: cc.strh(arg.as<GpR>(), mem); break;
    case ScalarDataType::I32:
    case ScalarDataType::I64: cc.str(arg.as<GpR>(), mem); break;
    case ScalarDataType::F32:
    case ScalarDataType::F64: cc.str(arg.as<VecR>(), mem); break;
    case ScalarDataType::I1:
    case ScalarDataType::I128: messed_up("Invalid type %s in scalar store", show_scalar_dtype(sdtype));
    }
}

static void load(aja64::Compiler &cc, const AnyR &arg, const aja64::Mem &mem, ScalarDataType sdtype) {
    switch (sdtype) {
    case ScalarDataType::I8: cc.ldrsb(arg.as<GpR>(), mem); break;
    case ScalarDataType::I16: cc.ldrsh(arg.as<GpR>(), mem); break;
    case ScalarDataType::I32:
    case ScalarDataType::I64: cc.ldr(arg.as<GpR>(), mem); break;
    case ScalarDataType::F32:
    case ScalarDataType::F64: cc.ldr(arg.as<VecR>(), mem); break;
    case ScalarDataType::I1:
    case ScalarDataType::I128: messed_up("Invalid type %s in scalar load", show_scalar_dtype(sdtype));
    }
}

static size_t vec_elem_size(const VecR &reg) {
    switch (reg.element_type()) {
    case asmjit::a64::VecElementType::kB: return 1;
    case asmjit::a64::VecElementType::kH: return 2;
    case asmjit::a64::VecElementType::kS: return 4;
    case asmjit::a64::VecElementType::kD: return 8;
    case asmjit::a64::VecElementType::kNone: SIMJIT_ASSERT(0); [[fallthrough]];
    default: messed_up("Invalid element type %u", (uint32_t)reg.element_type());
    }
    SIMJIT_UNREACHABLE();
}

static VecR vec_with_elem_size(const VecR &reg, size_t sz) noexcept {
    switch (sz) {
    case 1: return reg.b16();
    case 2: return reg.h8();
    case 4: return reg.s4();
    case 8: return reg.d2();
    default: break;
    }
    SIMJIT_UNREACHABLE();
}

static bool is_logical_imm(ScalarDataType sdtype, uint64_t imm) noexcept {
    return (sdtype == ScalarDataType::I8 && aja64::Utils::is_logical_imm(imm, 32)) ||
           (sdtype == ScalarDataType::I16 && aja64::Utils::is_logical_imm(imm, 32)) ||
           (sdtype == ScalarDataType::I32 && aja64::Utils::is_logical_imm(imm, 32)) ||
           (sdtype == ScalarDataType::I64 && aja64::Utils::is_logical_imm(imm, 64));
}

static uint64_t canonical_logical_imm(ScalarDataType sdtype, ConstData data) noexcept {
    data = data.retag(sdtype);
    switch (sdtype) {
    // Scalar i8/i16 values are kept sign-extended in 32-bit Gp registers. A logical immediate must match that
    // canonical register representation; using only the narrow bit pattern would turn values like i16 0x8000 into
    // positive 0x00008000 instead of negative 0xffff8000.
    case ScalarDataType::I8: return (uint32_t)(int32_t)data.as_i8();
    case ScalarDataType::I16: return (uint32_t)(int32_t)data.as_i16();
    case ScalarDataType::I32: return data.as_u32();
    case ScalarDataType::I64: return data.as_u64();
    default: SIMJIT_ASSERT(0);
    }
    SIMJIT_UNREACHABLE();
}

static bool can_cmp_signed_imm(int64_t value) noexcept {
    return value >= 0 && aja64::Utils::is_add_sub_imm((uint64_t)value);
}

static size_t mask_shift_writer_bit_count(const Step *step) noexcept {
    if (step->dtype == ScalarDataType::I1) { return 1; }
    return mask_dtype_bits(step->dtype.as_mask());
}

static bool is_vector_mask_shift_writer_store(const Step *step) noexcept {
    return step->is(StepKind::Store) && step->dtype.is_mask() &&
           (step->dtype == MaskDataType::M2 || step->dtype == MaskDataType::M4);
}

static bool is_scalar_mask_shift_writer_store(const Step *step) noexcept {
    return step->is(StepKind::Store) && step->dtype == ScalarDataType::I1;
}

static size_t step_row_width(const Step *step) noexcept {
    if (step->dtype.is_vec()) { return step->dtype.as_vec().nelems(); }
    if (step->dtype.is_mask()) { return mask_dtype_bits(step->dtype.as_mask()); }
    return 1;
}

static void bump_refcount(uint16_t &refcount) noexcept {
    if (refcount != UINT16_MAX) { ++refcount; }
}

static bool is_shift_rotate(ArithBinaryOp op) noexcept {
    return op == ArithBinaryOp::ShiftRightArith || op == ArithBinaryOp::ShiftLeftLogical ||
           op == ArithBinaryOp::ShiftRightLogical || op == ArithBinaryOp::RotateLeft ||
           op == ArithBinaryOp::RotateRight;
}

static bool scalar_binary_const_can_fold(ScalarDataType sdtype, ArithBinaryOp op, const ConstData &data) noexcept {
    if (!is_simple_int_dtype(sdtype)) { return false; }
    uint64_t imm = data.as_unsigned();
    if ((op == ArithBinaryOp::Add || op == ArithBinaryOp::Sub) && aja64::Utils::is_add_sub_imm(imm)) { return true; }
    if ((op == ArithBinaryOp::And || op == ArithBinaryOp::Or || op == ArithBinaryOp::AndNot ||
         op == ArithBinaryOp::Xor) &&
        is_logical_imm(sdtype, canonical_logical_imm(sdtype, data))) {
        return true;
    }
    return imm != 0 && imm < scalar_dtype_bits(sdtype) && is_shift_rotate(op);
}

static bool vector_binary_const_can_fold(VecDataType vdtype, ArithBinaryOp op, const ConstData &data) noexcept {
    if (!vdtype.is_int()) { return false; }
    uint64_t imm = data.as_unsigned();
    return imm != 0 && imm < vdtype.element_size_bits() && is_shift_rotate(op);
}

namespace {
struct CompileState {
    MemoryArena *arena;
    aja64::Compiler &cc;
    GpR counter{};
    GpR row_count{};
    ArenaArray<ArgInfo> args{};
    ArenaArray<AnyR> accs{};
    const mir::Function *mir_func = nullptr;
    ArenaArray<AnyR> step_map{};
    ArenaArray<uint16_t> refcounts{};
    ArenaArray<uint16_t> folded_const_refcounts{};
    ArenaArray<Step *> paired_ops{};
    ArenaArray<IndexRegState> index_regs{};
    std::vector<MSSWriterState> mask_shift_writers{};
    AnyR special_constants[(int)SpecialConstant::Count]{};
    GpR shifted_counter[7]{};

    aj::Label i8_pack_indices_label{};
    aj::Label i16_pack_indices_label{};
    aj::Label i32_pack_indices_label{};
    aj::Label i64_pack_indices_label{};
    aj::Label remainder_label{};
    aj::Label end_label{};

    void init(const mir::Function *func) {
        mir_func = func;
        step_map = arena->alloc_array<AnyR>(func->step_id_count);
        args = arena->alloc_array<ArgInfo>(func->args.size());
        refcounts = arena->alloc_array<uint16_t>(func->step_id_count);
        folded_const_refcounts = arena->alloc_array<uint16_t>(func->step_id_count);
        if (SIMJIT_A64_ASMJIT_INDEX_REGS) { index_regs = arena->alloc_array<IndexRegState>(func->args.size()); }
        accs = arena->alloc_array<AnyR>(func->accs.count);
        if (SIMJIT_A64_ASMJIT_STP || SIMJIT_A64_ASMJIT_LDP) {
            paired_ops = arena->alloc_array<Step *>(func->step_id_count);
        }

        row_count = cc.new_gp64("n");
        counter = cc.new_gp64("i");
        if (!func->remainder_roots.empty()) { remainder_label = cc.new_named_label("remainder"); }
        end_label = cc.new_named_label("end");
    }

    void clear_shifted_counters() noexcept {
        for (auto &c : shifted_counter) {
            c = {};
        }
    }

    GpR get_or_insert_shifted_counter(size_t shift) {
        SIMJIT_ASSERT(shift < std::size(shifted_counter));
        if (shifted_counter[shift].is_valid()) { return shifted_counter[shift]; }
        GpR idx = cc.new_gp64();
        cc.lsr(idx, counter, shift);
        shifted_counter[shift] = idx;
        return idx;
    }

    VecR vec_special_const(unsigned constant) noexcept {
        SIMJIT_ASSERT(constant < SpecialConstant::Count);
        return special_constants[constant].as<VecR>();
    }
    GpR gp_special_const(unsigned constant) noexcept {
        SIMJIT_ASSERT(constant < SpecialConstant::Count);
        return special_constants[constant].as<GpR>();
    }

    void add_u8_popcount_to_acc(const GpR &acc, const GpR &value) {
        VecR tmp = cc.new_vec128();
        cc.fmov(tmp.s(), value.w());
        cc.cnt(tmp.b8(), tmp.b8());
        cc.addv(tmp.b(), tmp.b8());
        cc.umov(value.w(), tmp.b(0));
        cc.add(acc, acc, value);
    }

    bool const_is_folded_root(const Step *step) const noexcept {
        if (!step->is(StepKind::Const)) return false;
        return refcounts[step->id] != 0 && refcounts[step->id] == folded_const_refcounts[step->id];
    }

    void record_folded_const_ref(Step *step) {
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::ArithBinary) {
                if (!SIMJIT_A64_ASMJIT_CONST_OPS || !data.right->is(StepKind::Const)) { return; }
                if (step->dtype.is_scalar()) {
                    if (scalar_binary_const_can_fold(step->dtype.as_scalar(), data.op,
                                                     data.right->step_data<StepKind::Const>())) {
                        bump_refcount(folded_const_refcounts[data.right->id]);
                    }
                } else if (step->dtype.is_vec()) {
                    if (vector_binary_const_can_fold(step->dtype.as_vec(), data.op,
                                                     data.right->step_data<StepKind::Const>())) {
                        bump_refcount(folded_const_refcounts[data.right->id]);
                    }
                }
                return;
            }
            SIMJIT_MATCH (StepKind::Compare) {
                if (!SIMJIT_A64_ASMJIT_CONST_OPS || !data.right->is(StepKind::Const)) { return; }
                if (data.left->dtype.is_scalar()) {
                    ScalarDataType sdtype = data.left->dtype.as_scalar();
                    if (is_float_dtype(sdtype) && step_is_zero(data.right)) {
                        bump_refcount(folded_const_refcounts[data.right->id]);
                    } else if (is_simple_int_dtype(sdtype) &&
                               aja64::Utils::is_add_sub_imm(data.right->step_data<StepKind::Const>().as_unsigned())) {
                        bump_refcount(folded_const_refcounts[data.right->id]);
                    }
                } else if (data.left->dtype.is_vec() && step_is_zero(data.right) && !data.is_unsigned) {
                    bump_refcount(folded_const_refcounts[data.right->id]);
                }
                return;
            }
            SIMJIT_MATCH (StepKind::Select) {
                if (!step->dtype.is_vec()) { return; }
                if (SIMJIT_A64_ASMJIT_BLEND_SUB && data.truthy->is(StepKind::ArithBinary)) {
                    const auto &bin_data = data.truthy->step_data<StepKind::ArithBinary>();
                    if (bin_data.op == ArithBinaryOp::Add && bin_data.right->is(StepKind::Const) &&
                        bin_data.right->step_data<StepKind::Const>().as_unsigned() == 1 &&
                        bin_data.left == data.falsy) {
                        bump_refcount(folded_const_refcounts[bin_data.right->id]);
                    }
                }
                if (SIMJIT_A64_ASMJIT_ZEROBLEND && step_is_zero(data.falsy)) {
                    bump_refcount(folded_const_refcounts[data.falsy->id]);
                }
                return;
            }
        default: return;
        }
    }

    VecR shrink_mask(const VecR &arg, size_t src_sz, size_t dst_sz) {
        SIMJIT_ASSERT(src_sz > dst_sz);
        aja64::Vec tmp = cc.new_vec128();
        if (src_sz == 8) {
            if (dst_sz == 4) {
                cc.xtn(tmp.s2(), arg.d2());
            } else if (dst_sz == 2) {
                cc.xtn(tmp.s2(), arg.d2());
                cc.xtn(tmp.h4(), tmp.s4());
            } else if (dst_sz == 1) {
                cc.xtn(tmp.s2(), arg.d2());
                cc.xtn(tmp.h4(), tmp.s4());
                cc.xtn(tmp.b8(), tmp.h8());
            } else {
                SIMJIT_ASSERT(0);
            }
        } else if (src_sz == 4) {
            if (dst_sz == 2) {
                cc.xtn(tmp.h4(), arg.s4());
            } else if (dst_sz == 1) {
                cc.xtn(tmp.h4(), arg.s4());
                cc.xtn(tmp.b8(), tmp.h8());
            } else {
                SIMJIT_ASSERT(0);
            }
        } else if (src_sz == 2) {
            SIMJIT_ASSERT(dst_sz == 1);
            cc.xtn(tmp.b8(), arg.h8());
        } else {
            SIMJIT_ASSERT(0);
        }
        return vec_with_elem_size(tmp, dst_sz);
    }

    VecR expand_mask(const VecR &arg, size_t src_sz, size_t dst_sz) {
        SIMJIT_ASSERT(dst_sz > src_sz);
        aja64::Vec tmp = cc.new_vec128();
        if (src_sz == 1) {
            if (dst_sz == 2) {
                cc.sxtl(tmp.h8(), arg.b8());
            } else if (dst_sz == 4) {
                cc.sxtl(tmp.h4(), arg.b8());
                cc.sxtl(tmp.s4(), tmp.h4());
            } else if (dst_sz == 8) {
                cc.sxtl(tmp.h2(), arg.b8());
                cc.sxtl(tmp.s2(), tmp.h2());
                cc.sxtl(tmp.d2(), tmp.s2());
            } else {
                SIMJIT_ASSERT(0);
            }
        } else if (src_sz == 2) {
            if (dst_sz == 4) {
                cc.sxtl(tmp.s4(), arg.h4());
            } else if (dst_sz == 8) {
                cc.sxtl(tmp.s2(), arg.h2());
                cc.sxtl(tmp.d2(), tmp.s2());
            } else {
                SIMJIT_ASSERT(0);
            }
        } else if (src_sz == 4) {
            SIMJIT_ASSERT(dst_sz == 8);
            cc.sxtl(tmp.d2(), arg.s2());
        } else {
            SIMJIT_ASSERT(0);
        }
        return vec_with_elem_size(tmp, dst_sz);
    }

    std::pair<VecR, VecR> masks_to_same_size(const VecR &left, const VecR &right) {
        size_t left_size = vec_elem_size(left);
        size_t right_size = vec_elem_size(right);
        if (left_size == right_size) { return {left, right}; }
        if (left_size > right_size) { return {shrink_mask(left, left_size, right_size), right}; }
        return {left, shrink_mask(right, right_size, left_size)};
    }

    aja64::Mem vec_mem(const ArgumentAddress &addr, VecDataType dtype, bool change_post_idx = true) {
        if (SIMJIT_A64_ASMJIT_INDEX_REGS) {
            if (auto &state = index_regs[addr.arg]; state.gp.is_valid()) {
                if (!change_post_idx && state.offset == addr.offset) { return aja64::ptr(state.gp); }
                if (state.offset == addr.offset) {
                    state.offset += dtype.nelems();
                    return aja64::ptr_post(state.gp, (int)dtype.size_bytes());
                }
            }
        }
        GpR base = args[addr.arg].gp;
        GpR tmp = cc.new_gp64();
        cc.add(tmp, base, counter, aja64::lsl(dtype.element_size_bytes_log2()));
        if (addr.offset != 0) { cc.add(tmp, tmp, addr.offset << dtype.element_size_bytes_log2()); }
        return aja64::ptr(tmp);
    }

    aja64::Mem vec_pair_mem(const ArgumentAddress &addr, VecDataType dtype) {
        if (SIMJIT_A64_ASMJIT_INDEX_REGS) {
            if (auto &state = index_regs[addr.arg]; state.gp.is_valid()) {
                if (state.offset == addr.offset) {
                    state.offset += dtype.nelems() * 2;
                    return aja64::ptr_post(state.gp, (int)dtype.size_bytes() * 2);
                }
            }
        }
        GpR base = args[addr.arg].gp;
        GpR tmp = cc.new_gp64();
        cc.add(tmp, base, counter, aja64::lsl(dtype.element_size_bytes_log2()));
        if (addr.offset != 0) { cc.add(tmp, tmp, addr.offset << dtype.element_size_bytes_log2()); }
        return aja64::ptr(tmp);
    }

    aja64::Mem scalar_mem(const ArgumentAddress &addr, ScalarDataType sdtype) {
        if (SIMJIT_A64_ASMJIT_INDEX_REGS) {
            if (auto &state = index_regs[addr.arg]; state.gp.is_valid()) {
                if (state.offset == addr.offset) {
                    state.offset += 1;
                    return aja64::ptr_post(state.gp, (int)scalar_dtype_size(sdtype));
                }
            }
        }
        GpR base = args[addr.arg].gp;
        GpR tmp = cc.new_gp64();
        cc.add(tmp, base, counter, aja64::lsl(scalar_dtype_size_log2(sdtype)));
        if (addr.offset != 0) { cc.add(tmp, tmp, addr.offset << scalar_dtype_size_log2(sdtype)); }
        return aja64::ptr(tmp);
    }

    aja64::Mem mask_mem(const ArgumentAddress &addr, MaskDataType mdtype) {
        size_t mask_size = scalar_dtype_size(mask_dtype_to_scalar(mdtype));
        if (SIMJIT_A64_ASMJIT_INDEX_REGS && mdtype != MaskDataType::M2 && mdtype != MaskDataType::M4) {
            if (auto &state = index_regs[addr.arg]; state.gp.is_valid()) {
                if (state.offset == addr.offset) {
                    state.offset += mask_dtype_bits(mdtype);
                    return aja64::ptr_post(state.gp, (int)mask_size);
                }
            }
        }
        GpR base = args[addr.arg].gp;
        GpR tmp = cc.new_gp64();
        cc.add(tmp, base, counter, aja64::lsr(3));
        if (addr.offset != 0) { cc.add(tmp, tmp, addr.offset >> 3); }
        return aja64::ptr(tmp);
    }

    GpR mask_row_offset(size_t offset) {
        if (offset == 0) { return counter; }
        GpR row = cc.new_gp64();
        cc.add(row, counter, offset);
        return row;
    }

    aja64::Mem small_mask_mem(const ArgumentAddress &addr, const GpR &row) {
        GpR base = args[addr.arg].gp;
        GpR tmp = cc.new_gp64();
        cc.add(tmp, base, row, aja64::lsr(3));
        return aja64::ptr(tmp);
    }

    void init_f32_const(const VecR &result, float val, bool is_vec = false) {
        if (aja64::Utils::is_fp32_imm8(val)) {
            cc.fmov(result, val);
        } else {
            if (is_vec && val == 0) {
                cc.movi(result, 0);
                return;
            }
            GpR gp = cc.new_gp32();
            init_int_const(gp, ConstData::f32(val), ScalarDataType::I32);
            if (is_vec) {
                cc.dup(result, gp);
            } else {
                cc.fmov(result, gp);
            }
        }
    }

    void init_f64_const(const VecR &result, double val, bool is_vec = false) {
        if (aja64::Utils::is_fp64_imm8(val)) {
            cc.fmov(result, val);
        } else {
            if (is_vec && val == 0) {
                cc.movi(result, 0);
                return;
            }
            GpR gp = cc.new_gp64();
            init_int_const(gp, ConstData::f64(val), ScalarDataType::I64);
            if (is_vec) {
                cc.dup(result, gp);
            } else {
                cc.fmov(result, gp);
            }
        }
    }

    void init_int_const(const GpR &result, ConstData data, ScalarDataType dtype) {
        data = data.retag(dtype);
        uint64_t bits = data.as_unsigned();
        switch (dtype) {
        case ScalarDataType::I8: cc.mov(result, (int8_t)data.as_signed()); break;
        case ScalarDataType::I16: cc.mov(result, (int16_t)data.as_signed()); break;
        case ScalarDataType::I32:
            cc.mov(result, (uint16_t)bits);
            if ((bits & 0xFFFF0000) != 0) { cc.movk(result, (uint16_t)(bits >> 16), 16); }
            break;
        case ScalarDataType::I64:
            cc.mov(result, (uint16_t)bits);
            if ((bits & 0x00000000FFFF0000) != 0) cc.movk(result, (uint16_t)(bits >> 16), 16);
            if ((bits & 0x0000FFFF00000000) != 0) cc.movk(result, (uint16_t)(bits >> 32), 32);
            if ((bits & 0xFFFF000000000000) != 0) cc.movk(result, (uint16_t)(bits >> 48), 48);
            break;
        case ScalarDataType::F32:
        case ScalarDataType::F64:
        case ScalarDataType::I1:
        case ScalarDataType::I128: messed_up("Invalid type %s in int const", show_scalar_dtype(dtype));
        }
    }

    GpR zero_extend_small_int_for_unsigned_cmp(const GpR &value, ScalarDataType dtype) {
        SIMJIT_ASSERT(dtype == ScalarDataType::I8 || dtype == ScalarDataType::I16);
        GpR result = cc.new_gp32();
        if (dtype == ScalarDataType::I8) {
            cc.uxtb(result, value.w());
        } else {
            cc.uxth(result, value.w());
        }
        return result;
    }

    GpR create_int_reg(ScalarDataType scalar) {
        SIMJIT_ASSERT(is_simple_int_dtype(scalar));
        return cc.new_gp(scalar_dtype_to_asmjit(scalar));
    }

    VecR create_float_reg(ScalarDataType scalar) {
        if (scalar == ScalarDataType::F32) { return cc.new_vec_s(); }
        if (scalar == ScalarDataType::F64) { return cc.new_vec_d(); }
        messed_up("Unexpected type %s in float context", show_scalar_dtype(scalar));
    }

    AnyR create_scalar_reg(ScalarDataType scalar) {
        if (scalar == ScalarDataType::F32) { return cc.new_vec_s(); }
        if (scalar == ScalarDataType::F64) { return cc.new_vec_d(); }
        return create_int_reg(scalar);
    }

    void emit_scalar_const_div(const ConstDivData &data, ScalarDataType sdtype, const GpR &result, const GpR &numer) {
        bool is_signed = data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::Mod;
        bool is_mod = data.op == ArithBinaryOp::Mod || data.op == ArithBinaryOp::UMod;
        size_t bits = scalar_dtype_bits(sdtype);
        GpR quotient = is_mod ? create_int_reg(sdtype) : result;

        if (!data.has_magic) {
            if (is_signed) {
                GpR tweak = create_int_reg(sdtype);
                GpR mask = int_subexpr(data.round_mask);
                if (sdtype == ScalarDataType::I32) {
                    cc.asr(tweak, numer, 31);
                    cc.and_(tweak, tweak, mask);
                    cc.add(quotient, numer, tweak);
                    if (data.shift != 0) { cc.asr(quotient, quotient, data.shift); }
                    if (data.negative_divisor) { cc.neg(quotient, quotient); }
                } else {
                    cc.asr(tweak, numer, bits - 1);
                    cc.and_(tweak, tweak, mask);
                    cc.add(quotient, numer, tweak);
                    if (data.shift != 0) { cc.asr(quotient, quotient, data.shift); }
                    if (data.negative_divisor) { cc.neg(quotient, quotient); }
                }
            } else if (data.shift == 0) {
                cc.mov(quotient, numer);
            } else {
                cc.lsr(quotient, numer, data.shift);
            }
        } else if (is_signed) {
            GpR magic = int_subexpr(data.magic);
            if (sdtype == ScalarDataType::I32) {
                GpR prod = cc.new_gp64();
                cc.smull(prod, numer, magic);
                cc.asr(prod, prod, 32);
                cc.mov(quotient.w(), prod.w());
                if (data.has_add) {
                    if (data.negative_divisor) {
                        cc.sub(quotient, quotient, numer);
                    } else {
                        cc.add(quotient, quotient, numer);
                    }
                }
                if (data.shift != 0) { cc.asr(quotient, quotient, data.shift); }
                GpR sign = cc.new_gp32();
                cc.lsr(sign, quotient, 31);
                cc.add(quotient, quotient, sign);
            } else {
                cc.smulh(quotient, numer, magic);
                if (data.has_add) {
                    if (data.negative_divisor) {
                        cc.sub(quotient, quotient, numer);
                    } else {
                        cc.add(quotient, quotient, numer);
                    }
                }
                if (data.shift != 0) { cc.asr(quotient, quotient, data.shift); }
                GpR sign = cc.new_gp64();
                cc.lsr(sign, quotient, bits - 1);
                cc.add(quotient, quotient, sign);
            }
        } else {
            GpR magic = int_subexpr(data.magic);
            if (sdtype == ScalarDataType::I32) {
                GpR prod = cc.new_gp64();
                cc.umull(prod, numer, magic);
                cc.lsr(prod, prod, 32);
                cc.mov(quotient.w(), prod.w());
                if (data.has_add) {
                    GpR tmp = cc.new_gp32();
                    cc.sub(tmp, numer, quotient);
                    cc.lsr(tmp, tmp, 1);
                    cc.add(quotient, tmp, quotient);
                }
                if (data.shift != 0) { cc.lsr(quotient, quotient, data.shift); }
            } else {
                cc.umulh(quotient, numer, magic);
                if (data.has_add) {
                    GpR tmp = cc.new_gp64();
                    cc.sub(tmp, numer, quotient);
                    cc.lsr(tmp, tmp, 1);
                    cc.add(quotient, tmp, quotient);
                }
                if (data.shift != 0) { cc.lsr(quotient, quotient, data.shift); }
            }
        }

        if (is_mod) {
            GpR divisor = int_subexpr(data.divisor);
            cc.msub(result, quotient, divisor, numer);
        }
    }

    VecR create_mask_reg(MaskDataType dtype) {
        VecR reg = cc.new_vec128();
        switch (dtype) {
        case MaskDataType::M2: return reg.d2();
        case MaskDataType::M4: return reg.s4();
        case MaskDataType::M8: return reg.h8();
        case MaskDataType::M16: return reg.b16();
        case MaskDataType::M32:
        case MaskDataType::M64: messed_up("Invalid mask type %s", show_mask_dtype(dtype));
        }
        SIMJIT_UNREACHABLE();
    }

    VecR create_vec_reg(VecDataType vdtype) {
        switch (vdtype.elem) {
        case VecElemType::I8: {
            VecR vec = cc.new_vec128();
            switch (vdtype.size) {
            case VecSize::X8: return vec.b8();
            case VecSize::X16: return vec.b16();
            case VecSize::X2:
            case VecSize::X4:
            case VecSize::X32:
            case VecSize::X64: break;
            }
            break;
        }
        case VecElemType::I16: {
            VecR vec = cc.new_vec128();
            switch (vdtype.size) {
            case VecSize::X2: return vec.h2();
            case VecSize::X4: return vec.h4();
            case VecSize::X8: return vec.h8();
            case VecSize::X16:
            case VecSize::X32:
            case VecSize::X64: break;
            }
            break;
        }
        case VecElemType::I32:
        case VecElemType::F32: {
            VecR vec = cc.new_vec128();
            switch (vdtype.size) {
            case VecSize::X2: return vec.s2();
            case VecSize::X4: return vec.s4();
            case VecSize::X8:
            case VecSize::X16:
            case VecSize::X32:
            case VecSize::X64: break;
            }
            break;
        }
        case VecElemType::I64:
        case VecElemType::F64: {
            VecR vec = cc.new_vec128();
            switch (vdtype.size) {
            case VecSize::X2: return vec.d2();
            case VecSize::X4:
            case VecSize::X8:
            case VecSize::X16:
            case VecSize::X32:
            case VecSize::X64: break;
            }
            break;
        }
        }
        messed_up("Invalid vector type %s", show_vec_dtype(vdtype));
    }

    VecR emit_vector_gather_direct(const GatherData &data, VecDataType vdtype, VecDataType idx_dtype) {
        SIMJIT_ASSERT(data.idx->is(StepKind::Load));
        SIMJIT_ASSERT(idx_dtype.nelems() == 2 || idx_dtype.nelems() == 4);

        auto idx_data = data.idx->step_data<StepKind::Load>();
        GpR indices[4];
        for (uint32_t lane = 0; lane != idx_dtype.nelems(); ++lane) {
            indices[lane] = cc.new_gp64();
        }

        if (SIMJIT_A64_ASMJIT_INDEX_REGS && index_regs[idx_data.addr.arg].gp.is_valid() &&
            index_regs[idx_data.addr.arg].offset == idx_data.addr.offset) {
            IndexRegState &state = index_regs[idx_data.addr.arg];
            state.offset += idx_dtype.nelems();
            int pair_size = (int)(idx_dtype.element_size_bytes() * 2);
            for (uint32_t lane = 0; lane != idx_dtype.nelems(); lane += 2) {
                aja64::Mem mem = aja64::ptr_post(state.gp, pair_size);
                if (idx_dtype.elem == VecElemType::I32) {
                    cc.ldpsw(indices[lane], indices[lane + 1], mem);
                } else {
                    cc.ldp(indices[lane], indices[lane + 1], mem);
                }
            }
        } else {
            const ArgInfo &idx_arg = args[idx_data.addr.arg];
            GpR base = cc.new_gp64();
            cc.add(base, idx_arg.gp, counter, aja64::lsl(idx_dtype.element_size_bytes_log2()));
            if (idx_data.addr.offset != 0) {
                cc.add(base, base, (int64_t)(idx_data.addr.offset << idx_dtype.element_size_bytes_log2()));
            }
            for (uint32_t lane = 0; lane != idx_dtype.nelems(); lane += 2) {
                aja64::Mem mem = aja64::ptr(base, (int32_t)(lane * idx_dtype.element_size_bytes()));
                if (idx_dtype.elem == VecElemType::I32) {
                    cc.ldpsw(indices[lane], indices[lane + 1], mem);
                } else {
                    cc.ldp(indices[lane], indices[lane + 1], mem);
                }
            }
        }

        const ArgInfo &arg = args[data.data];
        VecR result = create_vec_reg(vdtype);
        for (uint32_t lane = 0; lane != vdtype.nelems(); ++lane) {
            GpR idx = indices[lane];
            if (lane == 0) {
                aja64::Mem mem = mem_offset(arg.gp, idx, vdtype.to_scalar());
                switch (vdtype.elem) {
                case VecElemType::I8: cc.ldr(result.b(), mem); break;
                case VecElemType::I16: cc.ldr(result.h(), mem); break;
                case VecElemType::I32:
                case VecElemType::F32: cc.ldr(result.s(), mem); break;
                case VecElemType::I64:
                case VecElemType::F64: cc.ldr(result.d(), mem); break;
                }
                continue;
            }

            GpR addr = cc.new_gp64();
            cc.add(addr, arg.gp, idx.x(), aja64::lsl(vdtype.element_size_bytes_log2()));
            switch (vdtype.elem) {
            case VecElemType::I8: cc.ld1(result.b(lane), aja64::ptr(addr)); break;
            case VecElemType::I16: cc.ld1(result.h(lane), aja64::ptr(addr)); break;
            case VecElemType::I32:
            case VecElemType::F32: cc.ld1(result.s(lane), aja64::ptr(addr)); break;
            case VecElemType::I64:
            case VecElemType::F64: cc.ld1(result.d(lane), aja64::ptr(addr)); break;
            }
        }
        return result;
    }

    VecR emit_vector_gather_indirect(const GatherData &data, VecDataType vdtype, VecDataType idx_dtype) {
        const ArgInfo &arg = args[data.data];
        VecR idx_vec = vec_subexpr(data.idx);
        VecR result = create_vec_reg(vdtype);

        for (uint32_t lane = 0; lane != vdtype.nelems(); ++lane) {
            GpR idx = idx_dtype.elem == VecElemType::I32 ? cc.new_gp32() : cc.new_gp64();
            if (idx_dtype.elem == VecElemType::I32) {
                cc.umov(idx.w(), idx_vec.s(lane));
            } else {
                cc.umov(idx.x(), idx_vec.d(lane));
            }

            if (lane == 0) {
                aja64::Mem mem = mem_offset(arg.gp, idx, vdtype.to_scalar());
                switch (vdtype.elem) {
                case VecElemType::I8: cc.ldr(result.b(), mem); break;
                case VecElemType::I16: cc.ldr(result.h(), mem); break;
                case VecElemType::I32:
                case VecElemType::F32: cc.ldr(result.s(), mem); break;
                case VecElemType::I64:
                case VecElemType::F64: cc.ldr(result.d(), mem); break;
                }
                continue;
            }

            GpR addr = cc.new_gp64();
            cc.add(addr, arg.gp, idx.x(), aja64::lsl(vdtype.element_size_bytes_log2()));
            switch (vdtype.elem) {
            case VecElemType::I8: cc.ld1(result.b(lane), aja64::ptr(addr)); break;
            case VecElemType::I16: cc.ld1(result.h(lane), aja64::ptr(addr)); break;
            case VecElemType::I32:
            case VecElemType::F32: cc.ld1(result.s(lane), aja64::ptr(addr)); break;
            case VecElemType::I64:
            case VecElemType::F64: cc.ld1(result.d(lane), aja64::ptr(addr)); break;
            }
        }

        return result;
    }

    VecR emit_vector_gather(const GatherData &data, VecDataType vdtype) {
        VecDataType idx_dtype = data.idx->dtype.as_vec();
        if (idx_dtype.elem != VecElemType::I32 && idx_dtype.elem != VecElemType::I64) {
            unsupported("Do not support %s gather index", show_vec_dtype(idx_dtype));
        }
        if (idx_dtype.nelems() != vdtype.nelems()) {
            messed_up("Gather value/index lane count mismatch: value=%s index=%s", show_vec_dtype(vdtype),
                      show_vec_dtype(idx_dtype));
        }
        if (data.idx->is(StepKind::Load)) { return emit_vector_gather_direct(data, vdtype, idx_dtype); }
        return emit_vector_gather_indirect(data, vdtype, idx_dtype);
    }

    VecR emit_vec_narrow_combine(VecDataType vdtype, const VecR &low, const VecR &high) {
        VecR result = create_vec_reg(vdtype);
        switch (vdtype.elem) {
        case VecElemType::I8: cc.uzp1(result.b16(), low.b16(), high.b16()); break;
        case VecElemType::I16: cc.uzp1(result.h8(), low.h8(), high.h8()); break;
        case VecElemType::I32: cc.uzp1(result.s4(), low.s4(), high.s4()); break;
        case VecElemType::I64:
        case VecElemType::F32:
        case VecElemType::F64:
            messed_up("invalid vector narrow-combine target element %s", show_vec_elem_type(vdtype.elem));
        }
        return result;
    }

    VecR emit_vec_float_narrow_combine(VecDataType vdtype, const VecR &low, const VecR &high) {
        if (vdtype.elem != VecElemType::F32 || vdtype.size != VecSize::X4) {
            messed_up("invalid vector float narrow-combine target %s", show_vec_dtype(vdtype));
        }
        VecR result = create_vec_reg(vdtype);
        cc.fcvtn(result.s2(), low.d2());
        cc.fcvtn2(result.s4(), high.d2());
        return result;
    }

    // DO NOT CALL THIS DIRECTLY!!! Use subexpr() instead.
    SIMJIT_NO_ASAN AnyR scalar_subexpr(const Step *step, SuggestedReg *suggested_dst = nullptr) {
        ScalarDataType sdtype = step->dtype.as_scalar();
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::Const) {
                if (sdtype == ScalarDataType::F32) {
                    VecR result = cc.new_vec_s();
                    init_f32_const(result, data.as_f32());
                    return result;
                }
                if (sdtype == ScalarDataType::F64) {
                    VecR result = cc.new_vec_d();
                    init_f64_const(result, data.as_f64());
                    return result;
                }

                GpR result = create_int_reg(sdtype);
                init_int_const(result, data, sdtype);
                return result;
            }
            SIMJIT_MATCH (StepKind::Load) {
                const ArgInfo &arg = args[data.addr.arg];
                if (sdtype == ScalarDataType::I1) {
                    GpR result = cc.new_gp64();
                    GpR idx = get_or_insert_shifted_counter(6);
                    cc.ldr(result, mem_offset(arg.gp, idx, ScalarDataType::I64));
                    cc.lsr(result, result, counter);
                    cc.and_(result, result, 1);
                    return result.w();
                }
                AnyR result = create_scalar_reg(sdtype);
                aja64::Mem mem = scalar_mem(data.addr, sdtype);
                load(cc, result, mem, sdtype);
                return result;
            }
            SIMJIT_MATCH (StepKind::LoadSplat) {
                const ArgInfo &arg = args[data.addr.arg];
                sdtype = sdtype == ScalarDataType::I1 ? ScalarDataType::I8 : sdtype;
                AnyR result = create_scalar_reg(sdtype);
                load(cc, result, aja64::ptr(arg.gp), sdtype);
                if (step->dtype == ScalarDataType::I1) { cc.and_(result.as<GpR>(), result.as<GpR>(), 1); }
                return result;
            }
            SIMJIT_MATCH (StepKind::Gather) {
                const ArgInfo &arg = args[data.data];
                GpR idx = int_subexpr(data.idx);
                AnyR result = create_scalar_reg(sdtype);
                aja64::Mem mem = mem_offset(arg.gp, idx, sdtype);
                load(cc, result, mem, sdtype);
                return result;
            }
            SIMJIT_MATCH (StepKind::Store) {
                const ArgInfo &arg = args[data.addr.arg];
                if (sdtype == ScalarDataType::I1) {
                    GpR what = int_subexpr(data.what);
                    GpR idx = get_or_insert_shifted_counter(6);
                    GpR x = cc.new_gp64();
                    cc.ldr(x, mem_offset(arg.gp, idx, ScalarDataType::I64));
                    GpR bit = cc.new_gp64();
                    cc.mov(bit, 1);
                    cc.lsl(bit, bit, counter);
                    cc.bic(x, x, bit);
                    cc.lsl(bit, what.x(), counter);
                    cc.orr(x, x, bit);
                    cc.str(x, mem_offset(arg.gp, idx, ScalarDataType::I64));
                    return {};
                }
                AnyR what = subexpr(data.what);
                aja64::Mem mem = scalar_mem(data.addr, sdtype);
                store(cc, what, mem, sdtype);
                return {};
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                auto handle_i8_i16 = [&](const GpR &result) {
                    if (sdtype == ScalarDataType::I16) {
                        cc.sxth(result, result);
                    } else if (sdtype == ScalarDataType::I8) {
                        cc.sxtb(result, result);
                    }
                };

                if (is_float_dtype(sdtype)) {
                    VecR left = float_subexpr(data.left);
                    VecR right = float_subexpr(data.right);
                    VecR result =
                        suggested_dst != nullptr ? suggested_dst->take().as<VecR>() : create_float_reg(sdtype);
                    switch (data.op) {
                    case ArithBinaryOp::Add: cc.fadd(result, left, right); break;
                    case ArithBinaryOp::Sub: cc.fsub(result, left, right); break;
                    case ArithBinaryOp::Mul: cc.fmul(result, left, right); break;
                    case ArithBinaryOp::Div: cc.fdiv(result, left, right); break;
                    case ArithBinaryOp::Min: cc.fminnm(result, left, right); break;
                    case ArithBinaryOp::Max: cc.fmaxnm(result, left, right); break;
                    case ArithBinaryOp::And: cc.and_(result.b8(), left.b8(), right.b8()); break;
                    case ArithBinaryOp::Or: cc.orr(result.b8(), left.b8(), right.b8()); break;
                    case ArithBinaryOp::Xor: cc.eor(result.b8(), left.b8(), right.b8()); break;
                    case ArithBinaryOp::AndNot: cc.bic(result.b8(), right.b8(), left.b8()); break;
                    default: messed_up("Unexpected ArithBinary %s in float context", show_arith_binary_op(data.op));
                    }
                    return result;
                }

                GpR left = int_subexpr(data.left);
                GpR result = create_int_reg(sdtype);
                auto sign_extend_i8_i16_left = [&]() {
                    if (sdtype == ScalarDataType::I16) {
                        cc.sxth(result, left);
                        left = result;
                    } else if (sdtype == ScalarDataType::I8) {
                        cc.sxtb(result, left);
                        left = result;
                    }
                };
                auto sign_extend_i8_i16_left_in_place = [&]() {
                    if (sdtype == ScalarDataType::I16) {
                        cc.sxth(left, left);
                    } else if (sdtype == ScalarDataType::I8) {
                        cc.sxtb(left, left);
                    }
                };
                auto mask_i8_i16_shift_right_into_result = [&](GpR &value) {
                    if (sdtype == ScalarDataType::I16) {
                        cc.and_(result, value, 15);
                        value = result;
                    } else if (sdtype == ScalarDataType::I8) {
                        cc.and_(result, value, 7);
                        value = result;
                    }
                };
                if (SIMJIT_A64_ASMJIT_CONST_OPS && data.right->is(StepKind::Const)) {
                    ConstData right_const = data.right->step_data<StepKind::Const>();
                    auto imm = right_const.as_unsigned();
                    if (aja64::Utils::is_add_sub_imm(imm) &&
                        (data.op == ArithBinaryOp::Add || data.op == ArithBinaryOp::Sub)) {
                        switch (data.op) {
                        case ArithBinaryOp::Add:
                            cc.add(result, left, imm);
                            handle_i8_i16(result);
                            break;
                        case ArithBinaryOp::Sub:
                            cc.sub(result, left, imm);
                            handle_i8_i16(result);
                            break;
                        default: SIMJIT_UNREACHABLE();
                        }
                        return result;
                    }

                    auto logical_imm = canonical_logical_imm(sdtype, right_const);
                    if (is_logical_imm(sdtype, logical_imm) &&
                        (data.op == ArithBinaryOp::And || data.op == ArithBinaryOp::Or ||
                         data.op == ArithBinaryOp::AndNot || data.op == ArithBinaryOp::Xor)) {
                        switch (data.op) {
                        case ArithBinaryOp::And: cc.and_(result, left, logical_imm); break;
                        case ArithBinaryOp::Or: cc.orr(result, left, logical_imm); break;
                        case ArithBinaryOp::AndNot:
                            cc.mvn(result, left);
                            cc.and_(result, result, logical_imm);
                            break;
                        case ArithBinaryOp::Xor: cc.eor(result, left, logical_imm); break;
                        default: SIMJIT_UNREACHABLE();
                        }
                        return result;
                    }

                    bool is_shift = imm != 0 && imm < scalar_dtype_bits(sdtype);
                    if (is_shift &&
                        (data.op == ArithBinaryOp::ShiftRightArith || data.op == ArithBinaryOp::ShiftLeftLogical ||
                         data.op == ArithBinaryOp::ShiftRightLogical || data.op == ArithBinaryOp::RotateLeft ||
                         data.op == ArithBinaryOp::RotateRight)) {
                        switch (data.op) {
                        case ArithBinaryOp::ShiftRightArith:
                            sign_extend_i8_i16_left();
                            cc.asr(result, left, imm);
                            break;
                        case ArithBinaryOp::ShiftRightLogical:
                            if (sdtype == ScalarDataType::I16) {
                                cc.uxth(result, left);
                                left = result;
                            }
                            if (sdtype == ScalarDataType::I8) {
                                cc.uxtb(result, left);
                                left = result;
                            }
                            cc.lsr(result, left, imm);
                            handle_i8_i16(result);
                            break;
                        case ArithBinaryOp::ShiftLeftLogical:
                            cc.lsl(result, left, imm);
                            handle_i8_i16(result);
                            break;
                        case ArithBinaryOp::RotateLeft: cc.ror(result, left, scalar_dtype_bits(sdtype) - imm); break;
                        case ArithBinaryOp::RotateRight: cc.ror(result, left, imm); break;
                        default: SIMJIT_UNREACHABLE();
                        }
                        return result;
                    }
                }

                GpR right = int_subexpr(data.right);
                switch (data.op) {
                case ArithBinaryOp::Add:
                    cc.add(result, left, right);
                    handle_i8_i16(result);
                    break;
                case ArithBinaryOp::Sub:
                    cc.sub(result, left, right);
                    handle_i8_i16(result);
                    break;
                case ArithBinaryOp::Mul:
                    cc.mul(result, left, right);
                    handle_i8_i16(result);
                    break;
                case ArithBinaryOp::Mul64SE: cc.smull(result, left.w(), right.w()); break;
                case ArithBinaryOp::Mul64ZE: cc.umull(result, left.w(), right.w()); break;
                case ArithBinaryOp::Div: cc.sdiv(result, left, right); break;
                case ArithBinaryOp::UDiv: cc.udiv(result, left, right); break;
                case ArithBinaryOp::Mod:
                    cc.sdiv(result, left, right);
                    cc.msub(result, result, right, left);
                    break;
                case ArithBinaryOp::UMod:
                    cc.udiv(result, left, right);
                    cc.msub(result, result, right, left);
                    break;
                case ArithBinaryOp::Min:
                    cc.cmp(left, right);
                    cc.csel(result, left, right, aja64::CondCode::kLT);
                    break;
                case ArithBinaryOp::Max:
                    cc.cmp(left, right);
                    cc.csel(result, left, right, aja64::CondCode::kGT);
                    break;
                case ArithBinaryOp::UMin:
                    if (sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) {
                        GpR cmp_left = zero_extend_small_int_for_unsigned_cmp(left, sdtype);
                        GpR cmp_right = zero_extend_small_int_for_unsigned_cmp(right, sdtype);
                        cc.cmp(cmp_left, cmp_right);
                    } else {
                        cc.cmp(left, right);
                    }
                    cc.csel(result, left, right, aja64::CondCode::kLO);
                    break;
                case ArithBinaryOp::UMax:
                    if (sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) {
                        GpR cmp_left = zero_extend_small_int_for_unsigned_cmp(left, sdtype);
                        GpR cmp_right = zero_extend_small_int_for_unsigned_cmp(right, sdtype);
                        cc.cmp(cmp_left, cmp_right);
                    } else {
                        cc.cmp(left, right);
                    }
                    cc.csel(result, left, right, aja64::CondCode::kHI);
                    break;
                case ArithBinaryOp::And: cc.and_(result, left, right); break;
                case ArithBinaryOp::Or: cc.orr(result, left, right); break;
                case ArithBinaryOp::Xor: cc.eor(result, left, right); break;
                case ArithBinaryOp::AndNot: cc.bic(result, right, left); break;
                case ArithBinaryOp::ShiftRightArith:
                    mask_i8_i16_shift_right_into_result(right);
                    sign_extend_i8_i16_left_in_place();
                    cc.asr(result, left, right);
                    break;
                case ArithBinaryOp::ShiftRightLogical:
                    mask_i8_i16_shift_right_into_result(right);
                    if (sdtype == ScalarDataType::I16) {
                        if (refcounts[data.left->id] <= 1) {
                            cc.uxth(left, left);
                        } else {
                            GpR zero_left = create_int_reg(sdtype);
                            cc.uxth(zero_left, left);
                            left = zero_left;
                        }
                    } else if (sdtype == ScalarDataType::I8) {
                        if (refcounts[data.left->id] <= 1) {
                            cc.uxtb(left, left);
                        } else {
                            GpR zero_left = create_int_reg(sdtype);
                            cc.uxtb(zero_left, left);
                            left = zero_left;
                        }
                    }
                    cc.lsr(result, left, right);
                    handle_i8_i16(result);
                    break;
                case ArithBinaryOp::ShiftLeftLogical:
                    mask_i8_i16_shift_right_into_result(right);
                    cc.lsl(result, left, right);
                    handle_i8_i16(result);
                    break;
                case ArithBinaryOp::RotateLeft: {
                    mask_i8_i16_shift_right_into_result(right);
                    cc.neg(result, right);
                    cc.ror(result, left, result);
                    break;
                }
                case ArithBinaryOp::RotateRight:
                    mask_i8_i16_shift_right_into_result(right);
                    cc.ror(result, left, right);
                    break;
                }

                return result;
            }
            SIMJIT_MATCH (StepKind::ConstDiv) {
                GpR numerator = int_subexpr(data.numerator);
                GpR result = create_int_reg(sdtype);
                emit_scalar_const_div(data, sdtype, result, numerator);
                return result;
            }
            SIMJIT_MATCH (StepKind::FMA) {
                VecR x1 = float_subexpr(data.x1);
                VecR x2 = float_subexpr(data.x2);
                VecR x3 = float_subexpr(data.x3);
                VecR result = suggested_dst != nullptr ? suggested_dst->take().as<VecR>() : create_float_reg(sdtype);
                // AArch64 FMSUB is addend - product, while FNMSUB is product - addend.
                switch (data.kind) {
                case FmaKind::FMA: cc.fmadd(result, x1, x2, x3); break;
                case FmaKind::FMS: cc.fnmsub(result, x1, x2, x3); break;
                case FmaKind::FNMA: cc.fmsub(result, x1, x2, x3); break;
                case FmaKind::FNMS: cc.fnmadd(result, x1, x2, x3); break;
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                if (is_float_dtype(sdtype)) {
                    VecR arg = float_subexpr(data.arg);
                    VecR result = create_float_reg(sdtype);
                    switch (data.op) {
                    case ArithUnaryOp::Lzcnt:
                    case ArithUnaryOp::Tzcnt:
                    case ArithUnaryOp::Popcount:
                        messed_up("Unexpected ArithUnary %s in float context", show_arith_unary_op(data.op));
                    case ArithUnaryOp::Abs: cc.fabs(result, arg); break;
                    case ArithUnaryOp::Not: cc.mvn(result.b8(), arg.b8()); break;
                    case ArithUnaryOp::Negate: cc.fneg(result, arg); break;
                    case ArithUnaryOp::RoundNearest: cc.frintn(result, arg); break;
                    case ArithUnaryOp::RoundDown: cc.frintm(result, arg); break;
                    case ArithUnaryOp::RoundUp: cc.frintp(result, arg); break;
                    case ArithUnaryOp::RoundTruncate: cc.frintz(result, arg); break;
                    case ArithUnaryOp::Rcp: {
                        cc.frecpe(result, arg);
                        VecR tmp = create_float_reg(sdtype);
                        // 1 NR
                        cc.frecps(tmp, arg, result);
                        cc.fmul(result, result, tmp);
                        // 2 NR
                        cc.frecps(tmp, arg, result);
                        cc.fmul(result, result, tmp);
                        if (sdtype == ScalarDataType::F64) {
                            // 3 NR
                            cc.frecps(tmp, arg, result);
                            cc.fmul(result, result, tmp);
                        }
                        break;
                    }
                    case ArithUnaryOp::Sqrt: cc.fsqrt(result, arg); break;
                    case ArithUnaryOp::Rsqrt: {
                        // This also provides more precision than avx512
                        // Do two NR iterations to provide precision comparable to 1/sqrt
                        cc.frsqrte(result, arg);
                        VecR tmp = create_float_reg(sdtype);
                        // 1 NR
                        cc.fmul(tmp, result, result);
                        cc.frsqrts(tmp, arg, tmp);
                        cc.fmul(result, result, tmp);
                        // 2 NR
                        cc.fmul(tmp, result, result);
                        cc.frsqrts(tmp, arg, tmp);
                        cc.fmul(result, result, tmp);
                        if (sdtype == ScalarDataType::F64) {
                            // 3 NR
                            cc.fmul(tmp, result, result);
                            cc.frsqrts(tmp, arg, tmp);
                            cc.fmul(result, result, tmp);
                        }
                        break;
                    }
                    }
                    return result;
                }

                GpR arg = int_subexpr(data.arg);
                GpR result = create_int_reg(sdtype);
                auto handle_i8_i16 = [&]() {
                    if (sdtype == ScalarDataType::I16) {
                        cc.sxth(result, result);
                    } else if (sdtype == ScalarDataType::I8) {
                        cc.sxtb(result, result);
                    }
                };
                switch (data.op) {
                case ArithUnaryOp::Not: cc.mvn(result, arg); break;
                case ArithUnaryOp::Negate:
                    cc.neg(result, arg);
                    handle_i8_i16();
                    break;
                case ArithUnaryOp::Abs:
                    cc.cmp(arg, 0);
                    cc.csneg(result, arg, arg, aja64::CondCode::kGE);
                    handle_i8_i16();
                    break;
                case ArithUnaryOp::Lzcnt:
                    if (sdtype == ScalarDataType::I8) {
                        cc.uxtb(result, arg);
                        arg = result;
                    }
                    if (sdtype == ScalarDataType::I16) {
                        cc.uxth(result, arg);
                        arg = result;
                    }
                    cc.clz(result, arg);
                    if (sdtype == ScalarDataType::I8) { cc.sub(result, result, 24); }
                    if (sdtype == ScalarDataType::I16) { cc.sub(result, result, 16); }
                    break;
                case ArithUnaryOp::Tzcnt:
                    if (sdtype == ScalarDataType::I8) {
                        cc.uxtb(result, arg);
                        cc.orr(result, result, 0x100);
                        cc.rbit(result.r32(), result.r32());
                        cc.clz(result.r32(), result.r32());
                    } else if (sdtype == ScalarDataType::I16) {
                        cc.uxth(result, arg);
                        cc.orr(result, result, 0x10000);
                        cc.rbit(result.r32(), result.r32());
                        cc.clz(result.r32(), result.r32());
                    } else {
                        cc.rbit(result, arg);
                        cc.clz(result, result);
                    }
                    break;
                case ArithUnaryOp::Popcount: {
                    VecR vec = sdtype == ScalarDataType::I64 ? cc.new_vec128().d() : cc.new_vec128().s();
                    // popcount is NEON only
                    if (sdtype == ScalarDataType::I8) {
                        cc.uxtb(result, arg);
                        arg = result;
                    }
                    if (sdtype == ScalarDataType::I16) {
                        cc.uxth(result, arg);
                        arg = result;
                    }
                    cc.fmov(vec, arg);
                    cc.cnt(vec.b8(), vec.b8());
                    cc.addv(vec.b(), vec.b8());
                    cc.fmov(result, vec);
                } break;
                case ArithUnaryOp::RoundNearest:
                case ArithUnaryOp::RoundDown:
                case ArithUnaryOp::RoundUp:
                case ArithUnaryOp::RoundTruncate:
                case ArithUnaryOp::Rcp:
                case ArithUnaryOp::Sqrt:
                case ArithUnaryOp::Rsqrt:
                    messed_up("Unexpected ArithUnary %s in int context", show_arith_unary_op(data.op));
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                GpR result = create_int_reg(sdtype);
                ScalarDataType from = data.arg->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                bool is_sext = data.kind == IntCastKind::Sext;
                GpR arg = int_subexpr(data.arg);

                if (scalar_dtype_size(from) > scalar_dtype_size(to)) {
                    SIMJIT_ASSERT(data.kind == IntCastKind::Trunc);
                    switch (to) {
                    case ScalarDataType::I8: cc.sxtb(result, arg.w()); break;
                    case ScalarDataType::I16: cc.sxth(result, arg.w()); break;
                    case ScalarDataType::I32: cc.mov(result, arg.w()); break;
                    case ScalarDataType::I64:
                    case ScalarDataType::I1:
                    case ScalarDataType::I128:
                    case ScalarDataType::F32:
                    case ScalarDataType::F64:
                        messed_up("Invalid cast from %s to %s", show_scalar_dtype(from), show_scalar_dtype(to));
                    }
                    return result;
                }
                if (is_sext) {
                    if (from == ScalarDataType::I8) {
                        cc.sxtb(result, arg);
                    } else if (from == ScalarDataType::I16) {
                        cc.sxth(result, arg);
                    } else {
                        cc.sxtw(result, arg);
                    }
                } else {
                    if (from == ScalarDataType::I8) {
                        cc.uxtb(result.w(), arg);
                    } else if (from == ScalarDataType::I16) {
                        cc.uxth(result.w(), arg);
                    } else {
                        cc.mov(result.w(), arg);
                    }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                ScalarDataType from = data.arg->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                if (is_float_dtype(from) && is_float_dtype(to)) {
                    VecR arg = float_subexpr(data.arg);
                    VecR result = create_float_reg(sdtype);
                    cc.fcvt(result, arg);
                    return result;
                }
                if (is_float_dtype(to)) {
                    GpR arg = int_subexpr(data.arg);
                    VecR result = create_float_reg(sdtype);
                    if (data.is_unsigned) {
                        cc.ucvtf(result, arg);
                    } else {
                        cc.scvtf(result, arg);
                    }
                    return result;
                }
                VecR arg = float_subexpr(data.arg);
                GpR result = create_int_reg(sdtype);
                if (data.is_unsigned) {
                    cc.fcvtzu(result, arg);
                } else {
                    cc.fcvtzs(result, arg);
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::BitCast) {
                ScalarDataType from = data->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                (void)to;
                SIMJIT_ASSERT((is_simple_int_dtype(from) && is_float_dtype(to)) ||
                              (is_simple_int_dtype(to) && is_float_dtype(from)));
                if (is_simple_int_dtype(from)) {
                    SIMJIT_ASSERT(is_float_dtype(to));
                    GpR arg = int_subexpr(data);
                    VecR result = create_float_reg(sdtype);
                    cc.fmov(result, arg);
                    return result;
                }
                SIMJIT_ASSERT(is_simple_int_dtype(to));
                VecR arg = float_subexpr(data);
                GpR result = create_int_reg(sdtype);
                cc.fmov(result, arg);
                return result;
            }
            SIMJIT_MATCH (StepKind::Compare) {
                if (is_float_dtype(data.left->dtype.as_scalar())) {
                    VecR left = float_subexpr(data.left);
                    GpR result = cc.new_gp32();
                    if (SIMJIT_A64_ASMJIT_CONST_OPS && step_is_zero(data.right)) {
                        cc.fcmp(left, 0);
                    } else {
                        VecR right = float_subexpr(data.right);
                        cc.fcmp(left, right);
                    }
                    switch (data.op) {
                    case CmpOp::Less: cc.cset(result, aja64::CondCode::kMI); break;
                    case CmpOp::Greater: cc.cset(result, aja64::CondCode::kGT); break;
                    case CmpOp::LessEqual: cc.cset(result, aja64::CondCode::kLS); break;
                    case CmpOp::GreaterEqual: cc.cset(result, aja64::CondCode::kGE); break;
                    case CmpOp::Equal: cc.cset(result, aja64::CondCode::kEQ); break;
                    case CmpOp::NotEqual: cc.cset(result, aja64::CondCode::kNE); break;
                    }
                    return result;
                }
                GpR left = int_subexpr(data.left);
                GpR result = cc.new_gp32();

                const ConstData *right_const =
                    data.right->is(StepKind::Const) ? &data.right->step_data<StepKind::Const>() : nullptr;
                ScalarDataType cmp_dtype = data.left->dtype.as_scalar();
                bool can_cmp_imm = false;
                uint64_t cmp_imm = 0;
                if (SIMJIT_A64_ASMJIT_CONST_OPS && right_const != nullptr) {
                    if (data.is_unsigned) {
                        cmp_imm = right_const->retag(cmp_dtype).as_unsigned();
                        can_cmp_imm = aja64::Utils::is_add_sub_imm(cmp_imm);
                    } else {
                        int64_t signed_imm = right_const->retag(cmp_dtype).as_signed();
                        can_cmp_imm = can_cmp_signed_imm(signed_imm);
                        cmp_imm = (uint64_t)signed_imm;
                    }
                }
                GpR cmp_left = left;
                if (data.is_unsigned && (cmp_dtype == ScalarDataType::I8 || cmp_dtype == ScalarDataType::I16)) {
                    cmp_left = zero_extend_small_int_for_unsigned_cmp(left, cmp_dtype);
                }
                // Arm cmp is alias to subs and uses 12-bit immediate encoding.
                if (can_cmp_imm) {
                    cc.cmp(cmp_left, cmp_imm);
                } else if (right_const != nullptr) {
                    GpR right = create_int_reg(cmp_dtype);
                    init_int_const(right, *right_const, cmp_dtype);
                    if (data.is_unsigned && (cmp_dtype == ScalarDataType::I8 || cmp_dtype == ScalarDataType::I16)) {
                        right = zero_extend_small_int_for_unsigned_cmp(right, cmp_dtype);
                    }
                    cc.cmp(cmp_left, right);
                } else {
                    GpR right = int_subexpr(data.right);
                    if (data.is_unsigned && (cmp_dtype == ScalarDataType::I8 || cmp_dtype == ScalarDataType::I16)) {
                        right = zero_extend_small_int_for_unsigned_cmp(right, cmp_dtype);
                    }
                    cc.cmp(cmp_left, right);
                }
                if (data.is_unsigned) {
                    aja64::CondCode code = map_cmp_op_unsigned(data.op);
                    cc.cset(result, code);
                } else {
                    aja64::CondCode code = map_cmp_op_signed(data.op);
                    cc.cset(result, code);
                }

                return result;
            }
            SIMJIT_MATCH (StepKind::AggResult) {
                const ArgInfo &info = args[data.dst];
                AnyR arg = subexpr(data.arg);
                store(cc, arg, aja64::ptr(info.gp), sdtype);
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
                        cc.adds(low, low, lo_val);
                        cc.adc(high, high, zero);
                    } else {
                        VecR vec = vec_subexpr(lo_step);
                        GpR lo_val = cc.new_gp64();
                        cc.umov(lo_val, vec.d(0));
                        cc.adds(low, low, lo_val);
                        cc.adc(high, high, zero);
                        cc.umov(lo_val, vec.d(1));
                        cc.adds(low, low, lo_val);
                        cc.adc(high, high, zero);
                    }
                }

                cc.stp(low, high, aja64::ptr(info.gp));
                return {};
            }
            SIMJIT_MATCH (StepKind::AccLoad) { return accs[mir_func->accs.index(data)]; }
            SIMJIT_MATCH (StepKind::AccStore) {
                if (is_float_dtype(sdtype)) {
                    VecR acc_reg = accs[mir_func->accs.index(data.acc)].as<VecR>();
                    SuggestedReg suggest{acc_reg};
                    VecR arg = float_subexpr(data.arg, &suggest);
                    if (!suggest.acknowledge) cc.fmov(acc_reg, arg);
                    return {};
                }
                GpR acc_reg = accs[mir_func->accs.index(data.acc)].as<GpR>();
                SuggestedReg suggest{acc_reg};
                GpR arg = int_subexpr(data.arg, &suggest);
                if (!suggest.acknowledge) cc.mov(acc_reg, arg);
                return {};
            }
            SIMJIT_MATCH (StepKind::PredicateNot) {
                GpR arg = int_subexpr(data);
                GpR result = cc.new_gp32();
                cc.eor(result, arg, 1);
                return result;
            }
            SIMJIT_MATCH (StepKind::Select) {
                if (is_float_dtype(sdtype)) {
                    VecR falsy = float_subexpr(data.falsy);
                    VecR truthy = float_subexpr(data.truthy);
                    VecR result = create_float_reg(sdtype);
                    if (SIMJIT_A64_ASMJIT_INLINE_SCALAR_COND && data.cond->is(StepKind::Compare) &&
                        is_float_dtype(data.cond->step_data<StepKind::Compare>().left->dtype.as_scalar())) {
                        const auto &cond_data = data.cond->step_data<StepKind::Compare>();
                        if (cond_data.op == CmpOp::Less || cond_data.op == CmpOp::LessEqual ||
                            cond_data.op == CmpOp::Greater || cond_data.op == CmpOp::GreaterEqual ||
                            cond_data.op == CmpOp::Equal) {
                            if (SIMJIT_A64_ASMJIT_CONST_OPS && step_is_zero(cond_data.right)) {
                                cc.fcmp(vec_subexpr(cond_data.left), 0);
                            } else {
                                cc.fcmp(vec_subexpr(cond_data.left), vec_subexpr(cond_data.right));
                            }
                            aja64::CondCode cond;
                            switch (cond_data.op) {
                            case CmpOp::Less: cond = aja64::CondCode::kMI; break;
                            case CmpOp::Greater: cond = aja64::CondCode::kGT; break;
                            case CmpOp::LessEqual: cond = aja64::CondCode::kLS; break;
                            case CmpOp::GreaterEqual: cond = aja64::CondCode::kGE; break;
                            case CmpOp::Equal: cond = aja64::CondCode::kEQ; break;
                            default: SIMJIT_UNREACHABLE();
                            }
                            cc.fcsel(result, truthy, falsy, cond);
                            return result;
                        }
                    }
                    GpR cond = int_subexpr(data.cond);
                    cc.cmp(cond, 0);
                    cc.fcsel(result, truthy, falsy, aja64::CondCode::kNE);
                    return result;
                }
                GpR falsy = int_subexpr(data.falsy);
                GpR truthy = int_subexpr(data.truthy);
                GpR result = create_int_reg(sdtype);
                if (SIMJIT_A64_ASMJIT_INLINE_SCALAR_COND && data.cond->is(StepKind::Compare) &&
                    !is_float_dtype(data.cond->step_data<StepKind::Compare>().left->dtype.as_scalar())) {
                    auto &cond_data = data.cond->step_data<StepKind::Compare>();
                    const ConstData *right_const =
                        cond_data.right->is(StepKind::Const) ? &cond_data.right->step_data<StepKind::Const>() : nullptr;
                    ScalarDataType cmp_dtype = cond_data.left->dtype.as_scalar();
                    bool can_cmp_imm = false;
                    uint64_t cmp_imm = 0;
                    if (SIMJIT_A64_ASMJIT_CONST_OPS && right_const != nullptr) {
                        if (cond_data.is_unsigned) {
                            cmp_imm = right_const->retag(cmp_dtype).as_unsigned();
                            can_cmp_imm = aja64::Utils::is_add_sub_imm(cmp_imm);
                        } else {
                            int64_t signed_imm = right_const->retag(cmp_dtype).as_signed();
                            can_cmp_imm = can_cmp_signed_imm(signed_imm);
                            cmp_imm = (uint64_t)signed_imm;
                        }
                    }
                    GpR cmp_left = int_subexpr(cond_data.left);
                    if (cond_data.is_unsigned &&
                        (cmp_dtype == ScalarDataType::I8 || cmp_dtype == ScalarDataType::I16)) {
                        cmp_left = zero_extend_small_int_for_unsigned_cmp(cmp_left, cmp_dtype);
                    }
                    if (can_cmp_imm) {
                        cc.cmp(cmp_left, cmp_imm);
                    } else if (right_const != nullptr) {
                        GpR right = create_int_reg(cmp_dtype);
                        init_int_const(right, *right_const, cmp_dtype);
                        if (cond_data.is_unsigned &&
                            (cmp_dtype == ScalarDataType::I8 || cmp_dtype == ScalarDataType::I16)) {
                            right = zero_extend_small_int_for_unsigned_cmp(right, cmp_dtype);
                        }
                        cc.cmp(cmp_left, right);
                    } else {
                        GpR right = int_subexpr(cond_data.right);
                        if (cond_data.is_unsigned &&
                            (cmp_dtype == ScalarDataType::I8 || cmp_dtype == ScalarDataType::I16)) {
                            right = zero_extend_small_int_for_unsigned_cmp(right, cmp_dtype);
                        }
                        cc.cmp(cmp_left, right);
                    }
                    aja64::CondCode code =
                        cond_data.is_unsigned ? map_cmp_op_unsigned(cond_data.op) : map_cmp_op_signed(cond_data.op);
                    cc.csel(result, truthy, falsy, code);
                    return result;
                }
                GpR cond = int_subexpr(data.cond);
                cc.cmp(cond, 0);
                cc.csel(result, truthy, falsy, aja64::CondCode::kNE);
                return result;
            }
            SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
                GpR left = int_subexpr(data.left);
                GpR right = int_subexpr(data.right);
                GpR result = create_int_reg(sdtype);
                GpR overflow_flag = cc.new_gp32();
                bool is_small = sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16;
                auto extract_small_overflow = [&](ArithBinaryOp op) {
                    GpR different_signs = cc.new_gp32();
                    GpR changed_sign = cc.new_gp32();
                    cc.eor(different_signs, left, right);
                    cc.eor(changed_sign, left, result);
                    if (op == ArithBinaryOp::Add) {
                        cc.bic(changed_sign, changed_sign, different_signs);
                    } else {
                        cc.and_(changed_sign, changed_sign, different_signs);
                    }
                    cc.lsr(overflow_flag, changed_sign, scalar_dtype_bits(sdtype) - 1);
                    cc.and_(overflow_flag, overflow_flag, 1);
                };
                switch (data.op) {
                case ArithBinaryOp::Add:
                    if (is_small) {
                        cc.add(result, left, right);
                        extract_small_overflow(data.op);
                    } else {
                        cc.adds(result, left, right);
                        cc.cset(overflow_flag, aja64::CondCode::kOverflow);
                    }
                    break;
                case ArithBinaryOp::Sub:
                    if (is_small) {
                        cc.sub(result, left, right);
                        extract_small_overflow(data.op);
                    } else {
                        cc.subs(result, left, right);
                        cc.cset(overflow_flag, aja64::CondCode::kOverflow);
                    }
                    break;
                case ArithBinaryOp::Mul:
                    if (sdtype == ScalarDataType::I32) {
                        cc.smull(result.x(), left, right);
                        cc.cmp(result.x(), result.w(), aja64::sxtw(0));
                        cc.cset(overflow_flag, aja64::CondCode::kNE);
                    } else {
                        SIMJIT_ASSERT(sdtype == ScalarDataType::I64);
                        cc.mul(result, left, right);
                        GpR tmp = cc.new_gp64();
                        cc.smulh(tmp, left, right);
                        cc.cmp(tmp, result, aja64::asr(63));
                        cc.cset(overflow_flag, aja64::CondCode::kNE);
                    }
                    break;
                default: messed_up("Invalid overflow step %s", show_arith_binary_op(data.op));
                }
                if (data.mask != nullptr) { cc.and_(overflow_flag, overflow_flag, int_subexpr(data.mask)); }
                if (sdtype == ScalarDataType::I8) {
                    cc.sxtb(result, result);
                } else if (sdtype == ScalarDataType::I16) {
                    cc.sxth(result, result);
                }
                GpR overflow_acc = accs[mir_func->accs.index(data.overflow_flag)].as<GpR>();
                cc.orr(overflow_acc, overflow_acc, overflow_flag);
                return result;
            }
            SIMJIT_MATCH (StepKind::ScalarIndex) {
                if (sdtype == ScalarDataType::I32) { return counter.r32(); }
                return counter;
            }
            SIMJIT_MATCH (StepKind::ScalarPermute) {
                GpR result = create_int_reg(sdtype);
                GpR arg = int_subexpr(data.arg);
                if (data.is_bit) {
                    if (data.permute == REVERSE_BITS) {
                        cc.rbit(result, arg);
                        switch (sdtype) {
                        case ScalarDataType::I8: cc.lsr(result, result, 24); break;
                        case ScalarDataType::I16:
                            cc.lsr(result, result, 16);
                            cc.rev16(result, result);
                            break;
                        case ScalarDataType::I32: cc.rev32(result, result); break;
                        case ScalarDataType::I64: cc.rev64(result, result); break;
                        case ScalarDataType::F32:
                        case ScalarDataType::F64:
                        case ScalarDataType::I1:
                        case ScalarDataType::I128:
                            messed_up("Unexpected type %s in reverse_bits", show_scalar_dtype(sdtype));
                        }
                        return result;
                    }
                } else {
                    if (sdtype == ScalarDataType::I16 && data.permute == REVERSE_BYTES_I16) {
                        cc.rev16(result, arg);
                        return result;
                    }
                    if (sdtype == ScalarDataType::I32 && data.permute == REVERSE_BYTES_I32) {
                        cc.rev32(result, arg);
                        return result;
                    }
                    if (sdtype == ScalarDataType::I64 && data.permute == REVERSE_BYTES_I64) {
                        cc.rev64(result, arg);
                        return result;
                    }
                }

                GpR tmp = create_int_reg(sdtype);
                cc.mov(result, 0);

                size_t dtype_size = scalar_dtype_size(step->dtype.as_scalar());
                if (data.is_bit) {
                    for (size_t i = 0; i < dtype_size * 8; ++i) {
                        size_t permute_idx = (((data.permute >> ((i & 0x7) * 8)) - 1) & 0xff);
                        size_t imm = permute_idx + (i / 8 * 8);
                        cc.lsr(tmp, arg, imm);
                        cc.and_(tmp, tmp, 1);
                        if (i != 0) cc.lsl(tmp, tmp, i);
                        cc.orr(result, result, tmp);
                    }
                } else {
                    for (size_t i = 0; i < dtype_size; ++i) {
                        size_t imm = ((data.permute >> (i * 8)) & 0xff) * 8;
                        cc.lsr(tmp, arg, imm);
                        cc.and_(tmp, tmp, 0xff);
                        if (i != 0) cc.lsl(tmp, tmp, i * 8);
                        cc.orr(result, result, tmp);
                    }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                const ArgInfo &arg_info = args[data.dst];
                AnyR arg = subexpr(data.arg);
                GpR idx = int_subexpr(data.idx);
                aja64::Mem mem = mem_offset(arg_info.gp, idx, sdtype);
                store(cc, arg, mem, sdtype);
                return {};
            }
            SIMJIT_MATCH (StepKind::CondScatter) {
                const ArgInfo &arg_info = args[data.dst];
                AnyR arg = subexpr(data.arg);
                GpR cond = int_subexpr(data.cond);
                GpR idx = int_subexpr(data.idx);
                aja64::Mem mem = mem_offset(arg_info.gp, idx, sdtype);
                aj::Label skip_label = cc.new_anonymous_label("skip");
                cc.cmp(cond, 0);
                cc.b(aja64::CondCode::kEQ, skip_label);
                store(cc, arg, mem, sdtype);
                cc.bind(skip_label);
                return {};
            }
            SIMJIT_MATCH (StepKind::Pack) {
                const ArgInfo &arg_info = args[data.dst];
                GpR acc = accs[mir_func->accs.index(data.acc)].as<GpR>();
                AnyR arg = subexpr(data.arg);
                GpR cond = int_subexpr(data.cond);
                aja64::Mem mem = mem_offset(arg_info.gp, acc, sdtype);
                aj::Label skip_label = cc.new_anonymous_label("skip");
                cc.cbz(cond, skip_label);
                store(cc, arg, mem, sdtype);
                cc.add(acc, acc, 1);
                cc.bind(skip_label);
                return {};
            }
            SIMJIT_MATCH (StepKind::CondStore) {
                const ArgInfo &arg_info = args[data.addr.arg];
                AnyR arg = subexpr(data.arg);
                GpR cond = int_subexpr(data.cond);
                aja64::Mem mem = mem_offset(arg_info.gp, counter, sdtype);
                aj::Label skip_label = cc.new_anonymous_label("skip");
                cc.cbz(cond, skip_label);
                store(cc, arg, mem, sdtype);
                cc.bind(skip_label);
                return {};
            }
            SIMJIT_MATCH (StepKind::Fpclass) {
                bool is_f32 = data.arg->dtype == ScalarDataType::F32;
                VecR arg = float_subexpr(data.arg);
                std::optional<GpR> acc;
                auto accumulate = [&](const GpR &x) {
                    if (acc) {
                        cc.orr(*acc, *acc, x);
                    } else {
                        acc = x;
                    }
                };

                if (bool(data.flags & FpClass::FPC_INFINITE)) {
                    if (is_f32) {
                        GpR tmp1 = cc.new_gp32();
                        cc.fmov(tmp1, arg);
                        GpR tmp2 = cc.new_gp32();
                        cc.mov(tmp2, 0x7F800000);
                        cc.and_(tmp1, tmp1, 0x7fffffff);
                        cc.cmp(tmp1, tmp2);
                        GpR x = cc.new_gp32();
                        cc.cset(x, aja64::CondCode::kEQ);
                        accumulate(x);
                    } else {
                        GpR tmp1 = cc.new_gp64();
                        cc.fmov(tmp1, arg);
                        GpR tmp2 = cc.new_gp64();
                        cc.mov(tmp2, 0x7FF0000000000000);
                        cc.and_(tmp1, tmp1, 0x7fffffffffffffff);
                        cc.cmp(tmp1, tmp2);
                        GpR x = cc.new_gp32();
                        cc.cset(x, aja64::CondCode::kEQ);
                        accumulate(x);
                    }
                }
                if (bool(data.flags & FpClass::FPC_NAN)) {
                    cc.fcmp(arg, arg);
                    GpR x = cc.new_gp32();
                    cc.cset(x, aja64::CondCode::kNE);
                    accumulate(x);
                }

                if (bool(data.flags & FpClass::FPC_SUBNORMAL)) {
                    if (is_f32) {
                        GpR tmp1 = cc.new_gp32();
                        cc.fmov(tmp1, arg);
                        GpR exp = cc.new_gp32();
                        cc.and_(exp, tmp1, 0x7F800000);
                        GpR mant = cc.new_gp32();
                        GpR x = cc.new_gp32();
                        cc.cmp(exp, 0);
                        cc.cset(x, aja64::CondCode::kEQ);
                        cc.and_(mant, tmp1, 0x007FFFFF);
                        GpR y = cc.new_gp32();
                        cc.cmp(mant, 0);
                        cc.cset(y, aja64::CondCode::kNE);
                        cc.and_(x, x, y);
                        accumulate(x);
                    } else {
                        GpR tmp1 = cc.new_gp64();
                        cc.fmov(tmp1, arg);
                        GpR exp = cc.new_gp64();
                        cc.and_(exp, tmp1, 0x7FF0000000000000);
                        GpR mant = cc.new_gp64();
                        GpR x = cc.new_gp32();
                        cc.cmp(exp, 0);
                        cc.cset(x, aja64::CondCode::kEQ);
                        cc.and_(mant, tmp1, 0x000FFFFFFFFFFFFF);
                        GpR y = cc.new_gp32();
                        cc.cmp(mant, 0);
                        cc.cset(y, aja64::CondCode::kNE);
                        cc.and_(x, x, y);
                        accumulate(x);
                    }
                }
                if (bool(data.flags & FpClass::FPC_ZERO)) {
                    VecR tmp = create_float_reg(data.arg->dtype.as_scalar());
                    cc.fabs(tmp, arg);
                    cc.fcmp(tmp, 0);
                    GpR x = cc.new_gp32();
                    cc.cset(x, aja64::CondCode::kEQ);
                    accumulate(x);
                }
                SIMJIT_ASSERT(acc);
                return *acc;
            }

        default: messed_up("Unexpected step %s in scalar context", show_step_kind(step->kind));
        }
        SIMJIT_UNREACHABLE();
    }

    SIMJIT_ALWAYS_INLINE SIMJIT_NO_ASAN AnyR subexpr(const Step *step, SuggestedReg *suggested_dst = nullptr) {
        if (step_map[step->id].is_valid()) { return step_map[step->id]; }
        SIMJIT_ASSERT(!const_is_folded_root(step));
        AnyR v = subexpr_internal(step, suggested_dst);
        step_map[step->id] = v;
        return v;
    }

    SIMJIT_ALWAYS_INLINE SIMJIT_NO_ASAN GpR int_subexpr(const Step *step, SuggestedReg *suggested_dst = nullptr) {
        AnyR reg = subexpr(step, suggested_dst);
        if (!reg.is_gp()) {
            SIMJIT_ASSERT(0);
            messed_up("Expected int as result of step %s", show_step_kind(step->kind));
        }
        return reg.as<GpR>();
    }
    SIMJIT_ALWAYS_INLINE SIMJIT_NO_ASAN VecR float_subexpr(const Step *step, SuggestedReg *suggested_dst = nullptr) {
        AnyR reg = subexpr(step, suggested_dst);
        if (!reg.is_vec()) {
            SIMJIT_ASSERT(0);
            messed_up("Expected float as result of step %s", show_step_kind(step->kind));
        }
        return reg.as<VecR>();
    }

    SIMJIT_ALWAYS_INLINE SIMJIT_NO_ASAN VecR vec_subexpr(const Step *step, SuggestedReg *suggested_dst = nullptr) {
        AnyR reg = subexpr(step, suggested_dst);
        if (!reg.is_vec()) {
            SIMJIT_ASSERT(0);
            messed_up("Expected vec as result of step %s", show_step_kind(step->kind));
        }
        return reg.as<VecR>();
    }

    SIMJIT_NO_ASAN AnyR subexpr_internal(const Step *step, SuggestedReg *suggested_dst = nullptr) {
        if (step->dtype.is_scalar() && is_scalar_step(step->kind)) {
#if defined(__clang__)
            [[clang::musttail]]
#endif
            return scalar_subexpr(step, suggested_dst);
        }

        switch (step->kind) {
        case StepKind::AggResult:
        case StepKind::ScalarIndex:
        case StepKind::ScalarArithBinaryOverflow:
        case StepKind::ScalarPermute:
        case StepKind::StoreSum128:
        case StepKind::ConstDiv:
            SIMJIT_ASSERT(0);
            messed_up("Unexpected instruction %s", show_step_kind(step->kind));

            SIMJIT_MATCH (StepKind::Gather) return emit_vector_gather(data, step->dtype.as_vec());
            SIMJIT_MATCH (StepKind::Scatter) unsupported("Do not support scatter");
            SIMJIT_MATCH (StepKind::CondScatter) unsupported("Do not support cond scatter");
            SIMJIT_MATCH (StepKind::Ternarylogic) unsupported("Do not support ternarylogic");

            SIMJIT_MATCH (StepKind::Pack) {
                VecDataType vdtype = step->dtype.as_vec();
                if (vdtype.elem == VecElemType::F32) { vdtype.elem = VecElemType::I32; }
                if (vdtype.elem == VecElemType::F64) { vdtype.elem = VecElemType::I64; }
                GpR acc = accs[mir_func->accs.index(data.acc)].as<GpR>();
                VecR arg = vec_subexpr(data.arg);
                VecR cond = vec_subexpr(data.cond);
                // NOTE: We can implement special version for packing index, but it does not seem worth the
                // trouble
                if (vdtype.elem != VecElemType::I8 && vdtype.elem != VecElemType::I16 &&
                    vdtype.elem != VecElemType::I32 && vdtype.elem != VecElemType::I64) {
                    unsupported("Do not supprt %s pack (yet)", show_vec_dtype(step->dtype.as_vec()));
                }

                VecR tmp = create_vec_reg(vdtype);
                size_t cond_sz = vec_elem_size(cond);
                if (cond_sz != vdtype.element_size_bytes()) {
                    if (cond_sz < vdtype.element_size_bytes()) {
                        cond = expand_mask(cond, cond_sz, vdtype.element_size_bytes());
                    } else {
                        cond = shrink_mask(cond, cond_sz, vdtype.element_size_bytes());
                    }
                }
                GpR base = args[data.dst].gp;
                GpR addr = cc.new_gp64();
                if (vdtype.elem == VecElemType::I8) {
                    cc.and_(tmp.b16(), cond.b16(), vec_special_const(SpecialConstant::I8_MaskBits).b16());

                    VecR high_tmp = cc.new_vec128();
                    cc.ext(high_tmp.b16(), tmp.b16(), tmp.b16(), 8);

                    cc.addv(tmp.b(), tmp.b8());
                    GpR tmpGp = cc.new_gp64();
                    cc.umov(tmpGp.w(), tmp.b(0));

                    VecR idx = cc.new_vec128();
                    cc.ldr(idx.q(),
                           aja64::ptr(gp_special_const(SpecialConstant::I8_PackIndices), tmpGp.w(), aja64::uxtw(4)));
                    cc.tbl(idx.b16(), arg.b16(), idx.b16());
                    cc.add(addr, base, acc);
                    cc.str(idx.d(), aja64::ptr(addr));
                    add_u8_popcount_to_acc(acc, tmpGp);

                    cc.addv(high_tmp.b(), high_tmp.b8());
                    cc.umov(tmpGp.w(), high_tmp.b(0));
                    cc.ldr(idx.q(),
                           aja64::ptr(gp_special_const(SpecialConstant::I8_PackIndices), tmpGp.w(), aja64::uxtw(4)));
                    VecR high_arg = cc.new_vec128();
                    cc.ext(high_arg.b16(), arg.b16(), arg.b16(), 8);
                    cc.tbl(idx.b16(), high_arg.b16(), idx.b16());
                    cc.add(addr, base, acc);
                    cc.str(idx.d(), aja64::ptr(addr));
                    add_u8_popcount_to_acc(acc, tmpGp);
                    return {};
                }
                if (vdtype.elem == VecElemType::I16) {
                    cc.and_(tmp.b16(), cond.b16(), vec_special_const(SpecialConstant::I16_MaskBits).b16());
                    cc.addv(tmp.h(), tmp.h8());
                    GpR tmpGp = cc.new_gp64();
                    cc.umov(tmpGp.w(), tmp.h(0));

                    VecR idx = cc.new_vec128();
                    cc.ldr(idx.q(),
                           aja64::ptr(gp_special_const(SpecialConstant::I16_PackIndices), tmpGp.w(), aja64::uxtw(4)));
                    cc.tbl(idx.b16(), arg.b16(), idx.b16());
                    cc.add(addr, base, acc, aja64::lsl(1));
                    cc.str(idx.q(), aja64::ptr(addr));
                    add_u8_popcount_to_acc(acc, tmpGp);
                    return {};
                }

                auto mask =
                    vdtype.elem == VecElemType::I64 ? SpecialConstant::I64_MaskBits : SpecialConstant::I32_MaskBits;
                cc.and_(tmp.b16(), cond.b16(), vec_special_const(mask).b16());
                cc.addv(tmp.s(), tmp.s4());
                GpR tmpGp = cc.new_gp64();
                cc.fmov(tmpGp.w(), tmp.s());
                VecR idx = cc.new_vec128();
                auto pack_idx = vdtype.elem == VecElemType::I64 ? SpecialConstant::I64_PackIndices
                                                                : SpecialConstant::I32_PackIndices;
                cc.ldr(idx.q(), aja64::ptr(gp_special_const(pack_idx), tmpGp.w(), aja64::uxtw(4)));
                cc.tbl(idx.b16(), arg.b16(), idx.b16());
                cc.add(addr, base, acc, aja64::lsl(vdtype.elem == VecElemType::I64 ? 3 : 2));
                cc.str(idx.q(), aja64::ptr(addr));
                if (vdtype.elem == VecElemType::I64) {
                    // 2 bit integer. It is trivial to count manually
                    // Here we do: x - (x >> 1)
                    cc.sub(tmpGp.w(), tmpGp.w(), tmpGp.w(), aja64::lsr(1));
                } else {
                    GpR x = cc.new_gp32();
                    // Here we have 4 bit integer. Not so trivial to count manually, but still easy.
                    // Use same trick as for 2 bit integer, since it uses inline shifts nicely and looks smart.
                    cc.lsr(addr.w(), tmpGp.w(), 2);
                    cc.sub(x, addr.w(), addr.w(), aja64::lsr(1));
                    cc.and_(addr.w(), tmpGp.w(), 0b11);
                    cc.sub(tmpGp.w(), addr.w(), addr.w(), aja64::lsr(1));
                    cc.add(tmpGp.w(), tmpGp.w(), x);
                }
                cc.add(acc, acc, tmpGp);
                return {};
            }
            SIMJIT_MATCH (StepKind::Const) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    VecR result = cc.new_vec128();
                    uint8_t imm = data.is_zero() ? 0 : 0xff;
                    switch (mdtype) {
                    case MaskDataType::M2: result = result.d2(); break;
                    case MaskDataType::M4: result = result.s4(); break;
                    case MaskDataType::M8: result = result.h8(); break;
                    case MaskDataType::M16: result = result.b16(); break;
                    case MaskDataType::M32:
                    case MaskDataType::M64: messed_up("Invalid mask const of type %s", show_mask_dtype(mdtype));
                    }
                    cc.movi(result.b16(), imm);
                    return result;
                }

                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                switch (vdtype.elem) {
                case VecElemType::I8: cc.movi(reg, data.as_unsigned() & 0xff); break;
                case VecElemType::I16: {
                    if (data.as_unsigned() <= 0xff) {
                        cc.movi(reg, data.as_unsigned());
                        break;
                    }
                    GpR tmp = cc.new_gp32();
                    init_int_const(tmp, data, ScalarDataType::I16);
                    cc.dup(reg, tmp);
                    break;
                }
                case VecElemType::I32: {
                    if (data.as_unsigned() <= 0xff) {
                        cc.movi(reg, data.as_unsigned());
                        break;
                    }
                    GpR tmp = cc.new_gp32();
                    init_int_const(tmp, data, ScalarDataType::I32);
                    cc.dup(reg, tmp);
                    break;
                }
                case VecElemType::I64: {
                    if (aja64::Utils::is_byte_mask_imm(data.as_unsigned())) {
                        cc.movi(reg, data.as_unsigned());
                        break;
                    }
                    GpR tmp = cc.new_gp64();
                    init_int_const(tmp, data, ScalarDataType::I64);
                    cc.dup(reg, tmp);
                    break;
                }
                case VecElemType::F32: {
                    init_f32_const(reg, data.as_f32(), true);
                    break;
                }
                case VecElemType::F64: {
                    init_f64_const(reg, data.as_f64(), true);
                    break;
                }
                }
                return reg;
            }
            SIMJIT_MATCH (StepKind::VecConst) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                aja64::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, data.mem, vdtype.size_bytes());
                cc.ldr(reg, mem);
                return reg;
            }
            SIMJIT_MATCH (StepKind::Load) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    VecR result = cc.new_vec128();
                    switch (mdtype) {
                    case MaskDataType::M2: {
                        GpR row = mask_row_offset(data.addr.offset);
                        aja64::Mem mem = small_mask_mem(data.addr, row);
                        GpR loaded = cc.new_gp64();
                        cc.ldrb(loaded.w(), mem);
                        GpR tmp = cc.new_gp64();
                        cc.and_(tmp, row, 7);
                        cc.lsr(loaded, loaded, tmp);
                        cc.dup(result.d2(), loaded);
                        cc.cmtst(result.d2(), result.d2(), vec_special_const(SpecialConstant::I64_MaskBits).d2());
                        return result.d2();
                    }
                    case MaskDataType::M4: {
                        GpR row = mask_row_offset(data.addr.offset);
                        aja64::Mem mem = small_mask_mem(data.addr, row);
                        GpR loaded = cc.new_gp32();
                        cc.ldrb(loaded, mem);
                        GpR tmp = cc.new_gp32();
                        cc.and_(tmp, row.r32(), 7);
                        cc.lsr(loaded, loaded, tmp);
                        cc.dup(result.s4(), loaded);
                        cc.cmtst(result.s4(), result.s4(), vec_special_const(SpecialConstant::I32_MaskBits).s4());
                        return result.s4();
                    }
                    case MaskDataType::M8: {
                        aja64::Mem mem = mask_mem(data.addr, mdtype);
                        cc.ld1r(result.b8(), mem);
                        cc.cmtst(result.b8(), result.b8(), vec_special_const(SpecialConstant::Bits).b8());
                        return result.b8();
                    }
                    case MaskDataType::M16: {
                        aja64::Mem mem = mask_mem(data.addr, mdtype);
                        cc.ld1r(result.h4(), mem);
                        cc.tbl(result.b16(), result.b16(), vec_special_const(SpecialConstant::Zero8One8).b16());
                        cc.cmtst(result.b16(), result.b16(), vec_special_const(SpecialConstant::Bits).b16());
                        return result.b16();
                    }
                    case MaskDataType::M32:
                    case MaskDataType::M64: messed_up("unexpected large mask");
                    }
                    SIMJIT_UNREACHABLE();
                }
                VecDataType vdtype = step->dtype.as_vec();

                if (SIMJIT_A64_ASMJIT_LDP) {
                    if (auto *it = paired_ops[step->id]) {
                        VecR vec1 = create_vec_reg(vdtype);
                        VecR vec2 = create_vec_reg(vdtype);
                        aja64::Mem mem = vec_pair_mem(data.addr, vdtype);
                        cc.ldp(vec1.q(), vec2.q(), mem);
                        step_map[it->id] = vec2;
                        return vec1;
                    }
                }

                aja64::Mem mem = vec_mem(data.addr, step->dtype.as_vec());
                if (vdtype.size_bytes() != 16) messed_up("Invalid load of dtype %s", show_vec_dtype(vdtype));
                VecR reg = create_vec_reg(vdtype);
                // No special handling for data.kind
                cc.ldr(reg.q(), mem);
                return reg;
            }
            SIMJIT_MATCH (StepKind::LoadSplat) {
                const ArgInfo &arg = args[data.addr.arg];
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    if (mdtype != MaskDataType::M8 && mdtype != MaskDataType::M16) {
                        messed_up("Invalid mask load of type %s", show_mask_dtype(mdtype));
                    }
                    VecR result = cc.new_vec128();
                    aja64::Mem mem = aja64::ptr(arg.gp);
                    GpR w = cc.new_gp32();
                    cc.ldrb(w, mem);
                    cc.and_(w, w, 1);
                    if (mdtype == MaskDataType::M8) {
                        cc.dup(result.b8(), w);
                        cc.cmeq(result.b8(), result.b8(), 0);
                        cc.mvn(result.b8(), result.b8());
                        return result.b8();
                    }
                    cc.dup(result.b16(), w);
                    cc.cmeq(result.b16(), result.b16(), 0);
                    cc.mvn(result.b16(), result.b16());
                    return result.b16();
                }
                VecDataType vdtype = step->dtype.as_vec();
                VecR reg = create_vec_reg(vdtype);
                cc.ld1r(reg, aja64::ptr(arg.gp));
                return reg;
            }

            SIMJIT_MATCH (StepKind::Store) {
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    if (mdtype != MaskDataType::M8 && mdtype != MaskDataType::M16) {
                        messed_up("Invalid mask store of type %s", show_mask_dtype(mdtype));
                    }

                    VecR arg = vec_subexpr(data.what);
                    size_t sz = vec_elem_size(arg);
                    if (sz != 1) { arg = shrink_mask(arg, sz, 1); }

                    aja64::Mem mem = mask_mem(data.addr, mdtype);
                    if (mdtype == MaskDataType::M8) {
                        VecR tmp = cc.new_vec128().b8();
                        cc.and_(tmp, vec_special_const(SpecialConstant::Bits).b8(), arg.b8());
                        cc.addv(tmp.b(), tmp.b8());
                        aja64::Gp gp = cc.new_gp32();
                        cc.umov(gp, tmp.b(0));
                        cc.strb(gp, mem);
                    } else if (mdtype == MaskDataType::M16) {
                        VecR tmp = cc.new_vec128().b16();
                        cc.and_(tmp, vec_special_const(SpecialConstant::Bits).b16(), arg.b16());
                        VecR tmp1 = cc.new_vec128();
                        cc.ext(tmp1.b16(), tmp, tmp, 8);
                        cc.zip1(tmp.b16(), tmp.b16(), tmp1.b16());
                        cc.addv(tmp.h(), tmp.h8());
                        cc.str(tmp.h(), mem);
                    } else {
                        SIMJIT_ASSERT(0);
                    }
                    return {};
                }
                VecDataType vdtype = step->dtype.as_vec();
                aja64::Mem mem = vec_mem(data.addr, vdtype);
                VecR reg = vec_subexpr(data.what);
                if (vdtype.size_bytes() != 16) messed_up("Invalid store of dtype %s", show_vec_dtype(vdtype));
                // No special handling for data.kind
                cc.str(reg.q(), mem);
                return {};
            }

            SIMJIT_MATCH (StepKind::CondStore) {
                // There is no direct cond store instruction like on x86. We rewrite it manually to
                // load+blend+store. The performance is actually surprisingly good.
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = vec_subexpr(data.arg);
                VecR cond = vec_subexpr(data.cond);

                VecR tmp = cc.new_vec128();
                cc.ldr(tmp.q(), vec_mem(data.addr, vdtype, false));
                cc.bit(tmp.b16(), arg.b16(), cond.b16());
                cc.str(tmp.q(), vec_mem(data.addr, vdtype));
                return {};
            }

            SIMJIT_MATCH (StepKind::ArithBinary) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR left = vec_subexpr(data.left);
                if (vdtype.is_float()) {
                    VecR result = suggested_dst != nullptr ? suggested_dst->take().as<VecR>() : create_vec_reg(vdtype);
                    VecR right = vec_subexpr(data.right);
                    switch (data.op) {
                    case ArithBinaryOp::Add: cc.fadd(result, left, right); break;
                    case ArithBinaryOp::Sub: cc.fsub(result, left, right); break;
                    case ArithBinaryOp::Mul: cc.fmul(result, left, right); break;
                    case ArithBinaryOp::Div: cc.fdiv(result, left, right); break;
                    case ArithBinaryOp::Min: cc.fminnm(result, left, right); break;
                    case ArithBinaryOp::Max: cc.fmaxnm(result, left, right); break;
                    case ArithBinaryOp::And: cc.and_(result.b16(), left.b16(), right.b16()); break;
                    case ArithBinaryOp::Or: cc.orr(result.b16(), left.b16(), right.b16()); break;
                    case ArithBinaryOp::Xor: cc.eor(result.b16(), left.b16(), right.b16()); break;
                    case ArithBinaryOp::AndNot: cc.bic(result.b16(), right.b16(), left.b16()); break;
                    default: messed_up("Unexpected ArithBinary %s in float context", show_arith_binary_op(data.op));
                    }
                    return result;
                }
                if (SIMJIT_A64_ASMJIT_CONST_OPS && data.right->is(StepKind::Const)) {
                    VecR result = create_vec_reg(vdtype);
                    auto imm = data.right->step_data<StepKind::Const>().as_unsigned();
                    bool is_shift = imm != 0 && imm < vdtype.element_size_bits();
                    if (is_shift &&
                        (data.op == ArithBinaryOp::ShiftRightArith || data.op == ArithBinaryOp::ShiftLeftLogical ||
                         data.op == ArithBinaryOp::ShiftRightLogical || data.op == ArithBinaryOp::RotateLeft ||
                         data.op == ArithBinaryOp::RotateRight)) {
                        switch (data.op) {
                        case ArithBinaryOp::ShiftRightArith: cc.sshr(result, left, imm); break;
                        case ArithBinaryOp::ShiftRightLogical: cc.ushr(result, left, imm); break;
                        case ArithBinaryOp::ShiftLeftLogical: cc.shl(result, left, imm); break;
                        case ArithBinaryOp::RotateLeft:
                            cc.shl(result, left, imm);
                            cc.usra(result, left, vdtype.element_size_bits() - imm);
                            break;
                        case ArithBinaryOp::RotateRight:
                            cc.shl(result, left, vdtype.element_size_bits() - imm);
                            cc.usra(result, left, imm);
                            break;
                        default: SIMJIT_UNREACHABLE();
                        }
                        return result;
                    }
                }
                VecR right = vec_subexpr(data.right);

                // Forbid suggested dst in i64 cases because they will otherwise overwrite accumulator.
                // Other cases either work (add, and etc.) or not commutative (rotates), meaning they can't have
                // suggested_dst.
                //
                // This is way less clear than similar stuff in x86 backend. Reason for that is we need to
                // handle possibility of operation not supporting direct write to accumulator either way. Right now
                // it works because the only place we use suggested_dst is binary op. DO NOT EXTEND THIS TO OTHER
                // STEPS! Complexity will grow unbounded. Other cases do not usually have redudant moves, and even
                // if they do (e.g. select), they are way less frequent than stuff with accumulators. On top of that,
                // the only reason for doing it with accumulators is to make code nicer to look at.
                VecR result =
                    suggested_dst != nullptr && !(vdtype.elem == VecElemType::I64 &&
                                                  (data.op == ArithBinaryOp::Min || data.op == ArithBinaryOp::Max ||
                                                   data.op == ArithBinaryOp::UMin || data.op == ArithBinaryOp::UMax))
                        ? suggested_dst->take().as<VecR>()
                        : create_vec_reg(vdtype);

                switch (data.op) {
                case ArithBinaryOp::Add: cc.add(result, left, right); break;
                case ArithBinaryOp::Sub: cc.sub(result, left, right); break;
                case ArithBinaryOp::Mul:
                    if (vdtype.elem == VecElemType::I64) { unsupported("Do not support i64 mul"); }
                    cc.mul(result, left, right);
                    break;
                case ArithBinaryOp::Mul64SE:
                    if (vdtype.elem == VecElemType::I64) {
                        VecR left32 = cc.new_vec128().s2();
                        VecR right32 = cc.new_vec128().s2();
                        cc.xtn(left32, left.d2());
                        cc.xtn(right32, right.d2());
                        cc.smull(result.d2(), left32, right32);
                        break;
                    }
                    cc.smull(result, left, right);
                    break;
                case ArithBinaryOp::Mul64ZE:
                    if (vdtype.elem == VecElemType::I64) {
                        VecR left32 = cc.new_vec128().s2();
                        VecR right32 = cc.new_vec128().s2();
                        cc.xtn(left32, left.d2());
                        cc.xtn(right32, right.d2());
                        cc.umull(result.d2(), left32, right32);
                        break;
                    }
                    cc.umull(result, left, right);
                    break;
                case ArithBinaryOp::Min:
                    if (vdtype.elem == VecElemType::I64) {
                        // There is no direct encoding but this case is easy enough to write by hand
                        cc.cmgt(result, left, right);
                        cc.bsl(result.b16(), right.b16(), left.b16());
                        break;
                    }
                    cc.smin(result, left, right);
                    break;
                case ArithBinaryOp::Max:
                    if (vdtype.elem == VecElemType::I64) {
                        cc.cmgt(result, left, right);
                        cc.bsl(result.b16(), left.b16(), right.b16());
                        break;
                    }
                    cc.smax(result, left, right);
                    break;
                case ArithBinaryOp::UMin:
                    if (vdtype.elem == VecElemType::I64) {
                        cc.cmhi(result, left, right);
                        cc.bsl(result.b16(), right.b16(), left.b16());
                        break;
                    }
                    cc.umin(result, left, right);
                    break;
                case ArithBinaryOp::UMax:
                    if (vdtype.elem == VecElemType::I64) {
                        cc.cmhi(result, left, right);
                        cc.bsl(result.b16(), left.b16(), right.b16());
                        break;
                    }
                    cc.umax(result, left, right);
                    break;
                case ArithBinaryOp::And: cc.and_(result.b16(), left.b16(), right.b16()); break;
                case ArithBinaryOp::Or: cc.orr(result.b16(), left.b16(), right.b16()); break;
                case ArithBinaryOp::Xor: cc.eor(result.b16(), left.b16(), right.b16()); break;
                case ArithBinaryOp::AndNot: cc.bic(result.b16(), right.b16(), left.b16()); break;
                case ArithBinaryOp::ShiftRightArith: {
                    cc.neg(result, right);
                    cc.sshl(result, left, result);
                    break;
                }
                case ArithBinaryOp::ShiftRightLogical: {
                    cc.neg(result, right);
                    cc.ushl(result, left, result);
                    break;
                }
                case ArithBinaryOp::ShiftLeftLogical: cc.ushl(result, left, right); break;
                case ArithBinaryOp::RotateLeft: {
                    if (vdtype.elem != VecElemType::I32 && vdtype.elem != VecElemType::I64)
                        unsupported("Do not support rotate left for type %s", show_vec_dtype(vdtype));
                    cc.ushl(result, left, right);
                    VecR tmp1 = create_vec_reg(vdtype);
                    auto con = vdtype.elem == VecElemType::I32 ? SpecialConstant::I32_32 : SpecialConstant::I64_64;
                    cc.sub(tmp1, right, vec_special_const(con));
                    cc.ushl(tmp1, left, tmp1);
                    cc.orr(result.b16(), result.b16(), tmp1.b16());
                    break;
                }
                case ArithBinaryOp::RotateRight: {
                    if (vdtype.elem != VecElemType::I32 && vdtype.elem != VecElemType::I64)
                        unsupported("Do not support rotate right for type %s", show_vec_dtype(vdtype));
                    VecR tmp1 = create_vec_reg(vdtype);
                    cc.neg(tmp1, right);
                    cc.ushl(result, left, tmp1);
                    auto con = vdtype.elem == VecElemType::I32 ? SpecialConstant::I32_32 : SpecialConstant::I64_64;
                    cc.sub(tmp1, vec_special_const(con), right);
                    cc.ushl(tmp1, left, tmp1);
                    cc.orr(result.b16(), result.b16(), tmp1.b16());
                    break;
                }
                case ArithBinaryOp::UDiv:
                case ArithBinaryOp::Div:
                case ArithBinaryOp::UMod:
                case ArithBinaryOp::Mod: unsupported("Do not support int div");
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = float_subexpr(data.arg);
                VecR result = create_vec_reg(vdtype);
                if (vdtype.is_float()) {
                    switch (data.op) {
                    case ArithUnaryOp::Lzcnt:
                    case ArithUnaryOp::Tzcnt:
                    case ArithUnaryOp::Popcount:
                        messed_up("Unexpected ArithUnary %s in float context", show_arith_unary_op(data.op));
                    case ArithUnaryOp::Abs: cc.fabs(result, arg); break;
                    case ArithUnaryOp::Not: cc.mvn(result.b16(), arg.b16()); break;
                    case ArithUnaryOp::Negate: cc.fneg(result, arg); break;
                    case ArithUnaryOp::RoundNearest: cc.frintn(result, arg); break;
                    case ArithUnaryOp::RoundDown: cc.frintm(result, arg); break;
                    case ArithUnaryOp::RoundUp: cc.frintp(result, arg); break;
                    case ArithUnaryOp::RoundTruncate: cc.frintz(result, arg); break;
                    case ArithUnaryOp::Rcp: {
                        // This provides 22-24 bits of precision, more than avx512 version
                        cc.frecpe(result, arg);
                        VecR tmp = create_vec_reg(vdtype);
                        // 1 NR
                        cc.frecps(tmp, arg, result);
                        cc.fmul(result, result, tmp);
                        // 2 NR
                        cc.frecps(tmp, arg, result);
                        cc.fmul(result, result, tmp);
                        if (vdtype.elem == VecElemType::F64) {
                            // 3 NR
                            cc.frecps(tmp, arg, result);
                            cc.fmul(result, result, tmp);
                        }
                        break;
                    }
                    case ArithUnaryOp::Sqrt: cc.fsqrt(result, arg); break;
                    case ArithUnaryOp::Rsqrt: {
                        // This also provides more precision than avx512
                        cc.frsqrte(result, arg);
                        VecR tmp = create_vec_reg(vdtype);
                        // 1 NR
                        cc.fmul(tmp, result, result);
                        cc.frsqrts(tmp, arg, tmp);
                        cc.fmul(result, result, tmp);
                        // 2 NR
                        cc.fmul(tmp, result, result);
                        cc.frsqrts(tmp, arg, tmp);
                        cc.fmul(result, result, tmp);
                        if (vdtype.elem == VecElemType::F64) {
                            // 3 NR
                            cc.fmul(tmp, result, result);
                            cc.frsqrts(tmp, arg, tmp);
                            cc.fmul(result, result, tmp);
                        }
                        break;
                    }
                    }
                    return result;
                }

                switch (data.op) {
                case ArithUnaryOp::Not: cc.mvn(result.b16(), arg.b16()); break;
                case ArithUnaryOp::Negate: cc.neg(result, arg); break;
                case ArithUnaryOp::Abs: cc.abs(result, arg); break;
                case ArithUnaryOp::Lzcnt:
                    if (vdtype.elem == VecElemType::I64) {
                        // NOTE: I wrote this myself, didn't look into LLVM output. Probably can be optimized.
                        VecR tmp = cc.new_vec128();
                        cc.clz(tmp.s4(), arg.s4()); // [a0l a0h a1l a1h]
                        VecR tmp1 = cc.new_vec128();
                        cc.ushr(tmp1.d2(), tmp.d2(), 32); // [a0h 0   a1h 0  ]
                        cc.cmeq(result.s4(), tmp1.s4(),
                                vec_special_const(SpecialConstant::I32_32)); // [a0h == 32, false, a1h == 32, false]
                        // if (a0h == 32) ? 32 + a0l : a0h
                        cc.add(tmp.s4(), tmp1.s4(), tmp.s4());
                        cc.bsl(result.b16(), tmp.b16(), tmp1.b16());
                        break;
                    }
                    cc.clz(result, arg);
                    break;
                case ArithUnaryOp::Tzcnt: messed_up("vector tzcnt should have been rewritten");
                case ArithUnaryOp::Popcount: {
                    cc.cnt(result.b16(), arg.b16());
                    switch (vdtype.elem) {
                    case VecElemType::I8: break;
                    case VecElemType::I16: cc.uaddlp(result.h8(), result.b16()); break;
                    case VecElemType::I32:
                        cc.uaddlp(result.h8(), result.b16());
                        cc.uaddlp(result.s4(), result.h8());
                        break;
                    case VecElemType::I64: {
                        VecR tmp = cc.new_vec128();
                        cc.movi(tmp.d2(), 0);
                        cc.udot(tmp.s4(), result.b16(), vec_special_const(SpecialConstant::I8_1));
                        cc.uaddlp(result.d2(), tmp.s4());
                        break;
                    }
                    default: SIMJIT_UNREACHABLE();
                    }
                    break;
                }
                case ArithUnaryOp::RoundNearest:
                case ArithUnaryOp::RoundDown:
                case ArithUnaryOp::RoundUp:
                case ArithUnaryOp::RoundTruncate:
                case ArithUnaryOp::Rcp:
                case ArithUnaryOp::Sqrt:
                case ArithUnaryOp::Rsqrt:
                    messed_up("Unexpected ArithUnary %s in int context", show_arith_unary_op(data.op));
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR result = create_vec_reg(vdtype);
                VecElemType from = data.arg->dtype.as_vec().elem;
                VecElemType to = step->dtype.as_vec().elem;
                VecR arg = vec_subexpr(data.arg);

                if (vec_elem_size_bytes(from) > vec_elem_size_bytes(to)) {
                    SIMJIT_ASSERT(data.kind == IntCastKind::Trunc);
                    cc.xtn(result, arg);
                    return result;
                }
                if (data.kind == IntCastKind::Sext) {
                    cc.sxtl(result, arg);
                } else {
                    cc.uxtl(result, arg);
                }
                return result;
            }

            SIMJIT_MATCH (StepKind::FloatCast) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR result = create_vec_reg(vdtype);
                VecElemType from = data.arg->dtype.as_vec().elem;
                VecElemType to = step->dtype.as_vec().elem;
                VecR arg = vec_subexpr(data.arg);
                if (vec_elem_is_float(from) && vec_elem_is_float(to)) {
                    cc.fcvt(result, arg);
                    return result;
                }
                if (vec_elem_is_float(to)) {
                    if (data.is_unsigned) {
                        cc.ucvtf(result, arg);
                    } else {
                        cc.scvtf(result, arg);
                    }
                    return result;
                }
                if (data.is_unsigned) {
                    cc.fcvtzu(result, arg);
                } else {
                    cc.fcvtzs(result, arg);
                }
                return result;
            }
            // Yes, we don't do anything here
            SIMJIT_MATCH (StepKind::BitCast) return subexpr(data);
            SIMJIT_MATCH (StepKind::Compare) {
                VecDataType arg_dtype = data.left->dtype.as_vec();
                VecR left = vec_subexpr(data.left);
                VecR result = create_vec_reg(arg_dtype);
                if (SIMJIT_A64_ASMJIT_CONST_OPS && step_is_zero(data.right) && !data.is_unsigned) {
                    if (arg_dtype.is_float()) {
                        switch (data.op) {
                        case CmpOp::Less: cc.fcmlt(result, left, 0); break;
                        case CmpOp::Greater: cc.fcmgt(result, left, 0); break;
                        case CmpOp::LessEqual: cc.fcmle(result, left, 0); break;
                        case CmpOp::GreaterEqual: cc.fcmge(result, left, 0); break;
                        case CmpOp::Equal: cc.fcmeq(result, left, 0); break;
                        case CmpOp::NotEqual:
                            cc.fcmeq(result, left, 0);
                            cc.mvn(result.b16(), result.b16());
                            break;
                        }
                        return result;
                    }
                    switch (data.op) {
                    case CmpOp::Less: cc.cmlt(result, left, 0); break;
                    case CmpOp::Greater: cc.cmgt(result, left, 0); break;
                    case CmpOp::LessEqual: cc.cmle(result, left, 0); break;
                    case CmpOp::GreaterEqual: cc.cmge(result, left, 0); break;
                    case CmpOp::Equal: cc.cmeq(result, left, 0); break;
                    case CmpOp::NotEqual: cc.cmtst(result, left, left); break;
                    }
                    return result;
                }

                VecR right = vec_subexpr(data.right);
                if (arg_dtype.is_float()) {
                    // left >= right <=> right <= left
                    switch (data.op) {
                    case CmpOp::Less: cc.fcmgt(result, right, left); break;
                    case CmpOp::Greater: cc.fcmgt(result, left, right); break;
                    case CmpOp::LessEqual: cc.fcmge(result, right, left); break;
                    case CmpOp::GreaterEqual: cc.fcmge(result, left, right); break;
                    case CmpOp::Equal: cc.fcmeq(result, left, right); break;
                    case CmpOp::NotEqual:
                        cc.fcmeq(result, left, right);
                        cc.mvn(result.b16(), result.b16());
                        break;
                    }
                    return result;
                }
                if (data.is_unsigned) {
                    switch (data.op) {
                    case CmpOp::Less: cc.cmhi(result, right, left); break;
                    case CmpOp::Greater: cc.cmhi(result, left, right); break;
                    case CmpOp::LessEqual: cc.cmhs(result, right, left); break;
                    case CmpOp::GreaterEqual: cc.cmhs(result, left, right); break;
                    case CmpOp::Equal: cc.cmeq(result, left, right); break;
                    case CmpOp::NotEqual:
                        cc.cmeq(result, left, right);
                        cc.mvn(result.b16(), result.b16());
                        break;
                    }
                    return result;
                }
                switch (data.op) {
                case CmpOp::Less: cc.cmgt(result, right, left); break;
                case CmpOp::Greater: cc.cmgt(result, left, right); break;
                case CmpOp::LessEqual: cc.cmge(result, right, left); break;
                case CmpOp::GreaterEqual: cc.cmge(result, left, right); break;
                case CmpOp::Equal: cc.cmeq(result, left, right); break;
                case CmpOp::NotEqual:
                    cc.cmeq(result, left, right);
                    cc.mvn(result.b16(), result.b16());
                    break;
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::VecReduce) {
                // I hate this opcode...  A lot of operations are very cumbersome. Normally we would not implement
                // them, but this step is not performance critical.
                VecDataType vdtype = data.arg->dtype.as_vec();
                VecR arg = vec_subexpr(data.arg);
                VecR dst_v = create_vec_reg(vdtype);
                switch (vdtype.elem) {
                case VecElemType::I8: dst_v = dst_v.b(); break;
                case VecElemType::I16: dst_v = dst_v.h(); break;
                case VecElemType::I32:
                case VecElemType::F32: dst_v = dst_v.s(); break;
                case VecElemType::I64:
                case VecElemType::F64: dst_v = dst_v.d(); break;
                }

#define manual_impl(_flt, _int, _is_bool)                                                                             \
    do {                                                                                                              \
        if (vdtype.elem == VecElemType::F32) {                                                                        \
            if constexpr (_is_bool) {                                                                                 \
                unsupported("Do not support reduce %s of %s", show_arith_binary_op(data.op), show_vec_dtype(vdtype)); \
            }                                                                                                         \
            if (vdtype.size == VecSize::X4) {                                                                         \
                cc.ext(dst_v.b16(), arg.b16(), arg.b16(), 8);                                                         \
                cc._flt(dst_v.s2(), arg.s2(), dst_v.s2());                                                            \
                cc._flt(dst_v, dst_v.s(), dst_v.s(1));                                                                \
            } else if (vdtype.size == VecSize::X2) {                                                                  \
                cc._flt(dst_v, arg.s(), arg.s(1));                                                                    \
            } else {                                                                                                  \
                SIMJIT_UNREACHABLE();                                                                                 \
            }                                                                                                         \
            return dst_v;                                                                                             \
        } else if (vdtype.elem == VecElemType::F64) {                                                                 \
            if constexpr (_is_bool) {                                                                                 \
                unsupported("Do not support reduce %s of %s", show_arith_binary_op(data.op), show_vec_dtype(vdtype)); \
            }                                                                                                         \
            cc._flt(dst_v, arg.d(), arg.d(1));                                                                        \
            return dst_v;                                                                                             \
        } else if (vdtype.elem == VecElemType::I32) {                                                                 \
            if (vdtype.size == VecSize::X4) {                                                                         \
                cc.ext(dst_v.b16(), arg.b16(), arg.b16(), 8);                                                         \
                if constexpr (_is_bool) {                                                                             \
                    cc._int(dst_v.b8(), arg.b8(), dst_v.b8());                                                        \
                } else {                                                                                              \
                    cc._int(dst_v.s2(), arg.s2(), dst_v.s2());                                                        \
                }                                                                                                     \
                GpR l = cc.new_gp32();                                                                                \
                GpR r = cc.new_gp32();                                                                                \
                cc.fmov(l, dst_v.s());                                                                                \
                cc.mov(r, dst_v.s(1));                                                                                \
                cc._int(l, l, r);                                                                                     \
                return l;                                                                                             \
            }                                                                                                         \
            if (vdtype.size == VecSize::X2) {                                                                         \
                GpR l = cc.new_gp32();                                                                                \
                GpR r = cc.new_gp32();                                                                                \
                cc.fmov(l, arg.s());                                                                                  \
                cc.mov(r, arg.s(1));                                                                                  \
                cc._int(l, l, r);                                                                                     \
                return l;                                                                                             \
            }                                                                                                         \
            SIMJIT_UNREACHABLE();                                                                                     \
        } else if (vdtype.elem == VecElemType::I64) {                                                                 \
            GpR l = cc.new_gp64();                                                                                    \
            GpR r = cc.new_gp64();                                                                                    \
            cc.fmov(l, arg.d());                                                                                      \
            cc.mov(r, arg.d(1));                                                                                      \
            cc._int(l, l, r);                                                                                         \
            return l;                                                                                                 \
        }                                                                                                             \
        unsupported("Do not support reduce %s of %s", show_arith_binary_op(data.op), show_vec_dtype(vdtype));         \
    } while (0)

                switch (data.op) {
                case ArithBinaryOp::Add:
                    if (vdtype.elem == VecElemType::F32) {
                        if (vdtype.size == VecSize::X4) {
                            cc.faddp(dst_v.s4(), arg, arg);
                            cc.faddp(dst_v, dst_v.s2());
                        } else if (vdtype.size == VecSize::X2) {
                            cc.faddp(dst_v, arg);
                        } else {
                            SIMJIT_UNREACHABLE();
                        }
                    } else if (vdtype.elem == VecElemType::F64) {
                        cc.faddp(dst_v, arg);
                    } else if (vdtype.elem == VecElemType::I64) {
                        cc.addp(dst_v.d2(), arg, arg);
                    } else {
                        cc.addv(dst_v, arg);
                    }
                    break;
                case ArithBinaryOp::Mul: manual_impl(fmul, mul, false); break;
                case ArithBinaryOp::Min:
                    if (vdtype.elem == VecElemType::I64) {
                        VecR tmp = cc.new_vec128();
                        cc.mov(tmp.d(), arg.d(1));
                        cc.cmgt(dst_v.d2(), arg.d2(), tmp.d2());
                        cc.bsl(dst_v.b8(), tmp.b8(), arg.b8());
                        break;
                    }
                    if (vdtype.elem == VecElemType::F32) {
                        cc.fminv(dst_v, arg);
                        break;
                    }
                    if (vdtype.elem == VecElemType::F64) {
                        cc.fminp(dst_v, arg);
                        break;
                    }
                    cc.sminv(dst_v, arg);
                    break;
                case ArithBinaryOp::Max:
                    if (vdtype.elem == VecElemType::I64) {
                        VecR tmp = cc.new_vec128();
                        cc.mov(tmp.d(), arg.d(1));
                        cc.cmgt(dst_v.d2(), arg.d2(), tmp.d2());
                        cc.bsl(dst_v.b8(), arg.b8(), tmp.b8());
                        break;
                    }
                    if (vdtype.elem == VecElemType::F32) {
                        cc.fmaxv(dst_v, arg);
                        break;
                    }
                    if (vdtype.elem == VecElemType::F64) {
                        cc.fmaxp(dst_v, arg);
                        break;
                    }
                    cc.smaxv(dst_v, arg);
                    break;
                case ArithBinaryOp::UMin:
                    if (vdtype.elem == VecElemType::I64) {
                        VecR tmp = cc.new_vec128();
                        cc.mov(tmp.d(), arg.d(1));
                        cc.cmhi(dst_v.d2(), arg.d2(), tmp.d2());
                        cc.bsl(dst_v.b8(), tmp.b8(), arg.b8());
                        break;
                    }
                    cc.uminv(dst_v, arg);
                    break;
                case ArithBinaryOp::UMax:
                    if (vdtype.elem == VecElemType::I64) {
                        VecR tmp = cc.new_vec128();
                        cc.mov(tmp.d(), arg.d(1));
                        cc.cmhi(dst_v.d2(), arg.d2(), tmp.d2());
                        cc.bsl(dst_v.b8(), arg.b8(), tmp.b8());
                        break;
                    }
                    cc.umaxv(dst_v, arg);
                    break;
                case ArithBinaryOp::And: manual_impl(and_, and_, true); break;
                case ArithBinaryOp::Or: manual_impl(orr, orr, true); break;
                case ArithBinaryOp::Xor: manual_impl(eor, eor, true); break;
                default: messed_up("Invalid reduce %s of %s", show_arith_binary_op(data.op), show_vec_dtype(vdtype));
                }
                if (vdtype.is_float()) { return dst_v; }
                GpR dst = create_int_reg(vec_elem_to_scalar(vdtype.elem));
                cc.fmov(dst, dst_v);
                return dst;
            }
            SIMJIT_MATCH (StepKind::MaskReduce) {
                MaskDataType mdtype = data.arg->dtype.as_mask();
                VecR arg = vec_subexpr(data.arg);
                VecR tmp_vec = cc.new_vec128();
                GpR gp = cc.new_gp32();
                if (vec_elem_size(arg) != 1) { arg = shrink_mask(arg, vec_elem_size(arg), 1); }
                switch (data.op) {
                case PredicateBinaryOp::And:
                    cc.addv(tmp_vec.b(), arg.b16());
                    cc.umov(gp, tmp_vec.b(0));
                    cc.cmp(gp, 0x100 - (int)mask_dtype_bits(mdtype));
                    cc.cset(gp, aja64::CondCode::kEQ);
                    break;
                case PredicateBinaryOp::Or:
                    cc.addv(tmp_vec.b(), arg.b16());
                    cc.umov(gp, tmp_vec.b(0));
                    cc.cmp(gp, 0);
                    cc.cset(gp, aja64::CondCode::kNE);
                    break;
                case PredicateBinaryOp::Xor:
                    cc.addv(tmp_vec.b(), arg.b16());
                    cc.umov(gp, tmp_vec.b(0));
                    cc.and_(gp, gp, 1);
                    break;
                case PredicateBinaryOp::AndNot:
                case PredicateBinaryOp::XNor:
                    messed_up("Invalid reduce %s of %s", show_predicate_binary_op(data.op), show_mask_dtype(mdtype));
                }
                return gp;
            }

            SIMJIT_MATCH (StepKind::AccLoad) { return accs[mir_func->accs.index(data)]; }
            SIMJIT_MATCH (StepKind::AccStore) {
                VecR acc_reg = accs[mir_func->accs.index(data.acc)].as<VecR>();
                SuggestedReg suggest{acc_reg};
                VecR arg = vec_subexpr(data.arg, &suggest);
                if (data.arg->dtype.is_mask()) {
                    // Unlike other cases where dynamically change either lhs or rhs, here we need to treat accumulator
                    // as canon, and coerce arg to its format.
                    size_t acc_sz = vec_elem_size(acc_reg);
                    size_t arg_sz = vec_elem_size(arg);
                    if (acc_sz > arg_sz) {
                        arg = expand_mask(arg, arg_sz, acc_sz);
                    } else if (acc_sz < arg_sz) {
                        arg = shrink_mask(arg, arg_sz, acc_sz);
                    }
                    // Currently we shouldn't have any suggested regs for masks, although nothing will break if we do.
                    SIMJIT_ASSERT(!suggest.acknowledge);
                    cc.mov(acc_reg, arg);
                    return {};
                }
                if (!suggest.acknowledge) { cc.mov(acc_reg, arg); }
                return {};
            }
            SIMJIT_MATCH2 (StepKind::VecWidenHighHalf, StepKind::VecFloatWidenHighHalf) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = vec_subexpr(data.arg);
                VecR dst = create_vec_reg(step->dtype.as_vec());
                if (vdtype.is_float()) {
                    cc.fcvtl2(dst, arg);
                } else if (data.is_unsigned) {
                    cc.uxtl2(dst, arg);
                } else {
                    cc.sxtl2(dst, arg);
                }
                return dst;
            }
            SIMJIT_MATCH2 (StepKind::VecWidenLowHalf, StepKind::VecFloatWidenLowHalf) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = vec_subexpr(data.arg);
                switch (vdtype.elem) {
                case VecElemType::I8:
                    messed_up("invalid vector low-half widen target element %s", show_vec_elem_type(vdtype.elem));
                case VecElemType::I16: arg = arg.b8(); break;
                case VecElemType::I32: arg = arg.h4(); break;
                case VecElemType::I64: arg = arg.s2(); break;
                case VecElemType::F32:
                    messed_up("invalid vector low-half float widen target element %s", show_vec_elem_type(vdtype.elem));
                case VecElemType::F64: arg = arg.s2(); break;
                }

                VecR dst = create_vec_reg(step->dtype.as_vec());
                if (vdtype.is_float()) {
                    cc.fcvtl(dst, arg);
                } else if (data.is_unsigned) {
                    cc.uxtl(dst, arg);
                } else {
                    cc.sxtl(dst, arg);
                }
                return dst;
            }
            SIMJIT_MATCH (StepKind::VecNarrowCombine) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR low = vec_subexpr(data.low);
                VecR high = vec_subexpr(data.high);
                return emit_vec_narrow_combine(vdtype, low, high);
            }
            SIMJIT_MATCH (StepKind::VecFloatNarrowCombine) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR low = vec_subexpr(data.low);
                VecR high = vec_subexpr(data.high);
                return emit_vec_float_narrow_combine(vdtype, low, high);
            }
            SIMJIT_MATCH (StepKind::MaskBinary) {
                VecR left = vec_subexpr(data.left);
                VecR right = vec_subexpr(data.right);
                VecR result = cc.new_vec128();

                auto adjusted = masks_to_same_size(left, right);
                left = adjusted.first;
                right = adjusted.second;
                SIMJIT_ASSERT(vec_elem_size(left) == vec_elem_size(right));

                switch (data.op) {
                case PredicateBinaryOp::And: cc.and_(result.b16(), left.b16(), right.b16()); break;
                case PredicateBinaryOp::Or: cc.orr(result.b16(), left.b16(), right.b16()); break;
                case PredicateBinaryOp::Xor: cc.eor(result.b16(), left.b16(), right.b16()); break;
                case PredicateBinaryOp::AndNot: cc.bic(result.b16(), right.b16(), left.b16()); break;
                case PredicateBinaryOp::XNor:
                    cc.eor(result.b16(), left.b16(), right.b16());
                    cc.mvn(result.b16(), result.b16());
                    break;
                }

                MaskDataType mdtype = step->dtype.as_mask();
                size_t sz = vec_elem_size(left);
                switch (mdtype) {
                case MaskDataType::M2:
                    SIMJIT_ASSERT(sz == 8 || sz == 4 || sz == 2 || sz == 1);
                    if (sz == 8) { return result.d2(); }
                    if (sz == 4) { return result.s2(); }
                    if (sz == 2) { return result.h2(); }
                    return result.b8();
                case MaskDataType::M4:
                    SIMJIT_ASSERT(sz == 4 || sz == 2 || sz == 1);
                    if (sz == 4) { return result.s4(); }
                    if (sz == 2) { return result.h4(); }
                    return result.b8();
                case MaskDataType::M8:
                    SIMJIT_ASSERT(sz == 2 || sz == 1);
                    if (sz == 2) { return result.h8(); }
                    return result.b8();
                case MaskDataType::M16: SIMJIT_ASSERT(sz == 1); return result.b16();
                case MaskDataType::M32:
                case MaskDataType::M64: SIMJIT_UNREACHABLE();
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::MaskCount) {
                VecR arg = vec_subexpr(data);
                GpR result = create_int_reg(step->dtype.as_scalar());
                VecR tmp = cc.new_vec128();
                MaskDataType mdtype = data->dtype.as_mask();
                switch (mdtype) {
                case MaskDataType::M2:
                    cc.addp(tmp.d2(), arg.d2(), arg.d2());
                    cc.neg(tmp.d2(), tmp.d2());
                    break;
                case MaskDataType::M4:
                    cc.addv(tmp.s(), arg.s4());
                    cc.neg(tmp.s4(), tmp.s4());
                    break;
                case MaskDataType::M8:
                    cc.addv(tmp.h(), arg.h8());
                    cc.neg(tmp.h8(), tmp.h8());
                    break;
                case MaskDataType::M16:
                    cc.addv(tmp.b(), arg.b16());
                    cc.neg(tmp.b16(), tmp.b16());
                    break;
                case MaskDataType::M32:
                case MaskDataType::M64: SIMJIT_UNREACHABLE();
                }
                cc.umov(result.w(), tmp.b(0));
                return result;
            }
            SIMJIT_MATCH (StepKind::MaskCombine) {
                MaskDataType mdtype = step->dtype.as_mask();
                VecR left = vec_subexpr(data.left);
                VecR right = vec_subexpr(data.right);
                VecR result = cc.new_vec128();

                auto adjusted = masks_to_same_size(left, right);
                left = adjusted.first;
                right = adjusted.second;

                size_t sz = vec_elem_size(left);
                if (sz == 1) {
                    switch (mdtype) {
                    case MaskDataType::M4:
                        cc.movi(result, 0);
                        cc.mov(result.h(), left.h());
                        cc.mov(result.h(1), right.h());
                        return result.b8();
                    case MaskDataType::M8:
                        cc.movi(result, 0);
                        cc.mov(result.s(), left.s());
                        cc.mov(result.s(1), right.s());
                        return result.b8();
                    case MaskDataType::M16:
                        cc.movi(result, 0);
                        cc.mov(result.d(), left.d());
                        cc.mov(result.d(1), right.d());
                        return result.b16();
                    case MaskDataType::M2:
                    case MaskDataType::M32:
                    case MaskDataType::M64: SIMJIT_UNREACHABLE();
                    }
                } else if (sz == 2) {
                    switch (mdtype) {
                    case MaskDataType::M4:
                        cc.movi(result, 0);
                        cc.mov(result.s(), left.s());
                        cc.mov(result.s(1), right.s());
                        return result.b8();
                    case MaskDataType::M8:
                        cc.movi(result, 0);
                        cc.mov(result.d(), left.d());
                        cc.mov(result.d(1), right.d());
                        return result.b16();
                    case MaskDataType::M16:
                        cc.xtn(result.b8(), left.h8());
                        cc.xtn2(result.b16(), right.h8());
                        return result.b16();
                    case MaskDataType::M2:
                    case MaskDataType::M32:
                    case MaskDataType::M64: SIMJIT_UNREACHABLE();
                    }
                } else if (sz == 4) {
                    switch (mdtype) {
                    case MaskDataType::M4:
                        cc.movi(result, 0);
                        cc.mov(result.d(), left.d());
                        cc.mov(result.d(1), right.d());
                        return result.b16();
                    case MaskDataType::M8:
                        cc.xtn(result.h4(), left.s4());
                        cc.xtn2(result.h8(), right.s4());
                        return result.h8();
                    case MaskDataType::M2:
                    case MaskDataType::M16:
                    case MaskDataType::M32:
                    case MaskDataType::M64: SIMJIT_UNREACHABLE();
                    }
                } else if (sz == 8) {
                    switch (mdtype) {
                    case MaskDataType::M4:
                        cc.xtn(result.s2(), left.d2());
                        cc.xtn2(result.s4(), right.d2());
                        return result.s4();
                    case MaskDataType::M2:
                    case MaskDataType::M8:
                    case MaskDataType::M16:
                    case MaskDataType::M32:
                    case MaskDataType::M64: SIMJIT_UNREACHABLE();
                    }
                } else {
                    SIMJIT_UNREACHABLE();
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::PredicateNot) {
                MaskDataType mdtype = step->dtype.as_mask();
                VecR arg = vec_subexpr(data);
                size_t sz = vec_elem_size(arg);
                if (mdtype == MaskDataType::M8 && sz == 1) {
                    // XXX: This can happen after load... I am not sure if it needs proper fix since this whole dynamic
                    // mask size idea is not very nice..
                    sz = 2;
                    arg = expand_mask(arg, 1, 2);
                }
                VecR result = cc.new_similar_reg(arg);
                cc.mvn(result.b16(), arg.b16());
                switch (sz) {
                case 1: return result.b16();
                case 2: return result.h8();
                case 4: return result.s4();
                case 8: return result.d2();
                default: SIMJIT_UNREACHABLE();
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::Select) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR cond = vec_subexpr(data.cond);
                if (SIMJIT_A64_ASMJIT_BLEND_SUB && vec_elem_size(cond) == vdtype.element_size_bytes()) {
                    if (data.truthy->is(StepKind::ArithBinary)) {
                        auto &bin_data = data.truthy->step_data<StepKind::ArithBinary>();
                        if (bin_data.op == ArithBinaryOp::Add && bin_data.right->is(StepKind::Const) &&
                            bin_data.right->step_data<StepKind::Const>().as_unsigned() == 1 &&
                            bin_data.left == data.falsy) {
                            VecR falsy = vec_subexpr(data.falsy);
                            VecR result = create_vec_reg(vdtype);
                            cc.sub(result, falsy, cond);
                            return result;
                        }
                    }
                }
                VecR truthy = vec_subexpr(data.truthy);
                if (SIMJIT_A64_ASMJIT_ZEROBLEND && step_is_zero(data.falsy)) {
                    size_t cond_sz = vec_elem_size(cond);
                    VecR result{};
                    if (cond_sz != vdtype.element_size_bytes()) {
                        if (cond_sz < vdtype.element_size_bytes()) {
                            result = expand_mask(cond, cond_sz, vdtype.element_size_bytes());
                        } else {
                            result = shrink_mask(cond, cond_sz, vdtype.element_size_bytes());
                        }
                    } else {
                        result = create_vec_reg(vdtype);
                    }
                    cc.and_(result.b16(), cond.b16(), truthy.b16());
                    switch (vdtype.element_size_bytes()) {
                    case 1: return result.b16();
                    case 2: return result.h8();
                    case 4: return result.s4();
                    case 8: return result.d2();
                    default: SIMJIT_UNREACHABLE();
                    }
                    SIMJIT_UNREACHABLE();
                }

                VecR falsy = vec_subexpr(data.falsy);

                VecR result{};
                size_t cond_sz = vec_elem_size(cond);
                if (cond_sz != vdtype.element_size_bytes()) {
                    if (cond_sz < vdtype.element_size_bytes()) {
                        result = expand_mask(cond, cond_sz, vdtype.element_size_bytes());
                    } else {
                        result = shrink_mask(cond, cond_sz, vdtype.element_size_bytes());
                    }
                } else {
                    result = create_vec_reg(vdtype);
                    cc.mov(result.q(), cond.q());
                }

                cc.bsl(result.b16(), truthy.b16(), falsy.b16());
                switch (vdtype.element_size_bytes()) {
                case 1: return result.b16();
                case 2: return result.h8();
                case 4: return result.s4();
                case 8: return result.d2();
                default: SIMJIT_UNREACHABLE();
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::VecIndex) {
                VecDataType dtype = step->dtype.as_vec();
                VecR acc = accs[mir_func->accs.index(data.acc)].as<VecR>();
                VecR inc = vec_subexpr(data.inc);
                VecR result = create_vec_reg(dtype);
                cc.mov(result, acc);
                cc.add(acc, acc, inc);
                return result;
            }

            SIMJIT_MATCH (StepKind::FMA) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR a = vec_subexpr(data.x1);
                VecR b = vec_subexpr(data.x2);
                VecR c = vec_subexpr(data.x3);
                VecR result = suggested_dst != nullptr ? suggested_dst->take().as<VecR>() : create_vec_reg(vdtype);

                if (vdtype.is_float()) {
                    switch (data.kind) {
                    case FmaKind::FMA:
                        if (!(data.x3->is(StepKind::AccLoad) && suggested_dst != nullptr && suggested_dst->acknowledge))
                            cc.mov(result, c);
                        cc.fmla(result, a, b);
                        break;
                    case FmaKind::FMS:
                        cc.fneg(result, c);
                        cc.fmla(result, a, b);
                        break;
                    case FmaKind::FNMA:
                        if (!(data.x3->is(StepKind::AccLoad) && suggested_dst != nullptr && suggested_dst->acknowledge))
                            cc.mov(result, c);
                        cc.fmls(result, a, b);
                        break;
                    case FmaKind::FNMS:
                        cc.fneg(result, c);
                        cc.fmls(result, a, b);
                        break;
                    }
                } else {
                    if (vdtype.elem == VecElemType::I64) { unsupported("Do not support i64 mul"); }
                    switch (data.kind) {
                    case FmaKind::FMA:
                        if (!(data.x3->is(StepKind::AccLoad) && suggested_dst != nullptr && suggested_dst->acknowledge))
                            cc.mov(result, c);
                        cc.mla(result, a, b);
                        break;
                    case FmaKind::FMS:
                        cc.neg(result, c);
                        cc.mla(result, a, b);
                        break;
                    case FmaKind::FNMA:
                        if (!(data.x3->is(StepKind::AccLoad) && suggested_dst != nullptr && suggested_dst->acknowledge))
                            cc.mov(result, c);
                        cc.mls(result, a, b);
                        break;
                    case FmaKind::FNMS:
                        cc.neg(result, c);
                        cc.mls(result, a, b);
                        break;
                    }
                }
                return result;
            }
            SIMJIT_MATCH (StepKind::Fpclass) {
                VecDataType vdtype = data.arg->dtype.as_vec();
                bool is_f32 = vdtype.elem == VecElemType::F32;
                VecR arg = vec_subexpr(data.arg);
                std::optional<VecR> acc;
                auto accumulate = [&](const VecR &x) {
                    if (acc) {
                        cc.orr((*acc).b16(), (*acc).b16(), x.b16());
                    } else {
                        acc = x;
                    }
                };

                if (bool(data.flags & FpClass::FPC_INFINITE)) {
                    if (is_f32) {
                        VecR tmp = create_vec_reg(vdtype);
                        cc.fabs(tmp, arg);
                        cc.cmeq(tmp, tmp, vec_special_const(SpecialConstant::F32_Inf));
                        accumulate(tmp);
                    } else {
                        VecR tmp = create_vec_reg(vdtype);
                        cc.fabs(tmp, arg);
                        cc.cmeq(tmp, tmp, vec_special_const(SpecialConstant::F64_Inf));
                        accumulate(tmp);
                    }
                }
                if (bool(data.flags & FpClass::FPC_NAN)) {
                    VecR tmp = create_vec_reg(vdtype);
                    cc.fcmeq(tmp, arg, arg);
                    cc.mvn(tmp.b16(), tmp.b16());
                    accumulate(tmp);
                }

                if (bool(data.flags & FpClass::FPC_SUBNORMAL)) {
                    if (is_f32) {
                        VecR tmp = create_vec_reg(vdtype);
                        cc.and_(tmp.b16(), vec_special_const(SpecialConstant::F32_Inf).b16(), arg.b16());
                        cc.cmeq(tmp, tmp, 0);

                        VecR mant = create_vec_reg(vdtype);
                        cc.and_(mant.b16(), vec_special_const(SpecialConstant::F32_Mant).b16(), arg.b16());
                        cc.cmtst(mant, mant, mant);

                        cc.and_(tmp.b16(), tmp.b16(), mant.b16());
                        accumulate(tmp);
                    } else {
                        VecR tmp = create_vec_reg(vdtype);
                        cc.and_(tmp.b16(), vec_special_const(SpecialConstant::F64_Inf).b16(), arg.b16());
                        cc.cmeq(tmp, tmp, 0);

                        VecR mant = create_vec_reg(vdtype);
                        cc.and_(mant.b16(), vec_special_const(SpecialConstant::F64_Mant).b16(), arg.b16());
                        cc.cmtst(mant, mant, mant);

                        cc.and_(tmp.b16(), tmp.b16(), mant.b16());
                        accumulate(tmp);
                    }
                }
                if (bool(data.flags & FpClass::FPC_ZERO)) {
                    VecR tmp = create_vec_reg(vdtype);
                    cc.fabs(tmp, arg);
                    cc.fcmeq(tmp, tmp, 0);
                    accumulate(tmp);
                }
                SIMJIT_ASSERT(acc);
                return *acc;
            }

            SIMJIT_MATCH (StepKind::VecPermute) {
                VecDataType vdtype = step->dtype.as_vec();
                VecR arg = vec_subexpr(data.arg);
                VecR result = create_vec_reg(vdtype);

                if (data.is_bit) {
                    if (data.permute == REVERSE_BITS) {
                        cc.rbit(result.b16(), arg.b16());
                        return result;
                    }
                } else {
                    if (vdtype.elem == VecElemType::I16 && data.permute == REVERSE_BYTES_I16) {
                        cc.rev16(result.b16(), arg.b16());
                        return result;
                    }
                    if (vdtype.elem == VecElemType::I32 && data.permute == REVERSE_BYTES_I32) {
                        cc.rev32(result.b16(), arg.b16());
                        return result;
                    }
                    if (vdtype.elem == VecElemType::I64 && data.permute == REVERSE_BYTES_I64) {
                        cc.rev64(result.b16(), arg.b16());
                        return result;
                    }
                    VecR idxs = vec_subexpr(data.permute_idxs);
                    cc.tbl(result.b16(), arg.b16(), idxs.b16());
                    return result;
                }
                unsupported("Do not support arbitrary bit permutes");
            }
        }
        SIMJIT_UNREACHABLE();
    }

    void prepare_ldp(nonstd::span<Step *const> steps) {
        if (!SIMJIT_A64_ASMJIT_LDP) return;

        std::vector<std::vector<Step *>> loads;
        loads.resize(this->args.size());
        traverse_steps_postorder_unique(step_map.size(), steps, [&](Step *s) {
            if (s->is(StepKind::Load) && s->dtype.is_vec()) {
                auto &data = s->step_data<StepKind::Load>();
                loads[data.addr.arg].push_back(s);
            }
        });

        for (ArgumentIdx idx = 0; idx < loads.size(); ++idx) {
            const std::vector<Step *> &arg_loads = loads[idx];
            // SIMJIT_ASSERT(arg_loads.empty() || has_single_bit(arg_loads.size()));
            if (arg_loads.size() >= 2) {
                size_t offset = 0;
                for (;;) {
                    Step *first = nullptr;
                    for (Step *s : arg_loads) {
                        auto &data = s->step_data<StepKind::Load>();
                        if (data.addr.offset == offset) {
                            first = s;
                            break;
                        }
                    }
                    if (first == nullptr) { break; }
                    offset += first->dtype.as_vec().nelems();
                    Step *second = nullptr;
                    for (Step *s : arg_loads) {
                        auto &data = s->step_data<StepKind::Load>();
                        if (data.addr.offset == offset) {
                            second = s;
                            break;
                        }
                    }
                    if (second == nullptr) { break; }
                    offset += second->dtype.as_vec().nelems();
                    SIMJIT_ASSERT(first->dtype.as_vec() == second->dtype.as_vec());

                    // Record information about paired load to do it lazilly.
                    // If we do all loads here, they will take up all the registers.
                    paired_ops[first->id] = second;
                }
            }
        }
    }

    void disable_index_regs_for_loop_overreads(nonstd::span<Step *const> steps, size_t loop_width) {
        if (!SIMJIT_A64_ASMJIT_INDEX_REGS) { return; }

        traverse_steps_postorder_unique(step_map.size(), steps, [&](Step *step) {
            std::optional<ArgumentAddress> addr{};
            if (step->is(StepKind::Load)) {
                addr = step->step_data<StepKind::Load>().addr;
            } else if (step->is(StepKind::Store)) {
                addr = step->step_data<StepKind::Store>().addr;
            } else if (step->is(StepKind::CondStore)) {
                addr = step->step_data<StepKind::CondStore>().addr;
            }
            if (!addr.has_value()) { return; }
            if (addr->offset + step_row_width(step) <= loop_width) { return; }

            // Some normalized vector expressions read overlapping windows that extend past the rows produced by
            // one main-loop iteration. Post-index addressing would advance the argument pointer by those read-ahead
            // loads, while the loop counter advances only by loop_width.
            index_regs[addr->arg] = {};
        });
    }

    bool is_second_paired_load(Step *step) const {
        if (!SIMJIT_A64_ASMJIT_LDP || !step->is(StepKind::Load) || !step->dtype.is_vec()) { return false; }
        for (const Step *paired : paired_ops) {
            if (paired == step) { return true; }
        }
        return false;
    }

    std::optional<std::pair<ArgumentAddress, size_t>> post_index_access(Step *step) const {
        if (is_second_paired_load(step)) { return {}; }

        if (step->is(StepKind::Load)) {
            auto data = step->step_data<StepKind::Load>();
            if (step->dtype.is_vec()) {
                size_t width = step->dtype.as_vec().nelems();
                if (SIMJIT_A64_ASMJIT_LDP && paired_ops[step->id] != nullptr) { width *= 2; }
                return std::pair{data.addr, width};
            }
            if (step->dtype.is_scalar() && step->dtype != ScalarDataType::I1) {
                return std::pair{data.addr, size_t{1}};
            }
            if (step->dtype.is_mask() && step->dtype != MaskDataType::M2 && step->dtype != MaskDataType::M4) {
                return std::pair{data.addr, mask_dtype_bits(step->dtype.as_mask())};
            }
            return {};
        }
        if (step->is(StepKind::Store)) {
            auto data = step->step_data<StepKind::Store>();
            return std::pair{data.addr, step_row_width(step)};
        }
        if (step->is(StepKind::CondStore)) {
            auto data = step->step_data<StepKind::CondStore>();
            return std::pair{data.addr, step_row_width(step)};
        }
        return {};
    }

    void disable_index_regs_for_forward_gaps(nonstd::span<Step *const> roots) {
        if (!SIMJIT_A64_ASMJIT_INDEX_REGS) { return; }

        std::vector<size_t> cursors(index_regs.size(), 0);
        for (Step *root : roots) {
            traverse_steps_postorder(root, [&](Step *step) {
                auto access = post_index_access(step);
                if (!access.has_value()) { return; }
                ArgumentAddress addr = access->first;
                if (!index_regs[addr.arg].gp.is_valid()) { return; }
                if (addr.offset > cursors[addr.arg]) {
                    // Post-index registers only model a monotonic cursor. A forward gap means a later root may use
                    // explicit addressing for the gap, leaving the cursor behind for the next loop iteration.
                    index_regs[addr.arg] = {};
                    return;
                }
                if (addr.offset == cursors[addr.arg]) { cursors[addr.arg] += access->second; }
            });
        }
    }

    bool is_mask_shift_writer_root(const Step *root) const {
        if (root == nullptr || !root->is(StepKind::Store)) { return false; }
        if (!is_vector_mask_shift_writer_store(root) && !is_scalar_mask_shift_writer_store(root)) { return false; }
        auto data = root->step_data<StepKind::Store>();
        return mask_shift_writers[data.addr.arg].active;
    }

    void plan_mask_shift_writers(const mir::Function *func) {
        mask_shift_writers.assign(func->args.size(), {});
        std::vector<MSSCandidate> candidates(func->args.size());

        for (Step *root : func->main_loop_roots) {
            if (!is_vector_mask_shift_writer_store(root)) { continue; }
            auto data = root->step_data<StepKind::Store>();
            MSSCandidate &candidate = candidates[data.addr.arg];
            if (candidate.state == MSSCandidateState::Rejected) { continue; }

            unsigned bit_count = mask_shift_writer_bit_count(root);
            if (data.addr.offset != candidate.vector_cursor) {
                candidate.state = MSSCandidateState::Rejected;
                continue;
            }
            candidate.vector_cursor += bit_count;
            if (candidate.state == MSSCandidateState::Empty) {
                candidate.state = MSSCandidateState::VectorOnly;
            } else if (candidate.state == MSSCandidateState::ScalarOnly) {
                candidate.state = MSSCandidateState::VectorAndScalar;
            }
        }
        for (Step *root : func->remainder_roots) {
            if (!is_scalar_mask_shift_writer_store(root)) { continue; }
            auto data = root->step_data<StepKind::Store>();
            MSSCandidate &candidate = candidates[data.addr.arg];
            if (candidate.state == MSSCandidateState::Rejected) { continue; }

            if (data.addr.offset != candidate.scalar_cursor) {
                candidate.state = MSSCandidateState::Rejected;
                continue;
            }
            candidate.scalar_cursor += mask_shift_writer_bit_count(root);
            if (candidate.state == MSSCandidateState::Empty) {
                candidate.state = MSSCandidateState::ScalarOnly;
            } else if (candidate.state == MSSCandidateState::VectorOnly) {
                candidate.state = MSSCandidateState::VectorAndScalar;
            }
        }

        for (const Step *root : func->main_loop_roots) {
            if (!root->is(StepKind::Store)) { continue; }
            auto data = root->step_data<StepKind::Store>();
            if (!is_vector_mask_shift_writer_store(root)) {
                candidates[data.addr.arg].state = MSSCandidateState::Rejected;
            }
        }
        for (const Step *root : func->remainder_roots) {
            if (!root->is(StepKind::Store)) { continue; }
            auto data = root->step_data<StepKind::Store>();
            if (!is_scalar_mask_shift_writer_store(root)) {
                candidates[data.addr.arg].state = MSSCandidateState::Rejected;
            }
        }

        for (size_t i = 0; i < candidates.size(); ++i) {
            MSSCandidate candidate = candidates[i];
            // Rejected includes partially managed outputs. A common valid shape is an m8/m16 main-loop mask store
            // followed by scalar i1 tail stores for the same argument. The shift writer must not handle only
            // the tail, because its destination pointer would not be advanced by the normal main-loop stores.
            if (candidate.state != MSSCandidateState::ScalarOnly &&
                candidate.state != MSSCandidateState::VectorAndScalar) {
                continue;
            }
            if (candidate.state == MSSCandidateState::VectorAndScalar && candidate.vector_cursor != func->loop_width) {
                continue;
            }

            MSSWriterState &writer = mask_shift_writers[i];
            writer.active = true;
            writer.dst_ptr = cc.new_gp64();
            writer.acc = cc.new_gp32();
        }
    }

    void init_mask_shift_writers() {
        for (size_t i = 0; i < mask_shift_writers.size(); ++i) {
            MSSWriterState &writer = mask_shift_writers[i];
            if (!writer.active) { continue; }
            cc.mov(writer.dst_ptr, args[i].gp);
            cc.mov(writer.acc, 0);
        }
    }

    void emit_mask_shift_append(Step *root) {
        SIMJIT_ASSERT(is_mask_shift_writer_root(root));
        auto data = root->step_data<StepKind::Store>();
        MSSWriterState &writer = mask_shift_writers[data.addr.arg];
        SIMJIT_ASSERT(writer.active);

        unsigned bit_count = mask_shift_writer_bit_count(root);
        GpR bits{};
        if (root->dtype == ScalarDataType::I1) {
            GpR what = int_subexpr(data.what);
            bits = cc.new_gp32();
            cc.and_(bits, what.w(), 1);
        } else {
            VecR arg = vec_subexpr(data.what);
            if (root->dtype == MaskDataType::M2) {
                VecR selected = cc.new_vec128().d2();
                cc.and_(selected.b16(), arg.b16(), vec_special_const(SpecialConstant::I64_MaskBits).b16());
                cc.addp(selected.d2(), selected.d2(), selected.d2());
                bits = cc.new_gp64();
                cc.umov(bits, selected.d(0));
                bits = bits.w();
            } else if (root->dtype == MaskDataType::M4) {
                VecR selected = cc.new_vec128().s4();
                cc.and_(selected.b16(), arg.b16(), vec_special_const(SpecialConstant::I32_MaskBits).b16());
                cc.addv(selected.s(), selected.s4());
                bits = cc.new_gp32();
                cc.umov(bits, selected.s(0));
            } else {
                SIMJIT_UNREACHABLE();
            }
        }

        GpR shift = cc.new_gp32();
        if (data.addr.offset == 0) {
            cc.and_(shift, counter.w(), MASK_SHIFT_WRITER_FLUSH_BITS - 1);
        } else {
            GpR pos = cc.new_gp64();
            cc.add(pos, counter, data.addr.offset);
            cc.and_(shift, pos.w(), MASK_SHIFT_WRITER_FLUSH_BITS - 1);
        }
        GpR shifted_bits = cc.new_gp32();
        cc.lsl(shifted_bits, bits, shift);
        cc.orr(writer.acc, writer.acc, shifted_bits);

        GpR flush_pos = cc.new_gp64();
        cc.add(flush_pos, counter, data.addr.offset + bit_count);
        cc.tst(flush_pos, MASK_SHIFT_WRITER_FLUSH_BITS - 1);
        aj::Label no_flush = cc.new_label();
        cc.b(aja64::CondCode::kNE, no_flush);
        SIMJIT_ASSERT(is_valid_mask_shift_writer_flush_dtype(MASK_SHIFT_WRITER_FLUSH_DTYPE));
        store(cc, writer.acc, aja64::ptr_post(writer.dst_ptr, MASK_SHIFT_WRITER_FLUSH_BYTES),
              MASK_SHIFT_WRITER_FLUSH_DTYPE);
        cc.mov(writer.acc, 0);
        cc.bind(no_flush);
    }

    bool try_emit_store_pair(Step *left, Step *right) {
        bool can_pair_stores = SIMJIT_A64_ASMJIT_STP && left != nullptr && right != nullptr &&
                               left->is(StepKind::Store) && right->is(StepKind::Store) && left->dtype.is_vec() &&
                               right->dtype.is_vec() && left->dtype == right->dtype;
        if (!can_pair_stores) { return false; }

        auto left_data = left->step_data<StepKind::Store>();
        auto right_data = right->step_data<StepKind::Store>();
        VecDataType vdtype = left->dtype.as_vec();
        can_pair_stores = vdtype.size_bytes() == 16 && left_data.kind == LoadStoreKind::Aligned &&
                          right_data.kind == LoadStoreKind::Aligned && left_data.addr.arg == right_data.addr.arg &&
                          right_data.addr.offset == left_data.addr.offset + vdtype.nelems();
        if (!can_pair_stores) { return false; }

        VecR left_arg = vec_subexpr(left_data.what);
        VecR right_arg = vec_subexpr(right_data.what);
        cc.stp(left_arg.q(), right_arg.q(), vec_pair_mem(left_data.addr, vdtype));
        return true;
    }

    void emit_mask_shift_writer_epilogue() {
        for (MSSWriterState &writer : mask_shift_writers) {
            if (!writer.active) { continue; }
            SIMJIT_ASSERT(is_valid_mask_shift_writer_flush_dtype(MASK_SHIFT_WRITER_FLUSH_DTYPE));

            GpR pending = cc.new_gp32();
            cc.and_(pending, row_count.w(), MASK_SHIFT_WRITER_FLUSH_BITS - 1);
            aj::Label done = cc.new_label();
            cc.cbz(pending, done);

            if constexpr (MASK_SHIFT_WRITER_FLUSH_DTYPE == ScalarDataType::I16) {
                aj::Label partial_byte = cc.new_label();
                cc.cmp(pending, 8);
                cc.b(aja64::CondCode::kLO, partial_byte);
                cc.strb(writer.acc, aja64::ptr_post(writer.dst_ptr, 1));
                cc.lsr(writer.acc, writer.acc, 8);
                cc.sub(pending, pending, 8);
                cc.cbz(pending, done);
                cc.bind(partial_byte);
            }

            GpR mask = cc.new_gp32();
            cc.mov(mask, 1);
            cc.lsl(mask, mask, pending);
            cc.sub(mask, mask, 1);
            GpR old = cc.new_gp32();
            cc.ldrb(old, aja64::ptr(writer.dst_ptr));
            cc.bic(old, old, mask);
            cc.and_(writer.acc, writer.acc, mask);
            cc.orr(old, old, writer.acc);
            cc.strb(old, aja64::ptr(writer.dst_ptr));
            cc.bind(done);
        }
    }

    void compile_steps(nonstd::span<Step *const> steps) {
        clear_shifted_counters();
        for (size_t i = 0; i < steps.size(); ++i) {
            Step *left = steps[i];
            if (is_mask_shift_writer_root(left)) {
                emit_mask_shift_append(left);
                continue;
            }

            Step *right = i + 1 < steps.size() ? steps[i + 1] : nullptr;
            if (try_emit_store_pair(left, right)) {
                ++i;
                continue;
            }

            subexpr(left);
        }
    }

    void compile_prologue_steps(nonstd::span<Step *const> steps) {
        clear_shifted_counters();
        auto add_refcount = [&](Step *s) { bump_refcount(refcounts[s->id]); };
        for (Step *root : steps) {
            if (!root->is(StepKind::Const)) { step_recurse(root, add_refcount); }
        }
        for (const Step *root : steps) {
            if (const_is_folded_root(root)) { continue; }
            subexpr(root);
        }
    }

    void prepare_special_constants(nonstd::span<Step *const> steps, bool collect_special_constants = true) {
        uint64_t need_special_constants = 0;
        auto add_refcount = [&](Step *s) { bump_refcount(refcounts[s->id]); };
        traverse_steps_postorder_unique(step_map.size(), steps, [&](Step *s) {
            step_recurse(s, add_refcount);
            record_folded_const_ref(s);
            if (!collect_special_constants) { return; }

            if (s->dtype.is_mask() &&
                (s->is(StepKind::Load) ||
                 (s->is(StepKind::Store) && (s->dtype == MaskDataType::M8 || s->dtype == MaskDataType::M16)))) {
                need_special_constants |= 1 << SpecialConstant::Bits;
            }
            if (s->dtype == MaskDataType::M16 && s->is(StepKind::Load)) {
                need_special_constants |= 1 << SpecialConstant::Zero8One8;
            }
            if (s->dtype == MaskDataType::M2 && (s->is(StepKind::Load) || s->is(StepKind::Store))) {
                need_special_constants |= 1 << SpecialConstant::I64_MaskBits;
            }
            if (s->dtype == MaskDataType::M4 && (s->is(StepKind::Load) || s->is(StepKind::Store))) {
                need_special_constants |= 1 << SpecialConstant::I32_MaskBits;
            }
            if (s->is(StepKind::ArithBinary) && s->dtype.is_vec()) {
                auto op = s->step_data<StepKind::ArithBinary>().op;
                if (op == ArithBinaryOp::RotateLeft || op == ArithBinaryOp::RotateRight) {
                    VecDataType vdtype = s->dtype.as_vec();
                    if (vdtype.elem == VecElemType::I32) {
                        need_special_constants |= 1 << SpecialConstant::I32_32;
                    } else if (vdtype.elem == VecElemType::I64) {
                        need_special_constants |= 1 << SpecialConstant::I64_64;
                    }
                }
            }
            if (s->is(StepKind::ArithUnary) && s->dtype.is_vec()) {
                VecDataType vdtype = s->dtype.as_vec();
                auto op = s->step_data<StepKind::ArithUnary>().op;
                if (op == ArithUnaryOp::Lzcnt && vdtype.elem == VecElemType::I64) {
                    need_special_constants |= 1 << SpecialConstant::I32_32;
                }
                if (op == ArithUnaryOp::Popcount && vdtype.elem == VecElemType::I64) {
                    need_special_constants |= 1 << SpecialConstant::I8_1;
                }
            }
            if (s->is(StepKind::Fpclass) && s->dtype.is_mask()) {
                auto flags = s->step_data<StepKind::Fpclass>().flags;
                VecDataType vdtype = s->step_data<StepKind::Fpclass>().arg->dtype.as_vec();
                if (bool(flags & FpClass::FPC_INFINITE)) {
                    if (vdtype.elem == VecElemType::F32) need_special_constants |= 1 << SpecialConstant::F32_Inf;
                    if (vdtype.elem == VecElemType::F64) need_special_constants |= 1 << SpecialConstant::F64_Inf;
                }
                if (bool(flags & FpClass::FPC_SUBNORMAL)) {
                    if (vdtype.elem == VecElemType::F32)
                        need_special_constants |= (1 << SpecialConstant::F32_Inf) | (1 << SpecialConstant::F32_Mant);
                    if (vdtype.elem == VecElemType::F64)
                        need_special_constants |= (1 << SpecialConstant::F64_Inf) | (1 << SpecialConstant::F64_Mant);
                }
            }
            if (s->is(StepKind::Pack) && s->dtype.is_vec()) {
                VecDataType vdtype = s->dtype.as_vec();
                if (vdtype.elem == VecElemType::I8) {
                    need_special_constants |=
                        (1 << SpecialConstant::I8_PackIndices) | (1 << SpecialConstant::I8_MaskBits);
                } else if (vdtype.elem == VecElemType::I16) {
                    need_special_constants |=
                        (1 << SpecialConstant::I16_PackIndices) | (1 << SpecialConstant::I16_MaskBits);
                } else if (vdtype.elem == VecElemType::I32 || vdtype.elem == VecElemType::F32) {
                    need_special_constants |=
                        (1 << SpecialConstant::I32_PackIndices) | (1 << SpecialConstant::I32_MaskBits);
                } else if (vdtype.elem == VecElemType::I64 || vdtype.elem == VecElemType::F64) {
                    need_special_constants |=
                        (1 << SpecialConstant::I64_PackIndices) | (1 << SpecialConstant::I64_MaskBits);
                }
            }
        });

        if ((need_special_constants & (1 << SpecialConstant::Bits)) != 0) {
            GpR gp = cc.new_gp64();
            VecR reg = cc.new_vec128();
            init_int_const(gp, ConstData::u64(0x8040201008040201ull), ScalarDataType::I64);
            cc.dup(reg.d2(), gp);
            special_constants[SpecialConstant::Bits] = reg;
        }
        if ((need_special_constants & (1 << SpecialConstant::Zero8One8)) != 0) {
            GpR gp = cc.new_gp64();
            VecR reg = cc.new_vec128();
            cc.mov(gp, 0);
            cc.movi(reg.b16(), 1);
            cc.ins(reg.d(0), gp);
            special_constants[SpecialConstant::Zero8One8] = reg;
        }
        if ((need_special_constants & (1 << SpecialConstant::I32_32)) != 0) {
            VecR reg = cc.new_vec128();
            cc.movi(reg.s4(), 32);
            special_constants[SpecialConstant::I32_32] = reg.s4();
        }
        if ((need_special_constants & (1 << SpecialConstant::I64_64)) != 0) {
            GpR gp = cc.new_gp64();
            cc.mov(gp, 64);
            VecR reg = cc.new_vec128();
            cc.dup(reg.d2(), gp);
            special_constants[SpecialConstant::I64_64] = reg.d2();
        }
        if ((need_special_constants & (1 << SpecialConstant::F32_Inf)) != 0) {
            GpR tmpx = cc.new_gp32();
            VecR tmp1 = cc.new_vec128().s4();
            cc.mov(tmpx, 0x7F800000);
            cc.dup(tmp1, tmpx);
            special_constants[SpecialConstant::F32_Inf] = tmp1;
        }
        if ((need_special_constants & (1 << SpecialConstant::F32_Mant)) != 0) {
            GpR tmpx = cc.new_gp32();
            VecR tmp1 = cc.new_vec128().s4();
            cc.mov(tmpx, 0x007FFFFF);
            cc.dup(tmp1, tmpx);
            special_constants[SpecialConstant::F32_Mant] = tmp1;
        }
        if ((need_special_constants & (1 << SpecialConstant::F64_Inf)) != 0) {
            GpR tmpx = cc.new_gp64();
            VecR tmp1 = cc.new_vec128().d2();
            cc.mov(tmpx, 0x7FF0000000000000);
            cc.dup(tmp1, tmpx);
            special_constants[SpecialConstant::F64_Inf] = tmp1;
        }
        if ((need_special_constants & (1 << SpecialConstant::F64_Mant)) != 0) {
            GpR tmpx = cc.new_gp64();
            VecR tmp1 = cc.new_vec128().d2();
            cc.mov(tmpx, 0x000FFFFFFFFFFFFF);
            cc.dup(tmp1, tmpx);
            special_constants[SpecialConstant::F64_Mant] = tmp1;
        }
        if ((need_special_constants & (1 << SpecialConstant::I8_1)) != 0) {
            VecR reg = cc.new_vec128();
            cc.movi(reg.b16(), 1);
            special_constants[SpecialConstant::I8_1] = reg.b16();
        }
        if ((need_special_constants & (1 << SpecialConstant::I8_PackIndices)) != 0) {
            i8_pack_indices_label = cc.new_label();
            GpR gp = cc.new_gp64();
            cc.adr(gp, i8_pack_indices_label);
            special_constants[SpecialConstant::I8_PackIndices] = gp;
        }
        if ((need_special_constants & (1 << SpecialConstant::I8_MaskBits)) != 0) {
            const static uint8_t data[16] = {
                1, 2, 4, 8, 16, 32, 64, 128, //
                1, 2, 4, 8, 16, 32, 64, 128,
            };
            aja64::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, data, sizeof(data));
            VecR vec = cc.new_vec128();
            cc.ldr(vec.q(), mem);
            special_constants[SpecialConstant::I8_MaskBits] = vec;
        }
        if ((need_special_constants & (1 << SpecialConstant::I16_PackIndices)) != 0) {
            i16_pack_indices_label = cc.new_label();
            GpR gp = cc.new_gp64();
            cc.adr(gp, i16_pack_indices_label);
            special_constants[SpecialConstant::I16_PackIndices] = gp;
        }
        if ((need_special_constants & (1 << SpecialConstant::I16_MaskBits)) != 0) {
            const static uint8_t data[16] = {
                1, 0, 2, 0, 4, 0, 8, 0, 16, 0, 32, 0, 64, 0, 128, 0,
            };
            aja64::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, data, sizeof(data));
            VecR vec = cc.new_vec128();
            cc.ldr(vec.q(), mem);
            special_constants[SpecialConstant::I16_MaskBits] = vec;
        }
        if ((need_special_constants & (1 << SpecialConstant::I32_PackIndices)) != 0) {
            i32_pack_indices_label = cc.new_label();
            GpR gp = cc.new_gp64();
            cc.adr(gp, i32_pack_indices_label);
            special_constants[SpecialConstant::I32_PackIndices] = gp;
        }
        if ((need_special_constants & (1 << SpecialConstant::I32_MaskBits)) != 0) {
            const static uint8_t data[16] = {
                1, 0, 0, 0, //
                2, 0, 0, 0, //
                4, 0, 0, 0, //
                8, 0, 0, 0, //
            };
            aja64::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, data, sizeof(data));
            VecR vec = cc.new_vec128();
            cc.ldr(vec.q(), mem);
            special_constants[SpecialConstant::I32_MaskBits] = vec;
        }
        if ((need_special_constants & (1 << SpecialConstant::I64_PackIndices)) != 0) {
            i64_pack_indices_label = cc.new_label();
            GpR gp = cc.new_gp64();
            cc.adr(gp, i64_pack_indices_label);
            special_constants[SpecialConstant::I64_PackIndices] = gp;
        }
        if ((need_special_constants & (1 << SpecialConstant::I64_MaskBits)) != 0) {
            const static uint8_t data[16] = {
                1, 0, 0, 0, 0, 0, 0, 0, //
                2, 0, 0, 0, 0, 0, 0, 0, //
            };
            aja64::Mem mem = cc.new_const(aj::ConstPoolScope::kLocal, data, sizeof(data));
            VecR vec = cc.new_vec128();
            cc.ldr(vec.q(), mem);
            special_constants[SpecialConstant::I64_MaskBits] = vec;
        }
    }

    void emit_static_data() {
        static constexpr auto small_pack_indices = make_small_pack_index_tables();
        if (i8_pack_indices_label.is_valid()) {
            cc.align(aj::AlignMode::kData, 16);
            cc.bind(i8_pack_indices_label);
            cc.embed(small_pack_indices.i8, sizeof(small_pack_indices.i8));
        }
        if (i16_pack_indices_label.is_valid()) {
            cc.align(aj::AlignMode::kData, 16);
            cc.bind(i16_pack_indices_label);
            cc.embed(small_pack_indices.i16, sizeof(small_pack_indices.i16));
        }
        if (i32_pack_indices_label.is_valid()) {
            cc.align(aj::AlignMode::kData, 16);
            cc.bind(i32_pack_indices_label);
            // here 16 is used to mark out of range values
            const static uint8_t u8_indices[16 * 16] = {
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 0000 - []
                0,  1,  2,  3,  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 0001 - [a1]
                4,  5,  6,  7,  16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 0010 - [a2]
                0,  1,  2,  3,  4,  5,  6,  7,  16, 16, 16, 16, 16, 16, 16, 16, // 0011 - [a1, a2]
                8,  9,  10, 11, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 0100 - [a3]
                0,  1,  2,  3,  8,  9,  10, 11, 16, 16, 16, 16, 16, 16, 16, 16, // 0101 - [a1, a3]
                4,  5,  6,  7,  8,  9,  10, 11, 16, 16, 16, 16, 16, 16, 16, 16, // 0110 - [a2, a3]
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 16, 16, 16, 16, // 0111 - [a1, a2, a3]
                12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 1000 - [a4]
                0,  1,  2,  3,  12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, // 1001 - [a1, a4]
                4,  5,  6,  7,  12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, // 1010 - [a2, a4]
                0,  1,  2,  3,  4,  5,  6,  7,  12, 13, 14, 15, 16, 16, 16, 16, // 1011 - [a1, a2, a4]
                8,  9,  10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, // 1100 - [a3, a4]
                0,  1,  2,  3,  8,  9,  10, 11, 12, 13, 14, 15, 16, 16, 16, 16, // 1101 - [a1, a3, a4]
                4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 16, 16, 16, // 1110 - [a2, a3, a4]
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, // 1111 - [a1, a2, a3, a4]
            };
            cc.embed(u8_indices, sizeof(u8_indices));
        }
        if (i64_pack_indices_label.is_valid()) {
            cc.align(aj::AlignMode::kData, 16);
            cc.bind(i64_pack_indices_label);
            // here 16 is used to mark out of range values
            const static uint8_t u8_indices[16 * 4] = {
                16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, // 0000 - []
                0,  1,  2,  3,  4,  5,  6,  7,  16, 16, 16, 16, 16, 16, 16, 16, // 0001 - [a1]
                8,  9,  10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16, // 0010 - [a2]
                0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, // 0011 - [a1, a2]
            };
            cc.embed(u8_indices, sizeof(u8_indices));
        }
    }

    void init_args(nonstd::span<ArgumentDecl const> func_args, aj::FuncNode *func_node) {
        func_node->set_arg(0, row_count);

        for (const ArgumentDecl &arg : func_args) {
            GpR gp = cc.new_gp64();
            ArgInfo info{&arg, gp};
            func_node->set_arg(arg.idx + 1, gp);
            this->args[arg.idx] = std::move(info);
            if (SIMJIT_A64_ASMJIT_INDEX_REGS && (arg.kind == ArgumentKind::SrcArr || arg.kind == ArgumentKind::Dst ||
                                                 arg.kind == ArgumentKind::SrcIdxArr)) {
                index_regs[arg.idx] = {cc.new_gp64()};
            }
        }

        if (SIMJIT_A64_ASMJIT_INDEX_REGS) {
            for (size_t i = 0; i < index_regs.size(); ++i) {
                ArgumentIdx idx = i;
                if (!index_regs[idx].gp.is_valid()) continue;
                cc.mov(index_regs[idx].gp, args[idx].gp);
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
};
} // namespace

static aj::FuncNode *create_func_node(size_t arg_count, aja64::Compiler &cc) {
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

static void compile_asmjit(const mir::Function *func, aja64::Compiler &cc) {
    aj::FuncNode *func_node = create_func_node(func->args.size(), cc);

    CompileState state{func->ctx->arena, cc};
    // Note that we only pass func as argument, but don't save it inside CompileState. This makes it easier to see where
    // we interact with it.
    state.init(func);
    state.init_args(func->args, func_node);
    state.init_accs(func->prologue_roots);
    state.plan_mask_shift_writers(func);

    // Special constants are set of constant values that are used when evaluating certain expressions, but are not
    // reflected in MIR structure. They are usually ARM-specific hacks.
    // We only use them in SIMD code.
    if (!func->main_loop_roots.empty()) { state.prepare_special_constants(func->main_loop_roots); }
    if (!func->remainder_roots.empty()) { state.prepare_special_constants(func->remainder_roots, false); }

    // Prologue

    state.compile_prologue_steps(func->prologue_roots);
    state.init_mask_shift_writers();
    cc.cbz(state.row_count, state.end_label);
    cc.mov(state.counter, 0);

    // Main loop

    if (!func->main_loop_roots.empty()) {
        aj::Label main_loop_label = cc.new_named_label("main_loop");
        if (!func->remainder_roots.empty()) {
            cc.cmp(state.row_count, func->loop_width);
            cc.b(aja64::CondCode::kLO, state.remainder_label);
        }
        GpR last_vec_idx = cc.new_gp64();
        cc.and_(last_vec_idx, state.row_count, ~(func->loop_width - 1));
        cc.bind(main_loop_label);
        state.prepare_ldp(func->main_loop_roots);
        state.disable_index_regs_for_loop_overreads(func->main_loop_roots, func->loop_width);
        state.disable_index_regs_for_forward_gaps(func->main_loop_roots);
        state.compile_steps(func->main_loop_roots);
        cc.add(state.counter, state.counter, func->loop_width);
        cc.cmp(state.counter, last_vec_idx);
        cc.b(aja64::CondCode::kLO, main_loop_label);
    }

    // Scalar remainder

    aj::Label after_loops_label = cc.new_label();
    if (!func->remainder_roots.empty()) {
        if (!func->main_loop_roots.empty()) {
            cc.cmp(state.counter, state.row_count);
            cc.b(aja64::CondCode::kEQ, after_loops_label);
        }

        cc.bind(state.remainder_label);
        state.compile_steps(func->remainder_roots);
        cc.add(state.counter, state.counter, 1);
        cc.cmp(state.counter, state.row_count);
        cc.b(aja64::CondCode::kNE, state.remainder_label);
    }
    cc.bind(after_loops_label);

    // Epilogue

    state.emit_mask_shift_writer_epilogue();
    cc.bind(state.end_label);
    state.compile_steps(func->epilogue_roots);
    cc.ret();
    cc.end_func();
    state.emit_static_data();
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

    aja64::Compiler *cc = nullptr;
    aj::CodeHolder *code = nullptr;

    if (opts.session != nullptr) {
        if (opts.session->arch() != Arch::Arm64_NEON) { messed_up("asmjit a64 compile requires arm session"); }
        opts.session->reset();
        cc = static_cast<aja64::Compiler *>(&opts.session->compiler());
        code = &opts.session->code_holder();
    } else {
        tmp_state.session = std::make_unique<AsmjitSession>(Arch::Arm64_NEON);
        cc = static_cast<aja64::Compiler *>(&tmp_state.session->compiler());
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

} // namespace arm
} // namespace asmjit_backend

void compile_asmjit_arm(const mir::Function *func, const AsmjitCompileOptions &opts, AsmjitCompileResult &result) {
    asmjit_backend::arm::compile_asmjit(func, opts, result);
}

} // namespace simjit
