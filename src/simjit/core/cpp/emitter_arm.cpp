// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/cpp/emitter_internal.h"

#include <math.h>
#include <type_traits>

namespace simjit {
namespace cpp_backend {

#define messed_up(...) simjit_exception(ErrorModule::CPP, {}, {}, __VA_ARGS__)
#define unsupported(...) \
    simjit_exception(ErrorModule::CPP, ErrorKind::Unsupported, ErrorSubKind::UnsupportedBackendFeature, __VA_ARGS__)

static const char *arm_vec_dtype_cpp(VecDataType dtype) {
    switch (dtype.elem) {
    case VecElemType::I8:
        switch (dtype.size) {
        case VecSize::X8: return "int8x8_t";
        case VecSize::X16: return "int8x16_t";
        default: break;
        }
        break;
    case VecElemType::I16:
        switch (dtype.size) {
        case VecSize::X4: return "int16x4_t";
        case VecSize::X8: return "int16x8_t";
        default: break;
        }
        break;
    case VecElemType::I32:
        switch (dtype.size) {
        case VecSize::X2: return "int32x2_t";
        case VecSize::X4: return "int32x4_t";
        default: break;
        }
        break;
    case VecElemType::I64:
        switch (dtype.size) {
        case VecSize::X2: return "int64x2_t";
        default: break;
        }
        break;
    case VecElemType::F32:
        switch (dtype.size) {
        case VecSize::X2: return "float32x2_t";
        case VecSize::X4: return "float32x4_t";
        default: break;
        }
        break;
    case VecElemType::F64:
        switch (dtype.size) {
        case VecSize::X2: return "float64x2_t";
        default: break;
        }
        break;
    }
    messed_up("unsupported arm vector type %s", show_vec_dtype(dtype));
    SIMJIT_UNREACHABLE();
}

static const char *arm_mask_dtype_cpp(MaskDataType dtype) {
    switch (dtype) {
    case MaskDataType::M2: return "uint64x2_t";
    case MaskDataType::M4: return "uint32x4_t";
    case MaskDataType::M8: return "uint16x8_t";
    case MaskDataType::M16: return "uint8x16_t";
    case MaskDataType::M32:
    case MaskDataType::M64: messed_up("Invalid arm mask type %s", show_mask_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

static VecDataType arm_mask_vec_dtype(MaskDataType dtype) {
    switch (dtype) {
    case MaskDataType::M2: return VecDataType{VecSize::X2, VecElemType::I64};
    case MaskDataType::M4: return VecDataType{VecSize::X4, VecElemType::I32};
    case MaskDataType::M8: return VecDataType{VecSize::X8, VecElemType::I16};
    case MaskDataType::M16: return VecDataType{VecSize::X16, VecElemType::I8};
    case MaskDataType::M32:
    case MaskDataType::M64: messed_up("Invalid arm mask type %s", show_mask_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

static const char *arm_mask_lane_cpp(MaskDataType dtype) {
    switch (dtype) {
    case MaskDataType::M2: return "uint64_t";
    case MaskDataType::M4: return "uint32_t";
    case MaskDataType::M8: return "uint16_t";
    case MaskDataType::M16: return "uint8_t";
    case MaskDataType::M32:
    case MaskDataType::M64: messed_up("Invalid arm mask type %s", show_mask_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

static const char *arm_mask_lane_all_ones(MaskDataType dtype) {
    switch (dtype) {
    case MaskDataType::M2: return "~(uint64_t)0";
    case MaskDataType::M4: return "~(uint32_t)0";
    case MaskDataType::M8: return "0xffff";
    case MaskDataType::M16: return "0xff";
    case MaskDataType::M32:
    case MaskDataType::M64: messed_up("Invalid arm mask type %s", show_mask_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

static const char *arm_vec_suffix(VecElemType dtype) {
    switch (dtype) {
    case VecElemType::I8: return "s8";
    case VecElemType::I16: return "s16";
    case VecElemType::I32: return "s32";
    case VecElemType::I64: return "s64";
    case VecElemType::F32: return "f32";
    case VecElemType::F64: return "f64";
    }
    SIMJIT_UNREACHABLE();
}

static const char *arm_unsigned_vec_suffix(VecElemType dtype) {
    switch (dtype) {
    case VecElemType::I8: return "u8";
    case VecElemType::I16: return "u16";
    case VecElemType::I32: return "u32";
    case VecElemType::I64: return "u64";
    case VecElemType::F32: return "f32";
    case VecElemType::F64: return "f64";
    }
    SIMJIT_UNREACHABLE();
}

static const char *arm_mask_suffix(MaskDataType dtype) {
    switch (dtype) {
    case MaskDataType::M2: return "u64";
    case MaskDataType::M4: return "u32";
    case MaskDataType::M8: return "u16";
    case MaskDataType::M16:
    case MaskDataType::M32:
    case MaskDataType::M64: return "u8";
    }
    SIMJIT_UNREACHABLE();
}

static std::string arm_intrin_name(const char *op, VecDataType dtype, bool is_unsigned = false) {
    const char *q = dtype.size_bytes() == 16 ? "q" : "";
    const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
    return format("v%s%s_%s", op, q, suffix);
}

static std::string arm_float_intrin_name(const char *op, VecDataType dtype) {
    if (!dtype.is_float()) { messed_up("Expected float arm intrinsic type, got %s", show_vec_dtype(dtype)); }
    const char *q = dtype.size_bytes() == 16 ? "q" : "";
    const char *suffix = dtype.elem == VecElemType::F32 ? "f32" : "f64";
    return format("v%s%s_%s", op, q, suffix);
}

static std::string arm_dup_intrin_name(VecDataType dtype, bool is_unsigned = false) {
    const char *q = dtype.size_bytes() == 16 ? "q" : "";
    const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
    return format("vdup%s_n_%s", q, suffix);
}

static std::string arm_shift_imm_intrin_name(const char *op, VecDataType dtype, bool is_unsigned = false) {
    const char *q = dtype.size_bytes() == 16 ? "q" : "";
    const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
    return format("v%s%s_n_%s", op, q, suffix);
}

static std::string arm_lane_width_intrin_name(const char *op, VecDataType dtype, bool is_unsigned = false) {
    const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
    return format("v%s_%s", op, suffix);
}

static std::string arm_reduce_intrin_name(const char *op, VecDataType dtype, bool is_unsigned = false) {
    const char *q = dtype.size_bytes() == 16 ? "q" : "";
    const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
    return format("v%s%s_%s", op, q, suffix);
}

static std::string arm_pairwise_intrin_name(const char *op, VecDataType dtype, bool is_unsigned = false) {
    const char *q = dtype.size_bytes() == 16 ? "q" : "";
    const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
    return format("vp%s%s_%s", op, q, suffix);
}

static std::string arm_mask_intrin_name(const char *op, MaskDataType dtype) {
    return format("v%sq_%s", op, arm_mask_suffix(dtype));
}

static std::string arm_mask_dup_intrin_name(MaskDataType dtype) {
    return format("vdupq_n_%s", arm_mask_suffix(dtype));
}

struct ArmNeonCppEmitter : CppEmitterBase {
    using CppEmitterBase::CppEmitterBase;

    bool enable_vector_peepholes() const override { return false; }

    const char *cpp_dtype(DataType dtype) const override {
        switch (dtype.kind) {
        case DataTypeKind::Scalar: return scalar_dtype_to_cpp(dtype.scalar);
        case DataTypeKind::Vec: return arm_vec_dtype_cpp(dtype.vec);
        case DataTypeKind::Mask: return arm_mask_dtype_cpp(dtype.mask);
        }
        SIMJIT_UNREACHABLE();
    }

    bool step_supports_mask_pushdown(const Step *, bool) const { return false; }

    bool suppress_compacted_acc_store_args(const Step *step, ArenaBitmap &suppressed) const override {
        if (!step->is(StepKind::AccStore)) { return false; }

        const auto &store = step->step_data<StepKind::AccStore>();
        const Step *arg = store.arg;
        if (arg->dtype.is_scalar()) { return common_suppress_compacted_acc_store_args(step, suppressed); }
        if (use_counts[arg->id] != 1) { return false; }

        if (arg->is(StepKind::ArithBinary)) {
            const auto &bin = arg->step_data<StepKind::ArithBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return false; }
            suppressed.set(arg->id);
            suppressed.set(acc_load->id);
            return true;
        }

        if (arg->is(StepKind::MaskBinary)) {
            const auto &bin = arg->step_data<StepKind::MaskBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return false; }
            suppressed.set(arg->id);
            suppressed.set(acc_load->id);
            return true;
        }

        if (arg->is(StepKind::FMA)) {
            const auto &fma = arg->step_data<StepKind::FMA>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {fma.x1, fma.x2, fma.x3});
            if (acc_load == nullptr) { return false; }
            suppressed.set(arg->id);
            suppressed.set(acc_load->id);
            return true;
        }

        if (arg->is(StepKind::Select) && arg->dtype.is_vec()) {
            const auto &select = arg->step_data<StepKind::Select>();
            if (!select.falsy->is(StepKind::AccLoad) || select.falsy->step_data<StepKind::AccLoad>() != store.acc) {
                return false;
            }
            if (use_counts[select.falsy->id] == 0) { return false; }

            if (select.truthy->is(StepKind::ArithBinary) && use_counts[select.truthy->id] == 1) {
                const auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
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

            if (select.truthy->is(StepKind::FMA) && use_counts[select.truthy->id] == 1) {
                const auto &fma = select.truthy->step_data<StepKind::FMA>();
                const Step *acc_load = find_compacted_acc_load(store.acc, {fma.x1, fma.x2, fma.x3},
                                                               {fma.x1, fma.x2, fma.x3, select.falsy});
                if (acc_load == nullptr) { return false; }
                if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return false; }
                suppressed.set(arg->id);
                suppressed.set(select.truthy->id);
                suppressed.set(acc_load->id);
                suppressed.set(select.falsy->id);
                return true;
            }
        }

        return false;
    }

    std::string show_acc_store_operand(const Step *step, const Step *acc_load, AccId acc) {
        if (step == acc_load) { return format("acc%zu", func->accs.index(acc)); }
        if (can_inline_scalar_expr(step)) { return show_scalar_operand(step); }
        return show(step);
    }

    std::optional<std::string> backend_compact_acc_store_to_cpp(const Step *step) override {
        if (!step->is(StepKind::AccStore)) { return std::nullopt; }
        const auto &store = step->step_data<StepKind::AccStore>();
        const Step *arg = store.arg;
        if (arg->dtype.is_scalar()) { return common_compact_acc_store_to_cpp(step); }
        if (use_counts[arg->id] != 1) { return std::nullopt; }

        if (arg->is(StepKind::ArithBinary)) {
            if (!arg->dtype.is_vec()) { return std::nullopt; }
            const auto &bin = arg->step_data<StepKind::ArithBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return std::nullopt; }
            std::string left = show_acc_store_operand(bin.left, acc_load, store.acc);
            std::string right = show_acc_store_operand(bin.right, acc_load, store.acc);
            return binary_expr(arg, bin, left.c_str(), right.c_str());
        }

        if (arg->is(StepKind::MaskBinary)) {
            if (!arg->dtype.is_mask()) { return std::nullopt; }
            const auto &bin = arg->step_data<StepKind::MaskBinary>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
            if (acc_load == nullptr) { return std::nullopt; }
            std::string left = show_acc_store_operand(bin.left, acc_load, store.acc);
            std::string right = show_acc_store_operand(bin.right, acc_load, store.acc);
            return mask_binary_expr(arg->dtype.as_mask(), bin.op, left.c_str(), right.c_str());
        }

        if (arg->is(StepKind::FMA)) {
            if (!arg->dtype.is_vec()) { return std::nullopt; }
            const auto &fma = arg->step_data<StepKind::FMA>();
            const Step *acc_load = find_compacted_acc_load(store.acc, {fma.x1, fma.x2, fma.x3});
            if (acc_load == nullptr) { return std::nullopt; }
            std::string x1 = show_acc_store_operand(fma.x1, acc_load, store.acc);
            std::string x2 = show_acc_store_operand(fma.x2, acc_load, store.acc);
            std::string x3 = show_acc_store_operand(fma.x3, acc_load, store.acc);
            return fma_expr(arg->dtype.as_vec(), fma, x1.c_str(), x2.c_str(), x3.c_str());
        }

        if (arg->is(StepKind::Select) && arg->dtype.is_vec()) {
            const auto &select = arg->step_data<StepKind::Select>();
            if (!select.cond->dtype.is_mask() || !select.falsy->is(StepKind::AccLoad) ||
                select.falsy->step_data<StepKind::AccLoad>() != store.acc) {
                return std::nullopt;
            }

            if (select.truthy->is(StepKind::ArithBinary) && use_counts[select.truthy->id] == 1) {
                const auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
                const Step *acc_load =
                    find_compacted_acc_load(store.acc, {bin.left, bin.right}, {bin.left, bin.right, select.falsy});
                if (acc_load == nullptr) { return std::nullopt; }
                if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return std::nullopt; }
                std::string left = show_acc_store_operand(bin.left, acc_load, store.acc);
                std::string right = show_acc_store_operand(bin.right, acc_load, store.acc);
                auto truthy = binary_expr(select.truthy, bin, left.c_str(), right.c_str());
                if (!truthy.has_value()) { return std::nullopt; }
                std::string falsy = show_acc_store_operand(select.falsy, acc_load, store.acc);
                return vector_select_expr(arg->dtype.as_vec(), select.cond->dtype.as_mask(), show(select.cond),
                                          truthy->c_str(), falsy.c_str());
            }

            if (select.truthy->is(StepKind::FMA) && use_counts[select.truthy->id] == 1) {
                const auto &fma = select.truthy->step_data<StepKind::FMA>();
                const Step *acc_load = find_compacted_acc_load(store.acc, {fma.x1, fma.x2, fma.x3},
                                                               {fma.x1, fma.x2, fma.x3, select.falsy});
                if (acc_load == nullptr) { return std::nullopt; }
                if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return std::nullopt; }
                std::string x1 = show_acc_store_operand(fma.x1, acc_load, store.acc);
                std::string x2 = show_acc_store_operand(fma.x2, acc_load, store.acc);
                std::string x3 = show_acc_store_operand(fma.x3, acc_load, store.acc);
                std::string truthy = fma_expr(select.truthy->dtype.as_vec(), fma, x1.c_str(), x2.c_str(), x3.c_str());
                std::string falsy = show_acc_store_operand(select.falsy, acc_load, store.acc);
                return vector_select_expr(arg->dtype.as_vec(), select.cond->dtype.as_mask(), show(select.cond),
                                          truthy.c_str(), falsy.c_str());
            }
        }

        return std::nullopt;
    }

    enum class MaskShiftWriterCandidateState {
        Empty,
        VectorOnly,
        ScalarOnly,
        VectorAndScalar,
        Rejected,
    };

    struct MaskShiftWriterCandidate {
        MaskShiftWriterCandidateState state = MaskShiftWriterCandidateState::Empty;
        size_t vector_cursor = 0;
        size_t scalar_cursor = 0;
    };

    struct MaskShiftWriterState {
        bool active = false;
    };

    enum ArmSpecialConstant : uint32_t {
        ArmConstBits = 1u << 0,
        ArmConstZero8One8 = 1u << 1,
        ArmConstI16MaskBits = 1u << 2,
        ArmConstI32MaskBits = 1u << 3,
        ArmConstI64MaskBits = 1u << 4,
    };

    enum ArmHelper : uint32_t {
        ArmHelperFpclassF32 = 1u << 0,
        ArmHelperFpclassF64 = 1u << 1,
    };

    static constexpr unsigned MASK_SHIFT_WRITER_FLUSH_BITS = 16;
    static constexpr unsigned MASK_SHIFT_WRITER_FLUSH_MASK = MASK_SHIFT_WRITER_FLUSH_BITS - 1;

    std::vector<MaskShiftWriterState> mask_shift_writers{};
    uint32_t arm_special_constants = 0;
    uint32_t arm_helpers = 0;

    static size_t mask_shift_writer_bit_count(const Step *step) {
        if (step->dtype == ScalarDataType::I1) { return 1; }
        return mask_dtype_bits(step->dtype.as_mask());
    }

    static bool is_vector_mask_shift_writer_store(const Step *step) {
        return step->is(StepKind::Store) && step->dtype.is_mask() &&
               (step->dtype == MaskDataType::M2 || step->dtype == MaskDataType::M4);
    }

    static bool is_scalar_mask_shift_writer_store(const Step *step) {
        return step->is(StepKind::Store) && step->dtype == ScalarDataType::I1;
    }

    bool is_mask_shift_writer_root(const Step *step, EmitPhase phase) const {
        if (step == nullptr || !step->is(StepKind::Store)) { return false; }
        if (phase == EmitPhase::MainLoop) {
            if (!is_vector_mask_shift_writer_store(step)) { return false; }
        } else if (phase == EmitPhase::Remainder) {
            if (!is_scalar_mask_shift_writer_store(step)) { return false; }
        } else {
            return false;
        }
        const auto &data = step->step_data<StepKind::Store>();
        return data.addr.arg < mask_shift_writers.size() && mask_shift_writers[data.addr.arg].active;
    }

    void collect_arm_special_constants(nonstd::span<Step *const> step_roots, EmitPhase phase) {
        traverse_steps_postorder_unique(func->step_id_count, step_roots, [&](Step *step) {
            if (step->is(StepKind::Load) && step->dtype.is_mask()) {
                switch (step->dtype.as_mask()) {
                case MaskDataType::M2: arm_special_constants |= ArmConstI64MaskBits; break;
                case MaskDataType::M4: arm_special_constants |= ArmConstI32MaskBits; break;
                case MaskDataType::M8: arm_special_constants |= ArmConstI16MaskBits; break;
                case MaskDataType::M16: arm_special_constants |= ArmConstBits | ArmConstZero8One8; break;
                case MaskDataType::M32:
                case MaskDataType::M64: break;
                }
            }
            if (!step->is(StepKind::Store) || !step->dtype.is_mask()) { return; }
            switch (step->dtype.as_mask()) {
            case MaskDataType::M2:
                if (is_mask_shift_writer_root(step, phase)) { arm_special_constants |= ArmConstI64MaskBits; }
                break;
            case MaskDataType::M4:
                if (is_mask_shift_writer_root(step, phase)) { arm_special_constants |= ArmConstI32MaskBits; }
                break;
            case MaskDataType::M8:
            case MaskDataType::M16: arm_special_constants |= ArmConstBits; break;
            case MaskDataType::M32:
            case MaskDataType::M64: break;
            }
        });
    }

    void collect_arm_helpers(nonstd::span<Step *const> step_roots) {
        traverse_steps_postorder_unique(func->step_id_count, step_roots, [&](Step *step) {
            if (!step->is(StepKind::Fpclass)) { return; }
            const auto &data = step->step_data<StepKind::Fpclass>();
            if (!data.arg->dtype.is_vec()) { return; }
            switch (data.arg->dtype.as_vec().elem) {
            case VecElemType::F32: arm_helpers |= ArmHelperFpclassF32; break;
            case VecElemType::F64: arm_helpers |= ArmHelperFpclassF64; break;
            case VecElemType::I8:
            case VecElemType::I16:
            case VecElemType::I32:
            case VecElemType::I64: break;
            }
        });
    }

    void prepare_emit() override {
        mask_shift_writers.assign(func->args.size(), {});
        arm_special_constants = 0;
        arm_helpers = 0;
        std::vector<MaskShiftWriterCandidate> candidates(func->args.size());

        for (Step *root : func->main_loop_roots) {
            if (!is_vector_mask_shift_writer_store(root)) { continue; }
            const auto &data = root->step_data<StepKind::Store>();
            MaskShiftWriterCandidate &candidate = candidates[data.addr.arg];
            if (candidate.state == MaskShiftWriterCandidateState::Rejected) { continue; }

            unsigned bit_count = mask_shift_writer_bit_count(root);
            if (data.addr.offset != candidate.vector_cursor) {
                candidate.state = MaskShiftWriterCandidateState::Rejected;
                continue;
            }
            candidate.vector_cursor += bit_count;
            if (candidate.state == MaskShiftWriterCandidateState::Empty) {
                candidate.state = MaskShiftWriterCandidateState::VectorOnly;
            } else if (candidate.state == MaskShiftWriterCandidateState::ScalarOnly) {
                candidate.state = MaskShiftWriterCandidateState::VectorAndScalar;
            }
        }

        for (Step *root : func->remainder_roots) {
            if (!is_scalar_mask_shift_writer_store(root)) { continue; }
            const auto &data = root->step_data<StepKind::Store>();
            MaskShiftWriterCandidate &candidate = candidates[data.addr.arg];
            if (candidate.state == MaskShiftWriterCandidateState::Rejected) { continue; }

            if (data.addr.offset != candidate.scalar_cursor) {
                candidate.state = MaskShiftWriterCandidateState::Rejected;
                continue;
            }
            candidate.scalar_cursor += mask_shift_writer_bit_count(root);
            if (candidate.state == MaskShiftWriterCandidateState::Empty) {
                candidate.state = MaskShiftWriterCandidateState::ScalarOnly;
            } else if (candidate.state == MaskShiftWriterCandidateState::VectorOnly) {
                candidate.state = MaskShiftWriterCandidateState::VectorAndScalar;
            }
        }

        for (Step *root : func->main_loop_roots) {
            if (!root->is(StepKind::Store)) { continue; }
            const auto &data = root->step_data<StepKind::Store>();
            if (!is_vector_mask_shift_writer_store(root)) {
                candidates[data.addr.arg].state = MaskShiftWriterCandidateState::Rejected;
            }
        }
        for (Step *root : func->remainder_roots) {
            if (!root->is(StepKind::Store)) { continue; }
            const auto &data = root->step_data<StepKind::Store>();
            if (!is_scalar_mask_shift_writer_store(root)) {
                candidates[data.addr.arg].state = MaskShiftWriterCandidateState::Rejected;
            }
        }

        for (size_t arg = 0; arg < candidates.size(); ++arg) {
            MaskShiftWriterCandidate candidate = candidates[arg];
            if (candidate.state != MaskShiftWriterCandidateState::ScalarOnly &&
                candidate.state != MaskShiftWriterCandidateState::VectorAndScalar) {
                continue;
            }
            if (candidate.state == MaskShiftWriterCandidateState::VectorAndScalar &&
                candidate.vector_cursor != func->loop_width) {
                continue;
            }
            mask_shift_writers[arg].active = true;
        }

        collect_arm_special_constants(func->prologue_roots, EmitPhase::Prologue);
        collect_arm_special_constants(func->main_loop_roots, EmitPhase::MainLoop);
        collect_arm_special_constants(func->remainder_roots, EmitPhase::Remainder);
        collect_arm_special_constants(func->epilogue_roots, EmitPhase::Epilogue);

        collect_arm_helpers(func->prologue_roots);
        collect_arm_helpers(func->main_loop_roots);
        collect_arm_helpers(func->remainder_roots);
        collect_arm_helpers(func->epilogue_roots);
    }

    static constexpr const char *ARM_CONST_BITS = "simjit_arm_const_bits";
    static constexpr const char *ARM_CONST_ZERO8_ONE8 = "simjit_arm_const_zero8_one8";
    static constexpr const char *ARM_CONST_I16_MASK_BITS = "simjit_arm_const_i16_mask_bits";
    static constexpr const char *ARM_CONST_I32_MASK_BITS = "simjit_arm_const_i32_mask_bits";
    static constexpr const char *ARM_CONST_I64_MASK_BITS = "simjit_arm_const_i64_mask_bits";

    std::string arm_special_constants_to_cpp(size_t indent_level) const {
        const char *indent = indent_str(indent_level);
        std::string result{};
        if ((arm_special_constants & ArmConstBits) != 0) {
            format_to(result, "%sconst uint8x16_t %s = vreinterpretq_u8_u64(vdupq_n_u64(0x8040201008040201ULL));\n",
                      indent, ARM_CONST_BITS);
        }
        if ((arm_special_constants & ArmConstZero8One8) != 0) {
            format_to(result,
                      "%sconst uint8x16_t %s = uint8x16_t{0, 0, 0, 0, 0, 0, 0, 0, "
                      "1, 1, 1, 1, 1, 1, 1, 1};\n",
                      indent, ARM_CONST_ZERO8_ONE8);
        }
        if ((arm_special_constants & ArmConstI16MaskBits) != 0) {
            format_to(result, "%sconst uint16x8_t %s = uint16x8_t{1, 2, 4, 8, 16, 32, 64, 128};\n", indent,
                      ARM_CONST_I16_MASK_BITS);
        }
        if ((arm_special_constants & ArmConstI32MaskBits) != 0) {
            format_to(result, "%sconst uint32x4_t %s = uint32x4_t{1, 2, 4, 8};\n", indent, ARM_CONST_I32_MASK_BITS);
        }
        if ((arm_special_constants & ArmConstI64MaskBits) != 0) {
            format_to(result, "%sconst uint64x2_t %s = uint64x2_t{1, 2};\n", indent, ARM_CONST_I64_MASK_BITS);
        }
        return result;
    }

    std::string arm_helpers_to_cpp() const {
        std::string result{};
        if ((arm_helpers & ArmHelperFpclassF32) != 0) {
            result += R"CPP(static inline uint32x4_t simjit_arm_fpclass_f32(float32x4_t x, uint32_t flags) {
    uint32x4_t bits = vreinterpretq_u32_f32(x);
    uint32x4_t acc = vdupq_n_u32(0);
    if ((flags & 0x1u) != 0) {
        uint32x4_t abs_bits = vandq_u32(bits, vdupq_n_u32(0x7fffffffu));
        acc = vorrq_u32(acc, vceqq_u32(abs_bits, vdupq_n_u32(0x7f800000u)));
    }
    if ((flags & 0x2u) != 0) {
        uint32x4_t ordered = vceqq_f32(x, x);
        acc = vorrq_u32(acc, veorq_u32(ordered, vdupq_n_u32(~(uint32_t)0)));
    }
    if ((flags & 0x4u) != 0) {
        uint32x4_t exp_zero = vceqq_u32(vandq_u32(bits, vdupq_n_u32(0x7f800000u)), vdupq_n_u32(0));
        uint32x4_t mant = vandq_u32(bits, vdupq_n_u32(0x007fffffu));
        uint32x4_t mant_nonzero = veorq_u32(vceqq_u32(mant, vdupq_n_u32(0)), vdupq_n_u32(~(uint32_t)0));
        acc = vorrq_u32(acc, vandq_u32(exp_zero, mant_nonzero));
    }
    if ((flags & 0x8u) != 0) {
        acc = vorrq_u32(acc, vceqq_f32(vabsq_f32(x), vdupq_n_f32(0.0f)));
    }
    return acc;
}

)CPP";
        }
        if ((arm_helpers & ArmHelperFpclassF64) != 0) {
            result += R"CPP(static inline uint64x2_t simjit_arm_fpclass_f64(float64x2_t x, uint32_t flags) {
    uint64x2_t bits = vreinterpretq_u64_f64(x);
    uint64x2_t acc = vdupq_n_u64(0);
    if ((flags & 0x1u) != 0) {
        uint64x2_t abs_bits = vandq_u64(bits, vdupq_n_u64(0x7fffffffffffffffULL));
        acc = vorrq_u64(acc, vceqq_u64(abs_bits, vdupq_n_u64(0x7ff0000000000000ULL)));
    }
    if ((flags & 0x2u) != 0) {
        uint64x2_t ordered = vceqq_f64(x, x);
        acc = vorrq_u64(acc, veorq_u64(ordered, vdupq_n_u64(~(uint64_t)0)));
    }
    if ((flags & 0x4u) != 0) {
        uint64x2_t exp_zero = vceqq_u64(vandq_u64(bits, vdupq_n_u64(0x7ff0000000000000ULL)), vdupq_n_u64(0));
        uint64x2_t mant = vandq_u64(bits, vdupq_n_u64(0x000fffffffffffffULL));
        uint64x2_t mant_nonzero = veorq_u64(vceqq_u64(mant, vdupq_n_u64(0)), vdupq_n_u64(~(uint64_t)0));
        acc = vorrq_u64(acc, vandq_u64(exp_zero, mant_nonzero));
    }
    if ((flags & 0x8u) != 0) {
        acc = vorrq_u64(acc, vceqq_f64(vabsq_f64(x), vdupq_n_f64(0.0)));
    }
    return acc;
}

)CPP";
        }
        return result;
    }

    std::string source_prelude_to_cpp() const override { return arm_helpers_to_cpp(); }

    std::string function_prelude_to_cpp(size_t indent_level) const override {
        return arm_special_constants_to_cpp(indent_level);
    }

    std::string before_loops_to_cpp(size_t indent_level) override {
        const char *indent = indent_str(indent_level);
        std::string result{};
        for (size_t arg = 0; arg < mask_shift_writers.size(); ++arg) {
            if (!mask_shift_writers[arg].active) { continue; }
            format_to(result, "%suint8_t *mask_shift_writer_dst_%zu = arg%zu;\n", indent, arg, arg);
            format_to(result, "%suint32_t mask_shift_writer_acc_%zu = 0;\n", indent, arg);
        }
        return result;
    }

    std::string after_loops_to_cpp(size_t indent_level) override {
        const char *indent = indent_str(indent_level);
        std::string result{};
        for (size_t arg = 0; arg < mask_shift_writers.size(); ++arg) {
            if (!mask_shift_writers[arg].active) { continue; }
            format_to(result, "%sif ((nelems & %u) != 0) {\n", indent, MASK_SHIFT_WRITER_FLUSH_MASK);
            format_to(result, "%s    uint32_t mask_shift_writer_pending_%zu = (uint32_t)(nelems & %u);\n", indent, arg,
                      MASK_SHIFT_WRITER_FLUSH_MASK);
            format_to(result, "%s    if (mask_shift_writer_pending_%zu >= 8) {\n", indent, arg);
            format_to(result, "%s        *mask_shift_writer_dst_%zu++ = (uint8_t)mask_shift_writer_acc_%zu;\n", indent,
                      arg, arg);
            format_to(result, "%s        mask_shift_writer_acc_%zu >>= 8;\n", indent, arg);
            format_to(result, "%s        mask_shift_writer_pending_%zu -= 8;\n", indent, arg);
            format_to(result, "%s    }\n", indent);
            format_to(result, "%s    if (mask_shift_writer_pending_%zu != 0) {\n", indent, arg);
            format_to(result,
                      "%s        uint8_t mask_shift_writer_mask_%zu = "
                      "(uint8_t)((1u << mask_shift_writer_pending_%zu) - 1u);\n",
                      indent, arg, arg);
            format_to(result,
                      "%s        *mask_shift_writer_dst_%zu = "
                      "(uint8_t)((*mask_shift_writer_dst_%zu & ~mask_shift_writer_mask_%zu) | "
                      "(mask_shift_writer_acc_%zu & mask_shift_writer_mask_%zu));\n",
                      indent, arg, arg, arg, arg, arg);
            format_to(result, "%s    }\n", indent);
            format_to(result, "%s}\n", indent);
        }
        return result;
    }

    bool has_custom_step_to_cpp(const Step *step, EmitPhase phase) const override {
        return is_mask_shift_writer_root(step, phase);
    }

    std::string mask_shift_writer_bits_expr(const Step *step, const StoreData &data) {
        if (step->dtype == ScalarDataType::I1) {
            return format("((uint32_t)(%s) & 1u)", show_scalar_operand(data.what));
        }
        MaskDataType dtype = step->dtype.as_mask();
        if (dtype == MaskDataType::M2) {
            return format("((uint32_t)vaddvq_u64(vandq_u64(%s, %s)))", show(data.what), ARM_CONST_I64_MASK_BITS);
        }
        if (dtype == MaskDataType::M4) {
            return format("vaddvq_u32(vandq_u32(%s, %s))", show(data.what), ARM_CONST_I32_MASK_BITS);
        }
        SIMJIT_UNREACHABLE();
    }

    std::string custom_step_to_cpp(const Step *step, EmitPhase phase) override {
        (void)phase;
        SIMJIT_ASSERT(is_mask_shift_writer_root(step, phase));
        const auto &data = step->step_data<StepKind::Store>();
        size_t arg = data.addr.arg;
        size_t id = step->id;
        unsigned bit_count = mask_shift_writer_bit_count(step);
        std::string bits = mask_shift_writer_bits_expr(step, data);
        std::string shift_pos = data.addr.offset == 0 ? "i" : format("i + %zu", data.addr.offset);
        std::string flush_pos = format("i + %zu", data.addr.offset + bit_count);

        std::string result = "do {\n";
        format_to(result, "uint32_t mask_shift_writer_bits_%zu = %s;\n", id, bits.c_str());
        format_to(result,
                  "mask_shift_writer_acc_%zu |= "
                  "(mask_shift_writer_bits_%zu << ((%s) & %u));\n",
                  arg, id, shift_pos.c_str(), MASK_SHIFT_WRITER_FLUSH_MASK);
        format_to(result, "if (((%s) & %u) == 0) {\n", flush_pos.c_str(), MASK_SHIFT_WRITER_FLUSH_MASK);
        format_to(result, "uint16_t mask_shift_writer_flush_%zu = (uint16_t)mask_shift_writer_acc_%zu;\n", id, arg);
        format_to(result,
                  "__builtin_memcpy(mask_shift_writer_dst_%zu, &mask_shift_writer_flush_%zu, "
                  "sizeof(mask_shift_writer_flush_%zu));\n",
                  arg, id, id);
        format_to(result, "mask_shift_writer_dst_%zu += sizeof(mask_shift_writer_flush_%zu);\n", arg, id);
        format_to(result, "mask_shift_writer_acc_%zu = 0;\n", arg);
        result += "}\n";
        result += "} while (false)";
        return result;
    }

    static std::string offset_suffix(size_t offset) { return offset == 0 ? "" : format(" + %zu", offset); }

    std::string unexpected_step_dtype(const Step *step, const char *expected) const {
        messed_up("unexpected ARM C++ dtype %s for %s step, expected %s", show_dtype(step->dtype),
                  show_step_kind(step->kind), expected);
        SIMJIT_UNREACHABLE();
    }

    std::string mask_not_expr(MaskDataType dtype, const char *arg) const {
        if (dtype == MaskDataType::M2) {
            return format("%s(%s, %s(%s))", arm_mask_intrin_name("eor", dtype).c_str(), arg,
                          arm_mask_dup_intrin_name(dtype).c_str(), arm_mask_lane_all_ones(dtype));
        }
        return format("%s(%s)", arm_mask_intrin_name("mvn", dtype).c_str(), arg);
    }

    static const char *arm_mask_signed_suffix(MaskDataType dtype) {
        switch (dtype) {
        case MaskDataType::M2: return "s64";
        case MaskDataType::M4: return "s32";
        case MaskDataType::M8: return "s16";
        case MaskDataType::M16:
        case MaskDataType::M32:
        case MaskDataType::M64: return "s8";
        }
        SIMJIT_UNREACHABLE();
    }

    static size_t arm_mask_lane_size(MaskDataType dtype) {
        switch (dtype) {
        case MaskDataType::M2: return 8;
        case MaskDataType::M4: return 4;
        case MaskDataType::M8: return 2;
        case MaskDataType::M16: return 1;
        case MaskDataType::M32:
        case MaskDataType::M64: messed_up("Invalid arm mask type %s", show_mask_dtype(dtype));
        }
        SIMJIT_UNREACHABLE();
    }

    std::string mask_shrink_once_expr(MaskDataType from, const char *arg) const {
        auto maybe_to = double_mask(from);
        if (!maybe_to.has_value() || *maybe_to == MaskDataType::M32) {
            messed_up("Invalid arm mask shrink from %s", show_mask_dtype(from));
        }
        MaskDataType to = *maybe_to;
        return format("vcombine_%s(vmovn_%s(%s), vdup_n_%s(0))", arm_mask_suffix(to), arm_mask_suffix(from), arg,
                      arm_mask_suffix(to));
    }

    std::string mask_expand_once_expr(MaskDataType from, const char *arg) const {
        auto maybe_to = half_mask(from);
        if (!maybe_to.has_value()) { messed_up("Invalid arm mask expand from %s", show_mask_dtype(from)); }
        MaskDataType to = *maybe_to;
        return format("vreinterpretq_%s_%s(vmovl_%s(vreinterpret_%s_%s(vget_low_%s(%s))))", arm_mask_suffix(to),
                      arm_mask_signed_suffix(to), arm_mask_signed_suffix(from), arm_mask_signed_suffix(from),
                      arm_mask_suffix(from), arm_mask_suffix(from), arg);
    }

    std::string mask_adjust_expr(MaskDataType from, MaskDataType to, std::string arg) const {
        if (from == to) { return arg; }
        if (arm_mask_lane_size(from) > arm_mask_lane_size(to)) {
            for (MaskDataType cur = from; cur != to;) {
                arg = mask_shrink_once_expr(cur, arg.c_str());
                auto maybe_double = double_mask(cur);
                SIMJIT_ASSERT(maybe_double.has_value());
                cur = *maybe_double;
            }
            return arg;
        }
        for (MaskDataType cur = from; cur != to;) {
            arg = mask_expand_once_expr(cur, arg.c_str());
            auto maybe_half = half_mask(cur);
            SIMJIT_ASSERT(maybe_half.has_value());
            cur = *maybe_half;
        }
        return arg;
    }

    std::string mask_to_vec_mask_expr(MaskDataType from, VecDataType to, const char *arg) const {
        return mask_adjust_expr(from, to.mask(), arg);
    }

    static std::string arm_mask_addv_intrin_name(MaskDataType dtype) {
        return format("vaddvq_%s", arm_mask_suffix(dtype));
    }

    std::string mask_count_expr(MaskDataType dtype, const char *arg) const {
        return format("((%s)(0 - %s(%s)))", arm_mask_lane_cpp(dtype), arm_mask_addv_intrin_name(dtype).c_str(), arg);
    }

    static std::string mask_array_name(const char *prefix, size_t id) { return format("%s_%zu", prefix, id); }

    std::string store_mask_to_array(MaskDataType dtype, const char *array, const char *value) const {
        std::string store = arm_intrin_name("st1", arm_mask_vec_dtype(dtype), true);
        return format("%s %s[%zu];\n"
                      "%s(%s, %s);\n",
                      arm_mask_lane_cpp(dtype), array, mask_dtype_bits(dtype), //
                      store.c_str(), array, value);
    }

    std::string mask_load_to_cpp(const Step *step, const LoadData &data) const {
        MaskDataType dtype = step->dtype.as_mask();
        size_t id = step->id;
        std::string row = data.addr.offset == 0 ? "i" : format("(i + %zu)", data.addr.offset);
        if (dtype == MaskDataType::M2 || dtype == MaskDataType::M4) {
            std::string byte =
                format("(uint8_t)(arg%zu[%s >> 3] >> (%s & 7))", data.addr.arg, row.c_str(), row.c_str());
            if (dtype == MaskDataType::M2) {
                return cpp_var_decl(step) +
                       format("vtstq_u64(vdupq_n_u64(%s), %s)", byte.c_str(), ARM_CONST_I64_MASK_BITS);
            }
            return cpp_var_decl(step) + format("vtstq_u32(vdupq_n_u32(%s), %s)", byte.c_str(), ARM_CONST_I32_MASK_BITS);
        }
        if (dtype == MaskDataType::M8) {
            size_t byte_offset = data.addr.offset >> 3;
            std::string offset = byte_offset == 0 ? "" : format(" + %zu", byte_offset);
            return cpp_var_decl(step) + format("vtstq_u16(vdupq_n_u16(arg%zu[(i >> 3)%s]), %s)", data.addr.arg,
                                               offset.c_str(), ARM_CONST_I16_MASK_BITS);
        }
        std::string result = format("%s %s;\n"
                                    "{\n",
                                    cpp_dtype(step->dtype), show(step));
        if (dtype == MaskDataType::M16) {
            size_t byte_offset = data.addr.offset >> 3;
            std::string offset = byte_offset == 0 ? "" : format(" + %zu", byte_offset);
            format_to(result, "uint16_t mask_loaded_%zu;\n", id);
            format_to(result,
                      "__builtin_memcpy(&mask_loaded_%zu, arg%zu + ((i >> 3)%s), "
                      "sizeof(mask_loaded_%zu));\n",
                      id, data.addr.arg, offset.c_str(), id);
            format_to(result, "uint8x16_t mask_replicated_%zu = vreinterpretq_u8_u16(vdupq_n_u16(mask_loaded_%zu));\n",
                      id, id);
            result +=
                format("mask_replicated_%zu = vqtbl1q_u8(mask_replicated_%zu, %s);\n", id, id, ARM_CONST_ZERO8_ONE8);
            format_to(result,
                      "%s = vtstq_u8(mask_replicated_%zu, %s);\n"
                      "}",
                      show(step), id, ARM_CONST_BITS);
            return result;
        }
        unsupported("Do not support ARM C++ mask load of %s", show_mask_dtype(dtype));
        SIMJIT_UNREACHABLE();
    }

    std::string mask_store_to_cpp(const StoreData &data, const Step *step) const {
        MaskDataType dtype = step->dtype.as_mask();
        if (dtype == MaskDataType::M8 || dtype == MaskDataType::M16) {
            size_t id = step->id;
            std::string byte_mask = mask_adjust_expr(data.what->dtype.as_mask(), MaskDataType::M16, show(data.what));
            std::string offset = data.addr.offset == 0 ? "" : format(" + %zu", data.addr.offset >> 3);
            std::string weighted = format("vandq_u8(%s, %s)", ARM_CONST_BITS, byte_mask.c_str());
            if (dtype == MaskDataType::M8) {
                return format("arg%zu[(i >> 3)%s] = vaddvq_u8(%s)", data.addr.arg, offset.c_str(), weighted.c_str());
            }
            std::string weighted_name = format("mask_weighted_%zu", id);
            std::string high = format("vextq_u8(%s, %s, 8)", weighted_name.c_str(), weighted_name.c_str());
            std::string zipped = format("vzip1q_u8(%s, %s)", weighted_name.c_str(), high.c_str());
            std::string packed = format("vaddvq_u16(vreinterpretq_u16_u8(%s))", zipped.c_str());
            std::string result = "{\n";
            format_to(result, "uint8x16_t %s = %s;\n", weighted_name.c_str(), weighted.c_str());
            format_to(result, "uint16_t mask_packed_%zu = %s;\n", id, packed.c_str());
            format_to(result,
                      "__builtin_memcpy(arg%zu + ((i >> 3)%s), &mask_packed_%zu, "
                      "sizeof(mask_packed_%zu));\n}",
                      data.addr.arg, offset.c_str(), id, id);
            return result;
        }

        size_t id = step->id;
        std::string values = mask_array_name("mask_store_values", id);
        std::string result = "{\n";
        result += store_mask_to_array(dtype, values.c_str(), show(data.what));
        for (size_t lane = 0; lane < mask_dtype_bits(dtype); ++lane) {
            format_to(result, "size_t mask_row_%zu_%zu = i + %zu + %zu;\n", id, lane, data.addr.offset, lane);
            format_to(result, "uint8_t *mask_byte_%zu_%zu = arg%zu + (mask_row_%zu_%zu >> 3);\n", id, lane,
                      data.addr.arg, id, lane);
            result +=
                format("uint8_t mask_bit_%zu_%zu = (uint8_t)(1u << (mask_row_%zu_%zu & 7));\n", id, lane, id, lane);
            format_to(result, "if (%s[%zu] != 0) *mask_byte_%zu_%zu |= mask_bit_%zu_%zu;\n", values.c_str(), lane, id,
                      lane, id, lane);
            format_to(result, "else *mask_byte_%zu_%zu = (uint8_t)(*mask_byte_%zu_%zu & ~mask_bit_%zu_%zu);\n", id,
                      lane, id, lane, id, lane);
        }
        result += "}";
        return result;
    }

    std::string mask_splat_to_cpp(const Step *step, const LoadData &data) const {
        MaskDataType dtype = step->dtype.as_mask();
        return cpp_var_decl(step) + format("%s((*arg%zu & 1) ? %s : 0)", arm_mask_dup_intrin_name(dtype).c_str(),
                                           data.addr.arg, arm_mask_lane_all_ones(dtype));
    }

    bool can_inline_acc_store_arg(const Step *arg) const {
        if (use_counts[arg->id] != 1) { return false; }
        return arg->is(StepKind::Const) || arg->is(StepKind::LoadSplat);
    }

    bool suppress_inlined_acc_store_args(const Step *step, ArenaBitmap &suppressed) const override {
        if (!step->is(StepKind::AccStore)) { return false; }
        const auto &data = step->step_data<StepKind::AccStore>();
        if (!can_inline_acc_store_arg(data.arg)) { return false; }
        suppressed.set(data.arg->id);
        return true;
    }

    std::optional<std::string> inline_acc_store_arg_expr(const Step *arg) const {
        if (!can_inline_acc_store_arg(arg)) { return std::nullopt; }

        if (arg->is(StepKind::Const)) {
            const auto &data = arg->step_data<StepKind::Const>();
            if (arg->dtype.is_scalar()) { return const_data_to_cpp(data, arg->dtype.as_scalar()); }
            if (arg->dtype.is_mask()) {
                MaskDataType dtype = arg->dtype.as_mask();
                const char *value = data.as_unsigned() != 0 ? arm_mask_lane_all_ones(dtype) : "0";
                return format("%s(%s)", arm_mask_dup_intrin_name(dtype).c_str(), value);
            }
            VecDataType dtype = arg->dtype.as_vec();
            return format("%s(%s)", arm_dup_intrin_name(dtype).c_str(),
                          const_data_to_cpp(data, dtype.to_scalar()).c_str());
        }

        if (arg->is(StepKind::LoadSplat)) {
            const auto &data = arg->step_data<StepKind::LoadSplat>();
            if (arg->dtype.is_scalar()) { return format("*arg%zu", data.addr.arg); }
            if (arg->dtype.is_mask()) {
                MaskDataType dtype = arg->dtype.as_mask();
                return format("%s((*arg%zu & 1) ? %s : 0)", arm_mask_dup_intrin_name(dtype).c_str(), data.addr.arg,
                              arm_mask_lane_all_ones(dtype));
            }
            VecDataType dtype = arg->dtype.as_vec();
            return format("%s(*arg%zu)", arm_dup_intrin_name(dtype).c_str(), data.addr.arg);
        }

        SIMJIT_UNREACHABLE();
    }

    std::string acc_store_arg_expr(const Step *arg) const {
        if (auto expr = inline_acc_store_arg_expr(arg)) { return *expr; }
        return show(arg);
    }

    std::string mask_binary_expr(MaskDataType dtype, PredicateBinaryOp op, const char *left, const char *right) const {
        switch (op) {
        case PredicateBinaryOp::And:
            return format("%s(%s, %s)", arm_mask_intrin_name("and", dtype).c_str(), left, right);
        case PredicateBinaryOp::Or:
            return format("%s(%s, %s)", arm_mask_intrin_name("orr", dtype).c_str(), left, right);
        case PredicateBinaryOp::Xor:
            return format("%s(%s, %s)", arm_mask_intrin_name("eor", dtype).c_str(), left, right);
        case PredicateBinaryOp::AndNot:
            return format("%s(%s, %s)", arm_mask_intrin_name("bic", dtype).c_str(), right, left);
        case PredicateBinaryOp::XNor: {
            std::string xored = format("%s(%s, %s)", arm_mask_intrin_name("eor", dtype).c_str(), left, right);
            return mask_not_expr(dtype, xored.c_str());
        }
        }
        SIMJIT_UNREACHABLE();
    }

    std::string mask_binary_to_cpp(const Step *step, const PredicateBinData &data) const {
        MaskDataType dtype = step->dtype.as_mask();
        return cpp_var_decl(step) + mask_binary_expr(dtype, data.op, show(data.left), show(data.right));
    }

    std::string mask_count_to_cpp(const Step *step, const Step *arg) const {
        MaskDataType dtype = arg->dtype.as_mask();
        return cpp_var_decl(step) + format("(%s)%s", cpp_dtype(step->dtype), mask_count_expr(dtype, show(arg)).c_str());
    }

    std::string mask_reduce_to_cpp(const Step *step, const PredicateReduceData &data) const {
        MaskDataType dtype = data.arg->dtype.as_mask();
        std::string byte_mask = mask_adjust_expr(dtype, MaskDataType::M16, show(data.arg));
        std::string count = mask_count_expr(MaskDataType::M16, byte_mask.c_str());
        std::string expr;
        switch (data.op) {
        case PredicateBinaryOp::And: expr = format("(%s == %zu)", count.c_str(), mask_dtype_bits(dtype)); break;
        case PredicateBinaryOp::Or: expr = format("(%s != 0)", count.c_str()); break;
        case PredicateBinaryOp::Xor: expr = format("(%s & 1)", count.c_str()); break;
        case PredicateBinaryOp::AndNot:
        case PredicateBinaryOp::XNor:
            messed_up("Invalid reduce %s of %s", show_predicate_binary_op(data.op), show_mask_dtype(dtype));
        }
        return cpp_var_decl(step) + format("(%s)%s", cpp_dtype(step->dtype), expr.c_str());
    }

    std::string mask_combine_to_cpp(const Step *step, const CombineMaskData &data) const {
        MaskDataType dtype = step->dtype.as_mask();
        if (dtype == MaskDataType::M2 || dtype == MaskDataType::M32 || dtype == MaskDataType::M64) {
            messed_up("Invalid combine mask target %s", show_mask_dtype(dtype));
        }

        MaskDataType left_dtype = data.left->dtype.as_mask();
        MaskDataType right_dtype = data.right->dtype.as_mask();
        if (left_dtype != right_dtype) {
            messed_up("Invalid combine mask operands %s and %s", show_mask_dtype(left_dtype),
                      show_mask_dtype(right_dtype));
        }
        auto expected = double_mask(left_dtype);
        if (!expected.has_value() || *expected != dtype) {
            messed_up("Invalid combine mask target %s from %s", show_mask_dtype(dtype), show_mask_dtype(left_dtype));
        }
        return cpp_var_decl(step) + format("vmovn_high_%s(vmovn_%s(%s), %s)", arm_mask_suffix(left_dtype),
                                           arm_mask_suffix(left_dtype), show(data.left), show(data.right));
    }

    std::string vector_select_expr(VecDataType dtype, MaskDataType cond_dtype, const char *cond, const char *truthy,
                                   const char *falsy) const {
        std::string adjusted_cond = mask_to_vec_mask_expr(cond_dtype, dtype, cond);
        return format("%s(%s, %s, %s)", arm_intrin_name("bsl", dtype).c_str(), adjusted_cond.c_str(), truthy, falsy);
    }

    std::string vector_select_to_cpp(const Step *step, const SelectData &data) {
        if (!step->dtype.is_vec() || !data.cond->dtype.is_mask()) {
            return unexpected_step_dtype(step, "vector select with mask condition");
        }
        VecDataType dtype = step->dtype.as_vec();
        return cpp_var_decl(step) + vector_select_expr(dtype, data.cond->dtype.as_mask(), show(data.cond),
                                                       show(data.truthy), show(data.falsy));
    }

    std::string vector_cond_store_to_cpp(const Step *step, const CondStoreData &data) {
        if (!step->dtype.is_vec() || !data.arg->dtype.is_vec() || !data.cond->dtype.is_mask()) {
            return unexpected_step_dtype(step, "vector cond-store with vector argument and mask condition");
        }

        VecDataType dtype = step->dtype.as_vec();
        std::string offset = offset_suffix(data.addr.offset);
        size_t id = step->id;
        MaskDataType cond_dtype = data.cond->dtype.as_mask();
        std::string cond = mask_to_vec_mask_expr(cond_dtype, dtype, show(data.cond));
        return format(R"STR(do {
%s old_%zu = %s(arg%zu + i%s);
%s blend_%zu = %s(%s, %s, old_%zu);
%s(arg%zu + i%s, blend_%zu);
} while (false))STR",
                      cpp_dtype(step->dtype), id, arm_intrin_name("ld1", dtype).c_str(), data.addr.arg, offset.c_str(),
                      cpp_dtype(step->dtype), id, arm_intrin_name("bsl", dtype).c_str(), cond.c_str(), show(data.arg),
                      id, arm_intrin_name("st1", dtype).c_str(), data.addr.arg, offset.c_str(), id);
    }

    std::string vector_gather_to_cpp(const Step *step, const GatherData &data) {
        if (!step->dtype.is_vec() || !data.idx->dtype.is_vec()) {
            return unexpected_step_dtype(step, "vector gather with vector index");
        }

        VecDataType dtype = step->dtype.as_vec();
        VecDataType idx_dtype = data.idx->dtype.as_vec();
        if ((idx_dtype.elem != VecElemType::I32 && idx_dtype.elem != VecElemType::I64) ||
            idx_dtype.nelems() != dtype.nelems()) {
            return unexpected_step_dtype(step, "vector gather with matching i32 or i64 index");
        }

        size_t id = step->id;
        std::string values =
            format("const %s gather_values_%zu[%zu] = {", cpp_dtype(dtype.to_scalar()), id, dtype.nelems());
        for (size_t lane = 0; lane < dtype.nelems(); ++lane) {
            if (lane != 0) { values += ", "; }
            format_to(values, "arg%zu[%s]", data.data, lane_expr(data.idx, lane).c_str());
        }
        format_to(values, "};\n%s%s(gather_values_%zu)", cpp_var_decl(step).c_str(),
                  arm_intrin_name("ld1", dtype).c_str(), id);
        return values;
    }

    static std::string lane_get_intrin(VecDataType dtype, bool is_unsigned = false) {
        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(dtype.elem) : arm_vec_suffix(dtype.elem);
        return format("vget%s_lane_%s", q, suffix);
    }

    static std::string mask_lane_get_intrin(MaskDataType dtype) {
        return format("vgetq_lane_%s", arm_mask_suffix(dtype));
    }

    std::string lane_expr(const Step *step, size_t lane) {
        if (step->dtype.is_vec()) {
            VecDataType dtype = step->dtype.as_vec();
            return format("%s(%s, %zu)", lane_get_intrin(dtype).c_str(), show(step), lane);
        }
        if (step->dtype.is_mask()) {
            MaskDataType dtype = step->dtype.as_mask();
            return format("%s(%s, %zu)", mask_lane_get_intrin(dtype).c_str(), show(step), lane);
        }
        return show_scalar_operand(step);
    }

    std::string vector_bitwise_expr(VecDataType dtype, const char *op, const char *left, const char *right) const {
        if (dtype.is_int()) { return format("%s(%s, %s)", arm_intrin_name(op, dtype).c_str(), left, right); }

        VecElemType uint_elem = dtype.elem == VecElemType::F32 ? VecElemType::I32 : VecElemType::I64;
        VecDataType uint_dtype{dtype.size, uint_elem};
        const char *to_uint = dtype.elem == VecElemType::F32 ? "vreinterpretq_u32_f32" : "vreinterpretq_u64_f64";
        const char *from_uint = dtype.elem == VecElemType::F32 ? "vreinterpretq_f32_u32" : "vreinterpretq_f64_u64";
        return format("%s(%s(%s(%s), %s(%s)))", from_uint, arm_intrin_name(op, uint_dtype, true).c_str(), to_uint, left,
                      to_uint, right);
    }

    std::string vector_not_expr(VecDataType dtype, const char *arg) const {
        if (dtype.elem == VecElemType::I64) {
            std::string ones = format("%s(-1)", arm_dup_intrin_name(dtype).c_str());
            return vector_bitwise_expr(dtype, "eor", arg, ones.c_str());
        }
        if (dtype.is_int()) { return format("%s(%s)", arm_intrin_name("mvn", dtype).c_str(), arg); }

        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        return format("vreinterpret%s_%s_u8(vmvn%s_u8(vreinterpret%s_u8_%s(%s)))", q, arm_vec_suffix(dtype.elem), q, q,
                      arm_vec_suffix(dtype.elem), arg);
    }

    std::string variable_shift_expr(VecDataType dtype, ArithBinaryOp op, const char *left, const char *right) const {
        if (!dtype.is_int()) {
            unsupported("Do not support %s of %s", show_arith_binary_op(op), show_vec_dtype(dtype));
        }
        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        const char *signed_suffix = arm_vec_suffix(dtype.elem);
        const char *unsigned_suffix = arm_unsigned_vec_suffix(dtype.elem);
        switch (op) {
        case ArithBinaryOp::ShiftLeftLogical:
            return format("%s(%s, %s)", arm_intrin_name("shl", dtype).c_str(), left, right);
        case ArithBinaryOp::ShiftRightArith:
            return format("%s(%s, %s(%s))", arm_intrin_name("shl", dtype).c_str(), left,
                          arm_intrin_name("neg", dtype).c_str(), right);
        case ArithBinaryOp::ShiftRightLogical:
            return format("vreinterpret%s_%s_%s(%s(vreinterpret%s_%s_%s(%s), %s(%s)))", q, signed_suffix,
                          unsigned_suffix, arm_intrin_name("shl", dtype, true).c_str(), q, unsigned_suffix,
                          signed_suffix, left, arm_intrin_name("neg", dtype).c_str(), right);
        default: break;
        }
        SIMJIT_UNREACHABLE();
    }

    std::string logical_shift_right_imm_expr(VecDataType dtype, const char *left, uint64_t amount) const {
        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        const char *signed_suffix = arm_vec_suffix(dtype.elem);
        const char *unsigned_suffix = arm_unsigned_vec_suffix(dtype.elem);
        return format("vreinterpret%s_%s_%s(%s(vreinterpret%s_%s_%s(%s), %llu))", q, signed_suffix, unsigned_suffix,
                      arm_shift_imm_intrin_name("shr", dtype, true).c_str(), q, unsigned_suffix, signed_suffix, left,
                      (unsigned long long)amount);
    }

    std::string immediate_rotate_expr(VecDataType dtype, ArithBinaryOp op, const char *left, uint64_t amount) const {
        if (!dtype.is_int()) {
            unsupported("Do not support %s of %s", show_arith_binary_op(op), show_vec_dtype(dtype));
        }
        if (amount == 0) { return left; }

        uint64_t complement = dtype.element_size_bits() - amount;
        if (op == ArithBinaryOp::RotateLeft) {
            std::string lhs = format("%s(%s, %llu)", arm_shift_imm_intrin_name("shl", dtype).c_str(), left,
                                     (unsigned long long)amount);
            std::string rhs = logical_shift_right_imm_expr(dtype, left, complement);
            return vector_bitwise_expr(dtype, "orr", lhs.c_str(), rhs.c_str());
        }
        if (op == ArithBinaryOp::RotateRight) {
            std::string lhs = logical_shift_right_imm_expr(dtype, left, amount);
            std::string rhs = format("%s(%s, %llu)", arm_shift_imm_intrin_name("shl", dtype).c_str(), left,
                                     (unsigned long long)complement);
            return vector_bitwise_expr(dtype, "orr", lhs.c_str(), rhs.c_str());
        }

        SIMJIT_UNREACHABLE();
    }

    std::string variable_rotate_expr(VecDataType dtype, ArithBinaryOp op, const char *left, const char *right) const {
        if (!dtype.is_int()) {
            unsupported("Do not support %s of %s", show_arith_binary_op(op), show_vec_dtype(dtype));
        }

        std::string bits = format("%s(%zu)", arm_dup_intrin_name(dtype).c_str(), dtype.element_size_bits());
        std::string complement = format("%s(%s, %s)", arm_intrin_name("sub", dtype).c_str(), bits.c_str(), right);
        if (op == ArithBinaryOp::RotateLeft) {
            std::string lhs = variable_shift_expr(dtype, ArithBinaryOp::ShiftLeftLogical, left, right);
            std::string rhs = variable_shift_expr(dtype, ArithBinaryOp::ShiftRightLogical, left, complement.c_str());
            return vector_bitwise_expr(dtype, "orr", lhs.c_str(), rhs.c_str());
        }
        if (op == ArithBinaryOp::RotateRight) {
            std::string lhs = variable_shift_expr(dtype, ArithBinaryOp::ShiftRightLogical, left, right);
            std::string rhs = variable_shift_expr(dtype, ArithBinaryOp::ShiftLeftLogical, left, complement.c_str());
            return vector_bitwise_expr(dtype, "orr", lhs.c_str(), rhs.c_str());
        }

        SIMJIT_UNREACHABLE();
    }

    std::string int64_minmax_expr(VecDataType dtype, ArithBinaryOp op, const char *left, const char *right) const {
        SIMJIT_ASSERT(dtype.elem == VecElemType::I64);

        switch (op) {
        case ArithBinaryOp::Min:
            return format("%s(%s(%s, %s), %s, %s)", arm_intrin_name("bsl", dtype).c_str(),
                          arm_intrin_name("clt", dtype).c_str(), left, right, left, right);
        case ArithBinaryOp::Max:
            return format("%s(%s(%s, %s), %s, %s)", arm_intrin_name("bsl", dtype).c_str(),
                          arm_intrin_name("cgt", dtype).c_str(), left, right, left, right);
        case ArithBinaryOp::UMin:
            return format("%s(%s(vreinterpretq_u64_s64(%s), vreinterpretq_u64_s64(%s)), %s, %s)",
                          arm_intrin_name("bsl", dtype).c_str(), arm_intrin_name("clt", dtype, true).c_str(), left,
                          right, left, right);
        case ArithBinaryOp::UMax:
            return format("%s(%s(vreinterpretq_u64_s64(%s), vreinterpretq_u64_s64(%s)), %s, %s)",
                          arm_intrin_name("bsl", dtype).c_str(), arm_intrin_name("cgt", dtype, true).c_str(), left,
                          right, left, right);
        default: break;
        }
        SIMJIT_UNREACHABLE();
    }

    static VecElemType half_width_elem(VecElemType elem) {
        switch (elem) {
        case VecElemType::I16: return VecElemType::I8;
        case VecElemType::I32: return VecElemType::I16;
        case VecElemType::I64: return VecElemType::I32;
        case VecElemType::I8:
        case VecElemType::F32:
        case VecElemType::F64: break;
        }
        messed_up("No half-width element for %s", arm_vec_suffix(elem));
        SIMJIT_UNREACHABLE();
    }

    std::string low_half_expr(VecDataType dtype, const char *arg) const {
        if (dtype.size_bytes() == 8) { return arg; }
        return format("vget_low_%s(%s)", arm_vec_suffix(dtype.elem), arg);
    }

    std::string unsigned_reinterpret_expr(VecDataType dtype, const char *arg) const {
        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        return format("vreinterpret%s_%s_%s(%s)", q, arm_unsigned_vec_suffix(dtype.elem), arm_vec_suffix(dtype.elem),
                      arg);
    }

    std::string signed_reinterpret_from_unsigned_expr(VecDataType dtype, const char *arg) const {
        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        return format("vreinterpret%s_%s_%s(%s)", q, arm_vec_suffix(dtype.elem), arm_unsigned_vec_suffix(dtype.elem),
                      arg);
    }

    std::string mul64_arg_expr(const Step *arg, VecDataType dst, bool is_unsigned, std::string expr) const {
        VecDataType arg_dtype = arg->dtype.as_vec();
        VecElemType half_elem = half_width_elem(dst.elem);

        if (arg_dtype.elem == dst.elem && dst.elem == VecElemType::I64) {
            if (is_unsigned) { return format("vmovn_u64(vreinterpretq_u64_s64(%s))", expr.c_str()); }
            return format("vmovn_s64(%s)", expr.c_str());
        }

        if (arg_dtype.elem != half_elem) {
            unsupported("Do not support %s input for %s widened multiply", show_vec_dtype(arg_dtype),
                        show_vec_dtype(dst));
        }

        expr = low_half_expr(arg_dtype, expr.c_str());
        if (is_unsigned) {
            VecDataType low_dtype{VecSize::X8, half_elem};
            if (half_elem == VecElemType::I16) { low_dtype.size = VecSize::X4; }
            if (half_elem == VecElemType::I32) { low_dtype.size = VecSize::X2; }
            return unsigned_reinterpret_expr(low_dtype, expr.c_str());
        }
        return expr;
    }

    std::string mul64_arg_expr(const Step *arg, VecDataType dst, bool is_unsigned) const {
        return mul64_arg_expr(arg, dst, is_unsigned, show(arg));
    }

    std::string mul64_expr(VecDataType dtype, const Step *left, const char *left_expr, const Step *right,
                           const char *right_expr, bool is_unsigned) const {
        VecElemType half_elem = half_width_elem(dtype.elem);
        std::string lhs = mul64_arg_expr(left, dtype, is_unsigned, left_expr);
        std::string rhs = mul64_arg_expr(right, dtype, is_unsigned, right_expr);
        const char *suffix = is_unsigned ? arm_unsigned_vec_suffix(half_elem) : arm_vec_suffix(half_elem);
        std::string expr = format("vmull_%s(%s, %s)", suffix, lhs.c_str(), rhs.c_str());
        if (is_unsigned) { return signed_reinterpret_from_unsigned_expr(dtype, expr.c_str()); }
        return expr;
    }

    std::string mul64_expr(VecDataType dtype, const Step *left, const Step *right, bool is_unsigned) const {
        return mul64_expr(dtype, left, show(left), right, show(right), is_unsigned);
    }

    std::string lzcnt_i64_expr(const char *arg) const {
        return format(R"STR(([&]() {
int32x4_t clz32 = vclzq_s32(vreinterpretq_s32_s64(%s));
int32x4_t high32 = vreinterpretq_s32_s64(vshrq_n_s64(vreinterpretq_s64_s32(clz32), 32));
uint32x4_t high_zero = vceqq_s32(high32, vdupq_n_s32(32));
return vreinterpretq_s64_s32(vbslq_s32(high_zero, vaddq_s32(high32, clz32), high32));
}()))STR",
                      arg);
    }

    std::string popcount_expr(VecDataType dtype, const char *arg) const {
        std::string bytes = format("vcntq_u8(vreinterpretq_u8_%s(%s))", arm_vec_suffix(dtype.elem), arg);
        switch (dtype.elem) {
        case VecElemType::I8: return format("vreinterpretq_s8_u8(%s)", bytes.c_str());
        case VecElemType::I16: return format("vreinterpretq_s16_u16(vpaddlq_u8(%s))", bytes.c_str());
        case VecElemType::I32: return format("vreinterpretq_s32_u32(vpaddlq_u16(vpaddlq_u8(%s)))", bytes.c_str());
        case VecElemType::I64:
            return format("vreinterpretq_s64_u64(vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(%s))))", bytes.c_str());
        case VecElemType::F32:
        case VecElemType::F64: break;
        }
        SIMJIT_UNREACHABLE();
    }

    std::optional<std::string> binary_expr(const Step *step, const ArithBinData &data, const char *left,
                                           const char *right) const {
        VecDataType dtype = step->dtype.as_vec();

        switch (data.op) {
        case ArithBinaryOp::Add: return format("%s(%s, %s)", arm_intrin_name("add", dtype).c_str(), left, right);
        case ArithBinaryOp::Sub: return format("%s(%s, %s)", arm_intrin_name("sub", dtype).c_str(), left, right);
        case ArithBinaryOp::Mul:
            if (dtype.elem == VecElemType::I64) { unsupported("Do not support i64 mul"); }
            return format("%s(%s, %s)", arm_intrin_name("mul", dtype).c_str(), left, right);
        case ArithBinaryOp::Min:
            if (dtype.elem == VecElemType::I64) { return int64_minmax_expr(dtype, data.op, left, right); }
            if (dtype.is_float()) {
                return format("%s(%s, %s)", arm_float_intrin_name("minnm", dtype).c_str(), left, right);
            }
            return format("%s(%s, %s)", arm_intrin_name("min", dtype).c_str(), left, right);
        case ArithBinaryOp::Max:
            if (dtype.elem == VecElemType::I64) { return int64_minmax_expr(dtype, data.op, left, right); }
            if (dtype.is_float()) {
                return format("%s(%s, %s)", arm_float_intrin_name("maxnm", dtype).c_str(), left, right);
            }
            return format("%s(%s, %s)", arm_intrin_name("max", dtype).c_str(), left, right);
        case ArithBinaryOp::UMin:
            if (!dtype.is_int()) { return std::nullopt; }
            if (dtype.elem == VecElemType::I64) { return int64_minmax_expr(dtype, data.op, left, right); }
            return format("%s(%s, %s)", arm_intrin_name("min", dtype, true).c_str(), left, right);
        case ArithBinaryOp::UMax:
            if (!dtype.is_int()) { return std::nullopt; }
            if (dtype.elem == VecElemType::I64) { return int64_minmax_expr(dtype, data.op, left, right); }
            return format("%s(%s, %s)", arm_intrin_name("max", dtype, true).c_str(), left, right);
        case ArithBinaryOp::And: return vector_bitwise_expr(dtype, "and", left, right);
        case ArithBinaryOp::Or: return vector_bitwise_expr(dtype, "orr", left, right);
        case ArithBinaryOp::Xor: return vector_bitwise_expr(dtype, "eor", left, right);
        case ArithBinaryOp::AndNot: return vector_bitwise_expr(dtype, "bic", right, left);
        case ArithBinaryOp::ShiftLeftLogical:
        case ArithBinaryOp::ShiftRightLogical:
        case ArithBinaryOp::ShiftRightArith:
            if (!data.right->is(StepKind::Const)) { return variable_shift_expr(dtype, data.op, left, right); }
            {
                uint64_t amount = data.right->step_data<StepKind::Const>().as_unsigned();
                amount &= dtype.element_size_bits() - 1;
                if (amount == 0) { return left; }
                if (data.op == ArithBinaryOp::ShiftLeftLogical) {
                    return format("%s(%s, %llu)", arm_shift_imm_intrin_name("shl", dtype).c_str(), left,
                                  (unsigned long long)amount);
                }
                bool is_unsigned = data.op == ArithBinaryOp::ShiftRightLogical;
                if (is_unsigned && dtype.is_int()) { return logical_shift_right_imm_expr(dtype, left, amount); }
                return format("%s(%s, %llu)", arm_shift_imm_intrin_name("shr", dtype, is_unsigned).c_str(), left,
                              (unsigned long long)amount);
            }
        case ArithBinaryOp::Div:
            if (!dtype.is_float()) { unsupported("Do not support int div"); }
            return format("%s(%s, %s)", arm_intrin_name("div", dtype).c_str(), left, right);
        case ArithBinaryOp::Mul64SE: return mul64_expr(dtype, data.left, left, data.right, right, false);
        case ArithBinaryOp::Mul64ZE: return mul64_expr(dtype, data.left, left, data.right, right, true);
        case ArithBinaryOp::UDiv:
        case ArithBinaryOp::Mod:
        case ArithBinaryOp::UMod: unsupported("Do not support int div");
        case ArithBinaryOp::RotateLeft:
        case ArithBinaryOp::RotateRight:
            if (!data.right->is(StepKind::Const)) { return variable_rotate_expr(dtype, data.op, left, right); }
            {
                uint64_t amount = data.right->step_data<StepKind::Const>().as_unsigned();
                amount &= dtype.element_size_bits() - 1;
                return immediate_rotate_expr(dtype, data.op, left, amount);
            }
        }
        SIMJIT_UNREACHABLE();
    }

    std::optional<std::string> binary_expr(const Step *step, const ArithBinData &data) const {
        return binary_expr(step, data, show(data.left), show(data.right));
    }

    std::optional<std::string> scalar_refined_reciprocal_to_cpp(const Step *step, const ArithUnaryData &data) {
        if (data.op != ArithUnaryOp::Rcp && data.op != ArithUnaryOp::Rsqrt) { return std::nullopt; }
        ScalarDataType dtype = step->dtype.as_scalar();
        if (dtype != ScalarDataType::F32 && dtype != ScalarDataType::F64) {
            unsupported("Do not support %s of %s", show_arith_unary_op(data.op), show_scalar_dtype(dtype));
        }

        std::string arg = show_scalar_operand(data.arg);
        std::string result = show(step);
        std::string tmp = format("%s_tmp", result.c_str());
        const char *cpp_type = scalar_dtype_to_cpp(dtype);

        if (data.op == ArithUnaryOp::Rcp) {
            const char *est = dtype == ScalarDataType::F32 ? "vrecpes_f32" : "vrecped_f64";
            const char *step_intrin = dtype == ScalarDataType::F32 ? "vrecpss_f32" : "vrecpsd_f64";
            std::string code = cpp_var_decl(step) + format("%s(%s)", est, arg.c_str());
            code += format(";\n%s %s = %s(%s, %s)", cpp_type, tmp.c_str(), step_intrin, arg.c_str(), result.c_str());
            code += format(";\n%s = %s * %s", result.c_str(), result.c_str(), tmp.c_str());
            code += format(";\n%s = %s(%s, %s)", tmp.c_str(), step_intrin, arg.c_str(), result.c_str());
            code += format(";\n%s = %s * %s", result.c_str(), result.c_str(), tmp.c_str());
            if (dtype == ScalarDataType::F64) {
                code += format(";\n%s = %s(%s, %s)", tmp.c_str(), step_intrin, arg.c_str(), result.c_str());
                code += format(";\n%s = %s * %s", result.c_str(), result.c_str(), tmp.c_str());
            }
            return code;
        }

        const char *est = dtype == ScalarDataType::F32 ? "vrsqrtes_f32" : "vrsqrted_f64";
        const char *step_intrin = dtype == ScalarDataType::F32 ? "vrsqrtss_f32" : "vrsqrtsd_f64";
        std::string code = cpp_var_decl(step) + format("%s(%s)", est, arg.c_str());
        code += format(";\n%s %s = %s * %s", cpp_type, tmp.c_str(), result.c_str(), result.c_str());
        code += format(";\n%s = %s(%s, %s)", tmp.c_str(), step_intrin, arg.c_str(), tmp.c_str());
        code += format(";\n%s = %s * %s", result.c_str(), tmp.c_str(), result.c_str());
        code += format(";\n%s = %s * %s", tmp.c_str(), result.c_str(), result.c_str());
        code += format(";\n%s = %s(%s, %s)", tmp.c_str(), step_intrin, arg.c_str(), tmp.c_str());
        code += format(";\n%s = %s * %s", result.c_str(), tmp.c_str(), result.c_str());
        if (dtype == ScalarDataType::F64) {
            code += format(";\n%s = %s * %s", tmp.c_str(), result.c_str(), result.c_str());
            code += format(";\n%s = %s(%s, %s)", tmp.c_str(), step_intrin, arg.c_str(), tmp.c_str());
            code += format(";\n%s = %s * %s", result.c_str(), tmp.c_str(), result.c_str());
        }
        return code;
    }

    std::optional<std::string> vector_refined_reciprocal_to_cpp(const Step *step, const ArithUnaryData &data) const {
        if (data.op != ArithUnaryOp::Rcp && data.op != ArithUnaryOp::Rsqrt) { return std::nullopt; }
        VecDataType dtype = step->dtype.as_vec();
        if (!dtype.is_float()) {
            unsupported("Do not support %s of %s", show_arith_unary_op(data.op), show_vec_dtype(dtype));
        }

        std::string arg = show(data.arg);
        std::string result = show(step);
        std::string tmp = format("%s_tmp", result.c_str());
        const char *cpp_type = cpp_dtype(step->dtype);
        std::string mul = arm_intrin_name("mul", dtype);

        if (data.op == ArithUnaryOp::Rcp) {
            std::string recps = arm_intrin_name("recps", dtype);
            std::string code =
                cpp_var_decl(step) + format("%s(%s)", arm_intrin_name("recpe", dtype).c_str(), arg.c_str());
            code += format(";\n%s %s = %s(%s, %s)", cpp_type, tmp.c_str(), recps.c_str(), arg.c_str(), result.c_str());
            code += format(";\n%s = %s(%s, %s)", result.c_str(), mul.c_str(), result.c_str(), tmp.c_str());
            code += format(";\n%s = %s(%s, %s)", tmp.c_str(), recps.c_str(), arg.c_str(), result.c_str());
            code += format(";\n%s = %s(%s, %s)", result.c_str(), mul.c_str(), result.c_str(), tmp.c_str());
            if (dtype.elem == VecElemType::F64) {
                code += format(";\n%s = %s(%s, %s)", tmp.c_str(), recps.c_str(), arg.c_str(), result.c_str());
                code += format(";\n%s = %s(%s, %s)", result.c_str(), mul.c_str(), result.c_str(), tmp.c_str());
            }
            return code;
        }

        std::string rsqrts = arm_intrin_name("rsqrts", dtype);
        std::string code = cpp_var_decl(step) + format("%s(%s)", arm_intrin_name("rsqrte", dtype).c_str(), arg.c_str());
        code += format(";\n%s %s = %s(%s, %s)", cpp_type, tmp.c_str(), mul.c_str(), result.c_str(), result.c_str());
        code += format(";\n%s = %s(%s, %s)", tmp.c_str(), rsqrts.c_str(), arg.c_str(), tmp.c_str());
        code += format(";\n%s = %s(%s, %s)", result.c_str(), mul.c_str(), tmp.c_str(), result.c_str());
        code += format(";\n%s = %s(%s, %s)", tmp.c_str(), mul.c_str(), result.c_str(), result.c_str());
        code += format(";\n%s = %s(%s, %s)", tmp.c_str(), rsqrts.c_str(), arg.c_str(), tmp.c_str());
        code += format(";\n%s = %s(%s, %s)", result.c_str(), mul.c_str(), tmp.c_str(), result.c_str());
        if (dtype.elem == VecElemType::F64) {
            code += format(";\n%s = %s(%s, %s)", tmp.c_str(), mul.c_str(), result.c_str(), result.c_str());
            code += format(";\n%s = %s(%s, %s)", tmp.c_str(), rsqrts.c_str(), arg.c_str(), tmp.c_str());
            code += format(";\n%s = %s(%s, %s)", result.c_str(), mul.c_str(), tmp.c_str(), result.c_str());
        }
        return code;
    }

    std::optional<std::string> unary_expr(const Step *step, const ArithUnaryData &data) const {
        VecDataType dtype = step->dtype.as_vec();
        std::string arg = show(data.arg);

        switch (data.op) {
        case ArithUnaryOp::Not: return vector_not_expr(dtype, arg.c_str());
        case ArithUnaryOp::Negate: return format("%s(%s)", arm_intrin_name("neg", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::Abs: return format("%s(%s)", arm_intrin_name("abs", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::RoundNearest:
            if (!dtype.is_float()) { return std::nullopt; }
            return format("%s(%s)", arm_intrin_name("rndn", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::RoundDown:
            if (!dtype.is_float()) { return std::nullopt; }
            return format("%s(%s)", arm_intrin_name("rndm", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::RoundUp:
            if (!dtype.is_float()) { return std::nullopt; }
            return format("%s(%s)", arm_intrin_name("rndp", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::RoundTruncate:
            if (!dtype.is_float()) { return std::nullopt; }
            return format("%s(%s)", arm_intrin_name("rnd", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::Sqrt:
            if (!dtype.is_float()) { return std::nullopt; }
            return format("%s(%s)", arm_intrin_name("sqrt", dtype).c_str(), arg.c_str());
        case ArithUnaryOp::Rcp:
        case ArithUnaryOp::Rsqrt: return std::nullopt;
        case ArithUnaryOp::Tzcnt: messed_up("tzcnt should have been rewritten before C++ emission");
        case ArithUnaryOp::Lzcnt:
            if (dtype.elem == VecElemType::I64) { return lzcnt_i64_expr(arg.c_str()); }
            if (dtype.is_int()) { return format("%s(%s)", arm_intrin_name("clz", dtype).c_str(), arg.c_str()); }
            unsupported("Do not support %s of %s", show_arith_unary_op(data.op), show_vec_dtype(dtype));
        case ArithUnaryOp::Popcount:
            if (dtype.is_int()) { return popcount_expr(dtype, arg.c_str()); }
            unsupported("Do not support %s of %s", show_arith_unary_op(data.op), show_vec_dtype(dtype));
        }
        SIMJIT_UNREACHABLE();
    }

    std::optional<std::string> compare_expr(const Step *step, const CmpData &data) const {
        if (!data.left->dtype.is_vec()) { return std::nullopt; }
        VecDataType dtype = data.left->dtype.as_vec();
        std::string left = show(data.left);
        std::string right = show(data.right);
        bool is_unsigned = data.is_unsigned && dtype.is_int();

        switch (data.op) {
        case CmpOp::Equal:
            return format("%s(%s, %s)", arm_intrin_name("ceq", dtype, is_unsigned).c_str(), left.c_str(),
                          right.c_str());
        case CmpOp::NotEqual: {
            std::string equal =
                format("%s(%s, %s)", arm_intrin_name("ceq", dtype, is_unsigned).c_str(), left.c_str(), right.c_str());
            return mask_not_expr(step->dtype.as_mask(), equal.c_str());
        }
        case CmpOp::Greater:
            return format("%s(%s, %s)", arm_intrin_name("cgt", dtype, is_unsigned).c_str(), left.c_str(),
                          right.c_str());
        case CmpOp::Less:
            return format("%s(%s, %s)", arm_intrin_name("clt", dtype, is_unsigned).c_str(), left.c_str(),
                          right.c_str());
        case CmpOp::GreaterEqual:
            return format("%s(%s, %s)", arm_intrin_name("cge", dtype, is_unsigned).c_str(), left.c_str(),
                          right.c_str());
        case CmpOp::LessEqual:
            return format("%s(%s, %s)", arm_intrin_name("cle", dtype, is_unsigned).c_str(), left.c_str(),
                          right.c_str());
        }
        SIMJIT_UNREACHABLE();
    }

    std::string vec_const_to_cpp(const Step *step, const VecConstData &data) const {
        VecDataType dtype = step->dtype.as_vec();
        size_t idx = step->id;
        std::string result = format("static const %s const_mem_%zu[] = {", scalar_dtype_to_cpp(dtype.to_scalar()), idx);
        for (size_t i = 0; i < dtype.nelems(); ++i) {
            switch (dtype.elem) {
            case VecElemType::I8: format_to(result, "%d, ", int(((int8_t *)data.mem)[i])); break;
            case VecElemType::I16: format_to(result, "%d, ", int(((int16_t *)data.mem)[i])); break;
            case VecElemType::I32: format_to(result, "%d, ", int(((int32_t *)data.mem)[i])); break;
            case VecElemType::I64: format_to(result, "%lldLL, ", (long long)((int64_t *)data.mem)[i]); break;
            case VecElemType::F32:
            case VecElemType::F64: messed_up("unexpected float type in vecconst");
            }
        }
        result += "};\n";
        result += cpp_var_decl(step) + format("%s(const_mem_%zu)", arm_intrin_name("ld1", dtype).c_str(), idx);
        return result;
    }

    std::string bitcast_to_cpp(const Step *step, const Step *arg) const {
        if (!step->dtype.is_vec() || !arg->dtype.is_vec()) {
            messed_up("ARM C++ bitcast expects vector dtypes, got %s from %s", show_dtype(step->dtype),
                      show_dtype(arg->dtype));
            SIMJIT_UNREACHABLE();
        }
        VecDataType to = step->dtype.as_vec();
        VecDataType from = arg->dtype.as_vec();
        const char *q = to.size_bytes() == 16 ? "q" : "";
        return cpp_var_decl(step) +
               format("vreinterpret%s_%s_%s(%s)", q, arm_vec_suffix(to.elem), arm_vec_suffix(from.elem), show(arg));
    }

    std::string low_half_expr(const Step *arg) const {
        VecDataType dtype = arg->dtype.as_vec();
        if (dtype.size_bytes() != 16) { return show(arg); }
        return format("vget_low_%s(%s)", arm_vec_suffix(dtype.elem), show(arg));
    }

    std::string int_cast_to_cpp(const Step *step, const IntCastData &data) const {
        if (!step->dtype.is_vec() || !data.arg->dtype.is_vec()) {
            return unexpected_step_dtype(step, "vector int-cast");
        }
        VecDataType to = step->dtype.as_vec();
        VecDataType from = data.arg->dtype.as_vec();
        if (!to.is_int() || !from.is_int()) {
            messed_up("ARM C++ int-cast expects integer vectors, got %s from %s", show_vec_dtype(to),
                      show_vec_dtype(from));
        }

        if (to.element_size_bytes() == from.element_size_bytes()) { return bitcast_to_cpp(step, data.arg); }

        if (to.element_size_bytes() > from.element_size_bytes()) {
            bool is_unsigned = data.kind == IntCastKind::Zext;
            return cpp_var_decl(step) + format("%s(%s)", arm_lane_width_intrin_name("movl", from, is_unsigned).c_str(),
                                               low_half_expr(data.arg).c_str());
        }

        if (data.kind == IntCastKind::Trunc) {
            return cpp_var_decl(step) +
                   format("%s(%s)", arm_lane_width_intrin_name("movn", from).c_str(), show(data.arg));
        }
        unsupported("Do not support ARM C++ %s int-cast from %s to %s", show_int_cast_kind(data.kind),
                    show_vec_dtype(from), show_vec_dtype(to));
        SIMJIT_UNREACHABLE();
    }

    std::string float_cast_to_cpp(const Step *step, const FloatCastData &data) const {
        if (!step->dtype.is_vec() || !data.arg->dtype.is_vec()) {
            return unexpected_step_dtype(step, "vector float-cast");
        }
        VecDataType to = step->dtype.as_vec();
        VecDataType from = data.arg->dtype.as_vec();
        std::string arg = show(data.arg);

        if (to.is_float() && from.is_float()) {
            if (to.elem == from.elem) { return bitcast_to_cpp(step, data.arg); }
            if (to.elem == VecElemType::F64 && from.elem == VecElemType::F32) {
                return cpp_var_decl(step) + format("vcvt_f64_f32(%s)", low_half_expr(data.arg).c_str());
            }
            if (to.elem == VecElemType::F32 && from.elem == VecElemType::F64) {
                return cpp_var_decl(step) + format("vcvt_f32_f64(%s)", arg.c_str());
            }
        }

        if (to.is_float() && from.is_int() && to.element_size_bytes() == from.element_size_bytes()) {
            const char *q = to.size_bytes() == 16 ? "q" : "";
            const char *from_suffix = data.is_unsigned ? arm_unsigned_vec_suffix(from.elem) : arm_vec_suffix(from.elem);
            return cpp_var_decl(step) +
                   format("vcvt%s_%s_%s(%s)", q, arm_vec_suffix(to.elem), from_suffix, arg.c_str());
        }
        if (to.is_int() && from.is_float() && to.element_size_bytes() == from.element_size_bytes()) {
            const char *to_suffix = data.is_unsigned ? arm_unsigned_vec_suffix(to.elem) : arm_vec_suffix(to.elem);
            const char *q = to.size_bytes() == 16 ? "q" : "";
            return cpp_var_decl(step) +
                   format("vcvt%s_%s_%s(%s)", q, to_suffix, arm_vec_suffix(from.elem), arg.c_str());
        }
        unsupported("Do not support ARM C++ float-cast from %s to %s", show_vec_dtype(from), show_vec_dtype(to));
        SIMJIT_UNREACHABLE();
    }

    std::string widen_half_to_cpp(const Step *step, const HalfCast &data, bool high_half) const {
        if (!step->dtype.is_vec() || !data.arg->dtype.is_vec()) { return unexpected_step_dtype(step, "vector widen"); }
        VecDataType to = step->dtype.as_vec();
        VecDataType from = data.arg->dtype.as_vec();
        if (to.is_float()) {
            if (high_half) {
                return cpp_var_decl(step) + format("vcvt_high_%s_%s(%s)", arm_vec_suffix(to.elem),
                                                   arm_vec_suffix(from.elem), show(data.arg));
            }
            return cpp_var_decl(step) + format("vcvt_%s_%s(%s)", arm_vec_suffix(to.elem), arm_vec_suffix(from.elem),
                                               low_half_expr(data.arg).c_str());
        }

        bool is_unsigned = data.is_unsigned;
        if (high_half) {
            return cpp_var_decl(step) +
                   format("%s(%s)", arm_lane_width_intrin_name("movl_high", from, is_unsigned).c_str(), show(data.arg));
        }
        return cpp_var_decl(step) + format("%s(%s)", arm_lane_width_intrin_name("movl", from, is_unsigned).c_str(),
                                           low_half_expr(data.arg).c_str());
    }

    std::string narrow_combine_to_cpp(const Step *step, const VecNarrowCombineData &data) const {
        if (!step->dtype.is_vec() || !data.low->dtype.is_vec() || !data.high->dtype.is_vec()) {
            return unexpected_step_dtype(step, "vector narrow-combine");
        }
        VecDataType from = data.low->dtype.as_vec();
        return cpp_var_decl(step) + format("%s(%s(%s), %s)", arm_lane_width_intrin_name("movn_high", from).c_str(),
                                           arm_lane_width_intrin_name("movn", from).c_str(), show(data.low),
                                           show(data.high));
    }

    std::string float_narrow_combine_to_cpp(const Step *step, const VecFloatNarrowCombineData &data) const {
        if (!step->dtype.is_vec() || !data.low->dtype.is_vec() || !data.high->dtype.is_vec()) {
            return unexpected_step_dtype(step, "vector float narrow-combine");
        }
        VecDataType to = step->dtype.as_vec();
        VecDataType from = data.low->dtype.as_vec();
        return cpp_var_decl(step) + format("vcvt_high_%s_%s(vcvt_%s_%s(%s), %s)", arm_vec_suffix(to.elem),
                                           arm_vec_suffix(from.elem), arm_vec_suffix(to.elem),
                                           arm_vec_suffix(from.elem), show(data.low), show(data.high));
    }

    std::string pack_to_cpp(const PackData &data, const Step *step) {
        if (!step->dtype.is_vec() || !data.arg->dtype.is_vec() || !data.cond->dtype.is_mask()) {
            return unexpected_step_dtype(step, "vector pack with vector argument and mask condition");
        }
        VecDataType dtype = step->dtype.as_vec();
        size_t id = step->id;
        std::string result = format("{\n"
                                    "size_t base%zu = acc%zu;\n"
                                    "%s pack_values_%zu[%zu];\n",
                                    id, func->accs.index(data.acc), //
                                    scalar_dtype_to_cpp(dtype.to_scalar()), id, dtype.nelems());
        format_to(result, "%s(pack_values_%zu, %s);\n", arm_intrin_name("st1", dtype).c_str(), id, show(data.arg));
        for (size_t lane = 0; lane < dtype.nelems(); ++lane) {
            format_to(result, "if (%s) arg%zu[acc%zu++] = pack_values_%zu[%zu];\n", lane_expr(data.cond, lane).c_str(),
                      data.dst, func->accs.index(data.acc), id, lane);
        }
        result += "}";
        return result;
    }

    std::string vec_lane_expr(VecDataType dtype, const char *arg, size_t lane, bool is_unsigned = false) const {
        return format("%s(%s, %zu)", lane_get_intrin(dtype, is_unsigned).c_str(), arg, lane);
    }

    std::string pairwise_reduce_lane_expr(VecDataType dtype, const char *op, const char *arg,
                                          bool is_unsigned = false) const {
        std::string reduced = format("%s(%s, %s)", arm_pairwise_intrin_name(op, dtype, is_unsigned).c_str(), arg, arg);
        return vec_lane_expr(dtype, reduced.c_str(), 0, is_unsigned);
    }

    std::optional<std::string> direct_vec_reduce_expr(VecDataType dtype, ArithBinaryOp op, const char *arg) const {
        switch (op) {
        case ArithBinaryOp::Add:
            if (dtype.elem == VecElemType::I64 || dtype.elem == VecElemType::F64) {
                return pairwise_reduce_lane_expr(dtype, "add", arg);
            }
            return format("%s(%s)", arm_reduce_intrin_name("addv", dtype).c_str(), arg);
        case ArithBinaryOp::Min:
            if (dtype.elem == VecElemType::I64) { return std::nullopt; }
            if (dtype.elem == VecElemType::F64) { return pairwise_reduce_lane_expr(dtype, "minnm", arg); }
            if (dtype.elem == VecElemType::F32) {
                return format("%s(%s)", arm_float_intrin_name("minnmv", dtype).c_str(), arg);
            }
            return format("%s(%s)", arm_reduce_intrin_name("minv", dtype).c_str(), arg);
        case ArithBinaryOp::Max:
            if (dtype.elem == VecElemType::I64) { return std::nullopt; }
            if (dtype.elem == VecElemType::F64) { return pairwise_reduce_lane_expr(dtype, "maxnm", arg); }
            if (dtype.elem == VecElemType::F32) {
                return format("%s(%s)", arm_float_intrin_name("maxnmv", dtype).c_str(), arg);
            }
            return format("%s(%s)", arm_reduce_intrin_name("maxv", dtype).c_str(), arg);
        case ArithBinaryOp::UMin:
            if (!dtype.is_int()) { return std::nullopt; }
            if (dtype.elem == VecElemType::I64) { return std::nullopt; }
            return format("%s(%s)", arm_reduce_intrin_name("minv", dtype, true).c_str(),
                          unsigned_reinterpret_expr(dtype, arg).c_str());
        case ArithBinaryOp::UMax:
            if (!dtype.is_int()) { return std::nullopt; }
            if (dtype.elem == VecElemType::I64) { return std::nullopt; }
            return format("%s(%s)", arm_reduce_intrin_name("maxv", dtype, true).c_str(),
                          unsigned_reinterpret_expr(dtype, arg).c_str());
        default: return std::nullopt;
        }
    }

    std::optional<std::string> vector_fold_reduce_op_expr(VecDataType dtype, ArithBinaryOp op, const char *left,
                                                          const char *right) const {
        switch (op) {
        case ArithBinaryOp::Mul:
            if (dtype.elem == VecElemType::I64) { return std::nullopt; }
            return format("%s(%s, %s)", arm_intrin_name("mul", dtype).c_str(), left, right);
        case ArithBinaryOp::And:
            if (!dtype.is_int()) { return std::nullopt; }
            return vector_bitwise_expr(dtype, "and", left, right);
        case ArithBinaryOp::Or:
            if (!dtype.is_int()) { return std::nullopt; }
            return vector_bitwise_expr(dtype, "orr", left, right);
        case ArithBinaryOp::Xor:
            if (!dtype.is_int()) { return std::nullopt; }
            return vector_bitwise_expr(dtype, "eor", left, right);
        case ArithBinaryOp::Min:
        case ArithBinaryOp::Max:
        case ArithBinaryOp::UMin:
        case ArithBinaryOp::UMax:
            if (dtype.elem != VecElemType::I64) { return std::nullopt; }
            return int64_minmax_expr(dtype, op, left, right);
        default: return std::nullopt;
        }
    }

    static bool vector_fold_reduce_needs_shift_temp(VecDataType dtype, ArithBinaryOp op) {
        if (dtype.elem != VecElemType::I64) { return false; }
        return op == ArithBinaryOp::Min || op == ArithBinaryOp::Max || op == ArithBinaryOp::UMin ||
               op == ArithBinaryOp::UMax;
    }

    std::string scalar_i64_mul_reduce_expr(VecDataType dtype, const char *arg) const {
        SIMJIT_ASSERT(dtype.elem == VecElemType::I64);
        return format("(%s * %s)", vec_lane_expr(dtype, arg, 0).c_str(), vec_lane_expr(dtype, arg, 1).c_str());
    }

    std::string scalar_reduce_op_expr(ScalarDataType sdtype, ArithBinaryOp op, const char *left,
                                      const char *right) const {
        if (is_shift_or_rotate_op(op)) {
            unsupported("Do not support reduce %s of %s", show_arith_binary_op(op), show_scalar_dtype(sdtype));
        }
        return format_scalar_arith_binary_expr(sdtype, op, sdtype, sdtype, left, right, std::nullopt);
    }

    std::string scalar_fallback_reduce_to_cpp(const Step *step, VecDataType dtype, ArithBinaryOp op,
                                              const std::string &arg) const {
        ScalarDataType sdtype = dtype.to_scalar();
        size_t id = step->id;
        std::string values = format("reduce_values_%zu", id);
        std::string acc = format("reduce_acc_%zu", id);
        std::string result = format("%s %s;\n"
                                    "{\n"
                                    "alignas(16) %s %s[%zu];\n",
                                    cpp_dtype(step->dtype), show(step), //
                                    scalar_dtype_to_cpp(sdtype), values.c_str(), dtype.nelems());
        format_to(result, "%s(%s, %s);\n", arm_intrin_name("st1", dtype).c_str(), values.c_str(), arg.c_str());
        format_to(result, "%s %s = %s[0];\n", scalar_dtype_to_cpp(sdtype), acc.c_str(), values.c_str());
        for (size_t lane = 1; lane < dtype.nelems(); ++lane) {
            std::string right = format("%s[%zu]", values.c_str(), lane);
            format_to(result, "%s = %s;\n", acc.c_str(),
                      scalar_reduce_op_expr(sdtype, op, acc.c_str(), right.c_str()).c_str());
        }
        format_to(result,
                  "%s = %s;\n"
                  "}",
                  show(step), acc.c_str());
        return result;
    }

    std::string vector_fold_reduce_to_cpp(const Step *step, VecDataType dtype, ArithBinaryOp op,
                                          const std::string &arg) const {
        if (op == ArithBinaryOp::Mul && dtype.elem == VecElemType::I64) {
            return cpp_var_decl(step) + scalar_i64_mul_reduce_expr(dtype, arg.c_str());
        }

        size_t id = step->id;
        std::string result = format("%s %s;\n"
                                    "{\n"
                                    "%s reduce_vec_%zu = %s;\n",
                                    cpp_dtype(step->dtype), show(step), //
                                    arm_vec_dtype_cpp(dtype), id, arg.c_str());
        for (size_t shift = dtype.nelems() / 2; shift > 0; shift /= 2) {
            std::string shifted_expr =
                format("%s(reduce_vec_%zu, reduce_vec_%zu, %zu)", arm_intrin_name("ext", dtype).c_str(), id, id, shift);
            std::string shifted = shifted_expr;
            if (vector_fold_reduce_needs_shift_temp(dtype, op)) {
                shifted = format("reduce_shifted_%zu_%zu", id, shift);
                result +=
                    format("const %s %s = %s;\n", arm_vec_dtype_cpp(dtype), shifted.c_str(), shifted_expr.c_str());
            }
            std::string reduce_vec = format("reduce_vec_%zu", id);
            auto folded = vector_fold_reduce_op_expr(dtype, op, reduce_vec.c_str(), shifted.c_str());
            if (!folded.has_value()) { return scalar_fallback_reduce_to_cpp(step, dtype, op, arg); }
            format_to(result, "reduce_vec_%zu = %s;\n", id, folded->c_str());
        }
        std::string reduce_vec = format("reduce_vec_%zu", id);
        format_to(result,
                  "%s = %s;\n"
                  "}",
                  show(step), vec_lane_expr(dtype, reduce_vec.c_str(), 0).c_str());
        return result;
    }

    std::string vec_reduce_to_cpp(const Step *step, const ArithReduceData &data) const {
        if (!data.arg->dtype.is_vec()) { return unexpected_step_dtype(step, "vector reduce argument"); }
        VecDataType dtype = data.arg->dtype.as_vec();
        std::string arg = show(data.arg);
        auto direct = direct_vec_reduce_expr(dtype, data.op, arg.c_str());
        if (direct.has_value()) { return cpp_var_decl(step) + *direct; }
        return vector_fold_reduce_to_cpp(step, dtype, data.op, arg);
    }

    std::string fma_expr(VecDataType dtype, const FMAData &data, const char *x1, const char *x2, const char *x3) const {
        if (dtype.is_float()) {
            switch (data.kind) {
            case FmaKind::FMA: return format("%s(%s, %s, %s)", arm_intrin_name("fma", dtype).c_str(), x3, x1, x2);
            case FmaKind::FMS:
                return format("%s(%s(%s), %s, %s)", arm_intrin_name("fma", dtype).c_str(),
                              arm_intrin_name("neg", dtype).c_str(), x3, x1, x2);
            case FmaKind::FNMA: return format("%s(%s, %s, %s)", arm_intrin_name("fms", dtype).c_str(), x3, x1, x2);
            case FmaKind::FNMS:
                return format("%s(%s(%s), %s, %s)", arm_intrin_name("fms", dtype).c_str(),
                              arm_intrin_name("neg", dtype).c_str(), x3, x1, x2);
            }
        }
        if (dtype.elem == VecElemType::I64) { unsupported("Do not support i64 mul"); }
        switch (data.kind) {
        case FmaKind::FMA: return format("%s(%s, %s, %s)", arm_intrin_name("mla", dtype).c_str(), x3, x1, x2);
        case FmaKind::FMS:
            return format("%s(%s(%s), %s, %s)", arm_intrin_name("mla", dtype).c_str(),
                          arm_intrin_name("neg", dtype).c_str(), x3, x1, x2);
        case FmaKind::FNMA: return format("%s(%s, %s, %s)", arm_intrin_name("mls", dtype).c_str(), x3, x1, x2);
        case FmaKind::FNMS:
            return format("%s(%s(%s), %s, %s)", arm_intrin_name("mls", dtype).c_str(),
                          arm_intrin_name("neg", dtype).c_str(), x3, x1, x2);
        }
        SIMJIT_UNREACHABLE();
    }

    std::string fma_expr(VecDataType dtype, const FMAData &data) const {
        return fma_expr(dtype, data, show(data.x1), show(data.x2), show(data.x3));
    }

    std::string ternarylogic_to_cpp(const Step *step, const TernarylogicData &data) const {
        (void)step;
        (void)data;
        unsupported("Do not support ternarylogic");
        SIMJIT_UNREACHABLE();
    }

    std::string reinterpret_step_to_u8(const Step *step, bool want_q) const {
        if (!step->dtype.is_vec()) {
            messed_up("ARM C++ byte reinterpret expects vector dtype, got %s", show_dtype(step->dtype));
        }
        VecDataType dtype = step->dtype.as_vec();
        const char *suffix = arm_vec_suffix(dtype.elem);
        bool has_q = dtype.size_bytes() == 16;
        if (want_q == has_q) {
            const char *q = want_q ? "q" : "";
            return format("vreinterpret%s_u8_%s(%s)", q, suffix, show(step));
        }
        if (want_q) { return format("vcombine_u8(vreinterpret_u8_%s(%s), vdup_n_u8(0))", suffix, show(step)); }
        return format("vget_low_u8(vreinterpretq_u8_%s(%s))", suffix, show(step));
    }

    std::string reinterpret_u8_to_vec(VecDataType dtype, const char *expr) const {
        const char *q = dtype.size_bytes() == 16 ? "q" : "";
        return format("vreinterpret%s_%s_u8(%s)", q, arm_vec_suffix(dtype.elem), expr);
    }

    std::string vec_permute_to_cpp(const Step *step, const VecPermuteData &data) const {
        if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector permute"); }
        VecDataType dtype = step->dtype.as_vec();
        if (data.is_bit && data.permute == REVERSE_BITS) {
            return cpp_var_decl(step) + format("vreinterpretq_%s_u8(vrbitq_u8(vreinterpretq_u8_%s(%s)))",
                                               arm_vec_suffix(dtype.elem), arm_vec_suffix(dtype.elem), show(data.arg));
        }
        if (!data.is_bit && dtype.elem == VecElemType::I16 && data.permute == REVERSE_BYTES_I16) {
            return cpp_var_decl(step) + format("vreinterpretq_%s_u8(vrev16q_u8(vreinterpretq_u8_%s(%s)))",
                                               arm_vec_suffix(dtype.elem), arm_vec_suffix(dtype.elem), show(data.arg));
        }
        if (!data.is_bit && dtype.elem == VecElemType::I32 && data.permute == REVERSE_BYTES_I32) {
            return cpp_var_decl(step) + format("vreinterpretq_%s_u8(vrev32q_u8(vreinterpretq_u8_%s(%s)))",
                                               arm_vec_suffix(dtype.elem), arm_vec_suffix(dtype.elem), show(data.arg));
        }
        if (!data.is_bit && dtype.elem == VecElemType::I64 && data.permute == REVERSE_BYTES_I64) {
            return cpp_var_decl(step) + format("vreinterpretq_%s_u8(vrev64q_u8(vreinterpretq_u8_%s(%s)))",
                                               arm_vec_suffix(dtype.elem), arm_vec_suffix(dtype.elem), show(data.arg));
        }
        if (data.is_bit) { unsupported("Do not support arbitrary bit permutes"); }
        bool use_q = dtype.size_bytes() == 16;
        std::string table = reinterpret_step_to_u8(data.arg, use_q);
        std::string idxs = reinterpret_step_to_u8(data.permute_idxs, use_q);
        std::string tbl = use_q ? format("vqtbl1q_u8(%s, %s)", table.c_str(), idxs.c_str())
                                : format("vqtbl1_u8(%s, %s)", table.c_str(), idxs.c_str());
        return cpp_var_decl(step) + reinterpret_u8_to_vec(dtype, tbl.c_str());
    }

    std::string fpclass_to_cpp(const Step *step, const FpclassData &data) const {
        if (!data.arg->dtype.is_vec()) {
            messed_up("ARM C++ fpclass expects vector argument, got %s", show_dtype(data.arg->dtype));
        }
        VecDataType dtype = data.arg->dtype.as_vec();
        return cpp_var_decl(step) + format("simjit_arm_fpclass_%s(%s, 0x%x)", arm_vec_suffix(dtype.elem),
                                           show(data.arg), (unsigned)std::underlying_type_t<FpClass>(data.flags));
    }

    std::string store_sum128_to_cpp(const StoreSum128Data &data) override {
        std::string result = "{\n"
                             "uint64_t sum128_low = 0;\n"
                             "uint64_t sum128_carry = 0;\n"
                             "uint64_t sum128_prev = 0;\n";
        size_t low_idx = 0;
        for (auto *low : data.low_steps) {
            if (low->dtype.is_vec()) {
                VecDataType dtype = low->dtype.as_vec();
                std::string values = format("sum128_values_%zu", low_idx++);
                format_to(result,
                          "alignas(16) uint64_t %s[%zu];\n"
                          "%s(("
                          "%s *)"
                          "%s, "
                          "%s);\n",
                          values.c_str(), dtype.nelems(),         //
                          arm_intrin_name("st1", dtype).c_str(),  //
                          scalar_dtype_to_cpp(dtype.to_scalar()), //
                          values.c_str(),                         //
                          show(low));
                format_to(result,
                          "for (size_t sum128_i = 0; "
                          "sum128_i < %zu; "
                          "++sum128_i) {\n"
                          "sum128_prev = sum128_low;\n"
                          "sum128_low += %s[sum128_i];\n"
                          "sum128_carry += (sum128_low < sum128_prev);\n"
                          "}\n",
                          dtype.nelems(), //
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
        if (mask_pushdown != nullptr) {
            messed_up("ARM C++ emitter does not support mask pushdown for %s", show_step_kind(step->kind));
        }
        if (step->kind != StepKind::StoreSum128 && step->dtype.is_scalar() && is_scalar_step(step->kind)) {
            if (step->is(StepKind::ArithUnary)) {
                const auto &data = step->step_data<StepKind::ArithUnary>();
                if (auto code = scalar_refined_reciprocal_to_cpp(step, data)) { return *code; }
            }
            return scalar_step_to_cpp(step);
        }

        switch (step->kind) {
        case StepKind::AggResult:
        case StepKind::ScalarIndex:
        case StepKind::ScalarArithBinaryOverflow:
        case StepKind::ScalarPermute:
        case StepKind::ConstDiv:
            messed_up("Unexpected ARM C++ instruction %s", show_step_kind(step->kind));
            SIMJIT_UNREACHABLE();

        case StepKind::LoadSplat: {
            const auto &data = step->step_data<StepKind::LoadSplat>();
            if (step->dtype.is_mask()) { return mask_splat_to_cpp(step, data); }
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector or mask"); }
            VecDataType dtype = step->dtype.as_vec();
            return cpp_var_decl(step) + format("%s(*arg%zu)", arm_dup_intrin_name(dtype).c_str(), data.addr.arg);
        }
        case StepKind::Const: {
            const auto &data = step->step_data<StepKind::Const>();
            if (step->dtype.is_mask()) {
                MaskDataType dtype = step->dtype.as_mask();
                const char *value = data.as_unsigned() != 0 ? arm_mask_lane_all_ones(dtype) : "0";
                return cpp_var_decl(step) + format("%s(%s)", arm_mask_dup_intrin_name(dtype).c_str(), value);
            }
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector or mask"); }
            VecDataType dtype = step->dtype.as_vec();
            return cpp_var_decl(step) + format("%s(%s)", arm_dup_intrin_name(dtype).c_str(),
                                               const_data_to_cpp(data, dtype.to_scalar()).c_str());
        }
        case StepKind::VecConst: return vec_const_to_cpp(step, step->step_data<StepKind::VecConst>());
        case StepKind::VecIndex: {
            const auto &data = step->step_data<StepKind::VecIndex>();
            VecDataType dtype = step->dtype.as_vec();
            std::string add = arm_intrin_name("add", dtype);
            size_t acc = func->accs.index(data.acc);
            return cpp_var_decl(step) + format("acc%zu;\n"
                                               "acc%zu = %s(acc%zu, %s)",
                                               acc, //
                                               acc, add.c_str(), acc, show(data.inc));
        }
        case StepKind::Load: {
            const auto &data = step->step_data<StepKind::Load>();
            if (step->dtype.is_mask()) { return mask_load_to_cpp(step, data); }
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector or mask"); }
            VecDataType dtype = step->dtype.as_vec();
            std::string offset = offset_suffix(data.addr.offset);
            return cpp_var_decl(step) +
                   format("%s(arg%zu + i%s)", arm_intrin_name("ld1", dtype).c_str(), data.addr.arg, offset.c_str());
        }
        case StepKind::Store: {
            const auto &data = step->step_data<StepKind::Store>();
            if (step->dtype.is_mask()) { return mask_store_to_cpp(data, step); }
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector or mask"); }
            VecDataType dtype = step->dtype.as_vec();
            std::string offset = offset_suffix(data.addr.offset);
            return format("%s(arg%zu + i%s, %s)", arm_intrin_name("st1", dtype).c_str(), data.addr.arg, offset.c_str(),
                          show(data.what));
        }
        case StepKind::CondStore: {
            const auto &data = step->step_data<StepKind::CondStore>();
            return vector_cond_store_to_cpp(step, data);
        }
        case StepKind::ArithBinary: {
            const auto &data = step->step_data<StepKind::ArithBinary>();
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector"); }
            if (auto expr = binary_expr(step, data)) { return cpp_var_decl(step) + *expr; }
            unsupported("Do not support ARM C++ vector %s of %s", show_arith_binary_op(data.op),
                        show_vec_dtype(step->dtype.as_vec()));
        }
        case StepKind::ArithUnary: {
            const auto &data = step->step_data<StepKind::ArithUnary>();
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector"); }
            if (auto code = vector_refined_reciprocal_to_cpp(step, data)) { return *code; }
            if (auto expr = unary_expr(step, data)) { return cpp_var_decl(step) + *expr; }
            unsupported("Do not support ARM C++ vector %s of %s", show_arith_unary_op(data.op),
                        show_vec_dtype(step->dtype.as_vec()));
        }
        case StepKind::Compare: {
            const auto &data = step->step_data<StepKind::Compare>();
            if (auto expr = compare_expr(step, data)) { return cpp_var_decl(step) + *expr; }
            unsupported("Do not support ARM C++ compare %s", show_cmp_op(data.op));
        }
        case StepKind::Select: {
            const auto &data = step->step_data<StepKind::Select>();
            return vector_select_to_cpp(step, data);
        }
        case StepKind::AccLoad: {
            const auto &data = step->step_data<StepKind::AccLoad>();
            return cpp_var_decl(step) + format("acc%zu", func->accs.index(data));
        }
        case StepKind::AccStore: {
            const auto &data = step->step_data<StepKind::AccStore>();
            return format_acc_store(data.acc, acc_store_arg_expr(data.arg));
        }
        case StepKind::FMA: {
            const auto &data = step->step_data<StepKind::FMA>();
            if (!step->dtype.is_vec()) { return unexpected_step_dtype(step, "vector"); }
            VecDataType dtype = step->dtype.as_vec();
            return cpp_var_decl(step) + fma_expr(dtype, data);
        }
        case StepKind::BitCast: return bitcast_to_cpp(step, step->step_data<StepKind::BitCast>());
        case StepKind::PredicateNot: {
            if (!step->dtype.is_mask()) { return unexpected_step_dtype(step, "mask"); }
            const auto &data = step->step_data<StepKind::PredicateNot>();
            return cpp_var_decl(step) + mask_not_expr(step->dtype.as_mask(), show(data));
        }
        case StepKind::MaskBinary: {
            const auto &data = step->step_data<StepKind::MaskBinary>();
            if (!step->dtype.is_mask()) { return unexpected_step_dtype(step, "mask"); }
            return mask_binary_to_cpp(step, data);
        }
        case StepKind::IntCast: return int_cast_to_cpp(step, step->step_data<StepKind::IntCast>());
        case StepKind::FloatCast: return float_cast_to_cpp(step, step->step_data<StepKind::FloatCast>());
        case StepKind::Gather: return vector_gather_to_cpp(step, step->step_data<StepKind::Gather>());
        case StepKind::Scatter: unsupported("Do not support scatter");
        case StepKind::CondScatter: unsupported("Do not support cond scatter");
        case StepKind::Pack: return pack_to_cpp(step->step_data<StepKind::Pack>(), step);
        case StepKind::Ternarylogic: return ternarylogic_to_cpp(step, step->step_data<StepKind::Ternarylogic>());
        case StepKind::VecReduce: return vec_reduce_to_cpp(step, step->step_data<StepKind::VecReduce>());
        case StepKind::MaskReduce: return mask_reduce_to_cpp(step, step->step_data<StepKind::MaskReduce>());
        case StepKind::MaskCount: return mask_count_to_cpp(step, step->step_data<StepKind::MaskCount>());
        case StepKind::MaskCombine: return mask_combine_to_cpp(step, step->step_data<StepKind::MaskCombine>());
        case StepKind::VecWidenHighHalf:
            return widen_half_to_cpp(step, step->step_data<StepKind::VecWidenHighHalf>(), true);
        case StepKind::VecFloatWidenHighHalf:
            return widen_half_to_cpp(step, step->step_data<StepKind::VecFloatWidenHighHalf>(), true);
        case StepKind::VecWidenLowHalf:
            return widen_half_to_cpp(step, step->step_data<StepKind::VecWidenLowHalf>(), false);
        case StepKind::VecFloatWidenLowHalf:
            return widen_half_to_cpp(step, step->step_data<StepKind::VecFloatWidenLowHalf>(), false);
        case StepKind::VecNarrowCombine:
            return narrow_combine_to_cpp(step, step->step_data<StepKind::VecNarrowCombine>());
        case StepKind::VecFloatNarrowCombine:
            return float_narrow_combine_to_cpp(step, step->step_data<StepKind::VecFloatNarrowCombine>());
        case StepKind::VecPermute: return vec_permute_to_cpp(step, step->step_data<StepKind::VecPermute>());
        case StepKind::Fpclass: return fpclass_to_cpp(step, step->step_data<StepKind::Fpclass>());
        case StepKind::StoreSum128: return store_sum128_to_cpp(step->step_data<StepKind::StoreSum128>());
        }
        SIMJIT_UNREACHABLE();
    }
};

std::unique_ptr<CppEmitterBase> make_arm_neon_cpp_emitter(const Function *func) {
    return std::make_unique<ArmNeonCppEmitter>(func);
}

#undef unsupported
#undef messed_up

} // namespace cpp_backend
} // namespace simjit
