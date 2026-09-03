// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/compiler.h"
#include "simjit/core/expr.h"

namespace simjit {
namespace mir {

struct Step;

struct AccId {
    uint32_t local;
    int32_t group;

    // Special accumulators are whatever we decide to create internally in MIR without user-visible store
    constexpr bool is_special() const noexcept { return group == -1; }
    constexpr bool operator==(AccId right) const noexcept { return group == right.group && local == right.local; }
    constexpr bool operator!=(AccId right) const noexcept { return !(*this == right); }
};

static_assert(sizeof(AccId) == sizeof(uint64_t));

struct ArgumentAddress {
    ArgumentIdx arg;
    size_t offset;
};

struct ArithBinData {
    Step *left;
    Step *right;
    ArithBinaryOp op;
};

struct PredicateBinData {
    Step *left;
    Step *right;
    PredicateBinaryOp op;
};

struct ArithUnaryData {
    Step *arg;
    ArithUnaryOp op;
};

struct GatherData {
    ArgumentIdx data;
    Step *idx;
};

struct StoreData {
    ArgumentAddress addr;
    Step *what;
    LoadStoreKind kind;
};

struct CmpData {
    Step *left;
    Step *right;
    CmpOp op;
    bool is_unsigned = false;
};

struct AccStoreData {
    AccId acc;
    Step *arg;
};

struct AggResultData {
    Step *arg;
    ArgumentIdx dst;
};

struct ArithReduceData {
    Step *arg;
    ArithBinaryOp op;
};

struct PredicateReduceData {
    Step *arg;
    PredicateBinaryOp op;
};

struct CombineMaskData {
    Step *left;
    Step *right;
};

struct SelectData {
    Step *cond;
    Step *truthy;
    Step *falsy;
};

struct VecConstData {
    void *mem;
};

struct IndexData {
    AccId acc;
    Step *inc;
};

struct ScatterData {
    ArgumentIdx dst;
    Step *idx;
    Step *arg;
};

struct CondScatterData {
    ArgumentIdx dst;
    Step *idx;
    Step *arg;
    Step *cond;
};

struct PackData {
    ArgumentIdx dst;
    Step *arg;
    Step *cond;
    AccId acc;
};

struct CondStoreData {
    ArgumentAddress addr;
    Step *arg;
    Step *cond;
    LoadStoreKind kind;
};

struct TernarylogicData {
    Step *a;
    Step *b;
    Step *c;
    int fun;
    VecDataType lookup_type;
};

struct ScalarBinWithSafetyCheck {
    ArithBinaryOp op;
    Step *left;
    Step *right;
    AccId overflow_flag;
    Step *mask = nullptr;
};

struct IntCastData {
    Step *arg;
    IntCastKind kind;
};

struct FloatCastData {
    Step *arg;
    bool is_unsigned;
};

struct VecPermuteData {
    bool is_bit;
    Step *arg;
    Step *permute_idxs;
    uint64_t permute;
};

struct ScalarPermuteData {
    bool is_bit;
    Step *arg;
    uint64_t permute;
};

struct LoadData {
    ArgumentAddress addr;
    LoadStoreKind kind;
};

struct StoreSum128Data {
    ArenaArray<Step *> low_steps;
    Step *hi_combined;
    ArgumentIdx dst;
};

enum class FmaKind : uint8_t {
    FMA,  // result = (x1 * x2) + x3
    FMS,  // result = (x1 * x2) - x3
    FNMA, // result = -(x1 * x2) + x3
    FNMS  // result = -(x1 * x2) - x3
};

struct FMAData {
    Step *x1;
    Step *x2;
    Step *x3;
    FmaKind kind;
};

struct FpclassData {
    Step *arg;
    FpClass flags;
};

struct HalfCast {
    Step *arg;
    bool is_unsigned = false;
};

struct VecNarrowCombineData {
    Step *low;
    Step *high;
};

struct VecFloatNarrowCombineData {
    Step *low;
    Step *high;
};

struct ConstDivData {
    Step *numerator;
    Step *magic;
    Step *round_mask;
    Step *divisor;
    ArithBinaryOp op;
    uint8_t shift;
    bool has_magic;
    bool has_add;
    bool negative_divisor;
};

enum class StepKind : uint8_t {
    Const,
    Load,
    LoadSplat,
    Gather,
    Store,
    ArithBinary,
    ArithUnary,
    IntCast,
    FloatCast,
    BitCast,
    Compare,
    AggResult,
    VecReduce,
    MaskReduce,
    AccLoad,
    AccStore,
    VecWidenHighHalf,
    VecWidenLowHalf,
    VecFloatWidenHighHalf,
    VecFloatWidenLowHalf,
    MaskBinary,
    MaskCount,
    MaskCombine,
    PredicateNot,
    Select,
    VecIndex,
    Scatter,
    Pack,
    Ternarylogic,
    CondStore,
    ScalarIndex,
    ScalarArithBinaryOverflow,
    VecConst,
    VecPermute,
    ScalarPermute,
    StoreSum128,
    CondScatter,
    FMA,
    Fpclass,
    VecNarrowCombine,
    VecFloatNarrowCombine,
    ConstDiv,
};

struct Step {
    uint32_t id;
    StepKind kind;
    DataType dtype;

    template <StepKind Kind> struct Data;
    template <StepKind Kind> const typename Data<Kind>::T &step_data() const noexcept;
    template <StepKind Kind> typename Data<Kind>::T &step_data() noexcept;
    constexpr bool is(StepKind k) const noexcept { return kind == k; }

private:
    friend class StepMaker;

    struct ConstructorTag {};
    explicit Step(struct ConstructorTag) noexcept {}

    union {
        Step *arg;
        CmpData cmp;
        StoreData store;
        SelectData select;
        ArithBinData ab;
        GatherData gather;
        ArithUnaryData au;
        PredicateBinData mb;
        ConstData con;
        AccId acc_move;
        LoadData load;
        AccStoreData acc_store;
        AggResultData agg_result;
        ArithReduceData ab_reduce;
        PredicateReduceData lb_reduce;
        CombineMaskData combine_mask;
        VecConstData vec_const;
        IndexData index;
        ScatterData scatter;
        PackData pack;
        TernarylogicData ternarylogic;
        CondStoreData cond_store;
        ScalarBinWithSafetyCheck ab_overflow;
        IntCastData int_cast;
        FloatCastData float_cast;
        VecPermuteData vec_permute;
        ScalarPermuteData scalar_permute;
        StoreSum128Data sum128;
        CondScatterData cond_scatter;
        FMAData fma;
        FpclassData fpclass;
        HalfCast half_cast;
        VecNarrowCombineData narrow_combine;
        VecFloatNarrowCombineData float_narrow_combine;
        ConstDivData const_div;
    };
};

enum class AllowedDataTypes : uint8_t {
    Scalar,
    Vec,
    Mask,
    ScalarOrMask,
    ScalarOrVec,
    Any
};

#define MIR_STEP_DATA_LIST(X)                                                     \
    X(con, Const, con, Any)                                                       \
    X(load, Load, load, Any)                                                      \
    X(load_splat, LoadSplat, load, Any)                                           \
    X(gather, Gather, gather, ScalarOrVec)                                        \
    X(store, Store, store, Any)                                                   \
    X(arith_un, ArithUnary, au, ScalarOrVec)                                      \
    X(arith_bin, ArithBinary, ab, ScalarOrVec)                                    \
    X(mask_bin, MaskBinary, mb, Mask)                                             \
    X(int_cast, IntCast, int_cast, ScalarOrVec)                                   \
    X(widen_hi, VecWidenHighHalf, half_cast, Vec)                                 \
    X(widen_lo, VecWidenLowHalf, half_cast, Vec)                                  \
    X(float_widen_hi, VecFloatWidenHighHalf, half_cast, Vec)                      \
    X(float_widen_lo, VecFloatWidenLowHalf, half_cast, Vec)                       \
    X(count_mask, MaskCount, arg, Scalar)                                         \
    X(cmp, Compare, cmp, ScalarOrMask)                                            \
    X(acc_load, AccLoad, acc_move, Any)                                           \
    X(agg_result, AggResult, agg_result, Scalar)                                  \
    X(vec_reduce, VecReduce, ab_reduce, Scalar)                                   \
    X(mask_reduce, MaskReduce, lb_reduce, Scalar)                                 \
    X(acc_store, AccStore, acc_store, Any)                                        \
    X(combine_mask, MaskCombine, combine_mask, Mask)                              \
    X(predicate_not, PredicateNot, arg, ScalarOrMask)                             \
    X(select, Select, select, ScalarOrVec)                                        \
    X(vec_index, VecIndex, index, Vec)                                            \
    X(scatter, Scatter, scatter, ScalarOrVec)                                     \
    X(pack, Pack, pack, ScalarOrVec)                                              \
    X(ternarylogic, Ternarylogic, ternarylogic, Vec)                              \
    X(cond_store, CondStore, cond_store, ScalarOrVec)                             \
    X(scalar_index, ScalarIndex, index, Scalar)                                   \
    X(ab_overflow, ScalarArithBinaryOverflow, ab_overflow, Scalar)                \
    X(vec_const, VecConst, vec_const, Vec)                                        \
    X(vec_permute, VecPermute, vec_permute, Vec)                                  \
    X(scalar_permute, ScalarPermute, scalar_permute, Scalar)                      \
    X(sum128, StoreSum128, sum128, Scalar)                                        \
    X(cond_scatter, CondScatter, cond_scatter, ScalarOrVec)                       \
    X(float_cast, FloatCast, float_cast, ScalarOrVec)                             \
    X(fma, FMA, fma, ScalarOrVec)                                                 \
    X(bitcast, BitCast, arg, ScalarOrVec)                                         \
    X(fpclass, Fpclass, fpclass, ScalarOrMask)                                    \
    X(vec_narrow_combine, VecNarrowCombine, narrow_combine, Vec)                  \
    X(vec_float_narrow_combine, VecFloatNarrowCombine, float_narrow_combine, Vec) \
    X(const_div, ConstDiv, const_div, Scalar)

#define ASSOC_STEP_DATA(_0, _kind, _field, _2)       \
    template <> struct Step::Data<StepKind::_kind> { \
        using T = decltype(Step::_field);            \
    };

MIR_STEP_DATA_LIST(ASSOC_STEP_DATA)

#undef ASSOC_STEP_DATA

template <AllowedDataTypes Allowed> constexpr bool check_dtype(const Step *step) noexcept {
    switch (Allowed) {
    case AllowedDataTypes::Scalar: return step->dtype.is_scalar();
    case AllowedDataTypes::Vec: return step->dtype.is_vec();
    case AllowedDataTypes::Mask: return step->dtype.is_mask();
    case AllowedDataTypes::ScalarOrMask: return step->dtype.is_scalar() || step->dtype.is_mask();
    case AllowedDataTypes::ScalarOrVec: return step->dtype.is_scalar() || step->dtype.is_vec();
    case AllowedDataTypes::Any: return true;
    }
    SIMJIT_UNREACHABLE();
}

#define STEP_DATA(_0, _kind, _field, _allowed)                                                                   \
    template <> inline const Step::Data<StepKind::_kind>::T &Step::step_data<StepKind::_kind>() const noexcept { \
        SIMJIT_ASSERT(this->is(StepKind::_kind));                                                                \
        SIMJIT_ASSERT(check_dtype<AllowedDataTypes::_allowed>(this));                                            \
        return this->_field;                                                                                     \
    }                                                                                                            \
    template <> inline Step::Data<StepKind::_kind>::T &Step::step_data<StepKind::_kind>() noexcept {             \
        SIMJIT_ASSERT(this->is(StepKind::_kind));                                                                \
        SIMJIT_ASSERT(check_dtype<AllowedDataTypes::_allowed>(this));                                            \
        return this->_field;                                                                                     \
    }

MIR_STEP_DATA_LIST(STEP_DATA)

#undef STEP_DATA

template <typename Fn> SIMJIT_NO_ASAN void step_recurse(Step *step, Fn process) {
    switch (step->kind) {
    case StepKind::Const:
    case StepKind::VecConst:
    case StepKind::Load:
    case StepKind::AccLoad:
    case StepKind::LoadSplat:
    case StepKind::ScalarIndex:
        break;

        SIMJIT_MATCH (StepKind::VecIndex) {
            process(data.inc);
            break;
        }
        SIMJIT_MATCH (StepKind::MaskCombine) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::ArithBinary) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::ArithUnary) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::IntCast) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH2 (StepKind::VecWidenHighHalf, StepKind::VecWidenLowHalf) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH2 (StepKind::VecFloatWidenHighHalf, StepKind::VecFloatWidenLowHalf) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::VecNarrowCombine) {
            process(data.low);
            process(data.high);
            break;
        }
        SIMJIT_MATCH (StepKind::VecFloatNarrowCombine) {
            process(data.low);
            process(data.high);
            break;
        }
        SIMJIT_MATCH (StepKind::ConstDiv) {
            process(data.numerator);
            if (data.magic != nullptr) { process(data.magic); }
            if (data.round_mask != nullptr) { process(data.round_mask); }
            process(data.divisor);
            break;
        }
        SIMJIT_MATCH2 (StepKind::PredicateNot, StepKind::MaskCount) {
            process(data);
            break;
        }
        SIMJIT_MATCH (StepKind::VecReduce) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::MaskReduce) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::Store) {
            process(data.what);
            break;
        }
        SIMJIT_MATCH (StepKind::Compare) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::AccStore) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::Gather) {
            process(data.idx);
            break;
        }
        SIMJIT_MATCH (StepKind::MaskBinary) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::AggResult) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::Select) {
            process(data.cond);
            process(data.truthy);
            process(data.falsy);
            break;
        }
        SIMJIT_MATCH (StepKind::Scatter) {
            process(data.arg);
            process(data.idx);
            break;
        }
        SIMJIT_MATCH (StepKind::Pack) {
            process(data.arg);
            process(data.cond);
            break;
        }
        SIMJIT_MATCH (StepKind::CondStore) {
            process(data.arg);
            process(data.cond);
            break;
        }
        SIMJIT_MATCH (StepKind::Ternarylogic) {
            process(data.a);
            process(data.b);
            process(data.c);
            break;
        }
        SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
            process(data.left);
            process(data.right);
            if (data.mask) { process(data.mask); }
            break;
        }
        SIMJIT_MATCH (StepKind::VecPermute) {
            process(data.arg);
            process(data.permute_idxs);
            break;
        }
        SIMJIT_MATCH (StepKind::ScalarPermute) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::StoreSum128) {
            for (auto x : data.low_steps) {
                process(x);
            }
            process(data.hi_combined);
            break;
        }
        SIMJIT_MATCH (StepKind::CondScatter) {
            process(data.idx);
            process(data.arg);
            process(data.cond);
            break;
        }
        SIMJIT_MATCH (StepKind::FloatCast) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::FMA) {
            process(data.x1);
            process(data.x2);
            process(data.x3);
            break;
        }
        SIMJIT_MATCH (StepKind::BitCast) {
            process(data);
            break;
        }
        SIMJIT_MATCH (StepKind::Fpclass) {
            process(data.arg);
            break;
        }
    }
}

inline bool step_is_zero(const Step *step) noexcept {
    return step->is(StepKind::Const) && step->step_data<StepKind::Const>().is_zero();
}

template <typename Fn> SIMJIT_NO_ASAN void traverse_steps_postorder_impl(Step *step, Fn &fn);
template <typename Fn>
SIMJIT_NO_ASAN void traverse_steps_postorder_unique_impl(Step *step, Fn &fn, std::vector<uint8_t> &state);

template <typename Fn> struct PostorderRecurse {
    Fn &fn;

    SIMJIT_NO_ASAN void operator()(Step *s) { traverse_steps_postorder_impl(s, fn); }
};

template <typename Fn> struct UniquePostorderRecurse {
    Fn &fn;
    std::vector<uint8_t> &state;

    SIMJIT_NO_ASAN void operator()(Step *s) { traverse_steps_postorder_unique_impl(s, fn, state); }
};

template <typename Fn> SIMJIT_NO_ASAN void traverse_steps_postorder_impl(Step *step, Fn &fn) {
    step_recurse(step, PostorderRecurse<Fn>{fn});
    fn(step);
}

template <typename Fn> SIMJIT_NO_ASAN void traverse_steps_postorder(Step *step, Fn &&fn) {
    traverse_steps_postorder_impl(step, fn);
}

template <typename Fn>
SIMJIT_NO_ASAN void traverse_steps_postorder_unique_impl(Step *step, Fn &fn, std::vector<uint8_t> &state) {
    enum : uint8_t {
        White = 0,
        Gray = 1,
        Black = 2,
    };
    SIMJIT_ASSERT(step->id < state.size());
    if (state[step->id] == Black) return;
    if (state[step->id] == Gray) {
        simjit_exception(ErrorModule::MIR, {}, {}, "cycle detected during traversal at step id %u", step->id);
    }
    state[step->id] = Gray;
    step_recurse(step, UniquePostorderRecurse<Fn>{fn, state});
    state[step->id] = Black;
    fn(step);
}

template <typename Fn>
SIMJIT_NO_ASAN void traverse_steps_postorder_unique(Step *step, std::vector<uint8_t> &state, Fn &&fn) {
    traverse_steps_postorder_unique_impl(step, fn, state);
}

template <typename Fn>
SIMJIT_NO_ASAN void traverse_steps_postorder_unique(size_t step_count, nonstd::span<Step *const> roots, Fn fn) {
    std::vector<uint8_t> state(step_count, 0);
    for (Step *s : roots) {
        traverse_steps_postorder_unique(s, state, fn);
    }
}

struct AccumulatorInfo {
    // Prefix offsets for HIR accumulator groups. The final element equals agg_count.
    ArenaArray<size_t> group_offsets;
    size_t agg_count = 0;
    size_t special_count = 0;
    // agg_count + special_count
    size_t count = 0;

    size_t index(AccId id) const noexcept {
        if (id.is_special()) {
            SIMJIT_ASSERT(id.local < special_count);
            return agg_count + id.local;
        }
        SIMJIT_ASSERT(id.group >= 0 && size_t(id.group + 1) < group_offsets.size());
        size_t begin = group_offsets[id.group];
        size_t end = group_offsets[id.group + 1];
        SIMJIT_ASSERT(id.local < end - begin);
        return begin + id.local;
    }
};

struct Function {
    Context *ctx;
    ArenaArray<ArgumentDecl> args;
    AccumulatorInfo accs;
    ArenaArray<Step *> prologue_roots;
    ArenaArray<Step *> main_loop_roots;
    ArenaArray<Step *> remainder_roots;
    ArenaArray<Step *> epilogue_roots;
    // How many elements are processed per iteration?
    size_t loop_width = 1;
    // How many total steps are there?
    size_t step_id_count = 0;
};

const char *show_step_kind(StepKind kind) noexcept;

// This step can exist outside of vector context
constexpr bool is_scalar_step(StepKind kind) noexcept {
    switch (kind) {
    case StepKind::Const:
    case StepKind::Load:
    case StepKind::LoadSplat:
    case StepKind::Gather:
    case StepKind::Store:
    case StepKind::ArithBinary:
    case StepKind::FMA:
    case StepKind::ArithUnary:
    case StepKind::IntCast:
    case StepKind::FloatCast:
    case StepKind::Compare:
    case StepKind::AggResult:
    case StepKind::PredicateNot:
    case StepKind::Select:
    case StepKind::AccLoad:
    case StepKind::AccStore:
    case StepKind::ScalarIndex:
    case StepKind::Scatter:
    case StepKind::Pack:
    case StepKind::CondStore:
    case StepKind::ScalarArithBinaryOverflow:
    case StepKind::ScalarPermute:
    case StepKind::StoreSum128:
    case StepKind::CondScatter:
    case StepKind::BitCast:
    case StepKind::Fpclass:
    case StepKind::ConstDiv: return true;
    default: break;
    }
    return false;
}

std::vector<uint8_t> generate_bit_permute_lut(uint64_t func);

} // namespace mir
} // namespace simjit
