// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

// MIR is the expanded form of graphs we have in HIR and vectorizer. This is done to avoid guessing as much as possible
// in the backends. However, this stage adds redundancy to the expression graph. Two important entities we define here
// are 1. Assignment of row offsets 2. Explicit accumulators.
//
// Compared to vectorizer (which currently duplicates trees only for Widen), we expand all unrolls and duplications.
// Unroll repeats root evaluation, and mask and narrow combines repeat child node evaluation.
// To facilitatate assignment of row offsets during duplications we maintain RowBlock in LoweringContext.
// For the operations that change processing width (Widen, NarrowCombine), we update RowBlock indices and width.
//
// For aggregates, we allow AccSplit optimization, which will give each unrolled copy its own accumulator variable.
// During unroll this information is propagated in AccRemap.

#include "simjit/core/mir.h"
#include "simjit/compiler.h"
#include "simjit/core/expr.h"
#include "simjit/core/hir.h"
#include "simjit/core/vectorizer.h"
#include "simjit/core/x86.h"
#include "simjit/detail/base.h"
#include "simjit/simjit.h"

#include <algorithm>
#include <unordered_map>

#if SIMJIT_USE_LIBDIVIDE
#include "libdivide.h"
#endif

#define messed_up(...) simjit_exception(ErrorModule::MIR, {}, {}, __VA_ARGS__)

#define invariant(_check) SIMJIT_ASSERT(_check)
#define invariantm(_check, ...) SIMJIT_ASSERT(_check)

#define invariant_i1(_a) invariantm((_a)->dtype == ScalarDataType::I1, "Type is not i1: %s", show_dtype((_a)->dtype))
#define invariant_same_type(_a, _b)                                                                 \
    invariantm((_a)->dtype == (_b)->dtype, "Types are not same: %s vs %s", show_dtype((_a)->dtype), \
               show_dtype((_b)->dtype))
#define invariant_same_scalar_type(_a, _b)                                                                   \
    invariantm((_a)->dtype.is_scalar() && (_a)->dtype == (_b)->dtype, "Types are not same scalar: %s vs %s", \
               show_dtype((_a)->dtype), show_dtype((_b)->dtype))
#define invariant_same_vec_type(_a, _b)                                                                   \
    invariantm((_a)->dtype.is_vec() && (_a)->dtype == (_b)->dtype, "Types are not same vector: %s vs %s", \
               show_dtype((_a)->dtype), show_dtype((_b)->dtype))
#define invariant_same_vec_len(_a, _b)                                             \
    invariantm((_a)->dtype.is_vec() && (_b)->dtype.is_vec() &&                     \
                   (_a)->dtype.as_vec().nelems() == (_b)->dtype.as_vec().nelems(), \
               "Types are not vector of same len: %s vs %s", show_dtype((_a)->dtype), show_dtype((_b)->dtype))
#define invariant_same_mask_type(_a, _b)                                                                 \
    invariantm((_a)->dtype.is_mask() && (_a)->dtype == (_b)->dtype, "Types are not same mask: %s vs %s", \
               show_dtype((_a)->dtype), show_dtype((_b)->dtype))
#define invariant_vec_mask(_a, _b)                                                                                    \
    invariantm((_a)->dtype.is_vec() && (_b)->dtype.is_mask() && (_a)->dtype.as_vec().mask() == (_b)->dtype.as_mask(), \
               "Type %s is not mask for %s", show_dtype((_b)->dtype), show_dtype((_a)->dtype))
#define invariant_imply(_a, _b) invariant((_a) ? (_b) : true)

namespace simjit {
namespace {
struct ConstantKey {
    ConstData x;
    DataType step_dt;

    bool operator==(const ConstantKey &right) const { return x == right.x && step_dt == right.step_dt; }
};

struct ConstQuotientKey {
    const mir::Step *numerator;
    ConstData divisor;
    bool is_signed;

    bool operator==(const ConstQuotientKey &right) const noexcept {
        return numerator == right.numerator && divisor == right.divisor && is_signed == right.is_signed;
    }
};

struct ConstQuotientKeyHash {
    size_t operator()(const ConstQuotientKey &key) const noexcept {
        size_t result = std::hash<const mir::Step *>()(key.numerator);
        result = result * 31 + std::hash<uint64_t>()(key.divisor.raw_bits());
        result = result * 31 + std::hash<ScalarDataType>()(key.divisor.dtype);
        return result * 31 + std::hash<bool>()(key.is_signed);
    }
};

struct ContextualMirCacheKey {
    uint32_t node_id = 0;
    vect::RowBlock rows{};

    bool operator==(const ContextualMirCacheKey &right) const noexcept {
        return node_id == right.node_id && rows.width == right.rows.width && rows.idx == right.rows.idx;
    }
};

} // namespace
} // namespace simjit

template <> struct std::hash<simjit::ConstantKey> {
    size_t operator()(const simjit::ConstantKey &tt) const {
        return std::hash<uint64_t>()(tt.x.raw_bits()) + std::hash<simjit::ScalarDataType>()(tt.x.dtype) +
               std::hash<simjit::DataTypeKind>()(tt.step_dt.kind);
    }
};

template <> struct std::hash<simjit::ContextualMirCacheKey> {
    size_t operator()(const simjit::ContextualMirCacheKey &key) const noexcept {
        size_t result = std::hash<uint32_t>()(key.node_id);
        result = result * 31 + std::hash<size_t>()(key.rows.width);
        return result * 31 + std::hash<size_t>()(key.rows.idx);
    }
};

namespace simjit {
namespace mir {

namespace {
template <AllowedDataTypes Allowed> struct AllowedMap;

template <> struct AllowedMap<AllowedDataTypes::Scalar> {
    using Type = ScalarDataType;
};
template <> struct AllowedMap<AllowedDataTypes::Vec> {
    using Type = VecDataType;
};
template <> struct AllowedMap<AllowedDataTypes::Mask> {
    using Type = MaskDataType;
};
template <> struct AllowedMap<AllowedDataTypes::Any> {
    using Type = DataType;
};
template <> struct AllowedMap<AllowedDataTypes::ScalarOrVec> {
    using Type = DataType;
};
template <> struct AllowedMap<AllowedDataTypes::ScalarOrMask> {
    using Type = DataType;
};

} // namespace

class StepMaker {
public:
    explicit StepMaker(MemoryArena *arena) noexcept : arena_(arena) {}

#define STEP_DATA(_name, _kind, _field, _allowed)                                                          \
    Step *_name(Step::Data<StepKind::_kind>::T data, AllowedMap<AllowedDataTypes::_allowed>::Type dtype) { \
        Step *step = make_step(StepKind::_kind, dtype);                                                    \
        step->_field = data;                                                                               \
        check_dtype<AllowedDataTypes::_allowed>(step);                                                     \
        return step;                                                                                       \
    }
    MIR_STEP_DATA_LIST(STEP_DATA)

#undef STEP_DATA

    Step *copy(const Step *other) {
        Step *created = arena_->create<Step>(*other);
        created->id = id_counter_++;
        return created;
    }

    Step *make_vec_const(const void *mem, VecDataType vdtype) {
        void *allocd = arena_->alloc(64);
        memcpy(allocd, mem, 64);
        Step *created = vec_const({allocd}, vdtype);
        return created;
    }

    uint32_t max_id() const { return id_counter_; }

private:
    Step *make_step(StepKind kind, DataType dtype) {
        Step step{Step::ConstructorTag{}};
        step.id = id_counter_++;
        step.kind = kind;
        step.dtype = dtype;
        return arena_->create<Step>(step);
    }

    MemoryArena *arena_ = nullptr;
    uint32_t id_counter_ = 0;
};

namespace {
enum class AccKind : uint8_t {
    Agg,
    IndexInc,
    Pack,
    Overflow,
    OverflowMul,
    Sum128
};

struct AccRemap {
    std::unordered_map<AccIdx, uint32_t> map{};
};

class AccGroup {
public:
    AccGroup(AccKind k, ScalarDataType ty, AccIdx hir, const hir::Step *agg, ArgumentIdx dst) noexcept
        : kind(k), dst_dtype(ty), hir_idx(hir), agg_expr(agg), dst_arg(dst),
          slots_per_copy(kind == AccKind::Sum128 ? 2 : 1) {}

    // Fixed group description. These values are established with the HIR group and never change during lowering.
    const AccKind kind;
    const ScalarDataType dst_dtype;
    const AccIdx hir_idx;
    const hir::Step *const agg_expr;
    const ArgumentIdx dst_arg;
    const uint32_t slots_per_copy;

    uint32_t main_count() const noexcept { return main_count_; }
    AccId alloc_main(AccRemap *remap);
    std::pair<AccId, AccId> alloc_main_pair(AccRemap *remap);
    AccId alloc_remainder();
    std::pair<AccId, AccId> alloc_remainder_pair();
    AccId singleton();

    uint32_t storage_count() const noexcept {
        SIMJIT_ASSERT(remainder_allocated_);
        return (main_count_ + 1) * slots_per_copy;
    }
    AccId main(uint32_t copy, uint32_t slot = 0) const noexcept {
        SIMJIT_ASSERT(copy < main_count_ && slot < slots_per_copy);
        return {(copy + 1) * slots_per_copy + slot, int32_t(hir_idx)};
    }
    AccId remainder(uint32_t slot = 0) const noexcept {
        SIMJIT_ASSERT(remainder_allocated_ && slot < slots_per_copy);
        return {slot, int32_t(hir_idx)};
    }

private:
    uint32_t main_count_ = 0;
    // The remainder identity is fixed at local zero. This flag only enforces exactly-once allocation.
    bool remainder_allocated_ = false;
};

enum class SafetyCheckKind : uint8_t {
    // Add and sub on ternarylogic targets store the safety check flag in the sign bit and defer mask construction to
    // the epilogue.
    InSignBit,
    // Multiplication safety check is more compilcated than others - we store maxiumum number
    Mul,
    // Just a mask (can remove Simple here)
    SimpleMask,
    // Used in scalar
    Bool
};

struct SafetyCheckInfo {
    SafetyCheckKind kind;
    DataType acc_dtype;
    AccId acc;
};
} // namespace

const char *show_step_kind(StepKind kind) noexcept {
    switch (kind) {
    case StepKind::Const: return "const";
    case StepKind::Load: return "load";
    case StepKind::LoadSplat: return "load-splat";
    case StepKind::Gather: return "gather";
    case StepKind::ArithBinary: return "binary";
    case StepKind::ArithUnary: return "unary";
    case StepKind::Store: return "store";
    case StepKind::Compare: return "cmp";
    case StepKind::AggResult: return "aggresult";
    case StepKind::AccLoad: return "accload";
    case StepKind::AccStore: return "accstore";
    case StepKind::VecWidenHighHalf: return "widen-high";
    case StepKind::VecWidenLowHalf: return "widen-low";
    case StepKind::VecFloatWidenHighHalf: return "float-widen-high";
    case StepKind::VecFloatWidenLowHalf: return "float-widen-low";
    case StepKind::IntCast: return "int-cast";
    case StepKind::FloatCast: return "float-cast";
    case StepKind::MaskBinary: return "mask-binary";
    case StepKind::MaskCount: return "mask-count";
    case StepKind::VecReduce: return "reduce";
    case StepKind::MaskCombine: return "combine-mask";
    case StepKind::MaskReduce: return "mask-reduce";
    case StepKind::PredicateNot: return "predicate-not";
    case StepKind::Select: return "select";
    case StepKind::VecIndex: return "vec-index";
    case StepKind::ScalarIndex: return "scalar-index";
    case StepKind::Scatter: return "scatter";
    case StepKind::Pack: return "pack";
    case StepKind::Ternarylogic: return "ternarylogic";
    case StepKind::CondStore: return "cond-store";
    case StepKind::ScalarArithBinaryOverflow: return "ab-overflow";
    case StepKind::VecConst: return "vec-const";
    case StepKind::VecPermute: return "vec-permute";
    case StepKind::ScalarPermute: return "scalar-permute";
    case StepKind::StoreSum128: return "store-sum128";
    case StepKind::CondScatter: return "cond-scatter";
    case StepKind::FMA: return "fma";
    case StepKind::BitCast: return "bitcast";
    case StepKind::Fpclass: return "fpclass";
    case StepKind::VecNarrowCombine: return "narrow-combine";
    case StepKind::VecFloatNarrowCombine: return "float-narrow-combine";
    case StepKind::ConstDiv: return "const-div";
    }
    SIMJIT_UNREACHABLE();
}

constexpr bool is_root_step(StepKind kind) noexcept {
    switch (kind) {
    case StepKind::Store:
    case StepKind::AggResult:
    case StepKind::AccStore:
    case StepKind::Scatter:
    case StepKind::Pack:
    case StepKind::CondStore:
    case StepKind::StoreSum128:
    case StepKind::CondScatter: return true;
    default: break;
    }
    return false;
}

// Ternarylogic instruction only supports i32 and i64. However, it only matters if we pushdown masks to it - which we do
// not. So we just use i32 version all times. This functions returns i32 vector of same size for given vector type,
// which we can use to lookup ternarylogic.
static VecDataType ternarylogic_coerced_type(VecDataType dtype) {
    return x86::x86_to_vec(x86::Vector{x86::vec_to_x86(dtype).reg, VecElemType::I32});
}

static bool is_bit_binary_op(ArithBinaryOp op) noexcept {
    switch (op) {
    case ArithBinaryOp::And:
    case ArithBinaryOp::Or:
    case ArithBinaryOp::Xor:
    case ArithBinaryOp::AndNot: return true;
    default: break;
    }
    return false;
}

static bool is_division_op(ArithBinaryOp op) noexcept {
    return op == ArithBinaryOp::Div || op == ArithBinaryOp::UDiv || op == ArithBinaryOp::Mod ||
           op == ArithBinaryOp::UMod;
}

static bool is_signed_division_op(ArithBinaryOp op) noexcept {
    return op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod;
}

static ScalarDataType const_storage_dtype(DataType dtype) noexcept {
    switch (dtype.kind) {
    case DataTypeKind::Scalar: return dtype.scalar;
    case DataTypeKind::Vec: return dtype.vec.to_scalar();
    case DataTypeKind::Mask: return mask_dtype_to_scalar(dtype.mask);
    }
    SIMJIT_UNREACHABLE();
}

static const char *show_fma_kind(FmaKind kind) {
    switch (kind) {
    case FmaKind::FMA: return "fma";
    case FmaKind::FMS: return "fms";
    case FmaKind::FNMA: return "fnma";
    case FmaKind::FNMS: return "fnms";
    }
    SIMJIT_UNREACHABLE();
}

namespace {
struct ExprShowState {
    uint32_t counter = 1;
    std::vector<size_t> show_cache{};
    std::string buf{};

    void show_step(const Step *step) {
#define wr_(...) simjit::format_to(buf, __VA_ARGS__)
#define wr(...)           \
    do {                  \
        wr_(__VA_ARGS__); \
        return;           \
    } while (0)
#define ref(_step) ((unsigned)show_cache[(_step)->id])

        wr_("%s dtype=%s ", show_step_kind(step->kind), show_dtype(step->dtype));
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::Const)
                wr("value=%s",
                   show_const_data(data, step->dtype.is_scalar() ? step->dtype.as_scalar()
                                         : step->dtype.is_vec()  ? step->dtype.as_vec().to_scalar()
                                                                 : mask_dtype_to_scalar(step->dtype.as_mask()))
                       .c_str());
            SIMJIT_MATCH (StepKind::Load)
                wr("arg=@%zu offset=%zu kind=%s", data.addr.arg, data.addr.offset, show_load_store_kind(data.kind));
            SIMJIT_MATCH (StepKind::LoadSplat) wr("arg=@%zu", data.addr.arg);
            SIMJIT_MATCH (StepKind::Gather) wr("arg=%zu idx=%%%u", data.data, ref(data.idx));
            SIMJIT_MATCH (StepKind::ArithBinary)
                wr("op=%s left=%%%u right=%%%u", show_arith_binary_op(data.op), ref(data.left), ref(data.right));
            SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
                wr_("op=%s left=%%%u right=%%%u", show_arith_binary_op(data.op), ref(data.left), ref(data.right));
                if (data.mask != nullptr) wr_(" mask=%%%u", ref(data.mask));
                return;
            }
            SIMJIT_MATCH (StepKind::ArithUnary) wr("op=%s arg=%%%u", show_arith_unary_op(data.op), ref(data.arg));
            SIMJIT_MATCH (StepKind::IntCast) wr("arg=%%%u kind=%s", ref(data.arg), show_int_cast_kind(data.kind));
            SIMJIT_MATCH (StepKind::FloatCast)
                wr("arg=%%%u is_unsigned=%s", ref(data.arg), data.is_unsigned ? "true" : "false");
            SIMJIT_MATCH (StepKind::Store)
                wr("dst=@%zu src=%%%u offset=%zu kind=%s", data.addr.arg, ref(data.what), data.addr.offset,
                   show_load_store_kind(data.kind));
            SIMJIT_MATCH (StepKind::Compare)
                wr("op=%s left=%%%u right=%%%u is_unsigned=%s", show_cmp_op(data.op), ref(data.left), ref(data.right),
                   data.is_unsigned ? "true" : "false");
            SIMJIT_MATCH (StepKind::AggResult) wr("dst=%zu arg=%%%u", data.dst, ref(data.arg));
            SIMJIT_MATCH (StepKind::AccLoad)
                wr("acc=$%lld:%llu", (long long)data.group, (unsigned long long)data.local);
            SIMJIT_MATCH (StepKind::AccStore)
                wr("acc=$%lld:%llu arg=%%%u", (long long)data.acc.group, (unsigned long long)data.acc.local,
                   ref(data.arg));
            SIMJIT_MATCH2 (StepKind::VecWidenHighHalf, StepKind::VecWidenLowHalf)
                wr("arg=%%%u is_unsigned=%s", ref(data.arg), data.is_unsigned ? "true" : "false");
            SIMJIT_MATCH2 (StepKind::VecFloatWidenHighHalf, StepKind::VecFloatWidenLowHalf)
                wr("arg=%%%u is_unsigned=%s", ref(data.arg), data.is_unsigned ? "true" : "false");
            SIMJIT_MATCH (StepKind::PredicateNot) wr("arg=%%%u", ref(data));
            SIMJIT_MATCH (StepKind::MaskBinary)
                wr("op=%s left=%%%u right=%%%u", show_predicate_binary_op(data.op), ref(data.left), ref(data.right));
            SIMJIT_MATCH (StepKind::MaskCount) wr("arg=%%%u", ref(data));
            SIMJIT_MATCH (StepKind::VecReduce) wr("op=%s arg=%%%u", show_arith_binary_op(data.op), ref(data.arg));
            SIMJIT_MATCH (StepKind::MaskReduce) wr("op=%s arg=%%%u", show_predicate_binary_op(data.op), ref(data.arg));
            SIMJIT_MATCH (StepKind::MaskCombine) wr("left=%%%u right=%%%u", ref(data.left), ref(data.right));
            SIMJIT_MATCH (StepKind::Select)
                wr("cond=%%%u truthy=%%%u falsy=%%%u", ref(data.cond), ref(data.truthy), ref(data.falsy));
            SIMJIT_MATCH (StepKind::VecIndex) wr("inc=%%%u", ref(data.inc));
            SIMJIT_MATCH (StepKind::ScalarIndex) return;
            SIMJIT_MATCH (StepKind::Scatter) wr("idx=%%%u arg=%%%u", ref(data.idx), ref(data.arg));
            SIMJIT_MATCH (StepKind::CondScatter)
                wr("idx=%%%u arg=%%%u cond=%%%u", ref(data.idx), ref(data.arg), ref(data.cond));
            SIMJIT_MATCH (StepKind::Pack)
                wr("arg=%%%u cond=%%%u dst=%zu acc=$%lld:%llu", ref(data.arg), ref(data.cond), data.dst,
                   (long long)data.acc.group, (unsigned long long)data.acc.local);
            SIMJIT_MATCH (StepKind::CondStore)
                wr("arg=%%%u cond=%%%u dst=@%zu offset=%zu", ref(data.arg), ref(data.cond), data.addr.arg,
                   data.addr.offset);
            SIMJIT_MATCH (StepKind::Ternarylogic)
                wr("a=%%%u b=%%%u c=%%%u fn=%x", ref(data.a), ref(data.b), ref(data.c), data.fun);
            SIMJIT_MATCH (StepKind::VecConst) return;
            SIMJIT_MATCH (StepKind::VecPermute)
                wr("arg=%%%u permute_idxs=%%%u is_bit=%s", ref(data.arg), ref(data.permute_idxs),
                   data.is_bit ? "true" : "false");
            SIMJIT_MATCH (StepKind::ScalarPermute)
                wr("arg=%%%u permute_idxs=%llx is_bit=%s", ref(data.arg), (unsigned long long)data.permute,
                   data.is_bit ? "true" : "false");
            SIMJIT_MATCH (StepKind::StoreSum128) wr("dst=%zu", data.dst);
            SIMJIT_MATCH (StepKind::FMA)
                wr("kind=%s x1=%%%u x2=%%%u x3=%%%u", show_fma_kind(data.kind), ref(data.x1), ref(data.x2),
                   ref(data.x3));
            SIMJIT_MATCH (StepKind::BitCast) wr("arg=%%%u", ref(data));
            SIMJIT_MATCH (StepKind::Fpclass) wr("arg=%%%u flags=%s", ref(data.arg), show_fpclass(data.flags).c_str());
            SIMJIT_MATCH (StepKind::VecNarrowCombine) wr("low=%%%u high=%%%u", ref(data.low), ref(data.high));
            SIMJIT_MATCH (StepKind::VecFloatNarrowCombine) wr("low=%%%u high=%%%u", ref(data.low), ref(data.high));
            SIMJIT_MATCH (StepKind::ConstDiv) {
                std::string magic = data.magic != nullptr ? simjit::format("%%%u", ref(data.magic)) : "-";
                std::string round_mask =
                    data.round_mask != nullptr ? simjit::format("%%%u", ref(data.round_mask)) : "-";
                wr("op=%s numerator=%%%u magic=%s round_mask=%s divisor=%%%u shift=%u has_magic=%s has_add=%s "
                   "negative_divisor=%s",
                   show_arith_binary_op(data.op), ref(data.numerator), magic.c_str(), round_mask.c_str(),
                   ref(data.divisor), data.shift, data.has_magic ? "true" : "false", data.has_add ? "true" : "false",
                   data.negative_divisor ? "true" : "false");
            }
        }
#undef wr
#undef ref
        SIMJIT_UNREACHABLE();
    }

    void show(const Step *step) {
        SIMJIT_ASSERT(!show_cache[step->id]);
        uint32_t idx = show_cache[step->id] = counter++;
        simjit::format_to(buf, "%%%u <- ", idx);
        show_step(step);
        simjit::format_to(buf, "\n");
    }
};
} // namespace

std::string print_function(const Function *func) {
    ExprShowState show_state{};
    show_state.show_cache.resize(func->step_id_count, 0);
    show_state.buf.reserve(4096);
    auto show = [&](const Step *step) {
        if (show_state.show_cache[step->id] == 0) { show_state.show(step); }
    };
    if (!func->prologue_roots.empty()) {
        simjit::format_to(show_state.buf, "PROLOGUE\n");
        traverse_steps_postorder_unique(func->step_id_count, func->prologue_roots, show);
    }
    simjit::format_to(show_state.buf, "MAIN LOOP\n");
    traverse_steps_postorder_unique(func->step_id_count, func->main_loop_roots, show);

    simjit::format_to(show_state.buf, "REMAINDER\n");
    traverse_steps_postorder_unique(func->step_id_count, func->remainder_roots, show);

    if (!func->epilogue_roots.empty()) {
        simjit::format_to(show_state.buf, "EPILOGUE\n");
        traverse_steps_postorder_unique(func->step_id_count, func->epilogue_roots, show);
    }
    return show_state.buf;
}

namespace {
// Previously we had a lot of ad-hoc logic that tests target capabilities. They were refactored to this structure, which
// stores information about target architecture. Based on this information we tweak code gen a bit. This can be used to
// enable optimizations, and select instructions (this is necessary for correct codegen).
struct ArchTraits {
    bool has_ternarylogic = false;
    bool has_integer_fma = false;
    bool vec_shift_wraparound = false;
    bool vec_rotate_wraparound = false;
    bool has_unary_minus = false;
    bool has_float_not = false;
    bool has_float_abs = false;
    bool i8_gfni_shift = false;
    bool i8_ops = false;
    bool small_lzcnt = false;
    // false = backend has no refine, implement in MIR. true = backend handles, do nothing in MIR
    bool reciprocal_refine = false;
};
} // namespace

static ArchTraits get_arch_traits(Arch arch) {
    if (is_x86_arch(arch)) {
        ArchTraits traits;
        traits.has_ternarylogic = true;
        traits.has_integer_fma = false;
        traits.vec_rotate_wraparound = true;
        traits.vec_shift_wraparound = false;
        traits.has_unary_minus = false;
        traits.has_float_not = false;
        traits.has_float_abs = false;
        traits.i8_gfni_shift = true;
        traits.i8_ops = false;
        traits.small_lzcnt = false;
        traits.reciprocal_refine = false;
        return traits;
    }
    if (arch == Arch::Arm64_NEON) {
        ArchTraits traits;
        traits.has_ternarylogic = false;
        traits.has_integer_fma = true;
        traits.vec_rotate_wraparound = false;
        traits.vec_shift_wraparound = false;
        traits.has_unary_minus = true;
        traits.has_float_not = true;
        traits.has_float_abs = true;
        traits.i8_gfni_shift = false;
        traits.i8_ops = true;
        traits.small_lzcnt = true;
        traits.reciprocal_refine = true;
        return traits;
    }
    messed_up("unsupported arch %x", (unsigned)arch);
}

namespace {

AccId AccGroup::alloc_main(AccRemap *remap) {
    SIMJIT_ASSERT(slots_per_copy == 1);
    if (remap == nullptr) {
        if (main_count_ == 0) { ++main_count_; }
        return main(0);
    }
    if (auto it = remap->map.find(hir_idx); it != remap->map.end()) { return main(it->second); }
    uint32_t copy = main_count_++;
    remap->map[hir_idx] = copy;
    return main(copy);
}

std::pair<AccId, AccId> AccGroup::alloc_main_pair(AccRemap *remap) {
    SIMJIT_ASSERT(slots_per_copy == 2);
    if (remap == nullptr) {
        if (main_count_ == 0) { ++main_count_; }
        return {main(0, 0), main(0, 1)};
    }
    if (auto it = remap->map.find(hir_idx); it != remap->map.end()) {
        return {main(it->second, 0), main(it->second, 1)};
    }
    uint32_t copy = main_count_++;
    remap->map[hir_idx] = copy;
    return {main(copy, 0), main(copy, 1)};
}

AccId AccGroup::alloc_remainder() {
    SIMJIT_ASSERT(slots_per_copy == 1 && !remainder_allocated_);
    remainder_allocated_ = true;
    return remainder();
}

AccId AccGroup::singleton() {
    SIMJIT_ASSERT(slots_per_copy == 1 && main_count_ == 0);
    remainder_allocated_ = true;
    return remainder();
}

std::pair<AccId, AccId> AccGroup::alloc_remainder_pair() {
    SIMJIT_ASSERT(slots_per_copy == 2 && !remainder_allocated_);
    remainder_allocated_ = true;
    return {remainder(0), remainder(1)};
}

struct VectLoweringContext {
    vect::RowBlock rows{};
    AccRemap *remap = nullptr;
};

enum class TernarylogicRpnOp : uint8_t {
    Leaf0,
    Leaf1,
    Leaf2,
    Not,
    And,
    Or,
    Xor,
    AndNot,
};

struct TernarylogicRpn {
    static constexpr size_t MaxOps = 15;

    const vect::Node *leaves[3]{};
    uint8_t leaf_count = 0;
    uint8_t op_count = 0;
    uint8_t code_size = 0;
    TernarylogicRpnOp code[MaxOps]{};
};

static VecDataType widened_half_dtype(VecDataType dtype, VecElemType elem) {
    VecDataType wide_dtype = dtype;
    wide_dtype.elem = elem;
    auto result = vec_dtype_half(wide_dtype);
    if (!result) { messed_up("can't widen half of %s", show_vec_dtype(dtype)); }
    return *result;
}

static ConstData widen_small_divisor(ConstData value, ScalarDataType from, bool is_signed) noexcept {
    SIMJIT_ASSERT(from == ScalarDataType::I8 || from == ScalarDataType::I16);
    if (is_signed) {
        if (from == ScalarDataType::I8) { return ConstData::i32(value.retag(from).as_i8()); }
        return ConstData::i32(value.retag(from).as_i16());
    }
    if (from == ScalarDataType::I8) { return ConstData::u32(value.retag(from).as_u8()); }
    return ConstData::u32(value.retag(from).as_u16());
}

class VectMirCache {
public:
    void init(MemoryArena *arena, const vect::Function *func) {
        shared_ = arena->alloc_array<Step *>(func->node_id_count);
        std::fill(shared_.begin(), shared_.end(), nullptr);
        shareable_ = ArenaBitmap::create(arena, func->node_id_count);
        ArenaBitmap visited = ArenaBitmap::create(arena, func->node_id_count);
        for (const vect::Root &root : func->roots) {
            classify_shareability(root.node, visited);
        }
    }

    Step *lookup(const vect::Node *node, vect::RowBlock rows) const {
        if (has_accumulator_effect(node)) { return nullptr; }
        if (shareable_.get(node->id)) { return shared_[node->id]; }
        auto it = contextual_.find(ContextualMirCacheKey{node->id, rows});
        return it == contextual_.end() ? nullptr : it->second;
    }

    void store(const vect::Node *node, vect::RowBlock rows, Step *step) {
        if (has_accumulator_effect(node)) { return; }
        if (shareable_.get(node->id)) {
            SIMJIT_ASSERT(shared_[node->id] == nullptr);
            shared_[node->id] = step;
            return;
        }
        auto result = contextual_.emplace(ContextualMirCacheKey{node->id, rows}, step);
        SIMJIT_ASSERT(result.second);
    }

private:
    static bool step_kind_shareable_if_children_shareable(hir::StepKind kind) noexcept {
        switch (kind) {
        case hir::StepKind::Const:
        case hir::StepKind::LoadSplat: return true;
        case hir::StepKind::ArithBinary:
        case hir::StepKind::ArithUnary:
        case hir::StepKind::Select:
        case hir::StepKind::Compare:
        case hir::StepKind::PredicateNot:
        case hir::StepKind::IntCast:
        case hir::StepKind::FloatCast:
        case hir::StepKind::BitCast:
        case hir::StepKind::Fpclass:
        case hir::StepKind::PredicateBinary:
        case hir::StepKind::Permute: return true;
        default: return false;
        }
    }

    static bool has_accumulator_effect(const vect::Node *node) noexcept {
        if (!node->is(vect::NodeKind::Step)) { return false; }
        switch (node->step->kind) {
        case hir::StepKind::AccArithBinary:
        case hir::StepKind::AccPredicateBinary:
        case hir::StepKind::AccSum128:
        case hir::StepKind::Countif: return true;
        case hir::StepKind::CheckedOp: return true;
        default: return false;
        }
    }

    bool classify_shareability(const vect::Node *node, ArenaBitmap &visited) noexcept {
        if (visited.get(node->id)) { return shareable_.get(node->id); }
        visited.set(node->id);

        bool shareable = false;
        if (node->is(vect::NodeKind::Step)) {
            shareable = step_kind_shareable_if_children_shareable(node->step->kind);
        } else if (node->is(vect::NodeKind::CastDirect)) {
            shareable = true;
        }
        for (const vect::Node *child : node->children_span()) {
            shareable = shareable && classify_shareability(child, visited);
        }
        shareable_.set(node->id, shareable);
        return shareable;
    }

    ArenaArray<Step *> shared_{};
    std::unordered_map<ContextualMirCacheKey, Step *> contextual_{};
    ArenaBitmap shareable_{};
};

struct MirConstructState {
    StepMaker sm;
    const hir::Function *hir = nullptr;
    const vect::Function *vect_result = nullptr;
    std::optional<ArgumentIdx> safety_check_arg;
    VectMirCache vect_mir_cache{};
    std::vector<size_t> vect_node_ref_counts{};
    std::unordered_map<ConstantKey, Step *> constants{};
    std::unordered_map<ConstQuotientKey, Step *, ConstQuotientKeyHash> const_quotients{};
    std::vector<SafetyCheckInfo> safety_checks{};
    std::vector<Step *> checked_steps{};
    std::optional<AccId> saved_scalar_safety_check_acc{};
    std::vector<AccGroup> acc_groups{};
    uint32_t special_acc_count = 0;
    std::vector<Step *> prologue_roots{};
    std::vector<Step *> main_loop_roots{};
    std::vector<Step *> remainder_roots{};
    std::vector<Step *> epilogue_roots{};
    size_t loop_width = 0;
    ArchTraits arch_traits;

    MirConstructState() = delete;
    explicit MirConstructState(Context *ctx) : sm(ctx->arena), arch_traits(get_arch_traits(ctx->arch)) {}

    void count_vectorizer_node_refs_rec(const vect::Node *node, ArenaBitmap &visited) noexcept {
        if (visited.get(node->id)) { return; }
        visited.set(node->id);
        for (size_t i = 0; i < node->child_count; ++i) {
            const vect::Node *child = node->children[i];
            ++vect_node_ref_counts[child->id];
            count_vectorizer_node_refs_rec(child, visited);
        }
    }

    void count_vectorizer_node_refs() {
        vect_node_ref_counts.assign(vect_result->node_id_count, 0);
        ArenaBitmap visited = ArenaBitmap::create(hir->ctx->arena, vect_result->node_id_count);
        for (const vect::Root &root : vect_result->roots) {
            ++vect_node_ref_counts[root.node->id];
            count_vectorizer_node_refs_rec(root.node, visited);
        }
    }

    Step *make_const(ConstData x, DataType step_dt) {
        x = x.retag(const_storage_dtype(step_dt));
        ConstantKey key{x, step_dt};
        if (auto it = constants.find(key); it != constants.end()) { return it->second; }

        Step *s = sm.con(x, step_dt);
        constants[key] = s;
        return s;
    }

    Step *add_prologue_const(ConstData x, DataType step_dt) {
        Step *s = make_const(x, step_dt);
        if (std::find(prologue_roots.begin(), prologue_roots.end(), s) == prologue_roots.end()) {
            prologue_roots.push_back(s);
        }
        return s;
    }

    struct OnlyInt {
        int64_t v;
        OnlyInt(int x) noexcept : v(x) {}
        OnlyInt(long x) noexcept : v(x) {}
        OnlyInt(long long x) noexcept : v(x) {}
        OnlyInt(unsigned long x) noexcept : v((int64_t)(long)x) {}
        OnlyInt(unsigned long long x) noexcept : v((int64_t)(long long)x) {}
        OnlyInt(float x) = delete;
        OnlyInt(double x) = delete;
    };
    Step *make_const(OnlyInt x, DataType step_dt) { return make_const(ConstData::i64(x.v), step_dt); }

    Step *lower_reciprocal(Step *arg, ArithUnaryOp op, DataType dtype) {
        SIMJIT_ASSERT(op == ArithUnaryOp::Rcp || op == ArithUnaryOp::Rsqrt);
        ScalarDataType scalar_dtype = dtype.is_scalar() ? dtype.as_scalar() : dtype.as_vec().to_scalar();
        SIMJIT_ASSERT(is_float_dtype(scalar_dtype));

        Step *result = sm.arith_un({arg, op}, dtype);
        if (arch_traits.reciprocal_refine) { return result; }

        Step *one = make_const(scalar_dtype == ScalarDataType::F32 ? ConstData::f32(1.0f) : ConstData::f64(1.0), dtype);
        Step *x = arg;
        Step *y = result;

        if (op == ArithUnaryOp::Rcp) {
            auto refine_once = [&] {
                // y' = y + y * (1 - x * y)
                Step *r = sm.fma({x, y, one, FmaKind::FNMA}, dtype);
                y = sm.fma({y, r, y, FmaKind::FMA}, dtype);
                return;
            };
            // x86-64 estimates provide 14 bits. One iteration recovers full F32 precision. F64 needs two.
            refine_once();
            if (scalar_dtype == ScalarDataType::F64) { refine_once(); }
        } else {
            auto refine_once = [&] {
                // y' = y + 1/2 * y * (1 - x * y*y)
                Step *half =
                    make_const(scalar_dtype == ScalarDataType::F32 ? ConstData::f32(0.5f) : ConstData::f64(0.5), dtype);
                Step *half_y = sm.arith_bin({y, half, ArithBinaryOp::Mul}, dtype);
                Step *y_squared = sm.arith_bin({y, y, ArithBinaryOp::Mul}, dtype);
                Step *r = sm.fma({x, y_squared, one, FmaKind::FNMA}, dtype);
                y = sm.fma({half_y, r, y, FmaKind::FMA}, dtype);
            };
            // same logic as for rcp
            refine_once();
            if (scalar_dtype == ScalarDataType::F64) { refine_once(); }
        }

        return y;
    }

    Step *make_tzcnt(Step *arg, DataType dtype) {
        Step *below = sm.arith_bin({arg, make_const(1, dtype), ArithBinaryOp::Sub}, dtype);
        Step *low_bits = sm.arith_bin({arg, below, ArithBinaryOp::AndNot}, dtype);
        return sm.arith_un({low_bits, ArithUnaryOp::Popcount}, dtype);
    }

    Step *make_narrow_combine(Step *low, Step *high, VecDataType dtype) {
        invariant_same_vec_type(low, high);
        VecDataType wide_dtype = low->dtype.as_vec();
        invariantm(wide_dtype.nelems() * 2 == dtype.nelems() &&
                       wide_dtype.element_size_bytes() == dtype.element_size_bytes() * 2,
                   "invalid narrow-combine from %s to %s", show_vec_dtype(wide_dtype), show_vec_dtype(dtype));
        return sm.vec_narrow_combine({low, high}, dtype);
    }

#if SIMJIT_USE_LIBDIVIDE
    Step *mulhi_u16_part(Step *left32, uint16_t magic) {
        VecDataType wide_dtype = left32->dtype.as_vec();
        Step *right32 = make_const(ConstData::u64(magic), wide_dtype);
        Step *prod32 = sm.arith_bin({left32, right32, ArithBinaryOp::Mul}, wide_dtype);
        return sm.arith_bin({prod32, make_const(16, wide_dtype), ArithBinaryOp::ShiftRightLogical}, wide_dtype);
    }

    Step *mulhi_u16(Step *left, uint16_t magic) {
        VecDataType dtype = left->dtype.as_vec();
        VecDataType wide_dtype = widened_half_dtype(dtype, VecElemType::I32);
        Step *low = sm.widen_lo({left, true}, wide_dtype);
        Step *high = sm.widen_hi({left, true}, wide_dtype);
        return make_narrow_combine(mulhi_u16_part(low, magic), mulhi_u16_part(high, magic), dtype);
    }

    Step *mulhi_u32_part(Step *left64, uint32_t magic) {
        VecDataType wide_dtype = left64->dtype.as_vec();
        Step *right64 = make_const(ConstData::u64(magic), wide_dtype);
        Step *prod64 = sm.arith_bin({left64, right64, ArithBinaryOp::Mul64ZE}, wide_dtype);
        return sm.arith_bin({prod64, make_const(32, wide_dtype), ArithBinaryOp::ShiftRightLogical}, wide_dtype);
    }

    Step *mulhi_u32(Step *left, uint32_t magic) {
        VecDataType dtype = left->dtype.as_vec();
        VecDataType wide_dtype = widened_half_dtype(dtype, VecElemType::I64);
        Step *low = sm.widen_lo({left, true}, wide_dtype);
        Step *high = sm.widen_hi({left, true}, wide_dtype);
        return make_narrow_combine(mulhi_u32_part(low, magic), mulhi_u32_part(high, magic), dtype);
    }

    Step *mulhi_u64(Step *left, uint64_t magic) {
        VecDataType dtype = left->dtype.as_vec();
        Step *mask32 = make_const(ConstData::u64(0xffffffffull), dtype);
        Step *right_lo = make_const(ConstData::u64(magic & 0xffffffffull), dtype);
        Step *right_hi = make_const(ConstData::u64(magic >> 32), dtype);

        Step *left_lo = sm.arith_bin({left, mask32, ArithBinaryOp::And}, dtype);
        Step *left_hi = sm.arith_bin({left, make_const(32, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);

        Step *x0y0 = sm.arith_bin({left_lo, right_lo, ArithBinaryOp::Mul64ZE}, dtype);
        Step *x0y0_hi = sm.arith_bin({x0y0, make_const(32, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
        Step *x0y1 = sm.arith_bin({left_lo, right_hi, ArithBinaryOp::Mul64ZE}, dtype);
        Step *x1y0 = sm.arith_bin({left_hi, right_lo, ArithBinaryOp::Mul64ZE}, dtype);
        Step *x1y1 = sm.arith_bin({left_hi, right_hi, ArithBinaryOp::Mul64ZE}, dtype);

        Step *temp = sm.arith_bin({x1y0, x0y0_hi, ArithBinaryOp::Add}, dtype);
        Step *temp_lo = sm.arith_bin({temp, mask32, ArithBinaryOp::And}, dtype);
        Step *temp_hi = sm.arith_bin({temp, make_const(32, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
        Step *carry = sm.arith_bin({temp_lo, x0y1, ArithBinaryOp::Add}, dtype);
        Step *carry_hi = sm.arith_bin({carry, make_const(32, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);

        Step *sum = sm.arith_bin({x1y1, temp_hi, ArithBinaryOp::Add}, dtype);
        return sm.arith_bin({sum, carry_hi, ArithBinaryOp::Add}, dtype);
    }

    Step *udiv_const_peephole(Step *left, uint64_t rhs) {
        VecDataType dtype = left->dtype.as_vec();
        switch (dtype.to_scalar()) {
        case ScalarDataType::I16: {
            auto div = ::libdivide::libdivide_u16_gen((uint16_t)rhs);
            if (div.magic == 0) {
                return sm.arith_bin({left, make_const(div.more, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
            }
            Step *q = mulhi_u16(left, div.magic);
            if (div.more & ::libdivide::LIBDIVIDE_ADD_MARKER) {
                Step *sub = sm.arith_bin({left, q, ArithBinaryOp::Sub}, dtype);
                Step *half = sm.arith_bin({sub, make_const(1, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
                Step *t = sm.arith_bin({half, q, ArithBinaryOp::Add}, dtype);
                return sm.arith_bin({t, make_const(div.more & 0x0f, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
            }
            return sm.arith_bin({q, make_const(div.more, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
        }
        case ScalarDataType::I32: {
            auto div = ::libdivide::libdivide_u32_gen((uint32_t)rhs);
            if (div.magic == 0) {
                return sm.arith_bin({left, make_const(div.more, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
            }
            Step *q = mulhi_u32(left, div.magic);
            if (div.more & ::libdivide::LIBDIVIDE_ADD_MARKER) {
                Step *sub = sm.arith_bin({left, q, ArithBinaryOp::Sub}, dtype);
                Step *half = sm.arith_bin({sub, make_const(1, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
                Step *t = sm.arith_bin({half, q, ArithBinaryOp::Add}, dtype);
                return sm.arith_bin({t, make_const(div.more & 0x1f, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
            }
            return sm.arith_bin({q, make_const(div.more, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
        }
        case ScalarDataType::I64: {
            auto div = ::libdivide::libdivide_u64_gen(rhs);
            if (div.magic == 0) {
                return sm.arith_bin({left, make_const(div.more, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
            }
            Step *q = mulhi_u64(left, div.magic);
            if (div.more & ::libdivide::LIBDIVIDE_ADD_MARKER) {
                Step *sub = sm.arith_bin({left, q, ArithBinaryOp::Sub}, dtype);
                Step *half = sm.arith_bin({sub, make_const(1, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
                Step *t = sm.arith_bin({half, q, ArithBinaryOp::Add}, dtype);
                return sm.arith_bin({t, make_const(div.more & 0x3f, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
            }
            return sm.arith_bin({q, make_const(div.more, dtype), ArithBinaryOp::ShiftRightLogical}, dtype);
        }
        default: break;
        }
        return nullptr;
    }

    Step *sdiv_const_peephole(Step *left, int64_t rhs) {
        VecDataType dtype = left->dtype.as_vec();
        // Signed division is lowered through unsigned division of the absolute numerator:
        //   sign_x = x >> (bits - 1)
        //   q_abs = udiv(abs(x), abs(rhs))
        //   sign_q = rhs < 0 ? ~sign_x : sign_x
        //   q = (q_abs ^ sign_q) - sign_q
        size_t sign_shift = scalar_dtype_bits(dtype.to_scalar()) - 1;
        Step *sign_x = sm.arith_bin({left, make_const(sign_shift, dtype), ArithBinaryOp::ShiftRightArith}, dtype);
        Step *abs_x = sm.arith_un({left, ArithUnaryOp::Abs}, dtype);
        uint64_t rhs_abs = rhs < 0 ? (uint64_t)(-(uint64_t)rhs) : (uint64_t)rhs;
        Step *q_abs = udiv_const_peephole(abs_x, rhs_abs);
        if (q_abs == nullptr) return nullptr;

        Step *sign_q = sign_x;
        if (rhs < 0) {
            Step *sign_d = make_const(ConstData::u64(~0ull), dtype);
            sign_q = sm.arith_bin({sign_x, sign_d, ArithBinaryOp::Xor}, dtype);
        }
        Step *signed_flipped = sm.arith_bin({q_abs, sign_q, ArithBinaryOp::Xor}, dtype);
        return sm.arith_bin({signed_flipped, sign_q, ArithBinaryOp::Sub}, dtype);
    }

    Step *vector_const_quotient_peephole(Step *left, ConstData right, bool is_signed) {
        VecDataType dtype = left->dtype.as_vec();
        ScalarDataType sdtype = dtype.to_scalar();
        if (sdtype != ScalarDataType::I16 && sdtype != ScalarDataType::I32 && sdtype != ScalarDataType::I64) {
            return nullptr;
        }
        ConstData divisor = right.retag(sdtype);
        ConstQuotientKey key{left, divisor, is_signed};
        if (auto it = const_quotients.find(key); it != const_quotients.end()) { return it->second; }

        Step *q = nullptr;
        if (is_signed) {
            int64_t rhs = divisor.as_signed();
            if (rhs == 0) return nullptr;
            q = sdiv_const_peephole(left, rhs);
        } else {
            uint64_t rhs = divisor.as_unsigned();
            if (rhs == 0) return nullptr;
            q = udiv_const_peephole(left, rhs);
        }
        if (q != nullptr) { const_quotients.emplace(key, q); }
        return q;
    }

    Step *div_peephole(Step *left, ConstData right, ArithBinaryOp op) {
        bool is_signed = op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod;
        bool is_mod = op == ArithBinaryOp::Mod || op == ArithBinaryOp::UMod;
        Step *q = vector_const_quotient_peephole(left, right, is_signed);
        if (q == nullptr || !is_mod) { return q; }

        VecDataType dtype = left->dtype.as_vec();
        ConstData divisor = right.retag(dtype.to_scalar());
        Step *prod = sm.arith_bin({q, make_const(divisor, dtype), ArithBinaryOp::Mul}, dtype);
        return sm.arith_bin({left, prod, ArithBinaryOp::Sub}, dtype);
    }

    Step *scalar_wide_const_quotient_peephole(Step *left, ConstData right, bool is_signed) {
        ScalarDataType sdtype = left->dtype.as_scalar();
        if (sdtype != ScalarDataType::I32 && sdtype != ScalarDataType::I64) { return nullptr; }

        uint8_t shift = 0;
        bool has_magic = false;
        bool has_add = false;
        bool negative_divisor = false;
        ConstData magic{}; // initialized below
        ConstData round_mask = ConstData::u64(0);
        ConstData divisor{}; // initialized below

        if (is_signed) {
            int64_t rhs = right.retag(sdtype).as_signed();
            if (rhs == 0) return nullptr;
            if (sdtype == ScalarDataType::I32) {
                auto div = ::libdivide::libdivide_s32_gen((int32_t)rhs);
                shift = div.more & ::libdivide::LIBDIVIDE_32_SHIFT_MASK;
                has_magic = div.magic != 0;
                has_add = div.more & ::libdivide::LIBDIVIDE_ADD_MARKER;
                negative_divisor = div.more & ::libdivide::LIBDIVIDE_NEGATIVE_DIVISOR;
                magic = ConstData::i64(div.magic);
            } else {
                auto div = ::libdivide::libdivide_s64_gen(rhs);
                shift = div.more & ::libdivide::LIBDIVIDE_64_SHIFT_MASK;
                has_magic = div.magic != 0;
                has_add = div.more & ::libdivide::LIBDIVIDE_ADD_MARKER;
                negative_divisor = div.more & ::libdivide::LIBDIVIDE_NEGATIVE_DIVISOR;
                magic = ConstData::i64(div.magic);
            }
            round_mask = ConstData::u64(shift == 0 ? 0 : ((uint64_t)1 << shift) - 1);
            divisor = ConstData::i64(rhs);
        } else {
            uint64_t rhs = right.retag(sdtype).as_unsigned();
            if (rhs == 0) return nullptr;
            if (sdtype == ScalarDataType::I32) {
                auto div = ::libdivide::libdivide_u32_gen((uint32_t)rhs);
                shift = div.more & ::libdivide::LIBDIVIDE_32_SHIFT_MASK;
                has_magic = div.magic != 0;
                has_add = div.more & ::libdivide::LIBDIVIDE_ADD_MARKER;
                magic = ConstData::u64(div.magic);
                divisor = ConstData::u64((uint32_t)rhs);
            } else {
                auto div = ::libdivide::libdivide_u64_gen(rhs);
                shift = div.more & ::libdivide::LIBDIVIDE_64_SHIFT_MASK;
                has_magic = div.magic != 0;
                has_add = div.more & ::libdivide::LIBDIVIDE_ADD_MARKER;
                magic = ConstData::u64(div.magic);
                divisor = ConstData::u64(rhs);
            }
        }

        Step *magic_step = has_magic ? add_prologue_const(magic, sdtype) : nullptr;
        Step *round_mask_step = is_signed && !has_magic ? add_prologue_const(round_mask, sdtype) : nullptr;
        Step *divisor_step = make_const(divisor, sdtype);
        ArithBinaryOp op = is_signed ? ArithBinaryOp::Div : ArithBinaryOp::UDiv;
        return sm.const_div(
            {left, magic_step, round_mask_step, divisor_step, op, shift, has_magic, has_add, negative_divisor}, sdtype);
    }

    Step *scalar_const_div_peephole(Step *left, ConstData right, ArithBinaryOp op) {
        ScalarDataType sdtype = left->dtype.as_scalar();
        if (!is_simple_int_dtype(sdtype)) { return nullptr; }

        bool is_signed = op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod;
        bool is_mod = op == ArithBinaryOp::Mod || op == ArithBinaryOp::UMod;
        ConstData divisor = right.retag(sdtype);
        if ((is_signed && divisor.as_signed() == 0) || (!is_signed && divisor.as_unsigned() == 0)) { return nullptr; }

        ConstQuotientKey key{left, divisor, is_signed};
        Step *q = nullptr;
        if (auto it = const_quotients.find(key); it != const_quotients.end()) {
            q = it->second;
        } else if (sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) {
            Step *wide_left =
                sm.int_cast({left, is_signed ? IntCastKind::Sext : IntCastKind::Zext}, ScalarDataType::I32);
            ConstData wide_divisor = widen_small_divisor(divisor, sdtype, is_signed);
            Step *wide_q = scalar_wide_const_quotient_peephole(wide_left, wide_divisor, is_signed);
            if (wide_q != nullptr) { q = sm.int_cast({wide_q, IntCastKind::Trunc}, sdtype); }
        } else {
            q = scalar_wide_const_quotient_peephole(left, divisor, is_signed);
        }
        if (q == nullptr) { return nullptr; }
        const_quotients.emplace(key, q);
        if (!is_mod) { return q; }

        if (sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) {
            IntCastKind cast = is_signed ? IntCastKind::Sext : IntCastKind::Zext;
            Step *wide_left = sm.int_cast({left, cast}, ScalarDataType::I32);
            Step *wide_q = sm.int_cast({q, cast}, ScalarDataType::I32);
            Step *wide_divisor = make_const(widen_small_divisor(divisor, sdtype, is_signed), ScalarDataType::I32);
            Step *prod = sm.arith_bin({wide_q, wide_divisor, ArithBinaryOp::Mul}, ScalarDataType::I32);
            Step *remainder = sm.arith_bin({wide_left, prod, ArithBinaryOp::Sub}, ScalarDataType::I32);
            return sm.int_cast({remainder, IntCastKind::Trunc}, sdtype);
        }

        Step *prod = sm.arith_bin({q, make_const(divisor, sdtype), ArithBinaryOp::Mul}, sdtype);
        return sm.arith_bin({left, prod, ArithBinaryOp::Sub}, sdtype);
    }
#endif

    // FIXME: This takes surprisingly large time. See if we can do it on our own without separate step.
    void loop_invariant_code_motion() {
        auto move_const = [&](Step *x) {
            if ((x->is(StepKind::Const) && !x->dtype.is_scalar()) || x->is(StepKind::LoadSplat) ||
                x->is(StepKind::VecConst)) {
                prologue_roots.push_back(x);
            }
        };
        traverse_steps_postorder_unique(sm.max_id(), nonstd::span<mir::Step *const>{main_loop_roots}, move_const);
    }

    // In the end of aggregation we might have several accumulators that we still need to reduce into one. For this we
    // use tree of adders, which is implemented using this functions. Using tree of adders ((a + b) + (c + d)) instead
    // of 1 accumulator (((a + b) + c) + d) provides better hardware parallelisation opportunities.
    Step *fold_arith_accs(const AccGroup &group, uint32_t count, uint32_t component, DataType dtype,
                          ArithBinaryOp fold_op) {
        SIMJIT_ASSERT(count != 0);
        std::vector<Step *> values;
        values.reserve(count);
        for (uint32_t copy = 0; copy < count; ++copy) {
            values.push_back(sm.acc_load(group.main(copy, component), dtype));
        }
        size_t active = values.size();
        while (active > 1) {
            size_t output = 0;
            for (size_t input = 0; input + 1 < active; input += 2) {
                Step *left = values[input];
                Step *right = values[input + 1];
                values[output++] = dtype.is_vec() ? make_vec_arith_bin(left, right, fold_op, dtype.as_vec())
                                                  : sm.arith_bin({left, right, fold_op}, dtype);
            }
            if (active % 2 != 0) { values[output++] = values[active - 1]; }
            active = output;
        }
        return values[0];
    }

    void collapse_arith_accs(const AccGroup &group, ArithBinaryOp op) {
        const ArgumentDecl &arg = hir->args[group.dst_arg];
        ScalarDataType dtype = arg.dtype;
        auto combine = [&](Step *left, Step *right) {
            if ((dtype == ScalarDataType::I8 || dtype == ScalarDataType::I16) && op == ArithBinaryOp::Mul) {
                return small_size_arith_binary(left, right, op);
            }
            return sm.arith_bin({left, right, op}, dtype);
        };

        Step *result = nullptr;
        if (group.main_count() != 0) {
            DataType it_dtype = vect_result == nullptr ? dtype : vect_result->acc_dtypes[group.hir_idx];
            result = fold_arith_accs(group, group.main_count(), 0, it_dtype, op);
            if (it_dtype.is_vec()) {
                // It does not make sense to implement reduces for small types. However, we still need to handle them
                // for completeness. Do this by reusing 32-bit variants by zero-extending arguments to 32 bit and then
                // truncating. For this we use widen_lo and widen_hi, which perform cast (zero or sign extend) using
                // same register length. We can't change register length here, so it is our only option.
                VecDataType vdtype = it_dtype.as_vec();
                bool zext = op != ArithBinaryOp::Min && op != ArithBinaryOp::Max;
                if (vdtype.elem == VecElemType::I8) {
                    VecDataType widened1 = vdtype;
                    widened1.elem = VecElemType::I16;
                    auto maybe_widened1 = vec_dtype_half(widened1);
                    SIMJIT_ASSERT(maybe_widened1.has_value());
                    widened1 = *maybe_widened1;
                    VecDataType widened2 = widened1;
                    widened2.elem = VecElemType::I32;
                    auto maybe_widened2 = vec_dtype_half(widened2);
                    SIMJIT_ASSERT(maybe_widened2.has_value());
                    widened2 = *maybe_widened2;
                    Step *low = sm.widen_lo({result, zext}, widened1);
                    Step *high = sm.widen_hi({result, zext}, widened1);
                    Step *low_low = sm.widen_lo({low, zext}, widened2);
                    Step *low_high = sm.widen_hi({low, zext}, widened2);
                    Step *high_low = sm.widen_lo({high, zext}, widened2);
                    Step *high_high = sm.widen_hi({high, zext}, widened2);
                    low_low = sm.vec_reduce({low_low, op}, ScalarDataType::I32);
                    low_high = sm.vec_reduce({low_high, op}, ScalarDataType::I32);
                    high_low = sm.vec_reduce({high_low, op}, ScalarDataType::I32);
                    high_high = sm.vec_reduce({high_high, op}, ScalarDataType::I32);
                    low = sm.arith_bin({low_low, low_high, op}, ScalarDataType::I32);
                    high = sm.arith_bin({high_low, high_high, op}, ScalarDataType::I32);
                    result = sm.arith_bin({low, high, op}, ScalarDataType::I32);
                    result = sm.int_cast({result, IntCastKind::Trunc}, dtype);
                } else if (vdtype.elem == VecElemType::I16) {
                    VecDataType widened = vdtype;
                    widened.elem = VecElemType::I32;
                    auto maybe_widened = vec_dtype_half(widened);
                    SIMJIT_ASSERT(maybe_widened.has_value());
                    widened = *maybe_widened;
                    Step *low = sm.widen_lo({result, zext}, widened);
                    Step *high = sm.widen_hi({result, zext}, widened);
                    low = sm.vec_reduce({low, op}, ScalarDataType::I32);
                    high = sm.vec_reduce({high, op}, ScalarDataType::I32);
                    result = sm.arith_bin({low, high, op}, ScalarDataType::I32);
                    result = sm.int_cast({result, IntCastKind::Trunc}, dtype);
                } else {
                    result = sm.vec_reduce({result, op}, dtype);
                }
            }
        }
        Step *remainder_result = sm.acc_load(group.remainder(), dtype);
        if (result != nullptr) {
            result = combine(result, remainder_result);
        } else {
            result = remainder_result;
        }
        SIMJIT_ASSERT(result);
        epilogue_roots.push_back(sm.agg_result({result, arg.idx}, dtype));
    }

    // Similarly to fold_arith_accs, we fold mask accumulators using tree approach.
    Step *fold_predicate_accs(const AccGroup &group, uint32_t count, MaskDataType dtype, PredicateBinaryOp fold_op) {
        SIMJIT_ASSERT(count != 0);
        std::vector<Step *> values;
        values.reserve(count);
        for (uint32_t copy = 0; copy < count; ++copy) {
            values.push_back(sm.acc_load(group.main(copy), dtype));
        }
        size_t active = values.size();
        while (active > 1) {
            size_t output = 0;
            for (size_t input = 0; input + 1 < active; input += 2) {
                values[output++] = sm.mask_bin({values[input], values[input + 1], fold_op}, dtype);
            }
            if (active % 2 != 0) { values[output++] = values[active - 1]; }
            active = output;
        }
        return values[0];
    }

    void collapse_predicate_accs(const AccGroup &group, PredicateBinaryOp predicate_op, ArithBinaryOp fold_op) {
        const ArgumentDecl &arg = hir->args[group.dst_arg];
        ScalarDataType dt = ScalarDataType::I8;

        Step *result = nullptr;
        if (group.main_count() != 0) {
            MaskDataType mask_dtype = vect_result->acc_dtypes[group.hir_idx].as_mask();
            result = fold_predicate_accs(group, group.main_count(), mask_dtype, predicate_op);
            result = sm.mask_reduce({result, predicate_op}, dt);
        }
        Step *remainder_result = sm.acc_load(group.remainder(), dt);
        if (result != nullptr) {
            result = sm.arith_bin({result, remainder_result, fold_op}, dt);
        } else {
            result = remainder_result;
        }
        SIMJIT_ASSERT(result);
        epilogue_roots.push_back(sm.agg_result({result, arg.idx}, dt));
    }

    // This functions is used to construct code to output aggregate results. Since we might have several instances that
    // belong to one aggregate due to unrolling and remaineder code, we 'collapse' them first.
    void gen_agg_results() {
        for (const AccGroup &group : acc_groups) {
            const hir::Step *agg_expr = group.agg_expr;
            if (group.kind == AccKind::Sum128) {
                Step *hi = nullptr;
                if (group.main_count() != 0) {
                    VecDataType vdtype = vect_result->acc_dtypes[group.hir_idx].as_vec();
                    hi = fold_arith_accs(group, group.main_count(), 1, vdtype, ArithBinaryOp::Add);
                    hi = sm.vec_reduce({hi, ArithBinaryOp::Add}, ScalarDataType::I64);
                }
                Step *tmp = sm.acc_load(group.remainder(1), ScalarDataType::I64);
                if (hi != nullptr) {
                    hi = sm.arith_bin({hi, tmp, ArithBinaryOp::Add}, ScalarDataType::I64);
                } else {
                    hi = tmp;
                }
                std::vector<Step *> all_low{};
                for (uint32_t copy = 0; copy < group.main_count(); ++copy) {
                    Step *s = sm.acc_load(group.main(copy, 0), vect_result->acc_dtypes[group.hir_idx]);
                    all_low.push_back(s);
                }
                Step *s = sm.acc_load(group.remainder(0), ScalarDataType::I64);
                all_low.push_back(s);
                MemoryArena *arena = hir->ctx->arena;
                ArenaArray<Step *> all_low_arr = arena->copy_array<Step *>(all_low);

                Step *combined = sm.sum128({all_low_arr, hi, group.dst_arg}, ScalarDataType::I128);
                epilogue_roots.push_back(combined);
            } else if (agg_expr == nullptr) {
            } else if (agg_expr->is(hir::StepKind::AccArithBinary)) {
                ArithBinaryOp op = agg_expr->step_data<hir::StepKind::AccArithBinary>().op;
                // Andnot is applied on arguments, accumulators are combined with and
                if (op == ArithBinaryOp::AndNot) { op = ArithBinaryOp::And; }
                collapse_arith_accs(group, op);
            } else if (agg_expr->is(hir::StepKind::Countif)) {
                collapse_arith_accs(group, ArithBinaryOp::Add);
            } else if (agg_expr->is(hir::StepKind::AccPredicateBinary)) {
                auto op = agg_expr->step_data<hir::StepKind::AccPredicateBinary>().op;
                // Andnot is applied on arguments, accumulators are combined with and
                if (op == PredicateBinaryOp::AndNot) { op = PredicateBinaryOp::And; }
                auto maybe_arith_op = arith_op_from_predicate(op);
                if (!maybe_arith_op) messed_up("got invalid op %s in agg", show_predicate_binary_op(op));
                collapse_predicate_accs(group, op, *maybe_arith_op);
            } else if (agg_expr->is(hir::StepKind::Pack)) {
                SIMJIT_ASSERT(group.main_count() == 0);
                Step *acc = sm.acc_load(group.remainder(), ScalarDataType::I64);
                epilogue_roots.push_back(sm.agg_result({acc, group.dst_arg}, ScalarDataType::I64));
            } else {
                SIMJIT_ASSERT(0);
            }
        }

        if (!safety_checks.empty()) {
            Step *combined = nullptr;
            for (const SafetyCheckInfo &info : safety_checks) {
                Step *b = sm.acc_load(info.acc, info.acc_dtype);
                switch (info.kind) {
                case SafetyCheckKind::Bool: break;
                case SafetyCheckKind::SimpleMask: {
                    b = sm.mask_reduce({b, PredicateBinaryOp::Or}, ScalarDataType::I8);
                    break;
                }
                case SafetyCheckKind::InSignBit: {
                    VecDataType vdtype = b->dtype.as_vec();
                    MaskDataType mdtype = vdtype.mask();
                    Step *con = make_const(0, vdtype);
                    b = sm.cmp({b, con, CmpOp::Less}, mdtype);
                    b = sm.mask_reduce({b, PredicateBinaryOp::Or}, ScalarDataType::I8);
                    break;
                }
                case SafetyCheckKind::Mul: {
                    VecDataType vdtype = b->dtype.as_vec();
                    MaskDataType mdtype = vdtype.mask();
                    Step *con = make_const(vdtype.element_size_bits() + 1, vdtype);
                    b = sm.cmp({b, con, CmpOp::Less}, mdtype);
                    b = sm.mask_reduce({b, PredicateBinaryOp::Or}, ScalarDataType::I8);
                    break;
                }
                }
                if (combined == nullptr) {
                    combined = b;
                } else {
                    combined = sm.arith_bin({combined, b, ArithBinaryOp::Or}, ScalarDataType::I8);
                }
            }
            if (!safety_check_arg.has_value()) { messed_up("missing safety check argument"); }
            combined = sm.agg_result({combined, safety_check_arg.value()}, ScalarDataType::I8);
            epilogue_roots.push_back(combined);
        }
    }

    Step *agg_acc_init_value(DataType target_dtype, const hir::Step *agg_expr) {
        if (agg_expr->is(hir::StepKind::Countif)) { return make_const(0, target_dtype); }
        if (agg_expr->is(hir::StepKind::AccArithBinary)) {
            ArithBinaryOp op = agg_expr->step_data<hir::StepKind::AccArithBinary>().op;
            ScalarDataType s = target_dtype.is_scalar() ? target_dtype.as_scalar() : target_dtype.as_vec().to_scalar();
            switch (op) {
            case ArithBinaryOp::UMax:
            case ArithBinaryOp::Xor:
            case ArithBinaryOp::Or:
            case ArithBinaryOp::Add: return make_const(0, target_dtype);
            case ArithBinaryOp::Mul:
                if (s == ScalarDataType::F32) return make_const(ConstData::f32(1.0f), target_dtype);
                if (s == ScalarDataType::F64) return make_const(ConstData::f64(1.0), target_dtype);
                return make_const(1, target_dtype);
            case ArithBinaryOp::AndNot:
            case ArithBinaryOp::UMin:
            case ArithBinaryOp::And: return make_const(scalar_dtype_umax(s), target_dtype);
            case ArithBinaryOp::Min: return make_const(scalar_dtype_max(s), target_dtype);
            case ArithBinaryOp::Max: return make_const(scalar_dtype_min(s), target_dtype);
            default: break;
            }
            messed_up("unsupported initial value for arith agg %s", show_arith_binary_op(op));
        }
        if (agg_expr->is(hir::StepKind::AccPredicateBinary)) {
            // 0 and 1 values get expanded to mask splats, no need to do it manually.
            PredicateBinaryOp op = agg_expr->step_data<hir::StepKind::AccPredicateBinary>().op;
            switch (op) {
            case PredicateBinaryOp::AndNot:
            case PredicateBinaryOp::And: return make_const(1, target_dtype);
            case PredicateBinaryOp::Or:
            case PredicateBinaryOp::Xor: return make_const(0, target_dtype);
            default: break;
            }
            messed_up("unsupported initial value for predicate agg %s", show_predicate_binary_op(op));
        }
        messed_up("unexpected agg node %s", hir::show_step_kind(agg_expr->kind));
    }

    Step *vec_index_cache[(int)VecSize::X64 + 1][(int)VecElemType::F64 + 1]{};

    Step *make_vec_index(VecDataType vdtype) {
        if (auto it = vec_index_cache[(int)vdtype.size][(int)vdtype.elem]) { return it; }
        SIMJIT_ASSERT(vdtype.size_bytes() <= 64);
        alignas(64) unsigned char const_mem[64]{};
        auto fill_indices = [&](auto type_tag) {
            using Elem = decltype(type_tag);
            constexpr size_t capacity = sizeof(const_mem) / sizeof(Elem);
            for (size_t i = 0; i < vdtype.nelems() && i < capacity; ++i) {
                Elem value = static_cast<Elem>(i);
                memcpy(const_mem + i * sizeof(Elem), &value, sizeof(Elem));
            }
        };
        switch (vdtype.elem) {
        case VecElemType::I8: fill_indices(uint8_t{}); break;
        case VecElemType::I16: fill_indices(uint16_t{}); break;
        case VecElemType::I32: fill_indices(uint32_t{}); break;
        case VecElemType::I64: fill_indices(uint64_t{}); break;
        case VecElemType::F32: fill_indices(float{}); break;
        case VecElemType::F64: fill_indices(double{}); break;
        }
        auto step = sm.make_vec_const(const_mem, vdtype);
        vec_index_cache[(int)vdtype.size][(int)vdtype.elem] = step;
        return step;
    }

    Step *acc_init_value(AccKind kind, DataType dtype, const hir::Step *step) {
        switch (kind) {
        case AccKind::Agg: return agg_acc_init_value(dtype, step);
        case AccKind::Sum128:
        case AccKind::Overflow:
        case AccKind::Pack: return make_const(0, dtype);
        case AccKind::OverflowMul: {
            SIMJIT_ASSERT(dtype.is_vec());
            VecDataType vdtype = dtype.as_vec();
            return make_const(vdtype.element_size_bits() + 1, vdtype);
        }
        case AccKind::IndexInc:
            if (dtype.is_scalar()) { return make_const(0, dtype); }
            return make_vec_index(dtype.as_vec());
        }
        SIMJIT_UNREACHABLE();
    }

    AccId add_new_special_acc(const hir::Step *step, AccKind kind, DataType dtype) {
        AccId id{special_acc_count++, -1};
        Step *init = acc_init_value(kind, dtype, step);
        prologue_roots.push_back(sm.acc_store({id, init}, dtype));
        return id;
    }

    void gen_acc_inits() {
        for (const AccGroup &group : acc_groups) {
            ScalarDataType remainder_dtype = group.kind == AccKind::Sum128           ? ScalarDataType::I64
                                             : group.dst_dtype == ScalarDataType::I1 ? ScalarDataType::I8
                                                                                     : group.dst_dtype;
            Step *remainder_init = acc_init_value(group.kind, remainder_dtype, group.agg_expr);
            for (uint32_t slot = 0; slot < group.slots_per_copy; ++slot) {
                prologue_roots.push_back(sm.acc_store({group.remainder(slot), remainder_init}, remainder_dtype));
            }
            if (group.main_count() != 0) {
                DataType dtype = vect_result->acc_dtypes[group.hir_idx];
                Step *init = acc_init_value(group.kind, dtype, group.agg_expr);
                for (uint32_t copy = 0; copy < group.main_count(); ++copy) {
                    for (uint32_t slot = 0; slot < group.slots_per_copy; ++slot) {
                        prologue_roots.push_back(sm.acc_store({group.main(copy, slot), init}, dtype));
                    }
                }
            }
        }
    }

    AccId scalar_safety_check_acc() {
        if (saved_scalar_safety_check_acc.has_value()) { return *saved_scalar_safety_check_acc; }
        AccId idx = add_new_special_acc(nullptr, AccKind::Overflow, ScalarDataType::I8);
        SafetyCheckInfo info{SafetyCheckKind::Bool, ScalarDataType::I8, idx};
        safety_checks.push_back(info);
        saved_scalar_safety_check_acc = idx;
        return idx;
    }

    Step *vec_cast_overflow_condition(Step *arg, VecDataType to_vdtype) {
        VecDataType from_vdtype = arg->dtype.as_vec();
        MaskDataType mdtype = from_vdtype.mask();
        // Proper error checking should have been done before
        SIMJIT_ASSERT(to_vdtype.is_int());
        SIMJIT_ASSERT(scalar_dtype_size(from_vdtype.to_scalar()) > scalar_dtype_size(to_vdtype.to_scalar()));
        ConstData min = ConstData::i64(scalar_dtype_min(to_vdtype.to_scalar()).as_signed());
        ConstData max = ConstData::i64(scalar_dtype_max(to_vdtype.to_scalar()).as_signed());
        Step *less_min = sm.cmp({arg, make_const(min, from_vdtype), CmpOp::Less}, mdtype);
        Step *greater_max = sm.cmp({arg, make_const(max, from_vdtype), CmpOp::Greater}, mdtype);
        return sm.mask_bin({less_min, greater_max, PredicateBinaryOp::Or}, mdtype);
    }

    void add_vec_mask_overflow_check(Step *overflow) {
        MaskDataType mdtype = overflow->dtype.as_mask();
        AccId acc = add_new_special_acc(nullptr, AccKind::Overflow, mdtype);
        Step *load = sm.acc_load(acc, mdtype);
        safety_checks.push_back(SafetyCheckInfo{SafetyCheckKind::SimpleMask, mdtype, acc});
        Step *updated = sm.mask_bin({load, overflow, PredicateBinaryOp::Or}, mdtype);
        Step *check = sm.acc_store({acc, updated}, mdtype);
        main_loop_roots.push_back(check);
    }

    void add_scalar_cast_overflow_check(Step *arg, ScalarDataType to_dtype, Step *active_mask) {
        ScalarDataType from_dtype = arg->dtype.as_scalar();
        if (scalar_dtype_size(from_dtype) <= scalar_dtype_size(to_dtype)) {
            messed_up("this cast can't produce overflow");
        }
        SIMJIT_ASSERT(is_simple_int_dtype(to_dtype));
        ConstData min = ConstData::i64(scalar_dtype_min(to_dtype).as_signed());
        ConstData max = ConstData::i64(scalar_dtype_max(to_dtype).as_signed());
        Step *less_min = sm.cmp({arg, make_const(min, from_dtype), CmpOp::Less}, ScalarDataType::I1);
        Step *greater_max = sm.cmp({arg, make_const(max, from_dtype), CmpOp::Greater}, ScalarDataType::I1);
        Step *overflow = sm.arith_bin({less_min, greater_max, ArithBinaryOp::Or}, ScalarDataType::I8);
        if (active_mask != nullptr) {
            overflow = sm.arith_bin({overflow, active_mask, ArithBinaryOp::And}, ScalarDataType::I8);
        }

        AccId acc = scalar_safety_check_acc();
        Step *load = sm.acc_load(acc, ScalarDataType::I8);
        Step *updated = sm.arith_bin({load, overflow, ArithBinaryOp::Or}, ScalarDataType::I8);
        Step *check = sm.acc_store({acc, updated}, ScalarDataType::I8);
        remainder_roots.push_back(check);
    }

    void add_vec_unary_overflow_check(Step *arg, ArithUnaryOp op, Step *active_mask) {
        VecDataType vdtype = arg->dtype.as_vec();
        switch (op) {
        case ArithUnaryOp::Abs:
        case ArithUnaryOp::Negate: {
            MaskDataType mdtype = vdtype.mask();
            AccId acc = add_new_special_acc(nullptr, AccKind::Overflow, mdtype);
            Step *con = make_const(scalar_dtype_min(vdtype.to_scalar()), vdtype);
            Step *overflow = sm.cmp({arg, con, CmpOp::Equal}, mdtype);
            if (active_mask != nullptr) {
                SIMJIT_ASSERT(active_mask->dtype.is_mask());
                overflow = sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, mdtype);
            }
            Step *load = sm.acc_load(acc, mdtype);
            Step *updated = sm.mask_bin({load, overflow, PredicateBinaryOp::Or}, mdtype);
            safety_checks.push_back(SafetyCheckInfo{SafetyCheckKind::SimpleMask, mdtype, acc});
            Step *check = sm.acc_store({acc, updated}, mdtype);
            main_loop_roots.push_back(check);
            break;
        }
        default: messed_up("invalid unary overflow step %s", show_arith_unary_op(op));
        }
    }

    void add_scalar_unary_overflow_check(Step *arg, ArithUnaryOp op, Step *active_mask) {
        ScalarDataType dtype = arg->dtype.as_scalar();
        switch (op) {
        case ArithUnaryOp::Abs:
        case ArithUnaryOp::Negate: {
            AccId acc = scalar_safety_check_acc();
            Step *con = make_const(scalar_dtype_min(dtype), dtype);
            Step *overflow = sm.cmp({arg, con, CmpOp::Equal}, ScalarDataType::I1);
            if (active_mask != nullptr) {
                SIMJIT_ASSERT(active_mask->dtype == ScalarDataType::I1);
                overflow = sm.arith_bin({overflow, active_mask, ArithBinaryOp::And}, ScalarDataType::I8);
            }
            Step *load = sm.acc_load(acc, ScalarDataType::I8);
            Step *updated = sm.arith_bin({load, overflow, ArithBinaryOp::Or}, ScalarDataType::I8);
            Step *check = sm.acc_store({acc, updated}, ScalarDataType::I8);
            remainder_roots.push_back(check);
            break;
        }
        default: messed_up("invalid unary overflow step %s", show_arith_unary_op(op));
        }
    }

    void add_scalar_shift_overflow_check(Step *amount, Step *active_mask = nullptr) {
        ScalarDataType dtype = amount->dtype.as_scalar();

        Step *con = make_const(scalar_dtype_bits(dtype) - 1, dtype);
        Step *overflow = sm.cmp({amount, con, CmpOp::Greater, true}, ScalarDataType::I1);
        if (active_mask != nullptr) {
            SIMJIT_ASSERT(active_mask->dtype == ScalarDataType::I1);
            overflow = sm.arith_bin({overflow, active_mask, ArithBinaryOp::And}, ScalarDataType::I8);
        }

        AccId acc = scalar_safety_check_acc();
        Step *load = sm.acc_load(acc, ScalarDataType::I8);
        Step *updated = sm.arith_bin({load, overflow, ArithBinaryOp::Or}, ScalarDataType::I8);
        Step *check = sm.acc_store({acc, updated}, ScalarDataType::I8);
        remainder_roots.push_back(check);
    }

    Step *scalar_division_invalid_condition(Step *left, Step *right, ArithBinaryOp op) {
        ScalarDataType dtype = left->dtype.as_scalar();
        Step *rhs_zero = sm.cmp({right, make_const(0, dtype), CmpOp::Equal}, ScalarDataType::I1);
        if (!is_signed_division_op(op)) { return rhs_zero; }

        Step *lhs_min = sm.cmp({left, make_const(scalar_dtype_min(dtype), dtype), CmpOp::Equal}, ScalarDataType::I1);
        Step *rhs_mone = sm.cmp({right, make_const(-1, dtype), CmpOp::Equal}, ScalarDataType::I1);
        Step *overflow = sm.arith_bin({lhs_min, rhs_mone, ArithBinaryOp::And}, ScalarDataType::I8);
        return sm.arith_bin({rhs_zero, overflow, ArithBinaryOp::Or}, ScalarDataType::I8);
    }

    Step *vec_division_invalid_condition(Step *left, Step *right, ArithBinaryOp op) {
        VecDataType dtype = left->dtype.as_vec();
        MaskDataType mdtype = dtype.mask();
        Step *rhs_zero = sm.cmp({right, make_const(0, dtype), CmpOp::Equal}, mdtype);
        if (!is_signed_division_op(op)) { return rhs_zero; }

        Step *lhs_min = sm.cmp({left, make_const(scalar_dtype_min(dtype.to_scalar()), dtype), CmpOp::Equal}, mdtype);
        Step *rhs_mone = sm.cmp({right, make_const(-1, dtype), CmpOp::Equal}, mdtype);
        Step *overflow = sm.mask_bin({lhs_min, rhs_mone, PredicateBinaryOp::And}, mdtype);
        return sm.mask_bin({rhs_zero, overflow, PredicateBinaryOp::Or}, mdtype);
    }

    bool divisor_needs_safety(const hir::Step *right, ArithBinaryOp op) {
        if (!right->is(hir::StepKind::Const)) { return true; }
        ConstData data = right->step_data<hir::StepKind::Const>();
        if (is_signed_division_op(op)) {
            int64_t rhs = data.retag(right->dtype).as_signed();
            return rhs == 0 || rhs == -1;
        }
        return data.retag(right->dtype).as_unsigned() == 0;
    }

    void add_scalar_division_overflow_check(Step *invalid) {
        AccId acc = scalar_safety_check_acc();
        Step *load = sm.acc_load(acc, ScalarDataType::I8);
        Step *updated = sm.arith_bin({load, invalid, ArithBinaryOp::Or}, ScalarDataType::I8);
        Step *check = sm.acc_store({acc, updated}, ScalarDataType::I8);
        remainder_roots.push_back(check);
    }

    void add_vec_division_overflow_check(Step *invalid) {
        MaskDataType mdtype = invalid->dtype.as_mask();
        AccId acc = add_new_special_acc(nullptr, AccKind::Overflow, mdtype);
        Step *load = sm.acc_load(acc, mdtype);
        Step *updated = sm.mask_bin({load, invalid, PredicateBinaryOp::Or}, mdtype);
        safety_checks.push_back(SafetyCheckInfo{SafetyCheckKind::SimpleMask, mdtype, acc});
        Step *check = sm.acc_store({acc, updated}, mdtype);
        main_loop_roots.push_back(check);
    }

    void add_vec_sign_bit_overflow_check(Step *overflow) {
        VecDataType vdtype = overflow->dtype.as_vec();
        AccId acc = add_new_special_acc(nullptr, AccKind::Overflow, vdtype);
        Step *load = sm.acc_load(acc, vdtype);
        Step *updated = sm.arith_bin({load, overflow, ArithBinaryOp::Or}, vdtype);
        Step *check = sm.acc_store({acc, updated}, vdtype);
        safety_checks.push_back(SafetyCheckInfo{SafetyCheckKind::InSignBit, vdtype, acc});
        main_loop_roots.push_back(check);
    }

    void add_vec_addition_overflow_check(Step *left, Step *right, Step *result, Step *active_mask) {
        VecDataType vdtype = left->dtype.as_vec();
        if (arch_traits.has_ternarylogic) {
            Step *overflow = sm.ternarylogic({left, right, result, 0x42, ternarylogic_coerced_type(vdtype)}, vdtype);
            if (active_mask == nullptr) {
                add_vec_sign_bit_overflow_check(overflow);
                return;
            }
            MaskDataType mdtype = vdtype.mask();
            overflow = sm.cmp({overflow, make_const(0, vdtype), CmpOp::Less}, mdtype);
            overflow = sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, mdtype);
            add_vec_mask_overflow_check(overflow);
            return;
        }

        MaskDataType mdtype = vdtype.mask();
        Step *wrapped = sm.cmp({result, left, CmpOp::Less}, mdtype);
        Step *negative = sm.cmp({right, make_const(0, vdtype), CmpOp::Less}, mdtype);
        Step *overflow = sm.mask_bin({wrapped, negative, PredicateBinaryOp::Xor}, mdtype);
        if (active_mask != nullptr) { overflow = sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, mdtype); }
        add_vec_mask_overflow_check(overflow);
    }

    void add_vec_subtraction_overflow_check(Step *left, Step *right, Step *result, Step *active_mask) {
        VecDataType vdtype = left->dtype.as_vec();
        if (arch_traits.has_ternarylogic) {
            Step *overflow = sm.ternarylogic({left, right, result, 0x90, ternarylogic_coerced_type(vdtype)}, vdtype);
            if (active_mask == nullptr) {
                add_vec_sign_bit_overflow_check(overflow);
                return;
            }
            MaskDataType mdtype = vdtype.mask();
            overflow = sm.cmp({overflow, make_const(0, vdtype), CmpOp::Less}, mdtype);
            overflow = sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, mdtype);
            add_vec_mask_overflow_check(overflow);
            return;
        }

        MaskDataType mdtype = vdtype.mask();
        Step *wrapped = sm.cmp({result, left, CmpOp::Greater}, mdtype);
        Step *negative = sm.cmp({right, make_const(0, vdtype), CmpOp::Less}, mdtype);
        Step *overflow = sm.mask_bin({wrapped, negative, PredicateBinaryOp::Xor}, mdtype);
        if (active_mask != nullptr) { overflow = sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, mdtype); }
        add_vec_mask_overflow_check(overflow);
    }

    void add_vec_binary_safety_check(Step *left, Step *right, Step *result, ArithBinaryOp op,
                                     Step *active_mask = nullptr) {
        VecDataType vdtype = left->dtype.as_vec();
        switch (op) {
        case ArithBinaryOp::Add: {
            add_vec_addition_overflow_check(left, right, result, active_mask);
            break;
        }
        case ArithBinaryOp::Sub: {
            add_vec_subtraction_overflow_check(left, right, result, active_mask);
            break;
        }
        case ArithBinaryOp::Mul: {
            // See Hackers Delight 2-13 Overflow Detection. This gives approximation with possibility of false
            // positives. However, these false positives are expected to be rare, and in vectorized execution we assume
            // that values are kind of similar to each other, so we can expect real overflow to happen in same data set.
            // This requires double-checking, however.
            AccId acc = add_new_special_acc(nullptr, AccKind::OverflowMul, vdtype);
            Step *m = sm.arith_un({left, ArithUnaryOp::Abs}, vdtype);
            m = sm.arith_un({m, ArithUnaryOp::Lzcnt}, vdtype);
            Step *n = sm.arith_un({right, ArithUnaryOp::Abs}, vdtype);
            n = sm.arith_un({n, ArithUnaryOp::Lzcnt}, vdtype);
            Step *overflow = sm.arith_bin({m, n, ArithBinaryOp::Add}, vdtype);
            if (active_mask != nullptr) {
                Step *neutral = make_const(vdtype.element_size_bits() + 1, vdtype);
                overflow = sm.select({active_mask, overflow, neutral}, vdtype);
            }
            Step *load = sm.acc_load(acc, vdtype);
            Step *updated = sm.arith_bin({load, overflow, ArithBinaryOp::Min}, vdtype);
            safety_checks.push_back(SafetyCheckInfo{SafetyCheckKind::Mul, vdtype, acc});
            Step *check = sm.acc_store({acc, updated}, vdtype);
            main_loop_roots.push_back(check);
            break;
        }
        case ArithBinaryOp::ShiftRightArith:
        case ArithBinaryOp::ShiftRightLogical:
        case ArithBinaryOp::ShiftLeftLogical:
        case ArithBinaryOp::RotateLeft:
        case ArithBinaryOp::RotateRight: {
            MaskDataType mdtype = vdtype.mask();
            AccId acc = add_new_special_acc(nullptr, AccKind::Overflow, mdtype);
            Step *con = make_const(vdtype.element_size_bits() - 1, vdtype);
            Step *overflow = sm.cmp({right, con, CmpOp::Greater, true}, mdtype);
            Step *load = sm.acc_load(acc, mdtype);
            Step *updated = sm.mask_bin({load, overflow, PredicateBinaryOp::Or}, mdtype);
            safety_checks.push_back(SafetyCheckInfo{SafetyCheckKind::SimpleMask, mdtype, acc});
            Step *check = sm.acc_store({acc, updated}, mdtype);
            main_loop_roots.push_back(check);
            break;
        }
        default: break;
        }
    }

    Step *make_small_vec_mul_with_overflow_check(Step *left, Step *right, VecDataType vdtype,
                                                 Step *active_mask = nullptr) {
        SIMJIT_ASSERT(vdtype.elem == VecElemType::I8 || vdtype.elem == VecElemType::I16);
        VecElemType wide_elem = vdtype.elem == VecElemType::I8 ? VecElemType::I16 : VecElemType::I32;
        VecDataType wide_dtype = widened_half_dtype(vdtype, wide_elem);

        Step *left_low = sm.widen_lo({left, false}, wide_dtype);
        Step *left_high = sm.widen_hi({left, false}, wide_dtype);
        Step *right_low = sm.widen_lo({right, false}, wide_dtype);
        Step *right_high = sm.widen_hi({right, false}, wide_dtype);
        Step *low = sm.arith_bin({left_low, right_low, ArithBinaryOp::Mul}, wide_dtype);
        Step *high = sm.arith_bin({left_high, right_high, ArithBinaryOp::Mul}, wide_dtype);
        Step *result = sm.vec_narrow_combine({low, high}, vdtype);

        Step *low_overflow = vec_cast_overflow_condition(low, vdtype);
        Step *high_overflow = vec_cast_overflow_condition(high, vdtype);
        Step *overflow = nullptr;
        if (active_mask == nullptr) {
            overflow = sm.mask_bin({low_overflow, high_overflow, PredicateBinaryOp::Or}, low_overflow->dtype.as_mask());
        } else {
            auto combined_dtype = double_mask(low_overflow->dtype.as_mask());
            SIMJIT_ASSERT(combined_dtype && *combined_dtype == active_mask->dtype.as_mask());
            overflow = sm.combine_mask({low_overflow, high_overflow}, *combined_dtype);
            overflow = sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, *combined_dtype);
        }
        add_vec_mask_overflow_check(overflow);
        return result;
    }

    static bool ternarylogic_rpn_push(TernarylogicRpn &rpn, TernarylogicRpnOp op) {
        if (rpn.code_size == TernarylogicRpn::MaxOps) { return false; }
        rpn.code[rpn.code_size++] = op;
        return true;
    }

    static bool ternarylogic_rpn_leaf(TernarylogicRpn &rpn, const vect::Node *node) {
        for (uint8_t i = 0; i < rpn.leaf_count; ++i) {
            if (rpn.leaves[i] == node) {
                return ternarylogic_rpn_push(rpn, (TernarylogicRpnOp)((uint8_t)TernarylogicRpnOp::Leaf0 + i));
            }
        }
        if (rpn.leaf_count == 3) { return false; }
        uint8_t idx = rpn.leaf_count++;
        rpn.leaves[idx] = node;
        return ternarylogic_rpn_push(rpn, (TernarylogicRpnOp)((uint8_t)TernarylogicRpnOp::Leaf0 + idx));
    }

    bool compile_ternarylogic_rpn(const vect::Node *node, const vect::Node *root, TernarylogicRpn &rpn) {
        if (!node->dtype.is_vec() || node->dtype.as_vec() != root->dtype.as_vec()) {
            return ternarylogic_rpn_leaf(rpn, node);
        }

        if (node->is_step(hir::StepKind::ArithUnary)) {
            const auto &data = node->step->step_data<hir::StepKind::ArithUnary>();
            if (data.op != ArithUnaryOp::Not) { return ternarylogic_rpn_leaf(rpn, node); }
            if (node != root && vect_node_ref_counts[node->id] != 1) { return false; }
            if (!compile_ternarylogic_rpn(node->arg(), root, rpn)) { return false; }
            if (!ternarylogic_rpn_push(rpn, TernarylogicRpnOp::Not)) { return false; }
            ++rpn.op_count;
            return true;
        }

        if (!node->is_step(hir::StepKind::ArithBinary)) { return ternarylogic_rpn_leaf(rpn, node); }

        const auto &data = node->step->step_data<hir::StepKind::ArithBinary>();
        TernarylogicRpnOp rpn_op;
        switch (data.op) {
        case ArithBinaryOp::And: rpn_op = TernarylogicRpnOp::And; break;
        case ArithBinaryOp::Or: rpn_op = TernarylogicRpnOp::Or; break;
        case ArithBinaryOp::Xor: rpn_op = TernarylogicRpnOp::Xor; break;
        case ArithBinaryOp::AndNot: rpn_op = TernarylogicRpnOp::AndNot; break;
        default: return ternarylogic_rpn_leaf(rpn, node);
        }
        if (node != root && vect_node_ref_counts[node->id] != 1) { return false; }
        if (!compile_ternarylogic_rpn(node->left(), root, rpn)) { return false; }
        if (!compile_ternarylogic_rpn(node->right(), root, rpn)) { return false; }
        if (!ternarylogic_rpn_push(rpn, rpn_op)) { return false; }
        ++rpn.op_count;
        return true;
    }

    static uint8_t eval_ternarylogic_rpn(const TernarylogicRpn &rpn) {
        uint8_t stack[TernarylogicRpn::MaxOps]{};
        size_t stack_size = 0;

        auto pop = [&]() {
            SIMJIT_ASSERT(stack_size > 0);
            return stack[--stack_size];
        };
        auto push = [&](uint8_t value) {
            SIMJIT_ASSERT(stack_size < TernarylogicRpn::MaxOps);
            stack[stack_size++] = value;
        };

        for (size_t i = 0; i < rpn.code_size; ++i) {
            switch (rpn.code[i]) {
            case TernarylogicRpnOp::Leaf0: push(0xf0); break;
            case TernarylogicRpnOp::Leaf1: push(0xcc); break;
            case TernarylogicRpnOp::Leaf2: push(0xaa); break;
            case TernarylogicRpnOp::Not: push((uint8_t)~pop()); break;
            case TernarylogicRpnOp::And: {
                uint8_t rhs = pop();
                uint8_t lhs = pop();
                push(lhs & rhs);
                break;
            }
            case TernarylogicRpnOp::Or: {
                uint8_t rhs = pop();
                uint8_t lhs = pop();
                push(lhs | rhs);
                break;
            }
            case TernarylogicRpnOp::Xor: {
                uint8_t rhs = pop();
                uint8_t lhs = pop();
                push(lhs ^ rhs);
                break;
            }
            case TernarylogicRpnOp::AndNot: {
                uint8_t rhs = pop();
                uint8_t lhs = pop();
                push((uint8_t)(~lhs & rhs));
                break;
            }
            }
        }

        SIMJIT_ASSERT(stack_size == 1);
        return stack[0];
    }

    // Compile a local vector bit-op tree to compact RPN and evaluate it as a ternary truth table. Leaves are matched by
    // node identity and may repeat freely, but there can be at most three distinct leaves. Internal bit-op nodes must
    // be private to this tree, otherwise compiling them into RPN would duplicate shared producers. The RPN buffer is
    // capped at 15 entries to keep this bounded.
    Step *binary_op_ternarylogic_peephole(VecDataType vdtype, const vect::Node *node, const VectLoweringContext &ctx) {
        const hir::Step *step = node->step;

        SIMJIT_ASSERT(step->is(hir::StepKind::ArithBinary));
        const auto &data = step->step_data<hir::StepKind::ArithBinary>();
        if (!is_bit_binary_op(data.op)) return nullptr;

        TernarylogicRpn rpn;
        if (!compile_ternarylogic_rpn(node, node, rpn)) { return nullptr; }
        if (rpn.op_count < 2 || rpn.leaf_count == 0) { return nullptr; }

        const vect::Node *leaf0 = rpn.leaves[0];
        const vect::Node *leaf1 = rpn.leaf_count > 1 ? rpn.leaves[1] : leaf0;
        const vect::Node *leaf2 = rpn.leaf_count > 2 ? rpn.leaves[2] : leaf0;
        Step *a = lower_vect(leaf0, ctx);
        Step *b = lower_vect(leaf1, ctx);
        Step *c = lower_vect(leaf2, ctx);
        uint8_t imm = eval_ternarylogic_rpn(rpn);
        return sm.ternarylogic({a, b, c, imm, ternarylogic_coerced_type(vdtype)}, vdtype);
    }

    Step *binary_op_fma_peephole(VecDataType vdtype, const vect::Node *node, const VectLoweringContext &ctx) {
        const hir::Step *step = node->step;
        const auto &data = step->step_data<hir::StepKind::ArithBinary>();
        auto is_float_one = [&](const hir::Step *arg) {
            if (!arg->is(hir::StepKind::Const)) { return false; }
            ConstData con = arg->step_data<hir::StepKind::Const>();
            switch (vdtype.elem) {
            case VecElemType::F32: return con == ConstData::f32(1.0f);
            case VecElemType::F64: return con == ConstData::f64(1.0);
            default: return false;
            }
        };
        auto one_offset_mul_fma = [&](const vect::Node *mul_arg_node, const vect::Node *offset_node) -> Step * {
            if (!offset_node->is_step(hir::StepKind::ArithBinary)) { return nullptr; }
            const hir::Step *offset_step = offset_node->step;
            const auto &offset_data = offset_step->step_data<hir::StepKind::ArithBinary>();
            if (offset_data.op == ArithBinaryOp::Add) {
                const vect::Node *var_node = nullptr;
                if (is_float_one(offset_data.left)) {
                    var_node = offset_node->right();
                } else if (is_float_one(offset_data.right)) {
                    var_node = offset_node->left();
                }
                if (!var_node) { return nullptr; }
                Step *mul_arg = lower_vect(mul_arg_node, ctx);
                return sm.fma({lower_vect(var_node, ctx), mul_arg, mul_arg, FmaKind::FMA}, vdtype);
            }
            if (offset_data.op == ArithBinaryOp::Sub) {
                if (is_float_one(offset_data.left)) {
                    Step *mul_arg = lower_vect(mul_arg_node, ctx);
                    return sm.fma({lower_vect(offset_node->right(), ctx), mul_arg, mul_arg, FmaKind::FNMA}, vdtype);
                }
                if (is_float_one(offset_data.right)) {
                    Step *mul_arg = lower_vect(mul_arg_node, ctx);
                    return sm.fma({lower_vect(offset_node->left(), ctx), mul_arg, mul_arg, FmaKind::FMS}, vdtype);
                }
            }
            return nullptr;
        };

        if (data.op == ArithBinaryOp::Mul && vdtype.is_float()) {
            if (auto *it = one_offset_mul_fma(node->left(), node->right())) return it;
            if (auto *it = one_offset_mul_fma(node->right(), node->left())) return it;
        }

        if (data.left->is(hir::StepKind::ArithBinary)) {
            const vect::Node *left_node = node->left();
            const auto &left_data = data.left->step_data<hir::StepKind::ArithBinary>();
            if (left_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                return sm.fma({lower_vect(left_node->left(), ctx), lower_vect(left_node->right(), ctx),
                               lower_vect(node->right(), ctx), FmaKind::FMA},
                              vdtype);
            }
            if (left_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                return sm.fma({lower_vect(left_node->left(), ctx), lower_vect(left_node->right(), ctx),
                               lower_vect(node->right(), ctx), FmaKind::FMS},
                              vdtype);
            }
        }
        if (data.left->is(hir::StepKind::ArithUnary)) {
            const vect::Node *left_node = node->left();
            const auto &left_data = data.left->step_data<hir::StepKind::ArithUnary>();
            if (left_data.op == ArithUnaryOp::Negate && left_data.arg->is(hir::StepKind::ArithBinary)) {
                const vect::Node *inner_node = left_node->arg();
                const auto &inner_data = left_data.arg->step_data<hir::StepKind::ArithBinary>();
                if (inner_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                    return sm.fma({lower_vect(inner_node->left(), ctx), lower_vect(inner_node->right(), ctx),
                                   lower_vect(node->right(), ctx), FmaKind::FNMA},
                                  vdtype);
                }
                if (inner_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                    return sm.fma({lower_vect(inner_node->left(), ctx), lower_vect(inner_node->right(), ctx),
                                   lower_vect(node->right(), ctx), FmaKind::FNMS},
                                  vdtype);
                }
            }
        }
        if (data.right->is(hir::StepKind::ArithBinary)) {
            const vect::Node *right_node = node->right();
            const auto &right_data = data.right->step_data<hir::StepKind::ArithBinary>();
            if (right_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                return sm.fma({lower_vect(right_node->left(), ctx), lower_vect(right_node->right(), ctx),
                               lower_vect(node->left(), ctx), FmaKind::FMA},
                              vdtype);
            }
            if (right_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                return sm.fma({lower_vect(right_node->left(), ctx), lower_vect(right_node->right(), ctx),
                               lower_vect(node->left(), ctx), FmaKind::FNMA},
                              vdtype);
            }
        }
        if (data.right->is(hir::StepKind::ArithUnary)) {
            const vect::Node *right_node = node->right();
            const auto &right_data = data.right->step_data<hir::StepKind::ArithUnary>();
            if (right_data.op == ArithUnaryOp::Negate && right_data.arg->is(hir::StepKind::ArithBinary)) {
                const vect::Node *inner_node = right_node->arg();
                const auto &inner_data = right_data.arg->step_data<hir::StepKind::ArithBinary>();
                if (inner_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                    return sm.fma({lower_vect(inner_node->left(), ctx), lower_vect(inner_node->right(), ctx),
                                   lower_vect(node->left(), ctx), FmaKind::FNMA},
                                  vdtype);
                }
                if (inner_data.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                    return sm.fma({lower_vect(inner_node->left(), ctx), lower_vect(inner_node->right(), ctx),
                                   lower_vect(node->left(), ctx), FmaKind::FMA},
                                  vdtype);
                }
            }
        }

        return nullptr;
    }

    Step *arith_binary_i8_i16_widen(Step *left, Step *right, VecDataType vdtype, ArithBinaryOp op) {
        SIMJIT_ASSERT(op == ArithBinaryOp::Mul || op == ArithBinaryOp::ShiftRightArith ||
                      op == ArithBinaryOp::ShiftLeftLogical || op == ArithBinaryOp::ShiftRightLogical);
        bool is_unsigned =
            op == ArithBinaryOp::Mul || op == ArithBinaryOp::ShiftLeftLogical || op == ArithBinaryOp::ShiftRightLogical;
        SIMJIT_ASSERT(vdtype.elem == VecElemType::I8);
        VecDataType wide_dtype = widened_half_dtype(vdtype, VecElemType::I16);
        Step *left_low = sm.widen_lo({left, is_unsigned}, wide_dtype);
        Step *left_high = sm.widen_hi({left, is_unsigned}, wide_dtype);
        Step *right_low = sm.widen_lo({right, is_unsigned}, wide_dtype);
        Step *right_high = sm.widen_hi({right, is_unsigned}, wide_dtype);
        Step *low = sm.arith_bin({left_low, right_low, op}, wide_dtype);
        Step *high = sm.arith_bin({left_high, right_high, op}, wide_dtype);
        return make_narrow_combine(low, high, vdtype);
    }

    Step *lzcnt_i8_i16(Step *arg) {
        VecDataType vdtype = arg->dtype.as_vec();
        SIMJIT_ASSERT(vdtype.elem == VecElemType::I8 || vdtype.elem == VecElemType::I16);
        if (vdtype.elem == VecElemType::I8) {
            // y = x;
            // y |= y >> 1;
            // y |= y >> 2;
            // y |= y >> 4;
            // lz = 8 - popcnt8(y);
            Step *y = arg;
            y = sm.arith_bin({y, i8_srli(y, 1, vdtype), ArithBinaryOp::Or}, vdtype);
            y = sm.arith_bin({y, i8_srli(y, 2, vdtype), ArithBinaryOp::Or}, vdtype);
            y = sm.arith_bin({y, i8_srli(y, 4, vdtype), ArithBinaryOp::Or}, vdtype);
            y = sm.arith_bin(
                {make_const(8, vdtype), sm.arith_un({y, ArithUnaryOp::Popcount}, vdtype), ArithBinaryOp::Sub}, vdtype);
            return y;
        }
        // y = x;
        // y |= y >> 1;
        // y |= y >> 2;
        // y |= y >> 4;
        // y |= y >> 8;
        // lz = 16 - popcnt16(y);
        Step *y = arg;
        y = sm.arith_bin(
            {y, sm.arith_bin({y, make_const(1, vdtype), ArithBinaryOp::ShiftRightLogical}, vdtype), ArithBinaryOp::Or},
            vdtype);
        y = sm.arith_bin(
            {y, sm.arith_bin({y, make_const(2, vdtype), ArithBinaryOp::ShiftRightLogical}, vdtype), ArithBinaryOp::Or},
            vdtype);
        y = sm.arith_bin(
            {y, sm.arith_bin({y, make_const(4, vdtype), ArithBinaryOp::ShiftRightLogical}, vdtype), ArithBinaryOp::Or},
            vdtype);
        y = sm.arith_bin(
            {y, sm.arith_bin({y, make_const(8, vdtype), ArithBinaryOp::ShiftRightLogical}, vdtype), ArithBinaryOp::Or},
            vdtype);
        y = sm.arith_bin({make_const(16, vdtype), sm.arith_un({y, ArithUnaryOp::Popcount}, vdtype), ArithBinaryOp::Sub},
                         vdtype);
        return y;
    }

    Step *i8_permute_bits(Step *arg, uint64_t bits, VecDataType vdtype) {
        // Note that here order is different from that when we usually generate permute...
        // not sure if needs to be cleaned up, just keep in mind.
        uint8_t const_mem[64]{};
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                size_t permute_idx = ((bits >> ((j) * 8)) & 0xff);
                if (permute_idx != 0) { const_mem[i * 8 + j] = 1 << (permute_idx - 1); }
            }
        }
        Step *idxs = sm.make_vec_const(const_mem, vdtype);
        return sm.vec_permute({true, arg, idxs, bits}, vdtype);
    }

    Step *i8_srli(Step *arg, uint64_t imm, VecDataType vdtype) {
        uint64_t bits = (uint64_t)0x0102030405060708 << (imm * 8);
        return i8_permute_bits(arg, bits, vdtype);
    }

    Step *i8_shift_peephole(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is_step(hir::StepKind::ArithBinary));
        SIMJIT_ASSERT(node->dtype.is_vec());
        const auto &data = node->step->step_data<hir::StepKind::ArithBinary>();
        if (!data.right->is(hir::StepKind::Const)) { return nullptr; }
        uint64_t imm = data.right->step_data<hir::StepKind::Const>().retag(data.right->dtype).as_unsigned() & 7;
        Step *arg = lower_vect(node->left(), ctx);
        VecDataType vdtype = arg->dtype.as_vec();
        if (imm == 0) { return arg; }
        switch (data.op) {
        case ArithBinaryOp::ShiftLeftLogical: {
            uint64_t bits = (uint64_t)0x0102030405060708 >> (imm * 8);
            return i8_permute_bits(arg, bits, vdtype);
        }
        case ArithBinaryOp::ShiftRightLogical: {
            return i8_srli(arg, imm, vdtype);
        }
        case ArithBinaryOp::ShiftRightArith: {
            uint64_t sign_extend = ~((uint64_t)0xFFFFFFFFFFFFFFFF << (imm * 8)) & (uint64_t)0x0808080808080808;
            uint64_t bits = ((uint64_t)0x0102030405060708 << (imm * 8)) | sign_extend;
            return i8_permute_bits(arg, bits, vdtype);
        }
        case ArithBinaryOp::RotateLeft: {
            uint64_t bits = (uint64_t)0x0102030405060708;
            // Rotate is reversed (right rotate) because bit order is reversed
            bits = (bits >> (imm * 8)) | (bits << (64 - imm * 8));
            return i8_permute_bits(arg, bits, vdtype);
        }
        case ArithBinaryOp::RotateRight: {
            uint64_t bits = (uint64_t)0x0102030405060708;
            bits = (bits << (imm * 8)) | (bits >> (64 - imm * 8));
            return i8_permute_bits(arg, bits, vdtype);
        }
        default: break;
        }
        SIMJIT_UNREACHABLE();
    }

    void lower_vect_root_step(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is(vect::NodeKind::Step));
        SIMJIT_ASSERT(hir::is_root_step(node->step->kind));
        const hir::Step *step = node->step;
#define ref_arg() lower_vect(node->arg(), ctx)
#define ref_left() lower_vect(node->left(), ctx)
#define ref_right() lower_vect(node->right(), ctx)
#define ref_nth(_n) lower_vect(node->children[_n], ctx)
        switch (step->kind) {
            SIMJIT_MATCH (hir::StepKind::Store) {
                ArgumentAddress addr{data.addr, ctx.rows.row_offset()};
                Step *arg = ref_nth(0);
                invariant_same_type(node, arg);

                Step *root = nullptr;
                if (data.cond != nullptr) {
                    Step *cond = ref_nth(1);
                    invariant_vec_mask(node, cond);
                    root = sm.cond_store({addr, arg, cond, data.kind}, node->dtype);
                } else {
                    root = sm.store({addr, arg, data.kind}, node->dtype);
                }
                main_loop_roots.push_back(root);
                return;
            }
            SIMJIT_MATCH (hir::StepKind::AccArithBinary) {
                AccId acc_idx = acc_groups[data.acc].alloc_main(ctx.remap);
                Step *acc = sm.acc_load(acc_idx, node->dtype);
                invariant_same_type(node, acc);

                Step *updated = nullptr;
                // Duplicate FMA logic here because this code path does not go through FMA detection in ArithBinary
                if (data.cond == nullptr && bool(hir->ctx->transformations & CodeTransformations::FmaInst) &&
                    node->dtype.is_vec() && (node->dtype.as_vec().is_float() || arch_traits.has_integer_fma) &&
                    data.op == ArithBinaryOp::Add && data.arg->is(hir::StepKind::ArithBinary) &&
                    data.arg->step_data<hir::StepKind::ArithBinary>().op == ArithBinaryOp::Mul) {
                    const vect::Node *arg_node = node->arg();
                    Step *left = lower_vect(arg_node->left(), ctx);
                    Step *right = lower_vect(arg_node->right(), ctx);
                    invariant_same_type(left, right);
                    invariant_same_type(node, right);

                    updated = sm.fma({left, right, acc, FmaKind::FMA}, node->dtype.as_vec());
                } else if (data.cond) {
                    Step *arg = ref_left();
                    Step *cond = ref_right();
                    invariant_same_type(arg, node);
                    invariant_vec_mask(arg, cond);
                    // Try to have argument on the right to inline load. But for AndNot (none agg) argument is on
                    // the left. All other ops are commutative.
                    Step *left = acc;
                    Step *right = arg;
                    if (data.op == ArithBinaryOp::AndNot) { std::swap(left, right); }
                    updated = make_vec_arith_bin(left, right, data.op, node->dtype.as_vec());
                    // Actual conditional execution
                    updated = sm.select({cond, updated, acc}, node->dtype);
                } else {
                    Step *arg = ref_arg();
                    invariant_same_type(arg, node);
                    // Try to have argument on the right to inline load. But for AndNot (none agg) argument is on
                    // the left. All other ops are commutative.
                    Step *left = acc;
                    Step *right = arg;
                    if (data.op == ArithBinaryOp::AndNot) { std::swap(left, right); }
                    updated = make_vec_arith_bin(left, right, data.op, node->dtype.as_vec());
                }
                main_loop_roots.push_back(sm.acc_store({acc_idx, updated}, node->dtype));
                return;
            }
            SIMJIT_MATCH (hir::StepKind::Countif) {
                Step *arg = ref_arg();

                invariant(arg->dtype.is_mask());
                invariant(node->dtype.is_scalar());

                Step *count = sm.count_mask(arg, node->dtype.as_scalar());
                AccId acc_idx = acc_groups[data.acc].alloc_main(ctx.remap);
                Step *acc = sm.acc_load(acc_idx, node->dtype);
                invariant_same_type(node, acc);
                Step *updated = sm.arith_bin({acc, count, data.op}, node->dtype);
                main_loop_roots.push_back(sm.acc_store({acc_idx, updated}, node->dtype));
                return;
            }
            SIMJIT_MATCH (hir::StepKind::AccPredicateBinary) {
                AccId acc_idx = acc_groups[data.acc].alloc_main(ctx.remap);
                MaskDataType dt = node->dtype.as_mask();
                Step *acc = sm.acc_load(acc_idx, dt);
                Step *arg = ref_arg();
                invariant_same_mask_type(arg, node);
                invariant_same_mask_type(arg, acc);
                // Do not care about left/right order, because mask ops can't inline loads
                // (and even that case would be unimportant). ArgumentIdx is always first.
                Step *updated = sm.mask_bin({arg, acc, data.op}, dt);
                main_loop_roots.push_back(sm.acc_store({acc_idx, updated}, dt));
                return;
            }
            SIMJIT_MATCH (hir::StepKind::Scatter) {
                Step *idx = ref_nth(0);
                Step *arg = ref_nth(1);
                invariant_same_vec_type(arg, node);
                invariant_same_vec_len(idx, arg);

                Step *root = nullptr;
                if (data.cond != nullptr) {
                    Step *cond = ref_nth(2);
                    invariant_vec_mask(node, cond);
                    root = sm.cond_scatter({data.dst, idx, arg, cond}, node->dtype);
                } else {
                    root = sm.scatter({data.dst, idx, arg}, node->dtype);
                }
                main_loop_roots.push_back(root);
                return;
            }
            SIMJIT_MATCH (hir::StepKind::Pack) {
                Step *arg = ref_left();
                Step *cond = ref_right();

                invariant_same_vec_type(arg, node);
                invariant_vec_mask(node, cond);

                auto acc = acc_groups[data.dst_size_acc].singleton();
                main_loop_roots.push_back(sm.pack({data.dst, arg, cond, acc}, node->dtype));
                return;
            }
            SIMJIT_MATCH (hir::StepKind::AccSum128) {
                VecDataType vdtype = node->dtype.as_vec();
                MaskDataType mdtype = vdtype.mask();
                auto [low_acc, high_acc] = acc_groups[data.acc].alloc_main_pair(ctx.remap);
                Step *acc_lo = sm.acc_load(low_acc, vdtype);
                Step *acc_hi = sm.acc_load(high_acc, vdtype);
                Step *lo = ref_arg();
                Step *hi = sm.arith_bin({lo, make_const(63, vdtype), ArithBinaryOp::ShiftRightArith}, vdtype);
                Step *new_lo = sm.arith_bin({lo, acc_lo, ArithBinaryOp::Add}, vdtype);
                Step *new_hi = sm.arith_bin({hi, acc_hi, ArithBinaryOp::Add}, vdtype);
                Step *overflow = sm.cmp({new_lo, acc_lo, CmpOp::Less, true}, mdtype);
                Step *added = sm.arith_bin({new_hi, make_const(1, vdtype), ArithBinaryOp::Add}, vdtype);
                new_hi = sm.select({overflow, added, new_hi}, vdtype);
                main_loop_roots.push_back(sm.acc_store({high_acc, new_hi}, vdtype));
                main_loop_roots.push_back(sm.acc_store({low_acc, new_lo}, vdtype));
                return;
            }
        case hir::StepKind::Const:
        case hir::StepKind::ArithBinary:
        case hir::StepKind::CheckedOp:
        case hir::StepKind::ArithUnary:
        case hir::StepKind::Compare:
        case hir::StepKind::IntCast:
        case hir::StepKind::FloatCast:
        case hir::StepKind::BitCast:
        case hir::StepKind::PredicateBinary:
        case hir::StepKind::PredicateNot:
        case hir::StepKind::Select:
        case hir::StepKind::Index:
        case hir::StepKind::Permute:
        case hir::StepKind::Fpclass:
        case hir::StepKind::Load:
        case hir::StepKind::Gather:
        case hir::StepKind::LoadSplat: SIMJIT_UNREACHABLE();
        }
#undef ref_arg
#undef ref_left
#undef ref_right
#undef ref_nth

        SIMJIT_UNREACHABLE();
    }

    Step *lower_vect_step(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is(vect::NodeKind::Step));
        SIMJIT_ASSERT(!hir::is_root_step(node->step->kind));
        const hir::Step *step = node->step;
#define ref_arg() lower_vect(node->arg(), ctx)
#define ref_left() lower_vect(node->left(), ctx)
#define ref_right() lower_vect(node->right(), ctx)
#define ref_nth(_n) lower_vect(node->children[_n], ctx)
        switch (step->kind) {
            // Root steps
        case hir::StepKind::Store:
        case hir::StepKind::Pack:
        case hir::StepKind::Scatter:
        case hir::StepKind::AccArithBinary:
        case hir::StepKind::AccPredicateBinary:
        case hir::StepKind::AccSum128:
        case hir::StepKind::Countif:
            SIMJIT_UNREACHABLE();
            SIMJIT_MATCH (hir::StepKind::Const) { return make_const(data, node->dtype); }
            SIMJIT_MATCH (hir::StepKind::CheckedOp) {
                const hir::Step *operation = data.op;
                const vect::Node *op_node = node->children[0];
                const vect::Node *mask_node = node->child_count == 2 ? node->children[1] : nullptr;
                Step *active_mask = nullptr;
                if (mask_node != nullptr) { active_mask = lower_vect(mask_node, ctx); }
                if (operation->is(hir::StepKind::ArithBinary)) {
                    invariant(op_node->is_step(hir::StepKind::ArithBinary));
                    const auto &op_data = operation->step_data<hir::StepKind::ArithBinary>();
                    invariant(op_data.op == ArithBinaryOp::Add || op_data.op == ArithBinaryOp::Sub ||
                              op_data.op == ArithBinaryOp::Mul || op_data.op == ArithBinaryOp::ShiftRightArith ||
                              op_data.op == ArithBinaryOp::ShiftLeftLogical ||
                              op_data.op == ArithBinaryOp::ShiftRightLogical ||
                              op_data.op == ArithBinaryOp::RotateLeft || op_data.op == ArithBinaryOp::RotateRight ||
                              is_division_op(op_data.op));
                    invariant(op_data.flags == ArithBinaryOpFlags::No);

                    Step *left = lower_vect(op_node->left(), ctx);
                    Step *right = lower_vect(op_node->right(), ctx);
                    VecDataType vdtype = node->dtype.as_vec();
                    if (is_division_op(op_data.op)) {
                        Step *invalid = vec_division_invalid_condition(left, right, op_data.op);
                        if (active_mask != nullptr) {
                            invalid =
                                sm.mask_bin({invalid, active_mask, PredicateBinaryOp::And}, invalid->dtype.as_mask());
                        }
                        add_vec_division_overflow_check(invalid);
                        Step *result = lower_vect(op_node, ctx);
                        checked_steps.push_back(result);
                        return result;
                    }
                    if (op_data.op == ArithBinaryOp::Mul &&
                        (vdtype.elem == VecElemType::I8 || vdtype.elem == VecElemType::I16)) {
                        Step *result = make_small_vec_mul_with_overflow_check(left, right, vdtype, active_mask);
                        checked_steps.push_back(result);
                        return result;
                    }

                    Step *result = make_vec_arith_bin(left, right, op_data.op, vdtype);
                    add_vec_binary_safety_check(left, right, result, op_data.op, active_mask);
                    checked_steps.push_back(result);
                    return result;
                }
                if (operation->is(hir::StepKind::IntCast)) {
                    const auto &cast_data = operation->step_data<hir::StepKind::IntCast>();
                    invariant(cast_data.kind == IntCastKind::Trunc);
                    Step *result = lower_vect(op_node, ctx);
                    // This case is interesting because vectorizer might have rewritten the underlying cast, while hir
                    // step still tells it is just a basic cast.
                    Step *overflow = checked_vec_cast_overflow_condition(op_node, ctx);
                    if (mask_node != nullptr) {
                        invariant_same_mask_type(overflow, active_mask);
                        overflow =
                            sm.mask_bin({overflow, active_mask, PredicateBinaryOp::And}, overflow->dtype.as_mask());
                    }
                    add_vec_mask_overflow_check(overflow);
                    checked_steps.push_back(result);
                    return result;
                }
                if (operation->is(hir::StepKind::ArithUnary)) {
                    const auto &op_data = operation->step_data<hir::StepKind::ArithUnary>();
                    Step *arg = lower_vect(op_node->arg(), ctx);
                    Step *result = lower_vect(op_node, ctx);
                    add_vec_unary_overflow_check(arg, op_data.op, active_mask);
                    checked_steps.push_back(result);
                    return result;
                }
                messed_up("Unsupported checked op step kind %s", hir::show_step_kind(operation->kind));
            }
            SIMJIT_MATCH (hir::StepKind::ArithBinary) {
                VecDataType vdtype = node->dtype.as_vec();
                if (bool(hir->ctx->transformations & CodeTransformations::TernarylogicInst) &&
                    arch_traits.has_ternarylogic) {
                    if (auto *it = binary_op_ternarylogic_peephole(node->dtype.as_vec(), node, ctx)) return it;
                }
                // Contract mul/add into FMA/MLA when allowed.
                if (bool(hir->ctx->transformations & CodeTransformations::FmaInst) &&
                    (vdtype.is_float() || arch_traits.has_integer_fma)) {
                    if (auto *it = binary_op_fma_peephole(node->dtype.as_vec(), node, ctx)) return it;
                }

                if (bool(hir->ctx->transformations & CodeTransformations::SmallArith) &&
                    vdtype.elem == VecElemType::I8 && arch_traits.i8_gfni_shift &&
                    (data.op == ArithBinaryOp::RotateLeft || data.op == ArithBinaryOp::RotateRight ||
                     data.op == ArithBinaryOp::ShiftRightArith || data.op == ArithBinaryOp::ShiftLeftLogical ||
                     data.op == ArithBinaryOp::ShiftRightLogical)) {
                    if (auto *it = i8_shift_peephole(node, ctx)) return it;
                }
                Step *left = ref_left();
                Step *right = nullptr;
#if SIMJIT_USE_LIBDIVIDE
                if (bool(hir->ctx->transformations & CodeTransformations::ConstDiv) &&
                    data.right->is(hir::StepKind::Const) &&
                    (data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::UDiv || data.op == ArithBinaryOp::Mod ||
                     data.op == ArithBinaryOp::UMod) &&
                    is_simple_int_dtype(vdtype.to_scalar())) {
                    if (Step *rewritten = div_peephole(left, data.right->step_data<hir::StepKind::Const>(), data.op)) {
                        return rewritten;
                    }
                }
#endif
                if (vdtype.is_int() && (data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::UDiv ||
                                        data.op == ArithBinaryOp::Mod || data.op == ArithBinaryOp::UMod)) {
                    // Actually HIR already set scalar_only flag, but caller ignored it for some reason and still called
                    // vectorizer
                    simjit_exception(ErrorModule::MIR, ErrorKind::Unsupported, ErrorSubKind::UnsupportedFeature,
                                     "vectorization is not supported for non-constant divisions");
                }
                if (right == nullptr) { right = ref_right(); }
                invariant_same_vec_type(left, right);
                invariant_same_vec_type(node, left);
                invariant_imply(bool(data.flags & ArithBinaryOpFlags::ShiftWraparound),
                                data.op == ArithBinaryOp::ShiftRightArith ||
                                    data.op == ArithBinaryOp::ShiftLeftLogical ||
                                    data.op == ArithBinaryOp::ShiftRightLogical ||
                                    data.op == ArithBinaryOp::RotateLeft || data.op == ArithBinaryOp::RotateRight);
                invariant(!bool(data.flags & ArithBinaryOpFlags::SafetyCheck));

                if (bool(data.flags & ArithBinaryOpFlags::ShiftWraparound)) {
                    if (!(right->is(StepKind::Const) &&
                          right->step_data<StepKind::Const>().retag(vdtype.to_scalar()).as_unsigned() <
                              scalar_dtype_bits(vdtype.to_scalar())) &&
                        ((!arch_traits.vec_rotate_wraparound &&
                          (data.op == ArithBinaryOp::RotateRight || data.op == ArithBinaryOp::RotateLeft)) ||
                         (!arch_traits.vec_shift_wraparound &&
                          (data.op == ArithBinaryOp::ShiftRightArith || data.op == ArithBinaryOp::ShiftLeftLogical ||
                           data.op == ArithBinaryOp::ShiftRightLogical)))) {
                        right = sm.arith_bin(
                            {make_const(vdtype.element_size_bits() - 1, vdtype), right, ArithBinaryOp::And}, vdtype);
                    }
                }

                return make_vec_arith_bin(left, right, data.op, vdtype);
            }
            SIMJIT_MATCH (hir::StepKind::ArithUnary) {
                VecDataType vdtype = node->dtype.as_vec();
                Step *arg = ref_arg();

                invariant_same_vec_type(node, arg);
                invariant_imply(data.op == ArithUnaryOp::RoundNearest || data.op == ArithUnaryOp::RoundDown ||
                                    data.op == ArithUnaryOp::RoundUp || data.op == ArithUnaryOp::RoundTruncate ||
                                    data.op == ArithUnaryOp::Rcp || data.op == ArithUnaryOp::Sqrt ||
                                    data.op == ArithUnaryOp::Rsqrt,
                                vdtype.is_float());
                invariant_imply(data.op == ArithUnaryOp::Lzcnt || data.op == ArithUnaryOp::Tzcnt ||
                                    data.op == ArithUnaryOp::Popcount,
                                vdtype.is_int());

                Step *result = nullptr;
                switch (data.op) {
                case ArithUnaryOp::Negate: {
                    if (!arch_traits.has_unary_minus) {
                        if (vdtype.elem == VecElemType::F32) {
                            Step *sign = make_const(ConstData::f32(-0.0f), vdtype);
                            result = sm.arith_bin({sign, arg, ArithBinaryOp::Xor}, vdtype);
                        } else if (vdtype.elem == VecElemType::F64) {
                            Step *sign = make_const(ConstData::f64(-0.0), vdtype);
                            result = sm.arith_bin({sign, arg, ArithBinaryOp::Xor}, vdtype);
                        } else {
                            Step *zero = make_const(0, vdtype);
                            result = sm.arith_bin({zero, arg, ArithBinaryOp::Sub}, vdtype);
                        }
                    } else {
                        result = sm.arith_un({arg, data.op}, node->dtype);
                    }
                    break;
                }
                case ArithUnaryOp::Not:
                    // Similar situation to abs
                    if (!arch_traits.has_float_not && vdtype.is_float()) {
                        uint64_t con =
                            vdtype.elem == VecElemType::F32 ? (uint64_t)(uint32_t)(UINT32_MAX) : (uint64_t)(UINT64_MAX);
                        Step *con_step = make_const(con, vdtype);
                        result = sm.arith_bin({arg, con_step, ArithBinaryOp::AndNot}, node->dtype);
                    } else if (vdtype.is_int() && arch_traits.has_ternarylogic) {
                        VecDataType ty = ternarylogic_coerced_type(vdtype);
                        result = sm.ternarylogic({arg, arg, arg, 0x55, ty}, vdtype);
                    } else {
                        result = sm.arith_un({arg, data.op}, node->dtype);
                    }
                    break;
                case ArithUnaryOp::Tzcnt: result = make_tzcnt(arg, node->dtype); break;
                case ArithUnaryOp::Lzcnt:
                    if ((vdtype.elem == VecElemType::I8 || vdtype.elem == VecElemType::I16) &&
                        bool(hir->ctx->transformations & CodeTransformations::SmallArith) && !arch_traits.small_lzcnt) {
                        return lzcnt_i8_i16(arg);
                    }
                    [[fallthrough]];
                case ArithUnaryOp::Popcount: result = sm.arith_un({arg, data.op}, node->dtype); break;
                case ArithUnaryOp::Abs:
                    // AVX512 lacks abs instruction. Rewrite it here to use additional constants and not bother with
                    // it in backend
                    if (!arch_traits.has_float_abs && vdtype.is_float()) {
                        uint64_t con = vdtype.elem == VecElemType::F32 ? (uint64_t)~(uint32_t)(0x80000000u)
                                                                       : (uint64_t)(~0x8000000000000000);
                        Step *con_step = make_const(con, vdtype);
                        result = sm.arith_bin({arg, con_step, ArithBinaryOp::And}, node->dtype);
                    } else {
                        result = sm.arith_un({arg, data.op}, node->dtype);
                    }
                    break;
                case ArithUnaryOp::Rcp: result = lower_reciprocal(arg, data.op, node->dtype); break;
                case ArithUnaryOp::Rsqrt: result = lower_reciprocal(arg, data.op, node->dtype); break;
                case ArithUnaryOp::RoundNearest:
                case ArithUnaryOp::RoundDown:
                case ArithUnaryOp::RoundUp:
                case ArithUnaryOp::RoundTruncate:
                case ArithUnaryOp::Sqrt: result = sm.arith_un({arg, data.op}, node->dtype); break;
                }
                return result;
            }
            SIMJIT_MATCH (hir::StepKind::Select) {
                Step *cond = ref_nth(0);
                Step *truthy = ref_nth(1);
                Step *falsy = ref_nth(2);

                invariant_same_vec_type(truthy, falsy);
                invariant_same_vec_type(node, falsy);
                invariant_vec_mask(node, cond);

                return sm.select({cond, truthy, falsy}, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::Compare) {
                Step *left = ref_left();
                Step *right = ref_right();

                invariant_same_vec_type(left, right);
                invariant_vec_mask(left, node);

                return sm.cmp({left, right, data.op, data.is_unsigned}, node->dtype.as_mask());
            }
            SIMJIT_MATCH (hir::StepKind::PredicateNot) {
                Step *arg = ref_arg();

                invariant_same_mask_type(arg, node);

                return sm.predicate_not(arg, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::LoadSplat) {
                ArgumentAddress addr{data.idx, 0};
                return sm.load_splat({addr, data.kind}, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::IntCast) {
                Step *arg = ref_arg();
                VecDataType to_dtype = node->dtype.as_vec();

                invariant_same_vec_len(arg, node);
                invariant(arg->dtype.as_vec().is_int());
                invariant(node->dtype.as_vec().is_int());

                return sm.int_cast({arg, data.kind}, to_dtype);
            }
            SIMJIT_MATCH (hir::StepKind::FloatCast) {
                Step *arg = ref_arg();

                invariant_same_vec_len(arg, node);
                invariantm(arg->dtype.as_vec().is_float() || node->dtype.as_vec().is_float(),
                           "At least one type must be float in FloatCast, got %s -> %s", show_dtype(arg->dtype),
                           show_dtype(node->dtype));

                return sm.float_cast({arg, data.is_unsigned}, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::BitCast) {
                Step *arg = ref_arg();

                invariant_same_vec_len(arg, node);

                return sm.bitcast(arg, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::Fpclass) {
                Step *arg = ref_arg();

                invariant_vec_mask(arg, node);

                return sm.fpclass({arg, data.flags}, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::PredicateBinary) {
                Step *left = ref_left();
                Step *right = ref_right();

                invariant_same_mask_type(left, right);
                invariant_same_mask_type(left, node);

                return sm.mask_bin({left, right, data.op}, node->dtype.as_mask());
            }
            SIMJIT_MATCH (hir::StepKind::Load) {
                ArgumentAddress addr{data.idx, ctx.rows.row_offset()};
                return sm.load({addr, data.kind}, node->dtype);
            }
            SIMJIT_MATCH (hir::StepKind::Gather) {
                Step *idx = ref_arg();

                invariant_same_vec_len(idx, node);
                invariant(idx->dtype.as_vec().is_int());

                return sm.gather({data.data, idx}, node->dtype.as_vec());
            }
            SIMJIT_MATCH (hir::StepKind::Index) {
                VecDataType dtype = node->dtype.as_vec();
                VecDataType idx_ty = dtype;
                if (dtype.elem == VecElemType::F32) idx_ty.elem = VecElemType::I32;
                if (dtype.elem == VecElemType::F64) idx_ty.elem = VecElemType::I64;
                AccId idx_acc = add_new_special_acc(nullptr, AccKind::IndexInc, idx_ty);
                Step *inc = make_const(loop_width, idx_ty);
                Step *index = sm.vec_index({idx_acc, inc}, idx_ty);
                size_t row_offset = ctx.rows.row_offset();
                if (row_offset != 0) {
                    index = sm.arith_bin({index, make_const(row_offset, idx_ty), ArithBinaryOp::Add}, idx_ty);
                }
                if (dtype.is_float()) { return sm.float_cast({index, true}, dtype); }
                return index;
            }
            SIMJIT_MATCH (hir::StepKind::Permute) {
                VecDataType vdtype = node->dtype.as_vec();
                Step *arg = ref_arg();
                invariant_same_vec_type(arg, node);
                Step *idxs = nullptr;
                if (data.is_bit) {
                    uint8_t const_mem[64]{};
                    for (size_t i = 0; i < 8; ++i) {
                        for (size_t j = 0; j < 8; ++j) {
                            // Reverse order of bits
                            size_t permute_idx = ((data.permute >> ((7 - j) * 8)) & 0xff);
                            if (permute_idx != 0) { const_mem[i * 8 + j] = 1 << (permute_idx - 1); }
                        }
                    }
                    idxs = sm.make_vec_const(const_mem, vdtype);
                } else {
                    uint8_t const_mem[64]{};
                    for (size_t i = 0; i < 8; ++i) {
                        for (size_t j = 0; j < 8; ++j) {
                            size_t permute_idx = ((data.permute >> (j * 8)) & 0xff);
                            const_mem[i * 8 + j] = permute_idx + i * 8;
                        }
                    }
                    idxs = sm.make_vec_const(const_mem, vdtype);
                }
                return sm.vec_permute({data.is_bit, arg, idxs, data.permute}, vdtype);
            }
        }
#undef ref_arg
#undef ref_left
#undef ref_right
#undef ref_nth

        SIMJIT_UNREACHABLE();
    }

    Step *small_size_arith_binary(Step *l, Step *r, ArithBinaryOp op, const ConstData *right_const = nullptr) {
        ScalarDataType sdtype = l->dtype.as_scalar();
        bool is_signed = op == ArithBinaryOp::Div || op == ArithBinaryOp::Mod;
        IntCastKind cast = is_signed ? IntCastKind::Sext : IntCastKind::Zext;
        l = sm.int_cast({l, cast}, ScalarDataType::I32);
        r = sm.int_cast({r, cast}, ScalarDataType::I32);
        Step *tmp = nullptr;
#if SIMJIT_USE_LIBDIVIDE
        if (right_const != nullptr && (op == ArithBinaryOp::Div || op == ArithBinaryOp::UDiv ||
                                       op == ArithBinaryOp::Mod || op == ArithBinaryOp::UMod)) {
            ConstData widened = widen_small_divisor(*right_const, sdtype, is_signed);
            tmp = scalar_const_div_peephole(l, widened, op);
        }
#else
        (void)right_const;
#endif
        if (tmp == nullptr) { tmp = sm.arith_bin({l, r, op}, ScalarDataType::I32); }
        return sm.int_cast({tmp, IntCastKind::Trunc}, sdtype);
    }

    Step *small_size_mul_with_overflow_check(Step *l, Step *r, Step *active_mask = nullptr) {
        ScalarDataType sdtype = l->dtype.as_scalar();
        SIMJIT_ASSERT(sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16);
        l = sm.int_cast({l, IntCastKind::Sext}, ScalarDataType::I32);
        r = sm.int_cast({r, IntCastKind::Sext}, ScalarDataType::I32);
        Step *product = sm.arith_bin({l, r, ArithBinaryOp::Mul}, ScalarDataType::I32);
        add_scalar_cast_overflow_check(product, sdtype, active_mask);
        return sm.int_cast({product, IntCastKind::Trunc}, sdtype);
    }

    Step *make_vec_arith_bin(Step *l, Step *r, ArithBinaryOp op, VecDataType dtype) {
        if (dtype.elem == VecElemType::I8 && !arch_traits.i8_ops &&
            bool(hir->ctx->transformations & CodeTransformations::SmallArith)) {
            if (op == ArithBinaryOp::Mul || op == ArithBinaryOp::ShiftRightArith ||
                op == ArithBinaryOp::ShiftRightLogical || op == ArithBinaryOp::ShiftLeftLogical) {
                return arith_binary_i8_i16_widen(l, r, dtype, op);
            }
        }
        return sm.arith_bin({l, r, op}, dtype);
    }

    Step *scalar_float_cast(Step *arg, ScalarDataType to, bool is_unsigned) {
        if (arg->is(StepKind::FloatCast)) {
            const auto &inner = arg->step_data<StepKind::FloatCast>();
            ScalarDataType from = inner.arg->dtype.as_scalar();
            ScalarDataType mid = arg->dtype.as_scalar();
            if ((from == ScalarDataType::I32 || from == ScalarDataType::I64) && mid == to && is_float_dtype(to)) {
                return sm.float_cast({inner.arg, inner.is_unsigned}, to);
            }
        }
        if (arg->is(StepKind::IntCast) && is_float_dtype(to)) {
            const auto &inner = arg->step_data<StepKind::IntCast>();
            ScalarDataType from = inner.arg->dtype.as_scalar();
            if ((from == ScalarDataType::I32 || from == ScalarDataType::I64) &&
                (inner.kind == IntCastKind::Sext || inner.kind == IntCastKind::Zext)) {
                return sm.float_cast({inner.arg, inner.kind == IntCastKind::Zext}, to);
            }
        }
        return sm.float_cast({arg, is_unsigned}, to);
    }

    Step *scalar_binary_op_fma_peephole(const hir::Step *step, nonstd::span<Step *const> step_map) {
        const auto &data = step->step_data<hir::StepKind::ArithBinary>();
        ScalarDataType dtype = step->dtype;
        auto lower = [&](const hir::Step *arg) { return step_map[arg->id]; };
        auto is_float_one = [&](const hir::Step *arg) {
            if (!arg->is(hir::StepKind::Const)) { return false; }
            ConstData con = arg->step_data<hir::StepKind::Const>();
            return dtype == ScalarDataType::F32 ? con == ConstData::f32(1.0f) : con == ConstData::f64(1.0);
        };
        auto one_offset_mul_fma = [&](const hir::Step *mul_arg, const hir::Step *offset) -> Step * {
            if (!offset->is(hir::StepKind::ArithBinary)) { return nullptr; }
            const auto &offset_data = offset->step_data<hir::StepKind::ArithBinary>();
            if (offset_data.op == ArithBinaryOp::Add) {
                const hir::Step *variable = nullptr;
                if (is_float_one(offset_data.left)) {
                    variable = offset_data.right;
                } else if (is_float_one(offset_data.right)) {
                    variable = offset_data.left;
                }
                if (variable == nullptr) { return nullptr; }
                Step *lowered_mul_arg = lower(mul_arg);
                return sm.fma({lower(variable), lowered_mul_arg, lowered_mul_arg, FmaKind::FMA}, dtype);
            }
            if (offset_data.op == ArithBinaryOp::Sub) {
                if (is_float_one(offset_data.left)) {
                    Step *lowered_mul_arg = lower(mul_arg);
                    return sm.fma({lower(offset_data.right), lowered_mul_arg, lowered_mul_arg, FmaKind::FNMA}, dtype);
                }
                if (is_float_one(offset_data.right)) {
                    Step *lowered_mul_arg = lower(mul_arg);
                    return sm.fma({lower(offset_data.left), lowered_mul_arg, lowered_mul_arg, FmaKind::FMS}, dtype);
                }
            }
            return nullptr;
        };

        if (data.op == ArithBinaryOp::Mul) {
            if (auto *result = one_offset_mul_fma(data.left, data.right)) { return result; }
            if (auto *result = one_offset_mul_fma(data.right, data.left)) { return result; }
        }

        if (data.left->is(hir::StepKind::ArithBinary)) {
            const auto &left = data.left->step_data<hir::StepKind::ArithBinary>();
            if (left.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                return sm.fma({lower(left.left), lower(left.right), lower(data.right), FmaKind::FMA}, dtype);
            }
            if (left.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                return sm.fma({lower(left.left), lower(left.right), lower(data.right), FmaKind::FMS}, dtype);
            }
        }
        if (data.left->is(hir::StepKind::ArithUnary)) {
            const auto &left = data.left->step_data<hir::StepKind::ArithUnary>();
            if (left.op == ArithUnaryOp::Negate && left.arg->is(hir::StepKind::ArithBinary)) {
                const auto &inner = left.arg->step_data<hir::StepKind::ArithBinary>();
                if (inner.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                    return sm.fma({lower(inner.left), lower(inner.right), lower(data.right), FmaKind::FNMA}, dtype);
                }
                if (inner.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                    return sm.fma({lower(inner.left), lower(inner.right), lower(data.right), FmaKind::FNMS}, dtype);
                }
            }
        }
        if (data.right->is(hir::StepKind::ArithBinary)) {
            const auto &right = data.right->step_data<hir::StepKind::ArithBinary>();
            if (right.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                return sm.fma({lower(right.left), lower(right.right), lower(data.left), FmaKind::FMA}, dtype);
            }
            if (right.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                return sm.fma({lower(right.left), lower(right.right), lower(data.left), FmaKind::FNMA}, dtype);
            }
        }
        if (data.right->is(hir::StepKind::ArithUnary)) {
            const auto &right = data.right->step_data<hir::StepKind::ArithUnary>();
            if (right.op == ArithUnaryOp::Negate && right.arg->is(hir::StepKind::ArithBinary)) {
                const auto &inner = right.arg->step_data<hir::StepKind::ArithBinary>();
                if (inner.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Add) {
                    return sm.fma({lower(inner.left), lower(inner.right), lower(data.left), FmaKind::FNMA}, dtype);
                }
                if (inner.op == ArithBinaryOp::Mul && data.op == ArithBinaryOp::Sub) {
                    return sm.fma({lower(inner.left), lower(inner.right), lower(data.left), FmaKind::FMA}, dtype);
                }
            }
        }
        return nullptr;
    }

    void _lower_hir_root(const hir::Step *step, nonstd::span<Step *const> step_map) {
        SIMJIT_ASSERT(hir::is_root_step(step->kind));

#define ref(_step) step_map[(_step)->id]

        ScalarDataType sdtype = step->dtype;
        switch (step->kind) {
            SIMJIT_MATCH (hir::StepKind::Store) {
                Step *arg = ref(data.what);

                if (data.cond != nullptr) {
                    Step *cond = ref(data.cond);
                    return remainder_roots.push_back(sm.cond_store({{data.addr, 0}, arg, cond, data.kind}, arg->dtype));
                }
                return remainder_roots.push_back(sm.store({{data.addr, 0}, ref(data.what), data.kind}, sdtype));
            }
            SIMJIT_MATCH (hir::StepKind::AccArithBinary) {
                if (data.cond != nullptr) {
                    ScalarDataType dtype = sdtype;
                    AccId acc_idx = acc_groups[data.acc].alloc_remainder();
                    Step *acc = sm.acc_load(acc_idx, dtype);
                    // Try to have argument on the right to inline load. But for AndNot (none agg) argument is on
                    // the left. All other ops are commutative.
                    Step *left = acc;
                    Step *right = ref(data.arg);
                    if (data.op == ArithBinaryOp::AndNot) { std::swap(left, right); }
                    // See comment below
                    if (sdtype == ScalarDataType::I8 && data.op == ArithBinaryOp::Mul) {
                        Step *updated = small_size_arith_binary(left, right, data.op);
                        return remainder_roots.push_back(sm.acc_store({acc_idx, updated}, dtype));
                    }
                    Step *updated = sm.arith_bin({left, right, data.op}, dtype);
                    updated = sm.select({ref(data.cond), updated, acc}, dtype);
                    return remainder_roots.push_back(sm.acc_store({acc_idx, updated}, dtype));
                }
                ScalarDataType dtype = sdtype;
                AccId acc_idx = acc_groups[data.acc].alloc_remainder();
                Step *acc = sm.acc_load(acc_idx, dtype);
                // Try to have argument on the right to inline load. But for AndNot (none agg) argument is on the
                // left. All other ops are commutative.
                Step *left = acc;
                Step *right = ref(data.arg);
                if (data.op == ArithBinaryOp::AndNot) { std::swap(left, right); }
                // On x86 8-bit multiplication has special encodings, which I don't want to handle.
                if (sdtype == ScalarDataType::I8 && data.op == ArithBinaryOp::Mul) {
                    Step *updated = small_size_arith_binary(left, right, data.op);
                    return remainder_roots.push_back(sm.acc_store({acc_idx, updated}, dtype));
                }
                Step *updated = sm.arith_bin({left, right, data.op}, dtype);
                return remainder_roots.push_back(sm.acc_store({acc_idx, updated}, dtype));
            }
            SIMJIT_MATCH (hir::StepKind::Countif) {
                Step *arg = ref(data.arg);
                Step *count = sm.int_cast({arg, IntCastKind::Zext}, sdtype);
                ScalarDataType dtype = sdtype;
                AccId acc_idx = acc_groups[data.acc].alloc_remainder();
                Step *acc = sm.acc_load(acc_idx, dtype);
                Step *updated = sm.arith_bin({acc, count, data.op}, step->dtype);
                return remainder_roots.push_back(sm.acc_store({acc_idx, updated}, step->dtype));
            }
            SIMJIT_MATCH (hir::StepKind::AccSum128) {
                ScalarDataType dtype = ScalarDataType::I64;
                auto [low_acc, high_acc] = acc_groups[data.acc].alloc_remainder_pair();
                // TODO: This can be computed more efficiently using add with carry, right now it does the same as
                // vector version
                Step *acc_lo = sm.acc_load(low_acc, dtype);
                Step *acc_hi = sm.acc_load(high_acc, dtype);
                Step *lo = ref(data.arg);
                Step *hi = sm.arith_bin({lo, make_const(63, dtype), ArithBinaryOp::ShiftRightArith}, dtype);
                Step *new_lo = sm.arith_bin({lo, acc_lo, ArithBinaryOp::Add}, dtype);
                Step *new_hi = sm.arith_bin({hi, acc_hi, ArithBinaryOp::Add}, dtype);
                Step *overflow = sm.cmp({new_lo, acc_lo, CmpOp::Less, true}, ScalarDataType::I1);
                Step *added = sm.arith_bin({new_hi, make_const(1, dtype), ArithBinaryOp::Add}, dtype);
                new_hi = sm.select({overflow, added, new_hi}, dtype);
                Step *store_lo = sm.acc_store({low_acc, new_lo}, dtype);
                Step *store_hi = sm.acc_store({high_acc, new_hi}, dtype);
                // Note the order of operations - high part goes first. This is needed to process safety check
                // correctly, otherwise we can't guarantee that accumulator will not be updated before it is read
                remainder_roots.push_back(store_hi);
                remainder_roots.push_back(store_lo);
                return;
            }
            SIMJIT_MATCH (hir::StepKind::AccPredicateBinary) {
                ScalarDataType dtype = ScalarDataType::I8;
                AccId acc_idx = acc_groups[data.acc].alloc_remainder();
                Step *acc = sm.acc_load(acc_idx, dtype);
                auto maybe_arith_op = arith_op_from_predicate(data.op);
                if (!maybe_arith_op) messed_up("got invalid op %s in agg", show_predicate_binary_op(data.op));
                Step *updated = sm.arith_bin({ref(data.arg), acc, *maybe_arith_op}, dtype);
                return remainder_roots.push_back(sm.acc_store({acc_idx, updated}, dtype));
            }
            SIMJIT_MATCH (hir::StepKind::Scatter) {
                Step *arg = ref(data.arg);
                Step *idx = ref(data.idx);
                if (data.cond != nullptr) {
                    Step *cond = ref(data.cond);
                    return remainder_roots.push_back(sm.cond_scatter({data.dst, idx, arg, cond}, arg->dtype));
                }
                return remainder_roots.push_back(sm.scatter({data.dst, idx, arg}, arg->dtype));
            }
            SIMJIT_MATCH (hir::StepKind::Pack) {
                Step *arg = ref(data.arg);
                Step *cond = ref(data.cond);
                auto acc = acc_groups[data.dst_size_acc].singleton();
                return remainder_roots.push_back(sm.pack({data.dst, arg, cond, acc}, arg->dtype));
            }
        default: break;
        }
#undef ref
        SIMJIT_UNREACHABLE();
    }

    Step *_lower_hir_non_root(const hir::Step *step, nonstd::span<Step *const> step_map) {
#define ref(_step) step_map[(_step)->id]

        ScalarDataType sdtype = step->dtype;
        switch (step->kind) {
            // Root steps
        case hir::StepKind::Store:
        case hir::StepKind::Pack:
        case hir::StepKind::Scatter:
        case hir::StepKind::AccArithBinary:
        case hir::StepKind::AccPredicateBinary:
        case hir::StepKind::AccSum128:
        case hir::StepKind::Countif:
            SIMJIT_UNREACHABLE();
            SIMJIT_MATCH (hir::StepKind::Const) {
                ScalarDataType dt = sdtype == ScalarDataType::I1 ? ScalarDataType::I8 : sdtype;
                return make_const(data, dt);
            }
            SIMJIT_MATCH (hir::StepKind::ArithBinary) {
                Step *l = ref(data.left);
                Step *r = ref(data.right);
                invariant_same_scalar_type(l, r);
                invariant_same_scalar_type(l, step);
                invariant_imply(bool(data.flags & ArithBinaryOpFlags::ShiftWraparound),
                                data.op == ArithBinaryOp::ShiftRightArith ||
                                    data.op == ArithBinaryOp::ShiftLeftLogical ||
                                    data.op == ArithBinaryOp::ShiftRightLogical ||
                                    data.op == ArithBinaryOp::RotateLeft || data.op == ArithBinaryOp::RotateRight);
                invariant(!bool(data.flags & ArithBinaryOpFlags::SafetyCheck));
                if (bool(hir->ctx->transformations & CodeTransformations::FmaInst) && is_float_dtype(sdtype)) {
                    if (Step *fma = scalar_binary_op_fma_peephole(step, step_map)) { return fma; }
                }
                if (bool(data.flags & ArithBinaryOpFlags::SafeDivision) && is_division_op(data.op) &&
                    is_simple_int_dtype(sdtype) && divisor_needs_safety(data.right, data.op)) {
                    Step *invalid = scalar_division_invalid_condition(l, r, data.op);
                    r = sm.select({invalid, make_const(1, sdtype), r}, sdtype);
                }
                if ((sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) &&
                    (data.op == ArithBinaryOp::RotateLeft || data.op == ArithBinaryOp::RotateRight)) {
                    // For completeness, we provide implementation for these data types, even though nobody will
                    // actually use them. Since none of architectures natively support 8 bit and 16 bit versions (and we
                    // don't care much about perf in scalar code), use default versions.
                    Step *rotated = nullptr;
                    size_t bits = scalar_dtype_bits(sdtype);
                    // Wraparound
                    r = sm.arith_bin({r, make_const(bits - 1, sdtype), ArithBinaryOp::And}, sdtype);
                    if (data.op == ArithBinaryOp::RotateLeft) {
                        rotated = sm.arith_bin(
                            {sm.arith_bin({l, r, ArithBinaryOp::ShiftLeftLogical}, sdtype),
                             sm.arith_bin({l, sm.arith_bin({make_const(bits, sdtype), r, ArithBinaryOp::Sub}, sdtype),
                                           ArithBinaryOp::ShiftRightLogical},
                                          sdtype),
                             ArithBinaryOp::Or},
                            sdtype);
                    } else {
                        rotated = sm.arith_bin(
                            {sm.arith_bin({l, r, ArithBinaryOp::ShiftRightLogical}, sdtype),
                             sm.arith_bin({l, sm.arith_bin({make_const(bits, sdtype), r, ArithBinaryOp::Sub}, sdtype),
                                           ArithBinaryOp::ShiftLeftLogical},
                                          sdtype),
                             ArithBinaryOp::Or},
                            sdtype);
                    }
                    // Handle zero rhs
                    Step *rhs_zero = sm.cmp({r, make_const(0, sdtype), CmpOp::Equal}, ScalarDataType::I1);
                    return sm.select({rhs_zero, l, rotated}, sdtype);
                }
#if SIMJIT_USE_LIBDIVIDE
                if (bool(hir->ctx->transformations & CodeTransformations::ConstDiv) && r->is(StepKind::Const) &&
                    (data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::UDiv || data.op == ArithBinaryOp::Mod ||
                     data.op == ArithBinaryOp::UMod)) {
                    // SafeDivision may replace an invalid constant divisor such as -1 with a guarded/select value.
                    // Use the already-lowered RHS here; looking back at the original HIR constant can reintroduce
                    // INT_MIN / -1 poison in LLVM scalar remainder code.
                    if (Step *rewritten = scalar_const_div_peephole(l, r->step_data<StepKind::Const>(), data.op)) {
                        return rewritten;
                    }
                }
#endif
                // Since arm does not have native small size divisions, and x86-64 codegen for them is hard, it is
                // easier to implement them using 32 bit divisions.
                // UPD: Also added multiplications because they are annoying too.
                if (((sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) &&
                     (data.op == ArithBinaryOp::Div || data.op == ArithBinaryOp::UDiv ||
                      data.op == ArithBinaryOp::Mod || data.op == ArithBinaryOp::UMod)) ||
                    (sdtype == ScalarDataType::I8 && data.op == ArithBinaryOp::Mul)) {
                    ConstData right_const_data{};
                    const ConstData *right_const = nullptr;
                    if (r->is(StepKind::Const)) {
                        right_const_data = r->step_data<StepKind::Const>();
                        right_const = &right_const_data;
                    }
                    return small_size_arith_binary(l, r, data.op, right_const);
                }
                return sm.arith_bin({l, r, data.op}, sdtype);
            }
            SIMJIT_MATCH (hir::StepKind::CheckedOp) {
                Step *active_mask = data.mask != nullptr ? ref(data.mask) : nullptr;

                if (data.op->is(hir::StepKind::ArithBinary)) {
                    invariant(data.op->is(hir::StepKind::ArithBinary));
                    const auto &op_data = data.op->step_data<hir::StepKind::ArithBinary>();
                    invariant(op_data.op == ArithBinaryOp::Add || op_data.op == ArithBinaryOp::Sub ||
                              op_data.op == ArithBinaryOp::Mul || op_data.op == ArithBinaryOp::ShiftRightArith ||
                              op_data.op == ArithBinaryOp::ShiftLeftLogical ||
                              op_data.op == ArithBinaryOp::ShiftRightLogical ||
                              op_data.op == ArithBinaryOp::RotateLeft || op_data.op == ArithBinaryOp::RotateRight ||
                              is_division_op(op_data.op));
                    invariant(op_data.flags == ArithBinaryOpFlags::No);
                    Step *left = ref(op_data.left);
                    Step *right = ref(op_data.right);
                    if (is_division_op(op_data.op)) {
                        Step *invalid = scalar_division_invalid_condition(left, right, op_data.op);
                        Step *overflow = invalid;
                        if (active_mask != nullptr) {
                            overflow = sm.arith_bin({overflow, active_mask, ArithBinaryOp::And}, ScalarDataType::I8);
                        }
                        add_scalar_division_overflow_check(overflow);
                        if (divisor_needs_safety(op_data.right, op_data.op)) {
                            right = sm.select({invalid, make_const(1, sdtype), right}, sdtype);
                        }
#if SIMJIT_USE_LIBDIVIDE
                        if (bool(hir->ctx->transformations & CodeTransformations::ConstDiv) &&
                            right->is(StepKind::Const)) {
                            if (Step *rewritten =
                                    scalar_const_div_peephole(left, right->step_data<StepKind::Const>(), op_data.op)) {
                                return rewritten;
                            }
                        }
#endif
                        if (sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16) {
                            ConstData right_const_data{};
                            const ConstData *right_const = nullptr;
                            if (right->is(StepKind::Const)) {
                                right_const_data = right->step_data<StepKind::Const>();
                                right_const = &right_const_data;
                            }
                            return small_size_arith_binary(left, right, op_data.op, right_const);
                        }
                        return sm.arith_bin({left, right, op_data.op}, sdtype);
                    }
                    if (op_data.op == ArithBinaryOp::ShiftRightArith || op_data.op == ArithBinaryOp::ShiftLeftLogical ||
                        op_data.op == ArithBinaryOp::ShiftRightLogical || op_data.op == ArithBinaryOp::RotateLeft ||
                        op_data.op == ArithBinaryOp::RotateRight) {
                        add_scalar_shift_overflow_check(right, active_mask);
                        return ref(data.op);
                    }
                    if (op_data.op == ArithBinaryOp::Mul &&
                        (sdtype == ScalarDataType::I8 || sdtype == ScalarDataType::I16)) {
                        return small_size_mul_with_overflow_check(left, right, active_mask);
                    }
                    AccId acc = scalar_safety_check_acc();
                    return sm.ab_overflow({op_data.op, left, right, acc, active_mask}, sdtype);
                }
                if (data.op->is(hir::StepKind::IntCast)) {
                    const auto &cast_data = data.op->step_data<hir::StepKind::IntCast>();
                    invariant(cast_data.kind == IntCastKind::Trunc);
                    add_scalar_cast_overflow_check(ref(cast_data.arg), sdtype, active_mask);
                    return ref(data.op);
                }
                if (data.op->is(hir::StepKind::ArithUnary)) {
                    const auto &op_data = data.op->step_data<hir::StepKind::ArithUnary>();
                    add_scalar_unary_overflow_check(ref(op_data.arg), op_data.op, active_mask);
                    return ref(data.op);
                }
                messed_up("Unsupported checked op %s", hir::show_step_kind(data.op->kind));
            }
            SIMJIT_MATCH (hir::StepKind::ArithUnary) {
                Step *arg = ref(data.arg);
                invariant_same_scalar_type(arg, step);
                if (data.op == ArithUnaryOp::Abs && is_float_dtype(sdtype) && !arch_traits.has_float_abs) {
                    uint64_t con = sdtype == ScalarDataType::F32 ? (uint64_t)~(uint32_t)(0x80000000u)
                                                                 : (uint64_t)(~0x8000000000000000);
                    return sm.arith_bin({arg, make_const(con, sdtype), ArithBinaryOp::And}, sdtype);
                }
                if (data.op == ArithUnaryOp::Rcp || data.op == ArithUnaryOp::Rsqrt) {
                    return lower_reciprocal(arg, data.op, sdtype);
                }
                return sm.arith_un({arg, data.op}, sdtype);
            }
            SIMJIT_MATCH (hir::StepKind::Select) {
                Step *falsy = ref(data.falsy);
                Step *truthy = ref(data.truthy);
                Step *cond = ref(data.cond);
                invariant_same_scalar_type(falsy, truthy);
                invariant_same_scalar_type(falsy, step);
                return sm.select({cond, truthy, falsy}, sdtype);
            }
            SIMJIT_MATCH (hir::StepKind::Compare) {
                Step *l = ref(data.left);
                Step *r = ref(data.right);

                invariant_same_scalar_type(l, r);
                invariant_i1(step);

                return sm.cmp({l, r, data.op, data.is_unsigned}, ScalarDataType::I1);
            }
            SIMJIT_MATCH (hir::StepKind::Fpclass) {
                Step *arg = ref(data.arg);
                invariant_i1(step);
                invariant(is_float_dtype(arg->dtype.as_scalar()));

                return sm.fpclass({arg, data.flags}, ScalarDataType::I1);
            }
            SIMJIT_MATCH (hir::StepKind::PredicateBinary) {
                Step *l = ref(data.left);
                Step *r = ref(data.right);

                invariant_i1(step);

                // dtype can be either I1 or I8
                if (data.op == PredicateBinaryOp::XNor) {
                    Step *tmp = sm.arith_bin({l, r, ArithBinaryOp::Xor}, ScalarDataType::I8);
                    return sm.arith_bin({tmp, make_const(1, ScalarDataType::I8), ArithBinaryOp::Xor},
                                        ScalarDataType::I8);
                }
                auto maybe_op = arith_op_from_predicate(data.op);
                if (!maybe_op) messed_up("got invalid predicate op %s", show_predicate_binary_op(data.op));
                return sm.arith_bin({l, r, *maybe_op}, ScalarDataType::I8);
            }
            SIMJIT_MATCH (hir::StepKind::PredicateNot) {
                Step *arg = ref(data);
                invariant_i1(step);
                return sm.predicate_not(arg, sdtype);
            }
            SIMJIT_MATCH (hir::StepKind::IntCast) { return sm.int_cast({ref(data.arg), data.kind}, sdtype); }
            SIMJIT_MATCH (hir::StepKind::FloatCast) {
                return scalar_float_cast(ref(data.arg), sdtype, data.is_unsigned);
            }
            SIMJIT_MATCH (hir::StepKind::BitCast) { return sm.bitcast(ref(data), sdtype); }
            SIMJIT_MATCH (hir::StepKind::Load) {
                ArgumentAddress addr{data.idx, 0};
                return sm.load({addr, data.kind}, sdtype);
            }
            SIMJIT_MATCH (hir::StepKind::LoadSplat) {
                ArgumentAddress addr{data.idx, 0};
                return sm.load_splat({addr, data.kind}, sdtype);
            }
            SIMJIT_MATCH (hir::StepKind::Gather) return sm.gather({data.data, ref(data.idx)}, sdtype);
            SIMJIT_MATCH (hir::StepKind::Index) {
                ScalarDataType idx_ty = sdtype;
                if (sdtype == ScalarDataType::F32) idx_ty = ScalarDataType::I32;
                if (sdtype == ScalarDataType::F64) idx_ty = ScalarDataType::I64;
                Step *index = sm.scalar_index({}, idx_ty);
                if (is_float_dtype(sdtype)) { return sm.float_cast({index, true}, sdtype); }
                return index;
            }
            SIMJIT_MATCH (hir::StepKind::Permute) {
                Step *arg = ref(data.arg);
                return sm.scalar_permute({data.is_bit, arg, data.permute}, sdtype);
            }
        }
#undef ref
        SIMJIT_UNREACHABLE();
    }

    Step *lower_hir(const hir::Step *hir_step, nonstd::span<Step *> step_map) {
        if (auto *found = step_map[hir_step->id]) { return found; }
        // Unlike vectorized lowering, we eagerly initialize all children node into cache.
        // There is no much reason to do it since we recurse here anyway, but it is just how we do it.
        hir::step_recurse((hir::Step *)hir_step, [&](hir::Step *x) { lower_hir(x, step_map); });
        if (hir::is_root_step(hir_step->kind)) {
            _lower_hir_root(hir_step, step_map);
            return nullptr;
        }
        Step *result = _lower_hir_non_root(hir_step, step_map);
        step_map[hir_step->id] = result;
        return result;
    }

    void gen_remainder_loop() {
        std::vector<Step *> step_map(hir->step_id_count, nullptr);
        for (const hir::Step *hir_root : hir->step_roots) {
            Step *root = lower_hir(hir_root, step_map);
            SIMJIT_ASSERT(root == nullptr);
            (void)root;
        }
    }
    Step *fold_mask_combine_subtrees(nonstd::span<Step *> steps) {
        SIMJIT_ASSERT(!steps.empty());
        if (steps.size() == 1) { return steps[0]; }
        SIMJIT_ASSERT(steps.size() % 2 == 0);
        size_t middle = steps.size() / 2;
        Step *left = fold_mask_combine_subtrees(steps.subspan(0, middle));
        Step *right = fold_mask_combine_subtrees(steps.subspan(middle, steps.size() - middle));
        auto new_mask = double_mask(left->dtype.as_mask());
        if (!new_mask.has_value()) { messed_up("Mask grows too big"); }
        invariant_same_mask_type(left, right);
        return sm.combine_mask({left, right}, *new_mask);
    }

    Step *lower_vect_combine_masks(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is(vect::NodeKind::CombineMasks));
        const vect::Node *vect_tree = node->arg();
        size_t total_copies = size_t{1} << node->mask_combine_coef;
        SIMJIT_ASSERT(ctx.rows.width == node->item_width);
        SIMJIT_ASSERT(vect_tree->item_width * total_copies == node->item_width);
        std::vector<Step *> subtrees{};
        for (size_t i = 0; i < total_copies; ++i) {
            vect::RowBlock child_rows = ctx.rows.nth_slice(i, total_copies);
            SIMJIT_ASSERT(child_rows.width == vect_tree->item_width);
            VectLoweringContext child_ctx{child_rows, ctx.remap};
            subtrees.push_back(lower_vect(vect_tree, child_ctx));
        }

        return fold_mask_combine_subtrees(subtrees);
    }

    Step *checked_vec_cast_overflow_condition(const vect::Node *node, const VectLoweringContext &ctx) {
        if (node->is(vect::NodeKind::CastDirect)) {
            Step *arg = lower_vect(node->arg(), ctx);
            return vec_cast_overflow_condition(arg, node->dtype.as_vec());
        }
        if (node->is(vect::NodeKind::CastNarrowCombine)) {
            const vect::Node *arg_node = node->arg();
            VectLoweringContext low_ctx{ctx.rows.lhs_half(), ctx.remap};
            VectLoweringContext high_ctx{ctx.rows.rhs_half(), ctx.remap};
            Step *previous = nullptr;
            if (arg_node->is(vect::NodeKind::CastDirect) || arg_node->is(vect::NodeKind::CastNarrowCombine)) {
                Step *low = checked_vec_cast_overflow_condition(arg_node, low_ctx);
                Step *high = checked_vec_cast_overflow_condition(arg_node, high_ctx);
                auto combined_dtype = double_mask(low->dtype.as_mask());
                invariant(combined_dtype.has_value());
                invariant_same_mask_type(low, high);
                previous = sm.combine_mask({low, high}, *combined_dtype);
            }
            Step *low = lower_vect(arg_node, low_ctx);
            Step *high = lower_vect(arg_node, high_ctx);
            Step *low_overflow = vec_cast_overflow_condition(low, node->dtype.as_vec());
            Step *high_overflow = vec_cast_overflow_condition(high, node->dtype.as_vec());
            auto combined_dtype = double_mask(low_overflow->dtype.as_mask());
            invariant(combined_dtype.has_value());
            invariant_same_mask_type(low_overflow, high_overflow);
            Step *overflow = sm.combine_mask({low_overflow, high_overflow}, *combined_dtype);
            if (previous != nullptr) {
                invariant_same_mask_type(previous, overflow);
                overflow = sm.mask_bin({previous, overflow, PredicateBinaryOp::Or}, *combined_dtype);
            }
            return overflow;
        }
        if (node->is_step(hir::StepKind::IntCast)) {
            const auto &data = node->step->step_data<hir::StepKind::IntCast>();
            invariant(data.kind == IntCastKind::Trunc);
            Step *arg = lower_vect(node->arg(), ctx);
            return vec_cast_overflow_condition(arg, node->dtype.as_vec());
        }
        messed_up("checked integer truncation has invalid vectorizer node %u", unsigned(node->kind));
    }

    Step *lower_vect_narrow_combine(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is(vect::NodeKind::CastNarrowCombine));
        const vect::Node *arg_node = node->arg();
        SIMJIT_ASSERT(ctx.rows.width == node->item_width);
        SIMJIT_ASSERT(arg_node->item_width * 2 == node->item_width);
        VectLoweringContext low_ctx{ctx.rows.lhs_half(), ctx.remap};
        VectLoweringContext high_ctx{ctx.rows.rhs_half(), ctx.remap};
        Step *low = lower_vect(arg_node, low_ctx);
        Step *high = lower_vect(arg_node, high_ctx);
        if (node->cast_family == vect::CastFamily::Float) {
            return sm.vec_float_narrow_combine({low, high}, node->dtype.as_vec());
        }

        return sm.vec_narrow_combine({low, high}, node->dtype.as_vec());
    }

    Step *lower_vect_cast_direct(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is(vect::NodeKind::CastDirect));
        SIMJIT_ASSERT(node->cast_family == vect::CastFamily::Int);
        Step *arg = lower_vect(node->arg(), ctx);
        return sm.int_cast({arg, IntCastKind::Trunc}, node->dtype);
    }

    Step *lower_vect_widen(const vect::Node *node, const VectLoweringContext &ctx) {
        SIMJIT_ASSERT(node->is(vect::NodeKind::CastWidenPart));
        const vect::Node *arg_node = node->arg();
        SIMJIT_ASSERT(ctx.rows.width == node->item_width);
        SIMJIT_ASSERT(arg_node->item_width % node->item_width == 0);
        size_t part_count = arg_node->item_width / node->item_width;
        SIMJIT_ASSERT(part_count == 2);
        size_t part_idx = ctx.rows.index_in_containing_block(arg_node->item_width);
        VectLoweringContext child_ctx{ctx.rows.containing_block(arg_node->item_width), ctx.remap};
        Step *arg = lower_vect(arg_node, child_ctx);
        HalfCast data{arg, node->cast_family == vect::CastFamily::Int && node->synthetic_is_unsigned};
        if (node->cast_family == vect::CastFamily::Float) {
            if (part_idx != 0) { return sm.float_widen_hi(data, node->dtype.as_vec()); }
            return sm.float_widen_lo(data, node->dtype.as_vec());
        }
        if (part_idx != 0) { return sm.widen_hi(data, node->dtype.as_vec()); }
        return sm.widen_lo(data, node->dtype.as_vec());
    }

    Step *lower_vect(const vect::Node *node, const VectLoweringContext &ctx) {
        if (Step *cached = vect_mir_cache.lookup(node, ctx.rows)) { return cached; }

        Step *step = nullptr;
        switch (node->kind) {
        case vect::NodeKind::Step: {
            if (hir::is_root_step(node->step->kind)) {
                // Roots are not cached
                lower_vect_root_step(node, ctx);
                return nullptr;
            }
            step = lower_vect_step(node, ctx);
            break;
        }
        case vect::NodeKind::CombineMasks: step = lower_vect_combine_masks(node, ctx); break;
        case vect::NodeKind::CastNarrowCombine: step = lower_vect_narrow_combine(node, ctx); break;
        case vect::NodeKind::CastDirect: step = lower_vect_cast_direct(node, ctx); break;
        case vect::NodeKind::CastWidenPart: step = lower_vect_widen(node, ctx); break;
        }
        SIMJIT_ASSERT(step != nullptr);
        vect_mir_cache.store(node, ctx.rows, step);
        return step;
    }

    void gen_main_loop() {
        for (const vect::Root &root_info : vect_result->roots) {
            const vect::Node *root = root_info.node;
            uint8_t unroll_coef = root_info.unroll_coef;
            size_t total_copies = size_t(1) << unroll_coef;
            SIMJIT_ASSERT(loop_width % total_copies == 0);
            size_t copy_width = loop_width / total_copies;

            for (size_t x = 0; x < total_copies; ++x) {
                AccRemap remap{};
                VectLoweringContext ctx{root_info.block.nth_copy(x, copy_width), nullptr};
                if (bool(hir->ctx->transformations & CodeTransformations::AccSplit)) { ctx.remap = &remap; }
                Step *result = lower_vect(root, ctx);
                SIMJIT_ASSERT(result == nullptr);
                (void)result;
            }
        }
    }

    void create_acc_groups() {
        acc_groups.reserve(hir->accs.size());
        for (const hir::Accumulator &acc : hir->accs) {
            AccKind kind = acc.agg_expr->dtype == ScalarDataType::I128 ? AccKind::Sum128
                           : acc.agg_expr->kind == hir::StepKind::Pack ? AccKind::Pack
                                                                       : AccKind::Agg;
            acc_groups.emplace_back(kind, acc.dtype, acc.idx, acc.agg_expr, acc.dst_arg);
        }
    }

    Function *build() {
        MemoryArena *arena = hir->ctx->arena;

        for (const auto *node : prologue_roots) {
            switch (node->kind) {
            // We allow consts to appear in prologue through LICM
            case StepKind::Const:
            case StepKind::LoadSplat:
            case StepKind::VecConst: continue;
            default: break;
            }
            SIMJIT_ASSERT(is_root_step(node->kind));
        }
        for (const auto *node : main_loop_roots) {
            SIMJIT_ASSERT(is_root_step(node->kind));
        }
        for (const auto *node : remainder_roots) {
            SIMJIT_ASSERT(is_root_step(node->kind));
        }
        for (const auto *node : epilogue_roots) {
            SIMJIT_ASSERT(is_root_step(node->kind));
        }
        SIMJIT_ASSERT(has_single_bit(loop_width));

        Function *func = arena->create<Function>();
        func->ctx = hir->ctx;
        func->args = hir->args;
        std::vector<size_t> acc_group_offsets(acc_groups.size() + 1, 0);
        for (size_t i = 0; i < acc_groups.size(); ++i) {
            acc_group_offsets[i + 1] = acc_group_offsets[i] + acc_groups[i].storage_count();
        }
        func->accs.group_offsets = arena->copy_array<size_t>(acc_group_offsets);
        func->accs.agg_count = acc_group_offsets.back();
        func->accs.special_count = special_acc_count;
        func->accs.count = func->accs.agg_count + func->accs.special_count;
        func->prologue_roots = arena->copy_array<Step *>(prologue_roots);
        func->main_loop_roots = arena->copy_array<Step *>(main_loop_roots);
        func->remainder_roots = arena->copy_array<Step *>(remainder_roots);
        func->epilogue_roots = arena->copy_array<Step *>(epilogue_roots);
        func->loop_width = loop_width;
        func->step_id_count = sm.max_id();

        return func;
    }
};
} // namespace

Function *vect_to_mir(const vect::Function *vect) {
    MemoryArena *arena = vect->hir->ctx->arena;
    MirConstructState state(vect->hir->ctx);
    state.hir = vect->hir;
    state.create_acc_groups();
    state.vect_result = vect;
    state.safety_check_arg = vect->hir->safety_check_arg;
    state.vect_mir_cache.init(arena, vect);
    state.count_vectorizer_node_refs();
    state.loop_width = vect->loop_width;
    state.gen_main_loop();
    // We assume remainder loop has to be generated always
    state.gen_remainder_loop();
    state.gen_acc_inits();
    state.gen_agg_results();
    state.loop_invariant_code_motion();
    return state.build();
}

Function *hir_to_mir(const hir::Function *hir) {
    MirConstructState state(hir->ctx);
    state.hir = hir;
    state.create_acc_groups();
    state.safety_check_arg = hir->safety_check_arg;
    state.gen_remainder_loop();
    state.loop_width = 1;
    state.gen_acc_inits();
    state.gen_agg_results();
    state.loop_invariant_code_motion();
    return state.build();
}

std::vector<uint8_t> generate_bit_permute_lut(uint64_t func) {
    func = func - 0x0101010101010101;
    SIMJIT_ASSERT((func & 0x0707070707070707) == func);

    uint8_t loc1 = func & 0xff;
    uint8_t loc2 = (func >> 8) & 0xff;
    uint8_t loc3 = (func >> 16) & 0xff;
    uint8_t loc4 = (func >> 24) & 0xff;
    uint8_t loc5 = (func >> 32) & 0xff;
    uint8_t loc6 = (func >> 40) & 0xff;
    uint8_t loc7 = (func >> 48) & 0xff;
    uint8_t loc8 = (func >> 56) & 0xff;

    auto construct_value = [&](uint8_t src) -> uint8_t {
        uint8_t result = 0;
        result |= ((src >> loc1) & 1);
        result |= ((src >> loc2) & 1) << 1;
        result |= ((src >> loc3) & 1) << 2;
        result |= ((src >> loc4) & 1) << 3;
        result |= ((src >> loc5) & 1) << 4;
        result |= ((src >> loc6) & 1) << 5;
        result |= ((src >> loc7) & 1) << 6;
        result |= ((src >> loc8) & 1) << 7;
        return result;
    };

    std::vector<uint8_t> result(256, 0);
    for (size_t i = 0; i < 256; ++i) {
        result[i] = construct_value(i);
    }
    return result;
}

} // namespace mir

#if SIMJIT_ASMJIT_BACKEND
void compile_asmjit(const mir::Function *function, const AsmjitCompileOptions &opts, AsmjitCompileResult &result) {
#if SIMJIT_ASMJIT_BACKEND_ARM
    if (function->ctx->arch == Arch::Arm64_NEON) {
        compile_asmjit_arm(function, opts, result);
        return;
    }
#endif
#if SIMJIT_ASMJIT_BACKEND_X86
    if (is_x86_arch(function->ctx->arch)) {
        compile_asmjit_x86(function, opts, result);
        return;
    }
#endif
    messed_up("unsupported arch %x", (unsigned)function->ctx->arch);
}
#endif

} // namespace simjit
