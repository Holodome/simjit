// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/compiler.h"
#include "simjit/core/expr.h"

namespace simjit {
namespace hir {

struct Step;

struct ArithBinaryData {
    ArithBinaryOp op;
    Step *left;
    Step *right;
    ArithBinaryOpFlags flags = ArithBinaryOpFlags::No;
};

struct CheckedOpData {
    Step *op;
    Step *mask = nullptr;
};

struct ArithUnaryData {
    ArithUnaryOp op;
    Step *arg;
};

struct IntCastData {
    IntCastKind kind;
    Step *arg;
};

struct FloatCastData {
    Step *arg;
    bool is_unsigned = false;
};

struct StoreData {
    Step *what;
    ArgumentIdx addr;
    LoadStoreKind kind;
    Step *cond = nullptr;
};

struct CmpData {
    CmpOp op;
    Step *left;
    Step *right;
    bool is_unsigned = false;
};

struct PredicateBinaryData {
    Step *left;
    Step *right;
    PredicateBinaryOp op;
};

struct AccArithBinaryData {
    ArithBinaryOp op;
    Step *arg;
    AccIdx acc;
    Step *cond = nullptr;
};

struct AccPredicateBinData {
    PredicateBinaryOp op;
    Step *arg;
    AccIdx acc;
};

struct SelectData {
    Step *cond;
    Step *truthy;
    Step *falsy;
};

struct IndexData {};

struct ScatterData {
    Step *arg;
    Step *idx;
    ArgumentIdx dst;
    Step *cond = nullptr;
};

struct PackData {
    Step *arg;
    Step *cond;
    ArgumentIdx dst;
    AccIdx dst_size_acc;
};

struct GatherData {
    Step *idx;
    ArgumentIdx data;
};

struct PermuteData {
    Step *arg;
    uint64_t permute;
    bool is_bit;
};

struct LoadData {
    ArgumentIdx idx;
    LoadStoreKind kind;
};

struct FpclassData {
    FpClass flags;
    Step *arg;
};

enum class StepKind : uint8_t {
    Const,
    ArithBinary,
    CheckedOp,
    ArithUnary,
    Compare,
    IntCast,
    FloatCast,
    BitCast,
    PredicateBinary,
    PredicateNot,
    Select,
    Index,
    Permute,
    Fpclass,

    // Loads
    Load,
    Gather,
    LoadSplat,

    // Stores
    Store,
    Pack,
    Scatter,

    // Accumulator
    AccArithBinary,
    AccPredicateBinary,
    AccSum128,
    Countif
};

struct Step {
    ScalarDataType dtype;
    StepKind kind;
    // Incremented id, going from 0. Makes it easy to make associative arrays without unordered_map.
    // HIR steps are never copied and are guaranteed to be unique by CSE
    // (e.g. there will be only one Load for a given argument).
    uint32_t id;

    template <StepKind Kind> struct Data;
    template <StepKind Kind> typename Data<Kind>::T &step_data() noexcept;
    template <StepKind Kind> const typename Data<Kind>::T &step_data() const noexcept;
    constexpr bool is(StepKind k) const noexcept { return kind == k; }

private:
    struct ConstructorTag {};
    explicit Step(struct ConstructorTag) noexcept {}

    friend class StepMaker;

    union {
        Step *arg;
        CmpData cmp;
        StoreData store;
        SelectData select;
        ArithBinaryData ab;
        CheckedOpData checked_op;
        LoadData load;
        ArithUnaryData au;
        ConstData con;
        GatherData gather;
        PredicateBinaryData pb;
        AccArithBinaryData acc_ab;
        IndexData index;
        AccPredicateBinData acc_pb;
        ScatterData scatter;
        PackData pack;
        IntCastData int_cast;
        FloatCastData float_cast;
        PermuteData permute;
        FpclassData fpclass;
    };
};

#define HIR_STEP_DATA_LIST(X)                        \
    X(con, Const, con)                               \
    X(arith_bin, ArithBinary, ab)                    \
    X(checked_op, CheckedOp, checked_op)             \
    X(arith_un, ArithUnary, au)                      \
    X(cmp, Compare, cmp)                             \
    X(int_cast, IntCast, int_cast)                   \
    X(float_cast, FloatCast, float_cast)             \
    X(load, Load, load)                              \
    X(load_splat, LoadSplat, load)                   \
    X(gather, Gather, gather)                        \
    X(store, Store, store)                           \
    X(acc_arith_bin, AccArithBinary, acc_ab)         \
    X(acc_predicate_bin, AccPredicateBinary, acc_pb) \
    X(predicate_bin, PredicateBinary, pb)            \
    X(predicate_not, PredicateNot, arg)              \
    X(select, Select, select)                        \
    X(index, Index, index)                           \
    X(scatter, Scatter, scatter)                     \
    X(pack, Pack, pack)                              \
    X(sum128, AccSum128, acc_ab)                     \
    X(permute, Permute, permute)                     \
    X(bitcast, BitCast, arg)                         \
    X(fpclass, Fpclass, fpclass)                     \
    X(countif, Countif, acc_ab)

#define ASSOC_STEP_DATA(_0, _kind, _field)           \
    template <> struct Step::Data<StepKind::_kind> { \
        using T = decltype(Step::_field);            \
    };

HIR_STEP_DATA_LIST(ASSOC_STEP_DATA)

#undef ASSOC_STEP_DATA

#define STEP_DATA(_0, _kind, _field)                                                                             \
    template <> inline const Step::Data<StepKind::_kind>::T &Step::step_data<StepKind::_kind>() const noexcept { \
        SIMJIT_ASSERT(this->is(StepKind::_kind));                                                                \
        return this->_field;                                                                                     \
    }                                                                                                            \
    template <> inline Step::Data<StepKind::_kind>::T &Step::step_data<StepKind::_kind>() noexcept {             \
        SIMJIT_ASSERT(this->is(StepKind::_kind));                                                                \
        return this->_field;                                                                                     \
    }

HIR_STEP_DATA_LIST(STEP_DATA)

#undef STEP_DATA

class StepMaker {
public:
    StepMaker() = delete;
    explicit StepMaker(MemoryArena *arena) noexcept : arena_(arena) {}

    uint32_t max_id() const noexcept { return counter_; }

#define STEP_DATA(_name, _kind, _field)                                      \
    Step *_name(Step::Data<StepKind::_kind>::T data, ScalarDataType dtype) { \
        Step *step = make_step(StepKind::_kind, dtype);                      \
        step->_field = data;                                                 \
        return step;                                                         \
    }
    HIR_STEP_DATA_LIST(STEP_DATA)

#undef STEP_DATA

    Step *copy(const Step *other) {
        Step *created = arena_->create<Step>(*other);
        created->id = counter_++;
        return created;
    }

private:
    Step *make_step(StepKind kind, ScalarDataType dtype) {
        Step step{Step::ConstructorTag{}};
        step.id = counter_++;
        step.kind = kind;
        step.dtype = dtype;
        return arena_->create<Step>(step);
    }

    MemoryArena *arena_ = nullptr;
    uint32_t counter_ = 0;
};

struct Accumulator {
    ScalarDataType dtype;
    AccIdx idx;
    ArgumentIdx dst_arg;
    Step *agg_expr;
};

enum class SpecialOp : uint16_t {
    None = 0,
    I64Mul = 1 << 0,
    Gather = 1 << 1,
    Scatter = 1 << 2,
    CondScatter = 1 << 3,
    SmallPack = 1 << 4,
    ArbitraryBitPermute = 1 << 5,
    I8Mul = 1 << 6,
    I8VariableShift = 1 << 7,
    SmallLzcnt = 1 << 8,
    SmallGather = 1 << 9,
};
SIMJIT_DEFINE_ENUM_FLAGS(SpecialOp)

struct Function {
    Context *ctx = nullptr;
    size_t step_id_count = 0;
    ArenaArray<ArgumentDecl> args{};
    ArenaArray<Accumulator> accs{};
    ArenaArray<Step *> step_roots;
    std::optional<ArgumentIdx> safety_check_arg{};
    SpecialOp special_ops = SpecialOp::None;
    bool scalar_only = false;
};

const char *show_step_kind(StepKind kind) noexcept;
std::string show_special_ops(SpecialOp ops);

template <typename Fn> SIMJIT_NO_ASAN void step_recurse(Step *step, Fn process) {
    switch (step->kind) {
    case StepKind::Const:
    case StepKind::Load:
    case StepKind::LoadSplat:
    case StepKind::Index:
        break;

        SIMJIT_MATCH (StepKind::Gather) {
            process(data.idx);
            break;
        }
        SIMJIT_MATCH (StepKind::ArithBinary) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::CheckedOp) {
            process(data.op);
            if (data.mask) { process(data.mask); }
            break;
        }
        SIMJIT_MATCH (StepKind::ArithUnary) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::Store) {
            process(data.what);
            if (data.cond) { process(data.cond); }
            break;
        }
        SIMJIT_MATCH (StepKind::Compare) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::IntCast) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::FloatCast) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH (StepKind::PredicateNot) {
            process(data);
            break;
        }
        SIMJIT_MATCH (StepKind::PredicateBinary) {
            process(data.left);
            process(data.right);
            break;
        }
        SIMJIT_MATCH (StepKind::AccPredicateBinary) {
            process(data.arg);
            break;
        }
        SIMJIT_MATCH2 (StepKind::AccArithBinary, StepKind::AccSum128) {
            process(data.arg);
            if (data.cond) process(data.cond);
            break;
        }
        SIMJIT_MATCH (StepKind::Countif) {
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
            process(data.idx);
            process(data.arg);
            if (data.cond) process(data.cond);
            break;
        }
        SIMJIT_MATCH (StepKind::Pack) {
            process(data.arg);
            process(data.cond);
            break;
        }
        SIMJIT_MATCH (StepKind::Permute) {
            process(data.arg);
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

constexpr bool is_root_step(StepKind kind) {
    switch (kind) {
    case StepKind::Store:
    case StepKind::Pack:
    case StepKind::Scatter:
    case StepKind::AccArithBinary:
    case StepKind::AccPredicateBinary:
    case StepKind::AccSum128:
    case StepKind::Countif: return true;
    default: break;
    }
    return false;
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
        simjit_exception(ErrorModule::HIR, {}, {}, "cycle detected during traversal at step id %u", step->id);
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

} // namespace hir
} // namespace simjit
