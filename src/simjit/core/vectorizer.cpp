// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/vectorizer.h"
#include "simjit/compiler.h"
#include "simjit/core/expr.h"
#include "simjit/core/hir.h"
#include "simjit/detail/arena.h"
#include "simjit/detail/base.h"

#include <algorithm>
#include <limits.h>
#include <optional>
#include <string>

#define messed_up(...) simjit_exception(ErrorModule::Vectorizer, {}, {}, __VA_ARGS__)

namespace simjit {
namespace vect {

using namespace ::simjit::hir;

static SpecialOp supported_vector_special_ops(Arch arch, CodeTransformations transformations) noexcept {
    if (is_x86_arch(arch)) {
        SpecialOp result = SpecialOp::I64Mul | SpecialOp::Gather | SpecialOp::Scatter | SpecialOp::CondScatter |
                           SpecialOp::SmallPack | SpecialOp::ArbitraryBitPermute;
        if (bool(transformations & CodeTransformations::SmallArith)) {
            result |= SpecialOp::I8Mul | SpecialOp::I8VariableShift | SpecialOp::SmallLzcnt;
        }
        return result;
    }
    if (arch == Arch::Arm64_NEON) {
        return SpecialOp::Gather | SpecialOp::SmallPack | SpecialOp::I8Mul | SpecialOp::I8VariableShift |
               SpecialOp::SmallLzcnt;
    }
    SIMJIT_UNREACHABLE();
}

static SpecialOp unsupported_vector_special_ops(const hir::Function *func) noexcept {
    return func->special_ops & ~supported_vector_special_ops(func->ctx->arch, func->ctx->transformations);
}

struct NodePool {
    static constexpr size_t chunk_size = 64;
    static constexpr size_t chunk_shift = 6;
    static constexpr size_t chunk_mask = chunk_size - 1;
    static_assert((chunk_size & chunk_mask) == 0);

    explicit NodePool(MemoryArena *arena_) noexcept : arena(arena_) {}

    Node *allocate() {
        if (chunks.size() <= (size_ >> chunk_shift)) { chunks.push_back(arena->alloc_array<Node>(chunk_size)); }

        uint32_t id = size_++;

        Node *node = get(id);
        *node = Node();
        node->id = id;
        return node;
    }

    Node *get(uint32_t id) noexcept {
        SIMJIT_ASSERT(id != 0);
        SIMJIT_ASSERT(id < size_);
        return &chunks[id >> chunk_shift][id & chunk_mask];
    }

    const Node *get(uint32_t id) const noexcept {
        SIMJIT_ASSERT(id != 0);
        SIMJIT_ASSERT(id < size_);
        return &chunks[id >> chunk_shift][id & chunk_mask];
    }

    uint32_t size() const noexcept { return size_; }

private:
    MemoryArena *arena = nullptr;
    std::vector<ArenaArray<Node>> chunks{};
    uint32_t size_ = 1;
};

static int coef_adjust(ScalarDataType from, ScalarDataType to) noexcept {
    size_t to_size = scalar_dtype_size(to);
    size_t from_size = scalar_dtype_size(from);
    if (to_size == from_size) { return 0; }
    if (from_size > to_size) {
        size_t times = from_size / to_size;
        return (int)nonzero_log2(times);
    }
    size_t times = to_size / from_size;
    return -(int)nonzero_log2(times);
}

template <typename F> static SIMJIT_NO_ASAN void unique_postorder_traverse(Node *node, F f, ArenaBitmap &visited) {
    if (visited.get(node->id)) return;
    visited.set(node->id);
    for (auto child : node->children_span()) {
        unique_postorder_traverse(child, f, visited);
    }
    f(node);
}

static bool is_mask_combineable_step(const Step *step) noexcept {
    if (step->is(StepKind::PredicateBinary)) return true;
    if (step->is(StepKind::Compare)) return true;
    if (step->is(StepKind::Fpclass)) return true;
    return false;
}

static VecSize vec_size_from_nelems(size_t nelems) {
    switch (nelems) {
    case 2: return VecSize::X2;
    case 4: return VecSize::X4;
    case 8: return VecSize::X8;
    case 16: return VecSize::X16;
    case 32: return VecSize::X32;
    case 64: return VecSize::X64;
    default: messed_up("invalid vector lane count %zu", nelems);
    }
    SIMJIT_UNREACHABLE();
}

static VecDataType vec_dtype_from_coef(ScalarDataType scalar, int8_t coef) {
    auto maybe_elem = vec_elem_from_scalar(scalar);
    if (!maybe_elem) messed_up("got unexpected data type %s in vector context", show_scalar_dtype(scalar));
    SIMJIT_ASSERT(coef >= 0);
    size_t vector_bytes = (size_t)16 << (size_t)coef;
    size_t nelems = vector_bytes / scalar_dtype_size(scalar);
    return VecDataType{vec_size_from_nelems(nelems), *maybe_elem};
}

static bool find_mask_coef(Node *node, int8_t &coef, const std::vector<Node *> &mask_replacement_by_id) noexcept {
    if (node->is(NodeKind::CombineMasks)) {
        coef = (int8_t)node->mask_combine_coef;
        return true;
    }

    bool result = false;
    for (auto &child : node->children_span()) {
        if (mask_replacement_by_id[child->id] != nullptr) { child = mask_replacement_by_id[child->id]; }
        bool is_mask = find_mask_coef(child, coef, mask_replacement_by_id);
        result = result || is_mask;
    }
    return result;
}

static ScalarDataType scalar_dtype_with_size(size_t size) {
    switch (size) {
    case 1: return ScalarDataType::I8;
    case 2: return ScalarDataType::I16;
    case 4: return ScalarDataType::I32;
    case 8: return ScalarDataType::I64;
    default: messed_up("invalid integer scalar size %zu", size);
    }
    SIMJIT_UNREACHABLE();
}

static void find_mask_combineable_nodes(Node *node, std::vector<Node *> &mask_result_nodes,
                                        ArenaBitmap &is_mask_result_node) {
    if (node->is(NodeKind::Step)) {
        const Step *step = node->step;
        // degenerate cases of load
        if (step->dtype == ScalarDataType::I1 && step->is(StepKind::Store) && node->arg()->is(NodeKind::Step)) {
            const Step *arg = node->arg()->step;
            if (arg->is(StepKind::Load)) { return; }
            if (arg->is(StepKind::PredicateNot) && arg->step_data<StepKind::PredicateNot>()->is(StepKind::Load)) {
                return;
            }
        }
        if (step->is(StepKind::Select)) { return; }
        if (step->is(StepKind::Pack)) { return; }
        if (step->is(StepKind::Store) && step->step_data<StepKind::Store>().cond != nullptr) { return; }
        if (is_mask_combineable_step(step)) {
            if (!is_mask_result_node.get(node->id)) {
                is_mask_result_node.set(node->id);
                mask_result_nodes.push_back(node);
            }
            return;
        }
    }
    for (auto child : node->children_span()) {
        find_mask_combineable_nodes(child, mask_result_nodes, is_mask_result_node);
    }
}

namespace {
// Rewriting knows a root's position but not its concrete width. RowBlock is constructed only after type concretization.
struct RootState {
    Node *node = nullptr;
    size_t logical_root_idx = 0;
    size_t row_block_idx = 0;
    uint8_t unroll_coef{};
};

static void collect_assignment_nodes(Node *node, std::vector<Node *> &nodes, ArenaBitmap &visited) {
    unique_postorder_traverse(
        node,
        [&](Node *x) {
            if (!x->is(NodeKind::CombineMasks)) { nodes.push_back(x); }
        },
        visited);
}

static std::vector<Node *> collect_assignment_nodes(const std::vector<RootState> &roots, uint32_t id_counter,
                                                    MemoryArena *arena) {
    std::vector<Node *> nodes{};
    ArenaBitmap visited = ArenaBitmap::create(arena, id_counter);
    for (const RootState &root : roots) {
        collect_assignment_nodes(root.node, nodes, visited);
    }
    return nodes;
}

enum class CastDirection {
    Narrow,
    Widen,
    SameWidth,
};

struct VectCastInfo {
    CastFamily family = CastFamily::Int;
    CastDirection direction = CastDirection::SameWidth;
    ScalarDataType from = ScalarDataType::I1;
    ScalarDataType to = ScalarDataType::I1;
    bool is_unsigned = false;
};

static bool is_synthetic_cast_node(const Node *node) noexcept {
    return node->is(NodeKind::CastDirect) || node->is(NodeKind::CastNarrowCombine) || node->is(NodeKind::CastWidenPart);
}

static bool cast_info_supported(const VectCastInfo &info) noexcept {
    if (info.family == CastFamily::Int) { return is_simple_int_dtype(info.from) && is_simple_int_dtype(info.to); }
    return is_float_dtype(info.from) && is_float_dtype(info.to);
}

static bool graph_root_prefers_min_coef(const Node *root) noexcept {
    if (!root->is(NodeKind::Step)) { return false; }
    switch (root->step->kind) {
    case StepKind::Store: return root->step->dtype == ScalarDataType::I1;
    case StepKind::AccPredicateBinary:
    case StepKind::Countif: return true;
    default: return false;
    }
}

static bool graph_has_forbidden_mask(const std::vector<Node *> &mask_combineable_nodes,
                                     const ArenaBitmap &is_forbidden_mask) noexcept {
    for (const Node *node : mask_combineable_nodes) {
        if (is_forbidden_mask.get(node->id)) { return true; }
    }
    return false;
}

struct VectorizationFailure {
    ErrorSubKind kind = ErrorSubKind::None;
    std::string message{};
    bool internal = false;
};

struct GraphAssignment {
    std::vector<std::optional<ScalarDataType>> carrier_dtypes{};
    ArenaBitmap has_coef{};
    std::vector<int8_t> coefs{};
    std::vector<int> component_ids{};
    std::vector<std::vector<Node *>> parents{};
    std::string error{};
};

struct DowncastRewrite {
    Node *old_node;
    Node *new_node;
};

struct UpcastSubject {
    Node *node = nullptr;
    ScalarDataType split_dtype = ScalarDataType::I1;
    CastFamily family = CastFamily::Int;
    bool is_unsigned = false;
};

struct ComponentCoefRange {
    int8_t min_coef = INT8_MAX;
    int8_t max_coef = INT8_MIN;
};

enum class ComponentRangeCastKind {
    Downcast,
    Upcast,
};

struct ComponentRangeCast {
    ComponentRangeCastKind kind = ComponentRangeCastKind::Downcast;
    Node *node = nullptr;
    int score = 0;
};

// Upcast normalization can replace one HIR root with several Roots. A RootWidthGroup collects roots by their preserved
// logical-root identity, not by cloned node identity. RootState records each clone's future row-block index; its
// concrete RowBlock is built after type concretization supplies the width. Unrolling records a coefficient on every
// root in the group; it does not clone vectorizer roots.
struct RootWidthGroup {
    size_t logical_root_idx = 0;
    std::vector<size_t> root_idxs{};
    size_t width = 0;
};

static constexpr int proactive_unroll_node_threshold = 10;
static constexpr int proactive_unroll_aggregate_node_bonus = 10;
static constexpr int proactive_unroll_float_node_bonus = 5;
static constexpr int proactive_unroll_complex_op_penalty = 10;
static constexpr int graph_coef_abs_limit = 16;

struct ProactiveUnrollStats {
    size_t effective_node_count = 0;
    size_t agg_count = 0;
    size_t float_op_count = 0;
    size_t complex_op_cost = 0;
};

static bool is_aggregate_step(const Step *step) noexcept {
    switch (step->kind) {
    case StepKind::AccArithBinary:
    case StepKind::AccPredicateBinary:
    case StepKind::AccSum128:
    case StepKind::Countif: return true;
    default: return false;
    }
}

static bool is_complex_arith_binary_op(ArithBinaryOp op) noexcept {
    switch (op) {
    case ArithBinaryOp::Div:
    case ArithBinaryOp::UDiv:
    case ArithBinaryOp::Mod:
    case ArithBinaryOp::UMod: return true;
    default: return false;
    }
}

static bool is_complex_arith_unary_op(ArithUnaryOp op) noexcept {
    switch (op) {
    case ArithUnaryOp::Rcp:
    case ArithUnaryOp::Sqrt:
    case ArithUnaryOp::Rsqrt: return true;
    default: return false;
    }
}

static bool is_complex_step(const Step *step) noexcept {
    switch (step->kind) {
    case StepKind::ArithBinary: return is_complex_arith_binary_op(step->step_data<StepKind::ArithBinary>().op);
    case StepKind::ArithUnary: return is_complex_arith_unary_op(step->step_data<StepKind::ArithUnary>().op);
    default: return false;
    }
}

static bool is_float_op_step(const Step *step) noexcept {
    switch (step->kind) {
    case StepKind::ArithBinary:
    case StepKind::ArithUnary:
    case StepKind::AccArithBinary: return is_float_dtype(step->dtype);
    case StepKind::FloatCast:
    case StepKind::Fpclass: return true;
    case StepKind::Compare: return is_float_dtype(step->step_data<StepKind::Compare>().left->dtype);
    default: return false;
    }
}

static size_t proactive_unroll_special_op_cost(SpecialOp ops) noexcept {
    size_t cost = 0;
    if (bool(ops & SpecialOp::Gather) || bool(ops & SpecialOp::SmallGather)) { ++cost; }
    if (bool(ops & SpecialOp::Scatter) || bool(ops & SpecialOp::CondScatter)) { ++cost; }
    if (bool(ops & SpecialOp::SmallPack)) { ++cost; }
    if (bool(ops & SpecialOp::ArbitraryBitPermute)) { ++cost; }
    if (bool(ops & SpecialOp::I64Mul)) { ++cost; }
    if (bool(ops & SpecialOp::I8Mul)) { ++cost; }
    if (bool(ops & SpecialOp::I8VariableShift)) { ++cost; }
    if (bool(ops & SpecialOp::SmallLzcnt)) { ++cost; }
    return cost;
}

static bool graph_node_allows_no_carrier(const Node *node) noexcept {
    SIMJIT_ASSERT(node->is(NodeKind::Step));
    return node->step->dtype == ScalarDataType::I1 || node->is_step(StepKind::Countif);
}

static size_t graph_item_width_from_dtype(DataType dtype) noexcept {
    if (dtype.is_vec()) { return dtype.as_vec().nelems(); }
    if (dtype.is_mask()) { return mask_dtype_bits(dtype.as_mask()); }
    return 0;
}

static bool graph_component_ranges_fit(const std::vector<ComponentCoefRange> &ranges, int8_t vec_range) noexcept {
    for (const ComponentCoefRange &range : ranges) {
        SIMJIT_ASSERT(range.min_coef != INT8_MAX && range.max_coef != INT8_MIN);
        if (range.max_coef - range.min_coef > vec_range) { return false; }
    }
    return true;
}

static bool walk_graph_subtree_has_upcast_subject(Node *node, const std::vector<UpcastSubject> &subject_by_id,
                                                  ArenaBitmap &visited) noexcept {
    if (visited.get(node->id)) { return false; }
    visited.set(node->id);
    if (subject_by_id[node->id].node != nullptr) { return true; }
    for (Node *child : node->children_span()) {
        if (walk_graph_subtree_has_upcast_subject(child, subject_by_id, visited)) { return true; }
    }
    return false;
}

static bool walk_graph_subtree_has_downcast_combine(Node *node, ArenaBitmap &visited) noexcept {
    if (visited.get(node->id)) { return false; }
    visited.set(node->id);
    if (node->is(NodeKind::CastNarrowCombine)) { return true; }
    for (Node *child : node->children_span()) {
        if (walk_graph_subtree_has_downcast_combine(child, visited)) { return true; }
    }
    return false;
}

static int graph_proactive_unroll_threshold(const ProactiveUnrollStats &stats) noexcept {
    int threshold = proactive_unroll_node_threshold;
    if (stats.agg_count != 0) { threshold += proactive_unroll_aggregate_node_bonus; }
    if (stats.float_op_count != 0) { threshold += proactive_unroll_float_node_bonus; }
    threshold -= (int)stats.complex_op_cost * proactive_unroll_complex_op_penalty;
    return threshold;
}

static void graph_consider_component_range_cast(std::optional<ComponentRangeCast> &best, ComponentRangeCast candidate,
                                                int high_coef, int high_limit, int delta) noexcept {
    if (high_coef <= high_limit || delta <= 0) { return; }
    int producer_side_bonus = candidate.kind == ComponentRangeCastKind::Downcast ? 1000 : 0;
    candidate.score = producer_side_bonus + (high_coef - high_limit) * 16 + delta;
    if (!best || candidate.score > best->score) { best = candidate; }
}

static void graph_collect_proactive_unroll_stats(const Node *node, ProactiveUnrollStats &stats,
                                                 size_t multiplier) noexcept {
    if (!node->is(NodeKind::CombineMasks)) { stats.effective_node_count += multiplier; }
    for (const Node *child : node->children_span()) {
        graph_collect_proactive_unroll_stats(child, stats, multiplier);
    }
}

static size_t graph_proactive_unroll_extra_coef(const ProactiveUnrollStats &stats) noexcept {
    int threshold = graph_proactive_unroll_threshold(stats);
    if (threshold <= 0) { return 0; }
    size_t budget = (size_t)threshold;
    if (stats.effective_node_count >= budget) { return 0; }

    size_t extra_coef = 1;
    size_t estimated_nodes = stats.effective_node_count;
    while (estimated_nodes <= SIZE_MAX / 2) {
        estimated_nodes *= 2;
        if (estimated_nodes >= budget) { break; }
        ++extra_coef;
    }
    constexpr size_t max_unroll = 2;
    return std::min(extra_coef, max_unroll);
}

struct AlgebraicVectorizer {
    MemoryArena *arena;
    NodePool pool;
    const hir::Function *hir;
    std::vector<RootState> roots{};
    ArenaArray<Node *> node_map{};
    ArenaArray<DataType> acc_dtypes{};
    MaskDataType max_mask = MaskDataType::M2;
    int8_t vec_range = 0;
    size_t unroll_hint = 0;
    size_t loop_width = 0;
    VectorizationFailure graph_failure{};
    GraphAssignment assignment{};

    explicit AlgebraicVectorizer(const hir::Function *func) : arena(func->ctx->arena), pool(func->ctx->arena) {
        MemoryArena *mem = func->ctx->arena;
        hir = func;
        acc_dtypes = mem->alloc_array<DataType>(func->accs.size());
        node_map = mem->alloc_array<Node *>(func->step_id_count);
        if (func->ctx->arch == Arch::Arm64_NEON) {
            max_mask = MaskDataType::M16;
            vec_range = 0;
            unroll_hint = 4;
        } else if (func->ctx->arch == Arch::Amd64_AVX512) {
            max_mask = MaskDataType::M64;
            vec_range = 2;
            unroll_hint = 2;
        } else if (func->ctx->arch == Arch::Amd64_AVX512_YMM) {
            max_mask = MaskDataType::M32;
            vec_range = 1;
            unroll_hint = 2;
        }

        size_t logical_root_idx = 0;
        for (Step *root : func->step_roots) {
            Node *node = construct_tree(root);
            roots.push_back(RootState{node, logical_root_idx++});
        }
    }

    uint32_t id_counter() const noexcept { return pool.size(); }

    Node *create_node() { return pool.allocate(); }

    Node *construct_tree(Step *step) {
        if (auto it = node_map[step->id]) { return it; }
        Node *node = create_node();
        node_map[step->id] = node;
        node->step = step;
        step_recurse(step, [&](Step *ref) {
            SIMJIT_ASSERT(node->child_count < std::size(node->children));
            Node *child = construct_tree(ref);
            node->children[node->child_count++] = child;
        });
        return node;
    }

    bool fail_graph_assignment(ErrorSubKind kind, std::string message, bool internal = false) noexcept {
        assignment.error = message;
        graph_failure = VectorizationFailure{kind, std::move(message), internal};
        return false;
    }

    bool fail_graph_assignment(ErrorSubKind kind, const char *message, bool internal = false) {
        return fail_graph_assignment(kind, std::string(message), internal);
    }

    bool check_vector_root_limit(size_t root_count) {
        size_t max_roots = hir->ctx->build_limits.max_vector_roots;
        if (root_count <= max_roots) { return true; }
        return fail_graph_assignment(
            ErrorSubKind::TooManyRoots,
            simjit::format("vectorization creates too many roots (%zu > %zu)", root_count, max_roots));
    }

    void assign_carrier(Node *node, ScalarDataType dtype) noexcept {
        auto &slot = assignment.carrier_dtypes[node->id];
        if (slot.has_value()) {
            SIMJIT_ASSERT(*slot == dtype);
        } else {
            slot = dtype;
        }
    }

    std::optional<VectCastInfo> graph_cast_info(const Node *node) {
        if (node->is_step(StepKind::IntCast)) {
            auto data = node->step->step_data<StepKind::IntCast>();
            CastDirection direction = CastDirection::SameWidth;
            const Node *arg = node->arg();
            ScalarDataType from = data.arg->dtype;
            ScalarDataType to = node->step->dtype;
            if (auto x = assignment.carrier_dtypes[arg->id]) { from = *x; }

            if (data.kind == IntCastKind::Trunc) {
                direction = CastDirection::Narrow;
            } else if (data.kind == IntCastKind::Sext || data.kind == IntCastKind::Zext) {
                direction = CastDirection::Widen;
            } else {
                return {};
            }

            VectCastInfo info{CastFamily::Int, direction, from, to, data.kind == IntCastKind::Zext};
            if (!cast_info_supported(info)) { return {}; }
            return info;
        }

        if (node->is_step(StepKind::FloatCast)) {
            auto data = node->step->step_data<StepKind::FloatCast>();
            const Node *arg = node->arg();
            ScalarDataType from = data.arg->dtype;
            ScalarDataType to = node->step->dtype;
            if (auto x = assignment.carrier_dtypes[arg->id]) { from = *x; }
            if (!is_float_dtype(from) || !is_float_dtype(to)) { return {}; }

            CastDirection direction = CastDirection::SameWidth;
            if (scalar_dtype_size(from) > scalar_dtype_size(to)) {
                direction = CastDirection::Narrow;
            } else if (scalar_dtype_size(from) < scalar_dtype_size(to)) {
                direction = CastDirection::Widen;
            }
            return VectCastInfo{CastFamily::Float, direction, from, to, false};
        }

        return {};
    }

    bool graph_unify_carriers(Node *left, Node *right) noexcept {
        auto &left_carrier = assignment.carrier_dtypes[left->id];
        auto &right_carrier = assignment.carrier_dtypes[right->id];
        if (left_carrier.has_value() && !right_carrier.has_value()) {
            right_carrier = left_carrier;
            return true;
        }
        if (!left_carrier.has_value() && right_carrier.has_value()) {
            left_carrier = right_carrier;
            return true;
        }
        return false;
    }

    void graph_seed_carrier_dtypes(nonstd::span<Node *const> nodes) {
        for (Node *node : nodes) {
            if (is_synthetic_cast_node(node)) {
                assign_carrier(node, node->synthetic_dtype);
                continue;
            }

            if (!node->is(NodeKind::Step)) { continue; }
            const Step *step = node->step;
            if (step->dtype == ScalarDataType::I1 || step->dtype == ScalarDataType::I128) { continue; }
            if (step->is(StepKind::Countif)) { continue; }
            SIMJIT_ASSERT(vec_elem_from_scalar(step->dtype).has_value());
            assign_carrier(node, step->dtype);
        }
    }

    void graph_propagate_carrier_dtypes_once(nonstd::span<Node *> nodes) {
        bool changed = true;
        auto accumulate = [&](bool x) { changed = changed || x; };
        while (changed) {
            changed = false;
            for (Node *node : nodes) {
                if (!node->is(NodeKind::Step)) { continue; }
                switch (node->step->kind) {
                case StepKind::CheckedOp:
                    accumulate(graph_unify_carriers(node, node->children[0]));
                    if (node->child_count == 2) {
                        accumulate(graph_unify_carriers(node->children[0], node->children[1]));
                    }
                    break;
                case StepKind::Compare: accumulate(graph_unify_carriers(node, node->left())); break;
                case StepKind::Fpclass: accumulate(graph_unify_carriers(node, node->arg())); break;
                case StepKind::PredicateNot: accumulate(graph_unify_carriers(node, node->arg())); break;
                case StepKind::PredicateBinary:
                    accumulate(graph_unify_carriers(node, node->left()));
                    accumulate(graph_unify_carriers(node, node->right()));
                    break;
                case StepKind::Select:
                    if (node->step->dtype == ScalarDataType::I1) {
                        accumulate(graph_unify_carriers(node, node->children[1]));
                        accumulate(graph_unify_carriers(node->children[1], node->children[2]));
                    } else {
                        accumulate(graph_unify_carriers(node, node->children[1]));
                        accumulate(graph_unify_carriers(node->children[0], node->children[1]));
                    }
                    break;
                case StepKind::Store:
                    accumulate(graph_unify_carriers(node, node->children[0]));
                    if (node->child_count == 2) {
                        accumulate(graph_unify_carriers(node->children[0], node->children[1]));
                    }
                    break;
                case StepKind::Pack:
                    accumulate(graph_unify_carriers(node, node->left()));
                    accumulate(graph_unify_carriers(node->left(), node->right()));
                    break;
                case StepKind::AccPredicateBinary: accumulate(graph_unify_carriers(node, node->arg())); break;
                case StepKind::AccArithBinary:
                case StepKind::AccSum128:
                    accumulate(graph_unify_carriers(node, node->children[0]));
                    if (node->child_count == 2) {
                        accumulate(graph_unify_carriers(node->children[0], node->children[1]));
                    }
                    break;
                case StepKind::Countif: accumulate(graph_unify_carriers(node, node->arg())); break;
                case StepKind::Scatter:
                    accumulate(graph_unify_carriers(node, node->children[1]));
                    if (node->child_count == 3) {
                        accumulate(graph_unify_carriers(node->children[1], node->children[2]));
                    }
                    break;
                default: break;
                }
            }
        }
    }

    void graph_propagate_carrier_dtypes(nonstd::span<Node *> nodes) {
        bool defaulted_predicate_carrier = false;
        do {
            defaulted_predicate_carrier = false;
            graph_propagate_carrier_dtypes_once(nodes);
            for (const Node *node : nodes) {
                if (!assignment.carrier_dtypes[node->id].has_value() && node->is(NodeKind::Step) &&
                    node->step->dtype == ScalarDataType::I1) {
                    assignment.carrier_dtypes[node->id] = ScalarDataType::I8;
                    defaulted_predicate_carrier = true;
                }
            }
        } while (defaulted_predicate_carrier);
    }

    bool graph_assign_coef(const Node *node, int coef) {
        if (coef < -graph_coef_abs_limit || coef > graph_coef_abs_limit) {
            return fail_graph_assignment(
                ErrorSubKind::GraphCoefficientLimitExceeded,
                simjit::format("graph coefficient %d exceeds vectorizer limit +/- %d", coef, graph_coef_abs_limit));
        }
        int8_t stored_coef = (int8_t)coef;
        if (assignment.has_coef.get(node->id)) { return assignment.coefs[node->id] == stored_coef; }
        assignment.has_coef.set(node->id);
        assignment.coefs[node->id] = stored_coef;
        return true;
    }

    std::optional<int> graph_edge_delta_parent_to_child(const Node *parent, const Node *child) noexcept {
        if (parent->is(NodeKind::CastNarrowCombine) || parent->is(NodeKind::CastWidenPart)) { return 0; }
        const auto &parent_dtype = assignment.carrier_dtypes[parent->id];
        const auto &child_dtype = assignment.carrier_dtypes[child->id];
        if (!parent_dtype && !child_dtype) { return 0; }
        SIMJIT_ASSERT(parent_dtype.has_value());
        SIMJIT_ASSERT(child_dtype.has_value());
        return coef_adjust(*child_dtype, *parent_dtype);
    }

    bool graph_assign_component_coefs(Node *start, int component_id) {
        // Traversal order is irrelevant here; a vector is enough for the coefficient-propagation worklist.
        std::vector<Node *> worklist{};
        bool assigned = graph_assign_coef(start, 0);
        SIMJIT_ASSERT(assigned);
        assignment.component_ids[start->id] = component_id;
        worklist.push_back(start);
        while (!worklist.empty()) {
            Node *node = worklist.back();
            worklist.pop_back();
            int node_coef = assignment.coefs[node->id];
            for (Node *child : node->children_span()) {
                auto delta = graph_edge_delta_parent_to_child(node, child);
                if (!delta) { continue; }
                int child_coef = node_coef + *delta;
                bool was_assigned = assignment.has_coef.get(child->id);
                if (!graph_assign_coef(child, child_coef)) {
                    if (graph_failure.kind != ErrorSubKind::None) { return false; }
                    return fail_graph_assignment(ErrorSubKind::ConflictingGraphCoefficient,
                                                 "conflicting graph coefficient");
                }
                if (!was_assigned) {
                    assignment.component_ids[child->id] = component_id;
                    worklist.push_back(child);
                }
            }
            for (Node *parent : assignment.parents[node->id]) {
                auto delta = graph_edge_delta_parent_to_child(parent, node);
                if (!delta) { continue; }
                int parent_coef = node_coef - *delta;
                bool was_assigned = assignment.has_coef.get(parent->id);
                if (!graph_assign_coef(parent, parent_coef)) {
                    if (graph_failure.kind != ErrorSubKind::None) { return false; }
                    return fail_graph_assignment(ErrorSubKind::ConflictingGraphCoefficient,
                                                 "conflicting graph coefficient");
                }
                if (!was_assigned) {
                    assignment.component_ids[parent->id] = component_id;
                    worklist.push_back(parent);
                }
            }
        }
        return true;
    }

    bool graph_assign_coefs(const std::vector<Node *> &nodes) {
        int component_id = 0;
        for (const Node *node : nodes) {
            if (!assignment.carrier_dtypes[node->id].has_value() &&
                !(node->is(NodeKind::Step) && graph_node_allows_no_carrier(node)) && !is_synthetic_cast_node(node)) {
                return fail_graph_assignment(ErrorSubKind::UnresolvedCarrierDType, "unresolved carrier dtype");
            }
        }
        for (Node *node : nodes) {
            if (assignment.has_coef.get(node->id)) { continue; }
            if (!graph_assign_component_coefs(node, component_id)) { return false; }
            ++component_id;
        }
        return true;
    }

    void replace_node(const Node *old_node, Node *new_node) noexcept {
        for (Node *parent : assignment.parents[old_node->id]) {
            for (Node *&child : parent->children_span()) {
                if (child == old_node) { child = new_node; }
            }
        }
        for (RootState &root : roots) {
            if (root.node == old_node) { root.node = new_node; }
        }
    }

    Node *create_cast_direct(Node *arg, const VectCastInfo &info, ScalarDataType dtype) {
        Node *node = create_node();
        node->kind = NodeKind::CastDirect;
        node->cast_family = info.family;
        node->synthetic_dtype = dtype;
        node->synthetic_is_unsigned = info.is_unsigned;
        node->child_count = 1;
        node->children[0] = arg;
        return node;
    }

    Node *create_cast_narrow_combine(Node *arg, const VectCastInfo &info, ScalarDataType dtype) {
        Node *node = create_node();
        node->kind = NodeKind::CastNarrowCombine;
        node->cast_family = info.family;
        node->synthetic_dtype = dtype;
        node->synthetic_is_unsigned = info.is_unsigned;
        node->child_count = 1;
        node->children[0] = arg;
        return node;
    }

    Node *create_cast_widen_part(Node *arg, const VectCastInfo &info, ScalarDataType dtype) {
        Node *node = create_node();
        node->kind = NodeKind::CastWidenPart;
        node->cast_family = info.family;
        node->synthetic_dtype = dtype;
        node->synthetic_is_unsigned = info.is_unsigned;
        node->child_count = 1;
        node->children[0] = arg;
        return node;
    }

    Node *create_step_node(const hir::Step *step, Node *arg) {
        Node *node = create_node();
        node->kind = NodeKind::Step;
        node->step = step;
        node->child_count = 1;
        node->children[0] = arg;
        return node;
    }

    std::vector<DowncastRewrite> graph_plan_downcast_rewrites(const std::vector<Node *> &nodes) {
        std::vector<DowncastRewrite> rewrites{};
        for (Node *node : nodes) {
            std::optional<VectCastInfo> info = graph_cast_info(node);
            if (!info || info->direction != CastDirection::Narrow) { continue; }
            Node *arg = node->arg();
            ScalarDataType from = info->from;
            ScalarDataType to = info->to;

            int delta = assignment.coefs[arg->id] - assignment.coefs[node->id];
            if (delta <= vec_range) { continue; }
            SIMJIT_ASSERT(delta == (int)coef_adjust(from, to));

            if (info->family == CastFamily::Float && delta != 1) { continue; }

            int direct = vec_range;
            int indirect = delta - direct;
            SIMJIT_ASSERT(indirect > 0);

            Node *new_node = arg;
            size_t from_size = scalar_dtype_size(from);
            if (direct != 0) {
                ScalarDataType direct_dtype = scalar_dtype_with_size(from_size >> direct);
                new_node = create_cast_direct(new_node, *info, direct_dtype);
            }

            for (int i = 0; i < indirect; ++i) {
                int shift = direct + i + 1;
                ScalarDataType downcast_dtype =
                    info->family == CastFamily::Float ? to : scalar_dtype_with_size(from_size >> shift);
                new_node = create_cast_narrow_combine(new_node, *info, downcast_dtype);
            }

            SIMJIT_ASSERT(new_node->synthetic_dtype == to);
            rewrites.push_back({node, new_node});
        }
        return rewrites;
    }

    bool graph_assign() {
        assignment = GraphAssignment{};
        assignment.carrier_dtypes.resize(id_counter());
        assignment.has_coef = ArenaBitmap::create(arena, id_counter());
        assignment.coefs.assign(id_counter(), 0);
        assignment.component_ids.assign(id_counter(), -1);
        assignment.parents.resize(id_counter());

        std::vector<Node *> nodes = collect_assignment_nodes(roots, id_counter(), arena);
        for (Node *node : nodes) {
            for (const Node *child : node->children_span()) {
                assignment.parents[child->id].push_back(node);
            }
        }
        graph_seed_carrier_dtypes(nodes);
        graph_propagate_carrier_dtypes(nodes);
        return graph_assign_coefs(nodes);
    }

    bool graph_resolve_downcasts() {
        if (!graph_assign()) { return false; }

        std::vector<Node *> nodes = collect_assignment_nodes(roots, id_counter(), arena);
        std::vector<DowncastRewrite> rewrites = graph_plan_downcast_rewrites(nodes);
        if (rewrites.empty()) { return true; }

        for (const DowncastRewrite &rewrite : rewrites) {
            replace_node(rewrite.old_node, rewrite.new_node);
        }
        return graph_assign();
    }

    std::optional<UpcastSubject> graph_upcast_subject(Node *node) {
        std::optional<VectCastInfo> info = graph_cast_info(node);
        if (!info || info->direction != CastDirection::Widen) { return {}; }

        const Node *arg = node->arg();
        int delta = assignment.coefs[node->id] - assignment.coefs[arg->id];
        if (delta <= vec_range) { return {}; }

        size_t split_size = scalar_dtype_size(info->from) * 2;
        if (split_size > scalar_dtype_size(info->to)) { return {}; }
        ScalarDataType split_dtype = info->family == CastFamily::Int ? scalar_dtype_with_size(split_size) : info->to;
        if (info->family == CastFamily::Float &&
            (info->from != ScalarDataType::F32 || info->to != ScalarDataType::F64)) {
            return {};
        }
        return UpcastSubject{node, split_dtype, info->family, info->is_unsigned};
    }

    void graph_collect_upcast_subjects(Node *node, std::vector<UpcastSubject> &subjects, ArenaBitmap &visited) {
        if (visited.get(node->id)) { return; }
        visited.set(node->id);
        if (std::optional<UpcastSubject> subject = graph_upcast_subject(node)) {
            subjects.push_back(*subject);
            return;
        }
        for (Node *child : node->children_span()) {
            graph_collect_upcast_subjects(child, subjects, visited);
        }
    }

    bool graph_subtree_has_upcast_subject(Node *node, const std::vector<UpcastSubject> &subject_by_id) {
        ArenaBitmap visited = ArenaBitmap::create(arena, id_counter());
        return walk_graph_subtree_has_upcast_subject(node, subject_by_id, visited);
    }

    bool graph_subtree_has_downcast_combine(Node *node) {
        ArenaBitmap visited = ArenaBitmap::create(arena, id_counter());
        return walk_graph_subtree_has_downcast_combine(node, visited);
    }

    Node *graph_clone_upcast_split(Node *node, const std::vector<UpcastSubject> &subject_by_id,
                                   std::vector<Node *> &clone_by_id, bool force_clone) {
        if (subject_by_id[node->id].node != nullptr) {
            const UpcastSubject &subject = subject_by_id[node->id];
            Node *arg = subject.node->arg();
            VectCastInfo info{subject.family, CastDirection::Widen, ScalarDataType::I1, subject.node->step->dtype,
                              subject.is_unsigned};
            Node *split = create_cast_widen_part(arg, info, subject.split_dtype);
            if (subject.split_dtype == subject.node->step->dtype) { return split; }
            return create_step_node(subject.node->step, split);
        }
        if (!force_clone && !graph_subtree_has_upcast_subject(node, subject_by_id)) { return node; }
        if (clone_by_id[node->id] != nullptr) { return clone_by_id[node->id]; }

        Node *clone = create_node();
        clone->kind = node->kind;
        clone->step = node->step;
        clone->mask_combine_coef = node->mask_combine_coef;
        clone->cast_family = node->cast_family;
        clone->synthetic_dtype = node->synthetic_dtype;
        clone->synthetic_is_unsigned = node->synthetic_is_unsigned;
        clone->child_count = node->child_count;
        clone_by_id[node->id] = clone;
        for (size_t i = 0; i < node->child_count; ++i) {
            clone->children[i] = graph_clone_upcast_split(node->children[i], subject_by_id, clone_by_id, force_clone);
        }
        return clone;
    }

    bool graph_resolve_upcasts_once(bool &changed) {
        changed = false;
        if (!graph_assign()) { return false; }

        std::vector<RootState> new_roots{};
        new_roots.reserve(roots.size());
        for (const RootState &root : roots) {
            std::vector<UpcastSubject> subjects{};
            ArenaBitmap visited = ArenaBitmap::create(arena, id_counter());
            graph_collect_upcast_subjects(root.node, subjects, visited);
            if (subjects.empty()) {
                new_roots.push_back(root);
                continue;
            }
            std::vector<UpcastSubject> subject_by_id(id_counter());
            for (const UpcastSubject &subject : subjects) {
                subject_by_id[subject.node->id] = subject;
            }

            std::vector<Node *> low_clone_by_id(id_counter(), nullptr);
            std::vector<Node *> high_clone_by_id(id_counter(), nullptr);
            RootState low = root;
            low.node = graph_clone_upcast_split(root.node, subject_by_id, low_clone_by_id, true);
            low.row_block_idx *= 2;
            RootState high = root;
            high.node = graph_clone_upcast_split(root.node, subject_by_id, high_clone_by_id, true);
            high.row_block_idx = high.row_block_idx * 2 + 1;
            new_roots.push_back(low);
            new_roots.push_back(high);
            changed = true;
        }

        if (!changed) { return true; }
        if (!check_vector_root_limit(new_roots.size())) { return false; }
        roots = std::move(new_roots);
        return graph_assign();
    }

    bool graph_apply_upcast_subjects(const std::vector<UpcastSubject> &subject_by_id) {
        std::vector<RootState> new_roots{};
        new_roots.reserve(roots.size() + 1);
        bool changed = false;
        for (const RootState &root : roots) {
            if (!graph_subtree_has_upcast_subject(root.node, subject_by_id)) {
                new_roots.push_back(root);
                continue;
            }

            std::vector<Node *> low_clone_by_id(id_counter(), nullptr);
            std::vector<Node *> high_clone_by_id(id_counter(), nullptr);
            RootState low = root;
            low.node = graph_clone_upcast_split(root.node, subject_by_id, low_clone_by_id, true);
            low.row_block_idx *= 2;
            RootState high = root;
            high.node = graph_clone_upcast_split(root.node, subject_by_id, high_clone_by_id, true);
            high.row_block_idx = high.row_block_idx * 2 + 1;
            new_roots.push_back(low);
            new_roots.push_back(high);
            changed = true;
        }

        if (!changed) {
            return fail_graph_assignment(ErrorSubKind::UpcastSubjectUnreachable,
                                         "upcast normalization subject is not reachable from roots", true);
        }
        if (!check_vector_root_limit(new_roots.size())) { return false; }
        roots = std::move(new_roots);
        return graph_assign();
    }

    bool graph_apply_upcast_subject(const UpcastSubject &subject) {
        std::vector<UpcastSubject> subject_by_id(id_counter());
        subject_by_id[subject.node->id] = subject;
        return graph_apply_upcast_subjects(subject_by_id);
    }

    bool graph_resolve_upcasts() {
        for (size_t i = 0; i < 8; ++i) {
            bool changed = false;
            if (!graph_resolve_upcasts_once(changed)) { return false; }
            if (!changed) { return true; }
        }
        return fail_graph_assignment(ErrorSubKind::UpcastDidNotConverge, "upcast normalization did not converge");
    }

    MaskDataType graph_assigned_mask_dtype(const Node *node) {
        SIMJIT_ASSERT(assignment.has_coef.get(node->id));
        auto carrier = assignment.carrier_dtypes[node->id];
        if (!carrier) { return max_mask; }
        return vec_dtype_from_coef(*carrier, assignment.coefs[node->id]).mask();
    }

    MaskDataType graph_expected_mask_for_value(const Node *node) {
        if (is_synthetic_cast_node(node)) {
            SIMJIT_ASSERT(assignment.has_coef.get(node->id));
            return vec_dtype_from_coef(node->synthetic_dtype, assignment.coefs[node->id]).mask();
        }
        SIMJIT_ASSERT(node->is(NodeKind::Step));
        if (node->step->dtype == ScalarDataType::I1) { return graph_assigned_mask_dtype(node); }
        SIMJIT_ASSERT(assignment.has_coef.get(node->id));
        return vec_dtype_from_coef(node->step->dtype, assignment.coefs[node->id]).mask();
    }

    std::vector<ComponentCoefRange> graph_component_coef_ranges(nonstd::span<Node *const> nodes) const {
        int component_count = 0;
        for (const Node *node : nodes) {
            SIMJIT_ASSERT(assignment.component_ids[node->id] >= 0);
            component_count = std::max(component_count, assignment.component_ids[node->id] + 1);
        }

        std::vector<ComponentCoefRange> ranges(component_count);
        for (const Node *node : nodes) {
            SIMJIT_ASSERT(assignment.has_coef.get(node->id));
            int component_id = assignment.component_ids[node->id];
            ranges[component_id].min_coef = std::min(ranges[component_id].min_coef, assignment.coefs[node->id]);
            ranges[component_id].max_coef = std::max(ranges[component_id].max_coef, assignment.coefs[node->id]);
        }
        return ranges;
    }

    std::optional<ComponentRangeCast>
    graph_component_range_cast_candidate(const std::vector<Node *> &nodes,
                                         const std::vector<ComponentCoefRange> &ranges) {
        std::optional<ComponentRangeCast> best{};

        for (Node *node : nodes) {
            if (!node->is(NodeKind::Step)) { continue; }
            SIMJIT_ASSERT(assignment.has_coef.get(node->id));
            int component_id = assignment.component_ids[node->id];
            const ComponentCoefRange &range = ranges[component_id];
            if (range.max_coef - range.min_coef <= vec_range) { continue; }

            int high_limit = range.min_coef + vec_range;
            std::optional<VectCastInfo> info = graph_cast_info(node);
            if (!info) { continue; }
            const Node *arg = node->arg();

            if (info->direction == CastDirection::Narrow) {
                int delta = assignment.coefs[arg->id] - assignment.coefs[node->id];
                graph_consider_component_range_cast(best, ComponentRangeCast{ComponentRangeCastKind::Downcast, node},
                                                    assignment.coefs[arg->id], high_limit, delta);
                continue;
            }

            if (info->direction == CastDirection::Widen) {
                size_t split_size = scalar_dtype_size(info->from) * 2;
                if (split_size > scalar_dtype_size(info->to)) { continue; }
                int delta = assignment.coefs[node->id] - assignment.coefs[arg->id];
                graph_consider_component_range_cast(best, ComponentRangeCast{ComponentRangeCastKind::Upcast, node},
                                                    assignment.coefs[node->id], high_limit, delta);
                continue;
            }
        }

        return best;
    }

    Node *graph_create_component_range_downcast(Node *node) {
        std::optional<VectCastInfo> info = graph_cast_info(node);
        SIMJIT_ASSERT(info && info->direction == CastDirection::Narrow);
        Node *arg = node->arg();
        auto maybe_type = assignment.carrier_dtypes[arg->id];
        SIMJIT_ASSERT(maybe_type.has_value());
        ScalarDataType from = *maybe_type;
        ScalarDataType to = info->to;
        size_t from_size = scalar_dtype_size(from);
        size_t to_size = scalar_dtype_size(to);
        SIMJIT_ASSERT(from_size > to_size);

        Node *new_arg = arg;
        size_t combine_from_size = to_size * 2;
        if (from_size > combine_from_size) {
            new_arg = create_cast_direct(new_arg, *info, scalar_dtype_with_size(combine_from_size));
        }
        return create_cast_narrow_combine(new_arg, *info, to);
    }

    bool graph_apply_component_range_downcast(const Node *rewrite_node) {
        std::vector<Node *> nodes = collect_assignment_nodes(roots, id_counter(), arena);
        bool changed = false;
        for (Node *node : nodes) {
            if (!node->is(NodeKind::Step) || node->step != rewrite_node->step) { continue; }
            std::optional<VectCastInfo> info = graph_cast_info(node);
            if (!info || info->direction != CastDirection::Narrow) { continue; }
            const Node *arg = node->arg();
            auto from = assignment.carrier_dtypes[arg->id];
            if (!from || scalar_dtype_size(*from) <= scalar_dtype_size(info->to)) { continue; }
            replace_node(node, graph_create_component_range_downcast(node));
            changed = true;
        }

        if (!changed) {
            return fail_graph_assignment(ErrorSubKind::ComponentRangeSubjectUnreachable,
                                         "component range downcast subject is not reachable from roots", true);
        }
        return graph_assign();
    }

    std::optional<UpcastSubject> graph_component_range_upcast_subject(Node *node) {
        std::optional<VectCastInfo> info = graph_cast_info(node);
        if (!info || info->direction != CastDirection::Widen) { return {}; }
        size_t split_size = scalar_dtype_size(info->from) * 2;
        if (split_size > scalar_dtype_size(info->to)) { return {}; }
        ScalarDataType split_dtype = info->family == CastFamily::Int ? scalar_dtype_with_size(split_size) : info->to;
        return UpcastSubject{node, split_dtype, info->family, info->is_unsigned};
    }

    bool graph_apply_component_range_upcast(const ComponentRangeCast &rewrite) {
        std::vector<UpcastSubject> subject_by_id(id_counter());
        bool has_subject = false;
        std::vector<Node *> nodes = collect_assignment_nodes(roots, id_counter(), arena);
        for (Node *node : nodes) {
            if (!node->is(NodeKind::Step) || node->step != rewrite.node->step) { continue; }
            std::optional<UpcastSubject> subject = graph_component_range_upcast_subject(node);
            if (!subject) { continue; }
            subject_by_id[subject->node->id] = *subject;
            has_subject = true;
        }

        if (!has_subject) {
            return fail_graph_assignment(ErrorSubKind::ComponentRangeSubjectUnreachable,
                                         "component range upcast subject is not reachable from roots", true);
        }
        return graph_apply_upcast_subjects(subject_by_id);
    }

    bool graph_apply_component_range_cast(const ComponentRangeCast &rewrite) {
        switch (rewrite.kind) {
        case ComponentRangeCastKind::Downcast: return graph_apply_component_range_downcast(rewrite.node);
        case ComponentRangeCastKind::Upcast: return graph_apply_component_range_upcast(rewrite);
        }
        SIMJIT_UNREACHABLE();
    }

    bool graph_resolve_component_range_casts() {
        for (size_t i = 0; i < 64; ++i) {
            if (!graph_assign()) { return false; }
            std::vector<Node *> nodes = collect_assignment_nodes(roots, id_counter(), arena);
            std::vector<ComponentCoefRange> ranges = graph_component_coef_ranges(nodes);
            if (graph_component_ranges_fit(ranges, vec_range)) { return true; }

            std::optional<ComponentRangeCast> rewrite = graph_component_range_cast_candidate(nodes, ranges);
            if (!rewrite) {
                return fail_graph_assignment(ErrorSubKind::CoefficientRangeNeedsNormalization,
                                             "coefficient range needs normalization nodes");
            }
            if (!graph_apply_component_range_cast(*rewrite)) { return false; }
        }
        return fail_graph_assignment(ErrorSubKind::ComponentRangeDidNotConverge,
                                     "component range cast normalization did not converge");
    }

    std::optional<int> graph_min_coef_for_small_mask_root(const Node *root) const {
        if (mask_dtype_bits(max_mask) <= mask_dtype_bits(MaskDataType::M16)) { return {}; }
        bool needs_min_mask = (root->is_step(StepKind::Store) && root->step->dtype == ScalarDataType::I1) ||
                              root->is_step(StepKind::Countif);
        if (!needs_min_mask) { return {}; }

        auto carrier = assignment.carrier_dtypes[root->id];
        if (!carrier) { return {}; }
        for (int8_t coef = 0; coef <= vec_range; ++coef) {
            if (mask_dtype_bits(vec_dtype_from_coef(*carrier, coef).mask()) >= mask_dtype_bits(MaskDataType::M8)) {
                return coef;
            }
        }
        return {};
    }

    bool graph_normalize_coefs(const std::vector<Node *> &nodes) {
        // The expression-level root/list connects independent vector roots, but it should not make their arbitrary
        // graph zero-points fight each other. Each connected component can be shifted independently as long as its
        // internal coefficient range fits the target vector range.
        int component_count = 0;
        for (const Node *node : nodes) {
            SIMJIT_ASSERT(assignment.component_ids[node->id] >= 0);
            component_count = std::max(component_count, assignment.component_ids[node->id] + 1);
        }
        std::vector<int8_t> min_coefs(component_count, INT8_MAX);
        std::vector<int8_t> max_coefs(component_count, INT8_MIN);
        ArenaBitmap has_root = ArenaBitmap::create(arena, component_count);
        ArenaBitmap prefer_min_coef = ArenaBitmap::create(arena, component_count, true);
        std::vector<int8_t> min_adjusts(component_count, INT8_MIN);
        for (const Node *node : nodes) {
            SIMJIT_ASSERT(assignment.has_coef.get(node->id));
            int component_id = assignment.component_ids[node->id];

            min_coefs[component_id] = std::min(min_coefs[component_id], assignment.coefs[node->id]);
            max_coefs[component_id] = std::max(max_coefs[component_id], assignment.coefs[node->id]);
        }
        for (const RootState &root_info : roots) {
            const Node *root = root_info.node;
            int component_id = assignment.component_ids[root->id];
            SIMJIT_ASSERT(component_id >= 0);
            has_root.set(component_id);
            prefer_min_coef.set(component_id, prefer_min_coef.get(component_id) && graph_root_prefers_min_coef(root));
            if (auto min_coef = graph_min_coef_for_small_mask_root(root)) {
                min_adjusts[component_id] =
                    std::max(min_adjusts[component_id], int8_t(*min_coef - assignment.coefs[root->id]));
            }
        }
        std::vector<int8_t> adjusts(component_count, 0);
        for (int component_id = 0; component_id < component_count; ++component_id) {
            SIMJIT_ASSERT(min_coefs[component_id] != INT8_MAX && max_coefs[component_id] != INT8_MIN);
            if (max_coefs[component_id] - min_coefs[component_id] > vec_range) {
                return fail_graph_assignment(ErrorSubKind::CoefficientRangeNeedsNormalization,
                                             "coefficient range needs normalization nodes");
            }
            if (!has_root.get(component_id)) { prefer_min_coef.set(component_id, false); }
            if (prefer_min_coef.get(component_id)) {
                adjusts[component_id] = std::max(int8_t(-min_coefs[component_id]), min_adjusts[component_id]);
            } else {
                adjusts[component_id] = int8_t(vec_range - max_coefs[component_id]);
            }
            if (min_coefs[component_id] + adjusts[component_id] < 0 ||
                max_coefs[component_id] + adjusts[component_id] > vec_range) {
                return fail_graph_assignment(ErrorSubKind::CoefficientRangeNeedsNormalization,
                                             "coefficient range needs normalization nodes");
            }
        }

        for (const Node *node : nodes) {
            auto &x = assignment.coefs[node->id];
            x = int8_t(x + adjusts[assignment.component_ids[node->id]]);
            SIMJIT_ASSERT(x >= 0);
            SIMJIT_ASSERT(x <= vec_range);
        }
        return true;
    }

    std::optional<size_t> graph_calculate_expand_coef(const Node *node) {
        SIMJIT_ASSERT(node->is(NodeKind::Step));
        auto origin = assignment.carrier_dtypes[node->id];
        if (!origin) { return {}; }
        VecDataType vec_dtype = vec_dtype_from_coef(*origin, assignment.coefs[node->id]);
        MaskDataType mask_dtype = vec_dtype.mask();
        size_t mask_bits = mask_dtype_bits(mask_dtype);
        size_t target_bits = std::min(mask_dtype_bits(max_mask), mask_dtype_bits(MaskDataType::M32));
        if (target_bits % mask_bits != 0) { return {}; }
        size_t times = target_bits / mask_bits;
        return nonzero_log2(times);
    }

    void graph_update_coefs_according_to_mask(const Node *node, int8_t coef) {
        if (node->is(NodeKind::CombineMasks)) { return; }
        auto &x = assignment.coefs[node->id];
        x = int8_t(x + coef);

        for (auto child : node->children_span()) {
            graph_update_coefs_according_to_mask(child, coef);
        }
    }

    bool graph_has_shared_upper_tree(const Node *node) {
        std::vector<const Node *> worklist{node};
        ArenaBitmap visited = ArenaBitmap::create(arena, id_counter());
        while (!worklist.empty()) {
            const Node *cur = worklist.back();
            worklist.pop_back();
            if (visited.get(cur->id)) { continue; }
            visited.set(cur->id);
            if (assignment.parents[cur->id].size() > 1) { return true; }
            for (const Node *parent : assignment.parents[cur->id]) {
                worklist.push_back(parent);
            }
        }
        return false;
    }

    void graph_optimize_masks() {
        std::vector<Node *> mask_combineable_nodes{};
        std::vector<Node *> forbidden_mask_nodes{};
        ArenaBitmap is_mask_combineable_node = ArenaBitmap::create(arena, id_counter());
        ArenaBitmap is_forbidden_mask = ArenaBitmap::create(arena, id_counter());
        for (const RootState &root_info : roots) {
            Node *root = root_info.node;
            if (root->is_step(StepKind::AccPredicateBinary)) {
                find_mask_combineable_nodes(root->arg(), forbidden_mask_nodes, is_forbidden_mask);
                continue;
            }
            if (root->is_step(StepKind::Countif)) { continue; }
            if (root->is_step(StepKind::Store) && root->step->step_data<StepKind::Store>().cond != nullptr) {
                find_mask_combineable_nodes(root->right(), forbidden_mask_nodes, is_forbidden_mask);
                continue;
            }
            if (root->is_step(StepKind::AccArithBinary) &&
                root->step->step_data<StepKind::AccArithBinary>().cond != nullptr) {
                find_mask_combineable_nodes(root->right(), forbidden_mask_nodes, is_forbidden_mask);
                continue;
            }
            if (root->is_step(StepKind::Scatter) && root->step->step_data<StepKind::Scatter>().cond != nullptr) {
                find_mask_combineable_nodes(root->children[2], forbidden_mask_nodes, is_forbidden_mask);
                continue;
            }

            find_mask_combineable_nodes(root, mask_combineable_nodes, is_mask_combineable_node);
        }

        if (mask_combineable_nodes.empty()) return;
        if (graph_has_forbidden_mask(mask_combineable_nodes, is_forbidden_mask)) { return; }
        mask_combineable_nodes.erase(std::remove_if(mask_combineable_nodes.begin(), mask_combineable_nodes.end(),
                                                    [&](Node *node) { return graph_has_shared_upper_tree(node); }),
                                     mask_combineable_nodes.end());
        if (mask_combineable_nodes.empty()) return;

        std::vector<size_t> expand_coefs{};
        expand_coefs.reserve(mask_combineable_nodes.size());
        for (const Node *node : mask_combineable_nodes) {
            auto coef = graph_calculate_expand_coef(node);
            if (!coef) { return; }
            expand_coefs.push_back(*coef);
        }

        if (expand_coefs.size() > 1) {
            size_t min_coef = SIZE_MAX;
            for (size_t c : expand_coefs) {
                min_coef = std::min(c, min_coef);
            }
            if (min_coef != 0) {
                for (size_t &c : expand_coefs) {
                    c -= min_coef;
                }
            }
        }

        std::vector<Node *> mask_replacement_by_id(id_counter(), nullptr);
        bool has_replacement = false;
        size_t i = 0;
        for (Node *node : mask_combineable_nodes) {
            if (expand_coefs[i] == 0) {
                ++i;
                continue;
            }
            Node *combine_node = create_node();
            combine_node->kind = NodeKind::CombineMasks;
            combine_node->mask_combine_coef = expand_coefs[i];
            combine_node->child_count = 1;
            combine_node->children[0] = node;
            mask_replacement_by_id[node->id] = combine_node;
            has_replacement = true;
            ++i;
        }
        if (!has_replacement) { return; }

        for (const RootState &root_info : roots) {
            Node *root = root_info.node;
            int8_t coef = 0;
            if (!find_mask_coef(root, coef, mask_replacement_by_id)) continue;
            graph_update_coefs_according_to_mask(root, coef);
        }
    }

    DataType graph_value_dtype(const Node *node) {
        if (is_synthetic_cast_node(node)) {
            SIMJIT_ASSERT(assignment.coefs[node->id] >= 0);
            SIMJIT_ASSERT(assignment.coefs[node->id] <= vec_range);
            return vec_dtype_from_coef(node->synthetic_dtype, assignment.coefs[node->id]);
        }

        SIMJIT_ASSERT(node->is(NodeKind::Step));
        const Step *step = node->step;
        SIMJIT_ASSERT(assignment.coefs[node->id] >= 0);
        // FIXME: Enable this assert! Right now mask combine can mess up the normalization...
        // SIMJIT_ASSERT(assignment.coefs[node->id] <= vec_range);
        if (step->dtype == ScalarDataType::I1) {
            if (auto x = assignment.carrier_dtypes[node->id]) {
                VecDataType vec_dtype = vec_dtype_from_coef(*x, assignment.coefs[node->id]);
                return vec_dtype.mask();
            }
            return max_mask;
        }
        return vec_dtype_from_coef(step->dtype, assignment.coefs[node->id]);
    }

    bool graph_concretize_node(Node *node) {
        const Step *step = node->step;
        switch (step->kind) {
        case StepKind::PredicateBinary: node->dtype = node->left()->dtype; break;
        case StepKind::Compare: node->dtype = graph_value_dtype(node); break;
        case StepKind::Store: node->dtype = node->children[0]->dtype; break;
        case StepKind::Select: node->dtype = node->children[1]->dtype; break;
        case StepKind::Index:
        case StepKind::Const:
        case StepKind::ArithBinary:
        case StepKind::CheckedOp:
        case StepKind::ArithUnary:
        case StepKind::IntCast:
        case StepKind::FloatCast:
        case StepKind::Load:
        case StepKind::Gather:
        case StepKind::Fpclass:
        case StepKind::LoadSplat:
        case StepKind::BitCast: node->dtype = graph_value_dtype(node); break;
        case StepKind::PredicateNot:
        case StepKind::Permute: node->dtype = node->arg()->dtype; break;
        case StepKind::Scatter: node->dtype = node->children[1]->dtype; break;
        case StepKind::Pack: node->dtype = node->left()->dtype; break;
        case StepKind::AccPredicateBinary: {
            auto data = step->step_data<StepKind::AccPredicateBinary>();
            node->dtype = node->arg()->dtype;
            acc_dtypes[data.acc] = node->dtype;
            break;
        }
        case StepKind::Countif: {
            auto data = step->step_data<StepKind::Countif>();
            node->dtype = ScalarDataType::I64;
            acc_dtypes[data.acc] = ScalarDataType::I64;
            break;
        }
        case StepKind::AccArithBinary: {
            auto data = step->step_data<StepKind::AccArithBinary>();
            node->dtype = node->children[0]->dtype;
            acc_dtypes[data.acc] = node->dtype;
            break;
        }
        case StepKind::AccSum128: {
            auto data = step->step_data<StepKind::AccSum128>();
            node->dtype = node->arg()->dtype;
            acc_dtypes[data.acc] = node->dtype;
            break;
        }
        }

        node->coef = assignment.coefs[node->id];
        node->item_width = graph_item_width_from_dtype(node->dtype);
        if (node->item_width == 0 && node->child_count > 0) { node->item_width = node->children[0]->item_width; }
        SIMJIT_ASSERT(node->item_width != 0);
        return true;
    }

    bool graph_concretize_combine_masks(Node *node) {
        MaskDataType mask = node->arg()->dtype.as_mask();
        size_t item_width = node->arg()->item_width;
        for (size_t i = 0; i < node->mask_combine_coef; ++i) {
            auto new_mask = double_mask(mask);
            if (!new_mask.has_value()) {
                return fail_graph_assignment(ErrorSubKind::MaskCombineTooWide, "mask combine grows too wide");
            }
            mask = *new_mask;
            item_width *= 2;
        }
        node->dtype = mask;
        node->item_width = item_width;
        return true;
    }

    bool graph_concretize_cast_narrow_combine(Node *node) {
        node->dtype = graph_value_dtype(node);
        node->coef = assignment.coefs[node->id];
        node->item_width = graph_item_width_from_dtype(node->dtype);
        if (node->item_width != node->arg()->item_width * 2) {
            ErrorSubKind kind = node->cast_family == CastFamily::Float
                                    ? ErrorSubKind::FloatDowncastCombineItemWidthMismatch
                                    : ErrorSubKind::DowncastCombineItemWidthMismatch;
            return fail_graph_assignment(kind, "cast narrow combine item width mismatch", true);
        }
        return true;
    }

    bool graph_concretize_cast_direct(Node *node) {
        node->dtype = graph_value_dtype(node);
        node->coef = assignment.coefs[node->id];
        node->item_width = graph_item_width_from_dtype(node->dtype);
        if (node->item_width != node->arg()->item_width) {
            return fail_graph_assignment(ErrorSubKind::SyntheticIntCastItemWidthMismatch,
                                         "synthetic int cast item width mismatch", true);
        }
        return true;
    }

    bool graph_concretize_upcast_part(Node *node) {
        node->dtype = graph_value_dtype(node);
        node->coef = assignment.coefs[node->id];
        node->item_width = graph_item_width_from_dtype(node->dtype);
        if (node->arg()->item_width != node->item_width * 2) {
            return fail_graph_assignment(ErrorSubKind::UpcastHalfItemWidthMismatch, "upcast half item width mismatch",
                                         true);
        }
        return true;
    }

    bool graph_concretize_any(Node *node) {
        switch (node->kind) {
        case NodeKind::Step: return graph_concretize_node(node);
        case NodeKind::CombineMasks: return graph_concretize_combine_masks(node);
        case NodeKind::CastNarrowCombine: return graph_concretize_cast_narrow_combine(node);
        case NodeKind::CastDirect: return graph_concretize_cast_direct(node);
        case NodeKind::CastWidenPart: return graph_concretize_upcast_part(node);
        }
        SIMJIT_UNREACHABLE();
    }

    bool graph_concretize_types() {
        ArenaBitmap visited = ArenaBitmap::create(arena, id_counter());
        for (const RootState &root_info : roots) {
            Node *root = root_info.node;
            bool ok = true;
            unique_postorder_traverse(root, [&](Node *node) { ok = ok && graph_concretize_any(node); }, visited);
            if (!ok) { return false; }
        }
        return true;
    }

    bool graph_validate_same_width(const Node *node, const Node *left, const Node *right) {
        if (left->item_width != right->item_width) {
            return fail_graph_assignment(ErrorSubKind::WidthMismatch,
                                         simjit::format("width mismatch under %s (%u vs %u)",
                                                        show_step_kind(node->step->kind), left->item_width,
                                                        right->item_width),
                                         true);
        }
        return true;
    }

    bool graph_validate_node(const Node *node) {
        SIMJIT_ASSERT(node->is(NodeKind::Step));
        if (node->dtype.is_mask() && mask_dtype_bits(node->dtype.as_mask()) > mask_dtype_bits(max_mask)) {
            return fail_graph_assignment(ErrorSubKind::MaskDTypeTooWide,
                                         simjit::format("mask dtype %s is too wide", show_dtype(node->dtype)), true);
        }
        if (node->dtype.is_vec() && (node->coef < 0 || node->coef > vec_range)) {
            // XXX: This should be 'internal', but mask combine is buggy
            return fail_graph_assignment(ErrorSubKind::CoefficientRangeNeedsNormalization,
                                         simjit::format("coefficient %d is out of range", node->coef));
        }

        switch (node->step->kind) {
        case StepKind::ArithBinary:
        case StepKind::CheckedOp:
        case StepKind::Compare:
        case StepKind::PredicateBinary:
        case StepKind::Store:
        case StepKind::Pack:
            if (node->child_count == 2 && !graph_validate_same_width(node, node->left(), node->right())) {
                return false;
            }
            break;
        case StepKind::Select:
            if (!graph_validate_same_width(node, node->children[0], node->children[1])) { return false; }
            if (!graph_validate_same_width(node, node->children[1], node->children[2])) { return false; }
            break;
        case StepKind::Scatter:
            if (!graph_validate_same_width(node, node->children[0], node->children[1])) { return false; }
            if (node->child_count == 3 && !graph_validate_same_width(node, node->children[1], node->children[2])) {
                return false;
            }
            break;
        case StepKind::AccArithBinary:
        case StepKind::AccSum128:
            if (node->child_count == 2 && !graph_validate_same_width(node, node->left(), node->right())) {
                return false;
            }
            break;
        default: break;
        }
        return true;
    }

    bool graph_validate_any(const Node *node) {
        if (node->is(NodeKind::CombineMasks)) {
            if (node->dtype.is_mask() && mask_dtype_bits(node->dtype.as_mask()) > mask_dtype_bits(max_mask)) {
                return fail_graph_assignment(ErrorSubKind::MaskDTypeTooWide,
                                             simjit::format("mask dtype %s is too wide", show_dtype(node->dtype)),
                                             true);
            }
            return true;
        }
        if (is_synthetic_cast_node(node)) {
            if (node->dtype.is_vec() && (node->coef < 0 || node->coef > vec_range)) {
                return fail_graph_assignment(ErrorSubKind::CoefficientRangeNeedsNormalization,
                                             simjit::format("coefficient %d is out of range", node->coef), true);
            }
            return true;
        }
        return graph_validate_node(node);
    }

    std::vector<RootWidthGroup> graph_collect_root_width_groups(bool include_unroll) {
        std::vector<RootWidthGroup> groups{};
        for (size_t i = 0; i < roots.size(); ++i) {
            RootState &root_info = roots[i];
            Node *root = root_info.node;
            SIMJIT_ASSERT(root->is(NodeKind::Step));

            RootWidthGroup *group = nullptr;
            for (RootWidthGroup &candidate : groups) {
                if (candidate.logical_root_idx == root_info.logical_root_idx) {
                    group = &candidate;
                    break;
                }
            }
            if (group == nullptr) {
                groups.push_back(RootWidthGroup{root_info.logical_root_idx});
                group = &groups.back();
            }

            size_t width = root->item_width;
            if (include_unroll) { width <<= root_info.unroll_coef; }
            group->root_idxs.push_back(i);
            group->width += width;
        }
        return groups;
    }

    void graph_try_proactive_unroll(const ProactiveUnrollStats &stats) {
        size_t extra_coef = graph_proactive_unroll_extra_coef(stats);
        if (extra_coef == 0) { return; }
        SIMJIT_ASSERT(loop_width != 0);
        loop_width <<= extra_coef;
        for (RootState &root : roots) {
            root.unroll_coef += extra_coef;
        }
    }

    bool graph_validate(bool allow_proactive_unroll = false) {
        ProactiveUnrollStats stats{};
        stats.complex_op_cost = proactive_unroll_special_op_cost(hir->special_ops);
        ArenaBitmap visited = ArenaBitmap::create(arena, id_counter());
        for (RootState &root_info : roots) {
            Node *root = root_info.node;
            size_t stats_multiplier = size_t{1} << root_info.unroll_coef;
            graph_collect_proactive_unroll_stats(root, stats, stats_multiplier);

            bool ok = true;
            unique_postorder_traverse(
                root,
                [&](Node *node) {
                    if (node->is(NodeKind::Step)) {
                        if (is_aggregate_step(node->step)) { ++stats.agg_count; }
                        if (is_float_op_step(node->step)) { ++stats.float_op_count; }
                        if (is_complex_step(node->step)) { ++stats.complex_op_cost; }
                    }
                    ok = ok && graph_validate_any(node);
                },
                visited);
            if (!ok) { return false; }
        }

        if (allow_proactive_unroll && bool(hir->ctx->transformations & CodeTransformations::ProactiveUnroll)) {
            graph_try_proactive_unroll(stats);
        }

        for (const RootWidthGroup &group : graph_collect_root_width_groups(true)) {
            if (group.width != loop_width) {
                return fail_graph_assignment(
                    ErrorSubKind::RootWidthsMismatch,
                    simjit::format("root widths mismatch after unroll (%zu vs %zu)", loop_width, group.width), true);
            }
        }
        return true;
    }

    bool graph_unroll_roots() {
        std::vector<RootWidthGroup> groups = graph_collect_root_width_groups(false);
        SIMJIT_ASSERT(!groups.empty());

        size_t target_item_width = groups[0].width;
        bool any_smaller = false;
        for (const RootWidthGroup &group : groups) {
            if (group.width < target_item_width) {
                any_smaller = true;
            } else if (group.width > target_item_width) {
                any_smaller = true;
                target_item_width = group.width;
            }
        }
        loop_width = target_item_width;
        if (!has_single_bit(loop_width)) {
            return fail_graph_assignment(ErrorSubKind::RootWidthsNotPowerOfTwo,
                                         "root widths do not form power-of-two loop");
        }
        if (!any_smaller) { return true; }
        if (!bool(hir->ctx->transformations & CodeTransformations::Unroll)) {
            return fail_graph_assignment(ErrorSubKind::RootWidthsMismatch,
                                         "root widths require unroll, but unroll is disabled");
        }

        for (const RootWidthGroup &group : groups) {
            if (group.width == target_item_width) { continue; }
            if (target_item_width % group.width != 0) {
                return fail_graph_assignment(ErrorSubKind::RootWidthsNotDivisible,
                                             "root widths are not divisible for unroll");
            }
            for (size_t root_idx : group.root_idxs) {
                roots[root_idx].unroll_coef = nonzero_log2(target_item_width / group.width);
            }
        }
        return true;
    }

    void graph_validate_root_groups() {
        for (const RootWidthGroup &group : graph_collect_root_width_groups(false)) {
            SIMJIT_ASSERT(!group.root_idxs.empty());
            uint8_t unroll_coef = roots[group.root_idxs[0]].unroll_coef;
            for (size_t root_idx : group.root_idxs) {
                SIMJIT_ASSERT(roots[root_idx].unroll_coef == unroll_coef);
            }
            SIMJIT_ASSERT((group.width << unroll_coef) == loop_width);
        }
    }

    bool try_vectorize() {
        if (!graph_resolve_downcasts()) { return false; }
        if (!graph_resolve_upcasts()) { return false; }
        if (!graph_resolve_component_range_casts()) { return false; }
        if (!check_vector_root_limit(roots.size())) { return false; }
        std::vector<Node *> nodes = collect_assignment_nodes(roots, id_counter(), arena);
        if (!graph_normalize_coefs(nodes)) { return false; }
        if (bool(hir->ctx->transformations & CodeTransformations::MaskCombine)) { graph_optimize_masks(); }
        if (!graph_concretize_types()) { return false; }

        if (!graph_unroll_roots()) { return false; }
        if (!graph_validate(true)) { return false; }
        graph_validate_root_groups();
        return true;
    }
};

} // namespace

static void print_tree_rec(Node *node, std::string &str, std::vector<size_t> &print_index_by_id,
                           size_t &next_print_index, size_t depth = 0) {
    for (size_t i = 0; i < depth; ++i)
        str += " ";

    if (print_index_by_id[node->id] != SIZE_MAX) {
        simjit::format_to(str, "[%zu] ", print_index_by_id[node->id]);
    } else {
        print_index_by_id[node->id] = next_print_index++;
        simjit::format_to(str, "[%zu] ", print_index_by_id[node->id]);
    }

    switch (node->kind) {
    case NodeKind::Step:
        simjit::format_to(str, "node step='%s' coef=%d dtype=%s item_width=%u\n", show_step_kind(node->step->kind),
                          node->coef, show_dtype(node->dtype), node->item_width);
        break;
    case NodeKind::CombineMasks:
        simjit::format_to(str, "node combine_masks mask_coef=%u dtype=%s item_width=%u\n", node->mask_combine_coef,
                          show_dtype(node->dtype), node->item_width);
        break;
    case NodeKind::CastDirect:
    case NodeKind::CastNarrowCombine:
    case NodeKind::CastWidenPart: {
        const char *kind = "";
        switch (node->kind) {
        case NodeKind::CastDirect: kind = "cast_direct"; break;
        case NodeKind::CastNarrowCombine: kind = "cast_narrow_combine"; break;
        case NodeKind::CastWidenPart: kind = "cast_widen_part"; break;
        default: SIMJIT_UNREACHABLE();
        }
        simjit::format_to(str,
                          "node %s family=%s target=%s is_unsigned=%s coef=%d dtype=%s "
                          "item_width=%u\n",
                          kind, node->cast_family == CastFamily::Float ? "float" : "int",
                          show_scalar_dtype(node->synthetic_dtype), node->synthetic_is_unsigned ? "true" : "false",
                          node->coef, show_dtype(node->dtype), node->item_width);
        break;
    }
    }

    for (auto child : node->children_span()) {
        print_tree_rec(child, str, print_index_by_id, next_print_index, depth + 1);
    }
}

std::string print_function(const Function *result) {
    std::vector<size_t> print_index_by_id(result->node_id_count, SIZE_MAX);
    size_t next_print_index = 0;
    std::string str = "";
    for (const Root &root : result->roots) {
        simjit::format_to(str, "root logical=%zu block=(%zu,%zu) unroll=%u\n", root.logical_root_idx, root.block.width,
                          root.block.idx, root.unroll_coef);
        print_tree_rec(root.node, str, print_index_by_id, next_print_index);
    }
    return str;
}

static Function *make_vectorization_result(AlgebraicVectorizer &state) {
    MemoryArena *arena = state.arena;
    Function *result = arena->create<Function>();
    result->hir = state.hir;
    result->node_id_count = state.id_counter();
    result->acc_dtypes = arena->copy_array<DataType>(state.acc_dtypes);
    result->roots = arena->alloc_array<Root>(state.roots.size());
    for (size_t i = 0; i < state.roots.size(); ++i) {
        const RootState &root = state.roots[i];
        result->roots[i] = Root{root.node, root.logical_root_idx, RowBlock{root.node->item_width, root.row_block_idx},
                                root.unroll_coef};
    }
    result->loop_width = state.loop_width;
    return result;
}

nonstd::expected<Function *, ErrorInfo> try_hir_to_vect(const hir::Function *func) {
    SpecialOp unsupported_ops = unsupported_vector_special_ops(func);
    if (unsupported_ops != SpecialOp::None) {
        return nonstd::unexpected<ErrorInfo>(ErrorInfo{
            ErrorModule::Vectorizer,
            ErrorKind::VectorizationFailed,
            ErrorSubKind::UnsupportedSpecialOps,
            simjit::format("vectorization unsupported for %s", show_special_ops(unsupported_ops).c_str()),
        });
    }
    AlgebraicVectorizer algebraic = AlgebraicVectorizer{func};
    if (algebraic.try_vectorize()) { return make_vectorization_result(algebraic); }
    ErrorKind kind = algebraic.graph_failure.internal ? ErrorKind::InternalInvariant : ErrorKind::VectorizationFailed;
    return nonstd::unexpected<ErrorInfo>(ErrorInfo{
        ErrorModule::Vectorizer,
        kind,
        algebraic.graph_failure.kind,
        algebraic.graph_failure.message,
    });
}

Function *hir_to_vect(const hir::Function *func) {
    auto result = try_hir_to_vect(func);
    if (result) { return result.value(); }
    throw SimjitException(std::move(result.error()));
}

} // namespace vect
} // namespace simjit
