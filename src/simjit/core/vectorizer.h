// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/core/expr.h"
#include "simjit/core/hir.h"

namespace simjit {
namespace vect {

enum class NodeKind : uint8_t {
    Step,
    CombineMasks,
    CastDirect,
    CastNarrowCombine,
    CastWidenPart,
};

enum class CastFamily : uint8_t {
    Int,
    Float,
};

// Notice a bit funny difference in names: this is Node, unlike Step in HIR and MIR.
// This is because it does not have step_data and similar accessors, as well as it does not support SIMJIT_MATCH.
struct Node {
    const hir::Step *step = nullptr;
    Node *children[3];
    uint8_t item_width = 0;
    uint8_t mask_combine_coef = 0;
    uint8_t child_count = 0;
    uint32_t id = 0;
    DataType dtype;
    NodeKind kind = NodeKind::Step;
    int8_t coef = 0;
    CastFamily cast_family = CastFamily::Int;
    ScalarDataType synthetic_dtype = ScalarDataType::I1;
    bool synthetic_is_unsigned = false;

    constexpr bool is(NodeKind k) const noexcept { return kind == k; }
    constexpr bool is_step(hir::StepKind step_kind) const noexcept { return is(NodeKind::Step) && step->is(step_kind); }

    nonstd::span<Node *> children_span() noexcept {
        SIMJIT_ASSERT(child_count <= sizeof(children) / sizeof(children[0]));
        return nonstd::span{children, child_count};
    }

    nonstd::span<const Node *const> children_span() const noexcept {
        SIMJIT_ASSERT(child_count <= sizeof(children) / sizeof(children[0]));
        return nonstd::span{children, child_count};
    }

    Node *arg() noexcept {
        SIMJIT_ASSERT(child_count == 1);
        return children[0];
    }

    Node *left() noexcept {
        SIMJIT_ASSERT(child_count == 2);
        return children[0];
    }

    Node *right() noexcept {
        SIMJIT_ASSERT(child_count == 2);
        return children[1];
    }

    const Node *arg() const noexcept {
        SIMJIT_ASSERT(child_count == 1);
        return children[0];
    }

    const Node *left() const noexcept {
        SIMJIT_ASSERT(child_count == 2);
        return children[0];
    }

    const Node *right() const noexcept {
        SIMJIT_ASSERT(child_count == 2);
        return children[1];
    }
};

struct RowBlock {
    size_t width = 0;
    size_t idx = 0;

    size_t row_offset() const noexcept { return width * idx; }

    RowBlock nth_slice(size_t slice_idx, size_t slice_count) const noexcept {
        SIMJIT_ASSERT(width != 0 && slice_count != 0 && slice_idx < slice_count && width % slice_count == 0);
        return RowBlock{width / slice_count, idx * slice_count + slice_idx};
    }

    RowBlock lhs_half() const noexcept { return nth_slice(0, 2); }
    RowBlock rhs_half() const noexcept { return nth_slice(1, 2); }

    RowBlock containing_block(size_t containing_width) const noexcept {
        SIMJIT_ASSERT(width != 0 && containing_width >= width && containing_width % width == 0);
        size_t slice_count = containing_width / width;
        return RowBlock{containing_width, idx / slice_count};
    }

    size_t index_in_containing_block(size_t containing_width) const noexcept {
        SIMJIT_ASSERT(width != 0 && containing_width >= width && containing_width % width == 0);
        return idx % (containing_width / width);
    }

    RowBlock nth_copy(size_t copy_idx, size_t copy_width) const noexcept {
        SIMJIT_ASSERT(width != 0 && copy_width >= width && copy_width % width == 0);
        return RowBlock{width, idx + copy_idx * (copy_width / width)};
    }
};

struct Root {
    Node *node = nullptr;
    size_t logical_root_idx = 0;
    RowBlock block{};
    uint8_t unroll_coef{};
};

struct Function {
    const hir::Function *hir = nullptr;
    ArenaArray<DataType> acc_dtypes{};
    ArenaArray<Root> roots{};
    size_t node_id_count = 0;
    size_t loop_width = 0;
};

} // namespace vect
} // namespace simjit
