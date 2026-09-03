// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/hir.h"
#include "simjit/core/expr.h"
#include "simjit/detail/base.h"

#include <vector>

namespace simjit {
namespace hir {

const char *show_step_kind(StepKind kind) noexcept {
    switch (kind) {
    case StepKind::LoadSplat: return "load-splat";
    case StepKind::Const: return "const";
    case StepKind::Load: return "load";
    case StepKind::Gather: return "gather";
    case StepKind::ArithBinary: return "binary";
    case StepKind::CheckedOp: return "checked-op";
    case StepKind::PredicateBinary: return "predicate-binary";
    case StepKind::ArithUnary: return "unary";
    case StepKind::IntCast: return "int-cast";
    case StepKind::FloatCast: return "float-cast";
    case StepKind::Store: return "store";
    case StepKind::Compare: return "cmp";
    case StepKind::AccArithBinary: return "acc-arith-bin";
    case StepKind::AccPredicateBinary: return "acc-predicate-bin";
    case StepKind::PredicateNot: return "predicate-not";
    case StepKind::Select: return "select";
    case StepKind::Index: return "index";
    case StepKind::Scatter: return "scatter";
    case StepKind::Pack: return "pack";
    case StepKind::AccSum128: return "sum128";
    case StepKind::Permute: return "permute";
    case StepKind::BitCast: return "bitcast";
    case StepKind::Fpclass: return "fpclass";
    case StepKind::Countif: return "countif";
    }
    SIMJIT_UNREACHABLE();
}

std::string show_special_ops(SpecialOp ops) {
    if (ops == SpecialOp::None) { return "none"; }
    std::string result;
    result.reserve(64);
    auto append = [&](SpecialOp op, const char *name) {
        if (!bool(ops & op)) { return; }
        if (!result.empty()) { result += ","; }
        result += name;
    };
    append(SpecialOp::I64Mul, "i64-mul");
    append(SpecialOp::Gather, "gather");
    append(SpecialOp::Scatter, "scatter");
    append(SpecialOp::CondScatter, "cond-scatter");
    append(SpecialOp::SmallPack, "small-pack");
    append(SpecialOp::ArbitraryBitPermute, "arbitrary-bit-permute");
    append(SpecialOp::I8Mul, "i8-mul");
    append(SpecialOp::I8VariableShift, "i8-variable-shift");
    append(SpecialOp::SmallLzcnt, "small-lzcnt");
    append(SpecialOp::SmallGather, "small-gather");
    return result;
}

static void show_step(const Step *step, nonstd::span<uint32_t const> show_cache, std::string &buf) {
#define wr_(...) simjit::format_to(buf, __VA_ARGS__)
#define wr(...)           \
    do {                  \
        wr_(__VA_ARGS__); \
        return;           \
    } while (0)
#define ref(step_) ((unsigned)show_cache[(step_)->id])
    wr_("%s dtype=%s ", show_step_kind(step->kind), show_scalar_dtype(step->dtype));

    switch (step->kind) {
        SIMJIT_MATCH (StepKind::Index) return;
        SIMJIT_MATCH (StepKind::Const) wr("value=%s", show_const_data(data, step->dtype).c_str());
        SIMJIT_MATCH2 (StepKind::Load, StepKind::LoadSplat)
            wr("arg=@%zu kind=%s", data.idx, show_load_store_kind(data.kind));
        SIMJIT_MATCH (StepKind::BitCast) wr("arg=%%%u", ref(data));
        SIMJIT_MATCH (StepKind::Gather) wr("arg=@%zu idx=%%%u", data.data, ref(data.idx));
        SIMJIT_MATCH (StepKind::Scatter) {
            wr_("dst=@%zu idx=%%%u arg=%%%u", data.dst, ref(data.idx), ref(data.arg));
            if (data.cond != nullptr) wr_(" cond=%%%u", ref(data.cond));
            return;
        }
        SIMJIT_MATCH (StepKind::ArithBinary)
            wr("op=%s left=%%%u right=%%%u flags=%s", show_arith_binary_op(data.op), ref(data.left), ref(data.right),
               show_arith_binary_flags(data.flags).c_str());
        SIMJIT_MATCH (StepKind::CheckedOp) {
            wr_("op=%%%u", ref(data.op));
            if (data.mask != nullptr) wr_(" mask=%%%u", ref(data.mask));
            return;
        }
        SIMJIT_MATCH (StepKind::PredicateBinary)
            wr("op=%s left=%%%u right=%%%u", show_predicate_binary_op(data.op), ref(data.left), ref(data.right));
        SIMJIT_MATCH (StepKind::ArithUnary) wr("op=%s arg=%%%u", show_arith_unary_op(data.op), ref(data.arg));
        SIMJIT_MATCH (StepKind::IntCast) wr("arg=%%%u kind=%s", ref(data.arg), show_int_cast_kind(data.kind));
        SIMJIT_MATCH (StepKind::FloatCast)
            wr("arg=%%%u is_unsigned=%s", ref(data.arg), data.is_unsigned ? "true" : "false");
        SIMJIT_MATCH (StepKind::PredicateNot) wr("arg=%%%u", ref(data));
        SIMJIT_MATCH (StepKind::Store) {
            wr_("dst=@%zu src=%%%u kind=%s", data.addr, ref(data.what), show_load_store_kind(data.kind));
            if (data.cond != nullptr) wr_(" cond=%%%u", ref(data.cond));
            return;
        }
        SIMJIT_MATCH (StepKind::Compare)
            wr("op=%s left=%%%u right=%%%u is_unsigned=%s", show_cmp_op(data.op), ref(data.left), ref(data.right),
               data.is_unsigned ? "true" : "false");
        SIMJIT_MATCH2 (StepKind::AccArithBinary, StepKind::AccSum128) {
            wr_("op=%s acc=$%zu arg=%%%u", show_arith_binary_op(data.op), data.acc, ref(data.arg));
            if (data.cond != nullptr) wr_(" cond=%%%u", ref(data.cond));
            return;
        }
        SIMJIT_MATCH (StepKind::Countif)
            wr("op=%s acc=$%zu arg=%%%u", show_arith_binary_op(data.op), data.acc, ref(data.arg));
        SIMJIT_MATCH (StepKind::AccPredicateBinary)
            wr("op=%s acc=$%zu arg=%%%u", show_predicate_binary_op(data.op), data.acc, ref(data.arg));
        SIMJIT_MATCH (StepKind::Select)
            wr("cond=%%%u truthy=%%%u falsy=%%%u", ref(data.cond), ref(data.truthy), ref(data.falsy));
        SIMJIT_MATCH (StepKind::Pack)
            wr("arg=%%%u cond=%%%u dst=@%zu dst_size=$%zu", ref(data.arg), ref(data.cond), data.dst, data.dst_size_acc);
        SIMJIT_MATCH (StepKind::Permute)
            wr("is_bit=%s arg=%%%u permute=%llx", data.is_bit ? "true" : "false", ref(data.arg),
               (unsigned long long)data.permute);
        SIMJIT_MATCH (StepKind::Fpclass) wr("arg=%%%u flags=%s", ref(data.arg), show_fpclass(data.flags).c_str());
    }
#undef wr
#undef wr_
#undef ref
    SIMJIT_UNREACHABLE();
}

std::string print_function(const Function *func) {
    std::string buf;
    buf.reserve(1024);
    if (!func->args.empty()) {
        for (const ArgumentDecl &arg : func->args) {
            simjit::format_to(buf, "@%zu arg dtype=%s kind=%s\n", arg.idx, show_scalar_dtype(arg.dtype),
                              show_argument_kind(arg.kind).c_str());
        }
    }
    if (!func->accs.empty()) {
        for (const Accumulator &acc : func->accs) {
            simjit::format_to(buf, "$%zu acc dtype=%s arg=@%zu\n", acc.idx, show_scalar_dtype(acc.dtype), acc.dst_arg);
        }
    }
    if (func->special_ops != SpecialOp::None) {
        simjit::format_to(buf, "# special-ops=%s\n", show_special_ops(func->special_ops).c_str());
    }
    std::vector<uint32_t> show_cache(func->step_id_count, 0);
    std::vector<uint8_t> traversal_state(func->step_id_count, 0);
    uint32_t counter = 1;
    for (Step *root : func->step_roots) {
        traverse_steps_postorder_unique(root, traversal_state, [&](Step *step) {
            if (show_cache[step->id] == 0) {
                uint32_t idx = counter++;
                show_cache[step->id] = idx;
                simjit::format_to(buf, "%%%u <- ", idx);
                show_step(step, show_cache, buf);
                simjit::format_to(buf, "\n");
            }
        });
    }
    return buf;
}
} // namespace hir
} // namespace simjit
