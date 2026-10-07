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
        if (!result.empty()) { result += ','; }
        result += name;
    };
    append(SpecialOp::I64Mul, "i64-mul");
    append(SpecialOp::Gather, "gather");
    append(SpecialOp::Scatter, "scatter");
    append(SpecialOp::CondScatter, "cond-scatter");
    append(SpecialOp::SmallPack, "small-pack");
    append(SpecialOp::ArbitraryBitPermute, "arbitrary-bit-permute");
    append(SpecialOp::SmallGather, "small-gather");
    append(SpecialOp::SmallPopcount, "small-popcount");
    append(SpecialOp::LargePopcount, "large-popcount");
    append(SpecialOp::I8ConstantShift, "i8-constant-shift");
    return result;
}

SpecialOp supported_vector_special_ops_for_arch(Arch arch) noexcept {
    switch (arch) {
    case Arch::Amd64_AVX512:
    case Arch::Amd64_AVX512_YMM:
        // We treat Ice Lake as baseline supported version - it has everything we need. Other versions are treated as
        // 'reduced ice lake', where some special ops will be removed from supported list.
        return SpecialOp::I64Mul | SpecialOp::Gather | SpecialOp::Scatter | SpecialOp::CondScatter |
               SpecialOp::SmallPack | SpecialOp::ArbitraryBitPermute | SpecialOp::SmallPopcount |
               SpecialOp::LargePopcount | SpecialOp::I8ConstantShift;
    case Arch::Arm64_NEON:
        return SpecialOp::Gather | SpecialOp::SmallPack | SpecialOp::SmallPopcount | SpecialOp::LargePopcount |
               SpecialOp::I8ConstantShift;
    }
    SIMJIT_UNREACHABLE();
}

SpecialOp Function::unsupported_vector_special_ops() const noexcept {
    return special_ops & ~(supported_vector_special_ops_for_arch(ctx->arch) & ctx->host_supported_vector_special_ops);
}

namespace {

const char *show_argument_access(ArgumentUsage access) {
    switch (access) {
    case ArgumentUsage::SequentialRead: return "sequential read";
    case ArgumentUsage::SequentialWrite: return "sequential write";
    case ArgumentUsage::RandomRead: return "random read";
    case ArgumentUsage::RandomWrite: return "random write";
    case ArgumentUsage::AppendWrite: return "append write";
    case ArgumentUsage::AggregateWrite: return "aggregate write";
    default: break;
    }
    SIMJIT_UNREACHABLE();
}

constexpr ArgumentUsage compatible_argument_access_flags(ArgumentUsage access) {
    switch (access) {
    case ArgumentUsage::SequentialRead:
        return ArgumentUsage::SequentialRead | ArgumentUsage::SequentialWrite | ArgumentUsage::RandomRead;
    case ArgumentUsage::SequentialWrite: return ArgumentUsage::SequentialRead;
    case ArgumentUsage::RandomRead:
        return ArgumentUsage::SequentialRead | ArgumentUsage::RandomRead | ArgumentUsage::RandomWrite;
    case ArgumentUsage::RandomWrite: return ArgumentUsage::RandomRead | ArgumentUsage::RandomWrite;
    default: return ArgumentUsage::None;
    }
    SIMJIT_UNREACHABLE();
}

ArgumentUsage first_argument_access(ArgumentUsage flags) {
    if (bool(flags & ArgumentUsage::SequentialRead)) { return ArgumentUsage::SequentialRead; }
    if (bool(flags & ArgumentUsage::SequentialWrite)) { return ArgumentUsage::SequentialWrite; }
    if (bool(flags & ArgumentUsage::RandomRead)) { return ArgumentUsage::RandomRead; }
    if (bool(flags & ArgumentUsage::RandomWrite)) { return ArgumentUsage::RandomWrite; }
    if (bool(flags & ArgumentUsage::AppendWrite)) { return ArgumentUsage::AppendWrite; }
    if (bool(flags & ArgumentUsage::AggregateWrite)) { return ArgumentUsage::AggregateWrite; }
    SIMJIT_UNREACHABLE();
}

void add_argument_access(std::vector<ArgumentUsage> &usages, ArgumentIdx idx, ArgumentUsage access) {
    SIMJIT_ASSERT(idx < usages.size());
    ArgumentUsage &old_flags = usages[idx];
    ArgumentUsage incompatible_flags =
        (old_flags & ArgumentUsage::AccessMask) & ~compatible_argument_access_flags(access);
    if (incompatible_flags != ArgumentUsage::None) {
        ArgumentUsage old_access = first_argument_access(incompatible_flags);
        simjit_exception(ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::InvalidArgumentAccess,
                         "Argument %zu has incompatible accesses: %s and %s", idx, show_argument_access(old_access),
                         show_argument_access(access));
    }
    old_flags |= access;
}

void add_step_argument_accesses(std::vector<ArgumentUsage> &usages, const Step *step) {
    switch (step->kind) {
    case StepKind::Load:
        add_argument_access(usages, step->step_data<StepKind::Load>().idx, ArgumentUsage::SequentialRead);
        usages[step->step_data<StepKind::Load>().idx] |= ArgumentUsage::ArrayRead;
        return;
    case StepKind::LoadSplat:
        add_argument_access(usages, step->step_data<StepKind::LoadSplat>().idx, ArgumentUsage::SequentialRead);
        usages[step->step_data<StepKind::LoadSplat>().idx] |= ArgumentUsage::SplatRead;
        return;
    case StepKind::Gather:
        add_argument_access(usages, step->step_data<StepKind::Gather>().data, ArgumentUsage::RandomRead);
        return;
    case StepKind::Store:
        add_argument_access(usages, step->step_data<StepKind::Store>().addr, ArgumentUsage::SequentialWrite);
        return;
    case StepKind::Scatter:
        add_argument_access(usages, step->step_data<StepKind::Scatter>().dst, ArgumentUsage::RandomWrite);
        return;
    case StepKind::Pack:
        add_argument_access(usages, step->step_data<StepKind::Pack>().dst, ArgumentUsage::AppendWrite);
        return;
    default: return;
    }
}

void mark_bounded_index_inputs(std::vector<ArgumentUsage> &usages, Step *idx, size_t step_count) {
    std::vector<uint8_t> state(step_count, 0);
    traverse_steps_postorder_unique(idx, state, [&](Step *step) {
        if (step->is(StepKind::Load) || step->is(StepKind::LoadSplat)) {
            usages[step->step_data<StepKind::Load>().idx] |= ArgumentUsage::BoundedIndex;
        }
    });
}

} // namespace

void analyze_argument_usage(Function *func) {
    std::vector<ArgumentUsage> usages(func->args.size());
    std::vector<uint8_t> state(func->step_id_count, 0);
    for (Step *root : func->step_roots) {
        traverse_steps_postorder_unique(root, state, [&](Step *step) {
            add_step_argument_accesses(usages, step);
            if (step->is(StepKind::Gather)) {
                mark_bounded_index_inputs(usages, step->step_data<StepKind::Gather>().idx, func->step_id_count);
            } else if (step->is(StepKind::Scatter)) {
                mark_bounded_index_inputs(usages, step->step_data<StepKind::Scatter>().idx, func->step_id_count);
            }
        });
    }
    for (const Accumulator &acc : func->accs) {
        add_argument_access(usages, acc.dst_arg, ArgumentUsage::AggregateWrite);
    }
    if (func->safety_check_arg) {
        add_argument_access(usages, *func->safety_check_arg, ArgumentUsage::AggregateWrite);
        usages[*func->safety_check_arg] |= ArgumentUsage::SafetyCheck;
    }
    func->argument_usage = func->ctx->arena->copy_array<ArgumentUsage>(usages);
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
            simjit::format_to(buf, "@%zu arg dtype=%s\n", arg.idx, show_scalar_dtype(arg.dtype));
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
    if (func->vectorization_hint == VectorizationHint::UnsupportedSpecialOps) {
        simjit::format_to(buf, "# unsupported-vector-special-ops=%s\n",
                          show_special_ops(func->unsupported_vector_special_ops()).c_str());
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
