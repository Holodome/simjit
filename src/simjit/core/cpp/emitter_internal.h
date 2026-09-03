// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/core/expr.h"
#include "simjit/core/mir.h"
#include "simjit/detail/base.h"
#include "simjit/simjit.h"

#include <array>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace simjit {
namespace cpp_backend {

using namespace ::simjit::mir;

const char *scalar_dtype_to_cpp(ScalarDataType dtype);
const char *scalar_dtype_to_cpp_unsigned(ScalarDataType dtype);
const char *arith_binary_op_to_cpp(ArithBinaryOp op);
bool is_shift_or_rotate_op(ArithBinaryOp op);
std::string format_scalar_arith_binary_expr(ScalarDataType result_dtype, ArithBinaryOp op, ScalarDataType left_dtype,
                                            ScalarDataType right_dtype, const char *left, const char *right,
                                            const std::optional<std::string> &shift_rhs);
const char *cmp_op_to_cpp(CmpOp op);
const char *indent_str(size_t indent);
std::string const_data_to_cpp(ConstData data, ScalarDataType dt);

class CppEmitterBase {
public:
    explicit CppEmitterBase(const Function *f);
    virtual ~CppEmitterBase() noexcept;

    std::string emit_source();

protected:
    enum class StepNameKind {
        None,
        Temp,
        Const,
        Load,
        Acc,
    };

    enum class EmitPhase : uint8_t {
        Prologue = 0,
        MainLoop = 1,
        Remainder = 2,
        Epilogue = 3,
    };

    struct MaskPushdownInfo {
        const Step *mask{};
        const Step *merge{};
        const Step *result{};
    };

    enum class PrologueRootFilter {
        All,
        Immediate,
        Delayed
    };

    const Function *func;
    size_t const_count = 0;
    std::array<size_t, 4> load_counts{};
    std::array<size_t, 4> temp_counts{};
    std::vector<std::string> seen{};
    std::vector<size_t> use_counts{};
    std::vector<size_t> peephole_const_use_counts{};
    std::vector<size_t> peephole_step_use_counts{};
    std::vector<size_t> peephole_named_use_counts{};
    std::vector<std::optional<std::string>> inline_scalar_exprs{};
    std::vector<DataType> acc_dtypes{};
    ArenaBitmap acc_declared{};
    ArenaBitmap main_loop_acc_uses{};

    virtual bool enable_vector_peepholes() const;
    virtual const char *cpp_dtype(DataType dtype) const;
    virtual void prepare_emit();
    virtual std::string source_prelude_to_cpp() const;
    virtual std::string function_prelude_to_cpp(size_t indent_level) const;
    virtual std::string before_loops_to_cpp(size_t indent_level);
    virtual std::string after_loops_to_cpp(size_t indent_level);
    virtual bool has_custom_step_to_cpp(const Step *step, EmitPhase phase) const;
    virtual std::string custom_step_to_cpp(const Step *step, EmitPhase phase);
    virtual bool suppress_inlined_acc_store_args(const Step *step, ArenaBitmap &suppressed) const;
    virtual bool suppress_compacted_acc_store_args(const Step *step, ArenaBitmap &suppressed) const;
    virtual std::optional<std::string> backend_compact_acc_store_to_cpp(const Step *step);
    virtual std::string backend_step_to_cpp(const Step *step, const MaskPushdownInfo *mask_pushdown) = 0;
    virtual std::string store_sum128_to_cpp(const StoreSum128Data &data) = 0;
    virtual void record_backend_peephole_uses(Step *step);

    const char *show(const Step *step) const;
    const char *show_scalar_operand(const Step *step);
    const char *cpp_var_decl_qualifier(const Step *step) const;
    std::string cpp_var_decl(const Step *step) const;

    static const char *temp_prefix(EmitPhase phase);
    static const char *load_prefix(EmitPhase phase);
    static StepNameKind classify_step_name(const Step *step);
    std::optional<std::string> assign_step_name(const Step *step, EmitPhase phase);

    void init_use_counts();
    void init_accumulators();
    void mark_main_loop_acc_use(AccId acc);
    void record_main_loop_acc_uses();
    bool prologue_root_is_delayed(const Step *root) const;
    bool has_delayed_prologue_steps(nonstd::span<Step *const> steps) const;
    void init_inline_scalar_exprs();

    std::string show_inline_operand(const Step *step, const Step *acc_load, AccId acc);
    const Step *find_compacted_acc_load(AccId acc, std::initializer_list<const Step *> candidates,
                                        std::initializer_list<const Step *> counted_uses) const;
    const Step *find_compacted_acc_load(AccId acc, std::initializer_list<const Step *> operands) const;
    std::string format_acc_store(AccId acc, std::string_view rhs);

    bool can_inline_scalar_expr(const Step *step) const;
    std::optional<std::string> inline_scalar_leaf_expr(const Step *step);
    bool scalar_bool_const(const Step *step, uint64_t value) const;
    std::optional<std::string> scalar_select_bool_expr(const SelectData &data);
    std::string scalar_arith_binary_expr(ScalarDataType sdtype, const ArithBinData &data, const char *left,
                                         const char *right) const;
    std::string scalar_compare_expr(const CmpData &data, const char *left, const char *right) const;

    bool common_suppress_compacted_acc_store_args(const Step *step, ArenaBitmap &suppressed) const;
    std::optional<std::string> common_compact_acc_store_to_cpp(const Step *step);
    std::optional<std::string> compact_acc_store_to_cpp(const Step *step);

    bool can_inline_agg_result_arg(const Step *step) const;
    bool can_inline_store_value_arg(const Step *step) const;
    bool can_inline_store_arg(const Step *step) const;
    std::string scalar_output_arg_expr(const Step *step, bool allow_select);
    bool suppress_inlined_agg_result_args(const Step *step, ArenaBitmap &suppressed) const;
    bool suppress_inlined_store_args(const Step *step, ArenaBitmap &suppressed) const;
    std::string agg_result_arg_expr(const Step *step);
    std::string store_arg_expr(const Step *step);

    std::string scalar_step_to_cpp(const Step *step);
    std::string step_to_cpp(const Step *step, const MaskPushdownInfo *mask_pushdown = nullptr);
    std::string compile_steps(nonstd::span<Step *const> step_roots, size_t indent_level, EmitPhase phase,
                              PrologueRootFilter prologue_filter = PrologueRootFilter::All);
};

std::unique_ptr<CppEmitterBase> make_x86_cpp_emitter(const Function *func);
std::unique_ptr<CppEmitterBase> make_arm_neon_cpp_emitter(const Function *func);

} // namespace cpp_backend
} // namespace simjit
