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

const char *scalar_dtype_to_cpp(ScalarDataType dtype) {
    switch (dtype) {
    case ScalarDataType::I1:
    case ScalarDataType::I8: return "int8_t";
    case ScalarDataType::I16: return "int16_t";
    case ScalarDataType::I32: return "int32_t";
    case ScalarDataType::I64: return "int64_t";
    case ScalarDataType::I128: return "__int128";
    case ScalarDataType::F32: return "float";
    case ScalarDataType::F64: return "double";
    }
    SIMJIT_UNREACHABLE();
}

const char *scalar_dtype_to_cpp_unsigned(ScalarDataType dtype) {
    switch (dtype) {
    case ScalarDataType::I1:
    case ScalarDataType::I8: return "uint8_t";
    case ScalarDataType::I16: return "uint16_t";
    case ScalarDataType::I32: return "uint32_t";
    case ScalarDataType::I64: return "uint64_t";
    case ScalarDataType::I128: return "unsigned __int128";
    case ScalarDataType::F32:
    case ScalarDataType::F64: messed_up("unexpected float type %s in unsigned context", show_scalar_dtype(dtype));
    }
    SIMJIT_UNREACHABLE();
}

const char *arith_binary_op_to_cpp(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::Add: return "+";
    case ArithBinaryOp::Sub: return "-";
    case ArithBinaryOp::Div: return "/";
    case ArithBinaryOp::Mod: return "%";
    case ArithBinaryOp::Mul:
    case ArithBinaryOp::Mul64SE:
    case ArithBinaryOp::Mul64ZE: return "*";
    case ArithBinaryOp::And: return "&";
    case ArithBinaryOp::Or: return "|";
    case ArithBinaryOp::Xor: return "^";
    case ArithBinaryOp::ShiftRightArith: return ">>";
    case ArithBinaryOp::ShiftLeftLogical: return "<<";
    default: messed_up("binary op %s requires special handling", show_arith_binary_op(op));
    }
    SIMJIT_UNREACHABLE();
}

bool is_shift_or_rotate_op(ArithBinaryOp op) {
    switch (op) {
    case ArithBinaryOp::ShiftRightArith:
    case ArithBinaryOp::ShiftRightLogical:
    case ArithBinaryOp::ShiftLeftLogical:
    case ArithBinaryOp::RotateLeft:
    case ArithBinaryOp::RotateRight: return true;
    default: return false;
    }
}

std::string format_scalar_arith_binary_expr(ScalarDataType result_dtype, ArithBinaryOp op, ScalarDataType left_dtype,
                                            ScalarDataType right_dtype, const char *left, const char *right,
                                            const std::optional<std::string> &shift_rhs) {
    auto checked_shift_rhs = [&]() -> const char * {
        if (!shift_rhs.has_value()) {
            unsupported("Do not support scalar %s of %s without shift amount", show_arith_binary_op(op),
                        show_scalar_dtype(result_dtype));
            SIMJIT_UNREACHABLE();
        }
        return shift_rhs->c_str();
    };

    if (result_dtype == ScalarDataType::F32 &&
        (op == ArithBinaryOp::And || op == ArithBinaryOp::Or || op == ArithBinaryOp::Xor)) {
        return format("bit_cast<float>(bit_cast<uint32_t>(%s) %s bit_cast<uint32_t>(%s))", left,
                      arith_binary_op_to_cpp(op), right);
    }
    if (result_dtype == ScalarDataType::F32 && op == ArithBinaryOp::AndNot) {
        return format("bit_cast<float>(~bit_cast<uint32_t>(%s) & bit_cast<uint32_t>(%s))", left, right);
    }
    if (result_dtype == ScalarDataType::F64 &&
        (op == ArithBinaryOp::And || op == ArithBinaryOp::Or || op == ArithBinaryOp::Xor)) {
        return format("bit_cast<double>(bit_cast<uint64_t>(%s) %s bit_cast<uint64_t>(%s))", left,
                      arith_binary_op_to_cpp(op), right);
    }
    if (result_dtype == ScalarDataType::F64 && op == ArithBinaryOp::AndNot) {
        return format("bit_cast<double>(~bit_cast<uint64_t>(%s) & bit_cast<uint64_t>(%s))", left, right);
    }

    switch (op) {
    case ArithBinaryOp::Add:
    case ArithBinaryOp::Sub:
    case ArithBinaryOp::Mul:
    case ArithBinaryOp::Mul64SE:
    case ArithBinaryOp::Mul64ZE:
    case ArithBinaryOp::Div:
    case ArithBinaryOp::Mod:
    case ArithBinaryOp::And:
    case ArithBinaryOp::Or:
    case ArithBinaryOp::Xor: return format("%s %s %s", left, arith_binary_op_to_cpp(op), right);
    case ArithBinaryOp::UDiv:
        return format("(%s)%s / (%s)%s", scalar_dtype_to_cpp_unsigned(left_dtype), left,
                      scalar_dtype_to_cpp_unsigned(left_dtype), right);
    case ArithBinaryOp::UMod:
        return format("(%s)%s %% (%s)%s", scalar_dtype_to_cpp_unsigned(left_dtype), left,
                      scalar_dtype_to_cpp_unsigned(left_dtype), right);
    case ArithBinaryOp::Min:
        if (result_dtype == ScalarDataType::F32) { return format("fminf(%s, %s)", left, right); }
        if (result_dtype == ScalarDataType::F64) { return format("fmin(%s, %s)", left, right); }
        return format("%s < %s ? %s : %s", right, left, right, left);
    case ArithBinaryOp::Max:
        if (result_dtype == ScalarDataType::F32) { return format("fmaxf(%s, %s)", left, right); }
        if (result_dtype == ScalarDataType::F64) { return format("fmax(%s, %s)", left, right); }
        return format("%s > %s ? %s : %s", right, left, right, left);
    case ArithBinaryOp::UMin:
        return format("(%s)%s < (%s)%s ? %s : %s", scalar_dtype_to_cpp_unsigned(left_dtype), left,
                      scalar_dtype_to_cpp_unsigned(right_dtype), right, left, right);
    case ArithBinaryOp::UMax:
        return format("(%s)%s > (%s)%s ? %s : %s", scalar_dtype_to_cpp_unsigned(left_dtype), left,
                      scalar_dtype_to_cpp_unsigned(right_dtype), right, left, right);
    case ArithBinaryOp::AndNot: return format("~%s & %s", left, right);
    case ArithBinaryOp::ShiftLeftLogical:
        return format("(%s)%s << %s", scalar_dtype_to_cpp_unsigned(left_dtype), left, checked_shift_rhs());
    case ArithBinaryOp::ShiftRightArith:
        return format("%s %s %s", left, arith_binary_op_to_cpp(op), checked_shift_rhs());
    case ArithBinaryOp::ShiftRightLogical:
        return format("(%s)%s >> %s", scalar_dtype_to_cpp_unsigned(left_dtype), left, checked_shift_rhs());
    case ArithBinaryOp::RotateLeft: {
        const char *amount = checked_shift_rhs();
        return format("%s == 0 ? %s : (%s)(((%s)%s << %s) | ((%s)%s >> (%zu - %s)))", right, left,
                      scalar_dtype_to_cpp(result_dtype), scalar_dtype_to_cpp_unsigned(left_dtype), left, amount,
                      scalar_dtype_to_cpp_unsigned(left_dtype), left, scalar_dtype_bits(result_dtype), amount);
    }
    case ArithBinaryOp::RotateRight: {
        const char *amount = checked_shift_rhs();
        return format("%s == 0 ? %s : (%s)(((%s)%s >> %s) | ((%s)%s << (%zu - %s)))", right, left,
                      scalar_dtype_to_cpp(result_dtype), scalar_dtype_to_cpp_unsigned(left_dtype), left, amount,
                      scalar_dtype_to_cpp_unsigned(left_dtype), left, scalar_dtype_bits(result_dtype), amount);
    }
    }
    SIMJIT_UNREACHABLE();
}

const char *cmp_op_to_cpp(CmpOp op) {
    switch (op) {
    case CmpOp::Less: return "<";
    case CmpOp::Greater: return ">";
    case CmpOp::LessEqual: return "<=";
    case CmpOp::GreaterEqual: return ">=";
    case CmpOp::Equal: return "==";
    case CmpOp::NotEqual: return "!=";
    }
    SIMJIT_UNREACHABLE();
}

template <typename T> static std::string format_float_literal_internal(T value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(value));

    auto special = [bits]() {
        // Use integer representation + memcpy for special values
        if constexpr (std::is_same_v<T, float>) {
            return format("bit_cast<float>(0x%08xu)", uint32_t(bits));
        } else {
            return format("bit_cast<double>(0x%016llxllu)", (unsigned long long)bits);
        }
    };

    // Check positive/negative zero
    if (bits == 0) {
        if constexpr (std::is_same_v<T, float>) {
            return "0.0f";
        } else {
            return "0.0";
        }
    }
    if constexpr (std::is_same_v<T, float>) {
        if (bits == 0x80000000) { return "-0.0f"; }
    } else {
        if (bits == 0x8000000000000000) { return "-0.0"; }
    }

    // Recognize common sentinels by exact IEEE-754 bit pattern. Do not use
    // floating-point comparisons here: these values are often aggregate seeds.
    if constexpr (std::is_same_v<T, float>) {
        switch (uint32_t(bits)) {
        case 0x7f800000u: return "INFINITY";
        case 0xff800000u: return "-INFINITY";
        case 0x7f7fffffu: return "FLT_MAX";
        case 0xff7fffffu: return "-FLT_MAX";
        case 0x00800000u: return "FLT_MIN";
        case 0x80800000u: return "-FLT_MIN";
        default: break;
        }
    } else {
        switch (bits) {
        case 0x7ff0000000000000ull: return "INFINITY";
        case 0xfff0000000000000ull: return "-INFINITY";
        case 0x7fefffffffffffffull: return "DBL_MAX";
        case 0xffefffffffffffffull: return "-DBL_MAX";
        case 0x0010000000000000ull: return "DBL_MIN";
        case 0x8010000000000000ull: return "-DBL_MIN";
        default: break;
        }
    }

    // Check if the value is special or cannot be exactly represented
    if (std::isnan(value) || std::isinf(value) || std::fpclassify(value) == FP_SUBNORMAL) { return special(); }

    // Try to format as literal - check if it round-trips exactly
    T round_trip;
    std::string literal = format("%.17g", double(value));

    // Simple check: if formatting with sufficient precision preserves the value
    // For more robust checking, you might want to parse the literal back
    if constexpr (std::is_same_v<T, float>) {
        round_trip = std::stof(literal);
    } else {
        round_trip = std::stod(literal);
    }

    using Bits = std::conditional_t<sizeof(T) == sizeof(uint32_t), uint32_t, uint64_t>;
    auto value_bits = [](T x) {
        Bits b = 0;
        std::memcpy(&b, &x, sizeof(x));
        return b;
    };

    if (value_bits(value) == value_bits(round_trip)) {
        // Can be represented exactly as literal
        if (literal.find('.') == std::string::npos && literal.find('e') == std::string::npos) {
            literal += ".0"; // Ensure it's treated as float
        }
        if constexpr (std::is_same_v<T, float>) { literal += "f"; }
        return literal;
    }

    // Fall back to integer representation
    return special();
}

static std::string format_float_literal(float value) {
    return format_float_literal_internal<float>(value);
}

static std::string format_float_literal(double value) {
    return format_float_literal_internal<double>(value);
}

const char *indent_str(size_t indent) {
    if (indent == 0) return "";
    if (indent == 1) return "    ";
    if (indent == 2) return "        ";
    if (indent == 3) return "            ";
    messed_up("Too deep indent");
}

std::string const_data_to_cpp(ConstData data, ScalarDataType dt) {
    switch (dt) {
    case ScalarDataType::I8: {
        int8_t x = int8_t(data.as_signed());
        if (x == INT8_MIN) { return "INT8_MIN"; }
        if (x == INT8_MAX) { return "INT8_MAX"; }
        return format("%d", int(x));
    }
    case ScalarDataType::I16: {
        int16_t x = int16_t(data.as_signed());
        if (x == INT16_MIN) { return "INT16_MIN"; }
        if (x == INT16_MAX) { return "INT16_MAX"; }
        return format("%d", int(x));
    }
    case ScalarDataType::I32: {
        int32_t x = int32_t(data.as_signed());
        if (x == INT32_MIN) { return "INT32_MIN"; }
        if (x == INT32_MAX) { return "INT32_MAX"; }
        return format("%d", int(x));
    }
    case ScalarDataType::I64: {
        long long x = (long long)data.as_signed();
        if (x == INT64_MIN) { return "INT64_MIN"; }
        if (x == INT64_MAX) { return "INT64_MAX"; }
        if (x < (-2147483647LL - 1) || x > 2147483647LL) { return format("%lldLL", x); }
        return format("%lld", x);
    }
    case ScalarDataType::F32: return format_float_literal(data.as_f32());
    case ScalarDataType::F64: return format_float_literal(data.as_f64());
    case ScalarDataType::I1: return data.as_unsigned() ? "1" : "0";
    case ScalarDataType::I128: messed_up("unexpected i128 constant");
    }
    SIMJIT_UNREACHABLE();
}

CppEmitterBase::CppEmitterBase(const Function *f) : func(f) {
}

CppEmitterBase::~CppEmitterBase() noexcept = default;

bool CppEmitterBase::enable_vector_peepholes() const {
    return true;
}

const char *CppEmitterBase::cpp_dtype(DataType dtype) const {
    if (!dtype.is_scalar()) { messed_up("backend must provide C++ type for %s", show_dtype(dtype)); }
    return scalar_dtype_to_cpp(dtype.as_scalar());
}

void CppEmitterBase::prepare_emit() {
}

std::string CppEmitterBase::source_prelude_to_cpp() const {
    return {};
}

std::string CppEmitterBase::function_prelude_to_cpp(size_t) const {
    return {};
}

std::string CppEmitterBase::before_loops_to_cpp(size_t) {
    return {};
}

std::string CppEmitterBase::after_loops_to_cpp(size_t) {
    return {};
}

bool CppEmitterBase::has_custom_step_to_cpp(const Step *, EmitPhase) const {
    return false;
}

std::string CppEmitterBase::custom_step_to_cpp(const Step *, EmitPhase) {
    SIMJIT_UNREACHABLE();
}

bool CppEmitterBase::suppress_inlined_acc_store_args(const Step *, ArenaBitmap &) const {
    return false;
}

bool CppEmitterBase::suppress_compacted_acc_store_args(const Step *step, ArenaBitmap &suppressed) const {
    return common_suppress_compacted_acc_store_args(step, suppressed);
}

std::optional<std::string> CppEmitterBase::backend_compact_acc_store_to_cpp(const Step *step) {
    return common_compact_acc_store_to_cpp(step);
}

void CppEmitterBase::record_backend_peephole_uses(Step *) {
}

void CppEmitterBase::init_use_counts() {
    use_counts.assign(func->step_id_count, 0);
    peephole_const_use_counts.assign(func->step_id_count, 0);
    peephole_step_use_counts.assign(func->step_id_count, 0);
    peephole_named_use_counts.assign(func->step_id_count, 0);
    auto visit_roots = [&](nonstd::span<Step *const> roots) {
        traverse_steps_postorder_unique(func->step_id_count, roots, [&](Step *step) {
            step_recurse(step, [&](Step *child) { ++use_counts[child->id]; });
            record_backend_peephole_uses(step);
        });
    };
    visit_roots(func->prologue_roots);
    visit_roots(func->main_loop_roots);
    visit_roots(func->remainder_roots);
    visit_roots(func->epilogue_roots);
}

const char *CppEmitterBase::show(const Step *step) const {
    if (seen[step->id].empty()) {
        messed_up("step %s (id=%zu) does not have an assigned C++ name", show_step_kind(step->kind), size_t(step->id));
    }
    return seen[step->id].c_str();
}
const char *CppEmitterBase::show_scalar_operand(const Step *step) {
    auto &expr = (inline_scalar_exprs[step->id]);
    if (!expr.has_value() && can_inline_scalar_expr(step)) { expr = inline_scalar_leaf_expr(step); }
    if (expr.has_value()) { return expr->c_str(); }
    return show(step);
}
const char *CppEmitterBase::temp_prefix(EmitPhase phase) {
    switch (phase) {
    case EmitPhase::Prologue: return "a";
    case EmitPhase::MainLoop: return "x";
    case EmitPhase::Remainder: return "y";
    case EmitPhase::Epilogue: return "z";
    }
    SIMJIT_UNREACHABLE();
}
const char *CppEmitterBase::load_prefix(EmitPhase phase) {
    switch (phase) {
    case EmitPhase::Prologue: return "am";
    case EmitPhase::MainLoop: return "xm";
    case EmitPhase::Remainder: return "ym";
    case EmitPhase::Epilogue: return "zm";
    }
    SIMJIT_UNREACHABLE();
}
CppEmitterBase::StepNameKind CppEmitterBase::classify_step_name(const Step *step) {
    switch (step->kind) {
    case StepKind::Const:
    case StepKind::VecConst: return StepNameKind::Const;
    case StepKind::Load:
    case StepKind::LoadSplat:
    case StepKind::Gather: return StepNameKind::Load;
    case StepKind::AccLoad: return StepNameKind::Acc;
    case StepKind::Store:
    case StepKind::CondStore:
    case StepKind::Scatter:
    case StepKind::CondScatter:
    case StepKind::Pack:
    case StepKind::AccStore:
    case StepKind::AggResult:
    case StepKind::StoreSum128: return StepNameKind::None;
    default: return StepNameKind::Temp;
    }
}
std::optional<std::string> CppEmitterBase::assign_step_name(const Step *step, EmitPhase phase) {
    size_t phase_idx = size_t(phase);
    switch (classify_step_name(step)) {
    case StepNameKind::None: return std::nullopt;
    case StepNameKind::Temp: return format("%s%zu", temp_prefix(phase), temp_counts[phase_idx]++);
    case StepNameKind::Const: return format("c%zu", const_count++);
    case StepNameKind::Load: return format("%s%zu", load_prefix(phase), load_counts[phase_idx]++);
    case StepNameKind::Acc: return format("acc%zu", func->accs.index(step->step_data<StepKind::AccLoad>()));
    }
    SIMJIT_UNREACHABLE();
}
const char *CppEmitterBase::cpp_var_decl_qualifier(const Step *step) const {
    if (step->is(StepKind::Const) || step->is(StepKind::VecConst) || step->is(StepKind::LoadSplat)) { return "const "; }
    return "";
}
std::string CppEmitterBase::cpp_var_decl(const Step *step) const {
    return format("%s%s %s = ", cpp_var_decl_qualifier(step), cpp_dtype(step->dtype), show(step));
}
void CppEmitterBase::init_accumulators() {
    size_t count = func->accs.count;
    acc_dtypes.resize(count);
    acc_declared = ArenaBitmap::create(func->ctx->arena, count);
    main_loop_acc_uses = ArenaBitmap::create(func->ctx->arena, count);
    for (const Step *step : func->prologue_roots) {
        if (!step->is(StepKind::AccStore)) continue;
        const auto &store = step->step_data<StepKind::AccStore>();
        acc_dtypes[func->accs.index(store.acc)] = step->dtype;
    }
    record_main_loop_acc_uses();
}
void CppEmitterBase::mark_main_loop_acc_use(AccId acc) {
    size_t idx = func->accs.index(acc);
    if (idx >= main_loop_acc_uses.size()) { messed_up("accumulator %zu is out of range", idx); }
    main_loop_acc_uses.set(idx);
}
void CppEmitterBase::record_main_loop_acc_uses() {
    traverse_steps_postorder_unique(func->step_id_count, func->main_loop_roots, [&](Step *step) {
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::AccLoad) {
                mark_main_loop_acc_use(data);
                break;
            }
            SIMJIT_MATCH (StepKind::AccStore) {
                mark_main_loop_acc_use(data.acc);
                break;
            }
            SIMJIT_MATCH (StepKind::VecIndex) {
                mark_main_loop_acc_use(data.acc);
                break;
            }
            SIMJIT_MATCH (StepKind::Pack) {
                mark_main_loop_acc_use(data.acc);
                break;
            }
            SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
                mark_main_loop_acc_use(data.overflow_flag);
                break;
            }
        default: break;
        }
    });
}
bool CppEmitterBase::prologue_root_is_delayed(const Step *root) const {
    if (!root->is(StepKind::AccStore)) { return false; }
    if (!root->dtype.is_scalar()) { return false; }
    AccId acc = root->step_data<StepKind::AccStore>().acc;
    size_t idx = func->accs.index(acc);
    if (idx >= main_loop_acc_uses.size()) { messed_up("accumulator %zu is out of range", idx); }
    return !main_loop_acc_uses.get(idx);
}
bool CppEmitterBase::has_delayed_prologue_steps(nonstd::span<Step *const> steps) const {
    for (Step *root : steps) {
        if (prologue_root_is_delayed(root)) { return true; }
    }
    return false;
}
void CppEmitterBase::init_inline_scalar_exprs() {
    inline_scalar_exprs.assign(func->step_id_count, std::nullopt);
}
std::string CppEmitterBase::show_inline_operand(const Step *step, const Step *acc_load, AccId acc) {
    if (step == acc_load) { return format("acc%zu", func->accs.index(acc)); }
    if (can_inline_scalar_expr(step)) { return show_scalar_operand(step); }
    return show(step);
}
const Step *CppEmitterBase::find_compacted_acc_load(AccId acc, std::initializer_list<const Step *> candidates,
                                                    std::initializer_list<const Step *> counted_uses) const {
    for (const Step *candidate : candidates) {
        if (!candidate->is(StepKind::AccLoad)) { continue; }
        if (candidate->step_data<StepKind::AccLoad>() != acc) { continue; }

        size_t expected_uses = 0;
        for (const Step *use : counted_uses) {
            expected_uses += size_t(use == candidate);
        }
        if (use_counts[candidate->id] == expected_uses) { return candidate; }
    }
    return nullptr;
}
const Step *CppEmitterBase::find_compacted_acc_load(AccId acc, std::initializer_list<const Step *> operands) const {
    return find_compacted_acc_load(acc, operands, operands);
}
std::string CppEmitterBase::format_acc_store(AccId acc, std::string_view rhs) {
    size_t idx = func->accs.index(acc);
    std::string lhs = format("acc%zu", idx);
    if (idx >= acc_declared.size()) { messed_up("accumulator %zu is out of range", idx); }
    if (!acc_declared.get(idx)) {
        acc_declared.set(idx);
        return format("%s %s = %.*s", cpp_dtype(acc_dtypes[idx]), lhs.c_str(), (int)rhs.length(), rhs.data());
    }
    return format("%s = %.*s", lhs.c_str(), (int)rhs.length(), rhs.data());
}

bool CppEmitterBase::common_suppress_compacted_acc_store_args(const Step *step, ArenaBitmap &suppressed) const {
    if (!step->is(StepKind::AccStore)) { return false; }

    auto &store = step->step_data<StepKind::AccStore>();
    const Step *arg = store.arg;
    if (use_counts[arg->id] != 1) { return false; }

    if (arg->is(StepKind::ArithBinary) && arg->dtype.is_scalar()) {
        auto &bin = arg->step_data<StepKind::ArithBinary>();
        const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
        if (acc_load == nullptr) { return false; }
        suppressed.set(arg->id);
        suppressed.set(acc_load->id);
        return true;
    }

    if (arg->is(StepKind::Select) && arg->dtype.is_scalar()) {
        auto &select = arg->step_data<StepKind::Select>();
        if (!select.falsy->is(StepKind::AccLoad) || select.falsy->step_data<StepKind::AccLoad>() != store.acc) {
            return false;
        }

        if (select.truthy->is(StepKind::ArithBinary) && use_counts[select.truthy->id] == 1) {
            auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
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

        if (use_counts[select.falsy->id] != 1) { return false; }
        suppressed.set(arg->id);
        suppressed.set(select.falsy->id);
        return true;
    }

    return false;
}

std::optional<std::string> CppEmitterBase::common_compact_acc_store_to_cpp(const Step *step) {
    if (!step->is(StepKind::AccStore)) { return std::nullopt; }

    auto &store = step->step_data<StepKind::AccStore>();
    const Step *arg = store.arg;
    if (use_counts[arg->id] != 1) { return std::nullopt; }

    if (arg->is(StepKind::ArithBinary) && arg->dtype.is_scalar()) {
        auto &bin = arg->step_data<StepKind::ArithBinary>();
        const Step *acc_load = find_compacted_acc_load(store.acc, {bin.left, bin.right});
        if (acc_load == nullptr) { return std::nullopt; }
        std::string left = show_inline_operand(bin.left, acc_load, store.acc);
        std::string right = show_inline_operand(bin.right, acc_load, store.acc);
        return scalar_arith_binary_expr(arg->dtype.as_scalar(), bin, left.c_str(), right.c_str());
    }

    if (arg->is(StepKind::Select) && arg->dtype.is_scalar()) {
        auto &select = arg->step_data<StepKind::Select>();
        if (!select.falsy->is(StepKind::AccLoad) || select.falsy->step_data<StepKind::AccLoad>() != store.acc) {
            return std::nullopt;
        }

        std::string cond = show_scalar_operand(select.cond);
        if (select.truthy->is(StepKind::ArithBinary) && use_counts[select.truthy->id] == 1) {
            auto &bin = select.truthy->step_data<StepKind::ArithBinary>();
            const Step *acc_load =
                find_compacted_acc_load(store.acc, {bin.left, bin.right}, {bin.left, bin.right, select.falsy});
            if (acc_load == nullptr) { return std::nullopt; }
            if (select.falsy != acc_load && use_counts[select.falsy->id] != 1) { return std::nullopt; }

            std::string left = show_inline_operand(bin.left, acc_load, store.acc);
            std::string right = show_inline_operand(bin.right, acc_load, store.acc);
            std::string truthy = scalar_arith_binary_expr(arg->dtype.as_scalar(), bin, left.c_str(), right.c_str());
            std::string falsy = show_inline_operand(select.falsy, acc_load, store.acc);
            return format("%s ? %s : %s", cond.c_str(), truthy.c_str(), falsy.c_str());
        }

        if (use_counts[select.falsy->id] != 1) { return std::nullopt; }
        return format("%s ? %s : acc%zu", cond.c_str(), show_scalar_operand(select.truthy),
                      func->accs.index(store.acc));
    }

    return std::nullopt;
}

std::optional<std::string> CppEmitterBase::compact_acc_store_to_cpp(const Step *step) {
    return backend_compact_acc_store_to_cpp(step);
}

bool CppEmitterBase::can_inline_scalar_expr(const Step *step) const {
    if (!step->dtype.is_scalar() || !is_scalar_step(step->kind)) { return false; }

    switch (step->kind) {
        SIMJIT_MATCH (StepKind::Const) { return true; }
        SIMJIT_MATCH (StepKind::AccLoad) { return true; }
        SIMJIT_MATCH (StepKind::Load) {
            if (use_counts[step->id] != 1) { return false; }
            return step->dtype != ScalarDataType::I1;
        }
        SIMJIT_MATCH (StepKind::LoadSplat) {
            if (use_counts[step->id] != 1) { return false; }
            return step->dtype != ScalarDataType::I1;
        }
        SIMJIT_MATCH (StepKind::Compare) {
            return use_counts[step->id] == 1 && data.left->dtype.is_scalar() && data.right->dtype.is_scalar();
        }
    default: return false;
    }
}
std::optional<std::string> CppEmitterBase::inline_scalar_leaf_expr(const Step *step) {
    if (!can_inline_scalar_expr(step)) { return std::nullopt; }

    switch (step->kind) {
        SIMJIT_MATCH (StepKind::Const) { return const_data_to_cpp(data, step->dtype.as_scalar()); }
        SIMJIT_MATCH (StepKind::AccLoad) { return format("acc%zu", func->accs.index(data)); }
        SIMJIT_MATCH (StepKind::Load) {
            SIMJIT_ASSERT(step->dtype != ScalarDataType::I1);
            return format("arg%zu[i]", data.addr.arg);
        }
        SIMJIT_MATCH (StepKind::LoadSplat) { return format("*arg%zu", data.addr.arg); }
        SIMJIT_MATCH (StepKind::Compare) {
            return scalar_compare_expr(data, show_scalar_operand(data.left), show_scalar_operand(data.right));
        }
    default: return std::nullopt;
    }
}
bool CppEmitterBase::scalar_bool_const(const Step *step, uint64_t value) const {
    if (!step->is(StepKind::Const) || !step->dtype.is_scalar()) { return false; }
    ScalarDataType dtype = step->dtype.as_scalar();
    if (is_float_dtype(dtype) || dtype == ScalarDataType::I128) { return false; }
    return step->step_data<StepKind::Const>().as_unsigned() == value;
}
std::optional<std::string> CppEmitterBase::scalar_select_bool_expr(const SelectData &data) {
    if (scalar_bool_const(data.truthy, 1) && scalar_bool_const(data.falsy, 0)) {
        return show_scalar_operand(data.cond);
    }
    if (scalar_bool_const(data.truthy, 0) && scalar_bool_const(data.falsy, 1)) {
        return format("!%s", show_scalar_operand(data.cond));
    }
    return std::nullopt;
}
std::string CppEmitterBase::scalar_arith_binary_expr(ScalarDataType sdtype, const ArithBinData &data, const char *left,
                                                     const char *right) const {
    std::optional<std::string> shift_rhs;
    if (is_shift_or_rotate_op(data.op)) {
        if (data.right->is(StepKind::Const)) {
            uint64_t value = data.right->step_data<StepKind::Const>().as_unsigned();
            if (value < scalar_dtype_bits(sdtype)) {
                shift_rhs = format("%llu", (unsigned long long)(value));
            } else {
                shift_rhs = format("(%s & %zu)", right, scalar_dtype_bits(sdtype) - 1);
            }
        } else {
            shift_rhs = format("(%s & %zu)", right, scalar_dtype_bits(sdtype) - 1);
        }
    }

    return format_scalar_arith_binary_expr(sdtype, data.op, data.left->dtype.as_scalar(), data.right->dtype.as_scalar(),
                                           left, right, shift_rhs);
}
std::string CppEmitterBase::scalar_compare_expr(const CmpData &data, const char *left, const char *right) const {
    if (data.is_unsigned) {
        ScalarDataType s = data.left->dtype.as_scalar();
        return format("((%s)%s %s (%s)%s)", scalar_dtype_to_cpp_unsigned(s), left, cmp_op_to_cpp(data.op),
                      scalar_dtype_to_cpp_unsigned(s), right);
    }

    return format("(%s %s %s)", left, cmp_op_to_cpp(data.op), right);
}
bool CppEmitterBase::can_inline_agg_result_arg(const Step *step) const {
    return step->dtype.is_scalar() && step->is(StepKind::ArithBinary) && use_counts[step->id] == 1;
}
bool CppEmitterBase::can_inline_store_value_arg(const Step *step) const {
    return can_inline_agg_result_arg(step) ||
           (step->dtype.is_scalar() && step->is(StepKind::Select) && use_counts[step->id] == 1);
}
bool CppEmitterBase::can_inline_store_arg(const Step *step) const {
    if (!step->is(StepKind::Store) || !step->dtype.is_scalar() || step->dtype == ScalarDataType::I1) { return false; }
    const auto &data = step->step_data<StepKind::Store>();
    const ArgumentDecl &arg = func->args[data.addr.arg];
    return arg.dtype == step->dtype.as_scalar() && can_inline_store_value_arg(data.what);
}
std::string CppEmitterBase::scalar_output_arg_expr(const Step *step, bool allow_select) {
    if (can_inline_agg_result_arg(step)) {
        auto &data = step->step_data<StepKind::ArithBinary>();
        return scalar_arith_binary_expr(step->dtype.as_scalar(), data, show_scalar_operand(data.left),
                                        show_scalar_operand(data.right));
    }
    if (allow_select && can_inline_store_value_arg(step) && step->is(StepKind::Select)) {
        auto &data = step->step_data<StepKind::Select>();
        if (auto expr = scalar_select_bool_expr(data)) { return *expr; }
        return format("%s ? %s : %s", show_scalar_operand(data.cond), show_scalar_operand(data.truthy),
                      show_scalar_operand(data.falsy));
    }
    return show_scalar_operand(step);
}
bool CppEmitterBase::suppress_inlined_agg_result_args(const Step *step, ArenaBitmap &suppressed) const {
    if (!step->is(StepKind::AggResult)) { return false; }
    auto &data = step->step_data<StepKind::AggResult>();
    if (!can_inline_agg_result_arg(data.arg)) { return false; }
    suppressed.set(data.arg->id);
    return true;
}
bool CppEmitterBase::suppress_inlined_store_args(const Step *step, ArenaBitmap &suppressed) const {
    if (!can_inline_store_arg(step)) { return false; }
    const auto &data = step->step_data<StepKind::Store>();
    suppressed.set(data.what->id);
    return true;
}
std::string CppEmitterBase::agg_result_arg_expr(const Step *step) {
    return scalar_output_arg_expr(step, false);
}
std::string CppEmitterBase::store_arg_expr(const Step *step) {
    if (!can_inline_store_arg(step)) { return show_scalar_operand(step->step_data<StepKind::Store>().what); }
    return scalar_output_arg_expr(step->step_data<StepKind::Store>().what, true);
}
std::string CppEmitterBase::scalar_step_to_cpp(const Step *step) {
    ScalarDataType sdtype = step->dtype.as_scalar();
    switch (step->kind) {
        SIMJIT_MATCH (StepKind::ScalarIndex) { return cpp_var_decl(step) + "i"; }
        SIMJIT_MATCH (StepKind::Const) { return cpp_var_decl(step) + const_data_to_cpp(data, step->dtype.as_scalar()); }
        SIMJIT_MATCH (StepKind::Load) {
            const ArgumentDecl &arg = func->args[data.addr.arg];
            if (step->dtype == ScalarDataType::I1) {
                return cpp_var_decl(step) +
                       format("(((uint64_t *)arg%zu)[i >> 6] & ((uint64_t)1 << (i & 63))) != 0", data.addr.arg);
            }
            if (arg.dtype == step->dtype.as_scalar()) {
                return cpp_var_decl(step) + format("arg%zu[i]", data.addr.arg);
            }
            size_t size = scalar_dtype_size(step->dtype.as_scalar());
            std::string offset_str =
                data.addr.offset != 0 ? format(" + %zu * %zu", data.addr.offset / (size * 8), size) : "";
            return format("%s %s; memcpy(&%s, (char *)arg%zu + i * %zu%s, %zu)", cpp_dtype(step->dtype),
                          show_scalar_operand(step), show_scalar_operand(step), data.addr.arg, size, offset_str.c_str(),
                          size);
        }
        SIMJIT_MATCH (StepKind::LoadSplat) {
            if (sdtype == ScalarDataType::I1) { return cpp_var_decl(step) + format("*arg%zu & 1", data.addr.arg); }
            return cpp_var_decl(step) + format("*arg%zu", data.addr.arg);
        }
        SIMJIT_MATCH (StepKind::Gather) {
            return cpp_var_decl(step) + format("arg%zu[%s]", data.data, show_scalar_operand(data.idx));
        }
        SIMJIT_MATCH (StepKind::Store) {
            const ArgumentDecl &arg = func->args[data.addr.arg];
            if (step->dtype == ScalarDataType::I1) {
                return format("((uint64_t *)arg%zu)[i >> 6] &= ~((uint64_t)1 << (i & 63));\n"
                              "((uint64_t *)arg%zu)[i >> 6] |= (uint64_t)(%s) << (i & 63)",
                              data.addr.arg, //
                              data.addr.arg, show_scalar_operand(data.what));
            }
            if (arg.dtype == step->dtype.as_scalar()) {
                std::string what = store_arg_expr(step);
                return format("arg%zu[i] = %s", data.addr.arg, what.c_str());
            }
            size_t size = scalar_dtype_size(step->dtype.as_scalar());
            std::string offset_str = data.addr.offset != 0 ? format(" + %zu", data.addr.offset) : "";
            return format("memcpy((char *)arg%zu + i * %zu%s, &%s, %zu)", data.addr.arg, size, offset_str.c_str(),
                          show_scalar_operand(data.what), size);
        }
        SIMJIT_MATCH (StepKind::ArithBinary) {
            return cpp_var_decl(step) + scalar_arith_binary_expr(sdtype, data, show_scalar_operand(data.left),
                                                                 show_scalar_operand(data.right));
        }
        SIMJIT_MATCH (StepKind::FMA) {
            const char *function = sdtype == ScalarDataType::F32 ? "fmaf" : "fma";
            std::string x1 = show_scalar_operand(data.x1);
            std::string x2 = show_scalar_operand(data.x2);
            std::string x3 = show_scalar_operand(data.x3);
            switch (data.kind) {
            case FmaKind::FMA: break;
            case FmaKind::FMS: x3 = "-" + x3; break;
            case FmaKind::FNMA: x1 = "-" + x1; break;
            case FmaKind::FNMS:
                x1 = "-" + x1;
                x3 = "-" + x3;
                break;
            }
            return cpp_var_decl(step) + format("%s(%s, %s, %s)", function, x1.c_str(), x2.c_str(), x3.c_str());
        }
        SIMJIT_MATCH (StepKind::ConstDiv) {
            ArithBinData bin{data.numerator, data.divisor, data.op};
            return cpp_var_decl(step) + scalar_arith_binary_expr(sdtype, bin, show_scalar_operand(data.numerator),
                                                                 show_scalar_operand(data.divisor));
        }
        SIMJIT_MATCH (StepKind::ArithUnary) {
            bool is_f32 = step->dtype.as_scalar() == ScalarDataType::F32;
            switch (data.op) {
            case ArithUnaryOp::Not:
                if (sdtype == ScalarDataType::F32) {
                    return cpp_var_decl(step) +
                           format("bit_cast<float>(~bit_cast<uint32_t>(%s))", show_scalar_operand(data.arg));
                }
                if (sdtype == ScalarDataType::F64) {
                    return cpp_var_decl(step) +
                           format("bit_cast<double>(~bit_cast<uint64_t>(%s))", show_scalar_operand(data.arg));
                }
                return cpp_var_decl(step) + format("~%s", show_scalar_operand(data.arg));
            case ArithUnaryOp::Negate: return cpp_var_decl(step) + format("-%s", show_scalar_operand(data.arg));
            case ArithUnaryOp::Abs:
                return cpp_var_decl(step) + format("%s < 0 ? -%s : %s", show_scalar_operand(data.arg),
                                                   show_scalar_operand(data.arg), show_scalar_operand(data.arg));
            case ArithUnaryOp::Lzcnt: {
                switch (sdtype) {
                case ScalarDataType::I8:
                    return cpp_var_decl(step) + format("(uint8_t)%s == 0 ? 8 : __builtin_clz((uint8_t)%s) - 24",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::I16:
                    return cpp_var_decl(step) + format("(uint16_t)%s == 0 ? 16 : __builtin_clz((uint16_t)%s) - 16",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::I32:
                    return cpp_var_decl(step) + format("(uint32_t)%s == 0 ? 32 : __builtin_clz((uint32_t)%s)",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::I64:
                    return cpp_var_decl(step) + format("(uint64_t)%s == 0 ? 64 : __builtin_clzll((uint64_t)%s)",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::F32:
                case ScalarDataType::F64:
                case ScalarDataType::I1:
                case ScalarDataType::I128: messed_up("unsupported type %s for lzcnt", show_scalar_dtype(sdtype));
                }
                SIMJIT_UNREACHABLE();
            }
            case ArithUnaryOp::Tzcnt: {
                switch (sdtype) {
                case ScalarDataType::I8:
                    return cpp_var_decl(step) + format("(uint8_t)%s == 0 ? 8 : __builtin_ctz((uint8_t)%s)",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::I16:
                    return cpp_var_decl(step) + format("(uint16_t)%s == 0 ? 16 : __builtin_ctz((uint16_t)%s)",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::I32:
                    return cpp_var_decl(step) + format("(uint32_t)%s == 0 ? 32 : __builtin_ctz((uint32_t)%s)",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::I64:
                    return cpp_var_decl(step) + format("(uint64_t)%s == 0 ? 64 : __builtin_ctzll((uint64_t)%s)",
                                                       show_scalar_operand(data.arg), show_scalar_operand(data.arg));
                case ScalarDataType::F32:
                case ScalarDataType::F64:
                case ScalarDataType::I1:
                case ScalarDataType::I128: messed_up("unsupported type %s for tzcnt", show_scalar_dtype(sdtype));
                }
                SIMJIT_UNREACHABLE();
            }
            case ArithUnaryOp::Popcount: {
                if (sdtype == ScalarDataType::I8) {
                    return cpp_var_decl(step) +
                           format("__builtin_popcount((uint32_t)(uint8_t)%s)", show_scalar_operand(data.arg));
                }
                if (sdtype == ScalarDataType::I16) {
                    return cpp_var_decl(step) +
                           format("__builtin_popcount((uint32_t)(uint16_t)%s)", show_scalar_operand(data.arg));
                }
                if (sdtype == ScalarDataType::I32) {
                    return cpp_var_decl(step) + format("__builtin_popcount(%s)", show_scalar_operand(data.arg));
                }
                return cpp_var_decl(step) + format("__builtin_popcountll(%s)", show_scalar_operand(data.arg));
            }
            case ArithUnaryOp::RoundNearest:
                if (is_f32) return cpp_var_decl(step) + format("nearbyintf(%s)", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("nearbyint(%s)", show_scalar_operand(data.arg));
            case ArithUnaryOp::RoundDown:
                if (is_f32) return cpp_var_decl(step) + format("floorf(%s)", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("floor(%s)", show_scalar_operand(data.arg));
            case ArithUnaryOp::RoundUp:
                if (is_f32) return cpp_var_decl(step) + format("ceilf(%s)", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("ceil(%s)", show_scalar_operand(data.arg));
            case ArithUnaryOp::RoundTruncate:
                if (is_f32) return cpp_var_decl(step) + format("truncf(%s)", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("trunc(%s)", show_scalar_operand(data.arg));
            case ArithUnaryOp::Rcp:
                if (is_f32) return cpp_var_decl(step) + format("1.0f / %s", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("1.0 / (%s)", show_scalar_operand(data.arg));
            case ArithUnaryOp::Sqrt:
                if (is_f32) return cpp_var_decl(step) + format("sqrtf(%s)", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("sqrt(%s)", show_scalar_operand(data.arg));
            case ArithUnaryOp::Rsqrt:
                if (is_f32) return cpp_var_decl(step) + format("1.0f / sqrtf(%s)", show_scalar_operand(data.arg));
                return cpp_var_decl(step) + format("1.0 / sqrt(%s)", show_scalar_operand(data.arg));
            }
            SIMJIT_UNREACHABLE();
        }
        SIMJIT_MATCH (StepKind::IntCast) {
            switch (data.kind) {
            case IntCastKind::Trunc:
            case IntCastKind::Sext:
                return cpp_var_decl(step) + format("(%s)%s", cpp_dtype(step->dtype), show_scalar_operand(data.arg));
            case IntCastKind::Zext:
                // Code like
                // int8_t m1 = arg1[i];
                // uint16_t x1 = (uint16_t)m1;
                // still produces sign-extend, but if we first cast m1 to uint8_t zext is emitted
                return cpp_var_decl(step) + format("(%s)(%s)%s", scalar_dtype_to_cpp_unsigned(sdtype),
                                                   scalar_dtype_to_cpp_unsigned(data.arg->dtype.as_scalar()),
                                                   show_scalar_operand(data.arg));
            }
            SIMJIT_UNREACHABLE();
        }
        SIMJIT_MATCH (StepKind::FloatCast) {
            if (is_float_dtype(sdtype) && data.is_unsigned) {
                return cpp_var_decl(step) + format("(%s)(%s)%s", scalar_dtype_to_cpp(sdtype),
                                                   scalar_dtype_to_cpp_unsigned(data.arg->dtype.as_scalar()),
                                                   show_scalar_operand(data.arg));
            }
            const char *target_type = is_float_dtype(sdtype) ? scalar_dtype_to_cpp(sdtype)
                                      : data.is_unsigned     ? scalar_dtype_to_cpp_unsigned(sdtype)
                                                             : scalar_dtype_to_cpp(sdtype);
            return cpp_var_decl(step) + format("(%s)%s", target_type, show_scalar_operand(data.arg));
        }
        SIMJIT_MATCH (StepKind::BitCast) {
            return cpp_var_decl(step) + format("bit_cast<%s>(%s)", cpp_dtype(step->dtype), show_scalar_operand(data));
        }
        SIMJIT_MATCH (StepKind::Compare) {
            return cpp_var_decl(step) +
                   scalar_compare_expr(data, show_scalar_operand(data.left), show_scalar_operand(data.right));
        }
        SIMJIT_MATCH (StepKind::AggResult) {
            std::string arg = agg_result_arg_expr(data.arg);
            return format("*arg%zu = %s", data.dst, arg.c_str());
        }

        SIMJIT_MATCH (StepKind::StoreSum128) { return store_sum128_to_cpp(data); }
        SIMJIT_MATCH (StepKind::PredicateNot) { return cpp_var_decl(step) + format("!%s", show_scalar_operand(data)); }
        SIMJIT_MATCH (StepKind::Select) {
            if (auto expr = scalar_select_bool_expr(data)) { return cpp_var_decl(step) + *expr; }
            return cpp_var_decl(step) + format("%s ? %s : %s", show_scalar_operand(data.cond),
                                               show_scalar_operand(data.truthy), show_scalar_operand(data.falsy));
        }
        SIMJIT_MATCH (StepKind::AccLoad) return cpp_var_decl(step) + format("acc%zu", func->accs.index(data));
        SIMJIT_MATCH (StepKind::AccStore) { return format_acc_store(data.acc, show_scalar_operand(data.arg)); }
        SIMJIT_MATCH (StepKind::Scatter) {
            return format("arg%zu[%s] = %s", data.dst, show_scalar_operand(data.idx), show_scalar_operand(data.arg));
        }
        SIMJIT_MATCH (StepKind::CondScatter) {
            return format("if (%s) arg%zu[%s] = %s", show_scalar_operand(data.cond), data.dst,
                          show_scalar_operand(data.idx), show_scalar_operand(data.arg));
        }
        SIMJIT_MATCH (StepKind::Pack) {
            return format("if(%s) arg%zu[acc%zu++] = %s", show_scalar_operand(data.cond), data.dst,
                          func->accs.index(data.acc), show_scalar_operand(data.arg));
        }
        SIMJIT_MATCH (StepKind::CondStore) {
            return format("if (%s) arg%zu[i] = %s", show_scalar_operand(data.cond), data.addr.arg,
                          show_scalar_operand(data.arg));
        }
        SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
            const char *intrin = nullptr;
            switch (data.op) {
            case ArithBinaryOp::Add: intrin = "__builtin_add_overflow"; break;
            case ArithBinaryOp::Sub: intrin = "__builtin_sub_overflow"; break;
            case ArithBinaryOp::Mul: intrin = "__builtin_mul_overflow"; break;
            default: messed_up("binary op %s does not support safety checking", show_arith_binary_op(data.op));
            }

            std::string result = format("%s %s;\n", cpp_dtype(step->dtype), show_scalar_operand(step));
            std::string checked = format("%s(%s, %s, &%s)", intrin, show_scalar_operand(data.left),
                                         show_scalar_operand(data.right), show_scalar_operand(step));
            if (data.mask != nullptr) {
                checked = format("((%s != 0) & %s)", show_scalar_operand(data.mask), checked.c_str());
            }
            result += format("acc%zu |= %s", func->accs.index(data.overflow_flag), checked.c_str());
            return result;
        }
        SIMJIT_MATCH (StepKind::ScalarPermute) {
            size_t dtype_size = scalar_dtype_size(step->dtype.as_scalar());
            std::string result = cpp_var_decl(step);
            if (data.is_bit) {
                for (size_t i = 0; i < dtype_size * 8; ++i) {
                    size_t permute_idx = (((data.permute >> ((i & 0x7) * 8)) - 1) & 0xff);
                    result +=
                        format("(((((%s)%s >> %zu) & 0x1)) << (%s)%zu)",
                               scalar_dtype_to_cpp_unsigned(step->dtype.as_scalar()), show_scalar_operand(data.arg),
                               permute_idx + (i / 8 * 8), scalar_dtype_to_cpp_unsigned(step->dtype.as_scalar()), i);
                    if (i != dtype_size * 8 - 1) { result += " | "; }
                }
            } else {
                for (size_t i = 0; i < dtype_size; ++i) {
                    result +=
                        format("((((%s)%s >> %llu) & 0xff) << %zu)",
                               scalar_dtype_to_cpp_unsigned(step->dtype.as_scalar()), show_scalar_operand(data.arg),
                               (unsigned long long)(((data.permute >> (i * 8)) & 0xff) * 8), i * 8);
                    if (i != dtype_size - 1) { result += " | "; }
                }
            }
            return result;
        }
        SIMJIT_MATCH (StepKind::Fpclass) {
            std::string result;
            const char *arg = show_scalar_operand(data.arg);
            auto append = [&](const char *cl) {
                std::string call = format("fpclassify(%s) == %s", arg, cl);
                if (result.empty())
                    result = call;
                else
                    result += " || " + call;
            };
            if (bool(data.flags & FpClass::FPC_NAN)) { append("FP_NAN"); }
            if (bool(data.flags & FpClass::FPC_INFINITE)) { append("FP_INFINITE"); }
            if (bool(data.flags & FpClass::FPC_SUBNORMAL)) { append("FP_SUBNORMAL"); }
            if (bool(data.flags & FpClass::FPC_ZERO)) { append("FP_ZERO"); }
            SIMJIT_ASSERT(!result.empty());
            return cpp_var_decl(step) + result;
        }
    default: messed_up("unexpected step %s in scalar context", show_step_kind(step->kind));
    }
    SIMJIT_UNREACHABLE();
}

std::string CppEmitterBase::step_to_cpp(const Step *step, const MaskPushdownInfo *mask_pushdown) {
    if (mask_pushdown != nullptr) { return backend_step_to_cpp(step, mask_pushdown); }
    if (step->kind == StepKind::StoreSum128) { return store_sum128_to_cpp(step->step_data<StepKind::StoreSum128>()); }
    if (step->dtype.is_scalar() && is_scalar_step(step->kind)) { return backend_step_to_cpp(step, nullptr); }
    return backend_step_to_cpp(step, nullptr);
}

std::string CppEmitterBase::compile_steps(nonstd::span<Step *const> step_roots, size_t indent_level, EmitPhase phase,
                                          PrologueRootFilter prologue_filter) {
    std::string body{};
    const char *indent = indent_str(indent_level);
    std::vector<Step *> ordered{};
    std::vector<Step *> filtered_roots{};
    nonstd::span<Step *const> roots = step_roots;
    if (phase == EmitPhase::Prologue && prologue_filter != PrologueRootFilter::All) {
        for (Step *root : step_roots) {
            bool delayed = prologue_root_is_delayed(root);
            if ((prologue_filter == PrologueRootFilter::Delayed) == delayed) { filtered_roots.push_back(root); }
        }
        roots = filtered_roots.empty() ? nonstd::span<Step *const>(nullptr, 0)
                                       : nonstd::span<Step *const>(filtered_roots.data(), filtered_roots.size());
    }
    traverse_steps_postorder_unique(func->step_id_count, roots, [&](Step *x) {
        if (!seen[x->id].empty()) { return; }
        ordered.push_back(x);
    });

    ArenaBitmap suppressed = ArenaBitmap::create(func->ctx->arena, func->step_id_count);
    for (Step *step : ordered) {
        if (step->is(StepKind::AccLoad)) {
            suppressed.set(step->id);
            continue;
        }
        if (step->is(StepKind::Const) && peephole_named_use_counts[step->id] == 0 &&
            peephole_const_use_counts[step->id] != 0 && peephole_const_use_counts[step->id] == use_counts[step->id]) {
            suppressed.set(step->id);
            continue;
        }
        if (peephole_named_use_counts[step->id] == 0 && peephole_step_use_counts[step->id] != 0 &&
            peephole_step_use_counts[step->id] == use_counts[step->id]) {
            suppressed.set(step->id);
            continue;
        }
        if (!can_inline_scalar_expr(step)) { continue; }
        suppressed.set(step->id);
    }

    for (Step *step : ordered) {
        (void)suppress_compacted_acc_store_args(step, suppressed);
    }
    for (Step *step : ordered) {
        (void)suppress_inlined_acc_store_args(step, suppressed);
    }
    for (Step *step : ordered) {
        (void)suppress_inlined_agg_result_args(step, suppressed);
        (void)suppress_inlined_store_args(step, suppressed);
    }

    auto keep_name_for_suppressed = [&](const Step *step) {
        if (step->is(StepKind::Const) && peephole_named_use_counts[step->id] == 0 &&
            peephole_const_use_counts[step->id] != 0 && peephole_const_use_counts[step->id] == use_counts[step->id]) {
            return false;
        }
        if (peephole_named_use_counts[step->id] == 0 && peephole_step_use_counts[step->id] != 0 &&
            peephole_step_use_counts[step->id] == use_counts[step->id]) {
            return false;
        }
        return step->is(StepKind::AccLoad) || step->is(StepKind::Const) || step->is(StepKind::Load) ||
               step->is(StepKind::LoadSplat);
    };

    for (Step *step : ordered) {
        if (suppressed.get(step->id) && !keep_name_for_suppressed(step)) { continue; }
        if (auto name = assign_step_name(step, phase)) { seen[step->id] = std::move(*name); }
    }

    std::vector<std::optional<std::string>> compacted(func->step_id_count);
    for (Step *step : ordered) {
        compacted[step->id] = compact_acc_store_to_cpp(step);
    }

    for (Step *step : ordered) {
        if (suppressed.get(step->id)) { continue; }
        body += indent;
        std::string stmt{};
        std::optional<std::string> &compacted_step = compacted[step->id];
        if (compacted_step.has_value()) {
            auto &data = step->step_data<StepKind::AccStore>();
            stmt = format_acc_store(data.acc, compacted_step.value());
        } else if (has_custom_step_to_cpp(step, phase)) {
            stmt = custom_step_to_cpp(step, phase);
        } else {
            stmt = step_to_cpp(step);
        }
        for (char c : stmt) {
            body += c;
            if (c == '\n') { body += indent; }
        }
        body += ";\n";
    }
    return body;
}

std::string CppEmitterBase::emit_source() {
    seen.resize(func->step_id_count);
    init_use_counts();
    prepare_emit();
    init_inline_scalar_exprs();
    init_accumulators();
    bool has_delayed_prologue = has_delayed_prologue_steps(func->prologue_roots);

    std::string decl = format("void %s(size_t nelems", func->ctx->symbol_name.c_str());
    for (const ArgumentDecl &arg : func->args) {
        decl += ", ";
        bool writable = bool(arg.kind & ArgumentKind::Dst) || bool(arg.kind & ArgumentKind::DstAgg) ||
                        bool(arg.kind & ArgumentKind::DstSafetyCheck);
        if (!writable) { decl += "const "; }
        if (arg.dtype == ScalarDataType::I1) {
            decl += "uint8_t *";
        } else {
            decl += format("%s *", cpp_dtype(arg.dtype));
        }
        if (true) { decl += " __restrict "; }
        decl += format("arg%zu", arg.idx);
    }
    decl += ")";

    std::string code{};
    const std::string indent1(4, ' ');
    code += source_prelude_to_cpp();
    code += decl;
    code += " {\n";
    code += function_prelude_to_cpp(1);
    code += compile_steps(func->prologue_roots, 1, EmitPhase::Prologue,
                          has_delayed_prologue ? PrologueRootFilter::Immediate : PrologueRootFilter::All);
    code += indent1 + "size_t i = 0;\n";
    code += before_loops_to_cpp(1);
    if (!func->main_loop_roots.empty()) {
        code += indent1 + format("for (; i + %zu <= nelems; i += %zu) {\n", func->loop_width, func->loop_width);
        code += compile_steps(func->main_loop_roots, 2, EmitPhase::MainLoop);
        code += indent1 + "}\n";
    }
    if (has_delayed_prologue) {
        code += compile_steps(func->prologue_roots, 1, EmitPhase::Prologue, PrologueRootFilter::Delayed);
    }
    if (!func->remainder_roots.empty()) {
        if (!func->main_loop_roots.empty()) { code += indent1 + "if (i < nelems) {\n"; }
        if (true) { code += indent1 + "#pragma GCC unroll 1\n"; }
        code += indent1 + "for (; i < nelems; ++i) {\n";
        code += compile_steps(func->remainder_roots, 2, EmitPhase::Remainder);
        code += indent1 + "}\n";

        if (!func->main_loop_roots.empty()) { code += indent1 + "}\n"; }
    }
    code += after_loops_to_cpp(1);
    code += compile_steps(func->epilogue_roots, 1, EmitPhase::Epilogue);
    code += "}";
    return code;
}

#undef unsupported
#undef messed_up

} // namespace cpp_backend
} // namespace simjit
