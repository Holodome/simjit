// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/compiler.h"
#include "simjit/core/hir.h"
#include "simjit/core/mir.h"
#include "simjit/core/vectorizer.h"
#if SIMJIT_LLVM_BACKEND
#include "simjit/core/llvm/emitter.h"
#include "simjit/jit.h"
#endif
#include "simjit/simjit.h"

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace simjit;
using namespace simjit::types;

using IntegrationTest = std::function<void()>;

static hir::Step *single_stored_step(hir::Function *fn) {
    SIMJIT_ASSERT(fn->step_roots.size() == 1);
    hir::Step *root = fn->step_roots[0];
    SIMJIT_ASSERT(root->is(hir::StepKind::Store));
    return root->step_data<hir::StepKind::Store>().what;
}

static void add_pack_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
        FunctionBuilder b{ctx};
        Argument x = b.arg(I32);
        Value idx = b.input_arg(I32);
        Argument dst = b.arg(I32);
        Argument dst_size = b.arg(I64);
        Value tmp = b.gather(idx, x);
        Predicate cond = b.is_positive(tmp);
        b.pack(idx, cond, dst, dst_size);

        auto function = b.build();
        SIMJIT_ASSERT(function->accs.size() == 1);
        SIMJIT_ASSERT(function->accs[0].agg_expr == function->step_roots[0]);
        SIMJIT_ASSERT(function->accs[0].dst_arg == dst_size.idx_);
    });
}

static void add_checked_op_shape_tests(std::vector<IntegrationTest> &tests) {
    for (int32_t multiplier : {3, 4}) {
        tests.emplace_back([multiplier] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            builder.arg_safety_check();
            Value x = builder.input_arg(I32);
            Value checked = builder.checked_op(builder.mul(x, builder.i32(multiplier)));
            builder.store(checked, builder.arg(I32));

            hir::Step *step = single_stored_step(builder.build());
            SIMJIT_ASSERT(step->is(hir::StepKind::CheckedOp));
            hir::Step *operation = step->step_data<hir::StepKind::CheckedOp>().op;
            SIMJIT_ASSERT(operation->is(hir::StepKind::ArithBinary));
            SIMJIT_ASSERT(operation->step_data<hir::StepKind::ArithBinary>().op == ArithBinaryOp::Mul);
        });
    }

    for (ArithBinaryOp op : {ArithBinaryOp::Add, ArithBinaryOp::Sub, ArithBinaryOp::Mul,
                             ArithBinaryOp::ShiftLeftLogical, ArithBinaryOp::Div}) {
        tests.emplace_back([op] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            builder.arg_safety_check();
            Value left = builder.input_arg(I32);
            Value right = builder.input_arg(I32);
            Value checked = builder.arith_binary(left, right, op, ArithBinaryOpFlags::SafetyCheck);
            builder.store(checked, builder.arg(I32));

            hir::Step *step = single_stored_step(builder.build());
            SIMJIT_ASSERT(step->is(hir::StepKind::CheckedOp));
            hir::Step *operation = step->step_data<hir::StepKind::CheckedOp>().op;
            SIMJIT_ASSERT(operation->is(hir::StepKind::ArithBinary));
            const auto &arith = operation->step_data<hir::StepKind::ArithBinary>();
            SIMJIT_ASSERT(arith.op == op);
            SIMJIT_ASSERT(!bool(arith.flags & ArithBinaryOpFlags::SafetyCheck));
        });
    }

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
        FunctionBuilder builder{ctx};
        builder.arg_safety_check();
        Value checked = builder.trunc_checked(builder.input_arg(I64), I8);
        builder.store(checked, builder.arg(I8));

        hir::Step *step = single_stored_step(builder.build());
        SIMJIT_ASSERT(step->is(hir::StepKind::CheckedOp));
        hir::Step *operation = step->step_data<hir::StepKind::CheckedOp>().op;
        SIMJIT_ASSERT(operation->is(hir::StepKind::IntCast));
        SIMJIT_ASSERT(operation->step_data<hir::StepKind::IntCast>().kind == IntCastKind::Trunc);
    });

    for (bool is_signed : {false, true}) {
        tests.emplace_back([is_signed] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            builder.arg_safety_check();
            Value left = builder.input_arg(I32);
            Value right = builder.input_arg(I32);
            left = is_signed ? builder.sext(left, I64) : builder.zext(left, I64);
            right = is_signed ? builder.sext(right, I64) : builder.zext(right, I64);
            builder.store(builder.mul_checked(left, right), builder.arg(I64));

            hir::Step *step = single_stored_step(builder.build());
            SIMJIT_ASSERT(step->is(hir::StepKind::ArithBinary));
            ArithBinaryOp op = step->step_data<hir::StepKind::ArithBinary>().op;
            SIMJIT_ASSERT(op == (is_signed ? ArithBinaryOp::Mul64SE : ArithBinaryOp::Mul64ZE));
        });
    }
}

#if SIMJIT_LLVM_BACKEND
template <typename Predicate> static bool mir_has_step(mir::Function *fn, Predicate predicate) {
    bool found = false;
    std::vector<uint8_t> state(fn->step_id_count);
    auto visit_roots = [&](const auto &roots) {
        for (mir::Step *root : roots) {
            mir::traverse_steps_postorder_unique(root, state, [&](mir::Step *step) {
                if (predicate(step)) { found = true; }
            });
        }
    };
    visit_roots(fn->prologue_roots);
    visit_roots(fn->main_loop_roots);
    visit_roots(fn->remainder_roots);
    visit_roots(fn->epilogue_roots);
    return found;
}

static bool mir_has_arith_op(mir::Function *fn, ArithBinaryOp op) {
    return mir_has_step(fn, [op](mir::Step *step) {
        return step->is(mir::StepKind::ArithBinary) && step->step_data<mir::StepKind::ArithBinary>().op == op;
    });
}

static size_t count_substrings(std::string_view haystack, std::string_view needle) {
    size_t result = 0;
    size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string_view::npos) {
        ++result;
        pos += needle.size();
    }
    return result;
}

static void execute_llvm_add(void *fn) {
    alignas(64) int32_t input[32];
    alignas(64) int32_t output[32]{};
    for (int32_t i = 0; i < 32; ++i) {
        input[i] = i * 3 - 11;
    }
    void *args[] = {input, output};
    jit::call_fn_ptr(fn, 32, args);
    for (int32_t i = 0; i < 32; ++i) {
        SIMJIT_ASSERT(output[i] == input[i] + 7);
    }
}

static hir::Function *build_llvm_add(FunctionBuilder &builder) {
    Argument input = builder.arg(I32);
    Argument output = builder.arg(I32);
    builder.store(builder.add(builder.load(input), builder.i32(7)), output);
    return builder.build();
}

template <typename F> static void expect_exception(F &&fn) {
    bool threw = false;
    try {
        fn();
    } catch (const std::exception &) { threw = true; }
    SIMJIT_ASSERT(threw);
}

static void add_llvm_session_execution_tests(std::vector<IntegrationTest> &tests) {
    using llvm_backend::LLVMOptLevel;

    for (jit::CompilePolicy policy : {jit::CompilePolicy::Scalar, jit::CompilePolicy::Vectorized}) {
        for (ArithBinaryOp op : {ArithBinaryOp::Add, ArithBinaryOp::Sub, ArithBinaryOp::Mul}) {
            tests.emplace_back([policy, op] {
                MemoryArena arena;
                Context ctx{arena, "checked_mask", CodeTransformations::All, Arch::Native};
                FunctionBuilder builder{ctx};
                builder.arg_safety_check();
                Argument left_arg = builder.arg(I32);
                Argument right_arg = builder.arg(I32);
                Argument mask_arg = builder.arg(I1);
                Argument output_arg = builder.arg(I32);
                Value left = builder.load(left_arg);
                Value right = builder.load(right_arg);
                Predicate active = builder.load_predicate(mask_arg);
                Value operation = builder.arith_binary(left, right, op);
                builder.store(builder.checked_op(operation, active), output_arg);

                constexpr size_t count = 32;
                int32_t left_values[count]{};
                int32_t right_values[count]{};
                uint8_t masks[count]{};
                int32_t output[count]{};
                uint8_t overflow = 0;
                void *args[] = {&overflow, left_values, right_values, masks, output};

                llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
                void *fn = llvm_backend::compile_hir(builder.build(), session, policy);

                auto check = [&](int32_t left_value, int32_t right_value, bool a, bool expected_overflow,
                                 int32_t expected_output) {
                    left_values[0] = left_value;
                    right_values[0] = right_value;
                    masks[0] = a;
                    overflow = 0;
                    jit::call_fn_ptr(fn, count, args);
                    SIMJIT_ASSERT((overflow != 0) == expected_overflow);
                    SIMJIT_ASSERT(output[0] == expected_output);
                };

                int32_t max = std::numeric_limits<int32_t>::max();
                int32_t min = std::numeric_limits<int32_t>::min();
                int32_t overflow_left = op == ArithBinaryOp::Sub ? min : max;
                int32_t overflow_right = op == ArithBinaryOp::Mul ? 2 : 1;
                int32_t wrapped = op == ArithBinaryOp::Add ? min : op == ArithBinaryOp::Sub ? max : -2;
                check(overflow_left, overflow_right, false, false, wrapped);
                check(overflow_left, overflow_right, true, true, wrapped);
                if (op == ArithBinaryOp::Mul) {
                    check(123, 456, true, false, 56088);
                    // The vector check uses operand bit widths as a conservative filter. 32768 * 32768 fits in i32,
                    // but has the same widths as some overflowing products. Exact scalar lowering reports no overflow;
                    // the vector filter conservatively reports one.
                    check(32768, 32768, true, policy == jit::CompilePolicy::Vectorized, 1073741824);
                }
            });
        }

        tests.emplace_back([policy] {
            MemoryArena arena;
            Context ctx{arena, "checked_trunc", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            builder.arg_safety_check();
            Argument input_arg = builder.arg(I64);
            Argument mask_arg = builder.arg(I1);
            Argument output_arg = builder.arg(I8);
            Value input = builder.load(input_arg);
            Predicate active = builder.load_predicate(mask_arg);
            builder.store(builder.checked_op(builder.trunc(input, I8), active), output_arg);

            constexpr size_t count = 32;
            int64_t input_values[count]{};
            uint8_t masks[count]{};
            int8_t output[count]{};
            uint8_t overflow = 0;
            void *args[] = {&overflow, input_values, masks, output};

            llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
            void *fn = llvm_backend::compile_hir(builder.build(), session, policy);

            auto check = [&](int64_t input_value, bool active_value, int8_t expected_output, bool expected_overflow) {
                input_values[0] = input_value;
                masks[0] = active_value;
                overflow = 0;
                jit::call_fn_ptr(fn, count, args);
                SIMJIT_ASSERT(output[0] == expected_output);
                SIMJIT_ASSERT((overflow != 0) == expected_overflow);
            };

            check(127, true, 127, false);
            check(-128, true, -128, false);
            check(128, true, -128, true);
            check(-129, true, 127, true);
            check(int64_t{1} << 40, true, 0, true);
            check(int64_t{1} << 40, false, 0, false);
        });

        tests.emplace_back([policy] {
            MemoryArena arena;
            Context ctx{arena, "checked_rotate", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            builder.arg_safety_check();
            Argument value_arg = builder.arg(I32);
            Argument amount_arg = builder.arg(I32);
            Value value = builder.load(value_arg);
            Value amount = builder.load(amount_arg);
            builder.store(builder.rotl(value, amount), builder.arg(I32));
            builder.store(builder.rotl_checked(value, amount), builder.arg(I32));

            constexpr size_t count = 32;
            int32_t values[count]{};
            int32_t amounts[count]{};
            int32_t wrapped_output[count]{};
            int32_t checked_output[count]{};
            uint8_t overflow = 0;
            values[0] = int32_t(0xa0000000u);
            void *args[] = {&overflow, values, amounts, wrapped_output, checked_output};

            llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
            void *fn = llvm_backend::compile_hir(builder.build(), session, policy);

            auto check = [&](int32_t a, bool expected_overflow) {
                amounts[0] = a;
                overflow = 0;
                jit::call_fn_ptr(fn, count, args);
                SIMJIT_ASSERT((overflow != 0) == expected_overflow);
                SIMJIT_ASSERT(wrapped_output[0] == int32_t(0x40000001u));
                SIMJIT_ASSERT(checked_output[0] == int32_t(0x40000001u));
            };
            check(1, false);
            check(33, true);
        });

        tests.emplace_back([policy] {
            MemoryArena arena;
            Context ctx{arena, "checked_const_div", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            builder.arg_safety_check();
            Argument input_arg = builder.arg(I32);
            builder.store(builder.div_checked(builder.load(input_arg), builder.i32(7)), builder.arg(I32));

            constexpr size_t count = 32;
            int32_t input[count]{};
            int32_t output[count]{};
            for (int32_t i = 0; i < int32_t(count); ++i) {
                input[i] = (i - 11) * 21;
            }
            uint8_t overflow = 0;
            void *args[] = {&overflow, input, output};

            hir::Function *hir = builder.build();
            mir::Function *lowered = policy == jit::CompilePolicy::Scalar ? lower_scalar(hir) : lower_vectorized(hir);
#if SIMJIT_USE_LIBDIVIDE
            if (policy == jit::CompilePolicy::Scalar) {
                SIMJIT_ASSERT(mir_has_step(lowered, [](mir::Step *step) { return step->is(mir::StepKind::ConstDiv); }));
            }
            SIMJIT_ASSERT(!mir_has_arith_op(lowered, ArithBinaryOp::Div));
#endif
            llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
            void *fn = llvm_backend::compile_mir(lowered, session);
            jit::call_fn_ptr(fn, count, args);
            SIMJIT_ASSERT(overflow == 0);
            for (size_t i = 0; i < count; ++i) {
                SIMJIT_ASSERT(output[i] == input[i] / 7);
            }
        });
    }

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena, "checked_runtime_div", CodeTransformations::All, Arch::Native};
        FunctionBuilder builder{ctx};
        builder.arg_safety_check();
        Argument left_arg = builder.arg(I32);
        Argument right_arg = builder.arg(I32);
        builder.store(builder.div_checked(builder.load(left_arg), builder.load(right_arg)), builder.arg(I32));

        constexpr size_t count = 32;
        int32_t left[count]{};
        int32_t right[count];
        int32_t output[count]{};
        std::fill(std::begin(right), std::end(right), 1);
        uint8_t overflow = 0;
        void *args[] = {&overflow, left, right, output};

        mir::Function *lowered = lower_scalar(builder.build());
        SIMJIT_ASSERT(mir_has_arith_op(lowered, ArithBinaryOp::Div));
#if SIMJIT_USE_LIBDIVIDE
        SIMJIT_ASSERT(!mir_has_step(lowered, [](mir::Step *step) { return step->is(mir::StepKind::ConstDiv); }));
#endif
        llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
        void *fn = llvm_backend::compile_mir(lowered, session);

        auto check = [&](int32_t left_value, int32_t right_value, int32_t expected_output, bool expected_overflow) {
            left[0] = left_value;
            right[0] = right_value;
            overflow = 0;
            jit::call_fn_ptr(fn, count, args);
            SIMJIT_ASSERT(output[0] == expected_output);
            SIMJIT_ASSERT((overflow != 0) == expected_overflow);
        };
        check(84, 7, 12, false);
        check(84, 0, 84, true);
        int32_t min = std::numeric_limits<int32_t>::min();
        check(min, -1, min, true);
    });

    for (LLVMOptLevel level : {LLVMOptLevel::O1, LLVMOptLevel::O3}) {
        tests.emplace_back([level] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            hir::Function *hir = build_llvm_add(builder);
            llvm_backend::LLVMSession session{Arch::Native, level};
            SIMJIT_ASSERT(session.opt_level() == level);
            execute_llvm_add(llvm_backend::compile_hir(hir, session, jit::CompilePolicy::Scalar));
        });

        tests.emplace_back([level] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            hir::Function *hir = build_llvm_add(builder);
            llvm_backend::LLVMSession session{Arch::Native, level};
            execute_llvm_add(llvm_backend::compile_hir(hir, session, jit::CompilePolicy::Vectorized));
        });

        tests.emplace_back([level] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            mir::Function *mir = lower_scalar(build_llvm_add(builder));
            llvm_backend::LLVMSession session{Arch::Native, level};
            execute_llvm_add(llvm_backend::compile_mir(mir, session));
        });

        tests.emplace_back([level] {
            MemoryArena arena;
            Context ctx{arena, "expr", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            std::string ir = emit_llvm_ir(lower_scalar(build_llvm_add(builder)));
            llvm_backend::LLVMSession session{Arch::Native, level};
            execute_llvm_add(llvm_backend::compile_ir(ir, "expr", session));
        });
    }

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena, "expr", CodeTransformations::All & ~CodeTransformations::AccSplit, Arch::Native};
        FunctionBuilder builder{ctx};
        Argument f64_input = builder.arg(F64);
        Argument f64_output = builder.arg(F64);
        Argument i64_input = builder.arg(I64);
        Argument i128_output = builder.arg(I128);
        builder.sum(builder.load(f64_input), f64_output);
        builder.sum(builder.load(i64_input), i128_output);

        constexpr size_t count = 37;
        double f64_values[count];
        int64_t i64_values[count];
        double expected_f64 = 0;
        __int128 expected_i128 = 0;
        for (size_t i = 0; i < count; ++i) {
            f64_values[i] = double(int(i % 9) - 4);
            i64_values[i] = int64_t(i) * 100000000003LL - 1700000000000LL;
            expected_f64 += f64_values[i];
            expected_i128 += i64_values[i];
        }
        double actual_f64 = 0;
        __int128 actual_i128 = 0;
        void *args[] = {f64_values, &actual_f64, i64_values, &actual_i128};

        llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
        void *fn = llvm_backend::compile_hir(builder.build(), session, jit::CompilePolicy::Vectorized);
        jit::call_fn_ptr(fn, count, args);
        SIMJIT_ASSERT(actual_f64 == expected_f64);
        SIMJIT_ASSERT(actual_i128 == expected_i128);
    });

    tests.emplace_back([] {
        llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
        expect_exception([&] { (void)llvm_backend::compile_ir("not llvm ir", "expr", session); });
    });

    tests.emplace_back([] {
        static constexpr std::string_view ir = "define void @present(i64 %n) { ret void }";
        llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
        expect_exception([&] { (void)llvm_backend::compile_ir(ir, "missing", session); });
    });

    tests.emplace_back([] {
        static constexpr std::string_view ir = "define void @duplicate(i64 %n) { ret void }";
        llvm_backend::LLVMSession session{Arch::Native, LLVMOptLevel::O1};
        (void)llvm_backend::compile_ir(ir, "duplicate", session);
        expect_exception([&] { (void)llvm_backend::compile_ir(ir, "duplicate", session); });
    });

    tests.emplace_back([] {
#if defined(_M_X64) || defined(__x86_64__)
        constexpr Arch foreign_arch = Arch::Arm64_NEON;
#else
        constexpr Arch foreign_arch = Arch::Amd64_AVX512;
#endif
        expect_exception([&] { llvm_backend::LLVMSession session{foreign_arch, LLVMOptLevel::O1}; });
    });
}
#endif

static bool is_const_i1(hir::Step *step, bool value) {
    return step->is(hir::StepKind::Const) && step->step_data<hir::StepKind::Const>().dtype == ScalarDataType::I1 &&
           step->step_data<hir::StepKind::Const>().as_unsigned() == (value ? 1 : 0);
}

static bool is_const_zero(hir::Step *step) {
    return step->is(hir::StepKind::Const) && step->step_data<hir::StepKind::Const>().is_zero();
}

static void add_commutative_cse_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Value xy = local.add(x, y);
        Value yx = local.add(y, x);
        SIMJIT_ASSERT(xy.step_ == yx.step_);

        local.store(yx, local.arg(I32));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Value xy = local.min(x, y);
        Value yx = local.min(y, x);
        SIMJIT_ASSERT(xy.step_ == yx.step_);

        local.store(yx, local.arg(I32));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Predicate x = local.load_predicate(local.arg(I1));
        Predicate y = local.load_predicate(local.arg(I1));
        Predicate xy = local.and_(x, y);
        Predicate yx = local.and_(y, x);
        SIMJIT_ASSERT(xy.step_ == yx.step_);

        local.store(yx, local.arg(I1));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Value first = local.andnot(x, y);
        Value second = local.andnot(y, x);
        SIMJIT_ASSERT(first.step_ != second.step_);

        local.store(second, local.arg(I32));
    });
}

static void add_const_div_quotient_cache_tests(std::vector<IntegrationTest> &tests) {
#if SIMJIT_USE_LIBDIVIDE
    for (bool vectorized : {false, true}) {
        tests.emplace_back([vectorized] {
            MemoryArena arena;
            Context ctx{arena, "const_div_quotient_cache", CodeTransformations::All, Arch::Native};
            FunctionBuilder builder{ctx};
            Value x = builder.load(builder.arg(I32));
            Value y = builder.load(builder.arg(I32));
            Value divisor = builder.i32(100);
            Argument x_quotient_out = builder.arg(I32);
            Argument x_remainder_out = builder.arg(I32);
            Argument y_quotient_out = builder.arg(I32);
            builder.store(builder.div(x, divisor), x_quotient_out);
            builder.store(builder.mod(x, divisor), x_remainder_out);
            builder.store(builder.div(y, divisor), y_quotient_out);

            mir::Function *fn = vectorized ? lower_vectorized(builder.build()) : lower_scalar(builder.build());
            auto roots = vectorized ? fn->main_loop_roots : fn->remainder_roots;
            auto stored_values = [&](Argument output) {
                std::vector<mir::Step *> values;
                for (mir::Step *root : roots) {
                    if (root->is(mir::StepKind::Store)) {
                        const auto &store = root->step_data<mir::StepKind::Store>();
                        if (store.addr.arg == output.idx_) { values.push_back(store.what); }
                    } else if (root->is(mir::StepKind::CondStore)) {
                        const auto &store = root->step_data<mir::StepKind::CondStore>();
                        if (store.addr.arg == output.idx_) { values.push_back(store.arg); }
                    }
                }
                SIMJIT_ASSERT(!values.empty());
                return values;
            };
            std::vector<mir::Step *> x_quotients = stored_values(x_quotient_out);
            for (mir::Step *x_remainder : stored_values(x_remainder_out)) {
                bool found = false;
                mir::traverse_steps_postorder(x_remainder, [&](mir::Step *step) {
                    found |= std::find(x_quotients.begin(), x_quotients.end(), step) != x_quotients.end();
                });
                SIMJIT_ASSERT(found);
            }
            for (mir::Step *y_quotient : stored_values(y_quotient_out)) {
                bool found = false;
                mir::traverse_steps_postorder(y_quotient, [&](mir::Step *step) {
                    found |= std::find(x_quotients.begin(), x_quotients.end(), step) != x_quotients.end();
                });
                SIMJIT_ASSERT(!found);
            }
        });
    }
#else
    (void)tests;
#endif
}

static void add_logical_peephole_shape_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Predicate cond = local.load_predicate(local.arg(I1));
        Value truthy = local.load(local.arg(I32));
        Value falsy = local.load(local.arg(I32));
        local.store(local.select(local.not_(cond), truthy, falsy), local.arg(I32));

        hir::Step *selected = single_stored_step(local.build());
        SIMJIT_ASSERT(selected->is(hir::StepKind::Select));
        const auto &data = selected->step_data<hir::StepKind::Select>();
        SIMJIT_ASSERT(data.cond == cond.step_);
        SIMJIT_ASSERT(data.truthy == falsy.step_);
        SIMJIT_ASSERT(data.falsy == truthy.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value truthy = local.load(local.arg(I32));
        Value falsy = local.load(local.arg(I32));
        Predicate cond = local.load_predicate(local.arg(I1));

        SIMJIT_ASSERT(local.select(local.true_(), truthy, falsy).step_ == truthy.step_);
        SIMJIT_ASSERT(local.select(local.false_(), truthy, falsy).step_ == falsy.step_);
        SIMJIT_ASSERT(local.select(cond, truthy, truthy).step_ == truthy.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::ConstantConditionPeephole;
        FunctionBuilder local{ctx};

        Value truthy = local.load(local.arg(I32));
        Value falsy = local.load(local.arg(I32));
        local.store(local.select(local.true_(), truthy, falsy), local.arg(I32));

        hir::Step *selected = single_stored_step(local.build());
        SIMJIT_ASSERT(selected->is(hir::StepKind::Select));
        SIMJIT_ASSERT(is_const_i1(selected->step_data<hir::StepKind::Select>().cond, true));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value zero = local.i32(0);
        Value x = local.load(local.arg(I32));
        local.store(local.cmp_eq(zero, x), local.arg(I1));

        hir::Step *result = single_stored_step(local.build());
        SIMJIT_ASSERT(result->is(hir::StepKind::Compare));
        const auto &data = result->step_data<hir::StepKind::Compare>();
        SIMJIT_ASSERT(data.op == CmpOp::Equal);
        SIMJIT_ASSERT(data.left == x.step_);
        SIMJIT_ASSERT(data.right == zero.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value loaded = local.load(local.arg(I32));
        Value expr = local.add(local.load(local.arg(I32)), local.i32(1));
        local.store(local.cmp_ne(loaded, expr), local.arg(I1));

        hir::Step *result = single_stored_step(local.build());
        SIMJIT_ASSERT(result->is(hir::StepKind::Compare));
        const auto &data = result->step_data<hir::StepKind::Compare>();
        SIMJIT_ASSERT(data.op == CmpOp::NotEqual);
        SIMJIT_ASSERT(data.left == expr.step_);
        SIMJIT_ASSERT(data.right == loaded.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value zero = local.i32(0);
        Value x = local.load(local.arg(I32));
        Value truthy = local.load(local.arg(I32));
        Value falsy = local.load(local.arg(I32));
        Predicate cond = local.not_(local.cmp_eq(zero, x));
        local.store(local.select(cond, truthy, falsy), local.arg(I32));

        hir::Step *selected = single_stored_step(local.build());
        SIMJIT_ASSERT(selected->is(hir::StepKind::Select));
        hir::Step *cmp = selected->step_data<hir::StepKind::Select>().cond;
        SIMJIT_ASSERT(cmp->is(hir::StepKind::Compare));
        const auto &data = cmp->step_data<hir::StepKind::Compare>();
        SIMJIT_ASSERT(data.op == CmpOp::NotEqual);
        SIMJIT_ASSERT(data.left == x.step_);
        SIMJIT_ASSERT(data.right == zero.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Predicate left = local.load_predicate(local.arg(I1));
        Predicate right = local.load_predicate(local.arg(I1));
        local.store(local.andnot(local.not_(left), right), local.arg(I1));

        hir::Step *result = single_stored_step(local.build());
        SIMJIT_ASSERT(result->is(hir::StepKind::PredicateBinary));
        const auto &data = result->step_data<hir::StepKind::PredicateBinary>();
        SIMJIT_ASSERT(data.op == PredicateBinaryOp::And);
        SIMJIT_ASSERT(data.left == left.step_);
        SIMJIT_ASSERT(data.right == right.step_);
    });
}

static void add_binary_identity_peephole_shape_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.add(x, local.i32(0)), local.arg(I32));

        SIMJIT_ASSERT(single_stored_step(local.build()) == x.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::BinaryIdentityPeephole;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.add(x, local.i32(0)), local.arg(I32));

        hir::Step *result = single_stored_step(local.build());
        SIMJIT_ASSERT(result->is(hir::StepKind::ArithBinary));
        const auto &data = result->step_data<hir::StepKind::ArithBinary>();
        SIMJIT_ASSERT(data.op == ArithBinaryOp::Add);
        SIMJIT_ASSERT(data.left == x.step_);
        SIMJIT_ASSERT(data.right->is(hir::StepKind::Const));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value first = local.xor_(x, x);
        Value second = local.xor_(x, x);

        SIMJIT_ASSERT(first.step_ == second.step_);
        SIMJIT_ASSERT(is_const_zero(first.step_));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F32));
        Value mul_one = local.mul(x, local.f32(1.0f));
        local.store(local.mul(x, local.f32(0.0f)), local.arg(F32));

        SIMJIT_ASSERT(mul_one.step_ == x.step_);
        hir::Step *result = single_stored_step(local.build());
        SIMJIT_ASSERT(result->is(hir::StepKind::ArithBinary));
        SIMJIT_ASSERT(result->step_data<hir::StepKind::ArithBinary>().op == ArithBinaryOp::Mul);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Predicate p = local.load_predicate(local.arg(I1));

        SIMJIT_ASSERT(local.and_(p, p).step_ == p.step_);
        SIMJIT_ASSERT(local.andnot(local.false_(), p).step_ == p.step_);
        SIMJIT_ASSERT(is_const_i1(local.xor_(p, p).step_, false));
        SIMJIT_ASSERT(is_const_i1(local.xnor(p, p).step_, true));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));

        SIMJIT_ASSERT(is_const_i1(local.cmp_uge(x, local.i32(0)).step_, true));
        SIMJIT_ASSERT(is_const_i1(local.cmp_ult(x, local.i32(0)).step_, false));
        SIMJIT_ASSERT(is_const_i1(local.cmp_ne(x, x).step_, false));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value chosen = local.load(local.arg(I32));
        Value dead = local.add(local.load(local.arg(I32)), local.i32(17));
        Value cond_input = local.load(local.arg(I32));
        Predicate cond = local.and_(local.cmp_uge(cond_input, local.i32(0)), local.true_());
        local.store(local.select(cond, chosen, dead), local.arg(I32));

        SIMJIT_ASSERT(single_stored_step(local.build()) == chosen.step_);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::BinaryIdentityPeephole;
        FunctionBuilder local{ctx};

        Value chosen = local.load(local.arg(I32));
        Value dead = local.add(local.load(local.arg(I32)), local.i32(17));
        Value cond_input = local.load(local.arg(I32));
        Predicate cond = local.and_(local.cmp_uge(cond_input, local.i32(0)), local.true_());
        local.store(local.select(cond, chosen, dead), local.arg(I32));

        hir::Step *selected = single_stored_step(local.build());
        SIMJIT_ASSERT(selected->is(hir::StepKind::Select));
    });
}

static void add_conditional_identity_peephole_shape_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value dead_cond_input = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(dead_cond_input, dead_cond_input);
        local.cond_store(x, cond, local.arg(I32));

        hir::Step *root = local.build()->step_roots[0];
        SIMJIT_ASSERT(root->is(hir::StepKind::Store));
        SIMJIT_ASSERT(root->step_data<hir::StepKind::Store>().cond == nullptr);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value idx = local.index(I32);
        Value dead_cond_input = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(dead_cond_input, dead_cond_input);
        local.cond_scatter(x, idx, cond, local.arg(I32));

        hir::Step *root = local.build()->step_roots[0];
        SIMJIT_ASSERT(root->is(hir::StepKind::Scatter));
        SIMJIT_ASSERT(root->step_data<hir::StepKind::Scatter>().cond == nullptr);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I64));
        Value dead_cond_input = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(dead_cond_input, dead_cond_input);
        local.sum_if(x, cond, local.arg(I64));

        hir::Step *root = local.build()->step_roots[0];
        SIMJIT_ASSERT(root->is(hir::StepKind::AccArithBinary));
        SIMJIT_ASSERT(root->step_data<hir::StepKind::AccArithBinary>().cond == nullptr);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::ConstantConditionPeephole;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.cond_store(x, local.true_(), local.arg(I32));

        hir::Step *root = local.build()->step_roots[0];
        SIMJIT_ASSERT(root->is(hir::StepKind::Store));
        SIMJIT_ASSERT(is_const_i1(root->step_data<hir::StepKind::Store>().cond, true));
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Value chosen = local.load(local.arg(I32));
        Value dead = local.add(local.load(local.arg(I32)), local.i32(5));
        Value left_input = local.load(local.arg(I32));
        Value right_input = local.load(local.arg(I32));
        Predicate cond = local.or_(local.cmp_ult(left_input, local.i32(0)), local.cmp_eq(right_input, right_input));
        local.cond_store(local.select(cond, chosen, dead), cond, local.arg(I32));

        hir::Step *root = local.build()->step_roots[0];
        SIMJIT_ASSERT(root->is(hir::StepKind::Store));
        SIMJIT_ASSERT(root->step_data<hir::StepKind::Store>().cond == nullptr);
        SIMJIT_ASSERT(root->step_data<hir::StepKind::Store>().what == chosen.step_);
    });
}

static Value add_f64_constants(FunctionBuilder &b, Value expr, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        expr = b.add(expr, b.f64((double)i + 1.0));
    }
    return expr;
}

static void add_vectorizer_heuristic_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::SmallArith;
        FunctionBuilder local{ctx};

        Value x = local.input_arg(I8);
        Value y = local.input_arg(I8);
        local.store(local.mul(x, y), local.arg(I8));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(!result);
        SIMJIT_ASSERT(result.error().module == ErrorModule::Vectorizer);
        SIMJIT_ASSERT(result.error().kind == ErrorKind::VectorizationFailed);
        SIMJIT_ASSERT(result.error().subkind == ErrorSubKind::UnsupportedSpecialOps);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::SmallArith;
        FunctionBuilder local{ctx};

        Value x = local.input_arg(I8);
        Value y = local.input_arg(I8);
        local.store(local.mul(x, y), local.arg(I8));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(result);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::SmallArith;
        FunctionBuilder local{ctx};

        Value x8 = local.input_arg(I8);
        Value x16 = local.input_arg(I16);
        local.store(local.lzcnt(x8), local.arg(I8));
        local.store(local.lzcnt(x16), local.arg(I16));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(!result);
        SIMJIT_ASSERT(result.error().module == ErrorModule::Vectorizer);
        SIMJIT_ASSERT(result.error().kind == ErrorKind::VectorizationFailed);
        SIMJIT_ASSERT(result.error().subkind == ErrorSubKind::UnsupportedSpecialOps);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::SmallArith;
        FunctionBuilder local{ctx};

        Value x8 = local.input_arg(I8);
        Value x16 = local.input_arg(I16);
        local.store(local.lzcnt(x8), local.arg(I8));
        local.store(local.lzcnt(x16), local.arg(I16));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(result);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::SmallArith;
        FunctionBuilder local{ctx};

        Value x = local.input_arg(I8);
        Value amount = local.and_(local.input_arg(I8), local.i8(7));
        local.store(local.sll(x, amount), local.arg(I8));
        local.store(local.srl(x, amount), local.arg(I8));
        local.store(local.sra(x, amount), local.arg(I8));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(!result);
        SIMJIT_ASSERT(result.error().module == ErrorModule::Vectorizer);
        SIMJIT_ASSERT(result.error().kind == ErrorKind::VectorizationFailed);
        SIMJIT_ASSERT(result.error().subkind == ErrorSubKind::UnsupportedSpecialOps);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::SmallArith;
        FunctionBuilder local{ctx};

        Value x = local.input_arg(I8);
        Value amount = local.and_(local.input_arg(I8), local.i8(7));
        local.store(local.sll(x, amount), local.arg(I8));
        local.store(local.srl(x, amount), local.arg(I8));
        local.store(local.sra(x, amount), local.arg(I8));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(result);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        local.store(local.add(x, local.f64(1.0)), local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 32);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::ProactiveUnroll;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        local.store(local.add(x, local.f64(1.0)), local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 8);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::ProactiveUnroll;
        FunctionBuilder local{ctx};

        local.store(local.load(local.arg(F64)), local.arg(F64));
        local.store(local.load(local.arg(I32)), local.arg(I32));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(result);
        SIMJIT_ASSERT(result.value()->loop_width == 16);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        ctx.transformations = CodeTransformations::All & ~CodeTransformations::ProactiveUnroll;
        FunctionBuilder local{ctx};

        local.store(local.sext(local.input_arg(I8), I32), local.arg(I32));
        local.store(local.load(local.arg(I16)), local.arg(I16));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(result);
        const auto &roots = result.value()->roots;
        SIMJIT_ASSERT(roots.size() == 5);
        for (size_t i = 0; i < 4; ++i) {
            SIMJIT_ASSERT(roots[i].logical_root_idx == 0);
            SIMJIT_ASSERT(roots[i].block.width == 4);
            SIMJIT_ASSERT(roots[i].block.idx == i);
            SIMJIT_ASSERT(roots[i].unroll_coef == 0);
        }
        SIMJIT_ASSERT(roots[4].logical_root_idx == 1);
        SIMJIT_ASSERT(roots[4].block.width == 8);
        SIMJIT_ASSERT(roots[4].block.idx == 0);
        SIMJIT_ASSERT(roots[4].unroll_coef == 1);
        SIMJIT_ASSERT(result.value()->loop_width == 16);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        ctx.transformations =
            CodeTransformations::All & ~CodeTransformations::ProactiveUnroll & ~CodeTransformations::Unroll;
        FunctionBuilder local{ctx};

        local.store(local.load(local.arg(F64)), local.arg(F64));
        local.store(local.load(local.arg(I32)), local.arg(I32));

        auto result = vect::try_hir_to_vect(local.build());
        SIMJIT_ASSERT(!result);
        SIMJIT_ASSERT(result.error().module == ErrorModule::Vectorizer);
        SIMJIT_ASSERT(result.error().kind == ErrorKind::VectorizationFailed);
        SIMJIT_ASSERT(result.error().subkind == ErrorSubKind::RootWidthsMismatch);
    });

    tests.emplace_back([] {
        MemoryArena split_arena;
        Context split_ctx{split_arena};
        split_ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder split_builder{split_ctx};
        split_builder.sum(split_builder.load(split_builder.arg(F64)), split_builder.arg(F64));
        mir::Function *split = lower_vectorized(split_builder.build());

        MemoryArena shared_arena;
        Context shared_ctx{shared_arena};
        shared_ctx.arch = Arch::Amd64_AVX512;
        shared_ctx.transformations = CodeTransformations::All & ~CodeTransformations::AccSplit;
        FunctionBuilder shared_builder{shared_ctx};
        shared_builder.sum(shared_builder.load(shared_builder.arg(F64)), shared_builder.arg(F64));
        mir::Function *shared = lower_vectorized(shared_builder.build());

        SIMJIT_ASSERT(split->loop_width == shared->loop_width);
        SIMJIT_ASSERT(split->accs.agg_count > shared->accs.agg_count);
        SIMJIT_ASSERT(shared->accs.agg_count == 2);
    });

    tests.emplace_back([] {
        MemoryArena split_arena;
        Context split_ctx{split_arena};
        split_ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder split_builder{split_ctx};
        split_builder.sum(split_builder.load(split_builder.arg(I64)), split_builder.arg(I128));
        mir::Function *split = lower_vectorized(split_builder.build());

        MemoryArena shared_arena;
        Context shared_ctx{shared_arena};
        shared_ctx.arch = Arch::Amd64_AVX512;
        shared_ctx.transformations = CodeTransformations::All & ~CodeTransformations::AccSplit;
        FunctionBuilder shared_builder{shared_ctx};
        shared_builder.sum(shared_builder.load(shared_builder.arg(I64)), shared_builder.arg(I128));
        mir::Function *shared = lower_vectorized(shared_builder.build());

        SIMJIT_ASSERT(split->loop_width == shared->loop_width);
        SIMJIT_ASSERT(split->accs.agg_count > shared->accs.agg_count);
        SIMJIT_ASSERT(shared->accs.agg_count == 4);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value left = local.load(local.arg(F64));
        Value right = local.load(local.arg(F64));
        local.store(local.max(left, right), local.arg(F64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_max_pd") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_mov_pd(_mm512_max_pd") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_max_pd(") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        local.max_agg(local.load(local.arg(F64)), local.arg(F64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_max_pd") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_mov_pd(_mm512_max_pd") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_max_pd(") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        deserialize(R"SIMJIT(
(func
  (args
    (arg 0 i1 dst-scalar)
    (arg 1 i64 dst-scalar))
  (accs
    (acc 0 i1 (arg 0) (step 3))
    (acc 1 i64 (arg 1) (step 5)))
  (steps
    (step 0 const i32 "0x0")
    (step 1 index i32)
    (step 2 cmp i1 ge (step 0) (step 1) t)
    (step 3 acc-predicate-bin i1 xor (step 2) (acc 0))
    (step 4 const i1 "0x1")
    (step 5 countif i64 (step 4) (acc 1)))
  (roots (step 3) (step 5)))
)SIMJIT",
                    local);

        AsmjitCompileResult result{};
        compile_asmjit(lower_vectorized(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        bool saw_spilled_mask_to_gp = false;
        bool saw_mask_reg_to_gp = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            saw_spilled_mask_to_gp |= line.find("kmovw rax, word ptr") != std::string_view::npos;
            saw_mask_reg_to_gp |= line.find("kmovw rax, k") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(!saw_spilled_mask_to_gp);
        SIMJIT_ASSERT(saw_mask_reg_to_gp);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Value z = local.load(local.arg(I32));
        Predicate inner = local.and_(local.cmp_gt(y, local.i32(10)), local.cmp_eq(z, local.i32(0)));
        local.store(local.and_(local.cmp_lt(x, local.i32(0)), inner), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_mask_cmp_epi32_mask(,") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cmp_epi32_mask(,") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_mask_cmp_epi32_mask(x") != std::string::npos ||
                      cpp.find("_mm512_mask_cmp_epi32_mask(x") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Value z = local.load(local.arg(I32));
        Predicate left = local.and_(local.cmp_gt(x, local.i32(0)), local.cmp_le(y, local.i32(100)));
        Predicate right = local.cmp_lt(z, local.i32(50));
        local.store(local.and_(left, right), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_mask_cmp_epi32_mask(,") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cmp_epi32_mask(,") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value disc_price = local.load(local.arg(F64));
        Value tax = local.load(local.arg(F64));
        local.store(local.mul(disc_price, local.add(local.f64(1.0), tax)), local.arg(F64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_fmadd_pd") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_add_pd") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mul_pd") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value price = local.load(local.arg(F64));
        Value discount = local.load(local.arg(F64));
        local.store(local.mul(price, local.sub(local.f64(1.0), discount)), local.arg(F64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_fnmadd_pd") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_sub_pd") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mul_pd") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Predicate left = local.cmp_gt(x, local.i32(0));
        Predicate right = local.cmp_le(y, local.i32(100));
        local.store(local.and_(left, right), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_mask_cmp_epi32_mask(,") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cmp_epi32_mask(,") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_mask_cmp_epi32_mask(x") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value shared = local.add(x, local.f64(1.0));
        local.store(local.add(shared, shared), local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 16);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value f64_expr = add_f64_constants(local, local.load(local.arg(F64)), 3);
        local.store(f64_expr, local.arg(F64));
        Value i32_expr = local.add(local.load(local.arg(I32)), local.i32(1));
        local.store(i32_expr, local.arg(I32));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 16);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value expr = add_f64_constants(local, local.load(local.arg(F64)), 10);
        local.store(expr, local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 8);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value expr = add_f64_constants(local, local.load(local.arg(F64)), 9);
        local.sum(expr, local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 16);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value idx = local.load(local.arg(I32));
        Value expr = add_f64_constants(local, local.gather(idx, local.arg(F64)), 8);
        local.sum(expr, local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 8);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value expr = local.div(x, local.f64(2.0));
        expr = add_f64_constants(local, expr, 8);
        local.sum(expr, local.arg(F64));

        auto *mir = lower_vectorized(local.build());
        SIMJIT_ASSERT(mir->loop_width == 8);
    });
}

static void add_float_index_lowering_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        local.store(local.index(F32), local.arg(F32));
        mir::Function *fn = lower_vectorized(local.build());
        bool found = false;
        mir::traverse_steps_postorder_unique(fn->step_id_count, fn->main_loop_roots, [&](mir::Step *step) {
            if (!step->is(mir::StepKind::FloatCast)) { return; }
            SIMJIT_ASSERT(step->dtype.as_vec().elem == VecElemType::F32);
            mir::Step *integer_index = step->step_data<mir::StepKind::FloatCast>().arg;
            SIMJIT_ASSERT(integer_index->dtype.as_vec().elem == VecElemType::I32);
            bool found_vec_index = false;
            mir::traverse_steps_postorder(integer_index, [&](mir::Step *index) {
                if (!index->is(mir::StepKind::VecIndex)) { return; }
                SIMJIT_ASSERT(index->dtype.as_vec().elem == VecElemType::I32);
                mir::AccId acc = index->step_data<mir::StepKind::VecIndex>().acc;
                SIMJIT_ASSERT(acc.is_special());
                bool found_init = false;
                for (mir::Step *root : fn->prologue_roots) {
                    if (!root->is(mir::StepKind::AccStore)) continue;
                    const auto &store = root->step_data<mir::StepKind::AccStore>();
                    if (store.acc == acc) {
                        SIMJIT_ASSERT(root->dtype == index->dtype);
                        found_init = true;
                    }
                }
                SIMJIT_ASSERT(found_init);
                found_vec_index = true;
            });
            SIMJIT_ASSERT(found_vec_index);
            found = true;
        });
        SIMJIT_ASSERT(found);
    });
}

static void add_small_gather_scatter_index_tests(std::vector<IntegrationTest> &tests) {
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        Argument data = local.arg(I32);
        Value idx = local.input_arg(I8);
        local.store(local.gather(idx, data), local.arg(I32));

        hir::Function *fn = local.build();
        hir::Step *gather = single_stored_step(fn);
        SIMJIT_ASSERT(gather->is(hir::StepKind::Gather));
        hir::Step *cast = gather->step_data<hir::StepKind::Gather>().idx;
        SIMJIT_ASSERT(cast->is(hir::StepKind::IntCast));
        SIMJIT_ASSERT(cast->dtype == I32);
        const hir::IntCastData &cast_data = cast->step_data<hir::StepKind::IntCast>();
        SIMJIT_ASSERT(cast_data.kind == IntCastKind::Zext);
        SIMJIT_ASSERT(cast_data.arg->dtype == I8);
        SIMJIT_ASSERT(cast_data.arg->is(hir::StepKind::Load));
        ArgumentIdx idx_arg = cast_data.arg->step_data<hir::StepKind::Load>().idx;
        SIMJIT_ASSERT(fn->args[idx_arg].kind == ArgumentKind::SrcIdxArr);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        FunctionBuilder local{ctx};

        local.scatter(local.input_arg(I32), local.input_arg(I16), local.arg(I32));

        hir::Function *fn = local.build();
        SIMJIT_ASSERT(fn->step_roots.size() == 1);
        hir::Step *scatter = fn->step_roots[0];
        SIMJIT_ASSERT(scatter->is(hir::StepKind::Scatter));
        hir::Step *cast = scatter->step_data<hir::StepKind::Scatter>().idx;
        SIMJIT_ASSERT(cast->is(hir::StepKind::IntCast));
        SIMJIT_ASSERT(cast->dtype == I32);
        const hir::IntCastData &cast_data = cast->step_data<hir::StepKind::IntCast>();
        SIMJIT_ASSERT(cast_data.kind == IntCastKind::Zext);
        SIMJIT_ASSERT(cast_data.arg->dtype == I16);
        SIMJIT_ASSERT(cast_data.arg->is(hir::StepKind::Load));
        ArgumentIdx idx_arg = cast_data.arg->step_data<hir::StepKind::Load>().idx;
        SIMJIT_ASSERT(fn->args[idx_arg].kind == ArgumentKind::SrcIdxArr);
    });
}

#if SIMJIT_LLVM_BACKEND
static void build_mixed_type_index_case(FunctionBuilder &b) {
    Argument src_const = b.arg(I32);
    Argument scatter_dst = b.arg(I32);
    Argument gather32_src = b.arg(F64);
    Argument gather64_src = b.arg(F64);
    Argument count32_dst = b.arg(I64);
    Argument count64_dst = b.arg(I64);

    Value idx32 = b.index(I32);
    Value idx64 = b.index(I64);
    b.cond_scatter(b.load_splat(src_const), idx32, b.true_(), scatter_dst);

    Value gathered32_bits = b.bitcast(b.gather(idx32, gather32_src), I64);
    Value gathered64_bits = b.bitcast(b.gather(idx64, gather64_src), I64);
    b.countif(b.cmp_ugt(gathered32_bits, b.i64(0)), count32_dst);
    b.countif(b.cmp_ugt(gathered64_bits, b.i64(0)), count64_dst);
}

static void build_mixed_index_store_cast_case(FunctionBuilder &b) {
    Argument dst_i32 = b.arg(I32);
    Argument dst_f64 = b.arg(F64);
    Argument dst_i64_from_i32 = b.arg(I64);
    Argument dst_i64 = b.arg(I64);

    Value idx32 = b.index(I32);
    b.store(idx32, dst_i32);
    b.store(b.float_cast(idx32, F64, true), dst_f64);
    b.store(b.zext(idx32, I64), dst_i64_from_i32);
    b.store(b.index(I64), dst_i64);
}
#endif

static void add_llvm_emitter_format_tests(std::vector<IntegrationTest> &tests) {
#if SIMJIT_LLVM_BACKEND
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value mag = local.abs(x);
        local.store(local.select(local.isnan(mag), local.f64(1.0), mag), local.arg(F64));

        std::string llvm = emit_llvm_ir(lower_vectorized(local.build()));
        SIMJIT_ASSERT(llvm.find("@llvm.fabs.v2f64") != std::string::npos);
        SIMJIT_ASSERT(llvm.find("@llvm.is.fpclass.v2f64") != std::string::npos);
        SIMJIT_ASSERT(llvm.find(" fast") == std::string::npos);
    });
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        build_mixed_type_index_case(local);

        std::string llvm = emit_llvm_ir(lower_vectorized(local.build()));
        SIMJIT_ASSERT(count_substrings(llvm, "store <8 x i32> bitcast (<32 x i8>") == 1);
        SIMJIT_ASSERT(count_substrings(llvm, "store <8 x i64> bitcast (<64 x i8>") == 1);
        SIMJIT_ASSERT(llvm.find("store <8 x i64> bitcast (<32 x i8>") == std::string::npos);
        SIMJIT_ASSERT(llvm.find("store <8 x i32> bitcast (<64 x i8>") == std::string::npos);
    });
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        build_mixed_index_store_cast_case(local);

        std::string llvm = emit_llvm_ir(lower_vectorized(local.build()));
        SIMJIT_ASSERT(count_substrings(llvm, "store <4 x i32> bitcast (<16 x i8>") == 1);
        SIMJIT_ASSERT(count_substrings(llvm, "store <2 x i64> bitcast (<16 x i8>") == 2);
    });
#else
    (void)tests;
#endif
}

static void add_cpp_emitter_format_tests(std::vector<IntegrationTest> &tests) {
#if SIMJIT_CPP_BACKEND
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value idx = local.input_arg(I32);
        local.store(local.gather(idx, local.arg(I32)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("const int32_t gather_values_") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vgetq_lane_s32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vld1q_s32(gather_values_") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        local.store(local.cmp_lt(local.input_arg(I32), local.i32(0)), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(cpp.find("\n((uint64_t *)") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("\n        ((uint64_t *)") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value f32 = local.load(local.arg(F32));
        Value f64 = local.load(local.arg(F64));
        local.store(local.bitcast(local.not_(f32), I32), local.arg(I32));
        local.store(local.bitcast(local.not_(f64), I64), local.arg(I64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("vreinterpretq_f32_u8(vmvnq_u8(vreinterpretq_u8_f32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vreinterpretq_f64_u8(vmvnq_u8(vreinterpretq_u8_f64") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("simjit_arm_not_f32") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("simjit_arm_not_f64") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.cmp_lt(x, local.i32(0)), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(cpp.find("mask_shift_writer_acc_") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("((uint64_t *)") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("mask_store_values") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value selected = local.select(local.cmp_gt(x, local.i32(0)), local.i32(1), local.i32(0));
        local.store(selected, local.arg(I32));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(cpp.find("arg1[i] = (arg0[i] > 0)") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("? 1 : 0") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value selected = local.select(local.cmp_gt(x, local.i32(0)), local.i64(0), local.i64(1));
        local.store(selected, local.arg(I64));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(cpp.find("arg1[i] = !(arg0[i] > 0)") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("? 0 : 1") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.sll(x, local.i32(3)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(cpp.find("arg1[i] = (uint32_t)arg0[i] << 3") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("((uint32_t)(((uint32_t)") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.add(x, local.i32(1)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(
            cpp.find("void expr(size_t nelems, const int32_t * __restrict arg0, int32_t * __restrict arg1)") !=
            std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.add(x, local.i32(1)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("int32x4_t") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vld1q_s32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vaddq_s32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vst1q_s32") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.i32(1), local.i32(0)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("vceqq_s32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("vbslq_s32") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.zext(x, I64), local.arg(I64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("vmovl_u32") != std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Arm64_NEON;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.add(x, local.i32(1)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_scalar(local.build()));
        SIMJIT_ASSERT(cpp.find("arg0[i] + 1") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("int32x4_t") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Argument src = local.arg(I32);
        Value x = local.load(src);
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.add(x, local.i32(1)), local.i32(0)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_add_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512_YMM;
        FunctionBuilder local{ctx};

        Argument src = local.arg(I32);
        Value x = local.load(src);
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.add(x, local.i32(1)), local.i32(0)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_maskz_add_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_maskz_mov_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.cmp_eq(x, local.i32(0)), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_testn_epi32_mask") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_cmpeq_epi32_mask") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        local.store(local.bit_test(x, y), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_test_epi32_mask") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_cmpneq_epi32_mask") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Predicate mask = local.load_predicate(local.arg(I1));
        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        local.store(local.and_(mask, local.bit_testn(x, y)), local.arg(I1));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm256_mask_testn_epi32_mask") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_cmpeq_epi32_mask") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_kand_mask") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F32));
        Predicate cond = local.cmp_gt(x, local.f32(0.0f));
        local.store(local.select(cond, local.round_down(x), local.f32(0.0f)), local.arg(F32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_roundscale_ps") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_ps") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_roundscale_ps") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value y = local.load(local.arg(F64));
        Predicate cond = local.cmp_gt(x, local.f64(0.0));
        local.store(local.select(cond, local.round_toward_zero(x), y), local.arg(F64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_roundscale_pd") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_pd") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_roundscale_pd") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.cond_store(local.trunc(x, I16), cond, local.arg(I16), LoadStoreKind::Aligned);

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cvtepi32_storeu_epi16") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_cvtepi32_epi16") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.zext(x, I64), local.i64(0)), local.arg(I64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_cvtepu32_epi64") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi64") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I64));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.sext(x, I64), y), local.arg(I64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cvtepi32_epi64") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi64") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I16));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.trunc(x, I16), y), local.arg(I16));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cvtepi32_epi16") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi16") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_cvtepi32_epi16") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value truthy = local.zext(x, I64);
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, truthy, local.i64(0)), local.arg(I64));
        local.store(truthy, local.arg(I64));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_cvtepu32_epi64") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi64") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_cvtepu32_epi64") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.trunc(x, I16), local.arg(I16), LoadStoreKind::Aligned);

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cvtepi32_storeu_epi16") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_cvtepi32_epi16") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm256_store") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        local.store(local.trunc(x, I16), local.arg(I16), LoadStoreKind::Unaligned);

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cvtepi32_storeu_epi16") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_cvtepi32_epi16") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.cond_store(local.trunc(x, I16), cond, local.arg(I16));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_cvtepi32_storeu_epi16") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_cvtepi32_epi16") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_gt(x, local.i32(0));
        local.sum_if(x, cond, local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_add_epi32(acc") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.sll(x, local.i32(3)), local.i32(0)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_slli_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, y, local.i32(0)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_loadu_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.i32(-1), local.i32(0)), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_movm_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.add(x, local.i32(1)), y), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_add_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi32") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_add_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.sll(x, local.i32(3)), y), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_slli_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, y, x), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_loadu_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi32") == std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_loadu_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Argument table = local.arg(I32);
        Value idx = local.load(local.arg(I32));
        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, local.gather(idx, table), y), local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_mask_i32gather_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value truthy = local.add(x, local.i32(1));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, truthy, local.i32(0)), local.arg(I32));
        local.store(truthy, local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_add_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_mov_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_maskz_add_epi32") == std::string::npos);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I32));
        Value y = local.load(local.arg(I32));
        Value truthy = local.add(x, local.i32(1));
        Predicate cond = local.cmp_eq(x, local.i32(0));
        local.store(local.select(cond, truthy, y), local.arg(I32));
        local.store(truthy, local.arg(I32));

        std::string cpp = emit_cpp_source(lower_vectorized(local.build()));
        SIMJIT_ASSERT(cpp.find("_mm512_add_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_blend_epi32") != std::string::npos);
        SIMJIT_ASSERT(cpp.find("_mm512_mask_add_epi32") == std::string::npos);
    });

#else
    (void)tests;
#endif
}

static void add_asmjit_x86_peephole_tests(std::vector<IntegrationTest> &tests) {
#if SIMJIT_ASMJIT_BACKEND_X86
    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value y = local.load(local.arg(F64));
        Value z = local.load(local.arg(F64));
        local.store(local.sub(local.mul(x, y), z), local.arg(F64));

        AsmjitCompileResult result{};
        compile_asmjit(lower_scalar(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        bool saw_fma = false;
        bool saw_separate_arithmetic = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            saw_fma |= line.find("vfmsub213sd") != std::string_view::npos;
            saw_separate_arithmetic |=
                line.find("vmulsd") != std::string_view::npos || line.find("vsubsd") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(saw_fma);
        SIMJIT_ASSERT(!saw_separate_arithmetic);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value left = local.load(local.arg(F64));
        Value right = local.load(local.arg(F64));
        Predicate cond = local.load_predicate(local.arg(I1));
        local.store(local.select(cond, local.max(left, right), left), local.arg(F64));

        AsmjitCompileResult result{};
        compile_asmjit(lower_vectorized(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        bool saw_masked_max = false;
        bool saw_masked_mov_repair = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            saw_masked_max |=
                line.find("vmaxpd") != std::string_view::npos && line.find("{k") != std::string_view::npos;
            saw_masked_mov_repair |=
                line.find("vmovapd zmm") != std::string_view::npos && line.find("{k") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(saw_masked_max);
        SIMJIT_ASSERT(!saw_masked_mov_repair);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value y = local.load(local.arg(F64));
        local.sum(local.mul(x, y), local.arg(F64));
        local.sum(local.mul(x, x), local.arg(F64));
        local.sum(local.mul(y, y), local.arg(F64));

        AsmjitCompileResult result{};
        compile_asmjit(lower_vectorized(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        size_t fma_acc_updates = 0;
        bool saw_old_fma_form = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            if (line.find("vfmadd231pd") != std::string_view::npos) { ++fma_acc_updates; }
            saw_old_fma_form |= line.find("vfmadd213pd") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(fma_acc_updates >= 3);
        SIMJIT_ASSERT(!saw_old_fma_form);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        // This fuzzer case lowers the scalar predicate xor reduction into eight
        // two-lane F64 compare chunks inside a 16-lane AVX512 loop. Those F64
        // splats are XMM values, not ZMM values.
        deserialize(R"SIMJIT(
(func
  (args
    (arg 0 i1 dst-scalar)
    (arg 1 f64 src-const)
    (arg 2 i1 dst-scalar)
    (arg 3 i32 dst-arr))
  (accs
    (acc 0 i1 (arg 0) (step 6))
    (acc 1 i1 (arg 2) (step 11)))
  (steps
    (step 0 const i32 "0x0")
    (step 1 cmp i1 eq (step 0) (step 0) t)
    (step 2 const i1 "0x0")
    (step 3 predicate-binary i1 andnot (step 2) (step 2))
    (step 4 predicate-binary i1 andnot (step 1) (step 3))
    (step 5 predicate-not i1 (step 4))
    (step 6 acc-predicate-bin i1 or (step 5) (acc 0))
    (step 7 load-splat f64 (arg 1))
    (step 8 const f64 "0x4024000000000000")
    (step 9 cmp i1 lt (step 7) (step 8))
    (step 10 predicate-not i1 (step 9))
    (step 11 acc-predicate-bin i1 xor (step 10) (acc 1))
    (step 12 store i32 (step 0) (arg 3) unaligned))
  (roots (step 6) (step 11) (step 12)))
)SIMJIT",
                    local);

        AsmjitCompileResult result{};
        compile_asmjit(lower_vectorized(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        size_t xmm_duplicate_count = 0;
        bool saw_invalid_xmm_broadcast = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            if (line.find("vmovddup xmm") != std::string_view::npos) { ++xmm_duplicate_count; }
            saw_invalid_xmm_broadcast |= line.find("vbroadcastsd xmm") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(xmm_duplicate_count >= 2);
        SIMJIT_ASSERT(!saw_invalid_xmm_broadcast);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(I64));
        Value y = local.load(local.arg(I64));
        Value expr = local.mul(x, local.sub(local.i64(10000), y));
        local.store(expr, local.arg(I64));

        AsmjitCompileResult result{};
        compile_asmjit(lower_vectorized(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        bool saw_mem_mul = false;
        bool saw_standalone_load = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            saw_mem_mul |=
                line.find("vpmullq") != std::string_view::npos && line.find("word ptr") != std::string_view::npos;
            saw_standalone_load |= line.find("vmovdqu64 zmm") != std::string_view::npos &&
                                   line.find(", zmmword ptr") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(saw_mem_mul);
        SIMJIT_ASSERT(!saw_standalone_load);
    });

    tests.emplace_back([] {
        MemoryArena arena;
        Context ctx{arena};
        ctx.arch = Arch::Amd64_AVX512;
        FunctionBuilder local{ctx};

        Value x = local.load(local.arg(F64));
        Value y = local.load(local.arg(F64));
        Value z = local.load(local.arg(F64));
        Predicate cond1 = local.cmp_gt(local.mul(x, local.sub(local.f64(1.0), y)), local.f64(1000.0));
        Predicate cond2 = local.cmp_lt(z, local.f64(0.05));
        local.store(local.and_(cond1, cond2), local.arg(I1));

        AsmjitCompileResult result{};
        compile_asmjit(lower_vectorized(local.build()), AsmjitCompileOptions{false, true, nullptr}, result);

        auto last_operand = [](std::string_view line) {
            size_t comma = line.rfind(',');
            if (comma == std::string_view::npos) { return std::string_view{}; }
            line.remove_prefix(comma + 1);
            while (!line.empty() && line.front() == ' ') {
                line.remove_prefix(1);
            }
            return line;
        };

        bool saw_fma = false;
        bool saw_mem_cmp = false;
        bool saw_old_lt_cmp = false;
        size_t pos = 0;
        while (pos < result.asm_code.size()) {
            size_t end = result.asm_code.find('\n', pos);
            if (end == std::string::npos) { end = result.asm_code.size(); }
            std::string_view line{result.asm_code.data() + pos, end - pos};
            if (line.find("vcmppd") != std::string_view::npos) {
                std::string_view imm = last_operand(line);
                bool has_mem = line.find("word ptr") != std::string_view::npos;
                saw_mem_cmp |= has_mem && imm == "30";
                saw_old_lt_cmp |= has_mem && imm == "17";
            }
            saw_fma |= line.find("vfnmadd213pd") != std::string_view::npos;
            pos = end + 1;
        }

        SIMJIT_ASSERT(saw_fma);
        SIMJIT_ASSERT(saw_mem_cmp);
        SIMJIT_ASSERT(!saw_old_lt_cmp);
    });
#else
    (void)tests;
#endif
}

static std::vector<IntegrationTest> make_integration_tests() {
    std::vector<IntegrationTest> tests;
    add_pack_tests(tests);
    add_checked_op_shape_tests(tests);
    add_commutative_cse_tests(tests);
    add_const_div_quotient_cache_tests(tests);
    add_logical_peephole_shape_tests(tests);
    add_binary_identity_peephole_shape_tests(tests);
    add_conditional_identity_peephole_shape_tests(tests);
    add_vectorizer_heuristic_tests(tests);
    add_float_index_lowering_tests(tests);
    add_small_gather_scatter_index_tests(tests);
    add_llvm_emitter_format_tests(tests);
#if SIMJIT_LLVM_BACKEND
    add_llvm_session_execution_tests(tests);
#endif
    add_cpp_emitter_format_tests(tests);
    add_asmjit_x86_peephole_tests(tests);
    return tests;
}

int main() {
    std::vector<IntegrationTest> tests = make_integration_tests();
    size_t failures = 0;
    for (size_t i = 0; i < tests.size(); ++i) {
        try {
            tests[i]();
        } catch (const std::exception &e) {
            std::fprintf(stderr, "integration test %zu failed: %s\n", i, e.what());
            ++failures;
        } catch (...) {
            std::fprintf(stderr, "integration test %zu failed with an unknown exception\n", i);
            ++failures;
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "%zu of %zu integration tests failed\n", failures, tests.size());
        return 1;
    }
    std::printf("All %zu integration tests passed\n", tests.size());
    return 0;
}
