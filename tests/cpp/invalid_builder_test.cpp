// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

#include "simjit/compiler.h"

using namespace simjit;
using namespace simjit::types;

namespace {

static void assert_exception_metadata(const SimjitException &e, ErrorModule module, ErrorKind kind,
                                      ErrorSubKind subkind = ErrorSubKind::None) {
    SIMJIT_ASSERT(e.module() == module);
    SIMJIT_ASSERT(e.kind() == kind);
    SIMJIT_ASSERT(e.subkind() == subkind);
}

static void finish_valid_metadata_test(FunctionBuilder &b) {
    b.store(b.i32(1), b.arg(I32));
}

static void add_build_limit_tests(std::vector<Test> &tests) {
    add_unsupported(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        for (size_t i = 0; i < BuildLimits{}.max_hir_roots + 1; ++i) {
            b.store(x, b.arg(I32));
        }
    });
    add_unsupported(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        for (size_t i = 0; i < BuildLimits{}.max_hir_live_steps; ++i) {
            x = b.add(x, b.i32((int32_t)i + 1));
        }
        b.store(x, b.arg(I32));
    });

    tests.push_back(Test{[](FunctionBuilder &) {
                             MemoryArena arena;
                             Context ctx{arena};
                             ctx.build_limits.max_argument_count = 2;
                             FunctionBuilder limited{ctx};
                             (void)limited.arg(I32);
                             (void)limited.arg(I32);
                             (void)limited.arg(I32);
                         },
                         PASS_ALL,
                         {},
                         EXPECT_UNSUPPORTED,
                         relative_test_file(__builtin_FILE()),
                         __builtin_LINE()});
}

static void add_exception_metadata_tests(std::vector<Test> &tests) {
    add_valid(tests, [](FunctionBuilder &b) {
        try {
            MemoryArena arena;
            Context ctx{arena};
            FunctionBuilder local{ctx};
            local.arg_safety_check();
            local.arg_safety_check();
            SIMJIT_ASSERT(false);
        } catch (const SimjitException &e) {
            assert_exception_metadata(e, ErrorModule::HIR, ErrorKind::InvalidInput, ErrorSubKind::InvalidConfiguration);
        }
        finish_valid_metadata_test(b);
    });

    add_valid(tests, [](FunctionBuilder &b) {
        try {
            MemoryArena arena;
            Context ctx{arena};
            ctx.arch = Arch::Arm64_NEON;
            FunctionBuilder local{ctx};
            Value x = local.input_arg(I64);
            Value y = local.input_arg(I64);
            local.store(local.mul(x, y), local.arg(I64));
            hir::Function *fn = local.build();
            lower_vectorized(fn);
            SIMJIT_ASSERT(false);
        } catch (const SimjitException &e) {
            assert_exception_metadata(e, ErrorModule::Vectorizer, ErrorKind::VectorizationFailed,
                                      ErrorSubKind::UnsupportedSpecialOps);
        }
        finish_valid_metadata_test(b);
    });

    add_valid(tests, [](FunctionBuilder &b) {
        try {
            MemoryArena arena;
            Context ctx{arena};
            ctx.arch = Arch::Amd64_AVX512;
            FunctionBuilder local{ctx};
            Value x = local.input_arg(I32);
            for (size_t i = 0; i < 17; ++i) {
                local.store(x, local.arg(I32));
            }
            hir::Function *fn = local.build();
            lower_vectorized(fn);
            SIMJIT_ASSERT(false);
        } catch (const SimjitException &e) {
            assert_exception_metadata(e, ErrorModule::Vectorizer, ErrorKind::VectorizationFailed,
                                      ErrorSubKind::TooManyRoots);
        }
        finish_valid_metadata_test(b);
    });
}

static void add_lifecycle_tests(std::vector<Test> &tests) {
    add_invalid(tests, [](FunctionBuilder &b) {
        b.arg_safety_check();
        b.arg_safety_check();
    });
    add_invalid(tests, [](FunctionBuilder &b) { (void)b.arg(I32); });
    add_invalid(tests, [](FunctionBuilder &b) {
        (void)b.arg(I32);
        b.store(b.i32(1), b.arg(I32));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.add_checked(x, y), dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.negate_checked(x), dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I16);
        b.store(b.trunc_checked(x, I16), dst);
    });
}

static void add_argument_access_tests(std::vector<Test> &tests) {
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(x, dst);
        b.store(b.add(x, b.i32(1)), dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value idx = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(x, dst);
        b.scatter(x, idx, dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Argument dst = b.arg(I32);
        Value old = b.gather(idx, dst);
        b.store(old, dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate cond = b.input_predicate_arg();
        Argument dst = b.arg(I32);
        Argument size = b.arg(I64);
        b.pack(x, cond, dst, size);
        b.scatter(x, b.index(I32), dst);
    });
}

static void add_low_level_api_tests(std::vector<Test> &tests) {
    add_invalid(tests, [](FunctionBuilder &b) {
        Predicate x = b.input_predicate_arg();
        b.store(Value{x.step_}, b.arg(I1));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        b.store(Predicate{x.step_}, b.arg(I1));
    });
    add_invalid(tests, [](FunctionBuilder &b) { b.arith_agg(b.input_arg(I32), ArithBinaryOp::Div, b.arg(I32)); });
    add_invalid(tests, [](FunctionBuilder &b) {
        b.cond_arith_agg(b.input_arg(I32), b.input_predicate_arg(), ArithBinaryOp::RotateLeft, b.arg(I32));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        b.predicate_agg(b.input_predicate_arg(), PredicateBinaryOp::XNor, b.arg(I1));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.output_arg(b.arith_binary(x, y, ArithBinaryOp::Div, ArithBinaryOpFlags::SafetyCheck));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.output_arg(b.arith_binary(x, y, ArithBinaryOp::UMin, ArithBinaryOpFlags::SafetyCheck));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.output_arg(b.arith_binary(x, y, ArithBinaryOp::Add, ArithBinaryOpFlags::SafeDivision));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        ArithBinaryOpFlags flags = ArithBinaryOpFlags::SafetyCheck | ArithBinaryOpFlags::ShiftWraparound;
        b.output_arg(b.arith_binary(x, y, ArithBinaryOp::Add, flags));
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        ArithBinaryOpFlags flags = ArithBinaryOpFlags::SafeDivision | ArithBinaryOpFlags::ShiftWraparound;
        b.output_arg(b.arith_binary(x, y, ArithBinaryOp::Div, flags));
    });
    add_invalid(tests,
                [](FunctionBuilder &b) { b.output_arg(b.arith_unary(b.input_arg(I32), ArithUnaryOp::Not, true)); });
    add_invalid(tests,
                [](FunctionBuilder &b) { b.output_arg(b.int_cast(b.input_arg(I32), I64, IntCastKind::Sext, true)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.cmp_ugt(b.input_arg(F32), b.input_arg(F32))); });
    add_unsupported(tests, [](FunctionBuilder &b) { b.sum(b.input_arg(I32), b.arg(I128)); });
    add_unsupported(tests, [](FunctionBuilder &b) { b.product(b.input_arg(I64), b.arg(I128)); });
}

static void add_constant_tests(std::vector<Test> &tests) {
    struct ConstantCase {
        int64_t value;
        ScalarDataType dtype;
    };
    constexpr ConstantCase kCases[] = {
        {2, I1}, {-1, I1}, {-129, I8}, {-32769, I16}, {-2147483649LL, I32},
    };
    for (const auto &test_case : kCases) {
        add_invalid(tests, [test_case](FunctionBuilder &b) { b.output_arg(b.con(test_case.value, test_case.dtype)); });
    }
}

static void add_permute_index_tests(std::vector<Test> &tests) {
    add_invalid(tests,
                [](FunctionBuilder &b) { b.output_arg(b.permute_i64_i8(b.input_arg(I64), 8, 0, 0, 0, 0, 0, 0, 0)); });
    add_invalid(tests,
                [](FunctionBuilder &b) { b.output_arg(b.permute_i8_bits(b.input_arg(I8), 8, 0, 0, 0, 0, 0, 0, 0)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.replicate_ith_bit_i8(b.input_arg(I8), 8)); });
}

static void add_wrapper_tests(std::vector<Test> &tests) {
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.signed_cast(b.input_arg(I32), I32)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.unsigned_cast(b.input_arg(I32), I32)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.signed_cast(b.input_arg(I32), I1)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.unsigned_cast(b.input_arg(I32), I1)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.signed_cast(b.input_arg(I1), I32)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.unsigned_cast(b.input_arg(I1), I32)); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.log2_no_zero(b.input_arg(F32))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.log2(b.input_arg(I128))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.bit_floor(b.input_arg(F64))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.bit_ceil(b.input_arg(I1))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.has_single_bit(b.input_arg(F32))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.bit_test(b.input_arg(I32), b.input_arg(I64))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.bit_testn(b.input_arg(I16), b.input_arg(I32))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.copysign_no_zero(b.input_arg(I32), b.input_arg(I1))); });
    add_invalid(tests, [](FunctionBuilder &b) { b.output_arg(b.copysign(b.input_arg(I64), b.input_arg(I128))); });
}

static std::vector<Test> make_invalid_builder_tests() {
    std::vector<Test> tests;
    add_build_limit_tests(tests);
    add_exception_metadata_tests(tests);
    add_lifecycle_tests(tests);
    add_argument_access_tests(tests);
    add_low_level_api_tests(tests);
    add_constant_tests(tests);
    add_permute_index_tests(tests);
    add_wrapper_tests(tests);
    return tests;
}
} // namespace
std::vector<Test> invalid_builder_tests = make_invalid_builder_tests();
