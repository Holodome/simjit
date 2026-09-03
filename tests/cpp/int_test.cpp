// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;

std::vector<Test> int_tests{
    // constant zero
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.i32(0), x);
         },
         PASS_ALL, R"FOO(
def func(n, x): 
    for i in range(n):
        x[i] = 0;
        )FOO"},
    // constant non-zero
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.i32(100), x);
         },
         PASS_ALL, R"FOO(
def func(n, x): 
    for i in range(n):
        x[i] = 100;
        )FOO"},
    // I1 constant
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             b.store(b.false_(), x);
         },
         PASS_ALL, R"FOO(
def func(n, x): 
    for i in range(n):
        x[i] = 0;
        )FOO"},
    // out of range constant
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.con(0x1ffffffff, I32), x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // incompatible types
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.i8(0), x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.i16(0), x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.i64(0), x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             b.store(b.false_(), x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // var
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(y, x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(y, x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             b.store(y, x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(y, x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate y = b.input_predicate_arg();
             b.store(y, x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n//8):
        x[i] = y[i]
        )FOO"},
    // load_splat
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Argument y = b.arg(I32);
             b.store(b.load_splat(y), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[0]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Argument y = b.arg(I64);
             b.store(b.load_splat(y), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[0]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Argument y = b.arg(I8);
             b.store(b.load_splat(y), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[0]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Argument y = b.arg(I16);
             b.store(b.load_splat(y), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[0]
        )FOO"},
    // unaligned var
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Argument y = b.arg(I32);
             b.store(b.load(y, LoadStoreKind::Unaligned), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Argument y = b.arg(I64);
             b.store(b.load(y, LoadStoreKind::Unaligned), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Argument y = b.arg(I8);
             b.store(b.load(y, LoadStoreKind::Unaligned), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Argument y = b.arg(I16);
             b.store(b.load(y, LoadStoreKind::Unaligned), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Argument y = b.arg(I1);
             b.store(b.load_predicate(y), x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n//8):
        x[i] = y[i]
        )FOO"},
    // unaligned store
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(y, x, LoadStoreKind::Unaligned);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(y, x, LoadStoreKind::Unaligned);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             b.store(y, x, LoadStoreKind::Unaligned);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(y, x, LoadStoreKind::Unaligned);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate y = b.input_predicate_arg();
             b.store(y, x);
         },
         PASS_ALL, R"FOO(
def func(n, x, y): 
    for i in range(n//8):
        x[i] = y[i]
        )FOO"},
    // add
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.add(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] + z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.add(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] + z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.add(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] + z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.add(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] + z[i]
        )FOO"},
    // sub
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.sub(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] - z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.sub(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] - z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.sub(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] - z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.sub(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] - z[i]
        )FOO"},
    // mul
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] * z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_splat_arg(I8);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] * z[0]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value product = b.mul(y, z);
             Value tmp = b.select(b.cmp_gt(y, z), product, b.i8(127));
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] * z[i] if y[i] > z[i] else 127
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] * z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] * z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] * z[i]
        )FOO"},
    // div
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.div(y, z);
             b.store(tmp, x);
         },
         test_meta()
             .limitation(TestVariant::VectorAll)
             .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::UnsupportedSpecialOps)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        invalid = z[i] == 0 or (y[i] == -128 and z[i] == -1)
        zz = 1 if invalid else z[i]
        x[i] = int(y[i] / zz)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.div(y, z);
             b.store(tmp, x);
         },
         test_meta()
             .limitation(TestVariant::VectorAll)
             .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::UnsupportedSpecialOps)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        invalid = z[i] == 0 or (y[i] == -32768 and z[i] == -1)
        zz = 1 if invalid else z[i]
        x[i] = int(y[i] / zz)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.div(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        invalid = z[i] == 0 or (y[i] == -2147483648 and z[i] == -1)
        zz = 1 if invalid else z[i]
        x[i] = int(y[i] / zz)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.div(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    // mod
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.mod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        invalid = z[i] == 0 or (y[i] == -128 and z[i] == -1)
        zz = 1 if invalid else z[i]
        q = int(y[i] / zz)
        x[i] = y[i] - q * zz
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.mod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        invalid = z[i] == 0 or (y[i] == -32768 and z[i] == -1)
        zz = 1 if invalid else z[i]
        q = int(y[i] / zz)
        x[i] = y[i] - q * zz
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.mod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        invalid = z[i] == 0 or (y[i] == -2147483648 and z[i] == -1)
        zz = 1 if invalid else z[i]
        q = int(y[i] / zz)
        x[i] = y[i] - q * zz
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.mod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    // udiv
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.udiv(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.udiv(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.udiv(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.udiv(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    // umod
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.umod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.umod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.umod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.umod(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    // safe division edge cases
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.div(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.udiv(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value lhs = b.select(b.cmp_eq(x, x), b.i32(INT32_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i32(1));
             b.store(b.div(lhs, rhs), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value lhs = b.select(b.cmp_eq(x, x), b.i32(INT32_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i32(1));
             b.store(b.mod(lhs, rhs), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Value splat = b.load_splat(b.arg(I64));
             Value gathered = b.gather(b.index(I32), b.arg(I64));
             Value threshold = b.mod(b.i64(INT64_MIN), b.i64(-1));
             Predicate cond = b.cmp_ge(b.and_(splat, gathered), threshold);
             b.min_agg_if(b.popcnt(b.i16(INT16_MIN)), cond, b.arg(I16));
             b.and_agg(b.false_(), b.arg(I1));
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.arith_binary(x, zero, ArithBinaryOp::Div, ArithBinaryOpFlags::SafetyCheck), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value lhs = b.select(b.cmp_eq(x, x), b.i32(INT32_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i32(1));
             auto flags = ArithBinaryOpFlags::SafeDivision | ArithBinaryOpFlags::SafetyCheck;
             b.store(b.arith_binary(lhs, rhs, ArithBinaryOp::Mod, flags), dst);
         },
         ONLY_SCALAR},
    // checked division wrappers
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        b.store(b.div_checked(x, b.i32(7)), dst);
    }},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.div_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I64);
             Value x = b.input_arg(I64);
             Value zero = b.sub(x, x);
             b.store(b.div_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value lhs = b.select(b.cmp_eq(x, x), b.i32(INT32_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i32(1));
             b.store(b.div_checked(lhs, rhs), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I64);
             Value x = b.input_arg(I64);
             Value lhs = b.select(b.cmp_eq(x, x), b.i64(INT64_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i64(1));
             b.store(b.div_checked(lhs, rhs), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.mod_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I64);
             Value x = b.input_arg(I64);
             Value zero = b.sub(x, x);
             b.store(b.mod_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value lhs = b.select(b.cmp_eq(x, x), b.i32(INT32_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i32(1));
             b.store(b.mod_checked(lhs, rhs), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I64);
             Value x = b.input_arg(I64);
             Value lhs = b.select(b.cmp_eq(x, x), b.i64(INT64_MIN), x);
             Value rhs = b.sub(b.sub(x, x), b.i64(1));
             b.store(b.mod_checked(lhs, rhs), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.udiv_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I64);
             Value x = b.input_arg(I64);
             Value zero = b.sub(x, x);
             b.store(b.udiv_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I32);
             Value x = b.input_arg(I32);
             Value zero = b.sub(x, x);
             b.store(b.umod_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I64);
             Value x = b.input_arg(I64);
             Value zero = b.sub(x, x);
             b.store(b.umod_checked(x, zero), dst);
         },
         ONLY_SCALAR},
    // and
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.and_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.and_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.and_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.and_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] & z[i]
        )FOO"},
    // or
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.or_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] | z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.or_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] | z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.or_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] | z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.or_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] | z[i]
        )FOO"},
    // xor
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.xor_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] ^ z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.xor_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] ^ z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.xor_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] ^ z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.xor_(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] ^ z[i]
        )FOO"},
    // min
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.min(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = min(y[i], z[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.min(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = min(y[i], z[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.min(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = min(y[i], z[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.min(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = min(y[i], z[i])
        )FOO"},
    // max
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.max(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = max(y[i], z[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.max(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = max(y[i], z[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.max(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = max(y[i], z[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.max(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = max(y[i], z[i])
        )FOO"},
    // unsigned min
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.umin(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.umin(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.umin(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.umin(y, z);
        b.store(tmp, x);
    }},
    // unsigned max
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.umax(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.umax(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.umax(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.umax(y, z);
        b.store(tmp, x);
    }},
    // andnot
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.andnot(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = ~y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.andnot(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = ~y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value tmp = b.andnot(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = ~y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value tmp = b.andnot(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = ~y[i] & z[i]
        )FOO"},
    // sll
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value amount = b.and_(z, b.i8(7));
             Value tmp = b.sll(y, amount);
             b.store(tmp, x);
         },
         {},
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] << (z[i] & 7)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value amount = b.and_(z, b.i16(15));
             Value tmp = b.sll(y, amount);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] << (z[i] & 15)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value amount = b.and_(z, b.i32(31));
             Value tmp = b.sll(y, amount);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] << (z[i] & 31)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value amount = b.and_(z, b.i64(63));
             Value tmp = b.sll(y, amount);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] << (z[i] & 63)
        )FOO"},
    // srl
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value amount = b.and_(z, b.i8(7));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value amount = b.and_(z, b.i16(15));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Value amount = b.input_arg(I16);
        b.store(b.srl(x, amount), b.arg(I16));
        b.store(b.max(x, b.i16(0)), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value amount = b.and_(z, b.i32(31));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value amount = b.and_(z, b.i64(63));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    // sra
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value amount = b.and_(z, b.i8(7));
             Value tmp = b.sra(y, amount);
             b.store(tmp, x);
         },
         {},
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] >> (z[i] & 7)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value amount = b.and_(z, b.i16(15));
             Value tmp = b.sra(y, amount);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] >> (z[i] & 15)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value amount = b.and_(z, b.i32(31));
             Value tmp = b.sra(y, amount);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] >> (z[i] & 31)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value amount = b.and_(z, b.i64(63));
             Value tmp = b.sra(y, amount);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n):
        x[i] = y[i] >> (z[i] & 63)
        )FOO"},
    // rol
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.rotl(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.rotl(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.rotl(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.rotl(y, z);
        b.store(tmp, x);
    }},
    // ror
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.input_arg(I8);
             Value tmp = b.rotr(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value tmp = b.rotr(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.rotr(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.rotr(y, z);
        b.store(tmp, x);
    }},
    // index-driven variable shifts and rotates
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value amount = b.trunc(b.index(I32), I8);
        amount = b.and_(amount, b.i8(7));
        Value tmp = b.sll(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value amount = b.trunc(b.index(I32), I16);
        amount = b.and_(amount, b.i16(15));
        Value tmp = b.sll(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value amount = b.index(I32);
        amount = b.and_(amount, b.i32(31));
        Value tmp = b.sll(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value amount = b.index(I64);
        amount = b.and_(amount, b.i64(63));
        Value tmp = b.sll(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value amount = b.trunc(b.index(I32), I8);
        amount = b.and_(amount, b.i8(7));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value amount = b.trunc(b.index(I32), I16);
        amount = b.and_(amount, b.i16(15));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value amount = b.index(I32);
        amount = b.and_(amount, b.i32(31));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value amount = b.index(I64);
        amount = b.and_(amount, b.i64(63));
        Value tmp = b.srl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value amount = b.trunc(b.index(I32), I8);
        amount = b.and_(amount, b.i8(7));
        Value tmp = b.sra(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
             Argument src = b.arg(I8);
             Value amount = b.input_arg(I8);
             Argument dst = b.arg(I8);
             Value tmp = b.sra(b.load_splat(src), amount);
             b.scatter(tmp, b.index(I32), dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, src, amount, dst):
    for i in range(n):
        dst[i] = src[0] >> (amount[i] & 7)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value idx = b.index(I32);
             Value shifted = b.sra(b.input_arg(I16), b.input_arg(I16));
             Value splat = b.srl(b.load_splat(b.arg(I16)), b.load_splat(b.arg(I16)));
             Value mixed = b.xor_(shifted, splat);
             Argument dst = b.arg(I16);
             b.scatter(b.lzcnt(b.umod(b.i16(2), mixed)), idx, dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value amount = b.trunc(b.index(I32), I16);
        amount = b.and_(amount, b.i16(15));
        Value tmp = b.sra(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value amount = b.index(I32);
        amount = b.and_(amount, b.i32(31));
        Value tmp = b.sra(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value amount = b.index(I64);
        amount = b.and_(amount, b.i64(63));
        Value tmp = b.sra(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value amount = b.trunc(b.index(I32), I8);
             amount = b.and_(amount, b.i8(7));
             Value tmp = b.rotl(y, amount);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value amount = b.trunc(b.index(I32), I16);
             amount = b.and_(amount, b.i16(15));
             Value tmp = b.rotl(y, amount);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value amount = b.index(I32);
        amount = b.and_(amount, b.i32(31));
        Value tmp = b.rotl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value amount = b.index(I64);
        amount = b.and_(amount, b.i64(63));
        Value tmp = b.rotl(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value amount = b.trunc(b.index(I32), I8);
             amount = b.and_(amount, b.i8(7));
             Value tmp = b.rotr(y, amount);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value amount = b.trunc(b.index(I32), I16);
             amount = b.and_(amount, b.i16(15));
             Value tmp = b.rotr(y, amount);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value amount = b.index(I32);
        amount = b.and_(amount, b.i32(31));
        Value tmp = b.rotr(y, amount);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value amount = b.index(I64);
        amount = b.and_(amount, b.i64(63));
        Value tmp = b.rotr(y, amount);
        b.store(tmp, x);
    }},
    // sll const
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.i8(4);
             Value tmp = b.sll(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] << 4
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.i16(8);
             Value tmp = b.sll(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] << 8
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(16);
             Value tmp = b.sll(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] << 16
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.i64(16);
             Value tmp = b.sll(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] << 16
        )FOO"},
    // srl const
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value z = b.i8(4);
        Value tmp = b.srl(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value z = b.i16(8);
        Value tmp = b.srl(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.i32(16);
        Value tmp = b.srl(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.i64(16);
        Value tmp = b.srl(y, z);
        b.store(tmp, x);
    }},
    // sra const
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.i8(4);
             Value tmp = b.sra(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] >> 4
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.i16(8);
             Value tmp = b.sra(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] >> 8
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(16);
             Value tmp = b.sra(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] >> 16
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.i64(16);
             Value tmp = b.sra(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] >> 16
        )FOO"},
    // rol const
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value z = b.i8(3);
             Value tmp = b.rotl(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.i16(3);
             Value tmp = b.rotl(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.i32(16);
        Value tmp = b.rotl(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.i64(16);
        Value tmp = b.rotl(y, z);
        b.store(tmp, x);
    }},
    // ror const
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value z = b.i8(3);
        Value tmp = b.rotr(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.i16(7);
             Value tmp = b.rotr(y, z);
             b.store(tmp, x);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.i32(16);
        Value tmp = b.rotr(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.i64(16);
        Value tmp = b.rotr(y, z);
        b.store(tmp, x);
    }},
    // unary minus
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value tmp = b.negate(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = -y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value tmp = b.negate(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = -y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value tmp = b.negate(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = -y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value tmp = b.negate(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = -y[i]
        )FOO"},
    // bitwise not
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value tmp = b.not_(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = ~y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value tmp = b.not_(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = ~y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value tmp = b.not_(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = ~y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value tmp = b.not_(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = ~y[i]
        )FOO"},
    // lzcnt
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value tmp = b.lzcnt(y);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value tmp = b.lzcnt(y);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value tmp = b.lzcnt(y);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value tmp = b.lzcnt(y);
        b.store(tmp, x);
    }},
    // tzcnt
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        b.store(b.tzcnt(y), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        b.store(b.tzcnt(y), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        b.store(b.tzcnt(y), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        b.store(b.tzcnt(y), x);
    }},
    Test{[](FunctionBuilder &b) {
        b.scalar_only();

        Argument cmp_left = b.arg(I32);
        Argument cmp_right = b.arg(I32);
        Argument shift_src = b.arg(I8);
        Argument shift_amount = b.arg(I8);
        Argument div_left = b.arg(I8);
        Argument div_right = b.arg(I8);
        Argument store_dst = b.arg(I8);
        Argument count_dst = b.arg(I64);
        Argument min_dst = b.arg(I64);

        Predicate cmp = b.cmp_ugt(b.load(cmp_left, LoadStoreKind::Aligned), b.load(cmp_right));
        Predicate select_shift = b.andnot(b.true_(), cmp);
        Value shifted = b.sll(b.load(shift_src), b.load_splat(shift_amount));
        Value divided = b.div(b.load_splat(div_left), b.load_splat(div_right));
        Value selected = b.select(select_shift, shifted, b.max(b.i8(-1), divided));
        b.store(b.lzcnt(selected), store_dst, LoadStoreKind::Aligned);
        b.countif(b.false_(), count_dst);
        b.min_agg(b.index(I64), min_dst);
    }},
    // popcnt
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Value tmp = b.popcnt(y);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Value tmp = b.popcnt(y);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value tmp = b.popcnt(y);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Value tmp = b.popcnt(y);
        b.store(tmp, x);
    }},
    // abs
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value tmp = b.abs(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = abs(y[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value tmp = b.abs(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = abs(y[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value tmp = b.abs(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = abs(y[i])
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value tmp = b.abs(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = abs(y[i])
        )FOO"},
    // int_cast i8 -> i8
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Value tmp = b.sext(y, I8);
             b.store(tmp, x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // int_cast i8 -> i16
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I8);
             Value tmp = b.sext(y, I16);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I8);
        Value tmp = b.zext(y, I16);
        b.store(tmp, x);
    }},
    // int_cast i8 -> i32
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I8);
             Value tmp = b.sext(y, I32);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I8);
        Value tmp = b.zext(y, I32);
        b.store(tmp, x);
    }},
    // int_cast i8 -> i64
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I8);
        Value tmp = b.sext(y, I64);
        b.store(tmp, x);
    }},
    // int_cast i16 -> i8
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I16);
             Value tmp = b.trunc(y, I8);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    // int_cast i16 -> i16
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value tmp = b.sext(y, I16);
             b.store(tmp, x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // int_cast i16 -> i32
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I16);
             Value tmp = b.sext(y, I32);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I16);
        Value tmp = b.zext(y, I32);
        b.store(tmp, x);
    }},
    // int_cast i16 -> i64
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I16);
             Value tmp = b.sext(y, I64);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I16);
        Value tmp = b.zext(y, I64);
        b.store(tmp, x);
    }},

    // int_cast i32 -> i8
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I32);
             Value tmp = b.trunc(y, I8);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    // int_cast i32 -> i16
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I32);
        Value tmp = b.trunc(y, I16);
        b.store(tmp, x);
    }},
    // int_cast i32 -> i32
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value tmp = b.trunc(y, I32);
             b.store(tmp, x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // int_cast i32 -> i64
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I32);
             Value tmp = b.sext(y, I64);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I32);
        Value tmp = b.zext(y, I64);
        b.store(tmp, x);
    }},
    // int_cast i64 -> i8
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I64);
        Value tmp = b.trunc(y, I8);
        b.store(tmp, x);
    }},
    // int_cast i64 -> i16
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I64);
        Value tmp = b.trunc(y, I16);
        b.store(tmp, x);
    }},
    // int_cast i64 -> i32
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I64);
        Value tmp = b.trunc(y, I32);
        b.store(tmp, x);
    }},
    // int_cast i64 -> i64
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value tmp = b.trunc(y, I64);
             b.store(tmp, x);
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // chained trunc casts
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I64);
        Value y32 = b.trunc(y, I32);
        Value y16 = b.trunc(y32, I16);
        Value y8 = b.trunc(y16, I8);
        b.store(y8, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I32);
        Value y32 = b.trunc(y, I32);
        Value sum32 = b.add(y32, z);
        Value sum16 = b.trunc(sum32, I16);
        Value mixed16 = b.xor_(sum16, b.i16(0x55aa));
        Value result = b.trunc(mixed16, I8);
        b.store(result, x);
    }},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I8);
             Value narrow = b.input_arg(I8);
             for (size_t i = 0; i < 18; ++i) {
                 Value fallback = b.input_arg(I64);
                 Predicate cond = b.cmp_eq(fallback, b.i64((int64_t)i));
                 Value wide = b.select(cond, b.sext(narrow, I64), fallback);
                 narrow = b.trunc(wide, I8);
             }
             b.store(narrow, dst);
         },
         test_meta()
             .limitation(TestVariant::VectorAll)
             .vectorization_failure(TestVariant::VectorAll, simjit::ErrorSubKind::GraphCoefficientLimitExceeded)},
    // truncate after extend
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I16);
        Value wide = b.sext(y, I64);
        b.store(b.trunc(wide, I32), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I16);
        Value wide = b.zext(y, I64);
        b.store(b.trunc(wide, I32), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I32);
        Value wide = b.sext(y, I64);
        b.store(b.trunc(wide, I8), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I32);
        Value wide = b.zext(y, I64);
        b.store(b.trunc(wide, I8), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value wide = b.sext(y, I64);
        b.store(b.trunc(wide, I32), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Value wide = b.zext(y, I64);
        b.store(b.trunc(wide, I32), x);
    }},
    // chained extends
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I8);
        Value y16 = b.sext(y, I16);
        Value y32 = b.sext(y16, I32);
        b.store(b.sext(y32, I64), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I8);
        Value y16 = b.zext(y, I16);
        Value y32 = b.zext(y16, I32);
        b.store(b.zext(y32, I64), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I8);
        Value y16 = b.zext(y, I16);
        Value y32 = b.sext(y16, I32);
        b.store(b.sext(y32, I64), x);
    }},
    // upcast normalization with multiple subjects under one root
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        b.store(b.add(b.sext(y, I16), b.sext(z, I16)), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        b.store(b.add(b.sext(y, I64), b.sext(z, I64)), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.sext(b.input_arg(I8), I64);
        Predicate positive = b.cmp_gt(y, b.i64(0));
        b.cond_store(y, positive, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.sext(b.trunc(b.input_arg(I32), I8), I16);
        b.store(b.add(y, b.i16(1)), x);
    }},
    // different trunc widths in the same value expression
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I32);
        Value y32 = b.trunc(y, I32);
        Value z8 = b.trunc(z, I8);
        Value z32 = b.sext(z8, I32);
        b.store(b.add(y32, z32), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I32);
        Value y16 = b.trunc(y, I16);
        Value z8 = b.trunc(z, I8);
        Value y32 = b.sext(y16, I32);
        Value z32 = b.zext(z8, I32);
        b.store(b.sub(y32, z32), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I32);
        Value y32 = b.trunc(y, I32);
        Value z8 = b.trunc(z, I8);
        Value z32 = b.sext(z8, I32);
        Value mixed = b.xor_(y32, z32);
        b.store(b.trunc(mixed, I16), x);
    }},
    // cmp less
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    // cmp greater
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    // cmp less equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    // cmp greater equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    // cmp equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    // cmp not equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    // unsigned cmp less
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_ult(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_ult(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        b.scalar_only();
        Argument x8 = b.arg(I1);
        Argument x16 = b.arg(I1);
        Value y8 = b.input_arg(I8);
        Value z8 = b.input_arg(I8);
        Value y16 = b.input_arg(I16);
        Value z16 = b.input_arg(I16);
        b.store(b.cmp_ult(y8, z8), x8);
        b.store(b.cmp_ugt(y16, z16), x16);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ult(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_ult(y, z);
        b.store(tmp, x);
    }},
    // unsigned cmp greater
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_ugt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_ugt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ugt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_ugt(y, z);
        b.store(tmp, x);
    }},
    // unsigned cmp less equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_ule(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_ule(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ule(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_ule(y, z);
        b.store(tmp, x);
    }},
    // unsigned cmp greater equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_uge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_uge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_uge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_uge(y, z);
        b.store(tmp, x);
    }},
    // unsigned cmp equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_ueq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_ueq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ueq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_ueq(y, z);
        b.store(tmp, x);
    }},
    // unsigned cmp not equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Predicate tmp = b.cmp_une(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Predicate tmp = b.cmp_une(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_une(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Predicate tmp = b.cmp_une(y, z);
        b.store(tmp, x);
    }},
    // cmp less zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.i8(0);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.i16(0);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.i32(0);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.i64(0);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    // cmp greater zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.i8(0);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.i16(0);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.i32(0);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.i64(0);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    // cmp less equal zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.i8(0);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.i16(0);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.i32(0);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.i64(0);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    // cmp greater equal zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.i8(0);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.i16(0);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.i32(0);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.i64(0);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    // cmp equal zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.i8(0);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.i16(0);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.i32(0);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.i64(0);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    // cmp not equal zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I8);
        Value z = b.i8(0);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I16);
        Value z = b.i16(0);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.i32(0);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I64);
        Value z = b.i64(0);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    // Converting to bool
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I8);
             Predicate tmp = b.cmp_gt(y, b.i8(0));
             Value bool_ = b.select(tmp, b.i8(1), b.i8(0));
             b.store(bool_, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] > 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I16);
             Predicate tmp = b.is_positive(y);
             Value bool_ = b.select(tmp, b.i8(1), b.i8(0));
             b.store(bool_, x);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] > 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I32);
             Predicate tmp = b.is_positive(y);
             Value bool_ = b.select(tmp, b.i8(1), b.i8(0));
             b.store(bool_, x);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] > 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value y = b.input_arg(I64);
             Predicate tmp = b.is_positive(y);
             Value bool_ = b.select(tmp, b.i8(1), b.i8(0));
             b.store(bool_, x);
         },
         coefficient_range_limit(TestVariant::VectorAll)},
    // unary not
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate y = b.input_predicate_arg();
             Predicate tmp = b.not_(y);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n//8):
        x[i] = ~y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_gt(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not cmp less
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_lt(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not cmp greater
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_gt(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not cmp less equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_le(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not cmp greater equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ge(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not cmp equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_eq(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not cmp not equal
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate tmp = b.cmp_ne(y, z);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // logical binary and
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate w = b.input_predicate_arg();
        Predicate left = b.cmp_lt(y, z);
        Predicate tmp = b.and_(left, w);
        b.store(tmp, x);
    }},
    // logical binary or
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate w = b.input_predicate_arg();
        Predicate left = b.cmp_lt(y, z);
        Predicate tmp = b.or_(left, w);
        b.store(tmp, x);
    }},
    // logical binary xor
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.xor_(left, right);
        b.store(tmp, x);
    }},
    // logical binary xnor
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.xnor(left, right);
        b.store(tmp, x);
    }},
    // logical binary andnot
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.andnot(left, right);
        b.store(tmp, x);
    }},
    // not logical binary and
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.and_(left, right);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not logical binary or
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.or_(left, right);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not logical binary xor
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.xor_(left, right);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not logical binary xnor
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.xnor(left, right);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // not logical binary andnot
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value w = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_lt(w, a);
        Predicate tmp = b.andnot(left, right);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // Generate andnot from not and
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate w = b.input_predicate_arg();
        Predicate left = b.cmp_lt(y, z);
        Predicate tmp = b.and_(left, w);
        tmp = b.not_(tmp);
        b.store(tmp, x);
    }},
    // graph vectorizer: bitpacked predicate and i32 compare should agree on mask width
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Predicate mask = b.not_(b.input_predicate_arg());
        Value y = b.input_arg(I32);
        Predicate cmp = b.cmp_eq(y, b.i32(0));
        b.store(b.and_(mask, cmp), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Predicate mask = b.not_(b.input_predicate_arg());
        Value y = b.input_arg(I32);
        Predicate cmp = b.cmp_eq(y, b.i32(0));
        b.store(b.or_(mask, cmp), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Predicate mask = b.not_(b.input_predicate_arg());
        Value y = b.input_arg(I32);
        Predicate cmp = b.cmp_eq(y, b.i32(0));
        b.store(b.andnot(mask, cmp), x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Predicate mask = b.not_(b.input_predicate_arg());
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Predicate cmp = b.cmp_gt(y, b.i32(0));
        Predicate cond = b.andnot(mask, cmp);
        b.store(b.select(cond, y, z), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Predicate mask = b.not_(b.input_predicate_arg());
        Value y = b.input_arg(I32);
        Predicate cmp = b.cmp_gt(y, b.i32(0));
        Predicate cond = b.andnot(mask, cmp);
        b.cond_store(b.add(y, b.i32(1)), cond, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst8 = b.arg(I8);
        Argument dst64 = b.arg(I64);
        b.store(b.input_arg(I8), dst8);
        b.store(b.input_arg(I64), dst64);
    }},
    // independent roots can have different coefficient ranges.
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I8);
        Value wide = b.input_arg(I16);
        b.store(b.sext(narrow, I16), b.arg(I16));
        b.store(b.trunc(wide, I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I8);
        Value wide = b.input_arg(I32);
        b.store(b.sext(narrow, I32), b.arg(I32));
        b.store(b.trunc(wide, I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I8);
        Value wide = b.input_arg(I64);
        b.store(b.sext(narrow, I64), b.arg(I64));
        b.store(b.trunc(wide, I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I16);
        Value wide = b.input_arg(I32);
        b.store(b.sext(narrow, I32), b.arg(I32));
        b.store(b.trunc(wide, I16), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I16);
        Value wide = b.input_arg(I64);
        b.store(b.sext(narrow, I64), b.arg(I64));
        b.store(b.trunc(wide, I16), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I32);
        Value wide = b.input_arg(I64);
        b.store(b.sext(narrow, I64), b.arg(I64));
        b.store(b.trunc(wide, I32), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I8);
        Value wide = b.input_arg(I16);
        Value mid = b.input_arg(I32);
        Predicate pure_mask = b.and_(b.input_predicate_arg(), b.not_(b.input_predicate_arg()));
        Predicate carried_mask = b.cmp_gt(mid, b.i32(0));
        b.store(b.sext(narrow, I16), b.arg(I16));
        b.store(b.trunc(wide, I8), b.arg(I8));
        b.store(pure_mask, b.arg(I1));
        b.store(carried_mask, b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I8);
        Value wide = b.input_arg(I32);
        Value mid = b.input_arg(I32);
        Predicate pure_mask = b.and_(b.input_predicate_arg(), b.not_(b.input_predicate_arg()));
        Predicate carried_mask = b.cmp_gt(mid, b.i32(0));
        b.store(b.sext(narrow, I32), b.arg(I32));
        b.store(b.trunc(wide, I8), b.arg(I8));
        b.store(pure_mask, b.arg(I1));
        b.store(carried_mask, b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I8);
        Value wide = b.input_arg(I64);
        Value mid = b.input_arg(I32);
        Predicate pure_mask = b.and_(b.input_predicate_arg(), b.not_(b.input_predicate_arg()));
        Predicate carried_mask = b.cmp_gt(mid, b.i32(0));
        b.store(b.sext(narrow, I64), b.arg(I64));
        b.store(b.trunc(wide, I8), b.arg(I8));
        b.store(pure_mask, b.arg(I1));
        b.store(carried_mask, b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I16);
        Value wide = b.input_arg(I32);
        Value mid = b.input_arg(I32);
        Predicate pure_mask = b.and_(b.input_predicate_arg(), b.not_(b.input_predicate_arg()));
        Predicate carried_mask = b.cmp_gt(mid, b.i32(0));
        b.store(b.sext(narrow, I32), b.arg(I32));
        b.store(b.trunc(wide, I16), b.arg(I16));
        b.store(pure_mask, b.arg(I1));
        b.store(carried_mask, b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I16);
        Value wide = b.input_arg(I64);
        Value mid = b.input_arg(I32);
        Predicate pure_mask = b.and_(b.input_predicate_arg(), b.not_(b.input_predicate_arg()));
        Predicate carried_mask = b.cmp_gt(mid, b.i32(0));
        b.store(b.sext(narrow, I64), b.arg(I64));
        b.store(b.trunc(wide, I16), b.arg(I16));
        b.store(pure_mask, b.arg(I1));
        b.store(carried_mask, b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        Value narrow = b.input_arg(I32);
        Value wide = b.input_arg(I64);
        Value mid = b.input_arg(I32);
        Predicate pure_mask = b.and_(b.input_predicate_arg(), b.not_(b.input_predicate_arg()));
        Predicate carried_mask = b.cmp_gt(mid, b.i32(0));
        b.store(b.sext(narrow, I64), b.arg(I64));
        b.store(b.trunc(wide, I32), b.arg(I32));
        b.store(pure_mask, b.arg(I1));
        b.store(carried_mask, b.arg(I1));
    }},
    // mul peepholes
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(1);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 1
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(2);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 2
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(3);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 3
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(5);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 5
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(6);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 6
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.i64(11);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 11
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(132);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 132
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(133);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 133
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(124);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 124
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(255);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 255
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(257);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 257
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(258);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 258
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(130);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 130
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(126);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 126
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(176);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * 176
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-1);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -1
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-2);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -2
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-132);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -132
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-128);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -128
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-4);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -4
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-5);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -5
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.i32(-6);
             Value tmp = b.mul(y, z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n):
        x[i] = y[i] * -6
        )FOO"},
    // de-morgan !a && !b
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate y = b.input_predicate_arg();
             Predicate z = b.input_predicate_arg();
             Predicate tmp = b.and_(b.not_(y), b.not_(z));
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n//8):
        x[i] = ~(y[i] | z[i])
        )FOO"},
    // andnot conversion
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate y = b.input_predicate_arg();
             Predicate z = b.input_predicate_arg();
             Predicate tmp = b.and_(b.not_(y), z);
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n//8):
        x[i] = ~y[i] & z[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate tmp = b.andnot(b.false_(), b.true_());
             b.store(tmp, x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x):
    for i in range(n//8):
        x[i] = 0xff
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Predicate y = b.input_predicate_arg();
             Predicate z = b.input_predicate_arg();
             Predicate tmp = b.and_(y, b.not_(z));
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    for i in range(n//8):
        x[i] = y[i] & ~z[i]
        )FOO"},
    // not not
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I1);
             Predicate cond = b.input_predicate_arg();
             Predicate tmp = b.not_(b.not_(cond));
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y): 
    for i in range(n//8):
        x[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Predicate cond = b.input_predicate_arg();
        Predicate tmp = b.not_(cond);
        b.store(tmp, dst);
    }},
    // between a b
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_gt(y, z);
        Predicate right = b.cmp_lt(y, a);
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_lt(y, z);
        Predicate right = b.cmp_gt(y, a);
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_ge(y, z);
        Predicate right = b.cmp_le(y, a);
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_gt(y, z);
        Predicate right = b.cmp_le(y, a);
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    // between a b const
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Predicate left = b.cmp_gt(y, b.i32(100));
        Predicate right = b.cmp_lt(y, b.i32(120));
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Predicate left = b.cmp_gt(y, b.i32(100));
        Predicate right = b.cmp_le(y, b.i32(120));
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Predicate left = b.cmp_ge(y, b.i32(100));
        Predicate right = b.cmp_lt(y, b.i32(120));
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Predicate left = b.cmp_ge(y, b.i32(100));
        Predicate right = b.cmp_le(y, b.i32(120));
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    // between zero
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_gt(y, b.i32(0));
        Predicate right = b.cmp_lt(y, a);
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Value a = b.input_arg(I32);
        Predicate left = b.cmp_ge(y, b.i32(0));
        Predicate right = b.cmp_lt(y, a);
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Predicate left = b.cmp_gt(y, b.i32(0));
        Predicate right = b.cmp_lt(y, b.i32(120));
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(I32);
        Predicate left = b.cmp_ge(y, b.i32(0));
        Predicate right = b.cmp_lt(y, b.i32(120));
        Predicate tmp = b.and_(left, right);
        b.store(tmp, x);
    }},
    // permute i8
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        Value tmp = b.permute_i16_i8(x, 1, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        Value tmp = b.permute_i16_i8(x, 0, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        Value tmp = b.permute_i16_i8(x, 1, 1);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        Value tmp = b.permute_i32_i8(x, 1, 0, 3, 2);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        Value tmp = b.permute_i32_i8(x, 3, 2, 1, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i8(x, 7, 6, 5, 4, 3, 2, 1, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i8(x, 7, 6, 2, 4, 0, 3, 1, 4);
        b.store(tmp, dst);
    }},
    // permute i16
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        Value tmp = b.permute_i32_i16(x, 1, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i16(x, 1, 0, 2, 3);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i16(x, 3, 2, 1, 0);
        b.store(tmp, dst);
    }},
    // permute i32
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i32(x, 0, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i32(x, 0, 1);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i32(x, 1, 0);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.permute_i64_i32(x, 1, 1);
        b.store(tmp, dst);
    }},
    // permute bits
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value tmp = b.permute_i8_bits(x, 0, 1, 2, 3, 4, 5, 6, 7);
             b.store(tmp, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value tmp = b.permute_i8_bits(x, 1, 2, 3, 4, 7, 6, 5, 0);
             b.store(tmp, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // {
    //     [](FunctionBuilder &b) {
    //         Argument x = b.arg(I8);
    //         Argument dst = b.arg(I8);
    //         Value tmp = b.reverse_bits_i8(x);
    //         b.store(tmp, dst);
    //     },
    // },
    // {
    //     [](FunctionBuilder &b) {
    //         Argument x = b.arg(I64);
    //         Argument dst = b.arg(I64);
    //         Value tmp = b.reverse_bits_i8(b.permute_i64_i8(x, 7, 6, 5, 4, 3, 2, 1, 0));
    //         b.store(tmp, dst);
    //     },
    // },
    // gather 32
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I8);
             Argument dst = b.arg(I32);
             b.store(b.gather(idx, x), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst):
    for i in range(n):
        dst[i] = x[idx[i] & 0xff]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I16);
             Argument dst = b.arg(I32);
             b.store(b.gather(idx, x), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst):
    for i in range(n):
        dst[i] = x[idx[i] & 0xffff]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I8);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I16);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Value mask = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.select(b.cmp_gt(mask, b.i32(0)), b.gather(idx, x), b.i32(0));
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, mask, dst): 
    for i in range(n):
        dst[i] = x[idx[i]] if mask[i] > 0 else 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I64);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector,
                                                simjit::ErrorSubKind::CoefficientRangeNeedsNormalization),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    // gather 64
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I8);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I8);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         test_meta()
             .limitation(TestVariant::VectorAll)
             .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::UnsupportedSpecialOps)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I16);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         test_meta()
             .limitation(TestVariant::VectorAll)
             .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::UnsupportedSpecialOps)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I32);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector,
                                                simjit::ErrorSubKind::CoefficientRangeNeedsNormalization),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value tmp = b.gather(idx, x);
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[i] = x[idx[i]]
        )FOO"},
    // scatter 32
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I8);
             Argument dst = b.arg(I32);
             b.scatter(x, idx, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst):
    for i in range(n):
        dst[idx[i] & 0xff] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I16);
             Argument dst = b.arg(I32);
             b.scatter(x, idx, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst):
    for i in range(n):
        dst[idx[i] & 0xffff] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I8);
             b.scatter(x, idx, dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I16);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I16);
             b.scatter(x, idx, dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.scatter(x, idx, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I64);
             b.scatter(x, idx, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    // scatter 64
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I8);
             b.scatter(x, idx, dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I16);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I16);
             b.scatter(x, idx, dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I32);
             b.scatter(x, idx, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst): 
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.scatter(x, idx, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, x, idx, dst):
    for i in range(n):
        dst[idx[i]] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             b.scalar_only();

             Value idx = b.index(I32);
             Argument first_dst = b.arg(I32);
             Argument src_const = b.arg(I32);
             Argument gather_src = b.arg(I64);
             Argument second_dst = b.arg(I32);

             b.scatter(idx, idx, first_dst);
             Value rounded = b.float_cast(b.float_cast(b.i32(0x7fffffff), F32), I32);
             Predicate cond = b.cmp_uge(b.zext(rounded, I64), b.gather(idx, gather_src));
             b.cond_scatter(b.load_splat(src_const), idx, cond, second_dst);
         },
         // llvm freeze poison
         TestMetadata{}.unstable(TestVariant::All)},
    // cond_scatter
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I8);
             Argument dst = b.arg(I32);
             b.cond_scatter(x, idx, b.input_predicate_arg(), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I16);
             Argument dst = b.arg(I32);
             b.cond_scatter(x, idx, b.input_predicate_arg(), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I8);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I16);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I16);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I64);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // cond scatter 64
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I8);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I16);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I16);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I32);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value idx = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Predicate cond = b.input_predicate_arg();
             b.cond_scatter(x, idx, cond, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // cond_store
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.cond_store(x, b.cmp_ge(y, b.i8(0)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.cond_store(x, b.cmp_ge(y, b.i16(0)), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value y = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.cond_store(x, b.cmp_ge(y, b.i32(0)), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        if y[i] >= 0: dst[i] = x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.cond_store(x, b.cmp_ge(y, b.i64(0)), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        if y[i] >= 0: dst[i] = x[i]
        )FOO"},

    // cond_store unaligned
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.cond_store(x, b.cmp_ge(y, b.i8(0)), dst, LoadStoreKind::Unaligned);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.cond_store(x, b.cmp_ge(y, b.i16(0)), dst, LoadStoreKind::Unaligned);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.cond_store(x, b.cmp_ge(y, b.i32(0)), dst, LoadStoreKind::Unaligned);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.cond_store(x, b.cmp_ge(y, b.i64(0)), dst, LoadStoreKind::Unaligned);
    }},
    // index
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             b.store(b.index(I32), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst): 
    for i in range(n):
        dst[i] = i
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I64);
             b.store(b.index(I64), dst);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, dst): 
    for i in range(n):
        dst[i] = i
        )FOO"},
    // sum
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.sum(x, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, dst): 
    dst[0] = sum(x)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.sum(x, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, dst): 
    dst[0] = sum(x)
        )FOO"},
    // product
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.product(x, dst);
         },
         PASS_ALL,
         R"FOO(
import math
def func(n, x, dst): 
    dst[0] = math.prod(x)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.product(x, dst);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
import math
def func(n, x, dst): 
    dst[0] = math.prod(x)
        )FOO"},
    // Min
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.min_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import math
def func(n, x, dst): 
    dst[0] = min(x)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.min_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import math
def func(n, x, dst):
    dst[0] = min(x)
        )FOO"},
    Test{[](FunctionBuilder &b) {
        b.cond_store(b.i32(0), b.false_(), b.arg(I32), LoadStoreKind::Aligned);
        b.min_agg(b.andnot(b.i16(1), b.i16(-32768)), b.arg(I16));
    }},
    // Max
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.max_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import math
def func(n, x, dst): 
    dst[0] = max(x)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.max_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import math
def func(n, x, dst): 
    dst[0] = max(x)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I16);
             Argument dst = b.arg(I16);
             b.max_agg(b.srl(x, b.i16(0)), dst);
         },
         PASS_ALL},
    Test{[](FunctionBuilder &b) {
             Argument floats = b.arg(F32);
             Value x = b.input_arg(I8);
             Argument dst = b.arg(I8);
             b.store(b.f32(-2.5f), floats);
             b.max_agg(b.abs(x), dst);
         },
         PASS_ALL},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Argument dst = b.arg(I8);
             b.max_agg(b.negate(x), dst);
         },
         PASS_ALL},
    // Unsigned Min
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.umin_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.umin_agg(x, dst);
    }},
    // Unsigned Max
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.umax_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.umax_agg(x, dst);
    }},
    // And
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.and_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a & b, x, 0xFFFFFFFF)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.and_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a & b, x, 0xFFFFFFFFFFFFFFFF)
        )FOO"},
    // Or
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.or_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a | b, x, 0)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.or_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a | b, x, 0)
        )FOO"},
    // Xor
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.xor_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a ^ b, x, 0)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.xor_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a ^ b, x, 0)
        )FOO"},
    // AndNot
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.andnot_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a & ~b, x, 0xFFFFFFFF)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.andnot_agg(x, dst);
         },
         PASS_ALL,
         R"FOO(
import functools
def func(n, x, dst): 
    dst[0] = functools.reduce(lambda a,b: a & ~b, x, 0xFFFFFFFFFFFFFFFF)
        )FOO"},
    // countif
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Predicate tmp = b.cmp_gt(y, z);
             b.countif(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    s = 0
    for i in range(n):
        s += y[i] > z[i]
    x[0] = s
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value splat = b.load_splat(b.arg(I32));
             Value selected = b.select(b.false_(), b.i32(1), b.input_arg(I32));
             Value masked = b.andnot(splat, selected);
             Value div = b.div(b.i32(1), b.i32(-1));
             Value neg = b.negate(b.load_splat(b.arg(I32)));
             Value mixed = b.xor_(masked, b.max(div, neg));
             b.countif(b.cmp_ne(mixed, b.i32(0)), b.arg(I64));
         },
         PASS_ALL},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value chosen = b.input_arg(I32);
             Value dead_input = b.input_arg(I32);
             Value dead_branch = b.add(dead_input, b.i32(17));
             b.store(b.select(b.true_(), chosen, dead_branch), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_input):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value dead_input = b.input_arg(I32);
             Value chosen = b.input_arg(I32);
             Value dead_branch = b.mul(dead_input, b.i32(3));
             b.store(b.select(b.false_(), dead_branch, chosen), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, dead_input, chosen):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value chosen = b.input_arg(I32);
             Predicate dead_cond = b.input_predicate_arg();
             b.store(b.select(dead_cond, chosen, chosen), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_cond):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value chosen = b.input_arg(I32);
             Value dead_cond_input = b.input_arg(I32);
             Predicate cond = b.cmp_eq(dead_cond_input, dead_cond_input);
             b.cond_store(chosen, cond, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_cond_input):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value chosen = b.input_arg(I32);
             Value dead_cond_input = b.input_arg(I32);
             Predicate cond = b.cmp_eq(dead_cond_input, dead_cond_input);
             b.cond_scatter(chosen, b.index(I32), cond, dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, dst, chosen, dead_cond_input):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I64);
             Value chosen = b.input_arg(I64);
             Value dead_cond_input = b.input_arg(I32);
             Predicate cond = b.cmp_eq(dead_cond_input, dead_cond_input);
             b.sum_if(chosen, cond, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_cond_input):
    s = 0
    for i in range(n):
        s += chosen[i]
    dst[0] = s
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value chosen = b.input_arg(I32);
             Value dead_input = b.input_arg(I32);
             Value cond_input = b.input_arg(I32);
             Predicate cond = b.and_(b.cmp_uge(cond_input, b.i32(0)), b.true_());
             b.store(b.select(cond, chosen, b.add(dead_input, b.i32(17))), dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_input, cond_input):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I32);
             Value chosen = b.input_arg(I32);
             Value dead_input = b.input_arg(I32);
             Value left_input = b.input_arg(I32);
             Value right_input = b.input_arg(I32);
             Predicate cond = b.or_(b.cmp_ult(left_input, b.i32(0)), b.cmp_eq(right_input, right_input));
             b.cond_store(b.select(cond, chosen, b.add(dead_input, b.i32(5))), cond, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_input, left_input, right_input):
    for i in range(n):
        dst[i] = chosen[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I64);
             Value chosen = b.input_arg(I64);
             Value dead_input = b.input_arg(I64);
             Value cond_input = b.input_arg(I32);
             Predicate false_cond = b.cmp_ult(cond_input, b.i32(0));
             Predicate cond = b.xnor(false_cond, false_cond);
             b.sum_if(b.select(cond, chosen, b.add(dead_input, b.i64(11))), cond, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, dst, chosen, dead_input, cond_input):
    s = 0
    for i in range(n):
        s += chosen[i]
    dst[0] = s
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Predicate cond = b.cmp_le(b.max(b.i8(0x7f), x), b.u8(0xf0));
             b.countif(b.and_(b.true_(), cond), b.arg(I64));
         },
         PASS_ALL},
    Test{[](FunctionBuilder &b) {
             Argument dst64 = b.arg(I64);
             Argument dst_bool = b.arg(I1);
             Argument dst32 = b.arg(I32);
             Predicate never = b.false_();
             Value idx = b.index(I32);
             b.store(b.zext(b.select(never, idx, idx), I64), dst64);
             b.and_agg(never, dst_bool);
             b.store(b.mod(b.i32(0x12345678), b.i32(-1)), dst32);
         },
         PASS_ALL},
    // logical and agg
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Predicate tmp = b.cmp_gt(y, z);
             b.and_agg(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    x[0] = all(map(lambda it: it[0] > it[1], zip(y, z)))
        )FOO"},
    // logical or agg
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Predicate tmp = b.cmp_gt(y, z);
             b.or_agg(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    x[0] = any(map(lambda it: it[0] > it[1], zip(y, z)))
        )FOO"},
    // logical andnot agg
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Predicate tmp = b.cmp_gt(y, z);
             b.andnot_agg(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    x[0] = not any(map(lambda it: it[0] <= it[1], zip(y, z)))
        )FOO"},
    // logical xor agg
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I1);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Predicate tmp = b.cmp_gt(y, z);
             b.xor_agg(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z): 
    s = 0
    for i in range(n):
        s += y[i] > z[i]
    x[0] = s & 1
        )FOO"},
    // mul32
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument dst = b.arg(I64);
        Value tmp = b.mul(b.zext(x, I64), b.zext(y, I64));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value y = b.input_arg(I32);
             Argument dst = b.arg(I64);
             Value tmp = b.mul(b.sext(x, I64), b.sext(y, I64));
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        dst[i] = x[i] * y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I64);
        Value tmp = b.mul(b.zext(x, I64), b.zext(x, I64));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I64);
        Value tmp = b.mul(b.zext(x, I64), b.i64(123));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I64);
             Value tmp = b.mul(b.sext(x, I64), b.i64(123));
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, dst): 
    for i in range(n):
        dst[i] = x[i] * 123
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I64);
        Value tmp = b.mul(b.zext(x, I64), b.i64(123));
        b.sum(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value xl = b.zext(b.trunc(x, I32), I64);
             Value yl = b.zext(b.trunc(y, I32), I64);
             Value tmp = b.mul(xl, yl);
             b.store(tmp, dst);
         },
         PASS_ALL, R"FOO(
def func(n, x, y, dst):
    for i in range(n):
        xl = x[i] & 0xffffffff
        yl = y[i] & 0xffffffff
        dst[i] = xl * yl
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value xl = b.sext(b.trunc(x, I32), I64);
             Value yl = b.sext(b.trunc(y, I32), I64);
             Value tmp = b.mul(xl, yl);
             b.store(tmp, dst);
         },
         PASS_ALL, R"FOO(
def func(n, x, y, dst):
    for i in range(n):
        xl = ((x[i] & 0xffffffff) ^ 0x80000000) - 0x80000000
        yl = ((y[i] & 0xffffffff) ^ 0x80000000) - 0x80000000
        dst[i] = xl * yl
        )FOO"},
    // blends
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value y = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.select(b.cmp_gt(x, y), b.add(x, b.i32(10)), x);
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        dst[i] = x[i] + 10 if x[i] > y[i] else x[i] 
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.select(b.cmp_gt(x, b.i32(0)), b.add(x, b.i32(10)), x);
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, dst): 
    for i in range(n):
        dst[i] = x[i] + 10 if x[i] > 0 else x[i] 
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.select(b.cmp_gt(x, b.i32(0)), b.mul(x, b.i32(10)), x);
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, dst): 
    for i in range(n):
        dst[i] = x[i] * 10 if x[i] > 0 else x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate y = b.input_predicate_arg();
        Argument dst = b.arg(I32);
        Value tmp = b.select(y, b.add(x, b.i32(10)), x);
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Predicate y = b.input_predicate_arg();
        Argument dst = b.arg(I32);
        Value tmp = b.select(y, b.i32(-10), b.i32(10));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Predicate y = b.input_predicate_arg();
        Argument dst = b.arg(I32);
        Value tmp = b.select(y, b.i32(-1), b.i32(0));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Predicate y = b.input_predicate_arg();
        Argument dst = b.arg(I64);
        Value tmp = b.select(y, b.i64(-10), b.i64(10));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
             Argument y = b.arg(I32);
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.select(b.cmp_gt(x, b.i32(0)), b.load(y), x);
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, y, x, dst): 
    for i in range(n):
        dst[i] = y[i] if x[i] > 0 else x[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument y = b.arg(I32);
             Value x = b.input_arg(I32);
             Value mask = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Argument dst_copy = b.arg(I32);
             Value loaded = b.load(y);
             Value tmp = b.select(b.cmp_gt(mask, b.i32(0)), loaded, x);
             b.store(tmp, dst);
             b.store(loaded, dst_copy);
         },
         {},
         R"FOO(
def func(n, y, x, mask, dst, dst_copy): 
    for i in range(n):
        dst[i] = y[i] if mask[i] > 0 else x[i]
        dst_copy[i] = y[i]
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I64);
             Value tmp = b.select(b.cmp_gt(x, b.i32(0)), b.zext(x, I64), b.i64(0));
             b.store(tmp, dst);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, dst): 
    for i in range(n):
        dst[i] = x[i] & 0xffffffff if x[i] > 0 else 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value tmp = b.select(b.cmp_uge(x, y), b.i64(-10), b.i64(10));
             b.store(tmp, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        lhs = x[i] & 0xFFFFFFFFFFFFFFFF
        rhs = y[i] & 0xFFFFFFFFFFFFFFFF
        dst[i] = -10 if lhs >= rhs else 10
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Argument dst = b.arg(I64);
        Value tmp = b.select(b.cmp_ge(x, y), b.i64(-10), b.i64(10));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Predicate y = b.input_predicate_arg();
        Argument dst = b.arg(I16);
        Value tmp = b.select(y, b.i16(-10), b.i16(10));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Predicate y = b.input_predicate_arg();
        Argument dst = b.arg(I8);
        Value tmp = b.select(y, b.i8(-10), b.i8(10));
        b.store(tmp, dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Predicate cond = b.cmp_le(b.max(b.i8(0x7f), x), b.u8(0xf0));
             b.store(b.select(cond, b.i32(1), b.i32(0)), b.arg(I32));
         },
         ONLY_SCALAR},
    // blend different widths
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I64);
             Value tmp = b.select(b.cmp_le(x, b.i32(123)), b.i64(-10), b.i64(10));
             b.store(tmp, dst);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, dst): 
    for i in range(n):
        dst[i] = -10 if x[i] < 123 else 10
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Value tmp = b.select(b.cmp_le(x, b.i32(123)), y, b.i64(10));
             b.store(tmp, dst);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        dst[i] = y[i] if x[i] < 123 else 10
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Value tmp = b.select(b.cmp_le(x, b.i64(123)), y, b.i32(10));
             b.store(tmp, dst);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        dst[i] = y[i] if x[i] <= 123 else 10
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I16);
             Argument dst = b.arg(I16);
             Value tmp = b.select(b.cmp_le(x, b.i64(123)), y, b.i16(10));
             b.store(tmp, dst);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, x, y, dst): 
    for i in range(n):
        dst[i] =  y[i] if x[i] <= 123 else 10
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I8);
             Argument dst = b.arg(I8);
             Value tmp = b.select(b.cmp_le(x, b.i64(123)), y, b.i8(10));
             b.store(tmp, dst);
         },
         coefficient_range_limit(TestVariant::VectorAll)},
    // pack
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I8);
             Argument dst = b.arg(I8);
             Argument dst_size = b.arg(I64);
             b.pack(b.trunc(b.index(I32), I8), b.cmp_ge(y, b.i8(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = i
            c += 1
    dsts[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I16);
             Argument dst = b.arg(I16);
             Argument dst_size = b.arg(I64);
             b.pack(b.trunc(b.index(I32), I16), b.cmp_ge(y, b.i16(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = i
            c += 1
    dsts[0] = c
        )FOO"},

    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I8);
             Value y = b.input_arg(I8);
             Argument dst = b.arg(I8);
             Argument dst_size = b.arg(I64);
             b.pack(x, b.cmp_ge(y, b.i8(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = x[i]
            c += 1
    dsts[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I16);
             Value y = b.input_arg(I16);
             Argument dst = b.arg(I16);
             Argument dst_size = b.arg(I64);
             b.pack(x, b.cmp_ge(y, b.i16(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = x[i]
            c += 1
    dsts[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Argument dst_size = b.arg(I64);
             b.pack(b.index(I32), b.cmp_ge(y, b.i32(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = i
            c += 1
    dsts[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Value y = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Argument dst_size = b.arg(I64);
             b.pack(x, b.cmp_ge(y, b.i32(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = x[i]
            c += 1
    dsts[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             Argument dst_size = b.arg(I64);
             b.pack(x, b.cmp_ge(y, b.i64(0)), dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, dst, dsts): 
    c = 0
    for i in range(n):
        if y[i] >= 0: 
            dst[c] = x[i]
            c += 1
    dsts[0] = c
        )FOO"},
    // add safety check
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.store(b.add_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.add_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.add_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.add_checked(x, y), dst);
    }},
    // Compound checked operation, with and without an activity mask.
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.add(x, y), active), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.add(x, y), active), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.add(x, y), active), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.add(x, y), active), b.arg(I64));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        b.store(b.checked_op(b.add(x, y)), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        b.store(b.checked_op(b.add(x, y)), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.store(b.checked_op(b.add(x, y)), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        b.store(b.checked_op(b.add(x, y)), b.arg(I64));
    }},
    // sub safety check
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.store(b.sub_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.sub_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.sub_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.sub_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.sub(x, y), active), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.sub(x, y), active), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.sub(x, y), active), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.sub(x, y), active), b.arg(I64));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        b.store(b.checked_op(b.sub(x, y)), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        b.store(b.checked_op(b.sub(x, y)), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.store(b.checked_op(b.sub(x, y)), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        b.store(b.checked_op(b.sub(x, y)), b.arg(I64));
    }},
    // mul safety check
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.store(b.mul_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.mul_checked(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.mul(x, y), active), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.mul(x, y), active), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Predicate active = b.input_predicate_arg();
        b.store(b.checked_op(b.mul(x, y), active), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Predicate active = b.input_predicate_arg();
             b.store(b.checked_op(b.mul(x, y), active), b.arg(I64));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        b.store(b.checked_op(b.mul(x, y)), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        b.store(b.checked_op(b.mul(x, y)), b.arg(I16));
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.store(b.checked_op(b.mul(x, y)), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.checked_op(b.mul(x, y)), b.arg(I64));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // {
    //     [](FunctionBuilder &b) {
    //         b.arg_safety_check();
    //         Value x = b.input_arg(I32);
    //         Value y = b.input_arg(I32);
    //         Argument dst = b.arg(I32);
    //         b.store(b.mul_checked(x, y), dst);
    //     },
    // },
    // {
    //     [](FunctionBuilder &b) {
    //         b.arg_safety_check();
    //         Value x = b.input_arg(I64);
    //         Value y = b.input_arg(I64);
    //         Argument dst = b.arg(I64);
    //         b.store(b.mul_checked(x, y), dst);
    //     },
    // },
    // trunc safety check
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I8);
        b.store(b.trunc_checked(x, I8), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I8);
        b.store(b.trunc_checked(x, I8), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I16);
        b.store(b.trunc_checked(x, I16), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I8);
        b.store(b.trunc_checked(x, I8), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I32);
        b.store(b.trunc_checked(x, I32), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I16);
        b.store(b.trunc_checked(x, I16), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I8);
        b.store(b.trunc_checked(b.trunc_checked(x, I16), I8), dst);
    }},
    // negate safety check
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.store(b.negate_checked(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.negate_checked(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.negate_checked(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.negate_checked(x), dst);
    }},
    // abs safety check
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.store(b.abs_checked(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.abs_checked(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.abs_checked(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.abs_checked(x), dst);
    }},
    // 128-bit sum
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument dst = b.arg(I128);
             b.sum(x, dst);
         },
         PASS_ALL,
         R"FOO(
def func(n, arg, dst): 
    s = 0
    for i in range(n):
        s += int(arg[i])
    dst[0] = s
    dst[1] = s >> 64
)FOO"},
    // countif
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Predicate less = b.cmp_lt(x, b.i64(0));
        Argument dst = b.arg(I64);
        b.countif(less, dst);
    }},
    // reverse bits
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value tmp = b.reverse_bits_i8(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value tmp = b.reverse_bits_i8(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Value tmp = b.reverse_bits_i8(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.reverse_bits_i8(x);
        b.output_arg(tmp);
    }},
    // replicate bits
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value tmp = b.replicate_ith_bit_i8(x, 0);
             b.output_arg(tmp);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value tmp = b.replicate_ith_bit_i8(x, 1);
             b.output_arg(tmp);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value tmp = b.replicate_ith_bit_i8(x, 7);
             b.output_arg(tmp);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // slli_i8
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.sll(x, b.i8(7));
        b.output_arg(tmp);
    }},
    Test{
        [](FunctionBuilder &b) {
            Value x = b.input_arg(I8);
            Value tmp = b.sll(x, b.i8(0));
            b.output_arg(tmp);
        },
    },
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.sll(x, b.i8(1));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.sll(x, b.i8(4));
        b.output_arg(tmp);
    }},
    // srli_i8
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.srl(x, b.i8(7));
        b.output_arg(tmp);
    }},
    Test{
        [](FunctionBuilder &b) {
            Value x = b.input_arg(I8);
            Value tmp = b.srl(x, b.i8(0));
            b.output_arg(tmp);
        },
    },
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.srl(x, b.i8(1));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.srl(x, b.i8(4));
        b.output_arg(tmp);
    }},
    // srai_i8
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.sra(x, b.i8(7));
        b.output_arg(tmp);
    }},
    Test{
        [](FunctionBuilder &b) {
            Value x = b.input_arg(I8);
            Value tmp = b.sra(x, b.i8(0));
            b.output_arg(tmp);
        },
    },
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.sra(x, b.i8(1));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.sra(x, b.i8(4));
        b.output_arg(tmp);
    }},
    // rotli_i8
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotl(x, b.i8(7));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotl(x, b.i8(0));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotl(x, b.i8(1));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotl(x, b.i8(4));
        b.output_arg(tmp);
    }},
    // rotr_i8
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotr(x, b.i8(7));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotr(x, b.i8(0));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotr(x, b.i8(1));
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Value tmp = b.rotr(x, b.i8(4));
        b.output_arg(tmp);
    }},
    // log2_no_zero
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        x = b.or_(x, b.i64(1));
        Value tmp = b.log2_no_zero(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        x = b.or_(x, b.i32(1));
        Value tmp = b.log2_no_zero(x);
        b.output_arg(tmp);
    }},
    // log2
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value tmp = b.log2(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value tmp = b.log2(x);
        b.output_arg(tmp);
    }},
    // sign_no_zero
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value tmp = b.sign_no_zero(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value tmp = b.sign_no_zero(x);
        b.output_arg(tmp);
    }},
    // sign
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value tmp = b.sign(x);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value tmp = b.sign(x);
        b.output_arg(tmp);
    }},
    // copysign_no_zero
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Value tmp = b.copysign_no_zero(x, y);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Value tmp = b.copysign_no_zero(x, y);
        b.output_arg(tmp);
    }},
    // copysign
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Value tmp = b.copysign(x, y);
        b.output_arg(tmp);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Value tmp = b.copysign(x, y);
        b.output_arg(tmp);
    }},
    // strict addition with const
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate x_notnull = b.input_predicate_arg();
        Value c = b.i32(111);
        Argument x_dst = b.arg(I32);
        Argument x_notnull_dst = b.arg(I1);
        b.cond_store(b.add(x, c), x_notnull, x_dst);
        b.store(x_notnull, x_notnull_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Predicate x_notnull = b.input_predicate_arg();
        Value c = b.i64(111);
        Argument x_dst = b.arg(I64);
        Argument x_notnull_dst = b.arg(I1);
        b.cond_store(b.add(x, c), x_notnull, x_dst);
        b.store(x_notnull, x_notnull_dst);
    }},
    // binary strict addition
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate x_notnull = b.input_predicate_arg();
        Value y = b.input_arg(I32);
        Predicate y_notnull = b.input_predicate_arg();
        Argument x_dst = b.arg(I32);
        Argument x_notnull_dst = b.arg(I1);
        Predicate result_notnull = b.and_(x_notnull, y_notnull);
        b.cond_store(b.add(x, y), result_notnull, x_dst);
        b.store(result_notnull, x_notnull_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Predicate x_notnull = b.input_predicate_arg();
        Value y = b.input_arg(I64);
        Predicate y_notnull = b.input_predicate_arg();
        Argument x_dst = b.arg(I64);
        Argument x_notnull_dst = b.arg(I1);
        Predicate result_notnull = b.and_(x_notnull, y_notnull);
        b.cond_store(b.add(x, y), result_notnull, x_dst);
        b.store(result_notnull, x_notnull_dst);
    }},
    // binary strict addition with bools
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Predicate x_notnull = b.cmp_ne(b.input_arg(I8), b.i8(0));
             Value y = b.input_arg(I32);
             Predicate y_notnull = b.cmp_ne(b.input_arg(I8), b.i8(0));
             Argument x_dst = b.arg(I32);
             Argument x_notnull_dst = b.arg(I1);
             Predicate result_notnull = b.and_(x_notnull, y_notnull);
             b.cond_store(b.add(x, y), result_notnull, x_dst);
             b.store(result_notnull, x_notnull_dst);
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Predicate x_notnull = b.cmp_ne(b.input_arg(I8), b.i8(0));
             Value y = b.input_arg(I64);
             Predicate y_notnull = b.cmp_ne(b.input_arg(I8), b.i8(0));
             Argument x_dst = b.arg(I64);
             Argument x_notnull_dst = b.arg(I1);
             Predicate result_notnull = b.and_(x_notnull, y_notnull);
             b.cond_store(b.add(x, y), result_notnull, x_dst);
             b.store(result_notnull, x_notnull_dst);
         },
         coefficient_range_limit(TestVariant::VectorAll)},
    // binary strict addition with int_cast
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate x_notnull = b.input_predicate_arg();
        Value y = b.sext(b.input_arg(I16), I32);
        Predicate y_notnull = b.input_predicate_arg();
        Argument x_dst = b.arg(I32);
        Argument x_notnull_dst = b.arg(I1);
        Predicate result_notnull = b.and_(x_notnull, y_notnull);
        b.cond_store(b.add(x, y), result_notnull, x_dst);
        b.store(result_notnull, x_notnull_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.sext(b.input_arg(I32), I64);
        Predicate x_notnull = b.input_predicate_arg();
        Value y = b.input_arg(I64);
        Predicate y_notnull = b.input_predicate_arg();
        Argument x_dst = b.arg(I64);
        Argument x_notnull_dst = b.arg(I1);
        Predicate result_notnull = b.and_(x_notnull, y_notnull);
        b.cond_store(b.add(x, y), result_notnull, x_dst);
        b.store(result_notnull, x_notnull_dst);
    }},
    // Strict sum
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument x_dst = b.arg(I32);
        Predicate x_notnull = b.input_predicate_arg();
        b.sum_if(x, x_notnull, x_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Predicate x_notnull = b.input_predicate_arg();
        b.sum_if(x, x_notnull, x_dst);
    }},
    // Strict mul
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument x_dst = b.arg(I32);
        Predicate x_notnull = b.input_predicate_arg();
        b.product_if(x, x_notnull, x_dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument x_dst = b.arg(I64);
             Predicate x_notnull = b.input_predicate_arg();
             b.product_if(x, x_notnull, x_dst);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // Strict sum with bool
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument x_dst = b.arg(I32);
             Predicate x_notnull = b.cmp_ne(b.input_arg(I8), b.i8(0));
             b.sum_if(x, x_notnull, x_dst);
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Argument x_dst = b.arg(I64);
             Predicate x_notnull = b.cmp_ne(b.input_arg(I8), b.i8(0));
             b.sum_if(x, x_notnull, x_dst);
         },
         coefficient_range_limit(TestVariant::VectorAll)},

    // two aggs
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument x_dst = b.arg(I32);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument x_dst = b.arg(I32);
        Value y = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I16);
        Argument y_dst = b.arg(I32);
        b.sum(x, x_dst);
        b.sum(b.sext(y, I32), y_dst);
    }},
    // agg and expr
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        b.sum(x, x_dst);
        b.store(b.add(y, z), y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        b.sum(x, x_dst);
        b.store(b.add(y, z), y_dst);
    }},
    // three aggs
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        Value z = b.input_arg(I64);
        Argument z_dst = b.arg(I64);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
        b.sum(z, z_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        Value z = b.input_arg(I32);
        Argument z_dst = b.arg(I32);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
        b.sum(z, z_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        Value z = b.input_arg(I16);
        Argument z_dst = b.arg(I32);
        b.sum(x, x_dst);
        b.sum(y, y_dst);
        b.sum(b.sext(z, I32), z_dst);
    }},
    // tpch q1
    Test{[](FunctionBuilder &b) {
             Value l_quantity = b.input_arg(I64);
             Value l_extendedprice = b.input_arg(I64);
             Value l_discount = b.input_arg(I64);
             Argument sum_qty = b.arg(I64);
             Argument sum_base_price = b.arg(I64);
             Argument sum_disc_price = b.arg(I64);
             b.sum(l_quantity, sum_qty);
             b.sum(l_extendedprice, sum_base_price);
             b.sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), sum_disc_price);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value l_quantity = b.input_arg(I64);
             Value l_extendedprice = b.input_arg(I64);
             Value l_discount = b.sext(b.input_arg(I32), I64);
             Argument sum_qty = b.arg(I64);
             Argument sum_base_price = b.arg(I64);
             Argument sum_disc_price = b.arg(I64);
             b.sum(l_quantity, sum_qty);
             b.sum(l_extendedprice, sum_base_price);
             b.sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), sum_disc_price);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value l_quantity = b.sext(b.input_arg(I16), I64);
             Value l_extendedprice = b.input_arg(I64);
             Value l_discount = b.sext(b.input_arg(I32), I64);
             Argument sum_qty = b.arg(I64);
             Argument sum_base_price = b.arg(I64);
             Argument sum_disc_price = b.arg(I64);
             b.sum(l_quantity, sum_qty);
             b.sum(l_extendedprice, sum_base_price);
             b.sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), sum_disc_price);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
        Value l_quantity = b.sext(b.input_arg(I16), I32);
        Value l_extendedprice = b.trunc(b.input_arg(I64), I32);
        Value l_discount = b.input_arg(I32);
        Argument sum_qty = b.arg(I32);
        Argument sum_base_price = b.arg(I32);
        Argument sum_disc_price = b.arg(I32);
        b.sum(l_quantity, sum_qty);
        b.sum(l_extendedprice, sum_base_price);
        b.sum(b.mul(l_extendedprice, b.sub(b.i32(1), l_discount)), sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
             Value l_quantity = b.input_arg(I64);
             Value l_extendedprice = b.input_arg(I64);
             Value l_discount = b.input_arg(I64);
             Value l_tax = b.input_arg(I64);
             Argument sum_qty = b.arg(I64);
             Argument sum_base_price = b.arg(I64);
             Argument sum_disc_price = b.arg(I64);
             Argument sum_charge = b.arg(I64);
             b.sum(l_quantity, sum_qty);
             b.sum(l_extendedprice, sum_base_price);
             b.sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), sum_disc_price);
             b.sum(b.mul(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), b.add(b.i64(1), l_tax)), sum_charge);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             Value l_quantity = b.input_arg(I64);
             Value l_extendedprice = b.input_arg(I32);
             Value l_discount = b.input_arg(I32);
             Value l_tax = b.input_arg(I32);
             Argument sum_qty = b.arg(I64);
             Argument sum_base_price = b.arg(I64);
             Argument sum_disc_price = b.arg(I64);
             Argument sum_charge = b.arg(I64);
             b.sum(l_quantity, sum_qty);
             b.sum(b.sext(l_extendedprice, I64), sum_base_price);
             b.sum(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)), sum_disc_price);
             b.sum(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                         b.sext(b.add(b.i32(1), l_tax), I64)),
                   sum_charge);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // bit_test
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I32);
        b.store(b.bit_test(x, b.i32(0x48170000)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I32);
        b.store(b.bit_test(x, b.i32(2)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I64);
        b.store(b.bit_test(x, b.i64(2)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I8);
        b.store(b.bit_test(x, b.i8(2)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I16);
        b.store(b.bit_test(x, b.i16(2)), dst);
    }},
    // bit_testn
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I32);
        b.store(b.bit_testn(x, b.i32(0x48170000)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I32);
        b.store(b.bit_testn(x, b.i32(2)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I64);
        b.store(b.bit_testn(x, b.i64(2)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I8);
        b.store(b.bit_testn(x, b.i8(2)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I16);
        b.store(b.bit_testn(x, b.i16(2)), dst);
    }},
    // bit_test + mem
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        b.store(b.bit_test(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        b.store(b.bit_test(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.store(b.bit_test(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        b.store(b.bit_test(x, y), dst);
    }},
    // bit_testn + mem
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);
        b.store(b.bit_testn(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I16);
        Value y = b.input_arg(I16);
        b.store(b.bit_testn(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.store(b.bit_testn(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        b.store(b.bit_testn(x, y), dst);
    }},
    // movmask
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Predicate tmp = b.cmp_gt(y, b.i8(0));
        Value old = b.select(tmp, b.u8(0xFF), b.i8(0));
        b.store(old, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Predicate tmp = b.cmp_gt(y, b.i16(0));
        Value old = b.select(tmp, b.u16(0xFFFF), b.i16(0));
        b.store(old, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Predicate tmp = b.cmp_gt(y, b.i32(0));
        Value old = b.select(tmp, b.u32(0xFFFFFFFF), b.i32(0));
        b.store(old, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Predicate tmp = b.cmp_gt(y, b.i64(0));
        Value old = b.select(tmp, b.u64(0xFFFFFFFFFFFFFFFF), b.i64(0));
        b.store(old, x);
    }},
    // I originally wrote these 4 tests to mean 4 above, but messed up types (these do sign extend instead of zero
    // extend when process constants)
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I8);
        Predicate tmp = b.cmp_gt(y, b.i8(0));
        Value old = b.select(tmp, b.u8(0xFFu), b.i8(0));
        b.store(old, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I16);
        Predicate tmp = b.cmp_gt(y, b.i16(0));
        Value old = b.select(tmp, b.u16(0xFFFFu), b.i16(0));
        b.store(old, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I32);
        Predicate tmp = b.cmp_gt(y, b.i32(0));
        Value old = b.select(tmp, b.u32(0xFFFFFFFFu), b.i32(0));
        b.store(old, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I64);
        Predicate tmp = b.cmp_gt(y, b.i64(0));
        Value old = b.select(tmp, b.u64(0xFFFFFFFFFFFFFFFF), b.i64(0));
        b.store(old, x);
    }},
    // clamp
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I8);
        Value x = b.input_arg(I8);
        b.store(b.min(b.max(x, b.i8(-100)), b.ucon<I8>(200)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I16);
        Value x = b.input_arg(I16);
        b.store(b.min(b.max(x, b.i16(-100)), b.i16(200)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        b.store(b.min(b.max(x, b.i32(-100)), b.i32(200)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        b.store(b.min(b.max(x, b.i64(-100)), b.i64(200)), dst);
    }},
    // uclamp
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I8);
        Value x = b.input_arg(I8);
        b.store(b.umin(b.umax(x, b.i8(100)), b.ucon<I8>(200)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I16);
        Value x = b.input_arg(I16);
        b.store(b.umin(b.umax(x, b.i16(100)), b.i16(200)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        b.store(b.umin(b.umax(x, b.i32(100)), b.i32(200)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        b.store(b.umin(b.umax(x, b.i64(100)), b.i64(200)), dst);
    }},
    // has_single_bit
    Test{[](FunctionBuilder &b) {
             Argument dst = b.arg(I1);
             Value x = b.input_arg(I32);
             x = b.and_(x, b.zext(b.bit2bool(b.bit_testn(x, b.i32(63))), I32));
             b.store(b.has_single_bit(x), dst);
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        Value x = b.input_arg(I64);
        b.store(b.has_single_bit(x), dst);
    }},
    // bit_floor
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        b.store(b.bit_floor(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        b.store(b.bit_floor(x), dst);
    }},
    // bit_ceil
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        x = b.and_(x, b.i32(0x7FFFFFFF));
        b.store(b.bit_ceil(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        x = b.and_(x, b.i64(0x7FFFFFFFFFFFFFFF));
        b.store(b.bit_ceil(x), dst);
    }},
    // sll overflow
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I8);
        Value x = b.input_arg(I8);
        Value y = b.srl(b.input_arg(I8), b.i8(4));
        Value tmp = b.sll_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I16);
        Value x = b.input_arg(I16);
        Value y = b.srl(b.input_arg(I16), b.i16(11));
        Value tmp = b.sll_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.srl(b.input_arg(I32), b.i32(26));
        Value tmp = b.sll_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        Value y = b.srl(b.input_arg(I64), b.i64(57));
        Value tmp = b.sll_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    // srl overflow
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I8);
        Value x = b.input_arg(I8);
        Value y = b.srl(b.input_arg(I8), b.i8(4));
        Value tmp = b.srl_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I16);
        Value x = b.input_arg(I16);
        Value y = b.srl(b.input_arg(I16), b.i16(11));
        Value tmp = b.srl_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.srl(b.input_arg(I32), b.i32(26));
        Value tmp = b.srl_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        Value y = b.srl(b.input_arg(I64), b.i64(57));
        Value tmp = b.srl_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    // sra overflow
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I8);
        Value x = b.input_arg(I8);
        Value y = b.srl(b.input_arg(I8), b.i8(4));
        Value tmp = b.sra_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I16);
        Value x = b.input_arg(I16);
        Value y = b.srl(b.input_arg(I16), b.i16(11));
        Value tmp = b.sra_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.srl(b.input_arg(I32), b.i32(26));
        Value tmp = b.sra_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        Value y = b.srl(b.input_arg(I64), b.i64(57));
        Value tmp = b.sra_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    // rol overflow
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I8);
             Value x = b.input_arg(I8);
             Value y = b.srl(b.input_arg(I8), b.i8(4));
             Value tmp = b.rotl_checked(x, y);
             b.store(b.sub(tmp, tmp), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I16);
             Value x = b.input_arg(I16);
             Value y = b.srl(b.input_arg(I16), b.i16(11));
             Value tmp = b.rotl_checked(x, y);
             b.store(b.sub(tmp, tmp), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.srl(b.input_arg(I32), b.i32(26));
        Value tmp = b.rotl_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        Value y = b.srl(b.input_arg(I64), b.i64(57));
        Value tmp = b.rotl_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    // ror overflow
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I8);
             Value x = b.input_arg(I8);
             Value y = b.srl(b.input_arg(I8), b.i8(4));
             Value tmp = b.rotr_checked(x, y);
             b.store(b.sub(tmp, tmp), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
             b.arg_safety_check();
             Argument dst = b.arg(I16);
             Value x = b.input_arg(I16);
             Value y = b.srl(b.input_arg(I16), b.i16(11));
             Value tmp = b.rotr_checked(x, y);
             b.store(b.sub(tmp, tmp), dst);
         },
         ONLY_SCALAR},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.srl(b.input_arg(I32), b.i32(26));
        Value tmp = b.rotr_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Argument dst = b.arg(I64);
        Value x = b.input_arg(I64);
        Value y = b.srl(b.input_arg(I64), b.i64(57));
        Value tmp = b.rotr_checked(x, y);
        b.store(b.sub(tmp, tmp), dst);
    }},
    // pack + gather
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Argument dst_size = b.arg(I64);
             Value tmp = b.gather(idx, x);
             Predicate cond = b.is_positive(tmp);
             b.pack(tmp, cond, dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst, dst_sz): 
    c = 0
    for i in range(n):
        it = x[idx[i]]
        if it > 0:
            dst[c] = it
            c += 1
    dst_sz[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Argument dst_size = b.arg(I64);
             Value tmp = b.gather(idx, x);
             Predicate cond = b.is_positive(tmp);
             b.pack(b.index(I32), cond, dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst, dst_sz): 
    c = 0
    for i in range(n):
        if x[idx[i]] > 0:
            dst[c] = i
            c += 1
    dst_sz[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             Argument dst_size = b.arg(I64);
             Value tmp = b.gather(idx, x);
             Predicate cond = b.is_positive(tmp);
             b.pack(idx, cond, dst, dst_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, idx, dst, dst_sz): 
    c = 0
    for i in range(n):
        if x[idx[i]] > 0:
            dst[c] = idx[i]
            c += 1
    dst_sz[0] = c
        )FOO"},
    // fma
    Test{[](FunctionBuilder &b) {
        Value x1 = b.input_arg(I32);
        Value x2 = b.input_arg(I32);
        Value x3 = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.add(b.mul(x1, x2), x3), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x1 = b.input_arg(I64);
             Value x2 = b.input_arg(I64);
             Value x3 = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.store(b.add(b.mul(x1, x2), x3), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // fms
    Test{[](FunctionBuilder &b) {
        Value x1 = b.input_arg(I32);
        Value x2 = b.input_arg(I32);
        Value x3 = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.sub(b.mul(x1, x2), x3), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x1 = b.input_arg(I64);
             Value x2 = b.input_arg(I64);
             Value x3 = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.store(b.sub(b.mul(x1, x2), x3), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // fnma
    Test{[](FunctionBuilder &b) {
        Value x1 = b.input_arg(I32);
        Value x2 = b.input_arg(I32);
        Value x3 = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.add(b.negate(b.mul(x1, x2)), x3), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x1 = b.input_arg(I64);
             Value x2 = b.input_arg(I64);
             Value x3 = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.store(b.add(b.negate(b.mul(x1, x2)), x3), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // fnms
    Test{[](FunctionBuilder &b) {
        Value x1 = b.input_arg(I32);
        Value x2 = b.input_arg(I32);
        Value x3 = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.sub(b.negate(b.mul(x1, x2)), x3), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x1 = b.input_arg(I64);
             Value x2 = b.input_arg(I64);
             Value x3 = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.store(b.sub(b.negate(b.mul(x1, x2)), x3), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // dot
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.sum(b.mul(x, y), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I64);
             Value y = b.input_arg(I64);
             Argument dst = b.arg(I64);
             b.sum(b.mul(x, y), dst);
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // mask constant operand
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.store(b.xor_(x, b.u8(0xFFu)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.xor_(x, b.u16(uint16_t(0xFFFFu))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.xor_(x, b.i32(int(0xFFFFFFFF))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.xor_(x, b.i64(int64_t(0xFFFFFFFFFFFFFFFF))), dst);
    }},

    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.and_(x, b.u16(0xFF00)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.store(b.or_(x, b.i16(0x00FF)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.xor_(x, b.i32(int(0xFF00FF00))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.and_(x, b.i32(0x0000FF00)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.or_(x, b.i32(0x00FF00FF)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I32);
        b.store(b.xor_(x, b.i32(int32_t(0xFF000000))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.and_(x, b.i64(int64_t(0xFFFFFFFFFFFFFFFF))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.or_(x, b.i64(int64_t(0x00FF00FF00FF00FF))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.xor_(x, b.i64(int64_t(0xFF00FF00FF00FF00))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.and_(x, b.i64(int64_t(0xFF0000FF00FFFFFF))), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(I64);
        b.store(b.or_(x, b.i64(0x000000FF00FFFF00)), dst);
    }},
    // casts again (before we bit_test them with memory operands)
    // int_cast i8 -> i16
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.sext(b.add(y, z), I16);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.zext(b.add(y, z), I16);
        b.store(tmp, x);
    }},
    // int_cast i8 -> i32
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.sext(b.add(y, z), I32);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I8);
        Value tmp = b.zext(y, I32);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.zext(b.add(y, z), I32);
        b.store(tmp, x);
    }},
    // int_cast i8 -> i64
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I8);
        Value z = b.input_arg(I8);
        Value tmp = b.sext(b.add(y, z), I64);
        b.store(tmp, x);
    }},
    // int_cast i16 -> i8
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.trunc(b.add(y, z), I8);
        b.store(tmp, x);
    }},
    // int_cast i16 -> i16
    // int_cast i16 -> i32
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.sext(b.add(y, z), I32);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.zext(b.add(y, z), I32);
        b.store(tmp, x);
    }},
    // int_cast i16 -> i64
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.sext(b.add(y, z), I64);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.sext(b.add(y, z), I64);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I16);
        Value z = b.input_arg(I16);
        Value tmp = b.zext(b.add(y, z), I64);
        b.store(tmp, x);
    }},

    // int_cast i32 -> i8
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.trunc(b.add(y, z), I8);
        b.store(tmp, x);
    }},
    // int_cast i32 -> i16
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.trunc(b.add(y, z), I16);
        b.store(tmp, x);
    }},
    // int_cast i32 -> i64
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.sext(b.add(y, z), I64);
        b.store(tmp, x);
    }},
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Value tmp = b.zext(b.add(y, z), I64);
        b.store(tmp, x);
    }},

    // int_cast i64 -> i8
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I8);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.trunc(b.add(y, z), I8);
        b.store(tmp, x);
    }},
    // int_cast i64 -> i16
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I16);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.trunc(b.add(y, z), I16);
        b.store(tmp, x);
    }},
    // int_cast i64 -> i32
    Test{[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Value tmp = b.trunc(b.add(y, z), I32);
        b.store(tmp, x);
    }},
    // Aggregates for small types
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.sum(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.sum(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.product(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.product(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.min_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.min_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.max_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.max_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.umin_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.umin_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.umax_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.umax_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.and_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.and_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.or_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.or_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.xor_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.xor_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.andnot_agg(x, dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.andnot_agg(x, dst);
    }},
    // conditional aggregates for small types
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.sum_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.sum_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.product_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.product_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.min_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.min_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.max_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.max_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.umin_agg_if(x, b.cmp_gt(x, b.i8(100)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.umin_agg_if(x, b.cmp_gt(x, b.i16(10000)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.umax_agg_if(x, b.cmp_gt(x, b.i8(100)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.umax_agg_if(x, b.cmp_gt(x, b.i16(10000)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.and_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.and_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
             Value x = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.and_agg_if(b.i32(0), b.cmp_eq(x, b.i32(0)), dst);
         },
         PASS_ALL, R"FOO(
def func(n, x, dst):
    acc = -1
    for i in range(n):
        if x[i] == 0:
            acc = acc & 0
    dst[0] = acc
        )FOO"},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.or_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.or_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.xor_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.xor_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        b.andnot_agg_if(x, b.is_positive(x), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        b.andnot_agg_if(x, b.is_positive(x), dst);
    }},
    // splat i1
    Test([](FunctionBuilder &b) {
        Predicate x = b.input_predicate_splat_arg();
        b.output_arg(x);
    }),
    Test([](FunctionBuilder &b) {
        Predicate x = b.input_predicate_splat_arg();
        b.output_arg(b.bit2bool(x));
    }),
    // i1 select rewrite
    Test{[](FunctionBuilder &b) {
        Predicate cond = b.input_predicate_arg();
        Predicate x = b.input_predicate_arg();
        Predicate y = b.input_predicate_arg();
        Predicate s1 = b.select(cond, x, y);
        Predicate s2 = b.select(cond, x, y);
        b.output_arg(b.and_(s1, s2));
    }},
    // i1 cond_store rewrite
    Test{[](FunctionBuilder &b) {
        Predicate x = b.input_predicate_arg();
        Predicate cond = b.input_predicate_arg();
        b.cond_store(x, cond, b.arg(I1));
    }},
    // A64 vector gather coverage for a transformed result and an aggregate consumer.
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.store(b.add(b.gather(idx, x), b.i32(3)), dst);
         },
         PASS_ALL, R"FOO(
def func(n, x, idx, dst):
    for i in range(n):
        dst[i] = x[idx[i]] + 3
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value idx = b.input_arg(I32);
             Argument dst = b.arg(I32);
             b.sum(b.gather(idx, x), dst);
         },
         PASS_ALL, R"FOO(
def func(n, x, idx, dst):
    acc = 0
    for i in range(n):
        acc += x[idx[i]]
    dst[0] = acc
        )FOO"},
    Test{[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value y = b.input_arg(I32);
        b.store(b.trunc_checked(y, I16), b.arg(I16));
        b.store(b.trunc_checked(y, I8), b.arg(I8));
    }},
};
