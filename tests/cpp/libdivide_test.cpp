// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;

std::vector<Test> libdivide_tests{
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.div(y, b.i16(3)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 3
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.div(y, b.i16(-7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -7
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.div(y, b.i16(8)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 8
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.div(y, b.i16(-8)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -8
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.div(y, b.i32(7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 7
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.div(y, b.i32(-7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -7
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.div(y, b.i32(1024)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 1024
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.div(y, b.i32(-1024)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -1024
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 7
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(-7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -7
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(3)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 3
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(-3)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -3
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(10000)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 10000
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(-10000)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -10000
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.div(y, b.i32(21321321)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 21321321
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.div(y, b.i64(21321321)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 21321321
    for i in range(n):
        num = y[i]
        q = abs(num) // abs(den)
        x[i] = -q if ((num < 0) ^ (den < 0)) else q
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.udiv(y, b.i32(37)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 37
    mask = (1 << 32) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.udiv(y, b.i64(65537)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 65537
    mask = (1 << 64) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.udiv(y, b.u64(7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 7
    mask = (1 << 64) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.umod(y, b.u64(1)), x);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64, simjit::ErrorKind::Unsupported,
                               simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.mod(y, b.i32(-37)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -37
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], den)
        x[i] = y[i] - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.mod(y, b.i16(-1)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.mod(y, b.i32(-1)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = 0
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.mod(y, b.i64(1024)), x);
         },
         LIMIT_ARM_VECTOR.structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64,
                                           simjit::ErrorKind::Unsupported,
                                           simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, y):
    den = 1024
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], den)
        x[i] = y[i] - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.umod(y, b.i32(37)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 37
    mask = (1 << 32) - 1
    for i in range(n):
        num = y[i] & mask
        q = num // den
        x[i] = num - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.umod(y, b.i64(65537)), x);
         },
         LIMIT_ARM_VECTOR.structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64,
                                           simjit::ErrorKind::Unsupported,
                                           simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, y):
    den = 65537
    mask = (1 << 64) - 1
    for i in range(n):
        num = y[i] & mask
        q = num // den
        x[i] = num - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             Value z = b.input_arg(I16);
             Value left = b.div(y, b.i16(3));
             Value right = b.div(z, b.i16(-7));
             b.store(b.add(b.sub(left, right), b.i16(7)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        left = tdiv(y[i], 3)
        right = tdiv(z[i], -7)
        x[i] = left - right + 7
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value left = b.div(y, b.i32(5));
             Value right = b.div(z, b.i32(-37));
             b.store(b.add(b.sub(left, right), b.i32(19)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        left = tdiv(y[i], 5)
        right = tdiv(z[i], -37)
        x[i] = left - right + 19
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x_quotient = b.arg(I32);
             Argument x_remainder = b.arg(I32);
             Argument y_quotient = b.arg(I32);
             Value x = b.input_arg(I32);
             Value y = b.input_arg(I32);
             Value divisor = b.i32(100);
             b.store(b.div(x, divisor), x_quotient);
             b.store(b.mod(x, divisor), x_remainder);
             b.store(b.div(y, divisor), y_quotient);
         },
         PASS_ALL,
         R"FOO(
def func(n, x_quotient, x_remainder, y_quotient, x, y):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        xq = tdiv(x[i], 100)
        x_quotient[i] = xq
        x_remainder[i] = x[i] - xq * 100
        y_quotient[i] = tdiv(y[i], 100)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x_quotient = b.arg(I16);
             Argument x_remainder = b.arg(I16);
             Argument y_quotient = b.arg(I16);
             Value x = b.input_arg(I16);
             Value y = b.input_arg(I16);
             Value divisor = b.u16(100);
             b.store(b.udiv(x, divisor), x_quotient);
             b.store(b.umod(x, divisor), x_remainder);
             b.store(b.udiv(y, divisor), y_quotient);
         },
         PASS_ALL,
         R"FOO(
def func(n, x_quotient, x_remainder, y_quotient, x, y):
    mask = (1 << 16) - 1
    for i in range(n):
        xu = x[i] & mask
        yu = y[i] & mask
        xq = xu // 100
        x_quotient[i] = xq
        x_remainder[i] = xu - xq * 100
        y_quotient[i] = yu // 100
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value a = b.input_arg(I32);
             Value b1 = b.input_arg(I32);
             Value c = b.input_arg(I32);
             Value d = b.input_arg(I32);
             Value tmp = b.add(b.div(a, b.i32(97)), b.div(b1, b.i32(-257)));
             tmp = b.sub(tmp, b.div(c, b.i32(1024)));
             tmp = b.add(tmp, b.div(d, b.i32(65537)));
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, a, b1, c, d):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        x[i] = tdiv(a[i], 97) + tdiv(b1[i], -257) - tdiv(c[i], 1024) + tdiv(d[i], 65537)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Value z = b.input_arg(I32);
             Value q = b.udiv(y, b.i32(37));
             Value r = b.umod(z, b.i32(37));
             b.store(b.add(b.sub(q, r), b.i32(11)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z):
    den = 37
    mask = (1 << 32) - 1
    for i in range(n):
        q = (y[i] & mask) // den
        num = z[i] & mask
        r = num - (num // den) * den
        x[i] = q - r + 11
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value w = b.input_arg(I64);
             Value left = b.div(y, b.i64(97));
             Value middle = b.div(z, b.i64(-257));
             Value right = b.div(w, b.i64(1024));
             b.store(b.add(left, b.sub(middle, right)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y, z, w):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        left = tdiv(y[i], 97)
        middle = tdiv(z[i], -257)
        right = tdiv(w[i], 1024)
        x[i] = left + middle - right
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value a = b.input_arg(I64);
             Value b1 = b.input_arg(I64);
             Value c = b.input_arg(I64);
             Value d = b.input_arg(I64);
             Value tmp = b.add(b.div(a, b.i64(97)), b.div(b1, b.i64(-257)));
             tmp = b.sub(tmp, b.div(c, b.i64(1024)));
             tmp = b.add(tmp, b.div(d, b.i64(65537)));
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, a, b1, c, d):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        x[i] = tdiv(a[i], 97) + tdiv(b1[i], -257) - tdiv(c[i], 1024) + tdiv(d[i], 65537)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             Value z = b.input_arg(I64);
             Value q = b.div(y, b.i64(-257));
             Value r = b.mod(z, b.i64(97));
             b.store(b.add(q, r), x);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64, simjit::ErrorKind::Unsupported,
                               simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, y, z):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], -257)
        q2 = tdiv(z[i], 97)
        r = z[i] - q2 * 97
        x[i] = q + r
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.udiv(y, b.u16(11)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 11
    mask = (1 << 16) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.udiv(y, b.u16(16)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 16
    mask = (1 << 16) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.mod(y, b.i16(5)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 5
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], den)
        x[i] = y[i] - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.mod(y, b.i16(-5)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = -5
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], den)
        x[i] = y[i] - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.umod(y, b.u16(11)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 11
    mask = (1 << 16) - 1
    for i in range(n):
        num = y[i] & mask
        q = num // den
        x[i] = num - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I16);
             Value y = b.input_arg(I16);
             b.store(b.umod(y, b.u16(16)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 16
    mask = (1 << 16) - 1
    for i in range(n):
        num = y[i] & mask
        q = num // den
        x[i] = num - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.udiv(y, b.u32(1024)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 1024
    mask = (1 << 32) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.mod(y, b.i32(37)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 37
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], den)
        x[i] = y[i] - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             b.store(b.umod(y, b.u32(1024)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 1024
    mask = (1 << 32) - 1
    for i in range(n):
        num = y[i] & mask
        q = num // den
        x[i] = num - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.udiv(y, b.u64(1024)), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    den = 1024
    mask = (1 << 64) - 1
    for i in range(n):
        x[i] = (y[i] & mask) // den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.mod(y, b.i64(-257)), x);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64, simjit::ErrorKind::Unsupported,
                               simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, y):
    den = -257
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], den)
        x[i] = y[i] - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value y = b.input_arg(I64);
             b.store(b.umod(y, b.u64(1024)), x);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64, simjit::ErrorKind::Unsupported,
                               simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, y):
    den = 1024
    mask = (1 << 64) - 1
    for i in range(n):
        num = y[i] & mask
        q = num // den
        x[i] = num - q * den
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value a = b.input_arg(I32);
             Value b1 = b.input_arg(I32);
             Value c = b.input_arg(I32);
             Value d = b.input_arg(I32);
             Value tmp = b.add(b.div(a, b.i32(-37)), b.udiv(b1, b.u32(37)));
             tmp = b.sub(tmp, b.mod(c, b.i32(37)));
             tmp = b.add(tmp, b.umod(d, b.u32(37)));
             b.store(tmp, x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, a, b1, c, d):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    mask = (1 << 32) - 1
    for i in range(n):
        div_q = tdiv(a[i], -37)
        udiv_q = (b1[i] & mask) // 37
        mod_q = tdiv(c[i], 37)
        mod_r = c[i] - mod_q * 37
        unum = d[i] & mask
        umod_r = unum - (unum // 37) * 37
        x[i] = div_q + udiv_q - mod_r + umod_r
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I64);
             Value a = b.input_arg(I64);
             Value b1 = b.input_arg(I64);
             Value c = b.input_arg(I64);
             Value d = b.input_arg(I64);
             Value tmp = b.add(b.div(a, b.i64(-257)), b.udiv(b1, b.u64(65537)));
             tmp = b.sub(tmp, b.mod(c, b.i64(97)));
             tmp = b.add(tmp, b.umod(d, b.u64(1024)));
             b.store(tmp, x);
         },
         test_meta()
             .limitation(TestVariant::ArmVector)
             .structured_error(TestVariant::ArmVector, simjit::ErrorModule::A64, simjit::ErrorKind::Unsupported,
                               simjit::ErrorSubKind::UnsupportedBackendFeature),
         R"FOO(
def func(n, x, a, b1, c, d):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    mask = (1 << 64) - 1
    for i in range(n):
        div_q = tdiv(a[i], -257)
        udiv_q = (b1[i] & mask) // 65537
        mod_q = tdiv(c[i], 97)
        mod_r = c[i] - mod_q * 97
        unum = d[i] & mask
        umod_r = unum - (unum // 1024) * 1024
        x[i] = div_q + udiv_q - mod_r + umod_r
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value num = b.input_arg(I32);
             Value raw_den = b.input_arg(I32);
             Value den = b.or_(b.and_(raw_den, b.i32(255)), b.i32(1));
             Value fast = b.div(num, b.i32(37));
             Value dynamic = b.div(num, den);
             b.store(b.sub(fast, dynamic), x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, num, raw_den):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        den = (raw_den[i] & 255) | 1
        x[i] = tdiv(num[i], 37) - tdiv(num[i], den)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value num = b.input_arg(I32);
             Value raw_den = b.input_arg(I32);
             Value den = b.or_(b.and_(raw_den, b.u32(255)), b.u32(1));
             Value fast_q = b.udiv(num, b.u32(37));
             Value dynamic_r = b.umod(num, den);
             b.store(b.add(fast_q, dynamic_r), x);
         },
         ONLY_SCALAR,
         R"FOO(
def func(n, x, num, raw_den):
    mask = (1 << 32) - 1
    for i in range(n):
        den = (raw_den[i] & 255) | 1
        unum = num[i] & mask
        x[i] = (unum // 37) + (unum - (unum // den) * den)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument x = b.arg(I32);
             Value y = b.input_arg(I32);
             Predicate positive = b.cmp_ge(y, b.i32(0));
             Value quotient = b.div(y, b.i32(8));
             Value remainder = b.mod(y, b.i32(16));
             b.store(b.select(positive, quotient, remainder), x);
         },
         PASS_ALL,
         R"FOO(
def func(n, x, y):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        q = tdiv(y[i], 8)
        r = y[i] - tdiv(y[i], 16) * 16
        x[i] = q if y[i] >= 0 else r
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I32);
             Predicate cond = b.cmp_gt(b.mod(y, b.i32(8)), b.i32(3));
             b.cond_store(b.div(y, b.i32(16)), cond, b.arg(I32));
         },
         PASS_ALL,
         R"FOO(
def func(n, y, dst):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        r = y[i] - tdiv(y[i], 8) * 8
        cond = r > 3
        if cond:
            dst[i] = tdiv(y[i], 16)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I32);
             Value q = b.div(y, b.i32(8));
             b.sum(b.sext(q, I64), b.arg(I64));
             b.sum_if(b.sext(b.mod(y, b.i32(16)), I64), b.cmp_gt(y, b.i32(0)), b.arg(I64));
             b.countif(b.cmp_eq(b.umod(y, b.u32(8)), b.i32(0)), b.arg(I64));
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, y, sum_q, sum_r, count_even):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    mask = (1 << 32) - 1
    for i in range(n):
        sum_q[0] += tdiv(y[i], 8)
        if y[i] > 0:
            sum_r[0] += y[i] - tdiv(y[i], 16) * 16
        unum = y[i] & mask
        if unum - (unum // 8) * 8 == 0:
            count_even[0] += 1
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument table = b.arg(I64);
             Value raw_idx = b.input_arg(I32);
             Value y = b.input_arg(I32);
             Value idx = b.and_(raw_idx, b.i32(15));
             Value q = b.div(y, b.i32(8));
             Predicate cond = b.cmp_ne(b.umod(y, b.u32(4)), b.i32(0));
             b.grouped_sum_if(b.sext(q, I64), cond, idx, table);
         },
         coefficient_range_limit(TestVariant::ArmVector),
         R"FOO(
def func(n, table, raw_idx, y):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    mask = (1 << 32) - 1
    for i in range(n):
        idx = raw_idx[i] & 15
        unum = y[i] & mask
        if unum - (unum // 4) * 4 != 0:
            table[idx] += tdiv(y[i], 8)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument src = b.arg(I32);
             Value raw_idx = b.input_arg(I32);
             Argument packed = b.arg(I32);
             Argument packed_size = b.arg(I64);
             Value idx = b.and_(raw_idx, b.i32(15));
             Value gathered = b.gather(idx, src);
             Value q = b.div(gathered, b.i32(8));
             Predicate cond = b.cmp_ne(b.umod(gathered, b.u32(16)), b.i32(0));
             b.pack(q, cond, packed, packed_size);
         },
         PASS_ALL,
         R"FOO(
def func(n, src, raw_idx, packed, packed_size):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    mask = (1 << 32) - 1
    c = 0
    for i in range(n):
        gathered = src[raw_idx[i] & 15]
        unum = gathered & mask
        if unum - (unum // 16) * 16 != 0:
            packed[c] = tdiv(gathered, 8)
            c += 1
    packed_size[0] = c
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I32);
             Value q = b.div(y, b.i32(8));
             b.scatter(q, b.index(I32), b.arg(I32));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, y, dst):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        dst[i] = tdiv(y[i], 8)
        )FOO"},
    Test{[](FunctionBuilder &b) {
             Value y = b.input_arg(I32);
             Value raw_idx = b.input_arg(I32);
             Value idx = b.and_(raw_idx, b.i32(15));
             Value q = b.div(y, b.i32(8));
             Predicate cond = b.cmp_gt(b.mod(y, b.i32(16)), b.i32(0));
             b.cond_scatter(q, idx, cond, b.arg(I32));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps),
         R"FOO(
def func(n, y, raw_idx, dst):
    def tdiv(num, den):
        q = abs(num) // abs(den)
        return -q if ((num < 0) ^ (den < 0)) else q
    for i in range(n):
        r = y[i] - tdiv(y[i], 16) * 16
        if r > 0:
            dst[raw_idx[i] & 15] = tdiv(y[i], 8)
        )FOO"},
};
