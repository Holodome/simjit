// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;

template <typename MakeExpr>
static void add_ternarylogic_int_case(std::vector<Test> &tests, MakeExpr make_expr,
                                      const char *file = relative_test_file(__builtin_FILE()),
                                      int line = __builtin_LINE()) {
    tests.push_back(Test{
        [make_expr](FunctionBuilder &b) {
            Value a = b.input_arg(I32);
            Value x = b.input_arg(I32);
            Value y = b.input_arg(I32);
            b.store(make_expr(b, a, x, y), b.arg(I32));
        },
        PASS_ALL,
        {},
        EXPECT_SUCCESS,
        file,
        line,
    });
}

template <typename MakeExpr>
static void add_ternarylogic_int_case4(std::vector<Test> &tests, MakeExpr make_expr,
                                       const char *file = relative_test_file(__builtin_FILE()),
                                       int line = __builtin_LINE()) {
    tests.push_back(Test{
        [make_expr](FunctionBuilder &b) {
            Value a = b.input_arg(I32);
            Value x = b.input_arg(I32);
            Value y = b.input_arg(I32);
            Value z = b.input_arg(I32);
            b.store(make_expr(b, a, x, y, z), b.arg(I32));
        },
        PASS_ALL,
        {},
        EXPECT_SUCCESS,
        file,
        line,
    });
}

template <typename MakeExpr>
static void add_ternarylogic_int_case5(std::vector<Test> &tests, MakeExpr make_expr,
                                       const char *file = relative_test_file(__builtin_FILE()),
                                       int line = __builtin_LINE()) {
    tests.push_back(Test{
        [make_expr](FunctionBuilder &b) {
            Value a = b.input_arg(I32);
            Value x = b.input_arg(I32);
            Value y = b.input_arg(I32);
            Value z = b.input_arg(I32);
            Value w = b.input_arg(I32);
            b.store(make_expr(b, a, x, y, z, w), b.arg(I32));
        },
        PASS_ALL,
        {},
        EXPECT_SUCCESS,
        file,
        line,
    });
}

static std::vector<Test> make_ternarylogic_tests() {
    std::vector<Test> tests;

    add_ternarylogic_int_case(tests,
                              [](FunctionBuilder &b, Value a, Value x, Value y) { return b.or_(b.and_(a, x), y); });
    add_ternarylogic_int_case(tests,
                              [](FunctionBuilder &b, Value a, Value x, Value y) { return b.or_(y, b.and_(a, x)); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.or_(b.and_(a, x), b.and_(a, y)); });
    add_ternarylogic_int_case(tests,
                              [](FunctionBuilder &b, Value a, Value x, Value y) { return b.xor_(b.xor_(a, x), y); });
    add_ternarylogic_int_case(tests,
                              [](FunctionBuilder &b, Value a, Value x, Value y) { return b.andnot(y, b.and_(a, x)); });

    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.xor_(b.and_(a, x), b.or_(x, y)); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.and_(b.or_(a, x), b.xor_(a, y)); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.or_(b.xor_(a, x), b.and_(a, y)); });
    add_ternarylogic_int_case(tests,
                              [](FunctionBuilder &b, Value a, Value x, Value y) { return b.andnot(a, b.or_(x, y)); });
    add_ternarylogic_int_case(tests,
                              [](FunctionBuilder &b, Value a, Value x, Value y) { return b.andnot(b.and_(a, x), y); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.or_(b.and_(b.not_(a), x), y); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.xor_(b.or_(a, b.not_(x)), y); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.or_(b.not_(b.and_(a, x)), y); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.xor_(b.not_(b.or_(a, x)), y); });
    add_ternarylogic_int_case(
        tests, [](FunctionBuilder &b, Value a, Value x, Value y) { return b.and_(b.xor_(a, b.not_(x)), b.or_(x, y)); });

    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.and_(a, x);
        Value t1 = b.or_(t0, y);
        Value t2 = b.xor_(t1, a);
        return b.or_(b.and_(t2, x), b.andnot(y, t0));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.or_(a, x);
        Value t1 = b.xor_(x, y);
        Value t2 = b.and_(t0, t1);
        Value t3 = b.or_(b.and_(a, y), x);
        return b.xor_(t2, t3);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(a, x);
        Value t1 = b.xor_(a, y);
        Value t2 = b.and_(t0, t1);
        return b.or_(t2, b.andnot(y, x));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.andnot(a, x);
        Value t1 = b.andnot(y, a);
        Value t2 = b.or_(t0, t1);
        return b.xor_(t2, b.or_(a, y));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.or_(b.and_(a, x), b.and_(x, y));
        Value t1 = b.or_(b.and_(a, y), x);
        return b.xor_(t0, t1);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.or_(b.and_(a, b.not_(x)), b.and_(x, y));
        Value t1 = b.and_(b.or_(a, b.not_(x)), b.xor_(y, a));
        return b.xor_(t0, t1);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.andnot(b.not_(a), b.xor_(x, y));
        Value t1 = b.andnot(b.xor_(a, x), b.or_(x, y));
        return b.or_(t0, t1);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(b.andnot(a, x), b.or_(y, a));
        Value t1 = b.or_(b.andnot(a, x), b.andnot(y, a));
        return b.and_(t0, t1);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.and_(b.xor_(a, x), b.xor_(x, y));
        Value t1 = b.or_(b.xor_(a, x), b.xor_(a, y));
        return b.xor_(t0, t1);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(b.or_(b.and_(a, x), y), b.andnot(x, a));
        Value t1 = b.or_(b.and_(b.or_(a, x), y), b.xor_(a, x));
        return b.or_(t0, t1);
    });

    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        (void)y;
        Value t0 = b.xor_(b.xor_(a, x), x);
        Value t1 = b.or_(b.and_(a, x), b.and_(a, b.not_(x)));
        Value t2 = b.and_(b.or_(a, x), b.or_(a, b.not_(x)));
        return b.xor_(b.or_(t0, t1), t2);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(b.or_(a, x), b.and_(a, x));
        Value t1 = b.or_(b.and_(a, x), b.andnot(x, a));
        Value t2 = b.xor_(b.xor_(b.xor_(a, x), y), b.xor_(x, y));
        return b.and_(b.or_(t0, t1), b.or_(t2, a));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.or_(b.and_(a, x), b.and_(a, b.not_(x)));
        Value t1 = b.and_(b.or_(a, x), b.or_(a, b.not_(x)));
        Value t2 = b.xor_(b.xor_(t0, t1), b.xor_(a, y));
        return b.or_(b.and_(t2, x), b.andnot(y, t1));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(a, x);
        Value t1 = b.xor_(x, y);
        Value t2 = b.xor_(a, y);
        Value t3 = b.or_(b.and_(t0, t1), b.and_(t0, t2));
        return b.xor_(t3, b.and_(t1, t2));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.and_(a, x);
        Value t1 = b.and_(x, y);
        Value t2 = b.and_(a, y);
        Value t3 = b.or_(b.or_(t0, t1), t2);
        return b.xor_(t3, b.and_(t0, b.not_(t1)));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.or_(a, x);
        Value t1 = b.or_(x, y);
        Value t2 = b.or_(a, y);
        Value t3 = b.and_(b.and_(t0, t1), t2);
        return b.xor_(t3, b.or_(b.not_(t0), b.andnot(t1, t2)));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(a, x);
        Value t1 = b.xor_(t0, y);
        Value t2 = b.xor_(t1, a);
        Value t3 = b.xor_(t2, x);
        Value t4 = b.xor_(t3, y);
        return b.or_(b.and_(t4, a), b.andnot(x, t1));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.and_(a, x);
        Value t1 = b.or_(t0, y);
        Value t2 = b.xor_(t1, b.not_(a));
        Value t3 = b.andnot(t2, b.or_(x, y));
        Value t4 = b.xor_(t3, b.and_(t1, t2));
        return b.or_(t4, b.andnot(t0, t3));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.or_(a, x);
        Value t1 = b.and_(t0, y);
        Value t2 = b.xor_(t1, b.andnot(a, x));
        Value t3 = b.or_(t2, b.not_(t0));
        Value t4 = b.and_(t3, b.xor_(a, y));
        return b.xor_(t4, b.or_(t1, t2));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.xor_(a, x);
        Value t1 = b.and_(x, y);
        Value t2 = b.or_(t0, t1);
        Value t3 = b.andnot(t2, b.or_(a, y));
        Value t4 = b.xor_(t3, b.and_(t2, b.not_(t1)));
        return b.or_(b.and_(t4, t0), b.andnot(t1, t3));
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value t0 = b.and_(a, x);
        Value t1 = b.and_(t0, y);
        Value t2 = b.and_(t1, a);
        Value t3 = b.and_(t2, x);
        Value t4 = b.and_(t3, y);
        return b.or_(t4, b.xor_(t0, t3));
    });

    add_ternarylogic_int_case4(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z) {
        return b.or_(b.and_(a, x), b.and_(y, z));
    });
    add_ternarylogic_int_case4(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z) {
        return b.xor_(b.or_(b.and_(a, x), b.and_(y, z)), b.andnot(a, z));
    });
    add_ternarylogic_int_case4(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z) {
        Value t0 = b.or_(b.and_(a, x), b.and_(y, z));
        Value t1 = b.xor_(b.or_(a, y), b.and_(x, z));
        return b.and_(t0, t1);
    });
    add_ternarylogic_int_case4(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z) {
        Value t0 = b.xor_(a, x);
        Value t1 = b.xor_(y, z);
        Value t2 = b.or_(b.and_(t0, y), b.andnot(z, t1));
        return b.xor_(t2, b.or_(a, z));
    });
    add_ternarylogic_int_case4(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z) {
        Value t0 = b.and_(a, x);
        Value t1 = b.or_(y, z);
        Value t2 = b.xor_(t0, t1);
        Value t3 = b.andnot(b.or_(a, z), b.xor_(x, y));
        return b.or_(t2, t3);
    });
    add_ternarylogic_int_case4(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z) {
        Value t0 = b.or_(b.and_(a, b.not_(x)), b.and_(y, z));
        Value t1 = b.and_(b.or_(a, y), b.or_(x, z));
        Value t2 = b.xor_(t0, t1);
        return b.or_(b.and_(t2, a), b.andnot(z, t0));
    });
    add_ternarylogic_int_case5(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z, Value w) {
        return b.or_(b.xor_(b.and_(a, x), b.and_(y, z)), w);
    });
    add_ternarylogic_int_case5(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z, Value w) {
        Value t0 = b.and_(a, x);
        Value t1 = b.and_(y, z);
        Value t2 = b.or_(t0, t1);
        Value t3 = b.xor_(t2, w);
        return b.and_(t3, b.or_(a, z));
    });
    add_ternarylogic_int_case5(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z, Value w) {
        Value t0 = b.xor_(a, x);
        Value t1 = b.xor_(y, z);
        Value t2 = b.xor_(t0, t1);
        Value t3 = b.or_(b.and_(t2, w), b.andnot(z, a));
        return b.xor_(t3, b.or_(x, y));
    });
    add_ternarylogic_int_case5(tests, [](FunctionBuilder &b, Value a, Value x, Value y, Value z, Value w) {
        Value t0 = b.or_(a, x);
        Value t1 = b.and_(y, z);
        Value t2 = b.xor_(t0, t1);
        Value t3 = b.andnot(w, b.or_(t2, a));
        Value t4 = b.xor_(b.and_(z, w), b.or_(x, y));
        return b.or_(t3, t4);
    });
    add_ternarylogic_int_case(tests, [](FunctionBuilder &b, Value a, Value x, Value y) {
        Value ax = b.and_(a, x);
        b.store(ax, b.arg(I32));
        return b.or_(ax, y);
    });

    return tests;
}

std::vector<Test> ternarylogic_tests = make_ternarylogic_tests();
