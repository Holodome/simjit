// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;
static void add_binary_identity_peephole_runtime_tests(std::vector<Test> &tests) {
    tests.push_back(Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        b.store(b.add(b.mul(x, b.i32(1)), b.mul(x, b.i32(0))), b.arg(I32));
    }});

    tests.push_back(Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Value mask = b.i32(-1);
        b.store(b.xor_(b.or_(b.and_(x, mask), b.i32(0)), b.i32(0)), b.arg(I32));
    }});

    tests.push_back(Test{[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate cond = b.and_(b.cmp_uge(x, b.i32(0)), b.true_());
        b.store(b.or_(b.xor_(cond, b.false_()), b.andnot(cond, b.false_())), b.arg(I1));
    }});
}

static void add_input_output_alias_tests(std::vector<Test> &tests) {
    add_valid(tests, [](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        b.store(b.load(x), x);
    });
    add_valid(tests, [](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        b.store(b.load_predicate(x), x);
    });
    add_valid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        b.store(b.load(dst), dst);
    });
    add_valid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        b.store(b.load_predicate(dst), dst);
    });
}

static void add_multiple_write_tests(std::vector<Test> &tests) {
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        b.store(b.input_arg(I32), dst);
        b.store(b.input_arg(I32), dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        b.store(b.input_predicate_arg(), dst);
        b.store(b.input_predicate_arg(), dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        b.sum(b.input_arg(I32), dst);
        b.store(b.input_arg(I32), dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        b.or_agg(b.input_predicate_arg(), dst);
        b.store(b.input_predicate_arg(), dst);
    });
}

static void add_memory_offset_falsification_tests(std::vector<Test> &tests) {
    add_valid(tests, [](FunctionBuilder &b) {
        Argument src = b.arg(I32);
        Argument dst = b.arg(I32);
        Value a = b.load(src, LoadStoreKind::Unaligned);
        Value b1 = b.load(src, LoadStoreKind::Aligned);
        b.store(b.add(a, b1), dst);
    });
    add_valid(tests, [](FunctionBuilder &b) {
        Argument src = b.arg(I32);
        Argument dst1 = b.arg(I32);
        Argument dst2 = b.arg(I32);
        Value x = b.load(src);
        b.store(x, dst1);
        b.store(b.add(x, b.i32(1)), dst2);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Predicate cond = b.input_predicate_arg();
        Value y = b.input_arg(I32);
        b.cond_store(x, cond, dst);
        b.store(y, dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value x = b.input_arg(I32);
        Predicate cond = b.input_predicate_arg();
        Value y = b.input_arg(I32);
        b.store(x, dst);
        b.cond_store(y, cond, dst);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(F64);
        b.store(b.input_arg(F64), dst);
        b.store(b.input_arg(F64), dst);
    });
}

static void add_cross_expression_alias_tests(std::vector<Test> &tests) {
    add_valid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        b.store(b.input_arg(I32), dst);
        b.output_arg(b.add(b.load(dst), b.i32(1)));
    });
    add_valid(tests, [](FunctionBuilder &b) {
        Argument dst = b.arg(I1);
        b.store(b.input_predicate_arg(), dst);
        b.output_arg(b.select(b.input_predicate_arg(), b.load_predicate(dst), b.false_()));
    });
}

static void add_grouped_table_alias_tests(std::vector<Test> &tests) {
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument table = b.arg(I32);
        b.grouped_sum(b.input_arg(I32), b.input_arg(I32), table);
        b.store(b.input_arg(I32), table);
    });
    add_invalid(tests, [](FunctionBuilder &b) {
        Argument table = b.arg(I32);
        b.store(b.input_arg(I32), table);
        b.grouped_sum(b.input_arg(I32), b.input_arg(I32), table);
    });
}

static std::vector<Test> make_misc_tests() {
    std::vector<Test> tests;
    add_binary_identity_peephole_runtime_tests(tests);
    add_input_output_alias_tests(tests);
    add_multiple_write_tests(tests);
    add_memory_offset_falsification_tests(tests);
    add_cross_expression_alias_tests(tests);
    add_grouped_table_alias_tests(tests);
    return tests;
}

std::vector<Test> misc_tests = make_misc_tests();
