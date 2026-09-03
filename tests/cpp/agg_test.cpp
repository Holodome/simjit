// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;

std::vector<Test> agg_tests{
    // I32 grouped
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_sum(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] += input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_product(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] *= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_min(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] = min(ht[idxs[i]], input[i])
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_max(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] = max(ht[idxs[i]], input[i])
    )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_umin(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_umax(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_and(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] &= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_or(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] |= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_xor(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] ^= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_andnot(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] &= ~input[i]
    )FOO"},
    // I64 grouped
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_sum(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] += input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_product(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] *= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_min(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] = min(ht[idxs[i]], input[i])
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_max(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] = max(ht[idxs[i]], input[i])
    )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I64);
        Value input = b.input_arg(I64);
        b.grouped_umin(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I64);
        Value input = b.input_arg(I64);
        b.grouped_umax(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_and(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] &= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_or(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] |= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_xor(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] ^= input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I64);
             Value idxs = b.input_arg(I64);
             Value input = b.input_arg(I64);
             b.grouped_andnot(input, idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input):
    for i in range(n):
        ht[idxs[i]] &= ~input[i]
    )FOO"},
    // I64 with I32 indexes
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_sum(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_product(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_min(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_max(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_umin(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_umax(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_and(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_or(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_xor(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I64);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I64);
        b.grouped_andnot(input, idxs, ht);
    }},
    // I32 conditional grouped
    Test{[](FunctionBuilder &b) {
             Argument ht = b.arg(I32);
             Value idxs = b.input_arg(I32);
             Value input = b.input_arg(I32);
             b.grouped_sum_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
         },
         PASS_ALL, R"FOO(
def func(n, ht, idxs, input, cond):
    for i in range(n):
        if cond[i] & 1:
            ht[idxs[i]] += input[i]
    )FOO"},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_product_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_min_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_max_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_umin_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_umax_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_and_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_or_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_xor_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.input_arg(I32);
        Value input = b.input_arg(I32);
        b.grouped_andnot_if(input, b.bit_test(b.input_arg(I8), b.i8(1)), idxs, ht);
    }},
    // few groups (mainly for benchmark)
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.and_(b.input_arg(I32), b.i32(3));
        Value input = b.input_arg(I32);
        b.grouped_sum(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Value idxs = b.and_(b.input_arg(I32), b.i32(3));
        Value input = b.input_arg(I32);
        b.sum_if(input, b.cmp_eq(idxs, b.i32(0)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(1)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(2)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(3)), b.arg(I32));
    }},
    Test{[](FunctionBuilder &b) {
        Argument ht = b.arg(I32);
        Value idxs = b.and_(b.input_arg(I32), b.i32(7));
        Value input = b.input_arg(I32);
        b.grouped_sum(input, idxs, ht);
    }},
    Test{[](FunctionBuilder &b) {
        Value idxs = b.and_(b.input_arg(I32), b.i32(7));
        Value input = b.input_arg(I32);
        b.sum_if(input, b.cmp_eq(idxs, b.i32(0)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(1)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(2)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(3)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(4)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(5)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(6)), b.arg(I32));
        b.sum_if(input, b.cmp_eq(idxs, b.i32(7)), b.arg(I32));
    }},
    // stuff from int tests but grouped

    // two aggs
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I32);
        Argument x_dst = b.arg(I32);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I32);
        Argument x_dst = b.arg(I32);
        Value y = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I16);
        Argument y_dst = b.arg(I32);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(b.sext(y, I32), idx, y_dst);
    }},
    // agg and expr
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        b.grouped_sum(x, idx, x_dst);
        b.store(b.add(y, z), y_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        b.grouped_sum(x, idx, x_dst);
        b.store(b.add(y, z), y_dst);
    }},
    // three aggs
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I64);
        Argument y_dst = b.arg(I64);
        Value z = b.input_arg(I64);
        Argument z_dst = b.arg(I64);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
        b.grouped_sum(z, idx, z_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        Value z = b.input_arg(I32);
        Argument z_dst = b.arg(I32);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
        b.grouped_sum(z, idx, z_dst);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value x = b.input_arg(I64);
        Argument x_dst = b.arg(I64);
        Value y = b.input_arg(I32);
        Argument y_dst = b.arg(I32);
        Value z = b.input_arg(I16);
        Argument z_dst = b.arg(I32);
        b.grouped_sum(x, idx, x_dst);
        b.grouped_sum(y, idx, y_dst);
        b.grouped_sum(b.sext(z, I32), idx, z_dst);
    }},
    // tpch q1
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_extendedprice = b.input_arg(F64);
        Value l_discount = b.input_arg(F64);
        Argument sum_disc_price = b.arg(F64);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.f64(1), l_discount)), idx, sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_extendedprice = b.input_arg(I64);
        Value l_discount = b.input_arg(I64);
        Argument sum_disc_price = b.arg(I64);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), idx, sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_quantity = b.input_arg(I64);
        Value l_extendedprice = b.input_arg(I64);
        Value l_discount = b.input_arg(I64);
        Argument sum_qty = b.arg(I64);
        Argument sum_base_price = b.arg(I64);
        Argument sum_disc_price = b.arg(I64);
        b.grouped_sum(l_quantity, idx, sum_qty);
        b.grouped_sum(l_extendedprice, idx, sum_base_price);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), idx, sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_quantity = b.input_arg(I64);
        Value l_extendedprice = b.input_arg(I64);
        Value l_discount = b.sext(b.input_arg(I32), I64);
        Argument sum_qty = b.arg(I64);
        Argument sum_base_price = b.arg(I64);
        Argument sum_disc_price = b.arg(I64);
        b.grouped_sum(l_quantity, idx, sum_qty);
        b.grouped_sum(l_extendedprice, idx, sum_base_price);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), idx, sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_quantity = b.sext(b.input_arg(I16), I64);
        Value l_extendedprice = b.input_arg(I64);
        Value l_discount = b.sext(b.input_arg(I32), I64);
        Argument sum_qty = b.arg(I64);
        Argument sum_base_price = b.arg(I64);
        Argument sum_disc_price = b.arg(I64);
        b.grouped_sum(l_quantity, idx, sum_qty);
        b.grouped_sum(l_extendedprice, idx, sum_base_price);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), idx, sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_quantity = b.sext(b.input_arg(I16), I32);
        Value l_extendedprice = b.trunc(b.input_arg(I64), I32);
        Value l_discount = b.input_arg(I32);
        Argument sum_qty = b.arg(I32);
        Argument sum_base_price = b.arg(I32);
        Argument sum_disc_price = b.arg(I32);
        b.grouped_sum(l_quantity, idx, sum_qty);
        b.grouped_sum(l_extendedprice, idx, sum_base_price);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.i32(1), l_discount)), idx, sum_disc_price);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_quantity = b.input_arg(I64);
        Value l_extendedprice = b.input_arg(I64);
        Value l_discount = b.input_arg(I64);
        Value l_tax = b.input_arg(I64);
        Argument sum_qty = b.arg(I64);
        Argument sum_base_price = b.arg(I64);
        Argument sum_disc_price = b.arg(I64);
        Argument sum_charge = b.arg(I64);
        b.grouped_sum(l_quantity, idx, sum_qty);
        b.grouped_sum(l_extendedprice, idx, sum_base_price);
        b.grouped_sum(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), idx, sum_disc_price);
        b.grouped_sum(b.mul(b.mul(l_extendedprice, b.sub(b.i64(1), l_discount)), b.add(b.i64(1), l_tax)), idx,
                      sum_charge);
    }},
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        Value l_quantity = b.input_arg(I64);
        Value l_extendedprice = b.input_arg(I32);
        Value l_discount = b.input_arg(I32);
        Value l_tax = b.input_arg(I32);
        Argument sum_qty = b.arg(I64);
        Argument sum_base_price = b.arg(I64);
        Argument sum_disc_price = b.arg(I64);
        Argument sum_charge = b.arg(I64);
        b.grouped_sum(l_quantity, idx, sum_qty);
        b.grouped_sum(b.sext(l_extendedprice, I64), idx, sum_base_price);
        b.grouped_sum(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)), idx,
                      sum_disc_price);
        b.grouped_sum(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                            b.sext(b.add(b.i32(1), l_tax), I64)),
                      idx, sum_charge);
    }},
    // tpch few groups
    Test{[](FunctionBuilder &b) {
        Value idx = b.input_arg(I32);
        idx = b.and_(idx, b.i32(3));
        Value l_extendedprice = b.input_arg(I32);
        Value l_discount = b.input_arg(I32);
        Value l_tax = b.input_arg(I32);
        Argument sum_disc_price = b.arg(I64);
        Argument sum_charge = b.arg(I64);
        b.grouped_sum(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)), idx,
                      sum_disc_price);
        b.grouped_sum(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                            b.sext(b.add(b.i32(1), l_tax), I64)),
                      idx, sum_charge);
    }},
    Test{[](FunctionBuilder &b) {
             Value idx = b.input_arg(I32);
             idx = b.and_(idx, b.i32(3));
             Value l_extendedprice = b.input_arg(I32);
             Value l_discount = b.input_arg(I32);
             Value l_tax = b.input_arg(I32);
             b.sum_if(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                      b.cmp_eq(idx, b.i32(0)), b.arg(I64));
             b.sum_if(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                      b.cmp_eq(idx, b.i32(1)), b.arg(I64));
             b.sum_if(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                      b.cmp_eq(idx, b.i32(2)), b.arg(I64));
             b.sum_if(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                      b.cmp_eq(idx, b.i32(3)), b.arg(I64));
             b.sum_if(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                            b.sext(b.add(b.i32(1), l_tax), I64)),
                      b.cmp_eq(idx, b.i32(0)), b.arg(I64));
             b.sum_if(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                            b.sext(b.add(b.i32(1), l_tax), I64)),
                      b.cmp_eq(idx, b.i32(1)), b.arg(I64));
             b.sum_if(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                            b.sext(b.add(b.i32(1), l_tax), I64)),
                      b.cmp_eq(idx, b.i32(2)), b.arg(I64));
             b.sum_if(b.mul(b.mul(b.sext(l_extendedprice, I64), b.sext(b.sub(b.i32(1), l_discount), I64)),
                            b.sext(b.add(b.i32(1), l_tax), I64)),
                      b.cmp_eq(idx, b.i32(3)), b.arg(I64));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // fusing aggregates
    // Here we compute sum + count
    Test{[](FunctionBuilder &b) {
        Argument dst1 = b.arg(I32);
        Argument dst2 = b.arg(I32);
        Value idx = b.input_arg(I32);
        idx = b.and_(idx, b.i32(511));

        Value arg = b.input_arg(I32);
        b.grouped_sum(arg, idx, dst1);
        b.grouped_sum(b.i32(1), idx, dst2);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I32);
        Value idx = b.input_arg(I32);
        idx = b.and_(idx, b.i32(511));
        Value cell = b.mul(idx, b.i32(2));

        Value arg = b.input_arg(I32);
        b.grouped_sum(arg, cell, dst);
        b.grouped_sum(b.i32(1), b.add(cell, b.i32(1)), dst);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst1 = b.arg(I64);
        Argument dst2 = b.arg(I64);
        Value idx = b.input_arg(I32);
        idx = b.and_(idx, b.i32(511));

        Value arg = b.input_arg(I64);
        b.grouped_sum(arg, idx, dst1);
        b.grouped_sum(b.i64(1), idx, dst2);
    }},
    Test{[](FunctionBuilder &b) {
        Argument dst = b.arg(I64);
        Value idx = b.input_arg(I32);
        idx = b.and_(idx, b.i32(511));
        Value cell = b.mul(idx, b.i32(2));

        Value arg = b.input_arg(I64);
        b.grouped_sum(arg, cell, dst);
        b.grouped_sum(b.i64(1), b.add(cell, b.i32(1)), dst);
    }},
};
