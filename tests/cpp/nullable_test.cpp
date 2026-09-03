// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/nullable.h"
#include "test.h"

using namespace simjit;
using namespace simjit::types;
using namespace simjit::nullable;

std::vector<Test> nullable_tests{
    // SQL stuff
    // simple stuff
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        x = nb.add(x, x);
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
        x = nb.add(x, x);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    // strict add two nullable exprs store bit
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue x = nb.add(left, right);
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.add(left, right);
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.add(left, right);
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
        NullableValue x = nb.add(left, right);
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
             NullableValue x = nb.add(left, right);
             nb.nbit_store(x, b.arg(I32), b.arg(I1));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
        NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
        NullableValue x = nb.add(left, right);
        nb.nbit_store(x, b.arg(I32), b.arg(I1));
    }},
    // strict add two nullable exprs store bool
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue x = nb.add(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.add(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.add(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
             NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
             NullableValue x = nb.add(left, right);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
             NullableValue x = nb.add(left, right);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
             NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
             NullableValue x = nb.add(left, right);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // strict add two nullable exprs store nvar
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue x = nb.add(left, right);
        nb.nval_store(x, b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue x = nb.add(left, right);
             nb.nval_store(x, b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue x = nb.add(left, right);
             nb.nval_store(x, b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
        NullableValue x = nb.add(left, right);
        nb.nval_store(x, b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
             NullableValue x = nb.add(left, right);
             nb.nval_store(x, b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
        NullableValue right = nb.nval_load(b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
        NullableValue x = nb.add(left, right);
        nb.nval_store(x, b.arg(I32), b.i32(int32_t(0xAAAAAAAA)));
    }},
    // strict add with one null and other not null
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = b.input_arg(I32);
        NullableValue x = nb.greatest(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.null_value(I32);
        NullableValue right = b.input_arg(I32);
        NullableValue x = nb.greatest(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = b.i32(123);
        NullableValue x = nb.greatest(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    // greatest/least
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.greatest(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.least(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    // add+sub+mul
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue e4 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.add(e1, e2);
        x = nb.sub(x, e3);
        x = nb.mul(x, e4);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue e2 = b.input_arg(I32);
        NullableValue e3 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue e4 = b.input_arg(I32);
        NullableValue x = nb.add(e1, e2);
        x = nb.sub(x, e3);
        x = nb.mul(x, e4);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = b.input_arg(I32);
        NullableValue e2 = b.input_arg(I32);
        NullableValue e3 = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue e4 = b.input_arg(I32);
        NullableValue x = nb.add(e1, e2);
        x = nb.sub(x, e3);
        x = nb.mul(x, e4);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    // and or
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue x = nb.and_(e1, e2);
        nb.nbool_store(x, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue x = nb.or_(e1, e2);
        nb.nbool_store(x, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue x = nb.and_(e1, e2);
        x = nb.and_(x, e3);
        nb.nbool_store(x, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue x = nb.and_(e1, e2);
        x = nb.or_(x, e3);
        nb.nbool_store(x, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue x = nb.or_(e1, e2);
        x = nb.or_(x, e3);
        nb.nbool_store(x, b.arg(I8), b.arg(I8));
    }},
    // same stuff but convert to logical for whatever reason
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullablePredicate p1 = nb.bool2bit(e1);
        NullablePredicate p2 = nb.bool2bit(e2);
        NullablePredicate x = nb.and_(p1, p2);
        NullableValue y = nb.bit2bool(x);
        nb.nbool_store(y, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullablePredicate p1 = nb.bool2bit(e1);
        NullablePredicate p2 = nb.bool2bit(e2);
        NullablePredicate x = nb.or_(p1, p2);
        NullableValue y = nb.bit2bool(x);
        nb.nbool_store(y, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullablePredicate p1 = nb.bool2bit(e1);
        NullablePredicate p2 = nb.bool2bit(e2);
        NullablePredicate p3 = nb.bool2bit(e3);
        NullablePredicate x = nb.and_(p1, p2);
        x = nb.and_(x, p3);
        NullableValue y = nb.bit2bool(x);
        nb.nbool_store(y, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullablePredicate p1 = nb.bool2bit(e1);
        NullablePredicate p2 = nb.bool2bit(e2);
        NullablePredicate p3 = nb.bool2bit(e3);
        NullablePredicate x = nb.and_(p1, p2);
        x = nb.or_(x, p3);
        NullableValue y = nb.bit2bool(x);
        nb.nbool_store(y, b.arg(I8), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue e1 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e2 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue e3 = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullablePredicate p1 = nb.bool2bit(e1);
        NullablePredicate p2 = nb.bool2bit(e2);
        NullablePredicate p3 = nb.bool2bit(e3);
        NullablePredicate x = nb.or_(p1, p2);
        x = nb.or_(x, p3);
        NullableValue y = nb.bit2bool(x);
        nb.nbool_store(y, b.arg(I8), b.arg(I8));
    }},
    // casts
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        x = nb.cast(x, I64);
        nb.nbit_store(x, b.arg(I64), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        x = nb.cast(x, I16);
        nb.nbit_store(x, b.arg(I16), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue y = nb.nbit_load(b.arg(I64), b.arg(I1));
        x = nb.cast(x, I64);
        y = nb.cast(y, I16);
        nb.nbit_store(x, b.arg(I64), b.arg(I1));
        nb.nbit_store(y, b.arg(I16), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue y = nb.nbit_load(b.arg(I64), b.arg(I1));
        x = nb.cast(x, I16);
        y = nb.cast(y, I16);
        nb.nbit_store(x, b.arg(I16), b.arg(I1));
        nb.nbit_store(y, b.arg(I16), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue y = nb.nbit_load(b.arg(I16), b.arg(I1));
        x = nb.cast(x, I64);
        y = nb.cast(y, I64);
        nb.nbit_store(x, b.arg(I64), b.arg(I1));
        nb.nbit_store(y, b.arg(I64), b.arg(I1));
    }},
    // negate
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbit_load(b.arg(I32), b.arg(I1));
        left = nb.negate(left);
        NullableValue x = nb.mul(left, right);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        left = nb.negate(left);
        NullableValue x = nb.mul(left, right);
        x = nb.negate(x);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue x = nb.mul(left, right);
        x = nb.negate(x);
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    // cmp with zero
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = b.i32(0);
        NullablePredicate x = nb.cmp_lt(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = b.i32(0);
        NullablePredicate x = nb.cmp_le(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = b.i32(0);
        NullablePredicate x = nb.cmp_gt(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = b.i32(0);
        NullablePredicate x = nb.cmp_ge(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = b.i32(0);
        NullablePredicate x = nb.cmp_eq(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = b.i32(0);
        NullablePredicate x = nb.cmp_ne(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    // cmp
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullablePredicate x = nb.cmp_ne(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_ne(left, right);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbit_load(b.arg(I32), b.arg(I1));
             NullablePredicate x = nb.cmp_ne(left, right);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbit_load(b.arg(I32), b.arg(I1));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullablePredicate x = nb.cmp_ne(left, right);
        nb.nbit_store(x, b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I64), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I64), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::VectorAll)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I16), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I16), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I8), b.arg(I8));
        NullablePredicate x = nb.cmp_lt(left, right);
        nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
    }},
    // combining cmp (god help me)
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(left, front);
             x = nb.and_(x, y);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.and_(x, y);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.or_(x, y);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue horizontal = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue vertical = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             NullablePredicate z = nb.cmp_lt(horizontal, vertical);
             x = nb.and_(x, y);
             x = nb.and_(x, z);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue horizontal = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue vertical = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             NullablePredicate z = nb.cmp_lt(horizontal, vertical);
             x = nb.or_(x, y);
             x = nb.or_(x, z);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate z = nb.nbool_load_predicate(b.arg(I1), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.and_(x, y);
             x = nb.and_(x, z);
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue z = nb.nbool_load(b.arg(I8), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.and_(x, y);
             x = nb.and_(x, nb.bool2bit(z));
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue z = nb.nbool_load(b.arg(I8), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.and_(x, y);
             x = nb.and_(x, nb.not_(nb.bool2bit(z)));
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue z = nb.nbool_load(b.arg(I8), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.and_(x, y);
             x = nb.or_(x, nb.not_(nb.bool2bit(z)));
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue front = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue back = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue z = nb.nbool_load(b.arg(I8), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             NullablePredicate y = nb.cmp_lt(front, back);
             x = nb.or_(x, y);
             x = nb.or_(x, nb.not_(nb.bool2bit(z)));
             nb.nbool_store(nb.bit2bool(x), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // is true is false is distinct etc
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             x = nb.is_true(x);
             nb.nbool_store(x, b.arg(I1), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             x = nb.is_false(x);
             nb.nbool_store(x, b.arg(I1), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             x = nb.is_not_true(x);
             nb.nbool_store(x, b.arg(I1), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.cmp_lt(left, right);
             x = nb.is_not_false(x);
             nb.nbool_store(x, b.arg(I1), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.is_distinct(left, right);
             nb.nbool_store(x, b.arg(I1), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullablePredicate x = nb.is_not_distinct(left, right);
             nb.nbool_store(x, b.arg(I1), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // nullif
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             left = nb.and_(left, {b.i32(7)});
             NullableValue x = nb.nullif(left, {b.i32(0)});
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // coalesce
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             left = nb.and_(left, {b.i32(7)});
             NullableValue x = nb.nullif(left, {b.i32(0)});
             NullableValue args[2]{x, b.i32(-1)};
             x = nb.coalesce(args);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue y = nb.nbool_load(b.arg(I32), b.arg(I8));
             left = nb.and_(left, {b.i32(7)});
             NullableValue x = nb.nullif(left, {b.i32(0)});
             NullableValue args[3]{x, y, b.i32(-1)};
             x = nb.coalesce(args);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // if_else
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue x = nb.if_else({nb.is_not_null(left)}, left, right);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue x = nb.if_else(nb.cmp_lt(left, right), left, right);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue chained = nb.if_else(nb.cmp_gt(right, {b.i32(100)}), right, nb.negate(right));
             NullableValue x = nb.if_else(nb.cmp_lt(left, right), left, chained);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // array helpers
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue args[3] = {
            nb.nbool_load(b.arg(I32), b.arg(I8)),
            nb.nbool_load(b.arg(I32), b.arg(I8)),
            b.input_arg(I32),
        };
        nb.nbool_store(nb.greatest(args), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue args[3] = {
            nb.nbool_load(b.arg(I32), b.arg(I8)),
            nb.nbool_load(b.arg(I32), b.arg(I8)),
            b.i32(17),
        };
        nb.nbool_store(nb.least(args), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullablePredicate args[3] = {
                 nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(0)}),
                 nb.cmp_gt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(10)}),
                 nb.true_unless_null(nb.nbool_load(b.arg(I32), b.arg(I8))),
             };
             nb.nbool_store(nb.bit2bool(nb.and_(args)), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullablePredicate args[3] = {
                 nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(0)}),
                 nb.cmp_gt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(10)}),
                 nb.true_unless_null(nb.nbool_load(b.arg(I32), b.arg(I8))),
             };
             nb.nbool_store(nb.bit2bool(nb.or_(args)), b.arg(I8), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // simple extra unary/binary coverage
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue x = nb.zeronull(nb.nbool_load(b.arg(I32), b.arg(I8)));
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.zeronull(NullableValue{b.input_arg(I32)});
        nb.nbool_store(x, b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             Value idx = b.input_arg(I32);
             NullableValue x = nb.nbool_gather(b.arg(I32), b.arg(I8), idx);
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         test_meta()
             .limitation(TestVariant::VectorAll)
             .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::UnsupportedSpecialOps)
             .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             Value idx = b.input_arg(I32);
             NullableValue x = nb.nval_gather(b.arg(I32), idx, b.i32(-1));
             nb.nbool_store(x, b.arg(I32), b.arg(I8));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector,
                                                simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.div(left, right), b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.mod(left, right), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.div(left, right, true), b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.mod(left, right, true), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        nb.nbool_store(nb.abs(nb.nbool_load(b.arg(I32), b.arg(I8))), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I16), b.arg(I8));
        nb.nbool_store(nb.signed_cast(x, I32), b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.unsigned_cast(x, I32), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I16), b.arg(I8));
        nb.nbool_store(nb.sext(x, I64), b.arg(I64), b.arg(I8));
        nb.nbool_store(nb.zext(x, I64), b.arg(I64), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I16), b.arg(I8));
        nb.nbool_store(nb.int_cast(x, I64, IntCastKind::Sext), b.arg(I64), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.trunc(x, I16), b.arg(I16), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbool_store(nb.float_cast(x, F32), b.arg(F32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbit_store(nb.bit_test(left, right), b.arg(I1), b.arg(I1));
        nb.nbit_store(nb.bit_testn(left, right), b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbit_store(nb.is_positive(x), b.arg(I1), b.arg(I1));
        nb.nbit_store(nb.is_negative(x), b.arg(I1), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
        nb.nbit_store(nb.is_null(x), b.arg(I1), b.arg(I1));
        nb.nbit_store(nb.is_not_null(x), b.arg(I1), b.arg(I1));
        nb.nbool_store(nb.bit2bool(nb.true_unless_null(x)), b.arg(I8), b.arg(I8));
    }},
    // distinctness with statically not-null operands
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue right = b.input_arg(I32);
             nb.nbit_store(nb.is_distinct(left, right), b.arg(I1), b.arg(I1));
             nb.nbit_store(nb.is_not_distinct(left, right), b.arg(I1), b.arg(I1));
             nb.nbit_store(nb.is_distinct(b.i32(7), left), b.arg(I1), b.arg(I1));
             nb.nbit_store(nb.is_not_distinct(b.i32(7), left), b.arg(I1), b.arg(I1));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue left = b.input_arg(I32);
             NullableValue right = nb.nbool_load(b.arg(I32), b.arg(I8));
             nb.nbit_store(nb.is_distinct(left, right), b.arg(I1), b.arg(I1));
             nb.nbit_store(nb.is_not_distinct(left, right), b.arg(I1), b.arg(I1));
             nb.nbit_store(nb.is_distinct(b.i32(7), right), b.arg(I1), b.arg(I1));
             nb.nbit_store(nb.is_not_distinct(b.i32(7), right), b.arg(I1), b.arg(I1));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue left = b.input_arg(I32);
        NullableValue right = b.input_arg(I32);
        nb.nbit_store(nb.is_distinct(left, right), b.arg(I1), b.arg(I1));
        nb.nbit_store(nb.is_not_distinct(left, right), b.arg(I1), b.arg(I1));
        nb.nbit_store(nb.is_distinct(b.i32(7), right), b.arg(I1), b.arg(I1));
        nb.nbit_store(nb.is_not_distinct(b.i32(7), right), b.arg(I1), b.arg(I1));
    }},
    // if_else variations
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullablePredicate cond = nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(10)});
             NullableValue truthy = nb.nbool_load(b.arg(I32), b.arg(I8));
             nb.nbool_store(nb.if_else_null(cond, truthy), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullablePredicate cond = nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(10)});
             NullableValue falsy = nb.nbool_load(b.arg(I32), b.arg(I8));
             nb.nbool_store(nb.if_else(cond, b.i32(42), falsy), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullablePredicate cond = nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(10)});
             NullableValue truthy = nb.nbool_load(b.arg(I32), b.arg(I8));
             nb.nbool_store(nb.if_else(cond, truthy, b.i32(-5)), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // zero-argument error handling
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<NullableValue> args{};
             nb.nbool_store(nb.greatest(nonstd::span<const NullableValue>(args)), b.arg(I32), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<NullableValue> args{};
             nb.nbool_store(nb.least(nonstd::span<const NullableValue>(args)), b.arg(I32), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<NullablePredicate> args{};
             nb.nbool_store(nb.bit2bool(nb.and_(nonstd::span<const NullablePredicate>(args))), b.arg(I8), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<NullablePredicate> args{};
             nb.nbool_store(nb.bit2bool(nb.or_(nonstd::span<const NullablePredicate>(args))), b.arg(I8), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<NullableValue> args{};
             nb.nbool_store(nb.coalesce(nonstd::span<const NullableValue>(args)), b.arg(I32), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<std::pair<NullableValue, NullableValue>> cases{};
             nb.nbool_store(
                 nb.case_x_when(b.i32(0), nonstd::span<const std::pair<NullableValue, NullableValue>>(cases)),
                 b.arg(I32), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::vector<std::pair<NullablePredicate, NullableValue>> cases{};
             nb.nbool_store(nb.case_when(nonstd::span<const std::pair<NullablePredicate, NullableValue>>(cases)),
                            b.arg(I32), b.arg(I8));
         },
         PASS_ALL,
         {},
         EXPECT_INVALID_INPUT},
    // coalesce / case helpers
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue args[3] = {nb.nbool_load(b.arg(I32), b.arg(I8)), b.i32(5),
                                      nb.nbool_load(b.arg(I32), b.arg(I8))};
             nb.nbool_store(nb.coalesce(args), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue args[1] = {nb.nbool_load(b.arg(I32), b.arg(I8))};
        nb.nbool_store(nb.coalesce(args), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue args[3] = {b.input_arg(I32), nb.nbool_load(b.arg(I32), b.arg(I8)),
                                 nb.nbool_load(b.arg(I32), b.arg(I8))};
        nb.nbool_store(nb.coalesce(args), b.arg(I32), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::pair<NullableValue, NullableValue> cases[2] = {
                 {{b.i32(1)}, nb.nbool_load(b.arg(I32), b.arg(I8))},
                 {{b.i32(2)}, b.i32(20)},
             };
             nb.nbool_store(nb.case_x_when(nb.nbool_load(b.arg(I32), b.arg(I8)), cases, b.i32(-1)), b.arg(I32),
                            b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::pair<NullableValue, NullableValue> cases[2] = {
                 {{b.i32(1)}, nb.nbool_load(b.arg(I32), b.arg(I8))},
                 {{b.i32(2)}, b.i32(20)},
             };
             nb.nbool_store(nb.case_x_when(nb.nbool_load(b.arg(I32), b.arg(I8)), cases), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::pair<NullablePredicate, NullableValue> cases[2] = {
                 {nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(0)}), b.i32(-1)},
                 {nb.cmp_gt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(100)}), nb.nbool_load(b.arg(I32), b.arg(I8))},
             };
             nb.nbool_store(nb.case_when(cases, b.i32(0)), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             std::pair<NullablePredicate, NullableValue> cases[2] = {
                 {nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(0)}), b.i32(-1)},
                 {nb.cmp_gt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(100)}), nb.nbool_load(b.arg(I32), b.arg(I8))},
             };
             nb.nbool_store(nb.case_when(cases), b.arg(I32), b.arg(I8));
         },
         coefficient_range_limit(TestVariant::ArmVector)},
    // stores and scatters with statically non-null values
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        nb.nbit_store(NullableValue{b.i32(7)}, b.arg(I32), b.arg(I1));
        nb.nbit_store(NullablePredicate{b.cmp_lt(b.input_arg(I32), b.i32(0))}, b.arg(I1), b.arg(I1));
        nb.nbool_store(NullableValue{b.i32(8)}, b.arg(I32), b.arg(I8));
        nb.nbool_store(NullablePredicate{b.cmp_lt(b.input_arg(I32), b.i32(0))}, b.arg(I1), b.arg(I8));
        nb.nval_store(NullableValue{b.i32(9)}, b.arg(I32), b.i32(-1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullablePredicate p = nb.cmp_lt(nb.nbool_load(b.arg(I32), b.arg(I8)), {b.i32(0)});
        nb.nbool_store(p, b.arg(I1), b.arg(I8));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        Value idx = b.input_arg(I32);
        nb.nbool_scatter(NullableValue{b.input_arg(I32)}, b.arg(I32), b.arg(I8), idx);
        nb.nval_scatter(NullableValue{b.input_arg(I32)}, b.arg(I32), idx, b.i32(-1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        Value idx = b.input_arg(I32);
        NullableValue bit = nb.nbit_load_ext(b.arg(I32), b.arg(I1), false);
        NullableValue boolv = nb.nbool_load_ext(b.arg(I32), b.arg(I8), false);
        NullableValue bit_splat = nb.nbit_load_splat_ext(b.arg(I32), b.arg(I1), false);
        NullableValue bool_splat = nb.nbool_load_splat_ext(b.arg(I32), b.arg(I8), false);
        NullableValue gathered = nb.nbool_gather_ext(b.arg(I32), b.arg(I8), idx, false);
        NullablePredicate bit_pred = nb.nbit_load_predicate_ext(b.arg(I1), b.arg(I1), false);
        NullablePredicate bool_pred = nb.nbool_load_predicate_ext(b.arg(I1), b.arg(I8), false);
        NullablePredicate bit_pred_splat = nb.nbit_load_predicate_splat_ext(b.arg(I1), b.arg(I1), false);
        NullablePredicate bool_pred_splat = nb.nbool_load_predicate_splat_ext(b.arg(I1), b.arg(I8), false);
        nb.nbit_store_ext(nb.add(bit, bit_splat), b.arg(I32), b.arg(I1), false);
        nb.nbool_store_ext(nb.add(boolv, bool_splat), b.arg(I32), b.arg(I8), false);
        nb.nbool_scatter_ext(gathered, b.arg(I32), b.arg(I8), idx, false);
        nb.nbit_store_ext(nb.and_(bit_pred, bit_pred_splat), b.arg(I1), b.arg(I1), false);
        nb.nbool_store_ext(nb.or_(bool_pred, bool_pred_splat), b.arg(I1), b.arg(I8), false);
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             Value idx = b.input_arg(I32);
             NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
             nb.nbool_scatter(x, b.arg(I32), b.arg(I8), idx);
             nb.nval_scatter(x, b.arg(I32), idx, b.i32(-1));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // aggregate helpers
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue arg = nb.nbool_load(b.arg(I32), b.arg(I8));
        NullablePredicate pred = nb.cmp_gt(arg, {b.i32(10)});
        Value idx = b.input_arg(I32);
        nb.sum(arg, b.arg(I32));
        nb.min_agg(arg, b.arg(I32));
        nb.max_agg(arg, b.arg(I32));
        nb.grouped_sum(arg, idx, b.arg(I32));
        nb.grouped_min(arg, idx, b.arg(I32));
        nb.grouped_max(arg, idx, b.arg(I32));
        nb.countif(pred, b.arg(I64));
        nb.count_notnull(arg, b.arg(I64));
        nb.has_nulls(arg, b.arg(I1));
        nb.all(pred, b.arg(I1));
        nb.any(pred, b.arg(I1));
        nb.find_true_indices(pred, b.arg(I32), b.arg(I64));
        nb.update_true_indices(idx, pred, b.arg(I32), b.arg(I64));
        nb.find_notnull_indices(arg, b.arg(I32), b.arg(I64));
        nb.find_null_indices(arg, b.arg(I32), b.arg(I64));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue arg = b.input_arg(I32);
        NullablePredicate pred = nb.true_unless_null(arg);
        Value idx = b.input_arg(I32);
        nb.sum(arg, b.arg(I32));
        nb.grouped_sum(arg, idx, b.arg(I32));
        nb.countif(pred, b.arg(I64));
        nb.count_notnull(arg, b.arg(I64));
        nb.has_nulls(arg, b.arg(I1));
        nb.find_notnull_indices(arg, b.arg(I32), b.arg(I64));
        nb.find_null_indices(arg, b.arg(I32), b.arg(I64));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        NullableValue arg = b.input_arg(I32);
        Value idx = b.input_arg(I32);
        nb.min_agg(arg, b.arg(I32));
        nb.max_agg(arg, b.arg(I32));
        nb.grouped_min(arg, idx, b.arg(I32));
        nb.grouped_max(arg, idx, b.arg(I32));
    }},
    // complex combined cases
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        Value idx = b.input_arg(I32);
        NullableValue gathered = nb.nbool_gather(b.arg(I32), b.arg(I8), idx);
        NullableValue fallback = nb.if_else(nb.cmp_gt(gathered, {b.i32(100)}), gathered, b.i32(100));
        NullableValue args[3] = {nb.nullif(gathered, {b.i32(0)}), b.i32(7), fallback};
        NullableValue result = nb.coalesce(args);
        nb.nbool_scatter(result, b.arg(I32), b.arg(I8), idx);
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
             NullableValue y = nb.nval_gather(b.arg(I32), b.input_arg(I32), b.i32(-1));
             std::pair<NullablePredicate, NullableValue> cases[2] = {
                 {nb.is_negative(x), nb.abs(x)},
                 {nb.is_positive(y), nb.if_else_null(nb.true_unless_null(y), y)},
             };
             NullableValue result = nb.case_when(cases, nb.greatest(x, y));
             nb.nval_store(result, b.arg(I32), b.i32(-1));
         },
         LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector,
                                                simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    // Invalid API usage
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullableValue x = nb.greatest({});
            nb.nbool_store(x, b.arg(I32), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullableValue x = nb.least({});
            nb.nbool_store(x, b.arg(I32), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullablePredicate x = nb.and_({});
            nb.nbool_store(x, b.arg(I1), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullablePredicate x = nb.or_({});
            nb.nbool_store(x, b.arg(I1), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullableValue x = nb.coalesce({});
            nb.nbool_store(x, b.arg(I32), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullableValue x = nb.case_when({});
            nb.nbool_store(x, b.arg(I32), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            NullableValue right = b.input_arg(I32);
            NullableValue x = nb.case_x_when(right, {});
            nb.nbool_store(x, b.arg(I32), b.arg(I8));
        },
        PASS_ALL,
        {},
        EXPECT_INVALID_INPUT,
    },
// Checked operations
#define checked_op(_ty, _op)                                       \
    Test {                                                         \
        [](FunctionBuilder &b) {                                   \
            NullableBuilder nb(&b);                                \
            b.arg_safety_check();                                  \
            NullableValue x = nb.nbit_load(b.arg(_ty), b.arg(I1)); \
            x = nb.arith_binary_checked(x, x, _op);                \
            nb.nbit_store(x, b.arg(_ty), b.arg(I1));               \
        }                                                          \
    }
    checked_op(ScalarDataType::I8, ArithBinaryOp::Add),
    checked_op(ScalarDataType::I16, ArithBinaryOp::Add),
    checked_op(ScalarDataType::I32, ArithBinaryOp::Add),
    checked_op(ScalarDataType::I64, ArithBinaryOp::Add),
    checked_op(ScalarDataType::I8, ArithBinaryOp::Sub),
    checked_op(ScalarDataType::I16, ArithBinaryOp::Sub),
    checked_op(ScalarDataType::I32, ArithBinaryOp::Sub),
    checked_op(ScalarDataType::I64, ArithBinaryOp::Sub),
    checked_op(ScalarDataType::I8, ArithBinaryOp::Mul),
    checked_op(ScalarDataType::I16, ArithBinaryOp::Mul),
#undef checked_op
    // Didn't add i32 and i64 because they might have false positives and i64 does not work on arm
    // checked_op(ScalarDataType::I32, ArithBinaryOp::Mul),
    // checked_op(ScalarDataType::I64, ArithBinaryOp::Mul),
    // trunc checked
    Test{
        [](FunctionBuilder &b) {
            NullableBuilder nb(&b);
            b.arg_safety_check();
            NullableValue x = nb.nbool_load(b.arg(I32), b.arg(I8));
            nb.nbool_store(nb.trunc_checked(x, I16), b.arg(I16), b.arg(I8));
        },
        LIMIT_ALL_VECTOR.vectorization_failure(TestVariant::VectorAll,
                                               simjit::ErrorSubKind::CoefficientRangeNeedsNormalization),
    },
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        b.arg_safety_check();
        NullableValue x = nb.nbit_load(b.arg(I32), b.arg(I1));
        nb.nbit_store(nb.trunc_checked(x, I16), b.arg(I16), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
             NullableBuilder nb(&b);
             b.arg_safety_check();
             NullableValue x = nb.nbool_load(b.arg(I64), b.arg(I8));
             nb.nbool_store(nb.trunc_checked(x, I16), b.arg(I16), b.arg(I8));
         },
         LIMIT_ALL_VECTOR.vectorization_failure(TestVariant::VectorAll,
                                                simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        b.arg_safety_check();
        NullableValue x = nb.nbit_load(b.arg(I64), b.arg(I1));
        nb.nbit_store(nb.trunc_checked(x, I16), b.arg(I16), b.arg(I1));
    }},
    Test{[](FunctionBuilder &b) {
        NullableBuilder nb(&b);
        b.arg_safety_check();
        NullableValue x = nb.nbit_load(b.arg(I64), b.arg(I1));
        nb.nbit_store(nb.trunc_checked(x, I8), b.arg(I8), b.arg(I1));
    }},
#define checked_op(_op, _ty)                                       \
    Test {                                                         \
        [](FunctionBuilder &b) {                                   \
            NullableBuilder nb(&b);                                \
            b.arg_safety_check();                                  \
            NullableValue x = nb.nbit_load(b.arg(_ty), b.arg(I1)); \
            nb.nbit_store(nb._op(x), b.arg(_ty), b.arg(I1));       \
        }                                                          \
    }
    // checked_op(abs_checked, I8),
    // checked_op(abs_checked, I16),
    checked_op(abs_checked, I32),
    // checked_op(abs_checked, I64),
    // checked_op(negate_checked, I8),
    // checked_op(negate_checked, I16),
    // checked_op(negate_checked, I32),
    // checked_op(negate_checked, I64),
};
