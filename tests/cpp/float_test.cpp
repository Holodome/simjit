// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

#include <cstdlib>
#include <stdio.h>
#include <sys/stat.h>

using namespace simjit;
using namespace simjit::types;

std::vector<Test> float_tests{
    // constant zero
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         b.store(b.f32(0.0f), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 0.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         b.store(b.f64(0.0), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 0.0
        )FOO"},
    // index
    {[](FunctionBuilder &b) { b.store(b.index(F32), b.arg(F32)); }, PASS_ALL, R"FOO(
def func(n, dst):
    for i in range(n):
        dst[i] = float(i)
        )FOO"},
    {[](FunctionBuilder &b) { b.store(b.index(F64), b.arg(F64)); }, PASS_ALL, R"FOO(
def func(n, dst):
    for i in range(n):
        dst[i] = float(i)
        )FOO"},
    // other constants
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         b.store(b.f32(1.0f), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 1.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         b.store(b.f64(1.0), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 1.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         b.store(b.f32(2.0f), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 2.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         b.store(b.f64(2.0), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 2.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         b.store(b.f32(12345.0f), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 12345.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         b.store(b.f64(12345.0), x);
     },
     PASS_ALL, R"FOO(
def func(n, x):
    for i in range(n):
        x[i] = 12345.0
        )FOO"},
    // different kinds of zeros
    {[](FunctionBuilder &b) {
        // +0
        b.output_arg(b.con_internal(ConstData::u64(0x00000000), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x0000000000000000), F64));
        // -0
        b.output_arg(b.con_internal(ConstData::u64(0x80000000), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x8000000000000000), F64));
    }},
    // infs and qnan
    {[](FunctionBuilder &b) {
        // +inf
        b.output_arg(b.con_internal(ConstData::u64(0x7f800000u), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x7ff0000000000000llu), F64));
        // -inf
        b.output_arg(b.con_internal(ConstData::u64(0xff800000), F32));
        b.output_arg(b.con_internal(ConstData::u64(0xfff0000000000000llu), F64));
        // qnan
        // NAN is hard to insert into tests because it is not equal to itself...
        // b.output(b.con_internal((uint64_t)0x7fc00000u, F32));
        // b.output(b.con_internal((uint64_t)0x7ff8000000000000llu, F64));
    }},
    // denormals
    {[](FunctionBuilder &b) {
        // Smallest positive denormal
        b.output_arg(b.con_internal(ConstData::u64(0x00000001), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x0000000000000001), F64));
        // Smallest negative denormal
        b.output_arg(b.con_internal(ConstData::u64(0x80000001), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x8000000000000001), F64));
        // Largest positive denormal
        b.output_arg(b.con_internal(ConstData::u64(0x007fffff), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x000fffffffffffff), F64));
        // Smallest positive normal
        // This case is interesting because it gets printed as FLT_MIN/DBL_MIN
        b.output_arg(b.con_internal(ConstData::u64(0x00800000), F32));
        b.output_arg(b.con_internal(ConstData::u64(0x0010000000000000), F64));
    }},
    // var
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         b.store(y, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         b.store(y, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    // load const
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Argument y = b.arg(F32);
         b.store(b.load_splat(y), x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[0]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Argument y = b.arg(F64);
         b.store(b.load_splat(y), x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[0]
        )FOO"},
    // unaligned load
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.load(b.arg(F32), LoadStoreKind::Unaligned);
         b.store(y, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.load(b.arg(F64), LoadStoreKind::Unaligned);
         b.store(y, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    // unaligned store + unaligned load
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.load(b.arg(F32), LoadStoreKind::Unaligned);
         b.store(y, x, LoadStoreKind::Unaligned);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.load(b.arg(F64), LoadStoreKind::Unaligned);
         b.store(y, x, LoadStoreKind::Unaligned);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = y[i]
        )FOO"},
    // gather 32
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F32);
         Value tmp = b.gather(idx, x);
         b.store(tmp, dst);
     },
    PASS_ALL},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F64);
         Value tmp = b.gather(idx, x);
         b.store(tmp, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector,
                                           simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    // gather 64
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value idx = b.input_arg(I64);
         Argument dst = b.arg(F32);
         Value tmp = b.gather(idx, x);
         b.store(tmp, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector,
                                           simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value idx = b.input_arg(I64);
         Argument dst = b.arg(F64);
         Value tmp = b.gather(idx, x);
         b.store(tmp, dst);
     },
    PASS_ALL},
    // scatter 32
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F32);
         b.scatter(x, idx, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F64);
         b.scatter(x, idx, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F64);
         b.scatter(b.float_cast(x, F64), idx, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // scatter 64
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Value idx = b.input_arg(I64);
         Argument dst = b.arg(F32);
         b.scatter(x, idx, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Value idx = b.input_arg(I64);
         Argument dst = b.arg(F64);
         b.scatter(x, idx, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // cond_store
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F32));
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F64));
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst, LoadStoreKind::Unaligned);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F32));
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst, LoadStoreKind::Unaligned);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst, LoadStoreKind::Unaligned);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F64));
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_store(x, cond, dst, LoadStoreKind::Unaligned);
     }},
    {[](FunctionBuilder &b) {
         Value idx = b.index(I32);
         Value rounded = b.round_nearest_even(b.load_splat(b.arg(F32)));
         Value idx_as_float = b.bitcast(idx, F32);
         Value loaded = b.load(b.arg(F32), LoadStoreKind::Unaligned);
         b.cond_store(rounded, b.cmp_gt(idx_as_float, loaded), b.arg(F32), LoadStoreKind::Unaligned);
         b.countif(b.true_(), b.arg(I64));
         b.cond_store(b.max(idx, b.load(b.arg(I32), LoadStoreKind::Unaligned)), b.cmp_ge(idx, b.i32(-1)),
                      b.arg(I32), LoadStoreKind::Unaligned);
     }},
    // pack
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(F32);
         Argument dst_sz = b.arg(I64);
         Predicate cond = b.input_predicate_arg();
         b.pack(x, cond, dst, dst_sz);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(F64);
         Argument dst_sz = b.arg(I64);
         Predicate cond = b.input_predicate_arg();
         b.pack(x, cond, dst, dst_sz);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(F32);
        Argument dst_sz = b.arg(I64);
        b.pack(x, b.cmp_gt(x, b.f32(0)), dst, dst_sz);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(F64);
        Argument dst_sz = b.arg(I64);
        b.pack(x, b.cmp_gt(x, b.f64(0)), dst, dst_sz);
    }},
    // cond_scatter
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_scatter(x, idx, cond, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F32));
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_scatter(x, idx, cond, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F32));
         Value idx = b.input_arg(I64);
         Argument dst = b.arg(F32);
         Predicate cond = b.input_predicate_arg();
         b.cond_scatter(x, idx, cond, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_scatter(x, idx, cond, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F64));
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_scatter(x, idx, cond, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.abs(b.input_arg(F64));
         Value idx = b.input_arg(I64);
         Argument dst = b.arg(F64);
         Predicate cond = b.input_predicate_arg();
         b.cond_scatter(x, idx, cond, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // simple blend
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Value y = b.input_arg(F32);
         Predicate cond = b.input_predicate_arg();
         Argument dst = b.arg(F32);
         b.store(b.select(cond, y, x), dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Value y = b.input_arg(F64);
         Predicate cond = b.input_predicate_arg();
         Argument dst = b.arg(F64);
         b.store(b.select(cond, y, x), dst);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Argument dst = b.arg(F32);
        b.store(b.select(b.cmp_gt(z, x), y, x), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Argument dst = b.arg(F64);
        b.store(b.select(b.cmp_gt(z, x), y, x), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        b.output_arg(b.select(b.cmp_lt(z, x), y, x));
        b.output_arg(b.select(b.cmp_gt(z, x), y, x));
        b.output_arg(b.select(b.cmp_le(z, x), y, x));
        b.output_arg(b.select(b.cmp_ge(z, x), y, x));
        b.output_arg(b.select(b.cmp_eq(z, x), y, x));
        b.output_arg(b.select(b.cmp_ne(z, x), y, x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        b.output_arg(b.select(b.cmp_lt(z, x), y, x));
        b.output_arg(b.select(b.cmp_gt(z, x), y, x));
        b.output_arg(b.select(b.cmp_le(z, x), y, x));
        b.output_arg(b.select(b.cmp_ge(z, x), y, x));
        b.output_arg(b.select(b.cmp_eq(z, x), y, x));
        b.output_arg(b.select(b.cmp_ne(z, x), y, x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value y = b.input_arg(F32);
        b.output_arg(b.select(b.cmp_lt(x, b.f32(0.0f)), y, x));
        b.output_arg(b.select(b.cmp_gt(x, b.f32(0.0f)), y, x));
        b.output_arg(b.select(b.cmp_le(x, b.f32(0.0f)), y, x));
        b.output_arg(b.select(b.cmp_ge(x, b.f32(0.0f)), y, x));
        b.output_arg(b.select(b.cmp_eq(x, b.f32(0.0f)), y, x));
        b.output_arg(b.select(b.cmp_ne(x, b.f32(0.0f)), y, x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        b.output_arg(b.select(b.cmp_lt(x, b.f64(0.0)), y, x));
        b.output_arg(b.select(b.cmp_gt(x, b.f64(0.0)), y, x));
        b.output_arg(b.select(b.cmp_le(x, b.f64(0.0)), y, x));
        b.output_arg(b.select(b.cmp_ge(x, b.f64(0.0)), y, x));
        b.output_arg(b.select(b.cmp_eq(x, b.f64(0.0)), y, x));
        b.output_arg(b.select(b.cmp_ne(x, b.f64(0.0)), y, x));
    }},
    // add
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value z = b.input_arg(F32);
         Value tmp = b.add(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] + z[i]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value z = b.input_arg(F64);
         Value tmp = b.add(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] + z[i]
        )FOO"},
    // sub
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value z = b.input_arg(F32);
         Value tmp = b.sub(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] - z[i]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value z = b.input_arg(F64);
         Value tmp = b.sub(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] - z[i]
        )FOO"},
    // mul
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value z = b.input_arg(F32);
         Value tmp = b.mul(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] * z[i]
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value z = b.input_arg(F64);
         Value tmp = b.mul(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = y[i] * z[i]
        )FOO"},
    // div
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Value tmp = b.div(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.abs(b.input_arg(F32));
        Value z = b.abs(b.input_arg(F32));
        Value tmp = b.div(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Value tmp = b.div(y, z);
        b.store(tmp, x);
    }},
    // and
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Value tmp = b.and_(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Value tmp = b.and_(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // or
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Value tmp = b.or_(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.abs(b.input_arg(F32));
        Value z = b.abs(b.input_arg(F32));
        Value tmp = b.or_(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Value tmp = b.or_(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.abs(b.input_arg(F64));
        Value z = b.abs(b.input_arg(F64));
        Value tmp = b.or_(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // xor
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Value tmp = b.xor_(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.abs(b.input_arg(F32));
        Value z = b.abs(b.input_arg(F32));
        Value tmp = b.xor_(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Value tmp = b.xor_(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.abs(b.input_arg(F64));
        Value z = b.abs(b.input_arg(F64));
        Value tmp = b.xor_(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // andnot
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Value tmp = b.andnot(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Value tmp = b.andnot(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // andnot register variant
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.abs(b.input_arg(F32));
        Value z = b.abs(b.input_arg(F32));
        Value tmp = b.andnot(y, z);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.abs(b.input_arg(F64));
        Value z = b.abs(b.input_arg(F64));
        Value tmp = b.andnot(y, z);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // not
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.input_arg(F32);
        Value tmp = b.not_(y);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.input_arg(F64);
        Value tmp = b.not_(y);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // not register variant
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I32);
        Value y = b.abs(b.input_arg(F32));
        Value tmp = b.not_(y);
        b.store(b.bitcast(tmp, I32), x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I64);
        Value y = b.abs(b.input_arg(F64));
        Value tmp = b.not_(y);
        b.store(b.bitcast(tmp, I64), x);
    }},
    // min
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value z = b.input_arg(F32);
         Value tmp = b.min(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = min(y[i], z[i])
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value z = b.input_arg(F64);
         Value tmp = b.min(y, z);
         b.store(tmp, x);
     }},
    {[](FunctionBuilder &b) {
         Predicate never = b.cmp_eq(b.index(I32), b.i32(-1));
         Value nan = b.con_internal(ConstData::u64(0x7fc00000u), F32);
         Value truthy = b.min(b.f32(-2.5f), nan);
         Value selected = b.select(never, truthy, b.input_arg(F32));
         b.store(b.float_cast(selected, F64), b.arg(F64));
     }},
    // fmin/fmax ignore a single NaN regardless of operand order. This also exercises the x86 destination/right alias
    // path used by repaired vector min/max.
    {[](FunctionBuilder &b) {
         Value finite = b.input_arg(F32);
         Value nan = b.con_internal(ConstData::u64(0x7fc00000u), F32);
         b.store(b.isnan(b.min(finite, nan)), b.arg(I1));
         b.store(b.isnan(b.min(nan, finite)), b.arg(I1));
         b.store(b.isnan(b.max(finite, nan)), b.arg(I1));
         b.store(b.isnan(b.max(nan, finite)), b.arg(I1));
     },
     PASS_ALL, R"FOO(
def func(n, finite, min_rhs_nan, min_lhs_nan, max_rhs_nan, max_lhs_nan):
    for i in range(n):
        min_rhs_nan[i] = False
        min_lhs_nan[i] = False
        max_rhs_nan[i] = False
        max_lhs_nan[i] = False
        )FOO"},
    // max
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value z = b.input_arg(F32);
         Value tmp = b.max(y, z);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y, z):
    for i in range(n):
        x[i] = max(y[i], z[i])
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value z = b.input_arg(F64);
         Value tmp = b.max(y, z);
         b.store(tmp, x);
     }},
    // abs
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value tmp = b.abs(y);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = abs(y[i])
        )FOO"},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value tmp = b.abs(y);
         b.store(tmp, x);
     },
     PASS_ALL, R"FOO(
def func(n, x, y):
    for i in range(n):
        x[i] = abs(y[i])
        )FOO"},
    // round
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.round_nearest_even(y);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.round_nearest_even(y);
        b.store(tmp, x);
    }},
    // floor
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.round_down(y);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.round_down(y);
        b.store(tmp, x);
    }},
    // ceil
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.round_up(y);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.round_up(y);
        b.store(tmp, x);
    }},
    // round2zero
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.round_toward_zero(y);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.round_toward_zero(y);
        b.store(tmp, x);
    }},
    // sqrt
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.sqrt(y);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.sqrt(y);
        b.store(tmp, x);
    }},
    // rsqrt
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value tmp = b.rsqrt(y);
         b.store(tmp, x);
     }},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value tmp = b.rsqrt(y);
         b.store(tmp, x);
     }},
    // rcp
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F32);
         Value y = b.input_arg(F32);
         Value tmp = b.rcp(y);
         b.store(tmp, x);
     }},
    {[](FunctionBuilder &b) {
         Argument x = b.arg(F64);
         Value y = b.input_arg(F64);
         Value tmp = b.rcp(y);
         b.store(tmp, x);
     }},
    // cmp less
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Predicate tmp = b.cmp_lt(y, z);
        b.store(tmp, x);
    }},
    // cmp greater
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Predicate tmp = b.cmp_gt(y, z);
        b.store(tmp, x);
    }},
    // cmp less equal
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Predicate tmp = b.cmp_le(y, z);
        b.store(tmp, x);
    }},
    // cmp greater equal
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Predicate tmp = b.cmp_ge(y, z);
        b.store(tmp, x);
    }},
    // cmp equal
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Predicate tmp = b.cmp_eq(y, z);
        b.store(tmp, x);
    }},
    // cmp not equal
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Value z = b.input_arg(F32);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Value z = b.input_arg(F64);
        Predicate tmp = b.cmp_ne(y, z);
        b.store(tmp, x);
    }},
    // cmp less 0
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Predicate tmp = b.cmp_lt(y, b.f32(0.0f));
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Predicate tmp = b.cmp_lt(y, b.f64(0.0));
        b.store(tmp, x);
    }},
    // cmp greater 0
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Predicate tmp = b.cmp_gt(y, b.f32(0.0f));
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Predicate tmp = b.cmp_gt(y, b.f64(0.0));
        b.store(tmp, x);
    }},
    // cmp less equal 0
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Predicate tmp = b.cmp_le(y, b.f32(0.0f));
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Predicate tmp = b.cmp_le(y, b.f64(0.0));
        b.store(tmp, x);
    }},
    // cmp greater equal 0
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Predicate tmp = b.cmp_ge(y, b.f32(0.0f));
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Predicate tmp = b.cmp_ge(y, b.f64(0.0));
        b.store(tmp, x);
    }},
    // cmp equal
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Predicate tmp = b.cmp_eq(y, b.f32(0.0f));
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Predicate tmp = b.cmp_eq(y, b.f64(0.0));
        b.store(tmp, x);
    }},
    // cmp not equal
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F32);
        Predicate tmp = b.cmp_ne(y, b.f32(0.0f));
        b.store(tmp, x);
    }},
    {[](FunctionBuilder &b) {
        Argument x = b.arg(I1);
        Value y = b.input_arg(F64);
        Predicate tmp = b.cmp_ne(y, b.f64(0.0));
        b.store(tmp, x);
    }},
    // float casts
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(F64);
         b.store(b.float_cast(x, F64), dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(F64);
         Value tmp = b.select(b.cmp_gt(x, b.f32(0.0f)), b.float_cast(x, F64), b.f64(0.0));
         b.store(tmp, dst);
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(F32);
         b.store(b.float_cast(x, F32), dst);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(F32);
        b.store(b.float_cast(b.add(x, b.f64(1.5)), F32), dst);
    }},
    // float -> int
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I32);
        b.store(b.float_cast(x, I32), dst);
    }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(I64);
         b.store(b.float_cast(x, I64), dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(I32);
         b.store(b.float_cast(x, I32), dst);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(I64);
        b.store(b.float_cast(x, I64), dst);
    }},
    // float -> uint
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I32);
        b.store(b.float_cast(x, I32, true), dst);
    }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(I64);
         b.store(b.float_cast(x, I64, true), dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(I32);
         b.store(b.float_cast(x, I32, true), dst);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(I64);
        b.store(b.float_cast(x, I64, true), dst);
    }},
    // int -> float
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(F32);
        b.store(b.float_cast(x, F32), dst);
    }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I64);
         Argument dst = b.arg(F32);
         b.store(b.float_cast(x, F32), dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I32);
         Argument dst = b.arg(F64);
         b.store(b.float_cast(x, F64), dst);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(F64);
        b.store(b.float_cast(x, F64), dst);
    }},
    // uint -> float
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(F32);
        b.store(b.float_cast(x, F32, true), dst);
    }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I64);
         Argument dst = b.arg(F32);
         b.store(b.float_cast(x, F32, true), dst);
     }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I32);
         Argument dst = b.arg(F64);
         b.store(b.float_cast(x, F64, true), dst);
     }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(F64);
        b.store(b.float_cast(x, F64, true), dst);
    }},
    // small integer float casts are decomposed through i32/i64 bridge casts
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(F32);
        b.store(b.float_cast(x, F32), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(F64);
        b.store(b.float_cast(x, F64, true), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I8);
        b.store(b.float_cast(x, I8), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(I16);
        b.store(b.float_cast(x, I16, true), dst);
    }},
    // integer casts mixed with float casts
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(F32);
        b.store(b.float_cast(b.sext(x, I32), F32), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(F32);
        b.store(b.float_cast(b.zext(x, I32), F32, true), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(F32);
        Value wide = b.sext(x, I64);
        b.store(b.float_cast(b.float_cast(wide, F64), F32), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(F32);
        Value wide = b.zext(x, I64);
        b.store(b.float_cast(b.float_cast(wide, F64, true), F32), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I64);
        b.store(b.sext(b.float_cast(x, I32), I64), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I64);
        b.store(b.zext(b.float_cast(x, I32, true), I64), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(I32);
        b.store(b.trunc(b.float_cast(x, I64), I32), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(I32);
        b.store(b.trunc(b.float_cast(x, I64, true), I32), dst);
    }},
    // casts before and after float casts
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I8);
        Value widened = b.sext(x, I32);
        Value as_float = b.float_cast(widened, F32);
        b.store(b.trunc(b.float_cast(as_float, I32), I8), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I8);
        Argument dst = b.arg(I16);
        Value widened = b.zext(x, I32);
        Value as_float = b.float_cast(widened, F64, true);
        b.store(b.trunc(b.float_cast(as_float, I32, true), I16), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I16);
        Value widened = b.sext(x, I32);
        Value as_double = b.float_cast(widened, F64);
        Value as_float = b.float_cast(as_double, F32);
        b.store(b.trunc(b.float_cast(as_float, I32), I16), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I16);
        Argument dst = b.arg(I8);
        Value narrowed = b.trunc(x, I8);
        Value widened = b.sext(narrowed, I32);
        Value as_double = b.float_cast(widened, F64);
        Value as_float = b.float_cast(as_double, F32);
        b.store(b.trunc(b.float_cast(as_float, I32), I8), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I16);
        Value narrowed = b.trunc(x, I16);
        Value widened = b.sext(narrowed, I32);
        Value as_float = b.float_cast(widened, F32);
        b.store(b.trunc(b.float_cast(as_float, I32), I16), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(I8);
        Value narrowed = b.trunc(x, I8);
        Value widened = b.zext(narrowed, I32);
        Value as_float = b.float_cast(widened, F32, true);
        b.store(b.trunc(b.float_cast(as_float, I32, true), I8), dst);
    }},
    // multiple roots with mixed cast shapes
    {[](FunctionBuilder &b) {
        Value x8 = b.input_arg(I8);
        Value x16 = b.input_arg(I16);
        Argument dst8 = b.arg(I8);
        Argument dst16 = b.arg(I16);

        Value x8_float = b.float_cast(b.sext(x8, I32), F32);
        Value x16_float = b.float_cast(b.zext(x16, I32), F32, true);
        b.store(b.trunc(b.float_cast(x8_float, I32), I8), dst8);
        b.store(b.trunc(b.float_cast(x16_float, I32, true), I16), dst16);
    }},
    {[](FunctionBuilder &b) {
        Value x8 = b.input_arg(I8);
        Value x16 = b.input_arg(I16);
        Value x32 = b.input_arg(I32);
        Argument dst8 = b.arg(I8);
        Argument dst16 = b.arg(I16);
        Argument dstf32 = b.arg(F32);

        Value x8_float = b.float_cast(b.sext(x8, I32), F32);
        Value x16_double = b.float_cast(b.sext(x16, I32), F64);
        Value x32_small = b.sext(b.trunc(x32, I16), I32);
        b.store(b.trunc(b.float_cast(x8_float, I32), I8), dst8);
        b.store(b.trunc(b.float_cast(b.float_cast(x16_double, F32), I32), I16), dst16);
        b.store(b.float_cast(x32_small, F32), dstf32);
    }},
    {[](FunctionBuilder &b) {
        Value x8 = b.input_arg(I8);
        Value x16 = b.input_arg(I16);
        Value x32 = b.input_arg(I32);
        Argument dst8 = b.arg(I8);
        Argument dst16 = b.arg(I16);
        Argument dstf32 = b.arg(F32);
        Argument dstf64 = b.arg(F64);

        Value x8_float = b.float_cast(b.sext(x8, I32), F32);
        Value x16_double = b.float_cast(b.zext(x16, I32), F64, true);
        Value x32_i16 = b.sext(b.trunc(x32, I16), I32);
        Value x32_i8 = b.zext(b.trunc(x32, I8), I32);
        b.store(b.trunc(b.float_cast(x8_float, I32), I8), dst8);
        b.store(b.trunc(b.float_cast(b.float_cast(x16_double, F32), I32, true), I16), dst16);
        b.store(b.float_cast(x32_i16, F32), dstf32);
        b.store(b.float_cast(x32_i8, F64, true), dstf64);
    },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::TooManyRoots)},
    // negate
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F32);
         Argument dst = b.arg(F32);
         b.store(b.negate(x), dst);
     },
     PASS_ALL, /*R"FOO(
def func(n, dst, x):
    for i in range(n):
        dst[i] = -x[i]
        )FOO"*/ // for some reason python code always produces zeroes
    
    },
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.negate(x), dst);
     },
     PASS_ALL, /*R"FOO(
def func(n, dst, x):
    for i in range(n):
        dst[i] = -x[i]
        )FOO"*/},
    // fma
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F32);
         Value x2 = b.input_arg(F32);
         Value x3 = b.input_arg(F32);
         Argument dst = b.arg(F32);
         b.store(b.add(b.mul(x1, x2), x3), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.add(b.mul(x1, x2), x3), dst);
     },
     },
    // fms
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F32);
         Value x2 = b.input_arg(F32);
         Value x3 = b.input_arg(F32);
         Argument dst = b.arg(F32);
         b.store(b.sub(b.mul(x1, x2), x3), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.sub(b.mul(x1, x2), x3), dst);
     },
     },
    // fnma
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F32);
         Value x2 = b.input_arg(F32);
         Value x3 = b.input_arg(F32);
         Argument dst = b.arg(F32);
         b.store(b.add(b.negate(b.mul(x1, x2)), x3), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.add(b.negate(b.mul(x1, x2)), x3), dst);
     },
     },
    // fnms
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F32);
         Value x2 = b.input_arg(F32);
         Value x3 = b.input_arg(F32);
         Argument dst = b.arg(F32);
         b.store(b.sub(b.negate(b.mul(x1, x2)), x3), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.sub(b.negate(b.mul(x1, x2)), x3), dst);
     },
     },
    // fma with multiply on the right side
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.add(x3, b.mul(x1, x2)), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.sub(x3, b.mul(x1, x2)), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.add(x3, b.negate(b.mul(x1, x2))), dst);
     },
     },
    {[](FunctionBuilder &b) {
         Value x1 = b.input_arg(F64);
         Value x2 = b.input_arg(F64);
         Value x3 = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.store(b.sub(x3, b.negate(b.mul(x1, x2))), dst);
     },
     },
    // sum
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(F32);
        b.sum(x, dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(F64);
        b.sum(x, dst);
    }},
    // product
    {[](FunctionBuilder &b) {
         Value raw = b.input_arg(F32);
         Value x = b.select(b.cmp_gt(raw, b.f32(0.0f)), b.f32(1.0f), b.f32(-1.0f));
         Argument dst = b.arg(F32);
         b.product(x, dst);
     },
     PASS_ALL, R"FOO(
def func(n, raw, dst):
    for i in range(n):
        dst[0] *= 1.0 if raw[i] > 0.0 else -1.0
        )FOO"},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(F64);
         Argument dst = b.arg(F64);
         b.product(x, dst);
     }},
    // min_agg
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(F32);
        b.min_agg(x, dst);
    },
     PASS_ALL, R"FOO(
def func(n, x, dst):
    for i in range(n):
        dst[0] = min(dst[0], x[i])
        )FOO"},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(F64);
        b.min_agg(x, dst);
    }},
    // max_agg
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(F32);
        b.max_agg(x, dst);
    },
     PASS_ALL, R"FOO(
def func(n, x, dst):
    for i in range(n):
        dst[0] = max(dst[0], x[i])
        )FOO"},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(F64);
        b.max_agg(x, dst);
    }},
    // dot
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value y = b.input_arg(F32);
        Argument dst = b.arg(F32);
        b.sum(b.mul(x, y), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        Argument dst = b.arg(F64);
        b.sum(b.mul(x, y), dst);
    }},
    // bitcast
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Argument dst = b.arg(I32);
        b.store(b.bitcast(x, I32), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Argument dst = b.arg(F32);
        Value tmp = b.bitcast(x, F32);
        b.store(b.zero_select(tmp, b.isnormal(tmp)), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Argument dst = b.arg(I64);
        b.store(b.bitcast(x, I64), dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Argument dst = b.arg(F64);
        Value tmp = b.bitcast(x, F64);
        b.store(b.zero_select(tmp, b.isnormal(tmp)), dst);
    }},
    // tpch q1 for fun
    {[](FunctionBuilder &b) {
        Value l_quantity = b.input_arg(F64);
        Value l_extendedprice = b.input_arg(F64);
        Value l_discount = b.input_arg(F64);
        Argument sum_qty = b.arg(F64);
        Argument sum_base_price = b.arg(F64);
        Argument sum_disc_price = b.arg(F64);
        b.sum(l_quantity, sum_qty);
        b.sum(l_extendedprice, sum_base_price);
        b.sum(b.mul(l_extendedprice, b.sub(b.f64(1.0), l_discount)), sum_disc_price);
    }},
    {[](FunctionBuilder &b) {
         Value l_quantity = b.input_arg(F64);
         Value l_extendedprice = b.input_arg(F64);
         Value l_discount = b.float_cast(b.input_arg(F32), F64);
         Argument sum_qty = b.arg(F64);
         Argument sum_base_price = b.arg(F64);
         Argument sum_disc_price = b.arg(F64);
         b.sum(l_quantity, sum_qty);
         b.sum(l_extendedprice, sum_base_price);
         b.sum(b.mul(l_extendedprice, b.sub(b.f64(1.0), l_discount)), sum_disc_price);
     }},
    {[](FunctionBuilder &b) {
         Value l_quantity = b.float_cast(b.sext(b.input_arg(I16), I64), F64);
         Value l_extendedprice = b.input_arg(F64);
         Value l_discount = b.float_cast(b.input_arg(F32), F64);
         Argument sum_qty = b.arg(F64);
         Argument sum_base_price = b.arg(F64);
         Argument sum_disc_price = b.arg(F64);
         b.sum(l_quantity, sum_qty);
         b.sum(l_extendedprice, sum_base_price);
         b.sum(b.mul(l_extendedprice, b.sub(b.f64(1.0), l_discount)), sum_disc_price);
     }},
    {[](FunctionBuilder &b) {
         Value l_quantity = b.float_cast(b.sext(b.input_arg(I16), I32), F32);
         Value l_extendedprice = b.float_cast(b.input_arg(F64), F32);
         Value l_discount = b.input_arg(F32);
         Argument sum_qty = b.arg(F32);
         Argument sum_base_price = b.arg(F32);
         Argument sum_disc_price = b.arg(F32);
         b.sum(l_quantity, sum_qty);
         b.sum(l_extendedprice, sum_base_price);
         b.sum(b.mul(l_extendedprice, b.sub(b.f32(1.0f), l_discount)), sum_disc_price);
     }},
    {[](FunctionBuilder &b) {
        Value l_quantity = b.input_arg(F64);
        Value l_extendedprice = b.input_arg(F64);
        Value l_discount = b.input_arg(F64);
        Value l_tax = b.input_arg(F64);
        Argument sum_qty = b.arg(F64);
        Argument sum_base_price = b.arg(F64);
        Argument sum_disc_price = b.arg(F64);
        Argument sum_charge = b.arg(F64);
        b.sum(l_quantity, sum_qty);
        b.sum(l_extendedprice, sum_base_price);
        b.sum(b.mul(l_extendedprice, b.sub(b.f64(1.0), l_discount)), sum_disc_price);
        b.sum(b.mul(b.mul(l_extendedprice, b.sub(b.f64(1.0), l_discount)), b.add(b.f64(1.0), l_tax)), sum_charge);
    }},
    {[](FunctionBuilder &b) {
         Value l_quantity = b.input_arg(F64);
         Value l_extendedprice = b.input_arg(F32);
         Value l_discount = b.input_arg(F32);
         Value l_tax = b.input_arg(F32);
         Argument sum_qty = b.arg(F64);
         Argument sum_base_price = b.arg(F64);
         Argument sum_disc_price = b.arg(F64);
         Argument sum_charge = b.arg(F64);
         b.sum(l_quantity, sum_qty);
         b.sum(b.float_cast(l_extendedprice, F64), sum_base_price);
         b.sum(b.mul(b.float_cast(l_extendedprice, F64), b.float_cast(b.sub(b.f32(1.0f), l_discount), F64)),
               sum_disc_price);
         b.sum(b.mul(b.mul(b.float_cast(l_extendedprice, F64), b.float_cast(b.sub(b.f32(1.0f), l_discount), F64)),
                     b.float_cast(b.add(b.f32(1.0f), l_tax), F64)),
               sum_charge);
     }},
    // sign_no_zero
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value tmp = b.sign_no_zero(x);
        b.output_arg(tmp);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value tmp = b.sign_no_zero(x);
        b.output_arg(tmp);
    }},
    // sign
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value tmp = b.sign(x);
        b.output_arg(tmp);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value tmp = b.sign(x);
        b.output_arg(tmp);
    }},
    // copysign_no_zero
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.copysign_no_zero(x, y);
        b.output_arg(tmp);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.copysign_no_zero(x, y);
        b.output_arg(tmp);
    }},
    // copysign
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        Value tmp = b.copysign(x, y);
        b.output_arg(tmp);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value y = b.input_arg(F32);
        Value tmp = b.copysign(x, y);
        b.output_arg(tmp);
    }},
    // copysign_no_zero with ints
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I32);
         Value y = b.input_arg(F64);
         Value tmp = b.copysign_no_zero(x, y);
         b.output_arg(tmp);
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I64);
         Value y = b.input_arg(F32);
         Value tmp = b.copysign_no_zero(x, y);
         b.output_arg(tmp);
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    // copysign with ints
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I32);
         Value y = b.input_arg(F64);
         Value tmp = b.copysign(x, y);
         b.output_arg(tmp);
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I64);
         Value y = b.input_arg(F32);
         Value tmp = b.copysign(x, y);
         b.output_arg(tmp);
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    // permute bits in float (for whatever reason)
    {[](FunctionBuilder &b) {
         Value x = b.bitcast(b.input_arg(F64), I64);
         Argument dst = b.arg(F64);
         Value tmp = b.permute_i8_bits(x, 0, 1, 2, 3, 4, 5, 6, 7);
         tmp = b.bitcast(tmp, F64);
         tmp = b.zero_select(tmp, b.isnormal(tmp));
         b.store(tmp, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value x = b.bitcast(b.input_arg(F64), I64);
         Argument dst = b.arg(F64);
         Value tmp = b.permute_i8_bits(x, 1, 2, 3, 4, 7, 6, 5, 0);
         tmp = b.bitcast(tmp, F64);
         tmp = b.zero_select(tmp, b.isnormal(tmp));
         b.store(tmp, dst);
     },
    LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    // permute dwords in double
    {[](FunctionBuilder &b) {
        Value x = b.bitcast(b.input_arg(F64), I64);
        Argument dst = b.arg(F64);
        Value tmp = b.permute_i64_i32(x, 0, 0);
        tmp = b.bitcast(tmp, F64);
        tmp = b.zero_select(tmp, b.isnormal(tmp));
        b.store(tmp, dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.bitcast(b.input_arg(F64), I64);
        Argument dst = b.arg(F64);
        Value tmp = b.permute_i64_i32(x, 0, 1);
        tmp = b.bitcast(tmp, F64);
        tmp = b.zero_select(tmp, b.isnormal(tmp));
        b.store(tmp, dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.bitcast(b.input_arg(F64), I64);
        Argument dst = b.arg(F64);
        Value tmp = b.permute_i64_i32(x, 1, 0);
        tmp = b.bitcast(tmp, F64);
        tmp = b.zero_select(tmp, b.isnormal(tmp));
        b.store(tmp, dst);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.bitcast(b.input_arg(F64), I64);
        Argument dst = b.arg(F64);
        Value tmp = b.permute_i64_i32(x, 1, 1);
        tmp = b.bitcast(tmp, F64);
        tmp = b.zero_select(tmp, b.isnormal(tmp));
        b.store(tmp, dst);
    }},
    // isinf
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7f800000u), F32);
        b.output_arg(b.isinf(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7ff0000000000000llu), F64);
        b.output_arg(b.isinf(x));
    }},
    // isnan
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7fc00000u), F32);
        b.output_arg(b.isnan(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7ff8000000000000llu), F64);
        b.output_arg(b.isnan(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7fa00000u), F32);
        b.output_arg(b.isnan(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7ff0000000000001llu), F64);
        b.output_arg(b.isnan(x));
    }},
    // isfinite
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        b.output_arg(b.isfinite(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        b.output_arg(b.isfinite(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7f800000u), F32);
        b.output_arg(b.isfinite(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7ff0000000000000llu), F64);
        b.output_arg(b.isfinite(x));
    }},
    // isnormal
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        b.output_arg(b.isnormal(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        b.output_arg(b.isnormal(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7f800000u), F32);
        b.output_arg(b.isnormal(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.con_internal(ConstData::u64(0x7ff0000000000000llu), F64);
        b.output_arg(b.isnormal(x));
    }},
    // stuff for linear regression
    {[](FunctionBuilder &b) {
        Value x_start = b.load_splat(b.arg(F32));
        Value x_orig = b.input_arg(F32);
        Value y_start = b.load_splat(b.arg(F32));
        Value y_orig = b.input_arg(F32);

        Value x = b.sub(x_orig, x_start);
        Value xx = b.mul(x, x);
        Value y = b.sub(y_orig, y_start);
        Value xy = b.mul(x, y);

        b.sum(x, b.arg(F32));
        b.sum(xx, b.arg(F32));
        b.sum(y, b.arg(F32));
        b.sum(xy, b.arg(F32));
    }}};
