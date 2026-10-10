// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;

namespace {

// ---- Complex-arithmetic helpers over deinterleaved lane pairs ----

struct ComplexLanes {
    Value re;
    Value im;
};

ComplexLanes complex_add(FunctionBuilder &b, ComplexLanes x, ComplexLanes y) {
    return {b.add(x.re, y.re), b.add(x.im, y.im)};
}

ComplexLanes complex_mul(FunctionBuilder &b, ComplexLanes x, ComplexLanes y) {
    return {b.sub(b.mul(x.re, y.re), b.mul(x.im, y.im)), b.add(b.mul(x.re, y.im), b.mul(x.im, y.re))};
}

// (x.re + x.im*i) / (y.re + y.im*i), multiplied out to real divisions.
ComplexLanes complex_div(FunctionBuilder &b, ComplexLanes x, ComplexLanes y) {
    Value d = b.add(b.mul(y.re, y.re), b.mul(y.im, y.im));
    return {b.div(b.add(b.mul(x.re, y.re), b.mul(x.im, y.im)), d),
            b.div(b.sub(b.mul(x.im, y.re), b.mul(x.re, y.im)), d)};
}

// Deinterleave (re, im) pairs and store a complex result back to two outputs.
ComplexLanes load_complex(FunctionBuilder &b, Argument src) {
    DeinterleavedPair lanes = b.load2(src);
    return {lanes.even, lanes.odd};
}

void store_complex(FunctionBuilder &b, ComplexLanes z, Argument re_dst, Argument im_dst) {
    b.store(z.re, re_dst);
    b.store(z.im, im_dst);
}

// Deinterleave both lanes, add a constant to the even lane, store both lanes.
#define ADD_DEINTERLEAVE_ACCESS_TEST(tests, dt, const_fn)                         \
    tests.push_back(Test{[](FunctionBuilder &b) {                                 \
                             Argument src = b.arg(dt);                            \
                             Argument even_dst = b.arg(dt);                       \
                             Argument odd_dst = b.arg(dt);                        \
                             DeinterleavedPair lanes = b.load2(src);              \
                             b.store(b.add(lanes.even, b.const_fn(1)), even_dst); \
                             b.store(lanes.odd, odd_dst);                         \
                         },                                                       \
                         PASS_ALL, R"FOO(
def func(n, src, even_dst, odd_dst):
    for i in range(n):
        even_dst[i] = src[2*i] + 1
        odd_dst[i] = src[2*i+1]
        )FOO"});

// Consume a single lane of the interleave; the other lane must be dropped.
#define ADD_DEINTERLEAVE_SINGLE_LANE_TEST(tests, dt, const_fn, lane_value, py)        \
    tests.push_back(Test{[](FunctionBuilder &b) {                                     \
                             Argument src = b.arg(dt);                                \
                             Argument dst = b.arg(dt);                                \
                             DeinterleavedPair lanes = b.load2(src);                  \
                             Value lane = (lane_value) == 0 ? lanes.even : lanes.odd; \
                             b.store(b.add(lane, b.const_fn(2)), dst);                \
                         },                                                           \
                         PASS_ALL, py});

// Plain loads and deinterleave loads mixed on the same argument: rejected at analysis time. A plain
// access advances one element per row while the interleaved view advances lane_count, so the two
// access models cannot share an argument.
#define ADD_DEINTERLEAVE_MIXED_ACCESS_TEST(tests, dt, const_fn)                                             \
    tests.push_back(Test{[](FunctionBuilder &b) {                                                           \
                             Argument src = b.arg(dt);                                                      \
                             Argument plain_dst = b.arg(dt);                                                \
                             Argument lanes_dst = b.arg(dt);                                                \
                             Value plain = b.load(src);                                                     \
                             DeinterleavedPair lanes = b.load2(src);                                        \
                             b.store(b.add(plain, b.const_fn(3)), plain_dst);                               \
                             b.store(b.add(lanes.even, lanes.odd), lanes_dst);                              \
                         },                                                                                 \
                         test_meta()                                                                        \
                             .limitation(TestVariant::All)                                                  \
                             .structured_error(TestVariant::All, ErrorModule::HIR, ErrorKind::InvalidInput, \
                                               ErrorSubKind::InvalidArgumentAccess),                        \
                         R"FOO(
def func(n, src, plain_dst, lanes_dst):
    for i in range(n):
        plain_dst[i] = src[i] + 3
        lanes_dst[i] = src[2*i] + src[2*i+1]
        )FOO"});

std::vector<Test> make_interleaved_tests() {
    std::vector<Test> tests;

    // Basic access over every supported element type.
    ADD_DEINTERLEAVE_ACCESS_TEST(tests, I8, i8)
    ADD_DEINTERLEAVE_ACCESS_TEST(tests, I16, i16)
    ADD_DEINTERLEAVE_ACCESS_TEST(tests, I32, i32)
    ADD_DEINTERLEAVE_ACCESS_TEST(tests, I64, i64)
    ADD_DEINTERLEAVE_ACCESS_TEST(tests, F32, f32)
    ADD_DEINTERLEAVE_ACCESS_TEST(tests, F64, f64)

    // Single-lane consumption: even-only and odd-only.
    ADD_DEINTERLEAVE_SINGLE_LANE_TEST(tests, I32, i32, 0, R"FOO(
def func(n, src, dst):
    for i in range(n):
        dst[i] = src[2*i] + 2
        )FOO")
    ADD_DEINTERLEAVE_SINGLE_LANE_TEST(tests, I32, i32, 1, R"FOO(
def func(n, src, dst):
    for i in range(n):
        dst[i] = src[2*i+1] + 2
        )FOO")
    ADD_DEINTERLEAVE_SINGLE_LANE_TEST(tests, F64, f64, 0, R"FOO(
def func(n, src, dst):
    for i in range(n):
        dst[i] = src[2*i] + 2
        )FOO")
    ADD_DEINTERLEAVE_SINGLE_LANE_TEST(tests, F64, f64, 1, R"FOO(
def func(n, src, dst):
    for i in range(n):
        dst[i] = src[2*i+1] + 2
        )FOO")

    // The single-lane API mixes with the pair API: CSE unifies the leaves.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(I32);
                             Argument even_dst = b.arg(I32);
                             Argument odd_dst = b.arg(I32);
                             Value even = b.load2_0(src);
                             DeinterleavedPair lanes = b.load2(src);
                             b.store(b.add(even, b.i32(1)), even_dst);
                             b.store(b.add(lanes.odd, b.i32(2)), odd_dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, src, even_dst, odd_dst):
    for i in range(n):
        even_dst[i] = src[2*i] + 1
        odd_dst[i] = src[2*i+1] + 2
        )FOO"});

    // The interleaved buffer is also read by a plain (non-deinterleaving) load.
    ADD_DEINTERLEAVE_MIXED_ACCESS_TEST(tests, I32, i32)
    ADD_DEINTERLEAVE_MIXED_ACCESS_TEST(tests, I64, i64)

    // i64 arithmetic: lanes as (real, imag) of a complex number, squared.
    // Vectorized i64 mul is an x86-only special op; arm falls back to scalar.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(I64);
                             Argument re_dst = b.arg(I64);
                             Argument im_dst = b.arg(I64);
                             DeinterleavedPair lanes = b.load2(src);
                             Value re = lanes.even;
                             Value im = lanes.odd;
                             b.store(b.sub(b.mul(re, re), b.mul(im, im)), re_dst);
                             b.store(b.mul(b.add(im, im), re), im_dst);
                         },
                         test_meta()
                             .limitation(TestVariant::ArmVector)
                             .structured_error(TestVariant::ArmVector, ErrorModule::A64, ErrorKind::Unsupported,
                                               ErrorSubKind::UnsupportedBackendFeature),
                         R"FOO(
def func(n, src, re_dst, im_dst):
    for i in range(n):
        e = src[2*i]
        o = src[2*i+1]
        re_dst[i] = e*e - o*o
        im_dst[i] = 2*e*o
        )FOO"});

    // i64 arithmetic: division and modulo by constants (constant-division lowering).
    // Arm vector lowering of i64 const-div/-mod requires i64 mul, which is x86-only.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(I64);
                             Argument div_dst = b.arg(I64);
                             Argument mod_dst = b.arg(I64);
                             DeinterleavedPair lanes = b.load2(src);
                             b.store(b.div(lanes.even, b.i64(7)), div_dst);
                             b.store(b.mod(lanes.odd, b.i64(-7)), mod_dst);
                         },
                         test_meta()
                             .limitation(TestVariant::ArmVector)
                             .structured_error(TestVariant::ArmVector, ErrorModule::A64, ErrorKind::Unsupported,
                                               ErrorSubKind::UnsupportedBackendFeature),
                         R"FOO(
def func(n, src, div_dst, mod_dst):
    for i in range(n):
        div_dst[i] = src[2*i] // 7
        mod_dst[i] = src[2*i+1] % 7
        )FOO"});

    // i64 arithmetic: shifts and bit operations over both lanes.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(I64);
                             Argument dst = b.arg(I64);
                             DeinterleavedPair lanes = b.load2(src);
                             Value shifted = b.sra(lanes.even, b.i64(3));
                             b.store(b.xor_(shifted, b.and_(lanes.odd, b.i64(0xff))), dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, src, dst):
    for i in range(n):
        dst[i] = (src[2*i] >> 3) ^ (src[2*i+1] & 0xff)
        )FOO"});

    // Asymmetric lane consumers force different vectorizer widths per lane.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(I32);
                             Argument even_dst = b.arg(I16);
                             Argument odd_dst = b.arg(I32);
                             DeinterleavedPair lanes = b.load2(src);
                             b.store(b.trunc(lanes.even, I16), even_dst);
                             b.store(lanes.odd, odd_dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, src, even_dst, odd_dst):
    for i in range(n):
        even_dst[i] = src[2*i] & 0xffff
        odd_dst[i] = src[2*i+1]
        )FOO"});

    // Repeated load2 calls share lane steps (CSE).
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(I32);
                             Argument even_dst = b.arg(I32);
                             Argument odd_dst = b.arg(I32);
                             DeinterleavedPair first = b.load2(src);
                             DeinterleavedPair second = b.load2(src);
                             b.store(b.add(first.even, second.odd), even_dst);
                             b.store(b.add(first.odd, b.i32(7)), odd_dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, src, even_dst, odd_dst):
    for i in range(n):
        even_dst[i] = src[2*i] + src[2*i+1]
        odd_dst[i] = src[2*i+1] + 7
        )FOO"});

    // gather with interleaved index/value pairs (even slots hold indices)
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument pairs = b.arg(I32);
                             Argument data = b.arg(I32);
                             Argument dst = b.arg(I32);
                             DeinterleavedPair lanes = b.load2(pairs);
                             b.store(b.gather(lanes.even, data), dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, pairs, data, dst):
    for i in range(n):
        dst[i] = data[2*i]
        )FOO"});

    // scatter with interleaved index/value pairs (even slots hold indices)
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument pairs = b.arg(I32);
                             Value values = b.input_arg(I32);
                             Argument dst = b.arg(I32);
                             DeinterleavedPair lanes = b.load2(pairs);
                             b.scatter(values, lanes.even, dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, pairs, values, dst):
    for i in range(n):
        dst[2*i] = values[i]
        )FOO"});

    // f32 basic access with distinct lane expressions.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(F32);
                             Argument even_dst = b.arg(F32);
                             Argument odd_dst = b.arg(F32);
                             DeinterleavedPair lanes = b.load2(src);
                             b.store(b.add(lanes.even, b.f32(0.5f)), even_dst);
                             b.store(b.mul(lanes.odd, b.f32(2.0f)), odd_dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, src, even_dst, odd_dst):
    for i in range(n):
        even_dst[i] = src[2*i] + 0.5
        odd_dst[i] = src[2*i+1] * 2
        )FOO"});

    // f32 arithmetic: lanes as (real, imag), scaled rotation by a fixed angle.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument src = b.arg(F32);
                             Argument re_dst = b.arg(F32);
                             Argument im_dst = b.arg(F32);
                             DeinterleavedPair lanes = b.load2(src);
                             Value re = lanes.even;
                             Value im = lanes.odd;
                             Value half_sqrt2 = b.f32(0.7071067811865476f);
                             b.store(b.mul(b.sub(re, im), half_sqrt2), re_dst);
                             b.store(b.mul(b.add(re, im), half_sqrt2), im_dst);
                         },
                         PASS_ALL, R"FOO(
def func(n, src, re_dst, im_dst):
    for i in range(n):
        e = src[2*i]
        o = src[2*i+1]
        re_dst[i] = (e - o) * 0.7071067811865476
        im_dst[i] = (e + o) * 0.7071067811865476
        )FOO"});

    // f64 series impedance Z(R, L, C, f) = R + i*2*pi*f*L + 1/(i*2*pi*f*C).
    // R and L arrive as one interleaved stream, C and f as another.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument rl_arg = b.arg(F64);
                             Argument cf_arg = b.arg(F64);
                             Argument re_dst = b.arg(F64);
                             Argument im_dst = b.arg(F64);
                             ComplexLanes rl = load_complex(b, rl_arg);
                             ComplexLanes cf = load_complex(b, cf_arg);
                             Value omega = b.mul(cf.im, b.f64(6.283185307179586));
                             ComplexLanes resistance = {rl.re, b.f64(0)};
                             ComplexLanes inductor = {b.f64(0), b.mul(omega, rl.im)};
                             ComplexLanes capacitor =
                                 complex_div(b, {b.f64(1), b.f64(0)}, {b.f64(0), b.mul(cf.re, omega)});
                             ComplexLanes z = complex_add(b, complex_add(b, resistance, inductor), capacitor);
                             store_complex(b, z, re_dst, im_dst);
                         },
                         PASS_ALL});

    // f64 parallel impedance Z_parallel = 1 / (1/R + 1/(i*2*pi*f*L) + i*2*pi*f*C).
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument rl_arg = b.arg(F64);
                             Argument cf_arg = b.arg(F64);
                             Argument re_dst = b.arg(F64);
                             Argument im_dst = b.arg(F64);
                             ComplexLanes rl = load_complex(b, rl_arg);
                             ComplexLanes cf = load_complex(b, cf_arg);
                             Value omega = b.mul(cf.im, b.f64(6.283185307179586));
                             ComplexLanes inductor_term =
                                 complex_div(b, {b.f64(1), b.f64(0)}, {b.f64(0), b.mul(rl.im, omega)});
                             ComplexLanes denominator =
                                 complex_add(b, {b.div(b.f64(1), rl.re), b.mul(cf.re, omega)}, inductor_term);
                             ComplexLanes zp = complex_div(b, {b.f64(1), b.f64(0)}, denominator);
                             store_complex(b, zp, re_dst, im_dst);
                         },
                         PASS_ALL});

    // f64 low-pass filter H(w, R, L, C) = 1 / (1 - w*w*L*C + i*w*R*C).
    // R and L arrive as one interleaved stream, C and w as another.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument rl_arg = b.arg(F64);
                             Argument cw_arg = b.arg(F64);
                             Argument re_dst = b.arg(F64);
                             Argument im_dst = b.arg(F64);
                             ComplexLanes rl = load_complex(b, rl_arg);
                             ComplexLanes cw = load_complex(b, cw_arg);
                             Value omega = cw.im;
                             Value capacitance = cw.re;
                             Value real_denom = b.sub(b.f64(1), b.mul(b.mul(rl.im, capacitance), omega));
                             Value imag_denom = b.mul(b.mul(rl.re, capacitance), omega);
                             ComplexLanes h = complex_div(b, {b.f64(1), b.f64(0)}, {real_denom, imag_denom});
                             store_complex(b, h, re_dst, im_dst);
                         },
                         PASS_ALL});

    // f32 mandelbrot iteration: z_{k+1} = z_k^2 + c, four iterations per row.
    tests.push_back(Test{[](FunctionBuilder &b) {
                             Argument z_arg = b.arg(F32);
                             Argument c_arg = b.arg(F32);
                             Argument re_dst = b.arg(F32);
                             Argument im_dst = b.arg(F32);
                             ComplexLanes z = load_complex(b, z_arg);
                             ComplexLanes c = load_complex(b, c_arg);
                             for (int iteration = 0; iteration < 4; ++iteration) {
                                 z = complex_add(b, complex_mul(b, z, z), c);
                             }
                             store_complex(b, z, re_dst, im_dst);
                         },
                         PASS_ALL});

    return tests;
}

} // namespace

std::vector<Test> interleaved_tests = make_interleaved_tests();
