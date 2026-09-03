// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/nullable.h"
#include "test.h"

using namespace simjit;
using namespace simjit::nullable;
using namespace simjit::types;

std::vector<Test> general_tests{
    {[](FunctionBuilder &b) {
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);
        Value d = b.input_arg(I32);
        Argument result = b.arg(I32);
        b.store(b.mul(b.add(a, b_arg), b.sub(c, d)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test: (5 * x) + (y - 3)
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Argument res = b.arg(I64);
        b.store(b.add(b.mul(b.i64(5), x), b.sub(y, b.i64(3))), res);
    }},
    {[](FunctionBuilder &b) {
        // Test sum, min, max on same value
        Value val = b.input_arg(I32);
        Argument sum_result = b.arg(I32);
        Argument min_result = b.arg(I32);
        Argument max_result = b.arg(I32);

        b.sum(val, sum_result);
        b.min_agg(val, min_result);
        b.max_agg(val, max_result);
    }},

    {[](FunctionBuilder &b) {
        // Test: ((a & mask) | (b << 2)) ^ c
        Value a = b.input_arg(I16);
        Value b_arg = b.input_arg(I16);
        Value c = b.input_arg(I16);
        Value mask = b.i16(0xFF);
        Argument result = b.arg(I16);

        b.store(b.xor_(b.or_(b.and_(a, mask), b.sll(b_arg, b.i16(2))), c), result);
    }},
    {[](FunctionBuilder &b) {
        // Test: product(x * 2 + 1)
        Value x = b.sext(b.input_arg(I8), I32);
        Argument prod_result = b.arg(I32); // Wider type for product
        b.product(b.add(b.mul(x, b.i32(2)), b.i32(1)), prod_result);
    }},

    {[](FunctionBuilder &b) {
        // Test sum of products and product of sums
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument sum_prod = b.arg(I32);
        Argument prod_sum = b.arg(I32);

        b.sum(b.mul(x, y), sum_prod);
        b.product(b.add(x, y), prod_sum);
    }},
    {[](FunctionBuilder &b) {
        // Test: abs(negate(x)) + popcnt(y) + lzcnt(z)
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);
        Argument result = b.arg(I32);

        b.store(b.add(b.add(b.abs(b.negate(x)), b.popcnt(y)), b.lzcnt(z)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test: (x > 0 ? x : 0) using bitwise operations
        // Implementation: x & ~(x >> 31) for I32 (arithmetic shift)
        Value x = b.input_arg(I32);
        Argument result = b.arg(I32);

        Value sign_mask = b.sra(x, b.i32(31)); // All ones if negative
        b.store(b.and_(x, b.not_(sign_mask)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test storing same computation to multiple aggregates
        Value val = b.input_arg(I64);
        Argument sum_out = b.arg(I64);
        Argument min_out = b.arg(I64);
        Argument max_out = b.arg(I64);

        Value comp = b.mul(val, b.i64(100));

        b.store(comp, sum_out);
        b.store(comp, min_out);
        b.store(comp, max_out);
    }},
    {[](FunctionBuilder &b) {
         // Test: sum(x*y), product(x+y)
         Value x = b.input_arg(I16);
         Value y = b.input_arg(I16);
         Argument sum_result = b.arg(I32);
         Argument prod_result = b.arg(I64);

         b.sum(b.sext(b.mul(x, y), I32), sum_result);
         b.product(b.sext(b.add(x, y), I64), prod_result);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Test operations with min/max constants
        Value val = b.input_arg(I8);
        Argument result = b.arg(I8);

        // Test: (val + 127) - (-128)
        b.store(b.sub(b.add(val, b.i8(127)), b.i8(-128)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test: ((a ^ b) & mask) | ((c & ~mask) >> 1)
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);
        Value mask = b.i32(0xFFFF);
        Argument result = b.arg(I32);

        b.store(b.or_(b.and_(b.xor_(a, b_arg), mask), b.sra(b.and_(c, b.not_(mask)), b.i32(1))), result);
    }},
    {[](FunctionBuilder &b) {
        // Test different aggregate types on same computation chain
        Value base = b.input_arg(I64);
        Value comp = b.mul(base, b.i64(10));

        Argument sum_i64 = b.arg(I64);
        Argument min_i64 = b.arg(I64);
        Argument max_i64 = b.arg(I64);

        b.sum(comp, sum_i64);
        b.min_agg(comp, min_i64);
        b.max_agg(comp, max_i64);
    }},
    {[](FunctionBuilder &b) {
        // Test that input_arg creates unique arguments even with same variable name
        Value x1 = b.input_arg(I32);
        Value x2 = b.input_arg(I32); // Should be different argument
        Argument result = b.arg(I32);

        b.store(b.add(x1, x2), result);
    }},
    {[](FunctionBuilder &b) {
        // Test signed and unsigned min/max behavior
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Argument signed_min = b.arg(I32);
        Argument signed_max = b.arg(I32);
        Argument unsigned_min = b.arg(I32);
        Argument unsigned_max = b.arg(I32);

        b.store(b.min(a, b_arg), signed_min);
        b.store(b.max(a, b_arg), signed_max);
        b.store(b.umin(a, b_arg), unsigned_min);
        b.store(b.umax(a, b_arg), unsigned_max);
    }},

    {[](FunctionBuilder &b) {
        // Test and_agg three shift types
        Value val = b.input_arg(I32);
        Value shift = b.input_arg(I32);
        Argument sll_result = b.arg(I32);
        Argument srl_result = b.arg(I32);
        Argument sra_result = b.arg(I32);

        b.store(b.sll(val, shift), sll_result);
        b.store(b.srl(val, shift), srl_result);
        b.store(b.sra(val, shift), sra_result);
    }},
    {[](FunctionBuilder &b) {
         // Test using input_splat_arg for compile-time constants
         Value runtime_val = b.input_arg(I64);
         Value const_val = b.input_splat_arg(I64); // Fixed for and_agg rows
         Argument result = b.arg(I64);

         // Multiply runtime value by constant
         b.store(b.mul(runtime_val, const_val), result);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Test: min(val << 2, val >> 1)
        Value val = b.input_arg(I16);
        Argument result = b.arg(I16);

        b.store(b.min(b.sll(val, b.i16(2)), b.sra(val, b.i16(1))), result);
    }},
    {[](FunctionBuilder &b) {
        // Test multiple input_splat_arg with arithmetic
        Value a = b.input_arg(I32);
        Value const1 = b.input_splat_arg(I32);
        Value const2 = b.input_splat_arg(I32);
        Argument result = b.arg(I32);

        // (a * const1) + (const2 << 3)
        b.store(b.add(b.mul(a, const1), b.sll(const2, b.i32(3))), result);
    }},
    {[](FunctionBuilder &b) {
        // Test umin/umax with bit manipulation
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument umin_result = b.arg(I32);
        Argument umax_result = b.arg(I32);

        // Clear lower 4 bits before comparison
        Value mask = b.u32(0xFFFFFFF0);
        Value x_masked = b.and_(x, mask);
        Value y_masked = b.and_(y, mask);

        b.store(b.umin(x_masked, y_masked), umin_result);
        b.store(b.umax(x_masked, y_masked), umax_result);
    }},
    {[](FunctionBuilder &b) {
        // Test: ((val << 2) >> 1) arithmetic vs logical
        Value val = b.input_arg(I16);
        Argument sll_sra = b.arg(I16);
        Argument sll_srl = b.arg(I16);

        b.store(b.sra(b.sll(val, b.i16(2)), b.i16(1)), sll_sra);
        b.store(b.srl(b.sll(val, b.i16(2)), b.i16(1)), sll_srl);
    }},
    {[](FunctionBuilder &b) {
         // Test aggregates with input_splat_arg
         Value val = b.input_arg(I64);
         Value const_factor = b.input_splat_arg(I64);
         Argument sum_result = b.arg(I64);

         // Sum of (val * const_factor)
         b.sum(b.mul(val, const_factor), sum_result);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Test min/max against constant bounds
        Value val = b.input_arg(I32);
        Value lower_bound = b.input_splat_arg(I32);
        Value upper_bound = b.input_splat_arg(I32);
        Argument clamped = b.arg(I32);

        // Clamp value between lower_bound and upper_bound
        Value temp = b.max(val, lower_bound);
        b.store(b.min(temp, upper_bound), clamped);
    }},
    {[](FunctionBuilder &b) {
        // Test SRL in aggregate context
        Value val = b.input_arg(I32);
        Value shift_amount = b.input_splat_arg(I32);
        Argument sum_shifted = b.arg(I32);

        // Sum of (val >> shift_amount) logical shift
        b.sum(b.srl(val, shift_amount), sum_shifted);
    }},
    {[](FunctionBuilder &b) {
        // Test shifting by variable (not constant) amount
        Value value = b.input_arg(I64);
        Value shift_var = b.input_arg(I64);
        Argument result_left = b.arg(I64);
        Argument result_right = b.arg(I64);

        // Mask shift amount to prevent undefined behavior
        Value safe_shift = b.and_(shift_var, b.i64(63));
        b.store(b.sll(value, safe_shift), result_left);
        b.store(b.sra(value, safe_shift), result_right);
    }},
    {[](FunctionBuilder &b) {
        // Compare signed min/max with unsigned min/max on same values
        Value a = b.input_arg(I16);
        Value b_arg = b.input_arg(I16);
        Argument signed_comp = b.arg(I16);
        Argument unsigned_comp = b.arg(I16);

        // These can differ for negative numbers
        b.store(b.min(a, b_arg), signed_comp);
        b.store(b.umin(a, b_arg), unsigned_comp);
    }},
    {[](FunctionBuilder &b) {
        // Test complex bit manipulation using input_splat_arg
        Value val = b.input_arg(I32);
        Value mask1 = b.input_splat_arg(I32);
        Value mask2 = b.input_splat_arg(I32);
        Value shift = b.input_splat_arg(I32);
        Argument result = b.arg(I32);

        // ((val & mask1) << shift) | ((val & mask2) >> shift)
        b.store(b.or_(b.sll(b.and_(val, mask1), shift), b.srl(b.and_(val, mask2), shift)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test nested min/max: min(max(a,b), max(c,d))
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);
        Value d = b.input_arg(I32);
        Argument result = b.arg(I32);

        b.store(b.min(b.max(a, b_arg), b.max(c, d)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test aggregates on shifted values
        Value value = b.input_arg(I64);
        Argument sum_left = b.arg(I64);
        Argument min_right = b.arg(I64);
        Argument max_right = b.arg(I64);

        b.sum(b.sll(value, b.i64(1)), sum_left);      // sum of (value << 1)
        b.min_agg(b.sra(value, b.i64(2)), min_right); // min of (value >> 2) arithmetic
        b.max_agg(b.srl(value, b.i64(2)), max_right); // max of (value >> 2) logical
    }},
    {[](FunctionBuilder &b) {
        // Test: if const_flag then a << 2 else a >> 2
        Value a = b.input_arg(I32);
        Value const_flag = b.input_splat_arg(I32); // 0 or 1
        Argument result = b.arg(I32);

        // Using bitwise trick: result = ((const_flag & (a << 2)) | (~const_flag & (a >> 2)))
        Value shifted_left = b.sll(a, b.i32(2));
        Value shifted_right = b.sra(a, b.i32(2));
        Value mask = b.not_(const_flag);

        b.store(b.or_(b.and_(const_flag, shifted_left), b.and_(mask, shifted_right)), result);
    }},
    {[](FunctionBuilder &b) {
        // Test umax against constant threshold
        Value val = b.input_arg(I32);
        Value threshold = b.input_splat_arg(I32);
        Argument result = b.arg(I32);

        // result = umax(val, threshold)
        b.store(b.umax(val, threshold), result);
    }},
    {[](FunctionBuilder &b) {
        // Test: min(val << 1, umax(val >> 1, constant))
        Value val = b.input_arg(I16);
        Value constant = b.input_splat_arg(I16);
        Argument result = b.arg(I16);

        b.store(b.min(b.sll(val, b.i16(1)), b.umax(b.srl(val, b.i16(1)), constant)), result);
    }},
    {[](FunctionBuilder &b) {
        // Multiple independent stores with different types
        Value i8_val = b.input_arg(I8);
        Value i16_val = b.input_arg(I16);
        Value i32_val = b.input_arg(I32);
        Value i64_val = b.input_arg(I64);

        Argument result1 = b.arg(I8);
        Argument result2 = b.arg(I16);
        Argument result3 = b.arg(I32);
        Argument result4 = b.arg(I64);

        // I8: (val * 2) & 0x7F
        b.store(b.and_(b.sll(i8_val, b.i8(1)), b.i8(0x7F)), result1);

        // I16: umax(val << 1, 100)
        b.store(b.umax(b.sll(i16_val, b.i16(1)), b.i16(100)), result2);

        // I32: (val + constant) >> 2 (arithmetic)
        Value const32 = b.input_splat_arg(I32);
        b.store(b.sra(b.add(i32_val, const32), b.i32(2)), result3);

        // I64: min(val, val * 3)
        b.store(b.min(i64_val, b.mul(i64_val, b.i64(3))), result4);
    }},
    {[](FunctionBuilder &b) {
        // Different aggregates on different type computations
        Value base_i32 = b.input_arg(I32);
        Value offset_i32 = b.input_splat_arg(I32);

        Argument sum_result = b.arg(I32);
        Argument min_result = b.arg(I32);
        Argument max_result = b.arg(I32);
        Argument product_result = b.arg(I32);

        // Sum of: (base + offset) << 1
        b.sum(b.sll(b.add(base_i32, offset_i32), b.i32(1)), sum_result);

        // Min of: umax(base, 1000)
        b.min_agg(b.umax(base_i32, b.i32(1000)), min_result);

        // Max of: base & 0xFFFF0000 (clear lower 16 bits)
        b.max_agg(b.and_(base_i32, b.u32(0xFFFF0000)), max_result);

        // Product of: (base - offset) (promoted to I64 for product)
        b.product(b.sub(base_i32, offset_i32), product_result);
    }},
    {[](FunctionBuilder &b) {
         // Three completely independent computation chains
         // Chain 1: I8 operations
         Value a_i8 = b.input_arg(I8);
         Value b_i8 = b.input_arg(I8);
         Argument chain1_out = b.arg(I8);
         b.store(b.umin(a_i8, b_i8), chain1_out);

         // Chain 2: I16 operations
         Value x_i16 = b.input_arg(I16);
         Value y_i16 = b.input_arg(I16);
         Argument chain2_out = b.arg(I16);
         b.store(b.sra(b.add(x_i16, y_i16), b.i16(1)), chain2_out);

         // Chain 3: I32 operations
         Value p_i32 = b.input_arg(I32);
         Value q_i32 = b.input_splat_arg(I32);
         Argument chain3_out = b.arg(I32);
         b.store(b.mul(p_i32, q_i32), chain3_out);

         // Chain 4: I64 aggregates
         Value val_i64 = b.input_arg(I64);
         Argument sum_i64 = b.arg(I64);
         Argument prod_i64 = b.arg(I64);
         b.sum(val_i64, sum_i64);
         b.product(val_i64, prod_i64);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Multiple min/max aggregates on different expressions
        Value value = b.input_arg(I32);
        Value threshold = b.input_splat_arg(I32);

        Argument min_signed = b.arg(I32);
        Argument max_signed = b.arg(I32);
        Argument min_unsigned = b.arg(I32);
        Argument max_unsigned = b.arg(I32);

        // Signed min/max of (value * 2)
        Value doubled = b.mul(value, b.i32(2));
        b.store(b.min(doubled, threshold), min_signed);
        b.store(b.max(doubled, threshold), max_signed);

        // Unsigned min/max of (value >> 1)
        Value halved = b.srl(value, b.i32(1));
        b.store(b.umin(halved, threshold), min_unsigned);
        b.store(b.umax(halved, threshold), max_unsigned);
    }},
    {[](FunctionBuilder &b) {
         // Both store and aggregate the same computation
         Value input = b.input_arg(I64);
         Value multiplier = b.input_splat_arg(I64);

         Value intermediate = b.mul(input, multiplier);
         Argument stored_result = b.arg(I64);
         Argument aggregated_sum = b.arg(I64);
         Argument aggregated_min = b.arg(I64);

         // Store the intermediate value
         b.store(intermediate, stored_result);

         // Also aggregate it
         b.sum(intermediate, aggregated_sum);
         b.min_agg(intermediate, aggregated_min);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Different bit manipulation patterns per type
        Value i8_data = b.input_arg(I8);
        Value i16_data = b.input_arg(I16);
        Value i32_data = b.input_arg(I32);

        Argument i8_out = b.arg(I8);
        Argument i16_out = b.arg(I16);
        Argument i32_out = b.arg(I32);

        // I8: Clear high bit, set low bit
        b.store(b.or_(b.and_(i8_data, b.i8(0x7F)), b.i8(0x01)), i8_out);

        // I16: Swap bytes (rough approximation with shifts)
        Value high_byte = b.sll(i16_data, b.i16(8));
        Value low_byte = b.srl(i16_data, b.i16(8));
        b.store(b.or_(high_byte, low_byte), i16_out);

        // I32: Arithmetic shift with sign extension
        b.store(b.sra(b.sll(i32_data, b.i32(16)), b.i32(16)), i32_out);
    }},
    {[](FunctionBuilder &b) {
        // Multiple sum aggregates with different computation bases
        Value base_val = b.input_arg(I32);
        Value modifier = b.input_splat_arg(I32);

        Argument sum1 = b.arg(I32); // Sum of base_val
        Argument sum2 = b.arg(I32); // Sum of base_val + modifier
        Argument sum3 = b.arg(I32); // Sum of base_val << 1
        Argument sum4 = b.arg(I32); // Sum of base_val & 0xFF

        b.sum(base_val, sum1);
        b.sum(b.add(base_val, modifier), sum2);
        b.sum(b.sll(base_val, b.i32(1)), sum3);
        b.sum(b.and_(base_val, b.i32(0xFF)), sum4);
    }},
    {[](FunctionBuilder &b) {
         Value value = b.input_arg(I64);
         Value constant = b.input_splat_arg(I64);

         Argument store_result = b.arg(I64);
         Argument sum_result = b.arg(I64);
         Argument product_result = b.arg(I64);
         Argument min_result = b.arg(I64);
         Argument max_result = b.arg(I64);

         // Store: value * constant
         b.store(b.mul(value, constant), store_result);

         // Sum: value + constant
         b.sum(b.add(value, constant), sum_result);

         // Product: value - constant
         b.product(b.sub(value, constant), product_result);

         // Min: umin(value, constant)
         b.min_agg(b.umin(value, constant), min_result);

         // Max: value >> 2 (logical)
         b.max_agg(b.srl(value, b.i64(2)), max_result);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Process different fields independently (like in columnar processing)
         Value field1 = b.input_arg(I16);
         Value field2 = b.input_arg(I32);
         Value field3 = b.input_arg(I8);

         Argument out1 = b.arg(I16);
         Argument out2 = b.arg(I32);
         Argument out3 = b.arg(I8);
         Argument agg1 = b.arg(I32);
         Argument agg2 = b.arg(I64);

         // Field1 processing: clamp between 0 and 1000 (unsigned)
         Value clamped = b.umin(b.umax(field1, b.i16(0)), b.i16(1000));
         b.store(clamped, out1);

         // Field2 processing: (field2 * 3) >> 1
         b.store(b.sra(b.mul(field2, b.i32(3)), b.i32(1)), out2);

         // Field3 processing: popcount of field3
         b.store(b.popcnt(field3), out3);

         // Aggregate on field2: sum of abs(field2)
         b.sum(b.abs(field2), agg1);

         // Aggregate on field1: product of field1 + 1
         b.product(b.sext(b.add(field1, b.i16(1)), I64), agg2);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value mask = b.input_splat_arg(I32);

        Argument result1 = b.arg(I32); // Select a or b based on mask bit 0
        Argument result2 = b.arg(I32); // Select min or max based on mask bit 1
        Argument result3 = b.arg(I32); // Shift direction based on mask bit 2

        // result1: mask[0] ? a : b
        Value mask_bit0 = b.and_(mask, b.i32(1));
        Value select_a = b.and_(a, mask_bit0);
        Value not_mask = b.not_(mask_bit0);
        Value select_b = b.and_(b_arg, not_mask);
        b.store(b.or_(select_a, select_b), result1);

        // result2: mask[1] ? min(a,b) : max(a,b)
        Value mask_bit1 = b.and_(mask, b.i32(2));
        Value min_ab = b.min(a, b_arg);
        Value max_ab = b.max(a, b_arg);
        Value select_min = b.and_(min_ab, mask_bit1);
        Value not_mask1 = b.not_(mask_bit1);
        Value select_max = b.and_(max_ab, not_mask1);
        b.store(b.or_(select_min, select_max), result2);

        // result3: mask[2] ? a << 1 : a >> 1
        Value mask_bit2 = b.and_(mask, b.i32(4));
        Value shift_left = b.sll(a, b.i32(1));
        Value shift_right = b.sra(a, b.i32(1));
        Value select_left = b.and_(shift_left, mask_bit2);
        Value not_mask2 = b.not_(mask_bit2);
        Value select_right = b.and_(shift_right, not_mask2);
        b.store(b.or_(select_left, select_right), result3);
    }},
    {[](FunctionBuilder &b) {
         // Mix of I8, I16, I32 operations with stores and aggregates
         Value i8_input = b.input_arg(I8);
         Value i16_input = b.input_arg(I16);
         Value i32_input = b.input_arg(I32);

         Argument i8_store = b.arg(I8);
         Argument i16_store = b.arg(I16);
         Argument i32_sum = b.arg(I32);
         Argument i64_product = b.arg(I64);
         Argument i32_min = b.arg(I32);
         Argument i32_max = b.arg(I32);

         // I8 store: popcnt(i8_input)
         b.store(b.popcnt(i8_input), i8_store);

         // I16 store: i16_input with high byte cleared
         b.store(b.and_(i16_input, b.i16(0xFF)), i16_store);

         // I32 sum: i32_input * 10
         b.sum(b.mul(i32_input, b.i32(10)), i32_sum);

         // I64 product: zero-extended i16_input
         b.product(b.sext(i16_input, I64), i64_product);

         // I32 min/max on different expressions
         b.min_agg(b.sll(i32_input, b.i32(2)), i32_min);
         b.max_agg(b.sra(i32_input, b.i32(2)), i32_max);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Value base = b.input_arg(I64);
         Value offset = b.input_splat_arg(I64);

         // Dependent chain: out1 -> out2 -> out3
         Argument out1 = b.arg(I64); // base + offset
         Argument out2 = b.arg(I64); // out1 * 2
         Argument out3 = b.arg(I64); // out2 >> 1

         Value temp1 = b.add(base, offset);
         b.store(temp1, out1);
         Value temp2 = b.mul(temp1, b.i64(2));
         b.store(temp2, out2);
         b.store(b.sra(temp2, b.i64(1)), out3);

         // Independent aggregates
         Argument sum_out = b.arg(I64);  // sum of base
         Argument prod_out = b.arg(I64); // product of base + 1
         Argument min_out = b.arg(I64);  // min of base & 0xFF
         Argument max_out = b.arg(I64);  // max of umin(base, 1000)

         b.sum(base, sum_out);
         b.product(b.add(base, b.i64(1)), prod_out);
         b.min_agg(b.and_(base, b.i64(0xFF)), min_out);
         b.max_agg(b.umin(base, b.i64(1000)), max_out);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Same (x + y) used twice - should be CSE'd
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Argument sum1 = b.arg(I32);
        Argument sum2 = b.arg(I32);

        // These two adds are identical
        Value expr1 = b.add(x, y);
        Value expr2 = b.add(x, y); // Should be eliminated

        b.store(expr1, sum1);
        b.store(expr2, sum2);
    }},
    {[](FunctionBuilder &b) {
         // Same expression used in store, sum, and product
         Value a = b.input_arg(I64);
         Value b_arg = b.input_arg(I64);
         Value constant = b.i64(100);

         Value common_expr = b.mul(b.add(a, b_arg), constant); // (a + b) * 100

         Argument store_out = b.arg(I64);
         Argument sum_out = b.arg(I64);
         Argument product_out = b.arg(I64);
         Argument min_out = b.arg(I64);

         b.store(common_expr, store_out);     // Store it
         b.sum(common_expr, sum_out);         // Sum it
         b.product(common_expr, product_out); // Product it
         b.min_agg(common_expr, min_out);     // Min of it
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Common subexpression inside another expression
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);

        Value inner = b.mul(x, y); // x * y

        // Use inner in multiple places
        Value expr1 = b.add(inner, z);        // (x*y) + z
        Value expr2 = b.sub(inner, z);        // (x*y) - z
        Value expr3 = b.mul(inner, b.i32(2)); // (x*y) * 2

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);

        b.store(expr1, out1);
        b.store(expr2, out2);
        b.store(expr3, out3);
    }},
    {[](FunctionBuilder &b) {
        // Same computation expressed in different types - might or might not be CSE'd
        Value a = b.input_arg(I16);
        Value b_arg = b.input_arg(I16);

        // Same operation in different types
        Value add_i16 = b.add(a, b_arg); // I16 add
        Value add_i32 = b.add(a, b_arg); // Another I16 add - should be CSE'd

        // But if we explicitly cast to different type, might be different
        Value a_i32 = b.input_arg(I32);
        Value b_i32 = b.input_arg(I32);
        Value add_i32_2 = b.add(a_i32, b_i32); // Different type, different expression

        Argument out1 = b.arg(I16);
        Argument out2 = b.arg(I16);
        Argument out3 = b.arg(I32);

        b.store(add_i16, out1);
        b.store(add_i32, out2);
        b.store(add_i32_2, out3);
    }},
    {[](FunctionBuilder &b) {
         // Large expression with multiple shared subexpressions
         Value x = b.input_arg(I64);
         Value y = b.input_arg(I64);
         Value z = b.input_arg(I64);

         // Shared subexpressions
         Value x_plus_y = b.add(x, y);  // Used 3 times
         Value x_times_z = b.mul(x, z); // Used 2 times
         Value y_minus_z = b.sub(y, z); // Used 2 times

         // Expression 1: (x+y) * (x*z)
         Value expr1 = b.mul(x_plus_y, x_times_z);

         // Expression 2: (x+y) + (y-z)
         Value expr2 = b.add(x_plus_y, y_minus_z);

         // Expression 3: (x*z) - (y-z)
         Value expr3 = b.sub(x_times_z, y_minus_z);

         // Expression 4: (x+y) * 2
         Value expr4 = b.mul(x_plus_y, b.i64(2));

         Argument out1 = b.arg(I64);
         Argument out2 = b.arg(I64);
         Argument out3 = b.arg(I64);
         Argument out4 = b.arg(I64);

         b.store(expr1, out1);
         b.store(expr2, out2);
         b.store(expr3, out3);
         b.store(expr4, out4);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Expressions with same constant computations
        Value x = b.input_arg(I32);
        Value const_expr1 = b.mul(x, b.i32(100)); // x * 100
        Value const_expr2 = b.mul(x, b.i32(100)); // Same - should be CSE'd

        Value const_expr3 = b.add(x, b.i32(50)); // x + 50
        Value const_expr4 = b.add(x, b.i32(50)); // Same - should be CSE'd

        // Different constants - should NOT be CSE'd
        Value const_expr5 = b.mul(x, b.i32(200)); // Different constant

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);
        Argument out4 = b.arg(I32);
        Argument out5 = b.arg(I32);

        b.store(const_expr1, out1);
        b.store(const_expr2, out2);
        b.store(const_expr3, out3);
        b.store(const_expr4, out4);
        b.store(const_expr5, out5);
    }},
    {[](FunctionBuilder &b) {
        // Common bitmask operations
        Value val = b.input_arg(I32);
        Value mask1 = b.u32(0xFF00FF00);
        Value mask2 = b.i32(0x00FF00FF);

        // Common subexpressions
        Value masked_high = b.and_(val, mask1); // val & 0xFF00FF00
        Value masked_low = b.and_(val, mask2);  // val & 0x00FF00FF

        // Use masked_high multiple times
        Value expr1 = b.sll(masked_high, b.i32(8));   // (val & mask1) << 8
        Value expr2 = b.srl(masked_high, b.i32(8));   // (val & mask1) >> 8
        Value expr3 = b.or_(masked_high, masked_low); // (val & mask1) | (val & mask2)

        // Use masked_low multiple times
        Value expr4 = b.xor_(masked_low, mask2);     // (val & mask2) ^ mask2
        Value expr5 = b.add(masked_low, masked_low); // (val & mask2) + (val & mask2)

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);
        Argument out4 = b.arg(I32);
        Argument out5 = b.arg(I32);

        b.store(expr1, out1);
        b.store(expr2, out2);
        b.store(expr3, out3);
        b.store(expr4, out4);
        b.store(expr5, out5);
    }},
    {[](FunctionBuilder &b) {
         // Multiple aggregates on the same base expression
         Value base = b.input_arg(I64);
         Value offset = b.input_splat_arg(I64);

         Value common = b.mul(b.add(base, offset), b.i64(10)); // (base + offset) * 10

         Argument sum_result = b.arg(I64);
         Argument min_result = b.arg(I64);
         Argument max_result = b.arg(I64);
         Argument product_result = b.arg(I64);
         Argument store_result = b.arg(I64);

         // All using the same common expression
         b.sum(common, sum_result);
         b.min_agg(common, min_result);
         b.max_agg(common, max_result);
         b.product(common, product_result);
         b.store(common, store_result);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Common min/max expressions
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);

        Value min_ab = b.min(a, b_arg); // min(a, b)
        Value max_ab = b.max(a, b_arg); // max(a, b)

        // Use min_ab multiple times
        Value expr1 = b.add(min_ab, c);      // min(a,b) + c
        Value expr2 = b.sub(min_ab, c);      // min(a,b) - c
        Value expr3 = b.mul(min_ab, min_ab); // min(a,b) * min(a,b)

        // Use max_ab multiple times
        Value expr4 = b.umax(max_ab, c); // umax(max(a,b), c)
        Value expr5 = b.umin(max_ab, c); // umin(max(a,b), c)

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);
        Argument out4 = b.arg(I32);
        Argument out5 = b.arg(I32);

        b.store(expr1, out1);
        b.store(expr2, out2);
        b.store(expr3, out3);
        b.store(expr4, out4);
        b.store(expr5, out5);
    }},
    {[](FunctionBuilder &b) {
         // Common shift expressions
         Value val = b.input_arg(I64);
         Value shift1 = b.i64(4);
         Value shift2 = b.i64(2);

         Value shifted_left = b.sll(val, shift1);  // val << 4
         Value shifted_right = b.sra(val, shift2); // val >> 2 (arithmetic)

         // Use shifted_left multiple times
         Value expr1 = b.add(shifted_left, val);          // (val << 4) + val
         Value expr2 = b.sub(shifted_left, val);          // (val << 4) - val
         Value expr3 = b.mul(shifted_left, shifted_left); // (val << 4) * (val << 4)

         // Use shifted_right multiple times
         Value expr4 = b.and_(shifted_right, b.i64(0xFF)); // (val >> 2) & 0xFF
         Value expr5 = b.or_(shifted_right, shifted_left); // (val >> 2) | (val << 4)

         Argument out1 = b.arg(I64);
         Argument out2 = b.arg(I64);
         Argument out3 = b.arg(I64);
         Argument out4 = b.arg(I64);
         Argument out5 = b.arg(I64);

         b.store(expr1, out1);
         b.store(expr2, out2);
         b.store(expr3, out3);
         b.store(expr4, out4);
         b.store(expr5, out5);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // CSE with input_splat_arg (should work since constants are the same)
        Value runtime_val = b.input_arg(I32);
        Value const1 = b.input_splat_arg(I32); // Same constant in both uses
        Value const2 = b.input_splat_arg(I32); // Might be different or same

        // These should be CSE'd if const1 is the same in both positions
        Value expr1 = b.mul(runtime_val, const1); // val * const1
        Value expr2 = b.mul(runtime_val, const1); // Same - should be CSE'd

        // These might or might not be CSE'd depending if const1 == const2
        Value expr3 = b.add(runtime_val, const1); // val + const1
        Value expr4 = b.add(runtime_val, const2); // val + const2

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);
        Argument out4 = b.arg(I32);

        b.store(expr1, out1);
        b.store(expr2, out2);
        b.store(expr3, out3);
        b.store(expr4, out4);
    }},
    {[](FunctionBuilder &b) {
        // Complex DAG with multiple levels of sharing
        Value w = b.input_arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        Value z = b.input_arg(I32);

        // Level 1: Basic operations
        Value w_plus_x = b.add(w, x);  // w + x
        Value y_times_z = b.mul(y, z); // y * z

        // Level 2: Combine level 1 results (shared)
        Value sum1 = b.add(w_plus_x, y_times_z);  // (w+x) + (y*z)
        Value diff1 = b.sub(w_plus_x, y_times_z); // (w+x) - (y*z)

        // Level 3: Use level 2 results in multiple places
        // sum1 used twice
        Value expr1 = b.mul(sum1, b.i32(2)); // sum1 * 2
        Value expr2 = b.sra(sum1, b.i32(1)); // sum1 >> 1

        // diff1 used twice
        Value expr3 = b.umin(diff1, b.i32(100)); // umin(diff1, 100)
        Value expr4 = b.umax(diff1, b.i32(0));   // umax(diff1, 0)

        // Also reuse level 1 expressions
        Value expr5 = b.xor_(w_plus_x, y_times_z); // (w+x) ^ (y*z)

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);
        Argument out4 = b.arg(I32);
        Argument out5 = b.arg(I32);

        b.store(expr1, out1);
        b.store(expr2, out2);
        b.store(expr3, out3);
        b.store(expr4, out4);
        b.store(expr5, out5);
    }},
    {[](FunctionBuilder &b) {
        // Two identical chains of operations
        Value a = b.input_arg(I16);
        Value b_arg = b.input_arg(I16);

        // Chain 1
        Value step1_1 = b.add(a, b_arg);          // a + b
        Value step2_1 = b.mul(step1_1, b.i16(3)); // (a+b) * 3
        Value step3_1 = b.sll(step2_1, b.i16(1)); // ((a+b)*3) << 1

        // Chain 2 - identical to Chain 1
        Value step1_2 = b.add(a, b_arg);          // Same as step1_1
        Value step2_2 = b.mul(step1_2, b.i16(3)); // Same as step2_1
        Value step3_2 = b.sll(step2_2, b.i16(1)); // Same as step3_1

        // Chain 3 - different constant
        Value step1_3 = b.add(a, b_arg);          // Same as step1_1
        Value step2_3 = b.mul(step1_3, b.i16(4)); // Different constant
        Value step3_3 = b.sll(step2_3, b.i16(1)); // Same shift

        Argument out1 = b.arg(I16);
        Argument out2 = b.arg(I16);
        Argument out3 = b.arg(I16);

        b.store(step3_1, out1);
        b.store(step3_2, out2);
        b.store(step3_3, out3);
    }},
    {[](FunctionBuilder &b) {
         // Intermediate result used in multiple final expressions
         Value base = b.input_arg(I64);

         // Intermediate computations
         Value squared = b.mul(base, base);     // base²
         Value doubled = b.mul(base, b.i64(2)); // base * 2
         Value shifted = b.sll(base, b.i64(4)); // base << 4

         // Final expressions sharing intermediates
         // Expression A uses squared and doubled
         Value exprA = b.add(squared, doubled); // base² + (base*2)

         // Expression B uses squared and shifted
         Value exprB = b.sub(squared, shifted); // base² - (base<<4)

         // Expression C uses doubled and shifted
         Value exprC = b.mul(doubled, shifted); // (base*2) * (base<<4)

         // Expression D reuses squared
         Value exprD = b.mul(squared, b.i64(10)); // base² * 10

         // Expression E reuses shifted
         Value exprE = b.sra(shifted, b.i64(2)); // (base<<4) >> 2

         Argument outA = b.arg(I64);
         Argument outB = b.arg(I64);
         Argument outC = b.arg(I64);
         Argument outD = b.arg(I64);
         Argument outE = b.arg(I64);

         b.store(exprA, outA);
         b.store(exprB, outB);
         b.store(exprC, outC);
         b.store(exprD, outD);
         b.store(exprE, outE);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Test CSE detection with non-obvious identical expressions
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);

        // These should be recognized as identical (commutative property)
        Value expr1 = b.add(x, y); // x + y
        Value expr2 = b.add(y, x); // y + x - should be CSE'd with expr1

        // These should also be identical (associative property)
        Value expr3 = b.add(b.add(x, y), b.i32(5)); // (x + y) + 5
        Value expr4 = b.add(x, b.add(y, b.i32(5))); // x + (y + 5) - might not be CSE'd

        // Min/Max commutative
        Value expr5 = b.min(x, y); // min(x, y)
        Value expr6 = b.min(y, x); // min(y, x) - should be CSE'd

        // Multiplication commutative
        Value expr7 = b.mul(x, b.i32(3)); // x * 3
        Value expr8 = b.mul(b.i32(3), x); // 3 * x - should be CSE'd

        Argument out1 = b.arg(I32);
        Argument out2 = b.arg(I32);
        Argument out3 = b.arg(I32);
        Argument out4 = b.arg(I32);
        Argument out5 = b.arg(I32);
        Argument out6 = b.arg(I32);
        Argument out7 = b.arg(I32);
        Argument out8 = b.arg(I32);

        b.store(expr1, out1);
        b.store(expr2, out2);
        b.store(expr3, out3);
        b.store(expr4, out4);
        b.store(expr5, out5);
        b.store(expr6, out6);
        b.store(expr7, out7);
        b.store(expr8, out8);
    }},
    {[](FunctionBuilder &b) {
        // Deep chain of operations with proper type conversions
        Value i8_val = b.input_arg(I8);
        Value i32_val = b.input_arg(I32);

        // Chain: sext(i8) -> add with i32 -> mul by constant -> trunc to i16 -> sext to i64
        Value extended_i8 = b.sext(i8_val, I32);
        Value sum_i32 = b.add(extended_i8, i32_val);
        Value mul_i32 = b.mul(sum_i32, b.i32(100));
        Value trunc_i16 = b.trunc(mul_i32, I16);
        Value final_i64 = b.sext(trunc_i16, I64);

        Argument result = b.arg(I64);
        b.store(final_i64, result);
    }},
    {[](FunctionBuilder &b) {
        // I64 multiplication with sign-extended I32 values
        Value i32_a = b.input_arg(I32);
        Value i32_b = b.input_arg(I32);

        // Sign extend both to I64
        Value i64_a = b.sext(i32_a, I64);
        Value i64_b = b.sext(i32_b, I64);

        // Valid I64 multiplication (extended from I32)
        Value product = b.mul(i64_a, i64_b);

        // Also multiply by a constant (valid)
        Value scaled = b.mul(product, b.i64(3));

        Argument result = b.arg(I64);
        b.store(scaled, result);
    }},
    {[](FunctionBuilder &b) {
        // I128 sum with I64 input (the only cross-type aggregate allowed)
        Value i64_val = b.input_arg(I64);

        // Process the I64 value through some operations
        Value processed = b.add(i64_val, b.i64(1000));
        Value shifted = b.sll(processed, b.i64(2));

        // I128 sum of the I64 value
        Argument sum_result = b.arg(I128);
        b.sum(shifted, sum_result);
    }},
    {[](FunctionBuilder &b) {
        // Test type constraints in a single complex expression
        Value i16_val = b.input_arg(I16);
        Value i32_val = b.input_arg(I32);
        Value i64_val = b.input_arg(I64);

        // Convert everything to I64
        Value i64_from_i16 = b.sext(i16_val, I64);
        Value i64_from_i32 = b.sext(i32_val, I64);

        // All operations now on I64
        Value sum1 = b.add(i64_from_i16, i64_from_i32);
        Value sum2 = b.add(sum1, i64_val);

        // Multiply by constant (valid I64 multiplication)
        Value product = b.mul(sum2, b.i64(10));

        // Truncate to I32 for final store
        Value final_i32 = b.trunc(product, I32);

        Argument result = b.arg(I32);
        b.store(final_i32, result);
    }},
    {[](FunctionBuilder &b) {
        // Bitwise operations with proper type handling
        Value i8_val = b.input_arg(I8);
        Value i32_val = b.input_arg(I32);

        // Extend i8 to i32
        Value extended_i8 = b.zext(i8_val, I32);

        // Bitwise operations in i32
        Value and_result = b.and_(extended_i8, b.i32(0xF0));
        Value or_result = b.or_(and_result, i32_val);
        Value shifted = b.sll(or_result, b.i32(4));

        // Final store
        Argument result = b.arg(I32);
        b.store(shifted, result);
    }},
    {[](FunctionBuilder &b) {
        // Min/max operations with type conversions
        Value i16_a = b.input_arg(I16);
        Value i16_b = b.input_arg(I16);

        // Convert to I32 for operations
        Value i32_a = b.sext(i16_a, I32);
        Value i32_b = b.sext(i16_b, I32);

        // Find min and max
        Value min_val = b.min(i32_a, i32_b);
        Value max_val = b.max(i32_a, i32_b);

        // Average: (min + max) / 2 (using shift for division)
        Value sum = b.add(min_val, max_val);
        Value avg = b.sra(sum, b.i32(1));

        Argument result = b.arg(I32);
        b.store(avg, result);
    }},
    {[](FunctionBuilder &b) {
        // Complex I64 multiplication using only allowed patterns
        Value i32_val1 = b.input_arg(I32);
        Value i32_val2 = b.input_arg(I32);
        Value i64_val = b.input_arg(I64);

        // Sign extend I32 values
        Value i64_from_i32_1 = b.sext(i32_val1, I64);
        Value i64_from_i32_2 = b.sext(i32_val2, I64);

        // Multiply extended I32 values (valid)
        Value product1 = b.mul(i64_from_i32_1, i64_from_i32_2);

        // Multiply with constant (valid)
        Value scaled = b.mul(product1, b.i64(5));

        // Add with original I64 value
        Value final_sum = b.add(scaled, i64_val);

        Argument result = b.arg(I64);
        b.store(final_sum, result);
    }},
    {[](FunctionBuilder &b) {
        // Aggregates with type preservation
        Value i32_val = b.input_arg(I32);

        // Create a complex expression but keep as I32
        Value expr = b.add(i32_val, b.i32(100));
        expr = b.mul(expr, b.i32(3));
        expr = b.and_(expr, b.i32(0xFF));

        // Store (same type)
        Argument store_result = b.arg(I32);
        b.store(expr, store_result);

        // Sum (same type)
        Argument sum_result = b.arg(I32);
        b.sum(expr, sum_result);
    }},
    {[](FunctionBuilder &b) {
        // Chain of truncations and extensions
        Value i64_val = b.input_arg(I64);

        // Truncate to I16, then extend back to I64 with sign extension
        Value truncated = b.trunc(i64_val, I16);
        Value extended = b.sext(truncated, I32);
        Value reextended = b.sext(extended, I64);

        // Multiply by constant (valid)
        Value result_val = b.mul(reextended, b.i64(2));

        Argument result = b.arg(I64);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // I64 multiplication with zero and sign extensions
        Value i32_val = b.input_arg(I32);
        Value u32_val = b.input_arg(I32); // Treated as unsigned

        // Sign extend for signed multiplication
        Value signed_i64 = b.sext(i32_val, I64);

        // Zero extend for unsigned multiplication context
        Value unsigned_i64 = b.zext(u32_val, I64);

        // Multiply each with constants (valid)
        Value signed_product = b.mul(signed_i64, b.i64(-2));
        Value unsigned_product = b.mul(unsigned_i64, b.i64(3));

        // Combine results
        Value combined = b.add(signed_product, unsigned_product);

        Argument result = b.arg(I64);
        b.store(combined, result);
    }},
    {[](FunctionBuilder &b) {
        // Deep expression ending in I128 sum
        Value i64_base = b.input_arg(I64);

        // Complex I64 expression
        Value expr = b.add(i64_base, b.i64(1000));
        expr = b.sll(expr, b.i64(2));
        expr = b.sub(expr, b.i64(500));

        // I128 sum of the result
        Argument sum_result = b.arg(I128);
        b.sum(expr, sum_result);

        // Also store the I64 value
        Argument store_result = b.arg(I64);
        b.store(expr, store_result);
    }},
    {[](FunctionBuilder &b) {
        // Type-safe arithmetic with shifts and conversions
        Value i8_val = b.input_arg(I8);
        Value i16_val = b.input_arg(I16);

        // Convert to common type I32
        Value i32_from_i8 = b.sext(i8_val, I32);
        Value i32_from_i16 = b.sext(i16_val, I32);

        // Arithmetic in I32
        Value shifted = b.sll(i32_from_i8, b.i32(4));
        Value added = b.add(shifted, i32_from_i16);
        Value multiplied = b.mul(added, b.i32(2));

        // Truncate to I16 for store
        Value final_i16 = b.trunc(multiplied, I16);

        Argument result = b.arg(I16);
        b.store(final_i16, result);
    }},
    {[](FunctionBuilder &b) {
        // Only use valid I64 multiplication patterns
        Value i32_input = b.input_arg(I32);

        // Pattern 1: Extended I32 * Extended I32
        Value ext1 = b.sext(i32_input, I64);
        Value ext2 = b.sext(i32_input, I64);
        Value product1 = b.mul(ext1, ext2);

        // Pattern 2: Extended I32 * Constant
        Value product2 = b.mul(ext1, b.i64(10));

        // Pattern 3: Constant * Constant (via intermediate)
        Value const_product = b.mul(b.i64(3), ext1);

        // Combine and_agg products
        Value total = b.add(product1, product2);
        total = b.add(total, const_product);

        Argument result = b.arg(I64);
        b.store(total, result);
    }},
    {[](FunctionBuilder &b) {
        // Single top-level expression: I128 sum of complex I64 computation
        Value i64_val = b.input_arg(I64);

        // Start with I64 operations
        Value expr = b.add(i64_val, b.i64(100));
        expr = b.sll(expr, b.i64(1)); // Multiply by 2
        expr = b.sub(expr, b.i64(50));

        // Apply min/max bounds
        Value bounded = b.umax(expr, b.i64(0));
        bounded = b.umin(bounded, b.i64(10000));

        // I128 sum of the bounded value
        Argument sum_result = b.arg(I128);
        b.sum(bounded, sum_result);
    }},
    {[](FunctionBuilder &b) {
        // Type conversions for min/max operations
        Value i8_val = b.input_arg(I8);
        Value i32_val = b.input_arg(I32);

        // Convert i8 to i32 for comparison
        Value i32_from_i8 = b.sext(i8_val, I32);

        // Find signed min
        Value min_val = b.min(i32_from_i8, i32_val);

        // Find unsigned max (convert to unsigned comparison via bitwise)
        Value i32_from_i8_unsigned = b.zext(i8_val, I32);
        Value max_val = b.umax(i32_from_i8_unsigned, i32_val);

        // Combine results
        Value combined = b.add(min_val, max_val);

        Argument result = b.arg(I32);
        b.store(combined, result);
    }},
    {[](FunctionBuilder &b) {
        // Deep nesting of type casts in a single expression
        Value i64_input = b.input_arg(I64);

        // Multiple truncate/extend operations
        Value i32_trunc = b.trunc(i64_input, I32);
        Value i16_trunc = b.trunc(i32_trunc, I16);
        Value i8_trunc = b.trunc(i16_trunc, I8);

        // Extend back up with different signedness
        Value i16_sext = b.sext(i8_trunc, I16);
        Value i32_zext = b.zext(i16_sext, I32);
        Value i64_sext = b.sext(i32_zext, I64);

        // Final operation
        Value final_val = b.add(i64_sext, b.i64(1));

        Argument result = b.arg(I64);
        b.store(final_val, result);
    }},
    {[](FunctionBuilder &b) {
        // I8 addition with overflow at boundaries
        Value i8_val = b.input_arg(I8);

        // Add constant that can cause overflow
        Value add_near_max = b.add(i8_val, b.i8(127));  // Max I8 is 127
        Value sub_near_min = b.sub(i8_val, b.i8(-128)); // Min I8 is -128

        Argument result1 = b.arg(I8);
        Argument result2 = b.arg(I8);
        b.store(add_near_max, result1);
        b.store(sub_near_min, result2);
    }},
    {[](FunctionBuilder &b) {
        // I16 multiplication that can overflow
        Value i16_val = b.input_arg(I16);

        // Multiply by values near I16 max/min
        Value mul_near_max = b.mul(i16_val, b.i16(32767));  // Max I16
        Value mul_near_min = b.mul(i16_val, b.i16(-32768)); // Min I16

        Argument result1 = b.arg(I16);
        Argument result2 = b.arg(I16);
        b.store(mul_near_max, result1);
        b.store(mul_near_min, result2);
    }},
    {[](FunctionBuilder &b) {
        // Chain of operations that can overflow in I32
        Value i32_val = b.input_arg(I32);

        // Expression: (val + 0x7FFFFFFF) * 2 - 0x80000000
        Value add_max = b.add(i32_val, b.i32(0x7FFFFFFF));
        Value mul_two = b.mul(add_max, b.i32(2));
        Value sub_min = b.sub(mul_two, b.u32(0x80000000));

        Argument result = b.arg(I32);
        b.store(sub_min, result);
    }},
    {[](FunctionBuilder &b) {
        // I64 multiplication with overflow potential from extended I32 values
        Value i32_val = b.input_arg(I32);

        // Sign extend and multiply by large I64 constant
        Value extended = b.sext(i32_val, I64);
        Value product = b.mul(extended, b.i64(0x7FFFFFFFFFFFFFFF)); // Max I64

        Argument result = b.arg(I64);
        b.store(product, result);
    }},
    {[](FunctionBuilder &b) {
        // Test shift wrap-around (shifts defined to wrap around)
        Value i32_val = b.input_arg(I32);

        // Shift by amounts >= bit width (32)
        Value shift_by_32 = b.sll(i32_val, b.i32(32));       // Should wrap to 0
        Value shift_by_40 = b.srl(i32_val, b.i32(40));       // Should wrap
        Value shift_by_33_arith = b.sra(i32_val, b.i32(33)); // Arithmetic shift wrap

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);
        Argument result3 = b.arg(I32);
        b.store(shift_by_32, result1);
        b.store(shift_by_40, result2);
        b.store(shift_by_33_arith, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test aggregates (sum) with values that can overflow the accumulator
        Value i64_val = b.input_arg(I64);

        // I128 sum of large I64 values (I128 accumulator shouldn't overflow easily)
        Argument sum_result = b.arg(I128);
        b.sum(i64_val, sum_result);

        // Also store the I64 value after adding max I64
        Value processed = b.add(i64_val, b.i64(0x7FFFFFFFFFFFFFFF));
        Argument store_result = b.arg(I64);
        b.store(processed, store_result);
    }},
    {[](FunctionBuilder &b) {
        // Multiple operations each with overflow potential
        Value i16_val = b.input_arg(I16);

        // Chain: ((val * 32767) + 32767) << 2
        Value mul = b.mul(i16_val, b.i16(32767));
        Value add = b.add(mul, b.i16(32767));
        Value shift = b.sll(add, b.i16(2));

        Argument result = b.arg(I16);
        b.store(shift, result);
    }},
    {[](FunctionBuilder &b) {
        // Overflow in type conversions and operations
        Value i8_val = b.input_arg(I8);

        // Extend i8 to i32, then add large i32 constant
        Value extended = b.sext(i8_val, I32);
        Value sum = b.add(extended, b.i32(0x7FFFFFFF));

        // Multiply by 2 (potential overflow)
        Value doubled = b.mul(sum, b.i32(2));

        Argument result = b.arg(I32);
        b.store(doubled, result);
    }},
    {[](FunctionBuilder &b) {
        // Bitwise operations that simulate overflow conditions
        Value i32_val = b.input_arg(I32);

        // Clear top bit then add to potentially set it
        Value cleared_top = b.and_(i32_val, b.i32(0x7FFFFFFF));
        Value with_sign_bit = b.or_(cleared_top, b.u32(0x80000000));

        // Shift left by 1 moves sign bit out
        Value shifted = b.sll(with_sign_bit, b.i32(1));

        Argument result = b.arg(I32);
        b.store(shifted, result);
    }},
    {[](FunctionBuilder &b) {
        // Min/max operations with boundary constants
        Value i32_val = b.input_arg(I32);

        // Signed min/max with min and max I32 values
        Value signed_min = b.min(i32_val, b.u32(0x80000000)); // Min I32
        Value signed_max = b.max(i32_val, b.i32(0x7FFFFFFF)); // Max I32

        // Unsigned min/max (treat as unsigned comparison)
        Value unsigned_min = b.umin(i32_val, b.u32(0x80000000)); // 0x80000000 is 2147483648 unsigned
        Value unsigned_max = b.umax(i32_val, b.i32(0x7FFFFFFF)); // 0x7FFFFFFF is 2147483647 unsigned

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);
        Argument result3 = b.arg(I32);
        Argument result4 = b.arg(I32);
        b.store(signed_min, result1);
        b.store(signed_max, result2);
        b.store(unsigned_min, result3);
        b.store(unsigned_max, result4);
    }},
    {[](FunctionBuilder &b) {
        // Nested arithmetic with overflow across type boundaries
        Value i16_val = b.input_arg(I16);

        // Convert to I32, do overflow-prone ops, then trunc back
        Value extended = b.sext(i16_val, I32);
        Value scaled = b.mul(extended, b.i32(0x10000)); // 65536, shifts by 16
        Value truncated = b.trunc(scaled, I16);

        Argument result = b.arg(I16);
        b.store(truncated, result);
    }},
    {[](FunctionBuilder &b) {
        // Shift by very large constants (wrap-around behavior)
        Value i64_val = b.input_arg(I64);

        // Shift by amounts >= 64 (should wrap around)
        Value shift_left = b.sll(i64_val, b.i64(64));
        Value shift_right_logical = b.srl(i64_val, b.i64(65));
        Value shift_right_arith = b.sra(i64_val, b.i64(66));

        Argument result1 = b.arg(I64);
        Argument result2 = b.arg(I64);
        Argument result3 = b.arg(I64);
        b.store(shift_left, result1);
        b.store(shift_right_logical, result2);
        b.store(shift_right_arith, result3);
    }},
    {[](FunctionBuilder &b) {
        // Chain of additions that can overflow cumulatively
        Value i8_val = b.input_arg(I8);

        // Add multiple boundary values: val + 127 + 1 + (-128)
        Value step1 = b.add(i8_val, b.i8(127));
        Value step2 = b.add(step1, b.i8(1));
        Value step3 = b.add(step2, b.i8(-128));

        Argument result = b.arg(I8);
        b.store(step3, result);
    }},
    {[](FunctionBuilder &b) {
        // Multiplication that changes sign and overflows
        Value i32_val = b.input_arg(I32);

        // Multiply by -1 when value is min I32: -2147483648 * -1 overflows in two's complement
        Value multiplied = b.mul(i32_val, b.i32(-1));

        Argument result = b.arg(I32);
        b.store(multiplied, result);
    }},
    {[](FunctionBuilder &b) {
        // Complex expression with multiple overflow opportunities
        Value i16_val = b.input_arg(I16);

        // ((val & 0xFF) * 0x0101) + 0x8000
        Value masked = b.and_(i16_val, b.i16(0xFF));
        Value multiplied = b.mul(masked, b.i16(0x0101)); // 257 in decimal
        Value added = b.add(multiplied, b.u16(0x8000u)); // 32768 in decimal

        Argument result = b.arg(I16);
        b.store(added, result);
    }},
    {[](FunctionBuilder &b) {
        // I64 arithmetic with min and max constants
        Value i64_val = b.input_arg(I64);

        // Operations with min and max I64 values
        Value add_max = b.add(i64_val, b.i64(0x7FFFFFFFFFFFFFFF));
        Value sub_min = b.sub(i64_val, b.u64(0x8000000000000000));

        // Multiply by -1 (overflow for min value)
        Value mul_neg_one = b.mul(i64_val, b.i64(-1));

        Argument result1 = b.arg(I64);
        Argument result2 = b.arg(I64);
        Argument result3 = b.arg(I64);
        b.store(add_max, result1);
        b.store(sub_min, result2);
        b.store(mul_neg_one, result3);
    }},
    {[](FunctionBuilder &b) {
        // Common overflow pattern: (x << n) + x
        Value i32_val = b.input_arg(I32);

        // Left shift by 24 then add original (can overflow when x has high bits set)
        Value shifted = b.sll(i32_val, b.i32(24));
        Value sum = b.add(shifted, i32_val);

        Argument result = b.arg(I32);
        b.store(sum, result);
    }},
    {[](FunctionBuilder &b) {
        // Type conversion overflow when truncating large values
        Value i64_val = b.input_arg(I64);

        // Truncate I64 with large value to I8 (loses many bits)
        Value truncated = b.trunc(i64_val, I8);

        // Also test sign extension back
        Value extended_again = b.sext(truncated, I64);

        Argument result1 = b.arg(I8);
        Argument result2 = b.arg(I64);
        b.store(truncated, result1);
        b.store(extended_again, result2);
    }},
    {[](FunctionBuilder &b) {
        // Operations with and_agg-1s constants (like -1 in signed)
        Value i32_val = b.input_arg(I32);

        // Use 0xFFFFFFFF which is -1 in signed, UINT_MAX in unsigned
        Value add_all_ones = b.add(i32_val, b.u32(0xFFFFFFFF));
        Value sub_all_ones = b.sub(i32_val, b.u32(0xFFFFFFFF));
        Value mul_all_ones = b.mul(i32_val, b.u32(0xFFFFFFFF));

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);
        Argument result3 = b.arg(I32);
        b.store(add_all_ones, result1);
        b.store(sub_all_ones, result2);
        b.store(mul_all_ones, result3);
    }},
    {[](FunctionBuilder &b) {
        // Combined overflow scenarios in a single expression
        Value i16_val = b.input_arg(I16);

        // Expression designed to overflow in multiple ways:
        // ((val + 32767) * 2) >> 1) - 32767
        Value add_boundary = b.add(i16_val, b.i16(32767));
        Value mul_two = b.mul(add_boundary, b.i16(2));
        Value shift_right = b.sra(mul_two, b.i16(1));
        Value final_sub = b.sub(shift_right, b.i16(32767));

        Argument result = b.arg(I16);
        b.store(final_sub, result);
    }},
    {[](FunctionBuilder &b) {
        // Test basic rotate left and right
        Value i32_val = b.input_arg(I32);
        Value shift = b.input_arg(I32);

        // Rotate left by variable amount
        Value rotl_result = b.rotl(i32_val, shift);

        // Rotate right by constant amount (including wrap-around amounts)
        Value rotr_8 = b.rotr(i32_val, b.i32(8));
        Value rotr_40 = b.rotr(i32_val, b.i32(40)); // 40 mod 32 = 8

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);
        Argument result3 = b.arg(I32);
        b.store(rotl_result, result1);
        b.store(rotr_8, result2);
        b.store(rotr_40, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test rotates combined with other operations
        Value i64_val = b.input_arg(I64);

        // Rotate left by 1, then mask
        Value rotl_1 = b.rotl(i64_val, b.i64(1));
        Value masked = b.and_(rotl_1, b.u64(0xAAAAAAAAAAAAAAAA));

        // Rotate right by 8, then XOR with original
        Value rotr_8 = b.rotr(i64_val, b.i64(8));
        Value xored = b.xor_(i64_val, rotr_8);

        Argument result1 = b.arg(I64);
        Argument result2 = b.arg(I64);
        b.store(masked, result1);
        b.store(xored, result2);
    }},
    {[](FunctionBuilder &b) {
         // Test rotate with large shift constants (wrap-around behavior)
         Value i8_val = b.input_arg(I8);

         // Rotate left by amounts >= 8
         Value rotl_8 = b.rotl(i8_val, b.i8(8));   // Should be same as rotl by 0
         Value rotl_9 = b.rotl(i8_val, b.i8(9));   // Should be same as rotl by 1
         Value rotl_15 = b.rotl(i8_val, b.i8(15)); // Should be same as rotl by 7

         // Rotate right by negative equivalent (via large positive)
         Value rotr_1 = b.rotr(i8_val, b.i8(1));
         Value rotr_9 = b.rotr(i8_val, b.i8(9)); // Should be same as rotr by 1

         Argument result1 = b.arg(I8);
         Argument result2 = b.arg(I8);
         Argument result3 = b.arg(I8);
         Argument result4 = b.arg(I8);
         Argument result5 = b.arg(I8);
         b.store(rotl_8, result1);
         b.store(rotl_9, result2);
         b.store(rotl_15, result3);
         b.store(rotr_1, result4);
         b.store(rotr_9, result5);
     },
     ONLY_SCALAR},
    {[](FunctionBuilder &b) {
        // Combine rotate operations with arithmetic
        Value i32_val = b.input_arg(I32);

        // Rotate left by 1 (multiply by 2 with carry)
        Value rotl_1 = b.rotl(i32_val, b.i32(1));

        // Add original value (rotl_1 + val)
        Value sum = b.add(rotl_1, i32_val);

        // Rotate right by 2 of the sum
        Value final = b.rotr(sum, b.i32(2));

        Argument result = b.arg(I32);
        b.store(final, result);
    }},
    {[](FunctionBuilder &b) {
        // Use rotates to simulate byte swapping
        Value i32_val = b.input_arg(I32);

        // Rotate left by 8, then mask to keep only low 24 bits, OR with rotate right by 24
        Value rotl_8 = b.rotl(i32_val, b.i32(8));
        Value rotr_24 = b.rotr(i32_val, b.i32(24));

        // Clear high byte from rotl_8, clear low 24 bits from rotr_24
        Value low_24 = b.and_(rotl_8, b.i32(0x00FFFFFF));
        Value high_8 = b.and_(rotr_24, b.u32(0xFF000000));

        // Combine
        Value swapped = b.or_(low_24, high_8);

        Argument result = b.arg(I32);
        b.store(swapped, result);
    }},
    {[](FunctionBuilder &b) {
        // Test permute_i64_i8: shuffle 8 bytes within I64
        Value i64_val = b.input_arg(I64);

        // Reverse byte order: [0,1,2,3,4,5,6,7] -> [7,6,5,4,3,2,1,0]
        Value reversed_bytes = b.permute_i64_i8(i64_val, 7, 6, 5, 4, 3, 2, 1, 0);

        // Swap high and low 32-bit words: [0,1,2,3,4,5,6,7] -> [4,5,6,7,0,1,2,3]
        Value swapped_words = b.permute_i64_i8(i64_val, 4, 5, 6, 7, 0, 1, 2, 3);

        // Duplicate low byte to and_agg positions: [0,1,2,3,4,5,6,7] -> [0,0,0,0,0,0,0,0]
        Value broadcast_low = b.permute_i64_i8(i64_val, 0, 0, 0, 0, 0, 0, 0, 0);

        Argument result1 = b.arg(I64);
        Argument result2 = b.arg(I64);
        Argument result3 = b.arg(I64);
        b.store(reversed_bytes, result1);
        b.store(swapped_words, result2);
        b.store(broadcast_low, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test permute_i64_i16: shuffle 4 16-bit words within I64
        Value i64_val = b.input_arg(I64);

        // Reverse 16-bit word order: [0,1,2,3] -> [3,2,1,0] (where each is a 16-bit word)
        Value reversed_16bit = b.permute_i64_i16(i64_val, 3, 2, 1, 0);

        // Swap pairs of words: [0,1,2,3] -> [1,0,3,2]
        Value swap_pairs = b.permute_i64_i16(i64_val, 1, 0, 3, 2);

        // Rotate words left by 1: [0,1,2,3] -> [1,2,3,0]
        Value rotate_left = b.permute_i64_i16(i64_val, 1, 2, 3, 0);

        Argument result1 = b.arg(I64);
        Argument result2 = b.arg(I64);
        Argument result3 = b.arg(I64);
        b.store(reversed_16bit, result1);
        b.store(swap_pairs, result2);
        b.store(rotate_left, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test permute_i64_i32: shuffle 2 32-bit words within I64
        Value i64_val = b.input_arg(I64);

        // Swap 32-bit halves: [0,1] -> [1,0]
        Value swapped = b.permute_i64_i32(i64_val, 1, 0);

        // Keep original order: [0,1] -> [0,1] (identity)
        Value identity = b.permute_i64_i32(i64_val, 0, 1);

        // Duplicate low 32-bit: [0,1] -> [0,0]
        Value duplicate_low = b.permute_i64_i32(i64_val, 0, 0);

        Argument result1 = b.arg(I64);
        Argument result2 = b.arg(I64);
        Argument result3 = b.arg(I64);
        b.store(swapped, result1);
        b.store(identity, result2);
        b.store(duplicate_low, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test permute_i32_i8: shuffle 4 bytes within I32
        Value i32_val = b.input_arg(I32);

        // Reverse byte order in I32: [0,1,2,3] -> [3,2,1,0]
        Value reversed = b.permute_i32_i8(i32_val, 3, 2, 1, 0);

        // Rotate bytes left by 1: [0,1,2,3] -> [1,2,3,0]
        Value rotate_left = b.permute_i32_i8(i32_val, 1, 2, 3, 0);

        // Endian swap (common use case): [0,1,2,3] -> [3,2,1,0] (same as reverse)
        // But often we might want specific pattern for endianness

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);
        b.store(reversed, result1);
        b.store(rotate_left, result2);
    }},
    {[](FunctionBuilder &b) {
        // Test permute_i32_i16: shuffle 2 16-bit words within I32
        Value i32_val = b.input_arg(I32);

        // Swap 16-bit halves: [0,1] -> [1,0]
        Value swapped = b.permute_i32_i16(i32_val, 1, 0);

        // Identity: [0,1] -> [0,1]
        Value identity = b.permute_i32_i16(i32_val, 0, 1);

        // Duplicate high word: [0,1] -> [1,1]
        Value duplicate_high = b.permute_i32_i16(i32_val, 1, 1);

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);
        Argument result3 = b.arg(I32);
        b.store(swapped, result1);
        b.store(identity, result2);
        b.store(duplicate_high, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test permute_i16_i8: shuffle 2 bytes within I16
        Value i16_val = b.input_arg(I16);

        // Swap bytes: [0,1] -> [1,0]
        Value swapped = b.permute_i16_i8(i16_val, 1, 0);

        // Identity: [0,1] -> [0,1]
        Value identity = b.permute_i16_i8(i16_val, 0, 1);

        // Duplicate low byte: [0,1] -> [0,0]
        Value duplicate_low = b.permute_i16_i8(i16_val, 0, 0);

        Argument result1 = b.arg(I16);
        Argument result2 = b.arg(I16);
        Argument result3 = b.arg(I16);
        b.store(swapped, result1);
        b.store(identity, result2);
        b.store(duplicate_low, result3);
    }},
    {[](FunctionBuilder &b) {
         // Test permute_i8_bits: shuffle bits within each byte
         Value i32_val = b.input_arg(I32);

         // Reverse bits in each byte: [7,6,5,4,3,2,1,0]
         Value reversed_bits = b.permute_i8_bits(i32_val, 7, 6, 5, 4, 3, 2, 1, 0);

         // Rotate bits left by 1 in each byte: [1,2,3,4,5,6,7,0]
         Value rotate_left_bits = b.permute_i8_bits(i32_val, 1, 2, 3, 4, 5, 6, 7, 0);

         // Swap high and low nibbles: [4,5,6,7,0,1,2,3]
         Value swap_nibbles = b.permute_i8_bits(i32_val, 4, 5, 6, 7, 0, 1, 2, 3);

         Argument result1 = b.arg(I32);
         Argument result2 = b.arg(I32);
         Argument result3 = b.arg(I32);
         b.store(reversed_bits, result1);
         b.store(rotate_left_bits, result2);
         b.store(swap_nibbles, result3);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Complete bit reverse: reverse bits within bytes, then reverse bytes
        Value i64_val = b.input_arg(I64);

        // Step 1: Reverse bits within each byte
        Value reversed_bits = b.permute_i8_bits(i64_val, 7, 6, 5, 4, 3, 2, 1, 0);

        // Step 2: Reverse byte order
        Value full_reverse = b.permute_i64_i8(reversed_bits, 7, 6, 5, 4, 3, 2, 1, 0);

        Argument result = b.arg(I64);
        b.store(full_reverse, result);
    }},
    {[](FunctionBuilder &b) {
        // Chain multiple permutations
        Value i32_val = b.input_arg(I32);

        // First swap 16-bit words
        Value swapped_16bit = b.permute_i32_i16(i32_val, 1, 0);

        // Then permute bytes within each 16-bit word separately
        // Note: After 16-bit swap, bytes are in new positions
        Value permuted_bytes = b.permute_i32_i8(swapped_16bit, 1, 0, 3, 2);

        // Then reverse bits in each byte
        Value reversed_bits = b.permute_i8_bits(permuted_bytes, 7, 6, 5, 4, 3, 2, 1, 0);

        Argument result = b.arg(I32);
        b.store(reversed_bits, result);
    }},
    {[](FunctionBuilder &b) {
        // Combine permutations with arithmetic
        Value i64_val = b.input_arg(I64);

        // Permute bytes
        Value permuted = b.permute_i64_i8(i64_val, 1, 0, 3, 2, 5, 4, 7, 6);

        // Add original value
        Value sum = b.add(i64_val, permuted);

        // Then reverse bits in each byte
        Value result_bits = b.permute_i8_bits(sum, 7, 6, 5, 4, 3, 2, 1, 0);

        Argument result = b.arg(I64);
        b.store(result_bits, result);
    }},
    {[](FunctionBuilder &b) {
         // Create bit interleaving pattern using permutations
         Value i32_val = b.input_arg(I32);

         // First, spread bits using bit permutation
         // Pattern that spreads even bits to low nibble, odd bits to high nibble
         // This is a complex 8-bit permutation that needs to be designed carefully
         // Example: take every other bit
         // For simplicity, use a known pattern
         Value interleaved_bits = b.permute_i8_bits(i32_val, 0, 4, 1, 5, 2, 6, 3, 7);

         // Then permute bytes
         Value final_result = b.permute_i32_i8(interleaved_bits, 3, 2, 1, 0);

         Argument result = b.arg(I32);
         b.store(final_result, result);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Combine permutation with bit shifts
        Value i64_val = b.input_arg(I64);

        // First permute 32-bit words
        Value swapped_32bit = b.permute_i64_i32(i64_val, 1, 0);

        // Shift left by 8 bits
        Value shifted = b.sll(swapped_32bit, b.i64(8));

        // Then permute bytes within each 32-bit word
        Value permuted_bytes = b.permute_i64_i8(shifted, 1, 2, 3, 0, 5, 6, 7, 4);

        Argument result = b.arg(I64);
        b.store(permuted_bytes, result);
    }},
    {[](FunctionBuilder &b) {
         // Simulate byte-wise operations using permutations
         Value a = b.input_arg(I32);
         Value b_arg = b.input_arg(I32);

         // Permute bytes of both inputs
         Value a_permuted = b.permute_i32_i8(a, 3, 2, 1, 0);
         Value b_permuted = b.permute_i32_i8(b_arg, 0, 1, 2, 3);

         // Add byte-wise (but actually whole 32-bit add)
         Value sum = b.add(a_permuted, b_permuted);

         // Then apply bit permutation to each byte
         Value result_bits = b.permute_i8_bits(sum, 1, 0, 3, 2, 5, 4, 7, 6);

         Argument result = b.arg(I32);
         b.store(result_bits, result);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Use permutations to pack data
        Value i64_val = b.input_arg(I64);

        // Extract low bytes from each 16-bit word and pack them
        // Assuming i64 contains 8 bytes: [b0,b1,b2,b3,b4,b5,b6,b7]
        // We want to pack and_agg even-indexed bytes: [b0,b2,b4,b6]
        // First, permute to bring them to low 32 bits
        Value packed = b.permute_i64_i8(i64_val, 0, 2, 4, 6, 0, 0, 0, 0);

        // Then truncate to I32 (or use permute_i32 to extract low 32 bits)
        Argument result = b.arg(I32);
        b.store(b.trunc(packed, I32), result);
    }},
    {[](FunctionBuilder &b) {
         // Test bit permutations with various constant patterns
         Value i16_val = b.input_arg(I16);

         // Identity permutation
         Value identity = b.permute_i8_bits(i16_val, 0, 1, 2, 3, 4, 5, 6, 7);

         // Reverse bits
         Value reverse = b.permute_i8_bits(i16_val, 7, 6, 5, 4, 3, 2, 1, 0);

         // Spread bits (every other bit in low positions)
         Value spread = b.permute_i8_bits(i16_val, 0, 2, 4, 6, 1, 3, 5, 7);

         // Gather bits (opposite of spread)
         Value gather = b.permute_i8_bits(i16_val, 0, 4, 1, 5, 2, 6, 3, 7);

         Argument result1 = b.arg(I16);
         Argument result2 = b.arg(I16);
         Argument result3 = b.arg(I16);
         Argument result4 = b.arg(I16);
         b.store(identity, result1);
         b.store(reverse, result2);
         b.store(spread, result3);
         b.store(gather, result4);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Nested permutations across different types
        Value i64_val = b.input_arg(I64);

        // First, treat as 2 I32 values and swap them
        Value swapped_32bit = b.permute_i64_i32(i64_val, 1, 0);

        // Then treat each I32 as 2 I16 values and swap within each
        // We need to do this by extracting and recombining, or use byte permutation
        // Let's use byte permutation for more control
        // Original bytes after 32-bit swap: [4,5,6,7,0,1,2,3]
        // Now swap 16-bit within each 32-bit: [6,7,4,5,2,3,0,1]
        Value swapped_16bit = b.permute_i64_i8(swapped_32bit, 2, 3, 0, 1, 6, 7, 4, 5);

        // Then reverse bits in each byte
        Value reversed_bits = b.permute_i8_bits(swapped_16bit, 7, 6, 5, 4, 3, 2, 1, 0);

        Argument result = b.arg(I64);
        b.store(reversed_bits, result);
    }},
    {[](FunctionBuilder &b) {
        // Use permutation in aggregate operations
        Value i32_val = b.input_arg(I32);

        // Permute bytes
        Value permuted = b.permute_i32_i8(i32_val, 3, 2, 1, 0);

        // Sum the permuted values
        Argument sum_result = b.arg(I32);
        b.sum(permuted, sum_result);

        // Also store a bit-permuted version
        Value bit_permuted = b.permute_i8_bits(i32_val, 7, 6, 5, 4, 3, 2, 1, 0);
        Argument store_result = b.arg(I32);
        b.store(bit_permuted, store_result);
    }},
    {[](FunctionBuilder &b) {
        // Gather with scaled index (ensure within bounds)
        Argument x = b.arg(I32);
        Value idx = b.input_arg(I32);
        Argument dst = b.arg(I32);

        // Scale index by 2, but clamp to ensure < 1000
        Value scaled_idx = b.mul(idx, b.i32(2));
        Value clamped_idx = b.and_(scaled_idx, b.i32(0x3FF)); // Mask to 10 bits (0-1023)

        Value gathered = b.gather(clamped_idx, x);
        b.store(gathered, dst);
    }},
    {[](FunctionBuilder &b) {
         // Scatter with index arithmetic
         Value x = b.input_arg(I32);
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(I32);

         // Compute index as (idx + 10) % 256
         Value offset_idx = b.add(idx, b.i32(10));
         Value safe_idx = b.and_(offset_idx, b.i32(0xFF)); // Keep in range 0-255

         b.scatter(x, safe_idx, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Use index value in arithmetic expression
        Value index_val = b.index(I32);
        Value scaled = b.mul(index_val, b.i32(10));

        Argument dst = b.arg(I32);
        b.store(scaled, dst);
    }},
    {[](FunctionBuilder &b) {
        // Use index as index for gather
        Argument array = b.arg(I32);
        Value index_idx = b.index(I32);

        // Ensure index is within bounds (mod 1000)
        Value safe_idx = b.and_(index_idx, b.i32(0x3FF)); // 0-1023

        Value gathered = b.gather(safe_idx, array);
        Argument dst = b.arg(I32);
        b.store(gathered, dst);
    }},
    {[](FunctionBuilder &b) {
         // Use index as index for scatter
         Value value = b.input_arg(I32);
         Value index_idx = b.index(I32);

         // Ensure index is within bounds
         Value safe_idx = b.and_(index_idx, b.i32(0x1FF)); // 0-511

         Argument dst = b.arg(I32);
         b.scatter(value, safe_idx, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Gather from one array, process, scatter to another
         Argument src_array = b.arg(I32);
         Argument dst_array = b.arg(I32);
         Value idx = b.input_arg(I32);

         // Gather value
         Value gathered = b.gather(idx, src_array);

         // Process: multiply by 2
         Value processed = b.mul(gathered, b.i32(2));

         // Scatter to destination
         b.scatter(processed, idx, dst_array);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Multiple gathers from same array at different indices
        Argument array = b.arg(I32);
        Value idx1 = b.input_arg(I32);
        Value idx2 = b.input_arg(I32);

        // Gather two values
        Value val1 = b.gather(idx1, array);
        Value val2 = b.gather(idx2, array);

        // Add them
        Value sum = b.add(val1, val2);

        Argument dst = b.arg(I32);
        b.store(sum, dst);
    }},
    {[](FunctionBuilder &b) {
         // Scatter a computed value
         Value idx = b.input_arg(I32);
         Argument dst = b.arg(I32);

         // Compute value based on index
         Value value = b.mul(idx, b.i32(10));

         // Scatter computed value
         b.scatter(value, idx, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Gather using permuted index
         Argument array = b.arg(I32);
         Value idx = b.input_arg(I32);

         // Permute index bits to create different access pattern
         Value permuted_idx = b.permute_i8_bits(idx, 1, 0, 3, 2, 5, 4, 7, 6);
         Value safe_idx = b.and_(permuted_idx, b.i32(0x3FF)); // Keep in range

         Value gathered = b.gather(safe_idx, array);
         Argument dst = b.arg(I32);
         b.store(gathered, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Use index with shift to create pattern
        Value index_val = b.index(I32);
        Value shifted = b.sll(index_val, b.i32(2)); // Multiply by 4

        Argument dst = b.arg(I32);
        b.store(shifted, dst);
    }},
    {[](FunctionBuilder &b) {
         // Complex pattern: gather, process, scatter to different location
         Argument src = b.arg(I32);
         Argument dst = b.arg(I32);
         Value idx = b.input_arg(I32);

         // Gather value
         Value val = b.gather(idx, src);

         // Complex processing: (val * 3 + 7) & 0xFF
         Value processed = b.mul(val, b.i32(3));
         processed = b.add(processed, b.i32(7));
         processed = b.and_(processed, b.i32(0xFF));

         // Scatter to index + 100 (with bounds check)
         Value dst_idx = b.add(idx, b.i32(100));
         Value safe_dst_idx = b.and_(dst_idx, b.i32(0x3FF)); // 0-1023

         b.scatter(processed, safe_dst_idx, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // I64 gather and scatter
         Value idx_i64 = b.input_arg(I64);
         Argument array_i64 = b.arg(I64);
         Argument dst_i64 = b.arg(I64);

         // Gather I64 value
         Value gathered_i64 = b.gather(idx_i64, array_i64);

         // Scatter to different location (index + 1)
         Value dst_idx = b.add(idx_i64, b.i64(1));
         Value safe_dst_idx = b.and_(dst_idx, b.i64(0x3FF)); // Bound check

         b.scatter(gathered_i64, safe_dst_idx, dst_i64);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Use index as both index and value
         Argument dst = b.arg(I32);
         Value index_val = b.index(I32);

         // Use index as index (with modulo)
         Value idx = b.and_(index_val, b.i32(0xFF)); // 0-255

         // Use index as value
         Value value = b.add(index_val, b.i32(100));

         // Scatter
         b.scatter(value, idx, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Gather from array with offset
        Argument base_array = b.arg(I32);
        Value idx = b.input_arg(I32);
        Value offset = b.i32(100);

        // Compute index with offset, ensure within bounds
        Value offset_idx = b.add(idx, offset);
        Value safe_idx = b.and_(offset_idx, b.i32(0x3FF));

        Value gathered = b.gather(safe_idx, base_array);
        Argument dst = b.arg(I32);
        b.store(gathered, dst);
    }},
    {[](FunctionBuilder &b) {
         // Scatter same value to multiple indices
         Value value = b.input_arg(I32);
         Value idx1 = b.input_arg(I32);
         Value idx2 = b.input_arg(I32);
         Argument dst = b.arg(I32);
         Argument dst2 = b.arg(I32);

         // Scatter to first index
         b.scatter(value, idx1, dst);

         // Scatter to second index
         b.scatter(value, idx2, dst2);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Gather I32, convert to I64, scatter as I64
         Argument src_i32 = b.arg(I32);
         Argument dst_i64 = b.arg(I64);
         Value idx = b.input_arg(I32);

         // Gather I32 value
         Value gathered_i32 = b.gather(idx, src_i32);

         // Convert to I64
         Value gathered_i64 = b.sext(gathered_i32, I64);

         // Scale I64 value
         Value scaled_i64 = b.mul(gathered_i64, b.i64(100));

         // Scatter to I64 array (need to convert index to I64)
         Value idx_i64 = b.sext(idx, I64);
         Value safe_idx = b.and_(idx_i64, b.i64(0x1FF)); // 0-511

         b.scatter(scaled_i64, safe_idx, dst_i64);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Use index to generate complex pattern
         Argument dst = b.arg(I32);
         Value index_val = b.index(I32);

         // Generate pattern: (index * 3) % 256
         Value pattern = b.mul(index_val, b.i32(3));
         pattern = b.and_(pattern, b.i32(0xFF));

         // Scatter pattern at index index
         Value idx = b.and_(index_val, b.i32(0xFF));
         b.scatter(pattern, idx, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Basic integer comparisons producing I1 results
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);

        // Signed comparisons
        Predicate eq = b.cmp_eq(a, b_arg);
        Predicate ne = b.cmp_ne(a, b_arg);
        Predicate gt = b.cmp_gt(a, b_arg);
        Predicate ge = b.cmp_ge(a, b_arg);
        Predicate lt = b.cmp_lt(a, b_arg);
        Predicate le = b.cmp_le(a, b_arg);

        // Unsigned comparisons
        Predicate ugt = b.cmp_ugt(a, b_arg);
        Predicate uge = b.cmp_uge(a, b_arg);
        Predicate ult = b.cmp_ult(a, b_arg);
        Predicate ule = b.cmp_ule(a, b_arg);

        // Store I1 results
        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);
        Argument result4 = b.arg(I1);
        Argument result5 = b.arg(I1);
        Argument result6 = b.arg(I1);
        Argument result7 = b.arg(I1);
        Argument result8 = b.arg(I1);
        Argument result9 = b.arg(I1);
        Argument result10 = b.arg(I1);

        b.store(eq, result1);
        b.store(ne, result2);
        b.store(gt, result3);
        b.store(ge, result4);
        b.store(lt, result5);
        b.store(le, result6);
        b.store(ugt, result7);
        b.store(uge, result8);
        b.store(ult, result9);
        b.store(ule, result10);
    }},
    {[](FunctionBuilder &b) {
        // Basic blend operation: select between two values based on I1 condition
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Predicate cond = b.input_predicate_arg();

        // Blend: if cond then a else b
        Value blended = b.select(cond, a, b_arg); // Note: blend(falsy, truthy, cond)

        Argument result = b.arg(I32);
        b.store(blended, result);
    }},
    {[](FunctionBuilder &b) {
        // Comparisons with constant values
        Value val = b.input_arg(I64);

        // Compare with various constants
        Predicate eq_zero = b.cmp_eq(val, b.i64(0));
        Predicate gt_max = b.cmp_gt(val, b.i64(0x7FFFFFFFFFFFFFFF));
        Predicate lt_min = b.cmp_lt(val, b.u64(0x8000000000000000));
        Predicate uge_half = b.cmp_uge(val, b.u64(0x8000000000000000)); // unsigned >= 2^63

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);
        Argument result4 = b.arg(I1);

        b.store(eq_zero, result1);
        b.store(gt_max, result2);
        b.store(lt_min, result3);
        b.store(uge_half, result4);
    }},
    {[](FunctionBuilder &b) {
        // Complex blend with computed values
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);

        // Create condition: x > y
        Predicate cond = b.cmp_gt(x, y);

        // Compute two different expressions
        Value expr1 = b.mul(x, b.i32(3));  // x * 3
        Value expr2 = b.add(y, b.i32(10)); // y + 10

        // Blend based on condition
        Value result_val = b.select(cond, expr1, expr2); // if x > y then x*3 else y+10

        Argument result = b.arg(I32);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Test if bits are set (similar to x86 TEST instruction)
        Value val = b.input_arg(I32);
        Value mask = b.input_arg(I32);

        // test: (val & mask) != 0
        Predicate test_result = b.bit_test(val, mask);

        // testn: (val & mask) == 0
        Predicate testn_result = b.bit_testn(val, mask);

        // Test with constant mask
        Predicate test_bit0 = b.bit_test(val, b.i32(1));             // Test bit 0
        Predicate testn_bit31 = b.bit_testn(val, b.u32(0x80000000)); // Test bit 31 not set

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);
        Argument result4 = b.arg(I1);

        b.store(test_result, result1);
        b.store(testn_result, result2);
        b.store(test_bit0, result3);
        b.store(testn_bit31, result4);
    }},
    {[](FunctionBuilder &b) {
        // Nested blend operations for ternary logic
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);

        // Condition 1: a > b
        Predicate cond1 = b.cmp_gt(a, b_arg);

        // Condition 2: b > c
        Predicate cond2 = b.cmp_gt(b_arg, c);

        // First blend: if a > b then a else b
        Value temp = b.select(cond1, a, b_arg);

        // Second blend: if b > c then temp else c
        Value final = b.select(cond2, temp, c);

        Argument result = b.arg(I32);
        b.store(final, result);
    }},
    {[](FunctionBuilder &b) {
        // Chain of comparisons with blend
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);
        Value z = b.input_arg(I64);

        // Compare x with y and y with z
        Predicate y_gt_z = b.cmp_gt(y, z);

        // Find max of three values using blends
        // First find max of y and z
        Value max_y_z = b.select(y_gt_z, y, z);

        // Then compare with x
        Predicate x_gt_max_yz = b.cmp_gt(x, max_y_z);

        // Final max
        Value max_all = b.select(x_gt_max_yz, x, max_y_z);

        Argument result = b.arg(I64);
        b.store(max_all, result);
    }},
    {[](FunctionBuilder &b) {
        // Combine test operations with shifts
        Value val = b.input_arg(I32);

        // Test specific bit after shifting
        Value shifted = b.sll(val, b.i32(1));                // val << 1
        Predicate test_bit1 = b.bit_test(shifted, b.i32(2)); // Test bit 1 of original (now bit 2)

        // Testn with arithmetic shift
        Value arith_shifted = b.sra(val, b.i32(31));                 // Arithmetic shift right by 31
        Predicate testn_sign = b.bit_testn(arith_shifted, b.i32(1)); // Test if low bit is 0

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);

        b.store(test_bit1, result1);
        b.store(testn_sign, result2);
    }},
    {[](FunctionBuilder &b) {
        // Implement min/max using comparisons and blend
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);

        // Min: a < b ? a : b
        Predicate a_lt_b = b.cmp_lt(a, b_arg);
        Value min_val = b.select(a_lt_b, a, b_arg);

        // Max: a > b ? a : b
        Predicate a_gt_b = b.cmp_gt(a, b_arg);
        Value max_val = b.select(a_gt_b, a, b_arg);

        Argument result1 = b.arg(I32);
        Argument result2 = b.arg(I32);

        b.store(min_val, result1);
        b.store(max_val, result2);
    }},
    {[](FunctionBuilder &b) {
        // Unsigned comparisons with boundary constants
        Value val = b.input_arg(I32);

        // Common unsigned comparison patterns
        Predicate eq_max = b.cmp_ueq(val, b.u32(0xFFFFFFFF));  // val == UINT_MAX
        Predicate lt_half = b.cmp_ult(val, b.u32(0x80000000)); // val < 2^31
        Predicate ge_half = b.cmp_uge(val, b.u32(0x80000000)); // val >= 2^31

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);

        b.store(eq_max, result1);
        b.store(lt_half, result2);
        b.store(ge_half, result3);
    }},
    {[](FunctionBuilder &b) {
        // Complex condition using arithmetic
        Value x = b.input_arg(I64);
        Value y = b.input_arg(I64);

        // Condition: (x * 2) > (y + 10)
        Value x_times_2 = b.mul(x, b.i64(2));
        Value y_plus_10 = b.add(y, b.i64(10));
        Predicate cond = b.cmp_gt(x_times_2, y_plus_10);

        // Result: if condition then x else y
        Value result_val = b.select(cond, x, y);

        Argument result = b.arg(I64);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Multiple equality checks combined
        Value val = b.input_arg(I32);
        Value a = b.i32(1);
        Value b_arg = b.i32(2);
        Value c = b.i32(3);

        // Check which constant val equals
        Predicate eq_a = b.cmp_eq(val, a);
        Predicate eq_b = b.cmp_eq(val, b_arg);
        Predicate eq_c = b.cmp_eq(val, c);

        // Use blends to select result
        // First: if eq_a then 100 else 0
        Value temp1 = b.select(eq_a, b.i32(100), b.i32(0));
        // Then: if eq_b then 200 else temp1
        Value temp2 = b.select(eq_b, b.i32(200), temp1);
        // Finally: if eq_c then 300 else temp2
        Value final = b.select(eq_c, b.i32(300), temp2);

        Argument result = b.arg(I32);
        b.store(final, result);
    }},
    {[](FunctionBuilder &b) {
        // Test operation combined with bitwise ops
        Value val = b.input_arg(I32);
        Value mask1 = b.input_arg(I32);
        Value mask2 = b.input_arg(I32);

        // Test if or_agg bits in mask1 are set
        Predicate test1 = b.bit_test(val, mask1);

        // Test if no bits in mask2 are set
        Predicate testn2 = b.bit_testn(val, mask2);

        // Combine with bitwise AND of masks
        Value combined_mask = b.and_(mask1, mask2);
        Predicate test_combined = b.bit_test(val, combined_mask);

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);

        b.store(test1, result1);
        b.store(testn2, result2);
        b.store(test_combined, result3);
    }},
    {[](FunctionBuilder &b) {
        // Blend with unary operations in branches
        Value val = b.input_arg(I32);
        Predicate cond = b.input_predicate_arg();

        // Compute two different unary operations
        Value negated = b.negate(val); // -val
        Value abs_val = b.abs(val);    // |val|

        // Blend based on condition
        Value result_val = b.select(cond, negated, abs_val); // if cond then -val else |val|

        Argument result = b.arg(I32);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Show difference between signed and unsigned comparisons
        Value x = b.input_arg(I32);
        Value y = b.u32(0xFFFFFFFF); // -1 signed, UINT_MAX unsigned

        // Signed comparison: -1 < 0 is true
        Predicate signed_lt = b.cmp_lt(x, y);

        // Unsigned comparison: or_agg number < UINT_MAX is true (except UINT_MAX itself)
        Predicate unsigned_lt = b.cmp_ult(x, y);

        // These will differ when x is negative (high bit set)

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);

        b.store(signed_lt, result1);
        b.store(unsigned_lt, result2);
    }},
    {[](FunctionBuilder &b) {
        // Test operation after rotate
        Value val = b.input_arg(I32);

        // Rotate left by 4 bits
        Value rotated = b.rotl(val, b.i32(4));

        // Test specific bit pattern after rotation
        Predicate test_pattern = b.bit_test(rotated, b.i32(0xF0));   // Test if or_agg of bits 4-7 are set
        Predicate testn_pattern = b.bit_testn(rotated, b.i32(0x0F)); // Test if and_agg of bits 0-3 are 0

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);

        b.store(test_pattern, result1);
        b.store(testn_pattern, result2);
    }},
    {[](FunctionBuilder &b) {
        // Three-way conditional using nested blends
        Predicate cond1 = b.input_predicate_arg();
        Predicate cond2 = b.input_predicate_arg();

        // Three possible values
        Value val1 = b.i64(100);
        Value val2 = b.i64(200);
        Value val3 = b.i64(300);

        // If cond1 then val1 else (if cond2 then val2 else val3)
        Value inner = b.select(cond2, val2, val3);
        Value result_val = b.select(cond1, val1, inner);

        Argument result = b.arg(I64);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Compare results of arithmetic expressions
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);

        // Compute two expressions
        Value expr1 = b.mul(a, b.i32(3));      // a * 3
        Value expr2 = b.add(b_arg, b.i32(10)); // b + 10

        // Compare them
        Predicate gt = b.cmp_gt(expr1, expr2);
        Predicate eq = b.cmp_eq(expr1, expr2);
        Predicate lt = b.cmp_lt(expr1, expr2);

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);

        b.store(gt, result1);
        b.store(eq, result2);
        b.store(lt, result3);
    }},
    {[](FunctionBuilder &b) {
        // Test operations with population count and leading zeros
        Value val = b.input_arg(I32);

        // Test if popcnt is odd (bit 0 of popcnt is 1)
        Value pop = b.popcnt(val);
        Predicate pop_odd = b.bit_test(pop, b.i32(1));

        // Test if lzcnt is zero (no leading zeros)
        Value lz = b.lzcnt(val);
        Predicate lz_zero = b.cmp_eq(lz, b.i32(0));

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);

        b.store(pop_odd, result1);
        b.store(lz_zero, result2);
    }},
    {[](FunctionBuilder &b) {
        // Blend based on sign of value
        Value val = b.input_arg(I32);

        // Check if negative (using signed comparison)
        Predicate is_negative = b.cmp_lt(val, b.i32(0));

        // If negative, use absolute value, else use value as-is
        Value result_val = b.select(is_negative, b.abs(val), val);

        Argument result = b.arg(I32);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Test equality after overflow-prone operations
        Value x = b.input_arg(I8);
        Value y = b.input_arg(I8);

        // Add with potential overflow
        Value sum = b.add(x, y);

        // Compare sum with direct addition of constants (might overflow differently)
        Predicate eq_with_const = b.cmp_eq(sum, b.i8(127)); // Compare with max I8

        Argument result = b.arg(I1);
        b.store(eq_with_const, result);
    }},
    {[](FunctionBuilder &b) {
        // Basic logical operations on I1 values
        Predicate cond1 = b.input_predicate_arg();
        Predicate cond2 = b.input_predicate_arg();

        // All logical operations
        Predicate and_result = b.and_(cond1, cond2);
        Predicate or_result = b.or_(cond1, cond2);
        Predicate andnot_result = b.andnot(cond1, cond2); // cond1 && !cond2
        Predicate xnor_result = b.xnor(cond1, cond2);     // cond1 == cond2
        Predicate xor_result = b.xor_(cond1, cond2);      // cond1 != cond2
        Predicate not_result = b.not_(cond1);             // !cond1

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);
        Argument result4 = b.arg(I1);
        Argument result5 = b.arg(I1);
        Argument result6 = b.arg(I1);

        b.store(and_result, result1);
        b.store(or_result, result2);
        b.store(andnot_result, result3);
        b.store(xnor_result, result4);
        b.store(xor_result, result5);
        b.store(not_result, result6);
    }},
    {[](FunctionBuilder &b) {
        // Aggregate boolean conditions
        Predicate cond = b.input_predicate_arg();

        Argument all_dst = b.arg(I1);
        Argument any_dst = b.arg(I1);
        Argument none_dst = b.arg(I1);

        b.and_agg(cond, all_dst);     // True if and_agg rows have cond true
        b.or_agg(cond, any_dst);      // True if or_agg row has cond true
        b.andnot_agg(cond, none_dst); // True if no rows have cond true
    }},
    {[](FunctionBuilder &b) {
        // Complex logical expression using multiple logical operations
        Predicate a = b.input_predicate_arg();
        Predicate b_arg = b.input_predicate_arg();
        Predicate c = b.input_predicate_arg();

        Predicate a_and_not_b = b.andnot(a, b_arg);
        Predicate not_a_and_c = b.andnot(c, a);
        Predicate final = b.or_(a_and_not_b, not_a_and_c);

        Argument result = b.arg(I1);
        b.store(final, result);
    }},
    {[](FunctionBuilder &b) {
        // sum_if with condition built from multiple comparisons
        Value value = b.input_arg(I32);
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);

        // Condition: (x > 0) && (y < 100)
        Predicate x_gt_0 = b.cmp_gt(x, b.i32(0));
        Predicate y_lt_100 = b.cmp_lt(y, b.i32(100));
        Predicate cond = b.and_(x_gt_0, y_lt_100);

        Argument sum_dst = b.arg(I32);
        b.sum_if(value, cond, sum_dst);
    }},
    {[](FunctionBuilder &b) {
        // countif with OR condition
        Predicate cond1 = b.input_predicate_arg();
        Predicate cond2 = b.input_predicate_arg();

        // Count rows where either condition is true
        Predicate cond = b.or_(cond1, cond2);

        Argument count_dst = b.arg(I64);
        b.countif(cond, count_dst);
    }},
    {[](FunctionBuilder &b) {
        // Majority vote of 3 boolean inputs
        Predicate a = b.input_predicate_arg();
        Predicate b_arg = b.input_predicate_arg();
        Predicate c = b.input_predicate_arg();

        // Majority: at least 2 of 3 are true
        // (a && b) || (a && c) || (b && c)
        Predicate ab = b.and_(a, b_arg);
        Predicate ac = b.and_(a, c);
        Predicate bc = b.and_(b_arg, c);
        Predicate ab_or_ac = b.or_(ab, ac);
        Predicate majority = b.or_(ab_or_ac, bc);

        Argument result = b.arg(I1);
        b.store(majority, result);
    }},
    {[](FunctionBuilder &b) {
        // Calculate parity (xor of multiple conditions)
        Predicate cond1 = b.input_predicate_arg();
        Predicate cond2 = b.input_predicate_arg();
        Predicate cond3 = b.input_predicate_arg();
        Predicate cond4 = b.input_predicate_arg();

        // Parity: cond1 ^ cond2 ^ cond3 ^ cond4
        Predicate xor12 = b.xor_(cond1, cond2);
        Predicate xor123 = b.xor_(xor12, cond3);
        Predicate parity = b.xor_(xor123, cond4);

        Argument result = b.arg(I1);
        b.store(parity, result);
    }},
    {[](FunctionBuilder &b) {
        // sum_if with condition that requires arithmetic
        Value value = b.input_arg(I32);
        Value x = b.input_arg(I32);

        // Condition: (x & 0x0F) == 0x0A  (x's low nibble equals 10)
        Value low_nibble = b.and_(x, b.i32(0x0F));
        Predicate cond = b.cmp_eq(low_nibble, b.i32(0x0A));

        Argument sum_dst = b.arg(I32);
        b.sum_if(value, cond, sum_dst);
    }},
    {[](FunctionBuilder &b) {
        // Count values in a range using countif
        Value val = b.input_arg(I32);

        // Count values where 10 <= val <= 20
        Predicate ge_lower = b.cmp_ge(val, b.i32(10));
        Predicate le_upper = b.cmp_le(val, b.i32(20));
        Predicate in_range = b.and_(ge_lower, le_upper);

        Argument count_dst = b.arg(I64);
        b.countif(in_range, count_dst);
    }},
    {[](FunctionBuilder &b) {
        // Check if and_agg rows satisfy complex condition
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);

        // Condition: (x > 0) || (y == 0)
        Predicate x_gt_0 = b.cmp_gt(x, b.i32(0));
        Predicate y_eq_0 = b.cmp_eq(y, b.i32(0));
        Predicate cond = b.or_(x_gt_0, y_eq_0);

        Argument all_dst = b.arg(I1);
        b.and_agg(cond, all_dst);
    }},
    {[](FunctionBuilder &b) {
        // or_agg operation to check existence
        Value arr_idx = b.input_arg(I32);
        Value target = b.i32(42);

        // Check if or_agg row has arr_idx == 42
        Predicate cond = b.cmp_eq(arr_idx, target);

        Argument any_dst = b.arg(I1);
        b.or_agg(cond, any_dst);
    }},
    {[](FunctionBuilder &b) {
        // none to ensure no invalid values
        Value value = b.input_arg(I32);

        // Ensure no value is negative
        Predicate is_negative = b.cmp_lt(value, b.i32(0));

        Argument none_dst = b.arg(I1);
        b.andnot_agg(is_negative, none_dst);
    }},
    {[](FunctionBuilder &b) {
        // XNOR implements equality check for booleans
        Predicate a = b.input_predicate_arg();
        Predicate b_arg = b.input_predicate_arg();

        // a == b (using xnor)
        Predicate equal = b.xnor(a, b_arg);

        // Alternative using xor and not
        Predicate xor_ab = b.xor_(a, b_arg);
        Predicate not_xor = b.not_(xor_ab); // Same as xnor

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);

        b.store(equal, result1);
        b.store(not_xor, result2);
    }},
    {[](FunctionBuilder &b) {
        // andnot useful for set difference operations
        Predicate set_a = b.input_predicate_arg(); // Element in set A
        Predicate set_b = b.input_predicate_arg(); // Element in set B

        // A \ B (in A but not in B)
        Predicate a_minus_b = b.andnot(set_a, set_b);

        // B \ A (in B but not in A)
        Predicate b_minus_a = b.andnot(set_b, set_a);

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);

        b.store(a_minus_b, result1);
        b.store(b_minus_a, result2);
    }},
    {[](FunctionBuilder &b) {
        // Sum values based on multiple exclusive conditions
        Value value = b.input_arg(I32);
        Value type = b.input_arg(I32);

        Argument sum_type1 = b.arg(I32);
        Argument sum_type2 = b.arg(I32);

        // Condition for type == 1
        Predicate is_type1 = b.cmp_eq(type, b.i32(1));
        b.sum_if(value, is_type1, sum_type1);

        // Condition for type == 2
        Predicate is_type2 = b.cmp_eq(type, b.i32(2));
        b.sum_if(value, is_type2, sum_type2);
    }},
    {[](FunctionBuilder &b) {
        // Count multiple different conditions
        Value x = b.input_arg(I32);

        Argument count_neg = b.arg(I64);
        Argument count_zero = b.arg(I64);
        Argument count_pos = b.arg(I64);

        // Count negative values
        Predicate is_neg = b.cmp_lt(x, b.i32(0));
        b.countif(is_neg, count_neg);

        // Count zero values
        Predicate is_zero = b.cmp_eq(x, b.i32(0));
        b.countif(is_zero, count_zero);

        // Count positive values
        Predicate is_pos = b.cmp_gt(x, b.i32(0));
        b.countif(is_pos, count_pos);
    }},
    {[](FunctionBuilder &b) {
        // Test De Morgan's laws with logical operations
        Predicate a = b.input_predicate_arg();
        Predicate b_arg = b.input_predicate_arg();

        // De Morgan: !(a && b) == !a || !b
        Predicate not_a_and_b = b.not_(b.and_(a, b_arg));
        Predicate not_a_or_not_b = b.or_(b.not_(a), b.not_(b_arg));

        // De Morgan: !(a || b) == !a && !b
        Predicate not_a_or_b = b.not_(b.or_(a, b_arg));
        Predicate not_a_and_not_b = b.and_(b.not_(a), b.not_(b_arg));

        Argument result1 = b.arg(I1);
        Argument result2 = b.arg(I1);
        Argument result3 = b.arg(I1);
        Argument result4 = b.arg(I1);

        b.store(not_a_and_b, result1);
        b.store(not_a_or_not_b, result2);
        b.store(not_a_or_b, result3);
        b.store(not_a_and_not_b, result4);
    }},
    {[](FunctionBuilder &b) {
        // cond_store with negated condition
        Value value = b.input_arg(I64);
        Predicate cond = b.input_predicate_arg();
        Argument dst = b.arg(I64);

        // Store value when condition is false
        Predicate not_cond = b.not_(cond);
        b.cond_store(value, not_cond, dst);
    }},
    {[](FunctionBuilder &b) {
        // sum_if based on bit test
        Value value = b.input_arg(I32);
        Value flags = b.input_arg(I32);

        // Sum values where bit 3 is set
        Predicate bit3_set = b.bit_test(flags, b.i32(0x08));

        Argument sum_dst = b.arg(I32);
        b.sum_if(value, bit3_set, sum_dst);
    }},
    {[](FunctionBuilder &b) {
        // Complex boolean function: 3-input XOR with enable
        Predicate a = b.input_predicate_arg();
        Predicate b_arg = b.input_predicate_arg();
        Predicate c = b.input_predicate_arg();
        Predicate enable = b.input_predicate_arg();

        // If enabled, compute a ^ b ^ c, else false
        Predicate xor_ab = b.xor_(a, b_arg);
        Predicate xor_abc = b.xor_(xor_ab, c);
        Predicate result = b.and_(xor_abc, enable);

        Argument result_dst = b.arg(I1);
        b.store(result, result_dst);
    }},
    {[](FunctionBuilder &b) {
        // Check implication: if x > 0 then y > 0
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);

        // Implication: !(x > 0) || (y > 0)
        Predicate x_gt_0 = b.cmp_gt(x, b.i32(0));
        Predicate y_gt_0 = b.cmp_gt(y, b.i32(0));
        Predicate not_x_gt_0 = b.not_(x_gt_0);
        Predicate implication = b.or_(not_x_gt_0, y_gt_0);

        Argument all_dst = b.arg(I1);
        b.and_agg(implication, all_dst);
    }},
    {[](FunctionBuilder &b) {
        // Count values above threshold, but only if another condition holds
        Value value = b.input_arg(I32);
        Predicate valid = b.input_predicate_arg();
        Value threshold = b.i32(100);

        // Count valid values above threshold
        Predicate above_threshold = b.cmp_gt(value, threshold);
        Predicate cond = b.and_(above_threshold, valid);

        Argument count_dst = b.arg(I64);
        b.countif(cond, count_dst);
    }},
    {[](FunctionBuilder &b) {
        // Combine multiple comparisons with logical operations
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);

        // (x < y) && (x != 0) && (y != 0)
        Predicate x_lt_y = b.cmp_lt(x, y);
        Predicate x_ne_0 = b.cmp_ne(x, b.i32(0));
        Predicate y_ne_0 = b.cmp_ne(y, b.i32(0));

        Predicate temp = b.and_(x_lt_y, x_ne_0);
        Predicate cond = b.and_(temp, y_ne_0);

        Argument result = b.arg(I1);
        b.store(cond, result);
    }},
    {[](FunctionBuilder &b) {
        // Check if or_agg row satisfies one of multiple conditions
        Value val = b.input_arg(I32);

        // Check if val is 1, 3, or 7
        Predicate eq1 = b.cmp_eq(val, b.i32(1));
        Predicate eq3 = b.cmp_eq(val, b.i32(3));
        Predicate eq7 = b.cmp_eq(val, b.i32(7));

        Predicate cond1_or_cond2 = b.or_(eq1, eq3);
        Predicate cond = b.or_(cond1_or_cond2, eq7);

        Argument any_dst = b.arg(I1);
        b.or_agg(cond, any_dst);
    }},
    {[](FunctionBuilder &b) {
        // Store only if two conditions are met
        Value value = b.input_arg(I64);
        Predicate cond1 = b.input_predicate_arg();
        Predicate cond2 = b.input_predicate_arg();
        Argument dst = b.arg(I64);

        // Store only if both conditions true
        Predicate both_true = b.and_(cond1, cond2);
        b.cond_store(value, both_true, dst);
    }},
    {[](FunctionBuilder &b) {
        // Deep expression combining arithmetic, logical operations, and type conversions
        Value val = b.input_arg(I32);

        // Complex arithmetic expression
        Value expr = b.add(val, b.i32(100));
        expr = b.mul(expr, b.i32(3));
        expr = b.sub(expr, b.i32(50));
        expr = b.sll(expr, b.i32(2));
        expr = b.and_(expr, b.i32(0xFFF));

        // Logical condition based on arithmetic
        Predicate cond1 = b.cmp_gt(val, b.i32(0));
        Predicate cond2 = b.cmp_lt(val, b.i32(1000));
        Predicate cond = b.and_(cond1, cond2);

        // Conditional update
        Value updated = b.select(cond, b.i32(0), expr);

        // Convert to I64 for further processing
        Value i64_val = b.sext(updated, I64);
        i64_val = b.mul(i64_val, b.i64(10));

        // Bit manipulation
        i64_val = b.rotl(i64_val, b.i64(5));
        i64_val = b.and_(i64_val, b.i64(0xFFFFFFFF));

        // Final store
        Argument result = b.arg(I64);
        b.store(i64_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Deep tree with conditional aggregates and logical operations
        Value value = b.input_arg(I32);
        Value type = b.input_arg(I8);
        Value flags = b.input_arg(I32);

        // Multiple conditions
        Predicate type_cond1 = b.cmp_eq(b.sext(type, I32), b.i32(1));
        Predicate type_cond2 = b.cmp_eq(b.sext(type, I32), b.i32(2));
        Predicate flag_cond = b.bit_test(flags, b.i32(0x01));

        // Complex condition tree
        Predicate cond_a = b.and_(type_cond1, flag_cond);
        Predicate cond_b = b.and_(type_cond2, b.not_(flag_cond));
        Predicate cond = b.or_(cond_a, cond_b);

        // Process value based on condition tree
        Value processed = b.select(cond, b.mul(value, b.i32(10)), // if condition true
                                   b.add(value, b.i32(5))         // if condition false
        );

        // Further arithmetic
        processed = b.sra(processed, b.i32(1));
        processed = b.abs(processed);

        // Conditional sum and count
        Argument sum_dst = b.arg(I32);
        Argument count_dst = b.arg(I64);
        Argument any_dst = b.arg(I1);

        b.sum_if(processed, cond, sum_dst);
        b.countif(cond, count_dst);
        b.or_agg(cond, any_dst);
    }},
    {[](FunctionBuilder &b) {
        // Deep nested blend operations with arithmetic tree
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);
        Value d = b.input_arg(I32);

        // Complex arithmetic expressions for each branch
        Value expr1 = b.add(b.mul(a, b.i32(3)), b.i32(10));
        Value expr2 = b.sub(b.mul(b_arg, b.i32(2)), b.i32(5));
        Value expr3 = b.sll(c, b.i32(4));
        Value expr4 = b.sra(d, b.i32(2));

        // Conditions built from comparisons
        Predicate cond1 = b.cmp_gt(a, b_arg);
        Predicate cond2 = b.cmp_lt(c, d);
        Predicate cond3 = b.cmp_eq(b.add(a, b_arg), b.add(c, d));

        // Nested blend tree
        Value temp1 = b.select(cond1, expr1, expr2); // if a > b then expr1 else expr2
        Value temp2 = b.select(cond2, expr3, expr4); // if c < d then expr3 else expr4
        Value final = b.select(cond3, temp1, temp2); // if (a+b)==(c+d) then temp1 else temp2

        // Post-processing
        final = b.rotl(final, b.i32(8));
        final = b.and_(final, b.i32(0xFFFF));

        Argument result = b.arg(I32);
        b.store(final, result);
    }},
    {[](FunctionBuilder &b) {
         // Deep tree of bit manipulation operations
         Value val = b.input_arg(I64);

         // Bit permutation within each byte
         Value bits_permuted = b.permute_i8_bits(val, 1, 0, 3, 2, 5, 4, 7, 6);

         // Byte permutation
         Value bytes_permuted = b.permute_i64_i8(bits_permuted, 7, 6, 5, 4, 3, 2, 1, 0);

         // Bitwise operations
         Value bitwise = b.xor_(bytes_permuted, b.u64(0xAAAAAAAAAAAAAAAA));
         bitwise = b.and_(bitwise, b.i64(0x5555555555555555));
         bitwise = b.or_(bitwise, b.i64(0x1111111111111111));

         // Rotate and shift combination
         Value rotated = b.rotl(bitwise, b.i64(13));
         Value shifted = b.srl(rotated, b.i64(7));

         // Funnel shift with self
         Value high_part = b.sra(val, b.i64(32));
         Value funneled = b.add(high_part, shifted);

         // Logical condition based on bit test
         Predicate test_cond = b.bit_test(funneled, b.u64(0x8000000000000000));
         Predicate testn_cond = b.bit_testn(funneled, b.i64(0x00000000000000FF));
         Predicate final_cond = b.xor_(test_cond, testn_cond);

         // Conditional processing
         Value processed = b.select(final_cond, b.sub(funneled, b.i64(100)), b.add(funneled, b.i64(100)));

         Argument result = b.arg(I64);
         b.store(processed, result);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Deep tree with multiple aggregates and nested conditions
         Value value = b.input_arg(I64);
         Value category = b.input_arg(I16);
         Value status = b.input_arg(I8);

         // Convert types
         Value cat_i32 = b.sext(category, I32);
         Value status_i32 = b.sext(status, I32);

         // Multiple complex conditions
         Predicate cond_cat1 = b.cmp_eq(cat_i32, b.i32(1));
         Predicate cond_cat2 = b.cmp_eq(cat_i32, b.i32(2));
         Predicate cond_status = b.cmp_ugt(status_i32, b.i32(0)); // unsigned > 0

         // Nested condition tree
         Predicate cond_a = b.and_(cond_cat1, cond_status);
         Predicate cond_b = b.and_(cond_cat2, b.not_(cond_status));
         Predicate cond_c = b.or_(cond_a, cond_b);

         // Arithmetic processing for each branch
         Value processed_val = b.select(cond_c, b.mul(value, b.i64(2)), // if cond_c true
                                        b.add(value, b.i64(100))        // if cond_c false
         );

         // Further processing with min/max bounds
         processed_val = b.umax(processed_val, b.i64(0));
         processed_val = b.umin(processed_val, b.i64(10000));

         // Multiple aggregates
         Argument sum_dst = b.arg(I128);
         Argument count_dst = b.arg(I64);
         Argument min_dst = b.arg(I64);
         Argument max_dst = b.arg(I64);
         Argument any_dst = b.arg(I1);
         Argument all_dst = b.arg(I1);

         b.sum(processed_val, sum_dst);
         b.countif(cond_c, count_dst);
         b.min_agg(processed_val, min_dst);
         b.max_agg(processed_val, max_dst);
         b.or_agg(cond_c, any_dst);
         b.and_agg(cond_status, all_dst);
     },
     test_meta()
         .limitation(TestVariant::VectorAll)
         .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::TooManyRoots)},
    {[](FunctionBuilder &b) {
         // Deep tree with multiple type conversions and arithmetic
         Value i8_val = b.input_arg(I8);
         Value i16_val = b.input_arg(I16);
         Value i32_val = b.input_arg(I32);
         Value i64_val = b.input_arg(I64);

         // Convert everything to I64 with appropriate extensions
         Value i64_from_i8 = b.sext(i8_val, I64);
         Value i64_from_i16 = b.sext(i16_val, I64);
         Value i64_from_i32 = b.sext(i32_val, I64);

         // Complex arithmetic tree
         Value expr1 = b.add(i64_from_i8, i64_from_i16);
         Value expr2 = b.sub(i64_val, i64_from_i32);
         Value expr3 = b.mul(expr1, b.i64(3));     // Valid: constant multiplication
         Value expr4 = b.mul(expr2, i64_from_i32); // Valid: both extended from I32

         // Blend conditions based on comparisons
         Predicate cond1 = b.cmp_gt(i64_from_i8, i64_from_i16);
         Predicate cond2 = b.cmp_ult(i64_from_i32, i64_val); // unsigned comparison

         // Nested blend
         Value temp = b.select(cond1, expr3, expr4);
         Value final = b.select(cond2, expr2, temp);

         // Bit manipulation
         final = b.rotl(final, b.i64(19));
         final = b.and_(final, b.i64(0x0000FFFFFFFFFFFF));

         // Convert back to I32 for store
         Value final_i32 = b.trunc(final, I32);

         // Additional processing in I32
         final_i32 = b.add(final_i32, b.i32(1000));
         final_i32 = b.sll(final_i32, b.i32(1));

         Argument result = b.arg(I32);
         b.store(final_i32, result);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Complex expression involving gather/scatter with arithmetic
         Value idx = b.input_arg(I32);
         Value value = b.input_arg(I32);
         Argument src_array = b.arg(I32);
         Argument dst_array = b.arg(I32);

         // Ensure index is safe
         Value safe_idx = b.and_(idx, b.i32(0x3FF)); // 0-1023

         // Gather value from source array
         Value gathered = b.gather(safe_idx, src_array);

         // Complex arithmetic processing
         Value processed = b.add(gathered, value);
         processed = b.mul(processed, b.i32(3));
         processed = b.sra(processed, b.i32(2));

         // Condition based on bit test
         Predicate cond = b.bit_test(processed, b.i32(0x01)); // Test LSB

         // Conditional further processing
         processed = b.select(cond, b.add(processed, b.i32(100)), // if LSB set
                              b.sub(processed, b.i32(50))         // if LSB not set
         );

         // Scatter processed value
         Value dst_idx = b.add(safe_idx, b.i32(100));
         Value safe_dst_idx = b.and_(dst_idx, b.i32(0x3FF));
         b.scatter(processed, safe_dst_idx, dst_array);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
        // Deep tree of logical operations with arithmetic leaves
        Value a = b.input_arg(I32);
        Value b_arg = b.input_arg(I32);
        Value c = b.input_arg(I32);
        Value d = b.input_arg(I32);
        Value e = b.input_arg(I32);

        // Complex comparison tree
        Predicate cmp1 = b.cmp_gt(a, b_arg);
        Predicate cmp2 = b.cmp_lt(c, d);
        Predicate cmp3 = b.cmp_eq(e, b.i32(0));
        Predicate cmp4 = b.cmp_uge(a, c);     // unsigned >=
        Predicate cmp5 = b.cmp_ule(b_arg, d); // unsigned <=

        // Deep logical tree
        Predicate and1 = b.and_(cmp1, cmp2);
        Predicate or1 = b.or_(cmp3, cmp4);
        Predicate xor1 = b.xor_(and1, or1);
        Predicate andnot1 = b.andnot(cmp5, xor1);
        Predicate xnor1 = b.xnor(andnot1, cmp1);

        // Nested logical operations
        Predicate final_cond = b.or_(b.and_(xnor1, b.not_(cmp2)), b.andnot(cmp3, cmp4));

        // Arithmetic expressions based on conditions
        Value expr1 = b.add(b.mul(a, b.i32(3)), b.i32(10));
        Value expr2 = b.sub(b.mul(b_arg, b.i32(2)), b.i32(5));
        Value expr3 = b.sll(c, b.i32(4));
        Value expr4 = b.sra(d, b.i32(2));

        // Multi-level blend tree
        Value temp1 = b.select(cmp1, expr1, expr2);
        Value temp2 = b.select(cmp2, expr3, expr4);
        Value temp3 = b.select(final_cond, temp1, temp2);

        // Final processing
        Value result_val = b.add(temp3, e);

        Argument result = b.arg(I32);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
        // Complex expression using index and permutation
        Value index_val = b.index(I32);
        Value base_val = b.input_arg(I32);

        // Process index value
        Value processed = b.mul(index_val, b.i32(10));
        processed = b.add(processed, base_val);
        processed = b.and_(processed, b.i32(0xFF));

        // Permute bytes
        Value permuted = b.permute_i32_i8(processed, 3, 2, 1, 0);

        // Bit permutation within bytes
        Value bit_permuted = b.permute_i8_bits(permuted, 7, 6, 5, 4, 3, 2, 1, 0);

        // Complex condition based on index
        Predicate cond = b.cmp_ugt(b.and_(index_val, b.i32(0x03)), // index % 4
                                   b.i32(1));

        // Conditional arithmetic
        Value result_val = b.select(cond, b.add(bit_permuted, b.i32(100)), b.sub(bit_permuted, b.i32(50)));

        // Rotate based on index
        Value rotate_amount = b.and_(index_val, b.i32(0x07)); // 0-7
        result_val = b.rotl(result_val, rotate_amount);

        Argument result = b.arg(I32);
        b.store(result_val, result);
    }},
    {[](FunctionBuilder &b) {
         // Ultimate deep expression tree using and_agg operation categories
         Value a = b.input_arg(I64);
         Value b_arg = b.input_arg(I32);
         Value c = b.input_arg(I16);
         Value d = b.input_arg(I8);

         // Type conversion chain
         Value a_i64 = a;
         Value b_i64 = b.sext(b_arg, I64);
         Value c_i64 = b.sext(c, I64);
         Value d_i64 = b.sext(d, I64);

         // Complex arithmetic tree
         Value expr1 = b.add(a_i64, b_i64);
         Value expr2 = b.sub(c_i64, d_i64);
         Value expr3 = b.mul(expr1, b.i64(3)); // Valid: constant
         Value expr4 = b.mul(expr2, b_i64);    // Valid: extended from I32

         // Bit manipulation tree
         expr3 = b.rotl(expr3, b.i64(13));
         expr4 = b.sll(expr4, b.i64(2));

         // Permutation tree
         Value permuted = b.permute_i64_i8(expr3, 7, 6, 5, 4, 3, 2, 1, 0);
         Value bits_permuted = b.permute_i8_bits(expr4, 1, 0, 3, 2, 5, 4, 7, 6);

         Value funneled = b.add(permuted, bits_permuted);

         // Logical condition tree
         Predicate cond1 = b.cmp_gt(a_i64, b_i64);
         Predicate cond2 = b.cmp_ult(c_i64, d_i64); // unsigned
         Predicate cond3 = b.bit_test(funneled, b.u64(0x8000000000000000));
         Predicate and_cond = b.and_(cond1, cond2);
         Predicate or_cond = b.or_(and_cond, cond3);
         Predicate final_cond = b.xor_(or_cond, b.not_(cond1));

         // Multi-level blend with arithmetic
         Value branch1 = b.add(funneled, b.i64(1000));
         Value branch2 = b.sub(funneled, b.i64(500));
         Value branch3 = b.mul(funneled, b.i64(2)); // Valid: constant

         // Nested blend tree
         Value temp1 = b.select(cond1, branch1, branch2);
         Value temp2 = b.select(cond2, temp1, branch3);
         Value final_val = b.select(final_cond, funneled, temp2);

         // Final bit operations
         final_val = b.and_(final_val, b.i64(0x7FFFFFFFFFFFFFFF));
         final_val = b.or_(final_val, b.i64(0x00000000000000FF));

         // Conditional aggregates
         Predicate cond_agg = b.cmp_ugt(final_val, b.i64(1000));
         Argument sum_dst = b.arg(I64);
         Argument count_dst = b.arg(I64);
         Argument min_dst = b.arg(I64);
         Argument any_dst = b.arg(I1);

         b.sum_if(final_val, cond_agg, sum_dst);
         b.countif(cond_agg, count_dst);
         b.min_agg(final_val, min_dst);
         b.or_agg(cond_agg, any_dst);

         // Final store
         Argument store_dst = b.arg(I64);
         b.store(final_val, store_dst);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // TPC-H Q1 style: revenue with discounts and shipping
         Value l_quantity = b.input_arg(I32);
         Value l_extendedprice = b.input_arg(I64);
         Value l_discount = b.input_arg(I32);
         Value l_shipdate = b.input_arg(I32); // epoch days
         Value l_returnflag = b.input_arg(I8);
         Value l_linestatus = b.input_arg(I8);

         // Convert discount to decimal (stored as percentage * 100, e.g., 1500 for 15.00%)
         Value discount_decimal = l_discount;
         Value one_minus_discount = b.sub(b.i64(1), b.sext(discount_decimal, I64));

         // Filter: shipdate <= '1998-09-02'
         Predicate shipdate_filter = b.cmp_le(b.sext(l_shipdate, I64), b.i64(904)); // 1998-09-02 epoch

         // Revenue calculations
         Value revenue = b.mul(l_extendedprice, one_minus_discount);

         // Aggregates by return flag and line status
         Predicate returnflag_A = b.cmp_eq(l_returnflag, b.i8('A'));
         Predicate linestatus_F = b.cmp_eq(l_linestatus, b.i8('F'));
         Predicate group_cond = b.and_(returnflag_A, linestatus_F);
         Predicate filter_cond = b.and_(group_cond, shipdate_filter);

         Argument sum_qty = b.arg(I64);
         Argument sum_base_price = b.arg(I64);
         Argument sum_disc_price = b.arg(I64);
         Argument count_order = b.arg(I64);

         b.sum_if(b.sext(l_quantity, I64), filter_cond, sum_qty);
         b.sum_if(l_extendedprice, filter_cond, sum_base_price);
         b.sum_if(revenue, filter_cond, sum_disc_price);
         b.countif(filter_cond, count_order);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .limitation(TestVariant::X86Vector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)
         .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    {[](FunctionBuilder &b) {
         // TPC-DS Q14 style: cross sell analysis with promotions
         Value cs_quantity = b.input_arg(I32);
         Value cs_list_price = b.input_arg(I64);
         Value cs_sold_date_sk = b.input_arg(I32); // date key
         Value cs_item_sk = b.input_arg(I32);
         Value cs_promo_sk = b.input_arg(I32);
         Value d_date_sk = b.input_arg(I32);
         Value d_year = b.input_arg(I16);
         Value d_moy = b.input_arg(I8);
         Value i_item_sk = b.input_arg(I32);
         Value i_category = b.input_arg(I8);
         Value p_promo_sk = b.input_arg(I32);
         Value p_channel = b.input_arg(I8);

         // Join conditions
         Predicate date_join = b.cmp_eq(cs_sold_date_sk, d_date_sk);
         Predicate item_join = b.cmp_eq(cs_item_sk, i_item_sk);
         Predicate promo_join = b.cmp_eq(cs_promo_sk, p_promo_sk);

         // Filter conditions
         Predicate year_filter = b.cmp_eq(d_year, b.i16(2001));
         Predicate month_filter = b.cmp_eq(d_moy, b.i8(11));
         Predicate category_filter = b.cmp_eq(i_category, b.i8(2)); // Electronics
         Predicate channel_filter = b.cmp_eq(p_channel, b.i8('N')); // Newspaper

         // Combine conditions
         Predicate cond = date_join;
         cond = b.and_(cond, item_join);
         cond = b.and_(cond, promo_join);
         cond = b.and_(cond, year_filter);
         cond = b.and_(cond, month_filter);
         cond = b.and_(cond, category_filter);
         cond = b.and_(cond, channel_filter);

         // Calculate metrics
         Value sales_amount = b.mul(b.sext(cs_quantity, I64), cs_list_price);
         Value avg_quantity = b.sext(cs_quantity, I64); // Normalize

         // Multiple aggregates
         Argument total_sales = b.arg(I64);
         Argument avg_sales = b.arg(I64);
         Argument max_sales = b.arg(I64);
         Argument min_sales = b.arg(I64);
         Argument count_transactions = b.arg(I64);
         Argument count_items = b.arg(I64);

         b.sum_if(sales_amount, cond, total_sales);
         b.sum_if(avg_quantity, cond, avg_sales);
         b.max_agg_if(sales_amount, cond, max_sales);
         b.min_agg_if(sales_amount, cond, min_sales);
         b.countif(cond, count_transactions);
         b.countif(item_join, count_items); // Count distinct items
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .limitation(TestVariant::X86Vector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)
         .vectorization_failure(TestVariant::X86Vector, simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},
    {[](FunctionBuilder &b) {
         // ClickBench style: weather station analysis
         Value temperature = b.input_arg(I32); // in tenths of degree (e.g., 215 = 21.5°C)
         Value pressure = b.input_arg(I32);    // in hPa
         Value humidity = b.input_arg(I32);    // percentage
         Value timestamp = b.input_arg(I64);   // epoch seconds
         Value station_id = b.input_arg(I32);
         Value quality_flag = b.input_arg(I8);

         // Extract date parts from timestamp
         Value seconds_per_day = b.i64(86400);
         Value days_since_epoch = b.div(timestamp, seconds_per_day);
         Value day_of_week = b.and_(days_since_epoch, b.i64(7)); // 0-6 for Sunday-Saturday

         // Filter conditions
         Predicate temp_range = b.and_(b.cmp_gt(temperature, b.i32(-500)), // > -50.0°C
                                       b.cmp_lt(temperature, b.i32(5000))  // < 500.0°C
         );
         Predicate pressure_range = b.and_(b.cmp_gt(pressure, b.i32(80000)), // > 800 hPa
                                           b.cmp_lt(pressure, b.i32(110000)) // < 1100 hPa
         );
         Predicate quality_check = b.cmp_eq(quality_flag, b.i8(1)); // Good quality
         Predicate weekend = b.or_(b.cmp_eq(day_of_week, b.i64(0)), // Sunday
                                   b.cmp_eq(day_of_week, b.i64(6))  // Saturday
         );

         // Station-specific analysis
         Predicate station_nyc = b.cmp_eq(station_id, b.i32(12345));
         Predicate cond_nyc = b.and_(station_nyc, temp_range);
         cond_nyc = b.and_(cond_nyc, pressure_range);
         cond_nyc = b.and_(cond_nyc, quality_check);

         // Weekend vs weekday analysis
         Predicate cond_weekend = b.and_(cond_nyc, weekend);
         Predicate cond_weekday = b.and_(cond_nyc, b.not_(weekend));

         // Aggregates
         Argument avg_temp = b.arg(I32);
         Argument max_temp = b.arg(I32);
         Argument min_temp = b.arg(I32);
         Argument total_readings = b.arg(I64);
         Argument weekend_avg_temp = b.arg(I32);
         Argument weekday_avg_temp = b.arg(I32);
         Argument humidity_corr = b.arg(I64); // For correlation calculation

         // Temperature statistics
         b.sum_if(temperature, cond_nyc, avg_temp);
         b.max_agg_if(temperature, cond_nyc, max_temp);
         b.min_agg_if(temperature, cond_nyc, min_temp);
         b.countif(cond_nyc, total_readings);

         // Weekend vs weekday comparison
         b.sum_if(temperature, cond_weekend, weekend_avg_temp);
         b.sum_if(temperature, cond_weekday, weekday_avg_temp);

         // Correlation-like metric: sum(temp * humidity)
         Value temp_humidity = b.mul(b.sext(temperature, I64), b.sext(humidity, I64));
         b.sum_if(temp_humidity, cond_nyc, humidity_corr);
     },
     coefficient_range_limit(TestVariant::VectorAll)},
    {[](FunctionBuilder &b) {
         // E-commerce funnel: user behavior analysis
         Value event_type = b.input_arg(I8);  // 1=view, 2=add_to_cart, 3=checkout, 4=purchase
         Value event_time = b.input_arg(I64); // epoch milliseconds
         Value price = b.input_arg(I64);      // product price in cents
         Value category_id = b.input_arg(I16);
         Value device_type = b.input_arg(I8); // 1=mobile, 2=desktop, 3=tablet

         // Session time window (within 30 minutes)
         Value session_start = b.input_arg(I64);
         Predicate in_session =
             b.cmp_le(b.sub(event_time, session_start), b.i64(30ll * 60 * 1000) // 30 minutes in milliseconds
             );

         // Funnel steps
         Predicate viewed = b.cmp_eq(event_type, b.i8(1));
         Predicate added_to_cart = b.cmp_eq(event_type, b.i8(2));
         Predicate completed_purchase = b.cmp_eq(event_type, b.i8(4));

         // Device filters
         Predicate mobile_user = b.cmp_eq(device_type, b.i8(1));
         Predicate desktop_user = b.cmp_eq(device_type, b.i8(2));

         // Category filters
         Predicate electronics = b.cmp_eq(category_id, b.i16(1));
         Predicate clothing = b.cmp_eq(category_id, b.i16(2));

         // Complex conditions
         Predicate mobile_electronics_view = b.and_(mobile_user, electronics);
         mobile_electronics_view = b.and_(mobile_electronics_view, viewed);
         mobile_electronics_view = b.and_(mobile_electronics_view, in_session);

         Predicate desktop_clothing_purchase = b.and_(desktop_user, clothing);
         desktop_clothing_purchase = b.and_(desktop_clothing_purchase, completed_purchase);
         desktop_clothing_purchase = b.and_(desktop_clothing_purchase, in_session);

         // Funnel metrics
         Argument unique_users = b.arg(I64);
         Argument total_sessions = b.arg(I64);
         Argument mobile_views = b.arg(I64);
         Argument mobile_adds = b.arg(I64);
         Argument desktop_revenue = b.arg(I64);

         // User counting (approximate - would need distinct in real SQL)
         b.countif(viewed, unique_users);
         b.countif(in_session, total_sessions);

         // Funnel steps for mobile
         b.countif(mobile_electronics_view, mobile_views);
         Predicate mobile_add_cond = b.and_(mobile_user, added_to_cart);
         mobile_add_cond = b.and_(mobile_add_cond, electronics);
         b.countif(mobile_add_cond, mobile_adds);

         // Revenue for desktop purchases
         b.sum_if(price, desktop_clothing_purchase, desktop_revenue);
     },
     coefficient_range_limit(TestVariant::VectorAll)},
    {[](FunctionBuilder &b) {
         // Stock market time series analysis
         Value timestamp = b.input_arg(I64); // epoch seconds
         Value symbol = b.input_arg(I32);    // stock symbol encoded
         Value price = b.input_arg(I64);     // price in cents
         Value volume = b.input_arg(I64);    // trading volume

         // Time-based aggregations
         Predicate market_hours = b.and_(b.cmp_ge(timestamp, b.i64(9ll * 3600)), // 9 AM
                                         b.cmp_le(timestamp, b.i64(16ll * 3600)) // 4 PM
         );

         // Stock-specific filters
         Predicate tech_stock = b.cmp_eq(symbol, b.i32(1001));    // AAPL
         Predicate finance_stock = b.cmp_eq(symbol, b.i32(1002)); // JPM

         // High-volume periods (volume > 1M shares)
         Predicate high_volume = b.cmp_gt(volume, b.i64(1000000));

         // Conditions
         Predicate tech_market_hours = b.and_(tech_stock, market_hours);
         Predicate finance_high_volume = b.and_(finance_stock, high_volume);

         // Aggregates
         Argument avg_tech_price = b.arg(I64);
         Argument max_tech_price = b.arg(I64);
         Argument min_tech_price = b.arg(I64);
         Argument total_tech_volume = b.arg(I64);
         Argument finance_vwap = b.arg(I64); // Volume Weighted Average Price
         Argument num_trades = b.arg(I64);
         Argument large_trades = b.arg(I64);

         // Tech stock statistics during market hours
         b.sum_if(price, tech_market_hours, avg_tech_price);
         b.max_agg_if(price, tech_market_hours, max_tech_price);
         b.min_agg_if(price, tech_market_hours, min_tech_price);
         b.sum_if(volume, tech_market_hours, total_tech_volume);

         // Finance stock VWAP: sum(price * volume) / sum(volume)
         Value price_volume = b.mul(price, volume);
         b.sum_if(price_volume, finance_high_volume, finance_vwap);

         // Counts
         b.countif(tech_market_hours, num_trades);
         Predicate large_trade_cond = b.and_(tech_stock, b.cmp_gt(volume, b.i64(10000)));
         b.countif(large_trade_cond, large_trades);
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         // Ad tech: impression, click, conversion analysis
         Value timestamp = b.input_arg(I64);   // epoch milliseconds
         Value event_type = b.input_arg(I8);   // 1=impression, 2=click, 3=conversion
         Value revenue = b.input_arg(I64);     // in micro-dollars
         Value user_segment = b.input_arg(I8); // 1=new, 2=returning, 3=high_value
         Value device_id = b.input_arg(I8);    // 1=mobile, 2=desktop, 3=connected_tv

         // Time windows
         Predicate last_24h = b.cmp_ge(timestamp, b.i64(24ll * 3600 * 1000));

         // Event filters
         Predicate impression = b.cmp_eq(event_type, b.i8(1));
         Predicate click = b.cmp_eq(event_type, b.i8(2));
         Predicate conversion = b.cmp_eq(event_type, b.i8(3));

         // Segment filters
         Predicate high_value_user = b.cmp_eq(user_segment, b.i8(3));
         Predicate mobile_user = b.cmp_eq(device_id, b.i8(1));

         // CTR (Click Through Rate) calculation
         Predicate impression_click = b.and_(impression, click);
         Predicate ctr_numerator = b.and_(impression_click, last_24h);
         Predicate ctr_denominator = b.and_(impression, last_24h);

         // Revenue by user segment
         Predicate high_value_conversion = b.and_(conversion, high_value_user);
         Predicate mobile_conversion = b.and_(conversion, mobile_user);

         // Aggregates
         Argument total_impressions = b.arg(I64);
         Argument total_clicks = b.arg(I64);
         Argument total_conversions = b.arg(I64);
         Argument total_revenue = b.arg(I64);
         Argument high_value_revenue = b.arg(I64);
         Argument mobile_conversions = b.arg(I64);

         // Basic counts
         b.countif(impression, total_impressions);
         b.countif(click, total_clicks);
         b.countif(conversion, total_conversions);

         // Revenue metrics
         b.sum_if(revenue, conversion, total_revenue);
         b.sum_if(revenue, high_value_conversion, high_value_revenue);

         // Segment-specific
         b.countif(mobile_conversion, mobile_conversions);

         // Rate calculations (would need division in real implementation)
         // For simplicity, we'll count numerator and denominator separately
         Argument ctr_num = b.arg(I64);
         Argument ctr_den = b.arg(I64);
         b.countif(ctr_numerator, ctr_num);
         b.countif(ctr_denominator, ctr_den);
     },
     coefficient_range_limit(TestVariant::VectorAll)},
    {[](FunctionBuilder &b) {
         // Retail inventory and sales analysis
         Value store_id = b.input_arg(I32);
         Value date_id = b.input_arg(I32); // date key
         Value revenue = b.input_arg(I64);
         Value cost = b.input_arg(I64);
         Value inventory_level = b.input_arg(I32);
         Value category_id = b.input_arg(I16);
         Value promotion_flag = b.input_arg(I8);

         // Time periods
         Predicate current_week = b.cmp_eq(date_id, b.i32(202315)); // YYYYWW format
         Predicate previous_week = b.cmp_eq(date_id, b.i32(202314));

         // Business rules
         Predicate low_inventory = b.cmp_lt(inventory_level, b.i32(10));

         // Category filters
         Predicate fresh_category = b.cmp_eq(category_id, b.i16(1)); // Perishables

         // Promotion impact
         Predicate with_promotion = b.cmp_eq(promotion_flag, b.i8(1));
         Predicate without_promotion = b.cmp_eq(promotion_flag, b.i8(0));

         // Store performance
         Predicate store_123 = b.cmp_eq(store_id, b.i32(123));
         Predicate current_week_store = b.and_(store_123, current_week);

         // Complex conditions
         Predicate high_risk_items = b.and_(fresh_category, low_inventory);
         Predicate promoted_performance = b.and_(with_promotion, current_week);

         // Aggregates
         Argument total_revenue_current = b.arg(I64);
         Argument total_revenue_previous = b.arg(I64);
         Argument total_profit = b.arg(I64);
         Argument avg_inventory = b.arg(I32);
         Argument out_of_stock_count = b.arg(I64);

         // Revenue comparison
         b.sum_if(revenue, current_week_store, total_revenue_current);
         Predicate previous_week_store = b.and_(store_123, previous_week);
         b.sum_if(revenue, previous_week_store, total_revenue_previous);

         // Profitability
         Value profit = b.sub(revenue, cost);
         b.sum_if(profit, current_week, total_profit);

         // Inventory metrics
         b.sum_if(inventory_level, store_123, avg_inventory);
         b.countif(high_risk_items, out_of_stock_count);

         // Promotion analysis
         Argument promoted_revenue = b.arg(I64);
         Argument non_promoted_revenue = b.arg(I64);
         b.sum_if(revenue, promoted_performance, promoted_revenue);
         Predicate non_promoted_performance = b.and_(without_promotion, current_week);
         b.sum_if(revenue, non_promoted_performance, non_promoted_revenue);
     },
     coefficient_range_limit(TestVariant::VectorAll)},
    {[](FunctionBuilder &b) {
         // Web server log analysis
         Value timestamp = b.input_arg(I64); // epoch milliseconds
         Value status_code = b.input_arg(I16);
         Value response_size = b.input_arg(I32); // bytes
         Value request_time = b.input_arg(I32);  // milliseconds
         Value endpoint = b.input_arg(I32);      // encoded URL
         Value user_agent = b.input_arg(I8);     // 1=mobile, 2=desktop, 3=bot
         Value referrer = b.input_arg(I8);       // 1=direct, 2=search, 3=social

         // Time analysis
         Value hour_of_day = timestamp;
         Predicate business_hours = b.and_(b.cmp_ge(hour_of_day, b.i64(9)), b.cmp_le(hour_of_day, b.i64(17)));

         // Status categories
         Predicate success = b.cmp_eq(status_code, b.i16(200));
         Predicate server_error = b.cmp_ge(status_code, b.i16(500));

         // Performance thresholds
         Predicate slow_request = b.cmp_gt(request_time, b.i32(1000));       // > 1 second
         Predicate large_response = b.cmp_gt(response_size, b.i32(1000000)); // > 1MB

         // Traffic sources
         Predicate search_traffic = b.cmp_eq(referrer, b.i8(2));
         Predicate mobile_traffic = b.cmp_eq(user_agent, b.i8(1));

         // Endpoint analysis
         Predicate api_endpoint = b.cmp_eq(endpoint, b.i32(1001));
         Predicate static_endpoint = b.cmp_eq(endpoint, b.i32(1002));

         // Complex conditions
         Predicate mobile_api_slow = b.and_(mobile_traffic, api_endpoint);
         mobile_api_slow = b.and_(mobile_api_slow, slow_request);

         Predicate search_static_large = b.and_(search_traffic, static_endpoint);
         search_static_large = b.and_(search_static_large, large_response);

         // Aggregates
         Argument total_requests = b.arg(I64);
         Argument total_bandwidth = b.arg(I64);
         Argument avg_response_time = b.arg(I32);
         Argument error_rate = b.arg(I64);
         Argument mobile_errors = b.arg(I64);
         Argument peak_hour_traffic = b.arg(I64);

         // Basic metrics
         b.countif(success, total_requests); // Using success as proxy for valid requests
         b.sum_if(b.sext(response_size, I64), success, total_bandwidth);
         b.sum_if(request_time, success, avg_response_time);

         // Error analysis
         b.countif(server_error, error_rate);
         Predicate mobile_server_error = b.and_(mobile_traffic, server_error);
         b.countif(mobile_server_error, mobile_errors);

         // Traffic patterns
         b.countif(business_hours, peak_hour_traffic);
         b.countif(mobile_api_slow, b.arg(I64));
         b.countif(search_static_large, b.arg(I64));
     },
     test_meta()
         .limitation(TestVariant::ArmVector)
         .vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::CoefficientRangeNeedsNormalization)},

    // Missing-domain group 1: floating-point helpers
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        b.output_arg(b.round_nearest_even(x));
        b.output_arg(b.round_down(x));
        b.output_arg(b.round_up(x));
        b.output_arg(b.round_toward_zero(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F32);
        Value shifted = b.add(b.abs(x), b.f32(1.0f));
        Value root = b.sqrt(shifted);
        Value inv = b.rcp(root);
        Value inv_sqrt = b.rsqrt(shifted);
        Predicate ok = b.isfinite(inv);
        b.output_arg(b.select(ok, inv, b.f32(0.0f)));
        b.output_arg(b.select(ok, inv_sqrt, b.f32(0.0f)));
        b.output_arg(ok);
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(F64);
        Value y = b.input_arg(F64);
        Value int_bits = b.bitcast(x, I64);
        Value restored = b.bitcast(int_bits, F64);
        Value signed_i32 = b.signed_cast(b.input_arg(I32), F64);
        Value unsigned_i32 = b.unsigned_cast(b.input_arg(I32), F64);
        Value signed_mag = b.copysign_no_zero(y, b.abs(restored));
        Value blended = b.select(b.isnormal(signed_mag), signed_mag, b.add(signed_i32, unsigned_i32));
        b.output_arg(blended);
        b.output_arg(b.isinf(restored));
        b.output_arg(b.isnan(restored));
    }},

    // Missing-domain group 2: grouped aggregates
    {[](FunctionBuilder &b) {
        b.scalar_only();
        Argument sum_table = b.arg(I64);
        Argument min_table = b.arg(I64);
        Argument max_table = b.arg(I64);
        Value idx = b.and_(b.input_arg(I32), b.i32(15));
        Value value = b.input_arg(I64);
        b.grouped_sum(value, idx, sum_table);
        b.grouped_min(value, idx, min_table);
        b.grouped_max(value, idx, max_table);
    }},
    {[](FunctionBuilder &b) {
        b.scalar_only();
        Argument sum_table = b.arg(I64);
        Argument max_table = b.arg(I64);
        Argument and_table = b.arg(I64);
        Value idx = b.and_(b.input_arg(I32), b.i32(7));
        Value value = b.input_arg(I64);
        Value flags = b.input_arg(I64);
        Predicate enabled = b.cmp_eq(b.and_(flags, b.i64(1)), b.i64(0));
        b.grouped_sum_if(value, enabled, idx, sum_table);
        b.grouped_max_if(b.add(value, b.i64(10)), enabled, idx, max_table);
        b.grouped_and_if(b.or_(value, b.i64(0xFF)), enabled, idx, and_table);
    }},
    {[](FunctionBuilder &b) {
        b.scalar_only();
        Argument product_table = b.arg(I64);
        Argument xor_table = b.arg(I32);
        Argument umax_table = b.arg(I32);
        Value idx = b.and_(b.input_arg(I32), b.i32(31));
        Value left = b.input_arg(I32);
        Value right = b.input_arg(I32);
        Predicate left_gt_right = b.cmp_gt(left, right);
        Value mixed_i32 = b.xor_(b.add(left, right), b.rotl(left, b.i32(3)));
        Value mixed_i64 = b.sext(b.add(mixed_i32, b.i32(1)), I64);
        b.grouped_product_if(mixed_i64, left_gt_right, idx, product_table);
        b.grouped_xor(mixed_i32, idx, xor_table);
        b.grouped_umax(b.add(left, b.and_(right, b.i32(255))), idx, umax_table);
    }},

    // Missing-domain group 3: pack and conditional scatter
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        Predicate cond = b.input_predicate_arg();
        Argument packed_values = b.arg(I32);
        Argument packed_values_size = b.arg(I64);
        Argument packed_indices = b.arg(I32);
        Argument packed_indices_size = b.arg(I64);
        b.pack(x, cond, packed_values, packed_values_size);
        b.pack(b.index(I32), cond, packed_indices, packed_indices_size);
    }},
    {[](FunctionBuilder &b) {
         Value idx = b.and_(b.input_arg(I32), b.i32(15));
         Value x = b.input_arg(I32);
         Predicate cond = b.input_predicate_arg();
         Argument dst_true = b.arg(I32);
         Argument dst_false = b.arg(I32);
         b.cond_scatter(x, idx, cond, dst_true);
         b.cond_scatter(b.add(x, b.i32(1)), idx, b.not_(cond), dst_false);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},
    {[](FunctionBuilder &b) {
         Argument src = b.arg(I32);
         Argument dst = b.arg(I32);
         Argument packed = b.arg(I32);
         Argument packed_size = b.arg(I64);
         Value idx = b.and_(b.input_arg(I32), b.i32(15));
         Value bias = b.input_arg(I32);
         Predicate cond = b.input_predicate_arg();
         Value gathered = b.gather(idx, src);
         Value processed = b.select(cond, b.add(gathered, bias), b.sub(gathered, b.i32(1)));
         b.pack(processed, cond, packed, packed_size);
         b.cond_scatter(processed, idx, cond, dst);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},

    // Missing-domain group 4: safety-check helpers
    {[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I32);
        Value y = b.input_arg(I32);
        b.output_arg(b.add_checked(x, y));
        b.output_arg(b.sub_checked(x, y));
        b.output_arg(b.negate_checked(x));
    }},
    {[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value x = b.input_arg(I64);
        Value narrowed = b.trunc_checked(b.add(x, b.i64(123)), I16);
        Value widened = b.sext(narrowed, I64);
        b.output_arg(narrowed);
        b.output_arg(b.abs_checked(widened));
    }},
    {[](FunctionBuilder &b) {
        b.arg_safety_check();
        Value a = b.input_arg(I64);
        Value b_arg = b.input_arg(I64);
        Value c = b.input_arg(I64);
        Value sum = b.add_checked(a, b_arg);
        Value diff = b.sub_checked(sum, c);
        Predicate positive = b.cmp_gt(diff, b.i64(0));
        b.cond_store(diff, positive, b.arg(I64));
        b.output_arg(positive);
    }},

    // Missing-domain group 5: bit utilities and sign helpers
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I32);
        b.output_arg(b.log2(x));
        b.output_arg(b.bit_floor(x));
        b.output_arg(b.bit_ceil(x));
    }},
    {[](FunctionBuilder &b) {
        Value x = b.input_arg(I64);
        Value reversed = b.reverse_bits_full(x);
        Value swapped = b.byteswap(x);
        Predicate single = b.has_single_bit(b.bit_floor(x));
        b.output_arg(b.add(reversed, swapped));
        b.output_arg(single);
    }},
    {[](FunctionBuilder &b) {
         Value x = b.input_arg(I16);
         Value y = b.input_arg(I16);
         Value low_bits = b.trunc(b.and_(x, b.i16(0xFF)), I8);
         Value replicated = b.zext(b.replicate_ith_bit_i8(low_bits, 3), I16);
         Value signed_value = b.sign(x);
         Value copied = b.copysign_no_zero(signed_value, y);
         b.output_arg(replicated);
         b.output_arg(b.sign_no_zero(x));
         b.output_arg(copied);
     },
     LIMIT_ARM_VECTOR.vectorization_failure(TestVariant::ArmVector, simjit::ErrorSubKind::UnsupportedSpecialOps)},

    // Missing-domain group 6: lazy fox
    {[](FunctionBuilder &b) {
        b.scalar_only();
        Argument src_i32 = b.arg(I32);
        Argument scatter_dst = b.arg(I32);
        Argument packed_dst = b.arg(I32);
        Argument packed_size = b.arg(I64);
        Argument grouped_sum_dst = b.arg(I64);

        Value idx = b.and_(b.input_arg(I32), b.i32(15));
        Value base_i32 = b.input_arg(I32);
        Value base_f64 = b.input_arg(F64);
        Predicate enabled = b.input_predicate_arg();

        Value gathered = b.gather(idx, src_i32);
        Value index_val = b.index(I32);
        Value rotated = b.rotl(base_i32, b.i32(3));
        Value bit_floor_val = b.bit_floor(b.add(base_i32, b.i32(1)));
        Value fp_scaled = b.round_nearest_even(b.mul(b.abs(base_f64), b.f64(8.0)));
        Value fp_as_i32 = b.signed_cast(fp_scaled, I32);
        Value mixed = b.add(b.xor_(gathered, rotated), b.add(bit_floor_val, fp_as_i32));
        Value permuted = b.permute_i32_i8(mixed, 3, 2, 1, 0);
        Value bit_scrambled = b.permute_i8_bits(permuted, 1, 0, 3, 2, 5, 4, 7, 6);
        Predicate finite = b.isfinite(base_f64);
        Predicate positive = b.cmp_gt(base_i32, b.i32(0));
        Predicate cond = b.and_(enabled, b.and_(finite, positive));
        Value selected = b.select(cond, bit_scrambled, b.sub(mixed, index_val));

        b.output_arg(cond);
        b.output_arg(selected);
        b.cond_store(selected, cond, b.arg(I32));
        b.pack(selected, cond, packed_dst, packed_size);
        b.cond_scatter(selected, idx, cond, scatter_dst);
        b.grouped_sum(b.sext(selected, I64), idx, grouped_sum_dst);
        b.sum_if(b.sext(selected, I64), cond, b.arg(I64));
        b.countif(cond, b.arg(I64));
        b.cond_store(b.bitcast(b.bitcast(base_f64, I64), F64), cond, b.arg(F64));
    }}
    //
};
