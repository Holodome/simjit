// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include <array>

#include "simjit/simjit.h"
#include "test.h"

using namespace simjit;
using namespace simjit::types;

namespace {

#define SIMJIT_VALUE_BINARY_OPS(X) \
    X(add)                         \
    X(sub)                         \
    X(mul)                         \
    X(div)                         \
    X(udiv)                        \
    X(mod)                         \
    X(umod)                        \
    X(min)                         \
    X(max)                         \
    X(umin)                        \
    X(umax)                        \
    X(and_)                        \
    X(or_)                         \
    X(xor_)                        \
    X(andnot)                      \
    X(sll)                         \
    X(srl)                         \
    X(sra)                         \
    X(rotl)                        \
    X(rotr)

#define SIMJIT_UNSIGNED_CMP_OPS(X) \
    X(cmp_ueq)                     \
    X(cmp_une)                     \
    X(cmp_ugt)                     \
    X(cmp_uge)                     \
    X(cmp_ult)                     \
    X(cmp_ule)

#define SIMJIT_SIGNED_CMP_OPS(X) \
    X(cmp_eq)                    \
    X(cmp_ne)                    \
    X(cmp_gt)                    \
    X(cmp_ge)                    \
    X(cmp_lt)                    \
    X(cmp_le)

#define SIMJIT_ARITH_AGG_OPS(X) \
    X(sum, sum_if)              \
    X(product, product_if)      \
    X(min_agg, min_agg_if)      \
    X(max_agg, max_agg_if)      \
    X(umin_agg, umin_agg_if)    \
    X(umax_agg, umax_agg_if)    \
    X(and_agg, and_agg_if)      \
    X(or_agg, or_agg_if)        \
    X(xor_agg, xor_agg_if)      \
    X(andnot_agg, andnot_agg_if)

#define SIMJIT_PREDICATE_AGG_OPS(X) \
    X(and_agg)                      \
    X(or_agg)                       \
    X(xor_agg)                      \
    X(andnot_agg)

#define SIMJIT_GROUPED_AGG_OPS(X)          \
    X(grouped_sum, grouped_sum_if)         \
    X(grouped_product, grouped_product_if) \
    X(grouped_min, grouped_min_if)         \
    X(grouped_max, grouped_max_if)         \
    X(grouped_umin, grouped_umin_if)       \
    X(grouped_umax, grouped_umax_if)       \
    X(grouped_and, grouped_and_if)         \
    X(grouped_or, grouped_or_if)           \
    X(grouped_xor, grouped_xor_if)         \
    X(grouped_andnot, grouped_andnot_if)

#define SIMJIT_GENERIC_UNARY_OPS(X) \
    X(not_)                         \
    X(negate)                       \
    X(abs)

#define SIMJIT_INT_UNARY_OPS(X) \
    X(lzcnt)                    \
    X(popcnt)

#define SIMJIT_FLOAT_UNARY_OPS(X) \
    X(round_nearest_even)         \
    X(round_down)                 \
    X(round_up)                   \
    X(round_toward_zero)          \
    X(sqrt)                       \
    X(rsqrt)                      \
    X(rcp)

static constexpr std::array<ScalarDataType, 8> kAllDtypes{
    I8, I16, I32, I64, F32, F64, I1, I128,
};

static bool is_simple_int(ScalarDataType dtype) {
    return dtype == I8 || dtype == I16 || dtype == I32 || dtype == I64;
}

static bool is_float(ScalarDataType dtype) {
    return dtype == F32 || dtype == F64;
}

static bool is_valid_float_cast(ScalarDataType from, ScalarDataType to) {
    return (is_float(from) && is_float(to)) || (is_simple_int(from) && is_float(to)) ||
           (is_simple_int(to) && is_float(from));
}

static bool is_valid_bitcast(ScalarDataType from, ScalarDataType to) {
    return (from == F32 && to == I32) || (from == F64 && to == I64) || (from == I32 && to == F32) ||
           (from == I64 && to == F64);
}

static bool is_valid_int_cast(ScalarDataType from, ScalarDataType to, IntCastKind kind) {
    if (to == I1 || from == to) { return false; }
    if (!is_simple_int(from) || !is_simple_int(to)) { return false; }

    size_t from_size = 0;
    size_t to_size = 0;
    switch (from) {
    case I8: from_size = 1; break;
    case I16: from_size = 2; break;
    case I32: from_size = 4; break;
    case I64: from_size = 8; break;
    default: return false;
    }
    switch (to) {
    case I8: to_size = 1; break;
    case I16: to_size = 2; break;
    case I32: to_size = 4; break;
    case I64: to_size = 8; break;
    default: return false;
    }

    bool extend = from_size < to_size;
    switch (kind) {
    case IntCastKind::Trunc: return !extend;
    case IntCastKind::Sext: return extend;
    case IntCastKind::Zext: return extend;
    }
    return false;
}

static void add_load_predicate_tests(std::vector<Test> &tests) {
    for (ScalarDataType arg_dtype : kAllDtypes) {
        if (arg_dtype == I1) { continue; }
        add_invalid(tests, [arg_dtype](FunctionBuilder &b) {
            Predicate pred = b.load_predicate(b.arg(arg_dtype));
            b.output_arg(pred);
        });
    }
}

static void add_store_tests(std::vector<Test> &tests) {
    for (ScalarDataType arg_dtype : kAllDtypes) {
        for (ScalarDataType dst_dtype : kAllDtypes) {
            if (dst_dtype == arg_dtype) { continue; }
            add_invalid(tests, [arg_dtype, dst_dtype](FunctionBuilder &b) {
                b.store(b.input_arg(arg_dtype), b.arg(dst_dtype));
            });
            add_invalid(tests, [arg_dtype, dst_dtype](FunctionBuilder &b) {
                b.cond_store(b.input_arg(arg_dtype), b.input_predicate_arg(), b.arg(dst_dtype));
            });
        }
    }

    for (ScalarDataType dst_dtype : kAllDtypes) {
        if (dst_dtype == I1) { continue; }
        add_invalid(tests, [dst_dtype](FunctionBuilder &b) { b.store(b.input_predicate_arg(), b.arg(dst_dtype)); });
    }
}

static void add_pack_tests(std::vector<Test> &tests) {
    for (ScalarDataType arg_dtype : kAllDtypes) {
        for (ScalarDataType dst_dtype : kAllDtypes) {
            if (dst_dtype == arg_dtype) { continue; }
            add_invalid(tests, [arg_dtype, dst_dtype](FunctionBuilder &b) {
                b.pack(b.input_arg(arg_dtype), b.input_predicate_arg(), b.arg(dst_dtype), b.arg(I64));
            });
        }
        for (ScalarDataType size_dtype : kAllDtypes) {
            if (size_dtype == I64) { continue; }
            add_invalid(tests, [arg_dtype, size_dtype](FunctionBuilder &b) {
                b.pack(b.input_arg(arg_dtype), b.input_predicate_arg(), b.arg(arg_dtype), b.arg(size_dtype));
            });
        }
    }
}

static void add_gather_scatter_tests(std::vector<Test> &tests) {
    for (ScalarDataType idx_dtype : kAllDtypes) {
        bool valid_idx = is_simple_int(idx_dtype);
        for (ScalarDataType data_dtype : kAllDtypes) {
            if (!valid_idx) {
                add_invalid(tests, [idx_dtype, data_dtype](FunctionBuilder &b) {
                    b.output_arg(b.gather(b.input_arg(idx_dtype), b.arg(data_dtype)));
                });
                add_invalid(tests, [idx_dtype, data_dtype](FunctionBuilder &b) {
                    b.scatter(b.input_arg(data_dtype), b.input_arg(idx_dtype), b.arg(data_dtype));
                });
                add_invalid(tests, [idx_dtype, data_dtype](FunctionBuilder &b) {
                    b.cond_scatter(b.input_arg(data_dtype), b.input_arg(idx_dtype), b.input_predicate_arg(),
                                   b.arg(data_dtype));
                });
            }
        }
    }

    for (ScalarDataType arg_dtype : kAllDtypes) {
        for (ScalarDataType dst_dtype : kAllDtypes) {
            if (dst_dtype == arg_dtype) { continue; }
            add_invalid(tests, [arg_dtype, dst_dtype](FunctionBuilder &b) {
                b.scatter(b.input_arg(arg_dtype), b.input_arg(I32), b.arg(dst_dtype));
            });
            add_invalid(tests, [arg_dtype, dst_dtype](FunctionBuilder &b) {
                b.cond_scatter(b.input_arg(arg_dtype), b.input_arg(I32), b.input_predicate_arg(), b.arg(dst_dtype));
            });
        }
    }
}

static void add_aggregate_tests(std::vector<Test> &tests) {
    for (ScalarDataType arg_dtype : kAllDtypes) {
        for (ScalarDataType dst_dtype : kAllDtypes) {
            if (dst_dtype == arg_dtype || dst_dtype == I128) { continue; }
#define SIMJIT_ADD_INVALID_AGG_TEST(OP, OP_IF)                                                                   \
    add_invalid(tests,                                                                                           \
                [arg_dtype, dst_dtype](FunctionBuilder &b) { b.OP(b.input_arg(arg_dtype), b.arg(dst_dtype)); }); \
    add_invalid(tests, [arg_dtype, dst_dtype](FunctionBuilder &b) {                                              \
        b.OP_IF(b.input_arg(arg_dtype), b.input_predicate_arg(), b.arg(dst_dtype));                              \
    });
            SIMJIT_ARITH_AGG_OPS(SIMJIT_ADD_INVALID_AGG_TEST)
#undef SIMJIT_ADD_INVALID_AGG_TEST
        }
    }

    for (ScalarDataType dst_dtype : kAllDtypes) {
        if (dst_dtype == I1) { continue; }
#define SIMJIT_ADD_INVALID_PREDICATE_AGG_TEST(OP) \
    add_invalid(tests, [dst_dtype](FunctionBuilder &b) { b.OP(b.input_predicate_arg(), b.arg(dst_dtype)); });
        SIMJIT_PREDICATE_AGG_OPS(SIMJIT_ADD_INVALID_PREDICATE_AGG_TEST)
#undef SIMJIT_ADD_INVALID_PREDICATE_AGG_TEST
        if (dst_dtype != I64 && dst_dtype != I128) {
            add_invalid(tests,
                        [dst_dtype](FunctionBuilder &b) { b.countif(b.input_predicate_arg(), b.arg(dst_dtype)); });
        }
    }
}

static void add_grouped_aggregate_tests(std::vector<Test> &tests) {
    for (ScalarDataType arg_dtype : kAllDtypes) {
        for (ScalarDataType table_dtype : kAllDtypes) {
            if (table_dtype == arg_dtype) { continue; }
#define SIMJIT_ADD_INVALID_GROUPED_TABLE_TEST(OP, OP_IF)                                                \
    add_invalid(tests, [arg_dtype, table_dtype](FunctionBuilder &b) {                                   \
        b.OP(b.input_arg(arg_dtype), b.input_arg(I32), b.arg(table_dtype));                             \
    });                                                                                                 \
    add_invalid(tests, [arg_dtype, table_dtype](FunctionBuilder &b) {                                   \
        b.OP_IF(b.input_arg(arg_dtype), b.input_predicate_arg(), b.input_arg(I32), b.arg(table_dtype)); \
    });
            SIMJIT_GROUPED_AGG_OPS(SIMJIT_ADD_INVALID_GROUPED_TABLE_TEST)
#undef SIMJIT_ADD_INVALID_GROUPED_TABLE_TEST
        }
        for (ScalarDataType idx_dtype : kAllDtypes) {
            if (is_simple_int(idx_dtype)) { continue; }
#define SIMJIT_ADD_INVALID_GROUPED_INDEX_TEST(OP, OP_IF)                                                    \
    add_invalid(tests, [arg_dtype, idx_dtype](FunctionBuilder &b) {                                         \
        b.OP(b.input_arg(arg_dtype), b.input_arg(idx_dtype), b.arg(arg_dtype));                             \
    });                                                                                                     \
    add_invalid(tests, [arg_dtype, idx_dtype](FunctionBuilder &b) {                                         \
        b.OP_IF(b.input_arg(arg_dtype), b.input_predicate_arg(), b.input_arg(idx_dtype), b.arg(arg_dtype)); \
    });
            SIMJIT_GROUPED_AGG_OPS(SIMJIT_ADD_INVALID_GROUPED_INDEX_TEST)
#undef SIMJIT_ADD_INVALID_GROUPED_INDEX_TEST
        }
    }
}

static void add_binary_and_select_tests(std::vector<Test> &tests) {
    for (ScalarDataType left_dtype : kAllDtypes) {
        for (ScalarDataType right_dtype : kAllDtypes) {
            if (right_dtype == left_dtype) { continue; }
#define SIMJIT_ADD_INVALID_BINARY_TEST(OP)                                     \
    add_invalid(tests, [left_dtype, right_dtype](FunctionBuilder &b) {         \
        b.output_arg(b.OP(b.input_arg(left_dtype), b.input_arg(right_dtype))); \
    });
            SIMJIT_VALUE_BINARY_OPS(SIMJIT_ADD_INVALID_BINARY_TEST)
#undef SIMJIT_ADD_INVALID_BINARY_TEST
#define SIMJIT_ADD_INVALID_SIGNED_CMP_TEST(OP)                                 \
    add_invalid(tests, [left_dtype, right_dtype](FunctionBuilder &b) {         \
        b.output_arg(b.OP(b.input_arg(left_dtype), b.input_arg(right_dtype))); \
    });
            SIMJIT_SIGNED_CMP_OPS(SIMJIT_ADD_INVALID_SIGNED_CMP_TEST)
#undef SIMJIT_ADD_INVALID_SIGNED_CMP_TEST
            add_invalid(tests, [left_dtype, right_dtype](FunctionBuilder &b) {
                b.store(b.select(b.input_predicate_arg(), b.input_arg(left_dtype), b.input_arg(right_dtype)),
                        b.arg(left_dtype));
            });
        }
    }

    for (ScalarDataType dtype : {F32, F64}) {
#define SIMJIT_ADD_INVALID_UNSIGNED_CMP_TEST(OP) \
    add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.OP(b.input_arg(dtype), b.input_arg(dtype))); });
        SIMJIT_UNSIGNED_CMP_OPS(SIMJIT_ADD_INVALID_UNSIGNED_CMP_TEST)
#undef SIMJIT_ADD_INVALID_UNSIGNED_CMP_TEST
    }
}

static void add_cast_tests(std::vector<Test> &tests) {
    for (ScalarDataType from_dtype : kAllDtypes) {
        for (ScalarDataType to_dtype : kAllDtypes) {
            if (!is_valid_float_cast(from_dtype, to_dtype)) {
                add_invalid(tests, [from_dtype, to_dtype](FunctionBuilder &b) {
                    b.output_arg(b.float_cast(b.input_arg(from_dtype), to_dtype));
                });
            }
            if (!is_valid_bitcast(from_dtype, to_dtype)) {
                add_invalid(tests, [from_dtype, to_dtype](FunctionBuilder &b) {
                    b.output_arg(b.bitcast(b.input_arg(from_dtype), to_dtype));
                });
            }
        }
    }

    constexpr std::array<IntCastKind, 3> kIntCastKinds{
        IntCastKind::Trunc,
        IntCastKind::Sext,
        IntCastKind::Zext,
    };

    for (ScalarDataType from_dtype : kAllDtypes) {
        for (ScalarDataType to_dtype : kAllDtypes) {
            for (IntCastKind kind : kIntCastKinds) {
                if (is_valid_int_cast(from_dtype, to_dtype, kind)) { continue; }
                add_invalid(tests, [from_dtype, to_dtype, kind](FunctionBuilder &b) {
                    Value arg = b.input_arg(from_dtype);
                    switch (kind) {
                    case IntCastKind::Trunc: b.output_arg(b.trunc(arg, to_dtype)); break;
                    case IntCastKind::Sext: b.output_arg(b.sext(arg, to_dtype)); break;
                    case IntCastKind::Zext: b.output_arg(b.zext(arg, to_dtype)); break;
                    }
                });
            }
        }
    }
}

static void add_unary_tests(std::vector<Test> &tests) {
    for (ScalarDataType dtype : kAllDtypes) {
        if (dtype != I1 && dtype != I128) { continue; }
#define SIMJIT_ADD_INVALID_GENERIC_UNARY_TEST(OP) \
    add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.OP(b.input_arg(dtype))); });
        SIMJIT_GENERIC_UNARY_OPS(SIMJIT_ADD_INVALID_GENERIC_UNARY_TEST)
#undef SIMJIT_ADD_INVALID_GENERIC_UNARY_TEST
    }

    for (ScalarDataType dtype : kAllDtypes) {
        if (is_simple_int(dtype)) { continue; }
#define SIMJIT_ADD_INVALID_INT_UNARY_TEST(OP) \
    add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.OP(b.input_arg(dtype))); });
        SIMJIT_INT_UNARY_OPS(SIMJIT_ADD_INVALID_INT_UNARY_TEST)
#undef SIMJIT_ADD_INVALID_INT_UNARY_TEST
    }

    for (ScalarDataType dtype : kAllDtypes) {
        if (is_float(dtype)) { continue; }
#define SIMJIT_ADD_INVALID_FLOAT_UNARY_TEST(OP) \
    add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.OP(b.input_arg(dtype))); });
        SIMJIT_FLOAT_UNARY_OPS(SIMJIT_ADD_INVALID_FLOAT_UNARY_TEST)
#undef SIMJIT_ADD_INVALID_FLOAT_UNARY_TEST
    }

    for (ScalarDataType dtype : kAllDtypes) {
        if (!is_float(dtype)) { continue; }
        add_unsupported(tests, [dtype](FunctionBuilder &b) {
            b.arg_safety_check();
            b.output_arg(b.negate_checked(b.input_arg(dtype)));
        });
        add_unsupported(tests, [dtype](FunctionBuilder &b) {
            b.arg_safety_check();
            b.output_arg(b.abs_checked(b.input_arg(dtype)));
        });
    }
}

static void add_misc_tests(std::vector<Test> &tests) {
    for (ScalarDataType dtype : kAllDtypes) {
        if (!is_float(dtype)) {
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.isnan(b.input_arg(dtype))); });
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.isinf(b.input_arg(dtype))); });
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.isfinite(b.input_arg(dtype))); });
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.isnormal(b.input_arg(dtype))); });
        }
        if (!is_simple_int(dtype) && !is_float(dtype)) {
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.index(dtype)); });
        }
        if (!is_simple_int(dtype)) {
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.permute(b.input_arg(dtype), 0, false)); });
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.permute(b.input_arg(dtype), 0, true)); });
            add_invalid(tests, [dtype](FunctionBuilder &b) {
                b.output_arg(b.permute_i8_bits(b.input_arg(dtype), 0, 1, 2, 3, 4, 5, 6, 7));
            });
            add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.reverse_bits_i8(b.input_arg(dtype))); });
        }
    }

    for (ScalarDataType dtype : kAllDtypes) {
        if (dtype != I64) {
            add_invalid(tests, [dtype](FunctionBuilder &b) {
                b.output_arg(b.permute_i64_i16(b.input_arg(dtype), 0, 1, 2, 3));
            });
            add_invalid(tests,
                        [dtype](FunctionBuilder &b) { b.output_arg(b.permute_i64_i32(b.input_arg(dtype), 0, 1)); });
        }
        if (dtype != I32) {
            add_invalid(
                tests, [dtype](FunctionBuilder &b) { b.output_arg(b.permute_i32_i8(b.input_arg(dtype), 0, 1, 2, 3)); });
            add_invalid(tests,
                        [dtype](FunctionBuilder &b) { b.output_arg(b.permute_i32_i16(b.input_arg(dtype), 0, 1)); });
        }
        if (dtype != I16) {
            add_invalid(tests,
                        [dtype](FunctionBuilder &b) { b.output_arg(b.permute_i16_i8(b.input_arg(dtype), 0, 1)); });
        }
    }

    for (ScalarDataType dtype : kAllDtypes) {
        if (dtype == I16 || dtype == I32 || dtype == I64) { continue; }
        add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.byteswap(b.input_arg(dtype))); });
        add_invalid(tests, [dtype](FunctionBuilder &b) { b.output_arg(b.reverse_bits_full(b.input_arg(dtype))); });
    }
}

static void add_checked_tests(std::vector<Test> &tests) {
    for (ScalarDataType dtype : kAllDtypes) {
        bool valid_checked_binary = dtype == I32 || dtype == I64;
        bool valid_checked_add_sub = is_simple_int(dtype);
        bool valid_checked_mul = is_simple_int(dtype);
        bool valid_checked_shift = is_simple_int(dtype);
        if (dtype == ScalarDataType::I1) continue;
        if (!valid_checked_add_sub) {
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.add_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.sub_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
        }
        if (!valid_checked_mul) {
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.mul_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
        }
        if (!valid_checked_binary) {
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.div_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.udiv_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.mod_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.umod_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
        }
        if (!valid_checked_shift) {
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.sll_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.srl_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.sra_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.rotl_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
            add_unsupported(tests, [dtype](FunctionBuilder &b) {
                b.arg_safety_check();
                b.output_arg(b.rotr_checked(b.input_arg(dtype), b.input_arg(dtype)));
            });
        }
    }
}

static std::vector<Test> make_invalid_tests() {
    std::vector<Test> tests;
    add_load_predicate_tests(tests);
    add_store_tests(tests);
    add_pack_tests(tests);
    add_gather_scatter_tests(tests);
    add_aggregate_tests(tests);
    add_grouped_aggregate_tests(tests);
    add_binary_and_select_tests(tests);
    add_cast_tests(tests);
    add_unary_tests(tests);
    add_misc_tests(tests);
    add_checked_tests(tests);
    return tests;
}

} // namespace

#undef SIMJIT_VALUE_BINARY_OPS
#undef SIMJIT_SIGNED_CMP_OPS
#undef SIMJIT_UNSIGNED_CMP_OPS
#undef SIMJIT_ARITH_AGG_OPS
#undef SIMJIT_PREDICATE_AGG_OPS
#undef SIMJIT_GROUPED_AGG_OPS
#undef SIMJIT_GENERIC_UNARY_OPS
#undef SIMJIT_INT_UNARY_OPS
#undef SIMJIT_FLOAT_UNARY_OPS

std::vector<Test> invalid_type_tests = make_invalid_tests();
