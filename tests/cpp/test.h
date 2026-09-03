// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/simjit.h"
#include <functional>

using FuncType = std::function<void(simjit::FunctionBuilder &)>;

enum class TestVariant : uint8_t {
    None = 0,
    X86Scalar = 1 << 0,
    X86Vector = 1 << 1,
    ArmScalar = 1 << 2,
    ArmVector = 1 << 3,

    All = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3),
    X86All = (1 << 0) | (1 << 1),
    ArmAll = (1 << 2) | (1 << 3),
    ScalarAll = (1 << 0) | (1 << 2),
    VectorAll = (1 << 1) | (1 << 3),
};
SIMJIT_DEFINE_ENUM_FLAGS(TestVariant)

struct TestExpectation {
    bool has_error = false;
    simjit::ErrorKind kind = simjit::ErrorKind::InvalidInput;
};

constexpr bool operator==(TestExpectation left, TestExpectation right) {
    return left.has_error == right.has_error && left.kind == right.kind;
}

constexpr bool operator!=(TestExpectation left, TestExpectation right) {
    return !(left == right);
}

constexpr TestExpectation EXPECT_SUCCESS{};
constexpr TestExpectation EXPECT_INVALID_INPUT{true, simjit::ErrorKind::InvalidInput};
constexpr TestExpectation EXPECT_UNSUPPORTED{true, simjit::ErrorKind::Unsupported};

struct TestErrorInfoExpectation {
    bool has_error = false;
    simjit::ErrorModule module = simjit::ErrorModule::Generic;
    simjit::ErrorKind kind = simjit::ErrorKind::InvalidInput;
    simjit::ErrorSubKind subkind = simjit::ErrorSubKind::None;
};

constexpr uint8_t test_variant_bits(TestVariant mask) {
    return static_cast<uint8_t>(mask);
}

constexpr TestVariant normalize_test_variants(TestVariant mask) {
    return static_cast<TestVariant>(test_variant_bits(mask) & test_variant_bits(TestVariant::All));
}

constexpr bool test_variant_any(TestVariant mask) {
    return normalize_test_variants(mask) != TestVariant::None;
}

constexpr bool test_variant_contains(TestVariant set, TestVariant subset) {
    set = normalize_test_variants(set);
    subset = normalize_test_variants(subset);
    return (set & subset) == subset;
}

struct TestMetadata {
    TestVariant runnable = TestVariant::All;
    TestVariant passing = TestVariant::All;
    TestVariant limitations = TestVariant::None;
    TestVariant unstable_llvm = TestVariant::None;
    TestVariant unstable_asmjit = TestVariant::None;
    TestVariant unstable_cpp = TestVariant::None;
    simjit::ErrorSubKind x86_vectorization_failure = simjit::ErrorSubKind::None;
    simjit::ErrorSubKind arm_vectorization_failure = simjit::ErrorSubKind::None;
    TestErrorInfoExpectation x86_scalar_error = {};
    TestErrorInfoExpectation x86_vector_error = {};
    TestErrorInfoExpectation arm_scalar_error = {};
    TestErrorInfoExpectation arm_vector_error = {};

    constexpr TestMetadata only(TestVariant mask) const {
        mask = normalize_test_variants(mask);
        return TestMetadata{
            .runnable = runnable & mask,
            .passing = passing & mask,
            .limitations = limitations & mask,
            .unstable_llvm = unstable_llvm & mask,
            .unstable_asmjit = unstable_asmjit & mask,
            .unstable_cpp = unstable_cpp & mask,
            .x86_vectorization_failure = test_variant_contains(mask, TestVariant::X86Vector)
                                             ? x86_vectorization_failure
                                             : simjit::ErrorSubKind::None,
            .arm_vectorization_failure = test_variant_contains(mask, TestVariant::ArmVector)
                                             ? arm_vectorization_failure
                                             : simjit::ErrorSubKind::None,
            .x86_scalar_error =
                test_variant_contains(mask, TestVariant::X86Scalar) ? x86_scalar_error : TestErrorInfoExpectation{},
            .x86_vector_error =
                test_variant_contains(mask, TestVariant::X86Vector) ? x86_vector_error : TestErrorInfoExpectation{},
            .arm_scalar_error =
                test_variant_contains(mask, TestVariant::ArmScalar) ? arm_scalar_error : TestErrorInfoExpectation{},
            .arm_vector_error =
                test_variant_contains(mask, TestVariant::ArmVector) ? arm_vector_error : TestErrorInfoExpectation{},
        };
    }

    constexpr TestMetadata skip(TestVariant mask) const {
        TestVariant keep = normalize_test_variants(TestVariant::All & ~mask);
        return TestMetadata{
            .runnable = runnable & keep,
            .passing = passing & keep,
            .limitations = limitations & keep,
            .unstable_llvm = unstable_llvm & keep,
            .unstable_asmjit = unstable_asmjit & keep,
            .unstable_cpp = unstable_cpp & keep,
            .x86_vectorization_failure = test_variant_contains(keep, TestVariant::X86Vector)
                                             ? x86_vectorization_failure
                                             : simjit::ErrorSubKind::None,
            .arm_vectorization_failure = test_variant_contains(keep, TestVariant::ArmVector)
                                             ? arm_vectorization_failure
                                             : simjit::ErrorSubKind::None,
            .x86_scalar_error =
                test_variant_contains(keep, TestVariant::X86Scalar) ? x86_scalar_error : TestErrorInfoExpectation{},
            .x86_vector_error =
                test_variant_contains(keep, TestVariant::X86Vector) ? x86_vector_error : TestErrorInfoExpectation{},
            .arm_scalar_error =
                test_variant_contains(keep, TestVariant::ArmScalar) ? arm_scalar_error : TestErrorInfoExpectation{},
            .arm_vector_error =
                test_variant_contains(keep, TestVariant::ArmVector) ? arm_vector_error : TestErrorInfoExpectation{},
        };
    }

    constexpr TestMetadata bug(TestVariant mask) const {
        mask = normalize_test_variants(runnable & mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = normalize_test_variants(passing & ~mask),
            .limitations = normalize_test_variants(limitations & ~mask),
            .unstable_llvm = unstable_llvm,
            .unstable_asmjit = unstable_asmjit,
            .unstable_cpp = unstable_cpp,
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata limitation(TestVariant mask) const {
        mask = normalize_test_variants(runnable & mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = normalize_test_variants(passing & ~mask),
            .limitations = normalize_test_variants(limitations | mask),
            .unstable_llvm = unstable_llvm,
            .unstable_asmjit = unstable_asmjit,
            .unstable_cpp = unstable_cpp,
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata unstable(TestVariant mask) const {
        mask = normalize_test_variants(runnable & mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = passing,
            .limitations = limitations,
            .unstable_llvm = normalize_test_variants(unstable_llvm | mask),
            .unstable_asmjit = normalize_test_variants(unstable_asmjit | mask),
            .unstable_cpp = normalize_test_variants(unstable_cpp | mask),
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata unstable_llvm_only(TestVariant mask) const {
        mask = normalize_test_variants(runnable & mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = passing,
            .limitations = limitations,
            .unstable_llvm = normalize_test_variants(unstable_llvm | mask),
            .unstable_asmjit = unstable_asmjit,
            .unstable_cpp = unstable_cpp,
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata unstable_asmjit_only(TestVariant mask) const {
        mask = normalize_test_variants(runnable & mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = passing,
            .limitations = limitations,
            .unstable_llvm = unstable_llvm,
            .unstable_asmjit = normalize_test_variants(unstable_asmjit | mask),
            .unstable_cpp = unstable_cpp,
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata unstable_cpp_only(TestVariant mask) const {
        mask = normalize_test_variants(runnable & mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = passing,
            .limitations = limitations,
            .unstable_llvm = unstable_llvm,
            .unstable_asmjit = unstable_asmjit,
            .unstable_cpp = normalize_test_variants(unstable_cpp | mask),
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata vectorization_failure(TestVariant mask, simjit::ErrorSubKind kind) const {
        mask = normalize_test_variants(mask);
        return TestMetadata{
            .runnable = runnable,
            .passing = passing,
            .limitations = limitations,
            .unstable_llvm = unstable_llvm,
            .unstable_asmjit = unstable_asmjit,
            .unstable_cpp = unstable_cpp,
            .x86_vectorization_failure =
                test_variant_contains(mask, TestVariant::X86Vector) ? kind : x86_vectorization_failure,
            .arm_vectorization_failure =
                test_variant_contains(mask, TestVariant::ArmVector) ? kind : arm_vectorization_failure,
            .x86_scalar_error = x86_scalar_error,
            .x86_vector_error = x86_vector_error,
            .arm_scalar_error = arm_scalar_error,
            .arm_vector_error = arm_vector_error,
        };
    }

    constexpr TestMetadata structured_error(TestVariant mask, simjit::ErrorModule module, simjit::ErrorKind kind,
                                            simjit::ErrorSubKind subkind = simjit::ErrorSubKind::None) const {
        mask = normalize_test_variants(mask);
        TestErrorInfoExpectation error{true, module, kind, subkind};
        return TestMetadata{
            .runnable = runnable,
            .passing = passing,
            .limitations = limitations,
            .unstable_llvm = unstable_llvm,
            .unstable_asmjit = unstable_asmjit,
            .unstable_cpp = unstable_cpp,
            .x86_vectorization_failure = x86_vectorization_failure,
            .arm_vectorization_failure = arm_vectorization_failure,
            .x86_scalar_error = test_variant_contains(mask, TestVariant::X86Scalar) ? error : x86_scalar_error,
            .x86_vector_error = test_variant_contains(mask, TestVariant::X86Vector) ? error : x86_vector_error,
            .arm_scalar_error = test_variant_contains(mask, TestVariant::ArmScalar) ? error : arm_scalar_error,
            .arm_vector_error = test_variant_contains(mask, TestVariant::ArmVector) ? error : arm_vector_error,
        };
    }
};

constexpr TestMetadata test_meta() {
    return {};
}

// These shorthands preserve the existing declaration style while making the metadata explicit per variant.
constexpr TestMetadata PASS_ALL = test_meta();
constexpr TestMetadata BUG_X86_VECTOR = test_meta().bug(TestVariant::X86Vector);
constexpr TestMetadata BUG_ARM_VECTOR = test_meta().bug(TestVariant::ArmVector);
constexpr TestMetadata LIMIT_X86_VECTOR = test_meta().limitation(TestVariant::X86Vector);
constexpr TestMetadata LIMIT_ARM_VECTOR = test_meta().limitation(TestVariant::ArmVector);
constexpr TestMetadata ONLY_SCALAR = test_meta().only(TestVariant::ScalarAll);
constexpr TestMetadata BUG_ALL_VECTOR = test_meta().bug(TestVariant::VectorAll);
constexpr TestMetadata LIMIT_ALL_VECTOR = test_meta().limitation(TestVariant::VectorAll);
constexpr TestMetadata BUG_ALL = test_meta().bug(TestVariant::All);

constexpr TestMetadata coefficient_range_limit(TestVariant mask) {
    return test_meta().limitation(mask).vectorization_failure(mask,
                                                              simjit::ErrorSubKind::CoefficientRangeNeedsNormalization);
}

constexpr const char *relative_test_file(const char *path) {
    for (const char *p = path; *p != '\0'; ++p) {
        if (p[0] == 't' && p[1] == 'e' && p[2] == 's' && p[3] == 't' && p[4] == 's' && p[5] == '/') { return p; }
    }
    return path;
}

struct Test {
    FuncType builder;
    TestMetadata meta{};
    std::string_view python{};
    TestExpectation expectation = EXPECT_SUCCESS;
    const char *file = relative_test_file(__builtin_FILE());
    int line = __builtin_LINE();
};

extern std::vector<Test> float_tests;
extern std::vector<Test> int_tests;
extern std::vector<Test> nullable_tests;
extern std::vector<Test> tpcds_tests;
extern std::vector<Test> general_tests;
extern std::vector<Test> libdivide_tests;
extern std::vector<Test> agg_tests;
extern std::vector<Test> invalid_type_tests;
extern std::vector<Test> invalid_builder_tests;
extern std::vector<Test> misc_tests;
extern std::vector<Test> ternarylogic_tests;

inline void add_valid(std::vector<Test> &tests, FuncType builder,
                      const char *file = relative_test_file(__builtin_FILE()), int line = __builtin_LINE()) {
    tests.push_back(Test{std::move(builder), PASS_ALL, {}, EXPECT_SUCCESS, file, line});
}

inline void add_valid(std::vector<Test> &tests, FuncType builder, TestMetadata meta,
                      const char *file = relative_test_file(__builtin_FILE()), int line = __builtin_LINE()) {
    tests.push_back(Test{std::move(builder), meta, {}, EXPECT_SUCCESS, file, line});
}

inline void add_invalid(std::vector<Test> &tests, FuncType builder,
                        const char *file = relative_test_file(__builtin_FILE()), int line = __builtin_LINE()) {
    tests.push_back(Test{std::move(builder), PASS_ALL, {}, EXPECT_INVALID_INPUT, file, line});
}

inline void add_unsupported(std::vector<Test> &tests, FuncType builder,
                            const char *file = relative_test_file(__builtin_FILE()), int line = __builtin_LINE()) {
    tests.push_back(Test{std::move(builder), PASS_ALL, {}, EXPECT_UNSUPPORTED, file, line});
}
