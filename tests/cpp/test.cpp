// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"
#if SIMJIT_ASMJIT_BACKEND
#include "simjit/asmjit.h"
#endif
#include "simjit/core/hir.h"
#include "simjit/core/mir.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <format>
#include <fstream>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

// unity build because reasons
#include "agg_test.cpp"
#include "api_smoke_test.cpp"
#include "float_test.cpp"
#include "general_test.cpp"
#include "int_test.cpp"
#include "invalid_builder_test.cpp"
#include "invalid_type_test.cpp"
#include "libdivide_test.cpp"
#include "misc_test.cpp"
#include "nullable_test.cpp"
#include "ternarylogic_test.cpp"
#include "tpcds_test.cpp"

using namespace simjit;

#if SIMJIT_ASMJIT_BACKEND
using AsmjitState = AsmjitSession;
#else
struct AsmjitState {};
#endif

enum class RunnerArch : uint8_t {
    Native,
    X86,
    X86YMM,
    Arm,
    All,
};

enum class VectorMode : uint8_t {
    Auto,
    Scalar,
    NoVect,
    All,
};

enum class LogStage : uint8_t {
    Hir,
    Vectorizer,
    Mir,
};

enum class ExpectedOutcome : uint8_t {
    Pass,
    Bug,
    Limitation,
};

struct TestRunConfig {
    std::vector<std::string> selected_suites;
    RunnerArch arch = RunnerArch::Native;
    VectorMode vector_mode = VectorMode::Auto;
    bool emit_cpp = false;
    bool emit_llvm = false;
    bool emit_asmjit = false;
    bool validate_serialization = false;
    bool dump_json = false;
    std::string json_path = "tests.jsonl";
    bool log_failures = false;
    bool arena_stats = false;
    std::set<LogStage> log_stages;
    int iterations = 1;
    std::set<std::string> selected_test_ids;
    std::set<std::string> selected_locations;
    std::set<size_t> selected_test_numbers;
    bool has_positive_backend_selection = false;
    bool disable_cpp_backend = false;
    bool explicit_cpp_emit_requested = false;
};

struct SuiteDefinition {
    std::string_view id;
    size_t index;
    const std::vector<Test> *tests;
};

struct RegisteredTest {
    const Test *test = nullptr;
    std::string_view suite_id;
    size_t suite_index = 0;
    size_t case_index = 0;
    size_t transient_number = SIZE_MAX;
    std::string persistent_id;
    std::string location;
};

struct LoweringResult {
    mir::Function *mir = nullptr;
    vect::Function *vect = nullptr;
    bool used_vectorizer = false;
};

struct PipelineTiming {
    double hir_us = 0;
    double vectorizer_us = 0;
    double mir_us = 0;

    double total_us() const { return hir_us + vectorizer_us + mir_us; }
};

struct VariantResolution {
    TestVariant variant = TestVariant::None;

    bool runnable() const { return variant != TestVariant::None; }
};

struct RunResult {
    std::string json;
    bool failed = false;
    bool skipped = false;
    bool compiled = false;
    bool matched_expected_error = false;
    PipelineTiming pipeline_timing;
    double llvm_time = 0;
    double asmjit_time = 0;
    bool used_vectorizer = false;
    bool arena_sampled = false;
    size_t arena_used = 0;
    size_t arena_reserved = 0;
};

struct TimingSummary {
    double hir_total_us = 0;
    double vectorizer_total_us = 0;
    double mir_total_us = 0;
    double llvm_total_us = 0;
    double asmjit_total_us = 0;
    size_t hir_count = 0;
    size_t vectorizer_count = 0;
    size_t mir_count = 0;
    size_t llvm_count = 0;
    size_t asmjit_count = 0;
};

struct ArenaSizeSummary {
    size_t count = 0;
    size_t min_used = SIZE_MAX;
    size_t max_used = 0;
    size_t total_used = 0;
    size_t min_reserved = SIZE_MAX;
    size_t max_reserved = 0;
    size_t total_reserved = 0;
};

struct ConfigRunSummary {
    size_t executed = 0;
    size_t skipped = 0;
    size_t compiled = 0;
    size_t matched_expected_error = 0;
    int failures = 0;
    TimingSummary timing;
    ArenaSizeSummary arena;
};

struct UsageError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

enum class AnsiStyle : uint8_t {
    Plain,
    Bold,
    Heading,
    Success,
    Warning,
    Error,
    Accent,
};

static bool g_use_ansi = false;

static const std::array<SuiteDefinition, 11> kSuites{{
    {"int", 0, &int_tests},
    {"float", 1, &float_tests},
    {"nullable", 2, &nullable_tests},
    {"tpcds", 3, &tpcds_tests},
    {"general", 4, &general_tests},
    {"libdivide", 5, &libdivide_tests},
    {"agg", 6, &agg_tests},
    {"invalid_type", 7, &invalid_type_tests},
    {"invalid_builder", 8, &invalid_builder_tests},
    {"misc", 9, &misc_tests},
    {"ternarylogic", 10, &ternarylogic_tests},
}};

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

static bool detect_ansi_output() {
    if (std::getenv("NO_COLOR") != nullptr) { return false; }
    if (const char *term = std::getenv("TERM")) {
        if (std::string_view(term) == "dumb") { return false; }
    }
    return isatty(fileno(stdout)) != 0;
}

static const char *ansi_code(AnsiStyle style) {
    switch (style) {
    case AnsiStyle::Plain: return "";
    case AnsiStyle::Bold: return "\033[1m";
    case AnsiStyle::Heading: return "\033[1;36m";
    case AnsiStyle::Success: return "\033[1;32m";
    case AnsiStyle::Warning: return "\033[1;33m";
    case AnsiStyle::Error: return "\033[1;31m";
    case AnsiStyle::Accent: return "\033[1;34m";
    }
    return "";
}

static std::string_view normalize_suite_id(std::string_view suite_id) {
    if (suite_id == "llm") { return "general"; }
    return suite_id;
}

static std::string styled(std::string_view text, AnsiStyle style) {
    if (!g_use_ansi || style == AnsiStyle::Plain) { return std::string(text); }
    return std::string(ansi_code(style)) + std::string(text) + "\033[0m";
}

static std::string escape_json(std::string_view src) {
    std::string result;
    result.reserve(src.size());

    for (unsigned char c : src) {
        switch (c) {
        case '"': result += "\\\""; break;
        case '\\': result += "\\\\"; break;
        case '/': result += "\\/"; break;
        case '\b': result += "\\b"; break;
        case '\f': result += "\\f"; break;
        case '\n': result += "\\n"; break;
        case '\r': result += "\\r"; break;
        case '\t': result += "\\t"; break;
        default:
            if (std::isprint(c)) {
                result += static_cast<char>(c);
                break;
            }
            throw std::runtime_error(std::format("invalid character 0x{:x}", static_cast<unsigned>(c)));
        }
    }

    return result;
}

static std::string base64_encode(uint8_t const *buf, unsigned int buf_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];

    while (buf_len--) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; ++i) {
                ret += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; ++j) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; j < i + 1; ++j) {
            ret += base64_chars[char_array_4[j]];
        }

        while (i++ < 3) {
            ret += '=';
        }
    }

    return ret;
}

static std::string make_base64(std::span<uint8_t const> memory) {
    return base64_encode(memory.data(), memory.size());
}

static std::string schema_json(nonstd::span<ArgumentDecl> args) {
    std::string result = "{ \"args\": [";
    for (auto &arg : args) {
        std::string dtype;
        switch (arg.dtype) {
        case ScalarDataType::I8: dtype = "i8"; break;
        case ScalarDataType::I16: dtype = "i16"; break;
        case ScalarDataType::I32: dtype = "i32"; break;
        case ScalarDataType::I64: dtype = "i64"; break;
        case ScalarDataType::F32: dtype = "f32"; break;
        case ScalarDataType::F64: dtype = "f64"; break;
        case ScalarDataType::I1: dtype = "i1"; break;
        case ScalarDataType::I128: dtype = "i128"; break;
        }
        std::string kind;
        if ((arg.kind & simjit::ArgumentKind::SrcIdxArr) != simjit::ArgumentKind::Undefined) {
            kind = "sv";
        } else if ((arg.kind & (simjit::ArgumentKind::SrcGatherArr | simjit::ArgumentKind::SrcArr |
                                ArgumentKind::SrcConst)) != simjit::ArgumentKind::Undefined) {
            kind = "in";
        } else if ((arg.kind & simjit::ArgumentKind::DstSafetyCheck) != simjit::ArgumentKind::Undefined) {
            kind = "safety";
        } else if ((arg.kind & simjit::ArgumentKind::DstAgg) != simjit::ArgumentKind::Undefined) {
            kind = "outs";
        } else if ((arg.kind & simjit::ArgumentKind::Dst) != simjit::ArgumentKind::Undefined) {
            kind = "out";
        } else {
            throw std::runtime_error("invalid arg kind");
        }
        result += std::format("{{ \"dtype\": \"{}\", \"kind\": \"{}\"}}", dtype, kind);
        if (&arg != &args[args.size() - 1]) { result += ","; }
    }
    result += "]}";
    return result;
}

static std::string make_code_obj(std::string_view name, std::string_view code, bool comparison_unstable = false) {
    if (comparison_unstable) {
        return std::format("{{\"name\": \"{}\", \"code\": \"{}\", \"comparison_unstable\": true}}", name, code);
    }
    return std::format("{{\"name\": \"{}\", \"code\": \"{}\"}}", name, code);
}

enum class EmittedCodeKind : uint8_t {
    Llvm,
    Asmjit,
    Cpp,
};

static bool code_comparison_is_unstable(const TestMetadata &meta, EmittedCodeKind kind, TestVariant variant) {
    switch (kind) {
    case EmittedCodeKind::Llvm: return test_variant_contains(meta.unstable_llvm, variant);
    case EmittedCodeKind::Asmjit: return test_variant_contains(meta.unstable_asmjit, variant);
    case EmittedCodeKind::Cpp: return test_variant_contains(meta.unstable_cpp, variant);
    }
    SIMJIT_UNREACHABLE();
}

static std::optional<std::string_view> getenv_str(std::string_view name) {
    if (const char *value = std::getenv(std::string{name}.c_str())) { return std::string_view(value); }
    return std::nullopt;
}

static bool getenv_flag(std::string_view name) {
    return getenv_str(name).has_value();
}

static const SuiteDefinition *find_suite(std::string_view suite_id) {
    suite_id = normalize_suite_id(suite_id);
    for (const auto &suite : kSuites) {
        if (suite.id == suite_id) { return &suite; }
    }
    return nullptr;
}

static void ensure_suite_selected(TestRunConfig &config, std::string_view suite_id) {
    suite_id = normalize_suite_id(suite_id);
    if (std::find(config.selected_suites.begin(), config.selected_suites.end(), suite_id) ==
        config.selected_suites.end()) {
        config.selected_suites.emplace_back(suite_id);
    }
}

static std::vector<std::string_view> split_csv(std::string_view str) {
    std::vector<std::string_view> result;
    size_t start = 0;
    while (start <= str.size()) {
        size_t comma = str.find(',', start);
        size_t end = comma == std::string_view::npos ? str.size() : comma;
        std::string_view part = str.substr(start, end - start);
        if (part.empty()) { throw UsageError("empty value in comma-separated list"); }
        result.push_back(part);
        if (comma == std::string_view::npos) { break; }
        start = comma + 1;
    }
    return result;
}

static int parse_positive_int(std::string_view value, std::string_view flag_name) {
    int parsed = 0;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc{} || ptr != value.data() + value.size() || parsed < 1) {
        throw UsageError(std::format("invalid value for {}: {}", flag_name, value));
    }
    return parsed;
}

static size_t parse_non_negative_size(std::string_view value, std::string_view flag_name) {
    size_t parsed = 0;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc{} || ptr != value.data() + value.size()) {
        throw UsageError(std::format("invalid value for {}: {}", flag_name, value));
    }
    return parsed;
}

static std::string parse_test_id(std::string_view value) {
    size_t colon = value.find(':');
    if (colon == std::string_view::npos || colon == 0 || colon == value.size() - 1) {
        throw UsageError(std::format("invalid --test-id value: {}", value));
    }

    std::string_view suite = value.substr(0, colon);
    if (find_suite(suite) == nullptr) { throw UsageError(std::format("unknown suite in --test-id: {}", suite)); }

    size_t idx = parse_non_negative_size(value.substr(colon + 1), "--test-id");
    return std::format("{}:{}", suite, idx);
}

static std::string parse_test_location(std::string_view value) {
    size_t colon = value.rfind(':');
    if (colon == std::string_view::npos || colon == 0 || colon == value.size() - 1) {
        throw UsageError(std::format("invalid --test-at value: {}", value));
    }

    std::string_view path = value.substr(0, colon);
    int line = parse_positive_int(value.substr(colon + 1), "--test-at");
    return std::format("{}:{}", path, line);
}

static size_t parse_test_number(std::string_view value) {
    return parse_non_negative_size(value, "--test-num");
}

static RunnerArch parse_arch_value(std::string_view value) {
    if (value == "native") { return RunnerArch::Native; }
    if (value == "x86") { return RunnerArch::X86; }
    if (value == "x86-ymm" || value == "avx512-ymm") { return RunnerArch::X86YMM; }
    if (value == "arm") { return RunnerArch::Arm; }
    if (value == "all") { return RunnerArch::All; }
    throw UsageError(std::format("invalid --arch value: {}", value));
}

static VectorMode parse_mode_value(std::string_view value) {
    if (value == "auto") { return VectorMode::Auto; }
    if (value == "scalar") { return VectorMode::Scalar; }
    if (value == "novect") { return VectorMode::NoVect; }
    if (value == "all") { return VectorMode::All; }
    throw UsageError(std::format("invalid --mode value: {}", value));
}

static LogStage parse_log_stage(std::string_view value) {
    if (value == "hir") { return LogStage::Hir; }
    if (value == "vectorizer") { return LogStage::Vectorizer; }
    if (value == "mir") { return LogStage::Mir; }
    throw UsageError(std::format("invalid --log value: {}", value));
}

static void apply_suite_csv(TestRunConfig &config, std::string_view csv) {
    for (std::string_view value : split_csv(csv)) {
        if (value == "all") {
            for (const auto &suite : kSuites) {
                ensure_suite_selected(config, suite.id);
            }
            continue;
        }
        const auto *suite = find_suite(value);
        if (suite == nullptr) { throw UsageError(std::format("invalid --suite value: {}", value)); }
        ensure_suite_selected(config, suite->id);
    }
}

static void apply_emit_csv(TestRunConfig &config, std::string_view csv) {
    for (std::string_view value : split_csv(csv)) {
        if (value == "all") {
            config.emit_cpp = true;
            config.emit_llvm = true;
            config.emit_asmjit = true;
            config.explicit_cpp_emit_requested = true;
            config.has_positive_backend_selection = true;
        } else if (value == "cpp") {
            config.emit_cpp = true;
            config.explicit_cpp_emit_requested = true;
            config.has_positive_backend_selection = true;
        } else if (value == "llvm") {
            config.emit_llvm = true;
            config.has_positive_backend_selection = true;
        } else if (value == "asmjit") {
            config.emit_asmjit = true;
            config.has_positive_backend_selection = true;
        } else {
            throw UsageError(std::format("invalid --emit value: {}", value));
        }
    }
}

static void apply_log_csv(TestRunConfig &config, std::string_view csv) {
    for (std::string_view value : split_csv(csv)) {
        if (value == "all") {
            config.log_stages.insert(LogStage::Hir);
            config.log_stages.insert(LogStage::Vectorizer);
            config.log_stages.insert(LogStage::Mir);
            continue;
        }
        config.log_stages.insert(parse_log_stage(value));
    }
}

static std::string join_strings(const std::vector<std::string> &items) {
    if (items.empty()) { return "none"; }

    std::string result;
    for (size_t i = 0; i < items.size(); ++i) {
        if (i != 0) { result += ","; }
        result += items[i];
    }
    return result;
}

static std::vector<std::string> emit_names(const TestRunConfig &config) {
    std::vector<std::string> result;
    if (config.emit_cpp) { result.emplace_back("cpp"); }
    if (config.emit_llvm) { result.emplace_back("llvm"); }
    if (config.emit_asmjit) { result.emplace_back("asmjit"); }
    return result;
}

static std::vector<std::string> log_stage_names(const TestRunConfig &config) {
    std::vector<std::string> result;
    if (config.log_stages.contains(LogStage::Hir)) { result.emplace_back("hir"); }
    if (config.log_stages.contains(LogStage::Vectorizer)) { result.emplace_back("vectorizer"); }
    if (config.log_stages.contains(LogStage::Mir)) { result.emplace_back("mir"); }
    return result;
}

static std::string arch_name(RunnerArch arch) {
    switch (arch) {
    case RunnerArch::Native:
        if constexpr (Arch::Native == Arch::Amd64_AVX512) {
            return "native (x86)";
        } else {
            return "native (arm)";
        }
    case RunnerArch::X86: return "x86";
    case RunnerArch::X86YMM: return "x86-ymm";
    case RunnerArch::Arm: return "arm";
    case RunnerArch::All: return "all";
    }
    throw std::runtime_error("unknown arch");
}

static std::string mode_name(VectorMode mode) {
    switch (mode) {
    case VectorMode::Auto: return "auto";
    case VectorMode::Scalar: return "scalar";
    case VectorMode::NoVect: return "novect";
    case VectorMode::All: return "all";
    }
    throw std::runtime_error("unknown mode");
}

static Arch resolve_arch(RunnerArch arch) {
    switch (arch) {
    case RunnerArch::Native: return Arch::Native;
    case RunnerArch::X86: return Arch::Amd64_AVX512;
    case RunnerArch::X86YMM: return Arch::Amd64_AVX512_YMM;
    case RunnerArch::Arm: return Arch::Arm64_NEON;
    case RunnerArch::All: break;
    }
    throw std::runtime_error("unknown arch");
}

static bool is_arm_arch(RunnerArch arch) {
    return resolve_arch(arch) == Arch::Arm64_NEON;
}

static bool cpp_backend_available(RunnerArch arch) {
#if SIMJIT_CPP_BACKEND
    (void)arch;
    return true;
#else
    (void)arch;
    return false;
#endif
}

static bool cpp_backend_available(TestVariant variant) {
#if SIMJIT_CPP_BACKEND
    (void)variant;
    return true;
#else
    (void)variant;
    return false;
#endif
}

static bool llvm_backend_available() {
#if SIMJIT_LLVM_BACKEND
    return true;
#else
    return false;
#endif
}

static bool asmjit_backend_available() {
#if SIMJIT_ASMJIT_BACKEND
    return true;
#else
    return false;
#endif
}

static bool variant_is_scalar(TestVariant variant) {
    return variant == TestVariant::X86Scalar || variant == TestVariant::ArmScalar;
}

static TestVariant scalar_variant_for(TestVariant variant) {
    switch (variant) {
    case TestVariant::X86Vector:
    case TestVariant::X86Scalar: return TestVariant::X86Scalar;
    case TestVariant::ArmVector:
    case TestVariant::ArmScalar: return TestVariant::ArmScalar;
    case TestVariant::None:
    case TestVariant::All:
    case TestVariant::X86All:
    case TestVariant::ArmAll:
    case TestVariant::ScalarAll:
    case TestVariant::VectorAll: break;
    }
    throw std::runtime_error("invalid test variant");
}

static TestVariant preferred_variant(const TestRunConfig &config, bool force_scalar) {
    bool arm = is_arm_arch(config.arch);
    if (force_scalar || config.vector_mode == VectorMode::Scalar || config.vector_mode == VectorMode::NoVect) {
        return arm ? TestVariant::ArmScalar : TestVariant::X86Scalar;
    }
    return arm ? TestVariant::ArmVector : TestVariant::X86Vector;
}

static VariantResolution resolve_variant(const TestRunConfig &config, const Test &test, bool force_scalar) {
    TestVariant variant = preferred_variant(config, force_scalar);
    if (test_variant_contains(test.meta.runnable, variant)) { return VariantResolution{.variant = variant}; }

    if (!variant_is_scalar(variant)) {
        TestVariant scalar = scalar_variant_for(variant);
        if (test_variant_contains(test.meta.runnable, scalar)) { return VariantResolution{.variant = scalar}; }
    }

    return {};
}

static ExpectedOutcome expected_outcome(const Test &test, TestVariant variant) {
    if (!test_variant_contains(test.meta.runnable, variant)) {
        throw std::runtime_error("requested expectation for unrunnable variant");
    }
    if (test_variant_contains(test.meta.passing, variant)) { return ExpectedOutcome::Pass; }
    if (test_variant_contains(test.meta.limitations, variant)) { return ExpectedOutcome::Limitation; }
    return ExpectedOutcome::Bug;
}

static ErrorSubKind expected_vectorization_failure(const TestMetadata &meta, TestVariant variant) {
    switch (variant) {
    case TestVariant::X86Vector: return meta.x86_vectorization_failure;
    case TestVariant::ArmVector: return meta.arm_vectorization_failure;
    case TestVariant::X86Scalar:
    case TestVariant::ArmScalar: return ErrorSubKind::None;
    case TestVariant::None:
    case TestVariant::All:
    case TestVariant::X86All:
    case TestVariant::ArmAll:
    case TestVariant::ScalarAll:
    case TestVariant::VectorAll: break;
    }
    throw std::runtime_error("invalid test variant");
}

static TestErrorInfoExpectation expected_structured_error(const TestMetadata &meta, TestVariant variant) {
    switch (variant) {
    case TestVariant::X86Scalar: return meta.x86_scalar_error;
    case TestVariant::X86Vector: return meta.x86_vector_error;
    case TestVariant::ArmScalar: return meta.arm_scalar_error;
    case TestVariant::ArmVector: return meta.arm_vector_error;
    case TestVariant::None:
    case TestVariant::All:
    case TestVariant::X86All:
    case TestVariant::ArmAll:
    case TestVariant::ScalarAll:
    case TestVariant::VectorAll: break;
    }
    throw std::runtime_error("invalid test variant");
}

static bool expected_structured_error_source_checked(const TestRunConfig &config, TestVariant variant,
                                                     const TestErrorInfoExpectation &expected_error) {
    if (!expected_error.has_error) { return true; }

    switch (expected_error.module) {
    case ErrorModule::LLVM: return config.emit_llvm && llvm_backend_available();
    case ErrorModule::CPP: return config.emit_cpp && cpp_backend_available(variant);
    case ErrorModule::AsmJit:
    case ErrorModule::X86:
    case ErrorModule::A64: return config.emit_asmjit && asmjit_backend_available();
    case ErrorModule::Serialization: return config.dump_json || config.validate_serialization;
    case ErrorModule::Generic:
    case ErrorModule::HIR:
    case ErrorModule::Vectorizer:
    case ErrorModule::MIR:
    case ErrorModule::JIT:
    case ErrorModule::Nullable: return true;
    }
    SIMJIT_UNREACHABLE();
}

static bool structured_error_matches_expected(const TestRunConfig &config,
                                              const TestErrorInfoExpectation &expected_error,
                                              const ErrorInfo &observed_error) {
    if (observed_error.kind != expected_error.kind || observed_error.subkind != expected_error.subkind) {
        return false;
    }
    if (observed_error.module == expected_error.module) { return true; }

    bool expected_asmjit_family = expected_error.module == ErrorModule::AsmJit ||
                                  expected_error.module == ErrorModule::X86 ||
                                  expected_error.module == ErrorModule::A64;
    return config.emit_cpp && observed_error.module == ErrorModule::CPP && expected_asmjit_family &&
           expected_error.kind == ErrorKind::Unsupported;
}

static const char *expected_outcome_name(ExpectedOutcome outcome) {
    switch (outcome) {
    case ExpectedOutcome::Pass: return "pass";
    case ExpectedOutcome::Bug: return "bug";
    case ExpectedOutcome::Limitation: return "limitation";
    }
    throw std::runtime_error("unknown expected outcome");
}

static const char *variant_name(TestVariant variant) {
    switch (variant) {
    case TestVariant::X86Scalar: return "x86-scalar";
    case TestVariant::X86Vector: return "x86-vector";
    case TestVariant::ArmScalar: return "arm-scalar";
    case TestVariant::ArmVector: return "arm-vector";
    case TestVariant::None:
    case TestVariant::All:
    case TestVariant::X86All:
    case TestVariant::ArmAll:
    case TestVariant::ScalarAll:
    case TestVariant::VectorAll: break;
    }
    throw std::runtime_error("invalid test variant");
}

static std::string variant_mask_name(TestVariant mask) {
    mask = normalize_test_variants(mask);
    if (mask == TestVariant::None) { return "None"; }

    struct VariantBitName {
        TestVariant bit;
        const char *name;
    };
    static const std::array<VariantBitName, 4> bits{{
        {TestVariant::X86Scalar, "X86S"},
        {TestVariant::X86Vector, "X86V"},
        {TestVariant::ArmScalar, "A64S"},
        {TestVariant::ArmVector, "A64V"},
    }};

    std::string result = "(";
    bool first = true;
    for (const auto &entry : bits) {
        if (!test_variant_contains(mask, entry.bit)) { continue; }
        if (!first) { result += " | "; }
        result += entry.name;
        first = false;
    }
    result += ")";
    return result;
}

static std::string test_metadata_str(const TestMetadata &meta) {
    return std::format("run={} pass={} limitation={} unstable_llvm={} unstable_asmjit={} unstable_cpp={} "
                       "x86_vec_fail={} arm_vec_fail={}",
                       variant_mask_name(meta.runnable), variant_mask_name(meta.passing),
                       variant_mask_name(meta.limitations), variant_mask_name(meta.unstable_llvm),
                       variant_mask_name(meta.unstable_asmjit), variant_mask_name(meta.unstable_cpp),
                       (uint16_t)meta.x86_vectorization_failure, (uint16_t)meta.arm_vectorization_failure);
}

static const char *test_expectation_name(TestExpectation expectation) {
    if (expectation == EXPECT_SUCCESS) { return "success"; }
    if (expectation == EXPECT_INVALID_INPUT) { return "invalid-input"; }
    if (expectation == EXPECT_UNSUPPORTED) { return "unsupported-expr"; }
    return "error";
}

static void validate_test_metadata(const RegisteredTest &test) {
    TestVariant runnable = normalize_test_variants(test.test->meta.runnable);
    TestVariant passing = normalize_test_variants(test.test->meta.passing);
    TestVariant limitations = normalize_test_variants(test.test->meta.limitations);
    TestVariant unstable_llvm = normalize_test_variants(test.test->meta.unstable_llvm);
    TestVariant unstable_asmjit = normalize_test_variants(test.test->meta.unstable_asmjit);
    TestVariant unstable_cpp = normalize_test_variants(test.test->meta.unstable_cpp);

    if (!test_variant_any(runnable)) {
        throw UsageError(std::format("test [{}] {} has empty runnable mask", test.persistent_id, test.location));
    }
    if (!test_variant_contains(runnable, passing)) {
        throw UsageError(
            std::format("test [{}] {} has passing variants outside runnable mask", test.persistent_id, test.location));
    }
    if (!test_variant_contains(runnable, limitations)) {
        throw UsageError(std::format("test [{}] {} has limitation variants outside runnable mask", test.persistent_id,
                                     test.location));
    }
    if (!test_variant_contains(runnable, unstable_llvm)) {
        throw UsageError(std::format("test [{}] {} has llvm unstable variants outside runnable mask",
                                     test.persistent_id, test.location));
    }
    if (!test_variant_contains(runnable, unstable_asmjit)) {
        throw UsageError(std::format("test [{}] {} has asmjit unstable variants outside runnable mask",
                                     test.persistent_id, test.location));
    }
    if (!test_variant_contains(runnable, unstable_cpp)) {
        throw UsageError(std::format("test [{}] {} has cpp unstable variants outside runnable mask", test.persistent_id,
                                     test.location));
    }
    if (test_variant_any(passing & limitations)) {
        throw UsageError(std::format("test [{}] {} has overlapping passing and limitation variants", test.persistent_id,
                                     test.location));
    }
    if (test.test->meta.x86_vectorization_failure != ErrorSubKind::None &&
        test_variant_contains(passing, TestVariant::X86Vector)) {
        throw UsageError(std::format("test [{}] {} expects x86 vectorization failure on passing variant",
                                     test.persistent_id, test.location));
    }
    if (test.test->meta.arm_vectorization_failure != ErrorSubKind::None &&
        test_variant_contains(passing, TestVariant::ArmVector)) {
        throw UsageError(std::format("test [{}] {} expects arm vectorization failure on passing variant",
                                     test.persistent_id, test.location));
    }
}

static void print_usage(FILE *out) {
    fprintf(out, "usage: build/debug/test [options]\n"
                 "\n"
                 "Select suites:\n"
                 "  --suite <csv>          Comma-separated suites: int,float,nullable,tpcds,general,libdivide,agg,"
                 "ternarylogic,invalid,all\n"
                 "\n"
                 "Execution:\n"
                 "  --arch <native|x86|x86-ymm|arm|all>\n"
                 "  --mode <auto|scalar|novect|all>\n"
                 "  --iterations <n>\n"
                 "\n"
                 "Backend emission:\n"
                 "  --emit <csv>           Comma-separated emits: cpp,llvm,asmjit,all\n"
                 "  --validate-serialization\n"
                 "                         Run serialize/deserialize/lower checks without JSON dump\n"
                 "  --dump-json <path>     Write JSONL dump to the given path\n"
                 "  --arena-stats          Print per-test arena used/reserved min, max, and average\n"
                 "\n"
                 "Logging:\n"
                 "  --log-failures         Print HIR when a positive test fails\n"
                 "  --log <csv>            Comma-separated stages: hir,vectorizer,mir,all\n"
                 "\n"
                 "Selection:\n"
                 "  --test-id <suite:idx>\n"
                 "  --test-num <n>\n"
                 "  --test-at <file:line>\n"
                 "\n"
                 "Other:\n"
                 "  --help\n"
                 "\n"
                 "Environment compatibility:\n"
                 "  TI TF TDS TGENERAL TAGG select suites\n"
                 "  SCALAR NOVECT ARM map to mode/arch\n"
                 "  LLVM ASMJIT NOCPP control emission\n"
                 "  DUMP enables JSONL dump to tests.jsonl unless overridden by --dump-json\n"
                 "  LOG enables failure HIR logging\n"
                 "  N sets iteration count\n");
}

static void apply_env_config(TestRunConfig &config) {
    if (getenv_flag("TI")) { ensure_suite_selected(config, "int"); }
    if (getenv_flag("TF")) { ensure_suite_selected(config, "float"); }
    if (getenv_flag("TDS")) { ensure_suite_selected(config, "tpcds"); }
    if (getenv_flag("TGENERAL") || getenv_flag("TLLM")) { ensure_suite_selected(config, "general"); }
    if (getenv_flag("TAGG")) { ensure_suite_selected(config, "agg"); }
    if (getenv_flag("TINV")) { ensure_suite_selected(config, "invalid"); }

    if (getenv_flag("SCALAR")) { config.vector_mode = VectorMode::Scalar; }
    if (getenv_flag("NOVECT")) { config.vector_mode = VectorMode::NoVect; }
    if (getenv_flag("ARM")) { config.arch = RunnerArch::Arm; }

    if (getenv_flag("LLVM")) {
        config.emit_llvm = true;
        config.has_positive_backend_selection = true;
    }
    if (getenv_flag("ASMJIT")) {
        config.emit_asmjit = true;
        config.has_positive_backend_selection = true;
    }
    if (getenv_flag("NOCPP")) { config.disable_cpp_backend = true; }

    if (getenv_flag("DUMP")) {
        config.dump_json = true;
        config.json_path = "tests.jsonl";
    }
    if (getenv_flag("LOG")) { config.log_failures = true; }

    if (auto n = getenv_str("N")) { config.iterations = parse_positive_int(*n, "N"); }
}

static void finalize_backend_selection(TestRunConfig &config) {
    if (!config.has_positive_backend_selection) {
        config.emit_cpp = cpp_backend_available(config.arch);
        config.emit_llvm = llvm_backend_available();
        config.emit_asmjit = asmjit_backend_available();
    }

    if (config.disable_cpp_backend || !cpp_backend_available(config.arch)) { config.emit_cpp = false; }
}

static std::vector<RunnerArch> expand_archs(RunnerArch arch) {
    if (arch == RunnerArch::All) { return {RunnerArch::X86, RunnerArch::Arm}; }
    return {arch};
}

static std::vector<VectorMode> expand_modes(VectorMode mode) {
    if (mode == VectorMode::All) { return {VectorMode::Auto, VectorMode::Scalar}; }
    return {mode};
}

static std::vector<TestRunConfig> expand_run_configs(const TestRunConfig &base) {
    std::vector<TestRunConfig> result;
    for (RunnerArch arch : expand_archs(base.arch)) {
        for (VectorMode mode : expand_modes(base.vector_mode)) {
            TestRunConfig config = base;
            config.arch = arch;
            config.vector_mode = mode;
            finalize_backend_selection(config);
            result.push_back(std::move(config));
        }
    }
    return result;
}

static TestRunConfig parse_config(int argc, char **argv) {
    TestRunConfig config;
    apply_env_config(config);

    bool suite_cli_seen = false;

    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        std::optional<std::string_view> inline_value;
        if (size_t eq = arg.find('='); eq != std::string_view::npos) {
            inline_value = arg.substr(eq + 1);
            arg = arg.substr(0, eq);
            if (inline_value->empty()) { throw UsageError(std::format("missing value for {}", arg)); }
        }
        auto require_value = [&](std::string_view name) -> std::string_view {
            if (inline_value.has_value()) { return *inline_value; }
            if (i + 1 >= argc) { throw UsageError(std::format("missing value for {}", name)); }
            return argv[++i];
        };

        if (arg == "--help") {
            if (inline_value.has_value()) { throw UsageError("unexpected value for --help"); }
            print_usage(stdout);
            std::exit(0);
        } else if (arg == "--suite") {
            if (!suite_cli_seen) {
                config.selected_suites.clear();
                suite_cli_seen = true;
            }
            apply_suite_csv(config, require_value("--suite"));
        } else if (arg == "--arch") {
            config.arch = parse_arch_value(require_value("--arch"));
        } else if (arg == "--mode") {
            config.vector_mode = parse_mode_value(require_value("--mode"));
        } else if (arg == "--emit") {
            config.emit_asmjit = config.emit_cpp = config.emit_llvm = false;
            apply_emit_csv(config, require_value("--emit"));
        } else if (arg == "--no-cpp") {
            if (inline_value.has_value()) { throw UsageError("unexpected value for --no-cpp"); }
            config.disable_cpp_backend = true;
        } else if (arg == "--dump-json") {
            config.dump_json = true;
            config.json_path = std::string(require_value("--dump-json"));
        } else if (arg == "--arena-stats") {
            if (inline_value.has_value()) { throw UsageError("unexpected value for --arena-stats"); }
            config.arena_stats = true;
        } else if (arg == "--validate-serialization") {
            if (inline_value.has_value()) { throw UsageError("unexpected value for --validate-serialization"); }
#if !SIMJIT_ENABLE_SERIALIZATION
            throw UsageError("this binary was built without serialization support");
#endif
            config.validate_serialization = true;
        } else if (arg == "--log-failures") {
            if (inline_value.has_value()) { throw UsageError("unexpected value for --log-failures"); }
            config.log_failures = true;
        } else if (arg == "--log") {
            apply_log_csv(config, require_value("--log"));
        } else if (arg == "--test-id") {
            config.selected_test_ids.insert(parse_test_id(require_value("--test-id")));
        } else if (arg == "--test-num") {
            config.selected_test_numbers.insert(parse_test_number(require_value("--test-num")));
        } else if (arg == "--test-at") {
            config.selected_locations.insert(parse_test_location(require_value("--test-at")));
        } else if (arg == "--iterations") {
            config.iterations = parse_positive_int(require_value("--iterations"), "--iterations");
        } else {
            throw UsageError(std::format("unknown option: {}", arg));
        }
    }

    if (config.selected_suites.empty()) { throw UsageError("no suites selected"); }

    return config;
}

static std::vector<RegisteredTest> register_tests() {
    std::vector<RegisteredTest> result;

    for (const auto &suite : kSuites) {
        result.reserve(result.size() + suite.tests->size());
        for (size_t i = 0; i < suite.tests->size(); ++i) {
            const Test &test = suite.tests->at(i);
            result.push_back(RegisteredTest{
                .test = &test,
                .suite_id = suite.id,
                .suite_index = suite.index,
                .case_index = i,
                .persistent_id = std::format("{}:{}", suite.id, i),
                .location = std::format("{}:{}", test.file, test.line),
            });
            validate_test_metadata(result.back());
        }
    }

    return result;
}

static bool suite_selected(const TestRunConfig &config, std::string_view suite_id) {
    return std::find(config.selected_suites.begin(), config.selected_suites.end(), suite_id) !=
           config.selected_suites.end();
}

static bool has_test_filter(const TestRunConfig &config) {
    return !config.selected_test_ids.empty() || !config.selected_locations.empty() ||
           !config.selected_test_numbers.empty();
}

static std::vector<RegisteredTest> select_tests(const TestRunConfig &config,
                                                const std::vector<RegisteredTest> &all_tests,
                                                size_t &registered_in_selected_suites) {
    std::vector<RegisteredTest> suite_tests;
    suite_tests.reserve(all_tests.size());
    registered_in_selected_suites = 0;

    for (const auto &test : all_tests) {
        if (!suite_selected(config, test.suite_id)) { continue; }

        RegisteredTest numbered = test;
        numbered.transient_number = registered_in_selected_suites;
        ++registered_in_selected_suites;
        suite_tests.push_back(std::move(numbered));
    }

    std::vector<RegisteredTest> result;
    std::set<std::string> matched_ids;
    std::set<std::string> matched_locations;
    std::set<size_t> matched_numbers;

    for (const auto &test : suite_tests) {
        bool include = true;
        if (has_test_filter(config)) {
            include = config.selected_test_ids.contains(test.persistent_id) ||
                      config.selected_locations.contains(test.location) ||
                      config.selected_test_numbers.contains(test.transient_number);
        }

        if (!include) { continue; }

        if (config.selected_test_ids.contains(test.persistent_id)) { matched_ids.insert(test.persistent_id); }
        if (config.selected_locations.contains(test.location)) { matched_locations.insert(test.location); }
        if (config.selected_test_numbers.contains(test.transient_number)) {
            matched_numbers.insert(test.transient_number);
        }
        result.push_back(test);
    }

    if (has_test_filter(config)) {
        std::vector<std::string> missing;
        for (const auto &id : config.selected_test_ids) {
            if (!matched_ids.contains(id)) { missing.push_back(std::format("missing --test-id {}", id)); }
        }
        for (const auto &location : config.selected_locations) {
            if (!matched_locations.contains(location)) {
                missing.push_back(std::format("missing --test-at {}", location));
            }
        }
        for (size_t number : config.selected_test_numbers) {
            if (!matched_numbers.contains(number)) { missing.push_back(std::format("missing --test-num {}", number)); }
        }
        if (!missing.empty()) { throw UsageError(join_strings(missing)); }
    }

    return result;
}

static double elapsed_us(std::chrono::high_resolution_clock::time_point start,
                         std::chrono::high_resolution_clock::time_point end) {
    std::chrono::duration<double, std::micro> us_double = end - start;
    return us_double.count();
}

static LoweringResult lower_function(const hir::Function *fn, TestVariant variant, PipelineTiming *timing = nullptr) {
    if (variant_is_scalar(variant)) {
        auto t1 = std::chrono::high_resolution_clock::now();
        asm volatile("");
        mir::Function *mir = lower_scalar(fn);
        asm volatile("");
        auto t2 = std::chrono::high_resolution_clock::now();
        if (timing != nullptr) { timing->mir_us += elapsed_us(t1, t2); }
        return LoweringResult{.mir = mir, .vect = nullptr, .used_vectorizer = false};
    }

    auto vect_t1 = std::chrono::high_resolution_clock::now();
    asm volatile("");
    vect::Function *vec = vect::hir_to_vect(fn);
    asm volatile("");
    auto vect_t2 = std::chrono::high_resolution_clock::now();
    if (timing != nullptr) { timing->vectorizer_us += elapsed_us(vect_t1, vect_t2); }

    auto mir_t1 = std::chrono::high_resolution_clock::now();
    asm volatile("");
    mir::Function *mir = mir::vect_to_mir(vec);
    asm volatile("");
    auto mir_t2 = std::chrono::high_resolution_clock::now();
    if (timing != nullptr) { timing->mir_us += elapsed_us(mir_t1, mir_t2); }
    return LoweringResult{.mir = mir, .vect = vec, .used_vectorizer = true};
}

static void print_stage_log(std::string_view stage_name, std::string_view content, const RegisteredTest &descriptor,
                            TestVariant variant, int iteration, int total_iterations) {
    if (content.empty()) { return; }

    std::string title = styled(std::string(stage_name), AnsiStyle::Accent);
    if (total_iterations > 1) {
        printf("=== %s [n=%zu %s %s] %s (iteration %d/%d) ===\n", title.c_str(), descriptor.transient_number,
               descriptor.persistent_id.c_str(), variant_name(variant), descriptor.location.c_str(), iteration,
               total_iterations);
    } else {
        printf("=== %s [n=%zu %s %s] %s ===\n", title.c_str(), descriptor.transient_number,
               descriptor.persistent_id.c_str(), variant_name(variant), descriptor.location.c_str());
    }
    printf("%.*s\n", static_cast<int>(content.size()), content.data());
}

static void append_codes_json(const mir::Function *func, std::vector<std::string> &objs, AsmjitState &state,
                              const TestRunConfig &config, const TestMetadata &meta, TestVariant variant,
                              std::string_view suffix = {}) {
#if SIMJIT_LLVM_BACKEND
    if (config.emit_llvm) {
        std::string llvm = emit_llvm_ir(func);
        objs.push_back(make_code_obj(std::format("llvm{}", suffix), escape_json(llvm),
                                     code_comparison_is_unstable(meta, EmittedCodeKind::Llvm, variant)));
    }
#endif
#if SIMJIT_ASMJIT_BACKEND
    if (config.emit_asmjit) {
        AsmjitCompileOptions opts{true, true};
        opts.session = &state;
        AsmjitCompileResult asmjit_res{};
        compile_asmjit(func, opts, asmjit_res);

        assert(!asmjit_res.machine_code.empty());
        assert(!asmjit_res.asm_code.empty());

        objs.push_back(make_code_obj(std::format("asmjit{}", suffix), make_base64(asmjit_res.machine_code),
                                     code_comparison_is_unstable(meta, EmittedCodeKind::Asmjit, variant)));
        objs.push_back(make_code_obj(std::format("asmjit_asm{}", suffix), escape_json(asmjit_res.asm_code)));
    }
#endif
#if SIMJIT_CPP_BACKEND
    if (config.emit_cpp && cpp_backend_available(variant)) {
        std::string cpp = emit_cpp_source(func);
        objs.push_back(make_code_obj(std::format("cpp{}", suffix), escape_json(cpp),
                                     code_comparison_is_unstable(meta, EmittedCodeKind::Cpp, variant)));
    }
#endif
}

static std::string build_test_json(const RegisteredTest &descriptor, const mir::Function *mir, std::string_view hir_str,
                                   std::string_view mir_str, std::string_view serialized_str, TestVariant variant,
                                   ExpectedOutcome expected, const PipelineTiming &pipeline_timing, double asmjit_time,
                                   double llvm_time, std::vector<std::string> &codes, int iteration,
                                   int total_iterations) {
    std::string obj = "{";
    obj += std::format("\"n\": {}", descriptor.transient_number);
    obj += ",";
    obj += std::format("\"id\": \"{}\"", escape_json(descriptor.persistent_id));
    obj += ",";
    obj += std::format("\"suite\": \"{}\"", escape_json(descriptor.suite_id));
    obj += ",";
    obj += std::format("\"suite_idx\": {}", descriptor.suite_index);
    obj += ",";
    obj += std::format("\"case_idx\": {}", descriptor.case_index);
    if (total_iterations > 1) {
        obj += ",";
        obj += std::format("\"iteration\": {}", iteration);
    }
    obj += ",";
    obj += std::format("\"variant\": \"{}\"", variant_name(variant));
    obj += ",";
    obj += std::format("\"expected\": \"{}\"", expected_outcome_name(expected));
    obj += ",";
    obj += std::format("\"file\": \"{}\"", escape_json(descriptor.test->file));
    obj += ",";
    obj += std::format("\"line\": {}", descriptor.test->line);
    obj += ",";
    obj += std::format("\"schema\": {}", schema_json(mir->args));
    obj += ",";
    obj += std::format("\"src\": \"{}\"", escape_json(hir_str));
    obj += ",";
    obj += std::format("\"mir\": \"{}\"", escape_json(mir_str));
    obj += ",";
    obj += std::format("\"serialized\": \"{}\"", escape_json(serialized_str));
    obj += ",";
    obj += std::format("\"hir_time\": {}", pipeline_timing.hir_us);
    obj += ",";
    obj += std::format("\"vectorizer_time\": {}", pipeline_timing.vectorizer_us);
    obj += ",";
    obj += std::format("\"mir_time\": {}", pipeline_timing.mir_us);
    obj += ",";
    obj += std::format("\"asmjit_time\": {}", asmjit_time);
    obj += ",";
    obj += std::format("\"llvm_time\": {}", llvm_time);
    obj += ",";
    obj += "\"codes\": [";
    for (size_t i = 0; i < codes.size(); ++i) {
        obj += codes[i];
        if (i + 1 != codes.size()) { obj += ","; }
    }
    obj += "]}";
    return obj;
}

static std::string build_error_json(const RegisteredTest &descriptor, TestVariant variant, ExpectedOutcome expected,
                                    const ErrorInfo &error, std::string_view message, int iteration,
                                    int total_iterations) {
    std::string obj = "{";
    obj += std::format("\"n\": {}", descriptor.transient_number);
    obj += std::format(",\"id\": \"{}\"", escape_json(descriptor.persistent_id));
    obj += std::format(",\"suite\": \"{}\"", escape_json(descriptor.suite_id));
    obj += std::format(",\"suite_idx\": {}", descriptor.suite_index);
    obj += std::format(",\"case_idx\": {}", descriptor.case_index);
    if (total_iterations > 1) { obj += std::format(",\"iteration\": {}", iteration); }
    obj += std::format(",\"variant\": \"{}\"", variant_name(variant));
    obj += std::format(",\"expected\": \"{}\"", expected_outcome_name(expected));
    obj += std::format(",\"file\": \"{}\"", escape_json(descriptor.test->file));
    obj += std::format(",\"line\": {}", descriptor.test->line);
    obj += std::format(",\"error_module\": \"{}\"", show_error_module(error.module));
    obj += std::format(",\"error_kind\": \"{}\"", show_error_kind(error.kind));
    obj += std::format(",\"error_subkind\": \"{}\"", show_error_subkind(error.subkind));
    obj += std::format(",\"error_message\": \"{}\"", escape_json(message));
    obj += "}";
    return obj;
}

static RunResult run_test(const RegisteredTest &descriptor, const TestRunConfig &config, AsmjitState &state,
                          int iteration) {
    RunResult result;

    VariantResolution resolution = resolve_variant(config, *descriptor.test, false);
    if (!resolution.runnable()) {
        result.skipped = true;
        return result;
    }

    bool has_exception = false;
    std::string exception_str;
    std::string hir_str;
    std::string mir_str;
    std::string serialized_str;
    ErrorInfo observed_error{};
    PipelineTiming pipeline_timing;
    double llvm_time = 0;
    double asmjit_time = 0;
    bool used_vectorizer = false;
    ExpectedOutcome expected = expected_outcome(*descriptor.test, resolution.variant);
    bool checked = false;

    simjit::MemoryArena arena{};
    simjit::Context opts{arena, "expr"};
    opts.arch = resolve_arch(config.arch);

    simjit::FunctionBuilder builder{opts};
    try {
        auto hir_t1 = std::chrono::high_resolution_clock::now();
        asm volatile("");
        descriptor.test->builder(builder);
        hir::Function *fn = builder.build();
        asm volatile("");
        auto hir_t2 = std::chrono::high_resolution_clock::now();
        pipeline_timing.hir_us = elapsed_us(hir_t1, hir_t2);

        resolution = resolve_variant(config, *descriptor.test, fn->scalar_only);
        if (!resolution.runnable()) {
            result.skipped = true;
            return result;
        }
        expected = expected_outcome(*descriptor.test, resolution.variant);

        hir_str = simjit::hir::print_function(fn);
#if SIMJIT_ENABLE_SERIALIZATION
        serialized_str = simjit::serialize(fn);
#endif

        LoweringResult lowered = lower_function(fn, resolution.variant, &pipeline_timing);
        mir::Function *mir = lowered.mir;
        used_vectorizer = lowered.used_vectorizer;

        if (config.log_stages.contains(LogStage::Hir)) {
            print_stage_log("HIR", hir_str, descriptor, resolution.variant, iteration, config.iterations);
        }
        if (config.log_stages.contains(LogStage::Vectorizer) && lowered.used_vectorizer) {
            print_stage_log("Vectorizer", vect::print_function(lowered.vect), descriptor, resolution.variant, iteration,
                            config.iterations);
        }

        mir_str = mir::print_function(mir);
        if (config.log_stages.contains(LogStage::Mir)) {
            print_stage_log("MIR", mir_str, descriptor, resolution.variant, iteration, config.iterations);
        }

        [[maybe_unused]] bool emit_cpp_for_variant = config.emit_cpp && cpp_backend_available(resolution.variant);

#if SIMJIT_CPP_BACKEND
        if (emit_cpp_for_variant) {
            try {
                emit_cpp_source(mir);
            } catch (...) {
                if (!code_comparison_is_unstable(descriptor.test->meta, EmittedCodeKind::Cpp, resolution.variant)) {
                    throw;
                }
            }
            checked = true;
        }
#endif
#if SIMJIT_LLVM_BACKEND
        if (config.emit_llvm) {
            auto t1 = std::chrono::high_resolution_clock::now();
            asm volatile("");
            emit_llvm_ir(mir);
            asm volatile("");
            auto t2 = std::chrono::high_resolution_clock::now();
            llvm_time = pipeline_timing.total_us() + elapsed_us(t1, t2);
            checked = true;
        }
#endif
#if SIMJIT_ASMJIT_BACKEND
        if (config.emit_asmjit) {
            auto t1 = std::chrono::high_resolution_clock::now();
            asm volatile("");
            AsmjitCompileOptions aj_opts{};
            aj_opts.session = &state;
            AsmjitCompileResult compile_result{};
            compile_asmjit(mir, aj_opts, compile_result);
            asm volatile("");
            auto t2 = std::chrono::high_resolution_clock::now();
            asmjit_time = pipeline_timing.total_us() + elapsed_us(t1, t2);
            checked = true;
        }
#endif

        bool should_check_serialization = config.dump_json || config.validate_serialization;
        std::vector<std::string> codes;
        if (config.dump_json) {
            append_codes_json(mir, codes, state, config, descriptor.test->meta, resolution.variant);

            if (!variant_is_scalar(resolution.variant)) {
                mir::Function *scalar_mir = lower_scalar(fn);
                append_codes_json(scalar_mir, codes, state, config, descriptor.test->meta,
                                  scalar_variant_for(resolution.variant), "_s");
            }
        }

        if (should_check_serialization && !serialized_str.empty()) {
#if SIMJIT_ENABLE_SERIALIZATION
            try {
                FunctionBuilder other_builder{opts};
                simjit::deserialize(serialized_str, other_builder);
                hir::Function *other_fn = other_builder.build();
                VariantResolution other_resolution = resolve_variant(config, *descriptor.test, other_fn->scalar_only);
                if (!other_resolution.runnable()) {
                    throw std::runtime_error("deserialized test variant is unrunnable");
                }
                LoweringResult other_lowered = lower_function(other_fn, other_resolution.variant);
                [[maybe_unused]] bool emit_cpp_for_other_variant =
                    config.emit_cpp && cpp_backend_available(other_resolution.variant);

#if SIMJIT_CPP_BACKEND
                if (emit_cpp_for_other_variant && !config.dump_json) {
                    try {
                        emit_cpp_source(other_lowered.mir);
                    } catch (...) {
                        if (!code_comparison_is_unstable(descriptor.test->meta, EmittedCodeKind::Cpp,
                                                         other_resolution.variant)) {
                            throw;
                        }
                    }
                }
#endif
#if SIMJIT_LLVM_BACKEND
                if (config.emit_llvm && !config.dump_json) { emit_llvm_ir(other_lowered.mir); }
#endif
#if SIMJIT_ASMJIT_BACKEND
                if (config.emit_asmjit && !config.dump_json) {
                    AsmjitCompileOptions other_aj_opts{};
                    other_aj_opts.session = &state;
                    AsmjitCompileResult other_compile_result{};
                    compile_asmjit(other_lowered.mir, other_aj_opts, other_compile_result);
                }
#endif
                if (config.validate_serialization) { checked = true; }
            } catch (std::exception &e) {
                if (config.validate_serialization) {
                    has_exception = true;
                    exception_str = e.what();
                    observed_error = ErrorInfo{ErrorModule::Serialization, ErrorKind::SerializationFailure,
                                               ErrorSubKind::SerializationParseError, exception_str};
                } else {
                    std::string label = styled("failed to deserialize test", AnsiStyle::Warning);
                    printf("%s [n=%zu %s %s] %s: %s\n", label.c_str(), descriptor.transient_number,
                           descriptor.persistent_id.c_str(), variant_name(resolution.variant),
                           descriptor.location.c_str(), e.what());
                }
            }
#endif
        }

        if (config.dump_json) {
            if (!descriptor.test->python.empty()) {
                codes.push_back(make_code_obj("py", escape_json(descriptor.test->python)));
            }

            result.json =
                build_test_json(descriptor, mir, hir_str, mir_str, serialized_str, resolution.variant, expected,
                                pipeline_timing, asmjit_time, llvm_time, codes, iteration, config.iterations);
        }
    } catch (const SimjitException &e) {
        has_exception = true;
        exception_str = e.what();
        observed_error = e.info();
    } catch (std::exception &e) {
        has_exception = true;
        exception_str = e.what();
        observed_error =
            ErrorInfo{ErrorModule::Generic, ErrorKind::InternalInvariant, ErrorSubKind::ExternalFailure, exception_str};
    }

    TestErrorInfoExpectation expected_error = expected_structured_error(descriptor.test->meta, resolution.variant);
    bool expected_error_source_checked =
        expected_structured_error_source_checked(config, resolution.variant, expected_error);

    if (descriptor.test->expectation.has_error) {
        if (!has_exception || observed_error.kind != descriptor.test->expectation.kind) {
            std::string label = styled(has_exception ? "unexpected error kind" : "unexpected pass", AnsiStyle::Error);
            printf("%s [n=%zu %s %s] %s (expected %s observed_kind=%u): %s\n", label.c_str(),
                   descriptor.transient_number, descriptor.persistent_id.c_str(), variant_name(resolution.variant),
                   descriptor.location.c_str(), test_expectation_name(descriptor.test->expectation),
                   (unsigned)observed_error.kind, has_exception ? exception_str.c_str() : "");
            result.failed = true;
        } else {
            result.matched_expected_error = true;
        }
    } else if (expected == ExpectedOutcome::Pass && has_exception) {
        std::string label = styled("unexpected failure", AnsiStyle::Error);
        printf("%s [n=%zu %s %s] %s (expected pass, %s): %s\n", label.c_str(), descriptor.transient_number,
               descriptor.persistent_id.c_str(), variant_name(resolution.variant), descriptor.location.c_str(),
               test_metadata_str(descriptor.test->meta).c_str(), exception_str.c_str());
        if (config.log_failures && !hir_str.empty()) { printf("%s\n", hir_str.c_str()); }
        result.failed = true;
    } else if (expected != ExpectedOutcome::Pass && !has_exception && checked && expected_error_source_checked) {
        std::string label = styled("unexpected pass", AnsiStyle::Warning);
        printf("%s [n=%zu %s %s] %s (tagged as %s, %s): %s\n", label.c_str(), descriptor.transient_number,
               descriptor.persistent_id.c_str(), variant_name(resolution.variant), descriptor.location.c_str(),
               expected_outcome_name(expected), test_metadata_str(descriptor.test->meta).c_str(), hir_str.c_str());
        result.failed = true;
    }

    ErrorSubKind expected_vect_failure = expected_vectorization_failure(descriptor.test->meta, resolution.variant);
    if (!result.failed && expected_vect_failure != ErrorSubKind::None) {
        if (!has_exception || observed_error.module != ErrorModule::Vectorizer ||
            observed_error.kind != ErrorKind::VectorizationFailed || observed_error.subkind != expected_vect_failure) {
            std::string label = styled("unexpected vectorization failure metadata", AnsiStyle::Error);
            printf("%s [n=%zu %s %s] %s expected_kind=%u observed_module=%u observed_kind=%u observed_subkind=%u: %s\n",
                   label.c_str(), descriptor.transient_number, descriptor.persistent_id.c_str(),
                   variant_name(resolution.variant), descriptor.location.c_str(), (unsigned)expected_vect_failure,
                   (unsigned)observed_error.module, (unsigned)observed_error.kind, (unsigned)observed_error.subkind,
                   has_exception ? exception_str.c_str() : "");
            result.failed = true;
        } else {
            result.matched_expected_error = true;
        }
    } else if (!result.failed && expected != ExpectedOutcome::Pass && has_exception &&
               observed_error.module == ErrorModule::Vectorizer &&
               observed_error.kind == ErrorKind::VectorizationFailed) {
        std::string label = styled("missing vectorization failure metadata", AnsiStyle::Error);
        printf("%s [n=%zu %s %s] %s observed_subkind=%u: %s\n", label.c_str(), descriptor.transient_number,
               descriptor.persistent_id.c_str(), variant_name(resolution.variant), descriptor.location.c_str(),
               (unsigned)observed_error.subkind, exception_str.c_str());
        result.failed = true;
    }

    if (!result.failed && expected_error.has_error && expected_error_source_checked) {
        if (!has_exception || !structured_error_matches_expected(config, expected_error, observed_error)) {
            std::string label = styled("unexpected structured error metadata", AnsiStyle::Error);
            printf("%s [n=%zu %s %s] %s expected_module=%u expected_kind=%u expected_subkind=%u "
                   "observed_module=%u observed_kind=%u observed_subkind=%u: %s\n",
                   label.c_str(), descriptor.transient_number, descriptor.persistent_id.c_str(),
                   variant_name(resolution.variant), descriptor.location.c_str(), (unsigned)expected_error.module,
                   (unsigned)expected_error.kind, (unsigned)expected_error.subkind, (unsigned)observed_error.module,
                   (unsigned)observed_error.kind, (unsigned)observed_error.subkind,
                   has_exception ? exception_str.c_str() : "");
            result.failed = true;
        } else {
            result.matched_expected_error = true;
        }
    } else if (!result.failed && expected_error.has_error && has_exception && !expected_error_source_checked) {
        if (!structured_error_matches_expected(config, expected_error, observed_error)) {
            std::string label = styled("unexpected structured error metadata", AnsiStyle::Error);
            printf("%s [n=%zu %s %s] %s expected_kind=%u expected_subkind=%u "
                   "observed_module=%u observed_kind=%u observed_subkind=%u: %s\n",
                   label.c_str(), descriptor.transient_number, descriptor.persistent_id.c_str(),
                   variant_name(resolution.variant), descriptor.location.c_str(), (unsigned)expected_error.kind,
                   (unsigned)expected_error.subkind, (unsigned)observed_error.module, (unsigned)observed_error.kind,
                   (unsigned)observed_error.subkind, exception_str.c_str());
            result.failed = true;
        } else {
            result.matched_expected_error = true;
        }
    } else if (!result.failed && expected != ExpectedOutcome::Pass && has_exception &&
               (observed_error.module != ErrorModule::Vectorizer ||
                observed_error.kind != ErrorKind::VectorizationFailed)) {
        std::string label = styled("missing structured error metadata", AnsiStyle::Error);
        printf("%s [n=%zu %s %s] %s observed_module=%u observed_kind=%u observed_subkind=%u: %s\n", label.c_str(),
               descriptor.transient_number, descriptor.persistent_id.c_str(), variant_name(resolution.variant),
               descriptor.location.c_str(), (unsigned)observed_error.module, (unsigned)observed_error.kind,
               (unsigned)observed_error.subkind, exception_str.c_str());
        result.failed = true;
    }

    if (!result.failed && !has_exception && checked) { result.compiled = true; }

    if (config.dump_json && result.json.empty() && has_exception) {
        result.json = build_error_json(descriptor, resolution.variant, expected, observed_error, exception_str,
                                       iteration, config.iterations);
    }

    result.arena_sampled = !result.skipped;
    result.arena_used = arena.total_bytes_used();
    result.arena_reserved = arena.total_bytes_allocated();
    result.pipeline_timing = pipeline_timing;
    result.llvm_time = llvm_time;
    result.asmjit_time = asmjit_time;
    result.used_vectorizer = used_vectorizer;
    return result;
}

static void write_json_dump(const TestRunConfig &config, const std::vector<std::string> &test_jsons) {
    std::ofstream out(config.json_path, std::ios::binary);
    if (!out) { throw std::runtime_error(std::format("failed to open JSON output path: {}", config.json_path)); }
    for (const auto &test_json : test_jsons) {
        if (test_json.empty()) { continue; }
        out << test_json << '\n';
    }
}

static void print_timing_line(const char *name, double total_us, size_t count) {
    double avg_us = count == 0 ? 0.0 : total_us / static_cast<double>(count);
    printf("    %s: total_us=%.0f avg_us=%.1f samples=%zu\n", name, total_us, avg_us, count);
}

static void print_timing_summary(const TestRunConfig &config, const TimingSummary &timing) {
    if (!config.emit_cpp && !config.emit_llvm && !config.emit_asmjit && timing.hir_count == 0) { return; }

    std::string title = styled("compile timing:", AnsiStyle::Bold);
    printf("  %s\n", title.c_str());
    print_timing_line("hir", timing.hir_total_us, timing.hir_count);
    print_timing_line("vectorizer", timing.vectorizer_total_us, timing.vectorizer_count);
    print_timing_line("mir", timing.mir_total_us, timing.mir_count);
    if (config.emit_cpp) { printf("    cpp: not timed\n"); }
    if (config.emit_llvm) { print_timing_line("llvm_e2e", timing.llvm_total_us, timing.llvm_count); }
    if (config.emit_asmjit) { print_timing_line("asmjit_e2e", timing.asmjit_total_us, timing.asmjit_count); }
}

static void record_arena_sample(ArenaSizeSummary &summary, size_t used, size_t reserved) {
    ++summary.count;
    summary.min_used = std::min(summary.min_used, used);
    summary.max_used = std::max(summary.max_used, used);
    summary.total_used += used;
    summary.min_reserved = std::min(summary.min_reserved, reserved);
    summary.max_reserved = std::max(summary.max_reserved, reserved);
    summary.total_reserved += reserved;
}

static void merge_arena_summary(ArenaSizeSummary &dst, const ArenaSizeSummary &src) {
    if (src.count == 0) { return; }
    dst.count += src.count;
    dst.min_used = std::min(dst.min_used, src.min_used);
    dst.max_used = std::max(dst.max_used, src.max_used);
    dst.total_used += src.total_used;
    dst.min_reserved = std::min(dst.min_reserved, src.min_reserved);
    dst.max_reserved = std::max(dst.max_reserved, src.max_reserved);
    dst.total_reserved += src.total_reserved;
}

static void print_arena_summary(const ArenaSizeSummary &summary) {
    if (summary.count == 0) { return; }
    double avg_used = static_cast<double>(summary.total_used) / static_cast<double>(summary.count);
    double avg_reserved = static_cast<double>(summary.total_reserved) / static_cast<double>(summary.count);
    printf("  arena size:\n");
    printf("    used: min=%zu max=%zu avg=%.1f samples=%zu\n", summary.min_used, summary.max_used, avg_used,
           summary.count);
    printf("    reserved: min=%zu max=%zu avg=%.1f samples=%zu\n", summary.min_reserved, summary.max_reserved,
           avg_reserved, summary.count);
}

static void print_summary(const TestRunConfig &config, size_t registered_count, size_t executed_count, size_t skipped,
                          size_t compiled_count, size_t matched_expected_error_count, int failures,
                          const TimingSummary &timing, const ArenaSizeSummary &arena) {
    std::string title = styled("Summary:", AnsiStyle::Heading);
    std::string failure_line = failures == 0 ? styled(std::to_string(failures), AnsiStyle::Success)
                                             : styled(std::to_string(failures), AnsiStyle::Error);
    printf("%s\n", title.c_str());
    printf("  suites: %s\n", join_strings(config.selected_suites).c_str());
    printf("  arch: %s\n", arch_name(config.arch).c_str());
    printf("  mode: %s\n", mode_name(config.vector_mode).c_str());
    printf("  emits: %s\n", join_strings(emit_names(config)).c_str());
    printf("  validate serialization: %s\n", config.validate_serialization ? "yes" : "no");
    printf("  log stages: %s\n", join_strings(log_stage_names(config)).c_str());
    printf("  iterations: %d\n", config.iterations);
    printf("  registered tests: %zu\n", registered_count);
    printf("  executed tests: %zu\n", executed_count);
    printf("  compiled tests: %zu\n", compiled_count);
    printf("  expected error tests: %zu\n", matched_expected_error_count);
    printf("  skipped tests: %zu\n", skipped);
    printf("  failures: %s\n", failure_line.c_str());
    print_timing_summary(config, timing);
    if (config.arena_stats) { print_arena_summary(arena); }
    if (config.dump_json) { printf("  json output: %s\n", config.json_path.c_str()); }
}

static ConfigRunSummary execute_run_config(const TestRunConfig &config,
                                           const std::vector<RegisteredTest> &selected_tests,
                                           size_t registered_in_selected_suites, std::vector<std::string> &test_jsons) {
#if SIMJIT_ASMJIT_BACKEND
    AsmjitState state{resolve_arch(config.arch)};
#else
    AsmjitState state{};
#endif

    ConfigRunSummary summary;
    for (int iteration = 1; iteration <= config.iterations; ++iteration) {
        for (const auto &test : selected_tests) {
            RunResult result = run_test(test, config, state, iteration);
            if (result.skipped) {
                ++summary.skipped;
                continue;
            }
            ++summary.executed;
            if (!result.json.empty()) { test_jsons.push_back(std::move(result.json)); }
            if (result.failed) { ++summary.failures; }
            if (result.compiled) {
                ++summary.compiled;
                summary.timing.hir_total_us += result.pipeline_timing.hir_us;
                ++summary.timing.hir_count;
                if (result.used_vectorizer) {
                    summary.timing.vectorizer_total_us += result.pipeline_timing.vectorizer_us;
                    ++summary.timing.vectorizer_count;
                }
                summary.timing.mir_total_us += result.pipeline_timing.mir_us;
                ++summary.timing.mir_count;
            }
            if (result.matched_expected_error) { ++summary.matched_expected_error; }
            if (result.arena_sampled) { record_arena_sample(summary.arena, result.arena_used, result.arena_reserved); }
            if (config.emit_llvm && result.compiled) {
                summary.timing.llvm_total_us += result.llvm_time;
                ++summary.timing.llvm_count;
            }
            if (config.emit_asmjit && result.compiled) {
                summary.timing.asmjit_total_us += result.asmjit_time;
                ++summary.timing.asmjit_count;
            }
        }
    }

    print_summary(config, registered_in_selected_suites, summary.executed, summary.skipped, summary.compiled,
                  summary.matched_expected_error, summary.failures, summary.timing, summary.arena);
    return summary;
}

int main(int argc, char **argv) {
    g_use_ansi = detect_ansi_output();
    try {
        TestRunConfig config = parse_config(argc, argv);
        run_public_api_smoke_tests();
        std::vector<RegisteredTest> all_tests = register_tests();

        size_t registered_in_selected_suites = 0;
        std::vector<RegisteredTest> selected_tests = select_tests(config, all_tests, registered_in_selected_suites);
        std::vector<TestRunConfig> run_configs = expand_run_configs(config);

        std::vector<std::string> test_jsons;
        test_jsons.reserve(selected_tests.size() * static_cast<size_t>(config.iterations) * run_configs.size());

        int failures = 0;
        size_t executed = 0;
        size_t skipped = 0;
        size_t compiled = 0;
        size_t matched_expected_error = 0;
        ArenaSizeSummary arena;
        for (size_t i = 0; i < run_configs.size(); ++i) {
            if (run_configs.size() > 1) {
                std::string title = styled("Running", AnsiStyle::Heading);
                printf("=== %s arch=%s mode=%s ===\n", title.c_str(), arch_name(run_configs[i].arch).c_str(),
                       mode_name(run_configs[i].vector_mode).c_str());
            }
            ConfigRunSummary summary =
                execute_run_config(run_configs[i], selected_tests, registered_in_selected_suites, test_jsons);
            failures += summary.failures;
            executed += summary.executed;
            skipped += summary.skipped;
            compiled += summary.compiled;
            matched_expected_error += summary.matched_expected_error;
            merge_arena_summary(arena, summary.arena);
        }

        if (config.dump_json) { write_json_dump(config, test_jsons); }
        if (run_configs.size() > 1) {
            std::string title = styled("Overall Summary:", AnsiStyle::Heading);
            std::string failure_line = failures == 0 ? styled(std::to_string(failures), AnsiStyle::Success)
                                                     : styled(std::to_string(failures), AnsiStyle::Error);
            printf("%s\n", title.c_str());
            printf("  suites: %s\n", join_strings(config.selected_suites).c_str());
            printf("  arch: %s\n", arch_name(config.arch).c_str());
            printf("  mode: %s\n", mode_name(config.vector_mode).c_str());
            printf("  validate serialization: %s\n", config.validate_serialization ? "yes" : "no");
            printf("  iterations: %d\n", config.iterations);
            printf("  registered tests: %zu\n", registered_in_selected_suites);
            printf("  executed tests: %zu\n", executed);
            printf("  compiled tests: %zu\n", compiled);
            printf("  expected error tests: %zu\n", matched_expected_error);
            printf("  skipped tests: %zu\n", skipped);
            printf("  failures: %s\n", failure_line.c_str());
            if (config.arena_stats) { print_arena_summary(arena); }
            if (config.dump_json) { printf("  json output: %s\n", config.json_path.c_str()); }
        }
        return failures == 0 ? 0 : 1;
    } catch (const UsageError &e) {
        std::string label = styled("error:", AnsiStyle::Error);
        fprintf(stderr, "%s %s\n", label.c_str(), e.what());
        fprintf(stderr, "Use --help for usage.\n");
        return 1;
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
