// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "simjit/simjit.h"

namespace simjit::local_runner {

// model.cpp

enum class ArgumentKind : uint8_t {
    Input,
    Output,
    OutputScalar,
    SafetyCheck,
    Sequence
};
enum class Backend : uint8_t {
    Cpp,
    Llvm,
    Asmjit
};

struct ArgumentInfo {
    ScalarDataType dtype{};
    ArgumentKind kind{};
};

struct FunctionHandle {
    void *address = nullptr;
    std::shared_ptr<void> owner{};
};

struct Implementation {
    Backend backend{};
    std::string bundle_name;
    std::string code;
    std::string symbol;
    bool comparison_unstable = false;
    unsigned optimization_level = 1;
    FunctionHandle function{};
};

struct CaseResult {
    bool complete = false;
    bool failed = false;
    bool cancelled = false;
    bool comparison_skipped = false;
    std::string message;

    static constexpr CaseResult make_cancelled() {
        CaseResult x{};
        x.cancelled = true;
        x.complete = true;
        return x;
    }
};

struct BundleCase {
    size_t input_index = 0;
    size_t line_number = 0;
    int64_t number = -1;
    int64_t iteration = 0;
    std::string id;
    std::string suite;
    std::string variant;
    std::string file;
    int64_t source_line = 0;
    bool structured_error = false;
    std::string error_label;
    size_t skipped_python = 0;
    std::vector<ArgumentInfo> args;
    std::vector<Implementation> implementations;

    std::atomic<size_t> pending_compilations{0};
    std::atomic<bool> compilation_failed{false};
    mutable std::mutex mutex;
    CaseResult result{};

    std::string label() const;
};

using BundleCasePtr = std::shared_ptr<BundleCase>;

// bundle.cpp

struct RunnerOptions {
    enum class BenchmarkO3 : uint8_t {
        None,
        Scalar,
        All
    };

    std::string file = "tests.jsonl";
    int64_t first = 0;
    std::set<std::string> suites;
    size_t limit = 0;
    std::optional<int64_t> only_n;
    std::set<std::string> code_names;
    bool verbose_item = false;
    bool bench = false;
    BenchmarkO3 benchmark_o3 = BenchmarkO3::Scalar;
    bool timings = false;
    size_t workers = 0;
    uint64_t seed = 0x53494d4a4954ULL;
    size_t rows = 2048;
    std::vector<std::string> benchmark_arguments;
};

RunnerOptions parse_options(int argc, char **argv);
BundleCasePtr parse_bundle_line(std::string_view line, size_t line_number, size_t input_index);
bool case_selected(const BundleCase &item, const RunnerOptions &options);

// compiler.cpp

struct CompileRef {
    BundleCasePtr item;
    size_t implementation_index = 0;
    size_t weight_bytes = 0;
};

enum class CompilationPath : uint8_t {
    CppO1,
    CppO3,
    LlvmIrO1,
    LlvmIrO3,
    Count
};

struct CompilationTiming {
    size_t functions = 0;
    std::chrono::nanoseconds wall_time{};
    std::chrono::nanoseconds cpu_time{};
};

using CompilationTimings = std::array<CompilationTiming, static_cast<size_t>(CompilationPath::Count)>;

class CompilerPipeline {
public:
    CompilerPipeline();
    ~CompilerPipeline();

    CompilerPipeline(const CompilerPipeline &) = delete;
    CompilerPipeline &operator=(const CompilerPipeline &) = delete;

    void compile_clang_group(const std::vector<CompileRef> &group, unsigned optimization_level = 1);
    void compile_llvm_group(const std::vector<CompileRef> &group, unsigned optimization_level = 1);
    void materialize_asmjit_group(const std::vector<CompileRef> &group);
    CompilationTimings timings() const;

private:
    struct LLVMSessionPool;
    struct AsmjitRuntime;
    struct AsmjitPool;
    std::shared_ptr<LLVMSessionPool> llvm_;
    std::shared_ptr<AsmjitPool> asmjit_;
    mutable std::mutex timings_mutex_;
    CompilationTimings timings_{};

    void record_timing(CompilationPath path, size_t functions, std::chrono::nanoseconds wall_time,
                       std::chrono::nanoseconds cpu_time);
};

// planner.cpp

std::vector<std::vector<CompileRef>> balance_compile_groups(std::vector<CompileRef> refs, size_t target_bytes,
                                                            size_t minimum_groups);

// runtime.cpp

CaseResult execute_case(const BundleCase &item, const RunnerOptions &options);
class BenchmarkInvocation;
using BenchmarkInvocationPtr = std::shared_ptr<BenchmarkInvocation>;
BenchmarkInvocationPtr prepare_benchmark(const BundleCase &item, size_t implementation_index,
                                         const RunnerOptions &options);
size_t run_registered_benchmarks(const std::vector<BenchmarkInvocationPtr> &invocations,
                                 const std::vector<std::string> &arguments);

} // namespace simjit::local_runner
