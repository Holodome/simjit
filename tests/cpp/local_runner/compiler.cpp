// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "local_runner.h"

#include "simjit/asmjit.h"
#include "simjit/core/llvm/emitter.h"

#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/VirtualFileSystem.h"

#include <ctime>
#include <format>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>

#ifndef SIMJIT_LOCAL_RUNNER_CLANGXX
#define SIMJIT_LOCAL_RUNNER_CLANGXX "clang++"
#endif

namespace simjit::local_runner {

std::string error_string(llvm::Error error) {
    return llvm::toString(std::move(error));
}

namespace {

std::chrono::nanoseconds thread_cpu_time() {
#if defined(CLOCK_THREAD_CPUTIME_ID)
    timespec value{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0) {
        return std::chrono::seconds(value.tv_sec) + std::chrono::nanoseconds(value.tv_nsec);
    }
#endif
    return std::chrono::nanoseconds(static_cast<int64_t>(static_cast<long double>(std::clock()) * 1000000000.0L /
                                                         static_cast<long double>(CLOCKS_PER_SEC)));
}

void assign_functions(const std::vector<CompileRef> &group, const std::shared_ptr<llvm_backend::LLVMSession> &session) {
    for (const CompileRef &ref : group) {
        auto &impl = ref.item->implementations[ref.implementation_index];
        impl.function = FunctionHandle{session->lookup(impl.symbol), session};
    }
}

std::string cpp_prologue() {
    std::string result;
#if defined(__x86_64__) || defined(_M_X64)
    result += "#include <immintrin.h>\n";
#else
    result += "#include <arm_neon.h>\n";
#endif
    result += R"CPP(#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <bit>
#include <math.h>
#include <float.h>
#define bit_cast std::bit_cast
)CPP";
    return result;
}

std::string clang_source(const std::vector<CompileRef> &group) {
    std::string source = cpp_prologue();
    size_t reserve = source.size() + 16;
    for (const auto &ref : group)
        reserve += ref.item->implementations[ref.implementation_index].code.size() + 128;
    source.reserve(reserve);
    for (const CompileRef &ref : group) {
        const auto &item = *ref.item;
        const auto &impl = item.implementations[ref.implementation_index];
        std::string code = impl.code;
        constexpr std::string_view declaration = "void expr(";
        size_t position = code.find(declaration);
        if (position == std::string::npos || code.find(declaration, position + 1) != std::string::npos) {
            throw std::runtime_error(
                std::format("{} {} does not contain exactly one C++ expr declaration", item.label(), impl.bundle_name));
        }
        code.replace(position, declaration.size(), std::format("extern \"C\" void {}(", impl.symbol));
        source += std::format("\n#line 1 \"bundle-line-{}-{}\"\n", item.line_number, impl.bundle_name);
        source += std::format("namespace simjit_case_{}_{} {{\n", item.input_index, ref.implementation_index);
        source += code;
        source += "\n}\n";
    }
    return source;
}

struct ClangResult {
    std::unique_ptr<llvm::Module> module;
    std::unique_ptr<llvm::LLVMContext> context;
};

class CapturingEmitAction final : public clang::EmitLLVMOnlyAction {
public:
    explicit CapturingEmitAction(ClangResult &result) : result_(result) {}

protected:
    void EndSourceFileAction() override {
        clang::EmitLLVMOnlyAction::EndSourceFileAction();
        result_.module = takeModule();
        result_.context.reset(takeLLVMContext());
    }

private:
    ClangResult &result_;
};

class DiagnosticCollector final : public clang::DiagnosticConsumer {
public:
    explicit DiagnosticCollector(std::string &output) : output_(output) {}

    void HandleDiagnostic(clang::DiagnosticsEngine::Level level, const clang::Diagnostic &info) override {
        if (level < clang::DiagnosticsEngine::Warning) return;
        llvm::SmallString<256> message;
        info.FormatDiagnostic(message);
        if (!output_.empty()) output_ += '\n';
        output_.append(message.begin(), message.end());
    }

private:
    std::string &output_;
};

llvm_backend::LLVMModuleOwner compile_cpp_source(const std::string &source, unsigned optimization_level) {
    // Let CodeGenAction create the context and transfer both objects out after
    // compilation. Passing an externally-owned context here is unsafe with
    // some libclang-cpp builds because CompilerInstance teardown can retain
    // ownership state associated with the action.
    ClangResult result;
    auto action = std::make_unique<CapturingEmitAction>(result);

    auto memory_fs = llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();
    std::string input_path = "/__simjit_local_runner__/group.cpp";
    memory_fs->addFile(input_path, 0, llvm::MemoryBuffer::getMemBufferCopy(source, input_path));
    auto overlay = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(llvm::vfs::getRealFileSystem());
    overlay->pushOverlay(memory_fs);
    clang::FileSystemOptions fs_options;
    auto files = llvm::IntrusiveRefCntPtr<clang::FileManager>(new clang::FileManager(fs_options, overlay));

    std::vector<std::string> arguments{SIMJIT_LOCAL_RUNNER_CLANGXX,
                                       "-std=c++20",
                                       std::format("-O{}", optimization_level >= 3 ? 3 : 1),
                                       "-fwrapv",
                                       "-march=native",
                                       "-c",
                                       input_path};
    clang::tooling::ToolInvocation invocation(std::move(arguments), std::move(action), files.get());
    std::string diagnostics;
    // CompilerInstance takes ownership of this consumer. Its output storage
    // remains valid for the duration of the synchronous invocation.
    invocation.setDiagnosticConsumer(new DiagnosticCollector(diagnostics));
    if (!invocation.run()) {
        throw std::runtime_error(diagnostics.empty() ? "in-process Clang compilation failed" : diagnostics);
    }
    if (!result.module) {
        throw std::runtime_error(diagnostics.empty() ? "in-process Clang compilation produced no LLVM module"
                                                     : diagnostics);
    }
    if (!result.context) { throw std::runtime_error("in-process Clang compilation produced no LLVM context"); }
    return {std::move(result.context), std::move(result.module)};
}

} // namespace

struct CompilerPipeline::LLVMSessionPool {
    struct WorkerSessions {
        std::shared_ptr<llvm_backend::LLVMSession> o1;
        std::shared_ptr<llvm_backend::LLVMSession> o3;
    };

    std::shared_ptr<llvm_backend::LLVMSession> current_worker(unsigned optimization_level) {
        std::lock_guard lock(mutex);
        auto &sessions = workers[std::this_thread::get_id()];
        auto &session = optimization_level >= 3 ? sessions.o3 : sessions.o1;
        if (!session) {
            auto level = optimization_level >= 3 ? llvm_backend::LLVMOptLevel::O3 : llvm_backend::LLVMOptLevel::O1;
            session = std::make_shared<llvm_backend::LLVMSession>(Arch::Native, level);
        }
        return session;
    }

    std::mutex mutex;
    std::unordered_map<std::thread::id, WorkerSessions> workers;
};

struct CompilerPipeline::AsmjitRuntime {
    AsmjitRuntime() = default;
    ~AsmjitRuntime() = default;

    void *add(std::span<const char> bytes) {
        session.reset();
        auto &compiler = session.compiler();
        if (auto error = compiler.embed(bytes.data(), bytes.size()); error != asmjit::Error::kOk) {
            throw std::runtime_error(
                std::format("AsmJit embed failed: {}", asmjit::DebugUtils::error_as_string(error)));
        }
        if (auto error = compiler.finalize(); error != asmjit::Error::kOk) {
            throw std::runtime_error(
                std::format("AsmJit finalize failed: {}", asmjit::DebugUtils::error_as_string(error)));
        }
        return session.add_compiled_function();
    }

    simjit::AsmjitSession session;
};

struct CompilerPipeline::AsmjitPool {
    std::shared_ptr<AsmjitRuntime> current_worker() {
        std::lock_guard lock(mutex);
        auto &runtime = workers[std::this_thread::get_id()];
        if (!runtime) runtime = std::make_shared<AsmjitRuntime>();
        return runtime;
    }

    std::mutex mutex;
    std::unordered_map<std::thread::id, std::shared_ptr<AsmjitRuntime>> workers;
};

CompilerPipeline::CompilerPipeline()
    : llvm_(std::make_shared<LLVMSessionPool>()), asmjit_(std::make_shared<AsmjitPool>()) {
}
CompilerPipeline::~CompilerPipeline() = default;

void CompilerPipeline::record_timing(CompilationPath path, size_t functions, std::chrono::nanoseconds wall_time,
                                     std::chrono::nanoseconds cpu_time) {
    std::lock_guard lock(timings_mutex_);
    auto &timing = timings_[static_cast<size_t>(path)];
    timing.functions += functions;
    timing.wall_time += wall_time;
    timing.cpu_time += cpu_time;
}

CompilationTimings CompilerPipeline::timings() const {
    std::lock_guard lock(timings_mutex_);
    return timings_;
}

const char *backend_name(Backend backend) {
    switch (backend) {
    case Backend::Cpp: return "cpp";
    case Backend::Llvm: return "llvm";
    case Backend::Asmjit: return "asmjit";
    }
    return "unknown";
}

static void fail_compilation(const CompileRef &ref, std::string message) {
    ref.item->compilation_failed.store(true, std::memory_order_relaxed);
    std::lock_guard lock(ref.item->mutex);
    if (!ref.item->result.failed) {
        ref.item->result.failed = true;
        ref.item->result.message =
            std::format("{} {}: {}", backend_name(ref.item->implementations[ref.implementation_index].backend),
                        ref.item->implementations[ref.implementation_index].bundle_name, message);
    }
}

void CompilerPipeline::compile_clang_group(const std::vector<CompileRef> &group, unsigned optimization_level) {
    auto session = llvm_->current_worker(optimization_level);
    auto wall_start = std::chrono::steady_clock::now();
    auto cpu_start = thread_cpu_time();
    try {
        auto module = compile_cpp_source(clang_source(group), optimization_level);
        session->add_module(std::move(module));
        assign_functions(group, session);
    } catch (const std::exception &error) {
        for (const auto &ref : group)
            fail_compilation(ref, error.what());
    }
    auto path = optimization_level >= 3 ? CompilationPath::CppO3 : CompilationPath::CppO1;
    record_timing(path, group.size(), std::chrono::steady_clock::now() - wall_start, thread_cpu_time() - cpu_start);
}

void CompilerPipeline::compile_llvm_group(const std::vector<CompileRef> &group, unsigned optimization_level) {
    auto session = llvm_->current_worker(optimization_level);
    auto wall_start = std::chrono::steady_clock::now();
    auto cpu_start = thread_cpu_time();
    try {
        auto context = std::make_unique<llvm::LLVMContext>();
        std::unique_ptr<llvm::Module> combined;
        for (const CompileRef &ref : group) {
            const auto &impl = ref.item->implementations[ref.implementation_index];
            llvm::SMDiagnostic diagnostic;
            auto module = llvm::parseAssemblyString(impl.code, diagnostic, *context);
            if (!module) {
                std::string text;
                llvm::raw_string_ostream stream(text);
                diagnostic.print("local_runner", stream);
                throw std::runtime_error(text);
            }
            auto *function = module->getFunction("expr");
            if (!function) { throw std::runtime_error(std::format("{} has no @expr function", impl.bundle_name)); }
            function->setName(impl.symbol);
            if (!combined) {
                combined = std::move(module);
            } else if (llvm::Linker(*combined).linkInModule(std::move(module))) {
                throw std::runtime_error("failed to link LLVM expression group");
            }
        }
        if (!combined) return;
        llvm_backend::LLVMModuleOwner owner{std::move(context), std::move(combined)};
        session->optimize_module(owner);
        session->add_module(std::move(owner));
        assign_functions(group, session);
    } catch (const std::exception &error) {
        for (const auto &ref : group)
            fail_compilation(ref, error.what());
    }
    auto path = optimization_level >= 3 ? CompilationPath::LlvmIrO3 : CompilationPath::LlvmIrO1;
    record_timing(path, group.size(), std::chrono::steady_clock::now() - wall_start, thread_cpu_time() - cpu_start);
}

void CompilerPipeline::materialize_asmjit_group(const std::vector<CompileRef> &group) {
    auto runtime = asmjit_->current_worker();
    for (const CompileRef &ref : group) {
        try {
            std::vector<char> bytes;
            auto &impl = ref.item->implementations[ref.implementation_index];
            if (auto error = llvm::decodeBase64(impl.code, bytes)) {
                throw std::runtime_error(error_string(std::move(error)));
            }
            impl.function = FunctionHandle{runtime->add(bytes), runtime};
        } catch (const std::exception &error) { fail_compilation(ref, error.what()); }
    }
}

} // namespace simjit::local_runner
