// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "local_runner.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace simjit::local_runner {
namespace {

class CancellationState {
public:
    void cancel(size_t input_index) {
        size_t current = earliest_failure_.load(std::memory_order_relaxed);
        while (input_index < current &&
               !earliest_failure_.compare_exchange_weak(current, input_index, std::memory_order_release,
                                                        std::memory_order_relaxed)) {}
    }
    bool cancelled() const { return earliest_failure_.load(std::memory_order_acquire) != static_cast<size_t>(-1); }
    size_t earliest_failure() const { return earliest_failure_.load(std::memory_order_acquire); }

private:
    std::atomic<size_t> earliest_failure_{static_cast<size_t>(-1)};
};

enum class TaskQueue : size_t {
    Parse,
    Clang,
    Llvm,
    Asmjit,
    Execute,
    BenchmarkPrepare,
    Benchmark,
    Report,
    Count
};

struct TaskQueueStats {
    size_t submitted = 0;
    size_t completed = 0;
    size_t active = 0;
    size_t high_water = 0;
    std::chrono::nanoseconds wall_time{};
    std::chrono::nanoseconds task_wall_time{};
    std::chrono::nanoseconds cpu_time{};
    std::chrono::steady_clock::time_point wall_start{};
    std::chrono::steady_clock::time_point wall_finish{};
    bool wall_started = false;
};

class TaskPool {
public:
    explicit TaskPool(size_t worker_count);
    ~TaskPool();

    TaskPool(const TaskPool &) = delete;
    TaskPool &operator=(const TaskPool &) = delete;

    void set_limit(TaskQueue queue, size_t limit) {
        std::lock_guard lock(mutex_);
        limits_[static_cast<size_t>(queue)] = std::max<size_t>(1, limit);
    }
    void submit(TaskQueue queue, std::function<void()> task);
    void wait() {
        std::unique_lock lock(mutex_);
        idle_.wait(lock, [this] { return outstanding_ == 0; });
    }
    std::array<TaskQueueStats, static_cast<size_t>(TaskQueue::Count)> stats() const {
        std::lock_guard lock(mutex_);
        return stats_;
    }
    size_t worker_count() const noexcept { return workers_.size(); }

private:
    struct Entry {
        TaskQueue queue{};
        std::function<void()> task;
    };

    bool take_task(Entry &entry);
    void worker_loop();

    mutable std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable idle_;
    std::array<std::deque<std::function<void()>>, static_cast<size_t>(TaskQueue::Count)> queues_{};
    std::array<TaskQueueStats, static_cast<size_t>(TaskQueue::Count)> stats_{};
    std::array<size_t, static_cast<size_t>(TaskQueue::Count)> limits_{};
    std::vector<std::jthread> workers_;
    size_t outstanding_ = 0;
    size_t next_queue_ = 0;
    bool stopping_ = false;
};

namespace {
constexpr size_t queue_count = static_cast<size_t>(TaskQueue::Count);

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
} // namespace

TaskPool::TaskPool(size_t worker_count) {
    worker_count = std::max<size_t>(1, worker_count);
    limits_.fill(worker_count);
    workers_.reserve(worker_count);
    for (size_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this] { worker_loop(); });
    }
}

TaskPool::~TaskPool() {
    wait();
    {
        std::lock_guard lock(mutex_);
        stopping_ = true;
    }
    ready_.notify_all();
}

void TaskPool::submit(TaskQueue queue, std::function<void()> task) {
    if (!task) { throw std::invalid_argument("cannot submit an empty task"); }
    {
        std::lock_guard lock(mutex_);
        if (stopping_) { throw std::runtime_error("cannot submit to a stopped task pool"); }
        size_t index = static_cast<size_t>(queue);
        queues_[index].push_back(std::move(task));
        ++stats_[index].submitted;
        ++outstanding_;
    }
    ready_.notify_one();
}

bool TaskPool::take_task(Entry &entry) {
    for (size_t offset = 0; offset < queue_count; ++offset) {
        size_t index = (next_queue_ + offset) % queue_count;
        if (queues_[index].empty() || stats_[index].active >= limits_[index]) { continue; }
        entry.queue = static_cast<TaskQueue>(index);
        entry.task = std::move(queues_[index].front());
        queues_[index].pop_front();
        ++stats_[index].active;
        stats_[index].high_water = std::max(stats_[index].high_water, stats_[index].active);
        if (!stats_[index].wall_started) {
            stats_[index].wall_start = std::chrono::steady_clock::now();
            stats_[index].wall_started = true;
        }
        next_queue_ = (index + 1) % queue_count;
        return true;
    }
    return false;
}

void TaskPool::worker_loop() {
    while (true) {
        Entry entry;
        {
            std::unique_lock lock(mutex_);
            ready_.wait(lock, [this, &entry] { return stopping_ || take_task(entry); });
            if (stopping_ && !entry.task) { return; }
        }

        auto task_wall_start = std::chrono::steady_clock::now();
        auto cpu_start = thread_cpu_time();
        try {
            entry.task();
        } catch (...) {
            // Pipeline tasks translate failures into case results. Keep workers alive if a task violates that contract.
        }
        auto wall_finish = std::chrono::steady_clock::now();
        auto cpu_elapsed = thread_cpu_time() - cpu_start;

        {
            std::lock_guard lock(mutex_);
            size_t index = static_cast<size_t>(entry.queue);
            --stats_[index].active;
            ++stats_[index].completed;
            stats_[index].wall_finish = std::max(stats_[index].wall_finish, wall_finish);
            stats_[index].wall_time = stats_[index].wall_finish - stats_[index].wall_start;
            stats_[index].task_wall_time += wall_finish - task_wall_start;
            stats_[index].cpu_time += cpu_elapsed;
            --outstanding_;
            if (outstanding_ == 0) { idle_.notify_all(); }
        }
        ready_.notify_all();
    }
}

static const char *task_queue_name(TaskQueue queue) {
    switch (queue) {
    case TaskQueue::Parse: return "parse";
    case TaskQueue::Clang: return "clang";
    case TaskQueue::Llvm: return "llvm";
    case TaskQueue::Asmjit: return "asmjit";
    case TaskQueue::Execute: return "execute";
    case TaskQueue::BenchmarkPrepare: return "benchprep";
    case TaskQueue::Benchmark: return "benchmark";
    case TaskQueue::Report: return "report";
    case TaskQueue::Count: break;
    }
    return "unknown";
}

static bool host_variant(std::string_view variant) {
#if defined(__x86_64__) || defined(_M_X64)
    return variant.starts_with("x86-");
#else
    return variant.starts_with("arm-");
#endif
}

static void schedule_execute(TaskPool &pool, const BundleCasePtr &item, const RunnerOptions &options,
                             const std::shared_ptr<CancellationState> &control) {
    if (control->cancelled()) {
        std::lock_guard lock(item->mutex);
        if (!item->result.failed) item->result = CaseResult::make_cancelled();
        return;
    }
    pool.submit(TaskQueue::Execute, [&pool, item, &options, control] {
        if (control->cancelled()) {
            std::lock_guard lock(item->mutex);
            if (!item->result.failed) item->result = CaseResult::make_cancelled();
            return;
        }
        CaseResult result = execute_case(*item, options);
        if (result.failed) control->cancel(item->input_index);
        pool.submit(TaskQueue::Report, [item, result = std::move(result)]() mutable {
            std::lock_guard lock(item->mutex);
            item->result = std::move(result);
        });
    });
}

using BenchmarkSlots = std::vector<std::vector<BenchmarkInvocationPtr>>;

static void schedule_benchmark_prepare(TaskPool &pool, const BundleCasePtr &item, const RunnerOptions &options,
                                       const std::shared_ptr<CancellationState> &control,
                                       const std::shared_ptr<BenchmarkSlots> &slots) {
    if (control->cancelled()) {
        std::lock_guard lock(item->mutex);
        if (!item->result.failed) item->result = CaseResult::make_cancelled();
        return;
    }
    auto &case_slots = slots->at(item->input_index);
    if (case_slots.empty()) {
        item->result.complete = true;
        return;
    }
    for (size_t implementation_index = 0; implementation_index < case_slots.size(); ++implementation_index) {
        pool.submit(TaskQueue::BenchmarkPrepare, [item, implementation_index, &options, control, slots] {
            if (control->cancelled()) return;
            try {
                slots->at(item->input_index)[implementation_index] =
                    prepare_benchmark(*item, implementation_index, options);
            } catch (const std::exception &error) {
                {
                    std::lock_guard lock(item->mutex);
                    if (!item->result.failed) {
                        item->result.failed = true;
                        item->result.complete = true;
                        item->result.message = error.what();
                    }
                }
                control->cancel(item->input_index);
            }
        });
    }
}

void compilation_done(TaskPool &pool, const CompileRef &ref, const RunnerOptions &options,
                      const std::shared_ptr<CancellationState> &control,
                      const std::shared_ptr<BenchmarkSlots> &benchmark_slots) {
    if (ref.item->pending_compilations.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (options.bench)
            schedule_benchmark_prepare(pool, ref.item, options, control, benchmark_slots);
        else
            schedule_execute(pool, ref.item, options, control);
    }
}

template <typename CompileFn>
void submit_compile_group(TaskPool &pool, TaskQueue queue, std::vector<CompileRef> group, const RunnerOptions &options,
                          const std::shared_ptr<CancellationState> &control,
                          const std::shared_ptr<BenchmarkSlots> &benchmark_slots, CompileFn compile) {
    pool.submit(queue,
                [&pool, group = std::move(group), &options, control, benchmark_slots, compile = std::move(compile)] {
                    compile(group);
                    for (const auto &ref : group) {
                        if (ref.item->compilation_failed.load(std::memory_order_relaxed))
                            control->cancel(ref.item->input_index);
                        compilation_done(pool, ref, options, control, benchmark_slots);
                    }
                });
}

void print_timings(const TaskPool &pool, const CompilationTimings &compilation,
                   std::chrono::steady_clock::duration total) {
    llvm::outs() << "Pipeline timings:\n";
    auto stats = pool.stats();
    for (size_t i = 0; i < stats.size(); ++i) {
        const auto &value = stats[i];
        double wall_ms = std::chrono::duration<double, std::milli>(value.wall_time).count();
        double cpu_ms = std::chrono::duration<double, std::milli>(value.cpu_time).count();
        llvm::outs() << std::format(
            "  {:8}: submitted={} completed={} active_high_water={} cpu_ms={:.3f} wall_ms={:.3f}\n",
            task_queue_name(static_cast<TaskQueue>(i)), value.submitted, value.completed, value.high_water, cpu_ms,
            wall_ms);
    }
    llvm::outs() << std::format("  wall_ms: {:.3f}\n", std::chrono::duration<double, std::milli>(total).count());
    llvm::outs() << "Compilation averages:\n";
    auto print_compile_average = [&](CompilationPath path, std::string_view name) {
        const auto &value = compilation[static_cast<size_t>(path)];
        double wall_ms = value.functions == 0
                             ? 0.0
                             : std::chrono::duration<double, std::milli>(value.wall_time).count() / value.functions;
        double cpu_ms = value.functions == 0
                            ? 0.0
                            : std::chrono::duration<double, std::milli>(value.cpu_time).count() / value.functions;
        llvm::outs() << std::format("  {:14}: functions={} avg_wall_ms={:.3f} avg_cpu_ms={:.3f}\n", name,
                                    value.functions, wall_ms, cpu_ms);
    };
    print_compile_average(CompilationPath::CppO1, "c++ text o1");
    print_compile_average(CompilationPath::CppO3, "c++ text o3");
    print_compile_average(CompilationPath::LlvmIrO1, "llvm ir o1");
    print_compile_average(CompilationPath::LlvmIrO3, "llvm ir o3");
}

} // namespace

int run(int argc, char **argv) {
    auto start = std::chrono::steady_clock::now();
    RunnerOptions options = parse_options(argc, argv);
    std::ifstream input_file;
    std::istream *input = &std::cin;
    if (options.file != "-") {
        input_file.open(options.file);
        if (!input_file) throw std::runtime_error(std::format("failed to open bundle file '{}'", options.file));
        input = &input_file;
    }

    std::vector<std::string> lines;
    for (std::string line; std::getline(*input, line);) {
        if (!line.empty()) lines.push_back(std::move(line));
    }
    if (lines.empty()) throw std::runtime_error(std::format("bundle file is empty: {}", options.file));

    TaskPool pool(options.workers);
    pool.set_limit(TaskQueue::Clang, std::min<size_t>(2, options.workers));
    pool.set_limit(TaskQueue::Llvm,
                   std::min<size_t>(4, std::max<size_t>(1, options.workers > 2 ? options.workers - 2 : 1)));

    std::vector<BundleCasePtr> parsed(lines.size());
    std::vector<std::string> parse_errors(lines.size());
    for (size_t i = 0; i < lines.size(); ++i) {
        pool.submit(TaskQueue::Parse, [&, i] {
            try {
                parsed[i] = parse_bundle_line(lines[i], i + 1, i);
            } catch (const std::exception &error) { parse_errors[i] = error.what(); }
        });
    }
    pool.wait();
    for (const auto &error : parse_errors) {
        if (!error.empty()) throw std::runtime_error(error);
    }

    std::vector<BundleCasePtr> selected;
    size_t runnable_selected = 0;
    bool only_n_matched = false;
    for (const auto &item : parsed) {
        if (!case_selected(*item, options)) continue;
        if (options.only_n && item->number == *options.only_n) only_n_matched = true;
        if (!item->structured_error) {
            if (options.limit != 0 && runnable_selected >= options.limit) break;
            ++runnable_selected;
        }
        selected.push_back(item);
    }
    if (options.only_n && !only_n_matched)
        throw std::runtime_error(std::format("no bundle item found for n={}", *options.only_n));

    auto compiler = std::make_shared<CompilerPipeline>();
    auto control = std::make_shared<CancellationState>();
    auto benchmark_slots = std::make_shared<BenchmarkSlots>(lines.size());
    std::vector<CompileRef> clang_refs;
    std::vector<CompileRef> clang_o3_refs;
    std::vector<CompileRef> llvm_refs;
    std::vector<CompileRef> llvm_o3_refs;
    std::vector<CompileRef> asmjit_refs;
    for (const auto &item : selected) {
        if (item->structured_error) {
            item->result.complete = true;
            continue;
        }
        if (!host_variant(item->variant)) {
            item->result =
                CaseResult{.complete = true,
                           .failed = true,
                           .message = std::format("foreign architecture variant '{}' is not runnable on this host",
                                                  item->variant)};
            control->cancel(item->input_index);
            continue;
        }
        if (!options.code_names.empty()) {
            std::erase_if(item->implementations, [&](const Implementation &implementation) {
                return !options.code_names.contains(implementation.bundle_name);
            });
        }
        if (options.bench) {
            size_t original_count = item->implementations.size();
            for (size_t i = 0; i < original_count; ++i) {
                const auto &implementation = item->implementations[i];
                if (implementation.backend == Backend::Asmjit ||
                    options.benchmark_o3 == RunnerOptions::BenchmarkO3::None)
                    continue;
                if (options.benchmark_o3 == RunnerOptions::BenchmarkO3::Scalar &&
                    !implementation.bundle_name.ends_with("_s"))
                    continue;
                Implementation optimized = implementation;
                optimized.bundle_name += "_o3";
                optimized.symbol += "_o3";
                optimized.optimization_level = 3;
                optimized.function = {};
                item->implementations.push_back(std::move(optimized));
            }
            benchmark_slots->at(item->input_index).resize(item->implementations.size());
        }
        item->pending_compilations = item->implementations.size();
        if (item->implementations.empty()) {
            if (options.bench)
                schedule_benchmark_prepare(pool, item, options, control, benchmark_slots);
            else
                schedule_execute(pool, item, options, control);
            continue;
        }
        for (size_t i = 0; i < item->implementations.size(); ++i) {
            const auto &implementation = item->implementations[i];
            size_t weight = implementation.code.size();
            if (implementation.backend == Backend::Asmjit) weight = weight / 4 * 3;
            CompileRef ref{item, i, weight};
            switch (item->implementations[i].backend) {
            case Backend::Cpp:
                (implementation.optimization_level >= 3 ? clang_o3_refs : clang_refs).push_back(ref);
                break;
            case Backend::Llvm:
                (implementation.optimization_level >= 3 ? llvm_o3_refs : llvm_refs).push_back(ref);
                break;
            case Backend::Asmjit: asmjit_refs.push_back(ref); break;
            }
        }
    }

    size_t clang_minimum = clang_refs.size() >= 2 ? 2 : clang_refs.size();
    for (auto &group : balance_compile_groups(std::move(clang_refs), size_t(512) * 1024, clang_minimum)) {
        submit_compile_group(pool, TaskQueue::Clang, std::move(group), options, control, benchmark_slots,
                             [compiler](const auto &refs) { compiler->compile_clang_group(refs); });
    }
    for (auto &group : balance_compile_groups(std::move(clang_o3_refs), size_t(512) * 1024, 1)) {
        submit_compile_group(pool, TaskQueue::Clang, std::move(group), options, control, benchmark_slots,
                             [compiler](const auto &refs) { compiler->compile_clang_group(refs, 3); });
    }
    for (auto &group : balance_compile_groups(std::move(llvm_refs), size_t(1024) * 1024, 1)) {
        submit_compile_group(pool, TaskQueue::Llvm, std::move(group), options, control, benchmark_slots,
                             [compiler](const auto &refs) { compiler->compile_llvm_group(refs); });
    }
    for (auto &group : balance_compile_groups(std::move(llvm_o3_refs), size_t(1024) * 1024, 1)) {
        submit_compile_group(pool, TaskQueue::Llvm, std::move(group), options, control, benchmark_slots,
                             [compiler](const auto &refs) { compiler->compile_llvm_group(refs, 3); });
    }
    auto asm_groups = balance_compile_groups(std::move(asmjit_refs), size_t(256) * 1024, 1);
    for (auto &group : asm_groups) {
        std::vector<CompileRef> chunk;
        size_t chunk_bytes = 0;
        auto submit_chunk = [&] {
            if (chunk.empty()) return;
            submit_compile_group(pool, TaskQueue::Asmjit, std::move(chunk), options, control, benchmark_slots,
                                 [compiler](const auto &refs) { compiler->materialize_asmjit_group(refs); });
            chunk = {};
            chunk_bytes = 0;
        };
        for (auto &ref : group) {
            if (!chunk.empty() && (chunk.size() == 128 || chunk_bytes + ref.weight_bytes > size_t(256) * 1024)) {
                submit_chunk();
            }
            chunk_bytes += ref.weight_bytes;
            chunk.push_back(std::move(ref));
        }
        submit_chunk();
    }
    pool.wait();

    size_t structured_errors = 0;
    size_t skipped_python = 0;
    size_t comparison_skipped = 0;
    size_t failures = 0;
    const BundleCase *earliest_failure = nullptr;
    for (const auto &item : selected) {
        skipped_python += item->skipped_python;
        if (item->structured_error) {
            ++structured_errors;
            if (options.verbose_item) llvm::outs() << "Skipping " << item->label() << ": " << item->error_label << '\n';
            continue;
        }
        if (options.verbose_item) llvm::outs() << item->label() << '\n';
        if (item->result.comparison_skipped) ++comparison_skipped;
        if (item->result.failed) {
            ++failures;
            if (!earliest_failure) earliest_failure = item.get();
        }
    }

    if (earliest_failure) {
        llvm::errs() << earliest_failure->label() << ": " << earliest_failure->result.message << '\n';
    }

    size_t benchmarked = 0;
    if (options.bench && failures == 0) {
        std::vector<BenchmarkInvocationPtr> invocations;
        for (const auto &item : selected) {
            for (auto &invocation : benchmark_slots->at(item->input_index)) {
                if (invocation) invocations.push_back(std::move(invocation));
            }
        }
        std::string benchmark_error;
        llvm::outs().flush();
        pool.submit(TaskQueue::Benchmark, [&] {
            try {
                benchmarked = run_registered_benchmarks(invocations, options.benchmark_arguments);
            } catch (const std::exception &error) { benchmark_error = error.what(); }
        });
        pool.wait();
        if (!benchmark_error.empty()) throw std::runtime_error(benchmark_error);
    }

    if (options.bench) {
        llvm::outs() << std::format(
            "Summary: runnable={} structured_errors={} skipped_python={} benchmarked={} failed={}\n", runnable_selected,
            structured_errors, skipped_python, benchmarked, failures);
    } else {
        llvm::outs() << std::format(
            "Summary: runnable={} structured_errors={} skipped_python={} comparison_skipped={} failed={}\n",
            runnable_selected, structured_errors, skipped_python, comparison_skipped, failures);
    }
    if (options.timings) print_timings(pool, compiler->timings(), std::chrono::steady_clock::now() - start);
    if (failures == 0) llvm::outs() << "Testing complete!\n";
    return failures == 0 ? 0 : 1;
}

} // namespace simjit::local_runner

int main(int argc, char **argv) {
    try {
        return simjit::local_runner::run(argc, argv);
    } catch (const std::exception &error) {
        llvm::errs() << "error: " << error.what() << '\n';
        return 1;
    }
}
