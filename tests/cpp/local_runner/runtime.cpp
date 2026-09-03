// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "local_runner.h"

#include "simjit/core/expr.h"
#include "simjit/jit.h"

#include "benchmark/benchmark.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <limits>
#include <numbers>
#include <span>
#include <stdexcept>

namespace simjit::local_runner {
namespace {

// xorshift64*
class Random {
public:
    explicit Random(uint64_t seed) : state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}
    uint64_t next() {
        uint64_t x = state_;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state_ = x;
        return x * 0x2545f4914f6cdd1dULL;
    }
    uint64_t bounded(uint64_t bound) { return bound == 0 ? 0 : next() % bound; }
    double unit() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }
    double normal() {
        double u1 = std::max(unit(), std::numeric_limits<double>::min());
        double u2 = unit();
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
    }

private:
    uint64_t state_;
};

uint64_t hash_string(uint64_t seed, std::string_view text) {
    uint64_t result = seed ^ 1469598103934665603ULL;
    for (unsigned char c : text) {
        result ^= c;
        result *= 1099511628211ULL;
    }
    return result;
}

class Buffer {
public:
    explicit Buffer(size_t bytes)
        : bytes_(bytes), data_(static_cast<std::byte *>(::operator new(bytes, std::align_val_t(64)))) {
        std::memset(data_, 0, bytes_);
    }
    ~Buffer() { ::operator delete(data_, std::align_val_t(64)); }
    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&other) noexcept : bytes_(other.bytes_), data_(other.data_) { other.data_ = nullptr; }
    Buffer &operator=(Buffer &&other) noexcept {
        if (this != &other) {
            if (data_) ::operator delete(data_, std::align_val_t(64));
            bytes_ = other.bytes_;
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }
    void *data() const { return data_; }
    size_t size() const { return bytes_; }
    void zero() { std::memset(data_, 0, bytes_); }

private:
    size_t bytes_ = 0;
    std::byte *data_ = nullptr;
};

template <typename T> void write_integer_values(Buffer &buffer, size_t rows, Random &random) {
    auto *values = static_cast<T *>(buffer.data());
    static constexpr int64_t test_cases[]{-1000000000, -10000, -1, 0, 1, 10000, 1000000000};
    constexpr unsigned digits = std::numeric_limits<T>::digits;
    auto clamp = [](int64_t value) {
        if constexpr (sizeof(T) < sizeof(int64_t)) {
            value = std::clamp(value, static_cast<int64_t>(std::numeric_limits<T>::min()),
                               static_cast<int64_t>(std::numeric_limits<T>::max()));
        }
        return static_cast<T>(value);
    };
    for (size_t i = 0; i < rows; ++i) {
        uint64_t choice = random.bounded(44);
        if (choice == 0)
            values[i] = 0;
        else if (choice == 1)
            values[i] = std::numeric_limits<T>::min();
        else if (choice == 2)
            values[i] = std::numeric_limits<T>::max();
        else if (choice == 3)
            values[i] = clamp(test_cases[random.bounded(std::size(test_cases))]);
        else if (choice < 9) {
            int64_t value = 1;
            unsigned decimal = static_cast<unsigned>(random.bounded(std::max<unsigned>(1, digits / 3)));
            for (unsigned power = 0; power < decimal && value <= std::numeric_limits<int64_t>::max() / 10; ++power) {
                value *= 10;
            }
            values[i] = clamp(random.bounded(2) ? value : -value);
        } else if (choice < 14) {
            unsigned exponent = static_cast<unsigned>(random.bounded(digits));
            int64_t value = int64_t{1} << std::min<unsigned>(exponent, 62);
            values[i] = clamp(random.bounded(2) ? value : -value);
        } else {
            values[i] = static_cast<T>(
                std::llround(random.normal() * std::sqrt(static_cast<double>(std::numeric_limits<T>::max()))));
        }
    }
}

template <> void write_integer_values<__int128>(Buffer &buffer, size_t rows, Random &random) {
    auto *values = static_cast<__int128 *>(buffer.data());
    for (size_t i = 0; i < rows; ++i)
        values[i] = (static_cast<__int128>(random.next()) << 64) | random.next();
}

template <typename T> void write_float_values(Buffer &buffer, size_t rows, Random &random) {
    auto *values = static_cast<T *>(buffer.data());
    for (size_t i = 0; i < rows; ++i) {
        switch (random.bounded(37)) {
        case 0: values[i] = T(0); break;
        // Python's generator uses sys.float_info for both f32 and f64 and
        // lets ctypes perform the destination-width conversion.
        case 1: values[i] = static_cast<T>(std::numeric_limits<double>::min()); break;
        case 2: values[i] = static_cast<T>(std::numeric_limits<double>::max()); break;
        case 3: values[i] = static_cast<T>(std::numeric_limits<double>::epsilon()); break;
        case 4: values[i] = std::numeric_limits<T>::infinity(); break;
        case 5: values[i] = -std::numeric_limits<T>::infinity(); break;
        case 6: values[i] = std::numeric_limits<T>::quiet_NaN(); break;
        default:
            // Python assigns weights 10 and 20 to [0, 1) and normal values.
            if (random.bounded(30) < 10)
                values[i] = static_cast<T>(random.unit());
            else
                values[i] = static_cast<T>(random.normal() * 1.0e6);
            break;
        }
    }
}

void generate_input(Buffer &buffer, const ArgumentInfo &arg, size_t rows, Random &random) {
    if (arg.kind == ArgumentKind::Sequence) {
        switch (arg.dtype) {
        case ScalarDataType::I1:
        case ScalarDataType::I8:
            for (size_t i = 0; i < rows; ++i)
                static_cast<int8_t *>(buffer.data())[i] = static_cast<int8_t>(i);
            break;
        case ScalarDataType::I16:
            for (size_t i = 0; i < rows; ++i)
                static_cast<int16_t *>(buffer.data())[i] = static_cast<int16_t>(i);
            break;
        case ScalarDataType::I32:
            for (size_t i = 0; i < rows; ++i)
                static_cast<int32_t *>(buffer.data())[i] = static_cast<int32_t>(i);
            break;
        case ScalarDataType::I64:
            for (size_t i = 0; i < rows; ++i)
                static_cast<int64_t *>(buffer.data())[i] = static_cast<int64_t>(i);
            break;
        case ScalarDataType::I128:
            for (size_t i = 0; i < rows; ++i)
                static_cast<__int128 *>(buffer.data())[i] = i;
            break;
        case ScalarDataType::F32:
            for (size_t i = 0; i < rows; ++i)
                static_cast<float *>(buffer.data())[i] = static_cast<float>(i);
            break;
        case ScalarDataType::F64:
            for (size_t i = 0; i < rows; ++i)
                static_cast<double *>(buffer.data())[i] = static_cast<double>(i);
            break;
        }
        return;
    }
    if (arg.kind != ArgumentKind::Input) return;
    switch (arg.dtype) {
    case ScalarDataType::I1:
    case ScalarDataType::I8: write_integer_values<int8_t>(buffer, rows, random); break;
    case ScalarDataType::I16: write_integer_values<int16_t>(buffer, rows, random); break;
    case ScalarDataType::I32: write_integer_values<int32_t>(buffer, rows, random); break;
    case ScalarDataType::I64: write_integer_values<int64_t>(buffer, rows, random); break;
    case ScalarDataType::I128: write_integer_values<__int128>(buffer, rows, random); break;
    case ScalarDataType::F32: write_float_values<float>(buffer, rows, random); break;
    case ScalarDataType::F64: write_float_values<double>(buffer, rows, random); break;
    }
}

template <typename T> struct UIntOf;

template <> struct UIntOf<float> {
    using type = uint32_t;
};
template <> struct UIntOf<double> {
    using type = uint64_t;
};

template <typename T> using UIntOfT = typename UIntOf<T>::type;

template <typename T> UIntOfT<T> ordered_bits(T x) {
    using U = UIntOfT<T>;
    U u = std::bit_cast<U>(x);

    constexpr U sign = U(1) << (sizeof(T) * 8 - 1);

    // Map IEEE bit patterns into monotonically increasing integers.
    return (u & sign) ? ~u : (u | sign);
}

template <typename T> UIntOfT<T> ulp_distance(T a, T b) {
    using U = UIntOfT<T>;

    U ua = ordered_bits(a);
    U ub = ordered_bits(b);

    return ua > ub ? ua - ub : ub - ua;
}

bool row_value_equal(ScalarDataType dtype, const std::byte *left, const std::byte *right, size_t row) {
    if (dtype == ScalarDataType::I1) {
        uint8_t left_byte = std::to_integer<uint8_t>(left[row >> 3]);
        uint8_t right_byte = std::to_integer<uint8_t>(right[row >> 3]);
        uint8_t bit = uint8_t{1} << (row & 7);
        return (left_byte & bit) == (right_byte & bit);
    }

    size_t offset = row * scalar_dtype_size(dtype);
    left += offset;
    right += offset;
    if (dtype == ScalarDataType::F32) {
        float a, b;
        std::memcpy(&a, left, sizeof(a));
        std::memcpy(&b, right, sizeof(b));
        return (std::isnan(a) && std::isnan(b)) || ulp_distance(a, b) <= 3;
    }
    if (dtype == ScalarDataType::F64) {
        double a, b;
        std::memcpy(&a, left, sizeof(a));
        std::memcpy(&b, right, sizeof(b));
        return (std::isnan(a) && std::isnan(b)) || ulp_distance(a, b) <= 3;
    }
    return std::memcmp(left, right, scalar_dtype_size(dtype)) == 0;
}

std::string row_value_string(ScalarDataType dtype, const std::byte *values, size_t row) {
    if (dtype == ScalarDataType::I1) {
        uint8_t byte = std::to_integer<uint8_t>(values[row >> 3]);
        return std::format("{}", (byte >> (row & 7)) & 1);
    }

    const std::byte *value = values + row * scalar_dtype_size(dtype);
    switch (dtype) {
    case ScalarDataType::I8: return std::format("{}", *reinterpret_cast<const int8_t *>(value));
    case ScalarDataType::I16: return std::format("{}", *reinterpret_cast<const int16_t *>(value));
    case ScalarDataType::I32: return std::format("{}", *reinterpret_cast<const int32_t *>(value));
    case ScalarDataType::I64: return std::format("{}", *reinterpret_cast<const int64_t *>(value));
    case ScalarDataType::I128: return "<i128>";
    case ScalarDataType::F32: return std::format("{}", *reinterpret_cast<const float *>(value));
    case ScalarDataType::F64: return std::format("{}", *reinterpret_cast<const double *>(value));
    case ScalarDataType::I1: SIMJIT_UNREACHABLE();
    }
    return "?";
}

} // namespace

class BenchmarkInvocation {
public:
    BenchmarkInvocation(const BundleCase &item, size_t implementation_index, const RunnerOptions &options)
        : rows_(options.rows), arguments_(item.args), function_(item.implementations.at(implementation_index).function),
          seed_(hash_string(options.seed ^ static_cast<uint64_t>(item.iteration), item.id + item.variant)) {
        const auto &implementation = item.implementations.at(implementation_index);
        if (!function_.address) {
            throw std::runtime_error(std::format("{} has no compiled address", implementation.bundle_name));
        }
        name_ = std::format("{}/{}/{}/{}/{}", item.number, item.id.empty() ? "unknown" : item.id, item.variant,
                            item.iteration, implementation.bundle_name);
    }

    const std::string &name() const { return name_; }

    void run(benchmark::State &state) {
        Random random(seed_);
        std::vector<Buffer> buffers;
        std::vector<void *> argument_ptrs;
        buffers.reserve(arguments_.size());
        argument_ptrs.reserve(arguments_.size());
        for (const auto &argument : arguments_) {
            buffers.emplace_back(scalar_dtype_size(argument.dtype) * rows_);
            generate_input(buffers.back(), argument, rows_, random);
            argument_ptrs.push_back(buffers.back().data());
        }
        for (auto _ : state) {
            (void)_;
            simjit::jit::call_fn_ptr(function_.address, rows_,
                                     nonstd::span<void *>{argument_ptrs.data(), argument_ptrs.size()});
            benchmark::ClobberMemory();
        }
        state.SetItemsProcessed(state.iterations() * static_cast<int64_t>(rows_));
    }

private:
    size_t rows_ = 0;
    std::vector<ArgumentInfo> arguments_;
    FunctionHandle function_;
    uint64_t seed_ = 0;
    std::string name_;
};

BenchmarkInvocationPtr prepare_benchmark(const BundleCase &item, size_t implementation_index,
                                         const RunnerOptions &options) {
    return std::make_shared<BenchmarkInvocation>(item, implementation_index, options);
}

size_t run_registered_benchmarks(const std::vector<BenchmarkInvocationPtr> &invocations,
                                 const std::vector<std::string> &arguments) {
    benchmark::ClearRegisteredBenchmarks();
    for (const auto &invocation : invocations) {
        benchmark::RegisterBenchmark(invocation->name(),
                                     [invocation](benchmark::State &state) { invocation->run(state); });
    }

    std::vector<std::string> mutable_arguments;
    mutable_arguments.reserve(arguments.size() + 1);
    mutable_arguments.emplace_back("local_runner");
    mutable_arguments.insert(mutable_arguments.end(), arguments.begin(), arguments.end());
    std::vector<char *> argument_pointers;
    argument_pointers.reserve(mutable_arguments.size());
    for (auto &argument : mutable_arguments)
        argument_pointers.push_back(argument.data());
    int argument_count = static_cast<int>(argument_pointers.size());
    benchmark::Initialize(&argument_count, argument_pointers.data());
    if (benchmark::ReportUnrecognizedArguments(argument_count, argument_pointers.data())) {
        benchmark::ClearRegisteredBenchmarks();
        benchmark::Shutdown();
        throw std::runtime_error("unrecognized Google Benchmark option");
    }
    size_t count = benchmark::RunSpecifiedBenchmarks();
    benchmark::ClearRegisteredBenchmarks();
    benchmark::Shutdown();
    return count;
}

bool is_output(ArgumentKind kind) {
    return kind == ArgumentKind::Output || kind == ArgumentKind::OutputScalar || kind == ArgumentKind::SafetyCheck;
}

std::string_view lowering_kind(const BundleCase &item, const Implementation &implementation) {
    if (item.variant.ends_with("-scalar") || implementation.bundle_name.ends_with("_s")) return "scalar";
    return "vector";
}

CaseResult execute_case(const BundleCase &item, const RunnerOptions &options) {
    CaseResult result{};
    result.complete = true;
    try {
        if (item.compilation_failed.load(std::memory_order_relaxed)) {
            std::lock_guard lock(item.mutex);
            return item.result;
        }

        uint64_t seed = hash_string(options.seed ^ static_cast<uint64_t>(item.iteration), item.id + item.variant);
        Random random(seed);
        std::vector<Buffer> arguments;
        std::vector<void *> argument_ptrs;
        arguments.reserve(item.args.size());
        argument_ptrs.reserve(item.args.size());
        for (const auto &arg : item.args) {
            arguments.emplace_back(scalar_dtype_size(arg.dtype) * options.rows);
            generate_input(arguments.back(), arg, options.rows, random);
            argument_ptrs.push_back(arguments.back().data());
        }

        using Outputs = std::vector<std::vector<std::byte>>;
        std::vector<Outputs> implementation_outputs(item.implementations.size());
        for (size_t implementation_index = 0; implementation_index < item.implementations.size();
             ++implementation_index) {
            const auto &impl = item.implementations[implementation_index];
            if (!impl.function.address)
                throw std::runtime_error(std::format("{} has no compiled address", impl.bundle_name));
            for (size_t i = 0; i < item.args.size(); ++i) {
                if (is_output(item.args[i].kind)) arguments[i].zero();
            }
            simjit::jit::call_fn_ptr(impl.function.address, options.rows,
                                     nonstd::span<void *>{argument_ptrs.data(), argument_ptrs.size()});
            for (size_t i = 0; i < item.args.size(); ++i) {
                if (!is_output(item.args[i].kind)) continue;
                auto *begin = static_cast<std::byte *>(arguments[i].data());
                implementation_outputs[implementation_index].emplace_back(begin, begin + arguments[i].size());
            }
        }

        std::vector<size_t> stable;
        for (size_t i = 0; i < item.implementations.size(); ++i) {
            if (!item.implementations[i].comparison_unstable) stable.push_back(i);
        }
        if (stable.size() < 2) {
            result.comparison_skipped = true;
            return result;
        }

        size_t baseline = stable.front();
        for (size_t implementation_index : std::span(stable).subspan(1)) {
            size_t output_index = 0;
            for (size_t arg_index = 0; arg_index < item.args.size(); ++arg_index) {
                if (!is_output(item.args[arg_index].kind)) continue;
                const auto &left = implementation_outputs[implementation_index][output_index];
                const auto &right = implementation_outputs[baseline][output_index];
                size_t mismatch_count = 0;
                std::string samples;
                for (size_t row = 0; row < options.rows; ++row) {
                    if (row_value_equal(item.args[arg_index].dtype, left.data(), right.data(), row)) continue;
                    ++mismatch_count;
                    if (mismatch_count <= 40) {
                        samples += std::format(" (idx={} left={} right={})", row,
                                               row_value_string(item.args[arg_index].dtype, left.data(), row),
                                               row_value_string(item.args[arg_index].dtype, right.data(), row));
                    }
                }
                if (mismatch_count != 0) {
                    result.failed = true;
                    if (!result.message.empty()) result.message += '\n';
                    result.message += std::format("{} ({}) != {} ({}) on output {}: {} mismatches{}",
                                                  item.implementations[implementation_index].bundle_name,
                                                  lowering_kind(item, item.implementations[implementation_index]),
                                                  item.implementations[baseline].bundle_name,
                                                  lowering_kind(item, item.implementations[baseline]), output_index,
                                                  mismatch_count, samples);
                }
                ++output_index;
            }
        }
    } catch (const std::exception &error) {
        result.failed = true;
        result.message = error.what();
    }
    return result;
}

} // namespace simjit::local_runner
