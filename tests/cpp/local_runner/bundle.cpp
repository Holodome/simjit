// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "local_runner.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <charconv>
#include <format>
#include <stdexcept>
#include <thread>

namespace simjit::local_runner {

static std::string unique_symbol(const BundleCase &item, const Implementation &impl, size_t implementation_index) {
    std::string value = std::format("simjit_{}_{}_{}_{}_{}_{}_{}", item.line_number, item.input_index, item.number,
                                    item.iteration, item.variant, impl.bundle_name, implementation_index);
    for (char &c : value) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') { c = '_'; }
    }
    return value;
}

std::string BundleCase::label() const {
    std::string location = file.empty() ? "<no-location>" : file;
    if (source_line != 0) { location += std::format(":{}", source_line); }
    return std::format("Test {} {} {}", number, id.empty() ? "?" : id, location);
}

static int64_t parse_i64(std::string_view value, std::string_view option) {
    int64_t result = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error(std::format("invalid value for {}: {}", option, value));
    }
    return result;
}

static uint64_t parse_u64(std::string_view value, std::string_view option) {
    uint64_t result = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error(std::format("invalid value for {}: {}", option, value));
    }
    return result;
}

static std::string required_string(const llvm::json::Object &object, llvm::StringRef key, size_t line) {
    auto value = object.getString(key);
    if (!value) {
        throw std::runtime_error(std::format("bundle line {} is missing string field '{}'", line, key.str()));
    }
    return value->str();
}

static ScalarDataType parse_dtype(llvm::StringRef text, size_t line) {
    if (text == "i1") return ScalarDataType::I1;
    if (text == "i8") return ScalarDataType::I8;
    if (text == "i16") return ScalarDataType::I16;
    if (text == "i32") return ScalarDataType::I32;
    if (text == "i64") return ScalarDataType::I64;
    if (text == "i128") return ScalarDataType::I128;
    if (text == "f32") return ScalarDataType::F32;
    if (text == "f64") return ScalarDataType::F64;
    throw std::runtime_error(std::format("bundle line {} has unknown dtype '{}'", line, text.str()));
}

static ArgumentKind parse_kind(llvm::StringRef text, size_t line) {
    if (text == "in") return ArgumentKind::Input;
    if (text == "out") return ArgumentKind::Output;
    if (text == "outs") return ArgumentKind::OutputScalar;
    if (text == "safety") return ArgumentKind::SafetyCheck;
    if (text == "sv") return ArgumentKind::Sequence;
    throw std::runtime_error(std::format("bundle line {} has unknown argument kind '{}'", line, text.str()));
}

static std::string usage() {
    return "usage: local_runner [options]\n"
           "  --file, -f PATH       JSONL bundle or - for stdin (default tests.jsonl)\n"
           "  --first N             Skip records with n < N\n"
           "  --suite NAME          Select suite; repeatable\n"
           "  --limit N             Process at most N runnable records\n"
           "  --only-n N            Select one transient test number\n"
           "  --code NAME           Select native code kind; repeatable\n"
           "  --verbose-item         Print every selected record label\n"
           "  --bench                Benchmark native implementations\n"
           "  --bench-o3 MODE        O3 copies: none, scalar, or all\n"
           "  --workers N            Worker threads\n"
           "  --seed N               Deterministic base seed\n"
           "  --rows N               Rows per call (default 2048, benchmark 4096)\n"
           "  --benchmark_*          Forward Google Benchmark options\n"
           "  --timings              Print queue timings\n";
}

RunnerOptions parse_options(int argc, char **argv) {
    RunnerOptions options;
    bool rows_explicit = false;
    options.workers = std::max<unsigned>(1, std::thread::hardware_concurrency());
    auto require_value = [&](int &index, std::string_view name) -> std::string_view {
        if (++index >= argc) { throw std::runtime_error(std::format("{} requires a value", name)); }
        return argv[index];
    };
    for (int i = 1; i < argc; ++i) {
        std::string_view arg = argv[i];
        if (arg == "--help") {
            llvm::outs() << usage();
            std::exit(0);
        } else if (arg == "--file" || arg == "-f") {
            options.file = require_value(i, arg);
        } else if (arg == "--first") {
            options.first = parse_i64(require_value(i, arg), arg);
        } else if (arg == "--suite") {
            options.suites.emplace(require_value(i, arg));
        } else if (arg == "--limit") {
            options.limit = parse_u64(require_value(i, arg), arg);
        } else if (arg == "--only-n") {
            options.only_n = parse_i64(require_value(i, arg), arg);
        } else if (arg == "--code") {
            options.code_names.emplace(require_value(i, arg));
        } else if (arg == "--workers") {
            options.workers = parse_u64(require_value(i, arg), arg);
            if (options.workers == 0) { throw std::runtime_error("--workers must be positive"); }
        } else if (arg == "--seed") {
            options.seed = parse_u64(require_value(i, arg), arg);
        } else if (arg == "--rows") {
            options.rows = parse_u64(require_value(i, arg), arg);
            rows_explicit = true;
            if (options.rows == 0) { throw std::runtime_error("--rows must be positive"); }
        } else if (arg == "--verbose-item") {
            options.verbose_item = true;
        } else if (arg == "--bench") {
            options.bench = true;
        } else if (arg == "--bench-o3") {
            std::string_view value = require_value(i, arg);
            if (value == "none")
                options.benchmark_o3 = RunnerOptions::BenchmarkO3::None;
            else if (value == "scalar")
                options.benchmark_o3 = RunnerOptions::BenchmarkO3::Scalar;
            else if (value == "all")
                options.benchmark_o3 = RunnerOptions::BenchmarkO3::All;
            else
                throw std::runtime_error("--bench-o3 must be none, scalar, or all");
        } else if (arg.starts_with("--benchmark_")) {
            options.benchmark_arguments.emplace_back(arg);
        } else if (arg == "--timings") {
            options.timings = true;
        } else if (arg == "--isolate") {
            throw std::runtime_error(std::format("{} is not supported by local_runner", arg));
        } else {
            throw std::runtime_error(std::format("unknown option '{}'", arg));
        }
    }
    if (!options.benchmark_arguments.empty() && !options.bench) {
        throw std::runtime_error("--benchmark_* options require --bench");
    }
    if (!options.bench && options.benchmark_o3 != RunnerOptions::BenchmarkO3::Scalar) {
        throw std::runtime_error("--bench-o3 requires --bench");
    }
    if (options.bench && !rows_explicit) options.rows = 4096;
    return options;
}

BundleCasePtr parse_bundle_line(std::string_view line, size_t line_number, size_t input_index) {
    auto parsed = llvm::json::parse(line);
    if (!parsed) {
        throw std::runtime_error(std::format("bundle line {}: {}", line_number, llvm::toString(parsed.takeError())));
    }
    auto *object = parsed->getAsObject();
    if (!object) { throw std::runtime_error(std::format("bundle line {} is not a JSON object", line_number)); }

    auto item = std::make_shared<BundleCase>();
    item->input_index = input_index;
    item->line_number = line_number;
    if (auto value = object->getInteger("n")) item->number = *value;
    if (auto value = object->getInteger("iteration")) item->iteration = *value;
    if (auto value = object->getString("id")) item->id = value->str();
    if (auto value = object->getString("suite")) item->suite = value->str();
    if (auto value = object->getString("variant")) item->variant = value->str();
    if (auto value = object->getString("file")) item->file = value->str();
    if (auto value = object->getInteger("line")) item->source_line = *value;

    auto *schema = object->getObject("schema");
    auto *codes = object->getArray("codes");
    if (!schema || !codes) {
        static constexpr std::array fields{"error_module", "error_kind", "error_subkind", "error_message"};
        bool error = true;
        for (auto field : fields)
            error &= object->getString(field).has_value();
        if (!error) {
            throw std::runtime_error(
                std::format("bundle line {} is missing runnable schema or code data", line_number));
        }
        item->structured_error = true;
        item->error_label = std::format("{}/{}/{}", required_string(*object, "error_module", line_number),
                                        required_string(*object, "error_kind", line_number),
                                        required_string(*object, "error_subkind", line_number));
        return item;
    }

    auto *args = schema->getArray("args");
    if (!args) { throw std::runtime_error(std::format("bundle line {} schema is missing args", line_number)); }
    for (const auto &arg_value : *args) {
        auto *arg = arg_value.getAsObject();
        if (!arg) { throw std::runtime_error(std::format("bundle line {} has a non-object argument", line_number)); }
        item->args.push_back(ArgumentInfo{parse_dtype(required_string(*arg, "dtype", line_number), line_number),
                                          parse_kind(required_string(*arg, "kind", line_number), line_number)});
    }

    for (const auto &code_value : *codes) {
        auto *code = code_value.getAsObject();
        if (!code) { throw std::runtime_error(std::format("bundle line {} has a non-object code entry", line_number)); }
        std::string name = required_string(*code, "name", line_number);
        if (name == "py") {
            ++item->skipped_python;
            continue;
        }
        if (name == "asmjit_asm" || name == "asmjit_asm_s") { continue; }
        std::optional<Backend> backend;
        if (name == "cpp" || name == "cpp_s") backend = Backend::Cpp;
        if (name == "llvm" || name == "llvm_s") backend = Backend::Llvm;
        if (name == "asmjit" || name == "asmjit_s") backend = Backend::Asmjit;
        if (!backend) { continue; }
        Implementation impl;
        impl.backend = *backend;
        impl.bundle_name = std::move(name);
        impl.code = required_string(*code, "code", line_number);
        impl.comparison_unstable = code->getBoolean("comparison_unstable").value_or(false);
        item->implementations.push_back(std::move(impl));
    }
    for (size_t i = 0; i < item->implementations.size(); ++i) {
        item->implementations[i].symbol = unique_symbol(*item, item->implementations[i], i);
    }
    return item;
}

bool case_selected(const BundleCase &item, const RunnerOptions &options) {
    if (options.only_n && item.number != *options.only_n) return false;
    if (item.number >= 0 && item.number < options.first) return false;
    if (!options.suites.empty() && !options.suites.contains(item.suite)) return false;
    return true;
}

} // namespace simjit::local_runner
