// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/compiler.h"
#include "simjit/core/hir.h"
#include "simjit/simjit.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#if !SIMJIT_ENABLE_SERIALIZATION
#error "simjit-cli requires SIMJIT_ENABLE_SERIALIZATION"
#endif

simjit::MemoryArena arena{};
simjit::Context opts{arena, "expr", simjit::CodeTransformations::All, simjit::Arch::Native};

struct CliOptions {
    std::string input{};
    bool is_scalar = false;
    bool print_hir = false;
    bool print_mir = false;
    bool print_serialized = false;
    bool output_cpp = false;
    bool output_llvm = false;
    bool output_asmjit = false;
    bool dump_json = false;
    std::string json_path{};
};

static CliOptions cli_opts{};

struct OutputSection {
    std::string title;
    std::string content;
};

static void print_sections(const std::vector<OutputSection> &sections) {
    if (sections.empty()) { return; }

    if (sections.size() == 1) {
        printf("%s\n", sections[0].content.c_str());
        return;
    }

    for (const auto &section : sections) {
        printf("=== %s ===\n%s\n", section.title.c_str(), section.content.c_str());
    }
}

static std::string read_file(std::string_view path) {
    std::ifstream f{std::string(path)};
    if (!f) { throw std::runtime_error("failed to open input file '" + std::string(path) + "'"); }
    return std::string{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}

static simjit::Arch parse_arch(std::string_view value) {
    if (value == "native" || value == "auto") { return simjit::Arch::Native; }
    if (value == "amd64-avx512" || value == "x86" || value == "x86-64" || value == "x64" || value == "avx512") {
        return simjit::Arch::Amd64_AVX512;
    }
    if (value == "amd64-avx512-ymm" || value == "avx512-ymm" || value == "x86-ymm") {
        return simjit::Arch::Amd64_AVX512_YMM;
    }
    if (value == "arm64-neon" || value == "arm64" || value == "aarch64" || value == "arm" || value == "neon") {
        return simjit::Arch::Arm64_NEON;
    }
    throw std::runtime_error(std::string("unknown architecture '") + std::string(value) +
                             "'; expected one of: native, x86, x86-ymm, arm");
}

static void print_usage(FILE *out, const char *argv0) {
    fprintf(out,
            "Usage: %s (-s HIR | --serialized-string HIR | --serialized-file PATH) [options]\n"
            "\n"
            "Input:\n"
            "  -s, --serialized-string HIR\n"
            "                            Serialized HIR s-expression from --serialized or a bug report\n"
            "      --serialized-file PATH\n"
            "                            Read serialized HIR s-expression from a file\n"
            "      --name SYMBOL        Symbol name for generated code (default: expr)\n"
            "\n"
            "Compilation mode:\n"
            "      --scalar             Force scalar lowering\n"
            "      --arch ARCH          Target architecture: native, x86, x86-ymm, arm\n"
            "\n"
            "Argument formats:\n"
            "      --cpp                Print generated C++\n"
            "      --llvm               Print generated LLVM IR\n"
            "      --asmjit             Print generated asmjit assembly\n"
            "      --serialized         Print serialized HIR s-expression\n"
            "      --dump-json PATH     Write one tests.jsonl-compatible record; use '-' for stdout\n"
            "\n"
            "IR dumps:\n"
            "      --print-hir          Print readable HIR\n"
            "      --print-mir          Print readable MIR\n"
            "\n"
            "General:\n"
            "  -h, --help               Show this help message\n"
            "\n"
            "If no output format is selected explicitly, the CLI prints C++ by default.\n",
            argv0);
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
            if (std::isprint(c) != 0) {
                result += static_cast<char>(c);
                break;
            }
            throw std::runtime_error("cannot encode non-printable byte in JSON output");
        }
    }

    return result;
}

static const std::string base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

static std::string base64_encode(const uint8_t *buf, size_t buf_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    uint8_t char_array_3[3]{};
    uint8_t char_array_4[4]{};

    while (buf_len-- != 0) {
        char_array_3[i++] = *(buf++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = static_cast<uint8_t>(((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4));
            char_array_4[2] = static_cast<uint8_t>(((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6));
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; i < 4; ++i) {
                ret += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i != 0) {
        for (j = i; j < 3; ++j) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = static_cast<uint8_t>(((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4));
        char_array_4[2] = static_cast<uint8_t>(((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6));
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

static std::string make_base64(const std::vector<uint8_t> &memory) {
    return base64_encode(memory.data(), memory.size());
}

static const char *bundle_variant_name() {
    if (simjit::is_x86_arch(opts.arch)) { return cli_opts.is_scalar ? "x86-scalar" : "x86-vector"; }
    return cli_opts.is_scalar ? "arm-scalar" : "arm-vector";
}

static const char *bundle_arg_kind(simjit::ArgumentKind kind) {
    if ((kind & simjit::ArgumentKind::SrcIdxArr) != simjit::ArgumentKind::Undefined) { return "sv"; }
    if ((kind & (simjit::ArgumentKind::SrcGatherArr | simjit::ArgumentKind::SrcArr | simjit::ArgumentKind::SrcConst)) !=
        simjit::ArgumentKind::Undefined) {
        return "in";
    }
    if ((kind & simjit::ArgumentKind::DstSafetyCheck) != simjit::ArgumentKind::Undefined) { return "checked"; }
    if ((kind & simjit::ArgumentKind::DstAgg) != simjit::ArgumentKind::Undefined) { return "outs"; }
    if ((kind & simjit::ArgumentKind::Dst) != simjit::ArgumentKind::Undefined) { return "out"; }
    throw std::runtime_error("invalid argument kind for JSON dump");
}

static std::string schema_json(const simjit::ArenaArray<simjit::ArgumentDecl> &args) {
    std::string result = "{\"args\":[";
    for (size_t i = 0; i < args.size(); ++i) {
        const auto &arg = args[i];
        if (i != 0) { result += ","; }
        result += "{\"dtype\":\"";
        result += simjit::show_scalar_dtype(arg.dtype);
        result += "\",\"kind\":\"";
        result += bundle_arg_kind(arg.kind);
        result += "\"}";
    }
    result += "]}";
    return result;
}

static std::string make_code_obj(std::string_view name, std::string_view code) {
    return std::string("{\"name\":\"") + std::string(name) + "\",\"code\":\"" + std::string(code) + "\"}";
}

static bool wants_any_backend_output() {
    return cli_opts.output_cpp || cli_opts.output_llvm || cli_opts.output_asmjit;
}

static bool should_dump_cpp_code() {
#if SIMJIT_CPP_BACKEND
    return cli_opts.output_cpp || !wants_any_backend_output();
#else
    return false;
#endif
}

static bool should_dump_llvm_code() {
#if SIMJIT_LLVM_BACKEND
    return cli_opts.output_llvm || !wants_any_backend_output();
#else
    return false;
#endif
}

static bool should_dump_asmjit_code() {
#if SIMJIT_ASMJIT_BACKEND
    return cli_opts.output_asmjit || !wants_any_backend_output();
#else
    return false;
#endif
}

static void append_codes_json(const simjit::mir::Function *mir, std::vector<std::string> &codes) {
    const char *suffix = cli_opts.is_scalar ? "_s" : "";
    if (should_dump_cpp_code()) {
#if SIMJIT_CPP_BACKEND
        codes.push_back(make_code_obj(std::string("cpp") + suffix, escape_json(simjit::emit_cpp_source(mir))));
#endif
    }
    if (should_dump_llvm_code()) {
#if SIMJIT_LLVM_BACKEND
        codes.push_back(make_code_obj(std::string("llvm") + suffix, escape_json(simjit::emit_llvm_ir(mir))));
#endif
    }
    if (should_dump_asmjit_code()) {
#if SIMJIT_ASMJIT_BACKEND
        simjit::AsmjitCompileOptions ajopts{true, true};
        simjit::AsmjitCompileResult result{};
        simjit::compile_asmjit(mir, ajopts, result);
        codes.push_back(make_code_obj(std::string("asmjit") + suffix, make_base64(result.machine_code)));
        codes.push_back(make_code_obj(std::string("asmjit_asm") + suffix, escape_json(result.asm_code)));
#endif
    }
    if (codes.empty()) { throw std::runtime_error("JSON dump requested, but no selected code backend is enabled"); }
}

static std::string build_json_record(const simjit::hir::Function *hir, const simjit::mir::Function *mir,
                                     std::string_view hir_str, std::string_view mir_str,
                                     std::string_view serialized_str) {
    std::vector<std::string> codes{};
    append_codes_json(mir, codes);

    std::string obj = "{";
    obj += "\"n\":0";
    obj += ",\"id\":\"cli:0\"";
    obj += ",\"suite\":\"cli\"";
    obj += ",\"suite_idx\":0";
    obj += ",\"case_idx\":0";
    obj += ",\"variant\":\"";
    obj += bundle_variant_name();
    obj += "\"";
    obj += ",\"expected\":\"pass\"";
    obj += ",\"file\":\"simjit-cli\"";
    obj += ",\"line\":0";
    obj += ",\"schema\":";
    obj += schema_json(hir->args);
    obj += ",\"src\":\"";
    obj += escape_json(hir_str);
    obj += "\"";
    obj += ",\"mir\":\"";
    obj += escape_json(mir_str);
    obj += "\"";
    obj += ",\"serialized\":\"";
    obj += escape_json(serialized_str);
    obj += "\"";
    obj += ",\"hir_time\":0";
    obj += ",\"vectorizer_time\":0";
    obj += ",\"mir_time\":0";
    obj += ",\"asmjit_time\":0";
    obj += ",\"llvm_time\":0";
    obj += ",\"codes\":[";
    for (size_t i = 0; i < codes.size(); ++i) {
        if (i != 0) { obj += ","; }
        obj += codes[i];
    }
    obj += "]}";
    (void)hir;
    return obj;
}

static void write_json_record(std::string_view path, std::string_view record) {
    if (path == "-") {
        printf("%.*s\n", static_cast<int>(record.size()), record.data());
        return;
    }
    std::ofstream f{std::string(path)};
    if (!f) { throw std::runtime_error("failed to open JSON output file '" + std::string(path) + "'"); }
    f << record << '\n';
    if (!f) { throw std::runtime_error("failed to write JSON output file '" + std::string(path) + "'"); }
}

static bool wants_any_explicit_output_format() {
    return cli_opts.output_cpp || cli_opts.output_llvm || cli_opts.output_asmjit || cli_opts.print_serialized ||
           cli_opts.dump_json;
}

static bool wants_cpp_output() {
    if (cli_opts.output_cpp) { return true; }
    if (wants_any_explicit_output_format()) { return false; }
    return true;
}

static bool needs_mir() {
    return cli_opts.print_mir || wants_cpp_output() || cli_opts.output_llvm || cli_opts.output_asmjit ||
           cli_opts.dump_json;
}

static void validate_requested_outputs() {
    if (cli_opts.dump_json && cli_opts.json_path == "-" &&
        (cli_opts.print_hir || cli_opts.print_mir || cli_opts.print_serialized || wants_any_backend_output())) {
        throw std::runtime_error("--dump-json - cannot be combined with other stdout output options");
    }
    if (wants_cpp_output()) {
#if !SIMJIT_CPP_BACKEND
        throw std::runtime_error("C++ output requested, but this binary was built without the C++ backend");
#endif
    }
    if (cli_opts.output_llvm) {
#if !SIMJIT_LLVM_BACKEND
        throw std::runtime_error("LLVM output requested, but this binary was built without the LLVM backend");
#endif
    }
    if (cli_opts.output_asmjit) {
#if !SIMJIT_ASMJIT_BACKEND
        throw std::runtime_error("asmjit output requested, but this binary was built without the asmjit backend");
#endif
    }
}

static void process() {
    simjit::FunctionBuilder builder{opts};
    simjit::deserialize(cli_opts.input, builder);
    auto hir = builder.build();
    std::string hir_str;
    std::string mir_str;
    std::string serialized_str;
    std::vector<OutputSection> sections{};

    validate_requested_outputs();

    if (cli_opts.print_hir || cli_opts.dump_json) { hir_str = simjit::hir::print_function(hir); }
    if (cli_opts.print_serialized || cli_opts.dump_json) { serialized_str = simjit::serialize(hir); }
    if (cli_opts.print_hir) { sections.push_back({"HIR", hir_str}); }
    if (cli_opts.print_serialized) { sections.push_back({"Serialized HIR", serialized_str}); }
    if (!needs_mir()) {
        print_sections(sections);
        return;
    }

    simjit::mir::Function *mir = cli_opts.is_scalar ? simjit::lower_scalar(hir) : simjit::lower_vectorized(hir);

    if (cli_opts.print_mir || cli_opts.dump_json) { mir_str = simjit::mir::print_function(mir); }
    if (cli_opts.print_mir) { sections.push_back({"MIR", mir_str}); }
    if (wants_cpp_output()) {
#if SIMJIT_CPP_BACKEND
        sections.push_back({"C++", simjit::emit_cpp_source(mir)});
#endif
    }
    if (cli_opts.output_llvm) {
#if SIMJIT_LLVM_BACKEND
        sections.push_back({"LLVM IR", simjit::emit_llvm_ir(mir)});
#endif
    }
    if (cli_opts.output_asmjit) {
#if SIMJIT_ASMJIT_BACKEND
        simjit::AsmjitCompileOptions ajopts{false, true};
        simjit::AsmjitCompileResult result{};
        simjit::compile_asmjit(mir, ajopts, result);
        sections.push_back({"AsmJit", result.asm_code});
#endif
    }
    if (cli_opts.dump_json) {
        write_json_record(cli_opts.json_path, build_json_record(hir, mir, hir_str, mir_str, serialized_str));
    }

    print_sections(sections);
}

static std::string_view require_value(int argc, char **argv, int &cursor, std::string_view opt) {
    ++cursor;
    if (cursor >= argc) {
        throw std::runtime_error(std::string("option '") + std::string{opt} +
                                 "' requires a value\n\nTry --help' for usage.");
    }
    return argv[cursor];
}

static void parse_cli_opts(int argc, char **argv) {
    int cursor = 1;
    while (cursor < argc) {
        std::string_view opt = argv[cursor];
        if (opt == "-s" || opt == "--serialized-string" || opt == "--serialized-input") {
            if (!cli_opts.input.empty()) { throw std::runtime_error("only one input option can be specified"); }
            cli_opts.input = require_value(argc, argv, cursor, opt);
        } else if (opt == "--serialized-file") {
            if (!cli_opts.input.empty()) { throw std::runtime_error("only one input option can be specified"); }
            cli_opts.input = read_file(require_value(argc, argv, cursor, opt));
        } else if (opt == "--name") {
            opts.symbol_name = require_value(argc, argv, cursor, opt);
        } else if (opt == "--arch") {
            opts.arch = parse_arch(require_value(argc, argv, cursor, opt));
        } else if (opt == "--scalar") {
            cli_opts.is_scalar = true;
        } else if (opt == "--print-mir") {
            cli_opts.print_mir = true;
        } else if (opt == "--print-hir") {
            cli_opts.print_hir = true;
        } else if (opt == "--serialized") {
            cli_opts.print_serialized = true;
        } else if (opt == "--dump-json") {
            cli_opts.dump_json = true;
            cli_opts.json_path = require_value(argc, argv, cursor, opt);
        } else if (opt == "--cpp") {
            cli_opts.output_cpp = true;
        } else if (opt == "--llvm") {
#if SIMJIT_LLVM_BACKEND
            cli_opts.output_llvm = true;
#else
            throw std::runtime_error("this binary was built without LLVM backend support");
#endif
        } else if (opt == "--asmjit") {
#if SIMJIT_ASMJIT_BACKEND
            cli_opts.output_asmjit = true;
#else
            throw std::runtime_error("this binary was built without asmjit backend support");
#endif
        } else if (opt == "-h" || opt == "--help") {
            print_usage(stdout, argv[0]);
            exit(0);
        } else {
            throw std::runtime_error(
                std::string("unknown option '" + std::string{opt} + "'\n\nTry '--help' for usage."));
        }

        ++cursor;
    }

    if (cli_opts.input.empty()) {
        throw std::runtime_error(
            std::string("missing required --serialized-string argument\n\nTry '--help' for usage."));
    }
}

int main(int argc, char **argv) {
    try {
        parse_cli_opts(argc, argv);
    } catch (const std::exception &e) {
        fprintf(stderr, "error parsing CLI arguments: %s\n", e.what());
        return 1;
    }

    try {
        process();
    } catch (const std::exception &e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }

    return 0;
}
