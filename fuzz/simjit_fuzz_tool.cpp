// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "headerlisp.h"
#include "simjit/compiler.h"
#include "simjit/core/hir.h"
#include "simjit/core/mir.h"
#include "simjit/simjit.h"

#include <array>
#include <charconv>
#include <csignal>
#include <cstdlib>
#include <format>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace simjit;
namespace hl = headerlisp;

namespace {

enum class Preset : uint8_t {
    BuilderValidate,
    CompileValidate,
    CompileObserve,
    ScalarDiff,
    VectorObserve,
    VectorStrict,
};

struct Metadata {
    int64_t base_seed = -1;
    int64_t program_index = -1;
    int64_t program_seed = -1;
    std::string profile = "default";
    std::string preset = "scalar-diff";
    std::string arch = "native";
    Arch target_arch = Arch::Native;
};

struct CodeBlob {
    std::string name;
    std::string code;
    bool comparison_unstable = false;
};

struct EmitResult {
    bool ok = false;
    std::string code;
    std::string asm_code;
    std::string error;
    std::optional<ErrorInfo> error_info;
};

struct CompileState {
    Metadata meta;
    bool hir_scalar_only = false;
    std::string serialized;
    std::string scalar_status = "failed";
    std::string vector_status = "not-requested";
    std::string stage_error;
    std::string vector_lower_error;
    EmitResult asmjit_scalar;
    EmitResult llvm_scalar;
    EmitResult cpp_scalar;
    EmitResult asmjit_vector;
    EmitResult llvm_vector;
    EmitResult cpp_vector;
    std::string schema = R"({"args": []})";
    std::optional<ErrorInfo> stage_error_info;
    std::optional<ErrorInfo> vector_lower_error_info;
    std::vector<CodeBlob> codes;
    bool comparison_unstable = false;
};

struct ParseResult {
    std::string subcommand;
    Preset preset = Preset::ScalarDiff;
    Metadata meta;
    bool wait_for_debugger = false;
    bool emit_cpp = false;
    std::vector<int> roots;
};

static const std::string kBase64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789+/";

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
            if (c >= 0x20 && c <= 0x7e) {
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
                ret += kBase64Chars[char_array_4[i]];
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
            ret += kBase64Chars[char_array_4[j]];
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

template <typename T> static std::string enum_hex(T value) {
    using U = std::underlying_type_t<T>;
    return std::format("0x{:x}", static_cast<unsigned long long>(static_cast<U>(value)));
}

static std::string exception_message(const std::exception &e, std::optional<ErrorInfo> &info) {
    if (auto *simjit_error = dynamic_cast<const SimjitException *>(&e)) {
        info = simjit_error->info();
    } else {
        info.reset();
    }
    return e.what();
}

static std::string schema_json(nonstd::span<ArgumentDecl> args) {
    std::string result = "{ \"args\": [";
    for (size_t i = 0; i < args.size(); ++i) {
        const auto &arg = args[i];
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
        result += std::format("{{ \"dtype\": \"{}\", \"kind\": \"{}\"}}", show_scalar_dtype(arg.dtype), kind);
        if (i + 1 != args.size()) { result += ","; }
    }
    result += "]}";
    return result;
}

static std::string read_stdin() {
    std::istreambuf_iterator<char> begin(std::cin);
    std::istreambuf_iterator<char> end;
    return std::string(begin, end);
}

static Preset parse_preset(std::string_view arg) {
    if (arg == "builder-validate") return Preset::BuilderValidate;
    if (arg == "compile-validate") return Preset::CompileValidate;
    if (arg == "compile-observe") return Preset::CompileObserve;
    if (arg == "scalar-diff") return Preset::ScalarDiff;
    if (arg == "vector-observe") return Preset::VectorObserve;
    if (arg == "vector-strict") return Preset::VectorStrict;
    throw std::runtime_error(std::format("invalid preset '{}'", arg));
}

static std::vector<int> parse_root_list(std::string_view value) {
    std::vector<int> roots;
    size_t start = 0;
    while (start <= value.size()) {
        size_t end = value.find(',', start);
        if (end == std::string_view::npos) { end = value.size(); }
        std::string_view token = value.substr(start, end - start);
        while (!token.empty() && token.front() == ' ')
            token.remove_prefix(1);
        while (!token.empty() && token.back() == ' ')
            token.remove_suffix(1);
        if (token.empty()) { throw std::runtime_error("empty entry in --roots"); }
        int parsed = 0;
        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), parsed);
        if (ec != std::errc() || ptr != token.data() + token.size() || parsed < 0) {
            throw std::runtime_error(std::format("invalid root id '{}'", token));
        }
        roots.push_back(parsed);
        if (end == value.size()) { break; }
        start = end + 1;
    }
    if (roots.empty()) { throw std::runtime_error("--roots must not be empty"); }
    std::unordered_set<int> uniq;
    for (int root : roots) {
        if (!uniq.insert(root).second) { throw std::runtime_error(std::format("duplicate root id {}", root)); }
    }
    return roots;
}

static int64_t parse_i64(std::string_view name, std::string_view value) {
    int64_t parsed = 0;
    auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc() || ptr != value.data() + value.size()) {
        throw std::runtime_error(std::format("invalid value for {}: '{}'", name, value));
    }
    return parsed;
}

static Arch parse_arch(std::string_view value) {
    if (value == "native") return Arch::Native;
    if (value == "x86") return Arch::Amd64_AVX512;
    if (value == "arm") return Arch::Arm64_NEON;
    throw std::runtime_error(std::format("invalid arch '{}'", value));
}

static ParseResult parse_args(int argc, char **argv) {
    if (argc < 2) { throw std::runtime_error("usage: simjit-fuzz-tool <canonicalize|compile|minimize> [options]"); }

    ParseResult result;
    result.subcommand = argv[1];
    if (result.subcommand == "canonicalize") {
        if (argc != 2) { throw std::runtime_error("canonicalize takes no options"); }
        return result;
    }
    if (result.subcommand == "minimize") {
        for (int i = 2; i < argc; ++i) {
            std::string_view arg = argv[i];
            if (arg != "--roots") { throw std::runtime_error(std::format("unknown option '{}'", arg)); }
            if (!result.roots.empty()) { throw std::runtime_error("--roots specified more than once"); }
            if (i + 1 >= argc) { throw std::runtime_error("missing value for --roots"); }
            result.roots = parse_root_list(argv[++i]);
        }
        if (result.roots.empty()) { throw std::runtime_error("minimize requires --roots"); }
        return result;
    }
    if (result.subcommand == "produce") { result.subcommand = "compile"; }
    if (result.subcommand != "compile") {
        throw std::runtime_error(std::format("unknown subcommand '{}'", result.subcommand));
    }

    for (int i = 2; i < argc; ++i) {
        std::string_view arg = argv[i];
        auto require_value = [&](std::string_view flag) -> std::string_view {
            if (i + 1 >= argc) { throw std::runtime_error(std::format("missing value for {}", flag)); }
            ++i;
            return argv[i];
        };

        if (arg == "--preset") {
            std::string_view value = require_value(arg);
            result.preset = parse_preset(value);
            result.meta.preset = std::string(value);
        } else if (arg == "--base-seed") {
            result.meta.base_seed = parse_i64(arg, require_value(arg));
        } else if (arg == "--program-index") {
            result.meta.program_index = parse_i64(arg, require_value(arg));
        } else if (arg == "--program-seed") {
            result.meta.program_seed = parse_i64(arg, require_value(arg));
        } else if (arg == "--profile") {
            result.meta.profile = std::string(require_value(arg));
        } else if (arg == "--arch") {
            std::string_view value = require_value(arg);
            result.meta.arch = std::string(value);
            result.meta.target_arch = parse_arch(value);
        } else if (arg == "--wait-for-debugger") {
            result.wait_for_debugger = true;
        } else if (arg == "--cpp") {
            result.emit_cpp = true;
        } else {
            throw std::runtime_error(std::format("unknown option '{}'", arg));
        }
    }

    return result;
}

struct ArgDecl {
    int id;
    hl::value dtype;
    hl::value kind;
};

static ArgDecl parse_arg_decl(hl::value value) {
    auto [id, dtype, kind] = hl::require_match("arg declaration", value, "arg", hl::cap_as<int>, hl::cap, hl::cap);
    return ArgDecl{id, dtype, kind};
}

struct AccDecl {
    int id;
    hl::value dtype;
    hl::value dst_arg;
    hl::value step_ref;
};

static AccDecl parse_acc_decl(hl::value value) {
    auto [id, dtype, dst_arg, step_ref] =
        hl::require_match("acc declaration", value, "acc", hl::cap_as<int>, hl::cap, hl::cap, hl::cap);
    return AccDecl{id, dtype, dst_arg, step_ref};
}

struct StepRecord {
    int id;
    hl::value kind;
    std::string_view kind_name;
    hl::value dtype;
    hl::value payload;
};

static StepRecord parse_step_record(hl::value step) {
    auto [id, kind, dtype, payload] =
        hl::require_match("step record", step, "step", hl::cap_as<int>, hl::cap, hl::cap, hl::rest);
    return StepRecord{id, kind, kind.as_string_view(), dtype, payload};
}

static int parse_named_ref(hl::value value, const char *expected_name) {
    auto [idx] = hl::require_match(expected_name, value, expected_name, hl::cap_as<int>);
    return idx;
}

static bool arg_decl_is_checked(hl::value arg_decl) {
    return parse_arg_decl(arg_decl).kind.as_string_view() == "dst-checked";
}

struct StepUseInfo {
    std::vector<int> child_steps;
    std::vector<int> arg_refs;
    std::vector<int> acc_refs;
    bool needs_safety_check_arg = false;
};

static StepUseInfo collect_step_use_info(hl::value step) {
    StepUseInfo result;

    StepRecord record = parse_step_record(step);
    std::string_view kind = record.kind_name;
    hl::value payload = record.payload;

    auto add_step = [&](hl::value ref) { result.child_steps.push_back(parse_named_ref(ref, "step")); };
    auto add_arg = [&](hl::value ref) { result.arg_refs.push_back(parse_named_ref(ref, "arg")); };
    auto add_acc = [&](hl::value ref) { result.acc_refs.push_back(parse_named_ref(ref, "acc")); };
#define bind_step(...) hl::require_match(hl::print(step).c_str(), payload, __VA_ARGS__)

    if (kind == "const" || kind == "index") { return result; }
    if (kind == "load") {
        auto [arg] = bind_step(hl::cap, hl::_);
        add_arg(arg);
        return result;
    }
    if (kind == "load-splat") {
        auto [arg] = bind_step(hl::cap);
        add_arg(arg);
        return result;
    }
    if (kind == "gather") {
        auto [idx, arg] = bind_step(hl::cap, hl::cap);
        add_step(idx);
        add_arg(arg);
        return result;
    }
    if (kind == "binary") {
        auto [left, right, flags] = bind_step(hl::_, hl::cap, hl::cap, hl::maybe);
        add_step(left);
        add_step(right);
        if (flags) {
            hl::value x = flags.value();
            if (hl::is_string(x)) { x = hl::list(x); }
            for (auto it : x.iter()) {
                if (it.as_string_view() == "safety") {
                    result.needs_safety_check_arg = true;
                    break;
                }
            }
        }
        return result;
    }
    if (kind == "unary") {
        auto [arg, checked] = bind_step(hl::_, hl::cap, hl::or_nil_as<bool>);
        add_step(arg);
        if (checked) { result.needs_safety_check_arg = true; }
        return result;
    }
    if (kind == "cmp") {
        auto [left, right, flag] = bind_step(hl::_, hl::cap, hl::cap, hl::maybe);
        add_step(left);
        add_step(right);
        (void)flag;
        return result;
    }
    if (kind == "int-cast") {
        auto [arg, checked] = bind_step(hl::_, hl::cap, hl::or_nil_as<bool>);
        add_step(arg);
        if (checked) { result.needs_safety_check_arg = true; }
        return result;
    }
    if (kind == "float-cast") {
        auto [arg, flag] = bind_step(hl::cap, hl::maybe);
        add_step(arg);
        (void)flag;
        return result;
    }
    if (kind == "bitcast" || kind == "predicate-not") {
        auto [arg] = bind_step(hl::cap);
        add_step(arg);
        return result;
    }
    if (kind == "permute") {
        auto [arg] = bind_step(hl::cap, hl::_, hl::_);
        add_step(arg);
        return result;
    }
    if (kind == "fpclass") {
        auto [arg] = bind_step(hl::cap, hl::_);
        add_step(arg);
        return result;
    }
    if (kind == "predicate-binary") {
        auto [left, right] = bind_step(hl::_, hl::cap, hl::cap);
        add_step(left);
        add_step(right);
        return result;
    }
    if (kind == "select") {
        auto [cond, truthy, falsy] = bind_step(hl::cap, hl::cap, hl::cap);
        add_step(cond);
        add_step(truthy);
        add_step(falsy);
        return result;
    }
    if (kind == "store") {
        auto [arg, dst, cond] = bind_step(hl::cap, hl::cap, hl::_, hl::maybe);
        add_step(arg);
        add_arg(dst);
        if (cond) { add_step(cond.value()); }
        return result;
    }
    if (kind == "scatter") {
        auto [arg, idx, dst, cond] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        add_step(arg);
        add_step(idx);
        add_arg(dst);
        if (cond) { add_step(cond.value()); }
        return result;
    }
    if (kind == "pack") {
        auto [arg, cond, dst, dst_size] = bind_step(hl::cap, hl::cap, hl::cap, hl::cap);
        add_step(arg);
        add_step(cond);
        add_arg(dst);
        add_arg(dst_size);
        return result;
    }
    if (kind == "acc-arith-bin" || kind == "sum128") {
        auto [arg, acc, cond] = bind_step(hl::_, hl::cap, hl::cap, hl::maybe);
        add_step(arg);
        add_acc(acc);
        if (cond) { add_step(cond.value()); }
        return result;
    }
    if (kind == "countif") {
        auto [arg, acc] = bind_step(hl::cap, hl::cap);
        add_step(arg);
        add_acc(acc);
        return result;
    }
    if (kind == "acc-predicate-bin") {
        auto [arg, acc] = bind_step(hl::_, hl::cap, hl::cap);
        add_step(arg);
        add_acc(acc);
        return result;
    }

    throw std::runtime_error(std::format("unsupported step kind '{}'", kind));
}

static hl::value rewrite_step(hl::value step, const std::unordered_map<int, int> &step_id_map,
                              const std::unordered_map<int, int> &arg_id_map,
                              const std::unordered_map<int, int> &acc_id_map) {
    StepRecord record = parse_step_record(step);
    int old_id = record.id;
    hl::value kind = record.kind;
    std::string_view kind_str = record.kind_name;
    hl::value dtype = record.dtype;
    hl::value payload = record.payload;

    auto step_ref = [&](hl::value ref) { return hl::list("step", step_id_map.at(parse_named_ref(ref, "step"))); };
    auto arg_ref = [&](hl::value ref) { return hl::list("arg", arg_id_map.at(parse_named_ref(ref, "arg"))); };
    auto acc_ref = [&](hl::value ref) { return hl::list("acc", acc_id_map.at(parse_named_ref(ref, "acc"))); };
    auto step_form = [&](auto &&...payload_items) {
        return hl::list_dot("step", step_id_map.at(old_id), kind, dtype,
                            hl::list(std::forward<decltype(payload_items)>(payload_items)...));
    };

    if (kind_str == "const") {
        auto [value] = bind_step(hl::cap);
        return step_form(value);
    }
    if (kind_str == "load") {
        auto [arg, load_kind] = bind_step(hl::cap, hl::cap);
        return step_form(arg_ref(arg), load_kind);
    }
    if (kind_str == "load-splat") {
        auto [arg] = bind_step(hl::cap);
        return step_form(arg_ref(arg));
    }
    if (kind_str == "index") { return step_form(); }
    if (kind_str == "gather") {
        auto [idx, arg] = bind_step(hl::cap, hl::cap);
        return step_form(step_ref(idx), arg_ref(arg));
    }
    if (kind_str == "binary") {
        auto [op, left, right, flags] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        if (flags) { return step_form(op, step_ref(left), step_ref(right), *flags); }
        return step_form(op, step_ref(left), step_ref(right));
    }
    if (kind_str == "unary") {
        auto [op, arg, checked] = bind_step(hl::cap, hl::cap, hl::maybe);
        if (checked) { return step_form(op, step_ref(arg), *checked); }
        return step_form(op, step_ref(arg));
    }
    if (kind_str == "cmp") {
        auto [op, left, right, flag] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        if (flag) { return step_form(op, step_ref(left), step_ref(right), *flag); }
        return step_form(op, step_ref(left), step_ref(right));
    }
    if (kind_str == "int-cast") {
        auto [cast_kind, arg, checked] = bind_step(hl::cap, hl::cap, hl::maybe);
        if (checked) { return step_form(cast_kind, step_ref(arg), *checked); }
        return step_form(cast_kind, step_ref(arg));
    }
    if (kind_str == "float-cast") {
        auto [arg, flag] = bind_step(hl::cap, hl::maybe);
        if (flag) { return step_form(step_ref(arg), *flag); }
        return step_form(step_ref(arg));
    }
    if (kind_str == "bitcast" || kind_str == "predicate-not") {
        auto [arg] = bind_step(hl::cap);
        return step_form(step_ref(arg));
    }
    if (kind_str == "permute") {
        auto [arg, perm, bit] = bind_step(hl::cap, hl::cap, hl::cap);
        return step_form(step_ref(arg), perm, bit);
    }
    if (kind_str == "fpclass") {
        auto [arg, fp_flags] = bind_step(hl::cap, hl::cap);
        return step_form(step_ref(arg), fp_flags);
    }
    if (kind_str == "predicate-binary") {
        auto [op, left, right] = bind_step(hl::cap, hl::cap, hl::cap);
        return step_form(op, step_ref(left), step_ref(right));
    }
    if (kind_str == "select") {
        auto [cond, truthy, falsy] = bind_step(hl::cap, hl::cap, hl::cap);
        return step_form(step_ref(cond), step_ref(truthy), step_ref(falsy));
    }
    if (kind_str == "store") {
        auto [arg, dst, store_kind, cond] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        if (cond) { return step_form(step_ref(arg), arg_ref(dst), store_kind, step_ref(*cond)); }
        return step_form(step_ref(arg), arg_ref(dst), store_kind);
    }
    if (kind_str == "scatter") {
        auto [arg, idx, dst, cond] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        if (cond) { return step_form(step_ref(arg), step_ref(idx), arg_ref(dst), step_ref(*cond)); }
        return step_form(step_ref(arg), step_ref(idx), arg_ref(dst));
    }
    if (kind_str == "pack") {
        auto [arg, cond, dst, dst_size] = bind_step(hl::cap, hl::cap, hl::cap, hl::cap);
        return step_form(step_ref(arg), step_ref(cond), arg_ref(dst), arg_ref(dst_size));
    }
    if (kind_str == "acc-arith-bin" || kind_str == "sum128") {
        auto [op, arg, acc, cond] = bind_step(hl::cap, hl::cap, hl::cap, hl::maybe);
        if (cond) { return step_form(op, step_ref(arg), acc_ref(acc), step_ref(*cond)); }
        return step_form(op, step_ref(arg), acc_ref(acc));
    }
    if (kind_str == "countif") {
        auto [arg, acc] = bind_step(hl::cap, hl::cap);
        return step_form(step_ref(arg), acc_ref(acc));
    }
    if (kind_str == "acc-predicate-bin") {
        auto [op, arg, acc] = bind_step(hl::cap, hl::cap, hl::cap);
        return step_form(op, step_ref(arg), acc_ref(acc));
    }

    throw std::runtime_error(std::format("unsupported step kind '{}'", kind_str));
}

static std::string minimize_item(std::string_view input, const ParseResult &config) {
    hl::context_guard guard{};
    hl::value func = hl::read(input);
    auto [body] = hl::require_match("func", func, "func", hl::rest);

    hl::value args = hl::assoc_ref("args", body);
    hl::value steps = hl::assoc_ref("steps", body);
    hl::value roots = hl::assoc_ref("roots", body);
    hl::value accs = hl::assoc_ref("accs", body);
    bool has_scalar_only = !!hl::assoc("scalar-only", body);

    if (!args || !steps || !roots) {
        throw std::runtime_error("serialized function must contain args, steps, and roots sections");
    }

    std::unordered_map<int, hl::value> step_map;
    std::unordered_map<int, hl::value> arg_map;
    std::unordered_map<int, hl::value> acc_map;
    std::vector<int> step_order;
    std::vector<int> arg_order;
    std::vector<int> acc_order;
    std::unordered_set<int> actual_roots;

    for (auto arg : args.iter()) {
        ArgDecl decl = parse_arg_decl(arg);
        arg_map[decl.id] = arg;
        arg_order.push_back(decl.id);
    }
    for (auto acc : accs.iter()) {
        AccDecl decl = parse_acc_decl(acc);
        acc_map[decl.id] = acc;
        acc_order.push_back(decl.id);
    }
    for (auto step : steps.iter()) {
        StepRecord record = parse_step_record(step);
        step_map[record.id] = step;
        step_order.push_back(record.id);
    }
    for (auto root : roots.iter()) {
        actual_roots.insert(parse_named_ref(root, "step"));
    }

    for (int root : config.roots) {
        if (!actual_roots.contains(root)) {
            throw std::runtime_error(std::format("step {} is not listed in (roots ...)", root));
        }
        if (!step_map.contains(root)) {
            throw std::runtime_error(std::format("root step {} not found in steps", root));
        }
    }

    std::unordered_set<int> keep_steps;
    std::unordered_set<int> keep_args;
    std::unordered_set<int> keep_accs;
    bool needs_safety_check_arg = false;
    std::vector<int> stack(config.roots.begin(), config.roots.end());

    while (!stack.empty()) {
        int id = stack.back();
        stack.pop_back();
        if (!keep_steps.insert(id).second) { continue; }
        auto it = step_map.find(id);
        if (it == step_map.end()) { throw std::runtime_error(std::format("step {} not found in steps", id)); }
        StepUseInfo info = collect_step_use_info(it->second);
        for (int child : info.child_steps) {
            stack.push_back(child);
        }
        for (int arg : info.arg_refs) {
            keep_args.insert(arg);
        }
        for (int acc : info.acc_refs) {
            keep_accs.insert(acc);
        }
        needs_safety_check_arg = needs_safety_check_arg || info.needs_safety_check_arg;
    }

    for (int acc_id : keep_accs) {
        auto it = acc_map.find(acc_id);
        if (it == acc_map.end()) { throw std::runtime_error(std::format("acc {} not found in accs", acc_id)); }
        AccDecl decl = parse_acc_decl(it->second);
        keep_args.insert(parse_named_ref(decl.dst_arg, "arg"));
        keep_steps.insert(parse_named_ref(decl.step_ref, "step"));
    }

    if (needs_safety_check_arg) {
        for (int arg_id : arg_order) {
            if (arg_decl_is_checked(arg_map.at(arg_id))) { keep_args.insert(arg_id); }
        }
    }

    std::unordered_map<int, int> step_id_map;
    std::unordered_map<int, int> arg_id_map;
    std::unordered_map<int, int> acc_id_map;
    std::vector<int> kept_step_order;
    std::vector<int> kept_arg_order;
    std::vector<int> kept_acc_order;

    for (int arg_id : arg_order) {
        if (keep_args.contains(arg_id)) {
            arg_id_map[arg_id] = static_cast<int>(kept_arg_order.size());
            kept_arg_order.push_back(arg_id);
        }
    }
    for (int acc_id : acc_order) {
        if (keep_accs.contains(acc_id)) {
            acc_id_map[acc_id] = static_cast<int>(kept_acc_order.size());
            kept_acc_order.push_back(acc_id);
        }
    }
    for (int step_id : step_order) {
        if (keep_steps.contains(step_id)) {
            step_id_map[step_id] = static_cast<int>(kept_step_order.size());
            kept_step_order.push_back(step_id);
        }
    }

    hl::list_builder func_fields;
    {
        hl::list_builder b{};
        for (int arg_id : kept_arg_order) {
            ArgDecl decl = parse_arg_decl(arg_map.at(arg_id));
            b.add(hl::list("arg", arg_id_map.at(arg_id), decl.dtype, decl.kind));
        }
        func_fields.add(hl::cons("args", b.list()));
    }

    if (!kept_acc_order.empty()) {
        hl::list_builder b{};
        for (int acc_id : kept_acc_order) {
            AccDecl decl = parse_acc_decl(acc_map.at(acc_id));
            hl::value x = hl::list("acc", acc_id_map.at(acc_id), decl.dtype,
                                   hl::list("arg", arg_id_map.at(parse_named_ref(decl.dst_arg, "arg"))),
                                   hl::list("step", step_id_map.at(parse_named_ref(decl.step_ref, "step"))));
            b.add(x);
        }
        func_fields.add(hl::cons("accs", b.list()));
    }

    {
        hl::list_builder b{};
        for (int step_id : kept_step_order) {
            b.add(rewrite_step(step_map.at(step_id), step_id_map, arg_id_map, acc_id_map));
        }
        func_fields.add(hl::cons("steps", b.list()));
    }
    {
        hl::list_builder b{};
        for (int root_id : config.roots) {
            b.add(hl::list("step", step_id_map.at(root_id)));
        }
        func_fields.add(hl::cons("roots", b.list()));
    }

    if (has_scalar_only) { func_fields.add(hl::list("scalar-only")); }

    return hl::print(hl::cons("func", func_fields.list()));
}

static void wait_for_debugger_if_requested(const ParseResult &config) {
    if (!config.wait_for_debugger) { return; }

    const auto pid = static_cast<long>(::getpid());
    std::cerr << std::format(
        "simjit-fuzz-tool pid={} waiting for debugger; attach with 'lldb -p {}' and then continue the process\n", pid,
        pid);
    std::cerr.flush();
    ::raise(SIGSTOP);
}

static mir::Function *lower_for_preset(const hir::Function *fn, bool vectorized) {
    return vectorized ? lower_vectorized(fn) : lower_scalar(fn);
}

static EmitResult emit_llvm_code(const mir::Function *mir) {
    EmitResult result;
#ifdef SIMJIT_LLVM_BACKEND
    try {
        result.code = emit_llvm_ir(mir);
        result.ok = true;
    } catch (const std::exception &e) { result.error = exception_message(e, result.error_info); }
#else
    (void)mir;
    result.error = "llvm backend is disabled";
#endif
    return result;
}

static EmitResult emit_cpp_code(const mir::Function *mir) {
    EmitResult result;
#ifdef SIMJIT_CPP_BACKEND
    try {
        result.code = emit_cpp_source(mir);
        result.ok = true;
    } catch (const std::exception &e) { result.error = exception_message(e, result.error_info); }
#else
    (void)mir;
    result.error = "cpp backend is disabled";
#endif
    return result;
}

static EmitResult emit_asmjit_code(const mir::Function *mir) {
    EmitResult result;
#ifdef SIMJIT_ASMJIT_BACKEND
    try {
        AsmjitCompileOptions opts{true, true};
        AsmjitCompileResult compile_result{};
        compile_asmjit(mir, opts, compile_result);
        result.code = make_base64(compile_result.machine_code);
        result.asm_code = compile_result.asm_code;
        result.ok = true;
    } catch (const std::exception &e) { result.error = exception_message(e, result.error_info); }
#else
    (void)mir;
    result.error = "asmjit backend is disabled";
#endif
    return result;
}

static bool has_float_to_int_cast(const hir::Function *fn) {
    // Out-of-range float-to-int is undefined across LLVM IR, C++, and hardware backends. Keep emitting these programs
    // for compile/runtime observation, but do not use them as differential comparison oracles.
    std::vector<uint8_t> visited(fn->step_id_count);
    bool found = false;
    for (hir::Step *root : fn->step_roots) {
        hir::traverse_steps_postorder_unique(root, visited, [&](hir::Step *step) {
            if (!step->is(hir::StepKind::FloatCast)) { return; }
            auto &data = step->step_data<hir::StepKind::FloatCast>();
            if (is_float_dtype(data.arg->dtype) && is_simple_int_dtype(step->dtype)) { found = true; }
        });
    }
    return found;
}

static void maybe_add_code(std::vector<CodeBlob> &codes, std::string_view name, const EmitResult &emit,
                           bool comparison_unstable) {
    if (emit.ok) { codes.push_back(CodeBlob{std::string(name), emit.code, comparison_unstable}); }
}

static void maybe_add_asm_code(std::vector<CodeBlob> &codes, std::string_view name, const EmitResult &emit,
                               bool comparison_unstable) {
    if (emit.ok && !emit.asm_code.empty()) {
        codes.push_back(CodeBlob{std::string(name), emit.asm_code, comparison_unstable});
    }
}

static std::string error_object_json(const CompileState &state) {
    std::vector<std::pair<std::string_view, std::string_view>> items;
    auto maybe_push = [&](std::string_view key, const std::string &value) {
        if (!value.empty()) { items.emplace_back(key, value); }
    };

    maybe_push("stage", state.stage_error);
    maybe_push("vector_lower", state.vector_lower_error);
    maybe_push("asmjit_s", state.asmjit_scalar.error);
    maybe_push("llvm_s", state.llvm_scalar.error);
    maybe_push("cpp_s", state.cpp_scalar.error);
    maybe_push("asmjit", state.asmjit_vector.error);
    maybe_push("llvm", state.llvm_vector.error);
    maybe_push("cpp", state.cpp_vector.error);

    std::string result = "{";
    for (size_t i = 0; i < items.size(); ++i) {
        const auto &[key, value] = items[i];
        result += std::format("\"{}\": \"{}\"", key, escape_json(value));
        if (i + 1 != items.size()) { result += ","; }
    }
    result += "}";
    return result;
}

static std::string error_info_json(const ErrorInfo &info) {
    return std::format("{{\"module\": \"{}\", \"module_code\": \"{}\", \"kind\": \"{}\", \"kind_code\": \"{}\", "
                       "\"subkind\": \"{}\", \"subkind_code\": \"{}\", \"message\": \"{}\", \"verbose\": \"{}\"}}",
                       show_error_module(info.module), enum_hex(info.module), show_error_kind(info.kind),
                       enum_hex(info.kind), show_error_subkind(info.subkind), enum_hex(info.subkind),
                       escape_json(info.message), escape_json(info.verbose()));
}

static std::string error_metadata_json(const CompileState &state) {
    std::vector<std::pair<std::string_view, const ErrorInfo *>> items;
    auto maybe_push = [&](std::string_view key, const std::optional<ErrorInfo> &info) {
        if (info.has_value()) { items.emplace_back(key, &*info); }
    };

    maybe_push("stage", state.stage_error_info);
    maybe_push("vector_lower", state.vector_lower_error_info);
    maybe_push("asmjit_s", state.asmjit_scalar.error_info);
    maybe_push("llvm_s", state.llvm_scalar.error_info);
    maybe_push("cpp_s", state.cpp_scalar.error_info);
    maybe_push("asmjit", state.asmjit_vector.error_info);
    maybe_push("llvm", state.llvm_vector.error_info);
    maybe_push("cpp", state.cpp_vector.error_info);

    std::string result = "{";
    for (size_t i = 0; i < items.size(); ++i) {
        const auto &[key, info] = items[i];
        result += std::format("\"{}\": {}", key, error_info_json(*info));
        if (i + 1 != items.size()) { result += ","; }
    }
    result += "}";
    return result;
}

static std::string codes_json(const std::vector<CodeBlob> &codes) {
    std::string result = "[";
    for (size_t i = 0; i < codes.size(); ++i) {
        const auto &code = codes[i];
        result += std::format("{{\"name\": \"{}\", \"code\": \"{}\"", escape_json(code.name), escape_json(code.code));
        if (code.comparison_unstable) { result += ",\"comparison_unstable\": true"; }
        result += "}";
        if (i + 1 != codes.size()) { result += ","; }
    }
    result += "]";
    return result;
}

static std::string item_id(const Metadata &meta) {
    if (meta.base_seed >= 0 && meta.program_index >= 0) {
        return std::format("xsmith-{}-{}", meta.base_seed, meta.program_index);
    }
    if (meta.program_seed >= 0) { return std::format("xsmith-{}", meta.program_seed); }
    return "xsmith";
}

static std::string build_item_json(const CompileState &state) {
    const char *variant_mode = state.vector_status == "not-requested" ? "scalar" : "vector";
    std::string result = "{";
    result += std::format("\"n\": {}", state.meta.program_index);
    result += ",";
    result += std::format("\"id\": \"{}\"", escape_json(item_id(state.meta)));
    result += ",";
    result += "\"suite\": \"xsmith\"";
    result += ",";
    result += std::format("\"variant\": \"{}-{}\"", escape_json(state.meta.arch), variant_mode);
    result += ",";
    result += "\"expected\": \"pass\"";
    result += ",";
    result += "\"file\": \"fuzz/generated\"";
    result += ",";
    result += "\"line\": 0";
    result += ",";
    result += std::format("\"schema\": {}", state.schema);
    result += ",";
    result += std::format("\"serialized\": \"{}\"", escape_json(state.serialized));
    result += ",";
    result += std::format("\"codes\": {}", codes_json(state.codes));
    result += ",";
    result += std::format("\"base_seed\": {}", state.meta.base_seed);
    result += ",";
    result += std::format("\"program_index\": {}", state.meta.program_index);
    result += ",";
    result += std::format("\"program_seed\": {}", state.meta.program_seed);
    result += ",";
    result += std::format("\"profile\": \"{}\"", escape_json(state.meta.profile));
    result += ",";
    result += std::format("\"preset\": \"{}\"", escape_json(state.meta.preset));
    result += ",";
    result += std::format("\"arch\": \"{}\"", escape_json(state.meta.arch));
    result += ",";
    result += std::format("\"hir_scalar_only\": {}", state.hir_scalar_only ? "true" : "false");
    result += ",";
    result += std::format("\"scalar_status\": \"{}\"", escape_json(state.scalar_status));
    result += ",";
    result += std::format("\"vector_status\": \"{}\"", escape_json(state.vector_status));
    result += ",";
    result += std::format("\"errors\": {}", error_object_json(state));
    result += ",";
    result += std::format("\"error_metadata\": {}", error_metadata_json(state));
    result += "}";
    return result;
}

static CompileState compile_item(std::string_view input, const ParseResult &config) {
    CompileState state;
    state.meta = config.meta;
    state.serialized = std::string(input);

    MemoryArena arena{};
    Context opts{arena, "expr"};
    opts.arch = config.meta.target_arch;
    FunctionBuilder builder{opts};

    const hir::Function *fn = nullptr;
    try {
        deserialize(input, builder);
        fn = builder.build();
        state.hir_scalar_only = fn->scalar_only;
        state.comparison_unstable = has_float_to_int_cast(fn);
        state.serialized = serialize(fn);
        state.schema = schema_json(fn->args);
    } catch (const std::exception &e) {
        state.stage_error = exception_message(e, state.stage_error_info);
        return state;
    }

    if (config.preset == Preset::BuilderValidate) {
        state.scalar_status = "validated";
        state.vector_status = "not-requested";
        return state;
    }

    mir::Function *scalar_mir = nullptr;
    try {
        scalar_mir = lower_for_preset(fn, false);
        state.asmjit_scalar = emit_asmjit_code(scalar_mir);
        state.llvm_scalar = emit_llvm_code(scalar_mir);
        if (config.emit_cpp) { state.cpp_scalar = emit_cpp_code(scalar_mir); }
        if (state.asmjit_scalar.ok && state.llvm_scalar.ok) {
            state.scalar_status = "pass";
            maybe_add_code(state.codes, "asmjit_s", state.asmjit_scalar, state.comparison_unstable);
            maybe_add_asm_code(state.codes, "asmjit_s_asm", state.asmjit_scalar, state.comparison_unstable);
            maybe_add_code(state.codes, "llvm_s", state.llvm_scalar, state.comparison_unstable);
            if (config.emit_cpp) { maybe_add_code(state.codes, "cpp_s", state.cpp_scalar, state.comparison_unstable); }
        } else {
            state.scalar_status = "failed";
        }
    } catch (const std::exception &e) {
        state.stage_error = exception_message(e, state.stage_error_info);
        state.scalar_status = "failed";
        return state;
    }

    if (config.preset == Preset::ScalarDiff) {
        state.vector_status = "not-requested";
        return state;
    }

    if (state.hir_scalar_only) {
        state.vector_status = "scalar-only";
        return state;
    }

    try {
        mir::Function *vector_mir = lower_for_preset(fn, true);
        state.asmjit_vector = emit_asmjit_code(vector_mir);
        state.llvm_vector = emit_llvm_code(vector_mir);
        if (config.emit_cpp) { state.cpp_vector = emit_cpp_code(vector_mir); }
        maybe_add_code(state.codes, "asmjit", state.asmjit_vector, state.comparison_unstable);
        maybe_add_asm_code(state.codes, "asmjit_asm", state.asmjit_vector, state.comparison_unstable);
        maybe_add_code(state.codes, "llvm", state.llvm_vector, state.comparison_unstable);
        if (config.emit_cpp) { maybe_add_code(state.codes, "cpp", state.cpp_vector, state.comparison_unstable); }
        if (state.asmjit_vector.ok && state.llvm_vector.ok) {
            state.vector_status = "pass";
        } else if (state.asmjit_vector.ok || state.llvm_vector.ok) {
            state.vector_status = "partial-fail";
        } else {
            state.vector_status = "failed";
        }
    } catch (const std::exception &e) {
        state.vector_lower_error = exception_message(e, state.vector_lower_error_info);
        state.vector_status = "failed";
    }

    return state;
}

static bool compile_exit_success(const CompileState &state, Preset preset) {
    if (preset == Preset::BuilderValidate) { return state.scalar_status == "validated"; }
    if (state.scalar_status != "pass") { return false; }
    if (preset == Preset::CompileValidate) {
        return state.vector_status == "pass" || state.vector_status == "scalar-only";
    }
    if (preset == Preset::CompileObserve) { return true; }
    if (preset == Preset::VectorStrict) { return state.vector_status == "pass"; }
    return true;
}

} // namespace

int main(int argc, char **argv) {
    try {
        ParseResult config = parse_args(argc, argv);
        std::string input = read_stdin();
        if (input.empty()) { throw std::runtime_error("no serialized program provided on stdin"); }
        wait_for_debugger_if_requested(config);

        if (config.subcommand == "canonicalize") {
            MemoryArena arena{};
            Context opts{arena, "expr"};
            FunctionBuilder builder{opts};
            deserialize(input, builder);
            const hir::Function *fn = builder.build();
            std::cout << serialize(fn);
            return 0;
        }
        if (config.subcommand == "minimize") {
            std::cout << minimize_item(input, config) << "\n";
            return 0;
        }

        CompileState state = compile_item(input, config);
        std::cout << build_item_json(state) << "\n";
        return compile_exit_success(state, config.preset) ? 0 : 1;
    } catch (const std::exception &e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
