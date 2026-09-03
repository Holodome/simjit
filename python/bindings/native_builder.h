// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "resolver.h"

namespace simjit_python {

enum class NativeBuilderErrorKind : uint8_t {
    Type,
    Value,
    Index,
};

class NativeBuilderError : public std::exception {
public:
    NativeBuilderError(NativeBuilderErrorKind kind, std::string message) : message_(std::move(message)), kind_(kind) {}

    NativeBuilderErrorKind kind() const noexcept { return kind_; }

    const char *what() const noexcept override { return message_.c_str(); }

private:
    std::string message_{};
    NativeBuilderErrorKind kind_;
};

struct NativePointerBinding {
    size_t slot = 0;
    std::string_view name{};
    bool null_buffer = false;
    bool writable = false;
};

constexpr std::string_view NATIVE_SAFETY_CHECK_BUFFER = "__simjit_internal_safety_check";

bool native_program_requires_safety_check(const DslProgram &program);

std::vector<NativePointerBinding> build_native_pointer_plan(sj::FunctionBuilder &builder, NameMap<BufferDesc> buffers,
                                                            const DslProgram &program, size_t n);

} // namespace simjit_python
