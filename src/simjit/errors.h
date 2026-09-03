// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <cstdint>
#include <exception>
#include <string>
#include <utility>

#ifndef SIMJIT_ERROR_SOURCE_LOCATION
#define SIMJIT_ERROR_SOURCE_LOCATION 1
#endif

namespace simjit {

enum class ErrorModule : uint8_t {
    Generic,
    HIR,
    Vectorizer,
    MIR,
    LLVM,
    CPP,
    AsmJit,
    X86,
    A64,
    Serialization,
    JIT,
    Nullable,
};

enum class ErrorKind : uint8_t {
    InternalInvariant,
    InvalidInput,
    Unsupported,
    VectorizationFailed,
    SerializationFailure,
    JitFailure,
};

enum class ErrorSubKind : uint8_t {
    None,
    TypeError,                 // incompatible user-visible types
    LimitExceeded,             // configured resource limit exceeded
    InvalidConfiguration,      // invalid builder or context setup
    InvalidArgumentAccess,     // conflicting argument read/write usage
    MissingRequiredOutput,     // required output argument not provided
    ArgumentMismatch,          // compiled function signature mismatch
    CacheLimitExceeded,        // JIT function cache is full
    UnsupportedFeature,        // unsupported IR or API feature
    UnsupportedBackendFeature, // unsupported target backend feature
    UnsupportedHostFeature,    // unsupported host CPU or architecture
    SerializationParseError,   // serialized input could not be parsed
    SerializationFormatError,  // serialized input violates format rules
    ExternalFailure,           // failure from external library or runtime

    UnsupportedSpecialOps,                 // vectorizer target lacks required ops
    TooManyRoots,                          // vectorizer root expansion is too large
    UnresolvedCarrierDType,                // predicate carrier type is unknown
    ConflictingGraphCoefficient,           // graph coefficient assignment conflicts
    GraphCoefficientLimitExceeded,         // graph coefficient exceeds solver limit
    CoefficientRangeNeedsNormalization,    // coefficient span still too wide
    UpcastSubjectUnreachable,              // upcast rewrite target not reachable
    UpcastDidNotConverge,                  // upcast normalization exceeded limit
    ComponentRangeSubjectUnreachable,      // range rewrite target not reachable
    ComponentRangeDidNotConverge,          // component range rewrite exceeded limit
    MaskCombineTooWide,                    // mask combine exceeds max mask width
    DowncastCombineItemWidthMismatch,      // integer downcast combine width mismatch
    FloatDowncastCombineItemWidthMismatch, // float downcast combine width mismatch
    SyntheticIntCastItemWidthMismatch,     // synthetic int cast width mismatch
    UpcastHalfItemWidthMismatch,           // upcast half width mismatch
    RootWidthsNotPowerOfTwo,               // root widths cannot form binary unroll
    RootWidthsNotDivisible,                // root widths cannot evenly unroll
    RootWidthsMismatch,                    // vector roots have incompatible widths
    MaskDTypeTooWide,                      // mask dtype exceeds target range
    WidthMismatch,                         // vector item width mismatch
};

struct ErrorInfo {
    ErrorModule module = ErrorModule::Generic;
    ErrorKind kind = ErrorKind::InternalInvariant;
    ErrorSubKind subkind = ErrorSubKind::None;
    std::string message{};
#if SIMJIT_ERROR_SOURCE_LOCATION
    const char *file = "";
    int line = 0;
#endif

    std::string verbose() const;
};

class SimjitException : public std::exception {
public:
    SimjitException() = delete;
    explicit SimjitException(ErrorInfo info) noexcept : info_(std::move(info)) {}

    constexpr const ErrorInfo &info() const noexcept { return info_; }
    constexpr ErrorModule module() const noexcept { return info_.module; }
    constexpr ErrorKind kind() const noexcept { return info_.kind; }
    constexpr ErrorSubKind subkind() const noexcept { return info_.subkind; }
    constexpr ErrorSubKind vectorization_failure() const noexcept { return info_.subkind; }
    std::string verbose() const noexcept { return info_.verbose(); }

    const char *what() const noexcept override { return info_.message.c_str(); }

private:
    ErrorInfo info_{};
};

#if SIMJIT_ERROR_SOURCE_LOCATION
#define simjit_exception(_module, _kind, _subkind, ...) \
    throw ::simjit::SimjitException(                    \
        ::simjit::ErrorInfo{_module, _kind, _subkind, ::simjit::format(__VA_ARGS__), __FILE__, __LINE__})
#else

#define simjit_exception(_module, _kind, _subkind, ...) \
    throw ::simjit::SimjitException(::simjit::ErrorInfo{_module, _kind, _subkind, simjit::format(__VA_ARGS__)})
#endif

} // namespace simjit
