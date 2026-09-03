// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/detail/arena.h"
#include "simjit/simjit.h"

#include <cassert>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>
#include <unordered_map>
#include <vector>

#define SIMJIT_MATCH(_case) \
    case _case:             \
        if ([[maybe_unused]] auto &data = step->step_data<_case>(); true)

#define SIMJIT_MATCH2(_case1, _case2)                                                                               \
    case _case1:                                                                                                    \
    case _case2:                                                                                                    \
        if ([[maybe_unused]] auto &data = step->is(_case1) ? step->step_data<_case1>() : step->step_data<_case2>(); \
            true)

#if defined(_MSC_VER)
#define SIMJIT_ALWAYS_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define SIMJIT_ALWAYS_INLINE inline __attribute__((always_inline))
#else
#define SIMJIT_ALWAYS_INLINE inline
#endif

#if defined(__clang__)
#define SIMJIT_NO_ASAN __attribute__((no_sanitize("address")))
#elif defined(__GNUC__)
#define SIMJIT_NO_ASAN __attribute__((no_sanitize_address))
#else
#define SIMJIT_NO_ASAN
#endif

namespace simjit {

inline void vformat_to(std::string &out, const char *fmt, va_list ap) {
    size_t old_size = out.size();
    constexpr size_t initial_size = 256;

    out.resize(old_size + initial_size);

    va_list ap_copy;
    va_copy(ap_copy, ap);
    int n = std::vsnprintf(out.data() + old_size, initial_size, fmt, ap_copy);
    va_end(ap_copy);

    if (n < 0) {
        out.resize(old_size);
        out += "Formatting error";
        return;
    }

    if (size_t(n) < initial_size) {
        out.resize(old_size + size_t(n));
        return;
    }

    size_t required = size_t(n) + 1;

    out.resize(old_size + required);

    va_copy(ap_copy, ap);
    n = std::vsnprintf(out.data() + old_size, required, fmt, ap_copy);
    va_end(ap_copy);

    if (n < 0) {
        out.resize(old_size);
        out += "Formatting error";
        return;
    }

    out.resize(old_size + size_t(n));
}

inline std::string vformat(const char *fmt, va_list ap) {
    std::string result;
    vformat_to(result, fmt, ap);
    return result;
}

inline std::string format(const char *fmt, ...) __attribute__((format(printf, 1, 2)));

inline std::string format(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::string result = vformat(fmt, ap);
    va_end(ap);
    return result;
}

inline void format_to(std::string &out, const char *fmt, ...) __attribute__((format(printf, 2, 3)));

inline void format_to(std::string &out, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vformat_to(out, fmt, ap);
    va_end(ap);
}

using AccIdx = size_t;

// We use a lot of bitpacked arrays. Originally we used std::vector<bool>, but it has horrendous codegen.
// Instead we have decided to use arena-backed bitpacked arrays everywhere. They take extremely small amount of memory,
// while providing good performance out of the box.
class ArenaBitmap {
public:
    static ArenaBitmap create(MemoryArena *arena, size_t max_count, bool value = false) {
        ArenaBitmap result;
        result.max_count_ = max_count;
        size_t qword_count = (max_count + 63) / 64;
        if (value) {
            result.bits_ = arena->alloc_array<uint64_t>(qword_count, UINT64_MAX);
        } else {
            result.bits_ = arena->alloc_array<uint64_t>(qword_count);
        }
        return result;
    }

    ArenaBitmap() noexcept = default;
    ArenaBitmap(const ArenaBitmap &) noexcept = default;
    ArenaBitmap(ArenaBitmap &&) noexcept = default;
    ArenaBitmap &operator=(const ArenaBitmap &) noexcept = default;
    ArenaBitmap &operator=(ArenaBitmap &&) noexcept = default;

    size_t size() const noexcept { return max_count_; }

    bool get(size_t idx) const noexcept {
        SIMJIT_ASSERT(idx < max_count_);
        return (bits_[idx >> 6] & (1llu << (idx & 63))) != 0;
    }

    void set(size_t idx, bool value = true) noexcept {
        SIMJIT_ASSERT(idx < max_count_);
        uint64_t mask = 1llu << (idx & 63);
        if (value) {
            bits_[idx >> 6] |= mask;
        } else {
            bits_[idx >> 6] &= ~mask;
        }
    }

private:
    ArenaArray<uint64_t> bits_;
    size_t max_count_ = 0;
};

constexpr bool has_single_bit(size_t x) noexcept {
    return x && !(x & (x - 1));
}

constexpr size_t bit_width(size_t x) noexcept {
    return x == 0 ? 0 : (64 - __builtin_clzll(x));
}

constexpr size_t popcount(size_t x) noexcept {
    return __builtin_popcountll(x);
}

constexpr size_t nonzero_log2(size_t x) noexcept {
    SIMJIT_ASSERT(x != 0);
    return bit_width(x) - 1;
}

constexpr uint64_t combine_i8_to_i64(uint8_t a1, uint8_t a2, uint8_t a3, uint8_t a4, uint8_t a5, uint8_t a6, uint8_t a7,
                                     uint8_t a8) noexcept {
    return (((uint64_t)a8) << 56) | (((uint64_t)a7) << 48) | (((uint64_t)a6) << 40) | (((uint64_t)a5) << 32) |
           (((uint64_t)a4) << 24) | (((uint64_t)a3) << 16) | (((uint64_t)a2) << 8) | ((uint64_t)a1);
}

constexpr uint64_t REVERSE_BITS = combine_i8_to_i64(8, 7, 6, 5, 4, 3, 2, 1);
constexpr uint64_t REVERSE_BYTES_I16 = combine_i8_to_i64(1, 0, 3, 2, 5, 4, 7, 6);
constexpr uint64_t REVERSE_BYTES_I32 = combine_i8_to_i64(3, 2, 1, 0, 7, 6, 5, 4);
constexpr uint64_t REVERSE_BYTES_I64 = combine_i8_to_i64(7, 6, 5, 4, 3, 2, 1, 0);

static_assert(0x01'02'03'04'05'06'07'08ull == combine_i8_to_i64(8, 7, 6, 5, 4, 3, 2, 1));

} // namespace simjit
