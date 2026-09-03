// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include "simjit/detail/span.h"

namespace simjit {

template <typename T> using ArenaArray = nonstd::span<T>;

// This arena does not track destructors and simply frees whole blocks in clear()/destructor.
// It should only be used for types with trivial destructors.
class MemoryArena {
    struct MemoryArenaBlock {
        size_t size;
        size_t used;
        char *base;
        MemoryArenaBlock *next;
    };

public:
    constexpr static size_t alignment = 16;
    static_assert(alignment != 0 && ((alignment & (alignment - 1)) == 0));
    static_assert(alignment >= alignof(MemoryArenaBlock));
    static_assert(sizeof(MemoryArenaBlock) % alignment == 0);

    MemoryArena() noexcept = default;
    MemoryArena(const MemoryArena &) = delete;
    MemoryArena(MemoryArena &&other) noexcept : current_(other.current_), minimum_block_size(other.minimum_block_size) {
        other.current_ = nullptr;
    }
    MemoryArena &operator=(const MemoryArena &) = delete;
    MemoryArena &operator=(MemoryArena &&other) noexcept {
        if (this != &other) {
            clear();
            current_ = other.current_;
            minimum_block_size = other.minimum_block_size;
            other.current_ = nullptr;
        }
        return *this;
    }
    ~MemoryArena() noexcept { clear(); }

    size_t total_bytes_allocated() const noexcept {
        size_t result = 0;
        for (const MemoryArenaBlock *block = current_; block; block = block->next) {
            result += block->size + sizeof(MemoryArenaBlock);
        }
        return result;
    }

    size_t total_bytes_used() const noexcept {
        size_t result = 0;
        for (const MemoryArenaBlock *block = current_; block; block = block->next) {
            result += block->used;
        }
        return result;
    }

    void clear() noexcept {
        while (current_) {
            free_last_block();
        }
    }

    template <typename T> ArenaArray<T> alloc_array(size_t count) {
        static_assert(alignof(T) <= alignment);
        static_assert(std::is_trivially_destructible_v<T>);
        if (count == 0) { return ArenaArray<T>{}; }
        T *start = (T *)alloc(sizeof(T) * count);
        T *end = start + count;
        std::uninitialized_value_construct(start, end);
        return ArenaArray<T>{start, end};
    }

    template <typename T> ArenaArray<T> alloc_array(size_t count, const T &value) {
        static_assert(alignof(T) <= alignment);
        static_assert(std::is_trivially_destructible_v<T>);
        if (count == 0) { return ArenaArray<T>{}; }
        T *start = (T *)alloc(sizeof(T) * count);
        T *end = start + count;
        std::uninitialized_fill(start, end, value);
        return ArenaArray<T>{start, end};
    }

    template <typename T> ArenaArray<T> copy_array(nonstd::span<const T> other) {
        static_assert(alignof(T) <= alignment);
        static_assert(std::is_trivially_destructible_v<T>);
        size_t count = other.size();
        if (count == 0) { return ArenaArray<T>{}; }
        T *start = (T *)alloc(sizeof(T) * count);
        T *end = start + count;
        std::uninitialized_copy(other.begin(), other.end(), start);
        return ArenaArray<T>{start, end};
    }

    template <typename T, typename... Args> T *create(Args &&...args) {
        static_assert(alignof(T) <= alignment);
        static_assert(std::is_trivially_destructible_v<T>);
        void *mem = alloc(sizeof(T));
        return new (mem) T(std::forward<Args>(args)...);
    }

    void *alloc(size_t size) {
        void *mem = alloc_nothrow(size);
        if (mem == nullptr) throw std::bad_alloc{};
        return mem;
    }

private:
    void *alloc_nothrow(size_t size_init) noexcept {
        if (!size_init) return nullptr;

        const size_t aligned_size = (size_init + alignment - 1) & ~(alignment - 1);
        if (!current_ && !create_block(aligned_size)) return nullptr;

        uintptr_t current_ptr = reinterpret_cast<uintptr_t>(current_->base) + current_->used;

        // Check if allocation fits
        if (current_->used + aligned_size > current_->size) {
            if (!create_block(aligned_size)) return nullptr;
            current_ptr = reinterpret_cast<uintptr_t>(current_->base);
        }

        void *result = reinterpret_cast<void *>(current_ptr);
        current_->used += aligned_size;
        return result;
    }

    MemoryArenaBlock *create_block(size_t data_size) noexcept {
        data_size = std::max(data_size, minimum_block_size);
        data_size = (data_size + alignment - 1) & ~(alignment - 1);

        auto *block = (MemoryArenaBlock *)std::aligned_alloc(alignment, sizeof(MemoryArenaBlock) + data_size);
        if (!block) return nullptr;
        block->size = data_size;
        block->used = 0;
        block->base = reinterpret_cast<char *>(block + 1);
        block->next = current_;
        current_ = block;
        return block;
    }

    void free_last_block() noexcept {
        MemoryArenaBlock *block = current_;
        current_ = block->next;
        std::free(block);
    }

    MemoryArenaBlock *current_ = nullptr;
    size_t minimum_block_size = 1 << 14; // 16 KB
};

} // namespace simjit
