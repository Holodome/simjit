// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit.h"

#include <optional>

namespace simjit {

enum class DynamicValueKind : uint8_t {
    Dummy,
    Val,
    Pred,
};

class DynamicValue {
public:
    DynamicValue() noexcept = default;
    DynamicValue(const DynamicValue &) noexcept = default;
    DynamicValue(DynamicValue &&) noexcept = default;
    DynamicValue &operator=(const DynamicValue &) noexcept = default;
    DynamicValue &operator=(DynamicValue &&) noexcept = default;

    constexpr DynamicValue(Value x) noexcept : kind_(DynamicValueKind::Val), inner_(x.step_) {}
    constexpr DynamicValue(Predicate x) noexcept : kind_(DynamicValueKind::Pred), inner_(x.step_) {}

    constexpr std::optional<Value> as_value() const noexcept {
        if (!is_value()) return {};
        return Value{inner_};
    }

    constexpr std::optional<Predicate> as_predicate() const noexcept {
        if (!is_predicate()) return {};
        return Predicate{inner_};
    }

    constexpr std::optional<ScalarDataType> dtype() const noexcept {
        if (!is_valid()) return {};
        if (is_predicate()) return ScalarDataType::I1;
        return Value{inner_}.dtype();
    }

    constexpr bool is_predicate() const noexcept { return kind_ == DynamicValueKind::Pred; }
    constexpr bool is_value() const noexcept { return kind_ == DynamicValueKind::Val; }

    constexpr bool is_valid() const noexcept { return kind_ != DynamicValueKind::Dummy; }

private:
    DynamicValueKind kind_ = DynamicValueKind::Dummy;
    hir::Step *inner_ = nullptr;
};

} // namespace simjit
