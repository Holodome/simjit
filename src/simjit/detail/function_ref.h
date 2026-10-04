// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include <type_traits>
#include <utility>

namespace simjit {

template <typename> class function_ref;

// function_ref may only be invoked while the referenced callable is still alive, and must not escape the call.
template <typename R, typename... Args> class function_ref<R(Args...)> {
public:
    function_ref() = delete;

    function_ref(R (*function)(Args...)) noexcept : function_(function) {}

    template <typename F, typename T = std::remove_reference_t<F>,
              std::enable_if_t<!std::is_same_v<std::remove_cv_t<T>, function_ref> && !std::is_function_v<T> &&
                                   std::is_invocable_r_v<R, T &, Args...>,
                               int> = 0>
    function_ref(F &&function) noexcept
        : object_((void *)(&function)), callback_([](void *object, Args... args) -> R {
              return (*static_cast<T *>(object))(std::forward<Args>(args)...);
          }) {}

    R operator()(Args... args) const {
        if (function_ != nullptr) { return function_(std::forward<Args>(args)...); }
        return callback_(object_, std::forward<Args>(args)...);
    }

private:
    void *object_ = nullptr;
    R (*callback_)(void *, Args...) = nullptr;
    R (*function_)(Args...) = nullptr;
};

} // namespace simjit
