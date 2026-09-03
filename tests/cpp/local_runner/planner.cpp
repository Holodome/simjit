// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "local_runner.h"

#include <algorithm>

namespace simjit::local_runner {

std::vector<std::vector<CompileRef>> balance_compile_groups(std::vector<CompileRef> refs, size_t target_bytes,
                                                            size_t minimum_groups) {
    if (refs.empty()) return {};
    auto size_of = [](const CompileRef &ref) {
        return ref.weight_bytes != 0 ? ref.weight_bytes
                                     : ref.item->implementations[ref.implementation_index].code.size();
    };
    size_t total = 0;
    for (const auto &ref : refs)
        total += size_of(ref);
    size_t group_count = std::max(minimum_groups, (total + target_bytes - 1) / target_bytes);
    group_count = std::min(group_count, refs.size());
    std::sort(refs.begin(), refs.end(),
              [&](const CompileRef &a, const CompileRef &b) { return size_of(a) > size_of(b); });
    std::vector<std::vector<CompileRef>> groups(group_count);
    std::vector<size_t> sizes(group_count, 0);
    for (const auto &ref : refs) {
        size_t group = static_cast<size_t>(std::min_element(sizes.begin(), sizes.end()) - sizes.begin());
        groups[group].push_back(ref);
        sizes[group] += size_of(ref);
    }
    return groups;
}

} // namespace simjit::local_runner
