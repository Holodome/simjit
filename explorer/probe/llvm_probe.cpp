// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include <pybind11/pybind11.h>

#include "simjit/core/llvm/emitter.h"

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace {

static simjit::llvm_backend::LLVMOptLevel parse_opt_level(const std::string &opt) {
    if (opt == "O1" || opt == "1") { return simjit::llvm_backend::LLVMOptLevel::O1; }
    if (opt == "O3" || opt == "3") { return simjit::llvm_backend::LLVMOptLevel::O3; }
    throw std::invalid_argument("LLVM optimization level must be O1 or O3");
}

static py::dict compile_ir(const std::string &ir, const std::string &opt) {
    simjit::llvm_backend::LLVMSession session(simjit::Arch::Native, parse_opt_level(opt));

    auto start = std::chrono::steady_clock::now();
    std::atomic_signal_fence(std::memory_order_seq_cst);
    void *fn = simjit::llvm_backend::compile_ir(ir, "expr", session);
    std::atomic_signal_fence(std::memory_order_seq_cst);
    auto finish = std::chrono::steady_clock::now();
    if (fn == nullptr) { throw std::runtime_error("LLVM JIT returned a null executable function pointer"); }

    py::dict out;
    out["ok"] = true;
    out["compile_us"] = std::chrono::duration<double, std::micro>(finish - start).count();
    out["compile_boundary"] = "llvm-ir-to-executable-pointer";
    out["backend"] = "llvm";
    out["opt"] = opt;
    return out;
}

} // namespace

PYBIND11_MODULE(explorer_llvm_probe, m) {
    m.doc() = "Explorer LLVM IR-to-executable-pointer timing probe.";
    m.def("compile_ir", &compile_ir, py::arg("llvm_ir"), py::arg("opt") = "O1");
}
