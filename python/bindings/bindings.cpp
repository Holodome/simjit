// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "module_api.h"
#include "resolver.h"

#include <pybind11/typing.h>

namespace py = pybind11;
namespace sj = simjit;
using namespace simjit_python;

template <typename Tag> struct AnnotatedObject {
    py::object value;
};

struct PythonValueTag {
    static constexpr auto name = py::detail::const_name("builtins.object");
};
struct DslOutputsTag {
    static constexpr auto name = py::detail::const_name("typing.Sequence[tuple[str, simjit.ir.Expr]]");
};
struct InputMappingTag {
    static constexpr auto name = py::detail::const_name("typing.Mapping[str, builtins.object]");
};
struct OptionalInputMappingTag {
    static constexpr auto name = py::detail::const_name("typing.Optional[typing.Mapping[str, builtins.object]]");
};
struct IrBufferHandleTag {
    static constexpr auto name = py::detail::const_name("simjit.ir.BufferHandle");
};
struct SchemaMappingTag {
    static constexpr auto name =
        py::detail::const_name("typing.Mapping[str, simjit.ir.ScalarType | tuple[simjit.ir.ScalarType, bool]]");
};

using PythonValue = AnnotatedObject<PythonValueTag>;
using DslOutputs = AnnotatedObject<DslOutputsTag>;
using InputMapping = AnnotatedObject<InputMappingTag>;
using OptionalInputMapping = AnnotatedObject<OptionalInputMappingTag>;
using IrBufferHandle = AnnotatedObject<IrBufferHandleTag>;
using SchemaMapping = AnnotatedObject<SchemaMappingTag>;

namespace pybind11::detail {

template <typename Tag> struct type_caster<AnnotatedObject<Tag>> {
    PYBIND11_TYPE_CASTER(AnnotatedObject<Tag>, Tag::name);

    bool load(handle src, bool) {
        value.value = reinterpret_borrow<object>(src);
        return true;
    }
    static handle cast(const AnnotatedObject<Tag> &src, return_value_policy, handle) {
        return reinterpret_borrow<object>(src.value).release();
    }
};

} // namespace pybind11::detail

using ResultDict = py::typing::Dict<py::str, PythonValue>;
using BufferDict = py::typing::Dict<py::str, IrBufferHandle>;
using ObjectTuple = py::typing::Tuple<PythonValue, py::ellipsis>;

static ResultDict typed_dict(py::dict value) {
    return py::reinterpret_steal<ResultDict>(value.release());
}

static ObjectTuple typed_tuple(py::tuple value) {
    return py::reinterpret_steal<ObjectTuple>(value.release());
}

PYBIND11_MODULE(_simjit, m) {
    m.doc() = "simjit glue code";
    py::register_exception<SafetyCheckFailed>(m, "SafetyCheckFailed");

    py::enum_<sj::jit::CompilePolicy>(m, "CompilePolicy")
        .value("BestEffort", sj::jit::CompilePolicy::BestEffort)
        .value("Vectorized", sj::jit::CompilePolicy::Vectorized)
        .value("Scalar", sj::jit::CompilePolicy::Scalar);
    py::enum_<sj::jit::DebugStage>(m, "DebugStage")
        .value("HIR", sj::jit::DebugStage::HIR)
        .value("Vectorizer", sj::jit::DebugStage::Vectorizer)
        .value("MIR", sj::jit::DebugStage::MIR)
        .value("ASM", sj::jit::DebugStage::ASM)
        .value("MachineCode", sj::jit::DebugStage::MachineCode)
        .value("All", sj::jit::DebugStage::All);
    py::enum_<sj::CodeTransformations>(m, "CodeTransformations")
        .value("No", sj::CodeTransformations::No)
        .value("MuldqInst", sj::CodeTransformations::MuldqInst)
        .value("MulConstPeephole", sj::CodeTransformations::MulConstPeephole)
        .value("LogicalPeephole", sj::CodeTransformations::LogicalPeephole)
        .value("BetweenPeephole", sj::CodeTransformations::BetweenPeephole)
        .value("Unroll", sj::CodeTransformations::Unroll)
        .value("AccSplit", sj::CodeTransformations::AccSplit)
        .value("MaskCombine", sj::CodeTransformations::MaskCombine)
        .value("TernarylogicInst", sj::CodeTransformations::TernarylogicInst)
        .value("FmaInst", sj::CodeTransformations::FmaInst)
        .value("SmallArith", sj::CodeTransformations::SmallArith)
        .value("All", sj::CodeTransformations::All);
    py::enum_<sj::LoadStoreKind>(m, "LoadStoreKind")
        .value("Aligned", sj::LoadStoreKind::Aligned)
        .value("Unaligned", sj::LoadStoreKind::Unaligned);
    py::enum_<sj::ArithBinaryOp>(m, "ArithBinaryOp")
        .value("Add", sj::ArithBinaryOp::Add)
        .value("Sub", sj::ArithBinaryOp::Sub)
        .value("Mul", sj::ArithBinaryOp::Mul)
        .value("Mul64SE", sj::ArithBinaryOp::Mul64SE)
        .value("Mul64ZE", sj::ArithBinaryOp::Mul64ZE)
        .value("Div", sj::ArithBinaryOp::Div)
        .value("UDiv", sj::ArithBinaryOp::UDiv)
        .value("Mod", sj::ArithBinaryOp::Mod)
        .value("UMod", sj::ArithBinaryOp::UMod)
        .value("Min", sj::ArithBinaryOp::Min)
        .value("Max", sj::ArithBinaryOp::Max)
        .value("UMin", sj::ArithBinaryOp::UMin)
        .value("UMax", sj::ArithBinaryOp::UMax)
        .value("And", sj::ArithBinaryOp::And)
        .value("Or", sj::ArithBinaryOp::Or)
        .value("Xor", sj::ArithBinaryOp::Xor)
        .value("AndNot", sj::ArithBinaryOp::AndNot)
        .value("ShiftLeftLogical", sj::ArithBinaryOp::ShiftLeftLogical)
        .value("ShiftRightLogical", sj::ArithBinaryOp::ShiftRightLogical)
        .value("ShiftRightArith", sj::ArithBinaryOp::ShiftRightArith)
        .value("RotateLeft", sj::ArithBinaryOp::RotateLeft)
        .value("RotateRight", sj::ArithBinaryOp::RotateRight);
    py::enum_<sj::PredicateBinaryOp>(m, "PredicateBinaryOp")
        .value("And", sj::PredicateBinaryOp::And)
        .value("Or", sj::PredicateBinaryOp::Or)
        .value("Xor", sj::PredicateBinaryOp::Xor)
        .value("AndNot", sj::PredicateBinaryOp::AndNot)
        .value("XNor", sj::PredicateBinaryOp::XNor);
    py::enum_<sj::ArithUnaryOp>(m, "ArithUnaryOp")
        .value("Not", sj::ArithUnaryOp::Not)
        .value("Negate", sj::ArithUnaryOp::Negate)
        .value("Abs", sj::ArithUnaryOp::Abs)
        .value("Lzcnt", sj::ArithUnaryOp::Lzcnt)
        .value("Tzcnt", sj::ArithUnaryOp::Tzcnt)
        .value("Popcount", sj::ArithUnaryOp::Popcount)
        .value("RoundNearest", sj::ArithUnaryOp::RoundNearest)
        .value("RoundDown", sj::ArithUnaryOp::RoundDown)
        .value("RoundUp", sj::ArithUnaryOp::RoundUp)
        .value("RoundTruncate", sj::ArithUnaryOp::RoundTruncate)
        .value("Rcp", sj::ArithUnaryOp::Rcp)
        .value("Sqrt", sj::ArithUnaryOp::Sqrt)
        .value("Rsqrt", sj::ArithUnaryOp::Rsqrt);
    py::enum_<sj::CmpOp>(m, "CompareOp")
        .value("Less", sj::CmpOp::Less)
        .value("Greater", sj::CmpOp::Greater)
        .value("LessEqual", sj::CmpOp::LessEqual)
        .value("GreaterEqual", sj::CmpOp::GreaterEqual)
        .value("Equal", sj::CmpOp::Equal)
        .value("NotEqual", sj::CmpOp::NotEqual);
    py::enum_<DslIntCastKind>(m, "IntCastKind")
        .value("Cast", DslIntCastKind::Cast)
        .value("Signed", DslIntCastKind::Signed)
        .value("Unsigned", DslIntCastKind::Unsigned)
        .value("Trunc", DslIntCastKind::Trunc)
        .value("Sext", DslIntCastKind::Sext)
        .value("Zext", DslIntCastKind::Zext);
    py::enum_<sj::FpClass>(m, "FpClassFlags")
        .value("Infinite", sj::FpClass::FPC_INFINITE)
        .value("Nan", sj::FpClass::FPC_NAN)
        .value("Subnormal", sj::FpClass::FPC_SUBNORMAL)
        .value("Zero", sj::FpClass::FPC_ZERO)
        .def(
            "__or__", [](sj::FpClass lhs, sj::FpClass rhs) { return lhs | rhs; }, py::is_operator())
        .def("__and__", [](sj::FpClass lhs, sj::FpClass rhs) { return lhs & rhs; }, py::is_operator());
    py::enum_<DslFunctionKind>(m, "FunctionName")
        .value("Year", DslFunctionKind::Year)
        .value("Month", DslFunctionKind::Month)
        .value("Day", DslFunctionKind::Day)
        .value("Hour", DslFunctionKind::Hour)
        .value("Minute", DslFunctionKind::Minute)
        .value("Second", DslFunctionKind::Second)
        .value("DayOfWeek", DslFunctionKind::DayOfWeek)
        .value("Log2", DslFunctionKind::Log2)
        .value("Log2NoZero", DslFunctionKind::Log2NoZero)
        .value("Byteswap", DslFunctionKind::Byteswap)
        .value("BitFloor", DslFunctionKind::BitFloor)
        .value("BitCeil", DslFunctionKind::BitCeil)
        .value("Coalesce", DslFunctionKind::Coalesce)
        .value("NullIf", DslFunctionKind::NullIf)
        .value("IsNull", DslFunctionKind::IsNull)
        .value("IsNotNull", DslFunctionKind::IsNotNull);
    py::class_<sj::jit::DebugOptions>(m, "DebugOptions")
        .def(py::init<>())
        .def_readwrite("capture_on_error", &sj::jit::DebugOptions::capture_on_error)
        .def_readwrite("capture_on_success", &sj::jit::DebugOptions::capture_on_success)
        .def_readwrite("record_vectorization_fail_exception",
                       &sj::jit::DebugOptions::record_vectorization_fail_exception)
        .def_readwrite("stages", &sj::jit::DebugOptions::stages);
    py::class_<sj::jit::DebugSnapshot>(m, "DebugSnapshot")
        .def_readonly("vectorization_exception", &sj::jit::DebugSnapshot::vectorization_exception)
        .def_readonly("hir", &sj::jit::DebugSnapshot::hir)
        .def_readonly("serialized", &sj::jit::DebugSnapshot::serialized)
        .def_readonly("vectorizer", &sj::jit::DebugSnapshot::vectorizer)
        .def_readonly("mir", &sj::jit::DebugSnapshot::mir)
        .def_readonly("asm_code", &sj::jit::DebugSnapshot::asm_code)
        .def_readonly("machine_code", &sj::jit::DebugSnapshot::machine_code);
    py::class_<sj::jit::Statistics>(m, "Statistics")
        .def_readonly("function_count", &sj::jit::Statistics::function_count)
        .def_readonly("cache_hits", &sj::jit::Statistics::cache_hits)
        .def_readonly("cache_misses", &sj::jit::Statistics::cache_misses)
        .def_readonly("compilation_attempts", &sj::jit::Statistics::compilation_attempts)
        .def_readonly("compilation_successes", &sj::jit::Statistics::compilation_successes)
        .def_readonly("compilation_failures", &sj::jit::Statistics::compilation_failures)
        .def_readonly("last_compilation_arena_used_memory", &sj::jit::Statistics::last_compilation_arena_used_memory)
        .def_readonly("last_compilation_arena_reserved_memory",
                      &sj::jit::Statistics::last_compilation_arena_reserved_memory)
        .def_readonly("jit_memory_block_count", &sj::jit::Statistics::jit_memory_block_count)
        .def_readonly("jit_memory_allocation_count", &sj::jit::Statistics::jit_memory_allocation_count)
        .def_readonly("jit_used_memory", &sj::jit::Statistics::jit_used_memory)
        .def_readonly("jit_reserved_memory", &sj::jit::Statistics::jit_reserved_memory)
        .def_readonly("jit_overhead_memory", &sj::jit::Statistics::jit_overhead_memory);
    py::class_<PreparedKernel>(m, "PreparedKernel")
        .def("run", &PreparedKernel::run)
        .def_property_readonly("identifier", [](const PreparedKernel &self) { return self.identifier; });
    py::class_<PreparedProgram>(m, "PreparedProgram")
        .def(
            "run", [](PreparedProgram &self, const OptionalInputMapping &inputs) { self.run(inputs.value); },
            py::arg("inputs") = py::none())
        .def("run_fresh", [](PreparedProgram &self) { return typed_dict(self.run_fresh()); })
        .def("run_fresh_values", [](PreparedProgram &self) { return typed_tuple(self.run_fresh_values()); })
        .def("release_outputs", &PreparedProgram::release_outputs)
        .def("output_buffers", [](const PreparedProgram &self) { return typed_dict(self.output_buffers()); })
        .def("result", [](const PreparedProgram &self) { return typed_dict(self.result()); })
        .def_property_readonly("identifier", &PreparedProgram::identifier);
    py::class_<Session>(m, "Session")
        .def(py::init([m](std::string_view arch) { return Session{m, arch}; }), py::arg("arch") = "native")
        .def(
            "run_native",
            [](Session &self, const BufferDict &buffers, const DslOutputs &outputs, size_t n) {
                self.run_native(buffers, outputs.value, n);
            },
            py::arg("buffers"), py::arg("outputs"), py::arg("n"))
        .def(
            "run_program",
            [](Session &self, const DslOutputs &outputs, const InputMapping &inputs, std::string_view output) {
                return typed_dict(self.run_program(outputs.value, inputs.value, output));
            },
            py::arg("outputs"), py::arg("inputs"), py::arg("output") = "numpy")
        .def(
            "prepare_program",
            [](Session &self, const DslOutputs &outputs, const InputMapping &inputs, std::string_view output) {
                return self.prepare_program(outputs.value, inputs.value, output);
            },
            py::arg("outputs"), py::arg("inputs"), py::arg("output") = "numpy")
        .def_property(
            "policy", [](const Session &self) { return self.state->jit.policy(); },
            [](Session &self, sj::jit::CompilePolicy policy) { self.state->jit.set_policy(policy); })
        .def_property(
            "transformations", [](const Session &self) { return self.state->jit.transformations(); },
            [](Session &self, sj::CodeTransformations opts) { self.state->jit.set_transformations(opts); })
        .def_property_readonly(
            "debug_options", [](Session &self) { return &self.state->jit.debug_options(); },
            py::return_value_policy::reference_internal)
        .def_property_readonly(
            "debug_snapshot", [](const Session &self) { return &self.state->jit.debug_snapshot(); },
            py::return_value_policy::reference_internal)
        .def("statistics", [](const Session &self) { return self.state->jit.statistics(); })
        .def("bug_report", [](const Session &self) { return self.state->jit.bug_report(); })
        .def("function_identifiers", [](const Session &self) { return self.state->jit.function_identifiers(); })
        .def("release", &Session::release)
        .def("clear", &Session::clear);

    m.def(
        "run_native",
        [m](const BufferDict &buffers, const DslOutputs &outputs, size_t n) {
            run_native(m, buffers, outputs.value, n);
        },
        py::arg("buffers"), py::arg("outputs"), py::arg("n"));
    m.def(
        "_infer_native_length",
        [m](const BufferDict &buffers, const DslOutputs &outputs) {
            return infer_native_length(m, buffers, outputs.value);
        },
        py::arg("buffers"), py::arg("outputs"));
    m.def(
        "run_program",
        [m](const DslOutputs &outputs, const InputMapping &inputs, std::string_view output) {
            return typed_dict(run_program(m, outputs.value, inputs.value, output));
        },
        py::arg("outputs"), py::arg("inputs"), py::arg("output") = "numpy");
    m.def(
        "inspect_program",
        [](const DslOutputs &outputs, const InputMapping &inputs, std::string_view output, std::string_view policy,
           std::string_view arch) {
            return typed_dict(inspect_program_function(outputs.value, inputs.value, output, policy, arch));
        },
        py::arg("outputs"), py::arg("inputs"), py::arg("output") = "numpy", py::arg("policy") = "best_effort",
        py::arg("arch") = "native");
    m.def(
        "inspect_schema",
        [](const DslOutputs &outputs, const SchemaMapping &schema, std::string_view output, std::string_view policy,
           std::string_view arch) {
            return typed_dict(inspect_schema_function(outputs.value, schema.value, output, policy, arch));
        },
        py::arg("outputs"), py::arg("schema"), py::arg("output") = "numpy", py::arg("policy") = "best_effort",
        py::arg("arch") = "native");
    m.def(
        "benchmark_hir_jit_compile",
        [](const DslOutputs &outputs, const InputMapping &inputs, std::string_view output, std::string_view backend,
           sj::jit::CompilePolicy policy, std::string_view llvm_opt, std::string_view arch, int warmups, int runs) {
            return typed_dict(benchmark_hir_jit_compile(outputs.value, inputs.value, output, backend, policy, llvm_opt,
                                                        arch, warmups, runs));
        },
        py::arg("outputs"), py::arg("inputs"), py::arg("output") = "numpy", py::arg("backend") = "asmjit",
        py::arg("policy") = sj::jit::CompilePolicy::BestEffort, py::arg("llvm_opt") = "O1", py::arg("arch") = "native",
        py::arg("warmups") = 3, py::arg("runs") = 30);
    m.def(
        "inspect_serialized",
        [](std::string_view serialized, std::string_view policy, std::string_view arch) {
            return typed_dict(inspect_serialized_function(serialized, policy, arch));
        },
        py::arg("serialized"), py::arg("policy"), py::arg("arch"));
}
