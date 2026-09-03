# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

"""
simjit glue code
"""
from __future__ import annotations
import builtins
import simjit.ir
import typing
__all__: list[str] = ['ArithBinaryOp', 'ArithUnaryOp', 'CodeTransformations', 'CompareOp', 'CompilePolicy', 'DebugOptions', 'DebugSnapshot', 'DebugStage', 'FpClassFlags', 'FunctionName', 'IntCastKind', 'LoadStoreKind', 'PredicateBinaryOp', 'PreparedKernel', 'PreparedProgram', 'SafetyCheckFailed', 'Session', 'Statistics', 'benchmark_hir_jit_compile', 'inspect_program', 'inspect_schema', 'inspect_serialized', 'run_native', 'run_program']
class ArithBinaryOp:
    """
    Members:
    
      Add
    
      Sub
    
      Mul
    
      Mul64SE
    
      Mul64ZE
    
      Div
    
      UDiv
    
      Mod
    
      UMod
    
      Min
    
      Max
    
      UMin
    
      UMax
    
      And
    
      Or
    
      Xor
    
      AndNot
    
      ShiftLeftLogical
    
      ShiftRightLogical
    
      ShiftRightArith
    
      RotateLeft
    
      RotateRight
    """
    Add: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Add: 0>
    And: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.And: 13>
    AndNot: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.AndNot: 16>
    Div: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Div: 5>
    Max: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Max: 10>
    Min: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Min: 9>
    Mod: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Mod: 7>
    Mul: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Mul: 2>
    Mul64SE: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Mul64SE: 3>
    Mul64ZE: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Mul64ZE: 4>
    Or: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Or: 14>
    RotateLeft: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.RotateLeft: 20>
    RotateRight: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.RotateRight: 21>
    ShiftLeftLogical: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.ShiftLeftLogical: 19>
    ShiftRightArith: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.ShiftRightArith: 17>
    ShiftRightLogical: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.ShiftRightLogical: 18>
    Sub: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Sub: 1>
    UDiv: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.UDiv: 6>
    UMax: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.UMax: 12>
    UMin: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.UMin: 11>
    UMod: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.UMod: 8>
    Xor: typing.ClassVar[ArithBinaryOp]  # value = <ArithBinaryOp.Xor: 15>
    __members__: typing.ClassVar[dict[str, ArithBinaryOp]]  # value = {'Add': <ArithBinaryOp.Add: 0>, 'Sub': <ArithBinaryOp.Sub: 1>, 'Mul': <ArithBinaryOp.Mul: 2>, 'Mul64SE': <ArithBinaryOp.Mul64SE: 3>, 'Mul64ZE': <ArithBinaryOp.Mul64ZE: 4>, 'Div': <ArithBinaryOp.Div: 5>, 'UDiv': <ArithBinaryOp.UDiv: 6>, 'Mod': <ArithBinaryOp.Mod: 7>, 'UMod': <ArithBinaryOp.UMod: 8>, 'Min': <ArithBinaryOp.Min: 9>, 'Max': <ArithBinaryOp.Max: 10>, 'UMin': <ArithBinaryOp.UMin: 11>, 'UMax': <ArithBinaryOp.UMax: 12>, 'And': <ArithBinaryOp.And: 13>, 'Or': <ArithBinaryOp.Or: 14>, 'Xor': <ArithBinaryOp.Xor: 15>, 'AndNot': <ArithBinaryOp.AndNot: 16>, 'ShiftLeftLogical': <ArithBinaryOp.ShiftLeftLogical: 19>, 'ShiftRightLogical': <ArithBinaryOp.ShiftRightLogical: 18>, 'ShiftRightArith': <ArithBinaryOp.ShiftRightArith: 17>, 'RotateLeft': <ArithBinaryOp.RotateLeft: 20>, 'RotateRight': <ArithBinaryOp.RotateRight: 21>}
    @typing.overload
    def __eq__(self, other: ArithBinaryOp) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: ArithBinaryOp) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class ArithUnaryOp:
    """
    Members:
    
      Not
    
      Negate
    
      Abs
    
      Lzcnt
    
      Tzcnt

      Popcount
    
      RoundNearest
    
      RoundDown
    
      RoundUp
    
      RoundTruncate
    
      Rcp
    
      Sqrt
    
      Rsqrt
    """
    Abs: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Abs: 2>
    Lzcnt: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Lzcnt: 3>
    Negate: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Negate: 1>
    Not: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Not: 0>
    Popcount: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Popcount: 5>
    Rcp: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Rcp: 10>
    RoundDown: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.RoundDown: 7>
    RoundNearest: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.RoundNearest: 6>
    RoundTruncate: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.RoundTruncate: 9>
    RoundUp: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.RoundUp: 8>
    Rsqrt: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Rsqrt: 12>
    Sqrt: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Sqrt: 11>
    Tzcnt: typing.ClassVar[ArithUnaryOp]  # value = <ArithUnaryOp.Tzcnt: 4>
    __members__: typing.ClassVar[dict[str, ArithUnaryOp]]  # value = {'Not': <ArithUnaryOp.Not: 0>, 'Negate': <ArithUnaryOp.Negate: 1>, 'Abs': <ArithUnaryOp.Abs: 2>, 'Lzcnt': <ArithUnaryOp.Lzcnt: 3>, 'Tzcnt': <ArithUnaryOp.Tzcnt: 4>, 'Popcount': <ArithUnaryOp.Popcount: 5>, 'RoundNearest': <ArithUnaryOp.RoundNearest: 6>, 'RoundDown': <ArithUnaryOp.RoundDown: 7>, 'RoundUp': <ArithUnaryOp.RoundUp: 8>, 'RoundTruncate': <ArithUnaryOp.RoundTruncate: 9>, 'Rcp': <ArithUnaryOp.Rcp: 10>, 'Sqrt': <ArithUnaryOp.Sqrt: 11>, 'Rsqrt': <ArithUnaryOp.Rsqrt: 12>}
    @typing.overload
    def __eq__(self, other: ArithUnaryOp) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: ArithUnaryOp) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CodeTransformations:
    """
    Members:
    
      No
    
      MuldqInst
    
      MulConstPeephole
    
      LogicalPeephole
    
      BetweenPeephole
    
      Unroll
    
      AccSplit
    
      MaskCombine
    
      TernarylogicInst
    
      FmaInst
    
      SmallArith
    
      All
    """
    AccSplit: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.AccSplit: 32>
    All: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.All: 65535>
    BetweenPeephole: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.BetweenPeephole: 8>
    FmaInst: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.FmaInst: 256>
    LogicalPeephole: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.LogicalPeephole: 4>
    MaskCombine: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.MaskCombine: 64>
    MulConstPeephole: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.MulConstPeephole: 2>
    MuldqInst: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.MuldqInst: 1>
    No: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.No: 0>
    SmallArith: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.SmallArith: 512>
    TernarylogicInst: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.TernarylogicInst: 128>
    Unroll: typing.ClassVar[CodeTransformations]  # value = <CodeTransformations.Unroll: 16>
    __members__: typing.ClassVar[dict[str, CodeTransformations]]  # value = {'No': <CodeTransformations.No: 0>, 'MuldqInst': <CodeTransformations.MuldqInst: 1>, 'MulConstPeephole': <CodeTransformations.MulConstPeephole: 2>, 'LogicalPeephole': <CodeTransformations.LogicalPeephole: 4>, 'BetweenPeephole': <CodeTransformations.BetweenPeephole: 8>, 'Unroll': <CodeTransformations.Unroll: 16>, 'AccSplit': <CodeTransformations.AccSplit: 32>, 'MaskCombine': <CodeTransformations.MaskCombine: 64>, 'TernarylogicInst': <CodeTransformations.TernarylogicInst: 128>, 'FmaInst': <CodeTransformations.FmaInst: 256>, 'SmallArith': <CodeTransformations.SmallArith: 512>, 'All': <CodeTransformations.All: 65535>}
    @typing.overload
    def __eq__(self, other: CodeTransformations) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: CodeTransformations) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CompareOp:
    """
    Members:
    
      Less
    
      Greater
    
      LessEqual
    
      GreaterEqual
    
      Equal
    
      NotEqual
    """
    Equal: typing.ClassVar[CompareOp]  # value = <CompareOp.Equal: 4>
    Greater: typing.ClassVar[CompareOp]  # value = <CompareOp.Greater: 1>
    GreaterEqual: typing.ClassVar[CompareOp]  # value = <CompareOp.GreaterEqual: 3>
    Less: typing.ClassVar[CompareOp]  # value = <CompareOp.Less: 0>
    LessEqual: typing.ClassVar[CompareOp]  # value = <CompareOp.LessEqual: 2>
    NotEqual: typing.ClassVar[CompareOp]  # value = <CompareOp.NotEqual: 5>
    __members__: typing.ClassVar[dict[str, CompareOp]]  # value = {'Less': <CompareOp.Less: 0>, 'Greater': <CompareOp.Greater: 1>, 'LessEqual': <CompareOp.LessEqual: 2>, 'GreaterEqual': <CompareOp.GreaterEqual: 3>, 'Equal': <CompareOp.Equal: 4>, 'NotEqual': <CompareOp.NotEqual: 5>}
    @typing.overload
    def __eq__(self, other: CompareOp) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: CompareOp) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class CompilePolicy:
    """
    Members:
    
      BestEffort
    
      Vectorized
    
      Scalar
    """
    BestEffort: typing.ClassVar[CompilePolicy]  # value = <CompilePolicy.BestEffort: 0>
    Scalar: typing.ClassVar[CompilePolicy]  # value = <CompilePolicy.Scalar: 2>
    Vectorized: typing.ClassVar[CompilePolicy]  # value = <CompilePolicy.Vectorized: 1>
    __members__: typing.ClassVar[dict[str, CompilePolicy]]  # value = {'BestEffort': <CompilePolicy.BestEffort: 0>, 'Vectorized': <CompilePolicy.Vectorized: 1>, 'Scalar': <CompilePolicy.Scalar: 2>}
    @typing.overload
    def __eq__(self, other: CompilePolicy) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: CompilePolicy) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class DebugOptions:
    capture_on_error: bool
    capture_on_success: bool
    record_vectorization_fail_exception: bool
    stages: DebugStage
    def __init__(self) -> None:
        ...
class DebugSnapshot:
    @property
    def asm_code(self) -> str:
        ...
    @property
    def hir(self) -> str:
        ...
    @property
    def machine_code(self) -> list[int]:
        ...
    @property
    def mir(self) -> str:
        ...
    @property
    def serialized(self) -> str:
        ...
    @property
    def vectorization_exception(self) -> str:
        ...
    @property
    def vectorizer(self) -> str:
        ...
class DebugStage:
    """
    Members:
    
      HIR
    
      Vectorizer
    
      MIR
    
      ASM
    
      MachineCode
    
      All
    """
    ASM: typing.ClassVar[DebugStage]  # value = <DebugStage.ASM: 8>
    All: typing.ClassVar[DebugStage]  # value = <DebugStage.All: 31>
    HIR: typing.ClassVar[DebugStage]  # value = <DebugStage.HIR: 1>
    MIR: typing.ClassVar[DebugStage]  # value = <DebugStage.MIR: 4>
    MachineCode: typing.ClassVar[DebugStage]  # value = <DebugStage.MachineCode: 16>
    Vectorizer: typing.ClassVar[DebugStage]  # value = <DebugStage.Vectorizer: 2>
    __members__: typing.ClassVar[dict[str, DebugStage]]  # value = {'HIR': <DebugStage.HIR: 1>, 'Vectorizer': <DebugStage.Vectorizer: 2>, 'MIR': <DebugStage.MIR: 4>, 'ASM': <DebugStage.ASM: 8>, 'MachineCode': <DebugStage.MachineCode: 16>, 'All': <DebugStage.All: 31>}
    @typing.overload
    def __eq__(self, other: DebugStage) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: DebugStage) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FpClassFlags:
    """
    Members:
    
      Infinite
    
      Nan
    
      Subnormal
    
      Zero
    """
    Infinite: typing.ClassVar[FpClassFlags]  # value = <FpClassFlags.Infinite: 1>
    Nan: typing.ClassVar[FpClassFlags]  # value = <FpClassFlags.Nan: 2>
    Subnormal: typing.ClassVar[FpClassFlags]  # value = <FpClassFlags.Subnormal: 4>
    Zero: typing.ClassVar[FpClassFlags]  # value = <FpClassFlags.Zero: 8>
    __members__: typing.ClassVar[dict[str, FpClassFlags]]  # value = {'Infinite': <FpClassFlags.Infinite: 1>, 'Nan': <FpClassFlags.Nan: 2>, 'Subnormal': <FpClassFlags.Subnormal: 4>, 'Zero': <FpClassFlags.Zero: 8>}
    def __and__(self, arg0: FpClassFlags) -> FpClassFlags:
        ...
    @typing.overload
    def __eq__(self, other: FpClassFlags) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: FpClassFlags) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __or__(self, arg0: FpClassFlags) -> FpClassFlags:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class FunctionName:
    """
    Members:
    
      Year
    
      Month
    
      Day
    
      Hour
    
      Minute
    
      Second
    
      DayOfWeek
    
      Log2
    
      Log2NoZero
    
      Byteswap
    
      BitFloor
    
      BitCeil
    
      Coalesce
    
      NullIf
    
      IsNull
    
      IsNotNull
    """
    BitCeil: typing.ClassVar[FunctionName]  # value = <FunctionName.BitCeil: 12>
    BitFloor: typing.ClassVar[FunctionName]  # value = <FunctionName.BitFloor: 11>
    Byteswap: typing.ClassVar[FunctionName]  # value = <FunctionName.Byteswap: 10>
    Coalesce: typing.ClassVar[FunctionName]  # value = <FunctionName.Coalesce: 13>
    Day: typing.ClassVar[FunctionName]  # value = <FunctionName.Day: 3>
    DayOfWeek: typing.ClassVar[FunctionName]  # value = <FunctionName.DayOfWeek: 7>
    Hour: typing.ClassVar[FunctionName]  # value = <FunctionName.Hour: 4>
    IsNotNull: typing.ClassVar[FunctionName]  # value = <FunctionName.IsNotNull: 16>
    IsNull: typing.ClassVar[FunctionName]  # value = <FunctionName.IsNull: 15>
    Log2: typing.ClassVar[FunctionName]  # value = <FunctionName.Log2: 8>
    Log2NoZero: typing.ClassVar[FunctionName]  # value = <FunctionName.Log2NoZero: 9>
    Minute: typing.ClassVar[FunctionName]  # value = <FunctionName.Minute: 5>
    Month: typing.ClassVar[FunctionName]  # value = <FunctionName.Month: 2>
    NullIf: typing.ClassVar[FunctionName]  # value = <FunctionName.NullIf: 14>
    Second: typing.ClassVar[FunctionName]  # value = <FunctionName.Second: 6>
    Year: typing.ClassVar[FunctionName]  # value = <FunctionName.Year: 1>
    __members__: typing.ClassVar[dict[str, FunctionName]]  # value = {'Year': <FunctionName.Year: 1>, 'Month': <FunctionName.Month: 2>, 'Day': <FunctionName.Day: 3>, 'Hour': <FunctionName.Hour: 4>, 'Minute': <FunctionName.Minute: 5>, 'Second': <FunctionName.Second: 6>, 'DayOfWeek': <FunctionName.DayOfWeek: 7>, 'Log2': <FunctionName.Log2: 8>, 'Log2NoZero': <FunctionName.Log2NoZero: 9>, 'Byteswap': <FunctionName.Byteswap: 10>, 'BitFloor': <FunctionName.BitFloor: 11>, 'BitCeil': <FunctionName.BitCeil: 12>, 'Coalesce': <FunctionName.Coalesce: 13>, 'NullIf': <FunctionName.NullIf: 14>, 'IsNull': <FunctionName.IsNull: 15>, 'IsNotNull': <FunctionName.IsNotNull: 16>}
    @typing.overload
    def __eq__(self, other: FunctionName) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: FunctionName) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class IntCastKind:
    """
    Members:
    
      Cast
    
      Signed
    
      Unsigned
    
      Trunc
    
      Sext
    
      Zext
    """
    Cast: typing.ClassVar[IntCastKind]  # value = <IntCastKind.Cast: 0>
    Sext: typing.ClassVar[IntCastKind]  # value = <IntCastKind.Sext: 4>
    Signed: typing.ClassVar[IntCastKind]  # value = <IntCastKind.Signed: 1>
    Trunc: typing.ClassVar[IntCastKind]  # value = <IntCastKind.Trunc: 3>
    Unsigned: typing.ClassVar[IntCastKind]  # value = <IntCastKind.Unsigned: 2>
    Zext: typing.ClassVar[IntCastKind]  # value = <IntCastKind.Zext: 5>
    __members__: typing.ClassVar[dict[str, IntCastKind]]  # value = {'Cast': <IntCastKind.Cast: 0>, 'Signed': <IntCastKind.Signed: 1>, 'Unsigned': <IntCastKind.Unsigned: 2>, 'Trunc': <IntCastKind.Trunc: 3>, 'Sext': <IntCastKind.Sext: 4>, 'Zext': <IntCastKind.Zext: 5>}
    @typing.overload
    def __eq__(self, other: IntCastKind) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: IntCastKind) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class LoadStoreKind:
    """
    Members:
    
      Aligned
    
      Unaligned
    """
    Aligned: typing.ClassVar[LoadStoreKind]  # value = <LoadStoreKind.Aligned: 0>
    Unaligned: typing.ClassVar[LoadStoreKind]  # value = <LoadStoreKind.Unaligned: 1>
    __members__: typing.ClassVar[dict[str, LoadStoreKind]]  # value = {'Aligned': <LoadStoreKind.Aligned: 0>, 'Unaligned': <LoadStoreKind.Unaligned: 1>}
    @typing.overload
    def __eq__(self, other: LoadStoreKind) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: LoadStoreKind) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PredicateBinaryOp:
    """
    Members:
    
      And
    
      Or
    
      Xor
    
      AndNot
    
      XNor
    """
    And: typing.ClassVar[PredicateBinaryOp]  # value = <PredicateBinaryOp.And: 0>
    AndNot: typing.ClassVar[PredicateBinaryOp]  # value = <PredicateBinaryOp.AndNot: 3>
    Or: typing.ClassVar[PredicateBinaryOp]  # value = <PredicateBinaryOp.Or: 1>
    XNor: typing.ClassVar[PredicateBinaryOp]  # value = <PredicateBinaryOp.XNor: 4>
    Xor: typing.ClassVar[PredicateBinaryOp]  # value = <PredicateBinaryOp.Xor: 2>
    __members__: typing.ClassVar[dict[str, PredicateBinaryOp]]  # value = {'And': <PredicateBinaryOp.And: 0>, 'Or': <PredicateBinaryOp.Or: 1>, 'Xor': <PredicateBinaryOp.Xor: 2>, 'AndNot': <PredicateBinaryOp.AndNot: 3>, 'XNor': <PredicateBinaryOp.XNor: 4>}
    @typing.overload
    def __eq__(self, other: PredicateBinaryOp) -> bool:
        ...
    @typing.overload
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __int__(self) -> int:
        ...
    @typing.overload
    def __ne__(self, other: PredicateBinaryOp) -> bool:
        ...
    @typing.overload
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class PreparedKernel:
    def run(self) -> None:
        ...
    @property
    def identifier(self) -> str:
        ...
class PreparedProgram:
    def output_buffers(self) -> dict[str, builtins.object]:
        ...
    def release_outputs(self) -> None:
        ...
    def result(self) -> dict[str, builtins.object]:
        ...
    def run(self, inputs: typing.Mapping[str, builtins.object] | None = None) -> None:
        ...
    def run_fresh(self) -> dict[str, builtins.object]:
        ...
    def run_fresh_values(self) -> tuple[builtins.object, ...]:
        ...
    @property
    def identifier(self) -> str:
        ...
class SafetyCheckFailed(Exception):
    pass
class Session:
    policy: CompilePolicy
    transformations: CodeTransformations
    def __init__(self, arch: str = 'native') -> None:
        ...
    def bug_report(self) -> str:
        ...
    def clear(self) -> None:
        ...
    def function_identifiers(self) -> list[str]:
        ...
    def prepare_program(self, outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], inputs: typing.Mapping[str, builtins.object], output: str = 'numpy') -> PreparedProgram:
        ...
    def release(self, arg0: str) -> bool:
        ...
    def run_native(self, buffers: dict[str, simjit.ir.BufferHandle], outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], n: typing.SupportsInt | typing.SupportsIndex) -> None:
        ...
    def run_program(self, outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], inputs: typing.Mapping[str, builtins.object], output: str = 'numpy') -> dict[str, builtins.object]:
        ...
    def statistics(self) -> Statistics:
        ...
    @property
    def debug_options(self) -> DebugOptions:
        ...
    @property
    def debug_snapshot(self) -> DebugSnapshot:
        ...
class Statistics:
    @property
    def cache_hits(self) -> int:
        ...
    @property
    def cache_misses(self) -> int:
        ...
    @property
    def compilation_attempts(self) -> int:
        ...
    @property
    def compilation_failures(self) -> int:
        ...
    @property
    def compilation_successes(self) -> int:
        ...
    @property
    def function_count(self) -> int:
        ...
    @property
    def jit_memory_allocation_count(self) -> int:
        ...
    @property
    def jit_memory_block_count(self) -> int:
        ...
    @property
    def jit_overhead_memory(self) -> int:
        ...
    @property
    def jit_reserved_memory(self) -> int:
        ...
    @property
    def jit_used_memory(self) -> int:
        ...
    @property
    def last_compilation_arena_reserved_memory(self) -> int:
        ...
    @property
    def last_compilation_arena_used_memory(self) -> int:
        ...
def _infer_native_length(buffers: dict[str, simjit.ir.BufferHandle], outputs: typing.Sequence[tuple[str, simjit.ir.Expr]]) -> int:
    ...
def benchmark_hir_jit_compile(outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], inputs: typing.Mapping[str, builtins.object], output: str = 'numpy', backend: str = 'asmjit', policy: CompilePolicy = CompilePolicy.BestEffort, llvm_opt: str = 'O1', arch: str = 'native', warmups: typing.SupportsInt | typing.SupportsIndex = 3, runs: typing.SupportsInt | typing.SupportsIndex = 30) -> dict[str, builtins.object]:
    ...
def inspect_program(outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], inputs: typing.Mapping[str, builtins.object], output: str = 'numpy', policy: str = 'best_effort', arch: str = 'native') -> dict[str, builtins.object]:
    ...
def inspect_schema(outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], schema: typing.Mapping[str, simjit.ir.ScalarType | tuple[simjit.ir.ScalarType, bool]], output: str = 'numpy', policy: str = 'best_effort', arch: str = 'native') -> dict[str, builtins.object]:
    ...
def inspect_serialized(serialized: str, policy: str, arch: str) -> dict[str, builtins.object]:
    ...
def run_native(buffers: dict[str, simjit.ir.BufferHandle], outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], n: typing.SupportsInt | typing.SupportsIndex) -> None:
    ...
def run_program(outputs: typing.Sequence[tuple[str, simjit.ir.Expr]], inputs: typing.Mapping[str, builtins.object], output: str = 'numpy') -> dict[str, builtins.object]:
    ...
