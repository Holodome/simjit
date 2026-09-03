#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

#
# This file generates intrinsic function tables for C++ x86 backend.
# Generated files are checked out in source tree. There are intentionally no newlines.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

CPP_LICENSE_HEADER = """// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

"""

REGS = ("xmm", "ymm", "zmm")
DTYPES = ("i8", "i16", "i32", "i64", "f32", "f64")
MDTYPES = ("M2", "M4", "M8", "M16", "M32", "M64")


@dataclass(frozen=True)
class Vdtype:
    reg: str
    dtype: str

    def __post_init__(self) -> None:
        if self.reg not in REGS:
            raise ValueError(f"invalid reg {self.reg}")
        if self.dtype not in DTYPES:
            raise ValueError(f"invalid dtype {self.dtype}")


@dataclass(frozen=True)
class Intrin:
    dtype: Vdtype
    ret: object
    name: str
    args: tuple[object, ...]


@dataclass(frozen=True)
class IntrinMap:
    arity: int
    funcs: tuple[Intrin, ...]


def reg_size(reg: str) -> int:
    return {"xmm": 16, "ymm": 32, "zmm": 64}[reg]


def dtype_size(dtype: str) -> int:
    return {"i8": 1, "i16": 2, "i32": 4, "i64": 8, "f32": 4, "f64": 8}[dtype]


def vdtype_elem_count(vdtype: Vdtype) -> int:
    return reg_size(vdtype.reg) // dtype_size(vdtype.dtype)


def current_mask_type(dtype: Vdtype) -> str:
    return {2: "M2", 4: "M4", 8: "M8", 16: "M16", 32: "M32", 64: "M64"}[
        vdtype_elem_count(dtype)
    ]


def resolve_dtype(dtype: object, current_type: Vdtype) -> object:
    if dtype == "ct":
        return current_type
    if dtype == "cmt":
        return current_mask_type(current_type)
    if dtype == "cst":
        return current_type.dtype
    if dtype == "ptrct":
        return ("ptr", current_type)
    if dtype == "ptrelem":
        if current_type.reg == "zmm":
            return ("ptr", current_type.dtype)
        return ("ptr", current_type)
    if isinstance(dtype, Vdtype):
        return dtype
    if dtype in DTYPES or dtype in MDTYPES:
        return dtype
    raise ValueError(f"invalid type {dtype}")


def intrin_arity(intrin: Intrin) -> int:
    return len(intrin.args)


def resolve_intrin(
    vdtype: Vdtype, ret: object, name: str, args: tuple[object, ...]
) -> Intrin:
    return Intrin(
        vdtype,
        resolve_dtype(ret, vdtype),
        name,
        tuple(resolve_dtype(arg, vdtype) for arg in args),
    )


def make_intrin_map(funcs: list[Intrin]) -> IntrinMap:
    arities = [intrin_arity(func) for func in funcs]
    vdtypes = [func.dtype for func in funcs]
    if len(set(vdtypes)) != len(vdtypes):
        raise ValueError("some functions have same data type")
    if len(set(arities)) != 1:
        raise ValueError("functions have different arities")
    return IntrinMap(arities[0], tuple(funcs))


def reg_intrin_prefix(reg: str) -> str:
    return {"xmm": "_mm_", "ymm": "_mm256_", "zmm": "_mm512_"}[reg]


def vdtype_intrin_prefix(vec: Vdtype) -> str:
    return reg_intrin_prefix(vec.reg)


def dtype_intrin_postfix(dtype: str) -> str:
    return {
        "i8": "_epi8",
        "i16": "_epi16",
        "i32": "_epi32",
        "i64": "_epi64",
        "f32": "_ps",
        "f64": "_pd",
    }[dtype]


def dtype_intrin_u_postfix(dtype: str) -> str:
    if dtype not in ("i8", "i16", "i32", "i64"):
        raise ValueError(f"unexpected type {dtype}")
    return {
        "i8": "_epu8",
        "i16": "_epu16",
        "i32": "_epu32",
        "i64": "_epu64",
    }[dtype]


def vdtype_intrin_postfix(vec: Vdtype) -> str:
    return dtype_intrin_postfix(vec.dtype)


def vdtype_intrin_u_postfix(vec: Vdtype) -> str:
    return dtype_intrin_u_postfix(vec.dtype)


def vdtype_intrin_full_width_postfix(vec: Vdtype) -> str:
    if vec.dtype == "f32":
        return "_ps"
    if vec.dtype == "f64":
        return "_pd"
    return {"xmm": "_si128", "ymm": "_si256", "zmm": "_si512"}[vec.reg]


XMMI8 = Vdtype("xmm", "i8")
XMMI16 = Vdtype("xmm", "i16")
XMMI32 = Vdtype("xmm", "i32")
XMMI64 = Vdtype("xmm", "i64")
XMMF32 = Vdtype("xmm", "f32")
XMMF64 = Vdtype("xmm", "f64")

YMMI8 = Vdtype("ymm", "i8")
YMMI16 = Vdtype("ymm", "i16")
YMMI32 = Vdtype("ymm", "i32")
YMMI64 = Vdtype("ymm", "i64")
YMMF32 = Vdtype("ymm", "f32")
YMMF64 = Vdtype("ymm", "f64")

ZMMI8 = Vdtype("zmm", "i8")
ZMMI16 = Vdtype("zmm", "i16")
ZMMI32 = Vdtype("zmm", "i32")
ZMMI64 = Vdtype("zmm", "i64")
ZMMF32 = Vdtype("zmm", "f32")
ZMMF64 = Vdtype("zmm", "f64")


VDTYPE_NAME_MAP = {
    "XMMI8": XMMI8,
    "XMMI16": XMMI16,
    "XMMI32": XMMI32,
    "XMMI64": XMMI64,
    "XMMF32": XMMF32,
    "XMMF64": XMMF64,
    "YMMI8": YMMI8,
    "YMMI16": YMMI16,
    "YMMI32": YMMI32,
    "YMMI64": YMMI64,
    "YMMF32": YMMF32,
    "YMMF64": YMMF64,
    "ZMMI8": ZMMI8,
    "ZMMI16": ZMMI16,
    "ZMMI32": ZMMI32,
    "ZMMI64": ZMMI64,
    "ZMMF32": ZMMF32,
    "ZMMF64": ZMMF64,
}


def vdtype_symbol(vdtype: Vdtype) -> str:
    for name, value in VDTYPE_NAME_MAP.items():
        if value == vdtype:
            return name
    raise ValueError(f"unknown vdtype {vdtype}")


ALL_VDTYPES = [
    XMMI8,
    XMMI16,
    XMMI32,
    XMMI64,
    XMMF32,
    XMMF64,
    YMMI8,
    YMMI16,
    YMMI32,
    YMMI64,
    YMMF32,
    YMMF64,
    ZMMI8,
    ZMMI16,
    ZMMI32,
    ZMMI64,
    ZMMF32,
    ZMMF64,
]

INTEGER_VDTYPES = [
    XMMI8,
    XMMI16,
    XMMI32,
    XMMI64,
    YMMI8,
    YMMI16,
    YMMI32,
    YMMI64,
    ZMMI8,
    ZMMI16,
    ZMMI32,
    ZMMI64,
]

I32_I64_VDTYPES = [XMMI32, XMMI64, YMMI32, YMMI64, ZMMI32, ZMMI64]
FLOAT_VDTYPES = [XMMF32, XMMF64, YMMF32, YMMF64, ZMMF32, ZMMF64]
MASKED_BITWISE_VDTYPES = [
    vdtype for vdtype in ALL_VDTYPES if vdtype.dtype in ("i32", "i64", "f32", "f64")
]
I16_I32_I64_VDTYPES = [
    XMMI16,
    XMMI32,
    XMMI64,
    YMMI16,
    YMMI32,
    YMMI64,
    ZMMI16,
    ZMMI32,
    ZMMI64,
]
I64_VDTYPES = [XMMI64, YMMI64, ZMMI64]
I32_VDTYPES = [XMMI32, YMMI32, ZMMI32]


def make_def(
    name: str, dtype: Vdtype, infix: str, postfix: str, proto: tuple[object, ...]
) -> Intrin:
    prefix_str = vdtype_intrin_prefix(dtype)
    if postfix == "whole-width":
        postfix_str = vdtype_intrin_full_width_postfix(dtype)
    elif postfix == "dtype":
        postfix_str = vdtype_intrin_postfix(dtype)
    elif postfix == "udtype":
        postfix_str = vdtype_intrin_u_postfix(dtype)
    elif postfix == "mask":
        postfix_str = vdtype_intrin_postfix(dtype) + "_mask"
    elif postfix == "umask":
        postfix_str = vdtype_intrin_u_postfix(dtype) + "_mask"
    elif postfix == "i32":
        postfix_str = "_epi32"
    elif postfix == "u32":
        postfix_str = "_epu32"
    else:
        raise ValueError(f"invalid postfix {postfix}")
    intrin_name = prefix_str + infix + name + postfix_str
    return resolve_intrin(dtype, proto[0], intrin_name, proto[1:])


def load_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "whole-width", ("ct", "ptrelem"))


def maskz_load_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "maskz_", "dtype", ("ct", "cmt", "ptrelem"))


def mask_load_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ptrelem"))


def store_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "whole-width", ("ct", "ptrelem", "ct"))


def mask_store_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "mask_", "dtype", ("ct", "ptrelem", "cmt", "ct"))


def binary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "dtype", ("ct", "ct", "ct"))


def ubinary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "udtype", ("ct", "ct", "ct"))


def mul_def(dtype: Vdtype) -> Intrin:
    return make_def("mul", dtype, "", "i32", ("ct", "ct", "ct"))


def maskz_mul_def(dtype: Vdtype) -> Intrin:
    return make_def("mul", dtype, "maskz_", "i32", ("ct", "ct", "ct"))


def mask_mul_def(dtype: Vdtype) -> Intrin:
    return make_def("mul", dtype, "mask_", "i32", ("ct", "ct", "ct"))


def umul_def(dtype: Vdtype) -> Intrin:
    return make_def("mul", dtype, "", "u32", ("ct", "ct", "ct"))


def maskz_umul_def(dtype: Vdtype) -> Intrin:
    return make_def("mul", dtype, "maskz_", "u32", ("ct", "ct", "ct"))


def mask_umul_def(dtype: Vdtype) -> Intrin:
    return make_def("mul", dtype, "mask_", "u32", ("ct", "ct", "ct"))


def full_width_binary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "whole-width", ("ct", "ct", "ct"))


def maskz_binary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "maskz_", "dtype", ("ct", "cmt", "ct", "ct"))


def mask_binary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct", "ct"))


def maskz_ubinary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "maskz_", "udtype", ("ct", "cmt", "ct", "ct"))


def mask_ubinary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "mask_", "udtype", ("ct", "ct", "cmt", "ct", "ct"))


def cmp_mask_def(dtype: Vdtype) -> Intrin:
    return make_def("cmp", dtype, "", "mask", ("cmt", "ct", "ct", "i32"))


def mask_cmp_mask_def(dtype: Vdtype) -> Intrin:
    return make_def("cmp", dtype, "mask_", "mask", ("cmt", "cmt", "ct", "ct", "i32"))


def cmpu_mask_def(dtype: Vdtype) -> Intrin:
    return make_def("cmp", dtype, "", "umask", ("cmt", "ct", "ct", "i32"))


def mask_cmpu_mask_def(dtype: Vdtype) -> Intrin:
    return make_def("cmp", dtype, "mask_", "umask", ("cmt", "cmt", "ct", "ct", "i32"))


def maskz_mov_def(dtype: Vdtype) -> Intrin:
    return make_def("mov", dtype, "maskz_", "dtype", ("ct", "cmt", "ct"))


def mask_mov_def(dtype: Vdtype) -> Intrin:
    return make_def("mov", dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct"))


def mask_blend_def(dtype: Vdtype) -> Intrin:
    return make_def("blend", dtype, "mask_", "dtype", ("ct", "cmt", "ct", "ct"))


def unary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "dtype", ("ct", "ct"))


def maskz_unary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "maskz_", "dtype", ("ct", "cmt", "ct"))


def mask_unary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct"))


def shifti_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "dtype", ("ct", "ct", "i32"))


def maskz_shifti_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "maskz_", "dtype", ("ct", "cmt", "ct", "i32"))


def mask_shifti_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct", "i32"))


def test_def(dtype: Vdtype) -> Intrin:
    return make_def("test", dtype, "", "mask", ("cmt", "ct", "ct"))


def mask_test_def(dtype: Vdtype) -> Intrin:
    return make_def("test", dtype, "mask_", "mask", ("cmt", "cmt", "ct", "ct"))


def testn_def(dtype: Vdtype) -> Intrin:
    return make_def("testn", dtype, "", "mask", ("cmt", "ct", "ct"))


def mask_testn_def(dtype: Vdtype) -> Intrin:
    return make_def("testn", dtype, "mask_", "mask", ("cmt", "cmt", "ct", "ct"))


def ternarylogic_def(dtype: Vdtype) -> Intrin:
    return make_def("ternarylogic", dtype, "", "dtype", ("ct", "ct", "ct", "ct", "i32"))


def maskz_ternarylogic_def(dtype: Vdtype) -> Intrin:
    return make_def(
        "ternarylogic", dtype, "maskz_", "dtype", ("ct", "cmt", "ct", "ct", "ct", "i32")
    )


def mask_ternarylogic_def(dtype: Vdtype) -> Intrin:
    return make_def(
        "ternarylogic", dtype, "mask_", "dtype", ("ct", "cmt", "ct", "ct", "ct", "i32")
    )


def nullary_def(name: str, dtype: Vdtype) -> Intrin:
    return make_def(name, dtype, "", "whole-width", ("ct",))


def shuffle32_def(dtype: Vdtype) -> Intrin:
    return make_def("shuffle", dtype, "", "dtype", ("ct", "ct", "i32"))


def maskz_shuffle32_def(dtype: Vdtype) -> Intrin:
    return make_def(
        "shuffle", dtype, "maskz_", "dtype", ("ct", "ct", "cmt", "ct", "i32")
    )


def mask_shuffle32_def(dtype: Vdtype) -> Intrin:
    return make_def("shuffle", dtype, "mask_", "dtype", ("ct", "cmt", "ct", "i32"))


def mask_compress_def(dtype: Vdtype) -> Intrin:
    return make_def("compress", dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct"))


def maskz_compress_def(dtype: Vdtype) -> Intrin:
    return make_def("compress", dtype, "maskz_", "dtype", ("ct", "cmt", "ct"))


def conflict_def(dtype: Vdtype) -> Intrin:
    return make_def("conflict", dtype, "", "dtype", ("ct", "ct"))


def mask_conflict_def(dtype: Vdtype) -> Intrin:
    return make_def("conflict", dtype, "mask_", "dtype", ("ct", "cmt", "ct"))


def maskz_conflict_def(dtype: Vdtype) -> Intrin:
    return make_def("conflict", dtype, "maskz_", "dtype", ("ct", "ct", "cmt", "ct"))


def alignr_def(dtype: Vdtype) -> Intrin:
    return make_def("alignr", dtype, "", "dtype", ("ct", "ct", "ct", "i32"))


def maskz_alignr_def(dtype: Vdtype) -> Intrin:
    return make_def(
        "alignr", dtype, "maskz_", "dtype", ("ct", "cmt", "ct", "ct", "i32")
    )


def mask_alignr_def(dtype: Vdtype) -> Intrin:
    return make_def(
        "align", dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct", "ct", "i32")
    )


def funnel_left_def(dtype: Vdtype) -> Intrin:
    return make_def("shldv", dtype, "", "dtype", ("ct", "ct", "ct", "ct"))


def maskz_funnel_left_def(dtype: Vdtype) -> Intrin:
    return make_def("shldv", dtype, "maskz_", "dtype", ("ct", "cmt", "ct", "ct", "ct"))


def mask_funnel_left_def(dtype: Vdtype) -> Intrin:
    return make_def("shldv", dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct", "ct"))


def funnel_right_def(dtype: Vdtype) -> Intrin:
    return make_def("shrdv", dtype, "", "dtype", ("ct", "ct", "ct", "ct"))


def maskz_funnel_right_def(dtype: Vdtype) -> Intrin:
    return make_def("shrdv", dtype, "maskz_", "dtype", ("ct", "cmt", "ct", "ct", "ct"))


def mask_funnel_right_def(dtype: Vdtype) -> Intrin:
    return make_def("shrdv", dtype, "mask_", "dtype", ("ct", "ct", "cmt", "ct", "ct"))


def make_map(builder, dtypes):
    def inner(name: str, defs=None):
        if defs is None:
            defs = dtypes
        return make_intrin_map([builder(name, definition) for definition in defs])

    return inner


def make_nameless_map(builder, dtypes):
    def inner(defs=None):
        if defs is None:
            defs = dtypes
        return make_intrin_map([builder(definition) for definition in defs])

    return inner


FULL_WIDTH_BINARY_MAP = make_map(full_width_binary_def, ALL_VDTYPES)
BINARY_MAP = make_map(binary_def, ALL_VDTYPES)
MASKZ_BINARY_MAP = make_map(maskz_binary_def, ALL_VDTYPES)
MASK_BINARY_MAP = make_map(mask_binary_def, ALL_VDTYPES)
UBINARY_MAP = make_map(ubinary_def, INTEGER_VDTYPES)
MASKZ_UBINARY_MAP = make_map(maskz_ubinary_def, INTEGER_VDTYPES)
MASK_UBINARY_MAP = make_map(mask_ubinary_def, INTEGER_VDTYPES)
MUL_MAP = make_nameless_map(mul_def, I64_VDTYPES)
MASKZ_MUL_MAP = make_nameless_map(maskz_mul_def, I64_VDTYPES)
MASK_MUL_MAP = make_nameless_map(mask_mul_def, I64_VDTYPES)
UMUL_MAP = make_nameless_map(umul_def, I64_VDTYPES)
MASKZ_UMUL_MAP = make_nameless_map(maskz_umul_def, I64_VDTYPES)
MASK_UMUL_MAP = make_nameless_map(mask_umul_def, I64_VDTYPES)
LOAD_MAP = make_map(load_def, ALL_VDTYPES)
MASKZ_LOAD_MAP = make_map(maskz_load_def, ALL_VDTYPES)
MASK_LOAD_MAP = make_map(mask_load_def, ALL_VDTYPES)
STORE_MAP = make_map(store_def, ALL_VDTYPES)
MASK_STORE_MAP = make_map(mask_store_def, ALL_VDTYPES)
CMP_MASK_MAP = make_nameless_map(cmp_mask_def, ALL_VDTYPES)
MASK_CMP_MASK_MAP = make_nameless_map(mask_cmp_mask_def, ALL_VDTYPES)
MASKZ_MOV_MAP = make_nameless_map(maskz_mov_def, ALL_VDTYPES)
CMPU_MASK_MAP = make_nameless_map(cmpu_mask_def, INTEGER_VDTYPES)
MASK_CMPU_MASK_MAP = make_nameless_map(mask_cmpu_mask_def, INTEGER_VDTYPES)
MASK_MOV_MAP = make_nameless_map(mask_mov_def, ALL_VDTYPES)
MASK_BLEND_MAP = make_nameless_map(mask_blend_def, ALL_VDTYPES)
UNARY_MAP = make_map(unary_def, ALL_VDTYPES)
MASKZ_UNARY_MAP = make_map(maskz_unary_def, ALL_VDTYPES)
MASK_UNARY_MAP = make_map(mask_unary_def, ALL_VDTYPES)
SHIFTI_MAP = make_map(shifti_def, I32_I64_VDTYPES)
MASKZ_SHIFTI_MAP = make_map(maskz_shifti_def, I32_I64_VDTYPES)
MASK_SHIFTI_MAP = make_map(mask_shifti_def, I32_I64_VDTYPES)
SHIFTI16_MAP = make_map(shifti_def, I16_I32_I64_VDTYPES)
MASKZ_SHIFTI16_MAP = make_map(maskz_shifti_def, I16_I32_I64_VDTYPES)
MASK_SHIFTI16_MAP = make_map(mask_shifti_def, I16_I32_I64_VDTYPES)
TEST_MAP = make_nameless_map(test_def, ALL_VDTYPES)
MASK_TEST_MAP = make_nameless_map(mask_test_def, ALL_VDTYPES)
TESTN_MAP = make_nameless_map(testn_def, ALL_VDTYPES)
MASK_TESTN_MAP = make_nameless_map(mask_testn_def, ALL_VDTYPES)
TERNARYLOGIC_MAP = make_nameless_map(ternarylogic_def, I32_I64_VDTYPES)
MASKZ_TERNARYLOGIC_MAP = make_nameless_map(maskz_ternarylogic_def, I32_I64_VDTYPES)
MASK_TERNARYLOGIC_MAP = make_nameless_map(mask_ternarylogic_def, I32_I64_VDTYPES)
NULLARY_MAP = make_map(nullary_def, ALL_VDTYPES)
SHUFFLE32_MAP = make_nameless_map(shuffle32_def, I32_VDTYPES)
MASKZ_SHUFFLE32_MAP = make_nameless_map(maskz_shuffle32_def, I32_VDTYPES)
MASK_SHUFFLE32_MAP = make_nameless_map(mask_shuffle32_def, I32_VDTYPES)
MASK_COMPRESS_MAP = make_nameless_map(mask_compress_def, ALL_VDTYPES)
MASKZ_COMPRESS_MAP = make_nameless_map(maskz_compress_def, ALL_VDTYPES)
CONFLICT_MAP = make_nameless_map(conflict_def, I32_I64_VDTYPES)
MASK_CONFLICT_MAP = make_nameless_map(mask_conflict_def, I32_I64_VDTYPES)
MASKZ_CONFLICT_MAP = make_nameless_map(maskz_conflict_def, I32_I64_VDTYPES)
ALIGNR_MAP = make_nameless_map(alignr_def, I32_I64_VDTYPES)
MASKZ_ALIGNR_MAP = make_nameless_map(maskz_alignr_def, I32_I64_VDTYPES)
MASK_ALIGNR_MAP = make_nameless_map(mask_alignr_def, I32_I64_VDTYPES)
FUNNEL_LEFT_MAP = make_nameless_map(funnel_left_def, I16_I32_I64_VDTYPES)
MASKZ_FUNNEL_LEFT_MAP = make_nameless_map(maskz_funnel_left_def, I16_I32_I64_VDTYPES)
MASK_FUNNEL_LEFT_MAP = make_nameless_map(mask_funnel_left_def, I16_I32_I64_VDTYPES)
FUNNEL_RIGHT_MAP = make_nameless_map(funnel_right_def, I16_I32_I64_VDTYPES)
MASKZ_FUNNEL_RIGHT_MAP = make_nameless_map(maskz_funnel_right_def, I16_I32_I64_VDTYPES)
MASK_FUNNEL_RIGHT_MAP = make_nameless_map(mask_funnel_right_def, I16_I32_I64_VDTYPES)


ALL_MAPS = [
    ("min-map", BINARY_MAP("min")),
    ("maskz-min-map", MASKZ_BINARY_MAP("min")),
    ("mask-min-map", MASK_BINARY_MAP("min")),
    ("max-map", BINARY_MAP("max")),
    ("maskz-max-map", MASKZ_BINARY_MAP("max")),
    ("mask-max-map", MASK_BINARY_MAP("max")),
    ("umin-map", UBINARY_MAP("min")),
    ("maskz-umin-map", MASKZ_UBINARY_MAP("min")),
    ("mask-umin-map", MASK_UBINARY_MAP("min")),
    ("umax-map", UBINARY_MAP("max")),
    ("maskz-umax-map", MASKZ_UBINARY_MAP("max")),
    ("mask-umax-map", MASK_UBINARY_MAP("max")),
    ("add-map", BINARY_MAP("add")),
    ("maskz-add-map", MASKZ_BINARY_MAP("add")),
    ("mask-add-map", MASK_BINARY_MAP("add")),
    ("sub-map", BINARY_MAP("sub")),
    ("maskz-sub-map", MASKZ_BINARY_MAP("sub")),
    ("mask-sub-map", MASK_BINARY_MAP("sub")),
    ("mullo-map", BINARY_MAP("mullo", I16_I32_I64_VDTYPES)),
    ("maskz-mullo-map", MASKZ_BINARY_MAP("mullo", I16_I32_I64_VDTYPES)),
    ("mask-mullo-map", MASK_BINARY_MAP("mullo", I16_I32_I64_VDTYPES)),
    ("mul-map", MUL_MAP()),
    ("maskz-mul-map", MASKZ_MUL_MAP()),
    ("mask-mul-map", MASK_MUL_MAP()),
    ("umul-map", UMUL_MAP()),
    ("maskz-umul-map", MASKZ_UMUL_MAP()),
    ("mask-umul-map", MASK_UMUL_MAP()),
    ("and-map", FULL_WIDTH_BINARY_MAP("and")),
    ("maskz-and-map", MASKZ_BINARY_MAP("and", MASKED_BITWISE_VDTYPES)),
    ("mask-and-map", MASK_BINARY_MAP("and", MASKED_BITWISE_VDTYPES)),
    ("or-map", FULL_WIDTH_BINARY_MAP("or")),
    ("maskz-or-map", MASKZ_BINARY_MAP("or", MASKED_BITWISE_VDTYPES)),
    ("mask-or-map", MASK_BINARY_MAP("or", MASKED_BITWISE_VDTYPES)),
    ("xor-map", FULL_WIDTH_BINARY_MAP("xor")),
    ("maskz-xor-map", MASKZ_BINARY_MAP("xor", MASKED_BITWISE_VDTYPES)),
    ("mask-xor-map", MASK_BINARY_MAP("xor", MASKED_BITWISE_VDTYPES)),
    ("andnot-map", FULL_WIDTH_BINARY_MAP("andnot")),
    ("maskz-andnot-map", MASKZ_BINARY_MAP("andnot", MASKED_BITWISE_VDTYPES)),
    ("mask-andnot-map", MASK_BINARY_MAP("andnot", MASKED_BITWISE_VDTYPES)),
    ("loadu-map", LOAD_MAP("loadu")),
    ("maskz-loadu-map", MASKZ_LOAD_MAP("loadu")),
    ("mask-loadu-map", MASK_LOAD_MAP("loadu")),
    ("loada-map", LOAD_MAP("load")),
    ("maskz-loada-map", MASKZ_LOAD_MAP("load")),
    ("mask-loada-map", MASK_LOAD_MAP("load")),
    ("storeu-map", STORE_MAP("storeu")),
    ("mask-storeu-map", MASK_STORE_MAP("storeu")),
    ("storea-map", STORE_MAP("store")),
    ("cmp-mask-map", CMP_MASK_MAP()),
    ("mask-cmp-mask-map", MASK_CMP_MASK_MAP()),
    ("cmpu-mask-map", CMPU_MASK_MAP()),
    ("mask-cmpu-mask-map", MASK_CMPU_MASK_MAP()),
    ("maskz-mov-map", MASKZ_MOV_MAP()),
    ("mask-mov-map", MASK_MOV_MAP()),
    ("mask-blend-map", MASK_BLEND_MAP()),
    ("abs-map", UNARY_MAP("abs", ALL_VDTYPES)),
    ("maskz-abs-map", MASKZ_UNARY_MAP("abs")),
    ("mask-abs-map", MASK_UNARY_MAP("abs")),
    ("srai-map", SHIFTI16_MAP("srai")),
    ("maskz-srai-map", MASKZ_SHIFTI16_MAP("srai")),
    ("mask-srai-map", MASK_SHIFTI16_MAP("srai")),
    ("srli-map", SHIFTI16_MAP("srli")),
    ("maskz-srli-map", MASKZ_SHIFTI16_MAP("srli")),
    ("mask-srli-map", MASK_SHIFTI16_MAP("srli")),
    ("slli-map", SHIFTI16_MAP("slli")),
    ("maskz-slli-map", MASKZ_SHIFTI16_MAP("slli")),
    ("mask-slli-map", MASK_SHIFTI16_MAP("slli")),
    ("roli-map", SHIFTI_MAP("rol")),
    ("maskz-roli-map", MASKZ_SHIFTI_MAP("rol")),
    ("mask-roli-map", MASK_SHIFTI_MAP("rol")),
    ("rori-map", SHIFTI_MAP("ror")),
    ("maskz-rori-map", MASKZ_SHIFTI_MAP("ror")),
    ("mask-rori-map", MASK_SHIFTI_MAP("ror")),
    ("srav-map", BINARY_MAP("srav", I16_I32_I64_VDTYPES)),
    ("maskz-srav-map", MASKZ_BINARY_MAP("srav")),
    ("mask-srav-map", MASK_BINARY_MAP("srav")),
    ("srlv-map", BINARY_MAP("srlv", I16_I32_I64_VDTYPES)),
    ("maskz-srlv-map", MASKZ_BINARY_MAP("srlv")),
    ("mask-srlv-map", MASK_BINARY_MAP("srlv")),
    ("sllv-map", BINARY_MAP("sllv", I16_I32_I64_VDTYPES)),
    ("maskz-sllv-map", MASKZ_BINARY_MAP("sllv")),
    ("mask-sllv-map", MASK_BINARY_MAP("sllv")),
    ("rolv-map", BINARY_MAP("rolv", I32_I64_VDTYPES)),
    ("maskz-rolv-map", MASKZ_BINARY_MAP("rolv", I32_I64_VDTYPES)),
    ("mask-rolv-map", MASK_BINARY_MAP("rolv", I32_I64_VDTYPES)),
    ("rorv-map", BINARY_MAP("rorv", I32_I64_VDTYPES)),
    ("maskz-rorv-map", MASKZ_BINARY_MAP("rorv", I32_I64_VDTYPES)),
    ("mask-rorv-map", MASK_BINARY_MAP("rorv", I32_I64_VDTYPES)),
    ("test-map", TEST_MAP()),
    ("mask-test-map", MASK_TEST_MAP()),
    ("testn-map", TESTN_MAP()),
    ("mask-testn-map", MASK_TESTN_MAP()),
    ("ternarylogic-map", TERNARYLOGIC_MAP()),
    ("maskz-ternarylogic-map", MASKZ_TERNARYLOGIC_MAP()),
    ("mask-ternarylogic-map", MASK_TERNARYLOGIC_MAP()),
    ("popcnt-map", UNARY_MAP("popcnt", INTEGER_VDTYPES)),
    ("maskz-popcnt-map", MASKZ_UNARY_MAP("popcnt", INTEGER_VDTYPES)),
    ("mask-popcnt-map", MASK_UNARY_MAP("popcnt", INTEGER_VDTYPES)),
    ("lzcnt-map", UNARY_MAP("lzcnt", I32_I64_VDTYPES)),
    ("maskz-lzcnt-map", MASKZ_UNARY_MAP("lzcnt", I32_I64_VDTYPES)),
    ("mask-lzcnt-map", MASK_UNARY_MAP("lzcnt", I32_I64_VDTYPES)),
    ("setzero-map", NULLARY_MAP("setzero")),
    ("shuffle32-map", SHUFFLE32_MAP()),
    ("maskz-shuffle32-map", MASKZ_SHUFFLE32_MAP()),
    ("mask-shuffle32-map", MASK_SHUFFLE32_MAP()),
    ("mask-compress-map", MASK_COMPRESS_MAP()),
    ("maskz-compress-map", MASKZ_COMPRESS_MAP()),
    ("conflict-map", CONFLICT_MAP()),
    ("maskz-conflict-map", MASKZ_CONFLICT_MAP()),
    ("mask-conflict-map", MASK_CONFLICT_MAP()),
    ("alignr-map", ALIGNR_MAP()),
    ("maskz-alignr-map", MASKZ_ALIGNR_MAP()),
    ("mask-alignr-map", MASK_ALIGNR_MAP()),
    ("funnel-left-map", FUNNEL_LEFT_MAP()),
    ("maskz-funnel-left-map", MASKZ_FUNNEL_LEFT_MAP()),
    ("mask-funnel-left-map", MASK_FUNNEL_LEFT_MAP()),
    ("funnel-right-map", FUNNEL_RIGHT_MAP()),
    ("maskz-funnel-right-map", MASKZ_FUNNEL_RIGHT_MAP()),
    ("mask-funnel-right-map", MASK_FUNNEL_RIGHT_MAP()),
    ("float-mul-map", BINARY_MAP("mul", FLOAT_VDTYPES)),
    ("maskz-float-mul-map", MASKZ_BINARY_MAP("mul", FLOAT_VDTYPES)),
    ("mask-float-mul-map", MASK_BINARY_MAP("mul", FLOAT_VDTYPES)),
    ("div-map", BINARY_MAP("div", FLOAT_VDTYPES)),
    ("maskz-float-div-map", MASKZ_BINARY_MAP("div", FLOAT_VDTYPES)),
    ("mask-float-div-map", MASK_BINARY_MAP("div", FLOAT_VDTYPES)),
    ("rcp-map", UNARY_MAP("rcp14", FLOAT_VDTYPES)),
    ("maskz-rcp-map", MASKZ_UNARY_MAP("rcp14", FLOAT_VDTYPES)),
    ("mask-rcp-map", MASK_UNARY_MAP("rcp14", FLOAT_VDTYPES)),
    ("sqrt-map", UNARY_MAP("sqrt", FLOAT_VDTYPES)),
    ("maskz-sqrt-map", MASKZ_UNARY_MAP("sqrt", FLOAT_VDTYPES)),
    ("mask-sqrt-map", MASK_UNARY_MAP("sqrt", FLOAT_VDTYPES)),
    ("rsqrt-map", UNARY_MAP("rsqrt14", FLOAT_VDTYPES)),
    ("maskz-rsqrt-map", MASKZ_UNARY_MAP("rsqrt14", FLOAT_VDTYPES)),
    ("mask-rsqrt-map", MASK_UNARY_MAP("rsqrt14", FLOAT_VDTYPES)),
]


def sdtype_to_cpp(dtype: str) -> str:
    return {
        "i8": "I8",
        "i16": "I16",
        "i32": "I32",
        "i64": "I64",
        "f32": "F32",
        "f64": "F64",
    }[dtype]


def mdtype_to_cpp(dtype: str) -> str:
    return {
        "M2": "M2",
        "M4": "M4",
        "M8": "M8",
        "M16": "M16",
        "M32": "M32",
        "M64": "M64",
    }[dtype]


def dtype_to_cpp(dtype: object) -> str:
    if isinstance(dtype, Vdtype):
        return vdtype_symbol(dtype)
    if isinstance(dtype, tuple) and len(dtype) == 2 and dtype[0] == "ptr":
        return dtype_to_cpp(dtype[1])
    if dtype in DTYPES:
        return sdtype_to_cpp(dtype)
    if dtype in MDTYPES:
        return mdtype_to_cpp(dtype)
    raise ValueError(f"invalid data type {dtype}")


def intrin_to_cpp(intrin: Intrin) -> str:
    cpp_name = "I" + vdtype_symbol(intrin.dtype)
    result = f'{cpp_name}({dtype_to_cpp(intrin.ret)}, "{intrin.name}"'
    if intrin.args:
        result += "," + ",".join(dtype_to_cpp(arg) for arg in intrin.args)
    result += ")"
    return result


def print_map_to_cpp(map_def: tuple[str, IntrinMap]) -> str:
    name, intrin_map = map_def
    variable_name = name.replace("-", "_")
    return (
        f'const VecIntrinsicUnaryMap {variable_name} {{ "{variable_name}", '
        + ",".join(intrin_to_cpp(func) for func in intrin_map.funcs)
        + "};"
    )


def print_map_to_cpp_header(map_def: tuple[str, IntrinMap]) -> str:
    name, _ = map_def
    variable_name = name.replace("-", "_")
    return f"extern const VecIntrinsicUnaryMap {variable_name};"


def gen_cpp_src(dest_dir: Path) -> None:
    out_path = dest_dir / "x86_intrin.generated.cpp"
    with out_path.open("w", encoding="ascii", newline="") as out:
        out.write(CPP_LICENSE_HEADER)
        out.write("// GENERATED BY x86-intrin.py DO NOT EDIT\n")
        out.write("// clang-format off\n")
        out.write("// NOLINTBEGIN(bugprone-throwing-static-initialization)\n")
        for map_def in ALL_MAPS:
            out.write(print_map_to_cpp(map_def))
        out.write("\n// NOLINTEND(bugprone-throwing-static-initialization)\n")


def gen_cpp_header(dest_dir: Path) -> None:
    out_path = dest_dir / "x86_intrin.generated.h"
    with out_path.open("w", encoding="ascii", newline="") as out:
        out.write(CPP_LICENSE_HEADER)
        out.write("// GENERATED BY x86-intrin.py DO NOT EDIT\n")
        out.write("// clang-format off\n")
        for map_def in ALL_MAPS:
            out.write(print_map_to_cpp_header(map_def))


def main():
    parser = argparse.ArgumentParser(
        description="Generate x86_intrin.generated.{h,cpp} in the destination directory."
    )
    parser.add_argument(
        "dest_dir",
        nargs="?",
        default="src/simjit/core/cpp",
        help="Destination directory for generated files (default: src/simjit/core/cpp)",
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    gen_cpp_src(dest_dir)
    gen_cpp_header(dest_dir)


if __name__ == "__main__":
    main()
