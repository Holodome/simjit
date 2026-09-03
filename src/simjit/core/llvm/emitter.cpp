// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "simjit/core/llvm/emitter.h"
#include "simjit/core/expr.h"
#include "simjit/core/mir.h"
#include "simjit/core/x86.h"
#include "simjit/detail/base.h"
#include "simjit/jit.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/ExecutionEngine/Orc/ThreadSafeModule.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/IntrinsicsX86.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/Utils/Mem2Reg.h"

#include <stdexcept>
#include <string_view>

#define messed_up(...) simjit_exception(ErrorModule::LLVM, {}, {}, __VA_ARGS__)
#define unsupported(...) \
    simjit_exception(ErrorModule::LLVM, ErrorKind::Unsupported, ErrorSubKind::UnsupportedBackendFeature, __VA_ARGS__)

namespace simjit {
namespace llvm_backend {

using namespace ::simjit::mir;

using V = llvm::Value;
using Ty = llvm::Type;

namespace {
struct ArgInfo {
    V *data_ptr = nullptr;
    Ty *element_type = nullptr;
};

struct AccInfo {
    V *val_ptr = nullptr;
    Ty *type = nullptr;
};
} // namespace

static llvm::CmpInst::Predicate compare_to_llvm_int(CmpOp op, bool is_unsigned) {
    switch (op) {
    case CmpOp::Less: return is_unsigned ? llvm::CmpInst::Predicate::ICMP_ULT : llvm::CmpInst::Predicate::ICMP_SLT;
    case CmpOp::Greater: return is_unsigned ? llvm::CmpInst::Predicate::ICMP_UGT : llvm::CmpInst::Predicate::ICMP_SGT;
    case CmpOp::LessEqual: return is_unsigned ? llvm::CmpInst::Predicate::ICMP_ULE : llvm::CmpInst::Predicate::ICMP_SLE;
    case CmpOp::GreaterEqual:
        return is_unsigned ? llvm::CmpInst::Predicate::ICMP_UGE : llvm::CmpInst::Predicate::ICMP_SGE;
    case CmpOp::Equal: return llvm::CmpInst::Predicate::ICMP_EQ;
    case CmpOp::NotEqual: return llvm::CmpInst::Predicate::ICMP_NE;
    }
    SIMJIT_UNREACHABLE();
}

static llvm::CmpInst::Predicate compare_to_llvm_float(CmpOp op) {
    switch (op) {
    case CmpOp::Less: return llvm::CmpInst::Predicate::FCMP_OLT;
    case CmpOp::Greater: return llvm::CmpInst::Predicate::FCMP_OGT;
    case CmpOp::LessEqual: return llvm::CmpInst::Predicate::FCMP_OLE;
    case CmpOp::GreaterEqual: return llvm::CmpInst::Predicate::FCMP_OGE;
    case CmpOp::Equal: return llvm::CmpInst::Predicate::FCMP_OEQ;
    case CmpOp::NotEqual: return llvm::CmpInst::Predicate::FCMP_UNE;
    }
    SIMJIT_UNREACHABLE();
}

namespace {
struct LLVMBuilder {
    Context *ctx;
    const mir::Function *func = nullptr;
    std::unique_ptr<llvm::LLVMContext> context;
    std::unique_ptr<llvm::IRBuilder<>> b;
    std::unique_ptr<llvm::Module> module;
    ArenaArray<ArgInfo> func_args;
    ArenaArray<AccInfo> accs;

    llvm::Function *function;

    llvm::Value *counter_ptr;
    llvm::BasicBlock *function_block;

    ArenaArray<V *> value_map{};

    llvm::Function *ternarylogic_intrin(VecDataType vdtype) {
        x86::Vector vec = x86::vec_to_x86(vdtype);
        // only int32 because we don't use masking with this instruction
        static const unsigned map[3] = {
            llvm::Intrinsic::x86_avx512_pternlog_d_128,
            llvm::Intrinsic::x86_avx512_pternlog_d_256,
            llvm::Intrinsic::x86_avx512_pternlog_d_512,
        };
        return llvm::Intrinsic::getOrInsertDeclaration(module.get(), map[(int)vec.reg]);
    }

    V *call_gather_intrin(VecDataType vdtype, V *idxs, V *data_array) {
        llvm::VectorType *llvm_vec = vec_type_to_llvm(vdtype);
        V *ptr_vec = b->CreateGEP(scalar_type_to_llvm(vdtype.to_scalar()), data_array, idxs);

        llvm::IntegerType *intTy = b->getIntNTy(vdtype.nelems());
        V *all_ones_mask = llvm::ConstantVector::getAllOnesValue(intTy);
        all_ones_mask = b->CreateBitCast(
            all_ones_mask, llvm::VectorType::get(b->getInt1Ty(), llvm::ElementCount::getFixed(vdtype.nelems())));
        return b->CreateMaskedGather(llvm_vec, ptr_vec, llvm::Align(vdtype.element_size_bytes()), all_ones_mask,
                                     llvm::UndefValue::get(llvm_vec));
    }

    V *call_cond_scatter_intrin(VecDataType vdtype, V *arg, V *idxs, V *data_array, V *cond) {
        V *ptr_vec = b->CreateGEP(scalar_type_to_llvm(vdtype.to_scalar()), data_array, idxs);
        return b->CreateMaskedScatter(arg, ptr_vec, llvm::Align(vdtype.element_size_bytes()), cond);
    }

    V *call_scatter_intrin(VecDataType vdtype, V *arg, V *idxs, V *data_array) {
        V *ptr_vec = b->CreateGEP(scalar_type_to_llvm(vdtype.to_scalar()), data_array, idxs);

        llvm::IntegerType *intTy = b->getIntNTy(vdtype.nelems());
        V *all_ones_mask = llvm::ConstantVector::getAllOnesValue(intTy);
        all_ones_mask = b->CreateBitCast(
            all_ones_mask, llvm::VectorType::get(b->getInt1Ty(), llvm::ElementCount::getFixed(vdtype.nelems())));

        return b->CreateMaskedScatter(arg, ptr_vec, llvm::Align(vdtype.element_size_bytes()), all_ones_mask);
    }

    V *call_maskz_mov(V *mask, V *one, MaskDataType mdtype) {
        size_t elem_count = mask_dtype_bits(mdtype);
        llvm::VectorType *vec_ty = llvm::VectorType::get(b->getInt8Ty(), llvm::ElementCount::getFixed(elem_count));
        V *zero = llvm::ConstantAggregateZero::get(vec_ty);
        return b->CreateSelect(mask, one, zero);
    }

    V *call_kunpack(MaskDataType mdtype, V *left, V *right) {
        size_t total_size = mask_dtype_bits(mdtype);
        llvm::SmallVector<int, 64> shuf;
        for (int i = 0; i < int(total_size); ++i) {
            shuf.push_back(i);
        }
        return b->CreateShuffleVector(left, right, shuf);
    }

    Ty *scalar_type_to_llvm(ScalarDataType dtype) {
        switch (dtype) {
        // Keep under error until we use this path
        case ScalarDataType::I1:
        case ScalarDataType::I128: messed_up("Unexpected type %s in this context", show_scalar_dtype(dtype));
        case ScalarDataType::I8: return b->getInt8Ty();
        case ScalarDataType::I16: return b->getInt16Ty();
        case ScalarDataType::I32: return b->getInt32Ty();
        case ScalarDataType::I64: return b->getInt64Ty();
        case ScalarDataType::F32: return b->getFloatTy();
        case ScalarDataType::F64: return b->getDoubleTy();
        }
        SIMJIT_UNREACHABLE();
    }

    llvm::VectorType *vec_type_to_llvm(VecDataType dtype) {
        return llvm::VectorType::get(scalar_type_to_llvm(dtype.to_scalar()),
                                     llvm::ElementCount::getFixed(dtype.nelems()));
    }

    llvm::VectorType *mask_type_to_llvm(MaskDataType dtype) {
        size_t count = mask_dtype_bits(dtype);
        return llvm::VectorType::get(b->getInt1Ty(), llvm::ElementCount::getFixed(count));
    }

    V *with_float_contract(V *value, bool approximate = false, bool reassociate = false) {
        if (auto *inst = llvm::dyn_cast<llvm::Instruction>(value)) {
            if (reassociate) { inst->setHasAllowReassoc(true); }
            inst->setHasNoSignedZeros(true);
            inst->setHasAllowContract(true);
            if (approximate) {
                inst->setHasAllowReciprocal(true);
                inst->setHasApproxFunc(true);
            }
        }
        return value;
    }

    V *float_arith_binary(ArithBinaryOp op, V *left, V *right) {
        switch (op) {
        case ArithBinaryOp::Add: return with_float_contract(b->CreateFAdd(left, right));
        case ArithBinaryOp::Sub: return with_float_contract(b->CreateFSub(left, right));
        case ArithBinaryOp::Mul: return with_float_contract(b->CreateFMul(left, right));
        case ArithBinaryOp::Div: return with_float_contract(b->CreateFDiv(left, right));
        case ArithBinaryOp::Min:
            return with_float_contract(b->CreateIntrinsic(llvm::Intrinsic::minnum, {left->getType()}, {left, right}));
        case ArithBinaryOp::Max:
            return with_float_contract(b->CreateIntrinsic(llvm::Intrinsic::maxnum, {left->getType()}, {left, right}));
        case ArithBinaryOp::And: {
            Ty *ty = left->getType();
            Ty *intTy = nullptr;
            if (ty->isVectorTy()) {
                Ty *intType = b->getIntNTy(ty->getScalarSizeInBits());
                intTy = llvm::VectorType::get(intType, llvm::cast<llvm::VectorType>(ty)->getElementCount());
            } else {
                intTy = b->getIntNTy(ty->getPrimitiveSizeInBits());
            }
            V *lhsInt = b->CreateBitCast(left, intTy);
            V *rhsInt = b->CreateBitCast(right, intTy);
            V *resultInt = b->CreateAnd(lhsInt, rhsInt);
            return b->CreateBitCast(resultInt, ty);
        }
        case ArithBinaryOp::Or: {
            Ty *ty = left->getType();
            Ty *intTy = nullptr;
            if (ty->isVectorTy()) {
                Ty *intType = b->getIntNTy(ty->getScalarSizeInBits());
                intTy = llvm::VectorType::get(intType, llvm::cast<llvm::VectorType>(ty)->getElementCount());
            } else {
                intTy = b->getIntNTy(ty->getPrimitiveSizeInBits());
            }
            V *lhsInt = b->CreateBitCast(left, intTy);
            V *rhsInt = b->CreateBitCast(right, intTy);
            V *resultInt = b->CreateOr(lhsInt, rhsInt);
            return b->CreateBitCast(resultInt, ty);
        }
        case ArithBinaryOp::Xor: {
            Ty *ty = left->getType();
            Ty *intTy = nullptr;
            if (ty->isVectorTy()) {
                Ty *intType = b->getIntNTy(ty->getScalarSizeInBits());
                intTy = llvm::VectorType::get(intType, llvm::cast<llvm::VectorType>(ty)->getElementCount());
            } else {
                intTy = b->getIntNTy(ty->getPrimitiveSizeInBits());
            }
            V *lhsInt = b->CreateBitCast(left, intTy);
            V *rhsInt = b->CreateBitCast(right, intTy);
            V *resultInt = b->CreateXor(lhsInt, rhsInt);
            return b->CreateBitCast(resultInt, ty);
        }
        case ArithBinaryOp::AndNot: {
            Ty *ty = left->getType();
            Ty *intTy = nullptr;
            if (ty->isVectorTy()) {
                Ty *intType = b->getIntNTy(ty->getScalarSizeInBits());
                intTy = llvm::VectorType::get(intType, llvm::cast<llvm::VectorType>(ty)->getElementCount());
            } else {
                intTy = b->getIntNTy(ty->getPrimitiveSizeInBits());
            }
            V *lhsInt = b->CreateBitCast(left, intTy);
            V *rhsInt = b->CreateBitCast(right, intTy);
            V *resultInt = b->CreateAnd(b->CreateNot(lhsInt), rhsInt);
            return b->CreateBitCast(resultInt, ty);
        }
        default: messed_up("Unexpected ArithBinary %s in float context", show_arith_binary_op(op));
        }
        SIMJIT_UNREACHABLE();
    }
    V *arith_binary(ScalarDataType dtype, ArithBinaryOp op, V *left, V *right) {
        switch (op) {
        case ArithBinaryOp::Add: return b->CreateAdd(left, right);
        case ArithBinaryOp::Sub: return b->CreateSub(left, right);
        case ArithBinaryOp::Mul:
        case ArithBinaryOp::Mul64SE:
        case ArithBinaryOp::Mul64ZE: return b->CreateMul(left, right);
        case ArithBinaryOp::Div: return b->CreateSDiv(left, right);
        case ArithBinaryOp::UDiv: return b->CreateUDiv(left, right);
        case ArithBinaryOp::Mod: return b->CreateSRem(left, right);
        case ArithBinaryOp::UMod: return b->CreateURem(left, right);
        case ArithBinaryOp::Min: return b->CreateIntrinsic(llvm::Intrinsic::smin, {left->getType()}, {left, right});
        case ArithBinaryOp::Max: return b->CreateIntrinsic(llvm::Intrinsic::smax, {left->getType()}, {left, right});
        case ArithBinaryOp::UMin: return b->CreateIntrinsic(llvm::Intrinsic::umin, {left->getType()}, {left, right});
        case ArithBinaryOp::UMax: return b->CreateIntrinsic(llvm::Intrinsic::umax, {left->getType()}, {left, right});
        case ArithBinaryOp::And: return b->CreateAnd(left, right);
        case ArithBinaryOp::Or: return b->CreateOr(left, right);
        case ArithBinaryOp::Xor: return b->CreateXor(left, right);
        case ArithBinaryOp::AndNot: return b->CreateAnd(b->CreateNot(left), right);
        case ArithBinaryOp::ShiftRightArith:
            return b->CreateAShr(left, b->CreateAnd(right, scalar_dtype_bits(dtype) - 1));
        case ArithBinaryOp::ShiftRightLogical:
            return b->CreateLShr(left, b->CreateAnd(right, scalar_dtype_bits(dtype) - 1));
        case ArithBinaryOp::ShiftLeftLogical:
            return b->CreateShl(left, b->CreateAnd(right, scalar_dtype_bits(dtype) - 1));
        case ArithBinaryOp::RotateLeft:
            return b->CreateIntrinsic(llvm::Intrinsic::fshl, {left->getType()},
                                      {left, left, b->CreateAnd(right, scalar_dtype_bits(dtype) - 1)});
        case ArithBinaryOp::RotateRight:
            return b->CreateIntrinsic(llvm::Intrinsic::fshr, {left->getType()},
                                      {left, left, b->CreateAnd(right, scalar_dtype_bits(dtype) - 1)});
        }
        SIMJIT_UNREACHABLE();
    }

    V *predicate_binary(PredicateBinaryOp op, V *left, V *right) {
        switch (op) {
        case PredicateBinaryOp::And: return b->CreateAnd(left, right);
        case PredicateBinaryOp::Or: return b->CreateOr(left, right);
        case PredicateBinaryOp::Xor: return b->CreateXor(left, right);
        case PredicateBinaryOp::AndNot: return b->CreateAnd(b->CreateNot(left), right);
        case PredicateBinaryOp::XNor: return b->CreateNot(b->CreateXor(left, right));
        }
        SIMJIT_UNREACHABLE();
    }

    V *arith_unary(ArithUnaryOp op, V *arg, bool is_float) {
        switch (op) {
        case ArithUnaryOp::Not:
            if (is_float) {
                llvm::Type *arg_type = arg->getType();
                llvm::Type *int_type = nullptr;
                if (arg_type->isVectorTy()) {
                    llvm::Type *int_scalar_type = b->getIntNTy(arg_type->getScalarSizeInBits());
                    int_type = llvm::VectorType::get(int_scalar_type,
                                                     llvm::cast<llvm::VectorType>(arg_type)->getElementCount());
                } else {
                    int_type = b->getIntNTy(arg_type->getPrimitiveSizeInBits());
                }
                V *arg_bits = b->CreateBitCast(arg, int_type);
                V *not_bits = b->CreateNot(arg_bits);
                return b->CreateBitCast(not_bits, arg_type);
            }
            return b->CreateNot(arg);
        case ArithUnaryOp::Negate:
            if (is_float) return with_float_contract(b->CreateFNeg(arg));
            return b->CreateNeg(arg);
        case ArithUnaryOp::Abs:
            if (is_float) return b->CreateIntrinsic(llvm::Intrinsic::fabs, {arg->getType()}, {arg});
            return b->CreateIntrinsic(llvm::Intrinsic::abs, {arg->getType()},
                                      {arg, /* int min is poison */ b->getInt1(false)});
        case ArithUnaryOp::Lzcnt:
            return b->CreateIntrinsic(llvm::Intrinsic::ctlz, {arg->getType()},
                                      {arg, /* zero is poison */ b->getInt1(false)});
        case ArithUnaryOp::Tzcnt:
            return b->CreateIntrinsic(llvm::Intrinsic::cttz, {arg->getType()},
                                      {arg, /* zero is poison */ b->getInt1(false)});
        case ArithUnaryOp::Popcount: return b->CreateIntrinsic(llvm::Intrinsic::ctpop, {arg->getType()}, {arg});
        case ArithUnaryOp::RoundNearest: return b->CreateIntrinsic(llvm::Intrinsic::roundeven, {arg->getType()}, {arg});
        case ArithUnaryOp::RoundDown: return b->CreateIntrinsic(llvm::Intrinsic::floor, {arg->getType()}, {arg});
        case ArithUnaryOp::RoundUp: return b->CreateIntrinsic(llvm::Intrinsic::ceil, {arg->getType()}, {arg});
        case ArithUnaryOp::RoundTruncate: return b->CreateIntrinsic(llvm::Intrinsic::trunc, {arg->getType()}, {arg});
        case ArithUnaryOp::Rcp: {
            Ty *ty = arg->getType();
            if (ty->isVectorTy()) {
                llvm::VectorType *vty = llvm::cast<llvm::VectorType>(ty);
                V *one = llvm::ConstantVector::getSplat(vty->getElementCount(),
                                                        llvm::ConstantFP::get(vty->getElementType(), 1.0));
                return with_float_contract(b->CreateFDiv(one, arg), true);
            }
            return with_float_contract(b->CreateFDiv(llvm::ConstantFP::get(ty, 1.0), arg), true);
        }
        case ArithUnaryOp::Sqrt:
            return with_float_contract(b->CreateIntrinsic(llvm::Intrinsic::sqrt, {arg->getType()}, {arg}));
        case ArithUnaryOp::Rsqrt:
            Ty *ty = arg->getType();
            V *sq = with_float_contract(b->CreateIntrinsic(llvm::Intrinsic::sqrt, {ty}, {arg}), true);
            if (ty->isVectorTy()) {
                llvm::VectorType *vty = llvm::cast<llvm::VectorType>(ty);
                V *one = llvm::ConstantVector::getSplat(vty->getElementCount(),
                                                        llvm::ConstantFP::get(vty->getElementType(), 1.0));
                return with_float_contract(b->CreateFDiv(one, sq), true);
            }
            return with_float_contract(b->CreateFDiv(llvm::ConstantFP::get(ty, 1.0), sq), true);
        }
        SIMJIT_UNREACHABLE();
    }

    V *float_cast(V *arg, ScalarDataType from, ScalarDataType to, Ty *dest_type, bool is_unsigned) {
        switch (from) {
        case ScalarDataType::I32:
        case ScalarDataType::I64:
            switch (to) {
            case ScalarDataType::F32:
            case ScalarDataType::F64:
                if (is_unsigned)
                    return b->CreateUIToFP(arg, dest_type);
                else
                    return b->CreateSIToFP(arg, dest_type);
            default: break;
            }
            break;
        case ScalarDataType::F32:
        case ScalarDataType::F64:
            switch (to) {
            case ScalarDataType::I32:
            case ScalarDataType::I64:
                if (is_unsigned)
                    return b->CreateFreeze(b->CreateFPToUI(arg, dest_type));
                else
                    return b->CreateFreeze(b->CreateFPToSI(arg, dest_type));
                break;
            case ScalarDataType::F32:
            case ScalarDataType::F64: return b->CreateFPCast(arg, dest_type);
            default: break;
            }
            break;
        default: break;
        }
        messed_up("Invalid float cast from {} to {}");
    }

    V *index_arg(const ArgInfo &info, V *idx) {
        SIMJIT_ASSERT(info.element_type != b->getVoidTy());
        return b->CreateInBoundsGEP(info.element_type, info.data_ptr, idx);
    }

    V *load_acc(AccId id) {
        size_t idx = func->accs.index(id);
        SIMJIT_ASSERT(idx < accs.size());
        const AccInfo &acc = accs[idx];
        return b->CreateLoad(acc.type, acc.val_ptr);
    }
    void store_acc(AccId id, V *val) {
        size_t idx = func->accs.index(id);
        SIMJIT_ASSERT(idx < accs.size());
        const AccInfo &acc = accs[idx];
        b->CreateStore(val, acc.val_ptr);
    }

    V *fma_to_llvm(const Step *step, const FMAData &data) {
        V *x1 = step_to_llvm(data.x1);
        V *x2 = step_to_llvm(data.x2);
        V *x3 = step_to_llvm(data.x3);
        bool is_fp =
            step->dtype.is_scalar() ? is_float_dtype(step->dtype.as_scalar()) : step->dtype.as_vec().is_float();

        switch (data.kind) {
        case mir::FmaKind::FMA: break;
        case mir::FmaKind::FMS: x3 = is_fp ? with_float_contract(b->CreateFNeg(x3)) : b->CreateNeg(x3); break;
        case mir::FmaKind::FNMA: x1 = is_fp ? with_float_contract(b->CreateFNeg(x1)) : b->CreateNeg(x1); break;
        case mir::FmaKind::FNMS:
            x1 = is_fp ? with_float_contract(b->CreateFNeg(x1)) : b->CreateNeg(x1);
            x3 = is_fp ? with_float_contract(b->CreateFNeg(x3)) : b->CreateNeg(x3);
            break;
        }

        if (is_fp) {
            return with_float_contract(b->CreateIntrinsic(llvm::Intrinsic::fma, {x1->getType()}, {x1, x2, x3}));
        }
        return b->CreateAdd(b->CreateMul(x1, x2), x3);
    }

    V *step_to_llvm(const Step *step) {
        if (auto *it = value_map[step->id]) { return it; }
        V *v = step_to_llvm_internal(step);
        value_map[step->id] = v;
        return v;
    }

    V *scalar_step_to_llvm_internal(const Step *step) {
        switch (step->kind) {
            SIMJIT_MATCH (StepKind::Const) {
                if (step->dtype == ScalarDataType::I1) { return b->getInt1(data.as_unsigned() != 0); }
                if (step->dtype == ScalarDataType::F32) {
                    return llvm::ConstantFP::get(*context, llvm::APFloat(data.as_f32()));
                }
                if (step->dtype == ScalarDataType::F64) {
                    return llvm::ConstantFP::get(*context, llvm::APFloat(data.as_f64()));
                }

                return llvm::ConstantInt::getSigned(scalar_type_to_llvm(step->dtype.as_scalar()), data.as_signed());
            }
            SIMJIT_MATCH (StepKind::Load) {
                ArgInfo &info = func_args[data.addr.arg];
                V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                if (step->dtype == ScalarDataType::I1) {
                    V *idx = b->CreateLShr(off, 6);
                    V *result =
                        b->CreateLoad(b->getInt64Ty(), b->CreateInBoundsGEP(b->getInt64Ty(), info.data_ptr, idx));
                    result = b->CreateLShr(result, b->CreateAnd(off, 63));
                    result = b->CreateICmpEQ(b->CreateAnd(result, b->getInt64(1)), b->getInt64(1));
                    return result;
                }
                if (data.addr.offset != 0) { off = b->CreateNUWAdd(off, b->getInt64(data.addr.offset)); }
                return b->CreateLoad(info.element_type, index_arg(info, off));
            }
            SIMJIT_MATCH (StepKind::LoadSplat) {
                ArgInfo &info = func_args[data.addr.arg];
                if (step->dtype == ScalarDataType::I1) {
                    return b->CreateTrunc(b->CreateLoad(b->getInt8Ty(), info.data_ptr), b->getInt1Ty());
                }
                return b->CreateLoad(info.element_type, info.data_ptr);
            }
            SIMJIT_MATCH (StepKind::Gather) {
                ArgInfo &info = func_args[data.data];
                V *idx_value = step_to_llvm(data.idx);
                return b->CreateLoad(info.element_type, index_arg(info, idx_value));
            }
            SIMJIT_MATCH (StepKind::Store) {
                V *arg = step_to_llvm(data.what);
                V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                ArgInfo &info = func_args[data.addr.arg];
                if (step->dtype == ScalarDataType::I1) {
                    V *idx = b->CreateLShr(off, 6);
                    V *mem = b->CreateInBoundsGEP(b->getInt64Ty(), info.data_ptr, idx);
                    V *result = b->CreateLoad(b->getInt64Ty(), mem);
                    V *bit_idx = b->CreateAnd(off, 63);
                    V *bit = b->CreateShl(b->getInt64(1), bit_idx);
                    result = b->CreateAnd(result, b->CreateNot(bit));
                    bit = b->CreateShl(b->CreateIntCast(arg, b->getInt64Ty(), false), bit_idx);
                    result = b->CreateOr(result, bit);
                    b->CreateStore(result, mem);
                    return nullptr;
                }

                b->CreateStore(arg, index_arg(info, off));
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                V *arg = step_to_llvm(data.arg);
                ArgInfo &info = func_args[data.dst];
                V *off = step_to_llvm(data.idx);
                b->CreateStore(arg, index_arg(info, off));
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::CondScatter) {
                ArgInfo &info = func_args[data.dst];
                llvm::BasicBlock *then = llvm::BasicBlock::Create(*context, "then", function);
                llvm::BasicBlock *merge = llvm::BasicBlock::Create(*context, "merge", function);
                V *arg = step_to_llvm(data.arg);
                V *off = step_to_llvm(data.idx);
                V *cond = b->CreateIntCast(step_to_llvm(data.cond), b->getInt1Ty(), false);

                b->CreateCondBr(cond, then, merge);
                b->SetInsertPoint(then);

                b->CreateStore(arg, index_arg(info, off));
                b->CreateBr(merge);

                b->SetInsertPoint(merge);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::Pack) {
                ArgInfo &info = func_args[data.dst];
                llvm::BasicBlock *then = llvm::BasicBlock::Create(*context, "then", function);
                llvm::BasicBlock *merge = llvm::BasicBlock::Create(*context, "merge", function);
                V *arg = step_to_llvm(data.arg);
                V *cond = b->CreateIntCast(step_to_llvm(data.cond), b->getInt1Ty(), false);

                b->CreateCondBr(cond, then, merge);
                b->SetInsertPoint(then);

                AccInfo &acc = accs[func->accs.index(data.acc)];
                V *acc_value = b->CreateLoad(acc.type, acc.val_ptr);
                b->CreateStore(arg, index_arg(info, acc_value));
                V *new_acc_value = b->CreateAdd(acc_value, llvm::ConstantInt::get(acc_value->getType(), 1));
                b->CreateStore(new_acc_value, acc.val_ptr);
                b->CreateBr(merge);

                b->SetInsertPoint(merge);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::CondStore) {
                ArgInfo &info = func_args[data.addr.arg];
                llvm::BasicBlock *then = llvm::BasicBlock::Create(*context, "then", function);
                llvm::BasicBlock *merge = llvm::BasicBlock::Create(*context, "merge", function);
                V *arg = step_to_llvm(data.arg);
                V *cond = b->CreateIntCast(step_to_llvm(data.cond), b->getInt1Ty(), false);

                b->CreateCondBr(cond, then, merge);
                b->SetInsertPoint(then);
                V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                b->CreateStore(arg, index_arg(info, off));
                b->CreateBr(merge);
                b->SetInsertPoint(merge);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                V *left = step_to_llvm(data.left);
                V *right = step_to_llvm(data.right);
                // Implicitly convert i1 operands to i8. This is implicit in other backends, but LLVM requires us to be
                // pedantic.
                auto promote_i1 = [&](Step *s, V *v) -> V * {
                    if (s->dtype == ScalarDataType::I1) {
                        SIMJIT_ASSERT(step->dtype == ScalarDataType::I8);
                        return b->CreateZExt(v, b->getInt8Ty());
                    }
                    return v;
                };
                left = promote_i1(data.left, left);
                right = promote_i1(data.right, right);

                if (is_float_dtype(step->dtype.as_scalar())) return float_arith_binary(data.op, left, right);
                return arith_binary(step->dtype.as_scalar(), data.op, left, right);
            }
            SIMJIT_MATCH (StepKind::FMA) { return fma_to_llvm(step, data); }
            SIMJIT_MATCH (StepKind::ConstDiv) {
                return arith_binary(step->dtype.as_scalar(), data.op, step_to_llvm(data.numerator),
                                    step_to_llvm(data.divisor));
            }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                V *arg = step_to_llvm(data.arg);
                return arith_unary(data.op, arg, is_float_dtype(step->dtype.as_scalar()));
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                V *arg = step_to_llvm(data.arg);
                ScalarDataType to = step->dtype.as_scalar();
                Ty *to_type = scalar_type_to_llvm(to);

                switch (data.kind) {
                case IntCastKind::Trunc: return b->CreateTrunc(arg, to_type);
                case IntCastKind::Sext: return b->CreateSExt(arg, to_type);
                case IntCastKind::Zext: return b->CreateZExt(arg, to_type);
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                V *arg = step_to_llvm(data.arg);
                ScalarDataType from = data.arg->dtype.as_scalar();
                ScalarDataType to = step->dtype.as_scalar();
                Ty *to_type = scalar_type_to_llvm(to);

                return float_cast(arg, from, to, to_type, data.is_unsigned);
            }
            SIMJIT_MATCH (StepKind::BitCast) {
                return b->CreateBitCast(step_to_llvm(data), scalar_type_to_llvm(step->dtype.as_scalar()));
            }
            SIMJIT_MATCH (StepKind::Compare) {
                V *left = step_to_llvm(data.left);
                V *right = step_to_llvm(data.right);
                if (is_float_dtype(data.left->dtype.as_scalar()))
                    return b->CreateFCmp(compare_to_llvm_float(data.op), left, right);
                return b->CreateICmp(compare_to_llvm_int(data.op, data.is_unsigned), left, right);
            }
            SIMJIT_MATCH (StepKind::AggResult) {
                V *arg = step_to_llvm(data.arg);
                ArgInfo &info = func_args[data.dst];
                b->CreateStore(arg, info.data_ptr);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::StoreSum128) {
                V *high = step_to_llvm(data.hi_combined);
                V *combined = b->CreateShl(b->CreateZExt(high, b->getInt128Ty()), 64);
                for (Step *lo_step : data.low_steps) {
                    V *lo_val = step_to_llvm(lo_step);
                    if (lo_step->dtype.is_scalar()) {
                        lo_val = b->CreateZExt(lo_val, b->getInt128Ty());
                        combined = b->CreateAdd(combined, lo_val);
                    } else {
                        VecDataType vdtype = lo_step->dtype.as_vec();
                        for (size_t i = 0; i < vdtype.nelems(); ++i) {
                            V *it = b->CreateExtractElement(lo_val, b->getInt64(i));
                            it = b->CreateZExt(it, b->getInt128Ty());
                            combined = b->CreateAdd(combined, it);
                        }
                    }
                }

                ArgInfo &info = func_args[data.dst];
                b->CreateStore(combined, info.data_ptr);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::PredicateNot) {
                V *arg = step_to_llvm(data);
                // LLVM does not have predicate not
                return b->CreateICmpEQ(arg, llvm::ConstantInt::get(arg->getType(), 0));
            }
            SIMJIT_MATCH (StepKind::Select) {
                V *cond = step_to_llvm(data.cond);
                V *left = step_to_llvm(data.falsy);
                V *right = step_to_llvm(data.truthy);
                // cond might not be i1 because in MIR we converted predicate binary steps to arith binary
                auto truncate_i1 = [&](Step *s, V *v) -> V * {
                    if (s->dtype != ScalarDataType::I1) {
                        SIMJIT_ASSERT(s->dtype == ScalarDataType::I8);
                        return b->CreateTrunc(v, b->getInt1Ty());
                    }
                    return v;
                };
                cond = truncate_i1(data.cond, cond);
                return b->CreateSelect(cond, right, left);
            }
            SIMJIT_MATCH (StepKind::AccLoad) { return load_acc(data); }
            SIMJIT_MATCH (StepKind::AccStore) {
                V *arg = step_to_llvm(data.arg);
                store_acc(data.acc, arg);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::ScalarIndex) {
                V *v = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                if (step->dtype != ScalarDataType::I64) {
                    return b->CreateTrunc(v, scalar_type_to_llvm(step->dtype.as_scalar()));
                }
                return v;
            }
            SIMJIT_MATCH (StepKind::ScalarArithBinaryOverflow) {
                int intrin = 0;
                switch (data.op) {
                case ArithBinaryOp::Add: intrin = llvm::Intrinsic::sadd_with_overflow; break;
                case ArithBinaryOp::Sub: intrin = llvm::Intrinsic::ssub_with_overflow; break;
                case ArithBinaryOp::Mul: intrin = llvm::Intrinsic::smul_with_overflow; break;
                default: messed_up("Invalid overflow step %s", show_arith_binary_op(data.op));
                }

                V *left = step_to_llvm(data.left);
                V *right = step_to_llvm(data.right);
                V *called = b->CreateIntrinsic(intrin, scalar_type_to_llvm(step->dtype.as_scalar()), {left, right});
                V *overflow = b->CreateExtractValue(called, 1);
                if (data.mask != nullptr) {
                    V *mask = step_to_llvm(data.mask);
                    mask = b->CreateICmpNE(mask, llvm::ConstantInt::get(mask->getType(), 0));
                    overflow = b->CreateAnd(overflow, mask);
                }
                overflow = b->CreateZExt(overflow, b->getInt8Ty());

                V *acc_value = load_acc(data.overflow_flag);
                V *new_acc_value = b->CreateOr(acc_value, overflow);
                store_acc(data.overflow_flag, new_acc_value);

                V *result = b->CreateExtractValue(called, 0);
                return result;
            }
            SIMJIT_MATCH (StepKind::ScalarPermute) {
                ScalarDataType sdtype = step->dtype.as_scalar();
                V *arg = step_to_llvm(data.arg);
                if (data.is_bit) {
                    if (data.permute == REVERSE_BITS) {
                        Ty *ty = arg->getType();
                        llvm::VectorType *vty = llvm::VectorType::get(
                            b->getInt8Ty(), llvm::ElementCount::getFixed(scalar_dtype_size(sdtype)));
                        arg = b->CreateBitCast(arg, vty);
                        V *res = b->CreateIntrinsic(llvm::Intrinsic::bitreverse, {vty}, {arg});
                        res = b->CreateBitCast(res, ty);
                        return res;
                    }
                } else {
                    if ((sdtype == ScalarDataType::I16 && data.permute == REVERSE_BYTES_I16) ||
                        (sdtype == ScalarDataType::I32 && data.permute == REVERSE_BYTES_I32) ||
                        (sdtype == ScalarDataType::I64 && data.permute == REVERSE_BYTES_I64)) {
                        return b->CreateIntrinsic(llvm::Intrinsic::bswap, {arg->getType()}, {arg});
                    }
                }
                V *accumulated = nullptr;
                auto accumulate = [&](V *x) {
                    if (!accumulated) {
                        accumulated = x;
                    } else {
                        accumulated = b->CreateOr(accumulated, x);
                    }
                };
                size_t dtype_size = scalar_dtype_size(step->dtype.as_scalar());
                if (data.is_bit) {
                    for (size_t i = 0; i < dtype_size * 8; ++i) {
                        size_t permute_idx = (((data.permute >> ((i & 0x7) * 8)) - 1) & 0xff);
                        V *x = b->CreateLShr(arg, permute_idx + (i / 8 * 8));
                        x = b->CreateAnd(x, 1);
                        x = b->CreateShl(x, i);
                        accumulate(x);
                    }
                } else {
                    for (size_t i = 0; i < dtype_size; ++i) {
                        V *x = b->CreateLShr(arg, ((data.permute >> (i * 8)) & 0xff) * 8);
                        x = b->CreateAnd(x, 0xff);
                        x = b->CreateShl(x, i * 8);
                        accumulate(x);
                    }
                }
                return accumulated;
            }
            SIMJIT_MATCH (StepKind::Fpclass) { return fpclass(step_to_llvm(data.arg), data.flags); }
        default: messed_up("Unexpected step %s in scalar context", show_step_kind(step->kind));
        }
    }

    V *fpclass(V *arg, FpClass flags) {
        constexpr uint32_t LLVM_FC_SNAN = 1;
        constexpr uint32_t LLVM_FC_QNAN = 2;
        constexpr uint32_t LLVM_FC_POS_ZERO = 64;
        constexpr uint32_t LLVM_FC_NEG_ZERO = 32;
        constexpr uint32_t LLVM_FC_POS_INF = 512;
        constexpr uint32_t LLVM_FC_NEG_INF = 4;
        constexpr uint32_t LLVM_FC_POS_SUBNORMAL = 128;
        constexpr uint32_t LLVM_FC_NEG_SUBNORMAL = 16;

        uint32_t mask = 0;
        if (bool(flags & FpClass::FPC_INFINITE)) mask |= (LLVM_FC_POS_INF | LLVM_FC_NEG_INF);
        if (bool(flags & FpClass::FPC_NAN)) mask |= (LLVM_FC_QNAN | LLVM_FC_SNAN);
        if (bool(flags & FpClass::FPC_SUBNORMAL)) mask |= (LLVM_FC_POS_SUBNORMAL | LLVM_FC_NEG_SUBNORMAL);
        if (bool(flags & FpClass::FPC_ZERO)) mask |= (LLVM_FC_POS_ZERO | LLVM_FC_NEG_ZERO);

        return b->CreateIntrinsic(llvm::Intrinsic::is_fpclass, {arg->getType()}, {arg, b->getInt32(mask)});
    }

    V *step_to_llvm_internal(const Step *step) {
        if (step->dtype.is_scalar() && is_scalar_step(step->kind)) { return scalar_step_to_llvm_internal(step); }
        switch (step->kind) {
        case StepKind::AggResult:
        case StepKind::ScalarIndex:
        case StepKind::ScalarArithBinaryOverflow:
        case StepKind::ScalarPermute:
        case StepKind::StoreSum128:
        case StepKind::ConstDiv:
            SIMJIT_ASSERT(0);
            messed_up("Unexpected instruction %s", show_step_kind(step->kind));

            SIMJIT_MATCH (StepKind::LoadSplat) {
                ArgInfo &info = func_args[data.addr.arg];
                if (step->dtype.is_mask()) {
                    V *s = b->CreateLoad(b->getInt8Ty(), info.data_ptr);
                    s = b->CreateTrunc(s, b->getInt1Ty());
                    return b->CreateVectorSplat(mask_dtype_bits(step->dtype.as_mask()), s);
                }
                V *s = b->CreateLoad(info.element_type, info.data_ptr);
                return b->CreateVectorSplat(step->dtype.as_vec().nelems(), s);
            }
            SIMJIT_MATCH (StepKind::Const) {
                if (step->dtype.is_mask()) {
                    llvm::VectorType *llvm_type = mask_type_to_llvm(step->dtype.as_mask());
                    return llvm::ConstantVector::getSplat(
                        llvm_type->getElementCount(),
                        llvm::ConstantInt::getSigned(llvm_type->getElementType(), data.as_signed()));
                }

                VecDataType dtype = step->dtype.as_vec();
                llvm::VectorType *llvm_type = vec_type_to_llvm(dtype);
                if (data.is_zero()) { return llvm::ConstantAggregateZero::get(llvm_type); }
                if (dtype.elem == VecElemType::F32) {
                    return llvm::ConstantVector::getSplat(
                        llvm_type->getElementCount(),
                        llvm::ConstantFP::get(llvm_type->getElementType(), llvm::APFloat(data.as_f32())));
                }
                if (dtype.elem == VecElemType::F64) {
                    return llvm::ConstantVector::getSplat(
                        llvm_type->getElementCount(),
                        llvm::ConstantFP::get(llvm_type->getElementType(), llvm::APFloat(data.as_f64())));
                }
                return llvm::ConstantVector::getSplat(
                    llvm_type->getElementCount(),
                    llvm::ConstantInt::getSigned(llvm_type->getElementType(), data.as_signed()));
            }
            SIMJIT_MATCH (StepKind::VecIndex) {
                V *cur = load_acc(data.acc);
                V *inc = step_to_llvm(data.inc);
                V *updated = b->CreateNUWAdd(cur, inc);
                store_acc(data.acc, updated);
                return cur;
            }
            SIMJIT_MATCH (StepKind::Load) {
                ArgInfo &info = func_args[data.addr.arg];
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    llvm::VectorType *llvm_type = mask_type_to_llvm(mdtype);
                    V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                    if (data.addr.offset != 0) { off = b->CreateNUWAdd(off, b->getInt64(data.addr.offset)); }
                    if (mdtype == MaskDataType::M2 || mdtype == MaskDataType::M4) {
                        V *byte_off = b->CreateLShr(off, b->getInt64(3));
                        V *bit_off = b->CreateTrunc(b->CreateAnd(off, b->getInt64(7)), b->getInt8Ty());
                        V *byte = b->CreateLoad(b->getInt8Ty(),
                                                b->CreateInBoundsGEP(b->getInt8Ty(), info.data_ptr, byte_off));
                        V *shifted = b->CreateLShr(byte, bit_off);
                        V *packed = b->CreateTrunc(shifted, b->getIntNTy(mask_dtype_bits(mdtype)));
                        return b->CreateBitCast(packed, llvm_type);
                    }
                    off = b->CreateLShr(off, b->getInt64(mask_dtype_bits_log2(mdtype)));
                    V *v = b->CreateLoad(llvm_type, b->CreateInBoundsGEP(llvm_type, info.data_ptr, off));
                    return v;
                }
                llvm::VectorType *llvm_type = vec_type_to_llvm(step->dtype.as_vec());
                V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                if (data.addr.offset != 0) { off = b->CreateNUWAdd(off, b->getInt64(data.addr.offset)); }
                V *idx = index_arg(info, off);
                llvm::LoadInst *inst = nullptr;
                if (data.kind == LoadStoreKind::Unaligned) {
                    inst = b->CreateLoad(llvm_type, idx);
                    inst->setAlignment(llvm::Align{1});
                } else {
                    inst = b->CreateAlignedLoad(llvm_type, idx, {});
                }
                return inst;
            }
            SIMJIT_MATCH (StepKind::ArithBinary) {
                V *left = step_to_llvm(data.left);
                V *right = step_to_llvm(data.right);
                if (step->dtype.as_vec().is_float()) return float_arith_binary(data.op, left, right);
                return arith_binary(step->dtype.as_vec().to_scalar(), data.op, left, right);
            }
            SIMJIT_MATCH (StepKind::FMA) { return fma_to_llvm(step, data); }
            SIMJIT_MATCH (StepKind::ArithUnary) {
                V *arg = step_to_llvm(data.arg);
                return arith_unary(data.op, arg, step->dtype.as_vec().is_float());
            }
            SIMJIT_MATCH (StepKind::Store) {
                V *arg = step_to_llvm(data.what);
                ArgInfo &info = func_args[data.addr.arg];
                V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                if (data.addr.offset != 0) { off = b->CreateNUWAdd(off, b->getInt64(data.addr.offset)); }
                if (step->dtype.is_mask()) {
                    MaskDataType mdtype = step->dtype.as_mask();
                    llvm::VectorType *llvm_type = mask_type_to_llvm(mdtype);
                    if (mdtype == MaskDataType::M2 || mdtype == MaskDataType::M4) {
                        size_t bit_count = mask_dtype_bits(mdtype);
                        V *byte_off = b->CreateLShr(off, b->getInt64(3));
                        V *bit_off = b->CreateAnd(off, b->getInt64(7));
                        V *byte_ptr = b->CreateInBoundsGEP(b->getInt8Ty(), info.data_ptr, byte_off);
                        V *old = b->CreateLoad(b->getInt16Ty(), byte_ptr);
                        V *shift = b->CreateTrunc(bit_off, b->getInt16Ty());
                        V *mask = b->CreateShl(b->getInt16((1u << bit_count) - 1), shift);
                        V *packed = b->CreateBitCast(arg, b->getIntNTy(bit_count));
                        V *shifted = b->CreateShl(b->CreateZExt(packed, b->getInt16Ty()), shift);
                        V *result = b->CreateOr(b->CreateAnd(old, b->CreateNot(mask)), shifted);
                        b->CreateStore(result, byte_ptr);
                        return nullptr;
                    }
                    off = b->CreateAShr(off, b->getInt64(mask_dtype_bits_log2(mdtype)));
                    return b->CreateStore(arg, b->CreateInBoundsGEP(llvm_type, info.data_ptr, off));
                }
                V *idx = index_arg(info, off);
                if (data.kind == LoadStoreKind::Unaligned) {
                    auto *inst = b->CreateStore(arg, idx);
                    inst->setAlignment(llvm::Align{1});
                } else {
                    b->CreateAlignedStore(arg, idx, {});
                }
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::CondStore) {
                V *arg = step_to_llvm(data.arg);
                ArgInfo &info = func_args[data.addr.arg];
                V *off = b->CreateLoad(b->getInt64Ty(), counter_ptr);
                V *mask = step_to_llvm(data.cond);
                if (data.addr.offset != 0) { off = b->CreateNUWAdd(off, b->getInt64(data.addr.offset)); }
                if (data.kind == LoadStoreKind::Aligned) {
                    b->CreateMaskedStore(arg, index_arg(info, off), llvm::Align{step->dtype.as_vec().size_bytes()},
                                         mask);
                } else {
                    b->CreateMaskedStore(arg, index_arg(info, off), llvm::Align{1}, mask);
                }
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::Gather) {
                V *index = step_to_llvm(data.idx);
                VecDataType vec_dtype = step->dtype.as_vec();
                const ArgInfo *data_arg = &func_args[data.data];
                return call_gather_intrin(vec_dtype, index, data_arg->data_ptr);
            }
            SIMJIT_MATCH (StepKind::Compare) {
                V *left = step_to_llvm(data.left);
                V *right = step_to_llvm(data.right);
                if (data.left->dtype.as_vec().is_float())
                    return b->CreateFCmp(compare_to_llvm_float(data.op), left, right);
                return b->CreateICmp(compare_to_llvm_int(data.op, data.is_unsigned), left, right);
            }
            SIMJIT_MATCH (StepKind::AccLoad) { return load_acc(data); }
            SIMJIT_MATCH (StepKind::AccStore) {
                V *arg = step_to_llvm(data.arg);
                store_acc(data.acc, arg);
                return nullptr;
            }
            SIMJIT_MATCH2 (StepKind::VecWidenHighHalf, StepKind::VecFloatWidenHighHalf) {
                VecDataType vdtype = step->dtype.as_vec();
                auto maybe_extracted_type = vec_dtype_half(data.arg->dtype.as_vec());
                SIMJIT_ASSERT(maybe_extracted_type.has_value());
                VecDataType extracted_type = *maybe_extracted_type;
                V *half = b->CreateExtractVector(vec_type_to_llvm(extracted_type), step_to_llvm(data.arg),
                                                 b->getInt64(step->dtype.as_vec().nelems()));
                llvm::Type *dst_type = vec_type_to_llvm(vdtype);
                if (vdtype.is_float()) { return b->CreateFPCast(half, dst_type); }
                if (data.is_unsigned) { return b->CreateZExt(half, dst_type); }
                return b->CreateSExt(half, dst_type);
            }
            SIMJIT_MATCH2 (StepKind::VecWidenLowHalf, StepKind::VecFloatWidenLowHalf) {
                VecDataType vdtype = step->dtype.as_vec();
                auto maybe_extracted_type = vec_dtype_half(data.arg->dtype.as_vec());
                SIMJIT_ASSERT(maybe_extracted_type.has_value());
                VecDataType extracted_type = *maybe_extracted_type;
                V *half =
                    b->CreateExtractVector(vec_type_to_llvm(extracted_type), step_to_llvm(data.arg), b->getInt64(0));
                llvm::Type *dst_type = vec_type_to_llvm(vdtype);
                if (vdtype.is_float()) { return b->CreateFPCast(half, dst_type); }
                if (data.is_unsigned) { return b->CreateZExt(half, dst_type); }
                return b->CreateSExt(half, dst_type);
            }
            SIMJIT_MATCH (StepKind::VecNarrowCombine) {
                VecDataType vdtype = step->dtype.as_vec();
                auto maybe_half_type = vec_dtype_half(vdtype);
                SIMJIT_ASSERT(maybe_half_type.has_value());
                VecDataType half_type = *maybe_half_type;
                Ty *half_llvm = vec_type_to_llvm(half_type);
                V *low = b->CreateTrunc(step_to_llvm(data.low), half_llvm);
                V *high = b->CreateTrunc(step_to_llvm(data.high), half_llvm);
                llvm::SmallVector<int, 64> shuf;
                for (size_t i = 0; i < vdtype.nelems(); ++i) {
                    shuf.push_back((int)i);
                }
                return b->CreateShuffleVector(low, high, shuf);
            }
            SIMJIT_MATCH (StepKind::VecFloatNarrowCombine) {
                VecDataType vdtype = step->dtype.as_vec();
                auto maybe_half_type = vec_dtype_half(vdtype);
                SIMJIT_ASSERT(maybe_half_type.has_value());
                VecDataType half_type = *maybe_half_type;
                Ty *half_llvm = vec_type_to_llvm(half_type);
                V *low = b->CreateFPTrunc(step_to_llvm(data.low), half_llvm);
                V *high = b->CreateFPTrunc(step_to_llvm(data.high), half_llvm);
                llvm::SmallVector<int, 64> shuf;
                for (size_t i = 0; i < vdtype.nelems(); ++i) {
                    shuf.push_back((int)i);
                }
                return b->CreateShuffleVector(low, high, shuf);
            }
            SIMJIT_MATCH (StepKind::IntCast) {
                V *arg = step_to_llvm(data.arg);
                VecDataType vdtype = step->dtype.as_vec();
                switch (data.kind) {
                case IntCastKind::Trunc: return b->CreateTrunc(arg, vec_type_to_llvm(vdtype));
                case IntCastKind::Sext: return b->CreateSExt(arg, vec_type_to_llvm(vdtype));
                case IntCastKind::Zext: return b->CreateZExt(arg, vec_type_to_llvm(vdtype));
                }
                SIMJIT_UNREACHABLE();
            }
            SIMJIT_MATCH (StepKind::FloatCast) {
                VecDataType vdtype = step->dtype.as_vec();
                V *arg = step_to_llvm(data.arg);
                ScalarDataType from = data.arg->dtype.as_vec().to_scalar();
                ScalarDataType to = step->dtype.as_vec().to_scalar();
                Ty *to_type = vec_type_to_llvm(vdtype);
                return float_cast(arg, from, to, to_type, data.is_unsigned);
            }
            SIMJIT_MATCH (StepKind::PredicateNot) { return b->CreateNot(step_to_llvm(data)); }
            SIMJIT_MATCH (StepKind::MaskBinary) {
                return predicate_binary(data.op, step_to_llvm(data.left), step_to_llvm(data.right));
            }
            SIMJIT_MATCH (StepKind::MaskCount) {
                MaskDataType mdtype = data->dtype.as_mask();
                ScalarDataType sdtype = step->dtype.as_scalar();
                V *cond = step_to_llvm(data);
                V *mask_int = b->CreateBitCast(cond, b->getIntNTy(mask_dtype_bits(mdtype)));
                V *ctpop = b->CreateIntrinsic(llvm::Intrinsic::ctpop, {mask_int->getType()}, {mask_int});
                if (mask_dtype_bits(mdtype) != scalar_dtype_bits(sdtype)) {
                    return b->CreateZExt(ctpop, scalar_type_to_llvm(step->dtype.as_scalar()));
                }
                return ctpop;
            }
            SIMJIT_MATCH (StepKind::BitCast) {
                return b->CreateBitCast(step_to_llvm(data), vec_type_to_llvm(step->dtype.as_vec()));
            }
            SIMJIT_MATCH (StepKind::VecReduce) {
                VecDataType vdtype = data.arg->dtype.as_vec();
                Ty *ty = vec_type_to_llvm(data.arg->dtype.as_vec());
                llvm::VectorType *vty = llvm::cast<llvm::VectorType>(ty);
                auto unary_reduce = [&](unsigned id) { return b->CreateIntrinsic(id, {ty}, {step_to_llvm(data.arg)}); };

                switch (data.op) {
                case ArithBinaryOp::Add:
                    if (vdtype.is_float()) {
                        V *zero = llvm::ConstantFP::getNegativeZero(vty->getElementType());
                        llvm::CallInst *inst = b->CreateIntrinsic(llvm::Intrinsic::vector_reduce_fadd, {ty},
                                                                  {zero, step_to_llvm(data.arg)});
                        return with_float_contract(inst, false, true);
                    }
                    return unary_reduce(llvm::Intrinsic::vector_reduce_add);
                case ArithBinaryOp::Mul:
                    if (vdtype.is_float()) {
                        V *one = llvm::ConstantFP::get(vty->getElementType(), 1.0);
                        llvm::CallInst *inst = b->CreateIntrinsic(llvm::Intrinsic::vector_reduce_fmul, {ty},
                                                                  {one, step_to_llvm(data.arg)});
                        return with_float_contract(inst, false, true);
                    }
                    return unary_reduce(llvm::Intrinsic::vector_reduce_mul);
                case ArithBinaryOp::Min:
                    if (vdtype.is_float()) {
                        return with_float_contract(unary_reduce(llvm::Intrinsic::vector_reduce_fmin));
                    }
                    return unary_reduce(llvm::Intrinsic::vector_reduce_smin);
                case ArithBinaryOp::Max:
                    if (vdtype.is_float()) {
                        return with_float_contract(unary_reduce(llvm::Intrinsic::vector_reduce_fmax));
                    }
                    return unary_reduce(llvm::Intrinsic::vector_reduce_smax);
                case ArithBinaryOp::UMin: return unary_reduce(llvm::Intrinsic::vector_reduce_umin);
                case ArithBinaryOp::UMax: return unary_reduce(llvm::Intrinsic::vector_reduce_umax);
                case ArithBinaryOp::And: return unary_reduce(llvm::Intrinsic::vector_reduce_and);
                case ArithBinaryOp::Or: return unary_reduce(llvm::Intrinsic::vector_reduce_or);
                case ArithBinaryOp::Xor: return unary_reduce(llvm::Intrinsic::vector_reduce_xor);
                default: messed_up("Invalid reduce %s of %s", show_arith_binary_op(data.op), show_vec_dtype(vdtype));
                }
            }
            SIMJIT_MATCH (StepKind::MaskReduce) {
                SIMJIT_ASSERT(step->dtype == ScalarDataType::I8);
                unsigned intrin_id = 0;
                switch (data.op) {
                case PredicateBinaryOp::And: intrin_id = llvm::Intrinsic::vector_reduce_and; break;
                case PredicateBinaryOp::Or: intrin_id = llvm::Intrinsic::vector_reduce_or; break;
                case PredicateBinaryOp::Xor: intrin_id = llvm::Intrinsic::vector_reduce_xor; break;
                case PredicateBinaryOp::AndNot:
                case PredicateBinaryOp::XNor:
                    messed_up("Invalid reduce %s of %s", show_predicate_binary_op(data.op),
                              show_mask_dtype(data.arg->dtype.as_mask()));
                }
                V *result = b->CreateIntrinsic(intrin_id, {mask_type_to_llvm(data.arg->dtype.as_mask())},
                                               {step_to_llvm(data.arg)});
                result = b->CreateIntCast(result, b->getInt8Ty(), false);
                return result;
            }
            SIMJIT_MATCH (StepKind::MaskCombine) {
                return call_kunpack(step->dtype.as_mask(), step_to_llvm(data.left), step_to_llvm(data.right));
            }
            SIMJIT_MATCH (StepKind::Select) {
                return b->CreateSelect(step_to_llvm(data.cond), step_to_llvm(data.truthy), step_to_llvm(data.falsy));
            }
            SIMJIT_MATCH (StepKind::Ternarylogic) {
                VecDataType vdtype = step->dtype.as_vec();
                llvm::Function *f = ternarylogic_intrin(data.lookup_type);
                // for ternarylogic use int32 vector
                V *av = step_to_llvm(data.a);
                V *bv = step_to_llvm(data.b);
                V *cv = step_to_llvm(data.c);
                if (vdtype.elem != VecElemType::I32) {
                    Ty *coerce_type = vec_type_to_llvm(data.lookup_type);
                    av = b->CreateBitCast(av, coerce_type);
                    bv = b->CreateBitCast(bv, coerce_type);
                    cv = b->CreateBitCast(cv, coerce_type);
                }
                V *result = b->CreateCall(f, {av, bv, cv, b->getInt32(data.fun)});
                if (vdtype.elem != VecElemType::I32) { result = b->CreateBitCast(result, vec_type_to_llvm(vdtype)); }
                return result;
            }
            SIMJIT_MATCH (StepKind::VecPermute) {
                V *arg = step_to_llvm(data.arg);
                Ty *target_type = arg->getType();
                VecDataType vdtype = step->dtype.as_vec();
                if (data.is_bit) {
                    if (data.permute == REVERSE_BITS) {
                        llvm::VectorType *vty =
                            llvm::VectorType::get(b->getInt8Ty(), llvm::ElementCount::getFixed(vdtype.size_bytes()));
                        arg = b->CreateBitCast(arg, vty);
                        V *res = b->CreateIntrinsic(llvm::Intrinsic::bitreverse, {vty}, {arg});
                        return b->CreateBitCast(res, target_type);
                    }
                } else {
                    if ((vdtype.elem == VecElemType::I16 && data.permute == REVERSE_BYTES_I16) ||
                        (vdtype.elem == VecElemType::I32 && data.permute == REVERSE_BYTES_I32) ||
                        (vdtype.elem == VecElemType::I64 && data.permute == REVERSE_BYTES_I64)) {
                        return b->CreateIntrinsic(llvm::Intrinsic::bswap, {arg->getType()}, {arg});
                    }
                }

                if (ctx->arch == Arch::Arm64_NEON) {
                    if (!data.is_bit) {
                        Ty *ty =
                            llvm::VectorType::get(b->getInt8Ty(), llvm::ElementCount::getFixed(vdtype.size_bytes()));
                        V *right = step_to_llvm(data.permute_idxs);
                        V *left = b->CreateBitCast(arg, ty);
                        right = b->CreateBitCast(right, ty);
                        V *x = b->CreateIntrinsic(llvm::Intrinsic::aarch64_neon_tbl1, {ty}, {left, right});
                        return b->CreateBitCast(x, target_type);
                    }
                    unsupported("Do not support arbitrary bit permutes (ARM)");
                }

                V *left = arg;
                V *right = step_to_llvm(data.permute_idxs);
                x86::Vector vec = x86::vec_to_x86(vdtype);
                if (data.is_bit) {
                    unsigned intrin = 0;
                    Ty *ty = nullptr;
                    switch (vec.reg) {
                    case x86::VecRegisterKind::XMM:
                        intrin = llvm::Intrinsic::x86_vgf2p8affineqb_128;
                        ty = vec_type_to_llvm(x86::types::XMMI8);
                        break;
                    case x86::VecRegisterKind::YMM:
                        intrin = llvm::Intrinsic::x86_vgf2p8affineqb_256;
                        ty = vec_type_to_llvm(x86::types::YMMI8);
                        break;
                    case x86::VecRegisterKind::ZMM:
                        intrin = llvm::Intrinsic::x86_vgf2p8affineqb_512;
                        ty = vec_type_to_llvm(x86::types::ZMMI8);
                        break;
                    }
                    left = b->CreateBitCast(left, ty);
                    right = b->CreateBitCast(right, ty);
                    V *x = b->CreateIntrinsic(intrin, {}, {left, right, b->getInt8(0)});
                    return b->CreateBitCast(x, target_type);
                }
                unsigned intrin = 0;
                Ty *ty = nullptr;
                switch (vec.reg) {
                case x86::VecRegisterKind::XMM:
                    intrin = llvm::Intrinsic::x86_ssse3_pshuf_b_128;
                    ty = vec_type_to_llvm(x86::types::XMMI8);
                    break;
                case x86::VecRegisterKind::YMM:
                    intrin = llvm::Intrinsic::x86_avx2_pshuf_b;
                    ty = vec_type_to_llvm(x86::types::YMMI8);
                    break;
                case x86::VecRegisterKind::ZMM:
                    intrin = llvm::Intrinsic::x86_avx512_pshuf_b_512;
                    ty = vec_type_to_llvm(x86::types::ZMMI8);
                    break;
                }
                left = b->CreateBitCast(left, ty);
                right = b->CreateBitCast(right, ty);
                V *x = b->CreateIntrinsic(intrin, {}, {left, right});
                return b->CreateBitCast(x, target_type);
                // TODO: We can use shufflevector here
            }
            SIMJIT_MATCH (StepKind::VecConst) {
                VecDataType vdtype = step->dtype.as_vec();
                llvm::Constant *constants[64];
                for (size_t i = 0; i < vdtype.size_bytes(); ++i) {
                    constants[i] = b->getInt8(((uint8_t *)data.mem)[i]);
                }

                V *x = llvm::ConstantVector::get(llvm::ArrayRef<llvm::Constant *>{constants, vdtype.size_bytes()});
                return b->CreateBitCast(x, vec_type_to_llvm(vdtype));
            }
            SIMJIT_MATCH (StepKind::Pack) {
                MaskDataType mdtype = data.cond->dtype.as_mask();
                ArgInfo &info = func_args[data.dst];
                V *arg = step_to_llvm(data.arg);
                V *cond = step_to_llvm(data.cond);
                V *acc_value = load_acc(data.acc);
                V *off = index_arg(info, acc_value);
                b->CreateMaskedCompressStore(arg, off, {}, cond);
                V *mask_int = b->CreateBitCast(cond, b->getIntNTy(mask_dtype_bits(mdtype)));
                V *count = b->CreateIntrinsic(mask_int->getType(), llvm::Intrinsic::ctpop, {mask_int});
                if (mdtype != MaskDataType::M64) { count = b->CreateZExt(count, b->getInt64Ty()); }
                acc_value = b->CreateNUWAdd(acc_value, count);
                store_acc(data.acc, acc_value);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::Scatter) {
                VecDataType vdtype = step->dtype.as_vec();
                V *arg = step_to_llvm(data.arg);
                ArgInfo &info = func_args[data.dst];
                V *off = step_to_llvm(data.idx);
                call_scatter_intrin(vdtype, arg, off, info.data_ptr);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::CondScatter) {
                VecDataType vdtype = step->dtype.as_vec();
                V *arg = step_to_llvm(data.arg);
                ArgInfo &info = func_args[data.dst];
                V *off = step_to_llvm(data.idx);
                V *cond = step_to_llvm(data.cond);
                call_cond_scatter_intrin(vdtype, arg, off, info.data_ptr, cond);
                return nullptr;
            }
            SIMJIT_MATCH (StepKind::Fpclass) { return fpclass(step_to_llvm(data.arg), data.flags); }
        }
        SIMJIT_UNREACHABLE();
    }

    void steps_to_llvm(nonstd::span<Step *const> steps) {
        for (Step *root : steps) {
            step_to_llvm(root);
        }
    }
};
} // namespace

static void compile_function(LLVMBuilder &builder, const Function *func) {
    builder.func = func;
    std::vector<Ty *> function_args{};
    function_args.push_back(builder.b->getInt64Ty());
    for (size_t i = 0; i < func->args.size(); ++i) {
        function_args.push_back(llvm::PointerType::get(*builder.context, 0));
    }
    llvm::FunctionType *func_type = llvm::FunctionType::get(builder.b->getVoidTy(), function_args, false);
    builder.function =
        llvm::Function::Create(func_type, llvm::Function::ExternalLinkage, func->ctx->symbol_name, *builder.module);
    {
        llvm::AttributeList attrs = builder.function->getAttributes();
        for (size_t i = 0; i < func->args.size(); ++i) {
            size_t func_attr_idx = i + 1;

            // Create new attribute set for this argument
            llvm::AttrBuilder attr_builder{*builder.context};
            attr_builder.addAttribute(llvm::Attribute::NoAlias);

            // Apply the attributes to the argument
            attrs = attrs.addAttributesAtIndex(*builder.context, func_attr_idx + 1, attr_builder);
        }
        builder.function->setAttributes(attrs);
    }

    llvm::BasicBlock *entry = llvm::BasicBlock::Create(*builder.context, "entry", builder.function);
    builder.b->SetInsertPoint(entry);

    llvm::Argument *args_it = builder.function->arg_begin();
    V *rows_count_arg = args_it++;
    builder.func_args = func->ctx->arena->alloc_array<ArgInfo>(func->args.size());
    for (const ArgumentDecl &arg : func->args) {
        ArgInfo info{};
        llvm::Argument *llvm_arg = args_it++;
        SIMJIT_ASSERT(llvm_arg < builder.function->arg_end());
        info.data_ptr = llvm_arg;
        info.element_type = (arg.dtype == ScalarDataType::I1 || arg.dtype == ScalarDataType::I128)
                                ? builder.b->getVoidTy()
                                : builder.scalar_type_to_llvm(arg.dtype);
        builder.func_args[arg.idx] = info;
    }

    builder.accs = func->ctx->arena->alloc_array<AccInfo>(func->accs.count);
    for (const Step *step : func->prologue_roots) {
        if (!step->is(StepKind::AccStore)) continue;
        const auto &store = step->step_data<StepKind::AccStore>();
        size_t idx = func->accs.index(store.acc);
        switch (step->dtype.kind) {
        case DataTypeKind::Scalar: {
            ScalarDataType dtype = step->dtype.as_scalar();
            Ty *type = dtype == ScalarDataType::I1 ? builder.b->getInt1Ty() : builder.scalar_type_to_llvm(dtype);
            V *acc_value = builder.b->CreateAlloca(type);
            builder.accs[idx] = {acc_value, type};
            break;
        }
        case DataTypeKind::Vec: {
            VecDataType dtype = step->dtype.as_vec();
            Ty *type = builder.vec_type_to_llvm(dtype);
            V *acc_value = builder.b->CreateAlloca(type);
            builder.accs[idx] = {acc_value, type};
            break;
        }
        case DataTypeKind::Mask: {
            MaskDataType dtype = step->dtype.as_mask();
            Ty *type = builder.mask_type_to_llvm(dtype);
            V *acc_value = builder.b->CreateAlloca(type);
            builder.accs[idx] = {acc_value, type};
            break;
        }
        }
    }

    builder.steps_to_llvm(func->prologue_roots);

    llvm::BasicBlock *end = llvm::BasicBlock::Create(*builder.context, "end", builder.function);
    llvm::BasicBlock *remainder_entry =
        !func->remainder_roots.empty() ? llvm::BasicBlock::Create(*builder.context, "remainder_loop", builder.function)
                                       : end;
    builder.counter_ptr = builder.b->CreateAlloca(rows_count_arg->getType());
    builder.b->CreateStore(builder.b->getInt64(0), builder.counter_ptr);
    if (!func->main_loop_roots.empty()) {
        llvm::BasicBlock *loop = llvm::BasicBlock::Create(*builder.context, "loop", builder.function);
        llvm::Value *last_simd_idx =
            builder.b->CreateAnd(rows_count_arg, builder.b->getInt64(-(int64_t)func->loop_width));
        builder.b->CreateCondBr(builder.b->CreateICmpULT(rows_count_arg, builder.b->getInt64(func->loop_width)),
                                remainder_entry, loop);

        builder.b->SetInsertPoint(loop);
        builder.steps_to_llvm(func->main_loop_roots);
        V *old_counter = builder.b->CreateLoad(builder.b->getInt64Ty(), builder.counter_ptr);
        V *incremeted_counter = builder.b->CreateNUWAdd(old_counter, builder.b->getInt64(func->loop_width));
        builder.b->CreateStore(incremeted_counter, builder.counter_ptr);
        builder.b->CreateCondBr(builder.b->CreateICmpULT(incremeted_counter, last_simd_idx), loop, remainder_entry);
    } else {
        builder.b->CreateBr(remainder_entry);
    }
    if (!func->remainder_roots.empty()) {
        builder.b->SetInsertPoint(remainder_entry);
        V *old_counter = builder.b->CreateLoad(builder.b->getInt64Ty(), builder.counter_ptr);
        llvm::BasicBlock *remainder_block =
            llvm::BasicBlock::Create(*builder.context, "remainder_body", builder.function);
        builder.b->CreateCondBr(builder.b->CreateICmpEQ(old_counter, rows_count_arg), end, remainder_block);
        builder.b->SetInsertPoint(remainder_block);
        builder.steps_to_llvm(func->remainder_roots);
        V *incremeted_counter = builder.b->CreateNUWAdd(old_counter, builder.b->getInt64(1));
        builder.b->CreateStore(incremeted_counter, builder.counter_ptr);
        builder.b->CreateBr(remainder_entry);
    }
    builder.b->SetInsertPoint(end);
    builder.steps_to_llvm(func->epilogue_roots);

    builder.b->CreateRetVoid();

    llvm::verifyFunction(*builder.function);
}

std::unique_ptr<llvm::MemoryBuffer> generateA64MachineCode(llvm::Module *module) {
    // Initialize native target (only needed once per process)
    llvm::InitializeAllTargets();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllTargetMCs();

    // Get the ARM64 target
    llvm::Triple triple("aarch64-none-unknown");
    std::string Error;
    const llvm::Target *target = llvm::TargetRegistry::lookupTarget(triple, Error);
    if (!target) {
        llvm::errs() << "Error getting target: " << Error << "\n";
        return {};
    }

    // Create target machine
    llvm::TargetOptions options;

    std::unique_ptr<llvm::TargetMachine> target_machine{
        //
        target->createTargetMachine( //
            triple,
            "generic", // CPU type
            "+neon",   // Features
            options, llvm::Reloc::Model::PIC_, {}, llvm::CodeGenOptLevel::Less)};
    // Configure module
    module->setDataLayout(target_machine->createDataLayout());

    // Create output buffer
    llvm::SmallVector<char, 0> obj_buffer_sv;
    llvm::raw_svector_ostream obj_stream(obj_buffer_sv);

    // Create a pass manager
    llvm::legacy::PassManager pass_manager;

    // Ask the target machine to add passes to emit an object file
    if (target_machine->addPassesToEmitFile(pass_manager, obj_stream, nullptr, llvm::CodeGenFileType::ObjectFile)) {
        llvm::errs() << "TargetMachine can't emit an object file\n";
        return {};
    }
    pass_manager.run(*module);

    // Create MemoryBuffer from the generated code
    return llvm::MemoryBuffer::getMemBufferCopy(llvm::StringRef(obj_buffer_sv.data(), obj_buffer_sv.size()));
}

static std::string get_ir_string(llvm::Module &module) {
    std::string ir;
    llvm::raw_string_ostream os(ir);
    module.print(os, nullptr);
    os.flush();
    return ir;
}

LLVMModuleOwner::LLVMModuleOwner() noexcept = default;
LLVMModuleOwner::LLVMModuleOwner(std::unique_ptr<llvm::LLVMContext> context_init,
                                 std::unique_ptr<llvm::Module> module_init) noexcept
    : context(std::move(context_init)), module(std::move(module_init)) {
}
LLVMModuleOwner::~LLVMModuleOwner() noexcept = default;
LLVMModuleOwner::LLVMModuleOwner(LLVMModuleOwner &&) noexcept = default;
LLVMModuleOwner &LLVMModuleOwner::operator=(LLVMModuleOwner &&other) noexcept {
    if (this == &other) { return *this; }
    // Destroy the old module before its context. Defaulted member-wise move
    // assignment would assign context first and violate that requirement.
    module.reset();
    context.reset();
    context = std::move(other.context);
    module = std::move(other.module);
    return *this;
}

LLVMModuleOwner build_llvm_module(const mir::Function *function) {
    if (function == nullptr) { throw std::invalid_argument("LLVM module construction requires non-null MIR"); }
    llvm_backend::LLVMBuilder builder{};
    builder.ctx = function->ctx;
    builder.context = std::make_unique<llvm::LLVMContext>();
    builder.module = std::make_unique<llvm::Module>("simjit", *builder.context);
    builder.b = std::make_unique<llvm::IRBuilder<>>(*builder.context);
    builder.value_map = function->ctx->arena->alloc_array<llvm_backend::V *>(function->step_id_count);

    llvm_backend::compile_function(builder, function);
    builder.b.reset();
    return {std::move(builder.context), std::move(builder.module)};
}

} // namespace llvm_backend

std::string emit_llvm_ir(const mir::Function *function) {
    llvm_backend::LLVMModuleOwner owner = llvm_backend::build_llvm_module(function);
    if (llvm::verifyModule(*owner.module, &llvm::errs())) { messed_up("Verification failed"); }
    return llvm_backend::get_ir_string(*owner.module);
}

namespace llvm_backend {
namespace {

static std::runtime_error llvm_error(std::string_view prefix, llvm::Error error) {
    return std::runtime_error(std::string(prefix) + ": " + llvm::toString(std::move(error)));
}

static llvm::OptimizationLevel pass_opt_level(LLVMOptLevel level) {
    return level == LLVMOptLevel::O1 ? llvm::OptimizationLevel::O1 : llvm::OptimizationLevel::O3;
}

static llvm::CodeGenOptLevel codegen_opt_level(LLVMOptLevel level) {
    return level == LLVMOptLevel::O1 ? llvm::CodeGenOptLevel::Less : llvm::CodeGenOptLevel::Aggressive;
}

static void run_optimization_pipeline(llvm::Module &module, llvm::TargetMachine &target_machine, LLVMOptLevel level) {
    llvm::LoopAnalysisManager loops;
    llvm::FunctionAnalysisManager functions;
    llvm::CGSCCAnalysisManager cgscc;
    llvm::ModuleAnalysisManager modules;
    llvm::PipelineTuningOptions tuning;
    if (level == LLVMOptLevel::O1) {
        tuning.LoopInterleaving = false;
        tuning.LoopVectorization = false;
        tuning.SLPVectorization = false;
        tuning.LoopUnrolling = false;
    }
    llvm::PassBuilder pass_builder(&target_machine, tuning);
    pass_builder.registerModuleAnalyses(modules);
    pass_builder.registerCGSCCAnalyses(cgscc);
    pass_builder.registerFunctionAnalyses(functions);
    pass_builder.registerLoopAnalyses(loops);
    pass_builder.crossRegisterProxies(loops, functions, cgscc, modules);
    llvm::ModulePassManager pipeline = pass_builder.buildPerModuleDefaultPipeline(pass_opt_level(level));
    pipeline.run(module, modules);
}

static void verify_module(const llvm::Module &module) {
    std::string verify_text;
    llvm::raw_string_ostream verify_stream(verify_text);
    if (llvm::verifyModule(module, &verify_stream)) {
        throw std::runtime_error("LLVM module verification failed: " + verify_text);
    }
}

static void initialize_llvm_jit_once() {
    static const bool initialized = [] {
        if (llvm::InitializeNativeTarget()) { throw std::runtime_error("LLVM native target initialization failed"); }
        if (llvm::InitializeNativeTargetAsmPrinter()) {
            throw std::runtime_error("LLVM native assembly-printer initialization failed");
        }
        if (llvm::InitializeNativeTargetAsmParser()) {
            throw std::runtime_error("LLVM native assembly-parser initialization failed");
        }
        return true;
    }();
    (void)initialized;
}

static mir::Function *lower_hir(const hir::Function *function, jit::CompilePolicy policy) {
    if (function == nullptr) { throw std::invalid_argument("LLVM HIR compilation requires non-null HIR"); }
    switch (policy) {
    case jit::CompilePolicy::Scalar: return lower_scalar(function);
    case jit::CompilePolicy::Vectorized: return lower_vectorized(function);
    case jit::CompilePolicy::BestEffort: {
        auto vectorized = try_lower_vectorized(function);
        return vectorized ? vectorized.value() : lower_scalar(function);
    }
    }
    SIMJIT_UNREACHABLE();
}

} // namespace

struct LLVMSession::Impl {
    Arch arch = Arch::Native;
    LLVMOptLevel opt_level = LLVMOptLevel::O1;
    std::unique_ptr<llvm::TargetMachine> target_machine;
    std::unique_ptr<llvm::orc::LLJIT> jit;
};

LLVMSession::LLVMSession(Arch arch, LLVMOptLevel opt_level) : impl_(std::make_unique<Impl>()) {
    if (arch != Arch::Native) { throw std::invalid_argument("LLVM JIT requires the native architecture"); }
    initialize_llvm_jit_once();
    auto target = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!target) { throw llvm_error("LLVM host target detection failed", target.takeError()); }
    target->setCodeGenOptLevel(codegen_opt_level(opt_level));
    auto target_machine = target->createTargetMachine();
    if (!target_machine) { throw llvm_error("LLVM target-machine creation failed", target_machine.takeError()); }
    llvm::orc::LLJITBuilder builder;
    builder.setJITTargetMachineBuilder(std::move(*target));
    builder.setNumCompileThreads(0);
    auto jit = builder.create();
    if (!jit) { throw llvm_error("LLVM LLJIT creation failed", jit.takeError()); }
    impl_->arch = arch;
    impl_->opt_level = opt_level;
    impl_->target_machine = std::move(*target_machine);
    impl_->jit = std::move(*jit);
}

LLVMSession::~LLVMSession() noexcept = default;
Arch LLVMSession::arch() const noexcept {
    return impl_->arch;
}
LLVMOptLevel LLVMSession::opt_level() const noexcept {
    return impl_->opt_level;
}

void LLVMSession::optimize_module(LLVMModuleOwner &owner) {
    if (!owner.context || !owner.module) { throw std::invalid_argument("cannot optimize an empty LLVM module"); }
    owner.module->setDataLayout(impl_->jit->getDataLayout());
    owner.module->setTargetTriple(impl_->target_machine->getTargetTriple());
    verify_module(*owner.module);
    run_optimization_pipeline(*owner.module, *impl_->target_machine, impl_->opt_level);
}

void LLVMSession::add_module(LLVMModuleOwner owner) {
    if (!owner.context || !owner.module) { throw std::invalid_argument("cannot add an empty LLVM module"); }
    owner.module->setDataLayout(impl_->jit->getDataLayout());
    owner.module->setTargetTriple(impl_->target_machine->getTargetTriple());
    verify_module(*owner.module);
    llvm::orc::ThreadSafeModule thread_safe_module(std::move(owner.module), std::move(owner.context));
    if (llvm::Error error = impl_->jit->addIRModule(std::move(thread_safe_module))) {
        throw llvm_error("LLVM addIRModule failed", std::move(error));
    }
}

void *LLVMSession::lookup(std::string_view symbol) {
    auto address = impl_->jit->lookup(llvm::StringRef(symbol.data(), symbol.size()));
    if (!address) { throw llvm_error("LLVM symbol lookup failed", address.takeError()); }
    void *function = address->toPtr<void *>();
    if (function == nullptr) { throw std::runtime_error("LLVM returned a null executable function pointer"); }
    return function;
}

LLVMModuleOwner parse_llvm_ir(std::string_view ir) {
    auto context = std::make_unique<llvm::LLVMContext>();
    llvm::SMDiagnostic diagnostic;
    auto module = llvm::parseAssemblyString(llvm::StringRef(ir.data(), ir.size()), diagnostic, *context);
    if (!module) {
        std::string text;
        llvm::raw_string_ostream stream(text);
        diagnostic.print("simjit", stream);
        throw std::runtime_error(text);
    }
    return {std::move(context), std::move(module)};
}

void *compile_mir(const mir::Function *function, LLVMSession &session) {
    if (function == nullptr) { throw std::invalid_argument("LLVM MIR compilation requires non-null MIR"); }
    std::string symbol = function->ctx->symbol_name;
    LLVMModuleOwner owner = build_llvm_module(function);
    session.optimize_module(owner);
    session.add_module(std::move(owner));
    return session.lookup(symbol);
}

void *compile_hir(const hir::Function *function, LLVMSession &session, jit::CompilePolicy policy) {
    return compile_mir(lower_hir(function, policy), session);
}

void *compile_ir(std::string_view ir, std::string_view symbol, LLVMSession &session) {
    LLVMModuleOwner owner = parse_llvm_ir(ir);
    session.optimize_module(owner);
    session.add_module(std::move(owner));
    return session.lookup(symbol);
}

} // namespace llvm_backend
} // namespace simjit
