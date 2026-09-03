// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#pragma once

#include "simjit/core/x86.h"

#include <algorithm>
#include <variant>

namespace simjit {
namespace mir {
enum class FmaKind : uint8_t;
}
namespace x86 {

struct CurrentTypeTag {};
struct CurrentScalarTypeTag {};
struct CurrentMaskTypeTag {};
constexpr CurrentTypeTag CurrentType{};
constexpr CurrentScalarTypeTag CurrentScalarType{};
constexpr CurrentMaskTypeTag CurrentMaskType{};

template <VecRegisterKind R, VecElemType E> struct VecIntrinsicDataTypeDescription {
    VecIntrinsicDataTypeDescription() = delete;

    constexpr VecIntrinsicDataTypeDescription(CurrentTypeTag) noexcept : dtype(CurrentType) {}
    constexpr VecIntrinsicDataTypeDescription(CurrentScalarTypeTag) noexcept : dtype(CurrentScalarType) {}
    constexpr VecIntrinsicDataTypeDescription(CurrentMaskTypeTag) noexcept : dtype(CurrentMaskType) {}
    constexpr VecIntrinsicDataTypeDescription(DataType d) noexcept : dtype(d) {}
    constexpr VecIntrinsicDataTypeDescription(VecDataType d) noexcept : dtype(d) {}
    constexpr VecIntrinsicDataTypeDescription(MaskDataType d) noexcept : dtype(d) {}
    constexpr VecIntrinsicDataTypeDescription(ScalarDataType d) noexcept : dtype(d) {}

    constexpr static inline Vector x86_current = Vector{R, E};
    constexpr static inline VecDataType current = x86_to_vec(x86_current);

    constexpr DataType get() const {
        if (std::holds_alternative<CurrentTypeTag>(dtype)) { return current; }
        if (std::holds_alternative<CurrentScalarTypeTag>(dtype)) { return current.to_scalar(); }
        if (std::holds_alternative<CurrentMaskTypeTag>(dtype)) { return current.mask(); }
        return std::get<DataType>(dtype);
    }

    std::variant<CurrentTypeTag, CurrentScalarTypeTag, CurrentMaskTypeTag, DataType> dtype;
};

template <MaskDataType M> struct MaskIntrinsicDataTypeDescription {
    MaskIntrinsicDataTypeDescription() = delete;

    constexpr MaskIntrinsicDataTypeDescription(CurrentTypeTag) noexcept : dtype(CurrentType) {}
    constexpr MaskIntrinsicDataTypeDescription(DataType d) noexcept : dtype(d) {}
    constexpr MaskIntrinsicDataTypeDescription(VecDataType d) noexcept : dtype(d) {}
    constexpr MaskIntrinsicDataTypeDescription(MaskDataType d) noexcept : dtype(d) {}
    constexpr MaskIntrinsicDataTypeDescription(ScalarDataType d) noexcept : dtype(d) {}

    constexpr static MaskDataType current = M;

    constexpr DataType get() const {
        if (std::holds_alternative<CurrentTypeTag>(dtype)) { return current; }
        return std::get<DataType>(dtype);
    }

    std::variant<CurrentTypeTag, DataType> dtype;
};

template <ScalarDataType S> struct ScalarIntrinsicDataTypeDescription {
    ScalarIntrinsicDataTypeDescription() = delete;

    constexpr ScalarIntrinsicDataTypeDescription(CurrentTypeTag) noexcept : dtype(CurrentType) {}
    constexpr ScalarIntrinsicDataTypeDescription(DataType d) noexcept : dtype(d) {}
    constexpr ScalarIntrinsicDataTypeDescription(VecDataType d) noexcept : dtype(d) {}
    constexpr ScalarIntrinsicDataTypeDescription(MaskDataType d) noexcept : dtype(d) {}
    constexpr ScalarIntrinsicDataTypeDescription(ScalarDataType d) noexcept : dtype(d) {}

    constexpr static ScalarDataType current = S;

    constexpr DataType get() const {
        if (std::holds_alternative<CurrentTypeTag>(dtype)) { return current; }
        return std::get<DataType>(dtype);
    }

    std::variant<CurrentTypeTag, DataType> dtype;
};

class Intrinsic {
public:
    Intrinsic() = delete;
    Intrinsic(const Intrinsic &) = delete;
    Intrinsic(Intrinsic &&) = default;
    Intrinsic &operator=(const Intrinsic &) = delete;
    Intrinsic &operator=(Intrinsic &&) = default;

    Intrinsic(DataType return_dtype, const char *name, std::initializer_list<DataType> args)
        : return_dtype_(return_dtype), name_(name), arity_(args.size()), args_(args) {
        SIMJIT_ASSERT(args.size() <= 5);
    }

    const char *name() const noexcept { return name_; }
    DataType return_dtype() const noexcept { return return_dtype_; }
    size_t arity() const noexcept { return arity_; }
    DataType arg_dtype(size_t nth) const { return args_.at(nth); }

private:
    DataType return_dtype_;
    const char *name_;
    size_t arity_;
    std::vector<DataType> args_;
};

template <typename DT, typename D> struct IntrinsicDescription {
    using I = Intrinsic;

    I i;

    constexpr IntrinsicDescription(D ret, const char *name) : i(I(ret.get(), name, {})) {}
    constexpr IntrinsicDescription(D ret, const char *name, D a1) : i(I(ret.get(), name, {a1.get()})) {}
    constexpr IntrinsicDescription(D ret, const char *name, D a1, D a2) : i(I(ret.get(), name, {a1.get(), a2.get()})) {}
    constexpr IntrinsicDescription(D ret, const char *name, D a1, D a2, D a3)
        : i(I(ret.get(), name, {a1.get(), a2.get(), a3.get()})) {}
    constexpr IntrinsicDescription(D ret, const char *name, D a1, D a2, D a3, D a4)
        : i(I(ret.get(), name, {a1.get(), a2.get(), a3.get(), a4.get()})) {}
    constexpr IntrinsicDescription(D ret, const char *name, D a1, D a2, D a3, D a4, D a5)
        : i(I(ret.get(), name, {a1.get(), a2.get(), a3.get(), a4.get(), a5.get()})) {}

    constexpr I &&get() noexcept { return std::move(i); }

    constexpr static inline DT dtype = D::current;
};

template <VecRegisterKind R, VecElemType E>
using VecIntrinsicDescription = IntrinsicDescription<VecDataType, VecIntrinsicDataTypeDescription<R, E>>;

template <MaskDataType M>
using MaskIntrinsicDescription = IntrinsicDescription<MaskDataType, MaskIntrinsicDataTypeDescription<M>>;

template <ScalarDataType S>
using ScalarIntrinsicDescription = IntrinsicDescription<ScalarDataType, ScalarIntrinsicDataTypeDescription<S>>;

template <typename DT> class UnaryIntrinsicMap final {
    using I = Intrinsic;

public:
    UnaryIntrinsicMap() = delete;
    UnaryIntrinsicMap(const UnaryIntrinsicMap &) = delete;
    UnaryIntrinsicMap(UnaryIntrinsicMap &&) = delete;
    UnaryIntrinsicMap &operator=(const UnaryIntrinsicMap &) = delete;
    UnaryIntrinsicMap &operator=(UnaryIntrinsicMap &&) = delete;
    ~UnaryIntrinsicMap() noexcept = default;

    template <typename... Args> constexpr explicit UnaryIntrinsicMap(const char *name, Args &&...args) : name_(name) {
        (init_member(std::move(args)), ...);
        for (const Tagged &t : tagged_) {
            all_vec_.push_back(&t.i);
        }
    }

    const char *name() const noexcept { return name_; }
    const Intrinsic *lookup_nothrow(DT s) const noexcept {
        if (auto res = std::find_if(tagged_.cbegin(), tagged_.cend(), [=](const auto &test) { return test.x == s; });
            res != tagged_.end()) {
            return &res->i;
        }
        return nullptr;
    }
    const Intrinsic &lookup(DT s) const {
        if (auto *intrin = lookup_nothrow(s)) { return *intrin; }
        if constexpr (std::is_same_v<DT, VecDataType>) {
            x86_messed_up("Do not support %s for %s dtype", name_, show_vec_dtype(s));
        } else if constexpr (std::is_same_v<DT, ScalarDataType>) {
            x86_messed_up("Do not support %s for %s dtype", name_, show_scalar_dtype(s));
        } else {
            x86_messed_up("Do not support %s for %s dtype", name_, show_mask_dtype(s));
        }
    }

private:
    template <typename T> constexpr void init_member(T d) {
        SIMJIT_ASSERT(lookup_nothrow(d.dtype) == nullptr);
        tagged_.push_back(Tagged{d.get(), d.dtype});
    }

    struct Tagged {
        I i;
        DT x;
    };
    const char *name_;
    std::vector<Tagged> tagged_;
    std::vector<const I *> all_vec_;
};
using MaskIntrinsicUnaryMap = UnaryIntrinsicMap<MaskDataType>;
using ScalarIntrinsicUnaryMap = UnaryIntrinsicMap<ScalarDataType>;
using VecIntrinsicUnaryMap = UnaryIntrinsicMap<VecDataType>;

template <typename D> struct BinaryContainer {
    VecDataType vdtype;
    D d;
};

template <typename D> constexpr BinaryContainer<D> binary_intrin(VecDataType vdtype, D d) noexcept {
    return BinaryContainer<D>{vdtype, std::move(d)};
}

class VecBinaryIntrinsicMap final {
    template <VecRegisterKind R, VecElemType E> using D = VecIntrinsicDescription<R, E>;

public:
    VecBinaryIntrinsicMap() = delete;
    VecBinaryIntrinsicMap(const VecBinaryIntrinsicMap &) = delete;
    VecBinaryIntrinsicMap(VecBinaryIntrinsicMap &&) = default;
    VecBinaryIntrinsicMap &operator=(const VecBinaryIntrinsicMap &) = delete;
    VecBinaryIntrinsicMap &operator=(VecBinaryIntrinsicMap &&) = default;
    ~VecBinaryIntrinsicMap() noexcept = default;

    template <typename... Args>
    constexpr explicit VecBinaryIntrinsicMap(const char *name, Args &&...args) : name_(name) {
        (init_member(std::move(args)), ...);
        for (const Tagged &t : tagged_) {
            all_vec_.push_back(&t.i);
        }
    }

    constexpr const char *name() const noexcept { return name_; }

    const Intrinsic *lookup_nothrow(VecDataType x, VecDataType y) const noexcept {
        if (auto res = std::find_if(tagged_.cbegin(), tagged_.cend(),
                                    [=](const auto &test) { return test.x == x && test.y == y; });
            res != tagged_.end()) {
            return &res->i;
        }
        return nullptr;
    }

    const Intrinsic &lookup(VecDataType x, VecDataType y) const {
        if (auto *intrin = lookup_nothrow(x, y)) { return *intrin; }
        x86_messed_up("do not support %s for %s, %s dtypes", name(), show_vec_dtype(x), show_vec_dtype(y));
    }

private:
    template <VecRegisterKind R, VecElemType E> constexpr void init_member(BinaryContainer<D<R, E>> d) {
        tagged_.push_back(Tagged{d.d.get(), D<R, E>::dtype, d.vdtype});
    }

    struct Tagged {
        Intrinsic i;
        VecDataType x;
        VecDataType y;
    };
    const char *name_;
    std::vector<Tagged> tagged_;
    std::vector<const Intrinsic *> all_vec_;
};

#define DESC_FOR_INTRIN_REG(_iname, _reg)                                                       \
    using _iname##_reg##I8 = VecIntrinsicDescription<VecRegisterKind::_reg, VecElemType::I8>;   \
    using _iname##_reg##I16 = VecIntrinsicDescription<VecRegisterKind::_reg, VecElemType::I16>; \
    using _iname##_reg##I32 = VecIntrinsicDescription<VecRegisterKind::_reg, VecElemType::I32>; \
    using _iname##_reg##I64 = VecIntrinsicDescription<VecRegisterKind::_reg, VecElemType::I64>; \
    using _iname##_reg##F32 = VecIntrinsicDescription<VecRegisterKind::_reg, VecElemType::F32>; \
    using _iname##_reg##F64 = VecIntrinsicDescription<VecRegisterKind::_reg, VecElemType::F64>;

#define DESC_FOR_INTRIN(_iname)      \
    DESC_FOR_INTRIN_REG(_iname, XMM) \
    DESC_FOR_INTRIN_REG(_iname, YMM) \
    DESC_FOR_INTRIN_REG(_iname, ZMM)

DESC_FOR_INTRIN(I)

using IM8 = MaskIntrinsicDescription<MaskDataType::M8>;
using IM16 = MaskIntrinsicDescription<MaskDataType::M16>;
using IM32 = MaskIntrinsicDescription<MaskDataType::M32>;
using IM64 = MaskIntrinsicDescription<MaskDataType::M64>;

using ISI8 = ScalarIntrinsicDescription<ScalarDataType::I8>;
using ISI16 = ScalarIntrinsicDescription<ScalarDataType::I16>;
using ISI32 = ScalarIntrinsicDescription<ScalarDataType::I32>;
using ISI64 = ScalarIntrinsicDescription<ScalarDataType::I64>;

#include "x86_intrin.generated.h"

extern const ScalarIntrinsicUnaryMap scalar_popcnt_map;
extern const ScalarIntrinsicUnaryMap scalar_lzcnt_map;
extern const ScalarIntrinsicUnaryMap pext_map;
extern const ScalarIntrinsicUnaryMap pdep_map;
extern const ScalarIntrinsicUnaryMap andn_map;
extern const ScalarIntrinsicUnaryMap blsmsk_map;
extern const ScalarIntrinsicUnaryMap tzcnt_map;

const UnaryIntrinsicMap<VecDataType> &arith_binary_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &vector_immediate_shift_rotate_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &float_binary_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &maskz_vector_immediate_shift_rotate_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &mask_vector_immediate_shift_rotate_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &maskz_arith_binary_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &maskz_float_binary_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &mask_arith_binary_map(ArithBinaryOp op);
const UnaryIntrinsicMap<VecDataType> &mask_float_binary_map(ArithBinaryOp op);
extern const VecIntrinsicUnaryMap set1_map;
extern const VecBinaryIntrinsicMap cvt_map;
extern const VecBinaryIntrinsicMap maskz_cvt_map;
extern const VecBinaryIntrinsicMap mask_cvt_map;
extern const VecBinaryIntrinsicMap float_cast_map;
extern const VecBinaryIntrinsicMap float_ucast_map;
extern const VecBinaryIntrinsicMap zext_map;
extern const VecBinaryIntrinsicMap maskz_zext_map;
extern const VecBinaryIntrinsicMap mask_zext_map;
extern const VecBinaryIntrinsicMap cvt_storeu_map;
extern const VecBinaryIntrinsicMap signed_saturate_map;
extern const VecBinaryIntrinsicMap unsigned_saturate_map;
extern const VecIntrinsicUnaryMap bitcast_map;
const VecIntrinsicUnaryMap &fma_map(mir::FmaKind kind);
extern const VecIntrinsicUnaryMap mask_i32gather_map;
extern const VecIntrinsicUnaryMap i32scatter_map;
extern const VecIntrinsicUnaryMap i64scatter_map;
extern const VecIntrinsicUnaryMap mask_i32scatter_map;
extern const VecIntrinsicUnaryMap mask_i64scatter_map;
extern const VecIntrinsicUnaryMap mask_i64gather_map;
extern const VecIntrinsicUnaryMap compresstore_map;
extern const VecBinaryIntrinsicMap extract_map;

extern const VecBinaryIntrinsicMap compiler_downcast_map;
const VecIntrinsicUnaryMap &arith_unary_map(ArithUnaryOp op);
const VecIntrinsicUnaryMap &maskz_arith_unary_map(ArithUnaryOp op);
const VecIntrinsicUnaryMap &mask_arith_unary_map(ArithUnaryOp op);
const VecIntrinsicUnaryMap &reduce_map(ArithBinaryOp op);
const VecIntrinsicUnaryMap &unmasked_reduce_map(ArithBinaryOp op);

extern const VecIntrinsicUnaryMap mov_mask_map;
extern const VecIntrinsicUnaryMap movm_map;
extern const VecIntrinsicUnaryMap undefined_map;

const MaskIntrinsicUnaryMap &binary_op_mask_map(PredicateBinaryOp op);
extern const MaskIntrinsicUnaryMap not_mask_map;
extern const MaskIntrinsicUnaryMap cvtmask_u_map;
extern const MaskIntrinsicUnaryMap cvtu_mask_map;
extern const MaskIntrinsicUnaryMap kunpack_map;
extern const MaskIntrinsicUnaryMap ktestc_map;
extern const MaskIntrinsicUnaryMap ktestz_map;
extern const MaskIntrinsicUnaryMap kortestz_map;
extern const MaskIntrinsicUnaryMap kortestc_map;
extern const MaskIntrinsicUnaryMap load_mask_map;
extern const MaskIntrinsicUnaryMap store_mask_map;

extern const VecIntrinsicUnaryMap gp2affine_map;
extern const VecIntrinsicUnaryMap permb_map;
extern const VecIntrinsicUnaryMap shuffle8_map;

extern const VecIntrinsicUnaryMap roundscale_map;
extern const VecIntrinsicUnaryMap maskz_roundscale_map;
extern const VecIntrinsicUnaryMap mask_roundscale_map;
extern const VecIntrinsicUnaryMap fpclass_map;
extern const VecIntrinsicUnaryMap mask_storea_map;

} // namespace x86
} // namespace simjit
