// This file is part of Simjit project <https://simjit.org>
//
// See LICENSE for license and copyright information
// SPDX-License-Identifier: Zlib

#include "test.h"

using namespace simjit;
using namespace simjit::types;

std::vector<Test> tpcds_tests{
    // condition from Q10
    {[](FunctionBuilder &b) {
        Value c_customer_sk = b.input_arg(I32);
        Value ss_customer_sk = b.input_arg(I32);
        Value ss_sold_date_sk = b.input_arg(I32);
        Value d_date_sk = b.input_arg(I32);
        Value d_year = b.input_arg(I32);
        Value d_moy = b.input_arg(I32);

        Predicate cond = b.cmp_eq(c_customer_sk, ss_customer_sk);
        cond = b.and_(b.cmp_eq(ss_sold_date_sk, d_date_sk), cond);
        cond = b.and_(b.cmp_eq(d_year, b.i32(2002)), cond);
        cond = b.and_(b.and_(b.cmp_ge(d_moy, b.i32(4)), b.cmp_le(d_moy, b.i32(4 + 3))), cond);
        b.pack(b.index(I32), cond, b.arg(I32), b.arg(I64));
    }},
    {[](FunctionBuilder &b) {
         Value c_customer_sk = b.input_arg(I32);
         Value ss_customer_sk = b.input_arg(I32);
         Value ss_sold_date_sk = b.input_arg(I32);
         Value d_date_sk = b.input_arg(I32);
         Value d_year = b.input_arg(I16);
         Value d_moy = b.input_arg(I8);

         Predicate cond = b.cmp_eq(c_customer_sk, ss_customer_sk);
         cond = b.and_(b.cmp_eq(ss_sold_date_sk, d_date_sk), cond);
         cond = b.and_(b.cmp_eq(d_year, b.i16(2002)), cond);
         cond = b.and_(b.and_(b.cmp_ge(d_moy, b.i8(4)), b.cmp_le(d_moy, b.i8(4 + 3))), cond);
         b.pack(b.index(I32), cond, b.arg(I32), b.arg(I64));
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    // sum from Q11
    {[](FunctionBuilder &b) {
        Value ws_ext_list_price = b.input_arg(I32);
        Value ws_ext_discount_amt = b.input_arg(I32);
        b.sum(b.sub(ws_ext_list_price, ws_ext_discount_amt), b.arg(I32));
    }},
    // condition from Q11
    {[](FunctionBuilder &b) {
         Value sale_type = b.input_arg(I8);
         Value s_firstyear_year_total = b.input_arg(I64);
         Value w_firstyear_year_total = b.input_arg(I64);
         Value s_secyear_year_total = b.input_arg(I64);
         Value w_secyear_year_total = b.input_arg(I64);

         Predicate cond = b.cmp_eq(sale_type, b.input_splat_arg(I8));
         cond = b.and_(b.cmp_gt(s_firstyear_year_total, b.i64(0)), cond);
         cond = b.and_(b.cmp_gt(w_firstyear_year_total, b.i64(0)), cond);
         cond = b.and_(
             b.cmp_gt(
                 b.select(b.cmp_gt(w_firstyear_year_total, b.i64(0)),
                          b.div(b.float_cast(w_secyear_year_total, F64), b.float_cast(w_firstyear_year_total, F64)),
                          b.f64(0.0)),
                 b.select(b.cmp_gt(s_firstyear_year_total, b.i64(0)),
                          b.div(b.float_cast(s_secyear_year_total, F64), b.float_cast(s_firstyear_year_total, F64)),
                          b.f64(0.0))),
             cond);
         b.pack(b.index(I32), cond, b.arg(I32), b.arg(I64));
     },
     coefficient_range_limit(TestVariant::VectorAll)},
    // sum from Q14
    {[](FunctionBuilder &b) {
        Value cs_quantity = b.input_arg(I32);
        Value cs_list_price = b.input_arg(I32);
        b.sum(b.mul(b.sext(cs_quantity, I64), b.sext(cs_list_price, I64)), b.arg(I64));
    }},
    {[](FunctionBuilder &b) {
         b.arg_safety_check();
         Value cs_quantity = b.input_arg(I32);
         Value cs_list_price = b.input_arg(I32);
         b.sum(b.mul_checked(cs_quantity, cs_list_price), b.arg(I32));
     },
     test_meta().unstable(TestVariant::All)},
    {[](FunctionBuilder &b) {
         Value cs_quantity = b.input_arg(I32);
         Value cs_list_price = b.input_arg(I32);
         Value cs_item_sk = b.input_arg(I32);
         Value i_item_sk = b.input_arg(I32);
         Value d_year = b.input_arg(I16);
         Value d_moy = b.input_arg(I8);
         Predicate cond = b.cmp_eq(cs_item_sk, i_item_sk);
         cond = b.and_(cond, b.cmp_eq(d_year, b.i16(1999 + 2)));
         cond = b.and_(cond, b.cmp_eq(d_moy, b.i8(11)));
         Value m = b.mul(b.sext(cs_quantity, I64), b.sext(cs_list_price, I64));
         b.sum_if(m, cond, b.arg(I64));
     },
     coefficient_range_limit(TestVariant::VectorAll)},
    // Q15 with integer strings
    {[](FunctionBuilder &b) {
         Value ca_zip = b.input_arg(I64);
         ca_zip = b.srl(ca_zip, b.i64(3));

         auto make_int = [](std::string_view str) -> int64_t {
             uint64_t result = 0;
             for (auto it = str.rbegin(); it < str.rend(); ++it) {
                 result = (result << 8) | ((uint64_t)*it);
             }
             return (int64_t)result;
         };

         Predicate zip_ok =
             b.or_(b.or_(b.cmp_eq(ca_zip, b.i64(make_int("85669"))), b.cmp_eq(ca_zip, b.i64(make_int("86197")))),
                   b.or_(b.cmp_eq(ca_zip, b.i64(make_int("88274"))), b.cmp_eq(ca_zip, b.i64(make_int("83405")))));
         zip_ok = b.or_(
             zip_ok, //
             b.or_(b.or_(b.cmp_eq(ca_zip, b.i64(make_int("86475"))), b.cmp_eq(ca_zip, b.i64(make_int("85392")))),
                   b.or_(b.cmp_eq(ca_zip, b.i64(make_int("85460"))), b.cmp_eq(ca_zip, b.i64(make_int("80348"))))));
         zip_ok = b.or_(zip_ok, b.cmp_eq(ca_zip, b.i64(make_int("81792"))));

         Value ca_state = b.input_arg(I32);
         Predicate state_ok = b.or_(
             b.or_(b.cmp_eq(ca_state, b.i32((int)make_int("CA"))), b.cmp_eq(ca_state, b.i32((int)make_int("WA")))),
             b.cmp_eq(ca_state, b.i32((int)make_int("GA"))));

         Value cs_sales_price = b.input_arg(I32);
         Predicate price_ok = b.cmp_gt(cs_sales_price, b.i32(500));

         Predicate cond = b.and_(zip_ok, b.and_(state_ok, price_ok));
         b.pack(b.index(I32), cond, b.arg(I32), b.arg(I64));
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    // Q18 sum
    {[](FunctionBuilder &b) {
        Value cs_quantity = b.input_arg(I32);
        Value cs_list_price = b.input_arg(I32);
        Value cs_coupon_amt = b.input_arg(I32);
        Value cs_sales_price = b.input_arg(I32);
        Value cs_net_profit = b.input_arg(I32);
        Value c_birth_year = b.input_arg(I32);
        Value cd_dep_count = b.input_arg(I32);
        b.sum(cs_quantity, b.arg(I32));
        b.sum(cs_list_price, b.arg(I32));
        b.sum(cs_coupon_amt, b.arg(I32));
        b.sum(cs_sales_price, b.arg(I32));
        b.sum(cs_net_profit, b.arg(I32));
        b.sum(c_birth_year, b.arg(I32));
        b.sum(cd_dep_count, b.arg(I32));
    }},
    {[](FunctionBuilder &b) {
        Value cs_quantity = b.input_arg(I32);
        Value cs_list_price = b.input_arg(I32);
        Value cs_coupon_amt = b.input_arg(I32);
        Value cs_sales_price = b.input_arg(I32);
        Value cs_net_profit = b.input_arg(I32);
        Value c_birth_year = b.input_arg(I32);
        Value cd_dep_count = b.input_arg(I32);
        b.sum(b.sext(cs_quantity, I64), b.arg(I64));
        b.sum(b.sext(cs_list_price, I64), b.arg(I64));
        b.sum(b.sext(cs_coupon_amt, I64), b.arg(I64));
        b.sum(b.sext(cs_sales_price, I64), b.arg(I64));
        b.sum(b.sext(cs_net_profit, I64), b.arg(I64));
        b.sum(b.sext(c_birth_year, I64), b.arg(I64));
        b.sum(b.sext(cd_dep_count, I64), b.arg(I64));
    }},
    {[](FunctionBuilder &b) {
        Value cs_quantity = b.input_arg(I32);
        Value cs_list_price = b.input_arg(I32);
        Value cs_coupon_amt = b.input_arg(I32);
        Value cs_sales_price = b.input_arg(I32);
        Value cs_net_profit = b.input_arg(I32);
        Value c_birth_year = b.input_arg(I32);
        Value cd_dep_count = b.input_arg(I32);
        b.sum(b.mul(b.sext(cs_quantity, I64), b.i64(10)), b.arg(I64));
        b.sum(b.mul(b.sext(cs_list_price, I64), b.i64(100)), b.arg(I64));
        b.sum(b.mul(b.sext(cs_coupon_amt, I64), b.i64(1000)), b.arg(I64));
        b.sum(b.mul(b.sext(cs_sales_price, I64), b.i64(10000)), b.arg(I64));
        b.sum(b.mul(b.sext(cs_net_profit, I64), b.i64(100000)), b.arg(I64));
        b.sum(b.mul(b.sext(c_birth_year, I64), b.i64(1000000)), b.arg(I64));
        b.sum(b.mul(b.sext(cd_dep_count, I64), b.i64(10000000)), b.arg(I64));
    }},
    // Q66 sum
    {[](FunctionBuilder &b) {
         Value d_moy = b.input_arg(I8);
         Value ws_net_paid_inc_ship = b.input_arg(I32);
         Value ws_quantity = b.input_arg(I32);
         auto m = b.mul(b.sext(ws_net_paid_inc_ship, I64), b.sext(ws_quantity, I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(1)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(2)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(3)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(4)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(5)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(6)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(7)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(8)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(9)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(10)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(11)), b.arg(I64));
         b.sum_if(m, b.cmp_eq(d_moy, b.i8(12)), b.arg(I64));
     },
     LIMIT_ALL_VECTOR.vectorization_failure(TestVariant::VectorAll, simjit::ErrorSubKind::TooManyRoots)},
    // Q62 sum
    {[](FunctionBuilder &b) {
         Value ws_ship_date_sk = b.input_arg(I32);
         Value ws_sold_date_sk = b.input_arg(I32);
         Value d = b.sub(ws_ship_date_sk, ws_sold_date_sk);

         b.sum_if(b.i64(1), b.cmp_le(d, b.i32(30)), b.arg(I64));
         b.sum_if(b.i64(1), b.and_(b.cmp_gt(d, b.i32(30)), b.cmp_le(d, b.i32(60))), b.arg(I64));
         b.sum_if(b.i64(1), b.and_(b.cmp_gt(d, b.i32(60)), b.cmp_le(d, b.i32(90))), b.arg(I64));
         b.sum_if(b.i64(1), b.and_(b.cmp_gt(d, b.i32(90)), b.cmp_le(d, b.i32(120))), b.arg(I64));
         b.sum_if(b.i64(1), b.cmp_gt(d, b.i32(120)), b.arg(I64));
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    {[](FunctionBuilder &b) {
         Value ws_ship_date_sk = b.input_arg(I32);
         Value ws_sold_date_sk = b.input_arg(I32);
         Value d = b.sub(ws_ship_date_sk, ws_sold_date_sk);

         b.sum(b.zero_select(b.i64(1), b.cmp_le(d, b.i32(30))), b.arg(I64));
         b.sum(b.zero_select(b.i64(1), b.and_(b.cmp_gt(d, b.i32(30)), b.cmp_le(d, b.i32(60)))), b.arg(I64));
         b.sum(b.zero_select(b.i64(1), b.and_(b.cmp_gt(d, b.i32(60)), b.cmp_le(d, b.i32(90)))), b.arg(I64));
         b.sum(b.zero_select(b.i64(1), b.and_(b.cmp_gt(d, b.i32(90)), b.cmp_le(d, b.i32(120)))), b.arg(I64));
         b.sum(b.zero_select(b.i64(1), b.cmp_gt(d, b.i32(120))), b.arg(I64));
     },
     coefficient_range_limit(TestVariant::ArmVector)},
    {[](FunctionBuilder &b) {
        Value ws_ship_date_sk = b.input_arg(I32);
        Value ws_sold_date_sk = b.input_arg(I32);
        Value d = b.sub(ws_ship_date_sk, ws_sold_date_sk);

        b.countif(b.cmp_le(d, b.i32(30)), b.arg(I64));
        b.countif(b.and_(b.cmp_gt(d, b.i32(30)), b.cmp_le(d, b.i32(60))), b.arg(I64));
        b.countif(b.and_(b.cmp_gt(d, b.i32(60)), b.cmp_le(d, b.i32(90))), b.arg(I64));
        b.countif(b.and_(b.cmp_gt(d, b.i32(90)), b.cmp_le(d, b.i32(120))), b.arg(I64));
        b.countif(b.cmp_gt(d, b.i32(120)), b.arg(I64));
    }}

};
