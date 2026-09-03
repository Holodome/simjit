# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Sample:
    id: str
    title: str
    group: str
    query: str
    source: str = ""
    input_mode: str = "expression_sql"
    benchmarkable: bool = True
    tags: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class SampleGroup:
    title: str
    samples: tuple[Sample, ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(sample.id for sample in self.samples)


SAMPLES: tuple[Sample, ...] = (
    Sample(
        id="simple_arithmetic",
        title="Simple arithmetic",
        group="Basics",
        tags=("projection", "aggregate"),
        query="""INPUT (x i32, y i32);

SELECT
  (x + y) * 7 AS result,
  sum(x) AS total_x;
""",
    ),
    Sample(
        id="predicate_ifelse",
        title="Predicate ifelse",
        group="Basics",
        source="python/tests/e2e_test.py",
        tags=("predicate", "select"),
        query="""INPUT (a i32, b i32);

WITH
  keep = a > 0 AND b < 100

SELECT
  ifelse(keep, i32(1), i32(0)) AS flag,
  keep AS mask;
""",
    ),
    Sample(
        id="casts_and_promotions",
        title="Casts and promotions",
        group="Basics",
        tags=("casts", "promotion"),
        query="""INPUT (x i16, y i16, k i32);

SELECT
  i32(x) + i32(y) * k AS result;
""",
    ),
    Sample(
        id="multi_type_projection",
        title="Multi-type projection",
        group="Basics",
        tags=("projection", "casts", "multi-output"),
        query="""INPUT (
  i8_val i8,
  i16_val i16,
  i32_val i32,
  i64_val i64,
  offset i32
);

SELECT
  (i8_val << 1) & 127 AS byte_lane,
  greatest(i16_val << 1, 100) AS word_floor,
  (i32_val + offset) >> 2 AS shifted_i32,
  least(i64_val, i64_val * 3) AS signed_min;
""",
    ),
    Sample(
        id="deep_cast_chain",
        title="Deep cast chain",
        group="Basics",
        tags=("casts", "dag"),
        query="""INPUT (small i8, wide i32);

WITH
  extended = i32(small),
  summed = extended + wide,
  scaled = summed * 100

SELECT
  i64(i16(scaled)) AS roundtrip;
""",
    ),
    Sample(
        id="aggregate_pack",
        title="Aggregate pack",
        group="Aggregates",
        tags=("aggregate", "filter"),
        query="""INPUT (value i64, pred i1);

SELECT
  sum(value) AS total,
  sum(value) FILTER (WHERE pred) AS total_if,
  product(value) AS product_value,
  count_if(pred) AS count_true;
""",
    ),
    Sample(
        id="min_max_bounds",
        title="Min/max bounds",
        group="Aggregates",
        tags=("minmax", "projection"),
        query="""INPUT (value i32, lower i32, upper i32);

SELECT
  least(greatest(value, lower), upper) AS clamped;
""",
    ),
    Sample(
        id="shifted_aggregate_bundle",
        title="Shifted aggregate bundle",
        group="Aggregates",
        tags=("aggregate", "shift", "multi-output"),
        query="""INPUT (value i64);

SELECT
  sum(value << 1) AS sum_left,
  min(value >> 2) AS min_right_signed,
  max(srl(value, 2)) AS max_right_unsigned;
""",
    ),
    Sample(
        id="common_expr_aggregates",
        title="Common expr aggregates",
        group="Aggregates",
        tags=("aggregate", "cse", "projection"),
        query="""INPUT (a i64, b i64);

WITH
  common = (a + b) * 100

SELECT
  common AS projected,
  sum(common) AS total_common,
  product(common) AS product_common,
  min(common) AS min_common;
""",
    ),
    Sample(
        id="multi_sum_bases",
        title="Multi-sum bases",
        group="Aggregates",
        tags=("aggregate", "bitwise", "cse"),
        query="""INPUT (base i32, modifier i32);

SELECT
  sum(base) AS sum_base,
  sum(base + modifier) AS sum_offset,
  sum(base << 1) AS sum_doubled,
  sum(base & 255) AS sum_low_byte;
""",
    ),
    Sample(
        id="revenue_projection",
        title="Revenue projection",
        group="TPC-H-ish",
        tags=("tpch", "projection"),
        query="""INPUT (price f64, discount f64);

SELECT
  price * (1.0 - discount) AS revenue;
""",
    ),
    Sample(
        id="tpch_q6_filter",
        title="TPC-H Q6 filter",
        group="TPC-H-ish",
        tags=("tpch", "aggregate", "filter"),
        query="""INPUT (
  price f64,
  discount f64,
  quantity i32,
  shipdate i32
);

WITH
  revenue = price * (1.0 - discount),
  q6_filter = shipdate >= 9401
    AND shipdate <= 9501
    AND discount >= 0.05
    AND discount <= 0.07
    AND quantity < 24

SELECT
  sum(revenue) FILTER (WHERE q6_filter) AS q6_revenue;
""",
    ),
    Sample(
        id="complex_filter",
        title="Complex filter",
        group="TPC-H-ish",
        tags=("tpch", "predicate"),
        query="""INPUT (price f64, discount f64, tax f64);

WITH
  revenue = price * (1.0 - discount)

SELECT
  revenue > 1000.0 AND tax < 0.05 AS keep;
""",
    ),
    Sample(
        id="tpch_q1_pricing_summary",
        title="TPC-H Q1 pricing summary",
        group="TPC-H-ish",
        tags=("tpch", "aggregate", "float", "q1"),
        query="""INPUT (
  quantity f64,
  extended_price f64,
  discount f64,
  tax f64,
  shipdate i32,
  returnflag i8,
  linestatus i8
);

WITH
  q1_filter = shipdate <= 19980902,
  group_a_f = q1_filter AND returnflag = 1 AND linestatus = 1,
  group_n_o = q1_filter AND returnflag = 2 AND linestatus = 2,
  disc_price = extended_price * (1.0 - discount),
  charge = disc_price * (1.0 + tax)

SELECT
  sum(quantity) FILTER (WHERE q1_filter) AS sum_qty,
  sum(extended_price) FILTER (WHERE q1_filter) AS sum_base_price,
  sum(disc_price) FILTER (WHERE q1_filter) AS sum_disc_price,
  sum(charge) FILTER (WHERE q1_filter) AS sum_charge,
  sum(discount) FILTER (WHERE q1_filter) AS sum_discount,
  count_if(q1_filter) AS count_order,
  sum(quantity) FILTER (WHERE group_a_f) AS a_f_sum_qty,
  sum(charge) FILTER (WHERE group_a_f) AS a_f_sum_charge,
  count_if(group_a_f) AS a_f_count,
  sum(quantity) FILTER (WHERE group_n_o) AS n_o_sum_qty,
  sum(charge) FILTER (WHERE group_n_o) AS n_o_sum_charge,
  count_if(group_n_o) AS n_o_count;
""",
    ),
    Sample(
        id="lineitem_feature_pack",
        title="Lineitem feature pack",
        group="TPC-H-ish",
        tags=("tpch", "aggregate", "multi-output"),
        query="""INPUT (
  quantity i32,
  price i64,
  discount_bp i32,
  tax_bp i32,
  ship_delay_days i32
);

WITH
  gross = price * i64(quantity),
  discounted = gross * (10000 - i64(discount_bp)) / 10000,
  taxed = discounted * (10000 + i64(tax_bp)) / 10000,
  late = ship_delay_days > 30,
  large_order = quantity >= 25

SELECT
  taxed AS line_total,
  sum(taxed) FILTER (WHERE late) AS late_total,
  sum(taxed) FILTER (WHERE large_order) AS large_order_total,
  min(taxed) AS min_line_total,
  max(taxed) AS max_line_total;
""",
    ),
    Sample(
        id="lineitem_branchy_score",
        title="Lineitem branchy score",
        group="TPC-H-ish",
        tags=("tpch", "predicate", "select"),
        query="""INPUT (
  quantity i32,
  price i64,
  discount_bp i32,
  ship_delay_days i32
);

WITH
  gross = price * i64(quantity),
  discount_score = i64(discount_bp) * 17,
  late_penalty = ifelse(ship_delay_days > 30, i64(500), i64(0)),
  quantity_bonus = ifelse(quantity >= 25, i64(1000), i64(100)),
  score = gross - discount_score - late_penalty + quantity_bonus

SELECT
  score AS score,
  count_if(score > 10000) AS high_score_rows;
""",
    ),
    Sample(
        id="fp_dot_product_f64",
        title="F64 dot product",
        group="Float / HPC",
        tags=("float", "hpc", "dot"),
        query="""INPUT (x f64, y f64);

SELECT
  sum(x * y) AS dot_xy,
  sum(x * x) AS norm_x_sq,
  sum(y * y) AS norm_y_sq;
""",
    ),
    Sample(
        id="linear_regression_moments_f32",
        title="Linear regression moments",
        group="Float / HPC",
        tags=("float", "hpc", "regression"),
        query="""INPUT (x f32, y f32, x0 f32, y0 f32);

WITH
  centered_x = x - x0,
  centered_y = y - y0,
  xx = centered_x * centered_x,
  xy = centered_x * centered_y

SELECT
  sum(centered_x) AS sum_x,
  sum(xx) AS sum_xx,
  sum(centered_y) AS sum_y,
  sum(xy) AS sum_xy;
""",
    ),
    Sample(
        id="stddev_moments_f64",
        title="Stddev moments",
        group="Float / HPC",
        tags=("float", "hpc", "statistics"),
        query="""INPUT (value f64, valid bool);

SELECT
  sum(value) FILTER (WHERE valid) AS sum_value,
  sum(value * value) FILTER (WHERE valid) AS sum_value_sq,
  min(value) FILTER (WHERE valid) AS min_value,
  max(value) FILTER (WHERE valid) AS max_value,
  count_if(valid) AS valid_count;
""",
    ),
    Sample(
        id="covariance_moments_f64",
        title="Covariance moments",
        group="Float / HPC",
        tags=("float", "hpc", "statistics"),
        query="""INPUT (x f64, y f64);

WITH
  xy = x * y,
  xx = x * x,
  yy = y * y

SELECT
  sum(x) AS sum_x,
  sum(y) AS sum_y,
  sum(xy) AS sum_xy,
  sum(xx) AS sum_xx,
  sum(yy) AS sum_yy;
""",
    ),
    Sample(
        id="vector_norm3_f64",
        title="3D vector norms",
        group="Float / HPC",
        tags=("float", "hpc", "norm"),
        query="""INPUT (x f64, y f64, z f64);

WITH
  norm2 = x * x + y * y + z * z,
  norm = sqrt(norm2)

SELECT
  norm AS norm,
  sum(norm2) AS total_norm2,
  max(norm) AS max_norm;
""",
    ),
    Sample(
        id="stencil_residual_l2",
        title="Stencil residual L2",
        group="Float / HPC",
        tags=("float", "hpc", "stencil"),
        query="""INPUT (
  center f64,
  north f64,
  south f64,
  east f64,
  west f64,
  rhs f64
);

WITH
  laplacian = 4.0 * center - north - south - east - west,
  residual = laplacian - rhs,
  residual2 = residual * residual

SELECT
  residual AS residual,
  sum(residual2) AS residual_l2_sq,
  max(abs(residual)) AS max_abs_residual;
""",
    ),
    Sample(
        id="conjugate_gradient_step",
        title="CG update moments",
        group="Float / HPC",
        tags=("float", "hpc", "solver"),
        query="""INPUT (x f64, r f64, p f64, ap f64, alpha f64, beta f64);

WITH
  x_next = x + alpha * p,
  r_next = r - alpha * ap,
  p_next = r_next + beta * p

SELECT
  x_next AS x_next,
  r_next AS r_next,
  p_next AS p_next,
  sum(r_next * r_next) AS residual_norm2,
  sum(p * ap) AS p_ap_dot;
""",
    ),
    Sample(
        id="particle_energy",
        title="Particle energy",
        group="Float / HPC",
        tags=("float", "hpc", "simulation"),
        query="""INPUT (
  mass f64,
  vx f64,
  vy f64,
  vz f64,
  potential f64
);

WITH
  speed2 = vx * vx + vy * vy + vz * vz,
  kinetic = 0.5 * mass * speed2,
  total = kinetic + potential

SELECT
  total AS total_energy,
  sum(total) AS system_energy,
  max(speed2) AS max_speed2;
""",
    ),
    Sample(
        id="finite_volume_flux",
        title="Finite volume flux",
        group="Float / HPC",
        tags=("float", "hpc", "fluid"),
        query="""INPUT (rho f64, u f64, pressure f64, gamma f64);

WITH
  momentum = rho * u,
  kinetic = 0.5 * rho * u * u,
  internal_energy = pressure / (gamma - 1.0),
  total_energy = internal_energy + kinetic,
  mass_flux = momentum,
  momentum_flux = momentum * u + pressure,
  energy_flux = (total_energy + pressure) * u

SELECT
  mass_flux AS mass_flux,
  momentum_flux AS momentum_flux,
  energy_flux AS energy_flux,
  sum(energy_flux) AS total_energy_flux;
""",
    ),
    Sample(
        id="discounted_charge_f64",
        title="Discounted charge F64",
        group="Float / HPC",
        tags=("float", "hpc", "tpch"),
        query="""INPUT (
  quantity f64,
  extended_price f64,
  discount f64,
  tax f64
);

WITH
  disc_price = extended_price * (1.0 - discount),
  charge = disc_price * (1.0 + tax)

SELECT
  sum(quantity) AS sum_qty,
  sum(extended_price) AS sum_base_price,
  sum(disc_price) AS sum_disc_price,
  sum(charge) AS sum_charge;
""",
    ),
    Sample(
        id="nullable_projection",
        title="Nullable projection",
        group="Nullability / Arrow",
        tags=("nullable", "arrow"),
        query="""INPUT (x i32?, y i32?);

SELECT
  x + y AS sum_xy,
  coalesce(x, y, 0) AS filled,
  x IS NULL AS missing_x,
  y IS NOT NULL AS present_y;
""",
    ),
    Sample(
        id="nullable_aggregate",
        title="Nullable aggregate",
        group="Nullability / Arrow",
        tags=("nullable", "aggregate"),
        query="""INPUT (x i32?);

SELECT
  sum(x) AS total,
  count_if(x > 1) AS count_gt_one;
""",
    ),
    Sample(
        id="nullable_revenue",
        title="Nullable revenue stress",
        group="Nullability / Arrow",
        tags=("nullable", "aggregate", "stress"),
        query="""INPUT (
  qty i32?,
  unit_price i32?,
  discount_bp i32?,
  ship_delay_days i32?
);

WITH
  safe_qty = coalesce(qty, 0),
  price = nullif(unit_price, 0),
  gross = i64(safe_qty) * i64(price),
  net = gross * (10000 - i64(coalesce(discount_bp, 0))) / 10000,
  valid = price IS NOT NULL AND safe_qty > 0

SELECT
  net AS net,
  valid AS is_valid,
  sum(net) FILTER (WHERE valid) AS total_net,
  sum(net) FILTER (WHERE valid AND ship_delay_days > 30) AS late_net;
""",
    ),
    Sample(
        id="nullable_split_valids",
        title="Nullable split valids",
        group="Nullability / Arrow",
        tags=("nullable", "aggregate", "predicate"),
        query="""INPUT (
  qty i32?,
  unit_price i32?,
  discount_bp i32?,
  delay i32
);

WITH
  safe_qty = coalesce(qty, 0),
  price = nullif(unit_price, 0),
  price_value = coalesce(price, 0),
  discount = coalesce(discount_bp, 0),
  gross = i64(safe_qty) * i64(price_value),
  net = gross * (10000 - i64(discount)) / 10000,
  valid = price IS NOT NULL AND safe_qty > 0,
  high_qty = valid AND safe_qty >= 25,
  expensive = valid AND price_value >= 5000,
  late = valid AND delay > 30

SELECT
  sum(net) FILTER (WHERE high_qty) AS high_qty_net,
  sum(net) FILTER (WHERE expensive) AS expensive_net,
  count_if(late) AS late_count;
""",
    ),
    Sample(
        id="nullable_select_projection",
        title="Nullable select projection",
        group="Nullability / Arrow",
        tags=("nullable", "projection", "select"),
        query="""INPUT (
  qty i32?,
  unit_price i32?,
  discount_bp i32?,
  delay i32
);

WITH
  safe_qty = coalesce(qty, 0),
  price = nullif(unit_price, 0),
  price_value = coalesce(price, 0),
  discount = coalesce(discount_bp, 0),
  gross = i64(safe_qty) * i64(price_value),
  net = gross * (10000 - i64(discount)) / 10000,
  valid = price IS NOT NULL AND safe_qty > 0,
  high_qty = valid AND safe_qty >= 25,
  expensive = valid AND price_value >= 5000,
  late = valid AND delay > 30

SELECT
  ifelse(high_qty, net, i64(0)) AS high_qty_out,
  ifelse(expensive, net, i64(0)) AS expensive_out,
  ifelse(late, i64(1), i64(0)) AS late_out;
""",
    ),
    Sample(
        id="bitwise_mask_shift",
        title="Bitwise mask and shift",
        group="Bitwise / Vectorizer",
        tags=("bitwise", "vectorizer"),
        query="""INPUT (a i16, b i16, c i16);

SELECT
  ((a & 255) | (b << 2)) # c AS result;
""",
    ),
    Sample(
        id="popcnt_lzcnt",
        title="Popcnt and lzcnt",
        group="Bitwise / Vectorizer",
        tags=("bitwise", "unary"),
        query="""INPUT (x i32, y i32, z i32);

SELECT
  abs(-x) + popcnt(y) + lzcnt(z) AS result;
""",
    ),
    Sample(
        id="hash_mix",
        title="Hash mix",
        group="Bitwise / Vectorizer",
        tags=("bitwise", "hash"),
        query="""INPUT (a i32, discount_bp i32, quantity i32);

WITH
  left_hash = ((i64(a) * 31) # (i64(discount_bp) * 7)) # 4294967295,
  right_hash = (i64(quantity) * 13) & 4294967295

SELECT
  i32((left_hash # right_hash) & 255) AS hash_byte;
""",
    ),
    Sample(
        id="nested_bitmask_pipeline",
        title="Nested bitmask pipeline",
        group="Bitwise / Vectorizer",
        tags=("bitwise", "shift", "vectorizer"),
        query="""INPUT (a i32, b i32, c i32);

WITH
  low_mix = (a # b) & 65535,
  high_mix = (c & ~65535) >> 1

SELECT
  low_mix | high_mix AS mixed;
""",
    ),
    Sample(
        id="common_bitmask_operations",
        title="Common bitmask operations",
        group="Bitwise / Vectorizer",
        tags=("bitwise", "cse", "multi-output"),
        query="""INPUT (value i32);

WITH
  high = value & 65280,
  low = value & 255

SELECT
  high << 8 AS high_left,
  srl(high, 8) AS high_right,
  high | low AS recombined,
  low # 255 AS low_inverted,
  low + low AS low_doubled;
""",
    ),
    Sample(
        id="minmax_shift_mix",
        title="Min/max shift mix",
        group="Bitwise / Vectorizer",
        tags=("bitwise", "minmax", "shift"),
        query="""INPUT (value i16, constant i16);

SELECT
  least(value << 1, greatest(value >> 1, constant)) AS result;
""",
    ),
    Sample(
        id="shared_subexpression",
        title="Shared subexpression",
        group="CSE / DAGs",
        tags=("cse", "dag"),
        query="""INPUT (x i32, y i32, z i32);

WITH
  xy = x * y

SELECT
  xy + z AS plus_z,
  xy - z AS minus_z,
  xy * 2 AS doubled;
""",
    ),
    Sample(
        id="large_shared_dag",
        title="Large shared DAG",
        group="CSE / DAGs",
        tags=("cse", "dag"),
        query="""INPUT (x i64, y i64, z i64);

WITH
  x_plus_y = x + y,
  x_times_z = x * z,
  y_minus_z = y - z

SELECT
  x_plus_y * x_times_z AS product_path,
  x_plus_y + y_minus_z AS sum_path,
  x_times_z - y_minus_z AS difference_path,
  x_plus_y * 2 AS scaled_sum;
""",
    ),
    Sample(
        id="commutative_cse_probe",
        title="Commutative CSE",
        group="CSE / DAGs",
        tags=("cse", "commutative", "multi-output"),
        query="""INPUT (x i32, y i32);

SELECT
  x + y AS add_xy,
  y + x AS add_yx,
  (x + y) + 5 AS grouped_left,
  x + (y + 5) AS grouped_right,
  least(x, y) AS least_xy,
  least(y, x) AS least_yx,
  x * 3 AS mul_right,
  3 * x AS mul_left;
""",
    ),
    Sample(
        id="multi_level_dag",
        title="Multi-level DAG",
        group="CSE / DAGs",
        tags=("cse", "dag", "bitwise"),
        query="""INPUT (w i32, x i32, y i32, z i32);

WITH
  w_plus_x = w + x,
  y_times_z = y * z,
  sum1 = w_plus_x + y_times_z,
  diff1 = w_plus_x - y_times_z

SELECT
  sum1 * 2 AS scaled_sum,
  sum1 >> 1 AS halved_sum,
  least(diff1, 100) AS diff_floor,
  greatest(diff1, 0) AS diff_ceiling,
  w_plus_x # y_times_z AS mixed_inputs;
""",
    ),
)


# TODO: Add UI-visible samples once Expression SQL supports the required syntax:
# grouped aggregation and table names, gather/scatter/pack, splat inputs, and
# timestamp literals/types.

DEFAULT_SAMPLE_ID = "simple_arithmetic"
SAMPLES_BY_ID = {sample.id: sample for sample in SAMPLES}
DEFAULT_QUERY = SAMPLES_BY_ID[DEFAULT_SAMPLE_ID].query

_GROUP_ORDER = (
    "Basics",
    "Aggregates",
    "TPC-H-ish",
    "Float / HPC",
    "Nullability / Arrow",
    "Bitwise / Vectorizer",
    "CSE / DAGs",
)

SAMPLE_GROUPS = tuple(
    SampleGroup(
        title=group,
        samples=tuple(sample for sample in SAMPLES if sample.group == group),
    )
    for group in _GROUP_ORDER
)
