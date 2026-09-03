#!/usr/bin/env python3
# This file is part of Simjit project <https://simjit.org>
#
# See LICENSE for license and copyright information
# SPDX-License-Identifier: Zlib


import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SIMJIT_ROOT = SCRIPT_DIR.parent.parent


def env(name: str, default: str) -> str:
    return os.environ.get(name, default)


DUCKDB_DIR = os.environ.get("DUCKDB_DIR")
if not DUCKDB_DIR:
    print("supply DUCKDB_DIR")
    exit(1)
DUCKDB_DIR = Path(DUCKDB_DIR)

BUILD_TYPE = env("BUILD_TYPE", "release")
ROWS = int(env("ROWS", "10000000"))
HOT_RUNS = int(env("HOT_RUNS", env("RUNS", "7")))
SALES_FACTORIAL_HOT_RUNS = int(env("SALES_FACTORIAL_HOT_RUNS", "1"))
RESULT_DIR = Path(env("RESULT_DIR", "/private/tmp"))
SKIP_BUILD = env("SKIP_BUILD", "0") == "1"
NULL_DENSITIES = [int(x) for x in env("NULL_DENSITIES", "0,1,10,50,90").split(",") if x]
NULL_PATTERNS = [
    x for x in env("NULL_PATTERNS", "periodic,random,runs").split(",") if x
]

EXTENSION_CONFIG = SCRIPT_DIR / "extension_config.cmake"
DUCKDB_BIN = DUCKDB_DIR / "build" / BUILD_TYPE / "duckdb"
SIMJIT_EXTENSION = (
    DUCKDB_DIR
    / "build"
    / BUILD_TYPE
    / "extension"
    / "simjit"
    / "simjit.duckdb_extension"
)

TIMER_RE = re.compile(r"Run Time \(s\): real ([0-9.]+)")
BENCH_PREFIX = "__SIMJIT_BENCH__|"
CHECK_PREFIX = "__SIMJIT_CHECK__|"
META_PREFIX = "__SIMJIT_META__|"

SALES_FACTORS = {
    "numeric_layout": ["wide_i64", "mixed_narrow"],
    "price_validity": ["price_nn", "price_nullable_10pct"],
    "discount_representation": ["discount_column", "discount_constant"],
    "categorical_representation": [
        "cat_flat",
        "cat_dict_gather",
        "cat_dict_specialized",
    ],
}


def run(cmd, **kwargs):
    print("+", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def build_duckdb():
    if SKIP_BUILD:
        print("==> Skipping DuckDB build")
        return
    run(
        [
            "make",
            "-C",
            str(DUCKDB_DIR),
            BUILD_TYPE,
            f"EXTENSION_CONFIGS={EXTENSION_CONFIG}",
        ]
    )


def sql_quote(path):
    return str(path).replace("'", "''")


class SqlBench:
    def __init__(self):
        self.lines = [
            ".mode csv",
            ".headers off",
            ".timer on",
            f"LOAD '{sql_quote(SIMJIT_EXTENSION)}';",
            "PRAGMA threads=1;",
        ]
        self.bench_names = []

    def add(self, line):
        self.lines.append(line)

    def marker(self, prefix, name):
        self.lines.append(f"SELECT '{prefix}{name}';")

    def bench(self, name, query):
        self.bench_names.append(name)
        self.marker(BENCH_PREFIX, name)
        self.lines.append(f"COPY ({query}) TO '/dev/null';")

    def check(self, name, query):
        self.lines.append(
            f"SELECT '{CHECK_PREFIX}{name}|' || CASE WHEN ({query}) THEN 'ok' ELSE 'fail' END;"
        )

    def meta_compile(self, kernel_name):
        self.lines.append(
            f"SELECT '{META_PREFIX}{kernel_name}|compile_us|' || simjit_last_compile_us('{kernel_name}');"
        )
        self.lines.append(
            f"SELECT '{META_PREFIX}{kernel_name}|vectorizer_status|' || simjit_vectorizer_status('{kernel_name}');"
        )

    def reset(self):
        self.lines.append("SELECT simjit_reset_kernels();")

    def script(self):
        return "\n".join(self.lines) + "\n"


def null_condition(pattern, density):
    if density <= 0:
        return "false"
    if density >= 100:
        return "true"
    if pattern == "periodic":
        return f"(a % 100) < {density}"
    if pattern == "random":
        return f"(hash(a) % 100) < {density}"
    if pattern == "runs":
        return f"((a // 2048) % 100) < {density}"
    raise ValueError(f"unknown null pattern: {pattern}")


def q19_predicate(prefix=""):
    p = prefix
    return f"""
        (
          {p}brand = 12 AND {p}container BETWEEN 1 AND 4
          AND {p}quantity BETWEEN 1 AND 11 AND {p}size BETWEEN 1 AND 5
        )
        OR (
          {p}brand = 23 AND {p}container BETWEEN 5 AND 8
          AND {p}quantity BETWEEN 10 AND 20 AND {p}size BETWEEN 1 AND 10
        )
        OR (
          {p}brand = 34 AND {p}container BETWEEN 9 AND 12
          AND {p}quantity BETWEEN 20 AND 30 AND {p}size BETWEEN 1 AND 15
        )
    """


def sales_short(value):
    return {
        "wide_i64": "wide_i64",
        "mixed_narrow": "mixed_narrow",
        "price_nn": "nn",
        "price_nullable_10pct": "nullable10",
        "discount_column": "column",
        "discount_constant": "const",
        "cat_flat": "flat",
        "cat_dict_gather": "dict_gather",
        "cat_dict_specialized": "dict_spec",
    }[value]


def sales_kernel_name(numeric, price_validity, discount_repr, categorical_repr):
    return (
        f"simjit_sales_{numeric}__price_{sales_short(price_validity)}"
        f"__disc_{sales_short(discount_repr)}__cat_{sales_short(categorical_repr)}"
    )


def sales_benchmark_name(numeric, price_validity, discount_repr, categorical_repr):
    return (
        f"sales__num_{sales_short(numeric)}__price_{sales_short(price_validity)}"
        f"__disc_{sales_short(discount_repr)}__cat_{sales_short(categorical_repr)}"
    )


def sales_native_sql(numeric, price_validity, discount_repr):
    quantity = "quantity_i64" if numeric == "wide_i64" else "quantity_i16"
    price = "price_nullable" if price_validity == "price_nullable_10pct" else "price"
    if discount_repr == "discount_constant":
        discount = "500"
    else:
        discount = "discount_i64" if numeric == "wide_i64" else "discount_i16"
    return f"""
WITH x AS (
  SELECT
    {quantity}::BIGINT AS quantity,
    {price} AS price,
    {discount}::BIGINT AS discount_bp,
    return_flag,
    channel
  FROM sales_like
), y AS (
  SELECT
    quantity,
    price,
    return_flag,
    channel,
    quantity * price * (10000 - discount_bp) // 10000 AS net
  FROM x
), z AS (
  SELECT
    CASE WHEN return_flag = 1 THEN -net ELSE net END AS signed_net,
    quantity,
    channel
  FROM y
)
SELECT
  sum(signed_net) AS sum_revenue,
  sum(signed_net) FILTER (WHERE channel = 2) AS channel2_revenue,
  count(*) FILTER (WHERE quantity >= 10) AS bulk_count
FROM z
"""


def sales_simjit_sql(numeric, price_validity, discount_repr, categorical_repr):
    function = (
        "simjit_sales_wide_i64"
        if numeric == "wide_i64"
        else "simjit_sales_mixed_narrow"
    )
    quantity = "quantity_i64" if numeric == "wide_i64" else "quantity_i16"
    price = "price_nullable" if price_validity == "price_nullable_10pct" else "price"
    if discount_repr == "discount_constant":
        discount = "500::BIGINT" if numeric == "wide_i64" else "500::SMALLINT"
    else:
        discount = "discount_i64" if numeric == "wide_i64" else "discount_i16"
    if categorical_repr == "cat_dict_gather":
        return_flag = "simjit_dict_gather_i8(return_flag)"
        channel = "simjit_dict_gather_i8(channel)"
    elif categorical_repr == "cat_dict_specialized":
        return_flag = "simjit_dict_spec_i8(return_flag)"
        channel = "simjit_dict_spec_i8(channel)"
    else:
        return_flag = "return_flag"
        channel = "channel"
    return f"""
SELECT
  (r).sum_revenue AS sum_revenue,
  (r).channel2_revenue AS channel2_revenue,
  (r).bulk_count AS bulk_count
FROM (
  SELECT {function}({quantity}, {price}, {discount}, {return_flag}, {channel}) AS r
  FROM sales_like
)
"""


def build_sales_factorial(sql):
    cells = []
    sql.add(f"""
CREATE TEMP TABLE sales_like AS
SELECT
  i::BIGINT AS row_id,
  (1 + (i % 50))::BIGINT AS quantity_i64,
  (1 + (i % 50))::SMALLINT AS quantity_i16,
  (1000 + (i % 10000))::BIGINT AS price,
  CASE WHEN (hash(i) % 100) < 10 THEN NULL ELSE (1000 + (i % 10000))::BIGINT END AS price_nullable,
  (i % 1000)::BIGINT AS discount_i64,
  (i % 1000)::SMALLINT AS discount_i16,
  (i % 2)::TINYINT AS return_flag,
  (i % 4)::TINYINT AS channel
FROM range({ROWS}) tbl(i);
""")
    sql.add("SELECT '__SIMJIT_DATA__|sales_like|' || count(*) FROM sales_like;")

    for numeric in SALES_FACTORS["numeric_layout"]:
        for price_validity in SALES_FACTORS["price_validity"]:
            for discount_repr in SALES_FACTORS["discount_representation"]:
                for categorical_repr in SALES_FACTORS["categorical_representation"]:
                    name = sales_benchmark_name(
                        numeric, price_validity, discount_repr, categorical_repr
                    )
                    kernel = sales_kernel_name(
                        numeric, price_validity, discount_repr, categorical_repr
                    )
                    native = sales_native_sql(numeric, price_validity, discount_repr)
                    simjit = sales_simjit_sql(
                        numeric, price_validity, discount_repr, categorical_repr
                    )
                    cell = {
                        "name": name,
                        "kernel": kernel,
                        "numeric_layout": numeric,
                        "price_validity": price_validity,
                        "discount_representation": discount_repr,
                        "categorical_representation": categorical_repr,
                        "native_sql": native.strip(),
                        "simjit_sql": simjit.strip(),
                    }
                    cells.append(cell)
                    sql.check(
                        name,
                        f"""
EXISTS (
  WITH native AS ({native}), simjit AS ({simjit})
  SELECT 1 FROM native, simjit
  WHERE native.sum_revenue IS NOT DISTINCT FROM simjit.sum_revenue
    AND native.channel2_revenue IS NOT DISTINCT FROM simjit.channel2_revenue
    AND native.bulk_count IS NOT DISTINCT FROM simjit.bulk_count
)
""",
                    )

    for cell in cells:
        for i in range(SALES_FACTORIAL_HOT_RUNS):
            sql.bench(f"{cell['name']}.duckdb_sql.{i}", cell["native_sql"])
        sql.reset()
        sql.bench(f"{cell['name']}.simjit_cold.0", cell["simjit_sql"])
        sql.meta_compile(cell["kernel"])
        for i in range(SALES_FACTORIAL_HOT_RUNS):
            sql.bench(f"{cell['name']}.simjit_hot.{i}", cell["simjit_sql"])
    return cells


def add_sales_edge_checks(sql):
    cases = {
        "zero_rows": """
SELECT 1::BIGINT AS quantity, 1000::BIGINT AS price, 10::BIGINT AS discount_bp,
       0::TINYINT AS return_flag, 0::TINYINT AS channel
WHERE false
""",
        "all_channel2": """
SELECT (1 + (i % 50))::BIGINT AS quantity, (1000 + i)::BIGINT AS price, 10::BIGINT AS discount_bp,
       (i % 2)::TINYINT AS return_flag, 2::TINYINT AS channel
FROM range(128) tbl(i)
""",
        "no_channel2": """
SELECT (1 + (i % 50))::BIGINT AS quantity, (1000 + i)::BIGINT AS price, 10::BIGINT AS discount_bp,
       (i % 2)::TINYINT AS return_flag, 1::TINYINT AS channel
FROM range(128) tbl(i)
""",
        "all_returns_false": """
SELECT (1 + (i % 50))::BIGINT AS quantity, (1000 + i)::BIGINT AS price, 10::BIGINT AS discount_bp,
       0::TINYINT AS return_flag, (i % 4)::TINYINT AS channel
FROM range(128) tbl(i)
""",
        "all_returns_true": """
SELECT (1 + (i % 50))::BIGINT AS quantity, (1000 + i)::BIGINT AS price, 10::BIGINT AS discount_bp,
       1::TINYINT AS return_flag, (i % 4)::TINYINT AS channel
FROM range(128) tbl(i)
""",
        "all_price_null": """
SELECT (1 + (i % 50))::BIGINT AS quantity, NULL::BIGINT AS price, 10::BIGINT AS discount_bp,
       (i % 2)::TINYINT AS return_flag, (i % 4)::TINYINT AS channel
FROM range(128) tbl(i)
""",
    }
    for name, source in cases.items():
        native = f"""
WITH x AS ({source}), y AS (
  SELECT
    CASE WHEN return_flag = 1 THEN -(quantity * price * (10000 - discount_bp) // 10000)
         ELSE quantity * price * (10000 - discount_bp) // 10000
    END AS signed_net,
    quantity,
    channel
  FROM x
)
SELECT
  sum(signed_net) AS sum_revenue,
  sum(signed_net) FILTER (WHERE channel = 2) AS channel2_revenue,
  count(*) FILTER (WHERE quantity >= 10) AS bulk_count
FROM y
"""
        simjit = f"""
SELECT
  (r).sum_revenue AS sum_revenue,
  (r).channel2_revenue AS channel2_revenue,
  (r).bulk_count AS bulk_count
FROM (
  SELECT simjit_sales_wide_i64(quantity, price, discount_bp, return_flag, channel) AS r
  FROM ({source})
)
"""
        sql.check(
            f"sales_edge_{name}",
            f"""
EXISTS (
  WITH native AS ({native}), simjit AS ({simjit})
  SELECT 1 FROM native, simjit
  WHERE native.sum_revenue IS NOT DISTINCT FROM simjit.sum_revenue
    AND native.channel2_revenue IS NOT DISTINCT FROM simjit.channel2_revenue
    AND native.bulk_count IS NOT DISTINCT FROM simjit.bulk_count
)
""",
        )


def create_sql():
    sql = SqlBench()
    sql.add(f"""
CREATE TEMP TABLE t AS
SELECT
  i::BIGINT AS a,
  (i * 2)::BIGINT AS b,
  (1000 + (i % 10000))::BIGINT AS price,
  (i % 1000)::BIGINT AS discount_bp,
  (i % 700)::BIGINT AS tax_bp,
  (10000 + (i % 1000))::INTEGER AS shipdate,
  (1 + (i % 50))::INTEGER AS quantity_i32,
  (1 + (i % 50))::BIGINT AS quantity,
  (i % 100)::BIGINT AS delay,
  CASE WHEN i % 3 = 0 THEN 12 WHEN i % 3 = 1 THEN 23 ELSE 34 END::BIGINT AS brand,
  (1 + (i % 12))::BIGINT AS container,
  (1 + (i % 20))::BIGINT AS size
FROM range({ROWS}) tbl(i);
""")
    sql.add("SELECT '__SIMJIT_DATA__|t|' || count(*) FROM t;")

    for density in NULL_DENSITIES:
        for pattern in NULL_PATTERNS:
            table = f"nullable_{pattern}_{density}"
            cond = null_condition(pattern, density)
            sql.add(f"""
CREATE TEMP TABLE {table} AS
SELECT
  CASE WHEN {cond} THEN NULL ELSE quantity END AS quantity,
  CASE WHEN {cond} THEN NULL ELSE price END AS price,
  CASE WHEN {cond} THEN NULL ELSE discount_bp END AS discount_bp,
  delay
FROM t;
""")

    sales_cells = build_sales_factorial(sql)
    add_sales_edge_checks(sql)

    add_native = "SELECT sum(a + b) FROM t"
    add_simjit = "SELECT sum(simjit_add_i64_nn(a, b)) FROM t"
    add_simjit_fused = "SELECT simjit_sum_add_i64_nn(a, b) FROM t"
    net_native = "SELECT sum(price * (10000 - discount_bp) // 10000) FROM t"
    net_simjit = "SELECT sum(simjit_net_i64_nn(price, discount_bp)) FROM t"
    net_simjit_fused = "SELECT simjit_sum_net_i64_nn(price, discount_bp) FROM t"
    ship_native = "SELECT count(*) FROM t WHERE shipdate <= 10592 AND quantity_i32 < 24"
    ship_simjit = (
        "SELECT count(*) FROM t WHERE simjit_ship_filter_i32_nn(shipdate, quantity_i32)"
    )
    ship_simjit_fused = (
        "SELECT simjit_count_ship_filter_i32_nn(shipdate, quantity_i32) FROM t"
    )
    q1_native = """
SELECT
  sum(price * (10000 - discount_bp) // 10000) AS sum_net,
  sum(price * (10000 - discount_bp) * (10000 + tax_bp) // 100000000) AS sum_charge,
  count(*) AS row_count
FROM t
"""
    q1_simjit = """
SELECT (r).sum_net AS sum_net, (r).sum_charge AS sum_charge, (r).row_count AS row_count
FROM (SELECT simjit_q1_multi_i64(price, discount_bp, tax_bp) AS r FROM t)
"""
    q19_native = f"""
SELECT count(*) AS row_count, sum(price * (10000 - discount_bp) // 10000) AS revenue
FROM t WHERE {q19_predicate()}
"""
    q19_simjit = """
SELECT (r).row_count AS row_count, (r).revenue AS revenue
FROM (SELECT simjit_q19_mask_i64(brand, container, quantity, size, price, discount_bp) AS r FROM t)
"""

    sql.check("add_i64", f"(({add_native}) IS NOT DISTINCT FROM ({add_simjit}))")
    sql.check(
        "sum_add_i64", f"(({add_native}) IS NOT DISTINCT FROM ({add_simjit_fused}))"
    )
    sql.check("net_i64", f"(({net_native}) IS NOT DISTINCT FROM ({net_simjit}))")
    sql.check(
        "sum_net_i64", f"(({net_native}) IS NOT DISTINCT FROM ({net_simjit_fused}))"
    )
    sql.check("ship_filter", f"(({ship_native}) IS NOT DISTINCT FROM ({ship_simjit}))")
    sql.check(
        "count_ship_filter",
        f"(({ship_native}) IS NOT DISTINCT FROM ({ship_simjit_fused}))",
    )
    sql.check(
        "q1_multi",
        f"""
EXISTS (
  WITH native AS ({q1_native}), simjit AS ({q1_simjit})
  SELECT 1 FROM native, simjit
  WHERE native.sum_net IS NOT DISTINCT FROM simjit.sum_net
    AND native.sum_charge IS NOT DISTINCT FROM simjit.sum_charge
    AND native.row_count IS NOT DISTINCT FROM simjit.row_count
)
""",
    )
    sql.check(
        "q19_mask",
        f"""
EXISTS (
  WITH native AS ({q19_native}), simjit AS ({q19_simjit})
  SELECT 1 FROM native, simjit
  WHERE native.row_count IS NOT DISTINCT FROM simjit.row_count
    AND native.revenue IS NOT DISTINCT FROM simjit.revenue
)
""",
    )

    nullable_queries = {}
    for density in NULL_DENSITIES:
        for pattern in NULL_PATTERNS:
            table = f"nullable_{pattern}_{density}"
            native = f"""
WITH x AS (
  SELECT coalesce(quantity, 0) AS quantity,
         nullif(price, 0) AS price,
         coalesce(discount_bp, 0) AS discount_bp,
         delay
  FROM {table}
), y AS (
  SELECT quantity, price, discount_bp, delay,
         price IS NOT NULL AND quantity > 0 AS valid,
         quantity * price * (10000 - discount_bp) // 10000 AS net
  FROM x
)
SELECT
  sum(net) FILTER (WHERE valid) AS sum_valid_net,
  sum(net) FILTER (WHERE valid AND delay > 30) AS sum_delay_net,
  count(*) FILTER (WHERE valid) AS count_valid
FROM y
"""
            simjit = f"""
SELECT (r).sum_valid_net AS sum_valid_net, (r).sum_delay_net AS sum_delay_net, (r).count_valid AS count_valid
FROM (SELECT simjit_nullable_revenue_i64(quantity, price, discount_bp, delay) AS r FROM {table})
"""
            name = f"nullable_revenue_{pattern}_{density}"
            nullable_queries[name] = (native, simjit)
            sql.check(
                name,
                f"""
EXISTS (
  WITH native AS ({native}), simjit AS ({simjit})
  SELECT 1 FROM native, simjit
  WHERE native.sum_valid_net IS NOT DISTINCT FROM simjit.sum_valid_net
    AND native.sum_delay_net IS NOT DISTINCT FROM simjit.sum_delay_net
    AND native.count_valid IS NOT DISTINCT FROM simjit.count_valid
)
""",
            )

    benches = [
        ("scan_count", "scan_floor", "SELECT count(*) FROM t", None),
        ("scan_sum_price", "scan_floor", "SELECT sum(price) FROM t", None),
        ("add_i64", "simjit_add_i64_nn", add_native, add_simjit),
        ("sum_add_i64", "simjit_sum_add_i64_nn", add_native, add_simjit_fused),
        ("net_i64", "simjit_net_i64_nn", net_native, net_simjit),
        ("sum_net_i64", "simjit_sum_net_i64_nn", net_native, net_simjit_fused),
        ("ship_filter", "simjit_ship_filter_i32_nn", ship_native, ship_simjit),
        (
            "count_ship_filter",
            "simjit_count_ship_filter_i32_nn",
            ship_native,
            ship_simjit_fused,
        ),
        ("q1_multi", "simjit_q1_multi_i64", q1_native, q1_simjit),
        ("q19_mask", "simjit_q19_mask_i64", q19_native, q19_simjit),
    ]

    for name, kernel, native, simjit in benches:
        if kernel == "scan_floor":
            for i in range(HOT_RUNS):
                sql.bench(f"{name}.duckdb_sql.{i}", native)
            continue
        for i in range(HOT_RUNS):
            sql.bench(f"{name}.duckdb_sql.{i}", native)
        sql.reset()
        sql.bench(f"{name}.simjit_cold.0", simjit)
        sql.meta_compile(kernel)
        for i in range(HOT_RUNS):
            sql.bench(f"{name}.simjit_hot.{i}", simjit)

    for name, (native, simjit) in nullable_queries.items():
        for i in range(HOT_RUNS):
            sql.bench(f"{name}.duckdb_sql.{i}", native)
        sql.reset()
        sql.bench(f"{name}.simjit_cold.0", simjit)
        sql.meta_compile("simjit_nullable_revenue_i64")
        for i in range(HOT_RUNS):
            sql.bench(f"{name}.simjit_hot.{i}", simjit)

    return sql.script(), sales_cells


def parse_duckdb_output(output):
    timings = {}
    checks = {}
    meta = {}
    pending = None
    runtime_countdown = 0
    for raw in output.splitlines():
        line = raw.strip().strip("\r")
        if line.startswith(CHECK_PREFIX):
            _, name, status = line.split("|", 2)
            checks[name] = status
            continue
        if line.startswith(META_PREFIX):
            parts = line.split("|", 3)
            if len(parts) == 4:
                _, kernel, key, value = parts
                meta.setdefault(kernel, {})[key] = value
            continue
        if line.startswith(BENCH_PREFIX):
            pending = line[len(BENCH_PREFIX) :]
            runtime_countdown = 2
            continue
        match = TIMER_RE.search(line)
        if match and pending:
            runtime_countdown -= 1
            if runtime_countdown == 0:
                timings[pending] = float(match.group(1)) * 1_000_000.0
                pending = None
    return timings, checks, meta


def summarize_timings(timings):
    grouped = {}
    for name, us in timings.items():
        bench, variant, _ = name.rsplit(".", 2)
        grouped.setdefault(bench, {}).setdefault(variant, []).append(us)
    summary = {}
    for bench, variants in grouped.items():
        summary[bench] = {}
        for variant, values in variants.items():
            median_us = statistics.median(values)
            summary[bench][variant] = {
                "median_us": median_us,
                "min_us": min(values),
                "runs": len(values),
                "rows_per_s": ROWS / (median_us / 1_000_000.0)
                if median_us > 0
                else None,
            }
    return summary


def build_sales_factorial_report(summary, checks, meta, cells):
    rows = []
    for cell in cells:
        bench = summary.get(cell["name"], {})
        native = bench.get("duckdb_sql", {})
        cold = bench.get("simjit_cold", {})
        hot = bench.get("simjit_hot", {})
        native_us = native.get("median_us")
        hot_us = hot.get("median_us")
        speedup = native_us / hot_us if native_us and hot_us else None
        kernel_meta = meta.get(cell["kernel"], {})
        row = dict(cell)
        row.update(
            {
                "correctness": checks.get(cell["name"]),
                "duckdb_sql_us": native_us,
                "simjit_cold_us": cold.get("median_us"),
                "simjit_hot_us": hot_us,
                "hot_speedup": speedup,
                "compile_us": int(kernel_meta["compile_us"])
                if kernel_meta.get("compile_us", "").isdigit()
                else None,
                "vectorizer_status": kernel_meta.get("vectorizer_status"),
            }
        )
        rows.append(row)

    grouped = {}
    for factor, levels in SALES_FACTORS.items():
        grouped[factor] = {}
        for level in levels:
            values = [
                row["hot_speedup"]
                for row in rows
                if row[factor] == level and row["hot_speedup"] is not None
            ]
            grouped[factor][level] = {
                "cells": len([row for row in rows if row[factor] == level]),
                "avg_hot_speedup": statistics.mean(values) if values else None,
            }
    return {
        "factors": SALES_FACTORS,
        "timed_query_variants": len(cells) * 3,
        "cells": rows,
        "grouped": grouped,
    }


def write_reports(summary, checks, meta, sales_factorial):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = (
        RESULT_DIR / f"simjit_duckdb_fragments_{BUILD_TYPE}_{ROWS}_{timestamp}.json"
    )
    md_path = RESULT_DIR / f"simjit_duckdb_fragments_{BUILD_TYPE}_{ROWS}_{timestamp}.md"

    payload = {
        "rows": ROWS,
        "hot_runs": HOT_RUNS,
        "build_type": BUILD_TYPE,
        "duckdb_bin": str(DUCKDB_BIN),
        "extension": str(SIMJIT_EXTENSION),
        "checks": checks,
        "kernel_meta": meta,
        "benchmarks": summary,
        "sales_factorial": sales_factorial,
        "scope_note": "DuckDB-hosted expression fragments representative of analytical queries, not complete query execution.",
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines = [
        "# DuckDB-Hosted Simjit Expression Fragment POC",
        "",
        "DuckDB-hosted expression fragments representative of analytical queries, not complete query execution.",
        "",
        f"- rows: {ROWS}",
        f"- hot runs: {HOT_RUNS}",
        f"- sales factorial hot runs: {SALES_FACTORIAL_HOT_RUNS}",
        f"- build type: {BUILD_TYPE}",
        f"- extension: `{SIMJIT_EXTENSION}`",
        "",
        "## Sales Factorial Design",
        "",
        f"- cells: {len(sales_factorial['cells'])}",
        f"- timed query variants: {sales_factorial['timed_query_variants']}",
        "",
        "| Factor | Levels |",
        "| --- | --- |",
    ]
    for factor, levels in sales_factorial["factors"].items():
        lines.append(f"| {factor} | {', '.join(levels)} |")
    lines.extend(
        [
            "",
            "## Sales Factorial Results",
            "",
            "| Cell | DuckDB us | Simjit cold us | Simjit hot us | Hot speedup | Compile us | Vectorizer | Correct |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in sales_factorial["cells"]:
        speedup = "" if row["hot_speedup"] is None else f"{row['hot_speedup']:.2f}x"
        lines.append(
            f"| {row['name']} | {row['duckdb_sql_us'] or ''} | {row['simjit_cold_us'] or ''} | "
            f"{row['simjit_hot_us'] or ''} | {speedup} | {row['compile_us'] or ''} | "
            f"{row['vectorizer_status'] or ''} | {row['correctness'] or ''} |"
        )
    lines.extend(
        [
            "",
            "## Sales Factor Summary",
            "",
            "| Factor | Level | Cells | Avg hot speedup |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for factor, levels in sales_factorial["grouped"].items():
        for level, row in levels.items():
            avg = (
                ""
                if row["avg_hot_speedup"] is None
                else f"{row['avg_hot_speedup']:.2f}x"
            )
            lines.append(f"| {factor} | {level} | {row['cells']} | {avg} |")
    lines.extend(
        [
            "",
            "## Correctness",
            "",
        ]
    )
    for name in sorted(checks):
        lines.append(f"- {name}: {checks[name]}")
    lines.extend(
        [
            "",
            "## Timings",
            "",
            "| Benchmark | Variant | Median us | Rows/s | Runs |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for bench in sorted(summary):
        for variant in sorted(summary[bench]):
            row = summary[bench][variant]
            rows_per_s = "" if row["rows_per_s"] is None else f"{row['rows_per_s']:.0f}"
            lines.append(
                f"| {bench} | {variant} | {row['median_us']:.0f} | {rows_per_s} | {row['runs']} |"
            )
    lines.extend(
        [
            "",
            "## Kernel Metadata",
            "",
            "| Kernel | Compile us | Vectorizer status |",
            "| --- | ---: | --- |",
        ]
    )
    for kernel in sorted(meta):
        lines.append(
            f"| {kernel} | {meta[kernel].get('compile_us', '')} | {meta[kernel].get('vectorizer_status', '')} |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    return json_path, md_path


def main():
    if BUILD_TYPE not in {"release", "debug"}:
        raise SystemExit(f"BUILD_TYPE must be release or debug, got {BUILD_TYPE}")
    if not DUCKDB_DIR.exists():
        raise SystemExit(f"DuckDB directory does not exist: {DUCKDB_DIR}")

    print("Simjit DuckDB fragment benchmark")
    print(f"  simjit root: {SIMJIT_ROOT}")
    print(f"  duckdb dir:  {DUCKDB_DIR}")
    print(f"  rows:        {ROWS}")
    print(f"  hot runs:    {HOT_RUNS}")
    print(f"  sales hot runs: {SALES_FACTORIAL_HOT_RUNS}")

    build_duckdb()
    if not DUCKDB_BIN.exists() or not SIMJIT_EXTENSION.exists():
        raise SystemExit("DuckDB binary or simjit extension is missing after build")

    sql, sales_cells = create_sql()
    sql_path = RESULT_DIR / "simjit_duckdb_fragments_last.sql"
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    sql_path.write_text(sql)
    print(f"==> Running in-memory DuckDB SQL script: {sql_path}")
    result = run(
        [str(DUCKDB_BIN), "-unsigned", ":memory:"], input=sql, capture_output=True
    )
    raw_path = RESULT_DIR / "simjit_duckdb_fragments_last.out"
    raw_path.write_text(result.stdout + result.stderr)

    timings, checks, meta = parse_duckdb_output(result.stdout + result.stderr)
    failures = {name: status for name, status in checks.items() if status != "ok"}
    if failures:
        print(json.dumps(failures, indent=2), file=sys.stderr)
        raise SystemExit("correctness failures detected")

    summary = summarize_timings(timings)
    sales_factorial = build_sales_factorial_report(summary, checks, meta, sales_cells)
    json_path, md_path = write_reports(summary, checks, meta, sales_factorial)
    print(f"==> JSON report: {json_path}")
    print(f"==> Markdown report: {md_path}")
    print(f"==> Raw DuckDB output: {raw_path}")


if __name__ == "__main__":
    main()
