# Simjit

## Real-world example: nulls, conditionals, and aggregates

```python
import simjit as sj
import pyarrow as pa

qty = sj.coalesce(sj.col("qty"), 0)
unit_price = sj.nullif(sj.col("unit_price"), 0)
discount_bp = sj.coalesce(sj.col("discount_bp"), 0)

gross = sj.i64(qty) * sj.i64(unit_price)
net = gross * (10000 - sj.i64(discount_bp)) // 10000

is_valid = sj.is_not_null(unit_price) & (qty > 0)
is_late = sj.col("ship_delay_days") > 30

program = sj.query(
    net=net,
    is_valid=is_valid,
    total_net=net.sum(where=is_valid),
    late_net=net.sum(where=is_valid & is_late),
)

inputs = pa.table({
    "qty": [2, 0, 4],
    "unit_price": [100, None, 250],
    "discount_bp": [500, 0, 1000],
    "ship_delay_days": [5, 40, 35],
})

result = sj.run_program(program, inputs)
```

## Install

Build a release wheel from the repository root:

```bash
make py-release
```

Install the generated wheel:

```bash
pip install ./build/python-dist/simjit-*.whl
```

## Build

Build the local extension:

```bash
make py
```

LLVM IR and C++ inspection emitters are opt-in:

```bash
PY_LLVM=1 PY_CPP=1 make py
PY_LLVM=1 PY_CPP=1 make py-release
```

For reproducible Linux wheel artifacts:

```bash
docker buildx build \
  --platform=linux/amd64 \
  --file python/Dockerfile \
  --target export-wheel \
  --output type=local,dest=./build/python-dist \
  .
```
