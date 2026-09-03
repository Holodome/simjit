# AGENTS.md

## Python Workflow

Use the repository launcher for Python scripts. It selects the Python interpreter, adds the compiled extension build directory when present, and sets:

```text
PYTHONPATH=build/python-dev:python:python/src:python/tests
```

Run `make py` before scripts that need the compiled extension. The launcher looks for `build/python-dev/simjit/_simjit*.so` automatically.

Useful commands:

```sh
scripts/py -m pytest -q python/tests/test.py
scripts/py benchmarks/python/bench.py
scripts/py --debug-env -m pytest -q python/tests/test.py
make py-test
make py-e2e
scripts/dev python-bench
```

## Runtime Notes

`simjit.run_program` uses the native unresolved DSL path by default. The native path passes the `ir.py` expression graph, input objects, and requested output kind to the C++ extension, where input normalization, type resolution, output allocation, lowering, and compilation happen in one call.

The native resolver defines runtime semantics.

Use `Session.prepare_program(outputs, inputs, output)` when repeated benchmark or service calls should reuse the same native buffers and compiled function.
