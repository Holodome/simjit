# Simjit

Simjit is a low-latency JIT compiler for column expressions.

[Website and benchmarks](https://simjit.org)

## Requirements

Required for the C++ library:

- CMake 3.25 or newer
- C++17 compiler
- make
- git submodules checked out

Required for the Python package:

- Python 3.10 or newer
- scikit-build-core build environment
- NumPy or PyArrow when using those array/table adapters

## Install

Simjit installs as a CMake package. The `jit` profile exposes the public C++ API:

```sh
cmake -S . -B build/install-jit \
  -DSIMJIT_ENABLE_INSTALL=ON \
  -DSIMJIT_INSTALL_PROFILE=jit \
  -DCMAKE_INSTALL_PREFIX=/path/to/prefix

cmake --build build/install-jit --target install
````

Consumers can use the installed package through CMake:

```cmake
find_package(simjit 0.1 CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE simjit::simjit)
```

The `toolkit` install profile additionally exposes internal compiler headers.

## Quick Start

C++:

```cpp
#include <simjit/jit.h>

using namespace simjit;
using namespace simjit::jit;
using namespace simjit::types;

static void build_revenue(FunctionBuilder &b) {
    Value price = b.input_arg(F64);
    Value discount = b.input_arg(F64);
    Value revenue = b.mul(price, b.sub(b.f64(1.0), discount));
    b.store(revenue, b.arg(F64));
}

int main() {
    JitContext ctx{};

    auto func =
        vectorized_function<InputArr<F64>, InputArr<F64>, OutputArr<F64>>(
            ctx, "readme-revenue", build_revenue);

    double price[] = {10.0, 20.0, 30.0};
    double discount[] = {0.1, 0.2, 0.0};
    double revenue[3] = {};
    func(3, price, discount, revenue);

    return revenue[0] == 9.0 && revenue[1] == 16.0 && revenue[2] == 30.0 ? 0 : 1;
}
```

Python:

```py
import numpy as np
import simjit as sj

program = sj.query(
    revenue=sj.col("price") * (1.0 - sj.col("discount")),
)

result = sj.run_program(program, {
    "price": np.array([10.0, 20.0, 30.0]),
    "discount": np.array([0.1, 0.2, 0.0]),
})

print(result.revenue)
```

## License

Simjit is distributed under the Zlib license.
See [LICENSE](LICENSE) for the full license text and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for dependency notices.
