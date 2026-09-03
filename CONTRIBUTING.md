# Contributing

## Code Style

- Follow `.clang-format`.
- Use C-style C++: plain data, explicit control flow, and straightforward
  ownership.
- Classes, structs, and enum values use `PascalCase`.
- Functions and variables use `snake_case`.
- Prefer file-scoped functions over class-scoped helpers.
- Prefer file-scoped structures over nested structures.
- Use `private` and `public` when they preserve real invariants. Otherwise,
  prefer simple structures.
- Prefer local definitions in `.cpp` files instead of `.h` files.
- Avoid small functions that only add indirection.

## Testing

Before considering a merge request ready, run the complete validation on
both supported native runner architectures:

1. `SIMJIT_VALIDATE_ARCH=x86 scripts/validate-all` on an AVX-512 x86_64 host.
2. `SIMJIT_VALIDATE_ARCH=arm scripts/validate-all` on an AArch64 host.
