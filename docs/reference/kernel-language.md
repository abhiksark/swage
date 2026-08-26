<!-- docs/reference/kernel-language.md -->

# Kernel Language

The current Python frontend accepts one restricted AST shape for fixed-block
vector add. This page lists that syntax. Anything not listed fails closed with
a source-located `CompilationError`.

## Function shape

- One `def` captured by `@swage.jit` or `@jit`.
- No stacked decorators.
- Ordinary positional parameters only. Positional-only, keyword-only,
  variadic positional, and variadic keyword parameters are rejected.
- Compile-time parameters use the exact annotation `sl.constexpr`.
- A final empty `return` is optional. Return values and an earlier return are
  rejected.

Kernel bodies are parsed from source with `inspect.getsource`,
`textwrap.dedent`, and `ast.parse`. They do not execute as ordinary Python.

## Statements

The body accepts:

- single-name assignment, such as `offsets = ...`;
- `sl.store(...)` as an expression statement;
- an optional final bare `return`.

Attribute targets, tuple unpacking, control flow, loops, comprehensions, and
other statement forms are unsupported.

## Expressions and operators

The accepted expression forms are:

- a bound name;
- an integer literal that fits signed 64-bit;
- `+` for index arithmetic, pointer plus offset vector, or two f32 vectors;
- `*` for index arithmetic;
- one signed less-than comparison between an index-offset vector and an i32
  or index value;
- one of the symbolic calls below.

Pointer values support only addition with an offset vector. The frontend does
not accept subtraction, division, boolean operators, chained comparisons,
attribute access as a value, arbitrary calls, or Python control flow.

## Symbolic calls

```python
sl.program_id(axis_literal)
sl.arange(0, BLOCK)
sl.load(pointer + offsets, mask=mask, other=numeric_literal)
sl.store(pointer + offsets, value, mask=mask)
```

`program_id` requires one nonnegative integer literal that fits signed i32.
The public launch subset later requires axis zero. `arange` accepts only the
literal start `0` and the compile-time name `BLOCK`. `load` requires both
named arguments, and `store` requires its named mask. Keyword expansion and
duplicate keyword arguments are rejected.

`BLOCK` must be a positive signed 64-bit integer when present. Other
`sl.constexpr` values must be signed 64-bit integers. Vector operations
require `BLOCK`.

## Supported value categories

The emitter tracks only i32 scalar parameters, index scalars, index vectors,
boolean vectors, f32 vectors, f32 pointer descriptors, and transient pointer
plus offset addresses. It emits standard `arith`, `func`, `memref`, and
`vector` operations around the logical `swage.program_id` operation.

This language is the public M2 and M3 fixed-block subset. The segment
operations present in native MLIR are not exposed as Python language symbols.
Continue with [Public Python API](public-python-api.md) for input modes and
launch validation, or [Swage Dialect](swage-dialect.md) for the native
semantic IR boundary.
