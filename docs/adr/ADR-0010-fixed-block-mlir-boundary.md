# ADR-0010: Fixed-block MLIR boundary

- Status: accepted
- Date: 2026-08-19

## Context

The first Python AST to MLIR slice needs a fixed-block programming boundary.
That boundary must provide a logical coordinate for each program instance
without exposing GPU block or thread identities. It also needs vector and
memory operations, which MLIR already provides through standard dialects.

## Decision

`swage.program_id <axis> -> index` is the only Swage-specific fixed-block
operation. Its `axis` is a nonnegative `i32` attribute, and it is pure. The
result is a logical fixed-block coordinate, never a GPU block or thread ID.

Fixed vector and memory operations use the upstream `vector` and `memref`
dialects. The self-contained `mlir_swage` package embeds and registers their
pinned Python dialect wrappers and C APIs with the existing standard
dialects.

## Consequences

- The frontend has one logical-grid query without adding hardware identifiers
  to semantic Swage IR.
- Fixed-block load, store, and vector operations remain upstream MLIR
  operations rather than duplicating their semantics in the Swage dialect.
- New frontend operations can compose directly with the pinned `memref` and
  `vector` bindings without importing an external MLIR package.
