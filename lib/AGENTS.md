# lib/AGENTS.md

Rules for the C++ MLIR components (`include/swage/**`, `lib/**`,
`tools/**`).

- Operations, types, and attributes are defined in ODS/TableGen under
  `include/swage/Dialect/*/IR/`; no hand-written parsers unless the
  declarative format genuinely cannot express the syntax.
- Naming: `Swage_FooOp` defs, `swage.foo` mnemonics, `MLIRSwage*` CMake
  targets. Generated-file includes go through
  `swage/Dialect/Swage/IR/*.inc`.
- No string-based operation dispatch; use generated op classes.
- Every operation with invariants gets a C++ or ODS verifier and both a
  positive (`roundtrip.mlir`) and negative (`invalid.mlir`) lit test.
- Keep canonicalization (within a dialect) separate from conversion
  (between dialects); conversions live under `lib/Conversion/`.
- Use standard dialects (`arith`, `math`, `scf`, `memref`, `vector`) inside
  regions and lowerings — no custom ops for ordinary arithmetic.
- Format with `clang-format` (LLVM style) before committing; keep
  diagnostics actionable (say what was found *and* what was required).
