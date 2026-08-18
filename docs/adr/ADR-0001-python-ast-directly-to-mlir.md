# ADR-0001: Python AST directly to MLIR

- Status: accepted
- Date: 2026-08-18

## Context

A Python-embedded compiler needs a path from `@sw.jit` source to compiler
IR. Some systems introduce an intermediate Python-object IR (dataclasses)
between the AST and the backend IR. That creates two production type
systems, two verifiers, and two optimization stacks that must be kept in
sync.

## Decision

The frontend parses a restricted Python subset (`inspect.getsource` →
`ast.parse`) and constructs Swage MLIR directly through the MLIR Python
bindings and generated operation wrappers. MLIR is the only production IR
between Python and LLVM. Textual MLIR is reserved for debugging, tests, and
reproducers — the JIT path never serializes and reparses IR. Torch FX is
not the kernel frontend. User kernel bodies are never executed as normal
Python.

## Consequences

- One verifier and one optimization stack; source locations flow from
  Python into MLIR diagnostics.
- The frontend depends on the MLIR Python bindings, so the pinned LLVM
  build enables `MLIR_ENABLE_BINDINGS_PYTHON=ON`.
- Frontend work cannot begin until the dialect registers in Python
  (tracked for M2).
