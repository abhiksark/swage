# Swage

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded GPU compiler built on MLIR and
LLVM. You write a segment-local kernel in Python; Swage lowers it into
fixed-size GPU tile tasks and generates PTX through MLIR, LLVM, and NVPTX.

!!! warning "Pre-alpha"
    The MLIR dialect, pinned LLVM build, and test infrastructure work
    today. No Python kernel compiles or executes yet. The README status
    table is the single source of truth for what works.

Start with:

- [Quickstart](quickstart.md) — build and test on a machine with no GPU
- [Segments, Tasks, and Tiles](concepts/segments-tiles-tasks.md) — the
  three-level model everything else builds on
- [Compiler pipeline](architecture/compiler-pipeline.md) — from
  `@sw.jit` to PTX
- The ADRs — why the architecture is the way it is
