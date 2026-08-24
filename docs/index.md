<!-- docs/index.md -->

<p align="center">
  <img src="assets/images/swage-logo.png" alt="Swage logo" width="720">
</p>

# Swage

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded GPU compiler built on MLIR and
LLVM. You write a segment-local kernel in Python; Swage is designed to lower
it into fixed-size GPU tile tasks and generate PTX through MLIR, LLVM, and
NVPTX.

!!! warning "Pre-alpha"
    The MLIR dialect, pinned LLVM build, Python AST emitter, M3 fixed
    vector-add execution path, and internal M4–M8 segment qualification paths
    work today. Public segment syntax and launch plus general schedule
    selection remain planned. The README status table is the single source of
    truth for what works.

Start with:

- [Quickstart](quickstart.md) - build without a GPU or run the M3 CUDA
  walkthrough
- [Segments, Tasks, and Tiles](concepts/segments-tiles-tasks.md) - the
  three-level model everything else builds on
- [Compiler pipeline](architecture/compiler-pipeline.md) - from
  `@sw.jit` to PTX and launch
- The ADRs explain why the architecture is structured this way.
