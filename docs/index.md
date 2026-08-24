<!-- docs/index.md -->

<p align="center">
  <img src="assets/images/swage-logo.png" alt="Swage logo" width="720">
</p>

# Swage

Swage is an experimental Python-embedded MLIR/LLVM GPU compiler. Its public
execution boundary is one canonical fixed vector-add kernel. The wider
segment compiler exists as private qualification machinery or planned work,
not as public segmented Python syntax.

!!! warning "Pre-alpha boundary"

    Read status labels literally. Public today is supported application
    surface. Private qualification is tested contributor machinery. Planned
    describes work that has not passed a public gate.

## Public today

- The pure Python `swage` package, distributed as `swage-compiler`.
- `@swage.jit` capture and compile-only `emit_mlir()` for the restricted
  fixed-block vector-add subset, when build-tree native bindings are present.
- Keyword-only CUDA launch for the canonical fixed vector add.
- `python -m swage.env` environment diagnostics.
- Native `swage` MLIR parsing, verification, and registered compiler tools.

The published wheel does not include the native `mlir_swage` package or
compiler build output. Native wheel packaging remains deferred.

## Private qualification

- M4 segmented sum and max through a sequential CPU oracle and one CTA per
  segment on NVIDIA GPUs.
- M5 stable ragged softmax through the same private CPU and one-CTA GPU
  boundary.
- M6 to M8 canonical identity-sum planning, direct warp and CTA execution,
  fused mixed execution, and split-CTA partial and merge execution.

These paths have tests and recorded qualification evidence. They do not widen
the public Python language or launch contract.

## Planned

- Public segment syntax and public segmented launch.
- Packing several short segments into one warp allocation.
- Split max and split softmax.
- Device queues, persistent scheduling, and broader policy selection.

## Choose a path

| Goal | Start here |
|---|---|
| Install or run the supported example | [Installation](getting-started/installation.md), then [Quickstart](quickstart.md) |
| Learn the segment model | [Swage, Visually](concepts/swage-visual-guide.md), then [Segments, Tasks, and Tiles](concepts/segments-tiles-tasks.md) |
| Contribute to the compiler | [Compiler Pipeline](architecture/compiler-pipeline.md), [Compiler Tools and Passes](reference/compiler-tools.md), and [Contributing](https://github.com/abhiksark/swage/blob/main/CONTRIBUTING.md) |
| Audit qualification claims | [Private M4 to M8 Qualification](qualification/private-m4-m8.md) and [Verification Evidence](qualification/evidence.md) |

For exact public call contracts, use [Public Python API](reference/public-python-api.md).
For the rationale behind a boundary, use the [ADR Index](decisions/index.md).
