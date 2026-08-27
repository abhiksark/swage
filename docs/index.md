<!-- docs/index.md -->

<div class="doc-wordmark" markdown="1">

![Swage wordmark](assets/images/swage-logo.png)

</div>

# Swage

Swage is an experimental Python-embedded MLIR/LLVM GPU compiler. It studies
how one segment-local program can keep its meaning while task derivation
changes with runtime segment lengths. Its public execution boundary is one
canonical fixed vector-add kernel; the wider segment compiler exists as
private qualification machinery or planned work.

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

- Segmented sum and max through a sequential CPU oracle and one CTA per
  segment on NVIDIA GPUs.
- Stable ragged softmax through the same private CPU and one-CTA GPU
  boundary.
- Canonical identity-sum planning, direct warp and CTA execution, fused
  mixed execution, and split-CTA partial and merge execution.

These paths have tests and recorded qualification evidence. They do not
widen the public Python language or launch contract.

## Planned

- Public segment syntax and public segmented launch.
- Packing several short segments into one warp allocation.
- Split max and split softmax.
- Device queues, persistent scheduling, and broader policy selection.

The three lanes are status boundaries, not fallback paths.

<div class="doc-figure" tabindex="0" markdown="1">

![Public, private qualification, and planned capability lanes](assets/diagrams/capability-boundary.svg)

</div>

*Swage capability status at a glance. [Open the full-size figure](assets/diagrams/capability-boundary.svg).*

## Choose a path

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install the package, build the pinned toolchain, and run the
    supported example end to end.

    [Installation](getting-started/installation.md)

-   **User guide**

    ---

    The ideas behind Swage: ragged data, writing and launching kernels,
    and the execution model.

    [Start the guide](user-guide/index.md)

-   **API reference**

    ---

    Exact public contracts for the package, the kernel language, and
    the runtime.

    [Open the reference](reference/index.md)

-   **Internals**

    ---

    The compiler and runtime machinery behind the public surface, with
    its qualification evidence.

    [Read the internals](internals/index.md)

</div>

For the rationale behind a boundary, use the
[ADR index](decisions/index.md). For the exact public call surface, use
[swage](reference/swage.md).
