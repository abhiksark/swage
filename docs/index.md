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

!!! warning "Fixed-vector release boundary"

    These docs describe the implemented v0.5.2 native-wheel contract.
    Publication and production qualification remain pending; the latest
    released tag is v0.5.1. Public denotes application API, not evidence that
    a pending release has passed its gates. The broader segmented compiler
    remains experimental; private qualification is contributor machinery,
    and planned work has not passed a public gate.

## Public today

- Public `swage` and self-contained private `mlir_swage` in the v0.5.2
  native-wheel implementation, distributed as `swage-compiler`.
- `@swage.jit` capture and compile-only `emit_mlir()` for the restricted
  fixed-block vector-add subset, without a source tree or local LLVM install.
- Keyword-only launch of the canonical fixed vector add on explicitly selected
  CUDA or Native CPU backends.
- `python -m swage.env --json` diagnostics and explicit native/CPU/CUDA health checks.
- Native `swage` MLIR parsing and verification; compiler tools through source builds.

The fixed kernel's semantics are unchanged. CPU and CUDA selection never falls
back to the other backend. See the [runtime support matrix](reference/runtime-environment.md#support-matrix)
for supported wheels, optional PyTorch, and the distinction between admitted
CUDA targets and the A6000/`sm_86` release gate. Native wheels do not expose the
private segmented Python surface.

## Private qualification

- Segmented sum and max through a sequential CPU oracle and one CTA per
  segment on NVIDIA GPUs.
- Stable ragged softmax through the same private CPU and one-CTA GPU
  boundary.
- Canonical identity-sum planning, direct warp and CTA execution, fused
  mixed execution, and split-CTA partial and merge execution.

These paths have tests and recorded qualification evidence. They do not
widen the public Python language or launch contract.

## Experimental

- A private resident identity-sum kernel drains device task queues and
  publishes split completion correctly, but its predeclared A6000 performance
  gate failed. It is neither qualified nor public.

## Planned

- Public segment syntax and public segmented launch.
- Packing several short segments into one warp allocation.
- Split max and split softmax.
- Reusable device queues, qualified persistent scheduling, and broader policy
  selection.

These capability sections are status boundaries, not fallback paths.

<div class="doc-figure" tabindex="0" markdown="1">

![Public, private qualification, experimental, and planned capability lanes](assets/diagrams/capability-boundary.svg)

</div>

*Swage capability status at a glance: public, private qualification, experimental, and planned. [Open the full-size figure](assets/diagrams/capability-boundary.svg).*

## Choose a path

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install a native wheel and run the supported example end to end,
    or build from source against the exact pinned toolchain.

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

Continue with [Installation](getting-started/installation.md). For the
rationale behind a boundary, use the [ADR index](decisions/index.md);
for the exact public call surface, use [swage](reference/swage.md).
