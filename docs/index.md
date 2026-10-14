<!-- docs/index.md -->

<div class="doc-wordmark" markdown="1">

![Swage wordmark](assets/images/swage-logo.png)

</div>

# Swage

Swage is an experimental Python-embedded MLIR/LLVM GPU compiler. It studies
how one segment-local program can keep its meaning while task derivation
changes with runtime segment lengths. Its public surface is narrow: the
canonical fixed vector add and multiply on an explicitly selected CPU or
CUDA backend, and two segmented calls, `swage.segment_reduce` and
`swage.segment_softmax`, which run fixed programs over segments on one CUDA
device. The wider segment compiler is private research or planned work.

!!! warning "Release boundary"

    These pages describe the `v0.5.2` source tree and its native-wheel
    contract. That release is not published: publication and production
    qualification are pending, and the latest released tag is `v0.5.1`, a
    pure Python wheel that lacks most of what these pages describe.
    [Installation](getting-started/installation.md) lists the differences.
    Public denotes application API, not evidence that a pending release has
    passed its gates. Private qualification is tested contributor
    machinery, and planned work has not passed a public gate.

## Public today

- Public `swage` and the self-contained private `mlir_swage` bindings in one
  native wheel per CPython version, distributed as `swage-compiler`.
- `@swage.jit` capture and compile-only `emit_mlir()` for the restricted
  fixed-block elementwise subset, without a source tree or a local LLVM
  install.
- Keyword-only launch of the canonical fixed vector add or multiply over
  `float32`, `float16`, `float8_e4m3fn`, or `float8_e5m2` tensors, on an
  explicitly selected CUDA or Native CPU backend.
- `swage.segment_reduce` for `"sum"`, `"max"`, `"min"`, and `"mean"` over
  `float32` or `float64` values, and `swage.segment_softmax` over `float32`
  values, with `int32` or `int64` offsets on the current CUDA device. Both
  take values of rank one or `[N, D]` rows, and both record a gradient,
  with second derivatives, for values that require grad.
  [Segmented Calls](user-guide/segmented-calls.md) states the contract and
  the cost.
- `python -m swage.compile`, which writes the kernels of the two segmented
  calls ahead of time without a GPU, and `SWAGE_ARTIFACT_DIR`, which makes a
  process run the calls from such a directory with no compiler loaded.
- `python -m swage.env --json` diagnostics and explicit native, CPU, and
  CUDA health checks.
- `python -m swage.bench vector-add --output result.json` from the installed
  wheel, for the [frozen CUDA vector-add benchmark](reference/benchmarking.md);
  it is not independent release qualification.
- Native `swage` MLIR parsing and verification; compiler tools through
  source builds.

The two segmented calls take values of rank one or `[N, D]` rows that they
reduce or normalize per column. They record a gradient for values that
require grad, are refused under CUDA graph capture, and prepare their
offsets on the host at every call.
[Segmented Calls](user-guide/segmented-calls.md) states the contract and
the cost, and [Running Without the Compiler](user-guide/deployment.md)
states what an artifact delivers and what it does not.

The semantics of the fixed kernel are unchanged by native packaging. CPU and
CUDA selection never falls back to the other backend. The
[Support Matrix](reference/support-matrix.md) lists the supported wheels,
the optional PyTorch, and the difference between admitted CUDA targets and
the A6000 (`sm_86`) release gate.

## Private qualification

Everything below is private research: it is not public API, and it may
change or go away.

- Segmented sum, max, and min through a sequential CPU oracle and one CTA
  per segment on NVIDIA GPUs.
- Stable ragged softmax through the same private CPU and one-CTA GPU
  boundary.
- Planning, direct warp and CTA execution, fused mixed execution, and
  split-CTA partial and merge execution for capture-free, single-stage sum,
  max, and min programs over f32 or f64 values, with prepared launches and
  schedules chosen by hand.
- A persistent task queue for the identity sum, which drains device task
  queues and publishes split completion correctly, but whose predeclared
  A6000 performance gate failed.

These paths have tests and recorded qualification evidence. The two
segmented calls run a fixed sum, max, min, mean, and softmax through them
with default limits. Other programs, the prepared launches, the scheduling
policies, and the planning limits stay private, and none of it widens the
public kernel language or the `launch()` contract.

## Planned

- Public segment syntax, and public launch of a segment program that the
  caller writes.
- Packing several short segments into one warp allocation.
- Split softmax.
- Reusable device queues, qualified persistent scheduling, and broader
  policy selection.

These capability sections are status boundaries, not fallback paths.

<div class="doc-figure" tabindex="0" markdown="1">

![Public, private qualification, and planned capability lanes](assets/diagrams/capability-boundary.svg)

</div>

*Swage capability status at a glance. [Open the full-size figure](assets/diagrams/capability-boundary.svg).*

## Choose a path

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install a native wheel and run the supported examples end to end,
    or build from source against the exact pinned toolchain.

    [Installation](getting-started/installation.md)

-   **User guide**

    ---

    The ideas behind Swage: ragged data, the segmented calls, writing and
    launching kernels, and the execution model.

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
