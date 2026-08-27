<!-- docs/user-guide/launching.md -->

# Launching Kernels

`launch()` validates its entire host-visible boundary before it reads a
pointer or compiles a line of IR. This page narrates the journey from
the call to the GPU; every rule it mentions is stated exactly once, in
[Runtime and Environment](../reference/runtime-environment.md), which
is normative when the two disagree.

## Fail closed before anything else

The launch checks the canonical parameter names and order, tensor
dtype, rank, contiguity, and device placement, the `n` bound, the
`BLOCK` limit, and the required grid before anything else happens. A
launch that fails validation performs no allocation, no compilation,
and no driver call. This mirrors capture: the public surface refuses
early instead of failing late.

## Zero work returns early

For `n == 0` the required grid is `(0,)`, and the validated launch
returns before compilation, cache access, module loading, or enqueue.
Empty work is a contract, not an accident.

## Specialization and the cache

A launch is compiled per specialization: the normalized source, the
kernel name, the ABI, the compile-time values, the exact compute
capability, and the toolchain identity all participate in one key. The
first launch of a specialization compiles in process through LLVM
NVPTX; later launches reuse the loaded function. On an identified clean
checkout the compiled artifact also lands in a verified persistent
cache, so a fresh process skips compilation entirely. The key
composition and the verify-or-reject cache path are drawn on the
runtime page.

## Asynchronous by design

Admitted launches enqueue through the CUDA Driver API on the current
PyTorch stream and return immediately. Submitted tensors are retained
through `record_stream()`, storage stays owned by PyTorch, and nothing
synchronizes, copies, casts, or falls back behind your back. Emitted
kernels pin their launch width with `.reqntid`, so a geometry mismatch
fails at the driver instead of running wrong.

<div class="doc-figure" tabindex="0" markdown="1">

![Fail-closed validation, current-stream launch, and tensor retention](../assets/diagrams/runtime-lifecycle.svg)

</div>

*The journey of one launch: validate, specialize, compile or reuse, and
enqueue. [Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

## Seeing what happened

`python -m swage.env` reports the environment the runtime saw. Setting
`SWAGE_DUMP_MLIR=1` and `SWAGE_DUMP_PTX=1` writes the lowered MLIR and
emitted PTX per specialization, and `SWAGE_CACHE_DIR` isolates the
persistent cache; the [Quickstart](../getting-started/quickstart.md)
walks these switches end to end.

Continue with [Execution Model](execution-model.md) for how segments
become tasks and tiles, or [Runtime and Environment](../reference/runtime-environment.md)
for the exact rules behind this page.
