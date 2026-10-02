<!-- docs/user-guide/launching.md -->

# Launching Kernels

`launch()` validates its entire host-visible boundary before it reads a
pointer or compiles a line of IR. This page narrates the journey from
the call to the GPU; every rule it mentions is stated exactly once, in
[Runtime and Environment](../reference/runtime-environment.md), which
is normative when the two disagree.

## Fail closed before anything else

The launch first requires PyTorch 2.6 or newer with
`torch.Tensor.record_stream`, the method that retains submitted tensors.
It then checks each of the following before anything else happens:

- the canonical parameter names and order;
- tensor dtype, rank, contiguity, and device placement;
- that no tensor is a lazy negation or conjugate view, because the kernel
  reads the storage of the base tensor;
- that the output shares no memory with either input, which also rules out
  in-place use;
- that no tensor requires grad, because a launch records no gradient;
- the `n` bound, the `BLOCK` limit, and the required grid.

A launch that fails any of these performs no allocation, no compilation,
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
NVPTX; later launches reuse the loaded function. The process keeps a
bounded number of compiled and loaded kernels, and the runtime page states
the bound and when a loaded module is unloaded. When the process can
identify its frontend sources and native compiler libraries, which does
not require a clean git checkout, the compiled artifact also lands in a
verified persistent cache, so a fresh process skips compilation
entirely. The key composition and the verify-or-reject cache path are
drawn on the runtime page.

## Asynchronous by design

Admitted launches enqueue through the CUDA Driver API on the current
PyTorch stream and return immediately. Submitted tensors are retained
through `record_stream()`, storage stays owned by PyTorch, and the launch
does not copy, cast, or fall back. A launch of a kernel that is already
loaded does not synchronize. A launch that loads a kernel may synchronize
the context once, to unload modules that nothing holds any more. Emitted
kernels pin their launch width with `.reqntid`, so a geometry mismatch
fails at the driver instead of running with the wrong geometry.

<div class="doc-figure" tabindex="0" markdown="1">

![Fail-closed validation, current-stream launch, and tensor retention](../assets/diagrams/runtime-lifecycle.svg)

</div>

*The journey of one launch: validate, specialize, compile or reuse, and
enqueue. [Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

## Seeing what happened

`python -m swage.env` reports the environment the runtime saw, including
the standing of the device target and the state of the persistent cache.
Setting `SWAGE_DUMP_MLIR=1` and `SWAGE_DUMP_PTX=1` writes the lowered MLIR
and emitted PTX per specialization, and `SWAGE_CACHE_DIR` isolates the
persistent cache; the [Quickstart](../getting-started/quickstart.md)
walks these switches end to end.

Continue with [Execution Model](execution-model.md) for how segments
become tasks and tiles, or [Runtime and Environment](../reference/runtime-environment.md)
for the exact rules behind this page.
