<!-- docs/user-guide/launching.md -->

# Launching Kernels

`launch()` validates its entire host-visible boundary before it reads a
pointer or compiles a line of IR. This page narrates the journey from the
call to the explicitly selected backend; every rule it mentions is stated
exactly once in
[Runtime and Environment](../reference/runtime-environment.md), which is
normative when the two disagree.

## Fail closed before anything else

The launch accepts only `backend="cuda"` or `backend="cpu"`; CUDA remains
the default for existing calls. It requires PyTorch 2.6 or newer with
`torch.Tensor.record_stream`, the method that retains submitted tensors,
and `torch.autograd.graph.increment_version`, which marks the output as
written. It then checks each of the following before anything else
happens:

- the canonical parameter names and order;
- the tensor dtype, which is `float32`, `float16`, `float8_e4m3fn`, or
  `float8_e5m2` and the same for all three tensors, and the rank,
  contiguity, and placement on the device of the selected backend;
- that no tensor is a lazy negation or conjugate view, because the kernel
  reads the storage of the base tensor;
- that the output is exactly one of the inputs or shares no memory with
  either input in the elements the launch covers; any partial overlap is
  rejected, because one lane could store over an element another lane
  still has to read;
- that no tensor requires grad, because a launch records no gradient;
- the `n` bound, the `BLOCK` limit, and the required grid.

A launch that fails any of these performs no allocation, no compilation,
and no backend call. CPU failures never consult CUDA, and CUDA failures
never consult CPU. This mirrors capture: the public surface refuses early
instead of failing late.

## Zero work returns early

For `n == 0` the required grid is `(0,)`, and the validated launch
returns before compilation, cache access, module loading, or enqueue.
Empty work is a contract, not an accident.

## Specialization and the cache

A launch is compiled per specialization: the normalized source, the kernel
name, the ABI, the compile-time values, the dtype, the backend, the exact
backend target, and the toolchain identity all participate in one key. CPU
uses the literal target `native` and reuses its Native LLVM JIT executable
only in the current process. CUDA uses the compute capability of the
device. The process keeps a bounded number of compiled and loaded kernels,
and the runtime page states the bound and when a loaded module is unloaded.
When the process can identify its frontend sources and native compiler
libraries, which does not require a clean git checkout, verified CUDA PTX
also lands in a persistent cache, so a fresh process skips compilation. The
key composition and the verify-or-reject cache path are drawn on the
runtime page.

## Backend execution semantics

CUDA launches enqueue through the CUDA Driver API on the current PyTorch
stream and return immediately. Submitted tensors are retained through
`record_stream()`, storage stays owned by PyTorch, and a launch does not
synchronize. Each launch holds a lease on its loaded module; a module that
leaves the bounded cache is unloaded only after no lease holds it, no CUDA
graph captured it, and an event recorded on every stream that launched it
has completed. Emitted CUDA kernels pin their launch width with `.reqntid`,
so a geometry mismatch fails at the driver instead of running with the
wrong geometry.

CPU launches invoke the already-initialized Native LLVM JIT entry
synchronously and perform no stream operation; the admitted element range
runs sequentially. On both backends, the launch advances the version
counter of the output after the work is enqueued or run. Neither branch
copies, casts, changes devices, or falls back.

<div class="doc-figure" tabindex="0" markdown="1">

![CUDA fail-closed validation, current-stream launch, and tensor retention](../assets/diagrams/runtime-lifecycle.svg)

</div>

*The CUDA branch of one launch: validate, specialize, compile or reuse, and
enqueue. [Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

## Seeing what happened

`python -m swage.env --json` reports the environment the runtime saw,
including the standing of the CUDA device target and the state of the
persistent cache; add `--check cpu` or `--check cuda` to fail when the
selected backend is unavailable. These probes are not a launch or a
release-qualification test. `SWAGE_DUMP_MLIR=1` writes the lowered MLIR for
either backend, `SWAGE_DUMP_PTX=1` writes CUDA PTX, and `SWAGE_CACHE_DIR`
isolates the CUDA persistent cache. The
[Quickstart](../getting-started/quickstart.md) walks through these switches
end to end.

Continue with [Execution Model](execution-model.md) for how segments
become tasks and tiles, or [Runtime and Environment](../reference/runtime-environment.md)
for the exact rules behind this page.
