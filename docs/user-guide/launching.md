<!-- docs/user-guide/launching.md -->

# Launching Kernels

`launch()` validates its entire host-visible boundary before it reads a
pointer or compiles a line of IR. This page narrates the journey from
the call to the explicitly selected backend; every rule it mentions is
stated exactly once in
[Runtime and Environment](../reference/runtime-environment.md), which
is normative when the two disagree.

## Fail closed before anything else

The launch first accepts only `backend="cuda"` or `backend="cpu"`; CUDA
remains the default for existing calls. It then checks canonical parameter
order, tensor dtype, rank, contiguity, selected-backend device placement, the
`n` bound, the `BLOCK` limit, and the required logical grid. A launch that
fails validation performs no allocation, compilation, or backend call. CPU
failures never consult CUDA, and CUDA failures never consult CPU.

## Zero work returns early

For `n == 0` the required grid is `(0,)`, and the validated launch
returns before compilation, cache access, module loading, or enqueue.
Empty work is a contract, not an accident.

## Specialization and the cache

A launch is compiled per specialization: the normalized source, kernel name,
ABI, compile-time values, backend, artifact format, exact backend target, and
toolchain identity all participate in one key. CPU uses the literal target
`native` and reuses its Native LLVM JIT executable only in the current
process. CUDA uses the device compute capability, reuses loaded functions,
and, with validated clean packaged build identity (or an identified clean
source checkout), also stores verified PTX in a persistent cache so a fresh
process can skip compilation. Malformed packaged metadata disables persistent
reuse, not process-local compilation; see the normative
[identity rules](../reference/runtime-environment.md#native-build-identity).

## Backend execution semantics

CUDA launches enqueue through the CUDA Driver API on the current PyTorch
stream and return immediately. Submitted tensors are retained through
`record_stream()`. CPU launches invoke the already-initialized Native LLVM JIT
entry synchronously and perform no stream operation. Neither branch copies,
casts, changes devices, or falls back. CUDA kernels pin their launch width
with `.reqntid`; CPU executes the admitted element range sequentially.

<div class="doc-figure" tabindex="0" markdown="1">

![CUDA fail-closed validation, current-stream launch, and tensor retention](../assets/diagrams/runtime-lifecycle.svg)

</div>

*The CUDA branch of one launch: validate, specialize, compile or reuse, and
enqueue. [Open the full-size figure](../assets/diagrams/runtime-lifecycle.svg).*

## Seeing what happened

`python -m swage.env --json` reports the environment; add `--check cpu` or
`--check cuda` to fail when the selected prerequisite check is unavailable.
These probes are not a launch or a release-qualification test.
`SWAGE_DUMP_MLIR=1` writes lowered MLIR for either backend;
`SWAGE_DUMP_PTX=1` writes CUDA PTX. `SWAGE_CACHE_DIR` isolates the CUDA-only
persistent cache. The [Quickstart](../getting-started/quickstart.md) walks
these switches end to end.

Continue with [Execution Model](execution-model.md) for how segments
become tasks and tiles, or [Runtime and Environment](../reference/runtime-environment.md)
for the exact rules behind this page.
