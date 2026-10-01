# ADR-0012: Fixed vector-add launch ABI and execution boundary

- Status: accepted
- Date: 2026-08-20

## Context

The first execution boundary needs one executable kernel without turning the
fixed-block frontend into a general CUDA runtime. The boundary must preserve
PyTorch ownership of CUDA devices, contexts, streams, and tensor storage while
keeping the base package usable without PyTorch or a GPU.

## Decision

Swage exposes one deliberately narrow, keyword-only execution boundary for the
canonical one-dimensional fixed vector add. Lowering maps each vector lane to
one GPU x-thread and emits PTX in process through the pinned LLVM NVPTX
backend. Unsupported semantic shapes and ABIs fail before translation.

PyTorch continues to own tensor storage, the active device and context, and
the current stream. The runtime validates the complete host boundary before
compiler or driver work and never copies, casts, synchronizes, switches
devices, creates a CUDA context, or falls back. PyTorch, `mlir_swage`, and
`libcuda` remain lazy dependencies; `emit_mlir()` remains compile-only and
direct kernel calls remain unavailable.

Cache identity covers every compiler input and the exact target architecture.
Persistent reuse is limited to processes that can identify the compiler they
load, by a digest of the frontend sources and the identity of the native
libraries; it does not depend on a git checkout or a clean working tree.
Cached artifacts are verified before loading, and loaded modules remain
scoped to their CUDA context.

The private segmented kernels reload each segment range from device memory
at every launch, while host validation sees only the snapshot taken when the
launch was prepared. Every GPU kernel therefore clamps each range it loads,
as signed i32, so that `0 <= start <= end <= length`, before the range
becomes a loop bound. Here `length` is the element count of the buffer the
range indexes, an i32 that the ABI already carries. Ranges into the values
buffer use the value count: the direct, task-ID, fused, and persistent warp
and CTA paths, and the split and persistent partial ranges. Merge ranges into
scratch use the partial count, in the split merge kernel and in the
persistent merge. The ragged softmax kernel stores one output element per
input element under the same range, so its launch passes the length of the
shorter of the values and output buffers in the value-count slot, which
bounds the store as well as the read. Host validation remains the contract
for a correct result: for validated input every clamp is the identity, and
for a range that changed after validation the kernel produces a bounded
access and a clamped result, not a diagnostic. The sequential CPU lowering is
unchanged.

The clamp bounds ranges only. A segment ID loaded from a task buffer, the
merge ID of a persistent partial, and the output segment of a merge record
are used as loaded, so the offsets read, the completion counter update, and
the output store that they index are not bounded on the device. No ABI that
loads task IDs carries a segment count or an output length to bound them
with, and the persistent kernel does not read the merge count that its ABI
carries. These indices are trusted as validated on the host. The prepared
paths keep them in plan-owned storage, and only the private task-ID
qualification launch takes a caller-owned task buffer. The direct ABI needs
no such bound, because its segment ID is the block index, which the kernel
compares with the segment count.

The exact public call surface lives in
[Public Python API](../reference/swage.md). Current validation,
target, zero-work, stream, retention, and cache contracts live in
[Runtime and Environment](../reference/runtime-environment.md).

## Consequences

- The deterministic PTX compiler is an internal binding used by the runtime,
  not a public `emit_ptx()` API.
- The public runtime intentionally supports one fixed vector-add ABI.
  Segmented execution remains outside the public runtime.
- Driver diagnostics report the actual CUDA driver version separately from
  the CUDA version used to build PyTorch.
- Cache entries specialize the compiler inputs and target architecture, never
  tensor objects or data pointers.
