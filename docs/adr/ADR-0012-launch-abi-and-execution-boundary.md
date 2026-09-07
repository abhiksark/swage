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
Persistent reuse is limited to identified clean builds, cached artifacts are
verified before loading, and loaded modules remain scoped to their CUDA
context.

The exact public call surface lives in
[Public Python API](../reference/swage.md). Current validation,
target, zero-work, stream, retention, and cache contracts live in
[Runtime and Environment](../reference/runtime-environment.md).

## Subsequent extension

The original decision above established the CUDA boundary. The current public
API also accepts an explicit `backend="cpu"` selection for the same canonical
fixed vector-add ABI. That path lowers through the Native LLVM target and runs
synchronously through a process-local execution engine; it does not inherit
CUDA stream, context, module-cache, or tensor-retention behavior. The
backend-neutral generated contract and fail-closed backend selection are
recorded in [ADR-0019](ADR-0019-compiler-generated-kernel-contracts.md).

## Consequences

- The deterministic PTX compiler is an internal binding used by the runtime,
  not a public `emit_ptx()` API.
- The public runtime intentionally supports one fixed vector-add ABI.
  Segmented execution remains outside the public runtime.
- Driver diagnostics report the actual CUDA driver version separately from
  the CUDA version used to build PyTorch.
- Cache entries specialize the compiler inputs and target architecture, never
  tensor objects or data pointers.
