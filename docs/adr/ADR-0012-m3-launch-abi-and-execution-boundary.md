# ADR-0012: M3 launch ABI and execution boundary

- Status: accepted
- Date: 2026-08-20

## Context

M3 needs one executable kernel without turning the fixed-block frontend into
a general CUDA runtime. The boundary must preserve PyTorch ownership of CUDA
devices, contexts, streams, and tensor storage while keeping the base package
usable without PyTorch or a GPU.

## Decision

M3 exposes one deliberately narrow, keyword-only execution boundary for the
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

## Consequences

- The deterministic PTX compiler is an internal binding used by the runtime,
  not a public `emit_ptx()` API.
- M3 intentionally supports one fixed vector-add ABI. Segmented execution
  remains outside the public runtime.
- Driver diagnostics report the actual CUDA driver version separately from
  the CUDA version used to build PyTorch.
- Cache entries specialize the compiler inputs and target architecture, never
  tensor objects or data pointers.
