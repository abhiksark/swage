# ADR-0012: M3 launch ABI and execution boundary

- Status: accepted
- Date: 2026-08-20

## Context

M3 needs one executable kernel without turning the fixed-block frontend into
a general CUDA runtime. The boundary must preserve PyTorch ownership of CUDA
devices, contexts, streams, and tensor storage while keeping the base package
usable without PyTorch or a GPU.

## Decision

The M3 backend accepts only the canonical one-dimensional fixed vector add:
axis-zero `swage.program_id`, one rank-one vector whose width equals `BLOCK`,
three f32 pointers, and one i32 `n`. Lowering maps each vector lane to one GPU
x-thread and produces a kernel ABI of three raw device pointers followed by
`n` as i32. Unsupported operations, shapes, axes, and ABI types fail before
NVPTX translation. The target is the active device's exact `sm_*` compute
capability. PTX is emitted in-process through the pinned LLVM NVPTX backend,
without NVRTC, subprocesses, textual round-tripping, or a CUDA toolkit.

Execution is exposed only through keyword-only
`kernel.launch(arguments=..., constexprs=..., grid=...)`. It is asynchronous,
returns `None`, and launches on the current PyTorch CUDA stream. Launch
requires contiguous rank-one f32 CUDA tensors on the same current device,
`0 <= n <= tensor.numel()` for every pointer, a device-legal `BLOCK`, and
`grid == (ceildiv(n, BLOCK),)`. `n == 0` with `grid=(0,)` is a validated
no-op.

The runtime never copies, casts, synchronizes, switches devices, creates a
CUDA context, or falls back. It records the current stream on every tensor
after launch. PyTorch, `mlir_swage`, and `libcuda` remain lazy dependencies;
`emit_mlir()` remains compile-only and direct kernel calls remain unavailable.

PTX cache entries are keyed by normalized kernel source, kernel name, ordered
ABI descriptors, constexpr values, target architecture, code-generation
options, Swage revision, dialect version, and LLVM version. Persistent reuse
is disabled for dirty or unidentified builds. Verified PTX is cached on disk,
and loaded modules are cached in-process per CUDA context. Writes are atomic
and user-only; symlinked, world-writable, incomplete, or digest-mismatched
entries are rejected.

## Consequences

- The deterministic PTX compiler is an internal binding used by the runtime,
  not a public `emit_ptx()` API.
- M3 intentionally supports one fixed vector-add ABI. Segment lowering and a
  general kernel ABI remain later milestones.
- Driver diagnostics report the actual CUDA driver version separately from
  the CUDA version used to build PyTorch.
- Cache entries specialize the compiler inputs and target architecture, never
  tensor objects or data pointers.
