# ADR-0011: PyTorch metadata inference

- Status: accepted
- Date: 2026-08-19

## Context

The compile-only Python frontend needs a convenient way to derive its narrow
runtime signature from the PyTorch values already present in user code. The
base package must remain usable without PyTorch, and metadata inference must
not cross into the M3 runtime boundary.

## Decision

`emit_mlir(arguments=..., constexprs=...)` accepts a mapping whose keys match
the non-`constexpr` kernel parameters. A contiguous, strided, rank-one
`torch.float32` tensor on CPU or CUDA maps to
`sl.pointer(sl.float32)`. A non-boolean Python integer in the signed i32 range
maps to `sl.int32`. All other values fail with a source-located
`CompilationError`.

PyTorch is an optional `swage-compiler[pytorch]` dependency and is imported
only when `arguments=` is selected. The existing `signature=` form remains
available for callers that want explicit descriptors or a PyTorch-free path.
Callers must select exactly one form.

Inference reads only tensor layout, dtype, rank, device type, and contiguity.
It does not read data pointers or contents, infer scalar values from shapes,
retain arguments, compare tensor lengths or devices, cache specializations,
or execute a kernel.

## Consequences

- CPU and CUDA tensors are equivalent metadata providers. Mixed devices are
  accepted because no launch occurs.
- The inferred and explicit forms use the same descriptor validation and
  direct MLIR emitter, so they produce the same module.
- Import and metadata access failures remain inside the frontend diagnostic
  boundary and identify the optional installation extra.
- Data pointers, launch validation, specialization, PTX lowering, and CUDA
  execution remain M3 work.
