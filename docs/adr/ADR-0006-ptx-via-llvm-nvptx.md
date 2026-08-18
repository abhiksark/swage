# ADR-0006: PTX generation through LLVM NVPTX

- Status: accepted
- Date: 2026-08-18

## Context

Two common ways to produce NVIDIA GPU binaries from a JIT: generate CUDA
C++ and compile with NVRTC, or lower through MLIR's `gpu`/`nvvm` dialects
to LLVM IR and emit PTX with the LLVM NVPTX backend.

## Decision

The production path emits PTX directly via LLVM NVPTX
(`LLVM_TARGETS_TO_BUILD=Native;NVPTX`); modules are loaded and launched
through the CUDA Driver API on the current PyTorch stream. NVRTC is not a
production dependency. (The master plan retained NVRTC for the prototype
backend; with no prototype — ADR-0005 — NVRTC drops out entirely unless a
future differential backend earns it.)

## Consequences

- No dependency on the CUDA toolkit's compiler at runtime; the driver JITs
  PTX for the resident GPU.
- Kernel metadata (symbol, threads per CTA, dynamic shared memory, target
  arch) must come from Swage's own lowering, since there is no NVRTC
  round trip to inspect.
