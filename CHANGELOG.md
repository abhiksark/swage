# Changelog

All notable changes to Swage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning (`0.x`; anything may change).

## [Unreleased]

## [0.5.0] - 2026-08-24

### Added

- Region-based semantic ops `swage.map`, `swage.reduce`
  (kinds `sum`/`max`/`min`), `swage.map_store`, and `swage.yield`, with
  isolated regions, explicit captures, and the verifier set from ADR-0008.
- `swage` MLIR dialect: `!swage.segment<T>` type and `swage.segment_id`,
  `swage.make_segment`, `swage.extent` operations, with verifiers.
- `swage-opt` optimizer driver; lit/FileCheck test suite (`check-swage`).
- Exact LLVM/MLIR pin (`llvmorg-22.1.8`) with out-of-tree CMake build and
  `scripts/fetch_llvm.sh` / `build_llvm.sh` / `build_swage.sh`.
- Python package `swage-compiler` (import `swage`) with
  `python -m swage.env` environment diagnostics.
- Compile-only Python AST-to-MLIR emission for the fixed-block vector-add
  subset. `emit_mlir` accepts either explicit descriptors or a PyTorch
  argument mapping and returns a verified live build-tree
  `mlir_swage.ir.Module` with source locations and diagnostics.
- Optional `swage-compiler[pytorch]` metadata inference for contiguous,
  strided, rank-one f32 CPU or CUDA tensors and signed i32 Python integers.
- Native build-tree `mlir_swage` package with generated `swage` bindings,
  dialect registration, and the `check-swage-python` integration target.
- Bindings-enabled `ci-cpp` coverage that runs the native integration target
  after the lit suite and installs MLIR Python requirements on cold and warm
  LLVM cache paths.
- Deterministic in-process lowering of the canonical fixed vector add through
  GPU and NVVM dialects to exact-target LLVM NVPTX output.
- Keyword-only asynchronous `kernel.launch()` on the current PyTorch CUDA
  stream, with strict M3 ABI, device, bounds, block, and grid validation.
- Lazy `ctypes` CUDA Driver integration for context lookup, module loading,
  function lookup, launch, and stable driver diagnostics without a CUDA
  toolkit or link-time CUDA SDK dependency.
- Digest-validated PTX caching with complete specialization keys, atomic
  user-only writes, process-local reuse for dirty builds, per-context loaded
  module reuse, and opt-in MLIR/PTX dumps.
- Trusted dispatch and weekly GPU qualification for vector-add correctness,
  non-default streams, argument lifetime, and cache reuse. Pull-request code
  never runs on the self-hosted GPU runner.
- Fail-closed native lowering for canonical f32 segmented sum and max. The CPU
  oracle uses sequential `scf`/`memref` loops executed by upstream
  `mlir-runner`; the GPU path uses one CTA per segment and exact-target LLVM
  NVPTX output.
- An internal host-validated segmented-reduction qualification runner with
  explicit values, offsets, output, value-count, and segment-count ABI fields.
  It checks malformed offsets before launch and compares CPU and RTX A6000
  results with PyTorch across empty, boundary, large, uniform, and skewed
  distributions.
- Fail-closed single-consumer map fusion, ordered f32 reduction captures, and
  per-value `swage.map_store` lowering on the sequential CPU and one-CTA GPU
  paths while retaining the five-argument internal segmented ABI.
- Internal stable ragged-softmax qualification through maximum, exponential
  sum, and normalization/store phases. CPU results match PyTorch, and RTX
  A6000 `sm_86` results match both PyTorch and the CPU oracle across six
  adversarial segment distributions.
- Minimal `swage_plan` support for warp and CTA policy attributes, an opaque
  task-range type, and a classify operation. The fail-closed M6 conversion
  preserves one canonical identity segmented-sum function while adding a
  private planning companion, and the host classifier emits validated stable
  descriptors.
- Private M7 identity-sum preparation that clones the semantic module,
  consumes its planning threshold, and materializes stable warp and CTA task
  IDs before compiling or allocating GPU work. Pure schedules use 32-thread
  warp or 128-thread CTA kernels with the same task-ID ABI.
- One-launch fused M7 mixed execution with four one-segment warp slots per
  initial 128-thread block followed by one block per CTA task. The frozen RTX
  A6000 `sm_86` bimodal benchmark records a `0.939394`
  mixed-to-best-pure ratio, passing the predeclared `1.05` maximum.
- Project documentation (README, DESIGN, ROADMAP, concept docs, ADRs
  0001–0016), community health files, issue forms, and CPU CI.

### Notes

- `emit_mlir()` remains compile-only and direct kernel calls remain
  unavailable. The M3 `launch()` path executes only the canonical
  one-dimensional fixed vector add with f32 pointers and an i32 length.
- M4 segmented sum and max are native compiler qualification paths only. They
  do not add segment primitives to `swage.language` or widen public
  `kernel.launch()` behavior.
- M5 ragged softmax is also an internal qualification path. Public segment
  primitives, segmented launch, schedule selection, and multi-CTA execution
  remain planned; `v0.4.0` is eligible but not released.
- M7 mixed-policy execution remains an internal qualification path for one
  canonical identity segmented sum. Version 0.5.0 adds no public segmented
  launch, packed warps, split CTAs, queues, or persistent scheduling.
- The pip package remains GPU-free at import time. Execution requires Linux,
  CUDA-enabled PyTorch, `libcuda`, and the build-tree `mlir_swage` bindings.
