# Changelog

All notable changes to Swage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning (`0.x`; anything may change).

## [Unreleased]

### Added

- Installed-wheel `python -m swage.bench vector-add --output PATH [--enforce]`
  entry point, with dependency-free help and the unchanged frozen float32
  vector-add schema, correctness checks, and inclusive A6000 gates.
- Public canonical vector addition and multiplication now support matching
  `float16`, `float8_e4m3fn`, and `float8_e5m2` tensors on CPU and CUDA,
  alongside `float32`. Low-precision arithmetic rounds an FP32 sum or product
  to the storage format; software FP8 conversion works on RTX A6000 `sm_86`.
- Dtype-aware specialization and native warm-launch validation, public
  low-precision signature markers/stubs, and `--dtype` in the vector-add
  example. Numerical coverage exhausts both FP8 formats' encoding pairs
  on CPU/CUDA and checks FP16 encodings, rounding, tails, and dtype switches.
- v0.5.2 fixed-vector release hardening: self-contained Linux x86-64 native
  wheels for regular CPython 3.10–3.13, fixed-contract type stubs, bundled LLVM
  licensing and validated build provenance.
- Public `SwageError`, `CompilationError`, and `BackendUnavailableError`
  hierarchy, structured `swage.env --json --check native|cpu|cuda` diagnostics,
  and opt-in sanitized `swage.runtime` DEBUG events.
- Native artifact, minimum-PyTorch CPU, source/sdist reproducibility, sanitizer,
  CodeQL, dependency-review, and installed A6000 runtime SLO release gates.
  Manual release dispatch produces evidence without attestation or publication.

- Generated TikZ figure atlas covering the GPU execution approaches, and a
  benchmarks page backed by a committed RTX 5090 snapshot.
- Version-2 backend-neutral compiler launch contracts with strict canonical
  JSON validation, CUDA and Native CPU launch models, complete physical scalar
  kinds, typed runtime binding, and retained RTX A6000 host-marshalling gate
  evidence.
- A source-only independent-process Swage/Triton campaign runner with
  interleaved candidate timing, fresh-cache compile-only phase accounting,
  separate per-input planning and warm end-to-end measurements, and a padded
  PyTorch storage baseline. Strict schema-v1 evidence validates native/source
  identity, raw samples, child hashes, aggregates, and NVIDIA telemetry.
  Parameterized chart calculations use only raw child samples; archival
  figures require a successful exclusive campaign. The pinned,
  provenance-checked `soc-Epinions1` outgoing-degree trace is retained.
- Explicit `backend="cuda"` or `backend="cpu"` public launch selection for the
  canonical fixed vector add. The Native CPU path uses a synchronous,
  process-local LLVM JIT executable and never falls back to CUDA.

### Changed

- Native Python test subprocesses retain the active checkout's source package
  path, preventing an unrelated editable install from contaminating
  second-process persistent-cache verification in linked worktrees.

- Replaced the standalone fixed-runtime benchmark script with the installed
  module command. CI and release callers require the wheel-shipped parser and
  benchmark implementation; no compatibility wrapper or segmented selector
  is provided.
- Native wheel builds use pinned scikit-build-core and exact LLVM 22.1.8;
  source distributions and frontend-only editable installs remain CMake-free.
  Private segmented Python modules are excluded from wheels, not source trees.
- Validated packaged build identity now takes precedence over checkout identity
  for persistent CUDA caching. Malformed identity disables persistence without
  preventing process-local compilation.
- The optimizer driver registers the same upstream LLVM conversion
  interfaces as runtime codegen, so complete fixed GPU-to-NVVM pipelines
  do not abort on an unimplemented promised dialect interface.

- The documentation site is restructured into getting started, user guide,
  API reference, internals, and decisions sections; milestone codenames
  moved out of user-facing prose, and every previously published URL
  redirects to its new location.
- The research paper now treats the fixed mixed policy separately from its
  offline oracle, narrows public/private claims, expands related-work
  positioning, records byte-exact frozen artifact pins, distinguishes
  archival headline evidence from current-tree engineering campaigns, and
  replaces the redundant admission figure with a capability table.
- Split partial and merge kernels use 512 threads, sized so one
  4096-element chunk fully occupies a CTA at eight elements per thread.
- Warm CUDA launches dispatch through a compiled nanobind entry point that
  resolves `libcuda.so.1` with `dlopen`; ctypes remains the fallback.
  Canonical fixed warm dispatch now checks live tensor, device, stream, context,
  and cache state natively while retaining full validation on shortcut misses
  and PyTorch allocator stream recording.
- Legacy-default-stream module completion fences are deferred until retirement;
  caller-owned streams retain per-submission fences. Capture launches skip
  retirement polling so cache eviction cannot inject completion events into
  a captured graph.
- Fixed and segmented CUDA kernels share one ordered typed launcher,
  backend-aware artifact cache, and context-keyed module lifecycle; compiler
  contracts replace parameter names and positional pointer/scalar groups as
  physical ABI authority.
- Runtime orchestration is backend-neutral. CPU artifacts remain process-local;
  only verified CUDA PTX with validated clean compiler identity is persisted.
- Segmented semantic functions now contain only values, offsets, and output
  buffers. CPU and planning lowerings derive counts from memref dimensions;
  physical GPU contracts retain required runtime counts as derived bindings.
- Segmented compiler and Python internals are decomposed by admission,
  emission, validation, planning, execution, and oracle responsibilities.
  Process caches are bounded, compilation coalesces per specialization, and
  completed evicted modules unload through nonblocking same-context event
  polling while graph-captured modules remain context-owned.

### Release qualification

The v0.5.2 changes above are unreleased. Publication requires the repaired
four-ABI artifact set, byte-identical cp313 source/sdist wheels, security gates,
and trusted installed-wheel A6000 correctness and performance evidence.
FP32 behavior and the canonical kernel shape are unchanged; the newly
supported storage formats expand the public dtype contract. Existing frozen
FP32 performance evidence does not qualify low-precision performance.
No public segmented execution or persistent segmented performance gate
was changed.

## [0.5.1] - 2026-08-24

### Added

- Private split-CTA execution for the canonical identity segmented sum.
  Oversized segments become ordered partial ranges no larger than 4096
  elements followed by one compact scratch-range merge, using fixed 128-thread
  kernels and the existing CUDA Driver launch boundary.
- Exact and nontrivial f32 qualification on NVIDIA RTX A6000 `sm_86`, covering
  split-only, direct-only, mixed, zero-work, stream, lifetime, device drift,
  and fail-closed compile, allocation, and launch cases.
- Project logo and favicon assets for the README and MkDocs site.
- Read the Docs configuration with pinned documentation dependencies and
  canonical hosted URLs.

### Changed

- `swage_plan.classify` now records `cta_chunk_elements` and validates
  `0 < warp_max_elements <= cta_chunk_elements <= INT32_MAX`. The default CTA
  chunk limit is 4096; the warp and CTA policy set remains unchanged.

### Notes

- Split execution remains a private identity-sum correctness path. It adds no
  public segmented frontend or launch API, split max or softmax, and no change
  to the frozen mixed-policy benchmark or its `1.05` gate.
- Version 0.5.1 is a fix-forward release so tagged documentation and the PyPI
  project description include the current documentation configuration and
  branding. The immutable 0.5.0 release remains unchanged.

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
  stream, with strict fixed vector-add ABI, device, bounds, block, and grid
  validation.
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
  task-range type, and a classify operation. The fail-closed planning
  conversion preserves one canonical identity segmented-sum function while
  adding a private planning companion, and the host classifier emits
  validated stable descriptors.
- Private identity-sum preparation that clones the semantic module, consumes
  its planning threshold, and materializes stable warp and CTA task IDs before
  compiling or allocating GPU work. Pure schedules use 32-thread warp or
  128-thread CTA kernels with the same task-ID ABI.
- One-launch fused mixed execution with four one-segment warp slots per
  initial 128-thread block followed by one block per CTA task. The frozen RTX
  A6000 `sm_86` bimodal benchmark records a `0.939394`
  mixed-to-best-pure ratio, passing the predeclared `1.05` maximum.
- Project documentation (README, DESIGN, ROADMAP, concept docs, ADRs
  0001–0016), community health files, issue forms, and CPU CI.

### Notes

- `emit_mlir()` remains compile-only and direct kernel calls remain
  unavailable. The `launch()` path executes only the canonical
  one-dimensional fixed vector add with f32 pointers and an i32 length.
- Segmented sum and max are native compiler qualification paths only. They do
  not add segment primitives to `swage.language` or widen public
  `kernel.launch()` behavior.
- Ragged softmax is also an internal qualification path. Public segment
  primitives, segmented launch, schedule selection, and multi-CTA execution
  remain planned; `v0.4.0` is eligible but not released.
- Mixed-policy execution remains an internal qualification path for one
  canonical identity segmented sum. Version 0.5.0 adds no public segmented
  launch, packed warps, split CTAs, queues, or persistent scheduling.
- The pip package remains GPU-free at import time. Execution requires Linux,
  CUDA-enabled PyTorch, `libcuda`, and the build-tree `mlir_swage` bindings.
