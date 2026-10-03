# Changelog

All notable changes to Swage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning (`0.x`; anything may change).

## [Unreleased]

### Added

- `swage.segment_reduce` records a gradient for `values` that require grad
  while gradient recording is on, for every kind, both dtypes, and both
  ranks, with second derivatives. The backward of a reduction runs PyTorch
  operations on the device and copies nothing to the host: a sum copies the
  gradient of a segment to its elements, and a mean divides it once by the
  length. ADR-0024 records the decision.
- The gradient of a maximum or a minimum goes to the elements that equal
  the result, in equal shares, with `-0.0` equal to `0.0` and the NaN
  elements sharing when the result is NaN; every other element receives
  exactly `0.0`. On the CPU and on CUDA this differs from
  `torch.segment_reduce` only on a tie with a negative gradient, where
  PyTorch gives every tied element the whole gradient.
- `swage.segment_softmax` records a gradient, `y * (g - s)` with `s` the
  sum of `g * y` over the segment, and second derivatives. That sum, and
  the sum inside every second derivative, run the sum kernel of
  `segment_reduce`, so a backward that needs one is refused under CUDA
  graph capture with a message that names it. The user guide states the
  error bounds of the gradient and of the second derivative.
- The task-ids schedule plans a second kernel of a rank-two program, the
  row-stripe tile: `swage_plan.tasks policy<cta>` with `feature_count`, in
  which the row stripes of a group of up to 32 adjacent columns combine
  across a block through a butterfly and one shared-memory exchange. It
  admits the rank-two softmax with identity task ids. The split schedules
  plan the same tile for the partial and merge kernels of a rank-two
  reduction, with `feature_count` on `swage_plan.partial_tasks` and
  `swage_plan.merge_tasks` and one scratch row per partial task. The
  fused-mixed and persistent schedules refuse rank-two values by name. The
  digest matrix gains 50 pairs; no earlier pair moved (ADR-0023).
- `swage.segment_reduce` takes `kind="min"` and `kind="mean"`. An empty
  segment gives positive infinity for `min` and NaN for `mean`; a NaN
  element gives NaN. A mean is the sum of the same call divided by the
  segment length, bit for bit (ADR-0008 amended).
- `swage.segment_reduce` takes `torch.float64` values and returns float64
  on every static schedule, with sums within `k * eps64 * sum(|x|)`.
  `swage.segment_softmax` refuses float64 because the device has no 64-bit
  `exp2`. Persistent execution stays float32.
- Both public calls take `torch.int64` offsets, validated by their 64-bit
  values and narrowed on the host; the cap of `2**31 - 1` rows and segments
  stays.
- `swage.segment_reduce` takes `[N, D]` values and returns `[S, D]`;
  `swage.segment_softmax` takes `[N, D]` float32 values and normalizes per
  column. ADR-0022 records the wider data model.
- `swage.segment_reduce(values, offsets, kind, *, out=None)` for `"sum"` and
  `"max"`, and `swage.segment_softmax(values, offsets, *, out=None)`: the
  first public segmented calls, over rank-one f32 values and int32 offsets on
  the current CUDA device. They record no gradient, prepare their offsets on
  the host at every call, and are refused under CUDA graph capture. A
  Segmented Calls user-guide page, API reference entries, and
  `examples/segment_reduce.py` document them.
- `python -m swage.compile --target <processor> --output <directory>` writes
  the kernels of `swage.segment_reduce` and `swage.segment_softmax` for one
  NVPTX processor, with a small C runtime library and a manifest, without a
  GPU. `SWAGE_ARTIFACT_DIR` makes a process run the two calls from such a
  directory without `mlir_swage` and with no LLVM in the process; nothing is
  compiled, and a directory that cannot serve a call raises. The owner of an
  artifact directory is not compared with the current user; a directory or
  file with group or other write permission is refused.
  `python -m swage.env` reports the selected artifact. `libSwageRuntime` and
  `swage-c/Runtime.h` (the task classifier and a kernel launcher in plain C)
  are built and installed beside the bindings, for Linux x86-64 only.
- ADR-0021 records artifact format version 1: what the manifest holds, what
  the loader verifies, the trust rule, and what is out of scope.
- `scripts/build_native_wheel.sh` builds a `swage-compiler-native` wheel of
  `mlir_swage` from a checkout; no native wheel is published. The bindings
  record the `swage` version and source revision they were built from,
  `swage` refuses bindings built for another version, and
  `python -m swage.env` prints both.
- Committed records of the segmented-sum campaign at `453c56e` on the RTX
  A6000 (fresh offsets at three sizes and a frozen comparison with looped
  and planned Triton, five processes each), with a summary page and
  documentation fragments generated by `benchmarks/campaign_tables.py`.
- Committed records of the public `swage.segment_reduce` call at `c6099ec`
  on the RTX A6000 (fresh offsets on rank-one values at one size, and on
  `[N, D]` values with D of 3 and 64 at three sizes and with D of 768 at
  one, against `torch.segment_reduce` and looped Triton, five processes
  each), the baseline before a planned change to the rank-two schedule,
  with the PTX of the rank-two column kernel, a summary page, and
  documentation fragments generated by `benchmarks/public_call_tables.py`.
- A compile-only test compares the SHA-256 of the lowered MLIR and of the
  PTX of every compiled kernel (836) with a committed record.
- The planned per-function segmented lowering of ADR-0020: every segmented
  kernel and the CPU oracle are planned and then converted. The plan dialect
  gains `swage_plan.tasks`, `partial_tasks`, `merge_tasks`, `fused_tasks`,
  `persistent_tasks`, `swage_plan.yield`, the `swage_plan.block_threads`
  function attribute, and `#swage_plan.policy<sequential>`. New passes:
  `--swage-plan-to-gpu`, `--swage-plan-to-scf`, and `--swage-fuse-maps`.
  `--swage-to-plan` takes `schedule` (one or a list of `direct`, `task-ids`,
  `fused-mixed`, `split-partial`, `split-merge`, `persistent`,
  `sequential`), `block-threads`, and `function`. The C API gains
  `swageGetTargetDescription` and `swageEstimateElementWork`. No kernel text
  changed: the committed digests of all 552 kernels are unchanged.
- Generated TikZ figure atlas covering the GPU execution approaches, and a
  benchmarks page backed by a committed RTX 5090 snapshot.
- Private persistent task queue for the canonical identity segmented sum
  (ADR-0018). It is experimental: its predeclared performance gate failed and
  the path stays private.
- Private capture-free, single-stage f32 sum and max element programs and map
  chains across warp, CTA, mixed, and split execution (ADR-0019).
- `python -m swage.env` reports `revision`, `llvm_linked`, and whether the
  build-tree bindings import. The native `swage` submodule exposes
  `__llvm_version__`.
- `scripts/fetch_llvm.sh` verifies the pinned LLVM source tarball against a
  committed SHA-256 and accepts a `SWAGE_LLVM_URL` mirror. CMake rejects a
  non-pinned LLVM at configure time.
- Every segmented schedule, including split partial, split merge, and the
  experimental persistent queue, can be run from `swage-opt`, with FileCheck
  tests that pin the synchronization structure of the fused mixed, split,
  and persistent kernels.
- A `power-law` benchmark distribution with an uncapped heavy tail, a looped
  Triton baseline in the comparison harness, and
  `benchmarks/benchmark_fresh_offsets.py`, which times segmented sum on a new
  offsets layout every iteration with preparation inside the timed region.

- A support matrix page, an emit-only example that runs with the native
  build and no GPU or PyTorch, and user-guide sections for the offsets
  contract, layout conversions, and the empty and NaN results of the private
  segmented path.
- `swage-c/Dialects.h` adds a `!swage.segment<T>` constructor and predicate
  and a `swage_plan` registration handle. CMake installs the headers and a
  `find_package(Swage CONFIG)` package.
- `python -m swage.env` reports `swage_file`, `mlir_swage_file`, `target`
  (qualified, admitted, or not admitted), `cache_dir`, `cache`, and
  `compile_on_miss`.
- `SWAGE_CACHE_MAX_ENTRIES` (default 1024) bounds the disk cache;
  `SWAGE_CACHE_READ_ONLY=1` reads entries without changing the cache root;
  `SWAGE_NO_COMPILE=1` raises on a kernel that is not cached. The two
  cache variables govern the public `launch()`, the only path that uses the
  disk cache.
- Benchmarks: a driver repeats a harness in independent processes and
  reports the median and range of per-process medians; records carry native
  library and PTX hashes, CPU, driver, and GPU state; rows report effective
  GB/s and timer resolution; a pad-to-max PyTorch baseline and options for
  10^6 segments, seeds, values, and distributions; the gate scripts can
  rerun on another GPU without producing gate evidence.
- Tests and documentation for the numerics of the private segmented paths:
  bit reproducibility of each sum schedule, the dependence of sum bits on
  the schedule and how to pin it, a sum error bound checked at up to
  1,048,577 elements, sum special values, and the ragged-softmax error as a
  function of logit spread.
- A compile-only test covers every admitted processor, including `sm_87`.
  `ci-cpp` gains a clang-format check and an ASan and UBSan job.
- `THIRD_PARTY_NOTICES.md` lists what a native build contains.
- An opt-in NVIDIA Compute Sanitizer racecheck of the private segmented
  kernels (`SWAGE_RACECHECK=1`); no workflow runs it.

### Changed

- The public segmented calls take `values` that require grad instead of
  raising a `ValueError`. A call that records a gradient refuses `out` with
  a `ValueError`; under `torch.no_grad()` and inside
  `torch.inference_mode()` a call records nothing and takes `out`. An `out`
  that requires grad stays refused.
- `swage.segment_reduce` and `swage.segment_softmax` on `[N, D]` values run
  the row-stripe tile in place of the column tile. A reduction classifies
  the rows of each segment under the default limits divided by the
  column-group width, and splits a segment of more than `4096 / W` rows; a
  softmax launches one task per segment. The bits of a rank-two sum, mean,
  and softmax change, and a sum lies within the `k` of the tile; a maximum
  and a minimum keep their bits. An artifact holds the `cta`, `partial`,
  and `merge` roles of the rank-two programs in place of `column`, so an
  artifact written before is refused (ADR-0021 amended, ADR-0022
  superseded in part by ADR-0023).
- Inside `torch.compile`, each public segmented call is a graph break and
  runs eagerly. Before, rank-two calls with int64 offsets raised
  `AttributeError`, and parts of the host preparation were traced into
  compiled graphs.
- The bindings record a digest of the `swage` sources they were built
  beside. Outside a checkout `swage` refuses bindings built beside other
  sources; in a checkout it refuses differing native sources and warns once
  when only the frontend moved. `python -m swage.compile` refuses bindings
  built beside another frontend.
- The artifact loader refuses a runtime library that needs a newer glibc
  than the host has, before loading it. The deployment page states the
  requirement and gives build commands for AArch64 that have not been run
  there.
- `swage.segment_reduce` validates, classifies, and enqueues in one step. It
  compiles and loads only the kernels its batch launches, creates no CUDA
  event, accepts tensors created under `torch.inference_mode()`, and
  advances the version counter of its result only after an enqueue. Result
  bits are unchanged.
- The segmented calls raise a `RuntimeError` that names numpy when it is
  missing, and the `pytorch` extra declares it.
- A CUDA driver wrapper created while `SWAGE_ARTIFACT_DIR` is set takes its
  launcher from the artifact, so a process that runs the segmented calls
  from an artifact maps no LLVM library even when `mlir_swage` is
  importable.
- `SWAGE_ORACLE_BUILD_DIR` names the build directory of the private CPU
  oracle, for a `swage` that is not imported from a checkout.
- Benchmarks: `swage_public_call` in the fresh-offsets harness times the
  public call itself. No record holds it yet.
- A segment function declares its arguments with `swage.role`, in any order,
  and a module may hold any number of segment functions. A caller of a
  kernel or a symbol clash is diagnosed before a GPU lowering changes the
  module. `swageMaterializeSegmentedPlan` takes a kernel name.
- `swage.map` is specified as a lazy view, and `swage.map_store` declares
  recursive memory effects.
- The code generation C API runs the planner and the conversion as two
  passes; its six compile entry points, their signatures, and their
  diagnostics are unchanged. `swageMaterializeSegmentedPlan` classifies
  through `classifyTaskRecords` and reports an offset outside i32 before it
  classifies.
- `--swage-fixed-block-to-gpu` diagnoses a caller of its kernel or a symbol
  clash before it changes a module. `swage-opt` reports one rule for
  `block-threads`: a launch width the target admits.
- Block widths, claim batches, planning defaults, and admitted processors
  come from one target description read by the lowerings, the C API, and
  the private runner.
- Kernels pass through a curated LLVM pass pipeline (early-cse, instcombine,
  simplifycfg, loop-rotate, licm) before PTX emission. Results are
  bit-identical and each kernel's barrier, shuffle, fence, and atomic counts
  are unchanged; segment loops now close with one conditional branch.
  Committed benchmark records made before this change describe PTX generated
  without these passes.
- The benchmark pages report the regime with changing offsets and the rows
  where looped and planned Triton are faster, and the comparison study sets
  the older record beside the new one.
- Artifacts are format version 2: the role list is per program, the
  manifest records the target description, and the unlaunched warp kernel
  is gone. A version 1 artifact is refused and must be written again.
- The persistent cache key identifies native libraries by ELF build id or
  content digest instead of name, size, and modification time, and no longer
  includes the checkout's LLVM pin. Existing cache entries are not reused.
- `swage-opt` registers upstream dialect extensions, so the lowering
  pipeline of the code generation C API runs from text through the NVVM
  conversion.
- Benchmarks: candidate filters on both harnesses; fresh offsets warms each
  sample with calls of the same candidate and times planned Triton with its
  partition and a single-policy Swage call; a looping matched Triton
  comparator; process ratios against any named candidates and
  re-summarizing of existing records; the CPU frequency governor in every
  record.
- The documentation site is restructured into getting started, user guide,
  API reference, internals, and decisions sections; milestone codenames
  moved out of user-facing prose, and every previously published URL
  redirects to its new location.
- Split partial and merge kernels use 512 threads, sized so one
  4096-element chunk fully occupies a CTA at eight elements per thread.
- Warm launches dispatch through a compiled nanobind entry point that
  resolves `libcuda.so.1` with `dlopen`; ctypes remains the fallback.
- Segmented GPU kernels clamp each segment range loaded from device memory
  to the value count before indexing the values buffer, and each merge
  range to the partial count before indexing scratch. The softmax launch
  bounds its store by the shorter of the values and output buffers.
- Segmented GPU kernels also bound every index they load: a segment ID from
  a task buffer and the output segment of a merge record against the segment
  count, and a persistent merge ID against the merge count. An index outside
  its bound is skipped. The private task-ID, fused, persistent, and split
  merge kernels take a trailing segment count; valid-input results are
  unchanged (ADR-0012).
- The segmented GPU lowering and the private launch helpers reject block
  sizes whose warp count is not a power of two, and the lowering rejects
  `persistent` or `fused-mixed` combined with `use-task-ids`.
- `swage.extent`, `swage.map`, `swage.reduce`, and `swage.map_store` declare
  a memory read on their segment operand. `swage.extent` is no longer `Pure`,
  so upstream CSE and LICM no longer merge or hoist segment readers across
  writes.
- Private segmented kernels are compiled once per kernel, options, and target
  and loaded once per CUDA context. Prepared launches reject an offsets
  tensor modified in place, and validation rejects tensors that require grad.
- CUDA graph capture of a prepared launch needs an earlier launch that
  observed task storage ready: launch, synchronize, and launch again. The
  error message now says so.
- Disk cache entries are keyed on the frontend sources and the loaded native
  library instead of the checkout revision, persist without a clean git
  checkout, and are published atomically. Incomplete entries are recompiled,
  and entries not owned by the current user are rejected. Existing entries
  are not reused.
- The disk cache is used only when every frontend source and native library
  is older than the process, so a process that loaded an earlier compiler
  never publishes under the key of the files now on disk. A cache directory
  that cannot be read or written degrades to process-local reuse with one
  warning per process instead of failing the launch; unsafe and corrupt
  entries still raise.
- The process start time that gates the disk cache is sampled when `swage`
  is imported, so a child forked after import inherits it. `import swage`
  now also loads the private `swage._runtime` module; PyTorch and the
  native bindings are still not loaded at import.
- The private CPU oracle transports exact f32 bit patterns instead of
  six-digit text and takes its tools from the pinned LLVM install. Segmented
  runtime tests use position-dependent exact inputs, exact ownership checks
  on long segments, and randomized float64 accuracy checks for each static
  policy.
- The frontend accepts a docstring as the first statement of a kernel.
- The trusted GPU workflow runs the whole `python/tests/mlir` directory and
  observes disk cache reuse across two processes with the real compiler
  identity. The native binding and CUDA tests use a temporary kernel cache
  unless `SWAGE_CACHE_DIR` is set.
- `scripts/fetch_llvm.sh` unpacks the verified tarball through a temporary
  directory, refuses a leftover unpack directory, and records the verified
  digest in the source tree. An existing tree without that record is
  accepted with a notice.
- The fresh-offsets benchmark record identifies the imported `swage`
  package, the runtime compiler identity, the native libraries, and the
  linked LLVM version. A full run refuses a package from another checkout,
  and candidate order is a recorded seeded permutation per iteration.
- The codename test scans tracked files instead of walking the filesystem.
- `launch()` rejects lazy negation and conjugate views and an output that
  overlaps an input, including in-place use, and requires PyTorch 2.6 with
  `Tensor.record_stream`, all before any kernel is compiled or enqueued.
- Frontend: `emit_mlir()` checks the kernel before importing the native
  package, so a wheel-only install reports a kernel outside the language
  with a source-located `CompilationError`. The language module is matched
  by object under any import name, and `load other=` accepts a leading minus
  sign. Rejected now: parameter defaults, parameter annotations other than
  `constexpr`, return annotations other than `None`, assignment to a name
  bound to the language module, `other=` literals float32 cannot represent,
  and compile-time index arithmetic that leaves signed 64-bit. `sl.load` and
  `sl.store` declare their keywords as required. Several diagnostics name
  the operator, call, literal, or annotation the kernel wrote.
- Loaded CUDA modules are unloaded once no cache entry, prepared launch, or
  captured public launch holds them, and the in-process kernel caches keep
  128 entries. Warm launches no longer take the compile lock, and
  compilation releases the GIL. Prepared launches reject a different CUDA
  context, and a prepared persistent launch rejects an overlapping launch.
- Private segmented preparation validates and classifies offsets from one
  host int32 buffer with no per-element Python work and parses each program
  once per process; the Python plan derivation is now a property test
  against the native classifier. The private path now needs numpy.
- Private segmented preparation classifies each layout from the offsets
  buffer without the module (`swageClassifySegments`), admits a program once
  per pair of planning limits, uploads all task records in one copy, and
  shares the segment ids of the pure policies across preparations.
- The code generation C API states its contract in `swage-c/Codegen.h` and
  reports every failure with a diagnostic. The fixed-block lowering rejects
  scalable vectors and named memory spaces with a diagnostic, and several
  fixed-block and C API diagnostics say what was found.
- CMake refuses in-source builds and declares the NVVM dialect dependency of
  both conversions. The CPU runner lit tests require the `mlir-runner`
  feature.
- The `swage.reduce` description, ADR-0008, and the design invariants state
  that the combining order is unspecified and that an f32 sum depends on the
  schedule within rounding. No op syntax, trait, or verifier changed.
- The README states the release boundary before the install instructions,
  including which segmented calls are public.
- Hosted CI, the release workflow, and the docs build install one
  hash-locked tool set (`requirements-ci.txt`); the build backend is pinned;
  the release workflow runs the pure-Python tier before it builds; the
  `ci-cpp` LLVM cache key covers the build script, tarball digest, and
  runner image.
- `build_llvm.sh` and `build_swage.sh` build the bindings for the `python`
  on `PATH` and fall back to `python3`. The package description states that
  the wheel is pure Python.
- Every segmented lowering diagnostic has a negative lit test and states
  what it found. A rejected region operation is named with the accepted
  list, where it used to say "exponentials must use math.exp2". The lowering
  dispatches on operation classes, and its pass descriptions match what it
  emits. `_materialize_segmented_plan` accepts offsets only as an int32
  buffer. Two unreachable classifier checks are removed.
- The private segmented helpers reject lazy negation and conjugate views and
  honor `SWAGE_NO_COMPILE=1`. `launch()` errors name the kernel and where it
  is defined, and the missing-bindings error points to the installation
  page.
- Prepared private launches are bound to the storage they were prepared
  with and refuse a rebound tensor, and task IDs must not overlap the
  output. Interpreter exit and `os.fork()` wait for a compile in flight.
  `launch()` works on a thread that has not used CUDA. A failed module
  unload is retried and never fails a load.
- `launch()` rejects a tensor that requires grad with a `ValueError`; a
  launch records no gradient, so pass `tensor.detach()`. Every launch,
  public and private, advances the version counter of its output after the
  enqueue, so a backward pass that saved the output raises instead of using
  overwritten values. A replayed CUDA graph does not advance it. `launch()`
  requires `torch.autograd.graph.increment_version`, which PyTorch 2.6
  provides.
- `python -m swage.env` reports `revision` only when the package is
  `python/swage` of a Swage checkout that holds `cmake/llvm-version.txt`. A
  copy vendored inside another repository reports `None`.
- Documentation: the landing page, diagrams, reference, and internals pages
  match the merged runtime, frontend, CI, and benchmark changes;
  `installation.md` lists what the released `0.5.1` wheel lacks.
- Kernel-body integer literals are bounded to signed 64-bit with a
  source-located error, and kernel names that PTX cannot represent are
  rejected at capture and in PTX compilation. Both changes postdate
  `v0.5.1` and were missing from this list.

### Removed

- `swage_plan.classify`, `!swage_plan.task_range`, the planning companion
  function, and the limit options of `--swage-to-plan`. ADR-0014's dialect
  boundary is superseded by ADR-0020.
- `--swage-segmented-reduction-to-scf`, `--swage-segmented-reduction-to-gpu`,
  and `--swage-split-segmented-reduction-to-gpu`, with the
  `MLIRSwageSegmentedReduction` library and its header. Use
  `--swage-to-plan` followed by `--swage-plan-to-gpu` or
  `--swage-plan-to-scf`.

### Fixed

- `--swage-plan-to-gpu` checks plan functions in nested modules before
  converting them, so an invalid one is refused with a diagnostic instead of
  being converted.
- The artifact loader loads the runtime library from the bytes it verified,
  through an anonymous memory file, instead of reopening the file by path.
- The documentation states the signed-zero results of max and min (IEEE-754
  maximum and minimum; `torch.segment_reduce` returns the sign of the first
  zero), when the kernels read int32 offsets, and the design invariants,
  decision records, verification page, support matrix, comparison study,
  and README cost sentence as the code now stands.
- The private CPU oracle returned one unwritten value for a batch with zero
  segments. It now returns an empty result.
- A contiguous negation view passed to `launch()` was read with the
  opposite sign. It is now rejected.
- The documented sum bound of the default `mixed` schedule now covers the
  batches that automatic selection moves to the CTA schedule.
- The PTX arithmetic scan in the numerics tests never examined mixed-case
  modifiers such as `max.NaN.f32` and judged conversions by one of their
  types; it now checks every instruction against the element type of the
  kernel.
- An atomic cache write closed its file descriptor twice when the publish
  failed, which could close a descriptor owned by another thread.
- A prepared static launch on a thread with no CUDA context failed with an
  invalid-context driver error. The launch now makes the prepared device's
  context current.
- Documentation: the README release boundary labels reductions that postdate
  `v0.5.1` and records the failed two-launch mixed gate; installation lists
  the binding prerequisites, the `sm_80` minimum, and the hosted build cost;
  the cache, target-floor, and benchmark-baseline descriptions match the code
  and records.

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
