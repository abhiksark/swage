<!-- docs/internals/verification.md -->

# Verification

Status claims are grounded in executable tests and committed artifacts. This
matrix identifies the smallest evidence source for each current boundary and
does not turn private qualification into public API. The two public
segmented calls have rows of their own; the private rows below them cover
the paths the calls run through.

<div class="doc-figure" tabindex="0" markdown="1">

![One semantic module executed on the GPU path, the CPU oracle, and PyTorch, feeding a differential comparison](../assets/figures/oracle-topology.svg)

</div>

*The comparison topology behind every correctness claim. [Open the full-size figure](../assets/figures/oracle-topology.svg).*

| Boundary | Status | Primary evidence | Applicable command |
|---|---|---|---|
| Source distribution: release metadata, the CMake build resources, the exclusion of generated and checkout files, a byte-identical rebuild, and one pinned version of the build backend and the bindings | Release gate; 0.5.2 is unreleased | `tests/python/test_packaging.py` | `python -m pytest tests/python/test_packaging.py -q` |
| Environment report | Public today | `tests/python/test_env.py` | `python -m pytest tests/python/test_env.py -q` |
| Build identity of the bindings, the check that pairs `swage` with them, and the content identity of the native libraries in the cache key | Public today | `tests/python/test_native_identity.py`, `python/tests/mlir/test_native_identity.py` | `python -m pytest tests/python/test_native_identity.py -q`; `ninja -C build check-swage-python` |
| One native wheel per CPython 3.10 to 3.13: contents including the segmented modules, self-contained and relocatable libraries, license files, build identity, size, and reproducibility, and the refusal of a wheel that fails any of them | Release gate; 0.5.2 is unreleased | `tests/python/test_native_wheel.py`, `scripts/check_native_wheel.py`, `scripts/repair_native_wheel.py`, the `wheels` and `aggregate` jobs of `publish-pypi.yml` | `python -m pytest tests/python/test_native_wheel.py -q` |
| The local cp313 native wheel installed outside the checkout: CPU health, the Native CPU runtime suites, the CPU smoke, and the native suite with every CUDA test skipped | Hosted gate | The wheel steps of the `build-and-test` job of `ci-cpp.yml`, `scripts/smoke_installed_wheel.py` | The `build-and-test` job of `ci-cpp.yml` |
| Restricted AST to verified native module, and the kernel-language check without the native package | Public today, compile-only | `tests/python/test_frontend.py`, `python/tests/mlir/test_frontend.py` | `python -m pytest tests/python -q`; `ninja -C build check-swage-python` |
| Emit-only example without a GPU or PyTorch | Public today, compile-only | `python/tests/mlir/test_examples.py` | `ninja -C build check-swage-python` |
| Native `swage` dialect parsing, verification, and declared segment-read effects | Public today, compile-only | `test/Dialect/Swage`, including `effects.mlir` | `ninja -C build check-swage` |
| Fixed vector add and multiply: lowering for Native CPU and CUDA, launch on both backends over float32, float16, and both FP8 formats, and dispatch between the two operations | Public today | The `fixed-vector-*` files under `test/Conversion/SwageToCPU`, the `fixed-*` and `invalid-fixed-*` files under `test/Conversion/SwageToGPU`, `python/tests/mlir/test_cpu_runtime.py`, `python/tests/mlir/test_low_precision_runtime.py`, `python/tests/mlir/test_operation_dispatch.py`, `python/tests/mlir/test_runtime.py`, `python/tests/mlir/test_cache_process_reuse.py` | `ninja -C build check-swage`; `ninja -C build check-swage-python`; trusted GPU workflow |
| Launch validation, the PyTorch floor, and the cache bound, read-only mode, and no-compile mode | Public today | `tests/python/test_runtime.py`, `python/tests/mlir/test_runtime.py` | `python -m pytest tests/python/test_runtime.py -q`; trusted GPU workflow |
| Public segmented calls: names and signatures, the checks that need no native build, the PyTorch floor, and the wheel-only error | Public today | `tests/python/test_segments.py` | `python -m pytest tests/python/test_segments.py -q` |
| Public segmented calls: argument and offsets validation on host tensors | Public today | The tests of `python/tests/mlir/test_public_segments.py` that need no GPU | `ninja -C build check-swage-python` |
| Public segmented calls: results against `torch.segment_reduce`, `torch.softmax`, and float64 on the nine benchmark distributions, empty batches and segments, long segments, and special values; the schedule of a public sum; `out`; graph capture, streams, threads, inference mode, and no-compile mode; resource use over repeated calls; and the segmented example | Public today | The CUDA tests of `python/tests/mlir/test_public_segments.py`, `python/tests/mlir/test_examples.py` | Trusted GPU workflow |
| Installed-wheel segmented qualification: the public segmented calls and the artifact path, run from an installed wheel with `PYTHONPATH` unset, where a skipped test fails the qualification | Public today | `scripts/qualify_installed_segments.sh`, `python/tests/mlir/test_public_segments.py`, `python/tests/mlir/test_segment_columns.py`, `python/tests/mlir/test_artifact.py` | `scripts/qualify_installed_segments.sh PYTHON ORACLE_BUILD WORK_DIR` on an RTX A6000, with `ORACLE_BUILD` a bindings-off build that holds `bin/swage-opt`; the release `gpu` job of `publish-pypi.yml` and the `fixed-runtime-slo` job of `ci-gpu.yml` run it |
| Compile-only PTX emission for every admitted processor, public and private kernels | Public today, compile-only; private qualification | `python/tests/mlir/test_target_compile.py` | `ninja -C build check-swage-python` |
| Loaded-module lifetime, in-process cache bounds, cold-path locking, and the context and overlap guards of prepared launches | Public today; private qualification | `python/tests/mlir/test_module_lifetime.py` | `ninja -C build check-swage-python`; trusted GPU workflow |
| Module leases: a launch holds a lease on its loaded module, a retired module is unloaded only after no lease holds it, no graph captured it, and an event on every stream that launched it has completed; interpreter exit waits, bounded, and `os.fork()` waits for a compile in flight | Public today; private qualification | `tests/python/test_runtime.py`, `python/tests/mlir/test_module_lifetime.py`, `python/tests/mlir/test_segmented_cache.py` | `python -m pytest tests/python/test_runtime.py -k "lease or exit or fork"`; `python -m pytest python/tests/mlir/test_module_lifetime.py`; `python -m pytest python/tests/mlir/test_segmented_cache.py -k lease`; trusted GPU workflow |
| Concurrent compiles: compiles of different kernels run at the same time, and each kernel compiles once | Public today; private qualification | `tests/python/test_runtime.py`, `python/tests/mlir/test_segmented_cache.py` | `python -m pytest tests/python/test_runtime.py -k "concurrent or coalesced"`; `python -m pytest python/tests/mlir/test_segmented_cache.py -k "across_threads or at_once"` |
| C API contract: code generation entry points, failure reporting, and dialect handles | Compiler-facing, not public API | `unittests/CodegenCAPITest.cpp`, `unittests/DialectsCAPITest.cpp` | `ninja -C build check-swage-unit` |
| The C API lowering pipeline, through the nested NVVM conversion, run from text by `swage-opt` | Compiler-facing, not public API | `test/Conversion/SwageToGPU/nvvm-pipeline.mlir` | `ninja -C build check-swage` |
| Compiler-generated launch contracts: the schema, canonical JSON, validation against the lowered entry, and binding by origin | Compiler-facing, not public API | `unittests/KernelContractTest.cpp`, `python/tests/mlir/test_codegen.py`, `tests/python/test_abi.py` | `ninja -C build check-swage-unit`; `ninja -C build check-swage-python`; `python -m pytest tests/python/test_abi.py -q` |
| Kernel contracts on planned kernels: `--swage-plan-to-gpu` attaches `swage.kernel_contract` to every kernel, a user argument binds by the position of its parameter, and the driver checks each contract once and every launch against it | Compiler-facing, not public API | The files under `test/Conversion/SwagePlanToGPU` that check `swage.kernel_contract`, `python/tests/mlir/test_segmented_codegen.py`, `tests/python/test_runtime.py` | `ninja -C build check-swage`; `python -m pytest python/tests/mlir/test_segmented_codegen.py -k contract`; `python -m pytest tests/python/test_runtime.py -k contract` |
| Emitted kernel text: the SHA-256 of the lowered MLIR and of the PTX of every kernel in a matrix of programs, schedules, and admitted processors, against a committed record | Compiler-facing, not public API | `python/tests/mlir/test_ptx_digests.py` with `ptx_digests.json` | `ninja -C build check-swage-python` |
| Target description: one record of block widths, claim batches, planning defaults, and processors, read by the lowerings, the C API, and the private runner | Compiler-facing, not public API | `unittests/TargetDescriptionTest.cpp`, `python/tests/mlir/test_bindings.py` | `ninja -C build check-swage-unit`; `ninja -C build check-swage-python` |
| Segment functions: argument roles in any order, the kernel parameter layouts, any number of segment functions in a module, the `function` option, and the symbol checks before a GPU lowering changes a module | Compiler-facing, not public API | `test/Dialect/Swage/roles.mlir`, `invalid-roles.mlir`, the `roles-reordered.mlir`, `two-kernels.mlir`, `no-segment-function.mlir`, `invalid-function-option.mlir`, `invalid-kernel-symbols.mlir`, and `invalid-split-kernel-symbols.mlir` files under `test/Conversion`, `unittests/KernelLayoutTest.cpp`, `unittests/CodegenCAPITest.cpp` | `ninja -C build check-swage`; `ninja -C build check-swage-unit` |
| Map fusion: a `swage.map` with one consumer moves into that consumer, in the pass and in the lowerings, and nothing else changes | Compiler-facing, not public API | `test/Dialect/Swage/fuse-maps.mlir`, `test/Conversion/SwageToCPU/unused-reduction.mlir`, the map and softmax files under `test/Conversion` | `ninja -C build check-swage` |
| Segmented sum, max, and min CPU/GPU parity, device-side range and index bounds, and block-size admission | Private qualification | `test/Conversion/SwageToCPU`, `test/Conversion/SwageToGPU`, including `segment-bounds.mlir`, `segment-id-bounds.mlir`, `invalid-block-size.mlir`, and `invalid-unverified.mlir` (which turns parse-time verification off on purpose to reach the lowering's own checks), `python/tests/mlir/test_segmented_runtime.py`, `python/tests/mlir/test_segmented_bounds.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Stable ragged-softmax parity and edge cases | Private qualification | ragged-softmax lit files and `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Sum rounding and reproducibility by schedule, sum error bound, sum special values, and softmax accuracy by logit spread | Private qualification | `python/tests/mlir/test_segmented_numerics.py` | `ninja -C build check-swage-python`; trusted GPU workflow |
| Synchronization counts and loop shape of every kernel after the LLVM pass pipeline | Public today, compile-only; private qualification | `python/tests/mlir/test_kernel_optimization.py` | `ninja -C build check-swage-python` |
| Planning admission, limits, and descriptors, the record classifier against the descriptor classifier, and host validation and classification against a Python reference | Private qualification | `test/Conversion/SwageToPlan/invalid.mlir`, `unittests/TaskClassifierTest.cpp`, the plan cases of `unittests/CodegenCAPITest.cpp`, `python/tests/mlir/test_segmented_classification.py` | `ninja -C build check-swage`; `ninja -C build check-swage-unit`; `ninja -C build check-swage-python` |
| Plan stage: the plan dialect and its verifiers, the planner for the direct, task-id, fused-mixed, split-partial, split-merge, persistent, and sequential schedules and for schedule lists, the conversions of plan functions to kernels and of sequential plans to loops, and their checks on plan IR written by hand | Compiler-facing, not public API | `test/Dialect/SwagePlan`, `test/Conversion/SwageToPlan`, `test/Conversion/SwagePlanToGPU`, `test/Conversion/SwagePlanToSCF`, the oracle files under `test/Conversion/SwageToCPU` and the kernel goldens under `test/Conversion/SwageToGPU`, whose CHECK lines the plan stage left as they were, `python/tests/mlir/test_ptx_digests.py` | `ninja -C build check-swage`; `ninja -C build check-swage-python` |
| Pure and fused mixed capture-free sum, max, and min correctness | Private qualification | `test/Conversion/SwageToGPU/fused-mixed.mlir`, `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Prepared-launch kernel memoization, stale-offsets guard, and graph-capture protocol | Private qualification | `python/tests/mlir/test_segmented_cache.py`, `python/tests/mlir/test_prepared_capture.py` | `ninja -C build check-swage-python`; trusted GPU workflow |
| Frozen mixed-policy performance gate | Private qualification | `benchmarks/results/mixed-sum-a6000-sm86.json`, `tests/python/test_benchmark_mixed_sum.py` | `python -m pytest tests/python/test_benchmark_mixed_sum.py -q` |
| Composable split sum, max, and min, ordering, failures, and f32 parity | Private qualification | `unittests/TaskClassifierTest.cpp`, `test/Conversion/SwageToGPU/split-partial.mlir`, `split-merge.mlir`, and `invalid-split.mlir`, `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; `ninja -C build check-swage-unit`; trusted GPU workflow |
| Persistent claims, fenced split completion, poisoned scratch, graph replay, randomized plans, and failure paths | Experimental; predeclared performance gate failed | `test/Conversion/SwageToGPU/persistent.mlir` and `invalid-persistent.mlir`, `test/Conversion/SwageToPlan/persistent.mlir`, `test/Conversion/SwagePlanToGPU/persistent-tasks.mlir`, `python/tests/mlir/test_segmented_codegen.py`, `python/tests/mlir/test_segmented_runtime.py`, `python/tests/mlir/test_persistent_runtime.py`, `benchmarks/results/persistent-sum-a6000-sm86.json` | `ninja -C build check-swage`; `ninja -C build check-swage-python`; trusted GPU workflow after merge |
| Native AddressSanitizer and UndefinedBehaviorSanitizer over the lit suite and the C++ unit tests | Hosted gate | The `sanitizers` job of `ci-cpp.yml`, which instruments Swage against the uninstrumented Release LLVM on pull requests and pushes to `main`; the `sanitizers-instrumented-llvm` job of `sanitizers.yml`, which also instruments LLVM, on pushes to `main`, weekly, and by hand | `ninja -C <build> check-swage check-swage-unit` in a bindings-off build with the sanitizer flags of either job |
| Recorded RTX 5090 performance snapshot | Recorded evidence | `benchmarks/results/perf-5090-sm120.json` | Not re-executable in CI |
| Recorded fresh-offsets and frozen comparison runs on RTX A6000 at `453c56e`, their generated summary page, and the documentation fragments | Recorded evidence | `benchmarks/results/segmented-sum-a6000-sm86-453c56e/`, `tests/python/test_campaign_tables.py` | `python -m pytest tests/python/test_campaign_tables.py -q`; the measurements are not re-executable in CI |
| Public segment syntax, and public launch of a segment program that the caller writes | Planned | No executable public contract | No passing gate yet |
| Packed warps, queues, and persistent scheduling | Planned | No executable public contract | No passing gate yet |

A `python -m pytest` command on `python/tests/mlir` needs the native
bindings on `PYTHONPATH`, as `ninja -C build check-swage-python` sets them:
`PYTHONPATH="$PWD/python:$PWD/build/python_packages"`.

The sequential CPU oracle transports each f32 or f64 result as its exact
bit pattern, so oracle comparisons involve no decimal rounding.

Numerical claims have their own evidence in
`python/tests/mlir/test_segmented_numerics.py`:

- Reproducibility: each sum schedule returns the same f32 bits across
  launches, across a new preparation, in a second process, and at another
  position in the batch.
- Schedule dependence: the warp, CTA, and split trees return different bits
  for the same segment, automatic selection changes the bits of a segment
  when the batch reaches the SM count of the device, and the pinned
  schedules do not.
- Sum accuracy: every schedule stays within `k * eps32 * sum(|x|)` of a
  float64 reference at 100,003 to 1,048,577 elements, as does the CTA
  schedule that automatic selection substitutes on a batch of 8192-element
  segments, and every schedule propagates NaN, infinities, and subnormal
  values.
- Code generation: the PTX of every sum kernel and of the softmax kernel
  uses round-to-nearest f32 operations with no fused multiply-add and no
  flush-to-zero. This check needs no GPU.
- f64 reductions: every static schedule and the one-CTA path stay within
  `k * eps64 * sum(|x|)` of the exactly rounded sum, return the exact
  result on f64 values that are exactly summable and are not f32 values,
  and propagate NaN, infinities, signed zeros, and the subnormal values of
  f64. The PTX of every f64 kernel holds f64 round-to-nearest operations
  and no f32 instruction, and an f64 `math.exp2` is refused with a
  diagnostic by every compile function
  (`python/tests/mlir/test_segmented_codegen.py`).
- Mean: on every static schedule and on the one-CTA path, in both element
  types, a mean equals the sum of the same schedule divided by the length,
  bit for bit, at lengths that include 4097, 100,003, and 1,048,577, and an
  empty segment gives NaN. The public mean also stays within
  `(k + 1) * eps * sum(|x|) / n` of the exactly rounded mean, which
  `python/tests/mlir/test_public_segments.py` forms in rational arithmetic.
  The PTX of every mean kernel holds one conversion and one division per
  task region, and the partial kernel holds neither.
- Softmax accuracy: every output stays within a relative bound that grows
  linearly with logit spread, at spreads 8, 20, 50, and 80, and
  `ex2.approx.f32` is measured on its own.

[Segmented Reductions](segmented-reductions.md#sum-rounding) states the sum
trees, the bound, and how to pin a schedule.
[Ragged Softmax](ragged-softmax.md#accuracy) states the softmax bound and
the measured errors. All device measurements are from the RTX A6000
(`sm_86`); no other GPU has run them.

The public segmented calls have these checks in
`python/tests/mlir/test_public_segments.py`, on the RTX A6000 (`sm_86`):

- Differential results: sums stay within `k * eps32 * sum(|x|)` of a
  float64 reference and maxima equal `torch.segment_reduce` exactly, on
  seeded batches with the length shape of each of the nine benchmark
  distributions, and on single segments of 100,003 and 1,048,577 elements.
  Softmax outputs stay within the relative bound of
  [Ragged Softmax](ragged-softmax.md#accuracy) against float64
  `torch.softmax`.
- Schedule: a public sum, maximum, and minimum each equal the prepared
  private
  `mixed` launch with default limits and automatic selection bit for bit,
  on every batch of the differential suite, on both sides of the selection
  rule, and on one segment of 300,001 elements. The bits of a sum change
  between two batch sizes around the SM count of the device.
- Preparation: a call compiles and loads only the kernels its batch
  launches, never the pure warp kernel, and reads the shared segment ids
  only for a batch that the selection rule sends to the pure CTA kernel.
- Inference mode: both calls run with `values`, `offsets`, and `out`
  created under `torch.inference_mode()`.
- Resource use: 300 calls with offsets not seen before load no module,
  unload none, never synchronize the context, compile nothing, create no
  CUDA event, leave device memory where it was, and leave nothing for the
  cycle collector.

These checks were executed on that device from the branch that added the
calls. The trusted GPU workflow has not executed them yet.

Calls that run from an artifact, which
[Running Without the Compiler](../user-guide/deployment.md) describes, have
these checks:

- The writer, in `python/tests/mlir/test_artifact.py` without a device: an
  artifact is written for every admitted processor, each shipped kernel
  equals an independent compile of the same request, and each manifest
  entry states the entry name, the launch width, and the parameters that
  the PTX declares.
- The classifier of the runtime library, in `unittests/RuntimeTest.cpp` and
  `python/tests/mlir/test_segmented_classification.py`: it returns the
  records of the compiler's classifier on seeded layouts that hold every
  classification boundary, and refuses malformed input with the same
  message.
- The loader, in `tests/python/test_artifact.py` without the native build:
  selection, every refusal, and the permission rule.
- Execution, in `python/tests/mlir/test_artifact.py` on the RTX A6000
  (`sm_86`): a process that cannot import `mlir_swage` runs both calls from
  an artifact on batches with the length shape of each of the nine
  benchmark distributions, on a batch that reaches the direct-CTA
  selection, and on one segment of 100,003 elements. The process maps no
  file whose name contains `LLVM`, `MLIR`, `mlir`, `SwagePythonCAPI`,
  `swageDialects`, or `nanobind`. Its results pass the comparisons of the
  public calls above and equal the results of the compiled path bit for
  bit. A second process, in which `mlir_swage` is importable, runs a subset
  of those batches from the artifact with no such file mapped, and then
  launches the fixed vector add, which maps the compiler.

These checks were executed from the branch that added artifacts. The
trusted GPU workflow has not executed them yet.

The trusted GPU workflow, `ci-gpu.yml`, runs only on `main`, weekly and by
hand, through the self-hosted `swage-gpu` runner. Its `runtime-qualification`
job runs the whole `python/tests/mlir` directory from the build tree, so
every CUDA-gated test file runs there, including one added later; hosted CI
has no GPU and skips those tests. Its `fixed-runtime-slo` job builds a local
native wheel, installs it, and runs the installed CUDA checks, the
installed-wheel segmented qualification, and the fixed-runtime gate of
[Benchmarking](../reference/benchmarking.md). The release `gpu` job of
`publish-pypi.yml` runs the same qualification and gate on the release
wheel. Documentation in a branch can cite committed
evidence but cannot establish a new GPU result without executing that
workflow or an equivalent recorded qualification. Recorded evidence is a
citation status, not a boundary status: the snapshot row upgrades nothing,
and its numbers are presented on [Benchmarks](benchmarks.md).

One check is opt-in and no workflow runs it.
`python/tests/mlir/test_racecheck.py` runs the direct, task-ID, fused mixed,
split partial and merge, persistent, and softmax kernels under the
racecheck tool of NVIDIA Compute Sanitizer, beside a deliberately racy
kernel that the tool must report. It runs only when `SWAGE_RACECHECK=1` is
set, CUDA is available, and `compute-sanitizer` is found on `PATH` or in the
`compute-sanitizer` directory under `CUDA_HOME`, `CUDA_PATH`, or
`/usr/local/cuda`, where the development machine has version 2023.3 of it;
otherwise it is skipped. No row above relies on it, and this page claims no
result for it.

The hosted `ci-cpp` workflow also defines a `format` job, a `clang-format`
check of the C and C++ sources, that no row above relies on. The sanitizer
row names two jobs. The `sanitizers` job of `ci-cpp.yml` builds Swage with
AddressSanitizer and UndefinedBehaviorSanitizer against the cached Release
LLVM, so it reports a finding in Swage's own code but not inside an LLVM
call. The `sanitizers-instrumented-llvm` job of `sanitizers.yml` builds a
sanitizer-matched LLVM as well and reports both, and because that LLVM build
is long it does not run on pull requests. This page records no run of
either job or of the `format` job and claims no result for them. The
[Support Matrix](../reference/support-matrix.md) lists the versions each
workflow runs with.

## Release evidence and trusted GPU separation

Hosted Python CI covers regular-GIL CPython 3.10 to 3.13 without LLVM,
including the typed public fixed-vector contract and the checks of the
public segmented calls that need no native build. The native hosted job uses
exactly LLVM/MLIR 22.1.8 and also builds, installs, and tests a local cp313
native wheel. The release wheel lanes use their active CPython ABI and
`X86;NVPTX` LLVM targets in a digest-pinned manylinux 2.28 x86-64 container;
a workstation-built `linux_x86_64` wheel does not establish manylinux
compatibility. The supported release platform is Linux x86-64 with glibc
2.28 or newer, and optional PyTorch >=2.6,<3.

Each repaired wheel must include `swage` with its segmented modules and the
artifact writer, the self-contained `mlir_swage` extensions and runtime
libraries including `libSwageRuntime.so`, validated build identity, public
type metadata, and the license files `LICENSE`, `LICENSES/LLVM.txt`, and
`THIRD_PARTY_NOTICES.md`. The artifact checker rejects bytecode, external
MLIR runtime dependencies, absolute or non-`$ORIGIN`-relative
RPATH/RUNPATH, build-root leaks, and `libcuda.so.1` as a linked dependency.
The shared repair helper is the only release repair path. The cp313 sdist
rebuild uses the same toolchain and commit-derived `SOURCE_DATE_EPOCH`;
unequal repaired hashes block release, with no waiver.

The required checks on `main` are `test (3.10)`, `test (3.13)`, `docs`, and
`build-and-test`. Operator configuration of those requirements, protected
`main` and `v*` tags, signed tags, and the `pypi` reviewer and trusted
publisher requires separate authorization; this page is not evidence of
that setup. On a tag run, the publication workflow checks GitHub's
cryptographic tag verification as well as tag identity and main protection
and ancestry.

The scheduled and manual `ci-gpu` workflow runs only on trusted `main`
through the self-hosted `[linux, x64, swage-gpu]` runner. Its
`runtime-qualification` job retains build-tree and private research
coverage; its separate `fixed-runtime-slo` job installs a local native wheel,
qualifies the public segmented calls from it with
`scripts/qualify_installed_segments.sh`, and retains raw SLO JSON even on
failure. Neither replaces the release workflow's gate on the **actual
aggregated repaired cp313 artifact**, installed without `PYTHONPATH` in an
isolated environment reusing CUDA-enabled runner PyTorch. That gate checks
fixed native runtime behavior, boundary sizes `0, 1, 127, 128, 129, 4097`,
exact results, memory-cache reuse, second-process persistent hits, the
installed-wheel qualification of the public segmented calls, and the SLOs
below.

Only NVIDIA RTX A6000 / `sm_86` is release-qualification hardware. Other
admitted targets remain unqualified and best-effort; CUDA admission is not a
trusted performance result. Documentation in a branch can cite committed
evidence but cannot establish a new GPU result without executing the
trusted workflow or an equivalent recorded qualification. Recorded evidence
is a citation status, not a boundary status: historical snapshots upgrade
nothing, and their numbers are presented on [Benchmarks](benchmarks.md).
The private persistent performance gate remains failed; fixed-vector release
hardening neither completes it nor changes the v0.6.0 mapping.

## Multiplication regression coverage

The fixed runtime suites compare CPU and CUDA multiplication against widened
FP32 arithmetic rounded once to the storage dtype. FP16 uses PyTorch's cast;
FP8 uses a version-independent format oracle so PyTorch 2.13's saturating
E4M3FN overflow does not redefine Swage's non-saturating contract. Every
non-NaN output must match its storage encoding exactly, including signed
zeros and subnormals; NaN payloads are not part of the numerical contract.

`test_low_precision_runtime.py` separates numerical datasets from launch
layouts so that large datasets do not multiply the geometry test matrix:

- FP8 covers all 65,536 encoding pairs for each format and backend.
- FP16 covers all 65,536 encodings against 19 factors, including signed
  zeros, rounding boundaries, range limits, infinities, and NaN.
- FP32 covers 65,536 seeded raw-encoding pairs and both signs of every
  exponent with mantissas around rounding boundaries.
- Launch coverage spans blocks `1, 31, 32, 33, 128, 256, 512, 1024`, empty
  inputs, adjacent boundary sizes, multiple blocks, and 1,000,003 elements
  at block 256.
- Storage coverage checks contiguous offset views, guard bytes, shorter
  counts, and output aliasing with either input or both, using snapshots
  taken before an in-place launch.

Validation tests reject malformed metadata and geometry before compilation
or execution, on cold and warm calls including zero work. Compiler tests
independently corrupt gather and scatter masks and offsets and reject
arithmetic chains while checking source diagnostics, unchanged input
modules, deterministic output, and physical launch contracts.

`test_operation_dispatch.py` exercises all four dtypes through prepared
CUDA and forced ctypes launches, changed streams and storage, and graph
replay after input changes and cache eviction. Its two-process cache test
has identically named add and multiply kernels produce eight distinct
artifacts, then reuses them with compilation disabled in the second
process. A dirty checkout uses the persistent cache too; the test skips only
when the process would not read or write the cache, and
`SWAGE_REQUIRE_PERSISTENT_CACHE_TEST=1` turns that skip into a failure, as
the release `gpu` job sets it. CPU compilation coalescing is covered in
`test_cpu_runtime.py`; the installed CPU workflow also runs the numerical
and dispatch suites with CUDA cases skipped.

These correctness checks do not change the performance gates below. Local
A6000 results establish local evidence; sanitizer, hosted ABI and PyTorch,
manylinux, and additional-device qualification require their own runs.

## Fixed-runtime SLO gates

`python -m swage.bench vector-add --output fixed-runtime-slo.json` writes
raw JSON before returning, including failed gates and partial measurement
errors. The installed-wheel
[benchmark CLI contract](../reference/benchmarking.md) keeps the
vector-add-only scope; public multiplication and the segmented calls are not
benchmark selectors. `--enforce` refuses hardware other than exactly NVIDIA
RTX A6000 / `sm_86`. Correctness runs before every timing section; any
mismatch invalidates the record regardless of timing. Passing this CLI alone
does not qualify a release.

| Gate | Method | Passing threshold |
|---|---|---|
| Cold compile/load/first synchronized CUDA launch | Five fresh child processes, unique empty cache per process, `n=129`, `BLOCK=128` | Median <=250 ms and maximum <=400 ms |
| Warm host dispatch | `n=129`, `BLOCK=128`; 200 warmups, then 20 batches of 500 launches, synchronize before and after each batch | Median <=15 microseconds/call and p95 <=20 microseconds/call |
| Large-vector throughput | `n=2^18` and `2^20`, `BLOCK=256`; rotating interleaved CUDA-event batches of 32 launches, 25 warmups, 100 samples against `torch.add(out=...)` | Swage/PyTorch median ratio <=1.85 at `2^18` and <=1.50 at `2^20` |
| Native compiler memory | Linux `/proc/self/status` after PyTorch/tensor setup versus after first compile/launch | RSS increase <=512 MiB |

The `2^18` ceiling was 1.50. The
[gate calibration record](https://github.com/abhiksark/swage/blob/main/benchmarks/results/fixed-runtime-gate-a6000-sm86.md)
gives the alternating runs it was set from and the rule;
[Benchmarking](../reference/benchmarking.md#frozen-evidence-contract)
summarizes why it moved.

Retain raw samples, statistics, thresholds, correctness and pass fields,
package and build identity, Python, PyTorch, CUDA, and driver versions, and
hardware facts. A manual `publish-pypi` dry run must retain four wheels, one
sdist, checksums, one SPDX JSON SBOM, repair, reproducibility, and CPU
evidence, and trusted GPU evidence; it does not produce attestations or
publish. Signed-tag runs may attest and publish only after all gates pass.

### Local installed-wheel evidence

The 2026-09-05 repaired cp313 run on NVIDIA RTX A6000 / `sm_86`, CPython
3.13.13, and PyTorch `2.12.0+cu130` passed every SLO of that time and all 32
installed fixed-runtime tests.
[Raw samples and build identity](https://github.com/abhiksark/swage/blob/main/benchmarks/results/fixed-runtime-a6000-sm86.json)
record the following results:

| Measure | Observed |
|---|---|
| Cold compile/load/launch | 30.21 ms median, 31.12 ms maximum |
| Warm end-to-end host dispatch | 4.14 microseconds median, 4.92 microseconds p95 |
| Throughput ratio at `2^18` | 1.436 |
| Throughput ratio at `2^20` | 1.023 |
| Maximum native RSS increase | 78,839,808 bytes |

This is local engineering evidence, not official release qualification: the
candidate recorded an uncommitted source tree. No signed tag, publication,
or attestation is implied by this passing run. The gate calibration record
above measures later wheels on the same host.

Continue with [Benchmarks](benchmarks.md) for the recorded measurements. For
historical planning and release mapping, use
[`ROADMAP.md`](https://github.com/abhiksark/swage/blob/main/ROADMAP.md). For
why the boundaries were chosen, use the [ADR Index](../decisions/index.md).
