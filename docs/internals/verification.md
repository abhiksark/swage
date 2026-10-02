<!-- docs/internals/verification.md -->

# Verification

Status claims are grounded in executable tests and committed artifacts. This
matrix identifies the smallest evidence source for each current boundary and
does not turn private qualification into public API.

<div class="doc-figure" tabindex="0" markdown="1">

![One semantic module executed on the GPU path, the CPU oracle, and PyTorch, feeding a differential comparison](../assets/figures/oracle-topology.svg)

</div>

*The comparison topology behind every correctness claim. [Open the full-size figure](../assets/figures/oracle-topology.svg).*

| Boundary | Status | Primary evidence | Applicable command |
|---|---|---|---|
| Pure `swage` wheel contents and native-package exclusion | Public today | `tests/python/test_packaging.py` | `python -m pytest tests/python/test_packaging.py -q` |
| Environment report | Public today | `tests/python/test_env.py` | `python -m pytest tests/python/test_env.py -q` |
| Restricted AST to verified native module, and the kernel-language check without the native package | Public today, compile-only | `tests/python/test_frontend.py`, `python/tests/mlir/test_frontend.py` | `python -m pytest tests/python -q`; `ninja -C build check-swage-python` |
| Emit-only example without a GPU or PyTorch | Public today, compile-only | `python/tests/mlir/test_examples.py` | `ninja -C build check-swage-python` |
| Native `swage` dialect parsing, verification, and declared segment-read effects | Public today, compile-only | `test/Dialect/Swage`, including `effects.mlir` | `ninja -C build check-swage` |
| Fixed vector-add lowering and CUDA launch | Public today | `test/Conversion/SwageToGPU`, `python/tests/mlir/test_runtime.py`, `python/tests/mlir/test_cache_process_reuse.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Launch validation, the PyTorch floor, and the cache bound, read-only mode, and no-compile mode | Public today | `tests/python/test_runtime.py`, `python/tests/mlir/test_runtime.py` | `python -m pytest tests/python/test_runtime.py -q`; trusted GPU workflow |
| Compile-only PTX emission for every admitted processor, public and private kernels | Public today, compile-only; private qualification | `python/tests/mlir/test_target_compile.py` | `ninja -C build check-swage-python` |
| Loaded-module lifetime, in-process cache bounds, cold-path locking, and the context and overlap guards of prepared launches | Public today; private qualification | `python/tests/mlir/test_module_lifetime.py` | `ninja -C build check-swage-python`; trusted GPU workflow |
| C API contract: code generation entry points, failure reporting, and dialect handles | Compiler-facing, not public API | `unittests/CodegenCAPITest.cpp`, `unittests/DialectsCAPITest.cpp` | `ninja -C build check-swage-unit` |
| Segmented sum and max CPU/GPU parity, device-side range and index bounds, and block-size admission | Private qualification | `test/Conversion/SwageToCPU`, `test/Conversion/SwageToGPU`, including `segment-bounds.mlir`, `segment-id-bounds.mlir`, `invalid-block-size.mlir`, and `invalid-unverified.mlir` (which turns parse-time verification off on purpose to reach the lowering's own checks), `python/tests/mlir/test_segmented_runtime.py`, `python/tests/mlir/test_segmented_bounds.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Stable ragged-softmax parity and edge cases | Private qualification | ragged-softmax lit files and `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Sum rounding and reproducibility by schedule, sum error bound, sum special values, and softmax accuracy by logit spread | Private qualification | `python/tests/mlir/test_segmented_numerics.py` | `ninja -C build check-swage-python`; trusted GPU workflow |
| Planning admission, limits, and descriptors, the record classifier against the descriptor classifier, and host validation and classification against a Python reference | Private qualification | `test/Conversion/SwageToPlan`, `unittests/TaskClassifierTest.cpp`, `python/tests/mlir/test_segmented_classification.py` | `ninja -C build check-swage`; `ninja -C build check-swage-unit`; `ninja -C build check-swage-python` |
| Pure and fused mixed capture-free sum/max correctness | Private qualification | `test/Conversion/SwageToGPU/fused-mixed.mlir`, `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; trusted GPU workflow |
| Prepared-launch kernel memoization, stale-offsets guard, and graph-capture protocol | Private qualification | `python/tests/mlir/test_segmented_cache.py`, `python/tests/mlir/test_prepared_capture.py` | `ninja -C build check-swage-python`; trusted GPU workflow |
| Frozen mixed-policy performance gate | Private qualification | `benchmarks/results/mixed-sum-a6000-sm86.json`, `tests/python/test_benchmark_mixed_sum.py` | `python -m pytest tests/python/test_benchmark_mixed_sum.py -q` |
| Composable split sum/max, ordering, failures, and f32 parity | Private qualification | `unittests/TaskClassifierTest.cpp`, `test/Conversion/SwageToGPU/split-partial.mlir`, `split-merge.mlir`, and `invalid-split.mlir`, `python/tests/mlir/test_segmented_runtime.py` | `ninja -C build check-swage`; `ninja -C build check-swage-unit`; trusted GPU workflow |
| Persistent claims, fenced split completion, poisoned scratch, graph replay, randomized plans, and failure paths | Experimental; predeclared performance gate failed | `test/Conversion/SwageToGPU/persistent.mlir` and `invalid-persistent.mlir`, `python/tests/mlir/test_segmented_codegen.py`, `python/tests/mlir/test_segmented_runtime.py`, `python/tests/mlir/test_persistent_runtime.py`, `benchmarks/results/persistent-sum-a6000-sm86.json` | `ninja -C build check-swage`; `ninja -C build check-swage-python`; trusted GPU workflow after merge |
| Recorded RTX 5090 performance snapshot | Recorded evidence | `benchmarks/results/perf-5090-sm120.json` | Not re-executable in CI |
| Public segmented syntax and execution | Planned | No executable public contract | No passing gate yet |
| Packed warps, queues, and persistent scheduling | Planned | No executable public contract | No passing gate yet |

The sequential CPU oracle transports each f32 result as its exact bit
pattern, so oracle comparisons involve no decimal rounding.

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
  float64 reference at 100,003 to 1,048,577 elements, and propagates NaN,
  infinities, and subnormal values.
- Code generation: the PTX of every sum kernel and of the softmax kernel
  uses round-to-nearest f32 operations with no fused multiply-add and no
  flush-to-zero. This check needs no GPU.
- Softmax accuracy: every output stays within a relative bound that grows
  linearly with logit spread, at spreads 8, 20, 50, and 80, and
  `ex2.approx.f32` is measured on its own.

[Segmented Reductions](segmented-reductions.md#sum-rounding) states the sum
trees, the bound, and how to pin a schedule.
[Ragged Softmax](ragged-softmax.md#accuracy) states the softmax bound and
the measured errors. All device measurements are from the RTX A6000
(`sm_86`); no other GPU has run them.

The trusted GPU workflow runs only on `main` through the self-hosted
`swage-gpu` runner. It runs the whole `python/tests/mlir` directory, so every
CUDA-gated test file runs there, including one added later; hosted CI has no
GPU and skips those tests. Documentation in a branch can cite committed
evidence but cannot establish a new GPU result without executing that
workflow or an equivalent recorded qualification. Recorded evidence is a
citation status, not a boundary status: the snapshot row upgrades nothing,
and its numbers are presented on [Benchmarks](benchmarks.md).

The hosted `ci-cpp` workflow also defines two jobs that no row above relies
on: a `clang-format` check of the C and C++ sources, and a job that runs the
lit suite and the C++ unit tests under AddressSanitizer and
UndefinedBehaviorSanitizer against the uninstrumented LLVM install. Both
jobs have not yet run on hosted CI, so this page claims no result for them.
The [Support Matrix](../reference/support-matrix.md) lists the versions each
workflow runs with.

Continue with [Benchmarks](benchmarks.md) for the recorded measurements. For
historical planning and release mapping, use
[`ROADMAP.md`](https://github.com/abhiksark/swage/blob/main/ROADMAP.md). For
why the boundaries were chosen, use the [ADR Index](../decisions/index.md).
