# Roadmap

Phases are gates, not dates. A phase is done when its acceptance gate
passes with tests; nothing below claims more than the tests show.

| Phase | Scope | Gate | Status |
|---|---|---|---|
| P0: Python prototype freeze | Tag and freeze the pre-existing Python prototype | Prototype reproducible | **Not applicable**: the repository started empty; there is no prototype (ADR-0005) |
| M0: MLIR project foundation | LLVM pin, out-of-tree CMake, `swage` dialect skeleton, `swage-opt`, lit, CPU CI, community health | `swage-opt` round-trips Swage IR; lit + pytest green | **Complete** |
| M1: Swage dialect | Region-based `map`/`reduce`/`map_store`/`yield`, full verifier set | Positive and negative dialect tests | **Complete** |
| M2: Python AST → MLIR | `@sw.jit` frontend, `constexpr`, source locations, diagnostics | Vector-add kernel emits deterministic MLIR; no dataclass IR | **Complete** |
| M3: Fixed vector add via NVPTX | GPU lowering, PTX emission, driver launch, JIT cache | Python vector add returns correct CUDA results on PyTorch tensors | **Complete**; hosted CPU/native checks and trusted RTX A6000 `sm_86` workflow pass |
| M4: Segmented sum/max | Sequential CPU oracle, one-CTA-per-segment lowering, empty segments, offset validation | Matches PyTorch and CPU oracles | **Complete**; native checks and RTX A6000 `sm_86` qualification pass |
| M5: Ragged softmax parity | Fusion, captures, stable softmax, multi-stage execution | Matches PyTorch across adversarial distributions | **Complete**; sequential CPU oracle and RTX A6000 `sm_86` one-CTA qualification pass |
| M6: Minimal SwagePlan gate | Warp/CTA policy attribute, task-range type, classify operation, identity-sum conversion, host descriptors | One semantic kernel gains a private planning companion; validated host metadata produces exact warp/CTA descriptors | **Complete**; the planning artifact remains compile-only, and M7 consumes it only through a private qualification path |
| M7: Warp vs CTA policy | Classifier integration, task-range materialization, mixed-policy launch, benchmark comparison | Auto mixed policy beats or matches better pure policy on a predeclared distribution | **Complete**; private identity-sum qualification passes on RTX A6000 `sm_86` with a `0.939394` mixed-to-best-pure ratio, with no public segmented launch |
| M8: Split-CTA reductions | Long-segment partitioning, partial reductions, merges | Oversized segments execute with no dropped/duplicated elements | **Complete**; private identity-sum split-only and mixed paths match PyTorch and the CPU oracle on RTX A6000 `sm_86`, with no public segmented launch or benchmark retuning |
| M9: Persistent scheduling | Device task queue, resident CTAs, dependency handling | Correct under extreme skew; reduces long-tail idle on a predeclared benchmark | **In progress**; private identity-sum claims and split completion are correct, but the synchronized clean A6000 run was 1.06% faster and failed the predeclared 5% gate |
| M10: Research benchmark + release | Harness, raw data, plots, prior-art doc, tutorial | One kernel evaluated across distributions against all declared baselines, reproducibly | Not started |

Release mapping (semantic versioning, `0.x`): `v0.2.0` after M3, `v0.3.0`
after M4, `v0.4.0` after M5, `v0.5.0` after M7, `v0.6.0` after M9. `v0.1.0`
was reserved for the frozen Python prototype and will not be used
(ADR-0005).

M7 was the `v0.5.0` feature gate, and `v0.5.0` was published on 2026-08-24.
M8 is a post-release internal correctness milestone. The deferred `v0.2.0`,
`v0.3.0`, and `v0.4.0` versions remain outside this roadmap gate.

### Fixed-vector release hardening: v0.5.2

The source tree implements self-contained native wheels, fixed-contract typing,
stable backend errors, structured health checks, build provenance, and gated
artifact/security/SLO workflows. v0.5.2 remains unreleased until the four-ABI
CPU artifact checks, cp313 reproducibility check, and trusted installed-wheel
A6000 performance gates pass, followed by the protected signed-tag process.
No production-ready claim follows from implementation alone.

The source tree also extends canonical vector addition to matching
`float16`, `float8_e4m3fn`, and `float8_e5m2` tensors, alongside `float32`.
CPU and RTX A6000 `sm_86` execution pass exhaustive FP8 input-pair checks
and FP16 rounding/encoding coverage. These are correctness results, not
low-precision performance qualification; release artifacts must be rebuilt
and pass the expanded installed-wheel checks.

The canonical fixed vector-add kernel shape is unchanged; these additions
do not complete M9 or add a public segmented API. The v0.6.0 mapping remains
unchanged, and the failed persistent scheduling performance gate is not waived.

Tracked as GitHub milestones; per-phase issues carry the detailed
acceptance criteria.
