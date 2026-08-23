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
| M6: Minimal SwagePlan gate | Warp/CTA policy attribute, task-range type, classify operation, identity-sum conversion, host descriptors | One semantic kernel gains a private planning companion; validated host metadata produces exact warp/CTA descriptors | **Complete**; compile-only planning and unit-tested host descriptor generation, with no classifier integration, dispatch, or GPU policy execution |
| M7: Warp vs CTA policy | Classifier integration, task-range lowering, mixed-policy launch, benchmark comparison | Auto mixed policy beats or matches better pure policy on a predeclared distribution | Not started |
| M8: Split-CTA reductions | Long-segment partitioning, partial reductions, merges | Oversized segments execute with no dropped/duplicated elements | Not started |
| M9: Persistent scheduling | Device task queue, resident CTAs, dependency handling | Correct under extreme skew; reduces long-tail idle on a predeclared benchmark | Not started |
| M10: Research benchmark + release | Harness, raw data, plots, prior-art doc, tutorial | One kernel evaluated across distributions against all declared baselines, reproducibly | Not started |

Release mapping (semantic versioning, `0.x`): `v0.2.0` after M3, `v0.3.0`
after M4, `v0.4.0` after M5, `v0.5.0` after M7, `v0.6.0` after M9. `v0.1.0`
was reserved for the frozen Python prototype and will not be used
(ADR-0005).

M5 completion makes `v0.4.0` eligible. It is not released. The deferred
`v0.2.0` and `v0.3.0` releases, all tagging, and release publication remain
separate release work.

Tracked as GitHub milestones; per-phase issues carry the detailed
acceptance criteria.
