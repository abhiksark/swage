# Changelog

All notable changes to Swage are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
semantic versioning (`0.x` — anything may change).

## [Unreleased]

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
- Native build-tree `mlir_swage` package with generated `swage` bindings,
  dialect registration, and the `check-swage-python` integration target.
- Bindings-enabled `ci-cpp` coverage that runs the native integration target
  after the lit suite and installs MLIR Python requirements on cold and warm
  LLVM cache paths.
- Project documentation (README, DESIGN, ROADMAP, concept docs, ADRs
  0001–0009), community health files, issue forms, and CPU CI.

### Notes

- No Python kernel compiles or executes yet. The frontend remains deferred to
  Issue #4; see the README status table.
