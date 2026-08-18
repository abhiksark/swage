# ADR-0004: Exact LLVM pin

- Status: accepted
- Date: 2026-08-18

## Context

MLIR APIs move quickly; tracking `main` breaks contributors and makes
results irreproducible. The project needs one known-good LLVM/MLIR
revision, an out-of-tree build against it, and a deliberate upgrade
process.

## Decision

Pin the newest stable release at foundation time: **`llvmorg-22.1.8`**
(23.x was still in RC), recorded machine-readably in
`cmake/llvm-version.txt` and consumed by `scripts/fetch_llvm.sh` /
`scripts/build_llvm.sh` (release tarball, no git history, no vendored or
submoduled sources). Swage builds out-of-tree via `MLIR_DIR`/`LLVM_DIR`.
Build configuration: `LLVM_ENABLE_PROJECTS=mlir`,
`LLVM_TARGETS_TO_BUILD=Native;NVPTX`, `LLVM_ENABLE_ASSERTIONS=ON`,
`LLVM_INSTALL_UTILS=ON`, `MLIR_ENABLE_BINDINGS_PYTHON=ON`; default build
type `RelWithDebInfo`, overridable via `SWAGE_LLVM_BUILD_TYPE` (CI and
disk-constrained machines use `Release`, still with assertions).

GCC 11.4 satisfies the release's minimum (`GCC_MIN 7.4`, verified in
`llvm/cmake/modules/CheckCompilerVersion.cmake` of the pinned tree).

## Consequences

- Every build and benchmark names one exact LLVM revision.
- Pin updates are dedicated compatibility PRs, never side effects.
- Contributors pay a one-time ~1 hour LLVM build; CI caches the install
  tree keyed on the pin file.
