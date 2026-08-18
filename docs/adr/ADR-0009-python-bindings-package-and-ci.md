# ADR-0009: Self-contained Python bindings package and CI bindings build

- Status: accepted
- Date: 2026-08-18

## Context

The frontend constructs Swage MLIR directly through the MLIR Python
bindings (ADR-0001); textual IR is a debug format. That makes issue #5 —
registering the `swage` dialect in the bindings — the gate for all of M2.
Two designs needed a decision with alternatives:

1. How the dialect's Python package relates to MLIR's. A thin package
   that layers onto whatever `mlir` package the environment supplies,
   versus a self-contained package that embeds the pinned MLIR core.
2. Where the bindings are tested. The CI LLVM cache is currently built
   with `MLIR_ENABLE_BINDINGS_PYTHON=OFF` (the `nopy` cache key), so CI
   cannot exercise bindings at all without a cache rebuild.

## Decision

Follow the upstream `mlir/examples/standalone` pattern against the
installed pin:

- A C-API library `SwageCAPI` (`include/swage-c/Dialects.h`,
  `lib/CAPI/Dialects.cpp`) built on
  `MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Swage, swage)` and its
  `DEFINE` counterpart. No custom type helpers: Python constructs
  `!swage.segment<T>` by parsing, which suffices until the M2 emitter
  demonstrates a need for `SegmentType.get`.
- A **self-contained** package `mlir_swage`
  (`MLIR_PYTHON_PACKAGE_PREFIX=mlir_swage`) that embeds the pinned MLIR
  core Python sources plus the standard-dialect wrappers the semantic
  level composes with (`builtin`, `func`, `arith`, `math`), our
  generated `swage` op bindings, and a minimal nanobind extension whose
  only job is dialect registration (mirroring
  `StandaloneExtensionNanobind.cpp`). Import surface:
  `from mlir_swage import ir`, `from mlir_swage.dialects import swage`.
  Because the package never imports an external `mlir` distribution,
  version skew against the pin is impossible by construction.
- A `SWAGE_PYTHON_BINDINGS` CMake option defaulting to the MLIR
  install's `MLIR_ENABLE_BINDINGS_PYTHON`. Requesting it against an
  install built without bindings is a configure-time error, never a
  silent skip.
- Type stubs are not generated. They return with wheel packaging.
- Tests are pytest (`python/tests/mlir/`), importing `mlir_swage` from
  the build tree: a positive path that programmatically builds segment
  and region ops — captures and `kind` included — verifies, and
  round-trips the text against the lit suite's expectations; a negative
  path asserting a verifier failure surfaces as a Python exception. A
  `check-swage-python` target runs them with the correct `PYTHONPATH`.
- `ci-cpp` builds the LLVM cache with bindings ON: Python pinned via
  `actions/setup-python`, MLIR's `python/requirements.txt` installed on
  cache miss, and the cache key extended with the Python minor version
  (the extension is ABI-specific) and bumped to `v2`. pytest runs after
  the lit suite. `ci-python` keeps no LLVM dependency.

Out of scope until a consumer exists: wheel packaging of `mlir_swage`,
a public `swage.ir` API, injection into the upstream `mlir.dialects`
namespace, and Windows/macOS bindings builds.

## Consequences

- The M2 emitter (#4) imports `mlir_swage` from the build tree via
  `PYTHONPATH`; how the pip package `swage-compiler` ships or locates
  the native package is a packaging decision deferred to the wheel ADR.
- One LLVM cache serves lit and pytest. The flip costs one full CI
  LLVM rebuild (~3.5 h, as measured on the `nopy` cache) and grows the
  cache by the Python package; afterwards runs return to minutes.
- An LLVM pin bump rebuilds the bindings with the same single pin; no
  second version can drift.
- Changing the CI Python minor version invalidates the LLVM cache by
  design — that is the correctness property, not a bug.
