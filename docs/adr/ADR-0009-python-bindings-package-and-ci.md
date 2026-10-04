<!-- docs/adr/ADR-0009-python-bindings-package-and-ci.md -->
# ADR-0009: Self-contained Python bindings package and CI bindings build

- Status: accepted
- Date: 2026-08-18

## Context

The planned frontend constructs Swage MLIR directly through the MLIR Python
bindings (ADR-0001); textual IR is a debug format. At the time of this
decision, Issue #5 registered the `swage` dialect in those bindings and
Issue #4 remained the frontend follow-up.
Two designs needed a decision with alternatives:

1. How the dialect's Python package relates to MLIR's. A thin package
   that layers onto whatever `mlir` package the environment supplies,
   versus a self-contained package that embeds the pinned MLIR core.
2. Where the bindings are tested. The native CI path must build the pinned
   LLVM/MLIR with Python bindings and run an integration test from the
   Swage build tree.

## Decision

Follow the upstream `mlir/examples/standalone` pattern against the
installed pin:

- A C-API library `SwageCAPI` (`include/swage-c/Dialects.h`,
  `lib/CAPI/Dialects.cpp`) built on
  `MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Swage, swage)` and its
  `DEFINE` counterpart. No custom type helpers: Python constructs
  `!swage.segment<T>` by parsing, which suffices until the direct emitter
  demonstrates a need for `SegmentType.get`.
- A **self-contained** package `mlir_swage`
  (`MLIR_PYTHON_PACKAGE_PREFIX=mlir_swage`) that embeds the pinned MLIR
  core Python sources plus the standard-dialect wrappers the semantic
  level composes with (`builtin`, `func`, `arith`, `math`), our
  generated `swage` op bindings, and one internal nanobind extension.
  The extension began with dialect registration and now also exposes the
  compiler/runtime C APIs for PTX compilation, CUDA launch, and Native host
  execution. Import surface:
  `from mlir_swage import ir`, `from mlir_swage.dialects import swage`.
  Because the package never imports an external `mlir` distribution,
  version skew against the pin is impossible by construction.
- A `SWAGE_PYTHON_BINDINGS` CMake option defaulting to the MLIR
  install's `MLIR_ENABLE_BINDINGS_PYTHON`. Requesting it against an
  install built without bindings is a configure-time error, never a
  silent skip.
- At the original decision, type stubs were deferred to wheel packaging;
  the accepted v0.5.2 extension below supplies public `swage` stubs.
- Tests are pytest (`python/tests/mlir/`), importing `mlir_swage` from
  the build tree: a positive path that programmatically builds segment
  and region ops, captures and `kind` included, verifies, and
  round-trips the text against the lit suite's expectations; a negative
  path asserting a verifier failure surfaces as a Python exception. A
  `check-swage-python` target runs them with the correct `PYTHONPATH`.
- `ci-cpp` builds the LLVM cache with bindings ON: Python is pinned via
  `actions/setup-python`, a cache miss copies MLIR's
  `python/requirements.txt` into the cached install tree, and every cold or
  warm job installs that file plus `pytest` and `lit`. The cache key includes
  the Python minor version because the extension is ABI-specific; pytest runs
  after the lit suite. `ci-python` keeps no LLVM dependency.

The original decision deferred wheel packaging of `mlir_swage`; the
accepted extension below supersedes that deferral. A public `swage.ir` API,
injection into upstream `mlir.dialects`, and Windows/macOS bindings builds
remain out of scope.

## Accepted extension: fixed-vector native wheels

The v0.5.2 implementation extends this accepted decision rather than
introducing another packaging architecture. It is **unreleased pending
artifact, security, CPU, and trusted CUDA gates**; v0.5.1 remains the latest
tagged release. This extension overrides the original wheel/stub deferrals,
not the private status of segmented APIs or their research gates.

- `scikit-build-core==1.0.3` and `nanobind==2.15.0` build native wheels.
  CMake requires exactly LLVM/MLIR 22.1.8 (`llvmorg-22.1.8`). Normal source
  builds keep `build/python_packages/mlir_swage`; `SWAGE_WHEEL_BUILD=ON`
  changes the install prefix to site-packages-root `mlir_swage` under the
  existing `SwagePythonModules` target/component.
- Wheels contain public `swage`, self-contained private `mlir_swage`,
  the MLIR and Swage native extensions, runtime shared libraries, and
  MIT/LLVM licenses. No external `mlir` distribution or local LLVM tree is
  required. PyTorch is optional (`torch>=2.6,<3`, the `pytorch` extra),
  never bundled. Official wheels omit `swage/_segmented*.py` and bytecode;
  source checkouts and sdists preserve private research code. Retained
  internal native symbols do not create a supported public segmented API.
- The artifact set is four ABI-specific
  `manylinux_2_28_x86_64` wheels for regular-GIL CPython 3.10, 3.11, 3.12,
  and 3.13, plus one source distribution. Linux x86-64 / glibc >=2.28 is
  the only release platform; no universal, PyPy, free-threaded, musllinux,
  aarch64, Windows, or macOS wheel is admitted.
- Hand-authored `swage/__init__.pyi`, `language.pyi`, and `py.typed`
  describe the existing fixed-vector `jit`, `emit_mlir`, and `launch`
  contract, explicit CPU/CUDA backend literals, errors, and symbolic DSL
  values. They do not claim a generated typed public MLIR API or a broader
  DSL type system.
- Wheel builds require `Release`, a full 40-character lowercase source
  revision, and explicit clean-tree state. Installed
  `mlir_swage/_build_info.json` contains schema version 1, package version,
  source revision, `source_clean`, LLVM pin, and build type. Validated
  packaged identity takes precedence over checkout discovery for CUDA
  cache identity. A clean identity is required for persistence; malformed
  packaged metadata disables it, while absent metadata retains the
  checkout fallback. This JSON resource is data, not an executable module.
- Checkouts retain normal native builds; extracted sdists support native
  wheel builds with explicit verified source provenance.
  Frontend-only contributor installs use
  `pip install -e ".[dev]" -Cwheel.cmake=false`; sdist creation is also
  CMake-free. Source wheel builds still require the exact pinned native
  toolchain and explicit provenance.
- Hosted Python CI covers all four ABIs, stubs, and reproducible sdists;
  native cp313 CI retains bindings/lit/unit coverage, installed CPU smoke
  with current PyTorch, CodeQL C/C++, and a separate bindings-off
  ASan+UBSan build. Python CodeQL and dependency review add hosted security
  gates without giving pull requests access to the trusted GPU runner.
- Release jobs build LLVM for each active ABI inside a digest-pinned
  manylinux container with `X86;NVPTX` targets, strip and repair through
  the shared repair/check scripts, enforce the strictly sub-95,000,000-byte
  ceiling, inspect ELF dependencies and relative runtime paths, and reject
  leaked build roots. All four wheels pass installed CPU health/smoke with
  `torch==2.6.0+cpu`; the repaired cp313 source-tree and sdist builds must
  have identical hashes under the same `SOURCE_DATE_EPOCH`.
- Aggregation requires exactly five distributions, checksums, and one
  SPDX JSON SBOM. The actual repaired cp313 wheel must pass fixed-runtime
  correctness, second-process cache reuse, and enforced SLOs on trusted
  NVIDIA RTX A6000 / `sm_86`. Other admitted GPUs remain
  unqualified/best-effort. Private research results are not release evidence.
  Manual release-workflow runs never attest or publish; only protected
  signed annotated v0.5.2 tags, cryptographically verified by GitHub and
  checked against protected main ancestry, may reach build/SBOM attestation
  and reviewed OIDC publication.

See [Verification](../internals/verification.md) for gate/evidence boundaries
and [Contributing](https://github.com/abhiksark/swage/blob/main/CONTRIBUTING.md)
for operator prerequisites and the immutable yank-and-patch policy.

## Consequences

- The direct AST-to-MLIR emitter (#4) still uses build-tree bindings during
  native contributor tests. The accepted extension now also ships those
  bindings privately in the native wheel; installed consumers do not set
  `PYTHONPATH` or discover a separate MLIR installation.
- Contributors run `ninja -C build check-swage-python`, which sets
  `PYTHONPATH=build/python_packages`. An install built with bindings off must
  be rebuilt with `SWAGE_LLVM_PYTHON_BINDINGS=ON`, then Swage must be
  reconfigured with `-DSWAGE_PYTHON_BINDINGS=ON`; an incompatible install
  fails configuration instead of silently skipping bindings.
- One LLVM cache serves lit and pytest. The flip costs one full CI
  LLVM rebuild (~3.5 h, as measured on the `nopy` cache) and grows the
  cache by the Python package; afterwards runs return to minutes.
- An LLVM pin bump rebuilds the bindings with the same single pin; no
  second version can drift.
- Changing the CI Python minor version invalidates the LLVM cache by
  design. That is the correctness property, not a bug.
