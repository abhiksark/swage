<!-- CONTRIBUTING.md -->

# Contributing to Swage

The fixed-vector native-wheel implementation targets **v0.5.2**, which is
unreleased until the artifact, security, CPU, and trusted CUDA gates pass.
The latest tagged release is v0.5.1; the broader segmented compiler remains
experimental. Useful contributions distinguish public behavior, private
qualification, and planned work.

## Ground rules

- Preserve semantic correctness before performance.
- Follow the root and nearest scoped `AGENTS.md` files.
- Do not claim planned behavior as implemented.
- Keep internal milestone codenames in the roadmap, maintainer planning, and
  compatibility redirects; use capability names in project surfaces.
- Add tests with behavior changes and run the applicable tier.
- Keep the LLVM pin unchanged outside a dedicated compatibility change.
- Do not add Triton, a second production IR, or silent backend fallback.
- A GPU is not required for Python, documentation, dialect, and CPU-lowering
  contributions.

## Setup

```bash
git clone https://github.com/abhiksark/swage
cd swage

python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -Cwheel.cmake=false
PYTHONPATH="$PWD/python" python -m pytest tests/python -q
ruff check .

./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
SWAGE_BUILD_TYPE=Release ./scripts/build_swage.sh
```

See [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
for prerequisites, build overrides, and the published-package boundary.

## Native Python bindings

Normal source builds retain `build/python_packages/mlir_swage`.
The v0.5.2 wheel implementation instead installs public `swage` and
self-contained, private `mlir_swage` at site-packages root, with native
extensions, runtime libraries, typed public metadata, and build provenance.
It never depends on an external `mlir` package. Official wheels exclude
`swage/_segmented*.py` and bytecode; source checkouts and sdists retain
the private research code.

```bash
ninja -C build check-swage-python
```

This target supplies `build/python_packages` on `PYTHONPATH`. The selected
pinned MLIR install must include Python bindings. CMake requires exactly
LLVM/MLIR 22.1.8 (`llvmorg-22.1.8`), not a nearby compatible version.
Build LLVM with the same regular-GIL CPython interpreter used for the
wheel, using `SWAGE_PYTHON_EXECUTABLE`; official wheel jobs build only
`X86;NVPTX` LLVM targets. The build backend and binding dependency are pinned
to `scikit-build-core==1.0.3` and `nanobind==2.15.0`.

### Local native-wheel dry run

Use regular-GIL CPython 3.10–3.13 on Linux x86-64. The release platform is
glibc 2.28 or newer; no PyPy, free-threaded, musllinux, macOS, Windows, or
aarch64 wheel is in the release contract. PyTorch is optional
(`torch>=2.6,<3`, the `pytorch` extra), is never bundled, and is required
to launch either explicitly selected backend. Only canonical contiguous
rank-1 vector add or multiply is public, with matching `float32`, `float16`,
`float8_e4m3fn`, or `float8_e5m2` tensors. CUDA is the default and never
falls back to CPU. The installed smoke exercises every supported dtype.

After building the pinned LLVM for the active Python, run this from a
**clean committed checkout**. Keep generated output outside the checkout:

```bash
repo="$PWD"
work="$(mktemp -d)"
revision="$(git rev-parse HEAD)"
test -z "$(git status --porcelain)"
export SOURCE_DATE_EPOCH="$(git show -s --format=%ct "$revision")"
export MLIR_DIR="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}/install-llvmorg-22.1.8/lib/cmake/mlir"
export LLVM_DIR="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}/install-llvmorg-22.1.8/lib/cmake/llvm"
python -m build --sdist --outdir "$work/sdist"
python -m twine check --strict "$work"/sdist/*.tar.gz
python -m build --wheel --outdir "$work/unrepaired" \
  -Cbuild-dir="$work/native-build" -Ccmake.build-type=Release \
  -Ccmake.define.MLIR_DIR="$MLIR_DIR" -Ccmake.define.LLVM_DIR="$LLVM_DIR" \
  -Ccmake.define.SWAGE_SOURCE_REVISION="$revision" \
  -Ccmake.define.SWAGE_SOURCE_CLEAN=true
python -m venv "$work/venv"
"$work/venv/bin/python" -m pip install torch==2.6.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu/
"$work/venv/bin/python" -m pip install --no-deps "$work"/unrepaired/*.whl
(
  cd "$work"
  unset PYTHONPATH
  "$work/venv/bin/python" -m swage.env --json --check cpu
  "$work/venv/bin/python" "$repo/scripts/smoke_installed_wheel.py" \
    --backend cpu --expected-revision "$revision"
)
```

Wheel builds require `Release`, a full 40-character lowercase hexadecimal
`SWAGE_SOURCE_REVISION`, and an explicit `SWAGE_SOURCE_CLEAN`.
For local uncommitted work record `false`, never mislabel it clean; it cannot
qualify an official release or enable persistent CUDA caching. An extracted
sdist has no Git checkout: build its native wheel with the verified source
revision and clean flag explicitly, rather than the checkout build helper.
The frontend-only editable install and sdist creation remain CMake-free.

The host-built `linux_x86_64` wheel above is local evidence, **not** a
manylinux release artifact. Release repair must run inside the exact
digest-pinned `manylinux_2_28_x86_64` image in `publish-pypi.yml`, with that
job's Python and pinned LLVM. Do not repackage workstation binaries or
invoke an ad-hoc repair path. In that environment, using the paths above:

```bash
python scripts/repair_native_wheel.py "$work"/unrepaired/*.whl \
  --wheel-dir "$work/repaired" --expected-revision "$revision" \
  --forbid-prefix "$repo" --forbid-prefix "$work/native-build" \
  --rebuild-sdist "$work/sdist/swage_compiler-0.5.2.tar.gz"
python scripts/check_native_wheel.py "$work"/repaired/*.whl \
  --expected-revision "$revision" \
  --forbid-prefix "$repo" --forbid-prefix "$work/native-build"
```

The repair helper invokes `auditwheel`, gates the repaired wheel, and, with
`--rebuild-sdist`, requires a byte-identical rebuilt wheel using the same ABI,
toolchain, and `SOURCE_DATE_EPOCH`. The official cp313 lane requires that
comparison. Install the repaired artifact and repeat the outside-checkout
CPU smoke; a passing unrepaired smoke does not substitute for it.

### Applicable verification tiers

Run tiers in order; a later passing result never waives an earlier failure:

1. **Frontend, typing, and sdist:** `ruff check .`,
   `python -m pytest tests/python -q` (includes mypy fixtures),
   `python -m build --sdist`, and
   `python -m twine check --strict dist/*.tar.gz`, on Python 3.10–3.13.
2. **Native:** `SWAGE_BUILD_TYPE=Release ./scripts/build_swage.sh`,
   `ninja -C build check-swage-unit`, and
   `ninja -C build check-swage-python`. The build script also runs lit.
3. **Sanitizers:** use a separately cached LLVM install at the same pin,
   with matching ASan/UBSan instrumentation. Do not link sanitized Swage
   against ordinary LLVM: LLVM's inline allocator poisoning differs.
   The LLVM build needs Clang and its compiler-rt development files; hosted
   CI uses Ubuntu 24.04 with Clang 18.

   ```bash
   export SWAGE_LLVM_HOME="$HOME/.swage/llvm-sanitizers"
   export CC=clang CXX=clang++
   ./scripts/fetch_llvm.sh
   SWAGE_LLVM_BUILD_TYPE=Release SWAGE_LLVM_PYTHON_BINDINGS=OFF \
     SWAGE_LLVM_TARGETS='X86;NVPTX' \
     SWAGE_LLVM_SANITIZERS='Address;Undefined' \
     CMAKE_BUILD_PARALLEL_LEVEL=2 ./scripts/build_llvm.sh
   tag="$(cat cmake/llvm-version.txt)"
   cmake -G Ninja -S . -B build-sanitizers \
     -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DMLIR_DIR="$SWAGE_LLVM_HOME/install-$tag/lib/cmake/mlir" \
     -DLLVM_DIR="$SWAGE_LLVM_HOME/install-$tag/lib/cmake/llvm" \
     -DLLVM_EXTERNAL_LIT="$(command -v lit)" \
     -DSWAGE_PYTHON_BINDINGS=OFF -DSWAGE_ENABLE_SANITIZERS=ON
   ASAN_OPTIONS=detect_leaks=1:halt_on_error=1:abort_on_error=1 \
   UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1:abort_on_error=1 \
     ninja -C build-sanitizers check-swage check-swage-unit
   ```

4. **Installed artifact:** the local CPU dry run above, followed by the
   manylinux repair/check/reproducibility gates in the release workflow.
   Every ABI uses minimum `torch==2.6.0+cpu`; native cp313 CI additionally
   uses current `torch==2.13.0+cpu`.
5. **Docs:** `make docs` checks rendered diagrams/figures, builds MkDocs
   strictly, and checks the generated site. Native CI separately builds
   the MLIR documentation and runs
   `python scripts/sync_mlir_reference.py --build-dir build --check`.
6. **Trusted CUDA:** outside the checkout, with `PYTHONPATH` unset and the
   actual repaired cp313 wheel installed alongside CUDA-enabled PyTorch:
   `python -m swage.env --json --check cuda`,
   `python "$repo/scripts/smoke_installed_wheel.py" --backend cuda`,
   then the same smoke in a second process with `--require-persistent-hit`
   and the same private `SWAGE_CACHE_DIR`. Run
   `python "$repo/benchmarks/benchmark_fixed_runtime.py" --enforce --output fixed-runtime-slo.json`.
   The release workflow also runs `python/tests/mlir/test_runtime.py` against
   that installed wheel. Enforcement requires exactly NVIDIA RTX A6000 /
   `sm_86`; retain raw JSON even on failure.

See [Verification](docs/internals/verification.md) for the frozen SLO thresholds.
Local measurements that miss the warm-dispatch or throughput gates under
load do not qualify a release. The trusted installed-wheel run must pass;
neither private segmented qualification nor historical GPU snapshots replace it.

## Contributor paths

- Documentation: fix incorrect boundaries before improving presentation. Run
  `mkdocs build --strict` and `ruff check .`.
- Public Python frontend: work under `python/swage/`. The accepted AST and API
  are narrow fixed-vector elementwise contracts. Run the Python tier and native
  binding integration when emission changes.
- Native dialects and lowering: work under `include/swage/`, `lib/`, and
  `test/`. Run `ninja -C build check-swage`; run C++ or binding targets when
  their code changes.
- Runtime: public execution remains canonical fixed vector add or multiply on
  explicitly selected CUDA or Native CPU backends. Segmented helpers are private
  qualification. Runtime changes require the hosted tests and, where CUDA
  behavior changes, trusted GPU evidence.
- Benchmarks: preserve frozen inputs and gates. Prepare outside timing and
  commit raw evidence with the exact hardware and revision.

Start with [Compiler Pipeline](docs/internals/compiler-pipeline.md), then
use [Compiler Tools and Passes](docs/internals/compiler-tools.md) and
[Verification](docs/internals/verification.md) for the affected
surface.

## Pull requests

1. Branch from `main` and make one coherent change.
2. Run the smallest relevant test, then the full applicable tier.
3. Report files changed, semantic impact, tests run and skipped, GPU
   architecture used if any, limitations, and follow-up work.
4. Fill the pull request template with the same boundary information.
5. Leave unrelated work untouched. A maintainer reviews and merges.

## Reporting issues

Bug reports need a minimal reproducer, exact versions, and
`python -m swage.env --json` output. Performance reports also need hardware,
distribution, command, methodology, and raw measurements.

## Release checklist and rollback

The implemented `publish-pypi.yml` is a gate, not evidence that v0.5.2 has
been published or production-qualified. Complete these steps in order:

1. **Operator setup, separately authorized:** protect `main` and require
   `test (3.10)`, `test (3.11)`, `test (3.12)`, `test (3.13)`, `docs`,
   `CodeQL Python (3.13)`, `Native cp313 + CodeQL C/C++`,
   `Native ASan + UBSan`, and `dependency-review`.
   Protect `v*` tags and require signed tags. Configure the `pypi`
   environment with a required reviewer and the PyPI OIDC trusted publisher
   for this repository/workflow/environment. These are repository/account
   administration, not actions a contributor or automation should perform
   without separate authorization. Keep untrusted pull requests off the
   self-hosted GPU runner and away from publication credentials.
2. **Build-only rehearsal:** manually dispatch `publish-pypi` from the
   reviewed main-line revision. It checks a clean exact SHA, version 0.5.2,
   and ancestry from `main`, and uses the commit timestamp as
   `SOURCE_DATE_EPOCH`. It reuses the hosted Python/native/security workflows,
   builds four repaired manylinux wheels and one reproducible sdist, runs
   minimum-PyTorch CPU smoke on all four ABIs, and requires cp313
   source-tree/sdist wheel SHA-256 equality. The checker requires each wheel
   to be **less than 95,000,000 bytes**, validates provenance and licenses,
   rejects build-root leaks, absolute RPATH/RUNPATH, linked `libcuda.so.1`,
   bytecode, and private segmented Python modules, and verifies self-contained
   manylinux dependencies.
3. **Inspect complete dry-run evidence:** `release-distributions` must contain
   exactly `swage_compiler-0.5.2-cp310-cp310-manylinux_2_28_x86_64.whl`,
   the corresponding cp311/cp312/cp313 wheels, and
   `swage_compiler-0.5.2.tar.gz`. `release-artifact-evidence` contains
   `SHA256SUMS`, one `sbom.spdx.json`, per-ABI health/smoke/repair JSON, and
   cp313 reproducibility evidence. The SBOM scans unpacked copies of all five
   checked distributions and requires Swage metadata plus checksummed file
   inventory for each distribution; scanning unopened wheel archives misses
   their packages. `release-gpu-evidence` must show the
   aggregated cp313 wheel passing fixed runtime tests, CUDA health/smoke,
   second-process persistent cache reuse, and enforced A6000 SLOs.
   Manual dispatch **never attests or publishes**, even when every gate passes.
4. **Authorized signed-tag release:** only a push of protected annotated
   `v0.5.2` can proceed to publication. The workflow verifies tag-to-commit
   identity, main ancestry and protection, and GitHub's tag API
   `verification.verified == true` with reason `valid` and nonempty signature
   and payload. A lightweight tag, unsigned tag, unverifiable signature, or
   unprotected ref fails closed; tag text is not a substitute for
   cryptographic verification. Do not create or push a release tag without
   release authorization.
5. **Attest, approve, publish:** after the same artifact/security/CPU/GPU
   gates pass, signed-tag runs attest build provenance and the aggregate
   SPDX SBOM, retain `release-attestations`, and reach the protected `pypi`
   environment for reviewer approval and OIDC publication. All external
   workflow actions are pinned to immutable commits.
6. **Verify the published artifact:** use the successful signed-tag run's
   detached build-provenance bundle, rather than a stored attestation record:

   ```bash
   gh run download "$RELEASE_RUN_ID" --repo abhiksark/swage \
     --name release-attestations --dir attestations
   PROVENANCE_BUNDLE=<build-provenance-bundle-in-attestations>
   gh attestation verify <downloaded-wheel-or-sdist> \
     --bundle "$PROVENANCE_BUNDLE"
   ```

   Check `SHA256SUMS`, install from PyPI into a clean environment, and check
   build revision and selected backend health. Record filenames, sizes, hashes,
   reproducibility hash, SBOM/attestation identifiers, Python/PyTorch/glibc
   matrix, hardware, raw SLO results, and any skipped checks. No
   production-ready claim is valid before both installed-wheel CPU and trusted
   A6000 gates pass.

Rollback is immutable: with explicit publication authority, **yank a defective
v0.5.2 and ship v0.5.3 through the same gates** (updating the version-specific
release policy). Never replace files for an existing PyPI version. This
fixed-vector hardening does not complete the persistent research gate or
change the v0.6.0 mapping.

## License

Contributions are licensed under the [MIT License](LICENSE).
