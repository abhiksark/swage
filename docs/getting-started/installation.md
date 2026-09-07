<!-- docs/getting-started/installation.md -->

# Installation

!!! warning "v0.5.2 publication pending"

    This page describes the implemented v0.5.2 native-wheel contract, not an
    already published or production-qualified release. The latest released
    tag is v0.5.1; its older artifacts do not provide this contract. The
    version-pinned PyPI commands below apply only after v0.5.2 passes the
    release gates and is published. Until then, use a checked release-candidate
    wheel or the source-build path below.

## Install a native wheel

On a supported interpreter and platform, install the native package without a
source checkout or local LLVM installation:

```bash
python -m pip install --only-binary=:all: "swage-compiler==0.5.2"
python -m swage.env --json --check native
```

The wheel includes public `swage` and self-contained private `mlir_swage`
compiler bindings and runtime libraries. An independently installed `mlir`
package is neither required nor a substitute. The base package imports and
emits MLIR with an explicit signature without PyTorch. For metadata inference
or either launch backend, install the optional runtime extra:

```bash
python -m pip install --only-binary=:all: "swage-compiler[pytorch]==0.5.2"
python -m swage.env --json --check cpu
```

The release targets Linux x86-64 with glibc 2.28 or newer and regular-GIL
CPython 3.10–3.13. PyTorch is optional, not bundled, and constrained to
`torch>=2.6,<3`. Choose a CUDA-enabled PyTorch build using the
[PyTorch installation selector](https://pytorch.org/get-started/locally/) if
using CUDA; the extra alone does not promise a CUDA-enabled build. Then run:

```bash
python -m swage.env --json --check cuda
```

See the authoritative [support matrix](../reference/runtime-environment.md#support-matrix)
for ABI exclusions, driver requirements, and the A6000/`sm_86` release
qualification boundary. Other admitted CUDA targets are best-effort, not
release-qualified. CPU and CUDA are explicit choices; a failure never switches
backends. Compiler executables such as `swage-opt` remain source-build tools;
private segmented Python modules are excluded from wheels.

## Verify downloaded artifacts

After publication, download a wheel without installing or building it:

```bash
python -m pip download --only-binary=:all: --no-deps \
    --dest verified-wheel "swage-compiler==0.5.2"
```

Use the successful **signed-tag** `publish-pypi` run for `v0.5.2`, checking its
repository and commit identity. Set `RELEASE_RUN_ID` to that run's numeric ID.
Download its checksums and attestation bundles with an authenticated GitHub CLI:

```bash
gh run list --repo abhiksark/swage --workflow publish-pypi.yml --event push
gh run download "$RELEASE_RUN_ID" --repo abhiksark/swage \
    --name release-artifact-evidence --dir verified-wheel
gh run download "$RELEASE_RUN_ID" --repo abhiksark/swage \
    --name release-attestations --dir verified-wheel/attestations
cd verified-wheel
sha256sum --check --ignore-missing SHA256SUMS
```

This checks the downloaded wheel against the aggregate's manifest; the other
four distributions are intentionally absent. Set `PROVENANCE_BUNDLE` to the
build-provenance bundle in `attestations` (not the SPDX SBOM bundle), then verify
each downloaded wheel's signed provenance:

```bash
for wheel in ./*.whl; do
    gh attestation verify "$wheel" --repo abhiksark/swage \
        --bundle "$PROVENANCE_BUNDLE"
done
```

Inspect the verified source revision and signer workflow; they must identify
the intended release commit and `.github/workflows/publish-pypi.yml`. The
evidence also includes one SPDX JSON SBOM. Manual dry runs produce artifact
evidence but never publish or attest; no signed-tag attestation exists until
the release gates pass. Checksums alone are integrity checks, not proof of a
trusted publisher. Do not install on failed or mismatched verification.

Swage compiles native code from trusted inputs and is **not a sandbox** for
untrusted kernels or compiler artifacts. See the
[security policy](https://github.com/abhiksark/swage/blob/main/SECURITY.md).

## Build from source

A checkout supports a normal native build. An extracted source distribution
supports a native wheel build with explicit, verified source provenance.
The native build requires Linux x86-64, CMake 3.20 or newer, Ninja, and a C++17
compiler. LLVM/MLIR must be exactly **22.1.8** (`llvmorg-22.1.8`), as recorded
in `cmake/llvm-version.txt`; CMake rejects a mismatched installation. The first
pinned toolchain build uses about 25 GB and can take about an hour.

For a checkout, install the frontend and developer tools without triggering a
native wheel build:

```bash
git clone https://github.com/abhiksark/swage
cd swage
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -Cwheel.cmake=false
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
SWAGE_BUILD_TYPE=Release ./scripts/build_swage.sh
```

The final checkout helper configures Swage, builds `swage-opt` and native
bindings, and runs the lit suite. For an extracted sdist, use the
[provenance-bearing wheel build](https://github.com/abhiksark/swage/blob/main/CONTRIBUTING.md#local-native-wheel-dry-run);
it has no Git checkout from which to infer identity. An existing exact-pin
LLVM/MLIR installation can be selected for a checkout build instead:

```bash
MLIR_DIR=/path/to/lib/cmake/mlir \
LLVM_DIR=/path/to/lib/cmake/llvm \
SWAGE_BUILD_TYPE=Release ./scripts/build_swage.sh
```

The toolchain helper accepts these overrides:

- `SWAGE_LLVM_HOME`: LLVM source, build, and install root.
- `SWAGE_LLVM_BUILD_TYPE=Release`: smaller release build instead of the default
  `RelWithDebInfo`; assertions remain enabled.
- `SWAGE_PYTHON_EXECUTABLE`: interpreter whose native bindings are built
  (default `python3`); it must match the interpreter using those bindings.
- `SWAGE_LLVM_PYTHON_BINDINGS=OFF`: omit MLIR Python bindings; such an install
  cannot build `mlir_swage`.

## Use source-built bindings

Normal source builds retain `build/python_packages/mlir_swage`. Only this
source-build route needs a build-tree `PYTHONPATH`:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python -m swage.env --json --check native
PYTHONPATH=build/python_packages python examples/fixed_vector_add.py --backend cpu
PYTHONPATH=build/python_packages python examples/fixed_vector_multiply.py --backend cpu
```

`check-swage-python` supplies its own build-tree path. Explicit
`SWAGE_PYTHON_BINDINGS=ON` against an MLIR install without bindings fails
configuration rather than silently omitting the package.

Native wheel builds use the pinned `scikit-build-core==1.0.3` backend and
`nanobind==2.15.0`. They require Release mode, a 40-lowercase-hex
`SWAGE_SOURCE_REVISION`, and an explicit `SWAGE_SOURCE_CLEAN` value; do not
invent a clean identity for an extracted sdist or modified checkout. See
[Contributing](https://github.com/abhiksark/swage/blob/main/CONTRIBUTING.md)
for the native-wheel dry run and release gates. Frontend-only editable
installation is not a substitute for native bindings when emitting or launching.

Continue with the [Quickstart](quickstart.md), or use
[Troubleshooting](troubleshooting.md) when a tool or package cannot be found.
