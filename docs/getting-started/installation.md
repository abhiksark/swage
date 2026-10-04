<!-- docs/getting-started/installation.md -->

# Installation

!!! warning "v0.5.2 publication pending"

    This page describes the implemented `v0.5.2` native-wheel contract, not
    an already published or production-qualified release. The latest
    released tag is `v0.5.1`, a pure Python wheel that does not provide this
    contract; [What the released 0.5.1 wheel lacks](#what-the-released-051-wheel-lacks)
    lists the differences. The version-pinned PyPI commands below apply
    only after `v0.5.2` passes the release gates and is published. Until
    then, [build a local native wheel](#build-a-local-native-wheel) or use
    the [source build](#build-from-source).

The distribution `swage-compiler` is one native wheel per CPython version.
Each wheel holds public `swage`, the private `mlir_swage` compiler bindings,
the `libSwageRuntime.so` runtime library, and the private segmented modules
that `swage.segment_reduce` and `swage.segment_softmax` import. There is no
separate native distribution. Building a source distribution and the
frontend-only editable install of a checkout need no CMake; neither provides
the native bindings.

## Install a native wheel

On a supported interpreter and platform, install the native package without
a source checkout or a local LLVM installation:

```bash
python -m pip install --only-binary=:all: "swage-compiler==0.5.2"
python -m swage.env --json --check native
```

The wheel needs no independently installed `mlir` package, and such a
package is no substitute for the bundled bindings. The base package imports
and emits MLIR with an explicit signature without PyTorch. For metadata
inference, either launch backend, or the segmented calls, install the
optional runtime extra, which declares `torch>=2.6,<3` and `numpy`:

```bash
python -m pip install --only-binary=:all: "swage-compiler[pytorch]==0.5.2"
python -m swage.env --json --check cpu
```

The release targets Linux x86-64 with glibc 2.28 or newer
(`manylinux_2_28`) and regular-GIL CPython 3.10 to 3.13. PyTorch is
optional and never bundled. If you use CUDA, choose a CUDA-enabled PyTorch
build with the
[PyTorch installation selector](https://pytorch.org/get-started/locally/);
the extra alone does not promise a CUDA-enabled build. Then run:

```bash
python -m swage.env --json --check cuda
```

The [Support Matrix](../reference/support-matrix.md) lists the ABI
exclusions, the tested driver, and the A6000 (`sm_86`) qualification
boundary. Other admitted CUDA targets are best effort, not release
qualified. CPU and CUDA are explicit choices, and a failure never switches
backends. Compiler executables such as `swage-opt` remain source-build
tools; the wheel does not hold them.

## What the released 0.5.1 wheel lacks

The `0.5.1` wheel on PyPI was built from the `v0.5.1` tag. It is pure
Python, and it predates most of these pages:

- It has no `mlir_swage` bindings, so it cannot emit MLIR or launch a
  kernel.
- It launches only the vector add on CUDA. It has no `backend=` argument,
  no CPU backend, no multiply, and no `float16`, `float8_e4m3fn`, or
  `float8_e5m2` support.
- It has no `swage.segment_reduce`, no `swage.segment_softmax`, no
  `python -m swage.compile`, and no `SWAGE_ARTIFACT_DIR`.
- It exports `swage.jit` and `swage.CompilationError` only: no
  `swage.SwageError` and no `swage.BackendUnavailableError`.
- Its `python -m swage.env` prints a short text report and has no `--json`
  and no `--check`.
- Its `emit_mlir()` imports the native package before it checks the
  parameter list and the body, so without bindings it reports the missing
  bindings instead of a kernel-language error.
- It rejects a kernel that has a docstring.
- Its `sl.load` and `sl.store` declare their keywords with `None` defaults.

The [changelog](https://github.com/abhiksark/swage/blob/main/CHANGELOG.md)
lists every change since that release under Unreleased.

## Verify downloaded artifacts

After publication, download a wheel without installing or building it:

```bash
python -m pip download --only-binary=:all: --no-deps \
    --dest verified-wheel "swage-compiler==0.5.2"
```

Use the successful **signed-tag** `publish-pypi` run for `v0.5.2`, checking
its repository and commit identity. Set `RELEASE_RUN_ID` to that run's
numeric ID. Download its checksums and attestation bundles with an
authenticated GitHub CLI:

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
build-provenance bundle in `attestations` (not the SPDX SBOM bundle), then
verify each downloaded wheel's signed provenance:

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

A checkout supports two native builds: a build tree for compiler work, and
a local native wheel. Both require Linux x86-64, CMake 3.20 or newer, Ninja,
and a C++17 compiler, and both build against exactly LLVM/MLIR **22.1.8**
(`llvmorg-22.1.8`), as recorded in `cmake/llvm-version.txt`. CMake compares
the LLVM version of the install it finds with that file and stops with an
error for any other release. `scripts/fetch_llvm.sh` also uses `curl`,
`sha256sum` or `shasum`, and a `tar` that can extract `.tar.xz` archives.

First install the frontend and the developer tools without triggering a
native wheel build:

```bash
git clone https://github.com/abhiksark/swage
cd swage
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -Cwheel.cmake=false
```

The MLIR Python bindings are built by default and need Python packages to
build and to run. The pinned LLVM release lists them in its
`mlir/python/requirements.txt`. At the current pin that file requires:

- `nanobind>=2.9,<3.0`
- `PyYAML>=5.4.0,<=6.0.1`
- `typing_extensions>=4.12.2`
- `numpy>=1.19.5,<=2.1.2`
- `ml_dtypes>=0.1.0,<=0.6.0`, or `>=0.5.0,<=0.6.0` on Python 3.13 or newer

LLVM configuration stops if `nanobind` cannot be imported. The `dev` extra
provides `lit`, `pytest`, `build`, and the pinned `scikit-build-core` and
`nanobind`, but not the other binding requirements. Fetch the source,
install the requirements, and build LLVM and Swage:

```bash
./scripts/fetch_llvm.sh
LLVM_SRC="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}/src-$(cat cmake/llvm-version.txt)"
python -m pip install -r "$LLVM_SRC/mlir/python/requirements.txt"
./scripts/build_llvm.sh
./scripts/build_swage.sh
```

`build_llvm.sh` and `build_swage.sh` build the bindings for the `python`
found on `PATH`, which is the interpreter the commands on this page run.
Where `python` is absent or older than Python 3.10, they fall back to
`python3`. Each script prints the interpreter it chose as
`Python interpreter: <path>` and stops with an error when neither name is
Python 3.10 or newer. Install the binding requirements into that
interpreter. The final command configures Swage, builds `swage-opt` and the
native bindings, and runs the lit suite.

With the default `RelWithDebInfo` build type, the pinned LLVM/MLIR build
uses about 25 GB. Build time depends on the machine. The hosted CI workflow
uses the smaller `Release` build type and notes about three hours for the
first LLVM build on a new pin, under a 350-minute limit. Later runs reuse the
cached install tree.

`fetch_llvm.sh` downloads the source tarball of the pinned release from the
llvm-project GitHub releases and checks its SHA-256 against
`cmake/llvm-source-sha256.txt` before extraction, whether the tarball was
just downloaded or was already present. On a mismatch it stops and extracts
nothing. Set `SWAGE_LLVM_URL` to the full URL of that tarball to download it
from a mirror; the same digest check applies. The verified tarball is
unpacked into a temporary directory and renamed into place, so an
interrupted run leaves no source directory, and the script stops if a
leftover `llvm-project-<version>.src` directory is present. The script
records the verified digest in a `.swage-source-sha256` file inside the
source directory. When the source directory already exists, a matching
record is accepted without a download, a record with another digest or an
empty directory is an error, and a directory without the record is used
with a notice that the script did not verify it.

An existing install of the exact pinned LLVM/MLIR release can be selected
instead:

```bash
MLIR_DIR=/path/to/lib/cmake/mlir \
LLVM_DIR=/path/to/lib/cmake/llvm \
    ./scripts/build_swage.sh
```

The helpers accept these build-location and configuration overrides:

- `SWAGE_LLVM_HOME`: the LLVM source, build, and install root (default
  `~/.swage/llvm`).
- `SWAGE_LLVM_BUILD_TYPE=Release`: a smaller release build of LLVM instead
  of the default `RelWithDebInfo`; assertions remain enabled.
- `SWAGE_PYTHON_EXECUTABLE`: the interpreter whose MLIR Python bindings
  `build_llvm.sh` builds; it must match the interpreter that uses them.
- `SWAGE_LLVM_PYTHON_BINDINGS=OFF`: omit the MLIR Python bindings. Such an
  install cannot build `mlir_swage`.
- `SWAGE_BUILD_TYPE` and `SWAGE_BUILD_DIR`: the build type (default
  `RelWithDebInfo`) and the directory (default `build`) of the Swage build
  tree.

GPU execution additionally requires a CUDA-enabled PyTorch build, the
NVIDIA driver, and an NVIDIA GPU of compute capability 8.0 (`sm_80`) or
newer. An older device is rejected during compilation.
[Runtime and Environment](../reference/runtime-environment.md) states the
exact target-admission and zero-work rules.

### Use the build tree

A build tree keeps `build/python_packages/mlir_swage`. Only this route needs
a build-tree `PYTHONPATH`:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python -m swage.env --json --check native
PYTHONPATH=build/python_packages python examples/fixed_vector_add.py --backend cpu
PYTHONPATH=build/python_packages python examples/fixed_vector_multiply.py --backend cpu
```

`check-swage-python` supplies the build-tree path itself. Several binding
test modules import `torch`; the hosted CI job installs a CPU-only PyTorch
build for them, and the CUDA tests skip without a GPU. The segmented calls
and the private qualification helpers behind them also import `numpy`,
which the binding requirements above install. The `swage` that these
commands import is the editable install of the same checkout. If CMake is
asked for `SWAGE_PYTHON_BINDINGS=ON` against an MLIR install without Python
bindings, configuration fails instead of silently omitting the package.

A build tree is not relocatable: `build/python_packages/mlir_swage` holds
absolute symbolic links into the LLVM install and into the checkout. Build a
wheel to move the bindings to another environment.

### Build a local native wheel

A local wheel is built the way the `fixed-runtime-slo` job of
`.github/workflows/ci-gpu.yml` builds it: from a clean committed checkout,
without build isolation, against the pinned LLVM install that
`build_llvm.sh` wrote for the same `python`. Keep generated output outside
the checkout:

```bash
test -z "$(git status --porcelain)"
work="$(mktemp -d)"
llvm="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}/install-$(cat cmake/llvm-version.txt)"
python -m build --wheel --no-isolation --outdir "$work/dist" \
    -Cbuild-dir="$work/build" \
    -Ccmake.define.MLIR_DIR="$llvm/lib/cmake/mlir" \
    -Ccmake.define.LLVM_DIR="$llvm/lib/cmake/llvm" \
    -Ccmake.define.SWAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
    -Ccmake.define.SWAGE_SOURCE_CLEAN=true
python -m venv "$work/venv"
"$work/venv/bin/python" -m pip install "$work"/dist/*.whl
(cd "$work" && env -u PYTHONPATH "$work/venv/bin/python" -m swage.env --json --check native)
```

The `dev` extra supplies the build tools that `--no-isolation` needs, at
the pinned versions `scikit-build-core==1.0.3` and `nanobind==2.15.0`. A
wheel build requires the `Release` build type, which `pyproject.toml` sets,
a `SWAGE_SOURCE_REVISION` of 40 lowercase hexadecimal digits, and an
explicit `SWAGE_SOURCE_CLEAN` of `true` or `false`; CMake stops without
them. Record `false` for a modified tree and never label it clean: such a
wheel cannot qualify a release. Install
PyTorch into the new environment separately when a launch or a segmented
call needs it, and run checks from outside the checkout with `PYTHONPATH`
unset, so that the installed packages are the ones imported.

A wheel built this way carries the plain `linux_x86_64` platform tag and is
local evidence, not a `manylinux` release artifact. The release workflow
builds and repairs its wheels inside the digest-pinned
`manylinux_2_28_x86_64` image of `.github/workflows/publish-pypi.yml`. The
`build-and-test` job of `ci-cpp` builds and installs a local CPython 3.13
wheel the same way on every pull request, and runs the CPU checks and the
whole `python/tests/mlir` suite against it, with every CUDA test skipped.

## Run from an artifact

A host that only serves `swage.segment_reduce` and `swage.segment_softmax`
can run them from an artifact directory with no compiler loaded. It needs
three things:

- `swage` from the same source revision as the host that wrote the
  artifact: the native wheel, whose bindings a process with an artifact
  selected does not import, or the frontend-only install of that checkout.
  The released `0.5.1` wheel has no segmented calls and does not read
  artifacts.
- PyTorch 2.6 or newer with CUDA, and `numpy`.
- An artifact directory for the processor of its GPU, written on a host
  with the native bindings:

    ```bash
    python -m swage.compile --target sm_86 --output /path/to/artifact
    ```

On the serving host, name the directory and check it:

```bash
export SWAGE_ARTIFACT_DIR=/path/to/artifact
python -m swage.env
```

The report ends with an `artifact` line that names the directory, or gives
the reason it is rejected. Emitting MLIR and launching the fixed kernels
still need the native bindings.
[Running Without the Compiler](../user-guide/deployment.md) states what an
artifact holds, the rule for its permissions, and its limits, among them
that its runtime library has been built for Linux x86-64 only.

Installation is complete when the relevant health checks and test commands
succeed. Continue with the [Quickstart](quickstart.md), or use
[Troubleshooting](troubleshooting.md) when a tool or package cannot be
found.
