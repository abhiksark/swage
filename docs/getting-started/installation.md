<!-- docs/getting-started/installation.md -->

# Installation

Swage has two installation boundaries. The published `swage-compiler` wheel
contains the pure Python `swage` package. Compiler emission and execution also
require the native `mlir_swage` package from a build tree.

## Install the Python package

Python 3.10 or newer is required. Install the base package from PyPI:

```bash
python -m pip install swage-compiler
```

The base package imports without PyTorch. Install the optional PyTorch
dependency when using metadata inference or CUDA launch:

```bash
python -m pip install "swage-compiler[pytorch]"
```

For repository development, install the editable package and developer tools:

```bash
git clone https://github.com/abhiksark/swage
cd swage
python -m pip install -e ".[dev]"
```

The wheel does not contain compiler libraries, `swage-opt`, generated MLIR
bindings, or the native `mlir_swage` package. Native wheel packaging is
deferred. A wheel-only install can import `swage`, report package and
environment facts, and capture kernel source. It cannot emit MLIR or launch a
kernel.

## Build LLVM, MLIR, and Swage

The native build requires Linux x86-64, CMake 3.20 or newer, Ninja, and a
C++17 compiler. `scripts/fetch_llvm.sh` also uses `curl`, `sha256sum` or
`shasum`, and a `tar` that can extract `.tar.xz` archives.

The MLIR Python bindings are built by default and need Python packages to
build and to run. The pinned LLVM release lists them in its
`mlir/python/requirements.txt`. At the current pin that file requires:

- `nanobind>=2.9,<3.0`
- `PyYAML>=5.4.0,<=6.0.1`
- `typing_extensions>=4.12.2`
- `numpy>=1.19.5,<=2.1.2`
- `ml_dtypes>=0.1.0,<=0.6.0`, or `>=0.5.0,<=0.6.0` on Python 3.13 or newer

LLVM configuration stops if `nanobind` cannot be imported. `build_swage.sh`
runs the lit suite with the `lit` found on `PATH`, and the binding tests run
under `pytest`. The `dev` extra above provides `lit` and `pytest` but not the
binding requirements. `build_llvm.sh` configures the bindings for the
`python3` found on `PATH`, so `python` in the commands on this page must be
that interpreter.

With the default `RelWithDebInfo` build type, the pinned LLVM/MLIR build uses
about 25 GB. Build time depends on the machine. The hosted CI workflow uses
the smaller `Release` build type and notes about three hours for the first
LLVM build on a new pin, under a 350-minute limit. Later runs reuse the cached
install tree.

GPU execution additionally requires a CUDA-enabled PyTorch build, the NVIDIA
driver, and an NVIDIA GPU of compute capability 8.0 (`sm_80`) or newer. An
older device is rejected during compilation. Exact target-admission and
zero-work rules live in
[Runtime and Environment](../reference/runtime-environment.md).

```bash
./scripts/fetch_llvm.sh
LLVM_SRC="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}/src-$(cat cmake/llvm-version.txt)"
python -m pip install -r "$LLVM_SRC/mlir/python/requirements.txt" lit pytest
./scripts/build_llvm.sh
./scripts/build_swage.sh
```

The final command configures Swage, builds `swage-opt` and the native Python
bindings, and runs the lit suite. The LLVM pin is recorded in
`cmake/llvm-version.txt` and must not be changed as part of an unrelated
change.

`fetch_llvm.sh` downloads the source tarball of the pinned release from the
llvm-project GitHub releases and checks its SHA-256 against
`cmake/llvm-source-sha256.txt` before extraction, whether the tarball was
just downloaded or was already present. On a mismatch it stops and extracts
nothing. Set `SWAGE_LLVM_URL` to the full URL of that tarball to download it
from a mirror; the same digest check applies.

An existing install of the exact pinned LLVM/MLIR release can be selected
instead:

```bash
MLIR_DIR=/path/to/lib/cmake/mlir \
LLVM_DIR=/path/to/lib/cmake/llvm \
    ./scripts/build_swage.sh
```

CMake configuration compares the LLVM version of the install it finds with
`cmake/llvm-version.txt` and stops with an error for any other release.

The helper accepts these build-location and configuration overrides:

- `SWAGE_LLVM_HOME` changes the LLVM source, build, and install root.
- `SWAGE_LLVM_BUILD_TYPE=Release` selects a smaller release build. The default
  is `RelWithDebInfo`; assertions remain enabled.
- `SWAGE_LLVM_PYTHON_BINDINGS=OFF` omits MLIR Python bindings. Such an install
  cannot build `mlir_swage`.

## Use the native Python package

The native package is imported from `build/python_packages`, not from the
published wheel:

```bash
ninja -C build check-swage-python
PYTHONPATH=build/python_packages python -m pytest -q python/tests/mlir
```

`check-swage-python` supplies the build-tree `PYTHONPATH` itself. Both
commands need `pytest` and PyTorch, because several binding test modules
import `torch`. The hosted CI job installs a CPU-only PyTorch build for them,
and the CUDA tests skip without a GPU. The second command imports `swage`
from the installed package, so it needs the editable install from the same
checkout. The released `0.5.1` wheel does not match the tests on `main`.

If CMake is asked for `SWAGE_PYTHON_BINDINGS=ON` against an MLIR install
without Python bindings, configuration fails instead of silently omitting the
package.

Installation is complete when the relevant build and test commands succeed.
Continue with the [Quickstart](quickstart.md), or use
[Troubleshooting](troubleshooting.md) when a tool or package cannot be found.
