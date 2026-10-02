<!-- docs/getting-started/installation.md -->

# Installation

Swage has two installation boundaries. The published `swage-compiler` wheel
contains the pure Python `swage` package. Compiler emission and execution also
require the native `mlir_swage` package from a build tree.

## Install the Python package

Python 3.10 or newer is required. The
[Support Matrix](../reference/support-matrix.md) lists which Python,
PyTorch, driver, and GPU combinations are tested and which are only
admitted. Install the base package from PyPI:

```bash
python -m pip install swage-compiler
```

The base package imports without PyTorch. Install the optional PyTorch
dependency when using metadata inference, CUDA launch, or the segmented
calls:

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
environment facts, capture kernel source, and check a kernel against the
kernel language. It cannot emit MLIR, launch a kernel, or run a segmented
call.

These pages describe the current source tree. The released `0.5.1` wheel
predates part of them:

- It checks the inputs of `emit_mlir()` and then imports the native package
  before it checks the parameter list and the body, so a wheel-only install
  reports the missing bindings instead of a kernel-language error.
- It rejects a kernel that has a docstring.
- Its `sl.load` and `sl.store` declare their keywords with `None` defaults.
- Its environment report has fewer fields.
- It has no `swage.segment_reduce` and no `swage.segment_softmax`.

The [changelog](https://github.com/abhiksark/swage/blob/main/CHANGELOG.md)
lists every change since that release under Unreleased. To match these
pages, install the package from a checkout with the editable install above.

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
binding requirements.

`build_llvm.sh` and `build_swage.sh` build the bindings for the `python`
found on `PATH`, which is the interpreter the commands on this page run.
Where `python` is absent or older than Python 3.10, they fall back to
`python3`. Each script prints the interpreter it chose as
`Python interpreter: <path>` and stops with an error when neither name is
Python 3.10 or newer. Install the binding requirements into that
interpreter.

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
from a mirror; the same digest check applies. The verified tarball is
unpacked into a temporary directory and renamed into place, so an
interrupted run leaves no source directory, and the script stops if a
leftover `llvm-project-<version>.src` directory is present.

The script records the verified digest in a `.swage-source-sha256` file
inside the source directory. When the source directory already exists, a
matching record is accepted without a download, a record with another digest
or an empty directory is an error, and a directory without the record is
used with a notice that the script did not verify it. Remove the source
directory and run the script again to replace it with a verified tree.

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
and the CUDA tests skip without a GPU. The segmented calls and the private
qualification helpers behind them also import `numpy`, which the binding
requirements above already install. Importing `swage`, capturing a kernel,
emitting MLIR, and launching the fixed vector add do not need it. The
second command imports `swage`
from the installed package, so it needs the editable install from the same
checkout. The released `0.5.1` wheel does not match the tests of the
current source tree.

If CMake is asked for `SWAGE_PYTHON_BINDINGS=ON` against an MLIR install
without Python bindings, configuration fails instead of silently omitting the
package.

### Copying the native package

Native packaging is not provided yet. There is no native wheel, and no
deployment step is tested: `cmake --install` carries install rules for
`mlir_swage` that the MLIR build functions generate, and no test or
workflow runs them.

The build-tree package is not relocatable as it is.
`build/python_packages/mlir_swage` holds absolute symbolic links into the
LLVM install and into the checkout, so a copy that keeps the links stops
importing where those paths do not exist. A copy that follows the links
does not depend on them:

```bash
mkdir -p /path/to/site
cp -rL build/python_packages/mlir_swage /path/to/site/
cp -r python/swage /path/to/site/
PYTHONPATH=/path/to/site python -m swage.env
```

This was checked by hand on the machine that built it, with only that
directory on `PYTHONPATH` and with the build tree, the LLVM install, and the
checkout hidden from the process. MLIR emission, a vector-add launch, a
private segmented sum, and reuse of the persistent cache by a second process
all worked. These limits apply:

- No test in the repository covers the copy, and it was not tried on
  another machine. The native libraries still load the C++ runtime, `libz`,
  and `libzstd` from outside the copy.
- The copy is larger than the build-tree package, because each link is
  replaced by the file it points to.
- The copy is not a checkout, so `python -m swage.env` prints
  `revision: None` and `llvm_pin: None`, and the copy does not reuse
  persistent cache entries that the checkout wrote.
- A native build contains third-party code.
  [`THIRD_PARTY_NOTICES.md`](https://github.com/abhiksark/swage/blob/main/THIRD_PARTY_NOTICES.md)
  must go with any copy that leaves the machine that built it.

Installation is complete when the relevant build and test commands succeed.
Continue with the [Quickstart](quickstart.md), or use
[Troubleshooting](troubleshooting.md) when a tool or package cannot be found.
