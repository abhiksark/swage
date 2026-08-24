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
C++17 compiler. The pinned LLVM/MLIR build uses about 25 GB and can take about
an hour on its first build.

GPU execution additionally requires a CUDA-enabled PyTorch build and the
NVIDIA driver. Nonzero launches perform native compilation and require a
device with compute capability 8.0 or newer; the compiler currently accepts
exact targets from `sm_80` through `sm_129`. The validated `n == 0` public
launch returns before native target admission.

```bash
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh
```

The final command configures Swage, builds `swage-opt` and the native Python
bindings, and runs the lit suite. The LLVM pin is recorded in
`cmake/llvm-version.txt` and must not be changed as part of an unrelated
change.

An existing install of the exact pinned LLVM/MLIR release can be selected
instead:

```bash
MLIR_DIR=/path/to/lib/cmake/mlir \
LLVM_DIR=/path/to/lib/cmake/llvm \
    ./scripts/build_swage.sh
```

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

`check-swage-python` supplies the build-tree `PYTHONPATH` itself. If CMake is
asked for `SWAGE_PYTHON_BINDINGS=ON` against an MLIR install without Python
bindings, configuration fails instead of silently omitting the package.

Installation is complete when the relevant build and test commands succeed.
Continue with the [Quickstart](../quickstart.md), or use
[Troubleshooting](troubleshooting.md) when a tool or package cannot be found.
