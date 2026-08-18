# Quickstart

Everything on this page works today, on a machine with or without a GPU.

## Prerequisites

- Linux x86-64, Python ≥ 3.10, CMake ≥ 3.20, Ninja, a C++17 compiler
  (GCC ≥ 7.4 or Clang ≥ 5)
- ~25 GB of disk for the one-time LLVM build

## Python package (no LLVM required)

```bash
git clone https://github.com/abhiksark/swage
cd swage
make setup                     # pip install -e ".[dev]"
python -m pytest tests/python -q
python -m swage.env            # environment diagnostics
```

`swage.env` reports your Python, PyTorch, CUDA, GPU, and the LLVM pin —
paste its output into bug reports.

## MLIR components

Swage builds out-of-tree against one exact LLVM/MLIR release, pinned in
`cmake/llvm-version.txt`:

```bash
./scripts/fetch_llvm.sh        # downloads the pinned source tarball
./scripts/build_llvm.sh        # ~1 hour once; installs to ~/.swage/llvm/
./scripts/build_swage.sh       # builds swage-opt, runs the lit suite
```

Already have a matching MLIR install? Skip the first two steps and point
the build at it:

```bash
MLIR_DIR=/path/to/lib/cmake/mlir LLVM_DIR=/path/to/lib/cmake/llvm \
    ./scripts/build_swage.sh
```

Useful overrides for `build_llvm.sh`: `SWAGE_LLVM_BUILD_TYPE=Release`
(smaller/faster than the default `RelWithDebInfo`; assertions stay on),
`SWAGE_LLVM_PYTHON_BINDINGS=OFF` (skip the MLIR Python bindings),
`SWAGE_LLVM_HOME=/elsewhere` (move the ~25 GB out of `$HOME`).

## Trying the dialect

```bash
./build/bin/swage-opt test/Dialect/Swage/roundtrip.mlir
```

```mlir
func.func @segment_roundtrip(%values: memref<?xf32>, %offsets: memref<?xi32>) -> index {
  %sid = swage.segment_id 0
  %seg = swage.make_segment %values, %offsets, %sid
      : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  %n = swage.extent %seg : !swage.segment<f32>
  return %n : index
}
```

Re-run the test suites at any time:

```bash
ninja -C build check-swage     # MLIR lit tests
python -m pytest tests/python  # Python tests
make lint                      # ruff
```
