# Quickstart

The build and compile-only paths work without a GPU. The fixed vector-add
walkthrough also requires Linux, an NVIDIA GPU, `libcuda`, and a CUDA-enabled
PyTorch installation.

## Prerequisites

- Linux x86-64, Python ≥ 3.10, CMake ≥ 3.20, Ninja, a C++17 compiler
  (GCC ≥ 7.4 or Clang ≥ 5)
- ~25 GB of disk for the one-time LLVM build
- For execution only: an NVIDIA GPU supported by the installed driver and a
  CUDA-enabled PyTorch build

## Python package (no LLVM required)

```bash
git clone https://github.com/abhiksark/swage
cd swage
make setup                     # pip install -e ".[dev]"
python -m pytest tests/python -q
python -m swage.env            # environment diagnostics
```

`swage.env` reports your Python, PyTorch, GPU, LLVM pin, PyTorch CUDA build,
and actual CUDA driver version. Include its output in bug reports.

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

## Native Python bindings

The `mlir_swage` package is a native build artifact, not part of the pip
package. It requires the pinned LLVM/MLIR install with
`MLIR_ENABLE_BINDINGS_PYTHON=ON`, which `build_llvm.sh` enables by default.
The integration suite imports PyTorch. After building Swage, install the
optional dependency and run the dedicated target:

```bash
pip install -e ".[pytorch]"
ninja -C build check-swage-python
```

The target sets `PYTHONPATH=build/python_packages` and runs the integration
tests. To run one directly, use the same build-tree path:

```bash
PYTHONPATH=build/python_packages python -m pytest -q python/tests/mlir
```

If the LLVM install or existing CMake build was configured with bindings off,
rebuild the pinned LLVM/MLIR with bindings on and reconfigure Swage with
`-DSWAGE_PYTHON_BINDINGS=ON`. CMake reports an error if the selected MLIR
install lacks Python bindings.

```bash
SWAGE_LLVM_PYTHON_BINDINGS=ON ./scripts/build_llvm.sh
cmake -G Ninja -S . -B build \
    -DMLIR_DIR=/path/to/lib/cmake/mlir \
    -DLLVM_DIR=/path/to/lib/cmake/llvm \
    -DSWAGE_PYTHON_BINDINGS=ON
```

In `ci-cpp`, a cold LLVM cache fetches and stores MLIR's Python requirements
before the native build. Cold and warm cache jobs install that stored file
plus `pytest` and `lit`; no Python environment is restored from the cache.

## Execute fixed vector add

The installed PyTorch build must have CUDA support. Confirm the build, driver,
GPU, and compute capability before running the example:

```bash
python -m swage.env
```

The committed
[`examples/fixed_vector_add.py`](https://github.com/abhiksark/swage/blob/main/examples/fixed_vector_add.py)
contains the full kernel and launch. Run it from a clean checkout after the
native build, with dumps and cache files isolated in a temporary directory:

```bash
export SWAGE_WALKTHROUGH_DIR="$(mktemp -d)"
export SWAGE_CACHE_DIR="$SWAGE_WALKTHROUGH_DIR/cache"
export SWAGE_DUMP_DIR="$SWAGE_WALKTHROUGH_DIR/dumps"
export SWAGE_DUMP_MLIR=1
export SWAGE_DUMP_PTX=1

PYTHONPATH=build/python_packages python examples/fixed_vector_add.py
```

The program first calls `emit_mlir()` and prints the verified semantic MLIR.
It then calls the keyword-only `launch()` API. The runtime validates the
tensors, scalar, block size, and grid before reading any data pointers. It
lowers a clone of the semantic module through GPU, NVVM, and LLVM, emits PTX
for the active device's exact `sm_*` target, loads it through the CUDA Driver
API, and launches it on `torch.cuda.current_stream()`. The final assertion
consumes the asynchronous result and compares it with `torch.add`.

Inspect the lowered MLIR and PTX debug dumps:

```bash
find "$SWAGE_DUMP_DIR" -maxdepth 1 -type f -print
sed -n '1,120p' "$SWAGE_DUMP_DIR"/*.mlir
sed -n '1,80p' "$SWAGE_DUMP_DIR"/*.ptx
```

The persistent cache uses one digest-named directory per specialization. Its
metadata records the complete key and digests for the lowered MLIR and PTX:

```bash
find "$SWAGE_CACHE_DIR" -maxdepth 2 -type f -print
find "$SWAGE_CACHE_DIR" -name metadata.json \
    -exec python -m json.tool {} \;
```

Run the example again with the same environment. The second process verifies
and reuses the disk artifact, then loads a module for its current CUDA
context. Dirty or unidentified compiler builds deliberately use process-local
caching only, so a dirty checkout does not create persistent cache files.

The M3 boundary is intentionally narrow:

- Tensors must be contiguous rank-one f32 CUDA tensors on the same current
  device.
- `n` must be a nonnegative i32 no larger than any tensor, and `BLOCK` must be
  legal for the active device.
- The one-dimensional grid must equal `(ceildiv(n, BLOCK),)`. An empty launch
  uses `n == 0` and `grid == (0,)`.
- Launch never copies, casts, synchronizes, changes devices, creates a CUDA
  context, or falls back.

`emit_mlir()` remains available as a compile-only operation. It can infer
descriptors from supported CPU or CUDA tensor metadata, or accept explicit
`sl.pointer(sl.float32)` and `sl.int32` descriptors without PyTorch. There is
no public `emit_ptx()` API.

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
