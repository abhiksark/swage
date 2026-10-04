<!-- README.md -->

![Swage logo](https://raw.githubusercontent.com/abhiksark/swage/main/docs/assets/images/swage-logo.png)

# Swage

[![ci-python](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml)
[![ci-cpp](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml)
[![GPU runtime](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml)

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded MLIR/LLVM compiler with a
deliberately narrow public execution contract in two parts: the canonical
fixed vector add and multiply, launched on an explicitly selected CPU or CUDA
backend, and two segmented calls, `swage.segment_reduce` and
`swage.segment_softmax`, which run fixed programs over variable-sized
segments on one CUDA device. Everything else, including segment programs
that a caller writes, is private research: it is not public API, and it may
change or go away.

## Current capability boundary

The latest tagged release is `v0.5.1`, a pure Python wheel on PyPI. The
`v0.5.2` source tree implements the native-wheel contract below. It is not
published: publication and production qualification wait on the
installed-wheel release gates, and source changes alone are not release
qualification.

### Public today

- **Fixed vector add and multiply.** Canonical fixed vector add and multiply
  are the only kernels the kernel language launches. Contiguous rank-one
  tensors may use `float32`, `float16`, `float8_e4m3fn`, or `float8_e5m2`;
  both inputs and the output must match. Low-precision arithmetic widens to
  FP32 and rounds back to the storage dtype, and FP8 conversion runs in the
  compiled kernel, including on `sm_86`. A kernel contains exactly one
  `x + y` or `x * y` value operation; chains, floating vector and scalar
  arithmetic, broadcasting, and matrix multiplication are not accepted.
- **Compile-only emission.** The restricted Python frontend emits verified
  MLIR through the native bindings that the `v0.5.2` wheel bundles, with no
  GPU and no PyTorch.
- **Explicit backends.** `kernel.launch(..., backend="cuda")` lowers through
  LLVM NVPTX and enqueues through the CUDA Driver API on the current
  PyTorch stream. `kernel.launch(..., backend="cpu")` lowers the same
  program to a synchronous, process-local Native LLVM JIT entry for CPU
  tensors. Selection is explicit and fail-closed: no backend falls back to
  the other.
- **Segmented reduction.** `swage.segment_reduce(values, offsets, kind, *,
  out=None)` returns the sum, maximum, minimum, or mean of every segment,
  over `float32` or `float64` values.
- **Segmented softmax.** `swage.segment_softmax(values, offsets, *,
  out=None)` returns the softmax within every segment, over `float32`
  values only: the device has no 64-bit `exp2`, so `float64` is refused.
- **Ahead-of-time artifacts.** `python -m swage.compile --target <processor>
  --output <directory>` writes the kernels of both segmented calls for one
  NVPTX processor without a GPU. A process that sets
  `SWAGE_ARTIFACT_DIR=<directory>` runs the two calls from that directory
  with no compiler loaded. The artifact runtime library is built for Linux
  x86-64 only.
- **Diagnostics.** `python -m swage.env --json` reports the environment, and
  `--check native`, `--check cpu`, or `--check cuda` exits non-zero when the
  selected component is unavailable.

Both segmented calls take values of rank one, or `[N, D]` rows that are
reduced or normalized per column (a reduction of `[N, D]` values returns
`[S, D]` for `S` segments). They take `int32` or `int64` offsets, with fewer
than `2**31` rows and fewer than `2**31` segments. They share these limits:

- They run on the current CUDA device only.
- They record no gradient: a tensor that requires grad is refused.
- They are refused under CUDA graph capture.
- They validate and classify the offsets on the host at every call, so with
  offsets that change on every call a reduction is slower than
  `torch.segment_reduce`. The `pytorch` extra declares `numpy` for this.
- An empty segment gives the identity of its kind: a sum of `0`, a maximum
  of negative infinity, a minimum of positive infinity, and a mean of NaN.

[Segmented Calls](docs/user-guide/segmented-calls.md) states the contract
and the cost, and
[Running Without the Compiler](docs/user-guide/deployment.md) states what an
artifact holds and when it is refused. The `swage` dialect and `swage-opt`
are available to compiler contributors through source builds.

### Private qualification

Everything outside the public list is private research. The two segmented
calls run fixed programs through part of this machinery with default limits
and expose none of its controls.

- Canonical segmented sum, max, and stable ragged softmax run through
  sequential CPU oracles and one-CTA GPU paths.
- Capture-free, single-stage sum, max, and min programs over f32 or f64
  values, including element expressions and map chains, run through host
  classification, direct warp and CTA work, one fused mixed kernel, and
  split-CTA partial and merge kernels. Prepared launches and split-CTA and
  mixed schedules chosen by hand stay private.
- A persistent task queue for the identity sum has correctness evidence, but
  its predeclared performance gate failed; it is neither qualified nor
  public.
- The planner and its passes, the `swage-opt` flags, and everything under
  `swage._*` are private.

The frozen NVIDIA RTX A6000 `sm_86` mixed-policy record has a
mixed-to-best-pure ratio of `0.939394`, below its predeclared `1.05` limit.
Exact and nontrivial f32 split sums match PyTorch and the CPU oracle on the
same GPU. [Task Execution](docs/internals/task-execution.md) and
[Benchmarks](docs/internals/benchmarks.md) give the records and their
limits.

### Planned

- Public segment syntax, and public launch of a segment program that the
  caller writes.
- Packing several short segments into one warp, split softmax, reusable
  device queues, qualified persistent scheduling, and broader policies.

Private qualification is not a public segmented runtime: the public calls
expose two of its programs and none of its controls. Current status is
backed by the repository's executable tests and committed benchmark
records.

## Native wheel and source builds

The `v0.5.2` distribution, `swage-compiler`, is one native wheel per
CPython version, 3.10 to 3.13, for Linux x86-64 with glibc 2.28 or newer
(`manylinux_2_28`). Each wheel holds public `swage` with its type stubs, the
private `mlir_swage` compiler bindings, the `libSwageRuntime.so` runtime
library, and the private segmented modules that the two public calls
import. It needs no external MLIR installation and no CUDA toolkit compiler
at runtime. PyTorch is optional (`torch>=2.6,<3` through the `pytorch`
extra, which also declares `numpy`). There are no macOS, Windows, musl,
other-architecture, or free-threaded wheels.

The release workflow checks CPU execution on every wheel ABI. The
continuously qualified CUDA configuration is the NVIDIA RTX A6000 (`sm_86`).
On that GPU, the release `gpu` job and the `fixed-runtime-slo` job of the
GPU workflow are configured to qualify the segmented calls and the artifact
path from the installed wheel with `scripts/qualify_installed_segments.sh`.
Other admitted NVIDIA targets are best effort.

The released `0.5.1` wheel on PyPI is pure Python. It has no native
bindings, no CPU backend, no multiply or low-precision dtypes, no segmented
calls, and no artifact path.
[Installation](docs/getting-started/installation.md) lists the differences.

After `v0.5.2` passes its publication gates:

```bash
python -m pip install --only-binary=swage-compiler "swage-compiler[pytorch]==0.5.2"
python -m swage.env --json --check native
python -m swage.env --json --check cpu
# With a CUDA-enabled PyTorch build and an NVIDIA driver:
python -m swage.env --json --check cuda
```

Choose the backend explicitly. An unavailable backend raises
`swage.BackendUnavailableError` and never probes the other one. The
[Quickstart](docs/getting-started/quickstart.md) walks through the
canonical kernels and the segmented calls. The
[runtime environment reference](docs/reference/runtime-environment.md)
defines the build identity and the environment report, and
[Troubleshooting](docs/getting-started/troubleshooting.md) lists the error
codes.

Until then, a native wheel is built from a clean checkout against the
pinned LLVM, as the GPU workflow builds it.
[Installation](docs/getting-started/installation.md#build-a-local-native-wheel)
gives the full command:

```bash
python -m build --wheel --no-isolation -Cbuild-dir=<build directory> \
    -Ccmake.define.MLIR_DIR=<LLVM install>/lib/cmake/mlir \
    -Ccmake.define.LLVM_DIR=<LLVM install>/lib/cmake/llvm \
    -Ccmake.define.SWAGE_SOURCE_REVISION="$(git rev-parse HEAD)" \
    -Ccmake.define.SWAGE_SOURCE_CLEAN=true
```

The wheel also ships `python -m swage.bench vector-add --output result.json`
for the frozen CUDA vector-add benchmark. The
[benchmark CLI contract](docs/reference/benchmarking.md) gives its
prerequisites, raw evidence, and `--enforce`; this command alone does not
qualify a release.

Compiler contributors can instead build exactly `llvmorg-22.1.8` and use
the build tree:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -Cwheel.cmake=false
./scripts/fetch_llvm.sh
LLVM_SRC="${SWAGE_LLVM_HOME:-$HOME/.swage/llvm}/src-$(cat cmake/llvm-version.txt)"
python -m pip install -r "$LLVM_SRC/mlir/python/requirements.txt"
./scripts/build_llvm.sh
./scripts/build_swage.sh
ninja -C build check-swage-python
```

Source builds keep `build/python_packages/mlir_swage`, which a script uses
through `PYTHONPATH=build/python_packages`. The frontend-only editable
install captures kernels and reports the environment, but does not emit or
launch without those bindings. Source distributions and the frontend-only
editable install need no CMake.
[Installation](docs/getting-started/installation.md) gives the build
requirements and explains the health checks.

## Examples

Four committed examples run from the installed wheel, or from a source
build with `PYTHONPATH=build/python_packages`:

- `examples/emit_fixed_vector_add.py` emits MLIR and needs no GPU and no
  PyTorch.
- `examples/fixed_vector_add.py` and `examples/fixed_vector_multiply.py`
  launch the fixed kernels and take `--backend cpu|cuda` and `--dtype`.
- `examples/segment_reduce.py` runs both segmented calls on a CUDA GPU and
  compares every result with PyTorch:

```python
import swage
import torch

# Six values in four segments: [1, 2], [], [3, 4, 5], and [6].
values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], device="cuda")
offsets = torch.tensor([0, 2, 2, 5, 6], dtype=torch.int32, device="cuda")

totals = swage.segment_reduce(values, offsets, "sum")  # [3, 0, 12, 6]
weights = swage.segment_softmax(values, offsets)       # six weights
```

The [support matrix](docs/reference/support-matrix.md) lists which Python,
PyTorch, driver, and GPU combinations are tested.

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md), including
  [Segmented Calls](docs/user-guide/segmented-calls.md) and
  [Running Without the Compiler](docs/user-guide/deployment.md)
- [API Reference](docs/reference/index.md)
- [Support Matrix](docs/reference/support-matrix.md)
- [Compiler Pipeline](docs/internals/compiler-pipeline.md)
- [Internals](docs/internals/index.md)
- [Verification](docs/internals/verification.md)
- [DESIGN.md](DESIGN.md), [ROADMAP.md](ROADMAP.md), and
  [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Swage source: [MIT](LICENSE). Native wheels also redistribute LLVM under
[Apache-2.0 with LLVM exceptions](LICENSES/LLVM.txt) and the other
third-party code that [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
lists.
