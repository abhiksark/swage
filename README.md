<!-- README.md -->

![Swage logo](https://raw.githubusercontent.com/abhiksark/swage/main/docs/assets/images/swage-logo.png)

# Swage

[![ci-python](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml)
[![ci-cpp](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml)
[![GPU runtime](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml)

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is a Python-embedded MLIR/LLVM compiler with a deliberately narrow public
execution contract: canonical fixed vector addition or multiplication on
explicitly selected
CPU and CUDA backends. Variable-sized segment execution remains private
research.

## Current capability boundary

The latest tagged release recorded here is `v0.5.1`. The `v0.5.2` source tree
implements the native-wheel contract below; publication and production
qualification remain gated on the installed-artifact workflow. Source changes
alone are not release qualification.

### Public today

- Canonical fixed vector add and multiply are the only public execution subset.
  Contiguous rank-one tensors may use `float32`, `float16`,
  `float8_e4m3fn`, or `float8_e5m2`; both inputs and the output must match.
  Low-precision arithmetic widens to FP32 and rounds back to the storage
  dtype. FP8 conversion runs in the compiled kernel, including on `sm_86`.
  Kernels contain exactly one `x + y` or `x * y` value operation; chains,
  floating vector/scalar arithmetic, broadcasting, and matrix multiplication remain
  unsupported.
- The restricted Python frontend emits verified MLIR through self-contained
  native bindings, bundled in the v0.5.2 wheel contract.
- `kernel.launch(..., backend="cuda")` lowers through LLVM NVPTX and enqueues
  through the CUDA Driver API on the current PyTorch stream.
- `kernel.launch(..., backend="cpu")` lowers the same admitted program to a
  synchronous, process-local Native LLVM JIT entry for CPU tensors.
- Backend selection is explicit and fail-closed; no backend falls back to the
  other.
- The `swage` dialect, `swage-opt`, and environment diagnostics are available
  to compiler contributors.

### Private qualification

- Canonical segmented sum, max, and stable ragged softmax are qualified
  through sequential CPU oracles and one-CTA GPU paths.
- One canonical identity segmented sum is qualified through host
  classification, direct warp and CTA work, one fused mixed kernel that
  privately packs four warp task records per 128-thread block, and split-CTA
  partial and merge kernels.
- The frozen NVIDIA RTX A6000 `sm_86` mixed-policy record has a
  mixed-to-best-pure ratio of `0.939394`, below its predeclared `1.05` limit.
- Exact and nontrivial f32 split sums match PyTorch and the CPU oracle on
  NVIDIA RTX A6000 `sm_86`. Split execution is a correctness result and does
  not retune the frozen benchmark.
- An experimental private resident identity-sum path has correctness evidence,
  but its predeclared performance gate failed; it is not qualified or public.

### Planned

- Public segment syntax and public segmented launch.
- Public/general packed-warp planner policy, split max, split softmax, reusable
  device queues, qualified persistent scheduling, and broader policies.

Private qualification is not a public segmented runtime. Current status is
backed by the repository's executable tests and committed benchmark record.

## Native wheels and source builds

The v0.5.2 wheel contains public `swage`, its fixed-contract type stubs, and the
self-contained private `mlir_swage` compiler package. It does not need an
external MLIR installation or a CUDA toolkit compiler at runtime. Private
segmented Python modules are excluded; source distributions retain research
code.

The release matrix is regular CPython **3.10–3.13**, **Linux x86-64 with glibc
2.28 or newer**, and optional **PyTorch 2.6–2.x**. CPU execution is gated on every
wheel ABI. The continuously qualified CUDA configuration is **NVIDIA RTX
A6000, `sm_86`**; other admitted NVIDIA targets are best-effort, not equivalent
release evidence. There are no macOS, Windows, musl, other-architecture, or
free-threaded wheels in this contract.

After v0.5.2 passes publication gates:

```bash
python -m pip install --only-binary=swage-compiler "swage-compiler[pytorch]==0.5.2"
python -m swage.env --json --check native
python -m swage.env --json --check cpu
# With a suitable CUDA-enabled PyTorch installation and NVIDIA driver:
python -m swage.env --json --check cuda
```

Choose the backend explicitly; an unavailable backend raises
`swage.BackendUnavailableError`, never probes an alternative for execution.
See [Quickstart](docs/getting-started/quickstart.md) for the canonical kernels
and the runnable `examples/fixed_vector_add.py` and
`examples/fixed_vector_multiply.py` scripts. The
[runtime environment reference](docs/reference/runtime-environment.md)
defines error codes, build identity, opt-in `swage.runtime` DEBUG logging, and
wheel checksum/attestation verification.

Compiler contributors can instead build exactly `llvmorg-22.1.8`:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]" -Cwheel.cmake=false
./scripts/fetch_llvm.sh
SWAGE_PYTHON_EXECUTABLE="$(command -v python)" ./scripts/build_llvm.sh
./scripts/build_swage.sh
ninja -C build check-swage-python
```

Normal source builds retain `build/python_packages/mlir_swage`. The
frontend-only editable install does not itself provide native execution.
See [Installation](docs/getting-started/installation.md) for wheel candidates,
source-build requirements, and health-check interpretation.

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md)
- [Compiler Pipeline](docs/internals/compiler-pipeline.md)
- [API Reference](docs/reference/index.md)
- [Internals](docs/internals/index.md)
- [Verification](docs/internals/verification.md)
- [DESIGN.md](DESIGN.md), [ROADMAP.md](ROADMAP.md), and
  [CONTRIBUTING.md](CONTRIBUTING.md)

## License

Swage source: [MIT](LICENSE). Native wheels also redistribute LLVM under
[Apache-2.0 with LLVM exceptions](LICENSES/LLVM.txt).
