<!-- README.md -->

![Swage logo](https://raw.githubusercontent.com/abhiksark/swage/main/docs/assets/images/swage-logo.png)

# Swage

[![ci-python](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml)
[![ci-cpp](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml)
[![GPU runtime](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml)

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded MLIR/LLVM GPU compiler. It explores
whether one segment-local program can support different fixed GPU work shapes
as runtime segment lengths change.

Segments are not usable from Python yet. The public Python API compiles and
launches one kernel, a fixed-block vector add. Segmented sum, max, and softmax
run only through private qualification helpers in this repository. They have
no public syntax and no public launch call.

## Current release boundary

The current pre-alpha release is `v0.5.1`. The lists below describe that
release, except for the one item marked as completed after it.

### Public today

- The canonical fixed vector-add kernel is the only public execution subset.
- The restricted Python frontend can emit verified MLIR through build-tree
  native bindings.
- The fixed vector add can lower through LLVM NVPTX and launch through the
  CUDA Driver API on the current PyTorch stream.
- The `swage` dialect, `swage-opt`, and environment diagnostics are available
  to compiler contributors.

### Private qualification

Each item names the programs and the paths they run through:

- Canonical segmented sum, max, and stable ragged softmax run through
  sequential CPU oracles and one-CTA GPU paths.
- One canonical identity segmented sum runs through host classification,
  direct warp and CTA work, one fused mixed kernel, and split partial and
  merge kernels.
- Not part of `v0.5.1`, completed after that release: capture-free,
  single-stage f32 sum and max programs, including element expressions and
  map chains, run through the same host classification, direct warp and CTA
  work, fused mixed kernel, and split partial and merge kernels.

Each item of evidence was recorded on one NVIDIA RTX A6000 (`sm_86`):

- Mixed-policy gate: the frozen record, measured at revision `dcbcf39`, has
  a mixed-to-best-pure ratio of `0.939394`, below its predeclared `1.05`
  limit. The first schedule used two launches, measured `1.238806` on the
  same frozen input, and failed that gate. The fused one-launch schedule was
  predeclared in
  [ADR-0016](docs/adr/ADR-0016-fused-mixed-policy-schedule.md) before the
  passing run. Both ratios describe the PTX of the revision they were
  measured at. Kernels generated now also pass through an LLVM pass pipeline
  before PTX emission, and no committed record measures them.
- Split correctness: exact and nontrivial f32 split sums match PyTorch and
  the CPU oracle. Split execution is a correctness result and does not retune
  the frozen benchmark.

### Planned

- Public segment syntax and public segmented launch.
- Packed warps, split softmax, device queues, persistent
  scheduling, and broader policies.

Private qualification is not a public segmented runtime. Current status is
backed by the repository's executable tests and committed benchmark record.

## Package and native build

The `swage-compiler` wheel contains only the pure Python `swage` package. It
does not contain compiler libraries, build output, or the native
`mlir_swage` package. Native wheel packaging is deferred. A native build
contains third-party code, which
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists.

```bash
python -m pip install swage-compiler
python -m pip install "swage-compiler[pytorch]"  # optional
```

Compiler emission and execution require a native build against the pinned
LLVM/MLIR release. The MLIR Python binding requirements must be installed
between `fetch_llvm.sh` and `build_llvm.sh`;
[Installation](docs/getting-started/installation.md) lists them and gives the
command.

```bash
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh
ninja -C build check-swage-python
```

The native package is imported from `build/python_packages`. The published
wheel remains useful for package import, source capture, and diagnostics, but
does not independently emit MLIR or execute kernels.

Two committed examples use the native build.
`examples/emit_fixed_vector_add.py` emits MLIR with the native build alone and
needs no GPU and no PyTorch. `examples/fixed_vector_add.py` also launches the
kernel and needs a CUDA GPU. The
[support matrix](docs/reference/support-matrix.md) lists which Python,
PyTorch, driver, and GPU combinations are tested.

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/index.md)
- [Compiler Pipeline](docs/internals/compiler-pipeline.md)
- [API Reference](docs/reference/index.md)
- [Support Matrix](docs/reference/support-matrix.md)
- [Internals](docs/internals/index.md)
- [Verification](docs/internals/verification.md)
- [DESIGN.md](DESIGN.md), [ROADMAP.md](ROADMAP.md), and
  [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
