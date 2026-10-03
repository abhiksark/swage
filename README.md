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

Segments are usable from Python through two calls with fixed programs:
`swage.segment_reduce` computes a sum, a maximum, a minimum, or a mean per
segment, and `swage.segment_softmax` computes a softmax within each
segment, over int32 or int64 offsets on one CUDA device. Both take values
of rank one or of rank two, `[N, D]` rows that are reduced or normalized
per column. A reduction takes f32 or f64 values, and the softmax takes f32
values. Both are newer than the released wheel. They need a native build, or an artifact directory
that a native build wrote ahead of time. There is no public segment syntax:
the kernel language compiles and launches one kernel, a fixed-block vector
add.

## Current release boundary

The current pre-alpha release is `v0.5.1`. The lists below describe that
release, except for the items marked as completed after it.

### Public today

- The canonical fixed vector-add kernel is the only kernel the Python
  frontend compiles and launches.
- The restricted Python frontend can emit verified MLIR through build-tree
  native bindings.
- The fixed vector add can lower through LLVM NVPTX and launch through the
  CUDA Driver API on the current PyTorch stream.
- The `swage` dialect, `swage-opt`, and environment diagnostics are available
  to compiler contributors.
- Not part of `v0.5.1`, completed after that release: two segmented calls.
  `swage.segment_reduce(values, offsets, kind, *, out=None)` returns the
  sum, the maximum, the minimum, or the mean of every segment, and
  `swage.segment_softmax(values, offsets, *, out=None)` returns the softmax
  within every segment. They take int32 or int64 offsets on the current
  CUDA device: f32 or f64 values of rank one or two for a reduction, and
  f32 values of rank one or two for the softmax. Nothing else is admitted: no other
  dtype, kind, or rank. A sum and a mean record a gradient for values that
  require grad, with second derivatives; a maximum, a minimum, and the
  softmax have no backward yet. A call validates and classifies its offsets on the host
  every time, so with offsets that change on every call expect it to be
  slower than `torch.segment_reduce`. No committed record times the call:
  the one record of that regime times a private preparation, at an older
  revision, that prepares more than the call does.
  [Segmented Calls](docs/user-guide/segmented-calls.md) states the contract
  and the cost.
- Not part of `v0.5.1`, completed after that release: ahead-of-time
  artifacts for the two segmented calls. On a host with the native build,
  `python -m swage.compile --target <processor> --output <directory>`
  writes the kernels of both calls, a small runtime library, and a manifest,
  without a GPU. A process that sets `SWAGE_ARTIFACT_DIR` to the directory
  runs the two calls from it without `mlir_swage` and with no LLVM in the
  process. It still needs PyTorch and `numpy`, the artifact serves no other
  kernel, and the runtime library has been built for Linux x86-64 only.
  [Running Without the Compiler](docs/user-guide/deployment.md) states what
  an artifact holds, when it is refused, and what it does not deliver.

### Private qualification

The two public calls run through these paths with fixed programs and
default limits. Everything else about the paths is private: other programs,
the prepared launches, the pure policies, and the planning limits. Each item
names the programs and the paths they run through:

- Canonical segmented sum, max, and stable ragged softmax run through
  sequential CPU oracles and one-CTA GPU paths.
- One canonical identity segmented sum runs through host classification,
  direct warp and CTA work, one fused mixed kernel, and split partial and
  merge kernels.
- Not part of `v0.5.1`, completed after that release: capture-free,
  single-stage sum, max, and min programs over f32 or f64 values,
  including element expressions and map chains, run through the same host
  classification, direct warp and CTA work, fused mixed kernel, and split
  partial and merge kernels.

Each item of evidence was recorded on one NVIDIA RTX A6000 (`sm_86`):

- Mixed-policy gate: the frozen record, measured at revision `dcbcf39`, has
  a mixed-to-best-pure ratio of `0.939394`, below its predeclared `1.05`
  limit. The first schedule used two launches, measured `1.238806` on the
  same frozen input, and failed that gate. The fused one-launch schedule was
  predeclared in
  [ADR-0016](docs/adr/ADR-0016-fused-mixed-policy-schedule.md) before the
  passing run. Both ratios describe the PTX of the revision they were
  measured at. Kernels generated now also pass through an LLVM pass pipeline
  before PTX emission. The gate has not been rerun on them; a later
  committed record measures them with another harness.
- Split correctness: exact and nontrivial f32 split sums match PyTorch and
  the CPU oracle. Split execution is a correctness result and does not retune
  the frozen benchmark.

### Planned

- Public segment syntax, and public launch of a segment program that the
  caller writes.
- Packed warps, split softmax, device queues, persistent
  scheduling, and broader policies.

Private qualification is not a public segmented runtime: the public calls
expose two programs of it and none of its controls. Current status is
backed by the repository's executable tests and committed benchmark record.

## Package and native build

The `swage-compiler` wheel contains only the pure Python `swage` package. It
does not contain compiler libraries, build output, or the native
`mlir_swage` package. No native wheel is published. A native wheel,
`swage-compiler-native`, can be built from a checkout with
`scripts/build_native_wheel.sh`. A native build contains third-party code,
which [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists; the native
wheel carries that file.

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

The native package is imported from `build/python_packages`, or installed
into a fresh virtual environment from a native wheel built on the same
machine:

```bash
python -m build --wheel
./scripts/build_native_wheel.sh
python -m pip install --no-index --find-links dist swage-compiler-native
```

The native wheel records the `swage` version, the source revision, and the
LLVM version it was built from. `swage` refuses bindings built for another
version, and `python -m swage.env` prints all three. The wheel is built for
one Python version and carries the plain `linux_x86_64` tag; only a CPython
3.13 build has been tried, on the machine that built it.
[Installation](docs/getting-started/installation.md) states the limits.

The published wheel remains useful for package import, source capture, and
diagnostics, but does not independently emit MLIR, execute kernels, or run a
segmented call. The pure package of the current source tree runs the two
segmented calls from an artifact directory that
`python -m swage.compile` wrote on a host with the native build. That needs
PyTorch, `numpy`, and a CUDA GPU, and no `mlir_swage`;
[Running Without the Compiler](docs/user-guide/deployment.md) describes it.

Three committed examples use the native build.
`examples/emit_fixed_vector_add.py` emits MLIR with the native build alone and
needs no GPU and no PyTorch. `examples/fixed_vector_add.py` also launches the
kernel and needs a CUDA GPU. `examples/segment_reduce.py` runs the two
segmented calls and needs a CUDA GPU. The
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
