<!-- README.md -->

<p align="center">
  <img
    src="https://raw.githubusercontent.com/abhiksark/swage/a1a49346772b28b9658a2305b9206f85d8e0443e/docs/assets/images/swage-logo.png"
    alt="Swage logo"
    width="720"
  >
</p>

# Swage

[![ci-python](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-python.yml)
[![ci-cpp](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-cpp.yml)
[![GPU runtime](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml/badge.svg?branch=main)](https://github.com/abhiksark/swage/actions/workflows/ci-gpu.yml)

**Turn variable-sized dense segments into efficient GPU tile tasks.**

Swage is an experimental Python-embedded MLIR/LLVM GPU compiler. It explores
whether one segment-local program can support different fixed GPU work shapes
as runtime segment lengths change.

## Current release boundary

The current pre-alpha release is `v0.5.1`.

### Public today

- The canonical fixed vector-add kernel is the only public execution subset.
- The restricted Python frontend can emit verified MLIR through build-tree
  native bindings.
- The fixed vector add can lower through LLVM NVPTX and launch through the
  CUDA Driver API on the current PyTorch stream.
- The `swage` dialect, `swage-opt`, and environment diagnostics are available
  to compiler contributors.

### Private qualification

- M4 and M5 qualify canonical segmented sum, max, and stable ragged softmax
  through sequential CPU oracles and one-CTA GPU paths.
- M6 to M8 qualify one canonical identity segmented sum through host
  classification, direct warp and CTA work, one fused mixed kernel, and
  split-CTA partial and merge kernels.
- The frozen M7 NVIDIA RTX A6000 `sm_86` record has a
  mixed-to-best-pure ratio of `0.939394`, below its predeclared `1.05` limit.
- M8 exact and nontrivial f32 split sums match PyTorch and the CPU oracle on
  NVIDIA RTX A6000 `sm_86`. M8 is a correctness result and does not retune the
  M7 benchmark.

### Planned

- Public segment syntax and public segmented launch.
- Packed warps, split max, split softmax, device queues, persistent
  scheduling, and broader policies.

Private qualification is not a public segmented runtime. Current status is
backed by the repository's executable tests and committed benchmark record.

## Package and native build

The `swage-compiler` wheel contains only the pure Python `swage` package. It
does not contain compiler libraries, build output, or the native
`mlir_swage` package. Native wheel packaging is deferred.

```bash
python -m pip install swage-compiler
python -m pip install "swage-compiler[pytorch]"  # optional
```

Compiler emission and execution require a native build against the pinned
LLVM/MLIR release:

```bash
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh
ninja -C build check-swage-python
```

The native package is imported from `build/python_packages`. The published
wheel remains useful for package import, source capture, and diagnostics, but
does not independently emit MLIR or execute kernels.

## Documentation

- [Installation](docs/getting-started/installation.md)
- [Quickstart](docs/quickstart.md)
- [Swage, Visually](docs/concepts/swage-visual-guide.md)
- [Compiler Pipeline](docs/architecture/compiler-pipeline.md)
- [Public Python API](docs/reference/public-python-api.md)
- [Private M4 to M8 Qualification](docs/qualification/private-m4-m8.md)
- [Verification Evidence](docs/qualification/evidence.md)
- [DESIGN.md](DESIGN.md), [ROADMAP.md](ROADMAP.md), and
  [CONTRIBUTING.md](CONTRIBUTING.md)

## License

MIT. See [LICENSE](LICENSE).
