# Contributing to Swage

Thanks for considering a contribution. Swage is pre-alpha; the most useful
contributions right now are small, tested, and honest about what they do.

## Ground rules

- Correctness before performance; tests accompany behavior changes.
- Follow `AGENTS.md` (root and the nearest scoped one) — the invariants
  there bind humans and coding agents equally.
- No GPU is required to contribute. CPU-only development is a first-class
  path.

## Setup

```bash
git clone https://github.com/abhiksark/swage
cd swage

# Python side (fast, no LLVM needed)
make setup            # pip install -e ".[dev]"
make lint             # ruff check .
python -m pytest tests/python -q

# C++/MLIR side (one-time ~1 hour LLVM build)
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh     # builds swage-opt, runs check-swage
```

The LLVM pin lives in `cmake/llvm-version.txt`; never change it in a
feature PR (see ADR-0004). If you already have a matching MLIR install,
point `MLIR_DIR`/`LLVM_DIR` at it and skip the LLVM build.

The native `mlir_swage` bindings use the same pinned LLVM/MLIR install and
are enabled by default when that install has MLIR Python bindings enabled.
After the native build, run:

```bash
ninja -C build check-swage-python
```

This target provides the build-tree `PYTHONPATH` for `mlir_swage`. If the
install was built with `SWAGE_LLVM_PYTHON_BINDINGS=OFF`, rebuild it with that
variable set to `ON`, then reconfigure Swage with
`-DSWAGE_PYTHON_BINDINGS=ON`. CMake rejects an enabled binding build against
an MLIR install without Python bindings.

## Contributor paths

- **Documentation** — fix inaccuracies first, clarity second. Docs claiming
  more than the tests show are bugs.
- **Python frontend** — `python/swage/`; pytest + ruff; Google style,
  80 columns; see `python/AGENTS.md`. M2 emits a live MLIR module for the
  fixed-block vector-add subset from explicit descriptors or supported
  PyTorch metadata. It does not execute kernels.
- **Native bindings:** `python/mlir_swage/`; run `check-swage-python` after
  a bindings-enabled native build. This package is build-tree-only until
  wheel packaging is designed.
- **MLIR dialect** — `include/swage/`, `lib/`; ODS + verifiers + lit tests;
  see `lib/AGENTS.md` and `test/AGENTS.md`.
- **GPU backend** — arrives with M3+; watch issues labeled `area:runtime`.
- **Benchmarks** — arrives with M7+; correctness gates before timing,
  reproducible commands, raw data committed.

Look for issues labeled `good first issue` — they are real, bounded, and
mergeable.

## Pull requests

1. Branch from `main`; one coherent change per PR.
2. Run the applicable test tier (`pytest`, `check-swage`) before pushing.
3. Fill the PR template honestly, including what was *not* run.
4. CI must be green; a maintainer reviews and merges.

## Reporting issues

Use the issue forms. Bug reports need a minimal reproducer and exact
versions (`python -m swage.env` output helps). Performance reports need
hardware, distribution, command, and raw numbers.

## License

By contributing you agree your contributions are licensed under the MIT
License (see `LICENSE`).
