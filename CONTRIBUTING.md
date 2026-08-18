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

## Contributor paths

- **Documentation** — fix inaccuracies first, clarity second. Docs claiming
  more than the tests show are bugs.
- **Python frontend** — `python/swage/`; pytest + ruff; Google style,
  80 columns; see `python/AGENTS.md`.
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
