<!-- CONTRIBUTING.md -->

# Contributing to Swage

Swage is pre-alpha. Useful contributions are bounded, tested, and explicit
about whether they affect public behavior, private qualification, or planned
work.

## Ground rules

- Preserve semantic correctness before performance.
- Follow the root and nearest scoped `AGENTS.md` files.
- Do not claim planned behavior as implemented.
- Keep internal milestone codenames in the roadmap, maintainer planning, and
  compatibility redirects; use capability names in project surfaces.
- Add tests with behavior changes and run the applicable tier.
- Record every change to what the public frontend accepts or rejects, and
  every removal or rename of a public name, in `CHANGELOG.md`.
- Keep the LLVM pin unchanged outside a dedicated compatibility change.
- Do not add Triton, a second production IR, or silent backend fallback.
- A GPU is not required for Python, documentation, dialect, and CPU-lowering
  contributions.

## Setup

```bash
git clone https://github.com/abhiksark/swage
cd swage

python -m pip install -e ".[dev]"
PYTHONPATH="$PWD/python" python -m pytest tests/python -q
ruff check .

./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh
```

See [`docs/getting-started/installation.md`](docs/getting-started/installation.md)
for prerequisites, build overrides, and the published-package boundary. The
native build needs the MLIR Python binding requirements installed after
`fetch_llvm.sh` and before `build_llvm.sh`; that page lists them and gives
the command.

The hosted workflows and the docs build install one hash-locked tool set
from `requirements-ci.txt`, which pins every tool of the `dev` and `docs`
extras and of the `ci` dependency group. To reproduce a hosted job locally,
install it with `python -m pip install --require-hashes -r
requirements-ci.txt`. The comment above the `ci` group in `pyproject.toml`
gives the command that regenerates the file.

## Native Python bindings

The `swage-compiler` wheel contains only the pure Python `swage` package. The
native `mlir_swage` package is a build-tree artifact and native wheel
packaging is deferred. A native build contains third-party code;
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists it and must go with
any copy of the build that leaves the machine that built it.

```bash
ninja -C build check-swage-python
```

This target supplies `build/python_packages` on `PYTHONPATH`. The selected
pinned MLIR install must include Python bindings. CMake fails if native
bindings are enabled against an incompatible install.

## Contributor paths

- Documentation: fix incorrect boundaries before improving presentation. Run
  `make docs`, which checks the diagrams and figures, builds the site
  strictly, and checks its links, and `ruff check .`.
- Public Python frontend: work under `python/swage/`. The accepted AST is a
  narrow fixed-vector-add contract, and the public API adds two segmented
  calls with fixed programs. Run the Python tier and native binding
  integration when emission changes.
- Native dialects and lowering: work under `include/swage/`, `lib/`, and
  `test/`. Run `ninja -C build check-swage`; run C++ or binding targets when
  their code changes.
- Runtime: public execution is canonical fixed vector add and the two
  segmented calls, `swage.segment_reduce` and `swage.segment_softmax`. The
  helpers behind those calls are private qualification. Runtime changes
  require the hosted tests and, where CUDA behavior changes, trusted GPU
  evidence.
- Benchmarks: preserve frozen inputs and gates, and never overwrite a record
  under `benchmarks/results/`. Prepare outside timing, except in a benchmark
  that deliberately times preparation to measure per-layout cost
  (`benchmarks/benchmark_fresh_offsets.py`). Repeat a configuration in
  independent processes with `benchmarks/benchmark_processes.py`, which
  writes to a directory outside the checkout, and commit raw evidence with
  the exact hardware and revision.

Start with [Compiler Pipeline](docs/internals/compiler-pipeline.md), then
use [Compiler Tools and Passes](docs/internals/compiler-tools.md) and
[Verification](docs/internals/verification.md) for the affected
surface.

## Pull requests

1. Branch from `main` and make one coherent change.
2. Run the smallest relevant test, then the full applicable tier.
3. Report files changed, semantic impact, tests run and skipped, GPU
   architecture used if any, limitations, and follow-up work.
4. Fill the pull request template with the same boundary information.
5. Leave unrelated work untouched. A maintainer reviews and merges.

## Reporting issues

Bug reports need a minimal reproducer, exact versions, and
`python -m swage.env` output. Performance reports also need hardware,
distribution, command, methodology, and raw measurements.

## License

Contributions are licensed under the [MIT License](LICENSE).
