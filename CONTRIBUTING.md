<!-- CONTRIBUTING.md -->

# Contributing to Swage

Swage is pre-alpha. Useful contributions are bounded, tested, and explicit
about whether they affect public behavior, private qualification, or planned
work.

## Ground rules

- Preserve semantic correctness before performance.
- Follow the root and nearest scoped `AGENTS.md` files.
- Do not claim planned behavior as implemented.
- Add tests with behavior changes and run the applicable tier.
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
for prerequisites, build overrides, and the published-package boundary.

## Native Python bindings

The `swage-compiler` wheel contains only the pure Python `swage` package. The
native `mlir_swage` package is a build-tree artifact and native wheel
packaging is deferred.

```bash
ninja -C build check-swage-python
```

This target supplies `build/python_packages` on `PYTHONPATH`. The selected
pinned MLIR install must include Python bindings. CMake fails if native
bindings are enabled against an incompatible install.

## Contributor paths

- Documentation: fix incorrect boundaries before improving presentation. Run
  `mkdocs build --strict` and `ruff check .`.
- Public Python frontend: work under `python/swage/`. The accepted AST and API
  are narrow fixed-vector-add contracts. Run the Python tier and native
  binding integration when emission changes.
- Native dialects and lowering: work under `include/swage/`, `lib/`, and
  `test/`. Run `ninja -C build check-swage`; run C++ or binding targets when
  their code changes.
- Runtime: public execution remains canonical fixed vector add. M4 to M8
  segmented helpers are private qualification. Runtime changes require the
  hosted tests and, where CUDA behavior changes, trusted GPU evidence.
- Benchmarks: preserve frozen inputs and gates. Prepare outside timing and
  commit raw evidence with the exact hardware and revision.

Start with [Compiler Pipeline](docs/architecture/compiler-pipeline.md), then
use [Compiler Tools and Passes](docs/reference/compiler-tools.md) and
[Verification Evidence](docs/qualification/evidence.md) for the affected
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
