# AGENTS.md

## Mission

Swage is a Python-embedded MLIR/LLVM GPU compiler that lowers
variable-sized dense segment programs into fixed-tile GPU tasks.

## Required reading

- `README.md` — current status; what works vs. what is planned
- `DESIGN.md` — architecture and invariants
- `ROADMAP.md` — phase gates and their status
- `docs/concepts/segments-tiles-tasks.md` — the Segment/Task/Tile model
- `docs/architecture/compiler-pipeline.md` — the lowering pipeline

## Non-negotiable rules

- Preserve semantic correctness before performance.
- Do not claim planned features are implemented — in code, docs, or
  reports. Tests and executable examples are the source of truth.
- Do not add Triton as a dependency or copy Triton implementation code.
- Do not introduce a second production IR between Python and MLIR
  (ADR-0001).
- Do not represent a runtime-length segment as a runtime-sized register
  array.
- Do not place GPU thread/block indices in semantic Swage IR.
- Do not put runtime segment identity into an MLIR type (SSA values carry
  identity).
- Do not silently fall back between backends.
- Do not update the LLVM pin (`cmake/llvm-version.txt`) outside a dedicated
  compatibility PR (ADR-0004).
- Do not merge changes without running the applicable test tier.
- Do not add placeholder directories, empty passes, or scaffold "for
  later".

## Standard commands

```bash
# Python (no LLVM build required)
python -m pytest tests/python -q
ruff check .

# MLIR components (requires the pinned LLVM; ~1 hour once)
./scripts/fetch_llvm.sh
./scripts/build_llvm.sh
./scripts/build_swage.sh          # configure + build + check-swage
ninja -C build check-swage        # lit suite only, after a build exists

# Native Python bindings (requires bindings-enabled pinned LLVM/MLIR)
ninja -C build check-swage-python

# Environment diagnostics
python -m swage.env
```

## Change protocol

1. Read the nearest `AGENTS.md` and the relevant design docs.
2. Run baseline tests for the area you touch.
3. Make one coherent change.
4. Add or update tests with the change.
5. Run the smallest relevant test first, then the full applicable tier.
6. Report exactly what was and was not executed.
7. Leave the working tree clean when committing is part of the task.

## Status reporting

Every completed task reports: files changed; whether semantic behavior
changed; tests run / passed / skipped; GPU architecture used (if any);
known limitations; follow-up issue.

## Native Python bindings

`mlir_swage` is imported from `build/python_packages`, not from the
`swage-compiler` pip package. Use `check-swage-python` so CMake supplies the
build-tree `PYTHONPATH`. The bindings require an MLIR install built with
Python bindings; `SWAGE_PYTHON_BINDINGS=ON` against an incompatible install
is an error. The M2 Python frontend emits MLIR from explicit descriptors or
PyTorch metadata, but no Python kernel executes.
