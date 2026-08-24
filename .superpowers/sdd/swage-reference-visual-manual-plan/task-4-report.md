# Task 4 report: rendered-site integrity and CI gates

## Commit

- Reviewed base: `8b02512cb0e40bf06be1abec99574e4da53c6236`.
- Task 4 implementation: the `HEAD` commit containing this report. Its
  resolved SHA is recorded in the controller handoff after commit creation.
- Push or merge: not performed.

## Files changed

- `scripts/check_docs_site.py`
- `tests/python/test_check_docs_site.py`
- `mkdocs.yml`
- `Makefile`
- `.github/workflows/ci-python.yml`
- `.github/workflows/ci-cpp.yml`
- `.superpowers/sdd/swage-reference-visual-manual-plan/task-4-report.md`

The checker uses only Python standard-library `HTMLParser`, `Path`, and URL
parsing. It discovers rendered HTML deterministically, collects local `href`
and `src` references plus HTML fragments, resolves page-relative and
root-relative targets, and returns all missing targets and fragments in stable
order. It does not write to the rendered site or crawl external URLs.

MkDocs validation now treats omitted navigation pages, missing navigation and
link targets, absolute local paths, unrecognized relative links, and missing
anchors as warnings. Strict mode promotes those warnings to failures.

`make diagrams` writes the reviewed atlas. `make docs` checks diagram
freshness, builds MkDocs strictly, and checks the rendered `site/` tree without
repairing reviewed assets. Python CI calls `make docs`. Native CI builds the
four reviewed TableGen documentation targets and checks the committed MLIR
reference fragments.

## TDD evidence

RED command:

```bash
PYTHONPATH="$PWD/python" python -m pytest \
  tests/python/test_check_docs_site.py -q
```

Result: `3 failed in 0.02s`. Every test failed at the intended boundary
because `scripts/check_docs_site.py` did not exist.

GREEN command:

```bash
PYTHONPATH="$PWD/python" python -m pytest \
  tests/python/test_check_docs_site.py -q
```

Result: `3 passed in 0.01s` in the final focused run.

The tests cover nested page paths, valid relative assets, missing assets,
current-page and cross-page fragments, missing fragments, directory and
explicit `index.html` resolution, root-relative resolution, query strings,
external URL skipping, complete stable error reporting, a missing site, and a
site without HTML pages. They use temporary rendered trees and the real
checker without mocks.

## Verification

- `python scripts/render_docs_diagrams.py --check`: passed.
- `make diagrams`: passed; `git diff --exit-code -- docs/assets/diagrams`
  confirmed that reviewed SVG bytes did not change.
- `ninja -C build-task2 SwageDialectDocGen SwageOpsDocGen
  SwagePlanDialectDocGen SwagePlanOpsDocGen`: passed with
  `ninja: no work to do`.
- `python scripts/sync_mlir_reference.py --build-dir build-task2 --check`:
  passed.
- `make docs`: passed. MkDocs built strictly and the rendered-site checker
  returned zero failures. The Material team's upstream MkDocs 2 informational
  banner remains visible and was not suppressed.
- `PYTHONPATH="$PWD/python" python -m pytest tests/python -q`: `106 passed in
  3.58s`.
- `ruff check .`: `All checks passed!`.
- `git diff --check`: passed with no output.

Pytest emitted the environment's existing `pytest-asyncio` default
loop-scope deprecation warning.

The CI diff contains the exact four native documentation target names and the
required reference freshness command. Filtered comparisons against the
reviewed base confirmed unchanged action pins, permissions, and concurrency.
The GPU and publication workflows, LLVM pin, documentation content, generated
reference bytes, SVGs, and branding assets are unchanged.

`.readthedocs.yaml` remains at the repository root in MkDocs mode with Ubuntu
24.04, Python 3.13, `docs/requirements.txt`, and `fail_on_warning: true`.

## Semantic impact and skipped work

Compiler and runtime semantics, public APIs, release state, and GPU behavior
did not change. No GPU architecture was used and no GPU test was run because
this task changes documentation validation and CI wiring only.

C++ unit tests, lit, native Python bindings, and CUDA runtime tests were not
rerun. The native scope was limited to the four documentation targets and
generated-reference freshness check.

No browser stack was installed or launched. Final desktop, narrow-width,
theme, focus, logo, favicon, route, and console QA is intentionally reserved
for the controller's approved browser-testing workflow.

## Limitations and follow-up

The checker intentionally validates rendered local `href` and `src` targets
and HTML fragments only. It does not crawl the network, execute JavaScript,
validate CSS semantics, parse Markdown, or replace a general HTML validator.

The controller should run final browser QA and whole-branch review before any
push, merge, Read the Docs change, tag movement, or publication.
