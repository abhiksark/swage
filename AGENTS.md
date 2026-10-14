# AGENTS.md

## Mission

Swage is a Python-embedded MLIR/LLVM GPU compiler that lowers
variable-sized dense segment programs into fixed-tile GPU tasks.

## Task routing

Read this file for every task, then read each applicable scoped guide and the
relevant supporting documents below. Routes are cumulative for cross-area
tasks; read only the sections needed for the change.

- Python, frontend, bindings, and related tests:
  [python/AGENTS.md](python/AGENTS.md).
- MLIR, C++, TableGen, and compiler tools: [lib/AGENTS.md](lib/AGENTS.md).
- lit and FileCheck work: [test/AGENTS.md](test/AGENTS.md).
- Current or planned status claims: [README.md](README.md) and
  [ROADMAP.md](ROADMAP.md).
- Architecture or lowering changes: [DESIGN.md](DESIGN.md) plus the relevant
  [execution model](docs/user-guide/execution-model.md) or
  [compiler pipeline](docs/internals/compiler-pipeline.md).
- Commands, contribution workflow, and completion reporting:
  [CONTRIBUTING.md](CONTRIBUTING.md). For claim-specific evidence, use
  [verification.md](docs/internals/verification.md).

## Non-negotiable rules

- Preserve semantic correctness before performance.
- Inspect the worktree before editing and preserve all unrelated changes.
- Do not claim planned features are implemented, in code, docs, or
  reports. Tests and executable examples are the source of truth.
- Keep internal milestone codenames in `ROADMAP.md`, maintainer planning, and
  compatibility redirects. Use capability names everywhere else.
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

## Branch workflow

- Start ordinary feature and maintenance work from `dev` and target `dev`.
- Keep `main` as the default and release branch. Only this repository's
  `dev`, nonempty `hotfix/*` branches, and authenticated same-repository
  Dependabot PRs target `main`.
- Merge `dev` into `main` with a merge commit. Synchronize `dev` with every
  reviewed main change before accepting more development work, independently
  of whether a release is published. Preserve dirty local checkouts.
- Require applicable tests and explicit merge authorization. Release tags
  must identify a commit on the first-parent history of `main` and satisfy
  the existing signed-tag, native-wheel, and GPU gates.
- The metadata-only branch-policy job is advisory. Do not make it required
  without separately authorized operator changes and proof that the result
  is bound to the trusted workflow and current PR.

## Status reporting

Every completed task reports: files changed; whether semantic behavior
changed; tests run / passed / skipped; GPU architecture used (if any);
known limitations; follow-up issue.
