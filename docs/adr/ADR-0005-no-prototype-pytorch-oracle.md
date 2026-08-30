# ADR-0005: No Python prototype — PyTorch is the initial oracle

- Status: accepted
- Date: 2026-08-18

## Context

The project's master plan assumed a pre-existing Python prototype compiler
(AST → dataclass IR → CPU interpreter → CUDA-C/NVRTC backend) that would be
tagged `v0.1.0-python-prototype`, frozen, and kept as a differential-test
oracle during the MLIR migration.

**That prototype does not exist.** The repository started from an empty
directory on 2026-08-18. There is nothing to tag, freeze, or migrate.

## Decision

- Skip the prototype entirely rather than build a throwaway compiler
  first; build the MLIR path directly from the native compiler foundation.
- Differential oracles, in order of introduction: (1) PyTorch reference
  implementations of every supported kernel once GPU execution exists;
  (2) a small CPU reference lowering/interpreter for segment semantics,
  added with the first segment lowerings so GPU schedules can be checked
  against a sequential executor.
- `SWAGE_BACKEND=prototype` and the `prototype/` tree are dropped from the
  plan. The no-silent-fallback rule still applies between real backends.
- Version `v0.1.0` (reserved for the prototype freeze) is skipped; releases
  begin from the native compiler and runtime implementation instead.

## Consequences

- The roadmap's prototype-freeze phase is closed as "not applicable".
- Every correctness claim still requires an oracle comparison — the oracle
  is PyTorch (and later the CPU reference lowering), not a prototype.
