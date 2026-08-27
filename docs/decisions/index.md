<!-- docs/decisions/index.md -->

# Architecture Decision Records

ADRs explain why a boundary was chosen. Current interfaces belong in the
reference pages, current release status belongs in the README, and milestone
gates belong in the roadmap.

## Foundations and vocabulary

- [ADR-0001: Python AST directly to MLIR](../adr/ADR-0001-python-ast-directly-to-mlir.md)
- [ADR-0002: Segment, Task, and Tile terminology](../adr/ADR-0002-segment-task-tile-terminology.md)
- [ADR-0003: Two dialects](../adr/ADR-0003-two-dialects.md)
- [ADR-0004: Exact LLVM pin](../adr/ADR-0004-llvm-pin.md)
- [ADR-0005: No Python prototype, PyTorch oracle](../adr/ADR-0005-no-prototype-pytorch-oracle.md)
- [ADR-0006: PTX through LLVM NVPTX](../adr/ADR-0006-ptx-via-llvm-nvptx.md)
- [ADR-0007: Distribution and package naming](../adr/ADR-0007-distribution-naming.md)

## Semantic IR and frontend

- [ADR-0008: Region operations, isolation, and reduction kinds](../adr/ADR-0008-region-ops-isolation-and-kinds.md)
- [ADR-0009: Python bindings package and CI](../adr/ADR-0009-python-bindings-package-and-ci.md)
- [ADR-0010: Fixed-block MLIR boundary](../adr/ADR-0010-fixed-block-mlir-boundary.md)
- [ADR-0011: PyTorch metadata inference](../adr/ADR-0011-pytorch-metadata-inference.md)
- [ADR-0012: M3 launch ABI and execution boundary](../adr/ADR-0012-m3-launch-abi-and-execution-boundary.md)

## Private segmented qualification

- [ADR-0013: M5 fusion and map-store ABI](../adr/ADR-0013-m5-fusion-and-map-store-abi.md)
- [ADR-0014: M6 minimal SwagePlan gate](../adr/ADR-0014-m6-minimal-swage-plan-gate.md)
- [ADR-0015: M7 minimal mixed-policy execution](../adr/ADR-0015-m7-minimal-mixed-policy-execution.md)
- [ADR-0016: M7 fused mixed-policy schedule](../adr/ADR-0016-m7-fused-mixed-policy-schedule.md)
- [ADR-0017: M8 private split-CTA reductions](../adr/ADR-0017-m8-private-split-cta-reductions.md)

Start with [Compiler Pipeline](../internals/compiler-pipeline.md) when you
need current data flow, then use this index to find the decision that owns the
rationale.
