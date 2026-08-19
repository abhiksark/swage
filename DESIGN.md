# Swage design

This document records the architecture and the invariants that hold across
the codebase. Deep dives live in `docs/`; decisions with alternatives live
in `docs/adr/`.

## Problem

Fixed GPU tiles work extremely well for regular workloads. Variable-sized,
internally dense segments (ragged softmax rows, jagged batches, graph
neighborhoods) create padding waste and load imbalance. Existing systems
expose pieces of the solution — manual bucketing, hand-written persistent
kernels, per-shape specialization. Swage's research question is whether one
segment-local program can automatically produce competitive warp, CTA,
split-CTA, and persistent schedules as the runtime segment-length
distribution changes.

## Vocabulary

Three terms, never conflated (full treatment in
[docs/concepts/segments-tiles-tasks.md](docs/concepts/segments-tiles-tasks.md)):

- **Segment** — a logical, runtime-sized, internally dense data object:
  `segment i = values[offsets[i] : offsets[i+1]]`. Part of the semantic
  model.
- **Task** — a schedulable unit of execution: one short segment, several
  packed short segments, a chunk of a long segment, a partial reduction, a
  merge, a store stage. Part of the planning model.
- **Tile** — a fixed-size physical unit (`tile<32xf32>`, `tile<128xf32>`)
  processed by a warp or CTA. Part of the hardware-lowering model.

The **logical grid** is the semantic domain written in Python. The
**physical grid** is the generated GPU task domain. The **planner** converts
segment programs plus runtime extents into tasks and fixed tiles.

## Pipeline

```text
Python @sw.jit kernel
        ↓  inspect.getsource → ast.parse (restricted subset, M2 compile-only)
Swage semantic MLIR            (dialect: swage)
        ↓  canonicalization and fusion
SwagePlan task IR              (dialect: swage_plan — not yet started)
        ↓  task decomposition and policy selection
Fixed-size tile operations
        ↓
arith + math + scf + memref + vector
        ↓
gpu + nvgpu
        ↓
nvvm + llvm dialect
        ↓
LLVM IR → LLVM NVPTX → PTX → CUDA Driver API → current PyTorch stream
```

There is exactly one production IR between Python and LLVM: MLIR. The
Python frontend constructs Swage MLIR directly through the MLIR Python
bindings (ADR-0001). Textual MLIR is a debug, test, and reproducer format,
not the JIT construction path.

## Current Python frontend boundary

M2 is complete. The fixed-block vector-add subset captures a `@sw.jit`
function and emits a verified live `mlir_swage.ir.Module` directly through
the native bindings. Callers must provide exactly one of `signature=` or
`arguments=`; providing both or neither is invalid. From `arguments=`, the
frontend infers rank-one f32 pointers and i32 scalars. PyTorch stays optional
and is imported only for inference.

Inference reads metadata only. It does not inspect data pointers or contents,
retain arguments, infer lengths, validate cross-tensor relationships, or
cache specializations. Kernel calls and direct symbolic language operations
do not execute. The M3 backend can lower the canonical fixed vector add and
emit deterministic PTX internally, but no launch or runtime result exists
(ADR-0011, ADR-0012).

## The `swage` dialect (semantic level)

Implemented today: `!swage.segment<T>` plus `swage.segment_id`,
`swage.make_segment`, `swage.extent`, and the region-based `swage.map`,
`swage.reduce` (kinds `sum`/`max`/`min`), `swage.map_store`, and
`swage.yield`. Regions are isolated from above: outer values enter only
through explicit `captures(...)`; scalar math inside regions uses standard
`arith` / `math` operations. `swage.map_store` is the only effectful
operation (declared write on its output). Empty-segment behavior,
reduction identities, and the aliasing obligation are specified in
[ADR-0008](docs/adr/ADR-0008-region-ops-isolation-and-kinds.md).

Design rules (enforced by verifiers as the dialect grows):

- A segment value is symbolic. A runtime-length segment is never
  represented as a runtime-sized register array; consumers are tiled during
  lowering.
- Runtime segment identity lives in SSA values and operands, never in
  types. `!swage.segment<f32>` carries the element type only.
- No GPU thread or block indices in semantic IR. `swage.segment_id` is a
  logical-grid coordinate.
- No custom Swage operations for ordinary scalar arithmetic — regions use
  `arith` and `math`.
- Cross-segment effects must be explicit; a segment program may not
  silently touch another segment's output range.

## Python bindings

Python constructs Swage IR through the self-contained `mlir_swage`
package: the pinned MLIR core Python sources, the standard-dialect
wrappers regions compose with (`builtin`, `func`, `arith`, `math`), and
the generated `swage` op bindings, all built from the one pinned LLVM
(ADR-0009). The package never imports an external `mlir` distribution,
so bindings can never skew against the pin. During development it is
imported from the build tree; wheel packaging is deferred. Bindings are
optional at configure time (`SWAGE_PYTHON_BINDINGS`), and requesting
them against an MLIR install built without bindings is a configure
error, never a silent skip.

## The `swage_plan` dialect (planning level — not yet started)

Introduced only after the semantic dialect and one fixed GPU lowering work
end to end (ADR-0003). It will model tasks, policies (`warp`, `packed_warp`,
`cta`, `split_cta`, `merge`, `persistent`), queues, and dependencies. The
planner distinguishes compile-time decisions (legal schedule variants, tile
sizes, reduction structure, cost estimates) from runtime decisions (actual
lengths, skew, task counts, selected policies); compiler passes never
pretend to know runtime offset contents.

## Runtime

The runtime reuses PyTorch's CUDA context and stream: device pointers via
`tensor.data_ptr()`, launches on `torch.cuda.current_stream()`, module
loading through the CUDA Driver API, PTX emitted by LLVM NVPTX (no NVRTC on
the production path, ADR-0006). No silent copies, dtype changes, CPU
fallback, or backend fallback.

## Testing strategy

- Python: pytest for compile-only frontend emission and source diagnostics;
  JIT specialization and runtime validation follow with their implementations.
- MLIR: lit + FileCheck for parse/print round trips, verifier failures,
  and lowering; checks target semantic invariants, not incidental
  formatting.
- C++: GoogleTest for planner and cost-model units (when they exist).
- Differential: every public correctness claim is backed by at least one
  oracle comparison. With no Python prototype in this repository
  (ADR-0005), PyTorch reference implementations are the initial oracle;
  a CPU reference lowering joins it with the first segment lowerings.
- Property-based: generated segment distributions (empty, tiny, uniform,
  log-normal, bimodal, Zipf-like, one-outlier) checking coverage and
  no-overlap invariants.

## Dependency policy

One exact LLVM release, pinned in `cmake/llvm-version.txt`, built
out-of-tree via `MLIR_DIR`/`LLVM_DIR` (ADR-0004). No vendored LLVM sources.
Pin updates are dedicated compatibility PRs.
