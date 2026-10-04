<!-- docs/adr/ADR-0019-composable-private-reductions.md -->
# ADR-0019: Composable private segmented reductions

- Status: accepted
- Date: 2026-09-13

## Context

Direct segmented lowering already composes element regions and map chains.
Static planning and split execution admitted only identity sum, so the same
program could not use those schedules. The existing descriptor and scratch
layouts are sufficient for one scalar reduction per segment.

## Decision

Static planning admits one capture-free f32 sum or max, optionally preceded
by capture-free, single-consumer map chains, ending in one scalar store at
the segment index. Regions use the existing admitted f32 arithmetic and
`math.exp2` operations. Captures, multiple reductions, map-store terminals,
unsupported operations, and other ABIs remain rejected before mutation.

Direct warp, CTA, fused mixed, and split partial emitters reuse the existing
element program. Each valid input element is transformed before accumulation.
A partial writes one f32 scalar to its unique scratch slot. The merge uses
only the reduction kind and never evaluates the element expression on scratch.
Empty sums return zero; empty maxima return negative infinity. Max propagates
NaNs and preserves `maximumf` signed-zero semantics. Expressions are never
evaluated for empty ranges or inactive lanes.

Floating-point sums retain the existing tolerance-based qualification;
different reduction trees do not promise bitwise equality. This change adds
no fast-math flags or new algebraic rewrites. Native exponential approximation
continues to follow the existing direct-lowering contract.

The private `_prepare_planned_reduction` helper accepts semantic qualification
MLIR text and a kernel name. It returns the existing owned prepared
execution with its pure warp, pure CTA, and classified mixed launches.
`_prepare_planned_sum` remains a wrapper for existing callers and frozen
benchmarks. Public Python APIs do not change.

The general private helper additionally enables a conservative preparation-time
schedule choice: with default chunks, batches containing only 4097–8192-element
segments and at least one segment per device SM can use the pure CTA launch
in place of split execution. The element program must fit a relative work
budget of 32: simple arithmetic costs one, `exp2` eight, and division sixteen,
summed across all map and reduction regions. Constants and yields are free.
This avoids split compilation and scratch allocation for eligible programs.
Other batches keep the classified schedule. `select_schedule=False` opts out;
the legacy sum wrapper always opts out. This host-side choice reuses native
admission and kernels and does not change the classifier contract. See
[Task Planning](../internals/planning.md) for the current selection boundary.

Descriptor layouts, launch ABIs, the 32/4096 default limits, and ordered
direct/partial/merge launches on the current stream remain unchanged.
Persistent execution retains a separate identity-sum admission restriction
because its partial and merge emitters have not been generalized.

## Qualification and consequences

Shared native programs cover identity, squares, and chained affine maps for
sum and max. Compiler checks prove source preservation, deterministic output,
and that merge code omits element transformations. CPU and GPU checks cover
empty ranges, default and custom split boundaries, repeated launches,
sentinels, and max special values. Existing runtime validation, stream,
graph, retention, and failure-ordering tests remain applicable.

This is a composition and correctness extension. The frozen identity-sum
benchmark and its clean-source requirement are unchanged; correctness does
not establish a new performance result. Split softmax, captured stage results,
general cost inference, and public segmented execution require follow-up work.
