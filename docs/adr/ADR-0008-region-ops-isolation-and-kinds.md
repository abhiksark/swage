<!-- docs/adr/ADR-0008-region-ops-isolation-and-kinds.md -->
# ADR-0008: Region ops with explicit captures and kind-based reduction

- Status: accepted
- Date: 2026-08-18
- Amended: 2026-10-02, the statement about associativity
- Amended: 2026-10-03, a mean is a composition and not a kind

## Context

The initial semantic dialect adds the region-based operations `swage.map`,
`swage.reduce`, `swage.map_store`, and `swage.yield` (issue #1). Two designs
needed a decision with alternatives:

1. How outer SSA values enter a region. The linalg convention (regions
   may reference enclosing values freely, discipline by custom verifier)
   versus structural isolation with an explicit operand list.
2. How `swage.reduce` combines elements. A generic combiner region
   versus a fixed `kind` attribute.

## Decision

Regions on all three ops are `IsolatedFromAbove`. Outer values enter
only through `captures(...)`, whose operands append to the region block
arguments after the element argument, in order:

```mlir
%den = swage.reduce %s captures(%max : f32) kind<sum>
    : !swage.segment<f32> -> f32 {
^bb0(%x: f32, %m: f32):
  %sh = arith.subf %x, %m : f32
  %e = math.exp %sh : f32
  swage.yield %e : f32
}
```

`swage.reduce` combines with a `kind` enum, initially `sum`, `max`, and
`min`, while its region is the per-element transform. Every admitted
kind has a known identity and no fixed combining order. That freedom is
what lets a lowering split a reduction into partial results and a merge
([ADR-0017](ADR-0017-private-split-cta-reductions.md)). Identities per
element type: `sum` → 0, `max` → −∞ / minimum integer, `min` → +∞ /
maximum integer.

This record first said that every admitted kind must be associative and
commutative. That holds for `max` and `min`, whose result does not depend
on the order. It does not hold for floating-point `sum`, which is
commutative but not associative: the result depends on the order a
lowering and a schedule choose, within rounding, and two schedules need
not agree bit for bit
([ADR-0019](ADR-0019-composable-private-reductions.md)). Integer `sum`
has no lowering, and its overflow behavior is not defined. The operation
description in `SwageOps.td` states the current contract.

A mean is not a kind, and the kind enum does not grow for it
([ADR-0022](ADR-0022-wider-data-model-for-segmented-reductions.md)). A kind
is an identity and a combine with no fixed order, and that is what lets a
lowering merge partial results with the kind of the program. A mean has no
identity, and a mean of partial means is not the mean. A `kind<mean>`
reduction in the merge of a split would also read a range of partial
results and divide by a number that is not the extent of that range. A mean
is therefore written with what the dialect already has: a `kind<sum>`
reduction, `swage.extent` of the same segment, and one `arith.divf`, which
runs once per segment after the reduction. The lowerings admit exactly that
scalar epilogue, and the split lowering sums the partial results and
divides once. An empty segment gives NaN, zero divided by zero, where the
kinds give their identity.

Semantic contract:

- **Empty segment**: `map` yields an empty segment, `map_store` writes
  nothing, `reduce` returns the identity of its kind. This deliberately
  differs from PyTorch, where `max` of an empty tensor errors; oracle
  comparisons must account for it.
- **Floating max NaNs**: a non-empty f32 `max` reduction propagates NaN. The
  private segmented lowering uses `maximumf`, not
  `maxnumf`, and tests the behavior against the CPU oracle and PyTorch.
- **Effects**: `map`, `reduce`, and `map_store` read the values and
  offsets buffers behind the segment handle and declare that read on
  their segment operand, as `swage.extent` does, so common subexpression
  elimination does not merge two instances across a write. `map` and
  `reduce` expose their region's effects in addition to that read. A read
  alone does not keep an operation alive, so unused instances with pure
  `arith`/`math` bodies still fold away. `swage.map_store` also declares
  a write on its output operand and is never dead-code-eliminated.
- **Aliasing**: `map_store`'s output must not alias the segment's
  values buffer. A runtime obligation, documented, not statically
  checked.
- **Element types**: `map` may change the element type; the yield type
  defines the result segment's element type (`reduce`: the scalar
  result type; `map_store`: the output element type).

Statically verified: single-block region; block-argument count and
types against the element type and captures; yield-type agreement as
above; kind validity. Rank-1 outputs and scalar int-or-float captures
and results are ODS type constraints.

## Consequences

- Explicit dataflow is structural, not conventional: a region cannot
  name an outer buffer, so cross-segment stores are inexpressible and
  region extraction during task splitting needs no capture analysis.
- Constants used inside a region are defined inside it or captured.
- A future zipped multi-segment `map` is an additive signature change,
  not a redesign; it is omitted until a consumer exists.
- The fixed-block ops implied by the vector-add API (`program_id`,
  `arange`, masked load/store) are not part of this op set; their
  representation belongs to the fixed-block frontend design.
