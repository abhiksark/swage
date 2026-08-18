# ADR-0008: Region ops with explicit captures and kind-based reduction

- Status: accepted
- Date: 2026-08-18

## Context

M1 adds the region-based semantic ops `swage.map`, `swage.reduce`,
`swage.map_store`, and `swage.yield` (issue #1). Two designs needed a
decision with alternatives:

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

`swage.reduce` combines with a `kind` enum — initially `sum`, `max`,
`min` — while its region is the per-element transform. Every admitted
kind must be associative and commutative with a known identity; that
gate is what later licenses split reductions (M8). Identities per
element type: `sum` → 0, `max` → −∞ / minimum integer, `min` → +∞ /
maximum integer.

Semantic contract:

- **Empty segment**: `map` yields an empty segment, `map_store` writes
  nothing, `reduce` returns the identity of its kind. This deliberately
  differs from PyTorch, where `max` of an empty tensor errors; oracle
  comparisons must account for it.
- **Effects**: `map` and `reduce` expose only their region's effects, so
  unused instances with pure `arith`/`math` bodies fold away.
  `swage.map_store` declares a write on its output operand and is never
  dead-code-eliminated.
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
  representation is M2's first design question.
