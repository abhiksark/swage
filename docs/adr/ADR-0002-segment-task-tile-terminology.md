# ADR-0002: Segment, Task, and Tile terminology

- Status: accepted
- Date: 2026-08-18

## Context

The project needs a stable vocabulary separating what the programmer
means, what the runtime schedules, and what the hardware executes.
Overloading one word for all three (as early drafts did with
`RaggedBlock`) blurs the compiler's core boundary.

## Decision

Three terms, used consistently in APIs, IR, and docs:

- **Segment** — logical, runtime-sized, internally dense data object:
  `values[offsets[i] : offsets[i+1]]`. Semantic model.
- **Task** — schedulable unit of execution (a packed group of short
  segments, a chunk of a long one, a partial reduction, a merge…).
  Planning model.
- **Tile** — fixed-size physical unit (`tile<32xf32>`) processed by a warp
  or CTA. Hardware-lowering model.

`RaggedBlock` is not used in new APIs or IR. The Python-visible domain is
the *logical grid*; the generated GPU domain is the *physical grid*; the
*planner* maps between them.

## Consequences

- Dialect names follow the model: `swage` (segments), `swage_plan`
  (tasks), tiles appear at the fixed-size lowering level.
- Reviews reject changes that conflate the levels (e.g., thread ids in
  segment IR).
