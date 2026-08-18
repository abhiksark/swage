# Segments, Tasks, and Tiles

Swage's entire design hangs on keeping three concepts separate. Most
GPU-programming pain with ragged data comes from collapsing them into one.

## Segment — the semantic unit

A **segment** is a logical, runtime-sized, internally dense data object:

```text
segment i = values[offsets[i] : offsets[i + 1]]
```

Segments are what the programmer thinks about. A ragged softmax is "for
every segment: subtract the max, exponentiate, normalize" — no mention of
warps, blocks, or padding. Segment lengths are unknown at compile time,
may be zero, and may be wildly skewed within one launch.

In IR, a segment is the value produced by `swage.make_segment`; its type
`!swage.segment<f32>` carries only the element type. Which buffer, which
offsets, which index — that is runtime information and lives in SSA
operands, never in the type (see ADR-0002 and DESIGN.md).

## Tile — the physical unit

A **tile** is a fixed-size physical unit — `tile<32xf32>`,
`tile<128xf32>`, `tile<256xf32>` — processed by a warp or CTA. Tiles are
what GPUs are good at: statically-shaped loads, register allocation,
coalesced access, warp-synchronous reductions. Everything the hardware
executes is ultimately a tile-shaped operation.

## Task — the schedulable bridge

A **task** is a unit of execution the runtime can schedule. Tasks are how
variable-sized segments meet fixed-size tiles. A task may represent:

- one short segment,
- several short segments *packed* into one tile,
- one medium segment,
- one *chunk* of a long segment,
- a partial reduction, a reduction merge, or a normalize/store stage.

A long softmax segment might become:

```text
partial_max(segment=7, range=0:512)     partial_max(segment=7, range=512:1024)
merge_max(segment=7)
partial_sum(segment=7, range=0:512)     partial_sum(segment=7, range=512:1024)
merge_sum(segment=7)
normalize(segment=7, range=0:512)       normalize(segment=7, range=512:1024)
```

## Logical grid, physical grid, planner

The **logical grid** is the semantic domain the Python kernel is written
against (`sl.segment_id(0)` indexes it). The **physical grid** is the GPU
task domain actually launched. The **planner** converts segment programs
plus runtime extents (actual lengths, counts, skew) into tasks and tiles —
choosing packing, bucketing, partitioning, and static or persistent
scheduling.

The research bet: because the programmer specified *segment-local
semantics* rather than a schedule, the planner is free to re-derive the
schedule as the length distribution changes — without the kernel changing.

## What goes wrong when the levels blur

- *Segment = tile* (padding): every segment padded to the longest one;
  memory and FLOPs wasted on skewed distributions.
- *Segment = task* (one CTA per segment): tiny segments underutilize
  entire CTAs; one huge segment serializes the tail of the launch.
- *Hardware ids in semantics* (thread-level programming): the schedule is
  frozen into the kernel; changing distribution means rewriting it.

Swage's verifiers enforce the separation mechanically: no thread/block ids
in the `swage` dialect, no runtime identity in types, no runtime-sized
register arrays.
