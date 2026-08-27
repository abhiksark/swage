<!-- docs/internals/swage-dialect.md -->

# Swage Dialect

The `swage` dialect carries schedule-free semantic operations. It models a
logical fixed-block coordinate and symbolic runtime-sized segments without
placing GPU thread or block IDs in semantic IR.

## Current surface

The dialect currently defines:

- `!swage.segment<T>`, which carries only an element type;
- `swage.program_id`, a logical fixed-block coordinate;
- `swage.segment_id`, a logical segment coordinate;
- `swage.make_segment`, which binds values, offsets, and segment identity;
- `swage.extent`, which returns a runtime segment length;
- region-based `swage.map`, `swage.reduce`, and `swage.map_store`;
- `swage.yield`, the region terminator;
- reduction kinds `sum`, `max`, and `min` at the dialect level.

Region operations are isolated from above. Outer scalar values enter through
explicit `captures(...)` operands, in order. Ordinary scalar arithmetic uses
upstream `arith` and `math` operations inside regions. `map_store` is the only
effectful Swage operation and writes only the segment's corresponding output
range.

The type and operations parse, print, and verify independently of whether a
particular lowering admits them. Current segmented lowering supports a
narrower private subset described in [Segmented Reductions](segmented-reductions.md).

## Generated reference boundary

The detailed dialect and operation reference is generated from the TableGen
definitions in `include/swage/Dialect/Swage/IR`.

--8<-- "docs/reference/_generated/swage-dialect.inc"

--8<-- "docs/reference/_generated/swage-ops.inc"

Continue with [Segments, Tasks, and Tiles](../user-guide/execution-model.md)
for the semantic model, or [Compiler Tools and Passes](compiler-tools.md) for
the registered lowering surface.
