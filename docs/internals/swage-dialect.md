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
- the argument attribute `swage.role`, which says what an argument of a
  segment function is: `values`, `offsets`, `output`, `value_count`, or
  `segment_count`;
- reduction kinds `sum`, `max`, and `min` at the dialect level.

Region operations are isolated from above. Outer scalar values enter through
explicit `captures(...)` operands, in order. Ordinary scalar arithmetic uses
upstream `arith` and `math` operations inside regions. `extent`, `map`, `reduce`,
and `map_store` declare a read of their segment. `map_store` is the only Swage
operation that writes, and it writes only the segment's corresponding output
range.

A mapped segment is a lazy view that its consumer evaluates. The dialect
transform `--swage-fuse-maps` (`lib/Dialect/Swage/Transforms/FuseMaps.cpp`)
fuses a map that has one consumer into that consumer, and the segmented
lowerings apply the same rewrite to every map of an admitted function
before they emit code.

The type and operations parse, print, and verify independently of whether a
particular lowering admits them. Current segmented lowering supports a
narrower private subset described in [Segmented Reductions](segmented-reductions.md).

## Generated reference boundary

The detailed dialect and operation reference is generated from the TableGen
definitions in `include/swage/Dialect/Swage/IR`.

--8<-- "docs/reference/_generated/swage-dialect.inc"

--8<-- "docs/reference/_generated/swage-ops.inc"

Continue with [Textual Swage IR](../language/swage-ir.md) for the syntax,
the verifier contracts, and one complete module. For the semantic model go
back to [Execution Model](../user-guide/execution-model.md), and for the
registered lowering surface use
[Compiler Tools and Passes](compiler-tools.md).
