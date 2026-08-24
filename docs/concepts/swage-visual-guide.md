<!-- docs/concepts/swage-visual-guide.md -->

# Swage, Visually

Swage starts with ragged storage: one dense values buffer plus offsets that
divide it into logical segments.

```text
values  = [ 4.0, 1.0 | | 3.0, 8.0, 2.0 ]
offsets = [ 0,        2, 2,            5 ]
```

Segment `i` is the half-open slice
`values[offsets[i] : offsets[i + 1]]`. Repeated offsets create an empty
segment. The elements inside each nonempty segment remain dense even though
segment lengths differ.

## Three questions, three levels

```text
Segment: What logical data and computation does one program instance mean?
    |
    v
Task:    What schedulable work is needed for the observed segment length?
    |
    v
Tile:    What fixed warp or CTA step executes that task?
```

A segment is runtime-sized. A task is derived work. A tile is a fixed
physical step. Keeping the levels separate prevents the runtime shape from
leaking into semantic types or hard-coding GPU indices into the program.

## Why the separation matters

If a segment were treated as a fixed tile, short rows would be padded and
long rows would not fit. If a segment always became one task, tiny rows could
underuse a CTA and a very long row could dominate the launch tail. If GPU IDs
were part of semantic IR, changing the schedule would require rewriting the
kernel's meaning.

Swage instead preserves the segment-local meaning and allows task derivation
to be qualified separately. Today, that separation is public for canonical
fixed vector add and privately qualified for selected segmented modules.
General public segmented execution remains planned.

Continue with [Segments, Tasks, and Tiles](segments-tiles-tasks.md) for the
invariants and current qualification boundary. Then use
[Compiler Pipeline](../architecture/compiler-pipeline.md) to follow the
implemented data flow.
