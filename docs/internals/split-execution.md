<!-- docs/internals/split-execution.md -->

# Split Execution

Segments longer than the CTA chunk limit split into ordered partial
tasks and one merge. This page records the exact internal
contracts; none of them is a public API.

*Qualified on NVIDIA RTX A6000 (`sm_86`); see
[Verification](verification.md) and
[ADR-0017](../adr/ADR-0017-private-split-cta-reductions.md) and
[ADR-0019](../adr/ADR-0019-composable-private-reductions.md).*

Segments of at most 32 elements receive one direct warp descriptor. Segments
from 33 through 4096 elements receive one direct CTA descriptor. A longer
segment receives ordered stage-zero CTA chunks of at most 4096 input elements
and one stage-one merge descriptor. These are current defaults and must satisfy
the planning-limit invariant.

The figure uses one oversized identity-sum segment over absolute input range
`[100, 9500)`. Its three ordered partial CTAs own `[100, 4196)`,
`[4196, 8292)`, and `[8292, 9500)`. Each partial has one unique scratch
writer. The merge record names segment 7 and compact scratch range `[0, 3)`,
then one writer stores `output[7]`. Mixed execution submits direct fused work,
partial CTAs, and merge CTAs in that order on the current stream, skipping any
empty phase. This lifecycle supports private capture-free, single-stage
sum, max, and min over f32 or f64 values, including fused map chains. The
scratch has the element type of the program. Partial tasks evaluate the
element program on input values; merges combine scratch using the reduction
kind without reapplying that program. It does not support split softmax.

A mean is split as its sum. A partial task stores the raw sum of its chunk
and never divides. The merge sums the partial sums and divides once by the
extent of the segment, so no mean of partial means is formed.

<div class="doc-figure" tabindex="0" markdown="1">

![Absolute split ranges, unique scratch writers, and one merge writer](../assets/diagrams/split-lifecycle.svg)

</div>

*Private ownership and launch order for one split segment. [Open the full-size figure](../assets/diagrams/split-lifecycle.svg).*

Partial ABI:

```text
values*, partial_ranges*, scratch*, value_count:i32, partial_count:i32
```

Merge ABI:

```text
scratch*, output*, merge_records*, partial_count:i32, merge_count:i32,
segment_count:i32
```

Merge ABI of a program that divides by the extent of its segment:

```text
scratch*, output*, merge_records*, partial_ranges*, partial_count:i32,
merge_count:i32, segment_count:i32
```

The bound range of a merge is scratch, whose extent is a number of partial
results. The extent of the segment is not in a merge record, so this merge
kernel reads it from the range records the partial kernel reads: the chunks
of one segment are consecutive records, and the extent is the end of the
last one minus the begin of the first. The two records are addressed
through the range of partials after its clamp to `partial_count`, so a
stray merge record reads no range record outside the buffer, and an empty
range of partials reads none and has the extent zero. No existing kernel,
record, or launch tuple changed for this: the kernels of a sum, a maximum,
and a minimum keep the merge ABI above.

Both kernels use 512 threads, sized so one 4096-element chunk fully
occupies a CTA at eight elements per thread. Partial ranges are absolute
half-open input ranges, and each partial writes one unique scratch slot.
Merge records carry a segment ID and a compact half-open scratch range;
thread zero writes the final segment result once. The merge kernel compares
the segment ID with `segment_count` and stores nothing for an ID outside it.

If no split exists, the direct one-launch path remains unchanged. On NVIDIA
RTX A6000 `sm_86`, split sums of position-dependent values that are exact in
f32 match PyTorch and the sequential CPU oracle bit for bit, and split sums
of random f32 values stay within the bound stated in
[Sum rounding](segmented-reductions.md#sum-rounding).

Split execution does not implement packed warps, split softmax, captured
stages, device queues, persistent scheduling, or public segment syntax. The
public `swage.segment_reduce` reaches it for segments above the default
4096-element chunk limit, unless automatic selection replaces it by the CTA
schedule.

Continue with [Persistent Execution](persistent-execution.md) for the
experimental kernel that consumes the same partial and merge records in one
launch, or [Verification](verification.md) for the executable evidence
behind these claims.
