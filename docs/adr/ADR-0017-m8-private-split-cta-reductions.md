<!-- docs/adr/ADR-0017-m8-private-split-cta-reductions.md -->
# ADR-0017: M8 private split-CTA reductions

- Status: accepted
- Date: 2026-08-24

## Context

M7 executes one canonical identity segmented sum with private warp, CTA, and
fused mixed schedules. A segment assigned to CTA still runs in one block, so
an oversized segment can dominate the launch tail. M8 needs the smallest task
decomposition that proves complete multi-CTA coverage without adding another
policy, widening the public API, or changing the frozen M7 benchmark.

## Decision

### Planning and descriptor boundary

`swage_plan.classify` retains warp then CTA as its complete policy list and
adds `cta_chunk_elements`. The value defaults to 4096. Planning and host
classification require:

```text
0 < warp_max_elements <= cta_chunk_elements <= INT32_MAX
```

Segments no longer than `warp_max_elements` receive one direct warp
descriptor. Longer segments no greater than `cta_chunk_elements` receive one
direct CTA descriptor. An oversized segment receives ordered stage-zero CTA
descriptors that partition its absolute input range into chunks no larger
than `cta_chunk_elements`, followed by one stage-one CTA merge descriptor.

Stage-zero `begin` and `end` fields are absolute value indices. Stage-one
`begin` and `end` fields are compact half-open scratch indices covering every
partial for that segment. Every descriptor uses the segment ID as its
dependency group. All stage-zero descriptors are ordered by segment and chunk
start before all stage-one descriptors. Validation rejects malformed counts,
ranges, descriptor counts, scratch indices, and i32 narrowing before returning
any descriptor.

The private materialization boundary returns four flat i32 arrays: direct warp
segment IDs, direct CTA segment IDs, partial `[begin, end]` records, and merge
`[segment_id, partial_begin, partial_end]` records. This adds no operation,
type, policy, or public Python API.

The partial and merge lowering passes are available only through private
compiler factories. They are intentionally not registered as public
`swage-opt` pass arguments.

### Private GPU execution

The partial kernel uses 128 threads and this ABI:

```text
values*, partial_ranges*, scratch*, value_count:i32, partial_count:i32
```

Each block reduces one explicit absolute input range and writes one unique
scratch slot. The merge kernel also uses 128 threads and this ABI:

```text
scratch*, output*, merge_records*, partial_count:i32, merge_count:i32
```

Each block reduces one compact scratch range and thread zero writes the final
segment result once. Both kernels reuse the existing three-pointer, two-i32
CUDA Driver launch marshaller and the current PyTorch stream.

Pure `warp` and `cta` preparation continues to execute every segment as a
correctness control. Mixed execution submits direct fused work, partial CTAs,
then merge CTAs in that order on one current stream, skipping empty phases.
When no segment is split, mixed execution preserves the M7 one-launch path.
Validation and classification finish before compilation or allocation, and
failures do not select another policy or backend.

### Qualification gate

M8 is a correctness gate on NVIDIA RTX A6000 at `sm_86`. Exact all-one sums
and tolerant nontrivial f32 sums must match the sequential CPU oracle and
PyTorch. Tests cover boundary lengths, exact chunk multiples, one-element
remainders, repeated empty segments, split-only, direct-only, mixed and zero
work, stream ordering, repeated launches, retention, aliases, device drift,
and compile, allocation, and launch failures.

The M7 benchmark inputs, recorded result, and `1.05` gate do not change.

## Consequences

- M8 proves split-CTA identity sums without defining a third scheduling
  policy or a general task graph.
- Scratch storage is proportional to the number of partial chunks and is
  private to one prepared launch.
- The ordered stream boundary provides the only dependency mechanism needed
  for this two-stage reduction.
- Split max, split softmax, packed warps, queues, persistent scheduling, and
  public segmented execution remain outside M8.
