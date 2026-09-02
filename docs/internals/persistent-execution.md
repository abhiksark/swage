<!-- docs/internals/persistent-execution.md -->

# Persistent Execution

The experimental persistent identity-sum path uses resident CTAs to drain
device task queues and resolve split dependencies in one kernel. It consumes
the same host-classified task metadata as static mixed execution. This is a
private experiment, not a public API or completed qualification.

!!! warning "Performance gate failed"

    Correctness tests exercise the implementation, but the best clean NVIDIA
    RTX A6000 run was 2.53% faster than static mixed execution and missed the
    predeclared 5% requirement. Consequently
    [ADR-0018](../adr/ADR-0018-private-persistent-task-queue.md) remains
    proposed and no current release status depends on this path.

## Private ABI

The 512-thread kernel receives ten pointers followed by five signed-i32
counts:

```text
values*, offsets*, output*,
warp_task_ids*, cta_task_ids*,
partial_ranges*, partial_merge_ids*, merge_records*,
scratch*, counters*,
value_count:i32, warp_count:i32, cta_count:i32,
partial_count:i32, merge_count:i32
```

The flat record layouts remain those of [Split Execution](split-execution.md):
partial ranges are `[begin, end]` pairs and merge records are
`[segment_id, partial_begin, partial_end]` triples. `partial_merge_ids[i]`
identifies the merge record that depends on scratch slot `i`.

The counter array has this layout:

```text
[warp_claim, cta_claim, partial_claim, merge_completion...]
```

Preparation validates and materializes all metadata, allocates scratch and
counters, compiles and loads the kernel, and records a task-readiness event.
Every launch resets its private counters on the current PyTorch stream before
submitting the resident kernel. The reset is part of timed execution rather
than hidden preparation.

## Resident workers

The default launch requests two 512-thread blocks per SM and caps that count
at the number of available work groups. On the qualification RTX A6000 this
is at most 168 resident blocks. “Resident” describes a bounded physical grid
whose blocks claim multiple tasks; it does not promise that CUDA can
simultaneously place every requested block.

Each block proceeds through three queues without a grid-wide barrier:

1. **Direct CTA queue.** Thread zero atomically claims one segment ID and
   broadcasts it to the block. All threads perform a block-stride reduction.
2. **Partial queue.** Thread zero claims up to four consecutive materialized
   ranges. The block reduces each range and thread zero writes each task's
   unique scratch slot.
3. **Warp queue.** Each of the block's sixteen physical warps independently
   claims up to eight consecutive segment IDs. Lane zero broadcasts a claim
   within its warp.

A worker advances when it observes a queue empty even if other workers are
still completing already-claimed work. This permits short direct work to
overlap the tail of split work.

## Dependency publication

After a partial scratch store, the block converges and thread zero performs an
acquire-release atomic increment on that merge group's completion counter.
Only thread zero reads the dependency metadata. It broadcasts a ready merge
ID only when its increment completes the group; otherwise it broadcasts a
sentinel. Exactly one CTA therefore reduces the group's compact scratch range
and writes the segment output.

This protocol has no dependency spin loop:

- atomic queue increments give every task index one claimant;
- every partial has one scratch slot and one merge group;
- acquire-release publication orders scratch stores before the merge;
- only the final publisher owns the merge and output store;
- workers with no remaining queue work terminate.

The protocol is specialized to the admitted identity f32 sum. It is not a
general device task graph, and it does not establish persistent max or
softmax.

## Stream and graph behavior

Queue reset, resident execution, and tensor retention use the current PyTorch
stream. Launching on another device after preparation is rejected. CUDA graph
capture is supported after one ordinary initialized launch, matching the
prepared static path's task-readiness contract.

Failures do not fall back to static mixed execution. Unsupported semantics,
invalid metadata, invalid residency, compilation errors, allocation errors,
and CUDA launch errors propagate to the caller.

Continue with [Split Execution](split-execution.md) for static stage ordering,
[Task Execution](task-execution.md) for direct fixed tiles, or
[Verification](verification.md) for the current evidence boundary.
