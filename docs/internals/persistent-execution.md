<!-- docs/internals/persistent-execution.md -->

# Persistent Execution

The experimental persistent identity-sum path uses resident CTAs to drain
device task queues and resolve split dependencies in one kernel. It consumes
the same host-classified task metadata as static mixed execution. This is a
private experiment, not a public API or completed qualification.

!!! warning "Performance gate failed"

    Correctness tests exercise the implementation, but the semantically
    qualified NVIDIA RTX A6000 run was 1.06% faster than static mixed
    execution and missed the
    predeclared 5% requirement. Consequently
    [ADR-0018](../adr/ADR-0018-private-persistent-task-queue.md) remains
    proposed and no current release status depends on this path.

## Private ABI

The 512-thread kernel receives ten pointers followed by six signed-i32
counts:

    values*, offsets*, output*,
    warp_task_ids*, cta_task_ids*,
    partial_ranges*, partial_merge_ids*, merge_records*,
    scratch*, counters*,
    value_count:i32, warp_count:i32, cta_count:i32,
    partial_count:i32, merge_count:i32, segment_count:i32

The flat record layouts remain those of [Split Execution](split-execution.md):
partial ranges are `[begin, end]` pairs and merge records are
`[segment_id, partial_begin, partial_end]` triples. `partial_merge_ids[i]`
identifies the merge record that depends on scratch slot `i`.

Each count bounds what the kernel loads from the buffers:

- `value_count` bounds every range into `values`.
- `warp_count` bounds every claim on the warp queue.
- `cta_count` bounds every claim on the direct CTA queue.
- `partial_count` bounds every claim on the partial queue and every merge
  range into `scratch`.
- `merge_count` bounds every merge ID loaded from `partial_merge_ids`.
- `segment_count` bounds every segment ID loaded from a task queue or a
  merge record.

The counter array has this layout:

    [warp_claim, cta_claim, partial_claim, merge_completion...]

Preparation validates and materializes all metadata, allocates scratch and
counters, compiles and loads the kernel, and records a task-readiness event.
The compiled kernel is memoized per target and the loaded module per CUDA
context, so a later preparation compiles and loads nothing while the two
memos still hold the kernel. Each memo keeps 128 entries, and a module is
unloaded once no prepared launch holds it;
[Runtime and Environment](../reference/runtime-environment.md#module-lifetime)
states the rules. The prepared object keeps its kernel loaded for as long as
it is referenced.
Every launch resets its private counters on the current PyTorch stream before
submitting the resident kernel. The reset is part of timed execution rather
than hidden preparation.

## Resident workers

The default launch requests two 512-thread blocks per SM and caps that count
at the number of available work groups. On the qualification RTX A6000 this
is at most 168 resident blocks. "Resident" describes a bounded physical grid
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
overlap the tail of split work. A block barrier separates the direct-CTA and
partial phases because they reuse one shared claim-broadcast slot; it prevents
the first partial claim from racing the final CTA-claim read.

## Dependency publication

After a partial scratch store, the block converges and thread zero executes a
GPU-scope memory fence before performing an acquire-release atomic increment
on that merge group's completion counter. Only thread zero reads the
dependency metadata. It broadcasts a ready merge ID only when its increment
completes the group; otherwise it broadcasts a sentinel. Before any lane of
the final CTA reads scratch, it executes a second GPU-scope fence. Exactly one
CTA therefore reduces the group's compact scratch range and writes the
segment output.

The explicit fences are intentional. For the pinned LLVM/NVPTX path, the
lowered LLVM operation retains `acq_rel`, but PTX emission uses the legacy
unqualified `atom.global.add` spelling. Poisoned-scratch tests with two and
three resident CTAs exposed stale reads without the explicit `membar.gl`
pair.

This protocol has no dependency spin loop:

- atomic queue increments give every task index one claimant;
- every partial has one scratch slot and one merge group;
- explicit GPU fences and acquire-release publication order scratch stores
  before the merge;
- only the final publisher owns the merge and output store;
- workers with no remaining queue work terminate.

The protocol is specialized to the admitted identity f32 sum. It is not a
general device task graph, and it does not establish persistent max or
softmax.

## Stream and graph behavior

Queue reset, resident execution, and tensor retention use the current PyTorch
stream. Launching on another device after preparation is rejected, and so is
launching in a CUDA context other than the one the object was prepared in,
because the kernel is loaded in one context. A thread that has no current
CUDA context is given the context of the prepared device before that
comparison. A launch raises if the values, offsets, or output tensor was
rebound to other storage after preparation, and if the offsets tensor was
modified in place, which it detects through the tensor version counter. The
kernel clamps
every loaded range that indexes the values buffer to the value count and
every merge range that indexes scratch to the partial count. It skips a
segment ID outside the segment count and a merge ID outside the merge count
(ADR-0012). CUDA
graph capture is supported after an ordinary launch that observed task
storage ready, matching the prepared static path's task-readiness contract.
A first launch that only queued the wait for task storage does not count, so
the protocol is launch, synchronize, launch again, then capture.

One prepared object owns one counter array and one scratch buffer, so it
admits one launch at a time. The launch enforces that itself:

- It raises `RuntimeError` when another thread is inside a launch of the
  same object.
- It raises `RuntimeError` when an earlier launch of the object is still in
  flight on another stream. Synchronize that stream first.
- It admits a launch on the stream of the earlier one, because the stream
  orders the two.

Two cases are not checked. A launch recorded during a CUDA graph capture
skips the in-flight check, because CUDA forbids the event query during a
capture. A replay of a captured graph runs no host code, so nothing checks
it. A caller that replays a graph must keep replays of one prepared object
from overlapping.

Failures do not fall back to static mixed execution. Unsupported semantics,
invalid metadata, invalid residency, compilation errors, allocation errors,
and CUDA launch errors propagate to the caller.

Continue with [Compiler Tools and Passes](compiler-tools.md) for the driver
options that select each lowering mode, or [Verification](verification.md)
for the current evidence boundary. The static paths this kernel is compared
with are on [Task Execution](task-execution.md) and
[Split Execution](split-execution.md).
