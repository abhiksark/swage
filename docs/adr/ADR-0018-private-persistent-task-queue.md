<!-- docs/adr/ADR-0018-private-persistent-task-queue.md -->
# ADR-0018: Private persistent task queue

- Status: proposed; predeclared performance gate failed
- Date: 2026-08-30
- Last evaluated: 2026-09-02

## Context

Private identity-sum execution currently submits a fused direct kernel,
followed by split partial and merge kernels when oversized segments exist.
CUDA schedules blocks within each launch, but the three launch-wide phases
cannot overlap. In particular, direct short work completes before split work
starts, and every split partial must complete before any merge launch starts.
Extreme skew can therefore leave a launch tail even though independent work
exists.

A persistent experiment must preserve the semantic segmented program, consume
the existing host-classified descriptors, publish split dependencies safely,
and fail directly rather than falling back to the static schedule. It must not
add a public segmented API or put device coordinates in semantic Swage IR.

## Decision

### Private execution boundary

One private lowering compiles the admitted identity f32 segmented sum into a
512-thread resident kernel. The host materializes the existing stable direct,
partial, and merge metadata before compilation or launch. No new semantic or
planning operation is introduced.

The kernel receives separate device arrays for warp IDs, CTA IDs, partial
ranges, partial-to-merge IDs, merge records, scratch, and queue/dependency
counters. Its private ABI is documented in
[Persistent Execution](../internals/persistent-execution.md).

The host launches at most 168 blocks on the qualification GPU, two per SM.
Fewer blocks launch when the materialized work contains fewer groups. Each
launch first resets its private counters on the current PyTorch stream.

### Queue and dependency protocol

Resident CTAs perform these phases without a grid-wide barrier:

1. thread zero atomically claims direct CTA tasks; the claim is broadcast to
   the block;
2. thread zero atomically claims split partial tasks;
3. each partial writes one unique scratch slot, executes an explicit
   GPU-scope release fence, then publishes completion to its merge group with
   an acquire-release atomic increment;
4. the CTA observing the final completion performs the group's only scratch
   merge and output store;
5. each physical warp independently claims batches of direct warp tasks, with
   lane zero broadcasting each claim inside its warp.

Split claims cover bounded batches of four materialized tasks and direct warp
claims cover bounded batches of eight. Every task in a claimed batch still
has one owner and one output or scratch slot.

A block that observes an empty phase proceeds while blocks already processing
that phase may continue. No worker spins on a dependency. Atomic claims give
exactly-once task ownership; the final completion count gives exactly one
merge owner. A GPU-scope acquire fence before the merge scratch loads pairs
with each publisher's explicit release fence. These fences are required
because the pinned NVPTX backend emits the completion RMW as a legacy
unqualified `atom.global.add` instruction even though lowered LLVM IR retains
`acq_rel`.

The counter reset and resident launch are one ordered stream sequence. Inputs,
outputs, descriptors, scratch, and counters are retained on that stream.
Unsupported semantics, malformed descriptors, invalid devices, allocation
failures, compilation failures, and launches report their errors without a
policy or backend fallback.

### Predeclared performance experiment

The frozen experiment is named `persistent-tail-skew` and uses exactly:

- NVIDIA RTX A6000 at `sm_86`, with 84 SMs;
- 32,768 segments;
- seed 7 in an isolated `random.Random` instance;
- 32,767 short lengths sampled uniformly and inclusively from 1 through 32;
- one final oversized length of 16,777,216 elements;
- no position shuffle, because classification materializes policy queues;
- warp limit 32 and CTA chunk limit 4096;
- 512 threads and 168 resident blocks for persistent execution;
- all-one f32 values, so every expected sum is exact;
- 25 warmups and 100 interleaved CUDA-event samples per policy.

Compilation, classification, allocation, and module loading occur before
timing. Timed static execution is the existing prepared `mixed` sequence:
fused direct work, split partial work, then split merge work. Timed persistent
execution includes both the device counter reset and resident kernel. The raw
record must retain every sample, generated statistics, task counts, exact
source revision, worktree cleanliness, tool versions, driver, and GPU facts.

The performance gate passes only when:

```text
persistent_median <= 0.95 * static_mixed_median
```

The input, schedule, and ratio must not be changed after seeing a result. A
failed gate remains recorded and persistent qualification remains incomplete.

## Recorded outcome

Detailed adversarial testing invalidated the original `87fb65e` measurement:
with two or three resident CTAs, a final publisher could observe completion
before another CTA's scratch store became globally visible. The raw run is
retained as `persistent-sum-a6000-sm86-87fb65e-invalid.json`, but it is not
qualification evidence.

After adding explicit device fences, Compute Sanitizer exposed a second race:
the partial phase could overwrite the shared queue-claim slot before every
thread had read the terminating CTA claim. The clean `f8bb0a9` run is retained
as `persistent-sum-a6000-sm86-f8bb0a9-invalid.json`, but it also predates the
correctness fix and is not qualification evidence.

With both publication fences and the CTA-to-partial phase barrier, the clean
`205f629` run measured a persistent median of 117.520 microseconds and a static
mixed median of 118.784 microseconds. The corrected persistent path was 1.06%
faster, but its ratio of 0.9894 did not meet the predeclared maximum of 0.95.
The gate therefore failed and this ADR remains proposed.

The canonical raw record is
[`persistent-sum-a6000-sm86.json`](https://github.com/abhiksark/swage/blob/main/benchmarks/results/persistent-sum-a6000-sm86.json).
Earlier clean runs at `c34ab5c`, `bb99d25`, `9625aae`, `87fb65e`, and
`f8bb0a9` are retained next to it rather than discarded, but they predate one
or both synchronization fixes and are excluded from semantic qualification.

## Acceptance boundary

The decision may become accepted only when tests establish:

- exact direct warp, direct CTA, partial, and merge ownership;
- empty, boundary, repeated-empty, mixed, split-only, and extreme-skew
  correctness against PyTorch and the sequential CPU oracle;
- one merge writer after all partial scratch values are published, including
  runs that poison every scratch slot before submission;
- repeated launch, non-default stream, CUDA graph, retention, device drift,
  invalid resident count, allocation failure, compile failure, and launch
  failure behavior;
- no deadlock or starvation under repeated extreme-skew execution;
- the committed clean A6000 record passes the predeclared performance gate;
- all applicable Python, native, lit, C++, documentation, and GPU tests pass.

This remains private qualification even if accepted. Public segment syntax and
public segmented launch require separate decisions and tests.

## Consequences

- Direct and dependency-bearing tasks can execute in one resident kernel.
- Completion state is explicit device data rather than implicit launch order.
- The protocol specializes to the canonical identity sum; it is not a general
  task graph or queue API.
- A 512-thread shape favors split throughput and supplies sixteen warp
  workers, but may be inferior for direct-only distributions. The frozen
  experiment decides the declared gate rather than post-result tuning.
- One prepared object has one mutable counter array, so concurrent reuse on
  different streams is unsupported.
- Persistent max, persistent softmax, cross-kernel queues, multi-stream
  scheduling, and a public scheduler remain deferred.
