<!-- docs/adr/ADR-0014-minimal-swage-plan-gate.md -->
# ADR-0014: Minimal SwagePlan gate

- Status: accepted; extended by
  [ADR-0015](ADR-0015-minimal-mixed-policy-execution.md),
  [ADR-0017](ADR-0017-private-split-cta-reductions.md), and
  [ADR-0019](ADR-0019-composable-private-reductions.md); its planning
  dialect boundary is superseded by
  [ADR-0020](ADR-0020-planned-per-function-lowering.md); see
  [Later decisions](#later-decisions)
- Date: 2026-08-23

## Context

ADR-0003 separates segment semantics in `swage` from scheduling decisions in
`swage_plan`. The compiler needs the smallest executable proof of that
boundary without claiming general scheduling or changing the public frontend
and launch contracts. The proof must preserve an admitted semantic kernel,
describe its legal policies at compile time, and classify runtime segment
metadata without
executing either policy.

## Decision

### Planning dialect boundary

The private planning dialect contains only:

- `#swage_plan.policy<warp|cta>` for the two legal policies;
- `!swage_plan.task_range` for one runtime-produced descriptor range;
- `swage_plan.classify` for classification of one semantic kernel.

`swage_plan.classify` takes rank-one i32 offsets plus i32 value and segment
counts. It references the semantic kernel, records `warp_max_elements`, exposes
the legal policy order as warp then CTA, and returns one
`!swage_plan.task_range`. The operation does not contain runtime offset values
or materialized task descriptors.

The `--swage-to-plan` pass defaults `warp-max-elements` to 32. Its input module
contains exactly one single-block, void `func.func` with the existing
five-argument ABI, in order: rank-one f32 values, rank-one i32 offsets,
rank-one f32 output, i32 value count, and i32 segment count. The block contains
exactly one axis-zero `swage.segment_id`, one `swage.make_segment` binding the
values and offsets arguments at that segment ID, one capture-free
`swage.reduce` with `kind<sum>`, one scalar `memref.store` of that result to
`output[segment_id]`, and one void `func.return`. The reduction has one block
with one f32 element argument and only `swage.yield` of that same argument.

The pass therefore rejects max reductions, transformed sums, maps, captures,
multiple reductions, map-store terminals, extra operations, and every other
function, semantic, or ABI shape. Admission is read-only analysis. Every
unsupported input fails before module mutation, so failure leaves no companion
function or other partial Plan IR. After admission, the pass preserves the
semantic function and adds one private `<kernel>__swage_plan` companion
containing the classification operation. The existing SCF and GPU paths reuse
the same analysis, with optional region detachment occurring only after
admission succeeds.

Compatibility note: the generated-contract boundary later removed the two
runtime counts from the semantic function signature. Current semantic inputs
are only values, offsets, and output; the planning companion derives both
counts from memref dimensions, while host classification continues to receive
explicit wide counts for validation. The admitted operations and read-only
failure boundary above are otherwise unchanged.

### Compile-time and runtime responsibilities

Compile time:

- verifies the canonical semantic program and its ABI;
- records the semantic kernel reference and `warp_max_elements`;
- records warp then CTA as the complete legal policy order;
- emits the private planning companion without changing the semantic function.

Runtime:

- validates the actual offsets, value count, and segment count;
- computes each absolute segment range from adjacent offsets;
- selects warp or CTA from the actual segment length;
- returns stable task descriptors or an error without fallback.

This compile-only boundary does not lower either policy to GPU execution or
dispatch the returned task range.

### Host classifier contract

The internal host classifier returns
`llvm::Expected<SmallVector<TaskDescriptor>>`. Each `TaskDescriptor` contains:

- i32 `segment_id`;
- absolute half-open i32 `begin` and `end`;
- i32 `stage`;
- generated `TaskPolicy` `policy`;
- i32 `dependency_group`.

Valid metadata has a nonnegative i32 value count, segment count, and
`warp_max_elements`. Offsets are nonnegative signed i32 values, contain exactly
`segment_count + 1` entries, start at zero, are nondecreasing, and end no later
than the value count. The entry-count relationship is checked in a wider type
before addition, and every range length and descriptor field is computed in a
wider type and checked before conversion to i32.

Validation completes before descriptor construction. The classifier then
emits one descriptor per segment, including empty segments. Offsets `[0]` with
a zero segment count emit no descriptors. A segment uses warp when
`end - begin <= warp_max_elements`; otherwise it uses CTA. Every descriptor has
`stage` equal to zero and `dependency_group` equal to `segment_id`. Descriptor
order is segment order. Any violation returns an error with no descriptors and
does not fall back to another policy or backend.

## Acceptance boundary

The boundary is accepted when tests prove all of the following:

- the policy attribute, task-range type, and classify operation round-trip and
  reject invalid forms;
- the conversion accepts only the canonical identity segmented sum and leaves
  unsupported modules unchanged;
- the semantic function remains present and the private planning companion
  records the kernel reference, threshold, warp-then-CTA order, and one task
  range;
- the classifier returns exact stable descriptors for empty, boundary, skewed,
  and alternating segment distributions;
- malformed and out-of-i32 metadata returns an error without fallback or
  partial output.

## Explicit deferrals

This decision does not add:

- the Issue #13 `task`, `pack`, `partition`, or `make_task` dialect surface;
- packed warps, split CTAs, partial reductions, or merges;
- queues, dispatch, mixed-policy GPU execution, or benchmarks;
- ragged-softmax planning;
- public frontend, emission, or launch support;
- releases or tags;
- unrelated semantic-dialect backlog work.

## Consequences

- The first planning IR is intentionally limited to one classifier proof for
  one semantic program shape.
- Runtime metadata controls policy selection without placing runtime segment
  identity in types or hardware indices in semantic IR.
- This boundary establishes no performance claim because it performs no
  mixed-policy GPU execution or benchmark comparison.
- Later changes must extend this contract explicitly rather than treating
  deferred policies or execution as implied support.

## Later decisions

This record describes the gate as first accepted. Four of its statements
no longer match the code, and each is replaced by a later record:

- Dialect boundary: this record says the dialect holds the policy
  attribute, `!swage_plan.task_range`, and `swage_plan.classify`, and that
  the pass adds a `<kernel>__swage_plan` companion.
  [ADR-0020](ADR-0020-planned-per-function-lowering.md) removes the type,
  the operation, and the companion. Plan IR is now a plan function with one
  task operation that the GPU lowering consumes, and the planning limits
  are arguments of the host classifier. The split of responsibilities
  between compile time and run time, and the host descriptors, stand.

- Admission: this record says the pass rejects max reductions, transformed
  sums, and maps.
  [ADR-0019](ADR-0019-composable-private-reductions.md) admits one
  capture-free f32 sum or max with optional single-consumer map chains.
- Descriptors: this record says a segment is warp or CTA, that
  `warp_max_elements` may be zero, and that every descriptor has `stage`
  equal to zero.
  [ADR-0017](ADR-0017-private-split-cta-reductions.md) adds
  `cta_chunk_elements`, requires
  `0 < warp_max_elements <= cta_chunk_elements`, and gives a longer segment
  ordered stage-zero chunk descriptors plus one stage-one merge descriptor.
- Execution: this record says the boundary lowers no policy and dispatches
  no task range.
  [ADR-0015](ADR-0015-minimal-mixed-policy-execution.md) and
  [ADR-0016](ADR-0016-fused-mixed-policy-schedule.md) add private warp, CTA,
  and fused mixed execution of the classified tasks.

Captures, multiple reductions, and map-store terminals are still rejected.
[Task Planning](../internals/planning.md) states the current contract.
