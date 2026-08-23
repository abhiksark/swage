<!-- docs/adr/ADR-0014-m6-minimal-swage-plan-gate.md -->
# ADR-0014: M6 minimal SwagePlan gate

- Status: proposed
- Date: 2026-08-23

## Context

ADR-0003 separates segment semantics in `swage` from scheduling decisions in
`swage_plan`. M6 needs the smallest executable proof of that boundary without
claiming general scheduling or changing the public frontend and launch
contracts. The proof must preserve an admitted semantic kernel, describe its
legal policies at compile time, and classify runtime segment metadata without
executing either policy.

## Decision

### Planning dialect boundary

The M6 planning dialect contains only:

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

M6 does not lower either policy to GPU execution and does not dispatch the
returned task range.

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

M6 is accepted when tests prove all of the following:

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

M6 does not add:

- packed warps, split CTAs, partial reductions, or merges;
- queues, dispatch, mixed-policy GPU execution, or benchmarks;
- ragged-softmax planning;
- public frontend, emission, or launch support;
- releases or tags;
- unrelated M1 backlog work.

## Consequences

- The first planning IR is intentionally limited to one classifier proof for
  one semantic program shape.
- Runtime metadata controls policy selection without placing runtime segment
  identity in types or hardware indices in semantic IR.
- M6 establishes no performance claim because it performs no mixed-policy GPU
  execution or benchmark comparison.
- Later milestones must extend this contract explicitly rather than treating
  deferred policies or execution as implied support.
