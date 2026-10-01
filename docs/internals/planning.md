<!-- docs/internals/planning.md -->

# Task Planning

The SwagePlan gate turns an admitted capture-free sum or max into
classified tasks without executing them. This page records the
exact internal contracts; none of them is a public API.

*Qualified through compile-only compiler and classifier tests; see
[Verification](verification.md) and
[ADR-0019](../adr/ADR-0019-composable-private-reductions.md).*

`--swage-to-plan` admits a capture-free, single-stage f32 sum or max with
optional single-consumer map chains and a scalar output per segment. Element
regions use the existing admitted arithmetic and `math.exp2` operations.
Admission is read-only. On success it preserves the semantic function and adds
a private companion with `swage_plan.classify`. Captures, multiple reductions,
and map-store outputs remain outside static planning. Persistent execution
retains its separate identity-sum restriction.

The default legal policy order is warp then CTA. The default warp limit is 32
elements, and the default CTA chunk limit is 4096 elements. The host
classifier validates signed i32 counts and monotonic offsets before producing
stable descriptors. The planning gate does not execute a policy. Segments
above the CTA chunk limit classify as split work;
[Split Execution](split-execution.md) records the decomposition and the
validated planning-limit invariant.

The private `_prepare_planned_reduction` helper can select the existing pure
CTA implementation after validating the native plan. With default 4096-element
chunks, it avoids splitting when every segment has 4097–8192 elements and the
batch has at least as many segments as the device has SMs. The selected
`mixed` callable aliases `cta`, and preparation skips split kernel compilation
and scratch allocation. Selection does not execute or time the program.

The element program must also fit a 32-unit relative work budget. The helper
inspects typed operations in the admitted native MLIR: add, subtract, multiply,
minimum, and maximum cost one unit each; `math.exp2` costs eight and division
costs sixteen. Constants and yields are free. Work is counted across all map
and reduction regions, so splitting a long expression into maps does not
bypass the limit. These empirically chosen weights are scheduling hints, not
GPU instruction latency estimates. Larger programs retain split execution.

This is a conservative rule measured on A6000 and RTX 5090, not a general cost
model. Sparse batches, mixed direct/split batches, longer segments, and custom
chunk sizes retain the fixed schedule. `select_schedule=False` explicitly
disables selection. The legacy `_prepare_planned_sum` wrapper always uses the
fixed policy, preserving existing qualification and frozen benchmark inputs.
Native classification, descriptors, and public APIs are unchanged.

<div class="doc-figure" tabindex="0" markdown="1">

![Per-segment lengths classified into warp, CTA, and split task lists](../assets/figures/plan-classification.svg)

</div>

*SwagePlan classification buckets, including the split bucket, and the validated planning-limit invariant. [Open the full-size figure](../assets/figures/plan-classification.svg).*

Continue with [Task Execution](task-execution.md) for the qualified
warp, CTA, and fused paths. The planning IR itself is on the previous page,
[SwagePlan Dialect](swage-plan-dialect.md).
