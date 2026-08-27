<!-- docs/internals/planning.md -->

# Task Planning

The SwagePlan gate turns an admitted identity segmented sum into
classified tasks without executing them. This page records the
exact internal contracts; none of them is a public API.

*Qualified under the M6 gate on NVIDIA RTX A6000 (`sm_86`);
see [`ROADMAP.md`](https://github.com/abhiksark/swage/blob/main/ROADMAP.md) and [ADR-0014](../adr/ADR-0014-m6-minimal-swage-plan-gate.md).*

`--swage-to-plan` admits only a capture-free, map-free, single-stage identity
segmented sum. Admission is read-only. On success it preserves the semantic
function and adds a private companion with `swage_plan.classify`.

The default legal policy order is warp then CTA. The default warp limit is 32
elements, and the default CTA chunk limit is 4096 elements. The host
classifier validates signed i32 counts and monotonic offsets before producing
stable descriptors. The planning gate does not execute a policy.

<div class="doc-figure" tabindex="0" markdown="1">

![Per-segment lengths classified into warp, CTA, and split task lists](../assets/figures/plan-classification.svg)

</div>

*SwagePlan classification buckets, including the split bucket, and the validated planning-limit invariant. [Open the full-size figure](../assets/figures/plan-classification.svg).*

Continue with [Task Execution](task-execution.md) for the qualified
warp, CTA, and fused paths, or the
[SwagePlan Dialect](swage-plan-dialect.md) for the planning IR.
