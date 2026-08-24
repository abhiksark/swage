<!-- docs/reference/swage-plan-dialect.md -->

# SwagePlan Dialect

!!! warning "Private qualification"

    `swage_plan` is an internal compiler boundary for one admitted identity
    segmented sum. It is not a public Python API or a general task scheduler.

The current dialect contains only:

- `#swage_plan.policy<warp>` and `#swage_plan.policy<cta>`;
- `!swage_plan.task_range`, an opaque result type;
- `swage_plan.classify`, which records one semantic kernel, runtime offset and
  count operands, the legal policy order, a warp limit, and a CTA chunk limit.

The operation verifies its symbol, input ABI, limits, and policy list. Runtime
offset contents remain unknown to compiler passes. Private materialization
uses validated host metadata to construct direct segment IDs and M8 split
records. The dialect does not define packed-warp policies, queues, dependency
execution, persistent scheduling, or a general task-range lowering.

There is no public `mlir_swage.dialects.swage_plan` Python module contract.

## Generated reference boundary

The detailed dialect and operation reference is generated from TableGen in
`include/swage/Dialect/SwagePlan/IR`.

--8<-- "docs/reference/_generated/swage-plan-dialect.inc"

--8<-- "docs/reference/_generated/swage-plan-ops.inc"

Continue with [Private Qualification](../qualification/private-m4-m8.md) for
the M6 to M8 execution boundary or [Compiler Tools and Passes](compiler-tools.md)
for the registered planning pass.
