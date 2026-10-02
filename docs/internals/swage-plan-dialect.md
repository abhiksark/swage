<!-- docs/internals/swage-plan-dialect.md -->

# SwagePlan Dialect

!!! warning "Private qualification"

    `swage_plan` is an internal compiler boundary between the segment
    semantics of `swage` and a kernel. It is not a public Python API or a
    general task scheduler.

The dialect holds what a kernel lowering consumes:

- the function attribute `swage_plan.block_threads`, which makes a
  `func.func` a plan function and gives the launch width of its kernel in
  threads;
- `swage_plan.tasks`, the task operation of a kernel that reduces one
  segment per task, with or without a task buffer;
- `swage_plan.partial_tasks`, the task operation of the first stage of a
  split reduction, which reduces one chunk of a long segment per task into
  a scratch slot;
- `swage_plan.yield`, the terminator of a task region;
- `#swage_plan.policy<warp>` and `#swage_plan.policy<cta>`, which say how
  the threads of a task combine their partial results, and
  `#swage_plan.policy<sequential>`, the policy of the CPU oracle.

A plan function has the parameter list of its kernel as its signature and
one task operation, followed by a return, as its body:

```mlir
func.func @segmented_sum(
    %values: memref<?xf32>, %offsets: memref<?xi32>,
    %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
    attributes {swage_plan.block_threads = 128 : i32} {
  swage_plan.tasks policy<cta>
      segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
      value_count(%value_count : i32) segment_count(%segment_count : i32)
      into(%output : memref<?xf32>) {
  ^bb0(%segment: !swage.segment<f32>):
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    swage_plan.yield %sum : f32
  }
  return
}
```

The region of the task operation runs once per task, on the segment the
task binds. It holds the `swage.reduce` and `swage.map_store` operations of
the program, in the order the kernel runs them, and nothing else: the
planner has fused every map into its consumer, and the task operation has
absorbed the segment id, the segment construction, and the scalar store.

Every bound a kernel applies is an operand of the task operation, so a plan
cannot omit one. A loaded range is clamped to `value_count`. A task index is
compared with `task_count`, or with `segment_count` when there is no task
buffer. A segment id loaded from the task buffer is compared with
`segment_count`.

What is not in the dialect:

- The planning limits. `warp_max_elements` and `cta_chunk_elements` steer
  host classification and no kernel reads them, so they are arguments of
  the classifier and not part of plan IR.
- Runtime offset contents, which no compiler pass inspects.
- The fused mixed, split merge, and persistent kernels. Their lowerings emit
  them without a plan stage today;
  [ADR-0020](../adr/ADR-0020-planned-per-function-lowering.md) records the
  order in which they move.
- Packed-warp policies, queues, dependency execution, and a general task
  graph.

The CPU oracle is planned too. A task operation of `policy<sequential>`
visits every segment in order on one thread. Its function has no launch
width, keeps its signature and its callers, and takes no task buffer, and
`--swage-plan-to-scf` lowers it to loops over the memrefs. The consumers of
the region are lowered by the same patterns on both backends.

`--swage-to-plan` writes plan functions for the direct, task-id,
split-partial, and sequential schedules, one per schedule of its list, and
`--swage-plan-to-gpu` converts every plan function of a kernel to a
`gpu.module` that holds it. The plan function of a split partial kernel is
named after the kernel, `<function>__partial`, and the records its task
operation loads are laid out as `TaskRecords.h` says, which the host
classifier fills by the same fields. There is no public
`mlir_swage.dialects.swage_plan` Python module contract. The classification
buckets and task lists that the host produces are drawn in
[Task Planning](planning.md).

## Generated reference boundary

The detailed dialect and operation reference is generated from TableGen in
`include/swage/Dialect/SwagePlan/IR`.

--8<-- "docs/reference/_generated/swage-plan-dialect.inc"

--8<-- "docs/reference/_generated/swage-plan-ops.inc"

Continue with [Task Planning](planning.md) for planning admission and the
host classifier, or [Compiler Tools and Passes](compiler-tools.md) for the
registered planner and conversion passes.
