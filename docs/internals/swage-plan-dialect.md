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
- `swage_plan.merge_tasks`, the task operation of the second stage, which
  reduces the partial results of one split segment per task;
- `swage_plan.fused_tasks`, the task operation of the fused mixed kernel,
  with one region for a warp task and one for a block task;
- `swage_plan.persistent_tasks`, the task operation of the experimental
  persistent queue kernel, with one region each for a block task, a partial
  task, a merge, and a warp task;
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
the program, in the order the kernel runs them: the planner has fused every
map into its consumer, and the task operation has absorbed the segment id,
the segment construction, and the scalar store.

A program that divides its reduction by the extent of its segment, which is
how a mean is written, adds one thing. The planner absorbs `swage.extent`
too: the region takes the extent as a second argument, of type `index`, and
holds the scalar epilogue of the program after its consumers, the
`arith.index_cast`, `arith.sitofp`, and `arith.divf` that run once per task.
The region of `swage_plan.partial_tasks` never takes an extent: a chunk
yields its raw reduction, and the merge of its segment runs the epilogue
once, on an extent that `swage_plan.merge_tasks` reads from the range
records through its `ranges` operand.

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
- How the persistent kernel claims its tasks. `persistent_tasks` names
  the queues, the counters, and what each kind of task computes; the
  claims, barriers, and fences are written by its conversion pattern.
- Packed-warp policies, a reusable queue, and a general task graph.

The CPU oracle is planned too. A task operation of `policy<sequential>`
visits every segment in order on one thread. Its function has no launch
width, keeps its signature and its callers, and takes no task buffer, and
`--swage-plan-to-scf` lowers it to loops over the memrefs. The consumers of
the region are lowered by the same patterns on both backends.

`--swage-to-plan` writes plan functions for the direct, task-id,
fused-mixed, split-partial, split-merge, persistent, and sequential
schedules, one per schedule of its list, and
`--swage-plan-to-gpu` converts every plan function of a kernel to a
`gpu.module` that holds it. The plan function of a split stage is named
after its kernel, `<function>__partial` or `<function>__merge`, and the
records its task operation loads are laid out as `TaskRecords.h` says,
which the host classifier fills by the same fields. The region of a merge
task is an identity reduction over scratch: the merge combines partial
results and never runs the element program. The merge kernel of a program
with an epilogue takes the range records as a fourth buffer, after its
merge records. The persistent schedule admits
the identity f32 sum only, and its plan function reads the same records and
a counter buffer whose layout is in `TaskRecords.h` as well. There is no
public
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
