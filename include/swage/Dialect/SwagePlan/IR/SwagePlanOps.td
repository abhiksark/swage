// include/swage/Dialect/SwagePlan/IR/SwagePlanOps.td
//===- SwagePlanOps.td - SwagePlan operations ------------------*- tablegen -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS
#define SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.td"

def SwagePlan_TaskPolicyWarp : I32EnumAttrCase<"Warp", 0, "warp">;
def SwagePlan_TaskPolicyCTA : I32EnumAttrCase<"CTA", 1, "cta">;
// The policy of the CPU oracle. Host classification never produces it.
def SwagePlan_TaskPolicySequential
    : I32EnumAttrCase<"Sequential", 2, "sequential">;
// The policy of rank-two values: one thread reduces one column of the
// segment of its block, and no thread combines with another.
def SwagePlan_TaskPolicyColumn : I32EnumAttrCase<"Column", 3, "column">;

def SwagePlan_TaskPolicy
    : I32EnumAttr<"TaskPolicy", "task execution policy",
                  [SwagePlan_TaskPolicyWarp, SwagePlan_TaskPolicyCTA,
                   SwagePlan_TaskPolicySequential,
                   SwagePlan_TaskPolicyColumn]> {
  let cppNamespace = "::mlir::swage_plan";
  let genSpecializedAttr = 0;
}

def SwagePlan_TaskPolicyAttr
    : EnumAttr<SwagePlan_Dialect, SwagePlan_TaskPolicy, "policy"> {
  let assemblyFormat = "`<` $value `>`";
}

def SwagePlan_Buffer : MemRefRankOf<[AnySignlessInteger, AnyFloat], [1]>;
// The values and the output of `swage_plan.tasks`, which are rows of
// `feature_count` columns for rank-two values.
def SwagePlan_RowBuffer
    : MemRefRankOf<[AnySignlessInteger, AnyFloat], [1, 2]>;
def SwagePlan_WordBuffer : MemRefRankOf<[AnySignlessInteger], [1]>;

def SwagePlan_TasksOp
    : SwagePlan_Op<"tasks", [AttrSizedOperandSegments, SingleBlock,
                             RecursiveMemoryEffects]> {
  let summary = "Run a segment program once per task, on one segment each";
  let description = [{
    The body of a kernel that reduces one segment per task. Task `t` is the
    block of threads with index `t`. Without `ids` there is one task per
    segment and task `t` binds segment `t`. With `ids` there are
    `task_count` tasks and task `t` binds segment `ids[t]`.

    The region runs once per task. Its argument is the bound segment,
    `values[offsets[s] : offsets[s + 1]]` for the segment `s` of the task.
    It holds `swage.reduce` and `swage.map_store` operations that read that
    segment, in the order they run, and ends in `swage_plan.yield`. With
    `into`, the yielded scalar of segment `s` is stored at `output[s]`.

    The region may take a second argument of type `index`, the extent of the
    bound segment, `offsets[s + 1] - offsets[s]`. With it, the region may
    hold a scalar epilogue after its consumers: `arith.index_cast`,
    `arith.sitofp`, and `arith.divf` operations, which run once per task.
    The planner writes them for a program that divides its reduction by the
    extent of its segment.

    Every bound a kernel needs is an operand, so a plan cannot omit one:

    - a loaded range is clamped to `value_count`;
    - a task index is compared with `task_count`, or with `segment_count`
      when there are no `ids`;
    - a segment id loaded from `ids` is compared with `segment_count`, and
      one that fails the comparison stores nothing.

    `policy` says how the threads of a task combine their partial results:
    `cta` across the whole block, `warp` within one subgroup. The launch
    width is the `swage_plan.block_threads` attribute of the function.

    `policy<sequential>` is the CPU oracle: one thread visits the segments
    in order and runs the region on each. It takes no task buffer, its
    function has no launch width and keeps its callers, and it lowers to
    loops over the memrefs instead of to a kernel.

    With `feature_count`, the values are rank-two: `value_count` rows of
    `feature_count` columns. The region then runs once per segment and
    column, and its argument is one column of the rows of the segment,
    `values[offsets[s] : offsets[s + 1], c]`. With `into`, the yielded
    scalar is stored at `output[s, c]`, and `output` is rank-two as well.
    `policy<column>` is the kernel of such a program: task `t` is segment
    `t`, and thread `i` of a block runs the region for the columns `i`,
    `i + block_threads`, and so on, one after the other. Each thread reduces
    its columns alone and stores its own results, so no thread combines
    with another and a thread holds one scalar per reduction, whatever the
    number of columns. The policy takes no `ids`. `policy<cta>` over
    rank-two values is the row-stripe tile, with or without `ids`: a block
    runs one task and one group of adjacent columns at a time, its threads
    split into row stripes of each column of the group, and the stripes of
    one column combine across the block, so a thread still holds one scalar
    per reduction. `policy<sequential>` visits the columns of each segment
    in order. `policy<warp>` takes rank-one values only.

    Example:

    ```mlir
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
    ```
  }];
  let arguments = (ins
    Arg<SwagePlan_RowBuffer, "segment values", [MemRead]>:$values,
    Arg<SwagePlan_WordBuffer, "segment offsets", [MemRead]>:$offsets,
    AnySignlessInteger:$value_count,
    AnySignlessInteger:$segment_count,
    Arg<Optional<SwagePlan_WordBuffer>, "segment id of each task",
        [MemRead]>:$ids,
    Optional<AnySignlessInteger>:$task_count,
    Arg<Optional<SwagePlan_RowBuffer>, "one scalar per segment",
        [MemWrite]>:$output,
    Optional<AnySignlessInteger>:$feature_count,
    SwagePlan_TaskPolicyAttr:$policy);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `policy` `` $policy
    `segments` `(` $values `,` $offsets `:` type($values) `,` type($offsets) `)`
    `value_count` `(` $value_count `:` type($value_count) `)`
    `segment_count` `(` $segment_count `:` type($segment_count) `)`
    (`feature_count` `(` $feature_count^ `:` type($feature_count) `)`)?
    (`ids` `(` $ids^ `:` type($ids) `)`)?
    (`task_count` `(` $task_count^ `:` type($task_count) `)`)?
    (`into` `(` $output^ `:` type($output) `)`)?
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def SwagePlan_PartialTasksOp
    : SwagePlan_Op<"partial_tasks", [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Reduce one chunk of a long segment per task into scratch";
  let description = [{
    The body of the first stage of a split reduction. Task `t` is the block
    of threads with index `t`. It binds the half-open range of values that
    record `t` of `ranges` names, runs the region on it, and stores the
    yielded scalar at `scratch[t]`. A record is a `[begin, end]` pair of
    words, written by host classification.

    The region holds the `swage.reduce` of the program, which applies the
    element program to every element of the chunk, and ends in
    `swage_plan.yield` with its result. The threads of a task combine their
    partial results across the whole block. The region takes no extent and
    holds no scalar epilogue: a chunk yields the raw reduction, and the
    merge of its segment runs the epilogue once.

    Every bound the kernel needs is an operand: a loaded range is clamped to
    `value_count`, and a task index is compared with `partial_count`.

    Example:

    ```mlir
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %sum = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    ```
  }];
  let arguments = (ins
    Arg<SwagePlan_Buffer, "chunk values", [MemRead]>:$values,
    AnySignlessInteger:$value_count,
    Arg<SwagePlan_WordBuffer, "one range record per task", [MemRead]>:$ranges,
    AnySignlessInteger:$partial_count,
    Arg<SwagePlan_Buffer, "one scalar per task", [MemWrite]>:$scratch);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `values` `(` $values `:` type($values) `)`
    `value_count` `(` $value_count `:` type($value_count) `)`
    `ranges` `(` $ranges `:` type($ranges) `)`
    `partial_count` `(` $partial_count `:` type($partial_count) `)`
    `into` `(` $scratch `:` type($scratch) `)`
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def SwagePlan_MergeTasksOp
    : SwagePlan_Op<"merge_tasks", [SingleBlock, RecursiveMemoryEffects]> {
  let summary = "Reduce the partial results of one split segment per task";
  let description = [{
    The body of the second stage of a split reduction. Task `t` is the block
    of threads with index `t`. It reads record `t` of `merges`, a
    `[segment, partial_begin, partial_end]` triple of words written by host
    classification, binds the half-open range of scratch slots the record
    names, runs the region on it, and stores the yielded scalar at
    `output[segment]`.

    The region holds a `swage.reduce` over the bound range of scratch and
    ends in `swage_plan.yield` with its result. Scratch holds completed
    partial reductions, so the planner writes an identity reduction of the
    kind of the program: the merge combines partial results and never runs
    the element program. The threads of a task combine their partial
    results across the whole block.

    With `ranges`, the region takes a second argument of type `index`: the
    extent of the split segment, which is not the extent of the bound range
    of scratch. `ranges` holds the range records `swage_plan.partial_tasks`
    reads. The chunks of one segment are consecutive records, so the extent
    is the end of record `partial_end - 1` minus the begin of record
    `partial_begin`, and zero for an empty range of partials. The region may
    then hold a scalar epilogue after its reduction, as the region of
    `swage_plan.tasks` may: the partial tasks yield the raw reduction, and
    the merge divides once.

    Every bound the kernel needs is an operand: a loaded range is clamped to
    `partial_count`, a task index is compared with `merge_count`, and a
    loaded segment is compared with `segment_count`, and one that fails the
    comparison stores nothing. The two range records of an extent are
    addressed through the clamped range of partials.

    Example:

    ```mlir
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      %sum = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %sum : f32
    }
    ```
  }];
  let arguments = (ins
    Arg<SwagePlan_Buffer, "partial results", [MemRead]>:$scratch,
    AnySignlessInteger:$partial_count,
    Arg<SwagePlan_WordBuffer, "one merge record per task", [MemRead]>:$merges,
    AnySignlessInteger:$merge_count,
    AnySignlessInteger:$segment_count,
    Arg<SwagePlan_Buffer, "one scalar per split segment", [MemWrite]>:$output,
    Arg<Optional<SwagePlan_WordBuffer>, "one range record per partial task",
        [MemRead]>:$ranges);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `scratch` `(` $scratch `:` type($scratch) `)`
    `partial_count` `(` $partial_count `:` type($partial_count) `)`
    `merges` `(` $merges `:` type($merges) `)`
    `merge_count` `(` $merge_count `:` type($merge_count) `)`
    `segment_count` `(` $segment_count `:` type($segment_count) `)`
    (`ranges` `(` $ranges^ `:` type($ranges) `)`)?
    `into` `(` $output `:` type($output) `)`
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def SwagePlan_FusedTasksOp
    : SwagePlan_Op<"fused_tasks", [RecursiveMemoryEffects]> {
  let summary = "Run warp tasks and block tasks of one task buffer in one "
                "launch";
  let description = [{
    The body of the fused mixed kernel. The task buffer `ids` holds
    `warp_task_count` segment ids of short segments followed by
    `cta_task_count` segment ids of longer ones. The first blocks of the
    launch each run one warp task per subgroup, and every block after them
    runs one block task.

    The operation has two regions, each a task region over the bound
    segment of its task: `warp` for a warp task, whose threads combine
    within one subgroup, and `cta` for a block task, whose threads combine
    across the block. Both hold the `swage.reduce` of the program and end in
    `swage_plan.yield` with its result, which is stored at `output[s]` for
    the segment `s` of the task. Each may take the extent of its segment as
    a second argument and hold a scalar epilogue, as the region of
    `swage_plan.tasks` may.

    Every bound the kernel needs is an operand: a loaded range is clamped to
    `value_count`, a warp task index is compared with `warp_task_count` and
    a block task index with `cta_task_count`, and a segment id loaded from
    `ids` is compared with `segment_count`.

    Example:

    ```mlir
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) warp_task_count(%warp_task_count : i32)
        cta_task_count(%cta_task_count : i32)
        into(%output : memref<?xf32>) warp {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    } cta {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    ```
  }];
  let arguments = (ins
    Arg<SwagePlan_Buffer, "segment values", [MemRead]>:$values,
    Arg<SwagePlan_WordBuffer, "segment offsets", [MemRead]>:$offsets,
    AnySignlessInteger:$value_count,
    AnySignlessInteger:$segment_count,
    Arg<SwagePlan_WordBuffer, "segment id of each task", [MemRead]>:$ids,
    AnySignlessInteger:$warp_task_count,
    AnySignlessInteger:$cta_task_count,
    Arg<SwagePlan_Buffer, "one scalar per segment", [MemWrite]>:$output);
  let regions = (region SizedRegion<1>:$warp, SizedRegion<1>:$cta);
  let assemblyFormat = [{
    `segments` `(` $values `,` $offsets `:` type($values) `,` type($offsets) `)`
    `value_count` `(` $value_count `:` type($value_count) `)`
    `segment_count` `(` $segment_count `:` type($segment_count) `)`
    `ids` `(` $ids `:` type($ids) `)`
    `warp_task_count` `(` $warp_task_count `:` type($warp_task_count) `)`
    `cta_task_count` `(` $cta_task_count `:` type($cta_task_count) `)`
    `into` `(` $output `:` type($output) `)`
    attr-dict `warp` $warp `cta` $cta
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def SwagePlan_PersistentTasksOp
    : SwagePlan_Op<"persistent_tasks", [RecursiveMemoryEffects]> {
  let summary = "Drain the block, partial, and warp task queues from resident "
                "blocks";
  let description = [{
    The body of the persistent queue kernel. A launch holds a fixed number of
    resident blocks, and each block claims tasks until every queue is empty:
    first the block tasks, then the partial tasks with the merges they
    complete, then the warp tasks. A claim adds to a counter in device
    memory, so no task runs twice and no block waits for another.

    The operation has four regions, each a task region over the range its
    task binds, which hold a `swage.reduce` and end in `swage_plan.yield`
    with its result:

    - `cta` runs once per block task. Task `t` binds segment `cta_ids[t]`,
      its threads combine across the block, and the scalar is stored at
      `output[s]` for the segment `s` of the task.
    - `partial` runs once per partial task. Task `t` binds the half-open
      range of values that record `t` of `ranges` names and stores the
      scalar at `scratch[t]`. It then counts one completion for the merge
      `merge_ids[t]`.
    - `merge` runs in the block that completed the last partial task of a
      merge. Merge `m` binds the half-open range of scratch slots that
      record `m` of `merges` names and stores the scalar at
      `output[segment]` for the segment of that record.
    - `warp` runs once per warp task, on the lanes of one subgroup. Task `t`
      binds segment `warp_ids[t]` and stores the scalar at `output[s]`.

    `counters` holds one claim counter per queue followed by one completion
    counter per merge, all zero when the launch starts. The records
    of `ranges` and `merges` are the ones `swage_plan.partial_tasks` and
    `swage_plan.merge_tasks` read.

    Every bound the kernel needs is an operand: a loaded range of values is
    clamped to `value_count` and a loaded range of scratch to
    `partial_count`; a claimed task index is compared with the count of its
    queue; a loaded segment id is compared with `segment_count` and a
    loaded merge id with `merge_count`, and one that fails the comparison
    stores nothing and updates no counter.

    Example:

    ```mlir
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %sum = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %sum = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %sum : f32
    } warp {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    ```
  }];
  let arguments = (ins
    Arg<SwagePlan_Buffer, "segment values", [MemRead]>:$values,
    Arg<SwagePlan_WordBuffer, "segment offsets", [MemRead]>:$offsets,
    AnySignlessInteger:$value_count,
    AnySignlessInteger:$segment_count,
    Arg<SwagePlan_WordBuffer, "segment id of each warp task",
        [MemRead]>:$warp_ids,
    AnySignlessInteger:$warp_task_count,
    Arg<SwagePlan_WordBuffer, "segment id of each block task",
        [MemRead]>:$cta_ids,
    AnySignlessInteger:$cta_task_count,
    Arg<SwagePlan_WordBuffer, "one range record per partial task",
        [MemRead]>:$ranges,
    Arg<SwagePlan_WordBuffer, "merge id of each partial task",
        [MemRead]>:$merge_ids,
    AnySignlessInteger:$partial_count,
    Arg<SwagePlan_WordBuffer, "one merge record per merge", [MemRead]>:$merges,
    AnySignlessInteger:$merge_count,
    Arg<SwagePlan_Buffer, "one scalar per partial task",
        [MemRead, MemWrite]>:$scratch,
    Arg<SwagePlan_WordBuffer, "claim and completion counters",
        [MemRead, MemWrite]>:$counters,
    Arg<SwagePlan_Buffer, "one scalar per segment", [MemWrite]>:$output);
  let regions = (region SizedRegion<1>:$cta, SizedRegion<1>:$partial,
                        SizedRegion<1>:$merge, SizedRegion<1>:$warp);
  let assemblyFormat = [{
    `segments` `(` $values `,` $offsets `:` type($values) `,` type($offsets) `)`
    `value_count` `(` $value_count `:` type($value_count) `)`
    `segment_count` `(` $segment_count `:` type($segment_count) `)`
    `warp_ids` `(` $warp_ids `:` type($warp_ids) `)`
    `warp_task_count` `(` $warp_task_count `:` type($warp_task_count) `)`
    `cta_ids` `(` $cta_ids `:` type($cta_ids) `)`
    `cta_task_count` `(` $cta_task_count `:` type($cta_task_count) `)`
    `ranges` `(` $ranges `:` type($ranges) `)`
    `merge_ids` `(` $merge_ids `:` type($merge_ids) `)`
    `partial_count` `(` $partial_count `:` type($partial_count) `)`
    `merges` `(` $merges `:` type($merges) `)`
    `merge_count` `(` $merge_count `:` type($merge_count) `)`
    `scratch` `(` $scratch `:` type($scratch) `)`
    `counters` `(` $counters `:` type($counters) `)`
    `into` `(` $output `:` type($output) `)`
    attr-dict `cta` $cta `partial` $partial `merge` $merge `warp` $warp
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def SwagePlan_YieldOp
    : SwagePlan_Op<"yield", [Pure, Terminator,
                             ParentOneOf<["TasksOp", "PartialTasksOp",
                                          "MergeTasksOp", "FusedTasksOp",
                                          "PersistentTasksOp"]>]> {
  let summary = "End a task region, with the scalar result of the segment";
  let description = [{
    Terminates the region of a task operation. With an operand, the value is
    the scalar result of the bound segment, which the task operation stores
    at its `into` buffer. Without one, the region writes its results itself,
    through `swage.map_store`.
  }];
  let arguments =
      (ins Optional<AnyTypeOf<[AnySignlessInteger, AnyFloat]>>:$value);
  let assemblyFormat = "($value^ `:` type($value))? attr-dict";
}

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS
