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

def SwagePlan_TaskPolicy
    : I32EnumAttr<"TaskPolicy", "task execution policy",
                  [SwagePlan_TaskPolicyWarp, SwagePlan_TaskPolicyCTA]> {
  let cppNamespace = "::mlir::swage_plan";
  let genSpecializedAttr = 0;
}

def SwagePlan_TaskPolicyAttr
    : EnumAttr<SwagePlan_Dialect, SwagePlan_TaskPolicy, "policy"> {
  let assemblyFormat = "`<` $value `>`";
}

def SwagePlan_Buffer : MemRefRankOf<[AnySignlessInteger, AnyFloat], [1]>;
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

    Every bound a kernel needs is an operand, so a plan cannot omit one:

    - a loaded range is clamped to `value_count`;
    - a task index is compared with `task_count`, or with `segment_count`
      when there are no `ids`;
    - a segment id loaded from `ids` is compared with `segment_count`, and
      one that fails the comparison stores nothing.

    `policy` says how the threads of a task combine their partial results:
    `cta` across the whole block, `warp` within one subgroup. The launch
    width is the `swage_plan.block_threads` attribute of the function.

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
    Arg<SwagePlan_Buffer, "segment values", [MemRead]>:$values,
    Arg<SwagePlan_WordBuffer, "segment offsets", [MemRead]>:$offsets,
    AnySignlessInteger:$value_count,
    AnySignlessInteger:$segment_count,
    Arg<Optional<SwagePlan_WordBuffer>, "segment id of each task",
        [MemRead]>:$ids,
    Optional<AnySignlessInteger>:$task_count,
    Arg<Optional<SwagePlan_Buffer>, "one scalar per segment",
        [MemWrite]>:$output,
    SwagePlan_TaskPolicyAttr:$policy);
  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = [{
    `policy` `` $policy
    `segments` `(` $values `,` $offsets `:` type($values) `,` type($offsets) `)`
    `value_count` `(` $value_count `:` type($value_count) `)`
    `segment_count` `(` $segment_count `:` type($segment_count) `)`
    (`ids` `(` $ids^ `:` type($ids) `)`)?
    (`task_count` `(` $task_count^ `:` type($task_count) `)`)?
    (`into` `(` $output^ `:` type($output) `)`)?
    attr-dict-with-keyword $body
  }];
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def SwagePlan_YieldOp
    : SwagePlan_Op<"yield", [Pure, Terminator, HasParent<"TasksOp">]> {
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
