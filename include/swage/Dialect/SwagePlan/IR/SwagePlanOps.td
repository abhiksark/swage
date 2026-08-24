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

def SwagePlan_TaskPolicyArrayAttr
    : TypedArrayAttrBase<SwagePlan_TaskPolicyAttr,
                         "array of task execution policies">;

def SwagePlan_TaskRangeType
    : TypeDef<SwagePlan_Dialect, "TaskRange"> {
  let mnemonic = "task_range";
  let summary = "SwagePlan task range";
}

def SwagePlan_ClassifyOp : SwagePlan_Op<"classify"> {
  let summary = "Classify runtime segment ranges for one semantic kernel";
  let arguments = (ins
    Arg<MemRefRankOf<[I32], [1]>, "offset metadata", [MemRead]>:$offsets,
    I32:$value_count,
    I32:$segment_count,
    FlatSymbolRefAttr:$kernel,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$warp_max_elements,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$cta_chunk_elements,
    SwagePlan_TaskPolicyArrayAttr:$policies);
  let results = (outs SwagePlan_TaskRangeType:$result);
  let assemblyFormat = [{
    $offsets `,` $value_count `,` $segment_count attr-dict
    `:` type($offsets) `,` type($value_count) `,` type($segment_count)
    `->` qualified(type($result))
  }];
  let hasVerifier = 1;
}

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS
