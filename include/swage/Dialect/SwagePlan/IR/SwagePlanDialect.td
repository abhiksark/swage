// include/swage/Dialect/SwagePlan/IR/SwagePlanDialect.td
//===- SwagePlanDialect.td - SwagePlan dialect definition ------*- tablegen -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANDIALECT
#define SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANDIALECT

include "mlir/IR/OpBase.td"

def SwagePlan_Dialect : Dialect {
  let name = "swage_plan";
  let summary = "Planned kernels: a kernel signature and its task operation";
  let description = [{
    The `swage_plan` dialect holds what a kernel lowering consumes. A plan
    function is a `func.func` with the attribute `swage_plan.block_threads`,
    which gives the launch width of the kernel in threads:

    ```mlir
    func.func @segmented_sum(
        %values: memref<?xf32>, %offsets: memref<?xi32>,
        %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
        attributes {swage_plan.block_threads = 128 : i32} {
      swage_plan.tasks policy<cta> ... { ... }
      return
    }
    ```

    - The signature of a plan function is the parameter list of its kernel,
      in order: every buffer is a rank-one memref with a dynamic size, the
      identity layout, and the default memory space, and every count is a
      signless integer. There is no result.
    - The body is one task operation followed by a return. The task
      operation takes every buffer and every bound of the kernel as an
      operand and holds the reductions and stores of one bound segment.

    The dialect verifies both. A planning parameter that no kernel reads,
    such as a limit that steers host classification, is not part of plan IR.
  }];
  let cppNamespace = "::mlir::swage_plan";
  let useDefaultAttributePrinterParser = 1;
  let hasOperationAttrVerify = 1;
  let extraClassDeclaration = [{
    /// The name of the function attribute that makes a function a plan
    /// function and gives the launch width of its kernel in threads.
    static ::llvm::StringRef getBlockThreadsAttrName() {
      return "swage_plan.block_threads";
    }
  }];
}

class SwagePlan_Op<string mnemonic, list<Trait> traits = []>
    : Op<SwagePlan_Dialect, mnemonic, traits>;

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANDIALECT
