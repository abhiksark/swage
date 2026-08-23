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
  let summary = "Runtime task classification for admitted Swage kernels";
  let cppNamespace = "::mlir::swage_plan";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}

class SwagePlan_Op<string mnemonic, list<Trait> traits = []>
    : Op<SwagePlan_Dialect, mnemonic, traits>;

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANDIALECT
