// lib/Dialect/SwagePlan/IR/SwagePlanDialect.cpp
//===- SwagePlanDialect.cpp - SwagePlan dialect ----------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::swage_plan;

#include "swage/Dialect/SwagePlan/IR/SwagePlanOpsDialect.cpp.inc"

#include "swage/Dialect/SwagePlan/IR/SwagePlanEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanOpsTypes.cpp.inc"

#define GET_OP_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.cpp.inc"

void SwagePlanDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "swage/Dialect/SwagePlan/IR/SwagePlanAttributes.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "swage/Dialect/SwagePlan/IR/SwagePlanOpsTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.cpp.inc"
      >();
}

LogicalResult ClassifyOp::verify() {
  ArrayAttr policies = getPolicies();
  if (policies.size() != 2 ||
      cast<TaskPolicyAttr>(policies[0]).getValue() != TaskPolicy::Warp ||
      cast<TaskPolicyAttr>(policies[1]).getValue() != TaskPolicy::CTA)
    return emitOpError("policies must be ordered warp then CTA");
  if (!SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(getOperation(),
                                                          getKernelAttr()))
    return emitOpError("kernel must reference a func.func semantic kernel");
  return success();
}
