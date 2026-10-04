// include/swage/Dialect/SwagePlan/IR/SwagePlanOps.h
//===- SwagePlanOps.h - SwagePlan operations -------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS_H
#define SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanAttributes.h.inc"

#define GET_OP_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h.inc"

#include <optional>

namespace mlir::swage_plan {

/// How the threads that run `region` combine their partial results: the
/// policy of the task operation the region belongs to. Null when the region
/// is not the region of a task operation.
std::optional<TaskPolicy> policyOfRegion(Region *region);

} // namespace mlir::swage_plan

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_SWAGEPLANOPS_H
