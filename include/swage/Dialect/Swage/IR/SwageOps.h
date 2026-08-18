//===- SwageOps.h - Swage dialect operations --------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGE_IR_SWAGEOPS_H
#define SWAGE_DIALECT_SWAGE_IR_SWAGEOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "swage/Dialect/Swage/IR/SwageTypes.h"

#include "swage/Dialect/Swage/IR/SwageEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "swage/Dialect/Swage/IR/SwageAttributes.h.inc"

#define GET_OP_CLASSES
#include "swage/Dialect/Swage/IR/SwageOps.h.inc"

#endif // SWAGE_DIALECT_SWAGE_IR_SWAGEOPS_H
