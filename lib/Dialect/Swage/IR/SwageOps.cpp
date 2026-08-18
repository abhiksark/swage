//===- SwageOps.cpp - Swage dialect operations ------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/Swage/IR/SwageOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageTypes.h"

#define GET_OP_CLASSES
#include "swage/Dialect/Swage/IR/SwageOps.cpp.inc"

using namespace mlir;
using namespace mlir::swage;

LogicalResult MakeSegmentOp::verify() {
  auto valuesType = llvm::cast<MemRefType>(getValues().getType());
  auto segmentType = llvm::cast<SegmentType>(getResult().getType());
  if (valuesType.getElementType() != segmentType.getElementType())
    return emitOpError() << "values element type "
                         << valuesType.getElementType()
                         << " does not match segment element type "
                         << segmentType.getElementType();
  return success();
}
