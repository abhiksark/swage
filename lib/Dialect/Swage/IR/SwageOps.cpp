//===- SwageOps.cpp - Swage dialect operations ------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/Swage/IR/SwageOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_OP_CLASSES
#include "swage/Dialect/Swage/IR/SwageOps.cpp.inc"

using namespace mlir;
using namespace mlir::swage;

/// Shared region contract for map/reduce/map_store (ADR-0008): the entry
/// block takes the element type followed by the capture types, and the
/// terminating yield must produce `resultElementType`.
static LogicalResult verifySwageRegion(Operation *op, Region &body,
                                       Type elementType, ValueRange captures,
                                       Type resultElementType) {
  Block &block = body.front();
  unsigned expectedArgs = 1 + captures.size();
  if (block.getNumArguments() != expectedArgs)
    return op->emitOpError() << "region expects " << expectedArgs
                             << " arguments (element plus captures), got "
                             << block.getNumArguments();
  if (block.getArgument(0).getType() != elementType)
    return op->emitOpError()
           << "region argument #0 type " << block.getArgument(0).getType()
           << " does not match segment element type " << elementType;
  for (auto [index, capture] : llvm::enumerate(captures)) {
    Type argType = block.getArgument(index + 1).getType();
    if (argType != capture.getType())
      return op->emitOpError()
             << "region argument #" << (index + 1) << " type " << argType
             << " does not match capture type " << capture.getType();
  }
  auto yield = llvm::dyn_cast<YieldOp>(block.getTerminator());
  if (!yield)
    return op->emitOpError() << "region must terminate with swage.yield";
  if (yield.getValue().getType() != resultElementType)
    return op->emitOpError()
           << "yield type " << yield.getValue().getType()
           << " does not match the expected element type " << resultElementType;
  return success();
}

LogicalResult MapOp::verifyRegions() {
  auto segmentType = llvm::cast<SegmentType>(getSegment().getType());
  auto resultType = llvm::cast<SegmentType>(getResult().getType());
  return verifySwageRegion(getOperation(), getBody(),
                           segmentType.getElementType(), getCaptures(),
                           resultType.getElementType());
}

LogicalResult ReduceOp::verifyRegions() {
  auto segmentType = llvm::cast<SegmentType>(getSegment().getType());
  return verifySwageRegion(getOperation(), getBody(),
                           segmentType.getElementType(), getCaptures(),
                           getResult().getType());
}

#include "swage/Dialect/Swage/IR/SwageEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "swage/Dialect/Swage/IR/SwageAttributes.cpp.inc"

void SwageDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "swage/Dialect/Swage/IR/SwageAttributes.cpp.inc"
      >();
}

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
