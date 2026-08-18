//===- SwageTypes.cpp - Swage dialect types ---------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/Swage/IR/SwageTypes.h"

#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::swage;

#define GET_TYPEDEF_CLASSES
#include "swage/Dialect/Swage/IR/SwageOpsTypes.cpp.inc"

LogicalResult
SegmentType::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                    Type elementType) {
  if (!elementType.isIntOrFloat())
    return emitError() << "segment element type must be an integer or float "
                          "type, got "
                       << elementType;
  return success();
}

void SwageDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "swage/Dialect/Swage/IR/SwageOpsTypes.cpp.inc"
      >();
}
