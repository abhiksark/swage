//===- SwageDialect.cpp - Swage dialect -------------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/Swage/IR/SwageTypes.h"

using namespace mlir;
using namespace mlir::swage;

#include "swage/Dialect/Swage/IR/SwageOpsDialect.cpp.inc"

void SwageDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "swage/Dialect/Swage/IR/SwageOps.cpp.inc"
      >();
  registerTypes();
}
