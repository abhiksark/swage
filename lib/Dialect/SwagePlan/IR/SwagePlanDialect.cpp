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

namespace {

bool isRankOneMemRef(Type type, Type elementType) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType() == elementType &&
         memref.getLayout().isIdentity() && !memref.getMemorySpace();
}

bool hasCanonicalSemanticABI(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 3 || type.getNumResults() != 0)
    return false;

  MLIRContext *context = function.getContext();
  unsigned f32Buffers = 0;
  unsigned i32Buffers = 0;
  for (Type input : type.getInputs()) {
    if (isRankOneMemRef(input, Float32Type::get(context)))
      ++f32Buffers;
    else if (isRankOneMemRef(input, IntegerType::get(context, 32)))
      ++i32Buffers;
    else
      return false;
  }
  return f32Buffers == 2 && i32Buffers == 1;
}

} // namespace

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
  if (getWarpMaxElements() == 0)
    return emitOpError("warp_max_elements must be positive");
  if (getCtaChunkElements() == 0)
    return emitOpError("cta_chunk_elements must be positive");
  if (getWarpMaxElements() > getCtaChunkElements())
    return emitOpError("warp_max_elements must not exceed cta_chunk_elements");
  ArrayAttr policies = getPolicies();
  if (policies.size() != 2 ||
      cast<TaskPolicyAttr>(policies[0]).getValue() != TaskPolicy::Warp ||
      cast<TaskPolicyAttr>(policies[1]).getValue() != TaskPolicy::CTA)
    return emitOpError("policies must be ordered warp then CTA");
  Operation *symbol =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), getKernelAttr());
  auto kernel = dyn_cast_or_null<func::FuncOp>(symbol);
  if (!kernel)
    return emitOpError("kernel must reference a func.func semantic kernel");
  if (kernel == getOperation()->getParentOfType<func::FuncOp>())
    return emitOpError("kernel must not reference its containing function");
  if (!hasCanonicalSemanticABI(kernel))
    return emitOpError(
        "kernel must use the canonical three-buffer semantic ABI");
  return success();
}
