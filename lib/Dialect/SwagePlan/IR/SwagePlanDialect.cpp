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
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::swage_plan;

#include "swage/Dialect/SwagePlan/IR/SwagePlanOpsDialect.cpp.inc"

#include "swage/Dialect/SwagePlan/IR/SwagePlanEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanAttributes.cpp.inc"

#define GET_OP_CLASSES
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.cpp.inc"

void SwagePlanDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "swage/Dialect/SwagePlan/IR/SwagePlanAttributes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.cpp.inc"
      >();
}

std::optional<TaskPolicy> mlir::swage_plan::policyOfRegion(Region *region) {
  if (auto tasks = dyn_cast_or_null<TasksOp>(region->getParentOp()))
    return tasks.getPolicy();
  return std::nullopt;
}

/// Whether a kernel can take `type` as a parameter: a rank-one buffer it can
/// address through a pointer, or a count.
static bool isKernelParameterType(Type type) {
  if (type.isSignlessInteger())
    return true;
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getLayout().isIdentity() && !memref.getMemorySpace() &&
         (memref.getElementType().isSignlessInteger() ||
          isa<FloatType>(memref.getElementType()));
}

LogicalResult
SwagePlanDialect::verifyOperationAttribute(Operation *op,
                                           NamedAttribute attribute) {
  StringRef name = getBlockThreadsAttrName();
  if (attribute.getName() != name)
    return op->emitError()
           << "'" << attribute.getName().strref()
           << "' is not an operation attribute of the swage_plan dialect; the "
              "dialect defines "
           << name;
  auto function = dyn_cast<func::FuncOp>(op);
  if (!function)
    return op->emitError() << name << " belongs on a func.func, got '"
                           << op->getName() << "'";
  auto threads = dyn_cast<IntegerAttr>(attribute.getValue());
  if (!threads || !threads.getType().isSignlessInteger(32) ||
      !threads.getValue().isStrictlyPositive())
    return op->emitError() << name << " must be a positive i32, got "
                           << attribute.getValue();

  // The attribute makes the function a plan function: its signature is the
  // parameter list of a kernel, and its body is the task operation of that
  // kernel.
  FunctionType type = function.getFunctionType();
  if (type.getNumResults() != 0)
    return op->emitError() << "a plan function has no result, got " << type;
  for (auto [index, input] : llvm::enumerate(type.getInputs()))
    if (!isKernelParameterType(input))
      return op->emitError()
             << "plan function argument #" << index
             << " must be a signless integer or a rank-one memref of signless "
                "integers or floats with a dynamic size, the identity layout, "
                "and the default memory space, got "
             << input;
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return op->emitError("a plan function has a body of one block");
  Block &body = function.getBody().front();
  if (llvm::range_size(body) != 2 || !isa<TasksOp>(body.front()) ||
      !isa<func::ReturnOp>(body.back()))
    return op->emitError()
           << "a plan function holds one task operation followed by a return, "
              "found "
           << llvm::range_size(body) << " operations";
  if (cast<TasksOp>(body.front()).getPolicy() == TaskPolicy::Sequential)
    return op->emitError()
           << name << " gives the launch width of a kernel, and "
           << "policy<sequential> runs on one thread without a kernel; a "
              "function has one or the other";
  return success();
}

LogicalResult TasksOp::verify() {
  Type word = cast<MemRefType>(getOffsets().getType()).getElementType();
  auto requireWord = [&](StringRef name, Type type) -> LogicalResult {
    if (type == word)
      return success();
    return emitOpError() << name << " must have the element type of the "
                         << "offsets, " << word << ", got " << type;
  };
  if (failed(requireWord("value_count", getValueCount().getType())) ||
      failed(requireWord("segment_count", getSegmentCount().getType())))
    return failure();
  if (static_cast<bool>(getIds()) != static_cast<bool>(getTaskCount()))
    return emitOpError("ids and task_count are given together: the ids name "
                       "the segment of each task, and task_count bounds the "
                       "task index");
  if (getIds() && getPolicy() == TaskPolicy::Sequential)
    return emitOpError("policy<sequential> visits every segment in order and "
                       "takes no ids");
  if (getIds() &&
      (failed(requireWord("task_count", getTaskCount().getType())) ||
       failed(
           requireWord("an element of ids",
                       cast<MemRefType>(getIds().getType()).getElementType()))))
    return failure();
  return success();
}

LogicalResult TasksOp::verifyRegions() {
  Block &body = getBody().front();
  Type element = cast<MemRefType>(getValues().getType()).getElementType();
  auto segment =
      body.getNumArguments() == 1
          ? dyn_cast<swage::SegmentType>(body.getArgument(0).getType())
          : swage::SegmentType();
  if (!segment)
    return emitOpError("region takes the bound segment as its one argument, "
                       "of type !swage.segment<T>");
  if (segment.getElementType() != element)
    return emitOpError() << "region binds a segment of " << element
                         << ", the element type of the values, got "
                         << body.getArgument(0).getType();
  if (!body.mightHaveTerminator() || !isa<YieldOp>(body.getTerminator()))
    return emitOpError("region must end in swage_plan.yield");
  for (Operation &operation : body.without_terminator()) {
    if (!isa<swage::ReduceOp, swage::MapStoreOp>(operation))
      return operation.emitOpError(
          "is not allowed in a task region; the region holds swage.reduce "
          "and swage.map_store operations and ends in swage_plan.yield");
    if (operation.getOperand(0) != body.getArgument(0))
      return operation.emitOpError(
          "must read the bound segment, the argument of the task region");
  }
  auto yield = cast<YieldOp>(body.getTerminator());
  if (static_cast<bool>(yield.getValue()) != static_cast<bool>(getOutput()))
    return emitOpError("into and a yielded scalar are given together: the "
                       "scalar of each segment is stored in the into buffer");
  if (getOutput()) {
    Type slot = cast<MemRefType>(getOutput().getType()).getElementType();
    if (yield.getValue().getType() != slot)
      return emitOpError() << "region must yield " << slot
                           << ", the element type of the into buffer, got "
                           << yield.getValue().getType();
  }
  return success();
}
