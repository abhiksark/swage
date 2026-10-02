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
  Operation *task = region->getParentOp();
  if (auto tasks = dyn_cast_or_null<TasksOp>(task))
    return tasks.getPolicy();
  // The threads of a partial task and of a merge task combine across the
  // whole block.
  if (isa_and_nonnull<PartialTasksOp, MergeTasksOp>(task))
    return TaskPolicy::CTA;
  // The first region of a fused task operation is its warp region.
  if (isa_and_nonnull<FusedTasksOp>(task))
    return region->getRegionNumber() == 0 ? TaskPolicy::Warp : TaskPolicy::CTA;
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
  if (llvm::range_size(body) != 2 ||
      !isa<TasksOp, PartialTasksOp, MergeTasksOp, FusedTasksOp>(body.front()) ||
      !isa<func::ReturnOp>(body.back()))
    return op->emitError()
           << "a plan function holds one task operation followed by a return, "
              "found "
           << llvm::range_size(body) << " operations";
  auto tasks = dyn_cast<TasksOp>(body.front());
  if (tasks && tasks.getPolicy() == TaskPolicy::Sequential)
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

/// Verify what every task region shares: one argument, the bound segment,
/// with the element type of the buffer it is bound from; consumers that read
/// that argument; and a `swage_plan.yield` at the end. `buffer` names the
/// buffer in a diagnostic, and `allowStores` admits `swage.map_store`.
static LogicalResult verifyTaskRegion(Operation *task, Region &region,
                                      Type element, StringRef buffer,
                                      bool allowStores) {
  Block &body = region.front();
  auto segment =
      body.getNumArguments() == 1
          ? dyn_cast<swage::SegmentType>(body.getArgument(0).getType())
          : swage::SegmentType();
  if (!segment)
    return task->emitOpError(
        "region takes the bound segment as its one argument, of type "
        "!swage.segment<T>");
  if (segment.getElementType() != element)
    return task->emitOpError() << "region binds a segment of " << element
                               << ", the element type of the " << buffer
                               << ", got " << body.getArgument(0).getType();
  if (!body.mightHaveTerminator() || !isa<YieldOp>(body.getTerminator()))
    return task->emitOpError("region must end in swage_plan.yield");
  for (Operation &operation : body.without_terminator()) {
    bool admitted = allowStores
                        ? isa<swage::ReduceOp, swage::MapStoreOp>(operation)
                        : isa<swage::ReduceOp>(operation);
    if (!admitted)
      return operation.emitOpError()
             << "is not allowed in the region of '" << task->getName()
             << "'; the region holds "
             << (allowStores ? "swage.reduce and swage.map_store operations"
                             : "swage.reduce operations")
             << " and ends in swage_plan.yield";
    if (operation.getOperand(0) != body.getArgument(0))
      return operation.emitOpError(
          "must read the bound segment, the argument of the task region");
  }
  return success();
}

/// Require that the region of `task` yields a scalar of the element type of
/// `into`, the buffer that receives it.
static LogicalResult verifyYieldedScalar(Operation *task, Region &region,
                                         Value into) {
  auto yield = cast<YieldOp>(region.front().getTerminator());
  Type slot = cast<MemRefType>(into.getType()).getElementType();
  if (!yield.getValue() || yield.getValue().getType() != slot) {
    InFlightDiagnostic diagnostic =
        task->emitOpError() << "region must yield " << slot
                            << ", the element type of the into buffer, got ";
    if (yield.getValue())
      diagnostic << yield.getValue().getType();
    else
      diagnostic << "no value";
    return diagnostic;
  }
  return success();
}

LogicalResult TasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getValues().getType()).getElementType();
  if (failed(verifyTaskRegion(getOperation(), getBody(), element, "values",
                              /*allowStores=*/true)))
    return failure();
  auto yield = cast<YieldOp>(getBody().front().getTerminator());
  if (static_cast<bool>(yield.getValue()) != static_cast<bool>(getOutput()))
    return emitOpError("into and a yielded scalar are given together: the "
                       "scalar of each segment is stored in the into buffer");
  if (getOutput())
    return verifyYieldedScalar(getOperation(), getBody(), getOutput());
  return success();
}

LogicalResult MergeTasksOp::verify() {
  Type word = cast<MemRefType>(getMerges().getType()).getElementType();
  const std::pair<const char *, Type> words[] = {
      {"partial_count", getPartialCount().getType()},
      {"merge_count", getMergeCount().getType()},
      {"segment_count", getSegmentCount().getType()}};
  for (auto [name, type] : words)
    if (type != word)
      return emitOpError() << name << " must have the element type of the "
                           << "merges, " << word << ", got " << type;
  return success();
}

LogicalResult MergeTasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getScratch().getType()).getElementType();
  if (failed(verifyTaskRegion(getOperation(), getBody(), element, "scratch",
                              /*allowStores=*/false)))
    return failure();
  return verifyYieldedScalar(getOperation(), getBody(), getOutput());
}

LogicalResult FusedTasksOp::verify() {
  Type word = cast<MemRefType>(getOffsets().getType()).getElementType();
  const std::pair<const char *, Type> words[] = {
      {"value_count", getValueCount().getType()},
      {"segment_count", getSegmentCount().getType()},
      {"warp_task_count", getWarpTaskCount().getType()},
      {"cta_task_count", getCtaTaskCount().getType()},
      {"an element of ids",
       cast<MemRefType>(getIds().getType()).getElementType()}};
  for (auto [name, type] : words)
    if (type != word)
      return emitOpError() << name << " must have the element type of the "
                           << "offsets, " << word << ", got " << type;
  return success();
}

LogicalResult FusedTasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getValues().getType()).getElementType();
  for (Region *region : {&getWarp(), &getCta()})
    if (failed(verifyTaskRegion(getOperation(), *region, element, "values",
                                /*allowStores=*/false)) ||
        failed(verifyYieldedScalar(getOperation(), *region, getOutput())))
      return failure();
  return success();
}

LogicalResult PartialTasksOp::verify() {
  Type word = cast<MemRefType>(getRanges().getType()).getElementType();
  for (auto [name, type] :
       {std::pair("value_count", getValueCount().getType()),
        std::pair("partial_count", getPartialCount().getType())})
    if (type != word)
      return emitOpError() << name << " must have the element type of the "
                           << "ranges, " << word << ", got " << type;
  return success();
}

LogicalResult PartialTasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getValues().getType()).getElementType();
  if (failed(verifyTaskRegion(getOperation(), getBody(), element, "values",
                              /*allowStores=*/false)))
    return failure();
  return verifyYieldedScalar(getOperation(), getBody(), getScratch());
}
