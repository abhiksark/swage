// lib/Dialect/SwagePlan/IR/SwagePlanDialect.cpp
//===- SwagePlanDialect.cpp - SwagePlan dialect ----------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
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
  // The last region of a persistent task operation is its warp region. The
  // block, partial, and merge regions before it combine across the block.
  if (isa_and_nonnull<PersistentTasksOp>(task))
    return region->getRegionNumber() == 3 ? TaskPolicy::Warp : TaskPolicy::CTA;
  return std::nullopt;
}

/// Whether a kernel can take `type` as a parameter: a buffer it can address
/// through a pointer, or a count. A buffer has rank one, or rank two for
/// the rows of a column kernel.
static bool isKernelParameterType(Type type) {
  if (type.isSignlessInteger())
    return true;
  auto memref = dyn_cast<MemRefType>(type);
  return memref && (memref.getRank() == 1 || memref.getRank() == 2) &&
         llvm::all_of(memref.getShape(), ShapedType::isDynamic) &&
         memref.getLayout().isIdentity() && !memref.getMemorySpace() &&
         (memref.getElementType().isSignlessInteger() ||
          isa<FloatType>(memref.getElementType()));
}

LogicalResult
SwagePlanDialect::verifyRegionArgAttribute(Operation *op, unsigned regionIndex,
                                           unsigned argIndex,
                                           NamedAttribute attribute) {
  StringRef name = getSourceIndexAttrName();
  if (attribute.getName() != name)
    return op->emitError()
           << "'" << attribute.getName().strref()
           << "' is not an argument attribute of the swage_plan dialect; the "
              "dialect defines "
           << name;
  auto function = dyn_cast<func::FuncOp>(op);
  if (!function || !function->hasAttr(getBlockThreadsAttrName()))
    return op->emitError() << name
                           << " belongs on an argument of a plan function, a "
                              "func.func with "
                           << getBlockThreadsAttrName() << ", got argument #"
                           << argIndex << " of '" << op->getName() << "'";
  auto index = dyn_cast<IntegerAttr>(attribute.getValue());
  if (!index || !index.getType().isSignlessInteger(32) ||
      index.getValue().isNegative())
    return op->emitError() << name << " of argument #" << argIndex
                           << " must be a nonnegative i32, got "
                           << attribute.getValue();
  return success();
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
             << " must be a signless integer or a memref of rank one or two "
                "of signless integers or floats with dynamic sizes, the "
                "identity layout, and the default memory space, got "
             << input;
  if (function.isExternal() || !function.getBody().hasOneBlock())
    return op->emitError("a plan function has a body of one block");
  Block &body = function.getBody().front();
  if (llvm::range_size(body) != 2 ||
      !isa<TasksOp, PartialTasksOp, MergeTasksOp, FusedTasksOp,
           PersistentTasksOp>(body.front()) ||
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

  // Rank-two values are rows of `feature_count` columns, and a column
  // kernel is the kernel of such rows.
  auto values = cast<MemRefType>(getValues().getType());
  if ((values.getRank() == 2) != static_cast<bool>(getFeatureCount()))
    return emitOpError()
           << "feature_count is given exactly when the values have rank two, "
              "got "
           << values << (getFeatureCount() ? " with" : " without")
           << " a feature_count";
  bool column = getPolicy() == TaskPolicy::Column;
  if (!getFeatureCount()) {
    if (column)
      return emitOpError("policy<column> reduces the columns of rank-two "
                         "values and takes a feature_count");
    if (getOutput() && cast<MemRefType>(getOutput().getType()).getRank() != 1)
      return emitOpError()
             << "into must have rank one for rank-one values, got "
             << getOutput().getType();
    return success();
  }
  if (failed(requireWord("feature_count", getFeatureCount().getType())))
    return failure();
  if (getPolicy() == TaskPolicy::Warp)
    return emitOpError("rank-two values take policy<column>, policy<cta>, or "
                       "policy<sequential>, got policy<warp>");
  if (column && getIds())
    return emitOpError("policy<column> runs one task per segment, in order, "
                       "and takes no ids");
  if (getOutput() && cast<MemRefType>(getOutput().getType()).getRank() != 2)
    return emitOpError() << "into must have rank two for rank-two values, got "
                         << getOutput().getType();
  return success();
}

/// Whether the region of a task operation takes the extent of its segment,
/// its second argument.
static bool takesExtent(Region &region) {
  return region.front().getNumArguments() == 2;
}

/// Verify what every task region shares: one argument, the bound segment,
/// with the element type of the buffer it is bound from; consumers that read
/// that argument; and a `swage_plan.yield` at the end. `buffer` names the
/// buffer in a diagnostic, and `allowStores` admits `swage.map_store`.
///
/// With `allowExtent` the region may take the extent of the segment as a
/// second argument, of type `index`, and may then hold a scalar epilogue
/// after its consumers.
static LogicalResult verifyTaskRegion(Operation *task, Region &region,
                                      Type element, StringRef buffer,
                                      bool allowStores,
                                      bool allowExtent = false) {
  Block &body = region.front();
  bool extent = allowExtent && takesExtent(region);
  auto segment =
      body.getNumArguments() == 1 || extent
          ? dyn_cast<swage::SegmentType>(body.getArgument(0).getType())
          : swage::SegmentType();
  if (!segment)
    return task->emitOpError(
        "region takes the bound segment as its one argument, of type "
        "!swage.segment<T>");
  if (extent && !body.getArgument(1).getType().isIndex())
    return task->emitOpError()
           << "region takes the extent of the bound segment as its second "
              "argument, of type index, got "
           << body.getArgument(1).getType();
  if (segment.getElementType() != element)
    return task->emitOpError() << "region binds a segment of " << element
                               << ", the element type of the " << buffer
                               << ", got " << body.getArgument(0).getType();
  if (!body.mightHaveTerminator() || !isa<YieldOp>(body.getTerminator()))
    return task->emitOpError("region must end in swage_plan.yield");
  bool inEpilogue = false;
  for (Operation &operation : body.without_terminator()) {
    // The scalar epilogue of a region that takes an extent: ordinary
    // arithmetic, after every consumer.
    if (extent &&
        isa<arith::IndexCastOp, arith::SIToFPOp, arith::DivFOp>(operation)) {
      inEpilogue = true;
      continue;
    }
    bool admitted = allowStores
                        ? isa<swage::ReduceOp, swage::MapStoreOp>(operation)
                        : isa<swage::ReduceOp>(operation);
    if (!admitted) {
      InFlightDiagnostic diagnostic =
          operation.emitOpError()
          << "is not allowed in the region of '" << task->getName()
          << "'; the region holds "
          << (allowStores ? "swage.reduce and swage.map_store operations"
                          : "swage.reduce operations");
      if (extent)
        diagnostic << ", then the arith.index_cast, arith.sitofp, and "
                      "arith.divf operations of a scalar epilogue,";
      return diagnostic << " and ends in swage_plan.yield";
    }
    if (inEpilogue)
      return operation.emitOpError(
          "must come before the scalar epilogue of the task region");
    if (operation.getOperand(0) != body.getArgument(0))
      return operation.emitOpError(
          "must read the bound segment, the argument of the task region");
  }
  return success();
}

/// Require that the region of `task` yields a scalar of the element type of
/// `sink`, the buffer that receives it, which a diagnostic calls `name`.
static LogicalResult verifyYieldedScalar(Operation *task, Region &region,
                                         Value sink, StringRef name = "into") {
  auto yield = cast<YieldOp>(region.front().getTerminator());
  Type slot = cast<MemRefType>(sink.getType()).getElementType();
  if (!yield.getValue() || yield.getValue().getType() != slot) {
    InFlightDiagnostic diagnostic = task->emitOpError()
                                    << "region must yield " << slot
                                    << ", the element type of the " << name
                                    << " buffer, got ";
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
                              /*allowStores=*/true, /*allowExtent=*/true)))
    return failure();
  auto yield = cast<YieldOp>(getBody().front().getTerminator());
  if (static_cast<bool>(yield.getValue()) != static_cast<bool>(getOutput()))
    return emitOpError("into and a yielded scalar are given together: the "
                       "scalar of each segment is stored in the into buffer");
  if (getOutput())
    return verifyYieldedScalar(getOperation(), getBody(), getOutput());
  return success();
}

/// Require the buffers of a split task operation to be rows of
/// `featureCount` columns exactly when it is given, and of rank one
/// otherwise, and the count to be a word of the records.
static LogicalResult
verifyRowBuffers(Operation *task, Type word, Value featureCount,
                 ArrayRef<std::pair<StringRef, Value>> buffers) {
  int64_t rank = featureCount ? 2 : 1;
  for (auto [name, buffer] : buffers)
    if (cast<MemRefType>(buffer.getType()).getRank() != rank)
      return task->emitOpError()
             << name << " must have rank " << rank << " "
             << (featureCount ? "with" : "without") << " a feature_count, got "
             << buffer.getType();
  if (featureCount && featureCount.getType() != word)
    return task->emitOpError()
           << "feature_count must have the element type of the records, "
           << word << ", got " << featureCount.getType();
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
  if (getRanges() &&
      cast<MemRefType>(getRanges().getType()).getElementType() != word)
    return emitOpError()
           << "an element of ranges must have the element type of the merges, "
           << word << ", got "
           << cast<MemRefType>(getRanges().getType()).getElementType();
  return verifyRowBuffers(*this, word, getFeatureCount(),
                          {{"scratch", getScratch()}, {"into", getOutput()}});
}

LogicalResult MergeTasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getScratch().getType()).getElementType();
  if (failed(verifyTaskRegion(getOperation(), getBody(), element, "scratch",
                              /*allowStores=*/false, /*allowExtent=*/true)))
    return failure();
  // The bound range is scratch, so the extent of the split segment has to
  // come from the range records of its partial tasks.
  if (static_cast<bool>(getRanges()) != takesExtent(getBody()))
    return emitOpError("ranges and the extent argument of the region are "
                       "given together: the extent of a split segment is "
                       "read from the range records of its partial tasks");
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
                                /*allowStores=*/false, /*allowExtent=*/true)) ||
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
  return verifyRowBuffers(*this, word, getFeatureCount(),
                          {{"values", getValues()}, {"into", getScratch()}});
}

LogicalResult PartialTasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getValues().getType()).getElementType();
  if (failed(verifyTaskRegion(getOperation(), getBody(), element, "values",
                              /*allowStores=*/false)))
    return failure();
  return verifyYieldedScalar(getOperation(), getBody(), getScratch());
}

LogicalResult PersistentTasksOp::verify() {
  auto elementOf = [](Value buffer) {
    return cast<MemRefType>(buffer.getType()).getElementType();
  };
  Type word = elementOf(getOffsets());
  const std::pair<const char *, Type> words[] = {
      {"value_count", getValueCount().getType()},
      {"segment_count", getSegmentCount().getType()},
      {"an element of warp_ids", elementOf(getWarpIds())},
      {"warp_task_count", getWarpTaskCount().getType()},
      {"an element of cta_ids", elementOf(getCtaIds())},
      {"cta_task_count", getCtaTaskCount().getType()},
      {"an element of ranges", elementOf(getRanges())},
      {"an element of merge_ids", elementOf(getMergeIds())},
      {"partial_count", getPartialCount().getType()},
      {"an element of merges", elementOf(getMerges())},
      {"merge_count", getMergeCount().getType()},
      {"an element of counters", elementOf(getCounters())}};
  for (auto [name, type] : words)
    if (type != word)
      return emitOpError() << name << " must have the element type of the "
                           << "offsets, " << word << ", got " << type;
  return success();
}

LogicalResult PersistentTasksOp::verifyRegions() {
  Type element = cast<MemRefType>(getValues().getType()).getElementType();
  Type partial = cast<MemRefType>(getScratch().getType()).getElementType();
  Operation *task = getOperation();
  // A block task and a warp task reduce a segment of values into the output.
  for (Region *region : {&getCta(), &getWarp()})
    if (failed(verifyTaskRegion(task, *region, element, "values",
                                /*allowStores=*/false)) ||
        failed(verifyYieldedScalar(task, *region, getOutput())))
      return failure();
  // A partial task reduces a chunk of values into scratch, and a merge
  // reduces a range of scratch into the output.
  if (failed(verifyTaskRegion(task, getPartial(), element, "values",
                              /*allowStores=*/false)) ||
      failed(verifyYieldedScalar(task, getPartial(), getScratch(), "scratch")))
    return failure();
  if (failed(verifyTaskRegion(task, getMerge(), partial, "scratch",
                              /*allowStores=*/false)))
    return failure();
  return verifyYieldedScalar(task, getMerge(), getOutput());
}
