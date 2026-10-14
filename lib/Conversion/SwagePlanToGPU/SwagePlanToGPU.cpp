// lib/Conversion/SwagePlanToGPU/SwagePlanToGPU.cpp
//===- SwagePlanToGPU.cpp - Plan functions to GPU kernels -----------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// A dialect conversion with one pattern per operation. The function pattern
// builds the kernel shell, the task pattern binds one segment per task, and
// the consumer patterns lower the reductions and stores of the bound
// segment. What the patterns share is in Emission.h.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwagePlanToGPU/SwagePlanToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "swage/Conversion/SwagePlanToGPU/ConsumerPatterns.h"
#include "swage/Conversion/SwagePlanToGPU/Emission.h"
#include "swage/Conversion/SwageToPlan/Admission.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/SwagePlan/IR/KernelLayout.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Dialect/SwagePlan/IR/TaskRecords.h"
#include "swage/Support/KernelContract.h"
#include "swage/Target/TargetDescription.h"

namespace mlir::swage {
namespace {

using swage_plan::FusedTasksOp;
using swage_plan::MergeTasksOp;
using swage_plan::PartialTasksOp;
using swage_plan::PersistentTasksOp;
using swage_plan::SwagePlanDialect;
using swage_plan::TaskPolicy;
using swage_plan::TasksOp;

/// The launch width of a plan function, or a null attribute for any other
/// operation.
IntegerAttr blockThreadsOf(Operation *op) {
  if (!op || !isa<func::FuncOp>(op))
    return IntegerAttr();
  return op->getAttrOfType<IntegerAttr>(
      SwagePlanDialect::getBlockThreadsAttrName());
}

//===----------------------------------------------------------------------===//
// Kernel launch contracts
//===----------------------------------------------------------------------===//

using swage_plan::KernelKind;
using swage_plan::KernelLayout;
using LayoutArgument = swage_plan::KernelArgument;

/// The parameter types of the kernel a plan function becomes: every buffer
/// becomes a pointer and every count stays as it is.
SmallVector<Type> kernelInputsOf(func::FuncOp function) {
  Type pointer = LLVM::LLVMPointerType::get(function.getContext());
  SmallVector<Type> inputs;
  for (Type input : function.getFunctionType().getInputs())
    inputs.push_back(isa<MemRefType>(input) ? pointer : input);
  return inputs;
}

/// The kernel the task operation of a plan function becomes, which fixes the
/// parameter list in `KernelLayout.h`. None for a task operation that no
/// kernel takes in that form.
std::optional<KernelKind> kernelKindOf(Operation *task) {
  if (auto tasks = dyn_cast<TasksOp>(task)) {
    if (tasks.getFeatureCount())
      return tasks.getIds() ? KernelKind::TaskIdsColumns
                            : KernelKind::DirectColumns;
    return tasks.getIds() ? KernelKind::TaskIds : KernelKind::Direct;
  }
  if (isa<FusedTasksOp>(task))
    return KernelKind::FusedMixed;
  if (auto partial = dyn_cast<PartialTasksOp>(task))
    return partial.getFeatureCount() ? KernelKind::SplitPartialColumns
                                     : KernelKind::SplitPartial;
  if (auto merge = dyn_cast<MergeTasksOp>(task)) {
    if (merge.getFeatureCount())
      return merge.getRanges() ? KernelKind::SplitMergeExtentColumns
                               : KernelKind::SplitMergeColumns;
    return merge.getRanges() ? KernelKind::SplitMergeExtent
                             : KernelKind::SplitMerge;
  }
  if (isa<PersistentTasksOp>(task))
    return KernelKind::Persistent;
  return std::nullopt;
}

/// The name of a kernel in diagnostics.
StringRef kernelKindName(KernelKind kind) {
  switch (kind) {
  case KernelKind::Direct:
    return "direct";
  case KernelKind::DirectColumns:
    return "direct column";
  case KernelKind::TaskIds:
    return "task-id";
  case KernelKind::TaskIdsColumns:
    return "task-id column";
  case KernelKind::FusedMixed:
    return "fused mixed";
  case KernelKind::SplitPartial:
    return "split partial";
  case KernelKind::SplitMerge:
    return "split merge";
  case KernelKind::SplitMergeExtent:
    return "split merge with extents";
  case KernelKind::SplitPartialColumns:
    return "split partial column";
  case KernelKind::SplitMergeColumns:
    return "split merge column";
  case KernelKind::SplitMergeExtentColumns:
    return "split merge column with extents";
  case KernelKind::Persistent:
    return "persistent";
  }
  llvm_unreachable("unknown kernel kind");
}

/// The operand through which the task operation reads the kernel parameter
/// `argument`, or a null value when it takes none for it.
Value taskOperandOf(Operation *task, LayoutArgument argument) {
  if (auto tasks = dyn_cast<TasksOp>(task)) {
    switch (argument) {
    case LayoutArgument::Values:
      return tasks.getValues();
    case LayoutArgument::Offsets:
      return tasks.getOffsets();
    case LayoutArgument::Output:
      return tasks.getOutput();
    case LayoutArgument::TaskIds:
      return tasks.getIds();
    case LayoutArgument::ValueCount:
      return tasks.getValueCount();
    case LayoutArgument::TaskCount:
      return tasks.getTaskCount();
    case LayoutArgument::SegmentCount:
      return tasks.getSegmentCount();
    case LayoutArgument::FeatureCount:
      return tasks.getFeatureCount();
    default:
      return Value();
    }
  }
  if (auto fused = dyn_cast<FusedTasksOp>(task)) {
    switch (argument) {
    case LayoutArgument::Values:
      return fused.getValues();
    case LayoutArgument::Offsets:
      return fused.getOffsets();
    case LayoutArgument::Output:
      return fused.getOutput();
    case LayoutArgument::TaskIds:
      return fused.getIds();
    case LayoutArgument::ValueCount:
      return fused.getValueCount();
    case LayoutArgument::WarpTaskCount:
      return fused.getWarpTaskCount();
    case LayoutArgument::CtaTaskCount:
      return fused.getCtaTaskCount();
    case LayoutArgument::SegmentCount:
      return fused.getSegmentCount();
    default:
      return Value();
    }
  }
  if (auto partial = dyn_cast<PartialTasksOp>(task)) {
    switch (argument) {
    case LayoutArgument::Values:
      return partial.getValues();
    case LayoutArgument::PartialRanges:
      return partial.getRanges();
    case LayoutArgument::Scratch:
      return partial.getScratch();
    case LayoutArgument::ValueCount:
      return partial.getValueCount();
    case LayoutArgument::PartialCount:
      return partial.getPartialCount();
    case LayoutArgument::FeatureCount:
      return partial.getFeatureCount();
    default:
      return Value();
    }
  }
  if (auto merge = dyn_cast<MergeTasksOp>(task)) {
    switch (argument) {
    case LayoutArgument::Scratch:
      return merge.getScratch();
    case LayoutArgument::Output:
      return merge.getOutput();
    case LayoutArgument::MergeRecords:
      return merge.getMerges();
    case LayoutArgument::PartialRanges:
      return merge.getRanges();
    case LayoutArgument::PartialCount:
      return merge.getPartialCount();
    case LayoutArgument::MergeCount:
      return merge.getMergeCount();
    case LayoutArgument::SegmentCount:
      return merge.getSegmentCount();
    case LayoutArgument::FeatureCount:
      return merge.getFeatureCount();
    default:
      return Value();
    }
  }
  if (auto persistent = dyn_cast<PersistentTasksOp>(task)) {
    switch (argument) {
    case LayoutArgument::Values:
      return persistent.getValues();
    case LayoutArgument::Offsets:
      return persistent.getOffsets();
    case LayoutArgument::Output:
      return persistent.getOutput();
    case LayoutArgument::WarpIds:
      return persistent.getWarpIds();
    case LayoutArgument::CtaIds:
      return persistent.getCtaIds();
    case LayoutArgument::PartialRanges:
      return persistent.getRanges();
    case LayoutArgument::PartialMergeIds:
      return persistent.getMergeIds();
    case LayoutArgument::MergeRecords:
      return persistent.getMerges();
    case LayoutArgument::Scratch:
      return persistent.getScratch();
    case LayoutArgument::Counters:
      return persistent.getCounters();
    case LayoutArgument::ValueCount:
      return persistent.getValueCount();
    case LayoutArgument::WarpTaskCount:
      return persistent.getWarpTaskCount();
    case LayoutArgument::CtaTaskCount:
      return persistent.getCtaTaskCount();
    case LayoutArgument::PartialCount:
      return persistent.getPartialCount();
    case LayoutArgument::MergeCount:
      return persistent.getMergeCount();
    case LayoutArgument::SegmentCount:
      return persistent.getSegmentCount();
    default:
      return Value();
    }
  }
  return Value();
}

/// The contract kind of a count parameter.
std::optional<KernelArgumentKind> countKindOf(Type type) {
  switch (type.getIntOrFloatBitWidth()) {
  case 1:
    return KernelArgumentKind::I1;
  case 8:
    return KernelArgumentKind::I8;
  case 16:
    return KernelArgumentKind::I16;
  case 32:
    return KernelArgumentKind::I32;
  case 64:
    return KernelArgumentKind::I64;
  default:
    return std::nullopt;
  }
}

/// How the kernel `kind` uses its scratch buffer: a partial task writes its
/// slot, a merge reads the slots of its segment, and the persistent kernel
/// does both.
KernelArgumentAccess scratchAccess(KernelKind kind) {
  switch (kind) {
  case KernelKind::SplitPartial:
  case KernelKind::SplitPartialColumns:
    return KernelArgumentAccess::Write;
  case KernelKind::Persistent:
    return KernelArgumentAccess::ReadWrite;
  default:
    return KernelArgumentAccess::Read;
  }
}

/// The contract entry of one kernel parameter.
///
/// The arguments the segment function declares, which carry a role, bind
/// the caller's arguments at their source index: the values, the offsets,
/// the output, and the counts of values, segments, and features, which the
/// caller states (ADR-0020). The task records the host classifier produces
/// are plan arguments, the buffers the host allocates for the kernel are
/// scratch, and the record counts follow from the records and are derived.
/// Every keyed argument is keyed by its name in `KernelLayout.h`.
std::optional<KernelArgument> contractArgumentOf(LayoutArgument argument,
                                                 KernelKind kind, Type type,
                                                 uint32_t sourceIndex) {
  using Access = KernelArgumentAccess;
  using Origin = KernelArgumentOrigin;
  StringRef key = swage_plan::kernelArgumentName(argument);
  switch (argument) {
  case LayoutArgument::Values:
  case LayoutArgument::Offsets:
    return KernelArgument::userPointer(sourceIndex, Access::Read);
  case LayoutArgument::Output:
    return KernelArgument::userPointer(sourceIndex, Access::Write);
  case LayoutArgument::TaskIds:
  case LayoutArgument::WarpIds:
  case LayoutArgument::CtaIds:
  case LayoutArgument::PartialRanges:
  case LayoutArgument::PartialMergeIds:
  case LayoutArgument::MergeRecords:
    return KernelArgument::keyedPointer(Origin::Plan, key, Access::Read);
  case LayoutArgument::Scratch:
    return KernelArgument::keyedPointer(Origin::Scratch, key,
                                        scratchAccess(kind));
  case LayoutArgument::Counters:
    return KernelArgument::keyedPointer(Origin::Scratch, key,
                                        Access::ReadWrite);
  case LayoutArgument::ValueCount:
  case LayoutArgument::SegmentCount:
  case LayoutArgument::FeatureCount:
    if (std::optional<KernelArgumentKind> count = countKindOf(type))
      return KernelArgument::userScalar(*count, sourceIndex);
    return std::nullopt;
  case LayoutArgument::TaskCount:
  case LayoutArgument::WarpTaskCount:
  case LayoutArgument::CtaTaskCount:
  case LayoutArgument::PartialCount:
  case LayoutArgument::MergeCount:
    if (std::optional<KernelArgumentKind> count = countKindOf(type))
      return KernelArgument::keyedScalar(*count, Origin::Derived, key);
    return std::nullopt;
  }
  llvm_unreachable("unknown kernel argument");
}

/// The parameter of `function` that the task operation reads as the kernel
/// argument `argument`, or a null value when it reads none. The output of a
/// program that stores through `swage.map_store` is no operand of its task
/// operation: the map store in the region writes it.
BlockArgument parameterOf(func::FuncOp function, Operation *task,
                          LayoutArgument argument) {
  Value operand = taskOperandOf(task, argument);
  if (!operand && argument == LayoutArgument::Output && isa<TasksOp>(task))
    task->walk([&](MapStoreOp store) {
      operand = store.getOutput();
      return WalkResult::interrupt();
    });
  auto parameter = dyn_cast_or_null<BlockArgument>(operand);
  if (!parameter || parameter.getOwner() != &function.getBody().front())
    return BlockArgument();
  return parameter;
}

/// Build the launch contract of the kernel a plan function becomes.
///
/// The parameter layout of the kernel names the arguments it takes. Each
/// one is found among the parameters of the plan function through the
/// operand of the task operation that reads it, so the contract describes
/// the parameters in the order the function declares them, which is the
/// order of the kernel. Every parameter must be exactly one argument of the
/// layout. A failure is reported on the function.
FailureOr<KernelContract> buildKernelContract(func::FuncOp function) {
  Operation *task = &function.getBody().front().front();
  std::optional<KernelKind> kind = kernelKindOf(task);
  if (!kind)
    return function.emitError()
           << "plan function @" << function.getName()
           << " holds a task operation that no kernel layout describes, '"
           << task->getName() << "' with both ids and a feature_count";
  KernelLayout layout = swage_plan::kernelLayout(*kind);

  SmallVector<std::optional<LayoutArgument>> roles(function.getNumArguments());
  for (LayoutArgument argument : layout.arguments()) {
    StringRef name = swage_plan::kernelArgumentName(argument);
    BlockArgument parameter = parameterOf(function, task, argument);
    if (!parameter)
      return function.emitError()
             << "plan function @" << function.getName() << " must pass the "
             << name << " of the " << kernelKindName(*kind)
             << " kernel as a parameter of the function";
    std::optional<LayoutArgument> &role = roles[parameter.getArgNumber()];
    if (role)
      return function.emitError()
             << "plan function @" << function.getName() << " parameter #"
             << parameter.getArgNumber() << " is both the "
             << swage_plan::kernelArgumentName(*role) << " and the " << name
             << " of the " << kernelKindName(*kind)
             << " kernel; a kernel parameter has one meaning";
    role = argument;
  }

  SmallVector<Type> inputs = kernelInputsOf(function);
  KernelContract contract;
  contract.backend = KernelBackend::CUDA;
  contract.entry = function.getName().str();
  contract.launch = {
      KernelLaunchModel::SPMDGrid,
      std::array<int32_t, 3>{
          static_cast<int32_t>(blockThreadsOf(function).getInt()), 1, 1}};
  for (auto [index, role] : llvm::enumerate(roles)) {
    auto position = static_cast<unsigned>(index);
    if (!role)
      return function.emitError()
             << "plan function @" << function.getName() << " parameter #"
             << position << " is none of the " << layout.size()
             << " arguments of the " << kernelKindName(*kind)
             << " kernel, which its task operation reads";
    uint32_t sourceIndex = position;
    if (auto recorded = function.getArgAttrOfType<IntegerAttr>(
            position, SwagePlanDialect::getSourceIndexAttrName()))
      sourceIndex = static_cast<uint32_t>(recorded.getInt());
    std::optional<KernelArgument> entry =
        contractArgumentOf(*role, *kind, inputs[position], sourceIndex);
    if (!entry)
      return function.emitError()
             << "plan function @" << function.getName() << " parameter #"
             << position << ", the " << swage_plan::kernelArgumentName(*role)
             << ", must be an integer of 1, 8, 16, 32, or 64 bits, got "
             << inputs[position];
    contract.arguments.push_back(std::move(*entry));
  }

  FunctionType type = FunctionType::get(function.getContext(), inputs, {});
  if (llvm::Error error = validateCUDAKernelContract(
          contract, function.getName(), type, *contract.launch.block))
    return function.emitError() << "plan function @" << function.getName()
                                << " gives an invalid kernel contract: "
                                << llvm::toString(std::move(error));
  return contract;
}

/// The element type of the exchange buffer a task operation needs, or a
/// null type: the threads of a row-stripe tile of rank-two values combine
/// the results of one column through one element per thread.
Type exchangeElementOf(Operation *task) {
  auto elementOf = [](Value buffer) {
    return cast<MemRefType>(buffer.getType()).getElementType();
  };
  auto tasks = dyn_cast<TasksOp>(task);
  if (tasks && tasks.getFeatureCount() && tasks.getPolicy() == TaskPolicy::CTA)
    return elementOf(tasks.getValues());
  auto partial = dyn_cast<PartialTasksOp>(task);
  if (partial && partial.getFeatureCount())
    return elementOf(partial.getValues());
  auto merge = dyn_cast<MergeTasksOp>(task);
  if (merge && merge.getFeatureCount())
    return elementOf(merge.getScratch());
  return Type();
}

/// A plan function becomes a `gpu.module` named after it that holds the
/// kernel: every buffer argument becomes a pointer, every count stays as it
/// is, and the launch width is pinned through the target. The kernel carries
/// its launch contract as the discardable attribute `swage.kernel_contract`,
/// which code generation validates and removes. The kernel of a persistent
/// task operation also gets the two claim slots its blocks share, and the
/// kernel of a row-stripe tile its exchange buffer, one element per thread.
///
/// The body is legalized while the function is still its parent, so the
/// task pattern reads the launch width from the function it sits in, and
/// the converted operations are then moved into the kernel.
class PlanKernelFuncPattern : public OpConversionPattern<func::FuncOp> {
public:
  PlanKernelFuncPattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(func::FuncOp function, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(function);
    if (!threads)
      return rewriter.notifyMatchFailure(function, "not a plan function");
    MLIRContext *context = function.getContext();
    Location loc = function.getLoc();

    // The pass built the contract once before the conversion, which
    // reported any failure, so this build succeeds.
    FailureOr<KernelContract> contract = buildKernelContract(function);
    if (failed(contract))
      return failure();

    rewriter.setInsertionPoint(function);
    auto gpuModule = gpu::GPUModuleOp::create(
        rewriter, loc, function.getName().str() + "_module");
    rewriter.setInsertionPointToStart(gpuModule.getBody());
    auto kernel = gpu::GPUFuncOp::create(
        rewriter, loc, function.getName(),
        FunctionType::get(context, kernelInputsOf(function), {}));
    kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                    rewriter.getUnitAttr());
    target.pinLaunchWidth(kernel, static_cast<int32_t>(threads.getInt()));
    kernel->setDiscardableAttr(kernelContractAttrName,
                               buildKernelContractAttr(rewriter, *contract));

    Block &body = function.getBody().front();
    Block *entry = &kernel.getBody().front();
    for (auto [argument, parameter] :
         llvm::zip_equal(body.getArguments(), entry->getArguments()))
      rewriter.replaceAllUsesWith(argument, parameter);
    // One slot carries a claim from the leader of a block to its threads,
    // and the other the merge a partial task made ready.
    if (isa<PersistentTasksOp>(body.front()))
      kernel.addWorkgroupAttribution(
          MemRefType::get(
              {2}, rewriter.getI32Type(), AffineMap(),
              gpu::AddressSpaceAttr::get(
                  context, gpu::GPUDialect::getWorkgroupAddressSpace())),
          loc);
    if (Type element = exchangeElementOf(&body.front()))
      kernel.addWorkgroupAttribution(
          MemRefType::get(
              {threads.getInt()}, element, AffineMap(),
              gpu::AddressSpaceAttr::get(
                  context, gpu::GPUDialect::getWorkgroupAddressSpace())),
          loc);
    if (failed(legalizeInPlace(rewriter, operationsOf(body))))
      return failure();
    moveConvertedOperations<TasksOp, PartialTasksOp, MergeTasksOp, FusedTasksOp,
                            PersistentTasksOp, func::ReturnOp>(rewriter, body,
                                                               entry);
    rewriter.eraseOp(function);
    return success();
  }

private:
  const TargetDescription &target;
};

/// The return of a plan function becomes the return of its kernel.
class PlanKernelReturnPattern : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp returnOp, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!blockThreadsOf(returnOp->getParentOp()))
      return rewriter.notifyMatchFailure(returnOp, "not in a plan function");
    gpu::ReturnOp::create(rewriter, returnOp.getLoc());
    rewriter.eraseOp(returnOp);
    return success();
  }
};

/// What a task site needs to bind a segment by its ID: the buffers and
/// bounds of the task operation, as the kernel sees them, and the two index
/// constants of the kernel prelude.
struct SegmentSite {
  Value values;
  Value offsets;
  Value valueCount;
  Value segmentCount;
  Value output;
  Value zero;
  Value one;
};

/// Run the consumers of a task region on the segment of one task, at the
/// insertion point of `body`.
///
/// With `ids`, the segment ID is the word at `ids[taskId]`, compared with
/// the segment count. Without, it is `taskId` itself, which the caller
/// compared with the segment count. The segment is bound for the thread
/// `logicalThreadId` at `stride`; the region argument is replaced by the
/// four values of the binding; the consumers are legalized while the task
/// operation is still their parent, so each consumer pattern takes the
/// binding from its operands and the policy from the operation it sits in;
/// and the yielded scalar is stored by the thread whose `logicalThreadId`
/// is zero. Returns false when a consumer could not be legalized.
///
/// A region that takes the extent of its segment gets the clamped end minus
/// the clamped start, which is the same in every thread. Its scalar
/// epilogue is ordinary arithmetic, so it is legal as it stands and every
/// thread runs it after the reductions.
bool convertSegmentTask(ConversionPatternRewriter &rewriter, OpBuilder &body,
                        Location loc, Region &region, const SegmentSite &site,
                        Value ids, Value taskId, Value logicalThreadId,
                        Value stride) {
  Value segmentId = taskId;
  Value segmentInRange;
  if (ids) {
    Value segmentIdWord = loadTaskWord(body, loc, ids, taskId);
    segmentInRange =
        isLoadedIndexInRange(body, loc, segmentIdWord, site.segmentCount);
    segmentId = arith::IndexCastOp::create(body, loc, body.getIndexType(),
                                           segmentIdWord);
  }
  BoundSegment bound = emitSegmentBinding(
      body, loc, site.values, site.offsets, site.valueCount, segmentId,
      segmentInRange, logicalThreadId, stride, site.zero, site.one);
  Block &consumersBlock = region.front();
  rewriter.replaceAllUsesWith(consumersBlock.getArgument(0),
                              ValueRange{bound.segment.base,
                                         bound.segment.first, bound.segment.end,
                                         bound.segment.stride});
  if (consumersBlock.getNumArguments() == 2)
    rewriter.replaceAllUsesWith(
        consumersBlock.getArgument(1),
        arith::SubIOp::create(body, loc, bound.segment.end, bound.start)
            .getResult());

  auto yield = cast<swage_plan::YieldOp>(consumersBlock.getTerminator());
  SmallVector<Operation *> consumers = operationsOf(consumersBlock);
  consumers.pop_back();
  if (failed(legalizeInPlace(rewriter, consumers)))
    return false;
  moveConvertedOperations<ReduceOp, MapStoreOp, swage_plan::YieldOp>(
      rewriter, consumersBlock, body.getInsertionBlock());
  if (Value scalar = yield.getValue())
    emitScalarStore(body, loc, rewriter.getRemappedValue(scalar), site.output,
                    bound.segmentId64, logicalThreadId, site.zero,
                    segmentInRange);
  return true;
}

/// The column tile of rank-two values, at the insertion point of `body`,
/// for the segment `segment` of a block that passed its guard: thread `t`
/// reduces the columns `t`, `t + block`, and so on of the rows of the
/// segment, one after the other, and stores each result itself.
///
/// A column is a strided run of the row-order values: it starts at
/// `start * columns + column` and takes every `columns`-th element below
/// `end * columns`, where `[start, end)` are the rows of the segment after
/// their clamp to the row count. The indices are index values, 64 bits wide
/// in a kernel, so the element count may exceed what an i32 holds.
///
/// Nothing is combined across threads: the consumers run with
/// `ThreadCombination::None`, the kernel holds no shuffle, barrier, or
/// shared memory, and a thread holds one scalar per reduction whatever the
/// number of columns. The column loop is bounded by `feature_count`, which
/// is all that keeps a store inside the row of its segment. Returns false
/// when a consumer could not be legalized.
bool convertColumnTask(ConversionPatternRewriter &rewriter, OpBuilder &body,
                       Location loc, Region &region, const SegmentSite &site,
                       Value featureCount, Value segment, Value threadId,
                       Value block) {
  SegmentRange rows = emitSegmentRange(body, loc, site.offsets, site.valueCount,
                                       segment, Value(), site.zero, site.one);
  Value columns =
      arith::IndexCastOp::create(body, loc, body.getIndexType(), featureCount);
  Value firstRow = arith::MulIOp::create(body, loc, rows.start, columns);
  Value end = arith::MulIOp::create(body, loc, rows.end, columns);
  Value outputRow = arith::MulIOp::create(body, loc, segment, columns);
  Block &consumersBlock = region.front();
  Value extent;
  if (consumersBlock.getNumArguments() == 2)
    extent = arith::SubIOp::create(body, loc, rows.end, rows.start);

  bool converted = true;
  scf::ForOp::create(
      body, loc, threadId, columns, block, ValueRange(),
      [&](OpBuilder &loop, Location loopLoc, Value column, ValueRange) {
        Value first = arith::AddIOp::create(loop, loopLoc, firstRow, column);
        rewriter.replaceAllUsesWith(
            consumersBlock.getArgument(0),
            ValueRange{site.values, first, end, columns});
        if (extent)
          rewriter.replaceAllUsesWith(consumersBlock.getArgument(1), extent);
        auto yield = cast<swage_plan::YieldOp>(consumersBlock.getTerminator());
        SmallVector<Operation *> consumers = operationsOf(consumersBlock);
        consumers.pop_back();
        converted = succeeded(legalizeInPlace(rewriter, consumers));
        if (converted) {
          moveConvertedOperations<ReduceOp, MapStoreOp, swage_plan::YieldOp>(
              rewriter, consumersBlock, loop.getInsertionBlock());
          if (Value scalar = yield.getValue()) {
            Value total = rewriter.getRemappedValue(scalar);
            Type pointer = LLVM::LLVMPointerType::get(loop.getContext());
            Value slot =
                arith::AddIOp::create(loop, loopLoc, outputRow, column);
            Value slot64 = arith::IndexCastOp::create(loop, loopLoc,
                                                      loop.getI64Type(), slot);
            Value address = LLVM::GEPOp::create(
                loop, loopLoc, pointer, total.getType(), site.output, slot64);
            LLVM::StoreOp::create(loop, loopLoc, total, address);
          }
        }
        scf::YieldOp::create(loop, loopLoc);
      });
  return converted;
}

/// The geometry of the row-stripe tile of rank-two values, which every item
/// of a block shares. A block of `T` threads splits into `T / W` row stripes
/// of each of `W` adjacent columns: thread `t` is lane `t mod S` of subgroup
/// `t / S`, for the subgroup width `S`, owns column `lane mod W` of a group,
/// and is stripe `(t / S) * (S / W) + lane / W`. `W` is the column-group
/// width of the feature count, a power of two that divides `S`, and `T` is
/// a multiple of `S`, so the lanes of one column within a subgroup are an
/// XOR butterfly.
struct RowTile {
  Value features;  ///< The column count, as an index.
  Value width;     ///< `W`, the columns of one group.
  Value groups;    ///< The column groups, `ceil(features / W)`.
  Value column;    ///< The column of this thread within a group.
  Value stripe;    ///< The row stripe of this thread.
  Value rowStride; ///< The elements between two rows of one stripe.
  Value exchange;  ///< The workgroup buffer of the combination.
};

RowTile emitRowTile(OpBuilder &builder, Location loc,
                    const TargetDescription &target, Value featureCount,
                    Value threadId, int64_t threads, Value exchange) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value features = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), featureCount);
  Value width =
      emitColumnGroupWidth(builder, loc, features, target.subgroupWidth);
  // A feature count of zero or below names no column, so the block has no
  // item and nothing below reads the count.
  Value hasColumns = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sgt, features, zero);
  Value groups = arith::SelectOp::create(
      builder, loc, hasColumns,
      arith::DivUIOp::create(
          builder, loc,
          arith::AddIOp::create(
              builder, loc, features,
              arith::SubIOp::create(builder, loc, width, one)),
          width),
      zero);
  Value subgroupWidth =
      arith::ConstantIndexOp::create(builder, loc, target.subgroupWidth);
  Value lane = arith::RemUIOp::create(builder, loc, threadId, subgroupWidth);
  Value subgroup =
      arith::DivUIOp::create(builder, loc, threadId, subgroupWidth);
  Value column = arith::RemUIOp::create(builder, loc, lane, width);
  Value stripesPerSubgroup =
      arith::DivUIOp::create(builder, loc, subgroupWidth, width);
  Value stripe = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, subgroup, stripesPerSubgroup),
      arith::DivUIOp::create(builder, loc, lane, width));
  Value stripes = arith::DivUIOp::create(
      builder, loc, arith::ConstantIndexOp::create(builder, loc, threads),
      width);
  Value rowStride = arith::MulIOp::create(builder, loc, stripes, features);
  return {features, width, groups, column, stripe, rowStride, exchange};
}

/// Run `emitItem` for the items of this block at the insertion point of
/// `builder`: the items `b`, `b + G`, and so on below `taskCount * groups`,
/// for the block index `b` and the grid size `G`. Item `i` is task
/// `i / groups` and column group `i mod groups`. The bounds are the same in
/// every thread of a block, so the barriers of the item stay legal, and a
/// launch of one block per item runs one item per block.
void emitItemLoop(
    OpBuilder &builder, Location loc, Value taskCount, Value groups,
    function_ref<void(OpBuilder &, Location, Value, Value)> emitItem) {
  Value blockIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value gridSize = gpu::GridDimOp::create(builder, loc, gpu::Dimension::x);
  Value items = arith::MulIOp::create(builder, loc, taskCount, groups);
  scf::ForOp::create(
      builder, loc, blockIndex, items, gridSize, ValueRange(),
      [&](OpBuilder &loop, Location loopLoc, Value item, ValueRange) {
        Value task = arith::DivUIOp::create(loop, loopLoc, item, groups);
        Value group = arith::RemUIOp::create(loop, loopLoc, item, groups);
        emitItem(loop, loopLoc, task, group);
        scf::YieldOp::create(loop, loopLoc);
      });
}

/// The rows `[start, end)` of column group `group` for this thread of a
/// row-stripe tile: its column of the group and the binding of its stripe,
/// which reads `(start + stripe) * D + c` and every `rowStride`-th element
/// after it below `end * D`, for `D` features and the column `c`. A column
/// at or beyond `D` binds no element: its first element is the end. The
/// bound is on the address and not on the control flow, so every thread
/// reaches each shuffle and barrier of the stages that follow.
struct BoundRows {
  SegmentBinding binding;
  Value column;        ///< The column `c` of this thread.
  Value columnInRange; ///< Whether `c` is below `D`.
};

BoundRows bindRows(OpBuilder &builder, Location loc, const RowTile &tile,
                   Value base, Value start, Value end, Value group) {
  Value column = arith::AddIOp::create(
      builder, loc, arith::MulIOp::create(builder, loc, group, tile.width),
      tile.column);
  Value columnInRange = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ult, column, tile.features);
  Value firstRow = arith::AddIOp::create(builder, loc, start, tile.stripe);
  Value first = arith::AddIOp::create(
      builder, loc,
      arith::MulIOp::create(builder, loc, firstRow, tile.features), column);
  Value last = arith::MulIOp::create(builder, loc, end, tile.features);
  first = arith::SelectOp::create(builder, loc, columnInRange, first, last);
  return {{base, first, last, tile.rowStride, tile.width, tile.exchange},
          column,
          columnInRange};
}

/// Run the consumers of a task region on rows `rows` of one column group of
/// a row-stripe tile, at the insertion point of `body`, and return the
/// scalar the region yields, or a null value when the region yields none.
/// Sets `converted` to false when a consumer could not be legalized.
///
/// The region argument is replaced by the six values of the binding, so
/// each reduction combines per column, and an extent argument by `extent`,
/// a number of rows.
Value convertRowRegion(ConversionPatternRewriter &rewriter, OpBuilder &body,
                       Region &region, const BoundRows &rows, Value extent,
                       bool &converted) {
  Block &consumersBlock = region.front();
  const SegmentBinding &binding = rows.binding;
  rewriter.replaceAllUsesWith(consumersBlock.getArgument(0),
                              ValueRange{binding.base, binding.first,
                                         binding.end, binding.stride,
                                         binding.groupWidth, binding.exchange});
  if (extent)
    rewriter.replaceAllUsesWith(consumersBlock.getArgument(1), extent);
  auto yield = cast<swage_plan::YieldOp>(consumersBlock.getTerminator());
  SmallVector<Operation *> consumers = operationsOf(consumersBlock);
  consumers.pop_back();
  if (failed(legalizeInPlace(rewriter, consumers))) {
    converted = false;
    return Value();
  }
  moveConvertedOperations<ReduceOp, MapStoreOp, swage_plan::YieldOp>(
      rewriter, consumersBlock, body.getInsertionBlock());
  if (Value scalar = yield.getValue())
    return rewriter.getRemappedValue(scalar);
  return Value();
}

/// Store the result of one column of a row-stripe tile at
/// `sink[row * D + column]` from the thread of stripe zero of that column,
/// and only when the column is one of the features and `rowInRange`, if
/// given, holds.
void emitRowStore(OpBuilder &body, Location loc, const RowTile &tile,
                  const BoundRows &rows, Value total, Value sink, Value row,
                  Value rowInRange, Value zero) {
  Value slot = arith::AddIOp::create(
      body, loc, arith::MulIOp::create(body, loc, row, tile.features),
      rows.column);
  Value mayStore = rows.columnInRange;
  if (rowInRange)
    mayStore = arith::AndIOp::create(body, loc, mayStore, rowInRange);
  emitLeaderStore(body, loc, total, sink, slot, tile.stripe, zero, mayStore);
}

/// The kernel of a row-stripe tile finds its exchange buffer through the
/// kernel that owns `parameter`, one of its buffer parameters.
Value exchangeOf(Value parameter) {
  auto argument = dyn_cast<BlockArgument>(parameter);
  auto kernel =
      argument ? dyn_cast<gpu::GPUFuncOp>(argument.getOwner()->getParentOp())
               : gpu::GPUFuncOp();
  if (!kernel || kernel.getWorkgroupAttributions().empty())
    return Value();
  return kernel.getWorkgroupAttributions().front();
}

/// One block of threads per task: the kernel prelude, the guard on the task
/// index, and the segment of the task. The direct kernel uses the block
/// index as the segment ID; with a task buffer the ID is loaded from it. A
/// task of `policy<column>` is the column tile of rank-two values, and a
/// task of `policy<cta>` over rank-two values the row-stripe tile, which
/// loops over the items of its block: a task and a group of columns each.
class TasksPattern : public OpConversionPattern<TasksOp> {
public:
  TasksPattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(TasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(tasks->getParentOp());
    if (!threads)
      return rewriter.notifyMatchFailure(tasks, "not in a plan function");
    Location loc = tasks.getLoc();
    Value ids = adaptor.getIds();
    Value featureCount = adaptor.getFeatureCount();
    if (featureCount && tasks.getPolicy() == TaskPolicy::CTA)
      return convertRowTasks(tasks, adaptor, rewriter, threads.getInt());

    Value taskIndex = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value block =
        arith::ConstantIndexOp::create(rewriter, loc, threads.getInt());
    // One block per task with a task buffer, and one per segment without.
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(),
        ids ? adaptor.getTaskCount() : adaptor.getSegmentCount());
    Value inRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, taskIndex, taskCount);

    SegmentSite site{adaptor.getValues(),
                     adaptor.getOffsets(),
                     adaptor.getValueCount(),
                     adaptor.getSegmentCount(),
                     adaptor.getOutput(),
                     zero,
                     one};
    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
          converted =
              featureCount
                  ? convertColumnTask(rewriter, body, bodyLoc, tasks.getBody(),
                                      site, featureCount, taskIndex, threadId,
                                      block)
                  : convertSegmentTask(rewriter, body, bodyLoc, tasks.getBody(),
                                       site, ids, taskIndex, threadId, block);
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

private:
  /// The row-stripe tile of rank-two values: for each item of the block,
  /// the segment of its task, the rows of that segment after their clamp,
  /// the consumers on the column group of the item, and the store of each
  /// column of the group. With a task buffer the segment is loaded from it
  /// and compared with the segment count; without, task `t` is segment `t`.
  LogicalResult convertRowTasks(TasksOp tasks, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                int64_t threads) const {
    Value exchange = exchangeOf(adaptor.getValues());
    if (!exchange)
      return rewriter.notifyMatchFailure(tasks, "kernel has no exchange");
    Location loc = tasks.getLoc();
    Value ids = adaptor.getIds();
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    RowTile tile = emitRowTile(rewriter, loc, target, adaptor.getFeatureCount(),
                               threadId, threads, exchange);
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(),
        ids ? adaptor.getTaskCount() : adaptor.getSegmentCount());
    bool converted = true;
    emitItemLoop(
        rewriter, loc, taskCount, tile.groups,
        [&](OpBuilder &body, Location bodyLoc, Value task, Value group) {
          Value segment = task;
          Value segmentInRange;
          if (ids) {
            Value segmentWord = loadTaskWord(body, bodyLoc, ids, task);
            segmentInRange = isLoadedIndexInRange(body, bodyLoc, segmentWord,
                                                  adaptor.getSegmentCount());
            segment = arith::IndexCastOp::create(
                body, bodyLoc, body.getIndexType(), segmentWord);
          }
          SegmentRange range = emitSegmentRange(
              body, bodyLoc, adaptor.getOffsets(), adaptor.getValueCount(),
              segment, segmentInRange, zero, one);
          BoundRows rows = bindRows(body, bodyLoc, tile, adaptor.getValues(),
                                    range.start, range.end, group);
          // The extent of a segment is its number of rows.
          Value extent;
          if (tasks.getBody().front().getNumArguments() == 2)
            extent =
                arith::SubIOp::create(body, bodyLoc, range.end, range.start);
          Value total = convertRowRegion(rewriter, body, tasks.getBody(), rows,
                                         extent, converted);
          if (total)
            emitRowStore(body, bodyLoc, tile, rows, total, adaptor.getOutput(),
                         segment, segmentInRange, zero);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

  const TargetDescription &target;
};

/// Warp tasks and block tasks in one launch. The first blocks each run one
/// warp task per subgroup, on the lanes of that subgroup, and every block
/// after them runs one block task. A task index beyond its count runs
/// nothing.
class FusedTasksPattern : public OpConversionPattern<FusedTasksOp> {
public:
  FusedTasksPattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(FusedTasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(tasks->getParentOp());
    if (!threads)
      return rewriter.notifyMatchFailure(tasks, "not in a plan function");
    Location loc = tasks.getLoc();
    Value ids = adaptor.getIds();

    Value taskIndex = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value block =
        arith::ConstantIndexOp::create(rewriter, loc, threads.getInt());
    // One warp task per subgroup of the block: `four` slots, and `three` to
    // round the warp task count up to whole blocks.
    int64_t slots =
        target.slotsPerBlock(static_cast<int32_t>(threads.getInt()));
    Value three = arith::ConstantIndexOp::create(rewriter, loc, slots - 1);
    Value four = arith::ConstantIndexOp::create(rewriter, loc, slots);
    Value warp =
        arith::ConstantIndexOp::create(rewriter, loc, target.subgroupWidth);
    Value warpTaskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getWarpTaskCount());
    Value ctaTaskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getCtaTaskCount());
    Value roundedWarpTaskCount =
        arith::AddIOp::create(rewriter, loc, warpTaskCount, three);
    Value warpBlockCount =
        arith::DivUIOp::create(rewriter, loc, roundedWarpTaskCount, four);
    Value isWarpBlock = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::ult, taskIndex, warpBlockCount);

    SegmentSite site{adaptor.getValues(),
                     adaptor.getOffsets(),
                     adaptor.getValueCount(),
                     adaptor.getSegmentCount(),
                     adaptor.getOutput(),
                     zero,
                     one};
    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, isWarpBlock,
        [&](OpBuilder &warpBlock, Location warpLoc) {
          Value physicalWarp =
              arith::DivUIOp::create(warpBlock, warpLoc, threadId, warp);
          Value lane =
              arith::RemUIOp::create(warpBlock, warpLoc, threadId, warp);
          Value firstTask =
              arith::MulIOp::create(warpBlock, warpLoc, taskIndex, four);
          Value warpTaskId = arith::AddIOp::create(warpBlock, warpLoc,
                                                   firstTask, physicalWarp);
          Value inRange = arith::CmpIOp::create(warpBlock, warpLoc,
                                                arith::CmpIPredicate::ult,
                                                warpTaskId, warpTaskCount);
          scf::IfOp::create(warpBlock, warpLoc, inRange,
                            [&](OpBuilder &task, Location taskLoc) {
                              converted &= convertSegmentTask(
                                  rewriter, task, taskLoc, tasks.getWarp(),
                                  site, ids, warpTaskId, lane, warp);
                              scf::YieldOp::create(task, taskLoc);
                            });
          scf::YieldOp::create(warpBlock, warpLoc);
        },
        [&](OpBuilder &ctaBlock, Location ctaLoc) {
          Value ctaTaskId = arith::SubIOp::create(ctaBlock, ctaLoc, taskIndex,
                                                  warpBlockCount);
          Value inRange =
              arith::CmpIOp::create(ctaBlock, ctaLoc, arith::CmpIPredicate::ult,
                                    ctaTaskId, ctaTaskCount);
          scf::IfOp::create(ctaBlock, ctaLoc, inRange,
                            [&](OpBuilder &task, Location taskLoc) {
                              // The block tasks follow the warp tasks in the
                              // task buffer.
                              Value mixedTaskId = arith::AddIOp::create(
                                  task, taskLoc, warpTaskCount, ctaTaskId);
                              converted &= convertSegmentTask(
                                  rewriter, task, taskLoc, tasks.getCta(), site,
                                  ids, mixedTaskId, threadId, block);
                              scf::YieldOp::create(task, taskLoc);
                            });
          scf::YieldOp::create(ctaBlock, ctaLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

private:
  const TargetDescription &target;
};

/// Run the reduction of a task region on a range of a buffer, at the
/// insertion point of `body`, and return the scalar the region yields.
///
/// The region argument is replaced by the four values of `range`, and the
/// consumers are legalized while the task operation is still their parent,
/// so the reduction pattern takes the binding from its operand and the
/// policy from the operation it sits in. Returns a null value when a
/// consumer could not be legalized.
///
/// `extent` is given for a region that takes the extent of its segment: it
/// replaces the second argument, and the scalar epilogue of the region runs
/// after the reduction.
Value convertRangeTask(ConversionPatternRewriter &rewriter, OpBuilder &body,
                       Region &region, const SegmentBinding &range,
                       Value extent = Value()) {
  Block &consumersBlock = region.front();
  rewriter.replaceAllUsesWith(
      consumersBlock.getArgument(0),
      ValueRange{range.base, range.first, range.end, range.stride});
  if (extent)
    rewriter.replaceAllUsesWith(consumersBlock.getArgument(1), extent);
  auto yield = cast<swage_plan::YieldOp>(consumersBlock.getTerminator());
  SmallVector<Operation *> consumers = operationsOf(consumersBlock);
  consumers.pop_back();
  if (failed(legalizeInPlace(rewriter, consumers)))
    return Value();
  moveConvertedOperations<ReduceOp, swage_plan::YieldOp>(
      rewriter, consumersBlock, body.getInsertionBlock());
  return rewriter.getRemappedValue(yield.getValue());
}

/// The range of partial task `taskIndex`, loaded from its record in
/// `ranges` and clamped to `valueCount`, the values it indexes.
std::pair<Value, Value> loadPartialRange(OpBuilder &builder, Location loc,
                                         Value ranges, Value taskIndex,
                                         Value valueCount) {
  using namespace swage_plan::partial_record;
  Value fields = arith::ConstantIndexOp::create(builder, loc, Words);
  Value recordBase = arith::MulIOp::create(builder, loc, taskIndex, fields);
  Value beginWord = loadRecordField(builder, loc, ranges, recordBase, Begin);
  Value endWord = loadRecordField(builder, loc, ranges, recordBase, End);
  return clampRange(builder, loc, beginWord, endWord, valueCount);
}

/// One block of threads per chunk of a split segment: the kernel prelude,
/// the guard on the task index, the range of the chunk loaded from its
/// record and clamped to the value count, the reduction of the region, and
/// the store of its result in the scratch slot of the task.
///
/// Over rank-two values the chunk is a range of rows, the kernel is the
/// row-stripe tile, and task `p` stores the result of column `c` at
/// `scratch[p * D + c]`.
class PartialTasksPattern : public OpConversionPattern<PartialTasksOp> {
public:
  PartialTasksPattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(PartialTasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(tasks->getParentOp());
    if (!threads)
      return rewriter.notifyMatchFailure(tasks, "not in a plan function");
    if (adaptor.getFeatureCount())
      return convertRowPartials(tasks, adaptor, rewriter, threads.getInt());
    Location loc = tasks.getLoc();

    Value taskIndex = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value block =
        arith::ConstantIndexOp::create(rewriter, loc, threads.getInt());
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getPartialCount());
    Value inRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, taskIndex, taskCount);

    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
          auto [begin, end] =
              loadPartialRange(body, bodyLoc, adaptor.getRanges(), taskIndex,
                               adaptor.getValueCount());
          Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
          Value total =
              convertRangeTask(rewriter, body, tasks.getBody(),
                               {adaptor.getValues(), first, end, block});
          converted = static_cast<bool>(total);
          if (converted)
            emitLeaderStore(body, bodyLoc, total, adaptor.getScratch(),
                            taskIndex, threadId, zero, Value());
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

private:
  /// The row-stripe tile of the partial tasks: for each item of the block,
  /// the rows of the chunk of its task after their clamp to the row count,
  /// the consumers on the column group of the item, and the store of each
  /// column of the group in the scratch row of the task.
  LogicalResult convertRowPartials(PartialTasksOp tasks, OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter,
                                   int64_t threads) const {
    Value exchange = exchangeOf(adaptor.getValues());
    if (!exchange)
      return rewriter.notifyMatchFailure(tasks, "kernel has no exchange");
    Location loc = tasks.getLoc();
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    RowTile tile = emitRowTile(rewriter, loc, target, adaptor.getFeatureCount(),
                               threadId, threads, exchange);
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getPartialCount());
    bool converted = true;
    emitItemLoop(
        rewriter, loc, taskCount, tile.groups,
        [&](OpBuilder &body, Location bodyLoc, Value task, Value group) {
          auto [begin, end] =
              loadPartialRange(body, bodyLoc, adaptor.getRanges(), task,
                               adaptor.getValueCount());
          BoundRows rows = bindRows(body, bodyLoc, tile, adaptor.getValues(),
                                    begin, end, group);
          Value total = convertRowRegion(rewriter, body, tasks.getBody(), rows,
                                         Value(), converted);
          if (total)
            emitRowStore(body, bodyLoc, tile, rows, total, adaptor.getScratch(),
                         task, Value(), zero);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

  const TargetDescription &target;
};

/// The extent of a split segment, as an index: the end of range record
/// `partialEnd - 1` minus the begin of range record `partialBegin`, or zero
/// when the clamped range of partials `[partialBegin, partialEnd)` is empty.
/// The branch holds no barrier, so every thread may take it on its own.
Value emitSplitExtent(OpBuilder &builder, Location loc, Value ranges,
                      Value partialBegin, Value partialEnd, Value zero) {
  namespace record = swage_plan::partial_record;
  Value hasPartials = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, partialBegin, partialEnd);
  auto extent =
      scf::IfOp::create(builder, loc, TypeRange{builder.getIndexType()},
                        hasPartials, /*withElseRegion=*/true);
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&extent.getThenRegion().front());
  Value fields = arith::ConstantIndexOp::create(builder, loc, record::Words);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value firstBase = arith::MulIOp::create(builder, loc, partialBegin, fields);
  Value last = arith::SubIOp::create(builder, loc, partialEnd, one);
  Value lastBase = arith::MulIOp::create(builder, loc, last, fields);
  Value beginWord =
      loadRecordField(builder, loc, ranges, firstBase, record::Begin);
  Value endWord = loadRecordField(builder, loc, ranges, lastBase, record::End);
  Value extentWord = arith::SubIOp::create(builder, loc, endWord, beginWord);
  scf::YieldOp::create(builder, loc,
                       arith::IndexCastOp::create(
                           builder, loc, builder.getIndexType(), extentWord)
                           .getResult());
  builder.setInsertionPointToStart(&extent.getElseRegion().front());
  scf::YieldOp::create(builder, loc, zero);
  return extent.getResult(0);
}

/// The merge record of task `taskIndex`: the segment it names, whether
/// that segment is below `segmentCount`, and its range of partials clamped
/// to `partialCount`, the scratch slots or rows it indexes.
struct MergeRecord {
  Value segment;
  Value segmentInRange;
  Value begin;
  Value end;
};

MergeRecord loadMergeRecord(OpBuilder &builder, Location loc, Value records,
                            Value taskIndex, Value segmentCount,
                            Value partialCount) {
  using namespace swage_plan::merge_record;
  Value fields = arith::ConstantIndexOp::create(builder, loc, Words);
  Value recordBase = arith::MulIOp::create(builder, loc, taskIndex, fields);
  Value segmentWord =
      loadRecordField(builder, loc, records, recordBase, Segment);
  Value segmentInRange =
      isLoadedIndexInRange(builder, loc, segmentWord, segmentCount);
  Value segment = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), segmentWord);
  Value beginWord =
      loadRecordField(builder, loc, records, recordBase, PartialBegin);
  Value endWord =
      loadRecordField(builder, loc, records, recordBase, PartialEnd);
  auto [begin, end] =
      clampRange(builder, loc, beginWord, endWord, partialCount);
  return {segment, segmentInRange, begin, end};
}

/// One block of threads per split segment: the kernel prelude, the guard on
/// the task index, the merge record of the segment, the reduction of the
/// region over the scratch slots the record names, and the store of its
/// result at the segment the record names.
///
/// The segment comes from device memory, so it is compared with the segment
/// count, and one that fails stores nothing. The reduction itself stays
/// unconditional, which keeps its block-wide combination under the
/// block-uniform guard alone.
///
/// With `ranges`, the region takes the extent of the split segment. The
/// chunks of one segment are consecutive range records, so the extent is
/// the end of the last record minus the begin of the first. The two records
/// are addressed through the clamped range of partials, which keeps both
/// loads inside the `partial_count` records, and an empty range reads none
/// and has the extent zero. The words themselves are data: they reach a
/// division and never an address.
///
/// Over rank-two values scratch holds one row per partial task, the record
/// names rows of scratch and the extent is a number of rows, the kernel is
/// the row-stripe tile, and the result of column `c` is stored at
/// `output[segment * D + c]`.
class MergeTasksPattern : public OpConversionPattern<MergeTasksOp> {
public:
  MergeTasksPattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(MergeTasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(tasks->getParentOp());
    if (!threads)
      return rewriter.notifyMatchFailure(tasks, "not in a plan function");
    if (adaptor.getFeatureCount())
      return convertRowMerges(tasks, adaptor, rewriter, threads.getInt());
    Location loc = tasks.getLoc();

    Value taskIndex = gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value block =
        arith::ConstantIndexOp::create(rewriter, loc, threads.getInt());
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getMergeCount());
    Value inRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, taskIndex, taskCount);

    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
          MergeRecord record = loadMergeRecord(
              body, bodyLoc, adaptor.getMerges(), taskIndex,
              adaptor.getSegmentCount(), adaptor.getPartialCount());
          Value extent;
          if (Value ranges = adaptor.getRanges())
            extent = emitSplitExtent(body, bodyLoc, ranges, record.begin,
                                     record.end, zero);
          Value first =
              arith::AddIOp::create(body, bodyLoc, record.begin, threadId);
          Value total = convertRangeTask(
              rewriter, body, tasks.getBody(),
              {adaptor.getScratch(), first, record.end, block}, extent);
          converted = static_cast<bool>(total);
          if (converted)
            emitLeaderStore(body, bodyLoc, total, adaptor.getOutput(),
                            record.segment, threadId, zero,
                            record.segmentInRange);
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

private:
  /// The row-stripe tile of the merge tasks: for each item of the block,
  /// the merge record of its task, the rows of scratch the record names
  /// after their clamp to the partial count, the consumers on the column
  /// group of the item, and the store of each column of the group in the
  /// row of the segment, when that segment is below the segment count.
  LogicalResult convertRowMerges(MergeTasksOp tasks, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter,
                                 int64_t threads) const {
    Value exchange = exchangeOf(adaptor.getScratch());
    if (!exchange)
      return rewriter.notifyMatchFailure(tasks, "kernel has no exchange");
    Location loc = tasks.getLoc();
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    RowTile tile = emitRowTile(rewriter, loc, target, adaptor.getFeatureCount(),
                               threadId, threads, exchange);
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getMergeCount());
    bool converted = true;
    emitItemLoop(
        rewriter, loc, taskCount, tile.groups,
        [&](OpBuilder &body, Location bodyLoc, Value task, Value group) {
          MergeRecord record = loadMergeRecord(
              body, bodyLoc, adaptor.getMerges(), task,
              adaptor.getSegmentCount(), adaptor.getPartialCount());
          Value extent;
          if (Value ranges = adaptor.getRanges())
            extent = emitSplitExtent(body, bodyLoc, ranges, record.begin,
                                     record.end, zero);
          BoundRows rows = bindRows(body, bodyLoc, tile, adaptor.getScratch(),
                                    record.begin, record.end, group);
          Value total = convertRowRegion(rewriter, body, tasks.getBody(), rows,
                                         extent, converted);
          if (total)
            emitRowStore(body, bodyLoc, tile, rows, total, adaptor.getOutput(),
                         record.segment, record.segmentInRange, zero);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

  const TargetDescription &target;
};

/// The persistent queue kernel: resident blocks that drain the block queue,
/// then the partial queue with the merges it completes, then the warp
/// queue. A claim adds to a counter in device memory and tells every thread
/// of the claiming group which tasks it took.
///
/// The barriers and fences are the reason this kernel is one pattern: each
/// one sits where the claims and stores around it need it, so the order of
/// what is emitted here is the correctness argument of the kernel.
class PersistentTasksPattern : public OpConversionPattern<PersistentTasksOp> {
public:
  PersistentTasksPattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(PersistentTasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(tasks->getParentOp());
    if (!threads)
      return rewriter.notifyMatchFailure(tasks, "not in a plan function");
    // The function pattern gave the kernel its claim slots. The operands of
    // a task operation are parameters of that kernel, which lead to it.
    auto parameter = dyn_cast<BlockArgument>(adaptor.getCounters());
    auto kernel =
        parameter
            ? dyn_cast<gpu::GPUFuncOp>(parameter.getOwner()->getParentOp())
            : gpu::GPUFuncOp();
    if (!kernel || kernel.getWorkgroupAttributions().empty())
      return rewriter.notifyMatchFailure(tasks, "kernel has no claim slots");
    Value claimSlots = kernel.getWorkgroupAttributions().front();

    using namespace swage_plan::persistent_counter;
    Location loc = tasks.getLoc();
    Type pointer = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type i32 = rewriter.getI32Type();
    Value counters = adaptor.getCounters();
    Value valueCount = adaptor.getValueCount();
    Value segmentCount = adaptor.getSegmentCount();
    Value partialCount = adaptor.getPartialCount();
    Value mergeCount = adaptor.getMergeCount();

    // A resident block claims its tasks and never reads its block index. The
    // operation stays because the kernel text is pinned by its digest.
    gpu::BlockIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value threadId = gpu::ThreadIdOp::create(rewriter, loc, gpu::Dimension::x);
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value block =
        arith::ConstantIndexOp::create(rewriter, loc, threads.getInt());
    Value zeroI32 = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
    Value oneI32 = arith::ConstantIntOp::create(rewriter, loc, 1, 32);
    // The batch a block claims from the partial queue, and the batch a
    // subgroup claims from the warp queue.
    Value partialBatch = arith::ConstantIntOp::create(
        rewriter, loc, target.persistentPartialClaim, 32);
    Value warpBatch = arith::ConstantIntOp::create(
        rewriter, loc, target.persistentWarpClaim, 32);
    Value firstThread = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, threadId, zero);

    SegmentSite site{adaptor.getValues(),
                     adaptor.getOffsets(),
                     valueCount,
                     segmentCount,
                     adaptor.getOutput(),
                     zero,
                     one};
    bool converted = true;

    // Take `batch` tasks from the queue of `counter`: the leader adds the
    // batch to the counter, and every thread of its group receives the value
    // the counter had, which is the first task of the batch. A subgroup
    // hears its leader through a shuffle; a block hears it through the first
    // claim slot and a barrier.
    auto claim = [&](OpBuilder &body, Location claimLoc, Slot counter,
                     Value leader, bool subgroup, Value batch) {
      Value counterOffset =
          arith::ConstantIntOp::create(body, claimLoc, counter, 64);
      Value counterAddress = LLVM::GEPOp::create(body, claimLoc, pointer, i32,
                                                 counters, counterOffset);
      auto leaderClaim = scf::IfOp::create(body, claimLoc, TypeRange{i32},
                                           leader, /*withElseRegion=*/true);
      {
        OpBuilder::InsertionGuard guard(body);
        body.setInsertionPointToStart(&leaderClaim.getThenRegion().front());
        Value claimed = LLVM::AtomicRMWOp::create(
            body, claimLoc, LLVM::AtomicBinOp::add, counterAddress, batch,
            LLVM::AtomicOrdering::monotonic);
        scf::YieldOp::create(body, claimLoc, claimed);
        body.setInsertionPointToStart(&leaderClaim.getElseRegion().front());
        scf::YieldOp::create(body, claimLoc, zeroI32);
      }
      if (subgroup) {
        auto shuffled =
            gpu::ShuffleOp::create(body, claimLoc, leaderClaim.getResult(0), 0,
                                   target.subgroupWidth, gpu::ShuffleMode::IDX);
        return shuffled.getShuffleResult();
      }
      scf::IfOp::create(
          body, claimLoc, leader, [&](OpBuilder &store, Location storeLoc) {
            memref::StoreOp::create(store, storeLoc, leaderClaim.getResult(0),
                                    claimSlots, zero);
            scf::YieldOp::create(store, storeLoc);
          });
      gpu::BarrierOp::create(body, claimLoc);
      return Value(
          memref::LoadOp::create(body, claimLoc, claimSlots, ValueRange{zero}));
    };

    // Run `claimed` on each claim, starting with `first`, while the claim
    // names a task of the queue, which holds `count` tasks. `claimed` runs
    // the tasks of the claim and yields the next claim.
    using ClaimedFn = function_ref<void(OpBuilder &, Location, Value)>;
    auto drain = [&](Value first, Value count, ClaimedFn claimed) {
      scf::WhileOp::create(
          rewriter, loc, TypeRange{i32}, ValueRange{first},
          [&](OpBuilder &before, Location beforeLoc, ValueRange claims) {
            Value hasTask = arith::CmpIOp::create(before, beforeLoc,
                                                  arith::CmpIPredicate::ult,
                                                  claims.front(), count);
            scf::ConditionOp::create(before, beforeLoc, hasTask, claims);
          },
          [&](OpBuilder &after, Location afterLoc, ValueRange claims) {
            claimed(after, afterLoc, claims.front());
          });
    };

    // Run `task` on each task of the claim that starts at `first` and takes
    // `batch` tasks, as far as the queue holds them.
    auto forEachOfBatch = [&](OpBuilder &body, Location bodyLoc, Value first,
                              Value batch, Value count, ClaimedFn task) {
      Value batchEnd = arith::AddIOp::create(body, bodyLoc, first, batch);
      Value boundedBatchEnd =
          arith::MinUIOp::create(body, bodyLoc, batchEnd, count);
      Value firstIndex =
          arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(), first);
      Value endIndex = arith::IndexCastOp::create(
          body, bodyLoc, body.getIndexType(), boundedBatchEnd);
      scf::ForOp::create(
          body, bodyLoc, firstIndex, endIndex, one, ValueRange(),
          [&](OpBuilder &loop, Location loopLoc, Value index, ValueRange) {
            task(loop, loopLoc, index);
            scf::YieldOp::create(loop, loopLoc);
          });
    };

    // The leader counts the completion of partial task `partialIndex` for
    // its merge and writes the merge that became ready, or -1, to the second
    // claim slot. Only the leader reads the dependency metadata, and the
    // barrier that follows makes its decision uniform across the block
    // without a reload of the merge record in every lane.
    //
    // The merge ID addresses the completion counter and the merge record, so
    // the merge count bounds it first. A partial whose merge ID is out of
    // range updates no counter, reads no record, and publishes -1. This
    // branch runs in the leader alone and holds no barrier.
    Value completionSlot = one;
    auto publishCompletion = [&](OpBuilder &publish, Location publishLoc,
                                 Value partialIndex) {
      using namespace swage_plan::merge_record;
      Value mergeIdI32 = loadTaskWord(publish, publishLoc,
                                      adaptor.getMergeIds(), partialIndex);
      Value mergeInRange =
          isLoadedIndexInRange(publish, publishLoc, mergeIdI32, mergeCount);
      auto published =
          scf::IfOp::create(publish, publishLoc, TypeRange{i32}, mergeInRange,
                            /*withElseRegion=*/true);
      publish.setInsertionPointToStart(&published.getThenRegion().front());
      Value mergeId = arith::IndexCastOp::create(
          publish, publishLoc, publish.getIndexType(), mergeIdI32);
      Value completionBase =
          arith::ConstantIndexOp::create(publish, publishLoc, FirstCompletion);
      Value completionIndex =
          arith::AddIOp::create(publish, publishLoc, completionBase, mergeId);
      Value completionIndex64 = arith::IndexCastOp::create(
          publish, publishLoc, publish.getI64Type(), completionIndex);
      Value completionAddress = LLVM::GEPOp::create(
          publish, publishLoc, pointer, i32, counters, completionIndex64);

      Value fields = arith::ConstantIndexOp::create(publish, publishLoc, Words);
      Value recordBase =
          arith::MulIOp::create(publish, publishLoc, mergeId, fields);
      Value beginIndex =
          arith::AddIOp::create(publish, publishLoc, recordBase, one);
      Value endIndex =
          arith::AddIOp::create(publish, publishLoc, beginIndex, one);
      auto loadIndex = [&](Value wordIndex) {
        Value word =
            loadTaskWord(publish, publishLoc, adaptor.getMerges(), wordIndex);
        return Value(arith::IndexCastOp::create(publish, publishLoc,
                                                publish.getIndexType(), word));
      };
      Value begin = loadIndex(beginIndex);
      Value end = loadIndex(endIndex);
      Value expected = arith::SubIOp::create(publish, publishLoc, end, begin);
      Value expectedI32 = arith::IndexCastOp::create(
          publish, publishLoc, publish.getI32Type(), expected);

      // The device lowers this atomic to a form that does not order the
      // scratch store of the partial before it, so a fence makes scratch
      // visible before the completion is.
      target.emitDeviceFence(publish, publishLoc);
      Value previous = LLVM::AtomicRMWOp::create(
          publish, publishLoc, LLVM::AtomicBinOp::add, completionAddress,
          oneI32, LLVM::AtomicOrdering::acq_rel);
      Value completed =
          arith::AddIOp::create(publish, publishLoc, previous, oneI32);
      Value isLast =
          arith::CmpIOp::create(publish, publishLoc, arith::CmpIPredicate::eq,
                                completed, expectedI32);
      auto readyMerge = scf::IfOp::create(publish, publishLoc, TypeRange{i32},
                                          isLast, /*withElseRegion=*/true);
      publish.setInsertionPointToStart(&readyMerge.getThenRegion().front());
      scf::YieldOp::create(publish, publishLoc, mergeIdI32);
      publish.setInsertionPointToStart(&readyMerge.getElseRegion().front());
      Value noMerge = arith::ConstantIntOp::create(publish, publishLoc, -1, 32);
      scf::YieldOp::create(publish, publishLoc, noMerge);
      publish.setInsertionPointAfter(readyMerge);
      scf::YieldOp::create(publish, publishLoc, readyMerge.getResult(0));
      publish.setInsertionPointToStart(&published.getElseRegion().front());
      Value skippedMerge =
          arith::ConstantIntOp::create(publish, publishLoc, -1, 32);
      scf::YieldOp::create(publish, publishLoc, skippedMerge);
      publish.setInsertionPointAfter(published);
      memref::StoreOp::create(publish, publishLoc, published.getResult(0),
                              claimSlots, completionSlot);
    };

    // Reduce the scratch slots of the merge every thread read from the
    // second claim slot, and store the result at the segment of its record.
    //
    // The merge ID is in range here: the leader published it only after the
    // bound above. The output segment it names is loaded from the record, so
    // the segment count bounds it before the store. The reduction itself
    // stays unconditional, which keeps its block-wide combination under the
    // block-uniform guard alone.
    auto mergePartials = [&](OpBuilder &merge, Location mergeLoc,
                             Value readyMergeI32) {
      using namespace swage_plan::merge_record;
      // Pair with every partial publisher before any lane reads scratch.
      target.emitDeviceFence(merge, mergeLoc);
      Value mergeId = arith::IndexCastOp::create(
          merge, mergeLoc, merge.getIndexType(), readyMergeI32);
      Value fields = arith::ConstantIndexOp::create(merge, mergeLoc, Words);
      Value recordBase =
          arith::MulIOp::create(merge, mergeLoc, mergeId, fields);
      Value beginIndex =
          arith::AddIOp::create(merge, mergeLoc, recordBase, one);
      Value endIndex = arith::AddIOp::create(merge, mergeLoc, beginIndex, one);
      Value segmentWord =
          loadTaskWord(merge, mergeLoc, adaptor.getMerges(), recordBase);
      Value segmentInRange =
          isLoadedIndexInRange(merge, mergeLoc, segmentWord, segmentCount);
      Value segment = arith::IndexCastOp::create(
          merge, mergeLoc, merge.getIndexType(), segmentWord);
      Value beginWord =
          loadTaskWord(merge, mergeLoc, adaptor.getMerges(), beginIndex);
      Value endWord =
          loadTaskWord(merge, mergeLoc, adaptor.getMerges(), endIndex);
      // The range indexes scratch, so the partial count bounds it.
      Value begin;
      Value end;
      std::tie(begin, end) =
          clampRange(merge, mergeLoc, beginWord, endWord, partialCount);
      Value first = arith::AddIOp::create(merge, mergeLoc, begin, threadId);
      Value total = convertRangeTask(rewriter, merge, tasks.getMerge(),
                                     {adaptor.getScratch(), first, end, block});
      if (!total)
        return false;
      Value mayStore =
          arith::AndIOp::create(merge, mergeLoc, firstThread, segmentInRange);
      scf::IfOp::create(
          merge, mergeLoc, mayStore, [&](OpBuilder &store, Location storeLoc) {
            Value segment64 = arith::IndexCastOp::create(
                store, storeLoc, store.getI64Type(), segment);
            Value address =
                LLVM::GEPOp::create(store, storeLoc, pointer, total.getType(),
                                    adaptor.getOutput(), segment64);
            LLVM::StoreOp::create(store, storeLoc, total, address);
            scf::YieldOp::create(store, storeLoc);
          });
      return true;
    };

    // Reduce the chunk of partial task `partialIndex` into its scratch slot,
    // count its completion, and merge when it completed the last partial
    // task of a merge.
    auto runPartial = [&](OpBuilder &body, Location bodyLoc,
                          Value partialIndex) {
      using namespace swage_plan::partial_record;
      Value fields = arith::ConstantIndexOp::create(body, bodyLoc, Words);
      Value recordBase =
          arith::MulIOp::create(body, bodyLoc, partialIndex, fields);
      Value endIndex = arith::AddIOp::create(body, bodyLoc, recordBase, one);
      Value beginWord =
          loadTaskWord(body, bodyLoc, adaptor.getRanges(), recordBase);
      Value endWord =
          loadTaskWord(body, bodyLoc, adaptor.getRanges(), endIndex);
      // The range indexes the values, so the value count bounds it.
      Value begin;
      Value end;
      std::tie(begin, end) =
          clampRange(body, bodyLoc, beginWord, endWord, valueCount);
      Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
      Value total = convertRangeTask(rewriter, body, tasks.getPartial(),
                                     {adaptor.getValues(), first, end, block});
      if (!total) {
        converted = false;
        return;
      }
      scf::IfOp::create(
          body, bodyLoc, firstThread, [&](OpBuilder &store, Location storeLoc) {
            Value partialIndex64 = arith::IndexCastOp::create(
                store, storeLoc, store.getI64Type(), partialIndex);
            Value address =
                LLVM::GEPOp::create(store, storeLoc, pointer, total.getType(),
                                    adaptor.getScratch(), partialIndex64);
            LLVM::StoreOp::create(store, storeLoc, total, address);
            scf::YieldOp::create(store, storeLoc);
          });
      gpu::BarrierOp::create(body, bodyLoc);

      scf::IfOp::create(body, bodyLoc, firstThread,
                        [&](OpBuilder &publish, Location publishLoc) {
                          publishCompletion(publish, publishLoc, partialIndex);
                          scf::YieldOp::create(publish, publishLoc);
                        });
      gpu::BarrierOp::create(body, bodyLoc);
      Value readyMergeI32 = memref::LoadOp::create(body, bodyLoc, claimSlots,
                                                   ValueRange{completionSlot});
      Value hasReadyMerge = arith::CmpIOp::create(
          body, bodyLoc, arith::CmpIPredicate::sge, readyMergeI32, zeroI32);
      scf::IfOp::create(body, bodyLoc, hasReadyMerge,
                        [&](OpBuilder &merge, Location mergeLoc) {
                          converted &=
                              mergePartials(merge, mergeLoc, readyMergeI32);
                          scf::YieldOp::create(merge, mergeLoc);
                        });
    };

    // Block tasks are claimed first so a long tail can overlap the short
    // tasks that follow. A block that sees the queue empty moves on while
    // another block may still be reducing the last long segment.
    Value firstCta = claim(rewriter, loc, CtaClaim, firstThread, false, oneI32);
    drain(firstCta, adaptor.getCtaTaskCount(),
          [&](OpBuilder &body, Location bodyLoc, Value claimed) {
            Value taskIndex = arith::IndexCastOp::create(
                body, bodyLoc, body.getIndexType(), claimed);
            converted &= convertSegmentTask(
                rewriter, body, bodyLoc, tasks.getCta(), site,
                adaptor.getCtaIds(), taskIndex, threadId, block);
            gpu::BarrierOp::create(body, bodyLoc);
            Value next =
                claim(body, bodyLoc, CtaClaim, firstThread, false, oneI32);
            scf::YieldOp::create(body, bodyLoc, next);
          });
    // The claim that ended the loop is still read from the first claim slot.
    // The barrier keeps the first claim of the partial queue from writing
    // that slot until every thread has read it.
    gpu::BarrierOp::create(rewriter, loc);

    // Partial tasks share one queue. Each writes its own scratch slot and
    // then counts its completion with an acquire-release atomic. The block
    // that counts the last completion of a merge runs the merge and its
    // store, so no block waits for another.
    Value firstPartial =
        claim(rewriter, loc, PartialClaim, firstThread, false, partialBatch);
    drain(firstPartial, partialCount,
          [&](OpBuilder &body, Location bodyLoc, Value claimed) {
            forEachOfBatch(body, bodyLoc, claimed, partialBatch, partialCount,
                           runPartial);
            Value next = claim(body, bodyLoc, PartialClaim, firstThread, false,
                               partialBatch);
            scf::YieldOp::create(body, bodyLoc, next);
          });

    // Each subgroup drains the warp queue on its own. Its first lane claims
    // and the claim reaches the other lanes through a shuffle, so the
    // subgroups of a block need not stay in step.
    Value warp =
        arith::ConstantIndexOp::create(rewriter, loc, target.subgroupWidth);
    Value lane = arith::RemUIOp::create(rewriter, loc, threadId, warp);
    Value firstLane = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::eq, lane, zero);
    Value warpTaskCount = adaptor.getWarpTaskCount();
    Value firstWarp =
        claim(rewriter, loc, WarpClaim, firstLane, true, warpBatch);
    drain(firstWarp, warpTaskCount,
          [&](OpBuilder &body, Location bodyLoc, Value claimed) {
            forEachOfBatch(
                body, bodyLoc, claimed, warpBatch, warpTaskCount,
                [&](OpBuilder &task, Location taskLoc, Value taskIndex) {
                  converted &= convertSegmentTask(
                      rewriter, task, taskLoc, tasks.getWarp(), site,
                      adaptor.getWarpIds(), taskIndex, lane, warp);
                });
            Value next =
                claim(body, bodyLoc, WarpClaim, firstLane, true, warpBatch);
            scf::YieldOp::create(body, bodyLoc, next);
          });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }

private:
  const TargetDescription &target;
};

/// What the patterns rely on and the dialect verifier does not promise,
/// checked before anything is changed. The planner produces only plan
/// functions that pass; this is for plan IR that was written by hand. A
/// failed pattern could only say "failed to legalize", so each rule is
/// reported here, with the function and the rule.
LogicalResult verifyPlanFunction(ModuleOp module, func::FuncOp function,
                                 const TargetDescription &target) {
  StringRef name = SwagePlanDialect::getBlockThreadsAttrName();
  int64_t threads = blockThreadsOf(function).getInt();
  if (!target.admitsBlockThreads(threads))
    return function.emitError()
           << name << " must be a launch width the target admits, from 1 to "
           << target.maxBlockThreads
           << " threads with a power-of-two subgroup count, got " << threads;

  Operation *task = &function.getBody().front().front();
  if (auto persistent = dyn_cast<PersistentTasksOp>(task)) {
    // The claim batches and the number of resident blocks the host launches
    // are chosen for one width.
    if (threads != target.persistentBlockThreads)
      return persistent.emitError()
             << "a persistent task block has the persistent launch width of "
                "the target, "
             << target.persistentBlockThreads << " threads, got " << name
             << " = " << threads;
    Type values = persistent.getValues().getType();
    Type scratch = persistent.getScratch().getType();
    // The queue kernel has run with f32 values only.
    if (!cast<MemRefType>(values).getElementType().isF32())
      return persistent.emitError()
             << "a persistent task operation takes f32 values, got "
             << cast<MemRefType>(values).getElementType();
    Type offsets = persistent.getOffsets().getType();
    const std::pair<Region *, Type> regions[] = {
        {&persistent.getCta(), values},
        {&persistent.getPartial(), values},
        {&persistent.getMerge(), scratch},
        {&persistent.getWarp(), values}};
    for (auto [region, buffer] : regions) {
      if (failed(verifyTaskConsumers(persistent, buffer, offsets,
                                     region->front())))
        return failure();
      // The queue kernel is lowered for the one program the planner admits
      // for it. Every region runs that program, so every region is checked.
      Block &body = region->front();
      auto reduction = llvm::hasSingleElement(body.without_terminator())
                           ? dyn_cast<ReduceOp>(body.front())
                           : ReduceOp();
      auto isIdentity = [](Block &element) {
        return element.without_terminator().empty() &&
               cast<YieldOp>(element.getTerminator()).getValue() ==
                   element.getArgument(0);
      };
      if (!reduction || reduction.getKind() != ReductionKind::Sum ||
          !reduction.getCaptures().empty() ||
          !isIdentity(reduction.getBody().front()) ||
          cast<swage_plan::YieldOp>(body.getTerminator()).getValue() !=
              reduction.getResult())
        return persistent.emitError(
            "every region of a persistent task operation holds one "
            "capture-free kind<sum> reduction with an identity region and "
            "yields its result; the queue kernel is lowered for that program "
            "only");
    }
    return verifyKernelSymbols(module, function, "");
  }
  if (auto fused = dyn_cast<FusedTasksOp>(task)) {
    // Each subgroup of a warp block runs one task, so a block is a whole
    // number of subgroups.
    if (threads % target.subgroupWidth != 0)
      return fused.emitError()
             << "a fused task block is a whole number of subgroups of "
             << target.subgroupWidth << " threads, got " << name << " = "
             << threads;
    for (Region *region : {&fused.getWarp(), &fused.getCta()})
      if (failed(verifyTaskConsumers(fused, fused.getValues().getType(),
                                     fused.getOffsets().getType(),
                                     region->front())))
        return failure();
    return verifyKernelSymbols(module, function, "");
  }
  // Every lane of a row-stripe tile owns one row stripe of one column.
  auto verifyRowWidth = [&](Operation *rows) -> LogicalResult {
    if (!exchangeElementOf(rows) || threads % target.subgroupWidth == 0)
      return success();
    return rows->emitError()
           << "a row-stripe task of rank-two values runs whole subgroups of "
           << target.subgroupWidth << " threads, so " << name
           << " must be a multiple of " << target.subgroupWidth << ", got "
           << threads;
  };
  if (auto merge = dyn_cast<MergeTasksOp>(task)) {
    if (failed(verifyTaskConsumers(merge, merge.getScratch().getType(),
                                   merge.getMerges().getType(),
                                   merge.getBody().front())) ||
        failed(verifyRowWidth(merge)))
      return failure();
    return verifyKernelSymbols(module, function, "");
  }
  if (auto partial = dyn_cast<PartialTasksOp>(task)) {
    if (failed(verifyTaskConsumers(partial, partial.getValues().getType(),
                                   partial.getRanges().getType(),
                                   partial.getBody().front())) ||
        failed(verifyRowWidth(partial)))
      return failure();
    return verifyKernelSymbols(module, function, "");
  }
  auto tasks = cast<TasksOp>(task);
  if (failed(verifyTaskConsumers(tasks, tasks.getValues().getType(),
                                 tasks.getOffsets().getType(),
                                 tasks.getBody().front())))
    return failure();
  // A warp task reduces within one subgroup, so its block is one subgroup.
  if (tasks.getPolicy() == TaskPolicy::Warp && threads != target.subgroupWidth)
    return tasks.emitError()
           << "policy<warp> requires " << name << " to be the subgroup width, "
           << target.subgroupWidth << ", got " << threads;
  if (failed(verifyRowWidth(tasks)))
    return failure();
  return verifyKernelSymbols(module, function, "");
}

class SwagePlanToGPUPass
    : public PassWrapper<SwagePlanToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwagePlanToGPUPass)

  SwagePlanToGPUPass() = default;
  explicit SwagePlanToGPUPass(const TargetDescription &target)
      : target(&target) {}

  StringRef getArgument() const final { return "swage-plan-to-gpu"; }
  StringRef getDescription() const final {
    return "Convert every plan function to a GPU kernel module";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                memref::MemRefDialect, NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (failed(convertPlanToGPU(getOperation(), *target)))
      signalPassFailure();
  }

private:
  const TargetDescription *target = &nvidiaTarget();
};

} // namespace

LogicalResult convertPlanToGPU(ModuleOp module,
                               const TargetDescription &target) {
  // The conversion rewrites every plan function in the module, also one in
  // a nested module, so every one of them is checked before any is changed.
  WalkResult checked = module.walk([&](func::FuncOp function) {
    if (blockThreadsOf(function) &&
        (failed(verifyPlanFunction(module, function, target)) ||
         failed(buildKernelContract(function))))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (checked.wasInterrupted())
    return failure();

  auto isPlanFunction = [](Operation *op) {
    return static_cast<bool>(blockThreadsOf(op));
  };
  ConversionTarget legality(*module.getContext());
  // The conversion leaves everything outside plan functions as it is.
  legality.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  legality.addDynamicallyLegalOp<func::FuncOp>(
      [=](func::FuncOp function) { return !isPlanFunction(function); });
  legality.markOpRecursivelyLegal<func::FuncOp>(
      [=](func::FuncOp function) { return !isPlanFunction(function); });
  legality.addDynamicallyLegalOp<func::ReturnOp>([=](func::ReturnOp returnOp) {
    return !isPlanFunction(returnOp->getParentOp());
  });
  legality.addIllegalDialect<SwageDialect, SwagePlanDialect>();

  RewritePatternSet patterns(module.getContext());
  patterns.add<PlanKernelFuncPattern, TasksPattern, FusedTasksPattern,
               PartialTasksPattern, MergeTasksPattern, PersistentTasksPattern>(
      module.getContext(), target);
  patterns.add<PlanKernelReturnPattern>(module.getContext());
  populateSegmentConsumerPatterns(patterns, &target);
  return applyFullConversion(module, legality, std::move(patterns));
}

std::unique_ptr<Pass> createSwagePlanToGPUPass() {
  return std::make_unique<SwagePlanToGPUPass>();
}

std::unique_ptr<Pass>
createSwagePlanToGPUPass(const TargetDescription &target) {
  return std::make_unique<SwagePlanToGPUPass>(target);
}

void registerSwagePlanToGPUPass() { PassRegistration<SwagePlanToGPUPass>(); }

} // namespace mlir::swage
