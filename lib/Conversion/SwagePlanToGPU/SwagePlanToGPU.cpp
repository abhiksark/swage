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
// segment. What each pattern emits comes from Emission.h, which the
// schedules that are not planned yet share.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwagePlanToGPU/SwagePlanToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "swage/Conversion/SwagePlanToGPU/Emission.h"
#include "swage/Conversion/SwageToPlan/Admission.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Target/TargetDescription.h"

namespace mlir::swage {
namespace {

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

/// The operations of `block`, collected so that a pattern can legalize them
/// while they still sit under their parent and move the results afterwards.
SmallVector<Operation *> operationsOf(Block &block) {
  SmallVector<Operation *> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  return operations;
}

/// Legalize `operations` in place, in order. A nested pattern moves the
/// insertion point of the shared rewriter, so it is restored on return.
LogicalResult legalizeInPlace(ConversionPatternRewriter &rewriter,
                              ArrayRef<Operation *> operations) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (Operation *operation : operations)
    if (failed(rewriter.legalize(operation)))
      return failure();
  return success();
}

/// Move what the patterns created in `source` to the end of `destination`,
/// in order. The plan operations of `source` stay behind: they are replaced,
/// and they leave with their parent.
template <typename... PlanOps>
void moveConvertedOperations(ConversionPatternRewriter &rewriter, Block &source,
                             Block *destination) {
  for (Operation &operation : llvm::make_early_inc_range(source))
    if (!isa<PlanOps...>(operation))
      rewriter.moveOpBefore(&operation, destination, destination->end());
}

/// A plan function becomes a `gpu.module` named after it that holds the
/// kernel: every buffer argument becomes a pointer, every count stays as it
/// is, and the launch width is pinned through the target.
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

    rewriter.setInsertionPoint(function);
    auto gpuModule = gpu::GPUModuleOp::create(
        rewriter, loc, function.getName().str() + "_module");
    rewriter.setInsertionPointToStart(gpuModule.getBody());
    Type pointer = LLVM::LLVMPointerType::get(context);
    SmallVector<Type> inputs;
    for (Type input : function.getFunctionType().getInputs())
      inputs.push_back(isa<MemRefType>(input) ? pointer : input);
    auto kernel =
        gpu::GPUFuncOp::create(rewriter, loc, function.getName(),
                               FunctionType::get(context, inputs, {}));
    kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                    rewriter.getUnitAttr());
    target.pinLaunchWidth(kernel, static_cast<int32_t>(threads.getInt()));

    Block &body = function.getBody().front();
    Block *entry = &kernel.getBody().front();
    for (auto [argument, parameter] :
         llvm::zip_equal(body.getArguments(), entry->getArguments()))
      rewriter.replaceAllUsesWith(argument, parameter);
    if (failed(legalizeInPlace(rewriter, operationsOf(body))))
      return failure();
    moveConvertedOperations<TasksOp, func::ReturnOp>(rewriter, body, entry);
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

/// One block of threads per task: the kernel prelude, the guard on the task
/// index, the binding of the segment of the task, the consumers of the
/// region in order, and the store of the yielded scalar.
///
/// The region argument is replaced by the four values of the binding, and
/// the consumers are legalized while the task operation is still their
/// parent, so each consumer pattern takes the binding from its operands and
/// the policy from the operation it sits in.
class TasksPattern : public OpConversionPattern<TasksOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TasksOp tasks, OpAdaptor adaptor,
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
    // One block per task with a task buffer, and one per segment without.
    Value taskCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(),
        ids ? adaptor.getTaskCount() : adaptor.getSegmentCount());
    Value inRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, taskIndex, taskCount);

    Block &region = tasks.getBody().front();
    auto yield = cast<swage_plan::YieldOp>(region.getTerminator());
    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
          // The direct kernel uses the block index as the segment ID, which
          // the guard compared with the segment count. A segment ID loaded
          // from a task buffer is bounded here.
          Value segmentId = taskIndex;
          Value segmentInRange;
          if (ids) {
            Value segmentIdWord = loadTaskWord(body, bodyLoc, ids, taskIndex);
            segmentInRange = isLoadedIndexInRange(body, bodyLoc, segmentIdWord,
                                                  adaptor.getSegmentCount());
            segmentId = arith::IndexCastOp::create(
                body, bodyLoc, body.getIndexType(), segmentIdWord);
          }
          BoundSegment bound = emitSegmentBinding(
              body, bodyLoc, adaptor.getValues(), adaptor.getOffsets(),
              adaptor.getValueCount(), segmentId, segmentInRange, threadId,
              block, zero, one);
          rewriter.replaceAllUsesWith(
              region.getArgument(0),
              ValueRange{bound.segment.base, bound.segment.first,
                         bound.segment.end, bound.segment.stride});

          SmallVector<Operation *> consumers = operationsOf(region);
          consumers.pop_back();
          converted = succeeded(legalizeInPlace(rewriter, consumers));
          if (converted) {
            moveConvertedOperations<ReduceOp, MapStoreOp, swage_plan::YieldOp>(
                rewriter, region, body.getInsertionBlock());
            if (Value scalar = yield.getValue())
              emitScalarStore(body, bodyLoc, rewriter.getRemappedValue(scalar),
                              adaptor.getOutput(), bound.segmentId64, threadId,
                              zero, segmentInRange);
          }
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }
};

/// The four values of the bound segment a consumer reads, from the operand
/// the task pattern replaced.
std::optional<SegmentBinding> boundSegmentOf(ValueRange segment) {
  if (segment.size() != 4)
    return std::nullopt;
  return SegmentBinding{segment[0], segment[1], segment[2], segment[3]};
}

/// The capture operands of a consumer, one value each.
std::optional<SmallVector<Value>> capturesOf(ArrayRef<ValueRange> captures) {
  SmallVector<Value> values;
  for (ValueRange capture : captures) {
    if (capture.size() != 1)
      return std::nullopt;
    values.push_back(capture.front());
  }
  return values;
}

/// A reduction of the bound segment becomes one reduction stage: every
/// thread folds its elements through the element program, then the threads
/// combine their results as the policy of the task operation says.
class ReducePattern : public OpConversionPattern<ReduceOp> {
public:
  ReducePattern(MLIRContext *context, const TargetDescription &target)
      : OpConversionPattern(context), target(target) {}

  LogicalResult
  matchAndRewrite(ReduceOp reduce, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<TaskPolicy> policy =
        swage_plan::policyOfRegion(reduce->getParentRegion());
    std::optional<SegmentBinding> segment =
        boundSegmentOf(adaptor.getSegment());
    std::optional<SmallVector<Value>> captures =
        capturesOf(adaptor.getCaptures());
    if (!policy || !segment || !captures)
      return rewriter.notifyMatchFailure(reduce, "not in a task region");
    Type element =
        cast<SegmentType>(reduce.getSegment().getType()).getElementType();
    Value total = emitReductionStage(
        rewriter, reduce.getLoc(), target, reduce.getKind(), element, *segment,
        *policy == TaskPolicy::Warp, [&](OpBuilder &loop, Value value) {
          SmallVector<Value> arguments{value};
          arguments.append(*captures);
          return inlineRegion(loop, reduce.getBody(), arguments);
        });
    rewriter.replaceOp(reduce, total);
    return success();
  }

private:
  const TargetDescription &target;
};

/// A store of the bound segment becomes the loop that writes every element
/// program result to the same index of the output.
class MapStorePattern : public OpConversionPattern<MapStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MapStoreOp mapStore, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<SegmentBinding> segment =
        boundSegmentOf(adaptor.getSegment());
    std::optional<SmallVector<Value>> captures =
        capturesOf(adaptor.getCaptures());
    if (!segment || !captures || adaptor.getOutput().size() != 1)
      return rewriter.notifyMatchFailure(mapStore, "not in a task region");
    Type element =
        cast<SegmentType>(mapStore.getSegment().getType()).getElementType();
    emitMapStore(rewriter, mapStore.getLoc(), element, *segment,
                 adaptor.getOutput().front(),
                 [&](OpBuilder &loop, Value value) {
                   SmallVector<Value> arguments{value};
                   arguments.append(*captures);
                   return inlineRegion(loop, mapStore.getBody(), arguments);
                 });
    rewriter.eraseOp(mapStore);
    return success();
  }
};

class SwagePlanToGPUPass
    : public PassWrapper<SwagePlanToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwagePlanToGPUPass)

  StringRef getArgument() const final { return "swage-plan-to-gpu"; }
  StringRef getDescription() const final {
    return "Convert every plan function to a GPU kernel module";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (failed(convertPlanToGPU(getOperation(), nvidiaTarget())))
      signalPassFailure();
  }
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

  auto tasks = cast<TasksOp>(function.getBody().front().front());
  Type element = cast<MemRefType>(tasks.getValues().getType()).getElementType();
  Type word = cast<MemRefType>(tasks.getOffsets().getType()).getElementType();
  if (!isAdmittedElementType(element) || !isAdmittedIndexType(word))
    return tasks.emitError()
           << "the conversion lowers f32 values with i32 offsets and counts, "
              "got values of "
           << element << " and offsets of " << word;
  // A warp task reduces within one subgroup, so its block is one subgroup.
  if (tasks.getPolicy() == TaskPolicy::Warp && threads != target.subgroupWidth)
    return tasks.emitError()
           << "policy<warp> requires " << name << " to be the subgroup width, "
           << target.subgroupWidth << ", got " << threads;
  SegmentProgramAnalysis consumers;
  for (Operation &operation : tasks.getBody().front().without_terminator()) {
    if (auto reduction = dyn_cast<ReduceOp>(operation))
      consumers.reductions.push_back(reduction);
    else
      consumers.mapStores.push_back(cast<MapStoreOp>(operation));
  }
  if (failed(verifyConsumerPrograms(consumers)))
    return failure();
  return verifyKernelSymbols(module, function, "");
}

} // namespace

LogicalResult convertPlanToGPU(ModuleOp module,
                               const TargetDescription &target) {
  for (func::FuncOp function : module.getOps<func::FuncOp>())
    if (blockThreadsOf(function) &&
        failed(verifyPlanFunction(module, function, target)))
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
  patterns.add<PlanKernelFuncPattern, ReducePattern>(module.getContext(),
                                                     target);
  patterns.add<PlanKernelReturnPattern, TasksPattern, MapStorePattern>(
      module.getContext());
  return applyFullConversion(module, legality, std::move(patterns));
}

std::unique_ptr<Pass> createSwagePlanToGPUPass() {
  return std::make_unique<SwagePlanToGPUPass>();
}

void registerSwagePlanToGPUPass() { PassRegistration<SwagePlanToGPUPass>(); }

} // namespace mlir::swage
