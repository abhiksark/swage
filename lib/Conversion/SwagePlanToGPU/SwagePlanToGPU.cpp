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
#include "swage/Conversion/SwagePlanToGPU/ConsumerPatterns.h"
#include "swage/Conversion/SwagePlanToGPU/Emission.h"
#include "swage/Conversion/SwageToPlan/Admission.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Dialect/SwagePlan/IR/TaskRecords.h"
#include "swage/Target/TargetDescription.h"

namespace mlir::swage {
namespace {

using swage_plan::PartialTasksOp;
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
    moveConvertedOperations<TasksOp, PartialTasksOp, func::ReturnOp>(
        rewriter, body, entry);
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

/// One block of threads per chunk of a split segment: the kernel prelude,
/// the guard on the task index, the range of the chunk loaded from its
/// record and clamped to the value count, the reduction of the region, and
/// the store of its result in the scratch slot of the task.
class PartialTasksPattern : public OpConversionPattern<PartialTasksOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PartialTasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    IntegerAttr threads = blockThreadsOf(tasks->getParentOp());
    if (!threads)
      return rewriter.notifyMatchFailure(tasks, "not in a plan function");
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

    Block &region = tasks.getBody().front();
    auto yield = cast<swage_plan::YieldOp>(region.getTerminator());
    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
          using namespace swage_plan::partial_record;
          Value fields = arith::ConstantIndexOp::create(body, bodyLoc, Words);
          Value recordBase =
              arith::MulIOp::create(body, bodyLoc, taskIndex, fields);
          Value beginWord = loadRecordField(body, bodyLoc, adaptor.getRanges(),
                                            recordBase, Begin);
          Value endWord = loadRecordField(body, bodyLoc, adaptor.getRanges(),
                                          recordBase, End);
          // The range indexes the values, so the value count bounds it.
          Value begin;
          Value end;
          std::tie(begin, end) = clampRange(body, bodyLoc, beginWord, endWord,
                                            adaptor.getValueCount());
          Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
          rewriter.replaceAllUsesWith(
              region.getArgument(0),
              ValueRange{adaptor.getValues(), first, end, block});

          SmallVector<Operation *> consumers = operationsOf(region);
          consumers.pop_back();
          converted = succeeded(legalizeInPlace(rewriter, consumers));
          if (converted) {
            moveConvertedOperations<ReduceOp, swage_plan::YieldOp>(
                rewriter, region, body.getInsertionBlock());
            emitLeaderStore(
                body, bodyLoc, rewriter.getRemappedValue(yield.getValue()),
                adaptor.getScratch(), taskIndex, threadId, zero, Value());
          }
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
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

  Operation *task = &function.getBody().front().front();
  if (auto partial = dyn_cast<PartialTasksOp>(task)) {
    if (failed(verifyTaskConsumers(partial, partial.getValues().getType(),
                                   partial.getRanges().getType(),
                                   partial.getBody().front())))
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
  patterns.add<PlanKernelFuncPattern>(module.getContext(), target);
  patterns.add<PlanKernelReturnPattern, TasksPattern, PartialTasksPattern>(
      module.getContext());
  populateSegmentConsumerPatterns(patterns, &target);
  return applyFullConversion(module, legality, std::move(patterns));
}

std::unique_ptr<Pass> createSwagePlanToGPUPass() {
  return std::make_unique<SwagePlanToGPUPass>();
}

void registerSwagePlanToGPUPass() { PassRegistration<SwagePlanToGPUPass>(); }

} // namespace mlir::swage
