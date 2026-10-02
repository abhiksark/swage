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
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Dialect/SwagePlan/IR/TaskRecords.h"
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

/// A plan function becomes a `gpu.module` named after it that holds the
/// kernel: every buffer argument becomes a pointer, every count stays as it
/// is, and the launch width is pinned through the target. The kernel of a
/// persistent task operation also gets the two claim slots its blocks share.
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
    // One slot carries a claim from the leader of a block to its threads,
    // and the other the merge a partial task made ready.
    if (isa<PersistentTasksOp>(body.front()))
      kernel.addWorkgroupAttribution(
          MemRefType::get(
              {2}, rewriter.getI32Type(), AffineMap(),
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

/// One block of threads per task: the kernel prelude, the guard on the task
/// index, and the segment of the task. The direct kernel uses the block
/// index as the segment ID; with a task buffer the ID is loaded from it.
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
              convertSegmentTask(rewriter, body, bodyLoc, tasks.getBody(), site,
                                 ids, taskIndex, threadId, block);
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }
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
Value convertRangeTask(ConversionPatternRewriter &rewriter, OpBuilder &body,
                       Region &region, const SegmentBinding &range) {
  Block &consumersBlock = region.front();
  rewriter.replaceAllUsesWith(
      consumersBlock.getArgument(0),
      ValueRange{range.base, range.first, range.end, range.stride});
  auto yield = cast<swage_plan::YieldOp>(consumersBlock.getTerminator());
  SmallVector<Operation *> consumers = operationsOf(consumersBlock);
  consumers.pop_back();
  if (failed(legalizeInPlace(rewriter, consumers)))
    return Value();
  moveConvertedOperations<ReduceOp, swage_plan::YieldOp>(
      rewriter, consumersBlock, body.getInsertionBlock());
  return rewriter.getRemappedValue(yield.getValue());
}

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
};

/// One block of threads per split segment: the kernel prelude, the guard on
/// the task index, the merge record of the segment, the reduction of the
/// region over the scratch slots the record names, and the store of its
/// result at the segment the record names.
///
/// The segment comes from device memory, so it is compared with the segment
/// count, and one that fails stores nothing. The reduction itself stays
/// unconditional, which keeps its block-wide combination under the
/// block-uniform guard alone.
class MergeTasksPattern : public OpConversionPattern<MergeTasksOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MergeTasksOp tasks, OpAdaptor adaptor,
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
        rewriter, loc, rewriter.getIndexType(), adaptor.getMergeCount());
    Value inRange = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, taskIndex, taskCount);

    bool converted = true;
    scf::IfOp::create(
        rewriter, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
          using namespace swage_plan::merge_record;
          Value records = adaptor.getMerges();
          Value fields = arith::ConstantIndexOp::create(body, bodyLoc, Words);
          Value recordBase =
              arith::MulIOp::create(body, bodyLoc, taskIndex, fields);
          Value segmentWord =
              loadRecordField(body, bodyLoc, records, recordBase, Segment);
          Value segmentInRange = isLoadedIndexInRange(
              body, bodyLoc, segmentWord, adaptor.getSegmentCount());
          Value segment = arith::IndexCastOp::create(
              body, bodyLoc, body.getIndexType(), segmentWord);
          Value beginWord =
              loadRecordField(body, bodyLoc, records, recordBase, PartialBegin);
          Value endWord =
              loadRecordField(body, bodyLoc, records, recordBase, PartialEnd);
          // The range indexes scratch, so the partial count bounds it.
          Value begin;
          Value end;
          std::tie(begin, end) = clampRange(body, bodyLoc, beginWord, endWord,
                                            adaptor.getPartialCount());
          Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
          Value total =
              convertRangeTask(rewriter, body, tasks.getBody(),
                               {adaptor.getScratch(), first, end, block});
          converted = static_cast<bool>(total);
          if (converted)
            emitLeaderStore(body, bodyLoc, total, adaptor.getOutput(), segment,
                            threadId, zero, segmentInRange);
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }
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
  if (auto merge = dyn_cast<MergeTasksOp>(task)) {
    if (failed(verifyTaskConsumers(merge, merge.getScratch().getType(),
                                   merge.getMerges().getType(),
                                   merge.getBody().front())))
      return failure();
    return verifyKernelSymbols(module, function, "");
  }
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
  patterns
      .add<PlanKernelFuncPattern, FusedTasksPattern, PersistentTasksPattern>(
          module.getContext(), target);
  patterns.add<PlanKernelReturnPattern, TasksPattern, PartialTasksPattern,
               MergeTasksPattern>(module.getContext());
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
