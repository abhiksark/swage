// lib/Conversion/SwagePlanToGPU/ConsumerPatterns.cpp
//===- ConsumerPatterns.cpp - Patterns for a bound segment ----------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwagePlanToGPU/ConsumerPatterns.h"

#include "swage/Conversion/SwagePlanToGPU/Emission.h"
#include "swage/Conversion/SwageToPlan/Admission.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"

namespace mlir::swage {
namespace {

using swage_plan::TaskPolicy;

/// The values of the bound segment a consumer reads, from the operand the
/// task pattern replaced: the four of every binding, and for the row-stripe
/// tile of rank-two values the group width and the exchange buffer too.
std::optional<SegmentBinding> boundSegmentOf(ValueRange segment) {
  if (segment.size() == 4)
    return SegmentBinding{segment[0], segment[1], segment[2], segment[3]};
  if (segment.size() == 6)
    return SegmentBinding{segment[0], segment[1], segment[2],
                          segment[3], segment[4], segment[5]};
  return std::nullopt;
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

/// How the threads of a task of `policy` combine their accumulators.
ThreadCombination combinationOf(TaskPolicy policy) {
  switch (policy) {
  case TaskPolicy::Warp:
    return ThreadCombination::Subgroup;
  case TaskPolicy::CTA:
    return ThreadCombination::Block;
  case TaskPolicy::Sequential:
  case TaskPolicy::Column:
    // One thread reduces the whole range: the oracle, and a column.
    return ThreadCombination::None;
  }
  llvm_unreachable("unknown task policy");
}

/// A reduction of the bound segment becomes one reduction stage: every
/// thread folds its elements through the element program, then the threads
/// combine their results as the policy of the task operation says.
class ReducePattern : public OpConversionPattern<ReduceOp> {
public:
  ReducePattern(MLIRContext *context, const TargetDescription *target)
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
    // A block task of the row-stripe tile combines per column.
    ThreadCombination combination = combinationOf(*policy);
    if (combination == ThreadCombination::Block && segment->exchange)
      combination = ThreadCombination::ColumnGroup;
    if ((combination == ThreadCombination::Subgroup ||
         combination == ThreadCombination::ColumnGroup) &&
        !target)
      return rewriter.notifyMatchFailure(reduce,
                                         "a warp task or a column group needs "
                                         "a target");
    Type element =
        cast<SegmentType>(reduce.getSegment().getType()).getElementType();
    Value total = emitReductionStage(
        rewriter, reduce.getLoc(), target, reduce.getKind(), element, *segment,
        combination, [&](OpBuilder &loop, Value value) {
          SmallVector<Value> arguments{value};
          arguments.append(*captures);
          return inlineRegion(loop, reduce.getBody(), arguments);
        });
    rewriter.replaceOp(reduce, total);
    return success();
  }

private:
  const TargetDescription *target;
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

} // namespace

void populateSegmentConsumerPatterns(RewritePatternSet &patterns,
                                     const TargetDescription *target) {
  patterns.add<ReducePattern>(patterns.getContext(), target);
  patterns.add<MapStorePattern>(patterns.getContext());
}

LogicalResult verifyTaskConsumers(Operation *task, Type valuesType,
                                  Type offsetsType, Block &consumers) {
  Type element = cast<MemRefType>(valuesType).getElementType();
  Type word = cast<MemRefType>(offsetsType).getElementType();
  if (!isAdmittedElementType(element) || !isAdmittedIndexType(word))
    return task->emitError()
           << "the conversion lowers f32 or f64 values with i32 offsets and "
              "counts, got values of "
           << element << " and offsets of " << word;
  SegmentProgramAnalysis analysis;
  analysis.element = element;
  // What is neither is the scalar epilogue of the region: ordinary
  // arithmetic that no pattern lowers and the dialect verifier admitted.
  for (Operation &operation : consumers.without_terminator()) {
    if (auto reduction = dyn_cast<ReduceOp>(operation))
      analysis.reductions.push_back(reduction);
    else if (auto mapStore = dyn_cast<MapStoreOp>(operation))
      analysis.mapStores.push_back(mapStore);
  }
  return verifyConsumerPrograms(analysis);
}

SmallVector<Operation *> operationsOf(Block &block) {
  SmallVector<Operation *> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  return operations;
}

LogicalResult legalizeInPlace(ConversionPatternRewriter &rewriter,
                              ArrayRef<Operation *> operations) {
  OpBuilder::InsertionGuard guard(rewriter);
  for (Operation *operation : operations)
    if (failed(rewriter.legalize(operation)))
      return failure();
  return success();
}

} // namespace mlir::swage
