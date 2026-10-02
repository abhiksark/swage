// include/swage/Conversion/SwagePlanToGPU/ConsumerPatterns.h
//===- ConsumerPatterns.h - Patterns for a bound segment -------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// What the conversions of plan operations share: the patterns that lower
// the consumers of a bound segment, and the steps a task pattern takes to
// legalize its region in place and move the result. The kernel conversion
// and the sequential oracle use the same consumer patterns, so a reduction
// or a store is lowered by one piece of code on both backends. This header
// is internal to the Swage conversions.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SWAGEPLANTOGPU_CONSUMERPATTERNS_H
#define SWAGE_CONVERSION_SWAGEPLANTOGPU_CONSUMERPATTERNS_H

#include "mlir/Transforms/DialectConversion.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"

namespace mlir::swage {

struct TargetDescription;

/// Add the patterns for `swage.reduce` and `swage.map_store` in a task
/// region. A task pattern replaces the segment argument of its region by the
/// four values of a binding, and these patterns take the binding from their
/// segment operand and the policy from the task operation they sit in.
///
/// `target` gives the subgroup width of a warp task. It may be null for a
/// conversion that lowers `policy<sequential>` only.
void populateSegmentConsumerPatterns(RewritePatternSet &patterns,
                                     const TargetDescription *target);

/// What the consumer patterns rely on and the dialect verifier does not
/// promise: the element and word types the lowerings admit, and element
/// programs of admitted operations and kinds. `consumers` is the block of
/// the task region, and `values` and `offsets` are the buffers the segment
/// is bound from. Reported on `task`, before anything is changed.
LogicalResult verifyTaskConsumers(Operation *task, Type valuesType,
                                  Type offsetsType, Block &consumers);

/// The operations of `block`, collected so that a pattern can legalize them
/// while they still sit under their parent and move the results afterwards.
SmallVector<Operation *> operationsOf(Block &block);

/// Legalize `operations` in place, in order. A nested pattern moves the
/// insertion point of the shared rewriter, so it is restored on return.
LogicalResult legalizeInPlace(ConversionPatternRewriter &rewriter,
                              ArrayRef<Operation *> operations);

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

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SWAGEPLANTOGPU_CONSUMERPATTERNS_H
