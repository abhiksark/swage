// include/swage/Dialect/Swage/Transforms/FuseMaps.h
//===- FuseMaps.h - Fuse swage.map into its consumer ----------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// A `swage.map` result is a lazy view: the consumer of the result applies
// the region of the map to each element it reads. Fusion makes that explicit.
// It moves the region of a map with one consumer in front of the region of
// that consumer, so the consumer reads the operand segment of the map and
// the map disappears.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGE_TRANSFORMS_FUSEMAPS_H
#define SWAGE_DIALECT_SWAGE_TRANSFORMS_FUSEMAPS_H

#include "llvm/Support/LogicalResult.h"

#include <memory>

namespace mlir {
class Operation;
class Pass;
class RewritePatternSet;
class RewriterBase;
} // namespace mlir

namespace mlir::swage {

/// Fuse the `swage.map` that produces the segment of `consumer` into
/// `consumer`, which is a `swage.map`, a `swage.reduce`, or a
/// `swage.map_store`. The consumer then reads the operand segment of the
/// map, takes the captures of the map ahead of its own, and runs the
/// operations of the map ahead of its own on each element. The map is
/// erased.
///
/// Fails, and changes nothing, when the segment of `consumer` is not the
/// result of a map or when that result has another use.
llvm::LogicalResult fuseMapIntoConsumer(Operation *consumer,
                                        RewriterBase &rewriter);

/// Add `FuseMapIntoReduce`, `FuseMapIntoMapStore`, and `FuseMapIntoMap`,
/// each rooted at the consumer.
void populateFuseMapsPatterns(RewritePatternSet &patterns);

/// `--swage-fuse-maps`: apply the fusion patterns to a function until no
/// map with one consumer is left.
std::unique_ptr<Pass> createFuseMapsPass();
void registerFuseMapsPass();

} // namespace mlir::swage

#endif // SWAGE_DIALECT_SWAGE_TRANSFORMS_FUSEMAPS_H
