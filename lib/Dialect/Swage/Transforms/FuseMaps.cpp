// lib/Dialect/Swage/Transforms/FuseMaps.cpp
//===- FuseMaps.cpp - Fuse swage.map into its consumer --------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/Swage/Transforms/FuseMaps.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"

namespace mlir::swage {

LogicalResult fuseMapIntoConsumer(Operation *consumer, RewriterBase &rewriter) {
  if (!isa<MapOp, ReduceOp, MapStoreOp>(consumer))
    return failure();
  // The segment is the first operand of every consumer. A map_store takes
  // its output buffer between the segment and the captures.
  auto map = consumer->getOperand(0).getDefiningOp<MapOp>();
  if (!map || !map.getResult().hasOneUse())
    return failure();
  unsigned firstCapture = isa<MapStoreOp>(consumer) ? 2 : 1;

  Block &body = consumer->getRegion(0).front();
  Block &mapBody = map.getBody().front();
  auto yield = cast<YieldOp>(mapBody.getTerminator());
  SmallVector<Value> mapCaptures(map.getCaptures());

  // The fused block takes the element of the operand segment, then the
  // captures of the map, then the captures of the consumer. The old element
  // argument of the consumer stays until its uses are replaced.
  Value element = body.getArgument(0);
  SmallVector<Value> mapArguments;
  for (auto [index, argument] : llvm::enumerate(mapBody.getArguments()))
    mapArguments.push_back(body.insertArgument(
        static_cast<unsigned>(index), argument.getType(), argument.getLoc()));

  // The operations of the map run first. What the map yields is what the
  // operations of the consumer read as their element.
  rewriter.inlineBlockBefore(&mapBody, &body, body.begin(), mapArguments);
  rewriter.replaceAllUsesWith(element, yield.getValue());
  rewriter.eraseOp(yield);
  body.eraseArgument(mapArguments.size());

  rewriter.modifyOpInPlace(consumer, [&] {
    consumer->setOperand(0, map.getSegment());
    consumer->insertOperands(firstCapture, mapCaptures);
  });
  rewriter.eraseOp(map);
  return success();
}

namespace {

/// Fuse a single-consumer map into the consumer the pattern is rooted at.
template <typename ConsumerOp>
struct FuseMapInto : public OpRewritePattern<ConsumerOp> {
  using OpRewritePattern<ConsumerOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConsumerOp consumer,
                                PatternRewriter &rewriter) const override {
    return fuseMapIntoConsumer(consumer, rewriter);
  }
};

using FuseMapIntoReduce = FuseMapInto<ReduceOp>;
using FuseMapIntoMapStore = FuseMapInto<MapStoreOp>;
using FuseMapIntoMap = FuseMapInto<MapOp>;

class FuseMapsPass : public PassWrapper<FuseMapsPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FuseMapsPass)

  StringRef getArgument() const final { return "swage-fuse-maps"; }
  StringRef getDescription() const final {
    return "Fuse each swage.map that has one consumer into that consumer";
  }

  void runOnOperation() final {
    RewritePatternSet patterns(&getContext());
    populateFuseMapsPatterns(patterns);
    // The pass fuses and does nothing else: no folding, no merging of
    // constants, and no region simplification.
    GreedyRewriteConfig config;
    config.enableFolding(false);
    config.enableConstantCSE(false);
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    if (failed(applyPatternsGreedily(
            getOperation(), FrozenRewritePatternSet(std::move(patterns)),
            config)))
      signalPassFailure();
  }
};

} // namespace

void populateFuseMapsPatterns(RewritePatternSet &patterns) {
  patterns.add<FuseMapIntoReduce, FuseMapIntoMapStore, FuseMapIntoMap>(
      patterns.getContext());
}

std::unique_ptr<Pass> createFuseMapsPass() {
  return std::make_unique<FuseMapsPass>();
}

void registerFuseMapsPass() { PassRegistration<FuseMapsPass>(); }

} // namespace mlir::swage
