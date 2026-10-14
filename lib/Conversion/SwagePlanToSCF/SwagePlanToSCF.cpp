// lib/Conversion/SwagePlanToSCF/SwagePlanToSCF.cpp
//===- SwagePlanToSCF.cpp - Sequential plans to loops ---------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// The CPU oracle as a dialect conversion. One pattern lowers the sequential
// task operation to a loop over the segments, and the consumers of the
// bound segment are lowered by the patterns the kernel conversion uses, on
// memrefs instead of pointers and without a combination across threads.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwagePlanToSCF/SwagePlanToSCF.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "swage/Conversion/SwagePlanToGPU/ConsumerPatterns.h"
#include "swage/Conversion/SwagePlanToGPU/Emission.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"

namespace mlir::swage {
namespace {

using swage_plan::TaskPolicy;
using swage_plan::TasksOp;

/// Whether `op` is a task operation of the sequential policy.
bool isSequentialTasks(Operation *op) {
  auto tasks = dyn_cast_or_null<TasksOp>(op);
  return tasks && tasks.getPolicy() == TaskPolicy::Sequential;
}

/// One loop over the segments. Each iteration reads the range of its segment
/// from the offsets, runs the consumers of the region on it in order, and
/// stores the yielded scalar at the index of the segment.
///
/// The oracle trusts its offsets: it applies no clamp, because it runs on
/// host memory that the caller validated and nothing reloads it.
///
/// For rank-two values each iteration holds a second loop, over the
/// columns. A column is a strided run of the row-order view of the values:
/// it starts at `start * columns + column` and takes every `columns`-th
/// element below `end * columns`. The rows of a column are visited in
/// order, which is the order the column kernel adds them in.
class SequentialTasksPattern : public OpConversionPattern<TasksOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TasksOp tasks, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isSequentialTasks(tasks))
      return rewriter.notifyMatchFailure(tasks, "not a sequential task");
    Location loc = tasks.getLoc();
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value segmentCount = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), adaptor.getSegmentCount());

    Block &region = tasks.getBody().front();
    auto yield = cast<swage_plan::YieldOp>(region.getTerminator());
    bool converted = true;
    // Run the consumers of the region on `segment` at the insertion point
    // of `body` and store the yielded scalar at `output[slot]`.
    auto runRegion = [&](OpBuilder &body, Location bodyLoc, Value base,
                         Value first, Value end, Value stride, Value extent,
                         ValueRange slot) {
      rewriter.replaceAllUsesWith(region.getArgument(0),
                                  ValueRange{base, first, end, stride});
      // A region that takes the extent of its segment runs its scalar
      // epilogue after the reductions, once per segment.
      if (region.getNumArguments() == 2)
        rewriter.replaceAllUsesWith(region.getArgument(1), extent);
      SmallVector<Operation *> consumers = operationsOf(region);
      consumers.pop_back();
      converted = succeeded(legalizeInPlace(rewriter, consumers));
      if (!converted)
        return;
      moveConvertedOperations<ReduceOp, MapStoreOp, swage_plan::YieldOp>(
          rewriter, region, body.getInsertionBlock());
      if (Value scalar = yield.getValue())
        memref::StoreOp::create(body, bodyLoc,
                                rewriter.getRemappedValue(scalar),
                                adaptor.getOutput(), slot);
    };

    // The row-order view of rank-two values, and their number of columns.
    Value featureCount = adaptor.getFeatureCount();
    Value values = adaptor.getValues();
    Value columns;
    if (featureCount) {
      values = flattenRows(rewriter, loc, values);
      columns = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), featureCount);
    }
    scf::ForOp::create(
        rewriter, loc, zero, segmentCount, one, ValueRange(),
        [&](OpBuilder &body, Location bodyLoc, Value segmentId, ValueRange) {
          Value offsets = adaptor.getOffsets();
          Value startWord =
              memref::LoadOp::create(body, bodyLoc, offsets, segmentId);
          Value next = arith::AddIOp::create(body, bodyLoc, segmentId, one);
          Value endWord = memref::LoadOp::create(body, bodyLoc, offsets, next);
          Value start = arith::IndexCastOp::create(
              body, bodyLoc, body.getIndexType(), startWord);
          Value end = arith::IndexCastOp::create(body, bodyLoc,
                                                 body.getIndexType(), endWord);
          Value extent;
          if (region.getNumArguments() == 2)
            extent = arith::SubIOp::create(body, bodyLoc, end, start);
          if (!featureCount) {
            runRegion(body, bodyLoc, values, start, end, one, extent,
                      segmentId);
            scf::YieldOp::create(body, bodyLoc);
            return;
          }
          Value firstRow = arith::MulIOp::create(body, bodyLoc, start, columns);
          Value endRow = arith::MulIOp::create(body, bodyLoc, end, columns);
          scf::ForOp::create(
              body, bodyLoc, zero, columns, one, ValueRange(),
              [&](OpBuilder &loop, Location loopLoc, Value column, ValueRange) {
                Value first =
                    arith::AddIOp::create(loop, loopLoc, firstRow, column);
                runRegion(loop, loopLoc, values, first, endRow, columns, extent,
                          {segmentId, column});
                scf::YieldOp::create(loop, loopLoc);
              });
          scf::YieldOp::create(body, bodyLoc);
        });
    if (!converted)
      return failure();
    rewriter.eraseOp(tasks);
    return success();
  }
};

class SwagePlanToSCFPass
    : public PassWrapper<SwagePlanToSCFPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwagePlanToSCFPass)

  StringRef getArgument() const final { return "swage-plan-to-scf"; }
  StringRef getDescription() const final {
    return "Convert every sequential task operation to loops over its "
           "memrefs: the CPU oracle";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (failed(convertPlanToSCF(getOperation())))
      signalPassFailure();
  }
};

} // namespace

LogicalResult convertPlanToSCF(ModuleOp module) {
  // A failed pattern could only say "failed to legalize", so what the
  // patterns rely on is checked first and reported with its operation. The
  // functions that hold a sequential task operation are remembered: they
  // lose their roles once the operation is gone.
  SmallVector<func::FuncOp> functions;
  bool admitted = true;
  module.walk([&](TasksOp tasks) {
    if (!isSequentialTasks(tasks))
      return;
    admitted &= succeeded(verifyTaskConsumers(
        tasks, tasks.getValues().getType(), tasks.getOffsets().getType(),
        tasks.getBody().front()));
    if (auto function = tasks->getParentOfType<func::FuncOp>())
      functions.push_back(function);
  });
  if (!admitted)
    return failure();

  ConversionTarget legality(*module.getContext());
  // The conversion leaves everything outside sequential task operations as
  // it is, kernels that are planned for the GPU included.
  legality.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  legality.addDynamicallyLegalOp<TasksOp>(
      [](TasksOp tasks) { return !isSequentialTasks(tasks); });
  legality.markOpRecursivelyLegal<TasksOp>(
      [](TasksOp tasks) { return !isSequentialTasks(tasks); });
  legality.addDynamicallyLegalOp<ReduceOp, MapStoreOp>(
      [](Operation *op) { return !isSequentialTasks(op->getParentOp()); });

  RewritePatternSet patterns(module.getContext());
  patterns.add<SequentialTasksPattern>(module.getContext());
  populateSegmentConsumerPatterns(patterns, /*target=*/nullptr);
  if (failed(applyFullConversion(module, legality, std::move(patterns))))
    return failure();

  // The roles are consumed. What remains is ordinary upstream IR, which a
  // tool that does not know the Swage dialect must be able to parse.
  auto role =
      StringAttr::get(module.getContext(), SwageDialect::getRoleAttrName());
  for (func::FuncOp function : functions)
    for (unsigned index = 0; index < function.getNumArguments(); ++index)
      function.removeArgAttr(index, role);
  return success();
}

std::unique_ptr<Pass> createSwagePlanToSCFPass() {
  return std::make_unique<SwagePlanToSCFPass>();
}

void registerSwagePlanToSCFPass() { PassRegistration<SwagePlanToSCFPass>(); }

} // namespace mlir::swage
