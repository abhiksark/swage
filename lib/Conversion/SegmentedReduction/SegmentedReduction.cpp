// lib/Conversion/SegmentedReduction/SegmentedReduction.cpp
//===- SegmentedReduction.cpp - Segmented reduction lowering ------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Conversion/SwagePlanToGPU/Emission.h"
#include "swage/Conversion/SwageToPlan/Admission.h"

#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/Swage/Transforms/FuseMaps.h"
#include "swage/Dialect/SwagePlan/IR/KernelLayout.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "swage/Target/TargetDescription.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;

namespace mlir::swage {
namespace {

/// A per-element expression: the region of a consumer after map fusion,
/// which holds the bodies of the fused maps in application order followed
/// by the consumer's own. `captures` lists the reduction stages whose
/// results bind to the capture arguments of the region, in order.
struct ElementProgram {
  Region *region = nullptr;
  SmallVector<unsigned> captures;
};

/// One `swage.reduce` and the element expression feeding it.
struct ReductionStage {
  ElementProgram element;
  ReductionKind kind = ReductionKind::Sum;
};

/// Where an admitted program writes its result.
enum class TerminalKind {
  ScalarStore, ///< One f32 per segment, at output[segment_id].
  MapStore     ///< One f32 per element, at output[element index].
};

/// An admitted segment program. It holds no handle into the source function:
/// every region is detached into a RegionOwner and every capture is a stage
/// index, so an emitter may erase the source IR before emitting.
struct SegmentProgram {
  SmallVector<ReductionStage> reductions;
  TerminalKind terminal = TerminalKind::ScalarStore;
  unsigned storedReduction = 0; ///< ScalarStore: index into `reductions`.
  ElementProgram mapStore;      ///< MapStore: the per-element expression.
};

/// Owns the region bodies detached from the source function, and must
/// outlive the SegmentProgram pointing into it.
class RegionOwner {
public:
  Region *take(Region &body) {
    owned.push_back(std::make_unique<Region>());
    owned.back()->takeBody(body);
    return owned.back().get();
  }

private:
  SmallVector<std::unique_ptr<Region>> owned;
};

/// Fuse every map into its consumer and detach the consumer regions. This
/// changes the function, so it runs only after read-only admission has
/// succeeded. Admission gives every map one consumer, so fusion leaves no
/// map behind and every consumer reads the segment of `make_segment`.
void detachSegmentProgram(SegmentProgramAnalysis &analysis, RegionOwner &owner,
                          SegmentProgram &program) {
  // The fusion function is applied to the admitted consumers directly. The
  // greedy pattern driver would also delete dead operations, and a program
  // may hold a reduction that nothing reads, which is lowered as a stage.
  IRRewriter rewriter(analysis.reductions.front()->getContext());
  auto fuse = [&](Operation *consumer) {
    while (succeeded(fuseMapIntoConsumer(consumer, rewriter))) {
    }
  };
  for (ReduceOp reduction : analysis.reductions)
    fuse(reduction);
  for (MapStoreOp mapStore : analysis.mapStores)
    fuse(mapStore);
  analysis.maps.clear();

  DenseMap<Operation *, unsigned> stageOf;
  for (auto [index, reduction] : llvm::enumerate(analysis.reductions))
    stageOf[reduction.getOperation()] = index;
  program.terminal = analysis.mapStores.empty() ? TerminalKind::ScalarStore
                                                : TerminalKind::MapStore;
  if (analysis.mapStores.empty())
    program.storedReduction =
        stageOf.lookup(analysis.storedReduction.getOperation());
  auto takeElement = [&](Operation *consumer, ValueRange captures) {
    ElementProgram element;
    for (Value capture : captures)
      element.captures.push_back(stageOf.lookup(capture.getDefiningOp()));
    element.region = owner.take(consumer->getRegion(0));
    return element;
  };
  for (ReduceOp reduction : analysis.reductions) {
    ReductionStage stage;
    stage.kind = reduction.getKind();
    stage.element = takeElement(reduction, reduction.getCaptures());
    program.reductions.push_back(std::move(stage));
  }
  if (!analysis.mapStores.empty()) {
    MapStoreOp mapStore = analysis.mapStores.front();
    program.mapStore = takeElement(mapStore, mapStore.getCaptures());
  }
}

void buildPlanningCompanion(ModuleOp module, func::FuncOp semanticFunction,
                            const SegmentABI &abi, int32_t warpMaxElements,
                            int32_t ctaChunkElements) {
  OpBuilder builder(module.getContext());
  Location loc = semanticFunction.getLoc();
  Type taskRange = swage_plan::TaskRangeType::get(module.getContext());
  auto functionType = builder.getFunctionType(
      {semanticFunction.getArgument(abi.offsets).getType(),
       builder.getI32Type(), builder.getI32Type()},
      taskRange);

  builder.setInsertionPointAfter(semanticFunction);
  auto companion = func::FuncOp::create(
      builder, loc, semanticFunction.getName().str() + "__swage_plan",
      functionType);
  companion.setPrivate();
  Block *entry = companion.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  ArrayAttr policies = builder.getArrayAttr(
      {swage_plan::TaskPolicyAttr::get(module.getContext(),
                                       swage_plan::TaskPolicy::Warp),
       swage_plan::TaskPolicyAttr::get(module.getContext(),
                                       swage_plan::TaskPolicy::CTA)});
  auto tasks = swage_plan::ClassifyOp::create(
      builder, loc, taskRange, entry->getArgument(0), entry->getArgument(1),
      entry->getArgument(2), semanticFunction.getName(),
      static_cast<uint32_t>(warpMaxElements),
      static_cast<uint32_t>(ctaChunkElements), policies);
  func::ReturnOp::create(builder, loc, tasks.getResult());
}

/// Apply an admitted element expression to one loaded value.
Value evaluateElement(OpBuilder &builder, const ElementProgram &element,
                      Value value, ArrayRef<Value> reductions) {
  SmallVector<Value> arguments{value};
  for (unsigned stage : element.captures)
    arguments.push_back(reductions[stage]);
  return inlineRegion(builder, *element.region, arguments);
}

void buildSequentialProgram(func::FuncOp function, const SegmentABI &abi,
                            const SegmentProgram &program) {
  Block &entry = function.getBody().front();
  Value values = function.getArgument(abi.values);
  Value offsets = function.getArgument(abi.offsets);
  Value output = function.getArgument(abi.output);
  while (!entry.empty())
    entry.back().erase();

  OpBuilder builder(function.getContext());
  Location loc = function.getLoc();
  builder.setInsertionPointToEnd(&entry);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value segmentCount =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                 function.getArgument(abi.segmentCount));
  scf::ForOp::create(
      builder, loc, zero, segmentCount, one, ValueRange(),
      [&](OpBuilder &outer, Location outerLoc, Value segmentId, ValueRange) {
        Value startI32 =
            memref::LoadOp::create(outer, outerLoc, offsets, segmentId);
        Value next = arith::AddIOp::create(outer, outerLoc, segmentId, one);
        Value endI32 = memref::LoadOp::create(outer, outerLoc, offsets, next);
        Value start = arith::IndexCastOp::create(
            outer, outerLoc, outer.getIndexType(), startI32);
        Value end = arith::IndexCastOp::create(outer, outerLoc,
                                               outer.getIndexType(), endI32);
        SmallVector<Value> results;
        for (const ReductionStage &stage : program.reductions) {
          Value identity = identityFor(outer, outerLoc, stage.kind);
          auto reduction = scf::ForOp::create(
              outer, outerLoc, start, end, one, ValueRange(identity),
              [&](OpBuilder &inner, Location innerLoc, Value index,
                  ValueRange accumulator) {
                Value value =
                    memref::LoadOp::create(inner, innerLoc, values, index);
                value = evaluateElement(inner, stage.element, value, results);
                scf::YieldOp::create(inner, innerLoc,
                                     combine(inner, innerLoc, stage.kind,
                                             accumulator.front(), value));
              });
          results.push_back(reduction.getResult(0));
        }
        if (program.terminal == TerminalKind::ScalarStore) {
          memref::StoreOp::create(outer, outerLoc,
                                  results[program.storedReduction], output,
                                  segmentId);
        } else {
          scf::ForOp::create(
              outer, outerLoc, start, end, one, ValueRange(),
              [&](OpBuilder &inner, Location innerLoc, Value index,
                  ValueRange) {
                Value value =
                    memref::LoadOp::create(inner, innerLoc, values, index);
                value =
                    evaluateElement(inner, program.mapStore, value, results);
                memref::StoreOp::create(inner, innerLoc, value, output, index);
                scf::YieldOp::create(inner, innerLoc);
              });
        }
        scf::YieldOp::create(outer, outerLoc);
      });
  func::ReturnOp::create(builder, loc);
  // The roles are consumed. What remains is ordinary upstream IR, which a
  // tool that does not know the Swage dialect must be able to parse.
  for (unsigned index = 0; index < function.getNumArguments(); ++index)
    function.removeArgAttr(index,
                           StringAttr::get(function.getContext(),
                                           SwageDialect::getRoleAttrName()));
}

void buildGPUProgram(ModuleOp module, func::FuncOp source,
                     const SegmentProgram &program,
                     const TargetDescription &target, int64_t blockSize,
                     bool useTaskIds, bool fusedMixed, bool persistent) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(builder, loc,
                                            source.getName().str() + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  // The task-ID, fused, and persistent kernels load segment IDs from a task
  // buffer. The direct kernel uses the block index as the segment ID.
  using swage_plan::KernelArgument;
  swage_plan::KernelKind kind = swage_plan::KernelKind::Direct;
  if (persistent)
    kind = swage_plan::KernelKind::Persistent;
  else if (fusedMixed)
    kind = swage_plan::KernelKind::FusedMixed;
  else if (useTaskIds)
    kind = swage_plan::KernelKind::TaskIds;
  const swage_plan::KernelLayout layout = swage_plan::kernelLayout(kind);
  SmallVector<Type> inputs;
  for (KernelArgument parameter : layout.arguments())
    inputs.push_back(swage_plan::isBuffer(parameter) ? pointer : i32);
  auto kernelType = FunctionType::get(module.getContext(), inputs, {});
  auto kernel =
      gpu::GPUFuncOp::create(builder, loc, source.getName(), kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  target.pinLaunchWidth(kernel, static_cast<int32_t>(blockSize));
  Value claimBroadcast;
  if (persistent) {
    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        module.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto broadcastType = MemRefType::get({2}, i32, AffineMap(), workgroupSpace);
    claimBroadcast = kernel.addWorkgroupAttribution(broadcastType, loc);
  }

  Block *entry = &kernel.getBody().front();
  // The entry block may carry workgroup attributions after the parameters,
  // so a parameter is found through the layout and never from the end.
  auto argument = [&](KernelArgument parameter) {
    return Value(entry->getArgument(layout.indexOf(parameter)));
  };
  builder.setInsertionPointToStart(entry);
  Value taskIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value valueCount = argument(KernelArgument::ValueCount);
  // The direct kernel compares the segment count with the block index. The
  // others bound each loaded segment ID with it.
  Value segmentCount = argument(KernelArgument::SegmentCount);
  auto loadTaskIndex = [&](OpBuilder &body, Location bodyLoc, Value words,
                           Value wordIndex) {
    Value word = loadTaskWord(body, bodyLoc, words, wordIndex);
    return Value(
        arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(), word));
  };
  // Reduce one segment for the thread `logicalThreadId`: bind its range,
  // run one stage per reduction in program order, then the terminal.
  // `segmentInRange` is null when the segment ID is the block index, and the
  // result of `isLoadedIndexInRange` for an ID loaded from a task buffer.
  auto emitSegment = [&](OpBuilder &body, Location bodyLoc, Value segmentId,
                         Value segmentInRange, Value logicalThreadId,
                         Value stride, bool useWarpShuffle) {
    BoundSegment bound = emitSegmentBinding(
        body, bodyLoc, argument(KernelArgument::Values),
        argument(KernelArgument::Offsets), valueCount, segmentId,
        segmentInRange, logicalThreadId, stride, zero, one);
    SmallVector<Value> results;
    for (const ReductionStage &stage : program.reductions)
      results.push_back(emitReductionStage(
          body, bodyLoc, target, stage.kind, f32, bound.segment, useWarpShuffle,
          [&](OpBuilder &loop, Value value) {
            return evaluateElement(loop, stage.element, value, results);
          }));
    if (program.terminal == TerminalKind::ScalarStore) {
      emitScalarStore(body, bodyLoc, results[program.storedReduction],
                      argument(KernelArgument::Output), bound.segmentId64,
                      logicalThreadId, zero, segmentInRange);
      return;
    }
    emitMapStore(
        body, bodyLoc, f32, bound.segment, argument(KernelArgument::Output),
        [&](OpBuilder &loop, Value value) {
          return evaluateElement(loop, program.mapStore, value, results);
        });
  };

  auto emitTaskSegment = [&](OpBuilder &body, Location bodyLoc, Value taskIds,
                             Value taskId, Value logicalThreadId, Value stride,
                             bool useWarpShuffle) {
    Value segmentIdI32 = loadTaskWord(body, bodyLoc, taskIds, taskId);
    Value segmentInRange =
        isLoadedIndexInRange(body, bodyLoc, segmentIdI32, segmentCount);
    Value segmentId = arith::IndexCastOp::create(
        body, bodyLoc, body.getIndexType(), segmentIdI32);
    emitSegment(body, bodyLoc, segmentId, segmentInRange, logicalThreadId,
                stride, useWarpShuffle);
  };

  if (persistent) {
    Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value oneI32 = arith::ConstantIntOp::create(builder, loc, 1, 32);
    // The batch a block claims from the partial queue, and the batch a
    // subgroup claims from the warp queue.
    Value fourI32 = arith::ConstantIntOp::create(
        builder, loc, target.persistentPartialClaim, 32);
    Value eightI32 = arith::ConstantIntOp::create(
        builder, loc, target.persistentWarpClaim, 32);
    Value firstThread = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, threadId, zero);

    auto claim = [&](OpBuilder &body, Location claimLoc, int64_t counterIndex,
                     Value leader, bool warpBroadcast, Value increment) {
      Value counterOffset =
          arith::ConstantIntOp::create(body, claimLoc, counterIndex, 64);
      Value counterAddress = LLVM::GEPOp::create(
          body, claimLoc, pointer, i32, argument(KernelArgument::Counters),
          counterOffset);
      auto leaderClaim = scf::IfOp::create(body, claimLoc, TypeRange{i32},
                                           leader, /*withElseRegion=*/true);
      body.setInsertionPointToStart(&leaderClaim.getThenRegion().front());
      Value claimed = LLVM::AtomicRMWOp::create(
          body, claimLoc, LLVM::AtomicBinOp::add, counterAddress, increment,
          LLVM::AtomicOrdering::monotonic);
      scf::YieldOp::create(body, claimLoc, claimed);
      body.setInsertionPointToStart(&leaderClaim.getElseRegion().front());
      scf::YieldOp::create(body, claimLoc, zeroI32);
      body.setInsertionPointAfter(leaderClaim);
      if (warpBroadcast) {
        auto shuffled =
            gpu::ShuffleOp::create(body, claimLoc, leaderClaim.getResult(0), 0,
                                   target.subgroupWidth, gpu::ShuffleMode::IDX);
        return shuffled.getShuffleResult();
      }
      scf::IfOp::create(
          body, claimLoc, leader, [&](OpBuilder &store, Location storeLoc) {
            memref::StoreOp::create(store, storeLoc, leaderClaim.getResult(0),
                                    claimBroadcast, zero);
            scf::YieldOp::create(store, storeLoc);
          });
      gpu::BarrierOp::create(body, claimLoc);
      return Value(memref::LoadOp::create(body, claimLoc, claimBroadcast,
                                          ValueRange{zero}));
    };

    // CTA tasks are claimed first so a long tail can overlap subsequent
    // short-segment work. A CTA that observes the queue empty moves on while
    // another CTA may still be reducing its final long segment.
    Value firstCTA = claim(builder, loc, 1, firstThread, false, oneI32);
    Value ctaTaskCount = argument(KernelArgument::CtaTaskCount);
    auto ctaLoop = scf::WhileOp::create(builder, loc, TypeRange{i32},
                                        ValueRange{firstCTA});
    Block *ctaBefore =
        builder.createBlock(&ctaLoop.getBefore(), {}, {i32}, {loc});
    builder.setInsertionPointToEnd(ctaBefore);
    Value hasCTA =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                              ctaBefore->getArgument(0), ctaTaskCount);
    scf::ConditionOp::create(builder, loc, hasCTA, ctaBefore->getArguments());
    Block *ctaAfter =
        builder.createBlock(&ctaLoop.getAfter(), {}, {i32}, {loc});
    builder.setInsertionPointToEnd(ctaAfter);
    Value ctaTask = ctaAfter->getArgument(0);
    Value ctaTaskIndex = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), ctaTask);
    emitTaskSegment(builder, loc, argument(KernelArgument::CtaIds),
                    ctaTaskIndex, threadId, block, false);
    gpu::BarrierOp::create(builder, loc);
    Value nextCTA = claim(builder, loc, 1, firstThread, false, oneI32);
    scf::YieldOp::create(builder, loc, nextCTA);
    builder.setInsertionPointAfter(ctaLoop);
    // The terminating CTA claim is still read from the shared broadcast slot.
    // Keep the first partial-queue claim from overwriting that slot until every
    // CTA thread has consumed the terminating value.
    gpu::BarrierOp::create(builder, loc);

    // Split partials share one queue. Each partial writes a unique scratch
    // slot, then publishes completion with an acquire-release atomic. The
    // CTA observing the final completion performs the only merge and output
    // store for that dependency group, so workers never spin or deadlock.
    Value firstPartial = claim(builder, loc, 2, firstThread, false, fourI32);
    Value partialTaskCount = argument(KernelArgument::PartialCount);
    auto partialLoop = scf::WhileOp::create(builder, loc, TypeRange{i32},
                                            ValueRange{firstPartial});
    Block *partialBefore =
        builder.createBlock(&partialLoop.getBefore(), {}, {i32}, {loc});
    builder.setInsertionPointToEnd(partialBefore);
    Value hasPartial =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                              partialBefore->getArgument(0), partialTaskCount);
    scf::ConditionOp::create(builder, loc, hasPartial,
                             partialBefore->getArguments());
    Block *partialAfter =
        builder.createBlock(&partialLoop.getAfter(), {}, {i32}, {loc});
    builder.setInsertionPointToEnd(partialAfter);
    Value partialBatch = partialAfter->getArgument(0);
    Value partialBatchEnd =
        arith::AddIOp::create(builder, loc, partialBatch, fourI32);
    Value boundedPartialBatchEnd =
        arith::MinUIOp::create(builder, loc, partialBatchEnd, partialTaskCount);
    Value partialBatchIndex = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), partialBatch);
    Value partialBatchEndIndex = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), boundedPartialBatchEnd);
    auto partialBatchLoop = scf::ForOp::create(builder, loc, partialBatchIndex,
                                               partialBatchEndIndex, one);
    builder.setInsertionPointToStart(partialBatchLoop.getBody());
    Value partialIndex = partialBatchLoop.getInductionVar();
    Value two = arith::ConstantIndexOp::create(builder, loc, 2);
    Value partialBase = arith::MulIOp::create(builder, loc, partialIndex, two);
    Value partialEndIndex =
        arith::AddIOp::create(builder, loc, partialBase, one);
    Value partialBeginI32 = loadTaskWord(
        builder, loc, argument(KernelArgument::PartialRanges), partialBase);
    Value partialEndI32 = loadTaskWord(
        builder, loc, argument(KernelArgument::PartialRanges), partialEndIndex);
    // Partial ranges index the values buffer, so they take the same bound as
    // the direct ranges.
    Value partialBegin;
    Value partialEnd;
    std::tie(partialBegin, partialEnd) =
        clampRange(builder, loc, partialBeginI32, partialEndI32, valueCount);
    Value partialFirst =
        arith::AddIOp::create(builder, loc, partialBegin, threadId);
    Value partialIdentity = identityFor(builder, loc, ReductionKind::Sum);
    auto partialReduction = scf::ForOp::create(
        builder, loc, partialFirst, partialEnd, block,
        ValueRange(partialIdentity),
        [&](OpBuilder &loop, Location loopLoc, Value index,
            ValueRange accumulator) {
          Value index64 = arith::IndexCastOp::create(loop, loopLoc,
                                                     loop.getI64Type(), index);
          Value address =
              LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                  argument(KernelArgument::Values), index64);
          Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
          scf::YieldOp::create(loop, loopLoc,
                               combine(loop, loopLoc, ReductionKind::Sum,
                                       accumulator.front(), value));
        });
    auto add = gpu::AllReduceOperationAttr::get(module.getContext(),
                                                gpu::AllReduceOperation::ADD);
    Value partialTotal = gpu::AllReduceOp::create(
        builder, loc, partialReduction.getResult(0), add, true);
    scf::IfOp::create(
        builder, loc, firstThread, [&](OpBuilder &store, Location storeLoc) {
          Value partialIndex64 = arith::IndexCastOp::create(
              store, storeLoc, store.getI64Type(), partialIndex);
          Value scratchAddress = LLVM::GEPOp::create(
              store, storeLoc, pointer, f32, argument(KernelArgument::Scratch),
              partialIndex64);
          LLVM::StoreOp::create(store, storeLoc, partialTotal, scratchAddress);
          scf::YieldOp::create(store, storeLoc);
        });
    gpu::BarrierOp::create(builder, loc);

    // Only the leader reads dependency metadata and publishes completion.
    // It writes either the ready merge ID or -1 into a separate shared slot;
    // the CTA barrier makes that decision uniform without rereading merge
    // descriptors in every lane for every non-final partial.
    //
    // The merge ID addresses the completion counter and the merge record, so
    // the merge count bounds it first. A partial whose merge ID is out of
    // range updates no counter, reads no record, and publishes -1. This
    // branch runs in the leader alone and holds no barrier.
    Value completionSlot = one;
    Value mergeCount = argument(KernelArgument::MergeCount);
    scf::IfOp::create(
        builder, loc, firstThread,
        [&](OpBuilder &publish, Location publishLoc) {
          Value mergeIdI32 = loadTaskWord(
              publish, publishLoc, argument(KernelArgument::PartialMergeIds),
              partialIndex);
          Value mergeInRange =
              isLoadedIndexInRange(publish, publishLoc, mergeIdI32, mergeCount);
          auto published = scf::IfOp::create(publish, publishLoc,
                                             TypeRange{i32}, mergeInRange,
                                             /*withElseRegion=*/true);
          publish.setInsertionPointToStart(&published.getThenRegion().front());
          Value mergeId = arith::IndexCastOp::create(
              publish, publishLoc, publish.getIndexType(), mergeIdI32);
          Value completionBase =
              arith::ConstantIndexOp::create(publish, publishLoc, 3);
          Value completionIndex = arith::AddIOp::create(
              publish, publishLoc, completionBase, mergeId);
          Value completionIndex64 = arith::IndexCastOp::create(
              publish, publishLoc, publish.getI64Type(), completionIndex);
          Value completionAddress = LLVM::GEPOp::create(
              publish, publishLoc, pointer, i32,
              argument(KernelArgument::Counters), completionIndex64);

          Value three = arith::ConstantIndexOp::create(publish, publishLoc, 3);
          Value mergeBase =
              arith::MulIOp::create(publish, publishLoc, mergeId, three);
          Value mergeBeginIndex =
              arith::AddIOp::create(publish, publishLoc, mergeBase, one);
          Value mergeEndIndex =
              arith::AddIOp::create(publish, publishLoc, mergeBeginIndex, one);
          Value mergeBegin = loadTaskIndex(
              publish, publishLoc, argument(KernelArgument::MergeRecords),
              mergeBeginIndex);
          Value mergeEnd = loadTaskIndex(publish, publishLoc,
                                         argument(KernelArgument::MergeRecords),
                                         mergeEndIndex);
          Value expectedPartials =
              arith::SubIOp::create(publish, publishLoc, mergeEnd, mergeBegin);
          Value expectedPartialsI32 = arith::IndexCastOp::create(
              publish, publishLoc, publish.getI32Type(), expectedPartials);

          // NVPTX lowers the LLVM atomic to the legacy atom form on sm_86.
          // Make scratch publication explicit before exposing completion.
          target.emitDeviceFence(publish, publishLoc);
          Value previousCompletion = LLVM::AtomicRMWOp::create(
              publish, publishLoc, LLVM::AtomicBinOp::add, completionAddress,
              oneI32, LLVM::AtomicOrdering::acq_rel);
          Value completed = arith::AddIOp::create(publish, publishLoc,
                                                  previousCompletion, oneI32);
          Value isLastPartial = arith::CmpIOp::create(
              publish, publishLoc, arith::CmpIPredicate::eq, completed,
              expectedPartialsI32);
          auto readyMerge = scf::IfOp::create(publish, publishLoc,
                                              TypeRange{i32}, isLastPartial,
                                              /*withElseRegion=*/true);
          publish.setInsertionPointToStart(&readyMerge.getThenRegion().front());
          scf::YieldOp::create(publish, publishLoc, mergeIdI32);
          publish.setInsertionPointToStart(&readyMerge.getElseRegion().front());
          Value noMerge =
              arith::ConstantIntOp::create(publish, publishLoc, -1, 32);
          scf::YieldOp::create(publish, publishLoc, noMerge);
          publish.setInsertionPointAfter(readyMerge);
          scf::YieldOp::create(publish, publishLoc, readyMerge.getResult(0));
          publish.setInsertionPointToStart(&published.getElseRegion().front());
          Value skippedMerge =
              arith::ConstantIntOp::create(publish, publishLoc, -1, 32);
          scf::YieldOp::create(publish, publishLoc, skippedMerge);
          publish.setInsertionPointAfter(published);
          memref::StoreOp::create(publish, publishLoc, published.getResult(0),
                                  claimBroadcast, completionSlot);
          scf::YieldOp::create(publish, publishLoc);
        });
    gpu::BarrierOp::create(builder, loc);
    Value readyMergeI32 = memref::LoadOp::create(builder, loc, claimBroadcast,
                                                 ValueRange{completionSlot});
    Value hasReadyMerge = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::sge, readyMergeI32, zeroI32);

    scf::IfOp::create(
        builder, loc, hasReadyMerge, [&](OpBuilder &merge, Location mergeLoc) {
          // Pair with every partial publisher before any lane reads scratch.
          target.emitDeviceFence(merge, mergeLoc);
          Value mergeId = arith::IndexCastOp::create(
              merge, mergeLoc, merge.getIndexType(), readyMergeI32);
          Value three = arith::ConstantIndexOp::create(merge, mergeLoc, 3);
          Value mergeBase =
              arith::MulIOp::create(merge, mergeLoc, mergeId, three);
          Value mergeBeginIndex =
              arith::AddIOp::create(merge, mergeLoc, mergeBase, one);
          Value mergeEndIndex =
              arith::AddIOp::create(merge, mergeLoc, mergeBeginIndex, one);
          // The merge ID is in range here: the leader published it only
          // after the bound above. The output segment it names is loaded
          // from the record, so the segment count bounds it before the
          // store. The merge itself stays unconditional, which keeps its
          // all-reduce under the block-uniform guard alone.
          Value outputSegmentI32 =
              loadTaskWord(merge, mergeLoc,
                           argument(KernelArgument::MergeRecords), mergeBase);
          Value outputInRange = isLoadedIndexInRange(
              merge, mergeLoc, outputSegmentI32, segmentCount);
          Value outputSegment = arith::IndexCastOp::create(
              merge, mergeLoc, merge.getIndexType(), outputSegmentI32);
          Value mergeBeginI32 = loadTaskWord(
              merge, mergeLoc, argument(KernelArgument::MergeRecords),
              mergeBeginIndex);
          Value mergeEndI32 = loadTaskWord(
              merge, mergeLoc, argument(KernelArgument::MergeRecords),
              mergeEndIndex);
          // Merge ranges index scratch, which holds one slot per partial, so
          // the partial count bounds them.
          Value mergeBegin;
          Value mergeEnd;
          std::tie(mergeBegin, mergeEnd) = clampRange(
              merge, mergeLoc, mergeBeginI32, mergeEndI32, partialTaskCount);
          Value mergeFirst =
              arith::AddIOp::create(merge, mergeLoc, mergeBegin, threadId);
          Value mergeIdentity =
              identityFor(merge, mergeLoc, ReductionKind::Sum);
          auto mergeReduction = scf::ForOp::create(
              merge, mergeLoc, mergeFirst, mergeEnd, block,
              ValueRange(mergeIdentity),
              [&](OpBuilder &loop, Location loopLoc, Value index,
                  ValueRange accumulator) {
                Value index64 = arith::IndexCastOp::create(
                    loop, loopLoc, loop.getI64Type(), index);
                Value address = LLVM::GEPOp::create(
                    loop, loopLoc, pointer, f32,
                    argument(KernelArgument::Scratch), index64);
                Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
                scf::YieldOp::create(loop, loopLoc,
                                     combine(loop, loopLoc, ReductionKind::Sum,
                                             accumulator.front(), value));
              });
          Value mergeTotal = gpu::AllReduceOp::create(
              merge, mergeLoc, mergeReduction.getResult(0), add, true);
          Value mayStore = arith::AndIOp::create(merge, mergeLoc, firstThread,
                                                 outputInRange);
          scf::IfOp::create(
              merge, mergeLoc, mayStore,
              [&](OpBuilder &store, Location storeLoc) {
                Value outputIndex64 = arith::IndexCastOp::create(
                    store, storeLoc, store.getI64Type(), outputSegment);
                Value outputAddress = LLVM::GEPOp::create(
                    store, storeLoc, pointer, f32,
                    argument(KernelArgument::Output), outputIndex64);
                LLVM::StoreOp::create(store, storeLoc, mergeTotal,
                                      outputAddress);
                scf::YieldOp::create(store, storeLoc);
              });
          scf::YieldOp::create(merge, mergeLoc);
        });
    builder.setInsertionPointAfter(partialBatchLoop);
    Value nextPartial = claim(builder, loc, 2, firstThread, false, fourI32);
    scf::YieldOp::create(builder, loc, nextPartial);
    builder.setInsertionPointAfter(partialLoop);

    // Each physical warp independently drains the short-task queue. Only
    // lane zero performs the atomic claim and broadcasts the result within
    // that warp, so the sixteen workers do not require CTA-wide lockstep.
    Value warp =
        arith::ConstantIndexOp::create(builder, loc, target.subgroupWidth);
    Value lane = arith::RemUIOp::create(builder, loc, threadId, warp);
    Value firstLane = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, lane, zero);
    Value firstWarp = claim(builder, loc, 0, firstLane, true, eightI32);
    Value warpTaskCount = argument(KernelArgument::WarpTaskCount);
    auto warpLoop = scf::WhileOp::create(builder, loc, TypeRange{i32},
                                         ValueRange{firstWarp});
    Block *warpBefore =
        builder.createBlock(&warpLoop.getBefore(), {}, {i32}, {loc});
    builder.setInsertionPointToEnd(warpBefore);
    Value hasWarp =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult,
                              warpBefore->getArgument(0), warpTaskCount);
    scf::ConditionOp::create(builder, loc, hasWarp, warpBefore->getArguments());
    Block *warpAfter =
        builder.createBlock(&warpLoop.getAfter(), {}, {i32}, {loc});
    builder.setInsertionPointToEnd(warpAfter);
    Value warpBatch = warpAfter->getArgument(0);
    Value warpBatchEnd =
        arith::AddIOp::create(builder, loc, warpBatch, eightI32);
    Value boundedWarpBatchEnd =
        arith::MinUIOp::create(builder, loc, warpBatchEnd, warpTaskCount);
    Value warpBatchIndex = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), warpBatch);
    Value warpBatchEndIndex = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), boundedWarpBatchEnd);
    auto warpBatchLoop = scf::ForOp::create(builder, loc, warpBatchIndex,
                                            warpBatchEndIndex, one);
    builder.setInsertionPointToStart(warpBatchLoop.getBody());
    Value warpTaskIndex = warpBatchLoop.getInductionVar();
    emitTaskSegment(builder, loc, argument(KernelArgument::WarpIds),
                    warpTaskIndex, lane, warp, true);
    builder.setInsertionPointAfter(warpBatchLoop);
    Value nextWarp = claim(builder, loc, 0, firstLane, true, eightI32);
    scf::YieldOp::create(builder, loc, nextWarp);
    builder.setInsertionPointAfter(warpLoop);

    gpu::ReturnOp::create(builder, loc);
    source.erase();
    return;
  }

  if (fusedMixed) {
    // One warp task per subgroup of the block: `four` slots, and `three` to
    // round the warp task count up to whole blocks.
    int64_t slots = target.slotsPerBlock(static_cast<int32_t>(blockSize));
    Value three = arith::ConstantIndexOp::create(builder, loc, slots - 1);
    Value four = arith::ConstantIndexOp::create(builder, loc, slots);
    Value warp =
        arith::ConstantIndexOp::create(builder, loc, target.subgroupWidth);
    Value warpTaskCount =
        arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                   argument(KernelArgument::WarpTaskCount));
    Value ctaTaskCount =
        arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                   argument(KernelArgument::CtaTaskCount));
    Value roundedWarpTaskCount =
        arith::AddIOp::create(builder, loc, warpTaskCount, three);
    Value warpBlockCount =
        arith::DivUIOp::create(builder, loc, roundedWarpTaskCount, four);
    Value isWarpBlock = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ult, taskIndex, warpBlockCount);
    scf::IfOp::create(
        builder, loc, isWarpBlock,
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
                              emitTaskSegment(task, taskLoc,
                                              argument(KernelArgument::TaskIds),
                                              warpTaskId, lane, warp, true);
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
                              Value mixedTaskId = arith::AddIOp::create(
                                  task, taskLoc, warpTaskCount, ctaTaskId);
                              emitTaskSegment(task, taskLoc,
                                              argument(KernelArgument::TaskIds),
                                              mixedTaskId, threadId, block,
                                              false);
                              scf::YieldOp::create(task, taskLoc);
                            });
          scf::YieldOp::create(ctaBlock, ctaLoc);
        });
    gpu::ReturnOp::create(builder, loc);
    source.erase();
    return;
  }

  // One block per task with a task buffer, and one per segment without.
  Value taskCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      argument(useTaskIds ? KernelArgument::TaskCount
                          : KernelArgument::SegmentCount));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        taskIndex, taskCount);

  scf::IfOp::create(
      builder, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
        if (useTaskIds)
          emitTaskSegment(body, bodyLoc, argument(KernelArgument::TaskIds),
                          taskIndex, threadId, block,
                          blockSize == target.subgroupWidth);
        else
          emitSegment(body, bodyLoc, taskIndex, Value(), threadId, block,
                      false);

        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
}

void buildSplitGPUProgram(ModuleOp module, func::FuncOp source,
                          const ReductionStage &stage,
                          const TargetDescription &target, bool merge) {
  const int64_t blockSize = target.splitBlockThreads;
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  std::string suffix = merge ? "__merge" : "__partial";
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(
      builder, loc, source.getName().str() + suffix + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  // A merge loads its output segment from a record, so its parameters end
  // with the segment count that bounds it. A partial writes the scratch slot
  // of its own block index.
  using swage_plan::KernelArgument;
  const swage_plan::KernelLayout layout =
      swage_plan::kernelLayout(merge ? swage_plan::KernelKind::SplitMerge
                                     : swage_plan::KernelKind::SplitPartial);
  SmallVector<Type> inputs;
  for (KernelArgument parameter : layout.arguments())
    inputs.push_back(swage_plan::isBuffer(parameter) ? pointer : i32);
  auto kernelType = FunctionType::get(module.getContext(), inputs, {});
  auto kernel = gpu::GPUFuncOp::create(
      builder, loc, source.getName().str() + suffix, kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  target.pinLaunchWidth(kernel, static_cast<int32_t>(blockSize));

  Block *entry = &kernel.getBody().front();
  auto argument = [&](KernelArgument parameter) {
    return Value(entry->getArgument(layout.indexOf(parameter)));
  };
  // A partial reduces a range of values into scratch. A merge reduces a
  // range of scratch into the output.
  Value reduced =
      argument(merge ? KernelArgument::Scratch : KernelArgument::Values);
  Value reducedCount = argument(merge ? KernelArgument::PartialCount
                                      : KernelArgument::ValueCount);
  Value records = argument(merge ? KernelArgument::MergeRecords
                                 : KernelArgument::PartialRanges);
  Value sink =
      argument(merge ? KernelArgument::Output : KernelArgument::Scratch);
  builder.setInsertionPointToStart(entry);
  Value taskIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value taskCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      argument(merge ? KernelArgument::MergeCount
                     : KernelArgument::PartialCount));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        taskIndex, taskCount);

  scf::IfOp::create(
      builder, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
        Value fields =
            arith::ConstantIndexOp::create(body, bodyLoc, merge ? 3 : 2);
        Value recordBase =
            arith::MulIOp::create(body, bodyLoc, taskIndex, fields);
        Value recordPointer = records;
        auto loadRecord = [&](int64_t field) {
          Value index = recordBase;
          if (field)
            index = arith::AddIOp::create(
                body, bodyLoc, recordBase,
                arith::ConstantIndexOp::create(body, bodyLoc, field));
          Value index64 = arith::IndexCastOp::create(body, bodyLoc,
                                                     body.getI64Type(), index);
          Value address = LLVM::GEPOp::create(body, bodyLoc, pointer, i32,
                                              recordPointer, index64);
          return Value(LLVM::LoadOp::create(body, bodyLoc, i32, address));
        };

        // A merge loads its output segment from the record, so the segment
        // count bounds it before the store below. The merge itself stays
        // unconditional, which keeps its all-reduce under the block-uniform
        // guard alone.
        Value outputIndex = taskIndex;
        Value outputInRange;
        int64_t rangeField = 0;
        if (merge) {
          Value outputSegmentI32 = loadRecord(0);
          outputInRange =
              isLoadedIndexInRange(body, bodyLoc, outputSegmentI32,
                                   argument(KernelArgument::SegmentCount));
          outputIndex = arith::IndexCastOp::create(
              body, bodyLoc, body.getIndexType(), outputSegmentI32);
          rangeField = 1;
        }
        // The range indexes the buffer this kernel reduces, and that
        // buffer's count bounds it: values and the value count for a
        // partial, scratch and the partial count for a merge.
        Value beginI32 = loadRecord(rangeField);
        Value endI32 = loadRecord(rangeField + 1);
        Value begin;
        Value end;
        std::tie(begin, end) =
            clampRange(body, bodyLoc, beginI32, endI32, reducedCount);
        Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
        Value identity = identityFor(body, bodyLoc, stage.kind);
        auto local = scf::ForOp::create(
            body, bodyLoc, first, end, block, ValueRange(identity),
            [&](OpBuilder &loop, Location loopLoc, Value index,
                ValueRange accumulator) {
              Value index64 = arith::IndexCastOp::create(
                  loop, loopLoc, loop.getI64Type(), index);
              Value address = LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                                  reduced, index64);
              Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
              // Only input elements are transformed; scratch holds completed
              // partial reductions and must never run the element program.
              if (!merge)
                value = evaluateElement(loop, stage.element, value, {});
              scf::YieldOp::create(loop, loopLoc,
                                   combine(loop, loopLoc, stage.kind,
                                           accumulator.front(), value));
            });
        auto operation = gpu::AllReduceOperationAttr::get(
            module.getContext(), allReduceOperationFor(stage.kind));
        Value total = gpu::AllReduceOp::create(
            body, bodyLoc, local.getResult(0), operation, true);
        Value mayStore = arith::CmpIOp::create(
            body, bodyLoc, arith::CmpIPredicate::eq, threadId, zero);
        if (outputInRange)
          mayStore =
              arith::AndIOp::create(body, bodyLoc, mayStore, outputInRange);
        scf::IfOp::create(
            body, bodyLoc, mayStore, [&](OpBuilder &store, Location storeLoc) {
              Value outputIndex64 = arith::IndexCastOp::create(
                  store, storeLoc, store.getI64Type(), outputIndex);
              Value outputAddress = LLVM::GEPOp::create(
                  store, storeLoc, pointer, f32, sink, outputIndex64);
              LLVM::StoreOp::create(store, storeLoc, total, outputAddress);
              scf::YieldOp::create(store, storeLoc);
            });
        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
}

class SegmentedReductionToSCFPass
    : public PassWrapper<SegmentedReductionToSCFPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SegmentedReductionToSCFPass)

  SegmentedReductionToSCFPass() = default;
  SegmentedReductionToSCFPass(const SegmentedReductionToSCFPass &other)
      : PassWrapper(other) {
    selectedFunction = other.selectedFunction.getValue();
  }

  StringRef getArgument() const final {
    return "swage-segmented-reduction-to-scf";
  }
  StringRef getDescription() const final {
    return "Lower every segment function to sequential SCF loops";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    FailureOr<SmallVector<func::FuncOp>> functions =
        findSegmentFunctions(getOperation(), selectedFunction);
    if (failed(functions))
      return signalPassFailure();
    // Every function is admitted before any is changed, so a rejected module
    // is left as it was.
    SmallVector<SegmentProgramAnalysis, 1> analyses(functions->size());
    for (auto [function, analysis] : llvm::zip(*functions, analyses))
      if (failed(analyzeSegmentProgram(function, analysis)))
        return signalPassFailure();
    for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
      // The owner must outlive the program, which points into it.
      RegionOwner owner;
      SegmentProgram program;
      detachSegmentProgram(analysis, owner, program);
      buildSequentialProgram(function, analysis.abi, program);
    }
  }

private:
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Lower only this function instead of every function "
                     "that holds Swage operations")};
};

class SegmentedReductionToGPUPass
    : public PassWrapper<SegmentedReductionToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SegmentedReductionToGPUPass)

  SegmentedReductionToGPUPass() = default;
  SegmentedReductionToGPUPass(const SegmentedReductionToGPUPass &other)
      : PassWrapper(other), target(other.target) {
    blockSize = other.blockSize.getValue();
    useTaskIds = other.useTaskIds.getValue();
    fusedMixed = other.fusedMixed.getValue();
    persistent = other.persistent.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }
  SegmentedReductionToGPUPass(const TargetDescription &target,
                              int64_t requestedBlockSize, bool requestedTaskIds,
                              bool requestedFusedMixed,
                              bool requestedPersistent, StringRef function)
      : target(&target) {
    blockSize = requestedBlockSize;
    useTaskIds = requestedTaskIds;
    fusedMixed = requestedFusedMixed;
    persistent = requestedPersistent;
    selectedFunction = function.str();
  }

  StringRef getArgument() const final {
    return "swage-segmented-reduction-to-gpu";
  }
  StringRef getDescription() const final {
    return "Lower every segment function to a GPU kernel: one block per "
           "segment, or the task-id, fused mixed, or persistent schedule an "
           "option selects";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (blockSize <= 0) {
      getOperation().emitError()
          << "block-size must be a positive integer, got "
          << blockSize.getValue();
      return signalPassFailure();
    }
    if (blockSize > target->maxBlockThreads) {
      getOperation().emitError()
          << "block-size must be at most " << target->maxBlockThreads
          << ", got " << blockSize.getValue();
      return signalPassFailure();
    }
    if (fusedMixed && blockSize != target->ctaBlockThreads) {
      getOperation().emitError()
          << "fused mixed lowering requires block-size "
          << target->ctaBlockThreads << ", got " << blockSize.getValue();
      return signalPassFailure();
    }
    if (persistent && blockSize != target->persistentBlockThreads) {
      getOperation().emitError()
          << "persistent lowering requires block-size "
          << target->persistentBlockThreads << ", got " << blockSize.getValue();
      return signalPassFailure();
    }
    // The persistent and fused kernels have ABIs of their own and always
    // load segment IDs from their task buffers, so neither can honor the
    // task-ID ABI option.
    if (persistent && useTaskIds) {
      getOperation().emitError(
          "persistent lowering does not accept use-task-ids; the persistent "
          "kernel always loads segment IDs from its own task queues");
      return signalPassFailure();
    }
    if (fusedMixed && useTaskIds) {
      getOperation().emitError(
          "fused mixed lowering does not accept use-task-ids; the fused "
          "kernel always loads segment IDs from its own task buffer");
      return signalPassFailure();
    }
    if (!target->admitsBlockThreads(blockSize)) {
      // Read the option first: streaming the option object itself prints its
      // value as a character.
      int64_t requested = blockSize;
      getOperation().emitError()
          << "block-size must give a power-of-two warp count, got " << requested
          << " (" << target->subgroupCount(requested) << " warps)";
      return signalPassFailure();
    }
    ModuleOp module = getOperation();
    FailureOr<SmallVector<func::FuncOp>> functions =
        findSegmentFunctions(module, selectedFunction);
    if (failed(functions))
      return signalPassFailure();
    // Every function is admitted before any is changed, so a rejected module
    // is left as it was.
    SmallVector<SegmentProgramAnalysis, 1> analyses(functions->size());
    for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
      if (failed(analyzeSegmentProgram(function, analysis)))
        return signalPassFailure();
      if ((useTaskIds || fusedMixed || persistent) &&
          failed(verifyPlanningProgram(analysis)))
        return signalPassFailure();
      if (persistent && failed(verifyPersistentProgram(analysis)))
        return signalPassFailure();
      if (failed(verifyKernelSymbols(module, function, "")))
        return signalPassFailure();
    }
    for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
      RegionOwner owner;
      SegmentProgram program;
      detachSegmentProgram(analysis, owner, program);
      buildGPUProgram(module, function, program, *target, blockSize, useTaskIds,
                      fusedMixed, persistent);
    }
  }

private:
  const TargetDescription *target = &nvidiaTarget();
  Option<int64_t> blockSize{*this, "block-size",
                            llvm::cl::desc("CTA x block size"),
                            llvm::cl::init(0)};
  Option<bool> useTaskIds{
      *this, "use-task-ids",
      llvm::cl::desc("Load segment IDs through the internal task ABI"),
      llvm::cl::init(false)};
  Option<bool> fusedMixed{
      *this, "fused-mixed",
      llvm::cl::desc("Fuse warp and CTA task schedules into one kernel"),
      llvm::cl::init(false)};
  Option<bool> persistent{
      *this, "persistent",
      llvm::cl::desc("Emit the experimental private persistent queue kernel; "
                     "requires block-size 512"),
      llvm::cl::init(false)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Lower only this function instead of every function "
                     "that holds Swage operations")};
};

class SplitSegmentedReductionToGPUPass
    : public PassWrapper<SplitSegmentedReductionToGPUPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitSegmentedReductionToGPUPass)

  SplitSegmentedReductionToGPUPass() = default;
  SplitSegmentedReductionToGPUPass(
      const SplitSegmentedReductionToGPUPass &other)
      : PassWrapper(other) {
    merge = other.merge.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }
  SplitSegmentedReductionToGPUPass(bool requestedMerge, StringRef function) {
    merge = requestedMerge;
    selectedFunction = function.str();
  }

  StringRef getArgument() const final {
    return "swage-split-segmented-reduction-to-gpu";
  }
  StringRef getDescription() const final {
    return "Lower every capture-free sum or max function to a private split "
           "stage";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    FailureOr<SmallVector<func::FuncOp>> functions =
        findSegmentFunctions(module, selectedFunction);
    if (failed(functions))
      return signalPassFailure();
    // Every function is admitted before any is changed, so a rejected module
    // is left as it was.
    SmallVector<SegmentProgramAnalysis, 1> analyses(functions->size());
    for (auto [function, analysis] : llvm::zip(*functions, analyses))
      if (failed(analyzeSegmentProgram(function, analysis)) ||
          failed(verifyPlanningProgram(analysis)) ||
          failed(verifyKernelSymbols(module, function,
                                     merge ? "__merge" : "__partial")))
        return signalPassFailure();
    for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
      RegionOwner owner;
      SegmentProgram program;
      detachSegmentProgram(analysis, owner, program);
      buildSplitGPUProgram(module, function, program.reductions.front(),
                           nvidiaTarget(), merge);
    }
  }

private:
  Option<bool> merge{
      *this, "merge",
      llvm::cl::desc("Emit the merge stage over scratch partials instead of "
                     "the partial stage over input ranges"),
      llvm::cl::init(false)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Lower only this function instead of every function "
                     "that holds Swage operations")};
};

class SwageToPlanPass
    : public PassWrapper<SwageToPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwageToPlanPass)

  SwageToPlanPass() = default;
  SwageToPlanPass(const SwageToPlanPass &other) : PassWrapper(other) {
    warpMaxElements = other.warpMaxElements.getValue();
    ctaChunkElements = other.ctaChunkElements.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }
  SwageToPlanPass(int64_t requestedWarpMaxElements,
                  int64_t requestedCtaChunkElements, StringRef function) {
    warpMaxElements = requestedWarpMaxElements;
    ctaChunkElements = requestedCtaChunkElements;
    selectedFunction = function.str();
  }

  StringRef getArgument() const final { return "swage-to-plan"; }
  StringRef getDescription() const final {
    return "Add runtime classification for every capture-free sum or max "
           "function";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<func::FuncDialect, swage_plan::SwagePlanDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (warpMaxElements <= 0 || ctaChunkElements <= 0 ||
        warpMaxElements > ctaChunkElements ||
        ctaChunkElements > std::numeric_limits<int32_t>::max()) {
      module.emitError("planning limits must satisfy 0 < warp-max-elements <= "
                       "cta-chunk-elements <= INT32_MAX");
      return signalPassFailure();
    }

    FailureOr<SmallVector<func::FuncOp>> functions =
        findSegmentFunctions(module, selectedFunction);
    if (failed(functions))
      return signalPassFailure();
    // Every function is admitted before any companion is added, so a
    // rejected module is left as it was.
    SmallVector<SegmentProgramAnalysis, 1> analyses(functions->size());
    for (auto [function, analysis] : llvm::zip(*functions, analyses)) {
      std::string companionName = function.getName().str() + "__swage_plan";
      if (SymbolTable::lookupSymbolIn(module.getOperation(), companionName)) {
        module.emitError() << "planning companion symbol @" << companionName
                           << " already exists";
        return signalPassFailure();
      }
      if (failed(analyzeSegmentProgram(function, analysis)) ||
          failed(verifyPlanningProgram(analysis)))
        return signalPassFailure();
    }
    for (auto [function, analysis] : llvm::zip(*functions, analyses))
      buildPlanningCompanion(module, function, analysis.abi,
                             static_cast<int32_t>(warpMaxElements),
                             static_cast<int32_t>(ctaChunkElements));
  }

private:
  Option<int64_t> warpMaxElements{
      *this, "warp-max-elements",
      llvm::cl::desc("Maximum segment length admitted for warp policy"),
      llvm::cl::init(nvidiaTarget().defaultWarpMaxElements)};
  Option<int64_t> ctaChunkElements{
      *this, "cta-chunk-elements",
      llvm::cl::desc("Maximum input elements in one CTA task"),
      llvm::cl::init(nvidiaTarget().defaultCtaChunkElements)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Plan only this function instead of every function that "
                     "holds Swage operations")};
};

} // namespace

std::unique_ptr<Pass> createSegmentedReductionToSCFPass() {
  return std::make_unique<SegmentedReductionToSCFPass>();
}

std::unique_ptr<Pass> createSegmentedReductionToGPUPass(int64_t blockSize,
                                                        bool useTaskIds,
                                                        bool fusedMixed,
                                                        StringRef function) {
  return std::make_unique<SegmentedReductionToGPUPass>(
      nvidiaTarget(), blockSize, useTaskIds, fusedMixed, false, function);
}

std::unique_ptr<Pass>
createSegmentedReductionToGPUPass(int64_t blockSize, bool useTaskIds,
                                  bool fusedMixed,
                                  const TargetDescription &target) {
  return std::make_unique<SegmentedReductionToGPUPass>(
      target, blockSize, useTaskIds, fusedMixed, false, "");
}

std::unique_ptr<Pass>
createPersistentSegmentedReductionToGPUPass(StringRef function) {
  const TargetDescription &target = nvidiaTarget();
  return std::make_unique<SegmentedReductionToGPUPass>(
      target, target.persistentBlockThreads, false, false, true, function);
}

std::unique_ptr<Pass> createSplitPartialReductionToGPUPass(StringRef function) {
  return std::make_unique<SplitSegmentedReductionToGPUPass>(false, function);
}

std::unique_ptr<Pass> createSplitMergeReductionToGPUPass(StringRef function) {
  return std::make_unique<SplitSegmentedReductionToGPUPass>(true, function);
}

std::unique_ptr<Pass> createSwageToPlanPass(int64_t warpMaxElements,
                                            int64_t ctaChunkElements,
                                            StringRef function) {
  return std::make_unique<SwageToPlanPass>(warpMaxElements, ctaChunkElements,
                                           function);
}

void registerSegmentedReductionPasses() {
  PassRegistration<SegmentedReductionToSCFPass>();
  PassRegistration<SegmentedReductionToGPUPass>();
  PassRegistration<SplitSegmentedReductionToGPUPass>();
  PassRegistration<SwageToPlanPass>();
}

} // namespace mlir::swage
