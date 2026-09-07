// lib/Conversion/SegmentedReduction/GPU.cpp
//===- GPU.cpp - GPU segmented emitters -----------------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "SegmentProgram.h"

#include <cstdint>
#include <string>
#include <vector>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "swage/Support/KernelContract.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;

namespace mlir::swage::detail {
namespace {

std::vector<KernelArgument>
getSegmentedKernelArguments(const SemanticBufferRoles &roles,
                            SegmentedExecutionKind kind) {
  using Access = KernelArgumentAccess;
  using Origin = KernelArgumentOrigin;
  std::vector<KernelArgument> arguments{
      KernelArgument::userPointer(roles.valuesSourceIndex, Access::Read),
      KernelArgument::userPointer(roles.offsetsSourceIndex, Access::Read),
      KernelArgument::userPointer(roles.outputSourceIndex, Access::Write)};
  switch (kind) {
  case SegmentedExecutionKind::Persistent:
    arguments.insert(
        arguments.end(),
        {KernelArgument::keyedPointer(Origin::Plan, "warp_task_ids",
                                      Access::Read),
         KernelArgument::keyedPointer(Origin::Plan, "cta_task_ids",
                                      Access::Read),
         KernelArgument::keyedPointer(Origin::Plan, "partial_ranges",
                                      Access::Read),
         KernelArgument::keyedPointer(Origin::Plan, "partial_merge_ids",
                                      Access::Read),
         KernelArgument::keyedPointer(Origin::Plan, "merge_ranges",
                                      Access::Read),
         KernelArgument::keyedPointer(Origin::Scratch, "partials",
                                      Access::ReadWrite),
         KernelArgument::keyedPointer(Origin::Scratch, "counters",
                                      Access::ReadWrite),
         KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                     "value_count"),
         KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                     "warp_task_count"),
         KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                     "cta_task_count"),
         KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                     "partial_task_count"),
         KernelArgument::keyedScalar(KernelArgumentKind::I32, Origin::Derived,
                                     "merge_task_count")});
    return arguments;
  case SegmentedExecutionKind::FusedMixed:
  case SegmentedExecutionKind::TaskIds:
    arguments.push_back(
        KernelArgument::keyedPointer(Origin::Plan, "task_ids", Access::Read));
    break;
  case SegmentedExecutionKind::Direct:
    break;
  }
  arguments.push_back(KernelArgument::keyedScalar(
      KernelArgumentKind::I32, Origin::Derived, "value_count"));
  switch (kind) {
  case SegmentedExecutionKind::FusedMixed:
    arguments.push_back(KernelArgument::keyedScalar(
        KernelArgumentKind::I32, Origin::Derived, "warp_task_count"));
    arguments.push_back(KernelArgument::keyedScalar(
        KernelArgumentKind::I32, Origin::Derived, "cta_task_count"));
    break;
  case SegmentedExecutionKind::TaskIds:
    arguments.push_back(KernelArgument::keyedScalar(
        KernelArgumentKind::I32, Origin::Derived, "task_count"));
    break;
  case SegmentedExecutionKind::Direct:
    arguments.push_back(KernelArgument::keyedScalar(
        KernelArgumentKind::I32, Origin::Derived, "segment_count"));
    break;
  case SegmentedExecutionKind::Persistent:
    llvm_unreachable("persistent arguments returned above");
  }
  return arguments;
}

} // namespace

void buildGPUProgram(ModuleOp module, func::FuncOp source,
                     const SemanticBufferRoles &roles,
                     const SegmentProgram &program, int64_t blockSize,
                     SegmentedExecutionKind kind) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(builder, loc,
                                            source.getName().str() + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  std::vector<KernelArgument> arguments =
      getSegmentedKernelArguments(roles, kind);
  auto kernelType = FunctionType::get(
      module.getContext(),
      getKernelArgumentTypes(module.getContext(), arguments), {});
  auto kernel =
      gpu::GPUFuncOp::create(builder, loc, source.getName(), kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  KernelContract contract;
  contract.backend = KernelBackend::CUDA;
  contract.entry = source.getName().str();
  contract.launch = {
      KernelLaunchModel::SPMDGrid,
      std::array<int32_t, 3>{static_cast<int32_t>(blockSize), 1, 1}};
  contract.arguments = arguments;
  kernel->setDiscardableAttr(kernelContractAttrName,
                             buildKernelContractAttr(builder, contract));
  kernel->setAttr(NVVM::NVVMDialect::getReqntidAttrName(),
                  builder.getDenseI32ArrayAttr(*contract.launch.block));
  Value claimBroadcast;
  if (kind == SegmentedExecutionKind::Persistent) {
    auto workgroupSpace = gpu::AddressSpaceAttr::get(
        module.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
    auto broadcastType = MemRefType::get({2}, i32, AffineMap(), workgroupSpace);
    claimBroadcast = kernel.addWorkgroupAttribution(broadcastType, loc);
  }

  Block *entry = &kernel.getBody().front();
  builder.setInsertionPointToStart(entry);
  Value taskIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  auto loadSegmentId = [&](OpBuilder &body, Location bodyLoc, Value taskIds,
                           Value taskId) {
    Value taskId64 =
        arith::IndexCastOp::create(body, bodyLoc, body.getI64Type(), taskId);
    Value taskAddress =
        LLVM::GEPOp::create(body, bodyLoc, pointer, i32, taskIds, taskId64);
    Value segmentIdI32 = LLVM::LoadOp::create(body, bodyLoc, i32, taskAddress);
    return Value(arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(),
                                            segmentIdI32));
  };
  auto emitSegment = [&](OpBuilder &body, Location bodyLoc, Value segmentId,
                         Value logicalThreadId, Value stride,
                         bool useWarpShuffle) {
    Value segmentId64 =
        arith::IndexCastOp::create(body, bodyLoc, body.getI64Type(), segmentId);
    Value startAddress = LLVM::GEPOp::create(
        body, bodyLoc, pointer, i32, entry->getArgument(1), segmentId64);
    Value startI32 = LLVM::LoadOp::create(body, bodyLoc, i32, startAddress);
    Value nextSegment = arith::AddIOp::create(body, bodyLoc, segmentId, one);
    Value nextSegment64 = arith::IndexCastOp::create(
        body, bodyLoc, body.getI64Type(), nextSegment);
    Value endAddress = LLVM::GEPOp::create(
        body, bodyLoc, pointer, i32, entry->getArgument(1), nextSegment64);
    Value endI32 = LLVM::LoadOp::create(body, bodyLoc, i32, endAddress);
    Value start = arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(),
                                             startI32);
    Value end =
        arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(), endI32);
    Value first = arith::AddIOp::create(body, bodyLoc, start, logicalThreadId);

    SmallVector<Value> results;
    for (const ReductionStage &stage : program.reductions) {
      Value identity = identityFor(body, bodyLoc, stage.kind);
      auto local = scf::ForOp::create(
          body, bodyLoc, first, end, stride, ValueRange(identity),
          [&](OpBuilder &loop, Location loopLoc, Value index,
              ValueRange accumulator) {
            Value index64 = arith::IndexCastOp::create(
                loop, loopLoc, loop.getI64Type(), index);
            Value address = LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                                entry->getArgument(0), index64);
            Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
            value = evaluateElement(loop, stage.element, value, results);
            scf::YieldOp::create(
                loop, loopLoc,
                combine(loop, loopLoc, stage.kind, accumulator.front(), value));
          });
      gpu::AllReduceOperation gpuKind = stage.kind == ReductionKind::Sum
                                            ? gpu::AllReduceOperation::ADD
                                            : gpu::AllReduceOperation::MAXIMUMF;
      Value total = local.getResult(0);
      if (useWarpShuffle) {
        for (int32_t offset = 1; offset < 32; offset <<= 1) {
          auto shuffled = gpu::ShuffleOp::create(body, bodyLoc, total, offset,
                                                 32, gpu::ShuffleMode::XOR);
          total = combine(body, bodyLoc, stage.kind, total,
                          shuffled.getShuffleResult());
        }
      } else {
        auto operation =
            gpu::AllReduceOperationAttr::get(module.getContext(), gpuKind);
        // uniform = true, so the result is broadcast to every thread and the
        // lowering's trailing barrier fences this stage from the next.
        total = gpu::AllReduceOp::create(body, bodyLoc, total, operation, true);
      }
      results.push_back(total);
    }

    if (program.terminal == TerminalKind::ScalarStore) {
      Value total = results[program.storedReduction];
      Value firstThread = arith::CmpIOp::create(
          body, bodyLoc, arith::CmpIPredicate::eq, logicalThreadId, zero);
      scf::IfOp::create(
          body, bodyLoc, firstThread, [&](OpBuilder &store, Location storeLoc) {
            Value outputAddress =
                LLVM::GEPOp::create(store, storeLoc, pointer, f32,
                                    entry->getArgument(2), segmentId64);
            LLVM::StoreOp::create(store, storeLoc, total, outputAddress);
            scf::YieldOp::create(store, storeLoc);
          });
      return;
    }

    // Guard-free on purpose: every thread runs the same block-stride loop it
    // ran for each reduction stage, and an empty segment makes it zero-trip.
    // A thread-dependent guard here would put a predicate around code the
    // barriers above already made CTA-uniform.
    scf::ForOp::create(
        body, bodyLoc, first, end, stride, ValueRange(),
        [&](OpBuilder &loop, Location loopLoc, Value index, ValueRange) {
          Value index64 = arith::IndexCastOp::create(loop, loopLoc,
                                                     loop.getI64Type(), index);
          Value address = LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                              entry->getArgument(0), index64);
          Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
          value = evaluateElement(loop, program.mapStore, value, results);
          Value outputAddress = LLVM::GEPOp::create(
              loop, loopLoc, pointer, f32, entry->getArgument(2), index64);
          LLVM::StoreOp::create(loop, loopLoc, value, outputAddress);
          scf::YieldOp::create(loop, loopLoc);
        });
  };

  if (kind == SegmentedExecutionKind::Persistent) {
    Value zeroI32 = arith::ConstantIntOp::create(builder, loc, 0, 32);
    Value oneI32 = arith::ConstantIntOp::create(builder, loc, 1, 32);
    Value fourI32 = arith::ConstantIntOp::create(builder, loc, 4, 32);
    Value eightI32 = arith::ConstantIntOp::create(builder, loc, 8, 32);
    Value firstThread = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, threadId, zero);

    auto claim = [&](OpBuilder &body, Location claimLoc, int64_t counterIndex,
                     Value leader, bool warpBroadcast, Value increment) {
      Value counterOffset =
          arith::ConstantIntOp::create(body, claimLoc, counterIndex, 64);
      Value counterAddress = LLVM::GEPOp::create(
          body, claimLoc, pointer, i32, entry->getArgument(9), counterOffset);
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
                                   32, gpu::ShuffleMode::IDX);
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
    Value ctaTaskCount = entry->getArgument(12);
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
    Value ctaSegment =
        loadSegmentId(builder, loc, entry->getArgument(4), ctaTaskIndex);
    emitSegment(builder, loc, ctaSegment, threadId, block, false);
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
    Value partialTaskCount = entry->getArgument(13);
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
    Value partialBegin =
        loadSegmentId(builder, loc, entry->getArgument(5), partialBase);
    Value partialEnd =
        loadSegmentId(builder, loc, entry->getArgument(5), partialEndIndex);
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
          Value address = LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                              entry->getArgument(0), index64);
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
          Value scratchAddress =
              LLVM::GEPOp::create(store, storeLoc, pointer, f32,
                                  entry->getArgument(8), partialIndex64);
          LLVM::StoreOp::create(store, storeLoc, partialTotal, scratchAddress);
          scf::YieldOp::create(store, storeLoc);
        });
    gpu::BarrierOp::create(builder, loc);

    // Only the leader reads dependency metadata and publishes completion.
    // It writes either the ready merge ID or -1 into a separate shared slot;
    // the CTA barrier makes that decision uniform without rereading merge
    // descriptors in every lane for every non-final partial.
    Value completionSlot = one;
    scf::IfOp::create(
        builder, loc, firstThread,
        [&](OpBuilder &publish, Location publishLoc) {
          Value mergeId = loadSegmentId(publish, publishLoc,
                                        entry->getArgument(6), partialIndex);
          Value completionBase =
              arith::ConstantIndexOp::create(publish, publishLoc, 3);
          Value completionIndex = arith::AddIOp::create(
              publish, publishLoc, completionBase, mergeId);
          Value completionIndex64 = arith::IndexCastOp::create(
              publish, publishLoc, publish.getI64Type(), completionIndex);
          Value completionAddress =
              LLVM::GEPOp::create(publish, publishLoc, pointer, i32,
                                  entry->getArgument(9), completionIndex64);

          Value three = arith::ConstantIndexOp::create(publish, publishLoc, 3);
          Value mergeBase =
              arith::MulIOp::create(publish, publishLoc, mergeId, three);
          Value mergeBeginIndex =
              arith::AddIOp::create(publish, publishLoc, mergeBase, one);
          Value mergeEndIndex =
              arith::AddIOp::create(publish, publishLoc, mergeBeginIndex, one);
          Value mergeBegin = loadSegmentId(
              publish, publishLoc, entry->getArgument(7), mergeBeginIndex);
          Value mergeEnd = loadSegmentId(publish, publishLoc,
                                         entry->getArgument(7), mergeEndIndex);
          Value expectedPartials =
              arith::SubIOp::create(publish, publishLoc, mergeEnd, mergeBegin);
          Value expectedPartialsI32 = arith::IndexCastOp::create(
              publish, publishLoc, publish.getI32Type(), expectedPartials);

          // NVPTX lowers the LLVM atomic to the legacy atom form on sm_86.
          // Make scratch publication explicit before exposing completion.
          NVVM::MembarOp::create(publish, publishLoc, NVVM::MemScopeKind::GPU);
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
          Value mergeIdI32 = arith::IndexCastOp::create(
              publish, publishLoc, publish.getI32Type(), mergeId);
          scf::YieldOp::create(publish, publishLoc, mergeIdI32);
          publish.setInsertionPointToStart(&readyMerge.getElseRegion().front());
          Value noMerge =
              arith::ConstantIntOp::create(publish, publishLoc, -1, 32);
          scf::YieldOp::create(publish, publishLoc, noMerge);
          publish.setInsertionPointAfter(readyMerge);
          memref::StoreOp::create(publish, publishLoc, readyMerge.getResult(0),
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
          NVVM::MembarOp::create(merge, mergeLoc, NVVM::MemScopeKind::GPU);
          Value mergeId = arith::IndexCastOp::create(
              merge, mergeLoc, merge.getIndexType(), readyMergeI32);
          Value three = arith::ConstantIndexOp::create(merge, mergeLoc, 3);
          Value mergeBase =
              arith::MulIOp::create(merge, mergeLoc, mergeId, three);
          Value mergeBeginIndex =
              arith::AddIOp::create(merge, mergeLoc, mergeBase, one);
          Value mergeEndIndex =
              arith::AddIOp::create(merge, mergeLoc, mergeBeginIndex, one);
          Value outputSegment =
              loadSegmentId(merge, mergeLoc, entry->getArgument(7), mergeBase);
          Value mergeBegin = loadSegmentId(
              merge, mergeLoc, entry->getArgument(7), mergeBeginIndex);
          Value mergeEnd = loadSegmentId(merge, mergeLoc, entry->getArgument(7),
                                         mergeEndIndex);
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
                Value address =
                    LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                        entry->getArgument(8), index64);
                Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
                scf::YieldOp::create(loop, loopLoc,
                                     combine(loop, loopLoc, ReductionKind::Sum,
                                             accumulator.front(), value));
              });
          Value mergeTotal = gpu::AllReduceOp::create(
              merge, mergeLoc, mergeReduction.getResult(0), add, true);
          scf::IfOp::create(
              merge, mergeLoc, firstThread,
              [&](OpBuilder &store, Location storeLoc) {
                Value outputIndex64 = arith::IndexCastOp::create(
                    store, storeLoc, store.getI64Type(), outputSegment);
                Value outputAddress =
                    LLVM::GEPOp::create(store, storeLoc, pointer, f32,
                                        entry->getArgument(2), outputIndex64);
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
    Value warp = arith::ConstantIndexOp::create(builder, loc, 32);
    Value lane = arith::RemUIOp::create(builder, loc, threadId, warp);
    Value firstLane = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::eq, lane, zero);
    Value firstWarp = claim(builder, loc, 0, firstLane, true, eightI32);
    Value warpTaskCount = entry->getArgument(11);
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
    Value warpSegment =
        loadSegmentId(builder, loc, entry->getArgument(3), warpTaskIndex);
    emitSegment(builder, loc, warpSegment, lane, warp, true);
    builder.setInsertionPointAfter(warpBatchLoop);
    Value nextWarp = claim(builder, loc, 0, firstLane, true, eightI32);
    scf::YieldOp::create(builder, loc, nextWarp);
    builder.setInsertionPointAfter(warpLoop);

    gpu::ReturnOp::create(builder, loc);
    source.erase();
    return;
  }

  if (kind == SegmentedExecutionKind::FusedMixed) {
    Value three = arith::ConstantIndexOp::create(builder, loc, 3);
    Value four = arith::ConstantIndexOp::create(builder, loc, 4);
    Value warp = arith::ConstantIndexOp::create(builder, loc, 32);
    Value warpTaskCount = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), entry->getArgument(5));
    Value ctaTaskCount = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), entry->getArgument(6));
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
          scf::IfOp::create(
              warpBlock, warpLoc, inRange,
              [&](OpBuilder &task, Location taskLoc) {
                Value segmentId = loadSegmentId(
                    task, taskLoc, entry->getArgument(3), warpTaskId);
                emitSegment(task, taskLoc, segmentId, lane, warp, true);
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
          scf::IfOp::create(
              ctaBlock, ctaLoc, inRange,
              [&](OpBuilder &task, Location taskLoc) {
                Value mixedTaskId = arith::AddIOp::create(
                    task, taskLoc, warpTaskCount, ctaTaskId);
                Value segmentId = loadSegmentId(
                    task, taskLoc, entry->getArgument(3), mixedTaskId);
                emitSegment(task, taskLoc, segmentId, threadId, block, false);
                scf::YieldOp::create(task, taskLoc);
              });
          scf::YieldOp::create(ctaBlock, ctaLoc);
        });
    gpu::ReturnOp::create(builder, loc);
    source.erase();
    return;
  }

  Value taskCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(),
      entry->getArgument(kind == SegmentedExecutionKind::TaskIds ? 5 : 4));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        taskIndex, taskCount);

  scf::IfOp::create(
      builder, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
        Value segmentId = taskIndex;
        if (kind == SegmentedExecutionKind::TaskIds)
          segmentId =
              loadSegmentId(body, bodyLoc, entry->getArgument(3), taskIndex);
        emitSegment(body, bodyLoc, segmentId, threadId, block,
                    kind == SegmentedExecutionKind::TaskIds && blockSize == 32);

        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
}

} // namespace mlir::swage::detail
