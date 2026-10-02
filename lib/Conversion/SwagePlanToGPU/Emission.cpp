// lib/Conversion/SwagePlanToGPU/Emission.cpp
//===- Emission.cpp - Kernel emission helpers -----------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwagePlanToGPU/Emission.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "swage/Target/TargetDescription.h"

namespace mlir::swage {

Value inlineRegion(OpBuilder &builder, Region &region, ValueRange arguments) {
  Block &body = region.front();
  IRMapping mapping;
  mapping.map(body.getArguments(), arguments);
  for (Operation &operation : body.without_terminator())
    builder.clone(operation, mapping);
  return mapping.lookup(cast<YieldOp>(body.getTerminator()).getValue());
}

Value identityFor(OpBuilder &builder, Location loc, ReductionKind kind) {
  FloatType f32 = builder.getF32Type();
  APFloat identity = kind == ReductionKind::Sum
                         ? APFloat(f32.getFloatSemantics(), 0)
                         : APFloat::getInf(f32.getFloatSemantics(), true);
  return arith::ConstantFloatOp::create(builder, loc, f32, identity);
}

Value combine(OpBuilder &builder, Location loc, ReductionKind kind,
              Value accumulator, Value value) {
  if (kind == ReductionKind::Sum)
    return arith::AddFOp::create(builder, loc, accumulator, value).getResult();
  return arith::MaximumFOp::create(builder, loc, accumulator, value)
      .getResult();
}

gpu::AllReduceOperation allReduceOperationFor(ReductionKind kind) {
  return kind == ReductionKind::Sum ? gpu::AllReduceOperation::ADD
                                    : gpu::AllReduceOperation::MAXIMUMF;
}

std::pair<Value, Value> clampRange(OpBuilder &builder, Location loc,
                                   Value startI32, Value endI32, Value length) {
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value startFloored = arith::MaxSIOp::create(builder, loc, startI32, zero);
  Value startClamped =
      arith::MinSIOp::create(builder, loc, startFloored, length);
  Value endFloored = arith::MaxSIOp::create(builder, loc, endI32, startClamped);
  Value endClamped = arith::MinSIOp::create(builder, loc, endFloored, length);
  Value start = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                           startClamped);
  Value end = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                         endClamped);
  return {start, end};
}

Value isLoadedIndexInRange(OpBuilder &builder, Location loc, Value word,
                           Value count) {
  return arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ult, word,
                               count);
}

Value loadTaskWord(OpBuilder &builder, Location loc, Value words,
                   Value wordIndex) {
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  Type i32 = builder.getI32Type();
  Value wordIndex64 =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), wordIndex);
  Value wordAddress =
      LLVM::GEPOp::create(builder, loc, pointer, i32, words, wordIndex64);
  return LLVM::LoadOp::create(builder, loc, i32, wordAddress);
}

Value loadRecordField(OpBuilder &builder, Location loc, Value records,
                      Value recordBase, unsigned field) {
  Value index = recordBase;
  if (field)
    index = arith::AddIOp::create(
        builder, loc, recordBase,
        arith::ConstantIndexOp::create(builder, loc, field));
  return loadTaskWord(builder, loc, records, index);
}

void emitLeaderStore(OpBuilder &builder, Location loc, Value total, Value sink,
                     Value slot, Value threadId, Value zero,
                     Value slotInRange) {
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  Value mayStore = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         threadId, zero);
  if (slotInRange)
    mayStore = arith::AndIOp::create(builder, loc, mayStore, slotInRange);
  scf::IfOp::create(
      builder, loc, mayStore, [&](OpBuilder &store, Location storeLoc) {
        Value slot64 = arith::IndexCastOp::create(store, storeLoc,
                                                  store.getI64Type(), slot);
        Value address = LLVM::GEPOp::create(store, storeLoc, pointer,
                                            total.getType(), sink, slot64);
        LLVM::StoreOp::create(store, storeLoc, total, address);
        scf::YieldOp::create(store, storeLoc);
      });
}

BoundSegment emitSegmentBinding(OpBuilder &builder, Location loc, Value values,
                                Value offsets, Value valueCount,
                                Value segmentId, Value segmentInRange,
                                Value logicalThreadId, Value stride, Value zero,
                                Value one) {
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  // The offsets and the counts share one word type.
  Type word = valueCount.getType();
  Value segmentId64 =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), segmentId);
  Value startIndex64 = segmentId64;
  if (segmentInRange) {
    Value startIndex =
        arith::SelectOp::create(builder, loc, segmentInRange, segmentId, zero);
    startIndex64 = arith::IndexCastOp::create(builder, loc,
                                              builder.getI64Type(), startIndex);
  }
  Value startAddress =
      LLVM::GEPOp::create(builder, loc, pointer, word, offsets, startIndex64);
  Value startI32 = LLVM::LoadOp::create(builder, loc, word, startAddress);
  Value endIndex = arith::AddIOp::create(builder, loc, segmentId, one);
  if (segmentInRange)
    endIndex =
        arith::SelectOp::create(builder, loc, segmentInRange, endIndex, zero);
  Value endIndex64 =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), endIndex);
  Value endAddress =
      LLVM::GEPOp::create(builder, loc, pointer, word, offsets, endIndex64);
  Value endI32 = LLVM::LoadOp::create(builder, loc, word, endAddress);
  // The offsets come from the caller's buffer at launch time; bound them
  // here because host validation only saw an earlier snapshot.
  Value start;
  Value end;
  std::tie(start, end) = clampRange(builder, loc, startI32, endI32, valueCount);
  Value first = arith::AddIOp::create(builder, loc, start, logicalThreadId);
  return {{values, first, end, stride}, segmentId64};
}

/// Load the element at `index` of the buffer `base`. A pointer is addressed
/// through `index64`, which is set to the index as i64 for a later store at
/// the same position. A memref is indexed directly.
static Value loadElement(OpBuilder &builder, Location loc, Type elementType,
                         Value base, Value index, Value &index64) {
  if (isa<MemRefType>(base.getType()))
    return memref::LoadOp::create(builder, loc, base, index);
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  index64 =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), index);
  Value address =
      LLVM::GEPOp::create(builder, loc, pointer, elementType, base, index64);
  return LLVM::LoadOp::create(builder, loc, elementType, address);
}

Value emitReductionStage(OpBuilder &builder, Location loc,
                         const TargetDescription *target, ReductionKind kind,
                         Type elementType, const SegmentBinding &segment,
                         ThreadCombination combination,
                         ElementProgramFn element) {
  Value identity = identityFor(builder, loc, kind);
  auto local = scf::ForOp::create(
      builder, loc, segment.first, segment.end, segment.stride,
      ValueRange(identity),
      [&](OpBuilder &loop, Location loopLoc, Value index,
          ValueRange accumulator) {
        Value index64;
        Value value = loadElement(loop, loopLoc, elementType, segment.base,
                                  index, index64);
        value = element(loop, value);
        scf::YieldOp::create(
            loop, loopLoc,
            combine(loop, loopLoc, kind, accumulator.front(), value));
      });
  Value total = local.getResult(0);
  if (combination == ThreadCombination::None)
    return total;
  if (combination == ThreadCombination::Subgroup) {
    for (int32_t offset = 1; offset < target->subgroupWidth; offset <<= 1) {
      auto shuffled =
          gpu::ShuffleOp::create(builder, loc, total, offset,
                                 target->subgroupWidth, gpu::ShuffleMode::XOR);
      total = combine(builder, loc, kind, total, shuffled.getShuffleResult());
    }
    return total;
  }
  auto operation = gpu::AllReduceOperationAttr::get(
      builder.getContext(), allReduceOperationFor(kind));
  // uniform = true, so the result is broadcast to every thread and the
  // lowering's trailing barrier fences this stage from the next.
  return gpu::AllReduceOp::create(builder, loc, total, operation, true);
}

void emitScalarStore(OpBuilder &builder, Location loc, Value total,
                     Value output, Value segmentId64, Value logicalThreadId,
                     Value zero, Value segmentInRange) {
  Type pointer = LLVM::LLVMPointerType::get(builder.getContext());
  Value mayStore = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                         logicalThreadId, zero);
  if (segmentInRange)
    mayStore = arith::AndIOp::create(builder, loc, mayStore, segmentInRange);
  scf::IfOp::create(
      builder, loc, mayStore, [&](OpBuilder &store, Location storeLoc) {
        Value outputAddress = LLVM::GEPOp::create(
            store, storeLoc, pointer, total.getType(), output, segmentId64);
        LLVM::StoreOp::create(store, storeLoc, total, outputAddress);
        scf::YieldOp::create(store, storeLoc);
      });
}

void emitMapStore(OpBuilder &builder, Location loc, Type elementType,
                  const SegmentBinding &segment, Value output,
                  ElementProgramFn element) {
  // Guard-free on purpose: every thread runs the same block-stride loop it
  // ran for each reduction stage, and an empty segment makes it zero-trip.
  // A thread-dependent guard here would put a predicate around code the
  // barriers of the stages already made CTA-uniform. An out-of-range segment
  // ID needs no guard either, because its range is empty.
  scf::ForOp::create(
      builder, loc, segment.first, segment.end, segment.stride, ValueRange(),
      [&](OpBuilder &loop, Location loopLoc, Value index, ValueRange) {
        Value index64;
        Value value = loadElement(loop, loopLoc, elementType, segment.base,
                                  index, index64);
        value = element(loop, value);
        if (isa<MemRefType>(output.getType())) {
          memref::StoreOp::create(loop, loopLoc, value, output, index);
        } else {
          Type pointer = LLVM::LLVMPointerType::get(loop.getContext());
          Value outputAddress = LLVM::GEPOp::create(
              loop, loopLoc, pointer, value.getType(), output, index64);
          LLVM::StoreOp::create(loop, loopLoc, value, outputAddress);
        }
        scf::YieldOp::create(loop, loopLoc);
      });
}

} // namespace mlir::swage
