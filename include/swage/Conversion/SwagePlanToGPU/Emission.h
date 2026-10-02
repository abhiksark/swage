// include/swage/Conversion/SwagePlanToGPU/Emission.h
//===- Emission.h - Kernel emission helpers --------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// The pieces every segmented kernel is built from: the bounded range of a
// segment, one reduction stage, and the two sinks. Every conversion
// pattern calls the same functions, so a reduction or a store has the same
// text in every kernel. This header is internal to the Swage conversions.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SWAGEPLANTOGPU_EMISSION_H
#define SWAGE_CONVERSION_SWAGEPLANTOGPU_EMISSION_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"

#include <utility>

namespace mlir::swage {

struct TargetDescription;

/// The elements of one segment that one thread reads: the element at `first`
/// and every `stride`-th element after it below `end`, in the buffer at
/// `base`. A consumer of a bound segment needs these four values and
/// nothing else.
///
/// A kernel addresses its buffers through pointers. The sequential oracle
/// keeps them as memrefs. Loads and stores follow the type of the buffer
/// they are given.
struct SegmentBinding {
  Value base;   ///< The values buffer: a pointer in a kernel, or a memref.
  Value first;  ///< Index of the first element this thread reads.
  Value end;    ///< Index one past the last element of the segment.
  Value stride; ///< Index distance between two elements of one thread.
};

/// A bound segment and the index of its output slot.
struct BoundSegment {
  SegmentBinding segment;
  Value segmentId64; ///< The segment ID as i64, for the scalar store.
};

/// How the threads that ran one reduction loop combine their accumulators.
enum class ThreadCombination {
  /// One thread ran the whole loop; there is nothing to combine.
  None,
  /// A shuffle tree over the lanes of one subgroup.
  Subgroup,
  /// A block-wide reduction.
  Block,
};

/// Applies an element program to one loaded element at the insertion point
/// of the builder it is given.
using ElementProgramFn = function_ref<Value(OpBuilder &, Value)>;

/// Clone a verified region inline at the builder's insertion point and
/// return the mapped yielded value.
Value inlineRegion(OpBuilder &builder, Region &region, ValueRange arguments);

// What a reduction kind lowers to. A new kind is a case in each of these
// three functions. Each is a switch over every kind without a default, so a
// kind without a case is a compiler diagnostic and never the lowering of
// another kind.

/// The identity element of a reduction kind.
Value identityFor(OpBuilder &builder, Location loc, ReductionKind kind);
/// Combine an accumulator with one element.
Value combine(OpBuilder &builder, Location loc, ReductionKind kind,
              Value accumulator, Value value);
/// The block-wide reduction that combines the accumulators of all threads.
gpu::AllReduceOperation allReduceOperationFor(ReductionKind kind);

/// Clamp one half-open range loaded from device memory, as signed i32, so
/// that `0 <= start <= end <= length`, and return it as index values.
/// `length` is the i32 element count of the buffer the range indexes: the
/// value count for a range into values, the partial count for a range into
/// scratch.
///
/// Host validation sees a snapshot of the range, but the kernel reloads it at
/// every launch, so a range that changed after validation would otherwise
/// index outside that buffer. A decreasing pair becomes an empty range. For
/// validated ranges both clamps are the identity.
std::pair<Value, Value> clampRange(OpBuilder &builder, Location loc,
                                   Value startI32, Value endI32, Value length);

/// Whether `word`, an i32 index loaded from device memory, names one of
/// `count` elements, where `count` is an i32 count that the ABI carries. The
/// comparison is unsigned, so a negative word fails it too.
///
/// Host validation sees a snapshot of the buffer the index came from, but the
/// kernel reloads it at every launch. This applies to a segment ID from a task
/// buffer, the merge ID of a persistent partial, and the output segment of a
/// merge record. The caller skips an index that fails this test: no store or
/// counter update takes it. It is not clamped, because a clamped index would
/// store a wrong result in a valid slot. For validated indices the test is
/// always true.
Value isLoadedIndexInRange(OpBuilder &builder, Location loc, Value word,
                           Value count);

/// Load the i32 word at index `wordIndex` of the task buffer `words`.
Value loadTaskWord(OpBuilder &builder, Location loc, Value words,
                   Value wordIndex);

/// Load one field of a task record: the i32 word at index
/// `recordBase + field` of the task buffer `records`, where `recordBase` is
/// the word index of the record. The fields of each record are in
/// `TaskRecords.h`.
Value loadRecordField(OpBuilder &builder, Location loc, Value records,
                      Value recordBase, unsigned field);

/// Store `total` at `sink[slot]` from the thread whose `threadId` is zero,
/// and only when `slotInRange`, if given, holds. `slot` is an index; the
/// thread that stores converts it.
void emitLeaderStore(OpBuilder &builder, Location loc, Value total, Value sink,
                     Value slot, Value threadId, Value zero, Value slotInRange);

/// Bind segment `segmentId` for the thread `logicalThreadId`: load its range
/// from `offsets`, clamp it to `valueCount`, and give the thread its first
/// element.
///
/// `segmentInRange` is null when the segment ID is the block index, which
/// the caller already compared with the segment count. For a segment ID
/// loaded from a task buffer it is the result of `isLoadedIndexInRange`. The
/// bound is applied to the addresses and not to the control flow: an
/// out-of-range ID reads offsets[0] for both ends of its range, which makes
/// the range empty. Every thread therefore still reaches each barrier and
/// shuffle of the stages that follow, whatever the task buffer holds.
///
/// `zero` and `one` are the index constants of the kernel prelude.
BoundSegment emitSegmentBinding(OpBuilder &builder, Location loc, Value values,
                                Value offsets, Value valueCount,
                                Value segmentId, Value segmentInRange,
                                Value logicalThreadId, Value stride, Value zero,
                                Value one);

/// Reduce the bound segment with `kind`: every thread folds its elements,
/// then the threads combine their accumulators as `combination` says.
/// Returns the result, which every thread holds. `target` gives the width
/// of a subgroup and is read for `ThreadCombination::Subgroup` only, so it
/// may be null otherwise.
Value emitReductionStage(OpBuilder &builder, Location loc,
                         const TargetDescription *target, ReductionKind kind,
                         Type elementType, const SegmentBinding &segment,
                         ThreadCombination combination,
                         ElementProgramFn element);

/// Store `total` at `output[segmentId64]` from the thread whose
/// `logicalThreadId` is zero, and only when `segmentInRange`, if given,
/// holds.
void emitScalarStore(OpBuilder &builder, Location loc, Value total,
                     Value output, Value segmentId64, Value logicalThreadId,
                     Value zero, Value segmentInRange);

/// Write the element program of every element of the bound segment to the
/// same index of `output`, which is a pointer or a memref.
void emitMapStore(OpBuilder &builder, Location loc, Type elementType,
                  const SegmentBinding &segment, Value output,
                  ElementProgramFn element);

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SWAGEPLANTOGPU_EMISSION_H
