// include/swage/Dialect/SwagePlan/IR/KernelLayout.h
//===- KernelLayout.h - Parameter layouts of the emitted kernels -*- C++
//-*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// The parameter list of every kernel the plan conversion emits, as data.
// An emitter asks a layout where an argument is instead of counting
// positions, and the host launches with the same order. A kernel takes its
// buffers first, as pointers, and then its counts, as i32.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_DIALECT_SWAGEPLAN_IR_KERNELLAYOUT_H
#define SWAGE_DIALECT_SWAGEPLAN_IR_KERNELLAYOUT_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::swage_plan {

/// The kernels the plan conversion emits.
enum class KernelKind {
  /// One block per segment; the block index is the segment index.
  Direct,
  /// One block per segment of rank-two values; each thread reduces columns.
  DirectColumns,
  /// One block per task; the task buffer names the segment.
  TaskIds,
  /// The row-stripe tile of rank-two values over a task buffer: one block
  /// per task and group of adjacent columns, which the block index names.
  TaskIdsColumns,
  /// Warp tasks and block tasks in one launch.
  FusedMixed,
  /// One partial result per chunk of a long segment.
  SplitPartial,
  /// One result per split segment from its partial results.
  SplitMerge,
  /// The same for a program that needs the extent of the split segment: the
  /// merge also reads the range records of the partial tasks.
  SplitMergeExtent,
  /// Resident blocks that claim tasks from queues.
  Persistent,
};

/// What one kernel parameter is.
enum class KernelArgument {
  // Buffers, passed as pointers.
  Values,
  Offsets,
  Output,
  /// Segment ids, one per task.
  TaskIds,
  /// Segment ids of the warp tasks of a persistent launch.
  WarpIds,
  /// Segment ids of the block tasks of a persistent launch.
  CtaIds,
  /// One [begin, end] pair of value indices per partial task.
  PartialRanges,
  /// The merge record index of every partial task.
  PartialMergeIds,
  /// One [segment, partial_begin, partial_end] triple per split segment.
  MergeRecords,
  /// One f32 slot per partial task.
  Scratch,
  /// The queue claim counters and the completion counter of every merge.
  Counters,

  // Counts, passed as i32.
  ValueCount,
  TaskCount,
  WarpTaskCount,
  CtaTaskCount,
  PartialCount,
  MergeCount,
  SegmentCount,
  /// The number of columns of rank-two values.
  FeatureCount,
};

/// Whether the argument is a buffer, passed as a pointer. Every other
/// argument is a count, passed as i32.
constexpr bool isBuffer(KernelArgument argument) {
  return argument < KernelArgument::ValueCount;
}

/// The name of the argument in documentation and in host code.
constexpr llvm::StringLiteral kernelArgumentName(KernelArgument argument) {
  switch (argument) {
  case KernelArgument::Values:
    return "values";
  case KernelArgument::Offsets:
    return "offsets";
  case KernelArgument::Output:
    return "output";
  case KernelArgument::TaskIds:
    return "task_ids";
  case KernelArgument::WarpIds:
    return "warp_ids";
  case KernelArgument::CtaIds:
    return "cta_ids";
  case KernelArgument::PartialRanges:
    return "partial_ranges";
  case KernelArgument::PartialMergeIds:
    return "partial_merge_ids";
  case KernelArgument::MergeRecords:
    return "merge_records";
  case KernelArgument::Scratch:
    return "scratch";
  case KernelArgument::Counters:
    return "counters";
  case KernelArgument::ValueCount:
    return "value_count";
  case KernelArgument::TaskCount:
    return "task_count";
  case KernelArgument::WarpTaskCount:
    return "warp_task_count";
  case KernelArgument::CtaTaskCount:
    return "cta_task_count";
  case KernelArgument::PartialCount:
    return "partial_count";
  case KernelArgument::MergeCount:
    return "merge_count";
  case KernelArgument::SegmentCount:
    return "segment_count";
  case KernelArgument::FeatureCount:
    return "feature_count";
  }
  llvm_unreachable("unknown kernel argument");
}

/// The parameters of one kernel, in order.
class KernelLayout {
public:
  template <unsigned N>
  constexpr KernelLayout(const KernelArgument (&list)[N])
      : first(list), count(N) {}

  llvm::ArrayRef<KernelArgument> arguments() const { return {first, count}; }
  constexpr unsigned size() const { return count; }

  constexpr bool has(KernelArgument argument) const {
    for (unsigned index = 0; index < count; ++index)
      if (first[index] == argument)
        return true;
    return false;
  }

  /// The position of `argument`, which the kernel must take.
  constexpr unsigned indexOf(KernelArgument argument) const {
    for (unsigned index = 0; index < count; ++index)
      if (first[index] == argument)
        return index;
    llvm_unreachable("the kernel does not take this argument");
  }

private:
  const KernelArgument *first;
  unsigned count;
};

namespace detail {
inline constexpr KernelArgument directArguments[] = {
    KernelArgument::Values, KernelArgument::Offsets, KernelArgument::Output,
    KernelArgument::ValueCount, KernelArgument::SegmentCount};
inline constexpr KernelArgument directColumnsArguments[] = {
    KernelArgument::Values,       KernelArgument::Offsets,
    KernelArgument::Output,       KernelArgument::ValueCount,
    KernelArgument::SegmentCount, KernelArgument::FeatureCount};
inline constexpr KernelArgument taskIdArguments[] = {
    KernelArgument::Values,      KernelArgument::Offsets,
    KernelArgument::Output,      KernelArgument::TaskIds,
    KernelArgument::ValueCount,  KernelArgument::TaskCount,
    KernelArgument::SegmentCount};
inline constexpr KernelArgument taskIdColumnsArguments[] = {
    KernelArgument::Values,       KernelArgument::Offsets,
    KernelArgument::Output,       KernelArgument::TaskIds,
    KernelArgument::ValueCount,   KernelArgument::TaskCount,
    KernelArgument::SegmentCount, KernelArgument::FeatureCount};
inline constexpr KernelArgument fusedMixedArguments[] = {
    KernelArgument::Values,       KernelArgument::Offsets,
    KernelArgument::Output,       KernelArgument::TaskIds,
    KernelArgument::ValueCount,   KernelArgument::WarpTaskCount,
    KernelArgument::CtaTaskCount, KernelArgument::SegmentCount};
inline constexpr KernelArgument splitPartialArguments[] = {
    KernelArgument::Values, KernelArgument::PartialRanges,
    KernelArgument::Scratch, KernelArgument::ValueCount,
    KernelArgument::PartialCount};
inline constexpr KernelArgument splitMergeArguments[] = {
    KernelArgument::Scratch,      KernelArgument::Output,
    KernelArgument::MergeRecords, KernelArgument::PartialCount,
    KernelArgument::MergeCount,   KernelArgument::SegmentCount};
inline constexpr KernelArgument splitMergeExtentArguments[] = {
    KernelArgument::Scratch,      KernelArgument::Output,
    KernelArgument::MergeRecords, KernelArgument::PartialRanges,
    KernelArgument::PartialCount, KernelArgument::MergeCount,
    KernelArgument::SegmentCount};
inline constexpr KernelArgument persistentArguments[] = {
    KernelArgument::Values,          KernelArgument::Offsets,
    KernelArgument::Output,          KernelArgument::WarpIds,
    KernelArgument::CtaIds,          KernelArgument::PartialRanges,
    KernelArgument::PartialMergeIds, KernelArgument::MergeRecords,
    KernelArgument::Scratch,         KernelArgument::Counters,
    KernelArgument::ValueCount,      KernelArgument::WarpTaskCount,
    KernelArgument::CtaTaskCount,    KernelArgument::PartialCount,
    KernelArgument::MergeCount,      KernelArgument::SegmentCount};
} // namespace detail

/// The parameter layout of the kernel a schedule emits.
constexpr KernelLayout kernelLayout(KernelKind kind) {
  switch (kind) {
  case KernelKind::Direct:
    return KernelLayout(detail::directArguments);
  case KernelKind::DirectColumns:
    return KernelLayout(detail::directColumnsArguments);
  case KernelKind::TaskIds:
    return KernelLayout(detail::taskIdArguments);
  case KernelKind::TaskIdsColumns:
    return KernelLayout(detail::taskIdColumnsArguments);
  case KernelKind::FusedMixed:
    return KernelLayout(detail::fusedMixedArguments);
  case KernelKind::SplitPartial:
    return KernelLayout(detail::splitPartialArguments);
  case KernelKind::SplitMerge:
    return KernelLayout(detail::splitMergeArguments);
  case KernelKind::SplitMergeExtent:
    return KernelLayout(detail::splitMergeExtentArguments);
  case KernelKind::Persistent:
    return KernelLayout(detail::persistentArguments);
  }
  llvm_unreachable("unknown kernel kind");
}

} // namespace mlir::swage_plan

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_KERNELLAYOUT_H
