// include/swage/Dialect/SwagePlan/IR/TaskClassifier.h
#ifndef SWAGE_DIALECT_SWAGEPLAN_IR_TASKCLASSIFIER_H
#define SWAGE_DIALECT_SWAGEPLAN_IR_TASKCLASSIFIER_H

#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir::swage_plan {

struct TaskDescriptor {
  int32_t segment_id;
  int32_t begin;
  int32_t end;
  int32_t stage;
  TaskPolicy policy;
  int32_t dependency_group;
};

llvm::Expected<llvm::SmallVector<TaskDescriptor>>
classifyTasks(llvm::ArrayRef<int64_t> offsets, int64_t valueCount,
              int64_t segmentCount, int64_t warpMaxElements,
              int64_t ctaChunkElements);

/// The launch records of one classification in one buffer: the warp segment
/// ids, then the CTA segment ids, then one [begin, end] pair per partial
/// task, then one [segment_id, partial_begin, partial_end] triple per merge.
/// The counts are in ids, pairs, and triples.
struct TaskRecords {
  llvm::SmallVector<int32_t, 0> records;
  int32_t warpCount = 0;
  int32_t ctaCount = 0;
  int32_t partialCount = 0;
  int32_t mergeCount = 0;
};

/// Classifies i32 offsets straight into launch records, without building
/// descriptors. It admits and rejects what classifyTasks does, with the same
/// messages, and returns the records its descriptors regroup to: a warp
/// descriptor gives a warp id, a CTA descriptor of an unsplit segment a CTA
/// id, a CTA descriptor of a split segment a partial pair, and a stage-one
/// descriptor a merge triple.
llvm::Expected<TaskRecords> classifyTaskRecords(llvm::ArrayRef<int32_t> offsets,
                                                int64_t valueCount,
                                                int64_t segmentCount,
                                                int64_t warpMaxElements,
                                                int64_t ctaChunkElements);

} // namespace mlir::swage_plan

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_TASKCLASSIFIER_H
