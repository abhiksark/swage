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
              int64_t segmentCount, int64_t warpMaxElements);

} // namespace mlir::swage_plan

#endif // SWAGE_DIALECT_SWAGEPLAN_IR_TASKCLASSIFIER_H
