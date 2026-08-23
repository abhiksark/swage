// lib/Dialect/SwagePlan/IR/TaskClassifier.cpp
#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"

#include "llvm/ADT/Twine.h"

#include <cstddef>
#include <limits>
#include <utility>

namespace mlir::swage_plan {
namespace {

constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();

llvm::Error invalidMetadata(const llvm::Twine &message) {
  return llvm::createStringError(message);
}

llvm::Error validateI32Metadata(int64_t value, const llvm::Twine &name) {
  if (value < 0 || value > i32Max)
    return invalidMetadata(name + " must be a nonnegative i32 value");
  return llvm::Error::success();
}

} // namespace

llvm::Expected<llvm::SmallVector<TaskDescriptor>>
classifyTasks(llvm::ArrayRef<int64_t> offsets, int64_t valueCount,
              int64_t segmentCount, int64_t warpMaxElements) {
  if (llvm::Error error = validateI32Metadata(valueCount, "value count"))
    return std::move(error);
  if (llvm::Error error = validateI32Metadata(segmentCount, "segment count"))
    return std::move(error);
  if (llvm::Error error =
          validateI32Metadata(warpMaxElements, "warp max elements"))
    return std::move(error);

  const uint64_t expectedOffsetCount =
      static_cast<uint64_t>(segmentCount) + uint64_t{1};
  if (offsets.size() != expectedOffsetCount)
    return invalidMetadata("offset count must equal segment count plus one");

  int64_t previousOffset = 0;
  for (int64_t offset : offsets) {
    if (llvm::Error error = validateI32Metadata(offset, "offset"))
      return std::move(error);
    if (offset < previousOffset)
      return invalidMetadata("offsets must be nondecreasing");
    previousOffset = offset;
  }
  if (offsets.front() != 0)
    return invalidMetadata("offsets must start at zero");
  if (offsets.back() > valueCount)
    return invalidMetadata("final offset must not exceed value count");

  for (int64_t segmentId = 0; segmentId < segmentCount; ++segmentId) {
    const int64_t length = offsets[segmentId + 1] - offsets[segmentId];
    if (length < 0 || length > i32Max)
      return invalidMetadata("segment length must fit in i32");
  }

  llvm::SmallVector<TaskDescriptor> tasks;
  tasks.reserve(static_cast<size_t>(segmentCount));
  for (int64_t segmentId = 0; segmentId < segmentCount; ++segmentId) {
    const int64_t begin = offsets[segmentId];
    const int64_t end = offsets[segmentId + 1];
    const TaskPolicy policy =
        end - begin <= warpMaxElements ? TaskPolicy::Warp : TaskPolicy::CTA;
    tasks.push_back({static_cast<int32_t>(segmentId),
                     static_cast<int32_t>(begin), static_cast<int32_t>(end), 0,
                     policy, static_cast<int32_t>(segmentId)});
  }
  return tasks;
}

} // namespace mlir::swage_plan
