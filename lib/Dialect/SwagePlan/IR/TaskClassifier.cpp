// lib/Dialect/SwagePlan/IR/TaskClassifier.cpp
#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"

#include "llvm/ADT/Twine.h"

#include <algorithm>
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

llvm::Error validateClassifierConfiguration(int64_t valueCount,
                                            int64_t segmentCount,
                                            int64_t warpMaxElements,
                                            int64_t ctaChunkElements) {
  if (llvm::Error error = validateI32Metadata(valueCount, "value count"))
    return error;
  if (llvm::Error error = validateI32Metadata(segmentCount, "segment count"))
    return error;
  if (llvm::Error error =
          validateI32Metadata(warpMaxElements, "warp max elements"))
    return error;
  if (llvm::Error error =
          validateI32Metadata(ctaChunkElements, "CTA chunk elements"))
    return error;
  if (warpMaxElements == 0)
    return invalidMetadata("warp max elements must be positive");
  if (ctaChunkElements == 0)
    return invalidMetadata("CTA chunk elements must be positive");
  if (warpMaxElements > ctaChunkElements)
    return invalidMetadata(
        "warp max elements must not exceed CTA chunk elements");
  return llvm::Error::success();
}

/// Admits offsets that start at zero, never decrease, and end within the
/// value count. Every offset is then in [0, i32Max], so every segment length
/// `offsets[i + 1] - offsets[i]` is too, and the lengths sum to at most the
/// value count. The counting and narrowing below rely on both facts.
llvm::Error validateOffsets(llvm::ArrayRef<int64_t> offsets, int64_t valueCount,
                            int64_t segmentCount) {
  const uint64_t expectedOffsetCount =
      static_cast<uint64_t>(segmentCount) + uint64_t{1};
  if (offsets.size() != expectedOffsetCount)
    return invalidMetadata("offset count must equal segment count plus one");

  int64_t previousOffset = 0;
  for (int64_t offset : offsets) {
    if (llvm::Error error = validateI32Metadata(offset, "offset"))
      return error;
    if (offset < previousOffset)
      return invalidMetadata("offsets must be nondecreasing");
    previousOffset = offset;
  }
  if (offsets.front() != 0)
    return invalidMetadata("offsets must start at zero");
  if (offsets.back() > valueCount)
    return invalidMetadata("final offset must not exceed value count");
  return llvm::Error::success();
}

struct TaskCounts {
  uint64_t stageZero = 0;
  uint64_t merges = 0;
};

/// Counts descriptors before any is allocated. A chunk holds at least one
/// element, so the partial tasks of the split segments number at most the
/// value count and their scratch indices fit in i32 without a check of
/// their own. The descriptor count adds one task per unsplit segment and one
/// merge per split segment, and that sum can exceed i32.
llvm::Expected<TaskCounts> countTasks(llvm::ArrayRef<int64_t> offsets,
                                      int64_t segmentCount,
                                      int64_t ctaChunkElements) {
  TaskCounts counts;
  for (int64_t segmentId = 0; segmentId < segmentCount; ++segmentId) {
    const int64_t length = offsets[segmentId + 1] - offsets[segmentId];
    uint64_t taskCount = 1;
    if (length > ctaChunkElements) {
      taskCount = static_cast<uint64_t>(length / ctaChunkElements) +
                  static_cast<uint64_t>(length % ctaChunkElements != 0);
      ++counts.merges;
    }
    counts.stageZero += taskCount;
    if (counts.stageZero + counts.merges > static_cast<uint64_t>(i32Max))
      return invalidMetadata("descriptor count must fit in i32");
  }
  return counts;
}

void appendDirectTask(llvm::SmallVectorImpl<TaskDescriptor> &tasks,
                      int64_t segmentId, int64_t begin, int64_t end,
                      int64_t warpMaxElements) {
  const TaskPolicy policy =
      end - begin <= warpMaxElements ? TaskPolicy::Warp : TaskPolicy::CTA;
  tasks.push_back({static_cast<int32_t>(segmentId), static_cast<int32_t>(begin),
                   static_cast<int32_t>(end), 0, policy,
                   static_cast<int32_t>(segmentId)});
}

void appendSplitTasks(llvm::SmallVectorImpl<TaskDescriptor> &tasks,
                      llvm::SmallVectorImpl<TaskDescriptor> &merges,
                      int64_t segmentId, int64_t begin, int64_t end,
                      int64_t ctaChunkElements, int64_t &scratchIndex) {
  const int64_t partialBegin = scratchIndex;
  for (int64_t chunkBegin = begin; chunkBegin < end;
       chunkBegin += ctaChunkElements) {
    const int64_t chunkEnd = std::min(end, chunkBegin + ctaChunkElements);
    tasks.push_back({static_cast<int32_t>(segmentId),
                     static_cast<int32_t>(chunkBegin),
                     static_cast<int32_t>(chunkEnd), 0, TaskPolicy::CTA,
                     static_cast<int32_t>(segmentId)});
    ++scratchIndex;
  }
  merges.push_back({static_cast<int32_t>(segmentId),
                    static_cast<int32_t>(partialBegin),
                    static_cast<int32_t>(scratchIndex), 1, TaskPolicy::CTA,
                    static_cast<int32_t>(segmentId)});
}

llvm::SmallVector<TaskDescriptor>
materializeTasks(llvm::ArrayRef<int64_t> offsets, int64_t segmentCount,
                 int64_t warpMaxElements, int64_t ctaChunkElements,
                 const TaskCounts &counts) {
  llvm::SmallVector<TaskDescriptor> tasks;
  tasks.reserve(static_cast<size_t>(counts.stageZero + counts.merges));
  llvm::SmallVector<TaskDescriptor> merges;
  merges.reserve(static_cast<size_t>(counts.merges));
  int64_t scratchIndex = 0;
  for (int64_t segmentId = 0; segmentId < segmentCount; ++segmentId) {
    const int64_t begin = offsets[segmentId];
    const int64_t end = offsets[segmentId + 1];
    if (end - begin <= ctaChunkElements)
      appendDirectTask(tasks, segmentId, begin, end, warpMaxElements);
    else
      appendSplitTasks(tasks, merges, segmentId, begin, end, ctaChunkElements,
                       scratchIndex);
  }
  tasks.append(merges);
  return tasks;
}

/// Names the first offset that validateOffsets would reject for its sign or
/// its order. The caller has seen one, so the walk always finds it.
llvm::Error firstOffsetError(llvm::ArrayRef<int32_t> offsets) {
  int64_t previousOffset = 0;
  for (int64_t offset : offsets) {
    if (llvm::Error error = validateI32Metadata(offset, "offset"))
      return error;
    if (offset < previousOffset)
      break;
    previousOffset = offset;
  }
  return invalidMetadata("offsets must be nondecreasing");
}

} // namespace

llvm::Expected<TaskRecords> classifyTaskRecords(llvm::ArrayRef<int32_t> offsets,
                                                int64_t valueCount,
                                                int64_t segmentCount,
                                                int64_t warpMaxElements,
                                                int64_t ctaChunkElements) {
  if (llvm::Error error = validateClassifierConfiguration(
          valueCount, segmentCount, warpMaxElements, ctaChunkElements))
    return std::move(error);
  if (offsets.size() != static_cast<uint64_t>(segmentCount) + uint64_t{1})
    return invalidMetadata("offset count must equal segment count plus one");

  // One walk validates and counts. A negative or decreasing offset makes a
  // length negative, or is the first offset itself, because every offset
  // before the first invalid one is at least zero. The counts of an invalid
  // walk are never used.
  const int32_t *const data = offsets.data();
  bool ordered = data[0] >= 0;
  uint64_t warpCount = 0;
  uint64_t mergeCount = 0;
  uint64_t partialCount = 0;
  int64_t begin = data[0];
  for (int64_t segmentId = 0; segmentId < segmentCount; ++segmentId) {
    const int64_t end = data[segmentId + 1];
    const int64_t length = end - begin;
    ordered &= length >= 0;
    warpCount += length <= warpMaxElements;
    if (length > ctaChunkElements) {
      ++mergeCount;
      partialCount += static_cast<uint64_t>(length / ctaChunkElements) +
                      static_cast<uint64_t>(length % ctaChunkElements != 0);
    }
    begin = end;
  }
  if (!ordered)
    return firstOffsetError(offsets);
  if (offsets.front() != 0)
    return invalidMetadata("offsets must start at zero");
  if (offsets.back() > valueCount)
    return invalidMetadata("final offset must not exceed value count");
  // classifyTasks counts one descriptor per unsplit segment, per partial
  // task, and per merge, and reports an overflow only for valid offsets.
  const uint64_t ctaCount =
      static_cast<uint64_t>(segmentCount) - warpCount - mergeCount;
  if (warpCount + ctaCount + partialCount + mergeCount >
      static_cast<uint64_t>(i32Max))
    return invalidMetadata("descriptor count must fit in i32");

  TaskRecords result;
  result.warpCount = static_cast<int32_t>(warpCount);
  result.ctaCount = static_cast<int32_t>(ctaCount);
  result.partialCount = static_cast<int32_t>(partialCount);
  result.mergeCount = static_cast<int32_t>(mergeCount);
  result.records.resize_for_overwrite(static_cast<size_t>(
      warpCount + ctaCount + 3 * partialCount + 3 * mergeCount));
  int32_t *warp = result.records.data();
  int32_t *cta = warp + warpCount;
  int32_t *partial = cta + ctaCount;
  int32_t *merge = partial + 2 * partialCount;
  int32_t *partialMerge = merge + 3 * mergeCount;
  int32_t scratchIndex = 0;
  int32_t mergeIndex = 0;
  begin = 0;
  for (int64_t segmentId = 0; segmentId < segmentCount; ++segmentId) {
    const int64_t end = data[segmentId + 1];
    const int64_t length = end - begin;
    if (length <= warpMaxElements) {
      *warp++ = static_cast<int32_t>(segmentId);
    } else if (length <= ctaChunkElements) {
      *cta++ = static_cast<int32_t>(segmentId);
    } else {
      *merge++ = static_cast<int32_t>(segmentId);
      *merge++ = scratchIndex;
      for (int64_t chunkBegin = begin; chunkBegin < end;
           chunkBegin += ctaChunkElements) {
        *partial++ = static_cast<int32_t>(chunkBegin);
        *partial++ =
            static_cast<int32_t>(std::min(end, chunkBegin + ctaChunkElements));
        *partialMerge++ = mergeIndex;
        ++scratchIndex;
      }
      *merge++ = scratchIndex;
      ++mergeIndex;
    }
    begin = end;
  }
  return result;
}

llvm::Expected<llvm::SmallVector<TaskDescriptor>>
classifyTasks(llvm::ArrayRef<int64_t> offsets, int64_t valueCount,
              int64_t segmentCount, int64_t warpMaxElements,
              int64_t ctaChunkElements) {
  if (llvm::Error error = validateClassifierConfiguration(
          valueCount, segmentCount, warpMaxElements, ctaChunkElements))
    return std::move(error);
  if (llvm::Error error = validateOffsets(offsets, valueCount, segmentCount))
    return std::move(error);
  llvm::Expected<TaskCounts> counts =
      countTasks(offsets, segmentCount, ctaChunkElements);
  if (!counts)
    return counts.takeError();
  return materializeTasks(offsets, segmentCount, warpMaxElements,
                          ctaChunkElements, *counts);
}

} // namespace mlir::swage_plan
