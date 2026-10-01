// unittests/TaskClassifierTest.cpp
#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace mlir::swage_plan {
namespace {

void expectDescriptor(const TaskDescriptor &descriptor, int32_t segment_id,
                      int32_t begin, int32_t end, TaskPolicy policy,
                      int32_t stage = 0) {
  EXPECT_EQ(descriptor.segment_id, segment_id);
  EXPECT_EQ(descriptor.begin, begin);
  EXPECT_EQ(descriptor.end, end);
  EXPECT_EQ(descriptor.stage, stage);
  EXPECT_EQ(descriptor.policy, policy);
  EXPECT_EQ(descriptor.dependency_group, segment_id);
}

TEST(TaskClassifierTest, EmitsNoTasksForZeroSegments) {
  const std::array<int64_t, 1> offsets = {0};

  auto tasks = classifyTasks(offsets, 0, 0, 32, 4096);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  EXPECT_TRUE(tasks->empty());
}

TEST(TaskClassifierTest, ClassifiesWarpCtaAndChunkBoundariesInStageOrder) {
  const std::array<int64_t, 7> offsets = {0, 0, 32, 65, 4160, 8256, 12353};

  auto tasks = classifyTasks(offsets, 12353, 6, 32, 4096);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 8U);
  expectDescriptor((*tasks)[0], 0, 0, 0, TaskPolicy::Warp);
  expectDescriptor((*tasks)[1], 1, 0, 32, TaskPolicy::Warp);
  expectDescriptor((*tasks)[2], 2, 32, 65, TaskPolicy::CTA);
  expectDescriptor((*tasks)[3], 3, 65, 4160, TaskPolicy::CTA);
  expectDescriptor((*tasks)[4], 4, 4160, 8256, TaskPolicy::CTA);
  expectDescriptor((*tasks)[5], 5, 8256, 12352, TaskPolicy::CTA);
  expectDescriptor((*tasks)[6], 5, 12352, 12353, TaskPolicy::CTA);
  expectDescriptor((*tasks)[7], 5, 0, 2, TaskPolicy::CTA, 1);
}

TEST(TaskClassifierTest, SplitsExactMultiplesAndOneElementRemainders) {
  const std::array<int64_t, 3> offsets = {0, 8192, 16385};

  auto tasks = classifyTasks(offsets, 16385, 2, 32, 4096);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 7U);
  expectDescriptor((*tasks)[0], 0, 0, 4096, TaskPolicy::CTA);
  expectDescriptor((*tasks)[1], 0, 4096, 8192, TaskPolicy::CTA);
  expectDescriptor((*tasks)[2], 1, 8192, 12288, TaskPolicy::CTA);
  expectDescriptor((*tasks)[3], 1, 12288, 16384, TaskPolicy::CTA);
  expectDescriptor((*tasks)[4], 1, 16384, 16385, TaskPolicy::CTA);
  expectDescriptor((*tasks)[5], 0, 0, 2, TaskPolicy::CTA, 1);
  expectDescriptor((*tasks)[6], 1, 2, 5, TaskPolicy::CTA, 1);
}

TEST(TaskClassifierTest, RetainsRepeatedEmptyAndAlternatingSegmentIds) {
  const std::array<int64_t, 8> offsets = {0, 0, 0, 1, 1, 3, 3, 6};

  auto tasks = classifyTasks(offsets, 6, 7, 2, 4096);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 7U);
  expectDescriptor((*tasks)[0], 0, 0, 0, TaskPolicy::Warp);
  expectDescriptor((*tasks)[1], 1, 0, 0, TaskPolicy::Warp);
  expectDescriptor((*tasks)[2], 2, 0, 1, TaskPolicy::Warp);
  expectDescriptor((*tasks)[3], 3, 1, 1, TaskPolicy::Warp);
  expectDescriptor((*tasks)[4], 4, 1, 3, TaskPolicy::Warp);
  expectDescriptor((*tasks)[5], 5, 3, 3, TaskPolicy::Warp);
  expectDescriptor((*tasks)[6], 6, 3, 6, TaskPolicy::CTA);
}

TEST(TaskClassifierTest, PreservesI32MaximumDescriptorFields) {
  constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 2> offsets = {0, i32Max};

  auto tasks = classifyTasks(offsets, i32Max, 1, i32Max, i32Max);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 1U);
  expectDescriptor((*tasks)[0], 0, 0, static_cast<int32_t>(i32Max),
                   TaskPolicy::Warp);
}

TEST(TaskClassifierTest, KeepsAllStageZeroWorkBeforeOutlierMerge) {
  const std::array<int64_t, 5> offsets = {0, 1, 1, 10'001, 10'003};

  auto tasks = classifyTasks(offsets, 10'003, 4, 32, 4096);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 7U);
  expectDescriptor((*tasks)[0], 0, 0, 1, TaskPolicy::Warp);
  expectDescriptor((*tasks)[1], 1, 1, 1, TaskPolicy::Warp);
  expectDescriptor((*tasks)[2], 2, 1, 4097, TaskPolicy::CTA);
  expectDescriptor((*tasks)[3], 2, 4097, 8193, TaskPolicy::CTA);
  expectDescriptor((*tasks)[4], 2, 8193, 10'001, TaskPolicy::CTA);
  expectDescriptor((*tasks)[5], 3, 10'001, 10'003, TaskPolicy::Warp);
  expectDescriptor((*tasks)[6], 2, 0, 3, TaskPolicy::CTA, 1);
}

TEST(TaskClassifierTest, AssignsCompactScratchRangesToManyHugeSegments) {
  const std::array<int64_t, 4> offsets = {0, 4097, 8194, 12291};

  auto tasks = classifyTasks(offsets, 12291, 3, 32, 4096);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 9U);
  for (int32_t segmentId = 0; segmentId < 3; ++segmentId) {
    const int32_t begin = segmentId * 4097;
    expectDescriptor((*tasks)[segmentId * 2], segmentId, begin, begin + 4096,
                     TaskPolicy::CTA);
    expectDescriptor((*tasks)[segmentId * 2 + 1], segmentId, begin + 4096,
                     begin + 4097, TaskPolicy::CTA);
    expectDescriptor((*tasks)[6 + segmentId], segmentId, segmentId * 2,
                     segmentId * 2 + 2, TaskPolicy::CTA, 1);
  }
}

TEST(TaskClassifierTest, RejectsMalformedAndOutOfI32Metadata) {
  constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();
  constexpr int64_t i32Overflow = i32Max + 1;
  struct InvalidMetadata {
    const char *name;
    std::vector<int64_t> offsets;
    int64_t valueCount;
    int64_t segmentCount;
    int64_t warpMaxElements;
    int64_t ctaChunkElements;
    const char *message;
  };
  const char *const valueRange = "value count must be a nonnegative i32 value";
  const char *const segmentRange =
      "segment count must be a nonnegative i32 value";
  const char *const warpRange =
      "warp max elements must be a nonnegative i32 value";
  const char *const warpPositive = "warp max elements must be positive";
  const char *const ctaRange =
      "CTA chunk elements must be a nonnegative i32 value";
  const char *const ctaPositive = "CTA chunk elements must be positive";
  const char *const warpAboveCta =
      "warp max elements must not exceed CTA chunk elements";
  const char *const offsetCount =
      "offset count must equal segment count plus one";
  const char *const offsetRange = "offset must be a nonnegative i32 value";
  const char *const offsetOrder = "offsets must be nondecreasing";
  const char *const offsetStart = "offsets must start at zero";
  const char *const offsetEnd = "final offset must not exceed value count";
  const InvalidMetadata invalidInputs[] = {
      {"negative value count", {0}, -1, 0, 32, 4096, valueRange},
      {"value count above i32", {0}, i32Overflow, 0, 32, 4096, valueRange},
      {"negative segment count", {0}, 0, -1, 32, 4096, segmentRange},
      {"segment count above i32", {0}, 0, i32Overflow, 32, 4096, segmentRange},
      {"segment count addition overflow",
       {},
       0,
       std::numeric_limits<int64_t>::max(),
       32,
       4096,
       segmentRange},
      {"negative warp limit", {0}, 0, 0, -1, 4096, warpRange},
      {"zero warp limit", {0}, 0, 0, 0, 4096, warpPositive},
      {"warp limit above i32", {0}, 0, 0, i32Overflow, i32Overflow, warpRange},
      {"negative CTA chunk", {0}, 0, 0, 32, -1, ctaRange},
      {"zero CTA chunk", {0}, 0, 0, 32, 0, ctaPositive},
      {"CTA chunk above i32", {0}, 0, 0, 32, i32Overflow, ctaRange},
      {"warp limit above CTA chunk", {0}, 0, 0, 33, 32, warpAboveCta},
      {"empty offsets", {}, 0, 0, 32, 4096, offsetCount},
      {"missing offset", {0}, 0, 1, 32, 4096, offsetCount},
      {"extra offset", {0, 0}, 0, 0, 32, 4096, offsetCount},
      {"nonzero first offset", {1, 1}, 1, 1, 32, 4096, offsetStart},
      {"negative offset", {0, -1}, 0, 1, 32, 4096, offsetRange},
      {"offset above i32", {0, i32Overflow}, i32Max, 1, 32, 4096, offsetRange},
      {"decreasing offsets", {0, 2, 1}, 2, 2, 32, 4096, offsetOrder},
      {"final offset above value count", {0, 2}, 1, 1, 32, 4096, offsetEnd},
  };

  for (const InvalidMetadata &input : invalidInputs) {
    SCOPED_TRACE(input.name);
    auto tasks =
        classifyTasks(input.offsets, input.valueCount, input.segmentCount,
                      input.warpMaxElements, input.ctaChunkElements);
    ASSERT_FALSE(static_cast<bool>(tasks));
    // The message names the rejected quantity, so a case refused for a
    // different reason than the one it describes fails here.
    EXPECT_EQ(llvm::toString(tasks.takeError()), input.message);
  }
}

TEST(TaskClassifierTest, RejectsDescriptorCountOverflowBeforeAllocation) {
  constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 2> offsets = {0, i32Max};

  auto tasks = classifyTasks(offsets, i32Max, 1, 1, 1);

  ASSERT_FALSE(static_cast<bool>(tasks));
  EXPECT_EQ(llvm::toString(tasks.takeError()),
            "descriptor count must fit in i32");
}

} // namespace
} // namespace mlir::swage_plan
