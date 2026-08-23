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
                      int32_t begin, int32_t end, TaskPolicy policy) {
  EXPECT_EQ(descriptor.segment_id, segment_id);
  EXPECT_EQ(descriptor.begin, begin);
  EXPECT_EQ(descriptor.end, end);
  EXPECT_EQ(descriptor.stage, 0);
  EXPECT_EQ(descriptor.policy, policy);
  EXPECT_EQ(descriptor.dependency_group, segment_id);
}

TEST(TaskClassifierTest, EmitsNoTasksForZeroSegments) {
  const std::array<int64_t, 1> offsets = {0};

  auto tasks = classifyTasks(offsets, 0, 0, 32);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  EXPECT_TRUE(tasks->empty());
}

TEST(TaskClassifierTest, ClassifiesThresholdBoundaryInSegmentOrder) {
  const std::array<int64_t, 4> offsets = {0, 31, 63, 96};

  auto tasks = classifyTasks(offsets, 96, 3, 32);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 3U);
  expectDescriptor((*tasks)[0], 0, 0, 31, TaskPolicy::Warp);
  expectDescriptor((*tasks)[1], 1, 31, 63, TaskPolicy::Warp);
  expectDescriptor((*tasks)[2], 2, 63, 96, TaskPolicy::CTA);
}

TEST(TaskClassifierTest, RetainsRepeatedEmptyAndAlternatingSegmentIds) {
  const std::array<int64_t, 8> offsets = {0, 0, 0, 1, 1, 3, 3, 6};

  auto tasks = classifyTasks(offsets, 6, 7, 2);

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

TEST(TaskClassifierTest, UsesWarpOnlyForEmptySegmentsAtThresholdZero) {
  const std::array<int64_t, 3> offsets = {0, 0, 1};

  auto tasks = classifyTasks(offsets, 1, 2, 0);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 2U);
  expectDescriptor((*tasks)[0], 0, 0, 0, TaskPolicy::Warp);
  expectDescriptor((*tasks)[1], 1, 0, 1, TaskPolicy::CTA);
}

TEST(TaskClassifierTest, PreservesI32MaximumDescriptorFields) {
  constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();
  const std::array<int64_t, 2> offsets = {0, i32Max};

  auto tasks = classifyTasks(offsets, i32Max, 1, i32Max);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 1U);
  expectDescriptor((*tasks)[0], 0, 0, static_cast<int32_t>(i32Max),
                   TaskPolicy::Warp);
}

TEST(TaskClassifierTest, ClassifiesSkewWithoutSplittingSegments) {
  const std::array<int64_t, 5> offsets = {0, 1, 1, 1'000'001, 1'000'003};

  auto tasks = classifyTasks(offsets, 1'000'003, 4, 32);

  if (!tasks)
    FAIL() << llvm::toString(tasks.takeError());
  ASSERT_EQ(tasks->size(), 4U);
  expectDescriptor((*tasks)[0], 0, 0, 1, TaskPolicy::Warp);
  expectDescriptor((*tasks)[1], 1, 1, 1, TaskPolicy::Warp);
  expectDescriptor((*tasks)[2], 2, 1, 1'000'001, TaskPolicy::CTA);
  expectDescriptor((*tasks)[3], 3, 1'000'001, 1'000'003, TaskPolicy::Warp);
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
  };
  const InvalidMetadata invalidInputs[] = {
      {"negative value count", {0}, -1, 0, 32},
      {"value count above i32", {0}, i32Overflow, 0, 32},
      {"negative segment count", {0}, 0, -1, 32},
      {"segment count above i32", {0}, 0, i32Overflow, 32},
      {"segment count addition overflow",
       {},
       0,
       std::numeric_limits<int64_t>::max(),
       32},
      {"negative threshold", {0}, 0, 0, -1},
      {"threshold above i32", {0}, 0, 0, i32Overflow},
      {"empty offsets", {}, 0, 0, 32},
      {"missing offset", {0}, 0, 1, 32},
      {"extra offset", {0, 0}, 0, 0, 32},
      {"nonzero first offset", {1, 1}, 1, 1, 32},
      {"negative offset", {0, -1}, 0, 1, 32},
      {"offset above i32", {0, i32Overflow}, i32Max, 1, 32},
      {"decreasing offsets", {0, 2, 1}, 2, 2, 32},
      {"final offset above value count", {0, 2}, 1, 1, 32},
  };

  for (const InvalidMetadata &input : invalidInputs) {
    SCOPED_TRACE(input.name);
    auto tasks = classifyTasks(input.offsets, input.valueCount,
                               input.segmentCount, input.warpMaxElements);
    EXPECT_FALSE(static_cast<bool>(tasks));
    if (!tasks)
      llvm::consumeError(tasks.takeError());
  }
}

} // namespace
} // namespace mlir::swage_plan
