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
  };
  const InvalidMetadata invalidInputs[] = {
      {"negative value count", {0}, -1, 0, 32, 4096},
      {"value count above i32", {0}, i32Overflow, 0, 32, 4096},
      {"negative segment count", {0}, 0, -1, 32, 4096},
      {"segment count above i32", {0}, 0, i32Overflow, 32, 4096},
      {"segment count addition overflow",
       {},
       0,
       std::numeric_limits<int64_t>::max(),
       32,
       4096},
      {"negative warp limit", {0}, 0, 0, -1, 4096},
      {"zero warp limit", {0}, 0, 0, 0, 4096},
      {"warp limit above i32", {0}, 0, 0, i32Overflow, i32Overflow},
      {"negative CTA chunk", {0}, 0, 0, 32, -1},
      {"zero CTA chunk", {0}, 0, 0, 32, 0},
      {"CTA chunk above i32", {0}, 0, 0, 32, i32Overflow},
      {"warp limit above CTA chunk", {0}, 0, 0, 33, 32},
      {"empty offsets", {}, 0, 0, 32, 4096},
      {"missing offset", {0}, 0, 1, 32, 4096},
      {"extra offset", {0, 0}, 0, 0, 32, 4096},
      {"nonzero first offset", {1, 1}, 1, 1, 32, 4096},
      {"negative offset", {0, -1}, 0, 1, 32, 4096},
      {"offset above i32", {0, i32Overflow}, i32Max, 1, 32, 4096},
      {"decreasing offsets", {0, 2, 1}, 2, 2, 32, 4096},
      {"final offset above value count", {0, 2}, 1, 1, 32, 4096},
  };

  for (const InvalidMetadata &input : invalidInputs) {
    SCOPED_TRACE(input.name);
    auto tasks =
        classifyTasks(input.offsets, input.valueCount, input.segmentCount,
                      input.warpMaxElements, input.ctaChunkElements);
    EXPECT_FALSE(static_cast<bool>(tasks));
    if (!tasks)
      llvm::consumeError(tasks.takeError());
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
