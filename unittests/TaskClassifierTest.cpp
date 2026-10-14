// unittests/TaskClassifierTest.cpp
#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
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

/// The four record lists, as regrouping the descriptors of classifyTasks
/// by policy and stage gives them. This is the reference for
/// classifyTaskRecords.
struct RegroupedRecords {
  std::vector<int32_t> warp;
  std::vector<int32_t> cta;
  std::vector<int32_t> partial;
  std::vector<int32_t> merge;
};

RegroupedRecords regroup(llvm::ArrayRef<TaskDescriptor> tasks) {
  RegroupedRecords records;
  std::vector<int32_t> splitSegments;
  for (const TaskDescriptor &task : tasks)
    if (task.stage == 1)
      splitSegments.push_back(task.segment_id);
  auto isSplit = [&](int32_t segmentId) {
    return std::find(splitSegments.begin(), splitSegments.end(), segmentId) !=
           splitSegments.end();
  };
  for (const TaskDescriptor &task : tasks) {
    if (task.stage == 1)
      records.merge.insert(records.merge.end(),
                           {task.segment_id, task.begin, task.end});
    else if (task.policy == TaskPolicy::Warp)
      records.warp.push_back(task.segment_id);
    else if (isSplit(task.segment_id))
      records.partial.insert(records.partial.end(), {task.begin, task.end});
    else
      records.cta.push_back(task.segment_id);
  }
  return records;
}

std::vector<int32_t> narrow(const std::vector<int64_t> &offsets) {
  return std::vector<int32_t>(offsets.begin(), offsets.end());
}

/// Requires both classifiers to agree on one input: the same records when
/// the descriptors exist, the same message when they are refused.
void expectRecordsMatchDescriptors(const std::vector<int64_t> &offsets,
                                   int64_t valueCount, int64_t segmentCount,
                                   int64_t warpMaxElements,
                                   int64_t ctaChunkElements) {
  auto tasks = classifyTasks(offsets, valueCount, segmentCount, warpMaxElements,
                             ctaChunkElements);
  const std::vector<int32_t> narrowed = narrow(offsets);
  auto records = classifyTaskRecords(narrowed, valueCount, segmentCount,
                                     warpMaxElements, ctaChunkElements);
  if (!tasks) {
    const std::string message = llvm::toString(tasks.takeError());
    ASSERT_FALSE(static_cast<bool>(records)) << "expected: " << message;
    EXPECT_EQ(llvm::toString(records.takeError()), message);
    return;
  }
  if (!records)
    FAIL() << llvm::toString(records.takeError());
  const RegroupedRecords expected = regroup(*tasks);
  ASSERT_EQ(records->warpCount, static_cast<int32_t>(expected.warp.size()));
  ASSERT_EQ(records->ctaCount, static_cast<int32_t>(expected.cta.size()));
  ASSERT_EQ(2 * records->partialCount,
            static_cast<int32_t>(expected.partial.size()));
  ASSERT_EQ(3 * records->mergeCount,
            static_cast<int32_t>(expected.merge.size()));
  std::vector<int32_t> flat = expected.warp;
  flat.insert(flat.end(), expected.cta.begin(), expected.cta.end());
  flat.insert(flat.end(), expected.partial.begin(), expected.partial.end());
  flat.insert(flat.end(), expected.merge.begin(), expected.merge.end());
  // The merge of a partial task is the merge whose range holds it, and the
  // ranges follow one another, so each merge repeats over its range.
  for (size_t merge = 0; merge < expected.merge.size() / 3; ++merge)
    flat.insert(flat.end(),
                static_cast<size_t>(expected.merge[3 * merge + 2] -
                                    expected.merge[3 * merge + 1]),
                static_cast<int32_t>(merge));
  EXPECT_EQ(
      std::vector<int32_t>(records->records.begin(), records->records.end()),
      flat);
}

std::vector<int64_t> offsetsOf(const std::vector<int64_t> &lengths) {
  std::vector<int64_t> offsets = {0};
  for (int64_t length : lengths)
    offsets.push_back(offsets.back() + length);
  return offsets;
}

TEST(TaskRecordsTest, LaysOutWarpCtaPartialAndMergeRecordsInOneBuffer) {
  // A warp segment, a CTA segment, a segment of three chunks, an empty
  // segment, and a segment of exactly two chunks.
  const std::array<int32_t, 6> offsets = {0, 32, 132, 8325, 8325, 16517};

  auto records = classifyTaskRecords(offsets, 16517, 5, 32, 4096);

  if (!records)
    FAIL() << llvm::toString(records.takeError());
  EXPECT_EQ(records->warpCount, 2);
  EXPECT_EQ(records->ctaCount, 1);
  EXPECT_EQ(records->partialCount, 5);
  EXPECT_EQ(records->mergeCount, 2);
  EXPECT_EQ(
      std::vector<int32_t>(records->records.begin(), records->records.end()),
      // The last five values are the merge of each partial task.
      (std::vector<int32_t>{0,    3,    1,     132,   4228,  4228, 8324, 8324,
                            8325, 8325, 12421, 12421, 16517, 2,    0,    3,
                            4,    3,    5,     0,     0,     0,    1,    1}));
}

TEST(TaskRecordsTest, EmitsNoRecordsForZeroSegments) {
  const std::array<int32_t, 1> offsets = {0};

  auto records = classifyTaskRecords(offsets, 0, 0, 32, 4096);

  if (!records)
    FAIL() << llvm::toString(records.takeError());
  EXPECT_TRUE(records->records.empty());
  EXPECT_EQ(records->warpCount + records->ctaCount + records->partialCount +
                records->mergeCount,
            0);
}

TEST(TaskRecordsTest, EqualsRegroupedDescriptorsOnBoundaryLayouts) {
  constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();
  expectRecordsMatchDescriptors({0, 0, 32, 65, 4160, 8256, 12353}, 12353, 6, 32,
                                4096);
  expectRecordsMatchDescriptors({0, 8192, 16385}, 16385, 2, 32, 4096);
  expectRecordsMatchDescriptors({0, 0, 0, 1, 1, 3, 3, 6}, 6, 7, 2, 4096);
  expectRecordsMatchDescriptors({0, 1, 1, 10'001, 10'003}, 10'003, 4, 32, 4096);
  expectRecordsMatchDescriptors({0, 4097, 8194, 12291}, 12291, 3, 32, 4096);
  expectRecordsMatchDescriptors({0, i32Max}, i32Max, 1, i32Max, i32Max);
  expectRecordsMatchDescriptors({0, i32Max}, i32Max, 1, 1, i32Max);
  expectRecordsMatchDescriptors({0, 5, 9}, 12, 2, 1, 1);
}

TEST(TaskRecordsTest, EqualsRegroupedDescriptorsOnSeededRandomLayouts) {
  const std::pair<int64_t, int64_t> limits[] = {
      {32, 4096}, {1, 1}, {7, 100}, {64, 64}};
  for (const auto &[warpMax, chunk] : limits) {
    for (unsigned seed = 0; seed < 50; ++seed) {
      SCOPED_TRACE(testing::Message() << "warp " << warpMax << " chunk "
                                      << chunk << " seed " << seed);
      std::mt19937 generator(seed);
      auto draw = [&](int64_t low, int64_t high) {
        return std::uniform_int_distribution<int64_t>(low, high)(generator);
      };
      // Every boundary, then random lengths of every class.
      std::vector<int64_t> lengths = {0,
                                      0,
                                      warpMax,
                                      warpMax + 1,
                                      chunk,
                                      chunk + 1,
                                      draw(3, 6) * chunk + draw(0, chunk - 1)};
      for (int64_t count = draw(0, 150); count > 0; --count) {
        const int64_t kind = draw(0, 99);
        if (kind < 15)
          lengths.push_back(0);
        else if (kind < 60)
          lengths.push_back(draw(1, warpMax));
        else if (kind < 90)
          lengths.push_back(draw(warpMax, chunk));
        else
          lengths.push_back(draw(chunk + 1, 4 * chunk + 1));
      }
      std::shuffle(lengths.begin(), lengths.end(), generator);
      const std::vector<int64_t> offsets = offsetsOf(lengths);
      expectRecordsMatchDescriptors(offsets, offsets.back() + draw(0, 3),
                                    static_cast<int64_t>(lengths.size()),
                                    warpMax, chunk);
    }
  }
}

TEST(TaskRecordsTest, RejectsWhatDescriptorsRejectWithTheSameMessage) {
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
  // Every row of the descriptor table whose offsets fit in i32, and the
  // precedence between the offset checks.
  const InvalidMetadata invalidInputs[] = {
      {"negative value count",
       {0},
       -1,
       0,
       32,
       4096,
       "value count must be a nonnegative i32 value"},
      {"value count above i32",
       {0},
       i32Overflow,
       0,
       32,
       4096,
       "value count must be a nonnegative i32 value"},
      {"negative segment count",
       {0},
       0,
       -1,
       32,
       4096,
       "segment count must be a nonnegative i32 value"},
      {"segment count above i32",
       {0},
       0,
       i32Overflow,
       32,
       4096,
       "segment count must be a nonnegative i32 value"},
      {"negative warp limit",
       {0},
       0,
       0,
       -1,
       4096,
       "warp max elements must be a nonnegative i32 value"},
      {"zero warp limit",
       {0},
       0,
       0,
       0,
       4096,
       "warp max elements must be positive"},
      {"warp limit above i32",
       {0},
       0,
       0,
       i32Overflow,
       i32Overflow,
       "warp max elements must be a nonnegative i32 value"},
      {"negative CTA chunk",
       {0},
       0,
       0,
       32,
       -1,
       "CTA chunk elements must be a nonnegative i32 value"},
      {"zero CTA chunk",
       {0},
       0,
       0,
       32,
       0,
       "CTA chunk elements must be positive"},
      {"CTA chunk above i32",
       {0},
       0,
       0,
       32,
       i32Overflow,
       "CTA chunk elements must be a nonnegative i32 value"},
      {"warp limit above CTA chunk",
       {0},
       0,
       0,
       33,
       32,
       "warp max elements must not exceed CTA chunk elements"},
      {"empty offsets",
       {},
       0,
       0,
       32,
       4096,
       "offset count must equal segment count plus one"},
      {"missing offset",
       {0},
       0,
       1,
       32,
       4096,
       "offset count must equal segment count plus one"},
      {"extra offset",
       {0, 0},
       0,
       0,
       32,
       4096,
       "offset count must equal segment count plus one"},
      {"nonzero first offset",
       {1, 1},
       1,
       1,
       32,
       4096,
       "offsets must start at zero"},
      {"negative offset",
       {0, -1},
       0,
       1,
       32,
       4096,
       "offset must be a nonnegative i32 value"},
      {"negative first offset",
       {-1, 0},
       0,
       1,
       32,
       4096,
       "offset must be a nonnegative i32 value"},
      {"decreasing offsets",
       {0, 2, 1},
       2,
       2,
       32,
       4096,
       "offsets must be nondecreasing"},
      {"decrease before a negative offset",
       {0, 5, 3, -1},
       5,
       3,
       32,
       4096,
       "offsets must be nondecreasing"},
      {"negative offset before a decrease",
       {0, 5, -1, 3},
       5,
       3,
       32,
       4096,
       "offset must be a nonnegative i32 value"},
      {"decrease before a nonzero start",
       {5, 3},
       5,
       1,
       32,
       4096,
       "offsets must be nondecreasing"},
      {"final offset above value count",
       {0, 2},
       1,
       1,
       32,
       4096,
       "final offset must not exceed value count"},
      {"descriptor overflow",
       {0, i32Max},
       i32Max,
       1,
       1,
       1,
       "descriptor count must fit in i32"},
      {"decrease after a descriptor overflow",
       {0, i32Max, 5},
       i32Max,
       2,
       1,
       1,
       "offsets must be nondecreasing"},
      {"final offset above value count after a descriptor overflow",
       {0, i32Max},
       i32Max - 1,
       1,
       1,
       1,
       "final offset must not exceed value count"},
  };

  for (const InvalidMetadata &input : invalidInputs) {
    SCOPED_TRACE(input.name);
    auto records = classifyTaskRecords(
        narrow(input.offsets), input.valueCount, input.segmentCount,
        input.warpMaxElements, input.ctaChunkElements);
    ASSERT_FALSE(static_cast<bool>(records));
    EXPECT_EQ(llvm::toString(records.takeError()), input.message);
    expectRecordsMatchDescriptors(input.offsets, input.valueCount,
                                  input.segmentCount, input.warpMaxElements,
                                  input.ctaChunkElements);
  }
}

TEST(TaskRecordsTest, AgreesWithDescriptorsOnSeededMalformedOffsets) {
  constexpr int64_t i32Min = std::numeric_limits<int32_t>::min();
  constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();
  // Far above every other offset, and small enough that a layout it leaves
  // valid still splits into few tasks. The table above covers i32Max.
  constexpr int64_t far = 1 << 20;
  for (unsigned seed = 0; seed < 400; ++seed) {
    SCOPED_TRACE(seed);
    std::mt19937 generator(seed);
    auto draw = [&](int64_t low, int64_t high) {
      return std::uniform_int_distribution<int64_t>(low, high)(generator);
    };
    std::vector<int64_t> lengths(static_cast<size_t>(draw(0, 30)));
    for (int64_t &length : lengths)
      length = draw(0, 40);
    std::vector<int64_t> offsets = offsetsOf(lengths);
    for (int64_t changes = draw(0, 3); changes > 0; --changes) {
      int64_t &offset = offsets[static_cast<size_t>(
          draw(0, static_cast<int64_t>(offsets.size()) - 1))];
      const int64_t choices[] = {-1,
                                 -draw(1, 50),
                                 offset - draw(1, 50),
                                 offset + draw(1, 50),
                                 draw(0, 50),
                                 i32Min,
                                 far};
      offset = std::clamp(choices[draw(0, 6)], i32Min, i32Max);
    }
    const int64_t valueCount =
        std::clamp<int64_t>(offsets.back() + draw(-3, 40), 0, i32Max);
    expectRecordsMatchDescriptors(offsets, valueCount,
                                  static_cast<int64_t>(lengths.size()),
                                  draw(1, 8), draw(8, 20));
  }
}

} // namespace
} // namespace mlir::swage_plan
