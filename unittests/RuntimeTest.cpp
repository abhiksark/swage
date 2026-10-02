// unittests/RuntimeTest.cpp
#include "swage-c/Runtime.h"

#include "swage/Dialect/SwagePlan/IR/TaskClassifier.h"

#include "llvm/Support/Error.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr int64_t i32Min = std::numeric_limits<int32_t>::min();
constexpr int64_t i32Max = std::numeric_limits<int32_t>::max();

/// What the runtime library made of one input: the records and the four
/// counts, or the reason it refused.
struct RuntimeClassification {
  std::vector<int32_t> records;
  std::array<int64_t, 4> counts = {0, 0, 0, 0};
  std::string error;
};

RuntimeClassification classifyWithRuntime(const std::vector<int32_t> &offsets,
                                          int64_t valueCount,
                                          int64_t segmentCount,
                                          int64_t warpMaxElements,
                                          int64_t ctaChunkElements) {
  RuntimeClassification result;
  const char *error = swageRuntimeCountTasks(
      offsets.data(), static_cast<int64_t>(offsets.size()), valueCount,
      segmentCount, warpMaxElements, ctaChunkElements, result.counts.data());
  if (error) {
    result.error = error;
    return result;
  }
  // One value past the records keeps a sentinel, so a write past the count
  // the library reported fails the comparison below.
  const size_t size =
      static_cast<size_t>(result.counts[0] + result.counts[1] +
                          3 * result.counts[2] + 3 * result.counts[3]);
  result.records.assign(size + 1, -7);
  swageRuntimeWriteTasks(offsets.data(), segmentCount, warpMaxElements,
                         ctaChunkElements, result.counts.data(),
                         result.records.data());
  EXPECT_EQ(result.records.back(), -7);
  result.records.pop_back();
  return result;
}

/// Requires the runtime library and the compiler's record classifier to
/// agree on one input: the same records and counts when it is admitted, the
/// same message when it is refused.
void expectRuntimeMatchesCompiler(const std::vector<int64_t> &wideOffsets,
                                  int64_t valueCount, int64_t segmentCount,
                                  int64_t warpMaxElements,
                                  int64_t ctaChunkElements) {
  const std::vector<int32_t> offsets(wideOffsets.begin(), wideOffsets.end());
  auto expected = mlir::swage_plan::classifyTaskRecords(
      offsets, valueCount, segmentCount, warpMaxElements, ctaChunkElements);
  const RuntimeClassification actual = classifyWithRuntime(
      offsets, valueCount, segmentCount, warpMaxElements, ctaChunkElements);
  if (!expected) {
    EXPECT_EQ(actual.error, llvm::toString(expected.takeError()));
    return;
  }
  ASSERT_EQ(actual.error, "");
  EXPECT_EQ(actual.counts[0], expected->warpCount);
  EXPECT_EQ(actual.counts[1], expected->ctaCount);
  EXPECT_EQ(actual.counts[2], expected->partialCount);
  EXPECT_EQ(actual.counts[3], expected->mergeCount);
  EXPECT_EQ(actual.records, std::vector<int32_t>(expected->records.begin(),
                                                 expected->records.end()));
}

std::vector<int64_t> offsetsOf(const std::vector<int64_t> &lengths) {
  std::vector<int64_t> offsets = {0};
  for (int64_t length : lengths)
    offsets.push_back(offsets.back() + length);
  return offsets;
}

TEST(RuntimeTest, ReportsTheAbiVersionOfItsHeader) {
  EXPECT_EQ(swageRuntimeAbiVersion(), SWAGE_RUNTIME_ABI_VERSION);
}

TEST(RuntimeTest, LaysOutWarpCtaPartialAndMergeRecordsInOneBuffer) {
  // A warp segment, a CTA segment, a segment of three chunks, an empty
  // segment, and a segment of exactly two chunks.
  const RuntimeClassification result =
      classifyWithRuntime({0, 32, 132, 8325, 8325, 16517}, 16517, 5, 32, 4096);

  ASSERT_EQ(result.error, "");
  EXPECT_EQ(result.counts, (std::array<int64_t, 4>{2, 1, 5, 2}));
  // The last five values are the merge of each partial task.
  EXPECT_EQ(
      result.records,
      (std::vector<int32_t>{0,    3,    1,     132,   4228,  4228, 8324, 8324,
                            8325, 8325, 12421, 12421, 16517, 2,    0,    3,
                            4,    3,    5,     0,     0,     0,    1,    1}));
}

TEST(RuntimeTest, WritesNoRecordForZeroSegments) {
  const RuntimeClassification result = classifyWithRuntime({0}, 0, 0, 32, 4096);

  EXPECT_EQ(result.error, "");
  EXPECT_EQ(result.counts, (std::array<int64_t, 4>{0, 0, 0, 0}));
  EXPECT_TRUE(result.records.empty());
}

TEST(RuntimeTest, EqualsTheCompilerClassifierOnSeededLayouts) {
  const std::array<std::array<int64_t, 2>, 4> limits = {
      {{32, 4096}, {1, 1}, {7, 100}, {64, 64}}};
  for (unsigned seed = 0; seed < 200; ++seed) {
    SCOPED_TRACE(seed);
    std::mt19937 generator(seed);
    auto draw = [&](int64_t low, int64_t high) {
      return std::uniform_int_distribution<int64_t>(low, high)(generator);
    };
    const int64_t warpMax = limits[seed % limits.size()][0];
    const int64_t chunk = limits[seed % limits.size()][1];
    // Every layout holds each classification boundary and one segment of
    // several chunks, among random lengths of every class.
    std::vector<int64_t> lengths = {0,
                                    0,
                                    warpMax,
                                    warpMax + 1,
                                    chunk,
                                    chunk + 1,
                                    draw(3, 6) * chunk + draw(0, chunk - 1)};
    for (int64_t extra = draw(0, 200); extra > 0; --extra) {
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
    // Values past the final offset belong to no segment and are admitted.
    expectRuntimeMatchesCompiler(offsets, offsets.back() + draw(0, 5),
                                 static_cast<int64_t>(lengths.size()), warpMax,
                                 chunk);
  }
}

TEST(RuntimeTest, RefusesWhatTheCompilerClassifierRefusesWithItsMessage) {
  struct Input {
    std::vector<int64_t> offsets;
    int64_t valueCount;
    int64_t segmentCount;
    int64_t warpMaxElements;
    int64_t ctaChunkElements;
  };
  const Input inputs[] = {
      {{0, 1}, -1, 1, 32, 4096},
      {{0, 1}, i32Max + 1, 1, 32, 4096},
      {{0, 1}, 1, -1, 32, 4096},
      {{0, 1}, 1, 1, -1, 4096},
      {{0, 1}, 1, 1, 0, 4096},
      {{0, 1}, 1, 1, 32, 0},
      {{0, 1}, 1, 1, 32, i32Max + 1},
      {{0, 1}, 1, 1, 33, 32},
      {{0, 1}, 1, 2, 32, 4096},
      {{0, 1, 2}, 2, 1, 32, 4096},
      {{0, 2, 1}, 2, 2, 32, 4096},
      {{0, -1}, 0, 1, 32, 4096},
      {{-1, 1}, 1, 1, 32, 4096},
      {{1, 1}, 1, 1, 32, 4096},
      {{0, 2}, 1, 1, 32, 4096},
      {{0, i32Max, i32Min}, 5, 2, 32, 4096},
      {{0, i32Max}, i32Max, 1, 1, 1},
      {{0, i32Max, 5}, i32Max, 2, 1, 1},
      {{0, i32Max}, i32Max - 1, 1, 1, 1},
  };
  for (const Input &input : inputs) {
    SCOPED_TRACE(testing::PrintToString(input.offsets));
    const RuntimeClassification actual = classifyWithRuntime(
        std::vector<int32_t>(input.offsets.begin(), input.offsets.end()),
        input.valueCount, input.segmentCount, input.warpMaxElements,
        input.ctaChunkElements);
    EXPECT_NE(actual.error, "");
    expectRuntimeMatchesCompiler(input.offsets, input.valueCount,
                                 input.segmentCount, input.warpMaxElements,
                                 input.ctaChunkElements);
  }
}

TEST(RuntimeTest, EqualsTheCompilerClassifierOnSeededMalformedOffsets) {
  // Far above every other offset, and small enough that a layout it leaves
  // valid still splits into few tasks.
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
      const int64_t choices[] = {-1, -draw(1, 50), i32Min, far, offset - 1, 1};
      offset = choices[draw(0, 5)];
    }
    expectRuntimeMatchesCompiler(
        offsets, std::max<int64_t>(offsets.back(), 0) + draw(-1, 1),
        static_cast<int64_t>(lengths.size()) + (draw(0, 9) == 0), 7, 16);
  }
}

TEST(RuntimeTest, RefusesNullOffsets) {
  std::array<int64_t, 4> counts = {0, 0, 0, 0};

  EXPECT_STREQ(
      swageRuntimeCountTasks(nullptr, 1, 0, 0, 32, 4096, counts.data()),
      "offsets must not be null");
}

TEST(RuntimeTest, RefusesALaunchGeometryTheDriverCannotTake) {
  const std::array<uint64_t, 1> pointers = {0};
  const std::array<int32_t, 1> scalars = {0};
  const std::array<uint64_t, 17> manyPointers = {};

  EXPECT_EQ(
      swageRuntimeLaunch(0, 0, 32, 0, pointers.data(), 1, scalars.data(), 1),
      SWAGE_RUNTIME_ERROR_GRID);
  EXPECT_EQ(swageRuntimeLaunch(0, int64_t{1} << 32, 32, 0, pointers.data(), 1,
                               scalars.data(), 1),
            SWAGE_RUNTIME_ERROR_GRID);
  EXPECT_EQ(
      swageRuntimeLaunch(0, 1, 0, 0, pointers.data(), 1, scalars.data(), 1),
      SWAGE_RUNTIME_ERROR_BLOCK);
  EXPECT_EQ(
      swageRuntimeLaunch(0, 1, 1025, 0, pointers.data(), 1, scalars.data(), 1),
      SWAGE_RUNTIME_ERROR_BLOCK);
  EXPECT_EQ(swageRuntimeLaunch(0, 1, 32, 0, manyPointers.data(), 17,
                               scalars.data(), 0),
            SWAGE_RUNTIME_ERROR_ARGUMENTS);
  EXPECT_EQ(
      swageRuntimeLaunch(0, 1, 32, 0, pointers.data(), -1, scalars.data(), 1),
      SWAGE_RUNTIME_ERROR_ARGUMENTS);
}

TEST(RuntimeTest, DescribesItsOwnLaunchErrors) {
  for (int32_t code :
       {SWAGE_RUNTIME_ERROR_DRIVER, SWAGE_RUNTIME_ERROR_GRID,
        SWAGE_RUNTIME_ERROR_BLOCK, SWAGE_RUNTIME_ERROR_ARGUMENTS}) {
    const char *name = nullptr;
    const char *text = nullptr;
    swageRuntimeDescribe(code, &name, &text);
    ASSERT_NE(name, nullptr);
    ASSERT_NE(text, nullptr);
    EXPECT_NE(std::string(text), "");
  }
  const char *text = nullptr;
  const char *name = nullptr;
  swageRuntimeDescribe(SWAGE_RUNTIME_ERROR_GRID, &name, &text);
  EXPECT_STREQ(text, "grid_x must be a positive u32");
  swageRuntimeDescribe(SWAGE_RUNTIME_ERROR_BLOCK, &name, &text);
  EXPECT_STREQ(text, "block_x must be in 1..1024");
}

} // namespace
