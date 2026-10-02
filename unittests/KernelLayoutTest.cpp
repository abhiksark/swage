// unittests/KernelLayoutTest.cpp
//===- KernelLayoutTest.cpp - Kernel parameter layout tests ---------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Dialect/SwagePlan/IR/KernelLayout.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace mlir::swage_plan {
namespace {

std::vector<std::string> namesOf(KernelKind kind) {
  std::vector<std::string> names;
  for (KernelArgument argument : kernelLayout(kind).arguments())
    names.push_back(kernelArgumentName(argument).str());
  return names;
}

using Names = std::vector<std::string>;

// The layouts are private ABIs the host launches against. A change here is a
// change of every launch tuple of that kernel.
TEST(KernelLayoutTest, EachKernelTakesItsDocumentedParameters) {
  EXPECT_EQ(
      namesOf(KernelKind::Direct),
      (Names{"values", "offsets", "output", "value_count", "segment_count"}));
  EXPECT_EQ(namesOf(KernelKind::TaskIds),
            (Names{"values", "offsets", "output", "task_ids", "value_count",
                   "task_count", "segment_count"}));
  EXPECT_EQ(namesOf(KernelKind::FusedMixed),
            (Names{"values", "offsets", "output", "task_ids", "value_count",
                   "warp_task_count", "cta_task_count", "segment_count"}));
  EXPECT_EQ(namesOf(KernelKind::SplitPartial),
            (Names{"values", "partial_ranges", "scratch", "value_count",
                   "partial_count"}));
  EXPECT_EQ(namesOf(KernelKind::SplitMerge),
            (Names{"scratch", "output", "merge_records", "partial_count",
                   "merge_count", "segment_count"}));
  // The merge that reads the extent of a split segment takes the range
  // records after the merge records, and the three counts of the merge.
  EXPECT_EQ(namesOf(KernelKind::SplitMergeExtent),
            (Names{"scratch", "output", "merge_records", "partial_ranges",
                   "partial_count", "merge_count", "segment_count"}));
  EXPECT_EQ(
      namesOf(KernelKind::Persistent),
      (Names{"values", "offsets", "output", "warp_ids", "cta_ids",
             "partial_ranges", "partial_merge_ids", "merge_records", "scratch",
             "counters", "value_count", "warp_task_count", "cta_task_count",
             "partial_count", "merge_count", "segment_count"}));
}

TEST(KernelLayoutTest, EveryKernelTakesItsBuffersBeforeItsCounts) {
  for (KernelKind kind :
       {KernelKind::Direct, KernelKind::TaskIds, KernelKind::FusedMixed,
        KernelKind::SplitPartial, KernelKind::SplitMerge,
        KernelKind::SplitMergeExtent, KernelKind::Persistent}) {
    SCOPED_TRACE(static_cast<int>(kind));
    bool sawCount = false;
    for (KernelArgument argument : kernelLayout(kind).arguments()) {
      if (isBuffer(argument))
        EXPECT_FALSE(sawCount) << kernelArgumentName(argument).str();
      else
        sawCount = true;
    }
    EXPECT_TRUE(sawCount);
  }
}

TEST(KernelLayoutTest, FindsAnArgumentByWhatItIs) {
  constexpr KernelLayout taskIds = kernelLayout(KernelKind::TaskIds);
  static_assert(taskIds.indexOf(KernelArgument::TaskIds) == 3);
  static_assert(taskIds.indexOf(KernelArgument::SegmentCount) == 6);
  static_assert(!taskIds.has(KernelArgument::Scratch));

  const KernelLayout merge = kernelLayout(KernelKind::SplitMerge);
  EXPECT_EQ(merge.indexOf(KernelArgument::Scratch), 0U);
  EXPECT_EQ(merge.indexOf(KernelArgument::Output), 1U);
  EXPECT_EQ(merge.indexOf(KernelArgument::SegmentCount), 5U);
  EXPECT_FALSE(merge.has(KernelArgument::Values));

  const KernelLayout persistent = kernelLayout(KernelKind::Persistent);
  EXPECT_EQ(persistent.indexOf(KernelArgument::Counters), 9U);
  EXPECT_EQ(persistent.indexOf(KernelArgument::ValueCount), 10U);
  EXPECT_EQ(persistent.size(), 16U);
}

} // namespace
} // namespace mlir::swage_plan
