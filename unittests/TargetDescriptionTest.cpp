// unittests/TargetDescriptionTest.cpp
//===- TargetDescriptionTest.cpp - Swage target description tests ---------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Target/TargetDescription.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "swage-c/Target.h"
#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "llvm/Support/MathExtras.h"
#include "gtest/gtest.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::swage {
namespace {

constexpr const char *segmentedSum = R"mlir(
module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
)mlir";

/// The block sizes the lowering admitted before the description existed: a
/// positive size up to 1024 whose warp count, with a partly filled last warp
/// counted as one, is a power of two.
bool hasPowerOfTwoWarpCount(int64_t blockSize) {
  return blockSize > 0 && blockSize <= 1024 &&
         llvm::isPowerOf2_64((blockSize - 1) / 32 + 1);
}

/// The width and offset of every XOR shuffle a task-ID kernel of
/// `blockThreads` threads holds when it is lowered for `target`.
std::vector<std::pair<int64_t, int64_t>>
shufflesOf(const TargetDescription &target, int64_t blockThreads) {
  MLIRContext context;
  context.loadDialect<SwageDialect, func::FuncDialect, arith::ArithDialect,
                      memref::MemRefDialect>();
  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(segmentedSum, &context);
  EXPECT_TRUE(static_cast<bool>(module));
  PassManager manager(&context);
  manager.addPass(createSegmentedReductionToGPUPass(
      blockThreads, /*useTaskIds=*/true, /*fusedMixed=*/false, target));
  EXPECT_TRUE(succeeded(manager.run(*module)));

  std::vector<std::pair<int64_t, int64_t>> shuffles;
  module->walk([&](gpu::ShuffleOp shuffle) {
    llvm::APInt width;
    llvm::APInt offset;
    EXPECT_EQ(shuffle.getMode(), gpu::ShuffleMode::XOR);
    EXPECT_TRUE(matchPattern(shuffle.getWidth(), m_ConstantInt(&width)));
    EXPECT_TRUE(matchPattern(shuffle.getOffset(), m_ConstantInt(&offset)));
    shuffles.emplace_back(width.getSExtValue(), offset.getSExtValue());
  });
  return shuffles;
}

TEST(TargetDescriptionTest, DescribesTheOneTarget) {
  const TargetDescription &target = nvidiaTarget();

  EXPECT_EQ(target.name, "nvidia");
  EXPECT_EQ(target.triple, "nvptx64-nvidia-cuda");
  EXPECT_EQ(target.processorPrefix, "sm_");
  EXPECT_EQ(
      std::vector<uint16_t>(target.processors.begin(), target.processors.end()),
      (std::vector<uint16_t>{80, 86, 87, 88, 89, 90, 100, 101, 103, 110, 120,
                             121}));
  // The upstream all-reduce lowering and the NVVM conversion hard-code a
  // subgroup of 32 threads. A description that named another width would
  // disagree with the code they emit, so check both again when the LLVM pin
  // moves.
  EXPECT_EQ(target.subgroupWidth, 32);
  EXPECT_EQ(target.maxBlockThreads, 1024);
  EXPECT_EQ(target.ctaBlockThreads, 128);
  EXPECT_EQ(target.splitBlockThreads, 512);
  EXPECT_EQ(target.persistentBlockThreads, 512);
  EXPECT_EQ(target.persistentPartialClaim, 4);
  EXPECT_EQ(target.persistentWarpClaim, 8);
  EXPECT_EQ(target.defaultWarpMaxElements, 32);
  EXPECT_EQ(target.defaultCtaChunkElements, 4096);
  EXPECT_EQ(target.slotsPerBlock(target.ctaBlockThreads), 4);
  EXPECT_EQ(target.slotsPerBlock(target.persistentBlockThreads), 16);
}

TEST(TargetDescriptionTest, AdmitsTheBlockSizesOfThePowerOfTwoRule) {
  const TargetDescription &target = nvidiaTarget();
  int admitted = 0;
  for (int64_t threads = -64; threads <= 2048; ++threads) {
    SCOPED_TRACE(threads);
    EXPECT_EQ(target.admitsBlockThreads(threads),
              hasPowerOfTwoWarpCount(threads));
    admitted += target.admitsBlockThreads(threads);
  }
  // Six warp counts (1, 2, 4, 8, 16, 32), each reached by 32 block sizes.
  EXPECT_EQ(admitted, 192);
}

TEST(TargetDescriptionTest, TheWarpReductionReadsTheSubgroupWidth) {
  using Shuffles = std::vector<std::pair<int64_t, int64_t>>;

  EXPECT_EQ(shufflesOf(nvidiaTarget(), 32),
            (Shuffles{{32, 1}, {32, 2}, {32, 4}, {32, 8}, {32, 16}}));

  // A copy of the description with half the subgroup width. A block of 16
  // threads is then one subgroup, and its butterfly has four steps of width
  // 16. This shows the emitter reads the field; no device has this width.
  TargetDescription narrow = nvidiaTarget();
  narrow.subgroupWidth = 16;
  EXPECT_EQ(shufflesOf(narrow, 16),
            (Shuffles{{16, 1}, {16, 2}, {16, 4}, {16, 8}}));
}

TEST(TargetDescriptionTest, TheCRecordHoldsTheSameValues) {
  const TargetDescription &target = nvidiaTarget();

  SwageTargetDescription record = swageGetTargetDescription();

  auto text = [](MlirStringRef value) {
    return std::string(value.data, value.length);
  };
  EXPECT_EQ(text(record.name), target.name);
  EXPECT_EQ(text(record.triple), target.triple);
  EXPECT_EQ(text(record.processorPrefix), target.processorPrefix);
  EXPECT_EQ(std::vector<uint16_t>(record.processors,
                                  record.processors + record.processorCount),
            std::vector<uint16_t>(target.processors.begin(),
                                  target.processors.end()));
  EXPECT_EQ(record.subgroupWidth, target.subgroupWidth);
  EXPECT_EQ(record.maxBlockThreads, target.maxBlockThreads);
  EXPECT_EQ(record.ctaBlockThreads, target.ctaBlockThreads);
  EXPECT_EQ(record.splitBlockThreads, target.splitBlockThreads);
  EXPECT_EQ(record.persistentBlockThreads, target.persistentBlockThreads);
  EXPECT_EQ(record.persistentPartialClaim, target.persistentPartialClaim);
  EXPECT_EQ(record.persistentWarpClaim, target.persistentWarpClaim);
  EXPECT_EQ(record.defaultWarpMaxElements, target.defaultWarpMaxElements);
  EXPECT_EQ(record.defaultCtaChunkElements, target.defaultCtaChunkElements);
}

} // namespace
} // namespace mlir::swage
