// include/swage/Conversion/SegmentedReduction/SegmentedReduction.h
//===- SegmentedReduction.h - Segmented reduction lowering ----*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H
#define SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::swage {

struct TargetDescription;

std::unique_ptr<Pass> createSegmentedReductionToSCFPass();
std::unique_ptr<Pass> createSegmentedReductionToGPUPass(
    int64_t blockSize = 0, bool useTaskIds = false, bool fusedMixed = false);
/// The same pass for an explicit target description, which must outlive the
/// pass. The overload above lowers for `nvidiaTarget()`.
std::unique_ptr<Pass>
createSegmentedReductionToGPUPass(int64_t blockSize, bool useTaskIds,
                                  bool fusedMixed,
                                  const TargetDescription &target);
std::unique_ptr<Pass> createPersistentSegmentedReductionToGPUPass();
std::unique_ptr<Pass> createSplitPartialReductionToGPUPass();
std::unique_ptr<Pass> createSplitMergeReductionToGPUPass();
std::unique_ptr<Pass> createSwageToPlanPass(int64_t warpMaxElements,
                                            int64_t ctaChunkElements);
void registerSegmentedReductionPasses();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H
