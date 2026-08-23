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

std::unique_ptr<Pass> createSegmentedReductionToSCFPass();
std::unique_ptr<Pass>
createSegmentedReductionToGPUPass(int64_t blockSize = 0,
                                  bool useTaskIds = false);
void registerSegmentedReductionPasses();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H
