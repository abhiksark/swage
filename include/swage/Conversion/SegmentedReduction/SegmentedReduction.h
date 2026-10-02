// include/swage/Conversion/SegmentedReduction/SegmentedReduction.h
//===- SegmentedReduction.h - Segmented reduction lowering ----*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H
#define SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <memory>

namespace mlir {
class Pass;
}

namespace mlir::swage {

struct TargetDescription;

// Each pass lowers every function of the module that holds Swage operations
// and leaves the others as they are. A factory that takes `function` lowers
// only the function of that name when the name is not empty.

std::unique_ptr<Pass> createSegmentedReductionToSCFPass();
std::unique_ptr<Pass> createSegmentedReductionToGPUPass(
    int64_t blockSize = 0, bool useTaskIds = false, bool fusedMixed = false,
    llvm::StringRef function = "");
/// The same pass for an explicit target description, which must outlive the
/// pass. The overload above lowers for `nvidiaTarget()`.
std::unique_ptr<Pass>
createSegmentedReductionToGPUPass(int64_t blockSize, bool useTaskIds,
                                  bool fusedMixed,
                                  const TargetDescription &target);
std::unique_ptr<Pass>
createPersistentSegmentedReductionToGPUPass(llvm::StringRef function = "");
std::unique_ptr<Pass>
createSplitPartialReductionToGPUPass(llvm::StringRef function = "");
std::unique_ptr<Pass>
createSplitMergeReductionToGPUPass(llvm::StringRef function = "");
void registerSegmentedReductionPasses();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SEGMENTEDREDUCTION_SEGMENTEDREDUCTION_H
