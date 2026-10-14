// include/swage/Conversion/SwagePlanToGPU/SwagePlanToGPU.h
//===- SwagePlanToGPU.h - Plan functions to GPU kernels -------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SWAGEPLANTOGPU_SWAGEPLANTOGPU_H
#define SWAGE_CONVERSION_SWAGEPLANTOGPU_SWAGEPLANTOGPU_H

#include "llvm/Support/LogicalResult.h"

#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
} // namespace mlir

namespace mlir::swage {

struct TargetDescription;

/// Convert every plan function of `module` to a `gpu.module` that holds its
/// kernel, and leave every other operation as it is. A plan function is a
/// `func.func` with a `swage_plan.block_threads` attribute. The module is
/// unchanged when the conversion fails.
llvm::LogicalResult convertPlanToGPU(ModuleOp module,
                                     const TargetDescription &target);

/// `--swage-plan-to-gpu`: `convertPlanToGPU` for `nvidiaTarget()`.
std::unique_ptr<Pass> createSwagePlanToGPUPass();
/// The same pass for `target`, which must outlive the pass.
std::unique_ptr<Pass> createSwagePlanToGPUPass(const TargetDescription &target);
void registerSwagePlanToGPUPass();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SWAGEPLANTOGPU_SWAGEPLANTOGPU_H
