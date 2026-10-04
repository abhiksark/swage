// include/swage/Conversion/SwagePlanToSCF/SwagePlanToSCF.h
//===- SwagePlanToSCF.h - Sequential plans to loops ------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SWAGEPLANTOSCF_SWAGEPLANTOSCF_H
#define SWAGE_CONVERSION_SWAGEPLANTOSCF_SWAGEPLANTOSCF_H

#include "llvm/Support/LogicalResult.h"

#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
} // namespace mlir

namespace mlir::swage {

/// Convert every `swage_plan.tasks policy<sequential>` of `module` to loops
/// over its memrefs: the CPU oracle. The function that holds the operation
/// stays, with its signature and its callers, and loses its `swage.role`
/// attributes, so the result parses without the Swage dialects. Every other
/// operation is left as it is. The module is unchanged when the conversion
/// fails.
llvm::LogicalResult convertPlanToSCF(ModuleOp module);

/// `--swage-plan-to-scf`: `convertPlanToSCF`.
std::unique_ptr<Pass> createSwagePlanToSCFPass();
void registerSwagePlanToSCFPass();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SWAGEPLANTOSCF_SWAGEPLANTOSCF_H
