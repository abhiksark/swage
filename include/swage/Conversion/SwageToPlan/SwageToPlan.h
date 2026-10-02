// include/swage/Conversion/SwageToPlan/SwageToPlan.h
//===- SwageToPlan.h - Segment functions to plan functions ----*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SWAGETOPLAN_SWAGETOPLAN_H
#define SWAGE_CONVERSION_SWAGETOPLAN_SWAGETOPLAN_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include <cstdint>
#include <memory>

namespace mlir {
class ModuleOp;
class Pass;
} // namespace mlir

namespace mlir::swage {

struct TargetDescription;

/// The kernel a plan function describes.
enum class PlanSchedule {
  /// One block of threads per segment.
  Direct,
  /// One block of threads per task; a task buffer names the segment.
  TaskIds,
  /// Warp tasks and block tasks of one task buffer in one launch.
  FusedMixed,
  /// One block of threads per chunk of a long segment: the first stage of a
  /// split reduction, whose kernel is named `<function>__partial`.
  SplitPartial,
  /// One block of threads per split segment, which reduces its partial
  /// results: the second stage, whose kernel is named `<function>__merge`.
  SplitMerge,
  /// No kernel: one thread visits the segments in order. The CPU oracle.
  Sequential,
};

struct PlanOptions {
  /// The kernels to plan, one plan function each, in this order. Each
  /// kernel is named once, and the sequential schedule stands alone.
  llvm::SmallVector<PlanSchedule, 2> schedules;
  /// The launch width of the direct and task-id kernels, in threads. The
  /// target fixes the width of every other kernel.
  int64_t blockThreads = 0;
  /// Plan only the function of this name. Empty plans every function that
  /// holds Swage operations.
  llvm::StringRef function;
};

/// Replace each selected segment function of `module` by one plan function
/// per schedule: a function with the parameter list of the kernel, a
/// `swage_plan.block_threads` attribute, and one task operation that holds
/// the reductions and stores of the program.
///
/// The sequential schedule plans a function in place: it keeps its
/// signature, its roles, and its callers, has no launch width, and its body
/// becomes one task operation of `policy<sequential>`.
///
/// Every function is admitted before any is changed, so `module` is
/// unchanged when this fails. Other functions are left as they are.
llvm::LogicalResult planSegmentFunctions(ModuleOp module,
                                         const PlanOptions &options,
                                         const TargetDescription &target);

/// Whether the segment function named `function` holds a program that host
/// classification can turn into tasks: one capture-free sum or max whose
/// result is stored per segment. Reads `module` and never changes it.
llvm::LogicalResult admitTaskProgram(ModuleOp module, llvm::StringRef function);

/// `--swage-to-plan`: `planSegmentFunctions` for `nvidiaTarget()`.
std::unique_ptr<Pass> createSwageToPlanPass();
void registerSwageToPlanPass();

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SWAGETOPLAN_SWAGETOPLAN_H
