// lib/Conversion/SegmentedReduction/SegmentedReduction.cpp
//===- SegmentedReduction.cpp - Segmented reduction lowering ------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Conversion/SwagePlanToGPU/SwagePlanToGPU.h"
#include "swage/Conversion/SwagePlanToSCF/SwagePlanToSCF.h"
#include "swage/Conversion/SwageToPlan/SwageToPlan.h"

#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"
#include "swage/Target/TargetDescription.h"

using namespace mlir;

namespace mlir::swage {
namespace {

class SegmentedReductionToSCFPass
    : public PassWrapper<SegmentedReductionToSCFPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SegmentedReductionToSCFPass)

  SegmentedReductionToSCFPass() = default;
  SegmentedReductionToSCFPass(const SegmentedReductionToSCFPass &other)
      : PassWrapper(other) {
    selectedFunction = other.selectedFunction.getValue();
  }

  StringRef getArgument() const final {
    return "swage-segmented-reduction-to-scf";
  }
  StringRef getDescription() const final {
    return "Lower every segment function to sequential SCF loops";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, func::FuncDialect, memref::MemRefDialect,
                scf::SCFDialect, swage_plan::SwagePlanDialect>();
  }

  void runOnOperation() final {
    // The oracle is planned and then converted, like a kernel.
    PlanOptions options;
    options.schedules = {PlanSchedule::Sequential};
    options.function = selectedFunction;
    if (failed(planSegmentFunctions(getOperation(), options, nvidiaTarget())) ||
        failed(convertPlanToSCF(getOperation())))
      signalPassFailure();
  }

private:
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Lower only this function instead of every function "
                     "that holds Swage operations")};
};

class SegmentedReductionToGPUPass
    : public PassWrapper<SegmentedReductionToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SegmentedReductionToGPUPass)

  SegmentedReductionToGPUPass() = default;
  SegmentedReductionToGPUPass(const SegmentedReductionToGPUPass &other)
      : PassWrapper(other), target(other.target) {
    blockSize = other.blockSize.getValue();
    useTaskIds = other.useTaskIds.getValue();
    fusedMixed = other.fusedMixed.getValue();
    persistent = other.persistent.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }
  SegmentedReductionToGPUPass(const TargetDescription &target,
                              int64_t requestedBlockSize, bool requestedTaskIds,
                              bool requestedFusedMixed,
                              bool requestedPersistent, StringRef function)
      : target(&target) {
    blockSize = requestedBlockSize;
    useTaskIds = requestedTaskIds;
    fusedMixed = requestedFusedMixed;
    persistent = requestedPersistent;
    selectedFunction = function.str();
  }

  StringRef getArgument() const final {
    return "swage-segmented-reduction-to-gpu";
  }
  StringRef getDescription() const final {
    return "Lower every segment function to a GPU kernel: one block per "
           "segment, or the task-id, fused mixed, or persistent schedule an "
           "option selects";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    memref::MemRefDialect, NVVM::NVVMDialect, scf::SCFDialect,
                    swage_plan::SwagePlanDialect>();
  }

  void runOnOperation() final {
    if (blockSize <= 0) {
      getOperation().emitError()
          << "block-size must be a positive integer, got "
          << blockSize.getValue();
      return signalPassFailure();
    }
    if (blockSize > target->maxBlockThreads) {
      getOperation().emitError()
          << "block-size must be at most " << target->maxBlockThreads
          << ", got " << blockSize.getValue();
      return signalPassFailure();
    }
    if (fusedMixed && blockSize != target->ctaBlockThreads) {
      getOperation().emitError()
          << "fused mixed lowering requires block-size "
          << target->ctaBlockThreads << ", got " << blockSize.getValue();
      return signalPassFailure();
    }
    if (persistent && blockSize != target->persistentBlockThreads) {
      getOperation().emitError()
          << "persistent lowering requires block-size "
          << target->persistentBlockThreads << ", got " << blockSize.getValue();
      return signalPassFailure();
    }
    // The persistent and fused kernels have ABIs of their own and always
    // load segment IDs from their task buffers, so neither can honor the
    // task-ID ABI option.
    if (persistent && useTaskIds) {
      getOperation().emitError(
          "persistent lowering does not accept use-task-ids; the persistent "
          "kernel always loads segment IDs from its own task queues");
      return signalPassFailure();
    }
    if (fusedMixed && useTaskIds) {
      getOperation().emitError(
          "fused mixed lowering does not accept use-task-ids; the fused "
          "kernel always loads segment IDs from its own task buffer");
      return signalPassFailure();
    }
    if (!target->admitsBlockThreads(blockSize)) {
      // Read the option first: streaming the option object itself prints its
      // value as a character.
      int64_t requested = blockSize;
      getOperation().emitError()
          << "block-size must give a power-of-two warp count, got " << requested
          << " (" << target->subgroupCount(requested) << " warps)";
      return signalPassFailure();
    }
    ModuleOp module = getOperation();
    // Every schedule is planned and then converted.
    PlanOptions options;
    options.schedules = {persistent   ? PlanSchedule::Persistent
                         : fusedMixed ? PlanSchedule::FusedMixed
                         : useTaskIds ? PlanSchedule::TaskIds
                                      : PlanSchedule::Direct};
    options.blockThreads = blockSize;
    options.function = selectedFunction;
    if (failed(planSegmentFunctions(module, options, *target)) ||
        failed(convertPlanToGPU(module, *target)))
      signalPassFailure();
  }

private:
  const TargetDescription *target = &nvidiaTarget();
  Option<int64_t> blockSize{*this, "block-size",
                            llvm::cl::desc("CTA x block size"),
                            llvm::cl::init(0)};
  Option<bool> useTaskIds{
      *this, "use-task-ids",
      llvm::cl::desc("Load segment IDs through the internal task ABI"),
      llvm::cl::init(false)};
  Option<bool> fusedMixed{
      *this, "fused-mixed",
      llvm::cl::desc("Fuse warp and CTA task schedules into one kernel"),
      llvm::cl::init(false)};
  Option<bool> persistent{
      *this, "persistent",
      llvm::cl::desc("Emit the experimental private persistent queue kernel; "
                     "requires block-size 512"),
      llvm::cl::init(false)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Lower only this function instead of every function "
                     "that holds Swage operations")};
};

class SplitSegmentedReductionToGPUPass
    : public PassWrapper<SplitSegmentedReductionToGPUPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitSegmentedReductionToGPUPass)

  SplitSegmentedReductionToGPUPass() = default;
  SplitSegmentedReductionToGPUPass(
      const SplitSegmentedReductionToGPUPass &other)
      : PassWrapper(other) {
    merge = other.merge.getValue();
    selectedFunction = other.selectedFunction.getValue();
  }
  SplitSegmentedReductionToGPUPass(bool requestedMerge, StringRef function) {
    merge = requestedMerge;
    selectedFunction = function.str();
  }

  StringRef getArgument() const final {
    return "swage-split-segmented-reduction-to-gpu";
  }
  StringRef getDescription() const final {
    return "Lower every capture-free sum or max function to a private split "
           "stage";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect,
                    swage_plan::SwagePlanDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    // Each stage is planned and then converted.
    PlanOptions options;
    options.schedules = {merge ? PlanSchedule::SplitMerge
                               : PlanSchedule::SplitPartial};
    options.function = selectedFunction;
    if (failed(planSegmentFunctions(module, options, nvidiaTarget())) ||
        failed(convertPlanToGPU(module, nvidiaTarget())))
      signalPassFailure();
  }

private:
  Option<bool> merge{
      *this, "merge",
      llvm::cl::desc("Emit the merge stage over scratch partials instead of "
                     "the partial stage over input ranges"),
      llvm::cl::init(false)};
  Option<std::string> selectedFunction{
      *this, "function",
      llvm::cl::desc("Lower only this function instead of every function "
                     "that holds Swage operations")};
};

} // namespace

std::unique_ptr<Pass> createSegmentedReductionToSCFPass() {
  return std::make_unique<SegmentedReductionToSCFPass>();
}

std::unique_ptr<Pass> createSegmentedReductionToGPUPass(int64_t blockSize,
                                                        bool useTaskIds,
                                                        bool fusedMixed,
                                                        StringRef function) {
  return std::make_unique<SegmentedReductionToGPUPass>(
      nvidiaTarget(), blockSize, useTaskIds, fusedMixed, false, function);
}

std::unique_ptr<Pass>
createSegmentedReductionToGPUPass(int64_t blockSize, bool useTaskIds,
                                  bool fusedMixed,
                                  const TargetDescription &target) {
  return std::make_unique<SegmentedReductionToGPUPass>(
      target, blockSize, useTaskIds, fusedMixed, false, "");
}

std::unique_ptr<Pass>
createPersistentSegmentedReductionToGPUPass(StringRef function) {
  const TargetDescription &target = nvidiaTarget();
  return std::make_unique<SegmentedReductionToGPUPass>(
      target, target.persistentBlockThreads, false, false, true, function);
}

std::unique_ptr<Pass> createSplitPartialReductionToGPUPass(StringRef function) {
  return std::make_unique<SplitSegmentedReductionToGPUPass>(false, function);
}

std::unique_ptr<Pass> createSplitMergeReductionToGPUPass(StringRef function) {
  return std::make_unique<SplitSegmentedReductionToGPUPass>(true, function);
}

void registerSegmentedReductionPasses() {
  PassRegistration<SegmentedReductionToSCFPass>();
  PassRegistration<SegmentedReductionToGPUPass>();
  PassRegistration<SplitSegmentedReductionToGPUPass>();
}

} // namespace mlir::swage
