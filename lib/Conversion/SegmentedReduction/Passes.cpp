// lib/Conversion/SegmentedReduction/Passes.cpp
//===- Passes.cpp - Segmented reduction passes
//-------------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "SegmentProgram.h"
#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"

using namespace mlir;

namespace mlir::swage {
namespace {

using namespace detail;

void buildPlanningCompanion(ModuleOp module, func::FuncOp semanticFunction,
                            const SemanticBufferRoles &roles,
                            int32_t warpMaxElements, int32_t ctaChunkElements) {
  OpBuilder builder(module.getContext());
  Location loc = semanticFunction.getLoc();
  Type taskRange = swage_plan::TaskRangeType::get(module.getContext());
  auto functionType = builder.getFunctionType(
      {roles.values.getType(), roles.offsets.getType()}, taskRange);

  builder.setInsertionPointAfter(semanticFunction);
  auto companion = func::FuncOp::create(
      builder, loc, semanticFunction.getName().str() + "__swage_plan",
      functionType);
  companion.setPrivate();
  Block *entry = companion.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value valueCount =
      memref::DimOp::create(builder, loc, entry->getArgument(0), zero);
  Value offsetCount =
      memref::DimOp::create(builder, loc, entry->getArgument(1), zero);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value segmentCount = arith::SubIOp::create(builder, loc, offsetCount, one);
  Value valueCountI32 = arith::IndexCastOp::create(
      builder, loc, builder.getI32Type(), valueCount);
  Value segmentCountI32 = arith::IndexCastOp::create(
      builder, loc, builder.getI32Type(), segmentCount);
  ArrayAttr policies = builder.getArrayAttr(
      {swage_plan::TaskPolicyAttr::get(module.getContext(),
                                       swage_plan::TaskPolicy::Warp),
       swage_plan::TaskPolicyAttr::get(module.getContext(),
                                       swage_plan::TaskPolicy::CTA)});
  auto tasks = swage_plan::ClassifyOp::create(
      builder, loc, taskRange, entry->getArgument(1), valueCountI32,
      segmentCountI32, semanticFunction.getName(),
      static_cast<uint32_t>(warpMaxElements),
      static_cast<uint32_t>(ctaChunkElements), policies);
  func::ReturnOp::create(builder, loc, tasks.getResult());
}

class SegmentedReductionToSCFPass
    : public PassWrapper<SegmentedReductionToSCFPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SegmentedReductionToSCFPass)

  StringRef getArgument() const final {
    return "swage-segmented-reduction-to-scf";
  }
  StringRef getDescription() const final {
    return "Lower one canonical segmented sum or max to sequential SCF loops";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    FailureOr<func::FuncOp> function = findSegmentedReduction(getOperation());
    if (failed(function))
      return signalPassFailure();
    SegmentProgramAnalysis analysis;
    if (failed(analyzeSegmentProgram(*function, analysis)))
      return signalPassFailure();
    SegmentProgram program;
    detachSegmentProgram(analysis, program);
    buildSequentialProgram(*function, analysis.roles, program);
  }
};

class SegmentedReductionToGPUPass
    : public PassWrapper<SegmentedReductionToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SegmentedReductionToGPUPass)

  SegmentedReductionToGPUPass() = default;
  SegmentedReductionToGPUPass(const SegmentedReductionToGPUPass &other)
      : PassWrapper(other) {
    blockSize = other.blockSize.getValue();
    useTaskIds = other.useTaskIds.getValue();
    fusedMixed = other.fusedMixed.getValue();
    forcedKind = other.forcedKind;
  }
  SegmentedReductionToGPUPass(int64_t requestedBlockSize,
                              SegmentedExecutionKind requestedKind)
      : forcedKind(requestedKind) {
    blockSize = requestedBlockSize;
  }

  StringRef getArgument() const final {
    return "swage-segmented-reduction-to-gpu";
  }
  StringRef getDescription() const final {
    return "Lower one canonical segmented sum or max to one CTA per segment";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    SegmentedExecutionKind kind = executionKind();
    if (blockSize <= 0) {
      getOperation().emitError("block-size must be a positive integer");
      return signalPassFailure();
    }
    if (blockSize > 1024) {
      getOperation().emitError("block-size must be at most 1024");
      return signalPassFailure();
    }
    if (kind == SegmentedExecutionKind::FusedMixed && blockSize != 128) {
      getOperation().emitError("fused mixed lowering requires block-size 128");
      return signalPassFailure();
    }
    if (kind == SegmentedExecutionKind::Persistent && blockSize != 512) {
      getOperation().emitError("persistent lowering requires block-size 512");
      return signalPassFailure();
    }
    FailureOr<func::FuncOp> function = findSegmentedReduction(getOperation());
    if (failed(function))
      return signalPassFailure();
    SegmentProgramAnalysis analysis;
    if (failed(analyzeSegmentProgram(*function, analysis)))
      return signalPassFailure();
    if (kind != SegmentedExecutionKind::Direct &&
        failed(verifyPlanningProgram(analysis)))
      return signalPassFailure();
    SegmentProgram program;
    detachSegmentProgram(analysis, program);
    buildGPUProgram(getOperation(), *function, analysis.roles, program,
                    blockSize, kind);
  }

private:
  SegmentedExecutionKind executionKind() const {
    if (forcedKind)
      return *forcedKind;
    if (fusedMixed)
      return SegmentedExecutionKind::FusedMixed;
    if (useTaskIds)
      return SegmentedExecutionKind::TaskIds;
    return SegmentedExecutionKind::Direct;
  }

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
  // Factory-created passes provide a closed mode directly; registered CLI
  // passes decode the stable boolean options with fused-before-task precedence.
  std::optional<SegmentedExecutionKind> forcedKind;
};

class SplitSegmentedReductionToGPUPass
    : public PassWrapper<SplitSegmentedReductionToGPUPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SplitSegmentedReductionToGPUPass)

  SplitSegmentedReductionToGPUPass() = default;
  SplitSegmentedReductionToGPUPass(
      const SplitSegmentedReductionToGPUPass &other)
      : PassWrapper(other), merge(other.merge) {}
  explicit SplitSegmentedReductionToGPUPass(bool requestedMerge)
      : merge(requestedMerge) {}

  StringRef getArgument() const final {
    return "swage-split-segmented-reduction-to-gpu";
  }
  StringRef getDescription() const final {
    return "Lower one identity sum to a private split reduction stage";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    FailureOr<func::FuncOp> function = findSegmentedReduction(getOperation());
    if (failed(function))
      return signalPassFailure();
    SegmentProgramAnalysis analysis;
    if (failed(analyzeSegmentProgram(*function, analysis)) ||
        failed(verifyPlanningProgram(analysis)))
      return signalPassFailure();
    buildSplitGPUProgram(getOperation(), *function, analysis.roles, merge);
  }

private:
  bool merge = false;
};

class SwageToPlanPass
    : public PassWrapper<SwageToPlanPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SwageToPlanPass)

  SwageToPlanPass() = default;
  SwageToPlanPass(const SwageToPlanPass &other) : PassWrapper(other) {
    warpMaxElements = other.warpMaxElements.getValue();
    ctaChunkElements = other.ctaChunkElements.getValue();
  }
  SwageToPlanPass(int64_t requestedWarpMaxElements,
                  int64_t requestedCtaChunkElements) {
    warpMaxElements = requestedWarpMaxElements;
    ctaChunkElements = requestedCtaChunkElements;
  }

  StringRef getArgument() const final { return "swage-to-plan"; }
  StringRef getDescription() const final {
    return "Add runtime classification for one identity segmented sum";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect, swage_plan::SwagePlanDialect>();
  }

  void runOnOperation() final {
    ModuleOp module = getOperation();
    if (warpMaxElements <= 0 || ctaChunkElements <= 0 ||
        warpMaxElements > ctaChunkElements ||
        ctaChunkElements > std::numeric_limits<int32_t>::max()) {
      module.emitError("planning limits must satisfy 0 < warp-max-elements <= "
                       "cta-chunk-elements <= INT32_MAX");
      return signalPassFailure();
    }

    SmallVector<func::FuncOp> functions(module.getOps<func::FuncOp>());
    if (functions.size() != 1) {
      module.emitError() << "planning requires exactly one function, found "
                         << functions.size();
      return signalPassFailure();
    }
    func::FuncOp function = functions.front();
    std::string companionName = function.getName().str() + "__swage_plan";
    if (SymbolTable::lookupSymbolIn(module.getOperation(), companionName)) {
      module.emitError() << "planning companion symbol @" << companionName
                         << " already exists";
      return signalPassFailure();
    }
    SegmentProgramAnalysis analysis;
    if (failed(analyzeSegmentProgram(function, analysis)) ||
        failed(verifyPlanningProgram(analysis)))
      return signalPassFailure();

    buildPlanningCompanion(module, function, analysis.roles,
                           static_cast<int32_t>(warpMaxElements),
                           static_cast<int32_t>(ctaChunkElements));
  }

private:
  Option<int64_t> warpMaxElements{
      *this, "warp-max-elements",
      llvm::cl::desc("Maximum segment length admitted for warp policy"),
      llvm::cl::init(32)};
  Option<int64_t> ctaChunkElements{
      *this, "cta-chunk-elements",
      llvm::cl::desc("Maximum input elements in one CTA task"),
      llvm::cl::init(4096)};
};

} // namespace

std::unique_ptr<Pass> createSegmentedReductionToSCFPass() {
  return std::make_unique<SegmentedReductionToSCFPass>();
}

std::unique_ptr<Pass> createSegmentedReductionToGPUPass(int64_t blockSize,
                                                        bool useTaskIds,
                                                        bool fusedMixed) {
  detail::SegmentedExecutionKind kind = detail::SegmentedExecutionKind::Direct;
  if (fusedMixed)
    kind = detail::SegmentedExecutionKind::FusedMixed;
  else if (useTaskIds)
    kind = detail::SegmentedExecutionKind::TaskIds;
  return std::make_unique<SegmentedReductionToGPUPass>(blockSize, kind);
}

std::unique_ptr<Pass> createPersistentSegmentedReductionToGPUPass() {
  return std::make_unique<SegmentedReductionToGPUPass>(
      512, detail::SegmentedExecutionKind::Persistent);
}

std::unique_ptr<Pass> createSplitPartialReductionToGPUPass() {
  return std::make_unique<SplitSegmentedReductionToGPUPass>(false);
}

std::unique_ptr<Pass> createSplitMergeReductionToGPUPass() {
  return std::make_unique<SplitSegmentedReductionToGPUPass>(true);
}

std::unique_ptr<Pass> createSwageToPlanPass(int64_t warpMaxElements,
                                            int64_t ctaChunkElements) {
  return std::make_unique<SwageToPlanPass>(warpMaxElements, ctaChunkElements);
}

void registerSegmentedReductionPasses() {
  PassRegistration<SegmentedReductionToSCFPass>();
  PassRegistration<SegmentedReductionToGPUPass>();
  PassRegistration<SwageToPlanPass>();
}

} // namespace mlir::swage
