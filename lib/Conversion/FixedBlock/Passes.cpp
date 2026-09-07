// lib/Conversion/FixedBlock/Passes.cpp
//===- Passes.cpp - Fixed-block lowering passes -------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"
#include "swage/Conversion/FixedBlock/FixedBlock.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

#include <utility>

using namespace mlir;

namespace mlir::swage {
namespace {

FailureOr<std::pair<func::FuncOp, detail::FixedElementwiseKind>>
admitFixedElementwise(ModuleOp module, int64_t blockSize) {
  if (blockSize <= 0) {
    module.emitError("block-size must be a positive integer");
    return failure();
  }
  if (blockSize > 1024) {
    module.emitError("block-size must be at most 1024");
    return failure();
  }
  auto functions = llvm::to_vector(module.getOps<func::FuncOp>());
  if (functions.size() != 1) {
    module.emitError("expected exactly one kernel function");
    return failure();
  }
  auto kind = detail::verifyFixedElementwise(functions.front(), blockSize);
  if (failed(kind))
    return failure();
  return std::make_pair(functions.front(), *kind);
}

class FixedBlockToGPUPass
    : public PassWrapper<FixedBlockToGPUPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixedBlockToGPUPass)

  FixedBlockToGPUPass() = default;
  FixedBlockToGPUPass(const FixedBlockToGPUPass &other) : PassWrapper(other) {
    blockSize = other.blockSize.getValue();
  }
  explicit FixedBlockToGPUPass(int64_t requestedBlockSize) {
    blockSize = requestedBlockSize;
  }

  StringRef getArgument() const final { return "swage-fixed-block-to-gpu"; }
  StringRef getDescription() const final {
    return "Lower the fixed elementwise subset to one GPU x-thread per lane";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    NVVM::NVVMDialect, scf::SCFDialect>();
  }

  void runOnOperation() final {
    auto function = admitFixedElementwise(getOperation(), blockSize);
    if (failed(function))
      return signalPassFailure();
    detail::buildFixedGPUProgram(getOperation(), function->first, blockSize,
                                 function->second);
  }

private:
  Option<int64_t> blockSize{*this, "block-size",
                            llvm::cl::desc("fixed x block size"),
                            llvm::cl::init(0)};
};

class FixedBlockToHostPass
    : public PassWrapper<FixedBlockToHostPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FixedBlockToHostPass)

  FixedBlockToHostPass() = default;
  FixedBlockToHostPass(const FixedBlockToHostPass &other) : PassWrapper(other) {
    blockSize = other.blockSize.getValue();
  }
  explicit FixedBlockToHostPass(int64_t requestedBlockSize) {
    blockSize = requestedBlockSize;
  }

  StringRef getArgument() const final { return "swage-fixed-block-to-host"; }
  StringRef getDescription() const final {
    return "Lower the fixed elementwise subset to one sequential host call";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, func::FuncDialect, LLVM::LLVMDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    auto function = admitFixedElementwise(getOperation(), blockSize);
    if (failed(function))
      return signalPassFailure();
    detail::buildFixedHostProgram(getOperation(), function->first, blockSize,
                                  function->second);
  }

private:
  Option<int64_t> blockSize{*this, "block-size",
                            llvm::cl::desc("fixed vector width"),
                            llvm::cl::init(0)};
};

} // namespace

std::unique_ptr<Pass> createFixedBlockToGPUPass(int64_t blockSize) {
  return std::make_unique<FixedBlockToGPUPass>(blockSize);
}

std::unique_ptr<Pass> createFixedBlockToHostPass(int64_t blockSize) {
  return std::make_unique<FixedBlockToHostPass>(blockSize);
}

void registerFixedBlockPasses() {
  PassRegistration<FixedBlockToGPUPass>();
  PassRegistration<FixedBlockToHostPass>();
}

} // namespace mlir::swage
