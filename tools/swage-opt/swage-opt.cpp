// tools/swage-opt/swage-opt.cpp
//===- swage-opt.cpp - Swage optimizer driver -------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "swage/Conversion/FixedBlockToGPU/FixedBlockToGPU.h"
#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::swage::registerFixedBlockToGPUPass();
  mlir::swage::registerSegmentedReductionPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::swage::SwageDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect,
                  mlir::vector::VectorDialect, mlir::gpu::GPUDialect,
                  mlir::LLVM::LLVMDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Swage optimizer driver\n", registry));
}
