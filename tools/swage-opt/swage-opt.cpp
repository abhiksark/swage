// tools/swage-opt/swage-opt.cpp
//===- swage-opt.cpp - Swage optimizer driver -------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
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

#include "swage/Conversion/FixedBlock/FixedBlock.h"
#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::swage::registerFixedBlockPasses();
  mlir::swage::registerSegmentedReductionPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::swage::SwageDialect, mlir::swage_plan::SwagePlanDialect,
                  mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect, mlir::scf::SCFDialect,
                  mlir::memref::MemRefDialect, mlir::vector::VectorDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect>();
  // GPU-to-NVVM consults the LLVM interfaces promised by loaded dialects,
  // including semantic vector/memref dialects eliminated by Swage lowering.
  mlir::arith::registerConvertArithToLLVMInterface(registry);
  mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
  mlir::registerConvertFuncToLLVMInterface(registry);
  mlir::index::registerConvertIndexToLLVMInterface(registry);
  mlir::registerConvertMathToLLVMInterface(registry);
  mlir::registerConvertMemRefToLLVMInterface(registry);
  mlir::registerConvertNVVMToLLVMInterface(registry);
  mlir::ub::registerConvertUBToLLVMInterface(registry);
  mlir::vector::registerConvertVectorToLLVMInterface(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Swage optimizer driver\n", registry));
}
