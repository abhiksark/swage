//===- swage-opt.cpp - Swage optimizer driver -------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "swage/Dialect/Swage/IR/SwageDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::swage::SwageDialect, mlir::func::FuncDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::scf::SCFDialect, mlir::memref::MemRefDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Swage optimizer driver\n", registry));
}
