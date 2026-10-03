// lib/Conversion/FixedBlock/Analysis.h
//===- Analysis.h - Fixed-block admission ---------------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_LIB_CONVERSION_FIXEDBLOCK_ANALYSIS_H
#define SWAGE_LIB_CONVERSION_FIXEDBLOCK_ANALYSIS_H

#include "mlir/Support/LogicalResult.h"
#include <cstdint>

namespace mlir {
class Location;
class ModuleOp;
class OpBuilder;
class Type;
class Value;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace mlir::swage::detail {

enum class FixedElementwiseKind { Add, Multiply };

FailureOr<FixedElementwiseKind> verifyFixedElementwise(func::FuncOp function,
                                                       int64_t blockSize);
void buildFixedGPUProgram(ModuleOp module, func::FuncOp source,
                          int64_t blockSize, FixedElementwiseKind kind);
void buildFixedHostProgram(ModuleOp module, func::FuncOp source,
                           int64_t blockSize, FixedElementwiseKind kind);
void buildFixedScalarElementwise(OpBuilder &builder, Location loc,
                                 Type elementType, Value xBase, Value yBase,
                                 Value outputBase, Value offset,
                                 FixedElementwiseKind kind);

} // namespace mlir::swage::detail

#endif // SWAGE_LIB_CONVERSION_FIXEDBLOCK_ANALYSIS_H
