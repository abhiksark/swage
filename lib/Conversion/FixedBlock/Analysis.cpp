// lib/Conversion/FixedBlock/Analysis.cpp
//===- Analysis.cpp - Fixed-block admission -----------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "Analysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::swage::detail {
namespace {

LogicalResult verifyPointerType(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return success(
      memref && memref.getRank() == 1 &&
      (memref.getElementType().isF32() || memref.getElementType().isF16() ||
       isa<Float8E4M3FNType, Float8E5M2Type>(memref.getElementType())) &&
      memref.getLayout().isIdentity());
}

LogicalResult verifyVectorWidth(Operation *op, int64_t blockSize) {
  for (Type type :
       llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
    auto vector = dyn_cast<VectorType>(type);
    if (!vector)
      continue;
    if (vector.getRank() != 1 || vector.isScalable())
      return op->emitError("only fixed rank-one vectors are supported");
    if (vector.getShape().front() != blockSize)
      return op->emitError()
             << "vector width " << vector.getShape().front()
             << " does not match requested block size " << blockSize;
  }
  return success();
}

bool isConstantInteger(Value value, int64_t expected) {
  llvm::APInt constant;
  return matchPattern(value, m_ConstantInt(&constant)) &&
         constant.getSExtValue() == expected;
}

bool hasZeroOffsets(ValueRange offsets) {
  return llvm::all_of(offsets,
                      [](Value value) { return isConstantInteger(value, 0); });
}

bool hasCanonicalOffsetsAndMask(Value indices, Value mask, Value n,
                                Value programId, int64_t blockSize) {
  auto add = indices.getDefiningOp<arith::AddIOp>();
  if (!add || !add.getRhs().getDefiningOp<vector::StepOp>())
    return false;
  auto broadcast = add.getLhs().getDefiningOp<vector::BroadcastOp>();
  if (!broadcast)
    return false;
  auto multiply = broadcast.getSource().getDefiningOp<arith::MulIOp>();
  if (!multiply || multiply.getLhs() != programId ||
      !isConstantInteger(multiply.getRhs(), blockSize))
    return false;

  auto compare = mask.getDefiningOp<arith::CmpIOp>();
  if (!compare || compare.getPredicate() != arith::CmpIPredicate::slt ||
      compare.getLhs() != indices)
    return false;
  auto nBroadcast = compare.getRhs().getDefiningOp<vector::BroadcastOp>();
  if (!nBroadcast)
    return false;
  auto nCast = nBroadcast.getSource().getDefiningOp<arith::IndexCastOp>();
  return nCast && nCast.getIn() == n;
}

LogicalResult verifyFixedElementwiseSignature(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0 ||
      failed(verifyPointerType(type.getInput(0))) ||
      failed(verifyPointerType(type.getInput(1))) ||
      failed(verifyPointerType(type.getInput(2))) ||
      !type.getInput(3).isInteger(32))
    return function.emitError("fixed elementwise operation requires three "
                              "rank-one identity-layout memrefs "
                              "of f32, f16, f8E4M3FN, or f8E5M2 and one i32");
  Type elementType = cast<MemRefType>(type.getInput(0)).getElementType();
  if (!llvm::all_of(type.getInputs().take_front(3), [&](Type input) {
        return cast<MemRefType>(input).getElementType() == elementType;
      }))
    return function.emitError(
        "fixed elementwise operation requires identical pointer element types");
  if (!llvm::all_of(type.getInputs().take_front(3), [](Type input) {
        return cast<MemRefType>(input).getMemorySpaceAsInt() == 0;
      }))
    return function.emitError(
        "only default-memory-space pointers are supported");
  if (!function.getBody().hasOneBlock())
    return function.emitError(
        "fixed elementwise operation requires one straight-line block");
  return success();
}

struct FixedElementwiseOpCounts {
  unsigned programIds = 0;
  unsigned gathers = 0;
  unsigned scatters = 0;
  unsigned floatOperations = 0;
};

LogicalResult
classifyFixedElementwiseOperation(Operation *op, int64_t blockSize,
                                  FixedElementwiseOpCounts &counts) {
  if (failed(verifyVectorWidth(op, blockSize)))
    return failure();
  if (auto programId = dyn_cast<ProgramIdOp>(op)) {
    ++counts.programIds;
    if (programId.getAxis() != 0)
      return programId.emitError("only swage.program_id axis 0 is supported");
  } else if (isa<vector::GatherOp>(op)) {
    ++counts.gathers;
  } else if (isa<vector::ScatterOp>(op)) {
    ++counts.scatters;
  } else if (isa<arith::AddFOp, arith::MulFOp>(op)) {
    ++counts.floatOperations;
  } else if (!isa<arith::ConstantOp, arith::MulIOp, vector::StepOp,
                  vector::BroadcastOp, arith::AddIOp, arith::IndexCastOp,
                  arith::CmpIOp, func::ReturnOp>(op)) {
    return op->emitError(
        "operation is unsupported by fixed elementwise lowering");
  }
  return success();
}

LogicalResult
collectFixedElementwiseOperations(func::FuncOp function, int64_t blockSize,
                                  FixedElementwiseOpCounts &counts) {
  LogicalResult result = success();
  function.walk([&](Operation *op) {
    if (op == function.getOperation())
      return WalkResult::advance();
    if (failed(classifyFixedElementwiseOperation(op, blockSize, counts))) {
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

LogicalResult
verifyFixedElementwiseCounts(func::FuncOp function,
                             const FixedElementwiseOpCounts &counts) {
  if (counts.programIds != 1 || counts.gathers != 2 || counts.scatters != 1 ||
      counts.floatOperations != 1)
    return function.emitError("expected one program_id, two gathers, one "
                              "floating-point add or multiply, and one "
                              "scatter");
  return success();
}

LogicalResult verifyFixedElementwiseConnections(func::FuncOp function,
                                                vector::GatherOp lhsGather,
                                                vector::GatherOp rhsGather,
                                                vector::ScatterOp scatter,
                                                Operation *arithmetic) {
  if (lhsGather.getBase() != function.getArgument(0) ||
      rhsGather.getBase() != function.getArgument(1) ||
      scatter.getBase() != function.getArgument(2) ||
      arithmetic->getOperand(0) != lhsGather.getResult() ||
      arithmetic->getOperand(1) != rhsGather.getResult() ||
      lhsGather.getIndices() != rhsGather.getIndices() ||
      lhsGather.getIndices() != scatter.getIndices() ||
      lhsGather.getMask() != rhsGather.getMask() ||
      lhsGather.getMask() != scatter.getMask())
    return function.emitError("gathers, arithmetic, and scatter do not form a "
                              "fixed elementwise operation");
  return success();
}

LogicalResult verifyCanonicalElementwiseAccesses(func::FuncOp function,
                                                 vector::GatherOp lhsGather,
                                                 vector::GatherOp rhsGather,
                                                 vector::ScatterOp scatter,
                                                 int64_t blockSize) {
  Value indices = lhsGather.getIndices();
  Value mask = lhsGather.getMask();
  Value programId = (*function.getOps<ProgramIdOp>().begin()).getResult();
  if (!hasZeroOffsets(lhsGather.getOffsets()) ||
      !hasZeroOffsets(rhsGather.getOffsets()) ||
      !hasZeroOffsets(scatter.getOffsets()) ||
      !hasCanonicalOffsetsAndMask(indices, mask, function.getArgument(3),
                                  programId, blockSize))
    return scatter.emitError("fixed elementwise operation must use canonical "
                             "program offsets and bounds mask");
  return success();
}

FailureOr<FixedElementwiseKind>
verifyFixedElementwiseDataflow(func::FuncOp function, int64_t blockSize) {
  auto gathers = llvm::to_vector(function.getOps<vector::GatherOp>());
  auto scatter = *function.getOps<vector::ScatterOp>().begin();
  Operation *arithmetic = scatter.getValueToStore().getDefiningOp();
  if (!arithmetic || !isa<arith::AddFOp, arith::MulFOp>(arithmetic)) {
    scatter.emitError(
        "scatter value must be the vector floating-point add or multiply");
    return failure();
  }
  if (failed(verifyFixedElementwiseConnections(function, gathers[0], gathers[1],
                                               scatter, arithmetic)))
    return failure();
  if (failed(verifyCanonicalElementwiseAccesses(
          function, gathers[0], gathers[1], scatter, blockSize)))
    return failure();
  return isa<arith::AddFOp>(arithmetic) ? FixedElementwiseKind::Add
                                        : FixedElementwiseKind::Multiply;
}

} // namespace

FailureOr<FixedElementwiseKind> verifyFixedElementwise(func::FuncOp function,
                                                       int64_t blockSize) {
  if (failed(verifyFixedElementwiseSignature(function)))
    return failure();
  FixedElementwiseOpCounts counts;
  if (failed(collectFixedElementwiseOperations(function, blockSize, counts)) ||
      failed(verifyFixedElementwiseCounts(function, counts)))
    return failure();
  return verifyFixedElementwiseDataflow(function, blockSize);
}

} // namespace mlir::swage::detail
