//===- FixedBlockToGPU.cpp - Fixed-block GPU lowering --------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/FixedBlockToGPU/FixedBlockToGPU.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Pass/Pass.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::swage {
namespace {

LogicalResult verifyPointerType(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return success(memref && memref.getRank() == 1 &&
                 memref.getElementType().isF32());
}

LogicalResult verifyVectorWidth(Operation *op, int64_t blockSize) {
  for (Type type :
       llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
    auto vector = dyn_cast<VectorType>(type);
    if (!vector)
      continue;
    if (vector.getRank() != 1)
      return op->emitError("only rank-one vectors are supported");
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

LogicalResult verifyFixedVectorAdd(func::FuncOp function, int64_t blockSize) {
  FunctionType type = function.getFunctionType();
  if (type.getNumInputs() != 4 || type.getNumResults() != 0 ||
      failed(verifyPointerType(type.getInput(0))) ||
      failed(verifyPointerType(type.getInput(1))) ||
      failed(verifyPointerType(type.getInput(2))) ||
      !type.getInput(3).isInteger(32))
    return function.emitError(
        "fixed vector add requires three rank-one f32 memrefs and one i32");
  if (!llvm::all_of(type.getInputs().take_front(3), [](Type input) {
        return cast<MemRefType>(input).getMemorySpaceAsInt() == 0;
      }))
    return function.emitError(
        "only default-memory-space pointers are supported");
  if (!function.getBody().hasOneBlock())
    return function.emitError(
        "fixed vector add requires one straight-line block");

  unsigned programIds = 0;
  unsigned gathers = 0;
  unsigned scatters = 0;
  unsigned floatAdds = 0;
  LogicalResult result = success();
  function.walk([&](Operation *op) {
    if (op == function.getOperation())
      return WalkResult::advance();
    if (failed(result))
      return WalkResult::interrupt();
    if (failed(verifyVectorWidth(op, blockSize))) {
      result = failure();
      return WalkResult::interrupt();
    }
    if (auto programId = dyn_cast<ProgramIdOp>(op)) {
      ++programIds;
      if (programId.getAxis() != 0) {
        programId.emitError("only swage.program_id axis 0 is supported");
        result = failure();
        return WalkResult::interrupt();
      }
    } else if (isa<vector::GatherOp>(op)) {
      ++gathers;
    } else if (isa<vector::ScatterOp>(op)) {
      ++scatters;
    } else if (auto add = dyn_cast<arith::AddFOp>(op)) {
      if (isa<VectorType>(add.getType()))
        ++floatAdds;
    } else if (!isa<arith::ConstantOp, arith::MulIOp, vector::StepOp,
                    vector::BroadcastOp, arith::AddIOp, arith::IndexCastOp,
                    arith::CmpIOp, func::ReturnOp>(op)) {
      op->emitError("operation is unsupported by fixed vector-add lowering");
      result = failure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (failed(result))
    return failure();
  if (programIds != 1 || gathers != 2 || scatters != 1 || floatAdds != 1)
    return function.emitError(
        "expected one program_id, two gathers, one f32 add, and one scatter");

  auto gatherOps = llvm::to_vector(function.getOps<vector::GatherOp>());
  auto scatter = *function.getOps<vector::ScatterOp>().begin();
  auto add = scatter.getValueToStore().getDefiningOp<arith::AddFOp>();
  if (!add)
    return scatter.emitError("scatter value must be the vector f32 add");
  if (gatherOps[0].getBase() != function.getArgument(0) ||
      gatherOps[1].getBase() != function.getArgument(1) ||
      scatter.getBase() != function.getArgument(2) ||
      add.getLhs() != gatherOps[0].getResult() ||
      add.getRhs() != gatherOps[1].getResult() ||
      gatherOps[0].getIndices() != gatherOps[1].getIndices() ||
      gatherOps[0].getIndices() != scatter.getIndices() ||
      gatherOps[0].getMask() != gatherOps[1].getMask() ||
      gatherOps[0].getMask() != scatter.getMask())
    return function.emitError(
        "gathers, add, and scatter do not form a fixed vector add");
  Value indices = gatherOps[0].getIndices();
  Value mask = gatherOps[0].getMask();
  Value programId = (*function.getOps<ProgramIdOp>().begin()).getResult();
  if (!hasZeroOffsets(gatherOps[0].getOffsets()) ||
      !hasZeroOffsets(gatherOps[1].getOffsets()) ||
      !hasZeroOffsets(scatter.getOffsets()) ||
      !hasCanonicalOffsetsAndMask(indices, mask, function.getArgument(3),
                                  programId, blockSize))
    return scatter.emitError(
        "fixed vector add must use canonical program offsets and bounds mask");
  return success();
}

void buildKernel(ModuleOp module, func::FuncOp source, int64_t blockSize) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(builder, loc,
                                            source.getName().str() + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  auto kernelType = FunctionType::get(module.getContext(),
                                      {pointer, pointer, pointer, i32}, {});
  auto kernel =
      gpu::GPUFuncOp::create(builder, loc, source.getName(), kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());

  Block *entry = &kernel.getBody().front();
  builder.setInsertionPointToStart(entry);

  Value blockId = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value base = arith::MulIOp::create(builder, loc, blockId, block);
  Value offset = arith::AddIOp::create(builder, loc, base, threadId);
  Value n = arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                       entry->getArgument(3));
  Value inBounds =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt, offset, n);
  Value byteOffset =
      arith::IndexCastOp::create(builder, loc, builder.getI64Type(), offset);

  scf::IfOp::create(
      builder, loc, inBounds, [&](OpBuilder &body, Location bodyLoc) {
        Value xAddress = LLVM::GEPOp::create(body, bodyLoc, pointer, f32,
                                             entry->getArgument(0), byteOffset);
        Value yAddress = LLVM::GEPOp::create(body, bodyLoc, pointer, f32,
                                             entry->getArgument(1), byteOffset);
        Value outputAddress = LLVM::GEPOp::create(
            body, bodyLoc, pointer, f32, entry->getArgument(2), byteOffset);
        Value x = LLVM::LoadOp::create(body, bodyLoc, f32, xAddress);
        Value y = LLVM::LoadOp::create(body, bodyLoc, f32, yAddress);
        Value sum = arith::AddFOp::create(body, bodyLoc, x, y);
        LLVM::StoreOp::create(body, bodyLoc, sum, outputAddress);
        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
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
    return "Lower the M3 fixed vector-add subset to one GPU x-thread per lane";
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gpu::GPUDialect, LLVM::LLVMDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (blockSize <= 0) {
      getOperation().emitError("block-size must be a positive integer");
      return signalPassFailure();
    }
    auto functions = llvm::to_vector(getOperation().getOps<func::FuncOp>());
    if (functions.size() != 1) {
      getOperation().emitError("expected exactly one kernel function");
      return signalPassFailure();
    }
    if (failed(verifyFixedVectorAdd(functions.front(), blockSize)))
      return signalPassFailure();
    buildKernel(getOperation(), functions.front(), blockSize);
  }

private:
  Option<int64_t> blockSize{*this, "block-size",
                            llvm::cl::desc("fixed x block size"),
                            llvm::cl::init(0)};
};

} // namespace

std::unique_ptr<Pass> createFixedBlockToGPUPass(int64_t blockSize) {
  return std::make_unique<FixedBlockToGPUPass>(blockSize);
}

void registerFixedBlockToGPUPass() { PassRegistration<FixedBlockToGPUPass>(); }

} // namespace mlir::swage
