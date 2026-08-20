// lib/Conversion/SegmentedReduction/SegmentedReduction.cpp
//===- SegmentedReduction.cpp - Segmented reduction lowering ------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::swage {
namespace {

bool isRankOneMemRef(Type type, Type elementType) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType() == elementType &&
         memref.getLayout().isIdentity() && memref.getMemorySpaceAsInt() == 0;
}

/// Operations a Swage region may contain.
///
/// The list is a whitelist rather than a purity test because both backends
/// must lower every admitted operation. Exponentials are written as
/// `math.exp2` of a scaled operand; `math.exp` is deliberately absent
/// because `--convert-gpu-to-nvvm` turns every `math` operation into a
/// libdevice call, and the PTX path links no libdevice.
bool isAdmittedRegionOperation(Operation &operation) {
  static constexpr StringRef admitted[] = {
      "arith.constant", "arith.addf",     "arith.subf",     "arith.mulf",
      "arith.divf",     "arith.maximumf", "arith.minimumf", "math.exp2"};
  if (!isMemoryEffectFree(&operation) || operation.getNumRegions() != 0)
    return false;
  if (!llvm::is_contained(admitted, operation.getName().getStringRef()))
    return false;
  return llvm::all_of(operation.getResultTypes(),
                      [](Type type) { return type.isF32(); });
}

/// Verify a Swage region: one block, f32 element argument, admitted
/// operations only, and an f32 yield.
LogicalResult verifyRegion(Operation *owner, Region &region) {
  if (!region.hasOneBlock())
    return owner->emitError("segment region requires one block");
  Block &body = region.front();
  if (body.getNumArguments() != 1 || !body.getArgument(0).getType().isF32())
    return owner->emitError("segment region requires one f32 element argument");
  for (Operation &operation : body.without_terminator())
    if (!isAdmittedRegionOperation(operation))
      return operation.emitError("operation is unsupported inside a segment "
                                 "region; exponentials must use math.exp2");
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || !yield.getValue().getType().isF32())
    return owner->emitError("segment region must yield an f32 value");
  return success();
}

/// Clone a verified region inline at the builder's insertion point and
/// return the mapped yielded value.
Value inlineRegion(OpBuilder &builder, Region &region, ValueRange arguments) {
  Block &body = region.front();
  IRMapping mapping;
  mapping.map(body.getArguments(), arguments);
  for (Operation &operation : body.without_terminator())
    builder.clone(operation, mapping);
  return mapping.lookup(cast<YieldOp>(body.getTerminator()).getValue());
}

FailureOr<func::FuncOp> findSegmentedReduction(ModuleOp module) {
  SmallVector<func::FuncOp> candidates;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    bool hasSwageOperation = false;
    function.walk([&](Operation *operation) {
      hasSwageOperation |=
          operation->getName().getDialectNamespace() == "swage";
    });
    if (hasSwageOperation)
      candidates.push_back(function);
  }
  if (candidates.size() != 1) {
    module.emitError()
        << "expected exactly one function containing Swage segment operations, "
           "found "
        << candidates.size();
    return failure();
  }
  return candidates.front();
}

LogicalResult verifySegmentedReduction(func::FuncOp function,
                                       ReductionKind &kind,
                                       ReduceOp &reduceOp) {
  FunctionType type = function.getFunctionType();
  Builder builder(function.getContext());
  if (type.getNumInputs() != 5 || type.getNumResults() != 0 ||
      !isRankOneMemRef(type.getInput(0), builder.getF32Type()) ||
      !isRankOneMemRef(type.getInput(1), builder.getI32Type()) ||
      !isRankOneMemRef(type.getInput(2), builder.getF32Type()) ||
      !type.getInput(3).isInteger(32) || !type.getInput(4).isInteger(32))
    return function.emitError(
        "segmented reduction requires rank-one f32 values, rank-one i32 "
        "offsets, rank-one f32 output, i32 value count, and i32 segment "
        "count");
  if (!function.getBody().hasOneBlock())
    return function.emitError("segmented reduction requires one block");

  auto segmentIds = llvm::to_vector(function.getOps<SegmentIdOp>());
  auto segments = llvm::to_vector(function.getOps<MakeSegmentOp>());
  auto reductions = llvm::to_vector(function.getOps<ReduceOp>());
  auto stores = llvm::to_vector(function.getOps<memref::StoreOp>());
  auto returns = llvm::to_vector(function.getOps<func::ReturnOp>());
  if (segmentIds.size() != 1 || segments.size() != 1 ||
      reductions.size() != 1 || stores.size() != 1 || returns.size() != 1)
    return function.emitError(
        "segmented reduction requires one segment_id, make_segment, reduce, "
        "output store, and return");

  for (Operation &operation : function.getBody().front()) {
    if (!isa<SegmentIdOp, MakeSegmentOp, ReduceOp, memref::StoreOp,
             func::ReturnOp>(operation))
      return operation.emitError(
          "operation is unsupported by segmented reduction lowering");
  }

  SegmentIdOp segmentId = segmentIds.front();
  MakeSegmentOp segment = segments.front();
  ReduceOp reduction = reductions.front();
  memref::StoreOp store = stores.front();
  if (segmentId.getAxis() != 0)
    return segmentId.emitError("only swage.segment_id axis 0 is supported");
  if (segment.getValues() != function.getArgument(0) ||
      segment.getOffsets() != function.getArgument(1) ||
      segment.getSegmentId() != segmentId.getResult())
    return segment.emitError(
        "make_segment must bind the function values and offsets at segment_id");
  if (reduction.getSegment() != segment.getResult() ||
      !reduction.getCaptures().empty())
    return reduction.emitError(
        "segmented reduction requires a capture-free reduction");
  kind = reduction.getKind();
  if (kind != ReductionKind::Sum && kind != ReductionKind::Max)
    return reduction.emitError(
        "segmented reduction supports only kind<sum> and kind<max>");
  if (failed(verifyRegion(reduction, reduction.getBody())))
    return failure();
  if (store.getValue() != reduction.getResult() ||
      store.getMemRef() != function.getArgument(2) ||
      store.getIndices().size() != 1 ||
      store.getIndices().front() != segmentId.getResult())
    return store.emitError(
        "segmented reduction result must be stored at output[segment_id]");
  if (!returns.front().getOperands().empty())
    return returns.front().emitError("segmented reduction must return void");
  reduceOp = reduction;
  return success();
}

void buildSequentialReduction(func::FuncOp function, ReductionKind kind,
                              Region &region) {
  Block &entry = function.getBody().front();
  while (!entry.empty())
    entry.back().erase();

  OpBuilder builder(function.getContext());
  Location loc = function.getLoc();
  builder.setInsertionPointToEnd(&entry);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value segmentCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), function.getArgument(4));
  scf::ForOp::create(
      builder, loc, zero, segmentCount, one, ValueRange(),
      [&](OpBuilder &outer, Location outerLoc, Value segmentId, ValueRange) {
        Value startI32 = memref::LoadOp::create(
            outer, outerLoc, function.getArgument(1), segmentId);
        Value next = arith::AddIOp::create(outer, outerLoc, segmentId, one);
        Value endI32 = memref::LoadOp::create(outer, outerLoc,
                                              function.getArgument(1), next);
        Value start = arith::IndexCastOp::create(
            outer, outerLoc, outer.getIndexType(), startI32);
        Value end = arith::IndexCastOp::create(outer, outerLoc,
                                               outer.getIndexType(), endI32);
        FloatType f32 = outer.getF32Type();
        APFloat identityValue =
            kind == ReductionKind::Sum
                ? APFloat(f32.getFloatSemantics(), 0)
                : APFloat::getInf(f32.getFloatSemantics(), true);
        Value identity =
            arith::ConstantFloatOp::create(outer, outerLoc, f32, identityValue);
        auto reduction = scf::ForOp::create(
            outer, outerLoc, start, end, one, ValueRange(identity),
            [&](OpBuilder &inner, Location innerLoc, Value index,
                ValueRange accumulator) {
              Value value = memref::LoadOp::create(
                  inner, innerLoc, function.getArgument(0), index);
              value = inlineRegion(inner, region, value);
              Value nextAccumulator =
                  kind == ReductionKind::Sum
                      ? arith::AddFOp::create(inner, innerLoc,
                                              accumulator.front(), value)
                            .getResult()
                      : arith::MaximumFOp::create(inner, innerLoc,
                                                  accumulator.front(), value)
                            .getResult();
              scf::YieldOp::create(inner, innerLoc, nextAccumulator);
            });
        memref::StoreOp::create(outer, outerLoc, reduction.getResult(0),
                                function.getArgument(2), segmentId);
        scf::YieldOp::create(outer, outerLoc);
      });
  func::ReturnOp::create(builder, loc);
}

void buildGPUReduction(ModuleOp module, func::FuncOp source, ReductionKind kind,
                       Region &region, int64_t blockSize) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(builder, loc,
                                            source.getName().str() + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  auto kernelType = FunctionType::get(
      module.getContext(), {pointer, pointer, pointer, i32, i32}, {});
  auto kernel =
      gpu::GPUFuncOp::create(builder, loc, source.getName(), kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());

  Block *entry = &kernel.getBody().front();
  builder.setInsertionPointToStart(entry);
  Value segmentId = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value segmentCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), entry->getArgument(4));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        segmentId, segmentCount);

  scf::IfOp::create(
      builder, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
        Value segmentId64 = arith::IndexCastOp::create(
            body, bodyLoc, body.getI64Type(), segmentId);
        Value startAddress = LLVM::GEPOp::create(
            body, bodyLoc, pointer, i32, entry->getArgument(1), segmentId64);
        Value startI32 = LLVM::LoadOp::create(body, bodyLoc, i32, startAddress);
        Value nextSegment =
            arith::AddIOp::create(body, bodyLoc, segmentId, one);
        Value nextSegment64 = arith::IndexCastOp::create(
            body, bodyLoc, body.getI64Type(), nextSegment);
        Value endAddress = LLVM::GEPOp::create(
            body, bodyLoc, pointer, i32, entry->getArgument(1), nextSegment64);
        Value endI32 = LLVM::LoadOp::create(body, bodyLoc, i32, endAddress);
        Value start = arith::IndexCastOp::create(body, bodyLoc,
                                                 body.getIndexType(), startI32);
        Value end = arith::IndexCastOp::create(body, bodyLoc,
                                               body.getIndexType(), endI32);
        Value first = arith::AddIOp::create(body, bodyLoc, start, threadId);
        FloatType f32Type = body.getF32Type();
        APFloat identityValue =
            kind == ReductionKind::Sum
                ? APFloat(f32Type.getFloatSemantics(), 0)
                : APFloat::getInf(f32Type.getFloatSemantics(), true);
        Value identity = arith::ConstantFloatOp::create(body, bodyLoc, f32Type,
                                                        identityValue);
        auto local = scf::ForOp::create(
            body, bodyLoc, first, end, block, ValueRange(identity),
            [&](OpBuilder &loop, Location loopLoc, Value index,
                ValueRange accumulator) {
              Value index64 = arith::IndexCastOp::create(
                  loop, loopLoc, loop.getI64Type(), index);
              Value address = LLVM::GEPOp::create(
                  loop, loopLoc, pointer, f32, entry->getArgument(0), index64);
              Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
              value = inlineRegion(loop, region, value);
              Value nextAccumulator =
                  kind == ReductionKind::Sum
                      ? arith::AddFOp::create(loop, loopLoc,
                                              accumulator.front(), value)
                            .getResult()
                      : arith::MaximumFOp::create(loop, loopLoc,
                                                  accumulator.front(), value)
                            .getResult();
              scf::YieldOp::create(loop, loopLoc, nextAccumulator);
            });
        gpu::AllReduceOperation gpuKind =
            kind == ReductionKind::Sum ? gpu::AllReduceOperation::ADD
                                       : gpu::AllReduceOperation::MAXIMUMF;
        auto operation =
            gpu::AllReduceOperationAttr::get(module.getContext(), gpuKind);
        Value total = gpu::AllReduceOp::create(
            body, bodyLoc, local.getResult(0), operation, true);
        Value firstThread = arith::CmpIOp::create(
            body, bodyLoc, arith::CmpIPredicate::eq, threadId, zero);
        scf::IfOp::create(
            body, bodyLoc, firstThread,
            [&](OpBuilder &store, Location storeLoc) {
              Value outputAddress =
                  LLVM::GEPOp::create(store, storeLoc, pointer, f32,
                                      entry->getArgument(2), segmentId64);
              LLVM::StoreOp::create(store, storeLoc, total, outputAddress);
              scf::YieldOp::create(store, storeLoc);
            });
        scf::YieldOp::create(body, bodyLoc);
      });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
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
    ReductionKind kind = ReductionKind::Sum;
    ReduceOp reduction;
    if (failed(verifySegmentedReduction(*function, kind, reduction)))
      return signalPassFailure();
    // The entry block is rebuilt from scratch, so detach the region first.
    Region region;
    region.takeBody(reduction.getBody());
    buildSequentialReduction(*function, kind, region);
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
  }
  explicit SegmentedReductionToGPUPass(int64_t requestedBlockSize) {
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
                    scf::SCFDialect>();
  }

  void runOnOperation() final {
    if (blockSize <= 0) {
      getOperation().emitError("block-size must be a positive integer");
      return signalPassFailure();
    }
    FailureOr<func::FuncOp> function = findSegmentedReduction(getOperation());
    if (failed(function))
      return signalPassFailure();
    ReductionKind kind = ReductionKind::Sum;
    ReduceOp reduction;
    if (failed(verifySegmentedReduction(*function, kind, reduction)))
      return signalPassFailure();
    buildGPUReduction(getOperation(), *function, kind, reduction.getBody(),
                      blockSize);
  }

private:
  Option<int64_t> blockSize{*this, "block-size",
                            llvm::cl::desc("CTA x block size"),
                            llvm::cl::init(0)};
};

} // namespace

std::unique_ptr<Pass> createSegmentedReductionToSCFPass() {
  return std::make_unique<SegmentedReductionToSCFPass>();
}

std::unique_ptr<Pass> createSegmentedReductionToGPUPass(int64_t blockSize) {
  return std::make_unique<SegmentedReductionToGPUPass>(blockSize);
}

void registerSegmentedReductionPasses() {
  PassRegistration<SegmentedReductionToSCFPass>();
  PassRegistration<SegmentedReductionToGPUPass>();
}

} // namespace mlir::swage
