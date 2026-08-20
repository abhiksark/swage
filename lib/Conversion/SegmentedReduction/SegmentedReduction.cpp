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

/// Classification of one operation inside a Swage region.
enum class RegionOpStatus { Admitted, UnknownName, NonF32Result };

/// Operations a Swage region may contain.
///
/// The list is a whitelist rather than a purity test because both backends
/// must lower every admitted operation. Exponentials are written as
/// `math.exp2` of a scaled operand; `math.exp` is deliberately absent
/// because `--convert-gpu-to-nvvm` turns every `math` operation into a
/// libdevice call, and the PTX path links no libdevice.
RegionOpStatus classifyRegionOperation(Operation &operation) {
  static constexpr StringRef admitted[] = {
      "arith.constant", "arith.addf",     "arith.subf",     "arith.mulf",
      "arith.divf",     "arith.maximumf", "arith.minimumf", "math.exp2"};
  if (!llvm::is_contained(admitted, operation.getName().getStringRef()))
    return RegionOpStatus::UnknownName;
  if (!llvm::all_of(operation.getResultTypes(),
                    [](Type type) { return type.isF32(); }))
    return RegionOpStatus::NonF32Result;
  return RegionOpStatus::Admitted;
}

/// Verify a Swage region: an f32 element argument followed by one f32
/// argument per capture, admitted operations only, and an f32 yield.
LogicalResult verifyRegion(Operation *owner, Region &region,
                           unsigned captureCount) {
  Block &body = region.front();
  if (body.getNumArguments() != 1 + captureCount ||
      llvm::any_of(body.getArgumentTypes(),
                   [](Type type) { return !type.isF32(); }))
    return owner->emitError(
        "segment region requires an f32 element argument followed by f32 "
        "captures");
  for (Operation &operation : body.without_terminator()) {
    switch (classifyRegionOperation(operation)) {
    case RegionOpStatus::Admitted:
      break;
    case RegionOpStatus::UnknownName:
      return operation.emitError("operation is unsupported inside a segment "
                                 "region; exponentials must use math.exp2");
    case RegionOpStatus::NonF32Result:
      return operation.emitError("operation is unsupported inside a segment "
                                 "region; every result must be f32");
    }
  }
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

/// A per-element expression: the fused `swage.map` bodies in application
/// order, followed by the consumer's own body. `captures[i]` lists, for
/// `regions[i]`, the reduction stages whose results bind to that region's
/// capture arguments, in order.
struct ElementProgram {
  SmallVector<Region *> regions;
  SmallVector<SmallVector<unsigned>> captures;
};

/// One `swage.reduce` and the element expression feeding it.
struct ReductionStage {
  ElementProgram element;
  ReductionKind kind = ReductionKind::Sum;
};

/// An admitted segment program. It holds no handle into the source function:
/// every region is detached into a RegionOwner and every capture is a stage
/// index, so an emitter may erase the source IR before emitting.
struct SegmentProgram {
  SmallVector<ReductionStage> reductions;
  unsigned storedReduction = 0; ///< Index into `reductions`.
};

/// Owns the region bodies detached from the source function, and must
/// outlive the SegmentProgram pointing into it.
class RegionOwner {
public:
  Region *take(Region &body) {
    owned.push_back(std::make_unique<Region>());
    owned.back()->takeBody(body);
    return owned.back().get();
  }

private:
  SmallVector<std::unique_ptr<Region>> owned;
};

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

/// Walk back through fused maps to the root segment, collecting them in
/// application order. Every segment value in an admitted body is defined by
/// a map or by the single make_segment, so the walk always terminates.
SmallVector<MapOp> fusionChain(Value segment) {
  SmallVector<MapOp> chain;
  while (auto map = segment.getDefiningOp<MapOp>()) {
    chain.push_back(map);
    segment = map.getSegment();
  }
  std::reverse(chain.begin(), chain.end());
  return chain;
}

/// Admit one canonical segment program, or emit the first rule violation.
LogicalResult admitSegmentProgram(func::FuncOp function, RegionOwner &owner,
                                  SegmentProgram &program) {
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

  SmallVector<SegmentIdOp> segmentIds;
  SmallVector<MakeSegmentOp> segments;
  SmallVector<MapOp> maps;
  SmallVector<ReduceOp> reductions;
  SmallVector<memref::StoreOp> stores;
  SmallVector<func::ReturnOp> returns;
  for (Operation &operation : function.getBody().front()) {
    if (auto segmentId = dyn_cast<SegmentIdOp>(operation))
      segmentIds.push_back(segmentId);
    else if (auto segment = dyn_cast<MakeSegmentOp>(operation))
      segments.push_back(segment);
    else if (auto map = dyn_cast<MapOp>(operation))
      maps.push_back(map);
    else if (auto reduction = dyn_cast<ReduceOp>(operation))
      reductions.push_back(reduction);
    else if (auto store = dyn_cast<memref::StoreOp>(operation))
      stores.push_back(store);
    else if (auto returnOp = dyn_cast<func::ReturnOp>(operation))
      returns.push_back(returnOp);
    else
      return operation.emitError(
          "operation is unsupported by segmented reduction lowering");
  }
  if (segmentIds.size() != 1 || segments.size() != 1 || reductions.empty() ||
      returns.size() != 1)
    return function.emitError(
        "segmented reduction requires one segment_id, one make_segment, at "
        "least one reduce, and one return");
  SegmentIdOp segmentId = segmentIds.front();
  MakeSegmentOp segment = segments.front();
  if (segmentId.getAxis() != 0)
    return segmentId.emitError("only swage.segment_id axis 0 is supported");
  if (segment.getValues() != function.getArgument(0) ||
      segment.getOffsets() != function.getArgument(1) ||
      segment.getSegmentId() != segmentId.getResult())
    return segment.emitError(
        "make_segment must bind the function values and offsets at segment_id");
  if (stores.size() != 1)
    return function.emitError(
        "segmented reduction requires exactly one output terminal: a "
        "memref.store of a reduction at output[segment_id]");
  memref::StoreOp store = stores.front();

  for (MapOp map : maps) {
    if (!map.getResult().hasOneUse())
      return map.emitError(
          "swage.map result must have exactly one segment consumer; a mapped "
          "segment is never materialized");
    Operation *consumer = *map.getResult().getUsers().begin();
    if (!isa<MapOp, ReduceOp>(consumer))
      return map.emitError(
          "swage.map result must have exactly one segment consumer; a mapped "
          "segment is never materialized");
  }

  DenseMap<Operation *, unsigned> stageOf;
  for (auto [index, reduction] : llvm::enumerate(reductions))
    stageOf[reduction.getOperation()] = index;

  SmallVector<Operation *> capturing;
  for (MapOp map : maps)
    capturing.push_back(map.getOperation());
  for (ReduceOp reduction : reductions)
    capturing.push_back(reduction.getOperation());
  for (Operation *operation : capturing) {
    ValueRange captures = isa<MapOp>(operation)
                              ? cast<MapOp>(operation).getCaptures()
                              : cast<ReduceOp>(operation).getCaptures();
    for (Value capture : captures)
      if (!capture.getDefiningOp<ReduceOp>() || !capture.getType().isF32())
        return operation->emitError(
            "segment captures must be f32 results of a swage.reduce in the "
            "same function");
  }

  for (ReduceOp reduction : reductions) {
    ReductionKind kind = reduction.getKind();
    if (kind != ReductionKind::Sum && kind != ReductionKind::Max)
      return reduction.emitError(
          "segmented reduction supports only kind<sum> and kind<max>");
  }

  for (MapOp map : maps)
    if (failed(verifyRegion(map, map.getBody(), map.getCaptures().size())))
      return failure();
  for (ReduceOp reduction : reductions)
    if (failed(verifyRegion(reduction, reduction.getBody(),
                            reduction.getCaptures().size())))
      return failure();

  auto storedReduction = store.getValue().getDefiningOp<ReduceOp>();
  if (!storedReduction || !stageOf.contains(storedReduction.getOperation()) ||
      store.getMemRef() != function.getArgument(2) ||
      store.getIndices().size() != 1 ||
      store.getIndices().front() != segmentId.getResult())
    return store.emitError(
        "segmented reduction result must be stored at output[segment_id]");
  if (!returns.front().getOperands().empty())
    return returns.front().emitError("segmented reduction must return void");

  // Every rule passed, so the program may now take ownership of the regions.
  program.storedReduction = stageOf.lookup(storedReduction.getOperation());
  auto takeElement = [&](Operation *consumer, ValueRange consumerCaptures,
                         Value consumerSegment) {
    ElementProgram element;
    auto append = [&](Operation *stage, Region &body, ValueRange captures) {
      SmallVector<unsigned> indices;
      for (Value capture : captures)
        indices.push_back(stageOf.lookup(capture.getDefiningOp()));
      element.regions.push_back(owner.take(body));
      element.captures.push_back(std::move(indices));
      (void)stage;
    };
    for (MapOp map : fusionChain(consumerSegment))
      append(map, map.getBody(), map.getCaptures());
    append(consumer, consumer->getRegion(0), consumerCaptures);
    return element;
  };
  for (ReduceOp reduction : reductions) {
    ReductionStage stage;
    stage.kind = reduction.getKind();
    stage.element =
        takeElement(reduction, reduction.getCaptures(), reduction.getSegment());
    program.reductions.push_back(std::move(stage));
  }
  return success();
}

/// Apply an admitted element expression to one loaded value.
Value evaluateElement(OpBuilder &builder, const ElementProgram &element,
                      Value value, ArrayRef<Value> reductions) {
  for (auto [region, captures] :
       llvm::zip_equal(element.regions, element.captures)) {
    SmallVector<Value> arguments{value};
    for (unsigned stage : captures)
      arguments.push_back(reductions[stage]);
    value = inlineRegion(builder, *region, arguments);
  }
  return value;
}

/// The identity element of a reduction kind.
Value identityFor(OpBuilder &builder, Location loc, ReductionKind kind) {
  FloatType f32 = builder.getF32Type();
  APFloat identity = kind == ReductionKind::Sum
                         ? APFloat(f32.getFloatSemantics(), 0)
                         : APFloat::getInf(f32.getFloatSemantics(), true);
  return arith::ConstantFloatOp::create(builder, loc, f32, identity);
}

/// Combine an accumulator with one element.
Value combine(OpBuilder &builder, Location loc, ReductionKind kind,
              Value accumulator, Value value) {
  if (kind == ReductionKind::Sum)
    return arith::AddFOp::create(builder, loc, accumulator, value).getResult();
  return arith::MaximumFOp::create(builder, loc, accumulator, value)
      .getResult();
}

void buildSequentialProgram(func::FuncOp function,
                            const SegmentProgram &program) {
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
        SmallVector<Value> results;
        for (const ReductionStage &stage : program.reductions) {
          Value identity = identityFor(outer, outerLoc, stage.kind);
          auto reduction = scf::ForOp::create(
              outer, outerLoc, start, end, one, ValueRange(identity),
              [&](OpBuilder &inner, Location innerLoc, Value index,
                  ValueRange accumulator) {
                Value value = memref::LoadOp::create(
                    inner, innerLoc, function.getArgument(0), index);
                value = evaluateElement(inner, stage.element, value, results);
                scf::YieldOp::create(inner, innerLoc,
                                     combine(inner, innerLoc, stage.kind,
                                             accumulator.front(), value));
              });
          results.push_back(reduction.getResult(0));
        }
        memref::StoreOp::create(outer, outerLoc,
                                results[program.storedReduction],
                                function.getArgument(2), segmentId);
        scf::YieldOp::create(outer, outerLoc);
      });
  func::ReturnOp::create(builder, loc);
}

void buildGPUProgram(ModuleOp module, func::FuncOp source,
                     const SegmentProgram &program, int64_t blockSize) {
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

        SmallVector<Value> results;
        for (const ReductionStage &stage : program.reductions) {
          Value identity = identityFor(body, bodyLoc, stage.kind);
          auto local = scf::ForOp::create(
              body, bodyLoc, first, end, block, ValueRange(identity),
              [&](OpBuilder &loop, Location loopLoc, Value index,
                  ValueRange accumulator) {
                Value index64 = arith::IndexCastOp::create(
                    loop, loopLoc, loop.getI64Type(), index);
                Value address =
                    LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                        entry->getArgument(0), index64);
                Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
                value = evaluateElement(loop, stage.element, value, results);
                scf::YieldOp::create(loop, loopLoc,
                                     combine(loop, loopLoc, stage.kind,
                                             accumulator.front(), value));
              });
          gpu::AllReduceOperation gpuKind =
              stage.kind == ReductionKind::Sum
                  ? gpu::AllReduceOperation::ADD
                  : gpu::AllReduceOperation::MAXIMUMF;
          auto operation =
              gpu::AllReduceOperationAttr::get(module.getContext(), gpuKind);
          // uniform = true, so the result is broadcast to every thread and
          // the lowering's trailing barrier fences this stage from the next.
          results.push_back(gpu::AllReduceOp::create(
              body, bodyLoc, local.getResult(0), operation, true));
        }

        Value total = results[program.storedReduction];
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
    // The owner must outlive the program, which points into it.
    RegionOwner owner;
    SegmentProgram program;
    if (failed(admitSegmentProgram(*function, owner, program)))
      return signalPassFailure();
    buildSequentialProgram(*function, program);
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
    RegionOwner owner;
    SegmentProgram program;
    if (failed(admitSegmentProgram(*function, owner, program)))
      return signalPassFailure();
    buildGPUProgram(getOperation(), *function, program, blockSize);
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
