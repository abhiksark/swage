// lib/Conversion/SegmentedReduction/SegmentedReduction.cpp
//===- SegmentedReduction.cpp - Segmented reduction lowering ------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SegmentedReduction/SegmentedReduction.h"

#include <limits>
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
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "swage/Dialect/SwagePlan/IR/SwagePlanOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::swage {
namespace {

bool isRankOneMemRef(Type type, Type elementType) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType() == elementType &&
         memref.getLayout().isIdentity() && !memref.getMemorySpace();
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
  // Malformed IR that bypassed the dialect verifier must fail here rather
  // than reach the unchecked dereferences below.
  if (!region.hasOneBlock())
    return owner->emitError("segment region requires exactly one block");
  Block &body = region.front();
  if (!body.mightHaveTerminator())
    return owner->emitError("segment region must yield an f32 value");
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

/// Where an admitted program writes its result.
enum class TerminalKind {
  ScalarStore, ///< One f32 per segment, at output[segment_id].
  MapStore     ///< One f32 per element, at output[element index].
};

/// An admitted segment program. It holds no handle into the source function:
/// every region is detached into a RegionOwner and every capture is a stage
/// index, so an emitter may erase the source IR before emitting.
struct SegmentProgram {
  SmallVector<ReductionStage> reductions;
  TerminalKind terminal = TerminalKind::ScalarStore;
  unsigned storedReduction = 0; ///< ScalarStore: index into `reductions`.
  ElementProgram mapStore;      ///< MapStore: the per-element expression.
};

/// Read-only admission result shared by every segmented-program consumer.
struct SegmentProgramAnalysis {
  SmallVector<SegmentIdOp> segmentIds;
  SmallVector<MakeSegmentOp> segments;
  SmallVector<MapOp> maps;
  SmallVector<ReduceOp> reductions;
  SmallVector<memref::StoreOp> stores;
  SmallVector<MapStoreOp> mapStores;
  SmallVector<func::ReturnOp> returns;
  ReduceOp storedReduction;
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

/// Analyze one canonical segment program without mutating it.
LogicalResult analyzeSegmentProgram(func::FuncOp function,
                                    SegmentProgramAnalysis &analysis) {
  FunctionType type = function.getFunctionType();
  Builder builder(function.getContext());
  if (type.getNumInputs() != 5 || type.getNumResults() != 0 ||
      !isRankOneMemRef(type.getInput(0), builder.getF32Type()) ||
      !isRankOneMemRef(type.getInput(1), builder.getI32Type()) ||
      !isRankOneMemRef(type.getInput(2), builder.getF32Type()) ||
      !type.getInput(3).isSignlessInteger(32) ||
      !type.getInput(4).isSignlessInteger(32))
    return function.emitError(
        "segmented reduction requires rank-one f32 values, rank-one i32 "
        "offsets, rank-one f32 output, i32 value count, and i32 segment "
        "count");
  if (!function.getBody().hasOneBlock())
    return function.emitError("segmented reduction requires one block");

  for (Operation &operation : function.getBody().front()) {
    if (auto segmentId = dyn_cast<SegmentIdOp>(operation))
      analysis.segmentIds.push_back(segmentId);
    else if (auto segment = dyn_cast<MakeSegmentOp>(operation))
      analysis.segments.push_back(segment);
    else if (auto map = dyn_cast<MapOp>(operation))
      analysis.maps.push_back(map);
    else if (auto reduction = dyn_cast<ReduceOp>(operation))
      analysis.reductions.push_back(reduction);
    else if (auto store = dyn_cast<memref::StoreOp>(operation))
      analysis.stores.push_back(store);
    else if (auto mapStore = dyn_cast<MapStoreOp>(operation))
      analysis.mapStores.push_back(mapStore);
    else if (auto returnOp = dyn_cast<func::ReturnOp>(operation))
      analysis.returns.push_back(returnOp);
    else
      return operation.emitError(
          "operation is unsupported by segmented reduction lowering");
  }
  if (analysis.segmentIds.size() != 1 || analysis.segments.size() != 1 ||
      analysis.reductions.empty() || analysis.returns.size() != 1)
    return function.emitError(
        "segmented reduction requires one segment_id, one make_segment, at "
        "least one reduce, and one return");
  SegmentIdOp segmentId = analysis.segmentIds.front();
  MakeSegmentOp segment = analysis.segments.front();
  if (segmentId.getAxis() != 0)
    return segmentId.emitError("only swage.segment_id axis 0 is supported");
  if (segment.getValues() != function.getArgument(0) ||
      segment.getOffsets() != function.getArgument(1) ||
      segment.getSegmentId() != segmentId.getResult())
    return segment.emitError(
        "make_segment must bind the function values and offsets at segment_id");
  if (analysis.stores.size() + analysis.mapStores.size() != 1)
    return function.emitError(
        "segmented reduction requires exactly one output terminal: a "
        "memref.store of a reduction at output[segment_id] or a "
        "swage.map_store into the output");

  for (MapOp map : analysis.maps) {
    if (!map.getResult().hasOneUse())
      return map.emitError(
          "swage.map result must have exactly one segment consumer; a mapped "
          "segment is never materialized");
    Operation *consumer = *map.getResult().getUsers().begin();
    if (!isa<MapOp, ReduceOp, MapStoreOp>(consumer))
      return map.emitError(
          "swage.map result must have exactly one segment consumer; a mapped "
          "segment is never materialized");
  }

  DenseMap<Operation *, unsigned> stageOf;
  for (auto [index, reduction] : llvm::enumerate(analysis.reductions))
    stageOf[reduction.getOperation()] = index;

  SmallVector<Operation *> capturing;
  for (MapOp map : analysis.maps)
    capturing.push_back(map.getOperation());
  for (ReduceOp reduction : analysis.reductions)
    capturing.push_back(reduction.getOperation());
  for (MapStoreOp mapStore : analysis.mapStores)
    capturing.push_back(mapStore.getOperation());
  for (Operation *operation : capturing) {
    ValueRange captures =
        llvm::TypeSwitch<Operation *, ValueRange>(operation)
            .Case<MapOp>([](MapOp map) { return map.getCaptures(); })
            .Case<ReduceOp>(
                [](ReduceOp reduce) { return reduce.getCaptures(); })
            .Case<MapStoreOp>(
                [](MapStoreOp store) { return store.getCaptures(); });
    for (Value capture : captures)
      if (!capture.getDefiningOp<ReduceOp>() || !capture.getType().isF32())
        return operation->emitError(
            "segment captures must be f32 results of a swage.reduce in the "
            "same function");
  }

  for (ReduceOp reduction : analysis.reductions) {
    ReductionKind kind = reduction.getKind();
    if (kind != ReductionKind::Sum && kind != ReductionKind::Max)
      return reduction.emitError(
          "segmented reduction supports only kind<sum> and kind<max>");
  }

  for (MapOp map : analysis.maps)
    if (failed(verifyRegion(map, map.getBody(), map.getCaptures().size())))
      return failure();
  for (ReduceOp reduction : analysis.reductions)
    if (failed(verifyRegion(reduction, reduction.getBody(),
                            reduction.getCaptures().size())))
      return failure();
  for (MapStoreOp mapStore : analysis.mapStores)
    if (failed(verifyRegion(mapStore, mapStore.getBody(),
                            mapStore.getCaptures().size())))
      return failure();

  if (analysis.mapStores.empty()) {
    memref::StoreOp store = analysis.stores.front();
    analysis.storedReduction = store.getValue().getDefiningOp<ReduceOp>();
    if (!analysis.storedReduction ||
        !stageOf.contains(analysis.storedReduction.getOperation()) ||
        store.getMemRef() != function.getArgument(2) ||
        store.getIndices().size() != 1 ||
        store.getIndices().front() != segmentId.getResult())
      return store.emitError(
          "segmented reduction result must be stored at output[segment_id]");
  } else if (analysis.mapStores.front().getOutput() !=
             function.getArgument(2)) {
    return analysis.mapStores.front().emitError(
        "swage.map_store must write the function output buffer");
  }
  if (!analysis.returns.front().getOperands().empty())
    return analysis.returns.front().emitError(
        "segmented reduction must return void");

  return success();
}

/// Detach regions only after read-only admission has succeeded.
void detachSegmentProgram(SegmentProgramAnalysis &analysis, RegionOwner &owner,
                          SegmentProgram &program) {
  DenseMap<Operation *, unsigned> stageOf;
  for (auto [index, reduction] : llvm::enumerate(analysis.reductions))
    stageOf[reduction.getOperation()] = index;
  program.terminal = analysis.mapStores.empty() ? TerminalKind::ScalarStore
                                                : TerminalKind::MapStore;
  if (analysis.mapStores.empty())
    program.storedReduction =
        stageOf.lookup(analysis.storedReduction.getOperation());
  auto takeElement = [&](Operation *consumer, ValueRange consumerCaptures,
                         Value consumerSegment) {
    ElementProgram element;
    auto append = [&](Region &body, ValueRange captures) {
      SmallVector<unsigned> indices;
      for (Value capture : captures)
        indices.push_back(stageOf.lookup(capture.getDefiningOp()));
      element.regions.push_back(owner.take(body));
      element.captures.push_back(std::move(indices));
    };
    for (MapOp map : fusionChain(consumerSegment))
      append(map.getBody(), map.getCaptures());
    append(consumer->getRegion(0), consumerCaptures);
    return element;
  };
  for (ReduceOp reduction : analysis.reductions) {
    ReductionStage stage;
    stage.kind = reduction.getKind();
    stage.element =
        takeElement(reduction, reduction.getCaptures(), reduction.getSegment());
    program.reductions.push_back(std::move(stage));
  }
  if (!analysis.mapStores.empty()) {
    MapStoreOp mapStore = analysis.mapStores.front();
    program.mapStore =
        takeElement(mapStore, mapStore.getCaptures(), mapStore.getSegment());
  }
}

/// Accept only the M6 identity segmented-sum planning shape.
LogicalResult verifyPlanningProgram(SegmentProgramAnalysis &analysis) {
  if (!analysis.maps.empty())
    return analysis.maps.front().emitError(
        "planning does not support swage.map");
  for (ReduceOp reduction : analysis.reductions)
    if (!reduction.getCaptures().empty())
      return reduction.emitError("planning requires a capture-free reduction");
  if (analysis.reductions.size() != 1)
    return analysis.reductions.back().emitError(
        "planning requires exactly one reduction stage");
  if (!analysis.mapStores.empty())
    return analysis.mapStores.front().emitError(
        "planning requires memref.store of the reduction result");

  ReduceOp reduction = analysis.reductions.front();
  if (reduction.getKind() != ReductionKind::Sum)
    return reduction.emitError("planning requires kind<sum>");
  Block &body = reduction.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  if (!body.without_terminator().empty() ||
      yield.getValue() != body.getArgument(0))
    return reduction.emitError(
        "planning requires an identity reduction region");
  return success();
}

void buildPlanningCompanion(ModuleOp module, func::FuncOp semanticFunction,
                            int32_t warpMaxElements, int32_t ctaChunkElements) {
  OpBuilder builder(module.getContext());
  Location loc = semanticFunction.getLoc();
  Type taskRange = swage_plan::TaskRangeType::get(module.getContext());
  auto functionType =
      builder.getFunctionType({semanticFunction.getArgument(1).getType(),
                               builder.getI32Type(), builder.getI32Type()},
                              taskRange);

  builder.setInsertionPointAfter(semanticFunction);
  auto companion = func::FuncOp::create(
      builder, loc, semanticFunction.getName().str() + "__swage_plan",
      functionType);
  companion.setPrivate();
  Block *entry = companion.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  ArrayAttr policies = builder.getArrayAttr(
      {swage_plan::TaskPolicyAttr::get(module.getContext(),
                                       swage_plan::TaskPolicy::Warp),
       swage_plan::TaskPolicyAttr::get(module.getContext(),
                                       swage_plan::TaskPolicy::CTA)});
  auto tasks = swage_plan::ClassifyOp::create(
      builder, loc, taskRange, entry->getArgument(0), entry->getArgument(1),
      entry->getArgument(2), semanticFunction.getName(),
      static_cast<uint32_t>(warpMaxElements),
      static_cast<uint32_t>(ctaChunkElements), policies);
  func::ReturnOp::create(builder, loc, tasks.getResult());
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
        if (program.terminal == TerminalKind::ScalarStore) {
          memref::StoreOp::create(outer, outerLoc,
                                  results[program.storedReduction],
                                  function.getArgument(2), segmentId);
        } else {
          scf::ForOp::create(
              outer, outerLoc, start, end, one, ValueRange(),
              [&](OpBuilder &inner, Location innerLoc, Value index,
                  ValueRange) {
                Value value = memref::LoadOp::create(
                    inner, innerLoc, function.getArgument(0), index);
                value =
                    evaluateElement(inner, program.mapStore, value, results);
                memref::StoreOp::create(inner, innerLoc, value,
                                        function.getArgument(2), index);
                scf::YieldOp::create(inner, innerLoc);
              });
        }
        scf::YieldOp::create(outer, outerLoc);
      });
  func::ReturnOp::create(builder, loc);
}

void buildGPUProgram(ModuleOp module, func::FuncOp source,
                     const SegmentProgram &program, int64_t blockSize,
                     bool useTaskIds, bool fusedMixed) {
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(builder, loc,
                                            source.getName().str() + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  SmallVector<Type> inputs{pointer, pointer, pointer};
  if (useTaskIds || fusedMixed)
    inputs.push_back(pointer);
  inputs.append({i32, i32});
  if (fusedMixed)
    inputs.push_back(i32);
  auto kernelType = FunctionType::get(module.getContext(), inputs, {});
  auto kernel =
      gpu::GPUFuncOp::create(builder, loc, source.getName(), kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  kernel->setAttr(NVVM::NVVMDialect::getReqntidAttrName(),
                  builder.getDenseI32ArrayAttr(
                      {static_cast<int32_t>(blockSize), 1, 1}));

  Block *entry = &kernel.getBody().front();
  builder.setInsertionPointToStart(entry);
  Value taskIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  auto loadSegmentId = [&](OpBuilder &body, Location bodyLoc, Value taskId) {
    Value taskId64 =
        arith::IndexCastOp::create(body, bodyLoc, body.getI64Type(), taskId);
    Value taskAddress = LLVM::GEPOp::create(body, bodyLoc, pointer, i32,
                                            entry->getArgument(3), taskId64);
    Value segmentIdI32 = LLVM::LoadOp::create(body, bodyLoc, i32, taskAddress);
    return Value(arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(),
                                            segmentIdI32));
  };
  auto emitSegment = [&](OpBuilder &body, Location bodyLoc, Value segmentId,
                         Value logicalThreadId, Value stride,
                         bool useWarpShuffle) {
    Value segmentId64 =
        arith::IndexCastOp::create(body, bodyLoc, body.getI64Type(), segmentId);
    Value startAddress = LLVM::GEPOp::create(
        body, bodyLoc, pointer, i32, entry->getArgument(1), segmentId64);
    Value startI32 = LLVM::LoadOp::create(body, bodyLoc, i32, startAddress);
    Value nextSegment = arith::AddIOp::create(body, bodyLoc, segmentId, one);
    Value nextSegment64 = arith::IndexCastOp::create(
        body, bodyLoc, body.getI64Type(), nextSegment);
    Value endAddress = LLVM::GEPOp::create(
        body, bodyLoc, pointer, i32, entry->getArgument(1), nextSegment64);
    Value endI32 = LLVM::LoadOp::create(body, bodyLoc, i32, endAddress);
    Value start = arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(),
                                             startI32);
    Value end =
        arith::IndexCastOp::create(body, bodyLoc, body.getIndexType(), endI32);
    Value first = arith::AddIOp::create(body, bodyLoc, start, logicalThreadId);

    SmallVector<Value> results;
    for (const ReductionStage &stage : program.reductions) {
      Value identity = identityFor(body, bodyLoc, stage.kind);
      auto local = scf::ForOp::create(
          body, bodyLoc, first, end, stride, ValueRange(identity),
          [&](OpBuilder &loop, Location loopLoc, Value index,
              ValueRange accumulator) {
            Value index64 = arith::IndexCastOp::create(
                loop, loopLoc, loop.getI64Type(), index);
            Value address = LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                                entry->getArgument(0), index64);
            Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
            value = evaluateElement(loop, stage.element, value, results);
            scf::YieldOp::create(
                loop, loopLoc,
                combine(loop, loopLoc, stage.kind, accumulator.front(), value));
          });
      gpu::AllReduceOperation gpuKind = stage.kind == ReductionKind::Sum
                                            ? gpu::AllReduceOperation::ADD
                                            : gpu::AllReduceOperation::MAXIMUMF;
      Value total = local.getResult(0);
      if (useWarpShuffle) {
        for (int32_t offset = 1; offset < 32; offset <<= 1) {
          auto shuffled = gpu::ShuffleOp::create(body, bodyLoc, total, offset,
                                                 32, gpu::ShuffleMode::XOR);
          total = combine(body, bodyLoc, stage.kind, total,
                          shuffled.getShuffleResult());
        }
      } else {
        auto operation =
            gpu::AllReduceOperationAttr::get(module.getContext(), gpuKind);
        // uniform = true, so the result is broadcast to every thread and the
        // lowering's trailing barrier fences this stage from the next.
        total = gpu::AllReduceOp::create(body, bodyLoc, total, operation, true);
      }
      results.push_back(total);
    }

    if (program.terminal == TerminalKind::ScalarStore) {
      Value total = results[program.storedReduction];
      Value firstThread = arith::CmpIOp::create(
          body, bodyLoc, arith::CmpIPredicate::eq, logicalThreadId, zero);
      scf::IfOp::create(
          body, bodyLoc, firstThread, [&](OpBuilder &store, Location storeLoc) {
            Value outputAddress =
                LLVM::GEPOp::create(store, storeLoc, pointer, f32,
                                    entry->getArgument(2), segmentId64);
            LLVM::StoreOp::create(store, storeLoc, total, outputAddress);
            scf::YieldOp::create(store, storeLoc);
          });
      return;
    }

    // Guard-free on purpose: every thread runs the same block-stride loop it
    // ran for each reduction stage, and an empty segment makes it zero-trip.
    // A thread-dependent guard here would put a predicate around code the
    // barriers above already made CTA-uniform.
    scf::ForOp::create(
        body, bodyLoc, first, end, stride, ValueRange(),
        [&](OpBuilder &loop, Location loopLoc, Value index, ValueRange) {
          Value index64 = arith::IndexCastOp::create(loop, loopLoc,
                                                     loop.getI64Type(), index);
          Value address = LLVM::GEPOp::create(loop, loopLoc, pointer, f32,
                                              entry->getArgument(0), index64);
          Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
          value = evaluateElement(loop, program.mapStore, value, results);
          Value outputAddress = LLVM::GEPOp::create(
              loop, loopLoc, pointer, f32, entry->getArgument(2), index64);
          LLVM::StoreOp::create(loop, loopLoc, value, outputAddress);
          scf::YieldOp::create(loop, loopLoc);
        });
  };

  if (fusedMixed) {
    Value three = arith::ConstantIndexOp::create(builder, loc, 3);
    Value four = arith::ConstantIndexOp::create(builder, loc, 4);
    Value warp = arith::ConstantIndexOp::create(builder, loc, 32);
    Value warpTaskCount = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), entry->getArgument(5));
    Value ctaTaskCount = arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), entry->getArgument(6));
    Value roundedWarpTaskCount =
        arith::AddIOp::create(builder, loc, warpTaskCount, three);
    Value warpBlockCount =
        arith::DivUIOp::create(builder, loc, roundedWarpTaskCount, four);
    Value isWarpBlock = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::ult, taskIndex, warpBlockCount);
    scf::IfOp::create(
        builder, loc, isWarpBlock,
        [&](OpBuilder &warpBlock, Location warpLoc) {
          Value physicalWarp =
              arith::DivUIOp::create(warpBlock, warpLoc, threadId, warp);
          Value lane =
              arith::RemUIOp::create(warpBlock, warpLoc, threadId, warp);
          Value firstTask =
              arith::MulIOp::create(warpBlock, warpLoc, taskIndex, four);
          Value warpTaskId = arith::AddIOp::create(warpBlock, warpLoc,
                                                   firstTask, physicalWarp);
          Value inRange = arith::CmpIOp::create(warpBlock, warpLoc,
                                                arith::CmpIPredicate::ult,
                                                warpTaskId, warpTaskCount);
          scf::IfOp::create(
              warpBlock, warpLoc, inRange,
              [&](OpBuilder &task, Location taskLoc) {
                Value segmentId = loadSegmentId(task, taskLoc, warpTaskId);
                emitSegment(task, taskLoc, segmentId, lane, warp, true);
                scf::YieldOp::create(task, taskLoc);
              });
          scf::YieldOp::create(warpBlock, warpLoc);
        },
        [&](OpBuilder &ctaBlock, Location ctaLoc) {
          Value ctaTaskId = arith::SubIOp::create(ctaBlock, ctaLoc, taskIndex,
                                                  warpBlockCount);
          Value inRange =
              arith::CmpIOp::create(ctaBlock, ctaLoc, arith::CmpIPredicate::ult,
                                    ctaTaskId, ctaTaskCount);
          scf::IfOp::create(
              ctaBlock, ctaLoc, inRange,
              [&](OpBuilder &task, Location taskLoc) {
                Value mixedTaskId = arith::AddIOp::create(
                    task, taskLoc, warpTaskCount, ctaTaskId);
                Value segmentId = loadSegmentId(task, taskLoc, mixedTaskId);
                emitSegment(task, taskLoc, segmentId, threadId, block, false);
                scf::YieldOp::create(task, taskLoc);
              });
          scf::YieldOp::create(ctaBlock, ctaLoc);
        });
    gpu::ReturnOp::create(builder, loc);
    source.erase();
    return;
  }

  Value taskCount =
      arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                 entry->getArgument(useTaskIds ? 5 : 4));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        taskIndex, taskCount);

  scf::IfOp::create(builder, loc, inRange,
                    [&](OpBuilder &body, Location bodyLoc) {
                      Value segmentId = taskIndex;
                      if (useTaskIds)
                        segmentId = loadSegmentId(body, bodyLoc, taskIndex);
                      emitSegment(body, bodyLoc, segmentId, threadId, block,
                                  useTaskIds && blockSize == 32);

                      scf::YieldOp::create(body, bodyLoc);
                    });
  gpu::ReturnOp::create(builder, loc);
  source.erase();
}

void buildSplitGPUProgram(ModuleOp module, func::FuncOp source, bool merge) {
  constexpr int64_t blockSize = 128;
  OpBuilder builder(module.getContext());
  Location loc = source.getLoc();
  std::string suffix = merge ? "__merge" : "__partial";
  builder.setInsertionPoint(source);
  auto gpuModule = gpu::GPUModuleOp::create(
      builder, loc, source.getName().str() + suffix + "_module");

  builder.setInsertionPointToStart(gpuModule.getBody());
  Type pointer = LLVM::LLVMPointerType::get(module.getContext());
  Type i32 = builder.getI32Type();
  Type f32 = builder.getF32Type();
  auto kernelType = FunctionType::get(
      module.getContext(), {pointer, pointer, pointer, i32, i32}, {});
  auto kernel = gpu::GPUFuncOp::create(
      builder, loc, source.getName().str() + suffix, kernelType);
  kernel->setAttr(gpu::GPUDialect::getKernelFuncAttrName(),
                  builder.getUnitAttr());
  kernel->setAttr(NVVM::NVVMDialect::getReqntidAttrName(),
                  builder.getDenseI32ArrayAttr(
                      {static_cast<int32_t>(blockSize), 1, 1}));

  Block *entry = &kernel.getBody().front();
  builder.setInsertionPointToStart(entry);
  Value taskIndex = gpu::BlockIdOp::create(builder, loc, gpu::Dimension::x);
  Value threadId = gpu::ThreadIdOp::create(builder, loc, gpu::Dimension::x);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value block = arith::ConstantIndexOp::create(builder, loc, blockSize);
  Value taskCount = arith::IndexCastOp::create(
      builder, loc, builder.getIndexType(), entry->getArgument(4));
  Value inRange = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        taskIndex, taskCount);

  scf::IfOp::create(
      builder, loc, inRange, [&](OpBuilder &body, Location bodyLoc) {
        Value fields =
            arith::ConstantIndexOp::create(body, bodyLoc, merge ? 3 : 2);
        Value recordBase =
            arith::MulIOp::create(body, bodyLoc, taskIndex, fields);
        Value recordPointer = entry->getArgument(merge ? 2 : 1);
        auto loadRecord = [&](int64_t field) {
          Value index = recordBase;
          if (field)
            index = arith::AddIOp::create(
                body, bodyLoc, recordBase,
                arith::ConstantIndexOp::create(body, bodyLoc, field));
          Value index64 = arith::IndexCastOp::create(body, bodyLoc,
                                                     body.getI64Type(), index);
          Value address = LLVM::GEPOp::create(body, bodyLoc, pointer, i32,
                                              recordPointer, index64);
          return Value(LLVM::LoadOp::create(body, bodyLoc, i32, address));
        };

        Value outputIndex = taskIndex;
        int64_t rangeField = 0;
        if (merge) {
          outputIndex = arith::IndexCastOp::create(
              body, bodyLoc, body.getIndexType(), loadRecord(0));
          rangeField = 1;
        }
        Value begin = arith::IndexCastOp::create(
            body, bodyLoc, body.getIndexType(), loadRecord(rangeField));
        Value end = arith::IndexCastOp::create(
            body, bodyLoc, body.getIndexType(), loadRecord(rangeField + 1));
        Value first = arith::AddIOp::create(body, bodyLoc, begin, threadId);
        Value identity = identityFor(body, bodyLoc, ReductionKind::Sum);
        auto local = scf::ForOp::create(
            body, bodyLoc, first, end, block, ValueRange(identity),
            [&](OpBuilder &loop, Location loopLoc, Value index,
                ValueRange accumulator) {
              Value index64 = arith::IndexCastOp::create(
                  loop, loopLoc, loop.getI64Type(), index);
              Value address = LLVM::GEPOp::create(
                  loop, loopLoc, pointer, f32, entry->getArgument(0), index64);
              Value value = LLVM::LoadOp::create(loop, loopLoc, f32, address);
              scf::YieldOp::create(loop, loopLoc,
                                   combine(loop, loopLoc, ReductionKind::Sum,
                                           accumulator.front(), value));
            });
        auto operation = gpu::AllReduceOperationAttr::get(
            module.getContext(), gpu::AllReduceOperation::ADD);
        Value total = gpu::AllReduceOp::create(
            body, bodyLoc, local.getResult(0), operation, true);
        Value firstThread = arith::CmpIOp::create(
            body, bodyLoc, arith::CmpIPredicate::eq, threadId, zero);
        scf::IfOp::create(
            body, bodyLoc, firstThread,
            [&](OpBuilder &store, Location storeLoc) {
              Value outputIndex64 = arith::IndexCastOp::create(
                  store, storeLoc, store.getI64Type(), outputIndex);
              Value outputAddress = LLVM::GEPOp::create(
                  store, storeLoc, pointer, f32,
                  entry->getArgument(merge ? 1 : 2), outputIndex64);
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
    SegmentProgramAnalysis analysis;
    if (failed(analyzeSegmentProgram(*function, analysis)))
      return signalPassFailure();
    // The owner must outlive the program, which points into it.
    RegionOwner owner;
    SegmentProgram program;
    detachSegmentProgram(analysis, owner, program);
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
    useTaskIds = other.useTaskIds.getValue();
    fusedMixed = other.fusedMixed.getValue();
  }
  SegmentedReductionToGPUPass(int64_t requestedBlockSize, bool requestedTaskIds,
                              bool requestedFusedMixed) {
    blockSize = requestedBlockSize;
    useTaskIds = requestedTaskIds;
    fusedMixed = requestedFusedMixed;
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
    if (blockSize <= 0) {
      getOperation().emitError("block-size must be a positive integer");
      return signalPassFailure();
    }
    if (blockSize > 1024) {
      getOperation().emitError("block-size must be at most 1024");
      return signalPassFailure();
    }
    if (fusedMixed && blockSize != 128) {
      getOperation().emitError("fused mixed lowering requires block-size 128");
      return signalPassFailure();
    }
    FailureOr<func::FuncOp> function = findSegmentedReduction(getOperation());
    if (failed(function))
      return signalPassFailure();
    SegmentProgramAnalysis analysis;
    if (failed(analyzeSegmentProgram(*function, analysis)))
      return signalPassFailure();
    if ((useTaskIds || fusedMixed) && failed(verifyPlanningProgram(analysis)))
      return signalPassFailure();
    RegionOwner owner;
    SegmentProgram program;
    detachSegmentProgram(analysis, owner, program);
    buildGPUProgram(getOperation(), *function, program, blockSize, useTaskIds,
                    fusedMixed);
  }

private:
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
    buildSplitGPUProgram(getOperation(), *function, merge);
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
    registry.insert<func::FuncDialect, swage_plan::SwagePlanDialect>();
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

    buildPlanningCompanion(module, function,
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
  return std::make_unique<SegmentedReductionToGPUPass>(blockSize, useTaskIds,
                                                       fusedMixed);
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
