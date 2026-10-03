// lib/Conversion/SegmentedReduction/SegmentProgram.cpp
//===- SegmentProgram.cpp - Segmented program admission
//------------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "SegmentProgram.h"

#include <algorithm>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;

namespace mlir::swage::detail {
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
} // namespace

Region *SegmentProgram::takeRegion(Region &source) {
  ownedRegions.push_back(std::make_unique<Region>());
  ownedRegions.back()->takeBody(source);
  return ownedRegions.back().get();
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
namespace {

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

LogicalResult verifySegmentedFunctionShape(func::FuncOp function) {
  FunctionType type = function.getFunctionType();
  Builder builder(function.getContext());
  if (type.getNumInputs() != 3 || type.getNumResults() != 0)
    return function.emitError(
        "segmented reduction requires exactly three buffer arguments and "
        "returns void");

  unsigned f32Buffers = 0;
  unsigned i32Buffers = 0;
  for (Type input : type.getInputs()) {
    if (isRankOneMemRef(input, builder.getF32Type()))
      ++f32Buffers;
    else if (isRankOneMemRef(input, builder.getI32Type()))
      ++i32Buffers;
    else
      return function.emitError(
          "segmented reduction arguments must be dynamic rank-one identity "
          "memrefs in the default memory space");
  }
  if (f32Buffers != 2 || i32Buffers != 1)
    return function.emitError(
        "segmented reduction requires two rank-one f32 buffers and one "
        "rank-one i32 buffer");
  if (!function.getBody().hasOneBlock())
    return function.emitError("segmented reduction requires one block");
  return success();
}

LogicalResult collectSegmentOperations(func::FuncOp function,
                                       SegmentProgramAnalysis &analysis) {
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
  return success();
}

LogicalResult verifySegmentRoot(func::FuncOp function,
                                SegmentProgramAnalysis &analysis) {
  if (analysis.segmentIds.size() != 1 || analysis.segments.size() != 1 ||
      analysis.reductions.empty() || analysis.returns.size() != 1)
    return function.emitError(
        "segmented reduction requires one segment_id, one make_segment, at "
        "least one reduce, and one return");
  SegmentIdOp segmentId = analysis.segmentIds.front();
  MakeSegmentOp segment = analysis.segments.front();
  if (segmentId.getAxis() != 0)
    return segmentId.emitError("only swage.segment_id axis 0 is supported");
  if (segment.getSegmentId() != segmentId.getResult())
    return segment.emitError("make_segment must use the function's segment_id");

  Block &entry = function.getBody().front();
  Builder builder(function.getContext());
  auto values = dyn_cast<BlockArgument>(segment.getValues());
  auto offsets = dyn_cast<BlockArgument>(segment.getOffsets());
  if (!values || values.getOwner() != &entry ||
      !isRankOneMemRef(values.getType(), builder.getF32Type()) || !offsets ||
      offsets.getOwner() != &entry ||
      !isRankOneMemRef(offsets.getType(), builder.getI32Type()))
    return segment.emitError(
        "make_segment values and offsets must be the function's rank-one "
        "f32 and i32 input buffers");
  analysis.roles.values = values;
  analysis.roles.offsets = offsets;
  analysis.roles.valuesSourceIndex = values.getArgNumber();
  analysis.roles.offsetsSourceIndex = offsets.getArgNumber();

  if (analysis.stores.size() + analysis.mapStores.size() != 1)
    return function.emitError(
        "segmented reduction requires exactly one output terminal: a "
        "memref.store of a reduction at output[segment_id] or a "
        "swage.map_store into the output");
  return success();
}

LogicalResult verifyMapConsumers(SegmentProgramAnalysis &analysis) {
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
  return success();
}

LogicalResult verifyOperationCaptures(Operation *operation,
                                      ValueRange captures) {
  for (Value capture : captures)
    if (!capture.getDefiningOp<ReduceOp>() || !capture.getType().isF32())
      return operation->emitError(
          "segment captures must be f32 results of a swage.reduce in the "
          "same function");
  return success();
}

LogicalResult verifySegmentCaptures(SegmentProgramAnalysis &analysis) {
  for (MapOp map : analysis.maps)
    if (failed(verifyOperationCaptures(map, map.getCaptures())))
      return failure();
  for (ReduceOp reduction : analysis.reductions)
    if (failed(verifyOperationCaptures(reduction, reduction.getCaptures())))
      return failure();
  for (MapStoreOp mapStore : analysis.mapStores)
    if (failed(verifyOperationCaptures(mapStore, mapStore.getCaptures())))
      return failure();
  return success();
}

LogicalResult verifyReductionKinds(SegmentProgramAnalysis &analysis) {
  for (ReduceOp reduction : analysis.reductions) {
    ReductionKind kind = reduction.getKind();
    if (kind != ReductionKind::Sum && kind != ReductionKind::Max)
      return reduction.emitError(
          "segmented reduction supports only kind<sum> and kind<max>");
  }
  return success();
}

LogicalResult verifySegmentRegions(SegmentProgramAnalysis &analysis) {
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
  return success();
}

DenseMap<Operation *, unsigned>
indexReductionStages(SegmentProgramAnalysis &analysis) {
  DenseMap<Operation *, unsigned> stageOf;
  for (auto [index, reduction] : llvm::enumerate(analysis.reductions))
    stageOf[reduction.getOperation()] = index;
  return stageOf;
}

LogicalResult
verifySegmentTerminal(func::FuncOp function, SegmentProgramAnalysis &analysis,
                      const DenseMap<Operation *, unsigned> &stageOf) {
  SegmentIdOp segmentId = analysis.segmentIds.front();
  Value output;
  Operation *terminal;
  if (analysis.mapStores.empty()) {
    memref::StoreOp store = analysis.stores.front();
    analysis.storedReduction = store.getValue().getDefiningOp<ReduceOp>();
    if (!analysis.storedReduction ||
        !stageOf.contains(analysis.storedReduction.getOperation()) ||
        store.getIndices().size() != 1 ||
        store.getIndices().front() != segmentId.getResult())
      return store.emitError(
          "segmented reduction result must be stored at output[segment_id]");
    output = store.getMemRef();
    terminal = store.getOperation();
  } else {
    MapStoreOp mapStore = analysis.mapStores.front();
    output = mapStore.getOutput();
    terminal = mapStore.getOperation();
  }

  Block &entry = function.getBody().front();
  Builder builder(function.getContext());
  auto outputArgument = dyn_cast<BlockArgument>(output);
  if (!outputArgument || outputArgument.getOwner() != &entry ||
      !isRankOneMemRef(outputArgument.getType(), builder.getF32Type()))
    return terminal->emitError(
        "segmented reduction terminal must write a function rank-one f32 "
        "buffer");
  if (outputArgument == analysis.roles.values ||
      outputArgument == analysis.roles.offsets)
    return terminal->emitError(
        "segmented reduction values, offsets, and output roles must resolve "
        "to three distinct function arguments");

  analysis.roles.output = outputArgument;
  analysis.roles.outputSourceIndex = outputArgument.getArgNumber();
  return success();
}

} // namespace
/// Analyze one canonical segment program without mutating it.
LogicalResult analyzeSegmentProgram(func::FuncOp function,
                                    SegmentProgramAnalysis &analysis) {
  if (failed(verifySegmentedFunctionShape(function)) ||
      failed(collectSegmentOperations(function, analysis)) ||
      failed(verifySegmentRoot(function, analysis)) ||
      failed(verifyMapConsumers(analysis)) ||
      failed(verifySegmentCaptures(analysis)) ||
      failed(verifyReductionKinds(analysis)) ||
      failed(verifySegmentRegions(analysis)))
    return failure();

  DenseMap<Operation *, unsigned> stageOf = indexReductionStages(analysis);
  if (failed(verifySegmentTerminal(function, analysis, stageOf)))
    return failure();
  if (!analysis.returns.front().getOperands().empty())
    return analysis.returns.front().emitError(
        "segmented reduction must return void");
  return success();
}

/// Detach regions only after read-only admission has succeeded.
void detachSegmentProgram(SegmentProgramAnalysis &analysis,
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
      element.regions.push_back(program.takeRegion(body));
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

/// Admit a single reduction whose element program needs no other stage.
LogicalResult verifyPlanningProgram(SegmentProgramAnalysis &analysis) {
  for (MapOp map : analysis.maps)
    if (!map.getCaptures().empty())
      return map.emitError("planning requires capture-free maps");
  for (ReduceOp reduction : analysis.reductions)
    if (!reduction.getCaptures().empty())
      return reduction.emitError("planning requires a capture-free reduction");
  if (analysis.reductions.size() != 1)
    return analysis.reductions.back().emitError(
        "planning requires exactly one reduction stage");
  if (!analysis.mapStores.empty())
    return analysis.mapStores.front().emitError(
        "planning requires memref.store of the reduction result");
  return success();
}

/// Persistent partials and merges still implement only identity sum.
LogicalResult verifyPersistentProgram(SegmentProgramAnalysis &analysis) {
  if (failed(verifyPlanningProgram(analysis)))
    return failure();
  if (!analysis.maps.empty())
    return analysis.maps.front().emitError(
        "persistent execution does not support swage.map");

  ReduceOp reduction = analysis.reductions.front();
  if (reduction.getKind() != ReductionKind::Sum)
    return reduction.emitError("persistent execution requires kind<sum>");
  Block &body = reduction.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  if (!body.without_terminator().empty() ||
      yield.getValue() != body.getArgument(0))
    return reduction.emitError(
        "persistent execution requires an identity reduction region");
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

} // namespace mlir::swage::detail
