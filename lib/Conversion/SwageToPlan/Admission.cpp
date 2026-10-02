// lib/Conversion/SwageToPlan/Admission.cpp
//===- Admission.cpp - Segment program admission --------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "swage/Conversion/SwageToPlan/Admission.h"

#include <optional>
#include <string>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "swage/Dialect/Swage/IR/SwageDialect.h"
#include "swage/Dialect/Swage/Transforms/FuseMaps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::swage {

bool isAdmittedElementType(Type type) { return type.isF32() || type.isF64(); }
bool isAdmittedIndexType(Type type) { return type.isSignlessInteger(32); }

namespace {

/// The spelling of a type inside a sentence of a diagnostic, without the
/// quotes a diagnostic puts around a type it is given.
std::string spelled(Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << type;
  return text;
}

bool isRankOneMemRef(Type type, Type elementType) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && memref.getRank() == 1 && memref.isDynamicDim(0) &&
         memref.getElementType() == elementType &&
         memref.getLayout().isIdentity() && !memref.getMemorySpace();
}

/// A set of operation classes: the membership test and the names a
/// diagnostic lists come from the same type pack, so they cannot drift.
template <typename... Ops> struct OperationSet {
  static bool contains(Operation &operation) { return isa<Ops...>(&operation); }

  /// The operation names as prose: "a, b, and c".
  static std::string describe() {
    const StringRef names[] = {Ops::getOperationName()...};
    std::string text;
    for (auto [index, name] : llvm::enumerate(names)) {
      if (index)
        text += index + 1 == std::size(names) ? ", and " : ", ";
      text += name;
    }
    return text;
  }
};

/// Operations a Swage region may contain.
///
/// The list is a whitelist rather than a purity test because both backends
/// must lower every admitted operation. Exponentials are written as
/// `math.exp2` of a scaled operand; `math.exp` is deliberately absent
/// because `--convert-gpu-to-nvvm` turns every `math` operation into a
/// libdevice call, and the PTX path links no libdevice.
using RegionOperations =
    OperationSet<arith::ConstantOp, arith::AddFOp, arith::SubFOp, arith::MulFOp,
                 arith::DivFOp, arith::MaximumFOp, arith::MinimumFOp,
                 math::Exp2Op>;

/// Verify a Swage region of a program over `element` values: an element
/// argument followed by one argument of the same type per capture, admitted
/// operations only, and a yield of that type. A program has one element
/// type, the one of its values, so a region of another type is refused.
LogicalResult verifyRegion(Operation *owner, Region &region,
                           unsigned captureCount, Type element) {
  // Malformed IR that bypassed the dialect verifier must fail here rather
  // than reach the unchecked dereferences below.
  if (!region.hasOneBlock())
    return owner->emitError("segment region requires exactly one block");
  Block &body = region.front();
  if (!body.mightHaveTerminator())
    return owner->emitError()
           << "segment region must yield an " << spelled(element) << " value";
  if (body.getNumArguments() != 1 + captureCount ||
      llvm::any_of(body.getArgumentTypes(),
                   [&](Type type) { return type != element; }))
    return owner->emitError()
           << "segment region requires an " << spelled(element)
           << " element argument followed by " << spelled(element)
           << " captures";
  for (Operation &operation : body.without_terminator()) {
    if (!RegionOperations::contains(operation))
      return operation.emitError()
             << "operation '" << operation.getName()
             << "' is unsupported inside a segment region; a region accepts "
             << RegionOperations::describe();
    for (Type type : operation.getResultTypes())
      if (type != element)
        return operation.emitError()
               << "operation '" << operation.getName()
               << "' is unsupported inside a segment region; every result "
                  "must be "
               << spelled(element) << ", got " << type;
    // The device has an approximate exp2 for f32 and none for f64, and the
    // backend aborts on one it cannot select. The oracle follows the same
    // rule, so it never runs a program that no kernel can.
    if (isa<math::Exp2Op>(operation) && !element.isF32())
      return operation.emitError()
             << "operation 'math.exp2' is admitted for f32 values only: the "
                "device has no "
             << spelled(element) << " exp2";
  }
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getValue().getType() != element)
    return owner->emitError()
           << "segment region must yield an " << spelled(element) << " value";
  return success();
}

/// Whether `function` holds an operation of the Swage dialect.
bool holdsSwageOperation(func::FuncOp function) {
  bool found = false;
  function.walk([&](Operation *operation) {
    found |= isa_and_nonnull<SwageDialect>(operation->getDialect());
  });
  return found;
}

} // namespace

/// The functions a pass lowers: every function that holds a Swage operation,
/// or the one function that `selected` names. A module without a segment
/// function gives an empty list, and the pass leaves it as it is.
FailureOr<SmallVector<func::FuncOp>> findSegmentFunctions(ModuleOp module,
                                                          StringRef selected) {
  SmallVector<func::FuncOp> functions;
  if (selected.empty()) {
    for (func::FuncOp function : module.getOps<func::FuncOp>())
      if (holdsSwageOperation(function))
        functions.push_back(function);
    return functions;
  }
  auto function = module.lookupSymbol<func::FuncOp>(selected);
  if (!function) {
    module.emitError() << "function names @" << selected
                       << ", which is not a function of the module";
    return failure();
  }
  if (!holdsSwageOperation(function)) {
    function.emitError() << "function names @" << selected
                         << ", which holds no Swage segment operation";
    return failure();
  }
  functions.push_back(function);
  return functions;
}

/// A GPU lowering replaces a segment function by a `gpu.module` named after
/// its kernel, so nothing may refer to the function, and the names it
/// creates must be free. Checked before any function is changed.
LogicalResult verifyKernelSymbols(ModuleOp module, func::FuncOp function,
                                  StringRef kernelSuffix) {
  std::optional<SymbolTable::UseRange> uses = SymbolTable::getSymbolUses(
      function.getOperation(), module.getOperation());
  if (!uses)
    return function.emitError()
           << "cannot tell whether @" << function.getName()
           << " is referenced; lowering it to a GPU kernel removes it, so the "
              "module must hold only operations with known symbol uses";
  if (!uses->empty()) {
    InFlightDiagnostic diagnostic =
        function.emitError()
        << "segment function @" << function.getName() << " is referenced "
        << llvm::size(*uses)
        << " times; lowering it to a GPU kernel removes it, so it must have no "
           "symbol use";
    diagnostic.attachNote(uses->begin()->getUser()->getLoc())
        << "referenced here";
    return diagnostic;
  }
  std::string kernel = (function.getName() + kernelSuffix).str();
  SmallVector<std::string, 2> created = {kernel + "_module"};
  if (!kernelSuffix.empty())
    created.push_back(kernel);
  for (const std::string &name : created) {
    Operation *existing =
        SymbolTable::lookupSymbolIn(module.getOperation(), name);
    if (!existing)
      continue;
    InFlightDiagnostic diagnostic = function.emitError()
                                    << "lowering @" << function.getName()
                                    << " creates @" << name
                                    << ", which the module already defines";
    diagnostic.attachNote(existing->getLoc()) << "defined here";
    return diagnostic;
  }
  return success();
}

namespace {

/// Whether `type` is a rank-one buffer the lowerings can address: a dynamic
/// size, the identity layout, and the default memory space.
bool isSegmentBuffer(Type type) {
  auto memref = dyn_cast<MemRefType>(type);
  return memref && isRankOneMemRef(type, memref.getElementType());
}

/// Find the arguments of a segment function through their roles and check
/// their types. Every argument declares a role, each of the five roles is
/// declared once, and there is no positional default.
LogicalResult readSegmentABI(func::FuncOp function, SegmentABI &abi) {
  constexpr const char *shape =
      " with a dynamic size, the identity layout, and the default memory "
      "space, got ";
  FunctionType type = function.getFunctionType();
  StringRef attribute = SwageDialect::getRoleAttrName();
  std::optional<unsigned> declared[5];
  for (unsigned index = 0; index < type.getNumInputs(); ++index) {
    auto role = dyn_cast_or_null<ArgumentRoleAttr>(
        function.getArgAttr(index, attribute));
    if (!role)
      return function.emitError()
             << "segment function argument #" << index << " declares no "
             << attribute
             << "; every argument declares one of values, offsets, output, "
                "value_count, and segment_count";
    std::optional<unsigned> &slot =
        declared[static_cast<unsigned>(role.getValue())];
    // The dialect verifier rejects a repeated role; this guards IR that
    // bypassed it.
    if (slot)
      return function.emitError()
             << attribute << "<" << stringifyArgumentRole(role.getValue())
             << "> is declared by argument #" << *slot
             << " and again by argument #" << index;
    slot = index;
  }
  for (ArgumentRole role :
       {ArgumentRole::Values, ArgumentRole::Offsets, ArgumentRole::Output,
        ArgumentRole::ValueCount, ArgumentRole::SegmentCount})
    if (!declared[static_cast<unsigned>(role)])
      return function.emitError()
             << "segment function declares no " << attribute << "<"
             << stringifyArgumentRole(role)
             << ">; it declares values, offsets, output, value_count, and "
                "segment_count once each";
  abi.values = *declared[static_cast<unsigned>(ArgumentRole::Values)];
  abi.offsets = *declared[static_cast<unsigned>(ArgumentRole::Offsets)];
  abi.output = *declared[static_cast<unsigned>(ArgumentRole::Output)];
  abi.valueCount = *declared[static_cast<unsigned>(ArgumentRole::ValueCount)];
  abi.segmentCount =
      *declared[static_cast<unsigned>(ArgumentRole::SegmentCount)];

  // The element type comes from the values, and the index word type from
  // the offsets. The output and the counts follow them.
  Type valuesType = type.getInput(abi.values);
  if (!isSegmentBuffer(valuesType) ||
      !isAdmittedElementType(cast<MemRefType>(valuesType).getElementType()))
    return function.emitError()
           << attribute << "<values> requires a rank-one f32 or f64 memref"
           << shape << valuesType;
  Type offsetsType = type.getInput(abi.offsets);
  if (!isSegmentBuffer(offsetsType) ||
      !isAdmittedIndexType(cast<MemRefType>(offsetsType).getElementType()))
    return function.emitError()
           << attribute << "<offsets> requires a rank-one i32 memref" << shape
           << offsetsType;
  Type element = cast<MemRefType>(valuesType).getElementType();
  Type word = cast<MemRefType>(offsetsType).getElementType();
  Type outputType = type.getInput(abi.output);
  if (!isRankOneMemRef(outputType, element))
    return function.emitError()
           << attribute << "<output> requires a rank-one memref of " << element
           << ", the element type of the values," << shape << outputType;
  for (auto [role, index] : {std::pair("value_count", abi.valueCount),
                             std::pair("segment_count", abi.segmentCount)})
    if (type.getInput(index) != word)
      return function.emitError()
             << attribute << "<" << role << "> requires " << word
             << ", the element type of the offsets, got "
             << type.getInput(index);
  if (type.getNumResults() != 0)
    return function.emitError()
           << "segment function must have no result, got " << type;
  return success();
}

LogicalResult verifySegmentedFunctionShape(func::FuncOp function,
                                           SegmentABI &abi) {
  if (failed(readSegmentABI(function, abi)))
    return failure();
  if (!function.getBody().hasOneBlock())
    return function.emitError()
           << "segmented reduction requires one block, got "
           << function.getBody().getBlocks().size() << " blocks";
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
    else if (auto extent = dyn_cast<ExtentOp>(operation))
      analysis.extents.push_back(extent);
    else if (isa<arith::IndexCastOp, arith::SIToFPOp, arith::DivFOp>(operation))
      analysis.epilogue.push_back(&operation);
    else
      return operation.emitError()
             << "operation '" << operation.getName()
             << "' is unsupported by segmented reduction lowering";
  }
  return success();
}

LogicalResult verifySegmentRoot(func::FuncOp function,
                                SegmentProgramAnalysis &analysis) {
  if (analysis.segmentIds.size() != 1 || analysis.segments.size() != 1 ||
      analysis.reductions.empty() || analysis.returns.size() != 1)
    return function.emitError()
           << "segmented reduction requires one segment_id, one make_segment, "
              "at least one reduce, and one return, found "
           << analysis.segmentIds.size() << " segment_id, "
           << analysis.segments.size() << " make_segment, "
           << analysis.reductions.size() << " reduce, and "
           << analysis.returns.size() << " return";
  SegmentIdOp segmentId = analysis.segmentIds.front();
  MakeSegmentOp segment = analysis.segments.front();
  if (segmentId.getAxis() != 0)
    return segmentId.emitError()
           << "only swage.segment_id axis 0 is supported, got axis "
           << segmentId.getAxis();
  if (segment.getValues() != function.getArgument(analysis.abi.values) ||
      segment.getOffsets() != function.getArgument(analysis.abi.offsets) ||
      segment.getSegmentId() != segmentId.getResult())
    return segment.emitError(
        "make_segment must bind the function values and offsets at segment_id");
  if (analysis.stores.size() + analysis.mapStores.size() != 1)
    return function.emitError()
           << "segmented reduction requires exactly one output terminal: a "
              "memref.store of a reduction at output[segment_id] or a "
              "swage.map_store into the output, found "
           << analysis.stores.size() << " memref.store and "
           << analysis.mapStores.size() << " swage.map_store";
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

LogicalResult verifyOperationCaptures(Operation *operation, ValueRange captures,
                                      Type element) {
  for (Value capture : captures)
    if (!capture.getDefiningOp<ReduceOp>() || capture.getType() != element)
      return operation->emitError()
             << "segment captures must be " << spelled(element)
             << " results of a swage.reduce in the same function";
  return success();
}

LogicalResult verifySegmentCaptures(SegmentProgramAnalysis &analysis) {
  Type element = analysis.element;
  for (MapOp map : analysis.maps)
    if (failed(verifyOperationCaptures(map, map.getCaptures(), element)))
      return failure();
  for (ReduceOp reduction : analysis.reductions)
    if (failed(verifyOperationCaptures(reduction, reduction.getCaptures(),
                                       element)))
      return failure();
  for (MapStoreOp mapStore : analysis.mapStores)
    if (failed(
            verifyOperationCaptures(mapStore, mapStore.getCaptures(), element)))
      return failure();
  return success();
}

LogicalResult verifySegmentRegions(SegmentProgramAnalysis &analysis) {
  Type element = analysis.element;
  for (MapOp map : analysis.maps)
    if (failed(verifyRegion(map, map.getBody(), map.getCaptures().size(),
                            element)))
      return failure();
  for (ReduceOp reduction : analysis.reductions)
    if (failed(verifyRegion(reduction, reduction.getBody(),
                            reduction.getCaptures().size(), element)))
      return failure();
  for (MapStoreOp mapStore : analysis.mapStores)
    if (failed(verifyRegion(mapStore, mapStore.getBody(),
                            mapStore.getCaptures().size(), element)))
      return failure();
  return success();
}

/// Admit the scalar epilogue of a mean, or none: the extent of the segment,
/// cast to the count type and then to the element type, divides one
/// reduction result. The division runs once per segment, after the
/// reduction, so a lowering that splits a segment sums the partial results
/// and divides once.
///
/// The shape is fixed instead of open to any scalar arithmetic, because the
/// merge of a split segment has to rebuild it from partial results.
LogicalResult verifyEpilogue(func::FuncOp function,
                             SegmentProgramAnalysis &analysis) {
  if (analysis.extents.empty() && analysis.epilogue.empty())
    return success();
  Type word = function.getArgument(analysis.abi.valueCount).getType();
  Type element = analysis.element;
  auto count = analysis.epilogue.size() == 3
                   ? dyn_cast<arith::IndexCastOp>(analysis.epilogue[0])
                   : arith::IndexCastOp();
  auto divisor = count ? dyn_cast<arith::SIToFPOp>(analysis.epilogue[1])
                       : arith::SIToFPOp();
  auto division =
      divisor ? dyn_cast<arith::DivFOp>(analysis.epilogue[2]) : arith::DivFOp();
  if (analysis.extents.size() != 1 || !division)
    return function.emitError()
           << "a scalar epilogue divides one reduction by the extent of its "
              "segment: one swage.extent, then arith.index_cast, "
              "arith.sitofp, and arith.divf, in that order, found "
           << analysis.extents.size() << " swage.extent and "
           << analysis.epilogue.size()
           << " arith.index_cast, arith.sitofp, or arith.divf operations";
  // The one segment of the function is the only segment an extent can
  // read: a map has exactly one consumer, which consumes its elements.
  ExtentOp extent = analysis.extents.front();
  if (count.getIn() != extent.getResult() || count.getType() != word)
    return count.emitError()
           << "the arith.index_cast of a scalar epilogue casts the extent to "
           << word << ", the type of the counts, got "
           << count.getIn().getType() << " to " << count.getType();
  if (divisor.getIn() != count.getOut() || divisor.getType() != element)
    return divisor.emitError()
           << "the arith.sitofp of a scalar epilogue converts the extent "
              "count to "
           << spelled(element) << ", the element type, got "
           << divisor.getIn().getType() << " to " << divisor.getType();
  if (!division.getLhs().getDefiningOp<ReduceOp>() ||
      division.getRhs() != divisor.getOut())
    return division.emitError(
        "the arith.divf of a scalar epilogue divides the result of a "
        "swage.reduce by the converted extent");
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
  if (analysis.mapStores.empty()) {
    memref::StoreOp store = analysis.stores.front();
    analysis.storedValue = store.getValue();
    Value reduced = analysis.storedValue;
    // A program with an epilogue stores its last result, which divides a
    // reduction.
    if (!analysis.epilogue.empty()) {
      auto division = cast<arith::DivFOp>(analysis.epilogue.back());
      if (reduced != division.getResult())
        return store.emitError("a segment function with a scalar epilogue "
                               "stores the result of its arith.divf at "
                               "output[segment_id]");
      reduced = division.getLhs();
    }
    analysis.storedReduction = reduced.getDefiningOp<ReduceOp>();
    if (!analysis.storedReduction ||
        !stageOf.contains(analysis.storedReduction.getOperation()) ||
        store.getMemRef() != function.getArgument(analysis.abi.output) ||
        store.getIndices().size() != 1 ||
        store.getIndices().front() != segmentId.getResult())
      return store.emitError(
          "segmented reduction result must be stored at output[segment_id]");
  } else if (analysis.mapStores.front().getOutput() !=
             function.getArgument(analysis.abi.output)) {
    return analysis.mapStores.front().emitError(
        "swage.map_store must write the function output buffer");
  } else if (!analysis.epilogue.empty()) {
    return analysis.epilogue.back()->emitError(
        "a scalar epilogue needs a memref.store at output[segment_id]; a "
        "swage.map_store has no scalar to divide");
  }
  return success();
}

} // namespace

LogicalResult verifyConsumerPrograms(SegmentProgramAnalysis &analysis) {
  // Every kind of `swage.reduce` has a lowering, so the kind is not
  // checked here. The kind functions of the emission are switches over
  // every kind without a default.
  if (failed(verifySegmentCaptures(analysis)) ||
      failed(verifySegmentRegions(analysis)))
    return failure();
  return success();
}

/// Analyze one canonical segment program without mutating it.
LogicalResult analyzeSegmentProgram(func::FuncOp function,
                                    SegmentProgramAnalysis &analysis) {
  if (failed(verifySegmentedFunctionShape(function, analysis.abi)))
    return failure();
  analysis.element =
      cast<MemRefType>(function.getArgument(analysis.abi.values).getType())
          .getElementType();
  if (failed(collectSegmentOperations(function, analysis)) ||
      failed(verifySegmentRoot(function, analysis)) ||
      failed(verifyMapConsumers(analysis)) ||
      failed(verifyConsumerPrograms(analysis)) ||
      failed(verifyEpilogue(function, analysis)))
    return failure();

  DenseMap<Operation *, unsigned> stageOf = indexReductionStages(analysis);
  if (failed(verifySegmentTerminal(function, analysis, stageOf)))
    return failure();
  if (!analysis.returns.front().getOperands().empty())
    return analysis.returns.front().emitError(
        "segmented reduction must return void");
  return success();
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
  if (!analysis.epilogue.empty())
    return analysis.epilogue.back()->emitError(
        "persistent execution stores the reduction result as it is and takes "
        "no scalar epilogue");
  if (!analysis.element.isF32())
    return reduction.emitError()
           << "persistent execution requires f32 values, got "
           << spelled(analysis.element);
  Block &body = reduction.getBody().front();
  auto yield = cast<YieldOp>(body.getTerminator());
  if (!body.without_terminator().empty() ||
      yield.getValue() != body.getArgument(0))
    return reduction.emitError(
        "persistent execution requires an identity reduction region");
  return success();
}

std::optional<int64_t> estimateElementWork(Operation *root) {
  int64_t work = 0;
  bool weighted = true;
  root->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (!isa<MapOp, ReduceOp>(operation))
      return WalkResult::advance();
    for (Operation &instruction : operation->getRegion(0).front()) {
      if (isa<arith::ConstantOp, YieldOp>(instruction))
        continue;
      if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::MaximumFOp,
              arith::MinimumFOp>(instruction)) {
        work += 1;
      } else if (isa<math::Exp2Op>(instruction)) {
        work += 8;
      } else if (isa<arith::DivFOp>(instruction)) {
        work += 16;
      } else {
        weighted = false;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::skip();
  });
  if (!weighted)
    return std::nullopt;
  return work;
}

void fuseAdmittedMaps(SegmentProgramAnalysis &analysis) {
  // The fusion function is applied to the admitted consumers directly. The
  // greedy pattern driver would also delete dead operations, and a program
  // may hold a reduction that nothing reads, which is lowered as a stage.
  IRRewriter rewriter(analysis.reductions.front()->getContext());
  auto fuse = [&](Operation *consumer) {
    while (succeeded(fuseMapIntoConsumer(consumer, rewriter))) {
    }
  };
  for (ReduceOp reduction : analysis.reductions)
    fuse(reduction);
  for (MapStoreOp mapStore : analysis.mapStores)
    fuse(mapStore);
  analysis.maps.clear();
}

} // namespace mlir::swage
