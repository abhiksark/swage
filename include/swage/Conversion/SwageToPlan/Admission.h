// include/swage/Conversion/SwageToPlan/Admission.h
//===- Admission.h - Segment program admission -----------------*- C++ -*-===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//
//
// Read-only admission of segment functions: which functions a lowering
// handles, what each of their arguments is, and whether the program in a
// function has a shape the lowerings implement. Nothing here changes IR, so
// a caller admits every function before it changes any of them. This header
// is internal to the Swage conversions.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_CONVERSION_SWAGETOPLAN_ADMISSION_H
#define SWAGE_CONVERSION_SWAGETOPLAN_ADMISSION_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"

#include <cstdint>
#include <optional>

namespace mlir::swage {

/// Where a segment function takes each of its arguments. The positions come
/// from the `swage.role` argument attributes, so no lowering assumes an
/// argument order.
struct SegmentABI {
  unsigned values = 0;
  unsigned offsets = 0;
  unsigned output = 0;
  unsigned valueCount = 0;
  unsigned segmentCount = 0;
  /// The number of columns, which a function over rank-two values declares
  /// and a function over rank-one values does not.
  std::optional<unsigned> featureCount;
};

/// Read-only admission result shared by every segmented-program consumer.
struct SegmentProgramAnalysis {
  SegmentABI abi;
  /// The segment id of axis 0 and, for rank-two values, the one of axis
  /// 1, the column, in the order of their axes.
  SmallVector<SegmentIdOp> segmentIds;
  SmallVector<MakeSegmentOp> segments;
  SmallVector<MapOp> maps;
  SmallVector<ReduceOp> reductions;
  SmallVector<memref::StoreOp> stores;
  SmallVector<MapStoreOp> mapStores;
  SmallVector<func::ReturnOp> returns;
  /// The `swage.extent` of the program: none, or the one its epilogue reads.
  SmallVector<ExtentOp> extents;
  /// The scalar epilogue between the stored reduction and the store, in
  /// program order: empty, or the `arith.index_cast` of the extent, the
  /// `arith.sitofp` of that count, and the `arith.divf` of the reduction by
  /// it. A program with an epilogue stores a mean.
  SmallVector<Operation *> epilogue;
  /// The reduction whose result the program stores, as it is or divided by
  /// the extent.
  ReduceOp storedReduction;
  /// The value the program stores per segment: the result of
  /// `storedReduction`, or the result of the epilogue.
  Value storedValue;
  /// The element type of the values. Every region, capture, and result of
  /// the program has it.
  Type element;
};

/// The element types the lowerings admit for the values and the output, f32
/// and f64, and the index word types they admit for the offsets and the
/// counts, i32.
bool isAdmittedElementType(Type type);
bool isAdmittedIndexType(Type type);

/// The functions a pass lowers: every function that holds a Swage operation,
/// or the one function that `selected` names. A module without a segment
/// function gives an empty list, and the pass leaves it as it is.
FailureOr<SmallVector<func::FuncOp>> findSegmentFunctions(ModuleOp module,
                                                          StringRef selected);

/// A GPU lowering replaces a segment function by a `gpu.module` named after
/// its kernel, so nothing may refer to the function, and the names it
/// creates must be free. Checked before any function is changed.
/// `kernelSuffix` is what the kernel name adds to the function name.
LogicalResult verifyKernelSymbols(ModuleOp module, func::FuncOp function,
                                  StringRef kernelSuffix);

/// Analyze one canonical segment program without mutating it.
LogicalResult analyzeSegmentProgram(func::FuncOp function,
                                    SegmentProgramAnalysis &analysis);

/// Check the element programs of the maps, reductions, and stores listed in
/// `analysis`, against `analysis.element`: captures are results of
/// reductions of that type, and regions hold admitted operations of that
/// type only. `math.exp2` is admitted for f32 alone.
LogicalResult verifyConsumerPrograms(SegmentProgramAnalysis &analysis);

/// Admit a single reduction whose element program needs no other stage,
/// over rank-one values: host classification turns segments of scalars into
/// tasks, and a rank-two function has one kernel and no task buffer.
LogicalResult verifyPlanningProgram(SegmentProgramAnalysis &analysis);

/// The relative work of the element programs under `root`: one unit per
/// add, subtract, multiply, minimum, and maximum in a `swage.map` or
/// `swage.reduce` region, eight per `math.exp2`, and sixteen per division.
/// Constants and yields are free. Null when a region holds an operation
/// that has no weight.
///
/// The weights were calibrated on held-out GPU benchmarks. They are
/// scheduling hints for the host, which compares the sum with a budget, and
/// not instruction latency estimates.
std::optional<int64_t> estimateElementWork(Operation *root);

/// Fuse every map of an admitted function into its consumer. This is the
/// first change a lowering makes, so it runs only after every function has
/// been admitted. Admission gives every map one consumer, so no map is left
/// and every consumer reads the segment of `make_segment`; `analysis.maps`
/// is emptied.
void fuseAdmittedMaps(SegmentProgramAnalysis &analysis);

/// Persistent partials and merges still implement only the identity sum of
/// f32 values, stored as it is.
LogicalResult verifyPersistentProgram(SegmentProgramAnalysis &analysis);

} // namespace mlir::swage

#endif // SWAGE_CONVERSION_SWAGETOPLAN_ADMISSION_H
