// lib/Conversion/SegmentedReduction/SegmentProgram.h
//===- SegmentProgram.h - Internal segmented program
//-------------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#ifndef SWAGE_LIB_CONVERSION_SEGMENTEDREDUCTION_SEGMENTPROGRAM_H
#define SWAGE_LIB_CONVERSION_SEGMENTEDREDUCTION_SEGMENTPROGRAM_H

#include <cstdint>
#include <memory>
#include <vector>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "swage/Dialect/Swage/IR/SwageOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::swage::detail {

enum class SegmentedExecutionKind { Direct, TaskIds, FusedMixed, Persistent };

/// A per-element expression: the fused `swage.map` bodies in application
/// order, followed by the consumer's own body. `captures[i]` lists, for
/// `regions[i]`, the reduction stages whose results bind to that region's
/// capture arguments, in order.
struct ElementProgram {
  llvm::SmallVector<Region *> regions;
  llvm::SmallVector<llvm::SmallVector<unsigned>> captures;
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

/// An admitted segment program. It owns every region detached from the source
/// function, and every capture is a stage index, so an emitter may erase the
/// source IR before emitting. The allocated Region pointees remain stable when
/// this move-only program is moved.
struct SegmentProgram {
  std::vector<std::unique_ptr<Region>> ownedRegions;
  llvm::SmallVector<ReductionStage> reductions;
  TerminalKind terminal = TerminalKind::ScalarStore;
  unsigned storedReduction = 0; ///< ScalarStore: index into `reductions`.
  ElementProgram mapStore;      ///< MapStore: the per-element expression.

  Region *takeRegion(Region &source);
};

/// Resolved semantic entry-buffer roles. These are discovered from the
/// segment source and terminal write before any lowering mutates the function.
struct SemanticBufferRoles {
  BlockArgument values;
  BlockArgument offsets;
  BlockArgument output;
  unsigned valuesSourceIndex = 0;
  unsigned offsetsSourceIndex = 0;
  unsigned outputSourceIndex = 0;
};

/// Read-only admission result shared by every segmented-program consumer.
struct SegmentProgramAnalysis {
  llvm::SmallVector<SegmentIdOp> segmentIds;
  llvm::SmallVector<MakeSegmentOp> segments;
  llvm::SmallVector<MapOp> maps;
  llvm::SmallVector<ReduceOp> reductions;
  llvm::SmallVector<memref::StoreOp> stores;
  llvm::SmallVector<MapStoreOp> mapStores;
  llvm::SmallVector<func::ReturnOp> returns;
  ReduceOp storedReduction;
  SemanticBufferRoles roles;
};

FailureOr<func::FuncOp> findSegmentedReduction(ModuleOp module);
LogicalResult analyzeSegmentProgram(func::FuncOp function,
                                    SegmentProgramAnalysis &analysis);
void detachSegmentProgram(SegmentProgramAnalysis &analysis,
                          SegmentProgram &program);
LogicalResult verifyPlanningProgram(SegmentProgramAnalysis &analysis);
LogicalResult verifyPersistentProgram(SegmentProgramAnalysis &analysis);

Value evaluateElement(OpBuilder &builder, const ElementProgram &element,
                      Value value, ArrayRef<Value> reductions);
Value identityFor(OpBuilder &builder, Location loc, ReductionKind kind);
Value combine(OpBuilder &builder, Location loc, ReductionKind kind,
              Value accumulator, Value value);

void buildSequentialProgram(func::FuncOp function,
                            const SemanticBufferRoles &roles,
                            const SegmentProgram &program);
void buildGPUProgram(ModuleOp module, func::FuncOp source,
                     const SemanticBufferRoles &roles,
                     const SegmentProgram &program, int64_t blockSize,
                     SegmentedExecutionKind kind);
void buildSplitGPUProgram(ModuleOp module, func::FuncOp source,
                          const SemanticBufferRoles &roles,
                          const ReductionStage &stage, bool merge);

} // namespace mlir::swage::detail

#endif // SWAGE_LIB_CONVERSION_SEGMENTEDREDUCTION_SEGMENTPROGRAM_H
