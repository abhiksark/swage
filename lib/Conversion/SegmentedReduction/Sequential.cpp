// lib/Conversion/SegmentedReduction/Sequential.cpp
//===- Sequential.cpp - Sequential segmented emitter
//-----------------------------===//
//
// Part of the Swage project, under the MIT License.
// See LICENSE for license information.
//
//===----------------------------------------------------------------------===//

#include "SegmentProgram.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

using namespace mlir;

namespace mlir::swage::detail {

void buildSequentialProgram(func::FuncOp function,
                            const SemanticBufferRoles &roles,
                            const SegmentProgram &program) {
  Block &entry = function.getBody().front();
  while (!entry.empty())
    entry.back().erase();

  OpBuilder builder(function.getContext());
  Location loc = function.getLoc();
  builder.setInsertionPointToEnd(&entry);
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  // Materialize both semantic extents. The sequential loop needs only the
  // offset-derived segment count; value_count remains physical ABI metadata.
  memref::DimOp::create(builder, loc, roles.values, zero);
  Value offsetCount = memref::DimOp::create(builder, loc, roles.offsets, zero);
  Value segmentCount = arith::SubIOp::create(builder, loc, offsetCount, one);
  scf::ForOp::create(
      builder, loc, zero, segmentCount, one, ValueRange(),
      [&](OpBuilder &outer, Location outerLoc, Value segmentId, ValueRange) {
        Value startI32 =
            memref::LoadOp::create(outer, outerLoc, roles.offsets, segmentId);
        Value next = arith::AddIOp::create(outer, outerLoc, segmentId, one);
        Value endI32 =
            memref::LoadOp::create(outer, outerLoc, roles.offsets, next);
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
                Value value = memref::LoadOp::create(inner, innerLoc,
                                                     roles.values, index);
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
                                  roles.output, segmentId);
        } else {
          scf::ForOp::create(outer, outerLoc, start, end, one, ValueRange(),
                             [&](OpBuilder &inner, Location innerLoc,
                                 Value index, ValueRange) {
                               Value value = memref::LoadOp::create(
                                   inner, innerLoc, roles.values, index);
                               value = evaluateElement(inner, program.mapStore,
                                                       value, results);
                               memref::StoreOp::create(inner, innerLoc, value,
                                                       roles.output, index);
                               scf::YieldOp::create(inner, innerLoc);
                             });
        }
        scf::YieldOp::create(outer, outerLoc);
      });
  func::ReturnOp::create(builder, loc);
}

} // namespace mlir::swage::detail
