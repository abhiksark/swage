// test/Conversion/SwageToCPU/ragged-softmax.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// The canonical ragged softmax: a maximum stage, a sum over the shifted
// exponentials fused from a map, and a map_store terminal writing one value
// per element.
module {
  func.func @ragged_softmax(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @ragged_softmax(
// CHECK-SAME: %[[VALUES:[^:]+]]: memref<?xf32>
// CHECK-SAME: %[[OFFSETS:[^:]+]]: memref<?xi32>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?xf32>
// CHECK: scf.for %[[SID:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[START:.*]] = arith.index_cast
// CHECK:   %[[END:.*]] = arith.index_cast
// Stage one, the segment maximum.
// CHECK:   %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:   %[[MAX:.*]] = scf.for %{{.*}} = %[[START]] to %[[END]] step %{{.*}} iter_args(%[[MACC:.*]] = %[[NINF]]) -> (f32) {
// CHECK:     %[[MVALUE:.*]] = memref.load %[[VALUES]]
// CHECK:     %[[MNEXT:.*]] = arith.maximumf %[[MACC]], %[[MVALUE]] : f32
// CHECK:   }
// Stage two, the sum of shifted exponentials, with the map fused in.
// CHECK:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[TOTAL:.*]] = scf.for %{{.*}} = %[[START]] to %[[END]] step %{{.*}} iter_args(%[[SACC:.*]] = %[[ZERO]]) -> (f32) {
// CHECK:     %[[SVALUE:.*]] = memref.load %[[VALUES]]
// CHECK:     %[[SLOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK:     %[[SCENTERED:.*]] = arith.subf %[[SVALUE]], %[[MAX]] : f32
// CHECK:     %[[SSCALED:.*]] = arith.mulf %[[SCENTERED]], %[[SLOG2E]] : f32
// CHECK:     %[[SEXP:.*]] = math.exp2 %[[SSCALED]] : f32
// CHECK:     %[[SNEXT:.*]] = arith.addf %[[SACC]], %[[SEXP]] : f32
// CHECK:   }
// The terminal writes one value per element at its own index, not at the
// segment id, and carries no reduction accumulator.
// CHECK:   scf.for %[[I:.*]] = %[[START]] to %[[END]] step %{{.*}} {
// CHECK:     %[[TVALUE:.*]] = memref.load %[[VALUES]][%[[I]]]
// CHECK:     %[[TLOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK:     %[[TCENTERED:.*]] = arith.subf %[[TVALUE]], %[[MAX]] : f32
// CHECK:     %[[TSCALED:.*]] = arith.mulf %[[TCENTERED]], %[[TLOG2E]] : f32
// CHECK:     %[[TEXP:.*]] = math.exp2 %[[TSCALED]] : f32
// CHECK:     %[[NORMALIZED:.*]] = arith.divf %[[TEXP]], %[[TOTAL]] : f32
// CHECK:     memref.store %[[NORMALIZED]], %[[OUTPUT]][%[[I]]]
// CHECK:   }
// CHECK: }
