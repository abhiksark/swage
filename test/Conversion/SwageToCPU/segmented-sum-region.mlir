// test/Conversion/SwageToCPU/segmented-sum-region.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s

// A non-identity reduction region is inlined into the sequential element loop.
module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %square = arith.mulf %value, %value : f32
      %scaled = math.exp2 %square : f32
      swage.yield %scaled : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @segmented_sum(
// CHECK: scf.for
// CHECK:   %[[IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[SUM:.*]] = scf.for %{{.*}} iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:     %[[VALUE:.*]] = memref.load
// CHECK:     %[[SQUARE:.*]] = arith.mulf %[[VALUE]], %[[VALUE]] : f32
// CHECK:     %[[SCALED:.*]] = math.exp2 %[[SQUARE]] : f32
// CHECK:     %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[SCALED]] : f32
// CHECK:     scf.yield %[[NEXT_ACC]] : f32
// CHECK:   }
// CHECK:   memref.store %[[SUM]]
// CHECK: }
// CHECK-NOT: swage.
