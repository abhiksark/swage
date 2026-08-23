// test/Conversion/SwageToCPU/segmented-sum-map.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s

// Two reduction stages, where the second reduces a swage.map that captures the
// first stage's result. The map is fused into the consumer's element loop and
// no intermediate segment is materialized.
module {
  func.func @segmented_exp_sum(
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
    memref.store %total, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @segmented_exp_sum(
// CHECK: scf.for %[[SID:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[START:.*]] = arith.index_cast
// CHECK:   %[[END:.*]] = arith.index_cast
// The maximum stage runs first, with the negative-infinity identity.
// CHECK:   %[[NINF:.*]] = arith.constant 0xFF800000 : f32
// CHECK:   %[[MAX:.*]] = scf.for %{{.*}} iter_args(%[[MACC:.*]] = %[[NINF]]) -> (f32) {
// CHECK:     %[[MVALUE:.*]] = memref.load
// CHECK:     %[[MNEXT:.*]] = arith.maximumf %[[MACC]], %[[MVALUE]] : f32
// CHECK:     scf.yield %[[MNEXT]] : f32
// CHECK:   }
// The sum stage inlines the map body, binding the captured maximum.
// CHECK:   %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[TOTAL:.*]] = scf.for %{{.*}} iter_args(%[[SACC:.*]] = %[[ZERO]]) -> (f32) {
// CHECK:     %[[SVALUE:.*]] = memref.load
// CHECK:     %[[LOG2E:.*]] = arith.constant 1.44269502 : f32
// CHECK:     %[[CENTERED:.*]] = arith.subf %[[SVALUE]], %[[MAX]] : f32
// CHECK:     %[[SCALED:.*]] = arith.mulf %[[CENTERED]], %[[LOG2E]] : f32
// CHECK:     %[[EXP:.*]] = math.exp2 %[[SCALED]] : f32
// CHECK:     %[[SNEXT:.*]] = arith.addf %[[SACC]], %[[EXP]] : f32
// CHECK:     scf.yield %[[SNEXT]] : f32
// CHECK:   }
// CHECK:   memref.store %[[TOTAL]], %{{.*}}[%[[SID]]]
// CHECK: }
// CHECK-NOT: swage.
