// test/Conversion/SwageToCPU/segmented-sum-map-chain.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s

// A chain of two maps feeding one reduce. The two stages do not commute, so
// the emitted order is load-bearing: centering must happen before scaling.
module {
  func.func @segmented_chain(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %centered = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %shifted = arith.subf %value, %m : f32
      swage.yield %shifted : f32
    }
    %scaled = swage.map %centered
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      %two = arith.constant 2.000000e+00 : f32
      %doubled = arith.mulf %element, %two : f32
      swage.yield %doubled : f32
    }
    %total = swage.reduce %scaled kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %total, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @segmented_chain(
// CHECK: %[[MAX:.*]] = scf.for
// The chain is applied in source order inside one element loop: the scaling
// map consumes the centering map's result, never the loaded value.
// CHECK: scf.for %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK:   %[[VALUE:.*]] = memref.load
// CHECK:   %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[MAX]] : f32
// CHECK:   %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:   %[[SCALED:.*]] = arith.mulf %[[CENTERED]], %[[TWO]] : f32
// CHECK:   %[[NEXT:.*]] = arith.addf %[[ACC]], %[[SCALED]] : f32
// CHECK:   scf.yield %[[NEXT]] : f32
// CHECK: }
// CHECK-NOT: swage.
