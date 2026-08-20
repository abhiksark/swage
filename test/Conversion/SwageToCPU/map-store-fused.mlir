// test/Conversion/SwageToCPU/map-store-fused.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// A map fused into a map_store terminal, which is the one consumer kind the
// scalar-store programs never exercise.
module {
  func.func @centered_store(
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
    swage.map_store %centered, %output
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%element: f32):
      %two = arith.constant 2.000000e+00 : f32
      %doubled = arith.mulf %element, %two : f32
      swage.yield %doubled : f32
    }
    return
  }
}

// CHECK-LABEL: func.func @centered_store(
// CHECK: %[[MAX:.*]] = scf.for
// The map body runs inside the terminal's element loop, before the store.
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[VALUE:.*]] = memref.load
// CHECK:   %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[MAX]] : f32
// CHECK:   %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK:   %[[DOUBLED:.*]] = arith.mulf %[[CENTERED]], %[[TWO]] : f32
// CHECK:   memref.store %[[DOUBLED]], %{{.*}}[%[[I]]]
// CHECK: }
