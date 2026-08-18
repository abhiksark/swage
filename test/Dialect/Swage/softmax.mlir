// RUN: swage-opt %s | swage-opt | FileCheck %s

// The segmented-softmax semantic form from ADR-0008: two reductions and a
// store, with runtime identity flowing through SSA values only.

// CHECK-LABEL: func.func @segmented_softmax
func.func @segmented_softmax(%values: memref<?xf32>, %offsets: memref<?xi32>, %out: memref<?xf32>) {
  // CHECK: swage.segment_id 0
  %sid = swage.segment_id 0
  // CHECK: swage.make_segment
  %s = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
  // CHECK: swage.reduce %{{.*}} kind<max>
  %max = swage.reduce %s kind<max> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  // CHECK: swage.reduce %{{.*}} captures(%{{.*}} : f32) kind<sum>
  %den = swage.reduce %s captures(%max : f32) kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32, %m: f32):
    %sh = arith.subf %x, %m : f32
    %e = math.exp %sh : f32
    swage.yield %e : f32
  }
  // CHECK: swage.map_store %{{.*}}, %{{.*}} captures(%{{.*}}, %{{.*}} : f32, f32)
  swage.map_store %s, %out captures(%max, %den : f32, f32) : !swage.segment<f32>, memref<?xf32> {
  ^bb0(%x: f32, %m: f32, %d: f32):
    %sh = arith.subf %x, %m : f32
    %e = math.exp %sh : f32
    %r = arith.divf %e, %d : f32
    swage.yield %r : f32
  }
  return
}
