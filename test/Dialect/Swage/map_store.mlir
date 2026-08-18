// RUN: swage-opt %s | swage-opt | FileCheck %s

// CHECK-LABEL: func.func @map_store
func.func @map_store(%s: !swage.segment<f32>, %out: memref<?xf32>) {
  // CHECK: swage.map_store %{{.*}}, %{{.*}} : !swage.segment<f32>, memref<?xf32>
  swage.map_store %s, %out : !swage.segment<f32>, memref<?xf32> {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return
}

// CHECK-LABEL: func.func @map_store_with_captures
func.func @map_store_with_captures(%s: !swage.segment<f32>, %out: memref<?xf32>, %m: f32, %d: f32) {
  // CHECK: swage.map_store %{{.*}}, %{{.*}} captures(%{{.*}}, %{{.*}} : f32, f32) : !swage.segment<f32>, memref<?xf32>
  swage.map_store %s, %out captures(%m, %d : f32, f32) : !swage.segment<f32>, memref<?xf32> {
  ^bb0(%x: f32, %mm: f32, %dd: f32):
    %sh = arith.subf %x, %mm : f32
    %e = math.exp %sh : f32
    %r = arith.divf %e, %dd : f32
    swage.yield %r : f32
  }
  return
}
