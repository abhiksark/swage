// RUN: swage-opt --canonicalize %s | FileCheck %s

// An unused map with a pure body is trivially dead (recursive effects);
// map_store's declared write keeps it alive (ADR-0008).

// CHECK-LABEL: func.func @dce_contract
// CHECK-NOT: swage.map %
// CHECK: swage.map_store
// CHECK-NOT: swage.map %
func.func @dce_contract(%s: !swage.segment<f32>, %out: memref<?xf32>) {
  %dead = swage.map %s : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %e = math.exp %x : f32
    swage.yield %e : f32
  }
  swage.map_store %s, %out : !swage.segment<f32>, memref<?xf32> {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return
}
