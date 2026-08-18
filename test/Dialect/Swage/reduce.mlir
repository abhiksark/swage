// RUN: swage-opt %s | swage-opt | FileCheck %s

// CHECK-LABEL: func.func @reduce_kinds
func.func @reduce_kinds(%s: !swage.segment<f32>) -> (f32, f32, f32) {
  // CHECK: swage.reduce %{{.*}} kind<sum> : !swage.segment<f32> -> f32
  %sum = swage.reduce %s kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  // CHECK: swage.reduce %{{.*}} kind<max> : !swage.segment<f32> -> f32
  %max = swage.reduce %s kind<max> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  // CHECK: swage.reduce %{{.*}} kind<min> : !swage.segment<f32> -> f32
  %min = swage.reduce %s kind<min> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32):
    swage.yield %x : f32
  }
  return %sum, %max, %min : f32, f32, f32
}

// CHECK-LABEL: func.func @reduce_with_captures
func.func @reduce_with_captures(%s: !swage.segment<f32>, %m: f32) -> f32 {
  // CHECK: swage.reduce %{{.*}} captures(%{{.*}} : f32) kind<sum> : !swage.segment<f32> -> f32
  %den = swage.reduce %s captures(%m : f32) kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%x: f32, %mm: f32):
    %sh = arith.subf %x, %mm : f32
    %e = math.exp %sh : f32
    swage.yield %e : f32
  }
  return %den : f32
}
