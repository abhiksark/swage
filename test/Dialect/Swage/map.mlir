// RUN: swage-opt %s | swage-opt | FileCheck %s

// CHECK-LABEL: func.func @map_exp
func.func @map_exp(%s: !swage.segment<f32>) -> !swage.segment<f32> {
  // CHECK: %{{.*}} = swage.map %{{.*}} : !swage.segment<f32> -> !swage.segment<f32> {
  // CHECK: ^bb0(%{{.*}}: f32):
  // CHECK: swage.yield %{{.*}} : f32
  %r = swage.map %s : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32):
    %e = math.exp %x : f32
    swage.yield %e : f32
  }
  return %r : !swage.segment<f32>
}

// CHECK-LABEL: func.func @map_changes_element_type
func.func @map_changes_element_type(%s: !swage.segment<f32>) -> !swage.segment<f16> {
  // CHECK: swage.map %{{.*}} : !swage.segment<f32> -> !swage.segment<f16>
  %r = swage.map %s : !swage.segment<f32> -> !swage.segment<f16> {
  ^bb0(%x: f32):
    %t = arith.truncf %x : f32 to f16
    swage.yield %t : f16
  }
  return %r : !swage.segment<f16>
}

// CHECK-LABEL: func.func @map_with_captures
func.func @map_with_captures(%s: !swage.segment<f32>, %scale: f32) -> !swage.segment<f32> {
  // CHECK: swage.map %{{.*}} captures(%{{.*}} : f32) : !swage.segment<f32> -> !swage.segment<f32>
  %r = swage.map %s captures(%scale : f32) : !swage.segment<f32> -> !swage.segment<f32> {
  ^bb0(%x: f32, %m: f32):
    %y = arith.mulf %x, %m : f32
    swage.yield %y : f32
  }
  return %r : !swage.segment<f32>
}
