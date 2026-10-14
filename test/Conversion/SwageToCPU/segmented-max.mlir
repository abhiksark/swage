// test/Conversion/SwageToCPU/segmented-max.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' \
// RUN:   --swage-plan-to-scf %s | FileCheck %s

module {
  func.func @segmented_max(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @segmented_max(
// CHECK: %[[IDENTITY:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[MAXIMUM:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:   %[[VALUE:.*]] = memref.load
// CHECK:   %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[VALUE]] : f32
// CHECK:   scf.yield %[[NEXT]] : f32
// CHECK: }
// CHECK: memref.store %[[MAXIMUM]]
// CHECK-NOT: swage.
