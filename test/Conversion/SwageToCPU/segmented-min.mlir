// test/Conversion/SwageToCPU/segmented-min.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' \
// RUN:   --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=maximumf

module {
  func.func @segmented_min(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %minimum = swage.reduce %segment kind<min>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %minimum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @segmented_min(
// CHECK: %[[IDENTITY:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[MINIMUM:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:   %[[VALUE:.*]] = memref.load
// CHECK:   %[[NEXT:.*]] = arith.minimumf %[[ACC]], %[[VALUE]] : f32
// CHECK:   scf.yield %[[NEXT]] : f32
// CHECK: }
// CHECK: memref.store %[[MINIMUM]]
// CHECK-NOT: swage.
