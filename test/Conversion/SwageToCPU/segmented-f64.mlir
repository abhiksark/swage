// test/Conversion/SwageToCPU/segmented-f64.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' \
// RUN:   --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=f32

module {
  func.func @segmented_sum_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
  func.func @segmented_max_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<max>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
  func.func @segmented_min_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<min>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
}

// CHECK-LABEL: func.func @segmented_sum_f64(
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[SUM:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f64) {
// CHECK:   %[[VALUE:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<?xf64>
// CHECK:   %[[NEXT:.*]] = arith.addf %[[ACC]], %[[VALUE]] : f64
// CHECK:   scf.yield %[[NEXT]] : f64
// CHECK: }
// CHECK: memref.store %[[SUM]], %{{.*}}[%{{.*}}] : memref<?xf64>
// CHECK-LABEL: func.func @segmented_max_f64(
// CHECK: arith.constant 0xFFF0000000000000 : f64
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : f64
// CHECK-LABEL: func.func @segmented_min_f64(
// CHECK: arith.constant 0x7FF0000000000000 : f64
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : f64
// CHECK-NOT: swage.
