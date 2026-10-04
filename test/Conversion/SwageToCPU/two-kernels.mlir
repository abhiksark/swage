// test/Conversion/SwageToCPU/two-kernels.mlir
// The sequential lowering rewrites every function that holds Swage
// operations in place and leaves the other functions as they are. A function
// it lowers may have callers, because the function stays. The function
// option restricts the pass to one function.
//
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --check-prefix=ALL
// RUN: swage-opt --swage-to-plan='schedule=sequential function=second' \
// RUN:   --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --check-prefix=ONE

// ALL-NOT: swage.
// ALL-LABEL: func.func @first(
// ALL: scf.for
// ALL: arith.addf
// ALL-LABEL: func.func @caller(
// ALL: call @first(
// ALL-LABEL: func.func @second(
// ALL: scf.for
// ALL: arith.maximumf

// ONE-LABEL: func.func @first(
// ONE-SAME: {swage.role = #swage.role<values>}
// ONE: swage.reduce %{{.*}} kind<sum>
// ONE-LABEL: func.func @caller(
// ONE-LABEL: func.func @second(
// ONE-NOT: swage.
// ONE: arith.maximumf

module {
  func.func @first(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
  func.func @caller(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    call @first(%values, %offsets, %output, %value_count, %segment_count)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    return
  }
  func.func @second(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
}
