// test/Conversion/SwageToPlan/two-kernels.mlir
// Planning adds one companion per function that holds Swage operations and
// leaves the other functions as they are. The function option restricts it
// to one function.
//
// RUN: swage-opt --swage-to-plan %s | FileCheck %s --check-prefix=ALL
// RUN: swage-opt --swage-to-plan='function=second' %s \
// RUN:   | FileCheck %s --check-prefix=ONE \
// RUN:     --implicit-check-not=first__swage_plan

// ALL-LABEL: func.func @first(
// ALL-LABEL: func.func private @first__swage_plan(
// ALL: swage_plan.classify {{.*}} kernel = @first,
// ALL-LABEL: func.func @bystander(
// ALL-NOT: bystander__swage_plan
// ALL-LABEL: func.func @second(
// ALL-LABEL: func.func private @second__swage_plan(
// ALL: swage_plan.classify {{.*}} kernel = @second,

// ONE-LABEL: func.func @first(
// ONE-LABEL: func.func @bystander(
// ONE-LABEL: func.func @second(
// ONE-LABEL: func.func private @second__swage_plan(
// ONE: swage_plan.classify {{.*}} kernel = @second,

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
  func.func @bystander(%x: i32) -> i32 {
    return %x : i32
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
