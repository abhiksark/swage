// test/Conversion/SwageToPlan/two-kernels.mlir
// The planner plans every function that holds Swage operations and leaves
// the other functions as they are. The function option restricts it to one
// function, and a module without a segment function is left unchanged.
//
// RUN: swage-opt --swage-to-plan %s | FileCheck %s --check-prefix=ALL
// RUN: swage-opt --swage-to-plan='function=second' %s \
// RUN:   | FileCheck %s --check-prefix=ONE

// ALL: func.func @first({{.*}}) attributes {swage_plan.block_threads = 128 : i32} {
// ALL-NEXT: swage_plan.tasks policy<cta>
// ALL: swage.reduce %{{.*}} kind<sum>
// ALL: func.func @bystander(%[[X:.*]]: i32) -> i32 {
// ALL-NEXT: return %[[X]] : i32
// ALL: func.func @second({{.*}}) attributes {swage_plan.block_threads = 128 : i32} {
// ALL-NEXT: swage_plan.tasks policy<cta>
// ALL: swage.reduce %{{.*}} kind<max>

// The function that is not named keeps its segment operations.
// ONE: func.func @first(
// ONE-NOT: swage_plan
// ONE: swage.make_segment
// ONE: func.func @bystander(
// ONE: func.func @second({{.*}}) attributes {swage_plan.block_threads = 128 : i32} {
// ONE-NEXT: swage_plan.tasks policy<cta>

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
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
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
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %max, %output[%sid] : memref<?xf32>
    return
  }
}
