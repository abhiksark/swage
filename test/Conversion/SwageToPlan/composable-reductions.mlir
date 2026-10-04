// test/Conversion/SwageToPlan/composable-reductions.mlir
// What the task-id schedule admits: one capture-free sum or max, with an
// element expression and with map chains. The planner fuses each map into
// the reduction, so the task region holds one reduction over the bound
// segment and no map.
//
// RUN: swage-opt --swage-to-plan='schedule=task-ids' --split-input-file %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.map

// CHECK-LABEL: func.func @maximum(
// CHECK: swage_plan.tasks policy<cta>
// CHECK-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[MAX:.*]] = swage.reduce %[[SEGMENT]] kind<max>
// CHECK: swage_plan.yield %[[MAX]] : f32
module {
  func.func @maximum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @transformed_sum(
// CHECK: %[[SUM:.*]] = swage.reduce %{{.*}} kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: %[[DOUBLED:.*]] = arith.addf %[[ELEMENT]], %[[ELEMENT]] : f32
// CHECK-NEXT: swage.yield %[[DOUBLED]] : f32
// CHECK: swage_plan.yield %[[SUM]] : f32
module {
  func.func @transformed_sum(
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
      %doubled = arith.addf %element, %element : f32
      swage.yield %doubled : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// Two maps and the region of the reduction, in application order.
// CHECK-LABEL: func.func @mapped_sum(
// CHECK: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: %[[SQUARE:.*]] = arith.mulf %[[ELEMENT]], %[[ELEMENT]] : f32
// CHECK-NEXT: %[[DOUBLED:.*]] = arith.addf %[[SQUARE]], %[[SQUARE]] : f32
// CHECK-NEXT: %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-NEXT: %[[HALVED:.*]] = arith.divf %[[DOUBLED]], %[[TWO]] : f32
// CHECK-NEXT: swage.yield %[[HALVED]] : f32
// CHECK: swage_plan.yield %[[SUM]] : f32
module {
  func.func @mapped_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %squares = swage.map %segment : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      %square = arith.mulf %element, %element : f32
      swage.yield %square : f32
    }
    %doubles = swage.map %squares : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      %doubled = arith.addf %element, %element : f32
      swage.yield %doubled : f32
    }
    %sum = swage.reduce %doubles kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      %two = arith.constant 2.0 : f32
      %halved = arith.divf %element, %two : f32
      swage.yield %halved : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
