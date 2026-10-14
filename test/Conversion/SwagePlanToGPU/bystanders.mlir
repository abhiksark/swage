// test/Conversion/SwagePlanToGPU/bystanders.mlir
// The conversion changes plan functions and nothing else. A segment
// function that was not planned keeps its Swage operations, and other
// functions, globals, and kernel modules stay as they are. A plan function
// in a nested module is converted where it is.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s

// CHECK: module {
// CHECK-NEXT: memref.global "private" @table : memref<4xi32>
// CHECK-NEXT: gpu.module @earlier {
// CHECK-NEXT: }
// CHECK-NEXT: func.func @unplanned(%{{.*}}: memref<?xf32> {swage.role = #swage.role<values>},
// CHECK-NEXT: %{{.*}} = swage.segment_id 0
// CHECK-NEXT: %{{.*}} = swage.make_segment
// CHECK-NEXT: %{{.*}} = swage.reduce
// CHECK: memref.store
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: gpu.module @planned_module {
// CHECK-NEXT: gpu.func @planned(
// CHECK: gpu.return
// CHECK: func.func @bystander(%[[X:.*]]: i32) -> i32 {
// CHECK-NEXT: return %[[X]] : i32
// A plan function in a nested module becomes a kernel module beside it.
// CHECK: module @inner {
// CHECK-NEXT: gpu.module @nested_module {
// CHECK-NEXT: gpu.func @nested(
// CHECK: gpu.return

module {
  memref.global "private" @table : memref<4xi32>
  gpu.module @earlier {
  }
  func.func @unplanned(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
  func.func @planned(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
  func.func @bystander(%x: i32) -> i32 {
    return %x : i32
  }
  module @inner {
    func.func @nested(
        %values: memref<?xf32>, %offsets: memref<?xi32>,
        %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
        attributes {swage_plan.block_threads = 128 : i32} {
      swage_plan.tasks policy<cta>
          segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
          value_count(%value_count : i32) segment_count(%segment_count : i32)
          into(%output : memref<?xf32>) {
      ^bb0(%segment: !swage.segment<f32>):
        %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
        ^bb0(%value: f32):
          swage.yield %value : f32
        }
        swage_plan.yield %sum : f32
      }
      return
    }
  }
}
