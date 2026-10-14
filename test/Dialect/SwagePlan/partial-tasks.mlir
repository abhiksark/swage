// test/Dialect/SwagePlan/partial-tasks.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the plan function of a split
// partial kernel.

module {
  // CHECK-LABEL: func.func @partial(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>, %[[RANGES:.*]]: memref<?xi32>, %[[SCRATCH:.*]]: memref<?xf32>, %[[VALUE_COUNT:.*]]: i32, %[[PARTIAL_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 512 : i32} {
  // CHECK-NEXT: swage_plan.partial_tasks values(%[[VALUES]] : memref<?xf32>) value_count(%[[VALUE_COUNT]] : i32) ranges(%[[RANGES]] : memref<?xi32>) partial_count(%[[PARTIAL_COUNT]] : i32) into(%[[SCRATCH]] : memref<?xf32>) {
  // CHECK-NEXT: ^bb0(%[[CHUNK:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[TOTAL:.*]] = swage.reduce %[[CHUNK]] kind<sum> : !swage.segment<f32> -> f32 {
  // CHECK: swage_plan.yield %[[TOTAL]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @partial(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
