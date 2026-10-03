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

  // Over rank-two values the chunk is a range of rows, and each task stores
  // one scratch row of feature_count columns.
  // CHECK-LABEL: func.func @partial_rows(
  // CHECK-SAME: %[[R_VALUES:.*]]: memref<?x?xf64>, %[[R_RANGES:.*]]: memref<?xi32>, %[[R_SCRATCH:.*]]: memref<?x?xf64>, %[[R_VALUE_COUNT:.*]]: i32, %[[R_PARTIAL_COUNT:.*]]: i32, %[[R_FEATURE_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 512 : i32} {
  // CHECK-NEXT: swage_plan.partial_tasks values(%[[R_VALUES]] : memref<?x?xf64>) value_count(%[[R_VALUE_COUNT]] : i32) ranges(%[[R_RANGES]] : memref<?xi32>) partial_count(%[[R_PARTIAL_COUNT]] : i32) feature_count(%[[R_FEATURE_COUNT]] : i32) into(%[[R_SCRATCH]] : memref<?x?xf64>) {
  // CHECK-NEXT: ^bb0(%[[R_CHUNK:.*]]: !swage.segment<f64>):
  // CHECK-NEXT: %[[R_TOTAL:.*]] = swage.reduce %[[R_CHUNK]] kind<sum> : !swage.segment<f64> -> f64 {
  // CHECK: swage_plan.yield %[[R_TOTAL]] : f64
  func.func @partial_rows(
      %values: memref<?x?xf64>, %ranges: memref<?xi32>,
      %scratch: memref<?x?xf64>, %value_count: i32, %partial_count: i32,
      %feature_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.partial_tasks values(%values : memref<?x?xf64>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32)
        feature_count(%feature_count : i32)
        into(%scratch : memref<?x?xf64>) {
    ^bb0(%chunk: !swage.segment<f64>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f64> -> f64 {
      ^bb0(%value: f64):
        swage.yield %value : f64
      }
      swage_plan.yield %total : f64
    }
    return
  }
}
