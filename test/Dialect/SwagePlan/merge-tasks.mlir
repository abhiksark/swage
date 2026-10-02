// test/Dialect/SwagePlan/merge-tasks.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the plan function of a split merge
// kernel.

module {
  // CHECK-LABEL: func.func @merge(
  // CHECK-SAME: %[[SCRATCH:.*]]: memref<?xf32>, %[[OUTPUT:.*]]: memref<?xf32>, %[[MERGES:.*]]: memref<?xi32>, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 512 : i32} {
  // CHECK-NEXT: swage_plan.merge_tasks scratch(%[[SCRATCH]] : memref<?xf32>) partial_count(%[[PARTIAL_COUNT]] : i32) merges(%[[MERGES]] : memref<?xi32>) merge_count(%[[MERGE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) into(%[[OUTPUT]] : memref<?xf32>) {
  // CHECK-NEXT: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[TOTAL:.*]] = swage.reduce %[[PARTIALS]] kind<sum> : !swage.segment<f32> -> f32 {
  // CHECK: swage_plan.yield %[[TOTAL]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @merge(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
