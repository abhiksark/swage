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

  // Over rank-two values a merge reduces scratch rows into the row of its
  // segment. With the range records, the extent is a number of rows; the
  // two optional operands print in their own clauses.
  // CHECK-LABEL: func.func @merge_rows(
  // CHECK-SAME: %[[R_SCRATCH:.*]]: memref<?x?xf32>, %[[R_OUTPUT:.*]]: memref<?x?xf32>, %[[R_MERGES:.*]]: memref<?xi32>, %[[R_RANGES:.*]]: memref<?xi32>, %[[R_PARTIAL_COUNT:.*]]: i32, %[[R_MERGE_COUNT:.*]]: i32, %[[R_SEGMENT_COUNT:.*]]: i32, %[[R_FEATURE_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 512 : i32} {
  // CHECK-NEXT: swage_plan.merge_tasks scratch(%[[R_SCRATCH]] : memref<?x?xf32>) partial_count(%[[R_PARTIAL_COUNT]] : i32) merges(%[[R_MERGES]] : memref<?xi32>) merge_count(%[[R_MERGE_COUNT]] : i32) segment_count(%[[R_SEGMENT_COUNT]] : i32) feature_count(%[[R_FEATURE_COUNT]] : i32) ranges(%[[R_RANGES]] : memref<?xi32>) into(%[[R_OUTPUT]] : memref<?x?xf32>) {
  // CHECK-NEXT: ^bb0(%[[R_PARTIALS:.*]]: !swage.segment<f32>, %[[R_ROWS:.*]]: index):
  // CHECK: arith.index_cast %[[R_ROWS]] : index to i32
  // CHECK: %[[R_MEAN:.*]] = arith.divf
  // CHECK-NEXT: swage_plan.yield %[[R_MEAN]] : f32
  func.func @merge_rows(
      %scratch: memref<?x?xf32>, %output: memref<?x?xf32>,
      %merges: memref<?xi32>, %ranges: memref<?xi32>, %partial_count: i32,
      %merge_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.merge_tasks scratch(%scratch : memref<?x?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32) ranges(%ranges : memref<?xi32>)
        into(%output : memref<?x?xf32>) {
    ^bb0(%partials: !swage.segment<f32>, %rows: index):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      %count = arith.index_cast %rows : index to i32
      %divisor = arith.sitofp %count : i32 to f32
      %mean = arith.divf %total, %divisor : f32
      swage_plan.yield %mean : f32
    }
    return
  }
}
