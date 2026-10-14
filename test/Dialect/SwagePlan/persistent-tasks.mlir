// test/Dialect/SwagePlan/persistent-tasks.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the plan function of the persistent
// queue kernel: one task operation with a region for a block task, a
// partial task, a merge, and a warp task.

module {
  // CHECK-LABEL: func.func @queues(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xf32>, %[[WARP_IDS:.*]]: memref<?xi32>, %[[CTA_IDS:.*]]: memref<?xi32>, %[[RANGES:.*]]: memref<?xi32>, %[[MERGE_IDS:.*]]: memref<?xi32>, %[[MERGES:.*]]: memref<?xi32>, %[[SCRATCH:.*]]: memref<?xf32>, %[[COUNTERS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32, %[[WARP_TASKS:.*]]: i32, %[[CTA_TASKS:.*]]: i32, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 512 : i32} {
  // CHECK-NEXT: swage_plan.persistent_tasks segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) warp_ids(%[[WARP_IDS]] : memref<?xi32>) warp_task_count(%[[WARP_TASKS]] : i32) cta_ids(%[[CTA_IDS]] : memref<?xi32>) cta_task_count(%[[CTA_TASKS]] : i32) ranges(%[[RANGES]] : memref<?xi32>) merge_ids(%[[MERGE_IDS]] : memref<?xi32>) partial_count(%[[PARTIAL_COUNT]] : i32) merges(%[[MERGES]] : memref<?xi32>) merge_count(%[[MERGE_COUNT]] : i32) scratch(%[[SCRATCH]] : memref<?xf32>) counters(%[[COUNTERS]] : memref<?xi32>) into(%[[OUTPUT]] : memref<?xf32>) cta {
  // CHECK-NEXT: ^bb0(%[[CTA_SEGMENT:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[CTA_TOTAL:.*]] = swage.reduce %[[CTA_SEGMENT]] kind<sum>
  // CHECK: swage_plan.yield %[[CTA_TOTAL]] : f32
  // CHECK-NEXT: } partial {
  // CHECK-NEXT: ^bb0(%[[CHUNK:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[PARTIAL_TOTAL:.*]] = swage.reduce %[[CHUNK]] kind<sum>
  // CHECK: swage_plan.yield %[[PARTIAL_TOTAL]] : f32
  // CHECK-NEXT: } merge {
  // CHECK-NEXT: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[MERGE_TOTAL:.*]] = swage.reduce %[[PARTIALS]] kind<sum>
  // CHECK: swage_plan.yield %[[MERGE_TOTAL]] : f32
  // CHECK-NEXT: } warp {
  // CHECK-NEXT: ^bb0(%[[WARP_SEGMENT:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[WARP_TOTAL:.*]] = swage.reduce %[[WARP_SEGMENT]] kind<sum>
  // CHECK: swage_plan.yield %[[WARP_TOTAL]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @queues(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
