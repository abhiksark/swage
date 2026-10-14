// test/Dialect/SwagePlan/fused-tasks.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the plan function of the fused
// mixed kernel: one task operation with a warp region and a block region.

module {
  // CHECK-LABEL: func.func @fused(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xf32>, %[[IDS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32, %[[WARP_TASKS:.*]]: i32, %[[CTA_TASKS:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 128 : i32} {
  // CHECK-NEXT: swage_plan.fused_tasks segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) ids(%[[IDS]] : memref<?xi32>) warp_task_count(%[[WARP_TASKS]] : i32) cta_task_count(%[[CTA_TASKS]] : i32) into(%[[OUTPUT]] : memref<?xf32>) warp {
  // CHECK-NEXT: ^bb0(%[[WARP_SEGMENT:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[WARP_TOTAL:.*]] = swage.reduce %[[WARP_SEGMENT]] kind<sum>
  // CHECK: swage_plan.yield %[[WARP_TOTAL]] : f32
  // CHECK-NEXT: } cta {
  // CHECK-NEXT: ^bb0(%[[CTA_SEGMENT:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[CTA_TOTAL:.*]] = swage.reduce %[[CTA_SEGMENT]] kind<sum>
  // CHECK: swage_plan.yield %[[CTA_TOTAL]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @fused(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %warp_task_count: i32, %cta_task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) warp_task_count(%warp_task_count : i32)
        cta_task_count(%cta_task_count : i32)
        into(%output : memref<?xf32>) warp {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } cta {
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
