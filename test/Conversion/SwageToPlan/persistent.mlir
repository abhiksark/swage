// test/Conversion/SwageToPlan/persistent.mlir
// The persistent schedule plans the queue kernel. The plan function takes
// the parameters of that kernel at the persistent launch width of the
// target and holds one persistent task operation. Its block, partial, and
// warp regions hold the program, and its merge region an identity reduction
// of the same kind over scratch. The programs the schedule refuses are in
// ../SwageToGPU/invalid-persistent.mlir.
//
// RUN: swage-opt --swage-to-plan='schedule=persistent' %s | FileCheck %s
// RUN: swage-opt --swage-to-plan='schedule=persistent block-threads=32' %s \
// RUN:   | FileCheck %s
// RUN: not swage-opt --swage-to-plan='schedule=persistent,direct' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-KERNEL

// CHECK: func.func @segmented_sum(%[[VALUES:.*]]: memref<?xf32> {swage.role = #swage.role<values>}, %[[OFFSETS:.*]]: memref<?xi32> {swage.role = #swage.role<offsets>}, %[[OUTPUT:.*]]: memref<?xf32> {swage.role = #swage.role<output>}, %[[WARP_IDS:.*]]: memref<?xi32>, %[[CTA_IDS:.*]]: memref<?xi32>, %[[RANGES:.*]]: memref<?xi32>, %[[MERGE_IDS:.*]]: memref<?xi32>, %[[MERGES:.*]]: memref<?xi32>, %[[SCRATCH:.*]]: memref<?xf32>, %[[COUNTERS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>}, %[[WARP_TASKS:.*]]: i32, %[[CTA_TASKS:.*]]: i32, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>}) attributes {swage_plan.block_threads = 512 : i32} {
// CHECK-NEXT: swage_plan.persistent_tasks segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) warp_ids(%[[WARP_IDS]] : memref<?xi32>) warp_task_count(%[[WARP_TASKS]] : i32) cta_ids(%[[CTA_IDS]] : memref<?xi32>) cta_task_count(%[[CTA_TASKS]] : i32) ranges(%[[RANGES]] : memref<?xi32>) merge_ids(%[[MERGE_IDS]] : memref<?xi32>) partial_count(%[[PARTIAL_COUNT]] : i32) merges(%[[MERGES]] : memref<?xi32>) merge_count(%[[MERGE_COUNT]] : i32) scratch(%[[SCRATCH]] : memref<?xf32>) counters(%[[COUNTERS]] : memref<?xi32>) into(%[[OUTPUT]] : memref<?xf32>) cta {
// CHECK-NEXT: ^bb0(%[[CTA:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[CTA_SUM:.*]] = swage.reduce %[[CTA]] kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: swage.yield %[[ELEMENT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield %[[CTA_SUM]] : f32
// CHECK-NEXT: } partial {
// CHECK-NEXT: ^bb0(%[[CHUNK:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[CHUNK_SUM:.*]] = swage.reduce %[[CHUNK]] kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: swage.yield %[[ELEMENT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield %[[CHUNK_SUM]] : f32
// CHECK-NEXT: } merge {
// CHECK-NEXT: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[PARTIALS_SUM:.*]] = swage.reduce %[[PARTIALS]] kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: swage.yield %[[ELEMENT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield %[[PARTIALS_SUM]] : f32
// CHECK-NEXT: } warp {
// CHECK-NEXT: ^bb0(%[[WARP:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[WARP_SUM:.*]] = swage.reduce %[[WARP]] kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: swage.yield %[[ELEMENT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield %[[WARP_SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return

// SAME-KERNEL: error: schedules persistent and direct both name their kernel @<function>; a schedule list names each kernel once

module {
  func.func @segmented_sum(
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
}
