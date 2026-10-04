// test/Conversion/SwageToPlan/fused-mixed.mlir
// The fused-mixed schedule plans the kernel that serves warp tasks and block
// tasks in one launch. The plan function takes the parameters of that
// kernel at the block-task width of the target and holds one fused task
// operation, whose two regions hold the same program: a warp task and a
// block task differ in how their threads combine, not in what they compute.
//
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' %s | FileCheck %s
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed block-threads=32' %s \
// RUN:   | FileCheck %s
// RUN: not swage-opt --swage-to-plan='schedule=task-ids,fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-KERNEL

// CHECK: func.func @segmented_max(%[[VALUES:.*]]: memref<?xf32> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %[[OFFSETS:.*]]: memref<?xi32> {swage.role = #swage.role<offsets>, swage_plan.source_index = 1 : i32}, %[[OUTPUT:.*]]: memref<?xf32> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %[[IDS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>, swage_plan.source_index = 3 : i32}, %[[WARP_TASKS:.*]]: i32, %[[CTA_TASKS:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 4 : i32}) attributes {swage_plan.block_threads = 128 : i32} {
// CHECK-NEXT: swage_plan.fused_tasks segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) ids(%[[IDS]] : memref<?xi32>) warp_task_count(%[[WARP_TASKS]] : i32) cta_task_count(%[[CTA_TASKS]] : i32) into(%[[OUTPUT]] : memref<?xf32>) warp {
// CHECK-NEXT: ^bb0(%[[WARP_SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[WARP_MAX:.*]] = swage.reduce %[[WARP_SEGMENT]] kind<max> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: %[[SQUARE:.*]] = arith.mulf %[[ELEMENT]], %[[ELEMENT]] : f32
// CHECK-NEXT: swage.yield %[[SQUARE]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield %[[WARP_MAX]] : f32
// CHECK-NEXT: } cta {
// CHECK-NEXT: ^bb0(%[[CTA_SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[CTA_MAX:.*]] = swage.reduce %[[CTA_SEGMENT]] kind<max> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// CHECK-NEXT: %[[SQUARE:.*]] = arith.mulf %[[ELEMENT]], %[[ELEMENT]] : f32
// CHECK-NEXT: swage.yield %[[SQUARE]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield %[[CTA_MAX]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return

// SAME-KERNEL: error: schedules task-ids and fused-mixed both name their kernel @<function>; a schedule list names each kernel once

module {
  func.func @segmented_max(
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
    %max = swage.reduce %squares kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %max, %output[%sid] : memref<?xf32>
    return
  }
}
