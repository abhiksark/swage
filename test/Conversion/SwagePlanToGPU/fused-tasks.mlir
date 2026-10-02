// test/Conversion/SwagePlanToGPU/fused-tasks.mlir
// The plan function of the fused mixed kernel, written by hand. The first
// blocks each run one warp task per subgroup, on the lanes of the subgroup
// and through the warp region; every block after them runs one block task
// through the block region. Each task index is compared with its count, and
// each loaded segment id with the segment count.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s \
// RUN:   --implicit-check-not=swage --implicit-check-not=func.func
// RUN: not swage-opt --swage-plan-to-gpu %S/Inputs/fused-partial-subgroup.mlir \
// RUN:   2>&1 | FileCheck %s --check-prefix=SUBGROUPS

// CHECK: gpu.module @fused_module {
// CHECK-NEXT: gpu.func @fused(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %[[IDS:.*]]: !llvm.ptr, %{{.*}}: i32, %[[WARP_TASK_COUNT:.*]]: i32, %[[CTA_TASK_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 128, 1, 1>} {
// CHECK-NEXT: %[[BLOCK:.*]] = gpu.block_id x
// CHECK-NEXT: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[C128:.*]] = arith.constant 128 : index
// CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[C4:.*]] = arith.constant 4 : index
// CHECK-NEXT: %[[C32:.*]] = arith.constant 32 : index
// CHECK-NEXT: %[[WARP_TASKS:.*]] = arith.index_cast %[[WARP_TASK_COUNT]] : i32 to index
// CHECK-NEXT: %[[CTA_TASKS:.*]] = arith.index_cast %[[CTA_TASK_COUNT]] : i32 to index
// CHECK-NEXT: %[[ROUNDED:.*]] = arith.addi %[[WARP_TASKS]], %[[C3]] : index
// CHECK-NEXT: %[[WARP_BLOCKS:.*]] = arith.divui %[[ROUNDED]], %[[C4]] : index
// CHECK-NEXT: %[[IS_WARP_BLOCK:.*]] = arith.cmpi ult, %[[BLOCK]], %[[WARP_BLOCKS]] : index
// CHECK-NEXT: scf.if %[[IS_WARP_BLOCK]] {
// CHECK-NEXT: %[[SUBGROUP:.*]] = arith.divui %[[THREAD]], %[[C32]] : index
// CHECK-NEXT: %[[LANE:.*]] = arith.remui %[[THREAD]], %[[C32]] : index
// CHECK-NEXT: %[[FIRST_TASK:.*]] = arith.muli %[[BLOCK]], %[[C4]] : index
// CHECK-NEXT: %[[WARP_TASK:.*]] = arith.addi %[[FIRST_TASK]], %[[SUBGROUP]] : index
// CHECK-NEXT: %[[HAS_WARP_TASK:.*]] = arith.cmpi ult, %[[WARP_TASK]], %[[WARP_TASKS]] : index
// CHECK-NEXT: scf.if %[[HAS_WARP_TASK]] {
// CHECK: llvm.getelementptr %[[IDS]][
// CHECK-NEXT: %[[WARP_ID:.*]] = llvm.load
// CHECK-NEXT: arith.cmpi ult, %[[WARP_ID]], %[[SEGMENT_COUNT]] : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C32]] iter_args(
// CHECK: arith.maximumf
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK: arith.cmpi eq, %[[LANE]], %[[C0]] : index
// CHECK: } else {
// CHECK-NEXT: %[[CTA_TASK:.*]] = arith.subi %[[BLOCK]], %[[WARP_BLOCKS]] : index
// CHECK-NEXT: %[[HAS_CTA_TASK:.*]] = arith.cmpi ult, %[[CTA_TASK]], %[[CTA_TASKS]] : index
// CHECK-NEXT: scf.if %[[HAS_CTA_TASK]] {
// CHECK-NEXT: %[[MIXED_TASK:.*]] = arith.addi %[[WARP_TASKS]], %[[CTA_TASK]] : index
// CHECK: llvm.getelementptr %[[IDS]][
// CHECK-NEXT: %[[CTA_ID:.*]] = llvm.load
// CHECK-NEXT: arith.cmpi ult, %[[CTA_ID]], %[[SEGMENT_COUNT]] : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C128]] iter_args(
// CHECK: gpu.all_reduce maximumf
// CHECK: arith.cmpi eq, %[[THREAD]], %[[C0]] : index
// CHECK: gpu.return

// SUBGROUPS: error: a fused task block is a whole number of subgroups of 32 threads, got swage_plan.block_threads = 100

module {
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
      %total = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
