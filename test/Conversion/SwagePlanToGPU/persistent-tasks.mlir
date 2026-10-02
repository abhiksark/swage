// test/Conversion/SwagePlanToGPU/persistent-tasks.mlir
// A persistent plan function written by hand becomes the queue kernel. The
// function declares its parameters in an order of its own, so every check
// below follows an operand of the task operation to its use: the kernel
// takes its buffers and bounds from the operands and never from a position
// in the parameter list.
//
// The kernel text of the planned pipeline is pinned in full by
// ../SwageToGPU/persistent.mlir. This test pins what the conversion adds to
// the plan: the claim slots of the kernel, the counter of each queue, the
// bound each loaded index is compared with, and the buffer each region
// reads and writes.
//
// RUN: swage-opt --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=swage --implicit-check-not=func.func

// CHECK: gpu.module @drain_module {
// CHECK-NEXT: gpu.func @drain(%[[COUNTERS:.*]]: !llvm.ptr, %[[SCRATCH:.*]]: !llvm.ptr, %[[MERGES:.*]]: !llvm.ptr, %[[MERGE_COUNT:.*]]: i32, %[[RANGES:.*]]: !llvm.ptr, %[[MERGE_IDS:.*]]: !llvm.ptr, %[[PARTIAL_COUNT:.*]]: i32, %[[CTA_IDS:.*]]: !llvm.ptr, %[[CTA_TASKS:.*]]: i32, %[[WARP_IDS:.*]]: !llvm.ptr, %[[WARP_TASKS:.*]]: i32, %[[VALUES:.*]]: !llvm.ptr, %[[OFFSETS:.*]]: !llvm.ptr, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32, %[[OUTPUT:.*]]: !llvm.ptr) workgroup(%[[SLOTS:.*]] : memref<2xi32, #gpu.address_space<workgroup>>) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: %[[C0:.*]] = arith.constant 0 : index
// CHECK: %[[C1:.*]] = arith.constant 1 : index
// CHECK: %[[BLOCK:.*]] = arith.constant 512 : index
// CHECK: %[[ONE_TASK:.*]] = arith.constant 1 : i32
// CHECK: %[[PARTIAL_BATCH:.*]] = arith.constant 4 : i32
// CHECK: %[[WARP_BATCH:.*]] = arith.constant 8 : i32
// CHECK: %[[LEADER:.*]] = arith.cmpi eq, %[[THREAD]], %[[C0]] : index

// The block queue: counter 1, one task per claim, and the claim reaches
// the block through the first claim slot.
// CHECK: %[[CTA_COUNTER_INDEX:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[CTA_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[CTA_COUNTER_INDEX]]]
// CHECK: llvm.atomicrmw add %[[CTA_COUNTER]], %[[ONE_TASK]] monotonic
// CHECK: memref.store %{{.*}}, %[[SLOTS]][%[[C0]]]
// CHECK: gpu.barrier
// CHECK-NEXT: %[[FIRST_CTA:.*]] = memref.load %[[SLOTS]][%[[C0]]]
// CHECK-NEXT: scf.while (%[[CTA_CLAIM:.*]] = %[[FIRST_CTA]]) : (i32) -> i32 {
// CHECK-NEXT: %[[HAS_CTA:.*]] = arith.cmpi ult, %[[CTA_CLAIM]], %[[CTA_TASKS]] : i32
// CHECK: llvm.getelementptr %[[CTA_IDS]][
// CHECK-NEXT: %[[CTA_SEGMENT:.*]] = llvm.load
// CHECK-NEXT: %[[CTA_IN_RANGE:.*]] = arith.cmpi ult, %[[CTA_SEGMENT]], %[[SEGMENT_COUNT]] : i32
// CHECK: llvm.getelementptr %[[OFFSETS]][
// CHECK: arith.minsi %{{.*}}, %[[VALUE_COUNT]] : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[BLOCK]] iter_args
// CHECK: llvm.getelementptr %[[VALUES]][
// CHECK: %[[CTA_TOTAL:.*]] = gpu.all_reduce add %{{.*}} uniform
// CHECK: arith.andi %{{.*}}, %[[CTA_IN_RANGE]] : i1
// CHECK: %[[CTA_SLOT:.*]] = llvm.getelementptr %[[OUTPUT]][
// CHECK-NEXT: llvm.store %[[CTA_TOTAL]], %[[CTA_SLOT]] : f32, !llvm.ptr

// The partial queue: counter 2, a batch per claim.
// CHECK: %[[PARTIAL_COUNTER_INDEX:.*]] = arith.constant 2 : i64
// CHECK-NEXT: %[[PARTIAL_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[PARTIAL_COUNTER_INDEX]]]
// CHECK: llvm.atomicrmw add %[[PARTIAL_COUNTER]], %[[PARTIAL_BATCH]] monotonic
// CHECK: scf.while (%[[PARTIAL_CLAIM:.*]] = %{{.*}}) : (i32) -> i32 {
// CHECK-NEXT: arith.cmpi ult, %[[PARTIAL_CLAIM]], %[[PARTIAL_COUNT]] : i32
// CHECK: %[[BATCH_END:.*]] = arith.addi %{{.*}}, %[[PARTIAL_BATCH]] : i32
// CHECK-NEXT: arith.minui %[[BATCH_END]], %[[PARTIAL_COUNT]] : i32
// CHECK: scf.for %[[PARTIAL:.*]] = %{{.*}} to %{{.*}} step %[[C1]] {
// A partial task reduces its range of the values into its scratch slot.
// CHECK: llvm.getelementptr %[[RANGES]][
// CHECK: llvm.getelementptr %[[RANGES]][
// CHECK: arith.minsi %{{.*}}, %[[VALUE_COUNT]] : i32
// CHECK: llvm.getelementptr %[[VALUES]][
// CHECK: %[[PARTIAL_TOTAL:.*]] = gpu.all_reduce add %{{.*}} uniform
// CHECK: scf.if %[[LEADER]] {
// CHECK-NEXT: %[[PARTIAL_INDEX:.*]] = arith.index_cast %[[PARTIAL]] : index to i64
// CHECK-NEXT: %[[SCRATCH_SLOT:.*]] = llvm.getelementptr %[[SCRATCH]][%[[PARTIAL_INDEX]]]
// CHECK-NEXT: llvm.store %[[PARTIAL_TOTAL]], %[[SCRATCH_SLOT]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// The leader counts the completion of the merge of the partial task: the
// merge id is bounded by the merge count, the completion counters follow
// the three claim counters, and the fence comes before the count.
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK: llvm.getelementptr %[[MERGE_IDS]][
// CHECK-NEXT: %[[MERGE_ID:.*]] = llvm.load
// CHECK-NEXT: %[[MERGE_IN_RANGE:.*]] = arith.cmpi ult, %[[MERGE_ID]], %[[MERGE_COUNT]] : i32
// CHECK-NEXT: %[[PUBLISHED:.*]] = scf.if %[[MERGE_IN_RANGE]] -> (i32) {
// CHECK: %[[FIRST_COMPLETION:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[COMPLETION_INDEX:.*]] = arith.addi %[[FIRST_COMPLETION]], %{{.*}} : index
// CHECK-NEXT: %[[COMPLETION_INDEX64:.*]] = arith.index_cast %[[COMPLETION_INDEX]] : index to i64
// CHECK-NEXT: %[[COMPLETION:.*]] = llvm.getelementptr %[[COUNTERS]][%[[COMPLETION_INDEX64]]]
// CHECK: llvm.getelementptr %[[MERGES]][
// CHECK: llvm.getelementptr %[[MERGES]][
// CHECK: nvvm.memory.barrier <gpu>
// CHECK-NEXT: llvm.atomicrmw add %[[COMPLETION]], %[[ONE_TASK]] acq_rel
// CHECK: memref.store %[[PUBLISHED]], %[[SLOTS]][%[[C1]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[READY:.*]] = memref.load %[[SLOTS]][%[[C1]]]
// The block that completed a merge reduces its range of scratch into the
// segment of the merge record.
// CHECK: scf.if %{{.*}} {
// CHECK-NEXT: nvvm.memory.barrier <gpu>
// CHECK-NEXT: arith.index_cast %[[READY]] : i32 to index
// CHECK: llvm.getelementptr %[[MERGES]][
// CHECK-NEXT: %[[MERGE_SEGMENT:.*]] = llvm.load
// CHECK-NEXT: %[[MERGE_SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[MERGE_SEGMENT]], %[[SEGMENT_COUNT]] : i32
// CHECK: arith.minsi %{{.*}}, %[[PARTIAL_COUNT]] : i32
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[BLOCK]] iter_args
// CHECK: llvm.getelementptr %[[SCRATCH]][
// CHECK: %[[MERGE_TOTAL:.*]] = gpu.all_reduce add %{{.*}} uniform
// CHECK: arith.andi %[[LEADER]], %[[MERGE_SEGMENT_IN_RANGE]] : i1
// CHECK: %[[MERGE_SLOT:.*]] = llvm.getelementptr %[[OUTPUT]][
// CHECK-NEXT: llvm.store %[[MERGE_TOTAL]], %[[MERGE_SLOT]] : f32, !llvm.ptr

// The warp queue: counter 0, a batch per claim, and the claim reaches the
// lanes of the subgroup through a shuffle and no barrier.
// CHECK: %[[SUBGROUP:.*]] = arith.constant 32 : index
// CHECK-NEXT: %[[LANE:.*]] = arith.remui %[[THREAD]], %[[SUBGROUP]] : index
// CHECK-NEXT: %[[FIRST_LANE:.*]] = arith.cmpi eq, %[[LANE]], %[[C0]] : index
// CHECK-NEXT: %[[WARP_COUNTER_INDEX:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[WARP_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[WARP_COUNTER_INDEX]]]
// CHECK-NEXT: %[[WARP_LEADER_CLAIM:.*]] = scf.if %[[FIRST_LANE]] -> (i32) {
// CHECK-NEXT: llvm.atomicrmw add %[[WARP_COUNTER]], %[[WARP_BATCH]] monotonic
// CHECK: %[[FIRST_WARP:.*]], %{{.*}} = gpu.shuffle idx %[[WARP_LEADER_CLAIM]],
// CHECK-NEXT: scf.while (%[[WARP_CLAIM:.*]] = %[[FIRST_WARP]]) : (i32) -> i32 {
// CHECK-NEXT: arith.cmpi ult, %[[WARP_CLAIM]], %[[WARP_TASKS]] : i32
// CHECK-NOT: gpu.barrier
// CHECK: llvm.getelementptr %[[WARP_IDS]][
// CHECK-NEXT: %[[WARP_SEGMENT:.*]] = llvm.load
// CHECK-NEXT: arith.cmpi ult, %[[WARP_SEGMENT]], %[[SEGMENT_COUNT]] : i32
// CHECK-NOT: gpu.barrier
// CHECK: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[SUBGROUP]] iter_args
// CHECK-NOT: gpu.all_reduce
// CHECK: gpu.shuffle xor
// CHECK-NOT: gpu.barrier
// CHECK: gpu.return

module {
  func.func @drain(
      %counters: memref<?xi32>, %scratch: memref<?xf32>,
      %merges: memref<?xi32>, %merge_count: i32,
      %ranges: memref<?xi32>, %merge_ids: memref<?xi32>, %partial_count: i32,
      %cta_ids: memref<?xi32>, %cta_task_count: i32,
      %warp_ids: memref<?xi32>, %warp_task_count: i32,
      %values: memref<?xf32>, %offsets: memref<?xi32>, %value_count: i32,
      %segment_count: i32, %output: memref<?xf32>)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
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
      ^bb0(%partial: f32):
        swage.yield %partial : f32
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
