// test/Conversion/SwageToGPU/persistent.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --check-prefix=PERWARP
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | swage-opt | FileCheck %s --implicit-check-not=swage.
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=128 persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BLOCK-SIZE
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=128 persistent fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BLOCK-SIZE
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FUSED

// The experimental persistent kernel drains three device queues in one
// launch: direct CTA tasks, split partials with their merges, then direct
// warp tasks. ADR-0018 records two races this kernel had. Both fixes are
// orderings, which an operation count cannot see, so this test pins them:
//
//   - every block barrier is reached by all threads of the block: it follows
//     the closing brace of a leader-only branch instead of sitting inside it,
//     and the loops and branches around it are driven by values every thread
//     loaded from the shared claim slot after a barrier;
//   - a block barrier separates the direct CTA queue from the partial queue,
//     because both broadcast their claims through the same shared slot;
//   - a partial writes scratch, then fences, then publishes completion, and
//     the merging block fences before it reads scratch;
//   - the warp queue uses shuffles and no block-wide synchronization.
//
// The third RUN line reads the lowered kernel back through the driver, so the
// fences stay printable and parseable there.
module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK: gpu.module @segmented_sum_module
// CHECK: gpu.func @segmented_sum(
// CHECK-SAME: %[[VALUES:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[WARP_IDS:[^,]+]]: !llvm.ptr, %[[CTA_IDS:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[RANGES:[^,]+]]: !llvm.ptr, %[[MERGE_IDS:[^,]+]]: !llvm.ptr, %[[MERGES:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[SCRATCH:[^,]+]]: !llvm.ptr, %[[COUNTERS:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %{{[^,]+}}: i32, %[[WARP_COUNT:[^,]+]]: i32, %[[CTA_COUNT:[^,]+]]: i32, %[[PARTIAL_COUNT:[^,]+]]: i32, %{{[^)]+}}: i32)
// CHECK-SAME: workgroup(%[[SHARED:[^ ]+]] : memref<2xi32, #gpu.address_space<workgroup>>) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 512, 1, 1>
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[BLOCK:.*]] = arith.constant 512 : index
// CHECK-DAG: %[[ZERO_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE_I32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[FOUR_I32:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[EIGHT_I32:.*]] = arith.constant 8 : i32
// CHECK: %[[LEADER:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index

// Direct CTA queue, first claim. Only the leader claims. The claim reaches
// the other threads through shared slot 0: leader store, block barrier
// outside the leader branch, then a load by every thread.
// CHECK-NEXT: %[[CTA_QUEUE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[CTA_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[CTA_QUEUE]]]
// CHECK-NEXT: %[[CTA_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:   %[[CTA_CLAIMED:.*]] = llvm.atomicrmw add %[[CTA_COUNTER]], %[[ONE_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[CTA_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   memref.store %[[CTA_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[FIRST_CTA:.*]] = memref.load %[[SHARED]][%[[ZERO]]]

// The loop runs while the broadcast claim names a task, so every thread of
// the block agrees on each trip.
// CHECK-NEXT: %{{.*}} = scf.while (%[[CTA_HEAD:.*]] = %[[FIRST_CTA]]) : (i32) -> i32 {
// CHECK-NEXT:   %[[HAS_CTA:.*]] = arith.cmpi ult, %[[CTA_HEAD]], %[[CTA_COUNT]] : i32
// CHECK-NEXT:   scf.condition(%[[HAS_CTA]]) %[[CTA_HEAD]] : i32
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%[[CTA_TASK:.*]]: i32):
// CHECK-NEXT:   %[[CTA_TASK_INDEX:.*]] = arith.index_cast %[[CTA_TASK]] : i32 to index
// CHECK-NEXT:   %[[CTA_TASK_I64:.*]] = arith.index_cast %[[CTA_TASK_INDEX]] : index to i64
// CHECK-NEXT:   llvm.getelementptr %[[CTA_IDS]][%[[CTA_TASK_I64]]]
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK:     %[[CTA_LOCAL:.*]] = scf.for %{{.*}} step %[[BLOCK]] iter_args(
// CHECK:       llvm.getelementptr %[[VALUES]]
// CHECK:     }
// CHECK-NEXT: %[[CTA_TOTAL:.*]] = gpu.all_reduce add %[[CTA_LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32
// CHECK-NEXT: %[[CTA_WRITER:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index
// CHECK-NEXT: scf.if %[[CTA_WRITER]] {
// CHECK-NEXT:   %[[CTA_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]]
// CHECK-NEXT:   llvm.store %[[CTA_TOTAL]], %[[CTA_OUTPUT]] : f32, !llvm.ptr
// CHECK-NEXT: }
// The block converges again before the next claim reuses shared slot 0.
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[NEXT_CTA_QUEUE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[NEXT_CTA_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[NEXT_CTA_QUEUE]]]
// CHECK-NEXT: %[[NEXT_CTA_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:   %[[NEXT_CTA_CLAIMED:.*]] = llvm.atomicrmw add %[[NEXT_CTA_COUNTER]], %[[ONE_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[NEXT_CTA_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   memref.store %[[NEXT_CTA_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[NEXT_CTA:.*]] = memref.load %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: scf.yield %[[NEXT_CTA]] : i32
// CHECK-NEXT: }

// Phase handoff. The partial queue broadcasts through the same shared slot,
// so the block converges once more after the CTA loop. Without this barrier
// the leader's first partial claim could overwrite the slot before another
// thread had loaded the terminating CTA claim.
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[PARTIAL_QUEUE:.*]] = arith.constant 2 : i64
// CHECK-NEXT: %[[PARTIAL_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[PARTIAL_QUEUE]]]
// CHECK-NEXT: %[[PARTIAL_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:   %[[PARTIAL_CLAIMED:.*]] = llvm.atomicrmw add %[[PARTIAL_COUNTER]], %[[FOUR_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[PARTIAL_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   memref.store %[[PARTIAL_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[FIRST_PARTIAL:.*]] = memref.load %[[SHARED]][%[[ZERO]]]

// Partial queue. The outer loop and the batch bounds come from the broadcast
// claim and a launch argument, so both are block-uniform.
// CHECK-NEXT: %{{.*}} = scf.while (%[[PARTIAL_HEAD:.*]] = %[[FIRST_PARTIAL]]) : (i32) -> i32 {
// CHECK-NEXT:   %[[HAS_PARTIAL:.*]] = arith.cmpi ult, %[[PARTIAL_HEAD]], %[[PARTIAL_COUNT]] : i32
// CHECK-NEXT:   scf.condition(%[[HAS_PARTIAL]]) %[[PARTIAL_HEAD]] : i32
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%[[BATCH:.*]]: i32):
// CHECK-NEXT:   %[[BATCH_END:.*]] = arith.addi %[[BATCH]], %[[FOUR_I32]] : i32
// CHECK-NEXT:   %[[BOUNDED_END:.*]] = arith.minui %[[BATCH_END]], %[[PARTIAL_COUNT]] : i32
// CHECK-NEXT:   %[[BATCH_FIRST:.*]] = arith.index_cast %[[BATCH]] : i32 to index
// CHECK-NEXT:   %[[BATCH_LAST:.*]] = arith.index_cast %[[BOUNDED_END]] : i32 to index
// CHECK-NEXT:   scf.for %[[PARTIAL:.*]] = %[[BATCH_FIRST]] to %[[BATCH_LAST]] step %[[ONE]] {
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK:       llvm.getelementptr %[[RANGES]]
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK:       %[[PARTIAL_LOCAL:.*]] = scf.for %{{.*}} step %[[BLOCK]] iter_args(
// CHECK:         llvm.getelementptr %[[VALUES]]
// CHECK:       }
// CHECK-NEXT:  %[[PARTIAL_TOTAL:.*]] = gpu.all_reduce add %[[PARTIAL_LOCAL]] uniform {
// CHECK-NEXT:  } : (f32) -> f32

// Publication, writer side. The leader stores the partial into its own
// scratch slot, the block converges, and only then does the leader fence and
// publish completion. The fence directly precedes the completion atomic, and
// no atomic sits between the scratch store and the fence. A publisher that
// counted first could let another block observe the group complete and read
// scratch before this store became visible to it.
// CHECK-NEXT:  scf.if %[[LEADER]] {
// CHECK-NEXT:    %[[SLOT:.*]] = arith.index_cast %[[PARTIAL]] : index to i64
// CHECK-NEXT:    %[[SLOT_ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[SLOT]]]
// CHECK-NEXT:    llvm.store %[[PARTIAL_TOTAL]], %[[SLOT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT:  }
// CHECK-NEXT:  gpu.barrier
// CHECK-NEXT:  scf.if %[[LEADER]] {
// CHECK-NEXT:    %[[GROUP_SLOT:.*]] = arith.index_cast %[[PARTIAL]] : index to i64
// CHECK-NEXT:    %[[GROUP_ADDRESS:.*]] = llvm.getelementptr %[[MERGE_IDS]][%[[GROUP_SLOT]]]
// CHECK-NEXT:    %[[GROUP_I32:.*]] = llvm.load %[[GROUP_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT:    %[[GROUP:.*]] = arith.index_cast %[[GROUP_I32]] : i32 to index
// CHECK:         %[[COMPLETION_INDEX:.*]] = arith.addi %{{.*}}, %[[GROUP]] : index
// CHECK-NEXT:    %[[COMPLETION_I64:.*]] = arith.index_cast %[[COMPLETION_INDEX]] : index to i64
// CHECK-NEXT:    %[[COMPLETION:.*]] = llvm.getelementptr %[[COUNTERS]][%[[COMPLETION_I64]]]
// CHECK-NOT: llvm.atomicrmw
// CHECK-NOT: nvvm.memory.barrier
// CHECK-NOT: gpu.barrier
// CHECK-NOT: scf.if
// CHECK:         %[[EXPECTED:.*]] = arith.index_cast %{{.*}} : index to i32
// CHECK-NEXT:    nvvm.memory.barrier <gpu>
// CHECK-NEXT:    %[[PREVIOUS:.*]] = llvm.atomicrmw add %[[COMPLETION]], %[[ONE_I32]] acq_rel : !llvm.ptr, i32
// CHECK-NEXT:    %[[COMPLETED:.*]] = arith.addi %[[PREVIOUS]], %[[ONE_I32]] : i32
// CHECK-NEXT:    %[[IS_LAST:.*]] = arith.cmpi eq, %[[COMPLETED]], %[[EXPECTED]] : i32
// The leader broadcasts the ready merge, or -1, through shared slot 1.
// CHECK-NEXT:    %[[READY:.*]] = scf.if %[[IS_LAST]] -> (i32) {
// CHECK-NEXT:      %[[READY_ID:.*]] = arith.index_cast %[[GROUP]] : index to i32
// CHECK-NEXT:      scf.yield %[[READY_ID]] : i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      %[[NOT_READY:.*]] = arith.constant -1 : i32
// CHECK-NEXT:      scf.yield %[[NOT_READY]] : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    memref.store %[[READY]], %[[SHARED]][%[[ONE]]]
// CHECK-NEXT:  }
// CHECK-NEXT:  gpu.barrier
// CHECK-NEXT:  %[[READY_MERGE:.*]] = memref.load %[[SHARED]][%[[ONE]]]

// Publication, reader side. The merge guard is the value every thread loaded
// after the barrier, so the all-reduce under it is block-uniform. The fence
// is the first operation of the merge, ahead of every scratch load.
// CHECK-NEXT:  %[[HAS_MERGE:.*]] = arith.cmpi sge, %[[READY_MERGE]], %[[ZERO_I32]] : i32
// CHECK-NEXT:  scf.if %[[HAS_MERGE]] {
// CHECK-NEXT:    nvvm.memory.barrier <gpu>
// CHECK-NEXT:    %[[MERGE:.*]] = arith.index_cast %[[READY_MERGE]] : i32 to index
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK:         llvm.getelementptr %[[MERGES]]
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK:         %[[MERGE_LOCAL:.*]] = scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %[[BLOCK]] iter_args(%[[MERGE_ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK-NEXT:      %[[J_I64:.*]] = arith.index_cast %[[J]] : index to i64
// CHECK-NEXT:      %[[PUBLISHED_ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[J_I64]]]
// CHECK-NEXT:      %[[PUBLISHED:.*]] = llvm.load %[[PUBLISHED_ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT:      %[[MERGE_NEXT:.*]] = arith.addf %[[MERGE_ACC]], %[[PUBLISHED]] : f32
// CHECK-NEXT:      scf.yield %[[MERGE_NEXT]] : f32
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[MERGE_TOTAL:.*]] = gpu.all_reduce add %[[MERGE_LOCAL]] uniform {
// CHECK-NEXT:    } : (f32) -> f32
// CHECK-NEXT:    scf.if %[[LEADER]] {
// CHECK-NEXT:      %[[SEGMENT_I64:.*]] = arith.index_cast %{{.*}} : index to i64
// CHECK-NEXT:      %[[MERGE_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SEGMENT_I64]]]
// CHECK-NEXT:      llvm.store %[[MERGE_TOTAL]], %[[MERGE_OUTPUT]] : f32, !llvm.ptr
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  }

// Next partial claim, again through shared slot 0 behind a barrier.
// CHECK-NEXT:  %[[NEXT_PARTIAL_QUEUE:.*]] = arith.constant 2 : i64
// CHECK-NEXT:  %[[NEXT_PARTIAL_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[NEXT_PARTIAL_QUEUE]]]
// CHECK-NEXT:  %[[NEXT_PARTIAL_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:    %[[NEXT_PARTIAL_CLAIMED:.*]] = llvm.atomicrmw add %[[NEXT_PARTIAL_COUNTER]], %[[FOUR_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:    scf.yield %[[NEXT_PARTIAL_CLAIMED]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  scf.if %[[LEADER]] {
// CHECK-NEXT:    memref.store %[[NEXT_PARTIAL_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT:  }
// CHECK-NEXT:  gpu.barrier
// CHECK-NEXT:  %[[NEXT_PARTIAL:.*]] = memref.load %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT:  scf.yield %[[NEXT_PARTIAL]] : i32
// CHECK-NEXT: }

// Warp queue. Each warp claims on its own: lane zero performs the atomic and
// a shuffle broadcasts the claim inside the warp. Warps of one block diverge
// from here on.
// CHECK-NEXT: %[[WARP:.*]] = arith.constant 32 : index
// CHECK-NEXT: %[[LANE:.*]] = arith.remui %[[THREAD]], %[[WARP]] : index
// CHECK-NEXT: %[[LANE_LEADER:.*]] = arith.cmpi eq, %[[LANE]], %[[ZERO]] : index
// CHECK-NEXT: %[[WARP_QUEUE:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[WARP_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[WARP_QUEUE]]]
// CHECK-NEXT: %[[WARP_CLAIM:.*]] = scf.if %[[LANE_LEADER]] -> (i32) {
// CHECK-NEXT:   %[[WARP_CLAIMED:.*]] = llvm.atomicrmw add %[[WARP_COUNTER]], %[[EIGHT_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[WARP_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: %[[WIDTH:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[SOURCE_LANE:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[FIRST_WARP:[^,]+]], %{{.*}} = gpu.shuffle idx %[[WARP_CLAIM]], %[[SOURCE_LANE]], %[[WIDTH]] : i32
// CHECK-NEXT: %{{.*}} = scf.while (%[[WARP_HEAD:.*]] = %[[FIRST_WARP]]) : (i32) -> i32 {
// CHECK-NEXT:   %[[HAS_WARP:.*]] = arith.cmpi ult, %[[WARP_HEAD]], %[[WARP_COUNT]] : i32
// CHECK-NEXT:   scf.condition(%[[HAS_WARP]]) %[[WARP_HEAD]] : i32
// CHECK-NEXT: } do {
// CHECK:       llvm.getelementptr %[[WARP_IDS]]
// CHECK:       %[[WARP_LOCAL:.*]] = scf.for %{{.*}} step %[[WARP]] iter_args(
// CHECK:         llvm.getelementptr %[[VALUES]]
// CHECK:       }
// CHECK:       %[[S1:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_LOCAL]],
// CHECK-NEXT:  %[[T1:.*]] = arith.addf %[[WARP_LOCAL]], %[[S1]] : f32
// CHECK:       %[[S2:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T1]],
// CHECK-NEXT:  %[[T2:.*]] = arith.addf %[[T1]], %[[S2]] : f32
// CHECK:       %[[S3:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T2]],
// CHECK-NEXT:  %[[T3:.*]] = arith.addf %[[T2]], %[[S3]] : f32
// CHECK:       %[[S4:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T3]],
// CHECK-NEXT:  %[[T4:.*]] = arith.addf %[[T3]], %[[S4]] : f32
// CHECK:       %[[S5:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T4]],
// CHECK-NEXT:  %[[WARP_TOTAL:.*]] = arith.addf %[[T4]], %[[S5]] : f32
// CHECK-NEXT:  %[[WARP_WRITER:.*]] = arith.cmpi eq, %[[LANE]], %[[ZERO]] : index
// CHECK-NEXT:  scf.if %[[WARP_WRITER]] {
// CHECK-NEXT:    %[[WARP_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]]
// CHECK-NEXT:    llvm.store %[[WARP_TOTAL]], %[[WARP_OUTPUT]] : f32, !llvm.ptr
// CHECK-NEXT:  }
// CHECK-NEXT:  }
// CHECK-NEXT:  %[[NEXT_WARP_QUEUE:.*]] = arith.constant 0 : i64
// CHECK-NEXT:  %[[NEXT_WARP_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[NEXT_WARP_QUEUE]]]
// CHECK-NEXT:  %[[NEXT_WARP_CLAIM:.*]] = scf.if %[[LANE_LEADER]] -> (i32) {
// CHECK-NEXT:    %[[NEXT_WARP_CLAIMED:.*]] = llvm.atomicrmw add %[[NEXT_WARP_COUNTER]], %[[EIGHT_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:    scf.yield %[[NEXT_WARP_CLAIMED]] : i32
// CHECK-NEXT:  } else {
// CHECK-NEXT:    scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT:  }
// CHECK:       %[[NEXT_WARP:[^,]+]], %{{.*}} = gpu.shuffle idx %[[NEXT_WARP_CLAIM]],
// CHECK-NEXT:  scf.yield %[[NEXT_WARP]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// Once the warps diverge, nothing may require the whole block: no barrier,
// no all-reduce, no fence, and no use of the shared claim slots.
// PERWARP: arith.remui
// PERWARP-NOT: gpu.barrier
// PERWARP-NOT: gpu.all_reduce
// PERWARP-NOT: nvvm.memory.barrier
// PERWARP-NOT: memref.load
// PERWARP-NOT: memref.store
// PERWARP: gpu.return

// The persistent kernel is specialized to 512 threads, and the block-size
// requirements keep it from being combined with the fused mixed schedule.
// BLOCK-SIZE: error: persistent lowering requires block-size 512
// FUSED: error: fused mixed lowering requires block-size 128
