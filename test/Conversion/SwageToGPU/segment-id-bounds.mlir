// test/Conversion/SwageToGPU/segment-id-bounds.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=32 use-task-ids=true' %s \
// RUN:   | FileCheck %s --check-prefixes=TASKS,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128 use-task-ids=true' %s \
// RUN:   | FileCheck %s --check-prefixes=TASKS,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed=true' %s \
// RUN:   | FileCheck %s --check-prefixes=FUSED,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --check-prefixes=PERSISTENT,CHECK
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu='merge' %s \
// RUN:   | FileCheck %s --check-prefix=MERGE

// A kernel that loads an index from a buffer bounds it on the device before
// the index addresses anything. Host validation sees one snapshot of these
// buffers, and the kernel reloads them at every launch. segment-bounds.mlir
// pins the loaded ranges. This file pins the three loaded indices:
//
//   - a segment ID from a task buffer, in the task-ID, fused, and persistent
//     kernels, bounded by the segment count;
//   - the merge ID of a persistent partial, bounded by the merge count;
//   - the output segment of a merge record, in the split merge kernel and in
//     the persistent merge, bounded by the segment count.
//
// Each ABI passes the segment count as its last i32. Every bound is one
// unsigned comparison of the loaded i32 word, so a negative word fails it
// too. An out-of-range index is skipped, never clamped: it reads no element
// of values and stores nothing.
//
// A segment ID reaches three addresses. The two offsets reads take an index
// that is the ID when it is in range and zero otherwise, so an out-of-range
// ID reads offsets[0] twice and reduces an empty range. The output store is
// taken only when the ID is in range. No branch is added around the
// reduction, so every thread still reaches each barrier and shuffle.
//
// The direct kernel and the split partial kernel load no index: their segment
// and scratch slot are the block index, which they compare with a count.
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
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// TASKS: gpu.func @segmented_sum(%{{[^,]+}}: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[TASK_IDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %[[SEGMENT_COUNT:[^,)]+]]: i32) kernel
// FUSED: gpu.func @segmented_sum(%{{[^,]+}}: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[TASK_IDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %[[SEGMENT_COUNT:[^,)]+]]: i32) kernel
// The first task queue of the persistent kernel is the direct CTA queue.
// PERSISTENT: gpu.func @segmented_sum(
// PERSISTENT-SAME: %{{[^,]+}}: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr,
// PERSISTENT-SAME: %[[WARP_IDS:[^,]+]]: !llvm.ptr, %[[TASK_IDS:[^,]+]]: !llvm.ptr,
// PERSISTENT-SAME: %{{[^,]+}}: !llvm.ptr, %[[MERGE_IDS:[^,]+]]: !llvm.ptr, %[[MERGES:[^,]+]]: !llvm.ptr,
// PERSISTENT-SAME: %{{[^,]+}}: !llvm.ptr, %[[COUNTERS:[^,]+]]: !llvm.ptr,
// PERSISTENT-SAME: %{{[^,]+}}: i32, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %[[MERGE_COUNT:[^,]+]]: i32, %[[SEGMENT_COUNT:[^,)]+]]: i32)
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index

// The task word is compared before it is cast. Both offsets indices come
// from a select on that comparison, and the store predicate includes it.
// CHECK: %[[ID_ADDRESS:.*]] = llvm.getelementptr %[[TASK_IDS]]
// CHECK-NEXT: %[[ID_WORD:.*]] = llvm.load %[[ID_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[ID_IN_RANGE:.*]] = arith.cmpi ult, %[[ID_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT: %[[SEGMENT:.*]] = arith.index_cast %[[ID_WORD]] : i32 to index
// CHECK-NEXT: %[[SEGMENT_I64:.*]] = arith.index_cast %[[SEGMENT]] : index to i64
// CHECK-NEXT: %[[START_INDEX:.*]] = arith.select %[[ID_IN_RANGE]], %[[SEGMENT]], %[[ZERO]] : index
// CHECK-NEXT: %[[START_I64:.*]] = arith.index_cast %[[START_INDEX]] : index to i64
// CHECK-NEXT: %[[START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[START_I64]]]
// CHECK-NEXT: %{{.*}} = llvm.load %[[START_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[SEGMENT]], %[[ONE]] : index
// CHECK-NEXT: %[[END_INDEX:.*]] = arith.select %[[ID_IN_RANGE]], %[[NEXT]], %[[ZERO]] : index
// CHECK-NEXT: %[[END_I64:.*]] = arith.index_cast %[[END_INDEX]] : index to i64
// CHECK-NEXT: %[[END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[END_I64]]]
// CHECK-NEXT: %{{.*}} = llvm.load %[[END_ADDRESS]] : !llvm.ptr -> i32
// Nothing else indexes offsets or output for this task, and no branch opens
// before the reduction has finished.
// CHECK-NOT: llvm.getelementptr %[[OFFSETS]]
// CHECK-NOT: llvm.getelementptr %[[OUTPUT]]
// CHECK-NOT: scf.if
// CHECK: %[[WRITER:.*]] = arith.cmpi eq, %{{.*}}, %[[ZERO]] : index
// CHECK-NEXT: %[[MAY_STORE:.*]] = arith.andi %[[WRITER]], %[[ID_IN_RANGE]] : i1
// CHECK-NEXT: scf.if %[[MAY_STORE]] {
// CHECK-NEXT: %[[OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SEGMENT_I64]]]
// CHECK-NEXT: llvm.store %{{.*}}, %[[OUTPUT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }

// The task-ID kernel runs one task per block, so it has no other store.
// TASKS-NOT: llvm.store

// The fused kernel emits the segment body twice: the warp branch above and
// the CTA branch here. Both bound the task word the same way.
// FUSED: } else {
// FUSED: %[[CTA_ID_ADDRESS:.*]] = llvm.getelementptr %[[TASK_IDS]]
// FUSED-NEXT: %[[CTA_ID_WORD:.*]] = llvm.load %[[CTA_ID_ADDRESS]] : !llvm.ptr -> i32
// FUSED-NEXT: %[[CTA_ID_IN_RANGE:.*]] = arith.cmpi ult, %[[CTA_ID_WORD]], %[[SEGMENT_COUNT]] : i32
// FUSED-NEXT: %[[CTA_SEGMENT:.*]] = arith.index_cast %[[CTA_ID_WORD]] : i32 to index
// FUSED-NEXT: %[[CTA_SEGMENT_I64:.*]] = arith.index_cast %[[CTA_SEGMENT]] : index to i64
// FUSED-NEXT: %[[CTA_START_INDEX:.*]] = arith.select %[[CTA_ID_IN_RANGE]], %[[CTA_SEGMENT]], %[[ZERO]] : index
// FUSED-NEXT: %[[CTA_START_I64:.*]] = arith.index_cast %[[CTA_START_INDEX]] : index to i64
// FUSED-NEXT: %[[CTA_START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[CTA_START_I64]]]
// FUSED-NEXT: %{{.*}} = llvm.load %[[CTA_START_ADDRESS]] : !llvm.ptr -> i32
// FUSED-NEXT: %[[CTA_NEXT:.*]] = arith.addi %[[CTA_SEGMENT]], %[[ONE]] : index
// FUSED-NEXT: %[[CTA_END_INDEX:.*]] = arith.select %[[CTA_ID_IN_RANGE]], %[[CTA_NEXT]], %[[ZERO]] : index
// FUSED-NEXT: %[[CTA_END_I64:.*]] = arith.index_cast %[[CTA_END_INDEX]] : index to i64
// FUSED-NEXT: %[[CTA_END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[CTA_END_I64]]]
// FUSED-NEXT: %{{.*}} = llvm.load %[[CTA_END_ADDRESS]] : !llvm.ptr -> i32
// FUSED-NOT: llvm.getelementptr %[[OFFSETS]]
// FUSED-NOT: llvm.getelementptr %[[OUTPUT]]
// FUSED-NOT: scf.if
// FUSED: %[[CTA_WRITER:.*]] = arith.cmpi eq, %{{.*}}, %[[ZERO]] : index
// FUSED-NEXT: %[[CTA_MAY_STORE:.*]] = arith.andi %[[CTA_WRITER]], %[[CTA_ID_IN_RANGE]] : i1
// FUSED-NEXT: scf.if %[[CTA_MAY_STORE]] {
// FUSED-NEXT: %[[CTA_OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[CTA_SEGMENT_I64]]]
// FUSED-NEXT: llvm.store %{{.*}}, %[[CTA_OUTPUT_ADDRESS]] : f32, !llvm.ptr
// FUSED-NEXT: }
// FUSED-NOT: llvm.store

// Persistent merge ID. Only the block leader publishes a partial, so this
// branch is not around a barrier. The merge ID is compared with the merge
// count before it addresses the completion counter or a merge record. The
// counter update and both record reads sit inside the branch, and an
// out-of-range ID publishes "no merge ready" instead.
// PERSISTENT: %[[GROUP_ADDRESS:.*]] = llvm.getelementptr %[[MERGE_IDS]]
// PERSISTENT-NEXT: %[[GROUP_WORD:.*]] = llvm.load %[[GROUP_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT-NEXT: %[[GROUP_IN_RANGE:.*]] = arith.cmpi ult, %[[GROUP_WORD]], %[[MERGE_COUNT]] : i32
// PERSISTENT-NEXT: %[[PUBLISHED:.*]] = scf.if %[[GROUP_IN_RANGE]] -> (i32) {
// PERSISTENT-NEXT: %[[GROUP:.*]] = arith.index_cast %[[GROUP_WORD]] : i32 to index
// PERSISTENT-NEXT: %[[COMPLETION_BASE:.*]] = arith.constant 3 : index
// PERSISTENT-NEXT: %[[COMPLETION_INDEX:.*]] = arith.addi %[[COMPLETION_BASE]], %[[GROUP]] : index
// PERSISTENT-NEXT: %[[COMPLETION_I64:.*]] = arith.index_cast %[[COMPLETION_INDEX]] : index to i64
// PERSISTENT-NEXT: %[[COMPLETION:.*]] = llvm.getelementptr %[[COUNTERS]][%[[COMPLETION_I64]]]
// PERSISTENT: %[[GROUP_RECORD:.*]] = arith.muli %[[GROUP]], %{{.*}} : index
// PERSISTENT: llvm.getelementptr %[[MERGES]]
// PERSISTENT: llvm.getelementptr %[[MERGES]]
// PERSISTENT: llvm.atomicrmw add %[[COMPLETION]], %{{.*}} acq_rel
// PERSISTENT: %[[READY:.*]] = scf.if %{{.*}} -> (i32) {
// PERSISTENT: scf.yield %[[READY]] : i32
// PERSISTENT-NEXT: } else {
// PERSISTENT-NEXT: %[[SKIPPED:.*]] = arith.constant -1 : i32
// PERSISTENT-NEXT: scf.yield %[[SKIPPED]] : i32
// PERSISTENT-NEXT: }
// PERSISTENT-NEXT: memref.store %[[PUBLISHED]], %{{.*}}

// Persistent merge output. The first field of the ready merge's record names
// the output segment. It is compared when loaded, and the leader's store
// takes the comparison as part of its predicate. The merge itself runs
// unconditionally, so its all-reduce stays under the block-uniform guard.
// PERSISTENT: %[[MERGE_SEGMENT_ADDRESS:.*]] = llvm.getelementptr %[[MERGES]]
// PERSISTENT-NEXT: %[[MERGE_SEGMENT_WORD:.*]] = llvm.load %[[MERGE_SEGMENT_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT-NEXT: %[[MERGE_SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[MERGE_SEGMENT_WORD]], %[[SEGMENT_COUNT]] : i32
// PERSISTENT-NEXT: %[[MERGE_SEGMENT:.*]] = arith.index_cast %[[MERGE_SEGMENT_WORD]] : i32 to index
// PERSISTENT-NOT: llvm.getelementptr %[[OUTPUT]]
// PERSISTENT-NOT: scf.if
// PERSISTENT: gpu.all_reduce add
// PERSISTENT: %[[MERGE_MAY_STORE:.*]] = arith.andi %{{.*}}, %[[MERGE_SEGMENT_IN_RANGE]] : i1
// PERSISTENT-NEXT: scf.if %[[MERGE_MAY_STORE]] {
// PERSISTENT-NEXT: %[[MERGE_SEGMENT_I64:.*]] = arith.index_cast %[[MERGE_SEGMENT]] : index to i64
// PERSISTENT-NEXT: %[[MERGE_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[MERGE_SEGMENT_I64]]]
// PERSISTENT-NEXT: llvm.store %{{.*}}, %[[MERGE_OUTPUT]] : f32, !llvm.ptr
// PERSISTENT-NEXT: }

// Persistent warp queue: the same bound as the direct CTA queue, behind the
// warp task indirection.
// PERSISTENT: %[[WARP_ID_ADDRESS:.*]] = llvm.getelementptr %[[WARP_IDS]]
// PERSISTENT-NEXT: %[[WARP_ID_WORD:.*]] = llvm.load %[[WARP_ID_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT-NEXT: %[[WARP_ID_IN_RANGE:.*]] = arith.cmpi ult, %[[WARP_ID_WORD]], %[[SEGMENT_COUNT]] : i32
// PERSISTENT-NEXT: %[[WARP_SEGMENT:.*]] = arith.index_cast %[[WARP_ID_WORD]] : i32 to index
// PERSISTENT-NEXT: %[[WARP_SEGMENT_I64:.*]] = arith.index_cast %[[WARP_SEGMENT]] : index to i64
// PERSISTENT-NEXT: %[[WARP_START_INDEX:.*]] = arith.select %[[WARP_ID_IN_RANGE]], %[[WARP_SEGMENT]], %[[ZERO]] : index
// PERSISTENT-NEXT: %[[WARP_START_I64:.*]] = arith.index_cast %[[WARP_START_INDEX]] : index to i64
// PERSISTENT-NEXT: %[[WARP_START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[WARP_START_I64]]]
// PERSISTENT-NEXT: %{{.*}} = llvm.load %[[WARP_START_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT-NEXT: %[[WARP_NEXT:.*]] = arith.addi %[[WARP_SEGMENT]], %[[ONE]] : index
// PERSISTENT-NEXT: %[[WARP_END_INDEX:.*]] = arith.select %[[WARP_ID_IN_RANGE]], %[[WARP_NEXT]], %[[ZERO]] : index
// PERSISTENT-NEXT: %[[WARP_END_I64:.*]] = arith.index_cast %[[WARP_END_INDEX]] : index to i64
// PERSISTENT-NEXT: %[[WARP_END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[WARP_END_I64]]]
// PERSISTENT-NEXT: %{{.*}} = llvm.load %[[WARP_END_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT-NOT: llvm.getelementptr %[[OFFSETS]]
// PERSISTENT-NOT: llvm.getelementptr %[[OUTPUT]]
// PERSISTENT-NOT: scf.if
// PERSISTENT: %[[WARP_WRITER:.*]] = arith.cmpi eq, %{{.*}}, %[[ZERO]] : index
// PERSISTENT-NEXT: %[[WARP_MAY_STORE:.*]] = arith.andi %[[WARP_WRITER]], %[[WARP_ID_IN_RANGE]] : i1
// PERSISTENT-NEXT: scf.if %[[WARP_MAY_STORE]] {
// PERSISTENT-NEXT: %[[WARP_OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[WARP_SEGMENT_I64]]]
// PERSISTENT-NEXT: llvm.store %{{.*}}, %[[WARP_OUTPUT_ADDRESS]] : f32, !llvm.ptr
// PERSISTENT-NEXT: }
// PERSISTENT-NOT: llvm.store

// Split merge. The output segment is the first field of the block's merge
// record. It takes the same bound as in the persistent merge.
// MERGE: gpu.func @segmented_sum__merge(%{{[^,]+}}: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[RECORDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %[[SEGMENT_COUNT:[^,)]+]]: i32) kernel
// MERGE: %[[SEGMENT_ADDRESS:.*]] = llvm.getelementptr %[[RECORDS]]
// MERGE-NEXT: %[[SEGMENT_WORD:.*]] = llvm.load %[[SEGMENT_ADDRESS]] : !llvm.ptr -> i32
// MERGE-NEXT: %[[SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[SEGMENT_WORD]], %[[SEGMENT_COUNT]] : i32
// MERGE-NEXT: %[[SEGMENT:.*]] = arith.index_cast %[[SEGMENT_WORD]] : i32 to index
// MERGE-NOT: llvm.getelementptr %[[OUTPUT]]
// MERGE-NOT: scf.if
// MERGE: gpu.all_reduce add
// MERGE: %[[WRITER:.*]] = arith.cmpi eq, %{{.*}} : index
// MERGE-NEXT: %[[MAY_STORE:.*]] = arith.andi %[[WRITER]], %[[SEGMENT_IN_RANGE]] : i1
// MERGE-NEXT: scf.if %[[MAY_STORE]] {
// MERGE-NEXT: %[[SEGMENT_I64:.*]] = arith.index_cast %[[SEGMENT]] : index to i64
// MERGE-NEXT: %[[OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SEGMENT_I64]]]
// MERGE-NEXT: llvm.store %{{.*}}, %[[OUTPUT_ADDRESS]] : f32, !llvm.ptr
// MERGE-NEXT: }
// MERGE-NOT: llvm.store
