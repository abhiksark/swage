// test/Conversion/SwageToGPU/persistent.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --check-prefix=PERWARP
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --check-prefix=RANGE
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
//   - the warp queue uses shuffles and no block-wide synchronization;
//   - the bounds on loaded indices add no branch around a barrier or a
//     shuffle: a segment ID and an output segment are bounded through
//     selects and the store predicate, and the one new branch, on the merge
//     ID, sits inside the leader-only publication.
//
// The data that moves through that structure is pinned as well. The first
// run matches the kernel in order with captured operands: task word, range,
// bounded range, loop bounds, combine, block total, stored slot. For the two
// merge-record ranges it takes the bounds from the index casts above their
// use and leaves the cast operands open; the RANGE run follows those ranges
// forward from the record instead: each loaded word must be used by an index
// cast or an integer min or max, and that result must feed its use or another
// min or max. Together the two runs hold a merge range to its use whether or
// not a bound sits between them, and reject a bound taken from anywhere else.
//
// The fourth RUN line reads the lowered kernel back through the driver, so
// the fences stay printable and parseable there.
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

// CHECK: gpu.module @segmented_sum_module
// CHECK: gpu.func @segmented_sum(
// CHECK-SAME: %[[VALUES:[^,]+]]: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[WARP_IDS:[^,]+]]: !llvm.ptr, %[[CTA_IDS:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[RANGES:[^,]+]]: !llvm.ptr, %[[MERGE_IDS:[^,]+]]: !llvm.ptr, %[[MERGES:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[SCRATCH:[^,]+]]: !llvm.ptr, %[[COUNTERS:[^,]+]]: !llvm.ptr,
// CHECK-SAME: %[[VALUE_COUNT:[^,]+]]: i32, %[[WARP_COUNT:[^,]+]]: i32, %[[CTA_COUNT:[^,]+]]: i32, %[[PARTIAL_COUNT:[^,]+]]: i32, %[[MERGE_COUNT:[^,]+]]: i32, %[[SEGMENT_COUNT:[^)]+]]: i32)
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
// CHECK-NEXT: %[[FIRST_CTA_QUEUE:.*]] = arith.constant 1 : i64
// CHECK-NEXT: %[[FIRST_CTA_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[FIRST_CTA_QUEUE]]]
// CHECK-NEXT: %[[FIRST_CTA_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:   %[[FIRST_CTA_CLAIMED:.*]] = llvm.atomicrmw add %[[FIRST_CTA_COUNTER]], %[[ONE_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[FIRST_CTA_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   memref.store %[[FIRST_CTA_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[FIRST_CTA:.*]] = memref.load %[[SHARED]][%[[ZERO]]]

// The loop runs while the broadcast claim names a task, so every thread of
// the block agrees on each trip.
// CHECK-NEXT: %{{.*}} = scf.while (%[[CTA_HEAD:.*]] = %[[FIRST_CTA]]) : (i32) -> i32 {
// CHECK-NEXT:   %[[HAS_CTA:.*]] = arith.cmpi ult, %[[CTA_HEAD]], %[[CTA_COUNT]] : i32
// CHECK-NEXT:   scf.condition(%[[HAS_CTA]]) %[[CTA_HEAD]] : i32
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%[[CTA_CLAIMED_TASK:.*]]: i32):
// CHECK-NEXT:   %[[CTA_TASK:.*]] = arith.index_cast %[[CTA_CLAIMED_TASK]] : i32 to index
// CHECK-NEXT:   %[[CTA_TASK_I64:.*]] = arith.index_cast %[[CTA_TASK]] : index to i64

// The task word names the segment; its two offsets are the segment's range.
// The word is compared with the segment count as loaded. An out-of-range
// word selects offsets[0] for both ends, which is an empty range, so the
// bound opens no branch around the reduction.
// CHECK-NEXT: %[[CTA_ID_ADDRESS:.*]] = llvm.getelementptr %[[CTA_IDS]][%[[CTA_TASK_I64]]]
// CHECK-NEXT: %[[CTA_ID_WORD:.*]] = llvm.load %[[CTA_ID_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[CTA_ID_IN_RANGE:.*]] = arith.cmpi ult, %[[CTA_ID_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT: %[[CTA_SEGMENT:.*]] = arith.index_cast %[[CTA_ID_WORD]] : i32 to index
// CHECK-NEXT: %[[CTA_SEGMENT_I64:.*]] = arith.index_cast %[[CTA_SEGMENT]] : index to i64
// CHECK-NEXT: %[[CTA_START_INDEX:.*]] = arith.select %[[CTA_ID_IN_RANGE]], %[[CTA_SEGMENT]], %[[ZERO]] : index
// CHECK-NEXT: %[[CTA_START_INDEX_I64:.*]] = arith.index_cast %[[CTA_START_INDEX]] : index to i64
// CHECK-NEXT: %[[CTA_START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[CTA_START_INDEX_I64]]]
// CHECK-NEXT: %[[CTA_START_WORD:.*]] = llvm.load %[[CTA_START_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[CTA_NEXT_SEGMENT:.*]] = arith.addi %[[CTA_SEGMENT]], %[[ONE]] : index
// CHECK-NEXT: %[[CTA_END_INDEX:.*]] = arith.select %[[CTA_ID_IN_RANGE]], %[[CTA_NEXT_SEGMENT]], %[[ZERO]] : index
// CHECK-NEXT: %[[CTA_END_INDEX_I64:.*]] = arith.index_cast %[[CTA_END_INDEX]] : index to i64
// CHECK-NEXT: %[[CTA_END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[CTA_END_INDEX_I64]]]
// CHECK-NEXT: %[[CTA_END_WORD:.*]] = llvm.load %[[CTA_END_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[CTA_FLOOR:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[CTA_START_FLOORED:.*]] = arith.maxsi %[[CTA_START_WORD]], %[[CTA_FLOOR]] : i32
// CHECK-NEXT: %[[CTA_START_BOUND:.*]] = arith.minsi %[[CTA_START_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[CTA_END_FLOORED:.*]] = arith.maxsi %[[CTA_END_WORD]], %[[CTA_START_BOUND]] : i32
// CHECK-NEXT: %[[CTA_END_BOUND:.*]] = arith.minsi %[[CTA_END_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[CTA_START:.*]] = arith.index_cast %[[CTA_START_BOUND]] : i32 to index
// CHECK-NEXT: %[[CTA_END:.*]] = arith.index_cast %[[CTA_END_BOUND]] : i32 to index
// Each thread starts at its own offset into the range and strides by the
// number of threads, so the range is covered exactly once.
// CHECK-NEXT: %[[CTA_FIRST:.*]] = arith.addi %[[CTA_START]], %[[THREAD]] : index
// CHECK-NEXT: %[[CTA_IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[CTA_LOCAL:.*]] = scf.for %[[CTA_I:.*]] = %[[CTA_FIRST]] to %[[CTA_END]] step %[[BLOCK]] iter_args(%[[CTA_ACC:.*]] = %[[CTA_IDENTITY]]) -> (f32) {
// CHECK-NEXT:   %[[CTA_INDEX:.*]] = arith.index_cast %[[CTA_I]] : index to i64
// CHECK-NEXT:   %[[CTA_ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[CTA_INDEX]]]
// CHECK-NEXT:   %[[CTA_VALUE:.*]] = llvm.load %[[CTA_ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT:   %[[CTA_SUM:.*]] = arith.addf %[[CTA_ACC]], %[[CTA_VALUE]] : f32
// CHECK-NEXT:   scf.yield %[[CTA_SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: %[[CTA_TOTAL:.*]] = gpu.all_reduce add %[[CTA_LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32
// Thread zero of the block is the only writer, and it stores the total at the
// segment the task word named, unless that segment is out of range.
// CHECK-NEXT: %[[CTA_WRITER:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index
// CHECK-NEXT: %[[CTA_MAY_STORE:.*]] = arith.andi %[[CTA_WRITER]], %[[CTA_ID_IN_RANGE]] : i1
// CHECK-NEXT: scf.if %[[CTA_MAY_STORE]] {
// CHECK-NEXT:   %[[CTA_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[CTA_SEGMENT_I64]]]
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
// CHECK-NEXT: %[[FIRST_PARTIAL_QUEUE:.*]] = arith.constant 2 : i64
// CHECK-NEXT: %[[FIRST_PARTIAL_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[FIRST_PARTIAL_QUEUE]]]
// CHECK-NEXT: %[[FIRST_PARTIAL_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:   %[[FIRST_PARTIAL_CLAIMED:.*]] = llvm.atomicrmw add %[[FIRST_PARTIAL_COUNTER]], %[[FOUR_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[FIRST_PARTIAL_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   memref.store %[[FIRST_PARTIAL_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[FIRST_PARTIAL:.*]] = memref.load %[[SHARED]][%[[ZERO]]]

// Partial queue. A claim covers up to four consecutive partials. The outer
// loop and the batch bounds come from the broadcast claim and a launch
// argument, so both are block-uniform.
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

// Partial p reduces the [begin, end) pair at 2 * p in the planned ranges. The
// range indexes values, so both words are bounded by the value count, and the
// end is floored by the bounded begin.
// CHECK-NEXT: %[[RANGE_FIELDS:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[RANGE_RECORD:.*]] = arith.muli %[[PARTIAL]], %[[RANGE_FIELDS]] : index
// CHECK-NEXT: %[[RANGE_END_INDEX:.*]] = arith.addi %[[RANGE_RECORD]], %[[ONE]] : index
// CHECK-NEXT: %[[RANGE_BEGIN_FIELD:.*]] = arith.index_cast %[[RANGE_RECORD]] : index to i64
// CHECK-NEXT: %[[RANGE_BEGIN_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]][%[[RANGE_BEGIN_FIELD]]]
// CHECK-NEXT: %[[RANGE_BEGIN_WORD:.*]] = llvm.load %[[RANGE_BEGIN_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[RANGE_END_FIELD:.*]] = arith.index_cast %[[RANGE_END_INDEX]] : index to i64
// CHECK-NEXT: %[[RANGE_END_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]][%[[RANGE_END_FIELD]]]
// CHECK-NEXT: %[[RANGE_END_WORD:.*]] = llvm.load %[[RANGE_END_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[RANGE_FLOOR:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[RANGE_BEGIN_FLOORED:.*]] = arith.maxsi %[[RANGE_BEGIN_WORD]], %[[RANGE_FLOOR]] : i32
// CHECK-NEXT: %[[RANGE_BEGIN_BOUND:.*]] = arith.minsi %[[RANGE_BEGIN_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[RANGE_END_FLOORED:.*]] = arith.maxsi %[[RANGE_END_WORD]], %[[RANGE_BEGIN_BOUND]] : i32
// CHECK-NEXT: %[[RANGE_END_BOUND:.*]] = arith.minsi %[[RANGE_END_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[RANGE_BEGIN:.*]] = arith.index_cast %[[RANGE_BEGIN_BOUND]] : i32 to index
// CHECK-NEXT: %[[RANGE_END:.*]] = arith.index_cast %[[RANGE_END_BOUND]] : i32 to index
// CHECK-NEXT: %[[PARTIAL_FIRST:.*]] = arith.addi %[[RANGE_BEGIN]], %[[THREAD]] : index
// CHECK-NEXT: %[[PARTIAL_IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[PARTIAL_LOCAL:.*]] = scf.for %[[PARTIAL_I:.*]] = %[[PARTIAL_FIRST]] to %[[RANGE_END]] step %[[BLOCK]] iter_args(%[[PARTIAL_ACC:.*]] = %[[PARTIAL_IDENTITY]]) -> (f32) {
// CHECK-NEXT:   %[[PARTIAL_INDEX:.*]] = arith.index_cast %[[PARTIAL_I]] : index to i64
// CHECK-NEXT:   %[[PARTIAL_ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[PARTIAL_INDEX]]]
// CHECK-NEXT:   %[[PARTIAL_VALUE:.*]] = llvm.load %[[PARTIAL_ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT:   %[[PARTIAL_SUM:.*]] = arith.addf %[[PARTIAL_ACC]], %[[PARTIAL_VALUE]] : f32
// CHECK-NEXT:   scf.yield %[[PARTIAL_SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: %[[PARTIAL_TOTAL:.*]] = gpu.all_reduce add %[[PARTIAL_LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32

// Publication, writer side. The leader stores the partial into its own
// scratch slot, the block converges, and only then does the leader fence and
// publish completion. The fence directly precedes the completion atomic, and
// no atomic sits between the scratch store and the fence. A publisher that
// counted first could let another block observe the group complete and read
// scratch before this store became visible to it.
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   %[[SLOT:.*]] = arith.index_cast %[[PARTIAL]] : index to i64
// CHECK-NEXT:   %[[SLOT_ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[SLOT]]]
// CHECK-NEXT:   llvm.store %[[PARTIAL_TOTAL]], %[[SLOT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: scf.if %[[LEADER]] {
// The partial's merge group is compared with the merge count before it
// addresses anything. This branch runs in the leader alone, after the
// barrier above and before the next one, so no barrier depends on it. An
// out-of-range group updates no counter and reads no record.
// CHECK-NEXT:   %[[GROUP_SLOT:.*]] = arith.index_cast %[[PARTIAL]] : index to i64
// CHECK-NEXT:   %[[GROUP_ADDRESS:.*]] = llvm.getelementptr %[[MERGE_IDS]][%[[GROUP_SLOT]]]
// CHECK-NEXT:   %[[GROUP_WORD:.*]] = llvm.load %[[GROUP_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT:   %[[GROUP_IN_RANGE:.*]] = arith.cmpi ult, %[[GROUP_WORD]], %[[MERGE_COUNT]] : i32
// CHECK-NEXT:   %[[PUBLISHED:.*]] = scf.if %[[GROUP_IN_RANGE]] -> (i32) {
// The group has its completion counter at 3 + group.
// CHECK-NEXT:   %[[GROUP:.*]] = arith.index_cast %[[GROUP_WORD]] : i32 to index
// CHECK-NEXT:   %[[COMPLETION_BASE:.*]] = arith.constant 3 : index
// CHECK-NEXT:   %[[COMPLETION_INDEX:.*]] = arith.addi %[[COMPLETION_BASE]], %[[GROUP]] : index
// CHECK-NEXT:   %[[COMPLETION_I64:.*]] = arith.index_cast %[[COMPLETION_INDEX]] : index to i64
// CHECK-NEXT:   %[[COMPLETION:.*]] = llvm.getelementptr %[[COUNTERS]][%[[COMPLETION_I64]]]
// The group is complete when the count reaches the length of the scratch
// range in its merge record, end - begin. The two casts to index below are
// that range; RANGE pins what they cast.
// CHECK-NEXT:   %[[GROUP_FIELDS:.*]] = arith.constant 3 : index
// CHECK-NEXT:   %[[GROUP_RECORD:.*]] = arith.muli %[[GROUP]], %[[GROUP_FIELDS]] : index
// CHECK-NOT: llvm.atomicrmw
// CHECK-NOT: nvvm.memory.barrier
// CHECK-NOT: gpu.barrier
// CHECK-NOT: scf.if
// CHECK: %[[GROUP_BEGIN:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NOT: llvm.atomicrmw
// CHECK-NOT: nvvm.memory.barrier
// CHECK-NOT: gpu.barrier
// CHECK-NOT: scf.if
// CHECK: %[[GROUP_END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT:   %[[EXPECTED_INDEX:.*]] = arith.subi %[[GROUP_END]], %[[GROUP_BEGIN]] : index
// CHECK-NEXT:   %[[EXPECTED:.*]] = arith.index_cast %[[EXPECTED_INDEX]] : index to i32
// CHECK-NEXT:   nvvm.memory.barrier <gpu>
// CHECK-NEXT:   %[[PREVIOUS:.*]] = llvm.atomicrmw add %[[COMPLETION]], %[[ONE_I32]] acq_rel : !llvm.ptr, i32
// CHECK-NEXT:   %[[COMPLETED:.*]] = arith.addi %[[PREVIOUS]], %[[ONE_I32]] : i32
// CHECK-NEXT:   %[[IS_LAST:.*]] = arith.cmpi eq, %[[COMPLETED]], %[[EXPECTED]] : i32
// The leader broadcasts the ready merge, or -1, through shared slot 1. An
// out-of-range group broadcasts -1 as well, so the merge below only ever
// sees a group inside the merge count.
// CHECK-NEXT:   %[[READY:.*]] = scf.if %[[IS_LAST]] -> (i32) {
// CHECK-NEXT:     scf.yield %[[GROUP_WORD]] : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[NOT_READY:.*]] = arith.constant -1 : i32
// CHECK-NEXT:     scf.yield %[[NOT_READY]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.yield %[[READY]] : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[SKIPPED:.*]] = arith.constant -1 : i32
// CHECK-NEXT:     scf.yield %[[SKIPPED]] : i32
// CHECK-NEXT:   }
// CHECK-NEXT:   memref.store %[[PUBLISHED]], %[[SHARED]][%[[ONE]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[READY_MERGE:.*]] = memref.load %[[SHARED]][%[[ONE]]]

// Publication, reader side. The merge guard is the value every thread loaded
// after the barrier, so the all-reduce under it is block-uniform. The fence
// is the first operation of the merge, ahead of every scratch load.
// CHECK-NEXT: %[[HAS_MERGE:.*]] = arith.cmpi sge, %[[READY_MERGE]], %[[ZERO_I32]] : i32
// CHECK-NEXT: scf.if %[[HAS_MERGE]] {
// CHECK-NEXT:   nvvm.memory.barrier <gpu>
// The first field of the ready merge's record names the output segment. It
// is compared with the segment count as loaded; the comparison yields a
// value for the store predicate and opens no branch here.
// CHECK-NEXT:   %[[MERGE:.*]] = arith.index_cast %[[READY_MERGE]] : i32 to index
// CHECK-NEXT:   %[[MERGE_FIELDS:.*]] = arith.constant 3 : index
// CHECK-NEXT:   %[[MERGE_RECORD:.*]] = arith.muli %[[MERGE]], %[[MERGE_FIELDS]] : index
// CHECK-NEXT:   %[[MERGE_BEGIN_INDEX:.*]] = arith.addi %[[MERGE_RECORD]], %[[ONE]] : index
// CHECK-NEXT:   %[[MERGE_END_INDEX:.*]] = arith.addi %[[MERGE_BEGIN_INDEX]], %[[ONE]] : index
// CHECK-NEXT:   %[[SEGMENT_FIELD:.*]] = arith.index_cast %[[MERGE_RECORD]] : index to i64
// CHECK-NEXT:   %[[SEGMENT_ADDRESS:.*]] = llvm.getelementptr %[[MERGES]][%[[SEGMENT_FIELD]]]
// CHECK-NEXT:   %[[SEGMENT_WORD:.*]] = llvm.load %[[SEGMENT_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT:   %[[SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[SEGMENT_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT:   %[[SEGMENT:.*]] = arith.index_cast %[[SEGMENT_WORD]] : i32 to index
// The next two casts to index are the scratch range; RANGE pins what they
// cast. The merge sums the published partials themselves.
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK-NOT: nvvm.memory.barrier
// CHECK: %[[MERGE_BEGIN:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NOT: scf.if
// CHECK-NOT: gpu.barrier
// CHECK-NOT: nvvm.memory.barrier
// CHECK: %[[MERGE_END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT:   %[[MERGE_FIRST:.*]] = arith.addi %[[MERGE_BEGIN]], %[[THREAD]] : index
// CHECK-NEXT:   %[[MERGE_IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:   %[[MERGE_LOCAL:.*]] = scf.for %[[MERGE_I:.*]] = %[[MERGE_FIRST]] to %[[MERGE_END]] step %[[BLOCK]] iter_args(%[[MERGE_ACC:.*]] = %[[MERGE_IDENTITY]]) -> (f32) {
// CHECK-NEXT:     %[[MERGE_INDEX:.*]] = arith.index_cast %[[MERGE_I]] : index to i64
// CHECK-NEXT:     %[[PUBLISHED_ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[MERGE_INDEX]]]
// CHECK-NEXT:     %[[PUBLISHED:.*]] = llvm.load %[[PUBLISHED_ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT:     %[[MERGE_SUM:.*]] = arith.addf %[[MERGE_ACC]], %[[PUBLISHED]] : f32
// CHECK-NEXT:     scf.yield %[[MERGE_SUM]] : f32
// CHECK-NEXT:   }
// CHECK-NEXT:   %[[MERGE_TOTAL:.*]] = gpu.all_reduce add %[[MERGE_LOCAL]] uniform {
// CHECK-NEXT:   } : (f32) -> f32
// The leader is the only writer, and it stores the merged total at the
// segment the record names, unless that segment is out of range.
// CHECK-NEXT:   %[[MERGE_MAY_STORE:.*]] = arith.andi %[[LEADER]], %[[SEGMENT_IN_RANGE]] : i1
// CHECK-NEXT:   scf.if %[[MERGE_MAY_STORE]] {
// CHECK-NEXT:     %[[SEGMENT_I64:.*]] = arith.index_cast %[[SEGMENT]] : index to i64
// CHECK-NEXT:     %[[MERGE_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SEGMENT_I64]]]
// CHECK-NEXT:     llvm.store %[[MERGE_TOTAL]], %[[MERGE_OUTPUT]] : f32, !llvm.ptr
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NEXT: }

// Next partial claim, again through shared slot 0 behind a barrier.
// CHECK-NEXT: %[[NEXT_PARTIAL_QUEUE:.*]] = arith.constant 2 : i64
// CHECK-NEXT: %[[NEXT_PARTIAL_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[NEXT_PARTIAL_QUEUE]]]
// CHECK-NEXT: %[[NEXT_PARTIAL_CLAIM:.*]] = scf.if %[[LEADER]] -> (i32) {
// CHECK-NEXT:   %[[NEXT_PARTIAL_CLAIMED:.*]] = llvm.atomicrmw add %[[NEXT_PARTIAL_COUNTER]], %[[FOUR_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[NEXT_PARTIAL_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT:   memref.store %[[NEXT_PARTIAL_CLAIM]], %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: }
// CHECK-NEXT: gpu.barrier
// CHECK-NEXT: %[[NEXT_PARTIAL:.*]] = memref.load %[[SHARED]][%[[ZERO]]]
// CHECK-NEXT: scf.yield %[[NEXT_PARTIAL]] : i32
// CHECK-NEXT: }

// Warp queue. Each warp claims up to eight consecutive tasks on its own: lane
// zero performs the atomic and a shuffle broadcasts the claim inside the
// warp. Warps of one block diverge from here on.
// CHECK-NEXT: %[[WARP:.*]] = arith.constant 32 : index
// CHECK-NEXT: %[[LANE:.*]] = arith.remui %[[THREAD]], %[[WARP]] : index
// CHECK-NEXT: %[[LANE_LEADER:.*]] = arith.cmpi eq, %[[LANE]], %[[ZERO]] : index
// CHECK-NEXT: %[[FIRST_WARP_QUEUE:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[FIRST_WARP_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[FIRST_WARP_QUEUE]]]
// CHECK-NEXT: %[[FIRST_WARP_CLAIM:.*]] = scf.if %[[LANE_LEADER]] -> (i32) {
// CHECK-NEXT:   %[[FIRST_WARP_CLAIMED:.*]] = llvm.atomicrmw add %[[FIRST_WARP_COUNTER]], %[[EIGHT_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[FIRST_WARP_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: %[[FIRST_WARP_WIDTH:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[FIRST_WARP_SOURCE:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[FIRST_WARP:[^,]+]], %{{.*}} = gpu.shuffle idx %[[FIRST_WARP_CLAIM]], %[[FIRST_WARP_SOURCE]], %[[FIRST_WARP_WIDTH]] : i32
// CHECK-NEXT: %{{.*}} = scf.while (%[[WARP_HEAD:.*]] = %[[FIRST_WARP]]) : (i32) -> i32 {
// CHECK-NEXT:   %[[HAS_WARP:.*]] = arith.cmpi ult, %[[WARP_HEAD]], %[[WARP_COUNT]] : i32
// CHECK-NEXT:   scf.condition(%[[HAS_WARP]]) %[[WARP_HEAD]] : i32
// CHECK-NEXT: } do {
// CHECK-NEXT: ^bb0(%[[WARP_BATCH:.*]]: i32):
// CHECK-NEXT:   %[[WARP_BATCH_END:.*]] = arith.addi %[[WARP_BATCH]], %[[EIGHT_I32]] : i32
// CHECK-NEXT:   %[[WARP_BOUNDED_END:.*]] = arith.minui %[[WARP_BATCH_END]], %[[WARP_COUNT]] : i32
// CHECK-NEXT:   %[[WARP_BATCH_FIRST:.*]] = arith.index_cast %[[WARP_BATCH]] : i32 to index
// CHECK-NEXT:   %[[WARP_BATCH_LAST:.*]] = arith.index_cast %[[WARP_BOUNDED_END]] : i32 to index
// CHECK-NEXT:   scf.for %[[WARP_TASK:.*]] = %[[WARP_BATCH_FIRST]] to %[[WARP_BATCH_LAST]] step %[[ONE]] {
// CHECK-NEXT:   %[[WARP_TASK_I64:.*]] = arith.index_cast %[[WARP_TASK]] : index to i64

// The task word names the segment; its two offsets are the segment's range.
// The word is compared with the segment count as loaded. An out-of-range
// word selects offsets[0] for both ends, which is an empty range, so the
// bound opens no branch around the reduction.
// CHECK-NEXT: %[[WARP_ID_ADDRESS:.*]] = llvm.getelementptr %[[WARP_IDS]][%[[WARP_TASK_I64]]]
// CHECK-NEXT: %[[WARP_ID_WORD:.*]] = llvm.load %[[WARP_ID_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[WARP_ID_IN_RANGE:.*]] = arith.cmpi ult, %[[WARP_ID_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT: %[[WARP_SEGMENT:.*]] = arith.index_cast %[[WARP_ID_WORD]] : i32 to index
// CHECK-NEXT: %[[WARP_SEGMENT_I64:.*]] = arith.index_cast %[[WARP_SEGMENT]] : index to i64
// CHECK-NEXT: %[[WARP_START_INDEX:.*]] = arith.select %[[WARP_ID_IN_RANGE]], %[[WARP_SEGMENT]], %[[ZERO]] : index
// CHECK-NEXT: %[[WARP_START_INDEX_I64:.*]] = arith.index_cast %[[WARP_START_INDEX]] : index to i64
// CHECK-NEXT: %[[WARP_START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[WARP_START_INDEX_I64]]]
// CHECK-NEXT: %[[WARP_START_WORD:.*]] = llvm.load %[[WARP_START_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[WARP_NEXT_SEGMENT:.*]] = arith.addi %[[WARP_SEGMENT]], %[[ONE]] : index
// CHECK-NEXT: %[[WARP_END_INDEX:.*]] = arith.select %[[WARP_ID_IN_RANGE]], %[[WARP_NEXT_SEGMENT]], %[[ZERO]] : index
// CHECK-NEXT: %[[WARP_END_INDEX_I64:.*]] = arith.index_cast %[[WARP_END_INDEX]] : index to i64
// CHECK-NEXT: %[[WARP_END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]][%[[WARP_END_INDEX_I64]]]
// CHECK-NEXT: %[[WARP_END_WORD:.*]] = llvm.load %[[WARP_END_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[WARP_FLOOR:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[WARP_START_FLOORED:.*]] = arith.maxsi %[[WARP_START_WORD]], %[[WARP_FLOOR]] : i32
// CHECK-NEXT: %[[WARP_START_BOUND:.*]] = arith.minsi %[[WARP_START_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[WARP_END_FLOORED:.*]] = arith.maxsi %[[WARP_END_WORD]], %[[WARP_START_BOUND]] : i32
// CHECK-NEXT: %[[WARP_END_BOUND:.*]] = arith.minsi %[[WARP_END_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[WARP_START:.*]] = arith.index_cast %[[WARP_START_BOUND]] : i32 to index
// CHECK-NEXT: %[[WARP_END:.*]] = arith.index_cast %[[WARP_END_BOUND]] : i32 to index
// Each lane starts at its own offset into the range and strides by the
// number of lanes, so the range is covered exactly once.
// CHECK-NEXT: %[[WARP_FIRST:.*]] = arith.addi %[[WARP_START]], %[[LANE]] : index
// CHECK-NEXT: %[[WARP_IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT: %[[WARP_LOCAL:.*]] = scf.for %[[WARP_I:.*]] = %[[WARP_FIRST]] to %[[WARP_END]] step %[[WARP]] iter_args(%[[WARP_ACC:.*]] = %[[WARP_IDENTITY]]) -> (f32) {
// CHECK-NEXT:   %[[WARP_INDEX:.*]] = arith.index_cast %[[WARP_I]] : index to i64
// CHECK-NEXT:   %[[WARP_ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[WARP_INDEX]]]
// CHECK-NEXT:   %[[WARP_VALUE:.*]] = llvm.load %[[WARP_ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT:   %[[WARP_SUM:.*]] = arith.addf %[[WARP_ACC]], %[[WARP_VALUE]] : f32
// CHECK-NEXT:   scf.yield %[[WARP_SUM]] : f32
// CHECK-NEXT: }
// Five shuffle stages fold the 32 lanes; each stage adds the running total to
// its copy from the lane 1, 2, 4, 8, and 16 away.
// CHECK-NEXT: %[[WARP_W1:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[WARP_D1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[WARP_S1:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_LOCAL]], %[[WARP_D1]], %[[WARP_W1]] : f32
// CHECK-NEXT: %[[WARP_T1:.*]] = arith.addf %[[WARP_LOCAL]], %[[WARP_S1]] : f32
// CHECK-NEXT: %[[WARP_W2:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[WARP_D2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: %[[WARP_S2:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_T1]], %[[WARP_D2]], %[[WARP_W2]] : f32
// CHECK-NEXT: %[[WARP_T2:.*]] = arith.addf %[[WARP_T1]], %[[WARP_S2]] : f32
// CHECK-NEXT: %[[WARP_W3:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[WARP_D3:.*]] = arith.constant 4 : i32
// CHECK-NEXT: %[[WARP_S3:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_T2]], %[[WARP_D3]], %[[WARP_W3]] : f32
// CHECK-NEXT: %[[WARP_T3:.*]] = arith.addf %[[WARP_T2]], %[[WARP_S3]] : f32
// CHECK-NEXT: %[[WARP_W4:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[WARP_D4:.*]] = arith.constant 8 : i32
// CHECK-NEXT: %[[WARP_S4:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_T3]], %[[WARP_D4]], %[[WARP_W4]] : f32
// CHECK-NEXT: %[[WARP_T4:.*]] = arith.addf %[[WARP_T3]], %[[WARP_S4]] : f32
// CHECK-NEXT: %[[WARP_W5:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[WARP_D5:.*]] = arith.constant 16 : i32
// CHECK-NEXT: %[[WARP_S5:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_T4]], %[[WARP_D5]], %[[WARP_W5]] : f32
// CHECK-NEXT: %[[WARP_TOTAL:.*]] = arith.addf %[[WARP_T4]], %[[WARP_S5]] : f32
// Lane zero of the warp is the only writer, and it stores the total at the
// segment the task word named, unless that segment is out of range.
// CHECK-NEXT: %[[WARP_WRITER:.*]] = arith.cmpi eq, %[[LANE]], %[[ZERO]] : index
// CHECK-NEXT: %[[WARP_MAY_STORE:.*]] = arith.andi %[[WARP_WRITER]], %[[WARP_ID_IN_RANGE]] : i1
// CHECK-NEXT: scf.if %[[WARP_MAY_STORE]] {
// CHECK-NEXT:   %[[WARP_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[WARP_SEGMENT_I64]]]
// CHECK-NEXT:   llvm.store %[[WARP_TOTAL]], %[[WARP_OUTPUT]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: %[[NEXT_WARP_QUEUE:.*]] = arith.constant 0 : i64
// CHECK-NEXT: %[[NEXT_WARP_COUNTER:.*]] = llvm.getelementptr %[[COUNTERS]][%[[NEXT_WARP_QUEUE]]]
// CHECK-NEXT: %[[NEXT_WARP_CLAIM:.*]] = scf.if %[[LANE_LEADER]] -> (i32) {
// CHECK-NEXT:   %[[NEXT_WARP_CLAIMED:.*]] = llvm.atomicrmw add %[[NEXT_WARP_COUNTER]], %[[EIGHT_I32]] monotonic : !llvm.ptr, i32
// CHECK-NEXT:   scf.yield %[[NEXT_WARP_CLAIMED]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[ZERO_I32]] : i32
// CHECK-NEXT: }
// CHECK-NEXT: %[[NEXT_WARP_WIDTH:.*]] = arith.constant 32 : i32
// CHECK-NEXT: %[[NEXT_WARP_SOURCE:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[NEXT_WARP:[^,]+]], %{{.*}} = gpu.shuffle idx %[[NEXT_WARP_CLAIM]], %[[NEXT_WARP_SOURCE]], %[[NEXT_WARP_WIDTH]] : i32
// CHECK-NEXT: scf.yield %[[NEXT_WARP]] : i32
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

// The two merge-record ranges, followed forward from the record. The writer
// side subtracts its range to get the expected completion count; the reader
// side reduces scratch over its range.
// RANGE: gpu.func @segmented_sum(
// RANGE-SAME: %{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr,
// RANGE-SAME: %{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr,
// RANGE-SAME: %{{[^,]+}}: !llvm.ptr, %[[MERGE_IDS:[^,]+]]: !llvm.ptr, %[[MERGES:[^,]+]]: !llvm.ptr,
// RANGE: %[[ONE:.*]] = arith.constant 1 : index

// Writer side: fields one and two of the record of the partial's merge group.
// RANGE: %[[GROUP_ADDRESS:.*]] = llvm.getelementptr %[[MERGE_IDS]][
// RANGE-NEXT: %[[GROUP_WORD:.*]] = llvm.load %[[GROUP_ADDRESS]] : !llvm.ptr -> i32
// RANGE: %[[GROUP:.*]] = arith.index_cast %[[GROUP_WORD]] : i32 to index
// RANGE: %[[GROUP_RECORD:.*]] = arith.muli %[[GROUP]], %{{.*}} : index
// RANGE-NEXT: %[[GROUP_BEGIN_INDEX:.*]] = arith.addi %[[GROUP_RECORD]], %[[ONE]] : index
// RANGE-NEXT: %[[GROUP_END_INDEX:.*]] = arith.addi %[[GROUP_BEGIN_INDEX]], %[[ONE]] : index
// RANGE-NEXT: %[[GROUP_BEGIN_FIELD:.*]] = arith.index_cast %[[GROUP_BEGIN_INDEX]] : index to i64
// RANGE-NEXT: %[[GROUP_BEGIN_ADDRESS:.*]] = llvm.getelementptr %[[MERGES]][%[[GROUP_BEGIN_FIELD]]]
// RANGE-NEXT: %[[GROUP_BEGIN_WORD:.*]] = llvm.load %[[GROUP_BEGIN_ADDRESS]] : !llvm.ptr -> i32

// The begin word goes to an index cast or a bound. The end word is loaded
// either side of that, depending on whether a bound follows.
// RANGE-DAG: %[[GROUP_BEGIN_USE:[^ ]+]] = arith.{{index_cast|(max|min)[su]i}} %[[GROUP_BEGIN_WORD]]{{ : i32 to index|, }}
// RANGE-DAG: %[[GROUP_END_FIELD:[^ ]+]] = arith.index_cast %[[GROUP_END_INDEX]] : index to i64
// RANGE-DAG: %[[GROUP_END_ADDRESS:[^ ]+]] = llvm.getelementptr %[[MERGES]][%[[GROUP_END_FIELD]]]
// RANGE-DAG: %[[GROUP_END_WORD:[^ ]+]] = llvm.load %[[GROUP_END_ADDRESS]] : !llvm.ptr -> i32
// The end word is used after the begin word, never in its place. It goes to
// an index cast or a bound too. From there the begin goes to the right side
// of the subtraction and the end to the left side of the subtraction, or each
// to a further bound.
// RANGE-NOT: scf.if
// RANGE-DAG: %[[GROUP_END_USE:[^ ]+]] = arith.{{index_cast|(max|min)[su]i}} %[[GROUP_END_WORD]]{{ : i32 to index|, }}
// RANGE-DAG: {{ |(max|min)[su]i }}%[[GROUP_BEGIN_USE]]{{ : index$|, }}
// RANGE-DAG: arith.{{subi|(max|min)[su]i}} %[[GROUP_END_USE]],
// RANGE: nvvm.memory.barrier <gpu>

// Reader side: fields one and two of the ready merge's record.
// RANGE: nvvm.memory.barrier <gpu>
// RANGE-NEXT: %[[MERGE:.*]] = arith.index_cast %{{.*}} : i32 to index
// RANGE: %[[MERGE_RECORD:.*]] = arith.muli %[[MERGE]], %{{.*}} : index
// RANGE-NEXT: %[[MERGE_BEGIN_INDEX:.*]] = arith.addi %[[MERGE_RECORD]], %[[ONE]] : index
// RANGE-NEXT: %[[MERGE_END_INDEX:.*]] = arith.addi %[[MERGE_BEGIN_INDEX]], %[[ONE]] : index
// RANGE: %[[MERGE_BEGIN_FIELD:.*]] = arith.index_cast %[[MERGE_BEGIN_INDEX]] : index to i64
// RANGE-NEXT: %[[MERGE_BEGIN_ADDRESS:.*]] = llvm.getelementptr %[[MERGES]][%[[MERGE_BEGIN_FIELD]]]
// RANGE-NEXT: %[[MERGE_BEGIN_WORD:.*]] = llvm.load %[[MERGE_BEGIN_ADDRESS]] : !llvm.ptr -> i32

// The begin word goes to an index cast or a bound. The end word is loaded
// either side of that, depending on whether a bound follows.
// RANGE-DAG: %[[MERGE_BEGIN_USE:[^ ]+]] = arith.{{index_cast|(max|min)[su]i}} %[[MERGE_BEGIN_WORD]]{{ : i32 to index|, }}
// RANGE-DAG: %[[MERGE_END_FIELD:[^ ]+]] = arith.index_cast %[[MERGE_END_INDEX]] : index to i64
// RANGE-DAG: %[[MERGE_END_ADDRESS:[^ ]+]] = llvm.getelementptr %[[MERGES]][%[[MERGE_END_FIELD]]]
// RANGE-DAG: %[[MERGE_END_WORD:[^ ]+]] = llvm.load %[[MERGE_END_ADDRESS]] : !llvm.ptr -> i32
// The end word is used after the begin word, never in its place. It goes to
// an index cast or a bound too. From there the begin goes to the loop's first
// index and the end to the loop's upper bound, or each to a further bound.
// RANGE-NOT: scf.if
// RANGE-DAG: %[[MERGE_END_USE:[^ ]+]] = arith.{{index_cast|(max|min)[su]i}} %[[MERGE_END_WORD]]{{ : i32 to index|, }}
// RANGE-DAG: arith.{{addi|(max|min)[su]i}} %[[MERGE_BEGIN_USE]], %
// RANGE-DAG: {{ to|arith.(max|min)[su]i}} %[[MERGE_END_USE]]{{ step|, }}
// RANGE: gpu.all_reduce add

// The persistent kernel is specialized to 512 threads, and the block-size
// requirements keep it from being combined with the fused mixed schedule.
// BLOCK-SIZE: error: persistent lowering requires block-size 512, got 128
// FUSED: error: fused mixed lowering requires block-size 128, got 512
