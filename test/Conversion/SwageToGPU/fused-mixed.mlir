// test/Conversion/SwageToGPU/fused-mixed.mlir
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=SYNC
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed block-threads=64' \
// RUN:   --swage-plan-to-gpu %s | FileCheck %s --implicit-check-not=swage.
// RUN: not swage-opt --swage-to-plan='schedule=fused-mixed,task-ids' \
// RUN:   --swage-plan-to-gpu %s 2>&1 | FileCheck %s --check-prefix=TASK-IDS

// One kernel runs two schedules. The leading blocks pack four warp tasks
// each and reduce with shuffles; the remaining blocks run one CTA task each
// and reduce with a block-wide all-reduce, which contains block barriers. A
// barrier is legal only where every thread of the block reaches it, so this
// test pins which predicate guards each schedule.
//
// Every line from the schedule choice to the return is matched in order with
// captured operands, so each task's data path is pinned end to end: task
// word, segment offsets, bounded range, loop bounds, reduction, stored
// output.
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
// CHECK: gpu.func @segmented_sum(%[[VALUES:[^,]+]]: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[TASK_IDS:[^,]+]]: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %[[WARP_COUNT_I32:[^,]+]]: i32, %[[CTA_COUNT_I32:[^,]+]]: i32, %[[SEGMENT_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 128, 1, 1>
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "task_ids", kind = "ptr", origin = "plan"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "warp_task_count", kind = "i32", origin = "derived"}, {key = "cta_task_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_sum", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK: %[[BLOCK_ID:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[BLOCK:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[WARP:.*]] = arith.constant 32 : index
// CHECK: %[[WARP_COUNT:.*]] = arith.index_cast %[[WARP_COUNT_I32]] : i32 to index
// CHECK-NEXT: %[[CTA_COUNT:.*]] = arith.index_cast %[[CTA_COUNT_I32]] : i32 to index

// Four warp tasks share a block, so the warp schedule takes the first
// ceil(warp_count / 4) blocks. The choice depends only on the block index and
// a launch argument, so all threads of one block take the same branch.
// CHECK-NEXT: %[[ROUNDED:.*]] = arith.addi %[[WARP_COUNT]], %[[THREE]] : index
// CHECK-NEXT: %[[WARP_BLOCKS:.*]] = arith.divui %[[ROUNDED]], %[[FOUR]] : index
// CHECK-NEXT: %[[IS_WARP_BLOCK:.*]] = arith.cmpi ult, %[[BLOCK_ID]], %[[WARP_BLOCKS]] : index
// CHECK-NEXT: scf.if %[[IS_WARP_BLOCK]] {

// Warp schedule. Warp w of block b runs task 4 * b + w. The task guard
// depends on the physical warp, so it is uniform within a warp but not within
// the block.
// CHECK-NEXT: %[[PHYSICAL_WARP:.*]] = arith.divui %[[THREAD]], %[[WARP]] : index
// CHECK-NEXT: %[[LANE:.*]] = arith.remui %[[THREAD]], %[[WARP]] : index
// CHECK-NEXT: %[[FIRST_TASK:.*]] = arith.muli %[[BLOCK_ID]], %[[FOUR]] : index
// CHECK-NEXT: %[[WARP_TASK:.*]] = arith.addi %[[FIRST_TASK]], %[[PHYSICAL_WARP]] : index
// CHECK-NEXT: %[[WARP_IN_RANGE:.*]] = arith.cmpi ult, %[[WARP_TASK]], %[[WARP_COUNT]] : index
// CHECK-NEXT: scf.if %[[WARP_IN_RANGE]] {
// CHECK-NEXT: %[[WARP_TASK_I64:.*]] = arith.index_cast %[[WARP_TASK]] : index to i64

// The task word names the segment; its two offsets are the segment's range.
// The word is compared with the segment count as loaded. An out-of-range
// word selects offsets[0] for both ends, which is an empty range, so the
// bound opens no branch around the reduction.
// CHECK-NEXT: %[[WARP_ID_ADDRESS:.*]] = llvm.getelementptr %[[TASK_IDS]][%[[WARP_TASK_I64]]]
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
// CHECK-NEXT: } else {

// CTA schedule. The CTA tasks follow the warp tasks in the same task list,
// one block each. The task guard depends only on the block index and launch
// arguments, so it is block-uniform.
// CHECK-NEXT: %[[CTA_TASK:.*]] = arith.subi %[[BLOCK_ID]], %[[WARP_BLOCKS]] : index
// CHECK-NEXT: %[[CTA_IN_RANGE:.*]] = arith.cmpi ult, %[[CTA_TASK]], %[[CTA_COUNT]] : index
// CHECK-NEXT: scf.if %[[CTA_IN_RANGE]] {
// CHECK-NEXT: %[[MIXED_TASK:.*]] = arith.addi %[[WARP_COUNT]], %[[CTA_TASK]] : index
// CHECK-NEXT: %[[CTA_TASK_I64:.*]] = arith.index_cast %[[MIXED_TASK]] : index to i64

// The task word names the segment; its two offsets are the segment's range.
// The word is compared with the segment count as loaded. An out-of-range
// word selects offsets[0] for both ends, which is an empty range, so the
// bound opens no branch around the reduction.
// CHECK-NEXT: %[[CTA_ID_ADDRESS:.*]] = llvm.getelementptr %[[TASK_IDS]][%[[CTA_TASK_I64]]]
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
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// The warp branch holds shuffles and nothing that needs the whole block. The
// all-reduce sits directly under the block-uniform CTA guard, with no other
// conditional around it, and the CTA branch needs no shuffle of its own.
// SYNC: %[[IS_WARP_BLOCK:.*]] = arith.cmpi ult, %{{.*}} : index
// SYNC-NEXT: scf.if %[[IS_WARP_BLOCK]] {
// SYNC-NOT: gpu.barrier
// SYNC-NOT: gpu.all_reduce
// SYNC: gpu.shuffle xor
// SYNC-NOT: gpu.barrier
// SYNC-NOT: gpu.all_reduce
// SYNC: } else {
// SYNC: %[[CTA_IN_RANGE:.*]] = arith.cmpi ult, %{{.*}} : index
// SYNC-NEXT: scf.if %[[CTA_IN_RANGE]] {
// SYNC-NOT: scf.if
// SYNC-NOT: gpu.shuffle
// SYNC-NOT: gpu.barrier
// SYNC: gpu.all_reduce add
// SYNC-NOT: gpu.shuffle
// SYNC-NOT: gpu.barrier
// SYNC-NOT: gpu.all_reduce
// SYNC: gpu.return

// The fused kernel is specialized to four warps per block. The target fixes
// its launch width, so the third RUN line gives block-threads another value
// and matches the same kernel.

// The fused kernel and the task-id kernel both take the name of their
// function, so one schedule list cannot hold both. The list is refused
// before the module is admitted.
// TASK-IDS: error: schedules fused-mixed and task-ids both name their kernel @<function>; a schedule list names each kernel once
// TASK-IDS-NOT: gpu.func
