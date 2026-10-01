// test/Conversion/SwageToGPU/fused-mixed.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed' %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed' %s \
// RUN:   | FileCheck %s --check-prefix=SYNC
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=64 fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BLOCK-SIZE

// One kernel runs two schedules. The leading blocks pack four warp tasks
// each and reduce with shuffles; the remaining blocks run one CTA task each
// and reduce with a block-wide all-reduce, which contains block barriers. A
// barrier is legal only where every thread of the block reaches it, so this
// test pins which predicate guards each schedule.
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
// CHECK: gpu.func @segmented_sum(%[[VALUES:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[TASK_IDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %[[WARP_COUNT_I32:[^,]+]]: i32, %[[CTA_COUNT_I32:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 128, 1, 1>
// CHECK: %[[BLOCK_ID:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BLOCK:.*]] = arith.constant 128 : index
// CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[WARP:.*]] = arith.constant 32 : index
// CHECK: %[[WARP_COUNT:.*]] = arith.index_cast %[[WARP_COUNT_I32]] : i32 to index
// CHECK: %[[CTA_COUNT:.*]] = arith.index_cast %[[CTA_COUNT_I32]] : i32 to index
// CHECK: %[[ROUNDED:.*]] = arith.addi %[[WARP_COUNT]], %{{.*}} : index
// CHECK: %[[WARP_BLOCKS:.*]] = arith.divui %[[ROUNDED]], %[[FOUR]] : index

// The schedule is chosen from the block index and a launch argument, so all
// threads of one block take the same branch.
// CHECK: %[[IS_WARP_BLOCK:.*]] = arith.cmpi ult, %[[BLOCK_ID]], %[[WARP_BLOCKS]] : index
// CHECK: scf.if %[[IS_WARP_BLOCK]] {

// Warp schedule. The task guard depends on the physical warp, so it is
// uniform within a warp but not within the block.
// CHECK:   %[[PHYSICAL_WARP:.*]] = arith.divui %[[THREAD]], %[[WARP]] : index
// CHECK:   %[[LANE:.*]] = arith.remui %[[THREAD]], %[[WARP]] : index
// CHECK:   %[[FIRST_TASK:.*]] = arith.muli %[[BLOCK_ID]], %[[FOUR]] : index
// CHECK:   %[[WARP_TASK:.*]] = arith.addi %[[FIRST_TASK]], %[[PHYSICAL_WARP]] : index
// CHECK:   %[[WARP_IN_RANGE:.*]] = arith.cmpi ult, %[[WARP_TASK]], %[[WARP_COUNT]] : index
// CHECK:   scf.if %[[WARP_IN_RANGE]] {
// CHECK:     %[[WARP_TASK_I64:.*]] = arith.index_cast %[[WARP_TASK]] : index to i64
// CHECK:     llvm.getelementptr %[[TASK_IDS]][%[[WARP_TASK_I64]]]
// CHECK:     %[[WARP_LOCAL:.*]] = scf.for %{{.*}} step %[[WARP]] iter_args(
// CHECK:       llvm.getelementptr %[[VALUES]]
// CHECK:     }
// Five shuffle stages fold the 32 lanes; each stage combines the running
// total with its shuffled copy.
// CHECK:     %[[S1:[^,]+]], %{{.*}} = gpu.shuffle xor %[[WARP_LOCAL]],
// CHECK-NEXT: %[[T1:.*]] = arith.addf %[[WARP_LOCAL]], %[[S1]] : f32
// CHECK:     %[[S2:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T1]],
// CHECK-NEXT: %[[T2:.*]] = arith.addf %[[T1]], %[[S2]] : f32
// CHECK:     %[[S3:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T2]],
// CHECK-NEXT: %[[T3:.*]] = arith.addf %[[T2]], %[[S3]] : f32
// CHECK:     %[[S4:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T3]],
// CHECK-NEXT: %[[T4:.*]] = arith.addf %[[T3]], %[[S4]] : f32
// CHECK:     %[[S5:[^,]+]], %{{.*}} = gpu.shuffle xor %[[T4]],
// CHECK-NEXT: %[[WARP_TOTAL:.*]] = arith.addf %[[T4]], %[[S5]] : f32
// Lane zero of each warp is the only writer of that task's output.
// CHECK-NEXT: %[[FIRST_LANE:.*]] = arith.cmpi eq, %[[LANE]], %[[ZERO]] : index
// CHECK-NEXT: scf.if %[[FIRST_LANE]] {
// CHECK-NEXT:   %[[WARP_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]]
// CHECK-NEXT:   llvm.store %[[WARP_TOTAL]], %[[WARP_OUTPUT]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: } else {

// CTA schedule. The task guard depends only on the block index and launch
// arguments, so it is block-uniform.
// CHECK-NEXT: %[[CTA_TASK:.*]] = arith.subi %[[BLOCK_ID]], %[[WARP_BLOCKS]] : index
// CHECK-NEXT: %[[CTA_IN_RANGE:.*]] = arith.cmpi ult, %[[CTA_TASK]], %[[CTA_COUNT]] : index
// CHECK-NEXT: scf.if %[[CTA_IN_RANGE]] {
// CHECK-NEXT:   %[[MIXED_TASK:.*]] = arith.addi %[[WARP_COUNT]], %[[CTA_TASK]] : index
// CHECK-NEXT:   %[[MIXED_TASK_I64:.*]] = arith.index_cast %[[MIXED_TASK]] : index to i64
// CHECK-NEXT:   llvm.getelementptr %[[TASK_IDS]][%[[MIXED_TASK_I64]]]
// CHECK:     %[[CTA_LOCAL:.*]] = scf.for %{{.*}} step %[[BLOCK]] iter_args(
// CHECK:       llvm.getelementptr %[[VALUES]]
// CHECK:     }
// CHECK-NEXT: %[[CTA_TOTAL:.*]] = gpu.all_reduce add %[[CTA_LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32
// Thread zero is the only writer, after the all-reduce has broadcast.
// CHECK-NEXT: %[[FIRST_THREAD:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index
// CHECK-NEXT: scf.if %[[FIRST_THREAD]] {
// CHECK-NEXT:   %[[CTA_OUTPUT:.*]] = llvm.getelementptr %[[OUTPUT]]
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

// The fused kernel is specialized to four warps per block.
// BLOCK-SIZE: error: fused mixed lowering requires block-size 128
