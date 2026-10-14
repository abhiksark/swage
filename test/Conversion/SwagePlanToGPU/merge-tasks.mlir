// test/Conversion/SwagePlanToGPU/merge-tasks.mlir
// The plan function of a split merge kernel, written by hand. One block
// merges one split segment: the block index is compared with the merge
// count, the merge record is loaded, its segment is compared with the
// segment count and its range of scratch slots is clamped to the partial
// count, the block reduces as a whole, and thread 0 stores at the segment,
// and only when the segment is in range.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s \
// RUN:   --implicit-check-not=swage --implicit-check-not=func.func

// CHECK: gpu.module @merge_module {
// CHECK-NEXT: gpu.func @merge(%[[SCRATCH:.*]]: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr, %[[MERGES:.*]]: !llvm.ptr, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>,
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}} {
// CHECK-NEXT: %[[BLOCK:.*]] = gpu.block_id x
// CHECK-NEXT: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C512:.*]] = arith.constant 512 : index
// CHECK-NEXT: %[[TASKS:.*]] = arith.index_cast %[[MERGE_COUNT]] : i32 to index
// CHECK-NEXT: %[[HAS_TASK:.*]] = arith.cmpi slt, %[[BLOCK]], %[[TASKS]] : index
// CHECK-NEXT: scf.if %[[HAS_TASK]] {
// CHECK-NEXT: %[[C3:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[RECORD:.*]] = arith.muli %[[BLOCK]], %[[C3]] : index
// CHECK: llvm.getelementptr %[[MERGES]][
// CHECK-NEXT: %[[SEGMENT_WORD:.*]] = llvm.load
// CHECK-NEXT: %[[IN_RANGE:.*]] = arith.cmpi ult, %[[SEGMENT_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT: %[[SEGMENT:.*]] = arith.index_cast %[[SEGMENT_WORD]] : i32 to index
// CHECK: llvm.getelementptr %[[MERGES]][
// CHECK-NEXT: %[[BEGIN_WORD:.*]] = llvm.load
// CHECK: llvm.getelementptr %[[MERGES]][
// CHECK-NEXT: %[[END_WORD:.*]] = llvm.load
// CHECK: %[[BEGIN_FLOOR:.*]] = arith.maxsi %[[BEGIN_WORD]], %{{.*}} : i32
// CHECK-NEXT: %[[BEGIN:.*]] = arith.minsi %[[BEGIN_FLOOR]], %[[PARTIAL_COUNT]] : i32
// CHECK-NEXT: %[[END_FLOOR:.*]] = arith.maxsi %[[END_WORD]], %[[BEGIN]] : i32
// CHECK-NEXT: %[[END:.*]] = arith.minsi %[[END_FLOOR]], %[[PARTIAL_COUNT]] : i32
// CHECK: %[[LOCAL:.*]] = scf.for %[[INDEX:.*]] = %{{.*}} to %{{.*}} step %[[C512]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK-NEXT: %[[INDEX64:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[INDEX64]]]
// CHECK-NEXT: %[[PARTIAL:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT: %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[PARTIAL]] : f32
// CHECK-NEXT: scf.yield %[[NEXT]] : f32
// CHECK: %[[TOTAL:.*]] = gpu.all_reduce maximumf %[[LOCAL]] uniform {
// CHECK: %[[LEADER:.*]] = arith.cmpi eq, %[[THREAD]], %[[C0]] : index
// CHECK-NEXT: %[[MAY_STORE:.*]] = arith.andi %[[LEADER]], %[[IN_RANGE]] : i1
// CHECK-NEXT: scf.if %[[MAY_STORE]] {
// CHECK-NEXT: %[[SLOT:.*]] = arith.index_cast %[[SEGMENT]] : index to i64
// CHECK-NEXT: %[[SLOT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SLOT]]]
// CHECK-NEXT: llvm.store %[[TOTAL]], %[[SLOT_ADDRESS]] : f32, !llvm.ptr
// CHECK: gpu.return

module {
  func.func @merge(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
