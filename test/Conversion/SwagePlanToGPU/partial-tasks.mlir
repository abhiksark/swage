// test/Conversion/SwagePlanToGPU/partial-tasks.mlir
// The plan function of a split partial kernel, written by hand. One block
// reduces one chunk: the block index is compared with the partial count,
// the range record of the chunk is loaded and clamped to the value count,
// the block reduces as a whole, and thread 0 stores the result in the
// scratch slot of the task.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s \
// RUN:   --implicit-check-not=swage --implicit-check-not=func.func

// CHECK: gpu.module @partial_module {
// CHECK-NEXT: gpu.func @partial(%[[VALUES:.*]]: !llvm.ptr, %[[RANGES:.*]]: !llvm.ptr, %[[SCRATCH:.*]]: !llvm.ptr, %[[VALUE_COUNT:.*]]: i32, %[[PARTIAL_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>,
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {access = "write", key = "scratch", kind = "ptr", origin = "scratch"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "partial_count", kind = "i32", origin = "derived"}], backend = "cuda", entry = "partial", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}} {
// CHECK-NEXT: %[[BLOCK:.*]] = gpu.block_id x
// CHECK-NEXT: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[C512:.*]] = arith.constant 512 : index
// CHECK-NEXT: %[[TASKS:.*]] = arith.index_cast %[[PARTIAL_COUNT]] : i32 to index
// CHECK-NEXT: %[[HAS_TASK:.*]] = arith.cmpi slt, %[[BLOCK]], %[[TASKS]] : index
// CHECK-NEXT: scf.if %[[HAS_TASK]] {
// CHECK-NEXT: %[[C2:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[RECORD:.*]] = arith.muli %[[BLOCK]], %[[C2]] : index
// CHECK: llvm.getelementptr %[[RANGES]][
// CHECK-NEXT: %[[BEGIN_WORD:.*]] = llvm.load
// CHECK: %[[END_INDEX:.*]] = arith.addi %[[RECORD]], %{{.*}} : index
// CHECK: llvm.getelementptr %[[RANGES]][
// CHECK-NEXT: %[[END_WORD:.*]] = llvm.load
// CHECK: %[[BEGIN_FLOOR:.*]] = arith.maxsi %[[BEGIN_WORD]], %{{.*}} : i32
// CHECK-NEXT: %[[BEGIN:.*]] = arith.minsi %[[BEGIN_FLOOR]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[END_FLOOR:.*]] = arith.maxsi %[[END_WORD]], %[[BEGIN]] : i32
// CHECK-NEXT: %[[END:.*]] = arith.minsi %[[END_FLOOR]], %[[VALUE_COUNT]] : i32
// CHECK: %[[LOCAL:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %[[C512]] iter_args(
// CHECK: llvm.getelementptr %[[VALUES]][
// CHECK: arith.mulf
// CHECK: %[[TOTAL:.*]] = gpu.all_reduce maximumf %[[LOCAL]] uniform {
// CHECK: %[[LEADER:.*]] = arith.cmpi eq, %[[THREAD]], %[[C0]] : index
// CHECK-NEXT: scf.if %[[LEADER]] {
// CHECK-NEXT: %[[SLOT:.*]] = arith.index_cast %[[BLOCK]] : index to i64
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[SLOT]]]
// CHECK-NEXT: llvm.store %[[TOTAL]], %[[ADDRESS]] : f32, !llvm.ptr
// CHECK: gpu.return

module {
  func.func @partial(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        %square = arith.mulf %value, %value : f32
        swage.yield %square : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
