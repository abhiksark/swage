// test/Conversion/SwageToGPU/segmented-sum-task-ids.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=32 use-task-ids=true' %s \
// RUN:   | FileCheck %s

module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>) {
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

// CHECK-NOT: swage.
// CHECK: gpu.module @segmented_sum_module
// CHECK: gpu.func @segmented_sum(%[[VALUES:[^,]+]]: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[TASK_IDS:[^,]+]]: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %[[TASK_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "task_ids", kind = "ptr", origin = "plan"}, {key = "value_count", kind = "i32", origin = "derived"}, {key = "task_count", kind = "i32", origin = "derived"}], backend = "cuda", entry = "segmented_sum", launch = {block = array<i32: 32, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK: %[[TASK_INDEX:.*]] = gpu.block_id x
// CHECK: %[[TASKS:.*]] = arith.index_cast %[[TASK_COUNT]] : i32 to index
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[TASK_INDEX]], %[[TASKS]] : index
// CHECK: scf.if %[[IN_RANGE]] {
// CHECK:   %[[TASK_ADDRESS:.*]] = llvm.getelementptr %[[TASK_IDS]]
// CHECK:   %[[SID_I32:.*]] = llvm.load %[[TASK_ADDRESS]] : !llvm.ptr -> i32
// CHECK:   %[[SID:.*]] = arith.index_cast %[[SID_I32]] : i32 to index
// CHECK:   %[[START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// CHECK:   %[[START_I32:.*]] = llvm.load %[[START_ADDRESS]] : !llvm.ptr -> i32
// CHECK:   %[[END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// CHECK:   %[[END_I32:.*]] = llvm.load %[[END_ADDRESS]] : !llvm.ptr -> i32
// CHECK-COUNT-5: gpu.shuffle xor
// CHECK:   %[[OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]]
// CHECK:   llvm.store %{{.*}}, %[[OUTPUT_ADDRESS]] : f32, !llvm.ptr
// CHECK: }
// CHECK: gpu.return
