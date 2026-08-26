// test/Conversion/SwageToGPU/segmented-sum.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s

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

// CHECK-NOT: swage.
// CHECK: gpu.module @segmented_sum_module
// CHECK: gpu.func @segmented_sum(%[[VALUES:[^,]+]]: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %[[SEGMENT_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 128, 1, 1>
// CHECK: %[[SID:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: %[[SEGMENTS:.*]] = arith.index_cast %[[SEGMENT_COUNT]] : i32 to index
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[SID]], %[[SEGMENTS]] : index
// CHECK: scf.if %[[IN_RANGE]] {
// CHECK:   %[[START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// CHECK:   %[[START_I32:.*]] = llvm.load %[[START_ADDRESS]] : !llvm.ptr -> i32
// CHECK:   %[[END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// CHECK:   %[[END_I32:.*]] = llvm.load %[[END_ADDRESS]] : !llvm.ptr -> i32
// CHECK:   %[[START:.*]] = arith.index_cast %[[START_I32]] : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %[[END_I32]] : i32 to index
// CHECK:   %[[FIRST:.*]] = arith.addi %[[START]], %[[THREAD]] : index
// CHECK:   %[[IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[LOCAL:.*]] = scf.for %[[I:.*]] = %[[FIRST]] to %[[END]] step %{{.*}} iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:     %[[VALUE_ADDRESS:.*]] = llvm.getelementptr %[[VALUES]]
// CHECK:     %[[VALUE:.*]] = llvm.load %[[VALUE_ADDRESS]] : !llvm.ptr -> f32
// CHECK:     %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[VALUE]] : f32
// CHECK:     scf.yield %[[NEXT_ACC]] : f32
// CHECK:   }
// CHECK:   %[[TOTAL:.*]] = gpu.all_reduce add %[[LOCAL]] uniform
// CHECK:   %[[FIRST_THREAD:.*]] = arith.cmpi eq, %[[THREAD]], %{{.*}} : index
// CHECK:   scf.if %[[FIRST_THREAD]] {
// CHECK:     %[[OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]]
// CHECK:     llvm.store %[[TOTAL]], %[[OUTPUT_ADDRESS]] : f32, !llvm.ptr
// CHECK:   }
// CHECK: }
// CHECK: gpu.return
// CHECK-NOT: swage.
