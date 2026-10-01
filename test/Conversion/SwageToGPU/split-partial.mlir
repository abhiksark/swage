// test/Conversion/SwageToGPU/split-partial.mlir
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// The partial stage of a split reduction. Each block reduces one planned
// input range and writes one scratch slot. The element program (a fused map
// followed by the reduction region) runs here, on input values, and nowhere
// else in the split schedule; split-merge.mlir lowers the same module and
// checks that the merge stage does not run it again.
module {
  func.func @segmented_max(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %scaled = swage.map %segment
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      %two = arith.constant 2.000000e+00 : f32
      %doubled = arith.mulf %value, %two : f32
      swage.yield %doubled : f32
    }
    %maximum = swage.reduce %scaled kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %squared = arith.mulf %value, %value : f32
      swage.yield %squared : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK: gpu.module @segmented_max__partial_module
// CHECK: gpu.func @segmented_max__partial(%[[VALUES:[^,]+]]: !llvm.ptr, %[[RANGES:[^,]+]]: !llvm.ptr, %[[SCRATCH:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %[[PARTIAL_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 512, 1, 1>
// CHECK: %[[TASK:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BLOCK:.*]] = arith.constant 512 : index
// CHECK: %[[TASKS:.*]] = arith.index_cast %[[PARTIAL_COUNT]] : i32 to index

// The range guard depends only on the block index and a launch argument, so
// it is block-uniform. No other conditional may sit between it and the
// all-reduce, whose barriers every thread of the block must reach.
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[TASK]], %[[TASKS]] : index
// CHECK-NEXT: scf.if %[[IN_RANGE]] {
// CHECK-NOT: scf.if
// CHECK:   llvm.getelementptr %[[RANGES]]
// CHECK-NOT: scf.if
// CHECK:   %[[IDENTITY:.*]] = arith.constant 0xFF800000 : f32
// CHECK-NEXT: %[[LOCAL:.*]] = scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %[[BLOCK]] iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK-NEXT:   %[[INDEX:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK-NEXT:   %[[ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[INDEX]]]
// CHECK-NEXT:   %[[VALUE:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// The element program transforms each input value before the combine.
// CHECK-NEXT:   %[[TWO:.*]] = arith.constant 2.000000e+00 : f32
// CHECK-NEXT:   %[[DOUBLED:.*]] = arith.mulf %[[VALUE]], %[[TWO]] : f32
// CHECK-NEXT:   %[[SQUARED:.*]] = arith.mulf %[[DOUBLED]], %[[DOUBLED]] : f32
// CHECK-NEXT:   %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[SQUARED]] : f32
// CHECK-NEXT:   scf.yield %[[NEXT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: %[[TOTAL:.*]] = gpu.all_reduce maximumf %[[LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32
// Thread zero writes the block's scratch slot, indexed by the task itself, so
// every partial has one writer and one slot.
// CHECK-NEXT: %[[FIRST_THREAD:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index
// CHECK-NEXT: scf.if %[[FIRST_THREAD]] {
// CHECK-NEXT:   %[[SLOT:.*]] = arith.index_cast %[[TASK]] : index to i64
// CHECK-NEXT:   %[[SLOT_ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[SLOT]]]
// CHECK-NEXT:   llvm.store %[[TOTAL]], %[[SLOT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return
