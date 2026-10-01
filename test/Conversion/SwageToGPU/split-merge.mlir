// test/Conversion/SwageToGPU/split-merge.mlir
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu='merge' %s \
// RUN:   | FileCheck %s --implicit-check-not=swage. \
// RUN:       --implicit-check-not=arith.mulf

// The merge stage of a split reduction, lowered from the same module as
// split-partial.mlir. Each block combines one compact scratch range and one
// thread writes the segment result. Scratch already holds reduced partials,
// so the merge must combine them with the reduction kind alone: running the
// element program again would double and square every partial a second time.
// Both element operations are arith.mulf, which the RUN line forbids anywhere
// in this kernel.
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

// CHECK: gpu.module @segmented_max__merge_module
// CHECK: gpu.func @segmented_max__merge(%[[SCRATCH:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[RECORDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %[[MERGE_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 512, 1, 1>
// CHECK: %[[TASK:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BLOCK:.*]] = arith.constant 512 : index
// CHECK: %[[TASKS:.*]] = arith.index_cast %[[MERGE_COUNT]] : i32 to index

// The range guard depends only on the block index and a launch argument, so
// it is block-uniform. No other conditional may sit between it and the
// all-reduce, whose barriers every thread of the block must reach.
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[TASK]], %[[TASKS]] : index
// CHECK-NEXT: scf.if %[[IN_RANGE]] {
// The first field of the block's merge record names the output segment.
// CHECK-NEXT: %[[FIELDS:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[RECORD:.*]] = arith.muli %[[TASK]], %[[FIELDS]] : index
// CHECK-NEXT: %[[RECORD_I64:.*]] = arith.index_cast %[[RECORD]] : index to i64
// CHECK-NEXT: %[[SEGMENT_ADDRESS:.*]] = llvm.getelementptr %[[RECORDS]][%[[RECORD_I64]]]
// CHECK-NEXT: %[[SEGMENT_I32:.*]] = llvm.load %[[SEGMENT_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[SEGMENT:.*]] = arith.index_cast %[[SEGMENT_I32]] : i32 to index
// CHECK-NOT: scf.if
// CHECK:   %[[IDENTITY:.*]] = arith.constant 0xFF800000 : f32
// CHECK-NEXT: %[[LOCAL:.*]] = scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %[[BLOCK]] iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK-NEXT:   %[[INDEX:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK-NEXT:   %[[ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[INDEX]]]
// CHECK-NEXT:   %[[PARTIAL:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// The loaded partial is the combine operand itself; nothing transforms it.
// CHECK-NEXT:   %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[PARTIAL]] : f32
// CHECK-NEXT:   scf.yield %[[NEXT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: %[[TOTAL:.*]] = gpu.all_reduce maximumf %[[LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32
// Thread zero is the only writer of the segment result.
// CHECK-NEXT: %[[FIRST_THREAD:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index
// CHECK-NEXT: scf.if %[[FIRST_THREAD]] {
// CHECK-NEXT:   %[[SEGMENT_I64:.*]] = arith.index_cast %[[SEGMENT]] : index to i64
// CHECK-NEXT:   %[[OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SEGMENT_I64]]]
// CHECK-NEXT:   llvm.store %[[TOTAL]], %[[OUTPUT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return
