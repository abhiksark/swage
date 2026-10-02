// test/Conversion/SwageToGPU/split-partial.mlir
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// The partial stage of a split reduction. Each block reduces one planned
// input range and writes one scratch slot. The element program (a fused map
// followed by the reduction region) runs here, on input values, and nowhere
// else in the split schedule; split-merge.mlir lowers the same module and
// checks that the merge stage does not run it again.
//
// Every line from the range guard to the return is matched in order with
// captured operands, so the data path is pinned end to end: record address,
// loaded range, bounded range, loop bounds, combine, block total, stored
// slot.
module {
  func.func @segmented_max(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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
// CHECK: gpu.func @segmented_max__partial(%[[VALUES:[^,]+]]: !llvm.ptr, %[[RANGES:[^,]+]]: !llvm.ptr, %[[SCRATCH:[^,]+]]: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %[[PARTIAL_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 512, 1, 1>
// CHECK: %[[TASK:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BLOCK:.*]] = arith.constant 512 : index
// CHECK: %[[TASKS:.*]] = arith.index_cast %[[PARTIAL_COUNT]] : i32 to index

// The range guard depends only on the block index and a launch argument, so
// it is block-uniform. Everything up to the all-reduce, whose barriers every
// thread of the block must reach, sits directly under it.
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[TASK]], %[[TASKS]] : index
// CHECK-NEXT: scf.if %[[IN_RANGE]] {

// The block's record is the [begin, end) pair at twice its task index in the
// planned ranges.
// CHECK-NEXT: %[[FIELDS:.*]] = arith.constant 2 : index
// CHECK-NEXT: %[[RECORD:.*]] = arith.muli %[[TASK]], %[[FIELDS]] : index
// CHECK-NEXT: %[[BEGIN_FIELD:.*]] = arith.index_cast %[[RECORD]] : index to i64
// CHECK-NEXT: %[[BEGIN_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]][%[[BEGIN_FIELD]]]
// CHECK-NEXT: %[[BEGIN_WORD:.*]] = llvm.load %[[BEGIN_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[END_INDEX:.*]] = arith.addi %[[RECORD]], %[[ONE]] : index
// CHECK-NEXT: %[[END_FIELD:.*]] = arith.index_cast %[[END_INDEX]] : index to i64
// CHECK-NEXT: %[[END_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]][%[[END_FIELD]]]
// CHECK-NEXT: %[[END_WORD:.*]] = llvm.load %[[END_ADDRESS]] : !llvm.ptr -> i32

// The range indexes values, so both words are bounded by the value count,
// and the end is floored by the bounded begin.
// CHECK-NEXT: %[[ZERO_I32:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[BEGIN_FLOOR:.*]] = arith.maxsi %[[BEGIN_WORD]], %[[ZERO_I32]] : i32
// CHECK-NEXT: %[[BEGIN_BOUND:.*]] = arith.minsi %[[BEGIN_FLOOR]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[END_FLOOR:.*]] = arith.maxsi %[[END_WORD]], %[[BEGIN_BOUND]] : i32
// CHECK-NEXT: %[[END_BOUND:.*]] = arith.minsi %[[END_FLOOR]], %[[VALUE_COUNT]] : i32
// CHECK-NEXT: %[[BEGIN:.*]] = arith.index_cast %[[BEGIN_BOUND]] : i32 to index
// CHECK-NEXT: %[[END:.*]] = arith.index_cast %[[END_BOUND]] : i32 to index

// Thread t reduces begin + t, begin + t + 512, ... up to the end, so the 512
// threads cover the range exactly once.
// CHECK-NEXT: %[[FIRST:.*]] = arith.addi %[[BEGIN]], %[[THREAD]] : index
// CHECK-NEXT: %[[IDENTITY:.*]] = arith.constant 0xFF800000 : f32
// CHECK-NEXT: %[[LOCAL:.*]] = scf.for %[[I:.*]] = %[[FIRST]] to %[[END]] step %[[BLOCK]] iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
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
