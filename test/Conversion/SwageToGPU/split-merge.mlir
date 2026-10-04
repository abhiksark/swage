// test/Conversion/SwageToGPU/split-merge.mlir
// RUN: swage-opt --swage-to-plan='schedule=split-merge' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=swage. \
// RUN:       --implicit-check-not=arith.mulf
// RUN: swage-opt --swage-to-plan='schedule=split-merge' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=RANGE

// The merge stage of a split reduction, lowered from the same module as
// split-partial.mlir. Each block combines one compact scratch range and one
// thread writes the segment result. Scratch already holds reduced partials,
// so the merge must combine them with the reduction kind alone: running the
// element program again would double and square every partial a second time.
// Both element operations are arith.mulf, which the first RUN line forbids
// anywhere in this kernel.
//
// The data path is pinned by two runs over the same output. The first follows
// the kernel in order: output segment, loop, combine, block total, store. It
// takes the loop bounds from the two index casts above the loop and leaves
// their operands open. The second (RANGE) follows the scratch range forward
// from the merge record: each loaded word must be used by an index cast or
// an integer min or max, and that result must feed the loop or another min or
// max. Together they hold the loaded range to the loop bounds whether or not
// a bound sits between them, and reject a bound taken from anywhere else.
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

// CHECK: gpu.module @segmented_max__merge_module
// CHECK: gpu.func @segmented_max__merge(%[[SCRATCH:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[RECORDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: i32, %[[MERGE_COUNT:[^,]+]]: i32, %[[SEGMENT_COUNT:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 512, 1, 1>
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_max__merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
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
// The first field of the block's merge record names the output segment. It
// is compared with the segment count as loaded; the comparison yields a
// value for the store predicate and opens no branch here.
// CHECK-NEXT: %[[FIELDS:.*]] = arith.constant 3 : index
// CHECK-NEXT: %[[RECORD:.*]] = arith.muli %[[TASK]], %[[FIELDS]] : index
// CHECK-NEXT: %[[SEGMENT_FIELD:.*]] = arith.index_cast %[[RECORD]] : index to i64
// CHECK-NEXT: %[[SEGMENT_ADDRESS:.*]] = llvm.getelementptr %[[RECORDS]][%[[SEGMENT_FIELD]]]
// CHECK-NEXT: %[[SEGMENT_WORD:.*]] = llvm.load %[[SEGMENT_ADDRESS]] : !llvm.ptr -> i32
// CHECK-NEXT: %[[SEGMENT_IN_RANGE:.*]] = arith.cmpi ult, %[[SEGMENT_WORD]], %[[SEGMENT_COUNT]] : i32
// CHECK-NEXT: %[[SEGMENT:.*]] = arith.index_cast %[[SEGMENT_WORD]] : i32 to index
// The next two casts to index are the scratch range; RANGE pins what they
// cast. Thread t reduces begin + t, begin + t + 512, ... up to the end.
// CHECK-NOT: scf.if
// CHECK: %[[BEGIN:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NOT: scf.if
// CHECK: %[[END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT: %[[FIRST:.*]] = arith.addi %[[BEGIN]], %[[THREAD]] : index
// CHECK-NEXT: %[[IDENTITY:.*]] = arith.constant 0xFF800000 : f32
// CHECK-NEXT: %[[LOCAL:.*]] = scf.for %[[I:.*]] = %[[FIRST]] to %[[END]] step %[[BLOCK]] iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK-NEXT:   %[[INDEX:.*]] = arith.index_cast %[[I]] : index to i64
// CHECK-NEXT:   %[[ADDRESS:.*]] = llvm.getelementptr %[[SCRATCH]][%[[INDEX]]]
// CHECK-NEXT:   %[[PARTIAL:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// The loaded partial is the combine operand itself; nothing transforms it.
// CHECK-NEXT:   %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[PARTIAL]] : f32
// CHECK-NEXT:   scf.yield %[[NEXT]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: %[[TOTAL:.*]] = gpu.all_reduce maximumf %[[LOCAL]] uniform {
// CHECK-NEXT: } : (f32) -> f32
// Thread zero is the only writer, and it stores the block total at the
// segment the record names, unless that segment is out of range.
// CHECK-NEXT: %[[FIRST_THREAD:.*]] = arith.cmpi eq, %[[THREAD]], %[[ZERO]] : index
// CHECK-NEXT: %[[MAY_STORE:.*]] = arith.andi %[[FIRST_THREAD]], %[[SEGMENT_IN_RANGE]] : i1
// CHECK-NEXT: scf.if %[[MAY_STORE]] {
// CHECK-NEXT:   %[[SEGMENT_I64:.*]] = arith.index_cast %[[SEGMENT]] : index to i64
// CHECK-NEXT:   %[[OUTPUT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SEGMENT_I64]]]
// CHECK-NEXT:   llvm.store %[[TOTAL]], %[[OUTPUT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// The scratch range is fields one and two of the same record.
// RANGE: gpu.func @segmented_max__merge(%{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[RECORDS:[^,]+]]: !llvm.ptr,
// RANGE: %[[TASK:.*]] = gpu.block_id x
// RANGE: %[[FIELDS:.*]] = arith.constant 3 : index
// RANGE: %[[RECORD:.*]] = arith.muli %[[TASK]], %[[FIELDS]] : index
// RANGE: %[[ONE:.*]] = arith.constant 1 : index
// RANGE: %[[BEGIN_INDEX:.*]] = arith.addi %[[RECORD]], %[[ONE]] : index
// RANGE: %[[BEGIN_FIELD:.*]] = arith.index_cast %[[BEGIN_INDEX]] : index to i64
// RANGE: %[[BEGIN_ADDRESS:.*]] = llvm.getelementptr %[[RECORDS]][%[[BEGIN_FIELD]]]
// RANGE: %[[BEGIN_WORD:.*]] = llvm.load %[[BEGIN_ADDRESS]] : !llvm.ptr -> i32
// The begin word goes to an index cast or a bound. The end word is loaded
// either side of that, depending on whether a bound follows.
// RANGE-DAG: %[[BEGIN_USE:[^ ]+]] = arith.{{index_cast|(max|min)[su]i}} %[[BEGIN_WORD]]{{ : i32 to index|, }}
// RANGE-DAG: %[[TWO:[^ ]+]] = arith.constant 2 : index
// RANGE-DAG: %[[END_INDEX:[^ ]+]] = arith.addi %[[RECORD]], %[[TWO]] : index
// RANGE-DAG: %[[END_FIELD:[^ ]+]] = arith.index_cast %[[END_INDEX]] : index to i64
// RANGE-DAG: %[[END_ADDRESS:[^ ]+]] = llvm.getelementptr %[[RECORDS]][%[[END_FIELD]]]
// RANGE-DAG: %[[END_WORD:[^ ]+]] = llvm.load %[[END_ADDRESS]] : !llvm.ptr -> i32
// The end word is used after the begin word, never in its place. It goes to an
// index cast or a bound too. From there the begin goes to the loop's first
// index and the end to the loop's upper bound, or each to a further bound.
// RANGE-NOT: scf.if
// RANGE-DAG: %[[END_USE:[^ ]+]] = arith.{{index_cast|(max|min)[su]i}} %[[END_WORD]]{{ : i32 to index|, }}
// RANGE-DAG: arith.{{addi|(max|min)[su]i}} %[[BEGIN_USE]], %
// RANGE-DAG: {{ to|arith.(max|min)[su]i}} %[[END_USE]]{{ step|, }}
// RANGE: gpu.all_reduce maximumf
