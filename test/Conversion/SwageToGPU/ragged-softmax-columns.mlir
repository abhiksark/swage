// test/Conversion/SwageToGPU/ragged-softmax-columns.mlir
// The column kernel of the softmax over rank-two values: one block per
// segment, and thread t normalizes the columns t, t + 128, and so on of the
// rows of its segment, one after the other.
//
// - The three stages of the softmax run in the thread of the column: the
//   maximum, the sum of the shifted exponentials, and the store. Each is a
//   loop over the same strided run of the column, in row order.
// - A thread holds one scalar per reduction stage, the iteration argument of
//   its loop, whatever the number of columns and of rows.
// - The store writes the element it read from: the flat index of the row and
//   column in the values is the index in the output. The rows are clamped to
//   the row count and the columns bounded by the feature count, so a store
//   stays inside rows and columns that the output has.
// - Nothing is combined across threads. The rank-one softmax needs two block
//   reductions for the same program; this kernel holds no shuffle, no block
//   reduction, no barrier, and no workgroup buffer.
//
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=gpu.shuffle \
// RUN:       --implicit-check-not=gpu.all_reduce \
// RUN:       --implicit-check-not=gpu.barrier --implicit-check-not=swage. \
// RUN:       --implicit-check-not=memref --implicit-check-not=workgroup
// RUN: swage-opt \
// RUN:   --pass-pipeline='builtin.module(swage-to-plan{schedule=direct block-threads=128},swage-plan-to-gpu,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' %s \
// RUN:   | FileCheck %s --check-prefix=NVVM --implicit-check-not=nvvm.shfl \
// RUN:       --implicit-check-not=nvvm.barrier
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=STRIPES --implicit-check-not=swage. \
// RUN:       --implicit-check-not=gpu.all_reduce

module {
  func.func @ragged_softmax_r2(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index
          -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?x?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }
}

// CHECK: gpu.func @ragged_softmax_r2(%[[VALUES:.*]]: !llvm.ptr, %[[OFFSETS:.*]]: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32, %[[FEATURE_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 128, 1, 1>} {
// CHECK: %[[SEGMENT:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: %[[BLOCK:.*]] = arith.constant 128 : index
// CHECK: %[[SEGMENTS:.*]] = arith.index_cast %[[SEGMENT_COUNT]] : i32 to index
// CHECK: %[[IN_RANGE:.*]] = arith.cmpi slt, %[[SEGMENT]], %[[SEGMENTS]] : index
// CHECK: scf.if %[[IN_RANGE]] {
// The rows of the segment, clamped to the row count.
// CHECK:   %[[START_FLOOR:.*]] = arith.maxsi %{{.*}}, %{{.*}} : i32
// CHECK:   %[[START_CLAMPED:.*]] = arith.minsi %[[START_FLOOR]], %[[VALUE_COUNT]] : i32
// CHECK:   %[[END_FLOOR:.*]] = arith.maxsi %{{.*}}, %[[START_CLAMPED]] : i32
// CHECK:   %[[END_CLAMPED:.*]] = arith.minsi %[[END_FLOOR]], %[[VALUE_COUNT]] : i32
// CHECK:   %[[START:.*]] = arith.index_cast %[[START_CLAMPED]] : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %[[END_CLAMPED]] : i32 to index
// CHECK:   %[[COLUMNS:.*]] = arith.index_cast %[[FEATURE_COUNT]] : i32 to index
// CHECK:   %[[FIRST_ROW:.*]] = arith.muli %[[START]], %[[COLUMNS]] : index
// CHECK:   %[[LAST:.*]] = arith.muli %[[END]], %[[COLUMNS]] : index
// The columns of this thread, bounded by the feature count.
// CHECK:   scf.for %[[COLUMN:.*]] = %[[THREAD]] to %[[COLUMNS]] step %[[BLOCK]] {
// CHECK:     %[[FIRST:.*]] = arith.addi %[[FIRST_ROW]], %[[COLUMN]] : index
// Stage one: the maximum of the column, in row order.
// CHECK:     %[[MAX:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] iter_args(%{{.*}} = %{{.*}}) -> (f32) {
// CHECK:       arith.maximumf
// CHECK:     }
// Stage two: the sum of the shifted exponentials, over the same run.
// CHECK:     %[[TOTAL:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK:       %[[SUM_CENTERED:.*]] = arith.subf %{{.*}}, %[[MAX]] : f32
// CHECK:       %[[SUM_SCALED:.*]] = arith.mulf %[[SUM_CENTERED]], %{{.*}} : f32
// CHECK:       %[[SUM_EXPONENTIAL:.*]] = math.exp2 %[[SUM_SCALED]] : f32
// CHECK:       arith.addf %[[ACC]], %[[SUM_EXPONENTIAL]] : f32
// CHECK:     }
// Stage three: every element is stored at the index it was loaded from.
// CHECK:     scf.for %[[INDEX:.*]] = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] {
// CHECK-NEXT:  %[[INDEX64:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK-NEXT:  %[[ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[INDEX64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:  %[[VALUE:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// CHECK:       %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[MAX]] : f32
// CHECK:       %[[EXPONENTIAL:.*]] = math.exp2 %{{.*}} : f32
// CHECK-NEXT:  %[[NORMALIZED:.*]] = arith.divf %[[EXPONENTIAL]], %[[TOTAL]] : f32
// CHECK-NEXT:  %[[SLOT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[INDEX64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT:  llvm.store %[[NORMALIZED]], %[[SLOT]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// The upstream conversion finds nothing to synchronize.
// NVVM: llvm.func @ragged_softmax_r2(%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// NVVM-SAME: attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 128, 1, 1>}
// NVVM: llvm.return

// The task-ids schedule gives the row-stripe tile of the same program: the
// task buffer and the number of columns as parameters, one exchange buffer
// of 128 elements, the loop over the items of a block, and five shuffles
// and two barriers per reduction stage, which
// test/Conversion/SwagePlanToGPU/column-groups.mlir pins in detail.
// STRIPES: gpu.func @ragged_softmax_r2(%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32) workgroup(%{{[^ ]+}} : memref<128xf32, #gpu.address_space<workgroup>>) kernel
// STRIPES: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {{[{]$}}
// STRIPES-COUNT-5: gpu.shuffle xor
// STRIPES-COUNT-2: gpu.barrier
// STRIPES-COUNT-5: gpu.shuffle xor
// STRIPES-COUNT-2: gpu.barrier
// STRIPES: llvm.store
