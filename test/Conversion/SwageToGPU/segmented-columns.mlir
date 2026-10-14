// test/Conversion/SwageToGPU/segmented-columns.mlir
// The column kernel of rank-two values: one block per segment, and thread t
// reduces the columns t, t + 128, and so on of the rows of its segment, one
// after the other. The task-ids and split schedules give the row-stripe
// tile instead, at the end of the file.
//
// - The rows of the segment are clamped to the row count, and the column
//   loop is bounded by the feature count, so every load stays inside the
//   values and every store inside the row of its segment.
// - A column is the strided run that starts at start * columns + column and
//   takes every columns-th element below end * columns.
// - A thread holds one scalar accumulator per reduction, whatever the number
//   of columns: the accumulator is the one iteration argument of the row
//   loop.
// - Each thread stores its own results. Nothing is combined across threads,
//   so the kernel holds no shuffle, no block reduction, no barrier, and no
//   condition on the thread index.
//
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=gpu.shuffle \
// RUN:       --implicit-check-not=gpu.all_reduce \
// RUN:       --implicit-check-not=gpu.barrier --implicit-check-not=swage. \
// RUN:       --implicit-check-not=memref
// RUN: swage-opt \
// RUN:   --pass-pipeline='builtin.module(swage-to-plan{schedule=direct block-threads=128},swage-plan-to-gpu,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' %s \
// RUN:   | FileCheck %s --check-prefix=NVVM --implicit-check-not=nvvm.shfl \
// RUN:       --implicit-check-not=nvvm.barrier
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=STRIPES --implicit-check-not=swage. \
// RUN:       --implicit-check-not=gpu.all_reduce
// RUN: swage-opt --swage-to-plan='schedule=split-partial' --swage-plan-to-gpu \
// RUN:   %s | FileCheck %s --check-prefix=ROW-PARTIAL \
// RUN:       --implicit-check-not=swage. --implicit-check-not=arith.divf
// RUN: swage-opt --swage-to-plan='schedule=split-merge' --swage-plan-to-gpu \
// RUN:   %s | FileCheck %s --check-prefix=ROW-MERGE --implicit-check-not=swage.

module {
  func.func @segmented_sum_r2(
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
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid, %col] : memref<?x?xf32>
    return
  }
  func.func @segmented_max_r2(
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
    %result = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid, %col] : memref<?x?xf32>
    return
  }
  func.func @segmented_mean_f64_r2(
      %values: memref<?x?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf64>, memref<?xi32>, index, index
          -> !swage.segment<f64>
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    %extent = swage.extent %segment : !swage.segment<f64>
    %count = arith.index_cast %extent : index to i32
    %divisor = arith.sitofp %count : i32 to f64
    %mean = arith.divf %result, %divisor : f64
    memref.store %mean, %output[%sid, %col] : memref<?x?xf64>
    return
  }
}

// CHECK: gpu.func @segmented_sum_r2(%[[VALUES:.*]]: !llvm.ptr, %[[OFFSETS:.*]]: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32, %[[FEATURE_COUNT:.*]]: i32) kernel attributes {nvvm.reqntid = array<i32: 128, 1, 1>,
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 3 : i64}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_sum_r2", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}} {
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
// CHECK:   %[[OUTPUT_ROW:.*]] = arith.muli %[[SEGMENT]], %[[COLUMNS]] : index
// The columns of this thread, bounded by the feature count.
// CHECK:   scf.for %[[COLUMN:.*]] = %[[THREAD]] to %[[COLUMNS]] step %[[BLOCK]] {
// CHECK:     %[[FIRST:.*]] = arith.addi %[[FIRST_ROW]], %[[COLUMN]] : index
// CHECK:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:     %[[SUM:.*]] = scf.for %[[INDEX:.*]] = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f32) {
// CHECK:       %[[INDEX64:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK:       %[[ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[INDEX64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK:       %[[VALUE:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// CHECK:       %[[NEXT:.*]] = arith.addf %[[ACC]], %[[VALUE]] : f32
// CHECK:       scf.yield %[[NEXT]] : f32
// CHECK:     }
// Every thread stores the results of its own columns.
// CHECK-NEXT: %[[SLOT:.*]] = arith.addi %[[OUTPUT_ROW]], %[[COLUMN]] : index
// CHECK-NEXT: %[[SLOT64:.*]] = arith.index_cast %[[SLOT]] : index to i64
// CHECK-NEXT: %[[SLOT_ADDRESS:.*]] = llvm.getelementptr %[[OUTPUT]][%[[SLOT64]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// CHECK-NEXT: llvm.store %[[SUM]], %[[SLOT_ADDRESS]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

// CHECK: gpu.func @segmented_max_r2(
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 3 : i64}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_max_r2", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK: arith.constant 0xFF800000 : f32
// CHECK: arith.maximumf

// A mean of a column divides by the rows of its segment, in every thread.
// CHECK: gpu.func @segmented_mean_f64_r2(
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 3 : i64}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_mean_f64_r2", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK:   %[[MEAN_START_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// CHECK:   %[[MEAN_END_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: %[[MEAN_START:.*]] = arith.index_cast %[[MEAN_START_CLAMPED]] : i32 to index
// CHECK-NEXT: %[[MEAN_END:.*]] = arith.index_cast %[[MEAN_END_CLAMPED]] : i32 to index
// CHECK:   %[[ROWS:.*]] = arith.subi %[[MEAN_END]], %[[MEAN_START]] : index
// CHECK:   scf.for
// CHECK:     scf.for {{.*}} -> (f64) {
// CHECK:     %[[COUNT:.*]] = arith.index_cast %[[ROWS]] : index to i32
// CHECK-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f64
// CHECK-NEXT: %[[MEAN:.*]] = arith.divf %{{.*}}, %[[DIVISOR]] : f64
// CHECK:     llvm.store %[[MEAN]], %{{.*}} : f64, !llvm.ptr

// The upstream conversion finds nothing to synchronize.
// NVVM: llvm.func @segmented_sum_r2(%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32)
// NVVM-SAME: attributes {gpu.kernel, nvvm.kernel, nvvm.reqntid = array<i32: 128, 1, 1>, swage.kernel_contract = {{.+}}}
// NVVM: llvm.return

// The task-ids schedule gives the row-stripe tile of the same program: the
// task buffer and the number of columns as parameters, one exchange buffer
// of 128 elements, the loop over the items of a block, and five shuffles
// and two barriers per reduction stage, which
// test/Conversion/SwagePlanToGPU/column-groups.mlir pins in detail.
// STRIPES: gpu.func @segmented_sum_r2(%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32) workgroup(%{{[^ ]+}} : memref<128xf32, #gpu.address_space<workgroup>>) kernel
// STRIPES-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "task_ids", kind = "ptr", origin = "plan"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "task_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_sum_r2", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// STRIPES: scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} {{[{]$}}
// STRIPES-COUNT-5: gpu.shuffle xor
// STRIPES-COUNT-2: gpu.barrier
// STRIPES: llvm.store
// STRIPES: gpu.func @segmented_max_r2(
// STRIPES-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "task_ids", kind = "ptr", origin = "plan"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "task_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_max_r2", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// STRIPES: gpu.func @segmented_mean_f64_r2(
// STRIPES-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "task_ids", kind = "ptr", origin = "plan"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "task_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_mean_f64_r2", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}

// The split schedules give the same tile at the split width of 512 threads.
// A partial task stores the raw reduction of each column in its scratch
// row, so the partial kernel of the mean divides nothing.
// ROW-PARTIAL: gpu.func @segmented_sum_r2__partial(%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %[[SCRATCH:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32) workgroup(%{{[^ ]+}} : memref<512xf32, #gpu.address_space<workgroup>>) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>, swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {access = "write", key = "scratch", kind = "ptr", origin = "scratch"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "partial_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_sum_r2__partial", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}}
// ROW-PARTIAL-COUNT-5: gpu.shuffle xor
// ROW-PARTIAL-COUNT-2: gpu.barrier
// ROW-PARTIAL: llvm.getelementptr %[[SCRATCH]][
// ROW-PARTIAL-NEXT: llvm.store
// ROW-PARTIAL: gpu.func @segmented_max_r2__partial(
// ROW-PARTIAL-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {access = "write", key = "scratch", kind = "ptr", origin = "scratch"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "partial_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_max_r2__partial", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// ROW-PARTIAL: gpu.func @segmented_mean_f64_r2__partial(
// ROW-PARTIAL-SAME: workgroup(%{{[^ ]+}} : memref<512xf64, #gpu.address_space<workgroup>>)
// ROW-PARTIAL-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {access = "write", key = "scratch", kind = "ptr", origin = "scratch"}, {kind = "i32", origin = "user", source_index = 3 : i64}, {key = "partial_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_mean_f64_r2__partial", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}

// The merge of a mean reads the extent of each split segment from the range
// records, a number of rows, and divides once per column after the
// combination.
// ROW-MERGE: gpu.func @segmented_sum_r2__merge(
// ROW-MERGE-SAME: workgroup(%{{[^ ]+}} : memref<512xf32, #gpu.address_space<workgroup>>)
// ROW-MERGE-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_sum_r2__merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// ROW-MERGE-NOT: arith.divf
// ROW-MERGE: gpu.func @segmented_max_r2__merge(
// ROW-MERGE-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_max_r2__merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// ROW-MERGE: gpu.func @segmented_mean_f64_r2__merge(%{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %{{[^:]+}}: !llvm.ptr, %[[RANGES:[^:]+]]: !llvm.ptr, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32, %{{[^:]+}}: i32) workgroup(%{{[^ ]+}} : memref<512xf64, #gpu.address_space<workgroup>>)
// ROW-MERGE-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}, {kind = "i32", origin = "user", source_index = 5 : i64}], backend = "cuda", entry = "segmented_mean_f64_r2__merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// ROW-MERGE: %[[ROWS:.*]] = scf.if %{{.*}} -> (index) {
// ROW-MERGE: llvm.getelementptr %[[RANGES]][
// ROW-MERGE: llvm.getelementptr %[[RANGES]][
// ROW-MERGE-COUNT-5: gpu.shuffle xor
// ROW-MERGE-COUNT-2: gpu.barrier
// ROW-MERGE-NEXT: %[[COUNT:.*]] = arith.index_cast %[[ROWS]] : index to i32
// ROW-MERGE-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f64
// ROW-MERGE-NEXT: arith.divf %{{.*}}, %[[DIVISOR]] : f64
