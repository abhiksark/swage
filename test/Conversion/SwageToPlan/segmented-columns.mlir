// test/Conversion/SwageToPlan/segmented-columns.mlir
// A function over rank-two values reduces one column of one segment per
// program instance. The planner absorbs both segment ids and the column of
// make_segment: the task operation takes the number of columns, and its
// region binds one column of the rows of a segment, a segment of scalars.
//
// The direct schedule plans the column kernel, whose plan function takes the
// number of columns after its two other counts. The task-ids schedule plans
// the row-stripe kernel: policy<cta> with the task buffer, and the number of
// columns after the counts of the task-id kernel. Its block is a whole
// number of subgroups. The split schedules plan the row-stripe kernels of
// the split, which take the number of columns last, with scratch rows of
// that many columns. The other kernel schedules refuse rank-two values by
// name.
//
// RUN: swage-opt --swage-to-plan='schedule=direct' %s \
// RUN:   | FileCheck %s --check-prefix=COLUMN --implicit-check-not=segment_id \
// RUN:       --implicit-check-not=make_segment
// RUN: swage-opt --swage-to-plan='schedule=task-ids' %s \
// RUN:   | FileCheck %s --check-prefix=STRIPES --implicit-check-not=segment_id \
// RUN:       --implicit-check-not=make_segment
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' %s \
// RUN:   | FileCheck %s --check-prefix=ONE-SUBGROUP
// RUN: not swage-opt --swage-to-plan='schedule=task-ids block-threads=48' %s \
// RUN:   2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: swage-opt --swage-to-plan='schedule=sequential' %s \
// RUN:   | FileCheck %s --check-prefix=SEQUENTIAL \
// RUN:       --implicit-check-not=segment_id --implicit-check-not=make_segment
// RUN: not swage-opt --swage-to-plan='schedule=fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FUSED
// RUN: swage-opt --swage-to-plan='schedule=split-partial' %s \
// RUN:   | FileCheck %s --check-prefix=PARTIAL
// RUN: swage-opt --swage-to-plan='schedule=split-merge' %s \
// RUN:   | FileCheck %s --check-prefix=MERGE
// RUN: not swage-opt --swage-to-plan='schedule=persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERSISTENT

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

// COLUMN: func.func @segmented_sum_r2(%[[VALUES:.*]]: memref<?x?xf32> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %[[OFFSETS:.*]]: memref<?xi32> {swage.role = #swage.role<offsets>, swage_plan.source_index = 1 : i32}, %[[OUTPUT:.*]]: memref<?x?xf32> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %[[VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>, swage_plan.source_index = 3 : i32}, %[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 4 : i32}, %[[FEATURE_COUNT:.*]]: i32 {swage.role = #swage.role<feature_count>, swage_plan.source_index = 5 : i32}) attributes {swage_plan.block_threads = 128 : i32} {
// COLUMN-NEXT: swage_plan.tasks policy<column> segments(%[[VALUES]], %[[OFFSETS]] : memref<?x?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) feature_count(%[[FEATURE_COUNT]] : i32) into(%[[OUTPUT]] : memref<?x?xf32>) {
// COLUMN-NEXT: ^bb0(%[[COLUMN:.*]]: !swage.segment<f32>):
// COLUMN-NEXT: %[[SUM:.*]] = swage.reduce %[[COLUMN]] kind<sum> : !swage.segment<f32> -> f32 {
// COLUMN: swage_plan.yield %[[SUM]] : f32

// COLUMN: func.func @segmented_max_r2(
// COLUMN: swage_plan.tasks policy<column>
// COLUMN: swage.reduce %{{.*}} kind<max> : !swage.segment<f32> -> f32 {

// A mean of a column divides by the extent of its segment, a number of rows.
// COLUMN: func.func @segmented_mean_f64_r2(
// COLUMN: swage_plan.tasks policy<column>
// COLUMN: ^bb0(%[[MEAN_COLUMN:.*]]: !swage.segment<f64>, %[[ROWS:.*]]: index):
// COLUMN: arith.index_cast %[[ROWS]] : index to i32
// COLUMN: %[[MEAN:.*]] = arith.divf
// COLUMN-NEXT: swage_plan.yield %[[MEAN]] : f64

// The oracle keeps its function and its roles.
// SEQUENTIAL: func.func @segmented_sum_r2(
// SEQUENTIAL: swage_plan.tasks policy<sequential> segments(%{{.*}}, %{{.*}} : memref<?x?xf32>, memref<?xi32>) value_count(%{{.*}} : i32) segment_count(%{{.*}} : i32) feature_count(%{{.*}} : i32) into(%{{.*}} : memref<?x?xf32>) {

// The row-stripe kernel takes the task buffer and the number of columns
// after the counts of the task-id kernel, and its block combines per column.
// STRIPES: func.func @segmented_sum_r2(%[[R_VALUES:.*]]: memref<?x?xf32> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %[[R_OFFSETS:.*]]: memref<?xi32> {swage.role = #swage.role<offsets>, swage_plan.source_index = 1 : i32}, %[[R_OUTPUT:.*]]: memref<?x?xf32> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %[[R_IDS:.*]]: memref<?xi32>, %[[R_VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>, swage_plan.source_index = 3 : i32}, %[[R_TASK_COUNT:.*]]: i32, %[[R_SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 4 : i32}, %[[R_FEATURE_COUNT:.*]]: i32 {swage.role = #swage.role<feature_count>, swage_plan.source_index = 5 : i32}) attributes {swage_plan.block_threads = 128 : i32} {
// STRIPES-NEXT: swage_plan.tasks policy<cta> segments(%[[R_VALUES]], %[[R_OFFSETS]] : memref<?x?xf32>, memref<?xi32>) value_count(%[[R_VALUE_COUNT]] : i32) segment_count(%[[R_SEGMENT_COUNT]] : i32) feature_count(%[[R_FEATURE_COUNT]] : i32) ids(%[[R_IDS]] : memref<?xi32>) task_count(%[[R_TASK_COUNT]] : i32) into(%[[R_OUTPUT]] : memref<?x?xf32>) {
// STRIPES-NEXT: ^bb0(%[[R_COLUMN:.*]]: !swage.segment<f32>):
// STRIPES-NEXT: %[[R_SUM:.*]] = swage.reduce %[[R_COLUMN]] kind<sum> : !swage.segment<f32> -> f32 {
// STRIPES: swage_plan.yield %[[R_SUM]] : f32
// STRIPES: func.func @segmented_max_r2(
// STRIPES: swage_plan.tasks policy<cta>
// STRIPES: func.func @segmented_mean_f64_r2(
// STRIPES: swage_plan.tasks policy<cta>
// STRIPES: ^bb0(%{{.*}}: !swage.segment<f64>, %{{.*}}: index):

// A block of one subgroup is still a block task over rank-two values: the
// warp policy takes scalars only.
// ONE-SUBGROUP: attributes {swage_plan.block_threads = 32 : i32} {
// ONE-SUBGROUP-NEXT: swage_plan.tasks policy<cta>

// WIDTH: error: the task-ids kernel of rank-two values runs whole subgroups of 32 threads, so block-threads must be a multiple of 32, got 48

// A partial task reduces a range of rows into one scratch row of
// feature_count columns.
// PARTIAL: func.func @segmented_sum_r2__partial(%[[P_VALUES:.*]]: memref<?x?xf32> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %[[P_RANGES:.*]]: memref<?xi32>, %[[P_SCRATCH:.*]]: memref<?x?xf32>, %[[P_VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>, swage_plan.source_index = 3 : i32}, %[[P_PARTIAL_COUNT:.*]]: i32, %[[P_FEATURE_COUNT:.*]]: i32 {swage.role = #swage.role<feature_count>, swage_plan.source_index = 5 : i32}) attributes {swage_plan.block_threads = 512 : i32} {
// PARTIAL-NEXT: swage_plan.partial_tasks values(%[[P_VALUES]] : memref<?x?xf32>) value_count(%[[P_VALUE_COUNT]] : i32) ranges(%[[P_RANGES]] : memref<?xi32>) partial_count(%[[P_PARTIAL_COUNT]] : i32) feature_count(%[[P_FEATURE_COUNT]] : i32) into(%[[P_SCRATCH]] : memref<?x?xf32>) {
// PARTIAL: swage.reduce %{{.*}} kind<sum> : !swage.segment<f32> -> f32 {
// PARTIAL: func.func @segmented_max_r2__partial(
// PARTIAL: swage.reduce %{{.*}} kind<max> : !swage.segment<f32> -> f32 {
// The partial task of a mean yields the raw sum; its merge divides.
// PARTIAL: func.func @segmented_mean_f64_r2__partial(
// PARTIAL: feature_count(%{{.*}} : i32) into(%{{.*}} : memref<?x?xf64>) {
// PARTIAL-NEXT: ^bb0(%{{[^,]*}}: !swage.segment<f64>):
// PARTIAL-NOT: arith.divf
// PARTIAL: return

// A merge task reduces scratch rows into the row of its segment. The merge
// of a mean also reads the range records, whose extents are numbers of rows.
// MERGE: func.func @segmented_sum_r2__merge(%[[M_SCRATCH:.*]]: memref<?x?xf32>, %[[M_OUTPUT:.*]]: memref<?x?xf32> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %[[M_MERGES:.*]]: memref<?xi32>, %[[M_PARTIAL_COUNT:.*]]: i32, %[[M_MERGE_COUNT:.*]]: i32, %[[M_SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 4 : i32}, %[[M_FEATURE_COUNT:.*]]: i32 {swage.role = #swage.role<feature_count>, swage_plan.source_index = 5 : i32}) attributes {swage_plan.block_threads = 512 : i32} {
// MERGE-NEXT: swage_plan.merge_tasks scratch(%[[M_SCRATCH]] : memref<?x?xf32>) partial_count(%[[M_PARTIAL_COUNT]] : i32) merges(%[[M_MERGES]] : memref<?xi32>) merge_count(%[[M_MERGE_COUNT]] : i32) segment_count(%[[M_SEGMENT_COUNT]] : i32) feature_count(%[[M_FEATURE_COUNT]] : i32) into(%[[M_OUTPUT]] : memref<?x?xf32>) {
// MERGE: func.func @segmented_max_r2__merge(
// MERGE: swage.reduce %{{.*}} kind<max> : !swage.segment<f32> -> f32 {
// MERGE: func.func @segmented_mean_f64_r2__merge(%{{.*}}: memref<?x?xf64>, %{{.*}}: memref<?x?xf64> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %{{.*}}: memref<?xi32>, %[[E_RANGES:.*]]: memref<?xi32>, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 4 : i32}, %[[E_FEATURE_COUNT:.*]]: i32 {swage.role = #swage.role<feature_count>, swage_plan.source_index = 5 : i32})
// MERGE-NEXT: swage_plan.merge_tasks {{.*}} feature_count(%[[E_FEATURE_COUNT]] : i32) ranges(%[[E_RANGES]] : memref<?xi32>) into(%{{.*}} : memref<?x?xf64>) {
// MERGE-NEXT: ^bb0(%{{.*}}: !swage.segment<f64>, %[[E_ROWS:.*]]: index):
// MERGE: arith.index_cast %[[E_ROWS]] : index to i32
// MERGE: arith.divf

// FUSED: error: fused-mixed planning requires rank-one values: a function over rank-two values runs on the direct, task-ids, split-partial, and split-merge schedules
// PERSISTENT: error: persistent planning requires rank-one values: a function over rank-two values runs on the direct, task-ids, split-partial, and split-merge schedules
