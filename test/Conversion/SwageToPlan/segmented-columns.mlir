// test/Conversion/SwageToPlan/segmented-columns.mlir
// A function over rank-two values reduces one column of one segment per
// program instance. The planner absorbs both segment ids and the column of
// make_segment: the task operation takes the number of columns, and its
// region binds one column of the rows of a segment, a segment of scalars.
//
// The direct schedule plans the column kernel, whose plan function takes the
// number of columns after its two other counts. Every other kernel schedule
// needs a task buffer, which rank-two values do not have.
//
// RUN: swage-opt --swage-to-plan='schedule=direct' %s \
// RUN:   | FileCheck %s --check-prefix=COLUMN --implicit-check-not=segment_id \
// RUN:       --implicit-check-not=make_segment
// RUN: swage-opt --swage-to-plan='schedule=sequential' %s \
// RUN:   | FileCheck %s --check-prefix=SEQUENTIAL \
// RUN:       --implicit-check-not=segment_id --implicit-check-not=make_segment
// RUN: not swage-opt --swage-to-plan='schedule=task-ids' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=split-partial' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=split-merge' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED

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

// REFUSED: error: planning requires rank-one values: a function over rank-two values has one kernel, the direct schedule, and no task buffer
