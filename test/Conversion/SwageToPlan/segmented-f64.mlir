// test/Conversion/SwageToPlan/segmented-f64.mlir
// An f64 reduction plans like an f32 one. The scratch of a split holds
// elements, so it is an f64 buffer, and the merge region binds an f64
// segment. The persistent schedule admits f32 values only.
//
// RUN: swage-opt --swage-to-plan='schedule=task-ids' %s \
// RUN:   | FileCheck %s --check-prefix=TASKS --implicit-check-not=f32
// RUN: swage-opt --swage-to-plan='schedule=split-partial,split-merge' %s \
// RUN:   | FileCheck %s --check-prefix=SPLIT --implicit-check-not=f32
// RUN: not swage-opt --swage-to-plan='schedule=persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERSISTENT

module {
  func.func @segmented_sum_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
}

// TASKS: func.func @segmented_sum_f64(%{{.*}}: memref<?xf64> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %{{.*}}: memref<?xi32> {swage.role = #swage.role<offsets>, swage_plan.source_index = 1 : i32}, %{{.*}}: memref<?xf64> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %{{.*}}: memref<?xi32>,
// TASKS: swage_plan.tasks policy<cta>
// TASKS: %[[SUM:.*]] = swage.reduce %{{.*}} kind<sum> : !swage.segment<f64> -> f64 {
// TASKS: swage_plan.yield %[[SUM]] : f64

// SPLIT: func.func @segmented_sum_f64__partial(%[[VALUES:.*]]: memref<?xf64> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %[[RANGES:.*]]: memref<?xi32>, %[[SCRATCH:.*]]: memref<?xf64>,
// SPLIT: swage_plan.partial_tasks values(%[[VALUES]] : memref<?xf64>)
// SPLIT-SAME: into(%[[SCRATCH]] : memref<?xf64>)
// SPLIT: func.func @segmented_sum_f64__merge(%[[PARTIALS:.*]]: memref<?xf64>, %[[OUTPUT:.*]]: memref<?xf64> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32},
// SPLIT: swage_plan.merge_tasks scratch(%[[PARTIALS]] : memref<?xf64>)
// SPLIT: ^bb0(%[[RANGE:.*]]: !swage.segment<f64>):
// SPLIT-NEXT: swage.reduce %[[RANGE]] kind<sum> : !swage.segment<f64> -> f64 {

// PERSISTENT: error: persistent execution requires f32 values, got f64
