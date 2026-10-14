// test/Conversion/SwageToPlan/segmented-min.mlir
// A minimum plans like the other kinds. The task region holds the reduction
// of the program, and the merge region of a split reduces the partial
// results with the same kind. The persistent schedule admits a sum only.
//
// RUN: swage-opt --swage-to-plan='schedule=task-ids' %s \
// RUN:   | FileCheck %s --check-prefix=TASKS
// RUN: swage-opt --swage-to-plan='schedule=split-merge' %s \
// RUN:   | FileCheck %s --check-prefix=MERGE
// RUN: not swage-opt --swage-to-plan='schedule=persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERSISTENT

module {
  func.func @segmented_min(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %minimum = swage.reduce %segment kind<min>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %minimum, %output[%sid] : memref<?xf32>
    return
  }
}

// TASKS: func.func @segmented_min(
// TASKS: swage_plan.tasks policy<cta>
// TASKS: %[[MINIMUM:.*]] = swage.reduce %{{.*}} kind<min> : !swage.segment<f32> -> f32 {
// TASKS: swage_plan.yield %[[MINIMUM]] : f32

// MERGE: func.func @segmented_min__merge(
// MERGE: swage_plan.merge_tasks
// MERGE-NEXT: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f32>):
// MERGE-NEXT: %[[MINIMUM:.*]] = swage.reduce %[[PARTIALS]] kind<min> : !swage.segment<f32> -> f32 {
// MERGE-NEXT: ^bb0(%[[PARTIAL:.*]]: f32):
// MERGE-NEXT: swage.yield %[[PARTIAL]] : f32
// MERGE-NEXT: }
// MERGE-NEXT: swage_plan.yield %[[MINIMUM]] : f32

// PERSISTENT: error: persistent execution requires kind<sum>
