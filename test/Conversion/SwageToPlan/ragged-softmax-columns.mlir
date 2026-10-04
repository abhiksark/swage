// test/Conversion/SwageToPlan/ragged-softmax-columns.mlir
// The softmax over rank-two values normalizes one column of one segment per
// program instance. It plans like the reductions over rank-two values: the
// direct schedule gives the column kernel, whose task operation takes the
// number of columns and whose region binds one column of the rows of a
// segment. The region is the one of the rank-one softmax, two reductions and
// a map store, and the store writes the rank-two output itself.
//
// The task-ids schedule plans the same region as the row-stripe kernel,
// policy<cta> with a task buffer. Its ids only name segments, so the
// schedule admits the captures and the map store of the softmax over
// rank-two values, which a launch runs with one task per segment. Over
// rank-one values the same schedule refuses them, because host
// classification describes one capture-free reduction. The split schedules
// plan rank-two values but refuse the softmax as they do over rank-one
// values, and the other kernel schedules refuse rank-two values by name
// before they look at the program.
//
// RUN: swage-opt --swage-to-plan='schedule=direct' %s \
// RUN:   | FileCheck %s --implicit-check-not='swage.map ' \
// RUN:       --implicit-check-not=segment_id --implicit-check-not=make_segment
// RUN: swage-opt --swage-to-plan='schedule=task-ids' %s \
// RUN:   | FileCheck %s --check-prefix=STRIPES --implicit-check-not='swage.map '
// RUN: not swage-opt --swage-to-plan='schedule=fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=split-partial' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT
// RUN: not swage-opt --swage-to-plan='schedule=split-merge' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SPLIT
// RUN: not swage-opt --swage-to-plan='schedule=persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED

// CHECK: func.func @ragged_softmax_r2(%[[VALUES:.*]]: memref<?x?xf32> {{[{].*[}]}}, %[[OFFSETS:.*]]: memref<?xi32> {{[{].*[}]}}, %[[OUTPUT:.*]]: memref<?x?xf32> {{[{].*[}]}}, %[[VALUE_COUNT:.*]]: i32 {{[{].*[}]}}, %[[SEGMENT_COUNT:.*]]: i32 {{[{].*[}]}}, %[[FEATURE_COUNT:.*]]: i32 {{[{].*[}]}}) attributes {swage_plan.block_threads = 128 : i32} {
// CHECK-NEXT: swage_plan.tasks policy<column> segments(%[[VALUES]], %[[OFFSETS]] : memref<?x?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) feature_count(%[[FEATURE_COUNT]] : i32) {
// CHECK-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[MAX:.*]] = swage.reduce %[[SEGMENT]] kind<max> : !swage.segment<f32> -> f32 {
// CHECK: %[[TOTAL:.*]] = swage.reduce %[[SEGMENT]] captures(%[[MAX]] : f32) kind<sum> : !swage.segment<f32> -> f32 {
// CHECK: math.exp2
// CHECK: swage.map_store %[[SEGMENT]], %[[OUTPUT]] captures(%[[MAX]], %[[TOTAL]] : f32, f32) : !swage.segment<f32>, memref<?x?xf32> {
// CHECK: %[[NORMALIZED:.*]] = arith.divf %{{.*}}, %{{.*}} : f32
// CHECK-NEXT: swage.yield %[[NORMALIZED]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield{{$}}
// CHECK-NEXT: }
// CHECK-NEXT: return

// STRIPES: func.func @ragged_softmax_r2(%[[R_VALUES:.*]]: memref<?x?xf32> {{[{].*[}]}}, %[[R_OFFSETS:.*]]: memref<?xi32> {{[{].*[}]}}, %[[R_OUTPUT:.*]]: memref<?x?xf32> {{[{].*[}]}}, %[[R_IDS:.*]]: memref<?xi32>, %[[R_VALUE_COUNT:.*]]: i32 {{[{].*[}]}}, %[[R_TASK_COUNT:.*]]: i32, %[[R_SEGMENT_COUNT:.*]]: i32 {{[{].*[}]}}, %[[R_FEATURE_COUNT:.*]]: i32 {{[{].*[}]}}) attributes {swage_plan.block_threads = 128 : i32} {
// STRIPES-NEXT: swage_plan.tasks policy<cta> segments(%[[R_VALUES]], %[[R_OFFSETS]] : memref<?x?xf32>, memref<?xi32>) value_count(%[[R_VALUE_COUNT]] : i32) segment_count(%[[R_SEGMENT_COUNT]] : i32) feature_count(%[[R_FEATURE_COUNT]] : i32) ids(%[[R_IDS]] : memref<?xi32>) task_count(%[[R_TASK_COUNT]] : i32) {
// STRIPES-NEXT: ^bb0(%[[R_SEGMENT:.*]]: !swage.segment<f32>):
// STRIPES-NEXT: %[[R_MAX:.*]] = swage.reduce %[[R_SEGMENT]] kind<max> : !swage.segment<f32> -> f32 {
// STRIPES: %[[R_TOTAL:.*]] = swage.reduce %[[R_SEGMENT]] captures(%[[R_MAX]] : f32) kind<sum> : !swage.segment<f32> -> f32 {
// STRIPES: swage.map_store %[[R_SEGMENT]], %[[R_OUTPUT]] captures(%[[R_MAX]], %[[R_TOTAL]] : f32, f32) : !swage.segment<f32>, memref<?x?xf32> {
// STRIPES: swage_plan.yield{{$}}

// REFUSED: error: {{fused-mixed|persistent}} planning requires rank-one values: a function over rank-two values runs on the direct, task-ids, split-partial, and split-merge schedules
// SPLIT: error: planning requires capture-free maps

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
