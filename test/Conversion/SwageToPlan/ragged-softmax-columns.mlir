// test/Conversion/SwageToPlan/ragged-softmax-columns.mlir
// The softmax over rank-two values normalizes one column of one segment per
// program instance. It plans like the reductions over rank-two values: the
// direct schedule gives the column kernel, whose task operation takes the
// number of columns and whose region binds one column of the rows of a
// segment. The region is the one of the rank-one softmax, two reductions and
// a map store, and the store writes the rank-two output itself.
//
// Every schedule that needs a task buffer refuses rank-two values before it
// looks at the captures of the program.
//
// RUN: swage-opt --swage-to-plan='schedule=direct' %s \
// RUN:   | FileCheck %s --implicit-check-not='swage.map ' \
// RUN:       --implicit-check-not=segment_id --implicit-check-not=make_segment
// RUN: not swage-opt --swage-to-plan='schedule=task-ids' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=fused-mixed' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
// RUN: not swage-opt --swage-to-plan='schedule=split-partial' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=REFUSED
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

// REFUSED: error: planning requires rank-one values: a function over rank-two values has one kernel, the direct schedule, and no task buffer

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
