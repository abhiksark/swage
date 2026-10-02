// test/Conversion/SwageToPlan/ragged-softmax.mlir
// A program with several reductions and a map store plans under the direct
// schedule. The reductions come first, in program order, each capturing the
// results before it, and the store follows them: the order the kernel runs
// them in. The store writes the output itself, so the task operation has no
// into buffer and the region yields nothing.
//
// RUN: swage-opt --swage-to-plan %s \
// RUN:   | FileCheck %s --implicit-check-not='swage.map '
// RUN: not swage-opt --swage-to-plan='schedule=task-ids' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TASKS

// CHECK: func.func @ragged_softmax(%{{.*}}: memref<?xf32> {{[{].*[}]}}, %{{.*}}: memref<?xi32> {{[{].*[}]}}, %[[OUTPUT:.*]]: memref<?xf32> {{[{].*[}]}}, %{{.*}}: i32 {{[{].*[}]}}, %{{.*}}: i32 {{[{].*[}]}}) attributes {swage_plan.block_threads = 128 : i32} {
// CHECK-NEXT: swage_plan.tasks policy<cta> segments({{.*}}) value_count({{.*}}) segment_count(%{{.*}} : i32) {
// CHECK-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[MAX:.*]] = swage.reduce %[[SEGMENT]] kind<max> : !swage.segment<f32> -> f32 {
// CHECK: %[[TOTAL:.*]] = swage.reduce %[[SEGMENT]] captures(%[[MAX]] : f32) kind<sum> : !swage.segment<f32> -> f32 {
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: f32, %[[M:.*]]: f32):
// CHECK-NEXT: %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[M]] : f32
// CHECK-NEXT: %[[EXPONENTIAL:.*]] = math.exp2 %[[CENTERED]] : f32
// CHECK-NEXT: swage.yield %[[EXPONENTIAL]] : f32
// CHECK: swage.map_store %[[SEGMENT]], %[[OUTPUT]] captures(%[[MAX]], %[[TOTAL]] : f32, f32) : !swage.segment<f32>, memref<?xf32> {
// CHECK-NEXT: ^bb0(%[[VALUE:.*]]: f32, %[[M:.*]]: f32, %[[T:.*]]: f32):
// CHECK-NEXT: %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[M]] : f32
// CHECK-NEXT: %[[EXPONENTIAL:.*]] = math.exp2 %[[CENTERED]] : f32
// CHECK-NEXT: %[[NORMALIZED:.*]] = arith.divf %[[EXPONENTIAL]], %[[T]] : f32
// CHECK-NEXT: swage.yield %[[NORMALIZED]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: swage_plan.yield{{$}}
// CHECK-NEXT: }
// CHECK-NEXT: return

// A task buffer comes from host classification, which describes one
// capture-free reduction.
// TASKS: error: planning requires capture-free maps

module {
  func.func @ragged_softmax(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %centered = arith.subf %value, %m : f32
      %exponential = math.exp2 %centered : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %centered = arith.subf %value, %m : f32
      %exponential = math.exp2 %centered : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }
}
