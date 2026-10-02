// test/Dialect/SwagePlan/effects.mlir
// RUN: swage-opt --cse %s | FileCheck %s

// A task operation reads its values, offsets, and task buffer and writes
// its into buffer, and its effects include those of the consumers in its
// region. Common subexpression elimination therefore keeps a load on each
// side of one, and merges two loads when nothing writes between them.

// CHECK-LABEL: func.func @load_across_the_scalar_store(
// CHECK: %[[BEFORE:.*]] = memref.load
// CHECK: swage_plan.tasks
// CHECK: %[[AFTER:.*]] = memref.load
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @load_across_the_scalar_store(
    %values: memref<?xf32>, %offsets: memref<?xi32>,
    %output: memref<?xf32>, %value_count: i32, %segment_count: i32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = memref.load %output[%c0] : memref<?xf32>
  swage_plan.tasks policy<cta>
      segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
      value_count(%value_count : i32) segment_count(%segment_count : i32)
      into(%output : memref<?xf32>) {
  ^bb0(%segment: !swage.segment<f32>):
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    swage_plan.yield %sum : f32
  }
  %after = memref.load %output[%c0] : memref<?xf32>
  return %before, %after : f32, f32
}

// The store happens inside the region here, through swage.map_store.
// CHECK-LABEL: func.func @load_across_a_map_store(
// CHECK: %[[BEFORE:.*]] = memref.load
// CHECK: swage_plan.tasks
// CHECK: %[[AFTER:.*]] = memref.load
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @load_across_a_map_store(
    %values: memref<?xf32>, %offsets: memref<?xi32>,
    %output: memref<?xf32>, %value_count: i32, %segment_count: i32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = memref.load %output[%c0] : memref<?xf32>
  swage_plan.tasks policy<cta>
      segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
      value_count(%value_count : i32) segment_count(%segment_count : i32) {
  ^bb0(%segment: !swage.segment<f32>):
    swage.map_store %segment, %output : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    swage_plan.yield
  }
  %after = memref.load %output[%c0] : memref<?xf32>
  return %before, %after : f32, f32
}

// A persistent task operation writes scratch and its counters as well as
// its into buffer, so a load of scratch stays on each side of it.
// CHECK-LABEL: func.func @load_of_scratch_across_the_queues(
// CHECK: %[[BEFORE:.*]] = memref.load
// CHECK: swage_plan.persistent_tasks
// CHECK: %[[AFTER:.*]] = memref.load
// CHECK: return %[[BEFORE]], %[[AFTER]]
func.func @load_of_scratch_across_the_queues(
    %values: memref<?xf32>, %offsets: memref<?xi32>,
    %output: memref<?xf32>, %warp_ids: memref<?xi32>,
    %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
    %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
    %scratch: memref<?xf32>, %counters: memref<?xi32>,
    %value_count: i32, %warp_task_count: i32,
    %cta_task_count: i32, %partial_count: i32,
    %merge_count: i32, %segment_count: i32) -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = memref.load %scratch[%c0] : memref<?xf32>
  swage_plan.persistent_tasks
      segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
      value_count(%value_count : i32) segment_count(%segment_count : i32)
      warp_ids(%warp_ids : memref<?xi32>)
      warp_task_count(%warp_task_count : i32)
      cta_ids(%cta_ids : memref<?xi32>)
      cta_task_count(%cta_task_count : i32)
      ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
      partial_count(%partial_count : i32)
      merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
      scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
      into(%output : memref<?xf32>) cta {
  ^bb0(%segment: !swage.segment<f32>):
    %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    swage_plan.yield %total : f32
  } partial {
  ^bb0(%chunk: !swage.segment<f32>):
    %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    swage_plan.yield %total : f32
  } merge {
  ^bb0(%partials: !swage.segment<f32>):
    %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%partial: f32):
      swage.yield %partial : f32
    }
    swage_plan.yield %total : f32
  } warp {
  ^bb0(%segment: !swage.segment<f32>):
    %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    swage_plan.yield %total : f32
  }
  %after = memref.load %scratch[%c0] : memref<?xf32>
  return %before, %after : f32, f32
}

// CHECK-LABEL: func.func @loads_without_a_task_operation(
// CHECK: %[[ONLY:.*]] = memref.load
// CHECK-NOT: memref.load
// CHECK: return %[[ONLY]], %[[ONLY]]
func.func @loads_without_a_task_operation(%output: memref<?xf32>)
    -> (f32, f32) {
  %c0 = arith.constant 0 : index
  %before = memref.load %output[%c0] : memref<?xf32>
  %after = memref.load %output[%c0] : memref<?xf32>
  return %before, %after : f32, f32
}
