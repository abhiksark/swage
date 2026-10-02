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
