// test/Dialect/SwagePlan/roundtrip.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of plan functions: the launch width
// attribute, the task operation with and without a task buffer and an into
// buffer, and both forms of the yield.

module {
  // CHECK-LABEL: func.func @direct(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xf32>, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 128 : i32} {
  // CHECK-NEXT: swage_plan.tasks policy<cta> segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) into(%[[OUTPUT]] : memref<?xf32>) {
  // CHECK-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
  // CHECK: swage_plan.yield %[[SUM]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @direct(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
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
    return
  }

  // CHECK-LABEL: func.func @task_ids(
  // CHECK-SAME: %{{.*}}: memref<?xf32>, %{{.*}}: memref<?xi32>, %{{.*}}: memref<?xf32>, %[[IDS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32, %[[TASK_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 32 : i32} {
  // CHECK-NEXT: swage_plan.tasks policy<warp> segments({{.*}}) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) ids(%[[IDS]] : memref<?xi32>) task_count(%[[TASK_COUNT]] : i32) into(%{{.*}} : memref<?xf32>) {
  func.func @task_ids(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 32 : i32} {
    swage_plan.tasks policy<warp>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  // The oracle: no launch width, and no task buffer.
  // CHECK-LABEL: func.func @sequential(
  // CHECK-SAME: %{{.*}}: i32) {
  // CHECK-NEXT: swage_plan.tasks policy<sequential> segments({{.*}}) value_count({{.*}}) segment_count({{.*}}) into(%{{.*}} : memref<?xf32>) {
  func.func @sequential(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<sequential>
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
    return
  }

  // CHECK-LABEL: func.func @map_store(
  // CHECK: swage_plan.tasks policy<cta> segments({{.*}}) value_count({{.*}}) segment_count(%{{.*}} : i32) {
  // CHECK-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
  // CHECK: swage.map_store %[[SEGMENT]], %{{.*}} captures(
  // CHECK: swage_plan.yield{{$}}
  func.func @map_store(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<f32>):
      %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %total = swage.reduce %segment captures(%max : f32) kind<sum>
          : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32, %m: f32):
        %centered = arith.subf %value, %m : f32
        swage.yield %centered : f32
      }
      swage.map_store %segment, %output captures(%max, %total : f32, f32)
          : !swage.segment<f32>, memref<?xf32> {
      ^bb0(%value: f32, %m: f32, %t: f32):
        %centered = arith.subf %value, %m : f32
        %normalized = arith.divf %centered, %t : f32
        swage.yield %normalized : f32
      }
      swage_plan.yield
    }
    return
  }
}
