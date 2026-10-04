// test/Dialect/SwagePlan/extent.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of task regions that take the extent
// of their segment as a second argument and hold a scalar epilogue: the
// plan functions of a program that divides its sum by the extent.

module {
  // CHECK-LABEL: func.func @tasks(
  // CHECK: swage_plan.tasks policy<cta>
  // CHECK: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>, %[[EXTENT:.*]]: index):
  // CHECK-NEXT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
  // CHECK: %[[COUNT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
  // CHECK-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f32
  // CHECK-NEXT: %[[MEAN:.*]] = arith.divf %[[SUM]], %[[DIVISOR]] : f32
  // CHECK-NEXT: swage_plan.yield %[[MEAN]] : f32
  func.func @tasks(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>, %extent: index):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %count = arith.index_cast %extent : index to i32
      %divisor = arith.sitofp %count : i32 to f32
      %mean = arith.divf %sum, %divisor : f32
      swage_plan.yield %mean : f32
    }
    return
  }

  // A region may take the extent and leave it unread.
  // CHECK-LABEL: func.func @unread_extent(
  // CHECK: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>, %{{.*}}: index):
  // CHECK: swage_plan.yield %{{.*}} : f32
  func.func @unread_extent(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>, %extent: index):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  // CHECK-LABEL: func.func @fused(
  // CHECK: swage_plan.fused_tasks
  // CHECK: warp {
  // CHECK-NEXT: ^bb0(%{{.*}}: !swage.segment<f32>, %[[WARP_EXTENT:.*]]: index):
  // CHECK: arith.index_cast %[[WARP_EXTENT]] : index to i32
  // CHECK: } cta {
  // CHECK-NEXT: ^bb0(%{{.*}}: !swage.segment<f32>, %[[CTA_EXTENT:.*]]: index):
  // CHECK: arith.index_cast %[[CTA_EXTENT]] : index to i32
  func.func @fused(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %ids: memref<?xi32>, %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) warp_task_count(%warp_task_count : i32)
        cta_task_count(%cta_task_count : i32)
        into(%output : memref<?xf32>) warp {
    ^bb0(%segment: !swage.segment<f32>, %extent: index):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %count = arith.index_cast %extent : index to i32
      %divisor = arith.sitofp %count : i32 to f32
      %mean = arith.divf %sum, %divisor : f32
      swage_plan.yield %mean : f32
    } cta {
    ^bb0(%segment: !swage.segment<f32>, %extent: index):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %count = arith.index_cast %extent : index to i32
      %divisor = arith.sitofp %count : i32 to f32
      %mean = arith.divf %sum, %divisor : f32
      swage_plan.yield %mean : f32
    }
    return
  }

  // The merge of a split segment reads its extent from the range records.
  // CHECK-LABEL: func.func @merge(
  // CHECK-SAME: %[[SCRATCH:.*]]: memref<?xf64>, %[[OUTPUT:.*]]: memref<?xf64>, %[[MERGES:.*]]: memref<?xi32>, %[[RANGES:.*]]: memref<?xi32>, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32)
  // CHECK-NEXT: swage_plan.merge_tasks scratch(%[[SCRATCH]] : memref<?xf64>) partial_count(%[[PARTIAL_COUNT]] : i32) merges(%[[MERGES]] : memref<?xi32>) merge_count(%[[MERGE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) ranges(%[[RANGES]] : memref<?xi32>) into(%[[OUTPUT]] : memref<?xf64>) {
  // CHECK-NEXT: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f64>, %[[EXTENT:.*]]: index):
  // CHECK-NEXT: %[[TOTAL:.*]] = swage.reduce %[[PARTIALS]] kind<sum> : !swage.segment<f64> -> f64 {
  // CHECK: %[[COUNT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
  // CHECK-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f64
  // CHECK-NEXT: %[[MEAN:.*]] = arith.divf %[[TOTAL]], %[[DIVISOR]] : f64
  // CHECK-NEXT: swage_plan.yield %[[MEAN]] : f64
  func.func @merge(
      %scratch: memref<?xf64>, %output: memref<?xf64>,
      %merges: memref<?xi32>, %ranges: memref<?xi32>, %partial_count: i32,
      %merge_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    swage_plan.merge_tasks scratch(%scratch : memref<?xf64>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        ranges(%ranges : memref<?xi32>) into(%output : memref<?xf64>) {
    ^bb0(%partials: !swage.segment<f64>, %extent: index):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f64> -> f64 {
      ^bb0(%partial: f64):
        swage.yield %partial : f64
      }
      %count = arith.index_cast %extent : index to i32
      %divisor = arith.sitofp %count : i32 to f64
      %mean = arith.divf %total, %divisor : f64
      swage_plan.yield %mean : f64
    }
    return
  }
}
