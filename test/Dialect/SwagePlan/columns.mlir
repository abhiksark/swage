// test/Dialect/SwagePlan/columns.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of the plan functions of rank-two
// values: the column kernel, whose task operation takes the number of
// columns and has policy<column>, the row-stripe kernel, whose task
// operation has policy<cta> and may take a task buffer, and the oracle.

module {
  // CHECK-LABEL: func.func @row_stripes(
  // CHECK-SAME: %[[ROW_VALUES:.*]]: memref<?x?xf32>, %[[ROW_OFFSETS:.*]]: memref<?xi32>, %[[ROW_OUTPUT:.*]]: memref<?x?xf32>, %[[IDS:.*]]: memref<?xi32>, %[[ROW_VALUE_COUNT:.*]]: i32, %[[TASK_COUNT:.*]]: i32, %[[ROW_SEGMENT_COUNT:.*]]: i32, %[[ROW_FEATURE_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 128 : i32} {
  // CHECK-NEXT: swage_plan.tasks policy<cta> segments(%[[ROW_VALUES]], %[[ROW_OFFSETS]] : memref<?x?xf32>, memref<?xi32>) value_count(%[[ROW_VALUE_COUNT]] : i32) segment_count(%[[ROW_SEGMENT_COUNT]] : i32) feature_count(%[[ROW_FEATURE_COUNT]] : i32) ids(%[[IDS]] : memref<?xi32>) task_count(%[[TASK_COUNT]] : i32) into(%[[ROW_OUTPUT]] : memref<?x?xf32>) {
  func.func @row_stripes(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  // CHECK-LABEL: func.func @columns(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?x?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?x?xf32>, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32, %[[FEATURE_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 128 : i32} {
  // CHECK-NEXT: swage_plan.tasks policy<column> segments(%[[VALUES]], %[[OFFSETS]] : memref<?x?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) feature_count(%[[FEATURE_COUNT]] : i32) into(%[[OUTPUT]] : memref<?x?xf32>) {
  // CHECK-NEXT: ^bb0(%[[COLUMN:.*]]: !swage.segment<f32>):
  // CHECK-NEXT: %[[SUM:.*]] = swage.reduce %[[COLUMN]] kind<sum> : !swage.segment<f32> -> f32 {
  // CHECK: swage_plan.yield %[[SUM]] : f32
  func.func @columns(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }

  // The oracle visits the columns of each segment in order. A mean takes
  // the extent of its segment, which is a number of rows.
  // CHECK-LABEL: func.func @sequential_columns(
  // CHECK: swage_plan.tasks policy<sequential>
  // CHECK-SAME: feature_count(%{{.*}} : i32) into(%{{.*}} : memref<?x?xf64>) {
  // CHECK-NEXT: ^bb0(%{{.*}}: !swage.segment<f64>, %{{.*}}: index):
  func.func @sequential_columns(
      %values: memref<?x?xf64>, %offsets: memref<?xi32>,
      %output: memref<?x?xf64>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?x?xf64>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        into(%output : memref<?x?xf64>) {
    ^bb0(%column: !swage.segment<f64>, %rows: index):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f64> -> f64 {
      ^bb0(%value: f64):
        swage.yield %value : f64
      }
      %count = arith.index_cast %rows : index to i32
      %divisor = arith.sitofp %count : i32 to f64
      %mean = arith.divf %sum, %divisor : f64
      swage_plan.yield %mean : f64
    }
    return
  }

  // A region that writes its results itself stores into rank-two rows.
  // CHECK-LABEL: func.func @column_stores(
  // CHECK: swage_plan.tasks policy<column>
  // CHECK: swage.map_store %{{.*}}, %{{.*}} : !swage.segment<f32>, memref<?x?xf32> {
  func.func @column_stores(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32) {
    ^bb0(%column: !swage.segment<f32>):
      swage.map_store %column, %output
          : !swage.segment<f32>, memref<?x?xf32> {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield
    }
    return
  }
}
