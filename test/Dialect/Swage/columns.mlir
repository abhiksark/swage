// test/Dialect/Swage/columns.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s
// RUN: swage-opt --mlir-print-op-generic %s | swage-opt | FileCheck %s

// Parse -> print -> parse round trip of a segment function over rank-two
// values. A program instance is one segment and one column: the column is
// swage.segment_id 1, make_segment binds it, and the segment is a run of
// scalars, so its type carries neither the column nor the column count.

// CHECK-LABEL: func.func @column_sum(
// CHECK-SAME: %[[VALUES:[^:]+]]: memref<?x?xf32> {swage.role = #swage.role<values>}
// CHECK-SAME: %[[OFFSETS:[^:]+]]: memref<?xi32> {swage.role = #swage.role<offsets>}
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?x?xf32> {swage.role = #swage.role<output>}
// CHECK-SAME: %{{[^:]+}}: i32 {swage.role = #swage.role<feature_count>}
func.func @column_sum(
    %values: memref<?x?xf32> {swage.role = #swage.role<values>},
    %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
    %output: memref<?x?xf32> {swage.role = #swage.role<output>},
    %value_count: i32 {swage.role = #swage.role<value_count>},
    %segment_count: i32 {swage.role = #swage.role<segment_count>},
    %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
  // CHECK: %[[SID:.*]] = swage.segment_id 0
  // CHECK: %[[COLUMN:.*]] = swage.segment_id 1
  %sid = swage.segment_id 0
  %col = swage.segment_id 1
  // CHECK: %[[SEGMENT:.*]] = swage.make_segment %[[VALUES]], %[[OFFSETS]], %[[SID]] column(%[[COLUMN]]) : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
  %segment = swage.make_segment %values, %offsets, %sid column(%col)
      : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
  // CHECK: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
  %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
  ^bb0(%value: f32):
    swage.yield %value : f32
  }
  // CHECK: memref.store %[[SUM]], %[[OUTPUT]][%[[SID]], %[[COLUMN]]] : memref<?x?xf32>
  memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
  return
}

// A map store of a column segment writes the rows of the segment in its
// column of a rank-two output.
// CHECK-LABEL: func.func @column_store(
func.func @column_store(
    %values: memref<?x?xf32>, %offsets: memref<?xi32>,
    %output: memref<?x?xf32>, %sid: index, %col: index) {
  %segment = swage.make_segment %values, %offsets, %sid column(%col)
      : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
  // CHECK: swage.map_store %{{.*}}, %{{.*}} : !swage.segment<f32>, memref<?x?xf32> {
  swage.map_store %segment, %output : !swage.segment<f32>, memref<?x?xf32> {
  ^bb0(%value: f32):
    swage.yield %value : f32
  }
  return
}
