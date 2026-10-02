// test/Conversion/SwageToCPU/segmented-columns.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// The oracle of rank-two values: a loop over the segments, a loop over the
// columns, and a loop over the rows of the segment in that column. A column
// is a strided run of the row-order view of the values, so its rows are
// added in order, and each result is stored at its segment and column.

module {
  func.func @segmented_sum_r2(
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
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid, %col] : memref<?x?xf32>
    return
  }
  func.func @segmented_max_r2(
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
    %result = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid, %col] : memref<?x?xf32>
    return
  }
  func.func @segmented_mean_f64_r2(
      %values: memref<?x?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf64>, memref<?xi32>, index, index
          -> !swage.segment<f64>
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    %extent = swage.extent %segment : !swage.segment<f64>
    %count = arith.index_cast %extent : index to i32
    %divisor = arith.sitofp %count : i32 to f64
    %mean = arith.divf %result, %divisor : f64
    memref.store %mean, %output[%sid, %col] : memref<?x?xf64>
    return
  }
}

// CHECK: func.func @segmented_sum_r2(%[[VALUES:.*]]: memref<?x?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?x?xf32>, %{{.*}}: i32, %[[SEGMENT_COUNT:.*]]: i32, %[[FEATURE_COUNT:.*]]: i32)
// CHECK: %[[ROW_COUNT:.*]] = memref.dim %[[VALUES]], %{{.*}} : memref<?x?xf32>
// CHECK: %[[COLUMN_COUNT:.*]] = memref.dim %[[VALUES]], %{{.*}} : memref<?x?xf32>
// CHECK: %[[ELEMENTS:.*]] = arith.muli %[[ROW_COUNT]], %[[COLUMN_COUNT]] : index
// CHECK: %[[FLAT:.*]] = memref.reinterpret_cast %[[VALUES]] to offset: [0], sizes: [%[[ELEMENTS]]], strides: [1] : memref<?x?xf32> to memref<?xf32>
// CHECK: %[[COLUMNS:.*]] = arith.index_cast %[[FEATURE_COUNT]] : i32 to index
// CHECK: scf.for %[[SID:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[START:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:   %[[FIRST_ROW:.*]] = arith.muli %[[START]], %[[COLUMNS]] : index
// CHECK:   %[[LAST:.*]] = arith.muli %[[END]], %[[COLUMNS]] : index
// CHECK:   scf.for %[[COLUMN:.*]] = %{{.*}} to %[[COLUMNS]] step %{{.*}} {
// CHECK:     %[[FIRST:.*]] = arith.addi %[[FIRST_ROW]], %[[COLUMN]] : index
// CHECK:     %[[SUM:.*]] = scf.for %[[INDEX:.*]] = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK:       %[[VALUE:.*]] = memref.load %[[FLAT]][%[[INDEX]]] : memref<?xf32>
// CHECK:       arith.addf %[[ACC]], %[[VALUE]] : f32
// CHECK:     }
// CHECK:     memref.store %[[SUM]], %[[OUTPUT]][%[[SID]], %[[COLUMN]]] : memref<?x?xf32>
// CHECK:   }
// CHECK: }

// CHECK: func.func @segmented_max_r2(
// CHECK: arith.maximumf

// The extent of a mean is the number of rows of its segment.
// CHECK: func.func @segmented_mean_f64_r2(
// CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<?xi32>
// CHECK:   memref.load %{{.*}}[%{{.*}}] : memref<?xi32>
// CHECK-NEXT: %[[MEAN_START:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT: %[[MEAN_END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK-NEXT: %[[ROWS:.*]] = arith.subi %[[MEAN_END]], %[[MEAN_START]] : index
// CHECK:     %[[COUNT:.*]] = arith.index_cast %[[ROWS]] : index to i32
// CHECK:     %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f64
// CHECK:     %[[MEAN:.*]] = arith.divf %{{.*}}, %[[DIVISOR]] : f64
// CHECK:     memref.store %[[MEAN]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x?xf64>
