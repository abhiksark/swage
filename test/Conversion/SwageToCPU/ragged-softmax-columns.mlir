// test/Conversion/SwageToCPU/ragged-softmax-columns.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// The oracle of the softmax over rank-two values: a loop over the segments,
// a loop over the columns, and the three stages of the softmax over the rows
// of the segment in that column. A column is a strided run of the row-order
// view of the values, and the store writes the same index of the row-order
// view of the output.

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

// CHECK: func.func @ragged_softmax_r2(%[[VALUES:.*]]: memref<?x?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?x?xf32>, %{{.*}}: i32, %[[SEGMENT_COUNT:.*]]: i32, %[[FEATURE_COUNT:.*]]: i32)
// CHECK: %[[FLAT:.*]] = memref.reinterpret_cast %[[VALUES]] to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<?x?xf32> to memref<?xf32>
// CHECK: %[[COLUMNS:.*]] = arith.index_cast %[[FEATURE_COUNT]] : i32 to index
// CHECK: scf.for %[[SID:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[START:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK:   %[[FIRST_ROW:.*]] = arith.muli %[[START]], %[[COLUMNS]] : index
// CHECK:   %[[LAST:.*]] = arith.muli %[[END]], %[[COLUMNS]] : index
// CHECK:   scf.for %[[COLUMN:.*]] = %{{.*}} to %[[COLUMNS]] step %{{.*}} {
// CHECK:     %[[FIRST:.*]] = arith.addi %[[FIRST_ROW]], %[[COLUMN]] : index
// CHECK:     %[[MAX:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] iter_args(%{{.*}} = %{{.*}}) -> (f32) {
// CHECK:       arith.maximumf
// CHECK:     }
// CHECK:     %[[TOTAL:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK:       arith.subf %{{.*}}, %[[MAX]] : f32
// CHECK:       %[[SUM_EXPONENTIAL:.*]] = math.exp2 %{{.*}} : f32
// CHECK:       arith.addf %[[ACC]], %[[SUM_EXPONENTIAL]] : f32
// CHECK:     }
// CHECK:     %[[FLAT_OUTPUT:.*]] = memref.reinterpret_cast %[[OUTPUT]] to offset: [0], sizes: [%{{.*}}], strides: [1] : memref<?x?xf32> to memref<?xf32>
// CHECK:     scf.for %[[INDEX:.*]] = %[[FIRST]] to %[[LAST]] step %[[COLUMNS]] {
// CHECK:       %[[VALUE:.*]] = memref.load %[[FLAT]][%[[INDEX]]] : memref<?xf32>
// CHECK:       arith.subf %[[VALUE]], %[[MAX]] : f32
// CHECK:       %[[EXPONENTIAL:.*]] = math.exp2 %{{.*}} : f32
// CHECK:       %[[NORMALIZED:.*]] = arith.divf %[[EXPONENTIAL]], %[[TOTAL]] : f32
// CHECK:       memref.store %[[NORMALIZED]], %[[FLAT_OUTPUT]][%[[INDEX]]] : memref<?xf32>
// CHECK:     }
// CHECK:   }
// CHECK: }
