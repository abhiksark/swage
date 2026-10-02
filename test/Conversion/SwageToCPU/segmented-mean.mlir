// test/Conversion/SwageToCPU/segmented-mean.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// The oracle of a mean: the sequential sum of the segment, divided once by
// its extent, the end of the range minus its start.

module {
  func.func @segmented_mean(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %extent = swage.extent %segment : !swage.segment<f32>
    %count = arith.index_cast %extent : index to i32
    %divisor = arith.sitofp %count : i32 to f32
    %mean = arith.divf %sum, %divisor : f32
    memref.store %mean, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK: func.func @segmented_mean(%[[VALUES:.*]]: memref<?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xf32>, %{{.*}}: i32, %[[SEGMENT_COUNT:.*]]: i32)
// CHECK: scf.for %[[SID:.*]] = %{{.*}} to %{{.*}} step %{{.*}} {
// CHECK:   %[[START_WORD:.*]] = memref.load %[[OFFSETS]][%[[SID]]] : memref<?xi32>
// CHECK:   %[[END_WORD:.*]] = memref.load %[[OFFSETS]][%{{.*}}] : memref<?xi32>
// CHECK:   %[[START:.*]] = arith.index_cast %[[START_WORD]] : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %[[END_WORD]] : i32 to index
// CHECK:   %[[EXTENT:.*]] = arith.subi %[[END]], %[[START]] : index
// CHECK:   %[[SUM:.*]] = scf.for %{{.*}} = %[[START]] to %[[END]] step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (f32) {
// CHECK:     arith.addf
// CHECK:   }
// CHECK:   %[[COUNT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
// CHECK:   %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f32
// CHECK:   %[[MEAN:.*]] = arith.divf %[[SUM]], %[[DIVISOR]] : f32
// CHECK:   memref.store %[[MEAN]], %[[OUTPUT]][%[[SID]]] : memref<?xf32>
// CHECK: }
