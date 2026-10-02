// test/Conversion/SwageToCPU/segmented-sum.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s

module {
  func.func @segmented_sum(
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
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-LABEL: func.func @segmented_sum(
// CHECK-SAME: %[[VALUES:[^:]+]]: memref<?xf32>
// CHECK-SAME: %[[OFFSETS:[^:]+]]: memref<?xi32>
// CHECK-SAME: %[[OUTPUT:[^:]+]]: memref<?xf32>
// CHECK-SAME: %{{[^:]+}}: i32
// CHECK-SAME: %[[SEGMENT_COUNT:[^)]+]]: i32
// CHECK: %[[ZERO:.*]] = arith.constant 0 : index
// CHECK: %[[ONE:.*]] = arith.constant 1 : index
// CHECK: %[[SEGMENTS:.*]] = arith.index_cast %[[SEGMENT_COUNT]] : i32 to index
// CHECK: scf.for %[[SID:.*]] = %[[ZERO]] to %[[SEGMENTS]] step %[[ONE]] {
// CHECK:   %[[START_I32:.*]] = memref.load %[[OFFSETS]][%[[SID]]]
// CHECK:   %[[NEXT:.*]] = arith.addi %[[SID]], %[[ONE]] : index
// CHECK:   %[[END_I32:.*]] = memref.load %[[OFFSETS]][%[[NEXT]]]
// CHECK:   %[[START:.*]] = arith.index_cast %[[START_I32]] : i32 to index
// CHECK:   %[[END:.*]] = arith.index_cast %[[END_I32]] : i32 to index
// CHECK:   %[[IDENTITY:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:   %[[SUM:.*]] = scf.for %[[I:.*]] = %[[START]] to %[[END]] step %[[ONE]] iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:     %[[VALUE:.*]] = memref.load %[[VALUES]][%[[I]]]
// CHECK:     %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[VALUE]] : f32
// CHECK:     scf.yield %[[NEXT_ACC]] : f32
// CHECK:   }
// CHECK:   memref.store %[[SUM]], %[[OUTPUT]][%[[SID]]]
// CHECK: }
// CHECK-NOT: swage.
