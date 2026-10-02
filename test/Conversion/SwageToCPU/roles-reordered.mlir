// test/Conversion/SwageToCPU/roles-reordered.mlir
// The sequential lowering keeps the function and its argument order, reads
// each argument by its role, and removes the roles, which mean nothing once
// the Swage operations are gone.
//
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s

// CHECK-LABEL: func.func @segmented_sum(
// CHECK-SAME: %[[SEGMENTS:.*]]: i32, %[[OUTPUT:.*]]: memref<?xf32>, %{{.*}}: i32, %[[OFFSETS:.*]]: memref<?xi32>, %[[VALUES:.*]]: memref<?xf32>)
// CHECK-NOT: swage.role
// CHECK: %[[COUNT:.*]] = arith.index_cast %[[SEGMENTS]] : i32 to index
// CHECK: scf.for %[[SID:.*]] = %{{.*}} to %[[COUNT]]
// CHECK: memref.load %[[OFFSETS]][%[[SID]]] : memref<?xi32>
// CHECK: memref.load %[[VALUES]][%{{.*}}] : memref<?xf32>
// CHECK: memref.store %{{.*}}, %[[OUTPUT]][%[[SID]]] : memref<?xf32>

module {
  func.func @segmented_sum(
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %values: memref<?xf32> {swage.role = #swage.role<values>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
