// test/Conversion/SwageToPlan/sequential.mlir
// The sequential schedule plans the CPU oracle. The function is planned in
// place: it keeps its signature, the order of its arguments, its roles, and
// its callers, and has no launch width. The task operation takes each
// operand from the argument of that role, whatever the order.
//
// RUN: swage-opt --swage-to-plan='schedule=sequential' %s | FileCheck %s
// RUN: swage-opt --swage-to-plan='schedule=sequential block-threads=96' %s \
// RUN:   | FileCheck %s

// CHECK: func.func @reversed(%[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>}, %[[OUTPUT:.*]]: memref<?xf32> {swage.role = #swage.role<output>}, %[[VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>}, %[[OFFSETS:.*]]: memref<?xi32> {swage.role = #swage.role<offsets>}, %[[VALUES:.*]]: memref<?xf32> {swage.role = #swage.role<values>}) {
// CHECK-NEXT: swage_plan.tasks policy<sequential> segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) into(%[[OUTPUT]] : memref<?xf32>) {
// CHECK-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
// CHECK-NEXT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
// CHECK: swage_plan.yield %[[SUM]] : f32
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NEXT: }
// CHECK-NEXT: func.func @caller(
// CHECK: call @reversed(

module {
  func.func @reversed(
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
  func.func @caller(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    call @reversed(%segment_count, %output, %value_count, %offsets, %values)
        : (i32, memref<?xf32>, i32, memref<?xi32>, memref<?xf32>) -> ()
    return
  }
}
