// test/Conversion/SwagePlanToSCF/tasks.mlir
// Sequential plans written by hand. The conversion turns the task operation
// into one loop over the segments, lowers the consumers of the bound
// segment on the memrefs, and removes the roles of the function, which
// keeps its signature and its callers. A plan function of a kernel in the
// same module is left as it is.
//
// RUN: swage-opt --swage-plan-to-scf %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @scalar(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>, %[[OFFSETS:.*]]: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xf32>, %{{.*}}: i32, %[[SEGMENT_COUNT:.*]]: i32) {
  // CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
  // CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-NEXT: %[[SEGMENTS:.*]] = arith.index_cast %[[SEGMENT_COUNT]] : i32 to index
  // CHECK-NEXT: scf.for %[[SID:.*]] = %[[C0]] to %[[SEGMENTS]] step %[[C1]] {
  // CHECK-NEXT: %[[START_WORD:.*]] = memref.load %[[OFFSETS]][%[[SID]]] : memref<?xi32>
  // CHECK-NEXT: %[[NEXT:.*]] = arith.addi %[[SID]], %[[C1]] : index
  // CHECK-NEXT: %[[END_WORD:.*]] = memref.load %[[OFFSETS]][%[[NEXT]]] : memref<?xi32>
  // CHECK-NEXT: %[[START:.*]] = arith.index_cast %[[START_WORD]] : i32 to index
  // CHECK-NEXT: %[[END:.*]] = arith.index_cast %[[END_WORD]] : i32 to index
  // CHECK-NEXT: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: %[[SUM:.*]] = scf.for %[[INDEX:.*]] = %[[START]] to %[[END]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f32) {
  // CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[VALUES]][%[[INDEX]]] : memref<?xf32>
  // CHECK-NEXT: %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[VALUE]] : f32
  // CHECK-NEXT: scf.yield %[[NEXT_ACC]] : f32
  // CHECK-NEXT: }
  // CHECK-NEXT: memref.store %[[SUM]], %[[OUTPUT]][%[[SID]]] : memref<?xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @scalar(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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

  // The function keeps its callers.
  // CHECK-LABEL: func.func @caller(
  // CHECK: call @scalar(
  func.func @caller(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    call @scalar(%values, %offsets, %output, %value_count, %segment_count)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    return
  }

  // A reduction, then a store that captures its result and writes one value
  // per element.
  // CHECK-LABEL: func.func @map_store(
  // CHECK-SAME: %[[VALUES:.*]]: memref<?xf32>, %{{.*}}: memref<?xi32>, %[[OUTPUT:.*]]: memref<?xf32>, %{{.*}}: i32, %{{.*}}: i32) {
  // CHECK: %[[MAX:.*]] = scf.for %{{.*}} = %[[START:.*]] to %[[END:.*]] step %[[C1:.*]] iter_args(
  // CHECK: arith.maximumf
  // CHECK: scf.for %[[INDEX:.*]] = %[[START]] to %[[END]] step %[[C1]] {
  // CHECK-NEXT: %[[VALUE:.*]] = memref.load %[[VALUES]][%[[INDEX]]] : memref<?xf32>
  // CHECK-NEXT: %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[MAX]] : f32
  // CHECK-NEXT: memref.store %[[CENTERED]], %[[OUTPUT]][%[[INDEX]]] : memref<?xf32>
  // CHECK-NEXT: }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  func.func @map_store(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<f32>):
      %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage.map_store %segment, %output captures(%max : f32)
          : !swage.segment<f32>, memref<?xf32> {
      ^bb0(%value: f32, %m: f32):
        %centered = arith.subf %value, %m : f32
        swage.yield %centered : f32
      }
      swage_plan.yield
    }
    return
  }

  // A kernel plan is not the oracle's to lower.
  // CHECK-LABEL: func.func @kernel(
  // CHECK-SAME: attributes {swage_plan.block_threads = 128 : i32} {
  // CHECK-NEXT: swage_plan.tasks policy<cta>
  // CHECK: swage.reduce
  func.func @kernel(
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
}
