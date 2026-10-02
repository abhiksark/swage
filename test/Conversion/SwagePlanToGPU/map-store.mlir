// test/Conversion/SwagePlanToGPU/map-store.mlir
// A task region with two reductions and a map store. Each consumer takes
// the bound segment from the task operation and the results of the
// reductions before it as captures: one stage per reduction, in order, then
// the loop that writes every element.
//
// RUN: swage-opt --swage-plan-to-gpu %s | FileCheck %s \
// RUN:   --implicit-check-not=swage --implicit-check-not=func.func

// CHECK: gpu.func @map_store(%[[VALUES:.*]]: !llvm.ptr, %{{.*}}: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32) kernel
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: %[[FIRST:.*]] = arith.addi %{{.*}}, %[[THREAD]] : index
// CHECK: %[[LOCAL_MAX:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[END:.*]] step %[[STRIDE:.*]] iter_args(
// CHECK: arith.maximumf
// CHECK: %[[MAX:.*]] = gpu.all_reduce maximumf %[[LOCAL_MAX]] uniform {
// CHECK: %[[LOCAL_SUM:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[END]] step %[[STRIDE]] iter_args(
// CHECK: arith.subf %{{.*}}, %[[MAX]] : f32
// CHECK: %[[TOTAL:.*]] = gpu.all_reduce add %[[LOCAL_SUM]] uniform {
// CHECK: scf.for %[[INDEX:.*]] = %[[FIRST]] to %[[END]] step %[[STRIDE]] {
// CHECK-NEXT: %[[INDEX64:.*]] = arith.index_cast %[[INDEX]] : index to i64
// CHECK-NEXT: %[[ADDRESS:.*]] = llvm.getelementptr %[[VALUES]][%[[INDEX64]]]
// CHECK-NEXT: %[[VALUE:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// CHECK-NEXT: %[[CENTERED:.*]] = arith.subf %[[VALUE]], %[[MAX]] : f32
// CHECK-NEXT: %[[NORMALIZED:.*]] = arith.divf %[[CENTERED]], %[[TOTAL]] : f32
// CHECK-NEXT: %[[SLOT:.*]] = llvm.getelementptr %[[OUTPUT]][%[[INDEX64]]]
// CHECK-NEXT: llvm.store %[[NORMALIZED]], %[[SLOT]] : f32, !llvm.ptr
// CHECK-NEXT: }
// CHECK-NEXT: }
// CHECK-NEXT: gpu.return

module {
  func.func @map_store(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32) {
    ^bb0(%segment: !swage.segment<f32>):
      %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %total = swage.reduce %segment captures(%max : f32) kind<sum>
          : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32, %m: f32):
        %centered = arith.subf %value, %m : f32
        swage.yield %centered : f32
      }
      swage.map_store %segment, %output captures(%max, %total : f32, f32)
          : !swage.segment<f32>, memref<?xf32> {
      ^bb0(%value: f32, %m: f32, %t: f32):
        %centered = arith.subf %value, %m : f32
        %normalized = arith.divf %centered, %t : f32
        swage.yield %normalized : f32
      }
      swage_plan.yield
    }
    return
  }
}
