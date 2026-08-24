// test/Dialect/SwagePlan/effects.mlir
// RUN: swage-opt --cse %s | FileCheck %s

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)

  func.func @classify_around_offsets_write(
      %offsets: memref<?xi32>, %value_count: i32, %segment_count: i32)
      -> (!swage_plan.task_range, !swage_plan.task_range) {
    // CHECK-LABEL: func.func @classify_around_offsets_write(
    // CHECK: %[[BEFORE:.*]] = swage_plan.classify
    %before = swage_plan.classify %offsets, %value_count, %segment_count {cta_chunk_elements = 4096 : i32, kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    %c0 = arith.constant 0 : index
    // CHECK: memref.store
    memref.store %value_count, %offsets[%c0] : memref<?xi32>
    // CHECK: %[[AFTER:.*]] = swage_plan.classify
    %after = swage_plan.classify %offsets, %value_count, %segment_count {cta_chunk_elements = 4096 : i32, kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    // CHECK: return %[[BEFORE]], %[[AFTER]]
    return %before, %after : !swage_plan.task_range, !swage_plan.task_range
  }
}
