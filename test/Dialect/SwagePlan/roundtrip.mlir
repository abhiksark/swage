// test/Dialect/SwagePlan/roundtrip.mlir
// RUN: swage-opt %s | swage-opt | FileCheck %s

module {
  func.func private @semantic_sum(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32)

  func.func @classify(%offsets: memref<?xi32>, %value_count: i32,
                      %segment_count: i32) -> !swage_plan.task_range {
    // CHECK: %[[TASKS:.*]] = swage_plan.classify %[[OFFSETS:.*]], %[[VALUE_COUNT:.*]], %[[SEGMENT_COUNT:.*]] {cta_chunk_elements = 4096 : i32, kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    %tasks = swage_plan.classify %offsets, %value_count, %segment_count {cta_chunk_elements = 4096 : i32, kernel = @semantic_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
    return %tasks : !swage_plan.task_range
  }
}
