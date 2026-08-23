// test/Conversion/SwageToPlan/identity-sum.mlir
// RUN: swage-opt --swage-to-plan %s | FileCheck %s --check-prefix=DEFAULT
// RUN: swage-opt --swage-to-plan='warp-max-elements=64' %s | FileCheck %s --check-prefix=CUSTOM
// RUN: not swage-opt --swage-to-plan='warp-max-elements=-1' %s 2>&1 | FileCheck %s --check-prefix=BAD-THRESHOLD
// RUN: not swage-opt --swage-to-plan='warp-max-elements=2147483648' %s 2>&1 | FileCheck %s --check-prefix=BAD-THRESHOLD

module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %segment_id = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %segment_id
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%segment_id] : memref<?xf32>
    return
  }
}

// DEFAULT-LABEL: func.func @segmented_sum(
// DEFAULT: %[[SID:.*]] = swage.segment_id 0
// DEFAULT: %[[SEGMENT:.*]] = swage.make_segment
// DEFAULT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum>
// DEFAULT: memref.store %[[SUM]]
// DEFAULT: return
// DEFAULT-LABEL: func.func private @segmented_sum__swage_plan(
// DEFAULT-SAME: %[[OFFSETS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32) -> !swage_plan.task_range
// DEFAULT: %[[TASKS:.*]] = swage_plan.classify %[[OFFSETS]], %[[VALUE_COUNT]], %[[SEGMENT_COUNT]] {kernel = @segmented_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
// DEFAULT: return %[[TASKS]] : !swage_plan.task_range

// CUSTOM-LABEL: func.func private @segmented_sum__swage_plan(
// CUSTOM: swage_plan.classify {{.*}} warp_max_elements = 64 : i32

// BAD-THRESHOLD: error: warp-max-elements must be a nonnegative i32
