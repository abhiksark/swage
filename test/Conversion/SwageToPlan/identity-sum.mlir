// test/Conversion/SwageToPlan/identity-sum.mlir
// RUN: swage-opt --swage-to-plan %s | FileCheck %s --check-prefix=DEFAULT
// RUN: swage-opt --swage-to-plan='warp-max-elements=64 cta-chunk-elements=2048' %s | FileCheck %s --check-prefix=CUSTOM
// RUN: not swage-opt --swage-to-plan='warp-max-elements=0' %s 2>&1 | FileCheck %s --check-prefix=BAD-LIMIT
// RUN: not swage-opt --swage-to-plan='warp-max-elements=2147483648' %s 2>&1 | FileCheck %s --check-prefix=BAD-LIMIT
// RUN: not swage-opt --swage-to-plan='cta-chunk-elements=0' %s 2>&1 | FileCheck %s --check-prefix=BAD-LIMIT
// RUN: not swage-opt --swage-to-plan='cta-chunk-elements=2147483648' %s 2>&1 | FileCheck %s --check-prefix=BAD-LIMIT
// RUN: not swage-opt --swage-to-plan='warp-max-elements=33 cta-chunk-elements=32' %s 2>&1 | FileCheck %s --check-prefix=BAD-LIMIT

module {
  func.func @segmented_sum(
      %destination: memref<?xf32>, %boundaries: memref<?xi32>,
      %elements: memref<?xf32>) {
    %segment_id = swage.segment_id 0
    %segment = swage.make_segment %elements, %boundaries, %segment_id
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %destination[%segment_id] : memref<?xf32>
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
// DEFAULT-SAME: %[[VALUES:.*]]: memref<?xf32>, %[[OFFSETS:.*]]: memref<?xi32>) -> !swage_plan.task_range
// DEFAULT: %[[VALUE_COUNT_INDEX:.*]] = memref.dim %[[VALUES]], %{{.*}} : memref<?xf32>
// DEFAULT: %[[OFFSET_COUNT:.*]] = memref.dim %[[OFFSETS]], %{{.*}} : memref<?xi32>
// DEFAULT: %[[ONE:.*]] = arith.constant 1 : index
// DEFAULT: %[[SEGMENT_COUNT_INDEX:.*]] = arith.subi %[[OFFSET_COUNT]], %[[ONE]] : index
// DEFAULT: %[[VALUE_COUNT:.*]] = arith.index_cast %[[VALUE_COUNT_INDEX]] : index to i32
// DEFAULT: %[[SEGMENT_COUNT:.*]] = arith.index_cast %[[SEGMENT_COUNT_INDEX]] : index to i32
// DEFAULT: %[[TASKS:.*]] = swage_plan.classify %[[OFFSETS]], %[[VALUE_COUNT]], %[[SEGMENT_COUNT]] {cta_chunk_elements = 4096 : i32, kernel = @segmented_sum, policies = [#swage_plan.policy<warp>, #swage_plan.policy<cta>], warp_max_elements = 32 : i32} : memref<?xi32>, i32, i32 -> !swage_plan.task_range
// DEFAULT: return %[[TASKS]] : !swage_plan.task_range

// CUSTOM-LABEL: func.func private @segmented_sum__swage_plan(
// CUSTOM: swage_plan.classify {{.*}} {cta_chunk_elements = 2048 : i32, {{.*}} warp_max_elements = 64 : i32}

// BAD-LIMIT: error: planning limits must satisfy 0 < warp-max-elements <= cta-chunk-elements <= INT32_MAX
