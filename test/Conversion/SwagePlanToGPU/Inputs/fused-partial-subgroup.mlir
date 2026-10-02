// test/Conversion/SwagePlanToGPU/Inputs/fused-partial-subgroup.mlir
// Input of fused-tasks.mlir: a launch width the target admits, four
// subgroups with the last one partly filled, which a fused block cannot use.

module {
  func.func @fused(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %warp_task_count: i32, %cta_task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 100 : i32} {
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) warp_task_count(%warp_task_count : i32)
        cta_task_count(%cta_task_count : i32)
        into(%output : memref<?xf32>) warp {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
