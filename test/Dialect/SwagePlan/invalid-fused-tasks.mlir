// test/Dialect/SwagePlan/invalid-fused-tasks.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// A fused task operation ties every count and its task buffer to the word
// type of the offsets, and both of its regions hold the reduction of the
// bound segment and yield what the into buffer holds.

module {
  func.func @wide_warp_task_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %warp_task_count: i64, %cta_task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.fused_tasks' op warp_task_count must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) warp_task_count(%warp_task_count : i64)
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

// -----

module {
  func.func @wide_ids(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi64>, %value_count: i32,
      %warp_task_count: i32, %cta_task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.fused_tasks' op an element of ids must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi64>) warp_task_count(%warp_task_count : i32)
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

// -----

// The scalar of each task is stored by the task operation, so neither
// region holds a store.
module {
  func.func @store_in_the_block_region(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %warp_task_count: i32, %cta_task_count: i32, %segment_count: i32) {
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
      // expected-error@+1 {{'swage.map_store' op is not allowed in the region of 'swage_plan.fused_tasks'; the region holds swage.reduce operations and ends in swage_plan.yield}}
      swage.map_store %segment, %output : !swage.segment<f32>, memref<?xf32> {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @warp_region_without_a_result(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %warp_task_count: i32, %cta_task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.fused_tasks' op region must yield 'f32', the element type of the into buffer, got no value}}
    swage_plan.fused_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) warp_task_count(%warp_task_count : i32)
        cta_task_count(%cta_task_count : i32)
        into(%output : memref<?xf32>) warp {
    ^bb0(%segment: !swage.segment<f32>):
      swage_plan.yield
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

// -----

module {
  func.func @block_region_of_another_type(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %warp_task_count: i32, %cta_task_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.fused_tasks' op region binds a segment of 'f32', the element type of the values, got '!swage.segment<i32>'}}
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
    ^bb0(%segment: !swage.segment<i32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<i32> -> i32 {
      ^bb0(%value: i32):
        swage.yield %value : i32
      }
      swage_plan.yield %total : i32
    }
    return
  }
}
