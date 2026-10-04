// test/Dialect/SwagePlan/invalid-persistent-tasks.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// A persistent task operation ties every count, task buffer, and counter to
// the word type of the offsets. Its block, partial, and warp regions bind a
// range of the values and its merge region a range of scratch, and each
// yields what the buffer that receives its scalar holds.

module {
  func.func @wide_merge_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i64, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.persistent_tasks' op merge_count must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i64)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
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
  func.func @wide_counters(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi64>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.persistent_tasks' op an element of counters must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi64>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
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

// The scalar of each task is stored by the task operation, so no region
// holds a store.
module {
  func.func @store_in_the_merge_region(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32) {
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      // expected-error@+1 {{'swage.map_store' op is not allowed in the region of 'swage_plan.persistent_tasks'; the region holds swage.reduce operations and ends in swage_plan.yield}}
      swage.map_store %partials, %output : !swage.segment<f32>, memref<?xf32> {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
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

// The scalar of a partial task goes to scratch.
module {
  func.func @partial_without_a_scalar(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.persistent_tasks' op region must yield 'f32', the element type of the scratch buffer, got no value}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
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

// A merge reduces partial results, so its region binds a range of scratch.
module {
  func.func @merge_over_words(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.persistent_tasks' op region binds a segment of 'f32', the element type of the scratch, got '!swage.segment<i32>'}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<i32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<i32> -> i32 {
      ^bb0(%value: i32):
        swage.yield %value : i32
      }
      swage_plan.yield %total : i32
    } warp {
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
