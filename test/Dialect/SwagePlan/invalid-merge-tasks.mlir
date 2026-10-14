// test/Dialect/SwagePlan/invalid-merge-tasks.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// A merge task reduces a range of scratch into one output slot. Its counts
// have the word type of its merge records, its region binds a range of
// scratch and holds the reduction over it, and it yields what the output
// holds.

module {
  func.func @wide_merge_count(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i64,
      %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.merge_tasks' op merge_count must have the element type of the merges, 'i32', got 'i64'}}
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i64) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}

// -----

module {
  func.func @partials_of_another_type(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.merge_tasks' op region binds a segment of 'f32', the element type of the scratch, got '!swage.segment<i32>'}}
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<i32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<i32> -> i32 {
      ^bb0(%partial: i32):
        swage.yield %partial : i32
      }
      swage_plan.yield %total : i32
    }
    return
  }
}

// -----

module {
  func.func @store_in_a_merge_region(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32) {
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      // expected-error@+1 {{'swage.map_store' op is not allowed in the region of 'swage_plan.merge_tasks'; the region holds swage.reduce operations and ends in swage_plan.yield}}
      swage.map_store %partials, %output : !swage.segment<f32>, memref<?xf32> {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @no_merged_result(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.merge_tasks' op region must yield 'f32', the element type of the into buffer, got no value}}
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      swage_plan.yield
    }
    return
  }
}
