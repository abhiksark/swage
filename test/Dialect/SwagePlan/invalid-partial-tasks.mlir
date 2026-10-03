// test/Dialect/SwagePlan/invalid-partial-tasks.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// A partial task reduces one chunk of values into its scratch slot. Its
// counts have the word type of its range records, its region holds the
// reduction of the chunk, and it yields what the scratch buffer holds.

module {
  func.func @wide_value_count(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i64, %partial_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op value_count must have the element type of the ranges, 'i32', got 'i64'}}
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i64) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
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
  func.func @narrow_partial_count(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i16) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op partial_count must have the element type of the ranges, 'i32', got 'i16'}}
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i16) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
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
  func.func @chunk_of_another_type(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op region binds a segment of 'f32', the element type of the values, got '!swage.segment<i32>'}}
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<i32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<i32> -> i32 {
      ^bb0(%value: i32):
        swage.yield %value : i32
      }
      swage_plan.yield %total : i32
    }
    return
  }
}

// -----

// A partial task writes scratch and nothing else, so its region holds no
// store.
module {
  func.func @store_in_a_partial_region(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i32) {
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      // expected-error@+1 {{'swage.map_store' op is not allowed in the region of 'swage_plan.partial_tasks'; the region holds swage.reduce operations and ends in swage_plan.yield}}
      swage.map_store %chunk, %scratch : !swage.segment<f32>, memref<?xf32> {
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
  func.func @no_partial_result(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op region must yield 'f32', the element type of the into buffer, got no value}}
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      swage_plan.yield
    }
    return
  }
}

// -----

module {
  func.func @partial_result_of_another_type(
      %values: memref<?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf64>, %value_count: i32, %partial_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op region must yield 'f64', the element type of the into buffer, got 'f32'}}
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf64>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}

// -----

// Rank-two values are rows of feature_count columns, and scratch then holds
// one row per task.

module {
  func.func @rows_without_a_feature_count(
      %values: memref<?x?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?x?xf32>, %value_count: i32, %partial_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op values must have rank 1 without a feature_count, got 'memref<?x?xf32>'}}
    swage_plan.partial_tasks values(%values : memref<?x?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?x?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
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
  func.func @scalar_scratch_of_rows(
      %values: memref<?x?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?xf32>, %value_count: i32, %partial_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op into must have rank 2 with a feature_count, got 'memref<?xf32>'}}
    swage_plan.partial_tasks values(%values : memref<?x?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32)
        feature_count(%feature_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
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
  func.func @wide_feature_count(
      %values: memref<?x?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?x?xf32>, %value_count: i32, %partial_count: i32,
      %feature_count: i64) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op feature_count must have the element type of the records, 'i32', got 'i64'}}
    swage_plan.partial_tasks values(%values : memref<?x?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32)
        feature_count(%feature_count : i64) into(%scratch : memref<?x?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
