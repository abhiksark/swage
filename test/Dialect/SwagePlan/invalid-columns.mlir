// test/Dialect/SwagePlan/invalid-columns.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// A task operation over rank-two values takes the number of columns, has
// policy<column> or policy<sequential>, takes no task buffer, and stores its
// scalars into rank-two rows.

// The column policy is the kernel of rank-two values.
module {
  func.func @column_of_scalars(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op feature_count is given exactly when the values have rank two, got 'memref<?xf32>' with a feature_count}}
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @rows_without_a_feature_count(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op feature_count is given exactly when the values have rank two, got 'memref<?x?xf32>' without a feature_count}}
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @column_without_rows(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op policy<column> reduces the columns of rank-two values and takes a feature_count}}
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// Nothing is combined across the threads of a block of rank-two values.
module {
  func.func @rows_across_a_block(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op rank-two values take policy<column> or policy<sequential>, got policy<cta>}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @wide_feature_count(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i64) {
    // expected-error@+1 {{'swage_plan.tasks' op feature_count must have the element type of the offsets, 'i32', got 'i64'}}
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i64)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @rows_into_scalars(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op into must have rank two for rank-two values, got 'memref<?xf32>'}}
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  func.func @scalars_into_rows(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op into must have rank one for rank-one values, got 'memref<?x?xf32>'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// A task of rank-two values is one segment, in order.
module {
  func.func @rows_with_ids(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %ids: memref<?xi32>, %task_count: i32,
      %value_count: i32, %segment_count: i32,
      %feature_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op a task of rank-two values is one segment, in order, and takes no ids}}
    swage_plan.tasks policy<column>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}
