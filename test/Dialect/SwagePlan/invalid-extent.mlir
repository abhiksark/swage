// test/Dialect/SwagePlan/invalid-extent.mlir
// RUN: swage-opt --verify-diagnostics --split-input-file %s

// A task region may take the extent of its segment as a second argument,
// of type index, and may then hold a scalar epilogue after its consumers.
// A partial task and the regions of the queue kernel take no extent, and a
// merge takes one exactly when it is given the range records to read it
// from.

module {
  func.func @extent_of_another_type(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.tasks' op region takes the extent of the bound segment as its second argument, of type index, got 'i32'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>, %extent: i32):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// Without the extent argument a region holds consumers only.
module {
  func.func @epilogue_without_an_extent(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      // expected-error@+1 {{'arith.divf' op is not allowed in the region of 'swage_plan.tasks'; the region holds swage.reduce and swage.map_store operations and ends in swage_plan.yield}}
      %half = arith.divf %sum, %sum : f32
      swage_plan.yield %half : f32
    }
    return
  }
}

// -----

// The epilogue runs once per task, after the reductions.
module {
  func.func @reduction_after_the_epilogue(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>, %extent: index):
      %count = arith.index_cast %extent : index to i32
      // expected-error@+1 {{'swage.reduce' op must come before the scalar epilogue of the task region}}
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The epilogue is the cast, the conversion, and the division of a mean.
module {
  func.func @other_arithmetic(
      %values: memref<?xf32>, %offsets: memref<?xi32>, %output: memref<?xf32>,
      %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>, %extent: index):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      // expected-error@+1 {{'arith.mulf' op is not allowed in the region of 'swage_plan.tasks'; the region holds swage.reduce and swage.map_store operations, then the arith.index_cast, arith.sitofp, and arith.divf operations of a scalar epilogue, and ends in swage_plan.yield}}
      %square = arith.mulf %sum, %sum : f32
      swage_plan.yield %square : f32
    }
    return
  }
}

// -----

// A chunk yields its raw reduction: the merge runs the epilogue once.
module {
  func.func @extent_of_a_chunk(
      %values: memref<?xf32>, %ranges: memref<?xi32>, %scratch: memref<?xf32>,
      %value_count: i32, %partial_count: i32) {
    // expected-error@+1 {{'swage_plan.partial_tasks' op region takes the bound segment as its one argument, of type !swage.segment<T>}}
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>, %extent: index):
      %sum = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
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
  func.func @division_in_a_chunk(
      %values: memref<?xf32>, %ranges: memref<?xi32>, %scratch: memref<?xf32>,
      %value_count: i32, %partial_count: i32) {
    swage_plan.partial_tasks values(%values : memref<?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32) into(%scratch : memref<?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %sum = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      // expected-error@+1 {{'arith.divf' op is not allowed in the region of 'swage_plan.partial_tasks'; the region holds swage.reduce operations and ends in swage_plan.yield}}
      %mean = arith.divf %sum, %sum : f32
      swage_plan.yield %mean : f32
    }
    return
  }
}

// -----

// The bound range of a merge is scratch, whose extent is a number of
// partial results. The extent of the segment needs the range records.
module {
  func.func @extent_without_ranges(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.merge_tasks' op ranges and the extent argument of the region are given together: the extent of a split segment is read from the range records of its partial tasks}}
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>, %extent: index):
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
  func.func @ranges_without_an_extent(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %ranges: memref<?xi32>, %partial_count: i32,
      %merge_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.merge_tasks' op ranges and the extent argument of the region are given together: the extent of a split segment is read from the range records of its partial tasks}}
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        ranges(%ranges : memref<?xi32>) into(%output : memref<?xf32>) {
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
  func.func @wide_ranges(
      %scratch: memref<?xf32>, %output: memref<?xf32>,
      %merges: memref<?xi32>, %ranges: memref<?xi64>, %partial_count: i32,
      %merge_count: i32, %segment_count: i32) {
    // expected-error@+1 {{'swage_plan.merge_tasks' op an element of ranges must have the element type of the merges, 'i32', got 'i64'}}
    swage_plan.merge_tasks scratch(%scratch : memref<?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        ranges(%ranges : memref<?xi64>) into(%output : memref<?xf32>) {
    ^bb0(%partials: !swage.segment<f32>, %extent: index):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
