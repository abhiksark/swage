// test/Conversion/SwageToCPU/invalid-segmented-reduction.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s

module {
  func.func @bad_axis(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{only swage.segment_id axis 0 is supported}}
    %sid = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @unsupported_min(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segmented reduction supports only kind<sum> and kind<max>}}
    %minimum = swage.reduce %segment kind<min>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %minimum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{segmented reduction requires rank-one f32 values, rank-one i32 offsets}}
  func.func @bad_offsets(
      %values: memref<?xf32>, %offsets: memref<?xi64>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi64>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @captured_transform(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segment captures must be f32 results of a swage.reduce in the same function}}
    %sum = swage.reduce %segment captures(%value_count : i32) kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32, %capture: i32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// math.exp becomes a libdevice call the PTX path cannot resolve, so it is
// rejected on both backends; exponentials must be written as math.exp2.
module {
  func.func @unsupported_region_operation(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      // expected-error@+1 {{operation is unsupported inside a segment region}}
      %exponential = math.exp %value : f32
      swage.yield %exponential : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @extra_operation(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{operation is unsupported by segmented reduction lowering}}
    %extent = swage.extent %segment : !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// A mapped segment is fused into its consumer, so it may have exactly one.
module {
  func.func @multi_use_map(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{swage.map result must have exactly one segment consumer}}
    %doubled = swage.map %segment
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      %scaled = arith.mulf %value, %value : f32
      swage.yield %scaled : f32
    }
    %first = swage.reduce %doubled kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    %second = swage.reduce %doubled kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %first, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// An admitted operation name is not enough; every result must be f32.
module {
  func.func @integer_region_constant(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      // expected-error@+1 {{every result must be f32}}
      %count = arith.constant 3 : i32
      %widened = arith.sitofp %count : i32 to f32
      %scaled = arith.mulf %value, %widened : f32
      swage.yield %scaled : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// A capture defined by a reduce is still rejected unless its result is f32.
module {
  func.func @wide_capture(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %wide = swage.reduce %segment kind<max> : !swage.segment<f32> -> f64 {
    ^bb0(%value: f32):
      %widened = arith.extf %value : f32 to f64
      swage.yield %widened : f64
    }
    // expected-error@+1 {{segment captures must be f32 results of a swage.reduce in the same function}}
    %sum = swage.reduce %segment captures(%wide : f64) kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32, %w: f64):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// Rules fire in order, so a bad axis is reported even when the terminal count
// is also wrong.
module {
  func.func @bad_axis_and_terminals(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{only swage.segment_id axis 0 is supported}}
    %sid = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
