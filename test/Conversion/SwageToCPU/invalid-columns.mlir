// test/Conversion/SwageToCPU/invalid-columns.mlir
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu \
// RUN:   --verify-diagnostics --split-input-file %s

// What admission requires of a segment function over rank-two values, one
// rule per case. Both backends admit through the same code.

// The number of columns is an argument of its own: the function declares
// it exactly when its values are rows.
module {
  // expected-error@+1 {{a segment function declares swage.role<feature_count> exactly when its values have rank two, got 'memref<?x?xf32>' without a feature_count}}
  func.func @rows_without_a_feature_count(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{a segment function declares swage.role<feature_count> exactly when its values have rank two, got 'memref<?xf32>' with a feature_count}}
  func.func @feature_count_of_scalars(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
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
  // expected-error@+1 {{swage.role<values> requires a rank-two f32 or f64 memref with dynamic sizes, the identity layout, and the default memory space, got 'memref<?x4xf32>'}}
  func.func @rows_of_a_fixed_width(
      %values: memref<?x4xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x4xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{swage.role<values> requires a rank-two f32 or f64 memref with dynamic sizes, the identity layout, and the default memory space, got 'memref<?x?xf16>'}}
  func.func @rows_of_halves(
      %values: memref<?x?xf16> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf16> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf16>, memref<?xi32>, index, index -> !swage.segment<f16>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f16> -> f16 {
    ^bb0(%value: f16):
      swage.yield %value : f16
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf16>
    return
  }
}

// -----

// The output of a reduction over rows has one row per segment.
module {
  // expected-error@+1 {{swage.role<output> requires a rank-two memref of 'f32', the element type of the values, with dynamic sizes, the identity layout, and the default memory space, got 'memref<?xf32>'}}
  func.func @rows_into_scalars(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
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
  // expected-error@+1 {{swage.role<feature_count> requires 'i32', the element type of the offsets, got 'i64'}}
  func.func @wide_feature_count(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i64 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }
}

// -----

// A program instance over rows is one segment and one column.
module {
  // expected-error@+1 {{segmented reduction requires two segment_id, one make_segment, at least one reduce, and one return, found 1 segment_id, 1 make_segment, 1 reduce, and 1 return}}
  func.func @one_axis(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid column(%sid)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %sid] : memref<?x?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{a segment function over rank-two values requires one swage.segment_id 0, the segment, and one swage.segment_id 1, the column, found axes 0 0}}
  func.func @two_segment_axes(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{a segment function over rank-two values requires one swage.segment_id 0, the segment, and one swage.segment_id 1, the column, found axes 0 2}}
  func.func @third_axis(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 2
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }
}

// -----

// The offsets delimit rows, so the segment is axis 0 and the column axis 1.
module {
  func.func @column_as_the_segment(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    // expected-error@+1 {{make_segment must bind the function values and offsets at segment_id 0, with segment_id 1 as its column}}
    %segment = swage.make_segment %values, %offsets, %col column(%sid)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }
}

// -----

module {
  func.func @stored_in_the_wrong_slot(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    // expected-error@+1 {{the result of a reduction over rank-two values must be stored at output[segment_id 0, segment_id 1]}}
    memref.store %sum, %output[%col, %sid] : memref<?x?xf32>
    return
  }
}
