// test/Conversion/SwageToCPU/invalid-segmented-reduction.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s

module {
  func.func @bad_axis(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    // expected-error@+1 {{only swage.segment_id axis 0 is supported, got axis 1}}
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segmented reduction supports only kind<sum> and kind<max>, got kind<min>}}
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
  // expected-error@+1 {{swage.role<offsets> requires a rank-one i32 memref with a dynamic size, the identity layout, and the default memory space, got 'memref<?xi64>'}}
  func.func @bad_offsets(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi64> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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
// rejected on both backends; exponentials must be written as math.exp2. The
// diagnostic names the operation and lists the ones a region accepts.
module {
  func.func @unsupported_region_operation(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      // expected-error@+1 {{operation 'math.exp' is unsupported inside a segment region; a region accepts arith.constant, arith.addf, arith.subf, arith.mulf, arith.divf, arith.maximumf, arith.minimumf, and math.exp2}}
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{operation 'swage.extent' is unsupported by segmented reduction lowering}}
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      // expected-error@+1 {{operation 'arith.constant' is unsupported inside a segment region; every result must be f32, got 'i32'}}
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    // expected-error@+1 {{only swage.segment_id axis 0 is supported, got axis 1}}
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

// -----

// Exactly one output terminal, so a scalar store beside a map_store is an
// error rather than a silent choice between two output shapes.
module {
  // expected-error@+1 {{segmented reduction requires exactly one output terminal: a memref.store of a reduction at output[segment_id] or a swage.map_store into the output, found 1 memref.store and 1 swage.map_store}}
  func.func @two_terminals(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    swage.map_store %segment, %output : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    return
  }
}

// -----

// A map_store may only write the function's output buffer.
module {
  func.func @map_store_wrong_buffer(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    // expected-error@+1 {{swage.map_store must write the function output buffer}}
    swage.map_store %segment, %values : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    return
  }
}

// -----

// An unreachable second block is still a second block.
module {
  // expected-error@+1 {{segmented reduction requires one block, got 2 blocks}}
  func.func @two_blocks(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  ^unreachable:
    return
  }
}

// -----

module {
  // expected-error@+1 {{segmented reduction requires one segment_id, one make_segment, at least one reduce, and one return, found 2 segment_id, 1 make_segment, 1 reduce, and 1 return}}
  func.func @two_segment_ids(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %unused = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{segmented reduction requires one segment_id, one make_segment, at least one reduce, and one return, found 1 segment_id, 2 make_segment, 1 reduce, and 1 return}}
  func.func @two_segments(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %unused = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// A program that only maps and stores has no reduction to lower.
module {
  // expected-error@+1 {{segmented reduction requires one segment_id, one make_segment, at least one reduce, and one return, found 1 segment_id, 1 make_segment, 0 reduce, and 1 return}}
  func.func @map_store_without_reduce(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    swage.map_store %segment, %output : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32):
      %doubled = arith.addf %value, %value : f32
      swage.yield %doubled : f32
    }
    return
  }
}

// -----

// The segment must view the values argument. The output buffer has the same
// type, so only this rule keeps a kernel from reducing what it writes.
module {
  func.func @segment_of_the_output(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    // expected-error@+1 {{make_segment must bind the function values and offsets at segment_id}}
    %segment = swage.make_segment %output, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// The result goes to the output argument and nowhere else.
module {
  func.func @store_into_the_values(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    // expected-error@+1 {{segmented reduction result must be stored at output[segment_id]}}
    memref.store %sum, %values[%sid] : memref<?xf32>
    return
  }
}

// -----

// A segment function says what every argument is. An argument without a role
// is refused; there is no positional default.
module {
  // expected-error@+1 {{segment function argument #4 declares no swage.role; every argument declares one of values, offsets, output, value_count, and segment_count}}
  func.func @argument_without_a_role(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{segment function declares no swage.role<value_count>; it declares values, offsets, output, value_count, and segment_count once each}}
  func.func @missing_value_count(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// The element type comes from the values, and f32 is the one admitted today.
module {
  // expected-error@+1 {{swage.role<values> requires a rank-one f32 memref with a dynamic size, the identity layout, and the default memory space, got 'memref<?xf64>'}}
  func.func @double_values(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %sum, %output[%sid] : memref<?xf64>
    return
  }
}

// -----

// The output holds elements of the values, in a buffer the kernel can
// address: a fixed size is refused like any other layout.
module {
  // expected-error@+1 {{swage.role<output> requires a rank-one memref of 'f32', the element type of the values, with a dynamic size, the identity layout, and the default memory space, got 'memref<8xf32>'}}
  func.func @fixed_size_output(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<8xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<8xf32>
    return
  }
}

// -----

// The counts have the width of an offset.
module {
  // expected-error@+1 {{swage.role<segment_count> requires 'i32', the element type of the offsets, got 'i64'}}
  func.func @wide_segment_count(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i64 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// A kernel writes its result to the output and returns nothing.
module {
  // expected-error@+1 {{segment function must have no result, got '(memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> f32'}}
  func.func @returns_the_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) -> f32 {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return %sum : f32
  }
}

// -----

// The roles name the arguments, so their order is free. With the two f32
// buffers exchanged, the segment views the last argument and the result
// goes to the first.
module {
  func.func @segment_of_the_output_by_role(
      %first: memref<?xf32> {swage.role = #swage.role<output>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %last: memref<?xf32> {swage.role = #swage.role<values>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    // expected-error@+1 {{make_segment must bind the function values and offsets at segment_id}}
    %segment = swage.make_segment %first, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %last[%sid] : memref<?xf32>
    return
  }
}
