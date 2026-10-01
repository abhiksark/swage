// test/Conversion/SwageToCPU/invalid-unverified.mlir
// RUN: swage-opt --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --swage-segmented-reduction-to-scf \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --mlir-very-unsafe-disable-verifier-on-parsing \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s

// A pass manager verifies after each pass and never before the first one, so
// a caller that skips the verifier can hand the lowering IR the dialect would
// refuse. The lowering dereferences region bodies, so it repeats the checks
// it depends on and reports them instead of crashing. Every module here is
// invalid IR: the parse-time verifier is off for this file only.

module {
  func.func @region_with_two_blocks(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segment region requires exactly one block}}
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    ^bb1(%other: f32):
      swage.yield %other : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @region_without_terminator(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segment region must yield an f32 value}}
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %doubled = arith.addf %value, %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @region_with_an_extra_argument(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segment region requires an f32 element argument followed by f32 captures}}
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32, %uncaptured: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @region_ending_in_another_terminator(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segment region must yield an f32 value}}
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      llvm.unreachable
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// The function type has no result, which is all the signature check reads.
module {
  func.func @return_with_an_operand(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    // expected-error@+1 {{segmented reduction must return void}}
    return %sum : f32
  }
}

// -----

// A mapped segment has one use here, but the use is not a segment consumer.
module {
  func.func @returned_map(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{swage.map result must have exactly one segment consumer}}
    %mapped = swage.map %segment : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return %mapped : !swage.segment<f32>
  }
}
