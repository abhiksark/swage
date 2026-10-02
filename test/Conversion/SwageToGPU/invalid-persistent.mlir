// test/Conversion/SwageToGPU/invalid-persistent.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent use-task-ids' \
// RUN:   --split-input-file %s 2>&1 | FileCheck %s --check-prefix=TASK-IDS

// The experimental persistent kernel implements only the identity f32 sum.
// Static schedules admit max, map chains, and reduction regions; none of
// those may reach the persistent emitter, which would drop them.

// The persistent kernel has its own ABI and loads segment IDs from its own
// task queues, so it cannot honor the task-ID ABI option. The pair is refused
// by option, once for each of the five modules in this file and before any
// of them is admitted. The last module is one the persistent lowering accepts
// on its own, so the refusal is not one of the admission errors below.
// TASK-IDS-COUNT-5: error: persistent lowering does not accept use-task-ids
// TASK-IDS-NOT: gpu.func

module {
  func.func @persistent_map(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{persistent execution does not support swage.map}}
    %scaled = swage.map %segment
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      %two = arith.constant 2.000000e+00 : f32
      %doubled = arith.mulf %value, %two : f32
      swage.yield %doubled : f32
    }
    %sum = swage.reduce %scaled kind<sum>
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
  func.func @persistent_max(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{persistent execution requires kind<sum>}}
    %maximum = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @persistent_region(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{persistent execution requires an identity reduction region}}
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %squared = arith.mulf %value, %value : f32
      swage.yield %squared : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// Persistent execution consumes planned tasks, so the planning admission
// rules apply to it as well.
module {
  func.func @persistent_multi_stage(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %first = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    // expected-error@+1 {{planning requires exactly one reduction stage}}
    %second = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %second, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// An identity sum is admitted by the persistent lowering and produces no
// diagnostic there. It is refused only when use-task-ids is also requested.
module {
  func.func @persistent_identity_sum(
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
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
