// test/Conversion/SwageToGPU/invalid-segmented-task-ids.mlir
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' \
// RUN:   --swage-plan-to-gpu \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=128' \
// RUN:   --swage-plan-to-gpu \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' --swage-plan-to-gpu \
// RUN:   --verify-diagnostics --split-input-file %s

// A kernel that takes its segments from a task buffer is launched from a
// host plan, and a plan describes one capture-free reduction whose result is
// stored per segment. The warp and block task-ID kernels and the fused mixed
// kernel share that admission, so each case is refused by all three. The
// same four programs are refused by the split stages in invalid-split.mlir.

module {
  func.func @captured_map(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    // expected-error@+1 {{planning requires capture-free maps}}
    %shifted = swage.map %segment captures(%maximum : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %captured: f32):
      %centered = arith.subf %value, %captured : f32
      swage.yield %centered : f32
    }
    %sum = swage.reduce %shifted kind<sum>
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
  func.func @captured_reduction(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    // expected-error@+1 {{planning requires a capture-free reduction}}
    %sum = swage.reduce %segment captures(%maximum : f32) kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32, %captured: f32):
      %centered = arith.subf %value, %captured : f32
      swage.yield %centered : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @multi_stage(
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

module {
  func.func @map_store_terminal(
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
    // expected-error@+1 {{planning requires memref.store of the reduction result}}
    swage.map_store %segment, %output
        : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    return
  }
}
