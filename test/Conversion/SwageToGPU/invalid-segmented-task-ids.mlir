// test/Conversion/SwageToGPU/invalid-segmented-task-ids.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=32 use-task-ids=true' --verify-diagnostics --split-input-file %s

module {
  func.func @segmented_min(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{segmented reduction supports only kind<sum> and kind<max>}}
    %maximum = swage.reduce %segment kind<min>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

