// test/Conversion/SwageToPlan/invalid-function-count.mlir
// RUN: swage-opt --swage-to-plan --verify-diagnostics --split-input-file %s

// Planning names its companion after the one function of the module, so a
// module must hold exactly one. A bystander function counts.

// expected-error@+1 {{planning requires exactly one function, found 0}}
module {
}

// -----

// expected-error@+1 {{planning requires exactly one function, found 2}}
module {
  func.func @bystander() {
    return
  }
  func.func @segmented_sum(
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
  }
}
