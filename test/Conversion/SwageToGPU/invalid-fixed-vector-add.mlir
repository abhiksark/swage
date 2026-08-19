// test/Conversion/SwageToGPU/invalid-fixed-vector-add.mlir
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s

module {
  func.func @bad_axis(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    // expected-error@+1 {{only swage.program_id axis 0 is supported}}
    %pid = swage.program_id 1
    return
  }
}

// -----

module {
  func.func @bad_offsets(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    %pid = swage.program_id 0
    %offsets = arith.constant dense<0> : vector<128xindex>
    %mask = arith.constant dense<true> : vector<128xi1>
    %passthrough = arith.constant dense<0.0> : vector<128xf32>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %sum = arith.addf %lhs, %rhs : vector<128xf32>
    // expected-error@+1 {{fixed vector add must use canonical program offsets and bounds mask}}
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}
