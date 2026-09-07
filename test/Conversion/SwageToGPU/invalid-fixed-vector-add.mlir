// test/Conversion/SwageToGPU/invalid-fixed-vector-add.mlir
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-fixed-block-to-host='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' --verify-diagnostics --split-input-file --mlir-print-ir-after-failure %s 2>&1 | FileCheck %s --check-prefix=UNCHANGED --implicit-check-not=gpu.module
// RUN: swage-opt --swage-fixed-block-to-host='block-size=128' --verify-diagnostics --split-input-file --mlir-print-ir-after-failure %s 2>&1 | FileCheck %s --check-prefix=UNCHANGED --implicit-check-not=llvm.getelementptr

// UNCHANGED-LABEL: func.func @unsupported_element(
// UNCHANGED-SAME: memref<?xbf16>
// UNCHANGED-LABEL: func.func @mixed_inputs(
// UNCHANGED-SAME: memref<?xf16>
// UNCHANGED-SAME: memref<?xf32>
// UNCHANGED-LABEL: func.func @mixed_output(
// UNCHANGED-SAME: memref<?xf8E4M3FN>
// UNCHANGED-SAME: memref<?xf8E5M2>

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

// -----

module {
  // expected-error@+1 {{requires three rank-one identity-layout memrefs}}
  func.func @bad_layout(
      %x: memref<?xf32, strided<[2]>>, %y: memref<?xf32>,
      %output: memref<?xf32>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 128 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<128xindex>
    %base_vector = vector.broadcast %base : index to vector<128xindex>
    %offsets = arith.addi %base_vector, %lane : vector<128xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<128xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<128xindex>
    %zero = arith.constant 0.0 : f32
    %passthrough = vector.broadcast %zero : f32 to vector<128xf32>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32, strided<[2]>>, vector<128xindex>, vector<128xi1>,
          vector<128xf32> into vector<128xf32>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %sum = arith.addf %lhs, %rhs : vector<128xf32>
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{requires three rank-one identity-layout memrefs}}
  func.func @unsupported_element(
      %x: memref<?xbf16>, %y: memref<?xbf16>,
      %output: memref<?xbf16>, %n: i32) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{requires identical pointer element types}}
  func.func @mixed_inputs(
      %x: memref<?xf16>, %y: memref<?xf32>,
      %output: memref<?xf16>, %n: i32) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{requires identical pointer element types}}
  func.func @mixed_output(
      %x: memref<?xf8E4M3FN>, %y: memref<?xf8E4M3FN>,
      %output: memref<?xf8E5M2>, %n: i32) {
    return
  }
}
