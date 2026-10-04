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
    // expected-error@+1 {{only swage.program_id axis 0 is supported, got axis 1}}
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
    // expected-error@+1 {{fixed elementwise operation must use canonical program offsets and bounds mask}}
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{fixed elementwise operation requires three rank-one identity-layout memrefs of f32, f16, f8E4M3FN, or f8E5M2 and one i32, got '(memref<?xf32, strided<[2]>>, memref<?xf32>, memref<?xf32>, i32) -> ()'}}
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
  func.func @rank_two_vector(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    // expected-error@+1 {{only fixed-length rank-one vectors are supported, got 'vector<2x64xf32>'}}
    %zero = arith.constant dense<0.0> : vector<2x64xf32>
    return
  }
}

// -----

module {
  // A scalable vector of 128 has 128 lanes times a runtime factor, so it is
  // not the 128 threads of the block.
  func.func @scalable_vector(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    // expected-error@+1 {{only fixed-length rank-one vectors are supported, got 'vector<[128]xindex>'}}
    %lane = vector.step : vector<[128]xindex>
    return
  }
}

// -----

module {
  func.func @narrow_vector(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    // expected-error@+1 {{vector width 64 does not match requested block size 128}}
    %lane = vector.step : vector<64xindex>
    return
  }
}

// -----

module {
  // expected-error@+1 {{only default-memory-space pointers are supported, got 'memref<?xf32, 1>'}}
  func.func @device_memory_space(
      %x: memref<?xf32, 1>, %y: memref<?xf32>, %output: memref<?xf32>,
      %n: i32) {
    return
  }
}

// -----

module {
  // A memory space need not be an integer. It is still not the default one.
  // expected-error@+1 {{only default-memory-space pointers are supported, got 'memref<?xf32, "device">'}}
  func.func @named_memory_space(
      %x: memref<?xf32>, %y: memref<?xf32, "device">, %output: memref<?xf32>,
      %n: i32) {
    return
  }
}

// -----

module {
  // An unreachable second block is still a second block.
  // expected-error@+1 {{fixed elementwise operation requires one straight-line block, got 2 blocks}}
  func.func @two_blocks(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    return
  ^unreachable:
    return
  }
}

// -----

module {
  // A declaration has no block at all.
  // expected-error@+1 {{fixed elementwise operation requires one straight-line block, got 0 blocks}}
  func.func private @declaration(
      memref<?xf32>, memref<?xf32>, memref<?xf32>, i32)
}

// -----

module {
  func.func @unsupported_operation(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    %c0 = arith.constant 0 : index
    // expected-error@+1 {{operation 'memref.load' is unsupported by fixed elementwise lowering}}
    %value = memref.load %x[%c0] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{expected one program_id, two gathers, one floating-point add or multiply, and one scatter, found 1 program_id, 0 gathers, 0 floating-point adds or multiplies, and 0 scatters}}
  func.func @no_vector_add(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    %pid = swage.program_id 0
    return
  }
}

// -----

module {
  // The second gather reads %x again, so the kernel is not x + y.
  // expected-error@+1 {{gathers, arithmetic, and scatter do not form a fixed elementwise operation}}
  func.func @gathers_one_input_twice(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
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
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %rhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough
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
  func.func @stores_a_gather(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
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
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
          into vector<128xf32>
    %sum = arith.addf %lhs, %rhs : vector<128xf32>
    // The add is computed and dropped; the scatter writes %lhs unchanged.
    // expected-error@+1 {{scatter value must be the vector floating-point add or multiply}}
    vector.scatter %output[%c0] [%offsets], %mask, %lhs
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}

// -----

// expected-error@+1 {{expected exactly one kernel function, found 2}}
module {
  func.func @first(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    return
  }
  func.func @second(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    return
  }
}

// -----

// expected-error@+1 {{expected exactly one kernel function, found 0}}
module {
}

// -----

module {
  // expected-error@+1 {{fixed elementwise operation requires three rank-one identity-layout memrefs of f32, f16, f8E4M3FN, or f8E5M2 and one i32, got '(memref<?xbf16>, memref<?xbf16>, memref<?xbf16>, i32) -> ()'}}
  func.func @unsupported_element(
      %x: memref<?xbf16>, %y: memref<?xbf16>,
      %output: memref<?xbf16>, %n: i32) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{fixed elementwise operation requires identical pointer element types, got '(memref<?xf16>, memref<?xf32>, memref<?xf16>, i32) -> ()'}}
  func.func @mixed_inputs(
      %x: memref<?xf16>, %y: memref<?xf32>,
      %output: memref<?xf16>, %n: i32) {
    return
  }
}

// -----

module {
  // expected-error@+1 {{fixed elementwise operation requires identical pointer element types, got '(memref<?xf8E4M3FN>, memref<?xf8E4M3FN>, memref<?xf8E5M2>, i32) -> ()'}}
  func.func @mixed_output(
      %x: memref<?xf8E4M3FN>, %y: memref<?xf8E4M3FN>,
      %output: memref<?xf8E5M2>, %n: i32) {
    return
  }
}
