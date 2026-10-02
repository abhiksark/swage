// test/Conversion/SwageToGPU/invalid-fixed-kernel-symbols.mlir
// The fixed-block lowering replaces its kernel function by a gpu.module
// named after it. Nothing may refer to the function, and the name of the
// module must be free. Both are checked before the module is changed: the
// last RUN line prints each module after its failure and finds no kernel in
// it.
//
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file \
// RUN:   --mlir-print-ir-after-failure %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNCHANGED --implicit-check-not=gpu.func

// UNCHANGED-LABEL: func.func @clashes(
// UNCHANGED: vector.scatter
// UNCHANGED-LABEL: func.func @referenced(
// UNCHANGED: vector.scatter

module {
  // expected-note@+1 {{defined here}}
  memref.global "private" @clashes_module : memref<4xi32>
  // expected-error@+1 {{lowering @clashes creates @clashes_module, which the module already defines}}
  func.func @clashes(
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
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel function @referenced is referenced 1 times; lowering it to a GPU kernel removes it, so it must have no symbol use}}
  func.func @referenced(
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
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<128xindex>, vector<128xi1>, vector<128xf32>
    return
  }
  // expected-note@+1 {{referenced here}}
  gpu.module @launcher attributes {entry = @referenced} {
  }
}
