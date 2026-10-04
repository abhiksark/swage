// test/Conversion/SwageToGPU/invalid-fixed-block-size.mlir
// RUN: not swage-opt --swage-fixed-block-to-gpu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=NOT-POSITIVE,ZERO
// RUN: not swage-opt --swage-fixed-block-to-gpu='block-size=0' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=NOT-POSITIVE,ZERO
// RUN: not swage-opt --swage-fixed-block-to-gpu='block-size=-128' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefixes=NOT-POSITIVE,NEGATIVE
// RUN: not swage-opt --swage-fixed-block-to-gpu='block-size=1025' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOO-LARGE
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=1024' %s \
// RUN:   | FileCheck %s --check-prefix=LARGEST

// The block size is the launch width of the kernel. The option has no usable
// default, so a pipeline that omits it is refused, and 1024 is the largest
// block a CUDA device launches.
module {
  func.func @add_kernel(
      %x: memref<?xf32>, %y: memref<?xf32>, %output: memref<?xf32>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 1024 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<1024xindex>
    %base_vector = vector.broadcast %base : index to vector<1024xindex>
    %offsets = arith.addi %base_vector, %lane : vector<1024xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<1024xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<1024xindex>
    %zero = arith.constant 0.0 : f32
    %passthrough = vector.broadcast %zero : f32 to vector<1024xf32>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<1024xindex>, vector<1024xi1>,
          vector<1024xf32> into vector<1024xf32>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %passthrough
        : memref<?xf32>, vector<1024xindex>, vector<1024xi1>,
          vector<1024xf32> into vector<1024xf32>
    %sum = arith.addf %lhs, %rhs : vector<1024xf32>
    vector.scatter %output[%c0] [%offsets], %mask, %sum
        : memref<?xf32>, vector<1024xindex>, vector<1024xi1>,
          vector<1024xf32>
    return
  }
}

// ZERO: error: block-size must be a positive integer, got 0
// NEGATIVE: error: block-size must be a positive integer, got -128
// NOT-POSITIVE-NOT: gpu.func

// TOO-LARGE: error: block-size must be at most 1024, got 1025
// TOO-LARGE-NOT: gpu.func

// LARGEST: gpu.func @add_kernel
// LARGEST-SAME: nvvm.reqntid = array<i32: 1024, 1, 1>
