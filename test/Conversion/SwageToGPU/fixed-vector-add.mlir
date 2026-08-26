// test/Conversion/SwageToGPU/fixed-vector-add.mlir
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' %s | FileCheck %s

module {
  func.func @add_kernel(
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

// CHECK-NOT: swage.
// CHECK-NOT: vector.
// CHECK: gpu.module @add_kernel_module
// CHECK: gpu.func @add_kernel(%[[X:[^,]+]]: !llvm.ptr, %[[Y:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[N:[^)]+]]: i32) kernel
// CHECK-SAME: nvvm.reqntid = array<i32: 128, 1, 1>
// CHECK: %[[BLOCK:.*]] = gpu.block_id x
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: scf.if
// CHECK: llvm.load
// CHECK: arith.addf
// CHECK: llvm.store
// CHECK: gpu.return
// CHECK-NOT: swage.
// CHECK-NOT: vector.
