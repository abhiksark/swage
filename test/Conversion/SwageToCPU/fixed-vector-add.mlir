// test/Conversion/SwageToCPU/fixed-vector-add.mlir
// RUN: swage-opt --swage-fixed-block-to-host='block-size=128' %s | FileCheck %s --check-prefix=HOST
// RUN: swage-opt --pass-pipeline='builtin.module(swage-fixed-block-to-host{block-size=128},convert-scf-to-cf,convert-index-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' %s | FileCheck %s --check-prefix=LLVM

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

// HOST-NOT: swage.
// HOST-NOT: vector.
// HOST: func.func @add_kernel(%[[X:[^,]+]]: !llvm.ptr, %[[Y:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[N:[^)]+]]: i32)
// HOST-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 3 : i64}], backend = "cpu", entry = "add_kernel", launch = {model = "host-call"}, version = 2 : i64}
// HOST: %[[LOWER:.*]] = arith.constant 0 : index
// HOST: %[[UPPER:.*]] = arith.index_cast %[[N]] : i32 to index
// HOST: %[[STEP:.*]] = arith.constant 1 : index
// HOST: scf.for %[[IV:.*]] = %[[LOWER]] to %[[UPPER]] step %[[STEP]] {
// HOST: %[[OFFSET:.*]] = arith.index_cast %[[IV]] : index to i64
// HOST: %[[XPTR:.*]] = llvm.getelementptr %[[X]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// HOST: %[[YPTR:.*]] = llvm.getelementptr %[[Y]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// HOST: %[[OUTPTR:.*]] = llvm.getelementptr %[[OUTPUT]][%[[OFFSET]]] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// HOST: %[[XV:.*]] = llvm.load %[[XPTR]] : !llvm.ptr -> f32
// HOST: %[[YV:.*]] = llvm.load %[[YPTR]] : !llvm.ptr -> f32
// HOST: %[[SUM:.*]] = arith.addf %[[XV]], %[[YV]] : f32
// HOST: llvm.store %[[SUM]], %[[OUTPTR]] : f32, !llvm.ptr
// HOST: }
// HOST: return

// LLVM-NOT: swage.
// LLVM-NOT: vector.
// LLVM-NOT: scf.
// LLVM-NOT: arith.
// LLVM: llvm.func @add_kernel(%[[X:[^,]+]]: !llvm.ptr, %[[Y:[^,]+]]: !llvm.ptr, %[[OUTPUT:[^,]+]]: !llvm.ptr, %[[N:[^)]+]]: i32)
// LLVM: llvm.br ^bb1
// LLVM: ^bb1
// LLVM: llvm.cond_br
// LLVM: llvm.getelementptr %[[X]]
// LLVM: llvm.getelementptr %[[Y]]
// LLVM: llvm.getelementptr %[[OUTPUT]]
// LLVM: llvm.load
// LLVM: llvm.load
// LLVM: llvm.fadd
// LLVM: llvm.store
// LLVM: llvm.br ^bb1
// LLVM: llvm.return
