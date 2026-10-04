// test/Conversion/SwageToGPU/fixed-vector-multiply-low-precision.mlir
// RUN: swage-opt --swage-fixed-block-to-host='block-size=128' --split-input-file %s | FileCheck %s --check-prefix=SCALAR --implicit-check-not=vector. --implicit-check-not=f8E
// RUN: swage-opt --swage-fixed-block-to-gpu='block-size=128' --split-input-file %s | FileCheck %s --check-prefix=SCALAR --implicit-check-not=vector. --implicit-check-not=f8E
// RUN: swage-opt --pass-pipeline='builtin.module(swage-fixed-block-to-host{block-size=128},convert-scf-to-cf,convert-index-to-llvm,convert-arith-to-llvm,convert-cf-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)' --split-input-file %s | FileCheck %s --check-prefix=LLVM --implicit-check-not=arith. --implicit-check-not=llvm.call --implicit-check-not=f8E
// RUN: swage-opt --pass-pipeline='builtin.module(swage-fixed-block-to-gpu{block-size=128},gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' --split-input-file %s | FileCheck %s --check-prefix=LLVM --implicit-check-not=arith. --implicit-check-not=llvm.call --implicit-check-not=f8E

// The same semantic fixtures exercise both emitters and their upstream
// conversions. FP16 keeps two-byte storage and rounds an f32 product to f16.
module {
  func.func @multiply_f16(%x: memref<?xf16>, %y: memref<?xf16>, %out: memref<?xf16>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 128 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<128xindex>
    %base_vector = vector.broadcast %base : index to vector<128xindex>
    %offsets = arith.addi %base_vector, %lane : vector<128xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<128xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<128xindex>
    %zero = arith.constant dense<0.0> : vector<128xf16>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %zero : memref<?xf16>, vector<128xindex>, vector<128xi1>, vector<128xf16> into vector<128xf16>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %zero : memref<?xf16>, vector<128xindex>, vector<128xi1>, vector<128xf16> into vector<128xf16>
    %product = arith.mulf %lhs, %rhs : vector<128xf16>
    vector.scatter %out[%c0] [%offsets], %mask, %product : memref<?xf16>, vector<128xindex>, vector<128xi1>, vector<128xf16>
    return
  }
}

// SCALAR-LABEL: {{(func.func|gpu.func)}} @multiply_f16(
// SCALAR: %[[XP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// SCALAR: %[[YP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// SCALAR: %[[OP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, f16
// SCALAR: %[[X:.*]] = llvm.load %[[XP]] : !llvm.ptr -> f16
// SCALAR: %[[Y:.*]] = llvm.load %[[YP]] : !llvm.ptr -> f16
// SCALAR: %[[XF:.*]] = arith.extf %[[X]] : f16 to f32
// SCALAR: %[[YF:.*]] = arith.extf %[[Y]] : f16 to f32
// SCALAR: %[[PRODUCT:.*]] = arith.mulf %[[XF]], %[[YF]] : f32
// SCALAR: %[[OUT:.*]] = arith.truncf %[[PRODUCT]] : f32 to f16
// SCALAR: llvm.store %[[OUT]], %[[OP]] : f16, !llvm.ptr

// LLVM-LABEL: llvm.func @multiply_f16(
// LLVM: llvm.getelementptr {{.*}} !llvm.ptr, f16
// LLVM: llvm.getelementptr {{.*}} !llvm.ptr, f16
// LLVM: %[[OP:.*]] = llvm.getelementptr {{.*}} !llvm.ptr, f16
// LLVM: %[[X:.*]] = llvm.load {{.*}} -> f16
// LLVM: %[[Y:.*]] = llvm.load {{.*}} -> f16
// LLVM: %[[XF:.*]] = llvm.fpext %[[X]] : f16 to f32
// LLVM: %[[YF:.*]] = llvm.fpext %[[Y]] : f16 to f32
// LLVM: %[[PRODUCT:.*]] = llvm.fmul %[[XF]], %[[YF]] : f32
// LLVM: %[[OUT:.*]] = llvm.fptrunc %[[PRODUCT]] : f32 to f16
// LLVM: llvm.store %[[OUT]], %[[OP]] : f16, !llvm.ptr

// -----

// E4M3FN must become byte loads/stores and scalar software conversion, not
// f8 LLVM types or casts requiring native FP8 support.
module {
  func.func @add_e4m3(%x: memref<?xf8E4M3FN>, %y: memref<?xf8E4M3FN>, %out: memref<?xf8E4M3FN>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 128 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<128xindex>
    %base_vector = vector.broadcast %base : index to vector<128xindex>
    %offsets = arith.addi %base_vector, %lane : vector<128xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<128xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<128xindex>
    %zero = arith.constant dense<0.0> : vector<128xf8E4M3FN>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %zero : memref<?xf8E4M3FN>, vector<128xindex>, vector<128xi1>, vector<128xf8E4M3FN> into vector<128xf8E4M3FN>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %zero : memref<?xf8E4M3FN>, vector<128xindex>, vector<128xi1>, vector<128xf8E4M3FN> into vector<128xf8E4M3FN>
    %product = arith.mulf %lhs, %rhs : vector<128xf8E4M3FN>
    vector.scatter %out[%c0] [%offsets], %mask, %product : memref<?xf8E4M3FN>, vector<128xindex>, vector<128xi1>, vector<128xf8E4M3FN>
    return
  }
}

// SCALAR-LABEL: {{(func.func|gpu.func)}} @add_e4m3(
// SCALAR: %[[XP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// SCALAR: %[[YP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// SCALAR: %[[OP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// SCALAR: %[[X:.*]] = llvm.load %[[XP]] : !llvm.ptr -> i8
// SCALAR: %[[Y:.*]] = llvm.load %[[YP]] : !llvm.ptr -> i8
// SCALAR: arith.extui %[[X]] : i8 to i32
// SCALAR: arith.uitofp {{.*}} : i32 to f32
// SCALAR: arith.mulf {{.*}} : f32
// SCALAR: arith.extui %[[Y]] : i8 to i32
// SCALAR: arith.uitofp {{.*}} : i32 to f32
// SCALAR: arith.mulf {{.*}} : f32
// SCALAR: %[[PRODUCT:.*]] = arith.mulf {{.*}} : f32
// SCALAR: arith.bitcast %[[PRODUCT]] : f32 to i32
// SCALAR: arith.addf {{.*}} : f32
// SCALAR: %[[OUT:.*]] = arith.trunci {{.*}} : i32 to i8
// SCALAR: llvm.store %[[OUT]], %[[OP]] : i8, !llvm.ptr

// LLVM-LABEL: llvm.func @add_e4m3(
// LLVM: llvm.getelementptr {{.*}} !llvm.ptr, i8
// LLVM: llvm.getelementptr {{.*}} !llvm.ptr, i8
// LLVM: %[[OP:.*]] = llvm.getelementptr {{.*}} !llvm.ptr, i8
// LLVM: %[[X:.*]] = llvm.load {{.*}} -> i8
// LLVM: %[[Y:.*]] = llvm.load {{.*}} -> i8
// LLVM: llvm.zext %[[X]] : i8 to i32
// LLVM: llvm.zext %[[Y]] : i8 to i32
// LLVM: %[[PRODUCT:.*]] = llvm.fmul {{.*}} : f32
// LLVM: llvm.bitcast %[[PRODUCT]] : f32 to i32
// LLVM: %[[OUT:.*]] = llvm.trunc {{.*}} : i32 to i8
// LLVM: llvm.store %[[OUT]], %[[OP]] : i8, !llvm.ptr

// -----

// E5M2 has a distinct bias and infinity encoding but the same byte ABI.
module {
  func.func @add_e5m2(%x: memref<?xf8E5M2>, %y: memref<?xf8E5M2>, %out: memref<?xf8E5M2>, %n: i32) {
    %pid = swage.program_id 0
    %block = arith.constant 128 : index
    %base = arith.muli %pid, %block : index
    %lane = vector.step : vector<128xindex>
    %base_vector = vector.broadcast %base : index to vector<128xindex>
    %offsets = arith.addi %base_vector, %lane : vector<128xindex>
    %n_index = arith.index_cast %n : i32 to index
    %n_vector = vector.broadcast %n_index : index to vector<128xindex>
    %mask = arith.cmpi slt, %offsets, %n_vector : vector<128xindex>
    %zero = arith.constant dense<0.0> : vector<128xf8E5M2>
    %c0 = arith.constant 0 : index
    %lhs = vector.gather %x[%c0] [%offsets], %mask, %zero : memref<?xf8E5M2>, vector<128xindex>, vector<128xi1>, vector<128xf8E5M2> into vector<128xf8E5M2>
    %rhs = vector.gather %y[%c0] [%offsets], %mask, %zero : memref<?xf8E5M2>, vector<128xindex>, vector<128xi1>, vector<128xf8E5M2> into vector<128xf8E5M2>
    %product = arith.mulf %lhs, %rhs : vector<128xf8E5M2>
    vector.scatter %out[%c0] [%offsets], %mask, %product : memref<?xf8E5M2>, vector<128xindex>, vector<128xi1>, vector<128xf8E5M2>
    return
  }
}

// SCALAR-LABEL: {{(func.func|gpu.func)}} @add_e5m2(
// SCALAR: %[[XP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// SCALAR: %[[YP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// SCALAR: %[[OP:.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr, i64) -> !llvm.ptr, i8
// SCALAR: %[[X:.*]] = llvm.load %[[XP]] : !llvm.ptr -> i8
// SCALAR: %[[Y:.*]] = llvm.load %[[YP]] : !llvm.ptr -> i8
// SCALAR: arith.extui %[[X]] : i8 to i32
// SCALAR: arith.uitofp {{.*}} : i32 to f32
// SCALAR: arith.mulf {{.*}} : f32
// SCALAR: arith.extui %[[Y]] : i8 to i32
// SCALAR: arith.uitofp {{.*}} : i32 to f32
// SCALAR: arith.mulf {{.*}} : f32
// SCALAR: %[[PRODUCT:.*]] = arith.mulf {{.*}} : f32
// SCALAR: arith.bitcast %[[PRODUCT]] : f32 to i32
// SCALAR: arith.addf {{.*}} : f32
// SCALAR: %[[OUT:.*]] = arith.trunci {{.*}} : i32 to i8
// SCALAR: llvm.store %[[OUT]], %[[OP]] : i8, !llvm.ptr

// LLVM-LABEL: llvm.func @add_e5m2(
// LLVM: llvm.getelementptr {{.*}} !llvm.ptr, i8
// LLVM: llvm.getelementptr {{.*}} !llvm.ptr, i8
// LLVM: %[[OP:.*]] = llvm.getelementptr {{.*}} !llvm.ptr, i8
// LLVM: %[[X:.*]] = llvm.load {{.*}} -> i8
// LLVM: %[[Y:.*]] = llvm.load {{.*}} -> i8
// LLVM: llvm.zext %[[X]] : i8 to i32
// LLVM: llvm.zext %[[Y]] : i8 to i32
// LLVM: %[[PRODUCT:.*]] = llvm.fmul {{.*}} : f32
// LLVM: llvm.bitcast %[[PRODUCT]] : f32 to i32
// LLVM: %[[OUT:.*]] = llvm.trunc {{.*}} : i32 to i8
// LLVM: llvm.store %[[OUT]], %[[OP]] : i8, !llvm.ptr
