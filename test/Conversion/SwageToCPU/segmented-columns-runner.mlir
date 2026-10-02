// test/Conversion/SwageToCPU/segmented-columns-runner.mlir
// REQUIRES: mlir-runner
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | mlir-opt -pass-pipeline='builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   | FileCheck %s

// Four rows of three columns, [1, 10, 100], [2, 20, 200], [3, 30, 300], and
// [4, 40, 400], in the segments [], [row 0], [rows 1 to 3], and []. Each
// column has its own magnitude, so a result that mixed two columns, or two
// segments, shows in its digits. An empty segment gives 0 in every column.
module {
  func.func @segmented_sum_r2(
      %values: memref<?x?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?x?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %feature_count: i32 {swage.role = #swage.role<feature_count>}) {
    %sid = swage.segment_id 0
    %col = swage.segment_id 1
    %segment = swage.make_segment %values, %offsets, %sid column(%col)
        : memref<?x?xf32>, memref<?xi32>, index, index
          -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid, %col] : memref<?x?xf32>
    return
  }

  func.func @main() {
    %values_storage = memref.alloc() : memref<4x3xf32>
    %offsets_storage = memref.alloc() : memref<5xi32>
    %output_storage = memref.alloc() : memref<4x3xf32>
    %values = memref.cast %values_storage
        : memref<4x3xf32> to memref<?x?xf32>
    %offsets = memref.cast %offsets_storage : memref<5xi32> to memref<?xi32>
    %output = memref.cast %output_storage
        : memref<4x3xf32> to memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %ten = arith.constant 10.0 : f32
    // values[row, column] = (row + 1) * 10 ** column
    scf.for %row = %c0 to %c4 step %c1 {
      %next = arith.addi %row, %c1 : index
      %next_word = arith.index_cast %next : index to i32
      %base = arith.sitofp %next_word : i32 to f32
      %tens = arith.mulf %base, %ten : f32
      %hundreds = arith.mulf %tens, %ten : f32
      memref.store %base, %values[%row, %c0] : memref<?x?xf32>
      memref.store %tens, %values[%row, %c1] : memref<?x?xf32>
      memref.store %hundreds, %values[%row, %c2] : memref<?x?xf32>
    }
    %o0 = arith.constant 0 : i32
    %o1 = arith.constant 1 : i32
    %o3 = arith.constant 3 : i32
    %o4 = arith.constant 4 : i32
    memref.store %o0, %offsets[%c0] : memref<?xi32>
    memref.store %o0, %offsets[%c1] : memref<?xi32>
    memref.store %o1, %offsets[%c2] : memref<?xi32>
    memref.store %o4, %offsets[%c3] : memref<?xi32>
    memref.store %o4, %offsets[%c4] : memref<?xi32>
    call @segmented_sum_r2(%values, %offsets, %output, %o4, %o4, %o3)
        : (memref<?x?xf32>, memref<?xi32>, memref<?x?xf32>, i32, i32, i32)
          -> ()
    %unranked = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
    memref.dealloc %values_storage : memref<4x3xf32>
    memref.dealloc %offsets_storage : memref<5xi32>
    memref.dealloc %output_storage : memref<4x3xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
      attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 3] strides = [3, 1] data =
// CHECK-NEXT: {{\[\[}}0, 0, 0],
// CHECK-NEXT: [1, 10, 100],
// CHECK-NEXT: [9, 90, 900],
// CHECK-NEXT: [0, 0, 0]]
