// test/Conversion/SwageToCPU/ragged-softmax-columns-runner.mlir
// REQUIRES: mlir-runner
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | mlir-opt -pass-pipeline='builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   | FileCheck %s

// The softmax over rank-two values executed end to end. Six rows of three
// columns in the segments [row 0], [], and [rows 1 to 4]. Row 5 lies past
// the final offset and must keep its sentinel in every column.
//
// Every expected value is exact in f32, and each column of the long segment
// has its own pattern, so a result that mixed two columns shows:
// - Column 0 holds four values of 100, which give 0.25 each. They would
//   overflow to infinity without the shift by the maximum.
// - Column 1 holds 0 in row 2 and -1000 elsewhere, which gives 1 in row 2 and
//   0 elsewhere: exp2 of -1442.7 underflows to zero.
// - Column 2 holds 5 in rows 1 and 2 and -1000 in rows 3 and 4, which gives
//   0.5, 0.5, 0, and 0.
// The one row of the first segment normalizes to 1 in every column.
module {
  func.func @ragged_softmax_r2(
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
    %max = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %shifted = swage.map %segment captures(%max : f32)
        : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32, %m: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      swage.yield %exponential : f32
    }
    %total = swage.reduce %shifted kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    swage.map_store %segment, %output captures(%max, %total : f32, f32)
        : !swage.segment<f32>, memref<?x?xf32> {
    ^bb0(%value: f32, %m: f32, %t: f32):
      %log2e = arith.constant 1.44269502 : f32
      %centered = arith.subf %value, %m : f32
      %scaled = arith.mulf %centered, %log2e : f32
      %exponential = math.exp2 %scaled : f32
      %normalized = arith.divf %exponential, %t : f32
      swage.yield %normalized : f32
    }
    return
  }

  func.func @main() {
    %values_storage = memref.alloc() : memref<6x3xf32>
    %offsets_storage = memref.alloc() : memref<4xi32>
    %output_storage = memref.alloc() : memref<6x3xf32>
    %values = memref.cast %values_storage
        : memref<6x3xf32> to memref<?x?xf32>
    %offsets = memref.cast %offsets_storage : memref<4xi32> to memref<?xi32>
    %output = memref.cast %output_storage
        : memref<6x3xf32> to memref<?x?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %sentinel = arith.constant -1.0 : f32
    %zero = arith.constant 0.0 : f32
    %five = arith.constant 5.0 : f32
    %seven = arith.constant 7.0 : f32
    %hundred = arith.constant 1.0e+02 : f32
    %far = arith.constant -1.0e+03 : f32

    // The output is prefilled with a sentinel, so a store to a row that no
    // segment covers is observable. Column 0 is 100 and the other columns
    // are -1000 in every row, before the rows below are written.
    scf.for %row = %c0 to %c6 step %c1 {
      scf.for %column = %c0 to %c3 step %c1 {
        memref.store %sentinel, %output[%row, %column] : memref<?x?xf32>
        memref.store %far, %values[%row, %column] : memref<?x?xf32>
      }
      memref.store %hundred, %values[%row, %c0] : memref<?x?xf32>
    }
    memref.store %seven, %values[%c0, %c1] : memref<?x?xf32>
    memref.store %zero, %values[%c2, %c1] : memref<?x?xf32>
    memref.store %five, %values[%c1, %c2] : memref<?x?xf32>
    memref.store %five, %values[%c2, %c2] : memref<?x?xf32>

    // Segments: [row 0], [], [rows 1 to 4].
    %o0 = arith.constant 0 : i32
    %o1 = arith.constant 1 : i32
    %o3 = arith.constant 3 : i32
    %o5 = arith.constant 5 : i32
    %o6 = arith.constant 6 : i32
    memref.store %o0, %offsets[%c0] : memref<?xi32>
    memref.store %o1, %offsets[%c1] : memref<?xi32>
    memref.store %o1, %offsets[%c2] : memref<?xi32>
    memref.store %o5, %offsets[%c3] : memref<?xi32>
    call @ragged_softmax_r2(%values, %offsets, %output, %o6, %o3, %o3)
        : (memref<?x?xf32>, memref<?xi32>, memref<?x?xf32>, i32, i32, i32)
          -> ()
    %unranked = memref.cast %output : memref<?x?xf32> to memref<*xf32>
    call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
    memref.dealloc %values_storage : memref<6x3xf32>
    memref.dealloc %offsets_storage : memref<4xi32>
    memref.dealloc %output_storage : memref<6x3xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
      attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [6, 3] strides = [3, 1] data =
// CHECK-NEXT: {{\[\[}}1, 1, 1],
// CHECK-NEXT: [0.25, 0, 0.5],
// CHECK-NEXT: [0.25, 1, 0.5],
// CHECK-NEXT: [0.25, 0, 0],
// CHECK-NEXT: [0.25, 0, 0],
// CHECK-NEXT: [-1, -1, -1]]
