// test/Conversion/SwageToCPU/segmented-f64-runner.mlir
// REQUIRES: mlir-runner
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | mlir-opt -pass-pipeline='builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   | FileCheck %s

// The values 1, 2**-53, 2**-53, 5 in the segments [], [1, 2**-53, 2**-53],
// [5], []. In f64 the second sum is 1 exactly, because each addend is half
// a unit in the last place of 1 and rounds to even. An f32 accumulator
// would print the same, so the third value is 0.1, which prints 0.1 as f64
// and 0.100000001490116 after a round trip through f32.
module {
  func.func @segmented_sum_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }

  func.func @main() {
    %values_storage = memref.alloc() : memref<4xf64>
    %offsets_storage = memref.alloc() : memref<5xi32>
    %output_storage = memref.alloc() : memref<4xf64>
    %values = memref.cast %values_storage : memref<4xf64> to memref<?xf64>
    %offsets = memref.cast %offsets_storage : memref<5xi32> to memref<?xi32>
    %output = memref.cast %output_storage : memref<4xf64> to memref<?xf64>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %o0 = arith.constant 0 : i32
    %o3 = arith.constant 3 : i32
    %o4 = arith.constant 4 : i32
    %one = arith.constant 1.0 : f64
    %half_ulp = arith.constant 0x3CA0000000000000 : f64
    %tenth = arith.constant 0.1 : f64
    memref.store %one, %values[%c0] : memref<?xf64>
    memref.store %half_ulp, %values[%c1] : memref<?xf64>
    memref.store %half_ulp, %values[%c2] : memref<?xf64>
    memref.store %tenth, %values[%c3] : memref<?xf64>
    memref.store %o0, %offsets[%c0] : memref<?xi32>
    memref.store %o0, %offsets[%c1] : memref<?xi32>
    memref.store %o3, %offsets[%c2] : memref<?xi32>
    memref.store %o4, %offsets[%c3] : memref<?xi32>
    memref.store %o4, %offsets[%c4] : memref<?xi32>
    call @segmented_sum_f64(%values, %offsets, %output, %o4, %o4)
        : (memref<?xf64>, memref<?xi32>, memref<?xf64>, i32, i32) -> ()
    %unranked = memref.cast %output : memref<?xf64> to memref<*xf64>
    call @printMemrefF64(%unranked) : (memref<*xf64>) -> ()
    memref.dealloc %values_storage : memref<4xf64>
    memref.dealloc %offsets_storage : memref<5xi32>
    memref.dealloc %output_storage : memref<4xf64>
    return
  }

  func.func private @printMemrefF64(memref<*xf64>)
      attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// CHECK-NEXT: [0, 1, 0.1, 0]
