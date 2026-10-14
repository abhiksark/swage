// test/Conversion/SwageToCPU/segmented-min-runner.mlir
// REQUIRES: mlir-runner
// RUN: swage-opt --swage-to-plan='schedule=sequential' --swage-plan-to-scf %s \
// RUN:   | mlir-opt -pass-pipeline='builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   | FileCheck %s

// The values 3, 1, 2, 5 in the segments [], [3], [1, 2, 5], []. The minimum
// of the third segment, 1, is neither its first element nor its maximum,
// and an empty segment gives the identity, positive infinity.
module {
  func.func @segmented_min(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %minimum = swage.reduce %segment kind<min>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %minimum, %output[%sid] : memref<?xf32>
    return
  }

  func.func @main() {
    %values_storage = memref.alloc() : memref<4xf32>
    %offsets_storage = memref.alloc() : memref<5xi32>
    %output_storage = memref.alloc() : memref<4xf32>
    %values = memref.cast %values_storage : memref<4xf32> to memref<?xf32>
    %offsets = memref.cast %offsets_storage : memref<5xi32> to memref<?xi32>
    %output = memref.cast %output_storage : memref<4xf32> to memref<?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %o0 = arith.constant 0 : i32
    %o1 = arith.constant 1 : i32
    %o4 = arith.constant 4 : i32
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32
    %v5 = arith.constant 5.0 : f32
    memref.store %v3, %values[%c0] : memref<?xf32>
    memref.store %v1, %values[%c1] : memref<?xf32>
    memref.store %v2, %values[%c2] : memref<?xf32>
    memref.store %v5, %values[%c3] : memref<?xf32>
    memref.store %o0, %offsets[%c0] : memref<?xi32>
    memref.store %o0, %offsets[%c1] : memref<?xi32>
    memref.store %o1, %offsets[%c2] : memref<?xi32>
    memref.store %o4, %offsets[%c3] : memref<?xi32>
    memref.store %o4, %offsets[%c4] : memref<?xi32>
    call @segmented_min(%values, %offsets, %output, %o4, %o4)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    %unranked = memref.cast %output : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
    memref.dealloc %values_storage : memref<4xf32>
    memref.dealloc %offsets_storage : memref<5xi32>
    memref.dealloc %output_storage : memref<4xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
      attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [4] strides = [1] data =
// CHECK-NEXT: [inf, 3, 1, inf]
