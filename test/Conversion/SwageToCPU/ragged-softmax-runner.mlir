// test/Conversion/SwageToCPU/ragged-softmax-runner.mlir
// RUN: swage-opt --swage-segmented-reduction-to-scf %s \
// RUN:   | mlir-opt -pass-pipeline='builtin.module(func.func(convert-scf-to-cf,convert-math-to-llvm,convert-arith-to-llvm),finalize-memref-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)' \
// RUN:   | mlir-runner -e main -entry-point-result=void \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_runner_utils%shlibext \
// RUN:       -shared-libs=%llvm_lib_dir/libmlir_c_runner_utils%shlibext \
// RUN:   | FileCheck %s

// Ragged softmax executed end to end. Every expected value is exact in f32:
// a singleton normalizes to 1, the pair {0, ln2} gives exactly 1/3 and 2/3
// because exp2(-1) is exactly 0.5, and four equal values each give 0.25. The
// four values of 100 would overflow to infinity without the maximum shift.
// Slot 7 lies past the final offset and must keep its sentinel, which is what
// makes "map_store never writes past offsets[-1]" a checked invariant.
module {
  func.func @ragged_softmax(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
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
        : !swage.segment<f32>, memref<?xf32> {
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
    %values_storage = memref.alloc() : memref<8xf32>
    %offsets_storage = memref.alloc() : memref<5xi32>
    %output_storage = memref.alloc() : memref<8xf32>
    %values = memref.cast %values_storage : memref<8xf32> to memref<?xf32>
    %offsets = memref.cast %offsets_storage : memref<5xi32> to memref<?xi32>
    %output = memref.cast %output_storage : memref<8xf32> to memref<?xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %o0 = arith.constant 0 : i32
    %o1 = arith.constant 1 : i32
    %o3 = arith.constant 3 : i32
    %o4 = arith.constant 4 : i32
    %o7 = arith.constant 7 : i32
    %o8 = arith.constant 8 : i32
    %sentinel = arith.constant -1.0 : f32
    %five = arith.constant 5.0 : f32
    %zero = arith.constant 0.0 : f32
    %ln2 = arith.constant 0.693147182 : f32
    %hundred = arith.constant 1.0e+02 : f32
    %seven = arith.constant 7.0 : f32

    // The output is one slot longer than the final offset, prefilled with a
    // sentinel, so a stray store past the live range is observable.
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %sentinel, %output[%i] : memref<?xf32>
    }

    memref.store %five, %values[%c0] : memref<?xf32>
    memref.store %zero, %values[%c1] : memref<?xf32>
    memref.store %ln2, %values[%c2] : memref<?xf32>
    memref.store %hundred, %values[%c3] : memref<?xf32>
    memref.store %hundred, %values[%c4] : memref<?xf32>
    memref.store %hundred, %values[%c5] : memref<?xf32>
    memref.store %hundred, %values[%c6] : memref<?xf32>
    memref.store %seven, %values[%c7] : memref<?xf32>

    // Segments: {5}, {0, ln2}, empty, {100, 100, 100, 100}.
    memref.store %o0, %offsets[%c0] : memref<?xi32>
    memref.store %o1, %offsets[%c1] : memref<?xi32>
    memref.store %o3, %offsets[%c2] : memref<?xi32>
    memref.store %o3, %offsets[%c3] : memref<?xi32>
    memref.store %o7, %offsets[%c4] : memref<?xi32>

    call @ragged_softmax(%values, %offsets, %output, %o8, %o4)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    %unranked = memref.cast %output : memref<?xf32> to memref<*xf32>
    call @printMemrefF32(%unranked) : (memref<*xf32>) -> ()
    memref.dealloc %values_storage : memref<8xf32>
    memref.dealloc %offsets_storage : memref<5xi32>
    memref.dealloc %output_storage : memref<8xf32>
    return
  }

  func.func private @printMemrefF32(memref<*xf32>)
      attributes {llvm.emit_c_interface}
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [8] strides = [1] data =
// CHECK-NEXT: [1, 0.333333, 0.666667, 0.25, 0.25, 0.25, 0.25, -1]
