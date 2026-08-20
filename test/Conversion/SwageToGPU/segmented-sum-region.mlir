// test/Conversion/SwageToGPU/segmented-sum-region.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s

// A non-identity reduction region is inlined into the block-stride loop.
// The region survives as written; replacing the libdevice exponential with a
// native instruction happens in the PTX codegen path, not here.
module {
  func.func @segmented_sum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      %square = arith.mulf %value, %value : f32
      %scaled = math.exp2 %square : f32
      swage.yield %scaled : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-NOT: swage.
// CHECK: gpu.func @segmented_sum
// CHECK: scf.for %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK:   %[[VALUE:.*]] = llvm.load %{{.*}} : !llvm.ptr -> f32
// CHECK:   %[[SQUARE:.*]] = arith.mulf %[[VALUE]], %[[VALUE]] : f32
// CHECK:   %[[SCALED:.*]] = math.exp2 %[[SQUARE]] : f32
// CHECK:   %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[SCALED]] : f32
// CHECK:   scf.yield %[[NEXT_ACC]] : f32
// CHECK: }
// CHECK: gpu.all_reduce add
