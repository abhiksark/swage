// test/Conversion/SwageToGPU/segmented-sum-region.mlir
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --implicit-check-not=swage.

// A non-identity reduction region is inlined into the block-stride loop.
// The region survives as written; replacing the libdevice exponential with a
// native instruction happens in the PTX codegen path, not here.
module {
  func.func @segmented_sum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
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

// CHECK: gpu.func @segmented_sum
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {kind = "i32", origin = "user", source_index = 3 : i64}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_sum", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK: scf.for %{{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (f32) {
// CHECK:   %[[VALUE:.*]] = llvm.load %{{.*}} : !llvm.ptr -> f32
// CHECK:   %[[SQUARE:.*]] = arith.mulf %[[VALUE]], %[[VALUE]] : f32
// CHECK:   %[[SCALED:.*]] = math.exp2 %[[SQUARE]] : f32
// CHECK:   %[[NEXT_ACC:.*]] = arith.addf %[[ACC]], %[[SCALED]] : f32
// CHECK:   scf.yield %[[NEXT_ACC]] : f32
// CHECK: }
// CHECK: gpu.all_reduce add
