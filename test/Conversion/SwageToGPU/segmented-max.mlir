// test/Conversion/SwageToGPU/segmented-max.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s

module {
  func.func @segmented_max(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-NOT: swage.
// CHECK: gpu.module @segmented_max_module
// CHECK: gpu.func @segmented_max
// CHECK-SAME: swage.kernel_contract = {arguments = [{access = "read", kind = "ptr", origin = "user", source_index = 0 : i64}, {access = "read", kind = "ptr", origin = "user", source_index = 1 : i64}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {key = "value_count", kind = "i32", origin = "derived"}, {key = "segment_count", kind = "i32", origin = "derived"}], backend = "cuda", entry = "segmented_max", launch = {block = array<i32: 128, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// CHECK: %[[IDENTITY:.*]] = arith.constant 0xFF800000 : f32
// CHECK: %[[LOCAL:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:   %[[VALUE:.*]] = llvm.load {{.*}} : !llvm.ptr -> f32
// CHECK:   %[[NEXT:.*]] = arith.maximumf %[[ACC]], %[[VALUE]] : f32
// CHECK:   scf.yield %[[NEXT]] : f32
// CHECK: }
// CHECK: %[[TOTAL:.*]] = gpu.all_reduce maximumf %[[LOCAL]] uniform
// CHECK: llvm.store %[[TOTAL]]
// CHECK-NOT: swage.
