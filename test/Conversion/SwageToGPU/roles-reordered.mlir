// test/Conversion/SwageToGPU/roles-reordered.mlir
// The kernel takes its arguments in the order of its layout, whatever order
// the segment function declares them in: the roles name the arguments. This
// function is the one in segmented-sum.mlir with its arguments reversed, and
// it lowers to the same module.
//
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   %S/segmented-sum.mlir > %t.ordered
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | diff %t.ordered -
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s

// CHECK: gpu.func @segmented_sum(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32)
// CHECK-NOT: swage.role

module {
  func.func @segmented_sum(
      %segment_count: i32 {swage.role = #swage.role<segment_count>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %values: memref<?xf32> {swage.role = #swage.role<values>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
