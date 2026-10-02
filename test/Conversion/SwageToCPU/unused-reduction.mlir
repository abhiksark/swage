// test/Conversion/SwageToCPU/unused-reduction.mlir
// A program may hold a reduction that nothing reads. The lowerings fuse its
// map and lower it as a stage like any other: fusion removes maps and
// nothing else.
//
// RUN: swage-opt --swage-segmented-reduction-to-scf %s | FileCheck %s
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s --check-prefix=GPU

// CHECK-LABEL: func.func @unused_maximum(
// CHECK: scf.for
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.mulf
// CHECK: arith.maximumf
// CHECK: scf.for {{.*}} iter_args
// CHECK: arith.addf
// CHECK: memref.store

// GPU: gpu.func @unused_maximum(
// GPU: arith.mulf
// GPU: arith.maximumf
// GPU: arith.addf

module {
  func.func @unused_maximum(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %squares = swage.map %segment : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%value: f32):
      %square = arith.mulf %value, %value : f32
      swage.yield %square : f32
    }
    %unused = swage.reduce %squares kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
