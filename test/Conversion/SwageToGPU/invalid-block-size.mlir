// test/Conversion/SwageToGPU/invalid-block-size.mlir
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=0' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOT-POSITIVE
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=1025' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOO-LARGE
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=96' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=THREE-WARPS
// RUN: not swage-opt --swage-segmented-reduction-to-gpu='block-size=160' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FIVE-WARPS
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=100' %s \
// RUN:   | FileCheck %s --check-prefix=FOUR-WARPS
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=40' %s \
// RUN:   | FileCheck %s --check-prefix=TWO-WARPS

// The CTA reduction lowers through gpu.all_reduce, whose second stage combines
// the per-warp partials with an XOR butterfly and stores from every
// participating lane. Each lane holds the complete reduction only when the
// warp count is a power of two, so 96 (three warps) and 160 (five warps) are
// rejected. A partly filled last warp is fine: 100 and 40 round up to four and
// two warps.
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
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// NOT-POSITIVE: error: block-size must be a positive integer, got 0
// NOT-POSITIVE-NOT: gpu.func

// TOO-LARGE: error: block-size must be at most 1024, got 1025
// TOO-LARGE-NOT: gpu.func

// THREE-WARPS: error: block-size must give a power-of-two warp count, got 96 (3 warps)
// THREE-WARPS-NOT: gpu.func

// FIVE-WARPS: error: block-size must give a power-of-two warp count, got 160 (5 warps)
// FIVE-WARPS-NOT: gpu.func

// FOUR-WARPS: gpu.func @segmented_sum
// FOUR-WARPS-SAME: nvvm.reqntid = array<i32: 100, 1, 1>

// TWO-WARPS: gpu.func @segmented_sum
// TWO-WARPS-SAME: nvvm.reqntid = array<i32: 40, 1, 1>
