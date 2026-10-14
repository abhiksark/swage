// test/Conversion/SwageToGPU/invalid-block-size.mlir
// RUN: not swage-opt --swage-to-plan='schedule=direct block-threads=0' \
// RUN:   --swage-plan-to-gpu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOT-POSITIVE
// RUN: not swage-opt --swage-to-plan='schedule=direct block-threads=1025' \
// RUN:   --swage-plan-to-gpu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=TOO-LARGE
// RUN: not swage-opt --swage-to-plan='schedule=direct block-threads=96' \
// RUN:   --swage-plan-to-gpu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=THREE-WARPS
// RUN: not swage-opt --swage-to-plan='schedule=direct block-threads=160' \
// RUN:   --swage-plan-to-gpu %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=FIVE-WARPS
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=100' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=FOUR-WARPS
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=40' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=TWO-WARPS

// The CTA reduction lowers through gpu.all_reduce, whose second stage combines
// the per-warp partials with an XOR butterfly and stores from every
// participating lane. Each lane holds the complete reduction only when the
// warp count is a power of two, so 96 (three warps) and 160 (five warps) are
// rejected. A partly filled last warp is fine: 100 and 40 round up to four and
// two warps.
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
      swage.yield %value : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// NOT-POSITIVE: error: block-threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got 0
// NOT-POSITIVE-NOT: gpu.func

// TOO-LARGE: error: block-threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got 1025
// TOO-LARGE-NOT: gpu.func

// THREE-WARPS: error: block-threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got 96
// THREE-WARPS-NOT: gpu.func

// FIVE-WARPS: error: block-threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got 160
// FIVE-WARPS-NOT: gpu.func

// FOUR-WARPS: gpu.func @segmented_sum
// FOUR-WARPS-SAME: nvvm.reqntid = array<i32: 100, 1, 1>

// TWO-WARPS: gpu.func @segmented_sum
// TWO-WARPS-SAME: nvvm.reqntid = array<i32: 40, 1, 1>
