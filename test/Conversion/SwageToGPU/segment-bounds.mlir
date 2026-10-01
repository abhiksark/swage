// test/Conversion/SwageToGPU/segment-bounds.mlir
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=32' %s \
// RUN:   | FileCheck %s --check-prefixes=DIRECT,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s --check-prefixes=DIRECT,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=32 use-task-ids=true' %s \
// RUN:   | FileCheck %s --check-prefixes=TASKS,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128 use-task-ids=true' %s \
// RUN:   | FileCheck %s --check-prefixes=TASKS,CHECK
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed=true' %s \
// RUN:   | FileCheck %s --check-prefixes=FUSED,CHECK

// The kernel re-reads offsets[sid] and offsets[sid + 1] from device memory at
// every launch, so host validation cannot bound them. Both loaded offsets must
// pass through a signed clamp against the value count the ABI carries before
// they become loop bounds: start into [0, value_count], end into
// [start, value_count]. The value count sits at a different argument position
// in each ABI, so every ABI is pinned here.
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

// DIRECT: gpu.func @segmented_sum(%{{[^,]+}}: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %{{[^,)]+}}: i32) kernel
// TASKS: gpu.func @segmented_sum(%{{[^,]+}}: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %{{[^,)]+}}: i32) kernel
// FUSED: gpu.func @segmented_sum(%{{[^,]+}}: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[VALUE_COUNT:[^,]+]]: i32, %{{[^,)]+}}: i32, %{{[^,)]+}}: i32) kernel

// CHECK: %[[START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// CHECK: %[[START_RAW:.*]] = llvm.load %[[START_ADDRESS]] : !llvm.ptr -> i32
// CHECK: %[[END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// CHECK: %[[END_RAW:.*]] = llvm.load %[[END_ADDRESS]] : !llvm.ptr -> i32
// CHECK: %[[FLOOR:.*]] = arith.constant 0 : i32
// CHECK: %[[START_FLOORED:.*]] = arith.maxsi %[[START_RAW]], %[[FLOOR]] : i32
// CHECK: %[[START_I32:.*]] = arith.minsi %[[START_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK: %[[END_FLOORED:.*]] = arith.maxsi %[[END_RAW]], %[[START_I32]] : i32
// CHECK: %[[END_I32:.*]] = arith.minsi %[[END_FLOORED]], %[[VALUE_COUNT]] : i32
// CHECK: %[[START:.*]] = arith.index_cast %[[START_I32]] : i32 to index
// CHECK: %[[END:.*]] = arith.index_cast %[[END_I32]] : i32 to index
// CHECK: %[[FIRST:.*]] = arith.addi %[[START]], %{{.*}} : index
// CHECK: scf.for %{{.*}} = %[[FIRST]] to %[[END]] step
// Past the clamp the raw offsets have no further use.
// CHECK-NOT: %[[START_RAW]]{{[^0-9]}}
// CHECK-NOT: %[[END_RAW]]{{[^0-9]}}

// The fused kernel emits the segment body twice: the warp branch above and
// the CTA branch here. Both clamp against the same value count.
// FUSED: } else {
// FUSED: %[[CTA_START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// FUSED: %[[CTA_START_RAW:.*]] = llvm.load %[[CTA_START_ADDRESS]] : !llvm.ptr -> i32
// FUSED: %[[CTA_END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// FUSED: %[[CTA_END_RAW:.*]] = llvm.load %[[CTA_END_ADDRESS]] : !llvm.ptr -> i32
// FUSED: %[[CTA_FLOOR:.*]] = arith.constant 0 : i32
// FUSED: %[[CTA_START_FLOORED:.*]] = arith.maxsi %[[CTA_START_RAW]], %[[CTA_FLOOR]] : i32
// FUSED: %[[CTA_START_I32:.*]] = arith.minsi %[[CTA_START_FLOORED]], %[[VALUE_COUNT]] : i32
// FUSED: %[[CTA_END_FLOORED:.*]] = arith.maxsi %[[CTA_END_RAW]], %[[CTA_START_I32]] : i32
// FUSED: %[[CTA_END_I32:.*]] = arith.minsi %[[CTA_END_FLOORED]], %[[VALUE_COUNT]] : i32
// FUSED: %[[CTA_START:.*]] = arith.index_cast %[[CTA_START_I32]] : i32 to index
// FUSED: %[[CTA_END:.*]] = arith.index_cast %[[CTA_END_I32]] : i32 to index
// FUSED: %[[CTA_FIRST:.*]] = arith.addi %[[CTA_START]], %{{.*}} : index
// FUSED: scf.for %{{.*}} = %[[CTA_FIRST]] to %[[CTA_END]] step
// FUSED-NOT: %[[CTA_START_RAW]]{{[^0-9]}}
// FUSED-NOT: %[[CTA_END_RAW]]{{[^0-9]}}
