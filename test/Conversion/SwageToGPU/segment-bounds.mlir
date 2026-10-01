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
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=512 persistent' %s \
// RUN:   | FileCheck %s --check-prefixes=PERSISTENT,CHECK
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=PARTIAL,SPLIT
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu='merge' %s \
// RUN:   | FileCheck %s --check-prefixes=MERGE,SPLIT

// The kernel re-reads offsets[sid] and offsets[sid + 1] from device memory at
// every launch, so host validation cannot bound them. Both loaded offsets must
// pass through a signed clamp against the value count the ABI carries before
// they become loop bounds: start into [0, value_count], end into
// [start, value_count]. The value count sits at a different argument position
// in each ABI, so every ABI is pinned here.
//
// Plan-owned ranges take the same clamp against the length of the buffer they
// index. Partial ranges index values and are bounded by the value count.
// Merge ranges index scratch and are bounded by the partial count.
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
// PERSISTENT: gpu.func @segmented_sum(
// PERSISTENT-SAME: %[[VALUES:[^,]+]]: !llvm.ptr, %[[OFFSETS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr,
// PERSISTENT-SAME: %{{[^,]+}}: !llvm.ptr, %{{[^,]+}}: !llvm.ptr,
// PERSISTENT-SAME: %[[RANGES:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[MERGES:[^,]+]]: !llvm.ptr,
// PERSISTENT-SAME: %[[SCRATCH:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr,
// PERSISTENT-SAME: %[[VALUE_COUNT:[^,]+]]: i32, %{{[^,]+}}: i32, %{{[^,]+}}: i32, %[[PARTIAL_COUNT:[^,]+]]: i32, %{{[^,)]+}}: i32)

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

// The persistent kernel loads four ranges. The shared block above pinned the
// first, the direct CTA queue, and this all-reduce closes that task before
// its value names can be reused.
// PERSISTENT: gpu.all_reduce add

// Split partial queue: a plan-owned range into values.
// PERSISTENT: %[[PARTIAL_START_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]]
// PERSISTENT: %[[PARTIAL_START_RAW:.*]] = llvm.load %[[PARTIAL_START_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT: %[[PARTIAL_END_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]]
// PERSISTENT: %[[PARTIAL_END_RAW:.*]] = llvm.load %[[PARTIAL_END_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT: %[[PARTIAL_FLOOR:.*]] = arith.constant 0 : i32
// PERSISTENT: %[[PARTIAL_START_FLOORED:.*]] = arith.maxsi %[[PARTIAL_START_RAW]], %[[PARTIAL_FLOOR]] : i32
// PERSISTENT: %[[PARTIAL_START_I32:.*]] = arith.minsi %[[PARTIAL_START_FLOORED]], %[[VALUE_COUNT]] : i32
// PERSISTENT: %[[PARTIAL_END_FLOORED:.*]] = arith.maxsi %[[PARTIAL_END_RAW]], %[[PARTIAL_START_I32]] : i32
// PERSISTENT: %[[PARTIAL_END_I32:.*]] = arith.minsi %[[PARTIAL_END_FLOORED]], %[[VALUE_COUNT]] : i32
// PERSISTENT: %[[PARTIAL_START:.*]] = arith.index_cast %[[PARTIAL_START_I32]] : i32 to index
// PERSISTENT: %[[PARTIAL_END:.*]] = arith.index_cast %[[PARTIAL_END_I32]] : i32 to index
// PERSISTENT: %[[PARTIAL_FIRST:.*]] = arith.addi %[[PARTIAL_START]], %{{.*}} : index
// PERSISTENT: scf.for %[[PARTIAL_I:.*]] = %[[PARTIAL_FIRST]] to %[[PARTIAL_END]] step
// PERSISTENT-NEXT: %[[PARTIAL_I64:.*]] = arith.index_cast %[[PARTIAL_I]] : index to i64
// PERSISTENT-NEXT: llvm.getelementptr %[[VALUES]][%[[PARTIAL_I64]]]

// Merge: a plan-owned range into scratch, bounded by the partial count. The
// completion atomic separates the publisher, which only subtracts the two
// range fields, from the merge that indexes scratch with them. The first
// record field the merge loads is the output segment, not part of the range.
// PERSISTENT: llvm.atomicrmw add %{{.*}} acq_rel
// PERSISTENT: llvm.getelementptr %[[MERGES]]
// PERSISTENT: %[[MERGE_START_ADDRESS:.*]] = llvm.getelementptr %[[MERGES]]
// PERSISTENT: %[[MERGE_START_RAW:.*]] = llvm.load %[[MERGE_START_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT: %[[MERGE_END_ADDRESS:.*]] = llvm.getelementptr %[[MERGES]]
// PERSISTENT: %[[MERGE_END_RAW:.*]] = llvm.load %[[MERGE_END_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT: %[[MERGE_FLOOR:.*]] = arith.constant 0 : i32
// PERSISTENT: %[[MERGE_START_FLOORED:.*]] = arith.maxsi %[[MERGE_START_RAW]], %[[MERGE_FLOOR]] : i32
// PERSISTENT: %[[MERGE_START_I32:.*]] = arith.minsi %[[MERGE_START_FLOORED]], %[[PARTIAL_COUNT]] : i32
// PERSISTENT: %[[MERGE_END_FLOORED:.*]] = arith.maxsi %[[MERGE_END_RAW]], %[[MERGE_START_I32]] : i32
// PERSISTENT: %[[MERGE_END_I32:.*]] = arith.minsi %[[MERGE_END_FLOORED]], %[[PARTIAL_COUNT]] : i32
// PERSISTENT: %[[MERGE_START:.*]] = arith.index_cast %[[MERGE_START_I32]] : i32 to index
// PERSISTENT: %[[MERGE_END:.*]] = arith.index_cast %[[MERGE_END_I32]] : i32 to index
// PERSISTENT: %[[MERGE_FIRST:.*]] = arith.addi %[[MERGE_START]], %{{.*}} : index
// PERSISTENT: scf.for %[[MERGE_I:.*]] = %[[MERGE_FIRST]] to %[[MERGE_END]] step
// PERSISTENT-NEXT: %[[MERGE_I64:.*]] = arith.index_cast %[[MERGE_I]] : index to i64
// PERSISTENT-NEXT: llvm.getelementptr %[[SCRATCH]][%[[MERGE_I64]]]

// Direct warp queue: offsets again, behind the warp task indirection.
// PERSISTENT: %[[WARP_START_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// PERSISTENT: %[[WARP_START_RAW:.*]] = llvm.load %[[WARP_START_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT: %[[WARP_END_ADDRESS:.*]] = llvm.getelementptr %[[OFFSETS]]
// PERSISTENT: %[[WARP_END_RAW:.*]] = llvm.load %[[WARP_END_ADDRESS]] : !llvm.ptr -> i32
// PERSISTENT: %[[WARP_FLOOR:.*]] = arith.constant 0 : i32
// PERSISTENT: %[[WARP_START_FLOORED:.*]] = arith.maxsi %[[WARP_START_RAW]], %[[WARP_FLOOR]] : i32
// PERSISTENT: %[[WARP_START_I32:.*]] = arith.minsi %[[WARP_START_FLOORED]], %[[VALUE_COUNT]] : i32
// PERSISTENT: %[[WARP_END_FLOORED:.*]] = arith.maxsi %[[WARP_END_RAW]], %[[WARP_START_I32]] : i32
// PERSISTENT: %[[WARP_END_I32:.*]] = arith.minsi %[[WARP_END_FLOORED]], %[[VALUE_COUNT]] : i32
// PERSISTENT: %[[WARP_START:.*]] = arith.index_cast %[[WARP_START_I32]] : i32 to index
// PERSISTENT: %[[WARP_END:.*]] = arith.index_cast %[[WARP_END_I32]] : i32 to index
// PERSISTENT: %[[WARP_FIRST:.*]] = arith.addi %[[WARP_START]], %{{.*}} : index
// PERSISTENT: scf.for %[[WARP_I:.*]] = %[[WARP_FIRST]] to %[[WARP_END]] step
// PERSISTENT-NEXT: %[[WARP_I64:.*]] = arith.index_cast %[[WARP_I]] : index to i64
// PERSISTENT-NEXT: llvm.getelementptr %[[VALUES]][%[[WARP_I64]]]

// The split kernels pass the buffer their range indexes as the first pointer
// and its length as the first i32: values and the value count for a partial,
// scratch and the partial count for a merge. A merge record starts with the
// output segment, so its range is the second and third field.
// PARTIAL: gpu.func @segmented_sum__partial(%[[BUFFER:[^,]+]]: !llvm.ptr, %[[RECORDS:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[LENGTH:[^,]+]]: i32, %{{[^,)]+}}: i32) kernel
// MERGE: gpu.func @segmented_sum__merge(%[[BUFFER:[^,]+]]: !llvm.ptr, %{{[^,]+}}: !llvm.ptr, %[[RECORDS:[^,]+]]: !llvm.ptr, %[[LENGTH:[^,]+]]: i32, %{{[^,)]+}}: i32) kernel
// MERGE: llvm.getelementptr %[[RECORDS]]
// SPLIT: %[[SPLIT_START_ADDRESS:.*]] = llvm.getelementptr %[[RECORDS]]
// SPLIT: %[[SPLIT_START_RAW:.*]] = llvm.load %[[SPLIT_START_ADDRESS]] : !llvm.ptr -> i32
// SPLIT: %[[SPLIT_END_ADDRESS:.*]] = llvm.getelementptr %[[RECORDS]]
// SPLIT: %[[SPLIT_END_RAW:.*]] = llvm.load %[[SPLIT_END_ADDRESS]] : !llvm.ptr -> i32
// SPLIT: %[[SPLIT_FLOOR:.*]] = arith.constant 0 : i32
// SPLIT: %[[SPLIT_START_FLOORED:.*]] = arith.maxsi %[[SPLIT_START_RAW]], %[[SPLIT_FLOOR]] : i32
// SPLIT: %[[SPLIT_START_I32:.*]] = arith.minsi %[[SPLIT_START_FLOORED]], %[[LENGTH]] : i32
// SPLIT: %[[SPLIT_END_FLOORED:.*]] = arith.maxsi %[[SPLIT_END_RAW]], %[[SPLIT_START_I32]] : i32
// SPLIT: %[[SPLIT_END_I32:.*]] = arith.minsi %[[SPLIT_END_FLOORED]], %[[LENGTH]] : i32
// SPLIT: %[[SPLIT_START:.*]] = arith.index_cast %[[SPLIT_START_I32]] : i32 to index
// SPLIT: %[[SPLIT_END:.*]] = arith.index_cast %[[SPLIT_END_I32]] : i32 to index
// SPLIT: %[[SPLIT_FIRST:.*]] = arith.addi %[[SPLIT_START]], %{{.*}} : index
// SPLIT: scf.for %[[SPLIT_I:.*]] = %[[SPLIT_FIRST]] to %[[SPLIT_END]] step
// SPLIT-NEXT: %[[SPLIT_I64:.*]] = arith.index_cast %[[SPLIT_I]] : index to i64
// SPLIT-NEXT: llvm.getelementptr %[[BUFFER]][%[[SPLIT_I64]]]
// SPLIT-NOT: %[[SPLIT_START_RAW]]{{[^0-9]}}
// SPLIT-NOT: %[[SPLIT_END_RAW]]{{[^0-9]}}
