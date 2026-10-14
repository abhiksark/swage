// test/Conversion/SwageToGPU/segmented-mean.mlir
// The kernels of a mean. Every task kernel divides the combined sum by the
// extent of its segment, the clamped end minus the clamped start. The start
// of one thread is the start of the segment plus its thread index, so a
// subtraction from that value would be the extent on thread zero only.
//
// A split sums the chunks and divides once. The partial kernel holds no
// division. The merge kernel reads the extent of its segment from the range
// records of the partial tasks: the end of the last record minus the begin
// of the first, addressed through the clamped range of partials, and zero
// for an empty range.
//
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CTA
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WARP \
// RUN:       --implicit-check-not=gpu.all_reduce
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=FUSED
// RUN: swage-opt --swage-to-plan='schedule=split-partial' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=PARTIAL \
// RUN:       --implicit-check-not=arith.divf --implicit-check-not=arith.sitofp
// RUN: swage-opt --swage-to-plan='schedule=split-merge' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=MERGE

module {
  func.func @segmented_mean(
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
    %extent = swage.extent %segment : !swage.segment<f32>
    %count = arith.index_cast %extent : index to i32
    %divisor = arith.sitofp %count : i32 to f32
    %mean = arith.divf %sum, %divisor : f32
    memref.store %mean, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK: gpu.func @segmented_mean(
// CHECK: %[[THREAD:.*]] = gpu.thread_id x
// CHECK: %[[START_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// CHECK: %[[END_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// CHECK-NEXT: %[[START:.*]] = arith.index_cast %[[START_CLAMPED]] : i32 to index
// CHECK-NEXT: %[[END:.*]] = arith.index_cast %[[END_CLAMPED]] : i32 to index
// CHECK-NEXT: %[[FIRST:.*]] = arith.addi %[[START]], %[[THREAD]] : index
// CHECK-NEXT: %[[EXTENT:.*]] = arith.subi %[[END]], %[[START]] : index
// CHECK: %[[LOCAL:.*]] = scf.for %{{.*}} = %[[FIRST]] to %[[END]]
// CTA: %[[TOTAL:.*]] = gpu.all_reduce add %[[LOCAL]] uniform
// WARP: gpu.shuffle xor %[[LOCAL]]
// WARP: gpu.shuffle xor
// WARP: gpu.shuffle xor
// WARP: gpu.shuffle xor
// WARP: gpu.shuffle xor
// WARP-NEXT: %[[TOTAL:.*]] = arith.addf
// CHECK: %[[COUNT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
// CHECK-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f32
// CHECK-NEXT: %[[MEAN:.*]] = arith.divf %[[TOTAL]], %[[DIVISOR]] : f32
// CHECK: llvm.store %[[MEAN]], %{{.*}} : f32, !llvm.ptr

// Both branches of the fused kernel divide by the extent of their segment.
// FUSED: gpu.func @segmented_mean(
// FUSED: %[[WARP_START_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// FUSED: %[[WARP_END_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// FUSED-NEXT: %[[WARP_START:.*]] = arith.index_cast %[[WARP_START_CLAMPED]] : i32 to index
// FUSED-NEXT: %[[WARP_END:.*]] = arith.index_cast %[[WARP_END_CLAMPED]] : i32 to index
// FUSED-NEXT: %{{.*}} = arith.addi %[[WARP_START]], %{{.*}} : index
// FUSED-NEXT: %[[WARP_EXTENT:.*]] = arith.subi %[[WARP_END]], %[[WARP_START]] : index
// FUSED: gpu.shuffle xor
// FUSED: arith.index_cast %[[WARP_EXTENT]] : index to i32
// FUSED: %[[WARP_MEAN:.*]] = arith.divf
// FUSED: llvm.store %[[WARP_MEAN]]
// FUSED: } else {
// FUSED: %[[CTA_START_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// FUSED: %[[CTA_END_CLAMPED:.*]] = arith.minsi %{{.*}}, %{{.*}} : i32
// FUSED-NEXT: %[[CTA_START:.*]] = arith.index_cast %[[CTA_START_CLAMPED]] : i32 to index
// FUSED-NEXT: %[[CTA_END:.*]] = arith.index_cast %[[CTA_END_CLAMPED]] : i32 to index
// FUSED-NEXT: %{{.*}} = arith.addi %[[CTA_START]], %{{.*}} : index
// FUSED-NEXT: %[[CTA_EXTENT:.*]] = arith.subi %[[CTA_END]], %[[CTA_START]] : index
// FUSED: gpu.all_reduce add
// FUSED: arith.index_cast %[[CTA_EXTENT]] : index to i32
// FUSED: %[[CTA_MEAN:.*]] = arith.divf
// FUSED: llvm.store %[[CTA_MEAN]]

// PARTIAL: gpu.func @segmented_mean__partial(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32)
// PARTIAL: %[[PARTIAL_SUM:.*]] = gpu.all_reduce add
// PARTIAL: llvm.store %[[PARTIAL_SUM]], %{{.*}} : f32, !llvm.ptr

// MERGE: gpu.func @segmented_mean__merge(%[[SCRATCH:.*]]: !llvm.ptr, %[[OUTPUT:.*]]: !llvm.ptr, %[[MERGES:.*]]: !llvm.ptr, %[[RANGES:.*]]: !llvm.ptr, %[[PARTIAL_COUNT:.*]]: i32, %{{.*}}: i32, %{{.*}}: i32)
// MERGE-SAME: swage.kernel_contract = {arguments = [{access = "read", key = "scratch", kind = "ptr", origin = "scratch"}, {access = "write", kind = "ptr", origin = "user", source_index = 2 : i64}, {access = "read", key = "merge_records", kind = "ptr", origin = "plan"}, {access = "read", key = "partial_ranges", kind = "ptr", origin = "plan"}, {key = "partial_count", kind = "i32", origin = "derived"}, {key = "merge_count", kind = "i32", origin = "derived"}, {kind = "i32", origin = "user", source_index = 4 : i64}], backend = "cuda", entry = "segmented_mean__merge", launch = {block = array<i32: 512, 1, 1>, model = "spmd-grid"}, version = 2 : i64}
// The range of partials is clamped to the partial count before it is used.
// MERGE: %[[BEGIN_CLAMPED:.*]] = arith.minsi %{{.*}}, %[[PARTIAL_COUNT]] : i32
// MERGE: %[[END_CLAMPED:.*]] = arith.minsi %{{.*}}, %[[PARTIAL_COUNT]] : i32
// MERGE: %[[BEGIN:.*]] = arith.index_cast %[[BEGIN_CLAMPED]] : i32 to index
// MERGE-NEXT: %[[END:.*]] = arith.index_cast %[[END_CLAMPED]] : i32 to index
// MERGE-NEXT: %[[HAS_PARTIALS:.*]] = arith.cmpi slt, %[[BEGIN]], %[[END]] : index
// MERGE-NEXT: %[[EXTENT:.*]] = scf.if %[[HAS_PARTIALS]] -> (index) {
// MERGE:   %[[FIRST_BASE:.*]] = arith.muli %[[BEGIN]], %[[TWO:.*]] : index
// MERGE:   %[[LAST:.*]] = arith.subi %[[END]], %{{.*}} : index
// MERGE:   %[[LAST_BASE:.*]] = arith.muli %[[LAST]], %[[TWO]] : index
// MERGE:   %[[FIRST_INDEX:.*]] = arith.index_cast %[[FIRST_BASE]] : index to i64
// MERGE:   %[[FIRST_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]][%[[FIRST_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// MERGE:   %[[BEGIN_VALUE:.*]] = llvm.load %[[FIRST_ADDRESS]] : !llvm.ptr -> i32
// MERGE:   %[[LAST_END:.*]] = arith.addi %[[LAST_BASE]], %{{.*}} : index
// MERGE:   %[[LAST_INDEX:.*]] = arith.index_cast %[[LAST_END]] : index to i64
// MERGE:   %[[LAST_ADDRESS:.*]] = llvm.getelementptr %[[RANGES]][%[[LAST_INDEX]]] : (!llvm.ptr, i64) -> !llvm.ptr, i32
// MERGE:   %[[END_VALUE:.*]] = llvm.load %[[LAST_ADDRESS]] : !llvm.ptr -> i32
// MERGE:   %[[EXTENT_WORD:.*]] = arith.subi %[[END_VALUE]], %[[BEGIN_VALUE]] : i32
// MERGE:   %[[EXTENT_INDEX:.*]] = arith.index_cast %[[EXTENT_WORD]] : i32 to index
// MERGE:   scf.yield %[[EXTENT_INDEX]] : index
// MERGE: } else {
// MERGE:   scf.yield %{{.*}} : index
// MERGE: }
// MERGE: %[[MERGED:.*]] = gpu.all_reduce add
// MERGE: %[[COUNT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
// MERGE-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f32
// MERGE-NEXT: %[[MEAN:.*]] = arith.divf %[[MERGED]], %[[DIVISOR]] : f32
// MERGE: %[[SLOT:.*]] = llvm.getelementptr %[[OUTPUT]][%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f32
// MERGE-NEXT: llvm.store %[[MEAN]], %[[SLOT]] : f32, !llvm.ptr
