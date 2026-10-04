// test/Conversion/SwageToPlan/segmented-mean.mlir
// A mean is a sum, the extent of the segment, and one division. The planner
// absorbs swage.extent into the second argument of the task region and moves
// the division behind the reduction, so it runs once per task.
//
// A split sums the chunks and divides once: the partial region yields the
// raw sum and holds no division, and the merge region holds the division
// and takes the extent of the split segment, which the merge operation
// reads from the range records of the partial tasks. A mean of partial
// means would be wrong, and so would a division by the number of partials.
//
// RUN: swage-opt --swage-to-plan='schedule=task-ids' %s \
// RUN:   | FileCheck %s --check-prefix=TASKS --implicit-check-not=swage.extent
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' %s \
// RUN:   | FileCheck %s --check-prefix=FUSED --implicit-check-not=swage.extent
// RUN: swage-opt --swage-to-plan='schedule=split-partial' %s \
// RUN:   | FileCheck %s --check-prefix=PARTIAL \
// RUN:       --implicit-check-not=swage.extent --implicit-check-not=arith. \
// RUN:       --implicit-check-not='{{[^_]index}}'
// RUN: swage-opt --swage-to-plan='schedule=split-merge' %s \
// RUN:   | FileCheck %s --check-prefix=MERGE --implicit-check-not=swage.extent
// RUN: swage-opt --swage-to-plan='schedule=sequential' %s \
// RUN:   | FileCheck %s --check-prefix=SEQUENTIAL \
// RUN:       --implicit-check-not=swage.extent
// RUN: not swage-opt --swage-to-plan='schedule=persistent' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=PERSISTENT

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

// TASKS: swage_plan.tasks policy<cta>
// TASKS: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>, %[[EXTENT:.*]]: index):
// TASKS-NEXT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
// TASKS: %[[COUNT:.*]] = arith.index_cast %[[EXTENT]] : index to i32
// TASKS-NEXT: %[[DIVISOR:.*]] = arith.sitofp %[[COUNT]] : i32 to f32
// TASKS-NEXT: %[[MEAN:.*]] = arith.divf %[[SUM]], %[[DIVISOR]] : f32
// TASKS-NEXT: swage_plan.yield %[[MEAN]] : f32

// Both regions of the fused kernel run the program, with its division.
// FUSED: swage_plan.fused_tasks
// FUSED: warp {
// FUSED-NEXT: ^bb0(%[[WARP:.*]]: !swage.segment<f32>, %[[WARP_EXTENT:.*]]: index):
// FUSED: arith.index_cast %[[WARP_EXTENT]] : index to i32
// FUSED: %[[WARP_MEAN:.*]] = arith.divf
// FUSED-NEXT: swage_plan.yield %[[WARP_MEAN]] : f32
// FUSED: } cta {
// FUSED-NEXT: ^bb0(%[[CTA:.*]]: !swage.segment<f32>, %[[CTA_EXTENT:.*]]: index):
// FUSED: arith.index_cast %[[CTA_EXTENT]] : index to i32
// FUSED: %[[CTA_MEAN:.*]] = arith.divf
// FUSED-NEXT: swage_plan.yield %[[CTA_MEAN]] : f32

// The partial region takes the chunk alone and yields its raw sum. The run
// refuses any arith operation and any index, so no division and no extent
// reach a partial task.
// PARTIAL: func.func @segmented_mean__partial(
// PARTIAL: swage_plan.partial_tasks
// PARTIAL: ^bb0(%[[CHUNK:.*]]: !swage.segment<f32>):
// PARTIAL-NEXT: %[[PARTIAL_SUM:.*]] = swage.reduce %[[CHUNK]] kind<sum> : !swage.segment<f32> -> f32 {
// PARTIAL: swage_plan.yield %[[PARTIAL_SUM]] : f32

// The merge kernel takes the range records as a fourth buffer, after the
// merge records, and the merge region divides the sum of the partial sums.
// MERGE: func.func @segmented_mean__merge(%[[SCRATCH:.*]]: memref<?xf32>, %[[OUTPUT:.*]]: memref<?xf32> {swage.role = #swage.role<output>, swage_plan.source_index = 2 : i32}, %[[MERGES:.*]]: memref<?xi32>, %[[RANGES:.*]]: memref<?xi32>, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>, swage_plan.source_index = 4 : i32})
// MERGE: swage_plan.merge_tasks scratch(%[[SCRATCH]] : memref<?xf32>)
// MERGE-SAME: merges(%[[MERGES]] : memref<?xi32>)
// MERGE-SAME: ranges(%[[RANGES]] : memref<?xi32>)
// MERGE-SAME: into(%[[OUTPUT]] : memref<?xf32>)
// MERGE: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f32>, %[[SPLIT_EXTENT:.*]]: index):
// MERGE-NEXT: %[[MERGED:.*]] = swage.reduce %[[PARTIALS]] kind<sum> : !swage.segment<f32> -> f32 {
// MERGE-NEXT: ^bb0(%[[ONE:.*]]: f32):
// MERGE-NEXT: swage.yield %[[ONE]] : f32
// MERGE: %[[MERGE_COUNT_WORD:.*]] = arith.index_cast %[[SPLIT_EXTENT]] : index to i32
// MERGE-NEXT: %[[MERGE_DIVISOR:.*]] = arith.sitofp %[[MERGE_COUNT_WORD]] : i32 to f32
// MERGE-NEXT: %[[MERGE_MEAN:.*]] = arith.divf %[[MERGED]], %[[MERGE_DIVISOR]] : f32
// MERGE-NEXT: swage_plan.yield %[[MERGE_MEAN]] : f32

// SEQUENTIAL: swage_plan.tasks policy<sequential>
// SEQUENTIAL: ^bb0(%{{.*}}: !swage.segment<f32>, %[[ORACLE_EXTENT:.*]]: index):
// SEQUENTIAL: arith.index_cast %[[ORACLE_EXTENT]] : index to i32
// SEQUENTIAL: arith.divf

// PERSISTENT: error: persistent execution stores the reduction result as it is and takes no scalar epilogue
