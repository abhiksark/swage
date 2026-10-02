// test/Conversion/SwageToPlan/split-merge.mlir
// The split-merge schedule plans the second stage of a split reduction. The
// plan function is named after its kernel, <function>__merge, and holds one
// merge task operation. Its region is an identity reduction of the kind of
// the program over scratch: the element program of the function, a map and
// a transformed reduction here, is not in the merge plan, so the merge never
// runs it. The first RUN line checks that no arithmetic is left.
//
// A list of the two split schedules plans both stages of one function.
//
// RUN: swage-opt --swage-to-plan='schedule=split-merge' %s \
// RUN:   | FileCheck %s --check-prefix=MERGE --implicit-check-not=arith. \
// RUN:     --implicit-check-not=@segmented_max(
// RUN: swage-opt --swage-to-plan='schedule=split-partial,split-merge' %s \
// RUN:   | FileCheck %s --check-prefixes=PARTIAL,MERGE

// PARTIAL: func.func @segmented_max__partial({{.*}}) attributes {swage_plan.block_threads = 512 : i32} {
// PARTIAL-NEXT: swage_plan.partial_tasks
// PARTIAL: arith.mulf
// PARTIAL: arith.addf
// PARTIAL: return

// MERGE: func.func @segmented_max__merge(%[[SCRATCH:.*]]: memref<?xf32>, %[[OUTPUT:.*]]: memref<?xf32> {swage.role = #swage.role<output>}, %[[MERGES:.*]]: memref<?xi32>, %[[PARTIAL_COUNT:.*]]: i32, %[[MERGE_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>}) attributes {swage_plan.block_threads = 512 : i32} {
// MERGE-NEXT: swage_plan.merge_tasks scratch(%[[SCRATCH]] : memref<?xf32>) partial_count(%[[PARTIAL_COUNT]] : i32) merges(%[[MERGES]] : memref<?xi32>) merge_count(%[[MERGE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) into(%[[OUTPUT]] : memref<?xf32>) {
// MERGE-NEXT: ^bb0(%[[PARTIALS:.*]]: !swage.segment<f32>):
// MERGE-NEXT: %[[MAX:.*]] = swage.reduce %[[PARTIALS]] kind<max> : !swage.segment<f32> -> f32 {
// MERGE-NEXT: ^bb0(%[[PARTIAL:.*]]: f32):
// MERGE-NEXT: swage.yield %[[PARTIAL]] : f32
// MERGE-NEXT: }
// MERGE-NEXT: swage_plan.yield %[[MAX]] : f32
// MERGE-NEXT: }
// MERGE-NEXT: return

module {
  func.func @segmented_max(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %squares = swage.map %segment : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      %square = arith.mulf %element, %element : f32
      swage.yield %square : f32
    }
    %max = swage.reduce %squares kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      %doubled = arith.addf %element, %element : f32
      swage.yield %doubled : f32
    }
    memref.store %max, %output[%sid] : memref<?xf32>
    return
  }
}
