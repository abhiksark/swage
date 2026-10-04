// test/Conversion/SwageToPlan/split-partial.mlir
// The split-partial schedule plans the first stage of a split reduction.
// The plan function is named after its kernel, <function>__partial, takes
// the parameters of that kernel in layout order at the split width of the
// target, and holds one partial task operation. The element program, with
// its maps fused, stays in the reduction of the chunk.
//
// A schedule list plans one function per kernel, in the order of the list.
//
// RUN: swage-opt --swage-to-plan='schedule=split-partial' %s \
// RUN:   | FileCheck %s --check-prefix=PARTIAL --implicit-check-not=@segmented_sum(
// RUN: swage-opt --swage-to-plan='schedule=split-partial block-threads=32' \
// RUN:   %s | FileCheck %s --check-prefix=PARTIAL
// RUN: swage-opt \
// RUN:   --swage-to-plan='schedule=task-ids,split-partial block-threads=32' \
// RUN:   %s | FileCheck %s --check-prefixes=TASKS,PARTIAL
// RUN: not swage-opt --swage-to-plan='schedule=direct,task-ids' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=SAME-KERNEL
// RUN: not swage-opt --swage-to-plan='schedule=sequential,split-partial' \
// RUN:   %s 2>&1 | FileCheck %s --check-prefix=SEQUENTIAL

// TASKS: func.func @segmented_sum({{.*}}) attributes {swage_plan.block_threads = 32 : i32} {
// TASKS-NEXT: swage_plan.tasks policy<warp>
// TASKS: return

// The partial width is the target's, whatever block-threads says.
// PARTIAL: func.func @segmented_sum__partial(%[[VALUES:.*]]: memref<?xf32> {swage.role = #swage.role<values>, swage_plan.source_index = 0 : i32}, %[[RANGES:.*]]: memref<?xi32>, %[[SCRATCH:.*]]: memref<?xf32>, %[[VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>, swage_plan.source_index = 3 : i32}, %[[PARTIAL_COUNT:.*]]: i32) attributes {swage_plan.block_threads = 512 : i32} {
// PARTIAL-NEXT: swage_plan.partial_tasks values(%[[VALUES]] : memref<?xf32>) value_count(%[[VALUE_COUNT]] : i32) ranges(%[[RANGES]] : memref<?xi32>) partial_count(%[[PARTIAL_COUNT]] : i32) into(%[[SCRATCH]] : memref<?xf32>) {
// PARTIAL-NEXT: ^bb0(%[[CHUNK:.*]]: !swage.segment<f32>):
// PARTIAL-NEXT: %[[SUM:.*]] = swage.reduce %[[CHUNK]] kind<sum> : !swage.segment<f32> -> f32 {
// PARTIAL-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// PARTIAL-NEXT: %[[TWICE:.*]] = arith.addf %[[ELEMENT]], %[[ELEMENT]] : f32
// PARTIAL-NEXT: swage.yield %[[TWICE]] : f32
// PARTIAL-NEXT: }
// PARTIAL-NEXT: swage_plan.yield %[[SUM]] : f32
// PARTIAL-NEXT: }
// PARTIAL-NEXT: return

// SAME-KERNEL: error: schedules direct and task-ids both name their kernel @<function>; a schedule list names each kernel once
// SEQUENTIAL: error: the sequential schedule plans a function in place and keeps it, so it cannot share a schedule list with a kernel

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
    %doubled = swage.map %segment : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      %twice = arith.addf %element, %element : f32
      swage.yield %twice : f32
    }
    %sum = swage.reduce %doubled kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
