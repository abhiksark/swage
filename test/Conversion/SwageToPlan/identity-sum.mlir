// test/Conversion/SwageToPlan/identity-sum.mlir
// The planner replaces a segment function by the plan function of one
// kernel: the parameter list of the kernel, the launch width, and one task
// operation that takes every buffer and every bound as an operand.
//
// RUN: swage-opt --swage-to-plan %s | FileCheck %s --check-prefix=DIRECT
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=512' %s \
// RUN:   | FileCheck %s --check-prefix=WIDE
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' %s \
// RUN:   | FileCheck %s --check-prefix=WARP
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=128' %s \
// RUN:   | FileCheck %s --check-prefix=CTA
// RUN: not swage-opt --swage-to-plan='schedule=fused' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-SCHEDULE
// RUN: not swage-opt --swage-to-plan='block-threads=0' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-THREADS -DTHREADS=0
// RUN: not swage-opt --swage-to-plan='block-threads=96' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-THREADS -DTHREADS=96
// RUN: not swage-opt --swage-to-plan='block-threads=2048' %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=BAD-THREADS -DTHREADS=2048

// The direct schedule keeps the five arguments of the function, at 128
// threads unless block-threads says otherwise, and reduces across the block.
// The segment id, the segment construction, and the scalar store are gone:
// the task operation binds the segment and stores the yielded scalar.
// DIRECT: module {
// DIRECT-NEXT: func.func @segmented_sum(%[[VALUES:.*]]: memref<?xf32> {swage.role = #swage.role<values>}, %[[OFFSETS:.*]]: memref<?xi32> {swage.role = #swage.role<offsets>}, %[[OUTPUT:.*]]: memref<?xf32> {swage.role = #swage.role<output>}, %[[VALUE_COUNT:.*]]: i32 {swage.role = #swage.role<value_count>}, %[[SEGMENT_COUNT:.*]]: i32 {swage.role = #swage.role<segment_count>}) attributes {swage_plan.block_threads = 128 : i32} {
// DIRECT-NEXT: swage_plan.tasks policy<cta> segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) into(%[[OUTPUT]] : memref<?xf32>) {
// DIRECT-NEXT: ^bb0(%[[SEGMENT:.*]]: !swage.segment<f32>):
// DIRECT-NEXT: %[[SUM:.*]] = swage.reduce %[[SEGMENT]] kind<sum> : !swage.segment<f32> -> f32 {
// DIRECT-NEXT: ^bb0(%[[ELEMENT:.*]]: f32):
// DIRECT-NEXT: swage.yield %[[ELEMENT]] : f32
// DIRECT-NEXT: }
// DIRECT-NEXT: swage_plan.yield %[[SUM]] : f32
// DIRECT-NEXT: }
// DIRECT-NEXT: return
// DIRECT-NEXT: }
// DIRECT-NEXT: }

// WIDE: attributes {swage_plan.block_threads = 512 : i32} {
// WIDE-NEXT: swage_plan.tasks policy<cta> segments(

// The task-id schedule adds the task buffer after the three buffers and the
// task count between the two counts, which is the parameter order of the
// kernel. A block that is one subgroup reduces within the subgroup.
// WARP: func.func @segmented_sum(%[[VALUES:.*]]: memref<?xf32> {{[{].*[}]}}, %[[OFFSETS:.*]]: memref<?xi32> {{[{].*[}]}}, %[[OUTPUT:.*]]: memref<?xf32> {{[{].*[}]}}, %[[IDS:.*]]: memref<?xi32>, %[[VALUE_COUNT:.*]]: i32 {{[{].*[}]}}, %[[TASK_COUNT:.*]]: i32, %[[SEGMENT_COUNT:.*]]: i32 {{[{].*[}]}}) attributes {swage_plan.block_threads = 32 : i32} {
// WARP-NEXT: swage_plan.tasks policy<warp> segments(%[[VALUES]], %[[OFFSETS]] : memref<?xf32>, memref<?xi32>) value_count(%[[VALUE_COUNT]] : i32) segment_count(%[[SEGMENT_COUNT]] : i32) ids(%[[IDS]] : memref<?xi32>) task_count(%[[TASK_COUNT]] : i32) into(%[[OUTPUT]] : memref<?xf32>) {

// CTA: attributes {swage_plan.block_threads = 128 : i32} {
// CTA-NEXT: swage_plan.tasks policy<cta> segments({{.*}}) ids(%{{.*}} : memref<?xi32>) task_count(%{{.*}} : i32) into(

// BAD-SCHEDULE: error: schedule must be direct or task-ids, got 'fused'
// BAD-THREADS: error: block-threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got [[THREADS]]

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
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
