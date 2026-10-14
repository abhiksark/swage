// test/Conversion/SwagePlanToGPU/invalid-persistent-tasks.mlir
// What the conversion requires of a persistent plan function beyond what
// the dialect verifies. The planner produces only plan functions that pass;
// these are written by hand. Every rule is checked before anything is
// changed: the second RUN line prints each module after its failure and
// finds no kernel module that the conversion would have created.
//
// RUN: swage-opt --swage-plan-to-gpu --verify-diagnostics \
// RUN:   --split-input-file %s
// RUN: swage-opt --swage-plan-to-gpu --verify-diagnostics \
// RUN:   --split-input-file --mlir-print-ir-after-failure %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNCHANGED \
// RUN:     --implicit-check-not=gpu.func --implicit-check-not=nvvm.reqntid

// UNCHANGED-LABEL: func.func @narrow_block(
// UNCHANGED: swage_plan.persistent_tasks
// UNCHANGED-LABEL: func.func @maximum(
// UNCHANGED: swage_plan.persistent_tasks
// UNCHANGED-LABEL: func.func @squares_in_a_partial(
// UNCHANGED: swage_plan.persistent_tasks

// The claim batches and the resident blocks of a launch are chosen for one
// launch width.
module {
  func.func @narrow_block(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    // expected-error@+1 {{a persistent task block has the persistent launch width of the target, 512 threads, got swage_plan.block_threads = 128}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}

// -----

// The queue kernel is lowered for an identity sum, and every region runs
// that program.
module {
  func.func @maximum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    // expected-error@+1 {{every region of a persistent task operation holds one capture-free kind<sum> reduction with an identity region and yields its result; the queue kernel is lowered for that program only}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}

// -----

module {
  func.func @squares_in_a_partial(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %warp_ids: memref<?xi32>,
      %cta_ids: memref<?xi32>, %ranges: memref<?xi32>,
      %merge_ids: memref<?xi32>, %merges: memref<?xi32>,
      %scratch: memref<?xf32>, %counters: memref<?xi32>,
      %value_count: i32, %warp_task_count: i32,
      %cta_task_count: i32, %partial_count: i32,
      %merge_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 512 : i32} {
    // expected-error@+1 {{every region of a persistent task operation holds one capture-free kind<sum> reduction with an identity region and yields its result; the queue kernel is lowered for that program only}}
    swage_plan.persistent_tasks
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32)
        segment_count(%segment_count : i32)
        warp_ids(%warp_ids : memref<?xi32>)
        warp_task_count(%warp_task_count : i32)
        cta_ids(%cta_ids : memref<?xi32>)
        cta_task_count(%cta_task_count : i32)
        ranges(%ranges : memref<?xi32>) merge_ids(%merge_ids : memref<?xi32>)
        partial_count(%partial_count : i32)
        merges(%merges : memref<?xi32>) merge_count(%merge_count : i32)
        scratch(%scratch : memref<?xf32>) counters(%counters : memref<?xi32>)
        into(%output : memref<?xf32>) cta {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } partial {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        %square = arith.mulf %value, %value : f32
        swage.yield %square : f32
      }
      swage_plan.yield %total : f32
    } merge {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    } warp {
    ^bb0(%segment: !swage.segment<f32>):
      %total = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}
