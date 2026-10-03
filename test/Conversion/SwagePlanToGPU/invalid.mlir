// test/Conversion/SwagePlanToGPU/invalid.mlir
// What the conversion requires of a plan function beyond what the dialect
// verifies. The planner produces only plan functions that pass; these are
// written by hand. Every rule is checked before anything is changed: the
// last RUN line prints each module after its failure and finds no kernel
// module that the conversion would have created.
//
// RUN: swage-opt --swage-plan-to-gpu --verify-diagnostics \
// RUN:   --split-input-file %s
// RUN: swage-opt --swage-plan-to-gpu --verify-diagnostics \
// RUN:   --split-input-file --mlir-print-ir-after-failure %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNCHANGED \
// RUN:     --implicit-check-not=gpu.func --implicit-check-not=nvvm.reqntid

// UNCHANGED-LABEL: func.func @three_subgroups(
// UNCHANGED: swage_plan.tasks
// UNCHANGED-LABEL: func.func @too_wide(
// UNCHANGED-LABEL: func.func @warp_policy_on_a_wide_block(
// UNCHANGED-LABEL: func.func @rows_on_a_partial_subgroup(
// UNCHANGED-LABEL: func.func @partial_rows_on_a_partial_subgroup(
// UNCHANGED-LABEL: func.func @merge_rows_on_a_partial_subgroup(
// UNCHANGED-LABEL: func.func @half_values(
// UNCHANGED-LABEL: func.func @exponential_double(
// UNCHANGED-LABEL: func.func @wide_offsets(
// UNCHANGED-LABEL: func.func @exponential(
// UNCHANGED-LABEL: func.func @clashes(
// UNCHANGED-LABEL: func.func @convertible(
// UNCHANGED: swage_plan.tasks
// UNCHANGED-LABEL: func.func @called(
// UNCHANGED: swage_plan.tasks
// UNCHANGED-LABEL: func.func @nested_warp(
// UNCHANGED: swage_plan.tasks
// UNCHANGED-LABEL: func.func @nested_clash(
// UNCHANGED: swage_plan.tasks
// UNCHANGED-LABEL: func.func @nested_called(
// UNCHANGED: swage_plan.tasks

// The launch width is one the target admits.
module {
  // expected-error@+1 {{swage_plan.block_threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got 96}}
  func.func @three_subgroups(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 96 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{swage_plan.block_threads must be a launch width the target admits, from 1 to 1024 threads with a power-of-two subgroup count, got 2048}}
  func.func @too_wide(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 2048 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// A warp task reduces within one subgroup, so its block is one subgroup.
module {
  func.func @warp_policy_on_a_wide_block(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    // expected-error@+1 {{policy<warp> requires swage_plan.block_threads to be the subgroup width, 32, got 128}}
    swage_plan.tasks policy<warp>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// Every lane of a row-stripe tile owns one row stripe of one column, so its
// block is a whole number of subgroups. 48 threads are two subgroups, which
// a block-wide reduction of scalars admits.
module {
  func.func @rows_on_a_partial_subgroup(
      %values: memref<?x?xf32>, %offsets: memref<?xi32>,
      %output: memref<?x?xf32>, %ids: memref<?xi32>, %value_count: i32,
      %task_count: i32, %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 48 : i32} {
    // expected-error@+1 {{a row-stripe task of rank-two values runs whole subgroups of 32 threads, so swage_plan.block_threads must be a multiple of 32, got 48}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?x?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32)
        ids(%ids : memref<?xi32>) task_count(%task_count : i32)
        into(%output : memref<?x?xf32>) {
    ^bb0(%column: !swage.segment<f32>):
      %sum = swage.reduce %column kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The partial and merge kernels of a split over rank-two values are the
// same tile, under the same rule.
module {
  func.func @partial_rows_on_a_partial_subgroup(
      %values: memref<?x?xf32>, %ranges: memref<?xi32>,
      %scratch: memref<?x?xf32>, %value_count: i32, %partial_count: i32,
      %feature_count: i32)
      attributes {swage_plan.block_threads = 48 : i32} {
    // expected-error@+1 {{a row-stripe task of rank-two values runs whole subgroups of 32 threads, so swage_plan.block_threads must be a multiple of 32, got 48}}
    swage_plan.partial_tasks values(%values : memref<?x?xf32>)
        value_count(%value_count : i32) ranges(%ranges : memref<?xi32>)
        partial_count(%partial_count : i32)
        feature_count(%feature_count : i32)
        into(%scratch : memref<?x?xf32>) {
    ^bb0(%chunk: !swage.segment<f32>):
      %total = swage.reduce %chunk kind<sum> : !swage.segment<f32> -> f32 {
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
  func.func @merge_rows_on_a_partial_subgroup(
      %scratch: memref<?x?xf32>, %output: memref<?x?xf32>,
      %merges: memref<?xi32>, %partial_count: i32, %merge_count: i32,
      %segment_count: i32, %feature_count: i32)
      attributes {swage_plan.block_threads = 48 : i32} {
    // expected-error@+1 {{a row-stripe task of rank-two values runs whole subgroups of 32 threads, so swage_plan.block_threads must be a multiple of 32, got 48}}
    swage_plan.merge_tasks scratch(%scratch : memref<?x?xf32>)
        partial_count(%partial_count : i32) merges(%merges : memref<?xi32>)
        merge_count(%merge_count : i32) segment_count(%segment_count : i32)
        feature_count(%feature_count : i32) into(%output : memref<?x?xf32>) {
    ^bb0(%partials: !swage.segment<f32>):
      %total = swage.reduce %partials kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%partial: f32):
        swage.yield %partial : f32
      }
      swage_plan.yield %total : f32
    }
    return
  }
}

// -----

// The element and word types are the ones the lowerings admit.
module {
  func.func @half_values(
      %values: memref<?xf16>, %offsets: memref<?xi32>,
      %output: memref<?xf16>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    // expected-error@+1 {{the conversion lowers f32 or f64 values with i32 offsets and counts, got values of 'f16' and offsets of 'i32'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf16>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf16>) {
    ^bb0(%segment: !swage.segment<f16>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f16> -> f16 {
      ^bb0(%value: f16):
        swage.yield %value : f16
      }
      swage_plan.yield %sum : f16
    }
    return
  }
}

// -----

// An f64 exp2 has no instruction on the device.
module {
  func.func @exponential_double(
      %values: memref<?xf64>, %offsets: memref<?xi32>,
      %output: memref<?xf64>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf64>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf64>) {
    ^bb0(%segment: !swage.segment<f64>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f64> -> f64 {
      ^bb0(%value: f64):
        // expected-error@+1 {{operation 'math.exp2' is admitted for f32 values only: the device has no f64 exp2}}
        %exponential = math.exp2 %value : f64
        swage.yield %exponential : f64
      }
      swage_plan.yield %sum : f64
    }
    return
  }
}

// -----

module {
  func.func @wide_offsets(
      %values: memref<?xf32>, %offsets: memref<?xi64>,
      %output: memref<?xf32>, %value_count: i64, %segment_count: i64)
      attributes {swage_plan.block_threads = 128 : i32} {
    // expected-error@+1 {{the conversion lowers f32 or f64 values with i32 offsets and counts, got values of 'f32' and offsets of 'i64'}}
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi64>)
        value_count(%value_count : i64) segment_count(%segment_count : i64)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The consumers hold only what a kernel can run.
module {
  func.func @exponential(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        // expected-error@+1 {{operation 'math.exp' is unsupported inside a segment region; a region accepts arith.constant, arith.addf, arith.subf, arith.mulf, arith.divf, arith.maximumf, arith.minimumf, and math.exp2}}
        %exponential = math.exp %value : f32
        swage.yield %exponential : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The conversion creates a gpu.module named after the plan function and
// removes the function.
module {
  // expected-note@+1 {{defined here}}
  func.func private @clashes_module()
  // expected-error@+1 {{lowering @clashes creates @clashes_module, which the module already defines}}
  func.func @clashes(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The second plan function is refused, so the first is not converted
// either.
module {
  func.func @convertible(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
  // expected-error@+1 {{segment function @called is referenced 1 times; lowering it to a GPU kernel removes it, so it must have no symbol use}}
  func.func @called(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
      attributes {swage_plan.block_threads = 128 : i32} {
    swage_plan.tasks policy<cta>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
  func.func @caller(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-note@+1 {{referenced here}}
    call @called(%values, %offsets, %output, %value_count, %segment_count)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    return
  }
}

// -----

// A plan function in a nested module is checked as one at the top level is:
// the conversion rewrites every plan function it reaches.
module {
  module @inner {
    func.func @nested_warp(
          %values: memref<?xf32>, %offsets: memref<?xi32>,
          %output: memref<?xf32>, %ids: memref<?xi32>, %value_count: i32,
          %task_count: i32, %segment_count: i32)
          attributes {swage_plan.block_threads = 128 : i32} {
        // expected-error@+1 {{policy<warp> requires swage_plan.block_threads to be the subgroup width, 32, got 128}}
        swage_plan.tasks policy<warp>
            segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
            value_count(%value_count : i32) segment_count(%segment_count : i32)
            ids(%ids : memref<?xi32>) task_count(%task_count : i32)
            into(%output : memref<?xf32>) {
        ^bb0(%segment: !swage.segment<f32>):
          %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
          ^bb0(%value: f32):
            swage.yield %value : f32
          }
          swage_plan.yield %sum : f32
        }
        return
    }
  }
}

// -----

// The kernel module of a nested plan function is created beside it, in the
// nested module, so its name must be free there.
module {
  module @inner {
    // expected-note@+1 {{defined here}}
    gpu.module @nested_clash_module {
    }
    // expected-error@+1 {{lowering @nested_clash creates @nested_clash_module, which the module already defines}}
    func.func @nested_clash(
          %values: memref<?xf32>, %offsets: memref<?xi32>,
          %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
          attributes {swage_plan.block_threads = 128 : i32} {
        swage_plan.tasks policy<cta>
            segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
            value_count(%value_count : i32) segment_count(%segment_count : i32)
            into(%output : memref<?xf32>) {
        ^bb0(%segment: !swage.segment<f32>):
          %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
          ^bb0(%value: f32):
            swage.yield %value : f32
          }
          swage_plan.yield %sum : f32
        }
        return
    }
  }
}

// -----

// A nested plan function may not be referenced by a function beside it
// either.
module {
  module @inner {
    // expected-error@+1 {{segment function @nested_called is referenced 1 times; lowering it to a GPU kernel removes it, so it must have no symbol use}}
    func.func @nested_called(
          %values: memref<?xf32>, %offsets: memref<?xi32>,
          %output: memref<?xf32>, %value_count: i32, %segment_count: i32)
          attributes {swage_plan.block_threads = 128 : i32} {
        swage_plan.tasks policy<cta>
            segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
            value_count(%value_count : i32) segment_count(%segment_count : i32)
            into(%output : memref<?xf32>) {
        ^bb0(%segment: !swage.segment<f32>):
          %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
          ^bb0(%value: f32):
            swage.yield %value : f32
          }
          swage_plan.yield %sum : f32
        }
        return
    }
    func.func @nested_caller(
        %values: memref<?xf32>, %offsets: memref<?xi32>,
        %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
      // expected-note@+1 {{referenced here}}
      call @nested_called(%values, %offsets, %output, %value_count,
          %segment_count)
          : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
      return
    }
  }
}
