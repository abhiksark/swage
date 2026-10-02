// test/Conversion/SwagePlanToSCF/invalid.mlir
// What the oracle requires of a sequential plan beyond what the dialect
// verifies: the element and word types the lowerings admit, and element
// programs of admitted operations and kinds. Every rule is checked before
// anything is changed: the last RUN line prints each module after its
// failure and finds no loop in it.
//
// RUN: swage-opt --swage-plan-to-scf --verify-diagnostics \
// RUN:   --split-input-file %s
// RUN: swage-opt --swage-plan-to-scf --verify-diagnostics \
// RUN:   --split-input-file --mlir-print-ir-after-failure %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNCHANGED --implicit-check-not=scf.for

// UNCHANGED-LABEL: func.func @double_values(
// UNCHANGED: swage_plan.tasks policy<sequential>
// UNCHANGED-LABEL: func.func @minimum(
// UNCHANGED: swage_plan.tasks policy<sequential>
// UNCHANGED-LABEL: func.func @convertible(
// UNCHANGED: swage_plan.tasks policy<sequential>
// UNCHANGED-LABEL: func.func @exponential(

module {
  func.func @double_values(
      %values: memref<?xf64>, %offsets: memref<?xi32>,
      %output: memref<?xf64>, %value_count: i32, %segment_count: i32) {
    // expected-error@+1 {{the conversion lowers f32 values with i32 offsets and counts, got values of 'f64' and offsets of 'i32'}}
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?xf64>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf64>) {
    ^bb0(%segment: !swage.segment<f64>):
      %sum = swage.reduce %segment kind<sum> : !swage.segment<f64> -> f64 {
      ^bb0(%value: f64):
        swage.yield %value : f64
      }
      swage_plan.yield %sum : f64
    }
    return
  }
}

// -----

module {
  func.func @minimum(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<sequential>
        segments(%values, %offsets : memref<?xf32>, memref<?xi32>)
        value_count(%value_count : i32) segment_count(%segment_count : i32)
        into(%output : memref<?xf32>) {
    ^bb0(%segment: !swage.segment<f32>):
      // expected-error@+1 {{segmented reduction supports only kind<sum> and kind<max>, got kind<min>}}
      %sum = swage.reduce %segment kind<min> : !swage.segment<f32> -> f32 {
      ^bb0(%value: f32):
        swage.yield %value : f32
      }
      swage_plan.yield %sum : f32
    }
    return
  }
}

// -----

// The second function is refused, so the first is not converted either.
module {
  func.func @convertible(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<sequential>
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
  func.func @exponential(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    swage_plan.tasks policy<sequential>
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
