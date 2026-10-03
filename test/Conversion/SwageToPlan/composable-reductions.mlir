// test/Conversion/SwageToPlan/composable-reductions.mlir
// RUN: swage-opt --swage-to-plan --split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @maximum(
// CHECK: swage.reduce
// CHECK-LABEL: func.func private @maximum__swage_plan(
// CHECK: swage_plan.classify
module {
  func.func @maximum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                     %output: memref<?xf32>) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @transformed_sum(
// CHECK: swage.reduce
// CHECK-LABEL: func.func private @transformed_sum__swage_plan(
// CHECK: swage_plan.classify
module {
  func.func @transformed_sum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                             %output: memref<?xf32>) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      %doubled = arith.addf %element, %element : f32
      swage.yield %doubled : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// CHECK-LABEL: func.func @mapped_sum(
// CHECK: swage.reduce
// CHECK-LABEL: func.func private @mapped_sum__swage_plan(
// CHECK: swage_plan.classify
module {
  func.func @mapped_sum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                        %output: memref<?xf32>) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %mapped = swage.map %segment : !swage.segment<f32> -> !swage.segment<f32> {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    %sum = swage.reduce %mapped kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

