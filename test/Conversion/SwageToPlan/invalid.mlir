// test/Conversion/SwageToPlan/invalid.mlir
// RUN: swage-opt --swage-to-plan --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-to-plan --verify-diagnostics --split-input-file --mlir-print-ir-after-failure %s 2>&1 | FileCheck %s --check-prefix=NO-PLAN --implicit-check-not=swage_plan.classify

// NO-PLAN-LABEL: func.func @maximum(
// NO-PLAN: swage.reduce {{.*}} kind<max>
// NO-PLAN-LABEL: func.func @transformed_sum(
// NO-PLAN: arith.addf
// NO-PLAN-LABEL: func.func @mapped_sum(
// NO-PLAN: swage.map
// NO-PLAN-LABEL: func.func @captured_sum(
// NO-PLAN: captures(
// NO-PLAN-LABEL: func.func @multi_stage(
// NO-PLAN-COUNT-2: swage.reduce
// NO-PLAN-LABEL: func.func @map_store_terminal(
// NO-PLAN: swage.map_store
// NO-PLAN: memref.global "private" @companion_collision__swage_plan
// NO-PLAN-LABEL: func.func @companion_collision(
// NO-PLAN: swage.reduce
// NO-PLAN-LABEL: func.func @signed_value_count(
// NO-PLAN: swage.reduce
// NO-PLAN-LABEL: func.func @unsigned_segment_count(
// NO-PLAN: swage.reduce
// NO-PLAN-LABEL: func.func @non_default_memory_space(
// NO-PLAN: swage.reduce

module {
  func.func @maximum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                     %output: memref<?xf32>, %value_count: i32,
                     %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{planning requires kind<sum>}}
    %maximum = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %maximum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @transformed_sum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                             %output: memref<?xf32>, %value_count: i32,
                             %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{planning requires an identity reduction region}}
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

module {
  func.func @mapped_sum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                        %output: memref<?xf32>, %value_count: i32,
                        %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    // expected-error@+1 {{planning does not support swage.map}}
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

// -----

module {
  func.func @captured_sum(%values: memref<?xf32>, %offsets: memref<?xi32>,
                          %output: memref<?xf32>, %value_count: i32,
                          %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %maximum = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    // expected-error@+1 {{planning requires a capture-free reduction}}
    %sum = swage.reduce %segment captures(%maximum : f32) kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32, %captured: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @multi_stage(%values: memref<?xf32>, %offsets: memref<?xi32>,
                         %output: memref<?xf32>, %value_count: i32,
                         %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %first = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    // expected-error@+1 {{planning requires exactly one reduction stage}}
    %second = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %second, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  func.func @map_store_terminal(%values: memref<?xf32>, %offsets: memref<?xi32>,
                                %output: memref<?xf32>, %value_count: i32,
                                %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    // expected-error@+1 {{planning requires memref.store of the reduction result}}
    swage.map_store %segment, %output : !swage.segment<f32>, memref<?xf32> {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    return
  }
}

// -----

// expected-error@+1 {{planning companion symbol @companion_collision__swage_plan already exists}}
module {
  memref.global "private" @companion_collision__swage_plan : memref<1xi32>
  func.func @companion_collision(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{segmented reduction requires rank-one f32 values, rank-one i32 offsets, rank-one f32 output, i32 value count, and i32 segment count}}
  func.func @signed_value_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: si32, %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{segmented reduction requires rank-one f32 values, rank-one i32 offsets, rank-one f32 output, i32 value count, and i32 segment count}}
  func.func @unsigned_segment_count(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: ui32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

module {
  // expected-error@+1 {{segmented reduction requires rank-one f32 values, rank-one i32 offsets, rank-one f32 output, i32 value count, and i32 segment count}}
  func.func @non_default_memory_space(
      %values: memref<?xf32, #gpu.address_space<workgroup>>,
      %offsets: memref<?xi32>, %output: memref<?xf32>, %value_count: i32,
      %segment_count: i32) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid : memref<?xf32, #gpu.address_space<workgroup>>, memref<?xi32>, index -> !swage.segment<f32>
    %sum = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%element: f32):
      swage.yield %element : f32
    }
    memref.store %sum, %output[%sid] : memref<?xf32>
    return
  }
}
