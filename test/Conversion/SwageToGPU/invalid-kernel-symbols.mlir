// test/Conversion/SwageToGPU/invalid-kernel-symbols.mlir
// A GPU lowering replaces a segment function by a gpu.module named after its
// kernel. Nothing may refer to the function, the name of the module must be
// free, and the function option must name a segment function. Each rule is
// checked before any function is changed: the last RUN line prints the
// module after the failure and finds no gpu.module in it.
//
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128 use-task-ids' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=512 persistent' \
// RUN:   --verify-diagnostics --split-input-file %s
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   --verify-diagnostics --split-input-file \
// RUN:   --mlir-print-ir-after-failure %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=UNCHANGED --implicit-check-not=gpu.module

// UNCHANGED-LABEL: func.func @called(
// UNCHANGED: swage.reduce
// UNCHANGED-LABEL: func.func @caller(
// UNCHANGED-LABEL: func.func @clashes(
// UNCHANGED: swage.reduce
// UNCHANGED-LABEL: func.func @admitted(
// UNCHANGED: swage.reduce
// UNCHANGED-LABEL: func.func @referenced_after_admitted(
// UNCHANGED: swage.reduce

module {
  // expected-error@+1 {{segment function @called is referenced 1 times; lowering it to a GPU kernel removes it, so it must have no symbol use}}
  func.func @called(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
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

module {
  // expected-note@+1 {{defined here}}
  func.func private @clashes_module()
  // expected-error@+1 {{lowering @clashes creates @clashes_module, which the module already defines}}
  func.func @clashes(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
}

// -----

// The second function is refused, so the first, which is admitted, is not
// lowered either.
module {
  func.func @admitted(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
  // expected-error@+1 {{segment function @referenced_after_admitted is referenced 1 times; lowering it to a GPU kernel removes it, so it must have no symbol use}}
  func.func @referenced_after_admitted(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<sum> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
  func.func @caller(
      %values: memref<?xf32>, %offsets: memref<?xi32>,
      %output: memref<?xf32>, %value_count: i32, %segment_count: i32) {
    // expected-note@+1 {{referenced here}}
    call @referenced_after_admitted(%values, %offsets, %output, %value_count, %segment_count)
        : (memref<?xf32>, memref<?xi32>, memref<?xf32>, i32, i32) -> ()
    return
  }
}
