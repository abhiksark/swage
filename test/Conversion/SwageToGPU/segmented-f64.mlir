// test/Conversion/SwageToGPU/segmented-f64.mlir
// The three reductions over f64 values. Every schedule takes the element
// type from the segment: the identity is an f64 constant, the loads, the
// stores, and the address arithmetic use f64, and the shuffle and the
// block reduction carry f64. No run may hold an f32, which is what an
// identity or a load of the wrong width would be.
//
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CTA --implicit-check-not=f32
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WARP --implicit-check-not=f32 \
// RUN:       --implicit-check-not=gpu.all_reduce
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=FUSED --implicit-check-not=f32
// RUN: swage-opt --swage-to-plan='schedule=split-partial,split-merge' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=SPLIT --implicit-check-not=f32

module {
  func.func @segmented_sum_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<sum>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
  func.func @segmented_max_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<max>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
  func.func @segmented_min_f64(
      %values: memref<?xf64> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf64> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf64>, memref<?xi32>, index -> !swage.segment<f64>
    %result = swage.reduce %segment kind<min>
        : !swage.segment<f64> -> f64 {
    ^bb0(%value: f64):
      swage.yield %value : f64
    }
    memref.store %result, %output[%sid] : memref<?xf64>
    return
  }
}

// CHECK: gpu.func @segmented_sum_f64
// CHECK: %[[ZERO:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[LOCAL:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ZERO]]) -> (f64) {
// CHECK:   %[[ADDRESS:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CHECK:   %[[VALUE:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f64
// CHECK:   %[[NEXT:.*]] = arith.addf %[[ACC]], %[[VALUE]] : f64
// CHECK:   scf.yield %[[NEXT]] : f64
// CHECK: }
// CTA: %[[TOTAL:.*]] = gpu.all_reduce add %[[LOCAL]] uniform
// CTA-NEXT: } : (f64) -> f64
// CTA: %[[SLOT:.*]] = llvm.getelementptr %{{.*}}[%{{.*}}] : (!llvm.ptr, i64) -> !llvm.ptr, f64
// CTA-NEXT: llvm.store %[[TOTAL]], %[[SLOT]] : f64, !llvm.ptr
// The subgroup combines with five shuffle steps of f64 values.
// WARP: %[[S1:.*]], %{{.*}} = gpu.shuffle xor %[[LOCAL]], %{{.*}}, %{{.*}} : f64
// WARP-NEXT: %[[T1:.*]] = arith.addf %[[LOCAL]], %[[S1]] : f64
// WARP: %[[S2:.*]], %{{.*}} = gpu.shuffle xor %[[T1]], %{{.*}}, %{{.*}} : f64
// WARP-NEXT: %[[T2:.*]] = arith.addf %[[T1]], %[[S2]] : f64
// WARP: %[[S3:.*]], %{{.*}} = gpu.shuffle xor %[[T2]], %{{.*}}, %{{.*}} : f64
// WARP-NEXT: %[[T3:.*]] = arith.addf %[[T2]], %[[S3]] : f64
// WARP: %[[S4:.*]], %{{.*}} = gpu.shuffle xor %[[T3]], %{{.*}}, %{{.*}} : f64
// WARP-NEXT: %[[T4:.*]] = arith.addf %[[T3]], %[[S4]] : f64
// WARP: %[[S5:.*]], %{{.*}} = gpu.shuffle xor %[[T4]], %{{.*}}, %{{.*}} : f64
// WARP-NEXT: %[[T5:.*]] = arith.addf %[[T4]], %[[S5]] : f64
// WARP: llvm.store %[[T5]], %{{.*}} : f64, !llvm.ptr

// CHECK: gpu.func @segmented_max_f64
// CHECK: arith.constant 0xFFF0000000000000 : f64
// CHECK: arith.maximumf %{{.*}}, %{{.*}} : f64
// CTA: gpu.all_reduce maximumf
// WARP: arith.maximumf %{{.*}}, %{{.*}} : f64

// CHECK: gpu.func @segmented_min_f64
// CHECK: arith.constant 0x7FF0000000000000 : f64
// CHECK: arith.minimumf %{{.*}}, %{{.*}} : f64
// CTA: gpu.all_reduce minimumf
// WARP: arith.minimumf %{{.*}}, %{{.*}} : f64

// FUSED: gpu.func @segmented_sum_f64
// FUSED: gpu.shuffle xor %{{.*}}, %{{.*}}, %{{.*}} : f64
// FUSED: gpu.all_reduce add
// FUSED: gpu.func @segmented_max_f64
// FUSED: gpu.func @segmented_min_f64

// A partial task stores an f64 in scratch, and the merge loads f64 from it.
// SPLIT: gpu.func @segmented_sum_f64__partial
// SPLIT: llvm.load %{{.*}} : !llvm.ptr -> f64
// SPLIT: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
// SPLIT: gpu.func @segmented_sum_f64__merge
// SPLIT: llvm.load %{{.*}} : !llvm.ptr -> f64
// SPLIT: llvm.store %{{.*}}, %{{.*}} : f64, !llvm.ptr
