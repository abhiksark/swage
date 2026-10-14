// test/Conversion/SwageToGPU/segmented-min.mlir
// A minimum is lowered as a minimum on every schedule: the identity is
// positive infinity, an element is combined with arith.minimumf, a block
// combines with the minimumf block reduction, and a subgroup with a
// shuffle tree of arith.minimumf. No run may hold a maximum, which is what
// a kind that took the branch of another kind would emit.
//
// RUN: swage-opt --swage-to-plan='schedule=direct block-threads=128' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,CTA \
// RUN:       --implicit-check-not=maximumf --implicit-check-not=0xFF800000
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,WARP \
// RUN:       --implicit-check-not=maximumf --implicit-check-not=0xFF800000 \
// RUN:       --implicit-check-not=gpu.all_reduce
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=FUSED \
// RUN:       --implicit-check-not=maximumf --implicit-check-not=0xFF800000
// RUN: swage-opt --swage-to-plan='schedule=split-partial' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=SPLIT \
// RUN:       --implicit-check-not=maximumf --implicit-check-not=0xFF800000
// RUN: swage-opt --swage-to-plan='schedule=split-merge' \
// RUN:   --swage-plan-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=SPLIT \
// RUN:       --implicit-check-not=maximumf --implicit-check-not=0xFF800000

module {
  func.func @segmented_min(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %minimum = swage.reduce %segment kind<min>
        : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %minimum, %output[%sid] : memref<?xf32>
    return
  }
}

// CHECK-NOT: swage.
// CHECK: gpu.module @segmented_min_module
// CHECK: gpu.func @segmented_min
// CHECK: %[[IDENTITY:.*]] = arith.constant 0x7F800000 : f32
// CHECK: %[[LOCAL:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[IDENTITY]]) -> (f32) {
// CHECK:   %[[VALUE:.*]] = llvm.load {{.*}} : !llvm.ptr -> f32
// CHECK:   %[[NEXT:.*]] = arith.minimumf %[[ACC]], %[[VALUE]] : f32
// CHECK:   scf.yield %[[NEXT]] : f32
// CHECK: }
// CTA: %[[TOTAL:.*]] = gpu.all_reduce minimumf %[[LOCAL]] uniform
// CTA: llvm.store %[[TOTAL]]
// The subgroup combines with five shuffle steps, each a minimum.
// WARP: %[[S1:.*]], %{{.*}} = gpu.shuffle xor %[[LOCAL]],
// WARP-NEXT: %[[T1:.*]] = arith.minimumf %[[LOCAL]], %[[S1]] : f32
// WARP: %[[S2:.*]], %{{.*}} = gpu.shuffle xor %[[T1]],
// WARP-NEXT: %[[T2:.*]] = arith.minimumf %[[T1]], %[[S2]] : f32
// WARP: %[[S3:.*]], %{{.*}} = gpu.shuffle xor %[[T2]],
// WARP-NEXT: %[[T3:.*]] = arith.minimumf %[[T2]], %[[S3]] : f32
// WARP: %[[S4:.*]], %{{.*}} = gpu.shuffle xor %[[T3]],
// WARP-NEXT: %[[T4:.*]] = arith.minimumf %[[T3]], %[[S4]] : f32
// WARP: %[[S5:.*]], %{{.*}} = gpu.shuffle xor %[[T4]],
// WARP-NEXT: %[[T5:.*]] = arith.minimumf %[[T4]], %[[S5]] : f32
// WARP: llvm.store %[[T5]]
// CHECK-NOT: swage.

// The fused kernel holds both: the shuffle tree of a warp task and the
// block reduction of a block task.
// FUSED: gpu.func @segmented_min
// FUSED: arith.constant 0x7F800000 : f32
// FUSED: gpu.shuffle xor
// FUSED-NEXT: arith.minimumf
// FUSED: arith.constant 0x7F800000 : f32
// FUSED: gpu.all_reduce minimumf

// A partial task reduces its chunk with the kind of the program, and the
// merge reduces the partial results with the same kind.
// SPLIT: gpu.func @segmented_min__{{partial|merge}}
// SPLIT: arith.constant 0x7F800000 : f32
// SPLIT: arith.minimumf
// SPLIT: gpu.all_reduce minimumf
