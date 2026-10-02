// test/Conversion/SwageToGPU/two-kernels.mlir
// A module may hold any number of segment functions. Each pass lowers every
// function that holds Swage operations to a kernel module of its own and
// leaves the other functions as they are. The function option restricts a
// pass to one function.
//
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' %s \
// RUN:   | FileCheck %s --check-prefixes=ALL,DIRECT
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128 use-task-ids' %s \
// RUN:   | FileCheck %s --check-prefixes=ALL,TASKS
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed' %s \
// RUN:   | FileCheck %s --check-prefixes=ALL
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu %s \
// RUN:   | FileCheck %s --check-prefix=PARTIAL
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu='merge' %s \
// RUN:   | FileCheck %s --check-prefix=MERGE
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128 function=second' \
// RUN:   %s | FileCheck %s --check-prefix=ONE
// RUN: swage-opt \
// RUN:   --swage-split-segmented-reduction-to-gpu='function=first' %s \
// RUN:   | FileCheck %s --check-prefix=ONE-PARTIAL

// ALL-NOT: swage.
// ALL: gpu.module @first_module {
// DIRECT-NEXT: gpu.func @first(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32) kernel
// TASKS-NEXT: gpu.func @first(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32) kernel
// ALL: arith.addf
// ALL-NOT: func.func @first
// ALL: func.func @bystander(%{{.*}}: i32) -> i32 {
// ALL-NEXT: return %{{.*}} : i32
// ALL: gpu.module @second_module {
// ALL-NEXT: gpu.func @second(
// ALL: arith.maximumf
// ALL-NOT: func.func @second
// ALL-NOT: swage.

// PARTIAL: gpu.module @first__partial_module {
// PARTIAL-NEXT: gpu.func @first__partial(
// PARTIAL: func.func @bystander(
// PARTIAL: gpu.module @second__partial_module {
// PARTIAL-NEXT: gpu.func @second__partial(
// PARTIAL-NOT: swage.

// MERGE: gpu.module @first__merge_module {
// MERGE-NEXT: gpu.func @first__merge(
// MERGE: func.func @bystander(
// MERGE: gpu.module @second__merge_module {
// MERGE-NEXT: gpu.func @second__merge(
// MERGE-NOT: swage.

// The function that is not named keeps its Swage operations and its roles.
// ONE-NOT: gpu.module
// ONE: func.func @first(%{{.*}}: memref<?xf32> {swage.role = #swage.role<values>},
// ONE: swage.reduce %{{.*}} kind<sum>
// ONE: func.func @bystander(
// ONE: gpu.module @second_module {
// ONE-NEXT: gpu.func @second(
// ONE-NOT: swage.

// ONE-PARTIAL: gpu.module @first__partial_module {
// ONE-PARTIAL: func.func @bystander(
// ONE-PARTIAL-NOT: gpu.module
// ONE-PARTIAL: func.func @second(
// ONE-PARTIAL: swage.reduce %{{.*}} kind<max>

module {
  func.func @first(
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
  func.func @bystander(%x: i32) -> i32 {
    return %x : i32
  }
  func.func @second(
      %values: memref<?xf32> {swage.role = #swage.role<values>},
      %offsets: memref<?xi32> {swage.role = #swage.role<offsets>},
      %output: memref<?xf32> {swage.role = #swage.role<output>},
      %value_count: i32 {swage.role = #swage.role<value_count>},
      %segment_count: i32 {swage.role = #swage.role<segment_count>}) {
    %sid = swage.segment_id 0
    %segment = swage.make_segment %values, %offsets, %sid
        : memref<?xf32>, memref<?xi32>, index -> !swage.segment<f32>
    %result = swage.reduce %segment kind<max> : !swage.segment<f32> -> f32 {
    ^bb0(%value: f32):
      swage.yield %value : f32
    }
    memref.store %result, %output[%sid] : memref<?xf32>
    return
  }
}
