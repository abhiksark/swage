// test/Conversion/SwagePlanToGPU/schedule-list.mlir
// A schedule list plans three kernels of one function, and the conversion
// gives one gpu.module per kernel, in the order of the list.
//
// RUN: swage-opt \
// RUN:   --swage-to-plan='schedule=task-ids,split-partial,split-merge block-threads=32' \
// RUN:   --swage-plan-to-gpu %S/../SwageToGPU/segmented-sum.mlir \
// RUN:   | FileCheck %s --implicit-check-not=func.func

// CHECK: gpu.module @segmented_sum_module {
// CHECK-NEXT: gpu.func @segmented_sum({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 32, 1, 1>} {
// CHECK: gpu.shuffle xor
// CHECK: gpu.module @segmented_sum__partial_module {
// CHECK-NEXT: gpu.func @segmented_sum__partial({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
// CHECK: gpu.all_reduce add
// CHECK: gpu.module @segmented_sum__merge_module {
// CHECK-NEXT: gpu.func @segmented_sum__merge({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
// CHECK: gpu.all_reduce add
