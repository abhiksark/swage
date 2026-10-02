// test/Conversion/SwagePlanToGPU/split-pipeline.mlir
// The split partial stage as two passes of one pipeline, and a schedule
// list that plans two kernels of one function. The partial plan converts to
// the module the one-step split flag gives, and a list gives one gpu.module
// per kernel, in the order of the list.
//
// RUN: swage-opt --swage-split-segmented-reduction-to-gpu \
// RUN:   %S/../SwageToGPU/split-partial.mlir > %t.partial
// RUN: swage-opt --swage-to-plan='schedule=split-partial' --swage-plan-to-gpu \
// RUN:   %S/../SwageToGPU/split-partial.mlir | diff %t.partial -
// RUN: swage-opt \
// RUN:   --swage-to-plan='schedule=task-ids,split-partial block-threads=32' \
// RUN:   --swage-plan-to-gpu %S/../SwageToGPU/segmented-sum.mlir \
// RUN:   | FileCheck %s --implicit-check-not=func.func

// CHECK: gpu.module @segmented_sum_module {
// CHECK-NEXT: gpu.func @segmented_sum({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 32, 1, 1>} {
// CHECK: gpu.shuffle xor
// CHECK: gpu.module @segmented_sum__partial_module {
// CHECK-NEXT: gpu.func @segmented_sum__partial({{.*}}) kernel attributes {nvvm.reqntid = array<i32: 512, 1, 1>} {
// CHECK: gpu.all_reduce add
