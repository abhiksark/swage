// test/Conversion/SwagePlanToGPU/pipeline.mlir
// Planning and conversion as two passes of one pipeline. The planner writes
// a plan function and the conversion consumes it, which gives the module
// the one-step lowering flag gives. The last RUN line continues through the
// nested NVVM conversion the code generation C API runs.
//
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   %S/../SwageToGPU/segmented-sum.mlir > %t.direct
// RUN: swage-opt --swage-to-plan --swage-plan-to-gpu \
// RUN:   %S/../SwageToGPU/segmented-sum.mlir | diff %t.direct -
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=32 use-task-ids' \
// RUN:   %S/../SwageToGPU/segmented-max.mlir > %t.tasks
// RUN: swage-opt --swage-to-plan='schedule=task-ids block-threads=32' \
// RUN:   --swage-plan-to-gpu %S/../SwageToGPU/segmented-max.mlir \
// RUN:   | diff %t.tasks -
// RUN: swage-opt --swage-segmented-reduction-to-gpu='block-size=128' \
// RUN:   %S/../SwageToGPU/ragged-softmax.mlir > %t.softmax
// RUN: swage-opt --swage-to-plan --swage-plan-to-gpu \
// RUN:   %S/../SwageToGPU/ragged-softmax.mlir | diff %t.softmax -
// RUN: swage-opt %S/../SwageToGPU/segmented-sum.mlir \
// RUN:   --pass-pipeline='builtin.module(swage-to-plan{schedule=task-ids block-threads=32},swage-plan-to-gpu,gpu.module(convert-scf-to-cf,convert-gpu-to-nvvm{index-bitwidth=64}))' \
// RUN:   | FileCheck %s

// CHECK: gpu.module @segmented_sum_module {
// CHECK: llvm.func @segmented_sum(%{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: !llvm.ptr, %{{.*}}: i32, %{{.*}}: i32, %{{.*}}: i32)
// CHECK-SAME: nvvm.kernel
// CHECK: nvvm.shfl.sync
// CHECK-NOT: swage
