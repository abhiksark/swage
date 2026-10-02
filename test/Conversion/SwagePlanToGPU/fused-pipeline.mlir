// test/Conversion/SwagePlanToGPU/fused-pipeline.mlir
// The fused mixed kernel as two passes of one pipeline: the plan converts to
// the module the one-step fused flag gives.
//
// RUN: swage-opt \
// RUN:   --swage-segmented-reduction-to-gpu='block-size=128 fused-mixed' \
// RUN:   %S/../SwageToGPU/fused-mixed.mlir > %t.fused
// RUN: swage-opt --swage-to-plan='schedule=fused-mixed' --swage-plan-to-gpu \
// RUN:   %S/../SwageToGPU/fused-mixed.mlir | diff %t.fused -
